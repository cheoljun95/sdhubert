import torch
import torch.nn.functional as F
import math
import random
import math
import numpy as np

def stretch(x, attention=None,anchor_range=[0.2,0.8],rescale_range=[0.8,1.5] ):
    # x: (1, L, d)
    # attention: (1, d,)
    #assert x.shape[0] == 1
    x = x.transpose(1,2)
    if attention is None:
        _, _, L = x.shape
    else:
        L = attention.sum(1).long().min().cpu().numpy()
    anchor = np.random.uniform(low=anchor_range[0],
                               high=anchor_range[1])
    anchor = int(L*anchor)
    stretch_left = np.random.uniform()<0.5
    left=x[:,:,:anchor]
    right=x[:,:,anchor:L]
    
    if stretch_left:
        rescale_factor = np.random.uniform(low=rescale_range[0],
                               high=min( rescale_range[1], L/ left.shape[2]))
        stretched = torch.nn.functional.interpolate(left,scale_factor=rescale_factor,mode='linear',)
        rescale_factor_2 = max(0,(L-stretched.shape[2])/right.shape[2])
        if rescale_factor_2 == 0:
            x = stretched[:,:,:L]
        else:
            stretched_2 = torch.nn.functional.interpolate(right,size=(L-stretched.shape[2],),mode='linear',)
            x = torch.cat([stretched,stretched_2,x[:,:,L:]],2)
    else:
        rescale_factor = np.random.uniform(low=rescale_range[0],
                               high=min( rescale_range[1], L/ right.shape[2]))
        stretched = torch.nn.functional.interpolate(right,scale_factor=rescale_factor,mode='linear',)
        rescale_factor_2 = max(0,(L-stretched.shape[2])/left.shape[2])
        if rescale_factor_2 == 0:
            x = stretched[:,:,:L]
        else:
            stretched_2 = torch.nn.functional.interpolate(left,size=(L-stretched.shape[2],),mode='linear',)
            x = torch.cat([stretched_2,stretched,x[:,:,L:]],2)
    x = x.transpose(1,2)
    return x


def compute_mask_indices(shape, mask_prob, mask_length, attention_mask= None, min_masks = 0,):
    """
    Adopted from Fairseq implementation of SpecAugment (https://arxiv.org/abs/1904.08779)

    Args:
        shape: The shape for which to compute masks. This should be of a tuple of size 2 where
               the first element is the batch size and the second element is the length of the axis to span.
        mask_prob:  The percentage of the whole axis (between 0 and 1) which will be masked. The number of
                    independently generated mask spans of length `mask_length` is computed by
                    `mask_prob*shape[1]/mask_length`. Note that due to overlaps, `mask_prob` is an upper bound and the
                    actual percentage will be smaller.
        mask_length: size of the mask
        min_masks: minimum number of masked spans
        attention_mask: A (right-padded) attention mask which independently shortens the feature axis of
                        each batch dimension.
    """
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length}"
            f" and `sequence_length`: {sequence_length}`"
        )

    # epsilon is used for probabilistic rounding
    epsilon = np.random.rand(1).item()

    def compute_num_masked_span(input_length):
        """Given input length, compute how many spans should be masked"""
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)
        num_masked_span = max(num_masked_span, min_masks)

        # make sure num masked span <= sequence_length
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length

        # make sure num_masked span is also <= input_length - (mask_length - 1)
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)

        return num_masked_span

    # compute number of masked spans in batch
    input_lengths = (
        attention_mask.sum(-1).detach().tolist()
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]
    )

    # SpecAugment mask to fill
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)
    spec_aug_mask_idxs = []

    max_num_masked_span = compute_num_masked_span(sequence_length)

    if max_num_masked_span == 0:
        return spec_aug_mask

    for input_length in input_lengths:
        # compute num of masked spans for this input
        num_masked_span = compute_num_masked_span(input_length)

        # get random indices to mask
        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
        )

        # pick first sampled index that will serve as a dummy index to pad vector
        # to ensure same dimension for all batches due to probabilistic rounding
        # Picking first sample just pads those vectors twice.
        if len(spec_aug_mask_idx) == 0:
            # this case can only happen if `input_length` is strictly smaller then
            # `sequence_length` in which case the last token has to be a padding
            # token which we can use as a dummy mask id
            dummy_mask_idx = sequence_length - 1
        else:
            dummy_mask_idx = spec_aug_mask_idx[0]

        spec_aug_mask_idx = np.concatenate(
            [spec_aug_mask_idx, np.ones(max_num_masked_span - num_masked_span, dtype=np.int32) * dummy_mask_idx]
        )
        spec_aug_mask_idxs.append(spec_aug_mask_idx)

    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)

    # expand masked indices to masked spans
    spec_aug_mask_idxs = np.broadcast_to(
        spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, max_num_masked_span * mask_length)

    # add offset to the starting indexes so that indexes now create a span
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
        batch_size, max_num_masked_span * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # ensure that we cannot have indices larger than sequence_length
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1

    # scatter indices to mask
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    return spec_aug_mask
