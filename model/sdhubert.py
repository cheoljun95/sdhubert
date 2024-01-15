import math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from transformers import HubertModel
from .ema_module import EMAModule
from utils.specaugment import stretch, compute_mask_indices

DEFAULT_PERTURB_PLAN = {'mask0': {'plan_name': 'mask', 'plan_prob': 0.4, 'mask_prob': 0.2, 'mask_length':10},
                 'mask1': {'plan_name': 'mask', 'plan_prob': 0.4, 'mask_prob': 0.3, 'mask_length': 5},
                 'stretch': {'plan_name': 'stretch', 'plan_prob': 0.2, 'mask_prob': 0.3, 'mask_length': 5}}

class SDHuBERT(nn.Module):

    def __init__(self,
                 speech_upstream="facebook/hubert-base-ls960",
                 reinit_layers=[9,10,11],
                 ema_decay=0.999,
                 final_dim=2048,
                 mask_prob=0.2,
                 freeze_extractor=True,
                 freeze_pos_embedding=True,
                 perturb_teacher=True,
                 center_momentum=0.9,
                 perturb_plans=DEFAULT_PERTURB_PLAN,
                 perturb_subbatch_ratio=None,
                 **kwargs,
                ):
        super().__init__()
        self.speech_model = HubertModel.from_pretrained(speech_upstream)
        for l in reinit_layers:
            layer = self.speech_model.encoder.layers[l]
            nn.init.xavier_normal_(layer.attention.k_proj.weight)
            nn.init.xavier_normal_(layer.attention.q_proj.weight)
            nn.init.xavier_normal_(layer.attention.out_proj.weight)
            nn.init.xavier_normal_(layer.feed_forward.intermediate_dense.weight)
            nn.init.normal_(layer.feed_forward.intermediate_dense.bias,std=1.0/math.sqrt(np.prod(layer.feed_forward.intermediate_dense.bias.shape)))
            nn.init.xavier_normal_(layer.feed_forward.output_dense.weight)
            nn.init.normal_(layer.feed_forward.output_dense.bias,std=1.0/math.sqrt(np.prod(layer.feed_forward.output_dense.bias.shape)))

        self.reinit_layers = reinit_layers
        self.enc_dim = self.speech_model.config.hidden_size
        self.cls_token = nn.Parameter(torch.Tensor(self.enc_dim))
        self.masked_spec_embed = nn.Parameter(torch.Tensor(self.enc_dim))
        self.temp_s = nn.parameter.Parameter(torch.ones(1)*0.2,requires_grad=False)
        self.temp_t = nn.parameter.Parameter(torch.ones(1)*0.05,requires_grad=False)
        self.center = nn.Parameter(torch.Tensor(final_dim),requires_grad=False)
        self.center_momentum_ = nn.Parameter(torch.tensor(0.0),requires_grad=False)
        self.center_momentum = center_momentum
        
        nn.init.constant_(self.center,0)
        nn.init.normal_(self.cls_token,std=1.0/math.sqrt(self.enc_dim))
        nn.init.normal_(self.masked_spec_embed,std=1.0/math.sqrt(self.enc_dim))
        
        self.final_proj = nn.Sequential(nn.Linear(self.enc_dim ,2048),
                                nn.GELU(),
                                nn.Linear(2048 ,2048),
                                nn.GELU(),
                                nn.Linear(2048 ,2048),
                                nn.LayerNorm(2048),)
        
        self.final_lin = nn.Linear(2048 ,final_dim)
        self.ema = None
        self.ema_final = None
        self.ema_finallin = None
        self.ema_decay = ema_decay
        self.final_dim = final_dim
        self.layernorm = nn.LayerNorm(self.final_dim)
        
        if perturb_plans == 'default':
            perturb_plans = DEFAULT_PERTURB_PLAN
        self.perturb_plans = perturb_plans
        self.perturb_names = list(self.perturb_plans.keys()) 
        self.perturb_probs = [plan['plan_prob'] for _, plan in self.perturb_plans.items()]
            
        self.freeze_extractor = freeze_extractor
        self.freeze_pos_embedding = freeze_pos_embedding
        self.perturb_teacher =perturb_teacher
        
        if self.freeze_extractor:
            self.speech_model.feature_extractor.requires_grad_(False)
        if self.freeze_pos_embedding:
            self.speech_model.encoder.pos_conv_embed.requires_grad_(False)
            
        self.perturb_subbatch_ratio = perturb_subbatch_ratio
        
    def ema_step(self):
        if self.ema is None:
            self.ema = EMAModule(
                self.speech_model.encoder.layers,
                ema_decay=self.ema_decay,
            )
            self.ema_final = EMAModule(
                self.final_proj,
                ema_decay=self.ema_decay,
            )
            self.ema_finallin = EMAModule(
                self.final_lin,
                ema_decay=self.ema_decay,
            )
        else:
            self.ema.step(self)
            self.ema_final.step(self)
            self.ema_finallin.step(self)
            
    def _perturb_batch(self, x, attention=None):
        
        batch_size,L,d = x.shape
        x_perturbed = []
        if self.perturb_subbatch_ratio is None:
            subbatch_size = 1
        else:
            subbatch_size = int(batch_size*self.perturb_subbatch_ratio)
        for b in range(0,batch_size+subbatch_size-1,subbatch_size):
            plan_name = self.perturb_names[np.random.multinomial(1, self.perturb_probs, size=None).argmax()]
            plan_args = self.perturb_plans[plan_name]
            att_ = attention[b:b+subbatch_size] if attention is not None else None
            if plan_name =='stretch':
                x_perturbed.append(stretch(x[b:b+subbatch_size],attention=att_))
            elif plan_name == 'channel_mask':
                x_perturbed.append(x[b:b+subbatch_size]*((torch.rand(1,1,d,device=x.device)>plan_args['mask_prob'])*1.0))
            else:
                mask_time_indices= compute_mask_indices(
                                            (len(x[b:b+subbatch_size]), L),
                                            mask_prob=plan_args['mask_prob'],
                                            mask_length=plan_args['mask_length'],
                                            attention_mask=att_,
                                            min_masks=0,
                                            )
                mask_time_indices = torch.tensor(mask_time_indices, device=x.device, dtype=torch.bool)
                x[b:b+subbatch_size][mask_time_indices] = self.masked_spec_embed.to(x.dtype)
                x_perturbed.append(x[b:b+1])
                
        return torch.cat(x_perturbed,0)
            
        
    def forward(self,wav,wavlen=None, inference_mode=False):
        """
        """
        batch_size = len(wav)
        extract_features = self.speech_model.feature_extractor(wav)
        extract_features = extract_features.transpose(1, 2)
        hidden_states = self.speech_model.feature_projection(extract_features)
        teacher_hidden_states = hidden_states.detach().clone()
        
        if wavlen is not None:
            attention_mask =torch.zeros_like(hidden_states)
            for b, l in enumerate(wavlen):
                attention_mask[b,:int(np.ceil(l*50))+1] = 1
            attention_mask =attention_mask >0
            hidden_states[~attention_mask] = 0
            attention_mask = attention_mask[:,:,0]
            raw_attention_mask = attention_mask
            attention_mask = torch.cat([ torch.ones_like(attention_mask[:,:1])>0, attention_mask],1)
            attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )
        else:
            attention_mask = None
            raw_attention_mask = None
        
        if not inference_mode:
            hidden_states = self._perturb_batch(hidden_states,raw_attention_mask)
        position_embeddings = self.speech_model.encoder.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.speech_model.encoder.layer_norm(hidden_states)
        hidden_states = self.speech_model.encoder.dropout(hidden_states)
        
        hidden_states =[self.cls_token[None,None,:].repeat(batch_size,1,1), hidden_states,]
        hidden_states = torch.cat(hidden_states, 1)
        
        batch_size, sequence_length, _ = hidden_states.shape

        for li,layer in enumerate(self.speech_model.encoder.layers):
            dropout_probability = torch.rand([])
            skip_the_layer = True if self.speech_model.encoder.training and \
                    (dropout_probability < self.speech_model.encoder.config.layerdrop) else False
            if inference_mode or not skip_the_layer:
                layer_outputs = layer(
                        hidden_states, attention_mask=attention_mask, output_attentions=False
                    )
                hidden_states = layer_outputs[0]

        student_states = hidden_states
        student_cls_token = self.final_lin(self.final_proj(student_states[:,0]))
                         
        if not inference_mode:
            with torch.no_grad():
                self.ema.model.eval()
                self.ema_final.model.eval()
                self.ema_finallin.model.eval()
                hidden_states= teacher_hidden_states
                if self.perturb_teacher:
                    hidden_states = self._perturb_batch(hidden_states,raw_attention_mask)
                position_embeddings = self.speech_model.encoder.pos_conv_embed(hidden_states)
                hidden_states = hidden_states + position_embeddings
                hidden_states = self.speech_model.encoder.layer_norm(hidden_states)
                hidden_states = self.speech_model.encoder.dropout(hidden_states)
                hidden_states =[self.cls_token[None,None,:].repeat(batch_size,1,1), hidden_states,]
                hidden_states = torch.cat(hidden_states, 1)

                for li, layer in enumerate(self.ema.model):
                    layer_outputs = layer(
                                hidden_states, attention_mask=attention_mask, output_attentions=False
                            )
                    hidden_states = layer_outputs[0]

                teacher_cls_token = self.ema_finallin.model(self.ema_final.model(hidden_states[:,0]))
                self.center *= self.center_momentum_
                self.center += (1-self.center_momentum_)*teacher_cls_token.mean(0) 
                self.center_momentum_ += (self.center_momentum-self.center_momentum_) 
                
            distill_loss = (-torch.log((student_cls_token/self.temp_s).softmax(-1))*((teacher_cls_token-self.center[None,:])/self.temp_t).softmax(-1)).sum()
        else:
            distill_loss = None
            
        outputs = {'states': student_states,
                   'cls': student_cls_token,
                   'distill_loss': distill_loss,}

        return outputs

class SDHuBERTTrainer(LightningModule):

    def __init__(self, loss_coefs, lr=0.01, gamma=0.1,use_cosine_lr=True, T_max=200000, 
                 **model_configs):
        super().__init__()

        self.loss_coefs = loss_coefs
        self.lr = lr
        self.net = SDHuBERT(**model_configs).to(torch.float)
        self.gamma = gamma
        self.T_max = T_max
        
    def forward(self, **kwargs):
        return self.net(**kwargs)
    
    def training_step(self, batch, batch_idx):
        self.net.ema_step()
        outputs = self.net(**batch)
        
        loss_val = 0
        for coef_name, coef_val in self.loss_coefs.items():
            if coef_name in outputs.keys():
                loss_val += coef_val * outputs[coef_name]
                self.log(f'train_{coef_name}', outputs[coef_name],sync_dist=True)
        self.log(f'train_loss', loss_val)
        
        # For checking degeneration
        self.log(f'train_state_avg_std', outputs['states'].reshape(-1,outputs['states'].shape[-1]).std(0).mean(-1))
        self.log(f'train_token_avg_std', outputs['cls'].std(0).mean(-1))
        return loss_val

        
    def validation_step(self, batch, batch_idx):
        outputs = self.net(**batch)
        loss_val = 0
        
        for coef_name, coef_val in self.loss_coefs.items():
            if coef_name in outputs.keys():
                loss_val += coef_val * outputs[coef_name]
                self.log(f'val_{coef_name}', outputs[coef_name],sync_dist=True)
        self.log(f'val_loss', loss_val,sync_dist=True)
        
        # For checking degeneration
        self.log(f'val_state_avg_std', outputs['states'].reshape(-1,outputs['states'].shape[-1]).std(0).mean(-1))
        self.log(f'val_token_avg_std', outputs['cls'].std(0).mean(-1))
            
        return loss_val

    def configure_optimizers(self):       
        opt_fun = torch.optim.AdamW
        opt = opt_fun(self.net.parameters(),lr=self.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt,eta_min=self.lr*self.gamma,T_max=self.T_max)
        #return [opt], [{"scheduler": sch, "interval": "step"}]
        return {"optimizer":opt, "scheduler":sch, "interval": "step"}
