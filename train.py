from dataset.librispeech import SpeechDataModule
from model.sdhubert import SDHuBERTTrainer
import lightning as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import hydra
from pathlib import Path
import torch

torch.set_float32_matmul_precision('medium')

@hydra.main(config_path='configs', config_name='sdhubert_base')
def main(cfg):
    
    # datamodule
    datamodule = SpeechDataModule(**cfg['data'])

    # model
    model = SDHuBERTTrainer(**cfg['model'])
    
    # Callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # checkpoint best
    checkpoint_callback_topk = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        filename='best-{epoch}-{val_loss:.2f}'
    )
    
    # checkpoint every N epochs
    checkpoint_callback_by_epoch = ModelCheckpoint(
        every_n_epochs=cfg['checkpoint_epoch'],
    )
    
    # Trainer
    if cfg['gpus'] is not None:
        if not isinstance(cfg['gpus'],list):
            try:
                gpus = [int(cfg['gpus'])]
            except:
                gpus = [int(x) for x in cfg['gpus'].split(',')]
        else:
            gpus = cfg['gpus']
    else:
        gpus= None
    
    callbacks  = [checkpoint_callback_topk, checkpoint_callback_by_epoch,
                  LearningRateMonitor(logging_interval='step')]
    
    scaler = torch.cuda.amp.GradScaler()
    
    trainer = pl.Trainer(devices=gpus,
                         accelerator="gpu",
                         strategy="ddp_find_unused_parameters_true",
                         max_steps = cfg['max_steps'],
                         num_sanity_val_steps=0,
                         check_val_every_n_epoch=cfg['check_val_every_n_epoch'],
                         limit_val_batches=cfg['limit_val_batches'],
                         callbacks=callbacks,
                         gradient_clip_val=0.5,
                         default_root_dir=cfg.get('name', 'noname'),
                        )

    # fit model
    trainer.fit(model,datamodule,ckpt_path=cfg['resume_ckpt'],)

if __name__ =='__main__':
    main()
