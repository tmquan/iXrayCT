import os
import glob
from argparse import ArgumentParser
from natsort import natsorted

import datetime
import random
import numpy as np
import cv2
import skimage.io
import scipy.ndimage
import albumentations as AB

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision 

import pytorch_lightning as pl
from pytorch_lightning import Trainer, LightningModule, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateLogger


import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')

from models import * #PNet, INet, UNet, accumulate
from data import *

class Model(pl.LightningModule):
    def __init__(self, hparams, n_classes=1):
        super(Model, self).__init__()
        self.hparams = hparams
        self.example_input_array = torch.rand(self.hparams.batch_size, 
                                              self.hparams.dimz, 
                                              self.hparams.dimy, 
                                              self.hparams.dimx)
        
        self.save_hyperparameters()
        self.hparams = hparams
        # networks
        self.generator = self.init_generator()
        self.discriminator = self.init_discriminator()

    def forward(self, x):
        return self.generator(x)

    def configure_optimizers(self):
        lr = self.hparams.lr
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        return [opt_g, opt_d], []

    def init_discriminator(self):
        import torchvision.models as models
        discriminator = models.resnet18(pretrained=True)
        discriminator.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        discriminator.fc = nn.Linear(512, 1)
        return discriminator 

    def init_generator(self):
        generator = PNet(is_injecting=True, num_filters=8)
        return generator

    
    def discriminator_loss(self, x_real, i_fake):
        # train discriminator on real
        b = x_real.size(0)
        y_real = torch.ones(b, 1, device=self.device)
        y_fake = torch.zeros(b, 1, device=self.device)

        # calculate real score
        D_output = self.discriminator(x_real)
        # D_real_loss = F.binary_cross_entropy(D_output, y_real)
        D_real_loss = F.binary_cross_entropy_with_logits(D_output, y_real)
        
        # train discriminator on fake
        x_fake = self(i_fake) # Replace the input

        # calculate fake score
        D_output = self.discriminator(x_fake)
        # D_fake_loss = F.binary_cross_entropy(D_output, y_fake)
        D_fake_loss = F.binary_cross_entropy_with_logits(D_output, y_fake)
        
        # gradient backprop & optimize ONLY D's parameters
        D_loss = D_real_loss + D_fake_loss

        return D_loss

    def generator_loss(self, x_real, i_fake):
        # sample noise
        # z = torch.randn(x.shape[0], self.hparams.latent_dim, device=self.device)
        b = x_real.size(0)
        y = torch.ones(b, 1, device=self.device)

        # generate images
        x_fake = self(i_fake)
        D_output = self.discriminator(x_fake)

        # ground truth result (ie: all real)
        # g_loss = F.binary_cross_entropy(D_output, y)
        g_loss = F.binary_cross_entropy_with_logits(D_output, y)

        return g_loss


    def discriminator_step(self, x_real, i_fake):
        d_loss = self.discriminator_loss(x_real, i_fake)

        # log to prog bar on each step AND for the full epoch
        result = pl.TrainResult(minimize=d_loss)
        result.log('d_loss', d_loss, on_epoch=True, prog_bar=True)
        return result


    def generator_step(self, x_real, x_fake):
        g_loss = self.generator_loss(x_real, x_fake)

        # log to prog bar on each step AND for the full epoch
        # use the generator loss for checkpointing
        result = pl.TrainResult(minimize=g_loss, checkpoint_on=g_loss)
        result.log('g_loss', g_loss, on_epoch=True, prog_bar=True)
        return result

    
    def training_step(self, batch, batch_idx, optimizer_idx):
        ct3xr2_ct3_unpaired,\
        ct3xr2_xr2_unpaired,\
        xr2ct3_xr2_unpaired,\
        xr2ct3_ct3_unpaired,\
        ct3xr2_ct3_paired,\
        ct3xr2_xr2_paired,\
        xr2ct3_xr2_paired,\
        xr2ct3_ct3_paired,\
        ct3lung3_img_paired,\
        ct3lung3_lbl_paired,\
        ct3covid3_img_paired,\
        ct3covid3_lbl_paired,\
        xr2lung2_img_paired,\
        xr2lung2_lbl_paired,\
        xr2covid2_img_paired,\
        xr2covid2_lbl_paired = [ item / 128.0 - 1.0 for item in batch]

        # train generator
        result = None
        if optimizer_idx == 0:
            result = self.generator_step(ct3xr2_xr2_unpaired, ct3xr2_ct3_unpaired)

        # train discriminator
        if optimizer_idx == 1:
            result = self.discriminator_step(ct3xr2_xr2_unpaired, ct3xr2_ct3_unpaired)

        ct3xr2_xr2_fake = self.generator(ct3xr2_ct3_unpaired)
        grid = torchvision.utils.make_grid(ct3xr2_xr2_fake, 8, normalize=True, range=(-1, 1))
        self.logger.experiment.add_image('ct3xr2_xr2_fake', grid, self.current_epoch)
        grid = torchvision.utils.make_grid(ct3xr2_xr2_unpaired, 8, normalize=True, range=(-1, 1))
        self.logger.experiment.add_image('ct3xr2_xr2_unpaired', grid, self.current_epoch)
        
        ct3xr2_xr2_recon = self.generator(ct3xr2_ct3_paired)
        grid = torchvision.utils.make_grid(ct3xr2_xr2_recon, 8, normalize=True, range=(-1, 1))
        self.logger.experiment.add_image('ct3xr2_xr2_recon', grid, self.current_epoch)
        grid = torchvision.utils.make_grid(ct3xr2_xr2_paired, 8, normalize=True, range=(-1, 1))
        self.logger.experiment.add_image('ct3xr2_xr2_paired', grid, self.current_epoch)

        return result


    def validation_step(self, batch, batch_idx):
        ct3xr2_ct3_unpaired,\
        ct3xr2_xr2_unpaired,\
        xr2ct3_xr2_unpaired,\
        xr2ct3_ct3_unpaired,\
        ct3xr2_ct3_paired,\
        ct3xr2_xr2_paired,\
        xr2ct3_xr2_paired,\
        xr2ct3_ct3_paired,\
        ct3lung3_img_paired,\
        ct3lung3_lbl_paired,\
        ct3covid3_img_paired,\
        ct3covid3_lbl_paired,\
        xr2lung2_img_paired,\
        xr2lung2_lbl_paired,\
        xr2covid2_img_paired,\
        xr2covid2_lbl_paired = [ item / 128.0 - 1.0 for item in batch]

        
        ct3xr2_xr2_recon = self.generator(ct3xr2_ct3_paired)
        val_loss = nn.L1Loss()(ct3xr2_xr2_recon, ct3xr2_xr2_paired)
        return {'val_loss': val_loss}

    def __dataloader(self):
        train_tfm = AB.Compose([
            
            AB.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            AB.RandomGamma(gamma_limit=(80, 120), p=0.5),
            AB.GaussianBlur(p=0.5),
            AB.GaussNoise(p=0.5),
            AB.Resize(width=self.hparams.dimy, height=self.hparams.dimx, p=1.0),
        ])
        valid_tfm = AB.Compose([
            AB.Resize(width=self.hparams.dimy, height=self.hparams.dimx, p=1.0),
           
        ])
       

        train_ds = CustomNativeDataset(ct3xr2_xr2_unpaired_dir=train_ct3xr2_xr2_unpaired_dir,
                                   ct3xr2_ct3_unpaired_dir=train_ct3xr2_ct3_unpaired_dir,
                                   xr2ct3_xr2_unpaired_dir=train_xr2ct3_xr2_unpaired_dir,
                                   xr2ct3_ct3_unpaired_dir=train_xr2ct3_ct3_unpaired_dir,
                                   ct3xr2_ct3_paired_dir=train_ct3xr2_ct3_paired_dir,
                                   ct3xr2_xr2_paired_dir=train_ct3xr2_xr2_paired_dir,
                                   xr2ct3_ct3_paired_dir=train_xr2ct3_ct3_paired_dir,
                                   xr2ct3_xr2_paired_dir=train_xr2ct3_xr2_paired_dir,
                                   ct3lung3_img_paired_dir=train_ct3lung3_img_paired_dir,
                                   ct3lung3_lbl_paired_dir=train_ct3lung3_lbl_paired_dir,
                                   ct3covid3_img_paired_dir=train_ct3covid3_img_paired_dir,
                                   ct3covid3_lbl_paired_dir=train_ct3covid3_lbl_paired_dir,
                                   xr2lung2_img_paired_dir=train_xr2lung2_img_paired_dir,
                                   xr2lung2_lbl_paired_dir=train_xr2lung2_lbl_paired_dir,
                                   xr2covid2_img_paired_dir=train_xr2covid2_img_paired_dir,
                                   xr2covid2_lbl_paired_dir=train_xr2covid2_lbl_paired_dir,
                                   train_or_valid='train',
                                   size=1000,
                                   transforms=train_tfm,
                                   dimz=self.hparams.dimz,
                                   dimy=self.hparams.dimy,
                                   dimx=self.hparams.dimx
                                   )
        valid_ds = CustomNativeDataset(ct3xr2_xr2_unpaired_dir=valid_ct3xr2_xr2_unpaired_dir,
                                   ct3xr2_ct3_unpaired_dir=valid_ct3xr2_ct3_unpaired_dir,
                                   xr2ct3_xr2_unpaired_dir=valid_xr2ct3_xr2_unpaired_dir,
                                   xr2ct3_ct3_unpaired_dir=valid_xr2ct3_ct3_unpaired_dir,
                                   ct3xr2_ct3_paired_dir=valid_ct3xr2_ct3_paired_dir,
                                   ct3xr2_xr2_paired_dir=valid_ct3xr2_xr2_paired_dir,
                                   xr2ct3_ct3_paired_dir=valid_xr2ct3_ct3_paired_dir,
                                   xr2ct3_xr2_paired_dir=valid_xr2ct3_xr2_paired_dir,
                                   ct3lung3_img_paired_dir=valid_ct3lung3_img_paired_dir,
                                   ct3lung3_lbl_paired_dir=valid_ct3lung3_lbl_paired_dir,
                                   ct3covid3_img_paired_dir=valid_ct3covid3_img_paired_dir,
                                   ct3covid3_lbl_paired_dir=valid_ct3covid3_lbl_paired_dir,
                                   xr2lung2_img_paired_dir=valid_xr2lung2_img_paired_dir,
                                   xr2lung2_lbl_paired_dir=valid_xr2lung2_lbl_paired_dir,
                                   xr2covid2_img_paired_dir=valid_xr2covid2_img_paired_dir,
                                   xr2covid2_lbl_paired_dir=valid_xr2covid2_lbl_paired_dir,
                                   train_or_valid='valid',
                                   size=30,
                                   transforms=valid_tfm,
                                   dimz=self.hparams.dimz,
                                   dimy=self.hparams.dimy,
                                   dimx=self.hparams.dimx
                                   )

        

        train_loader = DataLoader(train_ds, 
            num_workers=self.hparams.num_workers, 
            batch_size=self.hparams.batch_size, 
            pin_memory=True, 
            shuffle=True, 
            worker_init_fn=worker_init_fn
        )
        valid_loader = DataLoader(valid_ds, 
            num_workers=self.hparams.num_workers, 
            batch_size=self.hparams.batch_size, 
            pin_memory=True, 
            shuffle=False, 
            worker_init_fn=worker_init_fn
        )
        return {
            'train': train_loader, 
            'valid': valid_loader, 
        }

    def train_dataloader(self):
        return self.__dataloader()['train']

    def val_dataloader(self):
        return self.__dataloader()['valid']

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])
        return parser

def main(hparams):
    model = Model(hparams)
    os.makedirs(hparams.lgdir, exist_ok=True)
    try:
        lgdir = sorted(os.listdir(hparams.lgdir))[-1]
    except IndexError:
        lgdir = os.path.join(hparams.lgdir, 'version_0')
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(lgdir, 'checkpoints'),
        save_top_k=5,
        verbose=True,
    )
    stop_callback = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=500,
        verbose=True,
    )
    trainer = Trainer(
        auto_lr_find=False,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=False, #stop_callback,
        gpus=hparams.gpus,
        callbacks=[LearningRateLogger()],
        max_epochs=hparams.epochs,
        accumulate_grad_batches=hparams.grad_batches,
        # gradient_clip_val=0.5,
        track_grad_norm=1,
        distributed_backend=hparams.distributed_backend,
        precision=16 if hparams.use_amp else 32, 
        profiler=True
    )

    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--dimx', type=int, default=256)
    parser.add_argument('--dimy', type=int, default=256)
    parser.add_argument('--dimz', type=int, default=64)
    parser.add_argument('--lgdir', type=str, default='lightning_logs')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--gpus", default=-1, help="number of available GPUs")
    parser.add_argument('--distributed-backend', type=str, default='ddp', choices=('dp', 'ddp', 'ddp2'),
                        help='supports three options dp, ddp, ddp2')
    parser.add_argument('--use_amp', default=True, action='store_true', help='if true uses 16 bit precision')
    parser.add_argument("--num_workers", type=int, default=8, help="size of the workers")
    parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
    parser.add_argument("--nb_layer", type=int, default=5, help="number of layers on u-net")
    parser.add_argument("--features", type=int, default=8, help="number of features in single layer")
    parser.add_argument("--bilinear", action='store_true', default=False,
                        help="whether to use bilinear interpolation or transposed")
    parser.add_argument("--grad_batches", type=int, default=1, help="number of batches to accumulate")
    parser.add_argument("--epochs", type=int, default=500, help="number of epochs to train")
    parser.add_argument("--log_wandb", action='store_true', help="log training on Weights & Biases")
    #
    parser.add_argument("--batch_size", type=int, default=4)
   

    parser = Model.add_model_specific_args(parser)
    hparams = parser.parse_args()

    if hparams.seed:
        seed_everything(hparams.seed)

    main(hparams) 