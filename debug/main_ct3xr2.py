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

# from model import INet
# from model import PNet
from model import Generator, Discriminator, PGNet #, Generator, Discriminator

train_image_paired_dir = ['/u01/data/iXrayCT_COVID/data_resized/train/paired/RADIOPAEDIA/pos/xr2/',
                          '/u01/data/iXrayCT_COVID/data_resized/train/paired/RADIOPAEDIA/neg/xr2/',
                          '/u01/data/iXrayCT_COVID/data_resized/train/paired/BIMCV/pos/xr2/',
                          ]
train_label_paired_dir = ['/u01/data/iXrayCT_COVID/data_resized/train/paired/RADIOPAEDIA/pos/ct3/',
                          '/u01/data/iXrayCT_COVID/data_resized/train/paired/RADIOPAEDIA/neg/ct3/',
                          '/u01/data/iXrayCT_COVID/data_resized/train/paired/BIMCV/pos/ct3/',
                          ]
valid_image_paired_dir = ['/u01/data/iXrayCT_COVID/data_resized/test/paired/RADIOPAEDIA/pos/xr2/',
                          '/u01/data/iXrayCT_COVID/data_resized/test/paired/RADIOPAEDIA/neg/xr2/',
                          '/u01/data/iXrayCT_COVID/data_resized/test/paired/BIMCV/pos/xr2/',
                          ]
valid_label_paired_dir = ['/u01/data/iXrayCT_COVID/data_resized/test/paired/RADIOPAEDIA/pos/ct3/',
                          '/u01/data/iXrayCT_COVID/data_resized/test/paired/RADIOPAEDIA/neg/ct3/',
                          '/u01/data/iXrayCT_COVID/data_resized/test/paired/BIMCV/pos/ct3/',
                          ]

train_image_unpaired_dir = ['/u01/data/iXrayCT_COVID/data_resized/train/paired/RADIOPAEDIA/pos/xr2/',
                            '/u01/data/iXrayCT_COVID/data_resized/train/paired/RADIOPAEDIA/neg/xr2/',
                            '/u01/data/iXrayCT_COVID/data_resized/train/unpaired/IEEE8023/pos/xr2/',
                            '/u01/data/iXrayCT_COVID/data_resized/train/unpaired/IEEE8023/neg/xr2/',
                            '/u01/data/iXrayCT_COVID/data_resized/train/unpaired/NLMMC/neg/xr2/',
                            '/u01/data/iXrayCT_COVID/data_resized/train/paired/BIMCV/pos/xr2/',
                            ]
train_label_unpaired_dir = ['/u01/data/iXrayCT_COVID/data_resized/train/paired/RADIOPAEDIA/pos/ct3/',
                            '/u01/data/iXrayCT_COVID/data_resized/train/paired/RADIOPAEDIA/neg/ct3/',
                            '/u01/data/iXrayCT_COVID/data_resized/train/unpaired/MOSMED/pos/ct3/',
                            '/u01/data/iXrayCT_COVID/data_resized/train/unpaired/LNDB/neg/ct3/',
                            '/u01/data/iXrayCT_COVID/data_resized/train/unpaired/DSB3/neg/ct3/',
                            '/u01/data/iXrayCT_COVID/data_resized/train/paired/BIMCV/pos/ct3/',
                            ]
valid_image_unpaired_dir = ['/u01/data/iXrayCT_COVID/data_resized/test/paired/RADIOPAEDIA/pos/xr2/',
                            '/u01/data/iXrayCT_COVID/data_resized/test/paired/RADIOPAEDIA/neg/xr2/',
                            '/u01/data/iXrayCT_COVID/data_resized/test/paired/BIMCV/pos/xr2/',
                            ]
valid_label_unpaired_dir = ['/u01/data/iXrayCT_COVID/data_resized/test/paired/RADIOPAEDIA/pos/ct3/',
                            '/u01/data/iXrayCT_COVID/data_resized/test/paired/RADIOPAEDIA/neg/ct3/',
                            '/u01/data/iXrayCT_COVID/data_resized/test/paired/BIMCV/pos/ct3/',
                            ]


def worker_init_fn(worker_id):
    torch.initial_seed()


class CustomNativeDataset(Dataset):
    def __init__(self,
                 imagepaireddir,
                 labelpaireddir,
                 imageunpaireddir,
                 labelunpaireddir,
                 train_or_valid='train',
                 size=10000,
                 transforms=None
                 ):
        self.size = size
        self.is_train = True if train_or_valid == 'train' else False
        self.imagepaireddir = imagepaireddir
        self.labelpaireddir = labelpaireddir
        self.imageunpaireddir = imageunpaireddir
        self.labelunpaireddir = labelunpaireddir

        self.imagepairedfiles = [glob.glob(os.path.join(folder, '*.*')) for folder in self.imagepaireddir]
        self.labelpairedfiles = [glob.glob(os.path.join(folder, '*.*')) for folder in self.labelpaireddir]
        self.imagepairedfiles = natsorted([item for sublist in self.imagepairedfiles for item in sublist])
        self.labelpairedfiles = natsorted([item for sublist in self.labelpairedfiles for item in sublist])
        self.transforms = transforms
        assert len(self.imagepairedfiles) == len(self.labelpairedfiles)

        self.imageunpairedfiles = [glob.glob(os.path.join(folder, '*.*')) for folder in self.imageunpaireddir]
        self.labelunpairedfiles = [glob.glob(os.path.join(folder, '*.*')) for folder in self.labelunpaireddir]
        self.imageunpairedfiles = natsorted([item for sublist in self.imageunpairedfiles for item in sublist])
        self.labelunpairedfiles = natsorted([item for sublist in self.labelunpairedfiles for item in sublist])

        print(len(self.imagepairedfiles), len(self.labelpairedfiles),
              len(self.imageunpairedfiles), len(self.labelunpairedfiles))

    def __len__(self):
        return self.size if self.is_train else len(self.imagepairedfiles)

    def __call__(self):
        np.random.seed(datetime.datetime.now().second +
                       datetime.datetime.now().millisecond)

    def __getitem__(self, idx):
        pidx = torch.randint(len(self.imagepairedfiles), (1, 1))
        imagepaired = skimage.io.imread(self.imagepairedfiles[pidx])
        labelpaired = skimage.io.imread(self.labelpairedfiles[pidx])

        aidx = torch.randint(len(self.imageunpairedfiles), (1, 1))
        bidx = torch.randint(len(self.labelunpairedfiles), (1, 1))
        imageunpaired = skimage.io.imread(self.imageunpairedfiles[aidx])
        labelunpaired = skimage.io.imread(self.labelunpairedfiles[bidx])

        if self.transforms is not None:
            untransformed = self.transforms(image=np.transpose(np.expand_dims(imageunpaired, 0), (1, 2, 0)))
            imageunpaired = np.squeeze(np.transpose(untransformed['image'], (2, 0, 1)), 0)
            untransformed = self.transforms(image=np.transpose(labelunpaired, (1, 2, 0)))
            labelunpaired = np.transpose(untransformed['image'], (2, 0, 1))

        scale = [64.0 / labelpaired.shape[0], 
                 256.0/labelpaired.shape[1], 
                 256.0/labelpaired.shape[2]]

        def bbox2(img):
            rows = np.any(img, axis=1)
            cols = np.any(img, axis=0)
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            return img[ymin:ymax+1, xmin:xmax+1]

        imagepaired = bbox2(imagepaired)                   
        imageunpaired = bbox2(imageunpaired)                   
        imagepaired = cv2.resize(imagepaired, (256, 256))
        labelpaired = scipy.ndimage.zoom(labelpaired, scale, order=3)
        imageunpaired = cv2.resize(imageunpaired, (256, 256))
        labelunpaired = scipy.ndimage.zoom(labelunpaired, scale, order=3)

        return torch.Tensor(imagepaired).float().unsqueeze_(0),   \
               torch.Tensor(labelpaired).float(),                 \
               torch.Tensor(imageunpaired).float().unsqueeze_(0), \
               torch.Tensor(labelunpaired).float(),  
########################################################################################################
def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = torch.autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True)
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / np.sqrt(fake_img.shape[2] * fake_img.shape[3])
    grad, = torch.autograd.grad(outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True)
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)
    path_penalty = (path_lengths - path_mean).pow(2).mean()
    return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)
    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)
    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)
    else:
        return [make_noise(batch, latent_dim, 1, device)]


########################################################################################################
class Model(pl.LightningModule):
    def __init__(self, hparams, n_classes=1):
        super(Model, self).__init__()
        self.hparams = hparams
        # self.example_input_array = torch.rand(self.hparams.batch_size,  1, 256, 256),  \
        #                            torch.rand(self.hparams.batch_size, 64, 256, 256),  \
        #                            torch.rand(self.hparams.batch_size,  1, 256, 256),  \
        #                            torch.rand(self.hparams.batch_size, 64, 256, 256)
        self.example_input_array = torch.rand(self.hparams.batch_size, 64, 256, 256)
        
        self.save_hyperparameters()
        self.hparams = hparams
        self.hparams.latent = 512
        self.hparams.n_mlp = 8
        self.generator, self.g_ema = self.init_generator()
        self.discriminator = self.init_discriminator()

        self.mean_path_length = 0
        self.accum = 0.5 ** (32 / (10 * 1000))
        self.register_buffer('sample_z', torch.randn(self.hparams.batch_size, self.hparams.latent))
        # self.example_input_array = torch.rand_like(self.sample_z)

    def forward(self, z):
        return self.generator(z)

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        # return [optimizer], [scheduler]
        g_reg_ratio = self.hparams.g_reg_every / (self.hparams.g_reg_every + 1)
        d_reg_ratio = self.hparams.d_reg_every / (self.hparams.d_reg_every + 1)
        g_optim = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr * g_reg_ratio, betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio))
        d_optim = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr * d_reg_ratio, betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio))
        return d_optim, g_optim

    def init_discriminator(self):
        discriminator = Discriminator(self.hparams.shape, channel_multiplier=self.hparams.channel_multiplier)
        return discriminator

    # def discriminator_loss(self, real_img):
    #     requires_grad(self.generator, False)
    #     requires_grad(self.discriminator, True)

    #     noise = mixing_noise(self.hparams.batch_size, self.hparams.latent, self.hparams.mixing, self.device)
    #     fake_img, _ = self.generator(noise)
    #     fake_pred = self.discriminator(fake_img)
    #     real_pred = self.discriminator(real_img)
    #     d_loss = d_logistic_loss(real_pred, fake_pred)

    #     real_score = real_pred.mean()
    #     fake_score = fake_pred.mean()

    #     return d_loss, real_score, fake_score

    # def discriminator_regularization_loss(self, real_img):
    #     real_img.requires_grad = True
    #     real_pred = self.discriminator(real_img)
    #     r1_loss = d_r1_loss(real_pred, real_img)
    #     d_reg_loss = (self.hparams.r1 / 2 * r1_loss * self.hparams.d_reg_every + 0 * real_pred[0]).mean()
    #     return d_reg_loss, r1_loss

    # def discriminator_step(self, real_img, regularize=False):
    #     d_loss, real_score, fake_score = self.discriminator_loss(real_img)

    #     tqdm_dict = {'d_loss': d_loss}
    #     log_dict = {'d_loss': d_loss, 'real_score': real_score, 'fake_score': fake_score}

    #     if regularize:
    #         d_reg_loss, r1_loss = self.discriminator_regularization_loss(real_img)
    #         d_loss += d_reg_loss
    #         log_dict.update({'r1': r1_loss})

    #     output = {
    #         'loss': d_loss,
    #         'progress_bar': tqdm_dict,
    #         'log': log_dict,
    #     }
    #     return output

    def init_generator(self):
        # generator = Generator(self.hparams.shape, self.hparams.latent, self.hparams.n_mlp, channel_multiplier=self.hparams.channel_multiplier)
        # g_ema = Generator(self.hparams.shape, self.hparams.latent, self.hparams.n_mlp, channel_multiplier=self.hparams.channel_multiplier)
        generator = PGNet(self.hparams)
        g_ema = PGNet(self.hparams)
        g_ema.eval()
        accumulate(g_ema, generator, 0)
        return generator, g_ema

    # def generator_loss(self):
    #     requires_grad(self.generator, True)
    #     requires_grad(self.discriminator, False)

    #     noise = mixing_noise(self.hparams.batch_size, self.hparams.latent, self.hparams.mixing, self.device)
    #     fake_img, _ = self.generator(noise)

    #     fake_pred = self.discriminator(fake_img)
    #     g_loss = g_nonsaturating_loss(fake_pred)
    #     return g_loss

    # def generator_regularization_loss(self):
    #     path_batch_size = max(1, self.hparams.batch_size // self.hparams.path_batch_shrink)
    #     noise = mixing_noise(path_batch_size, self.hparams.latent, self.hparams.mixing, self.device)
    #     fake_img, latents = self.generator(noise, return_latents=True)

    #     path_loss, self.mean_path_length, path_lengths = g_path_regularize(fake_img, latents, self.mean_path_length)
    #     weighted_path_loss = self.hparams.path_regularize * self.hparams.g_reg_every * path_loss

    #     if self.hparams.path_batch_shrink:
    #         weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

    #     return weighted_path_loss, path_loss, path_lengths

    # def generator_step(self, regularize=False):
    #     g_loss = self.generator_loss()
    #     tqdm_dict = {'g_loss': g_loss}
    #     log_dict = {'g_loss': g_loss}

    #     if regularize:
    #         weighted_path_loss, path_loss, path_lengths = self.generator_regularization_loss()
    #         g_loss += weighted_path_loss
    #         log_dict["path"] = path_loss
    #         log_dict["path_length"] = path_lengths.mean()

    #     output = {
    #         'loss': g_loss,
    #         'progress_bar': tqdm_dict,
    #         'log': log_dict,
    #     }
    #     return output

    def training_step(self, batch, batch_idx, optimizer_idx):
        d_regularize = (batch_idx % self.hparams.d_reg_every == 0)
        g_regularize = (batch_idx % self.hparams.g_reg_every == 0)
        pxr2, pct3, uxr2, uct3 = batch
        pxr2, pct3, uxr2, uct3 = pxr2/128-1, pct3/128-1, uxr2/128-1, uct3/128-1
        real_img = uxr2
        pair_img = pxr2

        result = None
        if optimizer_idx == 0: # Discriminator
            # result = self.discriminator_step(real_img, regularize=d_regularize)
            # d_loss, real_score, fake_score = self.discriminator_loss(real_img)
            requires_grad(self.generator, False)
            requires_grad(self.discriminator, True)

            # noise = mixing_noise(self.hparams.batch_size, self.hparams.latent, self.hparams.mixing, self.device)
            # print('noise', noise)
            # fake_img, _ = self.generator(noise)
            fake_img, _ = self.generator(uct3)
            fake_pred = self.discriminator(fake_img)
            real_pred = self.discriminator(real_img)
            d_loss = d_logistic_loss(real_pred, fake_pred)

            real_score = real_pred.mean()
            fake_score = fake_pred.mean()

            tqdm_dict = {'d_loss': d_loss}
            log_dict = {'d_loss': d_loss, 'real_score': real_score, 'fake_score': fake_score}

            if d_regularize:
                # d_reg_loss, r1_loss = self.discriminator_regularization_loss(real_img)
                real_img.requires_grad = True
                real_pred = self.discriminator(real_img)
                r1_loss = d_r1_loss(real_pred, real_img)
                d_reg_loss = (self.hparams.r1 / 2 * r1_loss * self.hparams.d_reg_every + 0 * real_pred[0]).mean()

                d_loss += d_reg_loss
                log_dict.update({'r1': r1_loss})

            output = {
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': log_dict,
            }

        if optimizer_idx == 1: # Generator
            # result = self.generator_step(regularize=g_regularize)
            # g_loss = self.generator_loss()
            requires_grad(self.generator, True)
            requires_grad(self.discriminator, False)

            # noise = mixing_noise(self.hparams.batch_size, self.hparams.latent, self.hparams.mixing, self.device)
            # print('noise.shape', noise.shape)
            # fake_img, _ = self.generator(noise)
            fake_img, _ = self.generator(uct3)
            exr2    , _ = self.generator(pct3)
            fake_pred = self.discriminator(fake_img)
            g_loss = g_nonsaturating_loss(fake_pred) + nn.L1Loss()(pxr2, exr2)

            tqdm_dict = {'g_loss': g_loss}
            log_dict = {'g_loss': g_loss}

            if g_regularize:
                # weighted_path_loss, path_loss, path_lengths = self.generator_regularization_loss()
                path_batch_size = max(1, self.hparams.batch_size // self.hparams.path_batch_shrink)
                # noise = mixing_noise(path_batch_size, self.hparams.latent, self.hparams.mixing, self.device)
                # fake_img, latents = self.generator(noise, return_latents=True)
                fake_img, latents = self.generator(uct3)

                path_loss, self.mean_path_length, path_lengths = g_path_regularize(fake_img, latents, self.mean_path_length)
                weighted_path_loss = self.hparams.path_regularize * self.hparams.g_reg_every * path_loss

                if self.hparams.path_batch_shrink:
                    weighted_path_loss += 0 * fake_img[0, 0, 0, 0]
                g_loss += weighted_path_loss
                log_dict["path"] = path_loss
                log_dict["path_length"] = path_lengths.mean()

            output = {
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': log_dict,
            }
            accumulate(self.g_ema, self.generator, self.accum)

        if batch_idx % self.hparams.img_log_frequency == 0:
            self.g_ema.eval()
            fake_img, _ = self.g_ema(uct3)
            grid = torchvision.utils.make_grid(fake_img, 8, normalize=True, range=(-1, 1))
            self.logger.experiment.add_image('fake_img', grid, self.current_epoch)
            grid = torchvision.utils.make_grid(real_img, 8, normalize=True, range=(-1, 1))
            self.logger.experiment.add_image('real_img', grid, self.current_epoch)
            grid = torchvision.utils.make_grid(pair_img, 8, normalize=True, range=(-1, 1))
            self.logger.experiment.add_image('pair_img', grid, self.current_epoch)
        return output


    def __dataloader(self):
        # train_tfm = None
        # valid_tfm = None
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
       

        train_ds = CustomNativeDataset(imagepaireddir=train_image_paired_dir, 
                                       labelpaireddir=train_label_paired_dir, 
                                       imageunpaireddir=train_image_unpaired_dir, 
                                       labelunpaireddir=train_label_unpaired_dir, 
                                       train_or_valid='train', 
                                       transforms=train_tfm)
        valid_ds = CustomNativeDataset(imagepaireddir=valid_image_paired_dir, 
                                       labelpaireddir=valid_label_paired_dir, 
                                       imageunpaireddir=valid_image_unpaired_dir, 
                                       labelunpaireddir=valid_label_unpaired_dir, 
                                       train_or_valid='valid', 
                                       transforms=valid_tfm)
        

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
    parser.add_argument('--use_amp', default=False, action='store_true', help='if true uses 16 bit precision')
    parser.add_argument("--num_workers", type=int, default=8, help="size of the workers")
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument("--nb_layer", type=int, default=5, help="number of layers on u-net")
    parser.add_argument("--features", type=int, default=8, help="number of features in single layer")
    parser.add_argument("--bilinear", action='store_true', default=False,
                        help="whether to use bilinear interpolation or transposed")
    parser.add_argument("--grad_batches", type=int, default=1, help="number of batches to accumulate")
    parser.add_argument("--epochs", type=int, default=500, help="number of epochs to train")
    parser.add_argument("--log_wandb", action='store_true', help="log training on Weights & Biases")
    #
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_sample", type=int, default=4)
    parser.add_argument("--shape", type=int, default=256)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--r1", type=float, default=10)
    parser.add_argument("--path_regularize", type=float, default=2)
    parser.add_argument("--path_batch_shrink", type=int, default=2)
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--g_reg_every", type=int, default=4)
    parser.add_argument("--mixing", type=float, default=0.9)
    parser.add_argument("--channel_multiplier", type=int, default=2)
    parser.add_argument("--img_log_frequency", type=int, default=1)

    parser = Model.add_model_specific_args(parser)
    hparams = parser.parse_args()

    if hparams.seed:
        # random.seed(hparams.seed)
        # os.environ['PYTHONHASHSEED'] = str(hparams.seed)
        # np.random.seed(hparams.seed)
        # torch.manual_seed(hparams.seed)
        # torch.cuda.manual_seed(hparams.seed)
        # torch.cuda.manual_seed_all(hparams.seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = True
        seed_everything(hparams.seed)

    main(hparams)