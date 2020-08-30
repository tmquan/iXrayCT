import os
import glob
from argparse import ArgumentParser

import logging
from collections import OrderedDict

from natsort import natsorted
from tqdm import tqdm

import random
import numpy as np
import cv2, skimage.io, skimage.transform, skimage.exposure
import albumentations as AB
# import volumentations as VL
# import torchio as TI
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

# import kornia
import scipy.ndimage
import warnings

# from model import ReconNet, INet
warnings.filterwarnings('ignore', '.*output shape of zoom.*')

train_image_dir = ['/u01/data/iXrayCT_COVID/data_resized/train/paired/RADIOPAEDIA/pos/xr2/', 
                   '/u01/data/iXrayCT_COVID/data_resized/train/paired/RADIOPAEDIA/neg/xr2/']
train_label_dir = ['/u01/data/iXrayCT_COVID/data_resized/train/paired/RADIOPAEDIA/pos/ct3/', 
                   '/u01/data/iXrayCT_COVID/data_resized/train/paired/RADIOPAEDIA/neg/ct3/']
valid_image_dir = ['/u01/data/iXrayCT_COVID/data_resized/test/paired/RADIOPAEDIA/pos/xr2/', 
                   '/u01/data/iXrayCT_COVID/data_resized/test/paired/RADIOPAEDIA/neg/xr2/']
valid_label_dir = ['/u01/data/iXrayCT_COVID/data_resized/test/paired/RADIOPAEDIA/pos/ct3/', 
                   '/u01/data/iXrayCT_COVID/data_resized/test/paired/RADIOPAEDIA/neg/ct3/']



class CustomNativeDataset(Dataset):
    def __init__(self, 
        imagedir, 
        labeldir, 
        train_or_valid='train',
        size=500, 
        transforms=None
    ):
        # print('\n')
        self.size = size
        self.is_train = True if train_or_valid=='train' else False
        self.imagedir = imagedir #if self.is_train else imagedir.replace('train', 'test')
        self.labeldir = labeldir #if self.is_train else labeldir.replace('train', 'test')
        
        self.imagefiles = [glob.glob(os.path.join(folder, '*.*')) for folder in self.imagedir]
        self.labelfiles = [glob.glob(os.path.join(folder, '*.*')) for folder in self.labeldir]
        self.imagefiles = natsorted([item for sublist in self.imagefiles for item in sublist])
        self.labelfiles = natsorted([item for sublist in self.labelfiles for item in sublist])
        self.transforms = transforms
        assert len(self.imagefiles) == len(self.labelfiles)
        # print(self.imagefiles)
    def __len__(self):
        if self.size > len(self.imagefiles) or self.size is None: 
            return len(self.imagefiles)
        else:
            return self.size
 
    def __getitem__(self, idx):
        # print(self.imagefiles[idx], self.labelfiles[idx])
        image = skimage.io.imread(self.imagefiles[idx])
        label = skimage.io.imread(self.labelfiles[idx])
        # if image.shape != label.shape:
        #     print(self.imagefiles[idx])
        #     print(self.labelfiles[idx])
        # assert image.shape == label.shape

        if self.transforms is not None:
            if np.random.randint(2) % 2:
                a = np.random.randint(0, 255)
                b = np.random.randint(a, 256)
                image = label[:,a:b,:]
                image = label.mean(axis=1).astype(np.uint8)
            transformed = self.transforms(image=np.transpose(np.expand_dims(image, 0), (1, 2, 0)))
            image = np.squeeze(np.transpose(transformed['image'], (2, 0, 1)), 0)
            transformed = self.transforms(image=np.transpose(label, (1, 2, 0)))
            label = np.transpose(transformed['image'], (2, 0, 1))
            
        scale = [64.0/label.shape[0], 256.0/label.shape[1], 256.0/label.shape[2]]
        # print(image.min(), image.max(), label.min(), label.max())

        # image = scipy.ndimage.zoom(image, scale, order=3)
        image = cv2.resize(image, (256, 256))
        label = scipy.ndimage.zoom(label, scale, order=3)
        # print(image.shape, label.shape)        
        # print(str(idx).zfill(3), image.shape, label.shape)
        return torch.Tensor(image).float().unsqueeze_(0), \
               torch.Tensor(label).float()

class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)

class Squeeze(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)

class WarmupConv2d(nn.Module):
    def __init__(self, 
        source_channels=32, 
        output_channels=32,
    ):
        super().__init__()
        self.pre = nn.Conv2d(source_channels, output_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.net = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(output_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        tmp = self.pre(x)
        ret = self.net(tmp) 
        return ret

class DoubleConv2d(nn.Module):
    def __init__(self, 
        source_channels=32, 
        output_channels=32,
    ):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(source_channels, output_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(output_channels),
            nn.LeakyReLU(inplace=True),
        )
        self.net = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(output_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        tmp = self.pre(x)
        ret = self.net(tmp) 
        return ret

class SingleDeconv3d(nn.Module):
    def __init__(self, 
        source_channels=32, 
        output_channels=32,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose3d(source_channels, output_channels, kernel_size=2, stride=2, padding=0, bias=False),
            nn.InstanceNorm3d(output_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        ret = self.net(x)
        return ret

class DoubleDeconv3d(nn.Module):
    def __init__(self, 
        source_channels=32, 
        output_channels=32,
    ):
        super().__init__()
        self.pre = nn.Sequential(
            nn.ConvTranspose3d(source_channels, output_channels, kernel_size=2, stride=2, padding=0, bias=False),
            nn.InstanceNorm3d(output_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.net = nn.Sequential(
            nn.ConvTranspose3d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(output_channels),
            nn.LeakyReLU(inplace=True),
        )
        
    def forward(self, x):
        tmp = self.pre(x)
        ret = self.net(tmp)  
        return ret

class INet(nn.Module):
    def __init__(self, source_channels=1, output_channels=1, num_filters=64):
        super().__init__()
        self.net = nn.Sequential(
            # 2D
            WarmupConv2d(source_channels, num_filters*4),
            DoubleConv2d(num_filters*4, num_filters*8),
            DoubleConv2d(num_filters*8, num_filters*16),
            DoubleConv2d(num_filters*16, num_filters*32),
            DoubleConv2d(num_filters*32, num_filters*64),
            

            # Transformation
            nn.Conv2d(num_filters*64, num_filters*64, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.LeakyReLU(inplace=True),
            nn.Tanh(),
            Reshape(num_filters*32, 2, 8, 8),
            nn.ConvTranspose3d(num_filters*32, num_filters*32, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.LeakyReLU(inplace=True),

            # 3D
            SingleDeconv3d(num_filters*32, num_filters*16), #2, 1024, 4, 16, 16]
            DoubleDeconv3d(num_filters*16, num_filters*8),
            DoubleDeconv3d(num_filters*8, num_filters*4),
            DoubleDeconv3d(num_filters*4, num_filters*2),
            DoubleDeconv3d(num_filters*2, num_filters*1),
            nn.ConvTranspose3d(num_filters*1, output_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(inplace=True),
            Squeeze(dim=1),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.net(x * 2.0 - 1.0) / 2.0 + 0.5

class Model(pl.LightningModule):
    def __init__(self, hparams, n_classes=1):
        super(Model, self).__init__()
        self.example_input_array = torch.rand(2, 1, 256, 256)
        self.hparams = hparams
        self.inet = INet(
            source_channels=1, 
            output_channels=1
        )
        print(self.inet)
    def forward(self, x):
        # return self.rnet(x)
        return self.inet(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        x = x / 255.0
        y = y / 255.0
        y_hat = self.forward(x)
        loss = nn.L1Loss()(y_hat, y)
        mid = int(y.shape[1]/2)
        vis_images = torch.cat([x, y[:,mid:mid+1,:,:], y_hat[:,mid:mid+1,:,:]], dim=-1)#[:8]
        grid = torchvision.utils.make_grid(vis_images, nrow=2, padding=0)
        self.logger.experiment.add_image('train_vis', grid, self.current_epoch)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        x, y = batch

        x = x / 255.0
        y = y / 255.0
        y_hat = self.forward(x)
        # print(x.shape, y.shape, y_hat.shape)
        loss = nn.L1Loss()(y_hat, y)
        mid = int(y.shape[1]/2)
        vis_images = torch.cat([x, y[:,mid:mid+1,:,:], y_hat[:,mid:mid+1,:,:]], dim=-1)#[:8]
        grid = torchvision.utils.make_grid(vis_images, nrow=2, padding=0)
        self.logger.experiment.add_image('valid_vis', grid, self.current_epoch)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def __dataloader(self):
        # train_tfm = None
        valid_tfm = None
        train_tfm = AB.Compose([
            # AB.ToFloat(), 
            # AB.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, p=1.0),
            # AB.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.15, rotate_limit=30, border_mode=cv2.BORDER_CONSTANT, p=0.8),
            # AB.Resize(height=512, width=512, p=1.0), 
            # AB.CropNonEmptyMaskIfExists(height=320, width=320, p=0.8), 
            # AB.RandomScale(scale_limit=(0.8, 1.2), p=0.8),
            # AB.Equalize(p=0.8),
            # AB.CLAHE(p=0.8),
            AB.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9),
            AB.RandomGamma(gamma_limit=(80, 120), p=0.9),
            AB.GaussianBlur(p=0.05),
            AB.GaussNoise(p=0.05),
            AB.Resize(width=self.hparams.dimy, height=self.hparams.dimx, p=1.0),
            # AB.ToTensor(),
        ])
        # valid_tfm = AB.Compose([
        #     # AB.ToFloat(), 
        #     # AB.Equalize(p=1.0),
        #     # AB.CLAHE(p=1.0),
        #     # AB.Resize(width=self.hparams.dimy, height=self.hparams.dimx),
        # ])

        train_ds = CustomNativeDataset(imagedir=train_image_dir, 
                                       labeldir=train_label_dir, 
                                       train_or_valid='train', 
                                       transforms=train_tfm)
        valid_ds = CustomNativeDataset(imagedir=valid_image_dir, 
                                       labeldir=valid_label_dir, 
                                       train_or_valid='valid', 
                                       transforms=valid_tfm)

        train_loader = DataLoader(train_ds, 
            num_workers=self.hparams.num_workers, 
            batch_size=self.hparams.batch_size, 
            pin_memory=False, 
            shuffle=True
        )
        valid_loader = DataLoader(valid_ds, 
            num_workers=self.hparams.num_workers, 
            batch_size=self.hparams.batch_size, 
            pin_memory=False, 
            shuffle=False
        )
        return {
            'train': train_loader, 
            'valid': valid_loader, 
        }

    # @pl.data_loader
    def train_dataloader(self):
        return self.__dataloader()['train']

    # @pl.data_loader
    def val_dataloader(self):
        return self.__dataloader()['valid']

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])
        return parser

def main(hparams):
    model = Model(hparams)
    
    # # ------------------------
    # # 2 SET LOGGER
    # # ------------------------
    # logger = False
    # if hparams.log_wandb:
    #     logger = WandbLogger()

    #     # optional: log model topology
    #     logger.watch(model)

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
        mode='auto',
        patience=50,
        verbose=True,
    )
    trainer = Trainer(
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=stop_callback,
        gpus=hparams.gpus,
        # logger=logger,
        max_epochs=hparams.epochs,
        accumulate_grad_batches=hparams.grad_batches,
        distributed_backend=hparams.distributed_backend,
        precision=16 if hparams.use_amp else 32, 
        profiler=True
    )

    trainer.fit(model)


if __name__ == '__main__':
    # print('Start')
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--dimx', type=int, default=256)
    parser.add_argument('--dimy', type=int, default=256)
    parser.add_argument('--dimz', type=int, default=64)
    parser.add_argument('--lgdir', type=str, default='lightning_logs')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--gpus", type=int, default=-1, help="number of available GPUs")
    parser.add_argument('--distributed-backend', type=str, default='dp', choices=('dp', 'ddp', 'ddp2'),
                        help='supports three options dp, ddp, ddp2')
    parser.add_argument('--use_amp', action='store_true', help='if true uses 16 bit precision')
    parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
    parser.add_argument("--num_workers", type=int, default=4, help="size of the workers")
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument("--nb_layer", type=int, default=5, help="number of layers on u-net")
    parser.add_argument("--features", type=int, default=24, help="number of features in single layer")
    parser.add_argument("--bilinear", action='store_true', default=False,
                        help="whether to use bilinear interpolation or transposed")
    parser.add_argument("--grad_batches", type=int, default=1, help="number of batches to accumulate")
    parser.add_argument("--epochs", type=int, default=300, help="number of epochs to train")
    parser.add_argument("--log_wandb", action='store_true', help="log training on Weights & Biases")
    
    parser = Model.add_model_specific_args(parser)
    hparams = parser.parse_args()

    if hparams.seed:
        random.seed(hparams.seed)
        os.environ['PYTHONHASHSEED'] = str(hparams.seed)
        np.random.seed(hparams.seed)
        torch.manual_seed(hparams.seed)
        torch.cuda.manual_seed(hparams.seed)
        torch.cuda.manual_seed_all(hparams.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    main(hparams)