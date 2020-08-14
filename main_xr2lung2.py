import os
import glob
from argparse import ArgumentParser

import logging
from collections import OrderedDict

from natsort import natsorted
from tqdm import tqdm

import numpy as np
import cv2
import albumentations as AB

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import kornia

train_image_dir = '/vinbrain/data/iXrayCT_COVID/data/train/xr2lung2/NLMMC/images/'
train_label_dir = '/vinbrain/data/iXrayCT_COVID/data/train/xr2lung2/NLMMC/labels/'
valid_image_dir = '/vinbrain/data/iXrayCT_COVID/data/test/xr2lung2/JSRT/images/'
valid_label_dir = '/vinbrain/data/iXrayCT_COVID/data/test/xr2lung2/JSRT/labels/'


class CustomNativeDataset(Dataset):
    def __init__(self, 
        imagedir, 
        labeldir, 
        train_or_valid='train',
        size=500, 
        transforms=None
    ):
        self.size = size
        self.is_train = True if train_or_valid=='train' else False
        self.imagedir = imagedir if self.is_train else imagedir.replace('train', 'test')
        self.labeldir = labeldir if self.is_train else labeldir.replace('train', 'test')
        
        self.imagefiles = natsorted(glob.glob(os.path.join(self.imagedir, '*.*')))
        self.labelfiles = natsorted(glob.glob(os.path.join(self.labeldir, '*.*')))
        self.transforms = transforms
        assert len(self.imagefiles) == len(self.labelfiles)
        
    def __len__(self):
        if self.size > len(self.imagefiles) or self.size is None: 
            return len(self.imagefiles)
        else:
            return self.size
 
    def __getitem__(self, idx):
        image = cv2.imread(self.imagefiles[idx], cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(self.labelfiles[idx], cv2.IMREAD_GRAYSCALE)

        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=label)
            image = transformed['image']
            label = transformed['mask']

        return kornia.image_to_tensor(image).float(), \
               kornia.image_to_tensor(label).float()


class UNet(nn.Module):
    """
    Architecture based on U-Net: Convolutional Networks for Biomedical Image Segmentation
    Link - https://arxiv.org/abs/1505.04597
    Parameters:
        num_classes: Number of output classes required (default 19 for KITTI dataset)
        num_layers: Number of layers in each side of U-net
        features_start: Number of features in first layer
        bilinear: Whether to use bilinear interpolation or transposed
            convolutions for upsampling.
    """
    def __init__(
            self, 
            num_classes: int = 1,
            num_layers: int = 5,
            features_start: int = 64,
            bilinear: bool = False
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers

        layers = [DoubleConv(1, features_start)]

        feats = features_start
        for _ in range(num_layers - 1):
            layers.append(Down(feats, feats * 2))
            feats *= 2

        for _ in range(num_layers - 1):
            layers.append(Up(feats, feats // 2, bilinear))
            feats //= 2

        layers.append(nn.Conv2d(feats, num_classes, kernel_size=1))
        self.layers = nn.ModuleList(layers)
        # self.output = nn.Sigmoid() #nn.ReLU(inplace=True)

    def forward(self, x):
        xi = [self.layers[0](x)]
        # Down path
        for layer in self.layers[1:self.num_layers]:
            xi.append(layer(xi[-1]))
        # Up path
        for i, layer in enumerate(self.layers[self.num_layers:-1]):
            xi[-1] = layer(xi[-1], xi[-2 - i])
        # return self.output(self.layers[-1](xi[-1]))
        return self.layers[-1](xi[-1])


class DoubleConv(nn.Module):
    """
    Double Convolution and BN and ReLU
    (3x3 conv -> BN -> ReLU) ** 2
    """
    def __init__(
            self, 
            in_ch: int, 
            out_ch: int
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """
    Combination of MaxPool2d and DoubleConv in series
    """

    def __init__(
            self, 
            in_ch: int, 
            out_ch: int
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """
    Upsampling (by either bilinear interpolation or transpose convolutions)
    followed by concatenation of feature map from contracting path,
    followed by double 3x3 convolution.
    """

    def __init__(
            self, 
            in_ch: int, 
            out_ch: int, 
            bilinear: bool = False
    ):
        super().__init__()
        self.upsample = None
        if bilinear:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, 
                            mode="bilinear", 
                            align_corners=True),
                nn.Conv2d(in_ch, 
                          in_ch // 2, 
                          kernel_size=1),
            )
        else:
            self.upsample = nn.ConvTranspose2d(in_ch, 
                                               in_ch // 2, 
                                               kernel_size=2, 
                                               stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)

        # Pad x1 to the size of x2
        diff_h = x2.shape[2] - x1.shape[2]
        diff_w = x2.shape[3] - x1.shape[3]

        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2, 
                        diff_h // 2, diff_h - diff_h // 2])

        # Concatenate along the channels axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    

class Model(pl.LightningModule):
    def __init__(self, hparams, n_classes=1):
        super(Model, self).__init__()
        self.hparams = hparams
        self.n_classes = n_classes
        self.unet = UNet()
        print(self.unet)
        
    def forward(self, x):
        return self.unet(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        x = x / 255.0
        y = y / 255.0
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y) if self.n_classes > 1 else \
            F.binary_cross_entropy_with_logits(y_hat, y)
        # loss = kornia.dice_loss(y_hat, y.long())
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        x = x / 255.0
        y = y / 255.0
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y) if self.n_classes > 1 else \
            F.binary_cross_entropy_with_logits(y_hat, y)
        # loss = kornia.dice_loss(y_hat, y.long())
        return {'val_loss': loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=1e-8)

    def __dataloader(self):
        train_tfm = AB.Compose([
            AB.Resize(width=self.hparams.shape, height=self.hparams.shape),
        ])
        valid_tfm = AB.Compose([
            AB.Resize(width=self.hparams.shape, height=self.hparams.shape),
        ])
        train_ds = CustomNativeDataset(imagedir=train_image_dir, 
                                       labeldir=train_label_dir, 
                                       train_or_valid='train', 
                                       transforms=train_tfm)
        valid_ds = CustomNativeDataset(imagedir=valid_image_dir, 
                                       labeldir=valid_label_dir, 
                                       train_or_valid='valid', 
                                       transforms=valid_tfm)

        train_loader = DataLoader(train_ds, batch_size=16, pin_memory=True, shuffle=True)
        valid_loader = DataLoader(valid_ds, batch_size=16, pin_memory=True, shuffle=False)
        return {
            'train': train_loader, 
            'valid': valid_loader, 
        }

    @pl.data_loader
    def train_dataloader(self):
        return self.__dataloader()['train']

    @pl.data_loader
    def val_dataloader(self):
        return self.__dataloader()['valid']

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])
        return parser

def main(hparams):
    model = Model(hparams)

    os.makedirs(hparams.logdir, exist_ok=True)
    try:
        logdir = sorted(os.listdir(hparams.logdir))[-1]
    except IndexError:
        logdir = os.path.join(hparams.logdir, 'version_0')
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(logdir, 'checkpoints'),
        save_top_k=5,
        verbose=True,
    )
    stop_callback = EarlyStopping(
        monitor='val_loss',
        mode='auto',
        patience=5,
        verbose=True,
    )
    trainer = Trainer(
        gpus=1,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=stop_callback,
    )

    trainer.fit(model)


if __name__ == '__main__':
    # print('Start')
    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument('--shape', default=256)
    parent_parser.add_argument('--logdir', default='lightning_logs')
    
    
    parser = Model.add_model_specific_args(parent_parser)
    hparams = parser.parse_args()

    main(hparams)