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
warnings.filterwarnings('ignore', '.*output shape of zoom.*')
# from model import UNet3D

train_image_dir = ['/u01/data/iXrayCT_COVID/data_resized/train/ct3lung3/NSCLC/neg/images/']
train_label_dir = ['/u01/data/iXrayCT_COVID/data_resized/train/ct3lung3/NSCLC/neg/labels/']
valid_image_dir = ['/u01/data/iXrayCT_COVID/data_resized/test/ct3lung3/MEDSEG/pos/images/', 
                   '/u01/data/iXrayCT_COVID/data_resized/test/ct3lung3/NSCLC/neg/images/']
valid_label_dir = ['/u01/data/iXrayCT_COVID/data_resized/test/ct3lung3/MEDSEG/pos/labels/', 
                   '/u01/data/iXrayCT_COVID/data_resized/test/ct3lung3/NSCLC/neg/labels/', ]

# based on:
# https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py

def dice_loss(input, target, eps=1e-8):
    r"""Function that computes Sørensen-Dice Coefficient loss.
    See :class:`~kornia.losses.DiceLoss` for details.
    """
    if not torch.is_tensor(input):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))

    if not len(input.shape) >= 4:
        raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                         .format(input.shape))

    if not input.shape[-3:] == target.shape[-3:]:
        raise ValueError("input and target shapes must be the same. Got: {} and {}"
                         .format(input.shape, input.shape))

    if not input.device == target.device:
        raise ValueError(
            "input and target must be in the same device. Got: {} and {}" .format(
                input.device, target.device))

    # compute softmax over the classes axis
    if input.shape[1] > 1:
        input_soft: torch.Tensor = F.softmax(input, dim=1)

        # create the labels one hot tensor
        target_one_hot: torch.Tensor = one_hot(
            target, nb_class=input.shape[1],
            device=input.device, dtype=input.dtype)
    else:
        input_soft = input
        target_one_hot = target

    # compute the actual dice score
    dims = (1, 2, 3)
    intersection = torch.sum(input_soft * target_one_hot, dims)
    cardinality = torch.sum(input_soft + target_one_hot, dims)

    dice_score = 2. * intersection / (cardinality + eps)
    return torch.mean(-dice_score + 1.)



class DiceLoss(nn.Module):
    r"""Criterion that computes Sørensen-Dice Coefficient loss.
    According to [1], we compute the Sørensen-Dice Coefficient as follows:
    .. math::
        \text{Dice}(x, class) = \frac{2 |X| \cap |Y|}{|X| + |Y|}
    where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the one-hot tensor with the class labels.
    the loss, is finally computed as:
    .. math::
        \text{loss}(x, class) = 1 - \text{Dice}(x, class)
    [1] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%64%93Dice_coefficient
    Shape:
        - Input: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.
    Examples:
        >>> N = 5  # nb_class
        >>> loss = kornia.losses.DiceLoss()
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.eps = 1e-6

    def forward(self, input, target):
        return dice_loss(input, target, self.eps)

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
        if image.shape != label.shape:
            print(self.imagefiles[idx])
            print(self.labelfiles[idx])
        assert image.shape == label.shape

        if self.transforms is not None:
            # for k in range(image.shape[0]):
            transformed = self.transforms(image=np.transpose(image, (1, 2, 0)), 
                                          mask=np.transpose(label, (1, 2, 0)))
            image = np.transpose(transformed['image'], (2, 0, 1))
            label = np.transpose(transformed['mask'], (2, 0, 1))
        scale = [64.0/image.shape[0], 256.0/image.shape[1], 256.0/image.shape[2]]
        # print(image.min(), image.max(), label.min(), label.max())

        # image = scipy.ndimage.zoom(image, scale, order=3)
        # label = scipy.ndimage.zoom(label, scale, order=3)
        # print(image.min(), image.max(), label.min(), label.max())

        p2, p98 = np.percentile(image, (2, 98))
        image = skimage.exposure.rescale_intensity(image, in_range=(p2, p98))
        label = 255*(label>128)
        # print(image.min(), image.max(), label.min(), label.max())

        
        # print(str(idx).zfill(3), image.shape, label.shape)
        return torch.Tensor(image).float().unsqueeze_(0), \
               torch.Tensor(label).float().unsqueeze_(0)


class VNet(nn.Module):
    def __init__(
            self, 
            nb_class = 1,
            nb_layer = 5,
            features = 64,
            bilinear = False
    ):
        super().__init__()
        self.nb_class = nb_class
        self.nb_layer = nb_layer

        layers = [DoubleConv(1, features)]

        feats = features
        for _ in range(nb_layer - 1):
            layers.append(Down(feats, feats * 2))
            feats *= 2

        for _ in range(nb_layer - 1):
            layers.append(Up(feats, feats // 2, bilinear))
            feats //= 2

        layers.append(nn.Conv3d(feats, nb_class, kernel_size=1))
        self.layers = nn.ModuleList(layers)
        self.output = nn.Tanh() #nn.LeakyReLU(inplace=True)

    def forward(self, x):
        xi = [self.layers[0](x * 2.0 - 1.0)]
        # Down path
        for layer in self.layers[1:self.nb_layer]:
            xi.append(layer(xi[-1]))
        # Up path
        for i, layer in enumerate(self.layers[self.nb_layer:-1]):
            xi[-1] = layer(xi[-1], xi[-2 - i])
        return self.output(self.layers[-1](xi[-1])) / 2.0 + 0.5
        # return self.layers[-1](xi[-1])


class DoubleConv(nn.Module):
    def __init__(
            self, 
            in_ch, 
            out_ch
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(
            self, 
            in_ch, 
            out_ch
    ):
        super().__init__()
        self.net = nn.Sequential(
            # nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(in_ch, in_ch, kernel_size=4, stride=2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(
            self, 
            in_ch, 
            out_ch, 
            bilinear = False
    ):
        super().__init__()
        self.upsample = None
        if bilinear:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, 
                            mode="bilinear", 
                            align_corners=True),
                nn.Conv3d(in_ch, 
                          out_ch, 
                          kernel_size=1),
            )
        else:
            self.upsample = nn.ConvTranspose3d(in_ch, 
                                               out_ch, 
                                               kernel_size=4, 
                                               stride=2)

        self.conv = DoubleConv(out_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)

        # Pad x1 to the size of x2
        diff_d = x2.shape[2] - x1.shape[2]
        diff_h = x2.shape[3] - x1.shape[3]
        diff_w = x2.shape[4] - x1.shape[4]

        x1 = F.pad(x1, [diff_d // 2, diff_d - diff_d // 2, 
                        diff_h // 2, diff_h - diff_h // 2, 
                        diff_w // 2, diff_w - diff_w // 2])

        # Concatenate along the channels axis
        # x = torch.cat([x2, x1], dim=1)
        x = x2 + x1
        return self.conv(x)
    

class Model(pl.LightningModule):
    def __init__(self, hparams, n_classes=1):
        super(Model, self).__init__()
        self.example_input_array = torch.rand(2, 1, 64, 256, 256)
        self.hparams = hparams
        self.n_classes = n_classes
        self.vnet = VNet(
            features=self.hparams.features, 
            bilinear=self.hparams.bilinear,
        )
        print(self.vnet)
       
    def forward(self, x):

        return self.vnet(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        x = x / 255.0
        y = y / 255.0
        y_hat = self.forward(x)
        loss = DiceLoss()(y_hat, y) + nn.L1Loss()(y_hat, y)
        vis_images = torch.cat([x, y, y_hat], dim=-1)#[:8]
        mid = int(y.shape[2]/2)
        vis_images = torch.cat([vis_images[:,:,mid,:,:], vis_images[:,:,mid+1,:,:]], dim=-2)
        grid = torchvision.utils.make_grid(vis_images, nrow=2, padding=0)
        self.logger.experiment.add_image('train_vis', grid, self.current_epoch)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        x, y = batch

        x = x / 255.0
        y = y / 255.0
        y_hat = self.forward(x)
        loss = DiceLoss()(y_hat, y) + nn.L1Loss()(y_hat, y)
        vis_images = torch.cat([x, y, y_hat], dim=-1)#[:8]
        mid = int(y.shape[2]/2)
        vis_images = torch.cat([vis_images[:,:,mid,:,:], vis_images[:,:,mid+1,:,:]], dim=-2)
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
        # valid_tfm = None
        train_tfm = AB.Compose([
            # AB.ToFloat(), 
            # AB.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, p=1.0),
            AB.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.15, rotate_limit=30, border_mode=cv2.BORDER_CONSTANT, p=0.8),
            # AB.Resize(height=512, width=512, p=1.0), 
            # AB.CropNonEmptyMaskIfExists(height=320, width=320, p=0.8), 
            # AB.RandomScale(scale_limit=(0.8, 1.2), p=0.8),
            # AB.Equalize(p=0.8),
            # AB.CLAHE(p=0.8),
            AB.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
            AB.RandomGamma(gamma_limit=(80, 120), p=0.8),
            AB.GaussianBlur(p=0.1),
            AB.GaussNoise(p=0.1),
            AB.Resize(width=self.hparams.dimy, height=self.hparams.dimx, p=1.0),
            # AB.ToTensor(),
        ])
        valid_tfm = AB.Compose([
            # AB.ToFloat(), 
            # AB.Equalize(p=1.0),
            # AB.CLAHE(p=1.0),
            # AB.Resize(width=self.hparams.dimy, height=self.hparams.dimx),
        ])

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
    parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
    parser.add_argument("--num_workers", type=int, default=4, help="size of the workers")
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument("--nb_layer", type=int, default=5, help="number of layers on u-net")
    parser.add_argument("--features", type=int, default=24, help="number of features in first layer")
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