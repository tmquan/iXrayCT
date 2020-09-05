import os
import glob
from argparse import ArgumentParser

import logging
from collections import OrderedDict

from natsort import natsorted
from tqdm import tqdm
import datetime
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
    # if input.shape[1] > 1:
    #     input_soft: torch.Tensor = F.softmax(input, dim=1)

    #     # create the labels one hot tensor
    #     target_one_hot: torch.Tensor = one_hot(
    #         target, nb_class=input.shape[1],
    #         device=input.device, dtype=input.dtype)
    # else:
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

def worker_init_fn(worker_id):                                                          
    # np.random.seed(np.random.get_state()[1][0] + worker_id)
    torch.initial_seed()

class CustomNativeDataset(Dataset):
    def __init__(self, 
        imagepaireddir, 
        labelpaireddir, 
        imageunpaireddir, 
        labelunpaireddir, 
        train_or_valid='train',
        size=1000, 
        transforms=None
    ):
        # print('\n')
        self.size = size
        self.is_train = True if train_or_valid=='train' else False
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
        
        print(len(self.imagepairedfiles), len(self.labelpairedfiles), \
              len(self.imageunpairedfiles), len(self.labelunpairedfiles))

    def __len__(self):
        return self.size if self.is_train else len(self.imagepairedfiles)
    
    def __call__(self):
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().millisecond)
        
    def __getitem__(self, idx):
        pidx = torch.randint(len(self.imagepairedfiles), (1, 1)) #np.random.randint(len(self.imagepairedfiles))
        imagepaired = skimage.io.imread(self.imagepairedfiles[pidx])
        labelpaired = skimage.io.imread(self.labelpairedfiles[pidx])

        aidx = torch.randint(len(self.imageunpairedfiles), (1, 1)) #np.random.randint(len(self.imageunpairedfiles))
        bidx = torch.randint(len(self.labelunpairedfiles), (1, 1)) #np.random.randint(len(self.labelunpairedfiles))
        imageunpaired = skimage.io.imread(self.imageunpairedfiles[aidx])
        labelunpaired = skimage.io.imread(self.labelunpairedfiles[bidx])

        if self.transforms is not None:
            transformed = self.transforms(image=np.transpose(np.expand_dims(imagepaired, 0), (1, 2, 0)))
            imagepaired = np.squeeze(np.transpose(transformed['image'], (2, 0, 1)), 0)
            transformed = self.transforms(image=np.transpose(labelpaired, (1, 2, 0)))
            labelpaired = np.transpose(transformed['image'], (2, 0, 1))

            untransformed = self.transforms(image=np.transpose(np.expand_dims(imageunpaired, 0), (1, 2, 0)))
            imageunpaired = np.squeeze(np.transpose(untransformed['image'], (2, 0, 1)), 0)
            untransformed = self.transforms(image=np.transpose(labelunpaired, (1, 2, 0)))
            labelunpaired = np.transpose(untransformed['image'], (2, 0, 1))
            
        scale = [64.0/labelpaired.shape[0], 256.0/labelpaired.shape[1], 256.0/labelpaired.shape[2]]
        imagepaired = cv2.resize(imagepaired, (256, 256))
        labelpaired = scipy.ndimage.zoom(labelpaired, scale, order=3)
        imageunpaired = cv2.resize(imageunpaired, (256, 256))
        labelunpaired = scipy.ndimage.zoom(labelunpaired, scale, order=3)

        return torch.Tensor(imagepaired).float().unsqueeze_(0),   \
               torch.Tensor(labelpaired).float(),                 \
               torch.Tensor(imageunpaired).float().unsqueeze_(0), \
               torch.Tensor(labelunpaired).float(),               

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

class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)

def pixel_shuffle(input, upscale_factor):
    r"""Rearranges elements in a tensor of shape
    ``[*, C, d_{1}, d_{2}, ..., d_{n}]`` to a tensor of shape
    ``[*, C/(r^n), d_{1}*r, d_{2}*r, ..., d_{n}*r]``. Where ``n`` is the
    dimensionality of the data.
    
    See :class:`~torch.nn.PixelShuffle` for details.
    Args:
        input (Variable): Input
        upscale_factor (int): factor to increase spatial resolution by
    Examples::
        # 1D example
        >>> ps = nn.PixelShuffle(2)
        >>> input = autograd.Variable(torch.Tensor(1, 4, 8))
        >>> output = ps(input)
        >>> print(output.size())
        torch.Size([1, 2, 16])
        # 2D example
        >>> ps = nn.PixelShuffle(3)
        >>> input = autograd.Variable(torch.Tensor(1, 9, 8, 8))
        >>> output = ps(input)
        >>> print(output.size())
        torch.Size([1, 1, 24, 24])
        # 3D example
        >>> ps = nn.PixelShuffle(2)
        >>> input = autograd.Variable(torch.Tensor(1, 8, 16, 16, 16))
        >>> output = ps(input)
        >>> print(output.size())
        torch.Size([1, 1, 32, 32, 32])
    """
    input_size = list(input.size())
    dimensionality = len(input_size) - 2

    input_size[1] //= (upscale_factor ** dimensionality)
    output_size = [dim * upscale_factor for dim in input_size[2:]]

    input_view = input.contiguous().view(
        input_size[0], input_size[1],
        *(([upscale_factor] * dimensionality) + input_size[2:])
    )

    indicies = list(range(2, 2 + 2 * dimensionality))
    indicies = indicies[1::2] + indicies[0::2]

    shuffle_out = input_view.permute(0, 1, *(indicies[::-1])).contiguous()
    shuffle_out = shuffle_out.view(input_size[0], input_size[1], *output_size)
    # print(shuffle_out.shape)
    return shuffle_out

class PixelShuffle(nn.Module):
    def __init__(self, scale=2):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return pixel_shuffle(x, self.scale)

class DoubleConv2d(nn.Module):
    def __init__(self, 
        source_channels=32, 
        output_channels=32,
    ):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Dropout(),
            nn.Conv2d(source_channels, output_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
        )
        self.net = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
        )

    def forward(self, x):
        tmp = self.pre(x)
        ret = self.net(tmp) + tmp
        return ret

class DoubleDeconv3d(nn.Module):
    def __init__(self, 
        source_channels=32, 
        output_channels=32,
    ):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Dropout(),
            # nn.ConvTranspose3d(source_channels, output_channels, kernel_size=2, stride=2, padding=0, bias=False),
            # nn.Conv3d(source_channels, output_channels*8, kernel_size=3, stride=1, padding=1, bias=False),
            # PixelShuffle(2),
            nn.Conv3d(source_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.BatchNorm3d(output_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
        )
        self.net = nn.Sequential(
            # nn.ConvTranspose3d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv3d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(output_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
        )
        
    def forward(self, x):
        tmp = self.pre(x)
        ret = self.net(tmp)  
        return ret

class DoubleConv3d(nn.Module):
    def __init__(self, 
        source_channels=32, 
        output_channels=32,
    ):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Dropout(),
            nn.Conv3d(source_channels, output_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(output_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
        )
        self.net = nn.Sequential(
            nn.Conv3d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(output_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
        )
        
    def forward(self, x):
        tmp = self.pre(x)
        ret = self.net(tmp) + tmp
        return ret

class DoubleDeconv2d(nn.Module):
    def __init__(self, 
        source_channels=32, 
        output_channels=32,
    ):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Dropout(),
            # nn.ConvTranspose2d(source_channels, output_channels, kernel_size=2, stride=2, padding=0, bias=False),
            # nn.Conv2d(source_channels, output_channels*4, kernel_size=3, stride=1, padding=1, bias=False),
            # PixelShuffle(2),
            nn.Conv2d(source_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
        )
        self.net = nn.Sequential(
            # nn.ConvTranspose2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
        )
        
    def forward(self, x):
        tmp = self.pre(x)
        ret = self.net(tmp)  
        return ret

class INet(nn.Module):
    def __init__(self, source_channels=1, output_channels=1, num_filters=32):
        super().__init__()
        self.net = nn.Sequential(
            # 2D
            DoubleConv2d(source_channels, num_filters*4), # 128
            DoubleConv2d(num_filters*4, num_filters*8), # 64
            DoubleConv2d(num_filters*8, num_filters*16), # 32
            DoubleConv2d(num_filters*16, num_filters*32), # 16
            DoubleConv2d(num_filters*32, num_filters*64), # 8
            

            # Transformation
            nn.Conv2d(num_filters*64, num_filters*64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_filters*64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
            Reshape(num_filters*32, 2, 8, 8),
            nn.Conv3d(num_filters*32, num_filters*32, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.ConvTranspose3d(num_filters*32, num_filters*32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(num_filters*32),
            nn.LeakyReLU(inplace=True),

            # 3D
            DoubleDeconv3d(num_filters*32, num_filters*16), #2, 1024, 4, 16, 16]
            DoubleDeconv3d(num_filters*16, num_filters*8),
            DoubleDeconv3d(num_filters*8, num_filters*4),
            DoubleDeconv3d(num_filters*4, num_filters*2),
            DoubleDeconv3d(num_filters*2, num_filters*1),
            # nn.ConvTranspose3d(num_filters*1, output_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv3d(num_filters*1, output_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            Squeeze(dim=1),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.net(x * 2.0 - 1.0) / 2.0 + 0.5

class PNet(nn.Module):
    def __init__(self, source_channels=1, output_channels=1, num_filters=32):
        super().__init__()
        # in: 1 x 64 x 128 x 128
        self.net = nn.Sequential(*[
            Unsqueeze(1),
            DoubleConv3d(1, num_filters*2), #128
            DoubleConv3d(num_filters*2, num_filters*4), #64
            DoubleConv3d(num_filters*4, num_filters*8), #32
            DoubleConv3d(num_filters*8, num_filters*16), #16 
            DoubleConv3d(num_filters*16, num_filters*32), #8
            
            # transformation
            nn.Conv3d(num_filters*32, num_filters*32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(num_filters*32),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),
            Reshape(num_filters*64, 8, 8),
            nn.Conv2d(num_filters*64, num_filters*64, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.ConvTranspose2d(num_filters*64, num_filters*64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_filters*64),
            nn.LeakyReLU(inplace=True),
            
            # 2D
            DoubleDeconv2d(num_filters*64, num_filters*32), #16
            DoubleDeconv2d(num_filters*32, num_filters*16), #32
            DoubleDeconv2d(num_filters*16, num_filters*8), #64
            DoubleDeconv2d(num_filters*8, num_filters*4), #128
            DoubleDeconv2d(num_filters*4, num_filters*2), #256
            # nn.ConvTranspose2d(num_filters*2, num_filters*1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(num_filters*2, num_filters*1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_filters*1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(num_filters*1, output_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh(),
        ])
    def forward(self, x):
        return self.net(x * 2.0 - 1.0) / 2.0 + 0.5

class Model(pl.LightningModule):
    def __init__(self, hparams, n_classes=1):
        super(Model, self).__init__()
        self.example_input_array = torch.rand(2, 1, 256, 256), torch.rand(2, 64, 256, 256), \
                                   torch.rand(2, 1, 256, 256), torch.rand(2, 64, 256, 256)
        self.hparams = hparams
        self.learning_rate = self.hparams.lr
        self.inet = INet(
            source_channels=1, 
            output_channels=1, num_filters=self.hparams.features
        )
        self.pnet = PNet(
            source_channels=1, 
            output_channels=1, num_filters=self.hparams.features
        )

    def forward(self, x, y, a, b):
        # return self.rnet(x)
        xy = self.inet(x)         # ct from xr
        yx = self.pnet(y)         # xr from ct
        ab = self.inet(a)         # ct from xr 
        aba = self.pnet(ab)     # xr from ct
        ba = self.pnet(b)         # xr from ct
        bab = self.inet(ba)     # ct from xr

        return xy, yx, ab, aba, ba, bab


    def training_step(self, batch, batch_nb):
        x, y, a, b = batch
        x = x / 255.0
        y = y / 255.0
        a = a / 255.0
        b = b / 255.0
        xy, yx, ab, aba, ba, bab = self.forward(x, y, a, b)
        loss = nn.L1Loss(reduction='mean')(xy, y) \
             + nn.L1Loss(reduction='mean')(yx, x) \
             + nn.L1Loss(reduction='mean')(aba, a) \
             + nn.L1Loss(reduction='mean')(bab, b) 
        # loss = DiceLoss()(xy, y) +  nn.L1Loss(reduction='mean')(xy, y) \
        #      + DiceLoss()(yx, x) +  nn.L1Loss(reduction='mean')(yx, x) \
        #      + DiceLoss()(aba, a) + nn.L1Loss(reduction='mean')(aba, a) \
        #      + DiceLoss()(bab, b) + nn.L1Loss(reduction='mean')(bab, b) 
        # DiceLoss()(xy, y) + 
        # DiceLoss()(yx, x) + 
        # DiceLoss()(aba, a) +
        # DiceLoss()(bab, b) +
        mid = int(y.shape[1]/2)
        vis_images = torch.cat([torch.cat([x, y[:,mid:mid+1,:,:], xy[:,mid:mid+1,:,:], yx], dim=-1), 
                                torch.cat([a, b[:,mid:mid+1,:,:], ab[:,mid:mid+1,:,:], aba], dim=-1), 
                                torch.cat([b[:,mid:mid+1,:,:], a, ba, bab[:,mid:mid+1,:,:]], dim=-1), 
                                ], dim=-2) 
        grid = torchvision.utils.make_grid(vis_images, nrow=2, padding=0)
        self.logger.experiment.add_image('train_vis', grid, self.current_epoch)
        tensorboard_logs = {'train_loss': loss, 'lr': self.learning_rate}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        x, y, a, b = batch
        x = x / 255.0
        y = y / 255.0
        a = a / 255.0
        b = b / 255.0
        xy, yx, ab, aba, ba, bab = self.forward(x, y, a, b)
        loss = nn.L1Loss(reduction='mean')(xy, y) \
             + nn.L1Loss(reduction='mean')(yx, x) \
             + nn.L1Loss(reduction='mean')(aba, a) \
             + nn.L1Loss(reduction='mean')(bab, b) 
        # loss = DiceLoss()(xy, y) +  nn.L1Loss(reduction='mean')(xy, y) \
        #      + DiceLoss()(yx, x) +  nn.L1Loss(reduction='mean')(yx, x) \
        #      + DiceLoss()(aba, a) + nn.L1Loss(reduction='mean')(aba, a) \
        #      + DiceLoss()(bab, b) + nn.L1Loss(reduction='mean')(bab, b) 
        # DiceLoss()(xy, y) + 
        # DiceLoss()(yx, x) + 
        # DiceLoss()(aba, a) +
        # DiceLoss()(bab, b) +
        mid = int(y.shape[1]/2)
        vis_images = torch.cat([torch.cat([x, y[:,mid:mid+1,:,:], xy[:,mid:mid+1,:,:], yx], dim=-1), 
                                torch.cat([a, b[:,mid:mid+1,:,:], ab[:,mid:mid+1,:,:], aba], dim=-1), 
                                torch.cat([b[:,mid:mid+1,:,:], a, ba, bab[:,mid:mid+1,:,:]], dim=-1), 
                                ], dim=-2) 
        grid = torchvision.utils.make_grid(vis_images, nrow=2, padding=0)
        self.logger.experiment.add_image('valid_vis', grid, self.current_epoch)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def __dataloader(self):
        # train_tfm = None
        # valid_tfm = None
        train_tfm = AB.Compose([
            # AB.ToFloat(), 
            # AB.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, p=1.0),
            # AB.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0., rotate_limit=0, border_mode=cv2.BORDER_CONSTANT, p=0.9),
            # AB.Resize(height=512, width=512, p=1.0), 
            # AB.CropNonEmptyMaskIfExists(height=320, width=320, p=0.8), 
            # AB.RandomScale(scale_limit=(0.8, 1.2), p=0.8),
            # AB.Equalize(p=0.8),
            # AB.CLAHE(p=0.8),
            AB.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.50),
            AB.RandomGamma(gamma_limit=(80, 120), p=0.50),
            AB.GaussianBlur(p=0.05),
            AB.GaussNoise(p=0.05),
            AB.Resize(width=self.hparams.dimy, height=self.hparams.dimx, p=1.0),
            # AB.ToTensor(),
        ])
        valid_tfm = AB.Compose([
            # AB.ToFloat(), 
            # AB.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, p=1.0),
            # AB.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0., rotate_limit=0, border_mode=cv2.BORDER_CONSTANT, p=0.9),
            # AB.Resize(height=512, width=512, p=1.0), 
            # AB.CropNonEmptyMaskIfExists(height=320, width=320, p=0.8), 
            # AB.RandomScale(scale_limit=(0.8, 1.2), p=0.8),
            # AB.Equalize(p=0.8),
            # AB.CLAHE(p=0.8),
            AB.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.50),
            AB.RandomGamma(gamma_limit=(80, 120), p=0.50),
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
        # print(len(train_loader), len(valid_loader))
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
        mode='min',
        patience=500,
        verbose=True,
    )
    trainer = Trainer(
        auto_lr_find=False,
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
    parser.add_argument('--distributed-backend', type=str, default='ddp', choices=('dp', 'ddp', 'ddp2'),
                        help='supports three options dp, ddp, ddp2')
    parser.add_argument('--use_amp', default=True, action='store_true', help='if true uses 16 bit precision')
    parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
    parser.add_argument("--num_workers", type=int, default=8, help="size of the workers")
    parser.add_argument("--lr", type=float, default=0.0005, help="learning rate")
    parser.add_argument("--nb_layer", type=int, default=5, help="number of layers on u-net")
    parser.add_argument("--features", type=int, default=32, help="number of features in single layer")
    parser.add_argument("--bilinear", action='store_true', default=False,
                        help="whether to use bilinear interpolation or transposed")
    parser.add_argument("--grad_batches", type=int, default=1, help="number of batches to accumulate")
    parser.add_argument("--epochs", type=int, default=500, help="number of epochs to train")
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
        torch.backends.cudnn.benchmark = True

    main(hparams)