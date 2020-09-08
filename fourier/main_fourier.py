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
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateLogger
from pytorch_lightning.loggers import WandbLogger
# from pytorch_lightning.callbacks import LearningRateLogger

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



def worker_init_fn(worker_id):                                                          
    # np.random.seed(np.random.get_state()[1][0] + worker_id)
    torch.initial_seed()

class TotalVariationLoss(nn.Module):
    def __init__(self, is_3d=False):
        super(TotalVariationLoss, self).__init__()
        self.is_3d = is_3d

    def tv(self, img):
        if not torch.is_tensor(img):
            raise TypeError(f"Input type is not a torch.Tensor. Got {type(img)}")
        img_shape = img.shape
        if len(img_shape) == 3 or len(img_shape) == 4:
            pixel_dif1 = img[...,:,1:,:] - img[...,:,:-1,:]
            pixel_dif2 = img[...,:,:,1:] - img[...,:,:,:-1]
            reduce_axes = (-3, -2, -1)
            if self.is_3d:
                pixel_dif3 = img[...,1:,:,:] - img[...,:-1,:,:]
        else:
            raise ValueError("Expected input tensor to be of ndim 3 or 4, but got " + str(len(img_shape)))
        if self.is_3d:
            return pixel_dif1.abs().mean(dim=reduce_axes) \
                 + pixel_dif2.abs().mean(dim=reduce_axes) \
                 + pixel_dif3.abs().mean(dim=reduce_axes)
        return pixel_dif1.abs().mean(dim=reduce_axes) \
             + pixel_dif2.abs().mean(dim=reduce_axes)

    def forward(self, img, dst):
        # return total_variation(img)
        # tv_img = self.tv(img)
        # tv_dst = self.tv(dst)
        # return nn.L1Loss()(tv_img, tv_dst)
        return self.tv(img-dst)


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4])
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9])
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16])
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23])
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize

    def forward(self, input, target):
        # if input.shape[1] != 3:
        #     input = input.repeat(1, 3, 1, 1)
        #     target = target.repeat(1, 3, 1, 1)

        input = input.view(-1, 1, input.shape[2], input.shape[3]) #.float()
        input = input.repeat(1, 3, 1, 1)
        target = target.view(-1, 1, target.shape[2], target.shape[3]) #.float()
        target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        return loss

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
            # transformed = self.transforms(image=np.transpose(np.expand_dims(imagepaired, 0), (1, 2, 0)))
            # imagepaired = np.squeeze(np.transpose(transformed['image'], (2, 0, 1)), 0)
            # transformed = self.transforms(image=np.transpose(labelpaired, (1, 2, 0)))
            # labelpaired = np.transpose(transformed['image'], (2, 0, 1))

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


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        channels = int(np.ceil(channels/2))
        if channels % 2 or channels == 0:
            channels += 1
        self.channels = channels
        inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")
        b, orig_ch, y, x = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb = torch.zeros((x,y,self.channels*2),device=tensor.device).type(tensor.type())
        emb[:,:,:self.channels] = emb_x
        emb[:,:,self.channels:2*self.channels] = emb_y

        return emb[:,:,:orig_ch].unsqueeze(0).repeat(b, 1, 1, 1).permute(0, 3, 2, 1)

class PositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        channels = int(np.ceil(channels/3))
        if channels % 2 or channels == 0:
            channels += 1
        self.channels = channels
        inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")
        b, orig_ch, z, y, x  = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_z = torch.arange(z, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=1).unsqueeze(1)
        emb_z = torch.cat((sin_inp_z.sin(), sin_inp_z.cos()), dim=1).unsqueeze(1).unsqueeze(1)
        emb = torch.zeros((1, z, y, x, self.channels*3),device=tensor.device).type(tensor.type())
        emb[:,:,:,:,:self.channels] = emb_x
        emb[:,:,:,:,self.channels:2*self.channels] = emb_y
        emb[:,:,:,:,2*self.channels:] = emb_z
        return emb[:,:,:,:,:orig_ch].permute(0, 4, 1, 2, 3).repeat(b, 1, 1, 1, 1)

class PositionalEncodingConv2d(nn.Module):
    def __init__(self, 
        source_channels=32, 
        output_channels=32,
        kernel_size=4, stride=2, padding=1, bias=False
    ):
        super().__init__()
        self.encoding = PositionalEncoding2D(source_channels)
        self.convolve = nn.Conv2d(source_channels*2, 
                                  output_channels, 
                                  kernel_size=kernel_size, 
                                  stride=stride, 
                                  padding=padding, 
                                  bias=bias)

    def forward(self, x):
        encode = self.encoding(x)
        concat = torch.cat((x, encode), dim=1)
        result = self.convolve(concat)
        return result

class PositionalEncodingConv3d(nn.Module):
    def __init__(self, 
        source_channels=32, 
        output_channels=32,
        kernel_size=4, stride=2, padding=1, bias=False
    ):
        super().__init__()
        self.encoding = PositionalEncoding3D(source_channels)
        self.convolve = nn.Conv3d(source_channels*2, 
                                  output_channels, 
                                  kernel_size=kernel_size, 
                                  stride=stride, 
                                  padding=padding, 
                                  bias=bias)

    def forward(self, x):
        encode = self.encoding(x)
        concat = torch.cat((x, encode), dim=1)
        result = self.convolve(concat)
        return result

class DoubleConv2d(nn.Module):
    def __init__(self, 
        source_channels=32, 
        output_channels=32,
    ):
        super().__init__()
        self.pre = nn.Sequential(
            PositionalEncodingConv2d(source_channels, output_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, output_channels),
            nn.LeakyReLU(inplace=True),
        )
        self.net = nn.Sequential(
            PositionalEncodingConv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(4, output_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        tmp = self.pre(x)
        ret = self.net(tmp) + tmp
        # print(ret.shape)
        return ret

class DoubleDeconv3d(nn.Module):
    def __init__(self, 
        source_channels=32, 
        output_channels=32,
    ):
        super().__init__()
        self.pre = nn.Sequential(
            PositionalEncodingConv3d(source_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.GroupNorm(4, output_channels),
            nn.LeakyReLU(inplace=True),
        )
        self.net = nn.Sequential(
            PositionalEncodingConv3d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(4, output_channels),
            nn.LeakyReLU(inplace=True),
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
            PositionalEncodingConv3d(source_channels, output_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, output_channels),
            nn.LeakyReLU(inplace=True),
        )
        self.net = nn.Sequential(
            PositionalEncodingConv3d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(4, output_channels),
            nn.LeakyReLU(inplace=True),
        )
        
    def forward(self, x):
        tmp = self.pre(x)
        ret = self.net(tmp) + tmp
        # print(ret.shape)
        return ret

class DoubleDeconv2d(nn.Module):
    def __init__(self, 
        source_channels=32, 
        output_channels=32,
    ):
        super().__init__()
        self.pre = nn.Sequential(
            PositionalEncodingConv2d(source_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.GroupNorm(4, output_channels),
            nn.LeakyReLU(inplace=True),
        )
        self.net = nn.Sequential(
            PositionalEncodingConv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(4, output_channels),
            nn.LeakyReLU(inplace=True),
        )
        
    def forward(self, x):
        tmp = self.pre(x)
        ret = self.net(tmp)  
        return ret

class INet(nn.Module):
    def __init__(self, source_channels=1, output_channels=1, num_filters=32):
        super().__init__()
        self.enc = nn.Sequential(
            # 2D
            DoubleConv2d(source_channels, num_filters*4),   # NF x 4 x 128 x 128
            DoubleConv2d(num_filters*4, num_filters*8),     # NF x 8 x 64 x 64
            DoubleConv2d(num_filters*8, num_filters*16),    # NF x 16 x 32 x 32
            DoubleConv2d(num_filters*16, num_filters*32),   # NF x 32 x 16 x 16
            DoubleConv2d(num_filters*32, num_filters*64),   # NF x 64 x 8 x 8
            DoubleConv2d(num_filters*64, num_filters*80),   # NF x 80 x 4 x 4
            

            # Transformation
            nn.Conv2d(num_filters*80, num_filters*80, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(4, num_filters*80),
            nn.Flatten(),
            nn.LeakyReLU(inplace=True),
            nn.Linear(10240, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 10240),
            nn.LeakyReLU(inplace=True),
        )
        self.dec = nn.Sequential(
            Reshape(num_filters*80, 1, 4, 4), # NF x 80 x 1 x 4 x 4
            nn.Conv3d(num_filters*80, num_filters*80, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(4, num_filters*80),
            nn.LeakyReLU(inplace=True),

            # 3D
            DoubleDeconv3d(num_filters*80, num_filters*64), # NF x 64 x 2 x 8 x 8
            DoubleDeconv3d(num_filters*64, num_filters*32), # NF x 32 x 4 x 16 x 16
            DoubleDeconv3d(num_filters*32, num_filters*16), # NF x 16 x 8 x 32 x 32
            DoubleDeconv3d(num_filters*16, num_filters*8),  # NF x 8 x 16 x 64 x 64
            DoubleDeconv3d(num_filters*8, num_filters*4),   # NF x 4 x 32 x 128 x 128
            DoubleDeconv3d(num_filters*4, num_filters*1),   # NF x 1 x 64 x 256 x 256
            # nn.ConvTranspose3d(num_filters*1, output_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv3d(num_filters*1, output_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),
            Squeeze(dim=1),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )
    
    def forward(self, x):
        # return self.net(x * 2.0 - 1.0) / 2.0 + 0.5
        feature = self.enc(x * 2.0 - 1.0) 
        outputs = self.dec(feature)/ 2.0 + 0.5
        return outputs, feature

class PNet(nn.Module):
    def __init__(self, source_channels=1, output_channels=1, num_filters=32):
        super().__init__()
        # in: 1 x 64 x 128 x 128
        self.enc = nn.Sequential(
            Unsqueeze(1),
            DoubleConv3d(1, num_filters*2),                 # NF x 2 x 32 x 128 x 128
            DoubleConv3d(num_filters*2, num_filters*4),     # NF x 4 x 16 x 64 x 64
            DoubleConv3d(num_filters*4, num_filters*8),     # NF x 8 x 8 x 32 x 32
            DoubleConv3d(num_filters*8, num_filters*16),    # NF x 16 x 4 x 16 x 16
            DoubleConv3d(num_filters*16, num_filters*32),   # NF x 32 x 2 x 8 x 8
            DoubleConv3d(num_filters*32, num_filters*64),   # NF x 64 x 1 x 4 x 4
            
            # transformation
            nn.Conv3d(num_filters*64, num_filters*80, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(4, num_filters*80),
            nn.LeakyReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(10240, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 10240),
            nn.LeakyReLU(inplace=True),
            Reshape(num_filters*80, 4, 4), # NF x 80 x 4 x 4
        )
        self.dec = nn.Sequential(
            nn.Conv2d(num_filters*80, num_filters*80, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(4, num_filters*80),
            nn.LeakyReLU(inplace=True),
            
            # 2D
            DoubleDeconv2d(num_filters*80, num_filters*64), # NF x 64 x 8 x 8
            DoubleDeconv2d(num_filters*64, num_filters*32), # NF x 32 x 16 x 16
            DoubleDeconv2d(num_filters*32, num_filters*16), # NF x 16 x 32 x 32
            DoubleDeconv2d(num_filters*16, num_filters*8),  # NF x 8 x 64 x 64
            DoubleDeconv2d(num_filters*8, num_filters*4),   # NF x 4 x 128 x 128
            DoubleDeconv2d(num_filters*4, num_filters*1),   # NF x 1 x 256 x 256
            # nn.ConvTranspose2d(num_filters*2, num_filters*1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(num_filters*1, num_filters*1, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(num_filters*1),
            nn.GroupNorm(4, num_filters*1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(num_filters*1, output_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        # return self.net(x * 2.0 - 1.0) / 2.0 + 0.5
        feature = self.enc(x * 2.0 - 1.0) 
        outputs = self.dec(feature)/ 2.0 + 0.5
        return outputs, feature

class Model(pl.LightningModule):
    def __init__(self, hparams, n_classes=1):
        super(Model, self).__init__()
        self.example_input_array = torch.rand(2, 1, 256, 256), torch.rand(2, 64, 256, 256), \
                                   torch.rand(2, 1, 256, 256), torch.rand(2, 64, 256, 256)
        self.hparams = hparams
        # self.learning_rate = self.hparams.lr
        self.inet = INet(
            source_channels=1, 
            output_channels=1, num_filters=self.hparams.features
        )
        self.pnet = PNet(
            source_channels=1, 
            output_channels=1, num_filters=self.hparams.features
        )
        self.l1loss = nn.L1Loss()
        # self.vggperceptualoss = VGGPerceptualLoss()
    def forward(self, x, y, a, b):
        # return self.rnet(x)
        xy, feat_xy = self.inet(x)         # ct from xr
        yx, feat_yx = self.pnet(y)         # xr from ct
        ab, feat_ab = self.inet(a)         # ct from xr 
        aba, feat_aba = self.pnet(ab)     # xr from ct
        ba, feat_ba = self.pnet(b)         # xr from ct
        bab, feat_bab = self.inet(ba)     # ct from xr

        return xy, yx, ab, aba, ba, bab, \
               feat_xy, feat_yx, feat_ab, feat_aba, feat_ba, feat_bab 


    def training_step(self, batch, batch_nb):
        x, y, a, b = batch
        x = x / 255.0
        y = y / 255.0
        a = a / 255.0
        b = b / 255.0
        xy, yx, ab, aba, ba, bab, \
        feat_xy, feat_yx, feat_ab, feat_aba, feat_ba, feat_bab  = self.forward(x, y, a, b)
        loss_l1_xy = self.l1loss(xy, y) 
        loss_l1_yx = self.l1loss(yx, x) 
        loss_l1_ab = self.l1loss(aba, a) 
        loss_l1_ba = self.l1loss(bab, b) 
        # loss_l1_fxy = self.l1loss(feat_xy, feat_yx) 
        # loss_l1_fab = self.l1loss(feat_ab, feat_aba) 
        # loss_l1_fba = self.l1loss(feat_ba, feat_bab) 
        # loss_perceptual_xy = self.vggperceptualoss(xy, y) 
        # loss_perceptual_yx = self.vggperceptualoss(yx, x) 
        # loss_perceptual_ab = self.vggperceptualoss(ab, b) 
        # loss_perceptual_ba = self.vggperceptualoss(ba, a) 
        loss_l1 = loss_l1_xy + loss_l1_yx  + loss_l1_ab + loss_l1_ba 
        # loss_ft = loss_l1_fxy + loss_l1_fab + loss_l1_fba 
        # loss_perceptual = loss_perceptual_xy + loss_perceptual_yx \
        #                 + loss_perceptual_ab + loss_perceptual_ba 
        loss = loss_l1 #+ loss_ft #+ loss_perceptual
     
        mid = int(y.shape[1]/2)
        vis_images = torch.cat([torch.cat([x, y[:,mid:mid+1,:,:], xy[:,mid:mid+1,:,:], yx], dim=-1), 
                                torch.cat([a, b[:,mid:mid+1,:,:], ab[:,mid:mid+1,:,:], aba], dim=-1), 
                                torch.cat([b[:,mid:mid+1,:,:], a, ba, bab[:,mid:mid+1,:,:]], dim=-1), 
                                ], dim=-2) 
        grid = torchvision.utils.make_grid(vis_images, nrow=2, padding=0)
        self.logger.experiment.add_image('train_vis', grid, self.current_epoch)
        tensorboard_logs = {'train_loss': loss, 
                            'loss_l1_xy': loss_l1_xy,
                            'loss_l1_yx': loss_l1_yx,
                            'loss_l1_ab': loss_l1_ab,
                            'loss_l1_ba': loss_l1_ba,
                            'loss_l1': loss_l1,
                            # 'loss_l1_fxy': loss_l1_fxy,
                            # 'loss_l1_fab': loss_l1_fab,
                            # 'loss_l1_fba': loss_l1_fba,
                            # 'loss_ft': loss_ft,
                            # 'loss_perceptual_xy': loss_perceptual_xy,
                            # 'loss_perceptual_yx': loss_perceptual_yx,
                            # 'loss_perceptual_ab': loss_perceptual_ab,
                            # 'loss_perceptual_ba': loss_perceptual_ba,
                            # 'loss_perceptual': loss_perceptual,
                            # 'lr': self.trainer.lr_schedulers[0].get_lr()[0]
                            }
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        x, y, a, b = batch
        x = x / 255.0
        y = y / 255.0
        a = a / 255.0
        b = b / 255.0
        xy, yx, ab, aba, ba, bab, \
        feat_xy, feat_yx, feat_ab, feat_aba, feat_ba, feat_bab  = self.forward(x, y, a, b)
        loss_l1_xy = self.l1loss(xy, y) 
        loss_l1_yx = self.l1loss(yx, x) 
        loss_l1_ab = self.l1loss(aba, a) 
        loss_l1_ba = self.l1loss(bab, b) 
        # loss_l1_fxy = self.l1loss(feat_xy, feat_yx) 
        # loss_l1_fab = self.l1loss(feat_ab, feat_aba) 
        # loss_l1_fba = self.l1loss(feat_ba, feat_bab) 
        # loss_perceptual_xy = self.vggperceptualoss(xy, y) 
        # loss_perceptual_yx = self.vggperceptualoss(yx, x) 
        # loss_perceptual_ab = self.vggperceptualoss(ab, b) 
        # loss_perceptual_ba = self.vggperceptualoss(ba, a) 
        loss_l1 = loss_l1_xy + loss_l1_yx  + loss_l1_ab + loss_l1_ba 
        # loss_ft = loss_l1_fxy + loss_l1_fab + loss_l1_fba 
        # loss_perceptual = loss_perceptual_xy + loss_perceptual_yx \
        #                 + loss_perceptual_ab + loss_perceptual_ba 
        loss = loss_l1 #+ loss_ft #+ loss_perceptual
       
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
        # return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # optimizer = torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=self.hparams.lr, eps=1e-08)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-4, total_steps=1000)
        # return [optimizer], [scheduler]
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]
        # optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9, nesterov=True)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, 
        #                                                 steps_per_epoch=1000, epochs=10)
        return [optimizer], [scheduler]

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
            AB.GaussianBlur(p=0.5),
            AB.GaussNoise(p=0.5),
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
            # AB.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.50),
            # AB.RandomGamma(gamma_limit=(80, 120), p=0.50),
            # AB.GaussianBlur(p=0.05),
            # AB.GaussNoise(p=0.05),
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
        callbacks=[LearningRateLogger()],
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
    parser.add_argument("--gpus", default=-1, help="number of available GPUs")
    parser.add_argument('--distributed-backend', type=str, default='ddp', choices=('dp', 'ddp', 'ddp2'),
                        help='supports three options dp, ddp, ddp2')
    parser.add_argument('--use_amp', default=True, action='store_true', help='if true uses 16 bit precision')
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--num_workers", type=int, default=4, help="size of the workers")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--nb_layer", type=int, default=5, help="number of layers on u-net")
    parser.add_argument("--features", type=int, default=8, help="number of features in single layer")
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

