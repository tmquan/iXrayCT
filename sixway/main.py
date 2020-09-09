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
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision 

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateLogger


import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')

from model import INet
from model import PNet
# Data preparation for 6 models
# Training dataset
# model A
train_ct3xr2_ct3_paired_dir = [
    '/u01/data/iXrayCT_COVID/data_resized/train/paired/RADIOPAEDIA/pos/ct3/',
    '/u01/data/iXrayCT_COVID/data_resized/train/paired/RADIOPAEDIA/neg/ct3/',
    '/u01/data/iXrayCT_COVID/data_resized/train/paired/BIMCV/pos/ct3/',
]

train_ct3xr2_xr2_paired_dir = [
    '/u01/data/iXrayCT_COVID/data_resized/train/paired/RADIOPAEDIA/pos/xr2/',
    '/u01/data/iXrayCT_COVID/data_resized/train/paired/RADIOPAEDIA/neg/xr2/',
    '/u01/data/iXrayCT_COVID/data_resized/train/paired/BIMCV/pos/xr2/',
]

# model B
train_xr2ct3_xr2_paired_dir = train_ct3xr2_xr2_paired_dir
train_xr2ct3_ct3_paired_dir = train_ct3xr2_ct3_paired_dir

# model C
train_ct3lung3_img_paired_dir = [
    '/u01/data/iXrayCT_COVID/data_resized/train/ct3covid3/MOSMED/pos/images/', 
    '/u01/data/iXrayCT_COVID/data_resized/train/ct3covid3/NSCLC/neg/images/'
]

train_ct3lung3_lbl_paired_dir = [
    '/u01/data/iXrayCT_COVID/data_resized/train/ct3covid3/MOSMED/pos/labels/', 
    '/u01/data/iXrayCT_COVID/data_resized/train/ct3covid3/NSCLC/neg/labels/',
]

# model D
train_ct3covid3_img_paired_dir = [
    '/u01/data/iXrayCT_COVID/data_resized/train/ct3lung3/NSCLC/neg/images/',
]   

train_ct3covid3_lbl_paired_dir = [
    '/u01/data/iXrayCT_COVID/data_resized/train/ct3lung3/NSCLC/neg/labels/',
]

# model E
train_xr2lung2_img_paired_dir = [
    '/u01/data/iXrayCT_COVID/data_resized/train/xr2lung2/NLMMC/neg/images/',
]

train_xr2lung2_lbl_paired_dir = [
    '/u01/data/iXrayCT_COVID/data_resized/train/xr2lung2/NLMMC/neg/labels/',
]

# model F
train_xr2covid2_img_paired_dir = [
    '/u01/data/iXrayCT_COVID/data_resized/train/xr2covid2/NLMMC/neg/images/',
    '/u01/data/iXrayCT_COVID/data_resized/train/xr2covid2/IEEE8023/neg/images/',
]

train_xr2covid2_lbl_paired_dir = [
    '/u01/data/iXrayCT_COVID/data_resized/train/xr2covid2/NLMMC/neg/labels/',
    '/u01/data/iXrayCT_COVID/data_resized/train/xr2covid2/IEEE8023/neg/labels/',
]

train_ct3xr2_xr2_unpaired_dir = [
    '/u01/data/iXrayCT_COVID/data_resized/train/unpaired/IEEE8023/pos/xr2/',
    '/u01/data/iXrayCT_COVID/data_resized/train/unpaired/IEEE8023/neg/xr2/',
    '/u01/data/iXrayCT_COVID/data_resized/train/unpaired/NLMMC/neg/xr2/',
]

train_ct3xr2_ct3_unpaired_dir = [
    '/u01/data/iXrayCT_COVID/data_resized/train/unpaired/MOSMED/pos/ct3/',
    '/u01/data/iXrayCT_COVID/data_resized/train/unpaired/LNDB/neg/ct3/',
    '/u01/data/iXrayCT_COVID/data_resized/train/unpaired/DSB3/neg/ct3/',
]

train_xr2ct3_xr2_unpaired_dir = train_ct3xr2_xr2_unpaired_dir
train_xr2ct3_ct3_unpaired_dir = train_ct3xr2_ct3_unpaired_dir

# Testing dataset
# model A
valid_ct3xr2_ct3_paired_dir = [
    '/u01/data/iXrayCT_COVID/data_resized/test/paired/RADIOPAEDIA/pos/ct3/',
    '/u01/data/iXrayCT_COVID/data_resized/test/paired/RADIOPAEDIA/neg/ct3/',
    '/u01/data/iXrayCT_COVID/data_resized/test/paired/BIMCV/pos/ct3/',
]

valid_ct3xr2_xr2_paired_dir = [
    '/u01/data/iXrayCT_COVID/data_resized/test/paired/RADIOPAEDIA/pos/xr2/',
    '/u01/data/iXrayCT_COVID/data_resized/test/paired/RADIOPAEDIA/neg/xr2/',
    '/u01/data/iXrayCT_COVID/data_resized/test/paired/BIMCV/pos/xr2/',
]

# model B
valid_xr2ct3_xr2_paired_dir = valid_ct3xr2_xr2_paired_dir
valid_xr2ct3_ct3_paired_dir = valid_ct3xr2_ct3_paired_dir

# model C
valid_ct3lung3_img_paired_dir = [
    '/u01/data/iXrayCT_COVID/data_resized/test/ct3lung3/MEDSEG/pos/images/', 
    '/u01/data/iXrayCT_COVID/data_resized/test/ct3lung3/NSCLC/neg/images/',
]

valid_ct3lung3_lbl_paired_dir = [
    '/u01/data/iXrayCT_COVID/data_resized/test/ct3lung3/MEDSEG/pos/labels/', 
    '/u01/data/iXrayCT_COVID/data_resized/test/ct3lung3/NSCLC/neg/labels/', 
]

# model D
valid_ct3covid3_img_paired_dir = [
    '/u01/data/iXrayCT_COVID/data_resized/test/ct3covid3/MEDSEG/pos/images/', 
    '/u01/data/iXrayCT_COVID/data_resized/test/ct3covid3/NSCLC/neg/images/',
]   

valid_ct3covid3_lbl_paired_dir = [
    '/u01/data/iXrayCT_COVID/data_resized/test/ct3covid3/MEDSEG/pos/labels/', 
    '/u01/data/iXrayCT_COVID/data_resized/test/ct3covid3/NSCLC/neg/labels/', 
]

# model E
valid_xr2lung2_img_paired_dir = [
    '/u01/data/iXrayCT_COVID/data_resized/test/xr2lung2/JSRT/neg/images/',
]

valid_xr2lung2_lbl_paired_dir = [
    '/u01/data/iXrayCT_COVID/data_resized/test/xr2lung2/JSRT/neg/labels/'
]

# model F
valid_xr2covid2_img_paired_dir = [
    '/u01/data/iXrayCT_COVID/data_resized/test/xr2covid2/IEEE8023/neg/images/',
    '/u01/data/iXrayCT_COVID/data_resized/test/xr2covid2/JSRT/neg/images/',
]

valid_xr2covid2_lbl_paired_dir = [
    '/u01/data/iXrayCT_COVID/data_resized/test/xr2covid2/IEEE8023/neg/labels/',
    '/u01/data/iXrayCT_COVID/data_resized/test/xr2covid2/JSRT/neg/labels/',
]

valid_ct3xr2_xr2_unpaired_dir = [
    '/u01/data/iXrayCT_COVID/data_resized/test/paired/RADIOPAEDIA/pos/xr2/',
    '/u01/data/iXrayCT_COVID/data_resized/test/paired/RADIOPAEDIA/neg/xr2/',
    '/u01/data/iXrayCT_COVID/data_resized/test/paired/BIMCV/pos/xr2/',
]

valid_ct3xr2_ct3_unpaired_dir = [
    '/u01/data/iXrayCT_COVID/data_resized/test/paired/RADIOPAEDIA/pos/ct3/',
    '/u01/data/iXrayCT_COVID/data_resized/test/paired/RADIOPAEDIA/neg/ct3/',
    '/u01/data/iXrayCT_COVID/data_resized/test/paired/BIMCV/pos/ct3/',
]

valid_xr2ct3_xr2_unpaired_dir = valid_ct3xr2_xr2_unpaired_dir
valid_xr2ct3_ct3_unpaired_dir = valid_ct3xr2_ct3_unpaired_dir


def worker_init_fn(worker_id):
    torch.initial_seed()


class CustomNativeDataset(Dataset):
    def __init__(self,
                 ct3xr2_xr2_unpaired_dir,
                 ct3xr2_ct3_unpaired_dir,
                 xr2ct3_xr2_unpaired_dir, 
                 xr2ct3_ct3_unpaired_dir,
                 ct3xr2_ct3_paired_dir, 
                 ct3xr2_xr2_paired_dir, 
                 xr2ct3_ct3_paired_dir, 
                 xr2ct3_xr2_paired_dir, 
                 ct3lung3_img_paired_dir, 
                 ct3lung3_lbl_paired_dir, 
                 ct3covid3_img_paired_dir, 
                 ct3covid3_lbl_paired_dir, 
                 xr2lung2_img_paired_dir, 
                 xr2lung2_lbl_paired_dir, 
                 xr2covid2_img_paired_dir, 
                 xr2covid2_lbl_paired_dir, 
                 train_or_valid='train',
                 size=1000,
                 transforms=None
                 ):
        self.size = size
        self.is_train = True if train_or_valid == 'train' else False
        print('Training') if self.is_train else print('Testing')
        self.transforms = transforms

        # Unpaired 
        self.ct3xr2_ct3_unpaired_files = [glob.glob(os.path.join(folder, '*.*')) for folder in ct3xr2_ct3_unpaired_dir]
        self.ct3xr2_ct3_unpaired_files = natsorted([item for sublist in self.ct3xr2_ct3_unpaired_files for item in sublist])
        self.ct3xr2_xr2_unpaired_files = [glob.glob(os.path.join(folder, '*.*')) for folder in ct3xr2_xr2_unpaired_dir]
        self.ct3xr2_xr2_unpaired_files = natsorted([item for sublist in self.ct3xr2_xr2_unpaired_files for item in sublist])
        print(len(self.ct3xr2_ct3_unpaired_files), len(self.ct3xr2_xr2_unpaired_files))

        self.xr2ct3_xr2_unpaired_files = [glob.glob(os.path.join(folder, '*.*')) for folder in xr2ct3_xr2_unpaired_dir]
        self.xr2ct3_xr2_unpaired_files = natsorted([item for sublist in self.xr2ct3_xr2_unpaired_files for item in sublist])
        self.xr2ct3_ct3_unpaired_files = [glob.glob(os.path.join(folder, '*.*')) for folder in xr2ct3_ct3_unpaired_dir]
        self.xr2ct3_ct3_unpaired_files = natsorted([item for sublist in self.xr2ct3_ct3_unpaired_files for item in sublist])
        print(len(self.xr2ct3_xr2_unpaired_files), len(self.xr2ct3_ct3_unpaired_files))


        self.ct3xr2_ct3_paired_files = [glob.glob(os.path.join(folder, '*.*')) for folder in ct3xr2_ct3_paired_dir]
        self.ct3xr2_ct3_paired_files = natsorted([item for sublist in self.ct3xr2_ct3_paired_files for item in sublist])
        self.ct3xr2_xr2_paired_files = [glob.glob(os.path.join(folder, '*.*')) for folder in ct3xr2_xr2_paired_dir]
        self.ct3xr2_xr2_paired_files = natsorted([item for sublist in self.ct3xr2_xr2_paired_files for item in sublist])
        assert len(self.ct3xr2_ct3_paired_files) == len(self.ct3xr2_xr2_paired_files)
        print(len(self.ct3xr2_ct3_paired_files), len(self.ct3xr2_xr2_paired_files))

        self.xr2ct3_xr2_paired_files = [glob.glob(os.path.join(folder, '*.*')) for folder in xr2ct3_xr2_paired_dir]
        self.xr2ct3_xr2_paired_files = natsorted([item for sublist in self.xr2ct3_xr2_paired_files for item in sublist])
        self.xr2ct3_ct3_paired_files = [glob.glob(os.path.join(folder, '*.*')) for folder in xr2ct3_ct3_paired_dir]
        self.xr2ct3_ct3_paired_files = natsorted([item for sublist in self.xr2ct3_ct3_paired_files for item in sublist])
        assert len(self.xr2ct3_xr2_paired_files) == len(self.xr2ct3_ct3_paired_files)
        print(len(self.xr2ct3_xr2_paired_files), len(self.xr2ct3_ct3_paired_files))

        self.ct3lung3_img_paired_files = [glob.glob(os.path.join(folder, '*.*')) for folder in ct3lung3_img_paired_dir]
        self.ct3lung3_img_paired_files = natsorted([item for sublist in self.ct3lung3_img_paired_files for item in sublist])
        self.ct3lung3_lbl_paired_files = [glob.glob(os.path.join(folder, '*.*')) for folder in ct3lung3_lbl_paired_dir]
        self.ct3lung3_lbl_paired_files = natsorted([item for sublist in self.ct3lung3_lbl_paired_files for item in sublist])
        assert len(self.ct3lung3_img_paired_files) == len(self.ct3lung3_lbl_paired_files)
        print(len(self.ct3lung3_img_paired_files), len(self.ct3lung3_lbl_paired_files))

        self.ct3covid3_img_paired_files = [glob.glob(os.path.join(folder, '*.*')) for folder in ct3covid3_img_paired_dir]
        self.ct3covid3_img_paired_files = natsorted([item for sublist in self.ct3covid3_img_paired_files for item in sublist])
        self.ct3covid3_lbl_paired_files = [glob.glob(os.path.join(folder, '*.*')) for folder in ct3covid3_lbl_paired_dir]
        self.ct3covid3_lbl_paired_files = natsorted([item for sublist in self.ct3covid3_lbl_paired_files for item in sublist])
        assert len(self.ct3covid3_img_paired_files) == len(self.ct3covid3_lbl_paired_files)
        print(len(self.ct3covid3_img_paired_files), len(self.ct3covid3_lbl_paired_files))

        self.xr2lung2_img_paired_files = [glob.glob(os.path.join(folder, '*.*')) for folder in xr2lung2_img_paired_dir]
        self.xr2lung2_img_paired_files = natsorted([item for sublist in self.xr2lung2_img_paired_files for item in sublist])
        self.xr2lung2_lbl_paired_files = [glob.glob(os.path.join(folder, '*.*')) for folder in xr2lung2_lbl_paired_dir]
        self.xr2lung2_lbl_paired_files = natsorted([item for sublist in self.xr2lung2_lbl_paired_files for item in sublist])
        assert len(self.xr2lung2_img_paired_files) == len(self.xr2lung2_lbl_paired_files)
        print(len(self.xr2lung2_img_paired_files), len(self.xr2lung2_lbl_paired_files))

        self.xr2covid2_img_paired_files = [glob.glob(os.path.join(folder, '*.*')) for folder in xr2covid2_img_paired_dir]
        self.xr2covid2_img_paired_files = natsorted([item for sublist in self.xr2covid2_img_paired_files for item in sublist])
        self.xr2covid2_lbl_paired_files = [glob.glob(os.path.join(folder, '*.*')) for folder in xr2covid2_lbl_paired_dir]
        self.xr2covid2_lbl_paired_files = natsorted([item for sublist in self.xr2covid2_lbl_paired_files for item in sublist])
        assert len(self.xr2covid2_img_paired_files) == len(self.xr2covid2_lbl_paired_files)
        print(len(self.xr2covid2_img_paired_files), len(self.xr2covid2_lbl_paired_files))

        

    def __len__(self):
        return self.size #if self.is_train else len(self.imagepairedfiles)

    def __call__(self):
        np.random.seed(datetime.datetime.now().second +
                       datetime.datetime.now().millisecond)

    def __getitem__(self, idx):
        # pidx = torch.randint(len(self.imagepairedfiles), (1, 1))
        # imagepaired = skimage.io.imread(self.imagepairedfiles[pidx])
        # labelpaired = skimage.io.imread(self.labelpairedfiles[pidx])

        # aidx = torch.randint(len(self.imageunpairedfiles), (1, 1))
        # bidx = torch.randint(len(self.labelunpairedfiles), (1, 1))
        # imageunpaired = skimage.io.imread(self.imageunpairedfiles[aidx])
        # labelunpaired = skimage.io.imread(self.labelunpairedfiles[bidx])

        # if self.transforms is not None:
        #     untransformed = self.transforms(image=np.transpose(np.expand_dims(imageunpaired, 0), (1, 2, 0)))
        #     imageunpaired = np.squeeze(np.transpose(untransformed['image'], (2, 0, 1)), 0)
        #     untransformed = self.transforms(image=np.transpose(labelunpaired, (1, 2, 0)))
        #     labelunpaired = np.transpose(untransformed['image'], (2, 0, 1))


        # imagepaired = cv2.resize(imagepaired, (256, 256))
        # labelpaired = scipy.ndimage.zoom(labelpaired, scale, order=3)
        # imageunpaired = cv2.resize(imageunpaired, (256, 256))
        # labelunpaired = scipy.ndimage.zoom(labelunpaired, scale, order=3)

        # return torch.Tensor(imagepaired).float().unsqueeze_(0),   \
        #     torch.Tensor(labelpaired).float(),                 \
        #     torch.Tensor(imageunpaired).float().unsqueeze_(0), \
        #     torch.Tensor(labelunpaired).float(),

        aidx = torch.randint(len(self.ct3xr2_ct3_unpaired_files), (1, 1))     
        bidx = torch.randint(len(self.ct3xr2_xr2_unpaired_files), (1, 1))     
        ct3xr2_ct3_unpaired = skimage.io.imread(self.ct3xr2_ct3_unpaired_files[aidx]).astype(np.uint8)         
        ct3xr2_xr2_unpaired = skimage.io.imread(self.ct3xr2_xr2_unpaired_files[bidx]).astype(np.uint8)    

        aidx = torch.randint(len(self.xr2ct3_xr2_unpaired_files), (1, 1))     
        bidx = torch.randint(len(self.xr2ct3_ct3_unpaired_files), (1, 1))     
        xr2ct3_xr2_unpaired = skimage.io.imread(self.xr2ct3_xr2_unpaired_files[aidx]).astype(np.uint8)    
        xr2ct3_ct3_unpaired = skimage.io.imread(self.xr2ct3_ct3_unpaired_files[bidx]).astype(np.uint8)    

        pidx = torch.randint(len(self.ct3xr2_ct3_paired_files), (1, 1))     
        ct3xr2_ct3_paired = skimage.io.imread(self.ct3xr2_ct3_paired_files[pidx]).astype(np.uint8)         
        ct3xr2_xr2_paired = skimage.io.imread(self.ct3xr2_xr2_paired_files[pidx]).astype(np.uint8)     

        pidx = torch.randint(len(self.xr2ct3_xr2_paired_files), (1, 1))     
        xr2ct3_xr2_paired = skimage.io.imread(self.xr2ct3_xr2_paired_files[pidx]).astype(np.uint8)         
        xr2ct3_ct3_paired = skimage.io.imread(self.xr2ct3_ct3_paired_files[pidx]).astype(np.uint8)       

        pidx = torch.randint(len(self.ct3lung3_img_paired_files), (1, 1))     
        ct3lung3_img_paired = skimage.io.imread(self.ct3lung3_img_paired_files[pidx]).astype(np.uint8)    
        ct3lung3_lbl_paired = skimage.io.imread(self.ct3lung3_lbl_paired_files[pidx]).astype(np.uint8)

        pidx = torch.randint(len(self.ct3covid3_img_paired_files), (1, 1))     
        ct3covid3_img_paired = skimage.io.imread(self.ct3covid3_img_paired_files[pidx]).astype(np.uint8)    
        ct3covid3_lbl_paired = skimage.io.imread(self.ct3covid3_lbl_paired_files[pidx]).astype(np.uint8)

        pidx = torch.randint(len(self.xr2lung2_img_paired_files), (1, 1))     
        xr2lung2_img_paired = skimage.io.imread(self.xr2lung2_img_paired_files[pidx]).astype(np.uint8)    
        xr2lung2_lbl_paired = skimage.io.imread(self.xr2lung2_lbl_paired_files[pidx]).astype(np.uint8)

        pidx = torch.randint(len(self.xr2covid2_img_paired_files), (1, 1))     
        xr2covid2_img_paired = skimage.io.imread(self.xr2covid2_img_paired_files[pidx]).astype(np.uint8)    
        xr2covid2_lbl_paired = skimage.io.imread(self.xr2covid2_lbl_paired_files[pidx]).astype(np.uint8)

        
        if self.transforms is not None:
            untransformed = self.transforms(image=np.transpose(xr2ct3_ct3_unpaired, (1, 2, 0)))
            ct3xr2_ct3_unpaired = np.transpose(untransformed['image'], (2, 0, 1))     
            untransformed = self.transforms(image=np.transpose(np.expand_dims(ct3xr2_xr2_unpaired, 0), (1, 2, 0)))
            ct3xr2_xr2_unpaired = np.squeeze(np.transpose(untransformed['image'], (2, 0, 1)), 0)
            untransformed = self.transforms(image=np.transpose(np.expand_dims(xr2ct3_xr2_unpaired, 0), (1, 2, 0)))
            xr2ct3_xr2_unpaired = np.squeeze(np.transpose(untransformed['image'], (2, 0, 1)), 0)
            untransformed = self.transforms(image=np.transpose(ct3xr2_ct3_unpaired, (1, 2, 0)))
            xr2ct3_ct3_unpaired = np.transpose(untransformed['image'], (2, 0, 1))     
            
        # scale = [64.0 / labelpaired.shape[0], 
        #          256.0/labelpaired.shape[1], 
        #          256.0/labelpaired.shape[2]]
        scale = [1, 1, 1]

        ct3xr2_ct3_paired = torch.Tensor( scipy.ndimage.zoom(ct3xr2_ct3_paired, scale, order=3) ).float()
        ct3xr2_xr2_paired = torch.Tensor( cv2.resize(ct3xr2_xr2_paired, (256, 256)) ).float().unsqueeze_(0)
        xr2ct3_xr2_paired = torch.Tensor( cv2.resize(xr2ct3_xr2_paired, (256, 256)) ).float().unsqueeze_(0)
        xr2ct3_ct3_paired = torch.Tensor( scipy.ndimage.zoom(xr2ct3_ct3_paired, scale, order=3) ).float()

        ct3xr2_ct3_unpaired = torch.Tensor( scipy.ndimage.zoom(ct3xr2_ct3_unpaired, scale, order=3) ).float()
        ct3xr2_xr2_unpaired = torch.Tensor( cv2.resize(ct3xr2_xr2_unpaired, (256, 256)) ).float().unsqueeze_(0)
        xr2ct3_xr2_unpaired = torch.Tensor( cv2.resize(xr2ct3_xr2_unpaired, (256, 256)) ).float().unsqueeze_(0)
        xr2ct3_ct3_unpaired = torch.Tensor( scipy.ndimage.zoom(xr2ct3_ct3_unpaired, scale, order=3) ).float()

        ct3lung3_img_paired = torch.Tensor( scipy.ndimage.zoom(ct3lung3_img_paired, scale, order=3) ).float()
        ct3lung3_lbl_paired = torch.Tensor( scipy.ndimage.zoom(ct3lung3_lbl_paired, scale, order=3) ).float()
        ct3covid3_img_paired = torch.Tensor( scipy.ndimage.zoom(ct3covid3_img_paired, scale, order=3) ).float()
        ct3covid3_lbl_paired = torch.Tensor( scipy.ndimage.zoom(ct3covid3_lbl_paired, scale, order=3) ).float()

        xr2lung2_img_paired = torch.Tensor( cv2.resize(xr2lung2_img_paired, (256, 256)) ).float().unsqueeze_(0)
        xr2lung2_lbl_paired = torch.Tensor( cv2.resize(xr2lung2_lbl_paired, (256, 256)) ).float().unsqueeze_(0)
        xr2covid2_img_paired = torch.Tensor( cv2.resize(xr2covid2_img_paired, (256, 256)) ).float().unsqueeze_(0)
        xr2covid2_lbl_paired = torch.Tensor( cv2.resize(xr2covid2_lbl_paired, (256, 256)) ).float().unsqueeze_(0)
                
        return [
            ct3xr2_ct3_paired,
            ct3xr2_xr2_paired,
            xr2ct3_xr2_paired,
            xr2ct3_ct3_paired,
            ct3xr2_ct3_unpaired,
            ct3xr2_xr2_unpaired,
            xr2ct3_xr2_unpaired,
            xr2ct3_ct3_unpaired,
            ct3lung3_img_paired,
            ct3lung3_lbl_paired,
            ct3covid3_img_paired,
            ct3covid3_lbl_paired,
            xr2lung2_img_paired,
            xr2lung2_lbl_paired,
            xr2covid2_img_paired,
            xr2covid2_lbl_paired,
        ]

class Model(pl.LightningModule):
    def __init__(self, hparams, n_classes=1):
        super(Model, self).__init__()
        self.hparams = hparams
        # self.example_input_array = torch.rand(self.hparams.batch_size, 1, 256, 256),  \
        #                            torch.rand(self.hparams.batch_size, 64, 256, 256), \
        #                            torch.rand(self.hparams.batch_size, 1, 256, 256),  \
        #                            torch.rand(self.hparams.batch_size, 64, 256, 256)
        
        self.inet = INet(
            source_channels=1, 
            output_channels=1, 
            num_filters=self.hparams.features,
        )
        self.pnet = PNet(
            source_channels=1, 
            output_channels=1, 
            num_filters=self.hparams.features,
        )
        self.l1loss = nn.L1Loss()

    def forward(self, 
                ct3xr2_ct3_paired,
                ct3xr2_xr2_paired,
                xr2ct3_xr2_paired,
                xr2ct3_ct3_paired,
                ct3xr2_ct3_unpaired,
                ct3xr2_xr2_unpaired,
                xr2ct3_xr2_unpaired,
                xr2ct3_ct3_unpaired,
                ct3lung3_img_paired,
                ct3lung3_lbl_paired,
                ct3covid3_img_paired,
                ct3covid3_lbl_paired,
                xr2lung2_img_paired,
                xr2lung2_lbl_paired,
                xr2covid2_img_paired,
                xr2covid2_lbl_paired
                ):
        pass
        # xy = self.inet(x)         # ct from xr
        # yx = self.pnet(y)         # xr from ct
        # ab = self.inet(a)         # ct from xr 
        # aba = self.pnet(ab)       # xr from ct
        # ba = self.pnet(b)         # xr from ct
        # bab = self.inet(ba)       # ct from xr

        # return xy, yx, ab, aba, ba, bab


    def training_step(self, batch, batch_nb):
        pass
        # x, y, a, b = batch
        # x = x / 255.0
        # y = y / 255.0
        # a = a / 255.0
        # b = b / 255.0
        # xy, yx, ab, aba, ba, bab = self.forward(x, y, a, b)
        # loss_l1_xy = self.l1loss(xy, y) 
        # loss_l1_yx = self.l1loss(yx, x) 
        # loss_l1_ab = self.l1loss(aba, a) 
        # loss_l1_ba = self.l1loss(bab, b) 
        # loss_l1 = loss_l1_xy + loss_l1_yx  + loss_l1_ab + loss_l1_ba 
        # loss = loss_l1 
     
        # mid = int(y.shape[1]/2)
        # vis_images = torch.cat([torch.cat([x, y[:,mid:mid+1,:,:], xy[:,mid:mid+1,:,:], yx], dim=-1), 
        #                         torch.cat([a, b[:,mid:mid+1,:,:], ab[:,mid:mid+1,:,:], aba], dim=-1), 
        #                         torch.cat([b[:,mid:mid+1,:,:], a, ba, bab[:,mid:mid+1,:,:]], dim=-1), 
        #                         ], dim=-2) 
        # grid = torchvision.utils.make_grid(vis_images, nrow=2, padding=0)
        # self.logger.experiment.add_image('train_vis', grid, self.current_epoch)
        # tensorboard_logs = {'train_loss': loss, 
        #                     'loss_l1_xy': loss_l1_xy,
        #                     'loss_l1_yx': loss_l1_yx,
        #                     'loss_l1_ab': loss_l1_ab,
        #                     'loss_l1_ba': loss_l1_ba,
        #                     'loss_l1': loss_l1,
        #                     }
        # return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        pass 
        # x, y, a, b = batch
        # x = x / 255.0
        # y = y / 255.0
        # a = a / 255.0
        # b = b / 255.0
        # xy, yx, ab, aba, ba, bab = self.forward(x, y, a, b)
        # loss_l1_xy = self.l1loss(xy, y) 
        # loss_l1_yx = self.l1loss(yx, x) 
        # loss_l1_ab = self.l1loss(aba, a) 
        # loss_l1_ba = self.l1loss(bab, b) 
        # loss_l1 = loss_l1_xy + loss_l1_yx  + loss_l1_ab + loss_l1_ba 

        # loss = loss_l1 
       
        # mid = int(y.shape[1]/2)
        # vis_images = torch.cat([torch.cat([x, y[:,mid:mid+1,:,:], xy[:,mid:mid+1,:,:], yx], dim=-1), 
        #                         torch.cat([a, b[:,mid:mid+1,:,:], ab[:,mid:mid+1,:,:], aba], dim=-1), 
        #                         torch.cat([b[:,mid:mid+1,:,:], a, ba, bab[:,mid:mid+1,:,:]], dim=-1), 
        #                         ], dim=-2) 
        # grid = torchvision.utils.make_grid(vis_images, nrow=2, padding=0)
        # self.logger.experiment.add_image('valid_vis', grid, self.current_epoch)
        # return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        pass 
        # avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        # tensorboard_logs = {'val_loss': avg_loss}
        # return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]
        
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
       

        # train_ds = CustomNativeDataset(imagepaireddir=train_img_paired_dir, 
        #                                labelpaireddir=train_lbl_paired_dir, 
        #                                imageunpaireddir=train_img_unpaired_dir, 
        #                                labelunpaireddir=train_lbl_unpaired_dir, 
        #                                train_or_valid='train', 
        #                                transforms=train_tfm)
        # valid_ds = CustomNativeDataset(imagepaireddir=valid_img_paired_dir, 
        #                                labelpaireddir=valid_lbl_paired_dir, 
        #                                imageunpaireddir=valid_img_unpaired_dir, 
        #                                labelunpaireddir=valid_lbl_unpaired_dir, 
        #                                train_or_valid='valid', 
        #                                transforms=valid_tfm)
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
        early_stop_callback=stop_callback,
        gpus=hparams.gpus,
        callbacks=[LearningRateLogger()],
        max_epochs=hparams.epochs,
        accumulate_grad_batches=hparams.grad_batches,
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
    parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
    parser.add_argument("--num_workers", type=int, default=8, help="size of the workers")
    parser.add_argument("--lr", type=float, default=0.02, help="learning rate")
    parser.add_argument("--nb_layer", type=int, default=5, help="number of layers on u-net")
    parser.add_argument("--features", type=int, default=10, help="number of features in single layer")
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