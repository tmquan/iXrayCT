# QuanTM
import os
import glob
from natsort import natsorted

import numpy as np
import cv2
import skimage.io
import scipy.ndimage
import torch
from torch.utils.data import Dataset

def worker_init_fn(worker_id):
    torch.initial_seed()

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

class CustomGlobbingFiles(Dataset):
    def __init__(self, folders=None):
        self.folders = folders

    def __call__(self, folders):
        self.folders = folders
        self.subfolders = [glob.glob(os.path.join(folder, '*.*')) for folder in self.folders]
        self.files = natsorted([item for sublist in self.subfolders for item in sublist])
        return self.files

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
                 shape=256,
                 dimx=256,
                 dimy=256,
                 dimz=64,
                 transforms=None
                 ):
        self.size = size
        self.dimx = dimx
        self.dimy = dimy
        self.dimz = dimz
        self.is_train = True if train_or_valid == 'train' else False
        print('Training') if self.is_train else print('Testing')
        self.transforms = transforms
        self.globber = CustomGlobbingFiles()

        # Unpaired
        self.ct3xr2_ct3_unpaired_files = self.globber(folders=ct3xr2_ct3_unpaired_dir)
        self.ct3xr2_xr2_unpaired_files = self.globber(folders=ct3xr2_xr2_unpaired_dir)
        print(len(self.ct3xr2_ct3_unpaired_files), len(self.ct3xr2_xr2_unpaired_files))

        self.xr2ct3_xr2_unpaired_files = self.globber(folders=xr2ct3_xr2_unpaired_dir)
        self.xr2ct3_ct3_unpaired_files = self.globber(folders=xr2ct3_ct3_unpaired_dir)
        print(len(self.xr2ct3_xr2_unpaired_files),
              len(self.xr2ct3_ct3_unpaired_files))

        self.ct3xr2_ct3_paired_files = self.globber(folders=ct3xr2_ct3_paired_dir)
        self.ct3xr2_xr2_paired_files = self.globber(folders=ct3xr2_xr2_paired_dir)
        assert len(self.ct3xr2_ct3_paired_files) == len(self.ct3xr2_xr2_paired_files)
        print(len(self.ct3xr2_ct3_paired_files), len(self.ct3xr2_xr2_paired_files))

        self.xr2ct3_xr2_paired_files = self.globber(folders=xr2ct3_xr2_paired_dir)
        self.xr2ct3_ct3_paired_files = self.globber(folders=xr2ct3_ct3_paired_dir)
        assert len(self.xr2ct3_xr2_paired_files) == len(self.xr2ct3_ct3_paired_files)
        print(len(self.xr2ct3_xr2_paired_files), len(self.xr2ct3_ct3_paired_files))

        self.ct3lung3_img_paired_files = self.globber(folders=ct3lung3_img_paired_dir)
        self.ct3lung3_lbl_paired_files = self.globber(folders=ct3lung3_lbl_paired_dir)
        assert len(self.ct3lung3_img_paired_files) == len(self.ct3lung3_lbl_paired_files)
        print(len(self.ct3lung3_img_paired_files), len(self.ct3lung3_lbl_paired_files))

        self.ct3covid3_img_paired_files = self.globber(folders=ct3covid3_img_paired_dir)
        self.ct3covid3_lbl_paired_files = self.globber(folders=ct3covid3_lbl_paired_dir)
        assert len(self.ct3covid3_img_paired_files) == len(self.ct3covid3_lbl_paired_files)
        print(len(self.ct3covid3_img_paired_files), len(self.ct3covid3_lbl_paired_files))

        self.xr2lung2_img_paired_files = self.globber(folders=xr2lung2_img_paired_dir)
        self.xr2lung2_lbl_paired_files = self.globber(folders=xr2lung2_lbl_paired_dir)
        assert len(self.xr2lung2_img_paired_files) == len(self.xr2lung2_lbl_paired_files)
        print(len(self.xr2lung2_img_paired_files), len(self.xr2lung2_lbl_paired_files))

        self.xr2covid2_img_paired_files = self.globber(folders=xr2covid2_img_paired_dir)
        self.xr2covid2_lbl_paired_files = self.globber(folders=xr2covid2_lbl_paired_dir)
        assert len(self.xr2covid2_img_paired_files) == len(self.xr2covid2_lbl_paired_files)
        print(len(self.xr2covid2_img_paired_files), len(self.xr2covid2_lbl_paired_files))
        print('\n')

    def __len__(self):
        return self.size  # if self.is_train else len(self.imagepairedfiles)

    def __call__(self):
        np.random.seed(datetime.datetime.now().second +
                       datetime.datetime.now().millisecond)

    def __getitem__(self, idx):
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
        # scale = [0.5, 0.5, 0.5]
        scale = [1, 1, 1]
        ct3xr2_ct3_paired = torch.Tensor(scipy.ndimage.zoom(ct3xr2_ct3_paired, scale, order=3)).float()
        ct3xr2_xr2_paired = torch.Tensor(cv2.resize(ct3xr2_xr2_paired, (self.dimy, self.dimx))).float().unsqueeze_(0)
        xr2ct3_xr2_paired = torch.Tensor(cv2.resize(xr2ct3_xr2_paired, (self.dimy, self.dimx))).float().unsqueeze_(0)
        xr2ct3_ct3_paired = torch.Tensor(scipy.ndimage.zoom(xr2ct3_ct3_paired, scale, order=3)).float()

        ct3xr2_ct3_unpaired = torch.Tensor(scipy.ndimage.zoom(ct3xr2_ct3_unpaired, scale, order=3)).float()
        ct3xr2_xr2_unpaired = torch.Tensor(cv2.resize(ct3xr2_xr2_unpaired, (self.dimy, self.dimx))).float().unsqueeze_(0)
        xr2ct3_xr2_unpaired = torch.Tensor(cv2.resize(xr2ct3_xr2_unpaired, (self.dimy, self.dimx))).float().unsqueeze_(0)
        xr2ct3_ct3_unpaired = torch.Tensor(scipy.ndimage.zoom(xr2ct3_ct3_unpaired, scale, order=3)).float()

        ct3lung3_img_paired = torch.Tensor(scipy.ndimage.zoom(ct3lung3_img_paired, scale, order=3)).float()
        ct3lung3_lbl_paired = torch.Tensor(scipy.ndimage.zoom(ct3lung3_lbl_paired, scale, order=3)).float()
        ct3covid3_img_paired = torch.Tensor(scipy.ndimage.zoom(ct3covid3_img_paired, scale, order=3)).float()
        ct3covid3_lbl_paired = torch.Tensor(scipy.ndimage.zoom(ct3covid3_lbl_paired, scale, order=3)).float()

        xr2lung2_img_paired = torch.Tensor(cv2.resize(xr2lung2_img_paired, (self.dimy, self.dimx))).float().unsqueeze_(0)
        xr2lung2_lbl_paired = torch.Tensor(cv2.resize(xr2lung2_lbl_paired, (self.dimy, self.dimx))).float().unsqueeze_(0)
        xr2covid2_img_paired = torch.Tensor(cv2.resize(xr2covid2_img_paired, (self.dimy, self.dimx))).float().unsqueeze_(0)
        xr2covid2_lbl_paired = torch.Tensor(cv2.resize(xr2covid2_lbl_paired, (self.dimy, self.dimx))).float().unsqueeze_(0)

        return [
            ct3xr2_ct3_unpaired,
            ct3xr2_xr2_unpaired,
            xr2ct3_xr2_unpaired,
            xr2ct3_ct3_unpaired,
            ct3xr2_ct3_paired,
            ct3xr2_xr2_paired,
            xr2ct3_xr2_paired,
            xr2ct3_ct3_paired,
            ct3lung3_img_paired,
            ct3lung3_lbl_paired,
            ct3covid3_img_paired,
            ct3covid3_lbl_paired,
            xr2lung2_img_paired,
            xr2lung2_lbl_paired,
            xr2covid2_img_paired,
            xr2covid2_lbl_paired,
        ]


if __name__ == '__main__':
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
                                   transforms=None,
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
                                   transforms=None,
                                   )
