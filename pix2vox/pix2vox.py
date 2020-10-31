# Setup 
import os
from argparse import ArgumentParser
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models
from torch.utils.data import DataLoader

import pytorch_lightning as pl

# DataModule
class CustomDataModule(pl.LightningDataModule):
    def __init__(self, 
    	data_folder: str = './', 
    	batch_size: int = 64, 
    	num_workers: int = 8
    ):
        super().__init__()
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.num_workers = num_workers
        pass

    def prepare_data(self):
        # download
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        # return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers)
        pass

    def val_dataloader(self):
        # return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)
        pass

    def test_dataloader(self):
        # return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)
        pass


# Native pytorch model
class Encoder(torch.nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.cfg = cfg

        # Layer Definition
        vgg16_bn = torchvision.models.vgg16_bn(pretrained=True)
        self.vgg = torch.nn.Sequential(*list(vgg16_bn.features.children()))[:27]
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3),
            torch.nn.BatchNorm2d(512),
            torch.nn.ELU(),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3),
            torch.nn.BatchNorm2d(512),
            torch.nn.ELU(),
            torch.nn.MaxPool2d(kernel_size=3)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, kernel_size=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ELU()
        )

        # Don't update params in VGG16
        for param in vgg16_bn.parameters():
            param.requires_grad = False

    def forward(self, rendering_images):
        # print(rendering_images.size())  # torch.Size([batch_size, n_views, img_c, img_h, img_w])
        rendering_images = rendering_images.permute(1, 0, 2, 3, 4).contiguous()
        rendering_images = torch.split(rendering_images, 1, dim=0)
        image_features = []

        for img in rendering_images:
            features = self.vgg(img.squeeze(dim=0))
            # print(features.size())    # torch.Size([batch_size, 512, 28, 28])
            features = self.layer1(features)
            # print(features.size())    # torch.Size([batch_size, 512, 26, 26])
            features = self.layer2(features)
            # print(features.size())    # torch.Size([batch_size, 512, 24, 24])
            features = self.layer3(features)
            # print(features.size())    # torch.Size([batch_size, 256, 8, 8])
            image_features.append(features)

        image_features = torch.stack(image_features).permute(1, 0, 2, 3, 4).contiguous()
        # print(image_features.size())  # torch.Size([batch_size, n_views, 256, 8, 8])
        return image_features

class Decoder(torch.nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        self.cfg = cfg

        # Layer Definition
        self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(2048, 512, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(512),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(512, 128, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.ReLU()
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 32, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU()
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 8, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(8),
            torch.nn.ReLU()
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(8, 1, kernel_size=1, bias=cfg.NETWORK.TCONV_USE_BIAS),
            torch.nn.Sigmoid()
        )

    def forward(self, image_features):
        image_features = image_features.permute(1, 0, 2, 3, 4).contiguous()
        image_features = torch.split(image_features, 1, dim=0)
        gen_volumes = []
        raw_features = []

        for features in image_features:
            gen_volume = features.view(-1, 2048, 2, 2, 2)
            # print(gen_volume.size())   # torch.Size([batch_size, 2048, 2, 2, 2])
            gen_volume = self.layer1(gen_volume)
            # print(gen_volume.size())   # torch.Size([batch_size, 512, 4, 4, 4])
            gen_volume = self.layer2(gen_volume)
            # print(gen_volume.size())   # torch.Size([batch_size, 128, 8, 8, 8])
            gen_volume = self.layer3(gen_volume)
            # print(gen_volume.size())   # torch.Size([batch_size, 32, 16, 16, 16])
            gen_volume = self.layer4(gen_volume)
            raw_feature = gen_volume
            # print(gen_volume.size())   # torch.Size([batch_size, 8, 32, 32, 32])
            gen_volume = self.layer5(gen_volume)
            # print(gen_volume.size())   # torch.Size([batch_size, 1, 32, 32, 32])
            raw_feature = torch.cat((raw_feature, gen_volume), dim=1)
            # print(raw_feature.size())  # torch.Size([batch_size, 9, 32, 32, 32])

            gen_volumes.append(torch.squeeze(gen_volume, dim=1))
            raw_features.append(raw_feature)

        gen_volumes = torch.stack(gen_volumes).permute(1, 0, 2, 3, 4).contiguous()
        raw_features = torch.stack(raw_features).permute(1, 0, 2, 3, 4, 5).contiguous()
        # print(gen_volumes.size())      # torch.Size([batch_size, n_views, 32, 32, 32])
        # print(raw_features.size())     # torch.Size([batch_size, n_views, 9, 32, 32, 32])
        return raw_features, gen_volumes

# Lightning model 
class GAN(pl.LightningModule):
    def __init__(
        self,
        channels,
        width,
        height,
        latent_dim: int = 100,
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        batch_size: int = 64,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        # networks
        # data_shape = (channels, width, height)
        # self.generator = Generator(latent_dim=self.hparams.latent_dim, img_shape=data_shape)
        # self.discriminator = Discriminator(img_shape=data_shape)

        self.validation_z = torch.randn(8, self.hparams.latent_dim)
        self.example_input_array = torch.zeros(2, self.hparams.latent_dim)

    def forward(self, imgs):
        return self.generator(imgs)

    # def adversarial_loss(self, y_hat, y):
    #     return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, vols = batch

        # # sample noise
        # z = torch.randn(imgs.shape[0], self.hparams.latent_dim)
        # z = z.type_as(imgs)

        # train generator
        if optimizer_idx == 0:
            # # generate images
            # self.generated_imgs = self(z)

            # # log sampled images
            # sample_imgs = self.generated_imgs[:6]
            # grid = torchvision.utils.make_grid(sample_imgs)
            # self.logger.experiment.add_image('generated_images', grid, 0)

            # # ground truth result (ie: all fake)
            # # put on GPU because we created this tensor inside training_loop
            # valid = torch.ones(imgs.size(0), 1)
            # valid = valid.type_as(imgs)

            # # adversarial loss is binary cross-entropy
            # g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
            # tqdm_dict = {'g_loss': g_loss}
            # output = OrderedDict({
            #     'loss': g_loss,
            #     'progress_bar': tqdm_dict,
            #     'log': tqdm_dict
            # })
            # return output

        # train discriminator
        if optimizer_idx == 1:
            # # Measure discriminator's ability to classify real from generated samples

            # # how well can it label as real?
            # valid = torch.ones(imgs.size(0), 1)
            # valid = valid.type_as(imgs)

            # real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

            # # how well can it label as fake?
            # fake = torch.zeros(imgs.size(0), 1)
            # fake = fake.type_as(imgs)

            # fake_loss = self.adversarial_loss(self.discriminator(self(z).detach()), fake)

            # # discriminator loss is the average of these
            # d_loss = (real_loss + fake_loss) / 2
            # tqdm_dict = {'d_loss': d_loss}
            # output = OrderedDict({
            #     'loss': d_loss,
            #     'progress_bar': tqdm_dict,
            #     'log': tqdm_dict
            # })
            # return output

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def on_epoch_end(self):
        # z = self.validation_z.type_as(self.generator.model[0].weight)

        # # log sampled images
        # sample_imgs = self(z)
        # grid = torchvision.utils.make_grid(sample_imgs)
        # self.logger.experiment.add_image('generated_images', grid, self.current_epoch)