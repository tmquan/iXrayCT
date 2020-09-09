import torch
import torch.nn as nn


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


class DoubleConv2d(nn.Module):
    def __init__(self,
                 source_channels=32,
                 output_channels=32,
                 number_grouping=4,
                 ):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels=source_channels,
                      out_channels=output_channels,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.GroupNorm(num_groups=number_grouping,
                         num_channels=output_channels),
            nn.LeakyReLU(inplace=True),
        )
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=output_channels,
                      out_channels=output_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.GroupNorm(num_groups=number_grouping,
                         num_channels=output_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        tmp = self.pre(x)
        ret = self.net(tmp) + tmp
        return ret


class DoubleConv3d(nn.Module):
    def __init__(self,
                 source_channels=32,
                 output_channels=32,
                 number_grouping=4,
                 ):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv3d(in_channels=source_channels,
                      out_channels=output_channels,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.GroupNorm(num_groups=number_grouping,
                         num_channels=output_channels),
            nn.LeakyReLU(inplace=True),
        )
        self.net = nn.Sequential(
            nn.Conv3d(in_channels=output_channels,
                      out_channels=output_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.GroupNorm(num_groups=number_grouping,
                         num_channels=output_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        tmp = self.pre(x)
        ret = self.net(tmp) + tmp
        return ret


class DoubleDeconv2d(nn.Module):
    def __init__(self,
                 source_channels=32,
                 output_channels=32,
                 number_grouping=4,
                 ):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels=source_channels,
                      out_channels=output_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.Upsample(scale_factor=2,
                        mode='bilinear',
                        align_corners=True),
            nn.GroupNorm(num_groups=number_grouping,
                         num_channels=output_channels),
            nn.LeakyReLU(inplace=True),
        )
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=output_channels,
                      out_channels=output_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.GroupNorm(num_groups=number_grouping,
                         num_channels=output_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        tmp = self.pre(x)
        ret = self.net(tmp) + tmp
        return ret


class DoubleDeconv3d(nn.Module):
    def __init__(self,
                 source_channels=32,
                 output_channels=32,
                 number_grouping=4,
                 ):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv3d(in_channels=source_channels,
                      out_channels=output_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.Upsample(scale_factor=2,
                        mode='trilinear',
                        align_corners=True),
            nn.GroupNorm(num_groups=number_grouping,
                         num_channels=output_channels),
            nn.LeakyReLU(inplace=True),
        )
        self.net = nn.Sequential(
            nn.Conv3d(in_channels=output_channels,
                      out_channels=output_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.GroupNorm(num_groups=number_grouping,
                         num_channels=output_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        tmp = self.pre(x)
        ret = self.net(tmp) + tmp
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
            DoubleConv2d(num_filters*64, num_filters*96),   # NF x 96 x 4 x 4
            
            # Transformation
            nn.Conv2d(num_filters*96, 
                      num_filters*96, 
                      kernel_size=1, 
                      stride=1, 
                      padding=0, 
                      bias=False),
            nn.GroupNorm(4, num_filters*96),
            nn.Flatten(),
            nn.LeakyReLU(inplace=True),
            nn.Linear(num_filters*1536, num_filters*192),
            nn.LeakyReLU(inplace=True),
            nn.Linear(num_filters*192, num_filters*24),
            nn.LeakyReLU(inplace=True),
            nn.Linear(num_filters*24, num_filters*192),
            nn.LeakyReLU(inplace=True),
            nn.Linear(num_filters*192, num_filters*1536),
            nn.LeakyReLU(inplace=True),
        )
        self.dec = nn.Sequential(
            Reshape(num_filters*96, 1, 4, 4), # NF x 96 x 1 x 4 x 4
            nn.Conv3d(num_filters*96, 
                      num_filters*96, 
                      kernel_size=1, 
                      stride=1, 
                      padding=0, 
                      bias=False),
            nn.GroupNorm(4, num_filters*96),
            nn.LeakyReLU(inplace=True),

            # 3D
            DoubleDeconv3d(num_filters*96, num_filters*64), # NF x 64 x 2 x 8 x 8
            DoubleDeconv3d(num_filters*64, num_filters*32), # NF x 32 x 4 x 16 x 16
            DoubleDeconv3d(num_filters*32, num_filters*16), # NF x 16 x 8 x 32 x 32
            DoubleDeconv3d(num_filters*16, num_filters*8),  # NF x 8 x 16 x 64 x 64
            DoubleDeconv3d(num_filters*8, num_filters*4),   # NF x 4 x 32 x 128 x 128
            DoubleDeconv3d(num_filters*4, num_filters*1),   # NF x 1 x 64 x 256 x 256
            nn.Conv3d(num_filters*1, 
                      output_channels, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1, 
                      bias=False),
            nn.LeakyReLU(inplace=True),
            Squeeze(dim=1),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )
    
    def forward(self, x):
        feature = self.enc(x * 2.0 - 1.0) 
        outputs = self.dec(feature)/ 2.0 + 0.5
        return outputs


class PNet(nn.Module):
    def __init__(self, source_channels=1, output_channels=1, num_filters=32):
        super().__init__()
        self.enc = nn.Sequential(
            # 3D
            Unsqueeze(1),
            DoubleConv3d(1, num_filters*2),                 # NF x 2 x 32 x 128 x 128
            DoubleConv3d(num_filters*2, num_filters*4),     # NF x 4 x 16 x 64 x 64
            DoubleConv3d(num_filters*4, num_filters*8),     # NF x 8 x 8 x 32 x 32
            DoubleConv3d(num_filters*8, num_filters*16),    # NF x 16 x 4 x 16 x 16
            DoubleConv3d(num_filters*16, num_filters*32),   # NF x 32 x 2 x 8 x 8
            DoubleConv3d(num_filters*32, num_filters*64),   # NF x 64 x 1 x 4 x 4
            
            # Transformation
            nn.Conv3d(num_filters*64, 
                      num_filters*96, 
                      kernel_size=1, 
                      stride=1, 
                      padding=0, 
                      bias=False),
            nn.GroupNorm(4, num_filters*96),
            nn.LeakyReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(num_filters*1536, num_filters*192),
            nn.LeakyReLU(inplace=True),
            nn.Linear(num_filters*192, num_filters*24),
            nn.LeakyReLU(inplace=True),
            nn.Linear(num_filters*24, num_filters*192),
            nn.LeakyReLU(inplace=True),
            nn.Linear(num_filters*192, num_filters*1536),
            nn.LeakyReLU(inplace=True),
            Reshape(num_filters*96, 4, 4), # NF x 96 x 4 x 4
        )
        self.dec = nn.Sequential(
            nn.Conv2d(num_filters*96, 
                      num_filters*96, 
                      kernel_size=1, 
                      stride=1, 
                      padding=0, 
                      bias=False),
            nn.GroupNorm(4, num_filters*96),
            nn.LeakyReLU(inplace=True),
            
            # 2D
            DoubleDeconv2d(num_filters*96, num_filters*64), # NF x 64 x 8 x 8
            DoubleDeconv2d(num_filters*64, num_filters*32), # NF x 32 x 16 x 16
            DoubleDeconv2d(num_filters*32, num_filters*16), # NF x 16 x 32 x 32
            DoubleDeconv2d(num_filters*16, num_filters*8),  # NF x 8 x 64 x 64
            DoubleDeconv2d(num_filters*8, num_filters*4),   # NF x 4 x 128 x 128
            DoubleDeconv2d(num_filters*4, num_filters*1),   # NF x 1 x 256 x 256
            nn.Conv2d(num_filters*1, 
                      num_filters*1, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1, 
                      bias=False),
            nn.GroupNorm(4, num_filters*1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(num_filters*1, output_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        feature = self.enc(x * 2.0 - 1.0) 
        outputs = self.dec(feature)/ 2.0 + 0.5
        return outputs
