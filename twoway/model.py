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
                 kernel_size=4,
                 stride=2,
                 padding=1,
                 bias=False
                 ):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels=source_channels,
                      out_channels=output_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      bias=bias),
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
                 kernel_size=4,
                 stride=2,
                 padding=1,
                 bias=False
                 ):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv3d(in_channels=source_channels,
                      out_channels=output_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      bias=bias),
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
                 scale_factor=2
                 ):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels=source_channels,
                      out_channels=output_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.Upsample(scale_factor=scale_factor,
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
                 scale_factor=2,
                 ):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv3d(in_channels=source_channels,
                      out_channels=output_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.Upsample(scale_factor=scale_factor,
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
        # 2D
        self.conv2d_0 = DoubleConv2d(source_channels, num_filters*2, kernel_size=3, stride=1) # NF x 2 x 256 x 256
        self.conv2d_1 = DoubleConv2d(num_filters*2,  num_filters*4)     # NF x 4 x 128 x 128
        self.conv2d_2 = DoubleConv2d(num_filters*4,  num_filters*8)     # NF x 8 x 64 x 64
        self.conv2d_3 = DoubleConv2d(num_filters*8,  num_filters*16)    # NF x 16 x 32 x 32
        self.conv2d_4 = DoubleConv2d(num_filters*16, num_filters*32)    # NF x 32 x 16 x 16
        self.conv2d_5 = DoubleConv2d(num_filters*32, num_filters*64)    # NF x 64 x 8 x 8
        self.conv2d_6 = DoubleConv2d(num_filters*64, num_filters*96)    # NF x 96 x 4 x 4

        # Transformation
        self.transformation = nn.Sequential(
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

            Reshape(num_filters*96, 1, 4, 4),  # NF x 96 x 1 x 4 x 4
            nn.Conv3d(num_filters*96,
                      num_filters*96,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.GroupNorm(4, num_filters*96),
            nn.LeakyReLU(inplace=True)
        )
        # 3D
        self.skip_6 = nn.Conv2d(num_filters*96, num_filters*96*1, kernel_size=1, stride=1, padding=0)
        self.skip_5 = nn.Conv2d(num_filters*64, num_filters*64*2, kernel_size=1, stride=1, padding=0)
        self.skip_4 = nn.Conv2d(num_filters*32, num_filters*48*4, kernel_size=1, stride=1, padding=0)
        self.skip_3 = nn.Conv2d(num_filters*16, num_filters*32*8, kernel_size=1, stride=1, padding=0)
        self.skip_2 = nn.Conv2d(num_filters*8, num_filters*16*16, kernel_size=1, stride=1, padding=0)
        self.skip_1 = nn.Conv2d(num_filters*4, num_filters*8*32, kernel_size=1, stride=1, padding=0)
        self.skip_0 = nn.Conv2d(num_filters*2, num_filters*4*64, kernel_size=1, stride=1, padding=0)

        self.deconv3d_6 = DoubleDeconv3d(num_filters*96, num_filters*64)
        self.deconv3d_5 = DoubleDeconv3d(num_filters*64, num_filters*48) 
        self.deconv3d_4 = DoubleDeconv3d(num_filters*48, num_filters*32)  
        self.deconv3d_3 = DoubleDeconv3d(num_filters*32, num_filters*16)  
        self.deconv3d_2 = DoubleDeconv3d(num_filters*16, num_filters*8)  
        self.deconv3d_1 = DoubleDeconv3d(num_filters*8,  num_filters*4)  
        self.deconv3d_0 = DoubleDeconv3d(num_filters*4,  num_filters*2, scale_factor=1)  

        self.output = nn.Sequential(
            nn.Conv3d(num_filters*2,
                      output_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.LeakyReLU(inplace=True),
            Squeeze(dim=1),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        conv2d_0 = self.conv2d_0(x * 2.0 - 1.0)
        conv2d_1 = self.conv2d_1(conv2d_0)
        conv2d_2 = self.conv2d_2(conv2d_1)
        conv2d_3 = self.conv2d_3(conv2d_2)
        conv2d_4 = self.conv2d_4(conv2d_3)
        conv2d_5 = self.conv2d_5(conv2d_4)
        conv2d_6 = self.conv2d_6(conv2d_5)

        transforms = self.transformation(conv2d_6)

        deconv3d_6 = self.deconv3d_6(transforms + self.skip_6(conv2d_6).view(transforms.shape))
        deconv3d_5 = self.deconv3d_5(deconv3d_6 + self.skip_5(conv2d_5).view(deconv3d_6.shape))
        deconv3d_4 = self.deconv3d_4(deconv3d_5 + self.skip_4(conv2d_4).view(deconv3d_5.shape))
        deconv3d_3 = self.deconv3d_3(deconv3d_4 + self.skip_3(conv2d_3).view(deconv3d_4.shape))
        deconv3d_2 = self.deconv3d_2(deconv3d_3 + self.skip_2(conv2d_2).view(deconv3d_3.shape))
        deconv3d_1 = self.deconv3d_1(deconv3d_2 + self.skip_1(conv2d_1).view(deconv3d_2.shape))
        deconv3d_0 = self.deconv3d_0(deconv3d_1 + self.skip_0(conv2d_0).view(deconv3d_1.shape))
        
        out = self.output(deconv3d_0) / 2.0 + 0.5
        # print('INet', x.shape, out.shape)
        return out


class PNet(nn.Module):
    def __init__(self, source_channels=1, output_channels=1, num_filters=32):
        super().__init__()
        # 3D
        self.unsqueeze = Unsqueeze(1)
        self.conv3d_0 = DoubleConv3d(source_channels, num_filters*2, kernel_size=3, stride=1) # NF x 2 x 256 x 256
        self.conv3d_1 = DoubleConv3d(num_filters*2,  num_filters*4)     # NF x 4 x 128 x 128
        self.conv3d_2 = DoubleConv3d(num_filters*4,  num_filters*8)     # NF x 8 x 64 x 64
        self.conv3d_3 = DoubleConv3d(num_filters*8,  num_filters*16)    # NF x 16 x 32 x 32
        self.conv3d_4 = DoubleConv3d(num_filters*16, num_filters*32)    # NF x 32 x 16 x 16
        self.conv3d_5 = DoubleConv3d(num_filters*32, num_filters*64)    # NF x 64 x 8 x 8
        self.conv3d_6 = DoubleConv3d(num_filters*64, num_filters*96)    # NF x 96 x 4 x 4

        # Transformation
        self.transformation = nn.Sequential(
            nn.Conv3d(num_filters*96,
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
            Reshape(num_filters*96, 4, 4), 
            
            nn.Conv2d(num_filters*96,
                      num_filters*96,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.GroupNorm(4, num_filters*96),
            nn.LeakyReLU(inplace=True),
        )

        self.skip_6 = nn.Conv2d(num_filters*96*1, num_filters*96, kernel_size=1, stride=1, padding=0)
        self.skip_5 = nn.Conv2d(num_filters*64*2, num_filters*64, kernel_size=1, stride=1, padding=0)
        self.skip_4 = nn.Conv2d(num_filters*32*4, num_filters*48, kernel_size=1, stride=1, padding=0)
        self.skip_3 = nn.Conv2d(num_filters*16*8, num_filters*32, kernel_size=1, stride=1, padding=0)
        self.skip_2 = nn.Conv2d(num_filters*8*16, num_filters*16, kernel_size=1, stride=1, padding=0)
        self.skip_1 = nn.Conv2d(num_filters*4*32, num_filters*8, kernel_size=1, stride=1, padding=0)
        self.skip_0 = nn.Conv2d(num_filters*2*64, num_filters*4, kernel_size=1, stride=1, padding=0)

        self.deconv2d_6 = DoubleDeconv2d(num_filters*96, num_filters*64)
        self.deconv2d_5 = DoubleDeconv2d(num_filters*64, num_filters*48) 
        self.deconv2d_4 = DoubleDeconv2d(num_filters*48, num_filters*32)  
        self.deconv2d_3 = DoubleDeconv2d(num_filters*32, num_filters*16)  
        self.deconv2d_2 = DoubleDeconv2d(num_filters*16, num_filters*8)  
        self.deconv2d_1 = DoubleDeconv2d(num_filters*8,  num_filters*4)  
        self.deconv2d_0 = DoubleDeconv2d(num_filters*4,  num_filters*2, scale_factor=1)  

        # # 2D
        self.output = nn.Sequential(
            nn.Conv2d(num_filters*2,
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
        x = self.unsqueeze(x)
        conv3d_0 = self.conv3d_0(x * 2.0 - 1.0)
        conv3d_1 = self.conv3d_1(conv3d_0)
        conv3d_2 = self.conv3d_2(conv3d_1)
        conv3d_3 = self.conv3d_3(conv3d_2)
        conv3d_4 = self.conv3d_4(conv3d_3)
        conv3d_5 = self.conv3d_5(conv3d_4)
        conv3d_6 = self.conv3d_6(conv3d_5)

        transforms = self.transformation(conv3d_6)
        
        deconv2d_6 = self.deconv2d_6(transforms + self.skip_6( conv3d_6.view(conv3d_6.shape[0], conv3d_6.shape[1]*conv3d_6.shape[2], conv3d_6.shape[3], conv3d_6.shape[4]) ) )
        deconv2d_5 = self.deconv2d_5(deconv2d_6 + self.skip_5( conv3d_5.view(conv3d_5.shape[0], conv3d_5.shape[1]*conv3d_5.shape[2], conv3d_5.shape[3], conv3d_5.shape[4]) ) )
        deconv2d_4 = self.deconv2d_4(deconv2d_5 + self.skip_4( conv3d_4.view(conv3d_4.shape[0], conv3d_4.shape[1]*conv3d_4.shape[2], conv3d_4.shape[3], conv3d_4.shape[4]) ) )
        deconv2d_3 = self.deconv2d_3(deconv2d_4 + self.skip_3( conv3d_3.view(conv3d_3.shape[0], conv3d_3.shape[1]*conv3d_3.shape[2], conv3d_3.shape[3], conv3d_3.shape[4]) ) )
        deconv2d_2 = self.deconv2d_2(deconv2d_3 + self.skip_2( conv3d_2.view(conv3d_2.shape[0], conv3d_2.shape[1]*conv3d_2.shape[2], conv3d_2.shape[3], conv3d_2.shape[4]) ) )
        deconv2d_1 = self.deconv2d_1(deconv2d_2 + self.skip_1( conv3d_1.view(conv3d_1.shape[0], conv3d_1.shape[1]*conv3d_1.shape[2], conv3d_1.shape[3], conv3d_1.shape[4]) ) )
        deconv2d_0 = self.deconv2d_0(deconv2d_1 + self.skip_0( conv3d_0.view(conv3d_0.shape[0], conv3d_0.shape[1]*conv3d_0.shape[2], conv3d_0.shape[3], conv3d_0.shape[4]) ) )

        out = self.output(deconv2d_0) / 2.0 + 0.5
        # print('PNet', x.shape, out.shape)
        return out
