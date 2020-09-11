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
        # print(ret.shape)
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
        # print(ret.shape)
        return ret


class DoubleDeconv3d(nn.Module):
    def __init__(self,
                 source_channels=32,
                 output_channels=32,
                 number_grouping=4,
                 scale_factor=2,
                 ):
        super().__init__()
        # print(scale_factor)
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
        # print(ret.shape)
        return ret


class Skip3d(nn.Sequential):
    def __init__(self, 
        source_channels=32,
        output_channels=32,
        number_grouping=4,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False, 
    ):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels=source_channels,
                                out_channels=output_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                bias=bias) 

    def forward(self, x):
        shape = x.shape
        y = self.conv2d(x)
        z = y.view([shape[0], shape[1], -1, shape[2], shape[3]]).transpose(2, 3) 
        return z


class Skip2d(nn.Sequential):
    def __init__(self, 
        source_channels=32,
        output_channels=32,
        number_grouping=4,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False, 
    ):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels=source_channels,
                                out_channels=output_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                bias=bias) 

    def forward(self, x):
        y = x.transpose(2, 3)
        shape = y.shape
        z = y.reshape([shape[0], shape[1]*shape[2], shape[3], shape[4]]) 
        z = self.conv2d(z)
        return z


class INet(nn.Module):
    def __init__(self, 
        source_channels=1, 
        output_channels=1, 
        num_filters=32, 
        num_factors=[1, 2, 4, 8, 16, 24, 32, 48, 64]):
        super().__init__()
        # 2D
        self.conv2d_0 = DoubleConv2d(source_channels, num_filters*num_factors[0], kernel_size=3, stride=1)  # NF x 1 x 256 x 256
        self.conv2d_1 = DoubleConv2d(num_filters*num_factors[0],  num_filters*num_factors[1])               # NF x 2 x 128 x 128
        self.conv2d_2 = DoubleConv2d(num_filters*num_factors[1],  num_filters*num_factors[2])               # NF x 4 x 64 x 64
        self.conv2d_3 = DoubleConv2d(num_filters*num_factors[2],  num_filters*num_factors[3])               # NF x 8 x 32 x 32
        self.conv2d_4 = DoubleConv2d(num_filters*num_factors[3],  num_filters*num_factors[4])               # NF x 16 x 16 x 16
        self.conv2d_5 = DoubleConv2d(num_filters*num_factors[4],  num_filters*num_factors[5])               # NF x 24 x 8 x 8
        self.conv2d_6 = DoubleConv2d(num_filters*num_factors[5],  num_filters*num_factors[6])               # NF x 32 x 4 x 4
        self.conv2d_7 = DoubleConv2d(num_filters*num_factors[6],  num_filters*num_factors[7])               # NF x 48 x 2 x 2
        self.conv2d_8 = DoubleConv2d(num_filters*num_factors[7],  num_filters*num_factors[8])               # NF x 64 x 1 x 1

        # Transformation
        self.transformation = nn.Sequential(
            nn.Conv2d(num_filters*num_factors[8],
                      num_filters*num_factors[8],
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.GroupNorm(4, num_filters*num_factors[8]),
            nn.Flatten(),
            nn.LeakyReLU(inplace=True),
            nn.Linear(num_filters*num_factors[8], num_filters*num_factors[8]),
            nn.LeakyReLU(inplace=True),
            nn.Linear(num_filters*num_factors[8], num_filters*num_factors[8]),
            nn.LeakyReLU(inplace=True),
            nn.Linear(num_filters*num_factors[8], num_filters*num_factors[8]),
            nn.LeakyReLU(inplace=True),
            nn.Linear(num_filters*num_factors[8], num_filters*num_factors[8]),
            nn.LeakyReLU(inplace=True),

            Reshape(num_filters*num_factors[8], 1, 1, 1),  # NF x 64 x 1 x 4 x 4
            nn.Conv3d(num_filters*num_factors[8],
                      num_filters*num_factors[8],
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.GroupNorm(4, num_filters*num_factors[8]),
            nn.LeakyReLU(inplace=True)
        )
        # 3D
        self.skip_0 = Skip3d(source_channels=num_filters*num_factors[0], output_channels=num_filters*num_factors[0]*64)
        self.skip_1 = Skip3d(source_channels=num_filters*num_factors[1], output_channels=num_filters*num_factors[1]*32)
        self.skip_2 = Skip3d(source_channels=num_filters*num_factors[2], output_channels=num_filters*num_factors[2]*16)
        self.skip_3 = Skip3d(source_channels=num_filters*num_factors[3], output_channels=num_filters*num_factors[3]*8 )
        self.skip_4 = Skip3d(source_channels=num_filters*num_factors[4], output_channels=num_filters*num_factors[4]*4 )
        self.skip_5 = Skip3d(source_channels=num_filters*num_factors[5], output_channels=num_filters*num_factors[5]*2 )
        self.skip_6 = Skip3d(source_channels=num_filters*num_factors[6], output_channels=num_filters*num_factors[6]*1 )
        self.skip_7 = Skip3d(source_channels=num_filters*num_factors[7], output_channels=num_filters*num_factors[7]*1 )
        self.skip_8 = Skip3d(source_channels=num_filters*num_factors[8], output_channels=num_filters*num_factors[8]*1 )

        self.deconv3d_8 = DoubleDeconv3d(num_filters*num_factors[8],  num_filters*num_factors[7], scale_factor=(2, 1, 2))
        self.deconv3d_7 = DoubleDeconv3d(num_filters*num_factors[7],  num_filters*num_factors[6], scale_factor=(2, 1, 2)) 
        self.deconv3d_6 = DoubleDeconv3d(num_filters*num_factors[6],  num_filters*num_factors[5])  
        self.deconv3d_5 = DoubleDeconv3d(num_filters*num_factors[5],  num_filters*num_factors[4])  
        self.deconv3d_4 = DoubleDeconv3d(num_filters*num_factors[4],  num_filters*num_factors[3])  
        self.deconv3d_3 = DoubleDeconv3d(num_filters*num_factors[3],  num_filters*num_factors[2])  
        self.deconv3d_2 = DoubleDeconv3d(num_filters*num_factors[2],  num_filters*num_factors[1])  
        self.deconv3d_1 = DoubleDeconv3d(num_filters*num_factors[1],  num_filters*num_factors[0])  
        self.deconv3d_0 = DoubleDeconv3d(num_filters*num_factors[0],  num_filters*1, scale_factor=1)  

        self.output = nn.Sequential(
            nn.Conv3d(num_filters*1,
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
        conv2d_7 = self.conv2d_7(conv2d_6)
        conv2d_8 = self.conv2d_8(conv2d_7)

        transforms = self.transformation(conv2d_8)

        deconv3d_8 = self.deconv3d_8(transforms + self.skip_8(conv2d_8))
        deconv3d_7 = self.deconv3d_7(deconv3d_8 + self.skip_7(conv2d_7))
        deconv3d_6 = self.deconv3d_6(deconv3d_7 + self.skip_6(conv2d_6))
        deconv3d_5 = self.deconv3d_5(deconv3d_6 + self.skip_5(conv2d_5))
        deconv3d_4 = self.deconv3d_4(deconv3d_5 + self.skip_4(conv2d_4))
        deconv3d_3 = self.deconv3d_3(deconv3d_4 + self.skip_3(conv2d_3))
        deconv3d_2 = self.deconv3d_2(deconv3d_3 + self.skip_2(conv2d_2))
        deconv3d_1 = self.deconv3d_1(deconv3d_2 + self.skip_1(conv2d_1))
        deconv3d_0 = self.deconv3d_0(deconv3d_1 + self.skip_0(conv2d_0))
        
        deconv3d_0 = deconv3d_0.transpose(2, 3)
        out = self.output(deconv3d_0) / 2.0 + 0.5
        # print('INet', out.shape)
        return out


class PNet(nn.Module):
    def __init__(self, source_channels=1, 
        output_channels=1, 
        num_filters=32, 
        num_factors=[1, 2, 4, 8, 16, 24, 32, 48, 64]):
        super().__init__()
        # 2D
        self.conv3d_0 = DoubleConv3d(source_channels, num_filters*num_factors[0], kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))         # NF x 1 x 256 x 256
        self.conv3d_1 = DoubleConv3d(num_filters*num_factors[0], num_filters*num_factors[1], kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))               # NF x 2 x 128 x 128
        self.conv3d_2 = DoubleConv3d(num_filters*num_factors[1], num_filters*num_factors[2], kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))               # NF x 4 x 64 x 64
        self.conv3d_3 = DoubleConv3d(num_filters*num_factors[2], num_filters*num_factors[3], kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))               # NF x 8 x 32 x 32
        self.conv3d_4 = DoubleConv3d(num_filters*num_factors[3], num_filters*num_factors[4], kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))               # NF x 16 x 16 x 16
        self.conv3d_5 = DoubleConv3d(num_filters*num_factors[4], num_filters*num_factors[5], kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))               # NF x 24 x 8 x 8
        self.conv3d_6 = DoubleConv3d(num_filters*num_factors[5], num_filters*num_factors[6], kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1))               # NF x 32 x 4 x 4
        self.conv3d_7 = DoubleConv3d(num_filters*num_factors[6], num_filters*num_factors[7], kernel_size=(4, 1, 4), stride=(2, 1, 2), padding=(1, 0, 1))               # NF x 48 x 2 x 2
        self.conv3d_8 = DoubleConv3d(num_filters*num_factors[7], num_filters*num_factors[8], kernel_size=(2, 1, 2), stride=(1, 1, 1), padding=(0, 0, 0))               # NF x 64 x 1 x 1

        # Transformation
        self.transformation = nn.Sequential(
            nn.Conv3d(num_filters*num_factors[8],
                      num_filters*num_factors[8],
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.GroupNorm(4, num_filters*num_factors[8]),
            nn.Flatten(),
            nn.LeakyReLU(inplace=True),
            nn.Linear(num_filters*num_factors[8], num_filters*num_factors[8]),
            nn.LeakyReLU(inplace=True),
            nn.Linear(num_filters*num_factors[8], num_filters*num_factors[8]),
            nn.LeakyReLU(inplace=True),
            nn.Linear(num_filters*num_factors[8], num_filters*num_factors[8]),
            nn.LeakyReLU(inplace=True),
            nn.Linear(num_filters*num_factors[8], num_filters*num_factors[8]),
            nn.LeakyReLU(inplace=True),

            Reshape(num_filters*num_factors[8], 1, 1),  # NF x 64 x 1 x 4 x 4
            nn.Conv2d(num_filters*num_factors[8],
                      num_filters*num_factors[8],
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.GroupNorm(4, num_filters*num_factors[8]),
            nn.LeakyReLU(inplace=True)
        )
        # 2D
        self.skip_0 = Skip2d(source_channels=num_filters*num_factors[0]*64, output_channels=num_filters*num_factors[0])
        self.skip_1 = Skip2d(source_channels=num_filters*num_factors[1]*32, output_channels=num_filters*num_factors[1])
        self.skip_2 = Skip2d(source_channels=num_filters*num_factors[2]*16, output_channels=num_filters*num_factors[2])
        self.skip_3 = Skip2d(source_channels=num_filters*num_factors[3]*8 , output_channels=num_filters*num_factors[3])
        self.skip_4 = Skip2d(source_channels=num_filters*num_factors[4]*4 , output_channels=num_filters*num_factors[4])
        self.skip_5 = Skip2d(source_channels=num_filters*num_factors[5]*2 , output_channels=num_filters*num_factors[5])
        self.skip_6 = Skip2d(source_channels=num_filters*num_factors[6]*1 , output_channels=num_filters*num_factors[6])
        self.skip_7 = Skip2d(source_channels=num_filters*num_factors[7]*1 , output_channels=num_filters*num_factors[7])
        self.skip_8 = Skip2d(source_channels=num_filters*num_factors[8]*1 , output_channels=num_filters*num_factors[8])

        self.deconv2d_8 = DoubleDeconv2d(num_filters*num_factors[8],  num_filters*num_factors[7])
        self.deconv2d_7 = DoubleDeconv2d(num_filters*num_factors[7],  num_filters*num_factors[6]) 
        self.deconv2d_6 = DoubleDeconv2d(num_filters*num_factors[6],  num_filters*num_factors[5])  
        self.deconv2d_5 = DoubleDeconv2d(num_filters*num_factors[5],  num_filters*num_factors[4])  
        self.deconv2d_4 = DoubleDeconv2d(num_filters*num_factors[4],  num_filters*num_factors[3])  
        self.deconv2d_3 = DoubleDeconv2d(num_filters*num_factors[3],  num_filters*num_factors[2])  
        self.deconv2d_2 = DoubleDeconv2d(num_filters*num_factors[2],  num_filters*num_factors[1])  
        self.deconv2d_1 = DoubleDeconv2d(num_filters*num_factors[1],  num_filters*num_factors[0])  
        self.deconv2d_0 = DoubleDeconv2d(num_filters*num_factors[0],  num_filters*1, scale_factor=1)  

        # # 2D
        self.output = nn.Sequential(
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
        # out = x.mean(dim=1, keepdim=True)
        y = x.unsqueeze(1).transpose(2, 3)
        conv3d_0 = self.conv3d_0(y * 2.0 - 1.0)
        conv3d_1 = self.conv3d_1(conv3d_0)
        conv3d_2 = self.conv3d_2(conv3d_1)
        conv3d_3 = self.conv3d_3(conv3d_2)
        conv3d_4 = self.conv3d_4(conv3d_3)
        conv3d_5 = self.conv3d_5(conv3d_4)
        conv3d_6 = self.conv3d_6(conv3d_5)
        conv3d_7 = self.conv3d_7(conv3d_6)
        conv3d_8 = self.conv3d_8(conv3d_7)

        transforms = self.transformation(conv3d_8)
        
        deconv2d_8 = self.deconv2d_8(transforms + self.skip_8(conv3d_8))
        deconv2d_7 = self.deconv2d_7(deconv2d_8 + self.skip_7(conv3d_7))
        deconv2d_6 = self.deconv2d_6(deconv2d_7 + self.skip_6(conv3d_6))
        deconv2d_5 = self.deconv2d_5(deconv2d_6 + self.skip_5(conv3d_5))
        deconv2d_4 = self.deconv2d_4(deconv2d_5 + self.skip_4(conv3d_4))
        deconv2d_3 = self.deconv2d_3(deconv2d_4 + self.skip_3(conv3d_3))
        deconv2d_2 = self.deconv2d_2(deconv2d_3 + self.skip_2(conv3d_2))
        deconv2d_1 = self.deconv2d_1(deconv2d_2 + self.skip_1(conv3d_1))
        deconv2d_0 = self.deconv2d_0(deconv2d_1 + self.skip_0(conv3d_0))
           
        out = self.output(deconv2d_0) / 2.0 + 0.5
        # print('PNet', out.shape)
        return out
