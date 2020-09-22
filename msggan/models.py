import math
import random
import functools
import operator

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

class EqualConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channels * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )


class EqualConv3d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1], kernel_size[2])
        )
        self.scale = 1 / math.sqrt(in_channels * kernel_size[0] * kernel_size[1] * kernel_size[2])

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv3d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )


class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            # import pdb
            # pdb.set_trace()
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)



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


class InjectNoises(nn.Module):
    def __init__(self, is_injecting):
        super().__init__()
        self.is_injecting = is_injecting

    def forward(self, x):
        if self.is_injecting:
            return x*(1+torch.randn_like(x)) + torch.randn_like(x)
        else:
            return x #torch.zeros_like(x)



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
            EqualConv2d(in_channels=source_channels,
                      out_channels=output_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      bias=bias),
            ScaledLeakyReLU(),
        )
        self.net = nn.Sequential(
            EqualConv2d(in_channels=output_channels,
                      out_channels=output_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            ScaledLeakyReLU(),
        )

    def forward(self, x):
        tmp = self.pre(x)
        ret = self.net(tmp) + tmp
        return ret


class DoubleConv3d(nn.Module):
    def __init__(self,
                 source_channels=32,
                 output_channels=32,
                 kernel_size=[4, 4, 4],
                 stride=2,
                 padding=1,
                 number_grouping=4,
                 bias=False
                 ):
        super().__init__()
        self.pre = nn.Sequential(
            EqualConv3d(in_channels=source_channels,
                      out_channels=output_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      bias=bias),
            ScaledLeakyReLU(),
        )
        self.net = nn.Sequential(
            EqualConv3d(in_channels=output_channels,
                      out_channels=output_channels,
                      kernel_size=[3, 3, 3],
                      stride=1,
                      padding=1,
                      bias=False),
            ScaledLeakyReLU(),
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
            nn.Upsample(scale_factor=scale_factor,
                        mode='bilinear',
                        align_corners=True),
            EqualConv2d(in_channels=source_channels,
                      out_channels=output_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            ScaledLeakyReLU(),
        )
        self.net = nn.Sequential(
            EqualConv2d(in_channels=output_channels,
                      out_channels=output_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            ScaledLeakyReLU(),
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
        self.conv2d = EqualConv2d(in_channels=source_channels,
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
        is_injecting=False,
    ):
        super().__init__()
        self.conv2d = EqualConv2d(in_channels=source_channels,
                                out_channels=output_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                bias=bias) 
        self.is_injecting = is_injecting
        self.injectnoises = InjectNoises(self.is_injecting)
    def forward(self, x):
        y = x.transpose(2, 3)
        shape = y.shape
        z = y.reshape([shape[0], shape[1]*shape[2], shape[3], shape[4]]) 
        z = self.injectnoises(self.conv2d(z)) if self.is_injecting else self.conv2d(z)
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
            EqualConv2d(num_filters*num_factors[8],
                      num_filters*num_factors[8],
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.Flatten(),
            ScaledLeakyReLU(),
            EqualLinear(num_filters*num_factors[8], num_filters*num_factors[8]),
            ScaledLeakyReLU(),
            EqualLinear(num_filters*num_factors[8], num_filters*num_factors[8]),
            ScaledLeakyReLU(),
            EqualLinear(num_filters*num_factors[8], num_filters*num_factors[8]),
            ScaledLeakyReLU(),
            EqualLinear(num_filters*num_factors[8], num_filters*num_factors[8]),
            ScaledLeakyReLU(),
            # nn.Dropout(),
            EqualLinear(num_filters*num_factors[8], num_filters*num_factors[8]),
            ScaledLeakyReLU(),
            EqualLinear(num_filters*num_factors[8], num_filters*num_factors[8]),
            ScaledLeakyReLU(),
            EqualLinear(num_filters*num_factors[8], num_filters*num_factors[8]),
            ScaledLeakyReLU(),
            EqualLinear(num_filters*num_factors[8], num_filters*num_factors[8]),
            ScaledLeakyReLU(),


            Reshape(num_filters*num_factors[8], 1, 1, 1),  # NF x 64 x 1 x 4 x 4
            EqualConv3d(num_filters*num_factors[8],
                      num_filters*num_factors[8],
                      kernel_size=[1, 1, 1],
                      stride=1,
                      padding=0,
                      bias=False),
            ScaledLeakyReLU()
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
            EqualConv3d(num_filters*1,
                      output_channels,
                      kernel_size=[3, 3, 3],
                      stride=1,
                      padding=1,
                      bias=False),
            ScaledLeakyReLU(),
            Squeeze(dim=1),
            EqualConv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.Tanh(),
        )

    def forward(self, x):
        conv2d_0 = self.conv2d_0(x)
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
        out = self.output(deconv3d_0)
        # print('INet', out.shape)
        return out


class PGNet(nn.Module):
    def __init__(self, source_channels=1, 
        output_channels=1, 
        num_filters=16, 
        num_factors=[1, 2, 4, 8, 16, 24, 32, 48, 64],
        is_skipping=False,
        is_injecting=False):
        super().__init__()
        self.is_skipping = is_skipping
        self.is_injecting = is_injecting
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
            EqualConv3d(num_filters*num_factors[8],
                      num_filters*num_factors[8],
                      kernel_size=[1, 1, 1],
                      stride=1,
                      padding=0,
                      bias=False),
            nn.Flatten(),
            ScaledLeakyReLU(),
            EqualLinear(num_filters*num_factors[8], num_filters*num_factors[8]),
            ScaledLeakyReLU(),
            EqualLinear(num_filters*num_factors[8], num_filters*num_factors[8]),
            ScaledLeakyReLU(),
            EqualLinear(num_filters*num_factors[8], num_filters*num_factors[8]),
            ScaledLeakyReLU(),
            EqualLinear(num_filters*num_factors[8], num_filters*num_factors[8]),
            ScaledLeakyReLU(),

            EqualLinear(num_filters*num_factors[8], num_filters*num_factors[8]),
            ScaledLeakyReLU(),
            EqualLinear(num_filters*num_factors[8], num_filters*num_factors[8]),
            ScaledLeakyReLU(),
            EqualLinear(num_filters*num_factors[8], num_filters*num_factors[8]),
            ScaledLeakyReLU(),
            EqualLinear(num_filters*num_factors[8], num_filters*num_factors[8]),
            ScaledLeakyReLU(),

            Reshape(num_filters*num_factors[8], 1, 1),  # NF x 64 x 1 x 4 x 4
            EqualConv2d(num_filters*num_factors[8],
                      num_filters*num_factors[8],
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            ScaledLeakyReLU()
        )
        
        # 2D
        self.skip_0 = Skip2d(source_channels=num_filters*num_factors[0]*64, output_channels=num_filters*num_factors[0], is_injecting=self.is_injecting)
        self.skip_1 = Skip2d(source_channels=num_filters*num_factors[1]*32, output_channels=num_filters*num_factors[1], is_injecting=self.is_injecting)
        self.skip_2 = Skip2d(source_channels=num_filters*num_factors[2]*16, output_channels=num_filters*num_factors[2], is_injecting=self.is_injecting)
        self.skip_3 = Skip2d(source_channels=num_filters*num_factors[3]*8 , output_channels=num_filters*num_factors[3], is_injecting=self.is_injecting)
        self.skip_4 = Skip2d(source_channels=num_filters*num_factors[4]*4 , output_channels=num_filters*num_factors[4], is_injecting=self.is_injecting)
        self.skip_5 = Skip2d(source_channels=num_filters*num_factors[5]*2 , output_channels=num_filters*num_factors[5], is_injecting=self.is_injecting)
        self.skip_6 = Skip2d(source_channels=num_filters*num_factors[6]*1 , output_channels=num_filters*num_factors[6], is_injecting=self.is_injecting)
        self.skip_7 = Skip2d(source_channels=num_filters*num_factors[7]*1 , output_channels=num_filters*num_factors[7], is_injecting=self.is_injecting)
        self.skip_8 = Skip2d(source_channels=num_filters*num_factors[8]*1 , output_channels=num_filters*num_factors[8], is_injecting=self.is_injecting)

        self.deconv2d_8 = DoubleDeconv2d(num_filters*num_factors[8],  num_filters*num_factors[7])
        self.deconv2d_7 = DoubleDeconv2d(num_filters*num_factors[7],  num_filters*num_factors[6]) 
        self.deconv2d_6 = DoubleDeconv2d(num_filters*num_factors[6],  num_filters*num_factors[5])  
        self.deconv2d_5 = DoubleDeconv2d(num_filters*num_factors[5],  num_filters*num_factors[4])  
        self.deconv2d_4 = DoubleDeconv2d(num_filters*num_factors[4],  num_filters*num_factors[3])  
        self.deconv2d_3 = DoubleDeconv2d(num_filters*num_factors[3],  num_filters*num_factors[2])  
        self.deconv2d_2 = DoubleDeconv2d(num_filters*num_factors[2],  num_filters*num_factors[1])  
        self.deconv2d_1 = DoubleDeconv2d(num_filters*num_factors[1],  num_filters*num_factors[0])  
        self.deconv2d_0 = DoubleDeconv2d(num_filters*num_factors[0],  num_filters*1, scale_factor=1)  

        self.scale_8 = EqualConv2d(num_filters*num_factors[8], 1, 1)
        self.scale_7 = EqualConv2d(num_filters*num_factors[7], 1, 1)
        self.scale_6 = EqualConv2d(num_filters*num_factors[6], 1, 1)
        self.scale_5 = EqualConv2d(num_filters*num_factors[5], 1, 1)
        self.scale_4 = EqualConv2d(num_filters*num_factors[4], 1, 1)
        self.scale_3 = EqualConv2d(num_filters*num_factors[3], 1, 1)
        self.scale_2 = EqualConv2d(num_filters*num_factors[2], 1, 1)
        self.scale_1 = EqualConv2d(num_filters*num_factors[1], 1, 1)
        self.scale_0 = EqualConv2d(num_filters*num_factors[0], 1, 1)


        # # 2D
        self.output = nn.Sequential(
            EqualConv2d(num_filters*1,
                      num_filters*1,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            ScaledLeakyReLU(),
            EqualConv2d(num_filters*1, output_channels, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.Tanh(),
        )

    def forward(self, x):
        # out = x.mean(dim=1, keepdim=True)
        y = x.unsqueeze(1).transpose(2, 3)
        conv3d_0 = self.conv3d_0(y)
        conv3d_1 = self.conv3d_1(conv3d_0)
        conv3d_2 = self.conv3d_2(conv3d_1)
        conv3d_3 = self.conv3d_3(conv3d_2)
        conv3d_4 = self.conv3d_4(conv3d_3)
        conv3d_5 = self.conv3d_5(conv3d_4)
        conv3d_6 = self.conv3d_6(conv3d_5)
        conv3d_7 = self.conv3d_7(conv3d_6)
        conv3d_8 = self.conv3d_8(conv3d_7)

        transforms = self.transformation(conv3d_8)
        
        if self.is_skipping:
            deconv2d_8 = self.deconv2d_8(transforms + self.skip_8(conv3d_8))
            deconv2d_7 = self.deconv2d_7(deconv2d_8 + self.skip_7(conv3d_7))
            deconv2d_6 = self.deconv2d_6(deconv2d_7 + self.skip_6(conv3d_6))
            deconv2d_5 = self.deconv2d_5(deconv2d_6 + self.skip_5(conv3d_5))
            deconv2d_4 = self.deconv2d_4(deconv2d_5 + self.skip_4(conv3d_4))
            deconv2d_3 = self.deconv2d_3(deconv2d_4 + self.skip_3(conv3d_3))
            deconv2d_2 = self.deconv2d_2(deconv2d_3 + self.skip_2(conv3d_2))
            deconv2d_1 = self.deconv2d_1(deconv2d_2 + self.skip_1(conv3d_1))
            deconv2d_0 = self.deconv2d_0(deconv2d_1 + self.skip_0(conv3d_0))
        else:
            deconv2d_8 = self.deconv2d_8(transforms)
            deconv2d_7 = self.deconv2d_7(deconv2d_8)
            deconv2d_6 = self.deconv2d_6(deconv2d_7)
            deconv2d_5 = self.deconv2d_5(deconv2d_6)
            deconv2d_4 = self.deconv2d_4(deconv2d_5)
            deconv2d_3 = self.deconv2d_3(deconv2d_4)
            deconv2d_2 = self.deconv2d_2(deconv2d_3)
            deconv2d_1 = self.deconv2d_1(deconv2d_2)
            deconv2d_0 = self.deconv2d_0(deconv2d_1)
           
        scale_8 = self.scale_8(transforms)
        scale_7 = self.scale_7(deconv2d_8)
        scale_6 = self.scale_6(deconv2d_7)
        scale_5 = self.scale_5(deconv2d_6)
        scale_4 = self.scale_4(deconv2d_5)
        scale_3 = self.scale_3(deconv2d_4)
        scale_2 = self.scale_2(deconv2d_3)
        scale_1 = self.scale_1(deconv2d_2)
        scale_0 = self.scale_0(deconv2d_1)

        out = self.output(deconv2d_0)
        # print(scale_0.shape)
        # print(scale_1.shape)
        # print(scale_2.shape)
        # print(scale_3.shape)
        # print(scale_4.shape)
        # print(scale_5.shape)
        # print(scale_6.shape)
        # print(scale_7.shape)
        # print(scale_8.shape)
        # print('PNet', out.shape)
        return out, [scale_0, scale_1, scale_2, scale_3, scale_4, scale_5, scale_6, scale_7, scale_8]

class PDNet(nn.Module):
    def __init__(self, 
        source_channels=1, 
        output_channels=1, 
        num_filters=32, 
        num_factors=[1, 2, 4, 8, 16, 24, 32, 48, 64],
        is_skipping=False,
        is_injecting=False):
        super().__init__()
        self.is_skipping = is_skipping
        self.is_injecting = is_injecting
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

        self.scale_8 = EqualConv2d(1, num_filters*num_factors[8], 1)
        self.scale_7 = EqualConv2d(1, num_filters*num_factors[7], 1)
        self.scale_6 = EqualConv2d(1, num_filters*num_factors[6], 1)
        self.scale_5 = EqualConv2d(1, num_filters*num_factors[5], 1)
        self.scale_4 = EqualConv2d(1, num_filters*num_factors[4], 1)
        self.scale_3 = EqualConv2d(1, num_filters*num_factors[3], 1)
        self.scale_2 = EqualConv2d(1, num_filters*num_factors[2], 1)
        self.scale_1 = EqualConv2d(1, num_filters*num_factors[1], 1)
        self.scale_0 = EqualConv2d(1, num_filters*num_factors[0], 1)

        # Transformation
        self.transformation = nn.Sequential(
            EqualConv2d(num_filters*num_factors[8],
                      num_filters*num_factors[8],
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.Flatten(),
            ScaledLeakyReLU(),
            nn.Linear(num_filters*num_factors[8], num_filters*num_factors[8]),
            ScaledLeakyReLU(),
            nn.Linear(num_filters*num_factors[8], num_filters*num_factors[8]),
            ScaledLeakyReLU(),
            nn.Linear(num_filters*num_factors[8], num_filters*num_factors[8]),
            ScaledLeakyReLU(),
            nn.Linear(num_filters*num_factors[8], 1),
            ScaledLeakyReLU(),
        )
       
    def forward(self, x, scales):
        # print(len(scales))
        # [print(s.shape)  for s in scales]
        scale_0, scale_1, scale_2, scale_3, scale_4, scale_5, scale_6, scale_7, scale_8 = scales

        conv2d_0 = self.conv2d_0(x       ) + self.scale_0(scale_0)
        conv2d_1 = self.conv2d_1(conv2d_0) + self.scale_1(scale_1)
        conv2d_2 = self.conv2d_2(conv2d_1) + self.scale_2(scale_2)
        conv2d_3 = self.conv2d_3(conv2d_2) + self.scale_3(scale_3)
        conv2d_4 = self.conv2d_4(conv2d_3) + self.scale_4(scale_4)
        conv2d_5 = self.conv2d_5(conv2d_4) + self.scale_5(scale_5)
        conv2d_6 = self.conv2d_6(conv2d_5) + self.scale_6(scale_6)
        conv2d_7 = self.conv2d_7(conv2d_6) + self.scale_7(scale_7)
        conv2d_8 = self.conv2d_8(conv2d_7) + self.scale_8(scale_8)

        # print(scale_0.shape, scale_1.shape, scale_2.shape, scale_3.shape, scale_4.shape, scale_5.shape, scale_6.shape, scale_7.shape, scale_8.shape)
        # print(conv2d_0.shape, conv2d_1.shape, conv2d_2.shape, conv2d_3.shape, conv2d_4.shape, conv2d_5.shape, conv2d_6.shape, conv2d_7.shape, conv2d_8.shape)

        transforms = self.transformation(conv2d_8)

        return transforms
