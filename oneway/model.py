import math
import random
import functools
import operator

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        # print(input.shape)
        # if len(input.shape) == 1:
            # input = input.unsqueeze(0)
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


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


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 1, 1, 1],
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})'
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 1, 1, 1],
        demodulate=True,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 1, 1, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 1, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 1, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        # print('ToRGB', input.shape, style.shape, out.shape)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out


class Generator(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 1, 1, 1],
        lr_mlp=0.01,
    ):
        super().__init__()

        self.size = size

        self.style_dim = style_dim

        layers = [PixelNorm()]
        # layers = []
        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )

        self.style = nn.Sequential(*layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f'noise_{layer_idx}', torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )

            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, style_dim))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    def make_noise(self):
        device = self.input.input.device

        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)

    def forward(
        self,
        styles,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
    ):
        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f'noise_{i}') for i in range(self.num_layers)
                ]

        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

            else:
                latent = styles[0]

        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        out = self.input(latent)
        out = self.conv1(out, latent[:, 0], noise=noise[0])

        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)

            i += 2

        image = skip

        if return_latents:
            return image, latent

        else:
            return image, None


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 1, 1, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))

            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 1, 1, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 1, 1, 1]):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(1, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out






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
            EqualConv2d(in_channels=source_channels,
                      out_channels=output_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      bias=bias),
            nn.GroupNorm(num_groups=number_grouping,
                         num_channels=output_channels),
            ScaledLeakyReLU(),
        )
        self.net = nn.Sequential(
            EqualConv2d(in_channels=output_channels,
                      out_channels=output_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.GroupNorm(num_groups=number_grouping,
                         num_channels=output_channels),
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
            nn.GroupNorm(num_groups=number_grouping,
                         num_channels=output_channels),
            ScaledLeakyReLU(),
        )
        self.net = nn.Sequential(
            EqualConv3d(in_channels=output_channels,
                      out_channels=output_channels,
                      kernel_size=[3, 3, 3],
                      stride=1,
                      padding=1,
                      bias=False),
            nn.GroupNorm(num_groups=number_grouping,
                         num_channels=output_channels),
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
            EqualConv2d(in_channels=source_channels,
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
            ScaledLeakyReLU(),
        )
        self.net = nn.Sequential(
            EqualConv2d(in_channels=output_channels,
                      out_channels=output_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.GroupNorm(num_groups=number_grouping,
                         num_channels=output_channels),
            ScaledLeakyReLU(),
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
            EqualConv3d(in_channels=source_channels,
                      out_channels=output_channels,
                      kernel_size=[3, 3, 3],
                      stride=1,
                      padding=1,
                      bias=False),
            nn.Upsample(scale_factor=scale_factor,
                        mode='trilinear',
                        align_corners=True),
            nn.GroupNorm(num_groups=number_grouping,
                         num_channels=output_channels),
            ScaledLeakyReLU(),
        )
        self.net = nn.Sequential(
            EqualConv3d(in_channels=output_channels,
                      out_channels=output_channels,
                      kernel_size=[3, 3, 3],
                      stride=1,
                      padding=1,
                      bias=False),
            nn.GroupNorm(num_groups=number_grouping,
                         num_channels=output_channels),
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
        # self.dropout = nn.Dropout()

    def forward(self, x):
        shape = x.shape
        y = self.conv2d(x)
        z = y.view([shape[0], shape[1], -1, shape[2], shape[3]]).transpose(2, 3) 
        # w = self.dropout(z)
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
        self.conv2d = EqualConv2d(in_channels=source_channels,
                                out_channels=output_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                bias=bias) 
        # self.dropout = nn.Dropout()
    def forward(self, x):
        y = x.transpose(2, 3)
        shape = y.shape
        z = y.reshape([shape[0], shape[1]*shape[2], shape[3], shape[4]]) 
        # w = self.dropout()
        return self.conv2d(z)


class PGNet(nn.Module):
    def __init__(self, 
        hparams=None,
        source_channels=1, 
        output_channels=1, 
        num_filters=8, 
        num_factors=[1, 2, 4, 8, 16, 24, 32, 48, 64], 
        is_skip=True
    ):
        super().__init__()
        self.hparams = hparams
        self.is_skip = is_skip

        # self.pixelnorm = PixelNorm()
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
            nn.GroupNorm(4, num_filters*num_factors[8]),
            nn.Flatten(),
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

            Reshape(num_filters*num_factors[8], 1, 1),  # NF x 64 x 1 x 4 x 4
            EqualConv2d(num_filters*num_factors[8],
                      num_filters*num_factors[8],
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.GroupNorm(4, num_filters*num_factors[8]),
            ScaledLeakyReLU()
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
            EqualConv2d(num_filters*1,
                      num_filters*1,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.GroupNorm(4, num_filters*1),
            ScaledLeakyReLU(),
            EqualConv2d(num_filters*1, output_channels, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.Tanh(),
        )

    def forward(self, x):
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
        
        if self.is_skip:
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
        # deconv2d_8 = self.deconv2d_8(transforms + self.skip_8(conv3d_8))
        # deconv2d_7 = self.deconv2d_7(deconv2d_8 + self.skip_7(conv3d_7))
        # deconv2d_6 = self.deconv2d_6(deconv2d_7 + self.skip_6(conv3d_6))
        # deconv2d_5 = self.deconv2d_5(deconv2d_6)
        # deconv2d_4 = self.deconv2d_4(deconv2d_5)
        # deconv2d_3 = self.deconv2d_3(deconv2d_4)
        # deconv2d_2 = self.deconv2d_2(deconv2d_3)
        # deconv2d_1 = self.deconv2d_1(deconv2d_2)
        # deconv2d_0 = self.deconv2d_0(deconv2d_1)
        out = self.output(deconv2d_0)
        # print('PNet', out.shape)
        return out


class PDNet(nn.Module):
    def __init__(self, 
        hparams=None,
        source_channels=1, 
        output_channels=1, 
        num_filters=8, 
        num_factors=[1, 2, 4, 8, 16, 24, 32, 48, 64]):
        super().__init__()
        self.pixelnorm = PixelNorm()
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
            nn.GroupNorm(4, num_filters*num_factors[8]),
            nn.Flatten(),
            ScaledLeakyReLU(),
            EqualLinear(num_filters*num_factors[8], num_filters*num_factors[8]),
            ScaledLeakyReLU(),
            EqualLinear(num_filters*num_factors[8], 1),
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

        return transforms