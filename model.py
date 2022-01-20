import collections
import jittor as jt

from jittor import nn
from jittor import init
from jittor import Function


from math import sqrt
import random
import numpy as np


class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.size(1) * weight[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module.weight
        module.weight_orig=weight
        module.register_pre_forward_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module,self.name,weight)  # setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module



class FusedUpsample(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        super().__init__()

        weight = jt.randn(in_channel, out_channel, kernel_size, kernel_size)
        bias = jt.zeros(out_channel)

        fan_in = in_channel * kernel_size * kernel_size
        self.multiplier = sqrt(2 / fan_in)

        self.weight = weight
        self.bias = bias

        self.pad = padding

    def execute(self, input):
        weight = nn.pad(self.weight * self.multiplier, [1, 1, 1, 1])
        weight = (
            weight[:, :, 1:, 1:]
            + weight[:, :, :-1, 1:]
            + weight[:, :, 1:, :-1]
            + weight[:, :, :-1, :-1]
        ) / 4

        out = nn.conv_transpose2d(input, weight, self.bias, stride=2, padding=self.pad)

        return out


class FusedDownsample(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        super().__init__()

        weight = jt.randn(out_channel, in_channel, kernel_size, kernel_size)
        bias = jt.zeros(out_channel)

        fan_in = in_channel * kernel_size * kernel_size
        self.multiplier = sqrt(2 / fan_in)

        self.weight = weight
        self.bias = bias

        self.pad = padding

    def execute(self, input):
        weight = nn.pad(self.weight * self.multiplier, [1, 1, 1, 1])
        weight = (
            weight[:, :, 1:, 1:]
            + weight[:, :, :-1, 1:]
            + weight[:, :, 1:, :-1]
            + weight[:, :, :-1, :-1]
        ) / 4

        out = nn.conv2d(input, weight, self.bias, stride=2, padding=self.pad)

        return out


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def execute(self, input):
        return input / jt.sqrt(jt.mean(input**2, dim=1, keepdims=True) + 1e-8)


class BlurFunctionBackward(Function):
    # @staticmethod
    def execute(self, grad_output, kernel, kernel_flip):
        # ctx.save_for_backward(kernel, kernel_flip)
        self.kernel=kernel  # self.kernel=kernel
        self.kernel_flip=kernel_flip  # self.kernel_flip=kernel_flip

        grad_input = nn.conv2d(
            grad_output, kernel_flip, padding=1, groups=grad_output.shape[1]
        )

        return grad_input

    # @staticmethod
    def grad(self, gradgrad_output):
        # kernel, kernel_flip = ctx.saved_tensors

        grad_input = nn.conv2d(
            gradgrad_output, self.kernel, padding=1, groups=gradgrad_output.shape[1]
        )

        return grad_input, None, None


class BlurFunction(Function):
    # @staticmethod
    def execute(self, input, kernel, kernel_flip):
        # ctx.save_for_backward(kernel, kernel_flip)
        self.kernel=kernel
        self.kernel_flip=kernel_flip

        output = nn.conv2d(input, kernel, padding=1, groups=input.shape[1])

        return output

    # @staticmethod
    def grad(self, grad_output):
        blurFunctionBackward = BlurFunctionBackward.apply
        grad_input = blurFunctionBackward(grad_output, self.kernel, self.kernel_flip)

        return grad_input, None, None


class Blur(nn.Module):
    def __init__(self, channel):
        super().__init__()

        weight = jt.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype='float32')
        weight = weight.view(1, 1, 3, 3)
        weight = weight / weight.sum()
        weight_flip = jt.flip(weight, [2, 3])
        self.weight = weight.repeat(channel, 1, 1, 1)
        self.weight_flip = weight_flip.repeat(channel, 1, 1, 1)

        # self.register_buffer('weight', weight.repeat(channel, 1, 1, 1))
        # self.register_buffer('weight_flip', weight_flip.repeat(channel, 1, 1, 1))

    def execute(self, input):
        blur = BlurFunction.apply
        return blur(input, self.weight, self.weight_flip)
        # return F.conv2d(input, self.weight, padding=1, groups=input.shape[1])



class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight=jt.normal(0,1,conv.weight.size())
        conv.bias=jt.zeros(conv.bias.shape)
        self.conv = equal_lr(conv)

    def execute(self, input):
        return self.conv(input)


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight=jt.normal(0,1,linear.weight.size())
        linear.bias=jt.zeros(linear.bias.shape)

        self.linear = equal_lr(linear)


    def execute(self, input):
        return self.linear(input)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        padding,
        kernel_size2=None,
        padding2=None,
        downsample=False,
        fused=False,
    ):
        super().__init__()

        pad1 = padding
        pad2 = padding
        if padding2 is not None:
            pad2 = padding2

        kernel1 = kernel_size
        kernel2 = kernel_size
        if kernel_size2 is not None:
            kernel2 = kernel_size2

        self.conv1 = nn.Sequential(
            EqualConv2d(in_channel, out_channel, kernel1, padding=pad1),
            nn.LeakyReLU(0.2),
        )

        if downsample:
            if fused:
                self.conv2 = nn.Sequential(
                    Blur(out_channel),
                    FusedDownsample(out_channel, out_channel, kernel2, padding=pad2),
                    nn.LeakyReLU(0.2),
                )

            else:
                self.conv2 = nn.Sequential(
                    Blur(out_channel),
                    EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),
                    nn.AvgPool2d(2),
                    nn.LeakyReLU(0.2),
                )

        else:
            self.conv2 = nn.Sequential(
                EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),
                nn.LeakyReLU(0.2),
            )

    def execute(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        return out


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channel,affine=False)
        self.style = EqualLinear(style_dim, in_channel * 2)

        self.style.linear.bias[:in_channel] = 1
        self.style.linear.bias[in_channel:] = 0

    def execute(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out


class NoiseInjection(nn.Module):
    def __init__(self, channel):
        super().__init__()
        # self._parameters=nn.ParameterList()

        self.weight = jt.zeros((1, channel, 1, 1))

    def execute(self, image, noise):
        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(jt.randn(1, channel, size, size))

    def execute(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class StyledConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        padding=1,
        style_dim=512,
        initial=False,
        upsample=False,
        fused=False,
    ):
        super().__init__()

        if initial:
            self.conv1 = ConstantInput(in_channel)

        else:
            if upsample:
                if fused:
                    self.conv1 = nn.Sequential(
                        FusedUpsample(
                            in_channel, out_channel, kernel_size, padding=padding
                        ),
                        Blur(out_channel),
                    )

                else:
                    self.conv1 = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='nearest'),
                        EqualConv2d(
                            in_channel, out_channel, kernel_size, padding=padding
                        ),
                        Blur(out_channel),
                    )

            else:
                self.conv1 = EqualConv2d(
                    in_channel, out_channel, kernel_size, padding=padding
                )

        self.noise1 = equal_lr(NoiseInjection(out_channel))
        self.adain1 = AdaptiveInstanceNorm(out_channel, style_dim)
        self.lrelu1 = nn.LeakyReLU(0.2)

        self.conv2 = EqualConv2d(out_channel, out_channel, kernel_size, padding=padding)
        self.noise2 = equal_lr(NoiseInjection(out_channel))
        self.adain2 = AdaptiveInstanceNorm(out_channel, style_dim)
        self.lrelu2 = nn.LeakyReLU(0.2)

    def execute(self, input, style, noise):
        out = self.conv1(input)
        out = self.noise1(out, noise)
        out = self.lrelu1(out)
        out = self.adain1(out, style)

        out = self.conv2(out)
        out = self.noise2(out, noise)
        out = self.lrelu2(out)
        out = self.adain2(out, style)

        return out


class Generator(nn.Module):
    def __init__(self, code_dim, fused=True):
        super().__init__()

        self.progression = nn.Sequential(
            [
                StyledConvBlock(512, 512, 3, 1, initial=True),  # 4
                StyledConvBlock(512, 512, 3, 1, upsample=True),  # 8
                StyledConvBlock(512, 512, 3, 1, upsample=True),  # 16
                StyledConvBlock(512, 512, 3, 1, upsample=True),  # 32
                StyledConvBlock(512, 256, 3, 1, upsample=True),  # 64
                StyledConvBlock(256, 128, 3, 1, upsample=True, fused=fused),  # 128
                StyledConvBlock(128, 64, 3, 1, upsample=True, fused=fused),  # 256
                StyledConvBlock(64, 32, 3, 1, upsample=True, fused=fused),  # 512
                StyledConvBlock(32, 16, 3, 1, upsample=True, fused=fused),  # 1024
            ]
        )

        self.to_rgb = nn.Sequential(
            [
                EqualConv2d(512, 3, 1),
                EqualConv2d(512, 3, 1),
                EqualConv2d(512, 3, 1),
                EqualConv2d(512, 3, 1),
                EqualConv2d(256, 3, 1),
                EqualConv2d(128, 3, 1),
                EqualConv2d(64, 3, 1),
                EqualConv2d(32, 3, 1),
                EqualConv2d(16, 3, 1),
            ]
        )

        # self.blur = Blur()

    def execute(self, style, noise, step=0, alpha=-1, mixing_range=(-1, -1)):
        out = noise[0]

        if len(style) < 2:
            inject_index = [len(self.progression) + 1]

        else:
            inject_index = sorted(random.sample(list(range(step)), len(style) - 1))

        crossover = 0

        for i, (conv, to_rgb) in enumerate(zip(self.progression, self.to_rgb)):
            if mixing_range == (-1, -1):
                if crossover < len(inject_index) and i > inject_index[crossover]:
                    crossover = min(crossover + 1, len(style))

                style_step = style[crossover]

            else:
                if mixing_range[0] <= i <= mixing_range[1]:
                    style_step = style[1]

                else:
                    style_step = style[0]

            if i > 0 and step > 0:
                out_prev = out
                
            out = conv(out, style_step, noise[i])

            if i == step:
                out = to_rgb(out)

                if i > 0 and 0 <= alpha < 1:
                    skip_rgb = self.to_rgb[i - 1](out_prev)
                    skip_rgb = nn.interpolate(skip_rgb, scale_factor=2, mode='nearest')
                    out = (1 - alpha) * skip_rgb + alpha * out

                break

        return out


class StyledGenerator(nn.Module):
    def __init__(self, code_dim=512, n_mlp=8):
        super().__init__()

        self.generator = Generator(code_dim)

        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(EqualLinear(code_dim, code_dim))
            layers.append(nn.LeakyReLU(0.2))

        self.style = nn.Sequential(*layers)

    def execute(
        self,
        input,
        noise=None,
        step=0,
        alpha=-1,
        mean_style=None,
        style_weight=0,
        mixing_range=(-1, -1),
    ):
        styles = []
        if type(input) not in (list, tuple):
            input = [input]

        for i in input:
            styles.append(self.style(i))

        batch = input[0].shape[0]

        if noise is None:
            noise = []

            for i in range(step + 1):
                size = 4 * 2 ** i
                noise.append(jt.randn(batch, 1, size, size))

        if mean_style is not None:
            styles_norm = []

            for style in styles:
                styles_norm.append(mean_style + style_weight * (style - mean_style))

            styles = styles_norm

        return self.generator(styles, noise, step, alpha, mixing_range=mixing_range)

    def mean_style(self, input):
        style = self.style(input).mean(0, keepdims=True)

        return style


class Discriminator(nn.Module):
    def __init__(self, fused=True, from_rgb_activate=False):
        super().__init__()

        self.progression = nn.Sequential(
            [
                ConvBlock(16, 32, 3, 1, downsample=True, fused=fused),  # 512
                ConvBlock(32, 64, 3, 1, downsample=True, fused=fused),  # 256
                ConvBlock(64, 128, 3, 1, downsample=True, fused=fused),  # 128
                ConvBlock(128, 256, 3, 1, downsample=True, fused=fused),  # 64
                ConvBlock(256, 512, 3, 1, downsample=True),  # 32
                ConvBlock(512, 512, 3, 1, downsample=True),  # 16
                ConvBlock(512, 512, 3, 1, downsample=True),  # 8
                ConvBlock(512, 512, 3, 1, downsample=True),  # 4
                ConvBlock(513, 512, 3, 1, 4, 0),
            ]
        )

        def make_from_rgb(out_channel):
            if from_rgb_activate:
                return nn.Sequential(EqualConv2d(3, out_channel, 1), nn.LeakyReLU(0.2))

            else:
                return EqualConv2d(3, out_channel, 1)

        self.from_rgb = nn.Sequential(
            [
                make_from_rgb(16),
                make_from_rgb(32),
                make_from_rgb(64),
                make_from_rgb(128),
                make_from_rgb(256),
                make_from_rgb(512),
                make_from_rgb(512),
                make_from_rgb(512),
                make_from_rgb(512),
            ]
        )

        # self.blur = Blur()

        self.n_layer = len(self.progression)

        self.linear = EqualLinear(512, 1)

    def execute(self, input, step=0, alpha=-1):
        for i in range(step, -1, -1):
            index = self.n_layer - i - 1

            if i == step:
                out = self.from_rgb[index](input)

            if i == 0:
                # out_std = out.std()  # jt.sqrt(out.var(0, unbiased=False) + 1e-8)
                out_std=((out-out.mean(0)).sqr().sum(0)+1e-8).sqrt()
                mean_std = out_std.mean()
                mean_std = mean_std.expand(out.size(0), 1, 4, 4)
                out = jt.concat([out, mean_std], 1)

            out = self.progression[index](out)

            if i > 0:
                if i == step and 0 <= alpha < 1:
                    skip_rgb = nn.avg_pool2d(input, 2)
                    skip_rgb = self.from_rgb[index + 1](skip_rgb)

                    out = (1 - alpha) * skip_rgb + alpha * out

        out = out.squeeze(2).squeeze(2)
        # print(input.size(), out.size(), step)
        out = self.linear(out)

        return out
