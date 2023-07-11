import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv4d_broadcast(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size,
                 padding,
                 stride=1,
                 padding_mode='circular',
                 dilation=1,
                 groups=1,
                 bias=True,
                 Nd=4,
                 bias_initializer=None,
                 kernel_initializer= None,
                 channels_last=False):
        super(Conv4d_broadcast, self).__init__()

        assert padding_mode == 'circular' or padding == 0 and padding_mode == 'zeros', \
            'Implemented only for circular or no padding'
        assert stride == 1, "not implemented"
        assert dilation == 1, "not implemented"
        assert groups == 1, "not implemented"
        assert Nd <= 4 and Nd > 2, "not implemented"

        if not isinstance(kernel_size, (tuple, list)):
            kernel_size = tuple(kernel_size for _ in range(Nd))
        if not isinstance(padding, (tuple, list)):
            padding = tuple(padding for _ in range(Nd))
        # assert np.all(np.array(padding) == np.array(kernel_size) - 1), "works only in circular mode"


        self.conv_f = (nn.Conv2d, nn.Conv3d)[Nd - 3]
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.padding_mode = padding_mode
        self.use_bias = bias

        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else self.register_parameter('bias', None)
        if bias_initializer is not None:
            bias_initializer(self.bias)

        self.conv_layers = torch.nn.ModuleList()
        for _ in range(self.kernel_size[0]):
            conv_layer = self.conv_f(
                in_channels=in_channels,
                out_channels=self.out_channels,
                bias=False,
                kernel_size=self.kernel_size[1:],
            )
            if kernel_initializer is not None:
                kernel_initializer(conv_layer.weight)

            if channels_last:
                channels_last = [torch.channels_last, torch.channels_last_3d][Nd-3]
                conv_layer.to(memory_format=channels_last)

            self.conv_layers.append(conv_layer)

    def do_padding(self, input):
        (b, c_i) = tuple(input.shape[0:2])
        size_i = tuple(input.shape[2:])
        size_p = [size_i[i] + self.padding[i] for i in range(len(size_i))]
        padding = tuple(np.array(
                [((self.padding[i+1]+1)//2, self.padding[i+1]//2) for i in range(len(size_i[1:]))]
                ).reshape(-1)[::-1])
        input = F.pad(  # Ls padding
            input.reshape(b, -1, *size_i[1:]),
            padding,
            'circular',
            0
            ).reshape(b, c_i, -1, *size_p[1:])
        return input

    def forward(self, input):
        if self.padding_mode == 'circular':
            input = self.do_padding(input)

        (b, c_i) = tuple(input.shape[0:2])
        size_i = tuple(input.shape[2:])
        size_k = self.kernel_size
        padding = list(self.padding)
        size_o = (size_i[0], ) + tuple([size_i[x+1] - size_k[x+1] + 1 for x in range(len(size_i[1:]))])

        result = torch.zeros((b, self.out_channels) + size_o, device=input.device)

        for i in range(size_k[0]):
            cinput = torch.transpose(input, 1, 2)  # 1 -> channels, 2 -> Lt
            cinput = cinput.reshape(-1, c_i, *size_i[1:])  # merge bs and Lt
            output = self.conv_layers[i](cinput)
            output = output.reshape(b, size_i[0], *output.shape[1:])
            output = torch.transpose(output, 1, 2)
            result = result + torch.roll(output, -1 * i, 2)

        if self.use_bias:
            resultShape = result.shape
            result = result.view(b, resultShape[1], -1)
            result += self.bias.reshape(1, self.out_channels, 1)
            result = result.view(resultShape)

        shift = math.ceil(padding[0] / 2)
        result = torch.roll(result, shift, 2)
        # after we rearranged 3D convolutions we can cut 4th dimention
        # depending on padding type circular or not
        dim_size = size_i[0] + self.padding[0] - size_k[0] + 1
        result = result[:, :, :dim_size, ]

        return result


class Conv4d_groups(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size,
                 padding,
                 stride=1,
                 padding_mode='circular',
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 Nd: int = 4,
                 bias_initializer=None,
                 kernel_initializer=None,
                 channels_last=False):
        super(Conv4d_groups, self).__init__()

        assert padding_mode == 'circular' or padding == 0, 'Implemented only for circular or no padding'
        assert stride == 1, "not implemented"
        assert dilation == 1, "not implemented"
        assert groups == 1, "not implemented"
        assert Nd <= 4, "not implemented"
        assert padding == kernel_size - 1, "works only in circular mode"

        if not isinstance(kernel_size, tuple):
            kernel_size = tuple(kernel_size for _ in range(Nd))
        if not isinstance(padding, tuple):
            padding = tuple(padding for _ in range(Nd))
        self.conv_f = (nn.Conv1d, nn.Conv2d, nn.Conv3d)[Nd - 2]

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.padding_mode = padding_mode
        self.use_bias = bias

        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else self.register_parameter('bias', None)
        if bias_initializer is not None:
            bias_initializer(self.bias)

        self.conv = self.conv_f(in_channels=in_channels*self.kernel_size[0],
                                out_channels=self.out_channels*self.kernel_size[0],
                                bias=False,
                                stride=stride,
                                kernel_size=self.kernel_size[1:],
                                padding_mode=self.padding_mode,
                                groups=self.kernel_size[0])
        if channels_last:
            channels_last = [torch.channels_last, torch.channels_last_3d][Nd-3]
            self.conv.to(memory_format=channels_last)

        if kernel_initializer is not None:
            kernel_initializer(self.conv.weight)

    def do_padding(self, input):
        (b, c_i) = tuple(input.shape[0:2])
        size_i = tuple(input.shape[2:])
        size_p = [size_i[i] + self.padding[i] for i in range(len(size_i))]
        input = F.pad(  # Ls padding
            input.reshape(b, -1, *size_i[1:]),
            tuple(np.array(
                # [(0, self.padding[i+1]) for i in range(len(size_i[1:]))]
                [((self.padding[i+1]+1)//2, self.padding[i+1]//2) for i in range(len(size_i[1:]))]
                ).reshape(-1)),
            'circular',
            0
            ).reshape(b, c_i, -1, *size_p[1:])
        return input

    def forward(self, input):
        if self.padding_mode == 'circular':
            input = self.do_padding(input)

        (b, c_i) = tuple(input.shape[0:2])
        size_i = tuple(input.shape[2:])
        size_k = self.kernel_size
        padding = list(self.padding)
        size_o = (size_i[0], ) + tuple([size_i[x+1] - size_k[x+1] + 1 for x in range(len(size_i[1:]))])

        # size_o = tuple([size_i[x] + padding[x] - size_k[x] + 1 for x in range(len(size_i))])

        cinput = torch.transpose(input, 1, 2)  # (bs, channels, Lt, ...) -> (bs, Lt, channels, ...)
        cinput = cinput.reshape(b * size_i[0], c_i, *size_i[1:])  # (bs, Lt, ...) -> (bs * Lt, ...)
        # # (bs * Lt, c_i, ...) -> (bs * Lt, 1, c_i, ...) -> (bs * Lt, k[0], c_i, ...) -> (bs * Lt, k[0] * c_i, ...)
        cinput = cinput[:,np.newaxis,:] \
            .expand(cinput.shape[0], self.kernel_size[0], *cinput.shape[1:]) \
            .reshape(cinput.shape[0], self.kernel_size[0] * cinput.shape[1], *cinput.shape[2:])
        out = self.conv(cinput)  # out.shape = (bs * Lt, k[0] * c_o, ...)
        # (bs * Lt, c_o * k[0], ...) -> (bs, Lt, k[0], c_o, ...)
        out = out.reshape(b, size_i[0], self.kernel_size[0], self.out_channels, *size_o[1:])
        out = out.transpose(1, 3)# (bs, Lt, k[0], c_o, ...) -> (bs, c_o, k[0], Lt...)
        out = out.split(1, dim=2)  # (bs, c_o, k[0], Lt...)-> list( (bs, c_o, 1, Lt...) )
        out = torch.stack([torch.roll(out[i].squeeze(2), -1 * i, 2) for i in range(len(out))], dim=0)
        result = torch.sum(out, dim=0)

        if self.use_bias:
            resultShape = result.shape
            result = result.view(b,resultShape[1],-1)
            result += self.bias.reshape(1, self.out_channels, 1)
            result =  result.view(resultShape)

        shift = math.ceil(padding[0] / 2)
        result = torch.roll(result, shift, 2)
        result = result[:, :, :size_o[0], ]

        return result
