import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvTransposeNd_groups(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0,
                 output_padding=0,
                 padding_mode="zeros",
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 Nd: int = 4,
                 device=None, dtype=None):
        super(ConvTransposeNd_groups, self).__init__()

        assert padding == 0, "not implemented"
        assert output_padding == 0, "not implemented"
        assert padding_mode == "zeros", "not implemented"
        assert dilation == 1, "not implemented"
        assert groups == 1, "not implemented"
        assert Nd <= 4, "not implemented"

        if not isinstance(kernel_size, tuple):
            kernel_size = tuple(kernel_size for _ in range(Nd))
        if not isinstance(padding, tuple):
            padding = tuple(padding for _ in range(Nd))
        self.convtranspose_f = (F.conv_transpose1d, 
                                F.conv_transpose2d, 
                                F.conv_transpose3d)[Nd - 2]
        self.in_channels = in_channels 
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.padding_mode = padding_mode
        self.use_bias = bias
        self.Nd = Nd

        # Following the initilization in 
        # https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
        init_norm = np.sqrt(1/self.out_channels/np.prod(kernel_size))
        self.weight = nn.Parameter(
                init_norm*(2*torch.randn(
                            (in_channels, out_channels, *kernel_size), 
                            device=device, dtype=dtype) - 1)
        )
        self.bias = nn.Parameter(
                      init_norm*(2*torch.randn(out_channels, device=device, dtype=dtype)
                      if bias else self.register_parameter('bias', None) - 1)
                    )

    def forward(self, input):
        # either (B, C, L_1, L_2, ...) or (C, L_1, L_2, ...)
        assert len(input.shape) ==  self.Nd+2 or len(input.shape) == self.Nd+1, 'wrong input shape'
        unbatched = len(input.shape) == self.Nd+1
        if unbatched:
            input = input.unsqueeze(0)

        B = input.shape[0]
        input_size = input.shape[2:]
        output_size =  [(i-1)*self.stride + (self.kernel_size[ic] - 1) + 1 for ic, i 
                        in enumerate(input_size)]
        output = torch.zeros([B, self.out_channels, *output_size], 
                              device=self.weight.device, dtype=self.weight.dtype)

        op = lambda x, y, groups: self.convtranspose_f(
                    x, y, bias=None, stride=self.stride, padding=0, 
                    output_padding=0, groups=groups, dilation=1) 
        
        # The order of repeat and reshape are important here
        x = input.transpose(1, 2) # from (B, C, L_1, L_2, ...) to (B, L_1, C, L_2, ...)
        x = x.repeat(1, self.kernel_size[0], *[1 for i in range(self.Nd)]) # (B, K_1*L_1, C, L_2, ...)
        x = x.reshape(B, self.kernel_size[0]*input_size[0]*self.in_channels, 
                      *input_size[1:]) # (B, K1*L1*C, L2, ...)

        input_kernel = self.weight.repeat(input_size[0], *[1 for i in range(self.Nd+1)]) # (L1*C, O, K_1, K_2, ...)
        input_kernel = input_kernel.permute(2, 0, 1, *[i for i in range(3, 3+self.Nd-1)]) # (K1, C*L1, O, K2, ...)
        input_kernel = input_kernel.reshape(self.kernel_size[0]*self.in_channels*input_size[0], 
                                            self.out_channels, 
                                            *self.weight.shape[3:]) # (K_1*L_1*C, O, K2, ...)
        # Fun with shapes
        op_output = op(x, input_kernel, groups=self.kernel_size[0]*input_size[0]) # (B, K_1*L_1*O, L_2, ...)

        op_output = op_output.reshape(B, self.kernel_size[0], input_size[0], self.out_channels,
                                      *output.shape[3:])  # (B, K_1, L_1, O, L_2, ...)
        op_output = op_output.transpose(2, 3) # (B, K1, O, L1, L2, ...)

        # Two methods of summing the final tensor - # TODO: optimize this operation with convNd
        # or other clever tricks(?) to avoid loops of any sorts
        # Method 1: loop over kernel size
        for ik in range(self.kernel_size[0]):
            output[:, :, ik:ik+input_size[0]*self.stride:self.stride] += op_output[:, ik]

        """
        # Method 2: loop over input size --- probably slower?
        op_output = op_output.permute(0, 2, 3, 1, 4) # (B, O, L1, K1, L2, ...)
        for iL in range(input_size[0]):
            output[:, :, iL*stride:iL*stride+kernel_size] += op_output[:, :, iL]
        """            
        if self.use_bias:
            original_shape = output.shape
            output = output.view(B, original_shape[1], -1)
            output += self.bias.reshape(1, self.out_channels, 1)
            output = output.view(original_shape)
        
        if unbatched:
            # Don't forget to remove the dummy batch dimension
            output = output.squeeze(0) 
        return output
