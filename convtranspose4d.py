import numpy as np
import math, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv2d_tranpose_test():
    batch_size = 5
    in_channels = 13
    out_channels = 17
    input_size = [7, 11]
    kernel_size = 3
    stride = 2
    input_2d = torch.randn(batch_size, in_channels, *input_size).to(torch.float64)
    weight_2d = torch.randn(in_channels, out_channels, kernel_size, kernel_size).to(torch.float64)
    
    true_result = F.conv_transpose2d(input_2d, weight_2d, 
                                     bias=None, stride=stride, padding=0, 
                                     output_padding=0, groups=1, dilation=1)

    output_size =  [(i-1)*stride + (kernel_size - 1) + 1 for i in input_size]
    output = torch.zeros([batch_size, out_channels, *output_size], dtype=torch.float64)

    op = lambda x, y, groups: F.conv_transpose1d(
                x, y, bias=None, stride=stride, padding=0, 
                output_padding=0, groups=groups, dilation=1) 
    
    # The order of repeat and reshape are important here
    x = input_2d.permute(0, 2, 1, 3) # from (B, C, L1, L2, ...) to (B, L1, C, L2, ...)
    x = x.repeat(1, kernel_size, 1, 1) # (B, K1*L1, C, L2, ...)
    input_x = x.reshape(batch_size, kernel_size*input_size[0]*in_channels, 
                        *input_size[1:]) # (B, K1*L1*C, L2, ...)

    input_kernel = weight_2d.repeat(input_size[0], 1, 1, 1) # (L1*C, O, K1, K2, ...)
    input_kernel = input_kernel.permute(2, 0, 1, 3) # (K1, C*L1, O, K2, ...)
    input_kernel = input_kernel.reshape(kernel_size*in_channels*input_size[0], 
                                        out_channels, 
                                        *weight_2d.shape[3:]) # (K1*L1*C, O, K2, ...)

    # Fun with shapes
    op_output = op(input_x, input_kernel, groups=kernel_size*input_size[0])

    op_output = op_output.reshape(batch_size, kernel_size, input_size[0], out_channels,
                                  *output.shape[3:])  # (B, K1, L1, O, L2, ...)
    op_output = op_output.transpose(2, 3) # (B, K1, O, L1, L2, ...)

    # Two methods of summing the final tensor - # TODO: optimize this operation with convNd
    # or other clever tricks(?) to avoid loops of any sorts

    # Method 1: loop over kernel size
    for ik in range(kernel_size):
        output[:, :, ik:ik+input_size[0]*stride:stride] += op_output[:, ik]

    """
    # Method 2: loop over input size
    op_output = op_output.permute(0, 2, 3, 1, 4) # (B, O, L1, K1, L2, ...)
    for iL in range(input_size[0]):
        output[:, :, iL*stride:iL*stride+kernel_size] += op_output[:, :, iL]
    """

    assert torch.allclose(true_result, output)
    print("tests passed")    

if __name__ == "__main__":
    conv2d_tranpose_test()