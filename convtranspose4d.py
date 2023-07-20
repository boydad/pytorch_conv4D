import numpy as np
import math, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv2d_tranpose_test():
    batch_size = 2
    in_channels = 2
    out_channels = 3
    input_size = [5, 7]
    kernel_size = 3
    stride = 5
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
    
    for iL in range(input_size[0]): # extra dimension!
        x = input_2d[...,iL,:]

        input_kernel = weight_2d.permute(2, 0, 1, 3)
        input_kernel = input_kernel.reshape(kernel_size*in_channels, 
                                            out_channels, 
                                            *weight_2d.shape[3:])

        # Fun with shapes
        input_x = x.repeat(1, kernel_size, 1)
        op_output = op(input_x, input_kernel, groups=kernel_size)

        op_output = op_output.reshape(batch_size, kernel_size, out_channels,
                                      *op_output.shape[2:])
        op_output = op_output.transpose(1, 2)
        output[:, :, iL*stride:iL*stride+kernel_size] += op_output    
    
    assert torch.allclose(true_result, output)
    print("tests passed")    

if __name__ == "__main__":
    conv2d_tranpose_test()