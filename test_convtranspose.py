import pytest
import numpy as np
import torch
from .convtranspose import ConvTransposeNd_groups

@pytest.mark.parametrize("Nd", [2, 3]) # only 1d, 2d, and 3d are implemented in pytorch
@pytest.mark.parametrize("test_numbers", 
    [(2, 3, 5, 7, 11, 13, 15, 17, 19, 23, 29, 31, 37), # primes
     (3, 2, 2, 5, 11, 7, 9, 3, 3, 5, 6,  2, 10), # some numbers
     (2, 7, 9, 17, 2, 21, 21, 13, 21, 16, 5, 15, 12),  # some numbers
     (11, 17, 8, 2, 15,  9,  9, 5, 3, 8, 3, 9, 2)  # some numbers
    ]
) 
def test_correctness_convtransposeNd(Nd, test_numbers):
    batch_size = test_numbers[0]
    in_channels = test_numbers[1]
    out_channels = test_numbers[2]
    stride = test_numbers[3]
    kernel_size = test_numbers[4:4+Nd]
    lattice_size = test_numbers[4+Nd:4+2*Nd]
    dtype = torch.float64
    device = 'cpu'

    native_convtranspose = (torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d)[Nd - 2]

    convT1 = native_convtranspose(in_channels, out_channels, 
                                  kernel_size, stride=stride, padding=0, 
                                  output_padding=0, groups=1, bias=True, dilation=1, 
                                  padding_mode='zeros', device=device, dtype=dtype)
    convT2 = ConvTransposeNd_groups(in_channels, out_channels, 
                                    kernel_size, stride=stride, padding=0, Nd=Nd,
                                    output_padding=0, groups=1, bias=True, dilation=1, 
                                    padding_mode='zeros', device=device, dtype=dtype)

    # Set the weight and biases to be the same
    convT2.weight = convT1.weight
    convT2.bias = convT1.bias

    test_input = torch.randn([batch_size, in_channels, *lattice_size], device=device, dtype=dtype)

    out1 = convT1(test_input)
    out2 = convT2(test_input)
    assert torch.allclose(out1, out2)