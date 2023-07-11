# PyTorch conv4D

The repository contains a four-dimensional convolutional layer for the PyTorch framework. The implementation of the 4D convolutions layer is done as a set of 3D convolutional layers. Up to my best knowledge first implementation of this idea [was done for TensorFlow](https://github.com/funkey/conv4d) where one can see some explanations. For PyTorch, there are implementations by [Timothy Gebhard](https://github.com/timothygebhard/pytorch-conv4d) and [Josué Page Vizcaíno](https://github.com/pvjosue/pytorch_convNd). 

The current repository aims to speed up naive implementations. Using broadcasting one can decrease the number of calls of 3D convolutions layers up to kernel size in the fourth direction. Using the `group` parameter one can decrease the number of calls even further. In the repository, two implementations are implemented and they show different performance. For example, for testing parameters, these implementations show 20% - 50% worse speed than PyTorch implementation, and much better than naive implementation(code is written in such a way that it uses Nd - 1 Pytorch convolution layers and, therefore, can be compared with Pytorch implementation in 3D). However, for a different set of parameters performance will be different.

The code is written only for `stride=1`, `dilation=1`, `groups=1` and zero or circular padding. Tests were done only for a specific set of parameters and no guarantee can be done. Before running add tests for your parameters in `test_conv4d.py`, to run tests use `pytest`. Performance measurement can be done with `ppython -m  pytorch_conv4D.test_conv4d`.

The code was tested using Nvidia GPUs and PyTorch 1.4. 
