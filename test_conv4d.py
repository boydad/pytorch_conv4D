# ks = 2 is not working due to bug in pytorch.nn.conv2d with padding=1
import math
import pytest
from functools import partial
import torch
import torch.nn as nn
from conv4d import Conv4d_broadcast, Conv4d_groups
import timeit
import numpy as np
import scipy.stats as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

def init(inChans, outChans, L, Nd, bs, ks, isBias, Conv4dClass):
    def init_broadcast(weights):
        def wgen(weights):
            for i in range(weights.shape[2]):
                yield weights[:, :, i, ]
        g = wgen(weights)

        def init(x):
            tmp = next(g)
            x.data = tmp
            return x

        return init

    def init_groups(x, weights):
        Lt = weights.shape[2]
        tmp = [weights[:, :, i, ...] for i in range(Lt)]
        tmp = torch.cat(tmp, dim=0)
        x.data = tmp
        return x

    def init_bias(x, bias):
        x.data = bias
        return x

    assert ks % 2 == 1, 'Since PT 1.5 works only with odd kernel size'

    padding_mode = 'circular'
    x = torch.randn(bs, inChans, *((L,)*Nd)).to(device)

    convPT = nn.Conv3d(
        inChans, outChans, ks, stride=1, padding=(ks-1)//2,
        bias=isBias, padding_mode=padding_mode
        ).to(device)

    conv = Conv4dClass(
        inChans, outChans, Nd=Nd, kernel_size=ks, padding=ks-1, bias=isBias,
        padding_mode=padding_mode,
        kernel_initializer=partial(init_groups, weights=tuple(convPT.parameters())[0])
        if Conv4dClass.__name__ == 'Conv4d_groups'
        else init_broadcast(tuple(convPT.parameters())[0]),
        bias_initializer=lambda x: init_bias(x, tuple(convPT.parameters())[1]) if isBias else None
        ).to(device)

    return x, convPT, conv


@pytest.mark.parametrize('inChans', [1, 2])
@pytest.mark.parametrize('outChans', [1, 2, 8])
@pytest.mark.parametrize('L', [8, 16])
@pytest.mark.parametrize('Nd', [3])
@pytest.mark.parametrize('bs', [256])
# ks = 2 is not working due to bug in pytorch.nn.conv2d with padding=1
@pytest.mark.parametrize('ks', [3, 5, 7])
@pytest.mark.parametrize('isBias', [True, False])
@pytest.mark.parametrize('Conv4dClass', [Conv4d_groups, Conv4d_broadcast])
def test_convNd(inChans, outChans, L, Nd, bs, ks, isBias, Conv4dClass):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    _data, _convPt, _convNd = init(inChans, outChans, L, Nd, bs, ks, isBias, Conv4dClass)
    outPT = _convPt(_data)
    out = _convNd(_data)

    # rotate data since padding in torch.conv2D -> (pad, pad)
    # and in convNd -> (0, 2 * pad) so output will be rotated by pad/2
    v = math.ceil((ks-1) / 2)
    out = torch.roll(out, v, 3)
    out = torch.roll(out, v, 4)


    diff = torch.abs((out-outPT)).max()
    print(f"convNd max error: {diff:.2g}")
    assert diff < 1e-5, f'err: {diff}'


def compare_time(inChans, outChans, L, Nd, bs, ks, isBias, Conv4dClass):
    import torch
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    _data, _convPt, _convNd = init(inChans, outChans, L, Nd, bs, ks, isBias, Conv4dClass)
    times = np.array(
        timeit.repeat(
            "torch.cuda.synchronize(); out = _convPt(_data);torch.cuda.synchronize();",
            globals=locals(), number=10)
        )
    print("ConvPt Forward time: ", f'{times[1:].mean():3g} pm {sns.sem(times[1:]):3g}')

    times = np.array(
        timeit.repeat(
            "torch.cuda.synchronize(); out = _convNd(_data);torch.cuda.synchronize();",
            globals=locals(), number=10)
        )
    print("ConvNd Forward time: ", f'{times[1:].mean():3g} pm {sns.sem(times[1:]):3g}')

    times = np.array(
        timeit.repeat(
            "torch.cuda.synchronize(); out = _convPt(_data); "
            "torch.sum(out, dim=tuple(range(len(out.shape)))).backward(); torch.cuda.synchronize();",
            globals=locals(), number=10)
        )
    print("ConvPt Forward+Backward time: ", f'{times[1:].mean():3g} pm {sns.sem(times[1:]):3g}')

    times = np.array(
        timeit.repeat(
            "torch.cuda.synchronize(); out = _convNd(_data); "
            "torch.sum(out, dim=tuple(range(len(out.shape)))).backward(); torch.cuda.synchronize();",
            globals=locals(), number=10)
        )
    print("ConvNd Forward+Backward time: ", f'{times[1:].mean():3g} pm {sns.sem(times[1:]):3g}')


def run_4d_benchmark(inChans, outChans, L, bs, ks, isBias, Conv4dClass):
    import torch
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    Nd = 4
    _data = torch.randn(bs, inChans, *((L,)*Nd)).to(device)
    _convNd = Conv4dClass(
        inChans, outChans, Nd=Nd, kernel_size=ks, padding=ks-1, bias=isBias,
        padding_mode='circular').to(device)

    times = np.array(
        timeit.repeat(
            "torch.cuda.synchronize(); out = _convNd(_data);torch.cuda.synchronize();",
            globals=locals(), number=10)
        )
    print("Forward time: ", f'{times[1:].mean():3g} pm {sns.sem(times[1:]):3g}')

    times = np.array(
        timeit.repeat(
            "torch.cuda.synchronize(); out = _convNd(_data); "
            "torch.sum(out, dim=tuple(range(len(out.shape)))).backward(); torch.cuda.synchronize();",
            globals=locals(), number=10)
        )
    print("Forward+Backward time: ", f'{times[1:].mean():3g} pm {sns.sem(times[1:]):3g}')


if __name__ == "__main__":
    for conv_type in [Conv4d_broadcast, Conv4d_groups]:
        print(conv_type)
        print("=======================================")
        print("--> Bechmark 3D")
        test_convNd(inChans=1, outChans=2, L=16, Nd=3, bs=256, ks=3, isBias=True, Conv4dClass=conv_type)
        compare_time(inChans=64, outChans=64, L=16, Nd=3, bs=256, ks=3, isBias=True, Conv4dClass=conv_type)
        print("--> Benchmark 4D")
        print("----> inChannels = 18, outChannels = 32")
        run_4d_benchmark(inChans=18, outChans=32, L=8, bs=64, ks=3, isBias=True, Conv4dClass=conv_type)
        print("----> inChannels = 32, outChannels = 32")
        run_4d_benchmark(inChans=32, outChans=32, L=8, bs=64, ks=3, isBias=True, Conv4dClass=conv_type)
        print("----> inChannels = 32, outChannels = 48")
        run_4d_benchmark(inChans=32, outChans=48, L=8, bs=64, ks=3, isBias=True, Conv4dClass=conv_type)
