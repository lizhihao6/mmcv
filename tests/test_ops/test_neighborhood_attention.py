# Copyright (c) OpenMMLab. All rights reserved.
# * Modified from
# *https://github.com/SHI-Labs/NATTEN/blob/main/tests/test_na2d.py
#
import pytest
import torch

from mmcv.ops import NeighborhoodAttention

def _priv_test_allclose_cpu_cuda(
    batch_size,
    height,
    width,
    kernel_sizes=[3, 5, 7, 9],
    dims=[4, 8],
    heads=[1, 2, 3],
    tol=1e-8,
):
    for kernel_size in kernel_sizes:
        for dim in dims:
            for num_heads in heads:
                for qkv_bias in [True, False]:
                    model_kwargs = {
                        "dim": dim * num_heads,
                        "kernel_size": kernel_size,
                        "num_heads": num_heads,
                        "qkv_bias": qkv_bias,
                    }

                    base_state_dict = NeighborhoodAttention(
                        **model_kwargs
                    ).state_dict()

                    x1 = torch.randn((batch_size, height, width, dim * num_heads))
                    x2 = x1.clone().detach().cuda(0)

                    nat1 = NeighborhoodAttention(**model_kwargs).eval()
                    nat1.load_state_dict(base_state_dict, strict=False)

                    nat2 = NeighborhoodAttention(**model_kwargs).cuda(0).eval()
                    nat2.load_state_dict(base_state_dict, strict=False)

                    y1 = nat1(x1)
                    y2 = nat2(x2)

                    forward_mse = ((y1.data - y2.cpu().data) ** 2).mean()

                    assert forward_mse < tol, (
                        f"FAIL: Forward MSE ({forward_mse}) was above the specified"
                        f" tolerance (tol) for heads={heads}, dim={dim},"
                        f" kernel_size={kernel_size}."
                    )

                    y1.sum().backward()
                    y2.sum().backward()

                    for name, n1w in nat1.named_modules():
                        if type(n1w) is not torch.nn.Linear:
                            continue
                        for name2, n2w in nat2.named_modules():
                            if name != name2:
                                continue
                            if n1w.weight.grad is None or n2w.weight.grad is None:
                                continue
                            mse = (
                                (n1w.weight.grad - n2w.weight.grad.cpu()) ** 2
                            ).mean()
                            if hasattr(n1w, "bias") and n1w.bias is not None:
                                if hasattr(n1w.bias, "grad") and hasattr(
                                    n2w.bias, "grad"
                                ):
                                    mse += (
                                        (n1w.bias.grad - n2w.bias.grad.cpu()) ** 2
                                    ).mean()

                            assert mse < tol, (
                                f"FAIL: {name} gradient MSE ({mse}) was above the"
                                f" specified tolerance ({tol}) for heads={heads},"
                                f" dim={dim}, kernel_size={kernel_size}."
                            )

@pytest.mark.parametrize('device', [
    pytest.param(
        'cuda',
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason='requires CUDA support'))
])
@pytest.mark.parametrize('dtype', [torch.float, torch.half])
def test_neighborhood_attention(device, dtype):
    b, li, lj = 4, 14, 16
    _priv_test_allclose_cpu_cuda(b, li, lj)
