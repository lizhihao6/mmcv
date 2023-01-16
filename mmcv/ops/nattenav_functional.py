# Copyright (c) OpenMMLab. All rights reserved.
from torch.autograd import Function

from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', ['nattenav_forward',
                                          'nattenav_backward'])


class NATTENAVFunction(Function):
    """
    AV autograd function
    Computes neighborhood attention outputs given attention weights,
    and values.
    This calls the `AV` kernel.
    """

    @staticmethod
    def forward(ctx, attn, value):
        attn = attn.contiguous()
        value = value.contiguous()
        out = ext_module.nattenav_forward(
            attn,
            value)
        ctx.save_for_backward(attn, value)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        outputs = ext_module.nattenav_backward(
            grad_out.contiguous(),
            ctx.saved_variables[0],
            ctx.saved_variables[1])
        d_attn, d_value = outputs
        return d_attn, d_value, None


nattenav = NATTENAVFunction.apply
