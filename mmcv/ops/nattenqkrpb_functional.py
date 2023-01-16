# Copyright (c) OpenMMLab. All rights reserved.
from torch.autograd import Function

from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', ['nattenqkrpb_forward', 'nattenqkrpb_backward'])


class NATTENQKRPBFunction(Function):
    """
    QK+RPB autograd function
    Computes neighborhood attention weights given queries and keys,
    and adds relative positional biases.
    This calls the `QKRPB` kernel.
    """

    @staticmethod
    def forward(ctx, query, key, rpb):
        query = query.contiguous()
        key = key.contiguous()
        attn = nattenqkrpb_forward(
            query,
            key,
            rpb)
        ctx.save_for_backward(query, key)
        return attn

    @staticmethod
    def backward(ctx, grad_out):
        outputs = nattenqkrpb_backward(
            grad_out.contiguous(), ctx.saved_variables[0], ctx.saved_variables[1])
        d_query, d_key, d_rpb = outputs
        return d_query, d_key, d_rpb, None


nattenqkrpb = NATTENQKRPBFunction.apply
