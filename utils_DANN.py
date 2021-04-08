#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torch.autograd import Function


class ReverseLayerF(Function):
    """
    Gradient Reversal Layer Reference:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """
    @staticmethod
    def forward(ctx, x, _lambda_):
        ctx._lambda_ = _lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, gradient_output):
        output = gradient_output.neg() * ctx._lambda_

        return output, None
