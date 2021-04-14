#!/usr/bin/env python

"""Tests for `minigrad` package."""


import unittest

from torch.nn.modules.loss import CosineEmbeddingLoss

import minigrad
from minigrad.nn import Linear, MSELoss, ReLU
import torch
from torch import nn

class TestMinigrad(unittest.TestCase):
    """Tests for `minigrad` package."""

    def test_simple_linear(self):

        # Do this multiple times
        for _ in range(10):
            d_in = 3
            d_out = 5

            w = torch.empty((d_out, d_in), requires_grad=True)
            b = torch.zeros(d_out, requires_grad=True)

            layer = Linear(d_in, d_out)
            # Set weights of the layer equal to pytorch ones
            with torch.no_grad():
                w[:, :] = layer.weights[:, :]

            # pytorch loss
            loss = nn.MSELoss()

            input = torch.tensor([1., 2., 3.])
            target = torch.randn(d_out)
            
            # Compute pytorch ground truth
            output = w @ input + b
            err = loss(output, target)
            err.backward()

            custom_loss = MSELoss()
            mse_grad = custom_loss.backward(output, target)

            layer.backward(mse_grad, input)
            custom_error = custom_loss.forward(layer.forward(input), target)
            # Check that error and gradients are the same (use pytorch value as ground truth)
            self.assertTrue( torch.allclose(err, custom_error) )
            self.assertTrue( torch.allclose(w.grad, layer.grad[0]) )
            self.assertTrue( torch.allclose(b.grad, layer.grad[1]) )

# ============================================================================================================

    def test_simple_lin_relu(self):
        d_in = 3
        d_out = 5

        w = torch.empty((d_out, d_in), requires_grad=True)
        b = torch.zeros(d_out, requires_grad=True)

        layer = Linear(d_in, d_out)
        # Set weights of the layer equal to pytorch ones
        with torch.no_grad():
            w[:, :] = layer.weights[:, :]

        input = torch.tensor([1., 2., 3.])
        target = torch.randn(d_out)

        # Compute pytorch ground truth
        loss = nn.MSELoss()
        relu = nn.ReLU()
        s = w @ input + b
        output = relu(s)

        err = loss(output, target)
        err.backward()

        custom_loss = MSELoss()
        custom_relu = ReLU()

        # Forward pass
        s_custom = layer.forward(input)
        out_custom = custom_relu.forward(s_custom)
        custom_error = custom_loss.forward(out_custom, target)

        # Backward pass
        mse_grad = custom_loss.backward(output, target)
        relu_grad = custom_relu.backward(mse_grad, s_custom)
        layer.backward(relu_grad, input)

        # Check that error and gradients are the same (use pytorch value as ground truth)

        self.assertTrue( torch.allclose(err, custom_error) )
        self.assertTrue( torch.allclose(w.grad, layer.grad[0]) )
        self.assertTrue( torch.allclose(b.grad, layer.grad[1]) )

# ============================================================================================================
# ============================================================================================================

if __name__ == "__main__":
    unittest.main()