import torch
from minigrad.nn import Linear, MSELoss, ReLU, Sequential
import minigrad
from torch import nn



d_in = 3
d_hidden = 5
d_out = 1

# PyTorch ground truth
w_1 = torch.empty((d_hidden, d_in), requires_grad=True)
b_1 = torch.zeros(d_hidden, requires_grad=True)

w_2 = torch.empty((d_hidden, d_hidden), requires_grad=True)
b_2 = torch.zeros(d_hidden, requires_grad=True)

w_3 = torch.empty((d_out, d_hidden), requires_grad=True)
b_3 = torch.zeros(d_out, requires_grad=True)


layer1 = Linear(d_in, d_hidden)
layer2 = Linear(d_hidden, d_hidden)
layer3 = Linear(d_hidden, d_out)

# Init ground truth weights as custom layers
with torch.no_grad():
    w_1[:, :] = layer1.weights[:, :]
    w_2[:, :] = layer2.weights[:, :]
    w_3[:, :] = layer3.weights[:, :]


input = torch.tensor([1., 2., 3.])
target = torch.randn(d_out)

# Compute pytorch ground truth
loss = nn.MSELoss()

x1 = w_1 @ input + b_1
x2 = w_2 @ x1 + b_2
x3 = w_3 @ x2 + b_3
print(x3.shape, target.shape)
err = loss(x3, target)
err.backward()



# Compute our gradients
custom_loss = MSELoss()

custom_seq = Sequential(layer1, layer2, layer3)
print(custom_seq.modules)
# Forward pass

custom_output = custom_seq.forward(input)
custom_error = custom_loss.forward(custom_output, target)

# Backward pass
mse_grad = custom_loss.backward(custom_output, target)
custom_seq.backward(mse_grad, input)

print(w_2.grad)

print(layer2.grad[0])

custom_seq.zero_grad()
print(layer1.grad)

# Check that error and gradients are the same (use pytorch value as ground truth)
"""
self.assertTrue( torch.allclose(err, custom_error) )
self.assertTrue( torch.allclose(w.grad, layer.grad[0]) )
self.assertTrue( torch.allclose(b.grad, layer.grad[1]) )
"""