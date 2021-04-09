import torch
from torch import empty

class Module(object):

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []

# ============================================================================================================

class Linear(Module):

    def __init__(self, input_features, output_features):
        super(Linear, self).__init__()
        self.weights = torch.randn(size=(output_features, input_features), requires_grad=False)
        self.bias = torch.zeros(output_features, requires_grad=False)
        self.grad = None

    def forward(self, x):
        return self.weights@x + self.bias 

    def backward(self, gradwrtoutput, x):
        dl_dw = gradwrtoutput.view(-1, 1)@x.view(1, -1)
        dl_db = gradwrtoutput
        if self.grad is None:
            self.grad = (dl_dw, dl_db)
        else:
            # accumulate gradient
            old_dw, old_db = self.grad
            self.grad = (old_dw + dl_dw, old_db + dl_db)
        # Return gradient wrt to input, used by parent layer
        return self.weights.T @ gradwrtoutput

    def params(self):
        return [self.weights, self.bias]

# ============================================================================================================

class Sequential(Module):

    def __init__(self, *args):
        super(Sequential, self).__init__()
        self.modules = args


    def forward(self, x):
        pass

# ============================================================================================================

class ReLU(Module):

    def forward(self, x):
        res = empty(size=x.shape)
        res[x <= 0] = 0
        res[x > 0] = x[x > 0]
        return res

    def backward(self, gradwrtoutput, x):
        mask = empty(size=x.shape)
        mask[x <= 0] = 0
        mask[x > 0] = 1
        return gradwrtoutput * mask

# ============================================================================================================

class Tanh(Module):
    def forward(self, x):
        ex = e ** x
        e_x = e ** (-x)
        return (ex - e_x) / (ex + e_x)

    def backward(self, gradwrtoutput, x):
        gradwrtoutput * (1 - self.forward(x) ** 2)

# ============================================================================================================

class MSELoss(Module):
    def forward(self, pred, target):
        return ((pred - target) ** 2).sum() / (len(pred))

    def backward(self, pred, target):
        return (pred - target) / len(pred) * 2

