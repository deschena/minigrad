import torch
from torch import empty
from math import e

class Module(object):

    def __init__(self):
        self.grad = None

    def forward(self, input):
        raise NotImplementedError

    def backward(self, gradwrtoutput):
        raise NotImplementedError

    def zero_grad(self):
        self.grad = None

    def __call__(self, input):
        return self.forward(input)

    @property
    def params(self):
        return []

    @params.setter
    def params(self, new_value):
        pass
# ============================================================================================================

class Linear(Module):
    def __init__(self, input_features, output_features):
        super(Linear, self).__init__()

        # Xavier initialization
        self.weights = torch.empty(size=(output_features, input_features), requires_grad=False).uniform_(- 1, 1) * (input_features) ** -0.5
        self.bias = torch.zeros(output_features, requires_grad=False)

    def forward(self, x):
        return self.weights @ x + self.bias 

    def backward(self, gradwrtoutput, x):
        dl_dw = gradwrtoutput.view(-1, 1) @ x.view(1, -1)
        dl_db = gradwrtoutput
        if self.grad is None:
            self.grad = [dl_dw, dl_db]
        else:
            # accumulate gradient
            self.grad[0] += dl_dw
            self.grad[1] += dl_db 
        # Return gradient wrt to input, used by parent layer (before activation)
        return self.weights.T @ gradwrtoutput

    @property
    def params(self):
        return [self.weights, self.bias]
# ============================================================================================================

class Sequential(Module):
    def __init__(self, *args):
        super().__init__()
        self.modules = args
        
    def forward(self, x):
        y = x
        for m in self.modules:
            y = m.forward(y)
        return y

    def backward(self, gradwrtoutput, x):
        forward_values = [x]
        y = x
        for m in self.modules:
            y = m.forward(y)
            forward_values.append(y)
        back_grad = gradwrtoutput
        
        for mod, layer_input in zip(self.modules[::-1], forward_values[-2::-1]):
            back_grad = mod.backward(back_grad, layer_input)

        return back_grad

    @property
    def grad(self):
        _grad = []
        for m in self.modules:
            if m.grad != None:
                _grad += m.grad
        return _grad

    @grad.setter
    def grad(self, new_value):
        pass
    
    @property
    def params(self):
        _params = []
        for m in self.modules:
            if m.params != []:
                _params += m.params
        return _params
    def zero_grad(self):
        for m in self.modules:
            m.zero_grad()
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

class Loss(object):

    def forward(self, pred, target):
        raise NotImplementedError

    def backward(self, pred, target):
        raise NotImplementedError

    def __call__(self, pred, target):
        return self.forward(pred, target)
        
    @property
    def params(self):
        return []
# ============================================================================================================

class MSELoss(Loss):
    def forward(self, pred, target):
        return ((pred - target) ** 2).sum() / (len(pred))

    def backward(self, pred, target):
        return (pred - target) / len(pred) * 2