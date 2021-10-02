

import torch
n = 4
m = 7
k = 11

target = torch.randn((n, k))
A = torch.randn((n, m), requires_grad=True)
B = torch.randn((m, k), requires_grad=True)
U = A @ B

def MSE(pred, target):
    return torch.norm((pred - target) ** 2)

saved = None
def extract(var):
    global saved
    saved = var
U.register_hook(extract)
loss = MSE(U, target)
loss.backward(retain_graph=True)

A_grad = saved @ B.T
B_grad = A.T @ saved

print((A.grad == A_grad).all())
print((B.grad == B_grad).all())


a = 123

func = lambda: a

a = 99

print(func())

a = 9931
print(func())