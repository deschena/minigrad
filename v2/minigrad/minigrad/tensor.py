import numpy as np
from typing import List

class Tensor:
    def __init__(
            self,
            data: np.ndarray,
    ):
        self._data = data
        self.grad = None
        self._backward = None

    @staticmethod
    def __check_grad__(tensor):
        if tensor.grad is None:
            tensor.grad = np.zeros_like(tensor._data)

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array([other]))
        new_data = self._data + other._data

        def _backward(original_tensor):
            Tensor.__check_grad__(self)
            Tensor.__check_grad__(other)

    def __sub__(self, other):
        pass

    def __mul__(self, other):
        pass

    def __truediv__(self, other):
        pass

    def __pow__(self, power, modulo=None):
        pass

    def __neg__(self):
        pass

    @property
    def shape(self):
        return self._data.shape

    def backward(self):
        assert self.shape == (1,), "Can only differentiate 1 dimensional expressions"
        reversed_tensor_list = sort_for_backward(self)

        for tensor in reversed_tensor_list:
            # TODO finish implementation
            pass


def sort_for_backward(tensor: Tensor) -> List[Tensor]:
    tensors_list = []
    visit(tensor, tensors_list)
    return tensors_list

def visit(tensor: Tensor, tensor_list: List[Tensor]):
    if tensor._parents is None:
        return
    for parent in tensor._parents:
        visit(parent, tensor_list)

    tensor_list.append(tensor)

class Operation:
        pass