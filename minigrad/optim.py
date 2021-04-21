class SGD:
    def __init__(self, model, lr, momentum=None):
        self.model = model
        self.lr = lr
        self.momentum = momentum

    def step(self):
        model_grad = self.model.grad
        model_params = self.model.params
        if len(model_grad) != len(model_params):
            raise RuntimeError(f"Gradient and parameters mismatch: {len(model_params)} parameter tensors and {len(model_grad)} gradient tensors.")

        for g, p in zip(model_grad, model_params):
            p -= self.lr * g

    def zero_grad(self):
        self.model.zero_grad()
