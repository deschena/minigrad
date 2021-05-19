class Optimizer:
    # Assumes instances of Optimizer have a model attribute representing the model to optimize
    def zero_grad(self):
        self.model.zero_grad()

class SGD(Optimizer):
    """SGD Optimizer
    """
    def __init__(self, model, lr=0.001, momentum=0.0):
        """Create an optimizer that will update parameters using SGD algorithm

        Args:
            model (nn.Module): Model whose parameters must be learnt
            lr (float, optional): Learning rate / Step size. Defaults to 0.001.
            momentum (float, optional): Momentum of SGD optimizer. Defaults to 0.
        """
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.momentum_ma = None

    def step(self):
        model_grad = self.model.grad
        model_params = self.model.params
        if len(model_grad) != len(model_params):
            raise RuntimeError(f"Gradient and parameters mismatch: {len(model_params)} parameter tensors and {len(model_grad)} gradient tensors.")

        # Create momentum array
        if self.momentum_ma == None:
            self.momentum_ma = []
            for _ in model_grad:
                self.momentum_ma.append(0)
        # Update parameters
        for i, (g, p, v) in enumerate(zip(model_grad, model_params, self.momentum_ma)):
            if self.momentum == 0.0: # Avoid unecessary computations
                v_t = g
            else:
                v_t = self.momentum * v + (1 - self.momentum) * g
                self.momentum_ma[i] = v_t
            p -= self.lr * v_t
# ============================================================================================================

class Adam(Optimizer):
    """Adam Optimizer
    """
    def __init__(self, model, lr=0.001, betas=(0.9, 0.999), eps=1e-08):
        self.model = model
        self.lr = lr
        self.momentum_ma = None # First order moving average
        self.sec_ma = None # Second order moving average
        self.betas = betas
        self.eps = eps

    def step(self):
        model_grad = self.model.grad
        model_params = self.model.params
        if len(model_grad) != len(model_params):
            raise RuntimeError(f"Gradient and parameters mismatch: {len(model_params)} parameter tensors and {len(model_grad)} gradient tensors.")

        b_0, b_1 = self.betas
        if self.momentum_ma == None: # First init, create the running averages for each group of params
            self.momentum_ma = []
            self.sec_ma = []
            for _ in model_grad:
                self.momentum_ma.append(0)
                self.sec_ma.append(0)
        # Update the moving average and update params
        for i, (g, p, m, v) in enumerate(zip(model_grad, model_params, self.momentum_ma, self.sec_ma)):
            m_t = b_0 * m + (1 - b_0) * g
            v_t = b_1 * v + (1 - b_1) * (g ** 2)
            # Save updated mov. avg. params for next iter
            self.momentum_ma[i] = m_t
            self.sec_ma[i] = v_t
            # Actual param update
            p -= self.lr / ( v_t ** 0.5 + self.eps ) * m_t