"""Optimization module"""
import needle as ndl
import numpy as np

class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad.data+p.data*self.weight_decay
            if p not in self.u:
                self.u[p] = 0

            self.u[p] = self.momentum*self.u[p] + \
                (1-self.momentum)*g

            p.data = p.data-self.lr*self.u[p]

        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        total_norm = np.linalg.norm(np.array([np.linalg.norm(p.grad.detach().numpy()).reshape((1,)) for p in self.params]))
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = min((np.asscalar(clip_coef), 1.0))
        for p in self.params:
            p.grad = p.grad.detach() * clip_coef_clamped


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad.data+p.data*self.weight_decay
            if p not in self.m:
                self.m[p] = 0
            if p not in self.v:
                self.v[p] = 0

            self.m[p] = self.beta1*self.m[p] + \
                (1-self.beta1)*g
            self.v[p] = self.beta2*self.v[p]+(1-self.beta2)*g*g

            u_hat = self.m[p]/(1-self.beta1**self.t)
            v_hat = self.v[p]/(1-self.beta2**self.t)

            p.data = p.data-self.lr*u_hat/(v_hat**(0.5)+self.eps)
            #p.data = p.data-self.lr*self.m[p]/(self.v[p]**(0.5)+self.eps)
            
        ### END YOUR SOLUTION
