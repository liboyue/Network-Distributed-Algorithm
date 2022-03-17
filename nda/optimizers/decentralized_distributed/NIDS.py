#!/usr/bin/env python
# coding=utf-8
try:
    import cupy as np
except ImportError:
    import numpy as np

from nda.optimizers import Optimizer


class NIDS(Optimizer):
    '''A Decentralized Proximal-Gradient Method with Network Independent Step-sizes and Separated Convergence Rates, https://arxiv.org/abs/1704.07807'''

    def __init__(self, p, eta=0.1, **kwargs):
        super().__init__(p, **kwargs)
        self.eta = eta
        self.tilde_W = (self.W + np.eye(self.p.n_agent)) / 2
        self.grad_last = None

    def update(self):
        self.comm_rounds += 1

        if self.t == 1:
            self.grad_last = self.grad(self.x)
            tmp = self.x - self.eta * self.grad_last

        else:
            tmp = 2 * self.x - self.x_last
            tmp += self.eta * self.grad_last
            self.grad_last = self.grad(self.x)
            tmp -= self.eta * self.grad_last
            tmp = tmp.dot(self.tilde_W)

        # Update variables
        self.x, self.x_last = tmp, self.x
