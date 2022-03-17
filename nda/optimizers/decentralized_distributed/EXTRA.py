#!/usr/bin/env python
# coding=utf-8
try:
    import cupy as xp
except ImportError:
    import numpy as xp

from nda.optimizers import Optimizer


class EXTRA(Optimizer):
    '''EXTRA: AN EXACT FIRST-ORDER ALGORITHM FOR DECENTRALIZED CONSENSUS OPTIMIZATION, https://arxiv.org/pdf/1404.6264.pdf'''

    def __init__(self, p, eta=0.1, **kwargs):
        super().__init__(p, **kwargs)
        self.eta = eta
        self.grad_last = None
        W_min_diag = min(np.diag(self.W))
        tmp = (1 - 1e-1) / (1 - W_min_diag)
        self.W_s = self.W * tmp + np.eye(self.p.n_agent) * (1 - tmp)

    def update(self):
        self.comm_rounds += 1

        if self.t == 1:
            self.grad_last = self.grad(self.x)
            tmp = self.x.dot(self.W) - self.eta * self.grad_last
        else:
            tmp = self.x.dot(self.W) + self.x - self.x_last.dot(self.W_s)
            tmp += self.eta * self.grad_last
            self.grad_last = self.grad(self.x)
            tmp -= self.eta * self.grad_last

        self.x, self.x_last = tmp, self.x
