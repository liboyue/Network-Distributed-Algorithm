#!/usr/bin/env python
# coding=utf-8
import numpy as np
from ..utils import NAG, GD
from .network_optimizer import NetworkOptimizer

class NetworkDANE(NetworkOptimizer):
    '''The Network DANE algorithm, https://arxiv.org/abs/1909.05844'''

    def __init__(self, p, n_iters=100, eta=0.1, mu=0.1, x_0=None, W=None, local_n_iters=100, local_optimizer='NAG', delta=None, verbose=False):
        super().__init__(p, n_iters, x_0, W, verbose)
        self.eta = eta
        self.mu = mu
        self.local_optimizer = local_optimizer
        self.local_n_iters = local_n_iters
        self.delta = delta


    def local_update(self):

        for i in range(self.n_agent):

            grad_y = self.p.grad(self.y[:, i], i)

            def _grad(tmp):
                return self.grad(tmp, i) - grad_y + self.eta * self.s[:, i] + self.mu * (tmp - self.y[:, i])

            if self.local_optimizer == "NAG":
                self.x[:, i], count = NAG(_grad, self.y[:, i].copy(), self.L + self.mu, self.sigma + self.mu, self.local_n_iters)
            else:
                if self.delta is not None:
                    self.x[:, i], count = GD(_grad, self.y[:, i].copy(), self.delta, self.local_n_iters)
                else:
                    self.x[:, i], count = GD(_grad, self.y[:, i].copy(), 2/(self.L + self.mu + self.sigma + self.mu), self.local_n_iters)
