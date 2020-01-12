#!/usr/bin/env python
# coding=utf-8
import numpy as np

from ..utils import NAG, GD
from .centralized_optimizer import CentralizedOptimizer

class ADMM(CentralizedOptimizer):
    '''ADMM for consensus optimization described in http://www.princeton.edu/~yc5/ele522_optimization/lectures/ADMM.pdf'''

    def __init__(self, p, rho=0.1, local_n_iters=100, delta=None, local_optimizer='NAG', **kwargs):
        super().__init__(p, **kwargs) 
        self.rho = rho
        self.local_optimizer = local_optimizer
        self.local_n_iters = local_n_iters
        self.Lambda = np.random.rand(self.dim, self.n_agent)
        self.delta = delta

    def update(self):
        self.n_comm[self.t] += 2*self.n_agent

        x = np.random.rand(self.dim, self.n_agent)
        z = self.x # Using notations from the tutorial

        for i in range(self.n_agent):

            def _grad(tmp):
                return self.grad(tmp, i) + self.rho / 2 * (tmp - z) + self.Lambda[:, i] / 2

            if self.local_optimizer == "NAG":
                x[:, i], _ = NAG(_grad, self.x.copy(), self.L + self.rho, self.sigma + self.rho, self.local_n_iters)
            else:
                if self.delta is not None:
                    x[:, i], _ = GD(_grad, self.x.copy(), self.delta, self.local_n_iters)
                else:
                    x[:, i], _ = GD(_grad, self.x.copy(), 2/(self.L + self.rho + self.sigma + self.rho), self.local_n_iters)

        z = (x + self.Lambda).mean(axis=1)
        for i in range(self.n_agent):
            self.Lambda[:, i] += self.rho * (x[:, i] - self.x)
        self.x = z # Update

