#!/usr/bin/env python
# coding=utf-8
import numpy as np

from ..utils import NAG, GD
from .centralized_optimizer import CentralizedOptimizer

class DANE(CentralizedOptimizer):
    '''The (inexact) DANE algorithm described in Communication Efficient Distributed Optimization using an Approximate Newton-type Method, https://arxiv.org/abs/1312.7853'''

    def __init__(self, p, mu=0.1, local_n_iters=100, local_optimizer='NAG', delta=None, **kwargs):
        super().__init__(p, **kwargs)
        self.mu = mu
        self.local_optimizer = local_optimizer
        self.local_n_iters = local_n_iters
        self.delta = delta


    def update(self):
        self.n_comm[self.t] += 2

        grad_x = self.grad_full(self.x)

        x_next = 0
        for i in range(self.p.n_agent):

            grad_x_i = self.grad(self.x, i)

            # "Exactly" solve local optimization problem using NAG
            def _grad(tmp):
                return self.grad(tmp, i) - grad_x_i + grad_x + self.mu * (tmp - self.x)

            if self.local_optimizer == "NAG":
                if self.delta is not None:
                    tmp, _ = NAG(_grad, self.x.copy(), self.delta, self.local_n_iters)
                else:
                    tmp, _ = NAG(_grad, self.x.copy(), self.p.L + self.mu, self.p.sigma + self.mu, self.local_n_iters)
            else:
                if self.delta is not None:
                    tmp, _ = GD(_grad, self.x.copy(), self.delta, self.local_n_iters)
                else:
                    tmp, _ = GD(_grad, self.x.copy(), 2/(self.p.L + self.mu + self.p.sigma + self.mu), self.local_n_iters)

            x_next += tmp

        self.x = x_next / self.p.n_agent
