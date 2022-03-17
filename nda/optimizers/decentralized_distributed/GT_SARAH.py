#!/usr/bin/env python
# coding=utf-8
try:
    import cupy as xp
except ImportError:
    import numpy as xp

from nda.optimizers import Optimizer


class GT_SARAH(Optimizer):
    '''A near-optimal stochastic gradient method for decentralized non-convex finite-sum optimization, https://arxiv.org/abs/2008.07428'''

    def __init__(self, p, n_inner_iters=100, eta=0.1, batch_size=1, **kwargs):
        super().__init__(p, **kwargs)
        self.eta = eta
        self.n_inner_iters = n_inner_iters
        self.batch_size = batch_size

        self.v = np.zeros((self.p.dim, self.p.n_agent))
        self.y = np.zeros((self.p.dim, self.p.n_agent))

    def update(self):

        self.v_last = self.v
        self.x_last = self.x

        self.v = self.grad(self.x)
        self.y = self.y.dot(self.W) + self.v - self.v_last
        self.x = self.x.dot(self.W) - self.eta * self.y
        self.comm_rounds += 1

        samples = xp.random.randint(0, self.p.m, (self.n_inner_iters, self.p.n_agent, self.batch_size))
        for inner_iter in range(self.n_inner_iters):

            self.v_last = self.v
            self.x_last = self.x

            self.v = self.v + self.grad(self.x, j=samples[inner_iter]) - self.grad(self.x_last, j=samples[inner_iter])

            self.y = self.y.dot(self.W) + self.v - self.v_last
            self.x = self.x.dot(self.W) - self.eta * self.y
            self.comm_rounds += 1

            if inner_iter < self.n_inner_iters - 1:
                self.save_metrics()
