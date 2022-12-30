#!/usr/bin/env python
# coding=utf-8
try:
    import cupy as xp
except ImportError:
    import numpy as xp

import numpy as np

from nda.optimizers import Optimizer

class D2(Optimizer):
    '''D2: Decentralized Training over Decentralized Data, https://arxiv.org/abs/1803.07068'''

    def __init__(self, p, eta=0.1, batch_size=1, **kwargs):
        super().__init__(p, **kwargs)
        self.eta = eta
        self.grad_last = None
        self.batch_size = batch_size
        self.tilde_W = (self.W + np.eye(self.p.n_agent)) / 2

    def update(self):
        self.comm_rounds += 1

        samples = xp.random.randint(0, self.p.m, (self.p.n_agent, self.batch_size))

        if self.t == 1:
            self.grad_last = self.grad(self.x, j=samples)
            tmp = self.x - self.eta * self.grad_last

        else:
            tmp = 2 * self.x - self.x_last
            tmp += self.eta * self.grad_last
            self.grad_last = self.grad(self.x, j=samples)
            tmp -= self.eta * self.grad_last
            tmp = tmp.dot(self.tilde_W)

        # Update variables
        self.x, self.x_last = tmp, self.x
