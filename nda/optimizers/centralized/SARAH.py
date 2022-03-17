#!/usr/bin/env python
# coding=utf-8
try:
    import cupy as xp
except ImportError:
    import numpy as xp

from nda.optimizers import Optimizer


class SARAH(Optimizer):
    '''The SARAH algorithm'''

    def __init__(self, p, n_inner_iters=20, batch_size=1, eta=0.01, opt=1, **kwargs):
        super().__init__(p, is_distributed=False, **kwargs)
        self.eta = eta
        self.n_inner_iters = n_inner_iters
        self.opt = opt
        self.batch_size = batch_size

    def update(self):
        v = self.grad(self.x)
        x_last = self.x.copy()
        self.x -= self.eta * v

        if self.opt == 1:
            n_inner_iters = self.n_inner_iters
        else:
            # Choose random stopping point from [1, n_inner_iters]
            n_inner_iters = xp.random.randint(1, self.n_inner_iters + 1)
            if type(n_inner_iters) is xp.ndarray:
                n_inner_iters = n_inner_iters.item()

        sample_list = xp.random.randint(0, self.p.m_total, (n_inner_iters, self.batch_size))
        for i in range(n_inner_iters - 1):
            v += self.grad(self.x, j=sample_list[i]) - self.grad(x_last, j=sample_list[i])
            x_last = self.x.copy()
            self.x -= self.eta * v
