#!/usr/bin/env python
# coding=utf-8
import numpy as np
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
            inner_iters = self.n_inner_iters
        else:
            # Choose random stopping point from [1, n_inner_iters]
            inner_iters = np.random.randint(1, self.n_inner_iters + 1)

        sample_list = np.random.randint(0, self.p.m_total, (inner_iters, self.batch_size))
        for i in range(inner_iters - 1):
            v += self.grad(self.x, j=sample_list[i]) - self.grad(x_last, j=sample_list[i])
            x_last = self.x.copy()
            self.x -= self.eta * v
