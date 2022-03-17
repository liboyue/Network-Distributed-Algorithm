#!/usr/bin/env python
# coding=utf-8
try:
    import cupy as xp
except ImportError:
    import numpy as xp

from nda.optimizers import Optimizer


class SGD(Optimizer):
    '''Stochastic Gradient Descent'''

    def __init__(self, p, batch_size=1, eta=0.1, diminishing_step_size=False, **kwargs):
        super().__init__(p, is_distributed=False, **kwargs)
        self.eta = eta
        self.batch_size = batch_size
        self.diminishing_step_size = diminishing_step_size

    def update(self):

        sample_list = xp.random.randint(0, self.p.m_total, self.batch_size)
        grad = self.grad(self.x, j=sample_list)

        if self.diminishing_step_size is True:
            self.x -= self.eta / self.t * grad
        else:
            self.x -= self.eta * grad
