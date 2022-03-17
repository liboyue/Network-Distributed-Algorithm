#!/usr/bin/env python
# coding=utf-8
try:
    import cupy as xp
except ImportError:
    import numpy as xp

from nda import log
from nda.optimizers import Optimizer


class GD(Optimizer):
    '''The vanilla GD'''

    def __init__(self, p, eta=0.1, **kwargs):
        super().__init__(p, is_distributed=False, **kwargs)
        self.eta = eta
        if self.p.is_smooth is False:
            log.info('Nonsmooth problem, running sub-gradient descent instead')
            self.update = self.subgd_update
            self.name = 'SubGD'

    def update(self):
        self.x -= self.eta * self.grad(self.x)

    def subgd_update(self):
        self.x -= self.eta / xp.sqrt(self.t) * self.grad(self.x)
