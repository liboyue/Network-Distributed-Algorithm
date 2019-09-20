#!/usr/bin/env python
# coding=utf-8
import numpy as np

from .centralized_optimizer import CentralizedOptimizer


class GD(CentralizedOptimizer):
    '''The vanilla GD'''

    def __init__(self, p, n_iters=100, eta=0.1, x_0=None, verbose=False):
        super().__init__(p, n_iters, x_0, None, verbose)
        self.eta = eta


    def update(self):
        self.x -= self.eta * self.grad(self.x)
