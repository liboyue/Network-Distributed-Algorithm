#!/usr/bin/env python
# coding=utf-8
import numpy as np

from .centralized_optimizer import CentralizedOptimizer


class GD(CentralizedOptimizer):
    '''The vanilla GD'''

    def __init__(self, p, eta=0.1, **kwargs):
        super().__init__(p, **kwargs)
        self.eta = eta


    def update(self):
        self.x -= self.eta * self.grad(self.x)
