#!/usr/bin/env python
# coding=utf-8
import numpy as np
from ..optimizer import Optimizer

class GD(Optimizer):
    '''The vanilla GD'''

    def __init__(self, p, eta=0.1, **kwargs):
        super().__init__(p, **kwargs)
        self.eta = eta


    def update(self):
        self.x -= self.eta * self.grad_full(self.x)
