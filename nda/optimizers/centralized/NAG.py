#!/usr/bin/env python
# coding=utf-8
import numpy as np
from nda import log
from nda.optimizers import Optimizer


class NAG(Optimizer):
    '''The Nesterov's Accelerated GD'''

    def __init__(self, p, **kwargs):
        super().__init__(p, is_distributed=False, **kwargs)

        if self.p.sigma > 0:
            self.update = self.update_strongly_convex
        else:
            log.error('NAG only supports strongly convex')

        if self.p.sigma > 0:
            self.x = self.y = self.x_0
            root_kappa = np.sqrt(self.p.L / self.p.sigma)
            r = (root_kappa - 1) / (root_kappa + 1)
            self.r_1 = 1 + r
            self.r_2 = r

    def update_convex(self):
        pass

    def update_strongly_convex(self):
        x_last = self.x.copy()
        self.x = self.y - self.grad(self.x) / self.p.L
        self.y = self.r_1 * self.x - self.r_2 * x_last
