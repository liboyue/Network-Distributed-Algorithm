#!/usr/bin/env python
# coding=utf-8
try:
    import cupy as xp
except ImportError:
    import numpy as xp

import numpy as np

from nda.optimizers.utils import NAG, GD, FISTA
from nda.optimizers import Optimizer


class ADMM(Optimizer):
    '''ADMM for consensus optimization described in http://www.princeton.edu/~yc5/ele522_optimization/lectures/ADMM.pdf'''

    def __init__(self, p, rho=0.1, local_n_iters=100, delta=None, local_optimizer='NAG', **kwargs):
        super().__init__(p, **kwargs)
        self.rho = rho
        self.local_optimizer = local_optimizer
        self.local_n_iters = local_n_iters
        self.delta = delta
        self.Lambda = np.random.rand(self.p.dim, self.p.n_agent)

    def update(self):
        self.comm_rounds += 1

        x = xp.random.rand(self.p.dim, self.p.n_agent)
        z = self.x.copy()  # Using notations from the tutorial

        for i in range(self.p.n_agent):

            # Non-smooth problems
            if self.p.is_smooth is not True:

                def _grad(tmp):
                    return self.grad_f(tmp, i) + self.rho / 2 * (tmp - z) + self.Lambda[:, i] / 2

                x[:, i], count = FISTA(_grad, self.x.copy(), self.p.L + self.rho, self.p.r, n_iters=self.local_n_iters, eps=1e-10)
            else:

                def _grad(tmp):
                    return self.grad(tmp, i) + self.rho / 2 * (tmp - z) + self.Lambda[:, i] / 2

                if self.local_optimizer == "NAG":
                    x[:, i], _ = NAG(_grad, self.x.copy(), self.p.L + self.rho, self.p.sigma + self.rho, self.local_n_iters)
                else:
                    if self.delta is not None:
                        x[:, i], _ = GD(_grad, self.x.copy(), self.delta, self.local_n_iters)
                    else:
                        x[:, i], _ = GD(_grad, self.x.copy(), 2 / (self.p.L + self.rho + self.p.sigma + self.rho), self.local_n_iters)

        z = (x + self.Lambda).mean(axis=1)

        for i in range(self.p.n_agent):
            self.Lambda[:, i] += self.rho * (x[:, i] - self.x)

        self.x = z
