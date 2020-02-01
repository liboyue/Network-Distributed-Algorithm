#!/usr/bin/env python
# coding=utf-8
import numpy as np
from numpy.linalg import norm

from ..centralized_distributed import CentralizedOptimizer
from ..utils import eps, relative_error


class DecentralizedOptimizer(CentralizedOptimizer):
    '''The base decentralized optimizer class, which handles logging, convergence/divergence checking.'''

    def __init__(self, p, W=None, x_0=None, **kwargs):

        super().__init__(p, x_0=x_0, **kwargs)

        self.x_0 = x_0 if x_0 is not None else np.random.rand(p.dim, p.n_agent)
        self.x = self.x_0.copy()

        if W is not None:
            self.W = W

            W_min_diag = min(np.diag(self.W))
            tmp = (1 - 1e-1) / (1 - W_min_diag)
            self.W_s = self.W*tmp + np.eye(self.p.n_agent) * (1 - tmp)
        

    def save_metric(self):
        self.func_error[self.t] = (self.f(self.x.mean(axis=1)) - self.p.f_min) / self.p.f_min
        self.var_error[self.t] = relative_error(self.x.mean(axis=1), self.p.x_min)


    def convergence_check(self):
        ''' Convergence check'''

        if norm(self.p.grad(self.x.mean(axis=1))) < eps:
            return True

        if norm(self.x.mean(axis=1) - self.p.x_min) > 5e1:
            return True
