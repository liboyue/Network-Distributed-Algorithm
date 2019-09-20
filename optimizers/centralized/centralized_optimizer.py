#!/usr/bin/env python
# coding=utf-8
import numpy as np
from numpy.linalg import norm

from ..optimizer import Optimizer
from ..utils import generate_mixing_matrix, eps, relative_error

class CentralizedOptimizer(Optimizer):
    '''The base centralized optimizer class, which handles logging, convergence/divergence checking.'''

    def __init__(self, p, n_iters=100, x_0=None, W=None, verbose=False):

        x_0 = x_0 if x_0 is not None else np.random.rand(p.dim)
        super().__init__(p, n_iters, x_0, W, verbose)


    def init(self):
        self.x = self.x_0


    def save_metric(self):
        self.func_error[self.t] = (self.f(self.x) - self.f(self.x_min)) / self.f(self.x_min)
        self.var_error[self.t] = relative_error(self.x, self.x_min)     


    def get_results(self):
        return {
                'x': self.x,
                'var_error': self.var_error[:self.t+1],
                'func_error': self.func_error[:self.t+1],
                'n_comm': self.n_comm[:self.t+1],
                'n_grad': self.n_grad[:self.t+1] / self.p.m / self.p.n_agent
                }


    def optimize(self):
        self.init()

        # Initial value
        if hasattr(self, 'x_min'):
            self.save_metric()

        if self.verbose == True:
            self.save_history()

        for self.t in range(1, self.n_iters+1):

            self.update()

            self.n_grad[self.t] += self.n_grad[self.t-1]
            self.n_comm[self.t] += self.n_comm[self.t-1]

            if hasattr(self.p, 'x_min'):
                self.save_metric()

            if self.verbose == True:
                self.save_history()

            if hasattr(self.p, 'x_min') and self.convergence_check() == True:
                break

        # endfor

        return self.get_results()


    def convergence_check(self):
        ''' Convergence check'''

        if norm(self.p.grad(self.x)) < eps:
            return True

        if norm(self.x - self.x_min) > 1e1:
            return True
