#!/usr/bin/env python
# coding=utf-8
import numpy as np

from ..optimizer import Optimizer

class CentralizedOptimizer(Optimizer):
    '''The base centralized optimizer class, which handles logging, convergence/divergence checking.'''

    def __init__(self, p, **kwargs):

        super().__init__(p, **kwargs)
        self.n_comm = np.zeros(self.n_iters+1)


    def get_results(self):
        res = {
                'x': self.x,
                'var_error': self.var_error[:self.t+1],
                'func_error': self.func_error[:self.t+1],
                'n_comm': self.n_comm[:self.t+1],
                'n_grad': self.n_grad[:self.t+1] / self.p.m_total
                }

        if self.verbose == True:
            res['history'] = self.history

        return res



    def init_iter(self):
        super().init_iter()
        self.n_comm[self.t] += self.n_comm[self.t-1]
