#!/usr/bin/env python
# coding=utf-8
import numpy as np
from ..optimizer import Optimizer

class SVRG(Optimizer):
    '''The SVRG algorithm'''

    def __init__(self, p, n_inner_iters=20, batch_size=1, eta=0.01, opt=1, **kwargs):
        super().__init__(p, **kwargs) 
        self.eta = eta
        self.n_inner_iters = n_inner_iters
        self.opt = opt

    def update(self):

        mu = self.grad(self.x)
        u = self.x.copy()
 
        if self.opt == 1:
            inner_iters = self.n_inner_iters
        else:
            # Choose random x^{(t)} from n_inner_iters
            inner_iters = np.random.randint(1, self.n_inner_iters+1)

        for i in range(inner_iters - 1):
            k_list = np.random.randint(0, self.p.m_total)
            v = self.grad_full(u, k_list) - self.grad_full(self.x, k_list) + mu
            u -= self.eta * v

        # Update
        self.x = u
