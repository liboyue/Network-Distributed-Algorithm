#!/usr/bin/env python
# coding=utf-8
import numpy as np

from .decentralized_optimizer import DecentralizedOptimizer

class DGD_tracking(DecentralizedOptimizer):
    '''The distributed gradient descent algorithm with gradient tracking, described in 'Harnessing Smoothness to Accelerate Distributed Optimization', Guannan Qu, Na Li'''

    def __init__(self, p, eta=0.1, **kwargs):
        super().__init__(p, **kwargs)
        self.eta = eta
        self.grad_last = None


    def init(self):
        super().init()
        self.s = np.zeros((self.p.dim, self.p.n_agent))
        for i in range(self.p.n_agent):
            self.s[:, i] = self.grad(self.x[:, i], i)

        self.grad_last = self.s.copy()


    def update(self):
        self.n_comm[self.t] += 1

        x_last = self.x.copy()
        y = self.x.dot(self.W)
        self.x = y - self.eta * self.s

        self.s = self.s.dot(self.W_s)
        # for i in range(self.p.n_agent):
            # self.s[:, i] += self.grad(self.x[:, i], i) - self.p.grad(x_last[:, i], i) # Don't count the last gradient evaluation!
        self.s -= self.grad_last
        for i in range(self.p.n_agent):
            self.grad_last[:, i] = self.grad(self.x[:, i], i)
        self.s += self.grad_last
            
