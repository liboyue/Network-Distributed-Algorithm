#!/usr/bin/env python
# coding=utf-8
import numpy as np
from ..optimizer import Optimizer

class DGD_tracking(Optimizer):
    '''The distributed gradient descent algorithm with gradient tracking, described in 'Harnessing Smoothness to Accelerate Distributed Optimization', Guannan Qu, Na Li'''

    def __init__(self, p, eta=None, **kwargs):
        super().__init__(p, **kwargs)

        if eta != None:
            self.eta = eta
        else:
        # self.W = ( self.W + np.eye(self.n_agent) ) / 2
            if hasattr('sigma', p): # Strongly convex
                self.eta = 2 / (p.L + p.sigma)
            else:
                self.eta = 1 / p.L

            print('NetworkGD chose theoratical largest step size eta = ' + str(self.eta))

        self.eta = eta

    def init(self):
        super().init()
        self.s = np.zeros((self.dim, self.n_agent))
        for i in range(self.n_agent):
            self.s[:, i] = self.grad(self.x[:, i], i)


    def update(self):
        self.n_comm[self.t] += 2*self.n_edges

        x_last = self.x.copy()
        y = self.x.dot(self.W)
        self.x = y - self.eta * self.s

        self.s = self.s.dot(self.W_s)
        for i in range(self.n_agent):
            self.s[:, i] += self.grad(self.x[:, i], i) - self.p.grad(x_last[:, i], i) # Don't count the last gradient evaluation!
