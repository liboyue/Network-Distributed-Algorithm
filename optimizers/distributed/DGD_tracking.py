#!/usr/bin/env python
# coding=utf-8
import numpy as np
from ..optimizer import Optimizer

class DGD_tracking(Optimizer):
    '''The distributed gradient descent algorithm with gradient tracking, described in 'Harnessing Smoothness to Accelerate Distributed Optimization', Guannan Qu, Na Li'''

    def __init__(self, p, n_iters=100, eta=0.1, x_0=None, W=None, auto_param_choosing=False, verbose=False):
        super().__init__(p, n_iters, x_0, W, verbose)

        if auto_param_choosing == True:
            max_eig = np.linalg.norm(W - np.ones((self.n_agent, self.n_agent)) / self.n_agent)
            if hasattr('sigma', p): # Strongly convex
                eta = 2 / (p.L + p.sigma)
            else:
                eta = 1 / p.L

            print('DGD_tracking auto_param_choosing is on')
            print('eta = ' + str(eta) + ', mu = ' + str(mu))

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
