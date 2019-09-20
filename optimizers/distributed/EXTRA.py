#!/usr/bin/env python
# coding=utf-8
import numpy as np
from ..optimizer import Optimizer

class EXTRA(Optimizer):
    '''EXTRA: AN EXACT FIRST-ORDER ALGORITHM FOR DECENTRALIZED CONSENSUS OPTIMIZATION, https://arxiv.org/pdf/1404.6264.pdf'''

    def __init__(self, p, n_iters=100, eta=0.1, x_0=None, W=None, verbose=False):
        super().__init__(p, n_iters, x_0, W, verbose)
        self.eta = eta

    # def init(self):
    #    pass

    def update(self):
        self.n_comm[self.t] += 2*self.n_edges

        if self.t == 1:
            tmp = self.x.dot(self.W)
            for i in range(self.n_agent):
                tmp[:, i] -= self.eta * self.grad(self.x[:, i], i)

        else:
            tmp = self.x.dot(self.W + np.eye(self.n_agent)) - self.x_last.dot(self.W_s)
            for i in range(self.n_agent):
                tmp[:, i] -= self.eta * (self.grad(self.x[:, i], i) - self.p.grad(self.x_last[:, i], i))

        # Update variables
        self.x, self.x_last = tmp, self.x
