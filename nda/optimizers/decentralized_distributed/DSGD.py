#!/usr/bin/env python
# coding=utf-8
try:
    import cupy as xp
except ImportError:
    import numpy as xp

from nda.optimizers import Optimizer


class DSGD(Optimizer):
    '''The Decentralized SGD (D-PSGD) described in https://arxiv.org/pdf/1808.07576.pdf'''

    def __init__(self, p, batch_size=1, eta=0.1, diminishing_step_size=False, **kwargs):

        super().__init__(p, **kwargs)
        self.eta = eta
        self.batch_size = batch_size
        self.diminishing_step_size = diminishing_step_size

    def update(self):
        self.comm_rounds += 1

        if self.diminishing_step_size is True:
            delta_t = self.eta / self.t
        else:
            delta_t = self.eta

        samples = xp.random.randint(0, self.p.m, (self.p.n_agent, self.batch_size))
        grad = self.grad(self.x, j=samples)
        self.x = self.x.dot(self.W) - delta_t * grad
        # self.x -= delta_t * grad
        # self.x = self.x.dot(self.W)
