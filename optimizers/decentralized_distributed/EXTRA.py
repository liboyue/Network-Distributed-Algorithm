#!/usr/bin/env python
# coding=utf-8
import numpy as np

from .decentralized_optimizer import DecentralizedOptimizer


class EXTRA(DecentralizedOptimizer):
    '''EXTRA: AN EXACT FIRST-ORDER ALGORITHM FOR DECENTRALIZED CONSENSUS OPTIMIZATION, https://arxiv.org/pdf/1404.6264.pdf'''

    def __init__(self, p, eta=0.1, **kwargs):
        super().__init__(p, **kwargs)
        self.eta = eta
        self.grad_last = None


    def update(self):
        self.n_comm[self.t] += 1

        if self.t == 1:
            tmp = self.x.dot(self.W)
            self.grad_last = np.zeros((self.p.dim, self.p.n_agent))
            # for i in range(self.p.n_agent):
                # tmp[:, i] -= self.eta * self.grad(self.x[:, i], i)
            for i in range(self.p.n_agent):
                self.grad_last[:, i] -= self.grad(self.x[:, i], i)
            tmp -= self.eta * self.grad_last

        else:
            tmp = self.x.dot(self.W) + self.x - self.x_last.dot(self.W_s)
            # for i in range(self.p.n_agent):
                # tmp[:, i] -= self.eta * (self.grad(self.x[:, i], i) - self.p.grad(self.x_last[:, i], i))
            tmp += self.eta * self.grad_last
            for i in range(self.p.n_agent):
                self.grad_last[:, i] = self.grad(self.x[:, i], i)
            tmp -= self.eta * self.grad_last

        # Update variables
        self.x, self.x_last = tmp, self.x
