#!/usr/bin/env python
# coding=utf-8
import numpy as np
from .network_optimizer import NetworkOptimizer

class NetworkSVRG(NetworkOptimizer):
    def __init__(self, p, n_iters=100, n_inner_iters=100, eta=0.1, opt=1, x_0=None, W=None, batch_size=1, verbose=False):
        super().__init__(p, n_iters, x_0, W, verbose)
        self.eta = eta
        self.opt = opt
        self.n_inner_iters = n_inner_iters
        self.batch_size = batch_size

    def local_update(self):
        for i in range(self.n_agent):
            u = self.y[:, i].copy()
            v = self.s[:, i]

            if self.opt == 1:
                inner_iters = self.n_inner_iters
            else:
                # Choose random x^{(t)} from n_inner_iters
                inner_iters = np.random.randint(1, self.n_inner_iters+1)

            for _ in range(inner_iters):
                u -= self.eta * v
                v = 0
                for j in range(self.batch_size):
                    k = np.random.randint(self.m)
                    v += self.grad(u, i, k) - self.grad(self.y[:, i], i, k)
                v /= self.batch_size
                v += self.s[:, i]

            self.x[:, i] = u
