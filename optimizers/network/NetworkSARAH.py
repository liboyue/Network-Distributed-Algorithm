#!/usr/bin/env python
# coding=utf-8
import numpy as np
from .network_optimizer import NetworkOptimizer

class NetworkSARAH(NetworkOptimizer):
    def __init__(self, p, n_inner_iters=100, eta=0.1, mu=0, opt=1, batch_size=1, **kwargs):
        super().__init__(p, **kwargs)
        self.eta = eta
        self.opt = opt
        self.mu = mu
        self.n_inner_iters = n_inner_iters
        self.batch_size = batch_size

    def local_update(self):
        for i in range(self.n_agent):
            u = self.y[:, i].copy()
            v = self.s[:, i].copy()

            if self.opt == 1:
                inner_iters = self.n_inner_iters
            else:
                # Choose random x^{(t)} from n_inner_iters
                inner_iters = np.random.randint(1, self.n_inner_iters+1)

            for _ in range(inner_iters):
                u_last = u.copy()
                u -= self.eta * v
                grad = 0
                for j in range(self.batch_size):
                    k = np.random.randint(self.m)
                    grad += self.grad(u, i, k) - self.grad(u_last, i, k) + self.mu * (u - u_last)
                grad /= self.batch_size
                v += grad

            self.x[:, i] = u
