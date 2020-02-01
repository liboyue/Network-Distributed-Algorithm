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
        for i in range(self.p.n_agent):
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

                k_list = np.random.randint(0, self.p.m[i], self.batch_size)
                v += self.grad(u, i, k_list) - self.grad(u_last, i, k_list) \
                        + self.mu * (u - self.y[:, i])

            self.x[:, i] = u
