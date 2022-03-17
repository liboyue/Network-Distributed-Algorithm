#!/usr/bin/env python
# coding=utf-8
try:
    import cupy as np
except ImportError:
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
        u = self.y.copy()
        v = self.s.copy()

        if self.opt == 1:
            n_inner_iters = self.n_inner_iters
        else:
            # Choose random x^{(t)} from n_inner_iters
            n_inner_iters = np.random.randint(1, self.n_inner_iters + 1)
            if type(n_inner_iters) is np.ndarray:
                n_inner_iters = n_inner_iters.item()

        for _ in range(n_inner_iters):
            u_last = u.copy()
            u -= self.eta * v

            sample_list = np.random.randint(0, self.p.m, (self.p.n_agent, self.batch_size))

            v += self.grad(u, j=sample_list) - self.grad(u_last, j=sample_list) \
                + self.mu * (u - self.y)

        self.x = u
