#!/usr/bin/env python
# coding=utf-8
import numpy as np

try:
    import cupy as xp
except ImportError:
    import numpy as xp

from nda.optimizers import Optimizer


def T(x, k):

    if k == 0:
        if type(x) is np.ndarray:
            return np.eye(x.shape[0])
        else:
            return 1

    if type(x) is np.ndarray:
        prev = np.eye(x.shape[0])
    else:
        prev = 1

    current = x
    for _ in range(k - 1):
        current, prev = 2 * np.dot(x, current) - prev, current

    return current


class Destress(Optimizer):
    def __init__(self, p, n_mix=1, n_inner_iters=100, eta=0.1, K_in=1, K_out=1, batch_size=1, opt=0, perturbation_threshould=None, perturbation_radius=None, perturbation_variance=None, **kwargs):
        super().__init__(p, **kwargs)

        self.K_in = K_in
        self.K_out = K_out

        self.eta = eta
        self.opt = opt
        self.n_inner_iters = n_inner_iters
        self.batch_size = batch_size

        average_matrix = np.ones((self.p.n_agent, self.p.n_agent)) / self.p.n_agent
        alpha = np.linalg.norm(self.W - average_matrix, 2)
        self.W_in = T(self.W / alpha, self.K_in) / T(1 / alpha, self.K_in)
        self.W_out = T(self.W / alpha, self.K_out) / T(1 / alpha, self.K_out)

    def init(self):

        super().init()

        # Equivalent mixing matrices after n_mix rounds of mixng
        # W_min_diag = min(np.diag(self.W))
        # tmp = (1 - 1e-1) / (1 - W_min_diag)
        # self.W_s = self.W * tmp + np.eye(self.p.n_agent) * (1 - tmp)

        if len(self.x_0.shape) == 2:
            self.x = xp.tile(self.x_0.mean(axis=1), (self.p.n_agent, 1)).T
        else:
            self.x = self.x_0.copy()

        self.grad_last = self.grad(self.x)
        self.s = self.grad_last.copy()
        self.s = xp.tile(self.s.mean(axis=1), (self.p.n_agent, 1)).T

    def local_update(self):
        if self.opt == 1:
            n_inner_iters = self.n_inner_iters
        else:
            # Choose random x^{(t)} from n_inner_iters
            n_inner_iters = xp.random.randint(1, self.n_inner_iters + 1)
            if type(n_inner_iters) is xp.ndarray:
                n_inner_iters = n_inner_iters.item()

        samples = xp.random.randint(0, self.p.m, (n_inner_iters, self.p.n_agent, self.batch_size))

        u = self.x.copy()
        v = self.s.copy()
        for inner_iter in range(n_inner_iters):

            u_last, u = u, (u - self.eta * v).dot(self.W_in)
            self.comm_rounds += self.K_in

            v += self.grad(u, j=samples[inner_iter]) - self.grad(u_last, j=samples[inner_iter])
            v = v.dot(self.W_in)
            self.comm_rounds += self.K_in

            if inner_iter < n_inner_iters - 1:
                self.save_metrics(x=u)

        self.x = u

    def update(self):

        self.local_update()

        self.s -= self.grad_last
        self.grad_last = self.grad(self.x)
        self.s += self.grad_last
        self.s = self.s.dot(self.W_out)
        self.comm_rounds += self.K_out
