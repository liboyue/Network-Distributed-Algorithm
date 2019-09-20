#!/usr/bin/env python
# coding=utf-8
import numpy as np
from .network_optimizer import NetworkOptimizer

class NetworkDANE_quadratic(NetworkOptimizer):
    '''The Network DANE algorithm for qudratic objectives.'''

    def __init__(self, p, n_iters=100, eta=0.1, mu=0.1, x_0=None, W=None, verbose=False):
        super().__init__(p, n_iters, x_0, W, verbose)
        self.eta = eta
        self.mu = mu

    def local_update(self):

        for i in range(self.n_agent):
            self.x[:, i] = self.y[:, i] - self.eta * np.linalg.solve(self.hessian(self.y[:, i], i) + self.mu * np.eye(self.dim), self.s[:, i])

