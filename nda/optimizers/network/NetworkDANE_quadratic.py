#!/usr/bin/env python
# coding=utf-8
try:
    import cupy as np
except ImportError:
    import numpy as np

from .network_optimizer import NetworkOptimizer


class NetworkDANE_quadratic(NetworkOptimizer):
    '''The Network DANE algorithm for qudratic objectives.'''

    def __init__(self, p, mu=0.1, **kwargs):
        super().__init__(p, **kwargs)
        self.mu = mu

    def local_update(self):

        for i in range(self.p.n_agent):
            self.x[:, i] = self.y[:, i] - np.linalg.solve(self.hessian(self.y[:, i], i) + self.mu * np.eye(self.p.dim), self.s[:, i])
