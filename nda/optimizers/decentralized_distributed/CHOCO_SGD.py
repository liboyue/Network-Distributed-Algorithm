#!/usr/bin/env python
# coding=utf-8
try:
    import cupy as np
except ImportError:
    import numpy as np

from nda.optimizers import Optimizer
from nda.optimizers import compressor


class CHOCO_SGD(Optimizer):
    '''Decentralized Stochastic Optimization and Gossip Algorithms with Compressed Communication'''

    def __init__(self, p, eta=0.1, gamma=0.1, batch_size=1, compressor_type=None, compressor_param=None, **kwargs):

        super().__init__(p, **kwargs)
        self.eta = eta
        self.gamma = gamma
        self.batch_size = batch_size

        # Compressor
        self.compressor_param = compressor_param
        if compressor_type == 'top':
            self.Q = compressor.top
        elif compressor_type == 'random':
            self.Q = compressor.random
        elif compressor_type == 'gsgd':
            self.Q = compressor.gsgd
        else:
            self.Q = compressor.identity

        self.x_hat = np.zeros_like(self.x)
        self.W_shifted = self.W - np.eye(self.p.n_agent)

    def update(self):
        self.comm_rounds += 1

        samples = np.random.randint(0, self.p.m, (self.p.n_agent, self.batch_size))
        grad = self.grad(self.x, j=samples)

        self.x -= self.eta * grad
        self.x_hat += self.Q(self.x - self.x_hat, self.compressor_param)
        self.x += self.gamma * self.x_hat.dot(self.W_shifted)
