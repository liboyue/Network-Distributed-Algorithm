#!/usr/bin/env python
# coding=utf-8
from nda.optimizers import Optimizer


class DGD_tracking(Optimizer):
    '''The distributed gradient descent algorithm with gradient tracking, described in 'Harnessing Smoothness to Accelerate Distributed Optimization', Guannan Qu, Na Li'''

    def __init__(self, p, eta=0.1, **kwargs):
        super().__init__(p, **kwargs)
        self.eta = eta
        self.grad_last = None

    def init(self):
        super().init()
        self.s = self.grad(self.x)
        self.grad_last = self.s.copy()

    def update(self):
        self.comm_rounds += 2

        self.x = self.x.dot(self.W) - self.eta * self.s
        grad_current = self.grad(self.x)

        self.s = self.s.dot(self.W) + grad_current - self.grad_last
        self.grad_last = grad_current
