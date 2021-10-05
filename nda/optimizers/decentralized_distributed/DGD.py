#!/usr/bin/env python
# coding=utf-8
from nda.optimizers import Optimizer


class DGD(Optimizer):

    def __init__(self, p, eta=0.1, **kwargs):

        super().__init__(p, **kwargs)
        self.eta = eta

    def update(self):
        self.comm_rounds += 1
        self.x = self.x.dot(self.W) - self.eta * self.grad(self.x)
