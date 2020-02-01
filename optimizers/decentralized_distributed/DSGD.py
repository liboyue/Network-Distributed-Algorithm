#!/usr/bin/env python
# coding=utf-8
import numpy as np

from .decentralized_optimizer import DecentralizedOptimizer


class DSGD(DecentralizedOptimizer):
    '''The Decentralized SGD (D-PSGD) described in https://arxiv.org/pdf/1808.07576.pdf'''

    def __init__(self, p, batch_size=1, eta=0.1, diminish=False, **kwargs):

        super().__init__(p, **kwargs)
        self.eta = eta
        self.batch_size = batch_size
        self.diminish = diminish


    def update(self):
        self.n_comm[self.t] += 1

        if self.diminish == True:
            delta_t = self.eta / self.t
        else:
            delta_t = self.eta

        tmp = self.x.dot(self.W)

        for i in range(self.p.n_agent):
            k_list = np.random.randint(0, self.p.m[i], self.batch_size)
            grad = self.grad(self.x[:, i], i, k_list)
            tmp[:, i] -= delta_t * grad

        self.x = tmp
