#!/usr/bin/env python
# coding=utf-8
from nda.optimizers.utils import NAG, GD, FISTA
from nda.optimizers import Optimizer


class DANE(Optimizer):
    '''The (inexact) DANE algorithm described in Communication Efficient Distributed Optimization using an Approximate Newton-type Method, https://arxiv.org/abs/1312.7853'''

    def __init__(self, p, mu=0.1, local_n_iters=100, local_optimizer='NAG', delta=None, **kwargs):
        super().__init__(p, **kwargs)
        self.mu = mu
        self.local_optimizer = local_optimizer
        self.local_n_iters = local_n_iters
        self.delta = delta

    def update(self):
        self.comm_rounds += 2

        grad_x = self.grad_h(self.x)

        x_next = 0
        for i in range(self.p.n_agent):

            if self.p.is_smooth is False:
                grad_x_i = self.grad_h(self.x, i)

                def _grad(tmp):
                    return self.grad_h(tmp, i) - grad_x_i + grad_x + self.mu * (tmp - self.x)
                tmp, count = FISTA(_grad, self.x.copy(), self.mu + 1, self.p.r, n_iters=self.local_n_iters, eps=1e-10)

            else:
                grad_x_i = self.grad_h(self.x, i)

                def _grad(tmp):
                    return self.grad_h(tmp, i) - grad_x_i + grad_x + self.mu * (tmp - self.x)

                if self.local_optimizer == "NAG":
                    if self.delta is not None:
                        tmp, count_ = NAG(_grad, self.x.copy(), self.delta, self.local_n_iters)
                    else:
                        tmp, count_ = NAG(_grad, self.x.copy(), self.p.L + self.mu, self.p.sigma + self.mu, self.local_n_iters)

                else:
                    if self.delta is not None:
                        tmp, count_ = GD(_grad, self.x.copy(), self.delta, self.local_n_iters)
                    else:
                        tmp, count_ = GD(_grad, self.x.copy(), 2 / (self.p.L + self.mu + self.p.sigma + self.mu), self.local_n_iters)

            x_next += tmp

        self.x = x_next / self.p.n_agent
