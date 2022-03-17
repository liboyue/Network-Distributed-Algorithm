#!/usr/bin/env python
# coding=utf-8
from nda.optimizers.utils import NAG, GD, FISTA
from .network_optimizer import NetworkOptimizer


class NetworkDANE(NetworkOptimizer):

    def __init__(self, p, mu=0.1, local_n_iters=100, local_optimizer='NAG', delta=None, **kwargs):
        super().__init__(p, **kwargs)
        self.mu = mu
        self.local_optimizer = local_optimizer
        self.local_n_iters = local_n_iters
        self.delta = delta

    def local_update(self):

        for i in range(self.p.n_agent):

            if self.p.is_smooth is False:
                grad_y = self.p.grad_f(self.y[:, i], i)

                def _grad(tmp):
                    return self.grad_f(tmp, i) - grad_y + self.s[:, i] + self.mu * (tmp - self.y[:, i])
                self.x[:, i], count = FISTA(_grad, self.y[:, i].copy(), self.p.L + self.mu, self.p.r, n_iters=self.local_n_iters, eps=1e-10)
            else:
                grad_y = self.p.grad(self.y[:, i], i)

                def _grad(tmp):
                    return self.grad(tmp, i) - grad_y + self.s[:, i] + self.mu * (tmp - self.y[:, i])

                if self.local_optimizer == 'NAG':
                    self.x[:, i], count = NAG(_grad, self.y[:, i].copy(), self.p.L + self.mu, self.p.sigma + self.mu, self.local_n_iters)
                else:
                    if self.delta is not None:
                        self.x[:, i], count = GD(_grad, self.y[:, i].copy(), self.delta, self.local_n_iters)
                    else:
                        self.x[:, i], count = GD(_grad, self.y[:, i].copy(), 2 / (self.p.L + self.mu + self.p.sigma + self.mu), self.local_n_iters)
