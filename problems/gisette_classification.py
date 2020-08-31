#!/usr/bin/env python
# coding=utf-8
import numpy as np
from scipy import optimize as opt
from . import LogisticRegression

import os

def NAG(grad, x_0, L, sigma):
    '''Nesterov's Accelerated Gradient Descent for strongly convex functions'''

    x = y = x_0
    root_kappa = np.sqrt(L / sigma)
    if root_kappa < 5:
        L *= 9
        root_kappa *= 3
    r = (root_kappa - 1) / (root_kappa + 1)
    r_1 = 1 + r
    r_2 = r

    while np.linalg.norm(grad(y)) > 1e-15:
        print(np.linalg.norm(grad(y)))
        y_last = y
        y = x - grad(x) / L
        x = r_1*y - r_2*y_last

    return y


class GisetteClassification(LogisticRegression):
    def __init__(self, n_agent, kappa=1, **kwargs):

        super().__init__(n_agent, int(6000/n_agent), 5000, kappa=kappa, noise_ratio=None, **kwargs)
        # self.LAMBDA = 1e-8

    def _generate_data(self):

        def _load(fname):
            print('Loading %s' % fname)
            data_path = os.path.abspath(os.path.expanduser(fname + '.data'))
            with open(data_path) as f:
                data = f.readlines()
            data = np.array([[int(x) for x in line.split()] for line in data], dtype=float)

            label_path = os.path.abspath(os.path.expanduser(fname + '.labels'))
            with open(label_path) as f:
                labels = np.array([int(x) for x in f.read().split()], dtype=float)

            labels[labels < 0] = 0
            # label= np.array([[int(x) for x in line.split()] for line in data])
            return data, labels

        self.X_total, self.Y_total = _load('~/gisette_data/gisette_train')
        self.X_val, self.Y_val = _load('~/gisette_data/gisette_valid')

        # norm = np.linalg.norm(self.X_total, 2) / np.sqrt(self.m_total)
        norm = 6422.51797924869151756866
        self.X_total /= norm + self.LAMBDA
 
        self.X = self.split_data(self.m, self.X_total)
        self.Y = self.split_data(self.m, self.Y_total)

        self.x_min = self.w_min = NAG(self.grad, np.random.randn(self.dim), self.L, self.sigma)

    def validate(self, w):
        Y_hat = self.X_val.dot(w)
        Y_hat[Y_hat >= 0] = 1
        Y_hat[Y_hat < 0] = 0
        return np.mean(Y_hat == self.Y_val)



if __name__ == '__main__':

    import matplotlib.pyplot as plt

    n = 10
    m = 1000
    dim = 10
    noise_ratio = 0.01

    p = GisetteClassification(n, m, balanced=False)
    p.grad_check()
    p.distributed_check()

    p = GisetteClassification(n, m, n_edges=4*n)
    p.grad_check()
    p.distributed_check()
    # p.plot_graph()

    print('w_min = ' + str(p.w_min))
    print('f(w_min) = ' + str(p.f(p.w_min)))
    print('f_0(w_min) = ' + str(p.f(p.w_min, 0)))
    print('|| g(w_min) || = ' + str(np.linalg.norm(p.grad(p.w_min))))
    print('|| g_0(w_min) || = ' + str(np.linalg.norm(p.grad(p.w_min, 0))))

    # plt.show()
