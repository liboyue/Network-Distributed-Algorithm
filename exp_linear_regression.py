#!/usr/bin/env python
# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt

from problems import LinearRegression
from optimizers import *
from utils import run_exp
from optimizers.utils import generate_mixing_matrix

import networkx as nx

n_agent = 20
m = 1000
dim = 40



kappa = 10000
mu = 5e-4


kappa = 10
mu = 5e-10

n_iters = 100


p = LinearRegression(n_agent, m, dim, noise_variance=1, kappa=kappa, prob=0.3)

print(p.n_edges)


x_0 = np.random.rand(dim, n_agent)
W = generate_mixing_matrix(p.G)
W_s = (W + np.eye(n_agent)) / 2
alpha = np.linalg.norm(W_s - np.ones((n_agent, n_agent))/n_agent, 2)
print('alpha = ' + str(alpha))


eta_2 = 2 / (p.L + p.sigma)
eta_1 = 1 / p.L

n_inner_iters = int(m * 0.05)
n_svrg_iters = n_iters * 20
n_dgd_iters = n_iters * 20
batch_size = int(m / 10)
n_dsgd_iters = int(n_iters * m / batch_size)

exps = [
    DGD_tracking(p, n_iters=n_dgd_iters, eta=eta_2/20, x_0=x_0, W=W),
    EXTRA(p, n_iters=n_dgd_iters, eta=eta_2/4, x_0=x_0, W=W),
    NetworkSVRG(p, n_iters=n_svrg_iters, n_inner_iters=n_inner_iters, eta=eta_2/20, mu=mu, x_0=x_0, W=W, opt=1, batch_size=1),
    NetworkSARAH(p, n_iters=n_svrg_iters, n_inner_iters=n_inner_iters, eta=eta_2/20, mu=mu, x_0=x_0, W=W),
    NetworkDANE(p, n_iters=n_iters, eta=1, mu=mu, x_0=x_0, W=W),
    ADMM(p, n_iters=n_iters, rho=1, x_0=x_0.mean(axis=1)),
    DANE(p, n_iters=n_iters, eta=1, mu=mu, x_0=x_0.mean(axis=1))
    ]


res = run_exp(exps, kappa=kappa, max_iter=n_iters, name='linear_regression')

plt.show()

