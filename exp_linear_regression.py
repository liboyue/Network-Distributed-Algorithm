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
n_iters = 100

# Well-conditioned experiment
kappa = 10
mu = 5e-10

# Ill-conditioned experiment
kappa = 10000
mu = 5e-6

p = LinearRegression(n_agent, m, dim, noise_variance=1, kappa=kappa, prob=0.3)


x_0 = np.random.rand(dim, n_agent)
x_0_mean = x_0.mean(axis=1)
W, alpha = generate_mixing_matrix(p)
print('alpha = ' + str(alpha))


eta_1 = 1 / p.L
eta_2 = 2 / (p.L + p.sigma)

n_inner_iters = int(m * 0.05)
n_svrg_iters = n_iters *  30
n_dgd_iters = n_iters * 30
batch_size = 5


distributed = [
    DGD_tracking(p, n_iters=n_dgd_iters, eta=eta_1/10, x_0=x_0, W=W),
    EXTRA(p, n_iters=n_dgd_iters, eta=eta_1/2, x_0=x_0, W=W),
    ADMM(p, n_iters=n_iters, rho=1, x_0=x_0.mean(axis=1)),
    DANE(p, n_iters=n_iters, mu=mu, x_0=x_0.mean(axis=1)),
    ]

network = [
    NetworkSVRG(p, n_iters=n_svrg_iters, n_inner_iters=n_inner_iters, eta=eta_1/10, x_0=x_0, W=W, opt=1),
    NetworkSARAH(p, n_iters=n_svrg_iters, n_inner_iters=n_inner_iters, eta=eta_1/10, x_0=x_0, W=W, opt=1),
    NetworkDANE(p, n_iters=n_iters, mu=mu, x_0=x_0, W=W),
    ]

exps = distributed + network

res = run_exp(exps, kappa=kappa, max_iter=n_iters, name='linear_regression', n_process=4, save=True)

plt.show()
