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


kappa = 10
mu = 5e-10


n_iters = 400

while True:
    p = LinearRegression(n_agent, m, dim, noise_variance=1, kappa=kappa, prob=0.2)
    W, alpha = generate_mixing_matrix(p)
    if alpha > 0.9:
        break

x_0 = np.random.rand(dim, n_agent)
print('alpha = ' + str(alpha))


eta_1 = 1 / p.L
eta_2 = 2 / (p.L + p.sigma)

n_inner_iters = int(m * 0.05)

n_mix = range(1, 20)


exps_dane = [
        NetworkDANE(p, n_iters=n_iters, n_mix=n, mu=mu, x_0=x_0, W=W)
        for n in n_mix 
        ]

exps_svrg = [
        NetworkSVRG(p, n_iters=n_iters, n_mix=n, n_inner_iters=n_inner_iters, eta=eta_1/10, x_0=x_0, W=W)
        for n in n_mix 
        ]

exps = exps_dane + exps_svrg

res = run_exp(exps, kappa=kappa, max_iter=n_iters, name='extra_comm_alpha_' + str(alpha), save=True)


plt.show()
