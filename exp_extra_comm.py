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
mu = 5e-2

n_iters = 100

p = LinearRegression(n_agent, m, dim, noise_variance=1, kappa=kappa, prob=0.2)

print(p.n_edges)


x_0 = np.random.rand(dim, n_agent)
W = generate_mixing_matrix(p.G)
W_s = (W + np.eye(n_agent)) / 2
alpha = np.linalg.norm(W_s - np.ones((n_agent, n_agent))/n_agent, 2)
print('alpha = ' + str(alpha))

K = int( - np.log(kappa) / np.log(alpha))


eta_2 = 2 / (p.L + p.sigma)
eta_1 = 1 / p.L

n_inner_iters = int(m * 0.05)
n_svrg_iters = n_iters * 20

n_mix = range(1, 10)
print([alpha** n for n in n_mix])


exps_dane = [
        NetworkDANE(p, n_iters=n_iters, n_mix=n, mu=mu, x_0=x_0, W=W) for n in range(1, 20) 
        ]

exps_svrg = [
        NetworkSVRG(p, n_iters=n_svrg_iters, n_inner_iters=n_inner_iters, eta=eta_2/20, mu=mu, x_0=x_0, n_mix=n, W=W)
        for n in range(1, 6) 
        ]

exps = exps_dane + exps_svrg

legends = [
        exp.name
        + ', n_mix = ' + str(exp.n_mix if hasattr(exp, 'n_mix') else '1')
        for exp in exps]

res = run_exp(exps, kappa=kappa, max_iter=n_iters, name='extra_comm_alpha_' + str(alpha), legend=legends)

plt.show()
