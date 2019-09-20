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
kappa = 10
n_iters = 100

# Generate problem, initial value and mixing matrix
p = LinearRegression(n_agent, m, dim, noise_variance=1, kappa=kappa, prob=0.3)
x_0 = np.random.rand(dim, n_agent)
W = generate_mixing_matrix(p.G)

print("alpha = " + str(np.linalg.norm(W - np.ones((n_agent, n_agent))/n_agent, 2) ))

eta_1 = 2 / (p.L + p.sigma)
eta_2 = 1 / p.L

mu = 0.5 / kappa # In practice, Network DANE allows almost arbitrary choices of \mu

n_inner_iters = int(m * 0.05)
n_svrg_iters = n_iters * 20
n_dgd_iters = n_iters * 20
batch_size = int(m / 10)
n_dsgd_iters = int(n_iters * m / batch_size)

exps = [
    DGD_tracking(p, n_dgd_iters, eta=eta_1/20, x_0=x_0, W=W, verbose=False),
    EXTRA(p, n_dgd_iters, eta=eta_1/4, x_0=x_0, W=W, verbose=False),

    NetworkSVRG(p, n_svrg_iters, n_inner_iters, eta=eta_1/40, x_0=x_0, W=W, opt=1, batch_size=1, verbose=False),
    NetworkDANE(p, n_iters, eta=1, mu=mu, x_0=x_0, W=W, verbose=False),
    # NetworkDANE_quadratic(p, n_iters, eta=1, mu=mu, x_0=x_0, W=W), # Behavior is identical to Network-DANE

    ADMM(p, n_iters, rho=1, x_0=x_0.mean(axis=1), verbose=False),
    DANE(p, n_iters, eta=1, mu=mu, x_0=x_0.mean(axis=1), verbose=False)
    ]


p.plot_graph()

res = run_exp(exps, kappa=kappa, max_iter=n_iters, name='linear_regression', save=False)


plt.show()
