#!/usr/bin/env python
# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt

from problems import *
from optimizers import *
from utils import run_exp
from optimizers.utils import generate_mixing_matrix


n_agent = 20
m = 1000
dim = 40

kappa = 10
mu = 5e-8

# kappa = 10000
# mu = 5e-3


n_iters = 100

p = LogisticRegression(n_agent, m, dim, noise_ratio=0.05, kappa=kappa, prob=0.3)
print(p.n_edges)


x_0 = np.random.rand(dim, n_agent)
W = generate_mixing_matrix(p.G)


eta = 2/(p.L + p.sigma)
n_inner_iters = int(m * 0.05)
batch_size = int(m / 10)
n_dgd_iters = n_iters * 20
n_svrg_iters = n_iters * 20 



exps = [
    DGD_tracking(p, n_iters=n_dgd_iters, eta=eta/10, x_0=x_0, W=W),
    EXTRA(p, n_iters=n_dgd_iters, eta=eta/2, x_0=x_0, W=W),

    NetworkSVRG(p, n_iters=n_svrg_iters, n_inner_iters=n_inner_iters, eta=eta/20, mu=mu, x_0=x_0, W=W),
    NetworkSARAH(p, n_iters=n_svrg_iters, n_inner_iters=n_inner_iters, eta=eta/20, mu=mu, x_0=x_0, W=W),
    NetworkDANE(p, n_iters=n_iters, eta=1, mu=mu, x_0=x_0, W=W),
    ADMM(p, n_iters=n_iters, rho=1, x_0=x_0.mean(axis=1)),
    DANE(p, n_iters=n_iters, eta=1, mu=mu, x_0=x_0.mean(axis=1))
    ]



res = run_exp(exps, kappa=kappa, max_iter=n_iters, name='logistic_regression')

plt.show()
