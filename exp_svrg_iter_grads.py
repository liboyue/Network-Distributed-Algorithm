#!/usr/bin/env python
# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time

from problems import *
from optimizers import *
from optimizers.utils import generate_mixing_matrix

from utils import multiprocess_run, run_exp

# NetworkSVRG = NetworkSARAH

n_agent = 20
m = 1000
dim = 40

kappa = 10000
kappa = 10

n_iters = 1000000 # Run till converge

p = LinearRegression(n_agent, m, dim, noise_variance=1, kappa=kappa, prob=0.3)

W = generate_mixing_matrix(p.G)

x_0 = np.random.rand(dim, n_agent)

eta = 2/(p.L + p.sigma)

params = [
        # n_inner_iters, eta
        (1, 0.15),
        (2, 0.20),
        (5, 0.30),
        (10, 0.15),
        (50, 0.05),
        (100, 0.05),
        (300, 0.02),
        (500, 0.01),
        (700, 0.01),
        (900, 0.01),
        ]

inner_iters = [x[0] for x in params]

exps = [NetworkSVRG(p, n_iters, n_inner_iters=x[0], eta=eta*x[1], x_0=x_0, W=W) for x in reversed(params)]

res = run_exp(exps, kappa=kappa, max_iter=n_iters, name='linear_regression')

table = np.zeros(len(inner_iters))
inner_iters_dict = {inner_iters[i]: i for i in range(len(inner_iters))}

for x in res:
    y = x.get_results()
    if len(y['func_error']) < n_iters and y['func_error'][-1] < 1: # Converged
        table[inner_iters_dict[x.n_inner_iters]] = len(y['func_error'])-1
    else: # Didn't converge
        table[inner_iters_dict[x.n_inner_iters]] = None

plt.figure()
plt.semilogy([x/m for x in inner_iters], table)
plt.xlabel('K/m')
plt.ylabel('#iters till converge')

plt.show()
