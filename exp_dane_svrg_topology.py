#!/usr/bin/env python
# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time

from problems import LinearRegression
from optimizers import *
from optimizers.utils import generate_mixing_matrix
from utils import multiprocess_run


n_agent = 20
m = 1000
dim = 40

kappa = 10000
kappa = 10

n_iters = 200

p = LinearRegression(n_agent, m, dim, noise_variance=1, kappa=kappa)
x_0 = np.random.rand(dim, n_agent)

eta = 2/(p.L + p.sigma)
mu = 1
n_inner_iters = int(m * 0.05)
n_svrg_iters = 20 * n_iters


def run(p, W, mu=0.01, dane_eta=1, svrg_eta=eta/40) :
    exps = [
        NetworkDANE(p, n_iters, eta=dane_eta, mu=mu, x_0=x_0, W=W, verbose=False),
        NetworkSVRG(p, n_svrg_iters, n_inner_iters, eta=svrg_eta, x_0=x_0, W=W, verbose=False)
        ]

    exps = multiprocess_run(2, exps)
    return exps


start = time.time()
alpha_list = []
results = []


# Erdos-Renyi random graph p=0.3
p.generate_erdos_renyi_graph(0.3)
W = generate_mixing_matrix(p.G)
alpha_list.append(
    np.linalg.norm(W - np.ones((n_agent, n_agent)) / n_agent, 2)
        )
results.append(run(p, W, mu=0.1))


# Ring topology
p.generate_ring_graph()
W = generate_mixing_matrix(p.G)
alpha_list.append(
    np.linalg.norm(W - np.ones((n_agent, n_agent)) / n_agent, 2)
        )
results.append(run(p, W, mu=3, dane_eta=1, svrg_eta=eta/600))


# Grid topology
p.generate_grid_graph(5, 4)
W = generate_mixing_matrix(p.G)
alpha_list.append(
    np.linalg.norm(W - np.ones((n_agent, n_agent)) / n_agent, 2)
        )
results.append(run(p, W, mu=1, dane_eta=0.8, svrg_eta=eta/400))


# Star topology
p.generate_star_graph()
W = generate_mixing_matrix(p.G)
alpha_list.append(
    np.linalg.norm(W - np.ones((n_agent, n_agent)) / n_agent, 2)
        )
results.append(run(p, W, mu=0.01, dane_eta=1))


end = time.time()
print('Total running time is {:.2f}s'.format(end - start))

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

plt.figure(0)
plt.figure(1)

for i in range(len(results)):
    res = results[i]
    res_dane = res[0].get_results()
    res_svrg = res[1].get_results()

    plt.figure(0)
    plt.semilogy(
            range(min(n_iters+1, len(res_dane['func_error']))),
                res_dane['func_error'][:n_iters+1], color=colors[i]
            )
    plt.semilogy(
            range(min(n_iters+1, len(res_svrg['func_error']))),
                res_svrg['func_error'][:n_iters+1], '--', color=colors[i]
            )

    plt.figure(1)
    plt.loglog( res_dane['n_grad'], res_dane['func_error'], color=colors[i] )
    plt.loglog( res_svrg['n_grad'], res_svrg['func_error'], '--', color=colors[i] )


legends = [alg + ', ' + topology for topology in ['ER(0.3)', 'Ring', 'Grid', 'Star'] for alg in ['Network-DANE', 'Network-SVRG']]

plt.figure(0)
plt.xlabel('#iters')
plt.ylabel(r"$\frac{f({\bar{\mathbf{x}}}^{(t)}) - f({\mathbf{x}}^\star)}{f({\mathbf{x}}^\star)}$")
plt.legend(legends)

plt.figure(1)
plt.xlabel('#grads')
plt.ylabel(r"$\frac{f({\bar{\mathbf{x}}}^{(t)}) - f({\mathbf{x}}^\star)}{f({\mathbf{x}}^\star)}$")
plt.legend(legends)
plt.show()
