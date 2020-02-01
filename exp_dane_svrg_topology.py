#!/usr/bin/env python
# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time

from problems import LinearRegression
from optimizers import *
from optimizers.utils import generate_mixing_matrix
from utils import run_exp


n_agent = 20
m = 1000
dim = 40

kappa = 10

n_iters = 200

p = LinearRegression(n_agent, m, dim, noise_variance=1, kappa=kappa)

x_0 = np.random.rand(dim, n_agent)


eta_1 = 1 / p.L
n_inner_iters = int(m * 0.05)
n_svrg_iters = 20 * n_iters


def save(res, alpha, topology):
    res_dane, res_svrg = res[0].get_results(), res[1].get_results()
    max_iters = max(
            len(res_dane['func_error']),
            len(res_svrg['func_error'])
            )

    table = np.zeros((max_iters, 5))
    table[:, 0] = np.arange(max_iters)
    table[:len(res_dane['n_grad']), 1] = res_dane['n_grad']
    table[:len(res_dane['func_error']), 2] = res_dane['func_error']
    table[:len(res_svrg['n_grad']), 3] = res_svrg['n_grad']
    table[:len(res_svrg['func_error']), 4] = res_svrg['func_error']

    np.savetxt('data/topology_' + topology + '_alpha_{:.2f}.txt'.format(alpha), table)
    with open('data/topology_' + topology + '_alpha_{:.2f}.txt'.format(alpha)) as f:
        data = f.read()
    with open('data/topology_' + topology + '_alpha_{:.2f}.txt'.format(alpha), 'w') as f:
        f.write('iter    dane_n_grad    dane_error    svrg_n_grad   svrg_error\n' + data)


start = time.time()

alpha_list = [0]
exps = [
    DANE(p, n_iters=n_iters, mu=5e-10, x_0=x_0.mean(axis=1)),
    SVRG(p, n_iters=n_svrg_iters, n_inner_iters=int(n_inner_iters*n_agent), eta=eta_1/10, x_0=x_0.mean(axis=1)),
    ]


# Erdos-Renyi random graph p=0.3
p.generate_erdos_renyi_graph(0.3)
W, alpha = generate_mixing_matrix(p)
alpha_list.append(alpha)

exps += [
    NetworkDANE(p, n_iters=n_iters, mu=5e-10, x_0=x_0, W=W),
    NetworkSVRG(p, n_iters=n_svrg_iters, n_inner_iters=n_inner_iters, eta=eta_1/10, x_0=x_0, W=W)
    ]


# Ring topology
p.generate_ring_graph()
W, alpha = generate_mixing_matrix(p)
alpha_list.append(alpha)

exps += [
    NetworkDANE(p, n_iters=n_iters, mu=50, x_0=x_0, W=W),
    NetworkSVRG(p, n_iters=n_svrg_iters, n_inner_iters=n_inner_iters, eta=eta_1/200, x_0=x_0, W=W),
    ]


# Grid topology
p.generate_grid_graph(5, 4)
W, alpha = generate_mixing_matrix(p)
alpha_list.append(alpha)

exps += [
    NetworkDANE(p, n_iters=n_iters, mu=10, x_0=x_0, W=W),
    NetworkSVRG(p, n_iters=n_svrg_iters, n_inner_iters=n_inner_iters, eta=eta_1/100, x_0=x_0, W=W),
    ]


# Star topology
p.generate_star_graph()
W, alpha = generate_mixing_matrix(p)
alpha_list.append(alpha)

exps += [
    NetworkDANE(p, n_iters=n_iters, mu=1, x_0=x_0, W=W),
    NetworkSVRG(p, n_iters=n_svrg_iters, n_inner_iters=n_inner_iters, eta=eta_1/40, x_0=x_0, W=W),
    ]


res = run_exp(exps, n_process=1, save=False, plot=False)

save(res[0:2], 0, 'centered')
save(res[2:4], alpha_list[1], 'er')
save(res[4:6], alpha_list[2], 'ring')
save(res[6:8], alpha_list[3], 'grid')
save(res[8:10], alpha_list[4], 'star')


end = time.time()
print('Total running time is {:.2f}s'.format(end - start))

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

plt.figure(0)
plt.figure(1)


for i in range(2, len(res), 2):
    res_dane = res[i].get_results()
    res_svrg = res[i+1].get_results()
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
