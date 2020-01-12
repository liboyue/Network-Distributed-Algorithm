#!/usr/bin/env python
# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt

from problems import *
from optimizers import *
from optimizers.utils import generate_mixing_matrix


n = 20
m = 1000
dim = 40
n_iters = 200 # Run till converge

n_edges_list = [65, 130, 190]
n_edges_list = [20, 40, 50, 65, 130, 190]

p_1 = LinearRegression(n, m, dim, noise_variance=1, kappa=1.001)
x_1 = np.random.rand(dim, n)

p_10 = LinearRegression(n, m, dim, noise_variance=1, kappa=10)
x_10 = np.random.rand(dim, n)

p_100 = LinearRegression(n, m, dim, noise_variance=1, kappa=100)
x_100 = np.random.rand(dim, n)

p_1000 = LinearRegression(n, m, dim, noise_variance=1, kappa=1000)
x_1000 = np.random.rand(dim, n)

def multi_process_helper(n_edges):
    # multiprocessing library will make a copy of everything

    p_1.n_edges = n_edges
    p_1.generate_connected_graph() # n_edges may change due to the random graph model
    n_edges = p_1000.n_edges = p_100.n_edges = p_10.n_edges = p_1.n_edges
    p_1000.G = p_100.G = p_10.G = p_1.G

    W = generate_mixing_matrix(p_1.G)
    alpha = np.linalg.norm(W - np.ones((n, n)) / n, 2)
    res = [n_edges, alpha]

    print('n_edges = ' + str(p_1.n_edges) + ', alpha = ' + str(alpha))


    def _run_exp(optimizer):
        optimizer.optimize()
        res = optimizer.get_results()

        if len(res['var_error']) < n_iters and res['func_error'][-1] < 1e-4: # Converged
            return len(res['var_error'])
        else:
            return n_iters


    # NetworkDANE, kappa = 1
    mu = 0.5
    res.append(_run_exp(
        NetworkDANE(p_1, n_iters=n_iters, eta=1, mu=mu, x_0=x_1, W=W, verbose=False)
        ))


    # NetworkDANE, kappa = 10
    # mu = 0.1/10
    # res.append(_run_exp(
    #     NetworkDANE(p_10, n_iters, eta=1, mu=mu, x_0=x_10, W=W, verbose=False)
    #     ))

    # NetworkDANE, kappa = 100
    # mu = 0.1/100
    # res.append(_run_exp(
    #    NetworkDANE(p_100, n_iters, eta=1, mu=mu, x_0=x_100, W=W, verbose=False)
    #    ))

    # NetworkDANE, kappa = 1000
    mu = 0.5/1000
    res.append(_run_exp(
        NetworkDANE(p_1000, n_iters=n_iters, eta=1, mu=mu, x_0=x_1000, W=W, verbose=False)
        ))

    # NetworkSVRG
    n_gd_iters = n_iters
    n_inner_iters = int(m / 10)

    # NetworkSVRG, kappa = 1
    res.append(_run_exp(
            NetworkSVRG(p_1, n_iters=n_gd_iters, n_inner_iters=n_inner_iters, eta=0.07/(p_1.L + p_1.sigma), x_0=x_1, W=W, verbose=False)
            ))

    # NetworkSVRG, kappa = 10
    res.append(_run_exp(
            NetworkSVRG(p_10, n_iters=n_gd_iters, n_inner_iters=n_inner_iters, eta=0.07/(p_10.L + p_10.sigma), x_0=x_10, W=W, verbose=False)
            ))

    # NetworkSVRG, kappa = 100
    res.append(_run_exp(
           NetworkSVRG(p_100, n_iters=n_gd_iters, n_inner_iters=n_inner_iters, eta=0.07/(p_100.L + p_100.sigma), x_0 = x_100, W=W, verbose=False)
           ))

    print('n_edges = ' + str(p_1.n_edges) + ' done')

    return res


from multiprocessing import Pool
with Pool(6) as pool:
    res = pool.map(multi_process_helper, n_edges_list)


# Remove duplicated items
ind = [x[0] for x in res]
if len(ind) != len(set(ind)): # There exists duplicated items
    ind = {x:0 for x in set(ind)}
    for x in res:
        if ind[x[0]] == 0:
            ind[x[0]] += 1
        else:
            res.remove(x)

res = np.array(sorted(res, key=lambda x: x[0])) # Sort according to n_edges

fig, ax1 = plt.subplots()
alpha_color = 'tomato'

ax1.plot(res[:, 0], res[:, 1]*200, 'x--', color=alpha_color)
for i in range(2, res.shape[1]):
    ax1.plot(res[:, 0], res[:, i])


ax1.set_xlabel('#edges of the graph')
ax1.set_ylabel('#iterations till converge')
ax1.tick_params(axis='y')
ax1.legend([
    r'$\alpha$',
    r'Network-DANE, $\kappa=10^0$',
    # r'Network-DANE $\kappa=10^1$',
    # r'Network-DANE $\kappa=10^2$',
    r'Network-DANE, $\kappa=10^3$',
    r'Network-SVRG, $\kappa=10^0$',
    r'Network-SVRG, $\kappa=10^1$',
    r'Network-SVRG, $\kappa=10^2$',
    ])


ax2 = ax1.twinx()
ax2.set_ylabel(r'$\alpha$', color=alpha_color)
# ax2.plot(res[:, 0], res[:, 1], 'cx--')
# ax2.stackplot(res[:, 0], res[:, 1],  baseline='zero', alpha=0.4)
ax2.tick_params(axis='y', labelcolor=alpha_color)
# ax2.legend(r'$\alpha')
ax1.set_ylim([-10, 210])
ax2.set_ylim([-0.05,1.05])



fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.show()
