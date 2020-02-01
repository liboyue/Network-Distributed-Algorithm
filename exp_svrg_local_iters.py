#!/usr/bin/env python
# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt

from problems import *
from optimizers import *
from optimizers.utils import generate_mixing_matrix
from utils import run_exp

n_agent = 20
m = 1000
dim = 40

kappa = 10
mu = 5e-10


n_iters = 1000

p = LinearRegression(n_agent, m, dim, noise_variance=1, kappa=kappa, prob=0.3)
W, alpha = generate_mixing_matrix(p)
x_0 = np.random.rand(dim, n_agent)
eta = 2/(p.L + p.sigma)


inner_iters = [1, 2, 5, 10, 50, 100]
batch_size = [1]
params = [(k, 1, 0.05) for k in inner_iters]


exps = [NetworkDANE(p, n_iters=n_iters, mu=mu, x_0=x_0, W=W)] \
    + [NetworkSVRG(p, n_iters=n_iters, n_inner_iters=x[0], batch_size=x[1], eta=eta*x[2], x_0=x_0, W=W) for x in params]


res = run_exp(exps, save=False, plot=False)


table = np.zeros((len(inner_iters), len(batch_size)*2+1))
table[:, 0] = inner_iters
table[:, 0] /= m

inner_iters_dict = {inner_iters[i]: i for i in range(len(inner_iters))}
batch_size_dict = {batch_size[i]: i for i in range(len(batch_size))}

for x in res[1:]:
    y = x.get_results()
    if len(y['func_error']) < n_iters and y['func_error'][-1] < 1: # Converged
        table[inner_iters_dict[x.n_inner_iters], batch_size_dict[x.batch_size]*2+1] = len(y['func_error'])-1
        table[inner_iters_dict[x.n_inner_iters], batch_size_dict[x.batch_size]*2+2] = y['n_grad'][-1]
    else: # Didn't converge
        table[inner_iters_dict[x.n_inner_iters], batch_size_dict[x.batch_size]*2+1] = None
        table[inner_iters_dict[x.n_inner_iters], batch_size_dict[x.batch_size]*2+2] = None

plt.figure()
for i in range(len(batch_size)):
    plt.semilogy([x/m for x in inner_iters], table[:, i*2+1])


plt.xlabel('K/m')
plt.ylabel('#iters till converge')
plt.legend(['b={:d}'.format(x) for x in batch_size])


plt.figure()
for i in range(len(batch_size)):
    plt.semilogy([x/m for x in inner_iters], table[:, i*2+2])


plt.xlabel('K/m')
plt.ylabel('#grads/#samples till converge')
plt.legend(['b={:d}'.format(x) for x in batch_size])



fname = 'data/svrg_iter_to_converge_ratio.txt'
np.savetxt(fname, table)

with open(fname) as f:
    data = f.read()

with open(fname, 'w') as f:
    f.write( 'ratio    ' \
            + '    '.join([x + str(y) for y in batch_size for x in ['n_iter_batch_size_', 'n_grad_batch_size_'] ]) \
            + '\n' + data)


plt.show()
