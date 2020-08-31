#!/usr/bin/env python
# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt

from problems import *
from optimizers import *
from utils import run_exp
from optimizers.utils import generate_mixing_matrix

n_agent = 20

kappa = 10
mu = 5e-8

kappa = 100
mu = 1e-1

kappa = 2
mu = 5e-9



# kappa = 1
# mu = 5e-8


n_iters = 300

p = GisetteClassification(n_agent, kappa=kappa, prob=0.3)
dim = p.dim



print(p.n_edges)


x_0 = np.random.rand(dim, n_agent)
x_0_mean = x_0.mean(axis=1)
W, alpha = generate_mixing_matrix(p)
print('alpha = ' + str(alpha))



eta = 2/(p.L + p.sigma)
n_inner_iters = int(p.m_mean * 0.05)
# n_inner_iters_2 = int(m / 10)
batch_size = int(p.m_mean / 10)
batch_size = 10
n_dgd_iters = n_iters * 20
n_svrg_iters = n_iters * 20 
n_dsgd_iters = int(n_iters * p.m_mean / batch_size)


single_machine = [
    # GD(p, n_iters=n_iters, eta=eta, x_0=x_0_mean, verbose=True),
    # SGD(p, n_iters=n_dsgd_iters, eta=eta*3, batch_size=batch_size, x_0=x_0_mean, diminish=True),
    # NAG(p, n_iters=n_iters, x_0=x_0_mean),
    # SVRG(p, n_iters=n_svrg_iters, n_inner_iters=n_inner_iters, eta=eta/20, x_0=x_0_mean),
    # SARAH(p, n_iters=n_svrg_iters, n_inner_iters=n_inner_iters, eta=eta/20, x_0=x_0_mean),
    ]


distributed = [
    DGD_tracking(p, n_iters=n_dgd_iters, eta=eta/10, x_0=x_0, W=W, verbose=True),
    # DSGD(p, n_iters=n_dsgd_iters, eta=eta*2, batch_size=batch_size, x_0=x_0, diminish=True),
    EXTRA(p, n_iters=n_dgd_iters, eta=eta/10, x_0=x_0, W=W, verbose=True),
    # NIDS(p, n_iters=n_dgd_iters, eta=eta, x_0=x_0, W=W),
    ADMM(p, n_iters=n_iters, rho=1, x_0=x_0_mean, verbose=True),
    DANE(p, n_iters=n_iters, mu=mu, x_0=x_0_mean, verbose=True)
    ]

network = [
    # NetworkGD(p, n_iters=n_dgd_iters, eta=eta, x_0=x_0, W=W),
    NetworkSVRG(p, n_iters=n_svrg_iters, n_inner_iters=n_inner_iters, eta=eta/20, mu=0, x_0=x_0, W=W, batch_size=batch_size, verbose=True),
    NetworkSARAH(p, n_iters=n_svrg_iters, n_inner_iters=n_inner_iters, eta=eta/20, mu=0, x_0=x_0, W=W, batch_size=batch_size, verbose=True),
    NetworkDANE(p, n_iters=n_iters, mu=mu, x_0=x_0, W=W, verbose=True),
    ]

# exps = [
    # NetworkGD(p, n_iters=n_dgd_iters, eta=eta, x_0=x_0, W=W),
    # NetworkSVRG(p, n_iters=n_svrg_iters, n_inner_iters=n_inner_iters, eta=eta/20, mu=0, x_0=x_0, W=W, batch_size=batch_size, verbose=True),
    # NetworkSARAH(p, n_iters=n_svrg_iters, n_inner_iters=n_inner_iters, eta=eta/20, mu=0, x_0=x_0, W=W, batch_size=batch_size, verbose=True),
    # NetworkDANE(p, n_iters=n_iters, mu=1e-2, x_0=x_0, W=W, verbose=True),
    # ]

exps = single_machine + distributed + network

res = run_exp(exps, kappa=kappa, max_iter=n_iters, name='gisette', n_process=1, save=True)

tmp = [ x.get_results() for x in res if x.get_name() == 'NetworkDANE'][0]
k = np.exp(np.log( tmp['func_error'][-1] / tmp['func_error'][0] ) / len(tmp['func_error']))
print('NetworkDANE\'s convergence rate is: ' + str(k))
print('1 - 1/ (2 kappa) = ' + str(1 - 1 / (2 * kappa)))


def accuracy(w):
    if len(w.shape) > 1:
        w = w.mean(axis=1)
    Y_hat = p.X_val.dot(w)
    Y_hat[Y_hat > 0] = 1
    Y_hat[Y_hat < 0] = 0
    return np.mean(Y_hat == p.Y_val)

b = [(x, x+1) for x in range(10)]
acc = []
for r in res:
    r = r.get_results()
    acc.append(r['n_grad'].tolist())
    acc.append([accuracy(x['x']) for x in r['history']])

max_len = 0
for data in acc:
    max_len = max(max_len, len(data))

for i in range(len(acc)):
    if len(acc[i]) < max_len:
        acc[i] += [np.nan] * (max_len - len(acc[i]))

acc = [np.arange(max_len)] + acc

acc = np.array(acc).T
fname = 'data/gisette_acc_kappa_%d.txt' % kappa
np.savetxt(fname, acc, delimiter='    ')
header = 'iter    ' + '    '.join(['    '.join([_.get_name() + '_n_grad', _.get_name() + '_acc']) for _ in res])

with open(fname, 'r') as f:
    content = f.read()

with open(fname, 'w') as f:
    f.write(header + "\n" + content)


plt.show()
