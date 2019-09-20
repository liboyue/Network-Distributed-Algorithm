#!/usr/bin/env python
# coding=utf-8
import numpy as np
import time
import matplotlib.pyplot as plt
from multiprocessing import Pool

from problems import NN
from optimizers import *
from optimizers.utils import generate_mixing_matrix
from utils import multi_process_helper


n_agent = 20
n_class = 10
img_dim = 785
n_hidden = 64
lr = 20

n_iters = 20

p = NN(n_agent, prob=0.3)
dim = (n_hidden+1) * (img_dim + n_class)
x_0 = np.random.randn(dim, n_agent) / 10
W = generate_mixing_matrix(p.G)


n_gd_iters = int(n_iters * 10)
n_inner_iters = int(p.m * 0.01)
batch_size = int(p.m / 100)
n_dsgd_iters = int(n_iters * p.m / batch_size)

centered_exps = [
    GD(p, n_gd_iters, eta=1, x_0=x_0.mean(axis=1), verbose=True),
    DANE(p, n_iters, eta=1, mu=0.01, x_0=x_0.mean(axis=1), local_n_iters=2, delta=1, verbose=True),
    ADMM(p, n_iters, rho=1, x_0=x_0.mean(axis=1), delta=1, local_optimizer='GD', local_n_iters=10, verbose=True)
    ]


distributd_exps = [
    DGD_tracking(p, n_gd_iters, eta=1, x_0=x_0, W=W, verbose=True),
    EXTRA(p, n_gd_iters, eta=1, x_0=x_0, W=W, verbose=True),
    NetworkSVRG(p, n_gd_iters, n_inner_iters, eta=0.1, x_0=x_0, W=W, opt=1, verbose=True),
    NetworkDANE(p, n_iters, eta=1, mu=0.001, x_0=x_0, W=W, local_n_iters=10, delta=0.1, local_optimizer='GD', verbose=True),
    ]

exps = centered_exps + distributd_exps


start = time.time()
with Pool(6) as pool:
     exps = pool.map(multi_process_helper, exps)
end = time.time()
print('Total ' + str(end-start) + 's')


max_iter = max(n_iters, n_gd_iters) + 1
table = np.zeros((max_iter, len(exps)*3))
for k in range(len(exps)):
    opt = exps[k]
    res = opt.get_results()
    for i in range(len(opt.history)):
        x = opt.history[i]['x']
        if len(x.shape) == 2:
            x = x.mean(axis=1)
        table[i, k*3] = res['n_grad'][i]
        table[i, k*3+1] = p.f(x)
        table[i, k*3+2] = p.accuracy(x)


# Plot
legends = [x.get_name() for x in exps]
x = np.array(range(n_iters+1)).reshape(-1, 1)

plt.figure()
for i in range(len(exps)):
    plt.plot(x, table[:, i*3+1])

plt.title('Loss')
plt.xlabel('#iters')
plt.legend(legends)
plt.savefig('figs/nn_iter_loss.eps', format='eps')

plt.figure()
for i in range(len(exps)):
    plt.plot(x, table[:, i*3+2])

plt.title('Accuracy')
plt.xlabel('#iters')
plt.legend(legends)
plt.savefig('figs/nn_iter_acc.eps', format='eps')

plt.figure()
for i in range(len(exps)):
    plt.plot(table[:, i*3], table[:, i*3+1])

plt.title('Loss')
plt.xlabel('#grad')
plt.legend(legends)
plt.savefig('figs/nn_grad_loss.eps', format='eps')

plt.figure()
for i in range(len(exps)):
    plt.plot(table[:, i*3], table[:, i*3+2])

plt.title('Accuracy')
plt.xlabel('#grad')
plt.legend(legends)
plt.savefig('figs/nn_grad_acc.eps', format='eps')

plt.show()
