#!/usr/bin/env python
# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

from problems import NN
from optimizers import *
from optimizers.utils import generate_mixing_matrix
from utils import run_exp
import time


if __name__ == '__main__': # Don't know why but this line is necessary

    n_agent = 20
    n_class = 10
    img_dim = 785
    n_hidden = 64
    dim = (n_hidden+1) * (img_dim + n_class)
    lr = 20
    mu = 0.001

    n_iters = 20

    p = NN(n_agent, prob=0.3)
    x_0 = np.random.randn(dim, n_agent) / 10
    W, alpha = generate_mixing_matrix(p)

    n_gd_iters = int(n_iters * 10)
    n_inner_iters = int(p.m_mean * 0.01)
    batch_size = int(p.m_mean / 100)
    n_dsgd_iters = int(n_iters * 20)


    distributed = [
        DSGD(p, n_iters=n_dsgd_iters, eta=1, batch_size=batch_size*3, x_0=x_0, W=W, diminish=False, verbose=True),

        ADMM(p, n_iters=n_iters*10, rho=1, x_0=x_0.mean(axis=1), delta=1, local_optimizer='GD', local_n_iters=10, verbose=True)
        ]

    network = [
        NetworkSVRG(p, n_iters=n_gd_iters, n_inner_iters=n_inner_iters, eta=0.1, x_0=x_0, W=W, verbose=True),
        NetworkSARAH(p, n_iters=n_gd_iters, n_inner_iters=n_inner_iters, eta=0.1, x_0=x_0, W=W, verbose=True),
        ]

    exps = distributed + network
    begin = time.time()
    res_list = run_exp(exps, max_iter=n_iters, name='nn', n_process=1, plot=False, save=True)
    end = time.time()
    print('Total {:.2f}s'.format(end-begin))


    print("Initial accuracy = " + str(p.accuracy(x_0.mean(axis=1))))

    max_iter = max(n_iters, n_gd_iters, n_dsgd_iters) + 1
    table = np.zeros((max_iter, len(exps)*3))

    for k in range(len(res_list)):
        res = res_list[k].get_results()
        for i in range(len(res['history'])):
            x = res['history'][i]['x']
            if len(x.shape) == 2:
                x = x.mean(axis=1)
            table[i, k*3] = res['n_grad'][i]
            table[i, k*3+2], table[i, k*3+1] = p.accuracy(x) # acc, loss


    # Plot
    legends = [x.get_name() for x in exps]
    x = np.array(range(n_iters+1)).reshape(-1, 1)

    plt.figure()
    for i in range(len(exps)):
        idx = table[:, i*3+1] != 0
        plt.plot(x, table[:, i*3+1][idx][:len(x)])

    plt.ylabel('Loss')
    plt.xlabel('#iters')
    plt.legend(legends)
    # plt.savefig('figs/nn_iter_loss.eps', format='eps')

    plt.figure()
    for i in range(len(exps)):
        idx = table[:, i*3+2] != 0
        plt.plot(x, table[:, i*3+2][idx][:len(x)])

    plt.ylabel('Accuracy')
    plt.xlabel('#iters')
    plt.legend(legends)
    # plt.savefig('figs/nn_iter_acc.eps', format='eps')

    plt.figure()
    for i in range(len(exps)):
        idx = table[:, i*3] != 0
        plt.plot(table[:, i*3][idx], table[:, i*3+1][idx])

    plt.ylabel('Loss')
    plt.xlabel('#grad')
    plt.legend(legends)
    # plt.savefig('figs/nn_grad_loss.eps', format='eps')

    plt.figure()
    for i in range(len(exps)):
        idx = table[:, i*3] != 0
        plt.plot(table[:, i*3][idx], table[:, i*3+2][idx])

    plt.ylabel('Accuracy')
    plt.xlabel('#grad')
    plt.legend(legends)
    # plt.savefig('figs/nn_grad_acc.eps', format='eps')
    plt.show()
    quit()


    # Save data
    x = np.array(range(max_iter)).reshape(-1, 1)
    np.savetxt('data/nn.txt', np.append(x, table, axis=1))
    with open('data/nn.txt', 'r') as f:
         data = f.read()

    with open('data/nn.txt', 'w') as f:
        n = [x.replace(' ', '') for x in legends]
        h = [x + y for x in n for y in ['_n_grad', '_loss', '_acc']]
        f.write('iter    ' + '    '.join(h) + '\n' + data)

    plt.show()
