#!/usr/bin/env python
# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import time

from nda import log
from nda.problems import LogisticRegression
from nda.optimizers import *
from nda.optimizers.utils import generate_mixing_matrix
from nda.experiment_utils import run_exp


def plot_gisette_exp(exps, topology, total_samples):
    results = [[exp.get_name()] + list(exp.get_metrics()) for exp in exps]
    with open('data/gisette_%s_res.data' % topology, 'wb') as f:
        pickle.dump(results, f)
    max_comm = min([results[i][2][-1, results[i][1].index('comm_rounds')] for i in range(len(results)) if 'comm_rounds' in results[i][1]])
    min_comm = 0
    min_grad = p.m_total / 2
    max_grad = min([results[i][2][-1, results[i][1].index('n_grads')] for i in range(len(results))])
    fig, axs = plt.subplots(1, 4)
    for (name, columns, data) in results:
        comm_idx = columns.index('comm_rounds')
        grad_idx = columns.index('n_grads')
        acc_idx = columns.index('test_accuracy')
        loss_idx = columns.index('f')
        if len(data) > 1000:
            skip = max(int(len(data) / 1000), 1)
            data = data[::skip]
        comm_mask = (data[:, comm_idx] <= max_comm) & (data[:, comm_idx] > min_comm)
        grad_mask = (data[:, grad_idx] <= max_grad) & (data[:, grad_idx] > min_grad)
        axs[0].loglog(data[:, comm_idx][comm_mask], data[:, loss_idx][comm_mask])
        axs[0].set(xlabel='\#communication rounds', ylabel='Loss')
        axs[1].semilogx(data[:, comm_idx][comm_mask], data[:, acc_idx][comm_mask])
        axs[1].set(xlabel='\#communication rounds', ylabel='Testing accuracy')
        axs[2].loglog(data[:, grad_idx][grad_mask] / total_samples, data[:, loss_idx][grad_mask])
        axs[2].set(xlabel='\#grads/\#samples', ylabel='Loss')
        axs[3].semilogx(data[:, grad_idx][grad_mask] / total_samples, data[:, acc_idx][grad_mask])
        axs[3].set(xlabel='\#grads/\#samples', ylabel='Testing accuracy')
    axs[3].legend([result[0].replace('_', '-') for result in results])
    plt.show()


if __name__ == '__main__':
    n_agent = 20

    # Experiment 1: Gisette classification
    p = LogisticRegression(n_agent, graph_type='er', graph_params=0.3, dataset='gisette', alpha=0.001)
    dim = p.dim

    os.system('mkdir data figs')
    if os.path.exists('data/gisette_initialization.npz'):
        x_0 = np.load('data/gisette_initialization.npz').get('x_0')
    else:
        x_0 = np.random.rand(dim, n_agent)
        np.savez('data/gisette_initialization.npz', x_0=x_0)
    x_0_mean = x_0.mean(axis=1)

    # Experiment 1.1: er topology
    W, alpha = generate_mixing_matrix(p)
    log.info('alpha = %.4f', alpha)

    exps = [
        DSGD(p, n_iters=20000, eta=1, x_0=x_0, W=W, diminishing_step_size=True),
        DESTRESS(p, n_iters=300, n_inner_iters=10, eta=1, K_in=2, K_out=2, batch_size=10, x_0=x_0, W=W),
        GT_SARAH(p, n_iters=300, n_inner_iters=10, batch_size=10, eta=0.1, x_0=x_0, W=W),
    ]

    begin = time.time()
    exps = run_exp(exps, name='gisette-er', n_process=1, plot=False, save=True)
    end = time.time()
    log.info('Total %.2fs', end - begin)

    plot_gisette_exp(exps, 'er', p.m_total)

    # Experiment 1.2: grid topology
    p.generate_graph('grid', (4, 5))
    W, alpha = generate_mixing_matrix(p)
    log.info('alpha = %.4f', alpha)

    exps = [
        DSGD(p, n_iters=20000, eta=1, x_0=x_0, W=W, diminishing_step_size=True),
        DESTRESS(p, n_iters=300, n_inner_iters=10, eta=1, K_in=2, K_out=2, batch_size=10, x_0=x_0, W=W),
        GT_SARAH(p, n_iters=300, n_inner_iters=10, batch_size=10, eta=0.01, x_0=x_0, W=W),
    ]

    begin = time.time()
    exps = run_exp(exps, name='gisette_grid', n_process=1, plot=False, save=True)
    end = time.time()
    log.info('Total %.2fs', end - begin)

    plot_gisette_exp(exps, 'grid', p.m_total)


    # Experiment 1.3: path topology
    p.generate_graph(graph_type='path')
    W, alpha = generate_mixing_matrix(p)
    log.info('alpha = %.4f', alpha)

    exps = [
        DSGD(p, n_iters=20000, eta=1, x_0=x_0, W=W, diminishing_step_size=True),
        DESTRESS(p, n_iters=300, n_inner_iters=10, eta=1, K_in=8, K_out=8, batch_size=10, x_0=x_0, W=W),
        GT_SARAH(p, n_iters=300, n_inner_iters=10, eta=0.001, batch_size=10, x_0=x_0, W=W),
    ]


    begin = time.time()
    exps = run_exp(exps, name='gisette_path', n_process=1, plot=False, save=True)
    end = time.time()
    log.info('Total %.2fs', end - begin)

    plot_gisette_exp(exps, 'path', p.m_total)


    # Experiment 2: MNIST classification
    p = NN(n_agent, graph_type='er', graph_params=0.3)
    dim = p.dim

    if os.path.exists('data/mnist_initialization.npz'):
        x_0 = np.load('data/mnist_initialization.npz').get('x_0')
    else:
        x_0 = np.random.rand(dim, n_agent)
        np.savez('data/mnist_initialization.npz', x_0=x_0)
    x_0_mean = x_0.mean(axis=1)

    # Experiment 2.1: er topology
    W, alpha = generate_mixing_matrix(p)
    log.info('alpha = %.4f', alpha)

    exps = [
        DSGD(p, n_iters=100000, eta=1, x_0=x_0, W=W, diminishing_step_size=True),
        DESTRESS(p, n_iters=30, n_inner_iters=10, eta=0.1, K_in=2, K_out=2, batch_size=100, x_0=x_0, W=W),
        GT_SARAH(p, n_iters=30, n_inner_iters=10, batch_size=100, eta=0.01, x_0=x_0, W=W),
    ]

    begin = time.time()
    exps = run_exp(exps, name='mnist_er', n_process=1, plot=False, save=True)
    end = time.time()
    log.info('Total %.2fs', end - begin)

    plot_gisette_exp(exps, 'er', p.m_total)

    # Experiment 2.2: grid topology
    p.generate_graph('grid', (4, 5))
    W, alpha = generate_mixing_matrix(p)
    log.info('alpha = %.4f', alpha)

    exps = [
        DSGD(p, n_iters=100000, eta=1, x_0=x_0, W=W, diminishing_step_size=True),
        DESTRESS(p, n_iters=30, n_inner_iters=10, eta=0.1, K_in=2, K_out=2, batch_size=100, x_0=x_0, W=W),
        GT_SARAH(p, n_iters=30, n_inner_iters=10, batch_size=100, eta=0.001, x_0=x_0, W=W),
    ]

    begin = time.time()
    exps = run_exp(exps, name='mnist_grid', n_process=1, plot=False, save=True)
    end = time.time()
    log.info('Total %.2fs', end - begin)

    plot_gisette_exp(exps, 'grid', p.m_total)


    # Experiment 2.3: path topology
    p.generate_graph(graph_type='path')
    W, alpha = generate_mixing_matrix(p)
    log.info('alpha = %.4f', alpha)

    exps = [
        DSGD(p, n_iters=100000, eta=1, x_0=x_0, W=W, diminishing_step_size=True),
        DESTRESS(p, n_iters=30, n_inner_iters=10, eta=0.1, K_in=8, K_out=8, batch_size=100, x_0=x_0, W=W),
        GT_SARAH(p, n_iters=30, n_inner_iters=10, batch_size=100, eta=0.001, x_0=x_0, W=W),
    ]


    begin = time.time()
    exps = run_exp(exps, name='mnist_path', n_process=1, plot=False, save=True)
    end = time.time()
    log.info('Total %.2fs', end - begin)

    plot_gisette_exp(exps, 'path', p.m_total)
