#!/usr/bin/env python
# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

from nda import log
from nda.problems import LogisticRegression
from nda.optimizers import *
from nda.optimizers.utils import generate_mixing_matrix
from nda.experiment_utils import run_exp

import time

def plot_exp(exps, configs, filename, dim, n_agent, logx=False, logy=False):

    colors = ['k', 'r', 'g', 'b', 'c', 'm', 'y']
    line_styles = ['-', '--', ':']
    # log.info("Initial accuracy = " + str(p.accuracy(x_0)))
    results = [[exp.get_name()] + list(exp.get_metrics()) for exp in exps]

    with open(f"data/{filename}", 'wb') as f:
        pickle.dump(results, f)

    row_index = {'t': 1, 'comm_rounds': 0}
    column_index = {'f': 0}

    all_columns = {column for (_, columns, _) in results for column in columns}
    if 'grad_norm' in all_columns:
        column_index.update({'grad_norm': len(column_index)})
    if 'test_accuracy' in all_columns:
        column_index.update({'test_accuracy': len(column_index)})
    if 'var_error' in all_columns:
        column_index.update({'var_error': len(column_index)})
    if 'f_test' in all_columns:
        column_index.update({'f_test': len(column_index)})


    fig, axs = plt.subplots(2, len(column_index))


    legends = []
    for (name, columns, data), config, (line_style, color) in zip(results, configs, product(line_styles, colors)):

        tmp = get_bits_per_round_per_agent(config, dim) * n_agent
        n_skip = max(int(len(data) / 1000), 1)
        # log.info(name)
        # log.info('n_skip = %d', n_skip)
        # log.info('len = %d', len(data))

        def _plot_iter(y, logx=False, logy=False):
            iter_ax = axs[row_index['t'], column_index[y]]
            iter_ax.plot(
                data[:, columns.index('t')][::n_skip],
                data[:, columns.index(y)][::n_skip],
                line_style + color
            )
            iter_ax.set(xlabel='Iterations', ylabel=y)
            if logy:
                iter_ax.set_yscale('log')
            if logx:
                iter_ax.set_xscale('log')

        def _plot_comm(y, logx=False, logy=False):
            comm_ax = axs[row_index['comm_rounds'], column_index[y]]
            comm_ax.plot(
                data[:, columns.index('comm_rounds')][::n_skip] * tmp,
                data[:, columns.index(y)][::n_skip],
                line_style + color
            )
            comm_ax.set(xlabel='Bits transferred', ylabel=y)
            if logy:
                comm_ax.set_yscale('log')
            if logx:
                comm_ax.set_xscale('log')

        for column in column_index.keys():
            if column in columns:
                _plot_iter(column, logx=logx, logy=logy)
                if 'comm_rounds' in columns:
                    _plot_comm(column, logx=logx, logy=logy)

        legends.append(name + ','.join([k + '=' + str(v) for k, v in config.items() if k in ['gamma', 'compressor_param', 'compressor_type', 'eta', 'batch_size']]))

    plt.legend(legends)


def plot_gisette_exp(exps, topology, total_samples):
    # print("Initial accuracy = " + str(p.accuracy(x_0.mean(axis=1))))
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
    # plt.show()
    tikzplotlib.save("data/gisette-%s.tex" % topology, standalone=True, externalize_tables=True, override_externals=True)

if __name__ == '__main__':
    n_agent = 20

    # Experiment for Gisette classification
    p = LogisticRegression(n_agent, graph_type='er', graph_params=0.3, dataset='gisette', alpha=0.001)
    dim = p.dim

    os.system('mkdir data figs')
    if os.path.exists('data/gisette_initialization.npz'):
        x_0 = np.load('data/gisette_initialization.npz').get('x_0')
    else:
        x_0 = np.random.rand(dim, n_agent)
        np.savez('data/gisette_initialization.npz', x_0=x_0)
    x_0_mean = x_0.mean(axis=1)

    extra_metrics = ['grad_norm', 'test_accuracy']
    # Experiment 1: er topology
    W, alpha = generate_mixing_matrix(p)
    log.info('alpha = %.4f', alpha)

    exps = [
        DSGD(p, n_iters=20000, eta=10, x_0=x_0, W=W, diminishing_step_size=True, extra_metrics=extra_metrics),
        Destress(p, n_iters=40, batch_size=10, n_inner_iters=10, eta=1, K_in=2, K_out=2, x_0=x_0, W=W, extra_metrics=extra_metrics),
        GT_SARAH(p, n_iters=40, batch_size=10, n_inner_iters=10, eta=0.1, x_0=x_0, W=W, extra_metrics=extra_metrics),
    ]

    begin = time.time()
    exps = run_exp(exps, name='gisette-er', n_process=1, plot=False, save=True)
    end = time.time()
    log.info('Total %.2fs', end - begin)

    plot_gisette_exp(exps, 'er', p.m_total)

    # Experiment 2: grid topology
    p.generate_graph('grid', (4, 5))
    W, alpha = generate_mixing_matrix(p)
    log.info('alpha = %.4f', alpha)

    exps = [
        DSGD(p, n_iters=20000, eta=1, x_0=x_0, W=W, diminishing_step_size=True, extra_metrics=extra_metrics),
        Destress(p, n_iters=40, batch_size=10, n_inner_iters=10, eta=1, K_in=2, K_out=2, x_0=x_0, W=W, extra_metrics=extra_metrics),
        GT_SARAH(p, n_iters=40, batch_size=10, n_inner_iters=10, eta=0.01, x_0=x_0, W=W, extra_metrics=extra_metrics),
    ]

    begin = time.time()
    exps = run_exp(exps, name='gisette-grid', n_process=1, plot=False, save=True)
    end = time.time()
    log.info('Total %.2fs', end - begin)

    plot_gisette_exp(exps, 'grid', p.m_total)


    # Experiment 3: path topology
    p.generate_graph(graph_type='path')
    W, alpha = generate_mixing_matrix(p)
    log.info('alpha = %.4f', alpha)

    exps = [
        DSGD(p, n_iters=20000, eta=1, x_0=x_0, W=W, diminishing_step_size=True, extra_metrics=extra_metrics),
        Destress(p, n_iters=40, batch_size=10, n_inner_iters=10, eta=1, K_in=8, K_out=8, x_0=x_0, W=W, extra_metrics=extra_metrics),
        GT_SARAH(p, n_iters=40, batch_size=10, n_inner_iters=10, eta=0.01, x_0=x_0, W=W, extra_metrics=extra_metrics),
    ]


    begin = time.time()
    exps = run_exp(exps, name='gisette_path', n_process=1, plot=False, save=True)
    end = time.time()
    log.info('Total %.2fs', end - begin)

    plot_gisette_exp(exps, 'path', p.m_total)


    # Experiment for MNIST classification
    p = NN(n_agent, graph_type='er', graph_params=0.3)
    dim = p.dim

    if os.path.exists('data/mnist_initialization.npz'):
        x_0 = np.load('data/mnist_initialization.npz').get('x_0')
    else:
        x_0 = np.random.rand(dim, n_agent)
        np.savez('data/mnist_initialization.npz', x_0=x_0)
    x_0_mean = x_0.mean(axis=1)

    # Experiment 1: er topology
    W, alpha = generate_mixing_matrix(p)
    log.info('alpha = %.4f', alpha)

    exps = [
        DSGD(p, n_iters=20000, eta=10, x_0=x_0, W=W, diminishing_step_size=True, extra_metrics=extra_metrics),
        Destress(p, n_iters=40, batch_size=10, n_inner_iters=10, eta=1, K_in=2, K_out=2, x_0=x_0, W=W, extra_metrics=extra_metrics),
        GT_SARAH(p, n_iters=40, batch_size=10, n_inner_iters=10, eta=0.1, x_0=x_0, W=W, extra_metrics=extra_metrics),
    ]

    begin = time.time()
    exps = run_exp(exps, name='gisette-er', n_process=1, plot=False, save=True)
    end = time.time()
    log.info('Total %.2fs', end - begin)

    plot_gisette_exp(exps, 'er', p.m_total)

    # Experiment 2: grid topology
    p.generate_graph('grid', (4, 5))
    W, alpha = generate_mixing_matrix(p)
    log.info('alpha = %.4f', alpha)

    exps = [
        DSGD(p, n_iters=20000, eta=1, x_0=x_0, W=W, diminishing_step_size=True, extra_metrics=extra_metrics),
        Destress(p, n_iters=40, batch_size=10, n_inner_iters=10, eta=1, K_in=2, K_out=2, x_0=x_0, W=W, extra_metrics=extra_metrics),
        GT_SARAH(p, n_iters=40, batch_size=10, n_inner_iters=10, eta=0.01, x_0=x_0, W=W, extra_metrics=extra_metrics),
    ]

    begin = time.time()
    exps = run_exp(exps, name='gisette-grid', n_process=1, plot=False, save=True)
    end = time.time()
    log.info('Total %.2fs', end - begin)

    plot_gisette_exp(exps, 'grid', p.m_total)


    # Experiment 3: path topology
    p.generate_graph(graph_type='path')
    W, alpha = generate_mixing_matrix(p)
    log.info('alpha = %.4f', alpha)

    exps = [
        DSGD(p, n_iters=20000, eta=1, x_0=x_0, W=W, diminishing_step_size=True, extra_metrics=extra_metrics),
        Destress(p, n_iters=40, batch_size=10, n_inner_iters=10, eta=1, K_in=8, K_out=8, x_0=x_0, W=W, extra_metrics=extra_metrics),
        GT_SARAH(p, n_iters=40, batch_size=10, n_inner_iters=10, eta=0.01, x_0=x_0, W=W, extra_metrics=extra_metrics),
    ]


    begin = time.time()
    exps = run_exp(exps, name='gisette_path', n_process=1, plot=False, save=True)
    end = time.time()
    log.info('Total %.2fs', end - begin)

    plot_gisette_exp(exps, 'path', p.m_total)
