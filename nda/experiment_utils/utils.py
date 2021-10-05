#!/usr/bin/env python
# coding=utf-8

import matplotlib.pyplot as plt
from multiprocessing import Pool
import itertools
import numpy as np
import time
import os

from nda import log


def LINE_STYLES():
    return itertools.cycle([
        line + color for line in ['-', '--', ':'] for color in ['k', 'r', 'b', 'c', 'y', 'g']
    ])


def multi_process_helper(opt):
    start = time.time()
    log.info('%s started', opt.get_name())
    opt.optimize()
    end = time.time()
    log.info('%s done, total %.2fs', opt.get_name(), end - start)
    return opt


def run_exp(exps, kappa=None, max_iter=None, name=None, save=False, plot=True, n_process=1, verbose=False):

    if n_process > 1:
        with Pool(n_process) as pool:
            exps = pool.map(multi_process_helper, exps)
    else:
        exps = [multi_process_helper(exp) for exp in exps]

    if save is True:
        os.system('mkdir -p data figs')

    if plot is True:
        plot_results(
            exps,
            kappa=kappa,
            max_iter=max_iter,
            name=name,
            save=save,
        )

    if save is True:  # Save data files too

        def _down_sample(table):
            if len(table[0]) > max_iter * 2:
                skip = int(len(table[:, 0]) / max_iter / 2)
                index = list(range((table[:, 0])))
                index = index[:max_iter + 1] + index[max_iter::skip]
                return table[index]
            return table

        # Save to txt file
        for exp in exps:

            if kappa is not None:
                fname = r'data/' + str(name) + '_kappa_' + str(int(kappa))
            else:
                fname = r'data/' + str(name)

            if hasattr(exp, 'n_mix'):
                fname += '_mix_' + str(exp.n_mix) + '_' + exp.get_name() + '.txt'
            else:
                fname += '_mix_1_' + exp.get_name() + '.txt'

            y = exp.get_metrics()
            np.savetxt(fname, y[1], delimiter='    ', header=','.join(y[0]))

    return exps


def plot_results(exps, kappa=None, max_iter=None, name=None, save=False):

    if kappa is not None:
        fig_path = r'figs/' + str(name) + '_kappa_' + str(int(kappa))
    else:
        fig_path = r'figs/' + str(name)

    results = [[exp.get_name()] + list(exp.get_metrics()) for exp in exps]

    plot_iters(results, fig_path, kappa=kappa, max_iter=max_iter, save=save)
    plot_grads(results, fig_path, exps[0].p.m_total, kappa=kappa, max_iter=max_iter, save=save)

    if any(['comm_rounds' in res[1] for res in results]):
        plot_comms(results, fig_path, kappa=kappa, max_iter=max_iter, save=save)


def plot_iters(results, path, kappa=None, max_iter=None, save=False):

    # iters vs. var_error
    legends = []

    plt.figure()
    for (name, columns, data), style in zip(results, LINE_STYLES()):
        if 'var_error' in columns:
            legends.append(name)
            plt.loglog(data[:, 0], data[:, columns.index('var_error')], style)
    # plt.title('Variable error vs. #outer iterations')
    plt.ylabel(r"$\frac{\Vert {\bar{\mathbf{x}}}^{(t)} - {\mathbf{x}}^\star \Vert}{\Vert {\mathbf{x}}^\star \Vert}$")
    plt.xlabel('#outer iterations')
    if kappa is not None:
        plt.title(r"$\kappa$ = " + str(int(kappa)))
    plt.legend(legends)
    if save is True:
        plt.savefig(path + '_var_iter.eps', format='eps')

    # iters vs. f
    legends = []
    if max_iter is None:
        max_iter = min([x[3][-1][-1] for res in results])

    plt.figure()
    for (name, columns, data), style in zip(results, LINE_STYLES()):
        legends.append(name)
        mask = data[:, 0] <= max_iter
        plt.loglog(data[:, 0][mask], data[:, 2][mask], style)
    # plt.title('Function value error vs. #outer iterations')
    plt.ylabel(r"$\frac{f({\bar{\mathbf{x}}}^{(t)}) - f({\mathbf{x}}^\star)}{f({\mathbf{x}}^\star)}$")
    plt.xlabel('#outer iterations')
    if kappa is not None:
        plt.title(r"$\kappa$ = " + str(int(kappa)))
    plt.legend(legends)
    if save is True:
        plt.savefig(path + '_fval_iter.eps', format='eps')


def plot_comms(results, path, kappa=None, max_iter=None, save=False):

    legends = []
    plt.figure()
    for (name, columns, data), style in zip(results, LINE_STYLES()):
        if 'var_error' in columns and 'comm_rounds' in columns:
            legends.append(name)
            plt.loglog(data[:, columns.index('comm_rounds')], data[:, columns.index('var_error')], style)

    # plt.title('Variable error vs. #communications')
    plt.ylabel(r"$\frac{\Vert {\bar{\mathbf{x}}}^{(t)} - {\mathbf{x}}^\star \Vert}{\Vert {\mathbf{x}}^\star \Vert}$")
    plt.xlabel('#communication rounds')
    if kappa is not None:
        plt.title(r"$\kappa$ = " + str(int(kappa)))
    plt.legend(legends)
    if save is True:
        plt.savefig(path + '_var_comm.eps', format='eps')

    legends = []
    plt.figure()
    for (name, columns, data), style in zip(results, LINE_STYLES()):
        if 'comm_rounds' in columns:
            legends.append(name)
            plt.semilogy(data[:, columns.index('comm_rounds')], data[:, 2], style)
    # plt.title('Function value error vs. #communications')
    plt.ylabel(r"$\frac{f({\bar{\mathbf{x}}}^{(t)}) - f({\mathbf{x}}^\star)}{f({\mathbf{x}}^\star)}$")
    plt.xlabel('#communication rounds')
    if kappa is not None:
        plt.title(r"$\kappa$ = " + str(int(kappa)))
    plt.legend(legends)
    if save is True:
        plt.savefig(path + '_fval_comm.eps', format='eps')


def plot_grads(results, path, m, kappa=None, max_iter=None, save=False):

    # n_grads vs. var_error
    legends = []
    plt.figure()
    for (name, columns, data), style in zip(results, LINE_STYLES()):
        if 'var_error' in columns:
            plt.loglog(data[:, 1] / m, data[:, columns.index('var_error')], style)
            legends.append(name)
    # plt.title('Variable error vs. #gradient evaluations')
    plt.ylabel(r"$\frac{\Vert {\bar{\mathbf{x}}}^{(t)} - {\mathbf{x}}^\star \Vert}{\Vert {\mathbf{x}}^\star \Vert}$")
    plt.xlabel('#gradient evaluations / #total samples')
    if kappa is not None:
        plt.title(r"$\kappa$ = " + str(int(kappa)))
    plt.legend(legends)
    if save is True:
        plt.savefig(path + '_var_grads.eps', format='eps')

    # n_grads vs. var_error
    legends = []
    plt.figure()
    for (name, columns, data), style in zip(results, LINE_STYLES()):
        plt.loglog(data[:, 1] / m, data[:, 2], style)
        legends.append(name)
    # plt.title('Function value error vs. #gradient evaluations')
    plt.ylabel(r"$\frac{f({\bar{\mathbf{x}}}^{(t)}) - f({\mathbf{x}}^\star)}{f({\mathbf{x}}^\star)}$")
    plt.xlabel('#gradient evaluations / #total samples')
    if kappa is not None:
        plt.title(r"$\kappa$ = " + str(int(kappa)))
    plt.legend(legends)
    if save is True:
        plt.savefig(path + '_fval_grads.eps', format='eps')
