#!/usr/bin/env python
# coding=utf-8
import matplotlib.pyplot as plt
from multiprocessing import Pool
import itertools
import seaborn as sns
import numpy as np
import time

def LINE_STYLES():
    return itertools.cycle([
        line+color for line in ['-', '--', ':'] for color in ['k', 'r', 'b', 'c', 'y', 'g']
        ])

def multi_process_helper(opt):
    start = time.time()
    print(opt.get_name() + ' started')
    _ = opt.optimize()
    end = time.time()
    print(opt.get_name() + ' done, total {:.2f}s'.format(end-start))
    return opt


def multiprocess_run(n, exps):
    with Pool(n) as pool:
        exps = pool.map(multi_process_helper, exps)
    return exps


def run_exp_centralized(exps, kappa=None, max_iter=None, name=None):

    with Pool(len(exps)) as pool:
        res = pool.map(multi_process_helper, exps)

    plot_results_centralized(
            [x.get_results() for x in res],
            [x.get_name() for x in res],
            kappa=kappa,
            max_iter=max_iter,
            name=name
            )

    return res


def run_exp(exps, kappa=None, max_iter=None, name=None, plot=True, legend=None, parallel=True):

    if parallel:
        with Pool(6) as pool:
            res = pool.map(multi_process_helper, exps)
    else:
        for exp in exps:
            exp = multi_process_helper(exp)
        res = exps

    for x in res:
        if x.verbose == True:
            x.plot_history()

    if plot == True:
        if legend is None:
            legend = [x.get_name() for x in res]
        plot_results(
                [x.get_results() for x in res],
                legend,
                kappa=kappa,
                max_iter=max_iter,
                name=name
                )

    return res


def plot_results(results, legend, kappa=None, max_iter=None, name=None):

    if max_iter == None:
        max_iter = max([len(res['var_error']) for res in results]) 

    plot_iters(results, legend, kappa=kappa, max_iter=max_iter, name=name)
    plot_comms(results, legend, kappa=kappa, max_iter=max_iter, name=name)
    plot_grads(results, legend, kappa=kappa, max_iter=max_iter, name=name)


def plot_results_centralized(results, legend, kappa=None, max_iter=None, name=None):
    plot_iters(results, legend, kappa=kappa, max_iter=max_iter, name=name)
    plot_grads(results, legend, kappa=kappa, max_iter=max_iter, name=name)


def plot_iters(results, legend, kappa=None, max_iter=None, name=None):
    plt.figure()

    for res, style in zip(results, LINE_STYLES()):
        plt.semilogy(range(1, len(res['var_error'][:max_iter])+1), res['var_error'][:max_iter], style)
    # plt.title('Variable error vs. #outer iterations')
    plt.ylabel(r"$\frac{\Vert {\bar{\mathbf{x}}}^{(t)} - {\mathbf{x}}^\star \Vert}{\Vert {\mathbf{x}}^\star \Vert}$")
    plt.xlabel('#outer iterations')
    if kappa is not None:
        plt.title(r"$\kappa$ = " + str(int(kappa)))
    plt.legend(legend)

    plt.figure()
    for res, style in zip(results, LINE_STYLES()):
        plt.semilogy(range(1, len(res['func_error'][:max_iter])+1), res['func_error'][:max_iter], style)
    # plt.title('Function value error vs. #outer iterations')
    plt.ylabel(r"$\frac{f({\bar{\mathbf{x}}}^{(t)}) - f({\mathbf{x}}^\star)}{f({\mathbf{x}}^\star)}$")
    plt.xlabel('#outer iterations')
    if kappa is not None:
        plt.title(r"$\kappa$ = " + str(int(kappa)))
    plt.legend(legend)


def plot_comms(results, legend, kappa=None, max_iter=None, name=None):
    plt.figure()
    for res, style in zip(results, LINE_STYLES()):
        plt.semilogy(res['n_comm'][:max_iter], res['var_error'][:max_iter], style)
    # plt.title('Variable error vs. #communications')
    plt.ylabel(r"$\frac{\Vert {\bar{\mathbf{x}}}^{(t)} - {\mathbf{x}}^\star \Vert}{\Vert {\mathbf{x}}^\star \Vert}$")
    plt.xlabel('#communications')
    if kappa is not None:
        plt.title(r"$\kappa$ = " + str(int(kappa)))
    plt.legend(legend)


    plt.figure()

    for res, style in zip(results, LINE_STYLES()):
        plt.semilogy(res['n_comm'][:max_iter], res['func_error'][:max_iter], style)
    # plt.title('Function value error vs. #communications')
    plt.ylabel(r"$\frac{f({\bar{\mathbf{x}}}^{(t)}) - f({\mathbf{x}}^\star)}{f({\mathbf{x}}^\star)}$")
    plt.xlabel('#communications')
    if kappa is not None:
        plt.title(r"$\kappa$ = " + str(int(kappa)))
    plt.legend(legend)


def plot_grads(results, legend, kappa=None, max_iter=None, name=None):
    plt.figure()

    for res, style in zip(results, LINE_STYLES()):
        plt.loglog(res['n_grad'][1:], res['var_error'][1:], style)
    # plt.title('Variable error vs. #gradient evaluations')
    plt.ylabel(r"$\frac{\Vert {\bar{\mathbf{x}}}^{(t)} - {\mathbf{x}}^\star \Vert}{\Vert {\mathbf{x}}^\star \Vert}$")
    plt.xlabel('#gradient evaluations / #total samples')
    if kappa is not None:
        plt.title(r"$\kappa$ = " + str(int(kappa)))
    plt.legend(legend)


    plt.figure()

    for res, style in zip(results, LINE_STYLES()):
        plt.loglog(res['n_grad'][1:], res['func_error'][1:], style)
    # plt.title('Function value error vs. #gradient evaluations')
    plt.ylabel(r"$\frac{f({\bar{\mathbf{x}}}^{(t)}) - f({\mathbf{x}}^\star)}{f({\mathbf{x}}^\star)}$")
    plt.xlabel('#gradient evaluations')
    if kappa is not None:
        plt.title(r"$\kappa$ = " + str(int(kappa)))
    plt.legend(legend)
