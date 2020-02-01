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
    opt.optimize()
    end = time.time()
    print(opt.get_name() + ' done, total {:.2f}s'.format(end-start))
    return opt


def run_exp(exps, kappa=None, max_iter=None, name=None, save=False, plot=True, n_process=1, verbose=False):

    if n_process > 1:
        with Pool(n_process) as pool:
            res = pool.map(multi_process_helper, exps)
    else:
        res = [multi_process_helper(exp) for exp in exps]

    if verbose == True:
        for x in res:
            x.plot_history()

    if plot == True:
        plot_results(
                res,
                kappa=kappa,
                max_iter=max_iter,
                name=name,
                save=save,
                )

    if save == True: # Save data files too
        # Save to txt file
        for x in res:

            if kappa is not None:
                fname = r'data/' + str(name) + '_kappa_' + str(int(kappa))
            else:
                fname = r'data/' + str(name)

            if hasattr(x, 'n_mix'):
                fname += '_mix_' + str(x.n_mix) + '_' + x.get_name() + '.txt'
            else:
                fname += '_mix_1_' + x.get_name() + '.txt'

            y = x.get_results()
            if 'n_comm' in y:
                tmp = [
                        list(range(1, len(y['var_error'])+1)),
                        y['var_error'],
                        y['func_error'],
                        y['n_comm'],
                        y['n_grad']
                        ]
            else:
                tmp = [
                        list(range(1, len(y['var_error'])+1)),
                        y['var_error'],
                        y['func_error'],
                        np.zeros(len(y['var_error'])),
                        y['n_grad']
                        ]

            tmp = np.array(tmp).T
            np.savetxt(fname, tmp, delimiter='    ')

            with open(fname, 'r') as f:
                content = f.read()

            with open(fname, 'w') as f:
                f.write("iter    var_error    func_error    n_comm    n_grad\n" + content)

    return res


def plot_results(res, kappa=None, max_iter=None, name=None, save=False):


    if kappa is not None:
        fig_path = r'figs/' + str(name) + '_kappa_' + str(int(kappa))
    else:
        fig_path = r'figs/' + str(name)

    full_legend = [x.get_name() for x in res]
    full_res = [x.get_results() for x in res]

    if max_iter == None:
        max_iter = max([len(x['var_error']) for x in full_res]) 

    plot_iters(full_res, full_legend, fig_path, kappa=kappa, max_iter=max_iter, save=save)
    plot_grads(full_res, full_legend, fig_path, kappa=kappa, max_iter=max_iter, save=save)


    partial_legend = [x.get_name() for x in res if 'n_comm' in x.get_results()]
    partial_res = [x.get_results() for x in res if 'n_comm' in x.get_results()]

    plot_comms(partial_res, partial_legend, fig_path, kappa=kappa, max_iter=max_iter, save=save)


def plot_iters(results, legend, path, kappa=None, max_iter=None, save=False):
    plt.figure()

    for res, style in zip(results, LINE_STYLES()):
        plt.semilogy(range(1, len(res['var_error'][:max_iter])+1), res['var_error'][:max_iter], style)
    # plt.title('Variable error vs. #outer iterations')
    plt.ylabel(r"$\frac{\Vert {\bar{\mathbf{x}}}^{(t)} - {\mathbf{x}}^\star \Vert}{\Vert {\mathbf{x}}^\star \Vert}$")
    plt.xlabel('#outer iterations')
    if kappa is not None:
        plt.title(r"$\kappa$ = " + str(int(kappa)))
    plt.legend(legend)
    if save == True:
        plt.savefig(path + '_var_iter.eps', format='eps')
    

    plt.figure()
    for res, style in zip(results, LINE_STYLES()):
        plt.semilogy(range(1, len(res['func_error'][:max_iter])+1), res['func_error'][:max_iter], style)
    # plt.title('Function value error vs. #outer iterations')
    plt.ylabel(r"$\frac{f({\bar{\mathbf{x}}}^{(t)}) - f({\mathbf{x}}^\star)}{f({\mathbf{x}}^\star)}$")
    plt.xlabel('#outer iterations')
    if kappa is not None:
        plt.title(r"$\kappa$ = " + str(int(kappa)))
    plt.legend(legend)
    if save == True:
        plt.savefig(path + '_fval_iter.eps', format='eps')


def plot_comms(results, legend, path, kappa=None, max_iter=None, save=False):
    plt.figure()
    for res, style in zip(results, LINE_STYLES()):
        plt.semilogy(res['n_comm'][:max_iter], res['var_error'][:max_iter], style)
    # plt.title('Variable error vs. #communications')
    plt.ylabel(r"$\frac{\Vert {\bar{\mathbf{x}}}^{(t)} - {\mathbf{x}}^\star \Vert}{\Vert {\mathbf{x}}^\star \Vert}$")
    plt.xlabel('#communications')
    if kappa is not None:
        plt.title(r"$\kappa$ = " + str(int(kappa)))
    plt.legend(legend)
    if save == True:
        plt.savefig(path + '_var_comm.eps', format='eps')


    plt.figure()

    for res, style in zip(results, LINE_STYLES()):
        plt.semilogy(res['n_comm'][:max_iter], res['func_error'][:max_iter], style)
    # plt.title('Function value error vs. #communications')
    plt.ylabel(r"$\frac{f({\bar{\mathbf{x}}}^{(t)}) - f({\mathbf{x}}^\star)}{f({\mathbf{x}}^\star)}$")
    plt.xlabel('#communications')
    if kappa is not None:
        plt.title(r"$\kappa$ = " + str(int(kappa)))
    plt.legend(legend)
    if save == True:
        plt.savefig(path + '_fval_comm.eps', format='eps')


def plot_grads(results, legend, path, kappa=None, max_iter=None, save=False):
    plt.figure()

    for res, style in zip(results, LINE_STYLES()):
        plt.loglog(res['n_grad'][1:], res['var_error'][1:], style)
    # plt.title('Variable error vs. #gradient evaluations')
    plt.ylabel(r"$\frac{\Vert {\bar{\mathbf{x}}}^{(t)} - {\mathbf{x}}^\star \Vert}{\Vert {\mathbf{x}}^\star \Vert}$")
    plt.xlabel('#gradient evaluations / #total samples')
    if kappa is not None:
        plt.title(r"$\kappa$ = " + str(int(kappa)))
    plt.legend(legend)
    if save == True:
        plt.savefig(path + '_var_grads.eps', format='eps')


    plt.figure()

    for res, style in zip(results, LINE_STYLES()):
        plt.loglog(res['n_grad'][1:], res['func_error'][1:], style)
    # plt.title('Function value error vs. #gradient evaluations')
    plt.ylabel(r"$\frac{f({\bar{\mathbf{x}}}^{(t)}) - f({\mathbf{x}}^\star)}{f({\mathbf{x}}^\star)}$")
    plt.xlabel('#gradient evaluations')
    if kappa is not None:
        plt.title(r"$\kappa$ = " + str(int(kappa)))
    plt.legend(legend)
    if save == True:
        plt.savefig(path + '_fval_grads.eps', format='eps')
