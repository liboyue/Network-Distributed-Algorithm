#!/usr/bin/env python
# coding=utf-8
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import time

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


def run_exp(exps, kappa=None, max_iter=None, name=None, save=False, plot=True, legend=None):

    with Pool(6) as pool:
        res = pool.map(multi_process_helper, exps)

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
                name=name,
                save=save,
                )

    if save == True: # Save data files too
        # Save to txt file
        for x in res:

            y = x.get_results()
            tmp = [
                    list(range(1, len(y['var_error'])+1)),
                    y['var_error'],
                    y['func_error'],
                    y['n_comm'],
                    y['n_grad']
                    ]
            tmp = np.array(tmp).T

            fname = r'data/' + str(name) + '_kappa_' + str(int(kappa)) + '_' + x.get_name() + '.txt'
            np.savetxt(fname, tmp, delimiter='    ')
            with open(fname, 'r') as f:
                content = f.read()
            with open(fname, 'w') as f:
                f.write("iter    var_error    func_error    n_comm    n_grad\n" + content)
    return res


def plot_results(results, legend, kappa=None, max_iter=None, name=None, save=False):

    if max_iter == None:
        max_iter = max([len(res['var_error']) for res in results]) 

    plot_iters(results, legend, kappa=kappa, max_iter=max_iter, name=name, save=save)
    plot_grads(results, legend, kappa=kappa, max_iter=max_iter, name=name, save=save)


def plot_iters(results, legend, kappa=None, max_iter=None, name=None, save=False):

    plt.figure()
    for res in results:
        plt.semilogy(range(1, len(res['var_error'][:max_iter])+1), res['var_error'][:max_iter])
    plt.title('Variable error vs. #outer iterations')
    plt.ylabel(r"$\frac{\Vert {\bar{\mathbf{x}}}^{(t)} - {\mathbf{x}}^\star \Vert}{\Vert {\mathbf{x}}^\star \Vert}$")
    plt.xlabel('#outer iterations')
    if kappa is not None:
        plt.title(r"$\kappa$ = " + str(int(kappa)))
    plt.legend(legend)
    if save == True:
        plt.savefig('figs/' + str(name) + '_kappa_' + str(int(kappa)) + '_var_iter.eps', format='eps')
    

    plt.figure()
    for res in results:
        plt.semilogy(range(1, len(res['func_error'][:max_iter])+1), res['func_error'][:max_iter])
    plt.title('Function value error vs. #outer iterations')
    plt.ylabel(r"$\frac{f({\bar{\mathbf{x}}}^{(t)}) - f({\mathbf{x}}^\star)}{f({\mathbf{x}}^\star)}$")
    plt.xlabel('#outer iterations')
    if kappa is not None:
        plt.title(r"$\kappa$ = " + str(int(kappa)))
    plt.legend(legend)
    if save == True:
        plt.savefig('figs/' + str(name) + '_kappa_' + str(int(kappa)) + '_fval_iter.eps', format='eps')


def plot_grads(results, legend, kappa=None, max_iter=None, name=None, save=False):
    plt.figure()
    for res in results:
        plt.loglog(res['n_grad'][1:], res['var_error'][1:])
    plt.title('Variable error vs. #gradient evaluations')
    plt.ylabel(r"$\frac{\Vert {\bar{\mathbf{x}}}^{(t)} - {\mathbf{x}}^\star \Vert}{\Vert {\mathbf{x}}^\star \Vert}$")
    plt.xlabel('#gradient evaluations / #total samples')
    if kappa is not None:
        plt.title(r"$\kappa$ = " + str(int(kappa)))
    plt.legend(legend)
    if save == True:
        plt.savefig('figs/' + str(name) + '_kappa_' + str(int(kappa)) + '_var_grads.eps', format='eps')


    plt.figure()
    for res in results:
        plt.loglog(res['n_grad'][1:], res['func_error'][1:])
    plt.title('Function value error vs. #gradient evaluations')
    plt.ylabel(r"$\frac{f({\bar{\mathbf{x}}}^{(t)}) - f({\mathbf{x}}^\star)}{f({\mathbf{x}}^\star)}$")
    plt.xlabel('#gradient evaluations')
    if kappa is not None:
        plt.title(r"$\kappa$ = " + str(int(kappa)))
    plt.legend(legend)
    if save == True:
        plt.savefig('figs/' + str(name) + '_kappa_' + str(int(kappa)) + '_fval_grads.eps', format='eps')
