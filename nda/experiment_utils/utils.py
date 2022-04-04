#!/usr/bin/env python
# coding=utf-8

import matplotlib.pyplot as plt
import multiprocessing as mp
import itertools, os, random, time
import numpy as np
import pandas as pd

from nda import log


def LINE_STYLES():
    return itertools.cycle([
        line + color for line in ['-', '--', ':'] for color in ['k', 'r', 'b', 'c', 'y', 'g']
    ])


def multi_process_helper(device_id, task_id, opt, res_queue):
    start = time.time()
    log.info(f'{opt.get_name()} started')
    log.debug(f'task {task_id} started on device {device_id}')
    np.random.seed(task_id)
    random.seed(task_id)

    try:
        import cupy as cp
        cp.cuda.Device(device=device_id).use()
        cp.random.seed(task_id)
        opt.cuda()
        opt.optimize()
        columns, metrics = opt.get_metrics()
        name = opt.get_name()
    except ModuleNotFoundError:
        opt.optimize()
        columns, metrics = opt.get_metrics()
        name = opt.get_name()

    end = time.time()
    log.info('%s done, total %.2fs', name, end - start)
    log.debug(f'task {task_id} on device {device_id} exited')

    res_queue.put([task_id, name, pd.DataFrame(metrics, columns=columns)])


def run_exp(exps, kappa=None, max_iter=None, name=None, save=False, plot=True, n_cpu_processes=None, n_gpus=None, processes_per_gpu=1, verbose=False):

    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    use_gpu = n_gpus is not None
    if use_gpu:
        pool = {n: [] for n in range(n_gpus)}
    else:
        pool = {n: [] for n in range(n_cpu_processes)}

    _exps = list(enumerate(exps))
    q = mp.Queue(len(_exps))
    res = []

    def _pop_queue():
        while q.empty() is False:
            res.append(q.get())
            # print(f'{res[-1][0]} stopped')

    def _remove_dead_process():
        for device_id in pool.keys():
            pool[device_id] = [process for process in pool[device_id] if process.is_alive()]

    while len(_exps) > 0:

        _pop_queue()
        _remove_dead_process()

        availabel_device_id = -1
        min_processes = processes_per_gpu if use_gpu else 1

        for device_id, processes in pool.items():
            n_processes = len(processes)
            if n_processes < min_processes:
                availabel_device_id = device_id
                min_processes = n_processes

        if availabel_device_id > -1:
            task_id, exp = _exps.pop(0)
            # print(f'{task_id} launched')
            pp = mp.Process(target=multi_process_helper, args=(availabel_device_id, task_id, exp, q,))
            pp.start()
            pool[availabel_device_id].append(pp)
            pp, task_id, exp = None, None, None
        else:
            time.sleep(0.1)

    while len(pool) > 0:
        _remove_dead_process()
        pool = {k: v for k, v in pool.items() if len(v) > 0}
        _pop_queue()
        time.sleep(0.1)

    res = [_[1:] for _ in sorted(res, key=lambda x: x[0])]

    if save is True:
        os.system('mkdir -p data figs')

    if plot is True:
        plot_results(
            res,
            exps[0].p.m_total,
            kappa=kappa,
            max_iter=max_iter,
            name=name,
            save=save,
        )

    if save is True:  # Save data files too

        # Save to txt file
        for (name, data), exp in zip(res, exps):

            if kappa is not None:
                fname = r'data/' + str(name) + '_kappa_' + str(int(kappa))
            else:
                fname = r'data/' + str(name)

            if hasattr(exp, 'n_mix'):
                fname += '_mix_' + str(exp.n_mix) + '_' + name + '.txt'
            else:
                fname += '_mix_1_' + name + '.txt'

            data.to_csv(fname, index=False)

    return res


def plot_results(results, m_total, kappa=None, max_iter=None, name=None, save=False):

    if kappa is not None:
        fig_path = r'figs/' + str(name) + '_kappa_' + str(int(kappa))
    else:
        fig_path = r'figs/' + str(name)

    plot_iters(results, fig_path, kappa=kappa, max_iter=max_iter, save=save)
    plot_grads(results, fig_path, m_total, kappa=kappa, max_iter=max_iter, save=save)

    if any(['comm_rounds' in res[1].columns for res in results]):
        plot_comms(results, fig_path, kappa=kappa, max_iter=max_iter, save=save)


def plot_iters(results, path=None, kappa=None, max_iter=None, save=False):

    # iters vs. var_error
    legends = []

    if any(['var_error' in result.columns for name, result in results]):
        plt.figure()
        for (name, result), style in zip(results, LINE_STYLES()):
            if 'var_error' in result.columns:
                legends.append(name)
                plt.loglog(result.t, result.var_error, style)
        plt.ylabel(r"$\frac{f({\bar{\mathbf{x}}}^{(t)}) - f({\mathbf{x}}^\star)}{f({\mathbf{x}}^\star)}$")
        plt.xlabel('#outer iterations')

        if kappa is not None:
            plt.title(r"$\kappa$ = " + str(int(kappa)))
        plt.legend(legends)
        if save is True:
            plt.savefig(path + '_var_iter.eps', format='eps')

    # iters vs. f
    legends = []
    if max_iter is None:
        max_iter = min([res[1].t.iloc[-1] for res in results])

    plt.figure()
    for (name, result), style in zip(results, LINE_STYLES()):
        legends.append(name)
        mask = result.t <= max_iter
        result = result.loc[mask]
        plt.loglog(result.t, result.f, style)
    # plt.title('Function value error vs. #outer iterations')
    plt.ylabel(r"Loss")
    plt.xlabel('#outer iterations')
    if kappa is not None:
        plt.title(r"$\kappa$ = " + str(int(kappa)))
    plt.legend(legends)
    if save is True:
        plt.savefig(path + '_fval_iter.eps', format='eps')


def plot_comms(results, path, kappa=None, max_iter=None, save=False):

    if any(['var_error' in result.columns for name, result in results]):
        legends = []
        plt.figure()
        for (name, data), style in zip(results, LINE_STYLES()):
            if 'var_error' in data.columns and 'comm_rounds' in data.columns:
                legends.append(name)
                plt.loglog(data.comm_rounds, data.var_error, style)

        plt.ylabel(r"$\frac{\Vert {\bar{\mathbf{x}}}^{(t)} - {\mathbf{x}}^\star \Vert}{\Vert {\mathbf{x}}^\star \Vert}$")
        plt.xlabel('#communication rounds')
        if kappa is not None:
            plt.title(r"$\kappa$ = " + str(int(kappa)))
        plt.legend(legends)
        if save is True:
            plt.savefig(path + '_var_comm.eps', format='eps')

    legends = []
    plt.figure()
    for (name, data), style in zip(results, LINE_STYLES()):
        if 'comm_rounds' in data.columns:
            legends.append(name)
            plt.semilogy(data.comm_rounds, data.f, style)
    # plt.title('Function value error vs. #communications')
    plt.ylabel(r"Loss")
    plt.xlabel('#communication rounds')
    if kappa is not None:
        plt.title(r"$\kappa$ = " + str(int(kappa)))
    plt.legend(legends)
    if save is True:
        plt.savefig(path + '_fval_comm.eps', format='eps')


def plot_grads(results, path, m, kappa=None, max_iter=None, save=False):

    # n_grads vs. var_error
    if any(['var_error' in result.columns for name, result in results]):
        legends = []
        plt.figure()
        for (name, data), style in zip(results, LINE_STYLES()):
            if 'var_error' in data.columns:
                plt.loglog(data.n_grads / m, data.var_error, style)
                legends.append(name)
        # plt.title('Variable error vs. #gradient evaluations')
        plt.ylabel(r"$\frac{\Vert {\bar{\mathbf{x}}}^{(t)} - {\mathbf{x}}^\star \Vert}{\Vert {\mathbf{x}}^\star \Vert}$")
        plt.xlabel('#gradient evaluations / #total samples')
        if kappa is not None:
            plt.title(r"$\kappa$ = " + str(int(kappa)))
        plt.legend(legends)
        if save is True:
            plt.savefig(path + '_var_grads.eps', format='eps')

    # n_grads vs. f
    legends = []
    plt.figure()
    for (name, data), style in zip(results, LINE_STYLES()):
        plt.loglog(data.n_grads / m, data.f, style)
        legends.append(name)
    # plt.title('Function value error vs. #gradient evaluations')
    plt.ylabel(r"Loss")
    plt.xlabel('#gradient evaluations / #total samples')
    if kappa is not None:
        plt.title(r"$\kappa$ = " + str(int(kappa)))
    plt.legend(legends)
    if save is True:
        plt.savefig(path + '_fval_grads.eps', format='eps')
