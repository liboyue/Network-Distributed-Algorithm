#!/usr/bin/env python
# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import os

from nda import log
from nda.problems import LogisticRegression
from nda.optimizers import *
from nda.optimizers.utils import generate_mixing_matrix
from nda.experiment_utils import run_exp


if __name__ == '__main__':

    n_agent = 20
    n_iters = 20

    p = LogisticRegression(n_agent, graph_type='er', graph_params=0.3, dataset='gisette', alpha=0.001)

    x_0 = np.random.rand(p.dim, n_agent)
    x_0_mean = x_0.mean(axis=1)

    batch_size = int(np.sqrt(p.m))
    n_inner_iters = 10

    W, alpha = generate_mixing_matrix(p)
    log.info('alpha = %.4f', alpha)

    exps = [
        GD(p, n_iters=n_iters, eta=100, x_0=x_0, W=W),
    ]

    exps = run_exp(exps, max_iter=n_iters, name='gisette', n_process=2, plot=True, save=False)
    plt.show()
