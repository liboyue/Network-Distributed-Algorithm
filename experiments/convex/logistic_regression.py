#!/usr/bin/env python
# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt

from nda import log
from nda.problems import LogisticRegression
from nda.optimizers import *
from nda.optimizers.utils import generate_mixing_matrix
from nda.experiment_utils import run_exp


if __name__ == '__main__':

    n_agent = 20
    m = 1000
    dim = 40

    kappa = 10
    mu = 5e-8
    n_iters = 30

    p = LogisticRegression(n_agent, m, dim, noise_ratio=0.05, kappa=kappa, graph_type='er', graph_params=0.3)
    W, alpha = generate_mixing_matrix(p)
    log.info('m = %d, n = %d, alpha = %.4f' % (m, n_agent, alpha))

    x_0 = np.random.rand(dim, n_agent)
    x_0_mean = x_0.mean(axis=1)

    eta = 2 / (p.L + p.sigma)
    n_inner_iters = int(m * 0.05)
    batch_size = int(m / 10)
    n_dgd_iters = n_iters * 20
    n_sarah_iters = n_iters * 20 
    n_dsgd_iters = int(n_iters * m / batch_size)

    centralized = [
        GD(p, n_iters=n_iters, eta=eta, x_0=x_0_mean),
        SGD(p, n_iters=n_dsgd_iters, eta=eta*3, batch_size=batch_size, x_0=x_0_mean, diminishing_step_size=True),
        NAG(p, n_iters=n_iters, x_0=x_0_mean),
        SARAH(p, n_iters=n_sarah_iters, n_inner_iters=n_inner_iters, eta=eta / 20, x_0=x_0_mean)
        ]

    distributed = [
        DGD_tracking(p, n_iters=n_dgd_iters, eta=eta / 10, x_0=x_0, W=W),
        DANE(p, n_iters=n_iters, mu=mu, x_0=x_0_mean),
        NetworkDANE(p, n_iters=n_iters, mu=mu, x_0=x_0, W=W),
        ]

    exps = centralized + distributed

    res = run_exp(exps, kappa=kappa, max_iter=n_iters, name='logistic_regression', n_process=1, save=True)

    plt.show()
