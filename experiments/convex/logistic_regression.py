#!/usr/bin/env python
# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt

from nda.problems import LogisticRegression
from nda.optimizers import *
from nda.optimizers.utils import generate_mixing_matrix

from nda.experiment_utils import run_exp

if __name__ == '__main__':
    n_agent = 20
    m = 1000
    dim = 40


    kappa = 10000
    mu = 5e-3

    kappa = 100
    mu = 5e-8

    n_iters = 10

    p = LogisticRegression(n_agent=n_agent, m=m, dim=dim, noise_ratio=0.05, graph_type='er', kappa=kappa, graph_params=0.3)
    print(p.n_edges)


    x_0 = np.random.rand(dim, n_agent)
    x_0_mean = x_0.mean(axis=1)
    W, alpha = generate_mixing_matrix(p)
    print('alpha = ' + str(alpha))


    eta = 2/(p.L + p.sigma)
    n_inner_iters = int(m * 0.05)
    batch_size = int(m / 10)
    batch_size = 10
    n_dgd_iters = n_iters * 20
    n_svrg_iters = n_iters * 20 
    n_dsgd_iters = int(n_iters * m / batch_size)


    single_machine = [
        GD(p, n_iters=n_iters, eta=eta, x_0=x_0_mean),
        SGD(p, n_iters=n_dsgd_iters, eta=eta*3, batch_size=batch_size, x_0=x_0_mean, diminishing_step_size=True),
        NAG(p, n_iters=n_iters, x_0=x_0_mean),
        SVRG(p, n_iters=n_svrg_iters, n_inner_iters=n_inner_iters, eta=eta/20, x_0=x_0_mean),
        SARAH(p, n_iters=n_svrg_iters, n_inner_iters=n_inner_iters, eta=eta/20, x_0=x_0_mean),
        ]


    distributed = [
        DGD_tracking(p, n_iters=n_dgd_iters, eta=eta/10, x_0=x_0, W=W),
        DSGD(p, n_iters=n_dsgd_iters, eta=eta*2, batch_size=batch_size, x_0=x_0, W=W, diminishing_step_size=True),
        EXTRA(p, n_iters=n_dgd_iters, eta=eta/2, x_0=x_0, W=W),
        NIDS(p, n_iters=n_dgd_iters, eta=eta, x_0=x_0, W=W),

        ADMM(p, n_iters=n_iters, rho=1, x_0=x_0_mean),
        DANE(p, n_iters=n_iters, mu=mu, x_0=x_0_mean)
        ]

    network = [
        NetworkSVRG(p, n_iters=n_svrg_iters, n_inner_iters=n_inner_iters, eta=eta/20, mu=mu, x_0=x_0, W=W, batch_size=batch_size),
        NetworkSARAH(p, n_iters=n_svrg_iters, n_inner_iters=n_inner_iters, eta=eta/20, mu=mu, x_0=x_0, W=W, batch_size=batch_size),
        NetworkDANE(p, n_iters=n_iters, mu=mu, x_0=x_0, W=W),
        ]

    exps = single_machine + distributed + network

    res = run_exp(exps, kappa=kappa, max_iter=n_iters, name='logistic_regression', n_cpu_processes=4, save=True)


    plt.show()
