#!/usr/bin/env python
# coding=utf-8
import numpy as np

try:
    import cupy as xp
except ImportError:
    import numpy as xp

import networkx as nx
import cvxpy as cvx
import os

eps = 1e-6

def NAG(grad, x_0, L, sigma, n_iters=100, eps=eps):
    '''Nesterov's Accelerated Gradient Descent for strongly convex functions'''

    x = y = x_0
    root_kappa = xp.sqrt(L / sigma)
    r = (root_kappa - 1) / (root_kappa + 1)
    r_1 = 1 + r
    r_2 = r

    for t in range(n_iters):
        y_last = y

        _grad = grad(y)
        if xp.linalg.norm(_grad) < eps:
            break

        y = x - _grad / L
        x = r_1*y - r_2*y_last

    return y, t


def GD(grad, x_0, eta, n_iters=100, eps=eps):
    '''Gradient Descent'''

    x = x_0
    for t in range(n_iters):

        _grad = grad(x)
        if xp.linalg.norm(_grad) < eps:
            break

        x -= eta * _grad

    return x, t + 1


def Sub_GD(grad, x_0, n_iters=100, eps=eps):
    '''Sub-gradient Descent'''

    R = xp.linalg.norm(x_0)
    x = x_0
    for t in range(n_iters):
        eta_t = R / xp.sqrt(t)

        g_t = grad(x)
        if xp.linalg.norm(g_t) < eps:
            break

        g_t /= xp.linalg.norm(g_t)
        x -= eta_t * g_t

    return x, t + 1


def FISTA(grad_f, x_0, L, LAMBDA, n_iters=100, eps=1e-10):
    '''FISTA'''
    r = xp.zeros(n_iters+1)

    for t in range(1, n_iters + 1):
        r[t] = 0.5 + xp.sqrt(1 + 4 * r[t - 1] ** 2) / 2

    gamma = (1 - r[:n_iters]) / r[1:]

    x = x_0.copy()
    y = x_0.copy()

    for t in range(1, n_iters):

        _grad = grad_f(x)
        if xp.linalg.norm(_grad) < eps:
            break

        x -= _grad / L
        y_new = xp.sign(x) * xp.maximum(xp.abs(x) - LAMBDA/L, 0)
        x = (1 - gamma[t]) * y_new + gamma[t] * y
        y = y_new

    return y, t + 1

def generate_mixing_matrix(p):
    return symmetric_fdla_matrix(p.G)

def asymmetric_fdla_matrix(G, m):
    n = G.number_of_nodes()

    ind = nx.adjacency_matrix(G).toarray() + np.eye(n)
    ind = ~ind.astype(bool)

    average_vec = m / m.sum()
    average_matrix = np.ones((n, 1)).dot(average_vec[np.newaxis, :]).T
    one_vec = np.ones(n)

    W = cvx.Variable((n, n))

    if ind.sum() == 0:
        prob = cvx.Problem(cvx.Minimize(cvx.norm(W - average_matrix)),
                            [
                                cvx.sum(W, axis=1) == one_vec,
                                cvx.sum(W, axis=0) == one_vec
                            ])
    else:
        prob = cvx.Problem(cvx.Minimize(cvx.norm(W - average_matrix)),
                            [
                                W[ind] == 0,
                                cvx.sum(W, axis=1) == one_vec,
                                cvx.sum(W, axis=0) == one_vec
                            ])
    prob.solve()

    W = W.value
    # W = (W + W.T) / 2
    W[ind] = 0
    W -= np.diag(W.sum(axis=1) - 1)
    alpha = np.linalg.norm(W - average_matrix, 2)

    return W, alpha



def symmetric_fdla_matrix(G):

    n = G.number_of_nodes()

    ind = nx.adjacency_matrix(G).toarray() + np.eye(n)
    ind = ~ind.astype(bool)

    average_matrix = np.ones((n, n)) / n
    one_vec = np.ones(n)

    W = cvx.Variable((n, n))

    if ind.sum() == 0:
        prob = cvx.Problem(cvx.Minimize(cvx.norm(W - average_matrix)),
                            [
                                W == W.T,
                                cvx.sum(W, axis=1) == one_vec
                            ])
    else:
        prob = cvx.Problem(cvx.Minimize(cvx.norm(W - average_matrix)),
                            [
                                W[ind] == 0,
                                W == W.T,
                                cvx.sum(W, axis=1) == one_vec
                            ])
    prob.solve()

    W = W.value
    W = (W + W.T) / 2
    W[ind] = 0
    W -= np.diag(W.sum(axis=1) - 1)
    alpha = np.linalg.norm(W - average_matrix, 2)

    return np.array(W), alpha

def relative_error(x, y):
    return xp.linalg.norm(x - y) / xp.linalg.norm(y)
