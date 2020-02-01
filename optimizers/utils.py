#!/usr/bin/env python
# coding=utf-8
import numpy as np
import networkx as nx
import cvxpy as cvx

eps = 1e-6

def NAG(grad, x_0, L, sigma, n_iters=100):
    '''Nesterov's Accelerated Gradient Descent for strongly convex functions'''

    x = y = x_0
    root_kappa = np.sqrt(L / sigma)
    r = (root_kappa - 1) / (root_kappa + 1)
    r_1 = 1 + r
    r_2 = r

    for t in range(1, n_iters+1):
        y_last = y
        y = x - grad(x) / L
        x = r_1*y - r_2*y_last

        if np.linalg.norm(y) < eps:
            break

    return y, t


def GD(grad, x_0, eta, n_iters=100):
    '''Gradient Descent'''

    x = x_0
    for t in range(1, n_iters+1):
        x -= eta * grad(x)

        if np.linalg.norm(x) < eps:
            break

    return x, t


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

    return W, alpha

def relative_error(x, y):
    return np.linalg.norm(x - y) / np.linalg.norm(y)
