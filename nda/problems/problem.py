#!/usr/bin/env python
# coding=utf-8
import numpy as np

try:
    import cupy as xp
except ImportError:
    xp = np

import networkx as nx
import matplotlib.pyplot as plt
from nda import log


class Problem(object):
    '''The base problem class, which generates the random problem and supports function value and gradient evaluation'''

    def __init__(self, n_agent=20, m=1000, dim=40, graph_type='er', graph_params=None, regularization=None, r=0, dataset='random', sort=False, shuffle=False, normalize_data=False, gpu=False):

        self.n_agent = n_agent          # Number of agents
        self.m = m                      # Number of samples per agent
        self.dim = dim                  # Dimension of the variable
        self.X_total = None             # All data
        self.Y_total = None             # All labels
        self.X = []                     # Distributed data
        self.Y = []                     # Distributed labels
        self.x_0 = None                 # The true varibal value
        self.x_min = None               # The minimizer varibal value
        self.f_min = None               # The optimal function value
        self.L = None                   # The smoothness constant
        self.sigma = 0                  # The strong convexity constant
        self.is_smooth = True           # If the problem is smooth or not
        self.r = r
        self.graph_params = graph_params
        self.graph_type = graph_type
        self.dataset = dataset

        if dataset == 'random':
            self.m_total = m * n_agent      # Total number of data samples of all agents
            self.generate_data()
        else:
            from nda import datasets

            if dataset == 'gisette':
                self.X_train, self.Y_train, self.X_test, self.Y_test = datasets.Gisette(normalize=normalize_data).load()
            elif dataset == 'mnist':
                self.X_train, self.Y_train, self.X_test, self.Y_test = datasets.MNIST(normalize=normalize_data).load()

            else:
                self.X_train, self.Y_train, self.X_test, self.Y_test = datasets.LibSVM(name=dataset, normalize=normalize_data).load()

            self.X_train = np.append(self.X_train, np.ones((self.X_train.shape[0], 1)), axis=1)
            self.X_test = np.append(self.X_test, np.ones((self.X_test.shape[0], 1)), axis=1)
            self.m = self.X_train.shape[0] // n_agent
            self.m_total = self.m * n_agent

            self.X_train = self.X_train[:self.m_total]
            self.Y_train = self.Y_train[:self.m_total]
            self.dim = self.X_train.shape[1]


        if sort or shuffle:
            if sort:
                if self.Y_train.ndim > 1:
                    order = self.Y_train.argmax(axis=1).argsort()
                else:
                    order = self.Y_train.argsort()
            elif shuffle:
                order = np.random.permutation(len(self.X_train))

            self.X_train = self.X_train[order].copy()
            self.Y_train = self.Y_train[order].copy()

        # Split data
        self.X = self.split_data(self.X_train)
        self.Y = self.split_data(self.Y_train)

        self.generate_graph(graph_type=graph_type, params=graph_params)

        if regularization == 'l1':
            self.grad_g = self._grad_regularization_l1
            self.is_smooth = False

        elif regularization == 'l2':
            self.grad_g = self._grad_regularization_l2

    def cuda(self):
        log.debug("Copying data to GPU")

        # Copy every np.array to GPU if needed
        for k in self.__dict__:
            if type(self.__dict__[k]) == np.ndarray:
                self.__dict__[k] = xp.array(self.__dict__[k])

    def split_data(self, X):
        '''Helper function to split data according to the number of training samples per agent.'''
        if self.m * self.n_agent != len(X):
            log.fatal('Data cannot be distributed equally to %d agents' % self.n_agent)
        if X.ndim == 1:
            return X.reshape(self.n_agent, -1)
        else:
            return X.reshape(self.n_agent, self.m, -1)

    def grad(self, w, i=None, j=None):
        '''(sub-)Gradient of f(x) = h(x) + g(x) at w. Depending on the shape of w and parameters i and j, this function behaves differently:
        1. If w is a vector of shape (dim,)
            1.1 If i is None and j is None
                returns the full gradient.
            1.2 If i is not None and j is None
                returns the gradient at the i-th agent.
            1.3 If i is None and j is not None
                returns the i-th gradient of all training data.
            1.4 If i is not None and j is not None
                returns the gradient of the j-th data sample at the i-th agent.
            Note i, j can be integers, lists or vectors.
        2. If w is a matrix of shape (dim, n_agent)
            2.1 if j is None
                returns the gradient of each parameter at the corresponding agent
            2.2 if j is not None
                returns the gradient of each parameter of the j-th sample at the corresponding agent.
            Note j can be lists of lists or vectors.
        '''
        return self.grad_h(w, i=i, j=j) + self.grad_g(w)

    def grad_h(self, w, i=None, j=None):
        '''Gradient of h(x) at w. Depending on the shape of w and parameters i and j, this function behaves differently:
        1. If w is a vector of shape (dim,)
            1.1 If i is None and j is None
                returns the full gradient.
            1.2 If i is not None and j is None
                returns the gradient at the i-th agent.
            1.3 If i is None and j is not None
                returns the i-th gradient of all training data.
            1.4 If i is not None and j is not None
                returns the gradient of the j-th data sample at the i-th agent.
            Note i, j can be integers, lists or vectors.
        2. If w is a matrix of shape (dim, n_agent)
            2.1 if j is None
                returns the gradient of each parameter at the corresponding agent
            2.2 if j is not None
                returns the gradient of each parameter of the j-th sample at the corresponding agent.
            Note j can be lists of lists or vectors.
        '''
        pass

    def grad_g(self, w):
        '''Sub-gradient of g(x) at w. Returns the sub-gradient of corresponding parameters. w can be a vector of shape (dim,) or a matrix of shape (dim, n_agent).
        '''
        return 0

    def f(self, w, i=None, j=None, split='train'):
        '''Function value of f(x) = h(x) + g(x) at w. If i is None, returns the global function value; if i is not None but j is, returns the function value in the i-th machine; otherwise,return the function value of j-th sample in i-th machine.'''
        return self.h(w, i=i, j=j, split=split) + self.g(w)

    def hessian(self, *args, **kwargs):
        raise NotImplementedError

    def h(self, w, i=None, j=None, split='train'):
        '''Function value at w. If i is None, returns h(x); if i is not None but j is, returns the function value in the i-th machine; otherwise,return the function value of j-th sample in i-th machine.'''
        raise NotImplementedError

    def g(self, w):
        '''Function value of g(x) at w. Returns 0 if no regularization.'''
        return 0

    def _regularization_l1(self, w):
        return self.r * xp.abs(w).sum(axis=0)

    def _regularization_l2(self, w):
        return self.r * (w * w).sum(axis=0)

    def _grad_regularization_l1(self, w):
        g = xp.zeros(w.shape)
        g[w > 1e-5] = 1
        g[w < -1e-5] = -1
        return self.r * g

    def _grad_regularization_l2(self, w):
        return 2 * self.r * w

    def grad_check(self):
        '''Check whether the full gradient equals to the gradient computed by finite difference at a random point.'''
        w = xp.random.randn(self.dim)
        delta = xp.zeros(self.dim)
        grad = xp.zeros(self.dim)
        eps = 1e-4

        for i in range(self.dim):
            delta[i] = eps
            grad[i] = (self.f(w + delta) - self.f(w - delta)) / 2 / eps
            delta[i] = 0

        error = xp.linalg.norm(grad - self.grad(w))
        if error > eps:
            log.warn('Gradient implementation check failed with difference %.4f!' % error)
            return False
        else:
            log.info('Gradient implementation check succeeded!')
            return True

    def distributed_check(self):
        '''Check the distributed function and gradient implementations are correct.'''

        def _check_1d_gradient():

            w = xp.random.randn(self.dim)
            g = self.grad(w)
            g_i = g_ij = 0
            res = True

            for i in range(self.n_agent):
                _tmp_g_i = self.grad(w, i)
                _tmp_g_ij = 0
                for j in range(self.m):
                    _tmp_g_ij += self.grad(w, i, j)

                if xp.linalg.norm(_tmp_g_i - _tmp_g_ij / self.m) > 1e-5:
                    log.warn('Distributed graident check failed! Difference between local graident at agent %d and average of all local sample gradients is %.4f' % (i, xp.linalg.norm(_tmp_g_i - _tmp_g_ij / self.m)))
                    res = False

                g_i += _tmp_g_i
                g_ij += _tmp_g_ij

            g_i /= self.n_agent
            g_ij /= self.m_total

            if xp.linalg.norm(g - g_i) > 1e-5:
                log.warn('Distributed gradient check failed! Difference between global graident and average of local gradients is %.4f', xp.linalg.norm(g - g_i))
                res = False

            if xp.linalg.norm(g - g_ij) > 1e-5:
                log.warn('Distributed graident check failed! Difference between global graident and average of all sample gradients is %.4f' % xp.linalg.norm(g - g_ij))
                res = False

            return res

        def _check_2d_gradient():

            res = True
            w_2d = xp.random.randn(self.dim, self.n_agent)

            g_1d = 0
            for i in range(self.n_agent):
                g_1d += self.grad(w_2d[:, i], i=i)

            g_1d /= self.n_agent
            g_2d = self.grad(w_2d).mean(axis=1)

            if xp.linalg.norm(g_1d - g_2d) > 1e-5:
                log.warn('Distributed graident check failed! Difference between global gradient and average of distributed graidents is %.4f' % xp.linalg.norm(g_1d - g_2d))
                res = False

            g_2d_sample = self.grad(w_2d, j=xp.arange(self.m).reshape(-1, 1).repeat(self.n_agent, axis=1).T).mean(axis=1)

            if xp.linalg.norm(g_1d - g_2d_sample) > 1e-5:
                log.warn('Distributed graident check failed! Difference between global graident and average of all sample gradients is %.4f' % xp.linalg.norm(g_1d - g_2d_sample))
                res = False

            samples = xp.random.randint(0, self.m, (self.n_agent, 10))
            g_2d_stochastic = self.grad(w_2d, j=samples)
            for i in range(self.n_agent):
                g_1d_stochastic = self.grad(w_2d[:, i], i=i, j=samples[i])
                if xp.linalg.norm(g_1d_stochastic - g_2d_stochastic[:, i]) > 1e-5:
                    log.warn('Distributed graident check failed! Difference between distributed stoachastic gradient at agent %d and average of sample gradients is %.4f' % (i, xp.linalg.norm(g_1d_stochastic - g_2d_stochastic[:, i])))
                    res = False

            return res

        def _check_function_value():
            w = xp.random.randn(self.dim)
            f = self.f(w)
            f_i = f_ij = 0
            res = True

            for i in range(self.n_agent):
                _tmp_f_i = self.f(w, i)
                _tmp_f_ij = 0
                for j in range(self.m):
                    _tmp_f_ij += self.f(w, i, j)

                if xp.abs(_tmp_f_i - _tmp_f_ij / self.m) > 1e-10:
                    log.warn('Distributed function value check failed! Difference between local function value at agent %d and average of all local sample function values %d is %.4f' % (i, i, xp.abs(_tmp_f_i - _tmp_f_ij / self.m)))
                    res = False

                f_i += _tmp_f_i
                f_ij += _tmp_f_ij

            f_i /= self.n_agent
            f_ij /= self.m_total

            if xp.abs(f - f_i) > 1e-10:
                log.warn('Distributed function value check failed! Difference between the global function value and average of local function values is %.4f' % xp.abs(f - f_i))
                res = False

            if xp.abs(f - f_ij) > 1e-10:
                log.warn('Distributed function value check failed! Difference between the global function value and average of all sample function values is %.4f' % xp.abs(f - f_ij))
                res = False

            return res

        res = _check_function_value() & _check_1d_gradient() & _check_2d_gradient()
        if res:
            log.info('Distributed check succeeded!')
            return True
        else:
            return False

    def generate_graph(self, graph_type='expander', params=None):
        '''Generate connected connectivity graph according to the params.'''

        if graph_type == 'expander':
            G = nx.paley_graph(self.n_agent).to_undirected()
        elif graph_type == 'grid':
            G = nx.grid_2d_graph(*params)
        elif graph_type == 'cycle':
            G = nx.cycle_graph(self.n_agent)
        elif graph_type == 'path':
            G = nx.path_graph(self.n_agent)
        elif graph_type == 'star':
            G = nx.star_graph(self.n_agent - 1)
        elif graph_type == 'er':
            if params < 2 / (self.n_agent - 1):
                log.fatal("Need higher probability to create a connected E-R graph!")
            G = None
            while G is None or nx.is_connected(G) is False:
                G = nx.erdos_renyi_graph(self.n_agent, params)
        else:
            log.fatal('Graph type %s not supported' % graph_type)

        self.n_edges = G.number_of_edges()
        self.G = G

    def plot_graph(self):
        '''Plot the generated connectivity graph.'''

        plt.figure()
        nx.draw(self.G)
