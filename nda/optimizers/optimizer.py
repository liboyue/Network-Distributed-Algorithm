#!/usr/bin/env python
# coding=utf-8
import numpy as np
from numpy.linalg import norm

from nda.optimizers.utils import eps


def relative_error(w, w_0):
    return norm(w - w_0) / norm(w_0)


class Optimizer(object):
    '''The base optimizer class, which handles logging, convergence/divergence checking.'''

    def __init__(self, p, n_iters=100, x_0=None, W=None, save_metric_frequency=1, is_distributed=True, verbose=False):

        self.name = self.__class__.__name__
        self.p = p
        self.n_iters = n_iters
        self.verbose = verbose
        self.save_metric_frequency = save_metric_frequency
        self.save_metric_counter = 0
        self.is_distributed = is_distributed

        if W is not None:
            self.W = W

        if x_0 is not None:
            self.x_0 = x_0
        else:
            if self.is_distributed:
                self.x_0 = np.random.rand(p.dim, p.n_agent)
            else:
                self.x_0 = np.random.rand(self.p.dim)

        self.x = self.x_0.copy()

        self.t = 0
        self.comm_rounds = 0
        self.n_grads = 0
        self.metrics = []
        self.history = []
        self.metrics_columns = ['t', 'n_grads', 'f']
        if self.p.f_min is not None:
            self.metrics_columns += ['var_error']
        if hasattr(self.p, 'accuracy'):
            self.metrics_columns += ['train_accuracy', 'test_accuracy']
        if self.is_distributed:
            self.metrics_columns += ['comm_rounds']

    def f(self, w, i=None, j=None):
        return self.p.f(w, i, j)

    def grad(self, w, i=None, j=None):
        '''Gradient wrapper. Provide logging function.'''

        return self.grad_h(w, i=i, j=j) + self.grad_g(w)

    def hessian(self, *args, **kwargs):
        return self.p.hessian(*args, **kwargs)

    def grad_h(self, w, i=None, j=None):
        '''Gradient wrapper. Provide logging function.'''

        if w.ndim == 1:
            if i is None and j is None:
                self.n_grads += self.p.m_total  # Works for agents is list or integer
            elif i is not None and j is None:
                self.n_grads += self.p.m
            elif j is not None:
                if type(j) is int:
                    j = [j]
                self.n_grads += len(j)
        elif w.ndim == 2:
            if j is None:
                self.n_grads += self.p.m_total  # Works for agents is list or integer
            elif j is not None:
                if type(j) is np.ndarray:
                    self.n_grads += j.size
                elif type(j) is list:
                    self.n_grads += sum([1 if type(j[i]) is int else len(j[i]) for i in range(self.p.n_agent)])
                else:
                    raise NotImplementedError
        else:
            raise NotImplementedError

        return self.p.grad_h(w, i=i, j=j)

    def grad_g(self, w):
        '''Gradient wrapper. Provide logging function.'''

        return self.p.grad_g(w)

    def init(self):
        pass

    def save_history(self, x=None):
        if self.verbose is False:
            return
        if x is None:
            x = self.x
        self.history.append({'x': x.copy()})

    def save_metrics(self, x=None):

        self.save_metric_counter %= self.save_metric_frequency
        self.save_metric_counter += 1

        if x is None:
            x = self.x

        if x.ndim > 1:
            x = x.mean(axis=1)

        metrics = [self.t, self.n_grads, self.f(x)]

        if 'var_error' in self.metrics_columns:
            metrics.append(relative_error(x, self.p.x_min))

        if 'train_accuracy' in self.metrics_columns:
            acc = self.p.accuracy(x, split='train')
            if type(acc) is tuple:
                acc = acc[0]
            metrics.append(acc)
        if 'test_accuracy' in self.metrics_columns:
            acc = self.p.accuracy(x, split='test')
            if type(acc) is tuple:
                acc = acc[0]
            metrics.append(acc)

        if 'comm_rounds' in self.metrics_columns:
            metrics.append(self.comm_rounds)

        self.metrics.append(metrics)

    def get_metrics(self):
        return self.metrics_columns, np.array(self.metrics)

    def get_history(self):
        return self.history

    def get_name(self):
        return self.name

    def optimize(self):
        self.init()

        # Initial value
        self.save_metrics()
        self.save_history()

        for self.t in range(1, self.n_iters + 1):

            # Initialized every iteration
            self.init_iter()

            # The actual update step for optimization variable
            self.update()

            self.save_metrics()
            self.save_history()

            if self.check_stopping_conditions() is True:
                break

        # endfor

        return self.get_metrics()

    def init_iter(self):
        pass

    def check_stopping_conditions(self):
        '''Check stopping conditions'''

        if self.x.ndim > 1:
            x = self.x.mean(axis=1)
        else:
            x = self.x

        if norm(self.p.grad(x)) < eps:
            return True

        if self.p.x_min is not None:
            distance = norm(x - self.p.x_min)
            if distance < eps:
                return True

            if distance > 5:
                return True

        return False

    def update(self):
        pass
