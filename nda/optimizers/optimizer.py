#!/usr/bin/env python
# coding=utf-8
import numpy as np

try:
    import cupy as xp
except ImportError:
    import numpy as xp

from nda.optimizers.utils import eps
from nda import log

norm = xp.linalg.norm


def relative_error(w, w_0):
    return norm(w - w_0) / norm(w_0)


class Optimizer(object):
    '''The base optimizer class, which handles logging, convergence/divergence checking.'''

    def __init__(self, p, n_iters=100, x_0=None, W=None, save_metric_frequency=1, is_distributed=True, extra_metrics=[], early_stopping=True, grad_eps=eps, var_eps=eps, f_eps=eps*eps):

        self.name = self.__class__.__name__
        self.p = p
        self.n_iters = n_iters
        self.save_metric_frequency = save_metric_frequency
        self.save_metric_counter = 0
        self.is_distributed = is_distributed
        self.early_stopping = early_stopping
        self.grad_eps = grad_eps
        self.var_eps = var_eps
        self.f_eps = f_eps
        self.is_initialized = False

        if W is not None:
            self.W = np.array(W)

        if x_0 is not None:
            self.x_0 = np.array(x_0)
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
        self.metric_names = ['t', 'n_grads', 'f']
        if self.is_distributed:
            self.metric_names += ['comm_rounds']
        self.metric_names += extra_metrics

        if self.p.x_min is not None:
            self.metric_names += ['var_error']

    def cuda(self):

        log.debug("Copying data to GPU")

        self.p.cuda()

        for k in self.__dict__:
            if type(self.__dict__[k]) == np.ndarray:
                self.__dict__[k] = xp.array(self.__dict__[k])

    def f(self, *args, **kwargs):
        return self.p.f(*args, **kwargs)

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
                if type(j) is xp.ndarray:
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

    def compute_metric(self, metric_name, x):

        if metric_name == 't':
            res = self.t
        elif metric_name == 'comm_rounds':
            res = self.comm_rounds
        elif metric_name == 'n_grads':
            res = self.n_grads
        elif metric_name == 'f':
            res = self.f(x)
        elif metric_name == 'f_test':
            res = self.f(x, split='test')
        elif metric_name == 'var_error':
            res = relative_error(x, self.p.x_min)
        elif metric_name == 'train_accuracy':
            acc = self.p.accuracy(x, split='train')
            if type(acc) is tuple:
                acc = acc[0]
            res = acc
        elif metric_name == 'test_accuracy':
            acc = self.p.accuracy(x, split='test')
            if type(acc) is tuple:
                acc = acc[0]
            res = acc
        elif metric_name == 'grad_norm':
            res = norm(self.p.grad(x))
        else:
            raise NotImplementedError(f'Metric {metric_name} is not implemented')

        return res

    def save_metrics(self, x=None):

        self.save_metric_counter %= self.save_metric_frequency
        self.save_metric_counter += 1

        if x is None:
            x = self.x

        if x.ndim > 1:
            x = x.mean(axis=1)

        self.metrics.append(
            [self.compute_metric(name, x) for name in self.metric_names]
        )

    def get_metrics(self):
        self.metrics = [[_metric.item() if type(_metric) is xp.ndarray else _metric for _metric in _metrics] for _metrics in self.metrics]
        return self.metric_names, np.array(self.metrics)

    def get_name(self):
        return self.name

    def optimize(self):

        self.init()

        # Initial value
        self.save_metrics()

        for self.t in range(1, self.n_iters + 1):

            # The actual update step for optimization variable
            self.update()

            self.save_metrics()

            if self.early_stopping is True and self.check_stopping_conditions() is True:
                break

        # endfor

        return self.get_metrics()

    def check_stopping_conditions(self):
        '''Check stopping conditions'''

        if self.x.ndim > 1:
            x = self.x.mean(axis=1)
        else:
            x = self.x

        if self.grad_eps is not None:
            grad_norm = norm(self.p.grad(x))
            if grad_norm < self.grad_eps:
                log.info('Gradient norm converged')
                return True
            elif grad_norm > 100 * self.p.dim:
                log.info('Gradient norm diverged')
                return True

        if self.p.x_min is not None and self.var_eps is not None:
            distance = norm(x - self.p.x_min) / norm(self.p.x_min)
            if distance < self.var_eps:
                log.info('Variable converged')
                return True

            if distance > 100:
                log.info('Variable diverged')
                return True

        if self.p.f_min is not None and self.f_eps is not None:
            distance = np.abs(self.p.f(x) / self.p.f_min - 1)
            if distance < self.f_eps:
                log.info('Function value converged')
                return True

            if distance > 100:
                log.info('Function value diverged')
                return True

        return False

    def init(self):
        pass

    def update(self):
        pass
