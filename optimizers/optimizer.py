#!/usr/bin/env python
# coding=utf-8
import numpy as np
from numpy.linalg import norm

from .utils import eps

def relative_error(w, w_0):
    return norm(w - w_0) / norm(w_0)

class Optimizer(object):
    '''The base optimizer class, which handles logging, convergence/divergence checking.'''

    def __init__(self, p, n_iters=100, x_0=None, W=None, verbose=False):

        self.name = self.__class__.__name__

        if x_0 is not None:
            if x_0.ndim == 1:
                self.x_0 = x_0
            else:
                self.x_0 = x_0.mean(axis=1)
        else:
            self.x_0 = np.random.rand(self.p.dim)

        self.n_iters = n_iters
        self.verbose = verbose
        self.history = []

        self.p = p

        self.t = 0
        self.x = self.x_0.copy()

        self.n_grad = np.zeros(self.n_iters+1)
        self.func_error = np.zeros(self.n_iters+1)
        self.var_error = np.zeros(self.n_iters+1)


    def f(self, w, i=None, j=None):
        return self.p.f(w, i, j)


    def grad(self, w, i=None, j=None):
        '''Gradient wrapper. Provide logging function.'''

        if i is None: # The full gradient
            self.n_grad[self.t] += self.p.m_total
        elif j is None: # The gradient at machine i
            self.n_grad[self.t] += self.p.m[i]
        else: # Return the gradient of sample j at machine i
            if type(j) is np.ndarray:
                self.n_grad[self.t] += len(j)
            else:
                self.n_grad[self.t] += 1

        return self.p.grad(w, i, j)


    def grad_full(self, w, i=None):
        '''Gradient wrapper. Provide logging function.'''

        if i is None: # The full gradient
            self.n_grad[self.t] += self.p.m_total
        else:
            if type(i) is np.ndarray:
                self.n_grad[self.t] += len(i)
            else:
                self.n_grad[self.t] += 1

        return self.p.grad_full(w, i)


    def hessian(self, w, i=None, j=None):
        return self.p.hessian(w, i, j)
 

    def init(self):
        pass


    def save_metric(self):
        self.func_error[self.t] = (self.f(self.x) - self.p.f_min) / self.p.f_min
        self.var_error[self.t] = relative_error(self.x, self.p.x_min)


    def save_history(self):
        self.history.append({
            'x': self.x.copy()
            })


    def plot_history(self):
        pass


    def get_results(self):

        res = {
                'x': self.x,
                'var_error': self.var_error[:self.t+1],
                'func_error': self.func_error[:self.t+1],
                'n_grad': self.n_grad[:self.t+1] / self.p.m_total
                }

        if self.verbose == True:
            res['history'] = self.history

        return res


    def get_name(self):
        return self.name


    def optimize(self):
        self.init()

        # Initial value
        if hasattr(self.p, 'x_min'):
            self.save_metric()

        if self.verbose == True:
            self.save_history()


        for self.t in range(1, self.n_iters+1):

            # Initialized every iteration
            self.init_iter()

            # The actual update step for optimization variable
            self.update()

            if hasattr(self.p, 'x_min'):
                self.save_metric()

            if self.verbose == True:
                self.save_history()

            if hasattr(self.p, 'x_min') and self.convergence_check() == True:
                break

        # endfor

        return self.get_results()


    def init_iter(self):
            self.n_grad[self.t] += self.n_grad[self.t-1]

    def convergence_check(self):
        ''' Convergence check'''

        if norm(self.p.grad(self.x)) < eps:
            return True

        if norm(self.x - self.p.x_min) > 5e1:
            return True


    def update(self):
        pass
