#!/usr/bin/env python
# coding=utf-8
import numpy as np
from numpy.linalg import norm
from .utils import generate_mixing_matrix, eps

def relative_error(w, w_0):
    return norm(w - w_0) / norm(w_0)

class Optimizer(object):
    '''The base optimizer class, which handles logging, convergence/divergence checking.'''

    def __init__(self, p, n_iters=100, x_0=None, W=None, verbose=False):

        self.x_0 = x_0
        self.n_iters = n_iters
        self.verbose = verbose
        if self.verbose == True:
            self.history = []

        self.name = self.__class__.__name__

        self.p = p
        if hasattr(p, 'x_min'):
            self.x_min = p.x_min
        self.n_agent = p.n_agent
        self.m = p.m
        self.dim = p.dim
        if hasattr(p, 'n_edges'):
            self.n_edges = p.n_edges
        self.L = p.L
        self.sigma = p.sigma

        self.t = 0

        self.x = None
        self.s = None

        self.n_comm = np.zeros(self.n_iters+1)
        self.n_grad = np.zeros(self.n_iters+1)
        self.func_error = np.zeros(self.n_iters+1)
        self.var_error = np.zeros(self.n_iters+1)

        if hasattr(p, 'G'):
            if W is None:
                self.W = generate_mixing_matrix(p.G)
            else:
                self.W = W

            W_min_diag = min(np.diag(self.W))
            tmp = (1 - 1e-1) / (1 - W_min_diag)
            self.W_s = self.W*tmp + np.eye(self.n_agent)*(1 - tmp)
        

    def f(self, w, i=None, j=None):
        return self.p.f(w, i, j)


    def grad(self, w, i=None, j=None):
        '''Gradient wrapper. Provide logging function.'''

        if (i == None): # The full gradient
            self.n_grad[self.t] += self.n_agent * self.m
        elif j == None: # The gradient in machine i
            self.n_grad[self.t] += self.m
        else: # Return the gradient of sample j in machine i
            self.n_grad[self.t] += 1

        return self.p.grad(w, i, j)


    def hessian(self, w, i=None, j=None):
        return self.p.hessian(w, i, j)
 

    def init(self):
        if self.x_0 is None:
            self.x_0 = np.random.rand(self.dim, self.n_agent)
        self.x = self.x_0.copy()

    def save_metric(self):
        self.func_error[self.t] = (self.f(self.x.mean(axis=1)) - self.f(self.x_min)) / self.f(self.x_min)
        self.var_error[self.t] = relative_error(self.x.mean(axis=1), self.x_min)


    def save_history(self):
        if len(self.x.shape) == 1:
            self.history.append({
                'x': self.x.copy()
                })
        else:
            self.history.append({
                'x': self.x.mean(axis=1).copy()
                })


    def plot_history(self):
        pass


    def get_results(self):
        return {
                'x': self.x.mean(axis=1),
                'var_error': self.var_error[:self.t+1],
                'func_error': self.func_error[:self.t+1],
                'n_comm': self.n_comm[:self.t+1],
                'n_grad': self.n_grad[:self.t+1] / self.p.m / self.p.n_agent
                }


    def get_name(self):
        return self.name


    def optimize(self):
        self.init()

        # Initial value
        if hasattr(self, 'x_min'):
            self.save_metric()

        if self.verbose == True:
            self.save_history()


        for self.t in range(1, self.n_iters+1):

            self.update()

            self.n_comm[self.t] += self.n_comm[self.t-1]
            self.n_grad[self.t] += self.n_grad[self.t-1]

            if hasattr(self, 'x_min'):
                self.save_metric()

            if self.verbose == True:
                self.save_history()

            if hasattr(self, 'x_min') and self.convergence_check() == True:
                break

        # endfor

        return self.get_results()


    def convergence_check(self):
        ''' Convergence check'''

        if norm(self.p.grad(self.x.mean(axis=1))) < eps:
            return True

        if norm(self.x.mean(axis=1) - self.x_min) > 5e1:
            return True


    def update(self):
        pass
