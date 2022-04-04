#!/usr/bin/env python
# coding=utf-8
import numpy as np

try:
    import cupy as xp
except ImportError:
    import numpy as xp

import multiprocessing as mp

from nda import log
from nda.problems import Problem


class LinearRegression(Problem):
    '''f(w) = 1/n \sum f_i(w) + r * g(w) = 1/n \sum 1/2m || Y_i - X_i w ||^2 + r * g(w)'''

    def __init__(self, noise_variance=0.1, kappa=10, **kwargs):

        self.noise_variance = noise_variance
        self.kappa = kappa

        super().__init__(**kwargs)

        # Pre-calculate matrix products to accelerate gradient and function value evaluations
        self.H = self.X_train.T.dot(self.X_train) / self.m_total
        self.X_T_Y = self.X_train.T.dot(self.Y_train) / self.m_total

        if xp.__name__ == 'cupy':
            log.info('Initializing using GPU')
            q = mp.Queue(2)
            pp = mp.Process(target=self._init, args=(q,))
            pp.start()
            pp.join()
            self.x_min = self.w_min = q.get()
            self.f_min = q.get()
        else:
            log.info('Initializing using CPU')
            self.x_min, self.f_min = self._init()

        # Pre-calculate matrix products to accelerate gradient and function value evaluations
        # After computing minimum to reduce memory copy
        self.H_list = np.einsum('ikj,ikl->ijl', self.X, self.X) / self.m
        self.X_T_Y_list = np.einsum('ikj,ik->ij', self.X, self.Y) / self.m
        log.info('beta = %.4f', np.linalg.norm(self.H_list - self.H, ord=2, axis=(1, 2)).max())
        log.info('Initialization done')

    def _init(self, result_queue=None):

        if xp.__name__ == 'cupy':
            self.cuda()

        if self.is_smooth is True:
            x_min = xp.linalg.solve(self.X_train.T.dot(self.X_train) + 2 * self.m_total * self.r * xp.eye(self.dim), self.X_train.T.dot(self.Y_train))

        else:
            from nda.optimizers.utils import FISTA
            x_min, _ = FISTA(self.grad_h, xp.random.randn(self.dim), self.L, self.r, n_iters=100000)

        f_min = self.f(x_min)
        log.info(f'f_min = {f_min}')

        if xp.__name__ == 'cupy':
            f_min = f_min.item()
            x_min = x_min.get()

        if result_queue is not None:
            result_queue.put(x_min)
            result_queue.put(f_min)

        return x_min, f_min

    def generate_data(self):

        def _generate_x(n_samples, dim, kappa):
            '''Helper function to generate data'''

            powers = - np.log(kappa) / np.log(dim) / 2

            S = np.power(np.arange(dim) + 1, powers)
            X = np.random.randn(n_samples, dim)  # Random standard Gaussian data
            X *= S                               # Conditioning
            X_list = self.split_data(X)

            max_norm = max([np.linalg.norm(X_list[i].T.dot(X_list[i]), 2) / X_list[i].shape[0] for i in range(self.n_agent)])
            X /= max_norm

            return X, 1, 1 / kappa, np.diag(S)

        # Generate X
        self.X_train, self.L, self.sigma, self.S = _generate_x(self.m_total, self.dim, self.kappa)

        # Generate Y and the optimal solution
        self.x_0 = self.w_0 = np.random.rand(self.dim)
        self.Y_0_train = self.X_train.dot(self.w_0)
        self.Y_train = self.Y_0_train + np.sqrt(self.noise_variance) * np.random.randn(self.m_total)

    def grad_h(self, w, i=None, j=None, split='train'):
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

        if w.ndim == 1:
            if type(j) is int:
                j = [j]

            if i is None and j is None:  # Return the full gradient
                return self.H.dot(w) - self.X_T_Y
            elif i is not None and j is None:  # Return the local gradient
                return self.H_list[i].dot(w) - self.X_T_Y_list[i]
            elif i is None and j is not None:  # Return the stochastic gradient
                return (self.X_train[j].dot(w) - self.Y_train[j]).dot(self.X_train[j]) / len(j)
            else:  # Return the stochastic gradient
                return (self.X[i][j].dot(w) - self.Y[i][j]).dot(self.X[i][j]) / len(j)

        elif w.ndim == 2:
            if i is None and j is None:  # Return the distributed gradient
                return xp.einsum('ijk,ki->ji', self.H_list, w) - self.X_T_Y_list.T
            elif i is None and j is not None:  # Return the stochastic gradient
                res = []
                for i in range(self.n_agent):
                    if type(j[i]) is int:
                        samples = [j[i]]
                    else:
                        samples = j[i]
                    res.append((self.X[i][samples].dot(w[:, i]) - self.Y[i][samples]).dot(self.X[i][samples]) / len(samples))
                return xp.array(res).T
            else:
                log.fatal('For distributed gradients j must be None')
        else:
            log.fatal('Parameter dimension should only be 1 or 2')

    def h(self, w, i=None, j=None, split='train'):
        '''Function value of h(x) at w. If i is None, returns h(x); if i is not None but j is, returns the function value at the i-th machine; otherwise,return the function value of j-th sample at the i-th machine.'''

        if i is None and j is None:  # Return the function value
            Z = xp.sqrt(2 * self.m_total)
            return xp.sum((self.Y_train / Z - (self.X_train / Z).dot(w)) ** 2)
        elif i is not None and j is None:  # Return the function value at machine i
            return xp.sum((self.Y[i] - self.X[i].dot(w)) ** 2) / 2 / self.m
        elif i is not None and j is not None:  # Return the function value of sample j at machine i
            return xp.sum((self.Y[i][j] - self.X[i][j].dot(w)) ** 2) / 2
        else:
            log.fatal('When i is None, j mush be None')

    def hessian(self, w=None, i=None, j=None):
        '''Hessian matrix at w. If i is None, returns the full Hessian matrix; if i is not None but j is, returns the hessian matrix at the i-th machine; otherwise,return the hessian matrix of j-th sample at the i-th machine.'''

        if i is None:  # Return the full hessian matrix
            return self.H
        elif j is None:  # Return the hessian matrix at machine i
            return self.H_list[i]
        else:  # Return the hessian matrix of sample j at machine i
            return self.X[i][xp.newaxis, j, :].T.dot(self.X[i][xp.newaxis, j, :])
