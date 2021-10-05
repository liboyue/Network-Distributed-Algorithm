#!/usr/bin/env python
# coding=utf-8
import numpy as np
from nda import log
from nda.problems import Problem


class LinearRegression(Problem):
    '''f(w) = 1/n \sum f_i(w) + r * g(w) = 1/n \sum 1/2m || Y_i - X_i w ||^2 + r * g(w)'''

    def __init__(self, n_agent, m, dim, noise_variance=0.1, kappa=10, **kwargs):

        super().__init__(n_agent, m, dim, **kwargs)

        self.noise_variance = noise_variance
        self.kappa = kappa

        # Generate X
        self.X_total, self.L, self.sigma, self.S = self.generate_x(self.m_total, self.dim, self.kappa)

        # Generate Y and the optimal solution
        self.x_0 = self.w_0 = np.random.rand(self.dim)
        self.Y_0_total = self.X_total.dot(self.w_0)
        self.Y_total = self.Y_0_total + np.sqrt(self.noise_variance) * np.random.randn(self.m_total)

        # Split data
        self.X = self.split_data(self.X_total)
        self.Y = self.split_data(self.Y_total)

        # Pre-calculate matrix products to accelerate gradient and function value evaluations
        self.H = self.X_total.T.dot(self.X_total) / self.m_total
        self.H_list = np.einsum('ikj,ikl->ijl', self.X, self.X) / self.m

        self.X_T_Y = self.X_total.T.dot(self.Y_total) / self.m_total
        self.X_T_Y_list = np.einsum('ikj,ik->ij', self.X, self.Y) / self.m

        if self.is_smooth is True:
            self.x_min = self.w_min = np.linalg.solve(self.X_total.T.dot(self.X_total) + 2 * self.m_total * self.r * np.eye(self.dim), self.X_total.T.dot(self.Y_total))
        else:
            import sys
            sys.path.append("..")
            from optimizers.utils import FISTA
            self.x_min, _ = FISTA(self.grad_h, np.random.randn(self.dim), self.L, self.r, n_iters=100000)
            self.w_min = self.x_min

        self.f_min = self.f(self.x_min)

        log.info('beta = %.4f', np.linalg.norm(self.H_list - self.H, ord=2, axis=(1, 2)).max())

    def generate_x(self, n_samples, dim, kappa):
        '''Helper function to generate data'''

        powers = - np.log(kappa) / np.log(dim) / 2

        S = np.power(np.arange(dim) + 1, powers)
        X = np.random.randn(n_samples, dim)  # Random standard Gaussian data
        X *= S                               # Conditioning
        X_list = self.split_data(X)

        max_norm = max([np.linalg.norm(X_list[i].T.dot(X_list[i]), 2) / X_list[i].shape[0] for i in range(self.n_agent)])
        X /= max_norm

        return X, 1, 1 / kappa, np.diag(S)

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

        if w.ndim == 1:
            if type(j) is int:
                j = [j]

            if i is None and j is None:  # Return the full gradient
                return self.H.dot(w) - self.X_T_Y
            elif i is not None and j is None:  # Return the local gradient
                return self.H_list[i].dot(w) - self.X_T_Y_list[i]
            elif i is None and j is not None:  # Return the stochastic gradient
                return (self.X_total[j].dot(w) - self.Y_total[j]).dot(self.X_total[j]) / len(j)
            else:  # Return the stochastic gradient
                return (self.X[i][j].dot(w) - self.Y[i][j]).dot(self.X[i][j]) / len(j)

        elif w.ndim == 2:
            if i is None and j is None:  # Return the distributed gradient
                return np.einsum('ijk,ki->ji', self.H_list, w) - self.X_T_Y_list.T
            elif i is None and j is not None:  # Return the stochastic gradient
                res = []
                for i in range(self.n_agent):
                    if type(j[i]) is int:
                        samples = [j[i]]
                    else:
                        samples = j[i]
                    res.append((self.X[i][samples].dot(w[:, i]) - self.Y[i][samples]).dot(self.X[i][samples]) / len(samples))
                return np.array(res).T
            else:
                log.fatal('For distributed gradients j must be None')
        else:
            log.fatal('Parameter dimension should only be 1 or 2')

    def h(self, w, i=None, j=None):
        '''Function value of h(x) at w. If i is None, returns h(x); if i is not None but j is, returns the function value at the i-th machine; otherwise,return the function value of j-th sample at the i-th machine.'''

        if i is None and j is None:  # Return the function value
            Z = np.sqrt(2 * self.m_total)
            return np.sum((self.Y_total / Z - (self.X_total / Z).dot(w)) ** 2)
        elif i is not None and j is None:  # Return the function value at machine i
            # return np.sum( (self.Y[i] - self.X[i].dot(w))**2 ) / 2 / self.m[i]
            return np.sum((self.Y[i] - self.X[i].dot(w)) ** 2) / 2 / self.m
        elif i is not None and j is not None:  # Return the function value of sample j at machine i
            return np.sum((self.Y[i][j] - self.X[i][j].dot(w)) ** 2) / 2
        else:
            log.fatal('When i is None, j mush be None')

    def hessian(self, w=None, i=None, j=None):
        '''Hessian matrix at w. If i is None, returns the full Hessian matrix; if i is not None but j is, returns the hessian matrix at the i-th machine; otherwise,return the hessian matrix of j-th sample at the i-th machine.'''

        if i is None:  # Return the full hessian matrix
            return self.H
        elif j is None:  # Return the hessian matrix at machine i
            return self.H_list[i]
        else:  # Return the hessian matrix of sample j at machine i
            return self.X[i][np.newaxis, j, :].T.dot(self.X[i][np.newaxis, j, :])


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    n = 10
    m = 1000
    dim = 10
    noise_variance = 0.01

    p = LinearRegression(n, m, dim, noise_variance=noise_variance, n_edges=4 * n, balanced=False)
    log.info(p.m)
    p.grad_check()
    p.distributed_check()

    # p = LinearRegression(n, m, dim, noise_variance=noise_variance, n_edges=4*n)
    p.plot_graph()

    log.info('w_min = ' + str(p.w_min))
    log.info('f(w_min) = ' + str(p.f(p.w_min)))
    log.info('f_0(w_min) = ' + str(p.f(p.w_min, 0)))
    log.info('|| g(w_min) || = ' + str(np.linalg.norm(p.grad(p.w_min))))
    log.info('|| g_0(w_min) || = ' + str(np.linalg.norm(p.grad(p.w_min, 0))))

    plt.show()
