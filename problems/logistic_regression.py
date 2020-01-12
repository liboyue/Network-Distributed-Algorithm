#!/usr/bin/env python
# coding=utf-8
import numpy as np
from scipy import optimize as opt
from .problem import Problem


class LogisticRegression(Problem):
    '''f(w) =  - (\sum y_i log(1/(1 + exp(w^T x_i))) + (1 - y_i) log (1 - 1/(1 + exp(w^T x_i)))) + \frac{\lambda}{2} \Vert w \Vert^2 '''
    
    def _logit(self, X, w):
        return 1 / (1 + np.exp(-X.dot(w)))

    def __init__(self, n, m, dim, kappa=10, noise_ratio=0.01, n_edges=None, prob=None):

        super().__init__(n, m, dim, n_edges=n_edges, prob=prob)

        self.noise_ratio = noise_ratio
        self.kappa = kappa
        if kappa == 1:
            self.LAMBDA = 100
        else:
            self.LAMBDA = 1 / (self.kappa - 1)
        self.L = 1
        self.sigma = self.LAMBDA

        def _generate_x():
            '''Helper function to generate data''' 

            X = np.random.randn(self.n_agent, self.m, self.dim)
            max_norm = 0
            for i in range(self.n_agent):
                max_norm = max(
                        np.sqrt(np.linalg.norm(X[i].T.dot(X[i]), 2) / m),
                        max_norm
                        )

            X /= max_norm + self.LAMBDA
            return X

        # Generate the problem
        self.X = _generate_x()

        w_0 = np.random.rand(dim)
        Y_0 = self._logit(self.X, w_0)
        Y_0[Y_0 > 0.5] = 1
        Y_0[Y_0 <= 0.5] = 0

        
        noise = np.random.binomial(1, self.noise_ratio, (self.n_agent, self.m))
        self.Y = np.multiply(noise - Y_0, noise) + np.multiply(Y_0, 1 - noise)


        self.w_min = opt.minimize(
                self.f,
                np.random.rand(dim),
                jac=self.grad,
                method='BFGS',
                options={'gtol' :1e-8}
                ).x

        self.x_min = self.w_min
        self.f_min = self.f(self.x_min)

    def grad(self, w, i=None, j=None):
        '''Gradient at w. If i is None, returns the full gradient; if i is not None but j is, returns the gradient in the i-th machine; otherwise,return the gradient of j-th sample in i-th machine. '''

        if i == None: # Return the full gradient
            tmp_X = self.X.reshape(-1, self.dim)
            tmp_Y = self.Y.reshape(-1)
            return tmp_X.T.dot(self._logit(tmp_X, w) - tmp_Y) / self.n_agent / self.m + w * self.LAMBDA
        elif j == None: # Return the gradient in machine i
                return self.X[i, :, :].T.dot(self._logit(self.X[i, :, :], w) - self.Y[i, :]) / self.m  + w * self.LAMBDA
        else: # Return the gradient of sample j in machine i
            return self.X[i, j, :] * (self._logit(self.X[i, j, :], w) - self.Y[i, j]) + w * self.LAMBDA

    def f(self, w, i=None, j=None):
        '''Function value at w. If i is None, returns f(x); if i is not None but j is, returns the function value in the i-th machine; otherwise,return the function value of j-th sample in i-th machine.'''

        if i == None: # Return the function value
            tmp = self.X.dot(w)
            return - np.sum(
                    (self.Y-1) * tmp - np.log(1 + np.exp(-tmp))
                    ) / self.n_agent / self.m + np.sum(w**2) * self.LAMBDA / 2

        elif j == None: # Return the function value in machine i
            tmp = self.X[i, :, :].dot(w)
            return - np.sum(
                    (self.Y[i, :]-1) * tmp - np.log(1 + np.exp(-tmp))
                    ) / self.m + np.sum(w**2) * self.LAMBDA / 2
        else: # Return the gradient of sample j in machine i
            tmp = self.X[i, j, :].dot(w)
            return -((self.Y[i, j]-1) * tmp - np.log(1 + np.exp(-tmp))) + np.sum(w**2) * self.LAMBDA / 2

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    n = 10
    m = 1000
    dim = 10
    noise_variance = 0.01

    p = LogisticRegression(n, m, dim, noise_variance=noise_variance)
    p.grad_check()
    p.distributed_check()

    p = LogisticRegression(n, m, dim, noise_variance=noise_variance, n_edges=4*n)
    p.plot_graph()

    print('w_min = ' + str(p.w_min))
    print('f(w_min) = ' + str(p.f(p.w_min)))
    print('f_0(w_min) = ' + str(p.f(p.w_min, 0)))
    print('|| g(w_min) || = ' + str(np.linalg.norm(p.grad(p.w_min))))
    print('|| g_0(w_min) || = ' + str(np.linalg.norm(p.grad(p.w_min, 0))))

    plt.show()
