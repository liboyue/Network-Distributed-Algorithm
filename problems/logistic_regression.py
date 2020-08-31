#!/usr/bin/env python
# coding=utf-8
import numpy as np
from scipy import optimize as opt
from .problem import Problem


class LogisticRegression(Problem):
    '''f(w) =  - (\sum y_i log(1/(1 + exp(w^T x_i))) + (1 - y_i) log (1 - 1/(1 + exp(w^T x_i)))) + \frac{\lambda}{2} \Vert w \Vert^2 '''
    
    def _logit(self, X, w):
        return 1 / (1 + np.exp(-X.dot(w)))

    def __init__(self, n_agent, m_mean, dim, kappa=10, noise_ratio=0.01, **kwargs):

        super().__init__(n_agent, m_mean, dim, **kwargs)

        self.noise_ratio = noise_ratio
        self.kappa = kappa
        if kappa == 1:
            self.LAMBDA = 100
        else:
            self.LAMBDA = 1 / (self.kappa - 1)
        self.L = 1
        self.sigma = self.LAMBDA

        # self.L = np.linalg.eigvals(X_tmp.T.dot(X_tmp) / self.m / self.n).max()

        self._generate_data()

        self.f_min = self.f(self.w_min)
        # print(w_0)
        # print(self.w_min)
        # print(np.linalg.norm(w_0 - self.w_min))


    def _generate_data(self):
        # Generate data
        X = np.random.randn(self.m_total, self.dim)
        norm = np.sqrt(np.linalg.norm(X.T.dot(X), 2) / self.m_total)
        X /= norm + self.LAMBDA
        self.X_total = X

        # Generate labels
        w_0 = np.random.rand(self.dim)
        Y_0_total = self._logit(self.X_total, w_0)
        Y_0_total[Y_0_total > 0.5] = 1
        Y_0_total[Y_0_total <= 0.5] = 0

        # P = self._logit(X, w_0, self.noise_variance * np.random.randn(n, m))
        #Y = numpy.random.binomial(1, p)
        # self.Y = np.random.binomial(1, P)
        # self.X = X
        
        if self.noise_ratio is not None:
            noise = np.random.binomial(1, self.noise_ratio, self.m_total)
            self.Y_total = np.multiply(noise - Y_0_total, noise) + np.multiply(Y_0_total, 1 - noise)
        else:
            self.Y_total = self.Y_0_total

        self.X = self.split_data(self.m, self.X_total)
        self.Y = self.split_data(self.m, self.Y_total)

        self.x_min = self.w_min = opt.minimize(
                self.f,
                np.random.rand(self.dim),
                jac=self.grad,
                method='BFGS',
                options={'gtol' :1e-8}
                ).x


    def grad(self, w, i=None, j=None):
        '''Gradient at w. If i is None, returns the full gradient; if i is not None but j is, returns the gradient in the i-th machine; otherwise,return the gradient of j-th sample in i-th machine. Note j can be a list.'''

        if i is None: # Return the full gradient
            return self.X_total.T.dot(self._logit(self.X_total, w) - self.Y_total) / self.m_total + w * self.LAMBDA
        elif j is None: # Return the gradient in machine i
                return self.X[i].T.dot(self._logit(self.X[i], w) - self.Y[i]) / self.m_mean  + w * self.LAMBDA
        else: # Return the gradient of sample j in machine i
            if type(j) is np.ndarray:
                return (self._logit(self.X[i][j], w) - self.Y[i][j]).dot(self.X[i][j]) / len(j) + w * self.LAMBDA
            else:
                return (self._logit(self.X[i][j], w) - self.Y[i][j]) * self.X[i][j] + w * self.LAMBDA


    def grad_full(self, w, i=None):
        '''Full gradient at w. If i is None, returns the full gradient; if i is not None, returns the gradient for the i-th sample in the whole dataset.'''

        if i is None: # Return the full gradient
            return self.X_total.T.dot(self._logit(self.X_total, w) - self.Y_total) / self.m_total + w * self.LAMBDA
        else: # Return the gradient of sample i
            if type(i) is np.ndarray:
                return (self._logit(self.X_total[i], w) - self.Y_total[i]).dot(self.X_total[i]) / len(i) + w * self.LAMBDA
            else:
                return (self._logit(self.X_total[i], w) - self.Y_total[i]) * self.X_total[i] + w * self.LAMBDA


    def f(self, w, i=None, j=None):
        '''Function value at w. If i is None, returns f(x); if i is not None but j is, returns the function value in the i-th machine; otherwise,return the function value of j-th sample in i-th machine.'''

        if i == None: # Return the function value
            tmp = self.X_total.dot(w)
            return - np.sum(
                    (self.Y_total - 1) * tmp - np.log(1 + np.exp(-tmp))
                    ) / self.m_total + np.sum(w**2) * self.LAMBDA / 2

        elif j == None: # Return the function value in machine i
            tmp = self.X[i].dot(w)
            return - np.sum(
                    (self.Y[i] - 1) * tmp - np.log(1 + np.exp(-tmp))
                    ) / self.m_mean + np.sum(w**2) * self.LAMBDA / 2
        else: # Return the gradient of sample j in machine i
            tmp = self.X[i][j].dot(w)
            return -((self.Y[i][j] - 1) * tmp - np.log(1 + np.exp(-tmp))) + np.sum(w**2) * self.LAMBDA / 2

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    n = 10
    m = 1000
    dim = 10
    noise_ratio = 0.01

    p = LogisticRegression(n, m, dim, noise_ratio=noise_ratio, balanced=False)
    p.grad_check()
    p.distributed_check()

    p = LogisticRegression(n, m, dim, noise_ratio=noise_ratio, n_edges=4*n)
    p.grad_check()
    p.distributed_check()
    # p.plot_graph()

    print('w_min = ' + str(p.w_min))
    print('f(w_min) = ' + str(p.f(p.w_min)))
    print('f_0(w_min) = ' + str(p.f(p.w_min, 0)))
    print('|| g(w_min) || = ' + str(np.linalg.norm(p.grad(p.w_min))))
    print('|| g_0(w_min) || = ' + str(np.linalg.norm(p.grad(p.w_min, 0))))

    # plt.show()
