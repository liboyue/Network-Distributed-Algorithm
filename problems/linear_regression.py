#!/usr/bin/env python
# coding=utf-8
import numpy as np
from .problem import Problem

class LinearRegression(Problem):
    '''f(w) = 1/n \sum f_i(w) = 1/n \sum 1/2m || Y_i - X_i w ||^2'''
    
    def __init__(self, n_agent, m_mean, dim, noise_variance=0.1, kappa=10, **kwargs):

        super().__init__(n_agent, m_mean, dim, **kwargs)

        self.noise_variance = noise_variance
        self.kappa = kappa

        # Generate X
        self.X_total, self.L, self.sigma, self.S = self.generate_x(self.m_total, self.dim, self.kappa)

        # Generate Y and the optimal solution
        self.x_0 = self.w_0 = np.random.rand(self.dim)
        self.Y_0_total = self.X_total.dot(self.w_0)
        self.Y_total = self.Y_0_total + np.sqrt(self.noise_variance) * np.random.randn(self.m_total)
        self.x_min = self.w_min = np.linalg.solve(self.X_total.T.dot(self.X_total), self.X_total.T.dot(self.Y_total))
        self.f_min = self.f(self.x_min)

        # Split data
        self.X = self.split_data(self.m, self.X_total)
        self.Y = self.split_data(self.m, self.Y_total)


        # Pre-calculate matrix products to accelerate gradient and function value evaluations
        self.H = self.X_total.T.dot(self.X_total) / self.m_total
        self.H_list = np.array([self.X[i].T.dot(self.X[i]) / self.m_mean for i in range(self.n_agent)])
        # self.H_list = np.array([self.X[i].T.dot(self.X[i]) / self.m[i] for i in range(self.n_agent)])
        self.X_T_Y = self.X_total.T.dot(self.Y_total) / self.m_total
        # self.X_T_Y_list = np.array([self.X[i].T.dot(self.Y[i]) / self.m[i] for i in range(self.n_agent)])
        self.X_T_Y_list = np.array([self.X[i].T.dot(self.Y[i]) / self.m_mean for i in range(self.n_agent)])

        print('beta = ' + str(max([np.linalg.norm(Hi - self.H, 2) for Hi in self.H_list])) )


    def generate_x(self, n_samples, dim, kappa):
        '''Helper function to generate data''' 

        powers = - np.log(kappa) / np.log(dim) / 2

        S = np.power(np.arange(dim)+1, powers)
        X = np.random.randn(n_samples, dim) # Random standard Gaussian data
        X *= S                              # Conditioning
        X_list = self.split_data(self.m, X)

        max_norm = max([np.linalg.norm(X_list[i].T.dot(X_list[i]), 2) / X_list[i].shape[0] for i in range(self.n_agent)])
        X /= max_norm

        return X, 1, 1/kappa, np.diag(S)


    def grad(self, w, i=None, j=None):
        '''Gradient at w. If i is None, returns the full gradient; if i is not None but j is, returns the gradient at the i-th machine; otherwise,return the gradient of j-th sample in i-th machine. Note i can be a vector if j is None, j can also be a vector.'''

        if i is None: # Return the full gradient
            return self.H.dot(w) - self.X_T_Y
        elif j is None: # Return the gradient at machine i
            return self.H_list[i].dot(w) - self.X_T_Y_list[i]
        else: # Return the gradient of sample j at machine i
            if type(j) is np.ndarray:
                return (self.X[i][j].dot(w) - self.Y[i][j]).dot(self.X[i][j]) / len(j)
            else:
                return (self.X[i][j].dot(w) - self.Y[i][j]) * self.X[i][j]


    def grad_full(self, w, i=None):
        '''Full gradient at w. If i is None, returns the full gradient; if i is not None, returns the gradient for the i-th sample in the whole dataset.'''

        if i is None: # Return the full gradient
            return self.H.dot(w) - self.X_T_Y
        else: # Return the gradient of sample i
            if type(i) is np.ndarray:
                return (self.X_total[i].dot(w) - self.Y_total[i]).dot(self.X_total[i]) / len(i)
            else:
                return (self.X_total[i].dot(w) - self.Y_total[i]) * self.X_total[i]



    def hessian(self, w=None, i=None, j=None):
        '''Hessian matrix at w. If i is None, returns the full Hessian matrix; if i is not None but j is, returns the hessian matrix in the i-th machine; otherwise,return the hessian matrix of j-th sample in i-th machine.'''

        if i is None: # Return the full hessian matrix
            return self.H
        elif j is None: # Return the hessian matrix at machine i
            return self.H_list[i]
        else: # Return the hessian matrix of sample j at machine i
            return self.X[i][np.newaxis, j, :].T.dot(self.X[i][np.newaxis, j, :])


    def f(self, w, i=None, j=None):
        '''Function value at w. If i is None, returns f(x); if i is not None but j is, returns the function value in the i-th machine; otherwise,return the function value of j-th sample in i-th machine.'''

        if i is None: # Return the function value
            Z = np.sqrt(2 * self.m_total)
            return np.sum((self.Y_total/Z - (self.X_total/Z).dot(w))**2)
        elif j is None: # Return the function value at machine i
            # return np.sum( (self.Y[i] - self.X[i].dot(w))**2 ) / 2 / self.m[i]
            return np.sum( (self.Y[i] - self.X[i].dot(w))**2 ) / 2 / self.m_mean
        else: # Return the function value of sample j at machine i
            return np.sum( (self.Y[i][j] - self.X[i][j].dot(w))**2 ) / 2


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    n = 10
    m = 1000
    dim = 10
    noise_variance = 0.01

    p = LinearRegression(n, m, dim, noise_variance=noise_variance, n_edges=4*n, balanced=False)
    print(p.m)
    p.grad_check()
    p.distributed_check()

    # p = LinearRegression(n, m, dim, noise_variance=noise_variance, n_edges=4*n)
    p.plot_graph()

    print('w_min = ' + str(p.w_min))
    print('f(w_min) = ' + str(p.f(p.w_min)))
    print('f_0(w_min) = ' + str(p.f(p.w_min, 0)))
    print('|| g(w_min) || = ' + str(np.linalg.norm(p.grad(p.w_min))))
    print('|| g_0(w_min) || = ' + str(np.linalg.norm(p.grad(p.w_min, 0))))

    plt.show()


