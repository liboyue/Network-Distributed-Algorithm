#!/usr/bin/env python
# coding=utf-8
import numpy as np
from .problem import Problem

class LinearRegression(Problem):
    '''f(w) = 1/n \sum f_i(w) = 1/n \sum 1/2m || Y_i - X_i w ||^2'''
    
    def __init__(self, n_agent, m, dim, noise_variance=0.1, kappa=10, n_edges=None, prob=None):

        super().__init__(n_agent, m, dim, n_edges=n_edges, prob=prob)

        self.noise_variance = noise_variance
        self.kappa = kappa

        def _generate_x(kappa):
            '''Helper function to generate data''' 
            power = - np.log(kappa) / np.log(self.dim) / 2

            X = np.random.randn(self.n_agent, self.m, self.dim)
            S = np.power(np.arange(self.dim)+1, power)
            X_ = X.reshape(-1, self.dim)
            X_ *= S
            X = X_.reshape(self.n_agent, self.m, self.dim)
            max_norm = 0
            min_norm = np.linalg.norm(X[0].T.dot(X[0]) / self.m, 2)
            for i in range(self.n_agent):
                max_norm = max(
                        np.linalg.norm(X[i].T.dot(X[i]) / self.m, 2),
                        max_norm
                        )
                min_norm = min(
                        np.linalg.norm(X[i].T.dot(X[i]) / self.m, -2),
                        min_norm
                        )

            X /= max_norm
            return X, 1, dim**(2*power) / max_norm

        # Generate the problem
        self.X, self.L, self.sigma = _generate_x(kappa)

        self.w_0 = np.random.rand(self.dim)
        self.Y_0 = self.X.dot(self.w_0)
        self.Y = self.Y_0 + self.noise_variance * np.random.randn(n_agent, m)

        X_tmp = self.X.reshape(-1, self.dim)
        Y_tmp = self.Y.reshape(-1)
        self.w_min = np.linalg.solve(X_tmp.T.dot(X_tmp), X_tmp.T.dot(Y_tmp))
        self.x_min = self.w_min

        self.H_list = [self.X[i].T.dot(self.X[i]) / self.m for i in range(self.n_agent)]
        self.H = X_tmp.T.dot(X_tmp) / self.m / self.n_agent
        self.X_T_Y = X_tmp.T.dot(Y_tmp) / self.m / self.n_agent
        self.X_T_Y_list = [self.X[i, :, :].T.dot(self.Y[i, :]) / self.m for i in range(self.n_agent)]


    def grad(self, w, i=None, j=None):
        '''Gradient at w. If i is None, returns the full gradient; if i is not None but j is, returns the gradient in the i-th machine; otherwise,return the gradient of j-th sample in i-th machine. '''

        if (i == None): # Return the full gradient
            return self.H.dot(w) - self.X_T_Y
        elif j == None: # Return the gradient in machine i
                return self.H_list[i].dot(w) - self.X_T_Y_list[i]
        else: # Return the gradient of sample j in machine i
                return (self.X[i, j, :].T.dot(w) - self.Y[i, j]) * self.X[i, j, :]

    def hessian(self, w=None, i=None, j=None):
        '''Hessian matrix at w. If i is None, returns the full Hessian matrix; if i is not None but j is, returns the hessian matrix in the i-th machine; otherwise,return the hessian matrix of j-th sample in i-th machine.'''

        if i == None: # Return the full hessian matrix
            return self.H
        elif j == None: # Return the hessian matrix in machine i
            return self.H_list[i]
        else: # Return the hessian matrix of sample j in machine i
            return self.X[np.newaxis, i, j, :].T.dot(self.X[np.newaxis, i, j, :])


    def f(self, w, i=None, j=None):
        '''Function value at w. If i is None, returns f(x); if i is not None but j is, returns the function value in the i-th machine; otherwise,return the function value of j-th sample in i-th machine.'''

        if i == None: # Return the function value
            Z = np.sqrt(2 * self.n_agent * self.m)
            return np.sum((self.Y/Z - (self.X/Z).dot(w))**2)
        elif j == None: # Return the function value in machine i
            return np.sum((self.Y[i, :] - self.X[i, :, :].dot(w))**2) / 2 / self.m
        else: # Return the function value in machine i
            return (self.Y[i, j] - self.X[i, j, :].dot(w))**2 / 2
