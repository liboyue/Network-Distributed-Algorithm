#!/usr/bin/env python
# coding=utf-8
import numpy as np
from scipy import optimize as opt
from nda import log, datasets
from nda.problems import Problem


def logit_1d(X, w):
    return 1 / (1 + np.exp(-X.dot(w)))


def logit_2d(X, w):
    tmp = np.einsum('ijk,ki->ij', X, w)
    return 1 / (1 + np.exp(-tmp))


def generate_data(m_total, dim, noise_ratio, m_test=None):
    if m_test is None:
        m_test = int(m_total / 10)

    # Generate data
    X = np.random.randn(m_total + m_test, dim)
    norm = np.sqrt(2 * np.linalg.norm(X.T.dot(X), 2) / (m_total + m_test))
    X /= 2 * norm

    # Generate labels
    w_0 = np.random.rand(dim)
    Y = logit_1d(X, w_0)
    Y[Y > 0.5] = 1
    Y[Y <= 0.5] = 0

    X_train, X_test = X[:m_total], X[:m_total]
    Y_train, Y_test = Y[:m_total], Y[:m_total]

    noise = np.random.binomial(1, noise_ratio, m_total)
    Y_train = (noise - Y_train) * noise + Y_train * (1 - noise)

    return X_train, Y_train, X_test, Y_test, w_0


class LogisticRegression(Problem):
    '''f(w) =  - 1 / N * (\sum y_i log(1/(1 + exp(w^T x_i))) + (1 - y_i) log (1 - 1/(1 + exp(w^T x_i)))) + \frac{\lambda}{2} \| w \|^2 + \alpha \sum w_i^2 / (1 + w_i^2)'''
    def grad_g(self, w):
        if self.alpha == 0:
            return 0
        return 2 * self.alpha * w / ((1 + w**2)**2)

    def g(self, w):
        if self.alpha == 0:
            return 0
        return (1 - 1 / (1 + w ** 2)).sum() * self.alpha

    def __init__(self, n_agent, m=None, dim=None, dataset='random', kappa=None, noise_ratio=None, LAMBDA=0, alpha=0, **kwargs):

        if dataset == 'random':
            self.X_train, self.Y_train, self.X_test, self.Y_test, self.w_0 = generate_data(n_agent * m, dim, noise_ratio)
        else:
            if dataset == 'gisette':
                self.X_train, self.Y_train, self.X_test, self.Y_test = datasets.Gisette(normalize=True).load()
            elif dataset == 'a9a':
                self.X_train, self.Y_train, self.X_test, self.Y_test = datasets.LibSVM(name='a9a', normalize=True).load()

            m = int(self.X_train.shape[0] / n_agent)
            self.X_train = self.X_train[:m * n_agent]
            self.Y_train = self.Y_train[:m * n_agent]
            dim = self.X_train.shape[1]

        super().__init__(n_agent, m, dim, **kwargs)

        self.noise_ratio = noise_ratio
        self.X = self.split_data(self.X_train)
        self.Y = self.split_data(self.Y_train)

        self.kappa = kappa
        self.alpha = alpha
        self.LAMBDA = LAMBDA

        if alpha == 0:
            if kappa == 1:
                self.LAMBDA = 100
            elif kappa is not None:
                self.LAMBDA = 1 / (self.kappa - 1)
            self.L = 1 + self.LAMBDA
            self.sigma = self.LAMBDA if self.LAMBDA != 0 else None 
        else:
            self.L = 1 + self.LAMBDA + 6 * self.alpha
            self.sigma = self.LAMBDA + 2 * self.alpha

        if self.kappa is not None:
            self.x_min = self.w_min = opt.minimize(
                self.f,
                np.random.rand(self.dim),
                jac=self.grad,
                method='BFGS',
                options={'gtol': 1e-8}
            ).x
            self.f_min = self.f(self.w_min)

        '''
    def _generate_data(self, find_minimum=True):
        # Generate data
        X = np.random.randn(self.m_total, self.dim)
        norm = np.sqrt(np.linalg.norm(X.T.dot(X), 2) / self.m_total)
        X /= norm
        self.X_train = X

        # Generate labels
        w_0 = np.random.rand(self.dim)
        Y_0_total = logit_1d(self.X_train, w_0)
        Y_0_total[Y_0_total > 0.5] = 1
        Y_0_total[Y_0_total <= 0.5] = 0

        noise = np.random.binomial(1, self.noise_ratio, self.m_total)
        self.Y_train = np.multiply(noise - Y_0_total, noise) + np.multiply(Y_0_total, 1 - noise)
        '''

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
                return self.X_train.T.dot(logit_1d(self.X_train, w) - self.Y_train) / self.m_total + w * self.LAMBDA
            elif i is not None and j is None:
                return self.X[i].T.dot(logit_1d(self.X[i], w) - self.Y[i]) / self.m + w * self.LAMBDA
            elif i is None and j is not None:  # Return the full gradient
                return self.X_train[j].T.dot(logit_1d(self.X_train[j], w) - self.Y_train[j]) / len(j) + w * self.LAMBDA
            else:  # Return the gradient of sample j at machine i
                return (logit_1d(self.X[i][j], w) - self.Y[i][j]).dot(self.X[i][j]) / len(j) + w * self.LAMBDA

        elif w.ndim == 2:
            if i is None and j is None:  # Return the distributed gradient
                tmp = logit_2d(self.X, w) - self.Y
                return np.einsum('ikj,ik->ji', self.X, tmp) / self.m + w * self.LAMBDA
            elif i is None and j is not None:  # Return the stochastic gradient
                res = []
                for i in range(self.n_agent):
                    if type(j[i]) is int:
                        samples = [j[i]]
                    else:
                        samples = j[i]
                    res.append(self.X[i][samples].T.dot(logit_1d(self.X[i][samples], w[:, i]) - self.Y[i][samples]) / len(samples) + w[:, i] * self.LAMBDA)
                return np.array(res).T
            else:
                log.fatal('For distributed gradients j must be None')
        else:
            log.fatal('Parameter dimension should only be 1 or 2')

    def h(self, w, i=None, j=None):
        '''Function value at w. If i is None, returns f(x); if i is not None but j is, returns the function value in the i-th machine; otherwise,return the function value of j-th sample in i-th machine.'''

        if i is None:  # Return the function value
            tmp = self.X_train.dot(w)
            return -np.sum(
                (self.Y_train - 1) * tmp - np.log(1 + np.exp(-tmp))
            ) / self.m_total + np.sum(w**2) * self.LAMBDA / 2

        elif j is None:  # Return the function value in machine i
            tmp = self.X[i].dot(w)
            return -np.sum((self.Y[i] - 1) * tmp - np.log(1 + np.exp(-tmp))) / self.m + np.sum(w**2) * self.LAMBDA / 2
        else:  # Return the gradient of sample j in machine i
            tmp = self.X[i][j].dot(w)
            return -((self.Y[i][j] - 1) * tmp - np.log(1 + np.exp(-tmp))) + np.sum(w**2) * self.LAMBDA / 2

    def accuracy(self, w, split='train'):
        if len(w.shape) > 1:
            w = w.mean(axis=1)
        if split == 'train':
            X = self.X_train
            Y = self.Y_train
        elif split == 'test':
            X = self.X_test
            Y = self.Y_test
        else:
            log.fatal('Data split %s is not supported' % split)

        Y_hat = X.dot(w)
        Y_hat[Y_hat > 0] = 1
        Y_hat[Y_hat <= 0] = 0
        return np.mean(Y_hat == Y)


if __name__ == '__main__':

    n = 10
    m = 1000
    dim = 10
    noise_ratio = 0.01

    p = LogisticRegression(n, m, dim, noise_ratio=noise_ratio, balanced=False)
    p.grad_check()
    p.distributed_check()

    p = LogisticRegression(n, m, dim, noise_ratio=noise_ratio, n_edges=4 * n)
    p.grad_check()
    p.distributed_check()
    # p.plot_graph()

    print('w_min = ' + str(p.w_min))
    print('f(w_min) = ' + str(p.f(p.w_min)))
    print('f_0(w_min) = ' + str(p.f(p.w_min, 0)))
    print('|| g(w_min) || = ' + str(np.linalg.norm(p.grad(p.w_min))))
    print('|| g_0(w_min) || = ' + str(np.linalg.norm(p.grad(p.w_min, 0))))
