#!/usr/bin/env python
# coding=utf-8
import numpy as np

try:
    import cupy as xp
except ImportError:
    xp = np

from nda.problems import Problem
from nda.datasets import MNIST
from nda import log


def sigmoid(x):
    return 1 / (1 + xp.exp(-x))


def softmax(x):
    tmp = xp.exp(x)
    return tmp / tmp.sum(axis=1, keepdims=True)


def softmax_loss(Y, score):
    return - xp.sum(xp.log(score[Y != 0])) / Y.shape[0]
    # return - xp.sum(Y * xp.log(score)) / Y.shape[0]


class NN(Problem):
    '''f(w) = 1/n \sum l_i(w), where l_i(w) is the logistic loss'''

    def __init__(self, n_hidden=64, dataset='mnist', **kwargs):

        super().__init__(dataset=dataset, **kwargs)

        self.n_hidden = n_hidden  # Number of neurons in hidden layer
        self.n_class = self.Y_train.shape[1]
        self.img_dim = self.X_train.shape[1]
        self.dim = (n_hidden + 1) * (self.img_dim + self.n_class)

        self.Y_train_labels = self.Y_train.argmax(axis=1)
        self.Y_test_labels = self.Y_test.argmax(axis=1)

        # Internal buffers
        self._dw = np.zeros(self.dim)
        self._dw1, self._dw2 = self.unpack_w(self._dw)  # Reference to the internal buffer

        log.info('Initialization done')

    def cuda(self):
        super().cuda()
        self._dw1, self._dw2 = self.unpack_w(self._dw)  # Renew the reference

    def unpack_w(self, W):
        # This function returns references
        return W[: self.img_dim * (self.n_hidden + 1)].reshape(self.img_dim, self.n_hidden + 1), \
            W[self.img_dim * (self.n_hidden + 1):].reshape(self.n_hidden + 1, self.n_class)

    def pack_w(self, W_1, W_2):
        # This function returns a new array
        return xp.append(W_1.reshape(-1), W_2.reshape(-1))

    def grad_h(self, w, i=None, j=None):
        '''Gradient at w. If i is None, returns the full gradient; if i is not None but j is, returns the gradient in the i-th machine; otherwise,return the gradient of j-th sample in i-th machine. '''

        if not(self._dw1.base is self._dw and self._dw2.base is self._dw):
            self._dw1, self._dw2 = self.unpack_w(self._dw)

        if w.ndim == 1:
            if type(j) is int:
                j = [j]

            if i is None and j is None:  # Return the full gradient
                return self.forward_backward(self.X_train, self.Y_train, w)[0]
            elif i is not None and j is None:  # Return the local gradient
                return self.forward_backward(self.X[i], self.Y[i], w)[0]
            elif i is None and j is not None:  # Return the stochastic gradient
                return self.forward_backward(self.X_train[j], self.Y_train[j], w)[0]
            else:  # Return the stochastic gradient
                return self.forward_backward(self.X[i][j], self.Y[i][j], w)[0]

        elif w.ndim == 2:
            if i is None and j is None:  # Return the distributed gradient
                return xp.array([self.forward_backward(self.X[i], self.Y[i], w[:, i])[0].copy() for i in range(self.n_agent)]).T
            elif i is None and j is not None:  # Return the stochastic gradient
                return xp.array([self.forward_backward(self.X[i][j[i]], self.Y[i][j[i]], w[:, i])[0].copy() for i in range(self.n_agent)]).T
            else:
                log.fatal('For distributed gradients j must be None')

        else:
            log.fatal('Parameter dimension should only be 1 or 2')

    def h(self, w, i=None, j=None, split='train'):
        '''Function value at w. If i is None, returns f(x); if i is not None but j is, returns the function value in the i-th machine; otherwise,return the function value of j-th sample in i-th machine.'''

        if split == 'train':
            X = self.X_train
            Y = self.Y_train
        elif split == 'test':
            if w.ndim > 1 or i is not None or j is not None:
                log.fatal("Function value on test set only applies to one parameter vector")
            X = self.X_test
            Y = self.Y_test

        if i is None and j is None:  # Return the function value
            return self.forward(X, Y, w)[0]
        elif i is not None and j is None:  # Return the function value at machine i
            return self.forward(self.X[i], self.Y[i], w)[0]
        else:  # Return the function value at machine i
            if type(j) is int:
                j = [j]
            return self.forward(self.X[i][j], self.Y[i][j], w)[0]

    def forward(self, X, Y, w):
        w1, w2 = self.unpack_w(w)
        A1 = sigmoid(X.dot(w1))
        A1[:, -1] = 1
        A2 = softmax(A1.dot(w2))

        return softmax_loss(Y, A2), A1, A2

    def forward_backward(self, X, Y, w):
        w1, w2 = self.unpack_w(w)
        loss, A1, A2 = self.forward(X, Y, w)

        dZ2 = A2 - Y
        xp.dot(A1.T, dZ2, out=self._dw2)
        dA1 = dZ2.dot(w2.T)
        dZ1 = dA1 * A1 * (1 - A1)
        xp.dot(X.T, dZ1, out=self._dw1)
        self._dw /= X.shape[0]

        return self._dw, loss

    def accuracy(self, w, split='test'):
        if w.ndim > 1:
            w = w.mean(axis=1)
        if split == 'train':
            X = self.X_train
            Y = self.Y_train
            labels = self.Y_train_labels
        elif split == 'test':
            X = self.X_test
            Y = self.Y_test
            labels = self.Y_test_labels
        else:
            log.fatal('Data split %s is not supported' % split)

        loss, _, A2 = self.forward(X, Y, w)
        pred = A2.argmax(axis=1)

        return sum(pred == labels) / len(pred), loss


if __name__ == '__main__':

    p = NN()
