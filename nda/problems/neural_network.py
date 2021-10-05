#!/usr/bin/env python
# coding=utf-8
import numpy as np
import os

from nda.problems import Problem
from nda.datasets import MNIST
from nda import log


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    tmp = np.exp(x)
    return tmp / tmp.sum(axis=1, keepdims=True)


def softmax_loss(Y, score):
    return - np.sum(np.log(score[Y != 0])) / Y.shape[0]
    # return - np.sum(Y * np.log(score)) / Y.shape[0]


class NN(Problem):
    '''f(w) = 1/n \sum l_i(w), where l_i(w) is the logistic loss'''

    def __init__(self, n_agent, n_hidden=64, shuffle=True, **kwargs):

        # Load data
        self.X_train, self.Y_train, self.X_test, self.Y_test = MNIST().load()

        self.X_train = np.append(self.X_train, np.ones((self.X_train.shape[0], 1)), axis=1)
        self.X_test = np.append(self.X_test, np.ones((self.X_test.shape[0], 1)), axis=1)

        self.n_hidden = n_hidden  # Number of neurons in hidden layer
        self.m = int(self.X_train.shape[0] / n_agent)
        self.n_class = self.Y_train.shape[1]
        self.img_dim = self.X_train.shape[1]

        log.info(self.img_dim)
        log.info(self.n_class)
        # Shuffle
        if shuffle is True:
            idx = np.random.permutation(len(self.X_train))
            self.X_train, self.Y_train = self.X_train[idx], self.Y_train[idx]

        super().__init__(n_agent, self.m, (n_hidden + 1) * (self.img_dim + self.n_class), **kwargs)

        # Split training data into n agents
        self.X = self.split_data(self.X_train)
        self.Y = self.split_data(self.Y_train)
        self.Y_train_labels = self.Y_train.argmax(axis=1)
        self.Y_test_labels = self.Y_test.argmax(axis=1)

        # Internal buffers
        self._dw = np.zeros(self.dim)
        self._dw1, self._dw2 = self.unpack_w(self._dw)  # Reference to the internal buffer

    def unpack_w(self, W):
        # This function returns references
        return W[: self.img_dim * (self.n_hidden + 1)].reshape(self.img_dim, self.n_hidden + 1), \
            W[self.img_dim * (self.n_hidden + 1):].reshape(self.n_hidden + 1, self.n_class)

    def pack_w(self, W_1, W_2):
        # This function returns a new array
        return np.append(W_1.reshape(-1), W_2.reshape(-1))

    def grad_h(self, w, i=None, j=None):
        '''Gradient at w. If i is None, returns the full gradient; if i is not None but j is, returns the gradient in the i-th machine; otherwise,return the gradient of j-th sample in i-th machine. '''

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
                return np.array([self.forward_backward(self.X[i], self.Y[i], w[:, i])[0].copy() for i in range(self.n_agent)]).T
            elif i is None and j is not None:  # Return the stochastic gradient
                return np.array([self.forward_backward(self.X[i][j[i]], self.Y[i][j[i]], w[:, i])[0].copy() for i in range(self.n_agent)]).T
            else:
                log.fatal('For distributed gradients j must be None')

        else:
            log.fatal('Parameter dimension should only be 1 or 2')

    def h(self, w, i=None, j=None):
        '''Function value at w. If i is None, returns f(x); if i is not None but j is, returns the function value in the i-th machine; otherwise,return the function value of j-th sample in i-th machine.'''

        if i is None and j is None:  # Return the function value
            return self.forward(self.X_train, self.Y_train, w)[0]
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
        np.dot(A1.T, dZ2, out=self._dw2)
        dA1 = dZ2.dot(w2.T)
        dZ1 = dA1 * A1 * (1 - A1)
        np.dot(X.T, dZ1, out=self._dw1)
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
