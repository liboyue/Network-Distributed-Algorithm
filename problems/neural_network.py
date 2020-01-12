#!/usr/bin/env python
# coding=utf-8
import numpy as np
from .problem import Problem

import os

img_dim = 785
n_class = 10

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    tmp = np.exp(x)
    return tmp / tmp.sum(axis=0)


def softmax_loss(Y, score):
    return - np.sum(Y * np.log(score)) / Y.shape[1]




class NN(Problem):
    '''f(w) = 1/n \sum l_i(w), where l_i(w) is the logistic loss'''
    
    def __init__(self, n, n_hidden=64, n_edges=None, prob=None):

        # Load data
        X_train, Y_train, self.X_test, self.Y_test = self.load_data()

        m = int(X_train.shape[1] / n)
        super().__init__(n, m, (n_hidden+1) * (img_dim + n_class), n_edges=n_edges, prob=prob)

        self.n_hidden = n_hidden # Number of neurons in hidden layer


        # Split training data into n agents
        self.m = int(X_train.shape[1] / self.n_agent)
        self.X = np.split(X_train, self.n_agent, axis=1)
        self.Y = np.split(Y_train, self.n_agent, axis=1)

        # Keep the whole data for easier gradient and function value computation
        self.X_train = X_train
        self.Y_train = Y_train


    def load_data(self):
        if os.path.exists('mnist.npz'):
            data = np.load('mnist.npz', allow_pickle=True)
            X = data['X']
            y = data['y']
        else:
            from sklearn.datasets import fetch_openml
            X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
            np.savez_compressed('mnist', X=X, y=y)

        # Subtract mean and normalize data
        X -= X.mean()
        X /= np.abs(X).max()
        X = X.T

        # Append '1' to every sample, so that the dimension increased by 1
        X = np.append(X, np.ones((1, X.shape[1])), 0)

        # One-hot encode labels
        Y = np.eye(n_class)[y.astype('int')].T

        # Split to train & test
        n_train = 60000
        X_train, X_test = X[:, :n_train], X[:, n_train:]
        Y_train, Y_test = Y[:, :n_train], Y[:, n_train:]

        # For reproductivity, we don't shuffle here
        # Shuffle
        # idx = np.random.permutation(n_train)
        # X_train, Y_train = X_train[:, idx], Y_train[:, idx]
        return X_train, Y_train, X_test, Y_test


    def unpack_w(self, W):
        # This function returns references
        return W[: img_dim * (self.n_hidden+1)].reshape(self.n_hidden+1, img_dim), \
                W[img_dim * (self.n_hidden+1) :].reshape(n_class, self.n_hidden+1)


    def pack_w(self, W_1, W_2):
        # This function returns a new array
        return np.append(W_1.reshape(-1), W_2.reshape(-1))


    def grad(self, w, i=None, j=None):
        '''Gradient at w. If i is None, returns the full gradient; if i is not None but j is, returns the gradient in the i-th machine; otherwise,return the gradient of j-th sample in i-th machine. '''

        if (i == None): # Return the full gradient
            grad, _ = self.forward_backward(self.X_train, self.Y_train, w)
        elif j == None: # Return the gradient in machine i
            grad, _ = self.forward_backward(self.X[i], self.Y[i], w)
        else: # Return the gradient of sample j in machine i
            grad, _ = self.forward_backward(self.X[i][:, [j]], self.Y[i][:, [j]], w)

        return grad


    def f(self, w, i=None, j=None):
        '''Function value at w. If i is None, returns f(x); if i is not None but j is, returns the function value in the i-th machine; otherwise,return the function value of j-th sample in i-th machine.'''

        if i == None: # Return the function value
            return self.forward(self.X_train, self.Y_train, w)
        elif j == None: # Return the function value in machine i
            return self.forward(self.X[i], self.Y[i], w)
        else: # Return the function value in machine i
            return self.forward(self.X[i][:, [j]], self.Y[i][:, [j]], w)


    def forward(self, X, Y, W):
        W1, W2 = self.unpack_w(W)
        A1 = sigmoid(W1.dot(X))
        A1[:, -1] = 1
        A2 = softmax(W2.dot(A1))

        return softmax_loss(Y, A2)


    def forward_backward(self, X, Y, W):
        W1, W2 = self.unpack_w(W)
        A1 = sigmoid(W1.dot(X))
        A1[:, -1] = 1
        A2 = softmax(W2.dot(A1))
        loss = softmax_loss(Y, A2)

        dZ2 = A2 - Y
        dW2 = dZ2.dot(A1.T) / X.shape[1]
        dA1 = W2.T.dot(dZ2)
        dZ1 = dA1 * A1 * (1 - A1)
        dW1 = dZ1.dot(X.T) / X.shape[1]
        dW = self.pack_w(dW1, dW2)
        return dW, loss

    def accuracy(self, W):
        W1, W2 = self.unpack_w(W)
        A1 = sigmoid(W1.dot(self.X_test))
        A1[:, -1] = 1
        A2 = softmax(W2.dot(A1))

        pred = A2.argmax(axis=0)
        labels = self.Y_test.argmax(axis=0)

        return sum(pred == labels) / len(pred)



if __name__ == '__main__':

    p = NN()


