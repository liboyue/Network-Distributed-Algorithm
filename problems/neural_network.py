#!/usr/bin/env python
# coding=utf-8
import numpy as np
import os

from .problem import Problem


img_dim = 785
n_class = 10

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
    
    def __init__(self, n_agent, n_hidden=64, data=None, **kwargs):

        if data is not None:
            self.X_train, self.Y_train, self.X_test, self.Y_test = data
        else:
            # Load data
            self.X_train, self.Y_train, self.X_test, self.Y_test = self.load_data()

            # self.X_train, self.X_train_buf = self.convert_to_shared(self.X_train)
            # self.X_test, self.X_test_buf = self.convert_to_shared(self.X_test)


        self.n_hidden = n_hidden # Number of neurons in hidden layer

        self.m_mean = int(self.X_train.shape[0] / n_agent)
        super().__init__(n_agent, self.m_mean, (n_hidden+1) * (img_dim + n_class), **kwargs)

        self.n_class = n_class

        # Split training data into n agents
        self.X = self.split_data(self.m, self.X_train)
        self.Y = self.split_data(self.m, self.Y_train)

        self.Y_test_labels = self.Y_test.argmax(axis=1)
        

        # Internal buffer
        # self._dW = np.zeros(self.dim)                               # dim = (n_hidden+1) * (img_dim + n_class)
        self._dw = np.zeros(self.dim)                               # dim = (n_hidden+1) * (img_dim + n_class)
        self._dw1, self._dw2 = self.unpack_w(self._dw) # Reference to the internal buffer
        # self._A1 = np.zeros((self.n_hidden+1, self.m_mean*self.n_agent)) # A_1 = W1.dot(X), dim = (self.n_hidden+1, img_dim).dot(img_dim, m*n)
        # self._A2 = np.zeros((self.n_class, self.m_mean*self.n_agent))    # A2 = W2.dot(A1),, dim = (n_class, self.n_hidden+1).dot(self.n_hidden+1, m)


    def load_data(self):
        if os.path.exists('mnist.npz'):
            data = np.load('mnist.npz', allow_pickle=True)
            X = data['X']
            Y = data['Y']
        else:
            from sklearn.datasets import fetch_openml
            X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
            # One-hot encode labels
            Y = np.eye(n_class)[y.astype('int')].T
            np.savez_compressed('mnist', X=X, Y=Y)

        # Subtract mean and normalize data
        X /= 255
        X -= X.mean()
        # X /= np.abs(X).max()
        X = X.T

        # Append '1' to every sample, so that the dimension increased by 1
        X = np.append(X, np.ones((1, X.shape[1])), 0)

        # Split to train & test
        n_train = 60000
        X_train, X_test = X[:, :n_train], X[:, n_train:]
        Y_train, Y_test = Y[:, :n_train], Y[:, n_train:]

        # Shuffle
        idx = np.random.permutation(n_train)
        X_train, Y_train = X_train[:, idx], Y_train[:, idx]
        return X_train.T.copy(), Y_train.T.copy(), X_test.T.copy(), Y_test.T.copy() # Return new copies to make sure they are C contiguous


    def unpack_w(self, W):
        # This function returns references
        return W[: img_dim * (self.n_hidden+1)].reshape(img_dim, self.n_hidden+1), \
                W[img_dim * (self.n_hidden+1) :].reshape(self.n_hidden+1, n_class)


    def pack_w(self, W_1, W_2):
        # This function returns a new array
        return np.append(W_1.reshape(-1), W_2.reshape(-1))


    def grad(self, w, i=None, j=None):
        '''Gradient at w. If i is None, returns the full gradient; if i is not None but j is, returns the gradient in the i-th machine; otherwise,return the gradient of j-th sample in i-th machine. '''

        if i is None: # Return the full gradient
            grad, _ = self.forward_backward(self.X_train, self.Y_train, w)
        elif j is None: # Return the gradient at machine i
            grad, _ = self.forward_backward(self.X[i], self.Y[i], w)
        else: # Return the gradient of sample j at machine i
            if type(j) is np.ndarray:
                grad, _ = self.forward_backward(self.X[i][j], self.Y[i][j], w)
            else:
                grad, _ = self.forward_backward(self.X[i][[j]], self.Y[i][[j]], w)

        return grad


    def grad_full(self, w, i=None):
        '''Full gradient at w. If i is None, returns the full gradient; if i is not None, returns the gradient for the i-th sample in the whole dataset.'''

        if i is None: # Return the full gradient
            grad, _ = self.forward_backward(self.X_train, self.Y_train, w)
        else: # Return the gradient of sample i
            if type(i) is np.ndarray:
                grad, _ = self.forward_backward(self.X_train[i], self.Y_train[i], w)
            else:
                grad, _ = self.forward_backward(self.X_train[[i]], self.Y_train[[i]], w)

        return grad


    def f(self, w, i=None, j=None):
        '''Function value at w. If i is None, returns f(x); if i is not None but j is, returns the function value in the i-th machine; otherwise,return the function value of j-th sample in i-th machine.'''

        if i is None: # Return the function value
            return self.forward(self.X_train, self.Y_train, w)[0]
        elif j is None: # Return the function value at machine i
            return self.forward(self.X[i], self.Y[i], w)[0]
        else: # Return the function value at machine i
            if type(j) is np.ndarray:
                return self.forward(self.X[i, j], self.Y[i, j], w)[0]
            else:
                return self.forward(self.X[i, [j]], self.Y[i, [j]], w)[0]


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
        # dw2 /= X.shape[0]
        # dw2 = A1.T.dot(dZ2) / X.shape[0]
        dA1 = dZ2.dot(w2.T)
        dZ1 = dA1 * A1 * (1 - A1)
        np.dot(X.T, dZ1, out=self._dw1)
        # dw1 = X.T.dot(dZ1) / X.shape[0]
        self._dw /= X.shape[0]
        return self._dw, loss

    def accuracy(self, w):
        loss, _, A2 = self.forward(self.X_test, self.Y_test, w)
        pred = A2.argmax(axis=1)

        return sum(pred == self.Y_test_labels) / len(pred), loss




if __name__ == '__main__':

    p = NN()


