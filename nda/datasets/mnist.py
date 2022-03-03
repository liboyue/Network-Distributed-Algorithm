#!/usr/bin/env python
# coding=utf-8
import os
import numpy as np
from sklearn.datasets import fetch_openml
from nda.datasets import Dataset


class MNIST(Dataset):

    def download(self):
        pass

    def load_raw(self):
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
        # One-hot encode labels
        y = y.astype('int')
        Y = np.eye(y.max() + 1)[y]

        # Split to train & test
        n_train = 60000

        self.X_train, self.X_test = X[:n_train], X[n_train:]
        self.Y_train, self.Y_test = Y[:n_train], Y[n_train:]

    def normalize_data(self):
        self.X_train /= 255
        self.X_test /= 255
        super().normalize_data()
