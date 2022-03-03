#!/usr/bin/env python
# coding=utf-8
import os
import numpy as np

from sklearn.datasets import load_svmlight_file
from nda.datasets import Dataset

from nda import log

class LibSVM(Dataset):
    def __init__(self, name='a9a', **kwargs):
        data_urls = [
            'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/%s' % name,
            'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/%s.t' %name
        ]
        super().__init__(root='~/data/LibSVM', name=name, data_urls=data_urls, **kwargs)

    def load_raw(self):
        def _load_raw(name, n_features=None):
            data_path = os.path.join(self.data_dir, name)
            X, Y = load_svmlight_file(data_path, n_features=n_features)
            X = X.toarray()
            Y[Y < 0] = 0
            return X, Y

        if self.name == 'a9a':
            n_features = 123
        elif self.name == 'a5a':
            n_features = 123
        else:
            n_features = None

        self.X_train, self.Y_train = _load_raw(self.name, n_features=n_features)
        self.X_test, self.Y_test = _load_raw('%s.t' % self.name, n_features=n_features)
