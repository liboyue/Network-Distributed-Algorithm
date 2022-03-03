#!/usr/bin/env python
# coding=utf-8
import os
import numpy as np
from nda.datasets import Dataset
from nda import log

class Gisette(Dataset):
    def __init__(self, **kwargs):
        data_urls = [
            'https://archive.ics.uci.edu/ml/machine-learning-databases/gisette/GISETTE/gisette_train.data',
            'https://archive.ics.uci.edu/ml/machine-learning-databases/gisette/GISETTE/gisette_train.labels',
            'https://archive.ics.uci.edu/ml/machine-learning-databases/gisette/GISETTE/gisette_valid.data',
            'https://archive.ics.uci.edu/ml/machine-learning-databases/gisette/gisette_valid.labels'
        ]

        super().__init__(data_urls=data_urls, **kwargs)

    def load_raw(self):
        def _load_raw(split):
            data_path = os.path.join(self.data_dir, 'gisette_%s.data' % split)
            with open(data_path) as f:
                data = f.readlines()
            data = np.array([[int(x) for x in line.split()] for line in data], dtype=float)

            label_path = os.path.join(self.data_dir, 'gisette_%s.labels' % split)
            with open(label_path) as f:
                labels = np.array([int(x) for x in f.read().split()], dtype=float)

            labels[labels < 0] = 0
            return data, labels

        self.X_train, self.Y_train = _load_raw('train')
        self.X_test, self.Y_test = _load_raw('valid')

    def normalize_data(self):
        self.X_train /= 999
        self.X_test /= 999
        super().normalize_data()
