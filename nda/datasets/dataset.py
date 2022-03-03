#!/usr/bin/env python
# coding=utf-8
import os
import numpy as np

from nda import log


class Dataset:
    def __init__(self, data_urls=None, name=None, normalize=False, root='~/data'):
        self.name = self.__class__.__name__ if name is None else name
        self.data_root = os.path.expanduser(root)
        self.data_dir = os.path.join(self.data_root, self.name)
        self.cache_path = os.path.join(self.data_dir, '%s.npz' % self.name)
        self.data_urls = data_urls
        self.normalize = normalize

    def load(self):
        if not os.path.exists(self.cache_path):
            log.info('Downloading %s dataset' % self.name)
            os.system('mkdir -p %s' % self.data_dir)
            self.download()
            self.load_raw()
            np.savez_compressed(
                self.cache_path,
                X_train=self.X_train, Y_train=self.Y_train,
                X_test=self.X_test, Y_test=self.Y_test
            )

        else:
            log.info('Loading %s dataset from cached file' % self.name)
            data = np.load(self.cache_path, allow_pickle=True)
            self.X_train = data['X_train']
            self.Y_train = data['Y_train']
            self.X_test = data['X_test']
            self.Y_test = data['Y_test']

        if self.normalize:
            self.normalize_data()
        
        return self.X_train, self.Y_train, self.X_test, self.Y_test

    def load_raw(self):
        raise NotImplementedError

    def normalize_data(self):
        mean = self.X_train.mean()
        std = self.X_train.std()
        self.X_train = (self.X_train - mean) / std
        self.X_test = (self.X_test - mean) / std

    def download(self):
        os.system("mkdir -p %s" % self.data_dir)
        for url in self.data_urls:
            os.system("wget -nc -P %s %s" % (self.data_dir, url))
