#!/usr/bin/env python
# coding=utf-8
import numpy as np
import os

from nda import log


class Dataset:
    def __init__(self, data_urls=None, name=None, normalize=False, root='~/data'):
        self.name = self.__class__.__name__ if name is None else name
        self.data_root = os.path.expanduser(root)
        self.data_dir = os.path.join(self.data_root, self.name)
        if normalize:
            self.cache_path = os.path.join(self.data_dir, '%s_normalized.npz' % self.name)
        else:
            self.cache_path = os.path.join(self.data_dir, '%s.npz' % self.name)
        self.data_urls = data_urls
        self.normalize = normalize

    def load(self):
        if not os.path.exists(self.cache_path):
            log.info('Downloading %s dataset' % self.name)
            os.system('mkdir -p %s' % self.data_dir)
            self.download()
            data = self.load_raw()
            np.savez_compressed(
                self.cache_path,
                X_train=data[0], Y_train=data[1],
                X_test=data[2], Y_test=data[3]
            )
            return data

        else:
            log.info('Loading %s dataset from cached file' % self.name)
            data = np.load(self.cache_path, allow_pickle=True)
            return [data[key] for key in ['X_train', 'Y_train', 'X_test', 'Y_test']]

    def load_raw(self):
        raise NotImplementedError

    def download(self):
        os.system("mkdir -p %s" % self.data_dir)
        for url in self.data_urls:
            os.system("wget -nc -P %s %s" % (self.data_dir, url))
