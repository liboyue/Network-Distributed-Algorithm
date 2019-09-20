# Network-Distributed Algorithm Experiments

This repository contains all code needed to reproduce experiments in "Communication-Efficient Distributed Optimization in Networks with Gradient Tracking" [[PDF](https://arxiv.org/pdf/1909.05844.pdf)].

Due to the random data generation procedure,
resulting graphs may be different from those appeared in the paper,
but conclusions remain the same.

If you find this code useful, please cite our paper:
```
@article{li2019communication,
  title={Communication-Efficient Distributed Optimization in Networks with Gradient Tracking},
  author={Li, Boyue and Cen, Shicong and Chen, Yuxin and Chi, Yuejie},
  journal={arXiv preprint arXiv:1909.05844},
  year={2019}
}
```


## Requirements

- Python 3.6
- Required packages are list in ``requirements.txt``.


## Experiments

- Linear regression (Fig. 1): file ``exp_linear_regression.py``
- Logistic regression (Fig. 2): file ``exp_logistic_regression.py``
- Computation-communication trade-off for Network-SVRG (Fig. 3): file ``exp_svrg_iter_grads.py``
- Network topology (Fig. 4): file ``exp_dane_svrg_topology.py``
- Neural networks (Fig. 5): file ``exp_nn.py``
