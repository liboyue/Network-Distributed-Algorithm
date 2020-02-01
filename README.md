# Network-Distributed Algorithm Experiments

This repository contains all code needed to reproduce experiments in "Communication-Efficient Distributed Optimization in Networks with Gradient Tracking and Variance Reduction" [[PDF](https://arxiv.org/abs/1909.05844v2)].

Due to the random data generation procedure,
resulting graphs may be different from those appeared in the paper,
but conclusions remain the same.

If you find this code useful, please cite our paper:
```
@article{li2019communication,
  title={Communication-Efficient Distributed Optimization in Networks with Gradient Tracking and Variance Reduction},
  author={Li, Boyue and Cen, Shicong and Chen, Yuxin and Chi, Yuejie},
  journal={arXiv preprint arXiv:1909.05844},
  year={2019}
}
```


## Requirements

- Python 3
- Required packages are list in ``requirements.txt``.


## Experiments

- Linear regression: file ``exp_linear_regression.py``
- Logistic regression: file ``exp_logistic_regression.py``
- Neural networks: file ``exp_nn.py``
- Network topology: file ``exp_dane_svrg_topology.py``
- Computation-communication trade-off for Network-SVRG: file ``exp_svrg_local_iters.py``
