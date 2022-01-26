# Network-Distributed Algorithm Experiments

This repository contains a set of optimization algorithms and objective functions, and all code needed to reproduce experiments in:

1. "DESTRESS: Computation-Optimal and Communication-Efficient Decentralized Nonconvex Finite-Sum Optimization" [[PDF](https://arxiv.org/abs/2110.01165)]. (code is in this file [[link](https://github.com/liboyue/Network-Distributed-Algorithm/blob/master/experiments/papers/destress.py)])

2. "Communication-Efficient Distributed Optimization in Networks with Gradient Tracking and Variance Reduction" [[PDF](https://arxiv.org/abs/1909.05844v2)]. (code is in the previous version of this repo [[link](https://github.com/liboyue/Network-Distributed-Algorithm/tree/08abe14f2a2d5929fc401ff99961ca3bae40ff60)])

Due to the random data generation procedure,
resulting graphs may be slightly different from those appeared in the paper,
but conclusions remain the same.

If you find this code useful, please cite our papers:

```
@article{li2021destress,
  title={DESTRESS: Computation-Optimal and Communication-Efficient Decentralized Nonconvex Finite-Sum Optimization},
  author={Li, Boyue and Li, Zhize and Chi, Yuejie},
  journal={arXiv preprint arXiv:2110.01165},
  year={2021}
}
```

```
@article{li2020communication,
  title={Communication-Efficient Distributed Optimization in Networks with Gradient Tracking and Variance Reduction},
  author={Li, Boyue and Cen, Shicong and Chen, Yuxin and Chi, Yuejie},
  journal={Journal of Machine Learning Research},
  volume={21},
  pages={1--51},
  year={2020}
}
```

## Implemented objective functions
The gradient implementations of all objective functions are checked numerically.

### Linear regression
Linear regression with random generated data.
The objective function is
<img src="https://render.githubusercontent.com/render/math?math=f(w) = \frac{1}{N} \sum_i (y_i - x_i^\top w)^2">

### Logistic regression
Logistic regression with $l$-2 or nonconvex regularization with random generated data or the Gisette dataset or datasets from `libsvmtools`.
The objective function is
<img src="https://render.githubusercontent.com/render/math?math=f(w) =  - \frac{1}{N} * \Big(\sum_i y_i \log \frac{1}{1 + exp(w^T x_i)} + (1 - y_i) \log \frac{exp(w^T x_i)}{1 + exp(w^T x_i)} \Big) + \frac{\lambda}{2} \| w \|_2^2 + \alpha \sum_j \frac{w_j^2}{1 + w_j^2}2">


### One-hidden-layer fully-connected neural netowrk
One-hidden-layer fully-connected neural network with softmax loss on the MNIST dataset.


## Implemented optimization algorithms

### Centralized optimization algorithms
- Gradient descent
- Stochastic gradient descent
- Nesterov's accelerated gradient descent
- SVRG
- SARAH

### Distributed optimization algorithms (i.e. with parameter server)
- ADMM
- DANE


### Decentralized optimization algorithms
- Decentralized gradient descent
- Decentralized stochastic gradient descent
- Decentralized gradient descent with gradient tracking
- EXTRA
- NIDS
- Network-DANE/SARAH/SVRG
- GT-SARAH
- DESTRESS
