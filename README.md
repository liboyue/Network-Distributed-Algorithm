# Network-Distributed Algorithm Experiments

This repository contains a set of optimization algorithms and objective functions, and all code needed to reproduce experiments in:

1. "DESTRESS: Computation-Optimal and Communication-Efficient Decentralized Nonconvex Finite-Sum Optimization" [[PDF](https://arxiv.org/abs/2110.01165)]. (code is in this file [[link](https://github.com/liboyue/Network-Distributed-Algorithm/blob/master/experiments/papers/destress.py)])

2. "Communication-Efficient Distributed Optimization in Networks with Gradient Tracking and Variance Reduction" [[PDF](https://arxiv.org/abs/1909.05844v2)]. (code is in the previous version of this repo [[link](https://github.com/liboyue/Network-Distributed-Algorithm/tree/08abe14f2a2d5929fc401ff99961ca3bae40ff60)])

Due to the random data generation procedure,
results may be slightly different from those appeared in papers,
but conclusions remain the same.

If you find this code useful, please cite our papers:

```
@article{li2022destress,
  title={DESTRESS: Computation-Optimal and Communication-Efficient Decentralized Nonconvex Finite-Sum Optimization},
  author={Li, Boyue and Li, Zhize and Chi, Yuejie},
  journal={SIAM Journal on Mathematics of Data Science},
  volume = {4},
  number = {3},
  pages = {1031-1051},
  year={2022}
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


## 1. Features
- Easy to use: come with several popular objective functions with optional regularization and compression, essential optimization algorithms, utilities to run experiments and plot results
- Extendability: easy to implement your own objective functions / optimization algorithms / datasets
- Correctness: numerically verified gradient implementation
- Performance: can run on both CPU and GPU
- Data preprocessing: shuffling, normalizing, splitting


## 2. Installation and usage
### 2.1 Installation

`pip install git+https://github.com/liboyue/Network-Distributed-Algorithm.git`

If you have Nvidia GPUs, please also install `cupy`.

### 2.2 Implementing your own objective function
### 2.3 Implementing your own optimizer


## 3. Objective functions
The gradient implementations of all objective functions are checked numerically.

### 3.1 Linear regression
Linear regression with random generated data.
The objective function is
<img src="https://render.githubusercontent.com/render/math?math=f(w) = \frac{1}{N} \sum_i (y_i - x_i^\top w)^2">

### 3.2 Logistic regression
Logistic regression with l-2 or nonconvex regularization with random generated data or the Gisette dataset or datasets from `libsvmtools`.
The objective function is
<img src="https://render.githubusercontent.com/render/math?math=f(w) = - \frac{1}{N} * \Big(\sum_i y_i \log \frac{1}{1 %2B exp(w^T x_i)} %2B (1 - y_i) \log \frac{exp(w^T x_i)}{1 %2B exp(w^T x_i)} \Big) %2B \frac{\lambda}{2} \| w \|_2^2 %2B \alpha \sum_j \frac{w_j^2}{1 %2B w_j^2}">

### 3.3 One-hidden-layer fully-connected neural netowrk
One-hidden-layer fully-connected neural network with softmax loss on the MNIST dataset.


## 4. Datasets
- MNIST
- Gisette
- LibSVM data
- Random generated data


## 5. Optimization algorithms

### 5.1 Centralized optimization algorithms
- Gradient descent
- Stochastic gradient descent
- Nesterov's accelerated gradient descent
- SVRG
- SARAH

### 5.2 Distributed optimization algorithms (i.e. with parameter server)
- ADMM
- DANE

### 5.3 Decentralized optimization algorithms
- Decentralized gradient descent
- Decentralized stochastic gradient descent
- Decentralized gradient descent with gradient tracking
- EXTRA
- NIDS
- D2
- CHOCO-SGD
- Network-DANE/SARAH/SVRG
- GT-SARAH
- DESTRESS


## 6. Change log

- Mar-03-2022: Add GPU support, refactor code
