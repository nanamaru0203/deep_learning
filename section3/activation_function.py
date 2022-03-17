import numpy as np


def step_function(x):
    y = x > 0
    return y.astype(np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ReLU(x):
    return np.maximum(0, x)


def identity_function(x):
    return x


def softmax(x):
    c = np.max(x)
    x_exp = np.exp(x - c)
    return x_exp / np.sum(x_exp)
