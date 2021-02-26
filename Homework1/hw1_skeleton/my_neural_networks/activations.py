import logging
import numpy as np
import torch

EPSILON = 1e-14


def cross_entropy(X, y_1hot, epsilon=EPSILON):
    """Cross Entropy Loss

        Cross Entropy Loss that assumes the input
        X is post-softmax, so this function only
        does negative loglikelihood. EPSILON is applied
        while calculating log.

    Args:
        X: (n_neurons, n_examples). softmax outputs
        y_1hot: (n_classes, n_examples). 1-hot-encoded labels

    Returns:
        a float number of Cross Entropy Loss (averaged)
    """
    t_log = torch.log(X + epsilon)
    t_prod = y_1hot * t_log
    return - torch.sum(t_prod)/X.shape[1]


def softmax(X):
    """Softmax

        Regular Softmax

    Args:
        X: (n_neurons, n_examples).

    Returns:
        (n_neurons, n_examples). probabilities
    """
    t_exp = torch.exp(X)
    t_sum = torch.sum(t_exp, dim=0, keepdim=True)
    return t_exp/t_sum


def stable_softmax(X):
    """Softmax

        Numerically stable Softmax

    Args:
        X: (n_neurons, n_examples).

    Returns:
        (n_neurons, n_examples). probabilities
    """
    t_max = X.max(dim=0, keepdim=True).values
    t_exp = torch.exp(X - t_max)
    t_sum = torch.sum(t_exp, dim=0, keepdim=True)
    return t_exp / t_sum


def relu(X):
    """Rectified Linear Unit

        Calculate ReLU

    Args:
        X: torch.Tensor

    Returns:
        A torch.Tenor whereThe shape is the same as X but clamped on 0
    """
    X = X.clone()
    X[X < 0] = 0
    return X

def sigmoid(X):
    """Sigmoid Function

        Calculate Sigmoid

    Args:
        X: torch.Tensor

    Returns:
        A torch.Tensor where each element is the sigmoid of the X.
    """
    X_nexp = torch.exp(-X)
    return 1.0 / (1 + X_nexp)