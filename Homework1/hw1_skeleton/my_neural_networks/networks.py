import logging
import math
import numpy as np
import torch
from copy import deepcopy
from collections import OrderedDict

from .activations import relu, softmax, cross_entropy, stable_softmax

logger = logging.getLogger(__name__)


class AutogradNeuralNetwork:
    """Implementation that uses torch.autograd

        Neural network classifier with cross-entropy loss
        and ReLU activations
    """
    def __init__(self, shape, gpu_id=-1):
        """Initialize the network

        Args:
            shape: a list of integers that specifieds
                    the number of neurons at each layer.
            gpu_id: -1 means using cpu. 
        """
        self.shape = shape
        # declare weights and biases
        if gpu_id == -1:
            self.weights = [torch.autograd.Variable(torch.FloatTensor(j, i),
                                requires_grad=True)
                            for i, j in zip(self.shape[:-1], self.shape[1:])]
            self.biases = [torch.autograd.Variable(torch.FloatTensor(i, 1),
                                requires_grad=True)
                           for i in self.shape[1:]]
        else:
            self.weights = [torch.autograd.Variable(torch.randn(j, i).cuda(gpu_id),
                                requires_grad=True)
                            for i, j in zip(self.shape[:-1], self.shape[1:])]
            self.biases = [torch.autograd.Variable(torch.randn(i, 1).cuda(gpu_id),
                                requires_grad=True)
                           for i in self.shape[1:]]
        # initialize weights and biases
        self.init_weights()

    def init_weights(self):
        """Initialize weights and biases

            Initialize self.weights and self.biases with
            Gaussian where the std is 1 / sqrt(n_neurons)
        """
        for i in range(len(self.weights)):
            w = self.weights[i]
            b = self.biases[i]
            stdv = 1. / math.sqrt(w.size(1))
            # in-place random
            w.data.uniform_(-stdv, stdv)
            b.data.uniform_(-stdv, stdv)

    def _feed_forward(self, X):
        """Forward pass

        Args:
            X: (n_neurons, n_examples)

        Returns:
            (outputs, act_outputs).

            "outputs" is a list of torch tensors. Each tensor is the Wx+b (weighted sum plus bias)
            of each layer in the shape (n_neurons, n_examples).

            "act_outputs" is also a list of torch tensors. Each tensor is the "activated" outputs
            of each layer in the shape(n_neurons, n_examples). If f(.) is the activation function,
            this should be f(ouptuts).
        """
        outputs = []
        act_outputs = []

        previous = X
        for i in range(len(self.shape)-1):
            w = self.weights[i]
            b = self.biases[i]
            outputs.append(torch.matmul(w, previous) + b)
            if i != len(self.shape) - 2:
                act_outputs.append(relu(outputs[i]))
                previous = act_outputs[i]
            else:
                act_outputs.append(stable_softmax(outputs[i]))

        return outputs, act_outputs

    def train_one_epoch(self, X, y, y_1hot, learning_rate):
        """Train for one epoch

        Args:
            X: (n_examples, n_features)
            y: (n_examples). don't care
            y_1hot: (n_examples, n_classes)

        Returns:
            Loss for the given input
        """
        X_t = X.t()
        y_1hot_t = y_1hot.t()

        X_t_train = X_t
        y_1hot_t_train = y_1hot_t
        
        # feed forward
        outputs, act_outputs = self._feed_forward(X_t_train)
        loss = cross_entropy(act_outputs[-1], y_1hot_t_train)

        # backward
        loss.backward()

        # update weights and biases
        for w, b in zip(self.weights, self.biases):
            w.data = w.data - (learning_rate * w.grad.data)
            b.data = b.data - (learning_rate * b.grad.data)
            w.grad.data.zero_()
            b.grad.data.zero_()

        return loss.data.item()
    
    def loss(self, X, y, y_1hot):
        """Compute feed forward loss

        Args:
            X: (n_examples, n_features)
            y: (n_examples). don't care
            y_1hot: (n_examples, n_classes)

        Returns:
            Loss for the given input
        """
        X_t = X.t()
        y_1hot_t = y_1hot.t()
        outputs, act_outputs = self._feed_forward(X_t)
        loss = cross_entropy(act_outputs[-1], y_1hot_t)
        return loss.data

    def predict(self, X):
        """Predict

            Make predictions for X using the current model

        Args:
            X: (n_examples, n_features)

        Returns:
            (n_examples, )
        """
        outputs, act_outputs = self._feed_forward(X.t())
        return torch.max(act_outputs[-1], 0)[1]


class BasicNeuralNetwork:
    """Implementation using only torch.Tensor

        Neural network classifier with cross-entropy loss
        and ReLU activations
    """
    def __init__(self, shape, gpu_id=-1):
        """Initialize the network

        Args:
            shape: a list of integers that specifieds
                    the number of neurons at each layer.
            gpu_id: -1 means using cpu. 
        """
        self.shape = shape
        # declare weights and biases
        if gpu_id == -1:
            self.weights = [torch.FloatTensor(j, i)
                            for i, j in zip(self.shape[:-1], self.shape[1:])]
            self.biases = [torch.FloatTensor(i, 1)
                           for i in self.shape[1:]]
        else:
            self.weights = [torch.randn(j, i).cuda(gpu_id)
                            for i, j in zip(self.shape[:-1], self.shape[1:])]
            self.biases = [torch.randn(i, 1).cuda(gpu_id)
                           for i in self.shape[1:]]

        # initialize weights and biases
        self.init_weights()

    def init_weights(self):
        """Initialize weights and biases

            Initialize self.weights and self.biases with
            Gaussian where the std is 1 / sqrt(n_neurons)
        """
        for i in range(len(self.weights)):
            w = self.weights[i]
            b = self.biases[i]
            stdv = 1. / math.sqrt(w.size(1))
            # in-place random
            w.uniform_(-stdv, stdv)
            b.uniform_(-stdv, stdv)

    def _feed_forward(self, X):
        """Forward pass

        Args:
            X: (n_neurons, n_examples)
        
        Returns:
            (outputs, act_outputs).

            "outputs" is a list of torch tensors. Each tensor is the Wx+b (weighted sum plus bias)
            of each layer in the shape (n_neurons, n_examples).

            "act_outputs" is also a list of torch tensors. Each tensor is the "activated" outputs
            of each layer in the shape(n_neurons, n_examples). If f(.) is the activation function,
            this should be f(ouptuts).
        """
        outputs = []
        act_outputs = []

        previous = X
        for i in range(len(self.shape) - 1):
            w = self.weights[i]
            b = self.biases[i]
            outputs.append(torch.matmul(w, previous) + b)
            if i != len(self.shape) - 2:
                act_outputs.append(relu(outputs[i]))
                previous = act_outputs[i]
            else:
                act_outputs.append(stable_softmax(outputs[i]))

        return outputs, act_outputs

    def _backpropagation(self, outputs, act_outputs, X, y_1hot):
        """Backward pass

        Args:
            outputs: (n_neurons, n_examples). get from _feed_forward()
            act_outputs: (n_neurons, n_examples). get from _feed_forward()
            X: (n_features, n_examples). input features
            y_1hot: (n_classes, n_examples). labels
        """

        n_layers = len(self.weights)

        y_cap = act_outputs[-1]
        grad_loss_zout = y_cap - y_1hot

        w_grads = [None for i in range(n_layers)]
        b_grads = [None for i in range(n_layers)]
        grad_loss_zi = grad_loss_zout
        for i in range(n_layers-1, 0, -1):
            w_grads[i] = torch.matmul(grad_loss_zi, act_outputs[i-1].t())
            b_grads[i] = grad_loss_zi.detach().clone().sum(dim=1, keepdim=True)

            grad_loss_hi = torch.matmul(self.weights[i].t(), grad_loss_zi)
            grad_loss_zi = grad_loss_hi.detach().clone()
            mask = act_outputs[i-1] <= 0
            grad_loss_zi[mask] = 0

        w_grads[0] = torch.matmul(grad_loss_zi, X.t())
        b_grads[0] = grad_loss_zi.detach().clone().sum(dim=1, keepdim=True)

        w_grads = [w/X.shape[1] for w in w_grads]
        b_grads = [b/X.shape[1] for b in b_grads]

        return w_grads, b_grads

    def train_one_epoch(self, X, y, y_1hot, learning_rate):
        """Train for one epoch

        Args:
            X: (n_examples, n_features)
            y: (n_examples). don't care
            y_1hot: (n_examples, n_classes)

        Returns:
            Loss for the given input
        """
        X_t = X.t()
        y_1hot_t = y_1hot.t()

        X_t_train = X_t
        y_1hot_t_train = y_1hot_t

        # feed forward
        outputs, act_outputs = self._feed_forward(X_t_train)
        loss = cross_entropy(act_outputs[-1], y_1hot_t_train)

        # backward
        # loss.backward()
        w_grads, b_grads = self._backpropagation(outputs, act_outputs,
                                                 X_t_train, y_1hot_t_train)

        # update weights and biases
        for i in range(len(self.weights)):
            w, b = self.weights[i], self.biases[i]
            w_grad, b_grad = w_grads[i], b_grads[i]
            w = w - (learning_rate * w_grad)
            b = b - (learning_rate * b_grad)
            self.weights[i], self.biases[i] = w, b

        return loss.data.item()

    def loss(self, X, y, y_1hot):
        """Compute feed forward loss

        Args:
            X: (n_examples, n_features)
            y: (n_examples). don't care
            y_1hot: (n_examples, n_classes)

        Returns:
            Loss for the given input
        """
        X_t = X.t()
        y_1hot_t = y_1hot.t()
        outputs, act_outputs = self._feed_forward(X_t)
        loss = cross_entropy(act_outputs[-1], y_1hot_t)
        return loss

    def predict(self, X):
        """Predict

            Make predictions for X using the current model

        Args:
            X: (n_examples, n_features)

        Returns:
            (n_examples, )
        """
        outputs, act_outputs = self._feed_forward(X.t())
        return torch.max(act_outputs[-1], 0)[1]
