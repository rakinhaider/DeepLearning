import logging
import unittest
import torch
from my_neural_networks.networks import BasicNeuralNetwork, AutogradNeuralNetwork


class TestBasicNeuralNetwork(unittest.TestCase):
    def test_backpropagation(self):
        torch.manual_seed(29)
        X = torch.tensor([[1, 2, 3], [-4, -5, -6]], dtype=torch.float32)
        y_1hot = torch.tensor([[1, 0, 0], [0, 1, 0]])
        print(X.t().shape)
        print(y_1hot.t().shape)
        bnn = BasicNeuralNetwork((3, 4, 3))

        outputs, act_outputs = bnn._feed_forward(X.t())
        print(*outputs, sep='\n')
        print(*act_outputs, sep='\n')
        bnn._backpropagation(outputs, act_outputs, X.t(), y_1hot.t())

    def test_train_one_epoch(self):
        torch.manual_seed(29)
        X = torch.tensor([[1, 2, 3], [-4, -5, -6]], dtype=torch.float32)
        y_1hot = torch.tensor([[1, 0, 0], [0, 1, 0]])
        ann = AutogradNeuralNetwork((3, 4, 3))
        bnn = BasicNeuralNetwork((3, 4, 3))
        for i in range(len(ann.weights)):
            bnn.weights[i] = ann.weights[i]
            bnn.biases[i] = ann.biases[i]

        for i in range(10):
            bnn_loss = bnn.train_one_epoch(X, y_1hot, y_1hot, 1e-4)
            ann_loss = ann.train_one_epoch(X, y_1hot, y_1hot, 1e-4)
            print(ann_loss, bnn_loss)
            assert ann_loss == bnn_loss
