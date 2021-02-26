import unittest
from my_neural_networks.networks import AutogradNeuralNetwork
import torch
from my_neural_networks.activations import cross_entropy
import logging

class TestAutogradNN(unittest.TestCase):
    def test_feed_forward(self):
        torch.manual_seed(29)
        X = torch.tensor([[1, 2, 3], [-4, -5, -6]], dtype=torch.float32)
        ann = AutogradNeuralNetwork((3, 4, 3))

        for i, w in enumerate(ann.weights):
            print(i)
            print(w)

        for i, b in enumerate(ann.biases):
            print(i)
            print(b)

        outputs, act_outputs = ann._feed_forward(X.t())
        print('outputs')
        print(outputs)
        print('act_outputs')
        print(act_outputs)

        print(act_outputs[-1].sum(dim=0))

        # TODO: Use assert to check each dot product and activation.

    def test_train_one_epoch(self):
        torch.manual_seed(29)
        X = torch.tensor([[1, 2, 3], [-4, -5, -6]], dtype=torch.float32)
        y_1hot = torch.tensor([[1, 0, 0], [0, 1, 0]])
        ann = AutogradNeuralNetwork((3, 4, 3))

        o, a = ann._feed_forward(X.t())
        loss = cross_entropy(a[-1], y_1hot.t())
        loss.backward()
        for i in range(len(ann.weights)):
            print(ann.weights[i].grad.data)
        for i in range(len(ann.weights)):
            print(ann.biases[i].grad.data)
