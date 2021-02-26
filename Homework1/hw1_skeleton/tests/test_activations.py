from unittest import TestCase
import torch
from my_neural_networks.activations import (
    relu, softmax,
    stable_softmax, cross_entropy
)


class TestActivations(TestCase):
    def test_relu(self):
        x = torch.tensor([1, 0, -3, 4, -5, 10], dtype=torch.float)
        relud = relu(x)
        assert relud[2] == 0 and relud[4] == 0

    def test_softmax(self):
        x = torch.tensor([[0.1, 0.0, -0.3, 0.4, -0.5, 1.0],
                          [-0.1, 0.0, 0.3, -0.4, 0.5, -1.0],
                          [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]],
                         dtype=torch.float)
        x = x.transpose(0, -1)
        sm = softmax(x)
        assert torch.allclose(sm.sum(dim=0), torch.ones(x.shape[1]))

    def test_stable_softmax(self):
        x = torch.tensor([[0.1, 0.0, -0.3, 0.4, -0.5, 1.0],
                          [-0.1, 0.0, 0.3, -0.4, 0.5, -1.0],
                          [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]],
                         dtype=torch.float)
        x = x.transpose(0, -1)
        ssm = stable_softmax(x)
        assert torch.allclose(ssm.sum(dim=0), torch.ones(x.shape[1]))

    def test_cross_entropy(self):
        x = torch.tensor([[0.9, 0.05, 0.05],
                          [0.25, 0.25, 0.5]])
        x = x.transpose(dim0=0, dim1=-1)
        y_1hot = torch.tensor([[1.0, 0, 0], [0, 0, 1]])
        y_1hot = y_1hot.transpose(dim0=0, dim1=-1)
        print(cross_entropy(x, y_1hot))
        print(torch.nn.BCELoss()(x, y_1hot))

