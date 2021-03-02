import unittest
import torch
from homework.template.mlp import GInvariantLinear


class TestMLP(unittest.TestCase):
    def test_forward(self):
        eigenvectors = torch.randn((4, 6))
        gl = GInvariantLinear(torch.Generator('cpu'),
                              torch.Generator('cpu'),
                              num_inputs=6,
                              num_outputs=10,
                              eigenvectors=eigenvectors)

        x = torch.randn((2, 6))

        print(gl.weight.shape)

        print(gl.forward(x).shape)

    def test_matmul(self):
        torch.manual_seed(47)
        weight = torch.randint(0, 5, (3, 4), dtype=torch.float32)
        print(weight)
        print(weight[0])
        eigenvectors = torch.randint(0, 5, (4, 6), dtype=torch.float32)
        for i in range(weight.shape[0]):
            print(weight[i])
            print(eigenvectors)
            print(weight[i].matmul(eigenvectors))

        wt = weight.matmul(eigenvectors)
        print(wt)
        x = torch.randint(0, 5, (2, 6), dtype=torch.float32)
        print(x.t())
        print(x.matmul(wt.t()))

        l = torch.nn.Linear(6, 3)
        print(l.forward(x).shape)