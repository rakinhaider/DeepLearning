import unittest
from homework.template.gcn import DenseGCN
import torch


class TestDenseGCN(unittest.TestCase):
    def test_forward(self):
        gcn = DenseGCN(torch.Generator('cpu'), torch.Generator('cpu'), 2, 2)
        gcn.weight = torch.nn.parameter.Parameter(torch.ones_like(gcn.weight))
        gcn.bias = torch.nn.parameter.Parameter(torch.ones_like(gcn.bias))

        node_feat = torch.tensor([[0.0, 1], [1, 0], [1, 1], [0, 0]])
        adjacency_feat = torch.tensor([[0.0, 1, 0, 1], [1, 0, 1, 1],
                                      [0, 1, 0, 0], [1, 1, 0, 0]])

        indices = torch.tensor([1, 3])

        forward = gcn.forward(node_feat, adjacency_feat, indices)
        assert forward.shape.__eq__(torch.tensor([2, 2]))
        assert torch.all(
            forward == torch.sigmoid(torch.tensor([[3.0, 3], [2, 2]]))
        )
