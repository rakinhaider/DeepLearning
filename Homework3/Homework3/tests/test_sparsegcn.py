import unittest
from homework.template.graphsage import SparseGCN
import torch

class TestSparseGCN(unittest.TestCase):
    def test_forward(self):
        gcn = SparseGCN(torch.Generator('cpu'), torch.Generator('cpu'), 2, 2)
        gcn.weight = torch.nn.parameter.Parameter(torch.ones_like(gcn.weight))
        gcn.bias = torch.nn.parameter.Parameter(torch.ones_like(gcn.bias))

        node_feat = torch.tensor([[0.0, 1], [1, 0], [1, 1], [0, 0]])
        # adjacency_feat = torch.tensor([[0.0, 1, 0, 1], [1, 0, 1, 1],
        #                                [0, 1, 0, 0], [1, 1, 0, 0]])

        edges = [(0, 1), (1, 0), (0, 3), (3, 0),
                 (1, 0), (0, 1), (1, 2), (2, 1),
                 (2, 1), (1, 2), (3, 0), (0, 3),
                 (3, 1), (1, 3)]
        edge_buf = list(set(edges))
        edge_mat = torch.LongTensor(edge_buf).view(-1, 2)

        # Remove self-loop connections.
        edge_mat = edge_mat[edge_mat[:, 0] != edge_mat[:, 1]]

        adjacency_feat = edge_mat

        indices = torch.tensor([1, 3])

        forward = gcn.forward(node_feat, adjacency_feat, indices)
        assert forward.shape.__eq__(torch.tensor([4, 2]))
        assert torch.all(
            forward == torch.sigmoid(
                torch.tensor([[2.5, 2.5], [3, 3], [4, 4], [2, 2]])
            )
        )
