import unittest
from homework.template.graphsage import SparseJanossy
import torch

class TestSparseJanossy(unittest.TestCase):
    def test_forward(self):
        torch.manual_seed(23)
        gcn = SparseJanossy(torch.Generator('cpu'), torch.Generator('cpu'), 2, 2,
                            kary=2, num_perms=20)
        gcn.weight = torch.nn.parameter.Parameter(torch.ones_like(gcn.weight))
        gcn.bias = torch.nn.parameter.Parameter(torch.ones_like(gcn.bias))

        node_feat = torch.tensor([[0.0, 1], [1, 0], [1, 1], [0, 0]])

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

        gcn.training = False
        forward = gcn.forward(node_feat, adjacency_feat, indices)
        assert forward.shape.__eq__(torch.tensor([4, 2]))
        # TODO: Need to add assestion for correct values.
        #  need to convert to a for loop with different
        #  combinations of k_ary and permutation.