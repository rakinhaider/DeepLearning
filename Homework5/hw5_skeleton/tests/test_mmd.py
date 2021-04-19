import unittest
from mmd import compute_mmd
import torch


class TestMMD(unittest.TestCase):
    def test_compute_mmd(self):
        torch.manual_seed(23)
        X = torch.randint(0, 10, (2, 2))
        Y = torch.randint(0, 10, (3, 2))
        mmd = compute_mmd(X, Y, 10)
        assert mmd == -0.043013811111450195
