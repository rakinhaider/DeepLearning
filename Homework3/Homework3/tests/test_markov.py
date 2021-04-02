import unittest
from homework.template.markov import Markov
import torch

class TestMarkov(unittest.TestCase):
    def test_forward(self):
        torch.manual_seed(23)
        vocab = 5
        m = Markov(torch.Generator('cpu'), torch.Generator('cpu'), vocab, 2)
        m.initialize(torch.Generator('cpu'))
        input = torch.randint(0, vocab + 1, (10, ))
        input[4] = 3
        input[6] = 0
        m.training = True
        m.forward(input)

        input = torch.randint(0, vocab + 1, (10,))
        input[6] = 2
        m.training = False
        predictions = m.forward(input)
        pred_orig = torch.tensor([0.5000, 1/9,
                                  0.2000, 0.2000,
                                  0.2000, 0.2000,
                                  0.2000, 0.2000,
                                  1/3, 0.1250])
        assert torch.allclose(predictions, pred_orig)
