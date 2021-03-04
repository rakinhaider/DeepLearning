import unittest
from homework.template.cnn import DualCNN
import torch


class TestCNN(unittest.TestCase):
    def test_forward(self):
        torch.manual_seed(23)
        cnn = DualCNN(
            rng_cpu=torch.Generator('cpu'), rng_gpu=torch.Generator('cpu'),
            num_input_channels=1, num_output_channels=3,
            num_internal_channels=2,
            conv_kernel=3, conv_stride=1,
            pool_kernel=3, pool_stride=1, padding=1
        )
        print(cnn)
        _debug_initializer(cnn.conv1)
        _debug_initializer(cnn.conv2)
        x = torch.randint(0, 5, (1, 1, 4, 4), dtype=torch.float32)
        print(x)

        print(cnn(x))

    def test_init(self):
        cnn = DualCNN(
            rng_cpu=torch.Generator('cpu'), rng_gpu=torch.Generator('cpu'),
            num_input_channels=1, num_output_channels=3,
            num_internal_channels=2,
            conv_kernel=3, conv_stride=1,
            pool_kernel=3, pool_stride=1, padding=1
        )
        _debug_initializer(cnn.conv1)
        _debug_initializer(cnn.conv2)
        print(list(cnn.conv1.named_parameters()))


def _debug_initializer(conv):
    conv.weight.data = 2 * torch.ones_like(conv.weight.data)
    conv.bias.data = torch.ones_like(conv.bias.data)
