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

        forward = cnn(x)
        result = torch.tensor([[[1453., 1621., 1621., 1621.],
                                [1453., 1621., 1621., 1621.],
                                [1453., 1621., 1621., 1621.],
                                [1453., 1621., 1621., 1621.]]])
        result = result.repeat([1, 3, 1, 1])
        assert torch.equal(result, forward)

    def test_filter_size(self):
        torch.manual_seed(23)
        cnn = DualCNN(
            rng_cpu=torch.Generator('cpu'), rng_gpu=torch.Generator('cpu'),
            num_input_channels=1, num_output_channels=3,
            num_internal_channels=2,
            conv_kernel=3, conv_stride=3,
            pool_kernel=3, pool_stride=1, padding=1
        )
        _debug_initializer(cnn.conv1)
        _debug_initializer(cnn.conv2)
        x = torch.randint(0, 5, (1, 1, 28, 28), dtype=torch.float32)

        forward = cnn(x)
        result = torch.tensor([[[1925., 1925., 1925., 1901.],
                                [1925., 1925., 1925., 1901.],
                                [1925., 1925., 1925., 1901.],
                                [1813., 1813., 1813., 1725.]]])
        result = result.repeat([1, 3, 1, 1])
        assert torch.equal(result, forward)

        cnn = DualCNN(
            rng_cpu=torch.Generator('cpu'), rng_gpu=torch.Generator('cpu'),
            num_input_channels=1, num_output_channels=3,
            num_internal_channels=2,
            conv_kernel=14, conv_stride=1,
            pool_kernel=3, pool_stride=1, padding=1
        )
        _debug_initializer(cnn.conv1)
        _debug_initializer(cnn.conv2)

        forward = cnn(x)
        result = torch.tensor([[[625761., 628361., 630897.,
                                 633249., 633249., 633249.],
                                [628873., 631073., 633193.,
                                 635105., 635105., 635105.],
                                [631689., 633417., 635049.,
                                 636449., 636449., 636449.],
                                [634305., 635577., 636737.,
                                 637641., 637641., 637641.],
                                [634305., 635577., 636737.,
                                 637641., 637641., 637641.],
                                [634305., 635577., 636737.,
                                 637641., 637641., 637641.]]])
        result = result.repeat([1, 3, 1, 1])

        assert torch.equal(result, forward)


def _debug_initializer(conv):
    conv.weight.data = 2 * torch.ones_like(conv.weight.data)
    conv.bias.data = torch.ones_like(conv.bias.data)
