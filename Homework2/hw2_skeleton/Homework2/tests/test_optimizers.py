import unittest
from torch.nn.parameter import Parameter
import torch
from homework.template.optimizers import (
    Momentum, SGD, Nesterov, Adam
)


class TestOptimizer(unittest.TestCase):
    def _get_test_params(self):
        x = torch.tensor([1, 2, 3, 4], dtype=torch.float32, requires_grad=True)
        p1 = Parameter(x, requires_grad=True)
        p1.grad = torch.tensor([0.1, 0.2, 0.3, 0.4])

        x = torch.tensor([5, 15, 20, 25], dtype=torch.float32, requires_grad=True)
        p2 = Parameter(x, requires_grad=True)
        p2.grad = torch.tensor([1.0, 2, 3, 4])

        x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, requires_grad=True)
        p3 = Parameter(x, requires_grad=True)
        p3.grad = torch.tensor([[1.0, 2, 3], [4, 5, 6]])

        return [{'params': p1}, {'params': [p2, p3]}]

    def test_sgd(self):
        parameters = self._get_test_params()
        optim = SGD(parameters, lr=0.5, weight_decay=0)
        optim.step()
        print(*parameters, end='\n')

    def test_momentum(self):
        parameters = self._get_test_params()
        optim = Momentum(parameters, lr=0.5, weight_decay=0,
                         momentum=0.9)
        optim.step()
        optim.step()

        results = [{'params': [torch.tensor([0.8550, 1.7100, 2.5650, 3.4200],
                                            requires_grad=True)]},
                   {'params': [torch.tensor([3.5500, 12.1000, 15.6500, 19.2000],
                                            requires_grad=True),
                               torch.tensor([[-0.4500, -0.9000, -1.3500],
                                             [-1.8000, -2.2500, -2.7000]],
                                            requires_grad=True)
                               ]
                    }
                   ]
        for i, g in enumerate(optim.param_groups):
            for j, p in enumerate(g['params']):
                assert torch.allclose(results[i]['params'][j], p)

    def test_nesterov(self):
        parameters = self._get_test_params()
        optim = Nesterov(parameters, lr=0.5, weight_decay=0,
                         momentum=0.9)
        optim.step()
        optim.step()

        results = [{'params': [torch.tensor([0.8550, 1.7100, 2.5650, 3.4200],
                                            requires_grad=True)]},
                   {'params': [torch.tensor([3.5500, 12.1000, 15.6500, 19.2000],
                                            requires_grad=True),
                               torch.tensor([[-0.4500, -0.9000, -1.3500],
                                             [-1.8000, -2.2500, -2.7000]],
                                            requires_grad=True)
                               ]
                    }
                   ]

        for i, g in enumerate(optim.param_groups):
            for j, p in enumerate(g['params']):
                assert torch.allclose(results[i]['params'][j], p)

    def test_adam(self):
        parameters = self._get_test_params()
        optim = Adam(parameters, lr=0.5, weight_decay=0,
                     beta1=0.1, beta2=0.2, epsilon=1)
        optim.step()
        print('m1', *optim.m1, sep='\n')
        print('beta1_powt', optim.beta1_powt)
        print('m2', *optim.m2, sep='\n')
        print('beta2_powt', optim.beta2_powt)
        print('params', *optim.param_groups, sep='\n')
        optim.step()

        print('m1', *optim.m1, sep='\n')
        print('beta1_powt', optim.beta1_powt)
        print('m2', *optim.m2, sep='\n')
        print('beta2_powt', optim.beta2_powt)
        print('params', *optim.param_groups, sep='\n')

        results = [{'params': [torch.tensor([0.8550, 1.7100, 2.5650, 3.4200],
                                            requires_grad=True)]},
                   {'params': [torch.tensor([3.5500, 12.1000, 15.6500, 19.2000],
                                            requires_grad=True),
                               torch.tensor([[-0.4500, -0.9000, -1.3500],
                                             [-1.8000, -2.2500, -2.7000]],
                                            requires_grad=True)
                               ]
                    }
                   ]

    def test_weight_decay(self):
        # test weight decay for all 4 optim.
        pass