# All imports
from typing import Any
from typing import Tuple, List, Dict
from typing import Iterable
from typing import Callable
from typing import Optional
import torch


# =============================================================================
# *****************************************************************************
# -----------------------------------------------------------------------------
# < Homework >
# -----------------------------------------------------------------------------
# *****************************************************************************
# =============================================================================


class SGD(
    torch.optim.Optimizer,
    metaclass=type,
):
    r"""
    SGD optimizer.
    """
    def __init__(
        self,
        /,
        parameters: Iterable[torch.nn.parameter.Parameter],
        *,
        lr: float, weight_decay: float,
    ) -> None:
        r"""
        Initialize the class.

        Args
        ----
        - parameters :
            All parameters being optimized by the optimizer.
        - lr :
            Learning rate.
        - weight_decay :
            Weight decay.

        Returns
        -------
        """
        # /
        # ANNOTATE
        # /
        self.lr: float
        self.weight_decay: float

        # Save necessary attributes.
        self.lr = lr
        self.weight_decay = weight_decay

        # Super call.
        super(SGD, self).__init__(parameters, dict())

    @torch.no_grad()
    def prev(
        self,
        /,
    ) -> None:
        r"""
        Operations before compute the gradient.
        PyTorch has design problem of compute Nesterov SGD gradient.
        PyTorch team avoid this problem by using an approximation of Nesterov
        SGD gradient.
        Also, using closure can also solve the problem, but it maybe a bit
        complicated for this homework.
        In our case, function is provided as auxiliary function for simplicity.
        It is called before `.backward()`.
        This function is only used for Nesterov SGD gradient.

        Args
        ----

        Returns
        -------
        """
        # Do nothing.
        pass

    @torch.no_grad()
    def step(
        self,
        /,
        closure: Optional[Callable[[], float]]=None,
    ) -> Optional[float]:
        r"""
        Performs a single optimization step.

        Args
        ----
        - closure :
            Just to fit PyTorch optimizer annotation.

        Returns
        -------
        """
        # /
        # ANNOTATE
        # /
        group: Dict[str, Any]
        parameter: torch.nn.parameter.Parameter
        gradient: torch.Tensor

        # Traverse parameters of each groups.
        for group in self.param_groups:
            for parameter in group['params']:
                # Get gradient without weight decaying.
                if (parameter.grad is None):
                    continue
                else:
                    gradient = parameter.grad

                # Apply weight decay.
                if (self.weight_decay != 0):
                    # /
                    # YOU SHOULD FILL IN THIS FUNCTION
                    # /
                    gradient += 2 * self.weight_decay * parameter.data
                else:
                    pass

                # Gradient Decay.
                parameter.data.add_(
                    gradient,
                    alpha=-self.lr,
                )
        return None


class Momentum(
    torch.optim.Optimizer,
    metaclass=type,
):
    r"""
    SGD optimizer with momentum.
    """
    def __init__(
        self,
        /,
        parameters: Iterable[torch.nn.parameter.Parameter],
        *,
        lr: float, weight_decay: float, momentum: float,
    ) -> None:
        r"""
        Initialize the class.

        Args
        ----
        - parameters :
            All parameters being optimized by the optimizer.
        - lr :
            Learning rate.
        - weight_decay :
            Weight decay.
        - momentum :
            Momentum.

        Returns
        -------
        """
        # /
        # YOU SHOULD FILL IN THIS FUNCTION
        # /
        self.lr: float
        self.weight_decay: float
        self.momentum: float

        # Save necessary attributes.
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum

        # Super call.
        super(Momentum, self).__init__(parameters, dict())

        # Initialize velocity vector
        self.velocity = []
        for group in self.param_groups:
            self.velocity.append([])
            for param in group['params']:
                print(param.device)
                vel = torch.zeros(param.shape,
                                  device=param.device)
                self.velocity[-1].append(vel)


    @torch.no_grad()
    def prev(
        self,
        /,
    ) -> None:
        r"""
        Operations before compute the gradient.
        PyTorch has design problem of compute Nesterov SGD gradient.
        PyTorch team avoid this problem by using an approximation of Nesterov
        SGD gradient.
        Also, using closure can also solve the problem, but it maybe a bit
        complicated for this homework.
        In our case, function is provided as auxiliary function for simplicity.
        It is called before `.backward()`.
        This function is only used for Nesterov SGD gradient.

        Args
        ----

        Returns
        -------
        """
        # Do nothing.
        pass

    @torch.no_grad()
    def step(
        self,
        /,
        closure: Optional[Callable[[], float]]=None,
    ) -> Optional[float]:
        r"""
        Performs a single optimization step.

        Args
        ----
        - closure :
            Just to fit PyTorch optimizer annotation.

        Returns
        -------
        """
        # /
        # YOU SHOULD FILL IN THIS FUNCTION
        # /
        group: Dict[str, Any]
        parameter: torch.nn.parameter.Parameter
        gradient: torch.Tensor

        # Traverse parameters of each groups.
        for i, group in enumerate(self.param_groups):
            for j, parameter in enumerate(group['params']):
                # Get gradient without weight decaying.
                if (parameter.grad is None):
                    continue
                else:
                    gradient = parameter.grad

                if self.weight_decay != 0:
                    gradient += 2 * self.weight_decay * parameter.data

                # Gradient Decay.
                self.velocity[i][j] = self.momentum * self.velocity[i][j]
                self.velocity[i][j] += self.lr * gradient
                parameter.data.add_(
                    self.velocity[i][j],
                    alpha=-1,
                )
        return None


class Nesterov(
    Momentum,
    metaclass=type,
):
    r"""
    SGD optimizer with Nesterov momentum.
    """
    @torch.no_grad()
    def prev(
        self,
        /,
    ) -> None:
        r"""
        Operations before compute the gradient.
        PyTorch has design problem of compute Nesterov SGD gradient.
        PyTorch team avoid this problem by using an approximation of Nesterov
        SGD gradient.
        Also, using closure can also solve the problem, but it maybe a bit
        complicated for this homework.
        In our case, function is provided as auxiliary function for simplicity.
        It is called before `.backward()`.
        This function is only used for Nesterov SGD gradient.

        Args
        ----

        Returns
        -------
        """
        # /
        # YOU SHOULD FILL IN THIS FUNCTION
        # /
        for i, group in enumerate(self.param_groups):
            for j, parameter in enumerate(group['params']):
                # Gradient Decay.
                parameter.data.add_(
                    self.velocity[i][j],
                    alpha=-self.momentum,
                )
        return None

    def step(
        self,
        /,
        closure: Optional[Callable[[], float]]=None,
    ) -> Optional[float]:
        for i, group in enumerate(self.param_groups):
            for j, parameter in enumerate(group['params']):
                # Gradient Decay.
                parameter.data.add_(
                    self.velocity[i][j],
                    alpha=self.momentum,
                )
        super(Nesterov, self).step(closure)
        return None


class Adam(
    torch.optim.Optimizer,
    metaclass=type,
):
    r"""
    Adam optimizer.
    """
    def __init__(
        self,
        /,
        parameters: Iterable[torch.nn.parameter.Parameter],
        *,
        lr: float, weight_decay: float, beta1: float, beta2: float,
        epsilon: float,
    ) -> None:
        r"""
        Initialize the class.

        Args
        ----
        - parameters :
            All parameters being optimized by the optimizer.
        - lr :
            Learning rate.
        - weight_decay :
            Weight decay.
        - beta1 :
            Beta 1.
        - beta2 :
            Beta 2.
        - Epsilon :
            Epsilon.

        Returns
        -------
        """
        # /
        # YOU SHOULD FILL IN THIS FUNCTION
        # /
        self.lr: float
        self.weight_decay: float
        self.beta1: float
        self.beta2: float
        self.epsilon: float

        # Save necessary attributes.
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.beta1_powt = 1
        self.beta2_powt = 1

        # Super call.
        super(Adam, self).__init__(parameters, dict())

        # Initialize m1, m2 vector
        self.m1 = []
        self.m2 = []
        for group in self.param_groups:
            self.m1.append([])
            self.m2.append([])
            for param in group['params']:
                vel = torch.zeros(param.shape, device=param.device)
                self.m1[-1].append(vel)
                vel = torch.zeros(param.shape, device=param.device)
                self.m2[-1].append(vel)

    @torch.no_grad()
    def prev(
        self,
        /,
    ) -> None:
        r"""
        Operations before compute the gradient.
        PyTorch has design problem of compute Nesterov SGD gradient.
        PyTorch team avoid this problem by using an approximation of Nesterov
        SGD gradient.
        Also, using closure can also solve the problem, but it maybe a bit
        complicated for this homework.
        In our case, function is provided as auxiliary function for simplicity.
        It is called before `.backward()`.
        This function is only used for Nesterov SGD gradient.

        Args
        ----

        Returns
        -------
        """
        # Do nothing.
        pass

    @torch.no_grad()
    def step(
        self,
        /,
        closure: Optional[Callable[[], float]]=None,
    ) -> Optional[float]:
        r"""
        Performs a single optimization step.

        Args
        ----
        - closure :
            Just to fit PyTorch optimizer annotation.

        Returns
        -------
        """
        # /
        # YOU SHOULD FILL IN THIS FUNCTION
        # /
        group: Dict[str, Any]
        parameter: torch.nn.parameter.Parameter
        gradient: torch.Tensor

        self.beta1_powt *= self.beta1
        self.beta2_powt *= self.beta2

        # Traverse parameters of each groups.
        for i, group in enumerate(self.param_groups):
            for j, parameter in enumerate(group['params']):
                # Get gradient without weight decaying.
                if (parameter.grad is None):
                    continue
                else:
                    gradient = parameter.grad

                if self.weight_decay != 0:
                    gradient += 2 * self.weight_decay * parameter.data

                m1 = self.m1[i][j]
                m1 = self.beta1 * m1 + (1 - self.beta1) * gradient
                self.m1[i][j] = m1

                m2 = self.m2[i][j]
                m2 = self.beta2 * m2 + (1 - self.beta2) * gradient
                self.m2[i][j] = m2

                u1 = m1.div(1 - self.beta1_powt)
                print('u1', u1)
                u2 = m2.div(1 - self.beta2_powt)
                print('u2', u2)
                update = (torch.sqrt(u2) + self.epsilon)
                print('update denom', update)
                update = u1.div(update)
                print('update', update)
                parameter.data.add_(
                    update,
                    alpha=-self.lr,
                )
        return None