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
                    raise NotImplementedError
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
        raise NotImplementedError

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
        raise NotImplementedError


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
        raise NotImplementedError


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
        raise NotImplementedError

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
        raise NotImplementedError