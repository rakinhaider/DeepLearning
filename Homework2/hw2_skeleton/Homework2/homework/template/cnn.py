# All imports
import torch
import math


# =============================================================================
# *****************************************************************************
# -----------------------------------------------------------------------------
# < Homework >
# -----------------------------------------------------------------------------
# *****************************************************************************
# =============================================================================


class DualCNN(
    torch.nn.Module,
    metaclass=type,
):
    r"""
    2-layer CNN Layer.
    """
    def __init__(
        self,
        /,
        rng_cpu: torch.Generator, rng_gpu: torch.Generator,
        *,
        num_input_channels: int, num_output_channels: int,
        num_internal_channels: int,
        conv_kernel: int, conv_stride: int,
        pool_kernel: int, pool_stride: int, padding: int,
    ) -> None:
        r"""
        Initialize the class.

        Args
        ----
        - rng_cpu :
            Random number generator for CPU.
        - rng_gpu :
            Random number generator for GPU.
        - num_input_channels :
            Number of input channels.
        - num_output_channels :
            Number of output channels.
        - num_internal_channels :
            Number of internal channels.
        - conv_kernel :
            Square kernel size for convolution.
        - conv_stride :
            Square stride size for convolution.
        - pool_kernel :
            Square kernel size for pooling.
        - pool_stride :
            Square stride size for pooling.
        - padding :
            Number of zero-paddings on each side.

        Returns
        -------
        """
        # /
        # ANNOTATE
        # /
        self.rng_cpu: torch.Generator
        self.rng_gpu: torch.Generator

        # Super call.
        super(DualCNN, self).__init__()

        # Save necessary attributes.
        self.rng_cpu = rng_cpu
        self.rng_gpu = rng_gpu

        # /
        # YOU SHOULD FILL IN THIS FUNCTION
        # /
        raise NotImplementedError

    def forward(
        self,
        /,
        input: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Forward.

        Args
        ----
        - input :
            Inputs.

        Returns
        -------
        - outputs :
            Outputs.
        """
        # /
        # YOU SHOULD FILL IN THIS FUNCTION
        # /
        raise NotImplementedError

    def initialize(
        self,
        /,
        rng_cpu: torch.Generator,
    ) -> None:
        r"""
        Initialize.

        Args
        ----
        - rng_cpu :
            Random number generator for CPU.

        Returns
        -------
        """
        # /
        # ANNOTATE
        # /
        conv: torch.nn.Conv2d
        chout: int
        chin: int
        reph: int
        repw: int
        fan_in: int
        fan_out: int

        # Use Xaiver uniform.
        for conv in [self.conv1, self.conv2]:
            chout, chin, reph, repw = conv.weight.data.size()
            fan_in = chin * reph * repw
            fan_out = chout * reph * repw
            bound = math.sqrt(6.0 / float(fan_in + fan_out))
            conv.weight.data.uniform_(
                -bound, bound,
                generator=rng_cpu,
            )
            conv.bias.data.uniform_(
                -bound, bound,
                generator=rng_cpu,
            )