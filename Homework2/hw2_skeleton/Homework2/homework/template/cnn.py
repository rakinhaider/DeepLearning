# All imports
import torch
import math
from torch.nn import Conv2d, MaxPool2d, Linear
import torch.backends.cudnn as cudnn

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
        self.num_input_channels: int
        self.num_output_channels: int
        self.num_internal_channels: int
        self.conv_kernel: int
        self.conv_stride: int
        self.pool_kernel: int
        self.pool_stride: int
        self.padding: int

        self.conv1 = Conv2d(in_channels=num_input_channels,
                            out_channels=num_internal_channels,
                            kernel_size=conv_kernel,
                            stride=conv_stride, padding=padding)
        self.pool = MaxPool2d(kernel_size=pool_kernel,
                              stride=pool_stride,
                              padding=padding)
        self.conv2 = Conv2d(in_channels=num_internal_channels,
                            out_channels=num_output_channels,
                            kernel_size=conv_kernel,
                            stride=conv_stride, padding=padding)
        cudnn.deterministic = True
        cudnn.benchmark = False

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
        x = torch.relu(self.conv1(input))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)

        return x


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
