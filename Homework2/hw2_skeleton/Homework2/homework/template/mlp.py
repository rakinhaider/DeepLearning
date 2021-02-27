# All imports
import torch


# =============================================================================
# *****************************************************************************
# -----------------------------------------------------------------------------
# < Homework >
# -----------------------------------------------------------------------------
# *****************************************************************************
# =============================================================================


class GInvariantLinear(
    torch.nn.Linear,
    metaclass=type,
):
    r"""
    G-Invariant Linear Layer.
    """
    def __init__(
        self,
        /,
        rng_cpu: torch.Generator, rng_gpu: torch.Generator,
        *,
        num_inputs: int, num_outputs: int,
        eigenvectors: torch.Tensor,
    ) -> None:
        r"""
        Initialize the class.

        Args
        ----
        - rng_cpu :
            Random number generator for CPU.
        - rng_gpu :
            Random number generator for GPU.
        - num_inputs :
            Number of inputs.
        - num_outputs :
            Number of outputs.
        - eigenvectors :
            Non-trival eigenvectors of invariant subspace.

        Returns
        -------
        """
        # /
        # ANNOTATE
        # /
        self.rng_cpu: torch.Generator
        self.rng_gpu: torch.Generator
        self.eigenvectors: torch.Tensor

        # Super call.
        super(GInvariantLinear, self).__init__(len(eigenvectors), num_outputs)

        # Save necessary attributes.
        self.rng_cpu = rng_cpu
        self.rng_gpu = rng_gpu
        self.register_buffer("eigenvectors", eigenvectors)

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