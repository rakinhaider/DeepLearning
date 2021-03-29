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


class DenseGCN(
    torch.nn.Module,
    metaclass=type,
):
    r"""
    GCN for dense adjacency input.
    """
    # =========================================================================
    # -------------------------------------------------------------------------
    # =========================================================================
    def __init__(
        self,
        rng_cpu: torch.Generator, rng_gpu: torch.Generator,
        num_inputs: int, num_outputs: int,
        /,
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

        Returns
        -------
        """
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # /
        # ANNOTATE
        # /
        self.rng_cpu: torch.Generator
        self.rng_gpu: torch.Generator
        self.num_inputs: int
        self.num_outputs: int

        # Super call.
        super(DenseGCN, self).__init__()

        # Save necessary attributes.
        self.rng_cpu = rng_cpu
        self.rng_gpu = rng_gpu
        self.num_inputs = num_inputs * 2
        self.num_outputs = num_outputs
        self.weight = torch.nn.parameter.Parameter(
            torch.zeros(self.num_inputs, self.num_outputs),
        )
        self.bias = torch.nn.parameter.Parameter(
            torch.zeros(self.num_outputs),
        )

    @staticmethod
    def xaiver_uniform(
        parameter: torch.Tensor, rng: torch.Generator,
        num_inputs: int, num_outputs: int,
        /,
    ) -> None:
        r"""
        Initialize given parameter by Xaiver uniform initialization.

        Args
        ----
        - parameter :
            Parameter data to be initialized.
        - rng :
            Random number generator.
        - num_inputs :
            Number of inputs.
        - num_outputs :
            Number of outputs.

        Returns
        -------
        """
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # /
        # ANNOTATE
        # /
        bound: float

        # Inplace initiailization.
        bound = math.sqrt(6.0 / float(num_inputs + num_outputs))
        parameter.uniform_(
            -bound, bound,
            generator=rng,
        )

    def initialize(
        self,
        rng_cpu: torch.Generator,
        /,
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
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # Initialzie linear layer.
        self.xaiver_uniform(
            self.weight.data, rng_cpu, self.num_inputs, self.num_outputs,
        )
        self.xaiver_uniform(
            self.bias.data, rng_cpu, self.num_inputs, self.num_outputs,
        )

    def forward(
        self,
        node_feat_input: torch.Tensor, adjacency_input: torch.Tensor,
        indices: torch.Tensor,
        /,
    ) -> torch.Tensor:
        r"""
        Forward.

        Args
        ----
        - node_feat_input :
            Inputs of node features.
        - adjacency_input :
            Inputs of adjacency.
        - indices :
            Active indices.

        Returns
        -------
        - outputs :
            Outputs.
        """
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # /
        # YOU SHOULD FILL IN THIS PART
        # /

        # Should compute sigmoid([H, diag(e*A)^-1 * A * H] * W + b)
        e = torch.ones(1, adjacency_input.shape[1], dtype=torch.float32)
        adjacency_input = adjacency_input.type(dtype=torch.float32)
        degree_vec = e.matmul(adjacency_input).reshape(-1)
        # print(torch.mean(degree_vec[indices]))
        # print(torch.max(degree_vec[indices]))
        # print(torch.min(degree_vec[indices]))
        degree_mat_inv = torch.diag(1/degree_vec)
        H_neigh = degree_mat_inv @ adjacency_input @ node_feat_input
        temp = torch.hstack([node_feat_input, H_neigh])
        return torch.sigmoid(temp.matmul(self.weight) + self.bias)
