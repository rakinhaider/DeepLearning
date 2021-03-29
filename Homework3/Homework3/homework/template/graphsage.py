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


class SparseGCN(
    torch.nn.Module,
    metaclass=type,
):
    r"""
    GCN for sparse adjacency input.
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
        super(SparseGCN, self).__init__()

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
        raise NotImplementedError


class SparseJanossy(
    torch.nn.Module,
    metaclass=type,
):
    r"""
    Janossy Pooling for sparse adjacency input.
    """
    # =========================================================================
    # -------------------------------------------------------------------------
    # =========================================================================
    def __init__(
        self,
        rng_cpu: torch.Generator, rng_gpu: torch.Generator,
        num_inputs: int, num_outputs: int,
        /,
        *,
        kary: int, num_perms: int,
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
        - kary :
            K-ary for Janossy Pooling.
        - num_perms :
            Number of permutations to be sampled in test stage.

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
        self.kary: int
        self.num_perms: int

        # Super call.
        super(SparseJanossy, self).__init__()

        # Save necessary attributes.
        self.rng_cpu = rng_cpu
        self.rng_gpu = rng_gpu
        self.num_inputs = num_inputs * 2
        self.num_outputs = num_outputs
        self.kary = kary
        self.num_perms = num_perms
        self.lstm = torch.nn.LSTM(num_inputs, num_inputs)
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

        # Initialize LSTM layer.
        self.xaiver_uniform(
            self.lstm.weight_ih_l0.data, rng_cpu,
            self.num_inputs // 2, self.num_inputs // 2,
        )
        self.xaiver_uniform(
            self.lstm.weight_hh_l0.data, rng_cpu,
            self.num_inputs // 2, self.num_inputs // 2,
        )
        self.xaiver_uniform(
            self.lstm.bias_ih_l0.data, rng_cpu,
            self.num_inputs // 2, self.num_inputs // 2,
        )
        self.xaiver_uniform(
            self.lstm.bias_hh_l0.data, rng_cpu,
            self.num_inputs // 2, self.num_inputs // 2,
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
        raise NotImplementedError