# All imports
import torch
import math
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import numpy as np
import random


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
        # TODO: simply implement GNN with sparse adjacency list
        #  create a n*n zero matrix
        #  use index_add_ to aggregate all the neighbors
        #  then simply multiply with weight matrix and add bias.
        dtype = node_feat_input.dtype
        device = node_feat_input.device
        n = node_feat_input.shape[0]
        f = node_feat_input.shape[1]
        sum_neighbors = torch.zeros((n, f), dtype=dtype, device=device)
        sum_neighbors.index_add_(0, adjacency_input[:, 0],
                                 node_feat_input[adjacency_input[:, 1]])

        degree = torch.zeros(n, dtype=dtype, device=device)
        ones = torch.ones_like(adjacency_input[:, 0],
                               dtype=dtype, device=device)
        degree.index_add_(0, adjacency_input[:, 0], ones)
        H_neigh = torch.diag(1 / degree) @ sum_neighbors
        temp = torch.hstack([node_feat_input, H_neigh]) @ self.weight
        temp = torch.sigmoid(temp + self.bias)
        return temp


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
        n = node_feat_input.shape[0]
        f = node_feat_input.shape[1]
        dtype = node_feat_input.dtype
        device = node_feat_input.device
        n_neighbors = torch.zeros(n, device=device, dtype=torch.int32)
        h_neigh = torch.zeros(n, f, device=device)
        if self.training:
            n_perms = 1
        else:
            n_perms = self.num_perms

        neighbor_indices = []
        lengths = []
        for i in range(n):
            edge_indices = (adjacency_input[:, 0] == i)
            neighbor_indices.append(adjacency_input[edge_indices][:, 1])
            n_neighbors[i] = len(neighbor_indices[-1])
            if n_neighbors[i] >= self.kary:
                lengths.append(self.kary)
            else:
                lengths.append(n_neighbors[i].item())

        for p in range(n_perms):
            neighbors_feats = []
            for i in range(n):
                nv = n_neighbors[i].item()
                if nv >= self.kary:
                    sel_indices = random.sample(range(nv), self.kary)
                    sel_indices = torch.tensor(sel_indices, device=device)
                else:
                    sel_indices = random.sample(range(nv), nv)
                    sel_indices = torch.tensor(sel_indices, device=device)
                selected = neighbor_indices[i][sel_indices]
                print(selected)
                neighbors_feats.append(node_feat_input[selected])

            padded_neighbors = pad_sequence(neighbors_feats, False,
                                            padding_value=-1)

            packed_neighbors = pack_padded_sequence(padded_neighbors,
                                                    lengths, False,
                                                    enforce_sorted=False)

            output, (h_n, c_n) = self.lstm(packed_neighbors)
            c_n = c_n.reshape(n, f)
            h_neigh.add_(c_n)

        h_neigh = h_neigh * (1 / n_perms)
        temp = torch.hstack([node_feat_input, h_neigh]) @ self.weight
        temp += self.bias

        return torch.sigmoid(temp)
