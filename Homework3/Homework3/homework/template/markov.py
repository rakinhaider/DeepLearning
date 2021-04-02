# All imports
import torch
import math
from typing import Union
from typing import Tuple
from typing import Set


# =============================================================================
# *****************************************************************************
# -----------------------------------------------------------------------------
# < Homework >
# -----------------------------------------------------------------------------
# *****************************************************************************
# =============================================================================


class Markov(
    torch.nn.Module,
    metaclass=type,
):
    r"""
    Markov for language modeling.
    """
    # =========================================================================
    # -------------------------------------------------------------------------
    # =========================================================================
    def __init__(
        self,
        rng_cpu: torch.Generator, rng_gpu: torch.Generator,
        num_words: int, num_internals: int,
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
        - num_words :
            Number of words.
        - num_internals :
            Number of internal neurons.

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
        self.num_words: int
        self.num_internals: int

        # Super call.
        super(Markov, self).__init__()

        # Save necessary attributes.
        self.rng_cpu = rng_cpu
        self.rng_gpu = rng_gpu
        self.num_words = num_words
        self.order = num_internals

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
        # /
        # ANNOTATE
        # /
        self.states: Dict[Tuple[int, ...], Dict[int, int]]
        self.totals: Dict[Tuple[int, ...], int]

        # Initialize chain states.
        self.states = dict()
        self.totals = dict()

    def forward(
        self,
        input: torch.Tensor,
        /,
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
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # /
        # ANNOTATE
        # /
        length: int
        observations: List[int]
        unknowns: Dict[Tuple[int, ...], Set[int]]
        estimations: List[float]
        c: Tuple[int, ...]
        w: int

        # Focus observations as a list.
        length = len(input)
        observations = [0] * self.order + input.tolist()

        # Update only for training.
        if (self.training):
            for i in range(self.order, length+self.order):
                c = tuple(observations[i - self.order:i])
                w = observations[i]
                if c not in self.totals:
                    self.totals[c] = 1
                    self.states[c] = {}
                else:
                    self.totals[c] += 1
                self.states[c][w] = self.states[c].get(w, 0) + 1

            # print(self.totals)
            # print(self.states)
        else:
            pass

        # Count for unknown paddings.
        # /
        # YOU SHOULD FILL IN THIS PART
        # /
        unknowns = dict()
        # all_words = set([i for i in range(self.num_words)])
        for c in self.totals:
            # unknowns[c] = all_words.difference(self.states[c].keys())
            unknowns[c] = self.num_words - len(self.states[c].keys())
        for i in range(self.order, length + self.order):
            c = tuple(observations[i - self.order:i])
            if c not in unknowns:
                # unknowns[c] = all_words
                unknowns[c] = self.num_words

        # Direct indexing to get probability predictions.
        # /
        # YOU SHOULD FILL IN THIS PART
        # /
        estimations = torch.zeros(length, device=input.device)
        for i in range(self.order, length + self.order):
            c = tuple(observations[i - self.order:i])
            w = observations[i]
            ct_total = self.totals.get(c, 0) + 1
            if ct_total == 1:
                ct_w = 0
            else:
                ct_w = self.states[c].get(w, 0)
            if ct_w > 0:
                estimations[i - self.order] = ct_w / ct_total
            else:
                # predictions[i - self.order] = 1 / (ct_total * len(unknowns[c]))
                estimations[i - self.order] = 1 / (ct_total * unknowns[c])

        return estimations

    def clear(
        self,
        /
    ) -> None:
        r"""
        Clear recurrent states.

        Args
        ----

        Returns
        -------
        """
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # Do nothing.
        pass