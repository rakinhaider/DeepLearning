# All imports
import torch
import math
from typing import cast
from typing import Union
from typing import Tuple


# =============================================================================
# *****************************************************************************
# -----------------------------------------------------------------------------
# < Homework >
# -----------------------------------------------------------------------------
# *****************************************************************************
# =============================================================================


def t_bptt_preprocess(
    length: int, input: torch.Tensor, target: torch.Tensor,
    batch_size: int, truncate: int,
) -> Tuple[int, torch.Tensor, torch.Tensor, int]:
    r"""
    Preprocess given data for truncated BPTT.

    Args
    ----
    - length :
        Number of words $N$ in the data.
    - input :
        A sequence of input words.
        It should be of length $N - 1$ since we predict one-step future.
    - target :
        A sequence of target words.
        It should be of length $N - 1$ since we predict one-step future.
    - batch_size :
        Batch size.
    - truncate :
        Truncated BPTT chunk size.

    Returns
    -------
    - length :
        Number of words $M$ in the processed data.
        It should be a multiplicaton of batch size and sub sequence length of
        each batch.
    - input :
        Processed input data.
        It should be of size "batch size, sub sequence length of each batch".
        It is possible that subsequences end by an incomplete sentence.
    - target :
        Processed target data.
        It should be of size "batch size, sub sequence length of each batch".
        It is possible that subsequences end by an incomplete sentence.
    - num_chunks :
        Number of BPTT chunks.
        It should be "ceil(sub sequence length of each batch / truncation
        size)".

    Use `input = input.contiguous()` and `target = target.contiguous()` to
    ensure generated data memory is contiguous.
    """
    # /
    # YOU SHOULD FILL IN THIS PART
    # /
    raise NotImplementedError


class LSTM(
    torch.nn.Module,
    metaclass=type,
):
    r"""
    LSTM for language modeling.
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
        self.word2vec: torch.nn.Module
        self.lstm: torch.nn.Module
        self.decode: torch.nn.Module

        # Super call.
        super(LSTM, self).__init__()

        # Save necessary attributes.
        self.rng_cpu = rng_cpu
        self.rng_gpu = rng_gpu
        self.num_words = num_words
        self.num_internals = num_internals
        self.word2vec = torch.nn.Embedding(self.num_words, self.num_internals)
        self.lstm = torch.nn.LSTM(
            self.num_internals, self.num_internals, 1,
            batch_first=False, dropout=0, bidirectional=False,
        )
        self.decode = torch.nn.Linear(self.num_internals, self.num_words)

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
        # Initialzie word2vec layer.
        self.xaiver_uniform(
            self.word2vec.weight.data, rng_cpu,
            self.num_words, self.num_internals,
        )

        # Initialize LSTM layer.
        self.xaiver_uniform(
            self.lstm.weight_ih_l0.data, rng_cpu,
            self.num_internals, self.num_internals,
        )
        self.xaiver_uniform(
            self.lstm.weight_hh_l0.data, rng_cpu,
            self.num_internals, self.num_internals,
        )
        self.xaiver_uniform(
            self.lstm.bias_ih_l0.data, rng_cpu,
            self.num_internals, self.num_internals,
        )
        self.xaiver_uniform(
            self.lstm.bias_hh_l0.data, rng_cpu,
            self.num_internals, self.num_internals,
        )

        # Initialzie linear layer.
        self.xaiver_uniform(
            self.decode.weight.data, rng_cpu,
            self.num_internals, self.num_words,
        )
        self.xaiver_uniform(
            self.decode.bias.data, rng_cpu,
            self.num_internals, self.num_words,
        )

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
        word_feats: torch.Tensor
        output: torch.Tensor

        # Translate word to word features (vectors).
        word_feats = self.word2vec(input)

        # Feed into LSTM model.
        word_feats, self.recurrent = self.lstm(word_feats, self.recurrent)
        self.recurrent[0].detach()
        self.recurrent[1].detach()
        self.recurrent = (
            self.recurrent[0].data,
            self.recurrent[1].data,
        )
        cast(Tuple[torch.Tensor, torch.Tensor], word_feats)

        # Feed into decoder.
        output = self.decode.forward(word_feats)
        return output

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
        # /
        # ANNOTATE
        # /
        self.recurrent: Union[None, Tuple[torch.Tensor, torch.Tensor]]

        # Initialize by null.
        self.recurrent = None