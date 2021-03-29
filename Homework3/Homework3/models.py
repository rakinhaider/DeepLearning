# All imports.
import abc
import torch
import math
import numpy as onp
from typing import Tuple, List, Dict
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


# =============================================================================
# *****************************************************************************
# -----------------------------------------------------------------------------
# < Models >
# -----------------------------------------------------------------------------
# *****************************************************************************
# =============================================================================


class MulticlassClassification(
    torch.nn.Module,
    metaclass=abc.ABCMeta,
):
    r"""
    Generic multi-class classification model.
    """
    # =========================================================================
    # -------------------------------------------------------------------------
    # =========================================================================
    # /
    # ANNOTATE
    # /
    EVALON: int
    EVALOFF: int
    METRICS: List[str]
    # -----
    LOSS: int
    ACCURACY: int
    # -----
    PRECISION_MICRO: int
    PRECISION_MACRO: int
    PRECISION_WEIGHT: int
    RECALL_MICRO: int
    RECALL_MACRO: int
    RECALL_WEIGHT: int
    F1_MICRO: int
    F1_MACRO: int
    F1_WEIGHT: int
    ROCAUC_MICRO: int
    ROCAUC_MACRO: int
    ROCAUC_WEIGHT: int

    # Define evaluation flag.
    EVALON = 1
    EVALOFF = -1

    # Define metric index.
    METRICS = [
        "Loss",
        "Acc",
        "Prec Mic",
        "Prec Mac",
        "Prec Wtd",
        "Recl Mic",
        "Recl Mac",
        "Recl Wtd",
        "F1 Mic",
        "F1 Mac",
        "F1 Wtd",
        "ROCAUC Mic",
        "ROCAUC Mac",
        "ROCAUC Wtd",
    ]
    LOSS = 0
    ACCURACY = 1
    PRECISION_MICRO = 2
    PRECISION_MACRO = 3
    PRECISION_WEIGHT = 4
    RECALL_MICRO = 5
    RECALL_MACRO = 6
    RECALL_WEIGHT = 7
    F1_MICRO = 8
    F1_MACRO = 9
    F1_WEIGHT = 10
    ROCAUC_MICRO = 11
    ROCAUC_MACRO = 12
    ROCAUC_WEIGHT = 13

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

    @abc.abstractmethod
    def loss_func(
        self,
        prediction: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor],
        /,
    ) -> Tuple[torch.Tensor, int]:
        r"""
        Compute loss function.

        Args
        ----
        - prediction :
            Prediction made by the model.
        - batch :
            Minibatch.

        Returns
        -------
        - loss :
            Computed loss tensor.
        - bsz :
            Batch size w.r.t. computed loss tensor.
        """
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # /
        # VIRTUAL
        # /
        ...

    @abc.abstractmethod
    def predict(
        self,
        batch: Dict[str, torch.Tensor],
        /,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""
        Predict on given batch.

        Args
        ----
        - batch :
            Training minibatch.

        Returns
        -------
        - pred_local :
            Prediction made by the model for local minibatch level usage.
            It is assumed to be scores for a classification task.
        - pred_global :
            Prediction made by the model for global epoch level usage.
            It is assumed to be scores and true labels for a classification
            task.
        """
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # /
        # VIRTUAL
        # /
        ...

    # =========================================================================
    # -------------------------------------------------------------------------
    # Compute gradient for given arguments.
    # -------------------------------------------------------------------------
    # =========================================================================

    def trainit(
        self,
        batch: Dict[str, torch.Tensor],
        /,
    ) -> None:
        r"""
        Train for gradient.

        Args
        ----
        - batch :
            Batch.

        Returns
        -------
        """
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # /
        # ANNOTATE
        # /
        pred_local: torch.Tensor
        loss: torch.Tensor

        # Feedforward.
        self.train()
        pred_local, _ = self.predict(batch)
        loss, _ = self.loss_func(pred_local, batch)

        # Backpropagate.
        loss.backward()

    # =========================================================================
    # -------------------------------------------------------------------------
    # Evaluate current model for given arguments.
    # -------------------------------------------------------------------------
    # =========================================================================

    def evaluate_reset(
        self,
        /,
    ) -> None:
        r"""
        Reset evaluation.

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
        self.evaluating: int

        # Initialize the evaluation flag.
        self.evaluating = self.EVALOFF

    def evaluatit(
        self,
        batch: Dict[str, torch.Tensor],
        /,
    ) -> None:
        r"""
        Evaluate for performance.

        Args
        ----
        - image :
            Flattened image vectors.
        - label :
            Target labels.

        Returns
        -------
        """
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # /
        # ANNOTATE
        # /
        pred_local: torch.Tensor
        pred_global: Tuple[torch.Tensor, torch.Tensor]
        loss: torch.Tensor
        bsz: int

        # Feedforward.
        self.eval()
        pred_local, pred_global = self.predict(batch)
        loss, bsz = self.loss_func(pred_local, batch)

        # Save output and target for final evaluation.
        # Keep it in device memory to avoid data transfer.
        self.buf_local.append(((loss * bsz).view(1), torch.Tensor([bsz])))
        self.buf_global.append(pred_global)

    def evaluaton(
        self,
        /,
    ) -> None:
        r"""
        Start a new evaluation.

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
        self.buf_local: List[Tuple[torch.Tensor, torch.Tensor]]
        self.buf_global: List[Tuple[torch.Tensor, torch.Tensor]]

        # Safety check.
        if (self.evaluating == self.EVALOFF):
            pass
        else:
            print("[\033[91mError\033[0m]: Evaluation is running.")
            raise RuntimeError

        # Change flag.
        self.evaluating = self.EVALON

        # Reset evaluation buffer.
        self.buf_local = []
        self.buf_global = []

    def evaluatoff(
        self,
        /,
    ) -> Tuple[float, torch.Tensor]:
        r"""
        Terminate current evaluation and return performance.

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
        val: torch.Tensor
        cnt: torch.Tensor
        buf_val: List[torch.Tensor]
        buf_cnt: List[torch.Tensor]
        # -----
        tensor_score: torch.Tensor
        tensor_label: torch.Tensor
        buf_score: List[torch.Tensor]
        buf_label: List[torch.Tensor]
        scores: torch.Tensor
        labels: torch.Tensor
        predicts: torch.Tensor
        array_score: onp.ndarray
        array_label: onp.ndarray
        array_predict: onp.ndarray
        # -----
        loss: float
        acc: float
        performance: torch.Tensor
        criterion: float

        # Safety check.
        if (self.evaluating == self.EVALON):
            pass
        else:
            print("[\033[91mError\033[0m]: Evaluation is not running.")
            raise RuntimeError

        # Change flag.
        self.evaluating = self.EVALOFF

        # Compute loss.
        buf_val, buf_cnt = [], []
        for val, cnt in self.buf_local:
            buf_val.append(val)
            buf_cnt.append(cnt)
        loss = sum(val).item() / sum(cnt).item()

        # Collect global scores and labels.
        buf_score, buf_label = [], []
        for tensor_score, tensor_label in self.buf_global:
            buf_score.append(tensor_score)
            buf_label.append(tensor_label)
        scores = torch.cat(
            buf_score,
            dim=0
        )
        labels = torch.cat(
            buf_label,
            dim=0
        )

        # Get predicted labels.
        _, predicts = torch.max(
            scores,
            dim=1,
        )

        # Transfer to evaluation device.
        array_score = scores.tolist()
        array_label = labels.tolist()
        array_predict = predicts.tolist()

        # Compute accuracy.
        acc = accuracy_score(
            y_true=array_label, y_pred=array_predict,
        )

        # Get performance and criterion.
        performance = torch.Tensor([float(loss), float(acc)])
        criterion = performance[self.criterion].item()
        return criterion, performance


class MulticlassNodeClassification(
    MulticlassClassification,
    metaclass=type,
):
    r"""
    Multi-class node classification model.
    """
    # =========================================================================
    # -------------------------------------------------------------------------
    # =========================================================================
    def __init__(
        self,
        rng_cpu: torch.Generator, rng_gpu: torch.Generator,
        /,
        *,
        num_inputs: int, num_internals: int, num_outputs: int,
        num_graph_layers: int, graph_kernel: type,
        criterion: int,
        janossy_kwargs: Dict[str, int]
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
            Number of input neurons.
        - num_internals :
            Number of internal neurons.
        - num_outputs :
            Number of input neurons.
        - num_graph_layers :
            Number of graph convolution layers.
        - graph_kernel :
            Graph convolution kernel.
        - criterion :
            Criterion used to evaluate model.
        - janossy_kwargs :
            Janossy Pooling keyword arguments.

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
        self.num_internals: int
        self.num_outputs: int
        self.num_graph_layers: int
        self.criterion: int
        # -----
        layer_buf: List[torch.nn.Module]
        l: int
        n_ins: int
        n_outs: int
        self.graph_convs: torch.nn.ModuleList
        self.linears: torch.nn.ModuleList

        # Super call.
        super(MulticlassNodeClassification, self).__init__()

        # Save necessary attributes.
        self.rng_cpu = rng_cpu
        self.rng_gpu = rng_gpu
        self.num_inputs = num_inputs
        self.num_internals = num_internals
        self.num_outputs = num_outputs
        self.num_graph_layers = num_graph_layers
        self.criterion = criterion

        # Create graph kernels.
        layer_buf = []
        for l in range(self.num_graph_layers):
            # Get input and output size of each graph kernel.
            if (l == 0):
                n_ins = self.num_inputs
            else:
                n_ins = self.num_internals
            n_outs = self.num_internals

            # Create graph kernel.
            if (len(janossy_kwargs) == 0):
                layer_buf.append(
                    graph_kernel(self.rng_cpu, self.rng_gpu, n_ins, n_outs),
                )
            else:
                layer_buf.append(
                    graph_kernel(
                        self.rng_cpu, self.rng_gpu, n_ins, n_outs,
                        **janossy_kwargs,
                    ),
                )
        self.graph_convs = torch.nn.ModuleList(layer_buf)

        # Create final linear layers.
        layer_buf = []
        for l in range(2):
            # Get input and output size of each linear layer.
            n_ins = self.num_internals
            if (l == 1):
                n_outs = self.num_outputs
            else:
                n_outs = self.num_internals

            # Create linear layer.
            layer_buf.append(torch.nn.Linear(n_ins, n_outs))
        self.linears = torch.nn.ModuleList(layer_buf)

        # Set evaluation flag.
        self.evaluate_reset()

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
        l: int
        n_ins: int
        n_outs: int

        # initialize graph kernels.
        for l in range(self.num_graph_layers):
            self.graph_convs[l].initialize(rng_cpu)

        # Initialize final linear layers.
        for l in range(2):
            # Get input and output size of each linear layer.
            n_ins = self.num_internals
            if (l == 1):
                n_outs = self.num_outputs
            else:
                n_outs = self.num_internals

            # Initialzie linear layer.
            self.xaiver_uniform(
                self.linears[l].weight.data, rng_cpu, n_ins, n_outs,
            )
            self.xaiver_uniform(
                self.linears[l].bias.data, rng_cpu, n_ins, n_outs,
            )

    def loss_func(
        self,
        prediction: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor],
        /,
    ) -> Tuple[torch.Tensor, int]:
        r"""
        Compute loss function.

        Args
        ----
        - prediction :
            Prediction made by the model.
        - batch :
            Minibatch.

        Returns
        -------
        - loss :
            Computed loss tensor.
        - bsz :
            Batch size w.r.t. computed loss tensor.
        """
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # /
        # ANNOTATE
        # /
        indices: torch.Tensor

        # Get focusing indices.
        indices = batch["indices"]

        # Compute loss only on given indices.
        return (
            torch.nn.functional.cross_entropy(
                prediction[indices], batch["node_target"][indices],
            ), 1,
        )

    def predict(
        self,
        batch: Dict[str, torch.Tensor],
        /,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""
        Predict on given batch.

        Args
        ----
        - batch :
            Training minibatch.

        Returns
        -------
        - pred_local :
            Prediction made by the model for local minibatch level usage.
            It is assumed to be scores for a classification task.
        - pred_global :
            Prediction made by the model for global epoch level usage.
            It is assumed to be scores and true labels for a classification
            task.
        """
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # /
        # ANNOTATE
        # /
        indices: torch.Tensor
        scores: torch.Tensor
        labels: torch.Tensor

        # Get focusing indices.
        indices = batch["indices"]

        # Get score directly.
        scores = self.forward(batch)
        return scores, (scores[indices], batch["node_target"][indices])

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        /,
    ) -> torch.Tensor:
        r"""
        Feedforward.

        Args
        ----
        - batch :
            Minibatch.

        Returns
        -------
        - scores :
            Scores of all labels to be predicted.
        """
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # /
        # ANNOTATE
        # /
        internal: torch.Tensor
        self.saved_graph_embedding: torch.Tensor
        l: int

        # Encode graph representations.
        internal = self.encode_graph(
            batch["node_feat"], batch["adjacency"], batch["indices"],
        )
        self.saved_graph_embedding = internal

        # Feedforward final linear layers.
        for l in range(2):
            # Pass one layer.
            internal = self.linears[l](internal)

            # Apply activation except for the last layer.
            if (l == 1):
                pass
            else:
                internal = torch.tanh(internal)
        return internal

    def encode_graph(
        self,
        node_feat: torch.Tensor, adjacency: torch.Tensor,
        indices: torch.Tensor,
        /,
    ) -> torch.Tensor:
        r"""
        Encode graph representations.

        Args
        ----
        - node_feat :
            Node feature.
        - adjacency :
            Adjacency.
        - indices :
            Active indices.

        Returns
        -------
        - node_repr :
            Node representation.
        """
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # /
        # ANNOTATE
        # /
        internal: torch.Tensor
        l: int

        # Feedforward graph kernels.
        internal = node_feat
        for l in range(self.num_graph_layers):
            # Pass one layer.
            internal = self.graph_convs[l](internal, adjacency, indices)

            # Apply activation except for the last layer.
            if (l == self.num_graph_layers - 1):
                pass
            else:
                internal = torch.tanh(internal)
        return internal


class MulticlassWordClassification(
    MulticlassClassification,
    metaclass=type,
):
    r"""
    Multi-class word classification model.
    """
    # =========================================================================
    # -------------------------------------------------------------------------
    # =========================================================================
    def __init__(
        self,
        rng_cpu: torch.Generator, rng_gpu: torch.Generator,
        /,
        *,
        num_words: int, num_internals: int,
        lm_kernel: type, noprop: bool,
        criterion: int,
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
            Number of input neurons.
        - num_internals :
            Number of internal neurons.
        - lm_kernel :
            Languange modeling kernel.
        - noprop :
            No backpropagation.
        - criterion :
            Criterion used to evaluate model.

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
        self.noprop: bool
        self.criterion: int

        # Super call.
        super(MulticlassWordClassification, self).__init__()

        # Save necessary attributes.
        self.rng_cpu = rng_cpu
        self.rng_gpu = rng_gpu
        self.num_words = num_words
        self.num_internals = num_internals
        self.noprop = noprop
        self.criterion = criterion

        # Create languange modeling kernel.
        self.lm = lm_kernel(
            self.rng_cpu, self.rng_gpu, self.num_words, self.num_internals,
        )

        # Set evaluation flag.
        self.evaluate_reset()

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
        # Initialize language modeling kernel.
        self.lm.initialize(self.rng_cpu)

    def loss_func(
        self,
        prediction: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor],
        /,
    ) -> Tuple[torch.Tensor, int]:
        r"""
        Compute loss function.

        Args
        ----
        - prediction :
            Prediction made by the model.
        - batch :
            Minibatch.

        Returns
        -------
        - loss :
            Computed loss tensor.
        - bsz :
            Batch size w.r.t. computed loss tensor.
        """
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # /
        # ANNOTATE
        # /
        target: torch.Tensor
        batch_size: int
        chunk_size: int

        # Get target labels.
        target = batch["target"]

        # Dected Markov and LSTM.
        if (self.noprop):
            return torch.exp(torch.mean(-torch.log(prediction))), 1
        else:
            # Reshape BPTT batch as common batch.
            chunk_size, batch_size = target.size()
            prediction = prediction.view(
                chunk_size * batch_size, self.num_words,
            )
            target = target.view(chunk_size * batch_size)

            # Compute perplexity as exponential of cross-entropy.
            return torch.exp(
                torch.nn.functional.cross_entropy(prediction, target),
            ), chunk_size

    def predict(
        self,
        batch: Dict[str, torch.Tensor],
        /,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""
        Predict on given batch.

        Args
        ----
        - batch :
            Training minibatch.

        Returns
        -------
        - pred_local :
            Prediction made by the model for local minibatch level usage.
            It is assumed to be scores for a classification task.
        - pred_global :
            Prediction made by the model for global epoch level usage.
            It is assumed to be scores and true labels for a classification
            task.
        """
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # /
        # ANNOTATE
        # /
        scores: torch.Tensor

        # Get score first.
        scores = self.forward(batch)
        return scores, (torch.Tensor([[1]]), torch.LongTensor([0]))

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        /,
    ) -> torch.Tensor:
        r"""
        Feedforward.

        Args
        ----
        - batch :
            Minibatch.

        Returns
        -------
        - scores :
            Scores of all labels to be predicted.
        """
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # Use the language model directly.
        return self.lm(batch["input"])

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
        # Only language model can be recurrent.
        self.lm.clear()

    # =========================================================================
    # -------------------------------------------------------------------------
    # Compute gradient for given arguments.
    # -------------------------------------------------------------------------
    # =========================================================================

    def trainit(
        self,
        batch: Dict[str, torch.Tensor],
        /,
    ) -> None:
        r"""
        Train for gradient.

        Args
        ----
        - batch :
            Batch.

        Returns
        -------
        """
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # /
        # ANNOTATE
        # /
        pred_local: torch.Tensor
        loss: torch.Tensor

        # Feedforward.
        self.train()
        pred_local, _ = self.predict(batch)
        loss, _ = self.loss_func(pred_local, batch)

        # Backpropagate.
        if (self.noprop):
            pass
        else:
            loss.backward()