# All imports.
import abc
from typing import Tuple, List
import torch
import numpy as onp
import math
from sklearn.metrics import balanced_accuracy_score
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


class MNISTClassification(
    torch.nn.Module,
    metaclass=abc.ABCMeta,
):
    r"""
    MNIST Classification framework.
    """
    # /
    # ANNOTATE
    # /
    EVALON: int
    EVALOFF: int
    METRICS: List[str]
    ACCURACY: int
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
        "Acc",
        "Loss",
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
    ACCURACY = 0
    LOSS = 1
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

    def optim(
        self,
        optimizer: torch.optim.Optimizer,
        /,
    ) -> None:
        r"""
        Build special optimizer link for Nesterov.

        Args
        ----
        - optimizer :
            Optimizer.

        Returns
        -------
        """
        # /
        # ANNOTATE
        # /
        self.optimizer: torch.optim.Optimizer

        # Link optimizer.
        self.optimizer = optimizer

    # -------------------------------------------------------------------------
    # < Training >
    # Compute gradient for given arguments.
    # -------------------------------------------------------------------------

    def trainit(
        self,
        /,
        image: torch.Tensor, label: torch.Tensor,
    ) -> None:
        r"""
        Train for gradient.

        Args
        ----
        - image :
            Flattened image vectors.
        - label :
            Target labels.

        Returns
        -------
        """
        # /
        # ANNOTATE
        # /
        scores: torch.Tensor
        loss: torch.Tensor

        # Enable training mode.
        self.train()

        # A special call in this homework.
        self.optimizer.prev()

        if (self.amprec):
            # Enable auto mixed precision mode.
            with torch.cuda.amp.autocast():
                # Forward MLPs.
                scores = self.forward(image)

                # Get loss assuming uniform label distribution.
                loss = torch.nn.functional.cross_entropy(scores, label)

            # Automatically compute gradient with auto mixed precision.
            self.gradscaler.scale(loss).backward()
        else:
            # Forward MLPs.
            scores = self.forward(image)

            # Get loss assuming uniform label distribution.
            loss = torch.nn.functional.cross_entropy(scores, label)

            # Automatically compute gradient.
            loss.backward()

    # -------------------------------------------------------------------------
    # < Evaluating >
    # Evaluate current model for given arguments.
    # -------------------------------------------------------------------------

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
        # /
        # ANNOTATE
        # /
        self.evaluating: int

        # Initialize the evaluation flag.
        self.evaluating = self.EVALOFF

    def evaluatit(
        self,
        /,
        image: torch.Tensor, label: torch.Tensor,
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
        # /
        # ANNOTATE
        # /
        scores: torch.Tensor
        probabilities: torch.Tensor
        predictions: torch.Tensor
        labels: torch.Tensor
        onehots: torch.Tensor

        # Forward 2-layer MLP.
        self.eval()
        scores = self.forward(image)

        # Get probability.
        probabilities = torch.softmax(
            scores,
            dim=1,
        )

        # Get predictions.
        _, predictions = torch.max(
            probabilities,
            dim=1,
        )

        # Get ground truth onehots.
        labels = label
        onehots = torch.eye(
            self.num_labels, self.num_labels,
            dtype=torch.long, device=label.device,
        )
        onehots = onehots[labels]

        # Save output and target for final evaluation.
        self.output_label_buf.extend(predictions.data.tolist())
        self.target_label_buf.extend(labels.data.tolist())
        self.output_proba_buf.extend(probabilities.data.tolist())
        self.target_proba_buf.extend(onehots.data.tolist())

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
        # /
        # ANNOTATE
        # /
        self.output_proba_buf: List[List[float]]
        self.output_label_buf: List[List[int]]
        self.target_proba_buf: List[List[float]]
        self.target_label_buf: List[List[int]]

        # Safety check.
        assert (
            self.evaluating == self.EVALOFF
        ), "[\033[91mError\033[0m]: Evaluation is running."

        # Change flag.
        self.evaluating = self.EVALON

        # Reset buffer.
        self.output_proba_buf = []
        self.output_label_buf = []
        self.target_proba_buf = []
        self.target_label_buf = []

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
        # /
        # ANNOTATE
        # /
        acc: float
        loss: float
        prec_mic: float
        prec_mac: float
        prec_wtd: float
        recl_mic: float
        recl_mac: float
        recl_wtd: float
        f1_mic: float
        f1_mac: float
        f1_wtd: float
        rocauc_mic: float
        rocauc_mac: float
        rocauc_wtd: float
        performance: torch.Tensor
        criterion: float

        # Safety check.
        assert (
            self.evaluating == self.EVALON
        ), "[\033[91mError\033[0m]: Evaluation is not running."

        # Change flag.
        self.evaluating = self.EVALOFF

        # Compute accuracy.
        acc = balanced_accuracy_score(
            y_true=self.target_label_buf, y_pred=self.output_label_buf,
        )

        # Compute loss.
        loss = torch.nn.functional.cross_entropy(
            torch.Tensor(self.output_proba_buf),
            torch.LongTensor(self.target_label_buf),
        ).item()

        # Compute precision.
        prec_mic = precision_score(
            y_true=self.target_label_buf, y_pred=self.output_label_buf,
            average="micro", zero_division=1,
        )
        prec_mac = precision_score(
            y_true=self.target_label_buf, y_pred=self.output_label_buf,
            average="macro", zero_division=1,
        )
        prec_wtd = precision_score(
            y_true=self.target_label_buf, y_pred=self.output_label_buf,
            average="weighted", zero_division=1,
        )

        # Compute recall.
        recl_mic = recall_score(
            y_true=self.target_label_buf, y_pred=self.output_label_buf,
            average="micro", zero_division=1,
        )
        recl_mac = recall_score(
            y_true=self.target_label_buf, y_pred=self.output_label_buf,
            average="macro", zero_division=1,
        )
        recl_wtd = recall_score(
            y_true=self.target_label_buf, y_pred=self.output_label_buf,
            average="weighted", zero_division=1,
        )

        # Compute F1.
        f1_mic = f1_score(
            y_true=self.target_label_buf, y_pred=self.output_label_buf,
            average="micro",
        )
        f1_mac = f1_score(
            y_true=self.target_label_buf, y_pred=self.output_label_buf,
            average="macro",
        )
        f1_wtd = f1_score(
            y_true=self.target_label_buf, y_pred=self.output_label_buf,
            average="weighted",
        )

        # Compute ROCAUC.
        rocauc_mic = roc_auc_score(
            y_true=self.target_proba_buf, y_score=self.output_proba_buf,
            average="micro",
        )
        rocauc_mac = roc_auc_score(
            y_true=self.target_proba_buf, y_score=self.output_proba_buf,
            average="macro",
        )
        rocauc_wtd = roc_auc_score(
            y_true=self.target_proba_buf, y_score=self.output_proba_buf,
            average="weighted",
        )

        # Get performance and criterion.
        performance = torch.Tensor([
            float(acc), float(loss),
            float(prec_mic), float(prec_mac), float(prec_wtd),
            float(recl_mic), float(recl_mac), float(recl_wtd),
            float(f1_mic), float(f1_mac), float(f1_wtd),
            float(rocauc_mic), float(rocauc_mac), float(rocauc_wtd),
        ])
        criterion = performance[self.criterion].item()
        return criterion, performance


class GInvariantMLP(
    MNISTClassification,
    metaclass=type,
):
    r"""
    G-Invariant MLP.
    """
    def __init__(
        self,
        /,
        rng_cpu: torch.Generator, rng_gpu: torch.Generator,
        gradscaler: torch.cuda.amp.grad_scaler.GradScaler,
        *,
        num_inputs: int, num_labels: int, num_internals: List[int],
        criterion: int,
        eigenvectors: onp.ndarray, ginvariant: bool,
        ginvariant_linear: type, amprec: bool,
    ) -> None:
        r"""
        Initialize the class.

        Args
        ----
        - rng_cpu :
            Random number generator for CPU.
        - rng_gpu :
            Random number generator for GPU.
        - gradscaler :
            Gradient scaler for automatically mixed precision.
        - num_inputs :
            Number of inputs.
        - num_labels :
            Number of labels.
        - num_internals :
            Number of internal neurons per hidden layer.
        - criterion :
            Criterion used to evaluate model.
        - eigenvectors :
            Non-trival eigenvectors of invariant subspace.
        - ginvariant :
            Signal to control G-invariant usage.
        - ginvariant_linear :
            G-invariant linear layer.
        - amprec :
            Use automatically mixed precision.

        Returns
        -------
        """
        # /
        # ANNOTATE
        # /
        self.rng_cpu: torch.Generator
        self.rng_gpu: torch.Generator
        self.gradscaler: torch.cuda.amp.grad_scaler.GradScaler
        self.eigenvectors: torch.Tensor
        self.linear0: torch.nn.Linear
        linears: List[torch.nn.Linear]
        self.linears: torch.nn.ModuleList
        self.criterion: int
        self.amprec: bool
        n_ins: int
        n_outs: int

        # Super call.
        super(GInvariantMLP, self).__init__()

        # Save necessary attributes.
        self.rng_cpu = rng_cpu
        self.rng_gpu = rng_gpu
        self.gradscaler = gradscaler
        self.num_labels = num_labels
        self.criterion = criterion
        self.amprec = amprec
        self.register_buffer(
            "eigenvectors",
            torch.from_numpy(eigenvectors).to(torch.get_default_dtype())
        )

        # Create the first linear layer (possibly be G-invariant).
        if (ginvariant):
            self.linear0 = ginvariant_linear(
                self.rng_cpu, self.rng_gpu,
                num_inputs=num_inputs, num_outputs=num_internals[0],
                eigenvectors=self.eigenvectors,
            )
        else:
            self.linear0 = torch.nn.Linear(num_inputs, num_internals[0])

        # Create following linear layers.
        linears = []
        for n_ins, n_outs in zip(
            num_internals, num_internals[1:] + [num_labels],
        ):
            linears.append(torch.nn.Linear(n_ins, n_outs))
        self.linears = torch.nn.ModuleList(linears)

        # Set evaluation flag.
        self.evaluate_reset()

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
        linear: torch.nn.Linear
        fan_in: int
        fan_out: int

        # Use Xaiver uniform.
        for linear in [self.linear0] + list(self.linears):
            fan_out, fan_in = linear.weight.data.size()
            bound = math.sqrt(6.0 / float(fan_in + fan_out))
            linear.weight.data.uniform_(
                -bound, bound,
                generator=rng_cpu,
            )
            linear.bias.data.uniform_(
                -bound, bound,
                generator=rng_cpu,
            )

    def forward(
        self,
        /,
        image: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Forward.

        Args
        ----
        - image :
            Flattened image vectors.

        Returns
        -------
        - scores :
            Label scores.
        """
        # /
        # ANNOTATE
        # /
        l: int
        internal: torch.Tensor
        scores: torch.Tensor

        # Forward 2-layer MLP.
        internal = image
        for l, linear in enumerate([self.linear0] + list(self.linears)):
            if (l == 0):
                pass
            else:
                internal = torch.relu(internal)
            internal = linear.forward(internal)
        scores = internal
        return scores


class StackCNN(
    MNISTClassification,
    metaclass=type,
):
    r"""
    Stack CNN.
    """
    def __init__(
        self,
        /,
        rng_cpu: torch.Generator, rng_gpu: torch.Generator,
        gradscaler: torch.cuda.amp.grad_scaler.GradScaler,
        *,
        num_input_channels: int, num_output_channels: int,
        num_internal_channels: int,
        conv_kernel: int, conv_stride: int,
        pool_kernel: int, pool_stride: int, padding: int,
        num_labels: int, num_internals: int,
        height: int, width: int,
        criterion: int,
        dual_cnn: torch.nn.Module, amprec: bool,
    ) -> None:
        r"""
        Initialize the class.

        Args
        ----
        - rng_cpu :
            Random number generator for CPU.
        - rng_gpu :
            Random number generator for GPU.
        - gradscaler :
            Gradient scaler for automatically mixed precision.
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
        - num_labels :
            Number of labels.
        - num_internals :
            Number of internal neurons per hidden layer.
        - height :
            Image height.
        - width :
            Image width.
        - criterion :
            Criterion used to evaluate model.
        - dual_cnn :
            Dual CNN layer.
        - amprec :
            Use automatically mixed precision.

        Returns
        -------
        """
        # /
        # ANNOTATE
        # /
        self.rng_cpu: torch.Generator
        self.rng_gpu: torch.Generator
        self.gradscaler: torch.cuda.amp.grad_scaler.GradScaler
        self.cnn0: torch.nn.Module
        linears: List[torch.nn.Linear]
        self.linears: torch.nn.ModuleList
        self.criterion: int
        self.amprec: bool
        self.channel: int
        self.height: int
        self.width: int
        self.num_flattens: int
        channel: int
        n_ins: int
        n_outs: int

        # Super call.
        super(StackCNN, self).__init__()

        # Save necessary attributes.
        self.rng_cpu = rng_cpu
        self.rng_gpu = rng_gpu
        self.gradscaler = gradscaler
        self.num_labels = num_labels
        self.criterion = criterion
        self.amprec = amprec
        self.channel = num_input_channels
        self.height = height
        self.width = width

        # Create the first CNN layer.
        self.cnn0 = dual_cnn(
            self.rng_cpu, self.rng_gpu,
            num_input_channels=num_input_channels,
            num_output_channels=num_output_channels,
            num_internal_channels=num_internal_channels,
            conv_kernel=conv_kernel, conv_stride=conv_stride,
            pool_kernel=pool_kernel, pool_stride=pool_stride,
            padding=padding,
        )

        # Pesudo forward for flatten size.
        _, channel, height, width = self.cnn0.forward(
            torch.zeros(1, num_input_channels, height, width),
        ).size()
        self.num_flattens = channel * height * width

        # Create following linear layers.
        linears = []
        for n_ins, n_outs in zip(
            [self.num_flattens, num_internals], [num_internals, num_labels],
        ):
            linears.append(torch.nn.Linear(n_ins, n_outs))
        self.linears = torch.nn.ModuleList(linears)

        # Set evaluation flag.
        self.evaluate_reset()

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
        linear: torch.nn.Linear
        fan_in: int
        fan_out: int

        # Initialize CNN.
        self.cnn0.initialize(rng_cpu)

        # Use Xaiver uniform.
        for linear in self.linears:
            fan_out, fan_in = linear.weight.data.size()
            bound = math.sqrt(6.0 / float(fan_in + fan_out))
            linear.weight.data.uniform_(
                -bound, bound,
                generator=rng_cpu,
            )
            linear.bias.data.uniform_(
                -bound, bound,
                generator=rng_cpu,
            )

    def forward(
        self,
        /,
        image: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Forward.

        Args
        ----
        - image :
            Flattened image vectors.

        Returns
        -------
        - scores :
            Label scores.
        """
        # /
        # ANNOTATE
        # /
        l: int
        internal: torch.Tensor
        scores: torch.Tensor

        # Forward 2-layer CNN and linears.
        internal = image.view(-1, self.channel, self.height, self.width)
        internal = self.cnn0.forward(internal)
        internal = internal.view(-1, self.num_flattens)
        for l, linear in enumerate(self.linears):
            if (l == 0):
                pass
            else:
                internal = torch.relu(internal)
            internal = linear.forward(internal)
        scores = internal
        return scores