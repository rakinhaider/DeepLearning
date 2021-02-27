# All imports
from typing import Any
from typing import Tuple, List, Dict
from typing import ClassVar
from typing import TextIO
import torch
import importlib
import os
import numpy as onp
import copy
import argparse
import time
import math
from datasets import MNISTDataset
from structures import MNISTPerturbDataStructure
from structures import mnist_dataset_collate
from structures import rotate, flip
from models import MNISTClassification
from models import GInvariantMLP
from models import StackCNN


# =============================================================================
# *****************************************************************************
# -----------------------------------------------------------------------------
# < Main >
# -----------------------------------------------------------------------------
# *****************************************************************************
# =============================================================================


class Main(
    object,
    metaclass=type,
):
    r"""
    Main interface.
    """
    def __init__(
        self,
        /,
    ) -> None:
        r"""
        Initialize the class.

        Args
        ----

        Returns
        -------
        """
        # /
        # ANNOTATE
        # /
        self.folder: str
        self.random_seed: int
        self.normalize: bool
        self.shuffle: bool
        self.batch_size: int
        self.num_workers: int
        self.num_internals: List[int]
        self.kernel: int
        self.stride: int
        self.ginvariant: bool
        self.cnn: bool
        self.lr: float
        self.wd: float
        self.num_epochs: int
        self.device: str
        self.sbatch: bool
        self.rng_cpu: torch.Generator
        self.rng_gpu: torch.Generator

        # Create training console arguments.
        self.console = argparse.ArgumentParser(
            description="Homework 2",
        )
        self.console.add_argument(
            "--sbatch",
            action="store_true",
            help="Submit by `sbatch`.",
        )
        self.console.add_argument(
            "--student",
            type=str, nargs=1, required=False,
            help="Student PUID.",
            default=["template"],
        )
        self.console.add_argument(
            "--mnist",
            type=str, nargs=1, required=False,
            help="Path to the MNIST data directory.",
            default=["../Data"],
        )
        self.console.add_argument(
            "--num-samples",
            type=int, nargs=1, required=False,
            help="Number of MNIST training samples to use.",
            default=[-1],
        )
        self.console.add_argument(
            "--random-seed",
            type=int, nargs=1, required=False,
            help="Random seed. It is also used as training-validation" \
                 " split index.",
            default=[0],
        )
        self.console.add_argument(
            "--shuffle-label",
            action="store_true",
            help="Shuffle training label data.",
        )
        self.console.add_argument(
            "--batch-size",
            type=int, nargs=1, required=False,
            help="Batch size.",
            default=[300],
        )
        self.console.add_argument(
            "--num-workers",
            type=int, nargs=1, required=False,
            help="Number of batch sampling processes.",
            default=[4],
        )
        self.console.add_argument(
            "--kernel",
            type=int, nargs=1, required=False,
            help="Size of square kernel (filter).",
            default=[5],
        )
        self.console.add_argument(
            "--stride",
            type=int, nargs=1, required=False,
            help="Size of square stride.",
            default=[1],
        )
        self.console.add_argument(
            "--ginvariant",
            action="store_true",
            help="Use G-Invariant layer for the first layer.",
        )
        self.console.add_argument(
            "--cnn",
            action="store_true",
            help="Use 2-layer CNN for the first layer.",
        )
        self.console.add_argument(
            "--amprec",
            action="store_true",
            help="Use Automatically Mixed Precision instead of FP32.",
        )
        self.console.add_argument(
            "--optim-alg",
            type=str, nargs=1, required=False,
            help="Optimizer algorithm.",
            default=["sgd"],
        )
        self.console.add_argument(
            "--learning-rate",
            type=float, nargs=1, required=False,
            help="Learning rate.",
            default=[float("nan")],
        )
        self.console.add_argument(
            "--l2-lambda",
            type=float, nargs=1, required=False,
            help="L2 regularization strength.",
            default=[0],
        )
        self.console.add_argument(
            "--num-epochs",
            type=int, nargs=1, required=False,
            help="Number of training epochs.",
            default=[100],
        )
        self.console.add_argument(
            "--device",
            type=str, nargs=1, required=False,
            choices=["cpu", "cuda"],
            help="Device to work on.",
            default=["cpu"],
        )

        # Parse the command line arguments.
        self.args = self.console.parse_args()
        self.sbatch = self.args.sbatch
        self.student = self.args.student[0]
        self.folder = self.args.mnist[0]
        self.num_samples = self.args.num_samples[0]
        self.random_seed = self.args.random_seed[0]
        self.shuffle = self.args.shuffle_label
        self.batch_size = self.args.batch_size[0]
        self.num_workers = self.args.num_workers[0]
        self.kernel = self.args.kernel[0]
        self.stride = self.args.stride[0]
        self.ginvariant = self.args.ginvariant
        self.cnn = self.args.cnn
        self.amprec = self.args.amprec
        self.optim_alg = self.args.optim_alg[0]
        self.lr = self.args.learning_rate[0]
        self.wd = self.args.l2_lambda[0]
        self.num_epochs = self.args.num_epochs[0]
        self.device = self.args.device[0]

        # Update learning rate.
        if (math.isnan(self.lr)):
            if (self.cnn):
                self.lr = 1e-1
            else:
                self.lr = 1e-4
        else:
            pass

        # Safety check.
        assert (
            not self.amprec or self.device == "cuda"
        ), "[\033[91mError\033[0m]: Automatically Mixed Precision is" \
           " designed only for GPU."

        # Generate title.
        self.generate_title()

        # Load implementation of a specific student.
        self.load_student_implementation(self.student)

        # Submit to cluster.
        if (self.sbatch):
            self.sbatch_submit()
            exit()
        else:
            pass

        # Get randomness.
        self.rng_cpu = torch.Generator("cpu")
        if torch.cuda.is_available():
            self.rng_gpu = torch.Generator("cuda")
        # Remove before submission.
        else:
            self.rng_gpu = torch.Generator("cpu")
        self.rng_cpu.manual_seed(self.random_seed)
        self.rng_gpu.manual_seed(self.random_seed)

        # Prepare datasets.
        self.load_datasets()

        # Prepare model.
        self.prepare_model()

        # Fit the model.
        self.fit()
        self.test()

    def sbatch_submit(
        self,
        /,
    ) -> None:
        r"""
        Submit by `sbatch`.

        Args
        ----

        Returns
        -------
        """
        # /
        # ANNOTATE
        # /
        sbatch_lines: List[str]
        path: str
        file: TextIO

        # Initialize sbatch submission file.
        sbatch_lines = ["#!/bin/bash"]

        # Hardware resources.
        if (self.device == "cuda"):
            sbatch_lines.append("#SBATCH -A gpu")
            sbatch_lines.append("#SBATCH --gres=gpu:1")
        elif (self.device == "cpu"):
            sbatch_lines.append("#SBATCH -A scholar")
        else:
            print(
                "[\033[91mError\033[0m]: Unknown device \"{:s}\".".format(
                    self.device
                ),
            )
            raise RuntimeError
        sbatch_lines.append(
            "#SBATCH --cpus-per-task={:d}".format(self.num_workers + 1),
        )
        sbatch_lines.append("#SBATCH --nodes=1")

        # Time limit.
        sbatch_lines.append("#SBATCH --job-name {:s}".format(self.title))
        sbatch_lines.append("#SBATCH --time=30:00")

        # IO redirection.
        sbatch_lines.append(
            "#SBATCH --output {:s}".format(
                os.path.join("logs", self.title, "output"),
            ),
        )
        sbatch_lines.append(
            "#SBATCH --error {:s}".format(
                os.path.join("logs", self.title, "error"),
            ),
        )

        # Python script.
        sbatch_lines.append(
            "python main.py \\",
        )
        sbatch_lines.append(
            "    " \
            "--student {:s} \\".format(
                self.student,
            ),
        )
        sbatch_lines.append(
            "    " \
            "--mnist {:s} --num-samples {:d} --random-seed {:d} \\".format(
                self.folder, self.num_samples, self.random_seed,
            ),
        )
        sbatch_lines.append(
            "    " \
            "--batch-size {:d} --num-workers {:d} {:s} \\".format(
                self.batch_size, self.num_workers,
                "--shuffle-label" if (self.shuffle) else "",
            ),
        )
        sbatch_lines.append(
            "    " \
            "--kernel {:d} --stride {:d} {:s} {:s} {:s} \\".format(
                self.kernel, self.stride,
                "--ginvariant" if (self.ginvariant) else "",
                "--cnn" if (self.cnn) else "",
                "--amprec" if (self.amprec) else "",
            ),
        )
        sbatch_lines.append(
            "    " \
            "--optim-alg {:s} --learning-rate {:s}" \
            " --l2-lambda {:s} \\".format(
                self.optim_alg,
                "{:.6f}".format(self.lr).rstrip("0"),
                "{:.6f}".format(self.wd).rstrip("0"),
            )
        )
        sbatch_lines.append(
            "    " \
            "--num-epochs {:d} --device {:s}".format(
                self.num_epochs, self.device,
            )
        )

        # Save to file.
        path = os.path.join("logs", self.title, "submit.sb")
        with open(path, "w") as file:
            file.write("\n".join(sbatch_lines) + "\n")

        # Run the command.
        print("[\033[31msbatch\033[0m] {:s}".format(path))
        os.system("sbatch {:s}".format(path))

    def generate_title(
        self,
        /,
    ) -> None:
        r"""
        Generate running process title.

        Args
        ----

        Returns
        -------
        """
        # /
        # ANNOTATE
        # /
        self.title: str

        # Get title directly.
        if (self.cnn):
            self.title = (
                "mnist_{:d}-{:d}-{:s}-{:d}-cnn_{:d}_{:d}-{:s}-{:s}-{:s}" \
                "-{:s}".format(
                    self.num_samples, self.random_seed,
                    "s" if (self.shuffle) else "0",
                    self.batch_size,
                    self.kernel, self.stride,
                    "m" if (self.amprec) else "0",
                    self.optim_alg,
                    "{:.6f}".format(self.lr).rstrip("0"),
                    "{:.6f}".format(self.wd).rstrip("0"),
                )
            )
        else:
            self.title = (
                "mnist_{:d}-{:d}-{:s}-{:d}-mlp_{:s}-{:s}-{:s}-{:s}" \
                "-{:s}".format(
                    self.num_samples, self.random_seed,
                    "s" if (self.shuffle) else "0",
                    self.batch_size,
                    "g" if (self.ginvariant) else "0",
                    "m" if (self.amprec) else "0",
                    self.optim_alg,
                    "{:.6f}".format(self.lr).rstrip("0"),
                    "{:.6f}".format(self.wd).rstrip("0"),
                )
            )

        # Allocate a directory.
        os.makedirs(
            os.path.join("logs", self.title),
            exist_ok=True,
        )

    def load_student_implementation(
        self,
        /,
        student: str,
    ) -> Any:
        r"""
        Load implementation of given student as a module.

        Args
        ----
        - student :
            Student PUID.

        Returns
        -------
        """
        # /
        # ANNOTATE
        # /
        self.homework_subspace: Any
        self.homework_optimizers: Any
        self.homework_mlp: Any
        self.homework_cnn: Any

        # Get student submitted homework.
        self.homework_subspace = importlib.import_module(
            ".subspace", "homework.{:s}".format(student),
        )
        self.homework_optimizers = importlib.import_module(
            ".optimizers", "homework.{:s}".format(student),
        )
        self.homework_mlp = importlib.import_module(
            ".mlp", "homework.{:s}".format(student),
        )
        self.homework_cnn = importlib.import_module(
            ".cnn", "homework.{:s}".format(student),
        )

    def get_structure(
        self,
        dataset_mnist: MNISTDataset,
        usage: int, perturbate: bool,
        /,
    ) -> MNISTPerturbDataStructure:
        r"""
        Get data structure for specific usage.

        Args
        ----
        - dataset :
            Dataset.
        - usage :
            Usage of data structure.
        - perturbate :
            If rotation or flip perturbation will be applied or not.

        Returns
        -------
        - struct :
            Data structure.
        """
        # Get directly.
        return MNISTPerturbDataStructure(
            memory=dataset_mnist,
            kfold_split=(11, 1, 2), kfold_index=(0, self.random_seed),
            kfold_usage=usage, perturbate=perturbate,
            random_seed=self.random_seed,
        )

    def get_train_loader(
        self,
        struct: MNISTPerturbDataStructure,
        /,
    ) -> torch.utils.data.DataLoader:
        r"""
        Get minibatch loader on given data structure for training.

        Args
        ----
        - struct :
            Data structure.

        Returns
        -------
        - loader :
            Minibatch loader.
        """
        # Get directly.
        return torch.utils.data.DataLoader(
            struct,
            sampler=torch.utils.data.RandomSampler(
                struct,
                replacement=False, generator=self.rng_cpu,
            ),
            batch_size=(
                len(struct) if (self.batch_size < 0) else self.batch_size
            ),
            num_workers=self.num_workers,
            collate_fn=mnist_dataset_collate,
            pin_memory=False, drop_last=False,
        )

    def get_evaluate_loader(
        self,
        struct: MNISTPerturbDataStructure,
        /,
    ) -> torch.utils.data.DataLoader:
        r"""
        Get minibatch loader on given data structure for evaluation.

        Args
        ----
        - struct :
            Data structure.

        Returns
        -------
        - loader :
            Minibatch loader.
        """
        # Get directly.
        return torch.utils.data.DataLoader(
            struct,
            sampler=torch.utils.data.SequentialSampler(struct),
            batch_size=(
                len(struct) if (self.batch_size < 0) else self.batch_size
            ),
            num_workers=self.num_workers,
            collate_fn=mnist_dataset_collate,
            pin_memory=False, drop_last=False,
        )

    def load_datasets(
        self,
        /,
    ) -> None:
        r"""
        Load all datasets used in the homework.

        Args
        ----

        Returns
        -------
        """
        # /
        # ANNOTATE
        # /
        maxlen: int
        dataset_mnist: MNISTDataset
        TRAIN: int
        VALID: int
        TEST: int
        # -----
        struct_train_raw: MNISTPerturbDataStructure
        struct_valid_raw: MNISTPerturbDataStructure
        struct_test_raw: MNISTPerturbDataStructure
        # -----
        struct_train_plus: MNISTPerturbDataStructure
        struct_valid_plus: MNISTPerturbDataStructure
        struct_test_plus: MNISTPerturbDataStructure
        # -----
        self.loader_train_raw: torch.utils.data.DataLoader
        self.loader_valid_raw: torch.utils.data.DataLoader
        self.loader_test_raw: torch.utils.data.DataLoader
        # -----
        self.loader_train_plus: torch.utils.data.DataLoader
        self.loader_valid_plus: torch.utils.data.DataLoader
        self.loader_test_plus: torch.utils.data.DataLoader
        # -----
        train_raw_len_msg: str
        valid_raw_len_msg: str
        test_raw_len_msg: str
        train_plus_len_msg: str
        valid_plus_len_msg: str
        test_plus_len_msg: str
        train_raw_bsz_msg: str
        valid_raw_bsz_msg: str
        test_raw_bsz_msg: str
        train_plus_bsz_msg: str
        valid_plus_bsz_msg: str
        test_plus_bsz_msg: str
        maxlen_len: int
        maxlen_bsz: int

        # Load full MNIST dataset into memory.
        dataset_mnist = MNISTDataset(
            self.folder,
            normalize=True, num_samples=self.num_samples,
            shuffle=self.shuffle, random_seed=self.random_seed,
        )
        TRAIN = MNISTPerturbDataStructure.TRAIN
        VALID = MNISTPerturbDataStructure.VALIDATE
        TEST = MNISTPerturbDataStructure.TEST

        # Output dataset info.
        maxlen = len(MNISTDataset.decolor(repr(dataset_mnist)))
        print("=" * 5, "=" * maxlen)
        print("MNIST", repr(dataset_mnist))
        print("=" * 5, "=" * maxlen)

        # Get raw data structres.
        struct_train_raw = self.get_structure(dataset_mnist, TRAIN, False)
        struct_valid_raw = self.get_structure(dataset_mnist, VALID, False)
        struct_test_raw = self.get_structure(dataset_mnist, TEST, False)

        # Get perturbated data structures.
        struct_train_plus = self.get_structure(dataset_mnist, TRAIN, True)
        struct_valid_plus = self.get_structure(dataset_mnist, VALID, True)
        struct_test_plus = self.get_structure(dataset_mnist, TEST, True)

        # Get raw dataset batch loaders.
        self.loader_train_raw = self.get_train_loader(struct_train_raw)
        self.loader_valid_raw = self.get_evaluate_loader(struct_valid_raw)
        self.loader_test_raw = self.get_evaluate_loader(struct_test_raw)

        # Get perturbated dataset batch loaders.
        self.loader_train_plus = self.get_train_loader(struct_train_plus)
        self.loader_valid_plus = self.get_evaluate_loader(struct_valid_plus)
        self.loader_test_plus = self.get_evaluate_loader(struct_test_plus)

        # Output data structure info.
        train_raw_len_msg = str(len(struct_train_raw))
        valid_raw_len_msg = str(len(struct_valid_raw))
        test_raw_len_msg = str(len(struct_test_raw))
        train_plus_len_msg = str(len(struct_train_plus))
        valid_plus_len_msg = str(len(struct_valid_plus))
        test_plus_len_msg = str(len(struct_test_plus))
        train_raw_bsz_msg = str(self.loader_train_raw.batch_size)
        valid_raw_bsz_msg = str(self.loader_valid_raw.batch_size)
        test_raw_bsz_msg = str(self.loader_test_raw.batch_size)
        train_plus_bsz_msg = str(self.loader_train_plus.batch_size)
        valid_plus_bsz_msg = str(self.loader_valid_plus.batch_size)
        test_plus_bsz_msg = str(self.loader_test_plus.batch_size)
        maxlen_len = max(
            len(train_raw_len_msg), len(valid_raw_len_msg),
            len(test_raw_len_msg),
            len(train_plus_len_msg), len(valid_plus_len_msg),
            len(test_plus_len_msg),
        )
        maxlen_bsz = max(
            len(train_raw_bsz_msg), len(valid_raw_bsz_msg),
            len(test_raw_bsz_msg),
            len(train_plus_bsz_msg), len(valid_plus_bsz_msg),
            len(test_plus_bsz_msg),
        )
        print("=" * 17, "=" * maxlen_len, "=" * maxlen_bsz)
        print(
            "Original Train   ",
            train_raw_len_msg.ljust(maxlen_len),
            train_raw_bsz_msg.ljust(maxlen_bsz),
        )
        print(
            "Original Validate",
            valid_raw_len_msg.ljust(maxlen_len),
            valid_raw_bsz_msg.ljust(maxlen_len),
        )
        print(
            "Original Test    ",
            test_raw_len_msg.ljust(maxlen_len),
            test_raw_bsz_msg.ljust(maxlen_len),
        )
        print(
            "Operated Train   ",
            train_plus_len_msg.ljust(maxlen_len),
            train_plus_bsz_msg.ljust(maxlen_len),
        )
        print(
            "Operated Validate",
            valid_plus_len_msg.ljust(maxlen_len),
            valid_plus_bsz_msg.ljust(maxlen_len),
        )
        print(
            "Operated Test    ",
            test_plus_len_msg.ljust(maxlen_len),
            test_plus_bsz_msg.ljust(maxlen_len),
        )
        print("=" * 17, "=" * maxlen_len, "=" * maxlen_bsz)

    def prepare_model(
        self,
        /,
    ) -> None:
        r"""
        Prepare all models used in the homework.

        Args
        ----

        Returns
        -------
        """
        # /
        # ANNOTATE
        # /
        self.gradscaler: torch.cuda.amp.grad_scaler.GradScaler
        self.model: MNISTClassification
        self.optimizer: torch.optim.Optimizer

        # Create gradient scaler for mix precision.
        self.gradscaler = torch.cuda.amp.GradScaler()

        # Create model.
        if (self.cnn):
            self.model = StackCNN(
                self.rng_cpu, self.rng_gpu, self.gradscaler,
                num_input_channels=1, num_output_channels=32,
                num_internal_channels=100,
                conv_kernel=self.kernel, conv_stride=self.stride,
                pool_kernel=3, pool_stride=1, padding=1,
                num_labels=MNISTDataset.NUM_LABELS, num_internals=100,
                height=MNISTDataset.HEIGHT, width=MNISTDataset.WIDTH,
                criterion=MNISTClassification.ACCURACY,
                dual_cnn=self.homework_cnn.DualCNN,
                amprec=self.amprec,
            )
        else:
            self.model = GInvariantMLP(
                self.rng_cpu, self.rng_gpu, self.gradscaler,
                num_inputs=MNISTDataset.NUM_PIXELS,
                num_labels=MNISTDataset.NUM_LABELS,
                num_internals=[300, 100],
                criterion=MNISTClassification.ACCURACY,
                eigenvectors=self.homework_subspace.InvariantSubspace(
                    rotate, flip,
                    shape=(MNISTDataset.HEIGHT, MNISTDataset.WIDTH),
                ).eigenvectors,
                ginvariant=self.ginvariant,
                ginvariant_linear=self.homework_mlp.GInvariantLinear,
                amprec=self.amprec,
            )
        self.model.initialize(self.rng_cpu)
        self.model = self.model.to(self.device)

        # Define optimization strategy.
        if (self.optim_alg == "sgd"):
            self.optimizer = self.homework_optimizers.SGD(
                self.model.parameters(),
                lr=self.lr, weight_decay=self.wd,
            )
        elif (self.optim_alg == "momentum"):
            self.optimizer = self.homework_optimizers.Momentum(
                self.model.parameters(),
                lr=self.lr, weight_decay=self.wd,
                momentum=0.9,
            )
        elif (self.optim_alg == "nesterov"):
            self.optimizer = self.homework_optimizers.Nesterov(
                self.model.parameters(),
                lr=self.lr, weight_decay=self.wd,
                momentum=0.9,
            )
        elif (self.optim_alg == "adam"):
            self.optimizer = self.homework_optimizers.Adam(
                self.model.parameters(),
                lr=self.lr, weight_decay=self.wd,
                beta1=0.9, beta2=0.999, epsilon=1e-8,
            )
        else:
            print(
                "[\033[91mError\033[0m]: Unknown optimizer algorithm \"{:s}\"."
                .format(self.optim_alg),
            )
            raise RuntimeError
        self.model.optim(self.optimizer)

    def train_minibatch(
        self,
        loader: torch.utils.data.DataLoader,
        /,
    ) -> float:
        r"""
        Train with minibatch.

        Args
        ----
        - loader :
            Minibatch loader.

        Returns
        -------
        - cost :
            Optimization time cost.
        """
        # /
        # ANNOTATE
        # /
        batch: Dict[str, torch.Tensor]
        key: str
        elapsed: float
        cost: float

        # Iterate batches once for training.
        cost = 0
        for batch in loader:
            # Common training.
            for key in batch:
                batch[key] = batch[key].to(
                    self.device,
                    non_blocking=True,
                )
            self.optimizer.zero_grad()
            elapsed = time.time()
            self.model.trainit(**batch)

            # Support for auto mixed precision.
            if (self.amprec):
                self.gradscaler.step(self.optimizer)
                self.gradscaler.update()
            else:
                self.optimizer.step()
            elapsed = time.time() - elapsed
            cost += elapsed
        return cost

    def evaluate_minibatch(
        self,
        loader: torch.utils.data.DataLoader,
        /,
    ) -> Tuple[float, torch.Tensor]:
        r"""
        Evaluate a model with minibatches

        Args
        ----
        - loader :
            Minibatch loader.

        Returns
        -------
        - criterion :
            Criterion for early stopping.
        - performance :
            Full evaluation performance report.
        """
        # /
        # ANNOTATE
        # /
        batch: Dict[str, torch.Tensor]
        key: str

        # Iterate batches once for evaluation.
        self.model.evaluaton()
        for batch in loader:
            for key in batch:
                batch[key] = batch[key].to(
                    self.device,
                    non_blocking=True,
                )
            self.model.evaluatit(**batch)
        return self.model.evaluatoff()

    def init_log(
        self,
        performance_train: torch.Tensor,
        performance_valid: torch.Tensor,
        performance_test: torch.Tensor,
        /,
    ) -> None:
        r"""
        Initialize log.

        Args
        ----
        - performance_train :
            Full evaluation performance report on training data.
        - performance_valid :
            Full evaluation performance report on validation data.
        - performance_test :
            Full evaluation performance report on test data.

        Returns
        -------
        """
        # /
        # ANNOTATE
        # /
        self.log: Dict[str, List[float]]
        metric_name: str
        metric_train: torch.Tensor
        metric_valid: torch.Tensor
        metric_test: torch.Tensor

        # Initialize all performance tracking together.
        self.log = dict()
        for metric_name, metric_train, metric_valid, metric_test in zip(
            MNISTClassification.METRICS,
            performance_train, performance_valid, performance_test,
        ):
            self.log[
                "Train {:s}".format(metric_name)
            ] = [metric_train.item()]
            self.log[
                "Validate {:s}".format(metric_name)
            ] = [metric_valid.item()]
            self.log[
                "Test {:s}".format(metric_name)
            ] = [metric_test.item()]
        self.log["Optimize Time"] = [0.0]
        torch.save(self.log, os.path.join("logs", self.title, "log.pt"))

    def update_log(
        self,
        performance_train: torch.Tensor,
        performance_valid: torch.Tensor,
        performance_test: torch.Tensor,
        elapsed: float,
        /,
    ) -> None:
        r"""
        Update log.

        Args
        ----
        - performance_train :
            Full evaluation performance report on training data.
        - performance_valid :
            Full evaluation performance report on validation data.
        - performance_test :
            Full evaluation performance report on test data.
        - elapsed :
            Optimization time cost.

        Returns
        -------
        """
        # /
        # ANNOTATE
        # /
        metric_name: str
        metric_train: torch.Tensor
        metric_valid: torch.Tensor
        metric_test: torch.Tensor

        # Update all performance together.
        for metric_name, metric_train, metric_valid, metric_test in zip(
            MNISTClassification.METRICS,
            performance_train, performance_valid, performance_test,
        ):
            self.log[
                "Train {:s}".format(metric_name)
            ].append(metric_train.item())
            self.log[
                "Validate {:s}".format(metric_name)
            ].append(metric_valid.item())
            self.log[
                "Test {:s}".format(metric_name)
            ].append(metric_test.item())
        self.log["Optimize Time"].append(elapsed)
        torch.save(self.log, os.path.join("logs", self.title, "log.pt"))

    def fit(
        self,
        /,
    ) -> None:
        r"""
        Tune model parameters.

        Args
        ----

        Returns
        -------
        """
        # /
        # ANNOTATE
        # /
        criterion_train: float
        criterion_valid: float
        criterion_test: float
        performance_train: torch.Tensor
        performance_valid: torch.Tensor
        performance_test: torch.Tensor
        elapsed: float
        best_valid: float
        self.best_state_dict: Dict[str, torch.Tensor]
        key: str
        val: torch.Tensor

        # Evaluation on initialization.
        criterion_train, performance_train = self.evaluate_minibatch(
            self.loader_train_raw,
        )
        criterion_valid, performance_valid = self.evaluate_minibatch(
            self.loader_valid_raw,
        )
        criterion_test, performance_test = self.evaluate_minibatch(
            self.loader_test_raw,
        )

        # Initialize early stopping.
        best_valid = criterion_valid
        self.best_state_dict = copy.deepcopy(self.model.state_dict())
        torch.save(
            {key: val.cpu() for key, val in self.best_state_dict.items()},
            os.path.join("logs", self.title, "init.pt"),
        )

        # Initialize log.
        self.init_log(
            performance_train, performance_valid, performance_test,
        )

        # Output progress.
        print("=" * 5, "=" * 8, "=" * 8, "=" * 8, "=" * 6)
        print("Epoch", "Train   ", "Validate", "Test    ", "Time  ")
        print("-" * 5, "-" * 8, "-" * 8, "-" * 8, "-" * 6)
        print(
            "{:s} {:s} {:s} {:s} {:s}".format(
                str(0).rjust(5),
                "{:.6f}".format(criterion_train)[0:8].rjust(8),
                "{:.6f}".format(criterion_valid)[0:8].rjust(8),
                "{:.6f}".format(criterion_test)[0:8].rjust(8),
                "{:.3f}".format(0.0)[0:8].rjust(6),
            ),
        )

        # Traverse given number of epochs.
        for e in range(1, 1 + self.num_epochs):
            # Iterate training batches once.
            elapsed = self.train_minibatch(self.loader_train_raw)

            # Evaluation after updating weights.
            criterion_train, performance_train = self.evaluate_minibatch(
                self.loader_train_raw,
            )
            criterion_valid, performance_valid = self.evaluate_minibatch(
                self.loader_valid_raw,
            )
            criterion_test, performance_test = self.evaluate_minibatch(
                self.loader_test_raw,
            )

            # Early stopping.
            if (criterion_valid > best_valid):
                best_valid = criterion_valid
                self.best_state_dict = copy.deepcopy(self.model.state_dict())
            else:
                pass

            # Log.
            self.update_log(
                performance_train, performance_valid, performance_test,
                elapsed,
            )

            # Output progress.
            print(
                "{:s} {:s} {:s} {:s} {:s}".format(
                    str(e).rjust(5),
                    "{:.6f}".format(criterion_train)[0:8].rjust(8),
                    "{:.6f}".format(criterion_valid)[0:8].rjust(8),
                    "{:.6f}".format(criterion_test)[0:8].rjust(8),
                    "{:.3f}".format(elapsed)[0:8].rjust(6),
                ),
            )
        torch.save(
            {key: val.cpu() for key, val in self.best_state_dict.items()},
            os.path.join("logs", self.title, "best.pt"),
        )
        print("=" * 5, "=" * 8, "=" * 8, "=" * 8, "=" * 6)

    def test(
        self,
        /,
    ) -> None:
        r"""
        Test performance.

        Args
        ----

        Returns
        -------
        """
        # /
        # ANNOTATE
        # /
        title: str
        loader: torch.utils.data.DataLoader
        criterion: float
        self.report: Dict[str, float]

        # Report on final model.
        self.model.load_state_dict(self.best_state_dict)
        self.report = dict()

        # Traverse and evaluate all loaders.
        print("=" * 17, "=" * 8)
        for title, loader in [
            ("Original Train   ", self.loader_train_raw),
            ("Original Validate", self.loader_valid_raw),
            ("Original Test    ", self.loader_test_raw),
        ]:
            criterion, _ = self.evaluate_minibatch(loader)
            self.report[title] = criterion
            print(
                "{:s} {:s}".format(
                    title, "{:.6f}".format(criterion)[0:8].rjust(8),
                ),
            )

        # Traverse and evaluate all loaders with perturbation.
        for title, loader in [
            ("Operated Train   ", self.loader_train_plus),
            ("Operated Validate", self.loader_valid_plus),
            ("Operated Test    ", self.loader_test_plus),
        ]:
            criterion, _ = self.evaluate_minibatch(loader)
            self.report[title] = criterion
            print(
                "{:s} {:s}".format(
                    title, "{:.6f}".format(criterion)[0:8].rjust(8),
                ),
            )
        print("=" * 17, "=" * 8)


# Run.
if (__name__ == "__main__"):
    Main()
else:
    pass