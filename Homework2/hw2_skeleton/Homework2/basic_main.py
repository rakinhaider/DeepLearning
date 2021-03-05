from main import Main
import torch
import math
import argparse
from typing import List


class BasicMain(Main):
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
            "--debug",
            action="store_true",
            help="Debug is turned on.",
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
        self.debug = self.args.debug

        # Update learning rate.
        if math.isnan(self.lr):
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

        # Get randomness.
        self.rng_cpu = torch.Generator("cpu")
        if torch.cuda.is_available():
            print('Cuda is available.')
            self.rng_gpu = torch.Generator("cuda")
        # Remove before submission.
        else:
            self.rng_gpu = torch.Generator("cpu")
        self.rng_cpu.manual_seed(self.random_seed)
        self.rng_gpu.manual_seed(self.random_seed)
