# All imports
import argparse
import torch
import importlib
import os
import copy
import time
import math
from typing import Any
from typing import Tuple, List, Dict
from datasets import Dataset
from datasets import CoraDataset
from datasets import PTBDataset
from structures import KFoldStructure
from structures import CoraDataStructure
from structures import PTBDataStructure
from models import MulticlassClassification
from models import MulticlassNodeClassification
from models import MulticlassWordClassification


# =============================================================================
# *****************************************************************************
# -----------------------------------------------------------------------------
# ## Main
# -----------------------------------------------------------------------------
# *****************************************************************************
# =============================================================================


class BasicMain(
    object,
    metaclass=type,
):
    r"""
    Main interface.
    """
    # =========================================================================
    # -------------------------------------------------------------------------
    # =========================================================================
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
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # /
        # ANNOTATE
        # /
        self.console: argparse.ArgumentParser
        argroup: argparse._MutuallyExclusiveGroup
        self.cora: bool
        self.ptb: bool
        self.sbatch: bool
        self.student: str
        self.folder: str
        self.random_seed: int
        self.dense: bool
        self.batch_size: int
        self.truncate: int
        self.num_workers: int
        self.janossy: bool
        self.num_perms: int
        self.markov: bool
        self.num_internals: int
        self.num_graph_layers: int
        self.lr: float
        self.wd: float
        self.clip: float
        self.num_epochs: int
        self.device: str
        # -----
        self.rng_cpu: torch.Generator
        self.rng_gpu: torch.Generator

        # Create training console arguments.
        self.console = argparse.ArgumentParser(
            description="Homework 3",
        )
        argroup = self.console.add_mutually_exclusive_group()
        argroup.add_argument(
            "--cora",
            action="store_true",
            help="Work on Cora related tasks.",
        )
        argroup.add_argument(
            "--ptb",
            action="store_true",
            help="Work on PTB related tasks.",
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
        )
        self.console.add_argument(
            "--data",
            type=str, nargs=1, required=False,
            help="Path to the data directory.",
            default=["../Data/Cora"],
        )
        self.console.add_argument(
            "--random-seed",
            type=int, nargs=1, required=False,
            help="Random seed."
        )
        self.console.add_argument(
            "--dense",
            action="store_true",
            help="Use dense instead of sparse adjacency data."
        )
        self.console.add_argument(
            "--batch-size",
            type=int, nargs=1, required=False,
            help="Batch size.",
        )
        self.console.add_argument(
            "--truncate",
            type=int, nargs=1, required=False,
            help="BPTT Truncation length.",
        )
        self.console.add_argument(
            "--num-workers",
            type=int, nargs=1, required=False,
            help="Number of batch sampling processes.",
        )
        self.console.add_argument(
            "--janossy",
            action="store_true",
            help="Use Janossy Pooling as sparse graph kernel."
        )
        self.console.add_argument(
            "--num-perms",
            type=int, nargs=1, required=False,
            help="Number of sampled permutations in Janossy Pooling test.",
        )
        self.console.add_argument(
            "--markov",
            action="store_true",
            help="Use Markovian as language modeling kernel."
        )
        self.console.add_argument(
            "--num-internals",
            type=int, nargs=1, required=False,
            help=(
                "Number of internal neurons. It works as chain order in" \
                "Markovian language modeling."
            ),
        )
        self.console.add_argument(
            "--num-graph-layers",
            type=int, nargs=1, required=False,
            help="Number of graph kernel layers.",
        )
        self.console.add_argument(
            "--learning-rate",
            type=float, nargs=1, required=False,
            help="Learning rate.",
        )
        self.console.add_argument(
            "--weight-decay",
            type=float, nargs=1, required=False,
            help="Weight decay.",
        )
        self.console.add_argument(
            "--clip",
            type=float, nargs=1, required=False,
            help="Gradient clipping.",
        )
        self.console.add_argument(
            "--num-epochs",
            type=int, nargs=1, required=False,
            help="Number of training epochs.",
        )
        self.console.add_argument(
            "--device",
            type=str, nargs=1, required=False,
            choices=["cpu", "cuda"],
            help="Device to work on.",
        )

        # Parse the command line arguments.
        self.args = self.console.parse_args()
        self.cora = self.args.cora
        self.ptb = self.args.ptb
        self.sbatch = self.args.sbatch
        self.student = self.args.student[0]
        self.folder = self.args.data[0]
        self.random_seed = self.args.random_seed[0]
        self.dense = self.args.dense
        self.batch_size = self.args.batch_size[0]
        self.truncate = self.args.truncate[0]
        self.num_workers = self.args.num_workers[0]
        self.janossy = self.args.janossy
        self.num_perms = self.args.num_perms[0]
        self.markov = self.args.markov
        self.num_internals = self.args.num_internals[0]
        self.num_graph_layers = self.args.num_graph_layers[0]
        self.lr = self.args.learning_rate[0]
        self.wd = self.args.weight_decay[0]
        self.clip = self.args.clip[0]
        self.num_epochs = self.args.num_epochs[0]
        self.device = self.args.device[0]

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
        self.rng_gpu = torch.Generator(self.device)
        self.rng_cpu.manual_seed(self.random_seed)
        self.rng_gpu.manual_seed(self.random_seed)


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
        sbatch_lines.append("#SBATCH --time=60:00")

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
        if (self.cora):
            sbatch_lines.append(
                "    " \
                "--cora --student {:s} \\".format(
                    self.student,
                ),
            )
        elif (self.ptb):
            sbatch_lines.append(
                "    " \
                "--ptb --student {:s} \\".format(
                    self.student,
                ),
            )
        else:
            print(
                "[\033[91mError\033[0m]: Reach impossible branch."
            )
            raise NotImplementedError
        sbatch_lines.append(
            "    " \
            "--data {root:s} --random-seed {seed:d}" \
            " {dense:s} \\".format(
                root=self.folder, seed=self.random_seed,
                dense="--dense" if (self.dense) else "",
            ),
        )
        sbatch_lines.append(
            "    " \
            "--batch-size {bsz:d} --truncate {trunc:d}" \
            " --num-workers {workers:d} \\".format(
                bsz=self.batch_size, trunc=self.truncate,
                workers=self.num_workers,
            ),
        )
        sbatch_lines.append(
            "    " \
            "--num-internals {intern:d} --num-graph-layers {layer:d}" \
            " --num-perms {perm:d} {jp:s} {markov:s} \\".format(
                intern=self.num_internals, layer=self.num_graph_layers,
                perm=self.num_perms,
                jp="--janossy" if (self.janossy) else "",
                markov="--markov" if (self.markov) else "",
            ),
        )
        sbatch_lines.append(
            "    " \
            "--learning-rate 0.01 --weight-decay 5e-5 --clip inf \\",
        )
        sbatch_lines.append(
            "    " \
            "--num-epochs {num:d} --device {device:s}".format(
                num=self.num_epochs, device=self.device,
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
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # /
        # ANNOTATE
        # /
        self.title: str

        # Get title directly.
        if (self.cora):
            self.title = (
                "{task:s}-{random_seed:d}-{dense:s}" \
                "-{janossy:s}-{num_perms:d}-{num_graph_layers:d}".format(
                    task="cora", random_seed=self.random_seed,
                    dense="d" if (self.dense) else "s",
                    janossy="jp" if (self.janossy) else "gc",
                    num_perms=self.num_perms,
                    num_graph_layers=self.num_graph_layers,
                )
            )
        elif (self.ptb):
            self.title = (
                "{task:s}-{random_seed:d}" \
                "-{markov:s}-{trunc:d}-{num_internals:d}".format(
                    task="ptb", random_seed=self.random_seed,
                    markov="mrkv" if (self.markov) else "lstm",
                    trunc=self.truncate,
                    num_internals=self.num_internals,
                )
            )
        else:
            print(
                "[\033[91mError\033[0m]: Reach impossible branch."
            )
            raise NotImplementedError

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
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # /
        # ANNOTATE
        # /
        self.homework_gcn: Any
        self.homework_graphsage: Any
        self.homework_markov: Any
        self.homework_lstm: Any

        # Get student submitted homework.
        self.homework_gcn = importlib.import_module(
            ".gcn", "homework.{:s}".format(student),
        )
        self.homework_graphsage = importlib.import_module(
            ".graphsage", "homework.{:s}".format(student),
        )
        self.homework_markov = importlib.import_module(
            ".markov", "homework.{:s}".format(student),
        )
        self.homework_lstm = importlib.import_module(
            ".lstm", "homework.{:s}".format(student),
        )

    # =========================================================================
    # -------------------------------------------------------------------------
    # Data preprocessing stage.
    # -------------------------------------------------------------------------
    # =========================================================================

    def load_memory(
        self,
        /,
    ) -> None:
        r"""
        Load full dataset into memory.

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
        msg: str
        msglen: int
        # -----
        self.dataset: Dataset
        datamsg: str

        # Load full dataset into memory.
        if (self.cora):
            self.dataset = CoraDataset(
                self.folder,
                dense=self.dense,
            )
            datamsg = "Cora"
        elif (self.ptb):
            self.dataset = PTBDataset(self.folder)
            datamsg = "PTB"
        else:
            print(
                "[\033[91mError\033[0m]: Reach impossible branch."
            )
            raise NotImplementedError

        # Output dataset info.
        msg = repr(self.dataset)
        msglen = len(Dataset.decolor(msg))
        print("=" * len(datamsg), "=" * msglen)
        print(datamsg, msg.ljust(msglen))
        print("=" * len(datamsg), "=" * msglen)

    def split_structs(
        self,
        /,
    ) -> None:
        r"""
        Split in-memory dataset into data structures for different usages.

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
        tr: int
        va: int
        te: int
        self.struct_train: KFoldStructure
        self.struct_valid: KFoldStructure
        self.struct_test: KFoldStructure
        # -----
        msg_train: str
        msg_valid: str
        msg_test: str
        msglen: int

        # Shorten constant names.
        tr = KFoldStructure.TRAIN
        va = KFoldStructure.VALIDATE
        te = KFoldStructure.TEST

        # Get data structres.
        self.struct_train = self.get_structure(self.dataset, tr)
        self.struct_valid = self.get_structure(self.dataset, va)
        self.struct_test = self.get_structure(self.dataset, te)

        # Output data structure info.
        msg_train = repr(self.struct_train)
        msg_valid = repr(self.struct_valid)
        msg_test = repr(self.struct_test)
        msglen = max(
            18,
            len(KFoldStructure.decolor(msg_train)),
            len(KFoldStructure.decolor(msg_valid)),
            len(KFoldStructure.decolor(msg_test)),
        )
        print("=" * 8, "=" * msglen)
        print("Usage   ", "Folding Statistics")
        print("-" * 8, "-" * msglen)
        print("Train   ", msg_train.ljust(msglen))
        print("Validate", msg_valid.ljust(msglen))
        print("Test    ", msg_test.ljust(msglen))
        print("=" * 8, "=" * msglen)

    def get_structure(
        self,
        dataset: Dataset, usage: int,
        /,
    ) -> KFoldStructure:
        r"""
        Get data structure for specific usage.

        Args
        ----
        - dataset :
            Dataset.
        - usage :
            Usage of data structure.

        Returns
        -------
        - struct :
            Data structure.
        """
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # /
        # ANNOTATE
        # /
        struct: KFoldStructure

        # Split full dataset for given usage.
        if (self.cora):
            struct = CoraDataStructure(
                self.dataset, usage,
                rest_test_split=4, train_valid_split=7,
                rest_test_index=0, train_valid_index=0,
            )
        elif (self.ptb):
            struct = PTBDataStructure(
                self.dataset, usage,
                rest_test_split=2, train_valid_split=1,
                rest_test_index=0, train_valid_index=0,
            )
            struct.preprocess(
                dict(batch_size=self.batch_size, truncate=self.truncate),
                self.homework_lstm.t_bptt_preprocess,
            )
        else:
            print(
                "[\033[91mError\033[0m]: Reach impossible branch."
            )
            raise NotImplementedError
        return struct

    def wrap_loaders(
        self,
        /,
    ) -> None:
        r"""
        Wrap data structures by minibatch loaders.

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
        train_shuffle: bool
        self.loader_train: torch.utils.data.DataLoader
        self.loader_valid: torch.utils.data.DataLoader
        self.loader_test: torch.utils.data.DataLoader
        # -----
        msg_train: str
        msg_valid: str
        msg_test: str
        msglen: int

        # Determine training minibatch shuffling.
        # PTB is sequential dataset, and has batch size 1 for simplicity, thus
        # there is no shuffling.
        if (self.cora):
            train_shuffle = True
        elif (self.ptb):
            train_shuffle = False
        else:
            print(
                "[\033[91mError\033[0m]: Reach impossible branch."
            )
            raise NotImplementedError

        # Pin shared minibatch data in memory as part of any minibatch.
        self.batch_pin = self.dataset.pin(self.device)

        # Get minibatch loaders.
        self.loader_train = self.get_loader(self.struct_train, train_shuffle)
        self.loader_valid = self.get_loader(self.struct_valid, False)
        self.loader_test = self.get_loader(self.struct_test, False)

        # Output minibatch info.
        msg_train = str(self.loader_train.batch_size)
        msg_valid = str(self.loader_valid.batch_size)
        msg_test = str(self.loader_test.batch_size)
        msglen = max(14, len(msg_train), len(msg_valid), len(msg_test))
        print("=" * 8, "=" * msglen)
        print("Usage   ", "Minibatch Size")
        print("-" * 8, "-" * msglen)
        print("Train   ", msg_train.rjust(msglen))
        print("Validate", msg_valid.rjust(msglen))
        print("Test    ", msg_test.rjust(msglen))
        print("=" * 8, "=" * msglen)

    def get_loader(
        self,
        struct: KFoldStructure, shuffle: bool,
        /,
    ) -> torch.utils.data.DataLoader:
        r"""
        Get minibatch loader on given data structure.

        Args
        ----
        - struct :
            Data structure.
        - shuffle :
            If True, it will shuffle sample order during minibatch sampling.

        Returns
        -------
        - loader :
            Minibatch loader.
        """
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # /
        # ANNOTATE
        # /
        sampler: torch.utils.data.Sampler
        batch_size: int

        # Get minibatch sampler.
        if (shuffle):
            sampler = torch.utils.data.RandomSampler(
                struct,
                replacement=False, generator=self.rng_cpu,
            )
        else:
            sampler = torch.utils.data.SequentialSampler(struct)

        # Get batch size.
        if (self.batch_size < 0):
            batch_size = len(struct)
        elif (self.ptb):
            batch_size = 1
        else:
            batch_size = self.batch_size

        # Get minibatch loader.
        return torch.utils.data.DataLoader(
            struct,
            sampler=sampler, batch_size=batch_size,
            num_workers=self.num_workers, collate_fn=struct.collate,
            pin_memory=False, drop_last=False,
        )

    # =========================================================================
    # -------------------------------------------------------------------------
    # Model preprocessing stage.
    # -------------------------------------------------------------------------
    # =========================================================================

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
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # /
        # ANNOTATE
        # /
        graph_kernel: type
        janossy_kwargs: Dict[str, int]
        lm_kernel: type
        self.model: MulticlassClassification
        self.optimizer: torch.optim.Optimizer

        # Create model.
        if (self.cora):
            if (self.dense):
                graph_kernel = self.homework_gcn.DenseGCN
                janossy_kwargs = dict()
            elif (self.janossy):
                graph_kernel = self.homework_graphsage.SparseJanossy
                janossy_kwargs = dict(kary=5, num_perms=self.num_perms)
            else:
                graph_kernel = self.homework_graphsage.SparseGCN
                janossy_kwargs = dict()
            self.model = MulticlassNodeClassification(
                self.rng_cpu, self.rng_gpu,
                num_inputs=getattr(self.dataset, "NUM_FEATS"),
                num_internals=self.num_internals,
                num_outputs=getattr(self.dataset, "NUM_LABELS"),
                num_graph_layers=self.num_graph_layers,
                graph_kernel=graph_kernel,
                criterion=MulticlassClassification.ACCURACY,
                janossy_kwargs=janossy_kwargs,
            )
        elif (self.ptb):
            if (self.markov):
                lm_kernel = self.homework_markov.Markov
            else:
                lm_kernel = self.homework_lstm.LSTM
            self.model = MulticlassWordClassification(
                self.rng_cpu, self.rng_gpu,
                num_words=getattr(self.dataset, "NUM_WORDS"),
                num_internals=self.num_internals,
                lm_kernel=lm_kernel, noprop=self.markov,
                criterion=MulticlassClassification.LOSS,
            )
        else:
            print(
                "[\033[91mError\033[0m]: Reach impossible branch."
            )
            raise NotImplementedError
        self.model.initialize(self.rng_cpu)
        self.model = self.model.to(self.device)
        if (self.markov):
            pass
        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.lr, weight_decay=self.wd,
            )

    # =========================================================================
    # -------------------------------------------------------------------------
    # Parameter tuning stage.
    # -------------------------------------------------------------------------
    # =========================================================================

    def train_minibatch(
        self,
        loader: torch.utils.data.DataLoader,
        /,
    ) -> float:
        r"""
        Train a model with minibatches.

        Args
        ----
        - loader :
            Minibatch loader.

        Returns
        -------
        - cost :
            Optimization time cost.
        """
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # /
        # ANNOTATE
        # /
        batch: Dict[str, torch.Tensor]
        key: str
        elapsed: float
        cost: float

        # Iterate batches once for training.
        if (self.ptb):
            self.model.clear()
        else:
            pass
        cost = 0
        for batch in loader:
            # Initialize an interation.
            for key in batch:
                batch[key] = batch[key].to(
                    self.device,
                    non_blocking=True,
                )
            batch.update(self.batch_pin)
            if (self.markov):
                pass
            else:
                self.optimizer.zero_grad()

            # Common training.
            elapsed = time.time()
            if (self.markov):
                self.model.trainit(batch)
            else:
                self.model.trainit(batch)
                if (math.isinf(self.clip)):
                    pass
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip,
                    )
                self.optimizer.step()
            elapsed = time.time() - elapsed

            # Update epoch time cost.
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
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # /
        # ANNOTATE
        # /
        batch: Dict[str, torch.Tensor]
        key: str

        # Iterate batches once for evaluation.
        if (self.ptb):
            self.model.clear()
        else:
            pass
        self.model.evaluaton()
        for batch in loader:
            for key in batch:
                batch[key] = batch[key].to(
                    self.device,
                    non_blocking=True,
                )
            batch.update(self.batch_pin)
            self.model.evaluatit(batch)
        return self.model.evaluatoff()

    def update_log(
        self,
        performance_valid: torch.Tensor,
        performance_test: torch.Tensor,
        elapsed: float,
        /,
    ) -> None:
        r"""
        Update log.

        Args
        ----
        - performance_valid :
            Full evaluation performance report on validation data.
        - performance_test :
            Full evaluation performance report on test data.
        - elapsed :
            Optimization time cost.

        Returns
        -------
        """
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # /
        # ANNOTATE
        # /
        metric_name: str
        metric_valid: torch.Tensor
        metric_test: torch.Tensor

        # Update all performance together.
        for metric_name, metric_valid, metric_test in zip(
            MulticlassClassification.METRICS,
            performance_valid, performance_test,
        ):
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
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # /
        # ANNOTATE
        # /
        criterion_valid: float
        criterion_test: float
        performance_valid: torch.Tensor
        performance_test: torch.Tensor
        elapsed: float
        # -----
        self.log: Dict[str, List[float]]
        metric_name: str
        # -----
        best_valid: float
        self.best_state_dict: Dict[str, torch.Tensor]
        key: str
        val: torch.Tensor

        # Evaluation on initialization.
        criterion_valid, performance_valid = self.evaluate_minibatch(
            self.loader_valid,
        )
        criterion_test, performance_test = self.evaluate_minibatch(
            self.loader_test,
        )

        # Initialize early stopping.
        best_valid = criterion_valid
        self.best_state_dict = copy.deepcopy(self.model.state_dict())

        # Initialize all performance tracking together.
        self.log = dict()
        for metric_name in MulticlassClassification.METRICS:
            self.log["Validate {:s}".format(metric_name)] = []
            self.log["Test {:s}".format(metric_name)] = []
        self.log["Optimize Time"] = []
        self.update_log(performance_valid, performance_test, 0.0)

        # Output progress.
        print("=" * 5, "=" * 8, "=" * 8, "=" * 8)
        print("Epoch", "Validate", "Test    ", "Time  ")
        print("-" * 5, "-" * 8, "-" * 8, "-" * 8)
        print(
            "{:s} {:s} {:s} {:s}".format(
                str(0).rjust(5),
                "{:.6f}".format(criterion_valid)[0:8].rjust(8),
                "{:.6f}".format(criterion_test)[0:8].rjust(8),
                "{:.3f}".format(0.0)[0:8].rjust(8),
            ),
        )

        # Traverse given number of epochs.
        for e in range(1, 1 + self.num_epochs):
            # Iterate training batches once.
            elapsed = self.train_minibatch(self.loader_train)

            # Evaluation after updating weights.
            criterion_valid, performance_valid = self.evaluate_minibatch(
                self.loader_valid,
            )
            criterion_test, performance_test = self.evaluate_minibatch(
                self.loader_test,
            )

            # Early stopping.
            if (criterion_valid > best_valid):
                best_valid = criterion_valid
                self.best_state_dict = copy.deepcopy(self.model.state_dict())
            else:
                pass

            # Log.
            self.update_log(performance_valid, performance_test, elapsed)

            # Output progress.
            print(
                "{:s} {:s} {:s} {:s}".format(
                    str(e).rjust(5),
                    "{:.6f}".format(criterion_valid)[0:8].rjust(8),
                    "{:.6f}".format(criterion_test)[0:8].rjust(8),
                    "{:.3f}".format(elapsed)[0:8].rjust(8),
                ),
            )
        torch.save(
            {key: val.cpu() for key, val in self.best_state_dict.items()},
            os.path.join("logs", self.title, "best.pt"),
        )
        print("=" * 5, "=" * 8, "=" * 8, "=" * 8)

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
        embedding: torch.Tensor
        i: int
        j: int
        mse: float
        mse_max: float

        # Report on final model.
        if (self.cora and not self.janossy):
            self.evaluate_minibatch(self.loader_test)
            embedding = getattr(self.model, "saved_graph_embedding").data.cpu()
            mse_max = 0
            for i in range(len(embedding)):
                for j in range(i + 1, len(embedding)):
                    mse = torch.mean((embedding[i] - embedding[j]) ** 2).item()
                    if (mse > mse_max):
                        mse_max = mse
                    else:
                        pass
                if ((i + 1) % (len(embedding) // 10) == 0):
                    print(
                        "MSE Maximum: [{:4d}/{:4d}]".format(
                            i + 1, len(embedding),
                        ),
                    )
                else:
                    pass
            print("MSE Maximum: {:.6f}".format(mse_max))
        else:
            pass