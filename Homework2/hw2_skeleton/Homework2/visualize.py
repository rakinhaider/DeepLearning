# All imports
from typing import Any
from typing import List
import argparse
import os
import pandas as pd
import torch
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt


# =============================================================================
# *****************************************************************************
# -----------------------------------------------------------------------------
# < Visualize >
# -----------------------------------------------------------------------------
# *****************************************************************************
# =============================================================================


class Visualize(
    object,
    metaclass=type,
):
    r"""
    Visualization interface.
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
        minibatch: bool
        optimizer: bool
        regularization: bool
        self.show_init: bool

        # Create training console arguments.
        self.console = argparse.ArgumentParser(
            description="Homework 2 (Visualization)",
        )
        self.console.add_argument(
            "--minibatch",
            action="store_true",
            help="Render minibatch plots.",
        )
        self.console.add_argument(
            "--optimizer",
            action="store_true",
            help="Render optimizer plots.",
        )
        self.console.add_argument(
            "--regularization",
            action="store_true",
            help="Render regularization plots.",
        )
        self.console.add_argument(
            "--show-init",
            action="store_true",
            help="Show the initialization epoch (epoch 0).",
        )

        # Parse the command line arguments.
        self.args = self.console.parse_args()
        minibatch = self.args.minibatch
        optimizer = self.args.optimizer
        regularization = self.args.regularization
        self.show_init = self.args.show_init

        # Allocate a directory.
        os.makedirs(
            "images",
            exist_ok=True,
        )

        # Use notebook theme.
        sns.set_theme(context="notebook")

        # Visualize minibatch performance.
        if (minibatch):
            for abbr, full in [("Acc", "Accuracy"), ("Loss", "Cross Entropy")]:
                self.visualize_minibatch(
                    abbr=abbr, full=full,
                    num_samples=-1, random_seed=0,
                    normalize=True, shuffle=False,
                    kernel=5, stride=1, ginvariant=False,
                    cnn=False, amprec=False,
                    optim_alg="sgd", wd=0,
                )
        else:
            pass

        # Visualize optimizer performance.
        if (optimizer):
            for abbr, full in [("Acc", "Accuracy"), ("Loss", "Cross Entropy")]:
                self.visualize_optimizer(
                    abbr=abbr, full=full,
                    num_samples=-1, random_seed=0,
                    normalize=True, shuffle=False, batch_size=300,
                    kernel=5, stride=1, ginvariant=False,
                    cnn=False, amprec=False,
                    lr=1e-4, wd=0,
                )
        else:
            pass

        # Visualize L2 regularization performance.
        if (regularization):
            for abbr, full in [("Acc", "Accuracy"), ("Loss", "Cross Entropy")]:
                self.visualize_l2_regularization(
                    abbr=abbr, full=full,
                    num_samples=-1, random_seed=0,
                    normalize=True, shuffle=False, batch_size=300,
                    kernel=5, stride=1, ginvariant=False,
                    cnn=False, amprec=False,
                    optim_alg="sgd", lr=1e-4,
                )
        else:
            pass

    def get_title(
        self,
        /,
        *,
        num_samples: int, random_seed: int,
        normalize: bool, shuffle: bool, batch_size: int,
        kernel: int, stride: int, ginvariant: bool, cnn: bool, amprec: bool,
        optim_alg: str, lr: float, wd: float,
    ) -> str:
        r"""
        Get log title.

        Args
        ----
        - num_samples :
            Number of used training samples.
        - random_seed :
            Random seed.
        - normalize :
            Normalize images.
        - shuffle :
            Shuffle labels.
        - batch_size :
            Batch size.
        - kernel :
            Kernel size for CNN.
        - stride :
            Stride size for CNN.
        - ginvariant :
            Use G-Invariance.
        - cnn :
            Use CNN.
        - amprec :
            Use Automatical Mixed Precision.
        - optim_alg :
            Optimizer algorithm.
        - lr :
            Learning rate.
        - wd :
            Weight decay.

        Returns
        -------
        - title :
            Title of given configuration.
        """
        # Get title directly.
        if (cnn):
            return (
                "mnist_{:d}-{:d}-{:s}-{:d}-cnn_{:d}_{:d}-{:s}-{:s}-{:s}" \
                "-{:s}".format(
                    num_samples, random_seed,
                    # "n" if (normalize) else "0",
                    "s" if (shuffle) else "0",
                    batch_size,
                    kernel, stride,
                    "m" if (amprec) else "0",
                    optim_alg,
                    "{:.6f}".format(lr).rstrip("0"),
                    "{:.6f}".format(wd).rstrip("0"),
                )
            )
        else:
            return (
                "mnist_{:d}-{:d}-{:s}-{:d}-mlp_{:s}-{:s}-{:s}-{:s}" \
                "-{:s}".format(
                    num_samples, random_seed,
                    # "n" if (normalize) else "0",
                    "s" if (shuffle) else "0",
                    batch_size,
                    "g" if (ginvariant) else "0",
                    "m" if (amprec) else "0",
                    optim_alg,
                    "{:.6f}".format(lr).rstrip("0"),
                    "{:.6f}".format(wd).rstrip("0"),
                )
            )

    def visualize_minibatch(
        self,
        /,
        *,
        abbr: str, full: str,
        **shared: Any,
    ) -> None:
        r"""
        Visualize optimizer.

        Args
        ----
        - abbr :
            Rendering metric abbreviation.
        - full
            Rendering metric full name.
        - shared :
            Shared configuration.

        Returns
        -------
        """
        # /
        # ANNOTATE
        # /
        batch_size: str
        lr: str
        path: str
        df_buf: List[pd.DataFrame]
        df: pd.DataFrame
        df_train: pd.DataFrame
        df_valid: pd.DataFrame
        df_test: pd.DataFrame
        grid: sns.axisgrid.FacetGrid
        i: int
        ax: mpl.axes.Axes
        ticklabel: mpl.text.Text

        # Load all logs.
        df_buf = []
        for batch_size in ("100", "500", "3000", "5000"):
            for lr in ("1e-3", "1e-4", "1e-5"):
                # Load log as a data frame.
                path = os.path.join(
                    "logs",
                    self.get_title(
                        batch_size=int(batch_size), lr=float(lr),
                        **shared,
                    ),
                    "log.pt",
                )
                df = pd.DataFrame(torch.load(path))

                # Hide the initialization.
                if (self.show_init):
                    df["Epoch"] = list(range(len(df)))
                else:
                    df = df[1:]
                    df["Epoch"] = list(range(1, len(df) + 1))

                # Extend loaded data.
                df["Batch Size"] = batch_size
                df["Learning Rate"] = lr

                # Focus specific data columns.
                df_train = df[
                    [
                        "Epoch", "Train {:s}".format(abbr),
                        "Batch Size", "Learning Rate",
                    ]
                ].copy().rename(columns={"Train {:s}".format(abbr): full})
                df_valid = df[
                    [
                        "Epoch", "Validate {:s}".format(abbr),
                        "Batch Size", "Learning Rate",
                    ]
                ].copy().rename(columns={"Validate {:s}".format(abbr): full})
                df_test = df[
                    [
                        "Epoch", "Test {:s}".format(abbr),
                        "Batch Size", "Learning Rate",
                    ]
                ].copy().rename(columns={"Test {:s}".format(abbr): full})

                # Extend focusing data.
                df_train["Usage"] = "Training"
                df_valid["Usage"] = "Validation"
                df_test["Usage"] = "Test"

                # Save processed data frame.
                df_buf.append(df_train)
                df_buf.append(df_valid)
                df_buf.append(df_test)
        df = pd.concat(df_buf)

        # Render the plot.
        grid = sns.relplot(
            data=df, x="Epoch", y=full, row="Batch Size", col="Learning Rate",
            hue="Usage",
            kind="line",
        )

        # Reset axis.
        for i, ax in enumerate(grid.axes.flat):
            ax.set_title(
                ax.get_title(),
                fontsize=15,
            )
            ax.set_xlabel(
                ax.get_xlabel(),
                fontsize=15,
            )
            if (i % 3 == 0):
                ax.set_ylabel(
                    ax.get_ylabel(),
                    fontsize=15,
                )
            else:
                pass
            for ticklabel in ax.get_xticklabels():
                ticklabel.set_fontsize(12)
            for ticklabel in ax.get_yticklabels():
                ticklabel.set_fontsize(12)

        # Reset legend.
        plt.setp(
            grid.legend.get_title(),
            fontsize=15,
        )
        plt.setp(
            grid.legend.get_texts(),
            fontsize=12,
        )

        # Save.
        grid.fig.subplots_adjust(wspace=0.08, hspace=0.08)
        grid.savefig(
            os.path.join("images", "minibatch-{:s}.png".format(abbr.lower())),
        )
        plt.close("all")

    def visualize_optimizer(
        self,
        /,
        *,
        abbr: str, full: str,
        **shared: Any,
    ) -> None:
        r"""
        Visualize optimizer.

        Args
        ----
        - abbr :
            Rendering metric abbreviation.
        - full
            Rendering metric full name.
        - shared :
            Shared configuration.

        Returns
        -------
        """
        # /
        # ANNOTATE
        # /
        optimizer: str
        path: str
        df_buf: List[pd.DataFrame]
        df: pd.DataFrame
        df_train: pd.DataFrame
        df_valid: pd.DataFrame
        df_test: pd.DataFrame
        grid: sns.axisgrid.FacetGrid
        i: int
        ax: mpl.axes.Axes
        ticklabel: mpl.text.Text

        # Load all logs.
        df_buf = []
        for optimizer in ("sgd", "momentum", "nesterov", "adam"):
            # Load log as a data frame.
            path = os.path.join(
                "logs",
                self.get_title(
                    optim_alg=optimizer,
                    **shared,
                ),
                "log.pt",
            )
            df = pd.DataFrame(torch.load(path))

            # Hide the initialization.
            if (self.show_init):
                df["Epoch"] = list(range(len(df)))
            else:
                df = df[1:]
                df["Epoch"] = list(range(1, len(df) + 1))

            # Extend loaded data.
            df["Optimizer"] = optimizer

            # Focus specific data columns.
            df_train = df[
                ["Epoch", "Train {:s}".format(abbr), "Optimizer"]
            ].copy().rename(columns={"Train {:s}".format(abbr): full})
            df_valid = df[
                ["Epoch", "Validate {:s}".format(abbr), "Optimizer"]
            ].copy().rename(columns={"Validate {:s}".format(abbr): full})
            df_test = df[
                ["Epoch", "Test {:s}".format(abbr), "Optimizer"]
            ].copy().rename(columns={"Test {:s}".format(abbr): full})

            # Extend focusing data.
            df_train["Usage"] = "Training"
            df_valid["Usage"] = "Validation"
            df_test["Usage"] = "Test"

            # Save processed data frame.
            df_buf.append(df_train)
            df_buf.append(df_valid)
            df_buf.append(df_test)
        df = pd.concat(df_buf)

        # Render the plot.
        grid = sns.relplot(
            data=df, x="Epoch", y=full, col="Usage", hue="Optimizer",
            col_wrap=3, kind="line",
        )

        # Reset axis.
        for i, ax in enumerate(grid.axes.flat):
            ax.set_title(
                ax.get_title(),
                fontsize=15,
            )
            ax.set_xlabel(
                ax.get_xlabel(),
                fontsize=15,
            )
            if (i % 3 == 0):
                ax.set_ylabel(
                    ax.get_ylabel(),
                    fontsize=15,
                )
            else:
                pass
            for ticklabel in ax.get_xticklabels():
                ticklabel.set_fontsize(12)
            for ticklabel in ax.get_yticklabels():
                ticklabel.set_fontsize(12)

        # Reset legend.
        plt.setp(
            grid.legend.get_title(),
            fontsize=15,
        )
        plt.setp(
            grid.legend.get_texts(),
            fontsize=12,
        )

        # Save.
        grid.fig.subplots_adjust(wspace=0.05, hspace=0.05)
        grid.savefig(
            os.path.join("images", "optimizer-{:s}.png".format(abbr.lower())),
        )
        plt.close("all")

    def visualize_l2_regularization(
        self,
        /,
        *,
        abbr: str, full: str,
        **shared: Any,
    ) -> None:
        r"""
        Visualize L2 regularization.

        Args
        ----
        - abbr :
            Rendering metric abbreviation.
        - full
            Rendering metric full name.
        - shared :
            Shared configuration.

        Returns
        -------
        """
        # /
        # ANNOTATE
        # /
        l2_lambda: str
        path: str
        df_buf: List[pd.DataFrame]
        df: pd.DataFrame
        df_train: pd.DataFrame
        df_valid: pd.DataFrame
        df_test: pd.DataFrame
        grid: sns.axisgrid.FacetGrid
        i: int
        ax: mpl.axes.Axes
        ticklabel: mpl.text.Text

        # Load all logs.
        df_buf = []
        for l2_lambda in ("1", "0.1", "0.01"):
            # Load log as a data frame.
            path = os.path.join(
                "logs",
                self.get_title(
                    wd=float(l2_lambda),
                    **shared,
                ),
                "log.pt",
            )
            df = pd.DataFrame(torch.load(path))

            # Hide the initialization.
            if (self.show_init):
                df["Epoch"] = list(range(len(df)))
            else:
                df = df[1:]
                df["Epoch"] = list(range(1, len(df) + 1))

            # Extend loaded data.
            df["L2 Regularization"] = l2_lambda

            # Focus specific data columns.
            df_train = df[
                ["Epoch", "Train {:s}".format(abbr), "L2 Regularization"]
            ].copy().rename(columns={"Train {:s}".format(abbr): full})
            df_valid = df[
                ["Epoch", "Validate {:s}".format(abbr), "L2 Regularization"]
            ].copy().rename(columns={"Validate {:s}".format(abbr): full})
            df_test = df[
                ["Epoch", "Test {:s}".format(abbr), "L2 Regularization"]
            ].copy().rename(columns={"Test {:s}".format(abbr): full})

            # Extend focusing data.
            df_train["Usage"] = "Training"
            df_valid["Usage"] = "Validation"
            df_test["Usage"] = "Test"

            # Save processed data frame.
            df_buf.append(df_train)
            df_buf.append(df_valid)
            df_buf.append(df_test)
        df = pd.concat(df_buf)

        # Render the plot.
        grid = sns.relplot(
            data=df, x="Epoch", y=full, col="Usage", hue="L2 Regularization",
            col_wrap=3, kind="line",
        )

        # Reset axis.
        for i, ax in enumerate(grid.axes.flat):
            ax.set_title(
                ax.get_title(),
                fontsize=15,
            )
            ax.set_xlabel(
                ax.get_xlabel(),
                fontsize=15,
            )
            if (i % 3 == 0):
                ax.set_ylabel(
                    ax.get_ylabel(),
                    fontsize=15,
                )
            else:
                pass
            for ticklabel in ax.get_xticklabels():
                ticklabel.set_fontsize(12)
            for ticklabel in ax.get_yticklabels():
                ticklabel.set_fontsize(12)

        # Reset legend.
        plt.setp(
            grid.legend.get_title(),
            fontsize=15,
        )
        plt.setp(
            grid.legend.get_texts(),
            fontsize=12,
        )

        # Save.
        grid.fig.subplots_adjust(wspace=0.05, hspace=0.05)
        grid.savefig(
            os.path.join(
                "images", "regularization-{:s}.png".format(abbr.lower()),
            ),
        )
        plt.close("all")


# Run.
if (__name__ == "__main__"):
    Visualize()
else:
    pass