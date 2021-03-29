# All imports
import argparse
import os
import pandas as pd
import torch
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Any
from typing import List, Dict
from typing import Union


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
        layer: bool
        janossy: bool
        t_bptt: bool
        self.show_init: bool

        # Create training console arguments.
        self.console = argparse.ArgumentParser(
            description="Homework 2 (Visualization)",
        )
        self.console.add_argument(
            "--layer",
            action="store_true",
            help="Render GNN layer plots.",
        )
        self.console.add_argument(
            "--janossy",
            action="store_true",
            help="Render Janossy Pooling plots.",
        )
        self.console.add_argument(
            "--t-bptt",
            action="store_true",
            help="Render truncated BPTT plots.",
        )
        self.console.add_argument(
            "--show-init",
            action="store_true",
            help="Show the initialization epoch (epoch 0).",
        )

        # Parse the command line arguments.
        self.args = self.console.parse_args()
        layer = self.args.layer
        janossy = self.args.janossy
        t_bptt = self.args.t_bptt
        self.show_init = self.args.show_init

        # Allocate a directory.
        os.makedirs(
            "images",
            exist_ok=True,
        )

        # Use notebook theme.
        sns.set_theme(context="notebook")

        # Visualize layer performance.
        if (layer):
            for abbr, full in [("Loss", "Cross Entropy"), ("Acc", "Accuracy")]:
                self.visualize_layer(
                    abbr=abbr, full=full,
                    cora=True, ptb=False, random_seed=0, dense=True,
                    janossy=False, num_perms=0,
                    markov=False, trunc=0, num_internals=16,
                )
        else:
            pass

        # Visualize Janossy Pooling performance.
        if (janossy):
            for abbr, full in [("Loss", "Cross Entropy"), ("Acc", "Accuracy")]:
                self.visualize_janossy(
                    abbr=abbr, full=full,
                    cora=True, ptb=False, random_seed=0, dense=False,
                    num_graph_layers=2,
                    markov=False, trunc=0, num_internals=16,
                )
        else:
            pass

        # Visualize T-BPTT performance.
        if (t_bptt):
            for abbr, full in [("Loss", "Perplexity")]:
                self.visualize_t_bptt(
                    abbr=abbr, full=full,
                    cora=False, ptb=True, random_seed=0, dense=False,
                    janossy=False, num_perms=0, num_graph_layers=0,
                    markov=False, num_internals=128,
                )
        else:
            pass

    def get_title(
        self,
        /,
        *,
        cora: bool, ptb: bool, random_seed: int, dense: bool,
        janossy: bool, num_perms: int, num_graph_layers: int,
        markov: bool, trunc: int, num_internals: int,
    ) -> str:
        r"""
        Get log title.

        Args
        ----
        - cora :
            Work on Cora dataset.
        - ptb :
            Work on PTB dataset.
        - random_seed :
            Random seed.
        - dense :
            Work on dense Cora dataset.
        - janossy :
            Use Janossy Pooling on Cora dataset.
        - num_perms :
            Number of sampled permutations for Janossy Pooling.
        - num_graph_layers :
            Number of graph convolution layers.
        - markov :
            Use Markov model on PTB dataset.
        - trunc :
            Length of truncated BPTT chunk.
        - num_internals :
            Internal size of language models.

        Returns
        -------
        - title :
            Title of given configuration.
        """
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # Get title directly.
        if (cora):
            return (
                "{task:s}-{random_seed:d}-{dense:s}" \
                "-{janossy:s}-{num_perms:d}-{num_graph_layers:d}".format(
                    task="cora", random_seed=random_seed,
                    dense="d" if (dense) else "s",
                    janossy="jp" if (janossy) else "gc",
                    num_perms=num_perms, num_graph_layers=num_graph_layers,
                )
            )
        elif (ptb):
            return (
                "{task:s}-{random_seed:d}" \
                "-{markov:s}-{trunc:d}-{num_internals:d}".format(
                    task="ptb", random_seed=random_seed,
                    markov="mrkv" if (markov) else "lstm",
                    trunc=trunc, num_internals=num_internals,
                )
            )
        else:
            print(
                "[\033[91mError\033[0m]: Reach impossible branch."
            )
            raise NotImplementedError

    @staticmethod
    def pseudo_early_stop(
        log: Dict[str, List[float]],
        on: str, scale: int,
        /,
    ) -> Dict[str, List[float]]:
        r"""
        Apply pesudo early stopping on given log.

        Args
        ----
        - log :
            Log data.
        - on :
            Criterion for early stopping.
        - scale :
            Scalue on early stopping criterion.

        Returns
        -------
        - log :
            Updated log data.
        """
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # /
        # ANNOTATE
        # /
        col_valid: List[float]
        best_valid: float

        # Get early stopping column.
        col_valid = log["Validate {:s}".format(on)].copy()
        best_valid = scale * col_valid[0]

        # Update current log.
        for i in range(1, len(col_valid)):
            if (scale * col_valid[i] < best_valid):
                best_valid = scale * col_valid[i]
            else:
                for key in log.keys():
                    log[key][i] = log[key][i - 1]
        return log

    def visualize_layer(
        self,
        /,
        *,
        abbr: str, full: str,
        **shared: Any,
    ) -> None:
        r"""
        Visualize layer.

        Args
        ----
        - abbr :
            Rendering metric abbreviation.
        - full :
            Rendering metric full name.
        - shared :
            Shared configuration.

        Returns
        -------
        """
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # /
        # ANNOTATE
        # /
        layer: str
        path: str
        df_buf: List[pd.DataFrame]
        df: pd.DataFrame
        df_valid: pd.DataFrame
        df_test: pd.DataFrame
        grid: sns.axisgrid.FacetGrid
        i: int
        ax: mpl.axes.Axes
        ticklabel: mpl.text.Text

        # Load all logs.
        df_buf = []
        for layer in ("2", "20"):
            # Load log as a data frame.
            path = os.path.join(
                "logs",
                self.get_title(
                    num_graph_layers=int(layer),
                    **shared,
                ),
                "log.pt",
            )
            df = pd.DataFrame(
                self.pseudo_early_stop(torch.load(path), "Acc", -1),
            )

            # Hide the initialization.
            if (self.show_init):
                df["Epoch"] = list(range(len(df)))
            else:
                df = df[1:]
                df["Epoch"] = list(range(1, len(df) + 1))

            # Extend loaded data.
            df["Layer"] = layer

            # Focus specific data columns.
            df_valid = df[
                [
                    "Epoch", "Validate {:s}".format(abbr),
                    "Layer",
                ]
            ].copy().rename(columns={"Validate {:s}".format(abbr): full})
            df_test = df[
                [
                    "Epoch", "Test {:s}".format(abbr),
                    "Layer",
                ]
            ].copy().rename(columns={"Test {:s}".format(abbr): full})

            # Extend focusing data.
            df_valid["Usage"] = "Validation"
            df_test["Usage"] = "Test"

            # Save processed data frame.
            df_buf.append(df_valid)
            df_buf.append(df_test)
        df = pd.concat(df_buf)

        # Render the plot.
        grid = sns.relplot(
            data=df, x="Epoch", y=full, col="Usage", col_wrap=2,
            hue="Layer",
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
            if (i % 2 == 0):
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
            os.path.join("images", "layer-{:s}.png".format(abbr.lower())),
        )
        plt.close("all")

    def visualize_janossy(
        self,
        /,
        *,
        abbr: str, full: str,
        **shared: Any,
    ) -> None:
        r"""
        Visualize Janossy Pooling.

        Args
        ----
        - abbr :
            Rendering metric abbreviation.
        - full :
            Rendering metric full name.
        - shared :
            Shared configuration.

        Returns
        -------
        """
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # /
        # ANNOTATE
        # /
        name: str
        specific: Dict[str, Union[bool, int]]
        path: str
        df_buf: List[pd.DataFrame]
        df: pd.DataFrame
        df_valid: pd.DataFrame
        df_test: pd.DataFrame
        grid: sns.axisgrid.FacetGrid
        i: int
        ax: mpl.axes.Axes
        ticklabel: mpl.text.Text

        # Load all logs.
        df_buf = []
        for name, specific in (
            ("GCN", dict(janossy=False, num_perms=0)),
            ("Janossy 1", dict(janossy=True, num_perms=1)),
            ("Janossy 20", dict(janossy=True, num_perms=20)),
        ):
            # Load log as a data frame.
            path = os.path.join(
                "logs",
                self.get_title(
                    **specific,
                    **shared,
                ),
                "log.pt",
            )
            df = pd.DataFrame(
                self.pseudo_early_stop(torch.load(path), "Acc", -1),
            )

            # Hide the initialization.
            if (self.show_init):
                df["Epoch"] = list(range(len(df)))
            else:
                df = df[1:]
                df["Epoch"] = list(range(1, len(df) + 1))

            # Extend loaded data.
            df["Model"] = name

            # Focus specific data columns.
            df_valid = df[
                [
                    "Epoch", "Validate {:s}".format(abbr),
                    "Model",
                ]
            ].copy().rename(columns={"Validate {:s}".format(abbr): full})
            df_test = df[
                [
                    "Epoch", "Test {:s}".format(abbr),
                    "Model",
                ]
            ].copy().rename(columns={"Test {:s}".format(abbr): full})

            # Extend focusing data.
            df_valid["Usage"] = "Validation"
            df_test["Usage"] = "Test"

            # Save processed data frame.
            df_buf.append(df_valid)
            df_buf.append(df_test)
        df = pd.concat(df_buf)

        # Render the plot.
        grid = sns.relplot(
            data=df, x="Epoch", y=full, col="Usage", col_wrap=2,
            hue="Model",
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
            if (i % 2 == 0):
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
            os.path.join("images", "janossy-{:s}.png".format(abbr.lower())),
        )
        plt.close("all")

    def visualize_t_bptt(
        self,
        /,
        *,
        abbr: str, full: str,
        **shared: Any,
    ) -> None:
        r"""
        Visualize truncated BPTT.

        Args
        ----
        - abbr :
            Rendering metric abbreviation.
        - full :
            Rendering metric full name.
        - shared :
            Shared configuration.

        Returns
        -------
        """
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # /
        # ANNOTATE
        # /
        trunc: str
        path: str
        df_buf: List[pd.DataFrame]
        df: pd.DataFrame
        df_valid: pd.DataFrame
        df_test: pd.DataFrame
        grid: sns.axisgrid.FacetGrid
        i: int
        ax: mpl.axes.Axes
        ticklabel: mpl.text.Text

        # Load all logs.
        df_buf = []
        for trunc in ("5", "35", "80"):
            # Load log as a data frame.
            path = os.path.join(
                "logs",
                self.get_title(
                    trunc=int(trunc),
                    **shared,
                ),
                "log.pt",
            )
            df = pd.DataFrame(
                self.pseudo_early_stop(torch.load(path), "Loss", 1),
            )

            # Hide the initialization.
            if (self.show_init):
                df["Epoch"] = list(range(len(df)))
            else:
                df = df[1:]
                df["Epoch"] = list(range(1, len(df) + 1))

            # Extend loaded data.
            df["Truncate"] = trunc

            # Focus specific data columns.
            df_valid = df[
                [
                    "Epoch", "Validate {:s}".format(abbr),
                    "Truncate",
                ]
            ].copy().rename(columns={"Validate {:s}".format(abbr): full})
            df_test = df[
                [
                    "Epoch", "Test {:s}".format(abbr),
                    "Truncate",
                ]
            ].copy().rename(columns={"Test {:s}".format(abbr): full})

            # Extend focusing data.
            df_valid["Usage"] = "Validation"
            df_test["Usage"] = "Test"

            # Save processed data frame.
            df_buf.append(df_valid)
            df_buf.append(df_test)
        df = pd.concat(df_buf)

        # Render the plot.
        grid = sns.relplot(
            data=df, x="Epoch", y=full, col="Usage", col_wrap=2,
            hue="Truncate",
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
            if (i % 2 == 0):
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
            os.path.join("images", "t_bptt-{:s}.png".format(abbr.lower())),
        )
        plt.close("all")


# Run.
if (__name__ == "__main__"):
    Visualize()
else:
    pass