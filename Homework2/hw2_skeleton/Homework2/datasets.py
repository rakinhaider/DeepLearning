# All imports.
from typing import Any
from typing import ClassVar
from typing import Tuple, List, Dict
from typing import Iterator
import torch
import re
import os


# =============================================================================
# *****************************************************************************
# -----------------------------------------------------------------------------
# < Memories >
# -----------------------------------------------------------------------------
# *****************************************************************************
# =============================================================================


# /
# ANNOTATE
# /
ELETYPE_MNIST: Any


# Define MNIST element type.
ELETYPE_MNIST = Tuple[torch.Tensor, torch.Tensor]


class MNISTDataset(
    object,
    metaclass=type,
):
    r"""
    MNIST Dataset.
    """
    # /
    # ANNOTATE
    # /
    COLORS: ClassVar[Dict[str, str]]
    PALATTE: ClassVar[List[str]]
    # -----
    HEIGHT: ClassVar[int]
    WIDTH: ClassVar[int]
    NUM_LABELS: ClassVar[int]
    # -----
    TRAIN: ClassVar[int]
    TEST: ClassVar[int]

    # Define representation colors
    COLORS = dict(
        orange="31",
        teal="32",
        olive="33",
        cadet="34",
        purple="35",
        navy="36",
        red="91",
        green="92",
        yellow="93",
        blue="94",
        pink="95",
        cyan="96",
    )
    PALATTE = [
        COLORS["red"], COLORS["green"], COLORS["yellow"], COLORS["blue"],
        COLORS["orange"], COLORS["purple"], COLORS["cyan"], COLORS["pink"],
        COLORS["teal"], COLORS["olive"], COLORS["cadet"], COLORS["navy"],
    ]

    # Constant properties.
    HEIGHT = 28
    WIDTH = 28
    NUM_PIXELS = HEIGHT * WIDTH
    NUM_LABELS = 10
    TRAIN = 0
    TEST = 1

    def __init__(
        self,
        /,
        root: str,
        *,
        normalize: bool, num_samples: int,
        shuffle: bool, random_seed: int,
    ) -> None:
        r"""
        Initialize the class.

        Args
        ----
        - root :
            Root directory holding raw data.
        - normalize :
            Do mean and standard deviation normalization.
        - num_samples :
            Number of training samples to use.
        - shuffle :
            Random shuffle labels.
        - random_seed :
            Random seed used for shuffling.

        Returns
        -------
        """
        # /
        # ANNOTATE
        # /
        train_images: torch.Tensor
        train_labels: torch.Tensor
        test_images: torch.Tensor
        test_labels: torch.Tensor
        train_mean: torch.Tensor
        train_std: torch.Tensor
        rng: torch.Generator
        self.memory: List[List[ELETYPE_MNIST]]

        # Load.
        (
            (train_images, train_labels),
            (test_images, test_labels),
        ) = torch.load(os.path.join(root, "mnist.ptds"))

        # Check memory format.
        assert (
            train_images.size()[1:] == (self.HEIGHT, self.WIDTH)
        ), "[\033[91mError\033[0m]: Training images have improper shape."
        assert (
            test_images.size()[1:] == (self.HEIGHT, self.WIDTH)
        ), "[\033[91mError\033[0m]: Test images have improper shape."

        # Formalize shape.
        train_images = train_images.view(
            len(train_images), self.HEIGHT, self.WIDTH,
        ).to(torch.get_default_dtype())
        train_labels = train_labels.view(len(train_labels), 1).long()
        test_images = test_images.view(
            len(test_images), self.HEIGHT, self.WIDTH,
        ).to(torch.get_default_dtype())
        test_labels = test_labels.view(len(test_labels), 1).long()

        # Normalize (Standardize).
        if (normalize):
            train_mean = torch.mean(train_images)
            train_std = torch.std(train_images)
            train_images = (train_images - train_mean) / train_std
            test_images = (test_images - train_mean) / train_std
        else:
            pass

        # Truncate.
        if (num_samples < 0):
            pass
        else:
            train_images = train_images[0:num_samples]
            train_labels = train_labels[0:num_samples]

        # Randomly shuffle labels.
        if (shuffle):
            rng = torch.Generator("cpu")
            rng.manual_seed(random_seed)
            train_labels = train_labels[
                torch.randperm(
                    len(train_labels),
                    generator=rng,
                )
            ]
            test_labels = test_labels[
                torch.randperm(
                    len(test_labels),
                    generator=rng,
                )
            ]
        else:
            pass

        # Save to memory.
        self.memory = [
            list(zip(train_images, train_labels)),
            list(zip(test_images, test_labels)),
        ]

    def __iter__(
        self,
        /,
    ) -> Iterator[ELETYPE_MNIST]:
        r"""
        Get an iterator.

        Args
        ----

        Returns
        -------
        - iterator :
            Iterator.
        """
        # Iterate on the memory.
        return iter(self.memory)

    def __getitem__(
        self,
        /,
        iu: Tuple[int, int],
    ) -> ELETYPE_MNIST:
        r"""
        Index an element.

        Args
        ----
        - iu :
            Element and usage index.
        - usage :
            Usage index.

        Returns
        -------
        - element :
            Element.
        """
        # /
        # ANNOTATE
        # /
        i: int
        usage: int

        # Directly locate in the memory.
        i, usage = iu
        return self.memory[usage][i]

    def __len__(
        self,
        /,
    ) -> int:
        r"""
        Get length.

        Args
        ----

        Returns
        -------
        - length :
            Length.
        """
        # Get length directly.
        return len(self.memory[self.TRAIN]) + len(self.memory[self.TEST])

    # -------------------------------------------------------------------------
    # < Generic Representation >
    # Generate representation string of memory by recursion.
    # -------------------------------------------------------------------------

    def __repr__(
        self,
        /,
    ) -> str:
        r"""
        Get representation string.

        Args
        ----

        Returns
        -------
        - msg :
            Representation string.
        """
        return self.repr(
            self.memory,
            depth=0,
        )

    @classmethod
    def repr(
        cls,
        /,
        item: Any,
        *,
        depth: int,
    ) -> str:
        r"""
        Recursively get representation string.

        Args
        ----
        - item :
            Item to generate representation string.

        Returns
        -------
        - msg :
            Representation string for given item.
        """
        # Check each case.
        if (isinstance(item, int)):
            return cls.colorful("1", depth)
        elif (isinstance(item, float)):
            return cls.colorful("1", depth)
        elif (isinstance(item, torch.Tensor)):
            return cls.colorful(
                "x".join(str(d) for d in item.size()),
                depth,
            )
        elif (isinstance(item, list)):
            left = "{:d}[".format(len(item))
            right = ", ...]"
            return "{:s}{:s}{:s}".format(
                cls.colorful(left, depth),
                cls.repr(
                    next(iter(item)),
                    depth=depth + 1,
                ),
                cls.colorful(right, depth),
            )
        elif (isinstance(item, tuple)):
            left = "("
            middle = ", "
            right = ")"
            return "{:s}{:s}{:s}".format(
                cls.colorful(left, depth),
                cls.colorful(middle, depth).join(
                    cls.repr(
                        itr,
                        depth=depth + 1,
                    ) for itr in item
                ),
                cls.colorful(right, depth),
            )
        else:
            print(
                "[\033[91mError\033[0m]: Unknown memory unit type \"{:s}\"."
                .format(str(type(item))),
            )
            raise RuntimeError

    @classmethod
    def colorful(
        cls,
        /,
        msg: str,
        palette: int,
    ) -> str:
        r"""
        Wrap string with color.

        Args
        ----
        - msg :
            Message string.

        Returns
        -------
        - msg :
            Colorful message string.
        """
        # Wrap with color style string.
        return "\033[{:s}m{:s}\033[0m".format(
            cls.PALATTE[palette % len(cls.PALATTE)], msg,
        )

    @classmethod
    def decolor(
        cls,
        /,
        msg: str,
    ) -> str:
        r"""
        Remove color from string.

        Args
        ----
        - msg :
            Message string.

        Returns
        -------
        - msg :
            Decolorized message string.
        """
        # Remove color style string.
        return re.sub(r"\033\[[^m]+m", "", msg)