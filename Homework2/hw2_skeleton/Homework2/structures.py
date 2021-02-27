# All imports.
from typing import Any
from typing import ClassVar
from typing import Tuple, List, Dict
import torch
import numpy as onp
from sklearn.model_selection import StratifiedKFold
from datasets import ELETYPE_MNIST
from datasets import MNISTDataset


# =============================================================================
# *****************************************************************************
# -----------------------------------------------------------------------------
# < Datasets >
# -----------------------------------------------------------------------------
# *****************************************************************************
# =============================================================================


class MNISTPerturbDataStructure(
    torch.utils.data.Dataset,
    metaclass=type,
):
    r"""
    MNIST Data Structure with Perturbations.
    Perturbations are rotations and flippings.
    """
    # /
    # ANNOTATE
    # /
    TRAIN: ClassVar[int]
    VALIDATE: ClassVar[int]
    TEST: ClassVar[int]

    # Train/Validation/Test split constants.
    TRAIN = 0
    VALIDATE = 1
    TEST = 2

    def __init__(
        self,
        /,
        memory: MNISTDataset, kfold_split: Tuple[int, int, int],
        *,
        kfold_index: Tuple[int, int], kfold_usage: int,
        perturbate: bool, random_seed: int,
    ) -> None:
        r"""
        Initialize the class.

        Args
        ----
        - memory :
            In-memory data.
        - kfold_split :
            Kfold split proportion of training, validation and test.
            MNIST has defined test split.
        - kfold_index :
            Kfold split index of validation and test.
            MNIST has defined test split.
        - kfold_usage :
            Fold usage.
        - perturbate :
            If True, an image will be randomly perturbated when it is sampled.
            If False, it will keep the same as original data.
        - random_seed :
            Random seed used to perturbation.

        Returns
        -------
        """
        # /
        # ANNOTATE
        # /
        self.memory: MNISTDataset
        self.kfold_split: Tuple[int, int, int]
        self.kfold_index: Tuple[int, int]
        self.kfold_usage: int
        self.virtual_split: onp.ndarray
        perturbs: onp.ndarray
        self.perturb_degrees: onp.ndarray
        self.perturb_numbers: onp.ndarray

        # Save necessary attributes.
        self.memory = memory
        self.kfold_split = kfold_split
        self.kfold_index = kfold_index
        self.kfold_usage = kfold_usage

        # Get memory virtual split table.
        self.virtual_split = self.get_virtual_split_table()

        # Generate a random or not-changing pertubations.
        if (perturbate):
            perturbs = onp.random.default_rng(random_seed).integers(
                low=1, high=16, size=(len(self.memory),)
            ).astype(int)
        else:
            perturbs = onp.zeros((len(self.memory),)).astype(int)
        self.perturb_degrees = perturbs % 4
        self.perturb_numbers = perturbs // 4

    def __getitem__(
        self,
        /,
        i: int,
    ) -> ELETYPE_MNIST:
        r"""
        Index an element.

        Args
        ----
        - i :
            Element index.

        Returns
        -------
        - element :
            Element.
        """
        # /
        # ANNOTATE
        # /
        index: int
        image: torch.Tensor
        label: torch.Tensor

        # Get image and label.
        index = self.virtual_split[i]
        if (self.kfold_usage == self.TEST):
            image, label = self.memory[(index, self.memory.TEST)]
        else:
            image, label = self.memory[(index, self.memory.TRAIN)]

        # Perturb if it is required.
        image = torch.from_numpy(
            flip(
                rotate(
                    image.numpy(),
                    degree=self.perturb_degrees[index],
                ),
                number=self.perturb_numbers[index],
            ).copy()
        ).to(image.dtype)
        return image, label

    def __len__(
        self,
        /,
    ) -> int:
        r"""
        Get length.
        """
        # Get length directly.
        return len(self.virtual_split)

    def get_virtual_split_table(
        self,
        /,
    ) -> onp.ndarray:
        r"""
        Get the virtual split table matching split index to memory index.

        Args
        ----

        Returns
        -------
        - virtual_split
            Virtual split table matching split index to memory index.
        """
        # /
        # ANNOTATE
        # /
        label: torch.Tensor
        # -----
        rest_inputs: onp.ndarray
        rest_labels: onp.ndarray
        test_inputs: onp.ndarray
        test_labels: onp.ndarray
        # -----
        cnt_valid: int
        cnt_rest: int
        cnt_test: int
        cnt_total: int
        # -----
        train_valid_op: StratifiedKFold
        # -----
        rest_test_index: int
        train_valid_index: int
        # -----
        train_indices: onp.ndarray
        valid_indices: onp.ndarray
        test_indices: onp.ndarray
        # -----
        train_split: onp.ndarray
        valid_split: onp.ndarray
        test_split: onp.ndarray

        # Get labels as a list.
        rest_inputs = onp.array(
            list(range(len(self.memory.memory[self.memory.TRAIN]))),
        )
        rest_labels = onp.array(
            [
                int(label.item())
                for _, label in self.memory.memory[self.memory.TRAIN]
            ],
        )
        test_inputs = onp.array(
            list(range(len(self.memory.memory[self.memory.TEST]))),
        )
        test_labels = onp.array(
            [
                int(label.item())
                for _, label in self.memory.memory[self.memory.TEST]
            ],
        )

        # Safety check.
        cnt_valid = self.kfold_split[self.VALIDATE]
        cnt_test = self.kfold_split[self.TEST]
        cnt_rest = self.kfold_split[self.TRAIN] + cnt_valid
        cnt_total = cnt_rest + cnt_test
        assert (
            cnt_total % cnt_test == 0
        ), "[\033[91mError\033[0m]: Test fold count {:d} should properly" \
           " divide total count {:d}.".format(cnt_test, cnt_total)
        assert (
            cnt_rest % cnt_valid == 0
        ), "[\033[91mError\033[0m]: Validation fold count {:d} should" \
           " properly divide rest count {:d}.".format(cnt_valid, cnt_rest)
        assert (
            cnt_rest == cnt_test * 6
        ), "[\033[91mError\033[0m]: Test fold and rest count must be" \
           " \"1:6\"."

        # Get fold split operaters.
        train_valid_op = StratifiedKFold(
            n_splits=cnt_rest // cnt_valid, shuffle=False,
        )

        # Get fold indices.
        rest_test_index, train_valid_index = self.kfold_index
        assert (
            rest_test_index == 0
        ), "[\033[91mError\033[0m]: Test fold is defined, thus rest-test" \
           " index must be 0."
        assert (
            train_valid_index < cnt_rest // cnt_valid
        ), "[\033[91mError\033[0m]: Training-validation index must be " \
           " smaller than {:d}.".format(cnt_rest // cnt_valid)

        # Get validation split.
        for _, (train_indices, valid_indices) in zip(
            range(train_valid_index + 1),
            iter(train_valid_op.split(rest_inputs, rest_labels)),
        ):
            pass

        # Get splits.
        train_split = rest_inputs[train_indices]
        valid_split = rest_inputs[valid_indices]
        test_split = test_inputs
        return (train_split, valid_split, test_split)[self.kfold_usage]


def mnist_dataset_collate(
    samples: List[ELETYPE_MNIST],
) -> Dict[str, torch.Tensor]:
    r"""
    Collate function for MNIST dataset.

    Args
    ----
    - samples :
        Samples to be collated as a batch.

    Returns
    -------
    - batch
        Batch.
    """
    # /
    # ANNOTATE
    # /
    buf_image: List[torch.Tensor]
    buf_label: List[torch.Tensor]
    image: torch.Tensor
    label: torch.Tensor

    # Stack samples.
    buf_image, buf_label = [], []
    for image, label in samples:
        buf_image.append(image.view(MNISTDataset.NUM_PIXELS).float())
        buf_label.append(label)
    return dict(
        image=torch.stack(buf_image), label=torch.cat(buf_label),
    )


def rotate(
    input: onp.ndarray,
    *,
    degree: int,
) -> onp.ndarray:
    r"""
    Rotate a batch of 2D matrices.

    Args
    ----
    - input :
        Input.
    - degree :
        Degree to rotate.
        Must be one of {0, 1, 2, 3}.

    Returns
    -------
    - output :
        Output.
    """
    # /
    # ANNOTATE
    # /
    output: onp.ndarray

    # Safety check.
    assert (
        input.ndim == 2
    ), "[\033[91mError\033[0m]: Operation input must be a 2D matrix."
    assert (
        degree in (0, 1, 2, 3)
    ), "[\033[91mError\033[0m]: Rotation must be one of {0, 1, 2," \
        " 3}."

    # Rotate input.
    output = onp.rot90(
        input, degree,
        axes=(0, 1),
    )
    return output


def flip(
    input: onp.ndarray,
    *,
    number: int,
) -> onp.ndarray:
    r"""
    Flip horizontally and vertically a batch of 2D matrices.

    Args
    ----
    - input :
        Input.
    - number :
        Number of flip times.
        Must be one of {0, 1, 2, 3} where:
        0. do nothing;
        1. flip horizontally;
        2. flip vertically;
        3. flip horizontally and vertically.

    Returns
    -------
    - output :
        Output.
    """
    # /
    # ANNOTATE
    # /
    output: onp.ndarray

    # Safety check.
    assert (
        input.ndim == 2
    ), "[\033[91mError\033[0m]: Operation input must be a 2D matrix."
    assert (
        number in (0, 1, 2, 3)
    ), "[\033[91mError\033[0m]: Number of flip times must be one of {0," \
        " 1, 2, 3}."

    # Flip input.
    if (number == 0):
        output = input
    elif (number == 1):
        output = onp.flip(input, 1)
    elif (number == 2):
        output = onp.flip(input, 0)
    elif (number == 3):
        output = onp.flip(input, (0, 1))
    else:
        print("[\033[91mError\033[0m]: Reach impossible code.")
        raise NotImplementedError
    return output