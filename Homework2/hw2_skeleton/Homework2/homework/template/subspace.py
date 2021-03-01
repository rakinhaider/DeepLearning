# All imports
from typing import List, Dict
from typing import Union
from typing import Protocol
import numpy as onp
import torch


# =============================================================================
# *****************************************************************************
# -----------------------------------------------------------------------------
# < Invariant Subspace >
# -----------------------------------------------------------------------------
# *****************************************************************************
# =============================================================================


class RotateProtocal(Protocol):
    r"""
    Protocal (Type) of Rotation Function.
    """
    def __call__(
        self,
        /,
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
            Must be one of {0, 1, 2, 3} where:
            0. Do nothing;
            1. Counter-clockwise rotate 90 degrees;
            2. Counter-clockwise rotate 180 degrees;
            3. Counter-clockwise rotate 270 degrees.

        Returns
        -------
        - output :
            Output.
        """
        if degree % 90 == 0:
            return onp.rot90(input, degree // 90)
        else:
            raise ValueError


class FlipProtocal(Protocol):
    r"""
    Protocal (Type) of Flip Function.
    """
    def __call__(
        self,
        /,
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
            0. Do nothing;
            1. Flip horizontally;
            2. Flip vertically;
            3. Flip horizontally and vertically.

        Returns
        -------
        - output :
            Output.
        """
        if number == 0:
            return onp.copy(input)
        elif number == 1:
            return onp.flip(input, 1)
        elif number == 2:
            return onp.flip(input, 0)
        else:
            return onp.flip(input)


class InvariantSubspace(
    object,
    metaclass=type,
):
    r"""
    Invariant subspace.
    """
    def __init__(
        self,
        /,
        rotate: RotateProtocal, flip: FlipProtocal,
        *,
        shape: List[int],
    ) -> None:
        r"""
        Initialize the class.

        Args
        ----
        - rotate :
            Rotate function.
            Please above protocol to under its usage.
        - flip :
            Flip function.
            Please above protocol to under its usage.
        - shape :
            Input shape of all operations.

        Returns
        -------
        """
        # /
        # ANNOTATE
        # /
        self.rotate: RotateProtocal
        self.flip: FlipProtocal
        self.shape: List[int]
        self.numel: int
        transformed_buf: List[onp.ndarray]
        self.transform_mat: onp.ndarray
        self.eigenvalues: onp.ndarray
        self.eigenvectors: onp.ndarray
        order: onp.ndarray

        # Save necessary attribtes.
        self.rotate = rotate
        self.flip = flip
        self.shape = shape
        self.numel = int(onp.prod(self.shape))

        # Run in single process.
        transformed_buf = []
        for index in range(self.numel):
            transformed_buf.append(self.reynold_operator(index))
        self.transform_mat = onp.stack(
            transformed_buf,
            axis=1,
        )

        # Eigenvectors is used to describe the transformation subspace.
        self.eigenvalues, self.eigenvectors = onp.linalg.eig(
            self.transform_mat,
        )
        self.eigenvalues = onp.real(self.eigenvalues)
        self.eigenvectors = onp.real(self.eigenvectors).T

        # Sorted by descending eigenvalues.
        # We focus only on non-trival eigenvalues and eigenvectors.
        order = self.eigenvalues.argsort()[::-1]
        rank = onp.linalg.matrix_rank(self.transform_mat)
        self.eigenvalues = self.eigenvalues[order[0:rank]]
        self.eigenvectors = self.eigenvectors[order[0:rank]]

    def reynold_operator(
        self,
        /,
        index: int,
    ) -> onp.ndarray:
        r"""
        Reynold operator that transforms a group of image functions by a given
        onehont vector.

        Args
        ----
        - index :
            Index $i$ of onehot vector $e_{i}$.

        Returns
        -------
        - output :
            Averaged output.
            It is equivalent to $\overline{T} e_{i}$.
        """
        # /
        # YOU SHOULD FILL IN THIS FUNCTION
        # /
        onehot = onp.zeros(self.numel)
        onehot[index] = 1

        reynold = onp.zeros(self.numel)

        for i in range(4):
            reynold += self.rotate(onehot, degree=90*i)

        for i in range(4):
            reynold += self.flip(onehot, number=i)

        reynold = reynold / 8
        return reynold
        # raise NotImplementedError