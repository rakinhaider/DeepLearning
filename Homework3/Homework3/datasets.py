# All imports.
import torch
import re
import os
import abc
from typing import Any
from typing import TypeVar, Generic
from typing import ClassVar
from typing import Tuple, List, Dict
from typing import Iterator
from typing import TextIO
from typing import Union


# =============================================================================
# *****************************************************************************
# -----------------------------------------------------------------------------
# ## Memories
# -----------------------------------------------------------------------------
# *****************************************************************************
# =============================================================================


# /
# GENERIC
# /
_ELEMENT = TypeVar("_ELEMENT")
_CORA = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class Dataset(
    Generic[_ELEMENT],
    metaclass=abc.ABCMeta,
):
    r"""
    Dataset.
    """
    # =========================================================================
    # -------------------------------------------------------------------------
    # =========================================================================
    # /
    # ANNOTATE
    # /
    COLORS: ClassVar[Dict[str, str]]
    PALATTE: ClassVar[List[str]]

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

    def __annotate__(
        self,
        /,
    ) -> None:
        r"""
        Annotate instance attributes.

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
        self.memory: List[_ELEMENT]
        self.valid_indices: Union[None, List[int]]
        self.test_indices: Union[None, List[int]]

    def __iter__(
        self,
        /,
    ) -> Iterator[_ELEMENT]:
        r"""
        Get an iterator.

        Args
        ----

        Returns
        -------
        - iterator :
            Iterator.
        """
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # Iterate on the memory.
        return iter(self.memory)

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
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # Get length directly.
        return len(self.memory)

    def __getitem__(
        self,
        /,
        i: int,
    ) -> _ELEMENT:
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
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # Directly locate in the memory.
        return self.memory[i]

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
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # Get representation string recursively.
        return self.repr(
            self.memory,
            depth=0,
        )

    # =========================================================================
    # -------------------------------------------------------------------------
    # Generate representation string of memory by recursion.
    # -------------------------------------------------------------------------
    # =========================================================================

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
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
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
            if (len(item) > 1):
                right = ", ...]"
            else:
                right = "]"
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
                "[\033[91mError\033[0m]: Unknown memory unit type" \
                " \"{:s}\".".format(str(type(item))),
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
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
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
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # Remove color style string.
        return re.sub(r"\033\[[^m]+m", "", msg)

    # =========================================================================
    # -------------------------------------------------------------------------
    # Function to pin shared data into device memory.
    # -------------------------------------------------------------------------
    # =========================================================================

    def pin(
        self,
        device: str,
        /,
    ) -> Dict[str, torch.Tensor]:
        r"""
        Pin shared data into device memory.

        Args
        ----
        - device :
            Device name if pin is required.

        Returns
        -------
        - batch_pin :
            Pinned memory as part of a minibatch.
        """
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # It is assumed that no sharing data requires pin.
        return dict()


class CoraDataset(
    Dataset[_CORA],
    metaclass=type,
):
    r"""
    Cora Dataset.
    """
    # /
    # ANNOTATE
    # /
    NUM_FEATS: ClassVar[int]
    NUM_LABELS: ClassVar[int]
    # -----
    LABEL_STR2INT: ClassVar[Dict[str, int]]

    # Constant properties.
    NUM_NODES = 2708
    NUM_FEATS = 1433
    NUM_LABELS = 7
    LABEL_STR2INT = {
        "Theory": 0,
        "Reinforcement_Learning": 1,
        "Genetic_Algorithms": 2,
        "Case_Based": 3,
        "Neural_Networks": 4,
        "Rule_Learning": 5,
        "Probabilistic_Methods": 6,
    }

    def __init__(
        self,
        /,
        root: str,
        *,
        dense: bool,
    ) -> None:
        r"""
        Initialize the class.

        Args
        ----
        - root :
            Root directory holding raw data.
        - dense :
            Use dense adjacency matrix instead of sparse adjacency list.

        Returns
        -------
        """
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # /
        # ANNOTATE
        # /
        file: TextIO
        line: str
        id_str2int: Dict[str, int]
        feat_buf: List[List[int]]
        label_buf: List[int]
        # -----
        str_id: str
        str_feat: List[str]
        str_label: str
        int_id: int
        int_feat: List[int]
        # -----
        str_id1: str
        str_id2: str
        edge_buf: List[Tuple[int, int]]
        edge_mat: torch.Tensor
        # -----
        feat_mat: torch.Tensor
        target: torch.Tensor
        adj_mat: torch.Tensor

        # read feature file and parse all node data.
        id_str2int = {}
        feat_buf = []
        label_buf = []
        with open(os.path.join(root, "cora.content"), "r") as file:
            while (True):
                line = file.readline().strip()
                if (line):
                    pass
                else:
                    break
                str_id, *str_feat, str_label = re.split(r"\s+", line)
                int_id = int(str_id)
                int_feat = [int(str_itr) for str_itr in str_feat]
                if (int_id in id_str2int):
                    print("[\033[91mError\033[0m]: Reach impossible branch.")
                    raise NotImplementedError
                else:
                    id_str2int[str_id] = len(id_str2int)
                feat_buf.append(int_feat)
                label_buf.append(self.LABEL_STR2INT[str_label])
        feat_mat = torch.Tensor(feat_buf).view(
            self.NUM_NODES, self.NUM_FEATS,
        ).to(torch.get_default_dtype())
        target = torch.LongTensor(label_buf).view(self.NUM_NODES)

        # read connection file and exclude duplication.
        edge_buf = []
        with open(os.path.join(root, 'cora.cites'), "r") as file:
            while (True):
                line = file.readline().strip()
                if (line):
                    pass
                else:
                    break
                str_id1, str_id2 = re.split(r"\s+", line)
                edge_buf.append((id_str2int[str_id1], id_str2int[str_id2]))
                edge_buf.append((id_str2int[str_id2], id_str2int[str_id1]))
        edge_buf = list(set(edge_buf))
        edge_mat = torch.LongTensor(edge_buf).view(-1, 2)

        # Remove self-loop connections.
        edge_mat = edge_mat[edge_mat[:, 0] != edge_mat[:, 1]]

        # Enforce adjacency form by given argument.
        if (dense):
            adj_mat = torch.zeros(
                self.NUM_NODES, self.NUM_NODES,
                dtype=torch.long,
            )
            adj_mat[edge_mat[:, 0], edge_mat[:, 1]] = 1
        else:
            adj_mat = edge_mat

        # Put the only graph in the memory.
        self.memory = [(feat_mat, adj_mat, target)]
        self.valid_indices = None
        self.test_indices = None

    def pin(
        self,
        device: str,
        /,
    ) -> Dict[str, torch.Tensor]:
        r"""
        Pin shared data into device memory.

        Args
        ----
        - device :
            Device name if pin is required.

        Returns
        -------
        - batch_pin :
            Pinned memory as part of a minibatch.
        """
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # /
        # ANNOTATE
        # /
        feat_mat: torch.Tensor
        target: torch.Tensor
        adj_mat: torch.Tensor

        # Pin the only graph in memory.
        if (len(self.memory) > 1):
            print(
                "[\033[91mError\033[0m]: Pinnable graph dataset must have a" \
                " single graph as element.",
            )
            raise RuntimeError
        else:
            feat_mat, adj_mat, target = next(iter(self.memory))
            return dict(
                node_feat=feat_mat.to(device), adjacency=adj_mat.to(device),
                node_target=target.to(device),
            )


class PTBDataset(
    Dataset[torch.Tensor],
    metaclass=type,
):
    r"""
    Cora Dataset.
    """
    # =========================================================================
    # -------------------------------------------------------------------------
    # =========================================================================
    # /
    # ANNOTATE
    # /
    NUM_WORDS: ClassVar[int]
    EOS: ClassVar[int]
    OOV: ClassVar[int]

    # Constant properties.
    NUM_WORDS = 10001
    EOS = 0
    OOV = -1

    def __init__(
        self,
        /,
        root: str,
    ) -> None:
        r"""
        Initialize the class.

        Args
        ----
        - root :
            Root directory holding raw data.

        Returns
        -------
        """
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # /
        # ANNOTATE
        # /
        self.word_str2int: Dict[Union[int, str], int]
        train_seq: List[int]
        valid_seq: List[int]
        test_seq: List[int]

        # Initialize dictionary.
        self.word_str2int = dict()
        self.word_str2int[self.EOS] = len(self.word_str2int)
        self.word_str2int[self.OOV] = len(self.word_str2int)

        # Load all dataset files.
        train_seq = self.load_file_as_seq(root, "ptb.train.txt", True)
        valid_seq = self.load_file_as_seq(root, "ptb.valid.txt", False)
        test_seq = self.load_file_as_seq(root, "ptb.test.txt", False)

        # Safety check.
        if (len(self.word_str2int) == self.NUM_WORDS):
            pass
        else:
            print(
                "[\033[91mError\033[0m]: Reach impossible branch.",
            )
            raise RuntimeError

        # Save into memory.
        self.memory = [
            torch.LongTensor(train_seq), torch.LongTensor(valid_seq),
            torch.LongTensor(test_seq),
        ]
        self.valid_indices = [1]
        self.test_indices = [2]

    def load_file_as_seq(
        self,
        root: str, filename: str, update: bool,
        /,
    ) -> List[int]:
        r"""
        Read a dataset file as a single sequence.

        Args
        ----
        - root :
            Root directory holding all dataset files.
        - filename :
            File name of a dataset file.
        - update :
            If True, update whole dataset word-to-int dictionary.
            If False, unknown word will be regarded as OOV.

        Returns
        -------
        - seq :
            Dataset as a single sequence.
        """
        # =====================================================================
        # ---------------------------------------------------------------------
        # =====================================================================
        # /
        # ANNOTATE
        # /
        file: TextIO
        line: str
        sentence: List[str]
        word_str: str
        word_int: int
        seq: List[int]

        # Read a dataset file as a single sequence.
        seq = []
        with open(os.path.join(root, filename), "r") as file:
            while (True):
                # Read each line as a sentence with specific ending symbol.
                line = file.readline().strip()
                if (line):
                    pass
                else:
                    break
                sentence = re.split(r"\s+", line)

                # Translate all words into integers.
                for word_str in sentence:
                    if (word_str in self.word_str2int):
                        word_int = self.word_str2int[word_str]
                    elif (update):
                        self.word_str2int[word_str] = len(self.word_str2int)
                        word_int = self.word_str2int[word_str]
                    else:
                        word_int = self.word_str2int[self.OOV]
                    seq.append(word_int)
        return seq