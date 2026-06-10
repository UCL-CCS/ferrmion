"""Ternary Tree fermion to qubit mappings.

The :class:`TernaryTree` class is a pure-Python builder for ternary-tree
structures. The encoding itself is represented by the Rust-backed
:class:`ferrmion.core.MajoranaEncoding`, obtained from a tree via
:meth:`TernaryTree.build_encoding`.
"""

import logging

import numpy as np
from numpy.typing import NDArray

from ferrmion import core
from ferrmion.core import FermionHamiltonian, MajoranaEncoding, QubitHamiltonian

from .ternary_tree_node import TTNode, node_sorter

logger = logging.getLogger(__name__)

"""Instructions to build a TernaryTree.

Each node in the tree is represented as a tuple of (qubit_index, (x_child, y_child, z_child)).

To ensure there are no clashes between qubit indices and majorana indices,
majorana indices should be offset by max(qubit_indices) + 1.
"""
type TTFlatpack = list[tuple[int, tuple[int | None, int | None, int | None]]]


class TernaryTree:
    """Builder for ternary tree encodings of fermionic operators.

    Attributes:
        n_modes (int): The number of fermionic modes to be encoded.
        n_qubits (int): The number of qubits in encoded operators.
        root_node (TTNode): The root node of the tree.
        enumeration_scheme (dict[str, tuple[int, int]] | None): The enumeration scheme.

    Methods:
        build_encoding(): Build the Rust-backed MajoranaEncoding for the tree.
        default_mode_op_map(): Create a default mode operator map for the tree.
        default_enumeration_scheme(): Create a default enumeration scheme for the tree.
        as_dict(): Return the tree structure as a dictionary.
        add_node(node_string: str): Add a node to the tree.
        branch_pauli_map(): Create a map from each branch string to a Pauli string.
        string_pairs(): Return the pair of branch strings which correspond to each node.

    Simple Example:
        >>> from ferrmion.encode.ternary_tree import TernaryTree
        >>> tree = TernaryTree(4)
        >>> tree.add_node('x')
        >>> tree.enumeration_scheme = tree.default_enumeration_scheme()
        >>> tree.as_dict()

    Advanced Usage:
        >>> from ferrmion.encode.ternary_tree import TernaryTree
        >>> jw_tree = TernaryTree.jordan_wigner(4)
        >>> encoding = jw_tree.build_encoding()
    """

    def __init__(
        self,
        n_modes: int,
        n_qubits: None | int = None,
        root_node: TTNode | None = None,
    ):
        """Initialise a ternary tree.

        Args:
            n_modes (int): How many fermionic modes in the encoding.
            n_qubits (int): Optional overwrite of number of qubits in target encoding.
            root_node (TTNode): The root node of the tree.
        """
        self.n_modes = n_modes
        self.n_qubits = n_modes if n_qubits is None else n_qubits
        self.root_node = TTNode() if root_node is None else root_node
        self._enumeration_scheme = {}
        self.default_mode_op_map = [*range(self.n_modes)]

        if None not in self.root_node.child_qubit_labels.values():
            self.enumeration_scheme: dict[str, tuple[int, int]] = {
                node: (mode, qubit)
                for mode, (node, qubit) in enumerate(
                    self.root_node.child_qubit_labels.items()
                )
            }

    def __eq__(self, other: object) -> bool:
        """Checks if two trees produce exactly equivalent encodings."""
        if isinstance(other, TernaryTree):
            return (
                self.n_modes == other.n_modes
                and self.n_qubits == other.n_qubits
                and self.build_encoding() == other.build_encoding()
            )
        return NotImplemented

    @property
    def default_mode_op_map(self) -> NDArray[np.uint]:
        """The default mode operator map for the tree."""
        return self._default_mode_op_map

    @default_mode_op_map.setter
    def default_mode_op_map(self, permutation: list[int]):
        """Set the default mode operator map.

        Args:
            permutation (list[int]): A list containing a permutation of mode indices.
        """
        logger.debug("Setting default mode operator map.")
        if set(permutation) != {*range(self.n_modes)}:
            error_string = "Default Mode op map does not cover all modes.\n"
            logger.error(error_string)
            logger.error(permutation)
            raise ValueError(error_string)

        self._default_mode_op_map = np.array(permutation, dtype=np.uint)

    @property
    def enumeration_scheme(self) -> dict[str, tuple[int, int]]:
        """Get the enumeration scheme for the tree.

        Note:
            The tuple is organised as (modes, qubits).

        Example:
            >>> from ferrmion.encode.ternary_tree import TernaryTree
            >>> tree = TernaryTree.jordan_wigner(3)
            >>> tree.enumeration_scheme
            {"": (0,0), "z": (1,1), "zz": (2,2)}
        """
        return self._enumeration_scheme

    @enumeration_scheme.setter
    def enumeration_scheme(self, enumeration_dict: dict[str, tuple[int, int]]):
        """Set the enumeration scheme.

        Args:
            enumeration_dict (dict[str, tuple[int, int]]): An dictionary mapping tree nodes to (mode, qubit) indices
        """
        logger.debug("Setting enumeration scheme.")
        error_string = ""
        if set(self.root_node.child_strings) != set(enumeration_dict.keys()):
            error_string += f"Enumeration scheme {enumeration_dict} must contain all nodes {self.root_node.child_strings}.\n"

        modes = set()
        qubits = set()
        for m, q in enumeration_dict.values():
            logger.debug(f"{m=}{q=}")
            modes.add(m)
            qubits.add(q)
        expected_modes = set(range(self.n_modes))
        if set(modes).symmetric_difference(expected_modes):
            error_string += f"Invalid mode labels {set(modes)} in enumeration scheme ({expected_modes=}).\n"
        if len(set(qubits)) != self.n_modes:
            error_string += f"Expected {self.n_modes} qubit labels, got {len(set(qubits))} in enumeration scheme.\n"

        if error_string != "":
            logger.error(error_string)
            raise ValueError(error_string)

        self.default_mode_op_map = [enum[0] for enum in enumeration_dict.values()]
        self._enumeration_scheme = enumeration_dict

    def build_encoding(self) -> MajoranaEncoding:
        """Build the Rust-backed encoding for the tree.

        The tree's ``default_mode_op_map`` is applied, so mode ``i`` of the
        returned encoding corresponds to the tree node enumerated as mode
        ``default_mode_op_map[i]``.

        Returns:
            MajoranaEncoding: The encoding represented by this tree.
        """
        encoding = MajoranaEncoding.from_flatpack(self.flatpack(), self.n_qubits)
        mode_op_map = [int(mode) for mode in self.default_mode_op_map]
        if mode_op_map != [*range(self.n_modes)]:
            encoding = encoding.apply_mode_enumeration(mode_op_map)
        return encoding

    @property
    def vacuum_state(self) -> NDArray[np.bool]:
        """The vacuum state of the encoding represented by this tree."""
        return self.build_encoding().vacuum_state

    def encode(self, fham: FermionHamiltonian) -> QubitHamiltonian:
        """Encode a fermionic Hamiltonian into a qubit Hamiltonian.

        Args:
            fham (FermionHamiltonian): The fermionic Hamiltonian to encode.

        Returns:
            QubitHamiltonian: The encoded qubit Hamiltonian.
        """
        return self.build_encoding().encode(fham)

    def encode_annealed(
        self,
        fham: FermionHamiltonian,
        temperature: int | None = None,
        initial_guess: list[int] | None = None,
        coefficient_weighted: bool = True,
        seed: int | None = None,
    ) -> tuple[QubitHamiltonian, MajoranaEncoding]:
        """Encode a Hamiltonian, optimising mode enumeration via simulated annealing.

        Args:
            fham (FermionHamiltonian): The fermionic Hamiltonian to encode.
            temperature (int | None): Initial annealing temperature. Defaults to ``n_modes // 2``.
            initial_guess (list[int] | None): Starting permutation. Defaults to identity.
            coefficient_weighted (bool): If True, minimise coefficient-weighted Pauli weight.
            seed (int | None): Seed for the RNG driving permutation moves.
                Defaults to ``1017`` when omitted.

        Returns:
            tuple[QubitHamiltonian, MajoranaEncoding]: The encoded qubit
            Hamiltonian and the encoding with the optimised enumeration.
        """
        return self.build_encoding().encode_annealed(
            fham,
            temperature=temperature,
            initial_guess=initial_guess,
            coefficient_weighted=coefficient_weighted,
            seed=seed,
        )

    def encode_topphatt(
        self,
        fham: FermionHamiltonian,
        parallelize: bool = True,
        heuristic: str = "min_weight",
        seed: int | None = None,
    ) -> tuple[QubitHamiltonian, MajoranaEncoding]:
        """Encode a Hamiltonian, using TOPP-HATT optimisation.

        Args:
            fham: The FermionHamiltonian to encode.
            parallelize: Whether to parallelize the encoding.
            heuristic: Node-selection strategy. One of ``"min_weight"``
                (evaluate every active node and keep the lowest Pauli weight),
                ``"x_first"`` (lowest-indexed active node), ``"z_first"``
                (highest-indexed active node), or ``"random"``
                (uniformly random active node using ``seed``).
            seed: RNG seed for ``heuristic="random"``. Ignored otherwise;
                defaults to ``0`` when omitted.

        Returns:
            tuple[QubitHamiltonian, MajoranaEncoding]: The encoded qubit
            Hamiltonian and the tree-optimised encoding.
        """
        return core.encode_topphatt(
            flatpack=self.flatpack(),
            n_qubits=self.n_qubits,
            fham=fham,
            parallelize=parallelize,
            heuristic=heuristic,
            seed=seed,
        )

    def decode(self, states: NDArray[np.bool]) -> NDArray[np.bool]:
        """Decode Z-basis states into fermionic occupation vectors.

        Args:
            states: 2D boolean array of shape ``(n_states, n_qubits)``.

        Returns:
            2D boolean array of shape ``(n_states, n_modes)``.

        Raises:
            ValueError: if any state cannot be decoded for this encoding.
        """
        return self.build_encoding().decode(states)

    def hartree_fock_state(
        self,
        fermionic_hf_state: NDArray[bool],
        mode_op_map: NDArray[np.uint] | list[int] | None = None,
    ) -> NDArray[np.bool]:
        """Find the Hartree-Fock state of a majorana string encoding.

        Args:
            fermionic_hf_state (NDArray[int]): An array of mode occupations.
            mode_op_map (dict[int, int]): An array mapping modes to pairs of majorana strings mode_op_map[i]=j => i -> (2j,2j+1)

        Returns:
            NDArray: The Hartree-Fock ground state in computational basis.
        """
        if mode_op_map is None:
            mode_op_map = self.default_mode_op_map

        mode_op_map = np.asarray(mode_op_map, dtype=np.uintp)

        return self.build_encoding().hartree_fock_state(
            np.asarray(fermionic_hf_state, dtype=bool), mode_op_map
        )

    def number_operator(
        self, mode: int, coeff: complex | float = 1.0
    ) -> QubitHamiltonian:
        """Return the number operator of a mode for this encoding."""
        return self.build_encoding().number_operator(mode, complex(coeff))

    def edge_operator(
        self,
        edge_indices: tuple[int, int],
        coeff: complex | float = 1.0,
        with_conjugate: bool = False,
    ) -> QubitHamiltonian:
        """Return the edge operator of a pair of modes for this encoding."""
        return self.build_encoding().edge_operator(
            tuple(edge_indices), complex(coeff), with_conjugate
        )

    def interaction_operator(
        self,
        mode_indices: tuple[int, int, int, int],
        coeff: complex | float = 1.0,
        physicist_notation: bool = True,
        with_conjugate: bool = False,
    ) -> QubitHamiltonian:
        """Return an interaction operator of four modes for this encoding."""
        return self.build_encoding().interaction_operator(
            tuple(mode_indices), complex(coeff), physicist_notation, with_conjugate
        )

    def to_json(self) -> dict:
        """Serialise the encoding represented by this tree.

        Returns:
            dict: Dictionary with ``"ipowers"``, ``"symplectics"`` and
            ``"vacuum_state"`` keys.
        """
        return self.build_encoding().to_json()

    def flatpack(self) -> TTFlatpack:
        """Create a TTFlatpack from the tree, which can be saved or passed to rust functions.

        Node children are represented by their qubit index (an int that also
        appears as the first element of some flatpack entry).  Leaf children
        with a known Majorana index are encoded as
        ``majorana_index + max_node_index + 1``, which is strictly greater than
        every node qubit index (including when the Majorana index is 0) and can
        therefore never be confused with a node.
        Leaves without a known Majorana index are represented as ``None``.

        Returns:
            list[tuple[int, tuple[int | None, int | None, int | None]]]

        Example:
            >>> TernaryTree.jordan_wigner(4).flatpack()
            >>> [(0, (4, 5,1)), (1, (6,7,2)), (2, (8,9,3)), (3, (10,11,4))]
        """
        max_node_index: int = max(
            qubit for _, qubit in self.enumeration_scheme.values()
        )
        flatpack: TTFlatpack = []

        to_flatten: list[TTNode] = [self.root_node]
        while len(to_flatten) > 0:
            node: TTNode = to_flatten.pop(0)
            children: list[int | None] = [None, None, None]
            for i, edge in enumerate(["x", "y", "z"]):
                child_node = getattr(node, edge)
                if isinstance(child_node, TTNode):
                    to_flatten.append(child_node)
                    children[i] = self.enumeration_scheme[child_node.root_path][1]
                else:
                    majorana_idx = node.leaf_majorana_indices.get(edge)
                    if majorana_idx is not None:
                        children[i] = majorana_idx + max_node_index + 1

            flatpack.append(
                (
                    int(self.enumeration_scheme[node.root_path][1]),
                    (children[0], children[1], children[2]),
                )
            )

        return flatpack

    @classmethod
    def from_flatpack(cls, flatpack: TTFlatpack) -> "TernaryTree":
        """Construct a TernaryTree from a TTFlatpack.

        Args:
            flatpack: The flatpack representation of the tree.

        Returns:
            A new TernaryTree instance.

        Raises:
            TypeError: If the flatpack is invalid.
        """
        if not flatpack:
            raise ValueError("Flatpack cannot be empty")
        node_qubit_indices: set[int] = {item[0] for item in flatpack}
        max_node_index: int = max(node_qubit_indices)
        used_qubit_indices = [flatpack[0][0]]
        for item in flatpack:
            if item[0] not in used_qubit_indices:
                raise TypeError("Cannot construct TTFlatpack from disconnected nodes.")
            children = item[1]
            if len(children) != 3:
                raise TypeError("TTFlatpack nodes must have three optional children.")
            for child in children:
                if isinstance(child, int):
                    if child in node_qubit_indices:
                        used_qubit_indices.append(child)
                    elif child > max_node_index:
                        pass  # leaf: Majorana index = child - (max_node_index + 1)
                    else:
                        raise TypeError(
                            f"TTFlatpack child {child} is not a node qubit index and is "
                            f"<= max_node_index ({max_node_index}); leaf values must be "
                            "> max_node_index (i.e. majorana_index + max_node_index + 1)."
                        )
                elif child is None:
                    continue
                else:
                    raise TypeError(
                        "TTFlatpack contains child which is not int | None."
                    )

        nodes = [TTNode() for _ in range(max_node_index + 1)]
        for qubit_index, children in flatpack:
            node = nodes[qubit_index]
            node.qubit_index = qubit_index
            for edge, child in zip(["x", "y", "z"], children):
                if isinstance(child, int) and child in node_qubit_indices:
                    setattr(node, edge, nodes[child])
                elif isinstance(child, int) and child > max_node_index:
                    node.leaf_majorana_indices[edge] = child - (max_node_index + 1)

        root = nodes[flatpack[0][0]]
        enumeration_scheme = {}
        mode_counter = [0]

        def assign_modes(node, path):
            node.root_path = path
            enumeration_scheme[path] = (mode_counter[0], node.qubit_index)
            mode_counter[0] += 1
            if node.x:
                assign_modes(node.x, path + "x")
            if node.y:
                assign_modes(node.y, path + "y")
            if node.z:
                assign_modes(node.z, path + "z")

        assign_modes(root, "")
        n_modes = len(enumeration_scheme)
        tree = cls(n_modes=n_modes, root_node=root)
        tree.enumeration_scheme = enumeration_scheme
        return tree

    def default_enumeration_scheme(self) -> dict[str, tuple[int, int]]:
        """Create a default enumeration scheme for the tree.

        Note:
            The tuple is organised as (modes, qubits).

        Example:
            >>> from ferrmion.encode.ternary_tree import TernaryTree
            >>> tree = TernaryTree.jordan_wigner(3)
            >>> tree.default_enumeration_scheme()
            {"": (0,0), "z": (1,1), "zz": (2,2)}
        """
        logger.debug("Setting default enumeration scheme")
        logger.debug("Child strings %s", self.root_node.child_strings)
        enumeration_scheme = {}
        child_labels = self.root_node.child_qubit_labels
        spare_labels: set[int] = set(range(len(child_labels))).difference(
            child_labels.values()
        )
        for mode, (child, qubit) in enumerate(child_labels.items()):
            if qubit is None:
                qubit = spare_labels.pop()
            enumeration_scheme[child] = (int(mode), int(qubit))
        return enumeration_scheme

    def as_dict(self):
        """Return the tree structure as a dictionary."""
        return self.root_node.as_dict()

    def add_node(self, node_string: str) -> "TernaryTree":
        """Add a node to the tree.

        Args:
            node_string (str): The string representation of the node.

        Returns:
            TernaryTree: The tree with the node added.

        Example:
            >>> from ferrmion.encode.ternary_tree import TernaryTree
            >>> tree = TernaryTree(3)
            >>> tree.add_node('x')
        """
        logger.debug("Adding node %s to TernaryTree", node_string)
        node_string = node_string.lower()
        valid_string = np.all([char in ["x", "y", "z"] for char in node_string])
        if not valid_string:
            raise ValueError("Branch string can only contain x,y,z")

        node = self.root_node
        for char in node_string:
            if isinstance(getattr(node, char), TTNode):
                node = getattr(node, char)
            else:
                node = node.add_child(
                    which_child=char,
                )
        return self

    @property
    def branch_pauli_map(self) -> dict[str, str]:
        """Create a map from each branch string to a Pauli string.

        Returns:
            dict[str, str]: A dictionary of all branch strings with their corresponding Pauli strings.

        Example:
            >>> from ferrmion.encode.ternary_tree import TernaryTree
            >>> tree = TernaryTree(3)
            >>> tree.add_node('x')
            >>> tree.add_node('xz')
            >>> tree.branch_pauli_map
            {'xx': 'XXI',
            'xzx': 'XZX',
            'y': 'YII',
            'xy': 'XYI',
            'xzz': 'XZZ',
            'xzy': 'XZY',
            'z': 'ZII'}
        """
        logger.debug("Building branch operator map for TernaryTree.")

        branches = self.root_node.branch_strings

        qubit_index = {
            node: qubit for node, (_, qubit) in self.enumeration_scheme.items()
        }
        branch_pauli_map = {}
        for branch in branches:
            branch_pauli_map[branch] = ["I"] * self.n_qubits
            node = self.root_node
            for char in branch:
                node_index = qubit_index[node.root_path]
                branch_pauli_map[branch][node_index] = char.upper()
                node = getattr(node, char, None)

            branch_pauli_map[branch] = "".join(branch_pauli_map[branch])
        logger.debug("Branch pauli map complete")
        logger.debug(branch_pauli_map)
        return branch_pauli_map

    @property
    def string_pairs(self) -> dict[str | int, tuple[str, str]]:
        """Return the pair of branch strings which correspond to each node.

        Returns:
            dict[str, tuple(str,str)]: A dictionary of all node labels, j,  with branch strings (2j, 2j+1).

        Example:
            >>> from ferrmion.encode.ternary_tree import TernaryTree
            >>> tree = TernaryTree(3)
            >>> tree.add_node('x')
            >>> tree.add_node('xz')
            >>> tree.string_pairs
            {'': ('xzz', 'y'), 'x': ('xx', 'xy'), 'xz': ('xzx', 'xzy')}
        """
        logger.debug("Building string pairs for TernaryTree.")
        node_set = self.root_node.child_strings

        pairs = {}
        for node_string in node_set:
            node = self.root_node
            for char in node_string:
                node = getattr(node, char)

            x_string = node_string + "x"
            y_string = node_string + "y"
            while x_string in node_set:
                x_string += "z"

            while y_string in node_set:
                y_string += "z"

            if x_string.count("y") % 2 == 0:
                pairs[node.root_path] = x_string, y_string
            elif y_string.count("y") % 2 == 0:
                pairs[node.root_path] = y_string, x_string

        return pairs

    @classmethod
    def jordan_wigner(cls, n_modes: int) -> "TernaryTree":
        """Create a Jordan-Wigner encoding tree.

        Example:
            >>> from ferrmion.encode.ternary_tree import TernaryTree
            >>> jw_tree = TernaryTree.jordan_wigner(3)
        """
        logger.debug("Creating Jordan-Wigner encoding tree")
        new_tree = cls(n_modes=n_modes)
        new_tree.add_node("z" * (n_modes - 1))
        new_tree.enumeration_scheme = new_tree.default_enumeration_scheme()
        return new_tree

    @classmethod
    def parity(cls, n_modes: int) -> "TernaryTree":
        """Create a parity encoding tree.

        Example:
            >>> from ferrmion.encode.ternary_tree import TernaryTree
            >>> parity_tree = TernaryTree.parity(3)
        """
        logger.debug("Creating parity encoding tree")
        new_tree = cls(n_modes=n_modes)
        new_tree.add_node("x" * (n_modes - 1))
        new_tree.enumeration_scheme = new_tree.default_enumeration_scheme()
        return new_tree

    @classmethod
    def bravyi_kitaev(cls, n_modes: int) -> "TernaryTree":
        """Create a Bravyi-Kitaev encoding tree.

        Example:
            >>> from ferrmion.encode.ternary_tree import TernaryTree
            >>> bk_tree = TernaryTree.bravyi_kitaev(3)
        """
        logger.debug("Creating Bravyi-Kitaev encoding tree")
        new_tree = cls(n_modes=n_modes)
        branches = ["x"]
        # one is used for root, which is defined
        remaining_qubits = n_modes - 1
        while remaining_qubits > 0:
            new_branches = set()
            for item in branches:
                if remaining_qubits > 0:
                    new_tree.add_node(item)
                    remaining_qubits -= 1
                else:
                    break

                new_branches.add(item + "x")
                new_branches.add(item + "z")
            branches = sorted(list(new_branches), key=node_sorter)
        new_tree.enumeration_scheme = new_tree.default_enumeration_scheme()
        return new_tree

    @classmethod
    def jkmn(cls, n_modes: int) -> "TernaryTree":
        """Create a JKMN (minimum-height) encoding tree.

        Example:
            >>> from ferrmion.encode.ternary_tree import TernaryTree
            >>> min_height_tree = TernaryTree.jkmn(3)
        """
        logger.debug("Creating JKMN encoding tree.")
        new_tree = cls(n_modes=n_modes)
        branches = ["x", "y", "z"]
        # one is used for root which is defined
        remaining_qubits = n_modes - 1
        while remaining_qubits > 0:
            new_branches = set()
            for item in branches:
                if remaining_qubits > 0:
                    new_tree.add_node(item)
                    remaining_qubits -= 1
                else:
                    break

                new_branches.add(item + "x")
                new_branches.add(item + "y")
                new_branches.add(item + "z")
            branches = sorted(list(new_branches), key=node_sorter)
        new_tree.enumeration_scheme = new_tree.default_enumeration_scheme()
        return new_tree

    def JordanWigner(self) -> MajoranaEncoding:
        """Create a Jordan-Wigner encoding with this tree's number of modes."""
        return JordanWigner(self.n_modes)

    def JW(self) -> MajoranaEncoding:
        """Alias for the Jordan-Wigner encoding."""
        return JordanWigner(self.n_modes)

    def ParityEncoding(self) -> MajoranaEncoding:
        """Create a parity encoding with this tree's number of modes."""
        return ParityEncoding(self.n_modes)

    def BravyiKitaev(self) -> MajoranaEncoding:
        """Create a Bravyi-Kitaev encoding with this tree's number of modes."""
        return BravyiKitaev(self.n_modes)

    def BK(self) -> MajoranaEncoding:
        """Alias for the Bravyi-Kitaev encoding."""
        return BravyiKitaev(self.n_modes)

    def JKMN(self) -> MajoranaEncoding:
        """Create a JKMN encoding with this tree's number of modes."""
        return JKMN(self.n_modes)


def string_pairing_algorithm(tree: TernaryTree):
    """String-pairing algoritm.

    This is used to produce a map from branches to majorana-indices
    of the root node.

    Args:
        tree (TernaryTree): A Ternary-tree encoding.

    Returns:
        dict[str, int]: A map from branches to majorana mdoe indices.
    """
    logger.debug("Running the string-pairing algorithm.")
    node_set = tree.root_node.child_strings

    branch_majorana_map = {}
    for node_string in node_set:
        # We want to set the majorana indices according to the
        # fermionic ones so that f_i -> (m_2i, m_2i+1)
        fermion_mode = tree.enumeration_scheme[node_string][0]

        x_string = node_string + "x"
        y_string = node_string + "y"
        while x_string in node_set:
            x_string += "z"

        while y_string in node_set:
            y_string += "z"

        if x_string.count("y") % 2 == 0:
            branch_majorana_map[x_string] = 2 * fermion_mode
            branch_majorana_map[y_string] = 2 * fermion_mode + 1
        elif y_string.count("y") % 2 == 0:
            branch_majorana_map[y_string] = 2 * fermion_mode
            branch_majorana_map[x_string] = 2 * fermion_mode + 1

    # We'll place the all-z string after all the required majorana modes
    all_z = "z"
    while all_z in node_set:
        all_z += "z"
    branch_majorana_map[all_z] = 2 * len(node_set) + 1

    logger.debug("String-pairing algorithm complete.")
    logger.debug(f"{branch_majorana_map=}")
    return branch_majorana_map


def JordanWigner(n_modes: int, n_qubits: int | None = None) -> MajoranaEncoding:
    """The Jordan-Wigner encoding.

    Args:
        n_modes (int): The number of fermionic modes.
        n_qubits (int | None): Optional number of qubits; defaults to ``n_modes``.

    Returns:
        MajoranaEncoding: The Jordan-Wigner encoding.

    Example:
        >>> from ferrmion.encode.ternary_tree import JordanWigner
        >>> jw = JordanWigner(3)
    """
    return MajoranaEncoding.jordan_wigner(n_modes, n_qubits)


def JW(n_modes: int, n_qubits: int | None = None) -> MajoranaEncoding:
    """Alias for the Jordan-Wigner encoding."""
    return JordanWigner(n_modes, n_qubits)


def ParityEncoding(n_modes: int, n_qubits: int | None = None) -> MajoranaEncoding:
    """The parity encoding.

    Args:
        n_modes (int): The number of fermionic modes.
        n_qubits (int | None): Optional number of qubits; defaults to ``n_modes``.

    Returns:
        MajoranaEncoding: The parity encoding.

    Example:
        >>> from ferrmion.encode.ternary_tree import ParityEncoding
        >>> parity = ParityEncoding(3)
    """
    return MajoranaEncoding.parity(n_modes, n_qubits)


def PE(n_modes: int, n_qubits: int | None = None) -> MajoranaEncoding:
    """Alias for ParityEncoding."""
    return ParityEncoding(n_modes, n_qubits)


def BravyiKitaev(n_modes: int, n_qubits: int | None = None) -> MajoranaEncoding:
    """The Bravyi-Kitaev encoding.

    Args:
        n_modes (int): The number of fermionic modes.
        n_qubits (int | None): Optional number of qubits; defaults to ``n_modes``.

    Returns:
        MajoranaEncoding: The Bravyi-Kitaev encoding.

    Example:
        >>> from ferrmion.encode.ternary_tree import BravyiKitaev
        >>> bk = BravyiKitaev(3)
    """
    return MajoranaEncoding.bravyi_kitaev(n_modes, n_qubits)


def BK(n_modes: int, n_qubits: int | None = None) -> MajoranaEncoding:
    """Alias for the Bravyi-Kitaev encoding."""
    return BravyiKitaev(n_modes, n_qubits)


def JKMN(n_modes: int, n_qubits: int | None = None) -> MajoranaEncoding:
    """The JKMN encoding.

    The JKMN encoding gives a ternary tree with the minimum Pauli-weight.

    Args:
        n_modes (int): The number of fermionic modes.
        n_qubits (int | None): Optional number of qubits; defaults to ``n_modes``.

    Returns:
        MajoranaEncoding: The JKMN encoding.

    Example:
        >>> from ferrmion.encode.ternary_tree import JKMN
        >>> min_height = JKMN(3)
    """
    return MajoranaEncoding.jkmn(n_modes, n_qubits)


def jordan_wigner(fham: FermionHamiltonian) -> QubitHamiltonian:
    """Naive Jordan-Wigner Encoding.

    Args:
        fham (FermionHamiltonian): Hamiltonian to encode.

    Returns:
        QubitHamiltonian: Encoded Hamiltonian.
    """
    return JordanWigner(fham.n_modes).encode(fham)


def bravyi_kitaev(fham: FermionHamiltonian) -> QubitHamiltonian:
    """Naive Bravyi-Kitaev Encoding.

    Args:
        fham (FermionHamiltonian): Hamiltonian to encode.

    Returns:
        QubitHamiltonian: Encoded Hamiltonian.
    """
    return BravyiKitaev(fham.n_modes).encode(fham)


def parity(fham: FermionHamiltonian) -> QubitHamiltonian:
    """Naive Parity Encoding.

    Args:
        fham (FermionHamiltonian): Hamiltonian to encode.

    Returns:
        QubitHamiltonian: Encoded Hamiltonian.
    """
    return ParityEncoding(fham.n_modes).encode(fham)


def jkmn(fham: FermionHamiltonian) -> QubitHamiltonian:
    """Naive Jiang-Kalev-Mruczkiewicz-Neven Encoding.

    Args:
        fham (FermionHamiltonian): Hamiltonian to encode.

    Returns:
        QubitHamiltonian: Encoded Hamiltonian.
    """
    return JKMN(fham.n_modes).encode(fham)


def jordan_wigner_topphatt(fham: FermionHamiltonian) -> QubitHamiltonian:
    """TOPP-HATT optimised Jordan-Wigner Encoding.

    Args:
        fham (FermionHamiltonian): Hamiltonian to encode.

    Returns:
        QubitHamiltonian: Encoded Hamiltonian.
    """
    return TernaryTree.jordan_wigner(fham.n_modes).encode_topphatt(fham)[0]


def bravyi_kitaev_topphatt(fham: FermionHamiltonian) -> QubitHamiltonian:
    """TOPP-HATT optimised Bravyi-Kitaev Encoding.

    Args:
        fham (FermionHamiltonian): Hamiltonian to encode.

    Returns:
        QubitHamiltonian: Encoded Hamiltonian.
    """
    return TernaryTree.bravyi_kitaev(fham.n_modes).encode_topphatt(fham)[0]


def parity_topphatt(fham: FermionHamiltonian) -> QubitHamiltonian:
    """TOPP-HATT optimised Parity Encoding.

    Args:
        fham (FermionHamiltonian): Hamiltonian to encode.

    Returns:
        QubitHamiltonian: Encoded Hamiltonian.
    """
    return TernaryTree.parity(fham.n_modes).encode_topphatt(fham)[0]


def jkmn_topphatt(fham: FermionHamiltonian) -> QubitHamiltonian:
    """TOPP-HATT optimised Jiang-Kalev-Mruczkiewicz-Neven Encoding.

    Args:
        fham (FermionHamiltonian): Hamiltonian to encode.

    Returns:
        QubitHamiltonian: Encoded Hamiltonian.
    """
    return TernaryTree.jkmn(fham.n_modes).encode_topphatt(fham)[0]


def jordan_wigner_annealed(
    fham: FermionHamiltonian,
    temperature: int | None = None,
    initial_guess: list[int] | None = None,
    coefficient_weighted: bool = True,
    seed: int | None = None,
) -> QubitHamiltonian:
    """Annealing-optimised Jordan-Wigner Encoding.

    Args:
        fham (FermionHamiltonian): Hamiltonian to encode.
        temperature (Optional[int]): Initial annealing temperature.
        initial_guess (Optional[list[int]]): Initial mode enumeration.
        coefficient_weighted (bool): True to minimise coefficient Pauli-weight.
        seed (Optional[int]): Seed for the annealing RNG. Defaults to ``1017``
            when omitted, preserving prior behaviour.

    Returns:
        QubitHamiltonian: Encoded Hamiltonian.
    """
    return JordanWigner(fham.n_modes).encode_annealed(
        fham, temperature, initial_guess, coefficient_weighted, seed
    )[0]


def bravyi_kitaev_annealed(
    fham: FermionHamiltonian,
    temperature: int | None = None,
    initial_guess: list[int] | None = None,
    coefficient_weighted: bool = True,
    seed: int | None = None,
) -> QubitHamiltonian:
    """Annealing-optimised Bravyi-Kitaev Encoding.

    Args:
        fham (FermionHamiltonian): Hamiltonian to encode.
        temperature (Optional[int]): Initial annealing temperature.
        initial_guess (Optional[list[int]]): Initial mode enumeration.
        coefficient_weighted (bool): True to minimise coefficient Pauli-weight.
        seed (Optional[int]): Seed for the annealing RNG. Defaults to ``1017``
            when omitted, preserving prior behaviour.

    Returns:
        QubitHamiltonian: Encoded Hamiltonian.
    """
    return BravyiKitaev(fham.n_modes).encode_annealed(
        fham, temperature, initial_guess, coefficient_weighted, seed
    )[0]


def parity_annealed(
    fham: FermionHamiltonian,
    temperature: int | None = None,
    initial_guess: list[int] | None = None,
    coefficient_weighted: bool = True,
    seed: int | None = None,
) -> QubitHamiltonian:
    """Annealing-optimised Parity Encoding.

    Args:
        fham (FermionHamiltonian): Hamiltonian to encode.
        temperature (Optional[int]): Initial annealing temperature.
        initial_guess (Optional[list[int]]): Initial mode enumeration.
        coefficient_weighted (bool): True to minimise coefficient Pauli-weight.
        seed (Optional[int]): Seed for the annealing RNG. Defaults to ``1017``
            when omitted, preserving prior behaviour.

    Returns:
        QubitHamiltonian: Encoded Hamiltonian.
    """
    return ParityEncoding(fham.n_modes).encode_annealed(
        fham, temperature, initial_guess, coefficient_weighted, seed
    )[0]


def jkmn_annealed(
    fham: FermionHamiltonian,
    temperature: int | None = None,
    initial_guess: list[int] | None = None,
    coefficient_weighted: bool = True,
    seed: int | None = None,
) -> QubitHamiltonian:
    """Annealing-optimised Jiang-Kalev-Mruczkiewicz-Neven Encoding.

    Args:
        fham (FermionHamiltonian): Hamiltonian to encode.
        temperature (Optional[int]): Initial annealing temperature.
        initial_guess (Optional[list[int]]): Initial mode enumeration.
        coefficient_weighted (bool): True to minimise coefficient Pauli-weight.
        seed (Optional[int]): Seed for the annealing RNG. Defaults to ``1017``
            when omitted, preserving prior behaviour.

    Returns:
        QubitHamiltonian: Encoded Hamiltonian.
    """
    return JKMN(fham.n_modes).encode_annealed(
        fham, temperature, initial_guess, coefficient_weighted, seed
    )[0]
