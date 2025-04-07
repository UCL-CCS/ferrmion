"""Ternary Tree fermion to qubit mappings"""

import numpy as np
from .devices import Qubit
from .base import FermionQubitEncoding
from .ternary_tree_node import TTNode, node_sorter
from .utils import symplectic_product, icount_to_sign
from functools import cached_property
import logging

logger = logging.getLogger(__name__)


class TernaryTree(FermionQubitEncoding):
    def __init__(
        self,
        one_e_coeffs: np.ndarray,
        two_e_coeffs: np.ndarray,
        qubits: set[Qubit],
        root_node: TTNode = TTNode(),
        enumeration_scheme: dict[str, tuple[int, int]] | None = None,
    ):
        super().__init__(one_e_coeffs, two_e_coeffs, qubits)
        self.root = root_node
        self.root.label = ""
        self.enumeration_scheme = enumeration_scheme

    def default_enumeration_scheme(self):
        node_strings = self.root.child_strings
        enumeration_scheme = {node: (None, None) for node in node_strings}
        enumeration_scheme = {
            k: {"mode": i, "qubit": i} for i, k in enumerate(enumeration_scheme)
        }
        return enumeration_scheme

    def _valid_qubit_number(self):
        return len(self.fermionic_modes) == len(self.qubits)

    def as_dict(self):
        return self.root.as_dict()

    def add_node(self, node_string: str):
        node_string = node_string.lower()
        valid_string = np.all([char in ["x", "y", "z"] for char in node_string])
        if not valid_string:
            raise ValueError("Branch string can only contain x,y,z")

        node = self.root
        for char in node_string:
            if isinstance(getattr(node, char), TTNode):
                node = getattr(node, char)
            else:
                self._next_qubit()
                node = node.add_child(char, node.label + char)
        return self

    @property
    def branch_operator_map(self):
        if self.enumeration_scheme is None:
            raise ValueError("enumeration scheme not set")

        branches = self.root.branch_strings

        nodes = self.root.child_strings
        node_indices = {node: i for i, node in enumerate(nodes)}

        branch_operator_map = {}
        num_qubits = len(self.qubits)
        for branch in branches:
            branch_operator_map[branch] = ["I"] * num_qubits
            node = self.root
            for char in branch:
                node_index = node_indices[node.label]
                # qubit_index = self.enumeration_scheme[node.label]["qubit"]
                branch_operator_map[branch][node_index] = char.upper()
                node = getattr(node, char, None)
            branch_operator_map[branch] = "".join(branch_operator_map[branch])

        return branch_operator_map

    @cached_property
    def string_pairs(self):
        """Return the pair of branch strings which correspond to each node.

        Returns:
            dict[str, tuple(str,str)]: A dictionary of all node labels, j,  with branch strings (2j, 2j+1).
        """
        node_set = self.root.child_strings

        pairs = {}
        for node_string in node_set:
            node = self.root
            for char in node_string:
                node = getattr(node, char)

            x_string = node_string + "x"
            y_string = node_string + "y"
            if x_string in node_set:
                while True:
                    x_string += "z"
                    if x_string not in node_set:
                        break

            if y_string in node_set:
                while True:
                    y_string += "z"
                    if y_string not in node_set:
                        break

            if x_string.count("y") % 2 == 0:
                pairs[node.label] = x_string, y_string
            elif y_string.count("y") % 2 == 0:
                pairs[node.label] = y_string, x_string

        return pairs

    def _build_symplectic_matrix(self):
        # If there isn't one provided, assume the naive one
        if self.enumeration_scheme is None:
            self.enumeration_scheme = self.default_enumeration_scheme()

        pauli_string_map = self.branch_operator_map
        n_qubits = len(self.qubits)
        symplectic = np.zeros((2 * n_qubits, 2 * n_qubits), dtype=np.byte)
        ipowers = np.zeros((2 * n_qubits), dtype=np.uint8)
        for node, operators in self.string_pairs.items():
            for offset, operator in enumerate(operators):
                operator = pauli_string_map[operator]
                operator = np.array(list(operator))
                # If the string is X or Y then assign 1
                term_ipower, symplectic_term = self._pauli_to_symplectic(operator)
                fermion_mode = self.enumeration_scheme[node]["mode"]
                ipowers[2 * fermion_mode + offset] = term_ipower
                symplectic[2 * fermion_mode + offset] = symplectic_term
        return ipowers, symplectic

    def JordanWigner(self):
        new_tree = TernaryTree(
            one_e_coeffs=self.one_e_coeffs,
            two_e_coeffs=self.two_e_coeffs,
            qubits=self.qubits,
            root_node=TTNode(),
        )
        new_tree.add_node("z" * (len(self.qubits) - 1))
        new_tree.enumeration_scheme = new_tree.default_enumeration_scheme()
        return new_tree

    def JW(self):
        return self.JordanWigner()

    def ParityEncoding(self):
        new_tree = TernaryTree(
            one_e_coeffs=self.one_e_coeffs,
            two_e_coeffs=self.two_e_coeffs,
            qubits=self.qubits,
            root_node=TTNode(),
        )
        new_tree.add_node("x" * (len(self.qubits) - 1))
        new_tree.enumeration_scheme = new_tree.default_enumeration_scheme()
        return new_tree

    def BravyiKitaev(self):
        new_tree = TernaryTree(
            one_e_coeffs=self.one_e_coeffs,
            two_e_coeffs=self.two_e_coeffs,
            qubits=self.qubits,
            root_node=TTNode(),
        )
        branches = ["x"]
        # one is used for root, which is defined
        remaining_qubits = len(self.qubits) - 1
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

    def BK(self):
        return self.BravyiKitaev()

    def JKMN(self):
        new_tree = TernaryTree(
            one_e_coeffs=self.one_e_coeffs,
            two_e_coeffs=self.two_e_coeffs,
            qubits=self.qubits,
            root_node=TTNode(),
        )
        branches = ["x", "y", "z"]
        # one is used for root which is defined
        remaining_qubits = len(self.qubits) - 1
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

    @cached_property
    def _build_one_e_hamiltonian(self):
        """Construct the symplectic representation of the one electron terms.

        NOTE: This assumes we are using the full Electronic Structure Hamiltonian.
        """
        icount, sym_products = self.symplectic_product_map
        symplectic_hamiltonian = {s: 0 for s in sym_products.values()}

        for m in range(self.one_e_coeffs.shape[0]):
            for n in range(self.one_e_coeffs.shape[1]):
                factor = 0.25  # if m == n else 0.25
                coefficient = factor * self.one_e_coeffs[m, n]
                # (gamma_2m -i gamma_2m+1)(gamma_2n +i gamma_2n+1)
                first_term = sym_products[(2 * m, 2 * n)]
                second_term = sym_products[(2 * m, 2 * n + 1)]
                third_term = sym_products[(2 * m + 1, 2 * n)]
                fourth_term = sym_products[(2 * m + 1, 2 * n + 1)]

                # first we add to the hamiltonian using the bitpack value
                # we'll unpack these once at the end for human
                # readable format
                symplectic_hamiltonian[first_term] += coefficient * icount_to_sign(
                    icount[2 * m, 2 * n]
                )
                symplectic_hamiltonian[second_term] += coefficient * icount_to_sign(
                    icount[2 * m, 2 * n + 1] + 1
                )
                symplectic_hamiltonian[third_term] += coefficient * icount_to_sign(
                    icount[2 * m + 1, 2 * n] + 3
                )
                symplectic_hamiltonian[fourth_term] += coefficient * icount_to_sign(
                    icount[2 * m + 1, 2 * n + 1]
                )

        return {k: v for k, v in symplectic_hamiltonian.items() if abs(v) > 1e-16}

    @cached_property
    def _build_two_e_hamiltonian(self):
        """Construct the symplectic representation of the two electron terms.

        NOTE: This uses PHYSICISTS's notation.
        """
        icount, sym_products = self.symplectic_product_map
        symplectic_hamiltonian = {}

        # am+ an+ ak- al-
        # (2m - i 2m+1)(2n -i 2n+1)(2k +i 2k+1)(2l +i 2l+1)
        # (l1 -i l2 -i l3 - l4)(r1 +i r2 +i r3 - r4)
        for m in range(self.two_e_coeffs.shape[0]):
            for n in range(self.two_e_coeffs.shape[1]):
                # Skip double applicatons of operators
                if m == n:
                    continue
                for k in range(self.two_e_coeffs.shape[2]):
                    for l in range(self.two_e_coeffs.shape[3]):
                        if k == l:
                            continue
                        coefficient = self.two_e_coeffs[m, n, k, l]
                        if coefficient == 0:
                            continue

                        # include the imaginary factors with terms in a tuple
                        creation_terms = [
                            (0 + icount[2 * m, 2 * n], sym_products[(2 * m, 2 * n)]),
                            (
                                3 + icount[2 * m, 2 * n + 1],
                                sym_products[(2 * m, 2 * n + 1)],
                            ),
                            (
                                3 + icount[2 * m + 1, 2 * n],
                                sym_products[(2 * m + 1, 2 * n)],
                            ),
                            (
                                2 + icount[2 * m + 1, 2 * n + 1],
                                sym_products[(2 * m + 1, 2 * n + 1)],
                            ),
                        ]
                        annihiliation_terms = [
                            (0 + icount[2 * k, 2 * l], sym_products[(2 * k, 2 * l)]),
                            (
                                1 + icount[2 * k, 2 * l + 1],
                                sym_products[(2 * k, 2 * l + 1)],
                            ),
                            (
                                1 + icount[2 * k + 1, 2 * l],
                                sym_products[(2 * k + 1, 2 * l)],
                            ),
                            (
                                2 + icount[2 * k + 1, 2 * l + 1],
                                sym_products[(2 * k + 1, 2 * l + 1)],
                            ),
                        ]

                        # In the symplectic form, the coefficients actually carry around an imaginary factor
                        # for pauli terms with an odd number of Ys
                        # So we need to account for taking the hermitian conjugate
                        # as we can arrive at the same 'product' from each term and its HC
                        prefactor = 1.0 / 16
                        for left_im, left_term in creation_terms:
                            for right_im, right_term in annihiliation_terms:
                                imaginary, product = symplectic_product(
                                    np.fromstring(
                                        left_term[1:-1], dtype=np.uint8, sep=" "
                                    ),
                                    np.fromstring(
                                        right_term[1:-1], dtype=np.uint8, sep=" "
                                    ),
                                )

                                product = np.array2string(product)

                                if symplectic_hamiltonian.get(product, False) is False:
                                    symplectic_hamiltonian[product] = 0

                                symplectic_hamiltonian[product] += (
                                    prefactor
                                    * coefficient
                                    * icount_to_sign(imaginary + left_im + right_im)
                                )
        return {k: v for k, v in symplectic_hamiltonian.items()}

    @property
    def symplectic_product_map(self):
        ipowers, symplectics = self._build_symplectic_matrix()
        product_ipowers = np.zeros(symplectics.shape, dtype=np.uint8)

        product_map = {}
        # For each product we need to keep track of the
        # imaginary factor, so that we can combine this with the
        # correct prefactor for each fermionic operation
        for m in range(symplectics.shape[0]):
            for n in range(symplectics.shape[0]):
                imaginary, term = symplectic_product(symplectics[m], symplectics[n])
                product_ipowers[m, n] = (imaginary + ipowers[m] + ipowers[n]) % 4
                product_map[(m, n)] = np.array2string(np.copy(term))

        return product_ipowers, product_map

    def to_symplectic_hamiltonian(self):
        """Output the hamiltonian in symplectic form.

        Remember, in symplectic form representation of XZ is literal.
        Convcerting to a Y will require an additional term.
        """
        one_e_ham = self._build_one_e_hamiltonian
        two_e_ham = self._build_two_e_hamiltonian

        total_ham = {k: v for k, v in one_e_ham.items() if v != 0}
        for k, v in two_e_ham.items():
            if v == 0:
                continue
            if total_ham.get(k, False) is False:
                total_ham[k] = 0
            total_ham[k] += v

        coeffs = []
        terms = []
        for term, coeff in total_ham.items():
            term = np.fromstring(term[1:-1], dtype=np.uint8, sep=" ")
            half_length = len(term) // 2
            y_count = np.sum(np.bitwise_and(term[half_length:], term[:half_length]))
            coeff = icount_to_sign(y_count * 3) * coeff
            if y_count % 2 == 1:
                coeff = (coeff + np.conj(coeff)) / 2

            if coeff != 0:
                coeffs.append(coeff)
                terms.append(term)

        terms = np.vstack(tuple(terms))
        return coeffs, terms

    def to_qubit_hamiltonian(self):
        one_e_ham = self._build_one_e_hamiltonian
        two_e_ham = self._build_two_e_hamiltonian

        total_ham = {k: v for k, v in one_e_ham.items() if v != 0}
        for k, v in two_e_ham.items():
            if v == 0:
                continue

            if total_ham.get(k, False) is False:
                total_ham[k] = 0
            total_ham[k] += v

        pauli_hamiltonian = {}
        for term, coefficient in total_ham.items():
            if np.real(coefficient) == 0:
                continue

            unhashed_symplectic = np.fromstring(term[1:-1], dtype=np.uint8, sep=" ")
            ipower, pauli_term = self._symplectic_to_pauli(unhashed_symplectic)
            coefficient = icount_to_sign(ipower) * coefficient
            coefficient = (coefficient + np.conj(coefficient)) / 2

            if coefficient == 0:
                continue

            if pauli_hamiltonian.get(pauli_term, None) is not None:
                pauli_hamiltonian[pauli_term] += coefficient
            else:
                pauli_hamiltonian[pauli_term] = coefficient
        return pauli_hamiltonian
