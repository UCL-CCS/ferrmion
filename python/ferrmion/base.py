"""Base FermionQubitEncoding class."""

import logging
from abc import ABC, abstractmethod
from functools import cached_property

import numpy as np
from numpy.typing import NDArray

from ferrmion import hartree_fock_state, symplectic_product

from .utils import (
    icount_to_sign,
    pauli_to_symplectic,
    symplectic_hash,
    symplectic_to_pauli,
    symplectic_unhash,
)

logger = logging.getLogger(__name__)


class FermionQubitEncoding(ABC):
    """Fermion Encodings for the Electronic Structure Hamiltonian in symplectic form.

    Attributes:
        one_e_coeffs (NDArray): One electron coefficients.
        two_e_coeffs (NDArray): Two electron coefficients.
        vaccum_state (NDArray | None): The vaccum state of the encoding.
        modes (set[int]): A set of modes.
        n_qubits (int): The number of qubits.

    Methods:
        one_e_coeffs: Get or set the one electron coefficients.
        two_e_coeffs: Get or set the two electron coefficients.
        vaccum_state: Get or set the vaccum state.
        default_mode_op_map: Get the default mode operator map.
        _build_symplectic_matrix: Build a symplectic matrix representing terms for each operator in the Hamiltonian.
        hartree_fock_state: Find the Hartree-Fock state of a majorana string encoding.
        _symplectic_to_pauli: Convert a symplectic matrix to a Pauli string.
        _pauli_to_symplectic: Convert a Pauli string to a symplectic matrix.
        _edge_operator_map: Build a map of operators in the full hamiltonian to their constituent majoranas.
        fill_template: Fill a template with Hamiltonian coefficients.
        to_symplectic_hamiltonian: Output the hamiltonian in symplectic form.
        to_qubit_hamiltonian: Create qubit representation Hamiltonian.

    NOTE: A 'Y' pauli operator is mapped to -iXY so a (0+n)**3 term is needed.
    """

    def __init__(
        self,
        one_e_coeffs: NDArray,
        two_e_coeffs: NDArray,
        vaccum_state: NDArray | None = None,
    ):
        """Initialise encoding.

        Args:
            one_e_coeffs (NDArray): One electron coefficients.
            two_e_coeffs (NDArray): Two electron coefficients.
            vaccum_state (NDArray | None): The vaccum state of the encoding.
        """
        self.one_e_coeffs: NDArray = one_e_coeffs
        self.two_e_coeffs: NDArray = two_e_coeffs
        self.vaccum_state = vaccum_state
        self.modes = {m for m in range(self.one_e_coeffs.shape[0])}

    def __post_init__(self):
        """Post init function to validate the encoding."""
        logger.debug("Post init of FermionQubitEncoding")
        self._one_e_hamiltonian_template
        self._two_e_hamiltonian_template

    @property
    def one_e_coeffs(self):
        """Return the one electron coefficients."""
        return self._one_e_coeffs

    @one_e_coeffs.setter
    def one_e_coeffs(self, coefficients):
        """Set the one electron coefficients.

        Args:
            coefficients (NDArray): The one electron coefficients.
        """
        logger.debug("Setting one electron coefficients as %s", coefficients)
        size_valid = np.all(
            [coefficients.shape[0] == size for size in coefficients.shape]
        )
        if size_valid:
            self._one_e_coeffs = coefficients
        else:
            raise ValueError("One electron integrals not valid.")

    @property
    def two_e_coeffs(self):
        """Return the two electron coefficients."""
        return self._two_e_coeffs

    @two_e_coeffs.setter
    def two_e_coeffs(self, coefficients):
        """Set the two electron coefficients.

        Args:
            coefficients (NDArray): The two electron coefficients.
        """
        logger.debug("Setting two electron coefficients as %s", coefficients)
        size_valid = np.all(
            [coefficients.shape[0] == size for size in coefficients.shape]
        )
        if size_valid:
            self._two_e_coeffs = coefficients
        else:
            raise ValueError("Two electron integrals not valid.")

    @property
    def vaccum_state(self):
        """Return the vaccum state."""
        return self._vaccum_state

    @vaccum_state.setter
    def vaccum_state(self, state: NDArray):
        """Validate and set the vaccum state.

        Args:
            state (NDArray): The vaccum state.
        """
        logger.debug("Setting vaccum state as %s", state)
        error_string = []
        state = np.array(state, dtype=np.float64)

        if len(state) != self.n_qubits:
            error_string.append("vaccum state must be length " + str(self.n_qubits))
        if state.ndim != 1:
            error_string.append("vaccum state must be vector (dimension==1)")

        if error_string != []:
            logger.error("\n".join(error_string))
            raise ValueError("\n".join(error_string))
        else:
            self._vaccum_state = state

    @property
    @abstractmethod
    def default_mode_op_map(self):
        """Define a default map from modes to majorana operator pairs i->(j,j+1)."""
        pass

    @abstractmethod
    def _build_symplectic_matrix(
        self,
    ) -> tuple[NDArray[np.number], NDArray[np.bool]]:
        """Build a symplectic matrix representing terms for each operator in the Hamitonian."""
        pass

    def hartree_fock_state(
        self, fermionic_hf_state: NDArray[np.bool], mode_op_map: dict | None = None
    ):
        """Find the Hartree-Fock state of a majorana string encoding.

        This function calls to the rust implementatin in `src/lib.rs`.
        It assumes that the vaccum state is a single state vector, though the HF state may not be
        The global phase so that the first component state has 0 phase.

        Args:
            fermionic_hf_state (NDArray[int]): An array of mode occupations.
            mode_op_map (dict[int, int]): A dictionary mapping modes to sets of majorana strings i->(j,j+1).

        Returns:
            NDArray: The Hartree-Fock ground state in computational basis.
        """
        if mode_op_map is None:
            mode_op_map = self.default_mode_op_map

        return hartree_fock_state(
            self.vaccum_state,
            fermionic_hf_state,
            mode_op_map,
            self._build_symplectic_matrix()[1],
        )

    @staticmethod
    def _symplectic_to_pauli(symplectic: NDArray) -> tuple[int, str]:
        """Convert a symplectic matrix to a Pauli string.

        Args:
            symplectic (NDArray): A symplectic vector.
        """
        return symplectic_to_pauli(symplectic)

    @staticmethod
    def _pauli_to_symplectic(pauli: str) -> tuple[int, NDArray[np.bool]]:
        """Convert a Pauli string to a symplectic matrix.

        Args:
            pauli (str): A Pauli-string.
        """
        return pauli_to_symplectic(pauli)

    def _edge_operator_map(self):
        return edge_operator_map(self)

    @cached_property
    def _one_e_hamiltonian_template(self):
        """Build a map of operators in the full hamiltonian to their constituent majoranas."""
        logger.debug("Building one electron hamiltonian template")
        majorana_symplectic = self._build_symplectic_matrix()[1]

        icount, sym_products = self.symplectic_product_map
        hamiltonian = {}

        n_modes = majorana_symplectic.shape[1] // 2

        for m in range(n_modes):
            for n in range(n_modes):
                # if self.one_e_coeffs[m,n] == 0:
                # continue

                # factor = 0.25  # if m == n else 0.25
                # (gamma_2m -i gamma_2m+1)(gamma_2n +i gamma_2n+1)
                first_term = sym_products[(2 * m, 2 * n)]
                second_term = sym_products[(2 * m, 2 * n + 1)]
                third_term = sym_products[(2 * m + 1, 2 * n)]
                fourth_term = sym_products[(2 * m + 1, 2 * n + 1)]

                factors = (
                    0.25 * icount_to_sign(icount[2 * m, 2 * n]),
                    0.25 * icount_to_sign(icount[2 * m, 2 * n + 1] + 1),
                    0.25 * icount_to_sign(icount[2 * m + 1, 2 * n] + 3),
                    0.25 * icount_to_sign(icount[2 * m + 1, 2 * n + 1]),
                )
                terms = [first_term, second_term, third_term, fourth_term]

                for t, f in zip(terms, factors):
                    hamiltonian[t] = hamiltonian.get(t, {})
                    hamiltonian[t][(m, n)] = hamiltonian[t].get((m, n), 0) + f

        return hamiltonian

    @cached_property
    def _two_e_hamiltonian_template(self):
        """Construct the symplectic representation of the two electron terms.

        NOTE: This uses PHYSICISTS's notation.

        Args:
            mode_op_map (dict): A dictionary mapping the mode indices to their corresponding qubit indices.
        """
        logger.debug("Building two electron hamiltonian template")
        icount, sym_products = self.symplectic_product_map
        hamiltonian = {}

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
                                    symplectic_unhash(left_term, 2 * self.n_qubits),
                                    symplectic_unhash(right_term, 2 * self.n_qubits),
                                )

                                product = symplectic_hash(product)

                                hamiltonian[product] = hamiltonian.get(product, {})
                                # ordered = (1 if m > n else -1) * (1 if k > l else -1)
                                weight = prefactor * icount_to_sign(
                                    imaginary + left_im + right_im
                                )
                                # index = tuple(sorted([m, n]) + sorted([k, l]))
                                index = (m, n, k, l)

                                hamiltonian[product][index] = (
                                    hamiltonian[product].get(index, 0) + weight
                                )
                                if hamiltonian[product][index] == 0:
                                    hamiltonian[product].pop(index)
                                if hamiltonian[product] == {}:
                                    hamiltonian.pop(product)
        return hamiltonian

    @property
    def symplectic_product_map(self):
        """Calculate the product of symplectic terms and cache them."""
        logger.debug("Building symplectic product map")
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
                product_map[(m, n)] = symplectic_hash(np.copy(term))

        return product_ipowers, product_map

    def fill_template(self, mode_op_map: dict) -> dict:
        """Fill a template with Hamiltonian coefficients.

        Args:
            mode_op_map (dict): A dictionary mapping the mode indices to their corresponding majorana operator indices.
        """
        logger.debug(f"Filling template with map\n{mode_op_map}")
        one_e_ham = self._one_e_hamiltonian_template
        two_e_ham = self._two_e_hamiltonian_template

        all_terms = set(one_e_ham).union(two_e_ham)

        total_ham = {t: 0 for t in all_terms}
        for term in total_ham:
            one_e_part = one_e_ham.get(term, {})
            for item, factor in one_e_part.items():
                total_ham[term] += (
                    factor * self.one_e_coeffs[*[mode_op_map[i] for i in item]]
                )

            two_e_part = two_e_ham.get(term, {})
            for item, factor in two_e_part.items():
                total_ham[term] += (
                    factor * self.two_e_coeffs[*[mode_op_map[i] for i in item]]
                )

            # print(total_ham[term])
            # if total_ham[term] == 0:
            # total_ham.pop(term)
        return total_ham

    def to_symplectic_hamiltonian(
        self, mode_op_map: dict | None = None
    ) -> tuple[list[complex], NDArray]:
        """Output the hamiltonian in symplectic form.

        Remember, in symplectic form representation of XZ is literal.
        Convcerting to a Y will require an additional term.

        Args:
            mode_op_map (dict): A dictionary mapping the mode indices to their corresponding majorana operator indices.

        Returns:
            tuple[list[complex], NDArray]: A tuple of coefficients and symplectic terms.
        """
        logger.debug("Creating symplectic Hamiltonian")
        if mode_op_map is None:
            logger.debug("No mode operator map provided, using default")
            mode_op_map = self.default_mode_op_map
        total_ham = self.fill_template(mode_op_map)

        coeffs = []
        terms = []
        for term, coeff in total_ham.items():
            term = symplectic_unhash(term, 2 * self.n_qubits)
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

    def to_qubit_hamiltonian(self, mode_op_map: dict | None = None) -> dict[str, float]:
        """Create qubit representation Hamiltonian.

        Args:
            mode_op_map (dict): A dictionary mapping the mode indices to their corresponding majorana operator indices.

        Returns:
            dict[str, float]: A dictionary of Pauli strings and their coefficients.
        """
        logger.debug("Creating qubit Hamiltonian")
        if mode_op_map is None:
            logger.debug("No mode operator map provided, using default")
            mode_op_map = self.default_mode_op_map

        total_ham = self.fill_template(mode_op_map)

        pauli_hamiltonian = {}
        for term, coefficient in total_ham.items():
            if np.real(coefficient) == 0:
                continue

            unhashed_symplectic = symplectic_unhash(term, 2 * self.n_qubits)
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


def edge_operator_map(encoding: FermionQubitEncoding) -> tuple[dict, dict]:
    """Build a map of operators in the full hamiltonian to their constituent majoranas.

    Args:
        encoding (FermionQubitEncoding): The encoding to use.

    Returns:
        tuple[dict, dict]: A tuple of the edge operator map and the weights.
    """
    logger.debug("Building edge operator map")
    majorana_symplectic = encoding._build_symplectic_matrix()[1]

    icount, sym_products = encoding.symplectic_product_map
    edge_map = {}

    n_modes = majorana_symplectic.shape[1] // 2

    for m in range(n_modes):
        for n in range(n_modes):
            # if self.one_e_coeffs[m,n] == 0:
            # continue

            # (gamma_2m -i gamma_2m+1)(gamma_2n +i gamma_2n+1)
            first_term = sym_products[(2 * m, 2 * n)]
            second_term = sym_products[(2 * m, 2 * n + 1)]
            third_term = sym_products[(2 * m + 1, 2 * n)]
            fourth_term = sym_products[(2 * m + 1, 2 * n + 1)]

            factors = (
                0.25 * icount_to_sign(icount[2 * m, 2 * n]),
                0.25 * icount_to_sign(icount[2 * m, 2 * n + 1] + 1),
                0.25 * icount_to_sign(icount[2 * m + 1, 2 * n] + 3),
                0.25 * icount_to_sign(icount[2 * m + 1, 2 * n + 1]),
            )
            terms = [first_term, second_term, third_term, fourth_term]

            if m <= n:
                edge_map[(m, n)] = {
                    term: factor for term, factor in zip(terms, factors)
                }
            # The other way round will always come second!
            else:
                for t, f in zip(terms, factors):
                    edge_map[(n, m)][t] += f
                    if edge_map[(n, m)][t] == 0:
                        edge_map[(n, m)].pop(t)

    logger.debug("Calculating mean weights")
    logger.debug(f"{edge_map}")
    weights = np.zeros((n_modes, n_modes))
    for k, v in edge_map.items():
        logger.debug(f"{[symplectic_unhash(op, 2*n_modes) for op in v.keys()]}")
        x_block, z_block = np.hsplit(
            np.vstack([symplectic_unhash(op, 2 * n_modes) for op in v.keys()]), 2
        )

        mean_weight = np.mean(
            np.sum(np.bitwise_or(x_block, z_block), axis=1)
            * [factor for factor in v.values()]
        )
        weights[k[0], k[1]] = mean_weight
        weights[k[1], k[0]] = mean_weight

    return edge_map, weights


def two_operator_product(creation: tuple[bool, bool], left, right) -> NDArray:
    """Calculate the product of two operators in symplectic form.

    Args:
        creation (tuple[bool, bool]): A tuple of two booleans indicating if the operators are creation operators.
        left (NDArray): The left operator in symplectic form.
        right (NDArray): The right operator in symplectic form.

    Returns:
        NDArray: The product of the two operators in symplectic form.

    Example:
        >>> left = np.array([[1, 0], [0, 1]])
        >>> right = np.array([[0, 1], [1, 0]])
        >>> creation = (True, False)
        >>> two_operator_product(creation, left, right)
        array([[0, 1],
               [1, 0]])
    """
    # (a+ib)(c+id) -> ac, iad, ibc, -bd
    first_term = symplectic_product(left[:, 0], right[:, 0])
    second_term = symplectic_product(left[:, 0], right[:, 1])
    third_term = symplectic_product(left[:, 1], right[:, 0])
    fourth_term = symplectic_product(left[:, 1], right[:, 1])

    # left creation -> -iad, +bd
    # right creation -> -ibc, +bd
    # both creation -> -iad, -ibc, -bd
    if creation[0] is True:
        second_term[0] += 2
        fourth_term[0] += 2
    if creation[1] is True:
        third_term[0] += 2
        fourth_term[0] += 2

    return np.vstack((first_term, second_term, third_term, fourth_term))
