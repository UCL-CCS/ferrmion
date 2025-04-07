import numpy as np
from functools import cached_property
from abc import ABC, abstractmethod
from typing import Hashable
from .devices import Qubit
import logging
from .utils import (
    icount_to_sign,
    symplectic_product,
    symplectic_to_pauli,
    pauli_to_symplectic,
)

logger = logging.getLogger(__name__)


class FermionQubitEncoding(ABC):
    """Fermion Encodings for the Electronic Structure Hamiltonian in symplectic form.

    NOTE: A 'Y' pauli operator is mapped to -iXY so a (0+n)**3 term is needed.
    """

    def __init__(
        self,
        one_e_coeffs: np.ndarray,
        two_e_coeffs: np.ndarray,
        qubit_labels: set[Qubit] = None,
        mode_labels: set[int] = None,
    ):
        self.one_e_coeffs: np.ndarray = one_e_coeffs
        self.two_e_coeffs: np.ndarray = two_e_coeffs
        self.qubits: set[Hashable] = qubit_labels
        self._qubit_iter: iter[Hashable] = iter(qubit_labels)

        self._validate_e_coeffs()
        if not self._valid_qubit_number:
            raise TypeError("Invalid number of qubits for encoding")

        self.modes = {m for m in range(self.one_e_coeffs.shape[0])}

    def _validate_e_coeffs(self):
        """Check that the one and two electron integral coefficients are the right shape."""
        one_e_valid = np.all(
            [self.one_e_coeffs.shape[0] == size for size in self.one_e_coeffs.shape]
        )
        two_e_valid = np.all(
            [self.one_e_coeffs.shape[0] == size for size in self.two_e_coeffs.shape]
        )

        if not one_e_valid and two_e_valid:
            raise ValueError(
                f"ERIs not valid {self.one_e_coeffs.size} {self.two_e_coeffs.size}"
            )

    def _next_qubit(self):
        """yield a label up to the number of available qubits"""
        try:
            label = next(self._qubit_iter)
            # logger.debug("Consuming qubit label %s", label)
            return label
        except StopIteration:
            # logger.warning("All qubits used.")
            raise StopIteration("All qubits have been assigned to modes.")

    @abstractmethod
    def _build_symplectic_matrix(
        self,
    ) -> tuple[np.ndarray[np.uint8], np.ndarray[np.uint8]]:
        """Build a symplectic matrix representing terms for each operator in the Hamitonian."""
        pass

    @property
    @abstractmethod
    def _valid_qubit_number(self):
        """Ensure the qubit number is suitable for the mapping."""
        pass

    @staticmethod
    def _symplectic_to_pauli(symplectic: np.ndarray) -> str:
        return symplectic_to_pauli(symplectic)

    @staticmethod
    def _pauli_to_symplectic(pauli: str) -> tuple[int, np.ndarray[np.uint8, np.uint8]]:
        return pauli_to_symplectic(pauli)

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

            if pauli_hamiltonian.get(pauli_term, None) is not None:
                pauli_hamiltonian[pauli_term] += coefficient
            else:
                pauli_hamiltonian[pauli_term] = coefficient

        pauli_hamiltonian = {
            k: v
            for k, v in pauli_hamiltonian.items()
            if np.real(v) != 0 or np.imag(v) != 0
        }

        return pauli_hamiltonian
