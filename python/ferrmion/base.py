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
        vacuum_state: np.ndarray | None = None,
    ):
        self.one_e_coeffs: np.ndarray = one_e_coeffs
        self.two_e_coeffs: np.ndarray = two_e_coeffs
        self.vacuum_state = vacuum_state
        self.modes = {m for m in range(self.one_e_coeffs.shape[0])}
    
    def __post_init__(self):
        self._one_e_hamiltonian_template
        self._two_e_hamiltonian_template

    @property
    def one_e_coeffs(self):
        return self._one_e_coeffs
    
    @one_e_coeffs.setter
    def one_e_coeffs(self, coefficients):
        size_valid = np.all(
            [coefficients.shape[0] == size for size in coefficients.shape]
        )
        if size_valid:
            self._one_e_coeffs = coefficients
        else:
            raise ValueError(
                f"One electron integrals not valid."
            )
        
    @property
    def two_e_coeffs(self):
        return self._two_e_coeffs
    
    @two_e_coeffs.setter
    def two_e_coeffs(self, coefficients):
        size_valid = np.all(
            [coefficients.shape[0] == size for size in coefficients.shape]
        )
        if size_valid:
            self._two_e_coeffs = coefficients
        else:
            raise ValueError(
                f"Two electron integrals not valid."
            )

    @property
    def vacuum_state(self):
        return self._vacuum_state

    @vacuum_state.setter
    def vacuum_state(self, state:np.ndarray):
        """Validate and set the vacuum state."""
        logger.debug("Setting vacuum state as %s", state)
        error_string = []
        state = np.array(state) if type(state) is not np.ndarray else state

        if len(state) != self.n_qubits:
            error_string.append("vacuum state must be length " + str(self.n_qubits))
        if state.ndim != 1:
            error_string.append("vacuum state must be vector (dimension==1)")
        
        if error_string != []:
            logger.error("\n".join(error_string))
            raise ValueError("\n".join(error_string))
        else:
            self._vacuum_state = state

    @property
    @abstractmethod
    def default_mode_op_map(self):
        """Define a default map from modes to majorana operator pairs i->(j,j+1)."""
        pass

    @abstractmethod
    def _build_symplectic_matrix(
        self,
    ) -> tuple[np.ndarray[np.uint8], np.ndarray[np.uint8]]:
        """Build a symplectic matrix representing terms for each operator in the Hamitonian."""
        pass
    
    def hartree_fock_state(self, fermionic_hf_state:np.ndarray, mode_op_map:dict | None = None):
        """Find the Hartree-Fock state of a majorana string encoding.
        
        Args:
            fermionic_hf_state (np.ndarray[int]): An array of mode occupations.
            mode_op_map (dict[int, int]): A dictionary mapping modes to sets of majorana strings i->(j,j+1).
    
        Returns:
            np.ndarray: The Hartree-Fock ground state in computational basis.
        """
        return hartree_fock_state(self, fermionic_hf_state, mode_op_map)


    @staticmethod
    def _symplectic_to_pauli(symplectic: np.ndarray) -> str:
        return symplectic_to_pauli(symplectic)

    @staticmethod
    def _pauli_to_symplectic(pauli: str) -> tuple[int, np.ndarray[np.uint8, np.uint8]]:
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

        n_modes = majorana_symplectic.shape[1]//2

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
                    0.25 * icount_to_sign(icount[2 * m, 2 * n+1]+1),
                    0.25 * icount_to_sign(icount[2 * m+1, 2 * n]+3), 
                    0.25 * icount_to_sign(icount[2 * m+1, 2 * n+1]), 
                                                    )
                terms = [first_term, second_term, third_term, fourth_term]

                for t, f in zip(terms, factors):
                    hamiltonian[t]= hamiltonian.get(t, {})
                    hamiltonian[t][(m,n)] = hamiltonian[t].get((m,n), 0) + f

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
                                    np.fromstring(
                                        left_term[1:-1], dtype=np.uint8, sep=" "
                                    ),
                                    np.fromstring(
                                        right_term[1:-1], dtype=np.uint8, sep=" "
                                    ),
                                )

                                product = np.array2string(product)

                                hamiltonian[product] = hamiltonian.get(
                                    product, {}
                                )
                                # ordered = (1 if m > n else -1) * (1 if k > l else -1)
                                weight = prefactor * icount_to_sign(imaginary + left_im + right_im)
                                # index = tuple(sorted([m, n]) + sorted([k, l]))
                                index = (m,n,k,l)
                                
                                hamiltonian[product][index] = hamiltonian[product].get(index, 0) + weight
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
                product_map[(m, n)] = np.array2string(np.copy(term))

        return product_ipowers, product_map

    def fill_template(self, mode_op_map:dict) -> dict:
        logger.debug(f"Filling template with map\n{mode_op_map}")
        one_e_ham = self._one_e_hamiltonian_template
        two_e_ham = self._two_e_hamiltonian_template

        all_terms = set(one_e_ham).union(two_e_ham)

        total_ham = {t:0 for t in all_terms}
        for term in total_ham:
            one_e_part = one_e_ham.get(term, {})
            for item, factor in one_e_part.items():
                total_ham[term] += factor * self.one_e_coeffs[*[mode_op_map[i] for i in item]]

            two_e_part = two_e_ham.get(term, {})
            for item, factor in two_e_part.items():
                total_ham[term] += factor * self.two_e_coeffs[*[mode_op_map[i] for i in item]]

            # print(total_ham[term])
            # if total_ham[term] == 0:
                # total_ham.pop(term)
        return total_ham

    def to_symplectic_hamiltonian(self, mode_op_map:dict=None):
        """Output the hamiltonian in symplectic form.

        Remember, in symplectic form representation of XZ is literal.
        Convcerting to a Y will require an additional term.

        Args:
            mode_op_map (dict): A dictionary mapping the mode indices to their corresponding qubit indices.
        """
        logger.debug("Creating symplectic Hamiltonian")
        if mode_op_map is None:
            logger.debug("No mode operator map provided, using default")
            mode_op_map = self.default_mode_op_map
        total_ham = self.fill_template(mode_op_map)

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


    def to_qubit_hamiltonian(self, mode_op_map:dict=None):
        """Create qubit representation Hamiltonian."""
        logger.debug("Creating qubit Hamiltonian")
        if mode_op_map is None:
            logger.debug("No mode operator map provided, using default")
            mode_op_map = self.default_mode_op_map
            
        total_ham = self.fill_template(mode_op_map)

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


def edge_operator_map(encoding: FermionQubitEncoding) -> tuple[dict, dict]:
    """Build a map of operators in the full hamiltonian to their constituent majoranas."""
    logger.debug("Building edge operator map")
    majorana_symplectic = encoding._build_symplectic_matrix()[1]

    icount, sym_products = encoding.symplectic_product_map
    edge_map = {}

    n_modes = majorana_symplectic.shape[1]//2

    for m in range(n_modes):
        for n in range(n_modes):
            # if self.one_e_coeffs[m,n] == 0:
                # continue
            
            factor = 0.25  # if m == n else 0.25
            # (gamma_2m -i gamma_2m+1)(gamma_2n +i gamma_2n+1)
            first_term = sym_products[(2 * m, 2 * n)]
            second_term = sym_products[(2 * m, 2 * n + 1)]
            third_term = sym_products[(2 * m + 1, 2 * n)]
            fourth_term = sym_products[(2 * m + 1, 2 * n + 1)]

            factors = (
                icount_to_sign(icount[2 * m, 2 * n]), 
                icount_to_sign(icount[2 * m, 2 * n+1]+1),
                icount_to_sign(icount[2 * m+1, 2 * n]+3), 
                icount_to_sign(icount[2 * m+1, 2 * n+1]), 
                                                )
            terms = [first_term, second_term, third_term, fourth_term]

            if m <= n:
                edge_map[(m,n)] = {term: factor for term, factor in zip(terms, factors)}
            # The other way round will always come second!
            else:
                for t, f in zip(terms, factors):
                    edge_map[(n,m)][t] += f
                    if edge_map[(n,m)][t] == 0:
                        edge_map[(n,m)].pop(t)


    logger.debug("Calculating mean weights")
    weights = np.zeros((n_modes, n_modes))
    for k,v in edge_map.items():
        x_block, z_block = np.hsplit(np.vstack([np.fromstring(op[1:-1], dtype=np.uint8, sep=" ") for op in v.keys()]),2)
        
        mean_weight = np.mean(np.sum(np.bitwise_or(x_block, z_block), axis=1) * [factor for factor in v.values()])
        weights[k[0], k[1]] = mean_weight
        weights[k[1], k[0]] = mean_weight

    return edge_map, weights

def two_operator_product(creation: tuple[bool, bool], left, right):
    # (a+ib)(c+id) -> ac, iad, ibc, -bd
    first_term = symplectic_product(left[:,0], right[:,0])
    second_term = symplectic_product(left[:,0], right[:,1])
    third_term = symplectic_product(left[:,1], right[:,0])
    fourth_term = symplectic_product(left[:,1], right[:,1])

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
    
def hartree_fock_state(encoding: FermionQubitEncoding, fermionic_hf_state:np.ndarray, mode_op_map:dict[int,int]) -> np.ndarray:
    """Find the Hartree-Fock state of a majorana string encoding.
    
    Args:
        encoding (FermionQubitEncoding): a Majoprana string encoding.
        fermionic_hf_state (np.ndarray[int]): An array of mode occupations.
        mode_op_map (dict[int, int]): A dictionary mapping modes to sets of majorana strings i->(j,j+1).
    
    Returns:
        np.ndarray: The Hartree-Fock ground state in computational basis.
    """
    mode_op_map = encoding.default_mode_op_map if mode_op_map is None else mode_op_map
    if len(fermionic_hf_state) != len(mode_op_map):
        error_string = "Fermionic HF state must be same length as Mode-Operator Map"
        logger.error(error_string)
        raise ValueError(error_string)

    matrices = {
        "X": np.array([[0,1],[1,0]]),
        "XZ": np.array([[0,-1j],[1j,0]]),
        "Z": np.array([[1,0],[0,-1]]),
        "": np.array([[1,0],[0,1]])
    }

    logger.debug("Creating computational basis HF state")

    modes = [mode_op_map[mode] for mode in np.where(fermionic_hf_state)[0]]
    symplectic_operators = encoding._build_symplectic_matrix()[1]
    symplectic_operators = np.vstack(symplectic_operators[[(2*mode, 2*mode+1) for mode in modes]])

    half_length = symplectic_operators.shape[1] // 2
    vacuum_state = [np.array([1,0]) if site==0 else np.array([0,1]) for site in encoding.vacuum_state]
    for index in range(0,symplectic_operators.shape[0],2):
        xlist = ["X" if line == 1 else "" for line in symplectic_operators[index,:half_length]]
        zlist = ["Z" if line == 1 else "" for line in symplectic_operators[index,half_length:]]
        left_operators = [matrices[f"{x}{z}"] for x, z in zip(xlist, zlist)]
        # ipower = (3 * y_count) % 4
        xlist = ["X" if line == 1 else "" for line in symplectic_operators[index+1,:half_length]]
        zlist = ["Z" if line == 1 else "" for line in symplectic_operators[index+1,half_length:]]
        right_operators = [1j*matrices[f"{x}{z}"] for x, z in zip(xlist, zlist)] 

        total_ops = [left-right for left,right in zip(left_operators, right_operators)]

        vacuum_state = [op @ state for op, state in zip(total_ops, vacuum_state)]
    # vacuum_state = [(v*np.conj(v))/np.linalg.norm(v*np.conj(v)) for v in vacuum_state]
    total_state = vacuum_state[0]
    for state in vacuum_state[1:]:
        total_state = np.kron(total_state, state)

    coeffs = (total_state*np.conj(total_state))/np.linalg.norm(total_state*np.conj(total_state))
    hf_components = np.vstack([np.array(list(np.binary_repr(val, width=len(vacuum_state))), dtype=np.uint8) for val in np.where(coeffs)[0]])
    coeffs = [c for c in coeffs if c != 0]

    return coeffs, hf_components