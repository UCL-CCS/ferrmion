"""Molecular Hamiltonian."""

import logging

import numpy as np
from numpy.typing import NDArray

from ferrmion import FermionQubitEncoding, symplectic_product
from ferrmion.utils import icount_to_sign, symplectic_hash, symplectic_unhash

from .utils import fill_template, symplectic_product_map, to_qubit_hamiltonian

logger = logging.getLogger(__name__)


def molecular_hamiltonian_template(
    ipowers: NDArray[np.number],
    majorana_symplectic: NDArray[bool],
) -> dict[str, dict[tuple[int, int] | tuple[int, int, int, int], np.complexfloating]]:
    """Build a map of operators in the full hamiltonian to their constituent majoranas."""
    logger.debug("Building hamiltonian template")
    n_modes = majorana_symplectic.shape[0] // 2
    n_qubits = majorana_symplectic.shape[1] // 2

    icount, sym_products = symplectic_product_map(
        ipowers=ipowers, symplectics=majorana_symplectic
    )
    hamiltonian: dict[
        str, dict[tuple[int, int] | tuple[int, int, int, int], np.complexfloating]
    ] = {}

    # am+ an+ ak- al-
    # (2m - i 2m+1)(2n -i 2n+1)(2k +i 2k+1)(2l +i 2l+1)
    # (l1 -i l2 -i l3 - l4)(r1 +i r2 +i r3 - r4)
    for m in range(n_modes):
        for n in range(n_modes):
            # Skip double applicatons of operators
            # if encoding.one_e_coeffs[m,n] == 0:
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

            # Two e terms cancel
            if m == n:
                continue
            for k in range(n_modes):
                for l in range(n_modes):
                    if k == l:
                        continue

                    # include the imaginary factors with terms in a tuple
                    creation_terms = [
                        (0 + icount[2 * m, 2 * n], first_term),
                        (
                            3 + icount[2 * m, 2 * n + 1],
                            second_term,
                        ),
                        (
                            3 + icount[2 * m + 1, 2 * n],
                            third_term,
                        ),
                        (
                            2 + icount[2 * m + 1, 2 * n + 1],
                            fourth_term,
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
                    # as we can arrive at the same `product` from each term and its HC
                    prefactor = 0.0625  # 1/16
                    for left_im, left_term in creation_terms:
                        for right_im, right_term in annihiliation_terms:
                            imaginary, product = symplectic_product(
                                symplectic_unhash(left_term, 2 * n_qubits),
                                symplectic_unhash(right_term, 2 * n_qubits),
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

    logger.debug("Completed Hamiltonian Template")
    return hamiltonian


def molecular_hamiltonian(
    encoding: FermionQubitEncoding,
    one_e_coeffs: NDArray,
    two_e_coeffs: NDArray,
    constant_energy: float,
):
    """Return an encoded electronic stucture hamiltonain with niave enumeration."""
    ipowers, majorana_symplectic = encoding._build_symplectic_matrix()
    template = molecular_hamiltonian_template(ipowers, majorana_symplectic)
    hashed_hamiltonian = fill_template(
        one_e_coeffs,
        two_e_coeffs,
        template,
        mode_op_map=encoding.default_mode_op_map,
        constant_energy=constant_energy,
    )
    return to_qubit_hamiltonian(encoding.n_qubits, hashed_hamiltonian)
