//! Shared helpers used by the HATT and TOPP-HATT tree constructors.

use std::ops::BitXorAssign;
use tinyvec::ArrayVec;

pub(crate) const MAJORANA_MAX: usize = 7;

/// Find the weight of a term on the qubit of a single node.
///
/// This function is used to assess the cost of each possible choice
/// of outward edges of a given node. Each outward edge has an associated
/// index. Either a Majorana-index, or a Node-index.
///
/// Each term is composed of some number of Majorana operators.
///
/// Where a Majorana operator is included in the _children_ of a given node,
/// the Majorana operator acts on that node's qubit with a non-Identity operator.
///
/// Additionally, using [`reduce_hamiltonian`] we guarantee that for
/// [`crate::encode::ternarytree::TernaryTree`]s,
/// no two distinct indices represent Majorana operators which
/// act with the same Pauli operator.
///
/// We wish to find out whether the product of Majorana operators in a given
/// Hamiltonian term require the application of non-Identity operator.
///
/// For each Majorana operator in a term:
/// - if it is not in _children_, it acts with the Identity.
/// - if it appears an even number of times, it acts with the Identity, as: PP=I forall P in {X,Y,Z,I}
///
/// if three Majorana operators appear in both the term and _children_ with odd parity,
/// together, they act with the identity as XYZ=-iI
#[inline(always)]
pub(crate) fn qubit_term_weight(
    term: &ArrayVec<[u16; MAJORANA_MAX]>,
    sorted_children: &[u16; 3],
) -> usize {
    let mut even_parity_paulis = [true, true, true];
    unsafe {
        for t in term {
            even_parity_paulis
                .get_unchecked_mut(0)
                .bitxor_assign(t == sorted_children.get_unchecked(0));
            even_parity_paulis
                .get_unchecked_mut(1)
                .bitxor_assign(t == sorted_children.get_unchecked(1));
            even_parity_paulis
                .get_unchecked_mut(2)
                .bitxor_assign(t == sorted_children.get_unchecked(2));
        }
    }
    let odd_parity_paulis = 3
        - (even_parity_paulis[0] as usize
            + even_parity_paulis[1] as usize
            + even_parity_paulis[2] as usize);

    !odd_parity_paulis.is_multiple_of(3) as usize
}

/// Simplify the Majorana operator Hamiltonian.
///
/// As we traverse from leaves to root, we can simplify the Hamiltonian.
/// For each node we pass, we can guarantee that all Majorana operators
/// passing through that node have taken the same path to that node.
///
/// They therefore act with the same Pauli operator on every node which
/// is on that path.
///
/// We can therefore substitute a single index, representing the node, in place of
/// all the individual Majorana operator indices.
pub(crate) fn reduce_hamiltonian(
    majorana_terms: Vec<ArrayVec<[u16; MAJORANA_MAX]>>,
    parent_majorana_index: u16,
    selection: [u16; 3],
) -> Vec<ArrayVec<[u16; MAJORANA_MAX]>> {
    let mut result: Vec<ArrayVec<[u16; MAJORANA_MAX]>> = majorana_terms
        .into_iter()
        .map(|mut term| {
            let original_len = term.len();
            term.retain(|&ind| !selection.contains(&ind));
            while term.len() < original_len {
                term.push(parent_majorana_index);
            }
            term.sort_unstable();
            term
        })
        .filter(|term| !term.iter().all(|&ind| ind == parent_majorana_index))
        .collect();
    // Use sort + dedup instead of BTreeSet for deduplication:
    // avoids per-element tree insertion overhead.
    result.sort_unstable();
    result.dedup();
    result
}
