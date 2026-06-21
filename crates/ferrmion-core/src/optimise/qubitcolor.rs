//! Encoding-agnostic parallelizability estimate for a [`QubitHamiltonian`].
//!
//! For a generic encoding we only have a `QubitHamiltonian` (a flat list of Pauli
//! strings), so parallelizability is estimated by colouring a conflict graph built
//! directly over the Pauli terms — an edge between two terms that cannot share a
//! parallel layer. Two notions of "cannot share":
//!
//! - [`QubitConflict::Support`]: their Pauli supports overlap (act on a common
//!   non-identity qubit). Disjoint-support terms trivially act in parallel.
//! - [`QubitConflict::Commutation`]: they anticommute. Commuting terms can be
//!   simultaneously diagonalised / grouped.
//!
//! There is no routing, so the conflict graph is fixed and a single deterministic
//! greedy colouring suffices. The graph is built and coloured directly over a
//! bit-packed symplectic representation of the Pauli terms, avoiding any hashing:
//! each term is two bitmasks (`x`, `z`) of `⌈n_qubits / 64⌉` `u64` words, the
//! adjacency is a packed bitset, and a degree-ordered greedy colours it in place.

use crate::hamiltonians::QubitHamiltonian;
use rayon::prelude::*;

/// Condition under which two Pauli terms conflict (need different colours) in
/// [`qubit_coloring`].
#[derive(Clone, Copy)]
pub enum QubitConflict {
    /// Conflict iff the terms' Pauli supports overlap (share a non-identity qubit).
    Support,
    /// Conflict iff the terms anticommute.
    Commutation,
}

/// Bit-packed symplectic form of a set of dense Pauli strings: for each term, the
/// `x` mask (set where the factor is `X` or `Y`) and the `z` mask (set where it is
/// `Z` or `Y`), stored row-major with `words` `u64`s per mask.
struct Symplectic {
    x: Vec<u64>,
    z: Vec<u64>,
    words: usize,
}

impl Symplectic {
    /// Pack the given Pauli strings (assumed equal length) into symplectic masks.
    fn pack(terms: &[&str]) -> Self {
        let n_qubits = terms.first().map_or(0, |t| t.len());
        let words = n_qubits.div_ceil(64).max(1);
        let mut x = vec![0u64; terms.len() * words];
        let mut z = vec![0u64; terms.len() * words];
        for (i, term) in terms.iter().enumerate() {
            let (xrow, zrow) = (&mut x[i * words..], &mut z[i * words..]);
            for (q, byte) in term.bytes().enumerate() {
                let (word, bit) = (q / 64, q % 64);
                match byte {
                    b'X' => xrow[word] |= 1 << bit,
                    b'Z' => zrow[word] |= 1 << bit,
                    b'Y' => {
                        xrow[word] |= 1 << bit;
                        zrow[word] |= 1 << bit;
                    }
                    _ => {}
                }
            }
        }
        Symplectic { x, z, words }
    }

    fn masks(&self, i: usize) -> (&[u64], &[u64]) {
        let r = i * self.words..(i + 1) * self.words;
        (&self.x[r.clone()], &self.z[r])
    }

    /// True if terms `i` and `j` anticommute (odd symplectic inner product).
    fn anticommute(&self, i: usize, j: usize) -> bool {
        let ((xi, zi), (xj, zj)) = (self.masks(i), self.masks(j));
        let mut parity = 0u32;
        for w in 0..self.words {
            parity ^= ((xi[w] & zj[w]) ^ (zi[w] & xj[w])).count_ones();
        }
        parity & 1 == 1
    }

    /// True if terms `i` and `j` act on a common non-identity qubit.
    fn supports_overlap(&self, i: usize, j: usize) -> bool {
        let ((xi, zi), (xj, zj)) = (self.masks(i), self.masks(j));
        (0..self.words).any(|w| (xi[w] | zi[w]) & (xj[w] | zj[w]) != 0)
    }

    fn conflicts(&self, i: usize, j: usize, conflict: QubitConflict) -> bool {
        match conflict {
            QubitConflict::Support => self.supports_overlap(i, j),
            QubitConflict::Commutation => self.anticommute(i, j),
        }
    }
}

/// Greedily colour the conflict graph over `terms` (already in the desired,
/// deterministic node order) under `conflict`, returning the colour of each term.
///
/// Builds a packed bitset adjacency (no hashing) in parallel, then runs a
/// degree-ordered greedy colouring directly on it. Returns an empty vector when
/// there are no terms.
fn colour_conflicts(terms: &[&str], conflict: QubitConflict) -> Vec<usize> {
    let n = terms.len();
    if n == 0 {
        return Vec::new();
    }
    let sym = Symplectic::pack(terms);

    // Packed symmetric adjacency: bit `j` of row `i` set iff `i` and `j` conflict.
    let row_words = n.div_ceil(64);
    let mut adj = vec![0u64; n * row_words];
    adj.par_chunks_mut(row_words)
        .enumerate()
        .for_each(|(i, row)| {
            for j in 0..n {
                if j != i && sym.conflicts(i, j, conflict) {
                    row[j / 64] |= 1 << (j % 64);
                }
            }
        });

    // Precompute each node's degree once (popcount of its adjacency row), in
    // parallel; recomputing it inside the sort comparator would re-scan every row
    // O(n log n) times.
    let degrees: Vec<usize> = adj
        .par_chunks(row_words)
        .map(|row| row.iter().map(|w| w.count_ones() as usize).sum())
        .collect();

    // Process highest-degree first; ties broken by ascending index (deterministic).
    let mut order: Vec<usize> = (0..n).collect();
    order.par_sort_unstable_by(|&a, &b| degrees[b].cmp(&degrees[a]).then(a.cmp(&b)));

    let mut colours = vec![usize::MAX; n];
    // `seen[c] == stamp` marks colour `c` as used by the node currently colouring,
    // avoiding a per-node allocation or clear.
    let mut seen = vec![usize::MAX; n];
    for (stamp, &node) in order.iter().enumerate() {
        let row = &adj[node * row_words..(node + 1) * row_words];
        for (w, &word) in row.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                let nbr = w * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let c = colours[nbr];
                if c != usize::MAX {
                    seen[c] = stamp;
                }
            }
        }
        let mut c = 0;
        while seen[c] == stamp {
            c += 1;
        }
        colours[node] = c;
    }
    colours
}

/// Estimate the parallelizability of a [`QubitHamiltonian`] by greedily colouring
/// the conflict graph of its Pauli terms under the chosen [`QubitConflict`],
/// returning the number of colours χ (parallel layers, or commuting groups).
///
/// Encoding-agnostic: works for the `QubitHamiltonian` of any encoding, but uses
/// only the flat Pauli structure (no routing). Identity-only terms are ignored.
/// Terms are assumed equal length (the qubit count of one Hamiltonian).
pub fn qubit_coloring(hamiltonian: &QubitHamiltonian, conflict: QubitConflict) -> usize {
    // Deterministic node order (HashMap iteration is unordered); drop identity terms.
    let mut terms: Vec<&str> = hamiltonian
        .0
        .keys()
        .map(String::as_str)
        .filter(|term| term.bytes().any(|c| c != b'I'))
        .collect();
    terms.sort_unstable();

    colour_conflicts(&terms, conflict)
        .into_iter()
        .max()
        .map_or(0, |c| c + 1)
}

/// Partition a [`QubitHamiltonian`]'s terms into parallel/commuting groups by
/// greedily colouring the conflict graph under the chosen [`QubitConflict`].
///
/// Returns one group per colour; each group lists the indices of its terms in the
/// Hamiltonian's iteration order (the order of `keys()`/`items()`). Every term
/// index appears in exactly one group. Identity-only terms conflict with nothing
/// and are grouped freely. The grouping is deterministic regardless of the
/// underlying hash-map iteration order (terms are coloured in sorted Pauli order
/// internally, then mapped back to their original indices).
pub fn qubit_coloring_groups(
    hamiltonian: &QubitHamiltonian,
    conflict: QubitConflict,
) -> Vec<Vec<usize>> {
    let terms: Vec<&str> = hamiltonian.0.keys().map(String::as_str).collect();

    // Colour in a deterministic order, independent of hash-map iteration order.
    let mut order: Vec<usize> = (0..terms.len()).collect();
    order.sort_by(|&a, &b| terms[a].cmp(terms[b]));
    let sorted_terms: Vec<&str> = order.iter().map(|&i| terms[i]).collect();

    let colours = colour_conflicts(&sorted_terms, conflict);
    let n_colors = colours.iter().copied().max().map_or(0, |c| c + 1);
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); n_colors];
    for (node, &colour) in colours.iter().enumerate() {
        // `node` indexes `order`; map back to the term's original position.
        groups[colour].push(order[node]);
    }
    for group in &mut groups {
        group.sort_unstable();
    }
    groups
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qubit_hamiltonian(terms: &[&str]) -> QubitHamiltonian {
        use num_complex::Complex64;
        QubitHamiltonian(
            terms
                .iter()
                .map(|t| (t.to_string(), Complex64::new(1.0, 0.0)))
                .collect(),
        )
    }

    #[test]
    fn qubit_disjoint_terms_single_group() {
        // Disjoint supports → parallelisable and trivially commuting.
        let ham = qubit_hamiltonian(&["XIII", "IIXI"]);
        assert_eq!(qubit_coloring(&ham, QubitConflict::Support), 1);
        assert_eq!(qubit_coloring(&ham, QubitConflict::Commutation), 1);
    }

    #[test]
    fn qubit_overlapping_but_commuting() {
        // ZZII and XXII overlap (qubits 0,1) but commute (two anticommuting
        // factors), so the two conditions disagree.
        let ham = qubit_hamiltonian(&["ZZII", "XXII"]);
        assert_eq!(qubit_coloring(&ham, QubitConflict::Support), 2);
        assert_eq!(qubit_coloring(&ham, QubitConflict::Commutation), 1);
    }

    #[test]
    fn qubit_anticommuting_terms_two_groups() {
        // X and Z on the same qubit anticommute and overlap.
        let ham = qubit_hamiltonian(&["XIII", "ZIII"]);
        assert_eq!(qubit_coloring(&ham, QubitConflict::Support), 2);
        assert_eq!(qubit_coloring(&ham, QubitConflict::Commutation), 2);
    }

    #[test]
    fn qubit_identity_terms_ignored() {
        // The all-identity term occupies nothing and is dropped.
        let ham = qubit_hamiltonian(&["IIII", "XIII"]);
        assert_eq!(qubit_coloring(&ham, QubitConflict::Support), 1);
        assert_eq!(qubit_coloring(&ham, QubitConflict::Commutation), 1);
    }

    #[test]
    fn qubit_coloring_is_deterministic() {
        let ham = qubit_hamiltonian(&["XYZI", "ZZXX", "IXYZ", "YYII", "ZIIX"]);
        for cond in [QubitConflict::Support, QubitConflict::Commutation] {
            let a = qubit_coloring(&ham, cond);
            let b = qubit_coloring(&ham, cond);
            assert_eq!(a, b);
        }
    }

    /// Indices covering every term exactly once, sorted for comparison.
    fn flatten_sorted(groups: &[Vec<usize>]) -> Vec<usize> {
        let mut all: Vec<usize> = groups.iter().flatten().copied().collect();
        all.sort_unstable();
        all
    }

    #[test]
    fn qubit_groups_partition_all_terms() {
        let ham = qubit_hamiltonian(&["XYZI", "ZZXX", "IXYZ", "YYII", "ZIIX"]);
        for cond in [QubitConflict::Support, QubitConflict::Commutation] {
            let groups = qubit_coloring_groups(&ham, cond);
            // Every term index appears exactly once across the groups.
            assert_eq!(
                flatten_sorted(&groups),
                (0..ham.0.len()).collect::<Vec<_>>()
            );
            // Group count matches the chromatic-number estimate.
            assert_eq!(groups.len(), qubit_coloring(&ham, cond));
        }
    }

    #[test]
    fn qubit_groups_match_conflict_condition() {
        // ZZII/XXII: overlap but commute. Native iteration order is unspecified,
        // so resolve indices via the actual keys.
        let ham = qubit_hamiltonian(&["ZZII", "XXII"]);
        let keys: Vec<&String> = ham.0.keys().collect();
        let idx = |s: &str| keys.iter().position(|k| k.as_str() == s).unwrap();

        // Support: the two overlapping terms land in different groups.
        let support = qubit_coloring_groups(&ham, QubitConflict::Support);
        assert_eq!(support.len(), 2);

        // Commutation: they commute, so they share a single group.
        let commutation = qubit_coloring_groups(&ham, QubitConflict::Commutation);
        assert_eq!(commutation.len(), 1);
        assert!(commutation[0].contains(&idx("ZZII")) && commutation[0].contains(&idx("XXII")));
    }

    #[test]
    fn qubit_groups_are_deterministic() {
        let ham = qubit_hamiltonian(&["XYZI", "ZZXX", "IXYZ", "YYII", "ZIIX"]);
        for cond in [QubitConflict::Support, QubitConflict::Commutation] {
            assert_eq!(
                qubit_coloring_groups(&ham, cond),
                qubit_coloring_groups(&ham, cond)
            );
        }
    }

    // Reference (slow, string-scanning) conflict predicates the bit-packed
    // `Symplectic` form must reproduce exactly.
    fn ref_anticommute(a: &str, b: &str) -> bool {
        let disagreements = a
            .bytes()
            .zip(b.bytes())
            .filter(|(x, y)| *x != b'I' && *y != b'I' && x != y)
            .count();
        disagreements % 2 == 1
    }

    fn ref_supports_overlap(a: &str, b: &str) -> bool {
        a.bytes()
            .zip(b.bytes())
            .any(|(x, y)| x != b'I' && y != b'I')
    }

    #[test]
    fn symplectic_predicates_match_reference() {
        // Mix of widths (single- and multi-word) and all I/X/Y/Z combinations.
        let terms: Vec<String> = {
            let mut v = vec![
                "XYZI".to_string(),
                "ZZXX".to_string(),
                "IXYZ".to_string(),
                "YYII".to_string(),
                "ZIIX".to_string(),
                "IIII".to_string(),
            ];
            // A 70-qubit case to exercise the multi-`u64` path.
            v.push("X".repeat(35) + &"Y".repeat(35));
            v.push("Z".repeat(70));
            v
        };
        // Group by length so packing sees equal-width strings.
        for width_terms in [&terms[..6], &terms[6..]] {
            let refs: Vec<&str> = width_terms.iter().map(String::as_str).collect();
            let sym = Symplectic::pack(&refs);
            for i in 0..refs.len() {
                for j in 0..refs.len() {
                    assert_eq!(
                        sym.anticommute(i, j),
                        ref_anticommute(refs[i], refs[j]),
                        "anticommute mismatch for {} vs {}",
                        refs[i],
                        refs[j]
                    );
                    assert_eq!(
                        sym.supports_overlap(i, j),
                        ref_supports_overlap(refs[i], refs[j]),
                        "support mismatch for {} vs {}",
                        refs[i],
                        refs[j]
                    );
                }
            }
        }
    }

    #[test]
    fn colouring_is_a_valid_proper_colouring() {
        // Every coloured pair that conflicts must receive different colours, and the
        // colour count must match `qubit_coloring`.
        let ham = qubit_hamiltonian(&[
            "XYZI", "ZZXX", "IXYZ", "YYII", "ZIIX", "XXYY", "ZIZI", "IYIY", "XZXZ", "YXYX",
        ]);
        for cond in [QubitConflict::Support, QubitConflict::Commutation] {
            let terms: Vec<&str> = {
                let mut t: Vec<&str> = ham.0.keys().map(String::as_str).collect();
                t.sort_unstable();
                t
            };
            let colours = colour_conflicts(&terms, cond);
            let sym = Symplectic::pack(&terms);
            for i in 0..terms.len() {
                for j in (i + 1)..terms.len() {
                    if sym.conflicts(i, j, cond) {
                        assert_ne!(
                            colours[i], colours[j],
                            "{} and {} conflict but share a colour",
                            terms[i], terms[j]
                        );
                    }
                }
            }
            let n_colours = colours.iter().copied().max().map_or(0, |c| c + 1);
            assert_eq!(n_colours, qubit_coloring(&ham, cond));
        }
    }
}
