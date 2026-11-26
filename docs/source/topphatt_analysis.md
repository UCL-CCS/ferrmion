# TOPP-HATT: analysis and new strategies

This page analyses the Pauli-weight optimisation done by TOPP-HATT
(`crates/ferrmion-core/src/optimise/ternarytree/topphatt.rs`). It restates the
problem in several equivalent forms, diagnoses where the greedy loses weight,
and documents the new strategies that came out of that analysis. Everything
described here is available through the `strategy`, `lookahead`,
`depth_scaled`, `dedup` and `contract_forced` arguments of
`TernaryTree.encode` / `TernaryTree.topphatt` (and `core.topphatt`,
`core.encode_topphatt`). The defaults reproduce the original method.

## The problem

TOPP-HATT keeps a ternary tree's topology and qubit labels fixed. It chooses
which Majorana operator sits on each leaf so that the encoded Hamiltonian
has as little Pauli weight as possible.

The leaves of the input tree come in vacuum-preserving pairs, one pair per
mode. Call the two leaves that carry $\gamma_{2k}$ and $\gamma_{2k+1}$ *slot*
$k$. Placing mode $m$ in slot $s$ (even Majorana on the even leaf) always
gives a valid encoding that preserves the vacuum and has real coefficients.
So the search space is the set of permutations $\sigma \in S_n$ of modes over
slots. (`geometry::SlotModel` implements this space.)

The original algorithm works bottom-up:
1. Repeatedly take the deepest node whose children are all finished.
2. Enumerate the leaf triples its restrictions allow.
3. Keep the triple with the lowest weight *on that node's qubit*.
4. Contract the three indices into a representative, and continue.

## Equivalent formulations

Write each Hamiltonian term as a parity set $A \subseteq [2n]$; repeated
indices cancel because $\gamma^2 = I$. For any set $B$ of Majoranas, let
$\mathrm{col}_B \in \mathbb{F}_2^T$ have bit $t$ equal to $|A_t \cap B| \bmod 2$.
For an even term, this bit is 1 exactly when $\gamma_{A_t}$ anticommutes with
$\gamma_B$.

The encoding is an algebra isomorphism. So the set of distinct Pauli strings
in the encoded Hamiltonian is the set of distinct parity sets with a
non-vanishing coefficient, whatever the encoding
(`geometry::parity_canonical_terms`).

### 1. Algebraic: anticommutation counting

For a Pauli string $P$,
$\mathrm{wt}(P) = \tfrac12 \sum_q \sum_{s\in\{X,Y,Z\}} [P \text{ anticommutes with } s_q]$.

Pull the single-qubit Paulis back through a ternary-tree encoding. Each
$s_q$ becomes a Majorana monomial $\gamma_B$, where $B$ is the set of leaves in
two of node $q$'s three child subtrees. For example, $Z_q$ pulls back to the
leaves below its X and Y edges. Hence

$$ W(\sigma) = \tfrac12 \sum_q \sum_{\text{pairs } \{e,e'\}} \#\{t : \gamma_{A_t} \text{ anticommutes with } \gamma_{B_{q,e}\cup B_{q,e'}}\}. $$

The cost is a sum of anticommutation counts against $3n$ "subtree-parity"
Majorana monomials. No Pauli strings need to be built.

### 2. Geometric: the Hamming cube

Give every subtree $v$ the point
$\pi_v = \bigoplus_{\ell \text{ below } v} \mathrm{col}_{\sigma(\ell)} \in \mathbb{F}_2^T$.
A parent's point is the XOR of its children's points. The weight on node
$v$'s qubit is the **half-perimeter of the triangle** formed by its children's
points:

$$ \mathrm{wt}_v = \tfrac12\big(d(\pi_X,\pi_Y)+d(\pi_Y,\pi_Z)+d(\pi_X,\pi_Z)\big) = \big|(\pi_X\vee\pi_Y\vee\pi_Z)\wedge\neg(\pi_X\wedge\pi_Y\wedge\pi_Z)\big|. $$

$W$ is the sum of these over the nodes. This is the identity the
`dense_transpose` kernel already computes with `popcount(any & !all)`. It is
now tested to equal the true encoded weight
(`half_perimeter_sum_equals_true_pauli_weight`).

### 3. Edge / T-join: a tree metric

For a fixed term, mark each tree edge whose lower side holds an odd number
of the term's leaves. Those edges form the unique $T$-join $J_A$ of the term's
leaves. Nodes have degree 0, 2 or 4 in $J_A$. Degree 2 costs one unit of
weight. Degree 4 costs nothing, because $XYZ \propto I$. Counting degrees
gives

$$ \mathrm{wt}(A) = |E(J_A)| - \tfrac{|A|}{2} - 2\,n_4(A), \qquad W = \sum_{e} |\pi_e| - \sum_t \tfrac{|A_t|}{2} - 2N_4 . $$

(Tested in `edge_tjoin_identity`.)

In a tree, $|E(J_A)|$ equals the minimum-weight perfect matching of
$\sigma(A)$ in the tree metric. So, for two-Majorana terms only, the problem
is exactly a **quadratic assignment problem on a tree metric**. On a path
(Jordan–Wigner) it becomes a minimum-linear-arrangement problem. Quartic
terms give a hypergraph generalisation. This is why greedy assignment is
near-optimal on caterpillar trees but not on balanced ones.

### 4. Group-theoretic

The leaf Pauli set is fixed. TOPP-HATT therefore searches the Majorana
permutations that keep the mode pairing: the slot permutations above, a
subgroup of the signed permutations that are both Clifford and fermionic
Gaussian. Dropping vacuum preservation would enlarge this to all of $S_{2n}$
and still give a valid encoding. That option is not implemented.

## Diagnosis

The forms above make it possible to check the greedy's own objective
against the truth. `topphatt_with_options` returns the greedy's running
estimate alongside the tree.

- **Forced nodes are dropped, not contracted.** Some nodes have all their
  leaves fixed by pair propagation from deeper nodes. The original loop
  drops such nodes without contracting them. Their parent then reads an
  all-zero representative column, so it scores that subtree as if it
  carried no terms. The option `contract_forced=True` queues these nodes
  and contracts their single forced combination. It is a correctness fix
  for the objective, but in practice it changes results only on the HATT
  tree.
- **Post-contraction deduplication undercounts.** HATT's
  `sort + dedup` merges terms that coincide *after* a contraction, e.g.
  $\{0,5,9\}$ and $\{1,5,9\}$ both become $\{r,5,9\}$. These are distinct
  Pauli strings, and each acts on every ancestor qubit. With
  `dedup=False, contract_forced=True`, the greedy's estimate equals the true
  weight exactly (`dedup_undercounts_and_exact_objective_matches_true_weight`).
  **But optimising the exact objective greedily is worse**: see the
  `exact_obj` column below. Down-weighting groups of terms that are already
  indistinguishable acts as a useful regulariser for a myopic greedy. The
  original rule is kept as the default for that reason.
- **The real loss is the bottom-up order.** By form 3, the qubits near the
  root carry the most weight. Bottom-up greedy fixes them last, using
  whatever leaves are left over. The exact optimum (exhaustive search, at
  most 10 modes) shows the gap. On H2/6-31g, TOPP-HATT is optimal on JW but
  1% above the optimum on BK and JKMN, and 7% above on the HATT tree, where
  it is *worse than the tree's own labelling*.

## New strategies

| `strategy` | Idea |
|---|---|
| `"bottom_up"` (default) | Original TOPP-HATT. `lookahead=λ` adds $\lambda\,|\pi_v|$: the number of terms the chosen children leave "open". By form 3, each open term pays for at least one more qubit. |
| `"top_down"` | Visit nodes from the root. A node's weight depends only on *which child subtree* each Majorana lands in, so at each node the modes are swapped between the child classes of slots (Kernighan–Lin style). The score is the exact half-perimeter plus `lookahead` × $\sum_{c}|\pi_c|$ over the child nodes, a lower bound on the weight still to come. Each swap is an $O(T/64)$ XOR update. |
| `"top_down_expected"` | Same traversal. A split is scored by the exact node weight plus the **expected weight of a uniformly random completion** of each child subtree. That expectation depends on a term only through its counts $(s,d)$ of unpaired Majoranas and whole mode pairs below the child. It is precomputed per node by Monte Carlo as $h_c(s,d)$ and updated incrementally per term. No parameters. |
| `"top_down_seeded"` | `top_down`, seeded with the bottom-up result. |
| `"best"` | Run the input labelling, bottom-up and both top-down variants (seeded with both the default labelling and the input tree's labelling). Score each exactly on the Hamming cube and keep the lowest. Never worse than the tree passed in. |
| `"exact"` | Exhaustive search over slot permutations (at most 10 modes). This is the optimality reference. |

## Results

Pauli weight; lower is better. Produced by
`docs/publications/topphatt_variants.py`; the raw data, including CPW and
timings, is in `docs/publications/data/topphatt_variants.json`. Column key:

- `exact_obj`: `dedup=False, contract_forced=True`.
- `lookahead`: bottom-up with `lookahead=0.5`.
- `top_down`: `lookahead=1`.

| Molecule | Tree | naive | TOPP-HATT | exact_obj | lookahead | top_down | top_down_expected | best | exact |
|---|---|---|---|---|---|---|---|---|---|
| H2 6-31g (8) | JW | 848 | 720 | 720 | 720 | 720 | 720 | 720 | 720 |
| | Parity | 868 | 744 | 752 | 744 | 744 | 744 | 744 | 738 |
| | BK | 826 | 756 | 836 | 756 | 748 | 748 | 748 | 748 |
| | JKMN | 840 | 808 | 832 | 808 | 800 | 800 | 800 | 800 |
| | Huffman | 862 | 782 | 848 | 782 | 782 | 782 | 782 | 782 |
| | HATT | 790 | 836 | 790 | 836 | 790 | 790 | 782 | 782 |
| H2O sto-3g (14) | JW | 14756 | 10946 | 12126 | 10932 | 12096 | 10880 | 10880 | – |
| | Parity | 15047 | 11432 | 12556 | 11396 | 12332 | 11206 | 11206 | – |
| | BK | 12656 | 12470 | 12534 | 12462 | 11611 | 11629 | 11611 | – |
| | JKMN | 12718 | 11952 | 12008 | 11886 | 11766 | 11759 | 11759 | – |
| | Huffman | 12960 | 12303 | 12404 | 12004 | 12123 | 11930 | 11930 | – |
| | HATT | 11169 | 12454 | 12379 | 12385 | 12055 | 11585 | 11169 | – |
| H2O 6-31g (26) | JW | 285512 | 214156 | 214910 | 214152 | 212926 | 212640 | 212640 | – |
| | Parity | 287756 | 217334 | 217614 | 217100 | 214739 | 214771 | 214739 | – |
| | BK | 203787 | 199925 | 200707 | 200094 | 187594 | 189501 | 187594 | – |
| | JKMN | 193914 | 184004 | 188873 | 186040 | 175889 | 174234 | 174234 | – |
| | Huffman | 195575 | 187616 | 191212 | 186440 | 171757 | 186246 | 171757 | – |

Relative to TOPP-HATT, `best` reduces Pauli weight as follows:

| | JW | Parity | BK | JKMN | Huffman | HATT |
|---|---|---|---|---|---|---|
| H2O sto-3g | 0.6% | 2.0% | 6.9% | 1.6% | 3.0% | 10.3% |
| H2O 6-31g | 0.7% | 1.2% | 6.2% | 5.3% | 8.5% | – |

The size of the reduction tracks how balanced the tree is, as form 3
predicts. On the H2O sto-3g Hamiltonian used for the published benchmarks,
`top_down_expected` on JW reaches 10880 Pauli weight. That is below both
TOPP-HATT (10946) and the published simulated-annealing result (10906),
without any annealing.

Cost: `top_down` runs faster than bottom-up (about 15 ms vs 30–60 ms at 26
modes). `top_down_expected` and `best` take roughly 0.3–0.5 s at 26 modes,
dominated by building the Monte Carlo tables.

**Caveats.**
- All new strategies minimise unweighted Pauli weight. Coefficient-weighted
  Pauli weight sometimes moves the other way (for example BK, H2O sto-3g:
  CPW 195 for TOPP-HATT vs 230 for top-down).
- Top-down with no lookahead is poor on chain-like trees (JW/Parity), where
  root-first decisions are themselves myopic. `top_down_expected` fixes that.
- `exact` is limited to 10 modes.

## Relation to other methods

- **HATT** (Liu et al., 2024) uses the same bottom-up greedy, but builds the
  topology as it goes. TOPP-HATT is HATT restricted to a fixed topology, so
  the dedup and ordering observations above apply to both.
- **Fermihedral** (Liu et al., 2024) finds exact optimal encodings with a SAT
  solver, for small systems. Our `exact` strategy is the analogue within a
  fixed topology, and is far cheaper because the space is $S_n$ rather than
  all Clifford-equivalent encodings.
- **Mode-ordering methods**, such as annealing over enumerations (in this
  repository) and the enumeration approach of Chiew & Strelchuk (2023),
  search the same space, $S_n$ acting on slots. The formulations above give
  them an $O(T/64)$ exact cost per node, instead of a full re-encoding per
  move.
- **Bonsai, Huffman and RETT** generate topologies. They are complementary,
  and their trees are inputs to these strategies.
- **The Clifford heuristic** (arXiv:2502.11933) acts after encoding, on the
  qubit Hamiltonian, and so is outside this search space.

## Further directions

- A coefficient-weighted objective: give each term's bit weight $|c_t|$ in
  forms 2 and 3. The top-down swap updates carry over unchanged.
- Branch-and-bound beyond 10 modes, using $\sum_e |\pi_e|$ over fixed edges
  as a lower bound.
- A vacuum-free search over all of $S_{2n}$ (form 4), for applications that
  prepare the initial state anyway.
- Exact `top_down_expected` tables. The Monte Carlo estimate $h_c(s,d)$ has
  closed forms on regular trees.
