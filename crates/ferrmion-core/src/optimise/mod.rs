//! Optimisation algorithms for fermion-qubit encodings.
//!
//! Provides the TOPP-HATT algorithm and simulated annealing for
//! minimising the Pauli weight of encoded qubit Hamiltonians.

mod encoding;
pub use encoding::*;
mod hamiltonian;
pub use hamiltonian::*;
mod ternarytree;
pub use ternarytree::*;
