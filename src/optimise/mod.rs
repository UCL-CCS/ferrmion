//! Optimisation algorithms for fermion-qubit encodings.
//!
//! Provides the TOPP-HATT algorithm and simulated annealing for
//! minimising the Pauli weight of encoded qubit Hamiltonians.

mod anneal;
pub use anneal::*;
mod topphatt;
pub use topphatt::*;
