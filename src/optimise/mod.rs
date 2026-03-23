//! Optimisation algorithms for fermion-qubit encodings.
//!
//! Provides simulated annealing and the TOPP-HATT algorithm for
//! minimising the Pauli weight of encoded qubit Hamiltonians.

mod anneal;
pub use anneal::*;
mod topphatt;
pub use topphatt::*;
