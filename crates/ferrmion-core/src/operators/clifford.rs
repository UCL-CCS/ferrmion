//! Clifford Operators
//!
//! Any gate which transforms a Pauli P to another Pauli Q
//! $C^{\dagger}PC = Q$

/// Clifford operator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Clifford {
    H(usize),
    S(usize),
    CNOT { control: usize, target: usize },
}
