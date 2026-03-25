//! Fast, reliable and easy optimisation of fermion-qubit encodings.
//!
//! Core Rust library for ferrmion. Contains all the computational logic
//! for encoding fermionic operators to qubit operators, ternary tree
//! optimisation, and related algorithms.

pub mod encoding;
pub mod hamiltonians;
pub mod operators;
pub mod optimise;
pub mod states;
pub mod ternarytree;
pub mod utils;
