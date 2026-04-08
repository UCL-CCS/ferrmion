//! Fermion-qubit encoding  methods.
//!
//! This crate provides the pure-Rust implementation of fermion-to-qubit
//! encoding algorithms. It has no Python or PyO3 dependencies and can be
//! used as a standalone Rust library.

pub mod encode;
pub mod hamiltonians;
pub mod operators;
pub mod optimise;
pub mod states;
pub mod utils;
