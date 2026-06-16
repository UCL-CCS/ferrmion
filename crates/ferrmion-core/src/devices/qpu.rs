//! Quantum processing unit (QPU) device.

use super::Qubit;
use rustworkx_core::graph::UnGraph;

/// Represents a quantum processing unit (QPU) device.
///
/// The only information currently stored is the graph of qubits.
/// Noise, gate-set and other device-specific parameters are not stored.
pub struct ProcessorTopology(UnGraph<Qubit, ()>);
