//! Qubit device.

/// Newtype storing a qubit index.
///
/// Note that this object does not store the qubit state.
/// States are handled separately in the `states` module.
/// This type is analogous to `Majorana` or `Mode`, and is used for type-safe indexing.
pub struct Qubit(u16);

impl Qubit {
    pub fn new(index: u16) -> Self {
        Self(index)
    }
}
