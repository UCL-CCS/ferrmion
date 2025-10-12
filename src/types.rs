/*
Types needed in multiple files.
*/

use ahash::RandomState;
use numpy::Complex64;
use pyo3::{FromPyObject, IntoPyObject};
use std::collections::HashMap;

pub type QubitHamiltonianTemplate =
    HashMap<String, HashMap<IntegralIndex, Complex64, RandomState>, RandomState>;

pub type QubitHamiltonian<'template> = HashMap<&'template String, Complex64, RandomState>;

pub enum Notation {
    Physicist,
    Chemist,
}

#[derive(Eq, PartialEq, Hash, IntoPyObject, FromPyObject, Debug)]
pub enum IntegralIndex {
    //TwoE terms are more common, and pyo3 tries from top to bottom
    //So putting them first in the Enum
    TwoE(usize, usize, usize, usize),
    OneE(usize, usize),
}
