/*
Operator types.
*/

// Individual operators
enum Operator {
    MajoranaOp,
    FermionOp,
    PauliOp,
}

struct MajoranaOp {
    val: Vec<usize>,
}

struct FermionOp {
    val: Vec<usize>,
    n_modes: usize,
}

struct PauliOp {
    val: &str,
    n_qubits: usize,
}

// Collections of operators
enum HamiltonianTerm {
    MajoranaTerm,
    FermionTerm,
    PauliTerm,
}
