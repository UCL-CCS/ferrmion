/*
Ternary tree encodings and methods.
*/
use std::num::NonZero;
const MAX_NODES: u8 = 85;

struct FastTernaryTree {
    parents: [u8; 2],
}
