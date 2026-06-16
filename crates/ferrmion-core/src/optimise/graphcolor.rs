//! Methods to find the parallelizability of an encoding
//! using graph coloring.
//! Bringewat & Davoudi et al. 2023
//! https://quantum-journal.org/papers/q-2023-04-13-975/

use crate::devices::Qubit;
use crate::operators::*;
use ndarray::{Axis, Zip};
use rustworkx_core::petgraph::graphmap::UnGraphMap;
use std::collections::HashSet;

/// Interaction graph of an Interaction Hamiltonian.
struct InteractionGraph(pub UnGraphMap<Mode, ()>);

impl InteractionGraph {
    fn append(&mut self, product: &InteractionProduct) {
        for operator in product.ops.iter() {
            match operator {
                InteractionOperator::Identity => {}
                // We assume the graph is going to be connected,
                // but it's possible that it won't be if we include vertices.
                InteractionOperator::Vertex(l) => {
                    self.0.add_node(*l);
                }
                InteractionOperator::Edge(l, r) => {
                    self.0.add_node(*l);
                    self.0.add_node(*r);
                    self.0.add_edge(*l, *r, ());
                }
            }
        }
    }

    fn append_sparse(&mut self, sparse: InteractionSparse) {
        let mut product = InteractionProduct::identity();
        for ind in sparse.indices.rows() {
            product.ops = ind
                .exact_chunks(2)
                .into_iter()
                .zip(sparse.ops.iter())
                .map(|(chunk, op)| match op {
                    InteractionBasis::Edge => InteractionOperator::Edge(chunk[0], chunk[1]),
                    InteractionBasis::Vertex => InteractionOperator::Vertex(chunk[0]),
                })
                .collect();
            self.append(&product);
            product = InteractionProduct::identity();
        }
    }
}

struct SystemGraph {
    graph: UnGraphMap<Mode, Qubit>,
    physical_modes: HashSet<Mode>,
    virtual_modes: HashSet<Mode>,
}

impl From<InteractionGraph> for SystemGraph {
    fn from(graph: InteractionGraph) -> Self {
        SystemGraph {
            graph: UnGraphMap::from_edges(
                graph
                    .0
                    .all_edges()
                    .enumerate()
                    .map(|(i, (l, r, _))| (l, r, Qubit::new(i as u16)))
                    .collect::<Vec<_>>(),
            ),
            physical_modes: HashSet::new(),
            virtual_modes: HashSet::new(),
        }
    }
}
