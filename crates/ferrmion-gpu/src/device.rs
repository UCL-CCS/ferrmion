//! The cuda-oxide encoding kernel and [`GpuEncoder`], its host-side driver.

use crate::packed::{PackedProblem, lane_weights};
use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;
use ferrmion_core::encode::majorana::MajoranaEncoding;
use ferrmion_core::operators::MajoranaSparse;
use std::error::Error;
use std::sync::Arc;

pub type GpuResult<T> = Result<T, Box<dyn Error>>;

#[cuda_module]
mod kernels {
    use super::*;

    /// One thread per `(permutation, lane)`; see [`PackedProblem`].
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn pauli_weights(
        x_rows: &[u64],
        z_rows: &[u64],
        term_indices: &[u32],
        term_lens: &[u32],
        norms: &[f64],
        perms: &[u32],
        n_terms: u32,
        n_modes: u32,
        words: u32,
        lanes: u32,
        mut plain_out: DisjointSlice<u32>,
        mut weighted_out: DisjointSlice<f64>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= plain_out.len() {
            return;
        }
        let (plain, weighted) = lane_weights(
            i,
            x_rows,
            z_rows,
            term_indices,
            term_lens,
            norms,
            perms,
            n_terms as usize,
            n_modes as usize,
            words as usize,
            lanes as usize,
        );
        if let Some(out) = plain_out.get_mut(idx) {
            *out = plain;
        }
        // `get_mut` consumes its witness, so the second output takes a fresh one.
        if let Some(out) = weighted_out.get_mut(thread::index_1d()) {
            *out = weighted;
        }
    }
}

/// Evaluates Pauli weights of one Hamiltonian under many mode permutations on
/// an NVIDIA GPU.
///
/// The encoding and Hamiltonian are uploaded once in [`GpuEncoder::new`]; each
/// [`GpuEncoder::batch_pauli_weights`] call uploads only the permutations.
pub struct GpuEncoder {
    problem: PackedProblem,
    stream: Arc<CudaStream>,
    module: kernels::LoadedModule,
    x_rows: DeviceBuffer<u64>,
    z_rows: DeviceBuffer<u64>,
    term_indices: DeviceBuffer<u32>,
    term_lens: DeviceBuffer<u32>,
    norms: DeviceBuffer<f64>,
}

impl GpuEncoder {
    /// Upload `encoding` and `msparse` to device `ordinal`.
    pub fn new(
        encoding: &MajoranaEncoding,
        msparse: &MajoranaSparse,
        ordinal: usize,
    ) -> GpuResult<Self> {
        let problem = PackedProblem::new(encoding, msparse);
        let ctx = CudaContext::new(ordinal)?;
        let stream = ctx.default_stream();
        let module = kernels::load(&ctx)?;
        Ok(Self {
            x_rows: DeviceBuffer::from_host(&stream, &problem.x_rows)?,
            z_rows: DeviceBuffer::from_host(&stream, &problem.z_rows)?,
            term_indices: DeviceBuffer::from_host(&stream, &problem.term_indices)?,
            term_lens: DeviceBuffer::from_host(&stream, &problem.term_lens)?,
            norms: DeviceBuffer::from_host(&stream, &problem.norms)?,
            problem,
            stream,
            module,
        })
    }

    /// Plain and coefficient-weighted Pauli weights for each permutation.
    ///
    /// Same semantics as [`MajoranaEncoding::batch_pauli_weights`]. Each
    /// kernel thread sums the weights of a strided share of the terms under
    /// one permutation, and the host adds the per-lane partials in order.
    pub fn batch_pauli_weights(
        &self,
        permutations: &[Vec<usize>],
    ) -> GpuResult<(Vec<f64>, Vec<f64>)> {
        let problem = &self.problem;
        if permutations.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        let n_threads = problem.n_threads(permutations.len());
        if n_threads > u32::MAX as usize {
            return Err("too many permutations for one launch; split the batch".into());
        }

        let perms =
            DeviceBuffer::from_host(&self.stream, &problem.pack_permutations(permutations))?;
        let mut plain_out = DeviceBuffer::<u32>::zeroed(&self.stream, n_threads)?;
        let mut weighted_out = DeviceBuffer::<f64>::zeroed(&self.stream, n_threads)?;

        // SAFETY: a 1-D launch, matching `thread::index_1d()`. Threads past
        // `n_threads` in the last block return before reading anything; every
        // other thread's reads are in bounds by construction of
        // `PackedProblem`, and it writes only its own output elements.
        unsafe {
            self.module.pauli_weights(
                &self.stream,
                LaunchConfig::for_num_elems(n_threads as u32),
                &self.x_rows,
                &self.z_rows,
                &self.term_indices,
                &self.term_lens,
                &self.norms,
                &perms,
                problem.n_terms as u32,
                problem.n_modes as u32,
                problem.words as u32,
                problem.lanes as u32,
                &mut plain_out,
                &mut weighted_out,
            )
        }?;

        let plain = plain_out.to_host_vec(&self.stream)?;
        let weighted = weighted_out.to_host_vec(&self.stream)?;
        Ok(problem.reduce(permutations.len(), &plain, &weighted))
    }
}
