//! Runtime-selectable expand-and-merge strategies for the parallel path of
//! `MajoranaHashMap::append_fermion_sparse`.
//!
//! Profiling showed that constructing a large `MajoranaSparse` spends most of
//! its threaded time merging the per-chunk hash maps produced by the parallel
//! term expansion. This module factors that expand/merge step out behind a set
//! of alternative algorithms so they can be compared empirically. The strategy
//! is chosen **at runtime** from environment variables, keeping a single
//! compiled artefact (Rust bench binary or Python wheel) able to run every
//! variant — which is what `scripts/bench_merge_sweep.sh` relies on.
//!
//! # Environment variables
//!
//! | Variable | Meaning | Default |
//! |---|---|---|
//! | `FERRMION_MERGE_STRATEGY` | one of the strategy names below | `baseline` |
//! | `FERRMION_MERGE_SHARDS` | shard/bin count for sharded strategies | rayon thread count |
//! | `FERRMION_PARALLEL_CHUNK` | terms expanded per phase-1 task | `64` |
//! | `FERRMION_MERGE_SERIAL_THRESHOLD` | total intermediate entries at or below which the merge runs serially | `0` (off) |
//! | `FERRMION_MERGE_PRESIZE` | `1`/`true` to pre-size hash maps from entry-count estimates | off |
//! | `FERRMION_MERGE_TIMING` | `1`/`true` to print per-call phase timings to stderr | off |
//!
//! The configuration is read once on first use and cached for the lifetime of
//! the process (mirroring how `RAYON_NUM_THREADS` pins the global thread
//! pool), so it must be set before the first `MajoranaSparse` construction.
//! Invalid values panic with a message listing the accepted ones: silently
//! falling back to a default would corrupt a benchmark comparison.
//!
//! # Strategies
//!
//! * `baseline` — the original algorithm: expand chunks into thread-local
//!   maps, then merge with one task per shard where every shard re-scans every
//!   entry of every partial map and re-hashes it (FNV-1a) to test ownership.
//!   Deterministic (each key is summed in chunk order).
//! * `hash_cache` — as `baseline`, but each partial is first flattened to a
//!   vector with the shard id computed **once** per entry, so the per-shard
//!   scans are cheap sequential passes with no re-hashing. Deterministic.
//! * `fx_hash` — as `baseline`, but the phase-1 partial maps use the
//!   `rustc-hash` (Fx) hasher instead of aHash, isolating the cost of the hash
//!   function itself. Deterministic.
//! * `shard_phase1` — phase-1 workers route each expanded term directly into
//!   one of `n_shards` local buckets, so phase 2 only concatenates same-shard
//!   buckets: every entry is visited once and never re-hashed for ownership.
//!   Deterministic.
//! * `tree_reduce` — rayon `reduce`: per-task maps are merged pairwise up a
//!   scheduling-dependent tree. Expansion and merging are fused, so the timing
//!   output reports a single duration. **Not deterministic** in floating-point
//!   summation order (results agree to rounding).
//! * `sort_scan` — terms are packed into `u128` keys, locally deduplicated,
//!   concatenated, sorted with rayon's **stable** parallel sort and summed in
//!   a single linear scan. Deterministic (stable sort preserves chunk order).
//! * `radix_partition` — locally deduplicated `u128` entries are partitioned
//!   into disjoint bins by a multiplicative hash; each bin is reduced into its
//!   own map in parallel. Deterministic (bins scan chunks in order).
//! * `kway_merge` — per-chunk entries are locally deduplicated, sorted, and
//!   combined with a serial binary-heap k-way merge that sums equal keys as
//!   they stream past. Deterministic (ties broken by chunk index).
use super::fermion::{expand_term, MajoranaHashMap, MajoranaKey, MAX_MAJORANAS};
use super::ladder::LadderOperator;
use ndarray::{Array1, Array2};
use num_complex::Complex64;
use rayon::prelude::*;
use rustc_hash::FxBuildHasher;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::collections::HashMap as StdHashMap;
use std::hash::BuildHasher;
use std::sync::OnceLock;
use std::time::{Duration, Instant};
use tinyvec::ArrayVec;

/// Hash map from Majorana keys to coefficients with a pluggable hasher.
type GMap<S> = StdHashMap<MajoranaKey, Complex64, S>;
/// The default (aHash) keyed map, matching `MajoranaHashMap`'s storage.
type AMap = GMap<ahash::RandomState>;
/// Map keyed by packed `u128` Majorana keys, used by the sort-based strategies.
type PackedMap = StdHashMap<u128, Complex64, ahash::RandomState>;

/// Default number of terms a single phase-1 task expands, so tiny per-term
/// work is batched rather than scheduled individually.
const DEFAULT_PARALLEL_CHUNK: usize = 64;

/// The merge algorithm to use for the parallel path. See the module docs for
/// descriptions and determinism notes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MergeStrategy {
    Baseline,
    HashCache,
    FxHash,
    ShardPhase1,
    TreeReduce,
    SortScan,
    RadixPartition,
    KwayMerge,
}

impl MergeStrategy {
    /// Every strategy, in the order they are documented. Used by the
    /// correctness sweep tests and kept in sync with [`Self::parse`].
    pub(crate) const ALL: [MergeStrategy; 8] = [
        MergeStrategy::Baseline,
        MergeStrategy::HashCache,
        MergeStrategy::FxHash,
        MergeStrategy::ShardPhase1,
        MergeStrategy::TreeReduce,
        MergeStrategy::SortScan,
        MergeStrategy::RadixPartition,
        MergeStrategy::KwayMerge,
    ];

    /// The environment-variable spelling of this strategy.
    pub(crate) fn name(&self) -> &'static str {
        match self {
            MergeStrategy::Baseline => "baseline",
            MergeStrategy::HashCache => "hash_cache",
            MergeStrategy::FxHash => "fx_hash",
            MergeStrategy::ShardPhase1 => "shard_phase1",
            MergeStrategy::TreeReduce => "tree_reduce",
            MergeStrategy::SortScan => "sort_scan",
            MergeStrategy::RadixPartition => "radix_partition",
            MergeStrategy::KwayMerge => "kway_merge",
        }
    }

    /// Parse a strategy name (case-insensitive), returning `None` for unknown names.
    fn parse(value: &str) -> Option<MergeStrategy> {
        let lowered = value.trim().to_ascii_lowercase();
        MergeStrategy::ALL.into_iter().find(|s| s.name() == lowered)
    }
}

/// Runtime configuration for [`expand_and_merge`], normally read once from the
/// environment via [`MergeConfig::get`]. Tests and benchmarks may construct
/// values directly to sweep strategies within one process.
#[derive(Debug, Clone)]
pub(crate) struct MergeConfig {
    pub(crate) strategy: MergeStrategy,
    /// Shard/bin count for the sharded strategies; `None` follows the rayon
    /// thread count (the pre-existing behaviour).
    pub(crate) n_shards: Option<usize>,
    /// Number of terms each phase-1 task expands.
    pub(crate) parallel_chunk: usize,
    /// If the intermediate entry count is at or below this, merge serially
    /// instead of sharding (0 disables the shortcut).
    pub(crate) serial_merge_threshold: usize,
    /// Pre-size hash maps from entry-count estimates.
    pub(crate) presize: bool,
    /// Print phase timings to stderr after every parallel merge.
    pub(crate) timing: bool,
}

impl Default for MergeConfig {
    fn default() -> Self {
        Self {
            strategy: MergeStrategy::Baseline,
            n_shards: None,
            parallel_chunk: DEFAULT_PARALLEL_CHUNK,
            serial_merge_threshold: 0,
            presize: false,
            timing: false,
        }
    }
}

static CONFIG: OnceLock<MergeConfig> = OnceLock::new();

impl MergeConfig {
    /// The process-wide configuration, read from the environment on first use.
    pub(crate) fn get() -> &'static MergeConfig {
        CONFIG.get_or_init(MergeConfig::from_env)
    }

    fn from_env() -> MergeConfig {
        let defaults = MergeConfig::default();
        let strategy = match std::env::var("FERRMION_MERGE_STRATEGY") {
            Ok(value) => MergeStrategy::parse(&value).unwrap_or_else(|| {
                let names: Vec<&str> = MergeStrategy::ALL.iter().map(|s| s.name()).collect();
                panic!(
                    "FERRMION_MERGE_STRATEGY={value:?} is not a merge strategy; expected one of {}",
                    names.join(", ")
                )
            }),
            Err(_) => defaults.strategy,
        };
        let parallel_chunk = env_usize("FERRMION_PARALLEL_CHUNK")
            .inspect(|&v| {
                assert!(v > 0, "FERRMION_PARALLEL_CHUNK must be at least 1");
            })
            .unwrap_or(defaults.parallel_chunk);
        MergeConfig {
            strategy,
            n_shards: env_usize("FERRMION_MERGE_SHARDS").inspect(|&v| {
                assert!(v > 0, "FERRMION_MERGE_SHARDS must be at least 1");
            }),
            parallel_chunk,
            serial_merge_threshold: env_usize("FERRMION_MERGE_SERIAL_THRESHOLD")
                .unwrap_or(defaults.serial_merge_threshold),
            presize: env_bool("FERRMION_MERGE_PRESIZE"),
            timing: env_bool("FERRMION_MERGE_TIMING"),
        }
    }

    /// Shard/bin count to use: the override if set, else the rayon pool size.
    fn effective_shards(&self) -> usize {
        self.n_shards
            .unwrap_or_else(rayon::current_num_threads)
            .max(1)
    }
}

/// Read a `usize` environment variable, panicking on unparsable values so a
/// mistyped benchmark configuration fails loudly instead of silently running
/// the default.
fn env_usize(name: &str) -> Option<usize> {
    let value = std::env::var(name).ok()?;
    Some(
        value
            .trim()
            .parse()
            .unwrap_or_else(|_| panic!("{name}={value:?} is not a valid non-negative integer")),
    )
}

/// Read a boolean environment variable (`1`/`true`/`yes` vs `0`/`false`/`no`),
/// panicking on anything else.
fn env_bool(name: &str) -> bool {
    match std::env::var(name) {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" => true,
            "" | "0" | "false" | "no" => false,
            _ => panic!("{name}={value:?} is not a valid boolean (use 1/true/yes or 0/false/no)"),
        },
        Err(_) => false,
    }
}

/// Wall-clock durations of the two phases of a parallel merge. Strategies that
/// fuse expansion and merging (`tree_reduce`) report everything under `merge`.
struct PhaseTimes {
    expand: Duration,
    merge: Duration,
}

/// Expand the rows of a `FermionSparse` (its shared `action`, per-row mode
/// `indices` and `coefficients`) into Majorana terms and merge them into
/// `dest`, using the algorithm selected by `cfg`.
///
/// Every strategy produces the same multiset of (key, coefficient-sum) pairs;
/// see the module docs for which strategies fix the floating-point summation
/// order and are therefore bit-for-bit deterministic.
pub(super) fn expand_and_merge(
    dest: &mut MajoranaHashMap,
    action: &[LadderOperator],
    indices: &Array2<usize>,
    coefficients: &Array1<Complex64>,
    cfg: &MergeConfig,
) {
    let n_terms = indices.nrows();
    let times = match cfg.strategy {
        MergeStrategy::Baseline => {
            rescan_strategy::<ahash::RandomState>(dest, action, indices, coefficients, cfg)
        }
        MergeStrategy::FxHash => {
            rescan_strategy::<FxBuildHasher>(dest, action, indices, coefficients, cfg)
        }
        MergeStrategy::HashCache => hash_cache(dest, action, indices, coefficients, cfg),
        MergeStrategy::ShardPhase1 => shard_phase1(dest, action, indices, coefficients, cfg),
        MergeStrategy::TreeReduce => tree_reduce(dest, action, indices, coefficients, cfg),
        MergeStrategy::SortScan => sort_scan(dest, action, indices, coefficients, cfg),
        MergeStrategy::RadixPartition => radix_partition(dest, action, indices, coefficients, cfg),
        MergeStrategy::KwayMerge => kway_merge(dest, action, indices, coefficients, cfg),
    };
    if cfg.timing {
        eprintln!(
            "[ferrmion-merge-timing] strategy={} n_terms={} threads={} shards={} expand_s={:.6} merge_s={:.6}",
            cfg.strategy.name(),
            n_terms,
            rayon::current_num_threads(),
            cfg.effective_shards(),
            times.expand.as_secs_f64(),
            times.merge.as_secs_f64(),
        );
    }
}

/// Stable (run-independent) FNV-1a hash of a Majorana key, used to assign it
/// to a merge shard. Uses `u64` (not `usize`) so the 64-bit FNV constants are
/// valid on 32-bit targets too.
#[inline]
fn fnv1a(key: &MajoranaKey) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &index in key.iter() {
        hash = (hash ^ index as u64).wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

/// Pack a canonicalized Majorana key into a single integer: seven 16-bit
/// indices followed by an 8-bit length (7 * 16 + 8 = 120 <= 128 bits). The
/// explicit length disambiguates shorter keys from padding, so packing is
/// injective; the numeric order of packed keys is not meaningful and does not
/// need to be (the final `MajoranaSparse` conversion re-sorts terms).
#[inline]
fn pack_key(key: &MajoranaKey) -> u128 {
    let mut packed = 0u128;
    for &index in key.iter() {
        packed = (packed << 16) | index as u128;
    }
    (packed << 8) | key.len() as u128
}

/// Inverse of [`pack_key`].
#[inline]
fn unpack_key(mut packed: u128) -> MajoranaKey {
    let len = (packed & 0xff) as usize;
    packed >>= 8;
    debug_assert!(len <= MAX_MAJORANAS);
    let mut buffer = [0u16; MAX_MAJORANAS];
    for slot in buffer[..len].iter_mut().rev() {
        *slot = (packed & 0xffff) as u16;
        packed >>= 16;
    }
    let mut key = MajoranaKey::new();
    key.extend_from_slice(&buffer[..len]);
    key
}

/// Phase 1 shared by the rescan-style strategies: expand chunks of rows into
/// thread-local maps, deduplicating within each chunk.
fn expand_partials<S: BuildHasher + Default + Send>(
    action: &[LadderOperator],
    indices: &Array2<usize>,
    coefficients: &Array1<Complex64>,
    cfg: &MergeConfig,
) -> Vec<GMap<S>> {
    let per_row = 1usize << action.len();
    (0..indices.nrows())
        .into_par_iter()
        .chunks(cfg.parallel_chunk)
        .map(|rows| {
            let capacity = if cfg.presize {
                rows.len() * per_row
            } else {
                cfg.parallel_chunk
            };
            let mut local = GMap::with_capacity_and_hasher(capacity, S::default());
            for r in rows {
                let row: ArrayVec<[usize; MAX_MAJORANAS]> =
                    indices.row(r).iter().copied().collect();
                expand_term(action, &row, coefficients[r], |key, value| {
                    *local.entry(key).or_insert(Complex64::ZERO) += value;
                });
            }
            local
        })
        .collect()
}

/// Serially fold a sequence of merged maps into the destination, optionally
/// reserving the exact entry count first.
fn fold_maps_into_dest<S: BuildHasher>(
    dest: &mut MajoranaHashMap,
    maps: Vec<GMap<S>>,
    presize: bool,
) {
    if presize {
        dest.operators
            .reserve(maps.iter().map(StdHashMap::len).sum());
    }
    for map in maps {
        for (key, value) in map {
            *dest.operators.entry(key).or_insert(Complex64::ZERO) += value;
        }
    }
}

/// `baseline` / `fx_hash` (ideas as-shipped, #9, plus knobs #11-#14): the
/// original sharded merge where each shard re-scans every partial and
/// re-hashes each key to test ownership. Generic over the phase-1 hasher.
fn rescan_strategy<S: BuildHasher + Default + Send + Sync>(
    dest: &mut MajoranaHashMap,
    action: &[LadderOperator],
    indices: &Array2<usize>,
    coefficients: &Array1<Complex64>,
    cfg: &MergeConfig,
) -> PhaseTimes {
    let start = Instant::now();
    let partials: Vec<GMap<S>> = expand_partials(action, indices, coefficients, cfg);
    let expanded = Instant::now();

    let total_entries: usize = partials.iter().map(StdHashMap::len).sum();
    let n_shards = cfg.effective_shards();
    if n_shards == 1 || total_entries <= cfg.serial_merge_threshold {
        if cfg.presize {
            dest.operators.reserve(total_entries);
        }
        for local in &partials {
            for (key, &value) in local.iter() {
                *dest.operators.entry(*key).or_insert(Complex64::ZERO) += value;
            }
        }
        return PhaseTimes {
            expand: expanded - start,
            merge: expanded.elapsed(),
        };
    }

    let shards: Vec<AMap> = (0..n_shards)
        .into_par_iter()
        .map(|shard| {
            let capacity = if cfg.presize {
                total_entries / n_shards + 1
            } else {
                0
            };
            let mut out = AMap::with_capacity_and_hasher(capacity, Default::default());
            for local in &partials {
                for (key, &value) in local.iter() {
                    if (fnv1a(key) % n_shards as u64) as usize == shard {
                        *out.entry(*key).or_insert(Complex64::ZERO) += value;
                    }
                }
            }
            out
        })
        .collect();
    fold_maps_into_dest(dest, shards, cfg.presize);
    PhaseTimes {
        expand: expanded - start,
        merge: expanded.elapsed(),
    }
}

/// `hash_cache` (idea #3): as `baseline`, but each partial is flattened once
/// into a vector carrying its precomputed shard id, so the per-shard scans are
/// sequential passes over contiguous memory with no re-hashing.
fn hash_cache(
    dest: &mut MajoranaHashMap,
    action: &[LadderOperator],
    indices: &Array2<usize>,
    coefficients: &Array1<Complex64>,
    cfg: &MergeConfig,
) -> PhaseTimes {
    let start = Instant::now();
    let partials: Vec<AMap> = expand_partials(action, indices, coefficients, cfg);
    let expanded = Instant::now();

    let total_entries: usize = partials.iter().map(StdHashMap::len).sum();
    let n_shards = cfg.effective_shards();
    if n_shards == 1 || total_entries <= cfg.serial_merge_threshold {
        if cfg.presize {
            dest.operators.reserve(total_entries);
        }
        for local in partials {
            for (key, value) in local {
                *dest.operators.entry(key).or_insert(Complex64::ZERO) += value;
            }
        }
        return PhaseTimes {
            expand: expanded - start,
            merge: expanded.elapsed(),
        };
    }

    // Compute each entry's shard assignment exactly once, in parallel.
    let tagged: Vec<Vec<(MajoranaKey, Complex64, u32)>> = partials
        .into_par_iter()
        .map(|local| {
            local
                .into_iter()
                .map(|(key, value)| {
                    let shard = (fnv1a(&key) % n_shards as u64) as u32;
                    (key, value, shard)
                })
                .collect()
        })
        .collect();
    let shards: Vec<AMap> = (0..n_shards)
        .into_par_iter()
        .map(|shard| {
            let capacity = if cfg.presize {
                total_entries / n_shards + 1
            } else {
                0
            };
            let mut out = AMap::with_capacity_and_hasher(capacity, Default::default());
            for part in &tagged {
                for &(key, value, tag) in part.iter() {
                    if tag as usize == shard {
                        *out.entry(key).or_insert(Complex64::ZERO) += value;
                    }
                }
            }
            out
        })
        .collect();
    fold_maps_into_dest(dest, shards, cfg.presize);
    PhaseTimes {
        expand: expanded - start,
        merge: expanded.elapsed(),
    }
}

/// `shard_phase1` (idea #1): phase-1 workers route each expanded term into one
/// of `n_shards` local buckets as they go, so phase 2 concatenates same-shard
/// buckets directly — every intermediate entry is visited exactly once and
/// never re-hashed for shard ownership.
fn shard_phase1(
    dest: &mut MajoranaHashMap,
    action: &[LadderOperator],
    indices: &Array2<usize>,
    coefficients: &Array1<Complex64>,
    cfg: &MergeConfig,
) -> PhaseTimes {
    let n_shards = cfg.effective_shards();
    let per_row = 1usize << action.len();
    let start = Instant::now();
    let partials: Vec<Vec<AMap>> = (0..indices.nrows())
        .into_par_iter()
        .chunks(cfg.parallel_chunk)
        .map(|rows| {
            let capacity = if cfg.presize {
                rows.len() * per_row / n_shards + 1
            } else {
                0
            };
            let mut buckets: Vec<AMap> = (0..n_shards)
                .map(|_| AMap::with_capacity_and_hasher(capacity, Default::default()))
                .collect();
            for r in rows {
                let row: ArrayVec<[usize; MAX_MAJORANAS]> =
                    indices.row(r).iter().copied().collect();
                expand_term(action, &row, coefficients[r], |key, value| {
                    let shard = (fnv1a(&key) % n_shards as u64) as usize;
                    *buckets[shard].entry(key).or_insert(Complex64::ZERO) += value;
                });
            }
            buckets
        })
        .collect();
    let expanded = Instant::now();

    let shards: Vec<AMap> = (0..n_shards)
        .into_par_iter()
        .map(|shard| {
            let capacity: usize = partials.iter().map(|buckets| buckets[shard].len()).sum();
            let mut out = AMap::with_capacity_and_hasher(capacity, Default::default());
            for buckets in &partials {
                for (key, &value) in buckets[shard].iter() {
                    *out.entry(*key).or_insert(Complex64::ZERO) += value;
                }
            }
            out
        })
        .collect();
    fold_maps_into_dest(dest, shards, cfg.presize);
    PhaseTimes {
        expand: expanded - start,
        merge: expanded.elapsed(),
    }
}

/// `tree_reduce` (idea #2): rayon `fold` accumulates per-task maps and
/// `reduce` merges them pairwise up a tree, always draining the smaller map
/// into the larger. Expansion and merging interleave, so the phases cannot be
/// timed separately. Floating-point summation order depends on the scheduling
/// tree, so results are only reproducible to rounding.
fn tree_reduce(
    dest: &mut MajoranaHashMap,
    action: &[LadderOperator],
    indices: &Array2<usize>,
    coefficients: &Array1<Complex64>,
    cfg: &MergeConfig,
) -> PhaseTimes {
    let per_row = 1usize << action.len();
    let start = Instant::now();
    let merged: AMap = (0..indices.nrows())
        .into_par_iter()
        .chunks(cfg.parallel_chunk)
        .map(|rows| {
            let capacity = if cfg.presize {
                rows.len() * per_row
            } else {
                cfg.parallel_chunk
            };
            let mut local = AMap::with_capacity_and_hasher(capacity, Default::default());
            for r in rows {
                let row: ArrayVec<[usize; MAX_MAJORANAS]> =
                    indices.row(r).iter().copied().collect();
                expand_term(action, &row, coefficients[r], |key, value| {
                    *local.entry(key).or_insert(Complex64::ZERO) += value;
                });
            }
            local
        })
        .reduce(AMap::default, |mut a, mut b| {
            if b.len() > a.len() {
                std::mem::swap(&mut a, &mut b);
            }
            for (key, value) in b {
                *a.entry(key).or_insert(Complex64::ZERO) += value;
            }
            a
        });
    if dest.operators.is_empty() {
        dest.operators = merged;
    } else {
        for (key, value) in merged {
            *dest.operators.entry(key).or_insert(Complex64::ZERO) += value;
        }
    }
    PhaseTimes {
        expand: Duration::ZERO,
        merge: start.elapsed(),
    }
}

/// Phase 1 shared by the sort-based strategies: expand chunks into locally
/// deduplicated vectors of `(packed_key, coefficient)` pairs.
fn expand_packed_partials(
    action: &[LadderOperator],
    indices: &Array2<usize>,
    coefficients: &Array1<Complex64>,
    cfg: &MergeConfig,
) -> Vec<Vec<(u128, Complex64)>> {
    let per_row = 1usize << action.len();
    (0..indices.nrows())
        .into_par_iter()
        .chunks(cfg.parallel_chunk)
        .map(|rows| {
            let capacity = if cfg.presize {
                rows.len() * per_row
            } else {
                cfg.parallel_chunk
            };
            let mut local = PackedMap::with_capacity_and_hasher(capacity, Default::default());
            for r in rows {
                let row: ArrayVec<[usize; MAX_MAJORANAS]> =
                    indices.row(r).iter().copied().collect();
                expand_term(action, &row, coefficients[r], |key, value| {
                    *local.entry(pack_key(&key)).or_insert(Complex64::ZERO) += value;
                });
            }
            local.into_iter().collect()
        })
        .collect()
}

/// `sort_scan` (ideas #6 + #8): concatenate the packed partials, sort them by
/// key with rayon's stable parallel sort, and sum runs of equal keys in one
/// linear scan. The stable sort preserves chunk order among equal keys, so
/// summation order — and hence the result — is deterministic.
fn sort_scan(
    dest: &mut MajoranaHashMap,
    action: &[LadderOperator],
    indices: &Array2<usize>,
    coefficients: &Array1<Complex64>,
    cfg: &MergeConfig,
) -> PhaseTimes {
    let start = Instant::now();
    let partials = expand_packed_partials(action, indices, coefficients, cfg);
    let expanded = Instant::now();

    let total: usize = partials.iter().map(Vec::len).sum();
    let mut flat: Vec<(u128, Complex64)> = Vec::with_capacity(total);
    for partial in partials {
        flat.extend(partial);
    }
    flat.par_sort_by_key(|&(key, _)| key);

    let mut i = 0;
    while i < flat.len() {
        let (key, mut sum) = flat[i];
        let mut j = i + 1;
        while j < flat.len() && flat[j].0 == key {
            sum += flat[j].1;
            j += 1;
        }
        *dest
            .operators
            .entry(unpack_key(key))
            .or_insert(Complex64::ZERO) += sum;
        i = j;
    }
    PhaseTimes {
        expand: expanded - start,
        merge: expanded.elapsed(),
    }
}

/// `radix_partition` (idea #7): partition the packed partials into disjoint
/// bins by a cheap multiplicative hash, then reduce each bin into its own map
/// in parallel. Bins scan chunks in order, so results are deterministic.
fn radix_partition(
    dest: &mut MajoranaHashMap,
    action: &[LadderOperator],
    indices: &Array2<usize>,
    coefficients: &Array1<Complex64>,
    cfg: &MergeConfig,
) -> PhaseTimes {
    let n_bins = cfg.effective_shards().next_power_of_two();
    // Fibonacci-style multiplicative hash of the packed key; the top bits pick
    // the bin. `n_bins == 1` would need a shift of 64 (UB), so bin 0 directly.
    let bin_of = move |key: u128| -> usize {
        if n_bins == 1 {
            return 0;
        }
        let mixed = (((key >> 64) as u64) ^ key as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        (mixed >> (64 - n_bins.trailing_zeros())) as usize
    };

    let per_row = 1usize << action.len();
    let start = Instant::now();
    let partials: Vec<Vec<Vec<(u128, Complex64)>>> = (0..indices.nrows())
        .into_par_iter()
        .chunks(cfg.parallel_chunk)
        .map(|rows| {
            let capacity = if cfg.presize {
                rows.len() * per_row
            } else {
                cfg.parallel_chunk
            };
            let mut local = PackedMap::with_capacity_and_hasher(capacity, Default::default());
            for r in rows {
                let row: ArrayVec<[usize; MAX_MAJORANAS]> =
                    indices.row(r).iter().copied().collect();
                expand_term(action, &row, coefficients[r], |key, value| {
                    *local.entry(pack_key(&key)).or_insert(Complex64::ZERO) += value;
                });
            }
            let mut bins: Vec<Vec<(u128, Complex64)>> = vec![Vec::new(); n_bins];
            for (key, value) in local {
                bins[bin_of(key)].push((key, value));
            }
            bins
        })
        .collect();
    let expanded = Instant::now();

    let bin_maps: Vec<PackedMap> = (0..n_bins)
        .into_par_iter()
        .map(|bin| {
            let capacity: usize = partials.iter().map(|bins| bins[bin].len()).sum();
            let mut out = PackedMap::with_capacity_and_hasher(capacity, Default::default());
            for bins in &partials {
                for &(key, value) in &bins[bin] {
                    *out.entry(key).or_insert(Complex64::ZERO) += value;
                }
            }
            out
        })
        .collect();
    if cfg.presize {
        dest.operators
            .reserve(bin_maps.iter().map(StdHashMap::len).sum());
    }
    for map in bin_maps {
        for (key, value) in map {
            *dest
                .operators
                .entry(unpack_key(key))
                .or_insert(Complex64::ZERO) += value;
        }
    }
    PhaseTimes {
        expand: expanded - start,
        merge: expanded.elapsed(),
    }
}

/// `kway_merge` (idea #10): sort each packed partial locally (keys within a
/// chunk are unique after deduplication), then combine them with a serial
/// binary-heap k-way merge, summing equal keys as they stream past. The heap
/// orders ties by chunk index, so summation order is deterministic.
fn kway_merge(
    dest: &mut MajoranaHashMap,
    action: &[LadderOperator],
    indices: &Array2<usize>,
    coefficients: &Array1<Complex64>,
    cfg: &MergeConfig,
) -> PhaseTimes {
    let start = Instant::now();
    let mut runs = expand_packed_partials(action, indices, coefficients, cfg);
    runs.par_iter_mut()
        .for_each(|run| run.sort_unstable_by_key(|&(key, _)| key));
    let expanded = Instant::now();

    let mut cursors = vec![0usize; runs.len()];
    let mut heap: BinaryHeap<Reverse<(u128, usize)>> = runs
        .iter()
        .enumerate()
        .filter(|(_, run)| !run.is_empty())
        .map(|(i, run)| Reverse((run[0].0, i)))
        .collect();

    let mut current: Option<(u128, Complex64)> = None;
    while let Some(Reverse((key, run_idx))) = heap.pop() {
        let value = runs[run_idx][cursors[run_idx]].1;
        cursors[run_idx] += 1;
        if cursors[run_idx] < runs[run_idx].len() {
            heap.push(Reverse((runs[run_idx][cursors[run_idx]].0, run_idx)));
        }
        current = match current {
            Some((current_key, sum)) if current_key == key => Some((key, sum + value)),
            Some((current_key, sum)) => {
                *dest
                    .operators
                    .entry(unpack_key(current_key))
                    .or_insert(Complex64::ZERO) += sum;
                Some((key, value))
            }
            None => Some((key, value)),
        };
    }
    if let Some((current_key, sum)) = current {
        *dest
            .operators
            .entry(unpack_key(current_key))
            .or_insert(Complex64::ZERO) += sum;
    }
    PhaseTimes {
        expand: expanded - start,
        merge: expanded.elapsed(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::c64;
    use tinyvec::array_vec;

    #[test]
    fn pack_key_roundtrips() {
        let keys: Vec<MajoranaKey> = vec![
            MajoranaKey::new(),
            array_vec!([u16; MAX_MAJORANAS] => 0),
            array_vec!([u16; MAX_MAJORANAS] => 0, 1),
            array_vec!([u16; MAX_MAJORANAS] => 3, 17, 42),
            array_vec!([u16; MAX_MAJORANAS] => 0, 0, 0, 0, 0, 0, 0),
            array_vec!([u16; MAX_MAJORANAS] => u16::MAX, u16::MAX, u16::MAX),
            array_vec!([u16; MAX_MAJORANAS] => 1, 2, 3, 4, 5, 6, u16::MAX),
        ];
        for key in keys {
            assert_eq!(
                unpack_key(pack_key(&key)),
                key,
                "roundtrip failed for {key:?}"
            );
        }
    }

    #[test]
    fn pack_key_distinguishes_padding_from_zeros() {
        // A key of two zeros must not collide with one, three, ... zeros.
        let lengths: Vec<MajoranaKey> = (0..=MAX_MAJORANAS)
            .map(|n| {
                let mut key = MajoranaKey::new();
                for _ in 0..n {
                    key.push(0);
                }
                key
            })
            .collect();
        let packed: Vec<u128> = lengths.iter().map(pack_key).collect();
        for i in 0..packed.len() {
            for j in (i + 1)..packed.len() {
                assert_ne!(packed[i], packed[j]);
            }
        }
    }

    #[test]
    fn strategy_names_roundtrip_through_parse() {
        for strategy in MergeStrategy::ALL {
            assert_eq!(MergeStrategy::parse(strategy.name()), Some(strategy));
            assert_eq!(
                MergeStrategy::parse(&strategy.name().to_ascii_uppercase()),
                Some(strategy)
            );
        }
        assert_eq!(MergeStrategy::parse("not_a_strategy"), None);
    }

    /// Two-body test fixture: `n_terms` rows of "++--" over `n_orb` modes with
    /// coefficients produced by `coeff(t)`.
    fn two_body_fixture(
        n_terms: usize,
        n_orb: usize,
        coeff: impl Fn(usize) -> Complex64,
    ) -> (Vec<LadderOperator>, Array2<usize>, Array1<Complex64>) {
        use LadderOperator::{Annihilation, Creation};
        let action = vec![Creation, Creation, Annihilation, Annihilation];
        let mut indices = Array2::<usize>::zeros((n_terms, 4));
        let mut coefficients = Array1::<Complex64>::zeros(n_terms);
        for t in 0..n_terms {
            indices[[t, 0]] = t % n_orb;
            indices[[t, 1]] = (t / n_orb) % n_orb;
            indices[[t, 2]] = (t / (n_orb * n_orb)) % n_orb;
            indices[[t, 3]] = (t / (n_orb * n_orb * n_orb)) % n_orb;
            coefficients[t] = coeff(t);
        }
        (action, indices, coefficients)
    }

    /// Serial reference: accumulate every row one term at a time.
    fn serial_reference(
        action: &[LadderOperator],
        indices: &Array2<usize>,
        coefficients: &Array1<Complex64>,
    ) -> MajoranaHashMap {
        let mut reference = MajoranaHashMap::new();
        for t in 0..indices.nrows() {
            let row: Vec<usize> = indices.row(t).iter().copied().collect();
            reference.append_term(action, &row, coefficients[t]);
        }
        reference
    }

    /// Knob combinations every strategy is swept over, exercising the shard
    /// override, non-default chunking, the serial-merge shortcut, and presizing.
    fn knob_combinations(strategy: MergeStrategy) -> Vec<MergeConfig> {
        let base = MergeConfig {
            strategy,
            ..MergeConfig::default()
        };
        vec![
            base.clone(),
            MergeConfig {
                presize: true,
                ..base.clone()
            },
            MergeConfig {
                n_shards: Some(3),
                parallel_chunk: 17,
                ..base.clone()
            },
            MergeConfig {
                n_shards: Some(1),
                ..base.clone()
            },
            MergeConfig {
                serial_merge_threshold: usize::MAX,
                ..base
            },
        ]
    }

    /// Every strategy, under every knob combination, must reproduce the serial
    /// reference **exactly**. Coefficients are small dyadic rationals, so all
    /// sums are exact in `f64` and the comparison is bit-for-bit regardless of
    /// accumulation order — this validates even the strategies that do not fix
    /// the floating-point summation order.
    #[test]
    fn every_strategy_matches_serial_exactly_on_dyadic_input() {
        let (action, indices, coefficients) =
            two_body_fixture(1200, 5, |t| c64((t % 4 + 1) as f64, (t % 3) as f64 * 0.5));
        let reference = serial_reference(&action, &indices, &coefficients);
        for strategy in MergeStrategy::ALL {
            for cfg in knob_combinations(strategy) {
                let mut dest = MajoranaHashMap::new();
                expand_and_merge(&mut dest, &action, &indices, &coefficients, &cfg);
                assert_eq!(
                    dest.operators,
                    reference.operators,
                    "strategy {} with config {cfg:?} diverged from the serial reference",
                    strategy.name()
                );
            }
        }
    }

    /// Merging into a non-empty destination must accumulate, not replace:
    /// running the same expansion twice doubles every coefficient.
    #[test]
    fn every_strategy_accumulates_into_nonempty_destination() {
        let (action, indices, coefficients) =
            two_body_fixture(600, 4, |t| c64((t % 8 + 1) as f64, 0.25 * (t % 5) as f64));
        let mut reference = serial_reference(&action, &indices, &coefficients);
        let second = serial_reference(&action, &indices, &coefficients);
        for (key, value) in second.operators {
            *reference.operators.entry(key).or_insert(Complex64::ZERO) += value;
        }
        for strategy in MergeStrategy::ALL {
            let cfg = MergeConfig {
                strategy,
                ..MergeConfig::default()
            };
            let mut dest = MajoranaHashMap::new();
            expand_and_merge(&mut dest, &action, &indices, &coefficients, &cfg);
            expand_and_merge(&mut dest, &action, &indices, &coefficients, &cfg);
            assert_eq!(
                dest.operators,
                reference.operators,
                "strategy {} did not accumulate into a non-empty destination",
                strategy.name()
            );
        }
    }

    /// With non-dyadic coefficients the summation order matters at the level of
    /// rounding, so order-nondeterministic strategies may differ from the
    /// serial reference in the last ulps: compare within a tolerance.
    #[test]
    fn every_strategy_matches_serial_within_tolerance_on_general_input() {
        let (action, indices, coefficients) = two_body_fixture(1500, 6, |t| {
            c64(1.0 + 0.001 * t as f64, 0.1 + 0.0003 * t as f64)
        });
        let reference = serial_reference(&action, &indices, &coefficients);
        for strategy in MergeStrategy::ALL {
            let cfg = MergeConfig {
                strategy,
                ..MergeConfig::default()
            };
            let mut dest = MajoranaHashMap::new();
            expand_and_merge(&mut dest, &action, &indices, &coefficients, &cfg);
            assert_eq!(
                dest.operators.len(),
                reference.operators.len(),
                "strategy {} produced a different key set",
                strategy.name()
            );
            for (key, expected) in reference.operators.iter() {
                let actual = dest
                    .operators
                    .get(key)
                    .unwrap_or_else(|| panic!("strategy {} lost key {key:?}", strategy.name()));
                let scale = 1.0 + expected.norm();
                assert!(
                    (actual - expected).norm() <= 1e-12 * scale,
                    "strategy {}: key {key:?} expected {expected}, got {actual}",
                    strategy.name()
                );
            }
        }
    }

    #[test]
    fn every_strategy_handles_empty_input() {
        let indices = Array2::<usize>::zeros((0, 4));
        let coefficients = Array1::<Complex64>::zeros(0);
        let action = [
            LadderOperator::Creation,
            LadderOperator::Creation,
            LadderOperator::Annihilation,
            LadderOperator::Annihilation,
        ];
        for strategy in MergeStrategy::ALL {
            let mut dest = MajoranaHashMap::new();
            let cfg = MergeConfig {
                strategy,
                ..MergeConfig::default()
            };
            expand_and_merge(&mut dest, &action, &indices, &coefficients, &cfg);
            assert!(dest.operators.is_empty(), "strategy {}", strategy.name());
        }
    }
}
