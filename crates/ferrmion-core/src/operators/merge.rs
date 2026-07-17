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
//! | `FERRMION_MERGE_STRATEGY` | one of the strategy names below | `radix_partition` |
//! | `FERRMION_MERGE_SHARDS` | shard/bin count for sharded strategies | rayon thread count |
//! | `FERRMION_PARALLEL_CHUNK` | terms expanded per phase-1 task | `64` |
//! | `FERRMION_MERGE_SERIAL_THRESHOLD` | total intermediate entries at or below which the merge runs serially | `0` (off) |
//! | `FERRMION_MERGE_PRESIZE` | pre-size hash maps from entry-count estimates | on |
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
//! All strategies fix the floating-point summation order of every key to chunk
//! order, so results are bit-for-bit deterministic and independent of the
//! thread count.
//!
//! * `radix_partition` — **the default**: locally deduplicated `u128`-packed
//!   entries are partitioned into disjoint bins by a multiplicative hash; each
//!   bin is reduced into its own map in parallel. Adopted after a benchmark
//!   campaign (M3 Pro 11-core and 4-core x86): with pre-sizing it was ~15%
//!   faster end-to-end and ~1.6-1.8x faster than `baseline` in merge-dominated
//!   micro-benchmarks at high thread counts, with no material single-thread
//!   regression.
//! * `baseline` — the original algorithm, kept as the reference for regression
//!   comparisons: expand chunks into thread-local maps, then merge with one
//!   task per shard where every shard re-scans every entry of every partial
//!   map and re-hashes it (FNV-1a) to test ownership. Combined with
//!   `FERRMION_MERGE_PRESIZE=0` this reproduces the pre-campaign behaviour
//!   exactly.
//! * `hash_cache` — as `baseline`, but each partial is first flattened to a
//!   vector with the shard id computed **once** per entry, so the per-shard
//!   scans are cheap sequential passes with no re-hashing. The closest
//!   runner-up in the campaign.
//! * `shard_phase1` — phase-1 workers route each expanded term directly into
//!   one of `n_shards` local buckets, so phase 2 only concatenates same-shard
//!   buckets. Best parallel efficiency in merge-dominated micro-benchmarks,
//!   but pays a 15-20% single-thread penalty; kept for future re-evaluation on
//!   higher-core hardware.
//!
//! Four further candidates (`fx_hash`, `tree_reduce`, `sort_scan`,
//! `kway_merge`) were benchmarked and removed after being dominated on both
//! runtime and scaling; see the repository history for their implementations
//! and `scripts/README.md` for the measurement method.
use super::fermion::{expand_term, MajoranaHashMap, MajoranaKey, MAX_MAJORANAS};
use super::ladder::LadderOperator;
use ndarray::{Array1, Array2};
use num_complex::Complex64;
use rayon::prelude::*;
use std::collections::HashMap as StdHashMap;
use std::sync::OnceLock;
use std::time::{Duration, Instant};
use tinyvec::ArrayVec;

/// Map from Majorana keys to coefficients, matching `MajoranaHashMap`'s storage.
type AMap = StdHashMap<MajoranaKey, Complex64, ahash::RandomState>;
/// Map keyed by packed `u128` Majorana keys, used by `radix_partition`.
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
    ShardPhase1,
    RadixPartition,
}

impl MergeStrategy {
    /// Every strategy, in the order they are documented. Used by the
    /// correctness sweep tests and kept in sync with [`Self::parse`].
    pub(crate) const ALL: [MergeStrategy; 4] = [
        MergeStrategy::Baseline,
        MergeStrategy::HashCache,
        MergeStrategy::ShardPhase1,
        MergeStrategy::RadixPartition,
    ];

    /// The environment-variable spelling of this strategy.
    pub(crate) fn name(&self) -> &'static str {
        match self {
            MergeStrategy::Baseline => "baseline",
            MergeStrategy::HashCache => "hash_cache",
            MergeStrategy::ShardPhase1 => "shard_phase1",
            MergeStrategy::RadixPartition => "radix_partition",
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
            strategy: MergeStrategy::RadixPartition,
            n_shards: None,
            parallel_chunk: DEFAULT_PARALLEL_CHUNK,
            serial_merge_threshold: 0,
            // Pre-sizing was a measured win for every strategy in the
            // benchmark campaign; `FERRMION_MERGE_PRESIZE=0` turns it off.
            presize: true,
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
            presize: env_bool("FERRMION_MERGE_PRESIZE").unwrap_or(defaults.presize),
            timing: env_bool("FERRMION_MERGE_TIMING").unwrap_or(defaults.timing),
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
/// panicking on anything else. Returns `None` when the variable is unset so
/// callers can apply their own default.
fn env_bool(name: &str) -> Option<bool> {
    match std::env::var(name) {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" => Some(true),
            "" | "0" | "false" | "no" => Some(false),
            _ => panic!("{name}={value:?} is not a valid boolean (use 1/true/yes or 0/false/no)"),
        },
        Err(_) => None,
    }
}

/// Wall-clock durations of the two phases of a parallel merge.
struct PhaseTimes {
    expand: Duration,
    merge: Duration,
}

/// Expand the rows of a `FermionSparse` (its shared `action`, per-row mode
/// `indices` and `coefficients`) into Majorana terms and merge them into
/// `dest`, using the algorithm selected by `cfg`.
///
/// Every strategy produces the same map, summing each key's contributions in
/// chunk order, so the result is bit-for-bit deterministic and independent of
/// the strategy, the thread count, and rayon's scheduling.
pub(super) fn expand_and_merge(
    dest: &mut MajoranaHashMap,
    action: &[LadderOperator],
    indices: &Array2<usize>,
    coefficients: &Array1<Complex64>,
    cfg: &MergeConfig,
) {
    let n_terms = indices.nrows();
    let times = match cfg.strategy {
        MergeStrategy::Baseline => rescan_strategy(dest, action, indices, coefficients, cfg),
        MergeStrategy::HashCache => hash_cache(dest, action, indices, coefficients, cfg),
        MergeStrategy::ShardPhase1 => shard_phase1(dest, action, indices, coefficients, cfg),
        MergeStrategy::RadixPartition => radix_partition(dest, action, indices, coefficients, cfg),
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
fn expand_partials(
    action: &[LadderOperator],
    indices: &Array2<usize>,
    coefficients: &Array1<Complex64>,
    cfg: &MergeConfig,
) -> Vec<AMap> {
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
        .collect()
}

/// Serially fold a sequence of merged maps into the destination, optionally
/// reserving the exact entry count first.
fn fold_maps_into_dest(dest: &mut MajoranaHashMap, maps: Vec<AMap>, presize: bool) {
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

/// `baseline`: the original sharded merge where each shard re-scans every
/// partial and re-hashes each key to test ownership.
fn rescan_strategy(
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
    /// accumulation order.
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

    /// With non-dyadic coefficients the summation order matters at the level
    /// of rounding. All current strategies sum in chunk order (which differs
    /// from the serial term order), so compare within a tolerance.
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
