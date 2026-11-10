//! Gravity: direct O(N^2) summation for small N, Barnes-Hut (`tree` module) for large N.
//!
//! Direct sum: positions and masses are copied into structure-of-arrays scratch buffers
//! padded to a multiple of `LANES` with zero-mass entries, so the inner loop is branch-free
//! and has a fixed trip count. The inner loop accumulates into `LANES` independent partial
//! sums (instead of one scalar), which lets LLVM vectorise it without fast-math
//! reassociation. Each body's row is computed independently on the rayon pool; there is no
//! shared mutable state and the summation order is fixed, so results are deterministic.
//!
//! On x86_64 an AVX-512 or AVX2 copy of the kernel is selected at runtime when available
//! (the crate is otherwise compiled for baseline SSE2, since dependents' RUSTFLAGS are
//! unknown). The same kernel evaluates Barnes-Hut interaction lists.
//!
//! Self-interaction needs no special case: dx = dy = 0 contributes zero force.
//!
//! Method selection uses hysteresis on the body count and is only changed by `select`
//! (called once per `World::step`, never between substeps), so one frame never mixes the
//! two force methods inside the leapfrog.

use crate::tree::Tree;
use crate::{Body, G, SOFTENING};
use rayon::prelude::*;

pub(crate) const LANES: usize = 16;
const EPS2: f32 = SOFTENING * SOFTENING;

/// Switch to Barnes-Hut when the body count reaches this... Measured on a 16-thread
/// Zen 5 (AVX-512 kernel), per force evaluation: 4k direct 0.22-0.25 ms vs tree
/// 0.26-0.34 ms; 5k 0.32-0.39 vs 0.30-0.38 (break-even); 6k 0.46-0.55 vs 0.33-0.46;
/// 10k 1.3 vs 0.45-0.56; 20k 5.0 vs 0.8; 50k 31 vs 1.4-1.6.
pub(crate) const TREE_ON: usize = 5000;
/// ...and back to direct summation only when it falls below this (hysteresis, so merges
/// near the threshold don't flip the method every frame).
pub(crate) const TREE_OFF: usize = 4500;

/// With the tree selected, an active subset still uses direct summation while
/// `n_active * N` stays below this many interactions (cheaper than building a tree:
/// ~25M interactions take ~0.3 ms on 16 threads, about one tree build at 50k bodies).
pub(crate) const DIRECT_SUBSET_BUDGET: u64 = 25_000_000;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Method {
    #[default]
    Direct,
    Tree,
}

#[derive(Default)]
pub(crate) struct GravityScratch {
    xs: Vec<f32>,
    ys: Vec<f32>,
    ms: Vec<f32>,
    tree: Tree,
    method: Method,
    /// Pairwise interactions evaluated by the last `compute`.
    pub(crate) interactions: u64,
}

impl GravityScratch {
    /// Pick the force method for `n` bodies (with hysteresis). Returns true if it changed.
    pub(crate) fn select(&mut self, n: usize) -> bool {
        let want = match self.method {
            Method::Direct if n >= TREE_ON => Method::Tree,
            Method::Tree if n < TREE_OFF => Method::Direct,
            m => m,
        };
        let changed = want != self.method;
        self.method = want;
        changed
    }

    pub(crate) fn method(&self) -> Method {
        self.method
    }

    /// Compute the gravitational acceleration of every body due to all others into `acc`,
    /// using the currently selected method.
    pub(crate) fn compute(&mut self, bodies: &[Body], acc: &mut Vec<[f32; 2]>) {
        match self.method {
            Method::Direct => self.direct(bodies, acc),
            Method::Tree => self.interactions = self.tree.compute(bodies, acc),
        }
    }

    /// Block timesteps: update `acc[i]` only for the active sinks (`mask[i]`, `count` of
    /// them), with every body as a source. Small active sets use exact direct summation
    /// (n_active * N interactions, no tree build); larger ones the tree with a sink mask
    /// (leaves without sinks are skipped). `active` is scratch for the index list.
    pub(crate) fn compute_active(
        &mut self,
        bodies: &[Body],
        mask: &[bool],
        count: usize,
        active: &mut Vec<u32>,
        acc: &mut [[f32; 2]],
    ) {
        let n = bodies.len();
        if count == 0 {
            self.interactions = 0;
            return;
        }
        let small = (count as u64) * (n as u64) <= DIRECT_SUBSET_BUDGET;
        if self.method == Method::Tree && !small {
            self.interactions = self.tree.compute_masked(bodies, acc, Some(mask));
            return;
        }
        // Order-preserving parallel collect: `active` stays ascending.
        active.clear();
        active.par_extend(
            (0..n as u32)
                .into_par_iter()
                .with_min_len(4096)
                .filter(|&i| mask[i as usize]),
        );
        let active = &active[..];
        self.fill_soa(bodies);
        let (xs, ys, ms) = (&self.xs[..], &self.ys[..], &self.ms[..]);
        let simd = simd_level();
        let rows: Vec<[f32; 2]> = active
            .par_iter()
            .with_min_len(4)
            .map(|&i| {
                let b = &bodies[i as usize];
                row(b.pos[0], b.pos[1], xs, ys, ms, simd)
            })
            .collect();
        for (&i, a) in active.iter().zip(rows) {
            acc[i as usize] = a;
        }
        self.interactions = (active.len() * xs.len()) as u64;
    }

    fn fill_soa(&mut self, bodies: &[Body]) {
        let padded = bodies.len().div_ceil(LANES) * LANES;
        self.xs.clear();
        self.ys.clear();
        self.ms.clear();
        self.xs.par_extend(bodies.par_iter().map(|b| b.pos[0]));
        self.ys.par_extend(bodies.par_iter().map(|b| b.pos[1]));
        self.ms.par_extend(bodies.par_iter().map(|b| b.mass));
        // Padding: zero mass => zero contribution regardless of position.
        self.xs.resize(padded, 0.0);
        self.ys.resize(padded, 0.0);
        self.ms.resize(padded, 0.0);
    }

    pub(crate) fn direct(&mut self, bodies: &[Body], acc: &mut Vec<[f32; 2]>) {
        let n = bodies.len();
        let padded = n.div_ceil(LANES) * LANES;
        self.xs.clear();
        self.ys.clear();
        self.ms.clear();
        self.xs.extend(bodies.iter().map(|b| b.pos[0]));
        self.ys.extend(bodies.iter().map(|b| b.pos[1]));
        self.ms.extend(bodies.iter().map(|b| b.mass));
        // Padding: zero mass => zero contribution regardless of position.
        self.xs.resize(padded, 0.0);
        self.ys.resize(padded, 0.0);
        self.ms.resize(padded, 0.0);

        acc.clear();
        acc.resize(n, [0.0; 2]);

        let (xs, ys, ms) = (&self.xs[..], &self.ys[..], &self.ms[..]);
        let simd = simd_level();
        acc.par_iter_mut()
            .enumerate()
            .with_min_len(16)
            .for_each(|(i, a)| *a = row(xs[i], ys[i], xs, ys, ms, simd));
        self.interactions = (n * padded) as u64;
    }
}

/// 0 = baseline, 1 = AVX2, 2 = AVX-512F.
#[cfg(target_arch = "x86_64")]
pub(crate) fn simd_level() -> u8 {
    if std::arch::is_x86_feature_detected!("avx512f") {
        2
    } else if std::arch::is_x86_feature_detected!("avx2") {
        1
    } else {
        0
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn simd_level() -> u8 {
    0
}

/// Acceleration on a body at (xi, yi) from all sources; slices must have equal length,
/// a multiple of `LANES` (pad with zero masses).
#[inline]
pub(crate) fn row(xi: f32, yi: f32, xs: &[f32], ys: &[f32], ms: &[f32], simd: u8) -> [f32; 2] {
    #[cfg(target_arch = "x86_64")]
    {
        if simd >= 2 {
            // SAFETY: AVX-512F support was verified at runtime by `simd_level`.
            return unsafe { row_avx512(xi, yi, xs, ys, ms) };
        }
        if simd == 1 {
            // SAFETY: AVX2 support was verified at runtime by `simd_level`.
            return unsafe { row_avx2(xi, yi, xs, ys, ms) };
        }
    }
    let _ = simd;
    row_generic(xi, yi, xs, ys, ms)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn row_avx2(xi: f32, yi: f32, xs: &[f32], ys: &[f32], ms: &[f32]) -> [f32; 2] {
    row_generic(xi, yi, xs, ys, ms)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma,avx512f")]
unsafe fn row_avx512(xi: f32, yi: f32, xs: &[f32], ys: &[f32], ms: &[f32]) -> [f32; 2] {
    row_generic(xi, yi, xs, ys, ms)
}

#[inline(always)]
fn row_generic(xi: f32, yi: f32, xs: &[f32], ys: &[f32], ms: &[f32]) -> [f32; 2] {
    let mut ax = [0.0f32; LANES];
    let mut ay = [0.0f32; LANES];
    // Inputs are padded to a multiple of LANES, so the remainders are empty.
    let (xs, _) = xs.as_chunks::<LANES>();
    let (ys, _) = ys.as_chunks::<LANES>();
    let (ms, _) = ms.as_chunks::<LANES>();
    for ((xc, yc), mc) in xs.iter().zip(ys).zip(ms) {
        for k in 0..LANES {
            let dx = xc[k] - xi;
            let dy = yc[k] - yi;
            let d2 = dx * dx + dy * dy + EPS2;
            let s = mc[k] / (d2 * d2.sqrt());
            ax[k] += dx * s;
            ay[k] += dy * s;
        }
    }
    [G * ax.iter().sum::<f32>(), G * ay.iter().sum::<f32>()]
}
