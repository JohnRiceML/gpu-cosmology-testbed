//! Hierarchical (block) individual timesteps with a synchronised kick-drift-kick leapfrog,
//! in the style of GADGET-2 / ChaNGa.
//!
//! # Scheme
//! The frame `dt` is the base step, split into `2^BIN_MAX` integer ticks. A body in bin
//! `b` takes steps of `dt / 2^b` (`ticks(b)` ticks). Bins come from the acceleration
//! criterion `dt_i = ETA * sqrt(SOFTENING / |a_i|)` (`a_i` = body-body gravity + external
//! field; the mouse well alone may refine only down to `WELL_BIN_MAX`), rounded down to a
//! power of two.
//!
//! Per frame:
//! 1. **Sync point (tick 0)**: every body is synchronised and has a fresh acceleration.
//!    Bins are reassigned (any finer bin; at most one level coarser than last frame, so a
//!    body that just had a close encounter, or a merge product, eases out), then every
//!    body gets its opening half kick `v += a * dt_i / 2`.
//! 2. **Loop**: let `L` be the step of the finest occupied bin. **Drift all bodies** by
//!    `L` (positions stay synchronised: in KDK a body's velocity is constant between its
//!    kicks, so drifting it in pieces is exactly its own leapfrog drift, and the tree and
//!    direct sums always see every source at the current time; no prediction needed).
//!    Bodies whose step ends at this tick are **active**: they alone get new forces
//!    (`GravityScratch::compute_active`: direct summation over all sources for small
//!    active sets, otherwise the Barnes-Hut tree built over all bodies but walked only for
//!    leaves containing active sinks), then their closing half kick. Unless the frame is
//!    over, each active body picks its next bin (finer: always allowed; coarser: only to a
//!    bin whose step boundary coincides with this tick) and gets its opening half kick.
//! 3. **Frame end**: all bins end together, so every body is synchronised again with a
//!    fresh acceleration at its final position (the cache for the next frame).
//!
//! With every body in the same bin this is exactly the uniform KDK leapfrog.
//!
//! # Collisions
//! Merges are resolved at the frame-end synchronisation point only, where every velocity
//! is a full-step (synchronised) velocity, so merges conserve momentum exactly as before.
//! The merged body takes the finest bin of its group. Light particles in SubhaloMerger
//! don't collide with each other, and a body would need to cross a core's radius (tens
//! of units) within one frame to be missed, so checking once per frame loses nothing
//! visible.

use crate::gravity::Method;
use crate::{Input, SOFTENING, World, accel_parts, confine};
use rayon::prelude::*;

/// Finest bin: steps down to `dt / 2^BIN_MAX` (1/7680 s at 60 fps).
pub(crate) const BIN_MAX: u8 = 7;
/// Timestep accuracy parameter of the acceleration criterion. Below the usual 0.025-0.05
/// on purpose: with radius = 2.5 sqrt(m) every body's surface gravity is G/2.5^2 = 6.4, so
/// outside merge radii |a| rarely exceeds ~10, and with eta = 0.025 everything would sit
/// in bin 0 (dt = 1/60), slightly *less* accurate than the old uniform 1/120 substeps
/// (Plummer test: 1.3e-5 vs 2e-6 energy drift). With 0.015 bodies near core surfaces and
/// in the inner disc take 1/120 or finer, matching or beating the uniform scheme in every
/// accuracy test at 0.5-0.9x the force evaluations.
pub(crate) const ETA: f32 = 0.015;

/// Which integrator `step` uses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Integrator {
    /// Block timesteps with the Barnes-Hut force method (large N), uniform substeps with
    /// direct summation (small N: unchanged behaviour, exact momentum conservation).
    #[default]
    Auto,
    Uniform,
    Block,
}

impl Integrator {
    pub(crate) fn use_block(self, method: Method) -> bool {
        match self {
            Integrator::Auto => method == Method::Tree,
            Integrator::Uniform => false,
            Integrator::Block => true,
        }
    }
}

#[inline]
pub(crate) fn ticks(bin: u8) -> u32 {
    (1u32 << BIN_MAX) >> bin
}

/// Bin for a body with squared total acceleration `a2` when the base step is `base`.
#[inline]
pub(crate) fn bin_for(a2: f32, base: f32) -> u8 {
    if !a2.is_finite() {
        return BIN_MAX;
    }
    if a2 <= 0.0 {
        return 0;
    }
    let want = ETA * (SOFTENING / a2.sqrt()).sqrt();
    let (mut bin, mut step) = (0u8, base);
    while step > want && bin < BIN_MAX {
        step *= 0.5;
        bin += 1;
    }
    bin
}

/// Finest bin the mouse well alone can force: 1/120 s at 60 fps, the old uniform substep.
/// The well is a gameplay force with a ~3850 peak acceleration and ~16 even 500 units
/// away; letting it drive bins 6-7 made every frame with the well held cost 64-128 passes
/// (~23 ms at 50k bodies), and bin 2 still cost 2.8 evaluations per body. Physical
/// forces (bodies, external field) can use every bin.
pub(crate) const WELL_BIN_MAX: u8 = 1;

/// Bin wanted for physical acceleration `phys` and total (incl. mouse well) `total`.
#[inline]
pub(crate) fn wanted_bin(phys: [f32; 2], total: [f32; 2], base: f32) -> u8 {
    let p = bin_for(phys[0] * phys[0] + phys[1] * phys[1], base);
    let t = bin_for(total[0] * total[0] + total[1] * total[1], base).min(WELL_BIN_MAX);
    p.max(t)
}

/// Totals over one block-timestep frame.
pub(crate) struct BlockReport {
    pub(crate) force_evals: u64,
    pub(crate) interactions: u64,
    /// Bit b set = bin b was occupied at some point during the frame.
    pub(crate) bins_used: u32,
}

impl World {
    /// Advance all bodies by `dt` with block timesteps. Precondition: `self.acc` holds the
    /// body-body acceleration of every body at its current position. Postcondition: the
    /// same, at the new positions. Does not resolve collisions.
    pub(crate) fn step_block(&mut self, dt: f32, input: &Input) -> BlockReport {
        let n = self.bodies.len();
        let total_ticks = 1u32 << BIN_MAX;
        let h = dt / total_ticks as f32;
        let field = self.external.as_ref();
        self.bins.resize(n, 0);
        let mut rep = BlockReport { force_evals: 0, interactions: 0, bins_used: 0 };

        // Each pass is fused into two parallel sweeps over the bodies (drift + active mask,
        // then kicks + rebin + occupied-bin OR) plus the force evaluation, so the per-pass
        // fork/join overhead stays small even when only a few bodies are active.

        // 1. Sync point: bins + opening half kicks.
        let mut used = self
            .bodies
            .par_iter_mut()
            .zip(&self.acc)
            .zip(&mut self.bins)
            .with_min_len(2048)
            .map(|((b, a), bin)| {
                let (phys, at) = accel_parts(b.pos, *a, input, field);
                let want = wanted_bin(phys, at, dt);
                *bin = want.max(bin.saturating_sub(1));
                let hk = 0.5 * h * ticks(*bin) as f32;
                b.vel[0] += at[0] * hk;
                b.vel[1] += at[1] * hk;
                1u32 << *bin
            })
            .reduce(|| 0, |x, y| x | y);

        let mut mask = std::mem::take(&mut self.active_mask);
        let mut active = std::mem::take(&mut self.active_idx);
        mask.resize(n, false);
        let mut tick = 0u32;
        while tick < total_ticks {
            // Finest occupied bin sets the next event.
            rep.bins_used |= used;
            let finest = 31 - used.leading_zeros();
            let step = ticks(finest as u8);
            // Every body starts its steps on multiples of its own step length, and the
            // finest step divides all others, so `tick` is a multiple of `step`.
            debug_assert_eq!(tick % step, 0);
            let next = tick + step;

            // 2a. Drift everyone to the next event; mark bodies whose step ends there.
            let dtd = step as f32 * h;
            let count: usize = self
                .bodies
                .par_iter_mut()
                .zip(&mut mask)
                .zip(&self.bins)
                .with_min_len(4096)
                .map(|((b, m), &bin)| {
                    b.pos[0] += b.vel[0] * dtd;
                    b.pos[1] += b.vel[1] * dtd;
                    confine(b);
                    *m = next.is_multiple_of(ticks(bin));
                    *m as usize
                })
                .sum();
            tick = next;

            // 2b. New forces for active sinks only.
            self.gravity
                .compute_active(&self.bodies, &mask, count, &mut active, &mut self.acc);
            rep.force_evals += count as u64;
            rep.interactions += self.gravity.interactions;

            // 2c. Closing half kick, then (unless the frame is over) rebin + opening kick.
            let last = tick == total_ticks;
            used = self
                .bodies
                .par_iter_mut()
                .zip(&self.acc)
                .zip(&mut self.bins)
                .zip(&mask)
                .with_min_len(2048)
                .map(|(((b, a), bin), &m)| {
                    if !m || last {
                        if m {
                            let at = accel_parts(b.pos, *a, input, field).1;
                            let hk = 0.5 * h * ticks(*bin) as f32;
                            b.vel[0] += at[0] * hk;
                            b.vel[1] += at[1] * hk;
                        }
                        return 1u32 << *bin;
                    }
                    let (phys, at) = accel_parts(b.pos, *a, input, field);
                    let hk = 0.5 * h * ticks(*bin) as f32;
                    b.vel[0] += at[0] * hk;
                    b.vel[1] += at[1] * hk;
                    let want = wanted_bin(phys, at, dt);
                    let mut nb = *bin;
                    if want > nb {
                        nb = want; // smaller steps: always allowed
                    } else {
                        // Larger steps only where that bin's step boundary is now.
                        while nb > want && tick.is_multiple_of(ticks(nb - 1)) {
                            nb -= 1;
                        }
                    }
                    *bin = nb;
                    let hk = 0.5 * h * ticks(nb) as f32;
                    b.vel[0] += at[0] * hk;
                    b.vel[1] += at[1] * hk;
                    1u32 << nb
                })
                .reduce(|| 0, |x, y| x | y);
        }
        self.active_mask = mask;
        self.active_idx = active;
        rep
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bin_criterion() {
        let base = 1.0 / 60.0;
        assert_eq!(bin_for(0.0, base), 0);
        assert_eq!(bin_for(1.0, base), 0); // wants 0.03 s
        // |a| = 100 wants 0.015*sqrt(0.04) = 0.003 s -> base/8 = 0.0021 -> bin 3.
        assert_eq!(bin_for(100.0 * 100.0, base), 3);
        assert_eq!(bin_for(1e12, base), BIN_MAX);
        assert_eq!(bin_for(f32::NAN, base), BIN_MAX);
        assert_eq!(ticks(0), 1 << BIN_MAX);
        assert_eq!(ticks(BIN_MAX), 1);
    }
}
