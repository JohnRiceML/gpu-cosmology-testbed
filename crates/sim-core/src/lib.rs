//! CPU physics for Gravity Sandbox: multithreaded N-body gravity, collisions and merging.
//!
//! This crate is the contract between the CPU sim and the GPU renderer.
//! The public types and function signatures below are FIXED; only bodies may change.
//!
//! # Implementation overview
//! - **Gravity** (`gravity` module): direct O(N^2) summation with Plummer softening,
//!   parallelised over bodies with rayon, using padded SoA scratch arrays and a
//!   vectorisation-friendly inner loop (runtime AVX-512 / AVX2 dispatch on x86_64).
//!   From 5000 bodies up (back below 4500, hysteresis) a Barnes-Hut quadtree (`tree`
//!   module, theta = 0.5, same softening) is rebuilt every evaluation and walked in
//!   parallel. The method is chosen once per `step`; if it changes, the cached
//!   accelerations are recomputed with the new method before the first kick.
//! - **Scenarios**: `Galaxy` (`World::new`) and `SubhaloMerger` (`scenario` module). In
//!   SubhaloMerger light particles (mass < 20) are collisionless with each other and only
//!   merge into heavy bodies they touch.
//! - **Integration**: kick-drift-kick leapfrog (velocity Verlet). `step(dt)` clamps `dt`
//!   to 1/30 s. With direct summation (small N) it splits the frame into equal substeps of
//!   at most 1/120 s, exactly as before. With Barnes-Hut (large N) it uses hierarchical
//!   block timesteps (`block` module): power-of-two bins under the frame step down to
//!   dt/128, chosen per body from `dt_i = 0.015 sqrt(SOFTENING / |a_i|)`, synchronised
//!   KDK, forces only for active bodies. The body-body acceleration is cached between
//!   steps; the mouse well and the external field are evaluated fresh at every kick.
//! - **External field** (`field` module): the optional `ExternalField` grid (GPU dust
//!   gravity) is bilinearly sampled at each body at every kick, clamped to the grid.
//! - **Collisions** (`collide` module): uniform-grid broad phase after every uniform
//!   substep, or at the frame-end synchronisation point with block timesteps; overlapping
//!   groups are merged (mass, momentum conserved; mass-weighted position; mass-weighted
//!   circular-mean hue; radius = `RADIUS_K * sqrt(mass)`; finest timestep bin).
//! - **Bounds**: reflective walls at ±WORLD_HALF that keep only part of the normal speed.
//!
//! Units: a typical Galaxy body has mass 0.5..2.0 (mean ~1), radius ~1.8..3.5, scaled by
//! 4000/n above 4000 bodies (constant total mass); the central body has mass 2000 and
//! radius ~112. SubhaloMerger: host core 3000 (radius ~137), satellite cores 60..240
//! (radius ~19..39), particle masses scale as 1/n (e.g. ~0.04..0.2 at 20k bodies).

use bytemuck::{Pod, Zeroable};
use rayon::prelude::*;
use std::time::Instant;

mod block;
mod collide;
mod field;
mod gravity;
mod rng;
mod scenario;
mod tree;

/// World is a square from -WORLD_HALF..WORLD_HALF on both axes (world units).
pub const WORLD_HALF: f32 = 1000.0;
/// Newtonian gravitational constant in world units.
pub const G: f32 = 40.0;
/// Plummer softening length, avoids infinite forces at close range.
pub const SOFTENING: f32 = 4.0;
/// Max number of heaviest bodies exported to the GPU as attractors for dust particles.
pub const MAX_ATTRACTORS: usize = 64;

/// One planet. `#[repr(C)]` + Pod so the renderer can upload a `&[Body]` straight into a
/// GPU storage/instance buffer. Layout: 8 x f32 = 32 bytes (WGSL-compatible, 16-byte aligned).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct Body {
    pub pos: [f32; 2],
    pub vel: [f32; 2],
    pub mass: f32,
    pub radius: f32,
    /// Display color hue in 0..1 (renderer converts to RGB).
    pub hue: f32,
    pub _pad: f32,
}

/// A point mass pulling on GPU dust particles. 16 bytes, WGSL `vec4<f32>`-compatible:
/// (x, y, strength, softening). Strength = G * mass (negative = repel).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct Attractor {
    pub pos: [f32; 2],
    pub strength: f32,
    pub softening: f32,
}

/// Player input for this frame, in world coordinates.
#[derive(Clone, Copy, Debug, Default)]
pub struct Input {
    pub cursor: [f32; 2],
    /// +1.0 = attract (left mouse), -1.0 = repel (right mouse), 0.0 = none.
    pub well: f32,
}

/// Mouse gravity well strength (multiplied by `Input::well`).
pub const WELL_STRENGTH: f32 = 4.0e6;

#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    pub bodies: usize,
    pub largest_mass: f32,
    pub merges_total: u64,
    /// Wall-clock time of the last `step` in milliseconds.
    pub step_ms: f32,
    /// True when gravity currently uses the Barnes-Hut tree (false = direct summation).
    pub tree: bool,
    /// Pairwise force interactions computed during the last `step` (all substeps / block
    /// passes, incl. SIMD padding).
    pub interactions: u64,
    /// Body force evaluations per frame divided by the body count (1.0 = every body once
    /// per substep with uniform steps; lower = block timesteps are saving work).
    pub force_evals_per_body: f32,
    /// Number of occupied hierarchical timestep bins in the last step.
    pub timestep_bins: u32,
}

/// A gravitational acceleration field sampled on a regular grid covering the whole world,
/// e.g. the GPU particle-mesh field of the self-gravitating dust, read back once per frame.
/// Cell (ix, iy) center is at `-WORLD_HALF + (i + 0.5) * cell` with `cell = 2*WORLD_HALF/size`.
/// `acc[iy * size + ix]` is the acceleration (world units / s^2), already multiplied by G.
#[derive(Clone, Debug, Default)]
pub struct ExternalField {
    pub size: usize,
    pub acc: Vec<[f32; 2]>,
}

/// Which initial conditions `World::with_scenario` builds. `reset` keeps the current one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Scenario {
    /// One rotating disc galaxy around a heavy central body (`World::new`).
    #[default]
    Galaxy,
    /// A massive host halo + disc with several satellite subhalos (dense core + bound cloud
    /// of bodies) falling in on eccentric orbits: tidal stripping, streams, and mergers.
    SubhaloMerger,
    /// Cosmological structure formation: a periodic, expanding box of dark matter (the GPU
    /// dust) grown from Zel'dovich initial conditions. All mass lives on the GPU, so the
    /// CPU world starts empty (bodies can still be spawned by the player).
    CosmicWeb,
}

impl Scenario {
    pub const ALL: [Scenario; 3] = [Scenario::Galaxy, Scenario::SubhaloMerger, Scenario::CosmicWeb];

    pub fn name(self) -> &'static str {
        match self {
            Scenario::Galaxy => "galaxy",
            Scenario::SubhaloMerger => "subhalo",
            Scenario::CosmicWeb => "cosmic",
        }
    }

    /// Parses `name()` output (case-insensitive).
    pub fn from_name(s: &str) -> Option<Scenario> {
        Scenario::ALL.into_iter().find(|sc| sc.name().eq_ignore_ascii_case(s))
    }
}

// ---- private tuning constants -------------------------------------------------------

/// radius = RADIUS_K * sqrt(mass): area is proportional to mass, so merges conserve area.
const RADIUS_K: f32 = 2.5;
/// Largest frame dt accepted by `step` (bigger values are clamped, e.g. after a hitch).
const MAX_DT: f32 = 1.0 / 30.0;
/// Largest integration substep.
const MAX_SUBSTEP: f32 = 1.0 / 120.0;
/// Softening length of the mouse well.
const WELL_SOFTENING: f32 = 20.0;
/// Fraction of the normal velocity kept when bouncing off a wall.
const WALL_RESTITUTION: f32 = 0.5;
/// Small body masses: MASS_MIN + (MASS_MAX - MASS_MIN) * u^2, mean ~1.0.
const MASS_MIN: f32 = 0.5;
const MASS_MAX: f32 = 2.0;
const CENTRAL_MASS: f32 = 2000.0;
const DISC_OUTER: f32 = 800.0;
const CLUSTER_RADIUS: f32 = 30.0;
/// Galaxy body count above which particle masses shrink as GALAXY_REF_N / n.
const GALAXY_REF_N: usize = 4000;
const CLUSTER_JITTER: f32 = 3.0;

#[inline]
pub(crate) fn radius_for_mass(mass: f32) -> f32 {
    RADIUS_K * mass.max(0.0).sqrt()
}

pub struct World {
    pub bodies: Vec<Body>,
    pub stats: Stats,
    /// Cached body-body gravitational acceleration for the current positions.
    acc: Vec<[f32; 2]>,
    acc_valid: bool,
    gravity: gravity::GravityScratch,
    grid: collide::Grid,
    rng: rng::SplitMix64,
    scenario: Scenario,
    /// A collision only merges if one body has at least this mass (0 = every contact
    /// merges). SubhaloMerger makes its light particles collisionless with each other.
    merge_min_mass: f32,
    /// Optional external field (GPU dust gravity) added to body accelerations.
    external: Option<ExternalField>,
    /// Block-timestep bin per body (kept across frames and through merges).
    bins: Vec<u8>,
    active_mask: Vec<bool>,
    active_idx: Vec<u32>,
    integrator: block::Integrator,
}

impl World {
    /// Build the initial conditions for `scenario` with about `n` bodies in total.
    /// Must handle n up to at least 50_000 (Barnes-Hut territory).
    pub fn with_scenario(scenario: Scenario, n: usize, seed: u64) -> Self {
        match scenario {
            Scenario::Galaxy => World::new(n, seed),
            Scenario::CosmicWeb => {
                let _ = n;
                let mut world = World::from_bodies(Vec::new(), rng::SplitMix64::new(seed));
                world.scenario = Scenario::CosmicWeb;
                world
            }
            Scenario::SubhaloMerger => {
                let mut rng = rng::SplitMix64::new(seed);
                let bodies = scenario::subhalo_merger(n, &mut rng);
                let mut world = World::from_bodies(bodies, rng);
                world.scenario = Scenario::SubhaloMerger;
                world.merge_min_mass = scenario::COLLISION_MIN_MASS;
                world
            }
        }
    }

    fn from_bodies(bodies: Vec<Body>, rng: rng::SplitMix64) -> Self {
        let mut world = World {
            bodies,
            stats: Stats::default(),
            acc: Vec::new(),
            acc_valid: false,
            gravity: gravity::GravityScratch::default(),
            grid: collide::Grid::default(),
            rng,
            scenario: Scenario::Galaxy,
            merge_min_mass: 0.0,
            external: None,
            bins: Vec::new(),
            active_mask: Vec::new(),
            active_idx: Vec::new(),
            integrator: block::Integrator::Auto,
        };
        world.gravity.select(world.bodies.len());
        world.refresh_stats();
        world
    }

    pub fn scenario(&self) -> Scenario {
        self.scenario
    }

    /// Galaxy-like starting disc of `n` bodies orbiting the origin.
    pub fn new(n: usize, seed: u64) -> Self {
        let mut rng = rng::SplitMix64::new(seed);
        let mut bodies = Vec::with_capacity(n);
        if n > 0 {
            bodies.push(Body {
                mass: CENTRAL_MASS,
                radius: radius_for_mass(CENTRAL_MASS),
                hue: 0.12,
                ..Body::default()
            });
            let r_min = radius_for_mass(CENTRAL_MASS) + 20.0;
            // Above GALAXY_REF_N bodies the total disc mass is kept constant (lighter, smaller
            // particles), so the dynamics don't change with n and the disc doesn't overfill.
            let mscale = (GALAXY_REF_N as f32 / n as f32).min(1.0);

            // (radius, angle, mass, hue), sorted by radius to accumulate enclosed mass.
            let mut disc: Vec<(f32, f32, f32, f32)> = (1..n)
                .map(|_| {
                    let t = rng.f32().powf(0.8);
                    let r = r_min + (DISC_OUTER - r_min) * t;
                    let th = rng.f32() * std::f32::consts::TAU;
                    let m = mscale * (MASS_MIN + (MASS_MAX - MASS_MIN) * rng.f32().powi(2));
                    let hue = (0.55 + 0.45 * t + rng.range(-0.08, 0.08)).rem_euclid(1.0);
                    (r, th, m, hue)
                })
                .collect();
            disc.sort_by(|a, b| a.0.total_cmp(&b.0));

            // Avoid a burst of merges on the first frame: re-roll the angle (keeping the
            // radius, hence the sort order) of bodies that start overlapping a neighbour.
            let mut grid = collide::Grid::default();
            for _ in 0..16 {
                let probe: Vec<Body> = disc
                    .iter()
                    .map(|&(r, th, m, _)| Body {
                        pos: [r * th.cos(), r * th.sin()],
                        radius: radius_for_mass(m),
                        ..Body::default()
                    })
                    .collect();
                let pairs = grid.find_pairs(&probe);
                if pairs.is_empty() {
                    break;
                }
                for (_, j) in pairs {
                    disc[j as usize].1 = rng.f32() * std::f32::consts::TAU;
                }
            }

            let mut enclosed = CENTRAL_MASS;
            for (r, th, m, hue) in disc {
                // Circular speed for a softened point mass equal to the enclosed mass.
                let d2 = r * r + SOFTENING * SOFTENING;
                let v = (G * enclosed * r * r / (d2 * d2.sqrt())).sqrt();
                let v = v * rng.range(0.97, 1.03);
                let vr = v * rng.range(-0.02, 0.02);
                let (s, c) = th.sin_cos();
                bodies.push(Body {
                    pos: [r * c, r * s],
                    vel: [-v * s + vr * c, v * c + vr * s],
                    mass: m,
                    radius: radius_for_mass(m),
                    hue,
                    _pad: 0.0,
                });
                enclosed += m;
            }

            // Remove net momentum so the galaxy doesn't drift.
            let (mut px, mut py, mut mt) = (0.0f64, 0.0f64, 0.0f64);
            for b in &bodies {
                px += (b.mass * b.vel[0]) as f64;
                py += (b.mass * b.vel[1]) as f64;
                mt += b.mass as f64;
            }
            let (vx, vy) = ((px / mt) as f32, (py / mt) as f32);
            for b in &mut bodies {
                b.vel[0] -= vx;
                b.vel[1] -= vy;
            }
        }

        World::from_bodies(bodies, rng)
    }

    /// Spawn a burst of `n` small bodies around `at` with velocity `vel` plus small jitter.
    pub fn spawn_cluster(&mut self, at: [f32; 2], vel: [f32; 2], n: usize) {
        self.bodies.reserve(n);
        // Grow the spawn disc with the count so bodies rarely start overlapping; at a fixed
        // radius a big cluster percolates and merges into a single blob on the first substep.
        let spread = CLUSTER_RADIUS * (n as f32 / 20.0).sqrt().max(1.0);
        for _ in 0..n {
            let off = self.rng.in_disc(spread);
            let jit = self.rng.in_disc(CLUSTER_JITTER);
            let m = MASS_MIN + (MASS_MAX - MASS_MIN) * self.rng.f32().powi(2);
            let hue = self.rng.f32();
            self.bodies.push(Body {
                pos: [
                    (at[0] + off[0]).clamp(-WORLD_HALF, WORLD_HALF),
                    (at[1] + off[1]).clamp(-WORLD_HALF, WORLD_HALF),
                ],
                vel: [vel[0] + jit[0], vel[1] + jit[1]],
                mass: m,
                radius: radius_for_mass(m),
                hue,
                _pad: 0.0,
            });
        }

        // Re-roll the position of new bodies that still overlap another new body.
        let start = self.bodies.len() - n;
        for _ in 0..16 {
            let pairs = self.grid.find_pairs(&self.bodies[start..]);
            if pairs.is_empty() {
                break;
            }
            for (_, j) in pairs {
                let off = self.rng.in_disc(spread);
                self.bodies[start + j as usize].pos = [
                    (at[0] + off[0]).clamp(-WORLD_HALF, WORLD_HALF),
                    (at[1] + off[1]).clamp(-WORLD_HALF, WORLD_HALF),
                ];
            }
        }
        self.acc_valid = false;
        self.refresh_stats();
    }

    /// Advance the simulation by `dt` seconds (may substep internally).
    /// Must be parallel (rayon) and keep all bodies inside the world bounds.
    pub fn step(&mut self, dt: f32, input: &Input) {
        let t0 = Instant::now();
        let dt = if dt.is_finite() { dt.clamp(0.0, MAX_DT) } else { 0.0 };
        let n0 = self.bodies.len();
        let (mut evals, mut interactions, mut bins_used) = (0u64, 0u64, 0u32);
        if dt > 0.0 && n0 > 0 {
            // The force method (direct / Barnes-Hut) is chosen once per step, with
            // hysteresis. When it changes, the cached accelerations came from the other
            // method, so recompute them: every kick of this step then uses one method.
            if self.gravity.select(n0) {
                self.acc_valid = false;
            }
            // `bodies` is public, so also revalidate if its length changed externally.
            if !self.acc_valid || self.acc.len() != n0 {
                self.gravity.compute(&self.bodies, &mut self.acc);
                self.acc_valid = true;
                evals += n0 as u64;
                interactions += self.gravity.interactions;
            }
            if self.bins.len() != n0 {
                self.bins.resize(n0, 0);
            }
            if self.integrator.use_block(self.gravity.method()) {
                let rep = self.step_block(dt, input);
                evals += rep.force_evals;
                interactions += rep.interactions;
                bins_used = rep.bins_used;
                // Collisions at the frame-end synchronisation point (see `block`).
                let pairs = self.grid.find_pairs_heavy(&self.bodies, self.merge_min_mass);
                self.stats.merges_total +=
                    collide::merge_pairs(&mut self.bodies, &mut self.acc, &mut self.bins, &pairs);
            } else {
                // Small epsilon so dt = 1/60 gives exactly 2 substeps despite rounding.
                let substeps = ((dt / MAX_SUBSTEP - 1e-4).ceil() as usize).max(1);
                let h = dt / substeps as f32;
                for _ in 0..substeps {
                    evals += self.bodies.len() as u64;
                    self.substep(h, input);
                    interactions += self.gravity.interactions;
                }
                bins_used = 1;
            }
        }
        self.refresh_stats();
        self.stats.force_evals_per_body = if n0 > 0 { evals as f32 / n0 as f32 } else { 0.0 };
        self.stats.timestep_bins = bins_used.count_ones();
        self.stats.interactions = interactions;
        self.stats.step_ms = t0.elapsed().as_secs_f32() * 1000.0;
    }

    /// Fill `out` with up to MAX_ATTRACTORS attractors: the mouse well first (if active),
    /// then the heaviest bodies. Returns the number written.
    pub fn attractors(&self, input: &Input, out: &mut [Attractor; MAX_ATTRACTORS]) -> usize {
        let mut count = 0;
        if input.well != 0.0 {
            out[0] = Attractor {
                pos: input.cursor,
                strength: input.well * WELL_STRENGTH,
                softening: WELL_SOFTENING,
            };
            count = 1;
        }
        let bodies = &self.bodies;
        let k = (MAX_ATTRACTORS - count).min(bodies.len());
        if k == 0 {
            return count;
        }
        // Heaviest first; ties broken by index for determinism.
        let cmp = |a: &u32, b: &u32| {
            bodies[*b as usize]
                .mass
                .total_cmp(&bodies[*a as usize].mass)
                .then(a.cmp(b))
        };
        let mut idx: Vec<u32> = (0..bodies.len() as u32).collect();
        if k < idx.len() {
            idx.select_nth_unstable_by(k - 1, cmp);
            idx.truncate(k);
        }
        idx.sort_unstable_by(cmp);
        for i in idx {
            let b = &bodies[i as usize];
            out[count] = Attractor {
                pos: b.pos,
                strength: G * b.mass,
                softening: b.radius.max(SOFTENING),
            };
            count += 1;
        }
        count
    }

    /// Remove all bodies and reseed.
    /// Rebuild the current scenario with `n` bodies.
    pub fn reset(&mut self, n: usize, seed: u64) {
        let external = self.external.take();
        *self = World::with_scenario(self.scenario, n, seed);
        self.external = external;
    }

    /// Set (or clear with `None`) the external acceleration field that bodies feel on top
    /// of body-body gravity. Called by the renderer about once per frame; the field is
    /// bilinearly sampled at body positions at every kick until replaced. `Some` with
    /// `size == 0` or `acc.len() != size*size` is treated as `None`.
    pub fn set_external_field(&mut self, field: Option<ExternalField>) {
        self.external = field.filter(|f| f.size > 0 && f.acc.len() == f.size * f.size);
    }

    /// The external field currently in use, if any (lets the caller reuse its allocation).
    pub fn take_external_field(&mut self) -> Option<ExternalField> {
        self.external.take()
    }

    /// Benchmark/test hook (not part of the game API): `None` = automatic (block timesteps
    /// with Barnes-Hut, uniform substeps with direct summation), `Some(true)` = always block
    /// timesteps, `Some(false)` = always uniform substeps.
    #[doc(hidden)]
    pub fn force_integrator(&mut self, block: Option<bool>) {
        self.integrator = match block {
            None => block::Integrator::Auto,
            Some(true) => block::Integrator::Block,
            Some(false) => block::Integrator::Uniform,
        };
    }

    // ---- private helpers ------------------------------------------------------------

    /// One kick-drift-kick leapfrog substep followed by collision handling.
    /// Precondition: `self.acc` holds the body-body acceleration for current positions.
    fn substep(&mut self, h: f32, input: &Input) {
        let half = 0.5 * h;
        let field = self.external.as_ref();
        kick(&mut self.bodies, &self.acc, input, field, half);
        self.bodies.par_iter_mut().with_min_len(4096).for_each(|b| {
            b.pos[0] += b.vel[0] * h;
            b.pos[1] += b.vel[1] * h;
            confine(b);
        });
        self.gravity.compute(&self.bodies, &mut self.acc);
        kick(&mut self.bodies, &self.acc, input, field, half);

        let pairs = self.grid.find_pairs_heavy(&self.bodies, self.merge_min_mass);
        self.stats.merges_total +=
            collide::merge_pairs(&mut self.bodies, &mut self.acc, &mut self.bins, &pairs);
    }

    fn refresh_stats(&mut self) {
        self.stats.bodies = self.bodies.len();
        self.stats.largest_mass = self.bodies.iter().map(|b| b.mass).fold(0.0, f32::max);
        self.stats.tree = self.gravity.method() == gravity::Method::Tree;
    }
}

/// Benchmark hook (not part of the game API): average wall time in ms of one gravity
/// evaluation of `bodies` with direct summation (`tree == false`) or Barnes-Hut.
#[doc(hidden)]
pub fn bench_gravity_ms(bodies: &[Body], tree: bool, reps: usize) -> f64 {
    let mut g = gravity::GravityScratch::default();
    if tree {
        g.select(usize::MAX);
    }
    let mut acc = Vec::new();
    g.compute(bodies, &mut acc); // warm-up
    let t = Instant::now();
    for _ in 0..reps.max(1) {
        g.compute(bodies, &mut acc);
    }
    t.elapsed().as_secs_f64() * 1000.0 / reps.max(1) as f64
}

/// Total acceleration at `pos`: cached body-body gravity `a` + mouse well + external field.
#[inline]
fn total_acc(pos: [f32; 2], a: [f32; 2], input: &Input, field: Option<&ExternalField>) -> [f32; 2] {
    accel_parts(pos, a, input, field).1
}

/// (physical acceleration = body-body + external field, total = physical + mouse well).
#[inline]
fn accel_parts(
    pos: [f32; 2],
    a: [f32; 2],
    input: &Input,
    field: Option<&ExternalField>,
) -> ([f32; 2], [f32; 2]) {
    let (mut ax, mut ay) = (a[0], a[1]);
    if let Some(f) = field {
        let e = field::sample(f, pos);
        ax += e[0];
        ay += e[1];
    }
    let phys = [ax, ay];
    let well = input.well * WELL_STRENGTH;
    if well != 0.0 {
        let dx = input.cursor[0] - pos[0];
        let dy = input.cursor[1] - pos[1];
        let d2 = dx * dx + dy * dy + WELL_SOFTENING * WELL_SOFTENING;
        let s = well / (d2 * d2.sqrt());
        ax += dx * s;
        ay += dy * s;
    }
    (phys, [ax, ay])
}

/// v += (a_gravity + a_well + a_external) * h
fn kick(bodies: &mut [Body], acc: &[[f32; 2]], input: &Input, field: Option<&ExternalField>, h: f32) {
    bodies.par_iter_mut().zip(acc).with_min_len(4096).for_each(|(b, a)| {
        let at = total_acc(b.pos, *a, input, field);
        b.vel[0] += at[0] * h;
        b.vel[1] += at[1] * h;
    });
}

/// Reflective, damped walls at ±WORLD_HALF. Also recovers from non-finite state.
fn confine(b: &mut Body) {
    for k in 0..2 {
        if !b.pos[k].is_finite() || !b.vel[k].is_finite() {
            b.pos[k] = b.pos[k].clamp(-WORLD_HALF, WORLD_HALF);
            if !b.pos[k].is_finite() {
                b.pos[k] = 0.0;
            }
            b.vel[k] = 0.0;
        }
        if b.pos[k] > WORLD_HALF {
            b.pos[k] = WORLD_HALF;
            b.vel[k] = -b.vel[k].abs() * WALL_RESTITUTION;
        } else if b.pos[k] < -WORLD_HALF {
            b.pos[k] = -WORLD_HALF;
            b.vel[k] = b.vel[k].abs() * WALL_RESTITUTION;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn momentum(w: &World) -> [f64; 2] {
        w.bodies.iter().fold([0.0; 2], |p, b| {
            [
                p[0] + (b.mass * b.vel[0]) as f64,
                p[1] + (b.mass * b.vel[1]) as f64,
            ]
        })
    }

    fn momentum_scale(w: &World) -> f64 {
        w.bodies
            .iter()
            .map(|b| (b.mass * b.vel[0].hypot(b.vel[1])) as f64)
            .sum()
    }

    fn total_mass(w: &World) -> f64 {
        w.bodies.iter().map(|b| b.mass as f64).sum()
    }

    fn body(pos: [f32; 2], vel: [f32; 2], mass: f32, hue: f32) -> Body {
        Body {
            pos,
            vel,
            mass,
            radius: radius_for_mass(mass),
            hue,
            _pad: 0.0,
        }
    }

    #[test]
    fn galaxy_setup() {
        let w = World::new(1000, 3);
        assert_eq!(w.bodies.len(), 1000);
        assert_eq!(w.stats.bodies, 1000);
        assert_eq!(w.stats.largest_mass, CENTRAL_MASS);
        let p = momentum(&w);
        assert!(p[0].hypot(p[1]) < 1e-3 * momentum_scale(&w));
        assert_eq!(World::new(0, 1).bodies.len(), 0);
        // Initial disc is de-overlapped: the first frame merges (almost) nothing.
        let mut w = w;
        w.step(1.0 / 60.0, &Input::default());
        assert!(w.stats.merges_total < 10, "merges {}", w.stats.merges_total);
    }

    #[test]
    fn momentum_conserved_without_input() {
        let mut w = World::new(1500, 7);
        let m0 = total_mass(&w);
        let scale = momentum_scale(&w);
        let p0 = momentum(&w);
        for _ in 0..30 {
            w.step(1.0 / 60.0, &Input::default());
        }
        let p1 = momentum(&w);
        let dp = (p1[0] - p0[0]).hypot(p1[1] - p0[1]);
        assert!(dp < 1e-3 * scale, "momentum drift {dp} vs scale {scale}");
        assert!(((total_mass(&w) - m0) / m0).abs() < 1e-5);
    }

    #[test]
    fn merge_conserves_mass_and_momentum() {
        let mut w = World::new(0, 1);
        w.bodies.push(body([0.0, 0.0], [10.0, 0.0], 2.0, 0.95));
        w.bodies.push(body([2.0, 0.0], [-4.0, 3.0], 1.0, 0.05));
        let p0 = momentum(&w);
        w.step(1e-4, &Input::default());
        assert_eq!(w.bodies.len(), 1);
        assert_eq!(w.stats.merges_total, 1);
        let b = w.bodies[0];
        assert!((b.mass - 3.0).abs() < 1e-5);
        assert!((b.radius - radius_for_mass(3.0)).abs() < 1e-5);
        let p1 = momentum(&w);
        assert!((p1[0] - p0[0]).abs() < 1e-3 && (p1[1] - p0[1]).abs() < 1e-3);
        // Mass-weighted position ~ (2/3, 0).
        assert!((b.pos[0] - 2.0 / 3.0).abs() < 0.05);
        // Hue wrap-around: blend of 0.95 and 0.05 must stay near 0/1, not 0.65.
        assert!(b.hue > 0.9 || b.hue < 0.05, "hue {}", b.hue);
    }

    #[test]
    fn one_body_overlapping_several_merges_all() {
        let mut w = World::new(0, 1);
        w.bodies.push(body([-500.0, -500.0], [0.0, 0.0], 1.0, 0.5)); // far away, untouched
        w.bodies.push(body([0.0, 0.0], [0.0, 0.0], 4.0, 0.3));
        w.bodies.push(body([4.0, 0.0], [1.0, 0.0], 1.0, 0.3));
        w.bodies.push(body([-4.0, 0.0], [-1.0, 0.0], 1.0, 0.3));
        w.bodies.push(body([0.0, 4.0], [0.0, 2.0], 1.0, 0.3));
        let p0 = momentum(&w);
        let m0 = total_mass(&w);
        w.step(1e-4, &Input::default());
        assert_eq!(w.bodies.len(), 2);
        assert_eq!(w.stats.merges_total, 3);
        assert!((total_mass(&w) - m0).abs() < 1e-4);
        let p1 = momentum(&w);
        assert!((p1[0] - p0[0]).abs() < 1e-3 && (p1[1] - p0[1]).abs() < 1e-3);
        assert!((w.bodies[1].mass - 7.0).abs() < 1e-5);
    }

    #[test]
    fn bodies_stay_in_bounds() {
        let mut w = World::new(500, 11);
        w.spawn_cluster([990.0, -990.0], [3000.0, -3000.0], 50);
        w.spawn_cluster([-990.0, 0.0], [-5000.0, 0.0], 50);
        for i in 0..240 {
            let input = Input {
                cursor: [900.0, 900.0],
                well: if i % 80 < 40 { 1.0 } else { -1.0 },
            };
            w.step(1.0 / 60.0, &input);
            for b in &w.bodies {
                assert!(b.pos[0].abs() <= WORLD_HALF && b.pos[1].abs() <= WORLD_HALF);
                assert!(b.vel[0].is_finite() && b.vel[1].is_finite());
            }
        }
        // Huge dt is clamped rather than exploding.
        w.step(10.0, &Input::default());
        assert!(w.bodies.iter().all(|b| b.pos[0].abs() <= WORLD_HALF));
    }

    #[test]
    fn spawned_cluster_stays_a_cluster() {
        for seed in 1..4 {
            let mut w = World::new(0, seed);
            w.spawn_cluster([0.0, 0.0], [0.0, 0.0], 200);
            w.step(1.0 / 60.0, &Input::default());
            assert!(w.bodies.len() > 150, "seed {seed}: only {} bodies left", w.bodies.len());
        }
    }

    #[test]
    fn attractors_ordering() {
        let mut w = World::new(300, 5);
        w.bodies[10].mass = 500.0;
        w.bodies[20].mass = 700.0;
        let mut out = [Attractor::default(); MAX_ATTRACTORS];

        let input = Input { cursor: [5.0, 6.0], well: -1.0 };
        let n = w.attractors(&input, &mut out);
        assert_eq!(n, MAX_ATTRACTORS);
        assert_eq!(out[0].pos, [5.0, 6.0]);
        assert_eq!(out[0].strength, -WELL_STRENGTH);
        assert_eq!(out[1].strength, G * CENTRAL_MASS);
        assert_eq!(out[2].strength, G * 700.0);
        assert_eq!(out[3].strength, G * 500.0);
        for a in &out[1..n] {
            assert!(a.softening >= SOFTENING);
        }
        for pair in out[1..n].windows(2) {
            assert!(pair[0].strength >= pair[1].strength);
        }
        // The cut-off is correct: nothing left out is heavier than the lightest exported.
        let min_exported = out[n - 1].strength / G;
        let heavier = w.bodies.iter().filter(|b| b.mass > min_exported).count();
        assert!(heavier < n);

        let n = w.attractors(&Input::default(), &mut out);
        assert_eq!(n, MAX_ATTRACTORS);
        assert_eq!(out[0].strength, G * CENTRAL_MASS);

        // Fewer bodies than slots.
        let small = World::new(3, 1);
        assert_eq!(small.attractors(&input, &mut out), 4);
    }

    #[test]
    fn subhalo_setup() {
        for (n, seed) in [(2000, 1), (6000, 2), (20000, 3)] {
            let mut w = World::with_scenario(Scenario::SubhaloMerger, n, seed);
            assert_eq!(w.scenario(), Scenario::SubhaloMerger);
            let len = w.bodies.len();
            assert!(len.abs_diff(n) * 20 <= n, "n {n}: built {len}");
            assert_eq!(w.stats.largest_mass, scenario::HOST_CORE_MASS);
            // Host core + 6..=10 satellite cores are the only heavy bodies.
            let heavy = w.bodies.iter().filter(|b| b.mass >= scenario::COLLISION_MIN_MASS).count();
            assert!((7..=11).contains(&heavy), "n {n}: {heavy} heavy bodies");
            let p = momentum(&w);
            assert!(p[0].hypot(p[1]) < 1e-3 * momentum_scale(&w), "n {n}: net momentum {p:?}");
            for b in &w.bodies {
                assert!(b.pos[0].abs() <= WORLD_HALF && b.pos[1].abs() <= WORLD_HALF);
                assert!(b.vel[0].is_finite() && b.vel[1].is_finite());
            }
            w.step(1.0 / 60.0, &Input::default());
            assert!(w.stats.merges_total * 100 < n as u64, "n {n}: merges {}", w.stats.merges_total);
            assert_eq!(w.stats.tree, n >= gravity::TREE_ON);
        }
        // reset keeps the scenario.
        let mut w = World::with_scenario(Scenario::SubhaloMerger, 3000, 1);
        w.reset(2500, 9);
        assert_eq!(w.scenario(), Scenario::SubhaloMerger);
        assert_eq!(w.bodies.len(), 2500);
        assert_eq!(World::with_scenario(Scenario::SubhaloMerger, 0, 1).bodies.len(), 0);
        for n in [1, 2, 10, 50] {
            assert_eq!(World::with_scenario(Scenario::SubhaloMerger, n, 1).bodies.len(), n);
        }
        assert_eq!(Scenario::from_name("SUBHALO"), Some(Scenario::SubhaloMerger));
    }

    #[test]
    fn force_method_switch_keeps_leapfrog_consistent() {
        let mut w = World::with_scenario(Scenario::SubhaloMerger, gravity::TREE_ON + 200, 4);
        let input = Input::default();
        w.step(1.0 / 60.0, &input);
        assert!(w.stats.tree);
        // Drop below the hysteresis band: the next step must switch to direct summation and
        // recompute the cached accelerations with it before the first kick.
        w.bodies.truncate(gravity::TREE_OFF - 100);
        // Same length as `acc` would hide the change from the length check; force it.
        w.acc.truncate(w.bodies.len());
        let m0 = total_mass(&w);
        let merges = w.stats.merges_total;
        w.step(1e-4, &input); // one tiny substep: no merges, so `acc` is untouched by them
        assert!(!w.stats.tree);
        assert_eq!(w.stats.merges_total, merges);
        let mut direct = Vec::new();
        gravity::GravityScratch::default().direct(&w.bodies, &mut direct);
        assert_eq!(bytemuck::cast_slice::<_, u8>(&w.acc), bytemuck::cast_slice::<_, u8>(&direct));
        // Inside the band nothing flips back.
        w.spawn_cluster([0.0, 600.0], [0.0, 0.0], 200);
        w.step(1.0 / 60.0, &input);
        assert!(!w.stats.tree);
        // Above TREE_ON it switches to the tree again.
        w.spawn_cluster([300.0, -600.0], [0.0, 0.0], gravity::TREE_ON);
        w.step(1.0 / 60.0, &input);
        assert!(w.stats.tree);
        for _ in 0..10 {
            w.step(1.0 / 60.0, &input);
        }
        assert!(((total_mass(&w) - m0) / m0) > -1e-5);
        assert!(w.bodies.iter().all(|b| b.vel[0].is_finite() && b.pos[0].abs() <= WORLD_HALF));
    }

    #[test]
    fn tree_momentum_and_determinism() {
        let run = || {
            let mut w = World::with_scenario(Scenario::SubhaloMerger, 8000, 42);
            let p0 = momentum(&w);
            let scale = momentum_scale(&w);
            for _ in 0..20 {
                w.step(1.0 / 60.0, &Input::default());
            }
            assert!(w.stats.tree);
            let p1 = momentum(&w);
            let dp = (p1[0] - p0[0]).hypot(p1[1] - p0[1]);
            // Barnes-Hut forces are not exactly antisymmetric, so allow a little drift.
            assert!(dp < 5e-3 * scale, "momentum drift {dp} vs scale {scale}");
            w.bodies
        };
        let (a, b) = (run(), run());
        assert_eq!(bytemuck::cast_slice::<_, u8>(&a), bytemuck::cast_slice::<_, u8>(&b));
    }

    /// Total energy (f64): kinetic + Plummer-softened pair potential (matches the force).
    fn energy(bodies: &[Body]) -> f64 {
        let eps2 = (SOFTENING as f64).powi(2);
        let mut e = 0.0f64;
        for (i, a) in bodies.iter().enumerate() {
            let v2 = (a.vel[0] as f64).powi(2) + (a.vel[1] as f64).powi(2);
            e += 0.5 * a.mass as f64 * v2;
            for b in &bodies[i + 1..] {
                let dx = (a.pos[0] - b.pos[0]) as f64;
                let dy = (a.pos[1] - b.pos[1]) as f64;
                e -= G as f64 * a.mass as f64 * b.mass as f64 / (dx * dx + dy * dy + eps2).sqrt();
            }
        }
        e
    }

    /// Runs `frames` frames of 1/60 s with the integrator forced, checking the energy every
    /// `every` frames. Returns (max |dE/E| over the run, final |dE/E|, mean force
    /// evaluations per body per frame, most bins occupied in one frame).
    fn energy_run(mut w: World, block: bool, frames: usize, every: usize) -> (f64, f64, f64, u32) {
        w.force_integrator(Some(block));
        // Collisionless: point particles still pass within 2e-3 of each other now and
        // then in 2D, and one merge would cascade (the product gets a real radius).
        w.merge_min_mass = f32::INFINITY;
        let e0 = energy(&w.bodies);
        let (mut evals, mut bins, mut max_err) = (0.0f64, 0u32, 0.0f64);
        for f in 1..=frames {
            w.step(1.0 / 60.0, &Input::default());
            evals += w.stats.force_evals_per_body as f64;
            bins = bins.max(w.stats.timestep_bins);
            if f % every == 0 {
                max_err = max_err.max(((energy(&w.bodies) - e0) / e0).abs());
            }
        }
        assert_eq!(w.stats.merges_total, 0);
        let fin = ((energy(&w.bodies) - e0) / e0).abs();
        (max_err.max(fin), fin, evals / frames as f64, bins)
    }

    /// Bodies with a tiny radius never touch, so energy tests are merge-free.
    fn point(pos: [f32; 2], vel: [f32; 2], mass: f32) -> Body {
        Body { pos, vel, mass, radius: 1e-3, hue: 0.0, _pad: 0.0 }
    }

    #[test]
    fn kepler_energy_uniform_vs_block() {
        // Eccentric (e = 0.9) orbit, semi-major axis 100, pericentre 10 (2.5 softening
        // lengths): peak acceleration ~350, period ~31 s. 10 orbits.
        let (m_big, m_small, a, e) = (1000.0f32, 1.0f32, 100.0f32, 0.9f32);
        let gm = G * (m_big + m_small);
        let r_apo = a * (1.0 + e);
        let v_apo = (gm * (1.0 - e) / r_apo).sqrt();
        let build = || {
            let mut w = World::new(0, 1);
            let f = m_small / (m_big + m_small);
            w.bodies.push(point([-r_apo * f, 0.0], [0.0, -v_apo * f], m_big));
            w.bodies.push(point([r_apo * (1.0 - f), 0.0], [0.0, v_apo * (1.0 - f)], m_small));
            w
        };
        let frames = 60 * 314;
        let (du, fu, eu, _) = energy_run(build(), false, frames, 1);
        let (db, fb, eb, bins) = energy_run(build(), true, frames, 1);
        println!(
            "kepler e=0.9, 10 orbits: uniform max|dE/E| {du:.2e} final {fu:.2e} ({eu:.2} evals/body/frame); \
             block max {db:.2e} final {fb:.2e} ({eb:.2} evals/body/frame, {bins} bins)"
        );
        assert!(db < 5e-4 && fb < 5e-4, "block energy error {db} / {fb}");
        assert!(db * 5.0 < du, "block {db} should beat uniform {du} at pericentre");
        assert!(eb < eu);
        assert!(bins >= 3);
    }

    #[test]
    fn plummer_energy_drift() {
        // 2000-body projected-Plummer cloud (scale 60, mass 1000), isotropic velocities
        // scaled to virial equilibrium. 10 s, direct forces (exact), no merges.
        let mut r = rng::SplitMix64::new(77);
        let (n, mass, scale) = (2000usize, 1000.0f32, 60.0f32);
        let mut bodies: Vec<Body> = (0..n)
            .map(|_| {
                let u = r.range(0.0, 0.95);
                let rr = scale * (u / (1.0 - u)).sqrt();
                let th = r.f32() * std::f32::consts::TAU;
                let v = r.in_disc(1.0);
                point([rr * th.cos(), rr * th.sin()], v, mass / n as f32)
            })
            .collect();
        let pot = energy(&bodies) - bodies.iter().map(|b| 0.5 * (b.mass * (b.vel[0].powi(2) + b.vel[1].powi(2))) as f64).sum::<f64>();
        let kin: f64 = bodies.iter().map(|b| 0.5 * (b.mass * (b.vel[0].powi(2) + b.vel[1].powi(2))) as f64).sum();
        let s = ((-pot / 2.0) / kin).sqrt() as f32; // 2K = -W
        for b in &mut bodies {
            b.vel = [b.vel[0] * s, b.vel[1] * s];
        }
        let build = || {
            let mut w = World::new(0, 1);
            w.bodies = bodies.clone();
            w
        };
        let (du, fu, eu, _) = energy_run(build(), false, 600, 20);
        let (db, fb, eb, bins) = energy_run(build(), true, 600, 20);
        println!(
            "plummer n=2000, 10 s: uniform max|dE/E| {du:.2e} final {fu:.2e} ({eu:.2} evals/body/frame); \
             block max {db:.2e} final {fb:.2e} ({eb:.2} evals/body/frame, {bins} bins)"
        );
        assert!(db < 1e-5, "block energy drift {db}");
        assert!(db < 1.5 * du, "block {db} should be about as accurate as uniform {du}");
        assert!(eb < eu, "block should do less work: {eb} vs {eu}");
    }

    #[test]
    fn external_field_sampling() {
        // Linear field a = (0.01 x + 1, -0.02 y + 3): bilinear sampling reproduces it exactly
        // between cell centres and clamps to the edge centres outside them.
        let size = 64;
        let cell = 2.0 * WORLD_HALF / size as f32;
        let centre = |i: usize| -WORLD_HALF + (i as f32 + 0.5) * cell;
        let lin = |x: f32, y: f32| [0.01 * x + 1.0, -0.02 * y + 3.0];
        let mut acc = Vec::new();
        for iy in 0..size {
            for ix in 0..size {
                acc.push(lin(centre(ix), centre(iy)));
            }
        }
        let f = ExternalField { size, acc };
        for p in [[0.0, 0.0], [123.4, -567.8], [-950.0, 960.0], [centre(3), centre(60)]] {
            let s = field::sample(&f, p);
            let want = lin(p[0], p[1]);
            assert!((s[0] - want[0]).abs() < 1e-3 && (s[1] - want[1]).abs() < 1e-3, "{p:?}: {s:?} vs {want:?}");
        }
        let edge = field::sample(&f, [5000.0, -5000.0]);
        let want = lin(centre(size - 1), centre(0));
        assert!((edge[0] - want[0]).abs() < 1e-3 && (edge[1] - want[1]).abs() < 1e-3);
        assert!(field::sample(&f, [f32::NAN, 0.0])[0].is_finite());

        // Constant field: a lone body (no body-body force) feels exactly that acceleration,
        // in both integrators, and the field survives `reset`.
        for block in [false, true] {
            let mut w = World::new(0, 1);
            w.force_integrator(Some(block));
            w.bodies.push(point([10.0, 20.0], [0.0, 0.0], 1.0));
            w.set_external_field(Some(ExternalField { size: 8, acc: vec![[3.0, -1.5]; 64] }));
            for _ in 0..60 {
                w.step(1.0 / 60.0, &Input::default());
            }
            let b = w.bodies[0];
            assert!((b.vel[0] - 3.0).abs() < 1e-4 && (b.vel[1] + 1.5).abs() < 1e-4, "{:?}", b.vel);
            assert!((b.pos[0] - 11.5).abs() < 1e-3 && (b.pos[1] - 19.25).abs() < 1e-3, "{:?}", b.pos);
        }
        let mut w = World::with_scenario(Scenario::SubhaloMerger, 500, 1);
        w.set_external_field(Some(ExternalField { size: 2, acc: vec![[1.0, 0.0]; 4] }));
        w.set_external_field(Some(ExternalField { size: 3, acc: vec![[1.0, 0.0]; 4] })); // invalid
        assert!(w.take_external_field().is_none());
        w.set_external_field(Some(ExternalField { size: 2, acc: vec![[1.0, 0.0]; 4] }));
        w.reset(400, 2);
        assert!(w.take_external_field().is_some());
    }

    #[test]
    fn merges_with_external_field() {
        // Merging stays mass/momentum-exact on top of a field: total momentum changes by
        // exactly M * a * t.
        let mut w = World::new(0, 1);
        w.bodies.push(body([0.0, 0.0], [10.0, 0.0], 2.0, 0.95));
        w.bodies.push(body([2.0, 0.0], [-4.0, 3.0], 1.0, 0.05));
        w.set_external_field(Some(ExternalField { size: 4, acc: vec![[0.0, -5.0]; 16] }));
        let p0 = momentum(&w);
        w.step(0.01, &Input::default());
        assert_eq!(w.bodies.len(), 1);
        let p1 = momentum(&w);
        assert!((p1[0] - p0[0]).abs() < 1e-3);
        assert!((p1[1] - (p0[1] - 3.0 * 5.0 * 0.01)).abs() < 1e-3, "{p1:?}");
    }

    #[test]
    fn block_bins_and_stats() {
        // Large subhalo world: block steps by default, ~1 force evaluation per body per
        // frame (vs 2 with uniform substeps). The mouse well near bodies forces finer bins.
        let mut w = World::with_scenario(Scenario::SubhaloMerger, 12000, 5);
        w.step(1.0 / 60.0, &Input::default());
        w.step(1.0 / 60.0, &Input::default());
        assert!(w.stats.tree);
        assert!(w.stats.force_evals_per_body < 1.5, "{}", w.stats.force_evals_per_body);
        let well = Input { cursor: [0.0, 400.0], well: 1.0 };
        w.step(1.0 / 60.0, &well);
        assert!(w.stats.timestep_bins >= 2, "bins {}", w.stats.timestep_bins);
        assert!(w.stats.force_evals_per_body >= 1.0);
        assert_eq!(w.bins.len(), w.bodies.len());
        // Uniform: exactly 2 evaluations per body per 1/60 s frame, one bin.
        w.force_integrator(Some(false));
        w.step(1.0 / 60.0, &Input::default());
        assert_eq!(w.stats.timestep_bins, 1);
        assert!((w.stats.force_evals_per_body - 2.0).abs() < 0.05);
    }

    #[test]
    fn deterministic() {
        let run = || {
            let mut w = World::new(800, 42);
            for _ in 0..20 {
                w.step(1.0 / 60.0, &Input { cursor: [100.0, 0.0], well: 1.0 });
            }
            w.bodies
        };
        let (a, b) = (run(), run());
        assert_eq!(bytemuck::cast_slice::<_, u8>(&a), bytemuck::cast_slice::<_, u8>(&b));
    }
}
