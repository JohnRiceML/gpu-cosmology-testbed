//! Initial conditions for `Scenario::SubhaloMerger`.
//!
//! Layout (masses are fixed, independent of `n`; only the particle mass changes, so the
//! dynamics look the same at 4k and 50k bodies, just at finer resolution):
//! - **Host**: a heavy core body at the origin plus a rotating disc (about 45% of `n`) out
//!   to `HOST_DISC_OUTER`, on near-circular orbits from the enclosed (softened) mass, with
//!   some velocity dispersion so it grows spiral structure instead of fragmenting.
//! - **Satellites** (6..=10): a dense core body (2-8% of the host core mass) plus a cloud
//!   of particles with a projected-Plummer surface density `S ~ (1 + R^2/a^2)^-2`, cut at
//!   an inner hole just outside the core body and truncated inside the tidal (Jacobi)
//!   radius at the starting position. Cloud velocities: isotropic dispersion plus partial
//!   rotation (random spin sense per satellite), with `<v^2> = v_circ^2(R)` from the
//!   enclosed core + cloud mass, so each subhalo starts bound and roughly virialised.
//! - Satellites start at radius 450..900 near apocentre: tangential speed 30-70% of the
//!   local circular speed, a small inward radial component, mixed prograde/retrograde,
//!   spread in phase. Each satellite has its own hue family; the host is gold.
//! - Collisions: in this scenario light particles (mass < `COLLISION_MIN_MASS`) are
//!   collisionless with each other, like stars and dark matter; anything that touches a
//!   heavy body (a core) merges into it. With merge-on-contact for everything, a 2D cloud
//!   or disc collides itself into blobs within seconds, and faster the larger `n` is.
//! - Light bodies overlapping a heavy one at the start are re-rolled within their own
//!   distribution; net momentum is removed at the end.

use crate::collide::Grid;
use crate::rng::SplitMix64;
use crate::{Body, G, SOFTENING, WORLD_HALF, radius_for_mass};
use std::f32::consts::TAU;

/// In this scenario a merge needs at least one body of at least this mass.
pub(crate) const COLLISION_MIN_MASS: f32 = 20.0;
/// Host core body mass (radius ~137).
pub(crate) const HOST_CORE_MASS: f32 = 3000.0;
/// Total mass of the host disc particles.
const HOST_DISC_MASS: f32 = 800.0;
pub(crate) const HOST_DISC_OUTER: f32 = 700.0;
/// Host disc random velocity (per component) as a fraction of the circular speed.
const HOST_DISC_DISPERSION: f32 = 0.05;
/// Fraction of `n` (core + disc) that belongs to the host.
const HOST_FRACTION: f32 = 0.45;
const HOST_HUE: f32 = 0.11;
/// Satellite core mass as a fraction of the host core mass.
const SAT_CORE_FRAC: (f32, f32) = (0.02, 0.08);
/// Satellite cloud mass as a multiple of its core mass.
const SAT_CLOUD_RATIO: (f32, f32) = (0.25, 0.5);
/// Starting (apocentre) radius of satellite centres.
pub(crate) const SAT_RADIUS: (f32, f32) = (450.0, 900.0);
/// Tangential speed as a fraction of the local circular speed.
const SAT_VT: (f32, f32) = (0.3, 0.7);
/// Plummer scale as a fraction of the starting Jacobi radius.
const CLOUD_SCALE: f32 = 0.15;
/// Cloud truncation radius as a fraction of the starting Jacobi radius.
const CLOUD_TRUNCATION: f32 = 0.7;
/// Cloud velocity dispersion per component, as a fraction of the local circular speed;
/// the rest of `<v^2> = v_circ^2` is ordered rotation.
const CLOUD_DISPERSION: f32 = 0.2;

/// Particle mass relative to the per-group mean, drawn as `0.5 + 1.5 u^2` (mean 1).
fn mass_shape(rng: &mut SplitMix64) -> f32 {
    0.5 + 1.5 * rng.f32().powi(2)
}

struct Satellite {
    centre: [f32; 2],
    vel: [f32; 2],
    core_mass: f32,
    cloud_mass: f32,
    a: f32,
    r_hole: f32,
    r_max: f32,
    hue: f32,
    spin: f32,
}

impl Satellite {
    /// Projected Plummer cumulative mass fraction.
    fn frac(&self, r: f32) -> f32 {
        r * r / (r * r + self.a * self.a)
    }

    fn sample(&self, rng: &mut SplitMix64) -> [f32; 2] {
        let (f0, f1) = (self.frac(self.r_hole), self.frac(self.r_max));
        let u = f0 + (f1 - f0) * rng.f32();
        let r = self.a * (u / (1.0 - u)).sqrt();
        let th = rng.f32() * TAU;
        [self.centre[0] + r * th.cos(), self.centre[1] + r * th.sin()]
    }
}

/// Host disc radial sampling: r = r_min + L * t^0.9 (surface density falling ~1/r).
fn disc_radius(r_min: f32, t: f32) -> f32 {
    r_min + (HOST_DISC_OUTER - r_min) * t.powf(0.9)
}

/// Host mass (core + disc) enclosed within radius `r`, from the sampling distribution.
fn host_enclosed(r_min: f32, r: f32) -> f32 {
    let t = ((r - r_min) / (HOST_DISC_OUTER - r_min)).clamp(0.0, 1.0);
    HOST_CORE_MASS + HOST_DISC_MASS * t.powf(1.0 / 0.9)
}

/// Circular speed around a softened point mass `m` at radius `r`.
fn v_circ(m: f32, r: f32) -> f32 {
    let d2 = r * r + SOFTENING * SOFTENING;
    (G * m * r * r / (d2 * d2.sqrt())).sqrt()
}

fn shuffle<T>(v: &mut [T], rng: &mut SplitMix64) {
    for i in (1..v.len()).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
}

/// Two independent standard normal samples (Box-Muller), capped at 2.5 sigma so no
/// particle starts far above the escape speed.
fn gaussian_pair(rng: &mut SplitMix64) -> (f32, f32) {
    let u1 = rng.f32().max(1e-7);
    let u2 = rng.f32();
    let r = (-2.0 * u1.ln()).sqrt().min(2.5);
    let (s, c) = (TAU * u2).sin_cos();
    (r * c, r * s)
}

/// Group of each generated body: a fixed core body, the host disc, or a satellite cloud.
#[derive(Clone, Copy, PartialEq)]
enum Group {
    Fixed,
    Disc,
    Cloud(usize),
}

pub(crate) fn subhalo_merger(n: usize, rng: &mut SplitMix64) -> Vec<Body> {
    if n == 0 {
        return Vec::new();
    }
    let host_r = radius_for_mass(HOST_CORE_MASS);
    let disc_rmin = host_r + 30.0;

    // ---- body budget --------------------------------------------------------------
    let n_host = ((n as f32 * HOST_FRACTION).round() as usize).clamp(1, n);
    let n_sat_total = n - n_host;
    let k_wanted = 6 + (rng.next_u64() % 5) as usize;
    // Each satellite needs a core and a few cloud particles to be worth having.
    let k = k_wanted.min(n_sat_total / 8);
    let n_disc = if k == 0 { n - 1 } else { n_host - 1 };

    // ---- satellites ---------------------------------------------------------------
    let mut radii: Vec<f32> = (0..k)
        .map(|i| {
            let t = (i as f32 + rng.range(0.15, 0.85)) / k as f32;
            SAT_RADIUS.0 + (SAT_RADIUS.1 - SAT_RADIUS.0) * t
        })
        .collect();
    shuffle(&mut radii, rng);
    let mut hues: Vec<f32> = (0..k).map(|i| 0.27 + 0.70 * (i as f32 + 0.5) / k as f32).collect();
    shuffle(&mut hues, rng);
    // Mix of prograde (+1, same sense as the host disc) and retrograde (-1) orbits.
    let n_retro = if k == 0 { 0 } else { ((k as f32 * 0.4).round() as usize).max(2).min(k) };
    let mut senses: Vec<f32> = (0..k).map(|i| if i < n_retro { -1.0 } else { 1.0 }).collect();
    shuffle(&mut senses, rng);

    let phase0 = rng.f32() * TAU;
    let mut sats: Vec<Satellite> = (0..k)
        .map(|i| {
            let phi = phase0 + TAU * (i as f32 + rng.range(-0.2, 0.2)) / k as f32;
            let r = radii[i];
            let (s, c) = phi.sin_cos();
            let core_mass = HOST_CORE_MASS * rng.range(SAT_CORE_FRAC.0, SAT_CORE_FRAC.1);
            let cloud_mass = core_mass * rng.range(SAT_CLOUD_RATIO.0, SAT_CLOUD_RATIO.1);
            let m_enc = host_enclosed(disc_rmin, r);
            let jacobi = r * ((core_mass + cloud_mass) / (3.0 * m_enc)).cbrt();
            let vc = v_circ(m_enc, r);
            let vt = vc * rng.range(SAT_VT.0, SAT_VT.1) * senses[i];
            let vr = -vc * rng.range(0.0, 0.15);
            let r_hole = 1.5 * radius_for_mass(core_mass) + 4.0;
            Satellite {
                centre: [r * c, r * s],
                vel: [-vt * s + vr * c, vt * c + vr * s],
                core_mass,
                cloud_mass,
                a: (CLOUD_SCALE * jacobi).max(1.2 * r_hole),
                r_hole,
                r_max: CLOUD_TRUNCATION * jacobi,
                hue: hues[i],
                spin: if rng.f32() < 0.5 { 1.0 } else { -1.0 },
            }
        })
        .collect();
    // Keep each cloud inside the walls and clear of its neighbours.
    for i in 0..k {
        let c = sats[i].centre;
        let wall = WORLD_HALF - 15.0 - c[0].abs().max(c[1].abs());
        let neighbour = (0..k)
            .filter(|&j| j != i)
            .map(|j| (sats[j].centre[0] - c[0]).hypot(sats[j].centre[1] - c[1]))
            .fold(f32::INFINITY, f32::min);
        let s = &mut sats[i];
        s.r_max = s.r_max.min(wall).min(0.48 * neighbour).max(s.r_hole + 10.0);
    }

    // Cloud particle counts proportional to cloud mass (largest remainder).
    let n_clouds = n - 1 - n_disc - k;
    let total_cloud: f32 = sats.iter().map(|s| s.cloud_mass).sum();
    let mut counts: Vec<usize> = Vec::with_capacity(k);
    let mut rems: Vec<(f32, usize)> = Vec::with_capacity(k);
    for (i, s) in sats.iter().enumerate() {
        let exact = n_clouds as f32 * s.cloud_mass / total_cloud;
        counts.push(exact.floor() as usize);
        rems.push((exact - exact.floor(), i));
    }
    let short = n_clouds.saturating_sub(counts.iter().sum::<usize>());
    rems.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    for &(_, i) in rems.iter().take(short) {
        counts[i] += 1;
    }

    // ---- positions ----------------------------------------------------------------
    let mut bodies: Vec<Body> = Vec::with_capacity(n);
    let mut groups: Vec<Group> = Vec::with_capacity(n);
    bodies.push(Body {
        mass: HOST_CORE_MASS,
        radius: host_r,
        hue: HOST_HUE,
        ..Body::default()
    });
    groups.push(Group::Fixed);

    let disc_m = HOST_DISC_MASS / n_disc.max(1) as f32;
    let sample_disc = |rng: &mut SplitMix64| {
        let r = disc_radius(disc_rmin, rng.f32());
        let th = rng.f32() * TAU;
        [r * th.cos(), r * th.sin()]
    };
    for _ in 0..n_disc {
        let m = disc_m * mass_shape(rng);
        let pos = sample_disc(rng);
        let t = ((pos[0].hypot(pos[1]) - disc_rmin) / (HOST_DISC_OUTER - disc_rmin)).clamp(0.0, 1.0);
        bodies.push(Body {
            pos,
            mass: m,
            radius: radius_for_mass(m),
            hue: (HOST_HUE - 0.03 + 0.06 * t + rng.range(-0.02, 0.02)).rem_euclid(1.0),
            ..Body::default()
        });
        groups.push(Group::Disc);
    }
    let mut core_idx = Vec::with_capacity(k);
    for (i, s) in sats.iter().enumerate() {
        core_idx.push(bodies.len());
        bodies.push(Body {
            pos: s.centre,
            mass: s.core_mass,
            radius: radius_for_mass(s.core_mass),
            hue: s.hue,
            ..Body::default()
        });
        groups.push(Group::Fixed);
        let pm = s.cloud_mass / counts[i].max(1) as f32;
        for _ in 0..counts[i] {
            let m = pm * mass_shape(rng);
            bodies.push(Body {
                pos: s.sample(rng),
                mass: m,
                radius: radius_for_mass(m),
                hue: (s.hue + rng.range(-0.035, 0.035)).rem_euclid(1.0),
                ..Body::default()
            });
            groups.push(Group::Cloud(i));
        }
    }
    debug_assert_eq!(bodies.len(), n);

    // ---- de-overlap: re-roll positions within each body's own distribution ---------
    // Only pairs that would actually merge (one side heavy) matter.
    let mut grid = Grid::default();
    for _ in 0..24 {
        let pairs = grid.find_pairs_heavy(&bodies, COLLISION_MIN_MASS);
        if pairs.is_empty() {
            break;
        }
        for (i, j) in pairs {
            let victim = (if groups[j as usize] != Group::Fixed { j } else { i }) as usize;
            bodies[victim].pos = match groups[victim] {
                Group::Fixed => continue,
                Group::Disc => sample_disc(rng),
                Group::Cloud(s) => sats[s].sample(rng),
            };
        }
    }

    // ---- velocities -----------------------------------------------------------------
    // Host disc: circular speed from the enclosed host mass (sorted cumulative sum).
    let mut order: Vec<(f32, usize)> = (0..bodies.len())
        .filter(|&i| groups[i] == Group::Disc)
        .map(|i| (bodies[i].pos[0].hypot(bodies[i].pos[1]), i))
        .collect();
    order.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut enclosed = HOST_CORE_MASS;
    for &(r, i) in &order {
        let vc = v_circ(enclosed, r);
        let (gx, gy) = gaussian_pair(rng);
        let v = vc + vc * HOST_DISC_DISPERSION * gx;
        let vr = vc * HOST_DISC_DISPERSION * gy;
        let b = &mut bodies[i];
        let (s, c) = (b.pos[1] / r, b.pos[0] / r);
        b.vel = [-v * s + vr * c, v * c + vr * s];
        enclosed += b.mass;
    }
    // Satellite clouds: rotation + dispersion from the enclosed core + cloud mass.
    let rot_frac = (1.0 - 2.0 * CLOUD_DISPERSION * CLOUD_DISPERSION).sqrt();
    for (si, s) in sats.iter().enumerate() {
        let mut order: Vec<(f32, usize)> = (0..bodies.len())
            .filter(|&i| groups[i] == Group::Cloud(si))
            .map(|i| {
                let p = bodies[i].pos;
                ((p[0] - s.centre[0]).hypot(p[1] - s.centre[1]), i)
            })
            .collect();
        order.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut enclosed = s.core_mass;
        for &(r, i) in &order {
            let vc = v_circ(enclosed, r);
            let b = &mut bodies[i];
            let (sn, cs) = ((b.pos[1] - s.centre[1]) / r, (b.pos[0] - s.centre[0]) / r);
            let vphi = vc * rot_frac * s.spin;
            let sig = vc * CLOUD_DISPERSION;
            let (gx, gy) = gaussian_pair(rng);
            b.vel = [
                s.vel[0] - vphi * sn + sig * gx,
                s.vel[1] + vphi * cs + sig * gy,
            ];
            enclosed += b.mass;
        }
    }
    for (s, &i) in sats.iter().zip(&core_idx) {
        bodies[i].vel = s.vel;
    }

    // ---- zero net momentum, stay inside the world -------------------------------------
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
        b.pos[0] = b.pos[0].clamp(-WORLD_HALF, WORLD_HALF);
        b.pos[1] = b.pos[1].clamp(-WORLD_HALF, WORLD_HALF);
    }
    bodies
}
