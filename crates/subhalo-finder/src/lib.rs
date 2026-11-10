//! Subhalo finder (2D SUBFIND) and merger-tree builder for 2D cosmological dark-matter
//! snapshots (`GSNAP001`).
//!
//! Pipeline per snapshot ([`find_catalog`]): periodic FoF hosts (b = 0.2 x mean spacing)
//! -> per group, in parallel: SPH densities, SUBFIND candidates, iterative unbinding
//! ([`subfind`]) -> subhalo properties, host R_200m-analogue radii, particle -> subhalo
//! and particle -> group maps. [`merger::Linker`] links consecutive catalogs through the
//! most-bound tracer particles and builds branches; [`output`] writes the CSV contract.
//!
//! Physics and units are derived in [`units`].

pub mod fof;
pub mod kdtree;
pub mod merger;
pub mod output;
pub mod run;
pub mod snapshot;
pub mod subfind;
pub mod units;

use rayon::prelude::*;
use std::time::Instant;
use units::Cosmo;

pub const NONE: u32 = u32::MAX;

/// Finder parameters (defaults = the documented production choices).
#[derive(Clone, Debug)]
pub struct Params {
    /// FoF linking length in units of the mean interparticle spacing L/sqrt(N).
    pub link_b: f64,
    /// Minimum number of bound particles of a subhalo (and of a FoF group).
    pub n_min: usize,
    /// Neighbours for the SPH density.
    pub k_dens: usize,
    /// Neighbours searched for denser particles in the candidate step.
    pub n_ngb: usize,
    /// Barnes-Hut opening angle for the unbinding potential.
    pub theta: f64,
    /// Reference radius for the 2D binding energy, in units of the candidate extent
    /// (default e: matches the 3D escape budget at the edge, see `units`).
    pub escape_factor: f64,
    /// At most this fraction of a candidate is removed per unbinding iteration.
    pub max_unbind_frac: f64,
    /// Most-bound particles kept per subhalo for tracking.
    pub n_tracers: usize,
    /// R_host encloses this many times the mean (background) surface density.
    pub host_overdensity: f64,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            link_b: 0.2,
            n_min: 20,
            k_dens: 32,
            n_ngb: 20,
            theta: 0.5,
            escape_factor: std::f64::consts::E,
            max_unbind_frac: 0.25,
            n_tracers: 20,
            host_overdensity: 200.0,
        }
    }
}

/// One subhalo of a snapshot catalog.
#[derive(Clone, Debug)]
pub struct SubRecord {
    /// FoF group index (groups sorted by size, descending).
    pub host: u32,
    pub is_central: bool,
    pub n_bound: u32,
    /// Position of the minimum-potential member (SUBFIND's subhalo position; comoving, in
    /// [0, L)). Radii (rmax, r_half) are measured from it.
    pub pos: [f64; 2],
    /// Centre of mass (periodic-safe, in [0, L)).
    pub com: [f64; 2],
    /// Mean peculiar velocity.
    pub vel: [f64; 2],
    /// Most-bound (lowest-energy) particle position.
    pub pos_most_bound: [f64; 2],
    pub vmax: f64,
    pub rmax: f64,
    pub sigma_v: f64,
    pub r_half: f64,
    /// IDs of the most-bound particles, most bound first (at most `n_tracers`).
    pub tracers: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct HostRecord {
    pub n_fof: u32,
    /// Index of the central subhalo in `Catalog::subs`, or `NONE`.
    pub central: u32,
    /// Centre: central's minimum-potential particle, else the FoF centre of mass.
    pub pos: [f64; 2],
    /// Largest radius around `pos` whose enclosed mean surface density is at least
    /// `host_overdensity` x the mean (FoF members only).
    pub r_host: f64,
    /// Same at 50x and 100x the mean (diagnostics).
    pub r_alt: [f64; 2],
}

#[derive(Clone, Debug, Default)]
pub struct Timings {
    pub fof_s: f64,
    pub subfind_s: f64,
    pub maps_s: f64,
}

pub struct Catalog {
    pub a: f64,
    pub subs: Vec<SubRecord>,
    pub hosts: Vec<HostRecord>,
    /// Particle -> subhalo index (`NONE` if unbound / not in a group).
    pub sub_of: Vec<u32>,
    /// Particle -> FoF group index (`NONE` if not in a group of >= n_min).
    pub group_of: Vec<u32>,
    pub n_grouped: u64,
    pub timings: Timings,
}

#[inline]
fn wrap(x: f64, l: f64) -> f64 {
    let y = x.rem_euclid(l);
    if y >= l { 0.0 } else { y }
}

struct GroupOut {
    subs: Vec<(SubRecord, Vec<u32>)>,
    host: HostRecord,
}

fn process_group(members: &[u32], pos: &[[f32; 2]], vel: &[[f32; 2]], cosmo: &Cosmo, params: &Params, g: u32) -> GroupOut {
    let l = cosmo.box_size;
    let half = 0.5 * l;
    let r0 = pos[members[0] as usize];
    let (rx, ry) = (r0[0] as f64, r0[1] as f64);
    let unwrap = |d: f64| {
        if d > half {
            d - l
        } else if d < -half {
            d + l
        } else {
            d
        }
    };
    let lpos: Vec<[f32; 2]> = members
        .iter()
        .map(|&i| {
            let p = pos[i as usize];
            [unwrap(p[0] as f64 - rx) as f32, unwrap(p[1] as f64 - ry) as f32]
        })
        .collect();
    let vf = cosmo.vel_factor as f32;
    let lvel: Vec<[f32; 2]> = members.iter().map(|&i| [vel[i as usize][0] * vf, vel[i as usize][1] * vf]).collect();
    let local = subfind::find_in_group(&lpos, &lvel, cosmo, params);

    let mut subs = Vec::with_capacity(local.len());
    for (k, s) in local.iter().enumerate() {
        let n = s.members.len();
        let mb = lpos[s.centre as usize];
        let (mut cx, mut cy, mut vx, mut vy) = (0.0f64, 0.0, 0.0, 0.0);
        for &i in &s.members {
            let (p, v) = (lpos[i as usize], lvel[i as usize]);
            cx += p[0] as f64;
            cy += p[1] as f64;
            vx += v[0] as f64;
            vy += v[1] as f64;
        }
        let nf = n as f64;
        let (vx, vy) = (vx / nf, vy / nf);
        let mut var = 0.0;
        let mut radii: Vec<f64> = Vec::with_capacity(n);
        for &i in &s.members {
            let (p, v) = (lpos[i as usize], lvel[i as usize]);
            var += (v[0] as f64 - vx).powi(2) + (v[1] as f64 - vy).powi(2);
            radii.push(((p[0] - mb[0]) as f64).hypot((p[1] - mb[1]) as f64));
        }
        radii.sort_unstable_by(f64::total_cmp);
        let (mut vmax2, mut rmax) = (0.0f64, 0.0f64);
        for (j, &r) in radii.iter().enumerate() {
            let v2 = cosmo.vc2((j + 1) as f64, r);
            if v2 > vmax2 {
                vmax2 = v2;
                rmax = r;
            }
        }
        let rec = SubRecord {
            host: g,
            is_central: k == 0,
            n_bound: n as u32,
            pos: [wrap(rx + mb[0] as f64, l), wrap(ry + mb[1] as f64, l)],
            com: [wrap(rx + cx / nf, l), wrap(ry + cy / nf, l)],
            vel: [vx, vy],
            pos_most_bound: {
                let q = lpos[s.members[0] as usize];
                [wrap(rx + q[0] as f64, l), wrap(ry + q[1] as f64, l)]
            },
            vmax: vmax2.sqrt(),
            rmax,
            sigma_v: (var / nf / 2.0).sqrt(),
            r_half: radii[(n - 1) / 2],
            tracers: s.members.iter().take(params.n_tracers).map(|&i| members[i as usize]).collect(),
        };
        let global: Vec<u32> = s.members.iter().map(|&i| members[i as usize]).collect();
        subs.push((rec, global));
    }

    // Host centre and R_host (all FoF members around the centre).
    let centre_local: [f64; 2] = match local.first() {
        Some(s) => {
            let p = lpos[s.centre as usize];
            [p[0] as f64, p[1] as f64]
        }
        None => {
            let n = lpos.len() as f64;
            [lpos.iter().map(|p| p[0] as f64).sum::<f64>() / n, lpos.iter().map(|p| p[1] as f64).sum::<f64>() / n]
        }
    };
    let mut d2: Vec<f64> = lpos
        .iter()
        .map(|p| (p[0] as f64 - centre_local[0]).powi(2) + (p[1] as f64 - centre_local[1]).powi(2))
        .collect();
    d2.sort_unstable_by(f64::total_cmp);
    // Largest radius whose enclosed mean surface density is >= Delta * rho_bar (rho_bar = 1
    // in area units): (k+1) m >= Delta pi r_k^2.
    let r_delta = |delta: f64| {
        let mut r = 0.0;
        for (k, &q) in d2.iter().enumerate() {
            if (k + 1) as f64 * cosmo.m_area >= delta * std::f64::consts::PI * q {
                r = q.sqrt();
            }
        }
        r
    };
    let r_host = r_delta(params.host_overdensity);
    let r_alt = [r_delta(50.0), r_delta(100.0)];
    GroupOut {
        subs,
        host: HostRecord {
            n_fof: members.len() as u32,
            central: NONE,
            pos: [wrap(rx + centre_local[0], l), wrap(ry + centre_local[1], l)],
            r_host,
            r_alt,
        },
    }
}

/// Find all subhalos of one snapshot. `vel` is the raw snapshot velocity column (converted
/// with `cosmo.vel_factor`).
pub fn find_catalog(pos: &[[f32; 2]], vel: &[[f32; 2]], cosmo: &Cosmo, params: &Params) -> Result<Catalog, String> {
    let np = pos.len();
    let t0 = Instant::now();
    let link = params.link_b * cosmo.box_size / (cosmo.n_total as f64).sqrt();
    let groups = fof::fof(pos, cosmo.box_size, link, params.n_min)?;
    let fof_s = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    let outs: Vec<GroupOut> = (0..groups.len())
        .into_par_iter()
        .with_max_len(1)
        .map(|g| process_group(groups.group(g), pos, vel, cosmo, params, g as u32))
        .collect();
    let subfind_s = t1.elapsed().as_secs_f64();

    let t2 = Instant::now();
    let mut sub_of = vec![NONE; np];
    let mut group_of = vec![NONE; np];
    let mut subs = Vec::new();
    let mut hosts = Vec::with_capacity(outs.len());
    for (g, out) in outs.into_iter().enumerate() {
        for &i in groups.group(g) {
            group_of[i as usize] = g as u32;
        }
        let mut host = out.host;
        for (k, (rec, global)) in out.subs.into_iter().enumerate() {
            let id = subs.len() as u32;
            if k == 0 {
                host.central = id;
            }
            for i in global {
                sub_of[i as usize] = id;
            }
            subs.push(rec);
        }
        hosts.push(host);
    }
    let n_grouped = groups.members.len() as u64;
    Ok(Catalog {
        a: cosmo.a,
        subs,
        hosts,
        sub_of,
        group_of,
        n_grouped,
        timings: Timings { fof_s, subfind_s, maps_s: t2.elapsed().as_secs_f64() },
    })
}

#[cfg(test)]
mod tests;
