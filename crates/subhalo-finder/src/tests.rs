//! Validation on synthetic data. Box L = 200 with N = 40000 particles in total, so the
//! mean density is 1 particle per unit area (m_area = 1), the FoF linking length is 0.2,
//! and C = 3/2, a = 1. Clumps are projected-Plummer discs `S ~ (1 + r^2/s^2)^-2`.

use super::*;
use crate::merger::{LinkKind, Linker, branches};

const L: f64 = 200.0;
const N: usize = 40_000;
/// Clump temperature in units of the virial dispersion (1 = virialised).
const TEMP: f64 = 1.0;

struct Rng(u64);

impl Rng {
    fn f64(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f64 {
        let u = self.f64().max(1e-300);
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * self.f64()).cos()
    }
}

fn cosmo() -> Cosmo {
    Cosmo {
        a: 1.0,
        box_size: L,
        n_total: N as u64,
        m_area: L * L / N as f64,
        particle_mass: 1.0,
        c_poisson: 1.5,
        softening: 0.01,
        vel_factor: 1.0,
    }
}

/// Per-component velocity dispersion of a virialised 2D log-gravity clump of n particles
/// (2K = G2 m^2 N^2 / 2  =>  sigma^2 = g_log N / 4), times `temp`.
fn sigma(n: usize, temp: f64) -> f64 {
    (cosmo().g_log() * n as f64 / 4.0 * temp).sqrt()
}

/// Append a truncated Plummer clump; returns the index range.
#[allow(clippy::too_many_arguments)]
fn clump(
    pos: &mut Vec<[f32; 2]>,
    vel: &mut Vec<[f32; 2]>,
    rng: &mut Rng,
    n: usize,
    centre: [f64; 2],
    scale: f64,
    trunc: f64,
    bulk: [f64; 2],
    sig: f64,
) -> std::ops::Range<usize> {
    let start = pos.len();
    let ft = trunc * trunc / (trunc * trunc + scale * scale);
    for _ in 0..n {
        let u = rng.f64() * ft;
        let r = scale * (u / (1.0 - u)).sqrt();
        let th = rng.f64() * std::f64::consts::TAU;
        pos.push([
            (centre[0] + r * th.cos()).rem_euclid(L) as f32,
            (centre[1] + r * th.sin()).rem_euclid(L) as f32,
        ]);
        vel.push([(bulk[0] + sig * rng.gauss()) as f32, (bulk[1] + sig * rng.gauss()) as f32]);
    }
    start..pos.len()
}

/// Fill up to N particles with a uniform (mean-density) Poisson background at rest.
fn background(pos: &mut Vec<[f32; 2]>, vel: &mut Vec<[f32; 2]>, rng: &mut Rng) {
    while pos.len() < N {
        pos.push([(rng.f64() * L) as f32, (rng.f64() * L) as f32]);
        vel.push([0.0, 0.0]);
    }
}

fn subs_containing(cat: &Catalog, ids: std::ops::Range<usize>) -> Vec<(&SubRecord, usize)> {
    let mut counts = vec![0usize; cat.subs.len()];
    for i in ids {
        if cat.sub_of[i] != NONE {
            counts[cat.sub_of[i] as usize] += 1;
        }
    }
    let mut v: Vec<(&SubRecord, usize)> =
        counts.iter().enumerate().filter(|(_, c)| **c > 0).map(|(s, &c)| (&cat.subs[s], c)).collect();
    v.sort_by_key(|e| std::cmp::Reverse(e.1));
    v
}

#[test]
fn central_plus_embedded_satellite() {
    let mut rng = Rng(0x1234_5678_9abc);
    let (mut pos, mut vel) = (Vec::new(), Vec::new());
    let hc = [100.0, 100.0];
    let host = clump(&mut pos, &mut vel, &mut rng, 8000, hc, 1.0, 2.5, [0.0, 0.0], sigma(8000, TEMP));
    // Satellite inside the host at r = 1.6, on a roughly circular orbit.
    let vc = (cosmo().vc2(8000.0 * 0.72, 1.6)).sqrt();
    let sat = clump(&mut pos, &mut vel, &mut rng, 800, [101.6, 100.0], 0.1, 0.25, [0.0, vc], sigma(800, TEMP));
    background(&mut pos, &mut vel, &mut rng);
    let cat = find_catalog(&pos, &vel, &cosmo(), &Params::default()).unwrap();
    assert_eq!(cat.hosts.len(), 1, "one FoF group expected, got {}", cat.hosts.len());
    let central = &cat.subs[cat.hosts[0].central as usize];
    assert!(central.is_central);
    let in_host = subs_containing(&cat, host.clone());
    let in_sat = subs_containing(&cat, sat.clone());
    let (sat_rec, _) = in_sat[0];
    assert!(!sat_rec.is_central, "satellite clump must be a satellite");
    assert!(in_host[0].0.is_central);
    let sats: Vec<&SubRecord> = cat.subs.iter().filter(|s| !s.is_central).collect();
    println!(
        "central n_bound {} (8000), satellite n_bound {} (800), {} satellites total; vmax {:.2}/{:.2}, r_half {:.3}/{:.3}",
        central.n_bound,
        sat_rec.n_bound,
        sats.len(),
        central.vmax,
        sat_rec.vmax,
        central.r_half,
        sat_rec.r_half
    );
    assert!((central.n_bound as f64 - 8000.0).abs() <= 0.05 * 8000.0, "central {}", central.n_bound);
    assert!((sat_rec.n_bound as f64 - 800.0).abs() <= 0.05 * 800.0, "satellite {}", sat_rec.n_bound);
    // Positions: most-bound particle near the clump centres.
    assert!((central.pos[0] - 100.0).hypot(central.pos[1] - 100.0) < 0.3);
    assert!((sat_rec.pos[0] - 101.6).hypot(sat_rec.pos[1] - 100.0) < 0.1);
    assert!((sat_rec.vel[1] - vc).abs() < 2.0, "satellite bulk vy {} vs {vc}", sat_rec.vel[1]);
    assert_eq!(sat_rec.tracers.len(), 20);
    assert!(sat_rec.tracers.iter().all(|&t| sat.contains(&(t as usize))));
    assert!(cat.hosts[0].r_host > 0.5);
    // Spurious satellites (noise) must be negligible.
    assert!(sats.len() <= 3, "{} satellites", sats.len());
}

#[test]
fn unbound_streaming_clump_is_removed() {
    // Same dense clump twice: cold (bound) and streaming apart (radial outflow far above
    // escape speed plus a hot dispersion). Only the cold one survives unbinding.
    for (stream, expect) in [(false, 1usize), (true, 0usize)] {
        let mut rng = Rng(99);
        let (mut pos, mut vel) = (Vec::new(), Vec::new());
        let r = clump(&mut pos, &mut vel, &mut rng, 1000, [50.0, 50.0], 0.2, 0.5, [5.0, 0.0], sigma(1000, TEMP));
        if stream {
            for i in r.clone() {
                let dx = pos[i][0] as f64 - 50.0;
                let dy = pos[i][1] as f64 - 50.0;
                let s = 400.0;
                vel[i] = [
                    (5.0 + s * dx + 10.0 * sigma(1000, 1.0) * rng.gauss()) as f32,
                    (s * dy + 10.0 * sigma(1000, 1.0) * rng.gauss()) as f32,
                ];
            }
        }
        background(&mut pos, &mut vel, &mut rng);
        let cat = find_catalog(&pos, &vel, &cosmo(), &Params::default()).unwrap();
        assert_eq!(cat.hosts.len(), 1);
        assert_eq!(cat.subs.len(), expect, "stream={stream}: {} subhalos", cat.subs.len());
        if expect == 1 {
            assert!(cat.subs[0].n_bound > 950, "cold clump bound {}", cat.subs[0].n_bound);
        }
    }
}

#[test]
fn uniform_field_has_no_subhalos() {
    // Perturbed lattice and Poisson field at the mean density, at rest.
    let side = (N as f64).sqrt() as usize;
    let mut rng = Rng(7);
    let lattice: Vec<[f32; 2]> = (0..side * side)
        .map(|k| {
            let (i, j) = ((k % side) as f64, (k / side) as f64);
            let s = L / side as f64;
            [((i + 0.5) * s + 0.1 * s * rng.gauss()).rem_euclid(L) as f32, ((j + 0.5) * s + 0.1 * s * rng.gauss()).rem_euclid(L) as f32]
        })
        .collect();
    let zeros = vec![[0.0f32; 2]; lattice.len()];
    let cat = find_catalog(&lattice, &zeros, &cosmo(), &Params::default()).unwrap();
    assert_eq!(cat.subs.len(), 0);
    let (mut pos, mut vel) = (Vec::new(), Vec::new());
    background(&mut pos, &mut vel, &mut rng);
    let cat = find_catalog(&pos, &vel, &cosmo(), &Params::default()).unwrap();
    assert_eq!(cat.subs.len(), 0, "{} subhalos in a Poisson field", cat.subs.len());
}

/// Particles in a fixed order across snapshots (index = ID): host 0..8000, satellite
/// 8000..8800, isolated clump 8800..9400, then background.
fn tree_snapshot(step: usize) -> (Vec<[f32; 2]>, Vec<[f32; 2]>) {
    let mut rng = Rng(1000 + step as u64);
    let (mut pos, mut vel) = (Vec::new(), Vec::new());
    let hc = [100.0, 100.0];
    clump(&mut pos, &mut vel, &mut rng, 8000, hc, 1.0, 2.5, [0.0, 0.0], sigma(8000, TEMP));
    let vc = (cosmo().vc2(8000.0 * 0.72, 1.6)).sqrt();
    // Satellite: at angle 0, 90 deg, (unresolved: spread through the host core), 180 deg.
    match step {
        0 => drop(clump(&mut pos, &mut vel, &mut rng, 800, [101.6, 100.0], 0.1, 0.25, [0.0, vc], sigma(800, TEMP))),
        1 => drop(clump(&mut pos, &mut vel, &mut rng, 800, [100.0, 101.6], 0.1, 0.25, [-vc, 0.0], sigma(800, TEMP))),
        2 => drop(clump(&mut pos, &mut vel, &mut rng, 800, hc, 0.8, 2.0, [0.0, 0.0], sigma(8000, TEMP))),
        _ => drop(clump(&mut pos, &mut vel, &mut rng, 800, [98.4, 100.0], 0.1, 0.25, [0.0, -vc], sigma(800, TEMP))),
    }
    // Isolated clump drifting in +x; at the last step it has fallen into the host core.
    if step < 3 {
        let c = [150.0 + 5.0 * step as f64, 60.0];
        clump(&mut pos, &mut vel, &mut rng, 600, c, 0.15, 0.4, [5.0, 0.0], sigma(600, TEMP));
    } else {
        clump(&mut pos, &mut vel, &mut rng, 600, hc, 0.8, 2.0, [0.0, 0.0], sigma(8000, TEMP));
    }
    let mut bg = Rng(424242);
    background(&mut pos, &mut vel, &mut bg);
    (pos, vel)
}

#[test]
fn synthetic_merger_tree() {
    let a_list = [0.4, 0.6, 0.8, 1.0];
    let mut linker = Linker::new();
    let mut cats = Vec::new();
    for (s, &a) in a_list.iter().enumerate() {
        let (pos, vel) = tree_snapshot(s);
        let mut c = cosmo();
        c.a = 1.0; // keep the same physics in every snapshot; only the label changes
        let mut cat = find_catalog(&pos, &vel, &c, &Params::default()).unwrap();
        cat.a = a;
        let find = |r: std::ops::Range<usize>| -> Option<u32> {
            let mut counts = std::collections::HashMap::new();
            for i in r {
                if cat.sub_of[i] != NONE {
                    *counts.entry(cat.sub_of[i]).or_insert(0usize) += 1;
                }
            }
            counts.into_iter().max_by_key(|e| (e.1, std::cmp::Reverse(e.0))).filter(|e| e.1 > 300).map(|e| e.0)
        };
        let ids = (find(0..8000), find(8000..8800), find(8800..9400));
        println!("snap {s}: {} subhalos, host {:?} sat {:?} clump {:?}", cat.subs.len(), ids.0, ids.1, ids.2);
        linker.add(&cat);
        cats.push((cat, ids));
    }
    linker.finish();
    let (h, sat, clump_ids) = (
        |s: usize| cats[s].1.0.unwrap(),
        |s: usize| cats[s].1.1,
        |s: usize| cats[s].1.2,
    );
    // Satellite: resolved at 0, 1, 3; dissolved into the host central at 2 (its particles
    // are then part of the central).
    assert!(sat(0).is_some() && sat(1).is_some() && sat(3).is_some());
    assert_eq!(sat(2), Some(h(2)));
    assert_eq!(clump_ids(3), Some(h(3)));
    assert!(cats[2].0.subs.len() == 2 && cats[3].0.subs.len() == 2);
    let d0 = linker.desc[0][sat(0).unwrap() as usize].unwrap();
    assert_eq!((d0.snap, d0.idx, d0.kind), (1, sat(1).unwrap(), LinkKind::Direct));
    let d1 = linker.desc[1][sat(1).unwrap() as usize].unwrap();
    assert_eq!((d1.snap, d1.idx, d1.kind), (3, sat(3).unwrap(), LinkKind::Skip), "skip link expected");
    // Host central links straight through.
    for s in 0..3 {
        let d = linker.desc[s][h(s) as usize].unwrap();
        assert_eq!((d.snap, d.idx), (s + 1, h(s + 1)));
    }
    // Isolated clump: linked 0 -> 1 -> 2, then merges into the host central at 3.
    let c2 = clump_ids(2).unwrap();
    assert_eq!(linker.desc[0][clump_ids(0).unwrap() as usize].unwrap().idx, clump_ids(1).unwrap());
    assert_eq!(linker.desc[1][clump_ids(1).unwrap() as usize].unwrap().idx, c2);
    let dc = linker.desc[2][c2 as usize].unwrap();
    assert_eq!((dc.snap, dc.idx), (3, h(3)));

    let forest = linker.forest();
    assert!(forest.is_mp[0][h(0) as usize] && forest.is_mp[2][h(2) as usize]);
    assert!(!forest.is_mp[2][c2 as usize], "the clump is not the host's main progenitor");
    let br = branches(&linker, &forest);
    let host_branch = br.iter().find(|b| b.end == (3, h(3))).unwrap();
    assert!(host_branch.surviving && host_branch.length == 4 && host_branch.a_infall.is_none());
    let sat_branch = br.iter().find(|b| b.end == (3, sat(3).unwrap())).unwrap();
    assert!(sat_branch.surviving);
    assert_eq!(sat_branch.length, 3);
    assert_eq!(sat_branch.a_infall, Some(0.4));
    assert!(sat_branch.n_peak >= 760, "satellite peak {}", sat_branch.n_peak);
    let clump_branch = br.iter().find(|b| b.end == (2, c2)).unwrap();
    assert!(!clump_branch.surviving);
    assert_eq!(clump_branch.host_at_last, Some(cats[3].0.subs[h(3) as usize].host));
    assert_eq!(clump_branch.last_a, 0.8);
    assert!(clump_branch.a_infall.is_none());
    assert!((clump_branch.pos[0] - 160.0).abs() < 0.5);
}

#[test]
fn softening_from_meta() {
    // short_range.eps (PP force softening) wins over the PM smoothing; without it, the old
    // keys and the 0.75-cell fallback still apply.
    let dir = std::env::temp_dir().join(format!("subfind_meta_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let base = r#""box": 2000.0, "particles": 1000, "mesh": 2048, "snapshots": [{"a": 1.0, "file": "s.bin"}],
        "snapshot_format": "... [f32 vx, f32 vy] peculiar velocities v = a dx/dt = p/a""#;
    let cases = [
        (format!(r#"{{{base}, "pm": {{"smoothing": "S(k) = exp(-(k r_s)^2), r_s = 0.610352"}},
            "short_range": {{"enabled": true, "eps": 0.026042}}}}"#), 0.026042, "short_range.eps"),
        (format!(r#"{{{base}, "pm": {{"smoothing": "S(k) = exp(-(k r_s)^2), r_s = 0.610352"}},
            "short_range": {{"enabled": false, "eps": 0.026042}}}}"#), 0.610352, "pm.smoothing"),
        (format!(r#"{{{base}}}"#), 0.75 * 2000.0 / 2048.0, "default"),
    ];
    for (json, want, src) in cases {
        std::fs::write(dir.join("meta.json"), json).unwrap();
        let m = units::RunMeta::load(&dir, None).unwrap();
        assert!((m.softening - want).abs() < 1e-9, "{src}: {}", m.softening);
        assert!(m.softening_source.contains(src), "{src}: {}", m.softening_source);
        assert_eq!(m.vel, units::VelConvention::Peculiar);
    }
    let _ = std::fs::remove_dir_all(&dir);
}
