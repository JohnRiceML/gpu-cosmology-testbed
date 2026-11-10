//! Validation tests: P(k) recovery on a Zel'dovich run, FoF on lattices and
//! hand-made clumps, and the analytic pieces of the mass function.

use crate::fof::{fof, halo_properties};
use crate::hmf::*;
use crate::meta::Spectrum;
use crate::pk::*;
use crate::snapshot;
use crate::synth::{make_za, za_particles};
use std::f64::consts::PI;

const SPEC: Spectrum = Spectrum { amp: 3.0, n: 0.0, r_c: 10.0 };

#[test]
fn bessel_j1_reference_values() {
    for (x, j) in [
        (0.5, 0.2422684577),
        (1.0, 0.4400505857),
        (3.0, 0.3390589585),
        (5.0, -0.3275791376),
        (10.0, 0.0434727462),
    ] {
        assert!((bessel_j1(x) - j).abs() < 2e-8, "J1({x})");
    }
    // power series J1(x) = sum_m (-1)^m (x/2)^{2m+1} / (m! (m+1)!) on [0, 14]
    for i in 0..=140 {
        let x = 0.1 * i as f64;
        let (mut term, mut sum) = (x / 2.0, x / 2.0);
        for m in 1..80 {
            term *= -(x / 2.0).powi(2) / (m as f64 * (m + 1) as f64);
            sum += term;
        }
        assert!((bessel_j1(x) - sum).abs() < 5e-8, "J1({x}) = {} vs series {sum}", bessel_j1(x));
    }
    assert!((tophat_w(1e-4) - 1.0).abs() < 1e-8);
    assert!((tophat_w(2.0) - 2.0 * 0.5767248078 / 2.0).abs() < 1e-8);
}

#[test]
fn sigma2_matches_analytic_limit() {
    // R -> 0, n = 0:  int A e^{-k^2 Rc^2} k dk / 2pi = A / (4 pi Rc^2)
    let s = sigma2_r(&SPEC, 1e-6, 1e-7, 1.0);
    let exact = SPEC.amp / (4.0 * PI * SPEC.r_c * SPEC.r_c);
    assert!((s / exact - 1.0).abs() < 1e-5, "{s} vs {exact}");
}

#[test]
fn multiplicity_functions_are_normalised() {
    let integ = |f: &dyn Fn(f64) -> f64| {
        let (l0, l1, n) = ((1e-10f64).ln(), 10f64.ln(), 40000);
        let h = (l1 - l0) / n as f64;
        (0..=n)
            .map(|i| {
                let c = if i == 0 || i == n { 1.0 } else if i % 2 == 1 { 4.0 } else { 2.0 };
                c * f((l0 + h * i as f64).exp())
            })
            .sum::<f64>()
            * h
            / 3.0
    };
    // integrated over nu in [1e-10, 10]; missing tails are negligible
    assert!((integ(&f_ps) - 1.0).abs() < 1e-4);
    // p > 0: the nu^{1-2p} small-nu tail below 1e-10 contributes ~6e-5
    assert!((integ(&|nu| f_st(nu, 0.707, 0.3)) - 1.0).abs() < 3e-4);
    assert!((gamma(0.5) - PI.sqrt()).abs() < 1e-12);
    assert!((gamma(5.0) - 24.0).abs() < 1e-9);
}

#[test]
fn tophat_collapse_thresholds() {
    let d3 = tophat_delta_c(3);
    assert!((d3 - 1.6865).abs() < 2e-3, "3D delta_c = {d3}");
    let d2 = tophat_delta_c(2);
    assert!(d2 > 1.0 && d2 < d3, "2D delta_c = {d2}");
}

fn edges_for(box_size: f64, n: usize) -> Vec<f64> {
    log_edges(2.0 * PI / box_size * 0.999, PI * n as f64 / box_size, 30)
}

#[test]
fn zeldovich_pk_recovers_linear_theory() {
    let (n, l, a_init) = (256usize, 2000.0, 0.02);
    let field = make_za(n, l, &SPEC, 7);
    let edges = edges_for(l, n);
    let opt = PkOptions { subtract_shot: false, deconvolve: true };
    // realisation spectrum (cosmic variance free reference)
    let l2 = l * l;
    let real = bin_modes(&field.delta_k, n, l, &edges, |_, _, v| Some((l2 * v.norm_sqr(), 0.0, 0.0, 0.0)));
    let measure = |a: f64| {
        let g = a / a_init;
        let (pos, _) = za_particles(&field, g);
        let counts = cic_counts(&pos, l, n);
        let plin = |k: f64| g * g * SPEC.p0(k);
        measure_pk(&counts, n, l, pos.len(), &edges, &plin, opt)
    };
    // (1) a = 0.1, 0.2 (Delta^2_lin peak 0.04, 0.18: well before shell crossing) against the
    //     ensemble linear theory (a/a_init)^2 P0, within the sampling (cosmic-variance) error
    for a in [0.1, 0.2] {
        let bins = measure(a);
        let (mut chi2, mut nb) = (0.0, 0);
        for b in bins.iter().filter(|b| b.k < 0.04) {
            let z = (b.p / b.p_lin - 1.0) / b.rel_err();
            assert!(z.abs() < 4.0, "a={a} k={} ratio={} z={z}", b.k, b.p / b.p_lin);
            chi2 += z * z;
            nb += 1;
        }
        assert!(nb >= 5);
        assert!(chi2 / (nb as f64) < 3.0, "chi2/bin = {}", chi2 / nb as f64);
    }
    // (2) against this realisation's own linear field (cosmic variance cancels). Residuals are
    //     the realisation-specific <delta1 delta2> cross term, which grows ~ linearly in D, so
    //     this tight check is done at D/D_init = 1 and 2.
    let b1 = measure(0.02);
    let b2 = measure(0.04);
    for (g, bins) in [(1.0, &b1), (2.0, &b2)] {
        for b in bins.iter().filter(|b| b.k < 0.035) {
            let r = real.iter().find(|r| (r.k / b.k - 1.0).abs() < 1e-9).unwrap();
            let rr = b.p / (g * g * r.p);
            assert!((rr - 1.0).abs() < 0.03, "D={g} k={} P/P_realisation={rr}", b.k);
        }
    }
    // (3) growth: P(a=0.04)/P(a=0.02) = (D2/D1)^2 = 4 bin by bin at low k
    for (x, y) in b1.iter().zip(&b2).filter(|(x, _)| x.k < 0.035) {
        let r = y.p / x.p;
        assert!((r / 4.0 - 1.0).abs() < 0.03, "k={} growth ratio {r}", x.k);
    }
}

#[test]
fn shot_noise_subtraction_on_poisson_particles() {
    let (n, l, np) = (128usize, 1000.0, 400_000usize);
    let mut st = 12345u64;
    let mut next = || {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        (st >> 11) as f64 / (1u64 << 53) as f64
    };
    let pos: Vec<[f32; 2]> = (0..np).map(|_| [(next() * l) as f32, (next() * l) as f32]).collect();
    let counts = cic_counts(&pos, l, n);
    let edges = edges_for(l, n);
    let bins = measure_pk(&counts, n, l, np, &edges, &|_| 0.0, PkOptions::default());
    let k_ny = PI * n as f64 / l;
    for b in bins.iter().filter(|b| b.k < 0.8 * k_ny && b.n_modes >= 20) {
        // residual must be consistent with zero relative to the shot level
        let z = b.p / (b.p_shot * b.rel_err());
        assert!(z.abs() < 4.5, "k={} P={} shot={} z={z}", b.k, b.p, b.p_shot);
    }
    let low = bins.iter().find(|b| b.n_modes >= 20).unwrap();
    assert!((low.p_shot / (l * l / np as f64) - 1.0).abs() < 0.01);
}

fn lattice(side: usize, spacing: f32) -> Vec<[f32; 2]> {
    (0..side * side)
        .map(|i| [((i % side) as f32 + 0.5) * spacing, ((i / side) as f32 + 0.5) * spacing])
        .collect()
}

#[test]
fn fof_uniform_lattice_has_no_groups() {
    let side = 400;
    let l = side as f64; // spacing 1 = mean interparticle spacing
    let pos = lattice(side, 1.0);
    let b = 0.2 * l / (pos.len() as f64).sqrt();
    assert!((b - 0.2).abs() < 1e-12);
    assert_eq!(fof(&pos, l, b, 2).unwrap().len(), 0, "no pair may link at b = 0.2");
    assert_eq!(fof(&pos, l, b, 20).unwrap().len(), 0);
    // just above the lattice spacing, everything percolates into one group
    let g = fof(&pos, l, 1.01, 20).unwrap();
    assert_eq!(g.len(), 1);
    assert_eq!(g.size(0), pos.len());
}

/// Brute-force O(N^2) periodic FoF partition (reference).
fn brute_partition(pos: &[[f32; 2]], l: f64, b: f64) -> Vec<Vec<u32>> {
    let n = pos.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn root(p: &mut [usize], mut x: usize) -> usize {
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    }
    let (lf, half, b2) = (l as f32, 0.5 * l as f32, (b * b) as f32);
    for i in 0..n {
        for j in i + 1..n {
            let mut dx = pos[i][0] - pos[j][0];
            let mut dy = pos[i][1] - pos[j][1];
            if dx > half { dx -= lf } else if dx < -half { dx += lf }
            if dy > half { dy -= lf } else if dy < -half { dy += lf }
            if dx * dx + dy * dy < b2 {
                let (ri, rj) = (root(&mut parent, i), root(&mut parent, j));
                if ri != rj {
                    parent[ri.max(rj)] = ri.min(rj);
                }
            }
        }
    }
    let mut groups: std::collections::BTreeMap<usize, Vec<u32>> = Default::default();
    for i in 0..n {
        let r = root(&mut parent, i);
        groups.entry(r).or_default().push(i as u32);
    }
    let mut v: Vec<Vec<u32>> = groups.into_values().collect();
    v.sort();
    v
}

fn fof_partition(pos: &[[f32; 2]], l: f64, b: f64, min: usize) -> Vec<Vec<u32>> {
    let g = fof(pos, l, b, min).unwrap();
    let mut v: Vec<Vec<u32>> = (0..g.len()).map(|i| g.group(i).to_vec()).collect();
    v.sort();
    v
}

#[test]
fn fof_hand_made_clumps_found_exactly() {
    let l = 1000.0f64;
    let b = 1.0;
    let mut pos: Vec<[f32; 2]> = Vec::new();
    let mut truth: Vec<Vec<u32>> = Vec::new();
    let wrap = |v: f64| v.rem_euclid(l) as f32;
    let add_block = |pos: &mut Vec<[f32; 2]>, x0: f64, y0: f64, nx: usize, ny: usize, s: f64| {
        let start = pos.len() as u32;
        for j in 0..ny {
            for i in 0..nx {
                pos.push([wrap(x0 + i as f64 * s), wrap(y0 + j as f64 * s)]);
            }
        }
        (start..pos.len() as u32).collect::<Vec<u32>>()
    };
    // A: dense 30x30 block (many particles per cell)
    truth.push(add_block(&mut pos, 100.0, 100.0, 30, 30, 0.3));
    // B: 10x10 block centred on the (0,0) corner -> wraps in x and y
    truth.push(add_block(&mut pos, -2.25, -2.25, 10, 10, 0.5));
    // C: 50-particle chain at spacing 0.95 b crossing x = L (transitive, periodic)
    truth.push(add_block(&mut pos, 980.0, 500.0, 50, 1, 0.95));
    // D/E: two 5x5 blocks separated by a gap of 1.1 b -> two groups
    truth.push(add_block(&mut pos, 300.0, 300.0, 5, 5, 0.5));
    truth.push(add_block(&mut pos, 300.0 + 2.0 + 1.1, 300.0, 5, 5, 0.5));
    // F: two 5x5 blocks separated by 0.9 b -> one group of 50
    let mut f = add_block(&mut pos, 600.0, 600.0, 5, 5, 0.5);
    f.extend(add_block(&mut pos, 600.0 + 2.0 + 0.9, 600.0, 5, 5, 0.5));
    truth.push(f);
    // G: 19-particle block (below the 20 threshold) and isolated field particles
    let _small = add_block(&mut pos, 800.0, 100.0, 19, 1, 0.5);
    let _field = add_block(&mut pos, 400.0, 700.0, 10, 10, 20.0);
    truth.iter_mut().for_each(|t| t.sort());
    truth.sort();

    let found = fof_partition(&pos, l, b, 20);
    assert_eq!(found, truth);

    // periodic-safe centre of mass of the corner clump is (0,0) mod L
    let g = fof(&pos, l, b, 20).unwrap();
    let vel = vec![[1.0f32, -2.0]; pos.len()];
    let halos = halo_properties(&g, &pos, &vel, l);
    let corner = (0..g.len()).find(|&i| g.size(i) == 100).unwrap();
    let c = halos[corner].com;
    let d = |x: f64| x.min(l - x);
    assert!(d(c[0]) < 1e-3 && d(c[1]) < 1e-3, "corner COM {c:?}");
    assert!(halos[corner].sigma_v.abs() < 1e-6);
    assert!((halos[corner].vmean[1] + 2.0).abs() < 1e-6);
    assert_eq!(halos[0].n, 900);
    let a = halos[0].com;
    assert!((a[0] - (100.0 + 14.5 * 0.3)).abs() < 1e-3 && (a[1] - (100.0 + 14.5 * 0.3)).abs() < 1e-3);
}

#[test]
fn fof_matches_brute_force_on_clustered_points() {
    let l = 200.0f64;
    let b = 0.8;
    let mut st = 99u64;
    let mut u = || {
        st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((st >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut pos = Vec::new();
    for _ in 0..20 {
        let (cx, cy, s) = (u() * l, u() * l, 0.5 + 2.0 * u());
        for _ in 0..150 {
            let r = (-2.0 * u().ln()).sqrt() * s;
            let t = 2.0 * PI * u();
            pos.push([(cx + r * t.cos()).rem_euclid(l) as f32, (cy + r * t.sin()).rem_euclid(l) as f32]);
        }
    }
    for _ in 0..2000 {
        pos.push([(u() * l) as f32, (u() * l) as f32]);
    }
    let reference = brute_partition(&pos, l, b);
    assert_eq!(fof_partition(&pos, l, b, 1), reference);
    let big: Vec<Vec<u32>> = reference.into_iter().filter(|g| g.len() >= 20).collect();
    assert_eq!(fof_partition(&pos, l, b, 20), big);
}

#[test]
fn hmf_bins_conserve_counts() {
    let sizes = vec![5000, 800, 400, 120, 60, 33, 21, 20, 20];
    let bins = hmf_bins(&sizes, 20, 1.0, 0.1);
    assert_eq!(bins.iter().map(|b| b.count).sum::<usize>(), sizes.len());
    assert!(bins.windows(2).all(|w| w[0].m_hi == w[1].m_lo));
}

#[test]
fn snapshot_roundtrip() {
    let dir = std::env::temp_dir().join(format!("cosmo_analysis_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("snap_a0.500.bin");
    let pos = vec![[1.0f32, 2.0], [3.5, 1999.9]];
    let vel = vec![[-1.0f32, 0.25], [7.0, 8.0]];
    snapshot::write_snapshot(&path, 0.5, 2000.0, &pos, &vel).unwrap();
    {
        let s = snapshot::Snapshot::open(&path).unwrap();
        assert_eq!((s.count, s.a, s.box_size), (2, 0.5, 2000.0));
        assert_eq!(s.positions(), &pos[..]);
        assert_eq!(s.velocities(), &vel[..]);
    }
    assert_eq!(snapshot::a_from_name("snap_a0.500.bin"), Some(0.5));
    std::fs::remove_dir_all(&dir).unwrap();
}

/// FoF stress/timing on 33M particles with very dense clumps:
/// `cargo test --release fof_stress -- --ignored --nocapture`
#[test]
#[ignore]
fn fof_stress_33m_dense_clumps() {
    use rayon::prelude::*;
    let (l, np) = (2000.0f64, 33_177_600usize);
    let link = 0.2 * l / (np as f64).sqrt();
    let pos: Vec<[f32; 2]> = (0..np)
        .into_par_iter()
        .map(|i| {
            let mut s = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xA5A5;
            let mut u = || {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 11) as f64 + 0.5) / (1u64 << 53) as f64
            };
            for _ in 0..4 {
                u();
            }
            if i % 2 == 0 {
                [(u() * l) as f32, (u() * l) as f32]
            } else {
                // 2000 blobs; one in ten is ultra-dense (sigma = 0.02, ~1e5 x mean density)
                let blob = (i / 2) % 2000;
                let mut bs = blob as u64 * 7919 + 17;
                let mut bu = || {
                    bs = bs.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    ((bs >> 11) as f64 + 0.5) / (1u64 << 53) as f64
                };
                let (cx, cy) = (bu() * l, bu() * l);
                let sig = if blob % 10 == 0 { 0.02 } else { 0.5 };
                let r = (-2.0 * u().ln()).sqrt() * sig;
                let t = 2.0 * PI * u();
                [(cx + r * t.cos()).rem_euclid(l) as f32, (cy + r * t.sin()).rem_euclid(l) as f32]
            }
        })
        .collect();
    let t = std::time::Instant::now();
    let g = fof(&pos, l, link, 20).unwrap();
    let dt = t.elapsed().as_secs_f64();
    let in_groups: usize = (0..g.len()).map(|i| g.size(i)).sum();
    println!("FoF stress: {np} particles, {} groups, largest {}, {in_groups} in groups, {dt:.2} s", g.len(), g.size(0));
    assert!(dt < 30.0);
}
