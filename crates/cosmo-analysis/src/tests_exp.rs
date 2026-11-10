//! Tests for the CDM vs WDM experiment: WDM theory, suppression-fit recovery, the
//! galaxy-formation threshold model, and recovery of an injected M_hm by the likelihood.

use crate::csvtab::Table;
use crate::galaxy::*;
use crate::wdm::*;
use std::f64::consts::PI;

#[test]
fn transfer_half_mode_and_fragmentation_constants() {
    let (alpha, nu) = (0.37, 1.12);
    let k = k_half_mode(alpha, nu);
    assert!((transfer(k, alpha, nu) - 0.5).abs() < 1e-12);
    assert!((alpha_from_khm(k, nu) - alpha).abs() < 1e-12);
    let sb = 16.78;
    let m = m_half_mode_2d(0.7, sb);
    assert!((k_hm_from_mhm_2d(m, sb) - 0.7).abs() < 1e-12);
    // the 2D fragmentation coefficient is the 3D one written per unit filament cross-section
    assert!((WW_KAPPA * PI * (PI / 2.0).powi(2) - 10.1).abs() < 1e-12);
    assert!((m_lim_2d(1.0, 1.0, 1.0) - 4.0 * 10.1 / (PI * PI)).abs() < 1e-12);
    // k^2 P peak of a Gaussian-cut k^-1 spectrum: argmax k e^{-k^2 Rc^2} = 1/(sqrt2 Rc)
    let rc = 2.0;
    let kp = k_peak_2d(&|k: f64| (-(k * rc).powi(2)).exp() / k, 1e-3, 10.0);
    assert!((kp * rc * 2f64.sqrt() - 1.0).abs() < 3e-3);
}

/// Thinned synthetic peak-mass catalog: CDM masses ~ dN/dlnM ~ M^-0.9, WDM keeps an object if its
/// fixed uniform deviate is below f(M). Returns binned (CDM, WDM) pairs for several M_hm.
fn synthetic_pairs(truth: &Supp, mhms: &[f64], n: usize, seed: u64) -> Vec<BinPair> {
    let mut r = Rng(seed);
    let (lo, hi) = (20.0f64, 1e6f64);
    let objs: Vec<(f64, f64)> = (0..n)
        .map(|_| {
            let u = r.uniform();
            let (a, b) = (lo.powf(-0.9), hi.powf(-0.9));
            ((a - u * (a - b)).powf(-1.0 / 0.9), r.uniform())
        })
        .collect();
    let edges: Vec<f64> = (0..=30).map(|i| 50.0 * 10f64.powf(0.15 * i as f64)).collect();
    let mut out = Vec::new();
    for &mhm in mhms {
        let mut nc = vec![0.0; 30];
        let mut nw = vec![0.0; 30];
        for &(m, u) in &objs {
            if m < edges[0] || m >= edges[30] {
                continue;
            }
            let b = edges.partition_point(|&e| e <= m) - 1;
            nc[b] += 1.0;
            if u < truth.f(m, mhm) {
                nw[b] += 1.0;
            }
        }
        for b in 0..30 {
            if nc[b] > 0.0 {
                out.push(BinPair { m_lo: edges[b], m_hi: edges[b + 1], m_hm: mhm, n_cdm: nc[b], n_wdm: nw[b] });
            }
        }
    }
    out
}

#[test]
fn suppression_fit_recovers_injected_parameters() {
    for truth in [Supp { alpha: 3.0, beta: 1.3, gamma: -0.6 }, LOVELL2020_CEN] {
        let data = synthetic_pairs(&truth, &[200.0, 1000.0, 5000.0], 3_000_000, 11);
        let (fit, _) = fit_supp(&data);
        // the (alpha, beta, gamma) valley is long, so check the curve itself and the parameters loosely
        for x in [0.3, 1.0, 3.0, 10.0, 30.0] {
            let (a, b) = (fit.f(x, 1.0), truth.f(x, 1.0));
            assert!((a / b - 1.0).abs() < 0.05 + 0.02 / b, "f({x}) fit {a} truth {b} ({fit:?})");
        }
        let (m_fit, m_true) = (m_half_suppression(&fit, 1.0), m_half_suppression(&truth, 1.0));
        assert!((m_fit / m_true - 1.0).abs() < 0.03, "M50 {m_fit} vs {m_true}");
        assert!((fit.alpha / truth.alpha - 1.0).abs() < 0.25, "{fit:?} vs {truth:?}");
        assert!((fit.beta / truth.beta - 1.0).abs() < 0.25, "{fit:?} vs {truth:?}");
        assert!((fit.gamma / truth.gamma - 1.0).abs() < 0.25, "{fit:?} vs {truth:?}");
    }
}

#[test]
fn galaxy_threshold_model() {
    assert!((f_gal(10.0, 10.0, 2.0) - 0.5).abs() < 1e-7);
    assert!((f_gal(12.0, 10.0, 2.0) - 0.841_344_7).abs() < 1e-6);
    assert_eq!(f_gal(9.99, 10.0, 0.0), 0.0);
    assert_eq!(f_gal(10.0, 10.0, 0.0), 1.0);
    assert!((erf(0.5) - 0.520_499_877_8).abs() < 2e-7);
    let mut last = 0.0;
    for i in 0..100 {
        let f = f_gal(i as f64 * 0.3, 15.0, 3.0);
        assert!(f >= last);
        last = f;
    }
    // expected LF equals sum of p_galaxy x P(L > cut); stochastic mock agrees with the expectation
    let mut r = Rng(5);
    let sats: Vec<Sat> = (0..20_000)
        .map(|_| {
            let m = 20.0 * r.uniform().powf(-1.0 / 0.9);
            Sat { m_peak: m, v_peak: m.sqrt() * 10f64.powf(0.05 * r.normal()), a_infall: 0.3 + 0.6 * r.uniform(), surviving: r.uniform() < 0.6 }
        })
        .collect();
    let gm = GalModel::new(100.0, 10.0, 0.5);
    let p = FIDUCIAL;
    let lf = gm.expected_lf(&sats, &p);
    let direct: f64 = sats
        .iter()
        .map(|s| gm.p_galaxy(s, &p) * p_above(gm.mean_log_l(s.m_peak), p.sigma_m, gm.lf_edges[0]))
        .sum();
    assert!((lf.iter().sum::<f64>() / direct - 1.0).abs() < 1e-9);
    let mut tot = vec![0.0; gm.nbins()];
    let reps = 40;
    for _ in 0..reps {
        for (t, x) in tot.iter_mut().zip(gm.mock_lf(&sats, &p, &mut r)) {
            *t += x;
        }
    }
    for b in 0..gm.nbins() {
        let mean = tot[b] / reps as f64;
        let err = (lf[b] / reps as f64).sqrt();
        assert!((mean - lf[b]).abs() < 4.0 * err + 0.05, "bin {b}: mock {mean} vs expected {}", lf[b]);
    }
    // higher threshold -> fewer galaxies; L_cut tuning hits the target
    let hi = GalParams { v50: 3.0, ..p };
    assert!(gm.expected_lf(&sats, &hi).iter().sum::<f64>() < lf.iter().sum::<f64>());
    let lc = tune_lcut(100.0, 10.0, &sats, 10, 30.0);
    let n = GalModel::new(100.0, 10.0, lc).expected_lf(&sats, &FIDUCIAL).iter().sum::<f64>() / 10.0;
    assert!((n - 30.0).abs() < 0.01, "{n}");
}

#[test]
fn likelihood_recovers_injected_half_mode_mass() {
    let truth = Supp { alpha: 3.0, beta: 1.3, gamma: -0.6 };
    let mut r = Rng(17);
    // ~40 MW-like hosts worth of satellites
    let sats: Vec<(Sat, f64)> = (0..60_000)
        .map(|_| {
            let m = 20.0 * r.uniform().powf(-1.0 / 0.9);
            let s = Sat { m_peak: m, v_peak: m.sqrt() * 10f64.powf(0.05 * r.normal()), a_infall: 0.3 + 0.6 * r.uniform(), surviving: r.uniform() < 0.6 };
            (s, r.uniform())
        })
        .collect();
    let base: Vec<Sat> = sats.iter().map(|x| x.0).collect();
    // luminosity cut at M_peak = 100 (log L = 0), below the threshold mass (1.5 V_res)^2 = 225, so the
    // faint end is threshold-limited: this is the regime where the threshold prior matters
    let v_res = 10.0;
    let gm = GalModel::new(100.0, v_res, 0.0);
    let tab = LikeTable::build(&gm, &base, &PRIOR_FIDUCIAL, 300, 3);
    let grid: Vec<f64> = (0..=60).map(|i| 10.0 * 10f64.powf(i as f64 * 4.0 / 60.0)).collect();
    let supp = |m: f64, mhm: f64| truth.f(m, mhm);
    for (k, &m_true) in [300.0, 1000.0, 3000.0].iter().enumerate() {
        let observed: Vec<Sat> = sats.iter().filter(|(s, u)| *u < truth.f(s.m_peak, m_true)).map(|x| x.0).collect();
        let obs = gm.mock_lf(&observed, &FIDUCIAL, &mut Rng(100 + k as u64));
        let ll = tab.loglike_curve(&obs, &supp, &grid);
        let ib = (0..grid.len()).max_by(|&a, &b| ll[a].total_cmp(&ll[b])).unwrap();
        let ll_true = tab.loglike_curve(&obs, &supp, &[m_true])[0];
        assert!(ll[ib] - ll_true < 3.0, "M_hm={m_true}: dlnL(truth) = {}", ll[ib] - ll_true);
        assert!((grid[ib] / m_true).log10().abs() < 0.35, "M_hm={m_true}: MLE {}", grid[ib]);
    }
    // CDM mock: the upper limit is finite and well below the largest grid value
    let obs = gm.mock_lf(&base, &FIDUCIAL, &mut Rng(9));
    let ll = tab.loglike_curve(&obs, &supp, &grid);
    let ul = posterior_upper_limit(&grid, &ll, 0.95);
    assert!(ul.is_finite() && ul < 1000.0, "{ul}");
    // wide threshold prior weakens the limit
    let tab_w = LikeTable::build(&gm, &base, &PRIOR_WIDE, 300, 4);
    let ul_w = posterior_upper_limit(&grid, &tab_w.loglike_curve(&obs, &supp, &grid), 0.95);
    assert!(ul_w > ul, "wide {ul_w} vs fiducial {ul}");
}

#[test]
fn catalog_csv_is_header_addressed() {
    let t = Table::parse("status,branch_id,m_peak\nsurviving,3,1.5e2\ndisrupted,4,nan\n").unwrap();
    assert_eq!(t.len, 2);
    assert_eq!(t.s("status").unwrap(), vec!["surviving", "disrupted"]);
    assert_eq!(t.f("branch_id").unwrap(), &[3.0, 4.0]);
    assert!(t.f("m_peak").unwrap()[1].is_nan());
    assert!(t.f("missing").is_err());
}

/// `cargo test --release sigma_lower_limit -- --ignored --nocapture`
#[test]
#[ignore]
fn sigma_lower_limit_check() {
    let spec = crate::meta::Spectrum { amp: 1.550091518e-2, n: -1.5, r_c: 0.170898 };
    let g2 = (1.0 / 0.01826229282444664f64).powi(2);
    let p = |k: f64| g2 * spec.p0(k);
    let r = 183.6555;
    let kny = PI * 6144.0 / 2000.0;
    for klo in [2.0 * PI / 2000.0, 1e-6] {
        println!("k_lo = {klo:.3e}: sigma(R=183.66) = {:.4}", crate::hmf::sigma2_r_with(&p, r, klo, kny).sqrt());
    }
}

#[test]
fn host_pairing_by_particle_ids() {
    use crate::experiment::match_groups;
    const N: u32 = u32::MAX;
    // CDM groups: 0 = ids 0..10, 1 = ids 10..20, 2 = ids 20..30
    let lc: Vec<u32> = (0..40).map(|i| if i < 10 { 0 } else if i < 20 { 1 } else if i < 30 { 2 } else { N }).collect();
    // WDM: group 0 is relabelled 5; group 1 split 4/6 between WDM 1 and 2; CDM group 2 merges into
    // WDM group 3 together with 10 unclustered CDM particles
    let lw: Vec<u32> = (0..40).map(|i| if i < 10 { 5 } else if i < 14 { 1 } else if i < 20 { 2 } else { 3 }).collect();
    let (bw, bc) = match_groups(&lc, &lw, 3, 6);
    assert_eq!(bw[0], Some((5, 10)));
    assert_eq!(bc[5], Some(0));
    // group 1: best partner WDM 2 with 6/10 shared, and WDM 2's best is CDM 1 (mutual)
    assert_eq!(bw[1], Some((2, 6)));
    assert_eq!(bc[2], Some(1));
    // group 2 -> WDM 3 (10 shared); WDM 3's best CDM group is 2 (the 10 unclustered ids don't count)
    assert_eq!(bw[2], Some((3, 10)));
    assert_eq!(bc[3], Some(2));
    assert_eq!(bc[4], None);
}

#[test]
fn m_complete_from_resolution_study() {
    let dir = std::env::temp_dir().join(format!("cosmo_mc_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let csv = "run,M_center,n_per_host\n\
        base,1.2589e2,28.1333\nbase,1.9953e2,21.2667\nbase,3.1623e2,15.0333\nbase,5.0119e2,10.6333\nbase,7.9433e2,6.9000\nbase,5.0119e3,1.8000\n\
        down2,1.2589e2,21.7333\ndown2,1.9953e2,17.0333\ndown2,3.1623e2,14.9667\ndown2,5.0119e2,9.6667\ndown2,7.9433e2,6.2667\ndown2,5.0119e3,1.4333\n";
    std::fs::write(dir.join("completeness.csv"), csv).unwrap();
    let (m, _) = crate::experiment::read_m_complete(&dir).unwrap();
    assert!((m - 316.23 / 2.0).abs() < 0.1, "{m}");
    std::fs::write(dir.join("completeness.csv"), "key,value\nM_complete,220\n").unwrap();
    assert_eq!(crate::experiment::read_m_complete(&dir).unwrap().0, 220.0);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn lagrangian_mapping_to_coarser_lattice() {
    use crate::convergence::coarse_index;
    let (sa, r) = (8usize, 2usize);
    let (l, sb) = (10.0f64, sa / r);
    for j in 0..sa * sa {
        let (qx, qy) = (((j % sa) as f64 + 0.5) * l / sa as f64, ((j / sa) as f64 + 0.5) * l / sa as f64);
        let i = coarse_index(j, sa, r);
        // the coarse particle's Lagrangian cell [i*h, (i+1)*h) contains the fine site
        let h = l / sb as f64;
        let (ix, iy) = (i % sb, i / sb);
        assert!(qx >= ix as f64 * h && qx < (ix + 1) as f64 * h && qy >= iy as f64 * h && qy < (iy + 1) as f64 * h);
    }
    assert_eq!(coarse_index(5, 8, 1), 5);
}
