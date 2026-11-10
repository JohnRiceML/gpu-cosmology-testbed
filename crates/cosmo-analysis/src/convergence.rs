//! `cosmo-analysis convergence <base_exp_dir> <conv_dir>`: numerical-convergence suite.
//!
//! Every variant (one change from base: particle number, PM mesh, softening, timestep, or seed)
//! is compared with the base run at a = 1, in PHYSICAL units (mass = count x particle_mass, box-area
//! units, mean surface density 1), separately for CDM and WDM, plus the WDM/CDM suppression itself.
//! Only snap_a1.000.bin, pk.csv, meta.json and subhalos/ are needed.
//!
//! Host matching base -> variant: FoF (b = 0.2) on both a = 1 snapshots; the variant labels are mapped
//! into the base lattice through the Lagrangian position q (lattice index -> q; a variant particle
//! of a lattice r times coarser covers r x r base cells); mutual best match with shared fraction >= 0.5.
//! A different seed cannot be matched: its own top-n hosts are compared statistically. Without
//! snapshots, or with incommensurate lattices, positional matching is used (flagged).

use crate::experiment::{Model, boot_mean_err, boot_ratio, fof_labels, host_groups, match_groups, mean_rows, pair_hosts};
use crate::meta;
use crate::snapshot::Snapshot;
use crate::wdm::{self, BinPair, Supp};
use rayon::prelude::*;
use serde_json::Value;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct ConvOpts {
    pub n_mw: usize,
    /// completeness in BASE particles (160 -> 17.0 mass units for the base particle mass)
    pub m_complete_base_particles: f64,
    pub host_min_base_particles: f64,
    pub n_boot: usize,
    pub seed: u64,
    pub out: Option<PathBuf>,
    /// beta, gamma held fixed in the single-model M50 fit (None: base experiment summary, else 6.66, -0.135)
    pub beta_gamma: Option<(f64, f64)>,
}

impl Default for ConvOpts {
    fn default() -> Self {
        ConvOpts {
            n_mw: 30,
            m_complete_base_particles: 160.0,
            host_min_base_particles: 1000.0,
            n_boot: 200,
            seed: 7071,
            out: None,
            beta_gamma: None,
        }
    }
}

struct Run {
    cdm: Model,
    wdm: Option<Model>,
    seed: Option<f64>,
}

fn read_seed(dir: &Path) -> Option<f64> {
    let s = std::fs::read_to_string(dir.join("meta.json")).ok()?;
    let v: Value = serde_json::from_str(&s).ok()?;
    v.get("seed").and_then(|x| x.as_f64())
}

fn load_run(name: &str, dir: &Path) -> Result<Run, String> {
    let cdm = Model::load(&dir.join("cdm"))?;
    let wdm_dir = std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.is_dir()
                && p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("wdm"))
                && p.join("subhalos").join("hosts_a1.000.csv").is_file()
        })
        .min();
    let wdm = match wdm_dir {
        Some(d) => Some(Model::load(&d)?),
        None => None,
    };
    let _ = name;
    Ok(Run { seed: read_seed(&dir.join("cdm")), cdm, wdm })
}

fn what_changed(v: &str) -> &'static str {
    match v {
        "np3072" => "3072^2 particles (particle mass x4), same initial field sampled coarser",
        "mesh2048" => "PM mesh 2048 instead of 4096 (PP range doubles)",
        "eps2x" => "softening 0.052 instead of 0.026",
        "eta_half" => "timestep eta 0.0125 instead of 0.025",
        "seed2" => "different random seed (cosmic variance, not convergence)",
        "down2" => "finder on a 1/2 down-sampled particle set (pseudo-variant, finder study)",
        "self" => "identical to base (self-test)",
        "control" => "nothing: base settings rerun with the current code (reproducibility floor)",
        _ => "variant",
    }
}

type Labels = (Vec<u32>, Vec<u32>);
/// (base host, variant host) index pairs
type HostPairs = Vec<(usize, usize)>;

fn snap_labels(m: &Model) -> Option<Labels> {
    let p = m.dir.join("snap_a1.000.bin");
    let s = Snapshot::open(&p).ok()?;
    if s.count as f64 != m.np {
        return None;
    }
    fof_labels(s.positions(), m.box_l, 20).ok()
}

/// Match hosts `ha` of A to hosts of B by shared Lagrangian particles (B lattice r times coarser or equal).
/// Returns per ha entry the matched B host and the shared fraction, or None when IDs cannot be mapped.
fn match_by_lagrangian(a: &Model, la: &Labels, ha: &[usize], b: &Model, lb: &Labels, hb: &[usize]) -> Option<(Vec<Option<usize>>, Vec<f64>)> {
    let (sa, sb) = (a.lattice, b.lattice);
    if sb == 0 || sa % sb != 0 || la.0.len() != sa * sa || lb.0.len() != sb * sb {
        return None;
    }
    let r = sa / sb;
    let mapped: Vec<u32> = if r == 1 {
        lb.0.clone()
    } else {
        (0..sa * sa).into_par_iter().map(|j| lb.0[coarse_index(j, sa, r)]).collect()
    };
    let (best_w, best_c) = match_groups(&la.0, &mapped, la.1.len(), lb.1.len());
    let ga = host_groups(a, ha, &la.0);
    let gb = host_groups(b, hb, &lb.0);
    let host_of: HashMap<u32, usize> = hb.iter().zip(&gb).filter_map(|(&h, g)| g.map(|g| (g, h))).collect();
    let mut out = vec![None; ha.len()];
    let mut frac = vec![f64::NAN; ha.len()];
    for (u, g) in ga.iter().enumerate() {
        let Some(c) = g else { continue };
        let Some((w, n)) = best_w[*c as usize] else {
            frac[u] = 0.0;
            continue;
        };
        frac[u] = n as f64 / la.1[*c as usize] as f64;
        if frac[u] >= 0.5 && best_c[w as usize] == Some(*c) {
            out[u] = host_of.get(&w).copied();
        }
    }
    Some((out, frac))
}

/// Index of the particle of a lattice r times coarser (side sa/r) whose Lagrangian cell contains
/// fine lattice site j (side sa). Sites are cell-centred, q = (i + 1/2) L / side, so coarse particle i
/// covers fine sites r i .. r i + r - 1 in each dimension.
pub fn coarse_index(j: usize, sa: usize, r: usize) -> usize {
    let sb = sa / r;
    (j / sa / r) * sb + (j % sa) / r
}

/// Hosts with at least `min_phys` mass units (FoF mass).
fn big_hosts(m: &Model, min_phys: f64) -> Vec<usize> {
    (0..m.cat.hosts.len()).filter(|&h| m.cat.hosts[h].m * m.mp >= min_phys).collect()
}

fn top_n(m: &Model, hosts: &[usize], n: usize) -> Vec<usize> {
    let mut o = hosts.to_vec();
    o.sort_by(|&a, &b| m.cat.hosts[b].m.total_cmp(&m.cat.hosts[a].m));
    o.truncate(n);
    o
}

/// Per-host counts of surviving subhaloes within R_host in physical M_peak bins.
fn sub_counts(m: &Model, h: usize, edges: &[f64]) -> Vec<f64> {
    let mut c = vec![0.0; edges.len() - 1];
    for &i in &m.sats[h] {
        let b = &m.cat.branches[i];
        if !b.surviving {
            continue;
        }
        let x = b.m_peak * m.mp;
        if x >= edges[0] && x < *edges.last().unwrap() {
            c[edges.partition_point(|&e| e <= x) - 1] += 1.0;
        }
    }
    c
}

fn bin_x(ed: &[f64], k: usize) -> f64 {
    if ed[k + 1].is_finite() { (ed[k] * ed[k + 1]).sqrt() } else { ed[k] }
}

fn log_edges(lo: f64, hi: f64, dex: f64) -> Vec<f64> {
    let n = (((hi / lo).log10() / dex).ceil() as usize).max(1);
    (0..=n).map(|i| lo * 10f64.powf(dex * i as f64)).collect()
}

/// One comparison quantity.
#[derive(Clone, Debug)]
struct Q {
    model: String,
    quantity: String,
    x: f64,
    base: f64,
    var: f64,
    ratio: f64,
    err: f64,
}

impl Q {
    fn sigma_dev(&self) -> f64 {
        if self.err > 0.0 { (self.ratio - 1.0) / self.err } else { f64::NAN }
    }
    /// within 10% or within 2 sigma
    fn converged(&self) -> bool {
        self.ratio.is_finite() && ((self.ratio - 1.0).abs() <= 0.10 || (self.err > 0.0 && self.sigma_dev().abs() <= 2.0))
    }
}

fn csvf(x: f64) -> String {
    if x.is_finite() { format!("{x:.6e}") } else { "nan".into() }
}

/// Ratio of per-host means (summed) with error: paired bootstrap if aligned, else independent.
fn compare_hosts(b: &[Vec<f64>], v: &[Vec<f64>], paired: bool, nb: usize, n_boot: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mb = mean_rows(b, nb);
    let mv = mean_rows(v, nb);
    if paired {
        let (r, e) = boot_ratio(b, v, nb, n_boot, seed);
        (mb, mv, r, e)
    } else {
        let (eb, ev) = (boot_mean_err(b, nb, n_boot, seed), boot_mean_err(v, nb, n_boot, seed ^ 1));
        let r: Vec<f64> = (0..nb).map(|k| if mb[k] > 0.0 { mv[k] / mb[k] } else { f64::NAN }).collect();
        let e: Vec<f64> = (0..nb).map(|k| r[k] * ((eb[k] / mb[k]).powi(2) + (ev[k] / mv[k]).powi(2)).sqrt()).collect();
        (mb, mv, r, e)
    }
}

/// Run-internal pieces for the comparison.
struct RunSample {
    /// MW-analog CDM hosts (aligned with the base top-n when matched; None = unmatched)
    mw: Vec<Option<usize>>,
    /// their WDM counterparts within the run
    mw_w: Vec<Option<usize>>,
    match_method: String,
    match_frac: Vec<f64>,
}

fn within_run_pairs(r: &Run, la: Option<&Labels>, lw: Option<&Labels>, hosts: &[Option<usize>], min_phys: f64) -> (Vec<Option<usize>>, String) {
    let Some(w) = &r.wdm else { return (vec![None; hosts.len()], "no WDM model".into()) };
    let list: Vec<usize> = hosts.iter().flatten().copied().collect();
    let hw = big_hosts(w, min_phys * 0.1);
    let res = match (la, lw) {
        (Some(a), Some(b)) => match_by_lagrangian(&r.cdm, a, &list, w, b, &hw).map(|x| (x.0, "particle IDs".to_string())),
        _ => None,
    };
    let (m, how) = res.unwrap_or_else(|| (pair_hosts(&r.cdm, w, &list), "position (fallback)".to_string()));
    let map: HashMap<usize, Option<usize>> = list.iter().copied().zip(m).collect();
    (hosts.iter().map(|h| h.and_then(|h| map.get(&h).copied().flatten())).collect(), how)
}

/// M50 with beta, gamma fixed (1D maximum likelihood in alpha), plus bootstrap error over host pairs.
fn m50_fixed(c: &[Vec<f64>], w: &[Vec<f64>], edges: &[f64], m_hm: f64, bg: (f64, f64), n_boot: usize, seed: u64) -> (f64, f64) {
    let nb = edges.len() - 1;
    let fit = |idx: &[usize]| -> f64 {
        let mut d = Vec::new();
        for k in 0..nb {
            let nc: f64 = idx.iter().map(|&i| c[i][k]).sum();
            let nw: f64 = idx.iter().map(|&i| w[i][k]).sum();
            if nc + nw > 0.0 {
                d.push(BinPair { m_lo: edges[k], m_hi: edges[k + 1], m_hm, n_cdm: nc, n_wdm: nw });
            }
        }
        if d.len() < 2 {
            return f64::NAN;
        }
        let nll = |la: f64| wdm::neg_loglike(&Supp { alpha: la.exp(), beta: bg.0, gamma: bg.1 }, &d);
        let (mut lo, mut hi) = (-8.0f64, 8.0f64);
        let g = 0.5 * (5f64.sqrt() - 1.0);
        for _ in 0..100 {
            let (x1, x2) = (hi - g * (hi - lo), lo + g * (hi - lo));
            if nll(x1) < nll(x2) { hi = x2 } else { lo = x1 }
        }
        wdm::m_half_suppression(&Supp { alpha: (0.5 * (lo + hi)).exp(), beta: bg.0, gamma: bg.1 }, 1.0) * m_hm
    };
    let all: Vec<usize> = (0..c.len()).collect();
    let best = fit(&all);
    if c.len() < 3 {
        return (best, f64::NAN);
    }
    let boots: Vec<f64> = (0..n_boot)
        .into_par_iter()
        .map(|i| {
            let mut r = crate::galaxy::Rng(seed ^ (i as u64 * 0x9E37));
            let pick: Vec<usize> = (0..c.len()).map(|_| r.below(c.len())).collect();
            fit(&pick)
        })
        .filter(|x| x.is_finite())
        .collect();
    let m = boots.iter().sum::<f64>() / boots.len().max(1) as f64;
    let sd = (boots.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (boots.len().max(2) - 1) as f64).sqrt();
    (best, sd)
}

pub fn run(base_dir: &Path, conv_dir: &Path, o: &ConvOpts) -> Result<(), String> {
    let t0 = std::time::Instant::now();
    let out = o.out.clone().unwrap_or_else(|| conv_dir.join("analysis"));
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let base = load_run("base", base_dir)?;
    let mp_b = base.cdm.mp;
    let mc_base = o.m_complete_base_particles * mp_b;
    let host_min = o.host_min_base_particles * mp_b;
    let m_hm_phys = base.wdm.as_ref().map(|w| w.m_hm * w.mp).unwrap_or(f64::NAN);
    let bg = o.beta_gamma.unwrap_or_else(|| {
        let s = std::fs::read_to_string(base_dir.join("analysis").join("experiment_summary.json")).ok();
        let v: Option<Value> = s.and_then(|s| serde_json::from_str(&s).ok());
        let f = v.as_ref().and_then(|v| v.pointer("/suppression/subhalo_2D_fit").cloned());
        f.as_ref().and_then(|f| Some((f.get("beta")?.as_f64()?, f.get("gamma")?.as_f64()?))).unwrap_or((6.66, -0.135))
    });
    eprintln!(
        "base: particle mass {mp_b:.6}, M_complete = {mc_base:.3} mass units ({} base particles), M_hm(WDM) = {m_hm_phys:.3} mass units, beta/gamma fixed at {:.3}/{:.3}",
        o.m_complete_base_particles, bg.0, bg.1
    );
    let mut variants: Vec<(String, PathBuf)> = std::fs::read_dir(conv_dir)
        .map_err(|e| format!("{}: {e}", conv_dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir() && p.join("cdm").join("subhalos").join("hosts_a1.000.csv").is_file())
        .map(|p| (p.file_name().unwrap().to_string_lossy().to_string(), p))
        .collect();
    variants.sort();
    if variants.is_empty() {
        return Err(format!("{}: no complete variant (needs <variant>/cdm/subhalos/hosts_a1.000.csv)", conv_dir.display()));
    }
    // base labels and base MW sample
    let t = std::time::Instant::now();
    let lb_c = snap_labels(&base.cdm);
    let lb_w = base.wdm.as_ref().and_then(snap_labels);
    eprintln!("base FoF labels: {:.1}s", t.elapsed().as_secs_f64());
    let base_big = big_hosts(&base.cdm, host_min);
    let base_top = top_n(&base.cdm, &base_big, o.n_mw);
    let base_mw: Vec<Option<usize>> = base_top.iter().map(|&h| Some(h)).collect();
    let (base_mw_w, base_pair_how) = within_run_pairs(&base, lb_c.as_ref(), lb_w.as_ref(), &base_mw, host_min);
    eprintln!(
        "base: top {} CDM hosts, {} with WDM counterparts ({base_pair_how})",
        base_top.len(),
        base_mw_w.iter().filter(|x| x.is_some()).count()
    );

    let mut rows: Vec<(String, Q)> = Vec::new();
    let mut summary: Vec<(String, Q)> = Vec::new();
    let mut notes: Vec<Value> = Vec::new();
    for (vname, vdir) in &variants {
        let t = std::time::Instant::now();
        let v = match load_run(vname, vdir) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("skipping {vname}: {e}");
                continue;
            }
        };
        let mp_v = v.cdm.mp;
        let mc_v = o.m_complete_base_particles * mp_v;
        let mc = mc_base.max(mc_v);
        let mc_flag = mc > mc_base * 1.0001;
        let same_seed = match (base.seed, v.seed) {
            (Some(a), Some(b)) => a == b,
            _ => true,
        };
        // ---- host matching base -> variant ----
        let lv_c = snap_labels(&v.cdm);
        let lv_w = v.wdm.as_ref().and_then(snap_labels);
        let v_big = big_hosts(&v.cdm, 0.1 * host_min);
        let sample = if !same_seed {
            let top = top_n(&v.cdm, &v_big, o.n_mw);
            RunSample { mw: top.iter().map(|&h| Some(h)).collect(), mw_w: vec![], match_method: "none: different seed, own top-n hosts compared statistically".into(), match_frac: vec![] }
        } else {
            let lag = match (&lb_c, &lv_c) {
                (Some(a), Some(b)) => match_by_lagrangian(&base.cdm, a, &base_top, &v.cdm, b, &v_big),
                _ => None,
            };
            match lag {
                Some((m, f)) => RunSample { mw: m, mw_w: vec![], match_method: format!("Lagrangian particle overlap (lattice {} -> {})", base.cdm.lattice, v.cdm.lattice), match_frac: f },
                None => RunSample { mw: pair_hosts(&base.cdm, &v.cdm, &base_top), mw_w: vec![], match_method: "position (fallback: no snapshot or incommensurate lattice)".into(), match_frac: vec![] },
            }
        };
        let (mw_w, v_pair_how) = within_run_pairs(&v, lv_c.as_ref(), lv_w.as_ref(), &sample.mw, host_min);
        let sample = RunSample { mw_w, ..sample };
        drop(lv_c);
        drop(lv_w);
        let n_matched = sample.mw.iter().filter(|x| x.is_some()).count();
        eprintln!(
            "{vname}: particle mass {mp_v:.6} ({:.2} x base), M_complete used {mc:.3}{}; MW matched {n_matched}/{} [{}]; WDM pairs {} [{v_pair_how}]; {:.1}s",
            mp_v / mp_b,
            if mc_flag { " (FLAG: raised to 160 variant particles)" } else { "" },
            base_top.len(),
            sample.match_method,
            sample.mw_w.iter().filter(|x| x.is_some()).count(),
            t.elapsed().as_secs_f64()
        );
        let mut fr: Vec<f64> = sample.match_frac.iter().copied().filter(|x| x.is_finite()).collect();
        fr.sort_by(|a, b| a.total_cmp(b));
        notes.push(serde_json::json!({
            "variant": vname, "what_changed": what_changed(vname), "particle_mass": mp_v, "particle_mass_over_base": mp_v / mp_b,
            "M_complete_used": mc, "M_complete_flag": mc_flag, "same_seed": same_seed,
            "host_matching": sample.match_method, "mw_matched": n_matched,
            "shared_frac_median": fr.get(fr.len() / 2), "wdm_pairing": v_pair_how,
        }));
        let push = |rows: &mut Vec<(String, Q)>, q: Q| rows.push((vname.clone(), q));

        // ---- P(k) ----
        for (lab, bm, vm) in [("cdm", Some(&base.cdm), Some(&v.cdm)), ("wdm", base.wdm.as_ref(), v.wdm.as_ref())] {
            let (Some(bm), Some(vm)) = (bm, vm) else { continue };
            let (Ok(pb), Ok(pv)) = (meta::read_pk_csv(&bm.dir.join("pk.csv")), meta::read_pk_csv(&vm.dir.join("pk.csv"))) else { continue };
            let (Some((_, rb)), Some((_, rv))) = (meta::rows_nearest_a(&pb, 1.0), meta::rows_nearest_a(&pv, 1.0)) else { continue };
            // pk.csv bins run to each run's mesh Nyquist: compare up to the lower one
            let k_max = rb.last().map(|r| r.k).unwrap_or(0.0).min(rv.last().map(|r| r.k).unwrap_or(0.0));
            let mut rs = Vec::new();
            for r in rb.iter().filter(|r| r.k <= k_max) {
                let pvk = meta::interp_pk(&rv, r.k);
                let ratio = pvk / r.p_meas;
                let err = if r.n_modes > 0.0 { ratio * (2.0 / r.n_modes).sqrt() * if same_seed { 0.0 } else { 2f64.sqrt() } } else { f64::NAN };
                if ratio.is_finite() && r.k <= 0.5 * k_max {
                    rs.push(ratio);
                }
                push(&mut rows, Q { model: lab.into(), quantity: "pk".into(), x: r.k, base: r.p_meas, var: pvk, ratio, err });
            }
            if !rs.is_empty() {
                rs.sort_by(|a, b| a.total_cmp(b));
                let med = rs[rs.len() / 2];
                let worst = rs.iter().cloned().fold(1.0, |a: f64, x| if (x - 1.0).abs() > (a - 1.0).abs() { x } else { a });
                summary.push((vname.clone(), Q { model: lab.into(), quantity: "pk_median_ratio_k_le_half_nyq".into(), x: f64::NAN, base: 1.0, var: med, ratio: med, err: 0.0 }));
                summary.push((vname.clone(), Q { model: lab.into(), quantity: "pk_worst_ratio_k_le_half_nyq".into(), x: f64::NAN, base: 1.0, var: worst, ratio: worst, err: 0.0 }));
            }
        }

        // ---- FoF halo mass function, cumulative N(>M) in physical mass ----
        for (lab, bm, vm) in [("cdm", Some(&base.cdm), Some(&v.cdm)), ("wdm", base.wdm.as_ref(), v.wdm.as_ref())] {
            let (Some(bm), Some(vm)) = (bm, vm) else { continue };
            let mut mb: Vec<f64> = bm.cat.hosts.iter().map(|h| h.m * bm.mp).collect();
            let mut mv: Vec<f64> = vm.cat.hosts.iter().map(|h| h.m * vm.mp).collect();
            mb.sort_by(|a, b| b.total_cmp(a));
            mv.sort_by(|a, b| b.total_cmp(a));
            let ngt = |v: &[f64], m: f64| v.partition_point(|&x| x >= m) as f64;
            for &m in &log_edges(mc, mb.first().copied().unwrap_or(mc * 2.0), 0.25) {
                let (nb, nv) = (ngt(&mb, m), ngt(&mv, m));
                if nb < 5.0 {
                    break;
                }
                let ratio = nv / nb;
                let err = ratio * (1.0 / nb + 1.0 / nv.max(1.0)).sqrt();
                let q = Q { model: lab.into(), quantity: "fof_N_gt_M".into(), x: m, base: nb, var: nv, ratio, err };
                if (m / mc - 1.0).abs() < 1e-9 {
                    summary.push((vname.clone(), Q { quantity: "fof_N_gt_Mcomplete".into(), ..q.clone() }));
                }
                if (m / (100.0 * mc) - 1.0).abs() < 1e-6 {
                    summary.push((vname.clone(), Q { quantity: "fof_N_gt_100Mcomplete".into(), ..q.clone() }));
                }
                push(&mut rows, q);
            }
        }

        // ---- subhalo mass functions per MW host (M_peak >= M_complete, physical) ----
        let hi = base.cdm.cat.branches.iter().map(|b| b.m_peak * mp_b).fold(mc * 2.0, f64::max) * 1.01;
        let edges = log_edges(mc, hi, 0.2);
        let tot_edges = vec![mc, f64::INFINITY];
        // aligned host lists: (base host, variant host) for CDM; and for WDM via the in-run pairs
        let (pairs_c, pairs_w): (HostPairs, HostPairs) = if same_seed {
            let pc: Vec<(usize, usize)> = base_top.iter().zip(&sample.mw).filter_map(|(&b, v)| v.map(|v| (b, v))).collect();
            let pw: Vec<(usize, usize)> = (0..base_top.len())
                .filter_map(|u| Some((base_mw_w.get(u).copied().flatten()?, sample.mw_w.get(u).copied().flatten()?)))
                .collect();
            (pc, pw)
        } else {
            (vec![], vec![])
        };
        for (lab, bm, vm, pairs, bh, vh) in [
            ("cdm", Some(&base.cdm), Some(&v.cdm), &pairs_c, base_top.clone(), sample.mw.iter().flatten().copied().collect::<Vec<_>>()),
            (
                "wdm",
                base.wdm.as_ref(),
                v.wdm.as_ref(),
                &pairs_w,
                base_mw_w.iter().flatten().copied().collect::<Vec<_>>(),
                sample.mw_w.iter().flatten().copied().collect::<Vec<_>>(),
            ),
        ] {
            let (Some(bm), Some(vm)) = (bm, vm) else { continue };
            let paired = same_seed && !pairs.is_empty();
            let (b_hosts, v_hosts): (Vec<usize>, Vec<usize>) = if paired { pairs.iter().copied().unzip() } else { (bh, vh) };
            if b_hosts.is_empty() || v_hosts.is_empty() {
                continue;
            }
            for (qname, ed) in [("shmf_per_host", &edges), ("n_sub_per_host_ge_Mcomplete", &tot_edges)] {
                let nbins = ed.len() - 1;
                let rb: Vec<Vec<f64>> = b_hosts.iter().map(|&h| sub_counts(bm, h, ed)).collect();
                let rv: Vec<Vec<f64>> = v_hosts.iter().map(|&h| sub_counts(vm, h, ed)).collect();
                let (mb_, mv_, r, e) = compare_hosts(&rb, &rv, paired, nbins, o.n_boot, o.seed);
                for k in 0..nbins {
                    let q = Q { model: lab.into(), quantity: qname.into(), x: bin_x(ed, k), base: mb_[k], var: mv_[k], ratio: r[k], err: e[k] };
                    if qname == "n_sub_per_host_ge_Mcomplete" {
                        summary.push((vname.clone(), Q { x: mc, ..q.clone() }));
                    }
                    push(&mut rows, q);
                }
            }
        }

        // ---- science: WDM/CDM suppression within each run, variant vs base ----
        if let (Some(bw), Some(vw)) = (&base.wdm, &v.wdm) {
            let pick = |cdm: &Model, w: &Model, mw: &[Option<usize>], mw_w: &[Option<usize>], ed: &[f64]| -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
                let pr: Vec<(usize, usize)> = mw.iter().zip(mw_w).filter_map(|(c, w)| Some(((*c)?, (*w)?))).collect();
                (pr.iter().map(|&(c, _)| sub_counts(cdm, c, ed)).collect(), pr.iter().map(|&(_, x)| sub_counts(w, x, ed)).collect())
            };
            for (qname, ed) in [("suppression_wdm_over_cdm", &edges), ("sat_count_ratio_wdm_over_cdm_per_host", &tot_edges)] {
                let nbins = ed.len() - 1;
                let (bc, bwc) = pick(&base.cdm, bw, &base_mw, &base_mw_w, ed);
                let (vc, vwc) = pick(&v.cdm, vw, &sample.mw, &sample.mw_w, ed);
                if bc.is_empty() || vc.is_empty() {
                    continue;
                }
                let (rb, eb) = boot_ratio(&bc, &bwc, nbins, o.n_boot, o.seed ^ 3);
                let (rv, ev) = boot_ratio(&vc, &vwc, nbins, o.n_boot, o.seed ^ 4);
                for k in 0..nbins {
                    let ratio = rv[k] / rb[k];
                    let err = ratio * ((eb[k] / rb[k]).powi(2) + (ev[k] / rv[k]).powi(2)).sqrt();
                    let q = Q { model: "wdm/cdm".into(), quantity: qname.into(), x: bin_x(ed, k), base: rb[k], var: rv[k], ratio, err };
                    if qname.starts_with("sat_count") {
                        summary.push((vname.clone(), Q { x: mc, ..q.clone() }));
                    }
                    push(&mut rows, q);
                }
                if qname == "suppression_wdm_over_cdm" {
                    let (m50b, e50b) = m50_fixed(&bc, &bwc, ed, m_hm_phys, bg, o.n_boot.min(100), o.seed ^ 5);
                    let (m50v, e50v) = m50_fixed(&vc, &vwc, ed, m_hm_phys, bg, o.n_boot.min(100), o.seed ^ 6);
                    let ratio = m50v / m50b;
                    let err = ratio * ((e50b / m50b).powi(2) + (e50v / m50v).powi(2)).sqrt();
                    let q = Q { model: "wdm/cdm".into(), quantity: "M50_fixed_beta_gamma".into(), x: m_hm_phys, base: m50b, var: m50v, ratio, err };
                    summary.push((vname.clone(), q.clone()));
                    push(&mut rows, q);
                }
            }
        }
    }

    // ---- outputs ----
    let mut c = String::from("variant,model,quantity,x,base,variant_value,ratio,err,sigma_dev,converged\n");
    for (v, q) in &rows {
        let _ = writeln!(
            c,
            "{v},{},{},{},{},{},{},{},{},{}",
            q.model,
            q.quantity,
            csvf(q.x),
            csvf(q.base),
            csvf(q.var),
            csvf(q.ratio),
            csvf(q.err),
            csvf(q.sigma_dev()),
            q.converged()
        );
    }
    std::fs::write(out.join("convergence.csv"), c).map_err(|e| e.to_string())?;
    // run-to-run reproducibility floor from a "control" variant (base settings rerun), if present
    let control: HashMap<(String, String), f64> = summary
        .iter()
        .filter(|(v, _)| v == "control")
        .map(|(_, q)| ((q.model.clone(), q.quantity.clone()), q.ratio))
        .collect();
    let vs_control = |v: &str, q: &Q| -> (f64, Option<bool>) {
        let Some(&rc) = control.get(&(q.model.clone(), q.quantity.clone())).filter(|_| v != "control") else {
            return (f64::NAN, None);
        };
        let floor = (q.err.max(0.0).powi(2) + (rc - 1.0).powi(2)).sqrt();
        (rc, Some(q.ratio.is_finite() && (q.ratio - 1.0).abs() <= (0.10f64).max(2.0 * floor)))
    };
    let mut s = String::from("variant,what_changed,quantity,ratio_to_base,err,converged,control_ratio,converged_vs_control\n");
    for (v, q) in &summary {
        let (rc, cc) = vs_control(v, q);
        let _ = writeln!(
            s,
            "{v},\"{}\",{}:{},{},{},{},{},{}",
            what_changed(v),
            q.model,
            q.quantity,
            csvf(q.ratio),
            csvf(q.err),
            q.converged(),
            csvf(rc),
            cc.map(|b| b.to_string()).unwrap_or_default()
        );
    }
    std::fs::write(out.join("convergence_summary.csv"), s).map_err(|e| e.to_string())?;
    let meta = serde_json::json!({
        "base": base_dir.display().to_string(), "conv_dir": conv_dir.display().to_string(),
        "units": "physical: mass = particle count x particle_mass (box-area units, mean surface density 1)",
        "M_complete_base": mc_base, "M_hm_wdm": m_hm_phys, "beta_gamma_fixed": [bg.0, bg.1],
        "converged_definition": "|ratio - 1| <= 10% or within 2 sigma (bootstrap / Poisson)",
        "converged_vs_control_definition": "|ratio - 1| <= max(10%, 2 sqrt(err^2 + (control_ratio - 1)^2)) with the control (rerun) variant as reproducibility floor",
        "pk_error_note": "P(k) ratios of same-seed variants have no cosmic variance (error 0: only the 10% criterion applies)",
        "variants": notes,
    });
    std::fs::write(out.join("convergence_meta.json"), serde_json::to_string_pretty(&meta).unwrap()).map_err(|e| e.to_string())?;

    println!();
    println!("{:<10} {:<9} {:<40} {:>9} {:>8} {:>8} {:>6} {:>9}", "variant", "model", "quantity", "ratio", "err", "sigma", "conv", "vs_ctrl");
    let opt = |x: f64, p: usize| if x.is_finite() { format!("{x:.p$}") } else { "-".into() };
    for (v, q) in &summary {
        println!(
            "{:<10} {:<9} {:<40} {:>9.4} {:>8} {:>8} {:>6} {:>9}",
            v,
            q.model,
            q.quantity,
            q.ratio,
            opt(if q.err > 0.0 { q.err } else { f64::NAN }, 4),
            opt(q.sigma_dev(), 2),
            if q.converged() { "yes" } else { "NO" },
            match vs_control(v, q).1 {
                Some(true) => "yes",
                Some(false) => "NO",
                None => "-",
            }
        );
    }
    println!("outputs in {} ({:.1}s)", out.display(), t0.elapsed().as_secs_f64());
    Ok(())
}
