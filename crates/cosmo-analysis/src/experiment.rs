//! `cosmo-analysis experiment <dir>`: CDM vs WDM subhalo / satellite experiment in the 2D universe.
//!
//! Layout: <dir>/<model>/{meta.json, snap_a1.000.bin?, subhalos/{hosts,branches,subhalos}_a1.000.csv}.
//! All masses are handled in particle-mass units (catalog mass / particle_mass).

use crate::catalog::{Branch, Catalog};
use crate::galaxy::{self, FIDUCIAL, GalModel, LikeTable, PRIOR_FIDUCIAL, PRIOR_WIDE, Rng, Sat};
use crate::meta::Meta;
use crate::snapshot::Snapshot;
use crate::wdm::{self, BinPair, Supp};
use crate::{hmf, image, pk};
use rayon::prelude::*;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::f64::consts::PI;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct ExpOpts {
    pub n_mw: usize,
    pub target_sats: f64,
    pub host_min: f64,
    pub fit_min: f64,
    /// completeness mass (particle masses); None = completeness.csv, else 150
    pub m_complete: Option<f64>,
    pub n_boot: usize,
    pub n_mc: usize,
    pub png: bool,
    /// spurious-halo cleaning: "lovell" (default), "shape", "none"
    pub clean: String,
    /// fractional satellite-count systematic (finder f_esc), marginalised in the direct likelihood
    pub count_sys: f64,
    pub seed: u64,
    pub out: Option<PathBuf>,
}

impl Default for ExpOpts {
    fn default() -> Self {
        ExpOpts {
            n_mw: 30,
            target_sats: 30.0,
            host_min: 1000.0,
            fit_min: 50.0,
            m_complete: None,
            n_boot: 200,
            n_mc: 1000,
            png: true,
            clean: "lovell".into(),
            count_sys: 0.10,
            seed: 20260,
            out: None,
        }
    }
}

pub struct Model {
    pub name: String,
    pub dir: PathBuf,
    pub is_cdm: bool,
    pub k_hm: f64,
    pub alpha: f64,
    pub nu: f64,
    /// 2D half-mode mass (particle masses), pi Sigma (pi/k_hm)^2
    pub m_hm: f64,
    pub m_hm_meta: Option<f64>,
    pub mp: f64,
    pub np: f64,
    pub box_l: f64,
    pub lattice: usize,
    pub sigma_bar: f64,
    pub meta: Option<Meta>,
    pub cat: Catalog,
    /// per host: indices of branches within R_host (projected)
    pub sats: Vec<Vec<usize>>,
    /// per branch: r / R_host (NaN if host unknown)
    pub r_over: Vec<f64>,
    pub k_peak: f64,
    pub m_lim: f64,
}

fn jnum(v: &Value, k: &str) -> Option<f64> {
    v.get(k).and_then(|x| x.as_f64().or_else(|| x.as_str().and_then(|s| s.trim().parse().ok())))
}

fn min_image(d: f64, l: f64) -> f64 {
    d - l * (d / l).round()
}

impl Model {
    pub fn load(dir: &Path) -> Result<Model, String> {
        let s = std::fs::read_to_string(dir.join("meta.json")).map_err(|e| format!("{}: {e}", dir.display()))?;
        let raw: Value = serde_json::from_str(s.trim_start_matches('\u{feff}')).map_err(|e| format!("{}: meta.json: {e}", dir.display()))?;
        let box_l = jnum(&raw, "box").unwrap_or(2000.0);
        let lattice_m = jnum(&raw, "lattice");
        let np = jnum(&raw, "particles").or(lattice_m.map(|x| x * x)).ok_or("meta.json: no particles/lattice")?;
        let lattice = lattice_m.unwrap_or(np.sqrt().round()) as usize;
        // finder convention: mass = count x particle_mass, or L^2/N (mean surface density 1) if absent
        let mp = jnum(&raw, "particle_mass").unwrap_or(box_l * box_l / np);
        let dm = raw.get("dark_matter").or_else(|| raw.get("dm")).cloned().unwrap_or(Value::Null);
        let name = dir.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        let model_s = dm.get("model").and_then(|x| x.as_str()).unwrap_or(&name).to_string();
        let (k0, a0, mhm_meta) = (jnum(&dm, "k_hm"), jnum(&dm, "alpha"), jnum(&dm, "M_hm"));
        let nu = jnum(&dm, "nu").unwrap_or(1.12);
        let sigma_bar = np / (box_l * box_l);
        let is_cdm = model_s.eq_ignore_ascii_case("cdm")
            || (k0.unwrap_or(0.0) <= 0.0 && a0.unwrap_or(0.0) <= 0.0 && mhm_meta.unwrap_or(0.0) <= 0.0);
        let (k_hm, alpha) = if is_cdm {
            (0.0, 0.0)
        } else {
            match (k0.filter(|k| *k > 0.0), a0.filter(|a| *a > 0.0)) {
                (Some(k), Some(a)) => (k, a),
                (Some(k), None) => (k, wdm::alpha_from_khm(k, nu)),
                (None, Some(a)) => (wdm::k_half_mode(a, nu), a),
                (None, None) => {
                    let k = wdm::k_hm_from_mhm_2d(mhm_meta.unwrap() / mp, sigma_bar);
                    (k, wdm::alpha_from_khm(k, nu))
                }
            }
        };
        let m_hm = if is_cdm { 0.0 } else { wdm::m_half_mode_2d(k_hm, sigma_bar) };
        let cat = Catalog::load(&dir.join("subhalos"), mp)?;
        let host_idx: HashMap<i64, usize> = cat.hosts.iter().enumerate().map(|(i, h)| (h.id, i)).collect();
        let mut sats = vec![Vec::new(); cat.hosts.len()];
        let mut r_over = vec![f64::NAN; cat.branches.len()];
        for (i, b) in cat.branches.iter().enumerate() {
            if let Some(&h) = host_idx.get(&b.host_id) {
                let host = &cat.hosts[h];
                // never a satellite (a_infall = -1) or the host's own central branch
                if (b.a_infall.is_finite() && b.a_infall < 0.0) || b.id == host.central_id {
                    continue;
                }
                let (dx, dy) = (min_image(b.x - host.x, box_l), min_image(b.y - host.y, box_l));
                let r = (dx * dx + dy * dy).sqrt() / host.r;
                r_over[i] = r;
                if r <= 1.0 {
                    sats[h].push(i);
                }
            }
        }
        let mut m = Model {
            name,
            dir: dir.to_path_buf(),
            is_cdm,
            k_hm,
            alpha,
            nu,
            m_hm,
            m_hm_meta: mhm_meta.map(|x| x / mp),
            mp,
            np,
            box_l,
            lattice,
            sigma_bar,
            meta: Meta::from_value(raw).ok(),
            cat,
            sats,
            r_over,
            k_peak: f64::NAN,
            m_lim: f64::NAN,
        };
        if let Some(meta) = &m.meta {
            let (spec, g2) = (meta.spectrum, (1.0 / meta.a_init).powi(2));
            let (alpha, nu) = (m.alpha, m.nu);
            let pw = move |k: f64| g2 * spec.p0(k) * wdm::transfer(k, alpha, nu).powi(2);
            let kf = 2.0 * PI / box_l;
            let kny = PI * lattice as f64 / box_l;
            m.k_peak = wdm::k_peak_2d(&pw, kf, kny);
            m.m_lim = wdm::m_lim_2d(sigma_bar, box_l / np.sqrt(), m.k_peak);
        }
        Ok(m)
    }

    /// sigma(M) at a = 1 for this model's linear spectrum (NaN without spectrum).
    pub fn sigma(&self, m: f64) -> f64 {
        let Some(meta) = &self.meta else { return f64::NAN };
        let (spec, g2, alpha, nu) = (meta.spectrum, (1.0 / meta.a_init).powi(2), self.alpha, self.nu);
        let p = move |k: f64| g2 * spec.p0(k) * wdm::transfer(k, alpha, nu).powi(2);
        let r = (m / (PI * self.sigma_bar)).sqrt();
        hmf::sigma2_r_with(&p, r, 2.0 * PI / self.box_l, PI * self.lattice as f64 / self.box_l).sqrt()
    }

    fn big_hosts(&self, host_min: f64) -> Vec<usize> {
        (0..self.cat.hosts.len()).filter(|&h| self.cat.hosts[h].n_fof >= host_min).collect()
    }

    fn sat(&self, i: usize) -> Sat {
        let b = &self.cat.branches[i];
        Sat {
            m_peak: b.m_peak,
            v_peak: b.v_peak,
            a_infall: if b.a_infall.is_finite() { b.a_infall } else { 0.5 },
            surviving: b.surviving,
        }
    }

    pub fn host_sats(&self, h: usize) -> Vec<Sat> {
        self.sats[h].iter().map(|&i| self.sat(i)).collect()
    }
}

/// Model directories (with meta.json and subhalos/hosts_a1.000.csv), CDM first, WDM by M_hm.
pub fn discover(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.join("meta.json").is_file() && p.join("subhalos").join("hosts_a1.000.csv").is_file())
        .collect();
    v.sort();
    Ok(v)
}

type KeyFn = fn(&Branch) -> f64;

fn log_edges(lo: f64, hi: f64, dex: f64) -> Vec<f64> {
    let n = (((hi / lo).log10() / dex).ceil() as usize).max(1);
    (0..=n).map(|i| lo * 10f64.powf(dex * i as f64)).collect()
}

fn bin_of(edges: &[f64], x: f64) -> Option<usize> {
    if x.is_nan() || x < edges[0] || x >= *edges.last().unwrap() {
        return None;
    }
    Some(edges.partition_point(|&e| e <= x) - 1)
}

/// Per-host counts of surviving satellites (inside R_host) in bins of `key`.
fn host_counts(md: &Model, hosts: &[usize], edges: &[f64], key: fn(&Branch) -> f64) -> Vec<Vec<f64>> {
    hosts
        .iter()
        .map(|&h| {
            let mut c = vec![0.0; edges.len() - 1];
            for &i in &md.sats[h] {
                let b = &md.cat.branches[i];
                if b.surviving && let Some(k) = bin_of(edges, key(b)) {
                    c[k] += 1.0;
                }
            }
            c
        })
        .collect()
}

pub(crate) fn mean_rows(rows: &[Vec<f64>], nb: usize) -> Vec<f64> {
    let mut m = vec![0.0; nb];
    for r in rows {
        for b in 0..nb {
            m[b] += r[b];
        }
    }
    let n = rows.len().max(1) as f64;
    m.iter().map(|x| x / n).collect()
}

fn std_of(samples: &[Vec<f64>], nb: usize) -> Vec<f64> {
    (0..nb)
        .map(|b| {
            let vals: Vec<f64> = samples.iter().map(|s| s[b]).filter(|x| x.is_finite()).collect();
            if vals.len() < 2 {
                return f64::NAN;
            }
            let m = vals.iter().sum::<f64>() / vals.len() as f64;
            (vals.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (vals.len() as f64 - 1.0)).sqrt()
        })
        .collect()
}

pub(crate) fn boot_mean_err(rows: &[Vec<f64>], nb: usize, n_boot: usize, seed: u64) -> Vec<f64> {
    if rows.len() < 2 {
        return vec![f64::NAN; nb];
    }
    let mut r = Rng(seed);
    let samples: Vec<Vec<f64>> = (0..n_boot)
        .map(|_| {
            let pick: Vec<Vec<f64>> = (0..rows.len()).map(|_| rows[r.below(rows.len())].clone()).collect();
            mean_rows(&pick, nb)
        })
        .collect();
    std_of(&samples, nb)
}

/// Paired ratio sum(wdm)/sum(cdm) per bin with bootstrap over pairs.
pub(crate) fn boot_ratio(c: &[Vec<f64>], w: &[Vec<f64>], nb: usize, n_boot: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let ratio = |idx: &[usize]| -> Vec<f64> {
        (0..nb)
            .map(|b| {
                let sc: f64 = idx.iter().map(|&i| c[i][b]).sum();
                let sw: f64 = idx.iter().map(|&i| w[i][b]).sum();
                if sc > 0.0 { sw / sc } else { f64::NAN }
            })
            .collect()
    };
    let all: Vec<usize> = (0..c.len()).collect();
    let r0 = ratio(&all);
    if c.len() < 2 {
        return (r0, vec![f64::NAN; nb]);
    }
    let mut r = Rng(seed);
    let samples: Vec<Vec<f64>> =
        (0..n_boot).map(|_| ratio(&(0..c.len()).map(|_| r.below(c.len())).collect::<Vec<_>>())).collect();
    (r0, std_of(&samples, nb))
}

/// Pair each CDM host with the nearest WDM host within 0.5 R_host and a factor 3 in mass.
pub(crate) fn pair_hosts(cdm: &Model, w: &Model, hosts: &[usize]) -> Vec<Option<usize>> {
    let l = cdm.box_l;
    hosts
        .par_iter()
        .map(|&c| {
            let hc = &cdm.cat.hosts[c];
            let mut best: Option<(f64, usize)> = None;
            for (j, hw) in w.cat.hosts.iter().enumerate() {
                let (dx, dy) = (min_image(hw.x - hc.x, l), min_image(hw.y - hc.y, l));
                let d = (dx * dx + dy * dy).sqrt();
                if d < 0.5 * hc.r && hw.m > hc.m / 3.0 && hw.m < hc.m * 3.0 && best.is_none_or(|b| d < b.0) {
                    best = Some((d, j));
                }
            }
            best.map(|b| b.1)
        })
        .collect()
}

/// Per-particle FoF labels (u32::MAX = not in a group with >= min_n members) and group sizes,
/// with the finder's linking length b = 0.2 L / sqrt(N).
pub(crate) fn fof_labels(pos: &[[f32; 2]], box_l: f64, min_n: usize) -> Result<(Vec<u32>, Vec<u32>), String> {
    let link = 0.2 * box_l / (pos.len() as f64).sqrt();
    let g = crate::fof::fof(pos, box_l, link, min_n)?;
    let mut lab = vec![u32::MAX; pos.len()];
    for k in 0..g.len() {
        for &i in g.group(k) {
            lab[i as usize] = k as u32;
        }
    }
    Ok((lab, (0..g.len()).map(|k| g.size(k) as u32).collect()))
}

/// (for each CDM group: (best WDM group, shared count); for each WDM group: best CDM group)
pub type GroupMatch = (Vec<Option<(u32, u32)>>, Vec<Option<u32>>);

/// Best partner of every group by shared particle IDs (particle index = stable lattice ID).
/// Returns (for each CDM group: (best WDM group, shared count); for each WDM group: best CDM group).
pub fn match_groups(lc: &[u32], lw: &[u32], nc: usize, nw: usize) -> GroupMatch {
    let mut keys: Vec<u64> = lc
        .par_iter()
        .zip(lw.par_iter())
        .filter(|(a, b)| **a != u32::MAX && **b != u32::MAX)
        .map(|(a, b)| ((*a as u64) << 32) | *b as u64)
        .collect();
    keys.par_sort_unstable();
    let mut best_w: Vec<Option<(u32, u32)>> = vec![None; nc];
    let mut best_c: Vec<Option<(u32, u32)>> = vec![None; nw];
    let mut i = 0;
    while i < keys.len() {
        let k = keys[i];
        let mut j = i;
        while j < keys.len() && keys[j] == k {
            j += 1;
        }
        let (c, w, n) = ((k >> 32) as u32, (k & 0xffff_ffff) as u32, (j - i) as u32);
        if best_w[c as usize].is_none_or(|b| n > b.1) {
            best_w[c as usize] = Some((w, n));
        }
        if best_c[w as usize].is_none_or(|b| n > b.1) {
            best_c[w as usize] = Some((c, n));
        }
        i = j;
    }
    (best_w, best_c.into_iter().map(|b| b.map(|x| x.0)).collect())
}

/// FoF group of each listed host: majority label of its central's most-bound tracer IDs.
pub(crate) fn host_groups(md: &Model, hosts: &[usize], labels: &[u32]) -> Vec<Option<u32>> {
    hosts
        .iter()
        .map(|&h| {
            let tr = md.cat.tracers.get(&md.cat.hosts[h].central_id)?;
            let mut votes: HashMap<u32, usize> = HashMap::new();
            for &id in tr {
                if let Some(&l) = labels.get(id as usize)
                    && l != u32::MAX
                {
                    *votes.entry(l).or_default() += 1;
                }
            }
            votes.into_iter().max_by_key(|&(l, n)| (n, std::cmp::Reverse(l))).map(|(l, _)| l)
        })
        .collect()
}

struct Pairing {
    method: String,
    /// per CDM big host (index into big[0]): paired WDM host index
    pairs: Vec<Option<usize>>,
    /// per CDM big host: shared fraction of the CDM FoF group with its best WDM group (NaN if unknown)
    frac: Vec<f64>,
}

/// Pair CDM and WDM hosts by shared particle IDs (mutual best match, shared fraction >= 0.5),
/// falling back to positional matching when snapshots or tracer files are missing.
fn pair_by_ids(models: &[Model], big: &[Vec<usize>], snaps: &[Option<PathBuf>]) -> Vec<Pairing> {
    let nm = models.len();
    let n0 = big[0].len();
    let identity = Pairing { method: "identity".into(), pairs: big[0].iter().map(|&h| Some(h)).collect(), frac: vec![1.0; n0] };
    let fallback = |j: usize, why: &str| {
        eprintln!("pairing {}: positional fallback ({why})", models[j].name);
        Pairing { method: format!("position (fallback: {why})"), pairs: pair_hosts(&models[0], &models[j], &big[0]), frac: vec![f64::NAN; n0] }
    };
    let ok_ids = |j: usize| snaps[j].is_some() && !models[j].cat.tracers.is_empty();
    let mut out = vec![identity];
    if nm == 1 {
        return out;
    }
    if !ok_ids(0) {
        for j in 1..nm {
            out.push(fallback(j, "CDM snapshot or tracers missing"));
        }
        return out;
    }
    let t = Instant::now();
    let labc = Snapshot::open(snaps[0].as_ref().unwrap()).and_then(|s| fof_labels(s.positions(), models[0].box_l, 20));
    let Ok((lc, szc)) = labc else {
        for j in 1..nm {
            out.push(fallback(j, "CDM FoF failed"));
        }
        return out;
    };
    let gc = host_groups(&models[0], &big[0], &lc);
    for j in 1..nm {
        if !ok_ids(j) {
            out.push(fallback(j, "snapshot or tracers missing"));
            continue;
        }
        let Ok((lw, szw)) = Snapshot::open(snaps[j].as_ref().unwrap()).and_then(|s| fof_labels(s.positions(), models[j].box_l, 20))
        else {
            out.push(fallback(j, "FoF failed"));
            continue;
        };
        if lw.len() != lc.len() {
            out.push(fallback(j, "particle count differs"));
            continue;
        }
        let (best_w, best_c) = match_groups(&lc, &lw, szc.len(), szw.len());
        let gw = host_groups(&models[j], &big[j], &lw);
        let host_of_group: HashMap<u32, usize> = big[j].iter().zip(&gw).filter_map(|(&h, g)| g.map(|g| (g, h))).collect();
        let mut pairs = vec![None; n0];
        let mut frac = vec![f64::NAN; n0];
        for u in 0..n0 {
            let Some(c) = gc[u] else { continue };
            let Some((w, n)) = best_w[c as usize] else {
                frac[u] = 0.0;
                continue;
            };
            let f = n as f64 / szc[c as usize] as f64;
            frac[u] = f;
            if f >= 0.5 && best_c[w as usize] == Some(c) {
                pairs[u] = host_of_group.get(&w).copied();
            }
        }
        out.push(Pairing { method: "particle IDs (FoF b=0.2 at a=1, mutual best, shared >= 0.5)".into(), pairs, frac });
    }
    eprintln!("ID pairing: {:.1}s", t.elapsed().as_secs_f64());
    out
}

#[derive(Clone, Copy, Debug)]
struct FitSummary {
    best: Supp,
    nll: f64,
    lo: Supp,
    hi: Supp,
    m50: f64,
    m50_lo: f64,
    m50_hi: f64,
    n_boot_ok: usize,
    /// parameters of the best fit sitting on a fit clamp
    at_bound: [bool; 3],
    /// fraction of bootstrap fits with any parameter on a clamp
    boot_at_bound: f64,
}

/// Which of (alpha, beta, gamma) are on the clamps used in wdm::fit_supp.
fn bound_hits(s: &Supp) -> [bool; 3] {
    let (la, lb, lg) = (s.alpha.ln(), s.beta.ln(), (-s.gamma).ln());
    [la.abs() > 7.99, !(-2.99..=wdm::LN_BETA_MAX - 0.01).contains(&lb), !(-4.99..=2.49).contains(&lg)]
}

fn pct(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    v[((v.len() - 1) as f64 * q).round() as usize]
}

/// Best fit plus bootstrap 16/84 percentiles. `make(sample)` builds the binned data for a resample.
fn fit_with_boot(
    make: &(dyn Fn(Option<&[usize]>) -> Vec<BinPair> + Sync),
    n_units: usize,
    n_boot: usize,
    seed: u64,
) -> Option<FitSummary> {
    let data = make(None);
    if data.iter().filter(|d| d.n_cdm + d.n_wdm > 0.0).count() < 4 {
        return None;
    }
    let (best, nll) = wdm::fit_supp(&data);
    let boots: Vec<Supp> = (0..n_boot)
        .into_par_iter()
        .filter_map(|i| {
            let mut r = Rng(seed ^ (i as u64).wrapping_mul(0x9E37_79B9));
            let pick: Vec<usize> = (0..n_units).map(|_| r.below(n_units)).collect();
            let d = make(Some(&pick));
            let x0 = [best.alpha.ln(), best.beta.ln(), (-best.gamma).ln()];
            let obj = |x: &[f64]| wdm::neg_loglike(&wdm::unpack(x), &d);
            let (x, v) = wdm::nelder_mead(&obj, &x0, 0.3, 1500);
            v.is_finite().then(|| wdm::unpack(&x))
        })
        .collect();
    let mut a: Vec<f64> = boots.iter().map(|s| s.alpha).collect();
    let mut b: Vec<f64> = boots.iter().map(|s| s.beta).collect();
    let mut g: Vec<f64> = boots.iter().map(|s| s.gamma).collect();
    let mut m: Vec<f64> = boots.iter().map(|s| wdm::m_half_suppression(s, 1.0)).collect();
    Some(FitSummary {
        best,
        nll,
        lo: Supp { alpha: pct(&mut a, 0.16), beta: pct(&mut b, 0.16), gamma: pct(&mut g, 0.16) },
        hi: Supp { alpha: pct(&mut a, 0.84), beta: pct(&mut b, 0.84), gamma: pct(&mut g, 0.84) },
        m50: wdm::m_half_suppression(&best, 1.0),
        m50_lo: pct(&mut m, 0.16),
        m50_hi: pct(&mut m, 0.84),
        n_boot_ok: boots.len(),
        at_bound: bound_hits(&best),
        boot_at_bound: boots.iter().filter(|s| bound_hits(s).iter().any(|&b| b)).count() as f64 / boots.len().max(1) as f64,
    })
}

fn supp_json(s: &Supp) -> Value {
    json!({"alpha": s.alpha, "beta": s.beta, "gamma": s.gamma, "M50_over_Mhm": wdm::m_half_suppression(s, 1.0)})
}

fn fit_json(f: &Option<FitSummary>) -> Value {
    match f {
        None => Value::Null,
        Some(f) => json!({
            "alpha": f.best.alpha, "alpha_16_84": [f.lo.alpha, f.hi.alpha],
            "beta": f.best.beta, "beta_16_84": [f.lo.beta, f.hi.beta],
            "gamma": f.best.gamma, "gamma_16_84": [f.lo.gamma, f.hi.gamma],
            "M50_over_Mhm": f.m50, "M50_over_Mhm_16_84": [f.m50_lo, f.m50_hi],
            "neg_log_like": f.nll, "bootstrap_samples": f.n_boot_ok,
            "best_fit_at_clamp_alpha_beta_gamma": f.at_bound, "bootstrap_fraction_at_clamp": f.boot_at_bound,
            "clamps": "ln alpha in [-8,8], ln beta in [-3,5] (beta <= 148), ln(-gamma) in [-5,2.5]",
        }),
    }
}

fn csvf(x: f64) -> String {
    if x.is_finite() { format!("{x:.6e}") } else { "nan".into() }
}

fn nums(x: f64) -> Value {
    if x.is_finite() { json!(x) } else { Value::Null }
}

/// Proto-halo axis ratios (Lovell et al. 2014 analogue): particles within r_half of each central at
/// a = 1 are mapped to their initial lattice positions q (particle index = lattice index), and
/// s = sqrt(lambda_min / lambda_max) of the 2D inertia tensor of q is returned (NaN if skipped).
fn proto_shapes(snap: &Snapshot, md: &Model, min_n: f64) -> Vec<f64> {
    let pos = snap.positions();
    let l = md.box_l;
    let side = md.lattice.max(1);
    let g = (side / 2).clamp(64, 4096);
    let cell = l / g as f64;
    let cidx = |x: f32, y: f32| -> usize {
        let cx = ((x as f64 / cell).floor() as i64).rem_euclid(g as i64) as usize;
        let cy = ((y as f64 / cell).floor() as i64).rem_euclid(g as i64) as usize;
        cy * g + cx
    };
    let mut start = vec![0u32; g * g + 1];
    for p in pos {
        start[cidx(p[0], p[1]) + 1] += 1;
    }
    for i in 0..g * g {
        start[i + 1] += start[i];
    }
    let mut fill = start.clone();
    let mut order = vec![0u32; pos.len()];
    for (i, p) in pos.iter().enumerate() {
        let c = cidx(p[0], p[1]);
        order[fill[c] as usize] = i as u32;
        fill[c] += 1;
    }
    let qs = l / side as f64;
    md.cat
        .subs
        .par_iter()
        .map(|s| {
            if s.n_bound < min_n || s.r_half.is_nan() || s.r_half <= 0.0 {
                return f64::NAN;
            }
            let r = s.r_half;
            let nc = (r / cell).ceil() as i64 + 1;
            let (cx0, cy0) = ((s.x / cell).floor() as i64, (s.y / cell).floor() as i64);
            let (mut n, mut q0) = (0.0f64, None::<(f64, f64)>);
            let (mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
            for dy in -nc..=nc {
                for dx in -nc..=nc {
                    let cx = (cx0 + dx).rem_euclid(g as i64) as usize;
                    let cy = (cy0 + dy).rem_euclid(g as i64) as usize;
                    let c = cy * g + cx;
                    for &pi in &order[start[c] as usize..start[c + 1] as usize] {
                        let p = pos[pi as usize];
                        let (ddx, ddy) = (min_image(p[0] as f64 - s.x, l), min_image(p[1] as f64 - s.y, l));
                        if ddx * ddx + ddy * ddy >= r * r {
                            continue;
                        }
                        let id = pi as usize;
                        let q = (((id % side) as f64 + 0.5) * qs, ((id / side) as f64 + 0.5) * qs);
                        let r0 = *q0.get_or_insert(q);
                        let (u, v) = (min_image(q.0 - r0.0, l), min_image(q.1 - r0.1, l));
                        n += 1.0;
                        sx += u;
                        sy += v;
                        sxx += u * u;
                        syy += v * v;
                        sxy += u * v;
                    }
                }
            }
            if n < 10.0 {
                return f64::NAN;
            }
            let (mx, my) = (sx / n, sy / n);
            let (a, b, c) = (sxx / n - mx * mx, syy / n - my * my, sxy / n - mx * my);
            let tr = 0.5 * (a + b);
            let disc = (0.25 * (a - b) * (a - b) + c * c).sqrt();
            let (l1, l2) = (tr + disc, (tr - disc).max(0.0));
            if l1 > 0.0 { (l2 / l1).sqrt() } else { f64::NAN }
        })
        .collect()
}

/// CIC counts of the particles in a square window (half-width w) centred on (cx, cy).
fn zoom_counts(pos: &[[f32; 2]], l: f64, cx: f64, cy: f64, w: f64, n: usize) -> Vec<f64> {
    let parts = rayon::current_num_threads().max(1);
    let chunk = pos.len().div_ceil(parts).max(1);
    let scale = n as f64 / (2.0 * w);
    let grids: Vec<Vec<f64>> = pos
        .par_chunks(chunk)
        .map(|ps| {
            let mut g = vec![0.0f64; n * n];
            for p in ps {
                let u = (min_image(p[0] as f64 - cx, l) + w) * scale - 0.5;
                let v = (min_image(p[1] as f64 - cy, l) + w) * scale - 0.5;
                if u < -1.0 || v < -1.0 || u >= n as f64 || v >= n as f64 {
                    continue;
                }
                let (fu, fv) = (u.floor(), v.floor());
                let (du, dv) = (u - fu, v - fv);
                for (oy, wy) in [(0i64, 1.0 - dv), (1, dv)] {
                    for (ox, wx) in [(0i64, 1.0 - du), (1, du)] {
                        let (ix, iy) = (fu as i64 + ox, fv as i64 + oy);
                        if ix >= 0 && iy >= 0 && (ix as usize) < n && (iy as usize) < n {
                            g[iy as usize * n + ix as usize] += wx * wy;
                        }
                    }
                }
            }
            g
        })
        .collect();
    let mut out = vec![0.0; n * n];
    out.par_chunks_mut(n).enumerate().for_each(|(r, o)| {
        for g in &grids {
            for (a, b) in o.iter_mut().zip(&g[r * n..(r + 1) * n]) {
                *a += b;
            }
        }
    });
    out
}

/// Completeness mass from the finder's completeness.csv: a column whose name contains
/// "complete"/"recommend" (first finite value), or a key,value row with such a key.
pub fn read_m_complete(dir: &Path) -> Option<(f64, String)> {
    let s = std::fs::read_to_string(dir.join("completeness.csv")).ok()?;
    let lines: Vec<&str> = s.lines().filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#')).collect();
    let key = |c: &str| {
        let c = c.to_ascii_lowercase();
        c.contains("m_complete") || c.contains("recommend")
    };
    let header: Vec<&str> = lines.first()?.split(',').map(|c| c.trim()).collect();
    if let Some(ci) = header.iter().position(|c| key(c)) {
        for l in &lines[1..] {
            if let Some(v) = l.split(',').nth(ci).and_then(|x| x.trim().parse::<f64>().ok()).filter(|v| v.is_finite() && *v > 0.0) {
                return Some((v, format!("completeness.csv column '{}'", header[ci])));
            }
        }
    }
    for l in &lines {
        let cells: Vec<&str> = l.split(',').map(|c| c.trim()).collect();
        if cells.first().is_some_and(|c| key(c))
            && let Some(v) = cells[1..].iter().find_map(|x| x.parse::<f64>().ok().filter(|v| v.is_finite() && *v > 0.0))
        {
            return Some((v, format!("completeness.csv row '{}'", cells[0])));
        }
    }
    resolution_test_m_complete(&lines)
}

/// Completeness from a resolution study in completeness.csv (columns run,M_center,n_per_host): the
/// down-sampled run "downK" (particle mass x K) is converged with the base run at all masses >= M_conv
/// when |n_downK / n_base - 1| <= 0.1 + 2/sqrt(30 n_base) (10% plus 2 sigma Poisson for ~30 hosts) for
/// every bin with n_base >= 1. The same particle number at full resolution gives M_complete = M_conv / K.
fn resolution_test_m_complete(lines: &[&str]) -> Option<(f64, String)> {
    let mut runs: Vec<(String, f64, f64)> = Vec::new();
    for l in lines.iter().skip(1) {
        let c: Vec<&str> = l.split(',').map(|x| x.trim()).collect();
        if c.len() >= 3
            && let (Ok(m), Ok(n)) = (c[1].parse::<f64>(), c[2].parse::<f64>())
        {
            runs.push((c[0].to_string(), m, n));
        }
    }
    let base_name = runs.iter().find(|r| r.0.contains("base")).or(runs.first())?.0.clone();
    let down_name = runs.iter().find(|r| r.0.starts_with("down"))?.0.clone();
    let k: f64 = down_name.trim_start_matches("down").parse().ok().filter(|k: &f64| *k > 1.0)?;
    let base: Vec<(f64, f64)> = runs.iter().filter(|r| r.0 == base_name).map(|r| (r.1, r.2)).collect();
    let down: HashMap<u64, f64> = runs.iter().filter(|r| r.0 == down_name).map(|r| (r.1.to_bits(), r.2)).collect();
    let ok: Vec<(f64, bool)> = base
        .iter()
        .filter(|(_, n)| *n >= 1.0)
        .map(|&(m, n)| {
            let d = down.get(&m.to_bits()).copied().unwrap_or(0.0);
            (m, (d / n - 1.0).abs() <= 0.1 + 2.0 / (30.0 * n).sqrt())
        })
        .collect();
    let first_bad_from_top = ok.iter().rposition(|x| !x.1)?;
    let m_conv = ok.get(first_bad_from_top + 1)?.0;
    Some((
        m_conv / k,
        format!("completeness.csv resolution test: '{down_name}' (particle mass x{k}) converges with '{base_name}' for M >= {m_conv:.0} m_p, so M_complete = {m_conv:.0}/{k}"),
    ))
}

pub fn run(dir: &Path, o_in: &ExpOpts) -> Result<(), String> {
    let t0 = Instant::now();
    let (m_complete, m_complete_src) = match o_in.m_complete {
        Some(v) => (v, "command line (--m-complete)".to_string()),
        None => read_m_complete(dir).unwrap_or((150.0, "default 150 m_p (completeness.csv not found or unparsed)".into())),
    };
    eprintln!("M_complete = {m_complete} particle masses ({m_complete_src})");
    // M_complete replaces the old fit_min everywhere: suppression fits, resolved counts, L_cut floor
    let o_own = ExpOpts { fit_min: m_complete, ..o_in.clone() };
    let o = &o_own;
    let dirs = discover(dir)?;
    let mut models: Vec<Model> = dirs.iter().map(|d| Model::load(d)).collect::<Result<_, _>>()?;
    models.sort_by(|a, b| (!a.is_cdm, a.m_hm).partial_cmp(&(!b.is_cdm, b.m_hm)).unwrap());
    if models.is_empty() || !models[0].is_cdm {
        return Err(format!("{}: need a cdm model directory with meta.json + subhalos/", dir.display()));
    }
    let out = o.out.clone().unwrap_or_else(|| dir.join("analysis"));
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let nm = models.len();
    for m in &models {
        eprintln!(
            "model {:<16} M_hm(2D) = {:>10.4e} mp  k_hm = {:.4}  hosts {}  branches {}  subhalos(a=1) {}",
            m.name,
            m.m_hm,
            m.k_hm,
            m.cat.hosts.len(),
            m.cat.branches.len(),
            m.cat.subs.len()
        );
    }
    // ---------- proto-halo shapes and spurious-halo cleaning (centrals AND subhaloes) ----------
    let snaps: Vec<Option<PathBuf>> =
        models.iter().map(|m| Some(m.dir.join("snap_a1.000.bin")).filter(|p| p.is_file())).collect();
    let shapes: Vec<Option<Vec<f64>>> = models
        .iter()
        .zip(&snaps)
        .map(|(m, s)| {
            let p = s.as_ref()?;
            if m.cat.subs.is_empty() {
                return None;
            }
            let snap = Snapshot::open(p).map_err(|e| eprintln!("warning: {e}")).ok()?;
            let t = Instant::now();
            let v = proto_shapes(&snap, m, 20.0);
            eprintln!("proto-halo shapes for {}: {:.2}s", m.name, t.elapsed().as_secs_f64());
            Some(v)
        })
        .collect();
    // Lovell et al. 2014: s_cut = 1st percentile of CDM (sub)haloes with > 100 particles (0.16 in 3D)
    let s_cut = shapes[0].as_ref().map(|v| {
        let mut x: Vec<f64> = v
            .iter()
            .zip(&models[0].cat.subs)
            .filter(|(s, c)| s.is_finite() && c.n_bound > 100.0)
            .map(|(s, _)| *s)
            .collect();
        pct(&mut x, 0.01)
    });
    // clean = "lovell": M < 0.5 M_lim AND s < s_cut; "shape": s < s_cut at any mass; "none"
    let spurious: Vec<Vec<bool>> = models
        .iter()
        .enumerate()
        .map(|(j, m)| {
            m.cat
                .subs
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    let low_s = match (&shapes[j], s_cut) {
                        (Some(v), Some(c)) => v[i].is_finite() && v[i] < c,
                        _ => false,
                    };
                    !m.is_cdm
                        && match o.clean.as_str() {
                            "none" => false,
                            "shape" => low_s,
                            _ => s.m < wdm::LOVELL_CLEAN * m.m_lim && low_s,
                        }
                })
                .collect()
        })
        .collect();
    // remove flagged surviving subhaloes (branch_id = id of its a = 1 subhalo) from the satellite lists
    let mut n_sub_cleaned = vec![0usize; nm];
    for (j, m) in models.iter_mut().enumerate() {
        let ids: std::collections::HashSet<i64> =
            m.cat.subs.iter().zip(&spurious[j]).filter(|(s, f)| **f && !s.is_central).map(|(s, _)| s.id).collect();
        let br = &m.cat.branches;
        for v in m.sats.iter_mut() {
            let before = v.len();
            v.retain(|&i| !ids.contains(&br[i].id));
            n_sub_cleaned[j] += before - v.len();
        }
    }
    let lows_frac: Vec<Option<f64>> = (0..nm)
        .map(|j| {
            let (v, c) = (shapes[j].as_ref()?, s_cut?);
            let sel: Vec<f64> = models[j].cat.subs.iter().zip(v).filter(|(s, x)| !s.is_central && s.m >= o.fit_min && x.is_finite()).map(|(_, x)| *x).collect();
            (!sel.is_empty()).then(|| sel.iter().filter(|&&x| x < c).count() as f64 / sel.len() as f64)
        })
        .collect();
    let cdm = &models[0];

    // ---------- host selection ----------
    let big: Vec<Vec<usize>> = models.iter().map(|m| m.big_hosts(o.host_min)).collect();
    let mut order = big[0].clone();
    order.sort_by(|&a, &b| cdm.cat.hosts[b].m.total_cmp(&cdm.cat.hosts[a].m));
    if order.is_empty() {
        return Err(format!("no CDM hosts with n_fof >= {}", o.host_min));
    }
    let n_mw = o.n_mw.min(order.len());
    let (mw_lo, mw_hi) = (cdm.cat.hosts[order[n_mw - 1]].m, cdm.cat.hosts[order[0]].m);
    // ---------- host pairing by particle IDs ----------
    let pairing = pair_by_ids(&models, &big, &snaps);
    let pairs: Vec<Vec<Option<usize>>> = pairing.iter().map(|p| p.pairs.clone()).collect();
    // position of each CDM big host in big[0]
    let upos: HashMap<usize, usize> = big[0].iter().enumerate().map(|(u, &h)| (h, u)).collect();
    // paired counterparts of a list of CDM hosts in model j: (cdm host, model-j host)
    let counterparts = |j: usize, cdm_hosts: &[usize]| -> Vec<(usize, usize)> {
        cdm_hosts.iter().filter_map(|&h| upos.get(&h).and_then(|&u| pairs[j][u]).map(|w| (h, w))).collect()
    };
    // MW analogues: the top-n_mw CDM hosts and their PAIRED counterparts in each model
    let mw_cdm: Vec<usize> = order[..n_mw].to_vec();
    let mw: Vec<Vec<usize>> = (0..nm).map(|j| counterparts(j, &mw_cdm).into_iter().map(|p| p.1).collect()).collect();
    // hosts paired in every model (common sample for the mock survey and the likelihood)
    let mw_common: Vec<usize> = mw_cdm.iter().copied().filter(|h| (1..nm).all(|j| !counterparts(j, &[*h]).is_empty())).collect();
    let pair_stats: Vec<Value> = (0..nm)
        .map(|j| {
            let mut f: Vec<f64> = pairing[j].frac.iter().copied().filter(|x| x.is_finite()).collect();
            let npair = pairs[j].iter().filter(|p| p.is_some()).count();
            json!({
                "model": models[j].name, "method": pairing[j].method,
                "n_cdm_hosts": big[0].len(), "n_paired": npair,
                "pairing_fraction": npair as f64 / big[0].len().max(1) as f64,
                "mw_paired": mw[j].len(),
                "shared_frac_percentiles_5_16_50_84_95": [pct(&mut f, 0.05), pct(&mut f, 0.16), pct(&mut f, 0.5), pct(&mut f, 0.84), pct(&mut f, 0.95)],
                "frac_hosts_with_shared_ge_0.5": f.iter().filter(|&&x| x >= 0.5).count() as f64 / f.len().max(1) as f64,
            })
        })
        .collect();
    {
        let mut pc = String::from("model,cdm_host_id,wdm_host_id,shared_frac\n");
        for j in 1..nm {
            for (u, &h) in big[0].iter().enumerate() {
                let w = pairs[j][u].map(|w| models[j].cat.hosts[w].id.to_string()).unwrap_or_default();
                let _ = writeln!(pc, "{},{},{},{}", models[j].name, cdm.cat.hosts[h].id, w, csvf(pairing[j].frac[u]));
            }
        }
        std::fs::write(out.join("pairing.csv"), pc).map_err(|e| e.to_string())?;
    }
    for (j, s) in pair_stats.iter().enumerate().skip(1) {
        eprintln!(
            "pairing {}: {} / {} CDM hosts paired ({:.1}%), MW analogues paired {} / {n_mw}",
            models[j].name,
            s["n_paired"],
            big[0].len(),
            100.0 * s["pairing_fraction"].as_f64().unwrap_or(0.0),
            mw[j].len()
        );
    }
    eprintln!(
        "MW analogues: top {n_mw} CDM hosts, M in [{mw_lo:.4e}, {mw_hi:.4e}] mp; paired in every model: {}",
        mw_common.len()
    );
    // host-mass bins (CDM-defined), plus MW and all
    let hb_edges = log_edges(o.host_min, mw_hi * 1.0001, 0.5);
    let mut host_bins: Vec<(String, f64, f64)> = vec![("MW".into(), mw_lo * 0.999, mw_hi * 1.001), ("all".into(), 0.0, f64::INFINITY)];
    for w in hb_edges.windows(2) {
        host_bins.push((format!("logM_{:.1}-{:.1}", w[0].log10(), w[1].log10()), w[0], w[1]));
    }

    // ---------- SHMF / VF per host bin ----------
    let m_hi = cdm.cat.branches.iter().map(|b| b.m_peak).fold(20.0, f64::max);
    let m_edges = log_edges(20.0, m_hi * 1.01, 0.2);
    let v20 = {
        let mut v: Vec<f64> = cdm.cat.branches.iter().filter(|b| b.m_peak >= 18.0 && b.m_peak <= 23.0).map(|b| b.v_peak).collect();
        let x = pct(&mut v, 0.5);
        if x.is_finite() { x } else { cdm.cat.branches.iter().map(|b| b.v_peak).fold(f64::INFINITY, f64::min) }
    };
    let v_hi = cdm.cat.branches.iter().map(|b| b.v_peak).fold(v20, f64::max);
    let v_edges = log_edges(v20, v_hi * 1.01, 0.1);
    let fit_edges = log_edges(o.fit_min, m_hi * 1.01, 0.15);

    // ---------- suppression fits ----------
    // subhalo: paired CDM hosts (n_fof >= host_min); units = CDM hosts
    let cdm_fit_counts = host_counts(cdm, &big[0], &fit_edges, |b| b.m_peak);
    let wdm_fit_counts: Vec<Vec<Option<Vec<f64>>>> = (1..nm)
        .map(|j| {
            pairs[j]
                .iter()
                .map(|p| p.map(|w| host_counts(&models[j], &[w], &fit_edges, |b| b.m_peak).remove(0)))
                .collect()
        })
        .collect();
    let nfb = fit_edges.len() - 1;
    // bootstrap units: CDM hosts paired in at least one WDM model (pair-level bootstrap)
    let fit_units: Vec<usize> = (0..big[0].len()).filter(|&u| (1..nm).any(|j| pairs[j][u].is_some())).collect();
    let sub_make = |pick: Option<&[usize]>| -> Vec<BinPair> {
        let idx: Vec<usize> = match pick {
            None => fit_units.clone(),
            Some(p) => p.iter().map(|&k| fit_units[k]).collect(),
        };
        let mut d = Vec::new();
        for j in 1..nm {
            let (mut nc, mut nw) = (vec![0.0; nfb], vec![0.0; nfb]);
            for &u in &idx {
                if let Some(wc) = &wdm_fit_counts[j - 1][u] {
                    for b in 0..nfb {
                        nc[b] += cdm_fit_counts[u][b];
                        nw[b] += wc[b];
                    }
                }
            }
            for b in 0..nfb {
                if nc[b] + nw[b] > 0.0 {
                    d.push(BinPair { m_lo: fit_edges[b], m_hi: fit_edges[b + 1], m_hm: models[j].m_hm, n_cdm: nc[b], n_wdm: nw[b] });
                }
            }
        }
        d
    };
    let t_fit = Instant::now();
    let sub_fit = if nm > 1 { fit_with_boot(&sub_make, fit_units.len(), o.n_boot, o.seed) } else { None };

    // ---------- suppression in sigma space ----------
    // x = sigma(M) / sigma(M_hm), both with the CDM linear spectrum in-box (2D top-hat, a = 1; the
    // growth factor cancels). The fit uses the same functional form with alpha M_hm/M -> alpha x,
    // implemented by mapping each bin to pseudo-masses m' = sigma(M_hm)/sigma(M) and M_hm' = 1.
    let sig_e: Vec<f64> = fit_edges.iter().map(|&m| cdm.sigma(m)).collect();
    let sig_hm: Vec<(f64, f64)> = models.iter().skip(1).map(|m| (m.m_hm, cdm.sigma(m.m_hm))).collect();
    let have_sigma = sig_e.iter().all(|s| s.is_finite()) && sig_hm.iter().all(|s| s.1.is_finite());
    let to_sigma = |d: Vec<BinPair>| -> Vec<BinPair> {
        d.into_iter()
            .map(|b| {
                let k = fit_edges.partition_point(|&e| e < b.m_lo * (1.0 - 1e-12)).min(fit_edges.len() - 2);
                let shm = sig_hm.iter().find(|s| s.0 == b.m_hm).map(|s| s.1).unwrap_or(f64::NAN);
                BinPair { m_lo: shm / sig_e[k], m_hi: shm / sig_e[k + 1], m_hm: 1.0, ..b }
            })
            .collect()
    };
    let sub_make_sigma = |pick: Option<&[usize]>| to_sigma(sub_make(pick));
    let sigma_fit = if nm > 1 && have_sigma { fit_with_boot(&sub_make_sigma, fit_units.len(), o.n_boot, o.seed ^ 99) } else { None };

    // halo (central) suppression: spatial 4x4 blocks as bootstrap units
    let nblk = 4usize;
    let centrals_mass = |j: usize| -> Vec<(f64, usize)> {
        let m = &models[j];
        let blk = |x: f64, y: f64| {
            let bx = ((x / m.box_l * nblk as f64).floor() as i64).rem_euclid(nblk as i64) as usize;
            let by = ((y / m.box_l * nblk as f64).floor() as i64).rem_euclid(nblk as i64) as usize;
            by * nblk + bx
        };
        if m.cat.subs.iter().any(|s| s.is_central) {
            m.cat.subs.iter().enumerate().filter(|(i, s)| s.is_central && !spurious[j][*i]).map(|(_, s)| (s.m, blk(s.x, s.y))).collect()
        } else {
            m.cat.hosts.iter().map(|h| (h.m, blk(h.x, h.y))).collect()
        }
    };
    let cen: Vec<Vec<(f64, usize)>> = (0..nm).map(centrals_mass).collect();
    // FoF groups below ~50 particles are mostly unbound noise without a bound central (finder caveat)
    let halo_min = o.fit_min.max(100.0);
    let hfe = log_edges(halo_min, cen[0].iter().map(|c| c.0).fold(halo_min * 1.5, f64::max) * 1.01, 0.15);
    let nhb = hfe.len() - 1;
    let blk_counts: Vec<Vec<Vec<f64>>> = cen
        .iter()
        .map(|c| {
            let mut v = vec![vec![0.0; nhb]; nblk * nblk];
            for &(m, b) in c {
                if let Some(k) = bin_of(&hfe, m) {
                    v[b][k] += 1.0;
                }
            }
            v
        })
        .collect();
    let halo_make = |pick: Option<&[usize]>| -> Vec<BinPair> {
        let all: Vec<usize> = (0..nblk * nblk).collect();
        let idx = pick.unwrap_or(&all);
        let mut d = Vec::new();
        for j in 1..nm {
            for b in 0..nhb {
                let nc: f64 = idx.iter().map(|&u| blk_counts[0][u][b]).sum();
                let nw: f64 = idx.iter().map(|&u| blk_counts[j][u][b]).sum();
                if nc + nw > 0.0 {
                    d.push(BinPair { m_lo: hfe[b], m_hi: hfe[b + 1], m_hm: models[j].m_hm, n_cdm: nc, n_wdm: nw });
                }
            }
        }
        d
    };
    let halo_fit = if nm > 1 { fit_with_boot(&halo_make, nblk * nblk, o.n_boot, o.seed ^ 77) } else { None };
    let t_fit = t_fit.elapsed().as_secs_f64();
    let sub_supp = sub_fit.map(|f| f.best).unwrap_or(wdm::COZMIC1_MPEAK);
    let halo_supp = halo_fit.map(|f| f.best).unwrap_or(wdm::LOVELL2020_CEN);

    // ---------- shmf.csv ----------
    let mut shmf = String::from("model,kind,host_bin,x_lo,x_hi,x_center,n_hosts,n_mean,n_err,ratio_to_cdm,ratio_err,fit\n");
    let kinds: [(&str, &Vec<f64>, KeyFn); 2] =
        [("subhalo_mpeak", &m_edges, |b| b.m_peak), ("subhalo_vpeak", &v_edges, |b| b.v_peak)];
    let mut mw_mean_nsub = vec![0.0; nm];
    for (kind, edges, key) in kinds {
        let nb = edges.len() - 1;
        for (hbl, lo, hi) in &host_bins {
            // CDM hosts in the bin (by CDM mass; MW = the top-n_mw list); other models use the PAIRED
            // counterparts of exactly these hosts
            let cdm_sel: Vec<usize> = if hbl == "MW" {
                mw_cdm.clone()
            } else {
                big[0].iter().copied().filter(|&h| cdm.cat.hosts[h].m >= *lo && cdm.cat.hosts[h].m <= *hi).collect()
            };
            if cdm_sel.is_empty() {
                continue;
            }
            for j in 0..nm {
                let pidx = counterparts(j, &cdm_sel);
                let hs: Vec<usize> = pidx.iter().map(|p| p.1).collect();
                let rows = host_counts(&models[j], &hs, edges, key);
                let mean = mean_rows(&rows, nb);
                let err = boot_mean_err(&rows, nb, o.n_boot, o.seed ^ j as u64);
                if kind == "subhalo_mpeak" && hbl == "MW" {
                    mw_mean_nsub[j] = (0..nb).filter(|&b| edges[b] >= o.fit_min).map(|b| mean[b]).sum();
                }
                // paired ratio
                let (ratio, rerr) = if j == 0 {
                    (vec![1.0; nb], vec![0.0; nb])
                } else {
                    let c: Vec<Vec<f64>> = pidx.iter().map(|&(h, _)| host_counts(cdm, &[h], edges, key).remove(0)).collect();
                    let w: Vec<Vec<f64>> = pidx.iter().map(|&(_, wv)| host_counts(&models[j], &[wv], edges, key).remove(0)).collect();
                    boot_ratio(&c, &w, nb, o.n_boot, o.seed ^ (j as u64 * 31))
                };
                for b in 0..nb {
                    let xc = (edges[b] * edges[b + 1]).sqrt();
                    let dl = (edges[b + 1] / edges[b]).log10();
                    let fit = if kind == "subhalo_mpeak" { sub_supp.f(xc, models[j].m_hm) } else { f64::NAN };
                    let _ = writeln!(
                        shmf,
                        "{},{},{},{},{},{},{},{},{},{},{},{}",
                        models[j].name,
                        kind,
                        hbl,
                        csvf(edges[b]),
                        csvf(edges[b + 1]),
                        csvf(xc),
                        hs.len(),
                        csvf(mean[b] / dl),
                        csvf(err[b] / dl),
                        csvf(ratio[b]),
                        csvf(rerr[b]),
                        csvf(fit)
                    );
                }
            }
        }
    }
    // field halo mass function (centrals), per unit area per dex
    let area = cdm.box_l * cdm.box_l;
    let h_edges = log_edges(20.0, cen[0].iter().map(|c| c.0).fold(21.0, f64::max) * 1.01, 0.2);
    let hn = |j: usize| {
        let mut c = vec![0.0; h_edges.len() - 1];
        for &(m, _) in &cen[j] {
            if let Some(k) = bin_of(&h_edges, m) {
                c[k] += 1.0;
            }
        }
        c
    };
    let hc0 = hn(0);
    for (j, model) in models.iter().enumerate() {
        let c = hn(j);
        for b in 0..c.len() {
            let dl = (h_edges[b + 1] / h_edges[b]).log10();
            let xc = (h_edges[b] * h_edges[b + 1]).sqrt();
            let r = if hc0[b] > 0.0 { c[b] / hc0[b] } else { f64::NAN };
            let re = if c[b] > 0.0 && hc0[b] > 0.0 { r * (1.0 / c[b] + 1.0 / hc0[b]).sqrt() } else { f64::NAN };
            let _ = writeln!(
                shmf,
                "{},halo_mass,field,{},{},{},0,{},{},{},{},{}",
                model.name,
                csvf(h_edges[b]),
                csvf(h_edges[b + 1]),
                csvf(xc),
                csvf(c[b] / area / dl),
                csvf(c[b].sqrt() / area / dl),
                csvf(r),
                csvf(if j == 0 { 0.0 } else { re }),
                csvf(halo_supp.f(xc, model.m_hm))
            );
        }
    }
    std::fs::write(out.join("shmf.csv"), shmf).map_err(|e| e.to_string())?;

    // ---------- suppression_fit.csv + curves ----------
    let mut sf = String::from("kind,source,alpha,alpha_lo,alpha_hi,beta,beta_lo,beta_hi,gamma,gamma_lo,gamma_hi,M50_over_Mhm,M50_lo,M50_hi\n");
    for (kind, fit) in [("subhalo", &sub_fit), ("halo", &halo_fit)] {
        if let Some(f) = fit {
            let _ = writeln!(
                sf,
                "{kind},2D_fit,{},{},{},{},{},{},{},{},{},{},{},{}",
                f.best.alpha, f.lo.alpha, f.hi.alpha, f.best.beta, f.lo.beta, f.hi.beta, f.best.gamma, f.lo.gamma, f.hi.gamma, f.m50, f.m50_lo, f.m50_hi
            );
        }
    }
    for (kind, src, s) in [
        ("subhalo", "Lovell2020b_3D_subhalo_bound_mass_sterile_nu", wdm::LOVELL2020B_SUB_BOUND),
        ("subhalo", "Lovell2014_3D_subhaloes_genuine", wdm::LOVELL2014_SUB),
        ("subhalo", "COZMIC_I_3D_Mpeak_Nadler2025", wdm::COZMIC1_MPEAK),
        ("halo", "Lovell2020_3D_centrals", wdm::LOVELL2020_CEN),
    ] {
        let m50 = wdm::m_half_suppression(&s, 1.0);
        let _ = writeln!(sf, "{kind},{src},{},,,{},,,{},,,{m50},,", s.alpha, s.beta, s.gamma);
    }
    std::fs::write(out.join("suppression_fit.csv"), sf).map_err(|e| e.to_string())?;
    let mut sc = String::from("M_over_Mhm,f_sub_2D,f_halo_2D,f_Lovell2020_sat,f_Lovell2020_cen,f_Lovell2014_sub,f_COZMIC_I_mpeak\n");
    for i in 0..=80 {
        let x = 10f64.powf(-2.0 + 0.05 * i as f64);
        let _ = writeln!(
            sc,
            "{:.5e},{:.5e},{:.5e},{:.5e},{:.5e},{:.5e},{:.5e}",
            x,
            sub_supp.f(x, 1.0),
            halo_supp.f(x, 1.0),
            wdm::LOVELL2020B_SUB_BOUND.f(x, 1.0),
            wdm::LOVELL2020_CEN.f(x, 1.0),
            wdm::LOVELL2014_SUB.f(x, 1.0),
            wdm::COZMIC1_MPEAK.f(x, 1.0)
        );
    }
    std::fs::write(out.join("suppression_curves.csv"), sc).map_err(|e| e.to_string())?;

    // ---------- suppression_sigma.csv ----------
    let n2d = cdm.meta.as_ref().map(|m| m.spectrum.n).unwrap_or(-1.5);
    let n3d = n2d - 1.0;
    let f3 = |s: &Supp, x: f64| s.f(1.0, wdm::mhm_over_m_3d(x, n3d));
    let x50_3d = |s: &Supp| (1.0 / wdm::m_half_suppression(s, 1.0)).powf((n3d + 3.0) / 6.0);
    let sig_supp = sigma_fit.map(|f| f.best);
    let mut ss = String::from(
        "model,M_center,x_sigma_ratio,ratio,ratio_err,f_2D_sigma_fit,f_2D_mass_fit,f_Lovell2014_3D,f_COZMIC_I_3D,f_Lovell2020b_3D\n",
    );
    let mut x_all: Vec<f64> = Vec::new();
    let mut x50_2d_mass: Vec<Value> = Vec::new();
    if have_sigma {
        let paired_units = |j: usize| -> Vec<usize> { fit_units.iter().copied().filter(|&u| pairs[j][u].is_some()).collect() };
        for j in 1..nm {
            let us = paired_units(j);
            let c: Vec<Vec<f64>> = us.iter().map(|&u| cdm_fit_counts[u].clone()).collect();
            let w: Vec<Vec<f64>> = us.iter().map(|&u| wdm_fit_counts[j - 1][u].clone().unwrap()).collect();
            let (ratio, rerr) = boot_ratio(&c, &w, nfb, o.n_boot, o.seed ^ (j as u64 * 131));
            let shm = sig_hm[j - 1].1;
            for b in 0..nfb {
                let mc = (fit_edges[b] * fit_edges[b + 1]).sqrt();
                let x = cdm.sigma(mc) / shm;
                x_all.push(x);
                let _ = writeln!(
                    ss,
                    "{},{},{},{},{},{},{},{},{},{}",
                    models[j].name,
                    csvf(mc),
                    csvf(x),
                    csvf(ratio[b]),
                    csvf(rerr[b]),
                    csvf(sig_supp.map(|s| s.f(1.0, x)).unwrap_or(f64::NAN)),
                    csvf(sub_supp.f(mc, models[j].m_hm)),
                    csvf(f3(&wdm::LOVELL2014_SUB, x)),
                    csvf(f3(&wdm::COZMIC1_MPEAK, x)),
                    csvf(f3(&wdm::LOVELL2020B_SUB_BOUND, x))
                );
            }
            let m50 = wdm::m_half_suppression(&sub_supp, 1.0) * models[j].m_hm;
            x50_2d_mass.push(json!({"model": models[j].name, "x50": cdm.sigma(m50) / shm}));
        }
        let (xl, xh) = (
            x_all.iter().cloned().fold(f64::INFINITY, f64::min).min(0.8),
            x_all.iter().cloned().fold(f64::NEG_INFINITY, f64::max).max(1.3),
        );
        for i in 0..=60 {
            let x = xl + (xh - xl) * i as f64 / 60.0;
            let _ = writeln!(
                ss,
                "curve,nan,{},nan,nan,{},nan,{},{},{}",
                csvf(x),
                csvf(sig_supp.map(|s| s.f(1.0, x)).unwrap_or(f64::NAN)),
                csvf(f3(&wdm::LOVELL2014_SUB, x)),
                csvf(f3(&wdm::COZMIC1_MPEAK, x)),
                csvf(f3(&wdm::LOVELL2020B_SUB_BOUND, x))
            );
        }
    }
    std::fs::write(out.join("suppression_sigma.csv"), ss).map_err(|e| e.to_string())?;
    let sigma_json = json!({
        "variable": "x = sigma_CDM(M) / sigma_CDM(M_hm), in-box 2D top-hat, CDM linear spectrum (growth cancels)",
        "form": "f = [1 + (alpha x)^beta]^gamma",
        "fit_2D": fit_json(&sigma_fit),
        "x50_2D_sigma_fit": sig_supp.map(|s| 1.0 / wdm::m_half_suppression(&s, 1.0)),
        "x50_2D_mass_fit_per_model": x50_2d_mass,
        "x_range_measured": [x_all.iter().cloned().fold(f64::INFINITY, f64::min), x_all.iter().cloned().fold(f64::NEG_INFINITY, f64::max)],
        "mapping_3D": format!("3D sigma ~ M^-(n3+3)/6 with n3 = n2D - 1 = {n3d} (Beacom et al. 1991), so M_hm/M = x^(6/(n3+3)) = x^{:.1}", 6.0 / (n3d + 3.0)),
        "x50_3D": {"Lovell2014": x50_3d(&wdm::LOVELL2014_SUB), "COZMIC_I": x50_3d(&wdm::COZMIC1_MPEAK), "Lovell2020b_bound": x50_3d(&wdm::LOVELL2020B_SUB_BOUND)},
    });

    // ---------- galaxy-halo connection ----------
    let v_res = {
        let mut v: Vec<f64> = cdm.cat.branches.iter().filter(|b| b.m_peak >= 80.0 && b.m_peak <= 125.0).map(|b| b.v_peak).collect();
        let x = pct(&mut v, 0.5);
        if x.is_finite() {
            x
        } else {
            let mut r: Vec<f64> = cdm.cat.branches.iter().map(|b| b.v_peak / b.m_peak.sqrt()).collect();
            pct(&mut r, 0.5) * 10.0
        }
    };
    let m_norm = 100.0;
    // V_peak per sqrt(M_peak) (2D: V^2 ~ M), to translate V50 into a peak mass
    let v_res_per_sqrt_m = {
        let mut r: Vec<f64> = cdm.cat.branches.iter().filter(|b| b.m_peak > 0.0).map(|b| b.v_peak / b.m_peak.sqrt()).collect();
        pct(&mut r, 0.5)
    };
    let cdm_mw_sats: Vec<Sat> = mw[0].iter().flat_map(|&h| cdm.host_sats(h)).collect();
    let log_lcut_tuned = galaxy::tune_lcut(m_norm, v_res, &cdm_mw_sats, mw[0].len(), o.target_sats);
    // never count galaxies fainter than the resolution limit (M_peak = fit_min)
    let log_lcut_res = 2.0 * (o.fit_min / m_norm).log10();
    let log_lcut = log_lcut_tuned.max(log_lcut_res);
    if log_lcut_tuned < log_lcut_res {
        eprintln!("note: target of {} satellites per MW host is not reachable; L_cut set at the resolution limit (M_peak = {} particles)", o.target_sats, o.fit_min);
    }
    let gm = GalModel::new(m_norm, v_res, log_lcut);
    let nlb = gm.nbins();
    let mut lf = String::from("model,host_id,logL_lo,logL_hi,n_expected,n_err\n");
    let mut sat_counts: Vec<(f64, f64)> = Vec::new(); // mean, std per host (fiducial)
    for j in 0..nm {
        let per_host: Vec<Vec<f64>> = mw[j].iter().map(|&h| gm.expected_lf(&models[j].host_sats(h), &FIDUCIAL)).collect();
        for (k, &h) in mw[j].iter().enumerate() {
            for (b, val) in per_host[k].iter().enumerate() {
                let hi = if b + 1 < nlb { gm.lf_edges[b + 1] } else { f64::INFINITY };
                let _ = writeln!(lf, "{},{},{:.3},{:.3},{:.4},", models[j].name, models[j].cat.hosts[h].id, gm.lf_edges[b], hi, val);
            }
        }
        let mean = mean_rows(&per_host, nlb);
        let sd = std_of(&per_host, nlb);
        for b in 0..nlb {
            let hi = if b + 1 < nlb { gm.lf_edges[b + 1] } else { f64::INFINITY };
            let _ = writeln!(lf, "{},stack_mean_per_host,{:.3},{:.3},{:.4},{:.4}", models[j].name, gm.lf_edges[b], hi, mean[b], sd[b]);
        }
        let tot: Vec<f64> = per_host.iter().map(|r| r.iter().sum()).collect();
        let m = tot.iter().sum::<f64>() / tot.len().max(1) as f64;
        let s = (tot.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (tot.len().max(2) - 1) as f64).sqrt();
        sat_counts.push((m, s));
    }
    let target_reachable = log_lcut_tuned >= log_lcut_res;
    // with the cut at the resolution limit these are resolved (not "observable") satellites
    let sat_label = if target_reachable { "observable" } else { "resolved" };
    // mock survey: CDM is the observed universe, on the hosts paired in every model
    let common_cdm_sats: Vec<Sat> = mw_common.iter().flat_map(|&h| cdm.host_sats(h)).collect();
    let mut rng = Rng(o.seed ^ 0xC0FFEE);
    let obs = gm.mock_lf(&common_cdm_sats, &FIDUCIAL, &mut rng);
    for (b, ob) in obs.iter().enumerate() {
        let hi = if b + 1 < nlb { gm.lf_edges[b + 1] } else { f64::INFINITY };
        let _ = writeln!(lf, "cdm,mock_observed_total,{:.3},{:.3},{},", gm.lf_edges[b], hi, ob);
    }
    std::fs::write(out.join("satellites_lf.csv"), lf).map_err(|e| e.to_string())?;

    // ---------- closure test: fitted-f emulator vs direct simulations ----------
    let t_like = Instant::now();
    let mut closure = String::from("model,M_hm,n_hosts,n_direct,n_direct_err,n_direct_host_boot_err,n_emulator,emulator_over_direct,pass\n");
    let mut closure_json = Vec::new();
    let mut all_pass = nm > 1 && sub_fit.is_some();
    for (j, mj) in models.iter().enumerate().skip(1) {
        let pr = counterparts(j, &mw_cdm);
        let d_host: Vec<f64> = pr.iter().map(|&(_, w)| gm.expected_lf(&mj.host_sats(w), &FIDUCIAL).iter().sum()).collect();
        let e_host: Vec<f64> = pr
            .iter()
            .map(|&(c, _)| {
                let mut mu = vec![0.0; nlb];
                for s in cdm.host_sats(c) {
                    gm.add_bin_probs(&s, &FIDUCIAL, sub_supp.f(s.m_peak, mj.m_hm), &mut mu);
                }
                mu.iter().sum()
            })
            .collect();
        let (direct, emu): (f64, f64) = (d_host.iter().sum(), e_host.iter().sum());
        // counting error of the simulated satellite number, and the host-to-host bootstrap of the sum
        let err = direct.max(1.0).sqrt();
        let boot = {
            let rows: Vec<Vec<f64>> = d_host.iter().map(|&x| vec![x]).collect();
            boot_mean_err(&rows, 1, o.n_boot, o.seed ^ (j as u64 * 7))[0] * pr.len() as f64
        };
        let pass = (emu - direct).abs() <= 2.0 * err;
        all_pass &= pass;
        let _ = writeln!(
            closure,
            "{},{},{},{:.3},{:.3},{:.3},{:.3},{:.4},{}",
            mj.name,
            mj.m_hm,
            pr.len(),
            direct,
            err,
            boot,
            emu,
            emu / direct,
            pass
        );
        closure_json.push(json!({"model": mj.name, "M_hm": mj.m_hm, "n_hosts": pr.len(),
            "n_direct": direct, "n_direct_err": err, "n_direct_host_boot_err": boot, "n_emulator": emu,
            "emulator_over_direct": emu / direct, "pass": pass}));
    }
    std::fs::write(out.join("closure.csv"), closure).map_err(|e| e.to_string())?;

    // ---------- likelihoods over M_hm ----------
    let sim_mhm: Vec<f64> = models.iter().filter(|m| !m.is_cdm).map(|m| m.m_hm).collect();
    let m_hm_min_sim = sim_mhm.iter().cloned().fold(f64::INFINITY, f64::min);
    let anchor = m_hm_min_sim / 10.0;
    let (g_lo, g_hi) = if sim_mhm.is_empty() { (10.0, 1e5) } else { (anchor, sim_mhm.iter().cloned().fold(0.0, f64::max) * 30.0) };
    let grid: Vec<f64> = (0..=80).map(|i| g_lo * (g_hi / g_lo).powf(i as f64 / 80.0)).collect();
    // (a) emulator: CDM satellites weighted by the fitted 2D suppression
    let supp_fn = |m: f64, mhm: f64| sub_supp.f(m, mhm);
    let tab_f = LikeTable::build(&gm, &common_cdm_sats, &PRIOR_FIDUCIAL, o.n_mc, o.seed ^ 1);
    let tab_w = LikeTable::build(&gm, &common_cdm_sats, &PRIOR_WIDE, o.n_mc, o.seed ^ 2);
    let ll_f = tab_f.loglike_curve(&obs, &supp_fn, &grid);
    let ll_w = tab_w.loglike_curve(&obs, &supp_fn, &grid);
    let ll_f0 = tab_f.loglike_curve(&obs, &supp_fn, &[0.0])[0];
    let ll_w0 = tab_w.loglike_curve(&obs, &supp_fn, &[0.0])[0];
    let lim = |g: &[f64], ll: &[f64]| (galaxy::posterior_upper_limit(g, ll, 0.95), galaxy::lr_upper_limit(g, ll, 1.353));
    let (post_f, lr_f) = lim(&grid, &ll_f);
    let (post_w, lr_w) = lim(&grid, &ll_w);
    // (b) direct: per-model predictions from each model's own paired satellites (same hosts, same
    //     galaxy-parameter draws), interpolated linearly in (ln M_hm, ln mu). CDM is anchored at
    //     M_hm = min(simulated)/10; no extrapolation beyond the largest simulated M_hm.
    let direct_tabs = |prior: &galaxy::Prior, seed: u64| -> Vec<Vec<Vec<f64>>> {
        (0..nm)
            .map(|j| {
                let sats: Vec<Sat> = counterparts(j, &mw_common).iter().flat_map(|&(_, w)| models[j].host_sats(w)).collect();
                let tb = LikeTable::build(&gm, &sats, prior, o.n_mc, seed);
                (0..o.n_mc).map(|k| tb.mu_total(k)).collect()
            })
            .collect()
    };
    let node_m: Vec<f64> = std::iter::once(anchor).chain(sim_mhm.iter().copied()).collect();
    // per-draw multiplicative count nuisance exp(sys z_k) (finder f_esc systematic), same z_k for all M_hm
    let zk: Vec<f64> = {
        let mut r = Rng(o.seed ^ 0x5157);
        (0..o.n_mc).map(|_| r.normal()).collect()
    };
    let direct_curve = |mus: &[Vec<Vec<f64>>], sys: f64| -> Vec<f64> {
        grid.iter()
            .map(|&m| {
                if nm < 2 || m < node_m[0] * (1.0 - 1e-9) || m > node_m[nm - 1] * (1.0 + 1e-9) {
                    return f64::NAN;
                }
                let i = node_m.partition_point(|&x| x <= m).clamp(1, nm - 1);
                let t = ((m / node_m[i - 1]).ln() / (node_m[i] / node_m[i - 1]).ln()).clamp(0.0, 1.0);
                let lls: Vec<f64> = (0..o.n_mc)
                    .map(|k| {
                        let mu: Vec<f64> = (0..nlb)
                            .map(|b| ((1.0 - t) * mus[i - 1][k][b].max(1e-12).ln() + t * mus[i][k][b].max(1e-12).ln() + sys * zk[k]).exp())
                            .collect();
                        galaxy::poisson_lnl(&obs, &mu)
                    })
                    .collect();
                let mx = lls.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                mx + (lls.iter().map(|l| (l - mx).exp()).sum::<f64>() / lls.len() as f64).ln()
            })
            .collect()
    };
    // suppression-only (conservative) variant: WDM predictions capped at the CDM prediction per bin and
    // per galaxy-parameter draw, so an excess of WDM satellites cannot drive the limit
    let cap = |mut mus: Vec<Vec<Vec<f64>>>| -> Vec<Vec<Vec<f64>>> {
        let c0 = mus[0].clone();
        for m in mus.iter_mut().skip(1) {
            for (mk, ck) in m.iter_mut().zip(&c0) {
                for (x, c) in mk.iter_mut().zip(ck) {
                    *x = x.min(*c);
                }
            }
        }
        mus
    };
    let (ld_f, ld_w, lc_f, lc_w, ls_f, ls_w) = if nm > 1 {
        let (tf, tw) = (direct_tabs(&PRIOR_FIDUCIAL, o.seed ^ 1), direct_tabs(&PRIOR_WIDE, o.seed ^ 2));
        let (a, b) = (direct_curve(&tf, 0.0), direct_curve(&tw, 0.0));
        let (cf, cw) = (cap(tf), cap(tw));
        (a, b, direct_curve(&cf, 0.0), direct_curve(&cw, 0.0), direct_curve(&cf, o.count_sys), direct_curve(&cw, o.count_sys))
    } else {
        let n = vec![f64::NAN; grid.len()];
        (n.clone(), n.clone(), n.clone(), n.clone(), n.clone(), n)
    };
    let sub_lim = |ll: &[f64]| {
        let idx: Vec<usize> = (0..grid.len()).filter(|&i| ll[i].is_finite()).collect();
        if idx.len() < 3 {
            return (f64::NAN, f64::NAN);
        }
        let g: Vec<f64> = idx.iter().map(|&i| grid[i]).collect();
        let l: Vec<f64> = idx.iter().map(|&i| ll[i]).collect();
        lim(&g, &l)
    };
    let (dpost_f, dlr_f) = sub_lim(&ld_f);
    let (dpost_w, dlr_w) = sub_lim(&ld_w);
    let (cpost_f, clr_f) = sub_lim(&lc_f);
    let (cpost_w, clr_w) = sub_lim(&lc_w);
    let (spost_f, slr_f) = sub_lim(&ls_f);
    let (spost_w, slr_w) = sub_lim(&ls_w);
    let mut lk = String::from(
        "M_hm,logL_fiducial,logL_wide,dlogL_fiducial,dlogL_wide,logL_direct_fiducial,logL_direct_wide,dlogL_direct_fiducial,dlogL_direct_wide,dlogL_capped_fiducial,dlogL_capped_wide\n",
    );
    let mx = |v: &[f64]| v.iter().cloned().filter(|x| x.is_finite()).fold(f64::NEG_INFINITY, f64::max);
    let (mf, mw_, mdf, mdw, mcf, mcw) = (mx(&ll_f), mx(&ll_w), mx(&ld_f), mx(&ld_w), mx(&lc_f), mx(&lc_w));
    for i in 0..grid.len() {
        let _ = writeln!(
            lk,
            "{:.6e},{:.4},{:.4},{:.4},{:.4},{},{},{},{},{},{}",
            grid[i],
            ll_f[i],
            ll_w[i],
            ll_f[i] - mf,
            ll_w[i] - mw_,
            csvf(ld_f[i]),
            csvf(ld_w[i]),
            csvf(ld_f[i] - mdf),
            csvf(ld_w[i] - mdw),
            csvf(lc_f[i] - mcf),
            csvf(lc_w[i] - mcw)
        );
    }
    std::fs::write(out.join("likelihood.csv"), lk).map_err(|e| e.to_string())?;
    let lim_json = |post: f64, lr: f64, ok: bool, what: &str| -> Value {
        if !ok {
            return json!({"posterior_95": Value::Null, "likelihood_ratio_95": Value::Null, "status": format!("not reported ({what})")});
        }
        json!({
            "posterior_95": nums(post),
            "likelihood_ratio_95": nums(lr),
            "status": if lr.is_infinite() {
                "likelihood ratio never drops by 1.353 within the grid: no limit below the grid maximum".to_string()
            } else if lr < m_hm_min_sim {
                format!("below the lowest simulated M_hm ({m_hm_min_sim}): set by interpolation towards the CDM anchor (anchor-dependent); quote as M_hm < {m_hm_min_sim}")
            } else {
                "reported".to_string()
            },
        })
    };
    let t_like = t_like.elapsed().as_secs_f64();
    // ---------- radial.csv ----------
    let r_edges: Vec<f64> = (0..=10).map(|i| i as f64 / 10.0).collect();
    let mut rad = String::from("model,host_bin,r_lo,r_hi,n_hosts,n_sub_mean,n_sub_err,n_gal_mean\n");
    for j in 0..nm {
        for (lab, hs) in [("MW", &mw[j]), ("all", &big[j])] {
            let m = &models[j];
            let rows: Vec<Vec<f64>> = hs
                .iter()
                .map(|&h| {
                    let mut c = vec![0.0; 10];
                    for &i in &m.sats[h] {
                        let b = &m.cat.branches[i];
                        if b.surviving && b.m_peak >= o.fit_min {
                            c[((m.r_over[i] * 10.0) as usize).min(9)] += 1.0;
                        }
                    }
                    c
                })
                .collect();
            let gal: Vec<Vec<f64>> = hs
                .iter()
                .map(|&h| {
                    let mut c = vec![0.0; 10];
                    for &i in &m.sats[h] {
                        let mut t = vec![0.0; nlb];
                        gm.add_bin_probs(&m.sat(i), &FIDUCIAL, 1.0, &mut t);
                        c[((m.r_over[i] * 10.0) as usize).min(9)] += t.iter().sum::<f64>();
                    }
                    c
                })
                .collect();
            let (mean, err, gmean) = (mean_rows(&rows, 10), boot_mean_err(&rows, 10, o.n_boot, o.seed ^ 5), mean_rows(&gal, 10));
            for b in 0..10 {
                let _ = writeln!(rad, "{},{lab},{:.2},{:.2},{},{:.4},{:.4},{:.4}", m.name, r_edges[b], r_edges[b + 1], hs.len(), mean[b], err[b], gmean[b]);
            }
        }
    }
    std::fs::write(out.join("radial.csv"), rad).map_err(|e| e.to_string())?;

    // ---------- hosts.csv ----------
    let mut hc = String::from("model,host_id,m_fof,n_fof,R_host,x,y,is_mw,paired_cdm_host_id,n_sub_surviving,n_disrupted,n_gal_fiducial\n");
    for j in 0..nm {
        let m = &models[j];
        let back: HashMap<usize, i64> =
            big[0].iter().enumerate().filter_map(|(u, &c)| pairs[j][u].map(|w| (w, cdm.cat.hosts[c].id))).collect();
        for &h in &big[j] {
            let host = &m.cat.hosts[h];
            let (ns, nd) = m.sats[h].iter().fold((0, 0), |(s, d), &i| {
                let b = &m.cat.branches[i];
                if b.m_peak < o.fit_min { (s, d) } else if b.surviving { (s + 1, d) } else { (s, d + 1) }
            });
            let ng: f64 = gm.expected_lf(&m.host_sats(h), &FIDUCIAL).iter().sum();
            let _ = writeln!(
                hc,
                "{},{},{:.6e},{},{:.4},{:.4},{:.4},{},{},{ns},{nd},{:.3}",
                m.name,
                host.id,
                host.m,
                host.n_fof,
                host.r,
                host.x,
                host.y,
                mw[j].contains(&h) as u8,
                back.get(&h).map(|x| x.to_string()).unwrap_or_default(),
                ng
            );
        }
    }
    std::fs::write(out.join("hosts.csv"), hc).map_err(|e| e.to_string())?;

    // ---------- fragmentation ----------
    let frag: Vec<Value> = (0..nm)
        .map(|j| {
            let m = &models[j];
            let res: Vec<usize> = (0..m.cat.subs.len()).filter(|&i| m.cat.subs[i].is_central && m.cat.subs[i].m >= o.fit_min).collect();
            let below = res.iter().filter(|&&i| m.cat.subs[i].m < m.m_lim).count();
            let flagged = res.iter().filter(|&&i| spurious[j][i]).count();
            let br: Vec<&Branch> = m.cat.branches.iter().filter(|b| b.m_peak >= o.fit_min).collect();
            let br_below = br.iter().filter(|b| b.m_peak < m.m_lim).count();
            let shaped = shapes[j].as_ref().map(|v| v.iter().filter(|x| x.is_finite()).count());
            let low_s = match (&shapes[j], s_cut) {
                (Some(v), Some(c)) => Some(res.iter().filter(|&&i| v[i].is_finite() && v[i] < c).count()),
                _ => None,
            };
            json!({
                "model": m.name, "k_peak": nums(m.k_peak), "M_lim_2D": nums(m.m_lim),
                "M_lim_over_Mhm": if m.is_cdm { Value::Null } else { nums(m.m_lim / m.m_hm) },
                "centrals_resolved": res.len(),
                "frac_centrals_below_Mlim": res.len().gt(&0).then(|| below as f64 / res.len() as f64),
                "frac_branches_below_Mlim": br.len().gt(&0).then(|| br_below as f64 / br.len() as f64),
                "centrals_with_shape": shaped,
                "centrals_with_s_below_cut": low_s,
                "frac_centrals_flagged_spurious": (shaped.is_some() && !res.is_empty()).then(|| flagged as f64 / res.len() as f64),
                "clean_mode": o.clean,
                "surviving_subhaloes_removed_from_hosts": n_sub_cleaned[j],
                "frac_resolved_subhaloes_s_below_cut": lows_frac[j],
                "subhaloes_flagged_spurious": m.cat.subs.iter().zip(&spurious[j]).filter(|(s, f)| **f && !s.is_central).count(),
            })
        })
        .collect();

    // ---------- images ----------
    let mut image_notes = Vec::new();
    if o.png && snaps.iter().any(|s| s.is_some()) {
        let n_box = 1024usize;
        let host0 = &cdm.cat.hosts[order[0]];
        let w = 1.5 * host0.r;
        let nz = 1024usize;
        let mut ranges: Option<((f32, f32), (f32, f32))> = None;
        for (j, s) in snaps.iter().enumerate() {
            let Some(p) = s else { continue };
            let Ok(snap) = Snapshot::open(p) else { continue };
            let pos = snap.positions();
            let m = &models[j];
            let cb = pk::cic_counts(pos, m.box_l, n_box);
            let vb = image::log_density(&cb, m.np / (n_box * n_box) as f64);
            drop(cb);
            let cz = zoom_counts(pos, m.box_l, host0.x, host0.y, w, nz);
            let vz = image::log_density(&cz, m.np / (m.box_l * m.box_l) * (2.0 * w / nz as f64).powi(2));
            let r = *ranges.get_or_insert_with(|| (image::auto_range(&vb), image::auto_range(&vz)));
            image::write_rgb_png(&out.join(format!("box_{}.png", m.name)), &vb, n_box, r.0.0, r.0.1)?;
            image::write_rgb_png(&out.join(format!("host_{}.png", m.name)), &vz, nz, r.1.0, r.1.1)?;
            image_notes.push(json!({"model": m.name, "box_png": format!("box_{}.png", m.name), "host_png": format!("host_{}.png", m.name)}));
        }
        if !image_notes.is_empty() {
            image_notes.push(json!({"zoom_center": [host0.x, host0.y], "zoom_half_width": w, "host_id_cdm": host0.id,
                "color_scale": "log10(rho/rho_bar), range fixed from the first (CDM) model"}));
        }
    }

    // ---------- summary ----------
    let theory: Vec<Value> = models
        .iter()
        .map(|m| {
            let masses: Vec<f64> = (0..=6).map(|i| 10f64.powi(i + 1)).collect();
            json!({
                "model": m.name, "is_cdm": m.is_cdm, "k_hm": m.k_hm, "alpha": m.alpha, "nu": m.nu,
                "M_hm_2D": m.m_hm, "M_hm_meta": m.m_hm_meta, "particle_mass": m.mp, "particles": m.np,
                "sigma_bar_particles_per_area": m.sigma_bar,
                "sigma_CDM_at_M_hm": if m.is_cdm { Value::Null } else { nums(cdm.sigma(m.m_hm)) },
                "sigma_of_M_a1": masses.iter().map(|&x| json!({"M": x, "sigma": nums(m.sigma(x))})).collect::<Vec<_>>(),
                "pred_suppression_at": masses.iter().map(|&x| json!({"M": x,
                    "f_halo_2D_fit": halo_supp.f(x, m.m_hm), "f_Lovell2020b_cen": wdm::LOVELL2020_CEN.f(x, m.m_hm),
                    "f_sub_2D_fit": sub_supp.f(x, m.m_hm), "f_Lovell2014_sub": wdm::LOVELL2014_SUB.f(x, m.m_hm),
                    "f_COZMIC_I": wdm::COZMIC1_MPEAK.f(x, m.m_hm)})).collect::<Vec<_>>(),
            })
        })
        .collect();
    let per_model: Vec<Value> = (0..nm)
        .map(|j| {
            json!({
                "model": models[j].name, "M_hm_2D": models[j].m_hm,
                "n_hosts_ge_host_min": big[j].len(), "n_mw_hosts_paired": mw[j].len(),
                "paired_with_cdm": pairs[j].iter().filter(|p| p.is_some()).count(),
                "mean_surviving_subhalos_per_mw_host_mpeak_ge_M_complete": mw_mean_nsub[j],
                "satellite_label": sat_label,
                "mean_resolved_or_observable_satellites_per_mw_host_fiducial": sat_counts[j].0,
                "mean_observable_satellites_per_mw_host_fiducial": sat_counts[j].0,
                "host_to_host_std": sat_counts[j].1,
            })
        })
        .collect();
    let summary = json!({
        "experiment_dir": dir.display().to_string(),
        "models": models.iter().map(|m| m.name.clone()).collect::<Vec<_>>(),
        "options": {"n_mw": o.n_mw, "target_sats": o.target_sats, "host_min_particles": o.host_min,
            "fit_min_particles": o.fit_min, "halo_fit_min_particles": o.fit_min.max(100.0), "n_boot": o.n_boot, "n_mc": o.n_mc, "seed": o.seed},
        "M_complete": m_complete, "M_complete_source": m_complete_src,
        "units": "masses in particle masses (catalog mass / particle_mass); lengths in box units; velocities as catalogued",
        "theory": theory,
        "pairing": pair_stats,
        "mw_analogues": {"selection": "top n_mw CDM hosts by m_fof; WDM = their particle-ID-paired counterparts",
            "mass_range": [mw_lo, mw_hi], "n_cdm": mw[0].len(), "n_paired_in_all_models": mw_common.len()},
        "suppression": {
            "form": "f(M) = [1 + (alpha M_hm / M)^beta]^gamma, M = M_peak (subhalo) or central mass (halo)",
            "subhalo_fit_sample": "surviving subhaloes within R_host of particle-ID-paired hosts, M_peak >= M_complete; pair-level bootstrap",
            "subhalo_2D_fit": fit_json(&sub_fit), "halo_2D_fit": fit_json(&halo_fit),
            "sigma_space": sigma_json,
            "reference_3D": {
                "Lovell2020b_subhalo_bound_mass_sterile_nu": supp_json(&wdm::LOVELL2020B_SUB_BOUND),
                "Lovell2020b_centrals": supp_json(&wdm::LOVELL2020_CEN),
                "Lovell2014_genuine_subhaloes": supp_json(&wdm::LOVELL2014_SUB),
                "COZMIC_I_Mpeak": supp_json(&wdm::COZMIC1_MPEAK)},
        },
        "fragmentation": {"formula": "M_lim,2D = kappa pi Sigma d / k_peak, kappa = 4*10.1/pi^3 (Wang & White 2007 transcribed)",
            "cleaning": format!("Lovell 2014: flag WDM haloes with M < {} M_lim and proto-halo s < s_cut (CDM 1st percentile, > 100 particles)", wdm::LOVELL_CLEAN),
            "proto_shape_cut_s": s_cut, "clean_mode": o.clean, "per_model": frag},
        "galaxy_model": {
            "V_res": v_res, "M_norm": m_norm, "lum_slope": gm.lum_slope, "log10_L_cut": log_lcut, "target_reachable": target_reachable,
            "satellite_label": sat_label,
            "M_peak_at_L_cut": m_norm * 10f64.powf(log_lcut / gm.lum_slope),
            "M_peak_at_V50_fiducial": (FIDUCIAL.v50 * v_res / v_res_per_sqrt_m).powi(2),
            "regime": if m_norm * 10f64.powf(log_lcut / gm.lum_slope) > 1.5 * (FIDUCIAL.v50 * v_res / v_res_per_sqrt_m).powi(2) { "luminosity-cut limited" } else { "threshold limited" },
            "lf_bin_edges_log10L": gm.lf_edges, "fiducial": {"V50_over_Vres": FIDUCIAL.v50, "sigma_gal_over_V50": FIDUCIAL.sigma_gal,
                "sigma_M_dex": FIDUCIAL.sigma_m, "p_orph": FIDUCIAL.p_orph, "tau_orph_a": FIDUCIAL.tau_orph},
            "prior_fiducial": format!("{PRIOR_FIDUCIAL:?}"), "prior_wide": format!("{PRIOR_WIDE:?}"),
        },
        "satellites_per_model": per_model,
        "closure_test": {
            "definition": "fiducial galaxy model on the top-n_mw CDM hosts' paired counterparts: n_direct = WDM simulation, n_emulator = CDM satellites x fitted 2D f(M_peak); pass if |emulator - direct| <= 2 sqrt(n_direct)",
            "all_pass": all_pass, "per_model": closure_json,
        },
        "mock_constraint": {
            "observed_counts_stacked": obs, "n_observed_total": obs.iter().sum::<f64>(), "n_hosts": mw_common.len(),
            "satellite_label": sat_label,
            "M_hm_grid": [g_lo, g_hi],
            "posterior_prior": "flat in log10 M_hm over the grid: the posterior limit depends on the lower grid bound; prefer the likelihood-ratio limit",
            "emulator_fitted_f": {
                "suppression": supp_json(&sub_supp),
                "logL_cdm_fiducial": ll_f0, "logL_cdm_wide": ll_w0,
                "fiducial_prior": lim_json(post_f, lr_f, all_pass, "emulator fails closure"),
                "wide_threshold_prior": lim_json(post_w, lr_w, all_pass, "emulator fails closure"),
            },
            "direct_interpolation": {
                "method": "per-model predictions from each simulation's own paired satellites, ln mu interpolated linearly in ln M_hm; CDM anchored at min(M_hm)/10; no extrapolation above the largest simulated M_hm",
                "fiducial_prior": lim_json(dpost_f, dlr_f, nm > 1, "no WDM models"),
                "wide_threshold_prior": lim_json(dpost_w, dlr_w, nm > 1, "no WDM models"),
            },
            "direct_interpolation_suppression_only": {
                "method": "as direct_interpolation, but each WDM prediction is capped at the CDM prediction (per LF bin and parameter draw): conservative, insensitive to WDM satellite excesses",
                "fiducial_prior": lim_json(cpost_f, clr_f, nm > 1, "no WDM models"),
                "wide_threshold_prior": lim_json(cpost_w, clr_w, nm > 1, "no WDM models"),
            },
            "direct_interpolation_suppression_only_with_count_systematic": {
                "method": format!("as suppression_only, with a lognormal satellite-count nuisance of {} (finder f_esc systematic) marginalised per draw", o.count_sys),
                "fiducial_prior": lim_json(spost_f, slr_f, nm > 1, "no WDM models"),
                "wide_threshold_prior": lim_json(spost_w, slr_w, nm > 1, "no WDM models"),
            },
            "M_hm_95_posterior_fiducial": if all_pass { nums(post_f) } else { Value::Null },
            "M_hm_95_likelihood_ratio_fiducial": if all_pass { nums(lr_f) } else { Value::Null },
            "M_hm_95_posterior_wide": if all_pass { nums(post_w) } else { Value::Null },
            "M_hm_95_likelihood_ratio_wide": if all_pass { nums(lr_w) } else { Value::Null },
        },
        "images": image_notes,
        "timings_s": {"fits": t_fit, "likelihood": t_like, "total": t0.elapsed().as_secs_f64()},
    });
    std::fs::write(out.join("experiment_summary.json"), serde_json::to_string_pretty(&summary).unwrap())
        .map_err(|e| e.to_string())?;

    // ---------- table ----------
    let nsat_h = if sat_label == "resolved" { "N_res/MW" } else { "N_obs/MW" };
    println!();
    println!(
        "M_complete = {m_complete} m_p ({m_complete_src}); satellites are '{sat_label}' (cut at M_peak = {:.0} m_p)",
        m_norm * 10f64.powf(log_lcut / gm.lum_slope)
    );
    println!("{:<16} {:>11} {:>9} {:>9} {:>9} {:>9} {:>11} {:>11} {:>10}", "model", "M_hm(2D)", "M_lim", "paired%", "MW pairs", "N_sub/MW", nsat_h, "+-host std", "f<M_lim");
    for j in 0..nm {
        let fr = frag[j]["frac_centrals_below_Mlim"].as_f64().unwrap_or(f64::NAN);
        println!(
            "{:<16} {:>11.4e} {:>9.3e} {:>9.1} {:>9} {:>9.1} {:>11.2} {:>11.2} {:>10.3}",
            models[j].name,
            models[j].m_hm,
            models[j].m_lim,
            100.0 * pair_stats[j]["pairing_fraction"].as_f64().unwrap_or(f64::NAN),
            mw[j].len(),
            mw_mean_nsub[j],
            sat_counts[j].0,
            sat_counts[j].1,
            fr
        );
    }
    let pf = |f: &Option<FitSummary>| match f {
        Some(f) => format!(
            "alpha={:.2} [{:.2},{:.2}] beta={:.2} [{:.2},{:.2}] gamma={:.2} [{:.2},{:.2}] M50/Mhm={:.3} [{:.3},{:.3}]{}",
            f.best.alpha,
            f.lo.alpha,
            f.hi.alpha,
            f.best.beta,
            f.lo.beta,
            f.hi.beta,
            f.best.gamma,
            f.lo.gamma,
            f.hi.gamma,
            f.m50,
            f.m50_lo,
            f.m50_hi,
            if f.at_bound.iter().any(|&b| b) { format!("  ** at clamp {:?}", f.at_bound) } else { String::new() }
        ),
        None => "-".into(),
    };
    println!("subhalo suppression 2D fit (mass):  {}", pf(&sub_fit));
    println!("subhalo suppression 2D fit (sigma): {}  (M50 column = 1/x50)", pf(&sigma_fit));
    println!(
        "   x50 = sigma(M50)/sigma(M_hm): 2D sigma fit {:.4}; 3D (n3 = {n3d}) Lovell2014 {:.4}, COZMIC I {:.4}",
        sig_supp.map(|s| 1.0 / wdm::m_half_suppression(&s, 1.0)).unwrap_or(f64::NAN),
        x50_3d(&wdm::LOVELL2014_SUB),
        x50_3d(&wdm::COZMIC1_MPEAK)
    );
    println!(
        "   3D M_peak refs: Lovell2014 M50/Mhm={:.3}, COZMIC I M50/Mhm={:.3}",
        wdm::m_half_suppression(&wdm::LOVELL2014_SUB, 1.0),
        wdm::m_half_suppression(&wdm::COZMIC1_MPEAK, 1.0)
    );
    println!("halo suppression 2D fit:            {}", pf(&halo_fit));
    for c in &closure_json {
        println!(
            "closure {}: direct {:.1} +- {:.1}, emulator {:.1} (ratio {:.3}) -> {}",
            c["model"].as_str().unwrap_or(""),
            c["n_direct"].as_f64().unwrap_or(f64::NAN),
            c["n_direct_err"].as_f64().unwrap_or(f64::NAN),
            c["n_emulator"].as_f64().unwrap_or(f64::NAN),
            c["emulator_over_direct"].as_f64().unwrap_or(f64::NAN),
            if c["pass"].as_bool() == Some(true) { "PASS" } else { "FAIL" }
        );
    }
    let show = |post: f64, lr: f64| format!("posterior {:.4e}, LR {:.4e}", post, lr);
    println!(
        "mock survey ({} {sat_label} satellites, {} hosts): emulator limit {}",
        obs.iter().sum::<f64>(),
        mw_common.len(),
        if all_pass {
            format!("fiducial [{}], wide [{}]", show(post_f, lr_f), show(post_w, lr_w))
        } else {
            "not reported (emulator fails closure)".into()
        }
    );
    println!(
        "   direct-interpolation limit: fiducial [{}], wide [{}]  (inf = no limit below max simulated M_hm)",
        show(dpost_f, dlr_f),
        show(dpost_w, dlr_w)
    );
    println!(
        "   suppression-only (capped) limit: fiducial [{}], wide [{}]",
        show(cpost_f, clr_f),
        show(cpost_w, clr_w)
    );
    println!(
        "   suppression-only + {:.0}% count systematic: fiducial [{}], wide [{}]",
        100.0 * o.count_sys,
        show(spost_f, slr_f),
        show(spost_w, slr_w)
    );
    for j in 1..nm {
        println!(
            "cleaning ({}) {}: {} subhaloes flagged, {} removed from host satellite lists; resolved subhaloes with s < s_cut: {}",
            o.clean,
            models[j].name,
            frag[j]["subhaloes_flagged_spurious"],
            n_sub_cleaned[j],
            lows_frac[j].map(|x| format!("{:.3}", x)).unwrap_or("-".into())
        );
    }
    println!("   CDM resolved subhaloes with s < s_cut: {}", lows_frac[0].map(|x| format!("{:.3}", x)).unwrap_or("-".into()));
    println!("outputs in {}", out.display());
    Ok(())
}