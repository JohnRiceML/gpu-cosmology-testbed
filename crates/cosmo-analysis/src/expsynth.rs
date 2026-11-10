//! Synthetic CDM/WDM subhalo catalogs with the subhalo-finder's columns (for validation
//! before the real data exists). All models share one random population ("same phases"):
//! WDM catalogs are the CDM catalog thinned with an injected suppression function, using one
//! fixed uniform deviate per object so the thinning is nested across M_hm.
//!
//! 2D scalings used: R_host from M = 200 Sigma pi R^2; V_peak ~ M_peak^(1/2) (in 2D the circular
//! velocity outside a mass M is v^2 = C M / (2 pi), independent of radius).

use crate::catalog::{Branch, Catalog, Host, Sub};
use crate::galaxy::Rng;
use crate::wdm::{Supp, alpha_from_khm, k_hm_from_mhm_2d};
use std::f64::consts::PI;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct SynthSpec {
    pub box_size: f64,
    pub lattice: usize,
    pub n_hosts: usize,
    pub n_field: usize,
    pub host_m_min: f64,
    pub host_m_max: f64,
    pub sub_norm: f64,
    pub seed: u64,
    pub supp_sub: Supp,
    pub supp_cen: Supp,
    /// half-mode masses (particle masses) of the WDM models
    pub m_hm: Vec<f64>,
}

impl Default for SynthSpec {
    fn default() -> Self {
        SynthSpec {
            box_size: 2000.0,
            lattice: 8192,
            n_hosts: 400,
            n_field: 150_000,
            host_m_min: 3000.0,
            host_m_max: 1e7,
            sub_norm: 0.15,
            seed: 2026,
            supp_sub: Supp { alpha: 3.0, beta: 1.3, gamma: -0.6 },
            supp_cen: Supp { alpha: 2.0, beta: 0.9, gamma: -1.1 },
            m_hm: vec![200.0, 1000.0, 5000.0],
        }
    }
}

const M_MIN: f64 = 20.0;

fn pareto(r: &mut Rng, lo: f64, hi: f64, idx: f64) -> f64 {
    // dN/dM ~ M^-(1+idx) on [lo, hi]
    let u = r.uniform();
    let a = lo.powf(-idx);
    let b = hi.powf(-idx);
    (a - u * (a - b)).powf(-1.0 / idx)
}

/// The CDM population plus the per-object keep deviates.
pub struct Population {
    pub cdm: Catalog,
    pub u_branch: Vec<f64>,
    pub u_sub: Vec<f64>,
    pub sigma_bar: f64,
}

pub fn population(spec: &SynthSpec) -> Population {
    let l = spec.box_size;
    let sigma_bar = (spec.lattice * spec.lattice) as f64 / (l * l);
    let mut r = Rng(spec.seed);
    let mut hosts = Vec::new();
    let mut subs = Vec::new();
    let mut u_sub = Vec::new();
    let mut branches = Vec::new();
    let mut u_branch = Vec::new();
    let mut next_id = 1i64;
    for h in 0..spec.n_hosts {
        let m = pareto(&mut r, spec.host_m_min, spec.host_m_max, 1.0);
        let (x, y) = (r.uniform() * l, r.uniform() * l);
        let rh = (m / (200.0 * PI * sigma_bar)).sqrt();
        let cid = next_id;
        next_id += 1;
        let host_id = h as i64 + 1;
        hosts.push(Host { id: host_id, central_id: cid, m, n_fof: m, m_central: 0.8 * m, vmax: (0.8 * m).sqrt(), x, y, r: rh });
        subs.push(Sub { id: cid, host_id, is_central: true, n_bound: 0.8 * m, m: 0.8 * m, x, y, vmax: (0.8 * m).sqrt(), r_half: 0.3 * rh });
        u_sub.push(r.uniform()); // host centrals are thinned like any central (hosts.csv keeps the host)
        // subhalo branches: N(>m) = sub_norm (m / M)^-0.9 on [M_MIN, 0.1 M]
        let mmax = 0.1 * m;
        if mmax <= M_MIN {
            continue;
        }
        let ntot = spec.sub_norm * ((M_MIN / m).powf(-0.9) - (0.1f64).powf(-0.9));
        let n = (ntot + r.uniform()).floor() as usize;
        for _ in 0..n {
            let mp = pareto(&mut r, M_MIN, mmax, 0.9);
            let vp = mp.sqrt() * 10f64.powf(0.05 * r.normal());
            let a_inf = 0.25 + 0.7 * r.uniform();
            let p_dis = (0.1 + 0.5 * (1.0 - a_inf) / 0.75 + 0.2 * (mp / (0.01 * m)).min(1.0)).min(0.95);
            let surviving = r.uniform() > p_dis;
            let rr = 1.3 * rh * r.uniform().powf(if surviving { 2.0 / 3.0 } else { 0.8 });
            let th = 2.0 * PI * r.uniform();
            let strip = 0.1 + 0.7 * r.uniform();
            let id = next_id;
            next_id += 1;
            let b = Branch {
                id,
                host_id,
                surviving,
                m_a1: if surviving { mp * strip } else { 0.0 },
                v_a1: if surviving { vp * strip.powf(0.3) } else { 0.0 },
                m_peak: mp,
                v_peak: vp,
                a_peak: a_inf,
                a_infall: a_inf,
                x: (x + rr * th.cos()).rem_euclid(l),
                y: (y + rr * th.sin()).rem_euclid(l),
                last_a: if surviving { 1.0 } else { a_inf + (1.0 - a_inf) * r.uniform() },
            };
            let uk = r.uniform();
            if surviving {
                subs.push(Sub {
                    id,
                    host_id,
                    is_central: false,
                    n_bound: b.m_a1,
                    m: b.m_a1,
                    x: b.x,
                    y: b.y,
                    vmax: b.v_a1,
                    r_half: 0.02 * rh,
                });
                u_sub.push(f64::NAN); // satellites follow their branch
            }
            branches.push(b);
            u_branch.push(uk);
        }
    }
    // field centrals (not hosts): dN/dlnM ~ M^-1
    for _ in 0..spec.n_field {
        let m = pareto(&mut r, M_MIN, spec.host_m_min, 1.0);
        let id = next_id;
        next_id += 1;
        subs.push(Sub {
            id,
            host_id: -id,
            is_central: true,
            n_bound: m,
            m,
            x: r.uniform() * l,
            y: r.uniform() * l,
            vmax: m.sqrt(),
            r_half: 0.3 * (m / (200.0 * PI * sigma_bar)).sqrt(),
        });
        u_sub.push(r.uniform());
    }
    Population { cdm: Catalog { hosts, branches, subs, tracers: Default::default() }, u_branch, u_sub, sigma_bar }
}

/// WDM catalog with half-mode mass m_hm (nested thinning of the CDM population).
pub fn thin(pop: &Population, spec: &SynthSpec, m_hm: f64) -> Catalog {
    let keep_b: Vec<bool> =
        pop.cdm.branches.iter().zip(&pop.u_branch).map(|(b, &u)| u < spec.supp_sub.f(b.m_peak, m_hm)).collect();
    let kept_ids: std::collections::HashSet<i64> =
        pop.cdm.branches.iter().zip(&keep_b).filter(|(_, k)| **k).map(|(b, _)| b.id).collect();
    let branches = pop.cdm.branches.iter().zip(&keep_b).filter(|(_, k)| **k).map(|(b, _)| b.clone()).collect();
    let subs = pop
        .cdm
        .subs
        .iter()
        .zip(&pop.u_sub)
        .filter(|(s, u)| {
            if s.is_central {
                **u < spec.supp_cen.f(s.m, m_hm)
            } else {
                kept_ids.contains(&s.id)
            }
        })
        .map(|(s, _)| s.clone())
        .collect();
    Catalog { hosts: pop.cdm.hosts.clone(), branches, subs, tracers: Default::default() }
}

pub fn write_experiment(dir: &Path, spec: &SynthSpec) -> Result<(), String> {
    let pop = population(spec);
    let nu = 1.12;
    let mut models: Vec<(String, f64, Catalog)> = vec![("cdm".into(), 0.0, pop.cdm.clone())];
    for &mhm in &spec.m_hm {
        let k = k_hm_from_mhm_2d(mhm, pop.sigma_bar);
        models.push((format!("wdm_khm{k:.3}"), mhm, thin(&pop, spec, mhm)));
    }
    for (name, mhm, cat) in &models {
        let mdir = dir.join(name);
        std::fs::create_dir_all(&mdir).map_err(|e| e.to_string())?;
        let dm = if *mhm > 0.0 {
            let k = k_hm_from_mhm_2d(*mhm, pop.sigma_bar);
            serde_json::json!({"model": "wdm", "k_hm": k, "alpha": alpha_from_khm(k, nu), "nu": nu, "M_hm": mhm})
        } else {
            serde_json::json!({"model": "cdm"})
        };
        let meta = serde_json::json!({
            "box": spec.box_size, "mesh": 4096, "particles": spec.lattice * spec.lattice,
            "lattice": spec.lattice, "seed": spec.seed, "a_init": 0.02, "a_final": 1.0,
            "spectrum": {"shape": "powerlaw_gauss", "A": 2.604764060e-2, "n": -1.0, "R_c": 1.464844},
            "growth": "D=a", "particle_mass": 1.0, "snapshots": [1.0], "dark_matter": dm,
            "synthetic": {"generator": "cosmo-analysis experiment-synth",
                "injected_subhalo_suppression": [spec.supp_sub.alpha, spec.supp_sub.beta, spec.supp_sub.gamma],
                "injected_halo_suppression": [spec.supp_cen.alpha, spec.supp_cen.beta, spec.supp_cen.gamma]},
        });
        std::fs::write(mdir.join("meta.json"), serde_json::to_string_pretty(&meta).unwrap()).map_err(|e| e.to_string())?;
        cat.write(&mdir.join("subhalos"), 1.0)?;
        eprintln!(
            "experiment-synth: {name}: {} hosts, {} branches, {} subhalos",
            cat.hosts.len(),
            cat.branches.len(),
            cat.subs.len()
        );
    }
    Ok(())
}

fn hash_u(id: i64, salt: u64) -> f64 {
    let mut r = Rng((id as u64) ^ salt.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    r.uniform()
}

/// Pseudo-WDM models from a REAL CDM model directory: the CDM catalogs are thinned with a known
/// suppression (fixed per-object hash deviates, nested in M_hm). Hosts are kept; the a = 1 snapshot is
/// hard-linked (CDM particles) so the shape/image code paths run. For real-format validation only.
pub fn inject(cdm_dir: &Path, out: &Path, mhms: &[f64], sub: Supp, cen: Supp) -> Result<(), String> {
    let md = crate::experiment::Model::load(cdm_dir)?;
    let raw: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(cdm_dir.join("meta.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let nu = 1.12;
    let mut list = vec![("cdm".to_string(), 0.0)];
    for &m in mhms {
        list.push((format!("wdm_khm{:.3}", k_hm_from_mhm_2d(m, md.sigma_bar)), m));
    }
    for (name, mhm) in list {
        let mdir = out.join(&name);
        std::fs::create_dir_all(&mdir).map_err(|e| e.to_string())?;
        let cat = if mhm == 0.0 {
            md.cat.clone()
        } else {
            let branches: Vec<Branch> =
                md.cat.branches.iter().filter(|b| hash_u(b.id, 1) < sub.f(b.m_peak, mhm)).cloned().collect();
            let kept: std::collections::HashSet<i64> = branches.iter().map(|b| b.id).collect();
            let subs = md
                .cat
                .subs
                .iter()
                .filter(|s| if s.is_central { hash_u(s.id, 2) < cen.f(s.m, mhm) } else { kept.contains(&s.id) })
                .cloned()
                .collect();
            Catalog { hosts: md.cat.hosts.clone(), branches, subs, tracers: md.cat.tracers.clone() }
        };
        let mut meta = raw.clone();
        meta["dark_matter"] = if mhm > 0.0 {
            let k = k_hm_from_mhm_2d(mhm, md.sigma_bar);
            serde_json::json!({"model": "wdm", "k_hm": k, "alpha": alpha_from_khm(k, nu), "nu": nu, "M_hm": mhm * md.mp,
                "injected": {"subhalo": [sub.alpha, sub.beta, sub.gamma], "halo": [cen.alpha, cen.beta, cen.gamma]}})
        } else {
            serde_json::json!({"model": "cdm"})
        };
        meta["particle_mass"] = serde_json::json!(md.mp);
        std::fs::write(mdir.join("meta.json"), serde_json::to_string_pretty(&meta).unwrap()).map_err(|e| e.to_string())?;
        cat.write(&mdir.join("subhalos"), md.mp)?;
        let snap = cdm_dir.join("snap_a1.000.bin");
        let link = mdir.join("snap_a1.000.bin");
        if snap.is_file() && !link.exists() && std::fs::hard_link(&snap, &link).is_err() {
            eprintln!("note: could not hard-link {}", snap.display());
        }
        eprintln!("experiment-inject: {name}: M_hm = {mhm} mp, {} branches, {} subhalos", cat.branches.len(), cat.subs.len());
    }
    Ok(())
}
