//! Whole-run driver: every snapshot of a model directory -> catalogs, trees, outputs.

use crate::merger::{self, Linker};
use crate::output;
use crate::snapshot::Snapshot;
use crate::units::{Cosmo, RunMeta};
use crate::{Params, find_catalog};
use rayon::prelude::*;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// True when every snapshot listed in `meta.json` exists and is complete.
pub fn run_complete(dir: &Path) -> bool {
    let Ok(text) = std::fs::read_to_string(dir.join("meta.json")) else { return false };
    let Ok(raw) = serde_json::from_str::<Value>(&text) else { return false };
    let Some(n) = raw.get("particles").and_then(|x| x.as_f64()) else { return false };
    let Some(snaps) = raw.get("snapshots").and_then(|x| x.as_array()) else { return false };
    !snaps.is_empty()
        && snaps.iter().all(|s| {
            let a = s.as_f64().or_else(|| s.get("a").and_then(|x| x.as_f64()));
            let file = s
                .get("file")
                .and_then(|f| f.as_str())
                .map(str::to_string)
                .or_else(|| a.map(|a| format!("snap_a{a:.3}.bin")));
            file.and_then(|f| std::fs::metadata(dir.join(f)).ok())
                .is_some_and(|m| m.len() >= 24 + 16 * n as u64)
        })
}

/// Options for studies; the defaults give the production run.
#[derive(Clone, Debug, Default)]
pub struct RunOptions {
    /// Output directory (default `<model_dir>/subhalos`).
    pub out_dir: Option<PathBuf>,
    /// Keep only particles with `hash(ID) % downsample == 0` (1 or 0 = all). The selection
    /// depends only on the ID, so it is the same subset in every snapshot; particle masses
    /// are scaled by N / N_kept.
    pub downsample: u32,
}

/// Deterministic ID hash (SplitMix64 finaliser).
pub fn id_hash(id: u64) -> u64 {
    let mut z = id.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

pub fn run_model(meta: &RunMeta, params: &Params, log: &mut dyn FnMut(&str)) -> Result<Value, String> {
    run_model_with(meta, params, &RunOptions::default(), log)
}

pub fn run_model_with(meta: &RunMeta, params: &Params, opts: &RunOptions, log: &mut dyn FnMut(&str)) -> Result<Value, String> {
    let t_all = Instant::now();
    let out = opts.out_dir.clone().unwrap_or_else(|| meta.dir.join("subhalos"));
    let ds = opts.downsample.max(1) as u64;
    let n_orig = meta.particles;
    // Down-sampled subset (original IDs in increasing order) and the matching meta.
    let keep: Option<Vec<u32>> = (ds > 1).then(|| {
        (0..meta.particles as u32).into_par_iter().filter(|&i| id_hash(i as u64).is_multiple_of(ds)).collect()
    });
    let mut meta_eff = meta.clone();
    if let Some(k) = &keep {
        meta_eff.particles = k.len() as u64;
        meta_eff.particle_mass = meta.particle_mass * meta.particles as f64 / k.len() as f64;
        log(&format!("down-sampling 1/{ds}: {} of {} particles, particle mass x{:.4}", k.len(), meta.particles, meta.particles as f64 / k.len() as f64));
    }
    let meta = &meta_eff;
    std::fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    let mut linker = Linker::new();
    let mut snaps_json = Vec::new();
    let mut a1_summary = json!(null);
    let nsnap = meta.snapshots.len();
    for (si, sinfo) in meta.snapshots.iter().enumerate() {
        let t0 = Instant::now();
        let snap = Snapshot::open(&sinfo.path)?;
        if snap.count as u64 != n_orig {
            return Err(format!("{}: {} particles, meta.json says {n_orig}", sinfo.path.display(), snap.count));
        }
        if ((snap.a as f64) - sinfo.a).abs() > 1e-3 {
            log(&format!("warning: {} header a = {} vs meta {}", sinfo.path.display(), snap.a, sinfo.a));
        }
        let cosmo = Cosmo::new(meta, sinfo.a);
        let cat = match &keep {
            // Particle "IDs" of a down-sampled run are indices into the kept subset.
            Some(k) => {
                let (p, v) = (snap.positions(), snap.velocities());
                let sp: Vec<[f32; 2]> = k.par_iter().map(|&i| p[i as usize]).collect();
                let sv: Vec<[f32; 2]> = k.par_iter().map(|&i| v[i as usize]).collect();
                find_catalog(&sp, &sv, &cosmo, params)?
            }
            None => find_catalog(snap.positions(), snap.velocities(), &cosmo, params)?,
        };
        drop(snap);
        output::write_subhalos(&out, si, &cat, meta.particle_mass)?;
        let n_sat = cat.subs.iter().filter(|s| !s.is_central).count();
        let n_cen = cat.subs.len() - n_sat;
        let bound: u64 = cat.subs.iter().map(|s| s.n_bound as u64).sum();
        if si + 1 == nsnap {
            output::write_hosts(&out, si, &cat, meta.particle_mass)?;
            // Satellite counts of the most massive hosts.
            let mut per_host = vec![0u32; cat.hosts.len()];
            for s in &cat.subs {
                if !s.is_central {
                    per_host[s.host as usize] += 1;
                }
            }
            let top: Vec<Value> = (0..cat.hosts.len().min(10))
                .map(|g| {
                    let h = &cat.hosts[g];
                    let c = (h.central != crate::NONE).then(|| &cat.subs[h.central as usize]);
                    json!({"host_id": g, "n_fof": h.n_fof, "n_central": c.map_or(0, |c| c.n_bound),
                           "vmax_central": c.map_or(0.0, |c| c.vmax), "R_host": h.r_host,
                           "R_50": h.r_alt[0], "R_100": h.r_alt[1],
                           "n_satellites": per_host[g]})
                })
                .collect();
            a1_summary = json!({"a": cat.a, "subhalos": cat.subs.len(), "centrals": n_cen,
                                "satellites": n_sat, "hosts": cat.hosts.len(), "top_hosts": top});
        }
        let secs = t0.elapsed().as_secs_f64();
        log(&format!(
            "a={:.3}: {} FoF groups ({} particles), {} subhalos ({} centrals, {} satellites), {} bound; \
             FoF {:.1}s, SUBFIND {:.1}s, total {:.1}s",
            cat.a,
            cat.hosts.len(),
            cat.n_grouped,
            cat.subs.len(),
            n_cen,
            n_sat,
            bound,
            cat.timings.fof_s,
            cat.timings.subfind_s,
            secs
        ));
        snaps_json.push(json!({
            "a": cat.a, "file": sinfo.path.file_name().map(|f| f.to_string_lossy().into_owned()),
            "fof_groups": cat.hosts.len(), "particles_in_groups": cat.n_grouped,
            "subhalos": cat.subs.len(), "centrals": n_cen, "satellites": n_sat, "bound_particles": bound,
            "seconds": {"fof": cat.timings.fof_s, "subfind": cat.timings.subfind_s, "maps": cat.timings.maps_s, "total": secs}
        }));
        let t_link = Instant::now();
        linker.add(&cat);
        log(&format!("  linked in {:.2}s", t_link.elapsed().as_secs_f64()));
    }
    linker.finish();
    let forest = linker.forest();
    let branches = merger::branches(&linker, &forest);
    output::write_tree(&out, &linker, &forest, meta.particle_mass)?;
    let (rows, surv, disr) = output::write_branches(&out, &linker, &branches, meta.particle_mass)?;
    let st = linker.stats;
    let linked = st.direct + st.skip + st.into_central;
    let searched = linked + st.lost;
    let multi = branches.iter().filter(|b| b.length >= 2).count();
    let summary = json!({
        "model": meta.model(),
        "dir": meta.dir.to_string_lossy(),
        "particles": meta.particles,
        "box": meta.box_size,
        "units": {
            "particle_mass": meta.particle_mass, "particle_mass_source": meta.particle_mass_source,
            "velocity_convention": meta.vel.name(), "velocity_source": meta.vel_source,
            "poisson_coefficient": meta.c_poisson,
            "softening": meta.softening, "softening_source": meta.softening_source,
            "mass_columns": "n x particle_mass",
            "vmax": "physical circular velocity, v_c^2 = C (M(<r) - pi rho_bar r^2) / (2 pi a), M in comoving area units (rho_bar = 1), box units per 1/H0",
            "lengths": "comoving box units",
            "binding_energy": "E = |v - v_bar|^2/2 + (C/(2 pi a)) [m sum_j ln sqrt(r_ij^2+eps^2) - M ln R_ref] - (C/(4a)) (|x - x_c|^2 - R_ref^2), bound if E < 0, R_ref = escape_factor x initial candidate extent",
            "R_host": format!("largest radius around the central's most-bound particle (FoF members) with mean enclosed surface density >= {} x mean", params.host_overdensity),
        },
        "params": {
            "link_b": params.link_b, "n_min": params.n_min, "k_dens": params.k_dens, "n_ngb": params.n_ngb,
            "theta": params.theta, "escape_factor": params.escape_factor, "max_unbind_frac": params.max_unbind_frac,
            "n_tracers": params.n_tracers, "tracer_weight": "1/(rank+1)", "min_link_score": merger::MIN_SCORE,
            "host_overdensity": params.host_overdensity, "threads": rayon::current_num_threads(),
            "downsample": ds, "ids": if ds > 1 { "indices into the ID-hash down-sampled subset" } else { "particle IDs" },
        },
        "snapshots": snaps_json,
        "links": {
            "direct": st.direct, "skip": st.skip, "into_central": st.into_central, "lost": st.lost,
            "searched": searched, "linked_fraction": if searched > 0 { linked as f64 / searched as f64 } else { 0.0 },
            "last_snapshot_subhalos": st.last,
        },
        "branches": {"total": branches.len(), "length_ge_2": multi, "rows_in_branch_file": rows,
                     "surviving": surv, "disrupted": disr},
        "at_last_snapshot": a1_summary,
        "dark_matter": meta.dark_matter.clone(),
        "seconds_total": t_all.elapsed().as_secs_f64(),
    });
    output::write_json(&out.join("subfind_meta.json"), &summary)?;
    Ok(summary)
}
