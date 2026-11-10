//! CSV/JSON outputs (the contract with the analysis side). All files go to
//! `<model_dir>/subhalos/`.
//!
//! - `subhalos_a{a:.3}.csv`: `id,host_id,is_central,n_bound,mass,x,y,vx,vy,vmax,rmax,sigma_v,r_half,a`
//!   (x,y = minimum-potential particle, SUBFIND's position; vx,vy = mean peculiar velocity;
//!   vmax physical, rmax/r_half comoving, measured from x,y; host_id = FoF group index at
//!   that snapshot, groups sorted by size).
//! - `subhalos_a{a:.3}_extra.csv`: `id,x_com,y_com,x_most_bound,y_most_bound,most_bound_ids`
//!   (space-separated IDs, most bound first) - centre of mass, lowest-energy particle and
//!   tracer particles, kept out of the contract file.
//! - `tree.csv`: `tree_id,snap_index,a,subhalo_id,desc_id,is_main_progenitor,mass,vmax`
//!   (desc_id = -1 if none; tree_id = id of the last subhalo of the descendant chain).
//! - `branches_a1.000.csv` (named after the last snapshot): `branch_id,host_id_at_a1,status,
//!   mass_a1,vmax_a1,m_peak,v_peak,a_peak,a_infall,x,y,vx,vy,last_a` (branch_id = id of the
//!   branch's last subhalo; a_infall = -1 if never a satellite; mass_a1 = vmax_a1 = 0 if
//!   disrupted; x..vx,vy,last_a = last known state).
//! - `hosts_a1.000.csv`: `host_id,central_subhalo_id,m_fof,n_fof,m_central,vmax_central,x,y,R_host`.

use crate::merger::{Branch, Forest, Linker, global_id};
use crate::{Catalog, NONE};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

fn write_file(path: &Path, text: &str) -> Result<(), String> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, text).map_err(|e| format!("{}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("{}: {e}", path.display()))
}

pub fn a_tag(a: f64) -> String {
    format!("{a:.3}")
}

pub fn subhalo_file(dir: &Path, a: f64) -> PathBuf {
    dir.join(format!("subhalos_a{}.csv", a_tag(a)))
}

pub fn write_subhalos(dir: &Path, snap: usize, cat: &Catalog, particle_mass: f64) -> Result<(), String> {
    let mut s = String::with_capacity(cat.subs.len() * 160 + 128);
    s.push_str("id,host_id,is_central,n_bound,mass,x,y,vx,vy,vmax,rmax,sigma_v,r_half,a\n");
    let mut e = String::with_capacity(cat.subs.len() * 260 + 64);
    e.push_str("id,x_com,y_com,x_most_bound,y_most_bound,most_bound_ids\n");
    for (i, r) in cat.subs.iter().enumerate() {
        let id = global_id(snap, i as u32);
        let _ = writeln!(
            s,
            "{id},{},{},{},{:.6e},{:.4},{:.4},{:.5e},{:.5e},{:.5e},{:.5e},{:.5e},{:.5e},{}",
            r.host,
            r.is_central as u8,
            r.n_bound,
            r.n_bound as f64 * particle_mass,
            r.pos[0],
            r.pos[1],
            r.vel[0],
            r.vel[1],
            r.vmax,
            r.rmax,
            r.sigma_v,
            r.r_half,
            a_tag(cat.a)
        );
        let ids: Vec<String> = r.tracers.iter().map(|t| t.to_string()).collect();
        let _ = writeln!(
            e,
            "{id},{:.4},{:.4},{:.4},{:.4},{}",
            r.com[0],
            r.com[1],
            r.pos_most_bound[0],
            r.pos_most_bound[1],
            ids.join(" ")
        );
    }
    write_file(&subhalo_file(dir, cat.a), &s)?;
    write_file(&dir.join(format!("subhalos_a{}_extra.csv", a_tag(cat.a))), &e)
}

pub fn write_tree(dir: &Path, linker: &Linker, forest: &Forest, particle_mass: f64) -> Result<(), String> {
    let mut s = String::from("tree_id,snap_index,a,subhalo_id,desc_id,is_main_progenitor,mass,vmax\n");
    for (si, sn) in linker.snaps.iter().enumerate() {
        for i in 0..sn.len() {
            let (rs, ri) = forest.root[si][i];
            let desc = linker.desc[si][i].map_or(-1i64, |d| global_id(d.snap, d.idx) as i64);
            let _ = writeln!(
                s,
                "{},{si},{},{},{desc},{},{:.6e},{:.5e}",
                global_id(rs, ri),
                a_tag(sn.a),
                global_id(si, i as u32),
                forest.is_mp[si][i] as u8,
                sn.n_bound[i] as f64 * particle_mass,
                sn.vmax[i]
            );
        }
    }
    write_file(&dir.join("tree.csv"), &s)
}

/// Writes the branch file; returns (rows, surviving, disrupted).
pub fn write_branches(dir: &Path, linker: &Linker, branches: &[Branch], particle_mass: f64) -> Result<(usize, usize, usize), String> {
    let last_a = linker.snaps.last().map_or(1.0, |s| s.a);
    let mut s = String::from(
        "branch_id,host_id_at_a1,status,mass_a1,vmax_a1,m_peak,v_peak,a_peak,a_infall,x,y,vx,vy,last_a\n",
    );
    let (mut rows, mut surv, mut disr) = (0, 0, 0);
    for b in branches {
        let Some(host) = b.host_at_last else { continue };
        rows += 1;
        if b.surviving {
            surv += 1;
        } else {
            disr += 1;
        }
        let _ = writeln!(
            s,
            "{},{host},{},{:.6e},{:.5e},{:.6e},{:.5e},{},{},{:.4},{:.4},{:.5e},{:.5e},{}",
            global_id(b.end.0, b.end.1),
            if b.surviving { "surviving" } else { "disrupted" },
            b.mass_last * particle_mass,
            b.vmax_last,
            b.n_peak as f64 * particle_mass,
            b.v_peak,
            a_tag(b.a_peak),
            b.a_infall.map_or("-1".to_string(), a_tag),
            b.pos[0],
            b.pos[1],
            b.vel[0],
            b.vel[1],
            a_tag(b.last_a)
        );
    }
    write_file(&dir.join(format!("branches_a{}.csv", a_tag(last_a))), &s)?;
    Ok((rows, surv, disr))
}

pub fn write_hosts(dir: &Path, snap: usize, cat: &Catalog, particle_mass: f64) -> Result<(), String> {
    let mut s = String::from("host_id,central_subhalo_id,m_fof,n_fof,m_central,vmax_central,x,y,R_host\n");
    for (g, h) in cat.hosts.iter().enumerate() {
        let (cid, mc, vc) = if h.central != NONE {
            let c = &cat.subs[h.central as usize];
            (global_id(snap, h.central) as i64, c.n_bound as f64 * particle_mass, c.vmax)
        } else {
            (-1, 0.0, 0.0)
        };
        let _ = writeln!(
            s,
            "{g},{cid},{:.6e},{},{:.6e},{:.5e},{:.4},{:.4},{:.4}",
            h.n_fof as f64 * particle_mass,
            h.n_fof,
            mc,
            vc,
            h.pos[0],
            h.pos[1],
            h.r_host
        );
    }
    write_file(&dir.join(format!("hosts_a{}.csv", a_tag(cat.a))), &s)
}

pub fn write_json(path: &Path, v: &serde_json::Value) -> Result<(), String> {
    write_file(path, &serde_json::to_string_pretty(v).map_err(|e| e.to_string())?)
}
