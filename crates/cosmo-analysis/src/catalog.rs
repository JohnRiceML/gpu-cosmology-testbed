//! Subhalo-finder catalogs (hosts, branches, a = 1 subhaloes): loading and writing.
//! Masses are converted to particle-mass units on load (mass / particle_mass).

use crate::csvtab::Table;
use std::fmt::Write as _;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct Host {
    pub id: i64,
    pub central_id: i64,
    pub m: f64,
    pub n_fof: f64,
    pub m_central: f64,
    pub vmax: f64,
    pub x: f64,
    pub y: f64,
    pub r: f64,
}

#[derive(Clone, Debug)]
pub struct Branch {
    pub id: i64,
    pub host_id: i64,
    pub surviving: bool,
    pub m_a1: f64,
    pub v_a1: f64,
    pub m_peak: f64,
    pub v_peak: f64,
    pub a_peak: f64,
    pub a_infall: f64,
    pub x: f64,
    pub y: f64,
    pub last_a: f64,
}

#[derive(Clone, Debug)]
pub struct Sub {
    pub id: i64,
    pub host_id: i64,
    pub is_central: bool,
    pub n_bound: f64,
    pub m: f64,
    pub x: f64,
    pub y: f64,
    pub vmax: f64,
    pub r_half: f64,
}

#[derive(Clone, Debug, Default)]
pub struct Catalog {
    pub hosts: Vec<Host>,
    pub branches: Vec<Branch>,
    /// a = 1 subhalo catalogue (centrals + surviving satellites), if present
    pub subs: Vec<Sub>,
    /// most-bound particle IDs per a = 1 subhalo id (from subhalos_a1.000_extra.csv), if present
    pub tracers: std::collections::HashMap<i64, Vec<u32>>,
}

fn col_or(t: &Table, name: &str, default: f64) -> Vec<f64> {
    t.f(name).map(|v| v.to_vec()).unwrap_or_else(|_| vec![default; t.len])
}

impl Catalog {
    pub fn load(subdir: &Path, mp: f64) -> Result<Catalog, String> {
        let h = Table::read(&subdir.join("hosts_a1.000.csv"))?;
        let (hid, cid, mf, nf) = (h.f("host_id")?, col_or(&h, "central_subhalo_id", -1.0), h.f("m_fof")?, col_or(&h, "n_fof", f64::NAN));
        let (mc, vc, hx, hy, hr) = (
            col_or(&h, "m_central", f64::NAN),
            col_or(&h, "vmax_central", f64::NAN),
            h.f("x")?,
            h.f("y")?,
            h.f("r_host")?,
        );
        let hosts = (0..h.len)
            .map(|i| Host {
                id: hid[i] as i64,
                central_id: cid[i] as i64,
                m: mf[i] / mp,
                n_fof: if nf[i].is_finite() { nf[i] } else { mf[i] / mp },
                m_central: mc[i] / mp,
                vmax: vc[i],
                x: hx[i],
                y: hy[i],
                r: hr[i],
            })
            .collect();
        let b = Table::read(&subdir.join("branches_a1.000.csv"))?;
        let st = b.s("status")?;
        let (bid, bh, ma, va) = (b.f("branch_id")?, b.f("host_id_at_a1")?, col_or(&b, "mass_a1", f64::NAN), col_or(&b, "vmax_a1", f64::NAN));
        let (mpk, vpk, apk, ainf) = (b.f("m_peak")?, b.f("v_peak")?, col_or(&b, "a_peak", f64::NAN), col_or(&b, "a_infall", f64::NAN));
        let (bx, by, la) = (b.f("x")?, b.f("y")?, col_or(&b, "last_a", 1.0));
        let branches = (0..b.len)
            .map(|i| Branch {
                id: bid[i] as i64,
                host_id: bh[i] as i64,
                surviving: st[i].trim().eq_ignore_ascii_case("surviving"),
                m_a1: ma[i] / mp,
                v_a1: va[i],
                m_peak: mpk[i] / mp,
                v_peak: vpk[i],
                a_peak: apk[i],
                a_infall: ainf[i],
                x: bx[i],
                y: by[i],
                last_a: la[i],
            })
            .collect();
        let mut subs = Vec::new();
        let sp = subdir.join("subhalos_a1.000.csv");
        if sp.is_file() {
            let s = Table::read(&sp)?;
            let (id, hid, ic, nb, m) = (s.f("id")?, s.f("host_id")?, s.f("is_central")?, col_or(&s, "n_bound", f64::NAN), s.f("mass")?);
            let (x, y, vm, rh) = (s.f("x")?, s.f("y")?, col_or(&s, "vmax", f64::NAN), col_or(&s, "r_half", f64::NAN));
            subs = (0..s.len)
                .map(|i| Sub {
                    id: id[i] as i64,
                    host_id: hid[i] as i64,
                    is_central: ic[i] > 0.5,
                    n_bound: if nb[i].is_finite() { nb[i] } else { m[i] / mp },
                    m: m[i] / mp,
                    x: x[i],
                    y: y[i],
                    vmax: vm[i],
                    r_half: rh[i],
                })
                .collect();
        }
        let mut tracers = std::collections::HashMap::new();
        let xp = subdir.join("subhalos_a1.000_extra.csv");
        if xp.is_file() {
            let x = Table::read(&xp)?;
            let (ids, tr) = (x.f("id")?, x.s("most_bound_ids")?);
            for (i, s) in tr.iter().enumerate() {
                let v: Vec<u32> = s.split_whitespace().filter_map(|t| t.parse().ok()).collect();
                if !v.is_empty() {
                    tracers.insert(ids[i] as i64, v);
                }
            }
        }
        Ok(Catalog { hosts, branches, subs, tracers })
    }

    pub fn write(&self, subdir: &Path, mp: f64) -> Result<(), String> {
        std::fs::create_dir_all(subdir).map_err(|e| e.to_string())?;
        let mut s = String::from("host_id,central_subhalo_id,m_fof,n_fof,m_central,vmax_central,x,y,R_host\n");
        for h in &self.hosts {
            let _ = writeln!(
                s,
                "{},{},{:.6e},{},{:.6e},{:.5},{:.5},{:.5},{:.5}",
                h.id, h.central_id, h.m * mp, h.n_fof.round(), h.m_central * mp, h.vmax, h.x, h.y, h.r
            );
        }
        std::fs::write(subdir.join("hosts_a1.000.csv"), s).map_err(|e| e.to_string())?;
        let mut s = String::from(
            "branch_id,host_id_at_a1,status,mass_a1,vmax_a1,m_peak,v_peak,a_peak,a_infall,x,y,vx,vy,last_a\n",
        );
        for b in &self.branches {
            let _ = writeln!(
                s,
                "{},{},{},{:.6e},{:.5},{:.6e},{:.5},{:.4},{:.4},{:.5},{:.5},0,0,{:.4}",
                b.id,
                b.host_id,
                if b.surviving { "surviving" } else { "disrupted" },
                b.m_a1 * mp,
                b.v_a1,
                b.m_peak * mp,
                b.v_peak,
                b.a_peak,
                b.a_infall,
                b.x,
                b.y,
                b.last_a
            );
        }
        std::fs::write(subdir.join("branches_a1.000.csv"), s).map_err(|e| e.to_string())?;
        let mut s = String::from("id,host_id,is_central,n_bound,mass,x,y,vx,vy,vmax,rmax,sigma_v,r_half,a\n");
        for c in &self.subs {
            let _ = writeln!(
                s,
                "{},{},{},{},{:.6e},{:.5},{:.5},0,0,{:.5},{:.5},0,{:.5},1.000",
                c.id,
                c.host_id,
                c.is_central as u8,
                c.n_bound.round(),
                c.m * mp,
                c.x,
                c.y,
                c.vmax,
                2.0 * c.r_half,
                c.r_half
            );
        }
        std::fs::write(subdir.join("subhalos_a1.000.csv"), s).map_err(|e| e.to_string())?;
        if !self.tracers.is_empty() {
            let mut s = String::from("id,x_com,y_com,x_most_bound,y_most_bound,most_bound_ids\n");
            for c in &self.subs {
                if let Some(t) = self.tracers.get(&c.id) {
                    let ids: Vec<String> = t.iter().map(|x| x.to_string()).collect();
                    let _ = writeln!(s, "{},{:.4},{:.4},{:.4},{:.4},{}", c.id, c.x, c.y, c.x, c.y, ids.join(" "));
                }
            }
            std::fs::write(subdir.join("subhalos_a1.000_extra.csv"), s).map_err(|e| e.to_string())?;
        }
        std::fs::write(
            subdir.join("tree.csv"),
            "tree_id,snap_index,a,subhalo_id,desc_id,is_main_progenitor,mass,vmax\n",
        )
        .map_err(|e| e.to_string())
    }
}
