//! Completeness and f_esc sensitivity study from finished runs.
//!
//! `cargo run --release --example completeness -- <experiment_dir>`
//!
//! Expects the production CDM run in `<exp>/cdm/subhalos` (f_esc = e, n_min = 20) and the
//! study runs in `<exp>/subfind_studies/cdm_{fesc2,fesc4,nmin40,nmin80,down2}`.
//!
//! - MW-mass hosts: the 30 most massive FoF hosts at a = 1 (by FoF mass), as in the
//!   analysis (`mw_analogues`).
//! - SHMF: surviving satellites (a = 1 subhalo with is_central = 0) of those hosts, binned in
//!   M_peak expressed in full-resolution particle masses (0.2 dex bins), per host.
//! - Writes `<exp>/completeness.csv` (run,M_center,n_per_host) and prints the cumulative
//!   comparison used to choose M_complete, plus the f_esc table (subhalos matched to the
//!   production run through their tracer particles).

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

const N_MW: usize = 30;

fn read_csv(path: &Path) -> (Vec<String>, Vec<Vec<String>>) {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut lines = text.lines();
    let head: Vec<String> = lines.next().unwrap().split(',').map(str::to_string).collect();
    let rows = lines.filter(|l| !l.is_empty()).map(|l| l.split(',').map(str::to_string).collect()).collect();
    (head, rows)
}

fn col(head: &[String], name: &str) -> usize {
    head.iter().position(|h| h == name).unwrap_or_else(|| panic!("no column {name}"))
}

struct Run {
    name: String,
    /// subhalo id -> (is_central, n_bound, host)
    subs: HashMap<u64, (bool, u64, u64)>,
    /// surviving branch id (= a1 subhalo id) -> m_peak in full-res particle masses
    mpeak: HashMap<u64, f64>,
    /// MW host ids
    mw: Vec<u64>,
    /// tracer id -> subhalo id (a = 1)
    tracers: HashMap<u64, u64>,
    /// subhalo id -> tracer ids (a = 1)
    sub_tracers: HashMap<u64, Vec<u64>>,
    /// mass per output particle relative to full resolution
    mass_scale: f64,
}

fn load(name: &str, dir: &Path, m0: f64) -> Run {
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("subfind_meta.json")).unwrap()).unwrap();
    let pm = meta["units"]["particle_mass"].as_f64().unwrap();
    let (h, rows) = read_csv(&dir.join("subhalos_a1.000.csv"));
    let (ci, cc, cn, ch) = (col(&h, "id"), col(&h, "is_central"), col(&h, "n_bound"), col(&h, "host_id"));
    let subs = rows
        .iter()
        .map(|r| (r[ci].parse().unwrap(), (r[cc] == "1", r[cn].parse().unwrap(), r[ch].parse().unwrap())))
        .collect();
    let (h, rows) = read_csv(&dir.join("branches_a1.000.csv"));
    let (bi, bs, bm) = (col(&h, "branch_id"), col(&h, "status"), col(&h, "m_peak"));
    let mpeak = rows
        .iter()
        .filter(|r| r[bs] == "surviving")
        .map(|r| (r[bi].parse().unwrap(), r[bm].parse::<f64>().unwrap() / m0))
        .collect();
    let (h, rows) = read_csv(&dir.join("hosts_a1.000.csv"));
    let (hi, hm) = (col(&h, "host_id"), col(&h, "m_fof"));
    let mut hosts: Vec<(f64, u64)> = rows.iter().map(|r| (r[hm].parse().unwrap(), r[hi].parse().unwrap())).collect();
    hosts.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let mw = hosts.iter().take(N_MW).map(|x| x.1).collect();
    let (h, rows) = read_csv(&dir.join("subhalos_a1.000_extra.csv"));
    let (ei, et) = (col(&h, "id"), col(&h, "most_bound_ids"));
    let mut tracers = HashMap::new();
    let mut sub_tracers = HashMap::new();
    for r in &rows {
        let id: u64 = r[ei].parse().unwrap();
        let t: Vec<u64> = r[et].split_whitespace().map(|x| x.parse().unwrap()).collect();
        for &x in &t {
            tracers.insert(x, id);
        }
        sub_tracers.insert(id, t);
    }
    Run { name: name.to_string(), subs, mpeak, mw, tracers, sub_tracers, mass_scale: pm / m0 }
}

/// M_peak (full-res particle masses) of surviving satellites in MW hosts.
fn shmf_masses(r: &Run) -> Vec<f64> {
    let mw: std::collections::HashSet<u64> = r.mw.iter().copied().collect();
    r.mpeak
        .iter()
        .filter(|(id, _)| r.subs.get(id).is_some_and(|s| !s.0 && mw.contains(&s.2)))
        .map(|(_, &m)| m)
        .collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() {
    let exp = PathBuf::from(std::env::args().nth(1).expect("experiment dir"));
    let m0 = 0.1059638129;
    let studies = exp.join("subfind_studies");
    let runs = vec![
        load("base_fesc_e_nmin20", &exp.join("cdm").join("subhalos"), m0),
        load("fesc2", &studies.join("cdm_fesc2"), m0),
        load("fesc4", &studies.join("cdm_fesc4"), m0),
        load("nmin40", &studies.join("cdm_nmin40"), m0),
        load("nmin80", &studies.join("cdm_nmin80"), m0),
        load("down2", &studies.join("cdm_down2"), m0),
    ];

    // ---- f_esc sensitivity (subhalos matched to the production run by tracer votes) ----
    let base = &runs[0];
    println!("f_esc sensitivity at a = 1 (CDM). Ratios are run / production (f = e), medians over matched subhalos.");
    println!("{:<20} {:>7} {:>7} {:>8} {:>14} {:>14} {:>12}", "run", "N_sub", "N_sat", "matched", "med Mb ratio", "med Mpk ratio", "MW sats");
    for r in runs.iter().take(3) {
        let (mut rb, mut rp, mut matched) = (Vec::new(), Vec::new(), 0usize);
        for (id, tr) in &r.sub_tracers {
            let mut votes: HashMap<u64, f64> = HashMap::new();
            for (k, t) in tr.iter().enumerate() {
                if let Some(&b) = base.tracers.get(t) {
                    *votes.entry(b).or_default() += 1.0 / (k as f64 + 1.0);
                }
            }
            let Some((&b, _)) = votes.iter().max_by(|a, b| a.1.total_cmp(b.1).then(b.0.cmp(a.0))) else { continue };
            matched += 1;
            rb.push(r.subs[id].1 as f64 * r.mass_scale / (base.subs[&b].1 as f64 * base.mass_scale));
            if let (Some(p), Some(q)) = (r.mpeak.get(id), base.mpeak.get(&b)) {
                rp.push(p / q);
            }
        }
        let nsat = r.subs.values().filter(|s| !s.0).count();
        println!(
            "{:<20} {:>7} {:>7} {:>8} {:>14.3} {:>14.3} {:>12}",
            r.name,
            r.subs.len(),
            nsat,
            matched,
            median(rb),
            median(rp),
            shmf_masses(r).len()
        );
    }

    // ---- SHMF per MW host, 0.2 dex bins in M_peak (full-res particle masses) ----
    let edges: Vec<f64> = (0..=19).map(|k| 10f64.powf(1.0 + 0.2 * k as f64)).collect();
    let mut csv = String::from("run,M_center,n_per_host\n");
    let mut cum: Vec<(String, Vec<f64>)> = Vec::new();
    for r in &runs {
        let m = shmf_masses(r);
        let mut c = Vec::new();
        for w in edges.windows(2) {
            let n = m.iter().filter(|&&x| x >= w[0] && x < w[1]).count();
            let _ = writeln!(csv, "{},{:.4e},{:.4}", r.name, (w[0] * w[1]).sqrt(), n as f64 / N_MW as f64);
        }
        for &e in &edges[..edges.len() - 1] {
            c.push(m.iter().filter(|&&x| x >= e).count() as f64 / N_MW as f64);
        }
        cum.push((r.name.clone(), c));
    }
    std::fs::write(exp.join("completeness.csv"), csv).unwrap();

    println!("\nCumulative satellites per MW host N(>M_peak), M_peak in full-res particle masses:");
    print!("{:>9}", "M_peak");
    for (n, _) in &cum {
        print!(" {:>18}", n);
    }
    println!();
    for (k, &e) in edges[..edges.len() - 1].iter().enumerate() {
        print!("{:>9.1}", e);
        for (_, c) in &cum {
            let rel = if cum[0].1[k] > 0.0 { c[k] / cum[0].1[k] - 1.0 } else { f64::NAN };
            print!(" {:>9.2} ({:>+5.0}%)", c[k], 100.0 * rel);
        }
        println!();
    }
    // M_complete: smallest edge from which on the 2x-degraded runs (n_min doubled; half the
    // particles) agree with the production run in N(>M) within max(10%, 2 sigma Poisson of the
    // production count), over all edges with >= 3 satellites per host (sparser bins are noise).
    let m_complete = |check: &[&str]| -> f64 {
        for (k, &e) in edges[..edges.len() - 1].iter().enumerate() {
            let ok = (k..edges.len() - 1).all(|j| {
                let n_tot = cum[0].1[j] * N_MW as f64;
                cum[0].1[j] < 3.0
                    || cum.iter().filter(|(n, _)| check.contains(&n.as_str())).all(|(_, c)| {
                        (c[j] / cum[0].1[j] - 1.0).abs() <= 0.10f64.max(2.0 / n_tot.sqrt())
                    })
            });
            if ok {
                return e;
            }
        }
        f64::NAN
    };
    println!(
        "\nM_complete from nmin40 + down2: {:.0}; including nmin80 (4x stricter cut): {:.0} (full-res particle masses)",
        m_complete(&["nmin40", "down2"]),
        m_complete(&["nmin40", "nmin80", "down2"])
    );
}
