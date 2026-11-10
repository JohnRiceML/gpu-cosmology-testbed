//! cosmo-analysis: power spectrum, FoF halos and halo mass function for the 2D
//! dark-matter simulation (GSNAP001 snapshots).

mod catalog;
mod convergence;
mod csvtab;
mod experiment;
mod expsynth;
mod fof;
mod galaxy;
mod hmf;
mod image;
mod meta;
mod pk;
mod snapshot;
mod synth;
mod wdm;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_exp;

use meta::{Meta, PkRow};
use serde_json::{Value, json};
use std::f64::consts::PI;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

const USAGE: &str = "\
usage:
  cosmo-analysis <run_dir | runs_root> [--snap A] [--out DIR] [--grid 2048] [--image-grid 2048]
                 [--b 0.2] [--min-members 20] [--fit-min 100] [--no-shot] [--no-png] [--top 2000]
      run_dir holds meta.json, pk.csv and snap_a*.bin. If a runs root (e.g. D:\\cosmo-runs) is
      given, the newest sub-directory with meta.json and snap_a1.000.bin is analysed.
  cosmo-analysis gen-test <out_dir> [--lattice 1024] [--mesh 512] [--seed 42] [--box 2000]
                 [--a-init 0.02] [--a 0.1,0.2,0.35,0.5,0.75,1.0] [--A 3.0] [--n 0.0] [--Rc 10.0]
      writes a synthetic 2D Zel'dovich run in the same file format.
  cosmo-analysis experiment <experiment_dir | experiments_root> [--out DIR] [--n-mw 30] [--target-sats 30]
                 [--host-min 1000] [--m-complete M] [--boot 200] [--mc 1000] [--seed N] [--no-png]
                 [--clean lovell|shape|none] [--count-sys 0.10]
      CDM vs WDM subhalo / satellite experiment (<dir>/<model>/{meta.json,subhalos/,snap_a1.000.bin}).
      Given a root holding several experiments, the newest complete one is used.
  cosmo-analysis experiment-synth <out_dir> [--seed N] [--hosts 400] [--lattice 8192]
      writes synthetic CDM + 3 WDM catalogs (injected suppression) in the subhalo-finder format.
  cosmo-analysis experiment-inject <cdm_model_dir> <out_dir> [--mhm 100,400,1600]
      pseudo-WDM models made by thinning a real CDM catalog with a known suppression (validation).
  cosmo-analysis convergence <base_exp_dir> <conv_dir> [--out DIR] [--n-mw 30] [--m-complete-particles 160]
                 [--boot 200] [--beta-gamma 6.66,-0.135]
      convergence suite: every <conv_dir>/<variant>/{cdm,wdm_*} vs base at a = 1, physical units.";

struct Opts {
    grid: usize,
    image_grid: usize,
    b_frac: f64,
    min_members: usize,
    fit_min: usize,
    shot: bool,
    png: bool,
    top: usize,
    snap: Option<f64>,
    out: Option<PathBuf>,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let r = match args.first().map(String::as_str) {
        None | Some("-h") | Some("--help") => {
            println!("{USAGE}");
            Ok(())
        }
        Some("gen-test") => cmd_gen(&args[1..]),
        Some("experiment") => cmd_experiment(&args[1..]),
        Some("experiment-synth") => cmd_experiment_synth(&args[1..]),
        Some("experiment-inject") => cmd_experiment_inject(&args[1..]),
        Some("convergence") => cmd_convergence(&args[1..]),
        _ => cmd_analyze(&args),
    };
    if let Err(e) = r {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn take_val<'a>(args: &'a [String], i: &mut usize, flag: &str) -> Result<&'a str, String> {
    *i += 1;
    args.get(*i).map(String::as_str).ok_or(format!("{flag} needs a value"))
}

fn parse<T: std::str::FromStr>(s: &str, flag: &str) -> Result<T, String> {
    s.parse().map_err(|_| format!("bad value for {flag}: {s}"))
}

fn cmd_gen(args: &[String]) -> Result<(), String> {
    let mut dir = None;
    let mut p = synth::GenParams {
        box_size: 2000.0,
        lattice: 1024,
        mesh: 512,
        seed: 42,
        a_init: 0.02,
        a_list: vec![0.1, 0.2, 0.35, 0.5, 0.75, 1.0],
        spec: meta::Spectrum { amp: 3.0, n: 0.0, r_c: 10.0 },
    };
    let mut i = 0;
    while i < args.len() {
        let f = args[i].as_str();
        match f {
            "--lattice" => p.lattice = parse(take_val(args, &mut i, f)?, f)?,
            "--mesh" => p.mesh = parse(take_val(args, &mut i, f)?, f)?,
            "--seed" => p.seed = parse(take_val(args, &mut i, f)?, f)?,
            "--box" => p.box_size = parse(take_val(args, &mut i, f)?, f)?,
            "--a-init" => p.a_init = parse(take_val(args, &mut i, f)?, f)?,
            "--A" => p.spec.amp = parse(take_val(args, &mut i, f)?, f)?,
            "--n" => p.spec.n = parse(take_val(args, &mut i, f)?, f)?,
            "--Rc" => p.spec.r_c = parse(take_val(args, &mut i, f)?, f)?,
            "--a" => {
                p.a_list = take_val(args, &mut i, f)?
                    .split(',')
                    .map(|x| parse(x.trim(), f))
                    .collect::<Result<_, _>>()?
            }
            _ if !f.starts_with("--") && dir.is_none() => dir = Some(PathBuf::from(f)),
            _ => return Err(format!("unknown argument {f}\n{USAGE}")),
        }
        i += 1;
    }
    let dir = dir.ok_or(format!("gen-test needs an output directory\n{USAGE}"))?;
    synth::gen_test(&dir, &p)
}

/// A run directory, or the newest complete run below a runs root.
fn resolve_run_dir(path: &Path) -> Result<PathBuf, String> {
    if path.join("meta.json").is_file() {
        return Ok(path.to_path_buf());
    }
    let mut cands: Vec<(bool, PathBuf)> = std::fs::read_dir(path)
        .map_err(|e| format!("{}: {e}", path.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir() && p.join("meta.json").is_file())
        .map(|p| (p.join("snap_a1.000.bin").is_file(), p))
        .collect();
    cands.sort();
    let pick = cands
        .iter()
        .rev()
        .find(|c| c.0)
        .or(cands.last())
        .ok_or(format!("{}: no run directory with meta.json", path.display()))?;
    eprintln!("using run directory {}", pick.1.display());
    Ok(pick.1.clone())
}

fn list_snapshots(dir: &Path) -> Result<Vec<(f64, PathBuf)>, String> {
    let mut v: Vec<(f64, PathBuf)> = std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            snapshot::a_from_name(&name).map(|a| (a, e.path()))
        })
        .collect();
    v.sort_by(|a, b| a.0.total_cmp(&b.0));
    Ok(v)
}

fn cmd_analyze(args: &[String]) -> Result<(), String> {
    let mut o = Opts {
        grid: 2048,
        image_grid: 2048,
        b_frac: 0.2,
        min_members: 20,
        fit_min: 100,
        shot: true,
        png: true,
        top: 2000,
        snap: None,
        out: None,
    };
    let mut run = None;
    let mut i = 0;
    while i < args.len() {
        let f = args[i].as_str();
        match f {
            "--snap" => o.snap = Some(parse(take_val(args, &mut i, f)?, f)?),
            "--out" => o.out = Some(PathBuf::from(take_val(args, &mut i, f)?)),
            "--grid" => o.grid = parse(take_val(args, &mut i, f)?, f)?,
            "--image-grid" => o.image_grid = parse(take_val(args, &mut i, f)?, f)?,
            "--b" => o.b_frac = parse(take_val(args, &mut i, f)?, f)?,
            "--min-members" => o.min_members = parse(take_val(args, &mut i, f)?, f)?,
            "--fit-min" => o.fit_min = parse(take_val(args, &mut i, f)?, f)?,
            "--top" => o.top = parse(take_val(args, &mut i, f)?, f)?,
            "--no-shot" => o.shot = false,
            "--no-png" => o.png = false,
            _ if !f.starts_with("--") && run.is_none() => run = Some(PathBuf::from(f)),
            _ => return Err(format!("unknown argument {f}\n{USAGE}")),
        }
        i += 1;
    }
    let run = resolve_run_dir(&run.ok_or(USAGE)?)?;
    let meta = Meta::load(&run.join("meta.json"))?;
    if !meta.growth.replace(' ', "").eq_ignore_ascii_case("D=a") {
        eprintln!("warning: meta growth = {:?}; this pipeline assumes D = a", meta.growth);
    }
    let gpu: Vec<PkRow> = match meta::read_pk_csv(&run.join("pk.csv")) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("warning: no GPU spectrum ({e})");
            vec![]
        }
    };
    let mut snaps = list_snapshots(&run)?;
    if snaps.is_empty() {
        return Err(format!("{}: no snap_a*.bin files", run.display()));
    }
    if let Some(a) = o.snap {
        let best = snaps
            .iter()
            .min_by(|x, y| (x.0 - a).abs().total_cmp(&(y.0 - a).abs()))
            .cloned()
            .unwrap();
        if (best.0 - a).abs() > 1e-3 {
            eprintln!("note: --snap {a} -> nearest snapshot a = {:.3}", best.0);
        }
        snaps = vec![best];
    }
    let out = o.out.clone().unwrap_or_else(|| run.join("analysis"));
    std::fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    let dc2 = hmf::tophat_delta_c(2);
    let dc3 = hmf::tophat_delta_c(3);
    eprintln!(
        "run {} | box {} | a_init {} | P0 = {}*k^{}*exp(-(k*{})^2) | delta_c(2D top-hat) = {:.4} | {} threads",
        run.display(),
        meta.box_size,
        meta.a_init,
        meta.spectrum.amp,
        meta.spectrum.n,
        meta.spectrum.r_c,
        dc2,
        rayon::current_num_threads()
    );
    let mut results = Vec::new();
    let mut rows = Vec::new();
    for (a, path) in &snaps {
        eprintln!("--- snapshot a = {a:.3}: {}", path.display());
        match analyze_snapshot(path, *a, &meta, &gpu, &o, &out, dc2) {
            Ok((v, row)) => {
                results.push(v);
                rows.push(row);
            }
            Err(e) => eprintln!("skipping {}: {e}", path.display()),
        }
        let summary = json!({
            "run_dir": run.display().to_string(),
            "meta": meta.raw,
            "meta_parsed": {
                "box": meta.box_size, "mesh": meta.mesh, "particles": meta.particles,
                "lattice": meta.lattice, "seed": meta.seed, "a_init": meta.a_init,
                "a_final": meta.a_final, "spectrum_shape": meta.spectrum_shape,
                "A": meta.spectrum.amp, "n": meta.spectrum.n, "R_c": meta.spectrum.r_c,
                "growth": meta.growth, "pk_convention": meta.pk_convention,
                "poisson_coefficient": meta.poisson_coefficient,
            },
            "settings": {
                "pk_grid": o.grid, "image_grid": o.image_grid, "fof_b": o.b_frac,
                "min_members": o.min_members, "hmf_fit_min_particles": o.fit_min, "shot_noise_subtracted": o.shot,
                "threads": rayon::current_num_threads(),
            },
            "conventions": {
                "delta_k": "(1/N^2) sum_x delta(x) exp(-ik.x)",
                "P": "L^2 <|delta_k|^2>; sigma^2 = int P k dk/(2pi)",
                "Delta2": "k^2 P(k) / (2 pi)  (2D dimensionless power per ln k)",
                "P_linear": "(a/a_init)^2 A k^n exp(-(k R_c)^2)",
                "pk_ratio": "P_particles / P_gpu (GPU log-log interpolated to the particle-bin k)",
                "shot_noise": "L^2/Np x prod_i[1-(2/3)sin^2(k_i dx/2)] / W_CIC^2(k) (-> L^2/Np at low k)",
                "sigma2": "D^2 int P0(k) [2 J1(kR)/(kR)]^2 k dk/(2pi), M = rho_bar pi R^2",
                "hmf": "dn/dlnM = (rho_bar/M) f(nu) |dln sigma/dln M| per unit area, nu = delta_c/sigma",
                "mass_units": "particle masses; mass_frac = M / Np",
            },
            "delta_c_2d_tophat": dc2,
            "delta_c_3d_tophat_check": dc3,
            "snapshots": results,
        });
        std::fs::write(out.join("summary.json"), serde_json::to_string_pretty(&summary).unwrap())
            .map_err(|e| e.to_string())?;
    }
    print_table(&rows, dc2);
    println!("outputs in {}", out.display());
    Ok(())
}

struct Row {
    a: f64,
    np: usize,
    k_nl: Option<f64>,
    k_nl_lin: Option<f64>,
    k_nl_gpu: Option<f64>,
    gpu_med: Option<f64>,
    lin_ratio: Option<f64>,
    n_halos: usize,
    m_max: usize,
    f_halo: f64,
    f_halo_ps: f64,
    dc_fit: Option<f64>,
    st: Option<(f64, f64)>,
    t_pk: f64,
    t_fof: f64,
}

fn fmt_opt(x: Option<f64>, prec: usize) -> String {
    x.map(|v| format!("{v:.prec$}")).unwrap_or_else(|| "-".into())
}

fn print_table(rows: &[Row], dc: f64) {
    println!();
    println!(
        "{:>6} {:>10} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>9} {:>7} {:>7} {:>7} {:>11} {:>6} {:>6}",
        "a", "Np", "k_nl", "k_nl_lin", "k_nl_gpu", "P/Pgpu", "P/Plin", "N_halo", "M_max", "f_halo",
        "f_PS", "dc_fit", "ST(q,p)", "t_pk", "t_fof"
    );
    for r in rows {
        println!(
            "{:>6.3} {:>10} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>9} {:>7.4} {:>7.4} {:>7} {:>11} {:>6.2} {:>6.2}",
            r.a,
            r.np,
            fmt_opt(r.k_nl, 4),
            fmt_opt(r.k_nl_lin, 4),
            fmt_opt(r.k_nl_gpu, 4),
            fmt_opt(r.gpu_med, 3),
            fmt_opt(r.lin_ratio, 3),
            r.n_halos,
            r.m_max,
            r.f_halo,
            r.f_halo_ps,
            fmt_opt(r.dc_fit, 3),
            r.st.map(|(q, p)| format!("{q:.2},{p:.2}")).unwrap_or_else(|| "-".into()),
            r.t_pk,
            r.t_fof
        );
    }
    println!(
        "k in rad per box length unit; Delta^2 = k^2 P/(2pi). P/Pgpu: median over k <= min(k_Nyq)/2 with P_gpu > 5 P_shot;\n\
         P/Plin: mode-weighted over bins with Delta^2_lin < 0.1 below the Delta^2_lin peak. delta_c(2D top-hat) = {dc:.4};\n\
         f_PS = PS mass fraction in halos >= min_members; fits use halos >= --fit-min particles; dc_fit \"-\" = no fit or PS shape\n         rejected (best fit on the [0.2,5] search bound); ST(q,p) fitted at delta_c(2D)."
    );
}

fn median(mut v: Vec<f64>) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    Some(if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) })
}

fn num(x: f64) -> Value {
    if x.is_finite() { json!(x) } else { Value::Null }
}

fn csvf(x: f64) -> String {
    if x.is_finite() { format!("{x:.6e}") } else { "nan".into() }
}

fn analyze_snapshot(
    path: &Path,
    a_name: f64,
    meta: &Meta,
    gpu: &[PkRow],
    o: &Opts,
    out: &Path,
    dc_theory: f64,
) -> Result<(Value, Row), String> {
    let t_total = Instant::now();
    let snap = snapshot::Snapshot::open(path)?;
    let a = if (snap.a as f64 - a_name).abs() < 1e-3 { a_name } else { snap.a as f64 };
    let l = meta.box_size;
    if (snap.box_size as f64 - l).abs() > 1e-4 * l {
        eprintln!("warning: snapshot box {} != meta box {}", snap.box_size, l);
    }
    let np = snap.count;
    if let Some(p) = meta.particles
        && p as usize != np
    {
        eprintln!("warning: meta particles {p} != snapshot count {np}");
    }
    let pos = snap.positions();
    let vel = snap.velocities();
    let tag = format!("{a:.3}");
    let kf = 2.0 * PI / l;
    let k_ny = PI * o.grid as f64 / l;
    let lattice = meta.lattice.unwrap_or_else(|| (np as f64).sqrt().round() as usize);
    let k_ny_ic = PI * lattice as f64 / l;

    // ---------------- P(k) ----------------
    let t = Instant::now();
    let counts = pk::cic_counts(pos, l, o.grid);
    let t_cic = t.elapsed().as_secs_f64();
    let gpu_sel = meta::rows_nearest_a(gpu, a);
    let mut edges = match &gpu_sel {
        Some((_, r)) if r.len() >= 3 => pk::edges_from_centers(&r.iter().map(|x| x.k).collect::<Vec<_>>()),
        _ => pk::log_edges(kf * 0.999, k_ny, 48),
    };
    // extend beyond the GPU range up to our Nyquist with the same log spacing
    if edges.len() >= 2 {
        let ratio = edges[edges.len() - 1] / edges[edges.len() - 2];
        while *edges.last().unwrap() < k_ny && ratio > 1.0 {
            let e = edges.last().unwrap() * ratio;
            edges.push(e);
        }
    }
    let g = meta.growth(a);
    let spec = meta.spectrum;
    let plin = |k: f64| meta.p_lin(k, a);
    let bins = pk::measure_pk(
        &counts,
        o.grid,
        l,
        np,
        &edges,
        &plin,
        pk::PkOptions { subtract_shot: o.shot, deconvolve: true },
    );
    let bins: Vec<pk::PkBin> = bins.into_iter().filter(|b| b.k <= k_ny).collect();
    let t_pk = t.elapsed().as_secs_f64();

    let ks: Vec<f64> = bins.iter().map(|b| b.k).collect();
    let d2: Vec<f64> = bins.iter().map(|b| b.delta2()).collect();
    // k_nl is searched only below the particle (lattice) Nyquist: above it the spectrum is
    // dominated by discreteness (lattice Bragg peaks at 2pi/spacing, shot noise)
    let k_search = k_ny.min(k_ny_ic);
    let n_search = ks.partition_point(|&k| k <= k_search);
    let knl = pk::k_nonlinear(&ks[..n_search], &d2[..n_search]);
    // linear-regime test range: rising side of Delta^2_lin, Delta^2_lin < 0.1
    let k_peak_lin = {
        let kk = pk::log_edges(kf, k_ny_ic.max(2.0 * kf), 2000);
        kk.iter().copied().max_by(|x, y| (x * x * plin(*x)).total_cmp(&(y * y * plin(*y)))).unwrap()
    };
    let k_ny_gpu = meta.mesh.map(|m| PI * m as f64 / l).unwrap_or(f64::INFINITY);
    let k_cmp = 0.5 * k_ny.min(k_ny_gpu);
    let knl_lin = pk::k_nonlinear_fn(plin, kf, k_ny.max(k_ny_ic));
    let (a_gpu, gpu_rows) = gpu_sel.clone().unwrap_or((f64::NAN, vec![]));
    let knl_gpu = if gpu_rows.is_empty() {
        Err("no GPU spectrum")
    } else {
        let gk: Vec<f64> = gpu_rows.iter().map(|r| r.k).collect();
        let gd: Vec<f64> = gpu_rows.iter().map(|r| r.k * r.k * r.p_meas / (2.0 * PI)).collect();
        pk::k_nonlinear(&gk, &gd)
    };

    // GPU self-consistency: its P_linear column vs the meta.json formula, and its mode counting
    let gpu_lin_vs_meta = median(
        gpu_rows
            .iter()
            .filter(|r| r.p_lin.is_finite() && r.p_lin > 0.0)
            .map(|r| r.p_lin / meta.p_lin(r.k, a_gpu))
            .collect(),
    );
    let gpu_modes_ratio = median(
        bins.iter()
            .filter_map(|b| {
                let r = gpu_rows.iter().find(|r| (r.k / b.k - 1.0).abs() < 0.03)?;
                (r.n_modes > 0.0).then(|| b.n_modes as f64 / r.n_modes)
            })
            .collect(),
    );
    let mut csv =
        String::from("k,P_particles,P_gpu,P_linear,ratio,ratio_linear,n_modes,P_shot,Delta2,P_before_shot\n");
    let mut gpu_ratios = Vec::new();
    let mut gpu_ratios_noshot = Vec::new();
    let mut gpu_ratios_low = Vec::new();
    let (mut sw_p, mut sw_l, mut chi2, mut nlin) = (0.0, 0.0, 0.0, 0usize);
    for b in &bins {
        let pg = meta::interp_pk(&gpu_rows, b.k);
        let ratio = b.p / pg;
        // compare where both grids resolve the mode well and the signal beats shot noise
        if ratio.is_finite() && pg > 5.0 * b.p_shot && b.k <= k_cmp {
            gpu_ratios.push(ratio);
            gpu_ratios_noshot.push(b.p_raw / pg);
            if b.k <= 0.2 * k_cmp {
                gpu_ratios_low.push(ratio);
            }
        }
        let d2l = b.k * b.k * b.p_lin / (2.0 * PI);
        if d2l < 0.1 && b.k <= k_peak_lin && b.p_lin > 0.0 {
            sw_p += b.n_modes as f64 * b.p;
            sw_l += b.n_modes as f64 * b.p_lin;
            chi2 += ((b.p - b.p_lin) / (b.p_lin * b.rel_err())).powi(2);
            nlin += 1;
        }
        let _ = writeln!(
            csv,
            "{},{},{},{},{},{},{},{},{},{}",
            csvf(b.k),
            csvf(b.p),
            csvf(pg),
            csvf(b.p_lin),
            csvf(ratio),
            csvf(b.p / b.p_lin),
            b.n_modes,
            csvf(b.p_shot),
            csvf(b.delta2()),
            csvf(b.p_raw)
        );
    }
    std::fs::write(out.join(format!("pk_{tag}.csv")), csv).map_err(|e| e.to_string())?;
    let gpu_med = median(gpu_ratios.clone());
    let gpu_mad = if gpu_ratios.is_empty() {
        None
    } else {
        Some(gpu_ratios.iter().map(|r| r.ln().abs()).sum::<f64>() / gpu_ratios.len() as f64)
    };
    let lin_ratio = (nlin > 0).then(|| sw_p / sw_l);

    // ---------------- image ----------------
    let t = Instant::now();
    if o.png {
        let png_path = out.join(format!("density_{tag}.png"));
        if o.image_grid == o.grid {
            image::write_density_png(&png_path, &counts, o.grid, np)?;
        } else {
            let c2 = pk::cic_counts(pos, l, o.image_grid);
            image::write_density_png(&png_path, &c2, o.image_grid, np)?;
        }
    }
    drop(counts);
    let t_png = t.elapsed().as_secs_f64();

    // ---------------- FoF ----------------
    let link = o.b_frac * l / (np as f64).sqrt();
    let t = Instant::now();
    let groups = fof::fof(pos, l, link, o.min_members)?;
    let t_fof = t.elapsed().as_secs_f64();
    if groups.is_empty() {
        eprintln!("FoF: no groups with >= {} members", o.min_members);
    } else {
        eprintln!("FoF: {} groups, largest {} particles", groups.len(), groups.size(0));
    }
    let t = Instant::now();
    let halos = fof::halo_properties(&groups, pos, vel, l);
    drop(groups);
    let t_props = t.elapsed().as_secs_f64();
    let n_in_halos: usize = halos.iter().map(|h| h.n).sum();
    let f_halo = n_in_halos as f64 / np as f64;

    let mut hcsv = String::from("rank,n_particles,mass_mp,mass_frac,x,y,vx,vy,sigma_v\n");
    for (r, h) in halos.iter().take(o.top).enumerate() {
        let _ = writeln!(
            hcsv,
            "{},{},{},{:.6e},{:.4},{:.4},{:.5e},{:.5e},{:.5e}",
            r + 1,
            h.n,
            h.n,
            h.n as f64 / np as f64,
            h.com[0],
            h.com[1],
            h.vmean[0],
            h.vmean[1],
            h.sigma_v
        );
    }
    std::fs::write(out.join(format!("halos_{tag}.csv")), hcsv).map_err(|e| e.to_string())?;

    // ---------------- HMF ----------------
    let t = Instant::now();
    let area = l * l;
    let sizes: Vec<usize> = halos.iter().map(|h| h.n).collect();
    let hb = hmf::hmf_bins(&sizes, o.min_members, area, 0.1);
    let table = hmf::SigmaTable::new(&spec, l, np as f64, k_ny_ic, np as f64);
    let ps_th = hmf::Model::Ps { delta_c: dc_theory };
    // fit only well-resolved halos (default >= 100 particles): the smallest FoF groups are
    // dominated by force softening and by spurious linkage along filaments
    let fit_bins: Vec<hmf::HmfBin> = hb.iter().copied().filter(|b| b.m_lo >= o.fit_min as f64 - 0.5).collect();
    let nonzero = fit_bins.iter().filter(|b| b.count > 0).count();
    let n_fit_halos: usize = fit_bins.iter().map(|b| b.count).sum();
    let fit = (n_fit_halos >= 10 && nonzero >= 3).then(|| hmf::fit_hmf(&table, &fit_bins, g, area, dc_theory));
    let mut mcsv = String::from("M_center,dn_dlnM,dn_dlnM_err,n_gt_M,PS_prediction,fitted_prediction,PS_bestfit_dc,N_halos\n");
    for b in &hb {
        let ps = hmf::dn_dlnm(&table, b.m_center, g, ps_th);
        let (st, psf) = match &fit {
            Some(f) => (
                hmf::dn_dlnm(&table, b.m_center, g, hmf::Model::St { delta_c: dc_theory, q: f.st_q, p: f.st_p }),
                hmf::dn_dlnm(&table, b.m_center, g, hmf::Model::Ps { delta_c: f.delta_c_bestfit }),
            ),
            None => (f64::NAN, f64::NAN),
        };
        let _ = writeln!(
            mcsv,
            "{:.4},{},{},{},{},{},{},{}",
            b.m_center,
            csvf(b.dn_dlnm),
            csvf(b.err),
            csvf(b.n_gt),
            csvf(ps),
            csvf(st),
            csvf(psf),
            b.count
        );
    }
    std::fs::write(out.join(format!("hmf_{tag}.csv")), mcsv).map_err(|e| e.to_string())?;
    let m_min = o.min_members as f64 - 0.5;
    let f_halo_ps = hmf::mass_fraction_above(&table, m_min, g, ps_th);
    let sigma_min = table.sigma0(m_min) * g;
    let m_star = {
        // sigma(M*) = delta_c
        let mut lo = 0.0f64;
        let mut hi = (np as f64).ln();
        if table.sigma0(hi.exp()) * g > dc_theory {
            f64::INFINITY
        } else if table.sigma0(1.0) * g < dc_theory {
            0.0
        } else {
            for _ in 0..60 {
                let mid = 0.5 * (lo + hi);
                if table.sigma0(mid.exp()) * g > dc_theory { lo = mid } else { hi = mid }
            }
            (0.5 * (lo + hi)).exp()
        }
    };
    let t_hmf = t.elapsed().as_secs_f64();
    let t_tot = t_total.elapsed().as_secs_f64();

    let largest = halos.first().map(|h| {
        json!({"n_particles": h.n, "mass_frac": h.n as f64 / np as f64, "com": h.com, "sigma_v": h.sigma_v})
    });
    let v = json!({
        "a": a,
        "file": path.display().to_string(),
        "n_particles": np,
        "growth_D_over_Dinit": g,
        "k_nl": knl.ok(),
        "k_nl_note": knl.err(),
        "k_nl_linear": knl_lin.ok(),
        "k_nl_gpu": knl_gpu.ok(),
        "pk": {
            "grid": o.grid, "k_nyquist": k_ny, "n_bins": bins.len(),
            "gpu_a": num(a_gpu),
            "gpu_comparison_k_max": k_cmp,
            "gpu_comparison_bins": gpu_ratios.len(),
            "median_ratio_to_gpu": gpu_med,
            "median_ratio_to_gpu_without_shot_subtraction": median(gpu_ratios_noshot),
            "median_ratio_to_gpu_low_k": median(gpu_ratios_low),
            "linear_test_k_max": k_peak_lin,
            "mean_abs_ln_ratio_to_gpu": gpu_mad,
            "linear_regime_bins": nlin,
            "ratio_to_linear_mode_weighted": lin_ratio,
            "chi2_per_bin_vs_linear": (nlin > 0).then(|| chi2 / nlin as f64),
            "gpu_P_linear_over_meta_formula": gpu_lin_vs_meta,
            "n_modes_ours_over_gpu": gpu_modes_ratio,
        },
        "fof": {
            "linking_length": link, "b": o.b_frac, "min_members": o.min_members,
            "n_halos": halos.len(),
            "largest_halo": largest,
            "halo_mass_fraction": f_halo,
            "n_particles_in_halos": n_in_halos,
        },
        "ps": {
            "delta_c_theory_2d_tophat": dc_theory,
            "sigma_at_min_mass": sigma_min,
            "M_star_particles": if m_star >= 1.0 { num(m_star) } else { Value::Null },
            "fit_min_particles": o.fit_min,
            "fit_halos": n_fit_halos,
            "ps_mass_fraction_above_min": f_halo_ps,
            "cash_ps_theory": fit.map(|f| f.cash_ps_theory),
            "delta_c_bestfit": fit.map(|f| f.delta_c_bestfit),
            "delta_c_bestfit_at_search_bound": fit.map(|f| f.delta_c_at_bound),
            "cash_ps_bestfit": fit.map(|f| f.cash_ps_fit),
            "st_q": fit.map(|f| f.st_q),
            "st_p": fit.map(|f| f.st_p),
            "st_delta_c_eff_sqrtq": fit.map(|f| dc_theory * f.st_q.sqrt()),
            "cash_st": fit.map(|f| f.cash_st),
            "n_mass_bins": hb.len(),
            "n_mass_bins_fitted": fit.map(|f| f.n_bins),
            "sigma_integral_k_range": [table.k_lo, table.k_hi],
        },
        "timings_s": {
            "cic_incl_first_read": t_cic, "pk_total": t_pk, "png": t_png, "fof": t_fof,
            "halo_props": t_props, "hmf_fit": t_hmf, "total": t_tot,
        },
    });
    eprintln!(
        "a={a:.3}: k_nl={} (lin {}), {} halos, largest {} particles, f_halo={:.4}, dc_fit={}, P(k) {:.2}s, FoF {:.2}s, total {:.2}s",
        fmt_opt(knl.ok(), 4),
        fmt_opt(knl_lin.ok(), 4),
        halos.len(),
        halos.first().map(|h| h.n).unwrap_or(0),
        f_halo,
        fmt_opt(fit.map(|f| f.delta_c_bestfit), 3),
        t_pk,
        t_fof,
        t_tot
    );
    let row = Row {
        a,
        np,
        k_nl: knl.ok(),
        k_nl_lin: knl_lin.ok(),
        k_nl_gpu: knl_gpu.ok(),
        gpu_med,
        lin_ratio,
        n_halos: halos.len(),
        m_max: halos.first().map(|h| h.n).unwrap_or(0),
        f_halo,
        f_halo_ps,
        dc_fit: fit.filter(|f| !f.delta_c_at_bound).map(|f| f.delta_c_bestfit),
        st: fit.map(|f| (f.st_q, f.st_p)),
        t_pk,
        t_fof,
    };
    Ok((v, row))
}

fn cmd_experiment(args: &[String]) -> Result<(), String> {
    let mut o = experiment::ExpOpts::default();
    let mut dir = None;
    let mut i = 0;
    while i < args.len() {
        let f = args[i].as_str();
        match f {
            "--out" => o.out = Some(PathBuf::from(take_val(args, &mut i, f)?)),
            "--n-mw" => o.n_mw = parse(take_val(args, &mut i, f)?, f)?,
            "--target-sats" => o.target_sats = parse(take_val(args, &mut i, f)?, f)?,
            "--host-min" => o.host_min = parse(take_val(args, &mut i, f)?, f)?,
            "--fit-min" | "--m-complete" => o.m_complete = Some(parse(take_val(args, &mut i, f)?, f)?),
            "--boot" => o.n_boot = parse(take_val(args, &mut i, f)?, f)?,
            "--mc" => o.n_mc = parse(take_val(args, &mut i, f)?, f)?,
            "--seed" => o.seed = parse(take_val(args, &mut i, f)?, f)?,
            "--no-png" => o.png = false,
            "--clean" => o.clean = take_val(args, &mut i, f)?.to_string(),
            "--count-sys" => o.count_sys = parse(take_val(args, &mut i, f)?, f)?,
            _ if !f.starts_with("--") && dir.is_none() => dir = Some(PathBuf::from(f)),
            _ => return Err(format!("unknown argument {f}\n{USAGE}")),
        }
        i += 1;
    }
    let dir = dir.ok_or(USAGE)?;
    let dir = if experiment::discover(&dir)?.is_empty() {
        // a root of experiments: newest sub-directory that has models
        let mut c: Vec<PathBuf> = std::fs::read_dir(&dir)
            .map_err(|e| e.to_string())?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.is_dir() && experiment::discover(p).map(|v| v.len() >= 2).unwrap_or(false))
            .collect();
        c.sort();
        let p = c.pop().ok_or(format!("{}: no experiment with model sub-directories", dir.display()))?;
        eprintln!("using experiment directory {}", p.display());
        p
    } else {
        dir
    };
    experiment::run(&dir, &o)
}

fn cmd_experiment_synth(args: &[String]) -> Result<(), String> {
    let mut s = expsynth::SynthSpec::default();
    let mut dir = None;
    let mut i = 0;
    while i < args.len() {
        let f = args[i].as_str();
        match f {
            "--seed" => s.seed = parse(take_val(args, &mut i, f)?, f)?,
            "--hosts" => s.n_hosts = parse(take_val(args, &mut i, f)?, f)?,
            "--lattice" => s.lattice = parse(take_val(args, &mut i, f)?, f)?,
            _ if !f.starts_with("--") && dir.is_none() => dir = Some(PathBuf::from(f)),
            _ => return Err(format!("unknown argument {f}\n{USAGE}")),
        }
        i += 1;
    }
    expsynth::write_experiment(&dir.ok_or(USAGE)?, &s)
}
fn cmd_experiment_inject(args: &[String]) -> Result<(), String> {
    let mut mhm = vec![100.0, 400.0, 1600.0];
    let mut pos = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let f = args[i].as_str();
        match f {
            "--mhm" => {
                mhm = take_val(args, &mut i, f)?.split(',').map(|x| parse(x.trim(), f)).collect::<Result<_, _>>()?
            }
            _ if !f.starts_with("--") => pos.push(PathBuf::from(f)),
            _ => return Err(format!("unknown argument {f}\n{USAGE}")),
        }
        i += 1;
    }
    if pos.len() != 2 {
        return Err(USAGE.into());
    }
    let s = expsynth::SynthSpec::default();
    expsynth::inject(&pos[0], &pos[1], &mhm, s.supp_sub, s.supp_cen)
}
fn cmd_convergence(args: &[String]) -> Result<(), String> {
    let mut o = convergence::ConvOpts::default();
    let mut pos = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let f = args[i].as_str();
        match f {
            "--out" => o.out = Some(PathBuf::from(take_val(args, &mut i, f)?)),
            "--n-mw" => o.n_mw = parse(take_val(args, &mut i, f)?, f)?,
            "--m-complete-particles" => o.m_complete_base_particles = parse(take_val(args, &mut i, f)?, f)?,
            "--boot" => o.n_boot = parse(take_val(args, &mut i, f)?, f)?,
            "--beta-gamma" => {
                let v: Vec<f64> = take_val(args, &mut i, f)?.split(',').map(|x| parse(x.trim(), f)).collect::<Result<_, _>>()?;
                if v.len() != 2 {
                    return Err("--beta-gamma needs two values".into());
                }
                o.beta_gamma = Some((v[0], v[1]));
            }
            _ if !f.starts_with("--") => pos.push(PathBuf::from(f)),
            _ => return Err(format!("unknown argument {f}\n{USAGE}")),
        }
        i += 1;
    }
    if pos.len() != 2 {
        return Err(USAGE.into());
    }
    convergence::run(&pos[0], &pos[1], &o)
}