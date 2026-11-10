//! `subfind`: subhalo catalogs and merger trees for 2D cosmological snapshots.
//!
//! ```text
//! subfind <model_dir>...                     run each model directory (needs meta.json)
//! subfind --scan D:\cosmo-runs\experiments    every complete <pair>/<model>/ under the root
//! options: --vel peculiar|canonical|comoving  override the velocity convention
//!          --nmin N  --fesc F (default e)  --host-delta D  --threads N`r`n//!          --out DIR (one model)  --downsample K (keep IDs with hash % K == 0, masses x N/N_kept)  --force (re-run models that already have output)
//! ```
//! Outputs go to `<model_dir>/subhalos/` (see `output` for the column contract).

use std::path::PathBuf;
use subhalo_finder::run::{RunOptions, run_complete, run_model_with};
use subhalo_finder::units::{RunMeta, VelConvention};
use subhalo_finder::Params;

fn main() {
    if let Err(e) = real_main() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn real_main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut dirs: Vec<PathBuf> = Vec::new();
    let mut params = Params::default();
    let mut vel = None;
    let mut threads = None;
    let mut force = false;
    let mut opts = RunOptions::default();
    let mut scan: Option<PathBuf> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = || it.next().cloned().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--vel" => vel = Some(VelConvention::parse(&val()?).ok_or("--vel: peculiar|canonical|comoving")?),
            "--nmin" => params.n_min = val()?.parse().map_err(|e| format!("--nmin: {e}"))?,
            "--escape" | "--fesc" => params.escape_factor = val()?.parse().map_err(|e| format!("{a}: {e}"))?,
            "--out" => opts.out_dir = Some(PathBuf::from(val()?)),
            "--downsample" => opts.downsample = val()?.parse().map_err(|e| format!("--downsample: {e}"))?,
            "--host-delta" => params.host_overdensity = val()?.parse().map_err(|e| format!("--host-delta: {e}"))?,
            "--threads" => threads = Some(val()?.parse::<usize>().map_err(|e| format!("--threads: {e}"))?),
            "--scan" => scan = Some(PathBuf::from(val()?)),
            "--force" => force = true,
            "-h" | "--help" => {
                println!("usage: subfind <model_dir>... | --scan <experiments_root> [--vel X] [--nmin N] [--fesc F] [--host-delta D] [--out DIR] [--downsample K] [--threads N] [--force]");
                return Ok(());
            }
            s => dirs.push(PathBuf::from(s)),
        }
    }
    if let Some(t) = threads {
        rayon::ThreadPoolBuilder::new().num_threads(t).build_global().map_err(|e| e.to_string())?;
    }
    if let Some(root) = scan {
        let mut found = Vec::new();
        for pair in std::fs::read_dir(&root).map_err(|e| format!("{}: {e}", root.display()))?.flatten() {
            if let Ok(models) = std::fs::read_dir(pair.path()) {
                for m in models.flatten() {
                    let p = m.path();
                    if p.join("meta.json").exists() {
                        if !run_complete(&p) {
                            eprintln!("skip (incomplete): {}", p.display());
                        } else if !force && p.join("subhalos").join("subfind_meta.json").exists() {
                            eprintln!("skip (done; --force to redo): {}", p.display());
                        } else {
                            found.push(p);
                        }
                    }
                }
            }
        }
        found.sort();
        dirs.extend(found);
    }
    if dirs.is_empty() {
        return Err("no model directories given (see --help)".into());
    }
    let n_dirs = dirs.len();
    for d in dirs {
        let meta = RunMeta::load(&d, vel)?;
        eprintln!(
            "== {} ({}; {} particles, {} snapshots, velocities: {} [{}], softening {} [{}], particle mass {} [{}])",
            d.display(),
            meta.model(),
            meta.particles,
            meta.snapshots.len(),
            meta.vel.name(),
            meta.vel_source,
            meta.softening,
            meta.softening_source,
            meta.particle_mass,
            meta.particle_mass_source
        );
        if opts.out_dir.is_some() && n_dirs > 1 {
            return Err("--out needs exactly one model directory".into());
        }
        let summary = run_model_with(&meta, &params, &opts, &mut |s| eprintln!("{s}"))?;
        println!("{}", serde_json::to_string(&summary).unwrap_or_default());
    }
    Ok(())
}
