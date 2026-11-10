//! Cosmology run setup (CDM/WDM models, science-mode normalisation, meta.json) and the
//! CDM-vs-WDM experiment batch.

use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::cosmo::{
    self, BOX, Cfl, CosmoConfig, ModeTable, RunSchedule, SNAPSHOT_AS, WDM_NU, science_normalization, science_snapshots,
    wdm_alpha_for_khm,
};
use crate::cosmo_io::{self, RunInfo, RunWriter};
use crate::gpu::Renderer;
use crate::settings::CosmoSettings;

/// Science-run defaults (see meta.json "normalization").
pub const SCI_SPECTRAL_INDEX: f64 = -1.5;
pub const SCI_HOST_PARTICLES: f64 = 1.0e6;
pub const SCI_RC_CELLS: f64 = 0.35;
pub const SCI_SIGMA_MESH_INIT: f64 = 0.1;
/// Half-mode masses of the experiment's WDM models, in particle masses.
pub const EXPERIMENT_MHM_PARTICLES: [f64; 3] = [3000.0, 300.0, 30.0];

/// One dark-matter model to run.
#[derive(Clone, Debug)]
pub struct ModelSpec {
    pub name: String,
    pub k_hm: Option<f64>,
    pub out_dir: Option<PathBuf>,
    pub pair_id: Option<String>,
    /// restart from this snapshot (a, path)
    pub resume: Option<(f64, PathBuf)>,
}

impl ModelSpec {
    pub fn cdm() -> Self {
        Self { name: "cdm".into(), k_hm: None, out_dir: None, pair_id: None, resume: None }
    }
    pub fn wdm(k_hm: f64) -> Self {
        Self { name: format!("wdm_khm{k_hm:.2}"), k_hm: Some(k_hm), out_dir: None, pair_id: None, resume: None }
    }
}

/// Half-mode quantities (2D): λ_hm = 2π/k_hm, M_hm = ρ̄ π (λ_hm/2)² with ρ̄ = 1 (code mass per
/// unit area, box mass = L²).
pub fn half_mode_mass(k_hm: f64) -> f64 {
    let lam = 2.0 * std::f64::consts::PI / k_hm;
    std::f64::consts::PI * (lam / 2.0) * (lam / 2.0)
}

/// k_hm whose half-mode mass is `m_particles` particle masses for `np` particles.
pub fn khm_for_mass(m_particles: f64, np: f64) -> f64 {
    let m = m_particles * BOX * BOX / np; // code mass
    let lam_half = (m / std::f64::consts::PI).sqrt();
    2.0 * std::f64::consts::PI / (2.0 * lam_half)
}

pub struct CosmoRun {
    pub schedule: RunSchedule,
    pub writer: RunWriter,
    pub psi_rms: f64,
    /// false once the mouse well has been used
    pub science: bool,
    /// uncapped adaptive-step run
    pub science_mode: bool,
    pub np: u32,
    pub model: String,
    pub started: Instant,
    pub step_ms_first: Option<(f64, f64)>,
    pub step_ms_last: Option<(f64, f64)>,
    pub last_report_step: u64,
    /// GPU error count when the run started (any new error fails the run)
    pub gpu_errors_at_start: u64,
}

/// Base experiment geometry (the convergence suite's reference).
pub const BASE_MESH: u32 = 4096;
pub const BASE_LATTICE: u32 = 6144;

/// Lattice side of a run: the explicit `--lattice` if given, else `lattice_side`.
pub fn run_lattice_side(cs: &CosmoSettings, mesh: u32, particles: u32, max_dust: u32) -> u32 {
    match cs.lattice {
        Some(n) => n.min((max_dust as f64).sqrt().floor() as u32),
        None => lattice_side(mesh, particles, max_dust),
    }
}

/// Lattice side used for `particles` on a `mesh` (multiple of mesh/2, within the adapter cap).
pub fn lattice_side(mesh: u32, particles: u32, max_dust: u32) -> u32 {
    let unit = (mesh / 2).max(1);
    let max_side = ((max_dust as f64).sqrt().floor() as u32 / unit).max(1) * unit;
    ((((particles as f64).sqrt() / unit as f64).round() as u32).max(1) * unit).min(max_side)
}

pub fn start_run(
    renderer: &mut Renderer,
    cs: &CosmoSettings,
    mesh: u32,
    particles: u32,
    seed: u32,
    model: &ModelSpec,
) -> Result<CosmoRun, String> {
    let gpu_errors_at_start = renderer.gpu_health().0;
    let science = cs.science || cs.experiment;
    let cell = BOX / mesh as f64;
    let n_index = cs.spectral_index.unwrap_or(if science { SCI_SPECTRAL_INDEX } else { -1.0 });
    let rc_cells = if science { cs.rc_cells.unwrap_or(SCI_RC_CELLS) } else { cs.rc_cells.unwrap_or(1.5) };
    // --base-norm: A, a_init and R_c (absolute) exactly as the base experiment, whatever this
    // run's mesh or lattice; R_c in this run's cells changes accordingly.
    let (norm_mesh, norm_side) = if cs.base_norm { (BASE_MESH, BASE_LATTICE) } else { (mesh, 0) };
    let rc = rc_cells * BOX / norm_mesh as f64;
    let rc_cells = rc / cell;
    let side = run_lattice_side(cs, mesh, particles, renderer.max_dust());
    let np_f = (side as f64) * (side as f64);
    let norm_np = if cs.base_norm { (norm_side as f64) * (norm_side as f64) } else { np_f };
    let (amp, a_init, norm) = if science {
        let nrm = science_normalization(norm_mesh, norm_np, n_index, rc, SCI_HOST_PARTICLES, cs.host_sigma, cs.sigma.unwrap_or(SCI_SIGMA_MESH_INIT));
        let a_init = cs.a_init.unwrap_or(nrm.a_init);
        // an explicit a_init keeps the host criterion: rescale A so that σ_host(a=1) is unchanged
        let amp = nrm.amp * (a_init / nrm.a_init).powi(2);
        (Some(amp), a_init, Some(nrm))
    } else {
        (None, cs.a_init.unwrap_or(0.02), None)
    };
    if cs.a_final <= a_init {
        return Err(format!("a_final {} must exceed a_init {a_init}", cs.a_final));
    }
    let alpha = model.k_hm.map(wdm_alpha_for_khm).unwrap_or(0.0);
    let table = ModeTable::build(mesh as usize, n_index, rc, cs.sigma.unwrap_or(0.05), amp, alpha);
    let spectrum = table.spectrum;
    let psi_rms = table.psi_rms;
    let eps = cs.eps_abs.unwrap_or(cs.eps_frac * BOX / side as f64);
    let snapshot_as: Vec<f64> = if !cs.snapshots {
        Vec::new()
    } else if science {
        science_snapshots(cs.a_final)
    } else {
        let mut v: Vec<f64> = SNAPSHOT_AS.iter().copied().filter(|&a| a <= cs.a_final + 1e-12).collect();
        if v.last().is_none_or(|&a| (a - cs.a_final).abs() > 1e-9) {
            v.push(cs.a_final);
        }
        v
    };
    let cfg = CosmoConfig {
        mesh,
        particles: side * side,
        seed,
        a_init,
        a_final: cs.a_final,
        run_seconds: cs.run_seconds,
        spectral_index: n_index,
        sigma_init: cs.sigma.unwrap_or(0.05),
        rc_cells,
        smooth_cells: 0.75,
        snapshots: cs.snapshots,
        out_root: PathBuf::from(&cs.runs_dir),
        flip_phases: cs.flip_phases,
        k_hm: model.k_hm,
        amp,
        pp: cs.pp.unwrap_or(science),
        eps,
        science,
        eta: cs.eta,
        min_dlna: cs.min_dlna,
        snapshot_as: snapshot_as.clone(),
        out_dir: model.out_dir.clone(),
        exact_lattice: cs.lattice.is_some(),
        ic_mesh: cs.ic_mesh,
    };
    let (side, np) = renderer.start_cosmo(&cfg, spectrum);
    let cfl = science.then_some(Cfl { eps, eta: cs.eta, min_dlna: cs.min_dlna });
    let mut schedule = RunSchedule::new(a_init, cs.a_final, cs.run_seconds, &snapshot_as, cfl);
    let all_names: Vec<(f64, String)> = schedule.snapshot_epochs().iter().map(|&a| (a, cosmo_io::snapshot_name(a))).collect();
    if let Some((a_r, path)) = &model.resume {
        let (a_file, state) = cosmo_io::read_snapshot_state(path, side).map_err(|e| format!("cannot read restart snapshot {}: {e}", path.display()))?;
        if state.len() != np as usize {
            return Err(format!("restart snapshot {} has {} particles, expected {np}", path.display(), state.len()));
        }
        renderer.cosmo_load_state(&state);
        schedule.resume_at(a_file.max(*a_r));
        println!("cosmic web [{}]: restarting from {} (a = {a_file:.6})", model.name, path.display());
    }
    // all epochs of the run (the restart keeps earlier files; COMPLETE checks every one)
    let snapshot_names = all_names;
    let sigma_in_box = norm.map(|n| {
        let sp = cosmo::Spectrum { amp: spectrum.amp, n: spectrum.n, rc: spectrum.rc, wdm_alpha: 0.0 };
        cosmo::sigma_tophat_in_box(|k| sp.p0(k), n.r_host, norm_mesh as usize, 1.0 / (a_init * a_init))
    });
    let meta = meta_json(&cfg, side, np, &table, &snapshot_names, norm, model, sigma_in_box);
    let meta = add_variant(meta, cs, &cfg);
    let info = RunInfo { a_init, particles: np as u64, side, snapshot_names };
    let writer = match &cfg.out_dir {
        Some(dir) => RunWriter::start_at(dir, &meta, table, info, model.resume.as_ref().map(|r| r.0)),
        None => RunWriter::start(&cfg.out_root, &meta, table, info),
    }
    .map_err(|e| format!("cannot create the run directory: {e}"))?;
    println!(
        "cosmic web [{}]: {np} particles ({side}^2 lattice), mesh {mesh}^2 periodic, seed {seed}, a {a_init:.5} -> {}, n = {n_index}, A = {:.4e}{}{}",
        model.name,
        cfg.a_final,
        spectrum.amp,
        if cfg.pp { format!(", P3M eps = {eps:.4}") } else { String::new() },
        if science { ", uncapped science run".to_string() } else { format!(" over {} s", cfg.run_seconds) },
    );
    Ok(CosmoRun {
        schedule,
        writer,
        psi_rms,
        science: true,
        science_mode: science,
        np,
        model: model.name.clone(),
        started: Instant::now(),
        step_ms_first: None,
        step_ms_last: None,
        last_report_step: 0,
        gpu_errors_at_start,
    })
}

/// GPU run-to-run determinism test of the P3M integration (`--determinism-test`): the science
/// set-up (n = -1.5, base-style normalisation on this mesh/lattice, eps = 0.08 spacing),
/// `steps` steps of dln a = 0.01 from a_init, run twice. Returns (differing particles, N).
pub fn determinism_test(renderer: &mut Renderer, cs: &CosmoSettings, mesh: u32, particles: u32, steps: u32) -> (usize, usize) {
    let side = run_lattice_side(cs, mesh, particles, renderer.max_dust());
    let np_f = (side as f64) * (side as f64);
    let rc = SCI_RC_CELLS * BOX / mesh as f64;
    let nrm = science_normalization(mesh, np_f, SCI_SPECTRAL_INDEX, rc, SCI_HOST_PARTICLES, cs.host_sigma, SCI_SIGMA_MESH_INIT);
    let table = ModeTable::build(mesh as usize, SCI_SPECTRAL_INDEX, rc, 0.05, Some(nrm.amp), 0.0);
    let eps = cs.eps_abs.unwrap_or(cs.eps_frac * BOX / side as f64);
    let cfg = CosmoConfig {
        mesh,
        particles: side * side,
        seed: cs.seed as u32,
        a_init: nrm.a_init,
        a_final: 1.0,
        run_seconds: 1.0,
        spectral_index: SCI_SPECTRAL_INDEX,
        sigma_init: 0.05,
        rc_cells: SCI_RC_CELLS,
        smooth_cells: 0.75,
        snapshots: false,
        out_root: PathBuf::from(&cs.runs_dir),
        flip_phases: false,
        k_hm: None,
        amp: Some(nrm.amp),
        pp: true,
        eps,
        science: true,
        eta: cs.eta,
        min_dlna: cs.min_dlna,
        snapshot_as: Vec::new(),
        out_dir: None,
        exact_lattice: cs.lattice.is_some(),
        ic_mesh: None,
    };
    renderer.cosmo_determinism_test(&cfg, table.spectrum, steps)
}

fn json_str(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

/// Convergence-suite fields at the top of meta.json ("variant", "ic_mesh", "normalization_reference").
fn add_variant(meta: String, cs: &CosmoSettings, cfg: &CosmoConfig) -> String {
    if cs.variant.is_none() && cfg.ic_mesh.is_none() && !cs.base_norm && !cfg.exact_lattice {
        return meta;
    }
    let mut extra = String::new();
    if let Some(v) = &cs.variant {
        extra += &format!(
            "  \"variant\": {{\"name\": {}, \"changed\": {}, \"reference\": \"exp-20261021-0022 (6144^2, mesh 4096, eps 0.026042, eta 0.025, floor dln a 0.001, seed 1)\"}},\n",
            json_str(v),
            json_str(cs.variant_note.as_deref().unwrap_or(""))
        );
    }
    extra += &format!(
        "  \"ic_mesh\": {},\n  \"exact_lattice\": {},\n  \"normalization_reference\": {},\n",
        cfg.ic_mesh.unwrap_or(cfg.mesh),
        cfg.exact_lattice,
        if cs.base_norm {
            format!("\"base experiment: A, a_init, R_c computed for mesh {BASE_MESH}, lattice {BASE_LATTICE}^2, R_c = 0.35 cells of the {BASE_MESH} mesh (identical to exp-20261021-0022)\"")
        } else {
            "\"this run's mesh and lattice\"".to_string()
        }
    );
    match meta.find("{\n") {
        Some(i) => format!("{}{}{}", &meta[..i + 2], extra, &meta[i + 2..]),
        None => meta,
    }
}

fn json_list(v: &[f64]) -> String {
    v.iter().map(|a| format!("{a:.6}")).collect::<Vec<_>>().join(", ")
}

fn meta_json(
    cfg: &CosmoConfig,
    side: u32,
    np: u32,
    table: &ModeTable,
    snaps: &[(f64, String)],
    norm: Option<cosmo::ScienceNorm>,
    model: &ModelSpec,
    sigma_in_box: Option<f64>,
) -> String {
    let s = table.spectrum;
    let _ = s.n;
    let cell = BOX / cfg.mesh as f64;
    let particle_mass = BOX * BOX / np as f64;
    let snap_as: Vec<f64> = snaps.iter().map(|x| x.0).collect();
    let snap_files: Vec<String> = snaps.iter().map(|x| format!("\"{}\"", x.1)).collect();
    let pp = cosmo::PpParams::for_mesh(cfg.mesh, cfg.eps);
    let dm = match cfg.k_hm {
        None => "{\"model\": \"cdm\"}".to_string(),
        Some(k) => {
            let lam = 2.0 * std::f64::consts::PI / k;
            let m = half_mode_mass(k);
            format!(
                "{{\"model\": \"wdm\", \"transfer\": \"T(k) = [1 + (alpha k)^(2 nu)]^(-5/nu), P_WDM = T^2 P_CDM (Viel et al. 2005), same phases as CDM\", \"nu\": {WDM_NU}, \"k_hm\": {k:.6}, \"alpha\": {:.6e}, \"lambda_hm\": {lam:.6}, \"M_hm\": {m:.6e}, \"M_hm_particles\": {:.3}, \"M_hm_definition\": \"rho_mean * pi * (lambda_hm/2)^2 with rho_mean = 1, lambda_hm = 2 pi / k_hm\"}}",
                wdm_alpha_for_khm(k),
                m / particle_mass
            )
        }
    };
    let normalization = match norm {
        Some(n) => format!(
            "{{\"method\": \"science\", \"host_particles\": {:.0}, \"R_host_lagrangian\": {:.4}, \"sigma_tophat_host_a1\": {}, \"sigma_tophat_host_a1_definition\": \"continuum integral of the linear P(k) from k = 0 (the normalisation target)\", \"sigma_tophat_host_a1_in_box\": {:.4}, \"sigma_tophat_host_a1_in_box_definition\": \"same top-hat, summed over the modes present in the periodic box (|k| >= 2 pi/L); this is the variance actually realised\", \"sigma_mesh_a_init\": {}, \"k_nl_a1\": {:.5}, \"k_fundamental\": {:.6}, \"reasoning\": \"A and a_init are set so that (1) the linear rms density contrast in a 2D top-hat containing 1e6 particles reaches {} at a = 1 when P(k) is integrated from k = 0. CORRECTION: the periodic box has no modes below 2 pi/L, so the in-box value is {:.3} (sigma_tophat_host_a1_in_box): 1e6-particle hosts are ~1.5-2 sigma rarer than the k = 0 value suggests and the largest hosts are somewhat below 1e6 particles. The normalisation was kept unchanged so that all models of the experiment are identical except for the WDM transfer function. k_nl(a=1) stays well inside the box, and (2) the rms delta on the mesh at a_init is {} so Zel'dovich ICs start far from shell crossing. n = {} gives sigma(R) ~ R^-(n+2)/2, a hierarchical spectrum with rich substructure; R_c = {} mesh cells keeps power up to near the mesh Nyquist so the WDM cutoffs are resolved; k_nl(a=1) = {:.3} is ~{:.0} k_fundamental, well inside the box.\"}}",
            SCI_HOST_PARTICLES,
            n.r_host,
            n.sigma_host_a1,
            sigma_in_box.unwrap_or(f64::NAN),
            n.sigma_mesh_init,
            n.k_nl,
            2.0 * std::f64::consts::PI / BOX,
            n.sigma_host_a1,
            sigma_in_box.unwrap_or(f64::NAN),
            n.sigma_mesh_init,
            s.n,
            cfg.rc_cells,
            n.k_nl,
            n.k_nl / (2.0 * std::f64::consts::PI / BOX)
        ),
        None => format!("{{\"method\": \"sigma_mesh_at_a_init\", \"sigma\": {}}}", cfg.sigma_init),
    };
    let pp_json = if cfg.pp {
        format!(
            "{{\"enabled\": true, \"split\": \"Gaussian: mesh phi_k exp(-k^2 r_s^2); short range g_pp r_vec/(r^2+eps^2) exp(-r^2/(4 r_s^2)) for r < r_cut, g_pp = C L^2/(2 pi N_p)\", \"r_s\": {:.6}, \"r_s_cells\": 1.25, \"r_cut\": {:.6}, \"r_cut_over_r_s\": 4.5, \"eps\": {:.6}, \"eps_over_mean_spacing\": {}, \"softening\": \"Plummer-like 1/(r^2+eps^2)\", \"neighbour_search\": \"chaining mesh, cells >= r_cut, GPU counting sort + prefix sum, 3x3 cells\"}}",
            pp.r_s, pp.r_cut, cfg.eps, cfg.eps / (BOX / side as f64)
        )
    } else {
        format!("{{\"enabled\": false, \"mesh_smoothing_r_s\": {:.6}}}", cfg.smooth_cells * cell)
    };
    let stepping = if cfg.science {
        format!(
            "\"uncapped science run: global step dt = sqrt(2 eta eps / A_max), A = |g|/a^3 (comoving acceleration), dln a = a^(-3/2) dt H0, eta = {}, growth <= 25%/step, floor dln a = {}, ceiling {}; floor/over-CFL statistics in run.log\"",
            cfg.eta,
            cfg.min_dlna,
            cosmo::MAX_DLNA
        )
    } else {
        format!("\"real-time: uniform in ln a over {} s, {} <= dln a <= {}\"", cfg.run_seconds, cosmo::MIN_DLNA, cosmo::MAX_DLNA)
    };
    format!(
        r#"{{
  "box": {box_l:.1},
  "mesh": {mesh},
  "particles": {np},
  "lattice": {side},
  "particle_mass": {pmass:.9e},
  "mass_units": "rho_mean = 1 per unit area, box mass = L^2 = {boxmass:.1}",
  "particle_ids": "a particle's index in every snapshot file is its stable ID = lattice index i (q = ((i % lattice) + 0.5, (i / lattice) + 0.5) * box/lattice)",
  "seed": {seed},
  "pair_id": {pair},
  "dm": {dm},
  "a_init": {a_init},
  "a_final": {a_final},
  "spectrum": {{"shape": "powerlaw_gauss", "A": {amp:.9e}, "A_from": "{amp_from}", "n": {n_cfg}, "modes_binned": {modes}, "R_c": {rc:.6}, "R_c_cells": {rc_cells}, "definition": "P0_CDM(k) = A k^n exp(-(k R_c)^2) at a_init (k in 1/box-length-units); WDM multiplies by T(k)^2"}},
  "normalization": {normalization},
  "phase_flipped": {flip},
  "growth": "D=a",
  "pk_convention": "delta_k=(1/N^2)sum delta e^-ikx; P=L^2<|delta_k|^2>",
  "poisson_coefficient": {c},
  "units": {{"length": "box units (box = {box_l:.1})", "time": "1/H0 with H0 = 1", "expansion": "Einstein-de Sitter: H = H0 a^-3/2, dt = a^(1/2) da / H0"}},
  "equations": "dx/dt = p/a^2, dp/dt = -grad(phi)/a, laplacian(phi) = C delta with C = 3/2 H0^2 (comoving x, canonical momentum p = a^2 dx/dt); linear growth a^2 D'' + 3/2 a D' - C/H0^2 D = 0 => D = a, D_- = a^-3/2",
  "integrator": "KDK leapfrog with merged kicks (Quinn et al. 1997): kick factor int dt/a = 2(a2^1/2 - a1^1/2), drift factor int dt/a^2 = 2(a1^-1/2 - a2^-1/2); steps clamped onto snapshot epochs",
  "stepping": {stepping},
  "pm": {{"periodic": true, "mesh": {mesh}, "cell": {cell:.6}, "assignment": "CIC", "green": "phi_k = -C delta_k S(k) / (k^2 W_cic(k)^2), k=0 removed", "force": "spectral gradient a_k = -i k phi_k", "nyquist_modes": "zeroed"}},
  "short_range": {pp_json},
  "initial_conditions": "Zel'dovich from a lattice x lattice particle grid (cell-centred; lattice a multiple of mesh/2); fixed-amplitude Gaussian field |delta_k| = sqrt(P0(k)/L^2) with random phases from a seeded white-noise field (Hermitian), k = 0 and Nyquist row/column zeroed; Psi_k = i k delta_k / k^2 deconvolved by the CIC window, CIC-interpolated to the lattice; x = q + (a/a_init) Psi, p = H0 a^(3/2) Psi / a_init",
  "pk_measurement": {{"bins": {bins}, "binning": "log-spaced from the fundamental 2pi/L to the Nyquist pi N/L; mean k per bin; Nyquist row/column excluded", "cic_deconvolved": true, "shot_noise_subtracted": {shot:.6e}, "cadence": "every {measure} in ln a plus at snapshot epochs and a_final", "p_linear": "(a/a_init)^2 * <P0(k)> over the bin's modes, P0 including T^2 for WDM"}},
  "snapshot_format": "little-endian: 8-byte magic GSNAP001, u64 count, f32 a, f32 box, count x [f32 x, f32 y] comoving positions wrapped into [0, box), count x [f32 vx, f32 vy] peculiar velocities v = a dx/dt = p/a in box units per (1/H0); record i is particle ID i in every snapshot",
  "snapshots": [{snap_as}],
  "snapshot_files": [{snap_files}],
  "snapshots_enabled": {snaps_on},
  "snapshot_epochs_requested": [{snap_req}],
  "notes": "2D cosmology (planar gravity with a 1/k^2 Poisson kernel). Lattice ICs have sub-Poisson noise, so subtracting the Poisson shot noise L^2/N_p under-estimates P at high k until shell crossing. Velocities in snapshots are synchronized with positions (partial kick). A run is complete when the file COMPLETE exists in its directory."
}}
"#,
        box_l = BOX,
        mesh = cfg.mesh,
        pmass = particle_mass,
        boxmass = BOX * BOX,
        seed = cfg.seed,
        pair = model.pair_id.as_ref().map_or("null".to_string(), |p| format!("\"{p}\"")),
        a_init = cfg.a_init,
        a_final = cfg.a_final,
        amp = s.amp,
        n_cfg = cfg.spectral_index,
        modes = table.count.iter().sum::<u64>(),
        amp_from = if cfg.amp.is_some() { "science normalization" } else { "sigma on the mesh at a_init" },
        rc = s.rc,
        rc_cells = cfg.rc_cells,
        flip = cfg.flip_phases,
        c = cosmo::POISSON_C,
        bins = cosmo::PK_BINS,
        shot = BOX * BOX / np as f64,
        measure = cosmo::MEASURE_DLNA,
        snap_as = json_list(&snap_as),
        snap_files = snap_files.join(", "),
        snaps_on = cfg.snapshots,
        snap_req = json_list(&cfg.snapshot_as),
    )
}

/// CDM + 3 WDM models sharing one seed/pair_id, written to `<root>/experiments/<pair_id>/<model>/`.
pub struct Experiment {
    pub pair_id: String,
    pub dir: PathBuf,
    pub models: Vec<ModelSpec>,
    pub next: usize,
    pub started: Instant,
    pub summary: Vec<String>,
    pub np: f64,
}

impl Experiment {
    pub fn new(root: &Path, pair_id: Option<String>, np: f64, cs: &CosmoSettings) -> Self {
        let pair_id = pair_id.unwrap_or_else(|| {
            let s = cosmo_io::utc_stamp(); // YYYYMMDD-HHMMSS
            format!("exp-{}", &s[..13])
        });
        let dir = match &cs.exp_dir {
            Some(d) => PathBuf::from(d),
            None => root.join("experiments").join(&pair_id),
        };
        let mut models = Vec::new();
        if cs.exp_cdm {
            models.push(ModelSpec::cdm());
        }
        match &cs.exp_wdm_mhm {
            // explicit list: M_hm in particle masses of the base 6144^2 lattice (k_hm as the base)
            Some(list) => {
                let base_np = (BASE_LATTICE as f64) * (BASE_LATTICE as f64);
                for &m in list {
                    models.push(ModelSpec::wdm(khm_for_mass(m, base_np)));
                }
            }
            None => {
                for m in EXPERIMENT_MHM_PARTICLES {
                    models.push(ModelSpec::wdm(khm_for_mass(m, np)));
                }
            }
        }
        for m in &mut models {
            m.out_dir = Some(dir.join(&m.name));
            m.pair_id = Some(pair_id.clone());
        }
        Self { pair_id, dir, models, next: 0, started: Instant::now(), summary: Vec::new(), np }
    }

    /// Next model whose directory is not COMPLETE (resumable).
    pub fn next_model(&mut self) -> Option<ModelSpec> {
        while self.next < self.models.len() {
            let m = self.models[self.next].clone();
            self.next += 1;
            let done = m.out_dir.as_ref().is_some_and(|d| d.join("COMPLETE").exists());
            if done {
                println!("experiment {}: {} already complete, skipping", self.pair_id, m.name);
                continue;
            }
            let mut m = m;
            if let Some(dir) = &m.out_dir {
                m.resume = cosmo_io::latest_snapshot(dir, self.np as u64);
                if let Some((a, p)) = &m.resume {
                    println!("experiment {}: {} interrupted earlier; resuming from {} (a = {a:.4})", self.pair_id, m.name, p.display());
                }
            }
            return Some(m);
        }
        None
    }
}
