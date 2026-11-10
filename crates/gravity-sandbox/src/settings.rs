//! Command-line settings (std::env::args, no external parser).

use sim_core::Scenario;

pub const BODIES_MIN: usize = 1_000;
pub const BODIES_MAX: usize = 64_000;

pub const HELP: &str = "\
Usage: gravity_sandbox [options]

Options:
  --preset normal|crazy|cosmic
                          normal: galaxy, 4000 bodies, 1M dust, windowed (default)
                          crazy:  subhalo merger, 50000 bodies, 32M dust (or the adapter
                                  max), 2048^2 PM mesh, borderless fullscreen, bloom
                          cosmic: periodic comoving cosmic-web run (Zel'dovich ICs, D=a),
                                  ~32M particles (6144^2 lattice: the multiple of mesh/2
                                  nearest to sqrt(32M)), 2048^2 mesh, fullscreen
  --scenario galaxy|subhalo|cosmic
  --bodies N              number of bodies (1000..64000), e.g. 20000 or 20k
  --dust N                dust particles, e.g. 16M, 500k, 2000000 (125k..adapter max, <=64M)
  --vsync                 start with VSync (Fifo) instead of uncapped
  --no-bloom              disable bloom
  --fullscreen            borderless fullscreen
  --dark-matter X         dust mass / total body mass (default 1.0 subhalo, 0.5 galaxy)
  --mesh N                PM mesh cells per side: 256|512|1024|2048 (default 1024,
                          crazy 2048; FFT grid is 2N x 2N, isolated boundaries)
  --field M               dust field size read back for the bodies (default 512; also the
                          bodies' mass-smoothing scale on the mesh)
  --no-pm                 no dust self-gravity (dust follows the 64 heaviest bodies)
  --pm-selftest           run the GPU PM point-mass validation at startup and exit
  --background            open the window without taking keyboard focus (benchmarks)

Cosmic web (--scenario cosmic):
  --spectral-index n      initial P0(k) = A k^n exp(-(k R_c)^2), default -1
  --a-init a              start scale factor (default 0.02)
  --a-final a             end scale factor (default 1.0); the run then holds
  --run-seconds s         real time from a_init to a_final (default 90)
  --sigma s               rms density contrast on the mesh at a_init (default 0.05)
  --seed N                run seed (default 1; R starts a new run with seed+1)
  --no-snapshots          skip particle snapshots (P(k) is still written)
  --flip-phases           paired-fixed partner run (delta -> -delta, same seed); average its
                          P(k) with the unflipped run to cancel realization-specific terms
  --runs-dir PATH         output root (default D:\\cosmo-runs)

Science / experiment (uncapped, P3M short-range force, adaptive steps):
  --science               uncapped science run of the cosmic web (P3M, CFL steps,
                          16 snapshots log-spaced in a from 0.1 to 1)
  --experiment            CDM + 3 WDM models (M_hm = 3000, 300, 30 particle masses), same
                          seed, 4096^2 mesh, 6144^2 particles, into
                          <runs-dir>\\experiments\\<pair_id>\\<model>\\ (resumable)
  --pair-id ID            experiment id (default exp-<UTC date-time>); reuse to resume
  --dm cdm|wdm            dark-matter model of a single run; --k-hm X sets the WDM half-mode
  --k-hm X                WDM half-mode wavenumber (T^2(k_hm) = 1/4, Viel et al. 2005)
  --eps-frac f            PP softening / mean interparticle spacing (default 0.08)
  --eta x                 CFL parameter, dt = sqrt(2 eta eps / a_max) (default 0.025)
  --min-dlna x            smallest step in ln a (default 0.002)
  --host-sigma s          linear sigma of a 1e6-particle top-hat at a = 1 (default 1.15)
  --rc-cells x            spectrum cutoff R_c in mesh cells
  --pp / --no-pp          force the short-range PP force on/off
  --force-test            GPU two-point force test (mesh-only vs P3M) and exit

Convergence suite (one change from the base experiment per variant):
  --exp-dir DIR           experiment directory (models go to DIR\\<model>\\)
  --exp-models LIST       e.g. cdm,3000: CDM and WDM models by M_hm in particle masses of
                          the 6144^2 base lattice (k_hm identical to the base experiment)
  --lattice N             exact N x N particle lattice (e.g. 3072 on the 4096 mesh)
  --eps X                 absolute PP softening (overrides --eps-frac)
  --base-norm             normalise exactly as the base (mesh 4096, 6144^2, R_c 0.35 cells)
  --ic-mesh N             generate the ICs on an N mesh (same phases), run on --mesh
  --variant NAME          variant name recorded in meta.json; --variant-note TEXT
  --determinism-test      run 100 P3M steps twice from the same ICs, require byte-identical
                          particles (exit code 0 = deterministic), and exit
  --help                  show this help

Explicit options override the preset regardless of their order.";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Preset {
    Normal,
    Crazy,
    Cosmic,
    Experiment,
}

#[derive(Clone, Debug)]
pub struct Settings {
    pub preset: Preset,
    pub scenario: Scenario,
    pub bodies: usize,
    pub dust: u32,
    pub vsync: bool,
    pub bloom: bool,
    pub fullscreen: bool,
    /// Dust mass / body mass; None = scenario default.
    pub dark_matter: Option<f32>,
    pub mesh: u32,
    pub field: u32,
    pub pm: bool,
    pub pm_selftest: bool,
    /// Open the window without taking keyboard focus (benchmark/capture runs).
    pub background: bool,
    pub cosmo: CosmoSettings,
}

#[derive(Clone, Debug)]
pub struct CosmoSettings {
    /// None = mode default (-1 real-time, -1.25 science)
    pub spectral_index: Option<f64>,
    /// None = mode default (0.02 real-time; science: from the normalisation)
    pub a_init: Option<f64>,
    pub a_final: f64,
    pub run_seconds: f64,
    /// rms δ on the mesh at a_init (None = 0.05 real-time, 0.1 science)
    pub sigma: Option<f64>,
    pub seed: u64,
    pub snapshots: bool,
    pub runs_dir: String,
    pub flip_phases: bool,
    /// WDM half-mode wavenumber (None = CDM)
    pub k_hm: Option<f64>,
    /// uncapped adaptive-step run with P³M
    pub science: bool,
    /// CDM + 3 WDM batch
    pub experiment: bool,
    pub pair_id: Option<String>,
    /// short-range PP force (None = on in science mode)
    pub pp: Option<bool>,
    /// PP softening in units of the mean interparticle spacing
    pub eps_frac: f64,
    /// CFL parameter
    pub eta: f64,
    /// smallest Δln a per step in science mode
    pub min_dlna: f64,
    /// linear top-hat σ of a 1e6-particle host at a = 1 (science normalisation)
    pub host_sigma: f64,
    /// spectrum cutoff R_c in mesh cells (None = 1.5 real-time, 0.35 science)
    pub rc_cells: Option<f64>,
    /// run the GPU two-point force test and exit
    pub force_test: bool,
    /// experiment directory (overrides <runs-dir>\experiments\<pair_id>)
    pub exp_dir: Option<String>,
    /// WDM models of the experiment as M_hm in particle masses of the 6144^2 base lattice
    /// (None = 3000, 300, 30 of the run's own lattice)
    pub exp_wdm_mhm: Option<Vec<f64>>,
    /// run only the CDM model of the experiment when false
    pub exp_cdm: bool,
    /// exact particle lattice side (no rounding to a multiple of mesh/2)
    pub lattice: Option<u32>,
    /// absolute PP softening (overrides eps_frac)
    pub eps_abs: Option<f64>,
    /// normalise as the base experiment (mesh 4096, lattice 6144, R_c = 0.35 cells of 4096)
    pub base_norm: bool,
    /// mesh on which the ICs are generated (None = run mesh)
    pub ic_mesh: Option<u32>,
    /// convergence-suite variant name and description (meta.json "variant")
    pub variant: Option<String>,
    pub variant_note: Option<String>,
    /// run the GPU P3M run-to-run determinism test and exit
    pub determinism_test: bool,
}

impl Default for CosmoSettings {
    fn default() -> Self {
        Self {
            spectral_index: None,
            a_init: None,
            a_final: 1.0,
            run_seconds: 90.0,
            sigma: None,
            seed: 1,
            snapshots: true,
            runs_dir: r"D:\cosmo-runs".to_string(),
            flip_phases: false,
            k_hm: None,
            science: false,
            experiment: false,
            pair_id: None,
            pp: None,
            eps_frac: 0.08,
            eta: 0.025,
            min_dlna: 0.002,
            host_sigma: 1.15,
            rc_cells: None,
            force_test: false,
            exp_dir: None,
            exp_wdm_mhm: None,
            exp_cdm: true,
            lattice: None,
            eps_abs: None,
            base_norm: false,
            ic_mesh: None,
            variant: None,
            variant_note: None,
            determinism_test: false,
        }
    }
}
impl Settings {
    pub fn preset(p: Preset) -> Self {
        match p {
            Preset::Experiment => {
                let mut s = Self::preset(Preset::Cosmic);
                s.preset = p;
                s.mesh = 4096;
                // 6144^2 (37.7M) rather than 8192^2 (67.1M): measured P3M cost per step at 8192^2 was
                // 0.15 s early and 0.7 s by a = 0.5 (rising with clustering), which puts the 4-model
                // experiment well over an hour; PP cost scales ~N^2 at fixed box.
                s.dust = 6144 * 6144;
                s.cosmo.min_dlna = 0.001;
                s.fullscreen = false;
                s.pm = false;
                s.cosmo.experiment = true;
                s.cosmo.science = true;
                s
            }
            Preset::Cosmic => Self {
                preset: p,
                scenario: Scenario::CosmicWeb,
                ..Self::preset(Preset::Crazy)
            },
            Preset::Normal => Self {
                preset: p,
                scenario: Scenario::Galaxy,
                bodies: 4_000,
                dust: 1_000_000,
                vsync: false,
                bloom: true,
                fullscreen: false,
                dark_matter: None,
                mesh: crate::pm::MESH_DEFAULT,
                field: crate::pm::FIELD_DEFAULT,
                pm: true,
                pm_selftest: false,
                background: false,
                cosmo: CosmoSettings::default(),
            },
            Preset::Crazy => Self {
                preset: p,
                scenario: Scenario::SubhaloMerger,
                bodies: 50_000,
                dust: 32_000_000,
                vsync: false,
                bloom: true,
                fullscreen: true,
                dark_matter: None,
                mesh: crate::pm::MESH_MAX,
                field: crate::pm::FIELD_DEFAULT,
                pm: true,
                pm_selftest: false,
                background: false,
                cosmo: CosmoSettings::default(),
            },
        }
    }
}

/// Parse "16M", "500k", "2_000_000", "1.5m" into a count.
pub fn parse_count(s: &str) -> Option<u64> {
    let s = s.trim().replace('_', "");
    let (num, mul) = match s.chars().last()? {
        'k' | 'K' => (&s[..s.len() - 1], 1e3),
        'm' | 'M' => (&s[..s.len() - 1], 1e6),
        'g' | 'G' | 'b' | 'B' => (&s[..s.len() - 1], 1e9),
        _ => (&s[..], 1.0),
    };
    let v: f64 = num.parse().ok()?;
    if !v.is_finite() || v < 0.0 {
        return None;
    }
    Some((v * mul).round() as u64)
}

pub enum ParseOutcome {
    Run(Settings),
    Help,
}

pub fn parse_args<I: IntoIterator<Item = String>>(args: I) -> Result<ParseOutcome, String> {
    let args: Vec<String> = args.into_iter().collect();

    // Pass 1: preset (so explicit options always win).
    let mut preset = Preset::Normal;
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        if a == "--help" || a == "-h" {
            return Ok(ParseOutcome::Help);
        }
        if a == "--experiment" {
            preset = Preset::Experiment;
        }
        if let Some(v) = value_of(&args, &mut i, "--preset")? {
            preset = match v.to_ascii_lowercase().as_str() {
                "normal" => Preset::Normal,
                "crazy" => Preset::Crazy,
                "cosmic" => Preset::Cosmic,
                other => return Err(format!("unknown preset '{other}' (normal|crazy|cosmic)")),
            };
        }
        i += 1;
    }
    let mut s = Settings::preset(preset);

    // Pass 2: everything else.
    let mut i = 0;
    while i < args.len() {
        let a = args[i].clone();
        if value_of(&args, &mut i, "--preset")?.is_some() {
        } else if let Some(v) = value_of(&args, &mut i, "--scenario")? {
            s.scenario = Scenario::from_name(&v)
                .ok_or_else(|| format!("unknown scenario '{v}' (galaxy|subhalo|cosmic)"))?;
        } else if let Some(v) = value_of(&args, &mut i, "--bodies")? {
            let n = parse_count(&v).ok_or_else(|| format!("bad --bodies value '{v}'"))?;
            s.bodies = (n as usize).clamp(BODIES_MIN, BODIES_MAX);
        } else if let Some(v) = value_of(&args, &mut i, "--dust")? {
            let n = parse_count(&v).ok_or_else(|| format!("bad --dust value '{v}'"))?;
            s.dust = n.min(u32::MAX as u64) as u32; // clamped to the adapter range later
        } else if let Some(v) = value_of(&args, &mut i, "--dark-matter")? {
            let x: f32 = v.parse().map_err(|_| format!("bad --dark-matter value '{v}'"))?;
            if !(0.0..=100.0).contains(&x) {
                return Err(format!("--dark-matter must be in 0..100, got {x}"));
            }
            s.dark_matter = Some(x);
        } else if let Some(v) = value_of(&args, &mut i, "--spectral-index")? {
            s.cosmo.spectral_index = Some(parse_f(&v, "--spectral-index", -3.0, 2.0)?);
        } else if let Some(v) = value_of(&args, &mut i, "--a-init")? {
            s.cosmo.a_init = Some(parse_f(&v, "--a-init", 1e-4, 0.5)?);
        } else if let Some(v) = value_of(&args, &mut i, "--a-final")? {
            s.cosmo.a_final = parse_f(&v, "--a-final", 0.01, 10.0)?;
        } else if let Some(v) = value_of(&args, &mut i, "--run-seconds")? {
            s.cosmo.run_seconds = parse_f(&v, "--run-seconds", 1.0, 36_000.0)?;
        } else if let Some(v) = value_of(&args, &mut i, "--sigma")? {
            s.cosmo.sigma = Some(parse_f(&v, "--sigma", 1e-4, 1.0)?);
        } else if let Some(v) = value_of(&args, &mut i, "--dm")? {
            match v.to_ascii_lowercase().as_str() {
                "cdm" => s.cosmo.k_hm = None,
                "wdm" => {
                    if s.cosmo.k_hm.is_none() {
                        s.cosmo.k_hm = Some(1.0);
                    }
                }
                other => return Err(format!("unknown --dm '{other}' (cdm|wdm)")),
            }
        } else if let Some(v) = value_of(&args, &mut i, "--k-hm")? {
            s.cosmo.k_hm = Some(parse_f(&v, "--k-hm", 1e-3, 100.0)?);
        } else if let Some(v) = value_of(&args, &mut i, "--pair-id")? {
            s.cosmo.pair_id = Some(v);
        } else if let Some(v) = value_of(&args, &mut i, "--eps-frac")? {
            s.cosmo.eps_frac = parse_f(&v, "--eps-frac", 0.001, 1.0)?;
        } else if let Some(v) = value_of(&args, &mut i, "--eta")? {
            s.cosmo.eta = parse_f(&v, "--eta", 1e-4, 1.0)?;
        } else if let Some(v) = value_of(&args, &mut i, "--min-dlna")? {
            s.cosmo.min_dlna = parse_f(&v, "--min-dlna", 1e-6, 0.01)?;
        } else if let Some(v) = value_of(&args, &mut i, "--host-sigma")? {
            s.cosmo.host_sigma = parse_f(&v, "--host-sigma", 0.1, 20.0)?;
        } else if let Some(v) = value_of(&args, &mut i, "--rc-cells")? {
            s.cosmo.rc_cells = Some(parse_f(&v, "--rc-cells", 0.0, 10.0)?);
        } else if let Some(v) = value_of(&args, &mut i, "--exp-dir")? {
            s.cosmo.exp_dir = Some(v);
        } else if let Some(v) = value_of(&args, &mut i, "--exp-models")? {
            // e.g. "cdm,3000": cdm and/or WDM half-mode masses in 6144^2-lattice particle masses
            let mut wdm = Vec::new();
            s.cosmo.exp_cdm = false;
            for t in v.split(',').map(str::trim).filter(|t| !t.is_empty()) {
                if t.eq_ignore_ascii_case("cdm") {
                    s.cosmo.exp_cdm = true;
                } else {
                    wdm.push(parse_f(t, "--exp-models", 1.0, 1e9)?);
                }
            }
            s.cosmo.exp_wdm_mhm = Some(wdm);
        } else if let Some(v) = value_of(&args, &mut i, "--lattice")? {
            let n: u32 = v.parse().map_err(|_| format!("bad --lattice value '{v}'"))?;
            if !(64..=11_585).contains(&n) {
                return Err(format!("--lattice must be in 64..11585, got {n}"));
            }
            s.cosmo.lattice = Some(n);
            s.dust = n * n;
        } else if let Some(v) = value_of(&args, &mut i, "--eps")? {
            s.cosmo.eps_abs = Some(parse_f(&v, "--eps", 1e-5, 100.0)?);
        } else if let Some(v) = value_of(&args, &mut i, "--ic-mesh")? {
            let n = parse_count(&v).ok_or_else(|| format!("bad --ic-mesh value '{v}'"))?;
            s.cosmo.ic_mesh = Some((n.clamp(256, 4096) as u32).next_power_of_two().min(4096));
        } else if let Some(v) = value_of(&args, &mut i, "--variant")? {
            s.cosmo.variant = Some(v);
        } else if let Some(v) = value_of(&args, &mut i, "--variant-note")? {
            s.cosmo.variant_note = Some(v);
        } else if let Some(v) = value_of(&args, &mut i, "--seed")? {
            s.cosmo.seed = v.parse().map_err(|_| format!("bad --seed value '{v}'"))?;
        } else if let Some(v) = value_of(&args, &mut i, "--runs-dir")? {
            s.cosmo.runs_dir = v;
        } else if let Some(v) = value_of(&args, &mut i, "--mesh")? {
            let n = parse_count(&v).ok_or_else(|| format!("bad --mesh value '{v}'"))?;
            // the isolated PM fits 2048 at most; the periodic cosmology mesh goes to 4096
            s.mesh = (n.clamp(256, 4096) as u32).next_power_of_two().min(4096);
        } else if let Some(v) = value_of(&args, &mut i, "--field")? {
            let n = parse_count(&v).ok_or_else(|| format!("bad --field value '{v}'"))?;
            s.field = (n.clamp(16, 1024) as u32).next_power_of_two();
        } else {
            match a.as_str() {
                "--vsync" => s.vsync = true,
                "--no-vsync" => s.vsync = false,
                "--no-bloom" => s.bloom = false,
                "--bloom" => s.bloom = true,
                "--fullscreen" => s.fullscreen = true,
                "--windowed" => s.fullscreen = false,
                "--no-pm" => s.pm = false,
                "--pm" => s.pm = true,
                "--pm-selftest" => s.pm_selftest = true,
                "--background" => s.background = true,
                "--no-snapshots" => s.cosmo.snapshots = false,
                "--flip-phases" => s.cosmo.flip_phases = true,
                "--experiment" => {}
                "--science" => {
                    s.cosmo.science = true;
                    s.scenario = Scenario::CosmicWeb;
                }
                "--pp" => s.cosmo.pp = Some(true),
                "--no-pp" => s.cosmo.pp = Some(false),
                "--force-test" => s.cosmo.force_test = true,
                "--base-norm" => s.cosmo.base_norm = true,
                "--determinism-test" => s.cosmo.determinism_test = true,
                other => return Err(format!("unknown argument '{other}' (see --help)")),
            }
        }
        i += 1;
    }
    Ok(ParseOutcome::Run(s))
}

fn parse_f(v: &str, name: &str, lo: f64, hi: f64) -> Result<f64, String> {
    let x: f64 = v.parse().map_err(|_| format!("bad {name} value '{v}'"))?;
    if !(lo..=hi).contains(&x) {
        return Err(format!("{name} must be in {lo}..{hi}, got {x}"));
    }
    Ok(x)
}

/// Matches `--name value` or `--name=value`; advances `i` past a separate value.
fn value_of(args: &[String], i: &mut usize, name: &str) -> Result<Option<String>, String> {
    let a = &args[*i];
    if a == name {
        let v = args
            .get(*i + 1)
            .ok_or_else(|| format!("{name} needs a value"))?
            .clone();
        *i += 1;
        Ok(Some(v))
    } else if let Some(rest) = a.strip_prefix(name).and_then(|r| r.strip_prefix('=')) {
        Ok(Some(rest.to_string()))
    } else {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(s: &str) -> Vec<String> {
        s.split_whitespace().map(String::from).collect()
    }

    #[test]
    fn counts() {
        assert_eq!(parse_count("16M"), Some(16_000_000));
        assert_eq!(parse_count("500k"), Some(500_000));
        assert_eq!(parse_count("1.5m"), Some(1_500_000));
        assert_eq!(parse_count("2_000"), Some(2_000));
        assert_eq!(parse_count("x"), None);
    }

    #[test]
    fn preset_then_override() {
        let ParseOutcome::Run(s) =
            parse_args(args("--dust 8M --preset crazy --no-bloom")).unwrap()
        else {
            panic!()
        };
        assert_eq!(s.scenario, Scenario::SubhaloMerger);
        assert_eq!(s.dust, 8_000_000);
        assert!(!s.bloom && s.fullscreen);
        assert_eq!(s.bodies, 50_000);
    }

    #[test]
    fn scenario_and_errors() {
        let ParseOutcome::Run(s) = parse_args(args("--scenario=subhalo --bodies 50k")).unwrap()
        else {
            panic!()
        };
        assert_eq!(s.scenario, Scenario::SubhaloMerger);
        assert_eq!(s.bodies, 50_000);
        assert!(parse_args(args("--bogus")).is_err());
        let ParseOutcome::Run(s) = parse_args(args("--dark-matter 2.5 --mesh 1000 --no-pm")).unwrap() else {
            panic!()
        };
        assert_eq!(s.dark_matter, Some(2.5));
        let ParseOutcome::Run(c) = parse_args(args("--preset cosmic --spectral-index -1.5 --a-final 0.5 --run-seconds 30 --no-snapshots")).unwrap() else {
            panic!()
        };
        assert_eq!(c.scenario, Scenario::CosmicWeb);
        assert_eq!(c.dust, 32_000_000);
        assert_eq!(c.mesh, 2048);
        assert_eq!(c.cosmo.spectral_index, Some(-1.5));
        assert_eq!(c.cosmo.a_final, 0.5);
        assert!(!c.cosmo.snapshots && c.fullscreen);
        assert!(parse_args(args("--a-final 50")).is_err());
        assert_eq!(s.mesh, 1024);
        assert!(!s.pm);
        assert!(parse_args(args("--dust")).is_err());
        assert!(matches!(parse_args(args("--help")), Ok(ParseOutcome::Help)));
    }
}
