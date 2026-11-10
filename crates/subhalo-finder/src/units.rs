//! Run metadata, unit conventions, and the 2D comoving binding-energy criterion.
//!
//! # Equations of motion (from the simulation's `meta.json`)
//! Comoving position `x` (box units), canonical momentum `p = a^2 dx/dt`, time in 1/H0,
//! Einstein-de Sitter background (H = H0 a^-3/2, H0 = 1):
//!
//! ```text
//! dx/dt = p / a^2,    dp/dt = -grad_x(phi) / a,    lap_x(phi) = C delta,   C = 3/2 H0^2
//! ```
//!
//! # Peculiar energy
//! With the peculiar velocity `v = a dx/dt = p/a` (what the snapshots store, in box units
//! per 1/H0) and physical coordinates `r = a x`:
//!
//! ```text
//! dv/dt = (dp/dt)/a - H v = -grad_x(phi)/a^2 - H v = -grad_r(Phi) - H v,   Phi = phi / a
//! ```
//!
//! so `E = v^2/2 + Phi` is the peculiar energy per unit mass, conserved apart from the
//! Hubble drag `-H v` (which only makes structures more bound; it is ignored, as in
//! SUBFIND). `Phi` must be the full peculiar potential, i.e. the solution of
//! `lap(phi) = C (rho/rho_bar - 1)`, including the uniform `-1` background term.
//!
//! # 2D potential of a particle set
//! Measure mass in comoving area units, `m = L^2 / N` per particle, so `rho_bar = 1` and
//! `rho/rho_bar = sum_j m delta_D(x - x_j)`. The 2D Green's function of the Laplacian is
//! `ln|x| / (2 pi)`, and `lap(|x|^2) = 4`, hence
//!
//! ```text
//! phi(x) = (C m / 2 pi) sum_j ln sqrt(|x - x_j|^2 + eps^2)  -  (C/4) |x - x_c|^2  (+ const)
//! ```
//!
//! with Plummer-like softening `eps` (the PM force smoothing length) and an arbitrary
//! centre `x_c` (a linear term is a uniform force, removed by the choice of frame).
//!
//! # Reference level (2D-specific)
//! The 2D log potential has no zero at infinity: nothing can escape to infinity, so the
//! 3D criterion `E < 0` with `Phi(inf) = 0` has no direct analogue. We measure energies
//! relative to the candidate's own potential at a reference radius
//! `R_ref = f_esc * R_ext`, where `R_ext` is the extent of the *initial* candidate
//! (distance of its farthest member from the potential minimum, i.e. the SUBFIND
//! saddle-point boundary): a particle is bound if, in the candidate's own field, it cannot
//! climb beyond `R_ref`. The default `f_esc = e` reproduces the 3D escape budget at the
//! boundary: in 3D, getting from the edge R to infinity costs `GM/R = v_c^2(R)`; in 2D,
//! getting from R to f R costs `v_c^2 ln f` (point-mass log potential), so `ln f = 1`.
//! (With `f_esc = 1`, i.e. "must not reach the saddle", large halos lose most of their
//! mass in a removal cascade: the depth of a 2D log well scales with the remaining mass.)
//! Using the monopole outside the candidate,
//!
//! ```text
//! E_i = 1/2 |v_i - v_bar|^2
//!     + (C / (2 pi a)) * [ m sum_{j != i} ln sqrt(r_ij^2 + eps^2) - M ln R_ref ]
//!     - (C / (4 a)) * ( |x_i - x_c|^2 - R_ref^2 )
//! ```
//!
//! with `M = n m` the current candidate mass, `x_c` its potential minimum, `v_bar` its
//! mean (peculiar) velocity, `r_ij` comoving. Bound <=> `E_i < 0`. Units: velocities in
//! box units per 1/H0 (physical peculiar), energies in their square. The comoving-vs-
//! physical length choice inside `ln` only shifts the potential by `C M ln(a)/(2 pi a)`,
//! which cancels against the same shift of the reference term.
//!
//! # Circular velocity
//! Gauss's law in 2D gives `|grad_x phi| = C (M(<r) - pi r^2) / (2 pi r)` (mass in area
//! units, the `pi r^2` is the background subtracted by `delta`). The physical circular
//! velocity `v_c^2 = r_phys |grad_r Phi| = (a r) |grad_x phi| / a^2` is therefore
//!
//! ```text
//! v_c^2(r) = C (M(<r) - pi rho_bar r^2) / (2 pi a)        (rho_bar = 1 in area units)
//! ```
//!
//! Note `C M(<r) / (2 pi r)` is the gravitational *acceleration*, not `v_c^2`. Without the
//! background term `v_c` would never decrease (in 2D a point mass has a flat rotation
//! curve), so Vmax/Rmax only exist because of the background subtraction: Rmax is where
//! the azimuthally averaged surface density of the subhalo falls to the cosmic mean.

use serde_json::Value;
use std::path::{Path, PathBuf};

/// How the snapshot velocity column relates to the peculiar velocity `a dx/dt`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VelConvention {
    /// `v = a dx/dt = p/a` (stored as is).
    Peculiar,
    /// Canonical momentum `p = a^2 dx/dt`: peculiar = p / a.
    Canonical,
    /// Comoving `dx/dt`: peculiar = a * dx/dt.
    Comoving,
}

impl VelConvention {
    pub fn factor(self, a: f64) -> f64 {
        match self {
            VelConvention::Peculiar => 1.0,
            VelConvention::Canonical => 1.0 / a,
            VelConvention::Comoving => a,
        }
    }

    pub fn parse(s: &str) -> Option<VelConvention> {
        match s.to_ascii_lowercase().as_str() {
            "peculiar" => Some(VelConvention::Peculiar),
            "canonical" => Some(VelConvention::Canonical),
            "comoving" => Some(VelConvention::Comoving),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            VelConvention::Peculiar => "peculiar v = a dx/dt",
            VelConvention::Canonical => "canonical p = a^2 dx/dt",
            VelConvention::Comoving => "comoving dx/dt",
        }
    }
}

#[derive(Clone, Debug)]
pub struct SnapInfo {
    pub a: f64,
    pub path: PathBuf,
}

#[derive(Clone, Debug)]
pub struct RunMeta {
    pub dir: PathBuf,
    pub box_size: f64,
    pub particles: u64,
    pub snapshots: Vec<SnapInfo>,
    /// Particle mass in the run's own mass units (from `particle_mass`), used for the
    /// `mass` columns. Defaults to the comoving-area unit L^2/N when absent.
    pub particle_mass: f64,
    pub particle_mass_source: String,
    /// Poisson coefficient C (3/2 H0^2 for EdS with H0 = 1).
    pub c_poisson: f64,
    /// Comoving softening length for the unbinding potential.
    pub softening: f64,
    pub softening_source: String,
    pub vel: VelConvention,
    pub vel_source: String,
    pub dark_matter: Option<Value>,
    pub raw: Value,
}

fn num(v: &Value, path: &[&str]) -> Option<f64> {
    let mut cur = v;
    for k in path {
        cur = cur.get(*k)?;
    }
    cur.as_f64().or_else(|| cur.as_str().and_then(|s| s.trim().parse().ok()))
}

/// Find the velocity convention from the velocity-specific descriptions in meta.json.
fn detect_velocity(raw: &Value) -> Option<(VelConvention, String)> {
    let mut texts = Vec::new();
    for k in ["velocity_units", "velocities", "velocity", "snapshot_velocity", "snapshot_format"] {
        if let Some(s) = raw.get(k).and_then(|x| x.as_str()) {
            texts.push((k, s.to_string()));
        } else if let Some(o) = raw.get(k).filter(|x| x.is_object()) {
            texts.push((k, o.to_string()));
        }
    }
    for (k, t) in texts {
        let t = t.to_ascii_lowercase().replace(' ', "");
        // Look only at the part describing velocities.
        let seg = match t.find("vel") {
            Some(i) => &t[i..],
            None => &t[..],
        };
        if seg.contains("v=adx/dt") || seg.contains("=p/a") || seg.contains("peculiar") {
            return Some((VelConvention::Peculiar, format!("meta.json `{k}`")));
        }
        if seg.contains("canonical") || seg.contains("p=a^2dx/dt") || seg.contains("p=a2dx/dt") {
            return Some((VelConvention::Canonical, format!("meta.json `{k}`")));
        }
        if seg.contains("dx/dt") {
            return Some((VelConvention::Comoving, format!("meta.json `{k}`")));
        }
    }
    None
}

impl RunMeta {
    pub fn load(dir: &Path, vel_override: Option<VelConvention>) -> Result<RunMeta, String> {
        let mp = dir.join("meta.json");
        let text = std::fs::read_to_string(&mp).map_err(|e| format!("{}: {e}", mp.display()))?;
        let raw: Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", mp.display()))?;
        let box_size = num(&raw, &["box"]).ok_or("meta.json: missing `box`")?;
        let particles = num(&raw, &["particles"]).ok_or("meta.json: missing `particles`")? as u64;
        let mut snapshots = Vec::new();
        for s in raw.get("snapshots").and_then(|x| x.as_array()).ok_or("meta.json: missing `snapshots`")? {
            let a = s.as_f64().or_else(|| num(s, &["a"])).ok_or("meta.json: snapshot without `a`")?;
            let file = s
                .get("file")
                .and_then(|f| f.as_str())
                .map(str::to_string)
                .unwrap_or_else(|| format!("snap_a{a:.3}.bin"));
            snapshots.push(SnapInfo { a, path: dir.join(file) });
        }
        snapshots.sort_by(|x, y| x.a.total_cmp(&y.a));
        let area_mass = box_size * box_size / particles as f64;
        let (particle_mass, particle_mass_source) = match num(&raw, &["particle_mass"]) {
            Some(m) if m > 0.0 => (m, "meta.json particle_mass".to_string()),
            _ => (area_mass, "default L^2/N (comoving area units, rho_bar = 1)".to_string()),
        };
        let c_poisson = num(&raw, &["poisson_coefficient"]).unwrap_or(1.5);
        // Priority: the particle-particle (short-range) force softening when the PP force is
        // on - then it is the real force resolution - then explicit softening keys, then the
        // PM smoothing length, then 0.75 PM cells.
        let pp_on = raw.get("short_range").and_then(|s| s.get("enabled")).and_then(|e| e.as_bool()) != Some(false);
        let (softening, softening_source) = if let Some(s) = num(&raw, &["short_range", "eps"]).filter(|s| pp_on && *s > 0.0) {
            (s, "meta.json short_range.eps (PP force softening)".to_string())
        } else if let Some(s) = num(&raw, &["softening"]) {
            (s, "meta.json softening".to_string())
        } else if let Some(s) = num(&raw, &["force", "softening"]) {
            (s, "meta.json force.softening".to_string())
        } else if let Some(s) = num(&raw, &["pm", "r_s"]).or_else(|| num(&raw, &["pm", "smoothing_length"])) {
            (s, "meta.json pm.r_s".to_string())
        } else if let Some(s) = raw.get("pm").and_then(|p| p.get("smoothing")).and_then(|x| x.as_str()).and_then(parse_rs)
        {
            (s, "meta.json pm.smoothing r_s".to_string())
        } else {
            let mesh = num(&raw, &["mesh"]).unwrap_or(box_size);
            (0.75 * box_size / mesh, "default 0.75 PM cells".to_string())
        };
        let (vel, vel_source) = match vel_override {
            Some(v) => (v, "command line".to_string()),
            None => detect_velocity(&raw).ok_or(
                "meta.json does not say how snapshot velocities are defined; pass --vel peculiar|canonical|comoving",
            )?,
        };
        Ok(RunMeta {
            dir: dir.to_path_buf(),
            box_size,
            particles,
            snapshots,
            particle_mass,
            particle_mass_source,
            c_poisson,
            softening,
            softening_source,
            vel,
            vel_source,
            dark_matter: raw.get("dark_matter").cloned(),
            raw,
        })
    }

    /// Model name for reports (dark_matter.model or directory name).
    pub fn model(&self) -> String {
        self.dark_matter
            .as_ref()
            .and_then(|d| d.get("model"))
            .and_then(|m| m.as_str())
            .map(str::to_string)
            .unwrap_or_else(|| self.dir.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default())
    }
}

/// Extract `r_s = <number>` from a description such as "S(k) = exp(-(k r_s)^2), r_s = 0.73".
fn parse_rs(s: &str) -> Option<f64> {
    let i = s.find("r_s =")?;
    let rest = s[i + 5..].trim_start();
    let end = rest.find(|c: char| !(c.is_ascii_digit() || c == '.' || c == 'e' || c == '-')).unwrap_or(rest.len());
    rest[..end].parse().ok()
}

/// Physical constants of one snapshot needed by the finder.
#[derive(Clone, Copy, Debug)]
pub struct Cosmo {
    pub a: f64,
    pub box_size: f64,
    /// Total particle count (sets the mean density).
    pub n_total: u64,
    /// Comoving-area mass per particle, L^2 / N (rho_bar = 1).
    pub m_area: f64,
    /// Output mass per particle.
    pub particle_mass: f64,
    pub c_poisson: f64,
    pub softening: f64,
    /// Snapshot velocity column x this = peculiar velocity.
    pub vel_factor: f64,
}

impl Cosmo {
    pub fn new(meta: &RunMeta, a: f64) -> Cosmo {
        Cosmo {
            a,
            box_size: meta.box_size,
            n_total: meta.particles,
            m_area: meta.box_size * meta.box_size / meta.particles as f64,
            particle_mass: meta.particle_mass,
            c_poisson: meta.c_poisson,
            softening: meta.softening,
            vel_factor: meta.vel.factor(a),
        }
    }

    /// `C m / (2 pi a)`: physical potential per particle per unit of `ln r`.
    pub fn g_log(&self) -> f64 {
        self.c_poisson * self.m_area / (2.0 * std::f64::consts::PI * self.a)
    }

    /// Physical circular velocity^2 at comoving radius `r` enclosing `n` particles.
    pub fn vc2(&self, n: f64, r: f64) -> f64 {
        self.c_poisson * (n * self.m_area - std::f64::consts::PI * r * r) / (2.0 * std::f64::consts::PI * self.a)
    }
}
