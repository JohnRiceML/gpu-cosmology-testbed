//! Run metadata (`meta.json`) and the in-game GPU power spectrum (`pk.csv`).
//!
//! Parsing is deliberately tolerant: unknown keys are ignored, numbers may be
//! given as JSON numbers or numeric strings, and a few common aliases are
//! accepted for every key we rely on.

use serde_json::Value;
use std::path::Path;

/// Initial power spectrum at `a_init`: P0(k) = A k^n exp(-(k R_c)^2).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spectrum {
    pub amp: f64,
    pub n: f64,
    pub r_c: f64,
}

impl Spectrum {
    pub fn p0(&self, k: f64) -> f64 {
        if k <= 0.0 {
            return 0.0;
        }
        self.amp * k.powf(self.n) * (-(k * self.r_c).powi(2)).exp()
    }
}

#[derive(Clone, Debug)]
pub struct Meta {
    pub box_size: f64,
    pub mesh: Option<usize>,
    pub particles: Option<u64>,
    pub lattice: Option<usize>,
    pub seed: Option<u64>,
    pub a_init: f64,
    pub a_final: Option<f64>,
    pub spectrum: Spectrum,
    pub spectrum_shape: String,
    pub growth: String,
    pub pk_convention: String,
    pub poisson_coefficient: Option<f64>,
    pub raw: Value,
}

fn num(v: &Value, keys: &[&str]) -> Option<f64> {
    for k in keys {
        if let Some(x) = v.get(*k) {
            if let Some(f) = x.as_f64() {
                return Some(f);
            }
            if let Some(s) = x.as_str()
                && let Ok(f) = s.trim().parse::<f64>()
            {
                return Some(f);
            }
        }
    }
    None
}

fn text(v: &Value, keys: &[&str]) -> Option<String> {
    for k in keys {
        if let Some(x) = v.get(*k) {
            return Some(match x {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            });
        }
    }
    None
}

impl Meta {
    pub fn from_value(v: Value) -> Result<Meta, String> {
        let box_size = num(&v, &["box", "box_size", "boxsize", "L"])
            .ok_or("meta.json: missing 'box'")?;
        let spec_v = v.get("spectrum").cloned().unwrap_or(Value::Null);
        let amp = num(&spec_v, &["A", "amp", "amplitude"])
            .or_else(|| num(&v, &["A", "amplitude"]))
            .ok_or("meta.json: missing spectrum.A")?;
        let n = num(&spec_v, &["n", "index", "n_s"]).unwrap_or(0.0);
        let r_c = num(&spec_v, &["R_c", "Rc", "r_c", "rc"]).unwrap_or(0.0);
        let a_init = num(&v, &["a_init", "a_start", "ainit", "a_i"])
            .ok_or("meta.json: missing 'a_init'")?;
        Ok(Meta {
            box_size,
            mesh: num(&v, &["mesh", "mesh_size", "grid"]).map(|x| x as usize),
            particles: num(&v, &["particles", "n_particles", "np"]).map(|x| x as u64),
            lattice: num(&v, &["lattice", "lattice_side"]).map(|x| x as usize),
            seed: num(&v, &["seed"]).map(|x| x as u64),
            a_init,
            a_final: num(&v, &["a_final", "a_end"]),
            spectrum: Spectrum { amp, n, r_c },
            spectrum_shape: text(&spec_v, &["shape"]).unwrap_or_else(|| "powerlaw_gauss".into()),
            growth: text(&v, &["growth"]).unwrap_or_else(|| "D=a".into()),
            pk_convention: text(&v, &["pk_convention"]).unwrap_or_default(),
            poisson_coefficient: num(&v, &["poisson_coefficient"]),
            raw: v,
        })
    }

    pub fn load(path: &Path) -> Result<Meta, String> {
        let s = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let v: Value =
            serde_json::from_str(s.trim_start_matches('\u{feff}')).map_err(|e| format!("{}: {e}", path.display()))?;
        Meta::from_value(v)
    }

    /// Linear growth with D = a, normalised so that D(a_init) = 1.
    pub fn growth(&self, a: f64) -> f64 {
        a / self.a_init
    }

    /// P_lin(k, a) = (a / a_init)^2 P0(k).
    pub fn p_lin(&self, k: f64, a: f64) -> f64 {
        self.growth(a).powi(2) * self.spectrum.p0(k)
    }
}

/// One row of the GPU `pk.csv`.
#[derive(Clone, Copy, Debug)]
pub struct PkRow {
    pub a: f64,
    pub k: f64,
    pub p_meas: f64,
    pub p_lin: f64,
    pub n_modes: f64,
}

pub fn read_pk_csv(path: &Path) -> Result<Vec<PkRow>, String> {
    let s = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut lines = s.lines().filter(|l| !l.trim().is_empty());
    let header = lines.next().ok_or("pk.csv is empty")?;
    let cols: Vec<String> = header.split(',').map(|c| c.trim().to_ascii_lowercase()).collect();
    let idx = |name: &str, fallback: usize| cols.iter().position(|c| c == name).unwrap_or(fallback);
    let (ia, ik, ipm, ipl, inm) = (
        idx("a", 0),
        idx("k", 1),
        idx("p_measured", 2),
        idx("p_linear", 3),
        idx("n_modes", 4),
    );
    let mut rows = Vec::new();
    for line in lines {
        let f: Vec<&str> = line.split(',').map(|x| x.trim()).collect();
        let get = |i: usize| f.get(i).and_then(|x| x.parse::<f64>().ok());
        if let (Some(a), Some(k), Some(pm)) = (get(ia), get(ik), get(ipm)) {
            rows.push(PkRow {
                a,
                k,
                p_meas: pm,
                p_lin: get(ipl).unwrap_or(f64::NAN),
                n_modes: get(inm).unwrap_or(f64::NAN),
            });
        }
    }
    Ok(rows)
}

/// Rows of the output time nearest to `a`, sorted by k.
pub fn rows_nearest_a(rows: &[PkRow], a: f64) -> Option<(f64, Vec<PkRow>)> {
    let best = rows
        .iter()
        .map(|r| r.a)
        .min_by(|x, y| (x - a).abs().total_cmp(&(y - a).abs()))?;
    let mut sel: Vec<PkRow> = rows
        .iter()
        .copied()
        .filter(|r| (r.a - best).abs() <= 1e-9 * best.abs().max(1.0))
        .filter(|r| r.k > 0.0 && r.k.is_finite())
        .collect();
    sel.sort_by(|x, y| x.k.total_cmp(&y.k));
    Some((best, sel))
}

/// Log-log interpolation of the GPU spectrum at k (linear fallback when a value is <= 0).
pub fn interp_pk(rows: &[PkRow], k: f64) -> f64 {
    if rows.is_empty() || k < rows[0].k * (1.0 - 1e-9) || k > rows[rows.len() - 1].k * (1.0 + 1e-9) {
        return f64::NAN;
    }
    let i = rows.partition_point(|r| r.k < k);
    if i == 0 {
        return rows[0].p_meas;
    }
    if i >= rows.len() {
        return rows[rows.len() - 1].p_meas;
    }
    let (r0, r1) = (rows[i - 1], rows[i]);
    if r0.p_meas > 0.0 && r1.p_meas > 0.0 {
        let t = (k / r0.k).ln() / (r1.k / r0.k).ln();
        (r0.p_meas.ln() + t * (r1.p_meas / r0.p_meas).ln()).exp()
    } else {
        let t = (k - r0.k) / (r1.k - r0.k);
        r0.p_meas + t * (r1.p_meas - r0.p_meas)
    }
}
