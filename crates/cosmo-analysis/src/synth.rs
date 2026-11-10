//! Synthetic 2D Zel'dovich runs in the simulation's file format (for validation).
//!
//! Gaussian field on the particle lattice (n x n, spacing s = L/n), contract convention:
//!   delta_k = W_k sqrt(P0(k)) / (L n)   with W_k the unnormalised FFT of unit white noise,
//! so <|delta_k|^2> = P0/L^2. Nyquist rows/columns and k = 0 are zeroed (keeps Psi real).
//! Zel'dovich: delta = -div Psi  =>  Psi_k = i k delta_k / k^2;  x = q + (a/a_init) Psi_init(q).
//! Velocities are written as dx/dln a = (a/a_init) Psi_init (code units).

use crate::meta::Spectrum;
use crate::pk::{bin_modes, fft2d, freq, log_edges};
use crate::snapshot::write_snapshot;
use rayon::prelude::*;
use rustfft::num_complex::Complex64;
use std::f64::consts::PI;
use std::path::Path;

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn uniform(state: &mut u64) -> f64 {
    ((splitmix(state) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

pub struct ZaField {
    pub n: usize,
    pub box_size: f64,
    /// realisation at a_init, contract convention, standard row-major [ky][kx] layout
    pub delta_k: Vec<Complex64>,
    /// displacement at a_init per lattice site (index iy*n + ix)
    pub psi: Vec<[f32; 2]>,
}

pub fn make_za(n: usize, box_size: f64, spec: &Spectrum, seed: u64) -> ZaField {
    let mut w: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); n * n];
    w.par_chunks_mut(n).enumerate().for_each(|(iy, row)| {
        let mut st = seed ^ (iy as u64).wrapping_mul(0xD1B5_4A32_D192_ED03) ^ 0x5851_F42D_4C95_7F2D;
        for pair in row.chunks_mut(2) {
            let (u1, u2) = (uniform(&mut st), uniform(&mut st));
            let r = (-2.0 * u1.ln()).sqrt();
            pair[0] = Complex64::new(r * (2.0 * PI * u2).cos(), 0.0);
            if pair.len() > 1 {
                pair[1] = Complex64::new(r * (2.0 * PI * u2).sin(), 0.0);
            }
        }
    });
    fft2d(&mut w, n, false);
    let kf = 2.0 * PI / box_size;
    let mut px = vec![Complex64::new(0.0, 0.0); n * n];
    let mut py = vec![Complex64::new(0.0, 0.0); n * n];
    w.par_chunks_mut(n)
        .zip(px.par_chunks_mut(n))
        .zip(py.par_chunks_mut(n))
        .enumerate()
        .for_each(|(iy, ((wr, xr), yr))| {
            let ky = kf * freq(iy, n) as f64;
            for ix in 0..n {
                let kx = kf * freq(ix, n) as f64;
                let k2 = kx * kx + ky * ky;
                if (ix == 0 && iy == 0) || ix == n / 2 || iy == n / 2 {
                    wr[ix] = Complex64::new(0.0, 0.0);
                    continue;
                }
                let d = wr[ix] * (spec.p0(k2.sqrt()).sqrt() / (box_size * n as f64));
                wr[ix] = d;
                let i_d = Complex64::new(-d.im, d.re); // i * delta_k
                xr[ix] = i_d * (kx / k2);
                yr[ix] = i_d * (ky / k2);
            }
        });
    fft2d(&mut px, n, true);
    fft2d(&mut py, n, true);
    let psi: Vec<[f32; 2]> = px.par_iter().zip(py.par_iter()).map(|(x, y)| [x.re as f32, y.re as f32]).collect();
    ZaField { n, box_size, delta_k: w, psi }
}

/// Lattice sites are q = (i + LATTICE_OFFSET) * L/n.
pub const LATTICE_OFFSET: f64 = 0.25;

/// Zel'dovich particles at growth factor g = a/a_init.
pub fn za_particles(f: &ZaField, g: f64) -> (Vec<[f32; 2]>, Vec<[f32; 2]>) {
    let n = f.n;
    let s = f.box_size / n as f64;
    let l = f.box_size;
    let pos = f
        .psi
        .par_iter()
        .enumerate()
        .map(|(i, p)| {
            let (ix, iy) = (i % n, i / n);
            let wrap = |v: f64| {
                let w = v.rem_euclid(l) as f32;
                if w >= l as f32 { 0.0 } else { w }
            };
            // the quarter-cell offset keeps the lattice off the nodes of any power-of-two
            // multiple measurement grid (a lattice sitting on CIC nodes rectifies sub-cell
            // displacements and biases P(k) by O(k dx) at early times)
            [
                wrap((ix as f64 + LATTICE_OFFSET) * s + g * p[0] as f64),
                wrap((iy as f64 + LATTICE_OFFSET) * s + g * p[1] as f64),
            ]
        })
        .collect();
    let vel = f.psi.par_iter().map(|p| [(g * p[0] as f64) as f32, (g * p[1] as f64) as f32]).collect();
    (pos, vel)
}

pub struct GenParams {
    pub box_size: f64,
    pub lattice: usize,
    pub mesh: usize,
    pub seed: u64,
    pub a_init: f64,
    pub a_list: Vec<f64>,
    pub spec: Spectrum,
}

/// Write meta.json, pk.csv (binned power of the linear realisation, standing in for the
/// GPU measurement) and one snapshot per a.
pub fn gen_test(dir: &Path, p: &GenParams) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let t0 = std::time::Instant::now();
    let f = make_za(p.lattice, p.box_size, &p.spec, p.seed);
    eprintln!("gen-test: Zel'dovich field {}^2 in {:.2}s", p.lattice, t0.elapsed().as_secs_f64());
    let a_final = p.a_list.iter().cloned().fold(p.a_init, f64::max);
    let meta = serde_json::json!({
        "box": p.box_size, "mesh": p.mesh, "particles": p.lattice * p.lattice, "lattice": p.lattice,
        "seed": p.seed, "a_init": p.a_init, "a_final": a_final,
        "spectrum": {"shape": "powerlaw_gauss", "A": p.spec.amp, "n": p.spec.n, "R_c": p.spec.r_c},
        "growth": "D=a",
        "pk_convention": "delta_k=(1/N^2) sum_x delta(x) exp(-ik.x); P(k)=L^2<|delta_k|^2>",
        "poisson_coefficient": 1.5,
        "generator": "cosmo-analysis gen-test (2D Zel'dovich, synthetic)",
    });
    std::fs::write(dir.join("meta.json"), serde_json::to_string_pretty(&meta).unwrap()).map_err(|e| e.to_string())?;

    // pk.csv: linear realisation, binned like a mesh of size `mesh`
    let kf = 2.0 * PI / p.box_size;
    let k_ny = PI * p.mesh.min(p.lattice) as f64 / p.box_size;
    let edges = log_edges(kf * 0.999, k_ny, 48);
    let l2 = p.box_size * p.box_size;
    let spec = p.spec;
    let bins = bin_modes(&f.delta_k, f.n, p.box_size, &edges, |kx, ky, v| {
        let k = (kx * kx + ky * ky).sqrt();
        let pr = l2 * v.norm_sqr();
        (pr > 0.0).then_some((pr, pr, 0.0, spec.p0(k)))
    });
    let mut csv = String::from("a,k,P_measured,P_linear,n_modes\n");
    for &a in &p.a_list {
        let g2 = (a / p.a_init).powi(2);
        for b in &bins {
            csv.push_str(&format!("{:.6},{:.8e},{:.8e},{:.8e},{}\n", a, b.k, g2 * b.p, g2 * b.p_lin, b.n_modes));
        }
    }
    std::fs::write(dir.join("pk.csv"), csv).map_err(|e| e.to_string())?;

    for &a in &p.a_list {
        let (pos, vel) = za_particles(&f, a / p.a_init);
        let path = dir.join(format!("snap_a{a:.3}.bin"));
        write_snapshot(&path, a as f32, p.box_size as f32, &pos, &vel).map_err(|e| e.to_string())?;
        eprintln!("gen-test: wrote {}", path.display());
    }
    Ok(())
}
