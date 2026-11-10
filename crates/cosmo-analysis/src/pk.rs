//! Particle power spectrum: CIC assignment, parallel 2D FFT, CIC deconvolution,
//! shot-noise subtraction and log-binning.
//!
//! Convention (matches the simulation contract):
//!   delta_k = (1/N^2) sum_x delta(x) exp(-i k.x),   P(k) = L^2 <|delta_k|^2>.
//! Then sigma^2 = sum_k |delta_k|^2 = int d^2k/(2pi)^2 P(k) = int P(k) k dk/(2pi),
//! so the 2D dimensionless power per ln k is Delta^2(k) = k^2 P(k) / (2 pi).

use rayon::prelude::*;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex64;
use std::f64::consts::PI;

/// Cloud-in-cell particle counts on an n x n periodic grid (grid point i at x = i L/n).
/// Uses one private f32 grid per worker, then a parallel row-wise reduction.
pub fn cic_counts(pos: &[[f32; 2]], box_size: f64, n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n * n];
    if pos.is_empty() {
        return out;
    }
    let threads = rayon::current_num_threads().max(1);
    // cap private-grid memory at ~2 GB
    let max_grids = ((2usize << 30) / (n * n * 4)).max(1);
    let nchunks = threads.min(max_grids);
    let chunk = pos.len().div_ceil(nchunks);
    let inv = n as f64 / box_size;
    let ni = n as i64;
    let grids: Vec<Vec<f32>> = pos
        .par_chunks(chunk)
        .map(|part| {
            let mut g = vec![0.0f32; n * n];
            for p in part {
                let ux = p[0] as f64 * inv;
                let uy = p[1] as f64 * inv;
                let fx = ux.floor();
                let fy = uy.floor();
                let dx = (ux - fx) as f32;
                let dy = (uy - fy) as f32;
                let ix = (fx as i64).rem_euclid(ni) as usize;
                let iy = (fy as i64).rem_euclid(ni) as usize;
                let ix1 = if ix + 1 == n { 0 } else { ix + 1 };
                let iy1 = if iy + 1 == n { 0 } else { iy + 1 };
                let (r0, r1) = (iy * n, iy1 * n);
                g[r0 + ix] += (1.0 - dx) * (1.0 - dy);
                g[r0 + ix1] += dx * (1.0 - dy);
                g[r1 + ix] += (1.0 - dx) * dy;
                g[r1 + ix1] += dx * dy;
            }
            g
        })
        .collect();
    out.par_chunks_mut(n).enumerate().for_each(|(row, o)| {
        for g in &grids {
            for (a, b) in o.iter_mut().zip(&g[row * n..(row + 1) * n]) {
                *a += *b as f64;
            }
        }
    });
    out
}

/// Blocked parallel out-of-place transpose of an n x n matrix.
fn transpose(src: &[Complex64], dst: &mut [Complex64], n: usize) {
    const B: usize = 32;
    dst.par_chunks_mut(n * B).enumerate().for_each(|(bi, chunk)| {
        let r0 = bi * B;
        let rows = chunk.len() / n;
        for c0 in (0..n).step_by(B) {
            for c in c0..(c0 + B).min(n) {
                let s = &src[c * n + r0..c * n + r0 + rows];
                for (r, v) in s.iter().enumerate() {
                    chunk[r * n + c] = *v;
                }
            }
        }
    });
}

/// In-place unnormalised 2D FFT (row-major, standard layout), rows/columns in parallel.
/// forward: X_k = sum_x x e^{-ik.x};  inverse: x = sum_k X_k e^{+ik.x}.
pub fn fft2d(data: &mut [Complex64], n: usize, inverse: bool) {
    assert_eq!(data.len(), n * n);
    let mut planner = FftPlanner::<f64>::new();
    let fft = if inverse { planner.plan_fft_inverse(n) } else { planner.plan_fft_forward(n) };
    let scratch_len = fft.get_inplace_scratch_len();
    let do_rows = |buf: &mut [Complex64]| {
        buf.par_chunks_mut(n).for_each_init(
            || vec![Complex64::new(0.0, 0.0); scratch_len],
            |s, row| fft.process_with_scratch(row, s),
        );
    };
    do_rows(data);
    let mut tmp = vec![Complex64::new(0.0, 0.0); n * n];
    transpose(data, &mut tmp, n);
    do_rows(&mut tmp);
    transpose(&tmp, data, n);
}

/// Signed integer frequency of FFT index i.
#[inline]
pub fn freq(i: usize, n: usize) -> i64 {
    if i <= n / 2 { i as i64 } else { i as i64 - n as i64 }
}

#[inline]
fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-8 { 1.0 } else { x.sin() / x }
}

pub fn log_edges(kmin: f64, kmax: f64, nbins: usize) -> Vec<f64> {
    let (l0, l1) = (kmin.ln(), kmax.ln());
    (0..=nbins).map(|i| (l0 + (l1 - l0) * i as f64 / nbins as f64).exp()).collect()
}

/// Bin edges reconstructed from a sorted list of (log-spaced) bin centres:
/// geometric midpoints between neighbours, extrapolated at both ends.
pub fn edges_from_centers(ks: &[f64]) -> Vec<f64> {
    let m = ks.len();
    if m == 0 {
        return vec![];
    }
    if m == 1 {
        return vec![ks[0] / 1.1, ks[0] * 1.1];
    }
    let mut e = Vec::with_capacity(m + 1);
    e.push(ks[0] / (ks[1] / ks[0]).sqrt());
    for i in 0..m - 1 {
        e.push((ks[i] * ks[i + 1]).sqrt());
    }
    e.push(ks[m - 1] * (ks[m - 1] / ks[m - 2]).sqrt());
    e
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PkBin {
    /// mode-weighted mean |k| in the bin
    pub k: f64,
    /// final estimate (deconvolved, shot noise subtracted if requested)
    pub p: f64,
    /// deconvolved, before shot-noise subtraction
    pub p_raw: f64,
    /// shot-noise level that was (or would be) subtracted, mode averaged
    pub p_shot: f64,
    /// mode-averaged linear theory
    pub p_lin: f64,
    /// number of grid modes in the bin (full k-plane: k and -k both counted)
    pub n_modes: u64,
}

impl PkBin {
    /// 2D dimensionless power Delta^2 = k^2 P / (2 pi).
    pub fn delta2(&self) -> f64 {
        self.k * self.k * self.p / (2.0 * PI)
    }
    /// Gaussian sampling error on P: P / sqrt(independent modes) = P sqrt(2 / n_modes).
    pub fn rel_err(&self) -> f64 {
        (2.0 / self.n_modes.max(1) as f64).sqrt()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PkOptions {
    pub subtract_shot: bool,
    pub deconvolve: bool,
}

impl Default for PkOptions {
    fn default() -> Self {
        PkOptions { subtract_shot: true, deconvolve: true }
    }
}

/// Bin |F_k|^2-derived quantities of a k-space grid. `f(kx, ky, v)` returns
/// (p, p_raw, p_shot, p_lin) for one mode or None to skip it.
pub fn bin_modes<F>(fk: &[Complex64], n: usize, box_size: f64, edges: &[f64], f: F) -> Vec<PkBin>
where
    F: Fn(f64, f64, Complex64) -> Option<(f64, f64, f64, f64)> + Sync,
{
    let nb = edges.len().saturating_sub(1);
    if nb == 0 {
        return vec![];
    }
    let kf = 2.0 * PI / box_size;
    let (emin, emax) = (edges[0], edges[nb]);
    let acc = fk
        .par_chunks(n)
        .enumerate()
        .fold(
            || vec![[0.0f64; 6]; nb],
            |mut acc, (iy, row)| {
                let ky = kf * freq(iy, n) as f64;
                for (ix, v) in row.iter().enumerate() {
                    if ix == 0 && iy == 0 {
                        continue;
                    }
                    let kx = kf * freq(ix, n) as f64;
                    let k = (kx * kx + ky * ky).sqrt();
                    if k < emin || k >= emax {
                        continue;
                    }
                    let b = edges.partition_point(|&e| e <= k) - 1;
                    if let Some((p, praw, ps, pl)) = f(kx, ky, *v) {
                        let a = &mut acc[b];
                        a[0] += k;
                        a[1] += p;
                        a[2] += praw;
                        a[3] += ps;
                        a[4] += pl;
                        a[5] += 1.0;
                    }
                }
                acc
            },
        )
        .reduce(
            || vec![[0.0f64; 6]; nb],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(&b) {
                    for j in 0..6 {
                        x[j] += y[j];
                    }
                }
                a
            },
        );
    acc.iter()
        .filter(|a| a[5] > 0.0)
        .map(|a| {
            let c = a[5];
            PkBin {
                k: a[0] / c,
                p: a[1] / c,
                p_raw: a[2] / c,
                p_shot: a[3] / c,
                p_lin: a[4] / c,
                n_modes: c as u64,
            }
        })
        .collect()
}

/// Power spectrum from CIC counts (as returned by [`cic_counts`]).
/// `p_lin(k)` is averaged over the same modes for a like-for-like comparison.
pub fn measure_pk(
    counts: &[f64],
    n: usize,
    box_size: f64,
    np: usize,
    edges: &[f64],
    p_lin: &(dyn Fn(f64) -> f64 + Sync),
    opt: PkOptions,
) -> Vec<PkBin> {
    let mean = np as f64 / (n * n) as f64;
    let mut grid: Vec<Complex64> =
        counts.par_iter().map(|&c| Complex64::new(c / mean - 1.0, 0.0)).collect();
    fft2d(&mut grid, n, false);
    let norm = 1.0 / (n * n) as f64;
    let l2 = box_size * box_size;
    let shot0 = l2 / np as f64;
    let h = 0.5 * box_size / n as f64; // Delta/2
    bin_modes(&grid, n, box_size, edges, |kx, ky, v| {
        let dk = v * norm;
        let p_meas = l2 * dk.norm_sqr();
        let (sx, sy) = ((kx * h).sin(), (ky * h).sin());
        // aliased CIC shot-noise shape (Jing 2005): prod_i [1 - (2/3) sin^2(k_i Delta/2)]
        let c = (1.0 - 2.0 / 3.0 * sx * sx) * (1.0 - 2.0 / 3.0 * sy * sy);
        let (praw, shot) = if opt.deconvolve {
            let w2 = (sinc(kx * h) * sinc(ky * h)).powi(4);
            (p_meas / w2, shot0 * c / w2)
        } else {
            (p_meas, shot0 * c)
        };
        let p = if opt.subtract_shot { praw - shot } else { praw };
        let k = (kx * kx + ky * ky).sqrt();
        Some((p, praw, shot, p_lin(k)))
    })
}

/// First up-crossing of Delta^2 = 1 (log-log interpolated). Err describes why none exists.
pub fn k_nonlinear(ks: &[f64], d2: &[f64]) -> Result<f64, &'static str> {
    if ks.is_empty() {
        return Err("no bins");
    }
    if d2[0] >= 1.0 {
        return Err("Delta^2 >= 1 already in the lowest-k bin");
    }
    for i in 1..ks.len() {
        if d2[i] >= 1.0 {
            let (d0, d1) = (d2[i - 1].max(1e-300), d2[i]);
            let t = (0.0 - d0.ln()) / (d1.ln() - d0.ln());
            return Ok((ks[i - 1].ln() + t * (ks[i] / ks[i - 1]).ln()).exp());
        }
    }
    Err("Delta^2 never reaches 1 in the measured range")
}

/// k_nl of a smooth function P(k) on [kmin, kmax], sampled finely in ln k.
pub fn k_nonlinear_fn(p: impl Fn(f64) -> f64, kmin: f64, kmax: f64) -> Result<f64, &'static str> {
    let ks = log_edges(kmin, kmax, 4000);
    let d2: Vec<f64> = ks.iter().map(|&k| k * k * p(k) / (2.0 * PI)).collect();
    k_nonlinear(&ks, &d2)
}
