//! Log-density PNG with a magma-style colormap.

use rayon::prelude::*;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

/// Magma anchors (matplotlib magma sampled every 0.1).
const MAGMA: [[f32; 3]; 11] = [
    [0.0, 0.0, 4.0],
    [20.0, 14.0, 54.0],
    [59.0, 15.0, 112.0],
    [100.0, 26.0, 128.0],
    [140.0, 41.0, 129.0],
    [183.0, 55.0, 121.0],
    [222.0, 73.0, 104.0],
    [246.0, 110.0, 92.0],
    [254.0, 159.0, 109.0],
    [254.0, 206.0, 145.0],
    [252.0, 253.0, 191.0],
];

pub fn magma(t: f32) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0) * 10.0;
    let i = (t.floor() as usize).min(9);
    let f = t - i as f32;
    let (a, b) = (MAGMA[i], MAGMA[i + 1]);
    [
        (a[0] + f * (b[0] - a[0])).round() as u8,
        (a[1] + f * (b[1] - a[1])).round() as u8,
        (a[2] + f * (b[2] - a[2])).round() as u8,
    ]
}

/// Write log10(1+delta) of CIC counts on an n x n grid (y axis up).
/// Colour range: 0.5th to 99.95th percentile; empty pixels floored at 0.1 particle.
pub fn write_density_png(path: &Path, counts: &[f64], n: usize, np: usize) -> Result<(), String> {
    let mean = np as f64 / (n * n) as f64;
    let v = log_density(counts, mean);
    let (lo, hi) = auto_range(&v);
    write_rgb_png(path, &v, n, lo, hi)
}

/// log10(rho / rho_bar) per pixel, floored at 0.1 particle per pixel. `mean` = mean count per pixel.
pub fn log_density(counts: &[f64], mean: f64) -> Vec<f32> {
    let floor = 0.1 / mean;
    counts.par_iter().map(|&c| ((c / mean).max(floor)).log10() as f32).collect()
}

/// Colour range: 0.5th to 99.95th percentile.
pub fn auto_range(v: &[f32]) -> (f32, f32) {
    let stride = (v.len() / 2_000_000).max(1);
    let mut sample: Vec<f32> = v.iter().step_by(stride).copied().collect();
    sample.par_sort_unstable_by(|a, b| a.total_cmp(b));
    let pct = |q: f64| sample[((sample.len() - 1) as f64 * q) as usize];
    let lo = pct(0.005);
    let mut hi = pct(0.9995);
    if hi - lo < 1e-6 {
        hi = lo + 1.0;
    }
    (lo, hi)
}

/// Write an n x n log-density field with a fixed colour range (row 0 = top = max y) as an
/// 8-bit indexed PNG with a 256-entry magma palette (about 3x smaller than RGB).
pub fn write_rgb_png(path: &Path, v: &[f32], n: usize, lo: f32, hi: f32) -> Result<(), String> {
    let mut buf = vec![0u8; n * n];
    buf.par_chunks_mut(n).enumerate().for_each(|(r, row)| {
        let iy = n - 1 - r;
        for (ix, px) in row.iter_mut().enumerate() {
            let t = ((v[iy * n + ix] - lo) / (hi - lo)).clamp(0.0, 1.0);
            *px = (t * 255.0).round() as u8;
        }
    });
    let palette: Vec<u8> = (0..256).flat_map(|i| magma(i as f32 / 255.0)).collect();
    let f = File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut enc = png::Encoder::new(BufWriter::new(f), n as u32, n as u32);
    enc.set_color(png::ColorType::Indexed);
    enc.set_depth(png::BitDepth::Eight);
    enc.set_palette(palette);
    enc.set_compression(png::Compression::High);
    let mut w = enc.write_header().map_err(|e| e.to_string())?;
    w.write_image_data(&buf).map_err(|e| e.to_string())?;
    w.finish().map_err(|e| e.to_string())
}