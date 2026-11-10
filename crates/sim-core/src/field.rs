//! Bilinear sampling of the external (GPU particle-mesh) acceleration field.

use crate::{ExternalField, WORLD_HALF};

/// Acceleration of the external field at `p` (already includes G). Cell centres sit at
/// `-WORLD_HALF + (i + 0.5) * cell`; outside the outermost centres the sample is clamped
/// to the edge (constant extrapolation), so points beyond the grid never read garbage.
#[inline]
pub(crate) fn sample(f: &ExternalField, p: [f32; 2]) -> [f32; 2] {
    let s = f.size;
    if s == 1 {
        return f.acc[0];
    }
    let inv_cell = s as f32 / (2.0 * WORLD_HALF);
    let max = (s - 1) as f32;
    // Continuous grid coordinate (0 at the first cell centre). NaN clamps to 0 below.
    let gx = ((p[0] + WORLD_HALF) * inv_cell - 0.5).clamp(0.0, max);
    let gy = ((p[1] + WORLD_HALF) * inv_cell - 0.5).clamp(0.0, max);
    let gx = if gx.is_nan() { 0.0 } else { gx };
    let gy = if gy.is_nan() { 0.0 } else { gy };
    let ix = (gx as usize).min(s - 2);
    let iy = (gy as usize).min(s - 2);
    let (fx, fy) = (gx - ix as f32, gy - iy as f32);
    let i00 = iy * s + ix;
    let (a00, a10, a01, a11) = (f.acc[i00], f.acc[i00 + 1], f.acc[i00 + s], f.acc[i00 + s + 1]);
    let mut out = [0.0f32; 2];
    for k in 0..2 {
        let bottom = a00[k] + (a10[k] - a00[k]) * fx;
        let top = a01[k] + (a11[k] - a01[k]) * fx;
        out[k] = bottom + (top - bottom) * fy;
    }
    out
}
