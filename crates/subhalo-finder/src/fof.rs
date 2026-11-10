//! Periodic 2D friends-of-friends with a cell-linked, lock-free parallel union-find.
//! (Copied from cosmo-analysis/src/fof.rs; the halo-property helper is not needed here.)
//!
//! Algorithm
//! 1. Cells of side c <= b/sqrt(2) (M = ceil(L sqrt2 / b) per side). Any two particles in the
//!    same cell are closer than b, so every cell is a complete sub-group: its particles are
//!    linked at initialisation (parent = first particle of the cell). This removes all
//!    O(n^2) work inside dense halo cores.
//! 2. Particles are sorted by (cell_y, cell_x) using packed u64 keys (cell << 32 | index), and
//!    positions are copied into that order for cache locality. Occupied cells are stored
//!    sparsely (at most Np of them), with a per-row index for neighbour look-ups.
//! 3. Each occupied cell visits a half stencil of neighbour cells whose minimum box separation
//!    is < b. If the two cells already share a root the pair is skipped, otherwise particle
//!    pairs are tested (minimum-image) until the first link, which merges the two cells.
//! 4. Union-find uses atomic parents: roots are always linked larger-index -> smaller-index with
//!    compare-and-swap, and `find` does path halving with CAS, so the parent of every element
//!    only ever decreases and the structure is safe under concurrent mutation.

use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering::*};

pub struct FofGroups {
    /// group g owns members[offsets[g]..offsets[g+1]] (original particle indices, sorted)
    pub offsets: Vec<usize>,
    pub members: Vec<u32>,
}

impl FofGroups {
    pub fn len(&self) -> usize {
        self.offsets.len() - 1
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn group(&self, g: usize) -> &[u32] {
        &self.members[self.offsets[g]..self.offsets[g + 1]]
    }
    pub fn size(&self, g: usize) -> usize {
        self.offsets[g + 1] - self.offsets[g]
    }
}

#[inline]
fn find(p: &[AtomicU32], mut x: u32) -> u32 {
    loop {
        let px = p[x as usize].load(Relaxed);
        if px == x {
            return x;
        }
        let gp = p[px as usize].load(Relaxed);
        if gp != px {
            let _ = p[x as usize].compare_exchange_weak(px, gp, Relaxed, Relaxed);
        }
        x = gp;
    }
}

#[inline]
fn union(p: &[AtomicU32], a: u32, b: u32) {
    let (mut a, mut b) = (a, b);
    loop {
        a = find(p, a);
        b = find(p, b);
        if a == b {
            return;
        }
        if a < b {
            std::mem::swap(&mut a, &mut b);
        }
        if p[a as usize].compare_exchange(a, b, AcqRel, Relaxed).is_ok() {
            return;
        }
    }
}

/// Run FoF with linking length `link` (same units as positions) and keep groups with at
/// least `min_members` particles, sorted by size (descending).
pub fn fof(pos: &[[f32; 2]], box_size: f64, link: f64, min_members: usize) -> Result<FofGroups, String> {
    let np = pos.len();
    if np >= u32::MAX as usize {
        return Err("too many particles for 32-bit indices".into());
    }
    if !(link > 0.0 && link < 0.5 * box_size) {
        return Err(format!("linking length {link} must be in (0, box/2)"));
    }
    let m = (box_size * std::f64::consts::SQRT_2 / link).ceil().max(1.0) as u64;
    if m > 65535 {
        return Err(format!("FoF cell grid {m}^2 exceeds 32-bit cell keys"));
    }
    let c = box_size / m as f64;
    let inv = m as f64 / box_size;
    let mi = m as i64;

    // 1-2. sort by cell
    let mut keys: Vec<u64> = pos
        .par_iter()
        .enumerate()
        .map(|(i, p)| {
            let cx = ((p[0] as f64 * inv).floor() as i64).rem_euclid(mi) as u64;
            let cy = ((p[1] as f64 * inv).floor() as i64).rem_euclid(mi) as u64;
            ((cy * m + cx) << 32) | i as u64
        })
        .collect();
    keys.par_sort_unstable();
    const LO: u64 = 0xffff_ffff;
    let spos: Vec<[f32; 2]> = keys.par_iter().map(|k| pos[(k & LO) as usize]).collect();
    let cell_start: Vec<u32> = (0..np)
        .into_par_iter()
        .filter(|&j| j == 0 || (keys[j] >> 32) != (keys[j - 1] >> 32))
        .map(|j| j as u32)
        .collect();
    let ncell = cell_start.len();
    let cell_key: Vec<u64> = cell_start.par_iter().map(|&s| keys[s as usize] >> 32).collect();
    let cell_end = |ci: usize| if ci + 1 < ncell { cell_start[ci + 1] as usize } else { np };
    let row_start: Vec<usize> = (0..=m).map(|r| cell_key.partition_point(|&k| k < r * m)).collect();

    let parent: Vec<AtomicU32> = (0..np).into_par_iter().map(|_| AtomicU32::new(0)).collect();
    cell_start.par_iter().enumerate().for_each(|(ci, &s)| {
        for pj in &parent[s as usize..cell_end(ci)] {
            pj.store(s, Relaxed);
        }
    });

    // 3. half stencil of neighbour cells that can contain a pair closer than b
    let reach = (link / c).ceil() as i64;
    let mut stencil = Vec::new();
    for dy in 0..=reach {
        for dx in -reach..=reach {
            if dy == 0 && dx <= 0 {
                continue;
            }
            let gx = (dx.abs() - 1).max(0) as f64 * c;
            let gy = (dy - 1).max(0) as f64 * c;
            if gx * gx + gy * gy < link * link {
                stencil.push((dx, dy));
            }
        }
    }
    let l2 = (link * link) as f32;
    let boxf = box_size as f32;
    let half = 0.5 * boxf;
    (0..ncell).into_par_iter().with_min_len(512).for_each(|ci| {
        let key = cell_key[ci];
        let (cy, cx) = ((key / m) as i64, (key % m) as i64);
        let (a0, a1) = (cell_start[ci] as usize, cell_end(ci));
        for &(dx, dy) in &stencil {
            let nx = (cx + dx).rem_euclid(mi) as u64;
            let ny = (cy + dy).rem_euclid(mi) as u64;
            let nkey = ny * m + nx;
            if nkey == key {
                continue;
            }
            let (rs, re) = (row_start[ny as usize], row_start[ny as usize + 1]);
            let Ok(off) = cell_key[rs..re].binary_search(&nkey) else { continue };
            let cj = rs + off;
            let (b0, b1) = (cell_start[cj] as usize, cell_end(cj));
            if find(&parent, a0 as u32) == find(&parent, b0 as u32) {
                continue;
            }
            'pairs: for (i, pa) in spos[a0..a1].iter().enumerate() {
                for (j, pb) in spos[b0..b1].iter().enumerate() {
                    let mut ddx = pa[0] - pb[0];
                    let mut ddy = pa[1] - pb[1];
                    if ddx > half {
                        ddx -= boxf
                    } else if ddx < -half {
                        ddx += boxf
                    }
                    if ddy > half {
                        ddy -= boxf
                    } else if ddy < -half {
                        ddy += boxf
                    }
                    if ddx * ddx + ddy * ddy < l2 {
                        union(&parent, (a0 + i) as u32, (b0 + j) as u32);
                        break 'pairs;
                    }
                }
            }
        }
    });
    drop(spos);
    drop(cell_start);
    drop(cell_key);

    // 4. flatten, count, select
    (0..np).into_par_iter().for_each(|j| {
        let r = find(&parent, j as u32);
        parent[j].store(r, Relaxed);
    });
    let counts: Vec<AtomicU32> = (0..np).into_par_iter().map(|_| AtomicU32::new(0)).collect();
    (0..np).into_par_iter().for_each(|j| {
        counts[parent[j].load(Relaxed) as usize].fetch_add(1, Relaxed);
    });
    let mut kept: Vec<(u32, u32)> = (0..np)
        .into_par_iter()
        .filter_map(|j| {
            let cnt = counts[j].load(Relaxed);
            (cnt as usize >= min_members.max(1)).then_some((cnt, j as u32))
        })
        .collect();
    // deterministic order: by size desc, then by smallest original index in the group (below)
    kept.par_sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    let ng = kept.len();
    // reuse `counts` as root -> group id map
    counts.par_iter().for_each(|c| c.store(u32::MAX, Relaxed));
    kept.par_iter().enumerate().for_each(|(g, &(_, r))| counts[r as usize].store(g as u32, Relaxed));
    let mut offsets = Vec::with_capacity(ng + 1);
    offsets.push(0usize);
    for &(cnt, _) in &kept {
        offsets.push(offsets.last().unwrap() + cnt as usize);
    }
    let total = *offsets.last().unwrap();
    let cursor: Vec<AtomicU32> = (0..ng).map(|_| AtomicU32::new(0)).collect();
    let members_at: Vec<AtomicU32> = (0..total).into_par_iter().map(|_| AtomicU32::new(0)).collect();
    (0..np).into_par_iter().for_each(|j| {
        let g = counts[parent[j].load(Relaxed) as usize].load(Relaxed);
        if g != u32::MAX {
            let slot = cursor[g as usize].fetch_add(1, Relaxed) as usize;
            members_at[offsets[g as usize] + slot].store((keys[j] & LO) as u32, Relaxed);
        }
    });
    let mut members: Vec<u32> = members_at.into_par_iter().map(|a| a.into_inner()).collect();
    {
        let mut slices: Vec<&mut [u32]> = Vec::with_capacity(ng);
        let mut rest = members.as_mut_slice();
        for g in 0..ng {
            let (h, t) = rest.split_at_mut(offsets[g + 1] - offsets[g]);
            slices.push(h);
            rest = t;
        }
        slices.par_iter_mut().for_each(|s| s.sort_unstable());
    }
    Ok(FofGroups { offsets, members })
}
