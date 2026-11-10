//! Collision detection (uniform grid broad phase) and merge resolution.
//!
//! Broad phase: a dense uniform grid covers the whole world (bodies are always kept inside
//! ±WORLD_HALF). Every body is inserted into each cell its bounding box touches, so large
//! bodies (e.g. the galactic core) work with a small cell size. Two overlapping circles
//! always share at least one cell. The grid is built sequentially with a counting sort
//! (O(N + cells), microseconds for a few thousand bodies); the pair queries run in parallel.
//!
//! Merge resolution is sequential and deterministic: overlapping pairs are sorted and
//! deduplicated, joined with union-find into connected groups (so one body touching several
//! others merges them all at once), and each group collapses into its lowest-index member.
//! Mass and momentum are conserved exactly (up to f32 rounding of the final values).

use crate::{Body, WORLD_HALF, radius_for_mass};
use rayon::prelude::*;

const CELL: f32 = 16.0;
const DIM: usize = ((2.0 * WORLD_HALF / CELL) as usize) + 1;

#[derive(Default)]
pub(crate) struct Grid {
    /// Prefix-summed start index into `items` for each cell (len DIM*DIM + 1).
    start: Vec<u32>,
    items: Vec<u32>,
    /// Inclusive cell range per body: [x0, x1, y0, y1].
    ranges: Vec<[u32; 4]>,
    cursor: Vec<u32>,
}

#[inline]
fn cell_of(v: f32) -> u32 {
    (((v + WORLD_HALF) / CELL).floor().max(0.0) as usize).min(DIM - 1) as u32
}

impl Grid {
    /// Returns all overlapping pairs (i < j), sorted and unique.
    pub(crate) fn find_pairs(&mut self, bodies: &[Body]) -> Vec<(u32, u32)> {
        let cells = DIM * DIM;
        self.ranges.clear();
        self.ranges.extend(bodies.iter().map(|b| {
            let r = b.radius;
            [
                cell_of(b.pos[0] - r),
                cell_of(b.pos[0] + r),
                cell_of(b.pos[1] - r),
                cell_of(b.pos[1] + r),
            ]
        }));

        // Counting sort of (body, cell) entries.
        self.start.clear();
        self.start.resize(cells + 1, 0);
        for rg in &self.ranges {
            for cy in rg[2]..=rg[3] {
                for cx in rg[0]..=rg[1] {
                    self.start[cy as usize * DIM + cx as usize + 1] += 1;
                }
            }
        }
        for c in 0..cells {
            self.start[c + 1] += self.start[c];
        }
        self.cursor.clear();
        self.cursor.extend_from_slice(&self.start[..cells]);
        self.items.clear();
        self.items.resize(self.start[cells] as usize, 0);
        for (i, rg) in self.ranges.iter().enumerate() {
            for cy in rg[2]..=rg[3] {
                for cx in rg[0]..=rg[1] {
                    let c = cy as usize * DIM + cx as usize;
                    self.items[self.cursor[c] as usize] = i as u32;
                    self.cursor[c] += 1;
                }
            }
        }

        let (start, items, ranges) = (&self.start, &self.items, &self.ranges);
        let mut pairs: Vec<(u32, u32)> = (0..bodies.len())
            .into_par_iter()
            .with_min_len(64)
            .flat_map_iter(|i| {
                let bi = &bodies[i];
                let rg = ranges[i];
                let mut found = Vec::new();
                for cy in rg[2]..=rg[3] {
                    for cx in rg[0]..=rg[1] {
                        let c = cy as usize * DIM + cx as usize;
                        for &j in &items[start[c] as usize..start[c + 1] as usize] {
                            if j as usize <= i {
                                continue;
                            }
                            let bj = &bodies[j as usize];
                            let dx = bj.pos[0] - bi.pos[0];
                            let dy = bj.pos[1] - bi.pos[1];
                            let rs = bi.radius + bj.radius;
                            if dx * dx + dy * dy < rs * rs {
                                found.push((i as u32, j));
                            }
                        }
                    }
                }
                found
            })
            .collect();
        // A pair sharing several cells is reported once per shared cell.
        pairs.sort_unstable();
        pairs.dedup();
        pairs
    }

    /// Like `find_pairs`, but only pairs where at least one body has `mass >= min_mass`
    /// ("heavy"); light bodies pass through each other. Only heavy bodies are binned, and
    /// every body queries the cells its bounding box touches, so the cost is O(N) even when
    /// thousands of light bodies crowd one region. `min_mass <= 0` = `find_pairs`.
    pub(crate) fn find_pairs_heavy(&mut self, bodies: &[Body], min_mass: f32) -> Vec<(u32, u32)> {
        if min_mass <= 0.0 {
            return self.find_pairs(bodies);
        }
        let heavy: Vec<u32> = (0..bodies.len() as u32)
            .filter(|&i| bodies[i as usize].mass >= min_mass)
            .collect();
        if heavy.is_empty() {
            return Vec::new();
        }
        let range = |b: &Body| {
            let r = b.radius;
            [
                cell_of(b.pos[0] - r),
                cell_of(b.pos[0] + r),
                cell_of(b.pos[1] - r),
                cell_of(b.pos[1] + r),
            ]
        };
        let cells = DIM * DIM;
        self.start.clear();
        self.start.resize(cells + 1, 0);
        self.ranges.clear();
        self.ranges.extend(heavy.iter().map(|&i| range(&bodies[i as usize])));
        for rg in &self.ranges {
            for cy in rg[2]..=rg[3] {
                for cx in rg[0]..=rg[1] {
                    self.start[cy as usize * DIM + cx as usize + 1] += 1;
                }
            }
        }
        for c in 0..cells {
            self.start[c + 1] += self.start[c];
        }
        self.cursor.clear();
        self.cursor.extend_from_slice(&self.start[..cells]);
        self.items.clear();
        self.items.resize(self.start[cells] as usize, 0);
        for (&i, rg) in heavy.iter().zip(&self.ranges) {
            for cy in rg[2]..=rg[3] {
                for cx in rg[0]..=rg[1] {
                    let c = cy as usize * DIM + cx as usize;
                    self.items[self.cursor[c] as usize] = i;
                    self.cursor[c] += 1;
                }
            }
        }

        let (start, items) = (&self.start, &self.items);
        let mut pairs: Vec<(u32, u32)> = (0..bodies.len())
            .into_par_iter()
            .with_min_len(256)
            .flat_map_iter(|i| {
                let bi = &bodies[i];
                let i_heavy = bi.mass >= min_mass;
                let rg = range(bi);
                let mut found = Vec::new();
                for cy in rg[2]..=rg[3] {
                    for cx in rg[0]..=rg[1] {
                        let c = cy as usize * DIM + cx as usize;
                        for &j in &items[start[c] as usize..start[c + 1] as usize] {
                            let ju = j as usize;
                            // Heavy-heavy pairs are found from both sides: keep i < j.
                            if ju == i || (i_heavy && ju < i) {
                                continue;
                            }
                            let bj = &bodies[ju];
                            let dx = bj.pos[0] - bi.pos[0];
                            let dy = bj.pos[1] - bi.pos[1];
                            let rs = bi.radius + bj.radius;
                            if dx * dx + dy * dy < rs * rs {
                                let i = i as u32;
                                found.push((i.min(j), i.max(j)));
                            }
                        }
                    }
                }
                found
            })
            .collect();
        pairs.sort_unstable();
        pairs.dedup();
        pairs
    }
}

fn find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        let p = parent[x as usize];
        parent[x as usize] = parent[p as usize]; // path halving
        x = p;
    }
    x
}

/// Merge every connected group of overlapping bodies into its lowest-index member.
/// `acc` (per-body gravitational acceleration) is kept in sync: the merged body gets the
/// mass-weighted acceleration, which is exact for the force from all other bodies.
/// Returns the number of bodies removed (= number of pairwise merges).
/// `bins` (block-timestep bin per body, larger = smaller step) is kept in sync when its
/// length matches: the merged body takes the finest bin of its group.
pub(crate) fn merge_pairs(
    bodies: &mut Vec<Body>,
    acc: &mut Vec<[f32; 2]>,
    bins: &mut Vec<u8>,
    pairs: &[(u32, u32)],
) -> u64 {
    if pairs.is_empty() {
        return 0;
    }
    let n = bodies.len();
    let mut parent: Vec<u32> = (0..n as u32).collect();
    for &(a, b) in pairs {
        let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
        if ra != rb {
            // Lowest index becomes the root => deterministic survivor.
            let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
            parent[hi as usize] = lo;
        }
    }

    // All bodies taking part in a merge, grouped by root (sorted => deterministic order).
    let mut members: Vec<(u32, u32)> = pairs
        .iter()
        .flat_map(|&(a, b)| [a, b])
        .map(|i| (find(&mut parent, i), i))
        .collect();
    members.sort_unstable();
    members.dedup();

    let mut dead = vec![false; n];
    let mut removed = 0u64;
    let track_bins = bins.len() == n;
    for group in members.chunk_by(|a, b| a.0 == b.0) {
        let root = group[0].0 as usize;
        if track_bins {
            bins[root] = group.iter().map(|&(_, i)| bins[i as usize]).max().unwrap_or(0);
        }
        // f64 accumulators: [mass, px, py, mx, my, max, may, hue_cos, hue_sin]
        let mut s = [0.0f64; 9];
        for &(_, i) in group {
            let b = &bodies[i as usize];
            let a = acc[i as usize];
            let m = b.mass as f64;
            let th = b.hue as f64 * std::f64::consts::TAU;
            s[0] += m;
            s[1] += m * b.vel[0] as f64;
            s[2] += m * b.vel[1] as f64;
            s[3] += m * b.pos[0] as f64;
            s[4] += m * b.pos[1] as f64;
            s[5] += m * a[0] as f64;
            s[6] += m * a[1] as f64;
            s[7] += m * th.cos();
            s[8] += m * th.sin();
            if i as usize != root {
                dead[i as usize] = true;
                removed += 1;
            }
        }
        let m = s[0];
        if m <= 0.0 {
            continue;
        }
        // Mass-weighted circular mean of hue (handles wrap-around at 0/1).
        let heaviest = group
            .iter()
            .map(|&(_, i)| &bodies[i as usize])
            .max_by(|a, b| a.mass.total_cmp(&b.mass))
            .map_or(0.0, |b| b.hue);
        let hue = if s[7].hypot(s[8]) > 1e-6 * m {
            (s[8].atan2(s[7]) / std::f64::consts::TAU).rem_euclid(1.0) as f32
        } else {
            heaviest
        };
        let b = &mut bodies[root];
        b.mass = m as f32;
        b.vel = [(s[1] / m) as f32, (s[2] / m) as f32];
        b.pos = [(s[3] / m) as f32, (s[4] / m) as f32];
        b.radius = radius_for_mass(b.mass);
        b.hue = hue;
        acc[root] = [(s[5] / m) as f32, (s[6] / m) as f32];
    }

    // Stable compaction keeps the remaining order (and thus results) deterministic.
    let mut w = 0;
    for r in 0..n {
        if !dead[r] {
            bodies[w] = bodies[r];
            acc[w] = acc[r];
            if track_bins {
                bins[w] = bins[r];
            }
            w += 1;
        }
    }
    bodies.truncate(w);
    acc.truncate(w);
    if track_bins {
        bins.truncate(w);
    }
    removed
}
