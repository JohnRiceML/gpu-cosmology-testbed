//! Barnes-Hut gravity: a quadtree rebuilt from scratch on every force evaluation.
//!
//! 1. **Sort**: each body gets a 32-bit Morton (Z-order) key from its position quantised
//!    on a 2^16 x 2^16 grid over the world square; `(key << 32 | index)` is sorted with
//!    rayon's parallel unstable sort (keys are unique, so the order is deterministic).
//!    Positions and masses are gathered into sorted structure-of-arrays buffers.
//! 2. **Build**: a quadtree over the sorted array. Every node owns a contiguous range of
//!    sorted bodies, so children are found with binary searches on the Morton bits; levels
//!    where all bodies fall into the same quadrant are skipped (path compression). A node
//!    with at most `LEAF` bodies (or at the maximum depth) is a leaf. Nodes are stored in
//!    depth-first pre-order with a `skip` index (next node after the subtree), so the
//!    traversal is a stackless loop over a flat array. Each node keeps its mass, centre of
//!    mass and the *tight* bounding box of its bodies. The build is sequential: it is
//!    O(N) after the sort and takes well under a millisecond at 50k bodies.
//! 3. **Walk** (parallel over leaves): each leaf is a group of nearby bodies that share
//!    one interaction list. A node is accepted as a point mass when the distance `d` from
//!    its centre of mass to the group's bounding box satisfies `d > s / THETA + delta`
//!    (`s` = node box size, `delta` = offset of the centre of mass from the box centre,
//!    Barnes 1994); opened leaves contribute their bodies directly. The list is then
//!    evaluated for every body in the group with the same vectorised Plummer-softened
//!    kernel as the direct sum (`gravity::row`).
//!
//! The result is deterministic (fixed tree, fixed summation order) and independent of the
//! number of threads.

use crate::gravity::{LANES, row, simd_level};
use crate::{Body, WORLD_HALF};
use rayon::prelude::*;

/// Opening angle. With the group criterion below, 0.5 gives ~0.6% norm-weighted RMS force
/// error and ~2% per-body RMS relative error on a uniform random set (see tests); 0.6 is
/// ~25% cheaper at 50k bodies but ~0.9% / 3.5%.
pub(crate) const THETA: f32 = 0.5;
/// Maximum bodies per leaf (= interaction-list group size).
const LEAF: usize = 16;
/// Morton bits per axis.
const BITS: u32 = 16;
/// Subtrees with at least this many bodies build their children in parallel.
const PAR_BUILD: usize = 8192;

#[derive(Clone, Copy, Default)]
struct Node {
    com: [f32; 2],
    mass: f32,
    /// Squared opening distance: accept as a point mass when the target group is farther.
    open2: f32,
    bmin: [f32; 2],
    bmax: [f32; 2],
    lo: u32,
    hi: u32,
    /// Index of the next node after this subtree (pre-order).
    skip: u32,
    leaf: bool,
}

#[derive(Default)]
pub(crate) struct Tree {
    keys: Vec<u64>,
    codes: Vec<u32>,
    sx: Vec<f32>,
    sy: Vec<f32>,
    sm: Vec<f32>,
    nodes: Vec<Node>,
    leaves: Vec<u32>,
    acc_sorted: Vec<[f32; 2]>,
    smask: Vec<bool>,
    /// Body indices in the last Morton order (seeds the next sort).
    order: Vec<u32>,
}

/// Spread the low 16 bits of `v` to the even bit positions.
#[inline]
fn spread(v: u32) -> u32 {
    let mut x = v & 0xFFFF;
    x = (x | (x << 8)) & 0x00FF_00FF;
    x = (x | (x << 4)) & 0x0F0F_0F0F;
    x = (x | (x << 2)) & 0x3333_3333;
    x = (x | (x << 1)) & 0x5555_5555;
    x
}

#[inline]
fn quantise(v: f32) -> u32 {
    // NaN -> 0 via the saturating float-to-int cast.
    ((v + WORLD_HALF) * (65536.0 / (2.0 * WORLD_HALF))).clamp(0.0, 65535.0) as u32
}

struct Builder<'a> {
    codes: &'a [u32],
    sx: &'a [f32],
    sy: &'a [f32],
    sm: &'a [f32],
    nodes: &'a mut Vec<Node>,
    leaves: &'a mut Vec<u32>,
}

impl Builder<'_> {
    fn build(&mut self, lo: usize, hi: usize, mut level: u32) -> usize {
        let codes = self.codes;
        // Path compression: skip levels where every body falls into the same quadrant.
        while level < BITS && hi - lo > LEAF {
            let shift = 2 * (BITS - 1 - level);
            if (codes[lo] >> shift) & 3 != (codes[hi - 1] >> shift) & 3 {
                break;
            }
            level += 1;
        }

        let idx = self.nodes.len();
        self.nodes.push(Node {
            lo: lo as u32,
            hi: hi as u32,
            ..Node::default()
        });

        let (mut m, mut mx, mut my) = (0.0f32, 0.0f32, 0.0f32);
        let mut bmin = [f32::INFINITY; 2];
        let mut bmax = [f32::NEG_INFINITY; 2];
        let leaf = hi - lo <= LEAF || level >= BITS;
        if leaf {
            for k in lo..hi {
                let (x, y, w) = (self.sx[k], self.sy[k], self.sm[k]);
                m += w;
                mx += w * x;
                my += w * y;
                bmin = [bmin[0].min(x), bmin[1].min(y)];
                bmax = [bmax[0].max(x), bmax[1].max(y)];
            }
            self.leaves.push(idx as u32);
        } else {
            let shift = 2 * (BITS - 1 - level);
            let mut ranges = [(0usize, 0usize); 4];
            let mut nr = 0;
            let mut s = lo;
            for c in 0..4u32 {
                let e = if c == 3 {
                    hi
                } else {
                    lo + codes[lo..hi].partition_point(|&k| (k >> shift) & 3 <= c)
                };
                if e > s {
                    ranges[nr] = (s, e);
                    nr += 1;
                }
                s = e;
            }
            let mut children = [0usize; 4];
            if hi - lo >= PAR_BUILD {
                // Big subtrees are built concurrently into their own buffers, then appended
                // in quadrant order with their indices shifted: the same pre-order layout
                // (and therefore the same results) as the sequential build.
                let ctx = (self.codes, self.sx, self.sy, self.sm);
                let subs: Vec<(Vec<Node>, Vec<u32>)> = ranges[..nr]
                    .par_iter()
                    .map(|&(s, e)| {
                        let (mut nodes, mut leaves) = (Vec::new(), Vec::new());
                        Builder {
                            codes: ctx.0,
                            sx: ctx.1,
                            sy: ctx.2,
                            sm: ctx.3,
                            nodes: &mut nodes,
                            leaves: &mut leaves,
                        }
                        .build(s, e, level + 1);
                        (nodes, leaves)
                    })
                    .collect();
                for (c, (nodes, leaves)) in subs.into_iter().enumerate() {
                    let off = self.nodes.len() as u32;
                    children[c] = off as usize;
                    self.nodes.extend(nodes.into_iter().map(|mut nd| {
                        nd.skip += off;
                        nd
                    }));
                    self.leaves.extend(leaves.into_iter().map(|l| l + off));
                }
            } else {
                for c in 0..nr {
                    children[c] = self.build(ranges[c].0, ranges[c].1, level + 1);
                }
            }
            for &child in &children[..nr] {
                let ch = self.nodes[child];
                m += ch.mass;
                mx += ch.mass * ch.com[0];
                my += ch.mass * ch.com[1];
                bmin = [bmin[0].min(ch.bmin[0]), bmin[1].min(ch.bmin[1])];
                bmax = [bmax[0].max(ch.bmax[0]), bmax[1].max(ch.bmax[1])];
            }
        }

        let centre = [0.5 * (bmin[0] + bmax[0]), 0.5 * (bmin[1] + bmax[1])];
        let com = if m > 0.0 { [mx / m, my / m] } else { centre };
        let size = (bmax[0] - bmin[0]).max(bmax[1] - bmin[1]);
        let delta = (com[0] - centre[0]).hypot(com[1] - centre[1]);
        let open = size / THETA + delta;
        let skip = self.nodes.len() as u32;
        let node = &mut self.nodes[idx];
        node.com = com;
        node.mass = m;
        node.open2 = open * open;
        node.bmin = bmin;
        node.bmax = bmax;
        node.leaf = leaf;
        node.skip = skip;
        idx
    }
}

impl Tree {
    /// Gravitational acceleration of every body into `acc`. Returns the number of
    /// pairwise (body-body or body-node) interactions evaluated, including SIMD padding.
    pub(crate) fn compute(&mut self, bodies: &[Body], acc: &mut Vec<[f32; 2]>) -> u64 {
        acc.clear();
        acc.resize(bodies.len(), [0.0; 2]);
        self.compute_masked(bodies, acc, None)
    }

    /// Like `compute`, but with `mask = Some(m)` only bodies with `m[i]` (the sinks) get
    /// their `acc[i]` written; all bodies are still sources. `acc.len()` must equal the
    /// body count. Leaves without sinks are skipped entirely.
    pub(crate) fn compute_masked(
        &mut self,
        bodies: &[Body],
        acc: &mut [[f32; 2]],
        mask: Option<&[bool]>,
    ) -> u64 {
        let n = bodies.len();
        assert_eq!(acc.len(), n);
        if n == 0 {
            return 0;
        }
        assert!(n < u32::MAX as usize);
        // 1. Morton sort. Bodies move little between evaluations, so when the body count is
        // unchanged the keys are generated in the previous sorted order and an adaptive
        // (run-detecting, merge-based) stable sort finishes the nearly sorted input quickly.
        // Keys are unique, so the result is the same as sorting from scratch.
        let key = |i: usize| {
            let b = &bodies[i];
            let code = spread(quantise(b.pos[0])) | (spread(quantise(b.pos[1])) << 1);
            ((code as u64) << 32) | i as u64
        };
        self.keys.clear();
        if self.order.len() == n {
            self.keys.par_extend(self.order.par_iter().map(|&i| key(i as usize)));
            self.keys.par_sort();
        } else {
            self.keys.par_extend((0..n).into_par_iter().map(key));
            self.keys.par_sort_unstable();
        }
        // Gather everything in sorted order in one parallel pass.
        let keys = &self.keys;
        for v in [&mut self.codes, &mut self.order] {
            v.resize(n, 0);
        }
        for v in [&mut self.sx, &mut self.sy, &mut self.sm] {
            v.resize(n, 0.0);
        }
        keys.par_iter()
            .zip(self.codes.par_iter_mut())
            .zip(self.order.par_iter_mut())
            .zip(self.sx.par_iter_mut())
            .zip(self.sy.par_iter_mut())
            .zip(self.sm.par_iter_mut())
            .with_min_len(2048)
            .for_each(|(((((&k, c), o), x), y), m)| {
                let b = &bodies[k as u32 as usize];
                *c = (k >> 32) as u32;
                *o = k as u32;
                *x = b.pos[0];
                *y = b.pos[1];
                *m = b.mass;
            });

        // 2. Build.
        self.nodes.clear();
        self.leaves.clear();
        Builder {
            codes: &self.codes,
            sx: &self.sx,
            sy: &self.sy,
            sm: &self.sm,
            nodes: &mut self.nodes,
            leaves: &mut self.leaves,
        }
        .build(0, n, 0);

        // 3. Walk + evaluate, one task per leaf. Leaves are in pre-order, so they cover the
        // sorted array contiguously and in order: split the output accordingly.
        self.acc_sorted.clear();
        self.acc_sorted.resize(n, [0.0; 2]);
        let nodes = &self.nodes[..];
        let mut parts: Vec<(&Node, &mut [[f32; 2]])> = Vec::with_capacity(self.leaves.len());
        let mut rest = &mut self.acc_sorted[..];
        for &l in &self.leaves {
            let nd = &nodes[l as usize];
            let (a, b) = rest.split_at_mut((nd.hi - nd.lo) as usize);
            parts.push((nd, a));
            rest = b;
        }
        debug_assert!(rest.is_empty());

        // Sink mask in sorted order (block timesteps: only active bodies need forces).
        self.smask.clear();
        if let Some(mask) = mask {
            self.smask.par_extend(keys.par_iter().map(|&k| mask[k as u32 as usize]));
        }
        let smask = &self.smask[..];
        let masked = mask.is_some();

        let (sx, sy, sm) = (&self.sx[..], &self.sy[..], &self.sm[..]);
        let simd = simd_level();
        let interactions: u64 = parts
            .into_par_iter()
            .map_init(
                || (Vec::new(), Vec::new(), Vec::new()),
                |(lx, ly, lm), (g, out)| {
                    let (lo, hi) = (g.lo as usize, g.hi as usize);
                    let sinks = if masked {
                        smask[lo..hi].iter().filter(|&&a| a).count()
                    } else {
                        hi - lo
                    };
                    if sinks == 0 {
                        return 0;
                    }
                    lx.clear();
                    ly.clear();
                    lm.clear();
                    let mut i = 0usize;
                    while i < nodes.len() {
                        let nd = &nodes[i];
                        let dx = (g.bmin[0] - nd.com[0]).max(nd.com[0] - g.bmax[0]).max(0.0);
                        let dy = (g.bmin[1] - nd.com[1]).max(nd.com[1] - g.bmax[1]).max(0.0);
                        if dx * dx + dy * dy > nd.open2 {
                            lx.push(nd.com[0]);
                            ly.push(nd.com[1]);
                            lm.push(nd.mass);
                            i = nd.skip as usize;
                        } else if nd.leaf {
                            let r = nd.lo as usize..nd.hi as usize;
                            lx.extend_from_slice(&sx[r.clone()]);
                            ly.extend_from_slice(&sy[r.clone()]);
                            lm.extend_from_slice(&sm[r]);
                            i = nd.skip as usize;
                        } else {
                            i += 1;
                        }
                    }
                    let padded = lx.len().div_ceil(LANES) * LANES;
                    lx.resize(padded, 0.0);
                    ly.resize(padded, 0.0);
                    lm.resize(padded, 0.0); // zero mass => no contribution
                    for (k, a) in (lo..).zip(out.iter_mut()) {
                        if !masked || smask[k] {
                            *a = row(sx[k], sy[k], lx, ly, lm, simd);
                        }
                    }
                    (padded * sinks) as u64
                },
            )
            .sum();

        // Scatter back to the original body order (only sinks when masked).
        for (k, &key) in self.keys.iter().enumerate() {
            if !masked || self.smask[k] {
                acc[key as u32 as usize] = self.acc_sorted[k];
            }
        }
        interactions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gravity::GravityScratch;
    use crate::rng::SplitMix64;

    fn rms_rel_error(bodies: &[Body]) -> (f64, f64) {
        let mut direct = Vec::new();
        GravityScratch::default().direct(bodies, &mut direct);
        let mut bh = Vec::new();
        Tree::default().compute(bodies, &mut bh);
        let (mut e2, mut a2, mut rel2) = (0.0f64, 0.0f64, 0.0f64);
        for (d, t) in direct.iter().zip(&bh) {
            let ex = (t[0] - d[0]) as f64;
            let ey = (t[1] - d[1]) as f64;
            let mag2 = (d[0] as f64).powi(2) + (d[1] as f64).powi(2);
            e2 += ex * ex + ey * ey;
            a2 += mag2;
            rel2 += (ex * ex + ey * ey) / mag2.max(1e-30);
        }
        let n = bodies.len() as f64;
        ((e2 / a2).sqrt(), (rel2 / n).sqrt())
    }

    fn body(x: f32, y: f32, m: f32) -> Body {
        Body {
            pos: [x, y],
            mass: m,
            radius: 1.0,
            ..Body::default()
        }
    }

    #[test]
    fn barnes_hut_matches_direct_sum() {
        let mut rng = SplitMix64::new(9);
        // Uniform random set.
        let uniform: Vec<Body> = (0..5000)
            .map(|_| body(rng.range(-990.0, 990.0), rng.range(-990.0, 990.0), rng.range(0.5, 2.0)))
            .collect();
        // Clustered set with a heavy body (closer to the game's mass distribution).
        let mut clustered = vec![body(0.0, 0.0, 2000.0)];
        for c in 0..8 {
            let centre = [rng.range(-700.0, 700.0), rng.range(-700.0, 700.0)];
            for _ in 0..600 {
                let o = rng.in_disc(80.0 + 20.0 * c as f32);
                clustered.push(body(centre[0] + o[0], centre[1] + o[1], rng.range(0.1, 1.0)));
            }
        }
        for _ in 0..200 {
            clustered.push(body(rng.range(-990.0, 990.0), rng.range(-990.0, 990.0), 0.5));
        }
        for (name, set) in [("uniform", &uniform), ("clustered", &clustered)] {
            let (global, per_body) = rms_rel_error(set);
            println!("{name}: global RMS rel err {global:.2e}, per-body RMS rel err {per_body:.2e}");
            // global = |a_bh - a_direct|_rms / |a_direct|_rms (norm-weighted);
            // per-body = rms_i(|a_bh - a_direct|_i / |a_direct|_i), which is inflated by
            // bodies where opposing pulls nearly cancel.
            assert!(global < 0.01, "{name}: global RMS rel err {global}");
            assert!(per_body < 0.025, "{name}: per-body RMS rel err {per_body}");
        }
    }

    #[test]
    fn degenerate_inputs() {
        let mut acc = Vec::new();
        let mut t = Tree::default();
        assert_eq!(t.compute(&[], &mut acc), 0);
        assert!(acc.is_empty());
        // Many coincident bodies (max-depth leaf bigger than LEAF) and one far away.
        let mut bodies: Vec<Body> = (0..100).map(|_| body(5.0, 5.0, 1.0)).collect();
        bodies.push(body(-500.0, 0.0, 1.0));
        t.compute(&bodies, &mut acc);
        let mut direct = Vec::new();
        GravityScratch::default().direct(&bodies, &mut direct);
        for (a, d) in acc.iter().zip(&direct) {
            assert!((a[0] - d[0]).abs() <= 1e-3 * d[0].abs().max(1e-3));
            assert!((a[1] - d[1]).abs() <= 1e-3 * d[1].abs().max(1e-3) + 1e-6);
        }
    }
}
