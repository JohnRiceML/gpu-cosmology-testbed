//! A static 2D kd-tree used for (a) k-nearest-neighbour queries (SPH densities and the
//! SUBFIND neighbour graph) and (b) Barnes-Hut evaluation of the softened 2D logarithmic
//! potential during unbinding.
//!
//! Built by recursive median splits along the longer side of each node's bounding box
//! (leaves of at most `LEAF` points). Points are copied into tree order for locality.
//! Every node stores its tight bounding box, point count and centre of mass (equal-mass
//! particles), so the same tree serves both kinds of queries. Queries are read-only and
//! run in parallel from rayon.

use rayon::prelude::*;

const LEAF: usize = 8;
const NONE: u32 = u32::MAX;

#[derive(Clone, Copy, Debug)]
struct Node {
    lo: u32,
    hi: u32,
    bmin: [f32; 2],
    bmax: [f32; 2],
    left: u32,
    right: u32,
    com: [f64; 2],
}

pub struct KdTree {
    /// `idx[k]` = original index of the k-th point in tree order.
    idx: Vec<u32>,
    /// Points in tree order.
    pts: Vec<[f32; 2]>,
    nodes: Vec<Node>,
}

#[inline]
fn bbox_dist2(q: [f32; 2], bmin: [f32; 2], bmax: [f32; 2]) -> f32 {
    let dx = (bmin[0] - q[0]).max(q[0] - bmax[0]).max(0.0);
    let dy = (bmin[1] - q[1]).max(q[1] - bmax[1]).max(0.0);
    dx * dx + dy * dy
}

impl KdTree {
    pub fn build(points: &[[f32; 2]]) -> KdTree {
        let n = points.len();
        assert!(n < NONE as usize);
        let mut t = KdTree { idx: (0..n as u32).collect(), pts: Vec::new(), nodes: Vec::with_capacity(2 * n / LEAF + 2) };
        if n > 0 {
            t.build_rec(points, 0, n);
        }
        t.pts = t.idx.iter().map(|&i| points[i as usize]).collect();
        t
    }

    fn build_rec(&mut self, points: &[[f32; 2]], lo: usize, hi: usize) -> u32 {
        let mut bmin = [f32::INFINITY; 2];
        let mut bmax = [f32::NEG_INFINITY; 2];
        let (mut sx, mut sy) = (0.0f64, 0.0f64);
        for &i in &self.idx[lo..hi] {
            let p = points[i as usize];
            bmin = [bmin[0].min(p[0]), bmin[1].min(p[1])];
            bmax = [bmax[0].max(p[0]), bmax[1].max(p[1])];
            sx += p[0] as f64;
            sy += p[1] as f64;
        }
        let cnt = (hi - lo) as f64;
        let id = self.nodes.len() as u32;
        self.nodes.push(Node {
            lo: lo as u32,
            hi: hi as u32,
            bmin,
            bmax,
            left: NONE,
            right: NONE,
            com: [sx / cnt, sy / cnt],
        });
        if hi - lo > LEAF && (bmax[0] > bmin[0] || bmax[1] > bmin[1]) {
            let axis = if bmax[0] - bmin[0] >= bmax[1] - bmin[1] { 0 } else { 1 };
            let mid = (lo + hi) / 2;
            self.idx[lo..hi].select_nth_unstable_by(mid - lo, |&a, &b| {
                points[a as usize][axis].total_cmp(&points[b as usize][axis]).then(a.cmp(&b))
            });
            let l = self.build_rec(points, lo, mid);
            let r = self.build_rec(points, mid, hi);
            self.nodes[id as usize].left = l;
            self.nodes[id as usize].right = r;
        }
        id
    }

    pub fn len(&self) -> usize {
        self.idx.len()
    }

    pub fn is_empty(&self) -> bool {
        self.idx.is_empty()
    }

    /// The `k` nearest points to `q` (including a point at `q` itself, if any), as
    /// (squared distance, original index), sorted by distance then index.
    pub fn knn(&self, q: [f32; 2], k: usize, out: &mut Vec<(f32, u32)>, stack: &mut Vec<u32>) {
        out.clear();
        if self.nodes.is_empty() || k == 0 {
            return;
        }
        stack.clear();
        stack.push(0);
        while let Some(ni) = stack.pop() {
            let nd = &self.nodes[ni as usize];
            let worst = if out.len() < k { f32::INFINITY } else { out[k - 1].0 };
            if bbox_dist2(q, nd.bmin, nd.bmax) > worst {
                continue;
            }
            if nd.left == NONE {
                for t in nd.lo as usize..nd.hi as usize {
                    let p = self.pts[t];
                    let d2 = (p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2);
                    let worst = if out.len() < k { f32::INFINITY } else { out[k - 1].0 };
                    if d2 < worst || (d2 == worst && out.len() < k) {
                        let item = (d2, self.idx[t]);
                        let pos = out.partition_point(|e| (e.0, e.1) < item);
                        out.insert(pos, item);
                        out.truncate(k);
                    }
                }
            } else {
                let (l, r) = (&self.nodes[nd.left as usize], &self.nodes[nd.right as usize]);
                let (dl, dr) = (bbox_dist2(q, l.bmin, l.bmax), bbox_dist2(q, r.bmin, r.bmax));
                // Push the farther child first so the nearer one is visited first.
                if dl <= dr {
                    stack.push(nd.right);
                    stack.push(nd.left);
                } else {
                    stack.push(nd.left);
                    stack.push(nd.right);
                }
            }
        }
    }

    /// For every point i (original index): sum over all other points j of
    /// `0.5 * ln(|x_i - x_j|^2 + eps2)` (the softened 2D log kernel, unit masses), using
    /// Barnes-Hut monopoles with opening criterion `d > s/theta + delta` (s = node box
    /// size, delta = centre-of-mass offset from the box centre). Small sets use the exact
    /// direct sum.
    pub fn log_potential(&self, eps2: f64, theta: f64) -> Vec<f64> {
        let n = self.len();
        let mut out = vec![0.0f64; n];
        if n <= 1 {
            return out;
        }
        if n <= 1024 {
            for a in 0..n {
                let pa = self.pts[a];
                let mut s = 0.0f64;
                for b in 0..n {
                    if a != b {
                        let pb = self.pts[b];
                        let d2 = (pa[0] - pb[0]) as f64 * (pa[0] - pb[0]) as f64 + (pa[1] - pb[1]) as f64 * (pa[1] - pb[1]) as f64;
                        s += (d2 + eps2).ln();
                    }
                }
                out[self.idx[a] as usize] = 0.5 * s;
            }
            return out;
        }
        let tree_order: Vec<f64> = (0..n)
            .into_par_iter()
            .with_min_len(256)
            .map_init(Vec::new, |stack: &mut Vec<u32>, t| self.walk_log(t, eps2, theta, stack))
            .collect();
        for (t, v) in tree_order.into_iter().enumerate() {
            out[self.idx[t] as usize] = v;
        }
        out
    }

    fn walk_log(&self, t: usize, eps2: f64, theta: f64, stack: &mut Vec<u32>) -> f64 {
        let q = self.pts[t];
        let (qx, qy) = (q[0] as f64, q[1] as f64);
        let mut s = 0.0f64;
        stack.clear();
        stack.push(0);
        while let Some(ni) = stack.pop() {
            let nd = &self.nodes[ni as usize];
            let dx = nd.com[0] - qx;
            let dy = nd.com[1] - qy;
            let d2 = dx * dx + dy * dy;
            let size = (nd.bmax[0] - nd.bmin[0]).max(nd.bmax[1] - nd.bmin[1]) as f64;
            let cx = 0.5 * (nd.bmin[0] + nd.bmax[0]) as f64;
            let cy = 0.5 * (nd.bmin[1] + nd.bmax[1]) as f64;
            let delta = ((nd.com[0] - cx).powi(2) + (nd.com[1] - cy).powi(2)).sqrt();
            let open = size / theta + delta;
            let cnt = (nd.hi - nd.lo) as f64;
            // The criterion implies the query lies outside the node's box (open > size).
            if size > 0.0 && d2 > open * open {
                s += 0.5 * cnt * (d2 + eps2).ln();
            } else if nd.left == NONE || size == 0.0 {
                for u in nd.lo as usize..nd.hi as usize {
                    if u != t {
                        let p = self.pts[u];
                        let ex = p[0] as f64 - qx;
                        let ey = p[1] as f64 - qy;
                        s += 0.5 * (ex * ex + ey * ey + eps2).ln();
                    }
                }
            } else {
                stack.push(nd.left);
                stack.push(nd.right);
            }
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 40) as f32) / (1u64 << 24) as f32
    }

    #[test]
    fn knn_matches_brute_force() {
        let mut s = 7u64;
        let pts: Vec<[f32; 2]> = (0..3000).map(|_| [lcg(&mut s) * 100.0, lcg(&mut s) * 30.0]).collect();
        let t = KdTree::build(&pts);
        let (mut out, mut st) = (Vec::new(), Vec::new());
        for q in (0..3000).step_by(97) {
            t.knn(pts[q], 32, &mut out, &mut st);
            let mut bf: Vec<(f32, u32)> = pts
                .iter()
                .enumerate()
                .map(|(i, p)| ((p[0] - pts[q][0]).powi(2) + (p[1] - pts[q][1]).powi(2), i as u32))
                .collect();
            bf.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            assert_eq!(out, bf[..32].to_vec());
        }
    }

    #[test]
    fn log_potential_tree_vs_direct() {
        let mut s = 3u64;
        let pts: Vec<[f32; 2]> = (0..5000)
            .map(|_| {
                let r = 20.0 * lcg(&mut s).max(1e-6).sqrt();
                let th = lcg(&mut s) * std::f32::consts::TAU;
                [r * th.cos(), r * th.sin()]
            })
            .collect();
        let t = KdTree::build(&pts);
        let tree = t.log_potential(0.5, 0.5);
        // direct reference
        let mut max_rel = 0.0f64;
        for i in (0..5000).step_by(53) {
            let mut d = 0.0f64;
            for j in 0..5000 {
                if i != j {
                    let dx = (pts[i][0] - pts[j][0]) as f64;
                    let dy = (pts[i][1] - pts[j][1]) as f64;
                    d += 0.5 * (dx * dx + dy * dy + 0.5).ln();
                }
            }
            // compare against the spread of the potential (it is shift-invariant)
            max_rel = max_rel.max((tree[i] - d).abs() / 5000.0);
        }
        assert!(max_rel < 2e-3, "max per-pair error {max_rel}");
    }
}
