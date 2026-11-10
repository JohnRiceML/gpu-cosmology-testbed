//! SUBFIND (Springel et al. 2001) adapted to 2D, applied to one FoF group.
//!
//! 1. **Density**: 2D SPH estimate from the `k_dens` nearest group members (kd-tree):
//!    `rho_i = sum_j m W(r_ij, h_i)` with `h_i` the distance to the `k_dens`-th neighbour
//!    and the 2D cubic spline `W = 40/(7 pi h^2) * {1 - 6q^2 + 6q^3 (q < 1/2); 2(1-q)^3
//!    (q < 1)}`, `q = r/h`.
//! 2. **Candidates**: particles are processed in order of decreasing density. For each, the
//!    two nearest of its `n_ngb` nearest neighbours that were already processed (denser)
//!    decide: none -> new subgroup (local maximum); one, or both in the same subgroup ->
//!    join it; two different subgroups -> the particle is a saddle point: the smaller
//!    subgroup is recorded as a subhalo candidate (if it has at least `n_min` members), the
//!    two are merged (smaller into larger) and the particle joins. Subgroups left at the end
//!    are recorded too, and if there is more than one of them (the neighbour graph is not
//!    connected) the whole group is added as the final background candidate. Candidates
//!    are therefore nested.
//! 3. **Unbinding**: candidates are processed in order of increasing size; each keeps only
//!    the particles not already claimed by a smaller accepted subhalo, then is iteratively
//!    unbound with the energy criterion documented in `units` (potential from a kd-tree
//!    Barnes-Hut sum of the softened log kernel, or the exact direct sum for small sets).
//!    Each iteration recomputes the potential, the potential minimum and the mean
//!    velocity, and removes the particles with E > 0 (at most `max_unbind_frac` of the set,
//!    the most unbound first). If at least `n_min` particles stay bound the candidate is
//!    accepted and its particles are claimed, so every particle ends up in at most one
//!    subhalo (the smallest bound structure containing it).
//! 4. The largest accepted subhalo of the group is its central; the rest are satellites.

use crate::kdtree::KdTree;
use crate::units::Cosmo;
use crate::Params;
use rayon::prelude::*;

const NONE: u32 = u32::MAX;

/// A bound subhalo in group-local terms.
#[derive(Clone, Debug)]
pub struct LocalSub {
    /// Group-local particle indices, most bound first.
    pub members: Vec<u32>,
    /// Total energy of each member (same order).
    pub energy: Vec<f64>,
    /// Group-local index of the minimum-potential member (the subhalo position, as in
    /// SUBFIND).
    pub centre: u32,
}

/// Group-local SPH densities and `n_ngb` nearest-neighbour lists (flat, `NONE`-padded).
pub fn densities(tree: &KdTree, pos: &[[f32; 2]], k_dens: usize, n_ngb: usize) -> (Vec<f32>, Vec<u32>) {
    let n = pos.len();
    let k = k_dens.min(n);
    let mut ngb = vec![NONE; n * n_ngb];
    let rho: Vec<f32> = ngb
        .par_chunks_mut(n_ngb)
        .enumerate()
        .with_min_len(64)
        .map_init(
            || (Vec::with_capacity(k + 1), Vec::new()),
            |(out, stack), (i, nl)| {
                tree.knn(pos[i], k, out, stack);
                let h2 = out.last().map_or(0.0, |e| e.0).max(1e-12);
                let h = h2.sqrt();
                let norm = 40.0 / (7.0 * std::f32::consts::PI * h2);
                let mut rho = 0.0f32;
                for &(d2, _) in out.iter() {
                    let q = d2.sqrt() / h;
                    let w = if q < 0.5 {
                        1.0 - 6.0 * q * q + 6.0 * q * q * q
                    } else if q < 1.0 {
                        2.0 * (1.0 - q).powi(3)
                    } else {
                        0.0
                    };
                    rho += norm * w;
                }
                let mut c = 0;
                for &(_, j) in out.iter() {
                    if j as usize != i && c < n_ngb {
                        nl[c] = j;
                        c += 1;
                    }
                }
                rho
            },
        )
        .collect();
    (rho, ngb)
}

/// Step 2: nested subhalo candidates (group-local member lists).
pub fn candidates(rho: &[f32], ngb: &[u32], n_ngb: usize, n_min: usize) -> Vec<Vec<u32>> {
    let n = rho.len();
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.par_sort_unstable_by(|&a, &b| rho[b as usize].total_cmp(&rho[a as usize]).then(a.cmp(&b)));
    let mut rank = vec![0u32; n];
    for (r, &i) in order.iter().enumerate() {
        rank[i as usize] = r as u32;
    }
    // Subgroups: head of each particle, singly linked member lists with tails and lengths.
    let mut head = vec![NONE; n];
    let mut next = vec![NONE; n];
    let mut tail = vec![NONE; n];
    let mut len = vec![0u32; n];
    let mut cands: Vec<Vec<u32>> = Vec::new();
    let collect = |h: u32, next: &[u32]| {
        let mut v = Vec::new();
        let mut p = h;
        while p != NONE {
            v.push(p);
            p = next[p as usize];
        }
        v
    };
    for &i in &order {
        let iu = i as usize;
        let r = rank[iu];
        let mut denser = [NONE; 2];
        let mut nd = 0;
        for &j in &ngb[iu * n_ngb..(iu + 1) * n_ngb] {
            if j == NONE {
                break;
            }
            if rank[j as usize] < r {
                denser[nd] = j;
                nd += 1;
                if nd == 2 {
                    break;
                }
            }
        }
        let target = match nd {
            0 => {
                head[iu] = i;
                tail[iu] = i;
                len[iu] = 1;
                continue;
            }
            1 => head[denser[0] as usize],
            _ => {
                let (ha, hb) = (head[denser[0] as usize], head[denser[1] as usize]);
                if ha == hb {
                    ha
                } else {
                    // Saddle point: record the smaller subgroup, merge it into the larger.
                    let (big, small) = if (len[ha as usize], hb) >= (len[hb as usize], ha) { (ha, hb) } else { (hb, ha) };
                    if len[small as usize] as usize >= n_min {
                        cands.push(collect(small, &next));
                    }
                    let mut p = small;
                    while p != NONE {
                        head[p as usize] = big;
                        p = next[p as usize];
                    }
                    next[tail[big as usize] as usize] = small;
                    tail[big as usize] = tail[small as usize];
                    len[big as usize] += len[small as usize];
                    big
                }
            }
        };
        let tu = target as usize;
        head[iu] = target;
        next[tail[tu] as usize] = i;
        tail[tu] = i;
        len[tu] += 1;
    }
    // Remaining top-level subgroups. The neighbour graph need not be connected (two dense
    // regions joined only by a thin FoF bridge), so there can be several; as in SUBFIND the
    // whole group is then the final (background) candidate as well.
    let mut tops = 0;
    for i in 0..n {
        if head[i] == i as u32 {
            tops += 1;
            if len[i] as usize >= n_min {
                cands.push(collect(i as u32, &next));
            }
        }
    }
    if tops > 1 {
        cands.push((0..n as u32).collect());
    }
    cands
}

/// Step 3 for one candidate: iterative unbinding. Returns bound members sorted by energy.
pub fn unbind(
    set: Vec<u32>,
    pos: &[[f32; 2]],
    vel: &[[f32; 2]],
    cosmo: &Cosmo,
    params: &Params,
) -> Option<LocalSub> {
    let mut cur = set;
    if cur.len() < params.n_min {
        return None;
    }
    let eps2 = cosmo.softening * cosmo.softening;
    let g = cosmo.g_log();
    let bg = cosmo.c_poisson / (4.0 * cosmo.a);
    let mut r_ref = 0.0f64;
    let mut energy: Vec<f64>;
    let mut centre;
    loop {
        let n = cur.len();
        let p: Vec<[f32; 2]> = cur.iter().map(|&i| pos[i as usize]).collect();
        let tree = KdTree::build(&p);
        let phi = tree.log_potential(eps2, params.theta);
        let c = (0..n).min_by(|&x, &y| phi[x].total_cmp(&phi[y]).then(x.cmp(&y))).unwrap();
        centre = cur[c];
        let xc = [p[c][0] as f64, p[c][1] as f64];
        let r2 = |q: [f32; 2]| (q[0] as f64 - xc[0]).powi(2) + (q[1] as f64 - xc[1]).powi(2);
        if r_ref == 0.0 {
            // Frozen at the initial candidate: its extent around the potential minimum.
            let r_ext = p.iter().map(|&q| r2(q)).fold(0.0f64, f64::max).sqrt();
            r_ref = (params.escape_factor * r_ext).max(cosmo.softening);
        }
        let (mut svx, mut svy) = (0.0f64, 0.0f64);
        for &i in &cur {
            svx += vel[i as usize][0] as f64;
            svy += vel[i as usize][1] as f64;
        }
        let vbar = [svx / n as f64, svy / n as f64];
        let shift = g * n as f64 * r_ref.ln() - bg * r_ref * r_ref;
        energy = cur
            .iter()
            .enumerate()
            .map(|(k, &i)| {
                let v = vel[i as usize];
                let ke = 0.5 * ((v[0] as f64 - vbar[0]).powi(2) + (v[1] as f64 - vbar[1]).powi(2));
                ke + g * phi[k] - bg * r2(p[k]) - shift
            })
            .collect();
        let unbound = energy.iter().filter(|&&e| e > 0.0).count();
        if unbound == 0 {
            break;
        }
        let cap = ((params.max_unbind_frac * n as f64) as usize).max(1);
        let keep: Vec<bool> = if unbound <= cap {
            energy.iter().map(|&e| e <= 0.0).collect()
        } else {
            // Remove only the `cap` most unbound this round.
            let mut by_e: Vec<usize> = (0..n).collect();
            by_e.sort_unstable_by(|&x, &y| energy[y].total_cmp(&energy[x]).then(x.cmp(&y)));
            let mut keep = vec![true; n];
            for &k in &by_e[..cap] {
                keep[k] = false;
            }
            keep
        };
        cur = cur.iter().zip(&keep).filter(|(_, k)| **k).map(|(&i, _)| i).collect();
        if cur.len() < params.n_min {
            return None;
        }
    }
    let mut idx: Vec<usize> = (0..cur.len()).collect();
    idx.sort_unstable_by(|&x, &y| energy[x].total_cmp(&energy[y]).then(cur[x].cmp(&cur[y])));
    Some(LocalSub {
        members: idx.iter().map(|&k| cur[k]).collect(),
        energy: idx.iter().map(|&k| energy[k]).collect(),
        centre,
    })
}

/// Steps 1-4 for one group in local (unwrapped) coordinates. Returns the accepted subhalos,
/// central (largest) first, then satellites by decreasing size.
pub fn find_in_group(pos: &[[f32; 2]], vel: &[[f32; 2]], cosmo: &Cosmo, params: &Params) -> Vec<LocalSub> {
    let n = pos.len();
    if n < params.n_min {
        return Vec::new();
    }
    let tree = KdTree::build(pos);
    let (rho, ngb) = densities(&tree, pos, params.k_dens, params.n_ngb);
    drop(tree);
    let mut cands = candidates(&rho, &ngb, params.n_ngb, params.n_min);
    drop(ngb);
    cands.sort_by(|a, b| a.len().cmp(&b.len()).then(a[0].cmp(&b[0])));
    let mut claimed = vec![false; n];
    let mut subs: Vec<LocalSub> = Vec::new();
    for cand in cands {
        let set: Vec<u32> = cand.into_iter().filter(|&i| !claimed[i as usize]).collect();
        if set.len() < params.n_min {
            continue;
        }
        if let Some(s) = unbind(set, pos, vel, cosmo, params) {
            for &i in &s.members {
                claimed[i as usize] = true;
            }
            subs.push(s);
        }
    }
    subs.sort_by(|a, b| b.members.len().cmp(&a.members.len()).then(a.members[0].cmp(&b.members[0])));
    subs
}
