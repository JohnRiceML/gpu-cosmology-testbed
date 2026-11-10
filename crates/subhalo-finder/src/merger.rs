//! Merger trees from most-bound tracer particles.
//!
//! **Descendants.** Each subhalo carries the IDs of its `n_tracers` most-bound particles,
//! weighted `w_k = 1/(k+1)` by binding rank k (most bound counts most). In the next
//! snapshot every tracer votes for the subhalo that now contains it, or, if it is in a
//! FoF group but in no subhalo, for that group's central (the host core). The descendant
//! is the candidate with the largest weighted vote, provided it holds at least
//! `MIN_SCORE` of the total tracer weight.
//!
//! **Skips.** A satellite whose tracers land in a host central, or nowhere, may just be
//! temporarily unresolved (e.g. near pericentre). Such links are kept pending and the
//! snapshot after next is searched: if a *satellite* there receives the vote, it becomes
//! the descendant (a one-snapshot skip); otherwise the provisional link to the central
//! stands (merger) or, with no votes at all, the branch ends (tracers lost).
//!
//! **Main progenitor.** Among all progenitors of a subhalo (including skip links), the one
//! with the most bound particles (ties: lower id).
//!
//! **Branches.** Following main-progenitor links forward gives disjoint chains; a branch
//! ends at a subhalo that is not the main progenitor of its descendant (merged/disrupted),
//! has no descendant (tracers lost), or is in the last snapshot (surviving). A branch is
//! identified by the global id of its last subhalo.

use crate::{Catalog, NONE};

/// Minimum fraction of a subhalo's total tracer weight a descendant must receive.
pub const MIN_SCORE: f64 = 0.1;
/// Global subhalo ids: `snap_index * ID_STRIDE + index within the snapshot`.
pub const ID_STRIDE: u64 = 100_000_000;

pub fn global_id(snap: usize, idx: u32) -> u64 {
    snap as u64 * ID_STRIDE + idx as u64
}

/// Per-snapshot data kept for tree building (particle maps are not kept).
#[derive(Clone, Debug)]
pub struct SnapSummary {
    pub a: f64,
    pub host: Vec<u32>,
    pub is_central: Vec<bool>,
    pub n_bound: Vec<u32>,
    pub vmax: Vec<f64>,
    pub pos: Vec<[f64; 2]>,
    pub vel: Vec<[f64; 2]>,
    pub tracers: Vec<Vec<u32>>,
    /// FoF group -> central subhalo index (or NONE).
    pub central_of_group: Vec<u32>,
}

impl SnapSummary {
    pub fn from_catalog(c: &Catalog) -> SnapSummary {
        SnapSummary {
            a: c.a,
            host: c.subs.iter().map(|s| s.host).collect(),
            is_central: c.subs.iter().map(|s| s.is_central).collect(),
            n_bound: c.subs.iter().map(|s| s.n_bound).collect(),
            vmax: c.subs.iter().map(|s| s.vmax).collect(),
            pos: c.subs.iter().map(|s| s.pos).collect(),
            vel: c.subs.iter().map(|s| s.vel).collect(),
            tracers: c.subs.iter().map(|s| s.tracers.clone()).collect(),
            central_of_group: c.hosts.iter().map(|h| h.central).collect(),
        }
    }

    pub fn len(&self) -> usize {
        self.host.len()
    }

    pub fn is_empty(&self) -> bool {
        self.host.is_empty()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinkKind {
    /// Direct link to the next snapshot.
    Direct,
    /// Link over one missing snapshot.
    Skip,
    /// Tracers ended up in a host central (no satellite reappeared): merger.
    IntoCentral,
}

#[derive(Clone, Copy, Debug)]
pub struct Desc {
    pub snap: usize,
    pub idx: u32,
    pub kind: LinkKind,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LinkStats {
    pub direct: u64,
    pub skip: u64,
    pub into_central: u64,
    pub lost: u64,
    /// Subhalos in the last snapshot (no descendant search).
    pub last: u64,
}

/// Vote result: (target index, target is a central, score fraction).
type Vote = Option<(u32, bool, f64)>;

fn vote(tracers: &[u32], sub_of: &[u32], group_of: &[u32], next: &SnapSummary) -> Vote {
    let mut tally: Vec<(u32, f64)> = Vec::with_capacity(tracers.len());
    let mut total = 0.0;
    for (k, &t) in tracers.iter().enumerate() {
        let w = 1.0 / (k as f64 + 1.0);
        total += w;
        let mut target = sub_of[t as usize];
        if target == NONE {
            let g = group_of[t as usize];
            if g != NONE {
                target = next.central_of_group[g as usize];
            }
        }
        if target == NONE {
            continue;
        }
        match tally.iter_mut().find(|e| e.0 == target) {
            Some(e) => e.1 += w,
            None => tally.push((target, w)),
        }
    }
    let best = tally.into_iter().max_by(|a, b| a.1.total_cmp(&b.1).then(b.0.cmp(&a.0)))?;
    let frac = best.1 / total;
    (frac >= MIN_SCORE).then_some((best.0, next.is_central[best.0 as usize], frac))
}

#[derive(Default)]
pub struct Linker {
    pub snaps: Vec<SnapSummary>,
    /// Descendant of every subhalo (per snapshot).
    pub desc: Vec<Vec<Option<Desc>>>,
    /// Subhalos of snapshot `pending_snap` waiting for the next-next snapshot, with their
    /// provisional (into-central) descendant in `pending_snap + 1`.
    pending: Vec<(u32, Option<u32>)>,
    pending_snap: usize,
    pub stats: LinkStats,
}

impl Linker {
    pub fn new() -> Linker {
        Linker::default()
    }

    /// Add the next snapshot's catalog (in increasing `a`).
    pub fn add(&mut self, cat: &Catalog) {
        let s = self.snaps.len();
        let summary = SnapSummary::from_catalog(cat);
        self.desc.push(vec![None; summary.len()]);
        // 1. Resolve skips pending from s-2.
        if s >= 2 && self.pending_snap == s - 2 {
            let pend = std::mem::take(&mut self.pending);
            for (i, fallback) in pend {
                let tr = &self.snaps[s - 2].tracers[i as usize];
                let d = match vote(tr, &cat.sub_of, &cat.group_of, &summary) {
                    Some((t, false, _)) => {
                        self.stats.skip += 1;
                        Some(Desc { snap: s, idx: t, kind: LinkKind::Skip })
                    }
                    _ => match fallback {
                        Some(t) => {
                            self.stats.into_central += 1;
                            Some(Desc { snap: s - 1, idx: t, kind: LinkKind::IntoCentral })
                        }
                        None => {
                            self.stats.lost += 1;
                            None
                        }
                    },
                };
                self.desc[s - 2][i as usize] = d;
            }
        }
        // 2. Link s-1 -> s.
        if s >= 1 {
            let prev = &self.snaps[s - 1];
            for i in 0..prev.len() {
                match vote(&prev.tracers[i], &cat.sub_of, &cat.group_of, &summary) {
                    Some((t, into_central, _)) if !(into_central && !prev.is_central[i]) => {
                        self.stats.direct += 1;
                        self.desc[s - 1][i] = Some(Desc { snap: s, idx: t, kind: LinkKind::Direct });
                    }
                    Some((t, _, _)) => self.pending.push((i as u32, Some(t))),
                    None => self.pending.push((i as u32, None)),
                }
            }
            self.pending_snap = s - 1;
        }
        self.snaps.push(summary);
    }

    /// Resolve what is still pending (no further snapshot to search).
    pub fn finish(&mut self) {
        let s = self.pending_snap;
        for (i, fallback) in std::mem::take(&mut self.pending) {
            self.desc[s][i as usize] = match fallback {
                Some(t) => {
                    self.stats.into_central += 1;
                    Some(Desc { snap: s + 1, idx: t, kind: LinkKind::IntoCentral })
                }
                None => {
                    self.stats.lost += 1;
                    None
                }
            };
        }
        if let Some(last) = self.snaps.last() {
            self.stats.last = last.len() as u64;
        }
    }

    /// Build the forest: main progenitors, tree ids and branches.
    pub fn forest(&self) -> Forest {
        let ns = self.snaps.len();
        // main progenitor of every subhalo: (snap, idx) of the chosen progenitor
        let mut mp: Vec<Vec<Option<(usize, u32)>>> = self.snaps.iter().map(|s| vec![None; s.len()]).collect();
        for s in 0..ns {
            for i in 0..self.snaps[s].len() {
                if let Some(d) = self.desc[s][i] {
                    let cand = (s, i as u32);
                    let better = match mp[d.snap][d.idx as usize] {
                        None => true,
                        Some((ps, pi)) => {
                            let (nb_new, nb_old) = (self.snaps[s].n_bound[i], self.snaps[ps].n_bound[pi as usize]);
                            nb_new > nb_old || (nb_new == nb_old && global_id(s, i as u32) < global_id(ps, pi))
                        }
                    };
                    if better {
                        mp[d.snap][d.idx as usize] = Some(cand);
                    }
                }
            }
        }
        let is_mp = |s: usize, i: usize| -> bool {
            self.desc[s][i].is_some_and(|d| mp[d.snap][d.idx as usize] == Some((s, i as u32)))
        };
        // tree root (end of the descendant chain) and branch end, computed backwards.
        let mut root: Vec<Vec<(usize, u32)>> = self.snaps.iter().map(|s| vec![(0, 0); s.len()]).collect();
        let mut bend: Vec<Vec<(usize, u32)>> = root.clone();
        for s in (0..ns).rev() {
            for i in 0..self.snaps[s].len() {
                let me = (s, i as u32);
                match self.desc[s][i] {
                    Some(d) => {
                        root[s][i] = root[d.snap][d.idx as usize];
                        bend[s][i] = if is_mp(s, i) { bend[d.snap][d.idx as usize] } else { me };
                    }
                    None => {
                        root[s][i] = me;
                        bend[s][i] = me;
                    }
                }
            }
        }
        let is_mp_all: Vec<Vec<bool>> = (0..ns).map(|s| (0..self.snaps[s].len()).map(|i| is_mp(s, i)).collect()).collect();
        Forest { mp, is_mp: is_mp_all, root, bend }
    }
}

pub struct Forest {
    pub mp: Vec<Vec<Option<(usize, u32)>>>,
    pub is_mp: Vec<Vec<bool>>,
    /// End of the descendant chain (tree root) of every subhalo.
    pub root: Vec<Vec<(usize, u32)>>,
    /// Last subhalo of the branch every subhalo belongs to.
    pub bend: Vec<Vec<(usize, u32)>>,
}

#[derive(Clone, Debug)]
pub struct Branch {
    pub end: (usize, u32),
    pub surviving: bool,
    /// FoF host (group index) at the last snapshot it ends up in, if any.
    pub host_at_last: Option<u32>,
    pub mass_last: f64,
    pub vmax_last: f64,
    pub n_peak: u32,
    pub v_peak: f64,
    pub a_peak: f64,
    /// First a at which it was a satellite, if ever.
    pub a_infall: Option<f64>,
    pub first_a: f64,
    pub pos: [f64; 2],
    pub vel: [f64; 2],
    pub last_a: f64,
    pub length: usize,
}

/// Summaries of all branches (keyed by their end subhalo).
pub fn branches(linker: &Linker, forest: &Forest) -> Vec<Branch> {
    let ns = linker.snaps.len();
    let mut map: std::collections::BTreeMap<(usize, u32), Branch> = std::collections::BTreeMap::new();
    for s in 0..ns {
        let sn = &linker.snaps[s];
        for i in 0..sn.len() {
            let end = forest.bend[s][i];
            let b = map.entry(end).or_insert_with(|| {
                let es = &linker.snaps[end.0];
                let e = end.1 as usize;
                Branch {
                    end,
                    surviving: end.0 == ns - 1,
                    host_at_last: None,
                    mass_last: 0.0,
                    vmax_last: 0.0,
                    n_peak: 0,
                    v_peak: 0.0,
                    a_peak: 0.0,
                    a_infall: None,
                    first_a: f64::INFINITY,
                    pos: es.pos[e],
                    vel: es.vel[e],
                    last_a: es.a,
                    length: 0,
                }
            });
            b.length += 1;
            b.first_a = b.first_a.min(sn.a);
            if sn.n_bound[i] > b.n_peak {
                b.n_peak = sn.n_bound[i];
                b.a_peak = sn.a;
            }
            b.v_peak = b.v_peak.max(sn.vmax[i]);
            if !sn.is_central[i] && b.a_infall.is_none_or(|a| sn.a < a) {
                b.a_infall = Some(sn.a);
            }
        }
    }
    let last = ns - 1;
    for b in map.values_mut() {
        let (es, ei) = b.end;
        if b.surviving {
            b.host_at_last = Some(linker.snaps[last].host[ei as usize]);
            b.mass_last = linker.snaps[last].n_bound[ei as usize] as f64;
            b.vmax_last = linker.snaps[last].vmax[ei as usize];
            continue;
        }
        // Disrupted: where did it end up? Its descendant's tree root, or (tracers lost) the
        // root of the central of its last host.
        let target = match linker.desc[es][ei as usize] {
            Some(d) => Some(forest.root[d.snap][d.idx as usize]),
            None => {
                let sn = &linker.snaps[es];
                let c = sn.central_of_group[sn.host[ei as usize] as usize];
                (c != NONE && c != ei).then(|| forest.root[es][c as usize])
            }
        };
        if let Some((rs, ri)) = target
            && rs == last
        {
            b.host_at_last = Some(linker.snaps[last].host[ri as usize]);
        }
    }
    map.into_values().collect()
}
