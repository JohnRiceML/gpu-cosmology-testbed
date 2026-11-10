//! 2D galaxy-halo connection (adapted from Nadler et al. 2020) and the satellite-count
//! likelihood for the half-mode mass.
//!
//! * Luminosity: log10 L = s_L log10(M_peak / M_norm) + N(0, sigma_M)  (power-law SMHM with
//!   lognormal scatter; s_L = 2 mimics M* ~ M_peak^2 for dwarfs). The zero point M_norm and the
//!   observability cut L_cut are model choices: L_cut is tuned so the fiducial CDM Milky-Way
//!   analogues host `target` observable satellites. "Luminosity" is a toy label in 2D.
//! * Galaxy formation (reionization) threshold: f_gal(V) = 1/2 [1 + erf((V - V50) / (sqrt2 sigma_gal))],
//!   with V50 in units of V_res (median V_peak of 100-particle subhaloes) and sigma_gal in units of V50.
//! * Disruption: surviving subhaloes keep their galaxy; a disrupted branch keeps an orphan with
//!   p = p_orph exp(-(1 - a_infall) / tau)  (earlier infall -> longer stripping -> lower survival).
//! * Expected counts are sums of per-branch probabilities, so no galaxy draws are needed except for
//!   the mock survey.

use rayon::prelude::*;
use std::f64::consts::SQRT_2;

/// erfc with ~1.2e-7 relative accuracy (Numerical Recipes erfcc).
pub fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let r = t * (-z * z - 1.26551223
        + t * (1.00002368
            + t * (0.37409196
                + t * (0.09678418
                    + t * (-0.18628806
                        + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
        .exp();
    if x >= 0.0 { r } else { 2.0 - r }
}

pub fn erf(x: f64) -> f64 {
    1.0 - erfc(x)
}

/// Galaxy-formation probability (reionization threshold).
pub fn f_gal(v: f64, v50: f64, sigma: f64) -> f64 {
    if sigma <= 0.0 {
        return if v >= v50 { 1.0 } else { 0.0 };
    }
    0.5 * (1.0 + erf((v - v50) / (SQRT_2 * sigma)))
}

/// P(x > thr) for x ~ N(mean, sigma).
pub fn p_above(mean: f64, sigma: f64, thr: f64) -> f64 {
    if sigma <= 0.0 {
        return if mean >= thr { 1.0 } else { 0.0 };
    }
    0.5 * erfc((thr - mean) / (SQRT_2 * sigma))
}

#[derive(Clone, Copy, Debug)]
pub struct Sat {
    /// peak mass in particle masses
    pub m_peak: f64,
    pub v_peak: f64,
    pub a_infall: f64,
    pub surviving: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GalParams {
    /// V50 / V_res
    pub v50: f64,
    /// sigma_gal / V50
    pub sigma_gal: f64,
    /// lognormal scatter of L at fixed M_peak (dex)
    pub sigma_m: f64,
    pub p_orph: f64,
    pub tau_orph: f64,
}

pub const FIDUCIAL: GalParams = GalParams { v50: 1.5, sigma_gal: 0.15, sigma_m: 0.2, p_orph: 0.5, tau_orph: 0.5 };

/// Ranges of the Monte Carlo priors (v50 log-uniform, the rest uniform).
#[derive(Clone, Copy, Debug)]
pub struct Prior {
    pub v50: (f64, f64),
    pub sigma_gal: (f64, f64),
    pub sigma_m: (f64, f64),
    pub p_orph: (f64, f64),
}

pub const PRIOR_FIDUCIAL: Prior = Prior { v50: (1.2, 1.9), sigma_gal: (0.0, 0.3), sigma_m: (0.1, 0.3), p_orph: (0.0, 1.0) };
pub const PRIOR_WIDE: Prior = Prior { v50: (0.5, 4.0), sigma_gal: (0.0, 0.3), sigma_m: (0.1, 0.3), p_orph: (0.0, 1.0) };

pub struct Rng(pub u64);
impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    pub fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.uniform(), self.uniform());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    pub fn below(&mut self, n: usize) -> usize {
        ((self.uniform() * n as f64) as usize).min(n.saturating_sub(1))
    }
}

impl Prior {
    pub fn draw(&self, r: &mut Rng) -> GalParams {
        let lu = |r: &mut Rng, (a, b): (f64, f64)| (a.ln() + r.uniform() * (b / a).ln()).exp();
        let un = |r: &mut Rng, (a, b): (f64, f64)| a + r.uniform() * (b - a);
        GalParams {
            v50: lu(r, self.v50),
            sigma_gal: un(r, self.sigma_gal),
            sigma_m: un(r, self.sigma_m),
            p_orph: un(r, self.p_orph),
            tau_orph: FIDUCIAL.tau_orph,
        }
    }
}

/// Fixed (non-marginalised) parts of the model.
#[derive(Clone, Debug)]
pub struct GalModel {
    pub m_norm: f64,
    pub lum_slope: f64,
    pub v_res: f64,
    /// log10 L edges of the luminosity-function bins; the first edge is the observability cut,
    /// the last bin is open-ended
    pub lf_edges: Vec<f64>,
}

impl GalModel {
    pub fn new(m_norm: f64, v_res: f64, log_lcut: f64) -> GalModel {
        GalModel { m_norm, lum_slope: 2.0, v_res, lf_edges: (0..8).map(|i| log_lcut + 0.5 * i as f64).collect() }
    }
    pub fn nbins(&self) -> usize {
        self.lf_edges.len()
    }
    pub fn mean_log_l(&self, m: f64) -> f64 {
        self.lum_slope * (m / self.m_norm).log10()
    }
    /// Probability the satellite hosts a galaxy that survives (no luminosity cut).
    pub fn p_galaxy(&self, s: &Sat, p: &GalParams) -> f64 {
        let v50 = p.v50 * self.v_res;
        let surv = if s.surviving { 1.0 } else { p.p_orph * (-(1.0 - s.a_infall).max(0.0) / p.tau_orph).exp() };
        f_gal(s.v_peak, v50, p.sigma_gal * v50) * surv
    }
    /// Per-LF-bin probability of an observable galaxy, added into `out` (len nbins) times `w`.
    pub fn add_bin_probs(&self, s: &Sat, p: &GalParams, w: f64, out: &mut [f64]) {
        let pg = self.p_galaxy(s, p) * w;
        if pg <= 0.0 {
            return;
        }
        let mu = self.mean_log_l(s.m_peak);
        let nb = self.nbins();
        for (b, o) in out.iter_mut().enumerate().take(nb) {
            let lo = p_above(mu, p.sigma_m, self.lf_edges[b]);
            let hi = if b + 1 < nb { p_above(mu, p.sigma_m, self.lf_edges[b + 1]) } else { 0.0 };
            *o += pg * (lo - hi);
        }
    }
    pub fn expected_lf(&self, sats: &[Sat], p: &GalParams) -> Vec<f64> {
        let mut out = vec![0.0; self.nbins()];
        for s in sats {
            self.add_bin_probs(s, p, 1.0, &mut out);
        }
        out
    }
    /// Stochastic mock survey (Bernoulli galaxy occupation + lognormal luminosity scatter).
    pub fn mock_lf(&self, sats: &[Sat], p: &GalParams, rng: &mut Rng) -> Vec<f64> {
        let mut out = vec![0.0; self.nbins()];
        for s in sats {
            if rng.uniform() >= self.p_galaxy(s, p) {
                continue;
            }
            let ll = self.mean_log_l(s.m_peak) + p.sigma_m * rng.normal();
            if ll < self.lf_edges[0] {
                continue;
            }
            let b = self.lf_edges.partition_point(|&e| e <= ll) - 1;
            out[b] += 1.0;
        }
        out
    }
}

/// Choose log10 L_cut so the fiducial expected number of observable satellites per host is `target`.
pub fn tune_lcut(m_norm: f64, v_res: f64, sats: &[Sat], n_hosts: usize, target: f64) -> f64 {
    let total = |lc: f64| {
        let g = GalModel::new(m_norm, v_res, lc);
        g.expected_lf(sats, &FIDUCIAL).iter().sum::<f64>() / n_hosts.max(1) as f64
    };
    let (mut lo, mut hi) = (-12.0, 12.0);
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if total(mid) > target { lo = mid } else { hi = mid }
    }
    0.5 * (lo + hi)
}

fn ln_factorial(n: f64) -> f64 {
    // Stirling series is plenty for n > 20; exact sum below
    if n < 20.0 {
        (1..=(n as u64)).map(|i| (i as f64).ln()).sum()
    } else {
        n * n.ln() - n + 0.5 * (2.0 * std::f64::consts::PI * n).ln() + 1.0 / (12.0 * n) - 1.0 / (360.0 * n.powi(3))
    }
}

pub fn poisson_lnl(obs: &[f64], mu: &[f64]) -> f64 {
    obs.iter()
        .zip(mu)
        .map(|(&n, &m)| {
            let m = m.max(1e-300);
            n * m.ln() - m - ln_factorial(n)
        })
        .sum()
}

/// Precomputed per-mass-bin LF weights S_jb(theta_k) for a sample of galaxy parameters, so the
/// prediction for any M_hm is mu_b = sum_j f(M_j; M_hm) S_jb.
pub struct LikeTable {
    pub m_centers: Vec<f64>,
    /// [k][j][b]
    pub s: Vec<Vec<Vec<f64>>>,
    pub nb: usize,
}

const MBIN_DEX: f64 = 0.02;

impl LikeTable {
    pub fn build(model: &GalModel, sats: &[Sat], prior: &Prior, n_draws: usize, seed: u64) -> LikeTable {
        let lmin = sats.iter().map(|s| s.m_peak.log10()).fold(f64::INFINITY, f64::min);
        let lmax = sats.iter().map(|s| s.m_peak.log10()).fold(f64::NEG_INFINITY, f64::max);
        let nj = (((lmax - lmin) / MBIN_DEX).floor() as usize + 1).max(1);
        let m_centers: Vec<f64> = (0..nj).map(|j| 10f64.powf(lmin + (j as f64 + 0.5) * MBIN_DEX)).collect();
        let nb = model.nbins();
        let mut rng = Rng(seed);
        let params: Vec<GalParams> = (0..n_draws).map(|_| prior.draw(&mut rng)).collect();
        let s = params
            .par_iter()
            .map(|p| {
                let mut t = vec![vec![0.0; nb]; nj];
                for sat in sats {
                    let j = (((sat.m_peak.log10() - lmin) / MBIN_DEX) as usize).min(nj - 1);
                    model.add_bin_probs(sat, p, 1.0, &mut t[j]);
                }
                t
            })
            .collect();
        LikeTable { m_centers, s, nb }
    }

    pub fn mu(&self, k: usize, weight: &[f64]) -> Vec<f64> {
        let mut mu = vec![0.0; self.nb];
        for (j, row) in self.s[k].iter().enumerate() {
            for b in 0..self.nb {
                mu[b] += weight[j] * row[b];
            }
        }
        mu
    }

    /// Expected LF of draw k with no suppression (all mass bins summed).
    pub fn mu_total(&self, k: usize) -> Vec<f64> {
        let mut mu = vec![0.0; self.nb];
        for row in &self.s[k] {
            for (m, r) in mu.iter_mut().zip(row) {
                *m += r;
            }
        }
        mu
    }

    /// Marginal log-likelihood (log of the prior-averaged likelihood) for each M_hm.
    pub fn loglike_curve(&self, obs: &[f64], supp: &(dyn Fn(f64, f64) -> f64 + Sync), mhm_grid: &[f64]) -> Vec<f64> {
        mhm_grid
            .par_iter()
            .map(|&mhm| {
                let w: Vec<f64> = self.m_centers.iter().map(|&m| supp(m, mhm)).collect();
                let lls: Vec<f64> = (0..self.s.len()).map(|k| poisson_lnl(obs, &self.mu(k, &w))).collect();
                let mx = lls.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                mx + (lls.iter().map(|l| (l - mx).exp()).sum::<f64>() / lls.len() as f64).ln()
            })
            .collect()
    }
}

/// 95% (or `cl`) upper limit on M_hm from the posterior with a prior flat in log M_hm over the grid.
pub fn posterior_upper_limit(grid: &[f64], logl: &[f64], cl: f64) -> f64 {
    let mx = logl.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let w: Vec<f64> = logl.iter().map(|l| (l - mx).exp()).collect();
    // trapezoid in ln M
    let mut cdf = vec![0.0; grid.len()];
    for i in 1..grid.len() {
        cdf[i] = cdf[i - 1] + 0.5 * (w[i] + w[i - 1]) * (grid[i] / grid[i - 1]).ln();
    }
    let tot = *cdf.last().unwrap();
    if tot <= 0.0 {
        return f64::NAN;
    }
    let target = cl * tot;
    let i = cdf.partition_point(|&c| c < target).clamp(1, grid.len() - 1);
    let t = (target - cdf[i - 1]) / (cdf[i] - cdf[i - 1]).max(1e-300);
    (grid[i - 1].ln() + t * (grid[i] / grid[i - 1]).ln()).exp()
}

/// Profile-likelihood one-sided limit: smallest M_hm above the maximum where lnL drops by `dlnl`
/// (1.353 for one-sided 95%).
pub fn lr_upper_limit(grid: &[f64], logl: &[f64], dlnl: f64) -> f64 {
    let ib = (0..grid.len()).max_by(|&a, &b| logl[a].total_cmp(&logl[b])).unwrap();
    let thr = logl[ib] - dlnl;
    for i in ib + 1..grid.len() {
        if logl[i] < thr {
            let t = (thr - logl[i - 1]) / (logl[i] - logl[i - 1]);
            return (grid[i - 1].ln() + t * (grid[i] / grid[i - 1]).ln()).exp();
        }
    }
    f64::INFINITY
}
