//! Warm-dark-matter theory in 2D: transfer function, half-mode scale and mass,
//! sigma(M) with T(k), the Lovell-style suppression function and its fit, and the
//! 2D analogue of the Wang & White (2007) artificial-fragmentation limit.
//!
//! T(k) = [1 + (alpha k)^(2 nu)]^(-5/nu),  P_WDM = T^2 P_CDM.
//! Half mode: T(k_hm) = 1/2  =>  k_hm = (2^(nu/5) - 1)^(1/(2 nu)) / alpha.
//! 3D: M_hm = (4 pi / 3) rho (lambda_hm / 2)^3.  2D analogue (disc of radius lambda_hm / 2, as in 3D):
//! M_hm,2D = pi Sigma (lambda_hm / 2)^2 = pi Sigma (pi / k_hm)^2.
//! Suppression: n_WDM / n_CDM = f(M) = [1 + (alpha_f M_hm / M)^beta]^gamma.

use std::f64::consts::PI;

pub fn transfer(k: f64, alpha: f64, nu: f64) -> f64 {
    if alpha <= 0.0 {
        return 1.0;
    }
    (1.0 + (alpha * k).powf(2.0 * nu)).powf(-5.0 / nu)
}

pub fn k_half_mode(alpha: f64, nu: f64) -> f64 {
    (2f64.powf(nu / 5.0) - 1.0).powf(1.0 / (2.0 * nu)) / alpha
}

pub fn alpha_from_khm(k_hm: f64, nu: f64) -> f64 {
    (2f64.powf(nu / 5.0) - 1.0).powf(1.0 / (2.0 * nu)) / k_hm
}

/// 2D half-mode mass: M = pi Sigma (lambda_hm/2)^2, lambda_hm = 2 pi / k_hm.
pub fn m_half_mode_2d(k_hm: f64, sigma_bar: f64) -> f64 {
    PI * sigma_bar * (PI / k_hm).powi(2)
}

/// Inverse of [`m_half_mode_2d`].
pub fn k_hm_from_mhm_2d(m_hm: f64, sigma_bar: f64) -> f64 {
    PI * (PI * sigma_bar / m_hm).sqrt()
}

/// Wavenumber where the 2D dimensionless power k^2 P(k) peaks.
pub fn k_peak_2d(p: &dyn Fn(f64) -> f64, kmin: f64, kmax: f64) -> f64 {
    let n = 4000;
    let (l0, l1) = (kmin.ln(), kmax.ln());
    (0..=n)
        .map(|i| (l0 + (l1 - l0) * i as f64 / n as f64).exp())
        .max_by(|a, b| (a * a * p(*a)).total_cmp(&(b * b * p(*b))))
        .unwrap()
}

/// Calibration constant carried over from Wang & White (2007):
/// 3D M_lim = 10.1 rho d k_peak^-2 = kappa rho d * pi (pi / (2 k_peak))^2  =>  kappa = 4 * 10.1 / pi^3.
pub const WW_KAPPA: f64 = 4.0 * 10.1 / (PI * PI * PI);

/// 2D artificial-fragmentation mass. Spurious fragments form by discreteness along filaments:
/// fragment spacing ~ d (mean interparticle separation), mass per unit filament length = Sigma x width,
/// width = pi / k_peak (the same filament diameter that reproduces the 3D coefficient). Hence
/// M_lim,2D = kappa Sigma d pi / k_peak  (= 4.09 Sigma d / k_peak).
pub fn m_lim_2d(sigma_bar: f64, d: f64, k_peak: f64) -> f64 {
    WW_KAPPA * sigma_bar * d * PI / k_peak
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Supp {
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
}

impl Supp {
    pub fn f(&self, m: f64, m_hm: f64) -> f64 {
        if m_hm <= 0.0 {
            return 1.0;
        }
        (1.0 + (self.alpha * m_hm / m).powf(self.beta)).powf(self.gamma)
    }
}

/// Published 3D references.
/// * Lovell et al. 2020b (arXiv:2003.01125): subhaloes vs SUBFIND *bound* mass (not M_peak), from
///   sterile-neutrino simulations: alpha = 4.2, beta = 2.5, gamma = -0.2.
/// * Lovell et al. 2020b centrals: alpha = 2.3, beta = 0.8, gamma = -1.0.
/// * Lovell et al. 2014 (arXiv:1308.1399), genuine subhaloes: (1 + 2.7 M_hm/M)^-0.99, i.e.
///   alpha = 2.7, beta = 1, gamma = -0.99 in our form. Nadler et al. 2021 (arXiv:2008.00022) applied
///   this fit to M_peak.
/// * COZMIC I (Nadler et al. 2025, arXiv:2410.03635), M_peak: [1 + (2.5 M_hm/M_peak)^0.9]^-1.0.
pub const LOVELL2020B_SUB_BOUND: Supp = Supp { alpha: 4.2, beta: 2.5, gamma: -0.2 };
pub const LOVELL2020_CEN: Supp = Supp { alpha: 2.3, beta: 0.8, gamma: -1.0 };
pub const LOVELL2014_SUB: Supp = Supp { alpha: 2.7, beta: 1.0, gamma: -0.99 };
pub const COZMIC1_MPEAK: Supp = Supp { alpha: 2.5, beta: 0.9, gamma: -1.0 };

/// Lovell et al. 2014 spurious-halo cleaning: remove WDM haloes with M_max < LOVELL_CLEAN x M_lim
/// AND proto-halo sphericity below the CDM 1st-percentile cut (s ~ 0.16 in 3D).
pub const LOVELL_CLEAN: f64 = 0.5;

/// 3D power-law sigma(M) on dwarf scales for mapping 3D fits into sigma space: P ~ k^n3 with
/// n3 = n2D - 1 (Beacom et al. 1991 correspondence), sigma ~ M^-(n3+3)/6.
/// Returns M_hm/M for a given x = sigma(M)/sigma(M_hm).
pub fn mhm_over_m_3d(x: f64, n3: f64) -> f64 {
    x.powf(6.0 / (n3 + 3.0))
}
/// Counts in one mass bin of one WDM model, paired with the CDM counts in the same bin.
#[derive(Clone, Copy, Debug)]
pub struct BinPair {
    pub m_lo: f64,
    pub m_hi: f64,
    pub m_hm: f64,
    pub n_cdm: f64,
    pub n_wdm: f64,
}

/// Bin-averaged suppression (average over ln M, weighted by a dN/dlnM ~ M^-0.9 CDM function).
pub fn f_bin(s: &Supp, b: &BinPair) -> f64 {
    let n = 6;
    let (l0, l1) = (b.m_lo.ln(), b.m_hi.ln());
    let (mut num, mut den) = (0.0, 0.0);
    for i in 0..n {
        let lm = l0 + (l1 - l0) * (i as f64 + 0.5) / n as f64;
        let w = (-0.9 * lm).exp();
        num += w * s.f(lm.exp(), b.m_hm);
        den += w;
    }
    num / den
}

/// -ln L for N_wdm ~ Pois(f mu), N_cdm ~ Pois(mu) with mu profiled out:
/// conditional binomial N_wdm | N_tot ~ Bin(N_tot, f / (1 + f)).
pub fn neg_loglike(s: &Supp, data: &[BinPair]) -> f64 {
    data.iter()
        .map(|b| {
            let f = f_bin(s, b).clamp(1e-12, 1e6);
            let p = f / (1.0 + f);
            -(b.n_wdm * p.ln() + b.n_cdm * (1.0 - p).ln())
        })
        .sum()
}

/// Nelder-Mead minimiser.
pub fn nelder_mead(f: &dyn Fn(&[f64]) -> f64, x0: &[f64], step: f64, iters: usize) -> (Vec<f64>, f64) {
    let n = x0.len();
    let mut simplex: Vec<Vec<f64>> = vec![x0.to_vec()];
    for i in 0..n {
        let mut x = x0.to_vec();
        x[i] += step;
        simplex.push(x);
    }
    let mut vals: Vec<f64> = simplex.iter().map(|x| f(x)).collect();
    for _ in 0..iters {
        let mut order: Vec<usize> = (0..=n).collect();
        order.sort_by(|&a, &b| vals[a].total_cmp(&vals[b]));
        simplex = order.iter().map(|&i| simplex[i].clone()).collect();
        vals = order.iter().map(|&i| vals[i]).collect();
        if (vals[n] - vals[0]).abs() < 1e-10 * (1.0 + vals[0].abs()) {
            break;
        }
        let cen: Vec<f64> = (0..n).map(|j| simplex[..n].iter().map(|x| x[j]).sum::<f64>() / n as f64).collect();
        let pt = |t: f64| -> Vec<f64> { (0..n).map(|j| cen[j] + t * (simplex[n][j] - cen[j])).collect() };
        let xr = pt(-1.0);
        let fr = f(&xr);
        if fr < vals[0] {
            let xe = pt(-2.0);
            let fe = f(&xe);
            if fe < fr {
                simplex[n] = xe;
                vals[n] = fe;
            } else {
                simplex[n] = xr;
                vals[n] = fr;
            }
        } else if fr < vals[n - 1] {
            simplex[n] = xr;
            vals[n] = fr;
        } else {
            let xc = if fr < vals[n] { pt(-0.5) } else { pt(0.5) };
            let fc = f(&xc);
            if fc < vals[n].min(fr) {
                simplex[n] = xc;
                vals[n] = fc;
            } else {
                let s0 = simplex[0].clone();
                for (xi, vi) in simplex.iter_mut().zip(vals.iter_mut()).skip(1) {
                    for (x, &x0) in xi.iter_mut().zip(&s0) {
                        *x = x0 + 0.5 * (*x - x0);
                    }
                    *vi = f(xi);
                }
            }
        }
    }
    let ib = (0..=n).min_by(|&a, &b| vals[a].total_cmp(&vals[b])).unwrap();
    (simplex[ib].clone(), vals[ib])
}

/// Clamp of ln(beta): beta <= e^5 = 148. In sigma space (x = sigma(M)/sigma(M_hm), sigma ~ M^-1/12 in the
/// 3D mapping) a mass slope beta_M corresponds to ~12 beta_M, so steep slopes must be allowed.
pub const LN_BETA_MAX: f64 = 5.0;

pub fn unpack(x: &[f64]) -> Supp {
    Supp { alpha: x[0].clamp(-8.0, 8.0).exp(), beta: x[1].clamp(-3.0, LN_BETA_MAX).exp(), gamma: -x[2].clamp(-5.0, 2.5).exp() }
}

/// Maximum-likelihood (alpha_f, beta, gamma) with multiple starts. Parameterised as
/// (ln alpha, ln beta, ln(-gamma)) so beta > 0 and gamma < 0 always.
pub fn fit_supp(data: &[BinPair]) -> (Supp, f64) {
    let obj = |x: &[f64]| neg_loglike(&unpack(x), data);
    let mut best = (vec![0.0; 3], f64::INFINITY);
    for &a0 in &[0.5f64, 2.0, 6.0] {
        for &b0 in &[0.7f64, 1.5, 3.0] {
            for &g0 in &[0.3f64, 1.0, 2.5] {
                let x0 = [a0.ln(), b0.ln(), g0.ln()];
                let (x, v) = nelder_mead(&obj, &x0, 0.4, 600);
                if v < best.1 {
                    best = (x, v);
                }
            }
        }
    }
    let (x, v) = nelder_mead(&obj, &best.0, 0.1, 2000);
    (unpack(&x), v)
}

/// Mass at which the suppression reaches 50% (useful, well-constrained summary of a fit).
pub fn m_half_suppression(s: &Supp, m_hm: f64) -> f64 {
    let (mut lo, mut hi) = ((m_hm * 1e-6).ln(), (m_hm * 1e6).ln());
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if s.f(mid.exp(), m_hm) < 0.5 { lo = mid } else { hi = mid }
    }
    (0.5 * (lo + hi)).exp()
}
