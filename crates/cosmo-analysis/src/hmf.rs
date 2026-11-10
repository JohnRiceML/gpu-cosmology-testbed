//! 2D halo mass function: sigma(M) with a 2D top-hat, Press-Schechter and
//! Sheth-Tormen multiplicities, the 2D circular top-hat collapse threshold, and
//! Poisson (Cash) fits to the FoF counts.
//!
//! sigma^2(M, a) = D(a)^2 int P0(k) W^2(kR) k dk / (2 pi),  W(x) = 2 J1(x)/x,  M = rho_bar pi R^2
//! dn/dlnM = (rho_bar / M) f(nu) |dln sigma / dln M|,  nu = delta_c / sigma
//! f_PS(nu) = sqrt(2/pi) nu exp(-nu^2/2)
//! f_ST(nu) = A(p) sqrt(2q/pi) nu [1 + (q nu^2)^-p] exp(-q nu^2/2)
//! The multiplicity functions are dimension independent (they follow from the Gaussian
//! excursion-set mass fraction); only sigma(M), rho_bar (per unit area) and delta_c are 2D.

use crate::meta::Spectrum;
use rayon::prelude::*;
use std::f64::consts::PI;

/// Bessel J1 (rational approximations of Numerical Recipes, |err| ~ 1e-8).
pub fn bessel_j1(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 8.0 {
        let y = x * x;
        let n = x
            * (72362614232.0
                + y * (-7895059235.0
                    + y * (242396853.1 + y * (-2972611.439 + y * (15704.48260 + y * (-30.16036606))))));
        let d = 144725228442.0
            + y * (2300535178.0 + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y))));
        n / d
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 2.356194491;
        let p1 = 1.0
            + y * (0.183105e-2 + y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * (-0.240337019e-6))));
        let p2 = 0.04687499995
            + y * (-0.2002690873e-3 + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6)));
        let ans = (std::f64::consts::FRAC_2_PI / ax).sqrt() * (xx.cos() * p1 - z * xx.sin() * p2);
        if x < 0.0 { -ans } else { ans }
    }
}

/// 2D top-hat window in Fourier space: W(x) = 2 J1(x) / x.
pub fn tophat_w(x: f64) -> f64 {
    if x.abs() < 1e-3 {
        let x2 = x * x;
        1.0 - x2 / 8.0 + x2 * x2 / 192.0
    } else {
        2.0 * bessel_j1(x) / x
    }
}

/// sigma^2(R) at a_init: int_{k_lo}^{k_hi} P0(k) W^2(kR) k^2/(2 pi) dln k  (Simpson in ln k).
pub fn sigma2_r(spec: &Spectrum, r: f64, k_lo: f64, k_hi: f64) -> f64 {
    sigma2_r_with(&|k| spec.p0(k), r, k_lo, k_hi)
}

/// sigma^2(R) for an arbitrary spectrum P(k) (e.g. P0 T^2 for WDM).
pub fn sigma2_r_with(p: &dyn Fn(f64) -> f64, r: f64, k_lo: f64, k_hi: f64) -> f64 {
    let n = 16384usize;
    let (l0, l1) = (k_lo.ln(), k_hi.ln());
    let h = (l1 - l0) / n as f64;
    let mut s = 0.0;
    for i in 0..=n {
        let k = (l0 + h * i as f64).exp();
        let w = tophat_w(k * r);
        let f = p(k) * w * w * k * k / (2.0 * PI);
        let c = if i == 0 || i == n { 1.0 } else if i % 2 == 1 { 4.0 } else { 2.0 };
        s += c * f;
    }
    s * h / 3.0
}

/// Tabulated sigma(M) at a_init (M in particle masses).
pub struct SigmaTable {
    ln_m: Vec<f64>,
    ln_s: Vec<f64>,
    pub rho_bar: f64,
    pub k_lo: f64,
    pub k_hi: f64,
}

impl SigmaTable {
    /// rho_bar = Np / L^2 particle masses per unit area. Integration limits: k_lo = 2 pi / L
    /// (box fundamental), k_hi = min(IC Nyquist, 8 / R_c).
    pub fn new(spec: &Spectrum, box_size: f64, np: f64, k_nyq_ic: f64, m_max: f64) -> SigmaTable {
        let rho_bar = np / (box_size * box_size);
        let k_lo = 2.0 * PI / box_size;
        let mut k_hi = k_nyq_ic;
        if spec.r_c > 0.0 {
            k_hi = k_hi.min(8.0 / spec.r_c);
        }
        let k_hi = k_hi.max(2.0 * k_lo);
        let nt = 240usize;
        let (lm0, lm1) = (0.25f64.ln(), (m_max.max(10.0) * 4.0).ln());
        let ln_m: Vec<f64> = (0..nt).map(|i| lm0 + (lm1 - lm0) * i as f64 / (nt - 1) as f64).collect();
        let ln_s: Vec<f64> = ln_m
            .par_iter()
            .map(|&lm| {
                let r = (lm.exp() / (PI * rho_bar)).sqrt();
                0.5 * sigma2_r(spec, r, k_lo, k_hi).max(1e-300).ln()
            })
            .collect();
        SigmaTable { ln_m, ln_s, rho_bar, k_lo, k_hi }
    }

    fn locate(&self, lm: f64) -> (usize, f64) {
        let n = self.ln_m.len();
        let i = self.ln_m.partition_point(|&x| x < lm).clamp(1, n - 1);
        let t = (lm - self.ln_m[i - 1]) / (self.ln_m[i] - self.ln_m[i - 1]);
        (i, t)
    }

    pub fn sigma0(&self, m: f64) -> f64 {
        let (i, t) = self.locate(m.ln());
        (self.ln_s[i - 1] + t * (self.ln_s[i] - self.ln_s[i - 1])).exp()
    }

    pub fn dlnsigma_dlnm(&self, m: f64) -> f64 {
        let h = 0.02;
        let lm = m.ln();
        ((self.sigma0((lm + h).exp())).ln() - (self.sigma0((lm - h).exp())).ln()) / (2.0 * h)
    }
}

pub fn f_ps(nu: f64) -> f64 {
    (2.0 / PI).sqrt() * nu * (-0.5 * nu * nu).exp()
}

/// Lanczos approximation of Gamma(x) with reflection for x < 0.5.
pub fn gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        PI / ((PI * x).sin() * gamma(1.0 - x))
    } else {
        let x = x - 1.0;
        let mut a = C[0];
        let t = x + G + 0.5;
        for (i, c) in C.iter().enumerate().skip(1) {
            a += c / (x + i as f64);
        }
        (2.0 * PI).sqrt() * t.powf(x + 0.5) * (-t).exp() * a
    }
}

/// Sheth-Tormen normalisation so that int f dln nu = 1.
pub fn st_norm(p: f64) -> f64 {
    1.0 / (1.0 + 2f64.powf(-p) * gamma(0.5 - p) / PI.sqrt())
}

pub fn f_st(nu: f64, q: f64, p: f64) -> f64 {
    let qn2 = q * nu * nu;
    st_norm(p) * (2.0 * q / PI).sqrt() * nu * (1.0 + qn2.powf(-p)) * (-0.5 * qn2).exp()
}

/// Multiplicity model.
#[derive(Clone, Copy, Debug)]
pub enum Model {
    Ps { delta_c: f64 },
    St { delta_c: f64, q: f64, p: f64 },
}

impl Model {
    pub fn f(&self, sigma: f64) -> f64 {
        match *self {
            Model::Ps { delta_c } => f_ps(delta_c / sigma),
            Model::St { delta_c, q, p } => f_st(delta_c / sigma, q, p),
        }
    }
}

/// dn/dlnM per unit area at mass m (particle units) and growth factor g = D(a)/D(a_init).
pub fn dn_dlnm(t: &SigmaTable, m: f64, g: f64, model: Model) -> f64 {
    let sigma = t.sigma0(m) * g;
    t.rho_bar / m * model.f(sigma) * t.dlnsigma_dlnm(m).abs()
}

/// Mass fraction in halos above m: int_{m}^inf f dln nu (numerically).
pub fn mass_fraction_above(t: &SigmaTable, m: f64, g: f64, model: Model) -> f64 {
    // int dn/dlnM * M / rho_bar dlnM from m to large M
    let (l0, l1) = (m.ln(), (t.ln_m.last().copied().unwrap_or(m.ln() + 10.0)).max(m.ln() + 1.0));
    let n = 400;
    let h = (l1 - l0) / n as f64;
    let mut s = 0.0;
    for i in 0..=n {
        let mm = (l0 + h * i as f64).exp();
        let f = dn_dlnm(t, mm, g, model) * mm / t.rho_bar;
        let c = if i == 0 || i == n { 1.0 } else if i % 2 == 1 { 4.0 } else { 2.0 };
        s += c * f;
    }
    s * h / 3.0
}

/// Linear overdensity at collapse of a top-hat in `dim` dimensions, for the EdS-like
/// comoving equations that give D = a:
///   y'' + (1/2) y' = -(3/(2 dim)) (y^{1-dim} - y),   ' = d/dln a,   y = r / q.
/// dim = 3 reproduces the classic 1.686; dim = 2 is the circular (2D) top-hat.
pub fn tophat_delta_c(dim: u32) -> f64 {
    let d = dim as f64;
    let eps0 = 1e-6f64;
    let acc = |y: f64, ys: f64| -0.5 * ys - (1.5 / d) * (y.powi(1 - dim as i32) - y);
    let (mut s, mut y, mut ys) = (0.0f64, 1.0 - eps0, -eps0);
    loop {
        let h = (0.002f64).min(0.01 * y / ys.abs().max(1e-300));
        let k1 = (ys, acc(y, ys));
        let k2 = (ys + 0.5 * h * k1.1, acc(y + 0.5 * h * k1.0, ys + 0.5 * h * k1.1));
        let k3 = (ys + 0.5 * h * k2.1, acc(y + 0.5 * h * k2.0, ys + 0.5 * h * k2.1));
        let k4 = (ys + h * k3.1, acc(y + h * k3.0, ys + h * k3.1));
        let yn = y + h / 6.0 * (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0);
        let ysn = ys + h / 6.0 * (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1);
        if yn <= 1e-9 || !yn.is_finite() {
            let sc = s + y / ys.abs();
            return d * eps0 * sc.exp();
        }
        s += h;
        y = yn;
        ys = ysn;
        if y < 1e-7 {
            let sc = s + y / ys.abs();
            return d * eps0 * sc.exp();
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct HmfBin {
    pub m_lo: f64,
    pub m_hi: f64,
    pub m_center: f64,
    pub count: usize,
    pub dn_dlnm: f64,
    pub err: f64,
    pub n_gt: f64,
}

/// Log bins of `dlog10` with integer-aligned (half-integer) edges starting at min_members.
pub fn hmf_bins(sizes_desc: &[usize], min_members: usize, area: f64, dlog10: f64) -> Vec<HmfBin> {
    let Some(&max) = sizes_desc.first() else { return vec![] };
    let mut ints = vec![min_members.max(1)];
    let mut i = 1;
    while *ints.last().unwrap() <= max {
        let b = (min_members.max(1) as f64 * 10f64.powf(dlog10 * i as f64)).round() as usize;
        if b > *ints.last().unwrap() {
            ints.push(b);
        }
        i += 1;
    }
    let mut bins: Vec<HmfBin> = ints
        .windows(2)
        .map(|w| {
            let (lo, hi) = (w[0] as f64 - 0.5, w[1] as f64 - 0.5);
            HmfBin { m_lo: lo, m_hi: hi, m_center: (lo * hi).sqrt(), count: 0, dn_dlnm: 0.0, err: 0.0, n_gt: 0.0 }
        })
        .collect();
    for &s in sizes_desc {
        let x = s as f64;
        let b = bins.partition_point(|b| b.m_hi <= x);
        if b < bins.len() {
            bins[b].count += 1;
        }
    }
    for b in bins.iter_mut() {
        let dl = (b.m_hi / b.m_lo).ln();
        b.dn_dlnm = b.count as f64 / area / dl;
        b.err = (b.count as f64).sqrt() / area / dl;
        b.n_gt = sizes_desc.iter().take_while(|&&s| s as f64 > b.m_center).count() as f64 / area;
    }
    bins
}

/// Expected halo count in a bin (Simpson over ln M).
pub fn expected_count(t: &SigmaTable, b: &HmfBin, g: f64, model: Model, area: f64) -> f64 {
    let (l0, l1) = (b.m_lo.ln(), b.m_hi.ln());
    let n = 8;
    let h = (l1 - l0) / n as f64;
    let mut s = 0.0;
    for i in 0..=n {
        let c = if i == 0 || i == n { 1.0 } else if i % 2 == 1 { 4.0 } else { 2.0 };
        s += c * dn_dlnm(t, (l0 + h * i as f64).exp(), g, model);
    }
    s * h / 3.0 * area
}

/// Poisson deviance (Cash statistic, offset so a perfect model gives 0).
pub fn cash(t: &SigmaTable, bins: &[HmfBin], g: f64, model: Model, area: f64) -> f64 {
    bins.iter()
        .map(|b| {
            let mu = expected_count(t, b, g, model, area).max(1e-300);
            let n = b.count as f64;
            2.0 * (mu - n + if n > 0.0 { n * (n / mu).ln() } else { 0.0 })
        })
        .sum()
}

#[derive(Clone, Copy, Debug)]
pub struct Fit {
    pub delta_c_bestfit: f64,
    pub cash_ps_theory: f64,
    pub cash_ps_fit: f64,
    pub st_q: f64,
    pub st_p: f64,
    pub cash_st: f64,
    pub n_bins: usize,
    /// best-fit delta_c sits on the search boundary [0.2, 5]: PS shape cannot describe the data
    pub delta_c_at_bound: bool,
}

/// Fit delta_c (PS) and (q, p) of Sheth-Tormen at fixed delta_c = dc_theory.
/// Note that ST depends on delta_c only through q delta_c^2.
pub fn fit_hmf(t: &SigmaTable, bins: &[HmfBin], g: f64, area: f64, dc_theory: f64) -> Fit {
    let golden = |f: &dyn Fn(f64) -> f64, mut lo: f64, mut hi: f64| {
        let r = 0.5 * (5f64.sqrt() - 1.0);
        for _ in 0..80 {
            let (x1, x2) = (hi - r * (hi - lo), lo + r * (hi - lo));
            if f(x1) < f(x2) { hi = x2 } else { lo = x1 }
        }
        0.5 * (lo + hi)
    };
    let ps = |dc: f64| cash(t, bins, g, Model::Ps { delta_c: dc }, area);
    // coarse scan then golden refinement
    let grid: Vec<f64> = (0..=480).map(|i| 0.2 + 0.01 * i as f64).collect();
    let vals: Vec<f64> = grid.par_iter().map(|&x| ps(x)).collect();
    let ib = (0..grid.len()).min_by(|&a, &b| vals[a].total_cmp(&vals[b])).unwrap();
    let dc_fit = golden(&ps, grid[ib.saturating_sub(1)], grid[(ib + 1).min(grid.len() - 1)]);

    let qs: Vec<f64> = (0..=390).map(|i| 0.1 + 0.01 * i as f64).collect();
    let ps_grid: Vec<f64> = (0..=49).map(|i| 0.01 * i as f64).collect();
    let best = qs
        .par_iter()
        .flat_map_iter(|&q| ps_grid.iter().map(move |&p| (q, p)))
        .map(|(q, p)| (q, p, cash(t, bins, g, Model::St { delta_c: dc_theory, q, p }, area)))
        .min_by(|a, b| a.2.total_cmp(&b.2))
        .unwrap();
    Fit {
        delta_c_bestfit: dc_fit,
        cash_ps_theory: ps(dc_theory),
        cash_ps_fit: ps(dc_fit),
        st_q: best.0,
        st_p: best.1,
        cash_st: best.2,
        n_bins: bins.len(),
        delta_c_at_bound: !(0.21..=4.99).contains(&dc_fit),
    }
}
