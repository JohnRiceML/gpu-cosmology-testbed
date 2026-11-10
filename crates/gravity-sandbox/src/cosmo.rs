//! 2D cosmological structure formation (Scenario::CosmicWeb): periodic particle-mesh
//! gravity in comoving coordinates with Einstein-de Sitter expansion.
//!
//! # Equations (code units: box L = 2000, H0 = 1, time t in units of 1/H0)
//!
//! Expansion (EdS-like): a(t) ∝ t^(2/3), H = da/dt / a = H0 a^(-3/2), so dt = a^(1/2) da / H0.
//!
//! Comoving position x, canonical momentum p = a^2 dx/dt. Equations of motion:
//!   dx/dt = p / a^2,        dp/dt = -(1/a) ∇φ,        ∇²φ = C δ,   δ = ρ/ρ̄ - 1
//! (φ is the comoving peculiar potential; the 2D Poisson equation is solved spectrally on the
//! periodic mesh: φ_k = -C δ_k / k², k = 0 removed.)
//!
//! Linear growth (derivation). Linearising, continuity gives dδ/dt = -∇·(dx/dt) = -∇·p / a²,
//! i.e. a² dδ/dt = -∇·p. Taking ∇· of the momentum equation:
//!   d/dt (a² dδ/dt) = -∇·(dp/dt) = ∇²φ / a = C δ / a.
//! Change variable to a (primes = d/da), ȧ = a H = H0 a^(-1/2):
//!   a² dδ/dt = H0 a^(3/2) δ'
//!   d/dt (H0 a^(3/2) δ') = H0 a^(-1/2) · H0 ((3/2) a^(1/2) δ' + a^(3/2) δ'')
//!                        = H0² (a δ'' + (3/2) δ')
//! so   a² δ'' + (3/2) a δ' - (C/H0²) δ = 0.   Power law δ ∝ a^s:  s² + s/2 - C/H0² = 0.
//! Requiring the growing mode s = 1 gives
//!                          C = (3/2) H0²     ⇒  D(a) = a exactly,
//! and then s² + s/2 - 3/2 = (s - 1)(s + 3/2): decaying mode D_- ∝ a^(-3/2) (∝ 1/t), as in
//! EdS. (This is the 2D analogue of 4πGρ̄a³ = (3/2)H0² in comoving form.)
//! Verified: unit test `linear_mode_grows_as_a` (exact 1D plane wave with this integrator)
//! and the measured P(k) growth on the GPU (pk.csv: P_measured / P_linear at low k).
//!//! Integrator (Quinn et al. 1997 KDK, merged kicks, step a_n -> a_{n+1}, a_{n±1/2} geometric
//! midpoints):
//!   p_{n+1/2} = p_{n-1/2} + g(x_n) K(a_{n-1/2}, a_{n+1/2}),   g = -∇φ
//!   x_{n+1}   = x_n + p_{n+1/2} D(a_n, a_{n+1})
//!   K(a1,a2) = ∫ dt/a   = (2/H0) (a2^(1/2) - a1^(1/2))
//!   D(a1,a2) = ∫ dt/a^2 = (2/H0) (a1^(-1/2) - a2^(-1/2))
//! First step starts from synchronized p(a_init) (a_{-1/2} = a_init); before snapshots p is
//! synchronized with a partial kick K(a_{n-1/2}, a_n).
//!
//! Zel'dovich ICs: δ_init(k) with fixed amplitude |δ_k| = sqrt(P0(k)/L²) and random phases;
//! Ψ_k = i k δ_k / k² (so δ = -∇·Ψ); x = q + (D(a)/D(a_init)) Ψ;
//! dx/dt = (Ḋ/D_init) Ψ = H0 a^(-1/2) Ψ / a_init  ⇒  p = a² dx/dt = H0 a^(3/2) Ψ / a_init,
//! i.e. p(a_init) = H0 sqrt(a_init) Ψ.  Peculiar velocity (snapshots): v = a dx/dt = p / a.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use bytemuck::{Pod, Zeroable};
use sim_core::WORLD_HALF;

use crate::cosmo_io::{Job, RunWriter};

pub const H0: f64 = 1.0;
/// Poisson coefficient in ∇²φ = C δ that makes linear growth exactly D(a) = a (see the
/// derivation above; verified by `linear_mode_grows_as_a` and the measured P(k)).
pub const POISSON_C: f64 = 1.5 * H0 * H0;
pub const BOX: f64 = 2.0 * WORLD_HALF as f64;
pub const PK_BINS: usize = 40;
pub const SNAPSHOT_AS: [f64; 6] = [0.1, 0.2, 0.333, 0.5, 0.75, 1.0];

const WG: u32 = 256;
const FIXED_BUDGET: f64 = 3.0e9;

pub fn kick_factor(a1: f64, a2: f64) -> f64 {
    2.0 / H0 * (a2.sqrt() - a1.sqrt())
}

pub fn drift_factor(a1: f64, a2: f64) -> f64 {
    2.0 / H0 * (1.0 / a1.sqrt() - 1.0 / a2.sqrt())
}

/// Initial spectrum at a_init: P0(k) = A k^n exp(-(k R_c)^2) (CDM), times T(k)^2 for WDM.
#[derive(Clone, Copy, Debug)]
pub struct Spectrum {
    pub amp: f64,
    pub n: f64,
    pub rc: f64,
    /// WDM transfer-function scale α (0 = CDM)
    pub wdm_alpha: f64,
}

/// Viel et al. (2005) shape exponent.
pub const WDM_NU: f64 = 1.12;

/// WDM transfer function T(k) = [1 + (αk)^(2ν)]^(-5/ν).
pub fn wdm_transfer(alpha: f64, k: f64) -> f64 {
    if alpha <= 0.0 {
        return 1.0;
    }
    (1.0 + (alpha * k).powf(2.0 * WDM_NU)).powf(-5.0 / WDM_NU)
}

/// α such that T(k_hm)² = 1/4:  (1 + (α k_hm)^(2ν))^(-10/ν) = 1/4  ⇒  α = (2^(ν/5) - 1)^(1/(2ν)) / k_hm.
pub fn wdm_alpha_for_khm(k_hm: f64) -> f64 {
    (2f64.powf(WDM_NU / 5.0) - 1.0).powf(1.0 / (2.0 * WDM_NU)) / k_hm
}

impl Spectrum {
    pub fn p0_cdm(&self, k: f64) -> f64 {
        self.amp * k.powf(self.n) * (-(k * self.rc) * (k * self.rc)).exp()
    }
    pub fn p0(&self, k: f64) -> f64 {
        let t = wdm_transfer(self.wdm_alpha, k);
        self.p0_cdm(k) * t * t
    }
}

/// First-order Bessel J1 (Abramowitz & Stegun 9.4.4 / 9.4.6), |error| < 1e-7.
pub fn bessel_j1(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 3.0 {
        let y = (x / 3.0) * (x / 3.0);
        x * (0.5 - 0.56249985 * y + 0.21093573 * y * y - 0.03954289 * y.powi(3) + 0.00443319 * y.powi(4)
            - 0.00031761 * y.powi(5)
            + 0.00001109 * y.powi(6))
    } else {
        let y = 3.0 / ax;
        let f1 = 0.79788456 + 0.00000156 * y + 0.01659667 * y * y + 0.00017105 * y.powi(3) - 0.00249511 * y.powi(4)
            + 0.00113653 * y.powi(5)
            - 0.00020033 * y.powi(6);
        let t1 = ax - 2.35619449 + 0.12499612 * y + 0.00005650 * y * y - 0.00637879 * y.powi(3)
            + 0.00074348 * y.powi(4)
            + 0.00079824 * y.powi(5)
            - 0.00029166 * y.powi(6);
        let r = f1 * t1.cos() / ax.sqrt();
        if x < 0.0 { -r } else { r }
    }
}

/// rms linear δ in a 2D top-hat of radius R for spectrum p (continuum integral):
/// σ²(R) = ∫ dk k P(k) W²(kR) / (2π),  W(x) = 2 J1(x)/x.
pub fn sigma_tophat(p: impl Fn(f64) -> f64, r: f64) -> f64 {
    let (k0, k1, n) = (1e-5f64, 200.0f64, 20_000);
    let (l0, l1) = (k0.ln(), k1.ln());
    let mut s = 0.0;
    for i in 0..n {
        let k = (l0 + (l1 - l0) * (i as f64 + 0.5) / n as f64).exp();
        let x = k * r;
        let w = if x < 1e-6 { 1.0 } else { 2.0 * bessel_j1(x) / x };
        s += k * k * p(k) * w * w / (2.0 * std::f64::consts::PI) * (l1 - l0) / n as f64;
    }
    s.sqrt()
}

/// User-facing run configuration.
#[derive(Clone, Debug)]
pub struct CosmoConfig {
    pub mesh: u32,
    pub particles: u32,
    pub seed: u32,
    pub a_init: f64,
    pub a_final: f64,
    pub run_seconds: f64,
    pub spectral_index: f64,
    /// rms δ on the mesh at a_init
    pub sigma_init: f64,
    /// spectrum cutoff R_c in mesh cells
    pub rc_cells: f64,
    /// Gaussian force smoothing in mesh cells
    pub smooth_cells: f64,
    pub snapshots: bool,
    pub out_root: PathBuf,
    /// Paired-fixed partner run: δ → -δ (Angulo & Pontzen 2016); averaging the P(k) of the
    /// two runs cancels the realization-specific odd-order terms.
    pub flip_phases: bool,
    /// WDM half-mode wavenumber (None = CDM)
    pub k_hm: Option<f64>,
    /// Spectrum amplitude override (science runs: chosen from the host-halo criterion);
    /// None = normalise to sigma_init on the mesh.
    pub amp: Option<f64>,
    /// Short-range PP force (P³M) with Gaussian split r_s = smooth_cells.
    pub pp: bool,
    /// PP softening (world units)
    pub eps: f64,
    /// Uncapped science run: adaptive CFL steps instead of a real-time schedule.
    pub science: bool,
    /// CFL accuracy parameter η: Δt = sqrt(2 η ε / A_max)
    pub eta: f64,
    /// Smallest allowed Δln a in science mode (budget floor)
    pub min_dlna: f64,
    /// Snapshot epochs
    pub snapshot_as: Vec<f64>,
    /// Exact output directory (experiments); None = out_root/<UTC stamp>
    pub out_dir: Option<PathBuf>,
    /// Use `particles` as an exact square lattice (no rounding to a multiple of mesh/2).
    pub exact_lattice: bool,
    /// Generate the ICs on this mesh (same white noise / phases as a run on that mesh), then
    /// run on `mesh`. None = the run mesh.
    pub ic_mesh: Option<u32>,
}

/// Mode binning for P(k) + spectrum normalisation (built once per run on the CPU).
pub struct ModeTable {
    pub bin_of: Vec<u16>,
    pub count: Vec<u64>,
    pub k_mean: Vec<f64>,
    /// mean of P0 (with the final amplitude A) over the modes of each bin
    pub p0_mean: Vec<f64>,
    pub spectrum: Spectrum,
    /// rms displacement |Ψ| at a_init (for colouring)
    pub psi_rms: f64,
}

impl ModeTable {
    /// Bins: PK_BINS log-spaced in k from the fundamental to the Nyquist frequency. Modes with
    /// k = 0, on the Nyquist row/column (zeroed in the ICs) or beyond k_Nyq are excluded.
    /// A is chosen so the rms δ on the mesh at a_init equals `sigma`.
    pub fn build(n: usize, n_index: f64, rc: f64, sigma: f64, amp_override: Option<f64>, wdm_alpha: f64) -> Self {
        let kf = 2.0 * std::f64::consts::PI / BOX;
        let k_nyq = kf * (n / 2) as f64;
        let (lmin, lmax) = (kf.ln(), k_nyq.ln() + 1e-9);
        // normalisation always from the CDM spectrum (same A for paired CDM/WDM runs)
        let unit = Spectrum { amp: 1.0, n: n_index, rc, wdm_alpha: 0.0 };
        let mut bin_of = vec![u16::MAX; n * n];
        let mut count = vec![0u64; PK_BINS];
        let mut ksum = vec![0f64; PK_BINS];
        let mut psum = vec![0f64; PK_BINS];
        let (mut var, mut psi2) = (0f64, 0f64);
        let m = |i: usize| if i >= n / 2 { i as i64 - n as i64 } else { i as i64 };
        for iy in 0..n {
            for ix in 0..n {
                if ix == n / 2 || iy == n / 2 || (ix == 0 && iy == 0) {
                    continue;
                }
                let (kx, ky) = (kf * m(ix) as f64, kf * m(iy) as f64);
                let k = (kx * kx + ky * ky).sqrt();
                let p_cdm = unit.p0(k);
                let t = wdm_transfer(wdm_alpha, k);
                let p = p_cdm * t * t;
                var += p_cdm / (BOX * BOX); // σ² = Σ |δ_k|² = Σ P/L² (CDM)
                psi2 += p / (BOX * BOX) / (k * k);
                if k > k_nyq {
                    continue;
                }
                let b = (((k.ln() - lmin) / (lmax - lmin)) * PK_BINS as f64) as usize;
                let b = b.min(PK_BINS - 1);
                bin_of[iy * n + ix] = b as u16;
                count[b] += 1;
                ksum[b] += k;
                psum[b] += p;
            }
        }
        let amp = amp_override.unwrap_or(sigma * sigma / var);
        let k_mean = (0..PK_BINS).map(|b| if count[b] > 0 { ksum[b] / count[b] as f64 } else { 0.0 }).collect();
        let p0_mean = (0..PK_BINS).map(|b| if count[b] > 0 { amp * psum[b] / count[b] as f64 } else { 0.0 }).collect();
        Self {
            bin_of,
            count,
            k_mean,
            p0_mean,
            spectrum: Spectrum { amp, n: n_index, rc, wdm_alpha },
            psi_rms: (amp * psi2).sqrt(),
        }
    }
}

/// Matches `CosmoParams` in cosmo.wgsl (144 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct CosmoParams {
    n_particles: u32,
    side: u32,
    units: u32,
    measure: u32,
    seed: u32,
    pp_on: u32,
    chain_n: u32,
    /// workgroup offset of a pp_force dispatch chunk (split submissions)
    pp_base: u32,
    box_l: f32,
    world_half: f32,
    cell: f32,
    inv_cell: f32,
    kick: f32,
    drift: f32,
    poisson_c: f32,
    smooth_r: f32,
    mass_norm: f32,
    disp_init: f32,
    mom_init: f32,
    spec_a: f32,
    spec_n: f32,
    spec_rc: f32,
    well_kick: f32,
    wdm_alpha: f32,
    wdm_nu: f32,
    chain_cell: f32,
    inv_chain_cell: f32,
    r_cut: f32,
    eps: f32,
    g_pp: f32,
    g_thresh: f32,
    _f2: f32,
    well: [f32; 4],
}
/// One frame of the cosmological integration, prepared by the run scheduler.
#[derive(Clone, Copy, Debug)]
pub struct CosmoStep {
    pub a: f64,
    /// kick before the (optional) snapshot copy
    pub kick1: f64,
    /// kick after the snapshot copy (0 when no snapshot)
    pub kick2: f64,
    pub drift: f64,
    pub measure: bool,
    pub snapshot: bool,
    pub well: [f32; 4],
    pub well_kick: f32,
    /// |g| above which a particle's own CFL step would be shorter than this step (counted)
    pub g_thresh: f64,
}

const FREE: u32 = 0;
const RECORDED: u32 = 1;
const MAPPING: u32 = 2;
const READY: u32 = 3;

struct MeasureSlot {
    buf: wgpu::Buffer,
    state: Arc<AtomicU32>,
    a: f64,
}

/// Snapshot readback is chunked to keep host (commit) memory small: at the snapshot step the
/// particle buffer is copied into a device-local snapshot copy (no host memory), which is then
/// read back over the following frames through two small MAP_READ staging buffers.
const SNAP_CHUNK: u64 = 4 * 1024 * 1024; // particles per chunk (64 MB)

struct ChunkSlot {
    buf: wgpu::Buffer,
    state: Arc<AtomicU32>,
    offset: u64,
    count: u64,
}

struct SnapActive {
    a: f64,
    next: u64,
    delivered: u64,
    begin_sent: bool,
}

/// Short-range (P³M) force parameters, all in world units.
#[derive(Clone, Copy, Debug)]
pub struct PpParams {
    pub r_s: f64,
    pub r_cut: f64,
    pub eps: f64,
}

impl PpParams {
    /// r_s = 1.25 mesh cells, r_cut = 4.5 r_s.
    pub fn for_mesh(n: u32, eps: f64) -> Self {
        let r_s = 1.25 * BOX / n as f64;
        Self { r_s, r_cut: 4.5 * r_s, eps }
    }
}

/// Per-step maxima read back from the GPU (one or two frames late).
#[derive(Clone, Copy, Debug, Default)]
pub struct StepStats {
    pub a: f64,
    /// max |g| over particles (g = -∇φ, kick force per unit K)
    pub g_max: f64,
    /// max |p| after the kick
    pub p_max: f64,
    /// particles whose |g| exceeded the CFL threshold of that step
    pub over_cfl: u64,
}

struct PpBuffers {
    chain_n: u32,
    chain_count: wgpu::Buffer,
    chain_off: wgpu::Buffer,
    block_sum: wgpu::Buffer,
    cell_rank: wgpu::Buffer,
    sorted_pos: wgpu::Buffer,
    sorted_idx: wgpu::Buffer,
    acc_pp: wgpu::Buffer,
}

pub struct Cosmo {
    pub n: u32,
    pub side: u32,
    pub np: u32,
    params: CosmoParams,
    params_a: wgpu::Buffer,
    params_b: wgpu::Buffer,
    dens: wgpu::Buffer,
    grid: wgpu::Buffer,
    tw: wgpu::Buffer,
    accel: wgpu::Buffer,
    power: wgpu::Buffer,
    stats: wgpu::Buffer,
    pp: Option<PpBuffers>,
    dummy: wgpu::Buffer,
    layout: wgpu::BindGroupLayout,
    bg_a: Option<wgpu::BindGroup>,
    bg_b: Option<wgpu::BindGroup>,
    p_rows_fwd: wgpu::ComputePipeline,
    p_rows_inv: wgpu::ComputePipeline,
    p_ic_noise: wgpu::ComputePipeline,
    p_ic_cols: wgpu::ComputePipeline,
    p_ic_particles: wgpu::ComputePipeline,
    p_deposit: wgpu::ComputePipeline,
    p_rows_density: wgpu::ComputePipeline,
    p_cols_poisson: wgpu::ComputePipeline,
    p_rows_accel: wgpu::ComputePipeline,
    p_integrate: wgpu::ComputePipeline,
    p_pp_clear: wgpu::ComputePipeline,
    p_pp_count: wgpu::ComputePipeline,
    p_scan_blocks: wgpu::ComputePipeline,
    p_scan_sums: wgpu::ComputePipeline,
    p_scan_add: wgpu::ComputePipeline,
    p_pp_scatter: wgpu::ComputePipeline,
    p_pp_rank: wgpu::ComputePipeline,
    p_pp_force: wgpu::ComputePipeline,
    p_test_place: wgpu::ComputePipeline,
    measure: Vec<MeasureSlot>,
    measure_recorded: Option<usize>,
    snap: Option<SnapActive>,
    particles: Option<wgpu::Buffer>,
    stats_ring: crate::readback::ReadbackRing,
    stats_a: f64,
    pub last_stats: Option<StepStats>,
    /// one reusable snapshot staging buffer (np * 16 bytes) and its busy flag (cleared by the
    /// writer thread once the file is written and the buffer unmapped)
    snap_copy: Option<wgpu::Buffer>,
    snap_slots: Vec<ChunkSlot>,
    /// a readback failed: the run must not be marked complete
    pub readback_failed: bool,
    snap_recorded: bool,
}

fn storage(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn buf(device: &wgpu::Device, label: &str, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(16),
        usage,
        mapped_at_creation: false,
    })
}

pub fn shader_source(n: u32) -> String {
    (include_str!("shaders/cosmo.wgsl").to_string() + include_str!("shaders/fft.wgsl"))
        .replace("{{P}}", &n.to_string())
        .replace("{{LOGP}}", &n.trailing_zeros().to_string())
        .replace("{{PT}}", &(n / WG).to_string())
}

impl Cosmo {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, n: u32) -> Self {
        let sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cosmo"),
            source: wgpu::ShaderSource::Wgsl(shader_source(n).into()),
        });
        use wgpu::BufferUsages as U;
        let cells = (n as u64) * (n as u64);
        let psize = std::mem::size_of::<CosmoParams>() as u64;
        let tw = buf(device, "cosmo twiddles", (n as u64 / 2) * 8, U::STORAGE | U::COPY_DST);
        queue.write_buffer(&tw, 0, bytemuck::cast_slice(&crate::pm::twiddles(n)));
        let mut entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }];
        for b in 1..=14u32 {
            entries.push(storage(b, b == 4));
        }
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cosmo layout"),
            entries: &entries,
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cosmo pl"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let make = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pl),
                module: &sm,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let measure = (0..2)
            .map(|_| MeasureSlot {
                buf: buf(device, "cosmo power readback", cells * 4, U::MAP_READ | U::COPY_DST),
                state: Arc::new(AtomicU32::new(FREE)),
                a: 0.0,
            })
            .collect();
        let cell = (BOX / n as f64) as f32;
        Self {
            n,
            side: 0,
            np: 0,
            params: CosmoParams {
                n_particles: 0,
                side: 0,
                units: 1,
                measure: 0,
                seed: 1,
                pp_on: 0,
                chain_n: 1,
                pp_base: 0,
                box_l: BOX as f32,
                world_half: WORLD_HALF,
                cell,
                inv_cell: 1.0 / cell,
                kick: 0.0,
                drift: 0.0,
                poisson_c: POISSON_C as f32,
                smooth_r: 0.0,
                mass_norm: 0.0,
                disp_init: 1.0,
                mom_init: 0.0,
                spec_a: 0.0,
                spec_n: -1.0,
                spec_rc: 0.0,
                well_kick: 0.0,
                wdm_alpha: 0.0,
                wdm_nu: WDM_NU as f32,
                chain_cell: BOX as f32,
                inv_chain_cell: 1.0 / BOX as f32,
                r_cut: 0.0,
                eps: 0.0,
                g_pp: 0.0,
                g_thresh: f32::MAX,
                _f2: 0.0,
                well: [0.0; 4],
            },
            params_a: buf(device, "cosmo params a", psize, U::UNIFORM | U::COPY_DST),
            params_b: buf(device, "cosmo params b", psize, U::UNIFORM | U::COPY_DST),
            dens: buf(device, "cosmo density", cells * 4, U::STORAGE),
            grid: buf(device, "cosmo fft grid", cells * 8, U::STORAGE),
            tw,
            accel: buf(device, "cosmo accel", cells * 8, U::STORAGE),
            power: buf(device, "cosmo power", cells * 4, U::STORAGE | U::COPY_SRC),
            stats: buf(device, "cosmo stats", 16, U::STORAGE | U::COPY_DST | U::COPY_SRC),
            pp: None,
            dummy: buf(device, "cosmo pp dummy", 16, U::STORAGE),
            layout,
            bg_a: None,
            bg_b: None,
            p_rows_fwd: make("rows_fwd"),
            p_rows_inv: make("rows_inv"),
            p_ic_noise: make("ic_noise"),
            p_ic_cols: make("ic_cols"),
            p_ic_particles: make("ic_particles"),
            p_deposit: make("deposit"),
            p_rows_density: make("rows_density"),
            p_cols_poisson: make("cols_poisson"),
            p_rows_accel: make("rows_accel"),
            p_integrate: make("integrate"),
            p_pp_clear: make("pp_clear"),
            p_pp_count: make("pp_count"),
            p_scan_blocks: make("scan_blocks"),
            p_scan_sums: make("scan_sums"),
            p_scan_add: make("scan_add"),
            p_pp_scatter: make("pp_scatter"),
            p_pp_rank: make("pp_rank"),
            p_pp_force: make("pp_force"),
            p_test_place: make("test_place"),
            measure,
            measure_recorded: None,
            snap: None,
            particles: None,
            stats_ring: crate::readback::ReadbackRing::new(device, "cosmo stats readback", 16, 4),
            stats_a: 0.0,
            last_stats: None,
            snap_copy: None,
            snap_slots: (0..2)
                .map(|_| ChunkSlot {
                    buf: buf(device, "cosmo snapshot chunk", SNAP_CHUNK * 16, U::MAP_READ | U::COPY_DST),
                    state: Arc::new(AtomicU32::new(FREE)),
                    offset: 0,
                    count: 0,
                })
                .collect(),
            readback_failed: false,
            snap_recorded: false,
        }
    }

    /// Allocate (or drop) the P³M buffers for `np` particles.
    fn setup_pp(&mut self, device: &wgpu::Device, np: u32, pp: Option<PpParams>) {
        let Some(pp) = pp else {
            self.pp = None;
            self.params.pp_on = 0;
            return;
        };
        use wgpu::BufferUsages as U;
        // chaining cells >= r_cut, at most 1024^2 cells (scan capacity)
        let chain_n = ((BOX / pp.r_cut).floor() as u32).clamp(3, 1024);
        let cells = (chain_n as u64) * (chain_n as u64);
        let npu = np as u64;
        self.pp = Some(PpBuffers {
            chain_n,
            chain_count: buf(device, "pp chain count", cells * 4, U::STORAGE),
            chain_off: buf(device, "pp chain offsets", cells * 4, U::STORAGE),
            block_sum: buf(device, "pp block sums", 1024 * 4, U::STORAGE),
            cell_rank: buf(device, "pp cell/rank", npu * 8, U::STORAGE),
            sorted_pos: buf(device, "pp sorted positions", npu * 8, U::STORAGE),
            sorted_idx: buf(device, "pp sorted ids", npu * 4, U::STORAGE),
            acc_pp: buf(device, "pp acceleration", npu * 8, U::STORAGE),
        });
        let p = &mut self.params;
        p.pp_on = 1;
        p.chain_n = chain_n;
        p.chain_cell = (BOX / chain_n as f64) as f32;
        p.inv_chain_cell = (chain_n as f64 / BOX) as f32;
        p.smooth_r = pp.r_s as f32;
        p.r_cut = pp.r_cut as f32;
        p.eps = pp.eps as f32;
        p.g_pp = (POISSON_C * BOX * BOX / (2.0 * std::f64::consts::PI * np as f64)) as f32;
    }

    pub fn bind(&mut self, device: &wgpu::Device, particles: &wgpu::Buffer) {
        let make = |params: &wgpu::Buffer| {
            fn e(i: u32, b: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
                wgpu::BindGroupEntry {
                    binding: i,
                    resource: b.as_entire_binding(),
                }
            }
            let d = &self.dummy;
            let pp = self.pp.as_ref();
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("cosmo bg"),
                layout: &self.layout,
                entries: &[
                    e(0, params),
                    e(1, particles),
                    e(2, &self.dens),
                    e(3, &self.grid),
                    e(4, &self.tw),
                    e(5, &self.accel),
                    e(6, &self.power),
                    e(7, pp.map_or(d, |p| &p.chain_count)),
                    e(8, pp.map_or(d, |p| &p.chain_off)),
                    e(9, pp.map_or(d, |p| &p.block_sum)),
                    e(10, pp.map_or(d, |p| &p.cell_rank)),
                    e(11, pp.map_or(d, |p| &p.sorted_pos)),
                    e(12, pp.map_or(d, |p| &p.sorted_idx)),
                    e(13, pp.map_or(d, |p| &p.acc_pp)),
                    e(14, &self.stats),
                ],
            })
        };
        self.particles = Some(particles.clone());
        self.bg_a = Some(make(&self.params_a));
        self.bg_b = Some(make(&self.params_b));
    }

    /// Configure for `np = side^2` particles (allocates the PP buffers) and bind.
    pub fn configure(&mut self, device: &wgpu::Device, particles: &wgpu::Buffer, side: u32, cfg: &CosmoConfig) {
        self.side = side;
        self.np = side * side;
        let pp = cfg.pp.then(|| PpParams::for_mesh(self.n, cfg.eps));
        self.setup_pp(device, self.np, pp);
        // reusable snapshot staging buffer, allocated once per run (no per-snapshot allocation)
        let bytes = self.np as u64 * 16;
        if cfg.snapshots && self.snap_copy.as_ref().map(|b| b.size()) != Some(bytes) {
            self.snap_copy = None;
            self.snap_copy = Some(buf(device, "cosmo snapshot copy (device)", bytes, wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC));
        }
        self.readback_failed = false;
        if !cfg.pp {
            self.params.smooth_r = (cfg.smooth_cells * BOX / self.n as f64) as f32;
        }
        self.bind(device, particles);
    }

    /// Generate the Zel'dovich initial conditions (call `configure` first).
    pub fn initial_conditions(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, cfg: &CosmoConfig, spectrum: Spectrum) {
        self.initial_conditions_with(device, queue, cfg, spectrum, true);
    }

    /// `generate = false`: set the run parameters only and keep the particles already in the
    /// buffer (ICs generated by another Cosmo on a different mesh; the particle state, a
    /// displacement from the lattice point plus a momentum, does not depend on the mesh).
    pub fn initial_conditions_with(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, cfg: &CosmoConfig, spectrum: Spectrum, generate: bool) {
        let p = &mut self.params;
        p.n_particles = self.np;
        p.side = self.side;
        p.units = (FIXED_BUDGET / self.np as f64).clamp(16.0, 65_536.0) as u32;
        p.mass_norm = ((self.n as f64 * self.n as f64) / (self.np as f64 * p.units as f64)) as f32;
        p.seed = cfg.seed;
        let sign = if cfg.flip_phases { -1.0 } else { 1.0 };
        p.disp_init = sign as f32;
        p.mom_init = (sign * H0 * cfg.a_init.sqrt()) as f32;
        p.spec_a = spectrum.amp as f32;
        p.spec_n = spectrum.n as f32;
        p.spec_rc = spectrum.rc as f32;
        p.wdm_alpha = spectrum.wdm_alpha as f32;
        p.measure = 0;
        p.kick = 0.0;
        p.drift = 0.0;
        p.well = [0.0; 4];
        queue.write_buffer(&self.params_a, 0, bytemuck::bytes_of(&self.params));
        self.snap = None;
        self.last_stats = None;
        if !generate {
            return;
        }

        let n = self.n;
        let bg = self.bg_a.as_ref().expect("cosmo configure before ICs");
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("cosmo ic") });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("cosmo ic"), timestamp_writes: None });
            cp.set_bind_group(0, bg, &[]);
            cp.set_pipeline(&self.p_ic_noise);
            cp.dispatch_workgroups(n / 16, n / 16, 1);
            cp.set_pipeline(&self.p_rows_fwd);
            cp.dispatch_workgroups(n, 1, 1);
            cp.set_pipeline(&self.p_ic_cols);
            cp.dispatch_workgroups(n, 1, 1);
            cp.set_pipeline(&self.p_rows_inv);
            cp.dispatch_workgroups(n, 1, 1);
            cp.set_pipeline(&self.p_ic_particles);
            let (gx, gy) = crate::pm::dispatch_dims(self.np);
            cp.dispatch_workgroups(gx, gy, 1);
        }
        queue.submit(Some(enc.finish()));
        self.snap = None;
        self.last_stats = None;
    }

    /// Mesh + (optional) PP force passes, then the integrate pass with `bg`.
    /// With `split = Some((device, queue))` the work before the integrate pass is submitted
    /// in several short command buffers (the PP force in chunks of PP_CHUNK_GROUPS
    /// workgroups): late in a run the PP force takes ~0.5 s per step, and one long submission
    /// risks the OS GPU watchdog (TDR / device lost). `enc` is replaced by a fresh encoder.
    fn encode_forces<'a>(
        &self,
        split: Option<(&wgpu::Device, &wgpu::Queue)>,
        enc: &mut wgpu::CommandEncoder,
        bg: &wgpu::BindGroup,
        ts: &dyn Fn(usize) -> Option<wgpu::ComputePassTimestampWrites<'a>>,
    ) {
        // Late in a run the PP force is ~1 s per step. As one submission (or a deep queue of
        // chunks) it repeatedly ended in 'device lost' (Windows GPU watchdog); chunks of 4096
        // workgroups (~0.1-0.3 s each) with at most 2 in flight are stable. Same kernel, same
        // particles, same order of operations: the forces are bit-identical to one dispatch.
        // Late-run chunks take ~25 ms at n = 4096 (and ~2-4x that at n = 2048, where r_cut covers
        // 4x the area): far below the watchdog limit, so the chunk size stays the same. Smaller
        // chunks cost a CPU-GPU round trip each (measured: 1024-group chunks at n = 2048 were
        // dominated by the waits).
        let pp_chunk_groups: u32 = 4096;
        // at most this many PP chunks queued before waiting: with deep queues of long PP work the
        // driver reported 'device lost' (Windows GPU watchdog); bounded queues never did.
        const PP_INFLIGHT: u32 = 2;
        let n = self.n;
        let (gx, gy) = crate::pm::dispatch_dims(self.np);
        fn pass<'e>(enc: &mut wgpu::CommandEncoder, bg: &wgpu::BindGroup, label: &str, tsw: Option<wgpu::ComputePassTimestampWrites<'e>>, f: &mut dyn FnMut(&mut wgpu::ComputePass<'_>)) {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(label), timestamp_writes: tsw });
            cp.set_bind_group(0, bg, &[]);
            f(&mut cp);
        }
        pass(enc, bg, "cosmo deposit", ts(crate::pm::ST_DEPOSIT), &mut |cp| {
            cp.set_pipeline(&self.p_deposit);
            cp.dispatch_workgroups(gx, gy, 1);
        });
        pass(enc, bg, "cosmo rows fwd", ts(crate::pm::ST_ROWS_FWD), &mut |cp| {
            cp.set_pipeline(&self.p_rows_density);
            cp.dispatch_workgroups(n, 1, 1);
        });
        pass(enc, bg, "cosmo cols poisson", ts(crate::pm::ST_COLS), &mut |cp| {
            cp.set_pipeline(&self.p_cols_poisson);
            cp.dispatch_workgroups(n, 1, 1);
        });
        pass(enc, bg, "cosmo rows accel", ts(crate::pm::ST_ROWS_INV), &mut |cp| {
            cp.set_pipeline(&self.p_rows_accel);
            cp.dispatch_workgroups(n, 1, 1);
        });
        if let Some(pp) = &self.pp {
            let cells = pp.chain_n * pp.chain_n;
            let (cx, cy) = crate::pm::dispatch_dims(cells);
            let blocks = cells.div_ceil(1024);
            let total_groups = self.np.div_ceil(256);
            match split {
                None => {
                    pass(enc, bg, "cosmo pp (sort + force)", ts(crate::pm::ST_GRADIENT), &mut |cp| {
                        cp.set_pipeline(&self.p_pp_clear);
                        cp.dispatch_workgroups(cx, cy, 1);
                        cp.set_pipeline(&self.p_pp_count);
                        cp.dispatch_workgroups(gx, gy, 1);
                        cp.set_pipeline(&self.p_scan_blocks);
                        cp.dispatch_workgroups(blocks, 1, 1);
                        cp.set_pipeline(&self.p_scan_sums);
                        cp.dispatch_workgroups(1, 1, 1);
                        cp.set_pipeline(&self.p_scan_add);
                        cp.dispatch_workgroups(cx, cy, 1);
                        cp.set_pipeline(&self.p_pp_scatter);
                        cp.dispatch_workgroups(gx, gy, 1);
                        // deterministic in-cell order (see pp_rank), then the final scatter
                        cp.set_pipeline(&self.p_pp_rank);
                        cp.dispatch_workgroups(gx, gy, 1);
                        cp.set_pipeline(&self.p_pp_scatter);
                        cp.dispatch_workgroups(gx, gy, 1);
                        cp.set_pipeline(&self.p_pp_force);
                        cp.dispatch_workgroups(gx, gy, 1);
                    });
                }
                Some((device, queue)) => {
                    let begin = ts(crate::pm::ST_GRADIENT).map(|t| wgpu::ComputePassTimestampWrites { end_of_pass_write_index: None, ..t });
                    pass(enc, bg, "cosmo pp sort", begin, &mut |cp| {
                        cp.set_pipeline(&self.p_pp_clear);
                        cp.dispatch_workgroups(cx, cy, 1);
                        cp.set_pipeline(&self.p_pp_count);
                        cp.dispatch_workgroups(gx, gy, 1);
                        cp.set_pipeline(&self.p_scan_blocks);
                        cp.dispatch_workgroups(blocks, 1, 1);
                        cp.set_pipeline(&self.p_scan_sums);
                        cp.dispatch_workgroups(1, 1, 1);
                        cp.set_pipeline(&self.p_scan_add);
                        cp.dispatch_workgroups(cx, cy, 1);
                        cp.set_pipeline(&self.p_pp_scatter);
                        cp.dispatch_workgroups(gx, gy, 1);
                        // deterministic in-cell order (see pp_rank), then the final scatter
                        cp.set_pipeline(&self.p_pp_rank);
                        cp.dispatch_workgroups(gx, gy, 1);
                        cp.set_pipeline(&self.p_pp_scatter);
                        cp.dispatch_workgroups(gx, gy, 1);
                    });
                    let fresh = || device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame (cont.)") });
                    queue.submit(Some(std::mem::replace(enc, fresh()).finish()));
                    let mut base = 0u32;
                    let mut chunk_i = 0u32;
                    let pp_inflight = PP_INFLIGHT;
                    while base < total_groups {
                        let groups = pp_chunk_groups.min(total_groups - base);
                        // pp_base is read by pp_force only; written between submissions
                        queue.write_buffer(&self.params_a, 28, &base.to_le_bytes());
                        let last = base + groups >= total_groups;
                        let tsw = if last { ts(crate::pm::ST_GRADIENT).map(|t| wgpu::ComputePassTimestampWrites { beginning_of_pass_write_index: None, ..t }) } else { None };
                        let mut ce = fresh();
                        pass(&mut ce, bg, "cosmo pp force chunk", tsw, &mut |cp| {
                            cp.set_pipeline(&self.p_pp_force);
                            cp.dispatch_workgroups(groups, 1, 1);
                        });
                        queue.submit(Some(ce.finish()));
                        chunk_i += 1;
                        if chunk_i % pp_inflight == 0 {
                            // bounded number of PP submissions in flight (see PP_CHUNK_GROUPS)
                            let _ = device.poll(wgpu::PollType::wait_indefinitely());
                        }
                        base += groups;
                    }
                    queue.write_buffer(&self.params_a, 28, &0u32.to_le_bytes());
                }
            }
        }
        pass(enc, bg, "cosmo integrate", ts(crate::pm::ST_INTEGRATE), &mut |cp| {
            cp.set_pipeline(&self.p_integrate);
            cp.dispatch_workgroups(gx, gy, 1);
        });
    }
    /// Record one PM step. `ts(stage)` gives optional timestamp writes per pass.
    pub fn encode<'a>(
        &mut self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        enc: &mut wgpu::CommandEncoder,
        step: &CosmoStep,
        ts: &dyn Fn(usize) -> Option<wgpu::ComputePassTimestampWrites<'a>>,
    ) {
        let (Some(bg_a), Some(bg_b), Some(particles)) = (self.bg_a.clone(), self.bg_b.clone(), self.particles.clone()) else {
            return;
        };
        let slot = if step.measure {
            self.measure.iter().position(|s| s.state.load(Ordering::Acquire) == FREE)
        } else {
            None
        };
        let snapshot = step.snapshot && self.snap.is_none() && self.snap_copy.is_some();

        let mut pa = self.params;
        pa.measure = slot.is_some() as u32;
        pa.kick = step.kick1 as f32;
        pa.drift = if snapshot { 0.0 } else { step.drift as f32 };
        pa.well = step.well;
        pa.well_kick = step.well_kick;
        pa.g_thresh = step.g_thresh as f32;
        queue.write_buffer(&self.params_a, 0, bytemuck::bytes_of(&pa));
        queue.write_buffer(&self.stats, 0, &[0u8; 16]);
        self.encode_forces(Some((_device, queue)), enc, &bg_a, ts);

        let n = self.n;
        let (gx, gy) = crate::pm::dispatch_dims(self.np);
        if snapshot {
            let mut pb = pa;
            pb.measure = 0;
            pb.kick = step.kick2 as f32;
            pb.drift = step.drift as f32;
            pb.well = [0.0; 4];
            // x is unchanged, so acc_pp (PP force at x_n) is reused by the second integrate
            queue.write_buffer(&self.params_b, 0, bytemuck::bytes_of(&pb));
            let bytes = self.np as u64 * 16;
            let copy = self.snap_copy.clone().unwrap();
            enc.copy_buffer_to_buffer(&particles, 0, &copy, 0, bytes);
            {
                // finish the step: remaining kick with the same forces (x unchanged) + drift
                let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("cosmo integrate (after snapshot)"), timestamp_writes: None });
                cp.set_bind_group(0, &bg_b, &[]);
                cp.set_pipeline(&self.p_integrate);
                cp.dispatch_workgroups(gx, gy, 1);
            }
            self.snap = Some(SnapActive { a: step.a, next: 0, delivered: 0, begin_sent: false });
        }
        if let Some(i) = slot {
            enc.copy_buffer_to_buffer(&self.power, 0, &self.measure[i].buf, 0, (n as u64) * (n as u64) * 4);
            self.measure[i].a = step.a;
            self.measure_recorded = Some(i);
        }
        if snapshot {
            self.snap_recorded = true;
        }
        self.encode_snapshot_chunks(enc);
        if let Some(staging) = self.stats_ring.begin() {
            enc.copy_buffer_to_buffer(&self.stats, 0, staging, 0, 16);
            self.stats_a = step.a;
        }
    }

    /// Whether the last `encode` actually recorded the requested measurement/snapshot.
    pub fn recorded(&self) -> (bool, bool) {
        (
            self.measure_recorded.is_some(),
            self.snap_recorded,
        )
    }

    /// Overwrite the particle state (restart from a snapshot).
    pub fn load_state(&mut self, queue: &wgpu::Queue, data: &[[f32; 4]]) {
        if let Some(p) = &self.particles {
            queue.write_buffer(p, 0, bytemuck::cast_slice(data));
        }
        self.snap = None;
        self.last_stats = None;
    }

    pub fn snapshot_in_flight(&self) -> bool {
        self.snap.is_some()
    }

    /// Record copies of the next snapshot chunks into free staging slots (any frame).
    pub fn encode_snapshot_chunks(&mut self, enc: &mut wgpu::CommandEncoder) {
        let (Some(s), Some(copy)) = (self.snap.as_mut(), self.snap_copy.as_ref()) else {
            return;
        };
        for slot in &mut self.snap_slots {
            if s.next >= self.np as u64 {
                break;
            }
            if slot.state.load(Ordering::Acquire) != FREE {
                continue;
            }
            let count = SNAP_CHUNK.min(self.np as u64 - s.next);
            enc.copy_buffer_to_buffer(copy, s.next * 16, &slot.buf, 0, count * 16);
            slot.offset = s.next;
            slot.count = count;
            slot.state.store(RECORDED, Ordering::Release);
            s.next += count;
        }
    }

    /// Call after the submit containing `encode`: start the async maps.
    pub fn after_submit(&mut self) {
        if let Some(i) = self.measure_recorded.take() {
            let s = &self.measure[i];
            s.state.store(MAPPING, Ordering::Release);
            let st = s.state.clone();
            s.buf.map_async(wgpu::MapMode::Read, .., move |r| {
                st.store(if r.is_ok() { READY } else { FREE }, Ordering::Release);
            });
        }
        for slot in &self.snap_slots {
            if slot.state.load(Ordering::Acquire) == RECORDED {
                slot.state.store(MAPPING, Ordering::Release);
                let st = slot.state.clone();
                slot.buf.map_async(wgpu::MapMode::Read, .., move |r| {
                    st.store(if r.is_ok() { READY } else { FREE + 10 }, Ordering::Release);
                });
            }
        }
        self.snap_recorded = false;        self.stats_ring.after_submit();
    }

    /// Hand finished readbacks to the writer thread and update `last_stats`.
    pub fn poll(&mut self, writer: &RunWriter) {
        let mut st = None;
        self.stats_ring.read_latest(|b| {
            let v: &[u32] = bytemuck::cast_slice(b);
            st = Some((f32::from_bits(v[0]) as f64, f32::from_bits(v[1]) as f64, v[2] as u64));
        });
        if let Some((g, p, c)) = st {
            self.last_stats = Some(StepStats { a: self.stats_a, g_max: g, p_max: p, over_cfl: c });
        }
        let n2 = (self.n as usize) * (self.n as usize);
        let mut ready: Vec<usize> = (0..self.measure.len())
            .filter(|&i| self.measure[i].state.load(Ordering::Acquire) == READY)
            .collect();
        ready.sort_by(|&x, &y| self.measure[x].a.total_cmp(&self.measure[y].a));
        for i in ready {
            let s = &self.measure[i];
            let power: Vec<f32> = match s.buf.slice(..).get_mapped_range() {
                Ok(v) => bytemuck::cast_slice::<u8, f32>(&v)[..n2].to_vec(),
                Err(_) => Vec::new(),
            };
            s.buf.unmap();
            s.state.store(FREE, Ordering::Release);
            if !power.is_empty() {
                writer.send(Job::Pk { a: s.a, power });
            }
        }
        if let Some(s) = self.snap.as_mut() {
            if !s.begin_sent {
                writer.send(Job::SnapBegin { a: s.a, count: self.np as u64 });
                s.begin_sent = true;
            }
            for slot in &self.snap_slots {
                match slot.state.load(Ordering::Acquire) {
                    READY => {
                        let data: Vec<[f32; 4]> = match slot.buf.slice(..slot.count * 16).get_mapped_range() {
                            Ok(v) => bytemuck::cast_slice::<u8, [f32; 4]>(&v).to_vec(),
                            Err(_) => Vec::new(),
                        };
                        slot.buf.unmap();
                        slot.state.store(FREE, Ordering::Release);
                        if data.len() as u64 != slot.count {
                            self.readback_failed = true;
                        }
                        s.delivered += slot.count;
                        writer.send(Job::SnapChunk { offset: slot.offset, data });
                    }
                    x if x == FREE + 10 => {
                        log::error!("snapshot chunk readback failed");
                        self.readback_failed = true;
                        slot.state.store(FREE, Ordering::Release);
                        s.delivered += slot.count;
                    }
                    _ => {}
                }
            }
            if s.delivered >= self.np as u64 {
                writer.send(Job::SnapEnd { a: s.a });
                self.snap = None;
            }
        }
    }
    /// Any GPU readback (P(k) measurement or snapshot) still outstanding?
    pub fn readbacks_pending(&self) -> bool {
        self.snap.is_some() || self.measure.iter().any(|s| s.state.load(Ordering::Acquire) != FREE)
    }

    /// GPU two-point force test: total acceleration on a probe at distance r from one other
    /// particle, with and without the PP term, averaged over 8 directions and sub-cell offsets.
    /// Uses its own 2-particle buffer; call before any run (it rebinds).
    /// Returns rows (r, mean a_total/a_exact, max |a_total/a_exact - 1|, mean a_mesh/a_exact).
    pub fn force_test(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, eps: f64, radii: &[f64]) -> Vec<[f64; 4]> {
        let parts = buf(device, "force test particles", 32, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        let staging = buf(device, "force test readback", 32, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
        self.side = 2;
        self.np = 2;
        self.setup_pp(device, 2, Some(PpParams::for_mesh(self.n, eps)));
        self.bind(device, &parts);
        let p = &mut self.params;
        p.n_particles = 2;
        p.side = 2;
        p.units = (FIXED_BUDGET / 2.0) as u32;
        p.mass_norm = ((self.n as f64 * self.n as f64) / (2.0 * p.units as f64)) as f32;
        p.kick = 1.0;
        p.drift = 0.0;
        p.measure = 0;
        p.g_thresh = f32::MAX;
        let g_pp = p.g_pp as f64;
        let cell = BOX / self.n as f64;
        let mut out = Vec::new();
        const DIRS: usize = 8;
        for &r in radii {
            let exact = g_pp * r / (r * r + eps * eps);
            let (mut sum_t, mut max_dev, mut sum_m) = (0.0f64, 0.0f64, 0.0f64);
            for d in 0..DIRS {
            let mut res = [0.0f64; 3];
            for (k, pp_on) in [(1usize, 1u32), (2, 0)] {
                // probe at varying sub-cell offsets and directions
                let x0 = [(3.3 + 0.13 * d as f64) * cell + 0.123, (-7.1 + 0.29 * d as f64) * cell + 0.37];
                let ang: f64 = 0.37 + d as f64 * std::f64::consts::TAU / DIRS as f64;
                let x1 = [x0[0] + r * ang.cos(), x0[1] + r * ang.sin()];
                let mut pr = self.params;
                pr.pp_on = pp_on;
                pr.well = [x0[0] as f32, x0[1] as f32, x1[0] as f32, x1[1] as f32];
                queue.write_buffer(&self.params_a, 0, bytemuck::bytes_of(&pr));
                let mut pw = pr;
                pw.well = [0.0; 4];
                queue.write_buffer(&self.params_b, 0, bytemuck::bytes_of(&pw));
                let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("force test") });
                {
                    let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("force test place"), timestamp_writes: None });
                    cp.set_bind_group(0, self.bg_a.as_ref().unwrap(), &[]);
                    cp.set_pipeline(&self.p_test_place);
                    cp.dispatch_workgroups(1, 1, 1);
                }
                let bg_b = self.bg_b.clone().unwrap();
                self.encode_forces(None, &mut enc, &bg_b, &|_| None);
                enc.copy_buffer_to_buffer(&parts, 0, &staging, 0, 32);
                queue.submit(Some(enc.finish()));
                staging.map_async(wgpu::MapMode::Read, .., |_| {});
                let _ = device.poll(wgpu::PollType::wait_indefinitely());
                {
                    let v = staging.slice(..).get_mapped_range().expect("map");
                    let f: &[f32] = bytemuck::cast_slice(&v);
                    // particle 1 momentum = its acceleration (kick 1); component towards particle 0
                    let (gx, gy) = (f[6] as f64, f[7] as f64);
                    res[k] = -(gx * ang.cos() + gy * ang.sin());
                }
                staging.unmap();
            }
            let t = res[1] / exact;
            sum_t += t;
            max_dev = max_dev.max((t - 1.0).abs());
            sum_m += res[2] / exact;
            }
            out.push([r, sum_t / DIRS as f64, max_dev, sum_m / DIRS as f64]);
        }
        self.pp = None;
        self.bg_a = None;
        self.bg_b = None;
        self.particles = None;
        out
    }
}
/// Normalisation of a science run (see meta.json "normalization").
#[derive(Clone, Copy, Debug)]
pub struct ScienceNorm {
    pub amp: f64,
    pub a_init: f64,
    /// Lagrangian radius of a host halo of `host_particles`
    pub r_host: f64,
    pub sigma_host_a1: f64,
    pub sigma_mesh_init: f64,
    /// Δ²(k_nl, a=1) = k² P / (2π) = 1
    pub k_nl: f64,
}

/// Choose A and a_init so that (1) the linear rms δ in a 2D top-hat holding `host_particles`
/// is `sigma_host` at a = 1 (several collapsed hosts of that mass by a = 1), and (2) the rms
/// δ on the mesh at a_init is `sigma_mesh_init` (Zel'dovich ICs well before shell crossing).
pub fn science_normalization(mesh: u32, np: f64, n_index: f64, rc: f64, host_particles: f64, sigma_host: f64, sigma_mesh_init: f64) -> ScienceNorm {
    let unit = Spectrum { amp: 1.0, n: n_index, rc, wdm_alpha: 0.0 };
    let rho_p = np / (BOX * BOX);
    let r_host = (host_particles / (std::f64::consts::PI * rho_p)).sqrt();
    let s_host = sigma_tophat(|k| unit.p0(k), r_host);
    // rms on the mesh: modes from the fundamental to the mesh Nyquist (continuum)
    let kf = 2.0 * std::f64::consts::PI / BOX;
    let k_nyq = std::f64::consts::PI * mesh as f64 / BOX;
    let (l0, l1, m) = (kf.ln(), k_nyq.ln(), 20_000);
    let mut v = 0.0;
    for i in 0..m {
        let k = (l0 + (l1 - l0) * (i as f64 + 0.5) / m as f64).exp();
        v += k * k * unit.p0(k) / (2.0 * std::f64::consts::PI) * (l1 - l0) / m as f64;
    }
    let s_mesh = v.sqrt();
    let amp = (sigma_mesh_init / s_mesh).powi(2);
    let a_init = amp.sqrt() * s_host / sigma_host;
    let spec = Spectrum { amp, n: n_index, rc, wdm_alpha: 0.0 };
    let delta2 = |k: f64| k * k * spec.p0(k) / (2.0 * std::f64::consts::PI) / (a_init * a_init);
    let (mut lo, mut hi) = (kf, k_nyq);
    let k_nl = if delta2(lo) >= 1.0 {
        lo
    } else if delta2(hi) < 1.0 {
        hi
    } else {
        for _ in 0..100 {
            let mid = (lo * hi).sqrt();
            if delta2(mid) < 1.0 { lo = mid } else { hi = mid }
        }
        (lo * hi).sqrt()
    };
    ScienceNorm { amp, a_init, r_host, sigma_host_a1: sigma_host, sigma_mesh_init, k_nl }
}
/// σ of a 2D top-hat of radius R summed over the modes that exist in the periodic box
/// (|k| >= 2π/L, mesh modes only, Nyquist row/column excluded), for spectrum `p` evaluated
/// at a = 1 via `growth2` = (1/a_init)². Unlike `sigma_tophat` (continuum from k = 0) this is
/// the variance actually realised in the simulation.
pub fn sigma_tophat_in_box(p: impl Fn(f64) -> f64, r: f64, mesh: usize, growth2: f64) -> f64 {
    let kf = 2.0 * std::f64::consts::PI / BOX;
    let m = |i: usize| if i >= mesh / 2 { i as f64 - mesh as f64 } else { i as f64 };
    let mut s = 0.0;
    for iy in 0..mesh {
        if iy == mesh / 2 {
            continue;
        }
        for ix in 0..mesh {
            if ix == mesh / 2 || (ix == 0 && iy == 0) {
                continue;
            }
            let k = kf * (m(ix) * m(ix) + m(iy) * m(iy)).sqrt();
            let x = k * r;
            let w = 2.0 * bessel_j1(x) / x;
            s += p(k) * growth2 / (BOX * BOX) * w * w;
        }
    }
    s.sqrt()
}
/// Steps between P(k) measurements (in ln a) and the largest step allowed per frame.
pub const MEASURE_DLNA: f64 = 0.05;
pub const MAX_DLNA: f64 = 0.01;
/// Smallest step: positions are f32, so very small steps (high frame rates) lose the
/// displacement increment to rounding (measured: 20k steps biased the lowest-k growth by
/// -7%). Real time is accumulated until a step of at least this size is due.
pub const MIN_DLNA: f64 = 0.004;

/// Science-mode (uncapped) step control: Δt = sqrt(2 η ε / A_max) with the comoving
/// acceleration A = |g| / a³ (dp/dt = g/a, dx/dt = p/a²), Δln a = H Δt = a^(-3/2) Δt,
/// clamped to [min_dlna, MAX_DLNA] and allowed to grow by at most 25% per step.
#[derive(Clone, Copy, Debug)]
pub struct Cfl {
    pub eps: f64,
    pub eta: f64,
    pub min_dlna: f64,
}

/// Scale-factor schedule. Real-time mode: uniform in ln a over `run_seconds`, MIN_DLNA..MAX_DLNA
/// per step (real time accumulates until a step is due). Science mode: one step per frame,
/// adaptive CFL. Steps are clamped to land exactly on snapshot epochs and a_final.
pub struct RunSchedule {
    pub a_init: f64,
    pub a_final: f64,
    pub run_seconds: f64,
    pub a: f64,
    a_half: f64,
    next_measure_lna: f64,
    snaps: Vec<f64>,
    pub finished: bool,
    pub steps: u64,
    acc_dl: f64,
    cfl: Option<Cfl>,
    last_dl: f64,
    /// statistics of the CFL criterion
    pub floor_steps: u64,
    pending: Option<(CosmoStep, f64, f64, bool)>,
}

impl RunSchedule {
    pub fn new(a_init: f64, a_final: f64, run_seconds: f64, snapshot_as: &[f64], cfl: Option<Cfl>) -> Self {
        let mut snaps: Vec<f64> = snapshot_as.iter().copied().filter(|&s| s > a_init && s <= a_final + 1e-12).collect();
        snaps.sort_by(|x, y| x.total_cmp(y));
        snaps.dedup_by(|x, y| (*x - *y).abs() < 1e-12);
        Self {
            a_init,
            a_final,
            run_seconds,
            a: a_init,
            a_half: a_init,
            next_measure_lna: a_init.ln(),
            snaps,
            finished: false,
            steps: 0,
            acc_dl: 0.0,
            cfl,
            last_dl: 1e-3,
            floor_steps: 0,
            pending: None,
        }
    }

    /// Continue from a synchronized state at `a` (snapshot restart): P(k) is measured again at
    /// `a` and snapshot epochs <= a are dropped.
    pub fn resume_at(&mut self, a: f64) {
        self.a = a;
        self.a_half = a;
        self.next_measure_lna = a.ln();
        self.snaps.retain(|&t| t > a * (1.0 + 1e-6));
    }

    pub fn snapshot_epochs(&self) -> &[f64] {
        &self.snaps
    }

    pub fn progress(&self) -> f64 {
        ((self.a / self.a_init).ln() / (self.a_final / self.a_init).ln()).clamp(0.0, 1.0)
    }

    /// Plan the next step (None = no step this frame).
    /// `stats`: latest GPU maxima (science mode); `snapshot_busy`: a snapshot readback is
    /// still in flight (a snapshot step must wait for it).
    pub fn plan(&mut self, real_dt: f64, stats: Option<StepStats>, snapshot_busy: bool) -> Option<CosmoStep> {
        if self.finished {
            return None;
        }
        let a = self.a;
        let snapshot = self.snaps.first().is_some_and(|&t| (t - a).abs() <= 1e-9 * t);
        if snapshot && snapshot_busy {
            self.pending = None;
            return None;
        }
        let final_step = a >= self.a_final * (1.0 - 1e-12);
        let measure = final_step || snapshot || a.ln() >= self.next_measure_lna - 1e-12;
        let next_target = self.snaps.iter().copied().find(|&t| t > a * (1.0 + 1e-12)).unwrap_or(self.a_final).min(self.a_final);
        let to_target = (next_target / a).ln().max(0.0);
        let mut g_thresh = f64::MAX;
        let step = if final_step {
            CosmoStep {
                a,
                kick1: kick_factor(self.a_half, a),
                kick2: 0.0,
                drift: 0.0,
                measure,
                snapshot,
                well: [0.0; 4],
                well_kick: 0.0,
                g_thresh,
            }
        } else {
            let dl = if let Some(cfl) = self.cfl {
                let want = match stats {
                    Some(s) if s.g_max > 0.0 => {
                        let acc = s.g_max / (a * a * a);
                        let dt = (2.0 * cfl.eta * cfl.eps / acc).sqrt();
                        dt / (a * a.sqrt()) * H0
                    }
                    _ => cfl.min_dlna.max(1e-3),
                };
                let mut dl = want.min(self.last_dl * 1.25).min(MAX_DLNA);
                if dl < cfl.min_dlna {
                    dl = cfl.min_dlna;
                    self.floor_steps += 1;
                }
                // particles whose own CFL step is shorter than this one: |g| > a³ 2ηε / Δt²
                let dt = dl * a * a.sqrt() / H0;
                g_thresh = a * a * a * 2.0 * cfl.eta * cfl.eps / (dt * dt);
                dl
            } else {
                let rate = (self.a_final / self.a_init).ln() / self.run_seconds;
                self.acc_dl += rate * real_dt.max(0.0);
                if self.acc_dl < MIN_DLNA.min(to_target) && !snapshot && !measure {
                    self.pending = None;
                    return None;
                }
                let dl = self.acc_dl.clamp(MIN_DLNA.min(to_target), MAX_DLNA);
                self.acc_dl = 0.0;
                dl
            };
            let dl = dl.min(to_target.max(1e-12));
            self.last_dl = dl.max(1e-6);
            let a_next = (a * dl.exp()).min(next_target);
            let a_half_next = (a * a_next).sqrt();
            let (kick1, kick2) = if snapshot {
                (kick_factor(self.a_half, a), kick_factor(a, a_half_next))
            } else {
                (kick_factor(self.a_half, a_half_next), 0.0)
            };
            let s = CosmoStep {
                a,
                kick1,
                kick2,
                drift: drift_factor(a, a_next),
                measure,
                snapshot,
                well: [0.0; 4],
                well_kick: 0.0,
                g_thresh,
            };
            self.pending = Some((s, a_next, a_half_next, false));
            return Some(s);
        };
        self.pending = Some((step, a, a, true));
        Some(step)
    }

    /// Apply the planned step if the GPU executed it.
    pub fn commit(&mut self, ran: bool, measured: bool, snapped: bool) {
        let Some((step, a_next, a_half_next, final_step)) = self.pending.take() else {
            return;
        };
        if !ran {
            return;
        }
        self.steps += 1;
        if step.measure && measured {
            self.next_measure_lna = step.a.ln() + MEASURE_DLNA;
        }
        if step.snapshot {
            if !snapped {
                log::warn!("snapshot at a = {:.4} skipped (previous snapshot still in flight)", step.a);
            }
            self.snaps.remove(0);
        }
        if final_step {
            self.finished = true;
        } else {
            self.a = a_next;
            self.a_half = a_half_next;
        }
    }
}

/// Snapshot epochs of a science run: 16 values log-spaced from 0.1 to 1.0 (both included).
pub fn science_snapshots(a_final: f64) -> Vec<f64> {
    (0..16)
        .map(|i| if i == 15 { 1.0 } else { 0.1 * 10f64.powf(i as f64 / 15.0) })
        .filter(|&a| a <= a_final + 1e-12)
        .collect()
}
#[cfg(test)]
mod tests {
    use super::*;

    /// Exact test of the integrator + Poisson coefficient: a 1D plane wave in the comoving
    /// frame. For planar symmetry the Poisson equation integrates exactly to g = C (x - q)
    /// (mass to the left of a sheet is fixed), so Zel'dovich is exact until shell crossing and
    /// the growing mode must follow x - q = (a / a_init) Ψ for any amplitude.
    #[test]
    fn linear_mode_grows_as_a() {
        for &dlna in &[0.01f64, 0.003] {
            let (a_init, a_final, psi) = (0.02f64, 1.0f64, 1.0f64);
            let q = 0.0;
            let mut x = q + psi;
            let mut p = H0 * a_init.sqrt() * psi;
            let mut a = a_init;
            let mut a_half = a_init;
            while a < a_final - 1e-12 {
                let a_next = (a * dlna.exp()).min(a_final);
                let a_half_next = (a * a_next).sqrt();
                let g = POISSON_C * (x - q);
                p += g * kick_factor(a_half, a_half_next);
                x += p * drift_factor(a, a_next);
                a = a_next;
                a_half = a_half_next;
            }
            let want = (a_final / a_init) * psi;
            let err = ((x - q) - want).abs() / want;
            eprintln!("dln a = {dlna}: growth {:.6} vs {:.6}, rel err {:.2e}", x - q, want, err);
            assert!(err < 2e-3, "growth error {err}");
        }
    }

    #[test]
    fn schedule_hits_epochs() {
        let mut s = RunSchedule::new(0.02, 1.0, 90.0, &SNAPSHOT_AS, None);
        assert_eq!(s.snapshot_epochs(), &SNAPSHOT_AS);
        let mut snaps = Vec::new();
        let mut measures = 0;
        let mut frames = 0;
        while !s.finished && frames < 1_000_000 {
            frames += 1;
            let Some(st) = s.plan(1.0 / 60.0, None, false) else { continue };
            if st.snapshot {
                snaps.push(st.a);
            }
            measures += st.measure as usize;
            s.commit(true, true, true);
        }
        assert!(s.steps > 900 && s.steps < 1200, "steps {}", s.steps);
        assert!(s.finished && (s.a - 1.0).abs() < 1e-12);
        assert_eq!(snaps.len(), 6);
        for (got, want) in snaps.iter().zip(SNAPSHOT_AS) {
            assert!((got - want).abs() < 1e-12);
        }
        // every 0.05 in ln a (+ snapshot epochs): ln(50)/0.05 ~ 78
        assert!(measures >= 78, "{measures}");
        // total drift factor = D(a_init, a_final) exactly (steps telescope)
    }

    #[test]
    fn factors_match_quadrature() {
        let (a1, a2) = (0.1f64, 0.3f64);
        let m = 100_000;
        let (mut k, mut d) = (0.0, 0.0);
        for i in 0..m {
            let a = a1 + (a2 - a1) * (i as f64 + 0.5) / m as f64;
            let dt_da = a.sqrt() / H0; // dt = a^(1/2) da / H0
            k += dt_da / a * (a2 - a1) / m as f64;
            d += dt_da / (a * a) * (a2 - a1) / m as f64;
        }
        assert!((k - kick_factor(a1, a2)).abs() < 1e-8);
        assert!((d - drift_factor(a1, a2)).abs() < 1e-8);
    }

    #[test]
    fn mode_table_normalisation() {
        let t = ModeTable::build(128, -1.0, 1.5 * BOX / 128.0, 0.07, None, 0.0);
        // Σ |δ_k|² over included modes = σ²
        let mut var = 0.0;
        let kf = 2.0 * std::f64::consts::PI / BOX;
        let m = |i: usize| if i >= 64 { i as i64 - 128 } else { i as i64 };
        for iy in 0..128 {
            for ix in 0..128 {
                if ix == 64 || iy == 64 || (ix == 0 && iy == 0) {
                    continue;
                }
                let k = kf * ((m(ix) * m(ix) + m(iy) * m(iy)) as f64).sqrt();
                var += t.spectrum.p0(k) / (BOX * BOX);
            }
        }
        assert!((var.sqrt() - 0.07).abs() < 1e-9);
        assert!(t.count.iter().sum::<u64>() > 10_000);
        assert!(t.count[0] >= 4);
    }
    // ---------------------------------------------------------------- P³M force split (CPU)

    use crate::pm::cpu::{C as Cx, fft2};

    /// CPU mirror of the GPU long-range mesh force (CIC deposit, Gaussian split, W² deconvolution,
    /// spectral gradient, CIC interpolation). Returns the acceleration grid (ax, ay) per cell.
    fn mesh_accel_grid(n: usize, pts: &[[f64; 2]], np_norm: f64, r_s: f64) -> Vec<[f64; 2]> {
        let h = BOX / n as f64;
        let w = BOX / 2.0;
        let mut g = vec![Cx::default(); n * n];
        let per = (n * n) as f64 / np_norm;
        for p in pts {
            let gx = (p[0] + w) / h - 0.5;
            let gy = (p[1] + w) / h - 0.5;
            let (bx, by) = (gx.floor(), gy.floor());
            let (fx, fy) = (gx - bx, gy - by);
            for (dx, dy, wt) in [(0, 0, (1.0 - fx) * (1.0 - fy)), (1, 0, fx * (1.0 - fy)), (0, 1, (1.0 - fx) * fy), (1, 1, fx * fy)] {
                let ix = (bx as i64 + dx).rem_euclid(n as i64) as usize;
                let iy = (by as i64 + dy).rem_euclid(n as i64) as usize;
                g[iy * n + ix].re += wt * per;
            }
        }
        fft2(&mut g, n, false);
        let kf = 2.0 * std::f64::consts::PI / BOX;
        let m = |i: usize| if i >= n / 2 { i as f64 - n as f64 } else { i as f64 };
        let sinc = |x: f64| if x.abs() < 1e-8 { 1.0 } else { x.sin() / x };
        for iy in 0..n {
            for ix in 0..n {
                let v = &mut g[iy * n + ix];
                let (kx, ky) = (kf * m(ix), kf * m(iy));
                let k2 = kx * kx + ky * ky;
                if k2 == 0.0 || ix == n / 2 || iy == n / 2 {
                    *v = Cx::default();
                    continue;
                }
                let d = Cx { re: v.re / (n * n) as f64, im: v.im / (n * n) as f64 };
                let wc = (sinc(0.5 * kx * h) * sinc(0.5 * ky * h)).powi(2);
                let s = (-k2 * r_s * r_s).exp();
                let f = -POISSON_C * s / (k2 * wc * wc);
                let phi = Cx { re: d.re * f, im: d.im * f };
                *v = Cx { re: phi.re * ky + phi.im * kx, im: phi.im * ky - phi.re * kx };
            }
        }
        fft2(&mut g, n, true);
        g.iter().map(|c| [c.re, c.im]).collect()
    }

    fn interp(grid: &[[f64; 2]], n: usize, x: [f64; 2]) -> [f64; 2] {
        let h = BOX / n as f64;
        let w = BOX / 2.0;
        let gx = (x[0] + w) / h - 0.5;
        let gy = (x[1] + w) / h - 0.5;
        let (bx, by) = (gx.floor(), gy.floor());
        let (fx, fy) = (gx - bx, gy - by);
        let mut a = [0.0; 2];
        for (dx, dy, wt) in [(0, 0, (1.0 - fx) * (1.0 - fy)), (1, 0, fx * (1.0 - fy)), (0, 1, (1.0 - fx) * fy), (1, 1, fx * fy)] {
            let ix = (bx as i64 + dx).rem_euclid(n as i64) as usize;
            let iy = (by as i64 + dy).rem_euclid(n as i64) as usize;
            a[0] += wt * grid[iy * n + ix][0];
            a[1] += wt * grid[iy * n + ix][1];
        }
        a
    }

    fn pp_force(g_pp: f64, pp: PpParams, xi: [f64; 2], xj: [f64; 2]) -> [f64; 2] {
        let d = [xj[0] - xi[0], xj[1] - xi[1]];
        let r2 = d[0] * d[0] + d[1] * d[1];
        if r2 >= pp.r_cut * pp.r_cut || r2 == 0.0 {
            return [0.0; 2];
        }
        let f = g_pp * (-r2 / (4.0 * pp.r_s * pp.r_s)).exp() / (r2 + pp.eps * pp.eps);
        [d[0] * f, d[1] * f]
    }

    fn exact(g_pp: f64, eps: f64, xi: [f64; 2], xj: [f64; 2]) -> [f64; 2] {
        let d = [xj[0] - xi[0], xj[1] - xi[1]];
        let f = g_pp / (d[0] * d[0] + d[1] * d[1] + eps * eps);
        [d[0] * f, d[1] * f]
    }

    /// Total (mesh + PP) force vs the exact softened point-mass force: point mass and a random set.
    #[test]
    fn force_split_matches_exact() {
        let n = 512;
        let h = BOX / n as f64;
        let eps = 0.1 * h;
        let pp = PpParams::for_mesh(n as u32, eps);
        // --- point mass (np_norm = 1)
        let x0 = [0.37 * h, -0.21 * h];
        let g_pp = POISSON_C * BOX * BOX / (2.0 * std::f64::consts::PI);
        let grid = mesh_accel_grid(n, &[x0], 1.0, pp.r_s);
        let mut worst = 0.0f64;
        let mut lines = Vec::new();
        for (i, rc) in [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0, 16.0].iter().enumerate() {
            let r = rc * h;
            let ang = 0.3 + 0.7 * i as f64;
            let x = [x0[0] + r * ang.cos(), x0[1] + r * ang.sin()];
            let am = interp(&grid, n, x);
            let ap = pp_force(g_pp, pp, x, x0);
            let ex = exact(g_pp, eps, x, x0);
            let ex_mag = (ex[0] * ex[0] + ex[1] * ex[1]).sqrt();
            let tot = [am[0] + ap[0], am[1] + ap[1]];
            let err = ((tot[0] - ex[0]).powi(2) + (tot[1] - ex[1]).powi(2)).sqrt() / ex_mag;
            let mesh_only = (am[0] * ex[0] + am[1] * ex[1]) / (ex_mag * ex_mag);
            lines.push(format!("r = {rc:5.2} cells: |total-exact|/exact = {:.3}%, mesh-only/exact = {mesh_only:.3}", err * 100.0));
            if r > 2.0 * eps {
                worst = worst.max(err);
            }
        }
        eprintln!("point mass (mesh {n}, r_s = 1.25 cells, r_cut = 4.5 r_s, eps = 0.1 cell):\n  {}", lines.join("\n  "));
        assert!(worst < 0.0125, "point-mass worst error {worst}");

        // --- random set of 300 particles in a 12-cell patch (np_norm = 300)
        let mut s: u64 = 12345;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let pts: Vec<[f64; 2]> = (0..300).map(|_| [(rnd() - 0.5) * 12.0 * h, (rnd() - 0.5) * 12.0 * h]).collect();
        let g_pp = POISSON_C * BOX * BOX / (2.0 * std::f64::consts::PI * 300.0);
        let grid = mesh_accel_grid(n, &pts, 300.0, pp.r_s);
        let (mut num, mut den, mut worst_rel) = (0.0f64, 0.0f64, 0.0f64);
        for (i, &xi) in pts.iter().enumerate() {
            let am = interp(&grid, n, xi);
            let (mut ex, mut ap) = ([0.0; 2], [0.0; 2]);
            for (j, &xj) in pts.iter().enumerate() {
                if i != j {
                    let e = exact(g_pp, eps, xi, xj);
                    let p = pp_force(g_pp, pp, xi, xj);
                    ex = [ex[0] + e[0], ex[1] + e[1]];
                    ap = [ap[0] + p[0], ap[1] + p[1]];
                }
            }
            // the mesh includes the particle's own (CIC) self-force, which is ~0
            let d2 = (am[0] + ap[0] - ex[0]).powi(2) + (am[1] + ap[1] - ex[1]).powi(2);
            let e2 = ex[0] * ex[0] + ex[1] * ex[1];
            num += d2;
            den += e2;
            worst_rel = worst_rel.max((d2 / e2).sqrt());
        }
        let rms = (num / den).sqrt();
        eprintln!("random set (300 particles): rms relative error {:.3}%, worst particle {:.3}%", rms * 100.0, worst_rel * 100.0);
        assert!(rms < 0.01, "random-set rms error {rms}");
    }

    #[test]
    fn wdm_half_mode() {
        for k_hm in [0.3, 1.0, 4.0] {
            let a = wdm_alpha_for_khm(k_hm);
            let t = wdm_transfer(a, k_hm);
            assert!((t * t - 0.25).abs() < 1e-12, "T^2(k_hm) = {}", t * t);
        }
        assert!((bessel_j1(1.0) - 0.4400505857).abs() < 1e-7);
        assert!((bessel_j1(5.0) + 0.3275791376).abs() < 1e-7);
    }
    #[test]
    fn print_science_normalizations() {
        let np = 8192f64 * 8192.0;
        for n in [-1.5, -1.25, -1.0] {
            for rc_cells in [0.35, 0.5] {
                let rc = rc_cells * BOX / 4096.0;
                for hs in [1.0, 1.15, 1.3] {
                let s = science_normalization(4096, np, n, rc, 1e6, hs, 0.1);
                eprintln!("n {n:5.2} rc {rc_cells} sigma_host {hs}: A {:.4e} a_init {:.4} R_host {:.1} k_nl(a=1)/k_f {:.2}", s.amp, s.a_init, s.r_host, s.k_nl / (2.0 * std::f64::consts::PI / BOX));
                }
            }
        }
    }
}