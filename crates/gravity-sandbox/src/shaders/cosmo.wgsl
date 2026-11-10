// Periodic 2D particle-mesh cosmology in comoving coordinates (CosmicWeb scenario).
// See src/cosmo.rs for the equations (EdS expansion, canonical momentum, D(a) = a).
//
// Mesh: N x N cells over the periodic box [-W, W)^2, L = 2W, cell h = L/N, cell centres at
// -W + (i + 0.5) h. FFT length P = N (no padding: periodic). Conventions:
//   delta_k = (1/N^2) sum_x delta(x) e^{-i k.x},  delta(x) = sum_k delta_k e^{+i k.x},
//   P(k) = L^2 <|delta_k|^2>,  k = (2 pi / L) * (m with m in [-N/2, N/2)).
// Our FFT is unnormalized in both directions, so a forward transform gives N^2 delta_k and
// an inverse transform of delta_k gives delta(x).
//
// {{P}}, {{LOGP}}, {{PT}} are substituted by the host; shaders/fft.wgsl is appended.

const P: u32 = {{P}}u;
const LOGP: u32 = {{LOGP}}u;
const WG: u32 = 256u;
const PT: u32 = {{PT}}u;
const PI: f32 = 3.14159265358979;

// Particles store their comoving DISPLACEMENT s from their lattice point q(i) (not the
// absolute position): early on the per-step increments are only a few f32 ulps of a
// coordinate ~1000, and the dead zone of round-to-nearest measurably damped the largest
// modes. |s| is small early, so it keeps full precision. x = wrap(q(i) + s).
struct Particle {
    pos: vec2<f32>, // displacement s = x - q (wrapped into [-W, W))
    vel: vec2<f32>, // canonical momentum p = a^2 dx/dt (code units, H0 = 1)
};

struct CosmoParams {
    n_particles: u32,
    side: u32,        // particle lattice side (n_particles = side^2)
    units: u32,       // fixed-point deposit units per particle
    measure: u32,     // 1 = write |delta_k|^2 / W_cic^2 into `power` this step
    seed: u32,
    pp_on: u32,       // 1 = add the short-range particle-particle force (P3M)
    chain_n: u32,     // chaining-mesh cells per side
    pp_base: u32,     // workgroup offset of a pp_force dispatch chunk
    box_l: f32,
    world_half: f32,
    cell: f32,
    inv_cell: f32,
    kick: f32,        // integral dt/a   over this kick (H0 = 1)
    drift: f32,       // integral dt/a^2 over this drift
    poisson_c: f32,   // laplacian(phi) = C delta, C = 3/2 H0^2  =>  D(a) = a
    smooth_r: f32,    // long-range Gaussian split / smoothing scale r_s (world units)
    mass_norm: f32,   // fixed-point units -> rho / rho_mean
    disp_init: f32,   // displacement factor at a_init (+-1: Psi is the a_init displacement)
    mom_init: f32,    // p = mom_init * Psi at a_init  (= +-sqrt(a_init))
    spec_a: f32,      // P0(k) = A k^n exp(-(k R_c)^2) at a_init (CDM)
    spec_n: f32,
    spec_rc: f32,
    well_kick: f32,   // momentum kick per unit well acceleration this step (non-science)
    wdm_alpha: f32,   // WDM transfer T(k) = [1 + (alpha k)^(2 nu)]^(-5/nu); 0 = CDM
    wdm_nu: f32,
    chain_cell: f32,  // chaining-mesh cell size (>= r_cut)
    inv_chain_cell: f32,
    r_cut: f32,       // short-range cutoff radius
    eps: f32,         // Plummer-like softening of the short-range force
    g_pp: f32,        // point-mass coupling C L^2 / (2 pi N_p): force = g_pp / r
    g_thresh: f32,    // particles with |g| above this violate the step's CFL (counted)
    _f2: f32,
    well: vec4<f32>,  // mouse well (x, y, strength, softening); strength 0 = off
};
@group(0) @binding(0) var<uniform> cp: CosmoParams;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(2) var<storage, read_write> dens: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> grid: array<vec2<f32>>;   // N*N complex
@group(0) @binding(4) var<storage, read> tw: array<vec2<f32>>;           // N/2 twiddles
@group(0) @binding(5) var<storage, read_write> accel: array<vec2<f32>>;  // N*N acceleration
@group(0) @binding(6) var<storage, read_write> power: array<f32>;        // N*N |delta_k|^2/W^2
// P3M short range: chaining mesh (counting sort by cell) and per-particle PP acceleration
@group(0) @binding(7) var<storage, read_write> chain_count: array<atomic<u32>>; // Nc*Nc
@group(0) @binding(8) var<storage, read_write> chain_off: array<u32>;           // Nc*Nc exclusive scan
@group(0) @binding(9) var<storage, read_write> block_sum: array<u32>;           // scan block totals
@group(0) @binding(10) var<storage, read_write> cell_rank: array<vec2<u32>>;    // per particle (cell, rank)
@group(0) @binding(11) var<storage, read_write> sorted_pos: array<vec2<f32>>;   // positions in cell order
@group(0) @binding(12) var<storage, read_write> sorted_idx: array<u32>;         // particle id in cell order
@group(0) @binding(13) var<storage, read_write> acc_pp: array<vec2<f32>>;       // per particle (by id)
@group(0) @binding(14) var<storage, read_write> stats: array<atomic<u32>, 4>;   // max|g| bits, max|p| bits, CFL count

// ------------------------------------------------------------------ helpers

fn linear_index(gid: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return gid.y * (nwg.x * 256u) + gid.x;
}

fn wrapi(i: i32) -> u32 {
    let n = i32(P);
    return u32(((i % n) + n) % n);
}

fn lattice_q(i: u32) -> vec2<f32> {
    let spacing = cp.box_l / f32(cp.side);
    return (vec2<f32>(f32(i % cp.side), f32(i / cp.side)) + 0.5) * spacing - vec2<f32>(cp.world_half);
}

// Minimum-image wrap of a displacement into [-W, W).
fn wrap_disp(s: vec2<f32>) -> vec2<f32> {
    return s - cp.box_l * floor((s + vec2<f32>(cp.world_half)) / cp.box_l);
}

fn wrap_pos(x: vec2<f32>) -> vec2<f32> {
    let l = cp.box_l;
    let w = cp.world_half;
    var y = x + vec2<f32>(w);
    y = y - l * floor(y / l);
    // floor() rounding can leave y == l
    y = select(y, vec2<f32>(0.0), y >= vec2<f32>(l));
    return y - vec2<f32>(w);
}

// Signed mode index and wave vector of FFT bin (ix, iy).
fn mode(i: u32) -> i32 {
    let s = i32(i);
    return select(s, s - i32(P), i >= P / 2u);
}

fn kvec(ix: u32, iy: u32) -> vec2<f32> {
    let kf = 2.0 * PI / cp.box_l;
    return kf * vec2<f32>(f32(mode(ix)), f32(mode(iy)));
}

fn sinc(x: f32) -> f32 {
    if (abs(x) < 1e-4) {
        return 1.0;
    }
    return sin(x) / x;
}

// CIC assignment window W(k) = prod_i sinc^2(k_i h / 2)
fn w_cic(k: vec2<f32>) -> f32 {
    let s = sinc(0.5 * k.x * cp.cell) * sinc(0.5 * k.y * cp.cell);
    return s * s;
}

fn nyquist(ix: u32, iy: u32) -> bool {
    return ix == P / 2u || iy == P / 2u;
}

fn mesh_coord(p: vec2<f32>) -> vec2<f32> {
    return (p + vec2<f32>(cp.world_half)) * cp.inv_cell - vec2<f32>(0.5);
}

fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand01(s: ptr<function, u32>) -> f32 {
    *s = pcg(*s);
    return (f32(*s >> 8u) + 0.5) * (1.0 / 16777216.0);
}

// Load row/column `line` of the grid into the workgroup arrays in bit-reversed order.
fn load_row(y: u32, lid: u32) {
    for (var x = lid; x < P; x = x + WG) {
        let v = grid[y * P + x];
        let r = brev(x);
        sre[r] = v.x;
        sim[r] = v.y;
    }
    workgroupBarrier();
}

fn load_col(kx: u32, lid: u32) {
    for (var y = lid; y < P; y = y + WG) {
        let v = grid[y * P + kx];
        let r = brev(y);
        sre[r] = v.x;
        sim[r] = v.y;
    }
    workgroupBarrier();
}

fn store_row(y: u32, lid: u32) {
    for (var x = lid; x < P; x = x + WG) {
        grid[y * P + x] = vec2<f32>(sre[x], sim[x]);
    }
}

fn store_col(kx: u32, lid: u32) {
    for (var y = lid; y < P; y = y + WG) {
        grid[y * P + kx] = vec2<f32>(sre[y], sim[y]);
    }
}

// ------------------------------------------------------------------ generic row FFTs

@compute @workgroup_size(256)
fn rows_fwd(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    load_row(wid.x, lid);
    fft_shared(lid, false);
    store_row(wid.x, lid);
}

@compute @workgroup_size(256)
fn rows_inv(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    load_row(wid.x, lid);
    fft_shared(lid, true);
    store_row(wid.x, lid);
}

// ------------------------------------------------------------------ initial conditions

// Unit Gaussian white noise (real), reproducible from the run seed.
@compute @workgroup_size(16, 16)
fn ic_noise(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= P || gid.y >= P) {
        return;
    }
    let i = gid.y * P + gid.x;
    var s: u32 = pcg(i ^ pcg(cp.seed * 0x9E3779B9u + 0x85EBCA6Bu));
    let u1 = rand01(&s);
    let u2 = rand01(&s);
    grid[i] = vec2<f32>(sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2), 0.0);
}

// Column pass of the IC transform: forward column FFT of the (row-transformed) noise, then
// per mode: keep only its phase (fixed amplitude |delta_k| = sqrt(P0/L^2), no cosmic
// variance), build the Zel'dovich displacement Psi_k = i k delta_k / k^2 packed as
// Psi_x,k + i Psi_y,k = (i kx - ky) delta_k / k^2, pre-deconvolved by the CIC window of the
// later mesh -> particle interpolation, and inverse-transform the column.
// Hermitian symmetry is inherited from the real white noise; k = 0 and the Nyquist row and
// column (where i k breaks Hermitian symmetry) are zeroed.
@compute @workgroup_size(256)
fn ic_cols(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let kx = wid.x;
    load_col(kx, lid);
    fft_shared(lid, false);

    var tmp: array<vec2<f32>, PT>;
    for (var j = 0u; j < PT; j = j + 1u) {
        let ky = lid + j * WG;
        let k = kvec(kx, ky);
        let k2 = dot(k, k);
        var out = vec2<f32>(0.0, 0.0);
        if (k2 > 0.0 && !nyquist(kx, ky)) {
            let w = vec2<f32>(sre[ky], sim[ky]);
            let wl = length(w);
            var ph = vec2<f32>(1.0, 0.0);
            if (wl > 0.0) {
                ph = w / wl;
            }
            let kk = sqrt(k2);
            var p0 = cp.spec_a * pow(kk, cp.spec_n) * exp(-(kk * cp.spec_rc) * (kk * cp.spec_rc));
            if (cp.wdm_alpha > 0.0) {
                // same phases, P_WDM = T^2 P_CDM (Viel et al. 2005)
                let t = pow(1.0 + pow(cp.wdm_alpha * kk, 2.0 * cp.wdm_nu), -5.0 / cp.wdm_nu);
                p0 = p0 * t * t;
            }
            let amp = sqrt(p0) / cp.box_l / w_cic(k);
            let d = ph * amp; // delta_k
            // (i kx - ky) * d / k^2 = (-kx d.im - ky d.re) + i (kx d.re - ky d.im)
            out = vec2<f32>(-k.x * d.y - k.y * d.x, k.x * d.x - k.y * d.y) / k2;
        }
        tmp[j] = out;
    }
    workgroupBarrier();
    for (var j = 0u; j < PT; j = j + 1u) {
        let r = brev(lid + j * WG);
        sre[r] = tmp[j].x;
        sim[r] = tmp[j].y;
    }
    workgroupBarrier();
    fft_shared(lid, true);
    store_col(kx, lid);
}

// Particles on a side x side lattice, displaced by the (interpolated) Zel'dovich field.
@compute @workgroup_size(256)
fn ic_particles(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = linear_index(gid, nwg);
    if (i >= cp.n_particles) {
        return;
    }
    let q = lattice_q(i);
    let psi = sample_grid(q);
    var p: Particle;
    p.pos = wrap_disp(cp.disp_init * psi);
    p.vel = cp.mom_init * psi;
    particles[i] = p;
}

fn sample_grid(x: vec2<f32>) -> vec2<f32> {
    let g = mesh_coord(x);
    let b = floor(g);
    let f = g - b;
    let x0 = wrapi(i32(b.x));
    let y0 = wrapi(i32(b.y));
    let x1 = (x0 + 1u) % P;
    let y1 = (y0 + 1u) % P;
    return grid[y0 * P + x0] * ((1.0 - f.x) * (1.0 - f.y))
        + grid[y0 * P + x1] * (f.x * (1.0 - f.y))
        + grid[y1 * P + x0] * ((1.0 - f.x) * f.y)
        + grid[y1 * P + x1] * (f.x * f.y);
}

// ------------------------------------------------------------------ PM step

fn add_mass(ix: i32, iy: i32, w: u32) {
    if (w != 0u) {
        atomicAdd(&dens[wrapi(iy) * P + wrapi(ix)], w);
    }
}

@compute @workgroup_size(256)
fn deposit(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = linear_index(gid, nwg);
    if (i >= cp.n_particles) {
        return;
    }
    let g = mesh_coord(wrap_pos(lattice_q(i) + particles[i].pos));
    let b = floor(g);
    let f = g - b;
    let ix = i32(b.x);
    let iy = i32(b.y);
    let s = f32(cp.units);
    add_mass(ix, iy, u32(round((1.0 - f.x) * (1.0 - f.y) * s)));
    add_mass(ix + 1, iy, u32(round(f.x * (1.0 - f.y) * s)));
    add_mass(ix, iy + 1, u32(round((1.0 - f.x) * f.y * s)));
    add_mass(ix + 1, iy + 1, u32(round(f.x * f.y * s)));
}

// Forward row FFT of rho / rho_mean (reading the fixed-point grid also clears it).
@compute @workgroup_size(256)
fn rows_density(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let y = wid.x;
    for (var x = lid; x < P; x = x + WG) {
        let v = f32(atomicExchange(&dens[y * P + x], 0u)) * cp.mass_norm;
        let r = brev(x);
        sre[r] = v;
        sim[r] = 0.0;
    }
    workgroupBarrier();
    fft_shared(lid, false);
    store_row(y, lid);
}

// Column pass: forward column FFT -> delta_k = F / N^2 -> (optional) power measurement ->
// acceleration a_k = -i k phi_k with phi_k = -C delta_k S(k) / (k^2 W^2), packed as
// a_x,k + i a_y,k = phi_k (ky - i kx) -> inverse column FFT.
@compute @workgroup_size(256)
fn cols_poisson(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let kx = wid.x;
    load_col(kx, lid);
    fft_shared(lid, false);

    let inv_n2 = 1.0 / (f32(P) * f32(P));
    var tmp: array<vec2<f32>, PT>;
    for (var j = 0u; j < PT; j = j + 1u) {
        let ky = lid + j * WG;
        let k = kvec(kx, ky);
        let k2 = dot(k, k);
        let d = vec2<f32>(sre[ky], sim[ky]) * inv_n2;
        let w = w_cic(k);
        if (cp.measure != 0u) {
            power[ky * P + kx] = dot(d, d) / (w * w);
        }
        var out = vec2<f32>(0.0, 0.0);
        if (k2 > 0.0 && !nyquist(kx, ky)) {
            let sm = exp(-k2 * cp.smooth_r * cp.smooth_r);
            let phi = d * (-cp.poisson_c * sm / (k2 * w * w));
            // phi * (ky - i kx)
            out = vec2<f32>(phi.x * k.y + phi.y * k.x, phi.y * k.y - phi.x * k.x);
        }
        tmp[j] = out;
    }
    workgroupBarrier();
    for (var j = 0u; j < PT; j = j + 1u) {
        let r = brev(lid + j * WG);
        sre[r] = tmp[j].x;
        sim[r] = tmp[j].y;
    }
    workgroupBarrier();
    fft_shared(lid, true);
    store_col(kx, lid);
}

// Inverse row FFT -> acceleration field (real part a_x, imaginary part a_y).
@compute @workgroup_size(256)
fn rows_accel(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let y = wid.x;
    load_row(y, lid);
    fft_shared(lid, true);
    for (var x = lid; x < P; x = x + WG) {
        accel[y * P + x] = vec2<f32>(sre[x], sim[x]);
    }
}

fn accel_at(x: vec2<f32>) -> vec2<f32> {
    let g = mesh_coord(x);
    let b = floor(g);
    let f = g - b;
    let x0 = wrapi(i32(b.x));
    let y0 = wrapi(i32(b.y));
    let x1 = (x0 + 1u) % P;
    let y1 = (y0 + 1u) % P;
    return accel[y0 * P + x0] * ((1.0 - f.x) * (1.0 - f.y))
        + accel[y0 * P + x1] * (f.x * (1.0 - f.y))
        + accel[y1 * P + x0] * ((1.0 - f.x) * f.y)
        + accel[y1 * P + x1] * (f.x * f.y);
}

// Comoving KDK leapfrog step (merged kicks): p += a(x) * kick; x += p * drift; wrap.
// With drift = 0 this is a pure (synchronizing) kick, used before snapshots.
@compute @workgroup_size(256)
fn integrate(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = linear_index(gid, nwg);
    if (i >= cp.n_particles) {
        return;
    }
    var p = particles[i];
    let x = wrap_pos(lattice_q(i) + p.pos);
    var g = accel_at(x);
    if (cp.pp_on != 0u) {
        g = g + acc_pp[i];
    }
    p.vel = p.vel + g * cp.kick;
    let gm = length(g);
    atomicMax(&stats[0], bitcast<u32>(gm));
    if (gm > cp.g_thresh) {
        atomicAdd(&stats[2], 1u);
    }
    if (cp.well.z != 0.0) {
        // Mouse well (breaks the science run): minimum-image distance in the periodic box.
        var d = cp.well.xy - x;
        d = d - cp.box_l * round(d / cp.box_l);
        let s = cp.well.w;
        let inv = inverseSqrt(dot(d, d) + s * s);
        p.vel = p.vel + cp.well.z * d * (inv * inv * inv) * cp.well_kick;
    }
    atomicMax(&stats[1], bitcast<u32>(length(p.vel)));
    p.pos = wrap_disp(p.pos + p.vel * cp.drift);
    if (!(abs(p.pos.x) <= cp.world_half && abs(p.pos.y) <= cp.world_half && abs(p.vel.x) < 1e9 && abs(p.vel.y) < 1e9)) {
        p.pos = vec2<f32>(0.0, 0.0);
        p.vel = vec2<f32>(0.0, 0.0);
    }
    particles[i] = p;
}

// ------------------------------------------------------------------ P3M short-range force
//
// Force split (2D Poisson kernel). A particle of mass m = L^2/N_p (rho_mean = 1) has the
// point-mass force g(r) = g_pp / r towards it, g_pp = C L^2 / (2 pi N_p). The mesh carries the
// long-range part phi_k exp(-k^2 r_s^2): in real space that is the point mass smeared by a
// Gaussian of variance 2 r_s^2, whose enclosed mass fraction in 2D is 1 - exp(-r^2/(4 r_s^2)).
// The short-range complement added here is therefore
//     g_sr(r) = g_pp r_vec / (r^2 + eps^2) * exp(-r^2 / (4 r_s^2)),   r < r_cut,
// (Plummer-like softening eps), truncated at r_cut = 4.5 r_s where exp(-5.06) = 0.6%.

fn chain_cell_of(x: vec2<f32>) -> vec2<u32> {
    let n = cp.chain_n;
    let c = vec2<u32>((x + vec2<f32>(cp.world_half)) * cp.inv_chain_cell);
    return min(c, vec2<u32>(n - 1u));
}

@compute @workgroup_size(256)
fn pp_clear(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = linear_index(gid, nwg);
    if (i < cp.chain_n * cp.chain_n) {
        atomicStore(&chain_count[i], 0u);
    }
}

@compute @workgroup_size(256)
fn pp_count(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = linear_index(gid, nwg);
    if (i >= cp.n_particles) {
        return;
    }
    let x = wrap_pos(lattice_q(i) + particles[i].pos);
    let c = chain_cell_of(x);
    let cell = c.y * cp.chain_n + c.x;
    cell_rank[i] = vec2<u32>(cell, atomicAdd(&chain_count[cell], 1u));
}

// Exclusive prefix sum of chain_count in blocks of 1024 (256 threads x 4).
var<workgroup> scan_tmp: array<u32, 256>;

fn block_scan(lid: u32, v: u32) -> u32 {
    // inclusive Hillis-Steele scan of one value per thread; returns the exclusive prefix
    scan_tmp[lid] = v;
    workgroupBarrier();
    for (var off = 1u; off < 256u; off = off * 2u) {
        var add = 0u;
        if (lid >= off) {
            add = scan_tmp[lid - off];
        }
        workgroupBarrier();
        scan_tmp[lid] = scan_tmp[lid] + add;
        workgroupBarrier();
    }
    return scan_tmp[lid] - v;
}

@compute @workgroup_size(256)
fn scan_blocks(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let total = cp.chain_n * cp.chain_n;
    let base = (wid.y * 65535u + wid.x) * 1024u + lid * 4u;
    var v: array<u32, 4>;
    var s = 0u;
    for (var j = 0u; j < 4u; j = j + 1u) {
        var x = 0u;
        if (base + j < total) {
            x = atomicLoad(&chain_count[base + j]);
        }
        v[j] = x;
        s = s + x;
    }
    let ex = block_scan(lid, s);
    var run = ex;
    for (var j = 0u; j < 4u; j = j + 1u) {
        if (base + j < total) {
            chain_off[base + j] = run;
        }
        run = run + v[j];
    }
    if (lid == 255u) {
        block_sum[wid.y * 65535u + wid.x] = ex + s;
    }
}

// Exclusive scan of the (<= 1024) block sums in one workgroup.
@compute @workgroup_size(256)
fn scan_sums(@builtin(local_invocation_index) lid: u32) {
    let nb = (cp.chain_n * cp.chain_n + 1023u) / 1024u;
    var v: array<u32, 4>;
    var s = 0u;
    for (var j = 0u; j < 4u; j = j + 1u) {
        let k = lid * 4u + j;
        var x = 0u;
        if (k < nb) {
            x = block_sum[k];
        }
        v[j] = x;
        s = s + x;
    }
    let ex = block_scan(lid, s);
    var run = ex;
    for (var j = 0u; j < 4u; j = j + 1u) {
        let k = lid * 4u + j;
        if (k < nb) {
            block_sum[k] = run;
        }
        run = run + v[j];
    }
}

@compute @workgroup_size(256)
fn scan_add(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = linear_index(gid, nwg);
    if (i < cp.chain_n * cp.chain_n) {
        chain_off[i] = chain_off[i] + block_sum[i / 1024u];
    }
}

@compute @workgroup_size(256)
fn pp_scatter(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = linear_index(gid, nwg);
    if (i >= cp.n_particles) {
        return;
    }
    let cr = cell_rank[i];
    let slot = chain_off[cr.x] + cr.y;
    sorted_pos[slot] = wrap_pos(lattice_q(i) + particles[i].pos);
    sorted_idx[slot] = i;
}

// Deterministic order inside each chaining cell. pp_count's atomic ranks depend on thread
// scheduling, so the first scatter puts a cell's particles in a run-dependent order, and the
// f32 force sums in pp_force then round differently from run to run. Here every slot
// replaces its particle's rank by the stable rank = number of particles of the same cell with
// a lower index (O(m^2) per cell of m particles, cheap next to the force); a second
// pp_scatter then writes every cell in ascending particle index, independent of scheduling.
@compute @workgroup_size(256)
fn pp_rank(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let t = linear_index(gid, nwg);
    if (t >= cp.n_particles) {
        return;
    }
    let i = sorted_idx[t];
    let cell = cell_rank[i].x;
    let start = chain_off[cell];
    let count = atomicLoad(&chain_count[cell]);
    var r = 0u;
    for (var j = 0u; j < count; j = j + 1u) {
        r = r + select(0u, 1u, sorted_idx[start + j] < i);
    }
    cell_rank[i] = vec2<u32>(cell, r);
}

fn pp_pair(xi: vec2<f32>, xj: vec2<f32>) -> vec2<f32> {
    var d = xj - xi;
    d = d - cp.box_l * round(d / cp.box_l); // minimum image
    let r2 = dot(d, d);
    if (r2 >= cp.r_cut * cp.r_cut || r2 == 0.0) {
        return vec2<f32>(0.0, 0.0);
    }
    let w = exp(-r2 / (4.0 * cp.smooth_r * cp.smooth_r));
    return d * (cp.g_pp * w / (r2 + cp.eps * cp.eps));
}

var<workgroup> tile: array<vec2<f32>, 256>;
var<workgroup> wg_cells: vec4<u32>;
var<workgroup> wg_range: vec2<u32>;

fn pp_pair_direct(xi: vec2<f32>, xj: vec2<f32>, rc2: f32, inv4rs2: f32, eps2: f32) -> vec2<f32> {
    let d = xj - xi;
    let r2 = dot(d, d);
    if (r2 >= rc2 || r2 == 0.0) {
        return vec2<f32>(0.0, 0.0);
    }
    return d * (cp.g_pp * exp(-r2 * inv4rs2) / (r2 + eps2));
}

// One thread per particle in cell order (workgroup = 256 consecutive sorted particles).
// Cells are sorted row-major, so when the workgroup's particles all lie in one row of
// chaining cells cx0..cx1 (no x wrap), every particle's 3x3 neighbourhood is contained in
// three CONTIGUOUS ranges of the sorted arrays: rows cy-1, cy, cy+1, columns cx0-1..cx1+1.
// Those ranges are streamed through shared memory in tiles of 256 positions (y-wrapped rows
// pre-shifted by the box so the inner loop needs no minimum-image test). Pairs farther
// than r_cut are rejected by the distance test (cells are >= r_cut). Otherwise (row change
// inside the workgroup or an x-wrapping column), each thread walks its own 3x3 cells directly.
@compute @workgroup_size(256)
fn pp_force(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let first = (cp.pp_base + wid.y * nwg.x + wid.x) * 256u;
    let t = first + lid;
    let valid = t < cp.n_particles;
    var xi = vec2<f32>(0.0, 0.0);
    if (valid) {
        xi = sorted_pos[t];
    }
    let n = cp.chain_n;
    let rc2 = cp.r_cut * cp.r_cut;
    let inv4rs2 = 1.0 / (4.0 * cp.smooth_r * cp.smooth_r);
    let eps2 = cp.eps * cp.eps;
    if (lid == 0u) {
        let last = min(first + 255u, cp.n_particles - 1u);
        let c0 = chain_cell_of(sorted_pos[first]);
        let c1 = chain_cell_of(sorted_pos[last]);
        wg_cells = vec4<u32>(c0.x, c0.y, c1.x, c1.y);
    }
    let wc = workgroupUniformLoad(&wg_cells);
    var acc = vec2<f32>(0.0, 0.0);
    let tiled = wc.y == wc.w && wc.x >= 1u && wc.z + 2u <= n && wc.z >= wc.x && wc.z - wc.x <= 8u;

    if (tiled) {
        for (var dy = 0u; dy < 3u; dy = dy + 1u) {
            let row = (wc.y + n + dy - 1u) % n;
            var shift = 0.0;
            if (wc.y == 0u && dy == 0u) {
                shift = -cp.box_l;
            }
            if (wc.y == n - 1u && dy == 2u) {
                shift = cp.box_l;
            }
            if (lid == 0u) {
                let c_lo = row * n + wc.x - 1u;
                let c_hi = row * n + wc.z + 1u;
                let start = chain_off[c_lo];
                let end = chain_off[c_hi] + atomicLoad(&chain_count[c_hi]);
                wg_range = vec2<u32>(start, end - start);
            }
            let rng = workgroupUniformLoad(&wg_range);
            workgroupBarrier(); // nobody may overwrite wg_range before all have read it
            for (var b = 0u; b < rng.y; b = b + 256u) {
                if (b + lid < rng.y) {
                    tile[lid] = sorted_pos[rng.x + b + lid] + vec2<f32>(0.0, shift);
                }
                workgroupBarrier();
                let m = min(256u, rng.y - b);
                if (valid) {
                    for (var j = 0u; j < m; j = j + 1u) {
                        acc = acc + pp_pair_direct(xi, tile[j], rc2, inv4rs2, eps2);
                    }
                }
                workgroupBarrier();
            }
        }
    } else if (valid) {
        let ci = chain_cell_of(xi);
        for (var dy = 0u; dy < 3u; dy = dy + 1u) {
            for (var dx = 0u; dx < 3u; dx = dx + 1u) {
                let nx = (ci.x + n + dx - 1u) % n;
                let ny = (ci.y + n + dy - 1u) % n;
                let cell = ny * n + nx;
                let start = chain_off[cell];
                let count = atomicLoad(&chain_count[cell]);
                for (var j = 0u; j < count; j = j + 1u) {
                    acc = acc + pp_pair(xi, sorted_pos[start + j]);
                }
            }
        }
    }
    if (valid) {
        acc_pp[sorted_idx[t]] = acc;
    }
}
// GPU two-point force test (Cosmo::force_test): n_particles = 2 on a 2x2 lattice; particle 0
// is placed at well.xy and particle 1 at well.zw (absolute positions), velocities zeroed.
@compute @workgroup_size(1)
fn test_place() {
    let x0 = cp.well.xy;
    let x1 = cp.well.zw;
    var p: Particle;
    p.vel = vec2<f32>(0.0, 0.0);
    p.pos = wrap_disp(x0 - lattice_q(0u));
    particles[0] = p;
    p.pos = wrap_disp(x1 - lattice_q(1u));
    particles[1] = p;
}