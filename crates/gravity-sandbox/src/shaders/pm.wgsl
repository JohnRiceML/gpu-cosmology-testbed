// Particle-mesh (PM) gravity: CIC deposit -> zero-padded 2D FFT (Hockney-Eastwood) ->
// multiply by the FFT of a softened Plummer kernel -> inverse FFT -> 4-point gradient ->
// CIC interpolation + KDK leapfrog for the dust.
//
// The mesh is N x N cells over [-WORLD_HALF, WORLD_HALF]^2 (cell centers at
// -W + (i + 0.5) * cell). The FFT grid is P x P with P = 2N (isolated boundaries).
// Dust mass goes in the real part and body mass in the imaginary part of one complex grid;
// the kernel is real and even, so after the inverse FFT re = phi_dust, im = phi_bodies.
//
// {{P}}, {{LOGP}}, {{PT}} are substituted by the host (P = FFT length, PT = P / WG).
// shaders/fft.wgsl (shared radix-2 FFT: workgroup arrays, brev, fft_shared) is appended.

const P: u32 = {{P}}u;
const LOGP: u32 = {{LOGP}}u;
const WG: u32 = 256u;
const PT: u32 = {{PT}}u;

struct Particle {
    pos: vec2<f32>,
    vel: vec2<f32>,
};

struct Body {
    pos: vec2<f32>,
    vel: vec2<f32>,
    mass: f32,
    radius: f32,
    hue: f32,
    pad: f32,
};

struct PmParams {
    n: u32,             // mesh cells per side
    m: u32,             // downsampled field size for the CPU
    n_particles: u32,
    n_bodies: u32,
    dust_units: u32,    // fixed-point units per dust particle
    _p0: u32,
    _p1: u32,
    _p2: u32,
    cell: f32,
    inv_cell: f32,
    world_half: f32,
    dust_unit_mass: f32, // mass of one dust fixed-point unit
    body_scale: f32,     // fixed-point units per unit of body mass
    inv_body_scale: f32,
    dt: f32,             // drift
    kick_dt: f32,        // 0.5 * (dt_prev + dt): merged KDK kicks
    eps: f32,            // Plummer softening (world units)
    grav_g: f32,
    max_speed: f32,
    restitution: f32,
    well: vec4<f32>,     // mouse well (x, y, strength, softening); strength 0 = off
};

@group(0) @binding(0) var<uniform> pm: PmParams;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(2) var<storage, read> bodies: array<Body>;
@group(0) @binding(3) var<storage, read_write> dens_d: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> dens_b: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> grid: array<vec2<f32>>;   // P*P complex
@group(0) @binding(6) var<storage, read_write> green: array<f32>;        // P*P, [kx*P + ky]
@group(0) @binding(7) var<storage, read> tw: array<vec2<f32>>;           // P/2 twiddles
@group(0) @binding(8) var<storage, read_write> phi: array<vec2<f32>>;    // N*N (dust, bodies)
@group(0) @binding(9) var<storage, read_write> force: array<vec4<f32>>;  // N*N (total.xy, dust.zw)
@group(0) @binding(10) var<storage, read_write> field: array<vec2<f32>>; // M*M dust-only accel

// ------------------------------------------------------------------ helpers

fn linear_index(gid: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return gid.y * (nwg.x * 256u) + gid.x;
}

// Mesh coordinate in which cell centers are integers.
fn mesh_coord(p: vec2<f32>) -> vec2<f32> {
    return (p + vec2<f32>(pm.world_half)) * pm.inv_cell - vec2<f32>(0.5);
}

fn in_mesh(ix: i32, iy: i32) -> bool {
    let n = i32(pm.n);
    return ix >= 0 && iy >= 0 && ix < n && iy < n;
}

fn add_dust(ix: i32, iy: i32, w: u32) {
    if (w != 0u && in_mesh(ix, iy)) {
        atomicAdd(&dens_d[u32(iy) * pm.n + u32(ix)], w);
    }
}

// ------------------------------------------------------------------ 1. deposit (CIC)

@compute @workgroup_size(256)
fn deposit_dust(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = linear_index(gid, nwg);
    if (i >= pm.n_particles) {
        return;
    }
    let g = mesh_coord(particles[i].pos);
    let base = floor(g);
    let f = g - base;
    let ix = i32(base.x);
    let iy = i32(base.y);
    let s = f32(pm.dust_units);
    add_dust(ix, iy, u32(round((1.0 - f.x) * (1.0 - f.y) * s)));
    add_dust(ix + 1, iy, u32(round(f.x * (1.0 - f.y) * s)));
    add_dust(ix, iy + 1, u32(round((1.0 - f.x) * f.y * s)));
    add_dust(ix + 1, iy + 1, u32(round(f.x * f.y * s)));
}
// Bodies feel the dust through the M x M field (box filter of b x b mesh cells, then
// bilinear on the CPU). For momentum conservation the body mass is deposited with the
// transpose of that operator: bilinear weights on the coarse grid, each coarse share spread
// uniformly over its b x b mesh cells. (b = 1 reduces to plain CIC.) The resulting body
// smoothing scale (~2 coarse cells) is comparable to the CPU body softening.
fn spread_body(cx: i32, cy: i32, units: f32) {
    let m = i32(pm.m);
    if (cx < 0 || cy < 0 || cx >= m || cy >= m || units <= 0.0) {
        return;
    }
    let b = pm.n / pm.m;
    let u = u32(round(units / f32(b * b)));
    if (u == 0u) {
        return;
    }
    for (var j = 0u; j < b; j = j + 1u) {
        let row = (u32(cy) * b + j) * pm.n + u32(cx) * b;
        for (var i = 0u; i < b; i = i + 1u) {
            atomicAdd(&dens_b[row + i], u);
        }
    }
}

@compute @workgroup_size(256)
fn deposit_bodies(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= pm.n_bodies) {
        return;
    }
    let b = bodies[i];
    let coarse_inv = f32(pm.m) / (2.0 * pm.world_half);
    let g = (b.pos + vec2<f32>(pm.world_half)) * coarse_inv - vec2<f32>(0.5);
    let base = floor(g);
    let f = g - base;
    let ix = i32(base.x);
    let iy = i32(base.y);
    let s = max(b.mass, 0.0) * pm.body_scale;
    spread_body(ix, iy, (1.0 - f.x) * (1.0 - f.y) * s);
    spread_body(ix + 1, iy, f.x * (1.0 - f.y) * s);
    spread_body(ix, iy + 1, (1.0 - f.x) * f.y * s);
    spread_body(ix + 1, iy + 1, f.x * f.y * s);
}

// ------------------------------------------------------------------ 2. forward row FFT
// One workgroup per physical row y < N. Rows N..P-1 of the padded grid are all zero and
// are never stored (the column pass treats them as zero). Reading the fixed-point mass
// also clears it for the next frame.

@compute @workgroup_size(256)
fn fft_rows_fwd(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let y = wid.x;
    let n = pm.n;
    for (var x = lid; x < P; x = x + WG) {
        var v = vec2<f32>(0.0, 0.0);
        if (x < n) {
            let c = y * n + x;
            v = vec2<f32>(
                f32(atomicExchange(&dens_d[c], 0u)) * pm.dust_unit_mass,
                f32(atomicExchange(&dens_b[c], 0u)) * pm.inv_body_scale,
            );
        }
        let r = brev(x);
        sre[r] = v.x;
        sim[r] = v.y;
    }
    workgroupBarrier();
    fft_shared(lid, false);
    for (var x = lid; x < P; x = x + WG) {
        grid[y * P + x] = vec2<f32>(sre[x], sim[x]);
    }
}

// ------------------------------------------------------------------ 3. column pass
// Fused: forward column FFT -> multiply by the (real) Green's function -> inverse column
// FFT. Only rows < N are needed afterwards, so only those are written back.

@compute @workgroup_size(256)
fn fft_cols_conv(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let kx = wid.x;
    let n = pm.n;
    for (var y = lid; y < P; y = y + WG) {
        var v = vec2<f32>(0.0, 0.0);
        if (y < n) {
            v = grid[y * P + kx];
        }
        let r = brev(y);
        sre[r] = v.x;
        sim[r] = v.y;
    }
    workgroupBarrier();
    fft_shared(lid, false);

    // Multiply and permute into bit-reversed order for the inverse transform.
    var tmp: array<vec2<f32>, PT>;
    for (var j = 0u; j < PT; j = j + 1u) {
        let k = lid + j * WG;
        let g = green[kx * P + k];
        tmp[j] = vec2<f32>(sre[k], sim[k]) * g;
    }
    workgroupBarrier();
    for (var j = 0u; j < PT; j = j + 1u) {
        let r = brev(lid + j * WG);
        sre[r] = tmp[j].x;
        sim[r] = tmp[j].y;
    }
    workgroupBarrier();
    fft_shared(lid, true);
    for (var y = lid; y < n; y = y + WG) {
        grid[y * P + kx] = vec2<f32>(sre[y], sim[y]);
    }
}

// ------------------------------------------------------------------ 4. inverse row FFT

@compute @workgroup_size(256)
fn fft_rows_inv(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let y = wid.x;
    let n = pm.n;
    for (var x = lid; x < P; x = x + WG) {
        let v = grid[y * P + x];
        let r = brev(x);
        sre[r] = v.x;
        sim[r] = v.y;
    }
    workgroupBarrier();
    fft_shared(lid, true);
    // 1/P^2 normalization is folded into `green`.
    for (var x = lid; x < n; x = x + WG) {
        phi[y * n + x] = vec2<f32>(sre[x], sim[x]);
    }
}

// ------------------------------------------------------------------ 5. gradient

fn phi_at(x: i32, y: i32) -> vec2<f32> {
    let n = i32(pm.n);
    let cx = clamp(x, 0, n - 1);
    let cy = clamp(y, 0, n - 1);
    return phi[u32(cy) * pm.n + u32(cx)];
}

// 4-point central difference: f' = (f[-2] - 8 f[-1] + 8 f[+1] - f[+2]) / (12 h)
@compute @workgroup_size(16, 16)
fn gradient(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= pm.n || gid.y >= pm.n) {
        return;
    }
    let x = i32(gid.x);
    let y = i32(gid.y);
    let k = pm.inv_cell / 12.0;
    let gx = (phi_at(x - 2, y) - 8.0 * phi_at(x - 1, y) + 8.0 * phi_at(x + 1, y) - phi_at(x + 2, y)) * k;
    let gy = (phi_at(x, y - 2) - 8.0 * phi_at(x, y - 1) + 8.0 * phi_at(x, y + 1) - phi_at(x, y + 2)) * k;
    // acceleration = -grad(phi). (.x = dust part, .y = body part)
    force[gid.y * pm.n + gid.x] = vec4<f32>(-(gx.x + gx.y), -(gy.x + gy.y), -gx.x, -gy.x);
}

// Box-filter the dust-only acceleration down to M x M for the CPU bodies.
@compute @workgroup_size(16, 16)
fn downsample(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= pm.m || gid.y >= pm.m) {
        return;
    }
    let b = pm.n / pm.m;
    var acc = vec2<f32>(0.0, 0.0);
    for (var j = 0u; j < b; j = j + 1u) {
        let row = (gid.y * b + j) * pm.n + gid.x * b;
        for (var i = 0u; i < b; i = i + 1u) {
            acc = acc + force[row + i].zw;
        }
    }
    field[gid.y * pm.m + gid.x] = acc / f32(b * b);
}

// ------------------------------------------------------------------ 6. interpolate + integrate

fn force_at(ix: i32, iy: i32) -> vec2<f32> {
    if (!in_mesh(ix, iy)) {
        return vec2<f32>(0.0, 0.0);
    }
    return force[u32(iy) * pm.n + u32(ix)].xy;
}

@compute @workgroup_size(256)
fn integrate(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = linear_index(gid, nwg);
    if (i >= pm.n_particles) {
        return;
    }
    var p = particles[i];

    // CIC interpolation with the same weights as the deposit (no self-force).
    let g = mesh_coord(p.pos);
    let base = floor(g);
    let f = g - base;
    let ix = i32(base.x);
    let iy = i32(base.y);
    var a = force_at(ix, iy) * ((1.0 - f.x) * (1.0 - f.y))
        + force_at(ix + 1, iy) * (f.x * (1.0 - f.y))
        + force_at(ix, iy + 1) * ((1.0 - f.x) * f.y)
        + force_at(ix + 1, iy + 1) * (f.x * f.y);

    // Mouse well: analytic, unsoftened by the mesh.
    if (pm.well.z != 0.0) {
        let d = pm.well.xy - p.pos;
        let s = pm.well.w;
        let inv = inverseSqrt(dot(d, d) + s * s);
        a = a + pm.well.z * d * (inv * inv * inv);
    }

    // KDK leapfrog with merged kicks: v(n+1/2) = v(n-1/2) + a(x_n) * (dt_prev + dt)/2.
    p.vel = p.vel + a * pm.kick_dt;
    let sp = length(p.vel);
    if (sp > pm.max_speed) {
        p.vel = p.vel * (pm.max_speed / sp);
    }
    p.pos = p.pos + p.vel * pm.dt;

    let wh = pm.world_half;
    if (p.pos.x > wh) {
        p.pos.x = 2.0 * wh - p.pos.x;
        p.vel.x = -abs(p.vel.x) * pm.restitution;
    } else if (p.pos.x < -wh) {
        p.pos.x = -2.0 * wh - p.pos.x;
        p.vel.x = abs(p.vel.x) * pm.restitution;
    }
    if (p.pos.y > wh) {
        p.pos.y = 2.0 * wh - p.pos.y;
        p.vel.y = -abs(p.vel.y) * pm.restitution;
    } else if (p.pos.y < -wh) {
        p.pos.y = -2.0 * wh - p.pos.y;
        p.vel.y = abs(p.vel.y) * pm.restitution;
    }
    p.pos = clamp(p.pos, vec2<f32>(-wh), vec2<f32>(wh));
    if (!(abs(p.pos.x) <= wh && abs(p.pos.y) <= wh && abs(p.vel.x) < 1e9 && abs(p.vel.y) < 1e9)) {
        p.pos = vec2<f32>(0.0, 0.0);
        p.vel = vec2<f32>(0.0, 0.0);
    }
    particles[i] = p;
}

// ------------------------------------------------------------------ Green's function
// Run once at startup: K(dx, dy) = -G / sqrt(dx^2 + dy^2 + eps^2) on the padded grid with
// wrap-around distances, forward 2D FFT, keep the real part / P^2 (stored transposed:
// green[kx*P + ky]; the kernel is symmetric so this equals the untransposed layout).

@compute @workgroup_size(16, 16)
fn green_fill(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= P || gid.y >= P) {
        return;
    }
    let dx = f32(min(gid.x, P - gid.x)) * pm.cell;
    let dy = f32(min(gid.y, P - gid.y)) * pm.cell;
    grid[gid.y * P + gid.x] = vec2<f32>(-pm.grav_g / sqrt(dx * dx + dy * dy + pm.eps * pm.eps), 0.0);
}

@compute @workgroup_size(256)
fn fft_rows_green(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let y = wid.x;
    for (var x = lid; x < P; x = x + WG) {
        let v = grid[y * P + x];
        let r = brev(x);
        sre[r] = v.x;
        sim[r] = v.y;
    }
    workgroupBarrier();
    fft_shared(lid, false);
    for (var x = lid; x < P; x = x + WG) {
        grid[y * P + x] = vec2<f32>(sre[x], sim[x]);
    }
}

@compute @workgroup_size(256)
fn fft_cols_green(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let kx = wid.x;
    for (var y = lid; y < P; y = y + WG) {
        let v = grid[y * P + kx];
        let r = brev(y);
        sre[r] = v.x;
        sim[r] = v.y;
    }
    workgroupBarrier();
    fft_shared(lid, false);
    let norm = 1.0 / (f32(P) * f32(P));
    for (var k = lid; k < P; k = k + WG) {
        green[kx * P + k] = sre[k] * norm;
    }
}
