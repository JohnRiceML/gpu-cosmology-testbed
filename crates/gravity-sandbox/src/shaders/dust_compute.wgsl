// GPU dust: one-way gravity from CPU-exported attractors.
// Particle state lives only in `particles`.

struct Particle {
    pos: vec2<f32>,
    vel: vec2<f32>,
};

const MAX_ATTRACTORS: u32 = 64u;

struct Params {
    dt: f32,
    damping: f32,
    world_half: f32,
    n_attr: u32,
    n_particles: u32,
    seed: u32,
    disc_radius: f32,
    spin: f32,
    n_bodies: u32,
    seed_mode: u32,     // 0 = spiral disc, 1 = sample around bodies
    grav_g: f32,
    total_weight: f32,  // last entry of body_cdf
    dust_mass: f32,     // total self-gravitating dust mass (0 = massless dust)
    _q0: f32,
    _q1: f32,
    _q2: f32,
    // (x, y, strength, softening); strength = G * mass, negative = repel
    attr: array<vec4<f32>, 64>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;

struct Body {
    pos: vec2<f32>,
    vel: vec2<f32>,
    mass: f32,
    radius: f32,
    hue: f32,
    pad: f32,
};
@group(0) @binding(2) var<storage, read> bodies: array<Body>;
// Inclusive prefix sum of the per-body selection weight (mass^SEED_MASS_EXP).
@group(0) @binding(3) var<storage, read> body_cdf: array<f32>;

// Linear particle index for 1D or 2D dispatches (x dimension is <= 65535 groups).
fn particle_index(gid: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return gid.y * (nwg.x * 256u) + gid.x;
}

// Adaptive substepping: substep length <= 1/90 s (1 substep above 90 FPS, max 4).
const SUBSTEP_RATE: f32 = 90.0;
const MIN_SOFT: f32 = 2.0;
const MAX_SPEED: f32 = 4000.0;
const RESTITUTION: f32 = 0.5;
const PI: f32 = 3.14159265;

fn accel(p: vec2<f32>) -> vec2<f32> {
    var a = vec2<f32>(0.0, 0.0);
    let n = min(params.n_attr, MAX_ATTRACTORS);
    for (var i = 0u; i < n; i = i + 1u) {
        let at = params.attr[i];
        let d = at.xy - p;
        let s = max(at.w, MIN_SOFT);
        let r2 = dot(d, d) + s * s;
        let inv = inverseSqrt(r2);
        // a = strength * d / (|d|^2 + soft^2)^(3/2)
        a = a + at.z * d * (inv * inv * inv);
    }
    return a;
}

@compute @workgroup_size(256)
fn update(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = particle_index(gid, nwg);
    if (i >= params.n_particles) {
        return;
    }
    var p = particles[i];
    let substeps = clamp(u32(ceil(params.dt * SUBSTEP_RATE)), 1u, 4u);
    let h = params.dt / f32(substeps);
    let damp = exp(-params.damping * h);

    for (var s = 0u; s < substeps; s = s + 1u) {
        // semi-implicit (symplectic) Euler
        p.vel = (p.vel + accel(p.pos) * h) * damp;
        let sp = length(p.vel);
        if (sp > MAX_SPEED) {
            p.vel = p.vel * (MAX_SPEED / sp);
        }
        p.pos = p.pos + p.vel * h;
    }

    // Bounce off the world walls.
    let wh = params.world_half;
    if (p.pos.x > wh) {
        p.pos.x = 2.0 * wh - p.pos.x;
        p.vel.x = -abs(p.vel.x) * RESTITUTION;
    } else if (p.pos.x < -wh) {
        p.pos.x = -2.0 * wh - p.pos.x;
        p.vel.x = abs(p.vel.x) * RESTITUTION;
    }
    if (p.pos.y > wh) {
        p.pos.y = 2.0 * wh - p.pos.y;
        p.vel.y = -abs(p.vel.y) * RESTITUTION;
    } else if (p.pos.y < -wh) {
        p.pos.y = -2.0 * wh - p.pos.y;
        p.vel.y = abs(p.vel.y) * RESTITUTION;
    }
    p.pos = clamp(p.pos, vec2<f32>(-wh), vec2<f32>(wh));

    // NaN guard (NaN fails every comparison).
    if (!(abs(p.pos.x) <= wh && abs(p.pos.y) <= wh && abs(p.vel.x) < 1e9 && abs(p.vel.y) < 1e9)) {
        p.pos = vec2<f32>(0.0, 0.0);
        p.vel = vec2<f32>(0.0, 0.0);
    }

    particles[i] = p;
}

// ---------------------------------------------------------------- init

fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand01(s: ptr<function, u32>) -> f32 {
    *s = pcg(*s);
    return f32(*s >> 8u) * (1.0 / 16777216.0);
}

fn gauss(s: ptr<function, u32>) -> f32 {
    let u1 = max(rand01(s), 1e-7);
    let u2 = rand01(s);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
}


// Swirling disc with two trailing spiral arms; near-circular orbital velocities
// computed from the enclosed attractor strength so it starts roughly in equilibrium.
// Stratified so that each warp (32 consecutive particles) lands in one small patch: the
// radius quantile follows the index and arm / angle are chosen per 32-particle group.
// This keeps the PM deposit/interpolation memory-coherent.
fn seed_spiral(s: ptr<function, u32>, i: u32) -> Particle {
    let big_r = params.disc_radius;
    let q = (f32(i) + rand01(s)) / f32(params.n_particles);
    let r = big_r * (0.02 + 0.98 * pow(q, 0.85));
    var gs: u32 = pcg((i >> 5u) ^ pcg(params.seed + 7u));

    var theta: f32;
    if (rand01(&gs) < 0.6) {
        let arm = floor(rand01(&gs) * 2.0);
        theta = arm * PI - params.spin * 2.4 * log(r / (0.05 * big_r)) + gauss(&gs) * 0.3 + gauss(s) * 0.02;
    } else {
        theta = (rand01(&gs) + 0.01 * rand01(s)) * 2.0 * PI;
    }
    let dir = vec2<f32>(cos(theta), sin(theta));
    let wh = params.world_half * 0.98;
    let pos = clamp(dir * r + vec2<f32>(gauss(s), gauss(s)) * (0.01 * big_r), vec2<f32>(-wh), vec2<f32>(wh));
    let pr = max(length(pos), 1.0);

    // Enclosed (positive) attractor strength -> circular velocity.
    var enclosed = 0.0;
    var soft = MIN_SOFT;
    let n = min(params.n_attr, MAX_ATTRACTORS);
    for (var k = 0u; k < n; k = k + 1u) {
        let at = params.attr[k];
        if (at.z > 0.0) {
            let ra = length(at.xy);
            enclosed = enclosed + at.z * smoothstep(-0.1 * pr, 0.1 * pr, pr - ra);
            soft = max(soft, at.w);
        }
    }
    // Self-gravitating dust: add the enclosed dust mass of the seeded disc profile.
    let fr = clamp((pr / big_r - 0.02) / 0.98, 0.0, 1.0);
    enclosed = enclosed + params.grav_g * params.dust_mass * pow(fr, 1.0 / 0.85);
    let a = enclosed * pr / pow(pr * pr + soft * soft, 1.5);
    let v = sqrt(max(a * pr, 0.0));
    let tangent = vec2<f32>(-pos.y, pos.x) / pr * params.spin;
    let jitter = vec2<f32>(gauss(s), gauss(s)) * (0.05 * v);

    var p: Particle;
    p.pos = pos;
    p.vel = tangent * v + jitter;
    return p;
}

// First body index whose inclusive prefix weight exceeds `t` (binary search).
fn find_body(t: f32) -> u32 {
    var lo = 0u;
    var hi = params.n_bodies - 1u;
    while (lo < hi) {
        let mid = (lo + hi) / 2u;
        if (body_cdf[mid] > t) {
            hi = mid;
        } else {
            lo = mid + 1u;
        }
    }
    return lo;
}

// Scenario-agnostic: pick a body (heavier = more likely), place the particle in a
// projected-Plummer cloud around it scaled by its radius (radius ~ sqrt(mass)), and give
// it the body's velocity plus a bound orbit around the body and a small dispersion.
fn seed_bodies(s: ptr<function, u32>, i: u32) -> Particle {
    // Stratified selection: consecutive particle indices pick the same / neighbouring bodies,
    // so warps touch nearby memory in the PM deposit, interpolation and rendering.
    let pick = (f32(i) + rand01(s)) / f32(params.n_particles);
    let bi = find_body(pick * params.total_weight);
    let b = bodies[bi];

    let a = b.radius * 1.2 + 3.0;
    let u = min(rand01(s), 0.97);
    let r = b.radius * 0.4 + a * sqrt(u / (1.0 - u));
    let theta = rand01(s) * 2.0 * PI;
    let dir = vec2<f32>(cos(theta), sin(theta));
    let wh = params.world_half * 0.98;
    let pos = clamp(b.pos + dir * r, vec2<f32>(-wh), vec2<f32>(wh));

    // Circular speed around the body (softened like the sim), random orbit sense.
    let soft = max(b.radius * 0.5, MIN_SOFT);
    // Enclosed mass = body + the part of its own dust cloud inside r (projected Plummer:
    // M(<r) = M_cloud r'^2/(r'^2+a^2), r' = r - 0.4 R). The cloud mass is this body's share of
    // the dust: dust_mass * w_b / W with the same weights as the selection CDF.
    var w_prev = 0.0;
    if (bi > 0u) {
        w_prev = body_cdf[bi - 1u];
    }
    let cloud = params.dust_mass * (body_cdf[bi] - w_prev) / max(params.total_weight, 1e-20);
    let rp = r - b.radius * 0.4;
    let gm = params.grav_g * (b.mass + cloud * rp * rp / (rp * rp + a * a));
    let vc = sqrt(max(gm * r * r / pow(r * r + soft * soft, 1.5), 0.0));
    var sense = 1.0;
    if (rand01(s) < 0.5) {
        sense = -1.0;
    }
    let tangent = vec2<f32>(-dir.y, dir.x) * sense;
    let disp = vec2<f32>(gauss(s), gauss(s)) * (0.25 * vc + 0.5);

    var p: Particle;
    p.pos = pos;
    p.vel = b.vel + tangent * (0.8 * vc) + disp;
    return p;
}

@compute @workgroup_size(256)
fn init(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = particle_index(gid, nwg);
    if (i >= params.n_particles) {
        return;
    }
    var s: u32 = pcg(i ^ pcg(params.seed * 0x9E3779B9u + 17u));
    if (params.seed_mode == 1u && params.n_bodies > 0u) {
        particles[i] = seed_bodies(&s, i);
    } else {
        particles[i] = seed_spiral(&s, i);
    }
}