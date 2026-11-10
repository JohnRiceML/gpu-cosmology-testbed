// Dust rendered as 1-pixel points, additively blended into the HDR target.

struct Particle {
    pos: vec2<f32>,
    vel: vec2<f32>,
};

struct Camera {
    center: vec2<f32>,
    half_extent: vec2<f32>,
    viewport: vec2<f32>,
    speed_ref: f32,
    dust_intensity: f32,
    lattice: f32,
    box_l: f32,
    _p0: f32,
    _p1: f32,
};

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec3<f32>,
};

fn speed_color(t: f32) -> vec3<f32> {
    // cool blue -> hot orange -> white
    let cold = vec3<f32>(0.18, 0.35, 1.0);
    let hot = vec3<f32>(1.0, 0.45, 0.12);
    let white = vec3<f32>(1.0, 0.95, 0.9);
    if (t < 1.0) {
        return mix(cold, hot, smoothstep(0.0, 1.0, t));
    }
    return mix(hot, white, smoothstep(1.0, 2.0, t));
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VOut {
    // Particles are stored spatially coherent (good for the PM deposit/integrate), but
    // drawing neighbours back-to-back serializes additive blending on the same pixels.
    // Draw them in a strided order instead (a bijection on [0, n)).
    // Permute 16-particle chunks (keeps vertex fetches coalesced).
    let n = arrayLength(&particles);
    let chunks = n / 16u;
    let q = chunks / 1024u;
    var idx = vi;
    let c = vi / 16u;
    if (c < q * 1024u) {
        idx = ((c % 1024u) * q + c / 1024u) * 16u + (vi % 16u);
    }
    let p = particles[idx];
    var pos = p.pos;
    if (cam.lattice > 0.0) {
        // CosmicWeb: .pos is the displacement from lattice point idx (periodic box)
        let side = u32(cam.lattice);
        let w = 0.5 * cam.box_l;
        let q = (vec2<f32>(f32(idx % side), f32(idx / side)) + 0.5) * (cam.box_l / cam.lattice) - vec2<f32>(w);
        let y = q + pos + vec2<f32>(w);
        pos = y - cam.box_l * floor(y / cam.box_l) - vec2<f32>(w);
    }
    var out: VOut;
    let ndc = (pos - cam.center) / cam.half_extent;
    out.clip = vec4<f32>(ndc, 0.0, 1.0);
    let t = clamp(length(p.vel) / cam.speed_ref, 0.0, 2.0);
    // Hot dust is only mildly brighter, so infalling cores don't blow out to white.
    out.color = speed_color(t) * cam.dust_intensity * (0.75 + 0.15 * t);
    return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
