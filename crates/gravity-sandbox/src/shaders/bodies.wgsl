// Bodies as instanced quads with an SDF circle + soft glow, additive into HDR.

struct Body {
    pos: vec2<f32>,
    vel: vec2<f32>,
    mass: f32,
    radius: f32,
    hue: f32,
    pad: f32,
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
@group(0) @binding(1) var<storage, read> bodies: array<Body>;

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) local: vec2<f32>,
    // (radius, extent, world-per-pixel)
    @location(1) @interpolate(flat) shape: vec3<f32>,
    @location(2) @interpolate(flat) color: vec3<f32>,
};

const GLOW_SCALE: f32 = 3.0;

fn hsv2rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let k = vec3<f32>(1.0, 2.0 / 3.0, 1.0 / 3.0);
    let p = abs(fract(vec3<f32>(h) + k) * 6.0 - vec3<f32>(3.0));
    return v * mix(vec3<f32>(1.0), clamp(p - vec3<f32>(1.0), vec3<f32>(0.0), vec3<f32>(1.0)), s);
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    let corner = corners[vi];
    let b = bodies[ii];
    let wpp = 2.0 * cam.half_extent.y / cam.viewport.y;
    // Keep tiny bodies visible (at least ~1 px radius).
    let r = max(b.radius, wpp);
    let ext = r * GLOW_SCALE + 2.0 * wpp;
    let world = b.pos + corner * ext;

    var out: VOut;
    out.clip = vec4<f32>((world - cam.center) / cam.half_extent, 0.0, 1.0);
    out.local = corner * ext;
    out.shape = vec3<f32>(r, ext, wpp);
    out.color = hsv2rgb(b.hue, 0.65, 1.0);
    return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    let r = in.shape.x;
    let ext = in.shape.y;
    let wpp = in.shape.z;
    let d = length(in.local);
    // Anti-aliased disc.
    let core = 1.0 - smoothstep(r - wpp, r + wpp, d);
    // Slightly hotter center (limb brightening reversed for a "star" look).
    let center = clamp(1.0 - d / r, 0.0, 1.0);
    // Soft glow that fades to exactly zero at the quad edge.
    let x = clamp((d - r) / max(ext - r, 1e-4), 0.0, 1.0);
    let glow = pow(1.0 - x, 3.0);
    let rgb = in.color * (core * (1.6 + 1.2 * center) + glow * 0.6 * (1.0 - core));
    return vec4<f32>(rgb, 1.0);
}
