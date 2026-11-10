// Post-processing: bloom bright-pass / separable blur, and the final tonemap.
// All passes share one bind group layout:
//   0: src texture, 1: linear sampler, 2: params, 3: bloom texture (tonemap only)

struct PostParams {
    // 1 / size of the src texture
    src_texel: vec2<f32>,
    // blur direction in texels (0,0 for non-blur passes)
    dir: vec2<f32>,
    bloom_strength: f32,
    exposure: f32,
    // 1.0 if the swapchain is not sRGB and the shader must encode gamma itself
    manual_gamma: f32,
    threshold: f32,
};

@group(0) @binding(0) var t_src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> pp: PostParams;
@group(0) @binding(3) var t_bloom: texture_2d<f32>;

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Fullscreen triangle; uv has origin at top-left like texture coordinates.
@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> VOut {
    let uv = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    var out: VOut;
    out.clip = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
    out.uv = uv;
    return out;
}

// Downsample (4 bilinear taps = 4x4 box) + soft threshold.
@fragment
fn fs_bright(in: VOut) -> @location(0) vec4<f32> {
    let o = pp.src_texel;
    var c = textureSample(t_src, samp, in.uv + vec2<f32>(-o.x, -o.y)).rgb;
    c = c + textureSample(t_src, samp, in.uv + vec2<f32>(o.x, -o.y)).rgb;
    c = c + textureSample(t_src, samp, in.uv + vec2<f32>(-o.x, o.y)).rgb;
    c = c + textureSample(t_src, samp, in.uv + vec2<f32>(o.x, o.y)).rgb;
    c = c * 0.25;
    let b = max(max(c.r, c.g), c.b);
    let knee = 0.5;
    let soft = clamp(b - pp.threshold + knee, 0.0, 2.0 * knee);
    let soft2 = soft * soft / (4.0 * knee + 1e-4);
    let contrib = max(soft2, b - pp.threshold) / max(b, 1e-4);
    return vec4<f32>(c * contrib, 1.0);
}

// 9-tap Gaussian using linear-sampling offsets (5 fetches).
@fragment
fn fs_blur(in: VOut) -> @location(0) vec4<f32> {
    let stp = pp.dir * pp.src_texel;
    var c = textureSample(t_src, samp, in.uv).rgb * 0.2270270270;
    c = c + textureSample(t_src, samp, in.uv + stp * 1.3846153846).rgb * 0.3162162162;
    c = c + textureSample(t_src, samp, in.uv - stp * 1.3846153846).rgb * 0.3162162162;
    c = c + textureSample(t_src, samp, in.uv + stp * 3.2307692308).rgb * 0.0702702703;
    c = c + textureSample(t_src, samp, in.uv - stp * 3.2307692308).rgb * 0.0702702703;
    return vec4<f32>(c, 1.0);
}

// Narkowicz ACES filmic fit.
fn aces(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_tonemap(in: VOut) -> @location(0) vec4<f32> {
    let hdr = textureSample(t_src, samp, in.uv).rgb;
    let bloom = textureSample(t_bloom, samp, in.uv).rgb;
    var c = aces((hdr + bloom * pp.bloom_strength) * pp.exposure);
    if (pp.manual_gamma > 0.5) {
        c = pow(c, vec3<f32>(1.0 / 2.2));
    }
    return vec4<f32>(c, 1.0);
}
