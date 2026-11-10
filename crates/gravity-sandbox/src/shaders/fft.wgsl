// Shared workgroup FFT, appended to pm.wgsl and cosmo.wgsl by the host.
// The including module must define: const P (FFT length, power of two), const LOGP,
// const WG (workgroup size, 256) and a storage binding `tw: array<vec2<f32>>` holding the
// P/2 twiddles e^{-2 pi i k / P}.

var<workgroup> sre: array<f32, P>;
var<workgroup> sim: array<f32, P>;

fn brev(i: u32) -> u32 {
    return reverseBits(i) >> (32u - LOGP);
}

// In-place iterative radix-2 DIT FFT of the workgroup arrays (input must already be in
// bit-reversed order). Forward uses e^{-2 pi i k/P}; inverse is unnormalized.
fn fft_shared(lid: u32, inverse: bool) {
    for (var s = 1u; s <= LOGP; s = s + 1u) {
        let half = 1u << (s - 1u);
        let tw_stride = P >> s;
        for (var b = lid; b < P / 2u; b = b + WG) {
            let pos = b & (half - 1u);
            let i0 = ((b >> (s - 1u)) << s) + pos;
            let i1 = i0 + half;
            var w = tw[pos * tw_stride];
            if (inverse) {
                w.y = -w.y;
            }
            let xr = sre[i1];
            let xi = sim[i1];
            let tr = xr * w.x - xi * w.y;
            let ti = xr * w.y + xi * w.x;
            let ar = sre[i0];
            let ai = sim[i0];
            sre[i0] = ar + tr;
            sim[i0] = ai + ti;
            sre[i1] = ar - tr;
            sim[i1] = ai - ti;
        }
        workgroupBarrier();
    }
}

