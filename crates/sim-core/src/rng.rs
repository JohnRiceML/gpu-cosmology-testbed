//! Tiny deterministic PRNG (SplitMix64). Good statistical quality for gameplay
//! purposes, no dependencies, and fully reproducible from a `u64` seed.

#[derive(Clone, Debug)]
pub(crate) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub(crate) fn f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 * (1.0 / (1u32 << 24) as f32)
    }

    /// Uniform in [lo, hi).
    pub(crate) fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.f32()
    }

    /// Uniform point inside a disc of radius `r` centred on the origin.
    pub(crate) fn in_disc(&mut self, r: f32) -> [f32; 2] {
        let rr = r * self.f32().sqrt();
        let th = self.f32() * std::f32::consts::TAU;
        [rr * th.cos(), rr * th.sin()]
    }
}
