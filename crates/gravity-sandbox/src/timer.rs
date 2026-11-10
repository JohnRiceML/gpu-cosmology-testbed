//! GPU pass timing with timestamp queries (when the adapter supports TIMESTAMP_QUERY).

use crate::readback::ReadbackRing;

pub const STAGES: usize = 8;
pub const ST_SCENE: usize = 6;
pub const ST_POST: usize = 7;
pub const STAGE_NAMES: [&str; STAGES] = [
    "deposit",
    "fft rows fwd",
    "fft cols+green+inv",
    "fft rows inv",
    "gradient / P3M short range",
    "interp+integrate",
    "render scene",
    "post (bloom+tonemap)",
];

pub struct GpuTimer {
    qs: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    ring: ReadbackRing,
    period_ns: f64,
    /// Smoothed milliseconds per stage (0 when the stage did not run).
    pub ms: [f32; STAGES],
}

impl GpuTimer {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Option<Self> {
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return None;
        }
        let count = (STAGES * 2) as u32;
        let size = count as u64 * 8;
        Some(Self {
            qs: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("stage timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count,
            }),
            resolve: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("timestamp resolve"),
                size,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            ring: ReadbackRing::new(device, "timestamp readback", size, 3),
            period_ns: queue.get_timestamp_period() as f64,
            ms: [0.0; STAGES],
        })
    }

    pub fn compute(&self, stage: usize) -> wgpu::ComputePassTimestampWrites<'_> {
        wgpu::ComputePassTimestampWrites {
            query_set: &self.qs,
            beginning_of_pass_write_index: Some(2 * stage as u32),
            end_of_pass_write_index: Some(2 * stage as u32 + 1),
        }
    }

    pub fn render(&self, stage: usize, begin: bool, end: bool) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        if !begin && !end {
            return None;
        }
        Some(wgpu::RenderPassTimestampWrites {
            query_set: &self.qs,
            beginning_of_pass_write_index: begin.then_some(2 * stage as u32),
            end_of_pass_write_index: end.then_some(2 * stage as u32 + 1),
        })
    }

    /// Record the resolve + copy at the end of the frame's encoder.
    pub fn resolve(&mut self, enc: &mut wgpu::CommandEncoder) {
        enc.resolve_query_set(&self.qs, 0..(STAGES * 2) as u32, &self.resolve, 0);
        if let Some(staging) = self.ring.begin() {
            enc.copy_buffer_to_buffer(&self.resolve, 0, staging, 0, (STAGES * 16) as u64);
        }
    }

    pub fn after_submit(&mut self) {
        self.ring.after_submit();
    }

    /// Fold the newest results into the smoothed per-stage times. `active[s]` says whether
    /// stage s ran in recent frames (stale query values are ignored otherwise).
    pub fn update(&mut self, active: &[bool; STAGES]) {
        let period = self.period_ns;
        let ms = &mut self.ms;
        self.ring.read_latest(|bytes| {
            let t: &[u64] = bytemuck::cast_slice(bytes);
            for s in 0..STAGES {
                if !active[s] {
                    ms[s] = 0.0;
                    continue;
                }
                let (a, b) = (t[2 * s], t[2 * s + 1]);
                if b > a && a != 0 {
                    let v = ((b - a) as f64 * period * 1e-6) as f32;
                    ms[s] = if ms[s] == 0.0 { v } else { 0.9 * ms[s] + 0.1 * v };
                }
            }
        });
    }

    pub fn pm_total(&self) -> f32 {
        self.ms[..6].iter().sum()
    }
}
