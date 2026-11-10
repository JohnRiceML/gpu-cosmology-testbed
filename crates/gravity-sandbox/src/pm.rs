//! GPU particle-mesh (PM) gravity for the self-gravitating dust ("dark matter").
//!
//! Pipeline per frame (all WGSL compute, see shaders/pm.wgsl):
//! 1. CIC deposit of dust and body mass into two fixed-point `atomic<u32>` N x N grids.
//! 2. Forward row FFTs of the zero-padded P x P (P = 2N) complex grid, dust = re, bodies = im
//!    (only the N non-zero rows are transformed; reading the atomics also clears them).
//! 3. Per column: forward FFT, multiply by the real Green's function, inverse FFT (fused).
//! 4. Inverse row FFTs of the N physical rows -> potentials (phi_dust, phi_bodies).
//! 5. 4-point gradient -> acceleration grids (dust feels dust+bodies, bodies feel dust).
//!    The dust-only field is box-filtered to M x M and read back for the CPU bodies.
//! 6. CIC interpolation + KDK leapfrog for the dust.
//!
//! Isolated (non-periodic) boundaries via Hockney-Eastwood zero padding. The kernel is the
//! softened Plummer potential -G / sqrt(r^2 + eps^2) (3D gravity in a plane), eps ~1.25 cells.

use bytemuck::{Pod, Zeroable};
use sim_core::{Attractor, ExternalField, G, WORLD_HALF};

use crate::readback::ReadbackRing;

const WG: u32 = 256;
pub const MESH_MIN: u32 = 256;
pub const MESH_MAX: u32 = 2048;
pub const MESH_DEFAULT: u32 = 1024;
/// Field read back for the CPU bodies. Its cell (3.9 units at 512) is also the body mass
/// smoothing scale on the mesh (momentum-conserving transpose deposit), ~ sim SOFTENING.
pub const FIELD_DEFAULT: u32 = 512;
/// Softening in mesh cells.
pub const EPS_CELLS: f32 = 1.25;
/// Total fixed-point units available per grid (kept below u32::MAX with headroom).
const FIXED_BUDGET: f64 = 3.0e9;
pub const DUST_MAX_SPEED: f32 = 6000.0;

/// Stages that get GPU timestamps (indices into the timer).
pub const ST_DEPOSIT: usize = 0;
pub const ST_ROWS_FWD: usize = 1;
pub const ST_COLS: usize = 2;
pub const ST_ROWS_INV: usize = 3;
pub const ST_GRADIENT: usize = 4;
pub const ST_INTEGRATE: usize = 5;

/// Matches `PmParams` in pm.wgsl (96 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct PmParams {
    n: u32,
    m: u32,
    n_particles: u32,
    n_bodies: u32,
    dust_units: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
    cell: f32,
    inv_cell: f32,
    world_half: f32,
    dust_unit_mass: f32,
    body_scale: f32,
    inv_body_scale: f32,
    dt: f32,
    kick_dt: f32,
    eps: f32,
    grav_g: f32,
    max_speed: f32,
    restitution: f32,
    well: [f32; 4],
}

pub struct Pm {
    pub n: u32,
    pub p: u32,
    pub m: u32,
    params: PmParams,
    params_buf: wgpu::Buffer,
    dens_d: wgpu::Buffer,
    dens_b: wgpu::Buffer,
    grid: wgpu::Buffer,
    green: wgpu::Buffer,
    tw: wgpu::Buffer,
    phi: wgpu::Buffer,
    force: wgpu::Buffer,
    field: wgpu::Buffer,
    layout: wgpu::BindGroupLayout,
    bg: Option<wgpu::BindGroup>,
    deposit_dust: wgpu::ComputePipeline,
    deposit_bodies: wgpu::ComputePipeline,
    rows_fwd: wgpu::ComputePipeline,
    cols_conv: wgpu::ComputePipeline,
    rows_inv: wgpu::ComputePipeline,
    gradient: wgpu::ComputePipeline,
    downsample: wgpu::ComputePipeline,
    integrate: wgpu::ComputePipeline,
    readback: ReadbackRing,
    /// Mass of one dust particle.
    pub particle_mass: f32,
    /// Previous drift dt (0 right after seeding: first kick is a half kick).
    dt_prev: f32,
}

/// Largest power-of-two mesh <= `want` whose P = 2N row fits in workgroup memory.
pub fn fit_mesh(want: u32, limits: &wgpu::Limits) -> u32 {
    let mut n = want.clamp(MESH_MIN, MESH_MAX).next_power_of_two().min(MESH_MAX);
    // shared memory: 2 f32 arrays of P = 2N
    while n > MESH_MIN && (16 * n) as u64 > limits.max_compute_workgroup_storage_size as u64 {
        n /= 2;
    }
    // P*P complex grid (8 bytes each) must fit in one storage binding
    while n > MESH_MIN && 32 * (n as u64) * (n as u64) > limits.max_storage_buffer_binding_size {
        n /= 2;
    }
    n
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
        size,
        usage,
        mapped_at_creation: false,
    })
}

/// Workgroup grid for `count` threads at 256 per group (2D above 65535 groups).
pub fn dispatch_dims(count: u32) -> (u32, u32) {
    let groups = count.div_ceil(WG).max(1);
    if groups <= 65_535 {
        (groups, 1)
    } else {
        (65_535, groups.div_ceil(65_535))
    }
}

/// Twiddles e^{-2 pi i k / P}, k < P/2, computed in f64.
pub fn twiddles(p: u32) -> Vec<[f32; 2]> {
    (0..p / 2)
        .map(|k| {
            let a = -2.0 * std::f64::consts::PI * k as f64 / p as f64;
            [a.cos() as f32, a.sin() as f32]
        })
        .collect()
}

impl Pm {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, n: u32, m: u32) -> Self {
        let p = 2 * n;
        let logp = p.trailing_zeros();
        let m = m.clamp(1, n).next_power_of_two().min(n);
        let src = (include_str!("shaders/pm.wgsl").to_string() + include_str!("shaders/fft.wgsl"))
            .replace("{{P}}", &p.to_string())
            .replace("{{LOGP}}", &logp.to_string())
            .replace("{{PT}}", &(p / WG).to_string());
        let sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pm"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });

        use wgpu::BufferUsages as U;
        let cells = (n * n) as u64;
        let params_buf = buf(device, "pm params", std::mem::size_of::<PmParams>() as u64, U::UNIFORM | U::COPY_DST);
        let dens_d = buf(device, "pm dens dust", cells * 4, U::STORAGE | U::COPY_DST);
        let dens_b = buf(device, "pm dens bodies", cells * 4, U::STORAGE | U::COPY_DST);
        let grid = buf(device, "pm fft grid", (p as u64) * (p as u64) * 8, U::STORAGE);
        let green = buf(device, "pm green", (p as u64) * (p as u64) * 4, U::STORAGE);
        let tw = buf(device, "pm twiddles", (p as u64 / 2) * 8, U::STORAGE | U::COPY_DST);
        let phi = buf(device, "pm phi", cells * 8, U::STORAGE);
        let force = buf(device, "pm force", cells * 16, U::STORAGE | U::COPY_SRC);
        let field_size = (m * m) as u64 * 8;
        let field = buf(device, "pm field", field_size, U::STORAGE | U::COPY_SRC);
        queue.write_buffer(&tw, 0, bytemuck::cast_slice(&twiddles(p)));

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pm layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage(1, false),
                storage(2, true),
                storage(3, false),
                storage(4, false),
                storage(5, false),
                storage(6, false),
                storage(7, true),
                storage(8, false),
                storage(9, false),
                storage(10, false),
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pm pl"),
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

        let cell = 2.0 * WORLD_HALF / n as f32;
        let params = PmParams {
            n,
            m,
            n_particles: 0,
            n_bodies: 0,
            dust_units: 1,
            _p0: 0,
            _p1: 0,
            _p2: 0,
            cell,
            inv_cell: 1.0 / cell,
            world_half: WORLD_HALF,
            dust_unit_mass: 0.0,
            body_scale: 1.0,
            inv_body_scale: 1.0,
            dt: 0.0,
            kick_dt: 0.0,
            eps: EPS_CELLS * cell,
            grav_g: G,
            max_speed: DUST_MAX_SPEED,
            restitution: 0.5,
            well: [0.0; 4],
        };

        let mut pm = Self {
            n,
            p,
            m,
            params,
            params_buf,
            dens_d,
            dens_b,
            grid,
            green,
            tw,
            phi,
            force,
            field,
            layout,
            bg: None,
            deposit_dust: make("deposit_dust"),
            deposit_bodies: make("deposit_bodies"),
            rows_fwd: make("fft_rows_fwd"),
            cols_conv: make("fft_cols_conv"),
            rows_inv: make("fft_rows_inv"),
            gradient: make("gradient"),
            downsample: make("downsample"),
            integrate: make("integrate"),
            readback: ReadbackRing::new(device, "pm field readback", field_size, 3),
            particle_mass: 0.0,
            dt_prev: 0.0,
        };

        // ---- Green's function (once). Needs a bind group: use tiny dummy particle/body
        // buffers that are replaced by `bind` before the first frame.
        let dummy_p = buf(device, "pm dummy", 16, U::STORAGE);
        let dummy_b = buf(device, "pm dummy", 32, U::STORAGE);
        pm.bind(device, &dummy_p, &dummy_b);
        queue.write_buffer(&pm.params_buf, 0, bytemuck::bytes_of(&pm.params));
        let green_fill = make("green_fill");
        let rows_green = make("fft_rows_green");
        let cols_green = make("fft_cols_green");
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pm green") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("pm green"), timestamp_writes: None });
            pass.set_bind_group(0, pm.bg.as_ref().unwrap(), &[]);
            pass.set_pipeline(&green_fill);
            pass.dispatch_workgroups(p / 16, p / 16, 1);
            pass.set_pipeline(&rows_green);
            pass.dispatch_workgroups(p, 1, 1);
            pass.set_pipeline(&cols_green);
            pass.dispatch_workgroups(p, 1, 1);
        }
        queue.submit(Some(enc.finish()));
        pm.bg = None;
        pm
    }

    /// (Re)create the bind group for the current particle and body buffers.
    pub fn bind(&mut self, device: &wgpu::Device, particles: &wgpu::Buffer, bodies: &wgpu::Buffer) {
        fn b(i: u32, buf: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry {
                binding: i,
                resource: buf.as_entire_binding(),
            }
        }
        self.bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pm bg"),
            layout: &self.layout,
            entries: &[
                b(0, &self.params_buf),
                b(1, particles),
                b(2, bodies),
                b(3, &self.dens_d),
                b(4, &self.dens_b),
                b(5, &self.grid),
                b(6, &self.green),
                b(7, &self.tw),
                b(8, &self.phi),
                b(9, &self.force),
                b(10, &self.field),
            ],
        }));
    }

    /// Set the dust mass after (re)seeding: `total_mass` spread over `n_particles`.
    pub fn set_dust(&mut self, n_particles: u32, total_mass: f32) {
        self.params.n_particles = n_particles;
        self.particle_mass = total_mass / n_particles.max(1) as f32;
        let units = (FIXED_BUDGET / n_particles.max(1) as f64).clamp(16.0, 65_536.0) as u32;
        self.params.dust_units = units;
        self.params.dust_unit_mass = self.particle_mass / units as f32;
        self.dt_prev = 0.0;
    }

    /// Per-frame uniforms. `well` = the mouse-well attractor if active.
    pub fn prepare(&mut self, queue: &wgpu::Queue, n_bodies: u32, body_mass: f32, dt: f32, well: Option<Attractor>) {
        let p = &mut self.params;
        p.n_bodies = n_bodies;
        p.body_scale = (FIXED_BUDGET / body_mass.max(1e-3) as f64).min(1.0e6) as f32;
        p.inv_body_scale = 1.0 / p.body_scale;
        p.dt = dt;
        p.kick_dt = 0.5 * (self.dt_prev + dt);
        p.well = match well {
            Some(a) => [a.pos[0], a.pos[1], a.strength, a.softening.max(1.0)],
            None => [0.0; 4],
        };
        self.dt_prev = dt;
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&self.params));
    }

    /// Record the PM passes. `ts(stage)` gives optional timestamp writes per pass.
    pub fn encode<'a>(
        &mut self,
        enc: &mut wgpu::CommandEncoder,
        ts: &dyn Fn(usize) -> Option<wgpu::ComputePassTimestampWrites<'a>>,
    ) {
        let Some(bg) = self.bg.as_ref() else {
            return;
        };
        let n = self.n;
        let (gx, gy) = dispatch_dims(self.params.n_particles);
        let mut pass = |stage: usize, label: &str, f: &mut dyn FnMut(&mut wgpu::ComputePass<'_>)| {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: ts(stage),
            });
            cp.set_bind_group(0, bg, &[]);
            f(&mut cp);
        };
        pass(ST_DEPOSIT, "pm deposit", &mut |cp| {
            cp.set_pipeline(&self.deposit_dust);
            cp.dispatch_workgroups(gx, gy, 1);
            if self.params.n_bodies > 0 {
                cp.set_pipeline(&self.deposit_bodies);
                cp.dispatch_workgroups(self.params.n_bodies.div_ceil(WG), 1, 1);
            }
        });
        pass(ST_ROWS_FWD, "pm fft rows fwd", &mut |cp| {
            cp.set_pipeline(&self.rows_fwd);
            cp.dispatch_workgroups(n, 1, 1);
        });
        pass(ST_COLS, "pm fft cols + green + inverse", &mut |cp| {
            cp.set_pipeline(&self.cols_conv);
            cp.dispatch_workgroups(self.p, 1, 1);
        });
        pass(ST_ROWS_INV, "pm fft rows inv", &mut |cp| {
            cp.set_pipeline(&self.rows_inv);
            cp.dispatch_workgroups(n, 1, 1);
        });
        let m = self.m;
        pass(ST_GRADIENT, "pm gradient", &mut |cp| {
            cp.set_pipeline(&self.gradient);
            cp.dispatch_workgroups(n.div_ceil(16), n.div_ceil(16), 1);
            cp.set_pipeline(&self.downsample);
            cp.dispatch_workgroups(m.div_ceil(16), m.div_ceil(16), 1);
        });
        pass(ST_INTEGRATE, "pm integrate", &mut |cp| {
            cp.set_pipeline(&self.integrate);
            cp.dispatch_workgroups(gx, gy, 1);
        });
        if let Some(staging) = self.readback.begin() {
            enc.copy_buffer_to_buffer(&self.field, 0, staging, 0, (self.m * self.m) as u64 * 8);
        }
    }

    pub fn after_submit(&mut self) {
        self.readback.after_submit();
    }

    /// Copy the newest ready dust-only field into `out`. Returns true if updated.
    pub fn read_field(&mut self, out: &mut ExternalField) -> bool {
        let m = self.m as usize;
        self.readback.read_latest(|bytes| {
            let src: &[[f32; 2]] = bytemuck::cast_slice(bytes);
            out.size = m;
            out.acc.clear();
            out.acc.extend_from_slice(&src[..m * m]);
        })
    }

    /// GPU validation: a point mass on the mesh vs. the analytic softened Plummer force.
    /// Blocks on the GPU (startup/debug only). Returns a human-readable report and the
    /// worst relative error at r >= 3 cells.
    pub fn self_test(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> (String, f32) {
        let n = self.n;
        let c = n / 2 + 3; // test mass cell (dust channel)
        let mass_units: u32 = 1_000_000;
        // A body mass elsewhere must NOT show up in the dust-only (zw) field.
        let bc = n / 4;
        let mut dd = vec![0u32; (n * n) as usize];
        let mut db = vec![0u32; (n * n) as usize];
        dd[(c * n + c) as usize] = mass_units;
        db[(bc * n + bc) as usize] = mass_units;
        queue.write_buffer(&self.dens_d, 0, bytemuck::cast_slice(&dd));
        queue.write_buffer(&self.dens_b, 0, bytemuck::cast_slice(&db));
        let saved = self.params;
        self.params.dust_unit_mass = 1.0 / mass_units as f32; // total dust mass 1
        self.params.inv_body_scale = 1.0 / mass_units as f32;
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&self.params));
        self.params = saved;

        let bytes = (n * n) as u64 * 16;
        let staging = buf(device, "pm selftest", bytes, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pm selftest") });
        {
            let bg = self.bg.as_ref().expect("bind before self_test");
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("pm selftest"), timestamp_writes: None });
            cp.set_bind_group(0, bg, &[]);
            cp.set_pipeline(&self.rows_fwd);
            cp.dispatch_workgroups(n, 1, 1);
            cp.set_pipeline(&self.cols_conv);
            cp.dispatch_workgroups(self.p, 1, 1);
            cp.set_pipeline(&self.rows_inv);
            cp.dispatch_workgroups(n, 1, 1);
            cp.set_pipeline(&self.gradient);
            cp.dispatch_workgroups(n.div_ceil(16), n.div_ceil(16), 1);
        }
        enc.copy_buffer_to_buffer(&self.force, 0, &staging, 0, bytes);
        queue.submit(Some(enc.finish()));
        staging.map_async(wgpu::MapMode::Read, .., |_| {});
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let view = staging.slice(..).get_mapped_range().expect("map");
        let f: &[[f32; 4]] = bytemuck::cast_slice(&view);

        let cell = 2.0 * WORLD_HALF / n as f32;
        let eps = EPS_CELLS * cell;
        let mut report = format!("PM self-test (mesh {n}, cell {cell:.3}, eps {eps:.3}):\n");
        let mut worst = 0.0f32;
        for r in [1u32, 2, 3, 5, 10, 20, 50, 100, 200] {
            if c + r >= n {
                continue;
            }
            let v = f[(c * n + c + r) as usize];
            let rr = r as f32 * cell;
            let analytic = G * 1.0 * rr / (rr * rr + eps * eps).powf(1.5);
            let got = -v[2]; // dust-only, pointing back towards the mass (-x)
            let err = (got - analytic).abs() / analytic;
            if r >= 3 {
                worst = worst.max(err);
            }
            report += &format!(
                "  r = {r:>3} cells: PM {got:.6e}  analytic {analytic:.6e}  err {:.3}%  (y-comp {:.1e}, total-x {:.6e})\n",
                err * 100.0,
                v[3],
                -v[0]
            );
        }
        // Body channel separation: 3 cells from the body mass, the dust-only field (zw) must
        // not contain the body's pull.
        let at_body = f[(bc * n + bc + 3) as usize];
        report += &format!(
            "  3 cells from the body mass: total |a| {:.3e}, dust-only |a| {:.3e} (expect dust-only << total)\n",
            (at_body[0].hypot(at_body[1])),
            (at_body[2].hypot(at_body[3]))
        );
        drop(view);
        staging.unmap();
        (report, worst)
    }
}

/// CPU reference used by the unit tests (mirrors the GPU algorithm exactly).
#[cfg(test)]
pub(crate) mod cpu {
    #[derive(Clone, Copy, Default, Debug)]
    pub struct C {
        pub re: f64,
        pub im: f64,
    }

    /// In-place iterative radix-2 DIT FFT (bit-reverse then butterflies), unnormalized.
    pub fn fft(x: &mut [C], inverse: bool) {
        let p = x.len();
        let logp = p.trailing_zeros();
        for i in 0..p {
            let r = i.reverse_bits() >> (usize::BITS - logp);
            if r > i {
                x.swap(i, r);
            }
        }
        let mut half = 1;
        while half < p {
            let step = p / (2 * half);
            for g in (0..p).step_by(2 * half) {
                for pos in 0..half {
                    let a = -2.0 * std::f64::consts::PI * (pos * step) as f64 / p as f64;
                    let (s, c) = a.sin_cos();
                    let w = C { re: c, im: if inverse { -s } else { s } };
                    let (i0, i1) = (g + pos, g + pos + half);
                    let t = C {
                        re: x[i1].re * w.re - x[i1].im * w.im,
                        im: x[i1].re * w.im + x[i1].im * w.re,
                    };
                    let a0 = x[i0];
                    x[i0] = C { re: a0.re + t.re, im: a0.im + t.im };
                    x[i1] = C { re: a0.re - t.re, im: a0.im - t.im };
                }
            }
            half *= 2;
        }
    }

    pub fn fft2(grid: &mut [C], p: usize, inverse: bool) {
        for y in 0..p {
            fft(&mut grid[y * p..(y + 1) * p], inverse);
        }
        let mut col = vec![C::default(); p];
        for x in 0..p {
            for y in 0..p {
                col[y] = grid[y * p + x];
            }
            fft(&mut col, inverse);
            for y in 0..p {
                grid[y * p + x] = col[y];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::cpu::*;
    use super::*;

    #[test]
    fn fft_round_trip() {
        let p = 64;
        let mut x: Vec<C> = (0..p)
            .map(|i| C { re: ((i * 37 % 11) as f64).sin(), im: (i as f64 * 0.3).cos() })
            .collect();
        let orig = x.clone();
        fft(&mut x, false);
        fft(&mut x, true);
        for (a, b) in x.iter().zip(&orig) {
            assert!((a.re / p as f64 - b.re).abs() < 1e-12);
            assert!((a.im / p as f64 - b.im).abs() < 1e-12);
        }
        // DFT check of one bin.
        let mut y = orig.clone();
        fft(&mut y, false);
        let k = 5;
        let (mut re, mut im) = (0.0, 0.0);
        for (n, v) in orig.iter().enumerate() {
            let a = -2.0 * std::f64::consts::PI * (k * n) as f64 / p as f64;
            re += v.re * a.cos() - v.im * a.sin();
            im += v.re * a.sin() + v.im * a.cos();
        }
        assert!((y[k].re - re).abs() < 1e-9 && (y[k].im - im).abs() < 1e-9);
    }

    #[test]
    fn twiddle_table_matches() {
        let t = twiddles(16);
        assert_eq!(t.len(), 8);
        assert!((t[4][0]).abs() < 1e-7 && (t[4][1] + 1.0).abs() < 1e-7); // e^{-i pi/2} = -i
    }

    /// Point mass through the full PM pipeline (padding, Plummer kernel, 4-point gradient)
    /// against the analytic softened force.
    #[test]
    fn point_mass_matches_plummer() {
        let n = 128usize;
        let p = 2 * n;
        let h = 2.0 * WORLD_HALF as f64 / n as f64;
        let eps = EPS_CELLS as f64 * h;
        let g = G as f64;

        // kernel
        let mut k = vec![C::default(); p * p];
        for y in 0..p {
            for x in 0..p {
                let dx = x.min(p - x) as f64 * h;
                let dy = y.min(p - y) as f64 * h;
                k[y * p + x].re = -g / (dx * dx + dy * dy + eps * eps).sqrt();
            }
        }
        fft2(&mut k, p, false);
        for v in &k {
            assert!(v.im.abs() < 1e-6 * (1.0 + v.re.abs()), "kernel FFT must be real");
        }

        // mass 1 at cell (c, c) (dust = re), mass 3 elsewhere in the body channel (im)
        let c = n / 2;
        let mut rho = vec![C::default(); p * p];
        rho[c * p + c].re = 1.0;
        rho[(n / 4) * p + n / 4].im = 3.0;
        fft2(&mut rho, p, false);
        for (r, kk) in rho.iter_mut().zip(&k) {
            r.re *= kk.re;
            r.im *= kk.re;
        }
        fft2(&mut rho, p, true);
        let norm = (p * p) as f64;
        let phi = |x: usize, y: usize| rho[y * p + x].re / norm; // dust channel only

        for r in [2usize, 3, 5, 10, 20, 40] {
            let x = c + r;
            let gx = (phi(x - 2, c) - 8.0 * phi(x - 1, c) + 8.0 * phi(x + 1, c) - phi(x + 2, c)) / (12.0 * h);
            let got = gx; // = -a_x; the force points to -x so -a_x > 0
            let rr = r as f64 * h;
            let analytic = g * rr / (rr * rr + eps * eps).powf(1.5);
            let err = (got - analytic).abs() / analytic;
            eprintln!("r={r} cells: rel err {:.3}%", err * 100.0);
            // 4-point FD of the softened kernel: ~8% at 2 cells, <2% from 3 cells out.
            assert!(err < if r >= 3 { 0.02 } else { 0.10 }, "r={r}: got {got}, analytic {analytic}, err {err}");
        }
        // isolated boundary: potential at distance d equals -G/sqrt(d^2+eps^2) (no images)
        let d = (n - 1 - c) as f64 * h;
        let expect = -g / (d * d + eps * eps).sqrt();
        let got = phi(n - 1, c);
        assert!(((got - expect) / expect).abs() < 1e-9, "periodic images leaked: {got} vs {expect}");
        // body channel separated into the imaginary part
        let im_at = rho[c * p + c + 5].im / norm;
        let bd = (((c + 5 - n / 4) as f64 * h).powi(2) + ((c - n / 4) as f64 * h).powi(2)).sqrt();
        assert!((im_at - (-3.0 * g / (bd * bd + eps * eps).sqrt())).abs() < 1e-9);
    }

    #[test]
    fn mesh_fits_limits() {
        let l = wgpu::Limits::default(); // 16 KiB workgroup memory, 128 MiB bindings
        assert_eq!(fit_mesh(2048, &l), 1024);
        let mut big = wgpu::Limits::default();
        big.max_compute_workgroup_storage_size = 49152;
        big.max_storage_buffer_binding_size = 1 << 30;
        assert_eq!(fit_mesh(2048, &big), 2048);
        assert_eq!(fit_mesh(1000, &big), 1024);
    }
}
