//! wgpu renderer: GPU dust compute, HDR scene (dust + bodies), bloom and tonemap.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use sim_core::{Attractor, Body, MAX_ATTRACTORS, WORLD_HALF};
use winit::window::Window;

use crate::camera::Camera;
use crate::cosmo::{Cosmo, CosmoConfig, CosmoStep, Spectrum};
use crate::cosmo_io::RunWriter;
use crate::pm::{self, Pm};
use crate::timer::{self, GpuTimer};

pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Default number of dust particles.
pub const DUST_DEFAULT: u32 = 1_000_000;
pub const DUST_MIN: u32 = 125_000;
/// Hard cap on dust (64M x 16 B = 1 GiB of VRAM). The real max may be lower if the
/// adapter's buffer limits are smaller; see `Renderer::max_dust`.
pub const DUST_CAP: u32 = 128 * 1024 * 1024;

const WORKGROUP_SIZE: u32 = 256;
/// Max workgroups per dispatch dimension guaranteed by WebGPU/wgpu.
const MAX_GROUPS_PER_DIM: u32 = 65_535;
/// Body selection weight for body-sampled dust seeding: weight = mass^SEED_MASS_EXP.
/// 1.0 would make the dust trace the body mass exactly (every well deepened by the same
/// factor, see Game::spin_up_for_dark_matter); 0.8 is close to that but keeps the heaviest
/// cores from hoarding particles (dense cores cost atomics/ROP throughput).
const SEED_MASS_EXP: f32 = 0.8;
/// Velocity damping for dust (per second, exponential).
const DUST_DAMPING: f32 = 0.02;
const INITIAL_BODY_CAPACITY: usize = 8192;

const BLOOM_THRESHOLD: f32 = 1.0;
pub const BLOOM_STRENGTH: f32 = 0.8;
const EXPOSURE: f32 = 1.0;

/// GPU particle layout: pos, vel (16 bytes).
const PARTICLE_SIZE: u64 = 16;

/// Matches `Params` in dust_compute.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SimParams {
    dt: f32,
    damping: f32,
    world_half: f32,
    n_attr: u32,
    n_particles: u32,
    seed: u32,
    disc_radius: f32,
    spin: f32,
    n_bodies: u32,
    /// 0 = spiral disc, 1 = sample around bodies (mass-weighted CDF)
    seed_mode: u32,
    grav_g: f32,
    total_weight: f32,
    /// Total dust mass included in the seeding equilibrium (0 = massless dust).
    dust_mass: f32,
    _q0: f32,
    _q1: f32,
    _q2: f32,
    attr: [Attractor; MAX_ATTRACTORS],
}

/// How `init_dust` distributes the particles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeedMode {
    /// Two-arm spiral disc around the origin with circular velocities (Galaxy).
    Spiral = 0,
    /// Each particle picks a body (mass-weighted) and is placed around it,
    /// inheriting its velocity: works for any scenario.
    Bodies = 1,
}

/// Construction options for the renderer.
#[derive(Clone, Copy, Debug)]
pub struct RendererOptions {
    pub dust: u32,
    pub vsync: bool,
    pub bloom: bool,
    /// Particle-mesh self-gravity for the dust (None = disabled / attractor mode only).
    pub pm_mesh: Option<u32>,
    /// Size of the dust-only field read back for the CPU bodies.
    pub pm_field: u32,
}

/// Workgroup grid for `count` particles: 1D when it fits, else 2D (x = 65535).
/// Shaders compute the linear index as `gid.y * num_workgroups.x * 256 + gid.x`.
fn dispatch_dims(count: u32) -> (u32, u32) {
    let groups = count.div_ceil(WORKGROUP_SIZE).max(1);
    if groups <= MAX_GROUPS_PER_DIM {
        (groups, 1)
    } else {
        (MAX_GROUPS_PER_DIM, groups.div_ceil(MAX_GROUPS_PER_DIM))
    }
}

/// Matches `Camera` in dust_render.wgsl / bodies.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraUniform {
    center: [f32; 2],
    half_extent: [f32; 2],
    viewport: [f32; 2],
    speed_ref: f32,
    dust_intensity: f32,
    /// > 0: particle .pos holds the displacement from lattice point i of a lattice x lattice
    /// grid over the periodic box (CosmicWeb); 0: .pos is the position.
    lattice: f32,
    box_l: f32,
    _p0: f32,
    _p1: f32,
}

/// Matches `PostParams` in post.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PostParams {
    src_texel: [f32; 2],
    dir: [f32; 2],
    bloom_strength: f32,
    exposure: f32,
    manual_gamma: f32,
    threshold: f32,
}

/// Per-reset description of the galaxy used to seed the dust disc.
#[derive(Clone, Copy, Debug)]
pub struct DustSeed {
    pub seed: u32,
    pub disc_radius: f32,
    /// +1 counter-clockwise, -1 clockwise.
    pub spin: f32,
}

/// Per-frame inputs to `Renderer::render`.
pub struct FrameData<'a> {
    pub bodies: &'a [Body],
    pub attractors: &'a [Attractor; MAX_ATTRACTORS],
    pub n_attractors: usize,
    pub dt: f32,
    pub simulate_dust: bool,
    pub camera: &'a Camera,
    /// Mouse well (analytic term for the PM dust), if active.
    pub well: Option<Attractor>,
    /// Total body mass (fixed-point scale for the body deposit).
    pub body_mass: f32,
    /// Cosmology step (CosmicWeb mode only; None = hold).
    pub cosmo: Option<CosmoStep>,
}

pub enum RenderOutcome {
    Ok,
    /// Frame skipped (occluded / timeout / zero-size); try again next frame.
    Skipped,
}

struct PostTargets {
    _hdr: wgpu::Texture,
    hdr_view: wgpu::TextureView,
    _bloom_a: wgpu::Texture,
    bloom_a_view: wgpu::TextureView,
    _bloom_b: wgpu::Texture,
    bloom_b_view: wgpu::TextureView,
    bg_bright: wgpu::BindGroup,
    bg_blur_h: wgpu::BindGroup,
    bg_blur_v: wgpu::BindGroup,
    bg_tonemap: wgpu::BindGroup,
}

pub struct Renderer {
    window: Arc<Window>,
    instance: wgpu::Instance,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    uncapped_mode: Option<wgpu::PresentMode>,
    vsync: bool,
    pub adapter_info: wgpu::AdapterInfo,
    max_dust: u32,

    // dust
    params: SimParams,
    params_buf: wgpu::Buffer,
    particle_buf: wgpu::Buffer,
    particle_count: u32,
    cdf_buf: wgpu::Buffer,
    cdf_scratch: Vec<f32>,
    compute_layout: wgpu::BindGroupLayout,
    compute_bg: wgpu::BindGroup,
    update_pipeline: wgpu::ComputePipeline,
    init_pipeline: wgpu::ComputePipeline,
    speed_ref: f32,

    // scene
    camera_buf: wgpu::Buffer,
    scene_layout: wgpu::BindGroupLayout,
    dust_bg: wgpu::BindGroup,
    dust_pipeline: wgpu::RenderPipeline,
    bodies_buf: wgpu::Buffer,
    bodies_capacity: usize,
    bodies_bg: wgpu::BindGroup,
    bodies_pipeline: wgpu::RenderPipeline,

    // post
    sampler: wgpu::Sampler,
    post_layout: wgpu::BindGroupLayout,
    post_bright_buf: wgpu::Buffer,
    post_blur_h_buf: wgpu::Buffer,
    post_blur_v_buf: wgpu::Buffer,
    post_tonemap_buf: wgpu::Buffer,
    bright_pipeline: wgpu::RenderPipeline,
    blur_pipeline: wgpu::RenderPipeline,
    tonemap_pipeline: wgpu::RenderPipeline,
    targets: PostTargets,
    pub bloom: bool,

    // particle-mesh self-gravity + GPU timing
    pm: Option<Pm>,
    pm_on: bool,
    pm_ran: bool,
    timer: Option<GpuTimer>,
    stage_active: [bool; timer::STAGES],

    // periodic comoving PM (CosmicWeb)
    cosmo: Option<Cosmo>,
    cosmo_on: bool,
    cosmo_ran: bool,
    cosmo_last: (bool, bool),
    bloom_strength: f32,
    surface_errors: u64,
    gpu_errors: Arc<std::sync::atomic::AtomicU64>,
    acquiring: Arc<std::sync::atomic::AtomicBool>,
    device_lost: Arc<std::sync::atomic::AtomicBool>,
}

fn uniform_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
    read_only: bool,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn buffer_resource(buf: &wgpu::Buffer) -> wgpu::BindingResource<'_> {
    buf.as_entire_binding()
}

const ADDITIVE: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
};

impl Renderer {
    pub async fn new(window: Arc<Window>, opts: RendererOptions) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let surface = instance
            .create_surface(window.clone())
            .expect("failed to create surface");
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .expect("no suitable GPU adapter");
        let adapter_info = adapter.get_info();

        // Raise the buffer limits to what the adapter supports so the dust can go huge.
        let adapter_limits = adapter.limits();
        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffer_binding_size = adapter_limits
            .max_storage_buffer_binding_size
            .max(limits.max_storage_buffer_binding_size);
        limits.max_buffer_size = adapter_limits.max_buffer_size.max(limits.max_buffer_size);
        // PM FFT rows live in workgroup memory (2 x P floats) and bind 10 storage buffers.
        limits.max_compute_workgroup_storage_size = adapter_limits
            .max_compute_workgroup_storage_size
            .max(limits.max_compute_workgroup_storage_size);
        limits.max_storage_buffers_per_shader_stage = adapter_limits
            .max_storage_buffers_per_shader_stage
            .max(limits.max_storage_buffers_per_shader_stage);
        let features = adapter.features() & wgpu::Features::TIMESTAMP_QUERY;
        let max_bytes = limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size);
        let max_dust = (max_bytes / PARTICLE_SIZE).min(DUST_CAP as u64) as u32;
        log::info!(
            "limits: max_storage_buffer_binding_size={} max_buffer_size={} -> max dust {}",
            limits.max_storage_buffer_binding_size,
            limits.max_buffer_size,
            max_dust
        );

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("gravity-sandbox device"),
                required_features: features,
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            })
            .await
            .expect("failed to create device");

        // Count GPU errors (science runs refuse to mark themselves complete after one). Errors raised
        // while acquiring the swapchain image (minimized window etc.) are reported separately.
        let gpu_errors = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let acquiring = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let device_lost = Arc::new(std::sync::atomic::AtomicBool::new(false));
        {
            let (errs, acq) = (gpu_errors.clone(), acquiring.clone());
            device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
                if acq.load(std::sync::atomic::Ordering::Acquire) {
                    log::debug!("surface acquire error: {e}");
                    return;
                }
                errs.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
                log::error!("wgpu uncaptured error: {e}");
            }));
            let lost = device_lost.clone();
            device.set_device_lost_callback(move |reason, msg| {
                lost.store(true, std::sync::atomic::Ordering::Release);
                log::error!("GPU device lost ({reason:?}): {msg}");
            });
        }

        // ---- surface
        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let uncapped_mode = [wgpu::PresentMode::Mailbox, wgpu::PresentMode::Immediate]
            .into_iter()
            .find(|m| caps.present_modes.contains(m));
        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: match uncapped_mode {
                Some(m) if !opts.vsync => m,
                _ => wgpu::PresentMode::Fifo,
            },
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);
        let vsync = config.present_mode == wgpu::PresentMode::Fifo;
        let dust_count = opts.dust.clamp(DUST_MIN, max_dust.max(DUST_MIN));

        // ---- shaders
        let compute_sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dust_compute"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/dust_compute.wgsl").into()),
        });
        let dust_sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dust_render"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/dust_render.wgsl").into()),
        });
        let bodies_sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bodies"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/bodies.wgsl").into()),
        });
        let post_sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("post"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/post.wgsl").into()),
        });

        // ---- dust compute
        let params = SimParams {
            dt: 0.0,
            damping: DUST_DAMPING,
            world_half: WORLD_HALF,
            n_attr: 0,
            n_particles: dust_count,
            seed: 1,
            disc_radius: WORLD_HALF * 0.8,
            spin: 1.0,
            n_bodies: 0,
            seed_mode: SeedMode::Spiral as u32,
            grav_g: sim_core::G,
            total_weight: 0.0,
            dust_mass: 0.0,
            _q0: 0.0,
            _q1: 0.0,
            _q2: 0.0,
            attr: [Attractor::default(); MAX_ATTRACTORS],
        };
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim params"),
            size: std::mem::size_of::<SimParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let compute_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compute layout"),
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::COMPUTE),
                storage_entry(1, wgpu::ShaderStages::COMPUTE, false),
                storage_entry(2, wgpu::ShaderStages::COMPUTE, true), // bodies
                storage_entry(3, wgpu::ShaderStages::COMPUTE, true), // body CDF
            ],
        });
        let compute_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("compute pl"),
            bind_group_layouts: &[Some(&compute_layout)],
            immediate_size: 0,
        });
        let make_compute = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&compute_pl),
                module: &compute_sm,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let update_pipeline = make_compute("update");
        let init_pipeline = make_compute("init");

        // ---- scene (dust points + bodies)
        let camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let scene_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene layout"),
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT),
                storage_entry(1, wgpu::ShaderStages::VERTEX, true),
            ],
        });
        let scene_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scene pl"),
            bind_group_layouts: &[Some(&scene_layout)],
            immediate_size: 0,
        });
        let make_scene_pipeline = |label: &str, sm: &wgpu::ShaderModule, topo: wgpu::PrimitiveTopology| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&scene_pl),
                vertex: wgpu::VertexState {
                    module: sm,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState {
                    topology: topo,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: sm,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: HDR_FORMAT,
                        blend: Some(ADDITIVE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let dust_pipeline = make_scene_pipeline("dust", &dust_sm, wgpu::PrimitiveTopology::PointList);
        let bodies_pipeline =
            make_scene_pipeline("bodies", &bodies_sm, wgpu::PrimitiveTopology::TriangleList);

        let particle_buf = Self::create_particle_buffer(&device, dust_count);
        let dust_bg = Self::make_pair_bg(&device, &scene_layout, &camera_buf, &particle_buf, "dust bg");

        let bodies_buf = Self::create_bodies_buffer(&device, INITIAL_BODY_CAPACITY);
        let cdf_buf = Self::create_cdf_buffer(&device, INITIAL_BODY_CAPACITY);
        let bodies_bg = Self::make_pair_bg(&device, &scene_layout, &camera_buf, &bodies_buf, "bodies bg");
        let compute_bg = Self::make_compute_bg(
            &device,
            &compute_layout,
            &params_buf,
            &particle_buf,
            &bodies_buf,
            &cdf_buf,
        );

        // ---- post
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("linear clamp"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let post_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post layout"),
            entries: &[
                texture_entry(0),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                uniform_entry(2, wgpu::ShaderStages::FRAGMENT),
                texture_entry(3),
            ],
        });
        let post_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("post pl"),
            bind_group_layouts: &[Some(&post_layout)],
            immediate_size: 0,
        });
        let make_post = |label: &str, fs: &str, format: wgpu::TextureFormat| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&post_pl),
                vertex: wgpu::VertexState {
                    module: &post_sm,
                    entry_point: Some("vs_fullscreen"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &post_sm,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let bright_pipeline = make_post("bloom bright", "fs_bright", HDR_FORMAT);
        let blur_pipeline = make_post("bloom blur", "fs_blur", HDR_FORMAT);
        let tonemap_pipeline = make_post("tonemap", "fs_tonemap", config.format);

        let make_post_buf = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: std::mem::size_of::<PostParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let post_bright_buf = make_post_buf("post bright");
        let post_blur_h_buf = make_post_buf("post blur h");
        let post_blur_v_buf = make_post_buf("post blur v");
        let post_tonemap_buf = make_post_buf("post tonemap");

        let targets = Self::create_targets(
            &device,
            &post_layout,
            &sampler,
            [&post_bright_buf, &post_blur_h_buf, &post_blur_v_buf, &post_tonemap_buf],
            config.width,
            config.height,
        );

        let mut r = Self {
            window,
            instance,
            surface,
            device,
            queue,
            config,
            uncapped_mode,
            vsync,
            adapter_info,
            max_dust,
            params,
            params_buf,
            particle_buf,
            particle_count: dust_count,
            cdf_buf,
            cdf_scratch: Vec::with_capacity(INITIAL_BODY_CAPACITY),
            compute_layout,
            compute_bg,
            update_pipeline,
            init_pipeline,
            speed_ref: 100.0,
            camera_buf,
            scene_layout,
            dust_bg,
            dust_pipeline,
            bodies_buf,
            bodies_capacity: INITIAL_BODY_CAPACITY,
            bodies_bg,
            bodies_pipeline,
            sampler,
            post_layout,
            post_bright_buf,
            post_blur_h_buf,
            post_blur_v_buf,
            post_tonemap_buf,
            bright_pipeline,
            blur_pipeline,
            tonemap_pipeline,
            targets,
            bloom: opts.bloom,
            pm: None,
            pm_on: false,
            pm_ran: false,
            timer: None,
            stage_active: [false; timer::STAGES],
            cosmo: None,
            cosmo_on: false,
            cosmo_ran: false,
            cosmo_last: (false, false),
            bloom_strength: BLOOM_STRENGTH,
            surface_errors: 0,
            gpu_errors,
            acquiring,
            device_lost,
        };
        r.write_post_params();
        r.timer = GpuTimer::new(&r.device, &r.queue);
        if let Some(want) = opts.pm_mesh {
            let n = pm::fit_mesh(want, &r.device.limits());
            if n != want {
                log::warn!("PM mesh {want} not supported by this adapter; using {n}");
            }
            let mut solver = Pm::new(&r.device, &r.queue, n, opts.pm_field);
            solver.bind(&r.device, &r.particle_buf, &r.bodies_buf);
            r.pm = Some(solver);
            r.pm_on = true;
        }
        r
    }

    // ------------------------------------------------------------ resources

    fn create_particle_buffer(device: &wgpu::Device, count: u32) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dust particles"),
            size: count as u64 * PARTICLE_SIZE,
            // COPY_SRC: cosmology snapshots are read back from this buffer
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn create_bodies_buffer(device: &wgpu::Device, capacity: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bodies"),
            size: (capacity * std::mem::size_of::<Body>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn create_cdf_buffer(device: &wgpu::Device, capacity: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("body cdf"),
            size: (capacity * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn make_compute_bg(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        params: &wgpu::Buffer,
        particles: &wgpu::Buffer,
        bodies: &wgpu::Buffer,
        cdf: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compute bg"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_resource(params),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer_resource(particles),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffer_resource(bodies),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffer_resource(cdf),
                },
            ],
        })
    }

    fn rebuild_compute_bg(&mut self) {
        self.compute_bg = Self::make_compute_bg(
            &self.device,
            &self.compute_layout,
            &self.params_buf,
            &self.particle_buf,
            &self.bodies_buf,
            &self.cdf_buf,
        );
    }

    fn make_pair_bg(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        uniform: &wgpu::Buffer,
        storage: &wgpu::Buffer,
        label: &str,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_resource(uniform),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer_resource(storage),
                },
            ],
        })
    }

    fn create_color_target(
        device: &wgpu::Device,
        label: &str,
        w: u32,
        h: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: w.max(1),
                height: h.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HDR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        (tex, view)
    }

    fn create_targets(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        bufs: [&wgpu::Buffer; 4],
        width: u32,
        height: u32,
    ) -> PostTargets {
        let (hdr, hdr_view) = Self::create_color_target(device, "hdr", width, height);
        let (bw, bh) = ((width / 2).max(1), (height / 2).max(1));
        let (bloom_a, bloom_a_view) = Self::create_color_target(device, "bloom a", bw, bh);
        let (bloom_b, bloom_b_view) = Self::create_color_target(device, "bloom b", bw, bh);

        let bg = |label: &str, src: &wgpu::TextureView, extra: &wgpu::TextureView, buf: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(src),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buffer_resource(buf),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(extra),
                    },
                ],
            })
        };
        // The "extra" (binding 3) texture is only read by the tonemap pass; for the bloom
        // passes we bind a texture that is not the pass's render target.
        let bg_bright = bg("bloom bright bg", &hdr_view, &bloom_b_view, bufs[0]); // hdr -> A
        let bg_blur_h = bg("bloom blur h bg", &bloom_a_view, &hdr_view, bufs[1]); // A -> B
        let bg_blur_v = bg("bloom blur v bg", &bloom_b_view, &hdr_view, bufs[2]); // B -> A
        let bg_tonemap = bg("tonemap bg", &hdr_view, &bloom_a_view, bufs[3]); // hdr + A -> swapchain

        PostTargets {
            _hdr: hdr,
            hdr_view,
            _bloom_a: bloom_a,
            bloom_a_view,
            _bloom_b: bloom_b,
            bloom_b_view,
            bg_bright,
            bg_blur_h,
            bg_blur_v,
            bg_tonemap,
        }
    }

    fn write_post_params(&self) {
        let (w, h) = (self.config.width.max(1) as f32, self.config.height.max(1) as f32);
        let (bw, bh) = (((self.config.width / 2).max(1)) as f32, ((self.config.height / 2).max(1)) as f32);
        let manual_gamma = if self.config.format.is_srgb() { 0.0 } else { 1.0 };
        let base = PostParams {
            src_texel: [1.0 / w, 1.0 / h],
            dir: [0.0, 0.0],
            bloom_strength: if self.bloom { self.bloom_strength } else { 0.0 },
            exposure: EXPOSURE,
            manual_gamma,
            threshold: BLOOM_THRESHOLD,
        };
        let blur_h = PostParams {
            src_texel: [1.0 / bw, 1.0 / bh],
            dir: [1.5, 0.0],
            ..base
        };
        let blur_v = PostParams {
            dir: [0.0, 1.5],
            ..blur_h
        };
        self.queue.write_buffer(&self.post_bright_buf, 0, bytemuck::bytes_of(&base));
        self.queue.write_buffer(&self.post_blur_h_buf, 0, bytemuck::bytes_of(&blur_h));
        self.queue.write_buffer(&self.post_blur_v_buf, 0, bytemuck::bytes_of(&blur_v));
        self.queue.write_buffer(&self.post_tonemap_buf, 0, bytemuck::bytes_of(&base));
    }

    // ------------------------------------------------------------ public API

    pub fn size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    pub fn vsync(&self) -> bool {
        self.vsync
    }

    pub fn dust_count(&self) -> u32 {
        self.particle_count
    }

    /// Largest dust count this adapter allows (<= DUST_CAP).
    pub fn max_dust(&self) -> u32 {
        self.max_dust.max(DUST_MIN)
    }

    /// Start a CosmicWeb run: (re)create the periodic mesh, put side^2 particles on a lattice
    /// and generate Zel'dovich ICs. Returns (lattice side, particle count).
    pub fn start_cosmo(&mut self, cfg: &CosmoConfig, spectrum: Spectrum) -> (u32, u32) {
        // Lattice side = nearest multiple of mesh/2: a lattice that is incommensurate with the
        // mesh (e.g. 5657 vs 2048) deposits a static beat pattern whose aliases couple linearly
        // to the displacements (measured: up to -12% in low-k bins by a = 0.1). With side a
        // multiple of N/2 the lattice harmonics alias only onto k = 0 / Nyquist.
        let unit = (cfg.mesh / 2).max(1);
        let max_side = ((self.max_dust() as f64).sqrt().floor() as u32 / unit).max(1) * unit;
        let side = if cfg.exact_lattice {
            ((cfg.particles as f64).sqrt().round() as u32).min((self.max_dust() as f64).sqrt().floor() as u32)
        } else {
            ((((cfg.particles as f64).sqrt() / unit as f64).round() as u32).max(1) * unit).min(max_side)
        };
        let np = side * side;
        if self.cosmo.as_ref().map(|c| c.n) != Some(cfg.mesh) {
            self.cosmo = Some(Cosmo::new(&self.device, &self.queue, cfg.mesh));
        }
        // set_dust_count works on the pinned range; a lattice count is set directly.
        if np != self.particle_count {
            self.particle_count = np;
            self.particle_buf.destroy();
            self.particle_buf = Self::create_particle_buffer(&self.device, np);
            self.rebuild_compute_bg();
            if let Some(pm) = self.pm.as_mut() {
                pm.bind(&self.device, &self.particle_buf, &self.bodies_buf);
            }
            self.dust_bg = Self::make_pair_bg(&self.device, &self.scene_layout, &self.camera_buf, &self.particle_buf, "dust bg");
        }
        match cfg.ic_mesh.filter(|&m| m != cfg.mesh) {
            None => {
                let c = self.cosmo.as_mut().unwrap();
                c.configure(&self.device, &self.particle_buf, side, cfg);
                c.initial_conditions(&self.device, &self.queue, cfg, spectrum);
            }
            Some(m) => {
                // ICs on the IC mesh (its white noise, phases and CIC deconvolution), written into
                // the shared particle buffer; then the run mesh takes over the same particles.
                self.cosmo = None;
                let mut ic_cfg = cfg.clone();
                ic_cfg.mesh = m;
                ic_cfg.pp = false;
                ic_cfg.snapshots = false;
                let mut ic = Cosmo::new(&self.device, &self.queue, m);
                ic.configure(&self.device, &self.particle_buf, side, &ic_cfg);
                ic.initial_conditions(&self.device, &self.queue, &ic_cfg, spectrum);
                let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
                drop(ic);
                let mut c = Cosmo::new(&self.device, &self.queue, cfg.mesh);
                c.configure(&self.device, &self.particle_buf, side, cfg);
                c.initial_conditions_with(&self.device, &self.queue, cfg, spectrum, false);
                self.cosmo = Some(c);
            }
        }
        self.cosmo_on = true;
        (side, np)
    }

    /// Run-to-run determinism test (startup only, blocking): generate the ICs of `cfg`, take
    /// `steps` P3M steps of dln a = 0.01, read the particles back; twice. Returns
    /// (particles whose 16-byte state differs, particle count).
    pub fn cosmo_determinism_test(&mut self, cfg: &CosmoConfig, spectrum: Spectrum, steps: u32) -> (usize, usize) {
        let mut states: Vec<Vec<u8>> = Vec::new();
        let mut np = 0;
        for _ in 0..2 {
            np = self.start_cosmo(cfg, spectrum).1 as usize;
            let mut a = cfg.a_init;
            for _ in 0..steps {
                let a2 = a * 0.01f64.exp();
                let step = crate::cosmo::CosmoStep {
                    a: a2,
                    kick1: crate::cosmo::kick_factor(a, a2),
                    kick2: 0.0,
                    drift: crate::cosmo::drift_factor(a, a2),
                    measure: false,
                    snapshot: false,
                    well: [0.0; 4],
                    well_kick: 0.0,
                    g_thresh: f64::MAX,
                };
                let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("determinism test") });
                let c = self.cosmo.as_mut().unwrap();
                c.encode(&self.device, &self.queue, &mut enc, &step, &|_| None);
                self.queue.submit(Some(enc.finish()));
                c.after_submit();
                let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
                a = a2;
            }
            let bytes = np as u64 * 16;
            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("determinism readback"),
                size: bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("determinism copy") });
            enc.copy_buffer_to_buffer(&self.particle_buf, 0, &staging, 0, bytes);
            self.queue.submit(Some(enc.finish()));
            staging.map_async(wgpu::MapMode::Read, .., |_| {});
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
            states.push(staging.slice(..).get_mapped_range().expect("map").to_vec());
            staging.unmap();
        }
        let differing = states[0].chunks_exact(16).zip(states[1].chunks_exact(16)).filter(|(x, y)| x != y).count();
        self.stop_cosmo();
        (differing, np)
    }

    /// GPU errors so far (excluding swapchain acquisition) and whether the device was lost.
    pub fn gpu_health(&self) -> (u64, bool) {
        (
            self.gpu_errors.load(std::sync::atomic::Ordering::Acquire),
            self.device_lost.load(std::sync::atomic::Ordering::Acquire) || self.cosmo.as_ref().is_some_and(|c| c.readback_failed),
        )
    }

    /// Latest GPU step maxima (science-mode CFL), 1-2 steps late.
    pub fn cosmo_stats(&self) -> Option<crate::cosmo::StepStats> {
        self.cosmo.as_ref().and_then(|c| c.last_stats)
    }

    pub fn cosmo_load_state(&mut self, data: &[[f32; 4]]) {
        if let Some(c) = self.cosmo.as_mut() {
            c.load_state(&self.queue, data);
        }
    }

    pub fn cosmo_snapshot_in_flight(&self) -> bool {
        self.cosmo.as_ref().is_some_and(|c| c.snapshot_in_flight())
    }

    pub fn cosmo_readbacks_pending(&self) -> bool {
        self.cosmo.as_ref().is_some_and(|c| c.readbacks_pending())
    }

    /// GPU two-point force test on a mesh of n cells per side (blocking; startup only).
    pub fn cosmo_force_test(&mut self, n: u32, eps: f64, radii: &[f64]) -> Vec<[f64; 4]> {
        if self.cosmo.as_ref().map(|c| c.n) != Some(n) {
            self.cosmo = Some(Cosmo::new(&self.device, &self.queue, n));
        }
        let c = self.cosmo.as_mut().unwrap();
        let r = c.force_test(&self.device, &self.queue, eps, radii);
        self.cosmo = None;
        r
    }

    pub fn stop_cosmo(&mut self) {
        self.cosmo_on = false;
    }

    pub fn cosmo_poll(&mut self, writer: &RunWriter) {
        if let Some(c) = self.cosmo.as_mut() {
            c.poll(writer);
        }
    }

    /// (measurement recorded, snapshot recorded) in the last frame.
    pub fn cosmo_recorded(&self) -> (bool, bool) {
        self.cosmo_last
    }

    /// Did the last render() execute the cosmology step it was given?
    pub fn cosmo_ran(&self) -> bool {
        self.cosmo_ran
    }

    pub fn set_speed_ref(&mut self, v: f32) {
        self.speed_ref = v.max(1e-6);
    }

    pub fn set_bloom_strength(&mut self, s: f32) {
        self.bloom_strength = s;
        self.write_post_params();
    }

    /// Particle-mesh self-gravity available (created at startup)?
    pub fn has_pm(&self) -> bool {
        self.pm.is_some()
    }

    /// Self-gravity currently on (G toggles it; off = attractor mode).
    pub fn pm_on(&self) -> bool {
        self.pm_on && self.pm.is_some()
    }

    pub fn set_pm_on(&mut self, on: bool) {
        self.pm_on = on && self.pm.is_some();
    }

    pub fn pm_mesh(&self) -> Option<u32> {
        self.pm.as_ref().map(|p| p.n)
    }

    /// Non-blocking: process finished GPU work (map callbacks) and fold in timings.
    pub fn poll(&mut self) {
        let _ = self.device.poll(wgpu::PollType::Poll);
        if let Some(t) = self.timer.as_mut() {
            t.update(&self.stage_active);
        }
    }

    /// Copy the newest dust-only PM field (if one arrived) into out.
    pub fn read_pm_field(&mut self, out: &mut sim_core::ExternalField) -> bool {
        match self.pm.as_mut() {
            Some(pm) if self.pm_on => pm.read_field(out),
            _ => false,
        }
    }

    /// Smoothed GPU stage times in ms (None without TIMESTAMP_QUERY).
    pub fn stage_ms(&self) -> Option<&[f32; timer::STAGES]> {
        self.timer.as_ref().map(|t| &t.ms)
    }

    pub fn pm_ms(&self) -> Option<f32> {
        self.timer.as_ref().map(|t| t.pm_total())
    }

    /// GPU PM validation (blocking; startup/debug only).
    pub fn pm_self_test(&mut self) -> Option<(String, f32)> {
        let pm = self.pm.as_mut()?;
        Some(pm.self_test(&self.device, &self.queue))
    }

    pub fn toggle_bloom(&mut self) {
        self.bloom = !self.bloom;
        self.write_post_params();
    }

    pub fn toggle_vsync(&mut self) {
        match self.uncapped_mode {
            Some(m) => {
                self.vsync = !self.vsync;
                self.config.present_mode = if self.vsync { wgpu::PresentMode::Fifo } else { m };
                self.surface.configure(&self.device, &self.config);
            }
            None => log::warn!("no Mailbox/Immediate present mode available; VSync stays on"),
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        if width == self.config.width && height == self.config.height {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.targets = Self::create_targets(
            &self.device,
            &self.post_layout,
            &self.sampler,
            [
                &self.post_bright_buf,
                &self.post_blur_h_buf,
                &self.post_blur_v_buf,
                &self.post_tonemap_buf,
            ],
            width,
            height,
        );
        self.write_post_params();
    }

    fn recreate_surface(&mut self) {
        match self.instance.create_surface(self.window.clone()) {
            Ok(s) => {
                self.surface = s;
                self.surface.configure(&self.device, &self.config);
                // If the window changed size meanwhile, also rebuild the HDR/bloom targets.
                let size = self.window.inner_size();
                self.resize(size.width, size.height);
            }
            Err(e) => log::error!("failed to recreate surface: {e}"),
        }
    }

    /// Change the dust particle count (re-creates the particle buffer); the caller must
    /// re-seed afterwards with `init_dust`.
    pub fn set_dust_count(&mut self, count: u32) {
        let count = count.clamp(DUST_MIN, self.max_dust());
        if count == self.particle_count {
            return;
        }
        self.particle_count = count;
        // Free the old buffer before allocating the new one (keeps peak VRAM down at 64M).
        self.particle_buf.destroy();
        self.particle_buf = Self::create_particle_buffer(&self.device, count);
        self.rebuild_compute_bg();
        if let Some(pm) = self.pm.as_mut() {
            pm.bind(&self.device, &self.particle_buf, &self.bodies_buf);
        }
        if let Some(c) = self.cosmo.as_mut() {
            c.bind(&self.device, &self.particle_buf);
        }
        self.dust_bg = Self::make_pair_bg(
            &self.device,
            &self.scene_layout,
            &self.camera_buf,
            &self.particle_buf,
            "dust bg",
        );
    }

    /// Seed the dust on the GPU.
    /// * `SeedMode::Spiral`: disc with circular speeds from `attractors` (no mouse well).
    /// * `SeedMode::Bodies`: each particle picks a body via a mass-weighted CDF (binary
    ///   search in the shader), is placed around it and inherits its velocity.
    #[allow(clippy::too_many_arguments)]
    pub fn init_dust(
        &mut self,
        mode: SeedMode,
        seed: DustSeed,
        bodies: &[Body],
        attractors: &[Attractor; MAX_ATTRACTORS],
        n_attr: usize,
        speed_ref: f32,
        dust_mass: f32,
    ) {
        // Bodies + CDF (seed time only; allocation here is fine).
        self.upload_bodies(bodies);
        self.cdf_scratch.clear();
        let mut acc = 0.0f64;
        for b in bodies {
            acc += (b.mass.max(0.0) as f64).powf(SEED_MASS_EXP as f64);
            self.cdf_scratch.push(acc as f32);
        }
        let mode = if bodies.is_empty() || acc <= 0.0 { SeedMode::Spiral } else { mode };
        if !self.cdf_scratch.is_empty() {
            self.queue
                .write_buffer(&self.cdf_buf, 0, bytemuck::cast_slice(&self.cdf_scratch));
        }

        self.speed_ref = speed_ref.max(1.0);
        self.params.dt = 0.0;
        self.params.n_attr = n_attr.min(MAX_ATTRACTORS) as u32;
        self.params.attr = *attractors;
        self.params.n_particles = self.particle_count;
        self.params.seed = seed.seed;
        self.params.disc_radius = seed.disc_radius.clamp(10.0, WORLD_HALF * 0.95);
        self.params.spin = if seed.spin < 0.0 { -1.0 } else { 1.0 };
        self.params.n_bodies = bodies.len() as u32;
        self.params.seed_mode = mode as u32;
        self.params.total_weight = acc as f32;
        // With self-gravity the seeding includes the dust's own mass in the circular speeds.
        let self_grav = self.pm.is_some() && self.pm_on;
        self.params.dust_mass = if self_grav { dust_mass.max(0.0) } else { 0.0 };
        if let Some(pm) = self.pm.as_mut() {
            pm.set_dust(self.particle_count, dust_mass.max(0.0));
        }
        self.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&self.params));

        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dust init"),
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("dust init"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.init_pipeline);
            pass.set_bind_group(0, &self.compute_bg, &[]);
            let (gx, gy) = dispatch_dims(self.particle_count);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        self.queue.submit(Some(enc.finish()));
    }

    fn upload_bodies(&mut self, bodies: &[Body]) {
        if bodies.len() > self.bodies_capacity {
            let cap = bodies.len().next_power_of_two();
            log::info!("growing bodies buffer {} -> {}", self.bodies_capacity, cap);
            self.bodies_capacity = cap;
            self.bodies_buf = Self::create_bodies_buffer(&self.device, cap);
            self.bodies_bg = Self::make_pair_bg(
                &self.device,
                &self.scene_layout,
                &self.camera_buf,
                &self.bodies_buf,
                "bodies bg",
            );
            self.cdf_buf = Self::create_cdf_buffer(&self.device, cap);
            self.rebuild_compute_bg();
            if let Some(pm) = self.pm.as_mut() {
                pm.bind(&self.device, &self.particle_buf, &self.bodies_buf);
            }
        }
        if !bodies.is_empty() {
            self.queue
                .write_buffer(&self.bodies_buf, 0, bytemuck::cast_slice(bodies));
        }
    }

    pub fn render(&mut self, frame: &FrameData<'_>) -> RenderOutcome {
        self.cosmo_ran = false;
        self.cosmo_last = (false, false);

        // The simulation (compute) is recorded and submitted even when no swapchain image is
        // available (minimized / occluded / lost surface), so unattended science runs never stall
        // on presentation; only the render passes are skipped.
        let mut reconfigure_after = false;
        let surface_tex = if self.config.width == 0 || self.config.height == 0 {
            None
        } else {
            self.acquiring.store(true, std::sync::atomic::Ordering::Release);
            let acquired = self.surface.get_current_texture();
            self.acquiring.store(false, std::sync::atomic::Ordering::Release);
            match acquired {
            wgpu::CurrentSurfaceTexture::Success(t) => Some(t),
            wgpu::CurrentSurfaceTexture::Suboptimal(t) => {
                // Use this frame, reconfigure after presenting it.
                reconfigure_after = true;
                Some(t)
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                let size = self.window.inner_size();
                if size.width > 0 && size.height > 0 {
                    if size.width != self.config.width || size.height != self.config.height {
                        self.resize(size.width, size.height);
                    } else {
                        self.surface.configure(&self.device, &self.config);
                    }
                }
                None
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                log::warn!("surface lost; recreating");
                self.recreate_surface();
                None
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => None,
            wgpu::CurrentSurfaceTexture::Validation => {
                // e.g. a minimized window on some drivers: reconfigure, keep simulating
                self.surface_errors += 1;
                if self.surface_errors == 1 || self.surface_errors % 1000 == 0 {
                    log::warn!("no swapchain image (surface validation error x{}); reconfiguring, simulation continues", self.surface_errors);
                }
                let size = self.window.inner_size();
                if size.width > 0 && size.height > 0 {
                    if size.width != self.config.width || size.height != self.config.height {
                        self.resize(size.width, size.height);
                    } else {
                        self.surface.configure(&self.device, &self.config);
                    }
                }
                None
            }
            }
        };

        // ---- uploads (fixed-size buffers, no allocations)
        self.upload_bodies(frame.bodies);

        self.params.dt = frame.dt;
        self.params.n_attr = frame.n_attractors.min(MAX_ATTRACTORS) as u32;
        self.params.attr = *frame.attractors;
        self.params.n_particles = self.particle_count;
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&self.params));

        let cam = frame.camera;
        let half = cam.half_extent();
        let zoom_gain = (crate::camera::INITIAL_HALF_HEIGHT / cam.half_height).clamp(0.25, 8.0);
        // Per-particle brightness falls almost as fast as the count rises, so total dust
        // light grows only ~count^0.2 (32M is ~2x as bright as 1M instead of washing out).
        let count_gain = (DUST_DEFAULT as f32 / self.particle_count as f32).powf(0.8);
        // More pixels = fewer particles per pixel; compensate (4K is ~3.7x 1600x900's gain).
        let res_gain = (cam.viewport[0] * cam.viewport[1] / (1600.0 * 900.0))
            .clamp(0.25, 16.0)
            .powf(0.75);
        let cam_u = CameraUniform {
            center: cam.center,
            half_extent: half,
            viewport: cam.viewport,
            speed_ref: self.speed_ref,
            lattice: if self.cosmo_on { self.cosmo.as_ref().map_or(0.0, |c| c.side as f32) } else { 0.0 },
            box_l: 2.0 * WORLD_HALF,
            _p0: 0.0,
            _p1: 0.0,
            dust_intensity: 0.10 * count_gain * zoom_gain * res_gain,
        };
        self.queue
            .write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(&cam_u));

        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("frame"),
        });

        // ---- 1. dust compute: PM self-gravity, or the analytic attractor field
        let run_dust = frame.simulate_dust && frame.dt > 0.0;
        let use_pm = self.pm_on && self.pm.is_some();
        self.stage_active = [false; timer::STAGES];
        self.pm_ran = false;
        self.cosmo_ran = false;
        self.cosmo_last = (false, false);
        if self.cosmo_on {
            // Steps are not taken every frame (MIN_DLNA); keep the cosmology stages active so the
            // smoothed timings persist between steps.
            for s in [0, 1, 2, 3, 4, 5] {
                self.stage_active[s] = true;
            }
            if let (Some(step), Some(c)) = (frame.cosmo.as_ref(), self.cosmo.as_mut()) {
                let t = self.timer.as_ref();
                let ts = |s: usize| t.map(|t| t.compute(s));
                c.encode(&self.device, &self.queue, &mut enc, step, &ts);
                self.cosmo_last = c.recorded();
                self.cosmo_ran = true;
            } else if let Some(c) = self.cosmo.as_mut() {
                // no step this frame: keep streaming an in-flight snapshot
                c.encode_snapshot_chunks(&mut enc);
                for s in [0, 1, 2, 3, 5] {
                    self.stage_active[s] = true;
                }
            }
        } else if run_dust && use_pm {
            let pm = self.pm.as_mut().unwrap();
            pm.prepare(&self.queue, frame.bodies.len() as u32, frame.body_mass, frame.dt, frame.well);
            let t = self.timer.as_ref();
            let ts = |s: usize| t.map(|t| t.compute(s));
            pm.encode(&mut enc, &ts);
            self.pm_ran = true;
            for s in 0..6 {
                self.stage_active[s] = true;
            }
        } else if run_dust {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("dust update"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.update_pipeline);
            pass.set_bind_group(0, &self.compute_bg, &[]);
            let (gx, gy) = dispatch_dims(self.particle_count);
            pass.dispatch_workgroups(gx, gy, 1);
        }

        let Some(surface_tex) = surface_tex else {
            // compute only
            if let Some(t) = self.timer.as_mut() {
                t.resolve(&mut enc);
            }
            self.queue.submit(Some(enc.finish()));
            self.after_submit_all();
            return RenderOutcome::Skipped;
        };
        let swap_view = surface_tex.texture.create_view(&wgpu::TextureViewDescriptor::default());

        // ---- 2. HDR scene
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("hdr scene"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.targets.hdr_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.002,
                            g: 0.002,
                            b: 0.006,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: self.timer.as_ref().and_then(|t| t.render(timer::ST_SCENE, true, true)),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.dust_pipeline);
            pass.set_bind_group(0, &self.dust_bg, &[]);
            pass.draw(0..self.particle_count, 0..1);

            if !frame.bodies.is_empty() {
                pass.set_pipeline(&self.bodies_pipeline);
                pass.set_bind_group(0, &self.bodies_bg, &[]);
                pass.draw(0..6, 0..frame.bodies.len() as u32);
            }
        }

        // ---- 3. bloom (bright-pass downsample, separable blur)
        self.stage_active[timer::ST_SCENE] = true;
        self.stage_active[timer::ST_POST] = true;
        let tm = self.timer.as_ref();
        if self.bloom {
            let t = &self.targets;
            let ts0 = tm.and_then(|t| t.render(timer::ST_POST, true, false));
            Self::fullscreen_pass(&mut enc, "bloom bright", &t.bloom_a_view, &self.bright_pipeline, &t.bg_bright, ts0);
            Self::fullscreen_pass(&mut enc, "bloom blur h", &t.bloom_b_view, &self.blur_pipeline, &t.bg_blur_h, None);
            Self::fullscreen_pass(&mut enc, "bloom blur v", &t.bloom_a_view, &self.blur_pipeline, &t.bg_blur_v, None);
        }

        // ---- 4. tonemap to swapchain
        Self::fullscreen_pass(
            &mut enc,
            "tonemap",
            &swap_view,
            &self.tonemap_pipeline,
            &self.targets.bg_tonemap,
            tm.and_then(|t| t.render(timer::ST_POST, !self.bloom, true)),
        );

        if let Some(t) = self.timer.as_mut() {
            t.resolve(&mut enc);
        }
        self.queue.submit(Some(enc.finish()));
        self.after_submit_all();
        self.window.pre_present_notify();
        self.queue.present(surface_tex);
        if reconfigure_after {
            // Only reconfigure on a real size change; a persistently suboptimal surface
            // would otherwise be reconfigured every frame.
            let size = self.window.inner_size();
            self.resize(size.width, size.height);
        }
        RenderOutcome::Ok
    }

    fn after_submit_all(&mut self) {
        if self.pm_ran {
            if let Some(pm) = self.pm.as_mut() {
                pm.after_submit();
            }
        }
        if let Some(t) = self.timer.as_mut() {
            t.after_submit();
        }
        if self.cosmo_on {
            if let Some(c) = self.cosmo.as_mut() {
                c.after_submit();
            }
        }
    }

    fn fullscreen_pass(
        enc: &mut wgpu::CommandEncoder,
        label: &str,
        target: &wgpu::TextureView,
        pipeline: &wgpu::RenderPipeline,
        bg: &wgpu::BindGroup,
        ts: Option<wgpu::RenderPassTimestampWrites<'_>>,
    ) {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: ts,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod shader_tests {
    use wgpu::naga;

    fn validate(name: &str, src: &str) {
        let module = naga::front::wgsl::parse_str(src)
            .unwrap_or_else(|e| panic!("{name}: parse error:\n{}", e.emit_to_string(src)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name}: validation error:\n{}", e.emit_to_string(src)));
    }

    #[test]
    fn all_shaders_validate() {
        validate("dust_compute", include_str!("shaders/dust_compute.wgsl"));
        validate("dust_render", include_str!("shaders/dust_render.wgsl"));
        validate("bodies", include_str!("shaders/bodies.wgsl"));
        validate("post", include_str!("shaders/post.wgsl"));
        for p in [512u32, 2048, 4096] {
            let src = (include_str!("shaders/pm.wgsl").to_string() + include_str!("shaders/fft.wgsl"))
                .replace("{{P}}", &p.to_string())
                .replace("{{LOGP}}", &p.trailing_zeros().to_string())
                .replace("{{PT}}", &(p / 256).to_string());
            validate("pm", &src);
        }
        for n in [256u32, 1024, 2048] {
            validate("cosmo", &crate::cosmo::shader_source(n));
        }
    }
}