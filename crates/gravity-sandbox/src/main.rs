//! Gravity Sandbox: CPU N-body planets (sim_core) + up to 64M GPU dust particles (wgpu compute).

mod camera;
mod cosmo;
mod cosmo_io;
mod cosmo_run;
mod gpu;
mod pm;
mod readback;
mod settings;
mod timer;

use std::sync::Arc;
use std::time::Instant;

use sim_core::{Attractor, Input, MAX_ATTRACTORS, Scenario, World};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Fullscreen, Window, WindowId};

use camera::Camera;
use gpu::{DustSeed, FrameData, Renderer, RendererOptions, SeedMode};
use cosmo_run::{CosmoRun, Experiment, ModelSpec};
use settings::{BODIES_MAX, BODIES_MIN, CosmoSettings, ParseOutcome, Settings};

const INITIAL_SEED: u64 = 1;
const CLUSTER_SIZE: usize = 200;
// Matches sim-core's internal clamp so dust and bodies advance by the same dt.
const MAX_DT: f32 = 1.0 / 30.0;
const HUD_INTERVAL: f32 = 0.25;
const TIMING_PRINT_INTERVAL: f32 = 3.0;
/// Momentum kick per second of holding the mouse well in the cosmic web (not science).
const COSMO_WELL_GAIN: f32 = 0.05;

const CONTROLS: &str = "\
Controls:
  Left mouse (hold)   attract (gravity well)
  Right mouse (hold)  repel
  Middle drag / WASD  pan
  Mouse wheel         zoom towards cursor
  Space               spawn a cluster of 200 bodies at the cursor
  R                   reset (new seed)
  P                   pause
  V                   toggle VSync
  B                   toggle bloom
  + / -               double / halve dust count (125k .. adapter max)
  [ / ]               halve / double body count (1k .. 64k, resets)
  1 / 2               scenario: galaxy / subhalo merger (resets)
  F11                 toggle borderless fullscreen
  G                   toggle dust self-gravity (particle-mesh) / attractor mode
  3                   scenario: cosmic web (periodic comoving PM run, writes D:\\cosmo-runs)
  Esc                 quit";

struct Game {
    window: Arc<Window>,
    renderer: Renderer,
    world: World,
    camera: Camera,
    seed: u64,
    scenario: Scenario,
    n_bodies: usize,

    attractors: [Attractor; MAX_ATTRACTORS],
    cursor_px: [f32; 2],
    left: bool,
    right: bool,
    middle: bool,
    keys_pan: [bool; 4], // W A S D
    paused: bool,
    /// Dust mass / body mass (None = scenario default).
    dark_matter: Option<f32>,
    timing_timer: f32,
    cosmo_settings: CosmoSettings,
    mesh: u32,
    /// Requested dust count for cosmic runs (rounded to a square lattice).
    cosmo_particles: u32,
    cosmo_run: Option<CosmoRun>,
    experiment: Option<Experiment>,
    /// set when an experiment has finished (or failed): the app exits
    quit: bool,

    last_frame: Instant,
    hud_timer: f32,
    hud_frames: u32,
    hud_frame_time: f32,
}

impl Game {
    fn input(&self) -> Input {
        let well = if self.left {
            1.0
        } else if self.right {
            -1.0
        } else {
            0.0
        };
        Input {
            cursor: self.camera.screen_to_world(self.cursor_px[0], self.cursor_px[1]),
            well,
        }
    }

    fn dark_matter_ratio(&self) -> f32 {
        self.dark_matter.unwrap_or(match self.world.scenario() {
            Scenario::Galaxy => 0.5,
            _ => 1.0,
        })
    }

    fn body_mass(&self) -> f32 {
        self.world.bodies.iter().map(|b| b.mass.max(0.0)).sum()
    }

    /// Seed the GPU dust disc from the current world (radius, rotation sense, speeds).
    fn reseed_dust(&mut self) {
        let dust_mass = self.dark_matter_ratio() * self.body_mass();
        let bodies = &self.world.bodies;
        let mut radii: Vec<f32> = bodies
            .iter()
            .map(|b| (b.pos[0] * b.pos[0] + b.pos[1] * b.pos[1]).sqrt())
            .collect();
        radii.sort_by(|a, b| a.total_cmp(b));
        let disc_radius = if radii.is_empty() {
            800.0
        } else {
            radii[(radii.len() * 95 / 100).min(radii.len() - 1)].max(100.0)
        };
        let mut ang_mom = 0.0f64;
        let mut v2 = 0.0f64;
        let mut m_sum = 0.0f64;
        for b in bodies {
            let m = b.mass as f64;
            ang_mom += m * (b.pos[0] * b.vel[1] - b.pos[1] * b.vel[0]) as f64;
            v2 += m * (b.vel[0] * b.vel[0] + b.vel[1] * b.vel[1]) as f64;
            m_sum += m;
        }
        let rms_speed = if m_sum > 0.0 { (v2 / m_sum).sqrt() as f32 } else { 50.0 };
        // Self-gravitating dust runs hotter than the bodies; widen the colour range.
        let speed_k = if self.renderer.pm_on() { 3.0 } else { 2.0 };
        let speed_ref = (rms_speed * speed_k).max(20.0);
        let spin = if ang_mom < 0.0 { -1.0 } else { 1.0 };

        let n = self.world.attractors(&Input::default(), &mut self.attractors);
        // The spiral disc reads best for the galaxy; any other scenario seeds dust around
        // its bodies (mass-weighted) so subhalos and their streams are traced.
        let mode = match self.world.scenario() {
            Scenario::Galaxy => SeedMode::Spiral,
            _ => SeedMode::Bodies,
        };
        self.renderer.init_dust(
            mode,
            DustSeed {
                seed: self.seed as u32,
                disc_radius,
                spin,
            },
            &self.world.bodies,
            &self.attractors,
            n,
            speed_ref,
            dust_mass,
        );
    }

    fn toggle_self_gravity(&mut self) {
        if !self.renderer.has_pm() {
            println!("self-gravity unavailable (started with --no-pm)");
            return;
        }
        let on = !self.renderer.pm_on();
        self.renderer.set_pm_on(on);
        if !on {
            self.world.set_external_field(None);
        }
        println!("dust self-gravity: {}", if on { "on (particle-mesh)" } else { "off (attractor mode)" });
    }

    fn reset(&mut self) {
        self.seed = self.seed.wrapping_add(1);
        self.rebuild_world();
    }

    fn rebuild_world(&mut self) {
        self.world = World::with_scenario(self.scenario, self.n_bodies, self.seed);
        if self.scenario == Scenario::CosmicWeb {
            self.world.set_external_field(None);
            self.start_cosmo_run();
        } else {
            if self.cosmo_run.take().is_some() {
                self.renderer.stop_cosmo();
                self.renderer.set_bloom_strength(gpu::BLOOM_STRENGTH);
                // back to the pinned dust range
                let n = self.cosmo_particles.clamp(gpu::DUST_MIN, self.renderer.max_dust());
                self.renderer.set_dust_count(n);
            }
            self.spin_up_for_dark_matter();
            self.reseed_dust();
        }
    }

    /// Start a new cosmological run with the current settings (experiment: the next model).
    fn start_cosmo_run(&mut self) {
        // Close the previous run first (flushes its files).
        self.cosmo_run = None;
        let seed = (self.cosmo_settings.seed as u32).wrapping_add(self.seed as u32 - INITIAL_SEED as u32);
        let model = if self.cosmo_settings.experiment {
            if self.experiment.is_none() {
                let side = cosmo_run::run_lattice_side(&self.cosmo_settings, self.mesh, self.cosmo_particles, self.renderer.max_dust());
                let np = (side as f64) * (side as f64);
                let exp = Experiment::new(std::path::Path::new(&self.cosmo_settings.runs_dir), self.cosmo_settings.pair_id.clone(), np, &self.cosmo_settings);
                println!("experiment {} -> {}", exp.pair_id, exp.dir.display());
                for m in &exp.models {
                    if let Some(k) = m.k_hm {
                        println!(
                            "  model {}: k_hm = {k:.4}, M_hm = {:.4e} ({:.0} particle masses)",
                            m.name,
                            cosmo_run::half_mode_mass(k),
                            cosmo_run::half_mode_mass(k) / (cosmo::BOX * cosmo::BOX / np)
                        );
                    } else {
                        println!("  model {}", m.name);
                    }
                }
                self.experiment = Some(exp);
            }
            match self.experiment.as_mut().unwrap().next_model() {
                Some(m) => m,
                None => {
                    self.finish_experiment();
                    return;
                }
            }
        } else {
            match self.cosmo_settings.k_hm {
                Some(k) => ModelSpec::wdm(k),
                None => ModelSpec::cdm(),
            }
        };
        match cosmo_run::start_run(&mut self.renderer, &self.cosmo_settings, self.mesh, self.cosmo_particles, seed, &model) {
            Ok(run) => {
                self.camera = Camera::new(self.camera.viewport[0] as u32, self.camera.viewport[1] as u32);
                self.camera.half_height = (cosmo::BOX as f32) * 0.5 * 1.02;
                self.renderer.set_bloom_strength(0.3);
                self.cosmo_run = Some(run);
            }
            Err(e) => {
                eprintln!("cosmic web: {e}");
                self.renderer.stop_cosmo();
                if self.experiment.is_some() {
                    self.quit = true;
                }
            }
        }
    }

    /// Called every frame: close a finished run (once all readbacks are handed over) and,
    /// in an experiment, start the next model.
    fn advance_cosmo_runs(&mut self) {
        let Some(run) = self.cosmo_run.as_ref() else {
            return;
        };
        let (errs, lost) = self.renderer.gpu_health();
        if errs > run.gpu_errors_at_start || lost {
            eprintln!("cosmic web [{}]: GPU error ({} new, device/readback lost: {lost}); the run is NOT marked complete. Stopping cleanly; rerun with the same --pair-id to resume. Run dir: {}", run.model, errs - run.gpu_errors_at_start, run.writer.dir.display());
            self.cosmo_run = None;
            self.renderer.stop_cosmo();
            self.quit = true;
            return;
        }
        if run.writer.failed() {
            eprintln!("cosmic web [{}]: output failed (disk space or write error); stopping cleanly. Run dir: {}", run.model, run.writer.dir.display());
            self.cosmo_run = None;
            self.renderer.stop_cosmo();
            if self.experiment.is_some() {
                self.quit = true;
            }
            return;
        }
        if !run.schedule.finished || self.renderer.cosmo_readbacks_pending() {
            return;
        }
        let run = self.cosmo_run.take().unwrap();
        run.writer.mark_complete();
        let wall = run.started.elapsed().as_secs_f64();
        let line = format!(
            "{}: {} steps in {:.1} min ({} at the dln a floor); pm+pp step cost early {} late {}; dir {}",
            run.model,
            run.schedule.steps,
            wall / 60.0,
            run.schedule.floor_steps,
            run.step_ms_first.map_or("-".into(), |(a, ms)| format!("{ms:.1} ms (a={a:.3})")),
            run.step_ms_last.map_or("-".into(), |(a, ms)| format!("{ms:.1} ms (a={a:.3})")),
            run.writer.dir.display()
        );
        {
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(run.writer.dir.join("run.log")) {
                let _ = writeln!(f, "  summary: {line}");
            }
        }
        drop(run); // joins the writer (writes COMPLETE)
        println!("cosmic web run finished: {line}");
        if let Some(exp) = self.experiment.as_mut() {
            exp.summary.push(line);
            self.start_cosmo_run();
        }
    }

    fn finish_experiment(&mut self) {
        if let Some(exp) = self.experiment.as_ref() {
            println!("experiment {} finished in {:.1} min: {}", exp.pair_id, exp.started.elapsed().as_secs_f64() / 60.0, exp.dir.display());
            for l in &exp.summary {
                println!("  {l}");
            }
            let _ = std::fs::write(exp.dir.join("experiment.log"), exp.summary.join("\n") + "\n");
        }
        self.renderer.stop_cosmo();
        self.quit = true;
    }
    /// The scenarios are built for the bodies' mass alone. With self-gravitating dust that
    /// traces the body mass (ratio f), every potential well is (1 + f) times deeper, so
    /// scaling all body velocities by sqrt(1 + f) keeps the initial orbits in equilibrium
    /// instead of letting everything fall in. Only applied to a freshly built world.
    fn spin_up_for_dark_matter(&mut self) {
        if !self.renderer.pm_on() {
            return;
        }
        let k = (1.0 + self.dark_matter_ratio()).sqrt();
        for b in &mut self.world.bodies {
            b.vel[0] *= k;
            b.vel[1] *= k;
        }
    }

    fn set_scenario(&mut self, scenario: Scenario) {
        self.scenario = scenario;
        println!("scenario: {}", scenario.name());
        self.reset();
    }

    fn change_bodies(&mut self, double: bool) {
        let next = if double { self.n_bodies * 2 } else { self.n_bodies / 2 }.clamp(BODIES_MIN, BODIES_MAX);
        if next != self.n_bodies {
            self.n_bodies = next;
            println!("bodies: {next}");
            self.reset();
        }
    }

    fn toggle_fullscreen(&mut self) {
        let fs = if self.window.fullscreen().is_some() {
            None
        } else {
            Some(Fullscreen::Borderless(None))
        };
        self.window.set_fullscreen(fs);
    }

    fn change_dust(&mut self, double: bool) {
        let cur = self.renderer.dust_count();
        let next = if double {
            cur.saturating_mul(2)
        } else {
            cur / 2
        }
        .clamp(gpu::DUST_MIN, self.renderer.max_dust());
        if self.scenario == Scenario::CosmicWeb {
            self.cosmo_particles = next;
            println!("cosmic web: restarting with ~{next} particles");
            self.start_cosmo_run();
            return;
        }
        if next != cur {
            self.cosmo_particles = next;
            self.renderer.set_dust_count(next);
            self.reseed_dust();
            println!("dust particles: {next}");
        }
    }

    fn frame(&mut self) {
        let now = Instant::now();
        let real_dt = now.duration_since(self.last_frame).as_secs_f32();
        self.last_frame = now;
        let dt = real_dt.min(MAX_DT);

        // Keyboard pan (speed scales with zoom).
        let pan_speed = self.camera.half_height * 1.2 * dt;
        let [w, a, s, d] = self.keys_pan;
        let px = (d as i32 - a as i32) as f32 * pan_speed;
        let py = (w as i32 - s as i32) as f32 * pan_speed;
        if px != 0.0 || py != 0.0 {
            self.camera.pan_world(px, py);
        }

        // Dust field from the GPU (non-blocking, 1-2 frames late) -> CPU bodies.
        self.renderer.poll();
        if let Some(run) = self.cosmo_run.as_ref() {
            self.renderer.cosmo_poll(&run.writer);
        } else if self.renderer.pm_on() {
            let mut field = self.world.take_external_field().unwrap_or_default();
            self.renderer.read_pm_field(&mut field);
            self.world.set_external_field(Some(field));
        }

        let input = self.input();
        if !self.paused {
            self.world.step(dt, &input);
        }
        let n_attr = self.world.attractors(&input, &mut self.attractors);
        // World::attractors puts the mouse well first when it is active.
        let well = (input.well != 0.0 && n_attr > 0).then(|| self.attractors[0]);
        let body_mass = self.body_mass();

        // Cosmology step (comoving KDK; a advances only if the GPU actually ran the step).
        let mut cosmo_step = None;
        if let Some(run) = self.cosmo_run.as_mut() {
            if input.well != 0.0 {
                run.science = false;
            }
            if !self.paused {
                let stats = self.renderer.cosmo_stats();
                let busy = self.renderer.cosmo_snapshot_in_flight();
                cosmo_step = run.schedule.plan(real_dt as f64, stats, busy).map(|mut st| {
                    if let Some(w) = well {
                        st.well = [w.pos[0], w.pos[1], w.strength, w.softening.max(1.0)];
                        st.well_kick = real_dt * COSMO_WELL_GAIN;
                    }
                    st
                });
            }
            let s = &run.schedule;
            // Colour scale: a few times the linear-theory rms momentum, so the coherent bulk
            // flows stay cool and only collapsed, hot regions turn orange/white.
            let lin_p = 4.0 * run.psi_rms * s.a.powf(1.5) / s.a_init;
            self.renderer.set_speed_ref(lin_p.max(1e-3) as f32);
        }

        let data = FrameData {
            bodies: &self.world.bodies,
            attractors: &self.attractors,
            n_attractors: n_attr,
            dt,
            simulate_dust: !self.paused,
            camera: &self.camera,
            well,
            body_mass,
            cosmo: cosmo_step,
        };
        let _ = self.renderer.render(&data);
        if let Some(run) = self.cosmo_run.as_mut() {
            let (measured, snapped) = self.renderer.cosmo_recorded();
            let was_finished = run.schedule.finished;
            run.schedule.commit(self.renderer.cosmo_ran(), measured, snapped);
            if run.science_mode && self.renderer.cosmo_ran() {
                if let Some(ms) = self.renderer.pm_ms() {
                    let a = run.schedule.a;
                    if run.step_ms_first.is_none() && run.schedule.steps >= 20 {
                        run.step_ms_first = Some((a, ms as f64));
                    }
                    run.step_ms_last = Some((a, ms as f64));
                    if run.schedule.steps >= run.last_report_step + 100 {
                        run.last_report_step = run.schedule.steps;
                        let st = self.renderer.cosmo_stats().unwrap_or_default();
                        if let Some(sm) = self.renderer.stage_ms() {
                            let pm: f32 = sm[0] + sm[1] + sm[2] + sm[3] + sm[5];
                            run.writer.log(&format!("step {} a = {:.4}: PP {:.1} ms/step, PM {:.1} ms/step, over-CFL particles {}, floor steps {}", run.schedule.steps, a, sm[4], pm, st.over_cfl, run.schedule.floor_steps));
                        }
                        println!("  [{}] step {} a = {:.4} pm+pp {:.1} ms | at a = {:.4}: |g|max {:.3e}, |p|max {:.3e}, over-CFL particles {} | floor steps {} | {:.1} min", run.model, run.schedule.steps, a, ms, st.a, st.g_max, st.p_max, st.over_cfl, run.schedule.floor_steps, run.started.elapsed().as_secs_f64() / 60.0);
                    }
                }
            }
            if run.schedule.finished && !was_finished {
                println!("cosmic web [{}]: reached a = {:.4} after {} steps; holding (run dir {})", run.model, run.schedule.a, run.schedule.steps, run.writer.dir.display());
            }
        }
        self.advance_cosmo_runs();

        // HUD in the window title.
        self.hud_timer += real_dt;
        self.hud_frames += 1;
        self.hud_frame_time += real_dt;
        if self.hud_timer >= HUD_INTERVAL {
            let fps = self.hud_frames as f32 / self.hud_timer;
            let ms = 1000.0 * self.hud_frame_time / self.hud_frames as f32;
            let st = &self.world.stats;
            let pm = match (self.renderer.pm_on(), self.renderer.pm_ms()) {
                (true, Some(t)) => format!(" | pm {:.1} ms ({}^2, DM x{:.1})", t, self.renderer.pm_mesh().unwrap_or(0), self.dark_matter_ratio()),
                (true, None) => format!(" | pm on ({}^2)", self.renderer.pm_mesh().unwrap_or(0)),
                _ => " | pm off".to_string(),
            };
            let cosmo = match self.cosmo_run.as_ref() {
                Some(run) => {
                    let s = &run.schedule;
                    format!(
                        " | {} a = {:.4} ({:.0}%{}){} | {}^2 particles{}",
                        run.model,
                        s.a,
                        100.0 * s.progress(),
                        if s.finished { ", done" } else { "" },
                        if run.science { " | SCIENCE RUN" } else { " | stirred" },
                        (run.np as f64).sqrt().round(),
                        if run.science_mode { format!(" | step {}", s.steps) } else { String::new() }
                    )
                }
                None => String::new(),
            };
            let pm = if self.cosmo_run.is_some() {
                match self.renderer.pm_ms() {
                    Some(t) => format!(" | pm {:.1} ms (periodic {}^2)", t, self.mesh),
                    None => format!(" | periodic pm {}^2", self.mesh),
                }
            } else {
                pm
            };
            let title = format!(
                "Gravity Sandbox | {}{} | {:.0} FPS | {:.2} ms | step {:.2} ms{} | bodies {} | largest {:.0} | dust {} | vsync {} | {}{}",
                self.world.scenario().name(),
                cosmo,
                fps,
                ms,
                st.step_ms,
                pm,
                st.bodies,
                st.largest_mass,
                self.renderer.dust_count(),
                if self.renderer.vsync() { "on" } else { "off" },
                self.renderer.adapter_info.name,
                if self.paused { " | PAUSED" } else { "" },
            );
            self.window.set_title(&title);
            self.hud_timer = 0.0;
            self.hud_frames = 0;
            self.hud_frame_time = 0.0;
        }

        self.timing_timer += real_dt;
        if self.timing_timer >= TIMING_PRINT_INTERVAL {
            self.timing_timer = 0.0;
            if let Some(ms) = self.renderer.stage_ms() {
                let mut line = String::from("gpu ms:");
                for (name, t) in timer::STAGE_NAMES.iter().zip(ms.iter()) {
                    line += &format!(" {name} {t:.2} |");
                }
                let st = &self.world.stats;
                println!("{line} cpu step {:.2} (evals/body {:.2}, bins {}) | bodies {} | dust {}", st.step_ms, st.force_evals_per_body, st.timestep_bins, st.bodies, self.renderer.dust_count());
            }
        }
    }

    fn key(&mut self, event_loop: &ActiveEventLoop, ev: &KeyEvent) {
        let PhysicalKey::Code(code) = ev.physical_key else {
            return;
        };
        let pressed = ev.state == ElementState::Pressed;
        match code {
            KeyCode::KeyW => self.keys_pan[0] = pressed,
            KeyCode::KeyA => self.keys_pan[1] = pressed,
            KeyCode::KeyS => self.keys_pan[2] = pressed,
            KeyCode::KeyD => self.keys_pan[3] = pressed,
            _ => {}
        }
        if !pressed || ev.repeat {
            return;
        }
        match code {
            KeyCode::Escape => event_loop.exit(),
            KeyCode::Space => {
                let at = self.input().cursor;
                self.world.spawn_cluster(at, [0.0, 0.0], CLUSTER_SIZE);
            }
            KeyCode::KeyR => self.reset(),
            KeyCode::KeyP => self.paused = !self.paused,
            KeyCode::KeyV => self.renderer.toggle_vsync(),
            KeyCode::KeyB => self.renderer.toggle_bloom(),
            KeyCode::Equal | KeyCode::NumpadAdd => self.change_dust(true),
            KeyCode::Minus | KeyCode::NumpadSubtract => self.change_dust(false),
            KeyCode::BracketLeft => self.change_bodies(false),
            KeyCode::BracketRight => self.change_bodies(true),
            KeyCode::Digit1 | KeyCode::Numpad1 => self.set_scenario(Scenario::Galaxy),
            KeyCode::Digit2 | KeyCode::Numpad2 => self.set_scenario(Scenario::SubhaloMerger),
            KeyCode::Digit3 | KeyCode::Numpad3 => self.set_scenario(Scenario::CosmicWeb),
            KeyCode::F11 => self.toggle_fullscreen(),
            KeyCode::KeyG => self.toggle_self_gravity(),
            _ => {}
        }
    }
}

struct App {
    settings: Settings,
    game: Option<Game>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.game.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("Gravity Sandbox")
            .with_inner_size(PhysicalSize::new(1600u32, 900u32))
            .with_resizable(true)
            .with_active(!self.settings.background)
            .with_fullscreen(if self.settings.fullscreen {
                Some(Fullscreen::Borderless(None))
            } else {
                None
            });
        let window = Arc::new(event_loop.create_window(attrs).expect("failed to create window"));
        let s = &self.settings;
        let renderer = pollster::block_on(Renderer::new(
            window.clone(),
            RendererOptions {
                dust: s.dust,
                vsync: s.vsync,
                bloom: s.bloom,
                pm_mesh: s.pm.then_some(s.mesh),
                pm_field: s.field,
            },
        ));
        let mut renderer = renderer;
        if s.cosmo.force_test {
            let side = cosmo_run::lattice_side(s.mesh, s.dust, renderer.max_dust());
            let h = cosmo::BOX / s.mesh as f64;
            let eps = s.cosmo.eps_frac * cosmo::BOX / side as f64;
            let radii: Vec<f64> = [0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0, 20.0, 40.0].iter().map(|c| c * h).collect();
            let rows = renderer.cosmo_force_test(s.mesh, eps, &radii);
            println!("GPU two-point force test: mesh {}^2 periodic, cell h = {h:.4}, r_s = 1.25 h, r_cut = 4.5 r_s, eps = {eps:.4} ({} x mean spacing {:.4})", s.mesh, s.cosmo.eps_frac, cosmo::BOX / side as f64);
            println!("   r/h   <a_P3M/a_exact>  max|a_P3M/a_exact-1|  <a_mesh_only/a_exact>   (8 directions / offsets)");
            let (mut worst_mean, mut worst_dir) = (0.0f64, 0.0f64);
            for r in &rows {
                if r[0] > 2.0 * eps {
                    worst_mean = worst_mean.max((r[1] - 1.0).abs());
                    worst_dir = worst_dir.max(r[2]);
                }
                println!("  {:6.2}   {:12.5}   {:14.3}%   {:16.5}", r[0] / h, r[1], r[2] * 100.0, r[3]);
            }
            println!("beyond 2 eps: worst direction-averaged error {:.3}%, worst single direction {:.3}%", worst_mean * 100.0, worst_dir * 100.0);
            let worst = worst_mean;
            std::process::exit(if worst < 0.02 { 0 } else { 1 });
        }
        if s.cosmo.determinism_test {
            let t = std::time::Instant::now();
            let (diff, n) = cosmo_run::determinism_test(&mut renderer, &s.cosmo, s.mesh, s.dust, 100);
            let (errs, lost) = renderer.gpu_health();
            println!(
                "P3M determinism test: mesh {}^2, {n} particles, 100 steps of dln a = 0.01 from a_init, run twice: {diff} particles differ ({:.1} s; GPU errors {errs}, device lost {lost})",
                s.mesh,
                t.elapsed().as_secs_f64()
            );
            std::process::exit(if diff == 0 && errs == 0 && !lost { 0 } else { 1 });
        }
        if s.pm_selftest {
            match renderer.pm_self_test() {
                Some((report, worst)) => {
                    println!("{report}worst relative error at r >= 3 cells: {:.3}%", worst * 100.0);
                    std::process::exit(if worst < 0.05 { 0 } else { 1 });
                }
                None => {
                    println!("PM disabled; nothing to test");
                    std::process::exit(1);
                }
            }
        }

        let info = &renderer.adapter_info;
        println!("Gravity Sandbox");
        println!(
            "Adapter: {} ({:?}, {:?}) driver: {} {}",
            info.name, info.backend, info.device_type, info.driver, info.driver_info
        );
        println!(
            "Present: {}",
            if renderer.vsync() { "Fifo (VSync)" } else { "uncapped (Mailbox/Immediate)" }
        );
        if s.dust > renderer.max_dust() {
            println!("Requested {} dust particles; adapter max is {}", s.dust, renderer.max_dust());
        }
        println!(
            "Settings: preset {:?} | scenario {} | bodies {} | dust {} (max {}) | vsync {} | bloom {} | fullscreen {} | PM mesh {} (field {}) | dark matter {} | timestamps {}",
            s.preset,
            s.scenario.name(),
            s.bodies,
            renderer.dust_count(),
            renderer.max_dust(),
            renderer.vsync(),
            renderer.bloom,
            s.fullscreen,
            renderer.pm_mesh().map_or("off".to_string(), |n| format!("{n}^2 (FFT {}^2)", 2 * n)),
            s.field,
            s.dark_matter.map_or("default".to_string(), |x| format!("x{x}")),
            renderer.stage_ms().is_some()
        );
        println!("{CONTROLS}");
        println!("\n{}", settings::HELP);

        let (w, h) = renderer.size();
        let world = World::with_scenario(s.scenario, s.bodies, INITIAL_SEED);
        let mut game = Game {
            window,
            renderer,
            world,
            camera: Camera::new(w, h),
            seed: INITIAL_SEED,
            scenario: s.scenario,
            n_bodies: s.bodies,
            attractors: [Attractor::default(); MAX_ATTRACTORS],
            cursor_px: [w as f32 * 0.5, h as f32 * 0.5],
            left: false,
            right: false,
            middle: false,
            keys_pan: [false; 4],
            paused: false,
            dark_matter: s.dark_matter,
            timing_timer: 0.0,
            cosmo_settings: s.cosmo.clone(),
            mesh: s.mesh,
            cosmo_particles: s.dust,
            cosmo_run: None,
            experiment: None,
            quit: false,
            last_frame: Instant::now(),
            hud_timer: 0.0,
            hud_frames: 0,
            hud_frame_time: 0.0,
        };
        if game.scenario == Scenario::CosmicWeb {
            game.start_cosmo_run();
        } else {
            game.spin_up_for_dark_matter();
            game.reseed_dust();
        }
        game.last_frame = Instant::now();
        self.game = Some(game);
        event_loop.set_control_flow(ControlFlow::Poll);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(game) = self.game.as_mut() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                game.renderer.resize(size.width, size.height);
                if size.width > 0 && size.height > 0 {
                    game.camera.resize(size.width, size.height);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => game.key(event_loop, &event),
            WindowEvent::CursorMoved { position, .. } => {
                let (x, y) = (position.x as f32, position.y as f32);
                if game.middle {
                    game.camera
                        .pan_pixels(x - game.cursor_px[0], y - game.cursor_px[1]);
                }
                game.cursor_px = [x, y];
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let down = state == ElementState::Pressed;
                match button {
                    MouseButton::Left => game.left = down,
                    MouseButton::Right => game.right = down,
                    MouseButton::Middle => game.middle = down,
                    _ => {}
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let steps = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 60.0,
                };
                let factor = 0.85f32.powf(steps);
                game.camera
                    .zoom_at(game.cursor_px[0], game.cursor_px[1], factor);
            }
            WindowEvent::Focused(false) => {
                game.left = false;
                game.right = false;
                game.middle = false;
                game.keys_pan = [false; 4];
            }
            WindowEvent::RedrawRequested => {
                game.frame();
                if game.quit {
                    event_loop.exit();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(game) = self.game.as_ref() {
            game.window.request_redraw();
        }
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
        .init();

    let settings = match settings::parse_args(std::env::args().skip(1)) {
        Ok(ParseOutcome::Run(s)) => s,
        Ok(ParseOutcome::Help) => {
            println!("{}", settings::HELP);
            return;
        }
        Err(e) => {
            eprintln!("error: {e}\n\n{}", settings::HELP);
            std::process::exit(2);
        }
    };

    let event_loop = EventLoop::new().expect("failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        settings,
        game: None,
    };
    event_loop.run_app(&mut app).expect("event loop error");
}

