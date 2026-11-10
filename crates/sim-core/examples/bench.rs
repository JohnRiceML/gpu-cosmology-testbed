//! Step-time benchmark.
//!
//! ```text
//! cargo run --release --example bench                      # table: both scenarios, 4k/10k/20k/50k
//! cargo run --release --example bench -- subhalo 20000     # one scenario, one or more counts
//! cargo run --release --example bench -- galaxy 4000 10000 --steps 300
//! cargo run --release --example bench -- subhalo 50000 --uniform   # force uniform substeps
//! cargo run --release --example bench -- subhalo 50000 --block     # force block timesteps
//! cargo run --release --example bench -- subhalo 50000 --field     # + synthetic 256^2 field
//! cargo run --release --example bench -- all 50000 --well          # mouse well held on
//! cargo run --release --example bench -- subhalo 20000 --timeline 120
//! ```
//! Default integrator is automatic (block timesteps with Barnes-Hut, uniform otherwise).
//! `--field` installs a point-mass-like external field (mass `FIELD_MASS`, softening 60)
//! sampled on a 256x256 grid, re-set every frame like the GPU readback would be.
//! `fast` = bodies faster than `FAST_SPEED` at the end (bodies flung out by close
//! encounters; the initial maximum speed is ~40-60).
//! `--timeline SECS` simulates SECS seconds of game time at 60 Hz and prints the state every
//! 5 s (bodies, merges, heavy bodies and their distance from the centre).

use sim_core::{ExternalField, G, Input, Scenario, WORLD_HALF, World};

const FAST_SPEED: f32 = 150.0;
const FIELD_MASS: f32 = 3000.0;

#[derive(Clone, Copy)]
struct Opts {
    steps: usize,
    well: bool,
    field: bool,
    block: Option<bool>,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut scenarios = Vec::new();
    let mut counts = Vec::new();
    let mut opts = Opts { steps: 200, well: false, field: false, block: None };
    let mut timeline: Option<f32> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--steps" => opts.steps = it.next().and_then(|s| s.parse().ok()).expect("--steps N"),
            "--timeline" => timeline = it.next().and_then(|s| s.parse().ok()),
            "--well" => opts.well = true,
            "--field" => opts.field = true,
            "--uniform" => opts.block = Some(false),
            "--block" => opts.block = Some(true),
            "all" => scenarios.extend(Scenario::ALL),
            s => {
                if let Some(sc) = Scenario::from_name(s) {
                    scenarios.push(sc);
                } else if let Ok(n) = s.replace('_', "").parse::<usize>() {
                    counts.push(n);
                } else {
                    eprintln!(
                        "unknown argument {s:?} (expected galaxy|subhalo|all, a count, --steps N, \
                         --uniform, --block, --field, --well, --timeline SECS)"
                    );
                    std::process::exit(2);
                }
            }
        }
    }
    if scenarios.is_empty() {
        scenarios.extend(Scenario::ALL);
    }
    if counts.is_empty() {
        counts.extend([4_000, 10_000, 20_000, 50_000]);
    }

    if let Some(secs) = timeline {
        for &sc in &scenarios {
            for &n in &counts {
                run_timeline(sc, n, secs, opts);
            }
        }
        return;
    }

    println!(
        "{:<8} {:>7} {:>5} {:>8} {:>8} {:>8} {:>8} {:>10} {:>5} {:>7} {:>7} {:>5}",
        "scenario", "n", "tree", "build", "first", "avg ms", "max ms", "evals/body", "bins", "end n",
        "merges", "fast"
    );
    for &sc in &scenarios {
        for &n in &counts {
            bench(sc, n, opts);
        }
    }
}

fn input(opts: Opts) -> Input {
    // `--well`: hold the attracting mouse well at (200, 200) the whole time, which pulls
    // everything into one dense knot (worst case for collisions and timestep bins).
    if opts.well {
        Input { cursor: [200.0, 200.0], well: 1.0 }
    } else {
        Input::default()
    }
}

/// Synthetic stand-in for the GPU particle-mesh field: a softened point mass at the origin.
fn synthetic_field(size: usize) -> ExternalField {
    let cell = 2.0 * WORLD_HALF / size as f32;
    let mut acc = Vec::with_capacity(size * size);
    for iy in 0..size {
        for ix in 0..size {
            let x = -WORLD_HALF + (ix as f32 + 0.5) * cell;
            let y = -WORLD_HALF + (iy as f32 + 0.5) * cell;
            let d2 = x * x + y * y + 60.0 * 60.0;
            let s = -G * FIELD_MASS / (d2 * d2.sqrt());
            acc.push([x * s, y * s]);
        }
    }
    ExternalField { size, acc }
}

fn new_world(sc: Scenario, n: usize, opts: Opts) -> World {
    let mut w = World::with_scenario(sc, n, 1);
    w.force_integrator(opts.block);
    if opts.field {
        w.set_external_field(Some(synthetic_field(256)));
    }
    w
}

fn frame(w: &mut World, opts: Opts) {
    if opts.field {
        // Mimic the renderer handing over a new field every frame (allocation reused).
        let f = w.take_external_field();
        w.set_external_field(f);
    }
    w.step(1.0 / 60.0, &input(opts));
}

fn fast(w: &World) -> usize {
    w.bodies.iter().filter(|b| b.vel[0].hypot(b.vel[1]) > FAST_SPEED).count()
}

fn bench(sc: Scenario, n: usize, opts: Opts) {
    let t = std::time::Instant::now();
    let mut w = new_world(sc, n, opts);
    let build_ms = t.elapsed().as_secs_f32() * 1000.0;
    // The first step includes the extra force evaluation that seeds the cached
    // accelerations (and rayon pool start-up on the very first run).
    frame(&mut w, opts);
    let first = w.stats.step_ms;
    for _ in 0..5 {
        frame(&mut w, opts);
    }
    let (mut sum, mut max, mut evals, mut bins) = (0.0f32, 0.0f32, 0.0f32, 0u32);
    for _ in 0..opts.steps {
        frame(&mut w, opts);
        sum += w.stats.step_ms;
        max = max.max(w.stats.step_ms);
        evals += w.stats.force_evals_per_body;
        bins = bins.max(w.stats.timestep_bins);
    }
    let steps = opts.steps as f32;
    println!(
        "{:<8} {:>7} {:>5} {:>8.2} {:>8.3} {:>8.3} {:>8.3} {:>10.3} {:>5} {:>7} {:>7} {:>5}",
        sc.name(),
        n,
        w.stats.tree,
        build_ms,
        first,
        sum / steps,
        max,
        evals / steps,
        bins,
        w.stats.bodies,
        w.stats.merges_total,
        fast(&w)
    );
}

fn run_timeline(sc: Scenario, n: usize, secs: f32, opts: Opts) {
    let mut w = new_world(sc, n, opts);
    println!("== {} n={} (built {} bodies)", sc.name(), n, w.bodies.len());
    let extent = w
        .bodies
        .iter()
        .map(|b| (b.pos[0].abs() + b.radius).max(b.pos[1].abs() + b.radius))
        .fold(0.0f32, f32::max);
    println!("initial half-extent (max |x|,|y| incl. radius): {extent:.0}");
    let frames = (secs * 60.0) as usize;
    let (mut ms_sum, mut ms_max, mut ev_sum, mut win_frames) = (0.0f32, 0.0f32, 0.0f32, 0usize);
    for f in 0..=frames {
        if f % 300 == 0 || f == 60 {
            let mut heavy: Vec<(f32, f32)> = w
                .bodies
                .iter()
                .filter(|b| b.mass >= 40.0)
                .map(|b| (b.mass, b.pos[0].hypot(b.pos[1])))
                .collect();
            heavy.sort_by(|a, b| b.0.total_cmp(&a.0));
            let heavy: Vec<String> = heavy
                .iter()
                .take(12)
                .map(|(m, r)| format!("{m:.0}@{r:.0}"))
                .collect();
            let wf = win_frames.max(1) as f32;
            // Subhalo hues: host family is gold (< 0.2), satellites 0.27..0.97.
            let host = w.bodies.iter().filter(|b| b.hue < 0.2).count();
            println!(
                "t={:5.1}s bodies {:6} (host-hue {:6}) merges {:6} fast {:4} step avg/max {:.2}/{:.2} ms, \
                 {:.2} evals/body, {} bins | heavy (mass@radius): {}",
                f as f32 / 60.0,
                w.stats.bodies,
                host,
                w.stats.merges_total,
                fast(&w),
                ms_sum / wf,
                ms_max,
                ev_sum / wf,
                w.stats.timestep_bins,
                heavy.join(" ")
            );
            (ms_sum, ms_max, ev_sum, win_frames) = (0.0, 0.0, 0.0, 0);
        }
        if f < frames {
            frame(&mut w, opts);
            ms_sum += w.stats.step_ms;
            ms_max = ms_max.max(w.stats.step_ms);
            ev_sum += w.stats.force_evals_per_body;
            win_frames += 1;
        }
    }
}
