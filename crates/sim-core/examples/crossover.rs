//! Direct-sum vs Barnes-Hut timing per force evaluation, to pick the crossover.
//! `cargo run --release --example crossover`

use sim_core::{Scenario, World, bench_gravity_ms};

fn main() {
    println!("{:<8} {:>7} {:>10} {:>10}", "scenario", "n", "direct ms", "tree ms");
    for sc in Scenario::ALL {
        for n in [1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 20000, 50000] {
            let w = World::with_scenario(sc, n, 1);
            let reps = (2_000_000_000 / (n * n)).clamp(3, 200);
            let d = bench_gravity_ms(&w.bodies, false, reps);
            let t = bench_gravity_ms(&w.bodies, true, reps);
            println!("{:<8} {:>7} {:>10.3} {:>10.3}", sc.name(), n, d, t);
        }
    }
}
