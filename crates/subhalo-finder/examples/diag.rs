//! Diagnostics on a real snapshot: virial ratios of FoF groups by size and enclosed-density
//! profiles of the largest groups. `cargo run --release --example diag -- <run_dir> <a>`

use subhalo_finder::snapshot::Snapshot;
use subhalo_finder::units::{Cosmo, RunMeta};
use subhalo_finder::{Params, fof};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = std::path::PathBuf::from(&args[0]);
    let a: f64 = args[1].parse().unwrap();
    let meta = RunMeta::load(&dir, None).unwrap();
    let si = meta.snapshots.iter().find(|s| (s.a - a).abs() < 1e-3).unwrap();
    let snap = Snapshot::open(&si.path).unwrap();
    let cosmo = Cosmo::new(&meta, a);
    let (pos, vel) = (snap.positions(), snap.velocities());
    let p = Params::default();
    let link = p.link_b * cosmo.box_size / (cosmo.n_total as f64).sqrt();
    let groups = fof::fof(pos, cosmo.box_size, link, 20).unwrap();
    let l = cosmo.box_size;
    let g2 = cosmo.g_log();
    println!("g_log (C m /(2 pi a)) = {g2:.4e}; link {link:.4}");
    // virial ratio per size bin: <|v - vbar|^2> / (g_log N / 2), plus extent and mean overdensity
    let bins = [20usize, 50, 100, 300, 1000, 3000, 10000, 100000, 1000000];
    for w in bins.windows(2) {
        let sel: Vec<usize> = (0..groups.len()).filter(|&g| (w[0]..w[1]).contains(&groups.size(g))).collect();
        if sel.is_empty() {
            continue;
        }
        let mut ratios = Vec::new();
        let mut deltas = Vec::new();
        for &g in sel.iter().take(400) {
            let m = groups.group(g);
            let r0 = pos[m[0] as usize];
            let lp: Vec<[f64; 2]> = m
                .iter()
                .map(|&i| {
                    let q = pos[i as usize];
                    let mut dx = q[0] as f64 - r0[0] as f64;
                    let mut dy = q[1] as f64 - r0[1] as f64;
                    dx -= l * (dx / l).round();
                    dy -= l * (dy / l).round();
                    [dx, dy]
                })
                .collect();
            let n = m.len() as f64;
            let cx = lp.iter().map(|q| q[0]).sum::<f64>() / n;
            let cy = lp.iter().map(|q| q[1]).sum::<f64>() / n;
            let vx = m.iter().map(|&i| vel[i as usize][0] as f64).sum::<f64>() / n;
            let vy = m.iter().map(|&i| vel[i as usize][1] as f64).sum::<f64>() / n;
            let v2 = m
                .iter()
                .map(|&i| (vel[i as usize][0] as f64 - vx).powi(2) + (vel[i as usize][1] as f64 - vy).powi(2))
                .sum::<f64>()
                / n;
            let mut r2: Vec<f64> = lp.iter().map(|q| (q[0] - cx).powi(2) + (q[1] - cy).powi(2)).collect();
            r2.sort_by(f64::total_cmp);
            let rh = r2[r2.len() / 2].sqrt();
            ratios.push(v2 / (g2 * n / 2.0));
            // mean overdensity inside the half-number radius
            deltas.push(0.5 * n * cosmo.m_area / (std::f64::consts::PI * rh * rh));
        }
        ratios.sort_by(f64::total_cmp);
        deltas.sort_by(f64::total_cmp);
        let med = |v: &Vec<f64>| v[v.len() / 2];
        println!(
            "n in [{}, {}): {} groups; median <v^2>/(g N/2) = {:.2} (p10 {:.2}, p90 {:.2}); median mean density inside r_half = {:.1} x mean",
            w[0],
            w[1],
            sel.len(),
            med(&ratios),
            ratios[ratios.len() / 10],
            ratios[ratios.len() * 9 / 10],
            med(&deltas)
        );
    }
    // Enclosed mean-density profile of the 3 largest groups around their densest region
    // (centre of mass of the innermost 1% by distance from the COM, iterated).
    for g in 0..3 {
        let m = groups.group(g);
        let r0 = pos[m[0] as usize];
        let lp: Vec<[f64; 2]> = m
            .iter()
            .map(|&i| {
                let q = pos[i as usize];
                let mut dx = q[0] as f64 - r0[0] as f64;
                let mut dy = q[1] as f64 - r0[1] as f64;
                dx -= l * (dx / l).round();
                dy -= l * (dy / l).round();
                [dx, dy]
            })
            .collect();
        let mut c = [lp.iter().map(|q| q[0]).sum::<f64>() / lp.len() as f64, lp.iter().map(|q| q[1]).sum::<f64>() / lp.len() as f64];
        for frac in [0.5, 0.2, 0.05, 0.01] {
            let mut d: Vec<(f64, usize)> = lp.iter().enumerate().map(|(k, q)| ((q[0] - c[0]).powi(2) + (q[1] - c[1]).powi(2), k)).collect();
            d.sort_by(|x, y| x.0.total_cmp(&y.0));
            let k = ((lp.len() as f64 * frac) as usize).max(10);
            c = [d[..k].iter().map(|e| lp[e.1][0]).sum::<f64>() / k as f64, d[..k].iter().map(|e| lp[e.1][1]).sum::<f64>() / k as f64];
        }
        let mut r2: Vec<f64> = lp.iter().map(|q| (q[0] - c[0]).powi(2) + (q[1] - c[1]).powi(2)).collect();
        r2.sort_by(f64::total_cmp);
        print!("group {g} (n={}): enclosed mean overdensity at r =", m.len());
        for r in [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0] {
            let k = r2.partition_point(|&q| q < r * r);
            print!(" {r}:{:.0}", k as f64 * cosmo.m_area / (std::f64::consts::PI * r * r));
        }
        println!();
        // Candidate / unbinding trace.
        let fp: Vec<[f32; 2]> = lp.iter().map(|q| [q[0] as f32, q[1] as f32]).collect();
        let fv: Vec<[f32; 2]> = m.iter().map(|&i| vel[i as usize]).collect();
        let tree = subhalo_finder::kdtree::KdTree::build(&fp);
        let (rho, ngb) = subhalo_finder::subfind::densities(&tree, &fp, p.k_dens, p.n_ngb);
        let cands = subhalo_finder::subfind::candidates(&rho, &ngb, p.n_ngb, p.n_min);
        let mut sizes: Vec<usize> = cands.iter().map(|c| c.len()).collect();
        sizes.sort_unstable();
        println!(
            "  {} candidates, largest {:?}",
            cands.len(),
            &sizes[sizes.len().saturating_sub(5)..]
        );
        let t = std::time::Instant::now();
        let subs = subhalo_finder::subfind::find_in_group(&fp, &fv, &cosmo, &p);
        let sz: Vec<usize> = subs.iter().take(8).map(|s| s.members.len()).collect();
        println!("  {} subhalos, largest {:?} ({:.2}s)", subs.len(), sz, t.elapsed().as_secs_f64());
    }
}
