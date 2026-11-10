//! Render PNG snapshots of a scenario over time (visual tuning aid, no GPU needed).
//!
//! `cargo run --release --example snapshot -- subhalo 20000 OUT_DIR 0 10 20 40 60`
//! writes OUT_DIR/subhalo_20000_t010.png etc. (whole world, 800x800, additive splats).

use sim_core::{Input, Scenario, WORLD_HALF, World};
use std::io::Write;

const SIZE: usize = 800;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let sc = Scenario::from_name(&args[0]).expect("scenario");
    let n: usize = args[1].parse().expect("n");
    let out = std::path::PathBuf::from(&args[2]);
    let times: Vec<f32> = args[3..].iter().map(|s| s.parse().unwrap()).collect();
    std::fs::create_dir_all(&out).unwrap();
    let mut w = World::with_scenario(sc, n, 1);
    let mut frame = 0usize;
    for t in times {
        while (frame as f32) < t * 60.0 {
            w.step(1.0 / 60.0, &Input::default());
            frame += 1;
        }
        let path = out.join(format!("{}_{}_t{:03}.png", sc.name(), n, t as u32));
        write_png(&path, &render(&w));
        println!("{} bodies {} -> {}", t, w.bodies.len(), path.display());
    }
}

fn hue_rgb(h: f32) -> [f32; 3] {
    let h = h.rem_euclid(1.0) * 6.0;
    let x = 1.0 - ((h % 2.0) - 1.0).abs();
    match h as u32 {
        0 => [1.0, x, 0.0],
        1 => [x, 1.0, 0.0],
        2 => [0.0, 1.0, x],
        3 => [0.0, x, 1.0],
        4 => [x, 0.0, 1.0],
        _ => [1.0, 0.0, x],
    }
}

fn render(w: &World) -> Vec<u8> {
    let mut acc = vec![[0.0f32; 3]; SIZE * SIZE];
    let scale = SIZE as f32 / (2.0 * WORLD_HALF);
    let light = (w.bodies.len() as f32 / 4000.0).max(1.0);
    for b in &w.bodies {
        let c = hue_rgb(b.hue);
        let px = (b.pos[0] + WORLD_HALF) * scale;
        let py = (WORLD_HALF - b.pos[1]) * scale;
        let r = b.radius * scale;
        let (gain, rr) = if r < 1.0 { (0.6 / light.sqrt(), 0.0) } else { (1.0, r) };
        let (x0, x1) = ((px - rr).floor() as i64, (px + rr).ceil() as i64);
        let (y0, y1) = ((py - rr).floor() as i64, (py + rr).ceil() as i64);
        for y in y0..=y1 {
            for x in x0..=x1 {
                if x < 0 || y < 0 || x >= SIZE as i64 || y >= SIZE as i64 {
                    continue;
                }
                let (dx, dy) = (x as f32 + 0.5 - px, y as f32 + 0.5 - py);
                if rr > 0.0 && dx * dx + dy * dy > rr * rr {
                    continue;
                }
                let p = &mut acc[y as usize * SIZE + x as usize];
                for k in 0..3 {
                    p[k] += c[k] * gain;
                }
            }
        }
    }
    let mut img = Vec::with_capacity(SIZE * SIZE * 3);
    for p in acc {
        for v in p {
            img.push((255.0 * (1.0 - (-v).exp())) as u8);
        }
    }
    img
}

fn crc32(data: &[u8]) -> u32 {
    let mut c = 0xFFFF_FFFFu32;
    for &b in data {
        c ^= b as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
        }
    }
    !c
}

/// Minimal PNG writer: RGB8, zlib "stored" (uncompressed) deflate blocks.
fn write_png(path: &std::path::Path, rgb: &[u8]) {
    let mut raw = Vec::with_capacity((SIZE * 3 + 1) * SIZE);
    for row in rgb.chunks(SIZE * 3) {
        raw.push(0);
        raw.extend_from_slice(row);
    }
    let mut z = vec![0x78, 0x01];
    let chunks: Vec<&[u8]> = raw.chunks(65535).collect();
    for (i, ch) in chunks.iter().enumerate() {
        z.push((i == chunks.len() - 1) as u8);
        let len = ch.len() as u16;
        z.extend_from_slice(&len.to_le_bytes());
        z.extend_from_slice(&(!len).to_le_bytes());
        z.extend_from_slice(ch);
    }
    let (mut a, mut b) = (1u32, 0u32);
    for &x in &raw {
        a = (a + x as u32) % 65521;
        b = (b + a) % 65521;
    }
    z.extend_from_slice(&((b << 16) | a).to_be_bytes());

    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(b"\x89PNG\r\n\x1a\n").unwrap();
    let mut chunk = |ty: &[u8], data: &[u8]| {
        f.write_all(&(data.len() as u32).to_be_bytes()).unwrap();
        let mut c = ty.to_vec();
        c.extend_from_slice(data);
        f.write_all(&c).unwrap();
        f.write_all(&crc32(&c).to_be_bytes()).unwrap();
    };
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&(SIZE as u32).to_be_bytes());
    ihdr.extend_from_slice(&(SIZE as u32).to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
    chunk(b"IHDR", &ihdr);
    chunk(b"IDAT", &z);
    chunk(b"IEND", &[]);
}
