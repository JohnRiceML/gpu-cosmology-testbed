//! `GSNAP001` snapshot files, memory-mapped (adapted from cosmo-analysis/src/snapshot.rs).
//!
//! Layout (little-endian): magic[8] | u64 count | f32 a | f32 box |
//! count x [f32 x, f32 y] comoving positions in [0, box) | count x [f32 vx, f32 vy].
//! The particle index is its stable ID across snapshots.

#[cfg(not(target_endian = "little"))]
compile_error!("snapshot reader casts little-endian file bytes directly; big-endian hosts unsupported");

use memmap2::Mmap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

pub const MAGIC: &[u8; 8] = b"GSNAP001";
pub const HEADER: usize = 24;

pub struct Snapshot {
    mmap: Mmap,
    pub count: usize,
    pub a: f32,
    pub box_size: f32,
}

impl Snapshot {
    pub fn open(path: &Path) -> Result<Snapshot, String> {
        let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        // SAFETY: read-only map; if another process truncates the file while we read we may
        // fault, the standard caveat of memory-mapped input.
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| format!("{}: {e}", path.display()))?;
        if mmap.len() < HEADER || &mmap[0..8] != MAGIC {
            return Err(format!("{}: not a GSNAP001 file", path.display()));
        }
        let count = u64::from_le_bytes(mmap[8..16].try_into().unwrap()) as usize;
        let a = f32::from_le_bytes(mmap[16..20].try_into().unwrap());
        let box_size = f32::from_le_bytes(mmap[20..24].try_into().unwrap());
        let need = HEADER + count * 16;
        if mmap.len() < need {
            return Err(format!(
                "{}: incomplete ({} of {need} bytes) - still being written?",
                path.display(),
                mmap.len()
            ));
        }
        Ok(Snapshot { mmap, count, a, box_size })
    }

    pub fn positions(&self) -> &[[f32; 2]] {
        bytemuck::cast_slice(&self.mmap[HEADER..HEADER + self.count * 8])
    }

    pub fn velocities(&self) -> &[[f32; 2]] {
        bytemuck::cast_slice(&self.mmap[HEADER + self.count * 8..HEADER + self.count * 16])
    }
}

/// Write a snapshot (used by tests and synthetic data).
pub fn write_snapshot(path: &Path, a: f32, box_size: f32, pos: &[[f32; 2]], vel: &[[f32; 2]]) -> std::io::Result<()> {
    assert_eq!(pos.len(), vel.len());
    let tmp = path.with_extension("bin.part");
    {
        let mut w = BufWriter::with_capacity(8 << 20, File::create(&tmp)?);
        w.write_all(MAGIC)?;
        w.write_all(&(pos.len() as u64).to_le_bytes())?;
        w.write_all(&a.to_le_bytes())?;
        w.write_all(&box_size.to_le_bytes())?;
        w.write_all(bytemuck::cast_slice(pos))?;
        w.write_all(bytemuck::cast_slice(vel))?;
        w.flush()?;
    }
    std::fs::rename(&tmp, path)
}
