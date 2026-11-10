//! Run directory + background writer for the cosmology run (P(k) table and particle snapshots).
//!
//! Files (contract with the analysis side): meta.json, pk.csv, snap_aX.XXX.bin, run.log.

use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;

use crate::cosmo::{BOX, ModeTable, PK_BINS};

pub enum Job {
    /// |δ_k|² / W_cic² on the full N x N mode grid (index ky*N + kx) measured at `a`.
    Pk { a: f64, power: Vec<f32> },
    /// Start a snapshot of `count` particles synchronized at `a`.
    SnapBegin { a: f64, count: u64 },
    /// Particles [offset, offset + data.len()) as (s.xy displacement, p.xy) records.
    SnapChunk { offset: u64, data: Vec<[f32; 4]> },
    /// All chunks sent.
    SnapEnd { a: f64 },
}

pub struct RunWriter {
    pub dir: PathBuf,
    tx: Option<Sender<Job>>,
    handle: Option<JoinHandle<()>>,
    complete: Arc<AtomicBool>,
    failed: Arc<AtomicBool>,
}

/// Minimum free space on the output drive before a snapshot is written.
pub const MIN_FREE_BYTES: u64 = 10 * 1024 * 1024 * 1024;

/// Free bytes available to the caller on the volume holding `path`.
#[cfg(windows)]
pub fn free_bytes(path: &Path) -> Option<u64> {
    use std::os::windows::ffi::OsStrExt;
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetDiskFreeSpaceExW(dir: *const u16, avail: *mut u64, total: *mut u64, free: *mut u64) -> i32;
    }
    let w: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
    let (mut a, mut t, mut f) = (0u64, 0u64, 0u64);
    // SAFETY: w is NUL-terminated; the out-pointers are valid u64s.
    let ok = unsafe { GetDiskFreeSpaceExW(w.as_ptr(), &mut a, &mut t, &mut f) };
    (ok != 0).then_some(a)
}

#[cfg(not(windows))]
pub fn free_bytes(_path: &Path) -> Option<u64> {
    None
}

/// Static facts the writer needs.
pub struct RunInfo {
    pub a_init: f64,
    pub particles: u64,
    /// lattice side: particle i has lattice point q(i) and the GPU stores s = x - q
    pub side: u32,
    pub snapshot_names: Vec<(f64, String)>,
}

/// UTC timestamp "YYYYMMDD-HHMMSS" (std only; civil-from-days, H. Hinnant).
pub fn utc_stamp() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}{m:02}{d:02}-{:02}{:02}{:02}", rem / 3600, (rem / 60) % 60, rem % 60)
}

pub fn snapshot_name(a: f64) -> String {
    format!("snap_a{a:.3}.bin")
}

impl RunWriter {
    /// Create `root/<UTC stamp>/`, write meta.json and the pk.csv header, start the thread.
    pub fn start(root: &Path, meta_json: &str, table: ModeTable, info: RunInfo) -> std::io::Result<Self> {
        let mut dir = root.join(utc_stamp());
        let mut k = 1;
        while dir.exists() {
            dir = root.join(format!("{}-{k}", utc_stamp()));
            k += 1;
        }
        Self::start_at(&dir, meta_json, table, info, None)
    }

    /// Start writing into exactly `dir` (created; a stale COMPLETE marker is removed).
    /// `resume_a`: restart from the snapshot at that a (keeps snapshots with a <= resume_a and
    /// the P(k) rows with a < resume_a; everything later is removed).
    pub fn start_at(dir: &Path, meta_json: &str, table: ModeTable, info: RunInfo, resume_a: Option<f64>) -> std::io::Result<Self> {
        let dir = dir.to_path_buf();
        fs::create_dir_all(&dir)?;
        let _ = fs::remove_file(dir.join("COMPLETE"));
        // remove snapshots of an earlier, interrupted attempt (all of them for a fresh run, the ones
        // after the restart epoch when resuming)
        let mut kept = 0usize;
        if let Ok(rd) = fs::read_dir(&dir) {
            for e in rd.flatten() {
                let n = e.file_name().to_string_lossy().to_string();
                if n.starts_with("snap_a") && n.ends_with(".bin") {
                    let keep = match (resume_a, snapshot_header(&e.path(), info.particles)) {
                        (Some(ar), Some(a)) => a <= ar + 1e-6,
                        _ => false,
                    };
                    if keep {
                        kept += 1;
                    } else {
                        let _ = fs::remove_file(e.path());
                    }
                } else if n == "run.log" && resume_a.is_none() {
                    let _ = fs::remove_file(e.path());
                }
            }
        }
        fs::write(dir.join("meta.json"), meta_json)?;
        let pk_path = dir.join("pk.csv");
        let old_rows: Vec<String> = match resume_a {
            Some(ar) => fs::read_to_string(&pk_path)
                .unwrap_or_default()
                .lines()
                .skip(1)
                .filter(|l| l.split(',').next().and_then(|a| a.parse::<f64>().ok()).is_some_and(|a| a < ar - 1e-7))
                .map(String::from)
                .collect(),
            None => Vec::new(),
        };
        let mut pk = BufWriter::new(File::create(&pk_path)?);
        writeln!(pk, "a,k,P_measured,P_linear,n_modes")?;
        for l in &old_rows {
            writeln!(pk, "{l}")?;
        }
        pk.flush()?;
        if let Some(ar) = resume_a {
            log_line(&dir, &format!("  RESTART from snapshot a = {ar:.6} ({kept} snapshots and {} P(k) rows kept) after an interrupted attempt (GPU device lost)", old_rows.len()));
        }
        println!("cosmo run directory: {}", dir.display());
        println!("  wrote {}", dir.join("meta.json").display());

        let (tx, rx) = channel();
        let thread_dir = dir.clone();
        let complete = Arc::new(AtomicBool::new(false));
        let failed = Arc::new(AtomicBool::new(false));
        let (c2, f2) = (complete.clone(), failed.clone());
        let handle = std::thread::Builder::new()
            .name("cosmo-writer".into())
            .spawn(move || writer_thread(rx, thread_dir, pk, table, info, c2, f2, kept))?;
        Ok(Self {
            dir,
            tx: Some(tx),
            handle: Some(handle),
            complete,
            failed,
        })
    }

    /// The run reached a_final and all readbacks were handed over: write COMPLETE when the
    /// queue drains without errors.
    pub fn mark_complete(&self) {
        self.complete.store(true, Ordering::Release);
    }

    /// Append a timestamped progress line to run.log (main thread, small).
    pub fn log(&self, line: &str) {
        if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(self.dir.join("run.log")) {
            let _ = writeln!(f, "[{}] {line}", utc_stamp());
        }
    }

    /// A write failed or the disk is below MIN_FREE_BYTES.
    pub fn failed(&self) -> bool {
        self.failed.load(Ordering::Acquire)
    }

    pub fn send(&self, job: Job) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(job);
        }
    }
}

impl Drop for RunWriter {
    fn drop(&mut self) {
        self.tx = None; // closes the channel; the thread drains pending jobs and exits
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn log_line(dir: &Path, line: &str) {
    println!("{line}");
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(dir.join("run.log")) {
        let _ = writeln!(f, "{line}");
    }
}

fn writer_thread(
    rx: Receiver<Job>,
    dir: PathBuf,
    mut pk: BufWriter<File>,
    table: ModeTable,
    info: RunInfo,
    complete: Arc<AtomicBool>,
    failed: Arc<AtomicBool>,
    kept_snapshots: usize,
) {
    let mut snaps_written = kept_snapshots;
    let mut snap: Option<SnapFile> = None;
    let shot = BOX * BOX / info.particles as f64;
    let mut n_pk = 0usize;
    for job in rx {
        match job {
            Job::Pk { a, power } => {
                let mut sum = [0f64; PK_BINS];
                let mut cnt = [0u64; PK_BINS];
                for (i, &p) in power.iter().enumerate() {
                    let b = table.bin_of[i];
                    if b != u16::MAX && p.is_finite() {
                        sum[b as usize] += p as f64;
                        cnt[b as usize] += 1;
                    }
                }
                let growth = (a / info.a_init) * (a / info.a_init);
                let mut low = Vec::new();
                for b in 0..PK_BINS {
                    if cnt[b] == 0 {
                        continue;
                    }
                    let pm = BOX * BOX * sum[b] / cnt[b] as f64 - shot;
                    let pl = growth * table.p0_mean[b];
                    let _ = writeln!(pk, "{a:.6},{:.6e},{pm:.6e},{pl:.6e},{}", table.k_mean[b], cnt[b]);
                    if b < 6 {
                        low.push(pm / pl);
                    }
                }
                let _ = pk.flush();
                if n_pk % 10 == 0 {
                    let r: Vec<String> = low.iter().map(|r| format!("{r:.3}")).collect();
                    log_line(&dir, &format!(
                        "  pk.csv: a = {a:.4} (measurement {n_pk}); low-k P_measured/P_linear = [{}]",
                        r.join(", ")
                    ));
                }
                n_pk += 1;
            }
            Job::SnapBegin { a, count } => {
                snap = None;
                let name = info
                    .snapshot_names
                    .iter()
                    .min_by(|x, y| (x.0 - a).abs().total_cmp(&(y.0 - a).abs()))
                    .map(|x| x.1.clone())
                    .unwrap_or_else(|| snapshot_name(a));
                let path = dir.join(&name);
                let need = 24 + count * 16 + MIN_FREE_BYTES;
                if let Some(free) = free_bytes(&dir) {
                    if free < need {
                        failed.store(true, Ordering::Release);
                        log_line(&dir, &format!("  NOT writing {}: only {:.1} GB free on the output drive (need >= 10 GB to remain); stopping the run", path.display(), free as f64 / 1e9));
                        continue;
                    }
                }
                match SnapFile::create(&path, a, count, info.side) {
                    Ok(s) => snap = Some(s),
                    Err(e) => {
                        failed.store(true, Ordering::Release);
                        log_line(&dir, &format!("  snapshot {} FAILED: {e}", path.display()));
                    }
                }
            }
            Job::SnapChunk { offset, data } => {
                if let Some(s) = snap.as_mut() {
                    if let Err(e) = s.write_chunk(offset, &data) {
                        failed.store(true, Ordering::Release);
                        log_line(&dir, &format!("  snapshot {} FAILED: {e}", s.path.display()));
                        snap = None;
                    }
                }
            }
            Job::SnapEnd { a } => {
                if let Some(s) = snap.take() {
                    let path = s.path.clone();
                    match s.finish() {
                        Ok((mom, nans, vrms, count)) => {
                            snaps_written += 1;
                            log_line(&dir, &format!(
                                "  wrote {} (a = {a:.4}, {count} particles; mean v = ({:.3e}, {:.3e}), rms v = {vrms:.4}, non-finite = {nans})",
                                path.display(),
                                mom[0],
                                mom[1]
                            ));
                        }
                        Err(e) => {
                            failed.store(true, Ordering::Release);
                            log_line(&dir, &format!("  snapshot {} FAILED: {e}", path.display()));
                        }
                    }
                }
            }
        }
    }    let _ = pk.flush();
    // COMPLETE only if every requested snapshot was written (and verified on disk) and P(k)
    // was measured; anything else leaves the run visibly incomplete.
    let expected = info.snapshot_names.len();
    let all_on_disk = info.snapshot_names.iter().all(|(_, n)| {
        fs::metadata(dir.join(n)).map(|m| m.len() == 24 + info.particles * 16).unwrap_or(false)
    });
    let ok = complete.load(Ordering::Acquire) && !failed.load(Ordering::Acquire) && snaps_written == expected && all_on_disk && n_pk > 0;
    if complete.load(Ordering::Acquire) && !ok {
        log_line(&dir, &format!("  run NOT complete: snapshots written {snaps_written}/{expected}, on disk {all_on_disk}, pk measurements {n_pk}, failed {}", failed.load(Ordering::Acquire)));
    }
    if ok {
        let _ = fs::write(dir.join("COMPLETE"), format!("snapshots {snaps_written}\npk_measurements {n_pk}\nfinished_utc {}\n", utc_stamp()));
        log_line(&dir, &format!("  run complete: {}", dir.display()));
    }
}

/// GSNAP001 | u64 count | f32 a | f32 box | count*[f32 x, f32 y] in [0, box) | count*[f32 vx, f32 vy]
/// Velocities are peculiar velocities v = a dx/dt = p / a (box length units per 1/H0).
/// Record i is particle (lattice) ID i. Written chunk by chunk with positioned writes.
struct SnapFile {
    path: PathBuf,
    file: File,
    a: f64,
    count: u64,
    side: u32,
    received: u64,
    sum: [f64; 3],
    nans: u64,
}

impl SnapFile {
    fn create(path: &Path, a: f64, count: u64, side: u32) -> std::io::Result<Self> {
        let mut file = File::create(path)?;
        file.set_len(24 + count * 16)?;
        let mut hdr = Vec::with_capacity(24);
        hdr.extend_from_slice(b"GSNAP001");
        hdr.extend_from_slice(&count.to_le_bytes());
        hdr.extend_from_slice(&(a as f32).to_le_bytes());
        hdr.extend_from_slice(&(BOX as f32).to_le_bytes());
        file.write_all(&hdr)?;
        Ok(Self { path: path.to_path_buf(), file, a, count, side, received: 0, sum: [0.0; 3], nans: 0 })
    }

    fn write_chunk(&mut self, offset: u64, data: &[[f32; 4]]) -> std::io::Result<()> {
        use std::io::{Seek, SeekFrom};
        let spacing = BOX / self.side as f64;
        let l = BOX as f32;
        let inv_a = 1.0 / self.a as f32;
        let mut pos = Vec::with_capacity(data.len() * 8);
        let mut vel = Vec::with_capacity(data.len() * 8);
        for (k, p) in data.iter().enumerate() {
            let i = offset + k as u64;
            let (ix, iy) = ((i % self.side as u64) as f64, (i / self.side as u64) as f64);
            let mut x = ((ix + 0.5) * spacing + p[0] as f64) as f32;
            let mut y = ((iy + 0.5) * spacing + p[1] as f64) as f32;
            x -= l * (x / l).floor();
            y -= l * (y / l).floor();
            if x >= l {
                x = 0.0;
            }
            if y >= l {
                y = 0.0;
            }
            pos.extend_from_slice(&x.to_le_bytes());
            pos.extend_from_slice(&y.to_le_bytes());
            let (vx, vy) = (p[2] * inv_a, p[3] * inv_a);
            if vx.is_finite() && vy.is_finite() && p[0].is_finite() && p[1].is_finite() {
                self.sum[0] += vx as f64;
                self.sum[1] += vy as f64;
                self.sum[2] += (vx * vx + vy * vy) as f64;
            } else {
                self.nans += 1;
            }
            vel.extend_from_slice(&vx.to_le_bytes());
            vel.extend_from_slice(&vy.to_le_bytes());
        }
        self.file.seek(SeekFrom::Start(24 + offset * 8))?;
        self.file.write_all(&pos)?;
        self.file.seek(SeekFrom::Start(24 + self.count * 8 + offset * 8))?;
        self.file.write_all(&vel)?;
        self.received += data.len() as u64;
        Ok(())
    }

    /// Returns (mean v, non-finite count, rms v, count).
    fn finish(self) -> std::io::Result<([f64; 2], u64, f64, u64)> {
        if self.received != self.count {
            return Err(std::io::Error::other(format!("received {} of {} particles", self.received, self.count)));
        }
        self.file.sync_all()?;
        let n = self.count.max(1) as f64;
        Ok(([self.sum[0] / n, self.sum[1] / n], self.nans, (self.sum[2] / n).sqrt(), self.count))
    }
}
/// a of a snapshot file if its header and size are valid for `count` particles.
pub fn snapshot_header(path: &Path, count: u64) -> Option<f64> {
    use std::io::Read;
    let meta = fs::metadata(path).ok()?;
    if meta.len() != 24 + count * 16 {
        return None;
    }
    let mut f = File::open(path).ok()?;
    let mut h = [0u8; 24];
    f.read_exact(&mut h).ok()?;
    if &h[..8] != b"GSNAP001" || u64::from_le_bytes(h[8..16].try_into().ok()?) != count {
        return None;
    }
    Some(f32::from_le_bytes(h[16..20].try_into().ok()?) as f64)
}

/// Latest valid snapshot in `dir` (restart point): (a, path).
pub fn latest_snapshot(dir: &Path, count: u64) -> Option<(f64, PathBuf)> {
    let mut best: Option<(f64, PathBuf)> = None;
    for e in fs::read_dir(dir).ok()?.flatten() {
        let n = e.file_name().to_string_lossy().to_string();
        if n.starts_with("snap_a") && n.ends_with(".bin") {
            if let Some(a) = snapshot_header(&e.path(), count) {
                if best.as_ref().is_none_or(|b| a > b.0) {
                    best = Some((a, e.path()));
                }
            }
        }
    }
    best
}

/// Read a snapshot back into the GPU particle layout: (s = x - q minimum image, p = a v).
pub fn read_snapshot_state(path: &Path, side: u32) -> std::io::Result<(f64, Vec<[f32; 4]>)> {
    let bytes = fs::read(path)?;
    let count = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    let a = f32::from_le_bytes(bytes[16..20].try_into().unwrap()) as f64;
    let pos: &[[f32; 2]] = bytemuck::cast_slice(&bytes[24..24 + count * 8]);
    let vel: &[[f32; 2]] = bytemuck::cast_slice(&bytes[24 + count * 8..24 + count * 16]);
    let spacing = BOX / side as f64;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let (ix, iy) = ((i % side as usize) as f64, (i / side as usize) as f64);
        let q = [(ix + 0.5) * spacing, (iy + 0.5) * spacing];
        let mut s = [pos[i][0] as f64 - q[0], pos[i][1] as f64 - q[1]];
        for v in &mut s {
            *v -= BOX * (*v / BOX).round();
        }
        out.push([s[0] as f32, s[1] as f32, (vel[i][0] as f64 * a) as f32, (vel[i][1] as f64 * a) as f32]);
    }
    Ok((a, out))
}