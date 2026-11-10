//! Non-blocking GPU -> CPU readback through a small ring of MAP_READ staging buffers.
//!
//! Per frame: `begin` picks a free slot and records a copy into it; after the submit,
//! `after_submit` starts `map_async` on it. The callback fires during a later
//! `device.poll(Poll)` / submit, so data arrives one or two frames late and the GPU is
//! never stalled waiting for the CPU.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

const FREE: u32 = 0;
const RECORDED: u32 = 1;
const MAPPING: u32 = 2;
const READY: u32 = 3;

struct Slot {
    buf: wgpu::Buffer,
    state: Arc<AtomicU32>,
    frame: u64,
}

pub struct ReadbackRing {
    slots: Vec<Slot>,
    size: u64,
    frame: u64,
    recorded: Option<usize>,
}

impl ReadbackRing {
    pub fn new(device: &wgpu::Device, label: &str, size: u64, count: usize) -> Self {
        let slots = (0..count)
            .map(|_| Slot {
                buf: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                state: Arc::new(AtomicU32::new(FREE)),
                frame: 0,
            })
            .collect();
        Self {
            slots,
            size,
            frame: 0,
            recorded: None,
        }
    }

    /// Returns a free staging buffer to copy into this frame (None if all are busy).
    pub fn begin(&mut self) -> Option<&wgpu::Buffer> {
        self.frame += 1;
        let idx = self
            .slots
            .iter()
            .position(|s| s.state.load(Ordering::Acquire) == FREE)?;
        let slot = &mut self.slots[idx];
        slot.state.store(RECORDED, Ordering::Release);
        slot.frame = self.frame;
        self.recorded = Some(idx);
        Some(&slot.buf)
    }

    /// Call after the queue submit that contains the copy recorded by `begin`.
    pub fn after_submit(&mut self) {
        if let Some(idx) = self.recorded.take() {
            let slot = &self.slots[idx];
            slot.state.store(MAPPING, Ordering::Release);
            let state = slot.state.clone();
            slot.buf.map_async(wgpu::MapMode::Read, .., move |r| {
                state.store(if r.is_ok() { READY } else { FREE }, Ordering::Release);
            });
        }
    }

    /// If a mapped result is ready, pass the newest one to `f` and recycle all ready
    /// slots. Returns whether `f` was called.
    pub fn read_latest(&mut self, f: impl FnOnce(&[u8])) -> bool {
        let newest = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.state.load(Ordering::Acquire) == READY)
            .max_by_key(|(_, s)| s.frame)
            .map(|(i, _)| i);
        let Some(newest) = newest else {
            return false;
        };
        {
            let slot = &self.slots[newest];
            match slot.buf.slice(..self.size).get_mapped_range() {
                Ok(view) => f(&view),
                Err(e) => log::warn!("readback map failed: {e:?}"),
            }
        }
        for s in &self.slots {
            if s.state.load(Ordering::Acquire) == READY {
                s.buf.unmap();
                s.state.store(FREE, Ordering::Release);
            }
        }
        true
    }
}
