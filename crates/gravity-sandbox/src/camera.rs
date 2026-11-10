//! 2D orthographic camera: pan + zoom-towards-cursor, screen <-> world conversion.
//!
//! World space is y-up; screen (window pixel) space is y-down with origin top-left.

/// Initial vertical half-extent of the view in world units (shows about +-900 vertically).
pub const INITIAL_HALF_HEIGHT: f32 = 900.0;
const MIN_HALF_HEIGHT: f32 = 10.0;
const MAX_HALF_HEIGHT: f32 = 20_000.0;

#[derive(Clone, Copy, Debug)]
pub struct Camera {
    /// World-space point at the center of the screen.
    pub center: [f32; 2],
    /// Half of the visible world height.
    pub half_height: f32,
    /// Viewport size in physical pixels.
    pub viewport: [f32; 2],
}

impl Camera {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            center: [0.0, 0.0],
            half_height: INITIAL_HALF_HEIGHT,
            viewport: [width.max(1) as f32, height.max(1) as f32],
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.viewport = [width.max(1) as f32, height.max(1) as f32];
    }

    pub fn aspect(&self) -> f32 {
        self.viewport[0] / self.viewport[1]
    }

    /// Visible half extents (x, y) in world units.
    pub fn half_extent(&self) -> [f32; 2] {
        [self.half_height * self.aspect(), self.half_height]
    }

    /// World units per physical pixel.
    pub fn world_per_pixel(&self) -> f32 {
        2.0 * self.half_height / self.viewport[1]
    }

    /// Pixel position -> normalized device coordinates (-1..1, y-up).
    fn screen_to_ndc(&self, px: f32, py: f32) -> [f32; 2] {
        [
            px / self.viewport[0] * 2.0 - 1.0,
            1.0 - py / self.viewport[1] * 2.0,
        ]
    }

    pub fn screen_to_world(&self, px: f32, py: f32) -> [f32; 2] {
        let ndc = self.screen_to_ndc(px, py);
        let h = self.half_extent();
        [self.center[0] + ndc[0] * h[0], self.center[1] + ndc[1] * h[1]]
    }

    /// Zoom by `factor` (<1 zooms in) keeping the world point under the cursor fixed.
    pub fn zoom_at(&mut self, px: f32, py: f32, factor: f32) {
        let anchor = self.screen_to_world(px, py);
        let ndc = self.screen_to_ndc(px, py);
        self.half_height = (self.half_height * factor).clamp(MIN_HALF_HEIGHT, MAX_HALF_HEIGHT);
        let h = self.half_extent();
        self.center = [anchor[0] - ndc[0] * h[0], anchor[1] - ndc[1] * h[1]];
    }

    /// Pan by a mouse drag of (dx, dy) pixels: the world moves with the cursor.
    pub fn pan_pixels(&mut self, dx: f32, dy: f32) {
        let wpp = self.world_per_pixel();
        self.center[0] -= dx * wpp;
        self.center[1] += dy * wpp;
    }

    /// Pan by a world-space offset.
    pub fn pan_world(&mut self, dx: f32, dy: f32) {
        self.center[0] += dx;
        self.center[1] += dy;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zoom_keeps_anchor_fixed() {
        let mut c = Camera::new(1600, 900);
        let before = c.screen_to_world(1200.0, 200.0);
        c.zoom_at(1200.0, 200.0, 0.5);
        let after = c.screen_to_world(1200.0, 200.0);
        assert!((before[0] - after[0]).abs() < 1e-3);
        assert!((before[1] - after[1]).abs() < 1e-3);
    }

    #[test]
    fn center_maps_to_center() {
        let c = Camera::new(1600, 900);
        let w = c.screen_to_world(800.0, 450.0);
        assert!(w[0].abs() < 1e-4 && w[1].abs() < 1e-4);
        let top = c.screen_to_world(800.0, 0.0);
        assert!((top[1] - INITIAL_HALF_HEIGHT).abs() < 1e-3);
    }
}
