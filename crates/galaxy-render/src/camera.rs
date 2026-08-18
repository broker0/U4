//! Camera with an absolute galactic position and a directional view matrix.
//!
//! The camera stores its position in [`GalacticCoord`] (exact `i128`). Stars are
//! rendered as a *sky sphere*: each star is projected by its direction from the
//! camera (its distance only modulates brightness), so there is no dependence on
//! galactic-scale near/far planes and no floating-point blow-up. The view matrix
//! therefore encodes only the camera orientation; positions are unit directions
//! on a sphere of radius 1 around the eye at the origin of clip space.

use galaxy_coord::GalacticCoord;
use glam::{DMat4, DQuat, DVec3, Mat4};

/// A free-flying 6DOF camera.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    /// Absolute position in the galaxy.
    pub position: GalacticCoord,
    /// Orientation (world-from-camera rotation).
    pub orientation: DQuat,
    /// Vertical field of view, radians.
    pub fov_y: f64,
}

impl Default for Camera {
    fn default() -> Self {
        Camera {
            position: GalacticCoord::ORIGIN,
            orientation: DQuat::IDENTITY,
            fov_y: 60f64.to_radians(),
        }
    }
}

impl Camera {
    #[inline]
    pub fn forward(&self) -> DVec3 {
        self.orientation * DVec3::NEG_Z
    }
    #[inline]
    pub fn right(&self) -> DVec3 {
        self.orientation * DVec3::X
    }
    #[inline]
    pub fn up(&self) -> DVec3 {
        self.orientation * DVec3::Y
    }

    /// View-projection matrix for the directional sky-sphere render.
    ///
    /// Stars are supplied as unit directions placed at radius 1, so the eye
    /// sits at the origin and near/far bracket that unit sphere. This removes
    /// all galactic-scale precision problems from the projection.
    pub fn view_proj(&self, aspect: f64) -> Mat4 {
        let eye = DVec3::ZERO;
        let center = self.forward();
        let view = DMat4::look_at_rh(eye, center, self.up());
        // Unit sphere sits at distance 1; a generous bracket keeps it inside.
        let proj = DMat4::perspective_rh(self.fov_y, aspect, 0.01, 10.0);
        (proj * view).as_mat4()
    }

    /// Apply yaw/pitch deltas (radians) using *local* axes.
    ///
    /// Right-multiplying by local rotations avoids the drift/roll accumulation
    /// that occurs when rotating about world axes derived from the current
    /// orientation. Roll is only introduced explicitly via [`Camera::roll`].
    pub fn rotate(&mut self, yaw: f64, pitch: f64) {
        let yaw_q = DQuat::from_axis_angle(DVec3::Y, yaw);
        let pitch_q = DQuat::from_axis_angle(DVec3::X, pitch);
        // Local-space application: orientation * yaw * pitch.
        self.orientation = (self.orientation * yaw_q * pitch_q).normalize();
    }

    /// Roll about the local view axis (radians). Positive rolls clockwise.
    pub fn roll(&mut self, angle: f64) {
        let roll_q = DQuat::from_axis_angle(DVec3::NEG_Z, angle);
        self.orientation = (self.orientation * roll_q).normalize();
    }

    /// Move by a local-space delta scaled by `speed` meters.
    pub fn translate_local(&mut self, local: DVec3, speed: f64) {
        let world = self.orientation * local * speed;
        self.position = self.position.offset_f64(world);
    }
}
