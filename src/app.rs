//! The eframe application: galaxy state, camera controls via egui input, and
//! an egui UI whose central panel hosts the wgpu star render via a paint
//! callback. This runs identically on desktop and web.

use eframe::egui_wgpu;
use egui::{Key, PointerButton};
use galaxy_coord::{GalacticCoord, METERS_PER_AU, METERS_PER_LIGHT_YEAR};
use galaxy_gen::{GalaxyParams, GalaxyType};
use galaxy_octree::{Galaxy, ViewParams};
use galaxy_render::{
    Camera, FrameInput, FrustumCallback, FrustumResources, StarCallback, StarResources,
    WireCallback, WireFrameInput, WireResources,
};
use glam::{DQuat, DVec3};

const GALAXY_CACHE_CAPACITY: usize = 65_536;

pub struct GalaxyApp {
    galaxy: Galaxy,
    camera: Camera,
    view: ViewParams,
    /// Movement speed in meters per second.
    speed_m_per_s: f64,
    last_stats: galaxy_octree::TraverseStats,
    /// Cached visible-star set, rebuilt only when the observer moves enough or
    /// the view parameters change.
    visible_stars: Vec<galaxy_octree::VisibleStar>,
    /// Observer position when `visible_stars` was last rebuilt.
    scene_anchor: GalacticCoord,
    /// True to force a scene rebuild next frame.
    scene_dirty: bool,
    /// Position that drives which stars/octree nodes are generated & visible
    /// (fed to `collect_visible`). Tracks the camera 1:1 unless
    /// `camera_detached` is true, in which case it stays frozen while the
    /// camera keeps moving/looking freely.
    observer: GalacticCoord,
    /// True: the camera can move freely without moving `observer`, so the
    /// visible star/node set stays frozen. Unchecking the UI toggle snaps the
    /// camera back to `observer`.
    camera_detached: bool,
    /// Frozen orientation associated with `observer` while detached.
    observer_orientation: DQuat,
    /// Instances drawn last frame (for the overlay).
    last_instances: u32,
    /// Visited node keys from the last scene rebuild (wireframe overlay source).
    visible_nodes: Vec<galaxy_gen::NodeKey>,
    /// True to draw the visited octree nodes as wireframe cubes.
    show_nodes: bool,
    /// True to draw observer frustum debug wireframe (in detach mode).
    show_frustums: bool,
    /// Cap on wireframe cubes drawn (bounds the per-frame upload).
    max_wire_nodes: usize,
}

impl GalaxyApp {
    /// Construct the app and register the star GPU resources with egui-wgpu.
    pub fn new(cc: &eframe::CreationContext<'_>) -> Option<Self> {
        // Grab egui's wgpu render state; we require the wgpu backend.
        let wgpu_state = cc.wgpu_render_state.as_ref()?;
        let device = &wgpu_state.device;
        let target_format = wgpu_state.target_format;

        // Build our pipeline/buffers and stash them in the callback resources.
        let resources = StarResources::new(device, target_format);
        wgpu_state
            .renderer
            .write()
            .callback_resources
            .insert(resources);
        let wire_resources = WireResources::new(device, target_format);
        wgpu_state
            .renderer
            .write()
            .callback_resources
            .insert(wire_resources);
        let frustum_resources = FrustumResources::new(device, target_format);
        wgpu_state
            .renderer
            .write()
            .callback_resources
            .insert(frustum_resources);

        let params = GalaxyParams::default();
        let galaxy = Galaxy::new(params, GALAXY_CACHE_CAPACITY);

        // Start somewhere inside the disk, a few thousand ly from center.
        let start = GalacticCoord::from_light_years_vec(8_000.0, 0.0, 200.0);
        let camera = Camera {
            position: start,
            ..Default::default()
        };

        Some(GalaxyApp {
            galaxy,
            camera,
            view: ViewParams::default(),
            speed_m_per_s: 500.0 * METERS_PER_LIGHT_YEAR,
            last_stats: galaxy_octree::TraverseStats::default(),
            visible_stars: Vec::new(),
            scene_anchor: start,
            scene_dirty: true,
            observer: start,
            camera_detached: false,
            observer_orientation: camera.orientation,
            last_instances: 0,
            visible_nodes: Vec::new(),
            show_nodes: false,
            show_frustums: false,
            max_wire_nodes: 4096,
        })
    }

    /// Read egui input and advance the camera. `look_delta` is the drag motion
    /// over the 3D viewport (in points); `dt` is the frame time in seconds.
    fn handle_input(&mut self, ctx: &egui::Context, look_delta: egui::Vec2, dt: f64) {
        ctx.input(|i| {
            // Speed modifiers.
            let mut speed = self.speed_m_per_s;
            if i.modifiers.shift {
                speed *= 20.0;
            }
            if i.modifiers.alt {
                speed *= 0.05;
            }

            // Look: drag over the viewport rotates the camera.
            if look_delta != egui::Vec2::ZERO {
                let sens = 0.0035;
                self.camera
                    .rotate(-look_delta.x as f64 * sens, -look_delta.y as f64 * sens);
            }

            // Roll (Q/E).
            let roll_speed = 1.5; // rad/s
            if i.key_down(Key::Q) {
                self.camera.roll(-roll_speed * dt);
            }
            if i.key_down(Key::E) {
                self.camera.roll(roll_speed * dt);
            }

            // Translation: WASD strafe, Space/Ctrl vertical.
            let mut local = DVec3::ZERO;
            if i.key_down(Key::W) {
                local.z -= 1.0;
            }
            if i.key_down(Key::S) {
                local.z += 1.0;
            }
            if i.key_down(Key::A) {
                local.x -= 1.0;
            }
            if i.key_down(Key::D) {
                local.x += 1.0;
            }
            if i.key_down(Key::Space) {
                local.y += 1.0;
            }
            if i.key_down(Key::C) || i.modifiers.ctrl {
                local.y -= 1.0;
            }
            if local != DVec3::ZERO {
                self.camera.translate_local(local.normalize(), speed * dt);
            }

            // Magnitude limit tweak with [ ].
            if i.key_pressed(Key::CloseBracket) {
                self.view.mag_limit += 0.5;
                self.scene_dirty = true;
            }
            if i.key_pressed(Key::OpenBracket) {
                self.view.mag_limit -= 0.5;
                self.scene_dirty = true;
            }

            // Scroll adjusts speed multiplicatively.
            let scroll = i.smooth_scroll_delta.y as f64;
            if scroll != 0.0 {
                let factor = 1.15f64.powf(scroll / 40.0);
                self.speed_m_per_s =
                    (self.speed_m_per_s * factor).clamp(0.1, 1.0e5 * METERS_PER_LIGHT_YEAR);
            }
        });
    }

    /// Rebuild the visible-star set if the observer moved enough or params changed.
    fn maybe_rebuild_scene(&mut self) {
        let moved = self.observer.distance_meters(self.scene_anchor);
        let threshold = (self.speed_m_per_s * 0.05).max(1.0e9);
        if self.scene_dirty || moved > threshold || self.visible_stars.is_empty() {
            let (stars, stats, nodes) = self.galaxy.collect_visible(self.observer, &self.view);
            self.visible_stars = stars;
            self.visible_nodes = nodes;
            self.last_stats = stats;
            self.scene_anchor = self.observer;
            self.scene_dirty = false;
        }
    }

    fn set_galaxy_type(&mut self, galaxy_type: GalaxyType) {
        if self.galaxy.params.galaxy_type == galaxy_type {
            return;
        }
        let mut params = self.galaxy.params;
        params.galaxy_type = galaxy_type;
        self.galaxy = Galaxy::new(params, GALAXY_CACHE_CAPACITY);
        self.visible_stars.clear();
        self.visible_nodes.clear();
        self.last_stats = galaxy_octree::TraverseStats::default();
        self.scene_anchor = self.observer;
        self.scene_dirty = true;
    }

    fn overlay_ui(&mut self, ui: &mut egui::Ui, fps: f32) {
        let cam = self.camera.position;
        let stats = self.last_stats;
        ui.label(format!("FPS: {fps:.1}"));
        ui.label(format!(
            "pos (ly): {:.0}, {:.0}, {:.0}",
            cam.x.to_meters_f64() / METERS_PER_LIGHT_YEAR,
            cam.y.to_meters_f64() / METERS_PER_LIGHT_YEAR,
            cam.z.to_meters_f64() / METERS_PER_LIGHT_YEAR,
        ));
        if self.camera_detached {
            let obs = self.observer;
            ui.label(format!(
                "observer (ly): {:.0}, {:.0}, {:.0}",
                obs.x.to_meters_f64() / METERS_PER_LIGHT_YEAR,
                obs.y.to_meters_f64() / METERS_PER_LIGHT_YEAR,
                obs.z.to_meters_f64() / METERS_PER_LIGHT_YEAR,
            ));
            let offset_ly = cam.distance_meters(obs) / METERS_PER_LIGHT_YEAR;
            ui.label(format!("camera offset from observer: {offset_ly:.2} ly"));
        }
        ui.separator();
        ui.label(format!(
            "visible {} / generated {}",
            stats.stars_visible, stats.stars_generated
        ));
        ui.label(format!("drawn instances: {}", self.last_instances));
        ui.label(format!(
            "nodes {} (subdiv {})",
            stats.nodes_visited, stats.nodes_subdivided
        ));
        ui.label(format!(
            "cache {}h / {}m",
            stats.cache_hits, stats.cache_misses
        ));
        ui.separator();

        let mut galaxy_type = self.galaxy.params.galaxy_type;
        egui::ComboBox::from_label("galaxy type")
            .selected_text(galaxy_type.label())
            .show_ui(ui, |ui| {
                ui.selectable_value(
                    &mut galaxy_type,
                    GalaxyType::UniformCube,
                    GalaxyType::UniformCube.label(),
                );
                ui.selectable_value(
                    &mut galaxy_type,
                    GalaxyType::Elliptical,
                    GalaxyType::Elliptical.label(),
                );
                ui.selectable_value(
                    &mut galaxy_type,
                    GalaxyType::Spiral,
                    GalaxyType::Spiral.label(),
                );
            });
        self.set_galaxy_type(galaxy_type);

        let mut dirty = false;
        dirty |= ui
            .add(egui::Slider::new(&mut self.view.mag_limit, -2.0..=20.0).text("mag limit"))
            .changed();
        dirty |= ui
            .add(egui::Slider::new(&mut self.view.fade_width, 0.1..=5.0).text("fade width"))
            .changed();
        dirty |= ui
            .add(
                egui::Slider::new(&mut self.view.subdivide_angle, 0.02..=1.0)
                    .text("subdivide angle"),
            )
            .changed();
        dirty |= ui
            .add(egui::Slider::new(&mut self.view.max_nodes, 500..=60_000).text("max nodes/frame"))
            .changed();
        if dirty {
            self.scene_dirty = true;
        }

        ui.add(
            egui::Slider::new(&mut self.speed_m_per_s, 1.0..=1.0e5 * METERS_PER_LIGHT_YEAR)
                .logarithmic(true)
                .text("speed")
                .custom_formatter(|v, _| format_speed(v)),
        );
        ui.label(format!("current speed: {}", format_speed(self.speed_m_per_s)));
        ui.separator();
        let detach_resp =
            ui.checkbox(&mut self.camera_detached, "detach camera (freeze observer)");
        if detach_resp.changed() {
            if self.camera_detached {
                // Detach entered: freeze observer pose for frustum debug.
                self.observer = self.camera.position;
                self.observer_orientation = self.camera.orientation;
            } else {
                // Just re-attached: snap the camera back to the frozen observer.
                self.camera.position = self.observer;
                self.camera.orientation = self.observer_orientation;
            }
        }
        ui.separator();
        ui.checkbox(&mut self.show_nodes, "show octree nodes");
        ui.checkbox(&mut self.show_frustums, "show observer frustum (detach)");
        ui.add(
            egui::Slider::new(&mut self.max_wire_nodes, 64..=20_000).text("wire nodes"),
        );
        ui.separator();
        ui.label("Drag viewport: look • WASD: move • Space/Ctrl: up/down");
        ui.label("Q/E: roll • wheel: speed • Shift: boost • Alt: slow • [ ]: mag");
    }
}

impl eframe::App for GalaxyApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        let dt = ctx.input(|i| i.stable_dt).min(0.1) as f64;
        let fps = if dt > 0.0 { (1.0 / dt) as f32 } else { 0.0 };

        // Overlay window (floating panel).
        egui::Window::new("U4 — galaxy")
            .default_pos([12.0, 12.0])
            .resizable(false)
            .show(&ctx, |ui| {
                self.overlay_ui(ui, fps);
            });

        // Central panel hosts the star render as a full-area paint callback.
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(egui::Color32::from_rgb(0, 1, 2)))
            .show(ui, |ui| {
                let (rect, response) =
                    ui.allocate_exact_size(ui.available_size(), egui::Sense::drag());

                // Look only while the primary button drags inside the viewport.
                let look_delta = if response.dragged_by(PointerButton::Primary) {
                    response.drag_delta()
                } else {
                    egui::Vec2::ZERO
                };

                self.handle_input(&ctx, look_delta, dt);
                if !self.camera_detached {
                    self.observer = self.camera.position;
                    self.observer_orientation = self.camera.orientation;
                }
                self.maybe_rebuild_scene();

                // Compute view-projection from the widget aspect (physical px).
                let ppp = ctx.pixels_per_point();
                let vp_px = [rect.width() * ppp, rect.height() * ppp];
                let aspect = (vp_px[0] / vp_px[1].max(1.0)) as f64;
                let view_proj = self.camera.view_proj(aspect);
                let (render_observer, observer_orientation) = if self.camera_detached {
                    (self.observer, self.observer_orientation)
                } else {
                    (self.camera.position, self.camera.orientation)
                };
                let observer_camera = Camera {
                    position: render_observer,
                    orientation: observer_orientation,
                    fov_y: self.camera.fov_y,
                };
                let observer_view_proj = observer_camera.view_proj(aspect);

                let input = FrameInput {
                    stars: &self.visible_stars,
                    camera: &self.camera,
                    mag_limit: self.view.mag_limit,
                };
                let cb = StarCallback::new(
                    &input,
                    view_proj,
                    render_observer,
                    observer_view_proj,
                    vp_px,
                    2.0,
                );
                self.last_instances = cb.instance_count();

                // Optional octree wireframe overlay. Additive like the stars, so
                // draw order is cosmetic.
                if self.show_nodes {
                    let winput = WireFrameInput {
                        galaxy: &self.galaxy,
                        nodes: &self.visible_nodes,
                        camera: &self.camera,
                        max_nodes: self.max_wire_nodes,
                        aspect,
                    };
                    let wcb = WireCallback::new(&winput);
                    ui.painter()
                        .add(egui_wgpu::Callback::new_paint_callback(rect, wcb));
                }

                if self.show_frustums {
                    let frustum_pose = if self.camera_detached {
                        Some((self.observer, self.observer_orientation))
                    } else {
                        None
                    };
                    let fcb = FrustumCallback::new(&self.camera, aspect, frustum_pose);
                    ui.painter()
                        .add(egui_wgpu::Callback::new_paint_callback(rect, fcb));
                }

                ui.painter()
                    .add(egui_wgpu::Callback::new_paint_callback(rect, cb));
            });

        // Keep animating.
        ctx.request_repaint();
    }
}

/// Format a speed given in meters/second using an adaptive astronomical unit.
fn format_speed(m_per_s: f64) -> String {
    let ly = METERS_PER_LIGHT_YEAR;
    let au = METERS_PER_AU;
    if m_per_s >= 0.001 * ly {
        format!("{:.3} ly/s", m_per_s / ly)
    } else if m_per_s >= 0.01 * au {
        format!("{:.3} AU/s", m_per_s / au)
    } else if m_per_s >= 1000.0 {
        format!("{:.2} km/s", m_per_s / 1000.0)
    } else {
        format!("{:.2} m/s", m_per_s)
    }
}
