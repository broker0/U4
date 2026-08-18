//! wgpu-based star renderer, embedded inside egui via `egui_wgpu::CallbackTrait`.
//!
//! Instead of owning a surface and running its own render pass, this renderer
//! draws into egui's render pass through a paint callback. The GPU resources
//! (pipeline, bind group, buffers) live in egui's `callback_resources` type
//! map so they share the render pass lifetime; per-frame data (view-projection,
//! viewport, star instances) is carried by a [`StarCallback`].
//!
//! Stars are drawn as additive billboards. Positions arrive as absolute
//! [`GalacticCoord`]; the CPU converts them to unit directions relative to a
//! floating origin (the camera) every frame, so projection stays in a tiny,
//! well-conditioned numeric range regardless of galactic scale.

pub mod camera;

pub use camera::Camera;

use bytemuck::{Pod, Zeroable};
use egui_wgpu::wgpu;
use galaxy_coord::{GalacticCoord, METERS_PER_LIGHT_YEAR, METERS_PER_PARSEC};
use galaxy_gen::NodeKey;
use galaxy_octree::Galaxy;
use glam::{DMat4, DQuat, DVec3, Mat4};

/// Per-star GPU instance data (unit direction from the camera).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct InstanceRaw {
    dir: [f32; 3],
    color: [f32; 3],
    intensity: f32,
    core_px: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Globals {
    view_proj: [[f32; 4]; 4],
    px_scale: f32,
    _pad0: f32,
    viewport: [f32; 2],
}

/// Convert apparent magnitude to a linear display intensity.
///
/// Magnitudes are logarithmic (each 5 mag = factor 100 in flux). We map the
/// visible band so that bright stars glow strongly and the `mag_limit` fades
/// to near zero.
fn magnitude_to_intensity(app_mag: f32, mag_limit: f32) -> f32 {
    // Flux ratio relative to the limit; clamp for stability.
    let x = (mag_limit - app_mag) * 0.4; // 0.4 = 1/2.5
    let flux = 10f32.powf(x.clamp(-4.0, 4.0));
    (flux * 0.02).clamp(0.0, 4.0)
}

/// Everything needed to draw one frame of stars, handed to the paint callback.
pub struct FrameInput<'a> {
    pub stars: &'a [galaxy_octree::VisibleStar],
    pub camera: &'a Camera,
    pub mag_limit: f32,
}

/// GPU resources for the star pass. Stored in egui's `callback_resources`.
pub struct StarResources {
    pipeline: wgpu::RenderPipeline,
    globals_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    instance_buf: wgpu::Buffer,
    instance_capacity: u64,
}

impl StarResources {
    /// Build the pipeline and buffers using egui's device and target format.
    ///
    /// Call once from `eframe::CreationContext` and insert the result into
    /// `wgpu_render_state.renderer.write().callback_resources`.
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("star.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("star.wgsl").into()),
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("globals-bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let globals_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals"),
            size: std::mem::size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globals-bg"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: globals_buf.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("star-pl"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        // Instance buffer layout: one InstanceRaw per star, stepped per instance.
        let instance_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<InstanceRaw>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                // dir
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // color
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // intensity
                wgpu::VertexAttribute {
                    offset: 24,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32,
                },
                // core_px
                wgpu::VertexAttribute {
                    offset: 28,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("star-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Some(instance_layout)],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    // Additive blending for glow accumulation.
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview_mask: None,
            cache: None,
        });

        let instance_capacity = 4096u64;
        let instance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instances"),
            size: instance_capacity * std::mem::size_of::<InstanceRaw>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        StarResources {
            pipeline,
            globals_buf,
            bind_group,
            instance_buf,
            instance_capacity,
        }
    }
}

/// Build the CPU-side instance buffer for a frame.
///
/// Culling and photometry are evaluated from the observer; final billboard
/// direction is still from the render camera so detached camera motion can
/// inspect the frozen observer-selected set.
fn build_instances(
    frame: &FrameInput,
    observer: GalacticCoord,
    observer_view_proj: Mat4,
) -> Vec<InstanceRaw> {
    let render_cam = frame.camera.position;
    let mut raw: Vec<InstanceRaw> = Vec::with_capacity(frame.stars.len());
    const NDC_MARGIN: f32 = 0.05;
    for v in frame.stars {
        let s = &v.star;
        let observer_rel = s.pos.relative_f64(observer);
        let observer_dist_m = observer_rel.length();
        if observer_dist_m < 1.0 {
            // Star essentially at observer origin; skip to avoid NaN direction.
            continue;
        }

        let observer_dir = (observer_rel / observer_dist_m).as_vec3();

        // CPU frustum cull: skip stars outside the current viewport.
        let clip = observer_view_proj * observer_dir.extend(1.0);
        if clip.w <= 0.0 {
            continue;
        }
        let inv_w = 1.0 / clip.w;
        let ndc_x = clip.x * inv_w;
        let ndc_y = clip.y * inv_w;
        if ndc_x.abs() > 1.0 + NDC_MARGIN || ndc_y.abs() > 1.0 + NDC_MARGIN {
            continue;
        }

        let d_pc = observer_dist_m / METERS_PER_PARSEC;
        let app_mag = s.abs_mag + 5.0 * (d_pc.log10() as f32 - 1.0);
        // Multiply by the octree's view-dependent fade so faint/LOD-boundary
        // stars ramp in smoothly instead of popping as hard walls.
        let intensity = magnitude_to_intensity(app_mag, frame.mag_limit) * v.fade;
        if intensity <= 0.0 {
            continue;
        }

        let render_rel = s.pos.relative_f64(render_cam);
        let render_dist_m = render_rel.length();
        if render_dist_m < 1.0 {
            // Star essentially at the render camera; skip to avoid NaN direction.
            continue;
        }
        let render_dir = (render_rel / render_dist_m).as_vec3();

        raw.push(InstanceRaw {
            dir: render_dir.into(),
            color: s.color,
            intensity,
            core_px: 0.6,
        });
    }
    raw
}

/// A per-frame paint callback that draws the star billboards inside egui's pass.
pub struct StarCallback {
    view_proj: Mat4,
    viewport: [f32; 2],
    px_scale: f32,
    instances: Vec<InstanceRaw>,
}

impl StarCallback {
    /// Prepare per-frame star data. `viewport` is the widget size in physical
    /// pixels; `aspect` should match it.
    pub fn new(
        frame: &FrameInput,
        view_proj: Mat4,
        observer: GalacticCoord,
        observer_view_proj: Mat4,
        viewport: [f32; 2],
        px_scale: f32,
    ) -> Self {
        StarCallback {
            view_proj,
            viewport,
            px_scale,
            instances: build_instances(frame, observer, observer_view_proj),
        }
    }

    /// Number of star instances that will be drawn.
    pub fn instance_count(&self) -> u32 {
        self.instances.len() as u32
    }
}

impl egui_wgpu::CallbackTrait for StarCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let res: &mut StarResources = resources.get_mut().expect("StarResources missing");

        // Upload globals.
        let g = Globals {
            view_proj: self.view_proj.to_cols_array_2d(),
            px_scale: self.px_scale,
            _pad0: 0.0,
            viewport: self.viewport,
        };
        queue.write_buffer(&res.globals_buf, 0, bytemuck::bytes_of(&g));

        // Grow the instance buffer if needed, then upload.
        let needed = self.instances.len() as u64;
        if needed > 0 {
            if needed > res.instance_capacity {
                res.instance_capacity = needed.next_power_of_two();
                res.instance_buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("instances"),
                    size: res.instance_capacity * std::mem::size_of::<InstanceRaw>() as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
            }
            queue.write_buffer(&res.instance_buf, 0, bytemuck::cast_slice(&self.instances));
        }

        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        if self.instances.is_empty() {
            return;
        }
        let res: &StarResources = resources.get().expect("StarResources missing");
        render_pass.set_pipeline(&res.pipeline);
        render_pass.set_bind_group(0, &res.bind_group, &[]);
        render_pass.set_vertex_buffer(0, res.instance_buf.slice(..));
        // 6 vertices (quad) per instance.
        render_pass.draw(0..6, 0..self.instances.len() as u32);
    }
}

// ───────────────────────────── Frustum overlay ─────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FrustumGlobals {
    view_proj: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FrustumVertexRaw {
    pos_rel: [f32; 3],
    color: [f32; 4],
}

/// GPU resources for frustum debug lines.
pub struct FrustumResources {
    pipeline: wgpu::RenderPipeline,
    globals_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    vertex_buf: wgpu::Buffer,
    vertex_capacity: u64,
}

impl FrustumResources {
    /// Build the frustum line pipeline and buffers using egui's device.
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("frustum.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("frustum.wgsl").into()),
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("frustum-globals-bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let globals_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frustum-globals"),
            size: std::mem::size_of::<FrustumGlobals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frustum-globals-bg"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: globals_buf.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("frustum-pl"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<FrustumVertexRaw>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("frustum-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Some(vertex_layout)],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview_mask: None,
            cache: None,
        });

        let vertex_capacity = 256u64;
        let vertex_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frustum-vertices"),
            size: vertex_capacity * std::mem::size_of::<FrustumVertexRaw>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        FrustumResources {
            pipeline,
            globals_buf,
            bind_group,
            vertex_buf,
            vertex_capacity,
        }
    }
}

const FRUSTUM_EDGES: [(usize, usize); 12] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0), // near
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4), // far
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7), // sides
];

fn append_frustum_vertices(
    out: &mut Vec<FrustumVertexRaw>,
    view_camera: &Camera,
    frustum_origin: GalacticCoord,
    frustum_orientation: glam::DQuat,
    fov_y: f64,
    aspect: f64,
    near: f64,
    far: f64,
    color: [f32; 4],
) {
    let t = (fov_y * 0.5).tan();
    let near_y = near * t;
    let near_x = near_y * aspect;
    let far_y = far * t;
    let far_x = far_y * aspect;

    let corners_local = [
        DVec3::new(-near_x, -near_y, -near),
        DVec3::new(near_x, -near_y, -near),
        DVec3::new(near_x, near_y, -near),
        DVec3::new(-near_x, near_y, -near),
        DVec3::new(-far_x, -far_y, -far),
        DVec3::new(far_x, -far_y, -far),
        DVec3::new(far_x, far_y, -far),
        DVec3::new(-far_x, far_y, -far),
    ];

    let origin_rel = frustum_origin.relative_f64(view_camera.position);
    let mut corners_rel = [DVec3::ZERO; 8];
    for (i, c) in corners_local.iter().enumerate() {
        corners_rel[i] = origin_rel + frustum_orientation * *c;
    }

    for (a, b) in FRUSTUM_EDGES {
        out.push(FrustumVertexRaw {
            pos_rel: corners_rel[a].as_vec3().into(),
            color,
        });
        out.push(FrustumVertexRaw {
            pos_rel: corners_rel[b].as_vec3().into(),
            color,
        });
    }
}

/// Per-frame frustum overlay callback.
pub struct FrustumCallback {
    vertices: Vec<FrustumVertexRaw>,
    view_proj: Mat4,
}

impl FrustumCallback {
    /// Build an optional frustum wireframe for debug visualization.
    pub fn new(
        camera: &Camera,
        aspect: f64,
        frustum_pose: Option<(GalacticCoord, DQuat)>,
    ) -> Self {
        let mut vertices = Vec::with_capacity(24);

        let sep = frustum_pose
            .map(|(origin, _)| camera.position.distance_meters(origin))
            .unwrap_or(0.0);
        // Debug frustum should read as effectively unbounded. Keep a very large
        // baseline span and grow further with camera-observer separation.
        let far = (sep * 2.0).max(20_000.0 * METERS_PER_LIGHT_YEAR);
        let near = (far * 0.01).max(0.2 * METERS_PER_LIGHT_YEAR);

        if let Some((origin, orientation)) = frustum_pose {
            append_frustum_vertices(
                &mut vertices,
                camera,
                origin,
                orientation,
                camera.fov_y,
                aspect,
                near,
                far,
                [1.00, 0.65, 0.20, 0.85],
            );
        }

        // Frustum debug geometry lives in galactic meter units, not on the unit
        // sky sphere used by the star pass, so it needs its own projection
        // bracket instead of `camera.view_proj`.
        let proj_near = (near * 0.25).max(1.0);
        let proj_far = (far * 5.0 + sep).max(proj_near * 1000.0);
        let view = DMat4::look_at_rh(DVec3::ZERO, camera.forward(), camera.up());
        let proj = DMat4::perspective_rh(camera.fov_y, aspect, proj_near, proj_far);

        FrustumCallback {
            vertices,
            view_proj: (proj * view).as_mat4(),
        }
    }
}

impl egui_wgpu::CallbackTrait for FrustumCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let res: &mut FrustumResources = resources.get_mut().expect("FrustumResources missing");

        queue.write_buffer(
            &res.globals_buf,
            0,
            bytemuck::bytes_of(&FrustumGlobals {
                view_proj: self.view_proj.to_cols_array_2d(),
            }),
        );

        let needed = self.vertices.len() as u64;
        if needed > 0 {
            if needed > res.vertex_capacity {
                res.vertex_capacity = needed.next_power_of_two();
                res.vertex_buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("frustum-vertices"),
                    size: res.vertex_capacity * std::mem::size_of::<FrustumVertexRaw>() as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
            }
            queue.write_buffer(&res.vertex_buf, 0, bytemuck::cast_slice(&self.vertices));
        }

        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        if self.vertices.is_empty() {
            return;
        }
        let res: &FrustumResources = resources.get().expect("FrustumResources missing");
        render_pass.set_pipeline(&res.pipeline);
        render_pass.set_bind_group(0, &res.bind_group, &[]);
        render_pass.set_vertex_buffer(0, res.vertex_buf.slice(..));
        render_pass.draw(0..self.vertices.len() as u32, 0..1);
    }
}

// ────────────────────────── Octree wireframe overlay ──────────────────────────

/// The 12 cube edges as corner pairs in `{-1, +1}^3` (LineList order). Shared
/// as the single instanced mesh: every node reuses these 24 vertices, only
/// translated/scaled per instance.
const CUBE_EDGES: [[DVec3; 2]; 12] = [
    // Bottom face.
    [DVec3::new(-1.0, -1.0, -1.0), DVec3::new(1.0, -1.0, -1.0)],
    [DVec3::new(1.0, -1.0, -1.0), DVec3::new(1.0, 1.0, -1.0)],
    [DVec3::new(1.0, 1.0, -1.0), DVec3::new(-1.0, 1.0, -1.0)],
    [DVec3::new(-1.0, 1.0, -1.0), DVec3::new(-1.0, -1.0, -1.0)],
    // Top face.
    [DVec3::new(-1.0, -1.0, 1.0), DVec3::new(1.0, -1.0, 1.0)],
    [DVec3::new(1.0, -1.0, 1.0), DVec3::new(1.0, 1.0, 1.0)],
    [DVec3::new(1.0, 1.0, 1.0), DVec3::new(-1.0, 1.0, 1.0)],
    [DVec3::new(-1.0, 1.0, 1.0), DVec3::new(-1.0, -1.0, 1.0)],
    // Vertical pillars.
    [DVec3::new(-1.0, -1.0, -1.0), DVec3::new(-1.0, -1.0, 1.0)],
    [DVec3::new(1.0, -1.0, -1.0), DVec3::new(1.0, -1.0, 1.0)],
    [DVec3::new(1.0, 1.0, -1.0), DVec3::new(1.0, 1.0, 1.0)],
    [DVec3::new(-1.0, 1.0, -1.0), DVec3::new(-1.0, 1.0, 1.0)],
];

/// Number of vertices in the shared cube mesh (12 edges x 2 endpoints).
const CUBE_VERTEX_COUNT: u32 = 24;

/// Uniform for the wireframe pass: only the view-projection now, since color
/// is per-instance (depth-coded).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct WireGlobals {
    view_proj: [[f32; 4]; 4],
}

/// Per-instance data for one octree node cube.
///
/// `center_rel` is the cube center relative to the camera, computed in `f64`
/// with a floating origin and cast to `f32` only at this point — its
/// magnitude is bounded by the node's own distance from the camera rather
/// than by absolute galactic coordinates, so the cast stays well-conditioned.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct WireInstanceRaw {
    center_rel: [f32; 3],
    half_size: f32,
    color: [f32; 4],
}

/// Deterministic, well-separated color per octree depth level.
///
/// Uses a golden-angle hue rotation so colors stay visually distinct however
/// many depth levels are actually visited, without a fixed-size palette.
fn depth_color(depth: u8) -> [f32; 4] {
    const GOLDEN_RATIO_CONJUGATE: f32 = 0.618_034;
    let hue = (depth as f32 * GOLDEN_RATIO_CONJUGATE).fract();
    let (r, g, b) = hsv_to_rgb(hue, 0.65, 1.0);
    // Alpha matches the previous single wire color's additive weight.
    [r, g, b, 0.55]
}

/// Minimal HSV -> RGB conversion (`h`, `s`, `v` in `[0, 1]`).
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let i = (h * 6.0).floor();
    let f = h * 6.0 - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    match (i as i64).rem_euclid(6) {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    }
}

/// GPU resources for the wireframe pass. Stored in egui's `callback_resources`.
pub struct WireResources {
    pipeline: wgpu::RenderPipeline,
    globals_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    /// Static unit-cube edge mesh, shared by every instance.
    mesh_buf: wgpu::Buffer,
    /// Per-node instance data, re-uploaded and grown as needed each frame.
    instance_buf: wgpu::Buffer,
    instance_capacity: u64,
}

impl WireResources {
    /// Build the instanced line pipeline and buffers using egui's device.
    ///
    /// Call once from `eframe::CreationContext` and insert the result into
    /// `wgpu_render_state.renderer.write().callback_resources` (under a key
    /// distinct from the star pass, or in the same map if only one wireframe
    /// overlay is used).
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("wire.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("wire.wgsl").into()),
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("wire-globals-bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let globals_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wire-globals"),
            size: std::mem::size_of::<WireGlobals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wire-globals-bg"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: globals_buf.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("wire-pl"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        // Vertex buffer 0: the shared unit-cube mesh (one vec3 per corner,
        // stepped per vertex).
        let mesh_layout = wgpu::VertexBufferLayout {
            array_stride: 12,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x3,
            }],
        };

        // Vertex buffer 1: per-node instance data, stepped per instance.
        let instance_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<WireInstanceRaw>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                // center_rel
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // half_size
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32,
                },
                // color
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("wire-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Some(mesh_layout), Some(instance_layout)],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    // Additive, matching the star pass; no depth buffer needed.
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview_mask: None,
            cache: None,
        });

        // Static mesh: upload once via mapped-at-creation (no queue available
        // here yet).
        let mesh_verts: [[f32; 3]; CUBE_VERTEX_COUNT as usize] = {
            let mut verts = [[0.0f32; 3]; CUBE_VERTEX_COUNT as usize];
            let mut i = 0;
            for edge in CUBE_EDGES {
                for corner in edge {
                    verts[i] = [corner.x as f32, corner.y as f32, corner.z as f32];
                    i += 1;
                }
            }
            verts
        };
        let mesh_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wire-cube-mesh"),
            size: (CUBE_VERTEX_COUNT as u64) * 12,
            usage: wgpu::BufferUsages::VERTEX,
            mapped_at_creation: true,
        });
        mesh_buf
            .slice(..)
            .get_mapped_range_mut()
            .expect("mesh buffer mapping")
            .copy_from_slice(bytemuck::cast_slice(&mesh_verts));
        mesh_buf.unmap();

        let instance_capacity = 4096u64;
        let instance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wire-instances"),
            size: instance_capacity * std::mem::size_of::<WireInstanceRaw>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        WireResources {
            pipeline,
            globals_buf,
            bind_group,
            mesh_buf,
            instance_buf,
            instance_capacity,
        }
    }
}

/// Per-frame wireframe data: the visited octree nodes to outline.
pub struct WireFrameInput<'a> {
    /// Galaxy owning the node bounds (params).
    pub galaxy: &'a Galaxy,
    /// Node keys to draw (e.g. the visited list from `collect_visible`).
    pub nodes: &'a [NodeKey],
    /// Camera the lines are projected from.
    pub camera: &'a Camera,
    /// Cap on nodes drawn (bounds the per-frame upload; the most important
    /// nodes come first, so truncation drops the least visible cubes).
    pub max_nodes: usize,
    /// Viewport aspect (width / height).
    pub aspect: f64,
}

/// Near-plane margin as a fraction of the closest node's distance.
const NEAR_MARGIN_FRACTION: f64 = 0.01;
/// Far-plane margin as a fraction of the farthest node's distance, plus an
/// absolute pad so a single very-close node set still gets a usable far plane.
const FAR_MARGIN_FRACTION: f64 = 1.05;
const FAR_MARGIN_ABSOLUTE: f64 = 10.0;
/// Minimum far/near ratio, so degenerate (near ≈ far) node sets stay usable.
const MIN_FAR_NEAR_RATIO: f64 = 1000.0;

/// Build per-node instance data (camera-relative, `f32`) plus the frame's
/// view-projection matrix.
///
/// Each node's center is computed in `f64` camera-relative coordinates
/// (floating origin at the cube's min corner) and only cast to `f32` once its
/// magnitude is already bounded by the node's own distance from the camera —
/// so instancing does not reintroduce the `f32` world-position precision loss
/// the original CPU-projected implementation avoided. `view_proj` encodes
/// only camera rotation and a dynamic near/far projection (no galactic-scale
/// translation), so casting it to `f32` for the shader is likewise safe.
/// Near/far planes bracket the actual node set for a sane projection.
fn build_wire_instances(
    galaxy: &Galaxy,
    nodes: &[NodeKey],
    camera: &Camera,
    aspect: f64,
) -> (Vec<WireInstanceRaw>, Mat4) {
    if nodes.is_empty() {
        return (Vec::new(), Mat4::IDENTITY);
    }

    let cam = camera.position;
    let mut min_near = f64::INFINITY;
    let mut max_far = 0.0f64;
    let mut instances: Vec<WireInstanceRaw> = Vec::with_capacity(nodes.len());

    for &key in nodes {
        let (min_m, size) = galaxy.node_bounds(key);
        // Floating origin at the box min keeps every delta exact in f64:
        // rel = camera - box_min, so center - camera = (box_min + size/2) - camera
        // = size/2 - rel (the box_min cancels).
        let origin = GalacticCoord::from_meters_f64(min_m);
        let rel = cam.relative_f64(origin);
        let center_rel = DVec3::splat(size * 0.5) - rel;
        min_near = min_near.min(galaxy.node_near_dist(key, cam).max(1.0));
        // Farthest cube corner: per axis, the farther of the two endpoints.
        let far_pt = DVec3::new(
            rel.x.abs().max((rel.x - size).abs()),
            rel.y.abs().max((rel.y - size).abs()),
            rel.z.abs().max((rel.z - size).abs()),
        );
        max_far = max_far.max(far_pt.length());

        instances.push(WireInstanceRaw {
            center_rel: center_rel.as_vec3().into(),
            half_size: (size * 0.5) as f32,
            color: depth_color(key.depth),
        });
    }

    // Near must stay inside the closest geometry; far outside the farthest.
    // No depth test is used, so the ratio only affects the divide.
    let near = (min_near * NEAR_MARGIN_FRACTION).max(1.0);
    let far = (max_far * FAR_MARGIN_FRACTION + FAR_MARGIN_ABSOLUTE).max(near * MIN_FAR_NEAR_RATIO);

    let view = DMat4::look_at_rh(DVec3::ZERO, camera.forward(), camera.up());
    let proj = DMat4::perspective_rh(camera.fov_y, aspect, near, far);
    let view_proj = (proj * view).as_mat4();

    (instances, view_proj)
}

/// A per-frame paint callback that draws the node wireframe inside egui's pass.
pub struct WireCallback {
    instances: Vec<WireInstanceRaw>,
    view_proj: Mat4,
}

impl WireCallback {
    /// Prepare per-frame instance data.
    pub fn new(input: &WireFrameInput<'_>) -> Self {
        let nodes = if input.nodes.len() > input.max_nodes {
            &input.nodes[..input.max_nodes]
        } else {
            input.nodes
        };
        let (instances, view_proj) =
            build_wire_instances(input.galaxy, nodes, input.camera, input.aspect);
        WireCallback {
            instances,
            view_proj,
        }
    }

    /// Number of node cubes that will be drawn this frame.
    pub fn instance_count(&self) -> usize {
        self.instances.len()
    }
}

impl egui_wgpu::CallbackTrait for WireCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let res: &mut WireResources = resources.get_mut().expect("WireResources missing");

        queue.write_buffer(
            &res.globals_buf,
            0,
            bytemuck::bytes_of(&WireGlobals {
                view_proj: self.view_proj.to_cols_array_2d(),
            }),
        );

        // Grow the instance buffer if needed, then upload.
        let needed = self.instances.len() as u64;
        if needed > 0 {
            if needed > res.instance_capacity {
                res.instance_capacity = needed.next_power_of_two();
                res.instance_buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("wire-instances"),
                    size: res.instance_capacity * std::mem::size_of::<WireInstanceRaw>() as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
            }
            queue.write_buffer(&res.instance_buf, 0, bytemuck::cast_slice(&self.instances));
        }

        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        if self.instances.is_empty() {
            return;
        }
        let res: &WireResources = resources.get().expect("WireResources missing");
        render_pass.set_pipeline(&res.pipeline);
        render_pass.set_bind_group(0, &res.bind_group, &[]);
        render_pass.set_vertex_buffer(0, res.mesh_buf.slice(..));
        render_pass.set_vertex_buffer(1, res.instance_buf.slice(..));
        render_pass.draw(0..CUBE_VERTEX_COUNT, 0..self.instances.len() as u32);
    }
}

#[cfg(test)]
mod wire_tests {
    use super::*;
    use galaxy_gen::GalaxyParams;
    use glam::Vec3;

    /// 8 m cube (min corner -4..+4), camera 20 m in front along +Z.
    fn fixture() -> (Galaxy, NodeKey, Camera) {
        let params = GalaxyParams {
            root_seed: 1,
            universe_size_m: 8.0,
            total_stars: 1.0e6,
            max_depth: 8,
        };
        let camera = Camera {
            position: GalacticCoord::from_meters_i128(0, 0, 20),
            ..Default::default()
        };
        (Galaxy::new(params, 16), NodeKey::ROOT, camera)
    }

    /// The camera looks down -Z with up +Y, so the view matrix is identity and
    /// view-space coordinates are recoverable from clip space: x = cx*t*aspect,
    /// y = cy*t, z = -w with t = tan(fov/2).
    fn to_view(p: [f32; 4], fov_y: f64, aspect: f64) -> DVec3 {
        let t = (fov_y * 0.5).tan();
        DVec3::new(p[0] as f64 * t * aspect, p[1] as f64 * t, -(p[3] as f64))
    }

    /// Re-derive the 24 clip-space edge endpoints for one instance exactly as
    /// the vertex shader would (mesh corner * half_size + center_rel, then
    /// view_proj), so geometry tests can reuse the CPU-projected assertions.
    fn instance_clip_points(inst: &WireInstanceRaw, view_proj: Mat4) -> Vec<[f32; 4]> {
        let center = Vec3::from(inst.center_rel);
        CUBE_EDGES
            .iter()
            .flat_map(|edge| edge.iter())
            .map(|corner| {
                let corner_f32 = Vec3::new(corner.x as f32, corner.y as f32, corner.z as f32);
                let world = center + corner_f32 * inst.half_size;
                let clip = view_proj * world.extend(1.0);
                [clip.x, clip.y, clip.z, clip.w]
            })
            .collect()
    }

    #[test]
    fn wire_instances_preserve_cube_geometry() {
        let (galaxy, key, camera) = fixture();
        let aspect = 1.6;
        let (instances, view_proj) = build_wire_instances(&galaxy, &[key], &camera, aspect);
        assert_eq!(instances.len(), 1);

        let pts = instance_clip_points(&instances[0], view_proj);
        assert_eq!(pts.len(), 24);
        assert!(pts.iter().all(|p| p[3] > 0.0), "all corners in front");

        // The 12 segments must keep the true edge length (8 m) in view space.
        for i in 0..12 {
            let a = to_view(pts[2 * i], camera.fov_y, aspect);
            let b = to_view(pts[2 * i + 1], camera.fov_y, aspect);
            assert!((a.distance(b) - 8.0).abs() < 1e-2, "edge {i} distorted");
        }

        // Each of the 8 true cube corners must appear among the endpoints.
        // View space is camera-relative (camera at z=+20): world z in
        // {-4, +4} maps to view z in {-24, -16}.
        let corners: Vec<DVec3> = pts
            .iter()
            .map(|p| to_view(*p, camera.fov_y, aspect))
            .collect();
        for sx in [-4.0, 4.0] {
            for sy in [-4.0, 4.0] {
                for sz in [-24.0, -16.0] {
                    let expected = DVec3::new(sx, sy, sz);
                    assert!(
                        corners.iter().any(|c| c.distance(expected) < 1e-2),
                        "corner {expected} missing"
                    );
                }
            }
        }
    }

    #[test]
    fn wire_instances_finite_when_camera_inside_cube() {
        let (galaxy, key, _) = fixture();
        let camera = Camera {
            position: GalacticCoord::ORIGIN, // inside the root cube
            ..Default::default()
        };
        let (instances, view_proj) = build_wire_instances(&galaxy, &[key], &camera, 1.6);
        assert_eq!(instances.len(), 1);
        assert!(instances[0].center_rel.iter().all(|c| c.is_finite()));
        assert!(instances[0].half_size.is_finite());

        let pts = instance_clip_points(&instances[0], view_proj);
        assert!(pts.iter().all(|p| p.iter().all(|c| c.is_finite())));
    }

    #[test]
    fn callback_respects_node_cap() {
        let (galaxy, key, camera) = fixture();
        let input = WireFrameInput {
            galaxy: &galaxy,
            nodes: &[key, key, key, key],
            camera: &camera,
            max_nodes: 2,
            aspect: 1.6,
        };
        let cb = WireCallback::new(&input);
        assert_eq!(cb.instance_count(), 2);
    }

    #[test]
    fn empty_node_list_yields_no_instances() {
        let (galaxy, _, camera) = fixture();
        let (instances, _) = build_wire_instances(&galaxy, &[], &camera, 1.6);
        assert!(instances.is_empty());
    }

    #[test]
    fn depth_colors_are_distinct_and_stable() {
        // Different depths should get visually distinct colors...
        let c0 = depth_color(0);
        let c1 = depth_color(1);
        let c2 = depth_color(2);
        assert_ne!(c0, c1);
        assert_ne!(c1, c2);
        // ...but the same depth always maps to the same color.
        assert_eq!(depth_color(3), depth_color(3));
        // Alpha stays constant; RGB stays in a valid, finite range.
        for c in [c0, c1, c2] {
            assert!((c[3] - 0.55).abs() < 1e-6);
            assert!(c[..3].iter().all(|v| (0.0..=1.0).contains(v)));
        }
    }
}

#[cfg(test)]
mod shader_tests {
    /// Parse every WGSL module with naga so syntax/type errors surface in
    /// `cargo test` instead of as a runtime panic in `create_shader_module`.
    #[test]
    fn wgsl_modules_parse() {
        for (name, src) in [
            ("star.wgsl", include_str!("star.wgsl")),
            ("frustum.wgsl", include_str!("frustum.wgsl")),
            ("wire.wgsl", include_str!("wire.wgsl")),
        ] {
            naga::front::wgsl::parse_str(src)
                .unwrap_or_else(|e| panic!("{name} failed to parse: {e}"));
        }
    }
}
