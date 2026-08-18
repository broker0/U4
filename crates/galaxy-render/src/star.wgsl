// Star billboard shader (directional sky-sphere) with additive glow.
//
// Each instance is a star given as a *unit direction* from the camera, placed
// on a sphere of radius 1. The camera sits at the origin looking down -Z, so
// projection never depends on galactic distances (that only feeds brightness,
// precomputed on the CPU into `intensity`).

struct Globals {
    view_proj : mat4x4<f32>,
    px_scale  : f32,
    _pad0     : f32,
    viewport  : vec2<f32>,
};

@group(0) @binding(0) var<uniform> globals : Globals;

struct Instance {
    @location(0) dir       : vec3<f32>,   // unit direction from camera
    @location(1) color     : vec3<f32>,
    @location(2) intensity : f32,         // linear brightness, >0
    @location(3) core_px   : f32,         // core radius in pixels
};

struct VsOut {
    @builtin(position) clip   : vec4<f32>,
    @location(0)       uv     : vec2<f32>,
    @location(1)       color  : vec3<f32>,
    @location(2)       intensity : f32,
};

// Two-triangle quad corners.
var<private> CORNERS : array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>( 1.0,  1.0),
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0,  1.0),
    vec2<f32>(-1.0,  1.0),
);

@vertex
fn vs_main(inst : Instance, @builtin(vertex_index) vid : u32) -> VsOut {
    let corner = CORNERS[vid];

    // Place the star on the unit sphere and project.
    var center = globals.view_proj * vec4<f32>(inst.dir, 1.0);

    // Behind the camera: clip it out.
    if (center.w <= 0.0) {
        var out : VsOut;
        out.clip = vec4<f32>(0.0, 0.0, 2.0, 1.0);
        out.uv = corner;
        out.color = inst.color;
        out.intensity = 0.0;
        return out;
    }

    // Keep most stars near point-like while allowing gentle growth for bright ones.
    let i = max(inst.intensity, 0.0);
    let size_gain = log2(1.0 + i * 2.0);
    let radius_px = min(inst.core_px + globals.px_scale * size_gain, 3.5);
    let ndc_per_px = 2.0 / globals.viewport;
    let offset_ndc = corner * radius_px * ndc_per_px * center.w;

    var out : VsOut;
    out.clip = vec4<f32>(center.xy + offset_ndc, center.z, center.w);
    out.uv = corner;
    out.color = inst.color;
    out.intensity = inst.intensity;
    return out;
}

@fragment
fn fs_main(in : VsOut) -> @location(0) vec4<f32> {
    let r2 = dot(in.uv, in.uv);
    if (r2 > 1.0) {
        discard;
    }
    // Tighter point core and weaker halo to avoid "ball" stars.
    let core = exp(-r2 * 20.0);
    let halo = exp(-r2 * 3.5) * 0.12;
    let a = (core + halo) * min(in.intensity, 1.5);
    return vec4<f32>(in.color * a, a);
}
