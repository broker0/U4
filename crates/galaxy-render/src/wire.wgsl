// Octree node wireframe: instanced cube edges.
//
// The mesh (vertex buffer 0) is the 12-edge unit cube in {-1, +1}^3, shared by
// every instance. Vertex buffer 1 carries one entry per octree node: its
// camera-relative center, half-size, and depth color. The camera-relative
// position is computed on the CPU in full f64 precision (floating origin at
// the camera), then cast to f32 only *after* subtracting the camera, so the
// magnitude staying in the f32 buffer is bounded by the node's own distance
// rather than by absolute galactic coordinates. `view_proj` only encodes
// rotation + a dynamic near/far projection (no galactic-scale translation),
// so it is safe to cast down to f32 as well. Additive blending matches the
// star pass and needs no depth buffer.

struct Globals {
    view_proj : mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> globals : Globals;

struct Instance {
    @location(1) center_rel : vec3<f32>,
    @location(2) half_size  : f32,
    @location(3) color      : vec4<f32>,
};

struct VsOut {
    @builtin(position) clip  : vec4<f32>,
    @location(0)       color : vec4<f32>,
};

@vertex
fn vs_main(@location(0) corner : vec3<f32>, inst : Instance) -> VsOut {
    let world_rel = inst.center_rel + corner * inst.half_size;
    var out : VsOut;
    out.clip = globals.view_proj * vec4<f32>(world_rel, 1.0);
    out.color = inst.color;
    return out;
}

@fragment
fn fs_main(in : VsOut) -> @location(0) vec4<f32> {
    return in.color;
}
