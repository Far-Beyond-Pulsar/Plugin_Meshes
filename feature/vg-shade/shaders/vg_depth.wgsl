// Final depth: writes the resolved VG depth with frag_depth. The pipeline
// depth-tests (reversed-Z, >=) against the classic G-buffer depth so VG and
// classic geometry composite in either order; non-VG pixels are discarded.
//
// `fs_main` also writes the view's plain tag (`gbuffer_view_tag`) to the
// surface target for the pixels it keeps: that target is Rg16Float, which
// cannot be a storage texture without an extra device feature, so it is
// written by this render pass instead of by the shade compute.
// `fs_depth_only` is the same without the colour output.

//!use helio_prelude
@group(0) @binding(0) var resolved_depth: texture_2d<f32>;
// The view the pass draws: its pixels carry it in the surface tag.
@group(0) @binding(1) var<storage, read> cameras: array<Camera, 2>;

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

struct DepthOut {
    @location(0) surface_tag: vec2<f32>,
    @builtin(frag_depth) depth: f32,
}

struct DepthOnly {
    @builtin(frag_depth) depth: f32,
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> DepthOut {
    let d = textureLoad(resolved_depth, vec2<i32>(pos.xy), 0).x;
    if d <= 0.0 {
        discard;
    }
    return DepthOut(gbuffer_view_tag(cameras[0], GBUFFER_TAG_PLAIN), d);
}

@fragment
fn fs_depth_only(@builtin(position) pos: vec4<f32>) -> DepthOnly {
    let d = textureLoad(resolved_depth, vec2<i32>(pos.xy), 0).x;
    if d <= 0.0 {
        discard;
    }
    return DepthOnly(d);
}
