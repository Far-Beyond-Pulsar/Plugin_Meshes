/// Depth-clear vertex shader: every serviced face of a layer in one draw.
///
/// `ShadowPass` writes one job per (layer, serviced face) and draws
/// `64 * jobs` instances of a six-vertex quad: instance `i` is page `i % 64`
/// of job `i / 64`. A page is one cell of the face's 8x8 grid; the quad is
/// placed in the whole atlas's clip space from the face's tile, so no viewport
/// or scissor is set per face (a viewport per face made the CPU record and wgpu
/// validate a state change per face and layer, every frame).
///
/// When the layer is gated on a dirty bit (`gate`: 1 dynamic, 2 static) a clean
/// face's quads are moved off screen, and the dynamic layer clears only the
/// pages in the face's page mask, so the rest of the face keeps its cached
/// depth. `gate` 0 clears every page.
///
/// Pipeline state:
///   - DepthCompare: Always  →  overwrites whatever is in the depth buffer.
///   - DepthWrite:   true
///   - Fragment:     none    →  depth write happens automatically.
///   - CullMode:     none    →  the quads are always front-facing.

/// `jobs[0].x` is the atlas's edge in texels; job `j` is `jobs[1 + j]`:
/// (face, gate, tile x | tile y << 16, tile size).
@group(0) @binding(0) var<storage, read> jobs: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> dirty: array<u32>;
/// Per-face 8x8 page masks of the dynamic layer (two words per face).
@group(0) @binding(2) var<storage, read> pages: array<vec2<u32>>;

const OFF_SCREEN: vec4f = vec4f(2.0, 2.0, 2.0, 1.0);

@vertex
fn vs_main(@builtin(vertex_index) vid: u32, @builtin(instance_index) instance: u32) -> @builtin(position) vec4<f32> {
    let job = jobs[1u + instance / 64u];
    let page = instance % 64u;
    let face = job.x;
    let gate = job.y;
    if gate != 0u && (dirty[face] & gate) == 0u { return OFF_SCREEN; }
    if gate == 1u {
        let mask = pages[face];
        let word = select(mask.y, mask.x, page < 32u);
        if ((word >> (page & 31u)) & 1u) == 0u { return OFF_SCREEN; }
    }
    // Page `page` is column `page % 8`, row `page / 8` (rows count up from the
    // bottom of the tile). Two triangles per page.
    var corners = array<vec2f, 6>(
        vec2f(0, 0), vec2f(1, 0), vec2f(0, 1),
        vec2f(0, 1), vec2f(1, 0), vec2f(1, 1),
    );
    let cell = vec2f(f32(page % 8u), f32(page / 8u));
    let local = (cell + corners[vid]) * 0.125;
    let atlas = f32(jobs[0].x);
    let origin = vec2f(f32(job.z & 0xffffu), f32(job.z >> 16u));
    // Texel position in the atlas (y down), then its clip position.
    let texel = origin + vec2f(local.x, 1.0 - local.y) * f32(job.w);
    let ndc = vec2f(texel.x / atlas * 2.0 - 1.0, 1.0 - texel.y / atlas * 2.0);
    return vec4f(ndc, 1.0, 1.0);
}
