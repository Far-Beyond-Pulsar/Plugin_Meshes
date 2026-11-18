//!use helio_prelude
//!use mesh_draw_rows
//!use mesh_skinning

// The selected meshes' depth into the selection mask
// (`helio_frame_api::selection::MASK_DEPTH`): placed exactly as the G-buffer
// places them (skinning, coordinate space, projection after view), but
// without the temporal jitter, so the outline drawn from the mask holds still
// while the scene resolves temporally. Depth only: no fragment stage.

@group(0) @binding(0) var<storage, read> cameras: array<Camera, 2>;
@group(0) @binding(1) var<storage, read> instance_data: array<GpuInstanceData>;
@group(0) @binding(2) var<storage, read> compacted_indices: array<u32>;
@group(0) @binding(3) var<storage, read> coordinate_spaces: array<mat4x4<f32>>;
// The skin cache (`helio_mesh_api::SKIN_CACHE`, `mesh_skinning`).
@group(0) @binding(4) var<storage, read> skin_cache: array<SkinnedVertex>;
@group(0) @binding(5) var<storage, read> skin_cache_rows: array<SkinCacheRow>;

// `slot` is per-instance vertex data, offset by the draw's `first_instance`
// on every backend (`helio_mesh_api::draw_slots`): an index into
// `compacted_indices`, as the selection's draws were written.
@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @builtin(vertex_index) vertex_index: u32,
    @location(15) slot: u32,
) -> @invariant @builtin(position) vec4<f32> {
    let inst = instance_data[compacted_indices[slot]];
    let local_position = mesh_skinned_position(inst, position, vertex_index);
    let space = coordinate_spaces[instance_coordinate_space(inst.flags)];
    let world = space * (inst.transform * vec4<f32>(local_position, 1.0));
    let clip = cameras[0].proj * (cameras[0].view * world);
    // The jitter is a translation in NDC (`jitter_frame.xy`): take it out.
    return vec4<f32>(clip.xy - cameras[0].jitter_frame.xy * clip.w, clip.zw);
}
