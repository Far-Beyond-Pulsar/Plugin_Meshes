// The meshes' skinning in WGSL (helio-mesh-api `SKINNING_SNIPPET`,
// `//!use mesh_skinning`): a skinned instance's vertex position, blended from
// its four influences in the shared palette. Rigid instances
// (`bone_palette_count == 0`) pass through.
//
// The includer binds `skin_vertices: array<SkinVertex>`
// (`helio_mesh_api::SKIN_VERTICES`) and `bone_palette: array<mat4x4<f32>>`
// (`helio_mesh_api::BONE_PALETTE`), and has a `GpuInstanceData` with the
// batch's skin fields (`//!use mesh_draw_rows`).

// `helio_mesh_api::SkinVertex`.
struct SkinVertex {
    joints: vec4<u32>,
    weights: vec4<f32>,
}

/// `position` (the vertex's, in the mesh's space) skinned by `inst`'s
/// current palette; `vertex_index` is the vertex's index in the mesh pool.
fn mesh_skinned_position(inst: GpuInstanceData, position: vec3<f32>, vertex_index: u32) -> vec3<f32> {
    if inst.bone_palette_count == 0u {
        return position;
    }
    let skin = skin_vertices[vertex_index + inst.skin_vertex_offset];
    var blended = mat4x4<f32>(vec4f(0.0), vec4f(0.0), vec4f(0.0), vec4f(0.0));
    var total = 0.0;
    for (var i = 0u; i < 4u; i = i + 1u) {
        if skin.weights[i] > 0.0 && skin.joints[i] < inst.bone_palette_count {
            blended += bone_palette[inst.bone_palette_offset + skin.joints[i]] * skin.weights[i];
            total += skin.weights[i];
        }
    }
    if total <= 0.00001 {
        return position;
    }
    return (blended * (1.0 / total) * vec4f(position, 1.0)).xyz;
}
