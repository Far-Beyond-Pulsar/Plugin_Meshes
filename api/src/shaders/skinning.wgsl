// The meshes' skinning in WGSL (helio-mesh-api `SKINNING_SNIPPET`,
// `//!use mesh_skinning`): the skin cache's rows and a skinned instance's
// vertex as a drawer reads it. The palette is blended in one place, the
// meshes' skin pass, which writes the cache once a frame for the instances
// whose pose changed; every drawer and the deforming BLASes read the cache,
// so what is rasterised is what is traced. Rigid instances
// (without `INSTANCE_FLAG_SKINNED`) pass through.
//
// The includer binds `skin_cache: array<SkinnedVertex>`
// (`helio_mesh_api::SKIN_CACHE`) and `skin_cache_rows: array<SkinCacheRow>`
// (`helio_mesh_api::SKIN_CACHE_ROWS`), has a `GpuInstanceData`
// (`//!use mesh_draw_rows`) and the frame's unit-vector packing
// (`scene_unpack_unit`, `//!use frame_ray_stream`).

// `helio_mesh_api::SkinnedVertex` (32 bytes): the vertex this frame, in the
// mesh's space; normal and tangent octahedral, 16 bits an axis
// (`scene_pack_unit`, the tangent's handedness the mesh vertex's bitangent
// sign); `previous` last frame's position.
struct SkinnedVertex {
    position: vec3<f32>,
    normal: u32,
    previous: vec3<f32>,
    tangent: u32,
}

// `helio_mesh_api::SkinCacheRow`, by mesh-instance entity row (`mesh_id`):
// mesh-pool vertex `v` is cache vertex `v + offset`; `resident` 0 for an
// instance the cache does not hold; `pose` changes with each new pose.
struct SkinCacheRow {
    offset: u32,
    resident: u32,
    pose: u32,
    // Non-zero once the pose stopped: the previous position is the current.
    settled: u32,
}

// A mesh vertex in its mesh's space, this frame and last.
struct MeshSkinnedVertex {
    position: vec3<f32>,
    previous: vec3<f32>,
    normal: vec3<f32>,
    tangent: vec3<f32>,
}

// The cache vertex of `inst`'s mesh-pool vertex `vertex_index`, if the
// instance is skinned and resident (the high bit: its pose has settled).
fn mesh_skin_cache_index(inst: GpuInstanceData, vertex_index: u32) -> u32 {
    if (inst.flags & INSTANCE_FLAG_SKINNED) == 0u || inst.mesh_id >= arrayLength(&skin_cache_rows) {
        return 0xFFFFFFFFu;
    }
    let row = skin_cache_rows[inst.mesh_id];
    let index = vertex_index + row.offset;
    if row.resident == 0u || index >= arrayLength(&skin_cache) {
        return 0xFFFFFFFFu;
    }
    return index | select(0u, 0x80000000u, row.settled != 0u);
}

// `inst`'s vertex `vertex_index` (its mesh-pool index) whose rest values are
// `position`, `normal` and `tangent`: skinned from the cache, or the rest
// values for a rigid (or not yet resident) instance.
fn mesh_skinned_vertex(inst: GpuInstanceData, vertex_index: u32, position: vec3<f32>, normal: vec3<f32>, tangent: vec3<f32>) -> MeshSkinnedVertex {
    let index = mesh_skin_cache_index(inst, vertex_index);
    if index == 0xFFFFFFFFu {
        return MeshSkinnedVertex(position, position, normal, tangent);
    }
    let v = skin_cache[index & 0x7FFFFFFFu];
    let previous = select(v.previous, v.position, (index & 0x80000000u) != 0u);
    return MeshSkinnedVertex(v.position, previous, scene_unpack_unit(v.normal), scene_unpack_unit(v.tangent));
}

// `inst`'s vertex `vertex_index` position this frame (`position` its rest
// position), for drawers that need no normal.
fn mesh_skinned_position(inst: GpuInstanceData, position: vec3<f32>, vertex_index: u32) -> vec3<f32> {
    let index = mesh_skin_cache_index(inst, vertex_index);
    if index == 0xFFFFFFFFu {
        return position;
    }
    return skin_cache[index & 0x7FFFFFFFu].position;
}
