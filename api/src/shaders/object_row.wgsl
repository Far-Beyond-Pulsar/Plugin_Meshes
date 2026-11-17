// The meshes' object row in WGSL (helio-mesh-api `OBJECT_ROW_SNIPPET`,
// `//!use mesh_object_row`): `helio_mesh_api::StaticObjectRow`, 252 bytes,
// the `static_objects` rows the scene join writes and every pass that reads
// an object's placement, mesh, material and flags. Matrices are arrays of
// columns (WGSL's `mat4x4` alignment would not match the packed Rust row).

struct StaticObjectRow {
    mesh_slot: u32,
    mesh_generation: u32,
    material_slot: u32,
    material_generation: u32,
    transform: array<array<f32, 4>, 4>,
    prev_transform: array<array<f32, 4>, 4>,
    normal_mat: array<array<f32, 4>, 3>,
    bounds: array<f32, 4>,
    index_count: u32,
    first_index: u32,
    vertex_offset: i32,
    material_class: u32,
    graph_hash_lo: u32,
    graph_hash_hi: u32,
    flags: u32,
    skin_vertex_offset: u32,
    bone_palette_offset: u32,
    bone_palette_count: u32,
    // Which section of its mesh the row draws.
    section_index: u32,
}
