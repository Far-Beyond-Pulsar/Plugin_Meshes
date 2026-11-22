// The meshes' object row in WGSL (helio-mesh-api `OBJECT_ROW_SNIPPET`,
// `//!use mesh_object_row`): `helio_mesh_api::StaticObjectRow`, 240 bytes,
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
    // Which section of its mesh the row draws.
    section_index: u32,
}

// `helio_mesh_api::RASTER_FLAG_REVERSED_WINDING` in a draw segment's
// material class (raster flags are its upper 16 bits): the segment's
// instances mirror, and a pass draws them with the opposite front face.
const SEGMENT_RASTER_REVERSED_WINDING: u32 = 1u << 31u;

// Whether the row's transform mirrors (a negative determinant): its
// triangles wind the other way in the world.
fn object_row_mirrors(row: StaticObjectRow) -> bool {
    let t = row.transform;
    let x = vec3<f32>(t[0][0], t[0][1], t[0][2]);
    let y = vec3<f32>(t[1][0], t[1][1], t[1][2]);
    let z = vec3<f32>(t[2][0], t[2][1], t[2][2]);
    return dot(cross(x, y), z) < 0.0;
}
