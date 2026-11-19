// The meshes' traced rows (helio-mesh-api `RAY_ROWS_SNIPPET`,
// `//!use mesh_ray_rows`): the instance rows the meshes publish beside the
// frame's TLAS (`helio_mesh_api::RAY_INSTANCES`) and their flags, each TLAS
// instance's transform (`RAY_INSTANCE_TRANSFORMS`) and the coordinate
// spaces' mask groups (`RAY_SPACE_GROUPS`). Needs no binding: a reader of
// the rows alone (a pass binning hits by program) takes this without the
// pools `mesh_ray_surface` reads.

// `helio_mesh_api::RayInstance`, a traced section's row (a hit's: its
// instance's custom data plus its `geometry_index`). `flags` holds the
// `RAY_INSTANCE_*` bits and, at `RAY_INSTANCE_SPACE_SHIFT`, the coordinate
// space slot; `instance` its TLAS instance (its `RayTransform`); `program`
// the frame's `RayPrograms` entry its material is evaluated by (0: the
// table-driven evaluation).
struct RayInstance {
    first_index: u32,
    base_vertex: u32,
    flags: u32,
    material: u32,
    transmission: vec3<f32>,
    instance: u32,
    skinned_vertex: u32,
    program: u32,
    _pad: vec2<u32>,
}

// `helio_mesh_api::RAY_INSTANCE_PROXY`: the instance's triangles are in the
// proxy pool, positions and UVs only (a virtual-geometry asset's coarse cut).
const RAY_INSTANCE_PROXY: u32 = 1u;
// `helio_mesh_api::RAY_INSTANCE_MASKED`: a cut-out, its material alpha-tested.
const RAY_INSTANCE_MASKED: u32 = 2u;
// `helio_mesh_api::RAY_INSTANCE_TWO_SIDED`: both faces are outside.
const RAY_INSTANCE_TWO_SIDED: u32 = 4u;
// `helio_mesh_api::RAY_INSTANCE_SKINNED`: positions and normals in the skin
// cache at `skinned_vertex`.
const RAY_INSTANCE_SKINNED: u32 = 8u;
// `helio_mesh_api::RAY_INSTANCE_SPACE_SHIFT`: the row's coordinate space slot.
const RAY_INSTANCE_SPACE_SHIFT: u32 = 8u;

fn ray_instance_space(instance: RayInstance) -> u32 {
    return (instance.flags >> RAY_INSTANCE_SPACE_SHIFT) & 0xffu;
}

// `helio_mesh_api::RayTransform`: a TLAS instance's object-to-frame matrix,
// three rows of a 3x4.
struct RayTransform {
    rows: array<vec4<f32>, 3>,
}

fn ray_transform_point(t: RayTransform, p: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(t.rows[0].xyz, p), dot(t.rows[1].xyz, p), dot(t.rows[2].xyz, p))
        + vec3<f32>(t.rows[0].w, t.rows[1].w, t.rows[2].w);
}

fn ray_transform_vector(t: RayTransform, v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(t.rows[0].xyz, v), dot(t.rows[1].xyz, v), dot(t.rows[2].xyz, v));
}

// An object-space normal to the frame: by the inverse transpose (the
// columns of the inverse are the rows' cross products over the
// determinant), not normalized.
fn ray_transform_normal(t: RayTransform, n: vec3<f32>) -> vec3<f32> {
    let r0 = t.rows[0].xyz;
    let r1 = t.rows[1].xyz;
    let r2 = t.rows[2].xyz;
    let c0 = cross(r1, r2);
    let det = dot(r0, c0);
    return vec3<f32>(dot(c0, n), dot(cross(r2, r0), n), dot(cross(r0, r1), n)) / select(det, 1e-30, det == 0.0);
}

// `helio_mesh_api::RAY_SPACE_GROUPS` entries: the group, shared, present.
const RAY_SPACE_GROUP: u32 = 3u;
const RAY_SPACE_SHARED: u32 = 4u;
const RAY_SPACE_PRESENT: u32 = 8u;

// The TLAS cull mask of a ray of caster classes `classes` (the frame's ray
// caster mask: bit 0 casters that never move, bit 1 those that may) in
// coordinate space `space` (`SceneRay::space`), `entry` that space's
// `RAY_SPACE_GROUPS` entry (unread for a ray of every space): x the mask (0:
// no instance of the space is traced), y 1 where the space's group is shared
// (its candidates are then tested by `ray_instance_space`).
fn ray_space_cull(classes: u32, space: u32, entry: u32) -> vec2<u32> {
    let c = classes & 3u;
    if space == 0u { return vec2<u32>(c * 0x55u, 0u); }
    if (entry & RAY_SPACE_PRESENT) == 0u { return vec2<u32>(0u); }
    return vec2<u32>(c << (2u * (entry & RAY_SPACE_GROUP)), select(0u, 1u, (entry & RAY_SPACE_SHARED) != 0u));
}
