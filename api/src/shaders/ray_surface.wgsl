// The meshes' surfaces as rays see them (helio-mesh-api `RAY_SURFACE_SNIPPET`,
// `//!use mesh_ray_surface`): the instance rows the meshes publish beside the
// frame's TLAS, the mesh pools' and the proxy pool's vertex decode, and a hit
// triangle's fetch. The reader declares the storage these read, under these
// names:
//   instances: array<RayInstance>   (`helio_mesh_api::RAY_INSTANCES`)
//   vertices: array<u32>            (`helio_mesh_api::MESH_VERTICES`)
//   indices: array<u32>             (`helio_mesh_api::MESH_INDICES`)
//   proxy_vertices: array<f32>      (`helio_mesh_api::RAY_PROXY_VERTICES`)
//   proxy_indices: array<u32>       (`helio_mesh_api::RAY_PROXY_INDICES`)
//   skinned_vertices: array<u32>    (`helio_mesh_api::RAY_SKINNED_VERTICES`)
// and, a pass that shades hits without the ray query, the instances'
// transforms (`helio_mesh_api::RAY_INSTANCE_TRANSFORMS`, `RayTransform`).

// `helio_mesh_api::RayInstance`, a traced section's row (a hit's: its
// instance's custom data plus its `geometry_index`). `flags` holds the
// `RAY_INSTANCE_*` bits and, at `RAY_INSTANCE_SPACE_SHIFT`, the coordinate
// space slot; `instance` its TLAS instance (its `RayTransform`).
struct RayInstance {
    first_index: u32,
    base_vertex: u32,
    flags: u32,
    material: u32,
    base_color: vec4<f32>,
    emissive: vec3<f32>,
    instance: u32,
    transmission: vec3<f32>,
    skinned_vertex: u32,
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
// `helio_mesh_api::SkinnedVertex` in `u32` words: position (words 0-2), the
// octahedral normal (3, `scene_pack_unit`: the reader has `frame_ray_stream`).
const SKINNED_VERTEX_WORDS: u32 = 8u;
// `helio_mesh_api::PackedVertex` in `u32` words: position (words 0-2), UV0
// (4-5), the snorm8x4 normal (8).
const MESH_VERTEX_WORDS: u32 = 10u;
// A proxy pool vertex in floats (`helio_mesh_api::RAY_PROXY_VERTEX_BYTES`):
// position, then UV0.
const RAY_PROXY_FLOATS: u32 = 5u;

fn mesh_vertex_position(v: u32) -> vec3<f32> {
    let at = v * MESH_VERTEX_WORDS;
    return vec3<f32>(bitcast<f32>(vertices[at]), bitcast<f32>(vertices[at + 1u]), bitcast<f32>(vertices[at + 2u]));
}

fn mesh_vertex_uv(v: u32) -> vec2<f32> {
    let at = v * MESH_VERTEX_WORDS + 4u;
    return vec2<f32>(bitcast<f32>(vertices[at]), bitcast<f32>(vertices[at + 1u]));
}

fn mesh_vertex_normal(v: u32) -> vec3<f32> {
    return unpack4x8snorm(vertices[v * MESH_VERTEX_WORDS + 8u]).xyz;
}

fn skinned_vertex_position(v: u32) -> vec3<f32> {
    let at = v * SKINNED_VERTEX_WORDS;
    return vec3<f32>(bitcast<f32>(skinned_vertices[at]), bitcast<f32>(skinned_vertices[at + 1u]), bitcast<f32>(skinned_vertices[at + 2u]));
}

fn skinned_vertex_normal(v: u32) -> vec3<f32> {
    return scene_unpack_unit(skinned_vertices[v * SKINNED_VERTEX_WORDS + 3u]);
}

fn ray_proxy_position(v: u32) -> vec3<f32> {
    let at = RAY_PROXY_FLOATS * v;
    return vec3<f32>(proxy_vertices[at], proxy_vertices[at + 1u], proxy_vertices[at + 2u]);
}

fn ray_proxy_uv(v: u32) -> vec2<f32> {
    let at = RAY_PROXY_FLOATS * v + 3u;
    return vec2<f32>(proxy_vertices[at], proxy_vertices[at + 1u]);
}

// A hit's triangle, in object space: its corners and UV0s, its pool
// vertices, whether it is a proxy's (no vertex normals), and for a skinned
// instance its skin cache vertices (where its corners and normals are).
struct RayTriangle {
    corners: array<vec3<f32>, 3>,
    uv: array<vec2<f32>, 3>,
    vertices: vec3<u32>,
    proxy: bool,
    skinned: bool,
    cached: vec3<u32>,
}

// Triangle `primitive` of `instance` (a hit's `primitive_index`), from the
// mesh pools or the proxy pool.
fn ray_triangle(instance: RayInstance, primitive: u32) -> RayTriangle {
    let base = instance.first_index + 3u * primitive;
    var t: RayTriangle;
    t.proxy = (instance.flags & RAY_INSTANCE_PROXY) != 0u;
    t.skinned = (instance.flags & RAY_INSTANCE_SKINNED) != 0u;
    if t.proxy {
        t.vertices = vec3<u32>(proxy_indices[base], proxy_indices[base + 1u], proxy_indices[base + 2u]) + instance.base_vertex;
        t.corners = array<vec3<f32>, 3>(ray_proxy_position(t.vertices.x), ray_proxy_position(t.vertices.y), ray_proxy_position(t.vertices.z));
        t.uv = array<vec2<f32>, 3>(ray_proxy_uv(t.vertices.x), ray_proxy_uv(t.vertices.y), ray_proxy_uv(t.vertices.z));
    } else {
        let local = vec3<u32>(indices[base], indices[base + 1u], indices[base + 2u]);
        t.vertices = local + instance.base_vertex;
        t.uv = array<vec2<f32>, 3>(mesh_vertex_uv(t.vertices.x), mesh_vertex_uv(t.vertices.y), mesh_vertex_uv(t.vertices.z));
        if t.skinned {
            t.cached = local + instance.skinned_vertex;
            t.corners = array<vec3<f32>, 3>(skinned_vertex_position(t.cached.x), skinned_vertex_position(t.cached.y), skinned_vertex_position(t.cached.z));
        } else {
            t.corners = array<vec3<f32>, 3>(mesh_vertex_position(t.vertices.x), mesh_vertex_position(t.vertices.y), mesh_vertex_position(t.vertices.z));
        }
    }
    return t;
}

// The side the mesh calls outside at barycentrics `b` (object space, not
// normalized): its vertex normals blended, or a proxy's winding (its face).
fn ray_triangle_outside(t: RayTriangle, b: vec3<f32>) -> vec3<f32> {
    if t.proxy {
        return cross(t.corners[1] - t.corners[0], t.corners[2] - t.corners[0]);
    }
    if t.skinned {
        return skinned_vertex_normal(t.cached.x) * b.x + skinned_vertex_normal(t.cached.y) * b.y + skinned_vertex_normal(t.cached.z) * b.z;
    }
    return mesh_vertex_normal(t.vertices.x) * b.x + mesh_vertex_normal(t.vertices.y) * b.y + mesh_vertex_normal(t.vertices.z) * b.z;
}
