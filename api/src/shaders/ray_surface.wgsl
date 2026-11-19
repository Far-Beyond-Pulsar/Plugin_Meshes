// The meshes' surfaces as rays see them (helio-mesh-api `RAY_SURFACE_SNIPPET`,
// `//!use mesh_ray_surface`, with `//!use mesh_ray_rows` and
// `//!use frame_ray_stream`): the mesh pools' and the proxy pool's vertex
// decode, a hit triangle's fetch, its tangent frame and the normals a hit
// is oriented by. The reader declares the storage these read, under these
// names:
//   instances: array<RayInstance>   (`helio_mesh_api::RAY_INSTANCES`)
//   vertices: array<u32>            (`helio_mesh_api::MESH_VERTICES`)
//   indices: array<u32>             (`helio_mesh_api::MESH_INDICES`)
//   proxy_vertices: array<f32>      (`helio_mesh_api::RAY_PROXY_VERTICES`)
//   proxy_indices: array<u32>       (`helio_mesh_api::RAY_PROXY_INDICES`)
//   skinned_vertices: array<u32>    (`helio_mesh_api::RAY_SKINNED_VERTICES`)
// and, a pass that shades hits without the ray query, the instances'
// transforms (`helio_mesh_api::RAY_INSTANCE_TRANSFORMS`, `RayTransform`).

// `helio_mesh_api::SkinnedVertex` in `u32` words: position (words 0-2), the
// octahedral normal (3, `scene_pack_unit`: the reader has `frame_ray_stream`),
// the octahedral tangent (7).
const SKINNED_VERTEX_WORDS: u32 = 8u;
// `helio_mesh_api::PackedVertex` in `u32` words: position (words 0-2), the
// bitangent's sign (3), UV0 (4-5), the snorm8x4 normal (8) and tangent (9).
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

fn mesh_vertex_tangent(v: u32) -> vec3<f32> {
    return unpack4x8snorm(vertices[v * MESH_VERTEX_WORDS + 9u]).xyz;
}

fn mesh_vertex_bitangent_sign(v: u32) -> f32 {
    return bitcast<f32>(vertices[v * MESH_VERTEX_WORDS + 3u]);
}

fn skinned_vertex_position(v: u32) -> vec3<f32> {
    let at = v * SKINNED_VERTEX_WORDS;
    return vec3<f32>(bitcast<f32>(skinned_vertices[at]), bitcast<f32>(skinned_vertices[at + 1u]), bitcast<f32>(skinned_vertices[at + 2u]));
}

fn skinned_vertex_normal(v: u32) -> vec3<f32> {
    return scene_unpack_unit(skinned_vertices[v * SKINNED_VERTEX_WORDS + 3u]);
}

fn skinned_vertex_tangent(v: u32) -> vec3<f32> {
    return scene_unpack_unit(skinned_vertices[v * SKINNED_VERTEX_WORDS + 7u]);
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

// The tangent along +u at barycentrics `b` (object space, not normalized;
// xyz) and the bitangent's sign (w), as the G-buffer's vertex stage hands
// them to a material: the vertices' tangents blended (a skinned instance's
// from the skin cache) and the first corner's sign (a mesh's sign is one a
// triangle). A proxy has no vertex tangents: its triangle's direction of
// +u on its face, the sign the side +v takes.
fn ray_triangle_tangent(t: RayTriangle, b: vec3<f32>) -> vec4<f32> {
    if t.proxy {
        let e1 = t.corners[1] - t.corners[0];
        let e2 = t.corners[2] - t.corners[0];
        let d1 = t.uv[1] - t.uv[0];
        let d2 = t.uv[2] - t.uv[0];
        let det = d1.x * d2.y - d1.y * d2.x;
        let r = select(1.0 / det, 0.0, abs(det) < 1e-20);
        let tangent = (e1 * d2.y - e2 * d1.y) * r;
        let bitangent = (e2 * d1.x - e1 * d2.x) * r;
        let n = cross(e1, e2);
        return vec4<f32>(tangent, select(1.0, -1.0, dot(cross(n, tangent), bitangent) < 0.0));
    }
    let sign = mesh_vertex_bitangent_sign(t.vertices.x);
    if t.skinned {
        return vec4<f32>(skinned_vertex_tangent(t.cached.x) * b.x + skinned_vertex_tangent(t.cached.y) * b.y + skinned_vertex_tangent(t.cached.z) * b.z, sign);
    }
    return vec4<f32>(mesh_vertex_tangent(t.vertices.x) * b.x + mesh_vertex_tangent(t.vertices.y) * b.y + mesh_vertex_tangent(t.vertices.z) * b.z, sign);
}

// The side triangle `t` calls outside at barycentrics `b`, in object space:
// its vertex normals blended; a proxy's winding (no vertex normals), unit.
fn mesh_ray_outside(t: RayTriangle, b: vec3<f32>) -> vec3<f32> {
    let local = ray_triangle_outside(t, b);
    return select(local, local / max(length(local), 1e-30), t.proxy);
}

// The face's and the shading normal at a hit, both on the side the ray
// met, unit: from `face_in` the triangle's frame-space cross product and
// `shading_in` its outside (`mesh_ray_outside`) taken to the frame by the
// inverse transpose; whether the ray met the back of a single-sided
// surface (`inside`); and whether the shading normal was turned to the
// side the ray met (`flipped`: a material's tangent frame turns with it).
struct MeshRayNormals {
    face: vec3<f32>,
    shading: vec3<f32>,
    inside: bool,
    flipped: bool,
}

fn mesh_ray_normals(instance: RayInstance, face_in: vec3<f32>, shading_in: vec3<f32>, dir: vec3<f32>) -> MeshRayNormals {
    var face = face_in / max(length(face_in), 1e-20);
    var shading = shading_in;
    if dot(shading, shading) < 1e-8 {
        // No vertex normals: two-sided.
        shading = -dir;
    }
    shading = normalize(shading);
    if dot(face, shading) < 0.0 { face = -face; }
    var flipped = false;
    // Two-sided: the side the ray meets is its outside.
    if (instance.flags & RAY_INSTANCE_TWO_SIDED) != 0u && dot(face, dir) > 0.0 {
        face = -face;
        shading = -shading;
        flipped = true;
    }
    // Its inside: the back of a single-sided surface (the ray starts in a
    // wall or a solid). The surface's normals are then those of the side
    // the ray met.
    let inside = dot(face, dir) > 0.0;
    if inside {
        face = -face;
        shading = -shading;
        flipped = !flipped;
    }
    return MeshRayNormals(face, shading, inside, flipped);
}
