/// Retires the faces `ShadowPass` serviced this frame, after their depth has
/// been drawn: publishes each face's matrix and clears its dirty state. One
/// dispatch for all of them, where the pass used to record a buffer copy and
/// five buffer clears per face.
/// `LightMatrix` is six `vec4<u32>`s (96 bytes); its matrix the first four.
const MATRIX_VECS: u32 = 4u;
const FACE_VECS: u32 = 6u;

/// `jobs[0]` is the face count, `jobs[1]` the faces a region of `face_geom_count`
/// holds (it has three regions: movable, static and transmissive casters), and
/// `jobs[2 + i]` the i-th serviced face.
@group(0) @binding(0) var<storage, read> jobs: array<u32>;
@group(0) @binding(1) var<storage, read> desired: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> published: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> face_dirty: array<u32>;
@group(0) @binding(4) var<storage, read_write> face_pages: array<u32>;
@group(0) @binding(5) var<storage, read_write> face_geom_count: array<u32>;

@compute @workgroup_size(64)
fn retire(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= jobs[0] { return; }
    let face = jobs[2u + gid.x];
    let max_faces = jobs[1];
    for (var i = 0u; i < MATRIX_VECS; i++) {
        published[face * FACE_VECS + i] = desired[face * FACE_VECS + i];
    }
    face_dirty[face] = 0u;
    face_pages[face * 2u] = 0u;
    face_pages[face * 2u + 1u] = 0u;
    face_geom_count[face] = 0u;
    face_geom_count[max_faces + face] = 0u;
    face_geom_count[2u * max_faces + face] = 0u;
}
