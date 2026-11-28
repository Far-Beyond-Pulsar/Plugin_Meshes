//!use mesh_draw_rows
//!use mesh_skinning
//!use frame_ray_stream

// The meshes' skin pass (`crate::skin_cache`): the one place a skinned
// instance's palette is blended. A job is a resident instance whose pose
// changed this frame (or that settles after one): each lane writes one of its
// vertices into the skin cache (`helio_mesh_api::SKIN_CACHE`), which every
// drawer and the instance's deforming BLAS read. The blend is the vertex
// shaders' it replaced: four influences, normalized by their total weight,
// the rest vertex where they sum to nothing.

// `helio_mesh_api::SkinVertex`.
struct SkinVertex {
    joints: vec4<u32>,
    weights: vec4<f32>,
}

// `crate::skin_cache::SkinJob` (32 bytes).
struct SkinJob {
    // First vertex of the instance's mesh in the mesh pool.
    mesh_vertex: u32,
    // First of its skin influences in the skin pool.
    skin_vertex: u32,
    // First matrix of its palette in the palette pool (the current joints,
    // then the previous pose's), and the joints.
    palette: u32,
    joints: u32,
    // First of its vertices in the cache, and their count.
    cache: u32,
    count: u32,
    // `SKIN_FRESH` or `SKIN_POSE`.
    mode: u32,
    // Its first workgroup of the dispatch.
    first_group: u32,
}

// `crate::skin_cache::SkinFrame`.
struct SkinFrame {
    jobs: u32,
    groups: u32,
    groups_x: u32,
    _pad: u32,
}

// Its first pose: no motion (the previous position is the new one).
const SKIN_FRESH: u32 = 0u;
// A new pose: the previous position is the previous palette's blend (the
// pose shown before it), computed here rather than read back from the cache:
// the pass reads only the rest mesh and the palettes and writes each vertex
// once.
const SKIN_POSE: u32 = 1u;
// `helio_mesh_api::PackedVertex` in `u32` words: position (0-2), the snorm8x4
// normal (6) and tangent (7).
const PACKED_VERTEX_WORDS: u32 = 8u;
const LANES: u32 = 64u;

@group(0) @binding(0) var<uniform> frame: SkinFrame;
@group(0) @binding(1) var<storage, read> jobs: array<SkinJob>;
@group(0) @binding(2) var<storage, read> mesh_vertices: array<u32>;
@group(0) @binding(3) var<storage, read> skin_vertices: array<SkinVertex>;
@group(0) @binding(4) var<storage, read> bone_palette: array<mat4x4<f32>>;
@group(0) @binding(5) var<storage, read_write> skin_cache: array<SkinnedVertex>;
@group(0) @binding(6) var<storage, read> skin_cache_rows: array<SkinCacheRow>;

@compute @workgroup_size(64)
fn skin(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let group = wg.y * frame.groups_x + wg.x;
    if group >= frame.groups || frame.jobs == 0u {
        return;
    }
    // The job whose groups hold this one (jobs in group order).
    var lo = 0u;
    var hi = frame.jobs - 1u;
    while lo < hi {
        let mid = (lo + hi + 1u) / 2u;
        if jobs[mid].first_group <= group {
            lo = mid;
        } else {
            hi = mid - 1u;
        }
    }
    let job = jobs[lo];
    let i = (group - job.first_group) * LANES + lane;
    if i >= job.count {
        return;
    }
    let out = job.cache + i;
    if out >= arrayLength(&skin_cache) {
        return;
    }
    let at = (job.mesh_vertex + i) * PACKED_VERTEX_WORDS;
    let rest = vec3<f32>(bitcast<f32>(mesh_vertices[at]), bitcast<f32>(mesh_vertices[at + 1u]), bitcast<f32>(mesh_vertices[at + 2u]));
    var position = rest;
    var normal = unpack4x8snorm(mesh_vertices[at + 6u]).xyz;
    var tangent = unpack4x8snorm(mesh_vertices[at + 7u]).xyz;
    let skin = skin_vertices[job.skin_vertex + i];
    let pose = job.mode == SKIN_POSE;
    var blended = mat4x4<f32>(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
    var before = blended;
    var total = 0.0;
    for (var k = 0u; k < 4u; k = k + 1u) {
        let weight = skin.weights[k];
        let joint = skin.joints[k];
        if weight > 0.0 && joint < job.joints {
            blended += bone_palette[job.palette + joint] * weight;
            if pose {
                before += bone_palette[job.palette + job.joints + joint] * weight;
            }
            total += weight;
        }
    }
    var previous = rest;
    if total > 0.00001 {
        blended = blended * (1.0 / total);
        position = (blended * vec4<f32>(rest, 1.0)).xyz;
        previous = (before * (1.0 / total) * vec4<f32>(rest, 1.0)).xyz;
        let rotation = mat3x3<f32>(blended[0].xyz, blended[1].xyz, blended[2].xyz);
        normal = normalize(rotation * normal);
        tangent = normalize(rotation * tangent);
    }
    if !pose {
        previous = position;
    }
    skin_cache[out] = SkinnedVertex(position, scene_pack_unit(normal), previous, scene_pack_unit(tangent));
}
