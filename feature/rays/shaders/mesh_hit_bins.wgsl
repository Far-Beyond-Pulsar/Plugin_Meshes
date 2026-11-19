//!use frame_ray_stream
//!use mesh_ray_rows

// The meshes' hits of a stream binned by the program that evaluates their
// material (helio-pass-mesh-rays `surfaces`), so each program's kernel runs
// over its own hits only: the wavefront renderers' sort by material (Laine,
// Karras and Aila 2013), what shader execution reordering does in hardware.
// Three passes over the stream's rows, no sort:
//
//   mesh_hit_count    each meshes hit takes its bin (`remap` of its row's
//                     program) and a rank in it: a workgroup counts its own
//                     hits a bin in shared memory and adds them to the bin
//                     once (one global atomic a bin a workgroup, not a ray)
//   mesh_hit_offsets  one invocation: each bin's list offset (a prefix sum
//                     over the bins), dispatch and count, and the counters
//                     cleared for the next frame
//   mesh_hit_scatter  each binned hit's row into its bin's list
//
// `bins` (`u32` words): bin `k` at `8k`: dispatch x, y, z (64 rays a
// workgroup, x wrapping at `HIT_ROW_GROUPS`), list offset, count, the
// count pass's counter; from `HIT_LIST` the list of ray rows, bin by bin.

@group(0) @binding(0) var<storage, read> header: RayStreamHeader;
@group(0) @binding(1) var<storage, read> hits: array<SceneHit>;
@group(0) @binding(2) var<storage, read> instances: array<RayInstance>;
// Program index (`RayInstance::program`) to bin; a program past its end is
// bin 0, the table-driven evaluation.
@group(0) @binding(3) var<storage, read> remap: array<u32>;
@group(0) @binding(4) var<storage, read_write> bins: array<atomic<u32>>;
// Each row's bin (bits 24..32) and rank (0..24); `HIT_NONE` for a row
// that is no meshes hit.
@group(0) @binding(5) var<storage, read_write> ranks: array<u32>;
// x: the tracer's id.
@group(0) @binding(6) var<uniform> hit_frame: vec4<u32>;

const HIT_BIN_WORDS: u32 = 8u;
const HIT_BINS: u32 = 64u;
const HIT_LIST: u32 = 512u;
const HIT_ROW_GROUPS: u32 = 32768u;
const HIT_NONE: u32 = 0xffffffffu;

var<workgroup> group_counts: array<atomic<u32>, 64>;
var<workgroup> group_bases: array<u32, 64>;

@compute @workgroup_size(64)
fn mesh_hit_count(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    atomicStore(&group_counts[lane], 0u);
    workgroupBarrier();
    var bin = HIT_NONE;
    var rank = 0u;
    if id.x < header.count && id.x < arrayLength(&hits) {
        let hit = hits[id.x];
        if hit.t > 0.0 && scene_hit_tracer(hit) == hit_frame.x {
            let row = scene_hit_payload(hit);
            let program = select(0u, instances[row].program, row < arrayLength(&instances));
            bin = select(0u, min(remap[program], HIT_BINS - 1u), program < arrayLength(&remap));
            rank = atomicAdd(&group_counts[bin], 1u);
        }
    }
    workgroupBarrier();
    let count = atomicLoad(&group_counts[lane]);
    if count > 0u {
        group_bases[lane] = atomicAdd(&bins[lane * HIT_BIN_WORDS + 5u], count);
    }
    workgroupBarrier();
    if id.x < arrayLength(&ranks) {
        ranks[id.x] = select((bin << 24u) | (group_bases[min(bin, HIT_BINS - 1u)] + rank), HIT_NONE, bin == HIT_NONE);
    }
}

@compute @workgroup_size(1)
fn mesh_hit_offsets() {
    var offset = 0u;
    for (var k = 0u; k < HIT_BINS; k++) {
        let at = k * HIT_BIN_WORDS;
        let count = atomicLoad(&bins[at + 5u]);
        let groups = (count + 63u) / 64u;
        atomicStore(&bins[at], min(groups, HIT_ROW_GROUPS));
        atomicStore(&bins[at + 1u], (groups + HIT_ROW_GROUPS - 1u) / HIT_ROW_GROUPS);
        atomicStore(&bins[at + 2u], 1u);
        atomicStore(&bins[at + 3u], offset);
        atomicStore(&bins[at + 4u], count);
        atomicStore(&bins[at + 5u], 0u);
        offset += count;
    }
}

@compute @workgroup_size(64)
fn mesh_hit_scatter(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= header.count || id.x >= arrayLength(&ranks) { return; }
    let r = ranks[id.x];
    if r == HIT_NONE { return; }
    let bin = r >> 24u;
    let at = HIT_LIST + atomicLoad(&bins[bin * HIT_BIN_WORDS + 3u]) + (r & 0xffffffu);
    atomicStore(&bins[at], id.x);
}
