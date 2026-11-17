//!use mesh_draw_rows
//!use portal_chain
// The meshes' occurrences through portal chains (`helio_portal_api`): which
// instances each live chain shows, packed per draw group so the G-buffer's
// occurrence draw is one indirect record per group, as its direct draw.
//
// An instance occurs once per chain whose route it may show through: its
// bounding sphere, in its own world (its coordinate space applied), lies
// beyond every opening of the chain (not wholly on the camera's side of
// one) and, mapped through the chain, inside the camera's frustum. The
// fragment stage clips exactly (the aperture mask and each stage's ray
// test); this only keeps what cannot show from being drawn.
//
// Three dispatches, so no group needs a reserved region:
//   1. `count_occurrences`: per (group, chain), how many occurrences.
//   2. `scan_occurrences`: one workgroup turns the per-group counts into
//      offsets and the per-group indirect records (clamped to the buffers'
//      capacity; the total goes back to the CPU, which grows them).
//   3. `write_occurrences`: per (group, chain), the occurrences themselves:
//      the instance slot and the chain, at the group's offset.

struct OccurrenceParams {
    frustum_planes: array<vec4<f32>, 6>,
    /// Rows `draw_calls` holds (the live count is `batch_counts[0]`).
    draw_capacity:  u32,
    chain_count:    u32,
    /// Occurrences the output buffers hold.
    capacity:       u32,
    /// Workgroups along x per row of z (group = x + z * this).
    groups_per_row: u32,
}

@group(0) @binding(0) var<uniform> params: OccurrenceParams;
@group(0) @binding(1) var<storage, read> instances: array<GpuInstanceData>;
@group(0) @binding(2) var<storage, read> draw_calls: array<GpuDrawCall>;
@group(0) @binding(3) var<storage, read> coordinate_spaces: array<mat4x4<f32>>;
@group(0) @binding(4) var<storage, read> chains: array<GpuPortalChain>;
@group(0) @binding(5) var<storage, read> stages: array<GpuPortalStage>;
// The object batch's GPU counts; word 0 is the live draw group count.
@group(0) @binding(6) var<storage, read> batch_counts: array<u32>;
@group(0) @binding(7) var<storage, read_write> group_counts: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> group_cursors: array<atomic<u32>>;
@group(0) @binding(9) var<storage, read_write> occurrence_instances: array<u32>;
@group(0) @binding(10) var<storage, read_write> occurrence_chains: array<u32>;
@group(0) @binding(11) var<storage, read_write> indirect: array<DrawIndexedIndirect>;
// [0]: occurrences found this frame (unclamped), read back to grow.
@group(0) @binding(12) var<storage, read_write> totals: array<u32>;

fn live_groups() -> u32 {
    return min(batch_counts[0], params.draw_capacity);
}

fn group_of(workgroup: vec3<u32>) -> u32 {
    return workgroup.x + workgroup.z * params.groups_per_row;
}

/// Whether `inst` may show through `chain`.
fn occurs(inst: GpuInstanceData, chain: GpuPortalChain) -> bool {
    // Another drawer draws it (virtual geometry, through its own chains run).
    if (inst.flags & INSTANCE_FLAG_CLASSIC_SUPPRESSED) != 0u {
        return false;
    }
    let space = coordinate_spaces[instance_coordinate_space(inst.flags)];
    let center = (space * vec4<f32>(inst.bounds.xyz, 1.0)).xyz;
    let radius = inst.bounds.w;
    for (var s = 0u; s < chain.stage_count; s += 1u) {
        let local = (stages[chain.first_stage + s].content_to_local * vec4<f32>(center, 1.0)).xyz;
        if local.z < -radius {
            return false;
        }
    }
    if (inst.flags & INSTANCE_FLAG_ALWAYS_VISIBLE) != 0u {
        return true;
    }
    let drawn = (chain.content_to_drawn * vec4<f32>(center, 1.0)).xyz;
    for (var i = 0u; i < 6u; i += 1u) {
        let plane = params.frustum_planes[i];
        if dot(plane.xyz, drawn) + plane.w + radius < 0.0 {
            return false;
        }
    }
    return true;
}

var<workgroup> found: atomic<u32>;

@compute @workgroup_size(64)
fn count_occurrences(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let group = group_of(workgroup);
    let chain_index = workgroup.y;
    if local.x == 0u {
        atomicStore(&found, 0u);
    }
    workgroupBarrier();
    let in_range = group < live_groups() && chain_index < params.chain_count;
    if in_range {
        let call = draw_calls[group];
        let chain = chains[chain_index];
        for (var i = local.x; i < call.instance_count; i += 64u) {
            if occurs(instances[call.first_instance + i], chain) {
                atomicAdd(&found, 1u);
            }
        }
    }
    workgroupBarrier();
    if in_range && local.x == 0u {
        let n = atomicLoad(&found);
        if n > 0u {
            atomicAdd(&group_counts[group], n);
        }
    }
}

const SCAN_THREADS: u32 = 256u;
var<workgroup> partial: array<u32, 256>;

@compute @workgroup_size(256)
fn scan_occurrences(@builtin(local_invocation_id) local: vec3<u32>) {
    let groups = params.draw_capacity;
    let live = live_groups();
    let per_thread = (groups + SCAN_THREADS - 1u) / SCAN_THREADS;
    let first = local.x * per_thread;
    let last = min(first + per_thread, groups);
    var sum = 0u;
    for (var g = first; g < last; g += 1u) {
        sum += atomicLoad(&group_counts[g]);
    }
    partial[local.x] = sum;
    workgroupBarrier();
    // Exclusive scan of the per-thread sums (Hillis-Steele, inclusive, then
    // shifted).
    for (var stride = 1u; stride < SCAN_THREADS; stride *= 2u) {
        var value = partial[local.x];
        if local.x >= stride {
            value += partial[local.x - stride];
        }
        workgroupBarrier();
        partial[local.x] = value;
        workgroupBarrier();
    }
    var offset = 0u;
    if local.x > 0u {
        offset = partial[local.x - 1u];
    }
    if local.x == SCAN_THREADS - 1u {
        totals[0] = partial[SCAN_THREADS - 1u];
    }
    for (var g = first; g < last; g += 1u) {
        let count = atomicLoad(&group_counts[g]);
        let room = select(0u, params.capacity - offset, offset < params.capacity);
        let kept = select(0u, min(count, room), g < live);
        let call = draw_calls[g];
        indirect[g] = DrawIndexedIndirect(call.index_count, kept, call.first_index, call.vertex_offset, offset);
        offset += count;
    }
}

@compute @workgroup_size(64)
fn write_occurrences(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let group = group_of(workgroup);
    let chain_index = workgroup.y;
    if group >= live_groups() || chain_index >= params.chain_count {
        return;
    }
    let call = draw_calls[group];
    let chain = chains[chain_index];
    let record = indirect[group];
    for (var i = local.x; i < call.instance_count; i += 64u) {
        let slot = call.first_instance + i;
        if occurs(instances[slot], chain) {
            let at = atomicAdd(&group_cursors[group], 1u);
            if at < record.instance_count {
                occurrence_instances[record.first_instance + at] = slot;
                occurrence_chains[record.first_instance + at] = chain_index;
            }
        }
    }
}
