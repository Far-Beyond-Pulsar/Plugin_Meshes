//!use mesh_draw_rows

// The meshes' selected draws: of this frame's culled draws (the G-buffer's,
// `CulledBatchFrameData::indirect`), every instance whose object is selected
// (`helio_frame_api::selection::OBJECT_SELECTED`, joined through
// `helio_mesh_api::STATIC_OBJECT_OWNERS`) becomes one draw of its own, for
// the selection mask. One invocation per culled draw: the work follows what
// is visible, and what is drawn after follows what is selected.

struct Params {
    // The most culled draws there can be (`ObjectBatchFrameData::all_draws`).
    draw_capacity: u32,
    // Room in `selected_draws`.
    out_capacity: u32,
    // Non-zero: `draw_count[count_word]` is the GPU's live count of culled draws.
    gpu_count: u32,
    count_word: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> draws: array<DrawIndexedIndirect>;
@group(0) @binding(2) var<storage, read> draw_count: array<u32>;
@group(0) @binding(3) var<storage, read> compacted_indices: array<u32>;
@group(0) @binding(4) var<storage, read> instances: array<GpuInstanceData>;
@group(0) @binding(5) var<storage, read> object_owners: array<u32>;
@group(0) @binding(6) var<storage, read> object_selected: array<u32>;
@group(0) @binding(7) var<storage, read_write> selected_draws: array<DrawIndexedIndirect>;
@group(0) @binding(8) var<storage, read_write> selected_count: array<atomic<u32>>;

/// Whether the object behind instance slot `slot` (an index into
/// `compacted_indices`) is selected.
fn slot_selected(slot: u32) -> bool {
    if slot >= arrayLength(&compacted_indices) {
        return false;
    }
    let instance = compacted_indices[slot];
    if instance >= arrayLength(&instances) {
        return false;
    }
    let row = instances[instance].object_row;
    if row >= arrayLength(&object_owners) {
        return false;
    }
    let owner = object_owners[row];
    return owner != 0u && owner - 1u < arrayLength(&object_selected)
        && object_selected[owner - 1u] != 0u;
}

@compute @workgroup_size(64)
fn cs_select(@builtin(global_invocation_id) gid: vec3<u32>) {
    var count = params.draw_capacity;
    if params.gpu_count != 0u && params.count_word < arrayLength(&draw_count) {
        count = min(count, draw_count[params.count_word]);
    }
    let d = gid.x;
    if d >= count || d >= arrayLength(&draws) {
        return;
    }
    let draw = draws[d];
    for (var i = 0u; i < draw.instance_count; i = i + 1u) {
        let slot = draw.first_instance + i;
        if !slot_selected(slot) {
            continue;
        }
        let k = atomicAdd(&selected_count[0], 1u);
        if k < params.out_capacity {
            selected_draws[k] = DrawIndexedIndirect(
                draw.index_count,
                1u,
                draw.first_index,
                draw.base_vertex,
                slot,
            );
        }
    }
}
