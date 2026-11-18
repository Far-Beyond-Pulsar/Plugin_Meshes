//!use portal_chain
//!use mesh_object_row
//!use mesh_draw_rows
// Virtual geometry through portal chains: per live chain, the frame's VG
// instances it can show, copied where the chain draws them
// (`VgPortalInstancesPass`, the instances of `helio_vg_format::VgRun::SECONDARY`).
//
// One workgroup per chain walks the frame's instance BVH (`VgRun::DIRECT`'s),
// a level at a time, its threads splitting each level's nodes, against the
// chain's volume (`GpuPortalVolume`: the frustum mapped back into
// the content, the pyramid through the chain's deepest opening, its plane), so
// the work follows what the openings show, not every instance. Each instance
// whose bounding sphere passes is appended: a `VgInstance` whose object rows
// (one per material section) are copies of its own with the transforms placed
// where the chain draws them (`content_to_drawn · space · transform`, last
// frame's with last frame's map, for motion vectors), the coordinate space
// folded in (space 0), and the chain's remap.
//
// Prepended: `helio_vg_format::WGSL` (`VgInstance`, the instance flags) and
// `helio_pass_vg_bvh::NODES_WGSL` (`VgBvhNode`, `VgBvhHeader`).

struct Params {
    chain_count:       u32,
    instance_capacity: u32,
    object_capacity:   u32,
    _pad:              u32,
}

const STACK: u32 = 64u;

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> nodes: array<VgBvhNode>;
@group(0) @binding(2) var<storage, read> header: VgBvhHeader;
@group(0) @binding(3) var<storage, read> items: array<u32>;
@group(0) @binding(4) var<storage, read> instances: array<VgInstance>;
@group(0) @binding(5) var<storage, read> objects: array<StaticObjectRow>;
@group(0) @binding(6) var<storage, read> coordinate_spaces: array<mat4x4<f32>>;
@group(0) @binding(7) var<storage, read> coordinate_spaces_prev: array<mat4x4<f32>>;
@group(0) @binding(8) var<storage, read> chains: array<GpuPortalChain>;
@group(0) @binding(9) var<storage, read> volumes: array<GpuPortalVolume>;
@group(0) @binding(10) var<storage, read_write> out_instances: array<VgInstance>;
@group(0) @binding(11) var<storage, read_write> out_objects: array<StaticObjectRow>;
@group(0) @binding(12) var<storage, read_write> out_remaps: array<u32>;
// [0] instances, [1] object rows: reserved (past the capacities when they do
// not fit: read back to grow).
@group(0) @binding(13) var<storage, read_write> counts: array<atomic<u32>, 2>;

fn to_mat(m: array<array<f32, 4>, 4>) -> mat4x4<f32> {
    return mat4x4<f32>(
        vec4<f32>(m[0][0], m[0][1], m[0][2], m[0][3]),
        vec4<f32>(m[1][0], m[1][1], m[1][2], m[1][3]),
        vec4<f32>(m[2][0], m[2][1], m[2][2], m[2][3]),
        vec4<f32>(m[3][0], m[3][1], m[3][2], m[3][3]),
    );
}

fn from_mat(m: mat4x4<f32>) -> array<array<f32, 4>, 4> {
    return array<array<f32, 4>, 4>(
        array<f32, 4>(m[0].x, m[0].y, m[0].z, m[0].w),
        array<f32, 4>(m[1].x, m[1].y, m[1].z, m[1].w),
        array<f32, 4>(m[2].x, m[2].y, m[2].z, m[2].w),
        array<f32, 4>(m[3].x, m[3].y, m[3].z, m[3].w),
    );
}

fn space_of(flags: u32) -> u32 {
    return min(instance_coordinate_space(flags), arrayLength(&coordinate_spaces) - 1u);
}

/// Whether the box may hold content the chain shows (not wholly outside a plane).
fn volume_holds_box(volume: GpuPortalVolume, bmin: vec3<f32>, bmax: vec3<f32>) -> bool {
    for (var i = 0u; i < min(volume.plane_count, 10u); i += 1u) {
        let plane = volume.planes[i];
        let far = select(bmin, bmax, plane.xyz >= vec3<f32>(0.0));
        if dot(plane.xyz, far) + plane.w < 0.0 {
            return false;
        }
    }
    return true;
}

/// Appends instance row `row` as `chain` shows it.
fn emit(row: u32, chain: GpuPortalChain) {
    let inst = instances[row];
    let sections = max((inst.flags >> VG_INST_SECTION_SHIFT) & VG_INST_SECTION_MASK, 1u);
    let slot = atomicAdd(&counts[0], 1u);
    let first_object = atomicAdd(&counts[1], sections);
    if slot >= params.instance_capacity || first_object + sections > params.object_capacity
        || inst.object_row + sections > arrayLength(&objects)
        || first_object + sections > arrayLength(&out_objects)
    {
        return;
    }
    for (var s = 0u; s < sections; s++) {
        var object = objects[inst.object_row + s];
        let space = space_of(object.flags);
        let place = chain.content_to_drawn * coordinate_spaces[space];
        let place_prev = chain.prev_content_to_drawn
            * coordinate_spaces_prev[min(space, arrayLength(&coordinate_spaces_prev) - 1u)];
        object.transform = from_mat(place * to_mat(object.transform));
        object.prev_transform = from_mat(place_prev * to_mat(object.prev_transform));
        // The normal matrix turns with the placement (rigid).
        let turn = mat3x3<f32>(place[0].xyz, place[1].xyz, place[2].xyz);
        for (var c = 0u; c < 3u; c++) {
            let n = object.normal_mat[c];
            let turned = turn * vec3<f32>(n[0], n[1], n[2]);
            object.normal_mat[c] = array<f32, 4>(turned.x, turned.y, turned.z, n[3]);
        }
        let b = object.bounds;
        let center = (place * vec4<f32>(b[0], b[1], b[2], 1.0)).xyz;
        object.bounds = array<f32, 4>(center.x, center.y, center.z, b[3]);
        // The space is folded into the transforms.
        object.flags = object.flags & ~(0xFFu << INSTANCE_COORDINATE_SPACE_SHIFT);
        out_objects[first_object + s] = object;
    }
    out_instances[slot] = VgInstance(inst.asset_slot, inst.flags, inst.lod_error_scale, first_object);
    out_remaps[slot] = chain.remap;
}

/// Tests the instances of leaf `node` against `volume`, appending those that pass.
fn visit_leaf(node: VgBvhNode, volume: GpuPortalVolume, chain: GpuPortalChain) {
    let first = vg_bvh_leaf_first(node);
    for (var i = 0u; i < vg_bvh_leaf_count(node); i++) {
        if first + i >= arrayLength(&items) {
            break;
        }
        let row = items[first + i];
        if row >= arrayLength(&instances) {
            continue;
        }
        let inst = instances[row];
        if (inst.flags & VG_INST_ENABLED) == 0u || inst.object_row >= arrayLength(&objects) {
            continue;
        }
        let object = objects[inst.object_row];
        let slot = space_of(object.flags);
        if slot != chain.destination_space {
            continue;
        }
        let b = object.bounds;
        let space = coordinate_spaces[slot];
        let center = (space * vec4<f32>(b[0], b[1], b[2], 1.0)).xyz;
        if portal_volume_holds_sphere(volume, center, b[3]) {
            emit(row, chain);
        }
    }
}

/// The whole subtree under `root`, by one thread (what the shared frontier
/// cannot hold).
fn walk_subtree(root: u32, volume: GpuPortalVolume, chain: GpuPortalChain) {
    var stack: array<u32, STACK>;
    var top = 1u;
    stack[0] = root;
    loop {
        if top == 0u {
            break;
        }
        top -= 1u;
        let node_index = stack[top];
        if node_index >= arrayLength(&nodes) {
            continue;
        }
        let node = nodes[node_index];
        if !volume_holds_box(volume, node.bmin, node.bmax) {
            continue;
        }
        if vg_bvh_is_leaf(node) {
            visit_leaf(node, volume, chain);
        } else if top + 2u <= STACK {
            stack[top] = node.a;
            stack[top + 1u] = node.b;
            top += 2u;
        }
    }
}

// The tree level being walked and the next, breadth first: one workgroup per
// chain, its threads splitting each level's nodes.
const FRONTIER: u32 = 512u;
var<workgroup> frontier: array<u32, 1024>;
var<workgroup> next_count: atomic<u32>;
var<workgroup> level_count: u32;

@compute @workgroup_size(64)
fn cs_portal_instances(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let chain_index = group.x;
    if chain_index >= params.chain_count || chain_index >= arrayLength(&chains)
        || chain_index >= arrayLength(&volumes) || header.node_count == 0u
    {
        return;
    }
    let chain = chains[chain_index];
    let volume = volumes[chain_index];
    if lane == 0u {
        frontier[0] = header.root_node;
        level_count = 1u;
    }
    var current = 0u;
    loop {
        let count = workgroupUniformLoad(&level_count);
        if count == 0u {
            break;
        }
        if lane == 0u {
            atomicStore(&next_count, 0u);
        }
        workgroupBarrier();
        let next = (1u - current) * FRONTIER;
        for (var i = lane; i < count; i += 64u) {
            let node_index = frontier[current * FRONTIER + i];
            if node_index >= arrayLength(&nodes) {
                continue;
            }
            let node = nodes[node_index];
            if !volume_holds_box(volume, node.bmin, node.bmax) {
                continue;
            }
            if vg_bvh_is_leaf(node) {
                visit_leaf(node, volume, chain);
                continue;
            }
            // Children in pairs (FRONTIER is even): a pair that does not fit
            // is walked here.
            let at = atomicAdd(&next_count, 2u);
            if at < FRONTIER {
                frontier[next + at] = node.a;
                frontier[next + at + 1u] = node.b;
            } else {
                walk_subtree(node.a, volume, chain);
                walk_subtree(node.b, volume, chain);
            }
        }
        workgroupBarrier();
        if lane == 0u {
            level_count = min(atomicLoad(&next_count), FRONTIER);
        }
        current = 1u - current;
    }
}
