//!use helio_prelude
// Virtual geometry cull: instance BVH -> per-asset cluster-group hierarchy ->
// cluster cut, with two-phase Hi-Z occlusion and tier binning.
//
// Composed in Rust (see `shader_source()`): helio prelude, generated `VGC_*`
// constants, vg_format.wgsl, helio-vg-stream bindings + helpers (binding 4..11),
// helio-pass-vg-bvh `NODES_WGSL`, then this
// file. Entry points: cs_tiles, cs_seed, cs_phase_init, cs_traverse, cs_finalize.
//
// Job = vec2<u32>(x, y), tag = y >> 30 (neither word is ever 0xffffffff: a
// queue slot stores the complement, and 0 marks it empty):
//   0  GCBVH node          x = 0,          y = node index
//   1  asset hierarchy item x = instance,   y = TAG | (parent_node << 2 | slot)
//   2  cluster item         x = instance,   y = TAG | cluster_slot (pool_slot << 11 | local)
//   3  instance             x = instance,   y = TAG
// Every job tests only its own bound. The cut is per cluster group: a group is
// refined when its projected error exceeds tau (a stable setting; no screen-wide
// multiplier), with hysteresis (`group_refines`). The only thing that raises
// the threshold is the per-tile overdraw valve (`tile_scale`), which acts on a
// tile that exceeds `tile_limit` triangles per covered pixel and nowhere else.
// The cut stays coverage complete: a coarser parent replaces geometry, never drops it.

struct CullParams {
    screen_width: u32,
    screen_height: u32,
    hiz_mip_count: u32,
    mode: u32,
    tau_px: f32,
    sw_edge_px: f32,
    instance_count: u32,
    bin_capacity: u32,
    queue_capacity: u32,
    history_bits_mask: u32,
    // Word offset in `state` of this frame's visibility/hysteresis marks (the
    // set being written; last frame's is `hist_ro[0 ..]`).
    hist_base: u32,
    // Word offset in `hist_ro` of the copy of the tile block.
    ro_tile_base: u32,
    flags: u32,
    dense_tiles_x: u32,
    dense_tiles_y: u32,
    max_groups: u32,
    // Overdraw valve: triangles per covered pixel one tile may hold before its
    // clusters are coarsened (0 = valve off).
    tile_limit: f32,
    // Prefetch band: child pages are requested/touched while the projected
    // error exceeds tau / prefetch_k (>= 1).
    prefetch_k: f32,
    // Tile grid of the screen (16x16 px tiles) and the word offset of its
    // per-tile block in `state` (3 words per tile: triangles, area, scale).
    tiles_x: u32,
    tiles_y: u32,
    tile_base: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct PhaseParams {
    phase: u32,
    hiz_valid: u32,
    pad0: u32,
    pad1: u32,
}

// GpuStaticObject row as published by the scene join (see
// helio-pass-object-batch `StaticObjectRow`); only `transform` is read.
struct VgObjectRow {
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
    skin_vertex_offset: u32,
    bone_palette_offset: u32,
    bone_palette_count: u32,
    _skin_pad: u32,
}

@group(0) @binding(0) var<storage, read> cameras: array<Camera, 2>;
@group(0) @binding(1) var<uniform> params: CullParams;
@group(0) @binding(2) var<storage, read> vg_instances: array<VgInstance>;
@group(0) @binding(3) var<storage, read> objects: array<VgObjectRow>;
// Bindings 4..=11 (page pool, residency, nodes, group refs, asset headers,
// bases, quant, feedback) are declared by `helio_vg_stream::bindings_wgsl(0, 4, true)`.
@group(0) @binding(12) var<storage, read> vg_gcbvh_nodes: array<VgBvhNode>;
@group(0) @binding(13) var<storage, read> vg_gcbvh_header: VgBvhHeader;
@group(0) @binding(14) var<storage, read> vg_gcbvh_items: array<u32>;
@group(0) @binding(15) var<storage, read> dense_mask: array<u32>;
@group(0) @binding(16) var hiz_tex: texture_2d<f32>;
@group(0) @binding(17) var hiz_samp: sampler;
// Counters, args and ranges, then this frame's visibility/hysteresis marks
// (`hist_base`) and the per-tile block (`tile_base`): every word the cull
// writes with atomics.
@group(0) @binding(18) var<storage, read_write> state: array<atomic<u32>>;
// Two job queues of `queue_capacity` slots (2 words each): 0 is the early
// phase's, 1 takes the early phase's deferred work and is the late phase's.
// A slot holds the complemented job; 0 = not written yet (`queue_put`,
// `queue_take`). Every slot a phase reserves is written and taken (and zeroed)
// before the phase ends, so the buffer is all zero between phases.
@group(0) @binding(19) var<storage, read_write> queues: array<atomic<u32>>;
@group(0) @binding(20) var<storage, read_write> visible: array<vec2<u32>>;
// What the traversal only reads, snapshotted from `state` before it starts
// (`VgCullCore::record_phase`): last frame's marks at 0, the tile block (with
// this frame's valve scales) at `ro_tile_base`. Plain loads, so they are served
// from the L1 instead of an L2 atomic per job.
@group(0) @binding(21) var<storage, read> hist_ro: array<u32>;
// GpuMaterial rows (24 words each); read only when VGC_FLAG_MASKED is set.
@group(0) @binding(22) var<storage, read> vgc_materials: array<u32>;

@group(1) @binding(0) var<uniform> rp: PhaseParams;

const WG: u32 = 64u;
const STACK_CAP: u32 = 1024u;
const STACK_MASK: u32 = 1023u;
const NO_JOB: vec2<u32> = vec2<u32>(0xffffffffu, 0xffffffffu);
// Bounds on spinning, so a broken invariant shows up as a counted overflow
// instead of a device loss (never reached when the invariants hold): a taken
// slot is written by a running workgroup right after it reserved it, and an
// idle group only waits while some running group holds work.
const SLOT_SPIN_MAX: u32 = 1048576u;
const IDLE_SPIN_MAX: u32 = 1048576u;
const INF_ERR: f32 = 1.0e30;
const HUGE_PX: f32 = 3.0e38;
const ROOT_ONLY_TAU: f32 = 1.0e30;

// The workgroup's local deque of jobs (logical indices, `& STACK_MASK`):
// pushed and popped at `wg_top`, shared from the bottom (`wg_bot`, the oldest
// and coarsest jobs, the biggest subtrees). `wg_bot` only moves between
// barriers, by thread 0.
var<workgroup> wg_stack: array<vec2<u32>, 1024>;
var<workgroup> wg_top: atomic<u32>;
var<workgroup> wg_bot: u32;

// One wave's plan, decided by thread 0 (`wave_plan`) and read by every thread.
//   mode 0: exit; 1: pop `n` jobs from the deque at `base`; 2: take `n` jobs
//   from the phase queue at slot `base`; 3: nothing to do yet (wait).
//   `share_n` jobs of the deque bottom (from `share_src`) go to queue slots
//   `share_dst ..` in the same wave.
struct Wave {
    mode: u32,
    n: u32,
    base: u32,
    share_n: u32,
    share_src: u32,
    share_dst: u32,
}
var<workgroup> wg_wave: Wave;
var<workgroup> wg_planes: array<vec4<f32>, 5>;

// Emission staging (`emit_cluster`): the clusters a wave emits are collected
// here and their bin slots reserved with one global atomic per bin per wave
// (`emit_reserve` / `emit_commit` in `cs_traverse`), instead of one per
// cluster. Entry: {instance, cluster slot, bin | rank << 3, triangles} and the
// world sphere (for the tile estimate). An emission past `EMIT_CAP` in one wave
// reserves its slot directly.
const EMIT_CAP: u32 = 128u;
var<workgroup> wg_emit: array<vec4<u32>, 128>;
var<workgroup> wg_emit_sph: array<vec4<f32>, 128>;
var<workgroup> wg_emit_n: atomic<u32>;
var<workgroup> wg_bin_n: array<atomic<u32>, VGC_BIN_COUNT>;
var<workgroup> wg_bin_base: array<u32, VGC_BIN_COUNT>;

// Tile-estimate aggregation (`tile_add`): a small open-addressed table of
// tiles touched this wave, summed in workgroup memory and added to `state`
// once per tile per wave (`tile_flush`). Integer sums of the same rounded
// contributions, so the estimate is bit-identical to per-cluster atomics.
const TILE_SLOTS: u32 = 64u;
const TILE_PROBES: u32 = 4u;
const TILE_EMPTY: u32 = 0xffffffffu;
var<workgroup> wg_tile_key: array<atomic<u32>, 64>;
var<workgroup> wg_tile_tris: array<atomic<u32>, 64>;
var<workgroup> wg_tile_area: array<atomic<u32>, 64>;

// ── Small helpers ──────────────────────────────────────────────────────────

// Counters are kept per thread and added to `state` once, when the thread
// leaves the traversal (`stats_flush`): a shared atomic per job put every
// thread of the dispatch on the same few cache lines.
var<private> p_stats: array<u32, VGC_STAT_COUNT>;

fn stat_add(i: u32) {
    if (params.flags & VGC_FLAG_STATS) != 0u {
        p_stats[i] += 1u;
    }
}

fn stats_flush() {
    for (var i = 0u; i < VGC_STAT_COUNT; i++) {
        let n = p_stats[i];
        if n != 0u { atomicAdd(&state[VGC_ST_STATS + i], n); }
    }
}

fn vg_hash(a: u32, b: u32) -> u32 {
    var h = (a * 0x9e3779b1u) ^ ((b + 0x7f4a7c15u) * 0x85ebca6bu);
    h ^= h >> 15u;
    h *= 0x2c1b3c6du;
    h ^= h >> 12u;
    h *= 0x297a2d39u;
    h ^= h >> 15u;
    return h;
}

fn job_key(job: vec2<u32>) -> u32 { return vg_hash(job.x, job.y); }

fn history_mark(key: u32) {
    if (params.flags & VGC_FLAG_TWO_PHASE) == 0u { return; }
    hyst_mark(key);
}

// Hysteresis memory: the same hash set, salted per decision kind and not gated
// by the two-phase flag. A bit says "this decision was taken last frame"; both
// the test and the mark are idempotent, so every job that reaches the same
// decision (the two sides of one group) agrees.
const SALT_REFINE: u32 = 0x1b873593u;
const SALT_SW: u32 = 0x51ed270bu;

// Last frame's marks: the read-only snapshot, never written during the frame.
fn hyst_test(key: u32) -> bool {
    let bit = key & params.history_bits_mask;
    let w = hist_ro[bit >> 5u];
    return ((w >> (bit & 31u)) & 1u) != 0u;
}

fn hyst_mark(key: u32) {
    let bit = key & params.history_bits_mask;
    let word = params.hist_base + (bit >> 5u);
    let mask = 1u << (bit & 31u);
    // These marks only accumulate during traversal and are cleared before the
    // next frame. Repeated group decisions need no further atomic writes.
    if (atomicLoad(&state[word]) & mask) == 0u {
        atomicOr(&state[word], mask);
    }
}

fn vp_row(r: u32) -> vec4<f32> {
    let m = cameras[0].view_proj;
    return vec4<f32>(m[0][r], m[1][r], m[2][r], m[3][r]);
}

// Normalised frustum planes, inside when dot(n, p) + d >= 0. No far plane
// (infinite reversed-Z projection); mirrors helio_core::depth::frustum_planes.
fn frustum_plane(i: u32) -> vec4<f32> {
    let r3 = vp_row(3u);
    var p: vec4<f32>;
    switch i {
        case 0u: { p = r3 + vp_row(0u); }
        case 1u: { p = r3 - vp_row(0u); }
        case 2u: { p = r3 + vp_row(1u); }
        case 3u: { p = r3 - vp_row(1u); }
        default: { p = r3 - vp_row(2u); }
    }
    return p / length(p.xyz);
}

fn sphere_in_frustum(c: vec3<f32>, r: f32) -> bool {
    for (var i = 0u; i < 5u; i++) {
        let p = wg_planes[i];
        if dot(p.xyz, c) + p.w < -r { return false; }
    }
    return true;
}

fn aabb_in_frustum(mn: vec3<f32>, mx: vec3<f32>) -> bool {
    for (var i = 0u; i < 5u; i++) {
        let p = wg_planes[i];
        let v = select(mn, mx, p.xyz > vec3<f32>(0.0));
        if dot(p.xyz, v) + p.w < 0.0 { return false; }
    }
    return true;
}

fn view_depth_of(c: vec3<f32>) -> f32 {
    return -(cameras[0].view * vec4<f32>(c, 1.0)).z;
}

// Projected size in pixels of an object-space error `err` (already scaled by
// instance scale) at a bounding sphere. Uses the sphere's nearest view depth,
// clamped to the near plane, so a containing sphere never projects smaller
// than a contained one (monotone => the cut is hole-free). Mirrors
// `reference::project_error`.
fn proj_err(err: f32, c: vec3<f32>, r: f32) -> f32 {
    if err >= INF_ERR { return HUGE_PX; }
    let d = max(view_depth_of(c) - r, cameras[0].position_near.w);
    return err * cameras[0].proj[1][1] * 0.5 * f32(params.screen_height) / d;
}

struct InstCtx {
    m: mat4x4<f32>,
    scale: f32,
    err_scale: f32,
    tau: f32,
    flags: u32,
    asset: u32,
}

fn load_inst(id: u32) -> InstCtx {
    let inst = vg_instances[id];
    let t = objects[inst.object_row].transform;
    let m = mat4x4<f32>(
        vec4<f32>(t[0][0], t[0][1], t[0][2], t[0][3]),
        vec4<f32>(t[1][0], t[1][1], t[1][2], t[1][3]),
        vec4<f32>(t[2][0], t[2][1], t[2][2], t[2][3]),
        vec4<f32>(t[3][0], t[3][1], t[3][2], t[3][3]),
    );
    var c: InstCtx;
    c.m = m;
    c.scale = max(length(m[0].xyz), max(length(m[1].xyz), length(m[2].xyz)));
    c.err_scale = max(inst.lod_error_scale, 1.0e-4);
    c.flags = inst.flags;
    c.asset = inst.asset_slot;
    c.tau = select(params.tau_px, ROOT_ONLY_TAU, (inst.flags & VG_INST_FORCE_ROOT_ONLY) != 0u);
    return c;
}

// ── Hi-Z (mirrors helio-pass-occlusion-cull, parameterised on the matrix) ─

fn sphere_near_depth(center: vec3<f32>, radius: f32, vp: mat4x4<f32>) -> f32 {
    let to_center = center - cameras[0].position_near.xyz;
    let dist_sq = dot(to_center, to_center);
    if dist_sq <= radius * radius { return 1.0; }
    let dir = to_center * (1.0 / sqrt(dist_sq));
    let near_clip = vp * vec4<f32>(center - dir * radius, 1.0);
    if near_clip.w <= 0.0 { return 1.0; }
    return clamp(near_clip.z / near_clip.w, 0.0, 1.0);
}

fn hiz_sphere_occluded(center: vec3<f32>, radius: f32, vp: mat4x4<f32>) -> bool {
    if radius <= 0.0 { return false; }
    let clip = vp * vec4<f32>(center, 1.0);
    if clip.w <= 0.0 { return false; }
    let ndc_r = max(
        abs(radius * cameras[0].proj[0][0] / clip.w),
        abs(radius * cameras[0].proj[1][1] / clip.w),
    );
    let ndc = clip.xyz / clip.w;
    if ndc.x + ndc_r < -1.0 || ndc.x - ndc_r > 1.0 || ndc.y + ndc_r < -1.0 || ndc.y - ndc_r > 1.0 {
        return false;
    }
    let near_z = sphere_near_depth(center, radius, vp);
    let r_px = abs(radius / clip.w * cameras[0].proj[1][1] * f32(params.screen_height) * 0.5);
    let mip = clamp(u32(ceil(log2(max(r_px * 2.0, 1.0)))), 0u, params.hiz_mip_count - 1u);
    let uv = helio_ndc_to_uv(ndc.xy);
    let uv_half = ndc_r * 0.5;
    let uv_00 = clamp(uv - vec2<f32>(uv_half), vec2<f32>(0.0), vec2<f32>(1.0));
    let uv_11 = clamp(uv + vec2<f32>(uv_half), vec2<f32>(0.0), vec2<f32>(1.0));
    let h00 = textureSampleLevel(hiz_tex, hiz_samp, uv_00, f32(mip)).r;
    let h01 = textureSampleLevel(hiz_tex, hiz_samp, vec2<f32>(uv_11.x, uv_00.y), f32(mip)).r;
    let h10 = textureSampleLevel(hiz_tex, hiz_samp, vec2<f32>(uv_00.x, uv_11.y), f32(mip)).r;
    let h11 = textureSampleLevel(hiz_tex, hiz_samp, uv_11, f32(mip)).r;
    let farthest = helio_depth_farthest(helio_depth_farthest(h00, h01), helio_depth_farthest(h10, h11));
    return helio_hiz_occludes(farthest, near_z);
}

const POLICY_PROCESS: u32 = 0u;
const POLICY_DEFER: u32 = 1u;
const POLICY_DROP: u32 = 2u;

// Two-phase occlusion policy for one job bound.
//   phase 0: only work that was visible last frame is tried, against last
//            frame's Hi-Z (old view-projection); everything else is deferred.
//            Wrong answers here are safe both ways: a false "visible" is just
//            an extra draw, a false "occluded" is retested in phase 1.
//   phase 1: retest deferred work against the Hi-Z built from phase-0 depth.
fn occlusion_policy(job: vec2<u32>, c: vec3<f32>, r: f32) -> u32 {
    if (params.flags & VGC_FLAG_TWO_PHASE) == 0u { return POLICY_PROCESS; }
    if rp.phase == 0u {
        if !hyst_test(job_key(job)) { return POLICY_DEFER; }
        if rp.hiz_valid != 0u && hiz_sphere_occluded(c, r * 1.02, cameras[0].prev_view_proj) {
            return POLICY_DEFER;
        }
        return POLICY_PROCESS;
    }
    if rp.hiz_valid != 0u && hiz_sphere_occluded(c, r, cameras[0].view_proj) {
        return POLICY_DROP;
    }
    return POLICY_PROCESS;
}

// ── Queues ──────────────────────────────────────────────────────────────────
//
// One persistent dispatch per phase traverses the whole tree (`cs_traverse`).
// A workgroup works depth-first from its local deque; the phase queue (an
// unbounded-producer, multi-consumer array: append-only within the phase) is
// how work moves between workgroups. `state[VGC_ST_OUTSTANDING]` counts the
// queued jobs that no workgroup has taken plus the workgroups that hold work;
// it is raised before work becomes visible and lowered after, so it reaches 0
// exactly when the phase is done, and every workgroup then exits.

fn queue_word(q: u32, slot: u32) -> u32 {
    return 2u * (q * params.queue_capacity + slot);
}

fn queue_put(q: u32, slot: u32, job: vec2<u32>) {
    let w = queue_word(q, slot);
    atomicStore(&queues[w], ~job.x);
    atomicStore(&queues[w + 1u], ~job.y);
}

// The job in a slot this workgroup has taken. Its producer reserved the slot
// and writes it without waiting on anything, so the spin is short; both words
// are checked, so a half-written slot is never read.
fn queue_take(q: u32, slot: u32) -> vec2<u32> {
    let w = queue_word(q, slot);
    for (var spin = 0u; spin < SLOT_SPIN_MAX; spin++) {
        let x = atomicLoad(&queues[w]);
        let y = atomicLoad(&queues[w + 1u]);
        if x != 0u && y != 0u {
            atomicStore(&queues[w], 0u);
            atomicStore(&queues[w + 1u], 0u);
            return vec2<u32>(~x, ~y);
        }
    }
    stat_add(VGC_STAT_JOB_OVERFLOW);
    return NO_JOB;
}

fn queue_tail(q: u32) -> u32 {
    return min(atomicLoad(&state[VGC_ST_Q_TAIL + q]), params.queue_capacity);
}

// Queued jobs not yet taken.
fn queue_backlog(q: u32) -> u32 {
    let t = queue_tail(q);
    let h = atomicLoad(&state[VGC_ST_Q_HEAD + q]);
    return select(0u, t - h, t > h);
}

// Reserves `n` consecutive slots for a share, or returns 0xffffffff when the
// queue cannot take them all (the jobs then stay in the deque: nothing is
// lost). `outstanding` is raised first, so it never undercounts.
fn queue_reserve(q: u32, n: u32) -> u32 {
    atomicAdd(&state[VGC_ST_OUTSTANDING], n);
    var t = atomicLoad(&state[VGC_ST_Q_TAIL + q]);
    loop {
        if t + n > params.queue_capacity {
            atomicSub(&state[VGC_ST_OUTSTANDING], n);
            return 0xffffffffu;
        }
        let r = atomicCompareExchangeWeak(&state[VGC_ST_Q_TAIL + q], t, t + n);
        if r.exchanged { break; }
        t = r.old_value;
    }
    atomicMax(&state[VGC_ST_STATS + VGC_STAT_PEAK_QUEUE], t + n);
    if (params.flags & VGC_FLAG_STATS) != 0u { p_stats[VGC_STAT_SPILLED] += n; }
    return t;
}

// Takes up to a guided-self-scheduling share of the backlog (backlog / groups,
// 1..WG jobs): large chunks while there is much, single jobs (the biggest
// subtrees, shared from deque bottoms) when work is scarce. Returns
// (first slot, count); count 0 when nothing is queued.
fn queue_claim(q: u32) -> vec2<u32> {
    var h = atomicLoad(&state[VGC_ST_Q_HEAD + q]);
    loop {
        let t = queue_tail(q);
        if h >= t { return vec2<u32>(0u, 0u); }
        let c = clamp((t - h) / params.max_groups, 1u, WG);
        let r = atomicCompareExchangeWeak(&state[VGC_ST_Q_HEAD + q], h, h + c);
        if r.exchanged { return vec2<u32>(h, c); }
        h = r.old_value;
    }
    return vec2<u32>(0u, 0u);
}

// A job whose deque is full goes to the phase queue. Lost only when that is
// full as well (the live frontier exceeds every queue and deque; counted, and
// the queue grows for the next frame).
fn share_job(job: vec2<u32>) {
    let q = rp.phase;
    atomicAdd(&state[VGC_ST_OUTSTANDING], 1u);
    let t = atomicAdd(&state[VGC_ST_Q_TAIL + q], 1u);
    if t < params.queue_capacity {
        queue_put(q, t, job);
        stat_add(VGC_STAT_SPILLED);
        atomicMax(&state[VGC_ST_STATS + VGC_STAT_PEAK_QUEUE], t + 1u);
    } else {
        atomicSub(&state[VGC_ST_OUTSTANDING], 1u);
        stat_add(VGC_STAT_JOB_OVERFLOW);
    }
}

fn push_job(job: vec2<u32>) {
    let i = atomicAdd(&wg_top, 1u);
    if i - wg_bot < STACK_CAP {
        wg_stack[i & STACK_MASK] = job;
    } else {
        share_job(job);
    }
}

// Early phase: queues work for the late phase. False when the late queue is
// full: the caller then processes the job now (drawn without the occlusion
// retest, never lost). The tail may pass the capacity; `cs_phase_init`
// clamps it and records it as the peak, so the queue grows.
fn defer_job(job: vec2<u32>) -> bool {
    let t = atomicAdd(&state[VGC_ST_Q_TAIL + 1u], 1u);
    if t >= params.queue_capacity { return false; }
    queue_put(1u, t, job);
    stat_add(VGC_STAT_DEFERRED);
    return true;
}

// ── Per-tile density: the overdraw valve and the dense-tile estimate ───────
//
// `state` carries, after the marks, three words per 16x16 tile: the triangles
// and the covered area the clusters emitted last frame estimate into the tile
// (fixed point), and the valve's coarsening scale `g >= 1` (f32 bits; 0 reads
// as 1). `cs_tiles` turns the estimate into the scale and clears it; the
// traversal reads the scales from the `hist_ro` copy taken after `cs_tiles`;
// `emit_cluster` adds to the estimate; the dense-tile classifier reads the
// triangle word (so hardware-class clusters count towards "this tile is dense"
// without the software raster having run).

const TILE_EST_MAX_R: f32 = 32.0;

struct Footprint {
    ok: bool,
    px: vec2<f32>,
    r: f32,
    lo: vec2<i32>,
    hi: vec2<i32>,
}

// The 16x16 tiles a projected sphere covers. `ok` is false for spheres behind
// the camera, off screen, or wider than `TILE_EST_MAX_R` px (a cluster that
// large cannot make a tile dense: at most 128 triangles over 4+ tiles).
fn tile_footprint(cw: vec3<f32>, rw: f32) -> Footprint {
    var f: Footprint;
    f.ok = false;
    if params.tiles_x == 0u || params.tiles_y == 0u { return f; }
    let clip = cameras[0].view_proj * vec4<f32>(cw, 1.0);
    if clip.w <= 0.0 { return f; }
    let d = max(view_depth_of(cw), cameras[0].position_near.w);
    let r = rw * cameras[0].proj[1][1] * 0.5 * f32(params.screen_height) / d;
    if r > TILE_EST_MAX_R { return f; }
    let uv = helio_ndc_to_uv(clip.xy / clip.w);
    let px = uv * vec2<f32>(f32(params.screen_width), f32(params.screen_height));
    let re = max(r, 0.5);
    if px.x + re <= 0.0 || px.y + re <= 0.0
        || px.x - re >= f32(params.screen_width) || px.y - re >= f32(params.screen_height) { return f; }
    let tmax = vec2<i32>(i32(params.tiles_x) - 1, i32(params.tiles_y) - 1);
    f.px = px;
    f.r = re;
    f.lo = clamp(vec2<i32>(floor((px - vec2<f32>(re)) / f32(VGC_TILE))), vec2<i32>(0), tmax);
    f.hi = clamp(vec2<i32>(floor((px + vec2<f32>(re)) / f32(VGC_TILE))), vec2<i32>(0), tmax);
    f.ok = true;
    return f;
}

fn tile_word(t: u32, w: u32) -> u32 { return params.tile_base + t * VGC_TILE_WORDS + w; }

// This frame's valve scale of tile `t`, from the read-only copy.
fn tile_scale_of(t: u32) -> f32 {
    return bitcast<f32>(hist_ro[params.ro_tile_base + t * VGC_TILE_WORDS + 2u]);
}

// The valve's coarsening scale for a sphere: the largest scale of the tiles it
// covers (1 when none is over the limit, or the valve is off).
fn tile_scale(cw: vec3<f32>, rw: f32) -> f32 {
    if params.tile_limit <= 0.0 { return 1.0; }
    let f = tile_footprint(cw, rw);
    if !f.ok { return 1.0; }
    var g = 1.0;
    for (var y = f.lo.y; y <= f.hi.y; y++) {
        for (var x = f.lo.x; x <= f.hi.x; x++) {
            let t = u32(y) * params.tiles_x + u32(x);
            g = max(g, tile_scale_of(t));
        }
    }
    return g;
}

// The smallest scale of the tiles a sphere covers. A hierarchy node may stop
// the descent at this scale and stay coverage complete: every cluster below it
// lies inside the node's sphere, so its own scale (the largest over its tiles,
// a subset of the node's) is never smaller. Where the node's tiles are all
// saturated (a distant tree: a few pixels, one tile) the whole subtree is
// skipped; a node straddling an unsaturated tile still descends.
fn tile_scale_min(cw: vec3<f32>, rw: f32) -> f32 {
    if params.tile_limit <= 0.0 { return 1.0; }
    let f = tile_footprint(cw, rw);
    if !f.ok { return 1.0; }
    var g = VGC_TILE_SCALE_MAX;
    for (var y = f.lo.y; y <= f.hi.y; y++) {
        for (var x = f.lo.x; x <= f.hi.x; x++) {
            let t = u32(y) * params.tiles_x + u32(x);
            g = min(g, tile_scale_of(t));
        }
    }
    return max(g, 1.0);
}

// Adds an emitted cluster's triangles to the tiles it covers, weighted by the
// overlap of each tile with the sphere's bounding square, and the disc area.
fn tile_accumulate(cw: vec3<f32>, rw: f32, tris: u32) {
    let f = tile_footprint(cw, rw);
    if !f.ok { return; }
    let sq = 4.0 * f.r * f.r;
    let disc = 3.14159265 * f.r * f.r;
    let tile = f32(VGC_TILE);
    for (var y = f.lo.y; y <= f.hi.y; y++) {
        let ty = f32(y) * tile;
        let oy = max(min(f.px.y + f.r, ty + tile) - max(f.px.y - f.r, ty), 0.0);
        for (var x = f.lo.x; x <= f.hi.x; x++) {
            let tx = f32(x) * tile;
            let ox = max(min(f.px.x + f.r, tx + tile) - max(f.px.x - f.r, tx), 0.0);
            let share = ox * oy / sq;
            let t = u32(y) * params.tiles_x + u32(x);
            tile_add(t, u32(f32(tris) * share * VGC_TILE_FIXED + 0.5), u32(disc * share * VGC_TILE_FIXED + 0.5));
        }
    }
}

// Adds to tile `t`'s estimate through the wave's tile table; straight to
// `state` when its probe window is taken by other tiles. A spuriously failed
// compare-exchange only moves on to the next slot: any split of a tile's sum
// between slots and `state` adds up to the same total.
fn tile_add(t: u32, tris_fx: u32, area_fx: u32) {
    if (tris_fx | area_fx) == 0u { return; }
    let h = (t * 0x9e3779b1u) >> 26u;
    for (var p = 0u; p < TILE_PROBES; p++) {
        let s = (h + p) & (TILE_SLOTS - 1u);
        let r = atomicCompareExchangeWeak(&wg_tile_key[s], TILE_EMPTY, t);
        if r.exchanged || r.old_value == t {
            atomicAdd(&wg_tile_tris[s], tris_fx);
            atomicAdd(&wg_tile_area[s], area_fx);
            return;
        }
    }
    atomicAdd(&state[tile_word(t, 0u)], tris_fx);
    atomicAdd(&state[tile_word(t, 1u)], area_fx);
}

// One slot of the tile table to `state`, emptied for the next wave. Called by
// thread `s` after the wave's last `tile_add` (behind a barrier).
fn tile_flush(s: u32) {
    let t = atomicLoad(&wg_tile_key[s]);
    if t == TILE_EMPTY { return; }
    let a = atomicExchange(&wg_tile_tris[s], 0u);
    let b = atomicExchange(&wg_tile_area[s], 0u);
    if a != 0u { atomicAdd(&state[tile_word(t, 0u)], a); }
    if b != 0u { atomicAdd(&state[tile_word(t, 1u)], b); }
    atomicStore(&wg_tile_key[s], TILE_EMPTY);
}

// ── Group refinement (the cut's one decision) ──────────────────────────────

// Whether the group with object-space sphere `sph` and error `err` (projected
// `px`) is refined, i.e. its children replace its clusters. Both sides of the
// decision (the group's own clusters, and the finer clusters it was made from)
// evaluate this with identical inputs, so the cut has neither holes nor
// overlaps. Hysteresis: a group refined last frame stays refined until `px`
// drops below `tau * VGC_TAU_HOLD`, so `px <= tau` still holds for every drawn
// group and only the finer side is ever held.
fn group_refines(inst: u32, sph: vec4<f32>, err: f32, px: f32, tau: f32) -> bool {
    if err >= INF_ERR { return true; }
    if !(err > 0.0) { return false; }
    // Below the hold threshold both histories choose coarse. No hash lookup
    // or mark is necessary; only decisions inside the band depend on history.
    if !(px > tau * VGC_TAU_HOLD) { return false; }
    var h = vg_hash(inst, bitcast<u32>(err));
    h = vg_hash(h, bitcast<u32>(sph.x));
    h = vg_hash(h, bitcast<u32>(sph.y));
    h = vg_hash(h, bitcast<u32>(sph.z));
    let key = h ^ SALT_REFINE;
    let refine = px > tau || hyst_test(key);
    if refine { hyst_mark(key); }
    return refine;
}

// ── Binning ────────────────────────────────────────────────────────────────

fn tri_class(tris: u32) -> u32 {
    if tris <= 32u { return 0u; }
    if tris <= 64u { return 1u; }
    if tris <= 96u { return 2u; }
    return 3u;
}

fn dense_overlap(cw: vec3<f32>, r_px: f32) -> bool {
    let clip = cameras[0].view_proj * vec4<f32>(cw, 1.0);
    if clip.w <= 0.0 { return false; }
    let uv = helio_ndc_to_uv(clip.xy / clip.w);
    let px = uv * vec2<f32>(f32(params.screen_width), f32(params.screen_height));
    let tmax = vec2<i32>(i32(params.dense_tiles_x) - 1, i32(params.dense_tiles_y) - 1);
    let lo = clamp(vec2<i32>(floor((px - vec2<f32>(r_px)) / f32(VGC_TILE))), vec2<i32>(0), tmax);
    let hi = min(clamp(vec2<i32>(floor((px + vec2<f32>(r_px)) / f32(VGC_TILE))), vec2<i32>(0), tmax), lo + vec2<i32>(3));
    for (var y = lo.y; y <= hi.y; y++) {
        for (var x = lo.x; x <= hi.x; x++) {
            let idx = u32(y) * params.dense_tiles_x + u32(x);
            if idx < arrayLength(&dense_mask) && dense_mask[idx] != 0u { return true; }
        }
    }
    return false;
}

// A cluster wholly covered by dense tiles can leave the raster bins entirely:
// every pixel it could touch will be supplied by Tier 3. Clusters crossing a
// dense-tile boundary stay rasterized for their pixels outside the mask.
fn dense_fully_covered(cw: vec3<f32>, r_px: f32) -> bool {
    let clip = cameras[0].view_proj * vec4<f32>(cw, 1.0);
    if clip.w <= 0.0 { return false; }
    let uv = helio_ndc_to_uv(clip.xy / clip.w);
    let px = uv * vec2<f32>(f32(params.screen_width), f32(params.screen_height));
    let tmax = vec2<i32>(i32(params.dense_tiles_x) - 1, i32(params.dense_tiles_y) - 1);
    let lo = vec2<i32>(floor((px - vec2<f32>(r_px)) / f32(VGC_TILE)));
    let hi = vec2<i32>(floor((px + vec2<f32>(r_px)) / f32(VGC_TILE)));
    // The screen edge is not a dense tile. Keep partially off-screen clusters
    // on raster so clipping at the viewport cannot create missing pixels.
    if any(lo < vec2<i32>(0)) || any(hi > tmax) { return false; }
    for (var y = lo.y; y <= hi.y; y++) {
        for (var x = lo.x; x <= hi.x; x++) {
            let idx = u32(y) * params.dense_tiles_x + u32(x);
            if idx >= arrayLength(&dense_mask) || dense_mask[idx] == 0u { return false; }
        }
    }
    return true;
}

fn choose_bin(tris: u32, ic: InstCtx, cw: vec3<f32>, rw: f32, key: u32) -> u32 {
    let hw_bin = tri_class(tris);
    if params.mode == VGC_MODE_FORCE_HW { return hw_bin; }
    let d = max(view_depth_of(cw), cameras[0].position_near.w);
    let r_px = rw * cameras[0].proj[1][1] * 0.5 * f32(params.screen_height) / d;
    // Use the sphere's nearest depth for the raster-removal proof. The cheaper
    // center-depth radius below still drives normal tier selection.
    let safe_d = max(d - rw, cameras[0].position_near.w);
    let safe_r_px = rw * cameras[0].proj[1][1] * 0.5 * f32(params.screen_height) / safe_d;
    if (params.flags & VGC_FLAG_DENSE_MASK) != 0u && d > rw + cameras[0].position_near.w && dense_fully_covered(cw, safe_r_px) {
        return VGC_BIN_DENSE;
    }
    let lock = (ic.flags >> VG_INST_TIER_SHIFT) & VG_INST_TIER_MASK;
    if lock == 1u { return hw_bin; }
    if lock == 2u { return VGC_BIN_SW; }
    if lock == 3u { return VGC_BIN_DENSE; }
    // Hysteresis on the SW/HW edge test: a cluster becomes software at `sw_edge_px`
    // (the stateless rule) and one that was software last frame stays so until its
    // edge exceeds the threshold by `VGC_EDGE_HYST`.
    let edge = VGC_EDGE_FACTOR * r_px / sqrt(f32(max(tris, 1u)));
    let sw_key = vg_hash(key, SALT_SW);
    let was_sw = hyst_test(sw_key);
    if edge <= params.sw_edge_px * select(1.0, 1.0 + VGC_EDGE_HYST, was_sw) {
        hyst_mark(sw_key);
        if (params.flags & VGC_FLAG_DENSE_MASK) != 0u && dense_overlap(cw, r_px) {
            return VGC_BIN_DENSE;
        }
        return VGC_BIN_SW;
    }
    return hw_bin;
}

// `GpuMaterial::flags` of section `section` of instance `inst` (0 when out of
// range), read from the materials' table as words (`//!use material_row`).
fn section_material_flags(inst: u32, section: u32) -> u32 {
    let row = vg_instances[inst].object_row + section;
    if row >= arrayLength(&objects) { return 0u; }
    let mslot = objects[row].material_slot;
    let w = mslot * MATERIAL_WORDS + MATERIAL_FLAGS_WORD;
    if w >= arrayLength(&vgc_materials) { return 0u; }
    return vgc_materials[w];
}

// Alpha-tested clusters leave the opaque bins: they are raster-ed by the masked
// pipelines, which sample the alpha texture. The opaque bins (and their
// pipelines) are untouched.
fn masked_bin(bin: u32) -> u32 {
    if bin == VGC_BIN_SW || bin == VGC_BIN_DENSE {
        if (params.flags & VGC_FLAG_SW_MASKED) != 0u { return VGC_BIN_SW_MASKED; }
    }
    return VGC_BIN_HW_MASKED;
}

// Stages an emitted cluster for the wave's bin reservation (`emit_reserve`,
// `emit_commit`). The stage slot is taken before the bin rank, so a cluster
// that finds the stage full holds no rank and every reserved slot is written.
fn emit_cluster(inst: u32, slot: u32, bin: u32, triangles: u32, cw: vec3<f32>, rw: f32) {
    let e = atomicAdd(&wg_emit_n, 1u);
    if e < EMIT_CAP {
        let rank = atomicAdd(&wg_bin_n[bin], 1u);
        wg_emit[e] = vec4<u32>(inst, slot, bin | (rank << 3u), triangles);
        wg_emit_sph[e] = vec4<f32>(cw, rw);
        return;
    }
    emit_store(bin, atomicAdd(&state[VGC_ST_BINS + bin], 1u), inst, slot, triangles, cw, rw);
}

// Writes a cluster to entry `k` of its bin (counted as overflow past the
// capacity: the bin counter keeps the attempted count for growth).
fn emit_store(bin: u32, k: u32, inst: u32, slot: u32, triangles: u32, cw: vec3<f32>, rw: f32) {
    if k >= params.bin_capacity {
        stat_add(VGC_STAT_VISIBLE_OVERFLOW);
        return;
    }
    visible[bin * params.bin_capacity + k] = vec2<u32>(inst, slot);
    // The selected triangle count is kept even when the other counters are off.
    p_stats[VGC_STAT_TRIANGLES] += triangles;
    tile_accumulate(cw, rw, triangles);
    stat_add(VGC_STAT_EMITTED);
}

// Thread `b` (< VGC_BIN_COUNT): one global reservation for the wave's staged
// clusters of bin `b`.
fn emit_reserve(b: u32) {
    let n = atomicLoad(&wg_bin_n[b]);
    var base = 0u;
    if n != 0u { base = atomicAdd(&state[VGC_ST_BINS + b], n); }
    wg_bin_base[b] = base;
}

// Writes the staged clusters `lid, lid + WG, ..` to their reserved entries.
fn emit_commit(lid: u32) {
    let staged = min(atomicLoad(&wg_emit_n), EMIT_CAP);
    for (var e = lid; e < staged; e += WG) {
        let s = wg_emit[e];
        let sph = wg_emit_sph[e];
        let bin = s.z & 7u;
        emit_store(bin, wg_bin_base[bin] + (s.z >> 3u), s.x, s.y, s.w, sph.xyz, sph.w);
    }
}

// ── Cluster test (cut rule, cone, frustum, occlusion, binning) ─────────────

fn try_cluster(inst: u32, cs: u32, ic: InstCtx) {
    stat_add(VGC_STAT_CLUSTERS_TESTED);
    let c = vg_load_cluster(cs >> VG_PAGE_CLUSTER_SHIFT, cs & VG_PAGE_CLUSTER_MASK);
    let cw = (ic.m * vec4<f32>(c.sphere.xyz, 1.0)).xyz;
    let rw = c.sphere.w * ic.scale;
    let pw = (ic.m * vec4<f32>(c.parent_sphere.xyz, 1.0)).xyz;
    let prw = c.parent_sphere.w * ic.scale;
    let k = ic.err_scale * ic.scale;
    let s_px = proj_err(c.self_error * k, cw, rw);
    let p_px = proj_err(c.parent_error * k, pw, prw);

    // The cut: this cluster is drawn when its parent group is refined (the
    // parent is too coarse) and its own group is not (it is fine enough), or
    // when the clusters it was simplified from (the members of its producer
    // group) are not resident, so it stands in for them. `group_id` is
    // asset-local.
    var kids_resident = true;
    var self_refined = false;
    if (c.flags & VGC_CLUSTER_FLAG_LEAF) == 0u && c.group_id != VG_NODE_INVALID {
        let gref = vg_group_refs[vg_group_ref_base(ic.asset) + c.group_id];
        let child_page = vg_global_page(ic.asset, gref);
        kids_resident = vg_page_resident(child_page);
        let tau_s = ic.tau * tile_scale(cw, rw);
        self_refined = group_refines(inst, c.sphere, c.self_error, s_px, tau_s);
        // Streaming demand: urgent while the children are wanted now, a
        // prefetch while the error is within `prefetch_k` of the threshold
        // (resident pages in the band are touched, so they are not evicted
        // before the cut needs them).
        if s_px * params.prefetch_k > tau_s {
            vg_demand_page(child_page, self_refined);
        }
    }
    let parent_refined = group_refines(inst, c.parent_sphere, c.parent_error, p_px, ic.tau * tile_scale(pw, prw));
    let draw = parent_refined && (!self_refined || !kids_resident);
    if !draw { return; }

    if (ic.flags & VG_INST_CONE_CULL) != 0u {
        let cone = unpack4x8snorm(c.cone);
        // cone.w = cos(half-angle of the normal cone); 1.0 disables.
        var two_sided = false;
        if cone.w < 0.9999 && (params.flags & VGC_FLAG_MASKED) != 0u {
            // A two-sided surface is visible from behind: no normal-cone test.
            two_sided = (section_material_flags(inst, c.counts_material >> 16u) & MATERIAL_FLAG_DOUBLE_SIDED) != 0u;
        }
        if cone.w < 0.9999 && !two_sided {
            let axis = normalize((ic.m * vec4<f32>(cone.xyz, 0.0)).xyz);
            let dv = cw - cameras[0].position_near.xyz;
            let sin_a = sqrt(max(1.0 - cone.w * cone.w, 0.0));
            if dot(dv, axis) >= sin_a * length(dv) + rw {
                stat_add(VGC_STAT_CULLED_CONE);
                return;
            }
        }
    }
    if !sphere_in_frustum(cw, rw) {
        stat_add(VGC_STAT_CULLED_FRUSTUM);
        return;
    }
    let job = vec2<u32>(inst, (2u << 30u) | cs);
    let pol = occlusion_policy(job, cw, rw);
    if pol == POLICY_DEFER && defer_job(job) { return; }
    if pol == POLICY_DROP { stat_add(VGC_STAT_CULLED_OCCLUSION); return; }
    history_mark(job_key(job));
    let triangles = c.counts_material & 0xffu;
    var bin = choose_bin(triangles, ic, cw, rw, job_key(job));
    if (params.flags & VGC_FLAG_MASKED) != 0u {
        let mflags = section_material_flags(inst, c.counts_material >> 16u);
        if (mflags & MATERIAL_FLAG_ALPHA_TEST) != 0u { bin = masked_bin(bin); }
    }
    emit_cluster(inst, cs, bin, triangles, cw, rw);
}

// `gidx` is an asset-local group-ref index.
// `demanded`: the group's own error is above tau, so the cut needs its page. A
// group reached only through the hysteresis band of the entry test is drawn when
// its page is resident but never requested (that is the prefetch band's job).
fn do_group(inst: u32, gidx: u32, ic: InstCtx, demanded: bool) {
    let g = vg_group_refs[vg_group_ref_base(ic.asset) + gidx];
    let page = vg_global_page(ic.asset, g);
    if !demanded && !vg_page_resident(page) { return; }
    let slot = vg_use_or_request_page(page);
    if slot == VG_PAGE_NOT_RESIDENT { return; }
    let first = g.first_cluster_count & 0xffffu;
    let count = g.first_cluster_count >> 16u;
    // Each cluster becomes its own job on the workgroup stack, so the 64
    // threads of the next wave test 64 clusters instead of this thread looping
    // over the group while the other 63 wait at the wave's barrier. The group
    // is reserved whole and only while it fits (compare-exchange), so the
    // deque never passes `STACK_CAP` on its account and later pushes are not
    // forced out to the phase queue. A group that does not fit is tested here.
    let bot = wg_bot;
    var base = atomicLoad(&wg_top);
    var reserved = false;
    loop {
        if base - bot + count > STACK_CAP { break; }
        let r = atomicCompareExchangeWeak(&wg_top, base, base + count);
        if r.exchanged {
            reserved = true;
            break;
        }
        base = r.old_value;
    }
    for (var i = 0u; i < count; i++) {
        let cs = (slot << VG_PAGE_CLUSTER_SHIFT) | (first + i);
        if reserved {
            wg_stack[(base + i) & STACK_MASK] = vec2<u32>(inst, (2u << 30u) | cs);
        } else {
            try_cluster(inst, cs, ic);
        }
    }
}

// ── Job kinds ──────────────────────────────────────────────────────────────

fn job_gcbvh(job: vec2<u32>) {
    let idx = job.y & 0x3fffffffu;
    if idx >= vg_gcbvh_header.node_count { return; }
    let n = vg_gcbvh_nodes[idx];
    if !aabb_in_frustum(n.bmin, n.bmax) {
        stat_add(VGC_STAT_CULLED_FRUSTUM);
        return;
    }
    let c = (n.bmin + n.bmax) * 0.5;
    let r = length(n.bmax - n.bmin) * 0.5;
    let pol = occlusion_policy(job, c, r);
    if pol == POLICY_DEFER && defer_job(job) { return; }
    if pol == POLICY_DROP { stat_add(VGC_STAT_CULLED_OCCLUSION); return; }
    history_mark(job_key(job));
    if vg_bvh_is_leaf(n) {
        let count = vg_bvh_leaf_count(n);
        let first = vg_bvh_leaf_first(n);
        for (var i = 0u; i < count; i++) {
            // Leaf items are rows of `vg_instances` through the items table.
            let inst = vg_gcbvh_items[first + i];
            if inst < params.instance_count {
                push_job(vec2<u32>(inst, 3u << 30u));
            }
        }
    } else {
        push_job(vec2<u32>(0u, vg_bvh_child(n, 0u)));
        push_job(vec2<u32>(0u, vg_bvh_child(n, 1u)));
    }
}

// Whole-asset early-out. The GPU view of the asset header holds up to three
// hierarchy leaves (group ref + 1, 0 = unused) that contain every root cluster
// (`helio_vg_format::gpu_header_with_root_leaves`; none stored = no early-out).
// When their pages are resident and every root cluster already projects under
// tau, the cut is exactly the root clusters (errors and nested spheres are
// monotone up the DAG: the property the per-node `max_parent_error` pruning
// also relies on), so they are tested directly and the hierarchy walk is
// skipped. Returns false to walk normally.
fn root_leaf_ok(inst_ctx: InstCtx, leaf1: u32) -> bool {
    if leaf1 == 0u { return true; }
    let g = vg_group_refs[vg_group_ref_base(inst_ctx.asset) + leaf1 - 1u];
    let slot = vg_page_slot(vg_global_page(inst_ctx.asset, g));
    if slot == VG_PAGE_NOT_RESIDENT { return false; }
    let first = g.first_cluster_count & 0xffffu;
    let count = g.first_cluster_count >> 16u;
    let k = inst_ctx.err_scale * inst_ctx.scale;
    for (var i = 0u; i < count; i++) {
        let c = vg_load_cluster(slot, first + i);
        let cw = (inst_ctx.m * vec4<f32>(c.sphere.xyz, 1.0)).xyz;
        if proj_err(c.self_error * k, cw, c.sphere.w * inst_ctx.scale) > inst_ctx.tau * VGC_TAU_HOLD { return false; }
    }
    return true;
}

fn try_root_shortcut(inst: u32, hdr: VgAssetHeader) -> bool {
    if (params.flags & VGC_FLAG_NO_ROOT_SHORTCUT) != 0u { return false; }
    // Root leaves live in the header words no shader otherwise reads.
    let l0 = hdr.cluster_count;
    let l1 = hdr.root_page_count;
    let l2 = hdr.flags;
    if l0 == 0u { return false; }
    let ic = load_inst(inst);
    if !root_leaf_ok(ic, l0) || !root_leaf_ok(ic, l1) || !root_leaf_ok(ic, l2) { return false; }
    stat_add(VGC_STAT_ROOT_SHORTCUT);
    do_group(inst, l0 - 1u, ic, true);
    if l1 != 0u { do_group(inst, l1 - 1u, ic, true); }
    if l2 != 0u { do_group(inst, l2 - 1u, ic, true); }
    return true;
}

fn job_instance(job: vec2<u32>) {
    if job.x >= params.instance_count { return; }
    let inst = vg_instances[job.x];
    if (inst.flags & VG_INST_ENABLED) == 0u { return; }
    if inst.asset_slot >= arrayLength(&vg_asset_headers) { return; }
    // Not VG-active until the root pages are resident: draw nothing (the
    // derivation keeps such instances on the classic path).
    if !vg_asset_ready(inst.asset_slot) { return; }
    let hdr = vg_asset_headers[inst.asset_slot];
    if hdr.node_count == 0u { return; }
    if try_root_shortcut(job.x, hdr) { return; }
    let root = vg_node_base(inst.asset_slot) + hdr.root_node;
    let children = vg_nodes[root].children;
    for (var s = 0u; s < 4u; s++) {
        if children[s] != VG_NODE_INVALID {
            push_job(vec2<u32>(job.x, (1u << 30u) | (root << 2u) | s));
        }
    }
}

fn job_asset_item(job: vec2<u32>) {
    let ic = load_inst(job.x);
    let payload = job.y & 0x3fffffffu;
    let parent = payload >> 2u;     // global node index
    let slot = payload & 3u;
    let child = vg_nodes[parent].children[slot];
    if child == VG_NODE_INVALID { return; }
    let sph = vg_nodes[parent].spheres[slot];
    let cw = (ic.m * vec4<f32>(sph.xyz, 1.0)).xyz;
    let rw = sph.w * ic.scale;
    // Enter only while something below may still need drawing, against the
    // smallest threshold a group below may be held at (hysteresis).
    let mpe = vg_nodes[parent].max_parent_error[slot];
    let mpe_px = proj_err(mpe * ic.err_scale * ic.scale, cw, rw);
    if mpe_px <= ic.tau * VGC_TAU_HOLD * tile_scale_min(cw, rw) {
        stat_add(VGC_STAT_CULLED_SSE);
        return;
    }
    if !sphere_in_frustum(cw, rw) {
        stat_add(VGC_STAT_CULLED_FRUSTUM);
        return;
    }
    let pol = occlusion_policy(job, cw, rw);
    if pol == POLICY_DEFER && defer_job(job) { return; }
    if pol == POLICY_DROP { stat_add(VGC_STAT_CULLED_OCCLUSION); return; }
    history_mark(job_key(job));
    if (child & VG_NODE_LEAF_BIT) != 0u {
        do_group(job.x, child & ~VG_NODE_LEAF_BIT, ic, mpe_px > ic.tau);
    } else {
        let gchild = vg_node_base(ic.asset) + child;
        let kids = vg_nodes[gchild].children;
        for (var s = 0u; s < 4u; s++) {
            if kids[s] != VG_NODE_INVALID {
                push_job(vec2<u32>(job.x, (1u << 30u) | (gchild << 2u) | s));
            }
        }
    }
}

fn job_cluster(job: vec2<u32>) {
    try_cluster(job.x, job.y & 0x3fffffffu, load_inst(job.x));
}

fn process_job(job: vec2<u32>) {
    stat_add(VGC_STAT_JOBS);
    switch job.y >> 30u {
        case 0u: { job_gcbvh(job); }
        case 1u: { job_asset_item(job); }
        case 2u: { job_cluster(job); }
        default: { job_instance(job); }
    }
}

// ── Entry points ───────────────────────────────────────────────────────────

// ── Tile pass ───────────────────────────────────────────────────────────────
// One thread per 16x16 tile, before traversal: turns last frame's emitted-triangle
// estimate into the overdraw valve's scale for this frame and clears the estimate.
//
// The valve is a slow integrator with a dead band, so it cannot oscillate: a
// tile over `tile_limit * (1 + VGC_TILE_BAND)` triangles per covered pixel
// coarsens by `VGC_TILE_STEP_UP` per frame (up to `VGC_TILE_SCALE_MAX`), a tile
// under `tile_limit / (1 + VGC_TILE_BAND)` relaxes by `VGC_TILE_STEP_DOWN`, and
// in between it holds. A tile the dense-tile mask marks dense is exempt: its
// scale only relaxes (dense is an outcome of the cut, not a failure of it).
@compute @workgroup_size(64)
fn cs_tiles(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if t >= params.tiles_x * params.tiles_y { return; }
    let tris = f32(atomicExchange(&state[tile_word(t, 0u)], 0u)) / VGC_TILE_FIXED;
    let area = f32(atomicExchange(&state[tile_word(t, 1u)], 0u)) / VGC_TILE_FIXED;
    var g = max(bitcast<f32>(atomicLoad(&state[tile_word(t, 2u)])), 1.0);
    if params.tile_limit > 0.0 {
        let d = tris / max(min(area, f32(VGC_TILE * VGC_TILE)), 1.0);
        // The mask has the raster's grid (`dense_tiles_*`), which is not this
        // grid for the frame after a resize: read it by tile coordinate.
        var dense = false;
        if (params.flags & VGC_FLAG_DENSE_MASK) != 0u {
            let x = t % params.tiles_x;
            let y = t / params.tiles_x;
            let m = y * params.dense_tiles_x + x;
            if x < params.dense_tiles_x && y < params.dense_tiles_y && m < arrayLength(&dense_mask) {
                dense = dense_mask[m] != 0u;
            }
        }
        if dense || d < params.tile_limit / (1.0 + VGC_TILE_BAND) {
            g = max(g * VGC_TILE_STEP_DOWN, 1.0);
        } else if d > params.tile_limit * (1.0 + VGC_TILE_BAND) {
            g = min(g * VGC_TILE_STEP_UP, VGC_TILE_SCALE_MAX);
        }
    } else {
        g = 1.0;
    }
    atomicStore(&state[tile_word(t, 2u)], bitcast<u32>(g));
}

// One thread, frame start: seed the traversal with the GCBVH root.
@compute @workgroup_size(1)
fn cs_seed() {
    if params.mode == VGC_MODE_OFF { return; }
    if vg_gcbvh_header.node_count == 0u || vg_gcbvh_header.root_node == VG_BVH_INVALID || params.instance_count == 0u { return; }
    queue_put(0u, 0u, vec2<u32>(0u, vg_gcbvh_header.root_node));
    atomicStore(&state[VGC_ST_Q_TAIL], 1u);
    atomicStore(&state[VGC_ST_OUTSTANDING], 1u);
}

// One thread, late phase start: the early phase's deferred jobs (queue 1) are
// the late phase's queued work. A tail past the capacity (deferrals that found
// the queue full and were processed early) is clamped and kept as the peak.
@compute @workgroup_size(1)
fn cs_phase_init() {
    let raw = atomicLoad(&state[VGC_ST_Q_TAIL + 1u]);
    let n = min(raw, params.queue_capacity);
    atomicMax(&state[VGC_ST_STATS + VGC_STAT_PEAK_QUEUE], raw);
    atomicStore(&state[VGC_ST_Q_TAIL + 1u], n);
    atomicStore(&state[VGC_ST_Q_HEAD + 1u], 0u);
    atomicStore(&state[VGC_ST_OUTSTANDING], n);
}

// Thread 0's plan for the next wave (see `Wave`). `holding` says this
// workgroup is counted in `outstanding` (it holds work); `idle` counts the
// waves it has waited.
fn wave_plan(q: u32, holding: ptr<function, bool>, idle: ptr<function, u32>) -> Wave {
    var w = Wave(3u, 0u, 0u, 0u, 0u, 0u);
    let bot = wg_bot;
    let size = min(atomicLoad(&wg_top) - bot, STACK_CAP);
    if size > 0u {
        *idle = 0u;
        // Work sharing: while fewer jobs are queued than there are
        // workgroups (some may be starving), give away half of what exceeds
        // one wave, from the deque bottom: the oldest, coarsest jobs, whose
        // subtrees are the largest (work stealing's split).
        if size > WG && queue_backlog(q) < params.max_groups {
            let d = (size - WG) / 2u;
            if d > 0u {
                let dst = queue_reserve(q, d);
                if dst != 0xffffffffu {
                    w.share_n = d;
                    w.share_src = bot;
                    w.share_dst = dst;
                }
            }
        }
        let take = min(size - w.share_n, WG);
        let top = bot + size - take;
        atomicStore(&wg_top, top);
        w.mode = 1u;
        w.n = take;
        w.base = top;
        return w;
    }
    if !*holding {
        if atomicLoad(&state[VGC_ST_OUTSTANDING]) == 0u {
            w.mode = 0u;
            return w;
        }
        if queue_backlog(q) == 0u {
            *idle += 1u;
            // Only a group holding nothing ever stops here, so giving up
            // loses parallelism, never work (a guard against a hang).
            if *idle >= IDLE_SPIN_MAX { w.mode = 0u; }
            return w;
        }
        // Counted before claiming, so `outstanding` cannot read 0 while the
        // claimed jobs are in flight.
        atomicAdd(&state[VGC_ST_OUTSTANDING], 1u);
        *holding = true;
    }
    let c = queue_claim(q);
    if c.y > 0u {
        atomicSub(&state[VGC_ST_OUTSTANDING], c.y);
        *idle = 0u;
        w.mode = 2u;
        w.n = c.y;
        w.base = c.x;
    } else {
        atomicSub(&state[VGC_ST_OUTSTANDING], 1u);
        *holding = false;
    }
    return w;
}

// The traversal of one phase, in a single persistent dispatch: each workgroup
// pops waves of up to 64 jobs from its local deque, depth-first, takes jobs
// from the phase queue when the deque is empty, shares the bottom of its deque
// when the queue runs low, and exits when `outstanding` reaches 0. Nothing
// depends on the dispatch size: a workgroup that starts late finds the work
// done, and a waiting one only waits on running ones.
//
// The barriers run every iteration and only the work is predicated: FXC (the
// only compiler on Windows on ARM, which has no DXC) rejects a barrier inside
// a branch it cannot prove uniform, and after the data-dependent `break` it
// can prove nothing. The wave's emissions ride on the same barriers: bin slots
// are reserved (one global atomic per bin) after the jobs ran, written in the
// next section, and the tile table flushed and the staging reset in the last,
// before the next wave can emit.
@compute @workgroup_size(64)
fn cs_traverse(@builtin(local_invocation_index) lid: u32) {
    if lid < 5u { wg_planes[lid] = frustum_plane(lid); }
    if lid == 0u {
        atomicStore(&wg_top, 0u);
        wg_bot = 0u;
        atomicStore(&wg_emit_n, 0u);
    }
    if lid < VGC_BIN_COUNT { atomicStore(&wg_bin_n[lid], 0u); }
    if lid < TILE_SLOTS {
        atomicStore(&wg_tile_key[lid], TILE_EMPTY);
        atomicStore(&wg_tile_tris[lid], 0u);
        atomicStore(&wg_tile_area[lid], 0u);
    }
    let q = rp.phase;
    var holding = false;
    var idle = 0u;
    loop {
        if lid == 0u { wg_wave = wave_plan(q, &holding, &idle); }
        let w = workgroupUniformLoad(&wg_wave);
        if w.mode == 0u { break; }
        // The shared jobs leave the deque bottom before any push this wave
        // (pushes are bounded by the old `wg_bot`, so they cannot reach them).
        for (var i = lid; i < w.share_n; i += WG) {
            queue_put(q, w.share_dst + i, wg_stack[(w.share_src + i) & STACK_MASK]);
        }
        var job = NO_JOB;
        if lid < w.n {
            if w.mode == 1u {
                job = wg_stack[(w.base + lid) & STACK_MASK];
            } else {
                job = queue_take(q, w.base + lid);
            }
        }
        // Pops are read before any push may reuse the slots.
        workgroupBarrier();
        if job.y != NO_JOB.y { process_job(job); }
        workgroupBarrier();
        if lid < VGC_BIN_COUNT { emit_reserve(lid); }
        if lid == 0u { wg_bot += w.share_n; }
        workgroupBarrier();
        emit_commit(lid);
        workgroupBarrier();
        if lid == 0u { atomicStore(&wg_emit_n, 0u); }
        if lid < VGC_BIN_COUNT { atomicStore(&wg_bin_n[lid], 0u); }
        if lid < TILE_SLOTS { tile_flush(lid); }
    }
    stats_flush();
}

// One thread per phase end: clamp counts, publish ranges and indirect args.
@compute @workgroup_size(1)
fn cs_finalize() {
    let cap = params.bin_capacity;
    let phase = rp.phase;
    // Work still counted at the end was abandoned by the spin guards (an
    // invariant broke): report it as lost.
    let left = atomicLoad(&state[VGC_ST_OUTSTANDING]);
    if left > 0u {
        atomicAdd(&state[VGC_ST_STATS + VGC_STAT_JOB_OVERFLOW], left);
        atomicStore(&state[VGC_ST_OUTSTANDING], 0u);
    }
    let args = VGC_ST_ARGS + phase * VGC_ARGS_PHASE_WORDS;
    let rng = VGC_ST_RANGES + phase * VGC_BIN_COUNT * 2u;
    for (var b = 0u; b < VGC_BIN_COUNT; b++) {
        let raw = min(atomicLoad(&state[VGC_ST_BINS + b]), cap);
        var skip = 0u;
        var c = raw;
        if phase == 0u {
            atomicStore(&state[VGC_ST_PHASE1 + b], raw);
        } else {
            skip = min(atomicLoad(&state[VGC_ST_PHASE1 + b]), raw);
            c = raw - skip;
        }
        let first = b * cap + skip;
        atomicStore(&state[rng + b * 2u], first);
        atomicStore(&state[rng + b * 2u + 1u], c);
        if b < 4u || b == VGC_BIN_HW_MASKED {
            atomicStore(&state[args + b * 4u], select(VGC_HW_MASKED_VERTS, 96u * (b + 1u), b < 4u));
            atomicStore(&state[args + b * 4u + 1u], c);
            atomicStore(&state[args + b * 4u + 2u], 0u);
            atomicStore(&state[args + b * 4u + 3u], first);
        } else {
            let o = args + b * 4u;
            atomicStore(&state[o], min(c, VGC_SW_GROUP_WIDTH));
            atomicStore(&state[o + 1u], (c + VGC_SW_GROUP_WIDTH - 1u) / VGC_SW_GROUP_WIDTH);
            atomicStore(&state[o + 2u], 1u);
            atomicStore(&state[o + 3u], 0u);
        }
    }
}
