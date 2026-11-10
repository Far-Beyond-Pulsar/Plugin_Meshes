//!use helio_prelude
// Virtual geometry cull: instance BVH -> per-asset cluster-group hierarchy ->
// cluster cut, with two-phase Hi-Z occlusion and tier binning.
//
// Composed in Rust (see `shader_source()`): helio prelude, generated `VGC_*`
// constants, vg_format.wgsl, helio-vg-stream bindings + helpers (binding 4..11),
// helio-pass-vg-bvh `NODES_WGSL`, then this
// file. Entry points: cs_tiles, cs_seed, cs_round_prepare, cs_traverse, cs_finalize.
//
// Job = vec2<u32>(x, y), tag = y >> 30:
//   0  GCBVH node          x = 0xffffffff, y = node index
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
    hist_read_base: u32,
    hist_write_base: u32,
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
    // per-tile block in `history` (3 words per tile: triangles, area, scale).
    tiles_x: u32,
    tiles_y: u32,
    tile_base: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct RoundParams {
    src: u32,
    dst: u32,
    phase: u32,
    hiz_valid: u32,
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
@group(0) @binding(18) var<storage, read_write> state: array<atomic<u32>>;
@group(0) @binding(19) var<storage, read_write> queues: array<vec2<u32>>;
@group(0) @binding(20) var<storage, read_write> visible: array<vec2<u32>>;
@group(0) @binding(21) var<storage, read_write> history: array<atomic<u32>>;
// GpuMaterial rows (24 words each); read only when VGC_FLAG_MASKED is set.
@group(0) @binding(22) var<storage, read> vgc_materials: array<u32>;

@group(1) @binding(0) var<uniform> rp: RoundParams;
@group(2) @binding(0) var<storage, read_write> dispatch_args: array<u32>;

const WG: u32 = 64u;
const SOURCE_CHUNK: u32 = 8u;
const REDISTRIBUTE_WAVES: u32 = 8u;
const STACK_CAP: u32 = 1024u;
const INF_ERR: f32 = 1.0e30;
const HUGE_PX: f32 = 3.0e38;
const ROOT_ONLY_TAU: f32 = 1.0e30;

var<workgroup> wg_stack: array<vec2<u32>, 1024>;
var<workgroup> wg_sp: atomic<u32>;
var<workgroup> wg_mode: u32;
var<workgroup> wg_n: u32;
var<workgroup> wg_base: u32;
var<workgroup> wg_planes: array<vec4<f32>, 5>;

// ── Small helpers ──────────────────────────────────────────────────────────

fn stat_add(i: u32) {
    if (params.flags & VGC_FLAG_STATS) != 0u {
        atomicAdd(&state[VGC_ST_STATS + i], 1u);
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

fn history_test(key: u32) -> bool {
    let bit = key & params.history_bits_mask;
    let w = atomicLoad(&history[params.hist_read_base + (bit >> 5u)]);
    return ((w >> (bit & 31u)) & 1u) != 0u;
}

fn history_mark(key: u32) {
    if (params.flags & VGC_FLAG_TWO_PHASE) == 0u { return; }
    let bit = key & params.history_bits_mask;
    atomicOr(&history[params.hist_write_base + (bit >> 5u)], 1u << (bit & 31u));
}

// Hysteresis memory: the same hash set, salted per decision kind and not gated
// by the two-phase flag. A bit says "this decision was taken last frame"; both
// the test and the mark are idempotent, so every job that reaches the same
// decision (the two sides of one group) agrees.
const SALT_REFINE: u32 = 0x1b873593u;
const SALT_SW: u32 = 0x51ed270bu;

fn hyst_test(key: u32) -> bool {
    let bit = key & params.history_bits_mask;
    let w = atomicLoad(&history[params.hist_read_base + (bit >> 5u)]);
    return ((w >> (bit & 31u)) & 1u) != 0u;
}

fn hyst_mark(key: u32) {
    let bit = key & params.history_bits_mask;
    atomicOr(&history[params.hist_write_base + (bit >> 5u)], 1u << (bit & 31u));
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
        if !history_test(job_key(job)) { return POLICY_DEFER; }
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

fn spill_job(job: vec2<u32>) {
    let t = atomicAdd(&state[VGC_ST_Q_TAIL + rp.dst], 1u);
    // dst 3 is the sink of a phase's last round: nothing is stored, finalize
    // reports the count as overflow.
    if rp.dst == 3u { return; }
    if t < params.queue_capacity {
        queues[rp.dst * params.queue_capacity + t] = job;
        stat_add(VGC_STAT_SPILLED);
        atomicMax(&state[VGC_ST_STATS + VGC_STAT_PEAK_QUEUE], t + 1u);
    } else {
        stat_add(VGC_STAT_JOB_OVERFLOW);
    }
}

fn push_job(job: vec2<u32>) {
    let i = atomicAdd(&wg_sp, 1u);
    if i < STACK_CAP {
        wg_stack[i] = job;
    } else {
        spill_job(job);
    }
}

fn defer_job(job: vec2<u32>) {
    let t = atomicAdd(&state[VGC_ST_Q_TAIL + 2u], 1u);
    if t < params.queue_capacity {
        queues[2u * params.queue_capacity + t] = job;
        stat_add(VGC_STAT_DEFERRED);
        atomicMax(&state[VGC_ST_STATS + VGC_STAT_PEAK_QUEUE], t + 1u);
    } else {
        stat_add(VGC_STAT_JOB_OVERFLOW);
    }
}

// ── Per-tile density: the overdraw valve and the dense-tile estimate ───────
//
// `history` carries, after its two visibility halves, three words per 16x16
// tile: the triangles and the covered area the clusters emitted last frame
// estimate into the tile (fixed point), and the valve's coarsening scale
// `g >= 1` (f32 bits; 0 reads as 1). `cs_tiles` turns the estimate into the
// scale and clears it; `emit_cluster` adds to it; the dense-tile classifier
// reads the triangle word (so hardware-class clusters count towards "this tile
// is dense" without the software raster having run).

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
            g = max(g, bitcast<f32>(atomicLoad(&history[tile_word(t, 2u)])));
        }
    }
    return g;
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
            atomicAdd(&history[tile_word(t, 0u)], u32(f32(tris) * share * VGC_TILE_FIXED + 0.5));
            atomicAdd(&history[tile_word(t, 1u)], u32(disc * share * VGC_TILE_FIXED + 0.5));
        }
    }
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
    var h = vg_hash(inst, bitcast<u32>(err));
    h = vg_hash(h, bitcast<u32>(sph.x));
    h = vg_hash(h, bitcast<u32>(sph.y));
    h = vg_hash(h, bitcast<u32>(sph.z));
    let key = h ^ SALT_REFINE;
    let refine = px > select(tau, tau * VGC_TAU_HOLD, hyst_test(key));
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

fn emit_cluster(inst: u32, slot: u32, bin: u32, triangles: u32, cw: vec3<f32>, rw: f32) {
    let k = atomicAdd(&state[VGC_ST_BINS + bin], 1u);
    if k >= params.bin_capacity {
        stat_add(VGC_STAT_VISIBLE_OVERFLOW);
        return;
    }
    visible[bin * params.bin_capacity + k] = vec2<u32>(inst, slot);
    // The selected triangle count is kept even when the other counters are off.
    atomicAdd(&state[VGC_ST_STATS + VGC_STAT_TRIANGLES], triangles);
    tile_accumulate(cw, rw, triangles);
    stat_add(VGC_STAT_EMITTED);
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
    if pol == POLICY_DEFER { defer_job(job); return; }
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
    for (var i = 0u; i < count; i++) {
        try_cluster(inst, (slot << VG_PAGE_CLUSTER_SHIFT) | (first + i), ic);
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
    if pol == POLICY_DEFER { defer_job(job); return; }
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
        push_job(vec2<u32>(0xffffffffu, vg_bvh_child(n, 0u)));
        push_job(vec2<u32>(0xffffffffu, vg_bvh_child(n, 1u)));
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
    if mpe_px <= ic.tau * VGC_TAU_HOLD {
        stat_add(VGC_STAT_CULLED_SSE);
        return;
    }
    if !sphere_in_frustum(cw, rw) {
        stat_add(VGC_STAT_CULLED_FRUSTUM);
        return;
    }
    let pol = occlusion_policy(job, cw, rw);
    if pol == POLICY_DEFER { defer_job(job); return; }
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
    let tris = f32(atomicExchange(&history[tile_word(t, 0u)], 0u)) / VGC_TILE_FIXED;
    let area = f32(atomicExchange(&history[tile_word(t, 1u)], 0u)) / VGC_TILE_FIXED;
    var g = max(bitcast<f32>(atomicLoad(&history[tile_word(t, 2u)])), 1.0);
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
    atomicStore(&history[tile_word(t, 2u)], bitcast<u32>(g));
}

// One thread, frame start: seed the traversal with the GCBVH root.
@compute @workgroup_size(1)
fn cs_seed() {
    if params.mode == VGC_MODE_OFF { return; }
    if vg_gcbvh_header.node_count == 0u || vg_gcbvh_header.root_node == VG_BVH_INVALID || params.instance_count == 0u { return; }
    queues[0] = vec2<u32>(0xffffffffu, vg_gcbvh_header.root_node);
    atomicStore(&state[VGC_ST_Q_TAIL], 1u);
}

// One thread per round: reset queue cursors and size the persistent dispatch.
@compute @workgroup_size(1)
fn cs_round_prepare() {
    let n = min(atomicLoad(&state[VGC_ST_Q_TAIL + rp.src]), params.queue_capacity);
    atomicStore(&state[VGC_ST_Q_HEAD + rp.src], 0u);
    atomicStore(&state[VGC_ST_Q_TAIL + rp.dst], 0u);
    atomicStore(&state[VGC_ST_Q_HEAD + rp.dst], 0u);
    dispatch_args[0] = min((n + SOURCE_CHUNK - 1u) / SOURCE_CHUNK, params.max_groups);
    dispatch_args[1] = 1u;
    dispatch_args[2] = 1u;
}

// Persistent workgroups: pull 64-job chunks from the source queue, traverse
// depth-first through a workgroup-local stack, spill to the next round's queue
// only when that stack is full.
@compute @workgroup_size(64)
fn cs_traverse(@builtin(local_invocation_index) lid: u32) {
    if lid < 5u { wg_planes[lid] = frustum_plane(lid); }
    if lid == 0u { atomicStore(&wg_sp, 0u); }
    let src = rp.src;
    let src_n = min(atomicLoad(&state[VGC_ST_Q_TAIL + src]), params.queue_capacity);
    workgroupBarrier();
    var waves = 0u;
    loop {
        if lid == 0u {
            let sp = min(atomicLoad(&wg_sp), STACK_CAP);
            if sp > 0u {
                let take = min(sp, WG);
                atomicStore(&wg_sp, sp - take);
                wg_n = take;
                wg_base = sp - take;
                wg_mode = 1u;
            } else {
                let base = atomicAdd(&state[VGC_ST_Q_HEAD + src], SOURCE_CHUNK);
                if base >= src_n {
                    wg_mode = 0u;
                    wg_n = 0u;
                    wg_base = 0u;
                } else {
                    wg_mode = 2u;
                    wg_n = min(SOURCE_CHUNK, src_n - base);
                    wg_base = base;
                }
            }
        }
        workgroupBarrier();
        let mode = workgroupUniformLoad(&wg_mode);
        if mode == 0u { break; }
        let n = workgroupUniformLoad(&wg_n);
        let base = workgroupUniformLoad(&wg_base);
        var job = vec2<u32>(0u);
        let have = lid < n;
        if have {
            if mode == 1u {
                job = wg_stack[base + lid];
            } else {
                job = queues[src * params.queue_capacity + base + lid];
            }
        }
        // Pops are read before any push may reuse the slots.
        workgroupBarrier();
        if have { process_job(job); }
        workgroupBarrier();
        waves += 1u;
        // The seed is one job: draining its entire subtree on one SM leaves
        // the rest of the GPU idle. Share a bounded frontier with the next
        // dispatch, whose workgroup count reflects that expanded frontier.
        // The final round drains normally; yielding there would drop work.
        if waves >= REDISTRIBUTE_WAVES && rp.dst != 3u {
            if lid == 0u { wg_n = min(atomicLoad(&wg_sp), STACK_CAP); }
            workgroupBarrier();
            let pending = workgroupUniformLoad(&wg_n);
            for (var i = lid; i < pending; i += WG) { spill_job(wg_stack[i]); }
            workgroupBarrier();
            if lid == 0u { atomicStore(&wg_sp, 0u); }
            workgroupBarrier();
            waves = 0u;
        }
    }
}

// One thread per phase end: clamp counts, publish ranges and indirect args.
@compute @workgroup_size(1)
fn cs_finalize() {
    let cap = params.bin_capacity;
    let phase = rp.phase;
    let left = atomicLoad(&state[VGC_ST_Q_TAIL + 3u]);
    if left > 0u {
        atomicAdd(&state[VGC_ST_STATS + VGC_STAT_JOB_OVERFLOW], left);
        atomicStore(&state[VGC_ST_Q_TAIL + 3u], 0u);
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
