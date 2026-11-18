//!use gbuffer_output
enable wgpu_binding_array;
//!use helio_prelude
//!use specular_aa
//!use pbr_eval
//!use material_row
//!use material_surface
//!use mesh_draw_rows
//!use mesh_skinning
//!use frame_ray_stream
//!use lightmap_regions
//!use portal_chain

//! The meshes' G-buffer program (GPU-driven): the meshes' vertex layout and
//! instance rows, shaded by the material's template. A material class's
//! template (`radiant_eval_surface`, `helio_mats::radiant`) and a material
//! graph are composed after it (`RadiantTemplate::compose`); the material
//! row, its texture table and `default_pbr_surface` are the materials'
//! snippets.
//!
//! Two entry pairs, one program: `vs_main`/`fs_main` draw the culled
//! instances where they are; `vs_occurrence`/`fs_occurrence` draw them again
//! through the frame's portal chains (`helio_portal_api`): each occurrence
//! is an instance and a chain, mapped by the chain's composed matrix,
//! projected once by the camera, clipped to what the chain shows (the
//! aperture mask, each deeper opening's ray test, the deeper chains' holes)
//! and tagged with the chain's remap, so the lighting shades it in its own
//! world.
//!
//! Rasterises scene geometry into screen-sized textures:
//!   target 0 – albedo    (Rgba8Unorm)
//!   target 1 – normal    (Rgba16Float)
//!   target 2 – orm       (Rgba8Unorm)
//!   target 3 – emissive  (Rgba16Float)
//!   target 7 – velocity  (Rgba16Float) — screen-space motion in pixels/frame
//!                                          (helio-pass-tsr MOTION_VECTORS)
//!
//! Resolved F0 is packed into unused alpha channels:
//!   normal.a   = F0.r
//!   orm.a      = F0.g
//!   emissive.a = F0.b


struct Globals {
    frame: u32,
    delta_time: f32,
    light_count: u32,
    ambient_intensity: f32,
    ambient_color: vec4<f32>,
    csm_splits: vec4<f32>,
    debug_mode: u32,
    screen_width: f32,
    screen_height: f32,
    time: f32, // RADIANT_GLOBALS_TIME: seconds since start, for graph time nodes
}

// Whether the graph requested baked lighting's lightmap UV as the G-buffer's
// surface payload (`helio_frame_api::gbuffer::payload`): the drawer then
// writes each lightmapped object's atlas UV there.
override LIGHTMAP_PAYLOAD: bool = false;
// Whether the view the pipeline draws has a clip plane
// (`helio_frame_api::view::CLIP_CONSTANT`): only then does the fragment
// stage test it, so the primary view keeps early depth.
override HELIO_VIEW_CLIP: bool = false;

@group(0) @binding(0) var<storage, read> cameras: array<Camera, 2>;
@group(0) @binding(1) var<uniform>          globals:                Globals;
@group(0) @binding(2) var<storage, read>    instance_data:          array<GpuInstanceData>;
// Baked lighting's per-object regions (`helio_bake_api::LIGHTMAP_REGIONS`),
// by `static_objects` row; a one-region stand-in without a bake.
@group(0) @binding(3) var<storage, read>    lightmap_regions:       array<LightmapRegion>;
// Per-draw-call-group compacted original instance slots, surviving both
// frustum culling (IndirectDispatchPass) and Hi-Z occlusion culling
// (OcclusionCullPass). The draw slot (`@location(15)`) ranges over the
// group's now-possibly-smaller compacted count, so it must be redirected
// through this buffer before indexing `instance_data` — it no longer equals
// the instance's real slot directly.
@group(0) @binding(4) var<storage, read>    compacted_indices:      array<u32>;
// Coordinate-space transforms — see `libhelio::{coordinate_space, set_coordinate_space}`.
// Slot 0 is always identity, so an untagged instance (the common case) is
// unaffected beyond one extra constant-buffer read + mat4x4 multiply.
// Sublevels and portals both work by tagging instances/draws with a non-zero
// slot here and moving the whole space with a single matrix write.
@group(0) @binding(5) var<storage, read>    coordinate_spaces:      array<mat4x4<f32>>;
@group(0) @binding(6) var<storage, read>    coordinate_spaces_prev: array<mat4x4<f32>>;
// The skin cache (`helio_mesh_api::SKIN_CACHE`, `mesh_skinning`): a skinned
// instance's vertices this frame and last, as the meshes' skin pass wrote them.
@group(0) @binding(7) var<storage, read>    skin_cache: array<SkinnedVertex>;
@group(0) @binding(8) var<storage, read>    skin_cache_rows: array<SkinCacheRow>;

@group(1) @binding(0) var<storage, read>    materials:          array<GpuMaterial>;
@group(1) @binding(1) var<storage, read>    material_textures:  array<MaterialTextureData>;
@group(1) @binding(2) var                   scene_sampler:      sampler;
@group(1) @binding(3) var                   scene_textures:     binding_array<texture_2d<f32>, 256>;

@group(2) @binding(0) var uv_reference_texture: texture_2d<f32>;
@group(2) @binding(1) var uv_reference_sampler: sampler;

// The occurrence pipelines' (`helio_mesh_api::PortalOccurrenceFrameData`):
// per occurrence its chain (the instance slot is `compacted_indices`), the
// frame's live chains and their stages and holes, and the aperture mask.
@group(3) @binding(0) var<storage, read> occurrence_chains: array<u32>;
@group(3) @binding(1) var<storage, read> portal_chains:     array<GpuPortalChain>;
@group(3) @binding(2) var<storage, read> portal_stages:     array<GpuPortalStage>;
@group(3) @binding(3) var portal_mask: texture_2d<u32>;

struct Vertex {
    @location(0) position:       vec3<f32>,
    @location(1) bitangent_sign: f32,
    @location(2) tex_coords:     vec2<f32>,  // UV0 — material/albedo channel (may tile)
    @location(3) normal:         u32,
    @location(4) tangent:        u32,
    @location(5) lightmap_uv:    vec2<f32>,  // UV1 — dedicated lightmap channel, non-overlapping [0,1]
}

struct VertexOutput {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position:     vec3<f32>,
    @location(1) world_normal:       vec3<f32>,
    @location(2) tex_coords:         vec2<f32>,
    @location(3) world_tangent:      vec3<f32>,
    @location(4) bitangent_sign:     f32,
    @location(5) @interpolate(flat) material_id:    u32,
    @location(6) lightmap_uv:        vec2<f32>,  // Lightmap atlas UV (or (0,0) if no lightmap)
    @location(7) prev_clip_position: vec4<f32>,  // Previous frame clip-space position (velocity)
    // 1 when the instance moves by itself (its transform, skin or coordinate
    // space changed since last frame), else 0.
    @location(8) @interpolate(flat) moving: u32,
    // An occurrence's chain (`portal_chains` row), and its point in its own
    // world (before the chain's map) for the clip. Unused by `vs_main`.
    @location(9) @interpolate(flat) chain: u32,
    @location(10) content_position: vec3<f32>,
}

fn decode_snorm8x4(packed: u32) -> vec3<f32> {
    return unpack4x8snorm(packed).xyz;
}

// `slot` is per-instance vertex data, which every backend offsets by the
// indirect draw's `first_instance` (`helio_pass_gbuffer::draw_slots`).
@vertex
fn vs_main(v: Vertex, @location(15) slot: u32, @builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let identity = mat4x4<f32>(
        vec4<f32>(1.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 1.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 1.0, 0.0), vec4<f32>(0.0, 0.0, 0.0, 1.0));
    var out = mesh_vertex(v, instance_data[compacted_indices[slot]], vertex_index, identity, identity);
    return out;
}

// An occurrence through a portal chain: the instance (`compacted_indices`)
// mapped by its chain's composed matrix (this frame's and last frame's, for
// its motion).
@vertex
fn vs_occurrence(v: Vertex, @location(15) slot: u32, @builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let chain_index = occurrence_chains[slot];
    let chain = portal_chains[chain_index];
    var out = mesh_vertex(
        v, instance_data[compacted_indices[slot]], vertex_index,
        chain.content_to_drawn, chain.prev_content_to_drawn);
    out.chain = chain_index;
    out.moving = select(out.moving, 1u, portal_chain_moved(chain));
    // The surface's tag: drawn through the chain's remap.
    out.lightmap_uv = gbuffer_remap_tag(chain.remap, GBUFFER_TAG_PLAIN);
    return out;
}

/// The meshes' vertex: `inst` in its coordinate space, then through `map`
/// (identity where it is drawn; a chain's composed matrix through a portal,
/// `prev_map` last frame's).
fn mesh_vertex(
    v: Vertex,
    inst: GpuInstanceData,
    vertex_index: u32,
    map: mat4x4<f32>,
    prev_map: mat4x4<f32>,
) -> VertexOutput {
    // A skinned instance's vertex from the skin cache (its pose this frame
    // and last); a rigid one's as it is.
    let skinned = mesh_skinned_vertex(
        inst, vertex_index, v.position, decode_snorm8x4(v.normal), decode_snorm8x4(v.tangent));
    let local_position = skinned.position;
    let local_normal = skinned.normal;
    let local_tangent = skinned.tangent;

    // Coordinate space: world space (slot 0, identity) for the overwhelming
    // common case, or wherever a sublevel/portal has placed this instance.
    // See the `coordinate_spaces` binding above.
    let space_id   = instance_coordinate_space(inst.flags);
    let space      = coordinate_spaces[space_id];
    let space_prev = coordinate_spaces_prev[space_id];
    let map_rot    = mat3x3<f32>(map[0].xyz, map[1].xyz, map[2].xyz);
    let space_rot  = map_rot * mat3x3<f32>(space[0].xyz, space[1].xyz, space[2].xyz);

    let content_pos = space * (inst.transform * vec4<f32>(local_position, 1.0));
    let world_pos  = map * content_pos;

    // Normals transform by the inverse-transpose (stored in normal_mat), then
    // by the coordinate space's rotation. Coordinate spaces are always rigid
    // (translation + rotation, never scaled), so `space_rot` is its own
    // inverse-transpose and needs no separate normal-matrix treatment.
    let normal_mat = space_rot * mat3x3<f32>(
        inst.normal_mat_0.xyz,
        inst.normal_mat_1.xyz,
        inst.normal_mat_2.xyz,
    );

    // Tangents are NOT normals — they transform by the regular upper-3×3 of
    // the model matrix (no inverse-transpose), then the same rigid rotation.
    let model_mat3 = space_rot * mat3x3<f32>(
        inst.transform[0].xyz,
        inst.transform[1].xyz,
        inst.transform[2].xyz,
    );

    // Previous-frame clip position for velocity buffer. Uses the coordinate
    // space's own previous-frame transform so a moving sublevel/portal still
    // produces correct motion vectors instead of a false one-frame pop.
    let prev_local_position = skinned.previous;
    let prev_world  = prev_map * (space_prev * (inst.prev_model * vec4<f32>(prev_local_position, 1.0)));
    let prev_clip   = cameras[0].prev_view_proj * prev_world;

    var out: VertexOutput;
    // Keep projection separate: precombining P*V rounds away depth precision
    // needed by world-position reconstruction and ray-query self-hit offsets.
    // The depth prepass and forward/transparent variants must match this order.
    out.clip_position      = cameras[0].proj * (cameras[0].view * world_pos);
    out.world_position     = world_pos.xyz;
    out.world_normal       = normalize(normal_mat  * local_normal);
    out.world_tangent      = normalize(model_mat3  * local_tangent);
    out.bitangent_sign     = v.bitangent_sign;
    out.tex_coords         = v.tex_coords;
    out.material_id        = inst.material_id;
    out.prev_clip_position = prev_clip;
    out.chain              = 0u;
    out.content_position   = content_pos.xyz;
    let moved = any(inst.prev_model[0] != inst.transform[0]) || any(inst.prev_model[1] != inst.transform[1])
        || any(inst.prev_model[2] != inst.transform[2]) || any(inst.prev_model[3] != inst.transform[3])
        || any(space_prev[0] != space[0]) || any(space_prev[1] != space[1])
        || any(space_prev[2] != space[2]) || any(space_prev[3] != space[3]);
    out.moving = select(0u, 1u, moved || (inst.flags & INSTANCE_FLAG_SKINNED) != 0u);
    
    // The object's lightmap atlas UV, as the surface payload, when the graph
    // requested it (baked lighting's `LIGHTMAP_UV`); otherwise the plain tag.
    //
    // The lightmap UV is the mesh's UV1 channel: a unique, non-overlapping
    // [0,1] layout (tiled UV0 is not one), which the bake used too. Only a
    // mesh baked with UV1 has a region; every other object reads a zero
    // region and keeps the plain tag. The region's half-texel clamp keeps
    // bilinear filtering inside it.
    out.lightmap_uv = GBUFFER_TAG_PLAIN;
    if LIGHTMAP_PAYLOAD && inst.object_row < arrayLength(&lightmap_regions) {
        let atlas = lightmap_atlas_uv(lightmap_regions[inst.object_row], v.lightmap_uv);
        if atlas.x >= 0.0 {
            out.lightmap_uv = atlas;
        }
    }
    return out;
}

// ── Fragment ─────────────────────────────────────────────────────────────────

// `GBufferOutput`: the frame's targets (`//!use gbuffer_output`).

fn compute_velocity(input: VertexOutput) -> vec4<f32> {
    let prev_ndc = input.prev_clip_position.xy / input.prev_clip_position.w;
    let prev_pixel_x = (prev_ndc.x * 0.5 + 0.5) * globals.screen_width;
    let prev_pixel_y = (0.5 - prev_ndc.y * 0.5) * globals.screen_height;
    let prev_pixel = vec2<f32>(prev_pixel_x, prev_pixel_y);
    let p=cameras[0].proj;
    var residual=0.0;
    var valid=0.0;
    if abs(p[2].w)==1.0 && p[3].w==0.0 && p[0].y==0.0 && p[1].x==0.0
        && p[0].z==0.0 && p[1].z==0.0 && p[0].w==0.0 && p[1].w==0.0
        && p[3].x==0.0 && p[3].y==0.0 {
        let reconstructed_z=p[3].z/(p[2].w*input.clip_position.z-p[2].z);
        let actual_z=(cameras[0].view*vec4<f32>(input.world_position,1.0)).z;
        residual=actual_z-reconstructed_z;
        valid=select(0.0,2.0,abs(residual)<60000.0);
        residual=select(0.0,residual,valid==2.0);
    }
    // Store only the small reconstruction error in FP16, not the large view
    // depth itself. This preserves contact-scale precision at long distances.
    // w: 2 when the residual is valid, plus 1 when the instance moves by
    // itself (helio-pass-tsr MOTION_VECTORS: static geometry leaves its
    // camera motion to TSR, which derives it from depth at full precision).
    return vec4<f32>(input.clip_position.xy-prev_pixel,residual,valid+f32(input.moving));
}

@fragment
fn fs_main(vertex: VertexOutput, @builtin(front_facing) front: bool) -> GBufferOutput {
    if HELIO_VIEW_CLIP && helio_view_clipped(cameras[0], vertex.world_position) { discard; }
    return mesh_fragment(vertex, front);
}

// What a chain shows: beyond its outermost opening where the aperture mask
// holds it, through each deeper opening, and not behind a deeper chain's
// opening (that chain draws there).
@fragment
fn fs_occurrence(vertex: VertexOutput, @builtin(front_facing) front: bool) -> GBufferOutput {
    let chain = portal_chains[vertex.chain];
    let p = vertex.content_position;
    let mask = textureLoad(portal_mask, vec2<i32>(vertex.clip_position.xy), 0).r;
    if !portal_outer_shows(chain, portal_stages[chain.first_stage], p, mask) { discard; }
    for (var s = 1u; s < chain.stage_count; s += 1u) {
        if !portal_stage_shows(portal_stages[chain.first_stage + s], p) { discard; }
    }
    for (var h = 0u; h < chain.hole_count; h += 1u) {
        if portal_hole_covers(portal_stages[chain.first_hole + h], p) { discard; }
    }
    return mesh_fragment(vertex, front);
}

fn mesh_fragment(vertex: VertexOutput, front: bool) -> GBufferOutput {
    var input = vertex;
    let material = material_at(input.material_id);
    let material_tex = material_texture_data(material,input.material_id);
    // Reject empty card pixels before normal/ORM/extension texture reads,
    // including debug views. Graph materials retain their post-evaluation test.
    if (material.flags & MATERIAL_FLAG_ALPHA_TEST) != 0u && (material.flags & MATERIAL_FLAG_HAS_CUSTOM_SHADER) == 0u && material.material_class == MATERIAL_CLASS_DEFAULT {
        let alpha = material.base_color.a * sample_texture(material_tex.base_color, input.tex_coords, vec4<f32>(1.0)).a;
        if alpha < material_tex.params.z { discard; }
    }
    if !front && (material.flags & MATERIAL_FLAG_DOUBLE_SIDED) != 0u {
        input.world_normal = -input.world_normal;
        input.bitangent_sign = -input.bitangent_sign;
    }

    // DEBUG MODE 1: Show the shared UV reference texture across UV0.
    if globals.debug_mode == 1u {
        let grid = textureSample(uv_reference_texture, uv_reference_sampler, input.tex_coords);
        return GBufferOutput(
            vec4<f32>(grid.rgb, 1.0),
            vec4<f32>(0.0, 0.0, 1.0, 0.0),
            vec4<f32>(0.0),
            vec4<f32>(0.0),
            vec2<f32>(0.0),
            vec4<f32>(0.0),
            vec4<f32>(0.0),
            compute_velocity(input)
        );
    }

    // DEBUG MODE 2: Show texture sample directly
    if globals.debug_mode == 2u {
        let base_sample = sample_texture(material_tex.base_color, input.tex_coords, vec4<f32>(1.0));
        return GBufferOutput(
            vec4<f32>(base_sample.rgb, 1.0),
            vec4<f32>(0.0, 0.0, 1.0, 0.0),
            vec4<f32>(0.0),
            vec4<f32>(0.0),
            vec2<f32>(0.0),
            vec4<f32>(0.0),
            vec4<f32>(0.0),
            compute_velocity(input)
        );
    }

    // DEBUG MODE 3: Geometry normals only (skip normal mapping)
    if globals.debug_mode == 3u {
        let uv = input.tex_coords;
        let base_sample = sample_texture(material_tex.base_color, uv, vec4<f32>(1.0));
        let albedo = material.base_color * base_sample;
        let N_geom = normalize(input.world_normal);
        let orm_sample = sample_texture(material_tex.roughness_metallic, uv, vec4<f32>(1.0));
        let roughness = clamp(material.roughness_metallic.x * orm_sample.g, 0.045, 1.0);
        let metallic = clamp(material.roughness_metallic.y * orm_sample.b, 0.0, 1.0);
        return GBufferOutput(
            vec4<f32>(albedo.rgb, albedo.a),
            vec4<f32>(N_geom, 0.0),
            vec4<f32>(1.0, roughness, metallic, 0.0),
            vec4<f32>(0.0),
            vec2<f32>(0.0),
            vec4<f32>(0.0),
            vec4<f32>(0.0),
            compute_velocity(input)
        );
    }

    let surface = radiant_eval_surface(material, material_tex, MaterialInput(
        input.clip_position, input.world_position, input.world_normal,
        input.world_tangent, input.bitangent_sign, input.tex_coords,
        cameras[0].position_near.xyz, globals.frame,
    ));

    // Alpha test
    if surface.alpha <= 0.001 { discard; }
    if surface.alpha < material_tex.params.z { discard; }

    var out: GBufferOutput;
    out.albedo = vec4<f32>(surface.albedo.rgb, surface.alpha);
    out.normal = vec4<f32>(surface.normal, surface.specular_f0.r);
    out.orm = vec4<f32>(surface.ao, surface.roughness, surface.metallic, surface.specular_f0.g);
    out.emissive = vec4<f32>(surface.emissive, surface.specular_f0.b);
    out.surface = input.lightmap_uv;
    out.sss = vec4<f32>(surface.subsurface_color, surface.subsurface_radius);
    out.extra = vec4<f32>(surface.roughness_aniso_x, surface.roughness_aniso_y,
                          surface.aniso_rotation, f32(surface.flags));
    out.velocity = compute_velocity(input);
    return out;
}
