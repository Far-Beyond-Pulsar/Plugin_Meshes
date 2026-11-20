//!use helio_prelude
//!use frame_bsdf
//!use light_list
//!use light_forward
//!use specular_aa
//!use material_row
//!use material_surface
//!use frame_surface
//!use mesh_draw_rows
//!use mesh_skinning
//!use frame_ray_stream

// The meshes' forward program: lit in its drawer by the lights' forward
// function. A material class's template (its surface function) is composed
// after it (`RadiantTemplate::compose`); the material row, its texture table
// and `default_pbr_surface` are the materials' snippets.

enable wgpu_binding_array;


struct Globals {
    frame:             u32,
    delta_time:        f32,
    light_count:       u32,
    ambient_intensity: f32,
    ambient_color:     vec4<f32>,
    num_tiles_x:       u32,
    num_tiles_y:       u32,
    screen_width:      f32,
    screen_height:     f32,
    _pad0: u32,
    time: f32, // RADIANT_GLOBALS_TIME: seconds since start, for graph time nodes
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> cameras: array<Camera, 2>;
@group(0) @binding(1) var<uniform>          globals:           Globals;
@group(0) @binding(2) var<storage, read>    instance_data:     array<GpuInstanceData>;
@group(0) @binding(3) var<storage, read>    compacted_indices: array<u32>;
@group(0) @binding(4) var<storage, read>    lights:            array<GpuLight>;
@group(0) @binding(5) var<storage, read>    tile_light_lists:  array<u32>;
@group(0) @binding(6) var<storage, read>    tile_light_counts: array<u32>;
// The skin cache (`helio_mesh_api::SKIN_CACHE`, `mesh_skinning`).
@group(0) @binding(9) var<storage, read> skin_cache: array<SkinnedVertex>;
@group(0) @binding(10) var<storage, read> skin_cache_rows: array<SkinCacheRow>;

@group(1) @binding(0) var<storage, read>    materials:         array<GpuMaterial>;
@group(1) @binding(1) var<storage, read>    material_textures: array<MaterialTextureData>;
@group(1) @binding(2) var                   scene_sampler:     sampler;
@group(1) @binding(3) var                   scene_textures:    binding_array<texture_2d<f32>, 256>;

struct Vertex {
    @location(0) position:       vec3<f32>,
    @location(1) bitangent_sign: f32,
    @location(2) tex_coords:     vec2<f32>,
    @location(3) normal:         u32,
    @location(4) tangent:        u32,
}

struct VertexOutput {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal:   vec3<f32>,
    @location(2) tex_coords:     vec2<f32>,
    @location(3) world_tangent:  vec3<f32>,
    @location(4) bitangent_sign: f32,
    @location(5) @interpolate(flat) material_id: u32,
}

fn decode_snorm8x4(packed: u32) -> vec3<f32> {
    return unpack4x8snorm(packed).xyz;
}

// `slot` is per-instance vertex data, which every backend offsets by the
// indirect draw's `first_instance` (`helio_mesh_api::draw_slots`).
@vertex
fn vs_main(v: Vertex, @location(15) slot: u32, @builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let inst       = instance_data[compacted_indices[slot]];
    // A skinned instance's vertex from the skin cache; a rigid one's as it is.
    let skinned = mesh_skinned_vertex(
        inst, vertex_index, v.position, decode_snorm8x4(v.normal), decode_snorm8x4(v.tangent));
    let local_position = skinned.position;
    let local_normal = skinned.normal;
    let local_tangent = skinned.tangent;
    let world_pos  = inst.transform * vec4<f32>(local_position, 1.0);

    let normal_mat = mat3x3<f32>(
        inst.normal_mat_0.xyz,
        inst.normal_mat_1.xyz,
        inst.normal_mat_2.xyz,
    );

    let model_mat3 = mat3x3<f32>(
        inst.transform[0].xyz,
        inst.transform[1].xyz,
        inst.transform[2].xyz,
    );

    var out: VertexOutput;
    out.clip_position  = cameras[0].proj * (cameras[0].view * world_pos);
    out.world_position = world_pos.xyz;
    out.world_normal   = normalize(normal_mat  * local_normal);
    out.world_tangent  = normalize(model_mat3  * local_tangent);
    out.bitangent_sign = v.bitangent_sign;
    out.tex_coords     = v.tex_coords;
    out.material_id    = inst.material_id;
    return out;
}

// Whether the view the pipeline draws has a clip plane
// (`helio_frame_api::view::CLIP_CONSTANT`).
override HELIO_VIEW_CLIP: bool = false;

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    if HELIO_VIEW_CLIP && helio_view_clipped(cameras[0], input.world_position) { discard; }
    let material = material_at(input.material_id);
    let material_tex = material_texture_data(material,input.material_id);

    let surface = radiant_eval_surface(material, material_tex, MaterialInput(
        input.clip_position, input.world_position, input.world_normal,
        input.world_tangent, input.bitangent_sign, input.tex_coords,
        cameras[0].position_near.xyz, globals.frame,
    ));

    if surface.alpha <= 0.001 { discard; }

    let V = normalize(cameras[0].position_near.xyz - input.world_position);
    let N = surface.normal;
    let NdV = max(dot(N, V), 0.0);
    let albedo = surface.albedo.rgb;
    let alpha = surface.alpha;
    let roughness = surface.roughness;
    let metallic = surface.metallic;
    let F0 = surface.specular_f0;
    let emissive = surface.emissive;
    let ao = surface.ao;

    var Lo = vec3<f32>(0.0);

    // The lights of the pixel's tile (the lights' forward function).
    Lo += light_forward(
        input.clip_position.xy, globals.num_tiles_x, input.world_position,
        N, V, F0, albedo, roughness, metallic,
    );

    let ambient = globals.ambient_color.rgb * globals.ambient_intensity * albedo * ao;
    let color = Lo + ambient + emissive;

    return vec4<f32>(color, alpha);
}
