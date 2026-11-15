//!use helio_prelude
//!use pbr_eval
//!use light_list
//!use light_forward
//!use specular_aa
//!use material_row
//!use material_surface
//!use mesh_draw_rows
enable wgpu_binding_array;

// The meshes' transparent program: the vertex layout and instance rows,
// shaded by the material's transparent template (`radiant_eval_transparent`,
// composed after this by `RadiantTemplate::compose`) and lit by the lights'
// forward function (the lights plugin's snippets).
// Camera struct matches the gbuffer's layout so position_near is correct.


struct Globals {
    frame:             u32,
    delta_time:        f32,
    light_count:       u32,
    ambient_intensity: f32,
    ambient_color:     vec4<f32>,
    csm_splits:        vec4<f32>,
    num_tiles_x:       u32,
    num_tiles_y:       u32,
    screen_width:      f32,
    screen_height:     f32,
    // 1 when `lights[i]`/`transforms[i]` share the same raw entity index
    // (SceneDB-direct); 0 when `lights` is `ctx.scene.lights`, a freshly
    // rebuilt dense array needing `light_entity_indices[i]`. See that
    // binding's doc.
    light_mode_direct_index: u32,
    time: f32, // RADIANT_GLOBALS_TIME: seconds since start, for graph time nodes
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> cameras: array<Camera, 2>;
@group(0) @binding(1) var<uniform>       globals:       Globals;
@group(0) @binding(2) var<storage, read> instance_data: array<GpuInstanceData>;
struct SkinVertex { joints: vec4<u32>, weights: vec4<f32> }
@group(0) @binding(3) var<storage, read> skin_vertices: array<SkinVertex>;
@group(0) @binding(4) var<storage, read> bone_palette: array<mat4x4<f32>>;

@group(1) @binding(0) var<storage, read> lights:            array<GpuLight>;
@group(1) @binding(1) var<storage, read> tile_light_lists:  array<u32>;
@group(1) @binding(2) var<storage, read> tile_light_counts: array<u32>;

// Parallel to `lights` when `globals.light_mode_direct_index == 0` -- entry
// `i` is the real SceneDB entity index `lights[i]` was built from this frame
// (`ctx.scene.lights` is a freshly-rebuilt dense array every frame, not
// itself entity-indexed). Ignored (identity) when
// `light_mode_direct_index == 1`, since `lights` is then the SceneDB
// `"scene_lights"` buffer directly, already entity-indexed the same way
// `transforms` is.
@group(1) @binding(3) var<storage, read> light_entity_indices: array<u32>;
// Legacy template ABI; the default path uses the world-space position in
// LightComponent.position_range, matching the other SceneDB render passes.
struct Transform {
    position: array<f32, 3>,
    rotation: array<f32, 3>,
    scale:    array<f32, 3>,
}
@group(1) @binding(4) var<storage, read> transforms: array<Transform>;

@group(1) @binding(5) var<storage, read> materials: array<GpuMaterial>;
@group(1) @binding(6) var<storage, read> material_textures: array<MaterialTextureData>;
@group(1) @binding(7) var scene_sampler: sampler;
@group(1) @binding(8) var scene_textures: binding_array<texture_2d<f32>, 256>;

struct Vertex {
    @location(0) position:       vec3<f32>,
    @location(1) bitangent_sign: f32,
    @location(2) tex_coords:     vec2<f32>,
    @location(3) normal:         u32,
    @location(4) tangent:        u32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal:   vec3<f32>,
    @location(2) tex_coords:     vec2<f32>,
    @location(3) world_tangent: vec3<f32>,
    @location(4) @interpolate(flat) bitangent_sign: f32,
    @location(5) @interpolate(flat) material_id: u32,
}

fn decode_snorm8x4(packed: u32) -> vec3<f32> {
    return unpack4x8snorm(packed).xyz;
}

// `slot` is per-instance vertex data, which every backend offsets by the
// indirect draw's `first_instance` (`helio_pass_gbuffer::draw_slots`).
@vertex
fn vs_main(vertex: Vertex, @location(15) slot: u32, @builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let inst      = instance_data[slot];
    var local_position = vertex.position;
    var local_normal = decode_snorm8x4(vertex.normal);
    var local_tangent = decode_snorm8x4(vertex.tangent);
    if inst.bone_palette_count > 0u {
        let skin = skin_vertices[vertex_index + inst.skin_vertex_offset];
        var blended = mat4x4<f32>(vec4f(0.0), vec4f(0.0), vec4f(0.0), vec4f(0.0));
        var total = 0.0;
        for (var i = 0u; i < 4u; i = i + 1u) {
            if skin.weights[i] > 0.0 && skin.joints[i] < inst.bone_palette_count {
                blended += bone_palette[inst.bone_palette_offset + skin.joints[i]] * skin.weights[i];
                total += skin.weights[i];
            }
        }
        if total > 0.00001 {
            blended = blended * (1.0 / total);
            local_position = (blended * vec4f(vertex.position, 1.0)).xyz;
            local_normal = normalize(mat3x3<f32>(blended[0].xyz, blended[1].xyz, blended[2].xyz) * local_normal);
            local_tangent = normalize(mat3x3<f32>(blended[0].xyz, blended[1].xyz, blended[2].xyz) * local_tangent);
        }
    }
    let world_pos = inst.transform * vec4<f32>(local_position, 1.0);
    let normal_mat = mat3x3<f32>(
        inst.normal_mat_0.xyz,
        inst.normal_mat_1.xyz,
        inst.normal_mat_2.xyz,
    );
    var out: VertexOutput;
    out.clip_position  = cameras[0].proj * (cameras[0].view * world_pos);
    out.world_position = world_pos.xyz;
    out.world_normal   = normalize(normal_mat * local_normal);
    out.world_tangent  = normalize(normal_mat * local_tangent);
    out.tex_coords     = vertex.tex_coords;
    out.bitangent_sign = vertex.bitangent_sign;
    out.material_id    = inst.material_id;
    return out;
}

// HELIO_TRANSPARENT_REACTIVITY: location 2 carries alpha coverage for TAA.
struct TransparentOutput {
    @location(0) accumulation: vec4<f32>,
    @location(1) revealage: f32,
    @location(2) reactivity: vec4<f32>,
}
@fragment
fn fs_main(input: VertexOutput) -> TransparentOutput {
    // RADIANT_OVERRIDE_TRANSPARENT
    // RADIANT_OVERRIDE_END

    let material = material_at(input.material_id);
    let material_tex = material_texture_data(material, input.material_id);
    let V = normalize(cameras[0].position_near.xyz - input.world_position);
    let material_input = MaterialInput(
        input.clip_position, input.world_position, input.world_normal,
        input.world_tangent, input.bitangent_sign, input.tex_coords,
        cameras[0].position_near.xyz, globals.frame,
    );
    var surface = radiant_eval_transparent(
        material, material_tex, material_input,
        globals.ambient_color.rgb * globals.ambient_intensity,
    );

    let pbr = default_pbr_surface(material, material_tex, material_input);
    let N = select(-pbr.normal, pbr.normal, dot(pbr.normal, V) >= 0.0);

    let Lo = light_forward(
        input.clip_position.xy, globals.num_tiles_x, input.world_position,
        N, V, pbr.specular_f0, pbr.albedo.rgb,
        pbr.roughness, pbr.metallic,
    );

    surface = vec4<f32>(surface.rgb + Lo, clamp(surface.a * pbr.alpha, 0.0, 1.0));
    // HELIO_VIEW_SURFACE_EFFECT
    let alpha = clamp(surface.a, 0.0, 1.0);
    if alpha <= 0.001 { discard; }
    let depth = clamp(input.clip_position.z, 0.0, 1.0);
    let weight = clamp(
        (alpha * 8.0 + 0.01) * pow(max(0.01, 1.0 - depth * 0.9), 3.0),
        0.01,
        8.0,
    );
    return TransparentOutput(
        vec4<f32>(surface.rgb * alpha * weight, alpha * weight),
        alpha,
        vec4<f32>(alpha),
    );
}
