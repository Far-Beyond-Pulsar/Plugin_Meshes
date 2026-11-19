enable wgpu_binding_array;
//!use frame_ray_stream
//!use frame_ray_surface
//!use frame_surface
//!use mesh_ray_rows
//!use mesh_ray_surface
//!use specular_aa
//!use pbr_eval
//!use material_row
//!use material_surface
//!use material_gradients

// The meshes' hit program (helio-pass-mesh-rays `HIT_PROGRAM`): the surface
// of a ray's hit on a mesh, its material evaluated as the G-buffer's drawers
// evaluate it. A material class's template (`radiant_eval_surface`) and a
// material graph are composed after it (`RadiantTemplate::compose`), and the
// whole lowered to explicit gradients (`helio_mats::shader::explicit_gradients`):
// a compute stage has no derivatives, so every texture is read at the ray
// cone's footprint on the surface (`mesh_hit_footprint`).
//
// One program a material program (`helio_mesh_api::RayProgram`), two
// pipelines a program (`MATERIAL_SURFACE_DIFFUSE`: the stream's
// `SurfaceDetail`). It runs over one bin of the stream's meshes hits (the
// binning pass's list, `mesh_hit_bins.wgsl`), or, where every hit takes
// the table-driven program, over the stream's rows themselves.

// Group 0: the stream, the meshes' rows, pools and transforms, the bins.
@group(0) @binding(0) var<storage, read> header: RayStreamHeader;
@group(0) @binding(1) var<storage, read> rays: array<SceneRay>;
@group(0) @binding(2) var<storage, read> hits: array<SceneHit>;
@group(0) @binding(3) var<storage, read_write> surfaces: array<SceneSurface>;
@group(0) @binding(4) var<storage, read> instances: array<RayInstance>;
@group(0) @binding(5) var<storage, read> transforms: array<RayTransform>;
@group(0) @binding(6) var<storage, read> vertices: array<u32>;
@group(0) @binding(7) var<storage, read> indices: array<u32>;
@group(0) @binding(8) var<storage, read> proxy_vertices: array<f32>;
@group(0) @binding(9) var<storage, read> proxy_indices: array<u32>;
@group(0) @binding(10) var<storage, read> skinned_vertices: array<u32>;
// The bins (`mesh_hit_bins.wgsl`: each bin's dispatch, list offset and
// count) and the list of ray rows, bin by bin.
@group(0) @binding(11) var<storage, read> bins: array<u32>;
@group(0) @binding(14) var<storage, read> list: array<u32>;
// x: the tracer's id, y: the frame.
@group(0) @binding(12) var<uniform> hit_frame: vec4<u32>;
// x: the bin this dispatch shades, `HIT_UNBINNED` for every row of the
// stream (a dynamic offset into a table of every bin).
@group(0) @binding(13) var<uniform> hit_bin: vec4<u32>;

// Group 1: the material table, as every drawer binds it
// (`material_surface`'s names).
@group(1) @binding(0) var<storage, read> materials: array<GpuMaterial>;
@group(1) @binding(1) var<storage, read> material_textures: array<MaterialTextureData>;
@group(1) @binding(2) var scene_sampler: sampler;
@group(1) @binding(3) var scene_textures: binding_array<texture_2d<f32>, 256>;

// `mesh_hit_bins.wgsl`'s layout: a bin's words (dispatch x, y, z, list
// offset, count, three unused).
const HIT_BIN_WORDS: u32 = 8u;
const HIT_BINS: u32 = 64u;
const HIT_UNBINNED: u32 = 0xffffffffu;
// A bin's workgroups wrap at this many along x (`mesh_hit_bins.wgsl`).
const HIT_ROW_GROUPS: u32 = 32768u;

// A point of a triangle by barycentrics.
fn mesh_hit_at3(p: array<vec3<f32>, 3>, b: vec3<f32>) -> vec3<f32> {
    return p[0] * b.x + p[1] * b.y + p[2] * b.z;
}

fn mesh_hit_at2(p: array<vec2<f32>, 3>, b: vec3<f32>) -> vec2<f32> {
    return p[0] * b.x + p[1] * b.y + p[2] * b.z;
}

// The barycentrics, in the triangle's plane, of `q` (frame space): `p`
// the corners, `f` the face's unit normal.
fn mesh_hit_barycentrics(p: array<vec3<f32>, 3>, f: vec3<f32>, q: vec3<f32>) -> vec3<f32> {
    let e1 = p[1] - p[0];
    let e2 = p[2] - p[0];
    let d = q - p[0];
    let area = dot(f, cross(e1, e2));
    let inv = select(1.0 / area, 0.0, abs(area) < 1e-30);
    let u = dot(f, cross(d, e2)) * inv;
    let v = dot(f, cross(e1, d)) * inv;
    return vec3<f32>(1.0 - u - v, u, v);
}

// The footprint of a ray cone on the surface it hits (Akenine-Moller et
// al., "Improved Shader and Texture Level of Detail Using Ray Cones",
// JCGT 2021, the anisotropic formulation that Ray Tracing Gems II ch. 7
// builds on): the cone of width `width` along unit `dir` meets the plane of
// face normal `f` at `q` in an ellipse; its two axes, `a1` along the
// plane's steepest slope to the ray and `a2` across it, each scaled so that
// it spans the width seen along the ray. Where the ray meets the face head
// on, any two perpendicular axes of the width.
struct MeshHitFootprint {
    a1: vec3<f32>,
    a2: vec3<f32>,
}

fn mesh_hit_footprint(f: vec3<f32>, dir: vec3<f32>, width: f32) -> MeshHitFootprint {
    var a1 = dir - dot(f, dir) * f;
    if dot(a1, a1) < 1e-12 {
        // Head on: any direction in the plane.
        a1 = select(cross(f, vec3<f32>(1.0, 0.0, 0.0)), cross(f, vec3<f32>(0.0, 1.0, 0.0)), abs(f.x) > 0.9);
    }
    let p1 = a1 - dot(dir, a1) * dir;
    a1 *= width / max(length(p1), 1e-12);
    var a2 = cross(f, a1);
    let p2 = a2 - dot(dir, a2) * dir;
    a2 *= width / max(length(p2), 1e-12);
    return MeshHitFootprint(a1, a2);
}

// The frame-space shading normal (unit) of a triangle at barycentrics `b`:
// its outside taken to the frame, oriented as `normals` was.
fn mesh_hit_normal_at(t: RayTriangle, x: RayTransform, b: vec3<f32>, flip: bool) -> vec3<f32> {
    let n = ray_transform_normal(x, mesh_ray_outside(t, b));
    let unit = n / max(length(n), 1e-20);
    return select(unit, -unit, flip);
}

@compute @workgroup_size(64)
fn mesh_hit_surface(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let local = (group.y * HIT_ROW_GROUPS + group.x) * 64u + lane;
    var i = local;
    if hit_bin.x == HIT_UNBINNED {
        if i >= header.count || i >= arrayLength(&rays) { return; }
        let h = hits[i];
        if h.t <= 0.0 || scene_hit_tracer(h) != hit_frame.x { return; }
    } else {
        let at = hit_bin.x * HIT_BIN_WORDS;
        if local >= bins[at + 4u] { return; }
        i = list[bins[at + 3u] + local];
    }
    let hit = hits[i];
    let row = scene_hit_payload(hit);
    if row >= arrayLength(&instances) { return; }
    let ray = rays[i];
    let instance = instances[row];
    let x = transforms[min(instance.instance, arrayLength(&transforms) - 1u)];
    let t = ray_triangle(instance, hit.b);
    let bary = unpack2x16unorm(hit.c);
    let b = vec3<f32>(1.0 - bary.x - bary.y, bary);
    let p = array<vec3<f32>, 3>(
        ray_transform_point(x, t.corners[0]),
        ray_transform_point(x, t.corners[1]),
        ray_transform_point(x, t.corners[2]),
    );
    let position = mesh_hit_at3(p, b);
    let outside = ray_transform_normal(x, mesh_ray_outside(t, b));
    let normals = mesh_ray_normals(instance, cross(p[1] - p[0], p[2] - p[0]), outside, ray.dir);
    let uv = mesh_hit_at2(t.uv, b);

    // Where the ray cone's footprint reaches on the surface, as the
    // barycentrics of its axes' ends: UV0 and the shading normal there
    // are the gradients the material reads its textures and widens its
    // roughness with (a pixel's derivatives on the raster).
    let footprint = mesh_hit_footprint(normals.face, ray.dir, scene_ray_footprint(ray, hit.t));
    let b1 = mesh_hit_barycentrics(p, normals.face, position + footprint.a1);
    let b2 = mesh_hit_barycentrics(p, normals.face, position + footprint.a2);
    material_duv_dx = mesh_hit_at2(t.uv, b1) - uv;
    material_duv_dy = mesh_hit_at2(t.uv, b2) - uv;
    let n = mesh_hit_normal_at(t, x, b, normals.flipped);
    material_dn_dx = mesh_hit_normal_at(t, x, b1, normals.flipped) - n;
    material_dn_dy = mesh_hit_normal_at(t, x, b2, normals.flipped) - n;

    // The tangent frame as the G-buffer's vertex stage builds it: tangents
    // by the upper 3x3, the bitangent's sign turned with the normal.
    let tangent = ray_triangle_tangent(t, b);
    let sign = select(tangent.w, -tangent.w, normals.flipped);

    let material = material_at(instance.material);
    let material_tex = material_texture_data(material, instance.material);
    // What the material looks from: the ray's origin (the eye of a camera
    // ray, the surface a bounce left).
    var surface = radiant_eval_surface(material, material_tex, MaterialInput(
        vec4<f32>(0.0), position, normals.shading, ray_transform_vector(x, tangent.xyz),
        sign, uv, ray.origin, hit_frame.y,
    ));
    // A shading normal turned away from the ray (an interpolated or mapped
    // normal at a silhouette's grazing hit) is the face's.
    if dot(surface.normal, ray.dir) >= 0.0 {
        surface.normal = normals.face;
    }
    surfaces[i] = scene_surface_pack(surface, normals.face);
}
