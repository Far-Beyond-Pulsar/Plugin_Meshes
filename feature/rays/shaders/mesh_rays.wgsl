//!use frame_ray_stream
//!use mesh_ray_surface
//!use material_row
enable wgpu_ray_query;
// The meshes' tracer of a ray stream (`helio_frame_api::rays::stream`,
// crate docs): each ray queries the frame's TLAS from its origin up to the
// nearest hit its row already holds (another tracer's), with its caster
// mask. Opaque triangles commit. Thin transmitting sheets (glass:
// non-opaque in their BLAS, `RayInstance::transmission`) are passed by
// nearest-hit rays and tint occlusion rays. Cut-outs
// (`RAY_INSTANCE_MASKED`: alpha-tested materials, non-opaque too) stop a
// ray only where their alpha says the surface is (`cut_out_surface`).
//
// A nearest-hit ray the meshes stop writes its hit and, in the same
// invocation, the surface there (`mesh_rays_nearest`): the surface needs
// the instance's object-to-frame transform, which only the committed
// intersection holds (the instance rows carry none). A later tracer that
// finds a nearer hit overwrites both; its own surface pass runs after every
// tracer traced, so the surface a row ends with is always its final hit's.
//
// The ray is traced from its origin exactly (`t` from 0): offsetting an
// origin off the surface it leaves is the consumer's, which knows that
// surface.

@group(0) @binding(0) var<storage, read> header: RayStreamHeader;
@group(0) @binding(1) var<storage, read> rays: array<SceneRay>;
@group(0) @binding(2) var<storage, read_write> hits: array<SceneHit>;
@group(0) @binding(3) var<uniform> trace: MeshRayFrame;
@group(0) @binding(4) var acc_struct: acceleration_structure;
// The meshes' surfaces (`mesh_ray_surface`, helio-mesh-api): their
// instance rows by TLAS instance, the scene's mesh pools (`PackedVertex`
// words, mesh-local indices) and the proxy pool.
@group(0) @binding(5) var<storage, read> instances: array<RayInstance>;
@group(0) @binding(6) var<storage, read> vertices: array<u32>;
@group(0) @binding(7) var<storage, read> indices: array<u32>;
@group(0) @binding(8) var<storage, read> proxy_vertices: array<f32>;
@group(0) @binding(9) var<storage, read> proxy_indices: array<u32>;
// A nearest-hit stream's surfaces (none in the occlusion pipeline).
@group(0) @binding(10) var<storage, read_write> surfaces: array<SceneSurface>;
// Group 1: the scene's material rows (`materials`, for a hit's metalness
// and roughness), then the same rows and the texture table for cut-outs'
// alpha (`helio_mats::vg_alpha`, spliced in before this file).
@group(1) @binding(0) var<storage, read> materials: array<GpuMaterial>;

struct MeshRayFrame {
    // x: this frame's seed, y: this tracer's id in the stream's hits.
    info: vec4<u32>,
}

// Cut-out candidates a ray tests at most (the cost bound in dense
// canopies); any further one stops it, as a surface.
const MAX_CUT_OUTS: u32 = 16u;
// Ray flag: an occlusion ray ends at its first opaque hit.
const FIRST_HIT: u32 = 0x04u;

// A hit's row of `instances`: its instance's first row (custom data) plus its
// segment (a BLAS holds a mesh's sections as segments, one row each).
fn ray_row(c: RayIntersection) -> u32 {
    return c.instance_custom_data + c.geometry_index;
}

// Whether row `i` is a cut-out.
fn is_cut_out(i: u32) -> bool {
    return i < arrayLength(&instances) && (instances[i].flags & RAY_INSTANCE_MASKED) != 0u;
}

fn hash(x: u32) -> u32 {
    var h = x * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    return (h >> 22u) ^ h;
}

// Whether the cut-out candidate `c`, met along unit `dir` by a ray whose
// footprint there is `width` across, is its surface: its material's alpha
// at the hit (uv0 by the barycentrics) passes the material's cutoff, as the
// raster's alpha test (`helio_mats::vg_alpha`). The alpha is sampled at the
// mip whose texels match the footprint on the triangle (ray cones). Past
// the finest mip a texel averages many, so the test turns stochastic there
// (alpha against a uniform threshold drawn from `seed`): a canopy then
// stops rays in proportion to its coverage at any distance, where a fixed
// cutoff would let its thin leaves vanish as their mips fall below it. A
// single-sided cut-out is not there from behind (the raster culls it).
fn cut_out_surface(c: RayIntersection, dir: vec3<f32>, width: f32, seed: u32) -> bool {
    let instance = instances[ray_row(c)];
    let t = ray_triangle(instance, c.primitive_index);
    let o = t.corners;
    let uv = t.uv;
    let b = vec3<f32>(1.0 - c.barycentrics.x - c.barycentrics.y, c.barycentrics);
    if (instance.flags & RAY_INSTANCE_TWO_SIDED) == 0u {
        // Its outside in object space, against the ray there.
        let outside = ray_triangle_outside(t, b);
        if dot(outside, c.world_to_object * vec4<f32>(dir, 0.0)) > 0.0 { return false; }
    }
    let a = vga_load(instance.material);
    // Twice the triangle's world area and twice its uv area.
    let w = c.object_to_world;
    let face = cross(w * vec4<f32>(o[1] - o[0], 0.0), w * vec4<f32>(o[2] - o[0], 0.0));
    let area = length(face);
    let du = uv[1] - uv[0];
    let dv = uv[2] - uv[0];
    let uv_area = abs(du.x * dv.y - du.y * dv.x);
    // The footprint on the triangle: wider the more it slants.
    let slant = max(abs(dot(face, dir)) / max(area, 1e-30), 0.05);
    let footprint = max(width * width / slant, 1e-12);
    let lod = vga_lod_for_area(a, uv_area, area / footprint);
    let alpha = vga_alpha_level(a, uv[0] * b.x + uv[1] * b.y + uv[2] * b.z, lod);
    let u = (f32(hash(seed ^ (c.primitive_index * 0x9e3779b9u) ^ (c.instance_index * 0x85ebca6bu)) >> 8u) + 0.5) / 16777216.0;
    return alpha > 0.001 && alpha >= mix(a.cutoff, u, saturate(lod));
}

// Ray `index`'s seed this frame: the frame's, never the stream's, so two
// consumers' rays through one canopy draw independently of each other.
fn ray_seed(index: u32) -> u32 {
    return hash(index ^ trace.info.x);
}

// The material row of a hit (the grey stand-in for a row the table lacks).
fn material_of(id: u32) -> GpuMaterial {
    if id < arrayLength(&materials) {
        return materials[id];
    }
    return material_stand_in();
}

@compute @workgroup_size(RAY_STREAM_GROUP)
fn mesh_rays_nearest(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= header.count || id.x >= arrayLength(&rays) { return; }
    let ray = rays[id.x];
    if ray.t_max <= 0.0 { return; }
    let reach = scene_hit_reach(ray, hits[id.x]);
    let seed = ray_seed(id.x);
    // Sheets are candidates, never confirmed: nearest-hit rays see through
    // glass. Cut-outs are confirmed where their surface is.
    var q: ray_query;
    rayQueryInitialize(&q, acc_struct, RayDesc(0u, scene_ray_mask(ray), 0.0, reach, ray.origin, ray.dir));
    var cut_outs = 0u;
    while rayQueryProceed(&q) {
        let candidate = rayQueryGetCandidateIntersection(&q);
        if !is_cut_out(ray_row(candidate)) { continue; }
        cut_outs++;
        if cut_outs > MAX_CUT_OUTS || cut_out_surface(candidate, ray.dir, scene_ray_footprint(ray, candidate.t), seed) {
            rayQueryConfirmIntersection(&q);
        }
    }
    let c = rayQueryGetCommittedIntersection(&q);
    if c.kind == RAY_QUERY_INTERSECTION_NONE || ray_row(c) >= arrayLength(&instances) { return; }
    // A hit no nearer than the row's stays the row's (the query reaches up
    // to it, and a tie keeps the earlier tracer's).
    let held = hits[id.x];
    if held.t > 0.0 && c.t >= held.t { return; }
    // The triangle: its frame corners for the face's normal, its vertex
    // normals (object space, to the frame by the inverse transpose) for the
    // side the mesh calls outside. A proxy has no vertex normals: its
    // winding says outside (its object-space face).
    let instance = instances[ray_row(c)];
    let t = ray_triangle(instance, c.primitive_index);
    let o = t.corners;
    var local = ray_triangle_outside(t, vec3<f32>(1.0 - c.barycentrics.x - c.barycentrics.y, c.barycentrics));
    if t.proxy {
        local = local / max(length(local), 1e-30);
    }
    let p0 = c.object_to_world * vec4<f32>(o[0], 1.0);
    let p1 = c.object_to_world * vec4<f32>(o[1], 1.0);
    let p2 = c.object_to_world * vec4<f32>(o[2], 1.0);
    var face = cross(p1 - p0, p2 - p0);
    face = face / max(length(face), 1e-20);
    let w = c.world_to_object;
    var shading = vec3<f32>(dot(w[0], local), dot(w[1], local), dot(w[2], local));
    if dot(shading, shading) < 1e-8 {
        // No vertex normals: two-sided.
        shading = -ray.dir;
    }
    shading = normalize(shading);
    if dot(face, shading) < 0.0 { face = -face; }
    // Two-sided: the side the ray meets is its outside.
    if (instance.flags & RAY_INSTANCE_TWO_SIDED) != 0u && dot(face, ray.dir) > 0.0 {
        face = -face;
        shading = -shading;
    }
    // Its inside: the back of a single-sided surface (the ray starts in a
    // wall or a solid). The surface's normals are then those of the side
    // the ray met.
    let inside = dot(face, ray.dir) > 0.0;
    if inside {
        face = -face;
        shading = -shading;
    }
    // The shading normal where it faces the ray; the face's where an
    // interpolated normal turns away (a silhouette's grazing hit).
    let n = select(face, shading, dot(shading, ray.dir) < 0.0);
    // A hit at the origin itself is still a hit (t 0 means none).
    hits[id.x] = scene_nearest(trace.info.y, max(c.t, 1e-6), inside, ray_row(c), c.primitive_index, pack2x16unorm(c.barycentrics));
    let material = material_of(instance.material);
    let metallic = select(0.0, material.roughness_metallic.y, material.workflow == MATERIAL_WORKFLOW_METALLIC);
    surfaces[id.x] = scene_surface(
        n,
        face,
        clamp(instance.base_color.rgb, vec3<f32>(0.0), vec3<f32>(1.0)),
        clamp(metallic, 0.0, 1.0),
        clamp(material.roughness_metallic.x, 0.0, 1.0),
        0.0,
        max(instance.emissive.rgb, vec3<f32>(0.0)),
    );
}

// The share of light that gets along `ray` (its whole length) past the
// meshes: none past an opaque mesh or a cut-out's surface, tinted by each
// sheet it crosses (its candidates; traversal order does not matter for a
// product).
fn transmittance(ray: SceneRay, seed: u32) -> vec3<f32> {
    var q: ray_query;
    rayQueryInitialize(&q, acc_struct, RayDesc(FIRST_HIT, scene_ray_mask(ray), 0.0, ray.t_max, ray.origin, ray.dir));
    var through = vec3<f32>(1.0);
    var cut_outs = 0u;
    while rayQueryProceed(&q) {
        let sheet = rayQueryGetCandidateIntersection(&q);
        if is_cut_out(ray_row(sheet)) {
            cut_outs++;
            if cut_outs <= MAX_CUT_OUTS && !cut_out_surface(sheet, ray.dir, scene_ray_footprint(ray, sheet.t), seed) { continue; }
            rayQueryTerminate(&q);
            return vec3<f32>(0.0);
        }
        through *= select(vec3<f32>(0.0), instances[ray_row(sheet)].transmission.rgb, ray_row(sheet) < arrayLength(&instances));
        if all(through <= vec3<f32>(0.0)) {
            rayQueryTerminate(&q);
            return vec3<f32>(0.0);
        }
    }
    if rayQueryGetCommittedIntersection(&q).kind != RAY_QUERY_INTERSECTION_NONE { return vec3<f32>(0.0); }
    return through;
}

@compute @workgroup_size(RAY_STREAM_GROUP)
fn mesh_rays_occlusion(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= header.count || id.x >= arrayLength(&rays) { return; }
    let ray = rays[id.x];
    if ray.t_max <= 0.0 { return; }
    let held = hits[id.x];
    // Already blocked by another tracer: nothing left to stop.
    if held.t > 0.0 { return; }
    let through = transmittance(ray, ray_seed(id.x));
    // Nothing of the meshes on the way: the row stays as it is.
    if all(through >= vec3<f32>(1.0)) { return; }
    hits[id.x] = scene_occlusion(scene_occlusion_through(held) * through);
}
