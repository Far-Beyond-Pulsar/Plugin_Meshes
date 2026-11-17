//! The meshes' tracer of ray streams against what a stream's rows must hold:
//! hits and surfaces compared with a CPU ray-triangle reference (distance,
//! tracer, inside, normals, colours, payload), the nearer of a row's hit and
//! the meshes', a ray's length and caster mask, cut-outs (exact at a fine
//! footprint, in proportion to coverage at a coarse one), occlusion through
//! sheets, cut-outs and opaque meshes, a proxy (virtual-geometry) instance,
//! rows past the header's count, and a frame without a TLAS. The last test
//! traces a stream in a mesh scene's graph, its TLAS from the standalone
//! projection. Needs hardware ray queries: without them each test returns
//! at once.

use std::sync::Arc;

use glam::{Mat3, Mat4, Quat, Vec2, Vec3};
use helio_core::{BlasGeometry, BlasManager, BlasSegment, TlasInstanceInput, TlasManager};
use helio_frame_api::rays::stream::{
    self, RayClass, SceneHit, SceneRay, SceneSurface, Stream, StreamBuffers, TracerId, MASK_ALL,
    MASK_MOVABLE, MASK_STATIC,
};
use helio_mesh_api::{
    RayInstance, RAY_INSTANCE_MASKED, RAY_INSTANCE_PROXY, RAY_INSTANCE_TWO_SIDED,
};
use helio_pass_mesh_rays::{payload, MeshRayScene, MeshRayTracer, StreamBindings};
use wgpu::util::DeviceExt;

struct Gpu {
    device: Arc<wgpu::Device>,
    queue: wgpu::Queue,
}

/// A device with ray queries (none without them).
fn gpu() -> Option<&'static Gpu> {
    static GPU: std::sync::OnceLock<Option<Gpu>> = std::sync::OnceLock::new();
    GPU.get_or_init(|| {
        let instance = wgpu::Instance::new(helio_core::wgpu_instance_descriptor());
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).ok()?;
        if !adapter
            .features()
            .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
        {
            eprintln!("no ray queries on {:?}: skipped", adapter.get_info().name);
            return None;
        }
        let limits = adapter.limits();
        // The material table as the renderer binds it (bindless when it can).
        let bindless = adapter.features() & helio_mats::BINDLESS_MATERIAL_FEATURES;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: wgpu::Features::EXPERIMENTAL_RAY_QUERY | bindless,
            required_limits: wgpu::Limits {
                max_buffer_size: limits.max_buffer_size.min(u32::MAX as u64),
                ..limits
            },
            // These tests exercise wgpu's experimental ray queries on purpose.
            experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
            ..Default::default()
        }))
        .ok()?;
        Some(Gpu {
            device: Arc::new(device),
            queue,
        })
    })
    .as_ref()
}

// ── Scenes ──────────────────────────────────────────────────────────────────

/// A quad of a scene, an instance of its own: its corners in object space
/// (a parallelogram: `corner`, plus `u`, plus `v`), its vertex normal (none:
/// zero normals), its surface, UVs from 0 to `uv` across it, its row's
/// `RAY_INSTANCE_*` flags and material row, its transmission (a sheet), its
/// object-to-frame transform and caster mask. A proxy's triangles go in the
/// proxy pool.
#[derive(Clone)]
struct Quad {
    corner: Vec3,
    u: Vec3,
    v: Vec3,
    normal: Option<Vec3>,
    base_color: Vec3,
    emissive: Vec3,
    uv: f32,
    flags: u32,
    material: u32,
    transmission: Vec3,
    transform: Mat4,
    mask: u8,
}

impl Default for Quad {
    fn default() -> Self {
        Self {
            corner: Vec3::new(-1.0, -1.0, 0.0),
            u: Vec3::X * 2.0,
            v: Vec3::Y * 2.0,
            normal: Some(Vec3::Z),
            base_color: Vec3::splat(0.5),
            emissive: Vec3::ZERO,
            uv: 1.0,
            flags: 0,
            material: 0,
            transmission: Vec3::ZERO,
            transform: Mat4::IDENTITY,
            mask: MASK_STATIC,
        }
    }
}

impl Quad {
    fn corners(&self) -> [Vec3; 4] {
        [
            self.corner,
            self.corner + self.u,
            self.corner + self.u + self.v,
            self.corner + self.v,
        ]
    }

    fn proxy(&self) -> bool {
        self.flags & RAY_INSTANCE_PROXY != 0
    }
}

/// Mesh-local indices of a quad's two triangles.
const QUAD_INDICES: [u32; 6] = [0, 1, 2, 0, 2, 3];

/// The quads as a TLAS (an instance each) with their rows, the way a
/// projection publishes them, and a material table.
struct Scene {
    tlas: TlasManager,
    _blas: BlasManager,
    instances: wgpu::Buffer,
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    proxies: Option<(wgpu::Buffer, wgpu::Buffer)>,
}

impl Scene {
    fn rays(&self) -> MeshRayScene<'_> {
        MeshRayScene {
            tlas: self.tlas.tlas().unwrap(),
            instances: &self.instances,
            vertices: &self.vertices,
            indices: &self.indices,
            proxies: self.proxies.as_ref().map(|(v, i)| (v, i)),
        }
    }
}

fn snorm8(n: Vec3) -> u32 {
    let b = |x: f32| ((x.clamp(-1.0, 1.0) * 127.0).round() as i8) as u8 as u32;
    b(n.x) | (b(n.y) << 8) | (b(n.z) << 16)
}

/// Row-major 3x4 of an affine `m`.
fn tlas_transform(m: Mat4) -> [f32; 12] {
    let r = m.transpose().to_cols_array();
    std::array::from_fn(|i| r[i])
}

fn scene(gpu: &Gpu, quads: &[Quad]) -> Scene {
    // `PackedVertex` rows: position, bitangent sign, two UVs, normal, tangent;
    // the proxy pool: position, UV.
    let mut vertices = Vec::<u32>::new();
    let mut indices = Vec::<u32>::new();
    let mut proxy_vertices = Vec::<f32>::new();
    let mut proxy_indices = Vec::<u32>::new();
    let mut rows = Vec::new();
    for q in quads {
        let uvs = [[0.0, 0.0], [q.uv, 0.0], [q.uv, q.uv], [0.0, q.uv]];
        let (first_index, base_vertex) = if q.proxy() {
            let at = (
                proxy_indices.len() as u32,
                (proxy_vertices.len() / 5) as u32,
            );
            for (p, [s, t]) in q.corners().into_iter().zip(uvs) {
                proxy_vertices.extend_from_slice(&[p.x, p.y, p.z, s, t]);
            }
            proxy_indices.extend_from_slice(&QUAD_INDICES);
            at
        } else {
            let at = (indices.len() as u32, (vertices.len() / 10) as u32);
            for (p, [s, t]) in q.corners().into_iter().zip(uvs) {
                vertices.extend_from_slice(&[
                    p.x.to_bits(),
                    p.y.to_bits(),
                    p.z.to_bits(),
                    1f32.to_bits(),
                    f32::to_bits(s),
                    f32::to_bits(t),
                    0,
                    0,
                    q.normal.map_or(0, snorm8),
                    0,
                ]);
            }
            indices.extend_from_slice(&QUAD_INDICES);
            at
        };
        rows.push(RayInstance {
            first_index,
            base_vertex,
            flags: q.flags,
            material: q.material,
            base_color: q.base_color.extend(1.0).to_array(),
            emissive: q.emissive.extend(0.0).to_array(),
            transmission: q.transmission.extend(0.0).to_array(),
        });
    }
    // Never-empty pools (a binding each).
    vertices.resize(vertices.len().max(10), 0);
    indices.resize(indices.len().max(4), 0);
    let usage =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::BLAS_INPUT | wgpu::BufferUsages::COPY_DST;
    let init = |label, contents: &[u8], usage| {
        gpu.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage,
            })
    };
    let vertex_buffer = init("mesh vertices", bytemuck::cast_slice(&vertices), usage);
    let index_buffer = init("mesh indices", bytemuck::cast_slice(&indices), usage);
    let proxies = (!proxy_indices.is_empty()).then(|| {
        (
            init(
                "proxy vertices",
                bytemuck::cast_slice(&proxy_vertices),
                usage,
            ),
            init("proxy indices", bytemuck::cast_slice(&proxy_indices), usage),
        )
    });
    let instances = init(
        "ray instances",
        bytemuck::cast_slice(&rows),
        wgpu::BufferUsages::STORAGE,
    );
    let mut blas = BlasManager::new(gpu.device.clone());
    let mut tlas = TlasManager::new(gpu.device.clone(), quads.len() as u32);
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    let mut inputs = Vec::new();
    for (k, (row, q)) in rows.iter().zip(quads).enumerate() {
        // Cut-outs and sheets are candidates, as the projections build them.
        let opaque = row.flags & RAY_INSTANCE_MASKED == 0 && q.transmission == Vec3::ZERO;
        let (vertices, indices, stride) = match &proxies {
            Some((v, i)) if q.proxy() => (v, i, 20),
            _ => (&vertex_buffer, &index_buffer, 40),
        };
        blas.build_segments(
            k as u64,
            &mut encoder,
            vec![BlasSegment {
                geometry: BlasGeometry {
                    revision: 0,
                    vertices,
                    first_vertex: row.base_vertex,
                    vertex_count: 4,
                    vertex_stride: stride,
                    indices: Some(indices),
                    first_index: row.first_index,
                    index_count: 6,
                },
                opaque,
            }],
        )
        .unwrap();
        inputs.push(TlasInstanceInput {
            geometry_id: k as u64,
            custom_data: k as u32,
            transform: tlas_transform(q.transform),
        });
    }
    let masks: Vec<u8> = quads.iter().map(|q| q.mask).collect();
    tlas.build_masked(&mut encoder, &inputs, &blas, &masks)
        .unwrap();
    gpu.queue.submit([encoder.finish()]);
    Scene {
        tlas,
        _blas: blas,
        instances,
        vertices: vertex_buffer,
        indices: index_buffer,
        proxies,
    }
}

/// A material table: `rows` (`GpuMaterial`s), the materials' stand-in
/// texture table.
struct Materials {
    rows: wgpu::Buffer,
    textures: Option<(wgpu::Buffer, wgpu::TextureView, wgpu::Sampler)>,
    stand_in: helio_mats::MaterialStandIn,
}

impl Materials {
    /// The texture table's views: the test's one texture in every slot.
    fn views(&self, gpu: &Gpu) -> Vec<&wgpu::TextureView> {
        let binding = helio_mats::MaterialBindingConfig::for_device(&gpu.device);
        match &self.textures {
            Some((_, view, _)) => vec![view; binding.max_textures],
            None => Vec::new(),
        }
    }

    fn bindings<'a>(
        &'a self,
        views: &'a [&'a wgpu::TextureView],
    ) -> (&'a wgpu::Buffer, helio_mats::MaterialTextureBindings<'a>) {
        let textures = match &self.textures {
            Some((metadata, _, sampler)) => helio_mats::MaterialTextureBindings {
                material_textures: metadata,
                texture_views: views,
                sampler,
                version: 0,
            },
            None => self.stand_in.textures(),
        };
        (&self.rows, textures)
    }
}

fn material(roughness: f32, metallic: f32) -> helio_mats::GpuMaterial {
    let none = helio_mats::GpuMaterial::NO_TEXTURE;
    helio_mats::GpuMaterial {
        base_color: [1.0; 4],
        emissive: [0.0; 4],
        roughness_metallic: [roughness, metallic, 1.5, 0.0],
        tex_base_color: none,
        tex_normal: none,
        tex_roughness: none,
        tex_emissive: none,
        tex_occlusion: none,
        workflow: 0,
        flags: 0,
        material_class: 0,
        class_params: [0.0; 4],
    }
}

fn materials(gpu: &Gpu, rows: &[helio_mats::GpuMaterial]) -> Materials {
    Materials {
        rows: gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("materials"),
                contents: bytemuck::cast_slice(rows),
                usage: wgpu::BufferUsages::STORAGE,
            }),
        textures: None,
        stand_in: helio_mats::MaterialStandIn::new(&gpu.device, &gpu.queue),
    }
}

/// A mipped RGBA8 texture whose alpha at mip 0 is `alpha(x, y)` over
/// `size` x `size` texels (white colour); its mips box-filtered.
fn alpha_texture(gpu: &Gpu, size: u32, alpha: impl Fn(u32, u32) -> u8) -> wgpu::Texture {
    let levels = size.ilog2() + 1;
    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("cut-out alpha"),
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
        mip_level_count: levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let mut level: Vec<f32> = (0..size * size)
        .map(|i| f32::from(alpha(i % size, i / size)) / 255.0)
        .collect();
    let mut side = size;
    for mip in 0..levels {
        let texels: Vec<u8> = level
            .iter()
            .flat_map(|a| [255, 255, 255, (a * 255.0).round() as u8])
            .collect();
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: mip,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &texels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * side),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: side,
                height: side,
                depth_or_array_layers: 1,
            },
        );
        if side == 1 {
            break;
        }
        let half = side / 2;
        level = (0..half * half)
            .map(|i| {
                let (x, y) = (2 * (i % half), 2 * (i / half));
                (level[(y * side + x) as usize]
                    + level[(y * side + x + 1) as usize]
                    + level[((y + 1) * side + x) as usize]
                    + level[((y + 1) * side + x + 1) as usize])
                    / 4.0
            })
            .collect();
        side = half;
    }
    texture
}

/// A material table whose row 0 is alpha-tested and double-sided, its base
/// colour texture `texture` (slot 0, repeating, linear), with the single-row
/// texture metadata whose `params.w = -1` takes the row's texture ids and
/// cutoff 0.5; row 1 opaque (roughness 0.25, metallic 1).
fn cut_out_materials(gpu: &Gpu, texture: &wgpu::Texture) -> Materials {
    let mut cut_out = material(0.75, 0.0);
    cut_out.base_color = [0.0, 0.0, 0.0, 1.0];
    cut_out.tex_base_color = 0;
    cut_out.flags = helio_mats::FLAG_ALPHA_TEST | helio_mats::FLAG_DOUBLE_SIDED;
    let mut table = materials(gpu, &[cut_out, material(0.25, 1.0)]);
    let mut metadata = [0u32; helio_mats::vg_alpha::MATERIAL_TEXTURE_WORDS as usize];
    metadata[metadata.len() - 1] = (-1f32).to_bits();
    let metadata = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("material texture metadata"),
            contents: bytemuck::cast_slice(&metadata),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        ..Default::default()
    });
    table.textures = Some((metadata, texture.create_view(&Default::default()), sampler));
    table
}

// ── Streams ─────────────────────────────────────────────────────────────────

fn ray(origin: Vec3, dir: Vec3, t_max: f32, spread: f32, mask: u8) -> SceneRay {
    SceneRay {
        origin: origin.to_array(),
        t_max,
        dir: dir.normalize().to_array(),
        cone: SceneRay::cone(0.0, spread, mask),
    }
}

fn read<T: bytemuck::Pod>(gpu: &Gpu, buffer: &wgpu::Buffer) -> Vec<T> {
    read_from(&gpu.device, &gpu.queue, buffer)
}

fn read_from<T: bytemuck::Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
) -> Vec<T> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: buffer.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
    queue.submit([encoder.finish()]);
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, |r| r.unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let words = staging.slice(..).get_mapped_range().unwrap();
    bytemuck::cast_slice(&words).to_vec()
}

/// The meshes' id in these tests' hits.
const MESHES: TracerId = TracerId(3);

/// What a stream's rows hold after the meshes traced it.
struct Traced {
    hits: Vec<SceneHit>,
    surfaces: Vec<SceneSurface>,
}

/// One frame of a stream of `class`: `rays` (the header counting `count`
/// of them), its hits starting as `hits` (cleared: none) and its surfaces
/// as `surfaces`, traced by the meshes of `scene` (none: a frame without a
/// TLAS).
#[allow(clippy::too_many_arguments)]
fn trace(
    gpu: &Gpu,
    class: RayClass,
    rays: &[SceneRay],
    count: u32,
    hits: Option<&[SceneHit]>,
    surfaces: Option<&[SceneSurface]>,
    scene: Option<&Scene>,
    materials: Option<&Materials>,
    frame: u64,
) -> Traced {
    let name = match class {
        RayClass::Nearest => "mesh_rays_test_nearest",
        RayClass::Occlusion => "mesh_rays_test_occlusion",
    };
    let stream = Stream::new(name, class, rays.len() as u32);
    let buffers = StreamBuffers::new(&gpu.device, &stream);
    gpu.queue
        .write_buffer(&buffers.rays, 0, bytemuck::cast_slice(rays));
    buffers.set_count(&gpu.queue, count);
    if let Some(hits) = hits {
        gpu.queue
            .write_buffer(&buffers.hits, 0, bytemuck::cast_slice(hits));
    }
    if let (Some(rows), Some(buffer)) = (surfaces, &buffers.surfaces) {
        gpu.queue
            .write_buffer(buffer, 0, bytemuck::cast_slice(rows));
    }
    let mut tracer = MeshRayTracer::new(&gpu.device, &gpu.queue, class, MESHES).unwrap();
    tracer.write_frame(frame, |buffer, bytes| {
        gpu.queue.write_buffer(buffer, 0, bytes)
    });
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    {
        let mut cmds = helio_core::CommandRecorder::from_encoder(&mut encoder);
        if hits.is_none() {
            buffers.clear(&mut cmds);
        }
        let views = materials.map(|m| m.views(gpu)).unwrap_or_default();
        let materials = materials.map(|m| m.bindings(&views));
        tracer.record(
            &gpu.device,
            &mut cmds,
            StreamBindings {
                header: &buffers.header,
                rays: &buffers.rays,
                hits: &buffers.hits,
                surfaces: buffers.surfaces.as_ref(),
            },
            scene.map(Scene::rays),
            materials,
        );
    }
    gpu.queue.submit([encoder.finish()]);
    Traced {
        hits: read(gpu, &buffers.hits),
        surfaces: buffers
            .surfaces
            .as_ref()
            .map(|s| read(gpu, s))
            .unwrap_or_default(),
    }
}

// ── The CPU reference ───────────────────────────────────────────────────────

/// A ray's nearest opaque hit on the CPU: its quad, triangle, distance and
/// barycentrics (of the second and third corners).
#[derive(Clone, Copy)]
struct Reference {
    quad: usize,
    primitive: u32,
    t: f32,
    bary: Vec2,
}

/// Möller-Trumbore, both faces.
fn intersect(o: Vec3, d: Vec3, [a, b, c]: [Vec3; 3]) -> Option<(f32, Vec2)> {
    let (e1, e2) = (b - a, c - a);
    let p = d.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1e-12 {
        return None;
    }
    let s = o - a;
    let u = s.dot(p) / det;
    let q = s.cross(e1);
    let v = d.dot(q) / det;
    let t = e2.dot(q) / det;
    (u >= 0.0 && v >= 0.0 && u + v <= 1.0 && t > 0.0).then_some((t, Vec2::new(u, v)))
}

fn triangle(q: &Quad, primitive: u32) -> [Vec3; 3] {
    let corners = q.corners();
    let i = &QUAD_INDICES[primitive as usize * 3..primitive as usize * 3 + 3];
    [0, 1, 2].map(|k| q.transform.transform_point3(corners[i[k] as usize]))
}

/// The ray's nearest hit, and whether it is unambiguous: no hit (of those
/// up to the nearest) within rounding of a triangle's edge, and none of
/// another triangle within rounding of the nearest's distance, where the GPU
/// may rightly take either.
fn nearest(quads: &[Quad], r: &SceneRay) -> (Option<Reference>, bool) {
    let (o, d) = (Vec3::from(r.origin), Vec3::from(r.dir));
    let mask = ((r.cone >> 16) & 0xff) as u8;
    let mut found = Vec::new();
    for (k, q) in quads.iter().enumerate() {
        if q.mask & mask == 0 {
            continue;
        }
        for primitive in 0..2 {
            if let Some((t, bary)) = intersect(o, d, triangle(q, primitive)) {
                if t <= r.t_max {
                    found.push(Reference {
                        quad: k,
                        primitive,
                        t,
                        bary,
                    });
                }
            }
        }
    }
    let Some(best) = found.iter().min_by(|a, b| a.t.total_cmp(&b.t)) else {
        return (None, true);
    };
    let tolerance = 1e-4 * best.t + 1e-5;
    let clear = found.iter().all(|h| {
        let margin = h.bary.x.min(h.bary.y).min(1.0 - h.bary.x - h.bary.y);
        let same = h.quad == best.quad && h.primitive == best.primitive;
        (h.t > best.t + tolerance || margin > 1e-4) && (same || h.t > best.t + tolerance)
    }) && r.t_max > best.t + tolerance;
    (Some(*best), clear)
}

/// The normals the surface must hold (shading, face, on the side the ray
/// met) and whether the ray met a single-sided surface's back.
fn expected_normals(q: &Quad, primitive: u32, dir: Vec3) -> (Vec3, Vec3, bool) {
    let [a, b, c] = triangle(q, primitive);
    let mut face = (b - a).cross(c - a).normalize();
    let linear = Mat3::from_mat4(q.transform);
    let local = if q.proxy() {
        q.u.cross(q.v).normalize()
    } else {
        q.normal.unwrap_or(Vec3::ZERO)
    };
    let mut shading = linear.inverse().transpose() * local;
    shading = if shading.length_squared() < 1e-8 {
        -dir
    } else {
        shading.normalize()
    };
    if face.dot(shading) < 0.0 {
        face = -face;
    }
    if q.flags & RAY_INSTANCE_TWO_SIDED != 0 && face.dot(dir) > 0.0 {
        face = -face;
        shading = -shading;
    }
    let inside = face.dot(dir) > 0.0;
    if inside {
        face = -face;
        shading = -shading;
    }
    let n = if shading.dot(dir) < 0.0 {
        shading
    } else {
        face
    };
    (n, face, inside)
}

fn angle(a: Vec3, b: Vec3) -> f32 {
    a.cross(b).length().atan2(a.dot(b))
}

/// `hit` and `surface` are the meshes' answer for `r`.
fn check_hit(
    quads: &[Quad],
    rows: &[helio_mats::GpuMaterial],
    k: usize,
    r: &SceneRay,
    hit: &SceneHit,
    surface: &SceneSurface,
) {
    let (want, clear) = nearest(quads, r);
    if !clear {
        return;
    }
    let Some(want) = want else {
        assert_eq!(*hit, SceneHit::default(), "ray {k}: no hit expected");
        return;
    };
    let q = &quads[want.quad];
    let dir = Vec3::from(r.dir);
    assert_eq!(hit.tracer(), MESHES, "ray {k}: {hit:?}");
    assert!(
        (hit.t - want.t).abs() <= 1e-4 * want.t + 1e-5,
        "ray {k}: t {} want {}",
        hit.t,
        want.t
    );
    assert_eq!(payload::row(hit), want.quad as u32, "ray {k}");
    assert_eq!(payload::primitive(hit), want.primitive, "ray {k}");
    let bary = Vec2::from(payload::barycentrics(hit));
    assert!(
        (bary - want.bary).abs().max_element() < 1e-3,
        "ray {k}: barycentrics {bary} want {}",
        want.bary
    );
    let (n, face, inside) = expected_normals(q, want.primitive, dir);
    assert_eq!(hit.inside(), inside, "ray {k}: inside");
    let got_n = Vec3::from(stream::unpack_unit(surface.normal));
    let got_face = Vec3::from(stream::unpack_unit(surface.face));
    assert!(angle(got_n, n) < 1e-3, "ray {k}: normal {got_n} want {n}");
    assert!(
        angle(got_face, face) < 1e-3,
        "ray {k}: face {got_face} want {face}"
    );
    let [r_, g, b, metallic] = surface.base_color();
    let want_rows = rows[q.material as usize];
    assert!(
        (Vec3::new(r_, g, b) - q.base_color).abs().max_element() <= 0.5 / 255.0 + 1e-6,
        "ray {k}: base colour {:?} want {}",
        [r_, g, b],
        q.base_color
    );
    assert!(
        (metallic - want_rows.roughness_metallic[1]).abs() <= 0.5 / 255.0 + 1e-6,
        "ray {k}: metallic {metallic}"
    );
    assert!(
        (surface.roughness() - want_rows.roughness_metallic[0]).abs() <= 0.5 / 255.0 + 1e-6,
        "ray {k}: roughness {}",
        surface.roughness()
    );
    assert_eq!(surface.transmission(), 0.0, "ray {k}");
    let emissive = Vec3::from(stream::unpack_radiance(surface.emissive));
    assert!(
        (emissive - q.emissive).abs().max_element() <= q.emissive.max_element() / 256.0 + 1e-6,
        "ray {k}: emission {emissive} want {}",
        q.emissive
    );
}

/// A deterministic stream of numbers in [0, 1).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }

    fn in_box(&mut self, half: f32) -> Vec3 {
        Vec3::new(self.next(), self.next(), self.next()) * (2.0 * half) - half
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

/// Quads placed by rotations, non-uniform scales and translations, some
/// single-sided and met from behind, one two-sided, one without vertex
/// normals; rays from all around: every hit is the CPU's nearest, and its
/// surface the instance's, with normals on the side the ray met.
#[test]
fn nearest_hits_and_surfaces_match_the_cpu() {
    let Some(gpu) = gpu() else { return };
    let rows = [material(0.5, 0.0), material(0.2, 1.0), material(0.9, 0.3)];
    let quads = vec![
        Quad {
            base_color: Vec3::new(0.8, 0.1, 0.1),
            transform: Mat4::from_scale_rotation_translation(
                Vec3::new(2.0, 0.5, 1.0),
                Quat::from_rotation_x(0.7),
                Vec3::new(0.0, 0.0, -1.5),
            ),
            ..Default::default()
        },
        Quad {
            base_color: Vec3::new(0.1, 0.7, 0.2),
            emissive: Vec3::new(3.0, 1.5, 0.25),
            material: 1,
            transform: Mat4::from_scale_rotation_translation(
                Vec3::new(0.7, 1.6, 1.0),
                Quat::from_rotation_y(1.2) * Quat::from_rotation_z(0.4),
                Vec3::new(1.2, 0.3, 0.5),
            ),
            ..Default::default()
        },
        Quad {
            base_color: Vec3::new(0.2, 0.3, 0.9),
            flags: RAY_INSTANCE_TWO_SIDED,
            material: 2,
            transform: Mat4::from_scale_rotation_translation(
                Vec3::new(1.0, 1.0, 3.0),
                Quat::from_rotation_y(-0.9),
                Vec3::new(-1.3, -0.2, 0.4),
            ),
            ..Default::default()
        },
        Quad {
            // No vertex normals: two-sided to its rays.
            normal: None,
            base_color: Vec3::splat(0.9),
            transform: Mat4::from_rotation_translation(
                Quat::from_rotation_x(-1.4),
                Vec3::new(0.2, -1.6, 0.0),
            ),
            ..Default::default()
        },
    ];
    let scene = scene(gpu, &quads);
    let table = materials(gpu, &rows);
    let mut lcg = Lcg(7);
    let rays: Vec<SceneRay> = (0..4096)
        .map(|_| {
            let origin = lcg.in_box(5.0);
            let target = lcg.in_box(1.8);
            ray(origin, target - origin, 50.0, 0.0, MASK_ALL)
        })
        .collect();
    let traced = trace(
        gpu,
        RayClass::Nearest,
        &rays,
        rays.len() as u32,
        None,
        None,
        Some(&scene),
        Some(&table),
        1,
    );
    let (mut hits, mut insides) = (0, 0);
    for (k, r) in rays.iter().enumerate() {
        check_hit(&quads, &rows, k, r, &traced.hits[k], &traced.surfaces[k]);
        hits += usize::from(traced.hits[k].t > 0.0);
        insides += usize::from(traced.hits[k].inside());
    }
    eprintln!("{hits} hits of {}, {insides} from inside", rays.len());
    assert!(
        hits > 1000 && insides > 100,
        "{hits} hits, {insides} inside"
    );
}

/// A quad 2 m down the rays: a row already holding a nearer hit (another
/// tracer's, 1 m) keeps it and its surface; one holding a farther hit (5 m)
/// takes the meshes'.
#[test]
fn the_nearer_of_the_row_s_hit_and_the_meshes_stands() {
    let Some(gpu) = gpu() else { return };
    let quads = [Quad {
        base_color: Vec3::new(0.25, 0.5, 0.75),
        ..Default::default()
    }];
    let scene = scene(gpu, &quads);
    let rows = [material(0.5, 0.0)];
    let table = materials(gpu, &rows);
    let rays: Vec<SceneRay> = (0..64)
        .map(|k| {
            let x = (k % 8) as f32 * 0.2 - 0.7;
            let y = (k / 8) as f32 * 0.2 - 0.7;
            ray(Vec3::new(x, y, 2.0), Vec3::NEG_Z, 100.0, 0.0, MASK_ALL)
        })
        .collect();
    let other = |t: f32| SceneHit {
        t,
        a: (7 << 28) | 12345,
        b: 9,
        c: 10,
    };
    let held: Vec<SceneHit> = (0..64)
        .map(|k| other(if k % 2 == 0 { 1.0 } else { 5.0 }))
        .collect();
    let sentinel = SceneSurface {
        normal: 1,
        face: 2,
        base_color: 3,
        params: 4,
        emissive: 5,
    };
    let traced = trace(
        gpu,
        RayClass::Nearest,
        &rays,
        64,
        Some(&held),
        Some(&vec![sentinel; 64]),
        Some(&scene),
        Some(&table),
        1,
    );
    for k in 0..64 {
        let (hit, surface) = (&traced.hits[k], &traced.surfaces[k]);
        if k % 2 == 0 {
            assert_eq!(*hit, held[k], "ray {k}: the nearer hit replaced");
            assert_eq!(
                *surface, sentinel,
                "ray {k}: the nearer hit's surface replaced"
            );
        } else {
            check_hit(&quads, &rows, k, &rays[k], hit, surface);
            assert!((hit.t - 2.0).abs() < 1e-4, "ray {k}: {hit:?}");
        }
    }
}

/// A static quad and a movable one behind it: a ray stops at neither past
/// its length, sees only the casters its mask holds, and a row with no ray
/// (length 0) or past the header's count is left as it was.
#[test]
fn length_masks_and_count_bound_what_a_ray_meets() {
    let Some(gpu) = gpu() else { return };
    let quads = [
        Quad {
            base_color: Vec3::new(1.0, 0.0, 0.0),
            mask: MASK_STATIC,
            ..Default::default()
        },
        Quad {
            base_color: Vec3::new(0.0, 1.0, 0.0),
            mask: MASK_MOVABLE,
            transform: Mat4::from_translation(Vec3::new(0.0, 0.0, -1.0)),
            ..Default::default()
        },
    ];
    let scene = scene(gpu, &quads);
    let rows = [material(0.5, 0.0)];
    let table = materials(gpu, &rows);
    let down = |t_max: f32, mask: u8| ray(Vec3::new(0.1, 0.2, 2.0), Vec3::NEG_Z, t_max, 0.0, mask);
    let rays = [
        down(1.5, MASK_ALL),      // short of both
        down(2.5, MASK_ALL),      // the static one at 2
        down(10.0, MASK_STATIC),  // the static one
        down(10.0, MASK_MOVABLE), // past the static one, the movable at 3
        down(2.5, MASK_MOVABLE),  // short of the movable one
        down(0.0, MASK_ALL),      // no ray
        down(10.0, MASK_ALL),     // past the count
        down(10.0, MASK_ALL),     // past the count
    ];
    let sentinel = SceneHit {
        t: 0.0,
        a: 0,
        b: 0xabcd,
        c: 0x1234,
    };
    let traced = trace(
        gpu,
        RayClass::Nearest,
        &rays,
        6,
        Some(&[sentinel; 8]),
        None,
        Some(&scene),
        Some(&table),
        1,
    );
    let t = |k: usize| traced.hits[k].t;
    assert_eq!(traced.hits[0], sentinel);
    assert!((t(1) - 2.0).abs() < 1e-4 && payload::row(&traced.hits[1]) == 0);
    assert!((t(2) - 2.0).abs() < 1e-4 && payload::row(&traced.hits[2]) == 0);
    assert!((t(3) - 3.0).abs() < 1e-4 && payload::row(&traced.hits[3]) == 1);
    assert_eq!(traced.hits[4], sentinel);
    assert_eq!(traced.hits[5], sentinel);
    assert_eq!(traced.hits[6], sentinel);
    assert_eq!(traced.hits[7], sentinel);
    for k in [1, 2, 3] {
        check_hit(
            &quads,
            &rows,
            k,
            &rays[k],
            &traced.hits[k],
            &traced.surfaces[k],
        );
    }
}

/// A frame without a TLAS: every row as it was.
#[test]
fn a_frame_without_a_tlas_leaves_the_rows() {
    let Some(gpu) = gpu() else { return };
    let rays = [ray(Vec3::Z, Vec3::NEG_Z, 10.0, 0.0, MASK_ALL); 4];
    let held = [SceneHit {
        t: 0.0,
        a: 0,
        b: 1,
        c: 2,
    }; 4];
    for class in [RayClass::Nearest, RayClass::Occlusion] {
        let traced = trace(gpu, class, &rays, 4, Some(&held), None, None, None, 1);
        assert_eq!(traced.hits, held, "{class:?}");
    }
}

/// A double-sided cut-out whose left half is leaves (alpha 1) and right
/// half holes (alpha 0) over an opaque floor: a fine ray stops on the
/// cut-out over the leaves only (with the cut-out's surface), and reaches
/// the floor through the holes. Coarse rays through a checker of leaves
/// and holes, whose far mips average to alpha 0.5, stop on about half
/// (stochastic past the finest mip), where a fixed cutoff would take every
/// one for leaves.
#[test]
fn cut_outs_stop_rays_only_where_their_alpha_passes() {
    let Some(gpu) = gpu() else { return };
    let cut_out = |uv: f32| Quad {
        flags: RAY_INSTANCE_MASKED | RAY_INSTANCE_TWO_SIDED,
        material: 0,
        uv,
        base_color: Vec3::new(0.0, 0.0, 0.0),
        ..Default::default()
    };
    let floor = Quad {
        corner: Vec3::new(-50.0, -50.0, 0.0),
        u: Vec3::X * 100.0,
        v: Vec3::Y * 100.0,
        material: 1,
        base_color: Vec3::splat(0.6),
        transform: Mat4::from_translation(Vec3::new(0.0, 0.0, -1.0)),
        ..Default::default()
    };
    // Fine rays: the left half of the texture leaves.
    let texture = alpha_texture(gpu, 64, |x, _| if x < 32 { 255 } else { 0 });
    let table = cut_out_materials(gpu, &texture);
    let rows = [material(0.75, 0.0), material(0.25, 1.0)];
    let quads = [cut_out(1.0), floor.clone()];
    let fine = scene(gpu, &quads);
    let rays: Vec<SceneRay> = (0..64)
        .map(|k| {
            // Clear of the halves' border (x 0) and of the quad's edges.
            let x = if k % 2 == 0 { -0.5 } else { 0.5 } + (k / 2) as f32 * 0.01;
            ray(Vec3::new(x, 0.3, 2.0), Vec3::NEG_Z, 100.0, 0.0, MASK_ALL)
        })
        .collect();
    let traced = trace(
        gpu,
        RayClass::Nearest,
        &rays,
        64,
        None,
        None,
        Some(&fine),
        Some(&table),
        1,
    );
    for (k, hit) in traced.hits.iter().enumerate() {
        let leaves = rays[k].origin[0] < 0.0;
        let (t, row) = if leaves { (2.0, 0) } else { (3.0, 1) };
        assert!(
            (hit.t - t).abs() < 1e-4 && payload::row(hit) == row,
            "ray {k} at x {}: {hit:?}",
            rays[k].origin[0]
        );
        let roughness = traced.surfaces[k].roughness();
        assert!((roughness - rows[row as usize].roughness_metallic[0]).abs() < 0.01);
    }
    // Coarse rays: a checker of 8-texel leaves and holes, tiled every 0.1 m,
    // seen by rays a metre wide at the cut-out.
    let texture = alpha_texture(
        gpu,
        64,
        |x, y| {
            if (x / 8 + y / 8) % 2 == 0 {
                255
            } else {
                0
            }
        },
    );
    let table = cut_out_materials(gpu, &texture);
    let quads = [cut_out(20.0), floor];
    let coarse = scene(gpu, &quads);
    let mut lcg = Lcg(3);
    let rays: Vec<SceneRay> = (0..4096)
        .map(|_| {
            let p = Vec3::new(lcg.next() * 1.6 - 0.8, lcg.next() * 1.6 - 0.8, 2.0);
            ray(p, Vec3::NEG_Z, 100.0, 0.5, MASK_ALL)
        })
        .collect();
    let stopped = |frame| {
        let traced = trace(
            gpu,
            RayClass::Nearest,
            &rays,
            rays.len() as u32,
            None,
            None,
            Some(&coarse),
            Some(&table),
            frame,
        );
        traced
            .hits
            .iter()
            .filter(|hit| payload::row(hit) == 0 && hit.t > 0.0)
            .count() as f32
            / rays.len() as f32
    };
    let (first, second) = (stopped(1), stopped(2));
    eprintln!("coarse rays stopped by the checker: {first} and {second} on two frames");
    assert!((first - 0.5).abs() < 0.1, "{first}");
    assert!((second - 0.5).abs() < 0.1, "{second}");
}

/// Occlusion rays through a tinted sheet, a sheet then an opaque quad, a
/// cut-out's holes and its leaves, a row another tracer tinted already, one
/// it blocked, and a ray whose mask sees none of them.
#[test]
fn occlusion_rows_multiply_what_the_meshes_let_through() {
    let Some(gpu) = gpu() else { return };
    let tint = Vec3::new(0.5, 0.25, 1.0);
    let at = |x: f32, z: f32, half: f32| {
        Mat4::from_translation(Vec3::new(x, 0.0, z)) * Mat4::from_scale(Vec3::new(half, 1.0, 1.0))
    };
    let texture = alpha_texture(gpu, 64, |x, _| if x < 32 { 255 } else { 0 });
    let table = cut_out_materials(gpu, &texture);
    let quads = [
        // The sheet over x in [-3, -1].
        Quad {
            transmission: tint,
            transform: at(-2.0, 0.0, 1.0),
            ..Default::default()
        },
        // An opaque quad under it, over x in [-1.75, -1.25].
        Quad {
            material: 1,
            transform: at(-1.5, -1.0, 0.25),
            ..Default::default()
        },
        // The cut-out over x in [1, 3]: leaves on [1, 2], holes on [2, 3].
        Quad {
            flags: RAY_INSTANCE_MASKED | RAY_INSTANCE_TWO_SIDED,
            transform: at(2.0, 0.0, 1.0),
            ..Default::default()
        },
    ];
    let scene = scene(gpu, &quads);
    let down = |x: f32, mask: u8| ray(Vec3::new(x, 0.1, 2.0), Vec3::NEG_Z, 10.0, 0.0, mask);
    let rays = [
        down(-2.5, MASK_ALL),
        down(-1.5, MASK_ALL),
        down(2.5, MASK_ALL),
        down(1.5, MASK_ALL),
        down(-2.5, MASK_ALL),
        down(-2.5, MASK_ALL),
        down(-2.5, MASK_MOVABLE),
    ];
    let clear = SceneHit::default();
    // Another tracer let half of each channel through.
    let half = SceneHit {
        t: 0.0,
        a: 0,
        b: 512 | (512 << 10) | (512 << 20),
        c: 0,
    };
    let blocked = SceneHit {
        t: 1.0,
        a: 0,
        b: 1023 | (1023 << 10) | (1023 << 20),
        c: 0,
    };
    let held = [clear, clear, clear, clear, half, blocked, clear];
    let traced = trace(
        gpu,
        RayClass::Occlusion,
        &rays,
        rays.len() as u32,
        Some(&held),
        None,
        Some(&scene),
        Some(&table),
        1,
    );
    let through = |k: usize| Vec3::from(traced.hits[k].through());
    let near = |k: usize, want: Vec3| {
        assert!(
            (through(k) - want).abs().max_element() < 2.0 / 1023.0,
            "ray {k}: {} want {want}",
            through(k)
        );
    };
    near(0, tint);
    assert!(
        traced.hits[0].t == 0.0,
        "the sheet blocked: {:?}",
        traced.hits[0]
    );
    assert!(traced.hits[1].t > 0.0, "{:?}", traced.hits[1]);
    near(1, Vec3::ZERO);
    assert_eq!(traced.hits[2], clear, "through the holes");
    assert!(
        traced.hits[3].t > 0.0,
        "the leaves let light by: {:?}",
        traced.hits[3]
    );
    near(4, tint * (1.0 - 512.0 / 1023.0));
    assert_eq!(traced.hits[5], blocked);
    assert_eq!(
        traced.hits[6], clear,
        "a movable-only ray met static casters"
    );
}

/// A virtual-geometry instance traced through its proxy (the proxy pool's
/// triangles, no vertex normals): its hits' normals are its face's, on the
/// side the ray met, and its back is an inside.
#[test]
fn a_proxy_instance_is_traced_by_its_face() {
    let Some(gpu) = gpu() else { return };
    let quads = [
        Quad {
            flags: RAY_INSTANCE_PROXY,
            base_color: Vec3::new(0.3, 0.6, 0.1),
            emissive: Vec3::new(0.0, 2.0, 0.0),
            transform: Mat4::from_scale_rotation_translation(
                Vec3::new(1.5, 0.5, 1.0),
                Quat::from_rotation_x(0.5),
                Vec3::new(0.0, 0.5, 0.0),
            ),
            ..Default::default()
        },
        // A plain mesh beside it: the two pools in one scene.
        Quad {
            transform: Mat4::from_translation(Vec3::new(4.0, 0.0, 0.0)),
            ..Default::default()
        },
    ];
    let scene = scene(gpu, &quads);
    let rows = [material(0.4, 0.0)];
    let table = materials(gpu, &rows);
    let mut lcg = Lcg(11);
    let rays: Vec<SceneRay> = (0..512)
        .map(|k| {
            let side = if k % 2 == 0 { 3.0 } else { -3.0 };
            let target = lcg.in_box(0.6) + Vec3::new(if k % 4 < 2 { 0.0 } else { 4.0 }, 0.5, 0.0);
            let origin = Vec3::new(target.x, target.y + 0.5, side);
            ray(origin, target - origin, 50.0, 0.0, MASK_ALL)
        })
        .collect();
    let traced = trace(
        gpu,
        RayClass::Nearest,
        &rays,
        rays.len() as u32,
        None,
        None,
        Some(&scene),
        Some(&table),
        1,
    );
    let mut proxy_hits = [0, 0];
    for (k, r) in rays.iter().enumerate() {
        check_hit(&quads, &rows, k, r, &traced.hits[k], &traced.surfaces[k]);
        if traced.hits[k].t > 0.0 && payload::row(&traced.hits[k]) == 0 {
            proxy_hits[usize::from(traced.hits[k].inside())] += 1;
        }
    }
    assert!(proxy_hits[0] > 30 && proxy_hits[1] > 30, "{proxy_hits:?}");
}

// ── In a graph ──────────────────────────────────────────────────────────────

mod graph {
    use super::*;
    use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};
    use helio_feature_api::{pass_node, Feature, FeatureRegistrar};
    use helio_frame_api::{FrameNode, Slot};
    use helio_mats::host_rows::MaterialComponent;
    use helio_mesh_host::ray_tracing::SceneDbRayTracing;
    use helio_mesh_host::{MeshComponent, StaticObjectComponent};

    /// The probe's rays: from the room's middle along the six axes.
    const EYE: Vec3 = Vec3::new(0.0, 1.0, 0.0);
    const DIRECTIONS: [Vec3; 6] = [
        Vec3::X,
        Vec3::NEG_X,
        Vec3::Y,
        Vec3::NEG_Y,
        Vec3::Z,
        Vec3::NEG_Z,
    ];
    /// Where each meets the room's inner faces.
    const DISTANCES: [f32; 6] = [5.0, 5.0, 3.0, 1.0, 5.0, 5.0];
    /// What the probe's rows hold before the tracers (no hit, words of
    /// its own).
    const HELD: SceneHit = SceneHit {
        t: 0.0,
        a: 0,
        b: 0x5eed,
        c: 0xf00d,
    };

    /// A consumer of a nearest-hit stream: its rays every frame, its rows
    /// set to [`HELD`], then every tracer's passes.
    struct Probe;

    struct ProbePass {
        stream: Stream,
        buffers: StreamBuffers,
        tracers: Vec<&'static str>,
        writes: &'static [&'static str],
    }

    impl Feature for Probe {
        fn id(&self) -> &'static str {
            "test.ray_probe"
        }

        fn register(&self, registrar: &mut FeatureRegistrar<'_>) {
            registrar.node(
                pass_node(
                    "ray_probe",
                    "Ray Probe",
                    "Passes: Test",
                    "A ray stream's consumer.",
                    |build| {
                        let stream = Stream::new("mesh_rays_probe", RayClass::Nearest, 6);
                        let mut writes = vec![stream.header, stream.rays, stream.hits];
                        writes.extend(stream.surfaces);
                        build.graph.add_pass(Box::new(ProbePass {
                            buffers: StreamBuffers::new(build.env.device, &stream),
                            tracers: stream::tracer_names(build.env),
                            writes: helio_frame_api::static_names(&writes),
                            stream,
                        }));
                        stream::trace(build, &stream)?;
                        Ok(())
                    },
                )
                .once_per_frame()
                .in_slot(Slot::Geometry, 50),
            );
        }
    }

    impl RenderPass for ProbePass {
        fn name(&self) -> &'static str {
            "RayProbe"
        }

        fn writes(&self) -> &'static [&'static str] {
            self.writes
        }

        fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
            let rays: Vec<SceneRay> = DIRECTIONS
                .iter()
                .map(|d| ray(EYE, *d, 100.0, 0.0, MASK_ALL))
                .collect();
            ctx.write_buffer(&self.buffers.rays, 0, bytemuck::cast_slice(&rays));
            ctx.write_buffer(
                &self.buffers.header,
                0,
                bytemuck::bytes_of(&stream::StreamHeader::of(6)),
            );
            ctx.write_buffer(&self.buffers.hits, 0, bytemuck::cast_slice(&[HELD; 6]));
            Ok(())
        }

        fn execute(&mut self, _ctx: &mut PassContext) -> HelioResult<()> {
            Ok(())
        }

        fn publish<'a>(&self, frame: &mut helio_core::ResourceRegistry<'a>) {
            self.buffers.publish(&self.stream, frame, "RayProbe");
        }
    }

    fn scene_db(device: &Arc<wgpu::Device>, queue: &Arc<wgpu::Queue>) -> pulsar_scenedb::SceneDb {
        let mut scene_db = pulsar_scenedb::SceneDb::new();
        let ctx = pulsar_scenedb::gpu::EngineGpuContext::new(device.clone(), queue.clone());
        let store = pulsar_scenedb::gpu::SceneGpuStore::new(
            &ctx,
            pulsar_scenedb::gpu::SceneGpuConfig {
                classes: Vec::new(),
                tombstone_headroom: 0,
                max_cells_metadata: 0,
            },
        );
        MeshComponent::register_gpu_columns_growable(&store, 64, device);
        MaterialComponent::register_gpu_columns_growable(&store, 64, device);
        StaticObjectComponent::register_gpu_columns_growable(&store, 64, device);
        scene_db
            .world
            .attach_gpu_mirror(pulsar_scenedb::gpu::GpuMirrorHandle::new(
                Arc::new(store),
                queue.clone(),
            ));
        scene_db
    }

    /// An axis-aligned box of half extent `half` at `centre` in `colour`,
    /// roughness 0.8.
    fn spawn_box(world: &mut pulsar_scenedb::World, centre: Vec3, half: Vec3, colour: [f32; 3]) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        for axis in 0..3 {
            for sign in [-1.0f32, 1.0] {
                let n =
                    Vec3::from_array(std::array::from_fn(|k| if k == axis { sign } else { 0.0 }));
                let (u, v) = (
                    Vec3::from_array(std::array::from_fn(|k| f32::from(k == (axis + 1) % 3))),
                    Vec3::from_array(std::array::from_fn(|k| f32::from(k == (axis + 2) % 3))),
                );
                let (u, v) = if sign > 0.0 { (u, v) } else { (v, u) };
                let base = vertices.len() as u32;
                for (a, b) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                    let p = (n + u * a + v * b) * half;
                    vertices.push(helio_mesh_api::PackedVertex::from_components(
                        p.to_array(),
                        n.to_array(),
                        [0.0; 2],
                        u.to_array(),
                        1.0,
                    ));
                }
                indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
            }
        }
        let mesh = world.spawn();
        world.insert(mesh, MeshComponent { vertices, indices });
        let material = world.spawn();
        let row = MaterialComponent::from(helio_mats::MaterialRow::new(
            [colour[0], colour[1], colour[2], 1.0],
            0.8,
            0.0,
            [0.0; 3],
            0.0,
        ));
        world.insert(material, row);
        let store = world.gpu_mirror().cloned().unwrap();
        let vertex_range = MeshComponent::vertices_gpu_handle(store.store(), mesh.index()).unwrap();
        let index_range = MeshComponent::indices_gpu_handle(store.store(), mesh.index()).unwrap();
        let object = StaticObjectComponent::from(helio_mesh_api::StaticObjectRow::new(
            mesh.index(),
            mesh.generation().wrapping_add(1),
            material.index(),
            material.generation().wrapping_add(1),
            Mat4::from_translation(centre),
            [centre.x, centre.y, centre.z, half.length()],
            index_range.count,
            index_range.offset,
            vertex_range.offset as i32,
            row.material_class,
            0,
            0,
        ));
        let entity = world.spawn();
        world.insert(entity, object);
    }

    /// A closed room: floor (top at 0), ceiling (bottom at 4), walls at
    /// +-5 (red at -x).
    fn room(world: &mut pulsar_scenedb::World) {
        let white = [0.7; 3];
        spawn_box(
            world,
            Vec3::new(0.0, -0.25, 0.0),
            Vec3::new(5.5, 0.25, 5.5),
            white,
        );
        spawn_box(
            world,
            Vec3::new(0.0, 4.25, 0.0),
            Vec3::new(5.5, 0.25, 5.5),
            white,
        );
        spawn_box(
            world,
            Vec3::new(-5.25, 2.0, 0.0),
            Vec3::new(0.25, 2.5, 5.5),
            [0.7, 0.1, 0.1],
        );
        spawn_box(
            world,
            Vec3::new(5.25, 2.0, 0.0),
            Vec3::new(0.25, 2.5, 5.5),
            white,
        );
        spawn_box(
            world,
            Vec3::new(0.0, 2.0, -5.25),
            Vec3::new(5.5, 2.5, 0.25),
            white,
        );
        spawn_box(
            world,
            Vec3::new(0.0, 2.0, 5.25),
            Vec3::new(5.5, 2.5, 0.25),
            white,
        );
    }

    /// The probe's stream in the deferred graph of a mesh scene: the meshes
    /// answer it (registered by their feature set), leave its rows as they
    /// were on frames without a TLAS, and on frames with one write each
    /// ray's room face with its surface (the material table's roughness).
    #[test]
    fn a_stream_in_a_mesh_scene_s_graph_is_answered_by_the_meshes() {
        let instance = wgpu::Instance::new(helio_core::wgpu_instance_descriptor());
        let Ok(adapter) =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            }))
        else {
            return;
        };
        let features = helio::required_wgpu_features(helio::usable_adapter_features(&adapter));
        if !features.contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY) {
            eprintln!("no ray queries: skipped");
            return;
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: features,
            required_limits: helio::required_wgpu_limits(adapter.limits()),
            experimental_features: helio::required_experimental_features(features),
            ..Default::default()
        }))
        .unwrap();
        let (device, queue) = (Arc::new(device), Arc::new(queue));
        let errors = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut scene = scene_db(&device, &queue);
        room(&mut scene.world);
        let config = helio::RendererConfig::new(64, 64, wgpu::TextureFormat::Rgba8Unorm);
        let mut renderer =
            helio::RendererBuilder::new(config.clone(), scene.world.gpu_mirror().cloned().unwrap())
                .with_graph(helio_mesh_host::presets::with(
                    helio_presets::Preset::Deferred,
                    &helio_feature_api::FeatureSet::new().with(Probe),
                ))
                .with_external_device()
                .build(
                    device.clone(),
                    queue.clone(),
                    config.width,
                    config.height,
                    config.surface_format,
                );
        let probe = renderer
            .find_pass::<ProbePass>()
            .expect("no probe in the graph");
        let meshes = probe
            .tracers
            .iter()
            .position(|name| *name == "meshes")
            .unwrap_or_else(|| panic!("the meshes do not trace: {:?}", probe.tracers));
        let id = TracerId(meshes as u32 + 1);
        assert!(
            renderer
                .find_pass::<helio_pass_mesh_rays::MeshRayPass>()
                .is_some(),
            "no mesh tracer in the graph"
        );
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 64,
                height: 64,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: config.surface_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = target.create_view(&Default::default());
        let camera = helio::Camera::perspective_look_at(
            Vec3::new(-2.0, 1.7, 3.0),
            Vec3::new(1.0, 1.0, -1.0),
            Vec3::Y,
            1.0,
            1.0,
            0.1,
            100.0,
        );
        let rows = |renderer: &helio::Renderer| {
            let probe = renderer.find_pass::<ProbePass>().unwrap();
            (
                read_from::<SceneHit>(&device, &queue, &probe.buffers.hits),
                read_from::<SceneSurface>(
                    &device,
                    &queue,
                    probe.buffers.surfaces.as_ref().unwrap(),
                ),
            )
        };
        // No TLAS published: the rows as the probe left them.
        for _ in 0..2 {
            scene.world.flush_gpu_mirror(&queue);
            renderer.render(&camera, &view).unwrap();
        }
        assert_eq!(rows(&renderer).0, [HELD; 6], "rows changed without a TLAS");
        let mut rays = SceneDbRayTracing::partial(device.clone(), queue.clone());
        for _ in 0..3 {
            scene.world.flush_gpu_mirror(&queue);
            rays.prepare(&scene.world).unwrap();
            rays.publish(&mut renderer);
            renderer.render(&camera, &view).unwrap();
        }
        let (hits, surfaces) = rows(&renderer);
        for (k, (hit, surface)) in hits.iter().zip(&surfaces).enumerate() {
            assert_eq!(hit.tracer(), id, "ray {k}: {hit:?}");
            assert!(
                (hit.t - DISTANCES[k]).abs() < 1e-3,
                "ray {k}: t {} want {}",
                hit.t,
                DISTANCES[k]
            );
            assert!(!hit.inside(), "ray {k}");
            let n = Vec3::from(stream::unpack_unit(surface.normal));
            assert!(angle(n, -DIRECTIONS[k]) < 1e-3, "ray {k}: normal {n}");
            assert!(
                (surface.roughness() - 0.8).abs() < 0.01,
                "ray {k}: {surface:?}"
            );
            let colour = surface.base_color();
            let want = if k == 1 { [0.7, 0.1, 0.1] } else { [0.7; 3] };
            for c in 0..3 {
                assert!((colour[c] - want[c]).abs() < 0.01, "ray {k}: {colour:?}");
            }
        }
        let error = pollster::block_on(errors.pop());
        assert!(error.is_none(), "{error:?}");
    }
}
