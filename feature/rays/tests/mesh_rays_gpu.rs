//! The meshes' tracer of ray streams against what a stream's rows must hold:
//! hits and surfaces compared with a CPU ray-triangle reference (distance,
//! tracer, inside, normals, the material's colours, payload), the surfaces
//! binned by material program (a template's surface, a program not compiled
//! yet, programs past the bound), textures read at the ray cone's
//! footprint, the nearer of a row's hit and
//! the meshes', a ray's length and caster mask, cut-outs (exact at a fine
//! footprint, in proportion to coverage at a coarse one), occlusion through
//! sheets, cut-outs and opaque meshes, a proxy (virtual-geometry) instance,
//! rows past the header's count, a frame without a TLAS, and coordinate
//! spaces (overlapping placements in different spaces, a group each and
//! shared groups). The last test
//! traces a stream in a mesh scene's graph, its TLAS from the standalone
//! projection. Needs hardware ray queries: without them each test returns
//! at once.

use std::sync::Arc;

use glam::{Mat3, Mat4, Quat, Vec2, Vec3};
use helio_core::{BlasGeometry, BlasManager, BlasSegment, TlasInstanceInput, TlasManager};
use helio_frame_api::rays::stream::SurfaceDetail;
use helio_frame_api::rays::stream::{
    self, space_of_slot, RayClass, SceneHit, SceneRay, SceneSurface, Stream, StreamBuffers,
    TracerId, MASK_ALL, MASK_MOVABLE, MASK_STATIC, SPACE_ANY,
};
use helio_mesh_api::{
    RayInstance, RayProgram, RayPrograms, RayTransform, RAY_INSTANCE_MASKED, RAY_INSTANCE_PROXY,
    RAY_INSTANCE_TWO_SIDED, RAY_SPACE_GROUP,
};
use helio_pass_mesh_rays::surfaces::MeshHitScene;
use helio_pass_mesh_rays::{
    payload, HitPrograms, MeshHitSurfaces, MeshRayScene, MeshRayTracer, StreamBindings,
};
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
/// zero normals), its material's base colour and emission ([`quad_rows`]
/// gives each quad a material row of its own, its test row `material` with
/// these colours), UVs from 0 to `uv` across it, its row's `RAY_INSTANCE_*`
/// flags and program, its transmission (a sheet), its object-to-frame
/// transform, caster class (`MASK_STATIC` or `MASK_MOVABLE`) and coordinate
/// space slot. A proxy's triangles go in the proxy pool.
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
    program: RayProgram,
    transmission: Vec3,
    transform: Mat4,
    mask: u8,
    space: u32,
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
            program: RayProgram::TABLE,
            transmission: Vec3::ZERO,
            transform: Mat4::IDENTITY,
            mask: MASK_STATIC,
            space: 0,
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
    transforms: wgpu::Buffer,
    space_groups: Option<wgpu::Buffer>,
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    proxies: Option<(wgpu::Buffer, wgpu::Buffer)>,
    programs: RayPrograms,
}

impl Scene {
    fn rays(&self) -> MeshRayScene<'_> {
        MeshRayScene {
            tlas: self.tlas.tlas().unwrap(),
            instances: &self.instances,
            space_groups: self.space_groups.as_ref(),
            vertices: &self.vertices,
            indices: &self.indices,
            proxies: self.proxies.as_ref().map(|(v, i)| (v, i)),
            skinned: None,
        }
    }

    fn hits(&self) -> MeshHitScene<'_> {
        MeshHitScene {
            instances: &self.instances,
            transforms: &self.transforms,
            vertices: &self.vertices,
            indices: &self.indices,
            proxies: self.proxies.as_ref().map(|(v, i)| (v, i)),
            skinned: None,
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

/// Words of a mesh pool vertex.
const VERTEX_WORDS: usize = std::mem::size_of::<helio_mesh_api::PackedVertex>() / 4;

/// A mesh pool vertex with a positive bitangent sign and these packed normal
/// and tangent words, as the pool holds it.
fn vertex_words(position: Vec3, uv: [f32; 2], normal: u32, tangent: u32) -> [u32; VERTEX_WORDS] {
    let mut vertex = helio_mesh_api::PackedVertex {
        position: position.to_array(),
        tex_coords0: uv,
        normal,
        tangent,
        ..Default::default()
    };
    vertex.set_bitangent_sign(1.0);
    bytemuck::cast(vertex)
}

fn scene(gpu: &Gpu, quads: &[Quad]) -> Scene {
    // The mesh pool in words; the proxy pool: position, UV.
    let mut vertices = Vec::<u32>::new();
    let mut indices = Vec::<u32>::new();
    let mut proxy_vertices = Vec::<f32>::new();
    let mut proxy_indices = Vec::<u32>::new();
    let mut rows = Vec::new();
    let mut programs = RayPrograms::default();
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
            let at = (indices.len() as u32, (vertices.len() / VERTEX_WORDS) as u32);
            for (p, [s, t]) in q.corners().into_iter().zip(uvs) {
                vertices.extend_from_slice(&vertex_words(
                    p,
                    [s, t],
                    q.normal.map_or(0, snorm8),
                    snorm8(q.u.normalize()),
                ));
            }
            indices.extend_from_slice(&QUAD_INDICES);
            at
        };
        rows.push(
            RayInstance {
                first_index,
                base_vertex,
                flags: q.flags,
                // Each quad's own material row (`quad_rows`).
                material: rows.len() as u32,
                transmission: q.transmission.to_array(),
                instance: rows.len() as u32,
                skinned_vertex: 0,
                program: programs.index(q.program),
                _pad: [0; 2],
            }
            .in_space(q.space),
        );
    }
    // Never-empty pools (a binding each).
    vertices.resize(vertices.len().max(VERTEX_WORDS), 0);
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
    // As a projection publishes them: each instance's transform, and the
    // spaces' mask groups where any instance is outside the world's space.
    let transforms: Vec<RayTransform> = quads
        .iter()
        .map(|q| RayTransform::from_rows(tlas_transform(q.transform)))
        .collect();
    let transforms = init(
        "ray instance transforms",
        bytemuck::cast_slice(&transforms),
        wgpu::BufferUsages::STORAGE,
    );
    let groups = helio_mesh_api::ray_space_groups(quads.iter().map(|q| q.space));
    let space_groups = quads.iter().any(|q| q.space != 0).then(|| {
        init(
            "ray space groups",
            bytemuck::cast_slice(&groups),
            wgpu::BufferUsages::STORAGE,
        )
    });
    let mut blas = BlasManager::new(gpu.device.clone());
    let mut tlas = TlasManager::new(gpu.device.clone(), quads.len() as u32);
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    let mut inputs = Vec::new();
    for (k, (row, q)) in rows.iter().zip(quads).enumerate() {
        // Cut-outs and sheets are candidates, as the projections build them.
        let opaque = row.flags & RAY_INSTANCE_MASKED == 0 && q.transmission == Vec3::ZERO;
        let (vertices, indices, stride) = match &proxies {
            Some((v, i)) if q.proxy() => (v, i, 20),
            _ => (
                &vertex_buffer,
                &index_buffer,
                helio_mesh_api::PackedVertex::STRIDE,
            ),
        };
        blas.build(
            k as u64,
            &mut encoder,
            vec![BlasSegment {
                geometry: BlasGeometry {
                    positions_revision: 0,
                    topology_revision: 0,
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
            helio_core::Movability::Static,
        )
        .unwrap();
        inputs.push(TlasInstanceInput {
            geometry_id: k as u64,
            custom_data: k as u32,
            transform: tlas_transform(q.transform),
        });
    }
    let masks: Vec<u8> = quads
        .iter()
        .map(|q| {
            helio_mesh_api::ray_instance_mask(
                groups[q.space as usize] & RAY_SPACE_GROUP,
                q.mask == MASK_MOVABLE,
            )
        })
        .collect();
    tlas.build_masked(&mut encoder, &inputs, &blas, &masks)
        .unwrap();
    gpu.queue.submit([encoder.finish()]);
    Scene {
        tlas,
        _blas: blas,
        instances,
        transforms,
        space_groups,
        vertices: vertex_buffer,
        indices: index_buffer,
        proxies,
        programs,
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

/// A material row a quad: quad `k`'s is its test row (`rows[q.material]`)
/// with the quad's base colour and emission (strength 1).
fn quad_rows(quads: &[Quad], rows: &[helio_mats::GpuMaterial]) -> Vec<helio_mats::GpuMaterial> {
    quads
        .iter()
        .map(|q| {
            let mut row = rows[q.material as usize];
            row.base_color = q.base_color.extend(row.base_color[3]).to_array();
            row.emissive = q.emissive.extend(1.0).to_array();
            row
        })
        .collect()
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

/// The test rows of [`cut_out_materials`]: row 0 alpha-tested and
/// double-sided, its base colour texture slot 0; row 1 opaque (roughness
/// 0.25, metallic 1).
fn cut_out_rows() -> [helio_mats::GpuMaterial; 2] {
    let mut cut_out = material(0.75, 0.0);
    cut_out.base_color = [0.0, 0.0, 0.0, 1.0];
    cut_out.tex_base_color = 0;
    cut_out.flags = helio_mats::FLAG_ALPHA_TEST | helio_mats::FLAG_DOUBLE_SIDED;
    [cut_out, material(0.25, 1.0)]
}

/// A material table of `quads` over [`cut_out_rows`], the base colour
/// texture `texture` (slot 0, repeating, linear), with the single-row
/// texture metadata whose `params.w = -1` takes the row's texture ids and
/// cutoff 0.5.
fn cut_out_materials(gpu: &Gpu, quads: &[Quad], texture: &wgpu::Texture) -> Materials {
    textured_materials(gpu, &quad_rows(quads, &cut_out_rows()), texture)
}

/// A material table of `rows` whose texture slots all hold `texture`
/// (repeating, trilinear), with the single-row texture metadata whose
/// `params.w = -1` takes each row's texture ids with identity UVs.
fn textured_materials(
    gpu: &Gpu,
    rows: &[helio_mats::GpuMaterial],
    texture: &wgpu::Texture,
) -> Materials {
    let mut table = materials(gpu, rows);
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
    SceneRay::new(
        origin.to_array(),
        t_max,
        dir.normalize().to_array(),
        SceneRay::cone(0.0, spread, mask),
    )
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
        let bindings = StreamBindings {
            header: &buffers.header,
            rays: &buffers.rays,
            hits: &buffers.hits,
            surfaces: buffers.surfaces.as_ref(),
        };
        tracer.record(
            &gpu.device,
            &mut cmds,
            bindings,
            scene.map(Scene::rays),
            materials,
        );
        // The surfaces of the hits that are the meshes', as `stream::trace`
        // adds the tracer's surface pass after every trace.
        surfaces_of(
            gpu,
            class,
            frame,
            &mut cmds,
            bindings,
            scene.map(Scene::hits),
            materials,
            scene.map(|s| &s.programs),
            SurfaceDetail::Full,
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

/// The hit programs of these tests' device (one set, as a graph shares
/// them).
fn hit_programs(gpu: &Gpu) -> Arc<HitPrograms> {
    static PROGRAMS: std::sync::OnceLock<Arc<HitPrograms>> = std::sync::OnceLock::new();
    PROGRAMS
        .get_or_init(|| Arc::new(HitPrograms::new(&gpu.device, &gpu.queue, None)))
        .clone()
}

/// Records the meshes' surface passes of a stream of `class` at `detail`
/// (nothing for an occlusion stream), its rows naming `programs` (every
/// program compiled first: a test has no frames to wait over).
#[allow(clippy::too_many_arguments)]
fn surfaces_of(
    gpu: &Gpu,
    class: RayClass,
    frame: u64,
    cmds: &mut helio_core::CommandRecorder<'_>,
    bindings: StreamBindings<'_>,
    scene: Option<MeshHitScene<'_>>,
    materials: Option<(&wgpu::Buffer, helio_mats::MaterialTextureBindings<'_>)>,
    programs: Option<&RayPrograms>,
    detail: SurfaceDetail,
) -> usize {
    if class != RayClass::Nearest {
        return 0;
    }
    let shared = hit_programs(gpu);
    let mut surfaces = MeshHitSurfaces::new(&gpu.device, shared.clone(), detail, MESHES);
    let write = |buffer: &wgpu::Buffer, offset: u64, bytes: &[u8]| {
        gpu.queue.write_buffer(buffer, offset, bytes)
    };
    // Ask once (starting the compiles), wait for them, ask again.
    surfaces.set_programs(&gpu.device, programs, write);
    shared.wait();
    surfaces.set_programs(&gpu.device, programs, write);
    surfaces.write_frame(frame, |buffer, bytes| {
        gpu.queue.write_buffer(buffer, 0, bytes)
    });
    surfaces.record(&gpu.device, cmds, bindings, scene, materials);
    surfaces.bin_count()
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
    let mask = r.mask();
    let mut found = Vec::new();
    for (k, q) in quads.iter().enumerate() {
        if q.mask & mask == 0 || !r.sees_space(q.space) {
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
    assert_eq!(surface.transmission(), [0.0; 3], "ray {k}");
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
    let table = materials(gpu, &quad_rows(&quads, &rows));
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
    let table = materials(gpu, &quad_rows(&quads, &rows));
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
        specular: 6,
        subsurface: 7,
        anisotropy: 8,
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
    let table = materials(gpu, &quad_rows(&quads, &rows));
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
    let rows = cut_out_rows();
    let quads = [cut_out(1.0), floor.clone()];
    let table = cut_out_materials(gpu, &quads, &texture);
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
    let quads = [cut_out(20.0), floor];
    let table = cut_out_materials(gpu, &quads, &texture);
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
    let table = cut_out_materials(gpu, &quads, &texture);
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
    let table = materials(gpu, &quad_rows(&quads, &rows));
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

/// A skinned instance is traced from the skin cache (`RAY_INSTANCE_SKINNED`):
/// its hit is where its pose put it and its surface's normal the skinned
/// one, not the rest mesh's; a new pose refits its deforming BLAS
/// (`Movability::Dynamic`) and the next stream sees it; an unchanged pose
/// records nothing; the policy's refit count makes the next change a
/// rebuild.
#[test]
fn a_skinned_instance_is_traced_where_its_pose_puts_it() {
    let Some(gpu) = gpu() else { return };
    // The rest mesh: a 2 m quad at z = 0 facing +z (UVs and normal from the
    // mesh pool), its corners in the skin cache at poses below.
    let rest = [
        Vec3::new(-1.0, -1.0, 0.0),
        Vec3::new(1.0, -1.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(-1.0, 1.0, 0.0),
    ];
    let mut vertices = Vec::<u32>::new();
    for (p, [s, t]) in rest
        .iter()
        .zip([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0f32]])
    {
        vertices.extend_from_slice(&vertex_words(*p, [s, t], snorm8(Vec3::Z), 0));
    }
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
    let vertex_pool = init("mesh vertices", bytemuck::cast_slice(&vertices), usage);
    let index_pool = init("mesh indices", bytemuck::cast_slice(&QUAD_INDICES), usage);
    // A pose: the rest quad lifted to `height` and tilted about y by `tilt`.
    let pose = |height: f32, tilt: f32| -> Vec<helio_mesh_api::SkinnedVertex> {
        let m = Mat4::from_translation(Vec3::Z * height) * Mat4::from_rotation_y(tilt);
        let n = m.transform_vector3(Vec3::Z).normalize();
        rest.iter()
            .map(|&p| helio_mesh_api::SkinnedVertex {
                position: m.transform_point3(p).to_array(),
                normal: stream::pack_unit(n.to_array()),
                previous: p.to_array(),
                tangent: 0,
            })
            .collect()
    };
    let skinned = init(
        "skin cache",
        bytemuck::cast_slice(&pose(1.0, 0.0)),
        usage | wgpu::BufferUsages::COPY_SRC,
    );
    let rows = [RayInstance {
        first_index: 0,
        base_vertex: 0,
        flags: helio_mesh_api::RAY_INSTANCE_SKINNED,
        material: 0,
        transmission: [0.0; 3],
        instance: 0,
        skinned_vertex: 0,
        program: 0,
        _pad: [0; 2],
    }];
    let instances = init(
        "ray instances",
        bytemuck::cast_slice(&rows),
        wgpu::BufferUsages::STORAGE,
    );
    let transforms = init(
        "ray instance transforms",
        bytemuck::bytes_of(&RayTransform::from_rows(tlas_transform(Mat4::IDENTITY))),
        wgpu::BufferUsages::STORAGE,
    );
    let mut blas = BlasManager::new(gpu.device.clone());
    blas.set_policy(helio_core::BlasPolicy {
        refits_per_rebuild: 2,
        ..Default::default()
    });
    let mut tlas = TlasManager::new(gpu.device.clone(), 1);
    let instance = [TlasInstanceInput {
        geometry_id: 0,
        custom_data: 0,
        transform: tlas_transform(Mat4::IDENTITY),
    }];
    // Build (or refit) the BLAS at `revision` from the skin cache, then the
    // TLAS; what the BLAS manager recorded.
    let build = |blas: &mut BlasManager, tlas: &mut TlasManager, revision: u64| {
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        let work = blas
            .build(
                0,
                &mut encoder,
                vec![BlasSegment {
                    geometry: BlasGeometry {
                        positions_revision: revision,
                        topology_revision: 0,
                        vertices: &skinned,
                        first_vertex: 0,
                        vertex_count: 4,
                        vertex_stride: std::mem::size_of::<helio_mesh_api::SkinnedVertex>() as u64,
                        indices: Some(&index_pool),
                        first_index: 0,
                        index_count: 6,
                    },
                    opaque: true,
                }],
                helio_core::Movability::Dynamic,
            )
            .unwrap();
        tlas.build_masked(&mut encoder, &instance, blas, &[MASK_MOVABLE])
            .unwrap();
        gpu.queue.submit([encoder.finish()]);
        (work, blas.counters())
    };
    let table = materials(gpu, &[material(0.5, 0.0)]);
    let rays: Vec<SceneRay> = (0..64)
        .map(|k| {
            let x = (k % 8) as f32 / 8.0 - 0.45;
            let y = (k / 8) as f32 / 8.0 - 0.45;
            ray(Vec3::new(x, y, 10.0), -Vec3::Z, 50.0, 0.0, MASK_ALL)
        })
        .collect();
    let shoot = |tlas: &TlasManager| {
        let scene = MeshRayScene {
            tlas: tlas.tlas().unwrap(),
            instances: &instances,
            space_groups: None,
            vertices: &vertex_pool,
            indices: &index_pool,
            proxies: None,
            skinned: Some(&skinned),
        };
        let hits = MeshHitScene {
            instances: &instances,
            transforms: &transforms,
            vertices: &vertex_pool,
            indices: &index_pool,
            proxies: None,
            skinned: Some(&skinned),
        };
        trace_scene(gpu, &rays, scene, hits, &table)
    };
    // Expect every ray to hit the plane through height `h` at the centre
    // tilted by `tilt` (x within half a metre: the tilt moves it by tan).
    let check = |traced: &Traced, height: f32, tilt: f32, what: &str| {
        let n = Vec3::new(tilt.sin(), 0.0, tilt.cos());
        for (k, r) in rays.iter().enumerate() {
            let hit = &traced.hits[k];
            let o = Vec3::from(r.origin);
            // The plane n.(p - h z) = 0 along -z from o.
            let t = (n.dot(o) - height * n.z) / n.z;
            assert_eq!(hit.tracer(), MESHES, "{what}, ray {k}: {hit:?}");
            assert!(
                (hit.t - t).abs() < 1e-3,
                "{what}, ray {k}: t {} want {t}",
                hit.t
            );
            let got = Vec3::from(stream::unpack_unit(traced.surfaces[k].normal));
            assert!(
                angle(got, n) < 1e-3,
                "{what}, ray {k}: normal {got} want {n}"
            );
        }
    };

    let (work, first) = build(&mut blas, &mut tlas, 1);
    assert_eq!(work, helio_core::BlasWork::Built);
    check(&shoot(&tlas), 1.0, 0.0, "first pose");

    // A new pose in the cache: a refit, and the rays see it.
    gpu.queue
        .write_buffer(&skinned, 0, bytemuck::cast_slice(&pose(2.0, 0.4)));
    let (work, refit) = build(&mut blas, &mut tlas, 2);
    assert_eq!(work, helio_core::BlasWork::Refit);
    assert_eq!(refit.refits, first.refits + 1);
    assert_eq!(refit.builds, first.builds);
    check(&shoot(&tlas), 2.0, 0.4, "second pose");

    // The same pose: nothing recorded.
    let (work, unchanged) = build(&mut blas, &mut tlas, 2);
    assert_eq!(work, helio_core::BlasWork::Unchanged);
    assert_eq!(unchanged, refit);

    // The policy's count (2 refits a rebuild): the next refit, then a
    // rebuild.
    let (work, _) = build(&mut blas, &mut tlas, 3);
    assert_eq!(work, helio_core::BlasWork::Refit);
    gpu.queue
        .write_buffer(&skinned, 0, bytemuck::cast_slice(&pose(3.0, -0.3)));
    let (work, rebuilt) = build(&mut blas, &mut tlas, 4);
    assert_eq!(work, helio_core::BlasWork::Built);
    assert_eq!(rebuilt.policy_rebuilds, 1);
    check(&shoot(&tlas), 3.0, -0.3, "after the rebuild");
}

/// [`trace`] of a nearest-hit stream against `scene` itself (a TLAS the test
/// built).
fn trace_scene(
    gpu: &Gpu,
    rays: &[SceneRay],
    scene: MeshRayScene<'_>,
    hits: MeshHitScene<'_>,
    materials: &Materials,
) -> Traced {
    let stream = Stream::new(
        "mesh_rays_test_skinned",
        RayClass::Nearest,
        rays.len() as u32,
    );
    let buffers = StreamBuffers::new(&gpu.device, &stream);
    gpu.queue
        .write_buffer(&buffers.rays, 0, bytemuck::cast_slice(rays));
    buffers.set_count(&gpu.queue, rays.len() as u32);
    let mut tracer =
        MeshRayTracer::new(&gpu.device, &gpu.queue, RayClass::Nearest, MESHES).unwrap();
    tracer.write_frame(1, |buffer, bytes| gpu.queue.write_buffer(buffer, 0, bytes));
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    {
        let mut cmds = helio_core::CommandRecorder::from_encoder(&mut encoder);
        buffers.clear(&mut cmds);
        let views = materials.views(gpu);
        let bindings = StreamBindings {
            header: &buffers.header,
            rays: &buffers.rays,
            hits: &buffers.hits,
            surfaces: buffers.surfaces.as_ref(),
        };
        tracer.record(
            &gpu.device,
            &mut cmds,
            bindings,
            Some(scene),
            Some(materials.bindings(&views)),
        );
        surfaces_of(
            gpu,
            RayClass::Nearest,
            1,
            &mut cmds,
            bindings,
            Some(hits),
            Some(materials.bindings(&views)),
            None,
            SurfaceDetail::Full,
        );
    }
    gpu.queue.submit([encoder.finish()]);
    Traced {
        hits: read(gpu, &buffers.hits),
        surfaces: read(gpu, buffers.surfaces.as_ref().unwrap()),
    }
}

// ── Materials at hits ───────────────────────────────────────────────────────

/// A mipped RGBA8 texture of `size` x `size` whose every texel of mip `k`
/// is `colour(k)`.
fn mip_texture(gpu: &Gpu, size: u32, colour: impl Fn(u32) -> [u8; 4]) -> wgpu::Texture {
    let levels = size.ilog2() + 1;
    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("mip colours"),
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
    for mip in 0..levels {
        let side = (size >> mip).max(1);
        let texels: Vec<u8> = (0..side * side).flat_map(|_| colour(mip)).collect();
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
    }
    texture
}

/// One frame of a nearest-hit stream at `detail`, its surfaces by the
/// rows' programs.
fn trace_detail(
    gpu: &Gpu,
    rays: &[SceneRay],
    scene: &Scene,
    materials: &Materials,
    detail: SurfaceDetail,
) -> (Traced, usize) {
    let stream = Stream::new(
        "mesh_rays_test_detail",
        RayClass::Nearest,
        rays.len() as u32,
    );
    let buffers = StreamBuffers::new(&gpu.device, &stream);
    gpu.queue
        .write_buffer(&buffers.rays, 0, bytemuck::cast_slice(rays));
    buffers.set_count(&gpu.queue, rays.len() as u32);
    let mut tracer =
        MeshRayTracer::new(&gpu.device, &gpu.queue, RayClass::Nearest, MESHES).unwrap();
    tracer.write_frame(1, |buffer, bytes| gpu.queue.write_buffer(buffer, 0, bytes));
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    let bins;
    {
        let mut cmds = helio_core::CommandRecorder::from_encoder(&mut encoder);
        buffers.clear(&mut cmds);
        let views = materials.views(gpu);
        let bindings = StreamBindings {
            header: &buffers.header,
            rays: &buffers.rays,
            hits: &buffers.hits,
            surfaces: buffers.surfaces.as_ref(),
        };
        tracer.record(
            &gpu.device,
            &mut cmds,
            bindings,
            Some(scene.rays()),
            Some(materials.bindings(&views)),
        );
        bins = surfaces_of(
            gpu,
            RayClass::Nearest,
            1,
            &mut cmds,
            bindings,
            Some(scene.hits()),
            Some(materials.bindings(&views)),
            Some(&scene.programs),
            detail,
        );
    }
    gpu.queue.submit([encoder.finish()]);
    (
        Traced {
            hits: read(gpu, &buffers.hits),
            surfaces: read(gpu, buffers.surfaces.as_ref().unwrap()),
        },
        bins,
    )
}

/// Rays down -z onto the quads laid side by side along x (2 m apart), `n`
/// a quad, clear of their edges.
fn rays_onto(quads: usize, n: usize, spread: f32) -> Vec<SceneRay> {
    (0..quads * n)
        .map(|k| {
            let (quad, i) = (k / n, k % n);
            let x = quad as f32 * 3.0 + (i % 8) as f32 * 0.2 - 0.7;
            let y = (i / 8) as f32 * 0.2 - 0.7;
            ray(Vec3::new(x, y, 2.0), Vec3::NEG_Z, 10.0, spread, MASK_ALL)
        })
        .collect()
}

/// Quads side by side along x (3 m apart), each with `program`.
fn side_by_side(programs: &[RayProgram]) -> Vec<Quad> {
    programs
        .iter()
        .enumerate()
        .map(|(k, program)| Quad {
            base_color: Vec3::new(0.2 + 0.1 * k as f32, 0.6, 0.4),
            program: *program,
            transform: Mat4::from_translation(Vec3::new(3.0 * k as f32, 0.0, 0.0)),
            ..Default::default()
        })
        .collect()
}

/// Hits binned by program: the table-driven quad, a clear-coat quad and a
/// two-sided foliage quad each shade by their own program (three bins), the
/// first two as the table describes them and the leaf with its share of its
/// colour let through (`material_transmitted`: the raster's and the hit's
/// one description); a program whose graph is not registered yet, and one
/// of a template no registry holds, shade by the table-driven program,
/// never left without a surface.
#[test]
fn hits_are_shaded_by_their_material_program() {
    let Some(gpu) = gpu() else { return };
    let programs = [
        RayProgram::TABLE,
        RayProgram {
            template: helio_mats::MATERIAL_CLASS_CLEAR_COAT,
            graph_hash: 0,
        },
        RayProgram {
            template: helio_mats::MATERIAL_CLASS_FOLIAGE,
            graph_hash: 0,
        },
        // A graph whose source is not registered: waits, the table shades.
        RayProgram {
            template: 0,
            graph_hash: 0x5eed_0000_dead_beef,
        },
        // A custom class no registry holds: the table's.
        RayProgram {
            template: helio_mats::FIRST_CUSTOM_MATERIAL_CLASS + 900,
            graph_hash: 0,
        },
    ];
    let quads = side_by_side(&programs);
    let mut foliage = material(0.6, 0.0);
    foliage.material_class = helio_mats::MATERIAL_CLASS_FOLIAGE;
    foliage.class_params = [0.4, 0.0, 0.0, 0.0];
    let mut clear_coat = material(0.3, 0.0);
    clear_coat.material_class = helio_mats::MATERIAL_CLASS_CLEAR_COAT;
    let mut quads = quads;
    quads[1].material = 1;
    quads[2].material = 2;
    let rows = [material(0.5, 0.0), clear_coat, foliage];
    let scene = scene(gpu, &quads);
    let table = materials(gpu, &quad_rows(&quads, &rows));
    let rays = rays_onto(quads.len(), 16, 0.0);
    for detail in [SurfaceDetail::Full, SurfaceDetail::Diffuse] {
        let (traced, bins) = trace_detail(gpu, &rays, &scene, &table, detail);
        assert_eq!(bins, 3, "{detail:?}: the table, clear coat and foliage");
        for (k, r) in rays.iter().enumerate() {
            let quad = k / 16;
            let (hit, surface) = (&traced.hits[k], &traced.surfaces[k]);
            assert_eq!(
                payload::row(hit),
                quad as u32,
                "{detail:?} ray {k}: {hit:?}"
            );
            let [red, green, blue, _] = surface.base_color();
            let albedo = Vec3::new(red, green, blue);
            assert!(
                (albedo - quads[quad].base_color).abs().max_element() <= 1.0 / 255.0,
                "{detail:?} ray {k}: {albedo} want {}",
                quads[quad].base_color
            );
            let through = Vec3::from(surface.transmission());
            if quad == 2 {
                let want = quads[2].base_color * 0.4;
                assert!(
                    (through - want).abs().max_element() < 0.01,
                    "{detail:?} ray {k}: the leaf lets {through} through, want {want}"
                );
            } else {
                assert_eq!(through, Vec3::ZERO, "{detail:?} ray {k}");
                if detail == SurfaceDetail::Full {
                    check_hit(&quads, &rows, k, r, hit, surface);
                }
            }
        }
    }
}

/// Textures at the ray cone's footprint: a texture whose mips are each a
/// colour of their own, on a 2 m quad across UV 0..1 (64 texels: 3.1 cm
/// each), met head on 2 m away by rays whose footprint there spans 1, 4
/// and 16 texels: each reads mip 0, 2 and 4, at either detail; a ray
/// meeting the quad at a slant reads a coarser mip than one head on.
#[test]
fn textures_are_read_at_the_ray_cone_s_footprint() {
    let Some(gpu) = gpu() else { return };
    let colours = [
        [255, 0, 0, 255],
        [255, 255, 0, 255],
        [0, 255, 0, 255],
        [0, 255, 255, 255],
        [0, 0, 255, 255],
        [255, 0, 255, 255],
        [255, 255, 255, 255],
    ];
    let texture = mip_texture(gpu, 64, |mip| colours[mip as usize]);
    let mut row = material(0.5, 0.0);
    row.tex_base_color = 0;
    let quads = [Quad {
        base_color: Vec3::ONE,
        ..Default::default()
    }];
    let scene = scene(gpu, &quads);
    let table = textured_materials(gpu, &quad_rows(&quads, &[row]), &texture);
    let texel = 2.0 / 64.0;
    for (texels, mip) in [(1.0f32, 0usize), (4.0, 2), (16.0, 4)] {
        // The footprint at 2 m is `texels` texels across.
        let rays = rays_onto(1, 16, texels * texel / 2.0);
        for detail in [SurfaceDetail::Full, SurfaceDetail::Diffuse] {
            let (traced, _) = trace_detail(gpu, &rays, &scene, &table, detail);
            let c = colours[mip].map(|c| f32::from(c) / 255.0);
            let want = Vec3::new(c[0], c[1], c[2]);
            for k in 0..rays.len() {
                let [r, g, b, _] = traced.surfaces[k].base_color();
                let got = Vec3::new(r, g, b);
                assert!(
                    (got - want).abs().max_element() < 0.1,
                    "{detail:?}, {texels} texels: ray {k} read {got}, want mip {mip} {want}"
                );
            }
        }
    }
    // At a slant (75 degrees off the normal) the same cone covers four
    // times the length along the slope: a coarser read than head on.
    let slant = 75f32.to_radians();
    let dir = Vec3::new(slant.sin(), 0.0, -slant.cos());
    let rays: Vec<SceneRay> = (0..16)
        .map(|i| {
            let target = Vec3::new(
                -0.5 + (i % 4) as f32 * 0.1,
                -0.5 + (i / 4) as f32 * 0.1,
                0.0,
            );
            ray(target - dir * 2.0, dir, 10.0, texel / 2.0, MASK_ALL)
        })
        .collect();
    let (traced, _) = trace_detail(gpu, &rays, &scene, &table, SurfaceDetail::Full);
    for k in 0..rays.len() {
        let [_, g, _, _] = traced.surfaces[k].base_color();
        // Mip 0 is pure red (no green); the slanted read is about mip 2
        // (the slope's axis about four texels long).
        assert!(traced.hits[k].t > 0.0, "slanted ray {k} missed");
        assert!(g > 0.5, "slanted ray {k}: green {g}: read at mip 0");
    }
}

/// `Full` evaluates the normal map; `Diffuse` keeps the vertex normal and
/// the material's constant roughness (no ORM texture read).
#[test]
fn diffuse_streams_skip_what_a_diffuse_consumer_does_not_read() {
    let Some(gpu) = gpu() else { return };
    // One texture for base colour, normal map and ORM: rgb (0.75, 0.5, 1.0),
    // a normal tilted toward +u; as ORM, roughness x0.5.
    let texture = mip_texture(gpu, 4, |_| [191, 128, 255, 255]);
    let mut row = material(0.8, 0.0);
    row.tex_base_color = 0;
    row.tex_normal = 0;
    row.tex_roughness = 0;
    row.flags = helio_mats::FLAG_HAS_NORMAL_MAP;
    let quads = [Quad {
        base_color: Vec3::ONE,
        ..Default::default()
    }];
    // The quad's tangent (+u: +x) and sign come from its vertices.
    let scene = scene(gpu, &quads);
    let table = textured_materials(gpu, &quad_rows(&quads, &[row]), &texture);
    let rays = rays_onto(1, 16, 0.0);
    let (full, _) = trace_detail(gpu, &rays, &scene, &table, SurfaceDetail::Full);
    let (diffuse, _) = trace_detail(gpu, &rays, &scene, &table, SurfaceDetail::Diffuse);
    for k in 0..rays.len() {
        let n_full = Vec3::from(stream::unpack_unit(full.surfaces[k].normal));
        let n_diffuse = Vec3::from(stream::unpack_unit(diffuse.surfaces[k].normal));
        assert!(
            angle(n_diffuse, Vec3::Z) < 1e-3,
            "ray {k}: diffuse normal {n_diffuse}"
        );
        assert!(
            angle(n_full, Vec3::Z) > 0.3,
            "ray {k}: the full surface's normal map unread: {n_full}"
        );
        assert!(
            (diffuse.surfaces[k].roughness() - 0.8).abs() < 0.01,
            "ray {k}: diffuse roughness {}",
            diffuse.surfaces[k].roughness()
        );
        assert!(
            full.surfaces[k].roughness() < 0.75,
            "ray {k}: the full surface's ORM unread: {}",
            full.surfaces[k].roughness()
        );
    }
}

// ── Cost ────────────────────────────────────────────────────────────────────

/// The surface passes' GPU cost on GI's stream shape (174,080 rays, all
/// hitting meshes: 81 textured, normal-mapped quads), by programs present
/// (1: the table-driven program alone, no binning; 2, 8, 64 and 72, past the
/// bound) and detail, as GPU milliseconds a run of the passes
/// (the throughput of the surface passes alone, recorded five times a frame
/// over 64 pipelined frames so the GPU never idles and runs at a loaded
/// frame's clocks; three windows, the median). Run by hand on a quiet GPU:
/// `mesh_rays_gpu hit_surface_cost --ignored --nocapture`.
#[test]
#[ignore = "a measurement; run by hand"]
fn hit_surface_cost() {
    let mut descriptor = helio_core::wgpu_instance_descriptor();
    // Measurement: wgpu's indirect-call validation (a dispatch and two
    // barriers before every indirect dispatch) off.
    if std::env::var_os("BENCH_NO_INDIRECT_VALIDATION").is_some() {
        descriptor
            .flags
            .remove(wgpu::InstanceFlags::VALIDATION_INDIRECT_CALL);
    }
    let instance = wgpu::Instance::new(descriptor);
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .unwrap();
    let wanted = wgpu::Features::EXPERIMENTAL_RAY_QUERY
        | wgpu::Features::TIMESTAMP_QUERY
        | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
    if !adapter.features().contains(wanted) {
        eprintln!("skipped: no ray queries or timestamps");
        return;
    }
    let limits = adapter.limits();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: wanted | (adapter.features() & helio_mats::BINDLESS_MATERIAL_FEATURES),
        required_limits: wgpu::Limits {
            max_buffer_size: limits.max_buffer_size.min(u32::MAX as u64),
            ..limits
        },
        experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
        ..Default::default()
    }))
    .unwrap();
    let gpu = Gpu {
        device: Arc::new(device),
        queue,
    };
    let gpu = &gpu;
    // 81 quads in a 9x9 grid at z = 0, each 2 m, 2.5 m apart.
    let texture = mip_texture(gpu, 256, |mip| {
        [255 - (mip * 30) as u8, 128, (mip * 30) as u8, 255]
    });
    let mut row = material(0.6, 0.0);
    row.tex_base_color = 0;
    row.tex_normal = 0;
    row.tex_roughness = 0;
    row.flags = helio_mats::FLAG_HAS_NORMAL_MAP;
    // Graph programs: each sets its own colour (registered sources).
    let graph = |k: u32| RayProgram {
        template: 0,
        graph_hash: 0x6a5e_0000_0000_0000 | u64::from(k),
    };
    for k in 1..80 {
        helio_mats::register_graph_source(
            graph(k).graph_hash,
            format!(
                "/*RADIANT_GRAPH_DECLARATIONS*/\n\n/*RADIANT_GRAPH_BODY*/\nalbedo = vec4<f32>({:.3}, 0.5, 0.5, 1.0) * albedo;\n",
                k as f32 / 80.0
            ),
        );
    }
    let lcg = std::cell::RefCell::new(Lcg(5));
    let rays: Vec<SceneRay> = (0..174_080)
        .map(|_| {
            let mut lcg = lcg.borrow_mut();
            let (qx, qy) = ((lcg.next() * 9.0) as u32, (lcg.next() * 9.0) as u32);
            let x = qx as f32 * 2.5 + lcg.next() * 1.8 - 0.9;
            let y = qy as f32 * 2.5 + lcg.next() * 1.8 - 0.9;
            ray(
                Vec3::new(x, y, 3.0),
                Vec3::new(lcg.next() * 0.2 - 0.1, lcg.next() * 0.2 - 0.1, -1.0),
                10.0,
                0.004,
                MASK_ALL,
            )
        })
        .collect();
    let stream = Stream::new("mesh_rays_cost", RayClass::Nearest, rays.len() as u32);
    let buffers = StreamBuffers::new(&gpu.device, &stream);
    gpu.queue
        .write_buffer(&buffers.rays, 0, bytemuck::cast_slice(&rays));
    buffers.set_count(&gpu.queue, rays.len() as u32);
    let bindings = StreamBindings {
        header: &buffers.header,
        rays: &buffers.rays,
        hits: &buffers.hits,
        surfaces: buffers.surfaces.as_ref(),
    };
    let programs_shared = Arc::new(HitPrograms::new(&gpu.device, &gpu.queue, None));
    println!("| Programs present | Bins | Count | Prefix | Scatter | Binning | Shading, Diffuse | Shading, Full |");
    println!("|---|---|---|---|---|---|---|---|");
    // (what, programs present, program of quad k, programs present that
    // no quad uses).
    type ProgramOf = Box<dyn Fn(u32) -> RayProgram>;
    let cases: Vec<(&str, u32, ProgramOf, Vec<RayProgram>)> = vec![
        ("table alone", 1, Box::new(|_| RayProgram::TABLE), vec![]),
        (
            "table, an unused program present",
            2,
            Box::new(|_| RayProgram::TABLE),
            vec![graph(1)],
        ),
        (
            "64 present, 2 with hits",
            64,
            Box::new(move |k| {
                if k % 2 == 0 {
                    RayProgram::TABLE
                } else {
                    graph(1)
                }
            }),
            (2..64).map(graph).collect(),
        ),
        (
            "2 programs",
            2,
            Box::new(move |k| {
                if k % 2 == 0 {
                    RayProgram::TABLE
                } else {
                    graph(1)
                }
            }),
            vec![],
        ),
        (
            "8 programs",
            8,
            Box::new(move |k| {
                if k % 8 == 0 {
                    RayProgram::TABLE
                } else {
                    graph(k % 8)
                }
            }),
            vec![],
        ),
        (
            "64 programs",
            64,
            Box::new(move |k| {
                if k % 64 == 0 {
                    RayProgram::TABLE
                } else {
                    graph(k % 64)
                }
            }),
            vec![],
        ),
    ];
    let only = std::env::var("BENCH_ONLY").unwrap_or_default();
    for (what, present, program_of, unused) in cases {
        if !only.is_empty() && !only.split(',').any(|o| what.starts_with(o)) {
            continue;
        }
        let quads: Vec<Quad> = (0..81u32)
            .map(|k| Quad {
                base_color: Vec3::splat(0.8),
                program: program_of(k),
                transform: Mat4::from_translation(Vec3::new(
                    (k % 9) as f32 * 2.5,
                    (k / 9) as f32 * 2.5,
                    0.0,
                )),
                ..Default::default()
            })
            .collect();
        let mut scene = scene(gpu, &quads);
        for program in unused {
            scene.programs.index(program);
        }
        let table = textured_materials(gpu, &quad_rows(&quads, &[row]), &texture);
        // Trace once: the hits every surface frame reads.
        let mut tracer =
            MeshRayTracer::new(&gpu.device, &gpu.queue, RayClass::Nearest, MESHES).unwrap();
        tracer.write_frame(1, |b, bytes| gpu.queue.write_buffer(b, 0, bytes));
        let views = table.views(gpu);
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        {
            let mut cmds = helio_core::CommandRecorder::from_encoder(&mut encoder);
            buffers.clear(&mut cmds);
            tracer.record(
                &gpu.device,
                &mut cmds,
                bindings,
                Some(scene.rays()),
                Some(table.bindings(&views)),
            );
        }
        gpu.queue.submit([encoder.finish()]);
        let mut row_out: Vec<[f64; 4]> = Vec::new();
        let mut bins = 0;
        for detail in [SurfaceDetail::Diffuse, SurfaceDetail::Full] {
            let mut surfaces =
                MeshHitSurfaces::new(&gpu.device, programs_shared.clone(), detail, MESHES);
            let write =
                |b: &wgpu::Buffer, o: u64, bytes: &[u8]| gpu.queue.write_buffer(b, o, bytes);
            surfaces.set_programs(&gpu.device, Some(&scene.programs), write);
            programs_shared.wait();
            surfaces.set_programs(&gpu.device, Some(&scene.programs), write);
            surfaces.write_frame(1, |b, bytes| gpu.queue.write_buffer(b, 0, bytes));
            // One frame whose bin counts come back, as a running stream's
            // do: programs present without hits then take no bin.
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            surfaces.record(
                &gpu.device,
                &mut helio_core::CommandRecorder::from_encoder(&mut encoder),
                bindings,
                Some(scene.hits()),
                Some(table.bindings(&views)),
            );
            gpu.queue.submit([encoder.finish()]);
            for _ in 0..2 {
                surfaces.set_programs(&gpu.device, Some(&scene.programs), write);
                gpu.device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .unwrap();
            }
            surfaces.set_programs(&gpu.device, Some(&scene.programs), write);
            surfaces.write_frame(1, |b, bytes| gpu.queue.write_buffer(b, 0, bytes));
            bins = surfaces.bin_count();
            // 24 pipelined frames of 16 runs each (the GPU kept busy), the
            // last two runs of each frame timed stage by stage. A pass's
            // beginning timestamp may be written before the work ahead of
            // it drains, so a stage is the time between the ends of passes:
            // the count from the previous run's shading end, each later
            // stage from the stage before; the median over the frames.
            const FRAMES: u32 = 24;
            const RUNS: u32 = 16;
            let queries = gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: None,
                ty: wgpu::QueryType::Timestamp,
                count: FRAMES * 16,
            });
            for frame in 0..FRAMES + 8 {
                let mut encoder = gpu.device.create_command_encoder(&Default::default());
                for run in 0..RUNS {
                    let timed = frame >= 8 && run >= RUNS - 2;
                    surfaces.set_stage_timestamps(
                        timed.then(|| (queries.clone(), (frame - 8) * 16 + 8 * (run + 2 - RUNS))),
                    );
                    let mut cmds = helio_core::CommandRecorder::from_encoder(&mut encoder);
                    surfaces.record(
                        &gpu.device,
                        &mut cmds,
                        bindings,
                        Some(scene.hits()),
                        Some(table.bindings(&views)),
                    );
                }
                gpu.queue.submit([encoder.finish()]);
            }
            surfaces.set_stage_timestamps(None);
            let resolve = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: u64::from(FRAMES) * 128,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            encoder.resolve_query_set(&queries, 0..FRAMES * 16, &resolve, 0);
            gpu.queue.submit([encoder.finish()]);
            let stamps = read::<u64>(gpu, &resolve);
            let period = f64::from(gpu.queue.get_timestamp_period());
            // Per stage (count, prefix, scatter, shading), from pass ends:
            // f[0..8] the run before, f[8..16] the timed run; an end is at
            // 2 * stage + 1.
            let binned = surfaces.bin_count() > 1;
            let stage = |k: usize| {
                let mut ms: Vec<f64> = stamps
                    .chunks(16)
                    .map(|f| {
                        let end = |run: usize, stage: usize| f[8 * run + 2 * stage + 1];
                        let (from, to) = match (binned, k) {
                            (false, 3) => (end(0, 3), end(1, 3)),
                            (false, _) => (0, 0),
                            (true, 0) => (end(0, 3), end(1, 0)),
                            (true, k) => (end(1, k - 1), end(1, k)),
                        };
                        to.saturating_sub(from) as f64 * period / 1e6
                    })
                    .collect();
                ms.sort_by(f64::total_cmp);
                ms[ms.len() / 2]
            };
            let run = {
                let mut ms: Vec<f64> = stamps
                    .chunks(16)
                    .map(|f| f[15].saturating_sub(f[7]) as f64 * period / 1e6)
                    .collect();
                ms.sort_by(f64::total_cmp);
                ms[ms.len() / 2]
            };
            eprintln!("{what} {detail:?}: a whole run {run:.3} ms");
            if binned && std::env::var_os("BENCH_BIN_STAMPS").is_some() {
                // Each bin in a pass of its own, its end stamped: a bin's
                // time is its end less the bin before's (the median over
                // frames of 16 runs, the last run stamped).
                let n = surfaces.bin_count() as u32;
                let queries = gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: None,
                    ty: wgpu::QueryType::Timestamp,
                    count: FRAMES * n,
                });
                for frame in 0..FRAMES + 8 {
                    let mut encoder = gpu.device.create_command_encoder(&Default::default());
                    for run in 0..RUNS {
                        surfaces.set_bin_timestamps(
                            (frame >= 8 && run == RUNS - 1)
                                .then(|| (queries.clone(), (frame - 8) * n)),
                        );
                        surfaces.record(
                            &gpu.device,
                            &mut helio_core::CommandRecorder::from_encoder(&mut encoder),
                            bindings,
                            Some(scene.hits()),
                            Some(table.bindings(&views)),
                        );
                    }
                    gpu.queue.submit([encoder.finish()]);
                }
                surfaces.set_bin_timestamps(None);
                let resolve = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: u64::from(FRAMES * n) * 8,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                let mut encoder = gpu.device.create_command_encoder(&Default::default());
                encoder.resolve_query_set(&queries, 0..FRAMES * n, &resolve, 0);
                gpu.queue.submit([encoder.finish()]);
                let stamps = read::<u64>(gpu, &resolve);
                let per_bin: Vec<f64> = (1..n as usize)
                    .map(|k| {
                        let mut us: Vec<f64> = stamps
                            .chunks(n as usize)
                            .map(|f| f[k].saturating_sub(f[k - 1]) as f64 * period / 1e3)
                            .collect();
                        us.sort_by(f64::total_cmp);
                        us[us.len() / 2]
                    })
                    .collect();
                let mut sorted = per_bin.clone();
                sorted.sort_by(f64::total_cmp);
                eprintln!(
                    "{what} {detail:?}: bins 1..{n} in passes of their own, us each: min {:.1} median {:.1} max {:.1} sum {:.1}; first 8 {:?}",
                    sorted[0],
                    sorted[sorted.len() / 2],
                    sorted[sorted.len() - 1],
                    per_bin.iter().sum::<f64>(),
                    per_bin.iter().take(8).map(|u| (u * 10.0).round() / 10.0).collect::<Vec<_>>()
                );
            }
            let stages = [stage(0), stage(1), stage(2), stage(3)];
            row_out.push(stages);
        }
        let [d, f] = [row_out[0], row_out[1]];
        println!(
            "| {what} ({present}) | {bins} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} |",
            d[0],
            d[1],
            d[2],
            d[0] + d[1] + d[2],
            d[3],
            f[3]
        );
    }
}

// ── Coordinate spaces ───────────────────────────────────────────────────────

/// Two quads at the same place in two coordinate spaces (`near_space` a
/// little nearer the rays than `far_space`'s), and the spaces in `others`
/// holding a quad far off the rays (they take mask groups).
fn overlapping(near_space: u32, far_space: u32, others: &[u32]) -> Vec<Quad> {
    let mut quads = vec![
        Quad {
            base_color: Vec3::new(0.0, 0.9, 0.0),
            transform: Mat4::from_translation(Vec3::Z * 0.01),
            space: near_space,
            ..Default::default()
        },
        Quad {
            base_color: Vec3::new(0.9, 0.0, 0.0),
            space: far_space,
            ..Default::default()
        },
    ];
    quads.extend(others.iter().map(|&space| Quad {
        transform: Mat4::from_translation(Vec3::new(100.0, 0.0, 0.0)),
        space,
        ..Default::default()
    }));
    quads
}

/// A ray down -z through the quads' middle in space `space`.
fn down_in(space: u32) -> SceneRay {
    ray(Vec3::new(0.1, 0.2, 5.0), -Vec3::Z, 50.0, 0.0, MASK_ALL).in_space(space)
}

/// Overlapping placements in two spaces: a ray of every space meets the
/// nearer, a ray of one space meets that space's quad alone, a ray of a
/// space with nothing traced meets nothing; with a group each and with
/// every group shared (more spaces than the mask has groups); for nearest
/// hits (with their surfaces) and for occlusion.
#[test]
fn a_ray_meets_only_its_own_space() {
    let Some(gpu) = gpu() else { return };
    for (what, quads) in [
        ("a group each", overlapping(4, 0, &[])),
        // Spaces 1 and 4 share group 1: the near quad (4) is a candidate of
        // the far space's rays, passed by its row's space.
        ("shared groups", overlapping(4, 1, &[2, 3, 5, 6])),
        ("the world nearer", overlapping(0, 9, &[2])),
    ] {
        let scene = scene(gpu, &quads);
        let table = materials(gpu, &quad_rows(&quads, &[material(0.5, 0.0)]));
        let (near, far) = (quads[0].space, quads[1].space);
        let rays = [
            down_in(SPACE_ANY),
            down_in(space_of_slot(near)),
            down_in(space_of_slot(far)),
            down_in(space_of_slot(77)),
        ];
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
        let colour = |k: usize| {
            let [r, g, b, _] = traced.surfaces[k].base_color();
            Vec3::new(r, g, b)
        };
        assert_eq!(payload::row(&traced.hits[0]), 0, "{what}: any space");
        assert!(
            (traced.hits[0].t - 4.99).abs() < 1e-3,
            "{what}: {:?}",
            traced.hits[0]
        );
        assert_eq!(payload::row(&traced.hits[1]), 0, "{what}: the near space");
        assert!(colour(1).y > 0.8, "{what}: {}", colour(1));
        assert_eq!(payload::row(&traced.hits[2]), 1, "{what}: the far space");
        assert!(
            (traced.hits[2].t - 5.0).abs() < 1e-3,
            "{what}: {:?}",
            traced.hits[2]
        );
        assert!(colour(2).x > 0.8, "{what}: {}", colour(2));
        assert_eq!(
            traced.hits[3],
            SceneHit::default(),
            "{what}: an empty space"
        );
        for (k, r) in rays.iter().enumerate() {
            check_hit(
                &quads,
                &[material(0.5, 0.0)],
                k,
                r,
                &traced.hits[k],
                &traced.surfaces[k],
            );
        }
        let occluded = trace(
            gpu,
            RayClass::Occlusion,
            &rays,
            rays.len() as u32,
            None,
            None,
            Some(&scene),
            Some(&table),
            1,
        );
        let through: Vec<[f32; 3]> = occluded.hits.iter().map(SceneHit::through).collect();
        assert_eq!(
            through[..3],
            [[0.0; 3]; 3],
            "{what}: blocked in a space that has a quad"
        );
        assert_eq!(through[3], [1.0; 3], "{what}: nothing in an empty space");
    }
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
        // A texture store: the materials' table the drawers and the hit
        // programs sample.
        let textures = Arc::new(std::sync::RwLock::new(
            pulsar_scenedb::gpu::TextureStore::with_reserved_slots(
                16,
                helio_core::TEXTURE_TABLE_RESERVED_SLOTS,
            ),
        ));
        scene_db.world.attach_gpu_mirror(
            pulsar_scenedb::gpu::GpuMirrorHandle::new(Arc::new(store), queue.clone())
                .with_texture_store(textures)
                .unwrap(),
        );
        scene_db
    }

    /// An axis-aligned box of half extent `half` at `centre` in `colour`,
    /// roughness 0.8.
    fn spawn_box(world: &mut pulsar_scenedb::World, centre: Vec3, half: Vec3, colour: [f32; 3]) {
        spawn_box_with(
            world,
            centre,
            half,
            MaterialComponent::from(helio_mats::MaterialRow::new(
                [colour[0], colour[1], colour[2], 1.0],
                0.8,
                0.0,
                [0.0; 3],
                0.0,
            )),
        );
    }

    /// An axis-aligned box of half extent `half` at `centre` drawn with
    /// `row`: each face UV 0..1, its tangent along +u.
    fn spawn_box_with(
        world: &mut pulsar_scenedb::World,
        centre: Vec3,
        half: Vec3,
        row: MaterialComponent,
    ) {
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
                        [(a + 1.0) * 0.5, (b + 1.0) * 0.5],
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

    /// The standalone projection (`SceneDbRayTracing`) culls by distance and
    /// angle from the eye its host gives, and the meshes' tracer then sees
    /// only what it holds: a 1 m box 1 km away (0.11 degrees) is left out,
    /// a 40 m wall at the same distance (2.3 degrees) and a box near the eye
    /// are traced; without an eye every object is.
    #[test]
    fn the_standalone_projection_traces_what_its_culling_keeps() {
        let Some(gpu) = gpu() else { return };
        let mut scene = scene_db(&gpu.device, &Arc::new(gpu.queue.clone()));
        let world = &mut scene.world;
        spawn_box(world, Vec3::new(0.0, 0.0, -5.0), Vec3::splat(1.0), [0.5; 3]);
        spawn_box(
            world,
            Vec3::new(1000.0, 0.0, 0.0),
            Vec3::splat(0.5),
            [0.5; 3],
        );
        spawn_box(
            world,
            Vec3::new(0.0, 0.0, 1000.0),
            Vec3::new(20.0, 20.0, 0.5),
            [0.5; 3],
        );
        world.flush_gpu_mirror(&gpu.queue);
        let mut projection =
            SceneDbRayTracing::partial(gpu.device.clone(), Arc::new(gpu.queue.clone()));
        let rays = [
            ray(Vec3::ZERO, -Vec3::Z, 2000.0, 0.0, MASK_ALL),
            ray(Vec3::ZERO, Vec3::X, 2000.0, 0.0, MASK_ALL),
            ray(Vec3::ZERO, Vec3::Z, 2000.0, 0.0, MASK_ALL),
        ];
        let table = materials(gpu, &[material(0.5, 0.0); 4]);
        let store = world.gpu_mirror().unwrap().store();
        let pool = |key: helio_core::BufferKey| store.resolve_buffer_handle(key).unwrap().buffer;
        let (vertices, indices) = (
            pool(helio_mesh_api::MESH_VERTICES.key()),
            pool(helio_mesh_api::MESH_INDICES.key()),
        );
        let shoot = |projection: &mut SceneDbRayTracing| {
            projection.prepare(world).unwrap();
            let scene_rays = projection.scene_rays().unwrap();
            let instances = scene_rays.buffer(helio_mesh_api::RAY_INSTANCES).unwrap();
            let scene = MeshRayScene {
                tlas: &scene_rays.tlas,
                instances,
                space_groups: scene_rays.buffer(helio_mesh_api::RAY_SPACE_GROUPS),
                vertices: &vertices,
                indices: &indices,
                proxies: None,
                skinned: None,
            };
            let hits = MeshHitScene {
                instances,
                transforms: scene_rays
                    .buffer(helio_mesh_api::RAY_INSTANCE_TRANSFORMS)
                    .unwrap(),
                vertices: &vertices,
                indices: &indices,
                proxies: None,
                skinned: None,
            };
            let traced = trace_scene(gpu, &rays, scene, hits, &table);
            traced.hits.iter().map(|h| h.t).collect::<Vec<f32>>()
        };
        projection.set_view(
            Some(glam::DVec3::ZERO),
            helio_pass_mesh_rays::RayTracingSettings::default(),
        );
        let culled = shoot(&mut projection);
        assert_eq!(projection.culled_count(), 1, "the far small box");
        assert!((culled[0] - 4.0).abs() < 1e-3, "the near box: {culled:?}");
        assert_eq!(culled[1], 0.0, "the far small box is culled: {culled:?}");
        assert!(
            (culled[2] - 999.5).abs() < 1e-2,
            "the far wall stays: {culled:?}"
        );
        projection.set_view(None, helio_pass_mesh_rays::RayTracingSettings::default());
        let all = shoot(&mut projection);
        assert_eq!(projection.culled_count(), 0);
        assert!(
            (all[1] - 999.5).abs() < 1e-2,
            "every object without an eye: {all:?}"
        );
    }

    // ── One surface, two encodings ─────────────────────────────────────────

    /// The view the comparison renders.
    const VIEW: u32 = 64;
    /// The pixels it compares: every fourth, each with its four neighbours
    /// (a sample counts only where all five meet the same section, clear of
    /// silhouettes).
    fn probe_pixels() -> Vec<[u32; 2]> {
        let mut pixels = Vec::new();
        for y in (4..VIEW - 4).step_by(4) {
            for x in (4..VIEW - 4).step_by(4) {
                for [dx, dy] in [[0i32, 0], [1, 0], [-1, 0], [0, 1], [0, -1]] {
                    pixels.push([(x as i32 + dx) as u32, (y as i32 + dy) as u32]);
                }
            }
        }
        pixels
    }

    /// Camera rays through [`probe_pixels`] (a `Full` stream, each as wide
    /// as its pixel), traced; then the G-buffer's surface at the same
    /// pixels, decoded with the frame's `gbuffer_surface_at` and packed as a
    /// hit's (`scene_surface_pack`), so both are read with one unpacker.
    struct SurfaceProbe;

    struct CameraRaysPass {
        stream: Stream,
        buffers: StreamBuffers,
        writes: &'static [&'static str],
    }

    struct GBufferAtPixelsPass {
        pipeline: wgpu::ComputePipeline,
        layout: wgpu::BindGroupLayout,
        pixels: wgpu::Buffer,
        surfaces: wgpu::Buffer,
    }

    const GBUFFER_AT_PIXELS: &str = "//!use frame_ray_stream
//!use frame_surface
//!use frame_ray_surface
@group(0) @binding(0) var albedo: texture_2d<f32>;
@group(0) @binding(1) var normal: texture_2d<f32>;
@group(0) @binding(2) var orm: texture_2d<f32>;
@group(0) @binding(3) var emissive: texture_2d<f32>;
@group(0) @binding(4) var sss: texture_2d<f32>;
@group(0) @binding(5) var extra: texture_2d<f32>;
@group(0) @binding(6) var<storage, read> pixels: array<vec2<u32>>;
@group(0) @binding(7) var<storage, read_write> surfaces: array<SceneSurface>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= arrayLength(&pixels) { return; }
    let dims = textureDimensions(albedo);
    let p = vec2<i32>(pixels[id.x] * dims / vec2<u32>(64u));
    let s = gbuffer_surface_at(GBufferSurfaceTexels(
        textureLoad(albedo, p, 0), textureLoad(normal, p, 0), textureLoad(orm, p, 0),
        textureLoad(emissive, p, 0), textureLoad(sss, p, 0), textureLoad(extra, p, 0)));
    surfaces[id.x] = scene_surface_pack(s, s.normal);
}
";

    /// What the deferred lighting reads the surface from: the G-buffer's
    /// group (albedo, normal, ORM, emissive) and its SSS and extra targets.
    const GBUFFER_SURFACE: [&str; 3] = [
        helio_frame_api::gbuffer::GROUP,
        helio_frame_api::gbuffer::SSS,
        helio_frame_api::gbuffer::EXTRA,
    ];

    impl Feature for SurfaceProbe {
        fn id(&self) -> &'static str {
            "test.surface_probe"
        }

        fn register(&self, registrar: &mut FeatureRegistrar<'_>) {
            registrar.node(
                pass_node(
                    "surface_probe",
                    "Surface Probe",
                    "Passes: Test",
                    "Camera rays and the G-buffer at the same pixels.",
                    |build| {
                        let count = probe_pixels().len() as u32;
                        let stream = Stream::new("mesh_rays_camera", RayClass::Nearest, count)
                            .with_detail(SurfaceDetail::Full);
                        let mut writes = vec![stream.header, stream.rays, stream.hits];
                        writes.extend(stream.surfaces);
                        build.graph.add_pass(Box::new(CameraRaysPass {
                            buffers: StreamBuffers::new(build.env.device, &stream),
                            writes: helio_frame_api::static_names(&writes),
                            stream,
                        }));
                        stream::trace(build, &stream)?;
                        let device = build.env.device;
                        let texture = |binding| wgpu::BindGroupLayoutEntry {
                            binding,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        };
                        let buffer = |binding, read_only| wgpu::BindGroupLayoutEntry {
                            binding,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        };
                        let mut entries: Vec<_> = (0..6).map(texture).collect();
                        entries.push(buffer(6, true));
                        entries.push(buffer(7, false));
                        let layout =
                            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                                label: Some("gbuffer at pixels"),
                                entries: &entries,
                            });
                        let pipeline_layout =
                            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                                label: None,
                                bind_group_layouts: &[Some(&layout)],
                                immediate_size: 0,
                            });
                        let module = helio_core::shader::module_with(
                            device,
                            "gbuffer at pixels",
                            GBUFFER_AT_PIXELS,
                            &[
                                helio_frame_api::rays::stream::SNIPPET,
                                helio_frame_api::gbuffer::SURFACE_SNIPPET,
                                helio_frame_api::rays::stream::SURFACE_SNIPPET,
                            ],
                        );
                        let pipeline = helio_core::shader::create_compute_pipeline(
                            device,
                            wgpu::ComputePipelineDescriptor {
                                label: Some("gbuffer at pixels"),
                                layout: Some(&pipeline_layout),
                                module: &module,
                                entry_point: Some("main"),
                                compilation_options: Default::default(),
                                cache: None,
                            },
                        );
                        let pixels: Vec<u32> = probe_pixels().into_iter().flatten().collect();
                        let pixels = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("probe pixels"),
                            contents: bytemuck::cast_slice(&pixels),
                            usage: wgpu::BufferUsages::STORAGE,
                        });
                        let surfaces = device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("gbuffer surfaces at pixels"),
                            size: u64::from(count) * 32,
                            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                            mapped_at_creation: false,
                        });
                        build.graph.add_pass(Box::new(GBufferAtPixelsPass {
                            pipeline,
                            layout,
                            pixels,
                            surfaces,
                        }));
                        Ok(())
                    },
                )
                .once_per_frame()
                .in_slot(Slot::Geometry, 50),
            );
        }
    }

    impl RenderPass for CameraRaysPass {
        fn name(&self) -> &'static str {
            "CameraRays"
        }

        fn writes(&self) -> &'static [&'static str] {
            self.writes
        }

        fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
            let camera = ctx.camera_data;
            let inverse = Mat4::from_cols_array(&camera.inv_view_proj);
            let eye = Vec3::from_slice(&camera.position_near[..3]);
            let through = |x: f32, y: f32| {
                let ndc = glam::Vec4::new(
                    2.0 * x / VIEW as f32 - 1.0,
                    1.0 - 2.0 * y / VIEW as f32,
                    0.5,
                    1.0,
                );
                let p = inverse * ndc;
                (p.truncate() / p.w - eye).normalize()
            };
            let rays: Vec<SceneRay> = probe_pixels()
                .iter()
                .map(|&[x, y]| {
                    // Where the raster sampled this pixel: its centre moved
                    // by the frame's jitter (the basis is unjittered).
                    let jitter = helio_core::temporal::ray_jitter_from_ndc(
                        [camera.jitter_frame[0], camera.jitter_frame[1]],
                        [VIEW, VIEW],
                    );
                    let (x, y) = (x as f32 + 0.5 + jitter[0], y as f32 + 0.5 + jitter[1]);
                    let dir = through(x, y);
                    // A pixel's angle: the cone the raster's derivatives span.
                    let spread = angle(dir, through(x + 1.0, y));
                    ray(eye, dir, 1000.0, spread, MASK_ALL)
                })
                .collect();
            ctx.write_buffer(&self.buffers.rays, 0, bytemuck::cast_slice(&rays));
            ctx.write_buffer(
                &self.buffers.header,
                0,
                bytemuck::bytes_of(&stream::StreamHeader::of(rays.len() as u32)),
            );
            ctx.write_buffer(
                &self.buffers.hits,
                0,
                bytemuck::cast_slice(&vec![SceneHit::default(); rays.len()]),
            );
            Ok(())
        }

        fn execute(&mut self, _ctx: &mut PassContext) -> HelioResult<()> {
            Ok(())
        }

        fn publish<'a>(&self, frame: &mut helio_core::ResourceRegistry<'a>) {
            self.buffers.publish(&self.stream, frame, "CameraRays");
        }
    }

    impl RenderPass for GBufferAtPixelsPass {
        fn name(&self) -> &'static str {
            "GBufferAtPixels"
        }

        fn reads(&self) -> &'static [&'static str] {
            &GBUFFER_SURFACE
        }

        fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
            let Some(group) = ctx.registry.read::<helio_core::ViewGroup<'_, 4>>(
                helio_core::ResourceKey::new(helio_frame_api::gbuffer::GROUP),
                "GBufferAtPixels",
            ) else {
                return Ok(());
            };
            let mut views: Vec<&wgpu::TextureView> = group.views.to_vec();
            for name in [
                helio_frame_api::gbuffer::SSS,
                helio_frame_api::gbuffer::EXTRA,
            ] {
                let Some(view) = ctx.registry.get(helio_core::ResourceKey::new(name)) else {
                    return Ok(());
                };
                views.push(view);
            }
            let mut entries: Vec<wgpu::BindGroupEntry<'_>> = views
                .iter()
                .enumerate()
                .map(|(binding, view)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: wgpu::BindingResource::TextureView(view),
                })
                .collect();
            entries.push(wgpu::BindGroupEntry {
                binding: 6,
                resource: self.pixels.as_entire_binding(),
            });
            entries.push(wgpu::BindGroupEntry {
                binding: 7,
                resource: self.surfaces.as_entire_binding(),
            });
            let group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.layout,
                entries: &entries,
            });
            let mut cmds = ctx.graphics_cmds();
            let mut pass = cmds.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, Some(&group), &[]);
            pass.dispatch_workgroups((probe_pixels().len() as u32).div_ceil(64), 1, 1);
            Ok(())
        }
    }

    /// An RGBA8 texture of `size` with mips box-filtered from `texel(x, y)`.
    fn register_texture(
        world: &pulsar_scenedb::World,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        size: u32,
        texel: impl Fn(u32, u32) -> [u8; 4],
    ) -> u32 {
        let levels = size.ilog2() + 1;
        let mut level: Vec<[f32; 4]> = (0..size * size)
            .map(|i| texel(i % size, i / size).map(|c| f32::from(c)))
            .collect();
        let mut mips = Vec::new();
        let mut side = size;
        loop {
            mips.push(
                level
                    .iter()
                    .flat_map(|t| t.map(|c| c.round() as u8))
                    .collect::<Vec<u8>>(),
            );
            if side == 1 {
                break;
            }
            let half = side / 2;
            level = (0..half * half)
                .map(|i| {
                    let (x, y) = (2 * (i % half), 2 * (i / half));
                    let at = |x: u32, y: u32| level[(y * side + x) as usize];
                    std::array::from_fn(|c| {
                        (at(x, y)[c] + at(x + 1, y)[c] + at(x, y + 1)[c] + at(x + 1, y + 1)[c])
                            / 4.0
                    })
                })
                .collect();
            side = half;
        }
        let descriptor = wgpu::TextureDescriptor {
            label: Some("probe texture"),
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
        };
        let levels: Vec<&[u8]> = mips.iter().map(Vec::as_slice).collect();
        world
            .gpu_mirror()
            .unwrap()
            .texture_store()
            .unwrap()
            .write()
            .unwrap()
            .register_mips(device, queue, &descriptor, &levels)
            .unwrap()
    }

    /// The test of "one surface, two encodings": a view rendered through the
    /// deferred graph and camera rays through the same pixels traced by the
    /// meshes as a `Full` stream; the G-buffer's surface (decoded with the
    /// frame's `gbuffer_surface_at`) and the hit's agree within the
    /// encodings' quantisation in base colour, normal, roughness, metalness
    /// and emission: a textured floor (a smooth gradient read at the
    /// pixel's footprint), a normal-mapped metallic wall and an emissive
    /// box.
    #[test]
    fn a_hit_s_surface_is_the_g_buffer_s_at_the_same_pixel() {
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
        let world = &mut scene.world;
        let gradient = register_texture(world, &device, &queue, 64, |x, y| {
            [(x * 4) as u8, (y * 4) as u8, 160, 255]
        });
        // A normal map tilted toward +u (alpha 1: no spread).
        let bumps = register_texture(world, &device, &queue, 4, |_, _| [200, 128, 230, 255]);
        let none = helio_mats::GpuMaterial::NO_TEXTURE;
        let mut floor = helio_mats::MaterialRow::new([1.0; 4], 0.6, 0.0, [0.0; 3], 0.0);
        floor.tex_base_color = gradient;
        let mut wall = helio_mats::MaterialRow::new([0.2, 0.4, 0.8, 1.0], 0.3, 1.0, [0.0; 3], 0.0);
        wall.tex_normal = bumps;
        wall.flags |= helio_mats::FLAG_HAS_NORMAL_MAP;
        // The emissive box's material class has a template of its own
        // (clear coat, no coat): its hits shade by that program, binned
        // apart from the table-driven ones.
        let mut glow = helio_mats::MaterialRow::new([0.9; 4], 0.5, 0.0, [1.0, 0.5, 0.25], 3.0);
        glow.material_class = helio_mats::MATERIAL_CLASS_CLEAR_COAT;
        assert_eq!(glow.tex_base_color, none);
        spawn_box_with(
            world,
            Vec3::new(0.0, -0.25, 0.0),
            Vec3::new(4.0, 0.25, 4.0),
            floor.into(),
        );
        spawn_box_with(
            world,
            Vec3::new(0.0, 1.5, -2.5),
            Vec3::new(2.0, 1.5, 0.25),
            wall.into(),
        );
        spawn_box_with(
            world,
            Vec3::new(1.2, 0.5, -0.5),
            Vec3::new(0.4, 0.5, 0.4),
            glow.into(),
        );
        // The G-buffer at the view's own size: a pixel is a texel.
        let config = helio::RendererConfig::new(VIEW, VIEW, wgpu::TextureFormat::Rgba8Unorm)
            .with_render_scale(1.0);
        let mut renderer =
            helio::RendererBuilder::new(config.clone(), scene.world.gpu_mirror().cloned().unwrap())
                .with_graph(helio_mesh_host::presets::with(
                    helio_presets::Preset::Deferred,
                    &helio_feature_api::FeatureSet::new().with(SurfaceProbe),
                ))
                .with_external_device()
                .build(
                    device.clone(),
                    queue.clone(),
                    config.width,
                    config.height,
                    config.surface_format,
                );
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: VIEW,
                height: VIEW,
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
            Vec3::new(0.3, 2.2, 4.0),
            Vec3::new(0.2, 0.6, -1.0),
            Vec3::Y,
            1.0,
            1.0,
            0.1,
            100.0,
        );
        let mut rays = SceneDbRayTracing::partial(device.clone(), queue.clone());
        for _ in 0..4 {
            scene.world.flush_gpu_mirror(&queue);
            rays.prepare(&scene.world).unwrap();
            rays.publish(&mut renderer);
            renderer.render(&camera, &view).unwrap();
            // The hit programs compile off the frame: wait for them between
            // frames, as a host's loading screen does.
            while helio_core::compile::in_flight() > 0 {
                std::thread::yield_now();
            }
        }
        let bins = renderer.frame_stats().latest().get("meshes.hit_bins");
        assert_eq!(
            bins,
            Some(2),
            "the hits are not binned by their programs (the table's and the clear coat's)"
        );
        let probe = renderer.find_pass::<CameraRaysPass>().unwrap();
        let hits = read_from::<SceneHit>(&device, &queue, &probe.buffers.hits);
        let traced =
            read_from::<SceneSurface>(&device, &queue, probe.buffers.surfaces.as_ref().unwrap());
        let gbuffer = renderer.find_pass::<GBufferAtPixelsPass>().unwrap();
        let raster = read_from::<SceneSurface>(&device, &queue, &gbuffer.surfaces);
        let drawn = raster.iter().filter(|s| s.base_color != 0).count();
        let traced_count = hits.iter().filter(|h| h.t > 0.0).count();
        eprintln!(
            "{drawn} G-buffer samples drawn, {traced_count} rays hit, of {}",
            hits.len()
        );
        let mut compared = [0usize; 3];
        for (sample, five) in hits.chunks(5).enumerate() {
            let row = payload::row(&five[0]);
            if five.iter().any(|h| h.t <= 0.0 || payload::row(h) != row) {
                continue;
            }
            let k = sample * 5;
            let (hit, gb) = (&traced[k], &raster[k]);
            let what = format!("sample {sample} (row {row})");
            let (a, b) = (hit.base_color(), gb.base_color());
            for c in 0..4 {
                // RGBA8 both; a texel's mip may differ by a level.
                assert!(
                    (a[c] - b[c]).abs() <= 3.0 / 255.0,
                    "{what}: base colour/metallic {a:?} vs {b:?}"
                );
            }
            assert!(
                (hit.roughness() - gb.roughness()).abs() <= 2.0 / 255.0,
                "{what}: roughness {} vs {}",
                hit.roughness(),
                gb.roughness()
            );
            let (n_hit, n_gb) = (
                Vec3::from(stream::unpack_unit(hit.normal)),
                Vec3::from(stream::unpack_unit(gb.normal)),
            );
            assert!(
                angle(n_hit, n_gb) < 0.01,
                "{what}: normal {n_hit} vs {n_gb}"
            );
            let (e_hit, e_gb) = (
                Vec3::from(stream::unpack_radiance(hit.emissive)),
                Vec3::from(stream::unpack_radiance(gb.emissive)),
            );
            assert!(
                (e_hit - e_gb).abs().max_element() <= 0.02 * e_gb.max_element() + 1e-3,
                "{what}: emission {e_hit} vs {e_gb}"
            );
            let kind = if e_gb.max_element() > 0.5 {
                2
            } else if b[3] > 0.5 {
                1
            } else {
                0
            };
            compared[kind] += 1;
        }
        eprintln!(
            "compared: floor {}, metallic wall {}, emissive box {}",
            compared[0], compared[1], compared[2]
        );
        assert!(
            compared.iter().all(|&n| n >= 3),
            "too few clear samples of a surface: {compared:?}"
        );
        let error = pollster::block_on(errors.pop());
        assert!(error.is_none(), "{error:?}");
    }
}
