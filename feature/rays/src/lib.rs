//! The meshes' tracer of ray streams (`helio_frame_api::rays::stream`): the
//! kind's answer to any consumer's rays, by hardware ray queries against the
//! frame's TLAS (`RenderEnvironment::tlas`, which the meshes' projections
//! publish as `helio_frame_api::rays::SCENE_RAYS`).
//!
//! - **Nearest-hit streams** (`mesh_rays_nearest`): each ray queries up to the
//!   nearest hit its row holds, with its caster mask. Opaque triangles
//!   commit; cut-outs (`RAY_INSTANCE_MASKED`) are tested as the raster's
//!   alpha test, at the mip matching the ray's footprint, stochastic past the
//!   finest mip; transmitting sheets (glass) are passed. A nearer hit is
//!   written with its [`SceneSurface`](helio_frame_api::rays::stream::SceneSurface)
//!   in the same invocation (below), in the hit payload of [`payload`].
//! - **Occlusion streams** (`mesh_rays_occlusion`): what gets through the
//!   meshes along the whole ray, multiplied into the row: nothing past an
//!   opaque triangle or a cut-out's surface, each sheet tinting.
//!
//! Both are one indirect dispatch over the stream's header. On a frame
//! without a TLAS or without the meshes' instance rows the pass records
//! nothing, and the rows stay as they were.
//!
//! **Why the trace pass writes the surface.** The surface's normals are the
//! triangle's object-space normals taken to the frame by the instance's
//! transform, and only a committed intersection holds that transform: the
//! instance rows (`RayInstance`) carry none, and a hit's 96 bits hold its row,
//! triangle and barycentrics. So the tracer's `surface` build adds no pass:
//! the trace pass writes a hit's surface when it commits the hit. A later
//! tracer that finds a nearer hit overwrites the row and writes its own
//! surface (its surface pass runs after every tracer traced), so a row always
//! ends with its final hit's surface. What that costs: the surface of a mesh
//! hit another kind then beats is wasted work. The alternative, a second
//! pass that re-fetches the triangle, needs each instance's transform on the
//! GPU by row; the projections publish it when the surface pass becomes the
//! materials' evaluator (r2: textured, binned by material), and the split
//! happens then.
//!
//! How others do it: DXR and Vulkan ray-tracing pipelines run a closest-hit
//! shader per hit group (Unreal's ray-tracing materials, RTXPT), which sees
//! the instance's transform and writes a payload; wavefront tracers split
//! intersection from shading through a hit buffer and keep the instance
//! table on the GPU (Laine et al. 2013). This tracer is the inline
//! equivalent of a closest-hit shader for the meshes' hit group.
//!
//! Cost scales with the rays a consumer writes, never with the scene: per
//! stream one dispatch over its count, bind groups made when their inputs
//! change, one 16-byte uniform written a frame.

pub mod feature;
mod settings;
pub use feature::MeshRays;
pub use settings::{RayTracingSettings, RAY_TRACING};

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use helio_core::{PassContext, PrepareContext, RenderPass, ResourceKey, Result as HelioResult};
use helio_frame_api::rays::stream::{RayClass, Stream, Tracer, TracerId};
use helio_mats::{MaterialBindingConfig, MaterialStandIn, MaterialTextureBindings};

/// The hit a nearest-hit stream's row holds for a mesh (`SceneHit`'s own
/// bits; the frame's `tracer` and `inside` fields aside).
///
/// - `a` payload (27 bits): the instance row, `instance_custom_data +
///   geometry_index` (the `ray_instances` row of the hit's section).
/// - `b`: the triangle (`primitive_index` in the section's BLAS geometry).
/// - `c`: the barycentrics of the hit's second and third corners,
///   `pack2x16unorm` (16 bits each, the second's in the low half).
///
/// `inside`: the ray met the back of a single-sided surface.
pub mod payload {
    /// The hit's `ray_instances` row.
    pub fn row(hit: &helio_frame_api::rays::stream::SceneHit) -> u32 {
        hit.payload()
    }

    /// The hit's triangle.
    pub fn primitive(hit: &helio_frame_api::rays::stream::SceneHit) -> u32 {
        hit.b
    }

    /// The hit's barycentrics (of the second and third corners).
    pub fn barycentrics(hit: &helio_frame_api::rays::stream::SceneHit) -> [f32; 2] {
        [hit.c & 0xffff, hit.c >> 16].map(|v| v as f32 / 65535.0)
    }
}

/// `MeshRayFrame` (`mesh_rays.wgsl`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuFrame {
    /// x: the frame's seed, y: the tracer's id.
    info: [u32; 4],
}

const FRAME_BYTES: u64 = std::mem::size_of::<GpuFrame>() as u64;

/// Whether `device` can trace rays against meshes (hardware ray queries and
/// a TLAS).
pub fn mesh_rays_supported(device: &wgpu::Device) -> bool {
    device
        .features()
        .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
        && device.limits().max_tlas_instance_count > 0
}

/// The meshes' tracer, as [`MeshRays`] registers it. A host without
/// features offers it to the graph being built
/// (`helio_frame_api::rays::stream::offer(env, tracer())`).
pub fn tracer() -> Tracer {
    Tracer {
        name: "meshes",
        supported: Arc::new(|env| mesh_rays_supported(env.device)),
        trace: Arc::new(|build, stream, id| {
            // One set of pipelines for every stream of the graph.
            let (device, queue) = (build.env.device.clone(), build.env.queue.clone());
            let pipelines = build
                .env
                .shared(|| SharedPipelines(None))
                .0
                .get_or_insert_with(|| Arc::new(Pipelines::new(&device, &queue)))
                .clone();
            let tracer =
                MeshRayTracer::with_pipelines(build.env.device, pipelines, stream.class, id);
            build
                .graph
                .add_pass(Box::new(MeshRayPass::new(*stream, tracer)));
            Ok(())
        }),
        // The trace pass writes the surfaces of the hits it commits (crate
        // docs): nothing to add once every tracer traced.
        surface: Arc::new(|_, _, _| Ok(())),
    }
}

/// The graph's [`Pipelines`], made by the first stream's pass.
struct SharedPipelines(Option<Arc<Pipelines>>);

/// What a tracer's dispatch reads and writes of its stream.
#[derive(Clone, Copy)]
pub struct StreamBindings<'a> {
    pub header: &'a wgpu::Buffer,
    pub rays: &'a wgpu::Buffer,
    pub hits: &'a wgpu::Buffer,
    /// A nearest-hit stream's; none for an occlusion stream.
    pub surfaces: Option<&'a wgpu::Buffer>,
}

/// The meshes as rays see them this frame: the TLAS, its instance rows
/// (`helio_mesh_api::RAY_INSTANCES`), the scene's mesh pools, the proxy
/// pool and the skin cache (`helio_mesh_api::RAY_SKINNED_VERTICES`), if any.
#[derive(Clone, Copy)]
pub struct MeshRayScene<'a> {
    pub tlas: &'a wgpu::Tlas,
    pub instances: &'a wgpu::Buffer,
    pub vertices: &'a wgpu::Buffer,
    pub indices: &'a wgpu::Buffer,
    pub proxies: Option<(&'a wgpu::Buffer, &'a wgpu::Buffer)>,
    pub skinned: Option<&'a wgpu::Buffer>,
}

/// The pipelines and stand-ins every stream's pass shares (one set a graph).
pub struct Pipelines {
    /// Group 0 of a nearest-hit stream (with its surfaces) and of an
    /// occlusion stream.
    nearest_layout: wgpu::BindGroupLayout,
    occlusion_layout: wgpu::BindGroupLayout,
    /// Group 1: material rows, then the cut-outs' alpha (`vg_alpha`).
    materials_layout: wgpu::BindGroupLayout,
    binding: MaterialBindingConfig,
    nearest: wgpu::ComputePipeline,
    occlusion: wgpu::ComputePipeline,
    /// Without the materials plugin: the materials' stand-in (one grey
    /// opaque row, a white texture table), as every drawer binds.
    no_materials: wgpu::BindGroup,
    /// No proxy pool: a binding for each of its buffers.
    no_proxies: wgpu::Buffer,
}

/// `mesh_rays.wgsl` with the cut-outs' alpha evaluator
/// (`helio_mats::vg_alpha`, group 1 from binding 1) in front.
pub fn shader_source(binding: MaterialBindingConfig) -> String {
    use helio_mats::vg_alpha;
    format!(
        "{}{}\n{}\n{}",
        vg_alpha::module_header(binding),
        vg_alpha::bindings_wgsl(binding, 1, 1),
        vg_alpha::WGSL,
        include_str!("../shaders/mesh_rays.wgsl")
    )
}

/// What the shader resolves (`//!use` markers).
pub const SNIPPETS: [helio_core::shader::ShaderSnippet; 3] = [
    helio_frame_api::rays::stream::SNIPPET,
    helio_mesh_api::RAY_SURFACE_SNIPPET,
    helio_mats::MATERIAL_ROW_SNIPPET,
];

fn storage(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

impl Pipelines {
    /// # Panics
    ///
    /// On a device without ray queries ([`mesh_rays_supported`]).
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        assert!(
            mesh_rays_supported(device),
            "the meshes' tracer needs hardware ray queries"
        );
        let mut entries = vec![
            storage(0, true),
            storage(1, true),
            storage(2, false),
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(FRAME_BYTES),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::AccelerationStructure {
                    vertex_return: false,
                },
                count: None,
            },
            storage(5, true),
            storage(6, true),
            storage(7, true),
            storage(8, true),
            storage(9, true),
            storage(11, true),
        ];
        let occlusion_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Mesh rays (occlusion)"),
            entries: &entries,
        });
        entries.push(storage(10, false));
        let nearest_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Mesh rays (nearest)"),
            entries: &entries,
        });
        let binding = MaterialBindingConfig::for_device(device);
        let mut material_entries = vec![storage(0, true)];
        material_entries.extend(helio_mats::vg_alpha::layout_entries(
            binding,
            1,
            wgpu::ShaderStages::COMPUTE,
        ));
        let materials_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Mesh rays materials"),
            entries: &material_entries,
        });
        let module = helio_core::shader::module_with(
            device,
            "Mesh rays",
            &shader_source(binding),
            &SNIPPETS,
        );
        let pipeline = |label: &str, layout: &wgpu::BindGroupLayout, entry: &str| {
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(layout), Some(&materials_layout)],
                immediate_size: 0,
            });
            helio_core::shader::create_compute_pipeline(
                device,
                wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                },
            )
        };
        let nearest = pipeline("mesh_rays_nearest", &nearest_layout, "mesh_rays_nearest");
        let occlusion = pipeline(
            "mesh_rays_occlusion",
            &occlusion_layout,
            "mesh_rays_occlusion",
        );
        let stand_in = MaterialStandIn::new(device, queue);
        let no_materials = materials_group(
            device,
            &materials_layout,
            binding,
            stand_in.rows(),
            stand_in.textures(),
        );
        let no_proxies = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Mesh rays no proxies"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        Self {
            nearest_layout,
            occlusion_layout,
            materials_layout,
            binding,
            nearest,
            occlusion,
            no_materials,
            no_proxies,
        }
    }
}

fn materials_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    binding: MaterialBindingConfig,
    rows: &wgpu::Buffer,
    textures: MaterialTextureBindings<'_>,
) -> wgpu::BindGroup {
    let mut entries = vec![wgpu::BindGroupEntry {
        binding: 0,
        resource: rows.as_entire_binding(),
    }];
    entries.extend(helio_mats::vg_alpha::bind_entries(
        binding, 1, rows, textures,
    ));
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Mesh rays materials"),
        layout,
        entries: &entries,
    })
}

/// Group 0's inputs, by identity: header, rays, hits, instances, vertices,
/// indices, proxy vertices, proxy indices; the surfaces; the TLAS.
struct Inputs {
    buffers: [wgpu::Buffer; 9],
    surfaces: Option<wgpu::Buffer>,
    tlas: wgpu::Tlas,
}

impl Inputs {
    fn matches(
        &self,
        buffers: &[&wgpu::Buffer; 9],
        surfaces: Option<&wgpu::Buffer>,
        tlas: &wgpu::Tlas,
    ) -> bool {
        self.tlas == *tlas
            && self.surfaces.as_ref() == surfaces
            && self.buffers.iter().zip(buffers).all(|(a, b)| a == *b)
    }
}

/// The meshes' dispatch for one stream: its class's pipeline, its uniform
/// (seed and tracer id) and its bind groups, made when their inputs change.
pub struct MeshRayTracer {
    pipelines: Arc<Pipelines>,
    class: RayClass,
    tracer: TracerId,
    frame: wgpu::Buffer,
    inputs: Option<(Inputs, wgpu::BindGroup)>,
    /// The scene's material table, by what it was made of.
    materials: Option<((wgpu::Buffer, wgpu::Buffer, u64), wgpu::BindGroup)>,
}

impl MeshRayTracer {
    /// The tracer of a stream of `class` whose hits name it `tracer`, with
    /// pipelines of its own (a test, a host without a graph); `None`
    /// without hardware ray queries.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        class: RayClass,
        tracer: TracerId,
    ) -> Option<Self> {
        mesh_rays_supported(device).then(|| {
            Self::with_pipelines(
                device,
                Arc::new(Pipelines::new(device, queue)),
                class,
                tracer,
            )
        })
    }

    pub fn with_pipelines(
        device: &wgpu::Device,
        pipelines: Arc<Pipelines>,
        class: RayClass,
        tracer: TracerId,
    ) -> Self {
        Self {
            pipelines,
            class,
            tracer,
            frame: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Mesh rays frame"),
                size: FRAME_BYTES,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            inputs: None,
            materials: None,
        }
    }

    /// Writes frame `frame`'s uniform through `write` (buffer, bytes): its
    /// seed (cut-outs' stochastic test) and the tracer's id.
    pub fn write_frame(&self, frame: u64, write: impl FnOnce(&wgpu::Buffer, &[u8])) {
        let seed = (frame as u32).wrapping_mul(0x9e37_79b9) ^ ((frame >> 32) as u32);
        let uniform = GpuFrame {
            info: [seed, self.tracer.0, 0, 0],
        };
        write(&self.frame, bytemuck::bytes_of(&uniform));
    }

    /// Records the meshes' answer to `stream`'s rays: nothing without a
    /// `scene` (no TLAS or no instance rows this frame). `materials`: the
    /// scene's material rows and texture table, if published (the
    /// stand-in's otherwise).
    pub fn record(
        &mut self,
        device: &wgpu::Device,
        cmds: &mut helio_core::CommandRecorder<'_>,
        stream: StreamBindings<'_>,
        scene: Option<MeshRayScene<'_>>,
        materials: Option<(&wgpu::Buffer, MaterialTextureBindings<'_>)>,
    ) {
        let Some(scene) = scene else {
            return;
        };
        let surfaces = match self.class {
            RayClass::Nearest => match stream.surfaces {
                Some(surfaces) => Some(surfaces),
                // A nearest-hit stream without its surfaces is no stream.
                None => return,
            },
            RayClass::Occlusion => None,
        };
        let (proxy_vertices, proxy_indices) = scene
            .proxies
            .unwrap_or((&self.pipelines.no_proxies, &self.pipelines.no_proxies));
        let skinned = scene.skinned.unwrap_or(&self.pipelines.no_proxies);
        let buffers = [
            stream.header,
            stream.rays,
            stream.hits,
            scene.instances,
            scene.vertices,
            scene.indices,
            proxy_vertices,
            proxy_indices,
            skinned,
        ];
        if !self
            .inputs
            .as_ref()
            .is_some_and(|(inputs, _)| inputs.matches(&buffers, surfaces, scene.tlas))
        {
            let mut entries: Vec<wgpu::BindGroupEntry<'_>> = [0, 1, 2, 5, 6, 7, 8, 9, 11]
                .into_iter()
                .zip(buffers)
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding,
                    resource: buffer.as_entire_binding(),
                })
                .collect();
            entries.push(wgpu::BindGroupEntry {
                binding: 3,
                resource: self.frame.as_entire_binding(),
            });
            entries.push(wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::AccelerationStructure(scene.tlas),
            });
            if let Some(surfaces) = surfaces {
                entries.push(wgpu::BindGroupEntry {
                    binding: 10,
                    resource: surfaces.as_entire_binding(),
                });
            }
            let layout = match self.class {
                RayClass::Nearest => &self.pipelines.nearest_layout,
                RayClass::Occlusion => &self.pipelines.occlusion_layout,
            };
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Mesh rays"),
                layout,
                entries: &entries,
            });
            let inputs = Inputs {
                buffers: buffers.map(Clone::clone),
                surfaces: surfaces.cloned(),
                tlas: scene.tlas.clone(),
            };
            self.inputs = Some((inputs, group));
        }
        match materials {
            Some((rows, textures)) => {
                let same = self.materials.as_ref().is_some_and(|((r, t, v), _)| {
                    r == rows && t == textures.material_textures && *v == textures.version
                });
                if !same {
                    let group = materials_group(
                        device,
                        &self.pipelines.materials_layout,
                        self.pipelines.binding,
                        rows,
                        textures,
                    );
                    let key = (
                        rows.clone(),
                        textures.material_textures.clone(),
                        textures.version,
                    );
                    self.materials = Some((key, group));
                }
            }
            None => self.materials = None,
        }
        let materials = self
            .materials
            .as_ref()
            .map_or(&self.pipelines.no_materials, |(_, group)| group);
        let Some((_, group)) = &self.inputs else {
            return;
        };
        let pipeline = match self.class {
            RayClass::Nearest => &self.pipelines.nearest,
            RayClass::Occlusion => &self.pipelines.occlusion,
        };
        let mut pass = cmds.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Mesh rays"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, Some(group), &[]);
        pass.set_bind_group(1, Some(materials), &[]);
        pass.dispatch_workgroups_indirect(stream.header, 0);
    }
}

/// The meshes' pass for one stream in the graph (where the consumer's node
/// called `stream::trace`).
pub struct MeshRayPass {
    stream: Stream,
    tracer: MeshRayTracer,
    name: &'static str,
    reads: &'static [&'static str],
    writes: &'static [&'static str],
}

impl MeshRayPass {
    pub fn new(stream: Stream, tracer: MeshRayTracer) -> Self {
        let reads = helio_frame_api::static_names(&[
            stream.header,
            stream.rays,
            stream.hits,
            helio_core::resource_keys::RENDER_ENVIRONMENT,
            helio_mesh_api::RAY_INSTANCES,
            helio_mesh_api::RAY_PROXY_VERTICES,
            helio_mesh_api::RAY_PROXY_INDICES,
            helio_mesh_api::RAY_SKINNED_VERTICES,
        ]);
        let mut writes = vec![stream.hits];
        writes.extend(stream.surfaces);
        Self {
            name: helio_frame_api::static_name(&format!("MeshRays {}", stream.name)),
            reads,
            writes: helio_frame_api::static_names(&writes),
            stream,
            tracer,
        }
    }

    /// The stream this pass answers.
    pub fn stream(&self) -> &Stream {
        &self.stream
    }
}

impl RenderPass for MeshRayPass {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reads(&self) -> &'static [&'static str] {
        self.reads
    }

    fn writes(&self) -> &'static [&'static str] {
        self.writes
    }

    fn scene_buffers(&self) -> &'static [&'static str] {
        const LIST: &[&str] = &[
            helio_mesh_api::MESH_VERTICES.name(),
            helio_mesh_api::MESH_INDICES.name(),
            helio_mats::MATERIALS.name(),
        ];
        LIST
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        self.tracer.write_frame(ctx.frame_num, |buffer, bytes| {
            ctx.write_buffer(buffer, 0, bytes)
        });
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let binding = |name: &str| ctx.registry.buffer_binding(name);
        let (Some(header), Some(rays), Some(hits)) = (
            binding(self.stream.header),
            binding(self.stream.rays),
            binding(self.stream.hits),
        ) else {
            return Ok(());
        };
        let surfaces = self.stream.surfaces.and_then(binding);
        let tlas = ctx
            .registry
            .get::<helio_core::RenderEnvironment>(helio_core::resource_keys::render_environment())
            .and_then(|environment| environment.tlas);
        let instances = ctx
            .registry
            .get::<&wgpu::Buffer>(ResourceKey::new(helio_mesh_api::RAY_INSTANCES));
        let pool = |key: helio_core::BufferKey| ctx.scene_buffers.get(key).map(|h| &h.buffer);
        let vertices = pool(helio_mesh_api::MESH_VERTICES.key());
        let indices = pool(helio_mesh_api::MESH_INDICES.key());
        let proxy = |name: &'static str| ctx.registry.get::<&wgpu::Buffer>(ResourceKey::new(name));
        let proxies =
            proxy(helio_mesh_api::RAY_PROXY_VERTICES).zip(proxy(helio_mesh_api::RAY_PROXY_INDICES));
        let scene = match (tlas, instances, vertices, indices) {
            (Some(tlas), Some(instances), Some(vertices), Some(indices)) => Some(MeshRayScene {
                tlas,
                instances,
                vertices,
                indices,
                proxies,
                skinned: proxy(helio_mesh_api::RAY_SKINNED_VERTICES),
            }),
            _ => None,
        };
        let materials = pool(helio_mats::MATERIALS.key()).zip(
            ctx.registry
                .get::<MaterialTextureBindings<'_>>(helio_mats::material_textures()),
        );
        let mut cmds = ctx.graphics_cmds();
        self.tracer.record(
            ctx.device,
            &mut cmds,
            StreamBindings {
                header,
                rays,
                hits,
                surfaces,
            },
            scene,
            materials,
        );
        Ok(())
    }
}

#[cfg(test)]
mod shader_tests {
    use super::*;

    #[test]
    fn the_shader_validates_in_both_binding_modes() {
        use helio_mats::MaterialBindingMode;
        for binding in [
            MaterialBindingConfig {
                mode: MaterialBindingMode::BindingArray,
                max_textures: 256,
            },
            MaterialBindingConfig {
                mode: MaterialBindingMode::Expanded,
                max_textures: 16,
            },
        ] {
            let source = shader_source(binding);
            let text = helio_core::shader::resolve_with(&source, &SNIPPETS);
            let module = wgpu::naga::front::wgsl::parse_str(&text)
                .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&text)));
            wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&text)));
            // Every ray runs it: no table indexed through a value (which
            // copies the whole table per invocation).
            let copies = helio_core::shader::value_array_copies(&text).unwrap();
            assert!(copies.is_empty(), "{copies:#?}");
            for entry in ["mesh_rays_nearest", "mesh_rays_occlusion"] {
                assert!(
                    module.entry_points.iter().any(|e| e.name == entry),
                    "no {entry}"
                );
            }
        }
    }
}
