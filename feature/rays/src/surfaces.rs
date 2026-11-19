//! The surfaces of a stream's meshes hits: each hit's material evaluated at
//! the hit, as the G-buffer's drawers evaluate it.
//!
//! **Programs.** A traced section's row names the program its material is
//! evaluated by ([`helio_mesh_api::RayInstance::program`], an index of the
//! frame's [`helio_mesh_api::RayPrograms`], which the projection publishes
//! with the TLAS): a material class's template, and for a material graph
//! the graph, keyed as the drawers key their pipelines
//! (`helio_mats::RadiantShaderKey` without raster flags). Each program is
//! the meshes' hit program ([`HIT_PROGRAM`]) with the template and graph
//! composed in (`RadiantTemplate::compose`, the drawers' composition),
//! lowered to explicit gradients (`helio_mats::shader::explicit_gradients`:
//! textures read at the ray cone's footprint). Program 0, the table-driven
//! evaluation (the default class, no graph: the row's factors and
//! textures), is built with the graph; every other program compiles off the
//! frame (`helio_core::compile::Compiling`) the first frame a row names it.
//!
//! **Detail.** One module a program, two pipelines
//! (`MATERIAL_SURFACE_DIFFUSE`): a [`SurfaceDetail::Diffuse`] stream reads
//! base colour and emission at the footprint with the vertex normal and the
//! material's constant roughness and metalness; a [`SurfaceDetail::Full`]
//! stream the whole material (normal maps, ORM, specular anti-aliasing at
//! the footprint). A program still runs at `Diffuse` (a graph's base colour
//! is the graph's), its unread inputs compiled away.
//!
//! **Binning.** Where every hit's program is the table-driven one (no other
//! program in the frame, or none ready yet), one dispatch over the stream's
//! rows writes them all. Otherwise the hits are binned by program
//! (`mesh_hit_bins.wgsl`: a count with workgroup-local histograms, a prefix
//! over the bins, a scatter into each bin's list) and each bin's program
//! runs over its own list by an indirect dispatch: the wavefront renderers'
//! sort by material and what shader execution reordering does in hardware,
//! as bins instead of a sort. At most [`MAX_HIT_PROGRAMS`] bins a frame, bin
//! 0 the table-driven one: a program not compiled yet, one that failed, and
//! programs past the bound are evaluated by the table-driven program (their
//! row's factors and textures), never left without a surface.
//!
//! Cost: the table-driven dispatch alone is one pass over the stream; with
//! programs, three binning passes plus one dispatch a bin (empty bins
//! dispatch no workgroups), each kernel over its own hits only.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use helio_core::compile::Compiling;
use helio_core::{PassContext, PrepareContext, RenderPass, ResourceKey, Result as HelioResult};
use helio_frame_api::rays::stream::{Stream, SurfaceDetail, TracerId};
use helio_mats::{
    MaterialBindingConfig, MaterialStandIn, MaterialTextureBindings, RadiantTemplate,
    SharedTemplateRegistry,
};
use helio_mesh_api::{RayProgram, RayPrograms};

use crate::{storage, StreamBindings};

/// The most bins a stream's hits take in a frame (bin 0 the table-driven
/// evaluation): programs past it are evaluated by the table-driven program
/// that frame. Matches `HIT_BINS` in the shaders.
pub const MAX_HIT_PROGRAMS: usize = 64;
/// `u32` words of a bin (`mesh_hit_bins.wgsl`).
const BIN_WORDS: u64 = 8;
/// The bin-table entry of a dispatch over every row of the stream.
const UNBINNED: u32 = u32::MAX;
/// The bin table's stride (the dynamic uniform offset alignment).
const BIN_STRIDE: u64 = 256;

/// The meshes' hit program (`shaders/mesh_hit_surface.wgsl`): a template's
/// surface function and a graph are composed after it.
pub const HIT_PROGRAM: &str = include_str!("../shaders/mesh_hit_surface.wgsl");
/// The snippets [`HIT_PROGRAM`] composes with besides the materials'.
pub const HIT_PROGRAM_SNIPPETS: [helio_core::shader::ShaderSnippet; 4] = [
    helio_frame_api::rays::stream::SNIPPET,
    helio_frame_api::rays::stream::SURFACE_SNIPPET,
    helio_mesh_api::RAY_ROWS_SNIPPET,
    helio_mesh_api::RAY_SURFACE_SNIPPET,
];
const BINS_SHADER: &str = include_str!("../shaders/mesh_hit_bins.wgsl");
const BINS_SNIPPETS: [helio_core::shader::ShaderSnippet; 2] = [
    helio_frame_api::rays::stream::SNIPPET,
    helio_mesh_api::RAY_ROWS_SNIPPET,
];

/// The graph's [`HitPrograms`], made by the first stream's surface pass.
pub(crate) struct SharedHitPrograms(pub Option<Arc<HitPrograms>>);

/// A program's pipelines, one a surface detail.
#[derive(Clone)]
pub struct ProgramPipelines {
    diffuse: wgpu::ComputePipeline,
    full: wgpu::ComputePipeline,
}

impl ProgramPipelines {
    fn of(&self, detail: SurfaceDetail) -> &wgpu::ComputePipeline {
        match detail {
            SurfaceDetail::Diffuse => &self.diffuse,
            SurfaceDetail::Full => &self.full,
        }
    }
}

/// The source of program `program` composed from `template` and `graph`
/// (empty: none), lowered for `binding` and to explicit gradients: what a
/// hit program compiles.
pub fn program_source(
    template: &RadiantTemplate,
    graph: &str,
    binding: MaterialBindingConfig,
) -> String {
    let composed = template.compose(
        HIT_PROGRAM,
        graph,
        binding.max_textures,
        &HIT_PROGRAM_SNIPPETS,
    );
    let bindless = binding.uses_binding_arrays();
    let composed = if bindless {
        composed
    } else {
        RadiantTemplate::apply_webgpu_fixups(&composed, binding.max_textures)
    };
    helio_mats::shader::explicit_gradients(&composed, binding.max_textures, bindless)
}

fn build_program(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    source: String,
    label: &str,
) -> ProgramPipelines {
    // Runtime-composed WGSL (template, graph, lowering): no source file to
    // watch, as the drawers' programs (`RadiantShaderCache`).
    #[allow(clippy::disallowed_methods)]
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = |diffuse: bool| {
        helio_core::shader::create_compute_pipeline(
            device,
            wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                module: &module,
                entry_point: Some("mesh_hit_surface"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("MATERIAL_SURFACE_DIFFUSE", if diffuse { 1.0 } else { 0.0 })],
                    ..Default::default()
                },
                cache: None,
            },
        )
    };
    ProgramPipelines {
        diffuse: pipeline(true),
        full: pipeline(false),
    }
}

/// What a program compiles from, or why it has nothing to compile yet.
enum Source {
    Ready(RadiantTemplate, String),
    /// A graph whose source is not registered yet: asked again next frame.
    Waiting,
    /// The table-driven program (a template the registry lacks).
    Table,
}

/// Every hit program of a graph (one set, shared by the streams' surface
/// passes): the layouts, the binning pipelines, the table-driven program
/// and the programs compiling or compiled.
pub struct HitPrograms {
    binding: MaterialBindingConfig,
    templates: Option<SharedTemplateRegistry>,
    surface_layout: wgpu::BindGroupLayout,
    materials_layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    bins_layout: wgpu::BindGroupLayout,
    count: wgpu::ComputePipeline,
    offsets: wgpu::ComputePipeline,
    scatter: wgpu::ComputePipeline,
    table: ProgramPipelines,
    compiling: Mutex<HashMap<RayProgram, Arc<Compiling<ProgramPipelines>>>>,
    /// The bin each dynamic offset names: entry `k` (at `k * BIN_STRIDE`)
    /// holds `k`, the last [`UNBINNED`].
    bin_table: wgpu::Buffer,
    stand_in: MaterialStandIn,
    /// A binding for each absent pool.
    no_pool: wgpu::Buffer,
}

impl HitPrograms {
    /// The layouts, the binning and the table-driven program (built now);
    /// `templates`, the materials' opaque registry (the built-ins without).
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        templates: Option<SharedTemplateRegistry>,
    ) -> Self {
        let binding = MaterialBindingConfig::for_device(device);
        let uniform = |binding, dynamic| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: dynamic,
                min_binding_size: wgpu::BufferSize::new(16),
            },
            count: None,
        };
        let mut entries: Vec<_> = (0..12).map(|b| storage(b, b != 3)).collect();
        entries.push(uniform(12, false));
        entries.push(uniform(13, true));
        entries.push(storage(14, true));
        let surface_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Mesh hit surfaces"),
            entries: &entries,
        });
        let mut material_entries = vec![storage(0, true), storage(1, true)];
        binding.append_layout_entries(&mut material_entries, 2, wgpu::ShaderStages::COMPUTE);
        let materials_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Mesh hit surfaces materials"),
            entries: &material_entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Mesh hit surfaces"),
            bind_group_layouts: &[Some(&surface_layout), Some(&materials_layout)],
            immediate_size: 0,
        });
        let bins_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Mesh hit bins"),
            entries: &[
                storage(0, true),
                storage(1, true),
                storage(2, true),
                storage(3, true),
                storage(4, false),
                storage(5, false),
                uniform(6, false),
                storage(7, false),
                storage(8, false),
            ],
        });
        let bins_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Mesh hit bins"),
            bind_group_layouts: &[Some(&bins_layout)],
            immediate_size: 0,
        });
        let bins_module =
            helio_core::shader::module_with(device, "Mesh hit bins", BINS_SHADER, &BINS_SNIPPETS);
        let bins_pipeline = |entry: &str| {
            helio_core::shader::create_compute_pipeline(
                device,
                wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&bins_pipeline_layout),
                    module: &bins_module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                },
            )
        };
        let table_template = template_of(templates.as_ref(), 0).unwrap_or_else(default_template);
        let table = build_program(
            device,
            &pipeline_layout,
            program_source(&table_template, "", binding),
            "Mesh hit surfaces (table)",
        );
        let bin_table = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Mesh hit bin table"),
            size: (MAX_HIT_PROGRAMS as u64 + 1) * BIN_STRIDE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut table_words = vec![0u32; ((MAX_HIT_PROGRAMS as u64 + 1) * BIN_STRIDE / 4) as usize];
        for k in 0..=MAX_HIT_PROGRAMS {
            table_words[k * (BIN_STRIDE / 4) as usize] = if k == MAX_HIT_PROGRAMS {
                UNBINNED
            } else {
                k as u32
            };
        }
        helio_core::upload::write_buffer(queue, &bin_table, 0, bytemuck::cast_slice(&table_words));
        Self {
            binding,
            templates,
            surface_layout,
            materials_layout,
            pipeline_layout,
            bins_layout,
            count: bins_pipeline("mesh_hit_count"),
            offsets: bins_pipeline("mesh_hit_offsets"),
            scatter: bins_pipeline("mesh_hit_scatter"),
            table,
            compiling: Mutex::new(HashMap::new()),
            bin_table,
            stand_in: MaterialStandIn::new(device, queue),
            no_pool: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Mesh hit surfaces no pool"),
                size: 48,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
        }
    }

    /// What `program` compiles from.
    fn source(&self, program: RayProgram) -> Source {
        let Some(template) = template_of(self.templates.as_ref(), program.template)
            .or_else(|| {
                // The built-in classes without a registry; a custom class
                // without its registry is the table's.
                (program.template < helio_mats::FIRST_CUSTOM_MATERIAL_CLASS)
                    .then(|| {
                        helio_mats::RadiantTemplateRegistry::new()
                            .get(program.template)
                            .cloned()
                    })
                    .flatten()
            })
            // A class no registry holds with a graph (a material graph's own
            // class, `MATERIAL_CLASS_CUSTOM`): the graph over the default
            // class's template, as the drawers compose it.
            .or_else(|| {
                (program.graph_hash != 0).then(|| {
                    template_of(self.templates.as_ref(), 0).unwrap_or_else(default_template)
                })
            })
        else {
            return Source::Table;
        };
        if program.graph_hash == 0 {
            return Source::Ready(template, String::new());
        }
        match helio_mats::graph_source(program.graph_hash) {
            Some(graph) => Source::Ready(template, graph),
            None => Source::Waiting,
        }
    }

    /// `program`'s pipelines once compiled (its compile started the first
    /// time it is asked for); `None` while it compiles, waits for its graph,
    /// failed, or is the table-driven program.
    pub fn ready(&self, device: &wgpu::Device, program: RayProgram) -> Option<ProgramPipelines> {
        if program == RayProgram::TABLE {
            return None;
        }
        let mut compiling = self
            .compiling
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(job) = compiling.get(&program) {
            return job.ready().cloned();
        }
        let Source::Ready(template, graph) = self.source(program) else {
            return None;
        };
        let (layout, binding) = (self.pipeline_layout.clone(), self.binding);
        let label = format!(
            "Mesh hit surfaces (template {}, graph {:016x})",
            program.template, program.graph_hash
        );
        let job = Compiling::start(device, label.clone(), move |device| {
            build_program(
                device,
                &layout,
                program_source(&template, &graph, binding),
                &label,
            )
        });
        compiling.insert(program, Arc::new(job));
        None
    }

    /// Whether `program` waits for its graph's source (not registered yet).
    fn waits_for_graph(&self, program: RayProgram) -> bool {
        program.graph_hash != 0
            && !self
                .compiling
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .contains_key(&program)
            && matches!(self.source(program), Source::Waiting)
    }

    /// Programs compiling and programs whose compile failed.
    pub fn compile_states(&self) -> (usize, usize) {
        let compiling = self
            .compiling
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let pending = compiling.values().filter(|job| job.is_pending()).count();
        let failed = compiling
            .values()
            .filter(|job| job.failed().is_some())
            .count();
        (pending, failed)
    }

    /// Blocks until every program asked for so far has compiled: for tests
    /// and offline tools, never a frame.
    pub fn wait(&self) {
        let jobs: Vec<_> = self
            .compiling
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .values()
            .cloned()
            .collect();
        for job in jobs {
            let _ = job.wait();
        }
    }
}

fn template_of(templates: Option<&SharedTemplateRegistry>, class: u32) -> Option<RadiantTemplate> {
    templates?.read().ok()?.get(class).cloned()
}

fn default_template() -> RadiantTemplate {
    helio_mats::RadiantTemplateRegistry::new()
        .get(0)
        .cloned()
        .expect("the default material class's template")
}

/// The meshes as a hit's surface reads them this frame: the instance rows
/// and transforms, the scene's mesh pools, the proxy pool and the skin
/// cache, if any.
#[derive(Clone, Copy)]
pub struct MeshHitScene<'a> {
    pub instances: &'a wgpu::Buffer,
    pub transforms: &'a wgpu::Buffer,
    pub vertices: &'a wgpu::Buffer,
    pub indices: &'a wgpu::Buffer,
    pub proxies: Option<(&'a wgpu::Buffer, &'a wgpu::Buffer)>,
    pub skinned: Option<&'a wgpu::Buffer>,
}

/// One stream's surface passes: its detail, this frame's bins (bin `k`'s
/// program), the program-to-bin table, the binning buffers and bind
/// groups, made when their inputs change.
pub struct MeshHitSurfaces {
    programs: Arc<HitPrograms>,
    detail: SurfaceDetail,
    tracer: TracerId,
    /// x: the tracer's id, y: the frame.
    frame: wgpu::Buffer,
    /// Program index to bin, as last uploaded.
    remap: wgpu::Buffer,
    remap_rows: Vec<u32>,
    /// This frame's bins: bin 0 the table-driven program, bin `k` the
    /// program of its entry.
    bins: Vec<ProgramPipelines>,
    /// This frame's programs handed over, and those waiting for their
    /// graph's source (frame statistics).
    seen: usize,
    waiting: usize,
    /// The ranks and the list, for a stream of the capacity they were made
    /// for.
    binning: Option<(u64, wgpu::Buffer, wgpu::Buffer)>,
    /// Each bin's indirect arguments, list offset and count.
    bin_args: wgpu::Buffer,
    /// The count pass's counter a bin.
    counters: wgpu::Buffer,
    inputs: Option<(Vec<wgpu::Buffer>, wgpu::BindGroup, wgpu::BindGroup)>,
    materials: Option<((wgpu::Buffer, wgpu::Buffer, u64), wgpu::BindGroup)>,
}

impl MeshHitSurfaces {
    /// The surface passes of a stream needing `detail`, whose hits name the
    /// meshes `tracer`.
    pub fn new(
        device: &wgpu::Device,
        programs: Arc<HitPrograms>,
        detail: SurfaceDetail,
        tracer: TracerId,
    ) -> Self {
        let table = programs.table.clone();
        Self {
            programs,
            detail,
            tracer,
            frame: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Mesh hit surfaces frame"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            remap: remap_buffer(device, 1),
            remap_rows: vec![0],
            bins: vec![table],
            seen: 0,
            waiting: 0,
            binning: None,
            bin_args: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Mesh hit bins"),
                size: MAX_HIT_PROGRAMS as u64 * BIN_WORDS * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
                mapped_at_creation: false,
            }),
            counters: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Mesh hit bin counters"),
                size: MAX_HIT_PROGRAMS as u64 * 4,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
            inputs: None,
            materials: None,
        }
    }

    /// Writes frame `frame`'s uniform through `write` (buffer, bytes).
    pub fn write_frame(&self, frame: u64, write: impl FnOnce(&wgpu::Buffer, &[u8])) {
        let words = [self.tracer.0, frame as u32, 0, 0];
        write(&self.frame, bytemuck::cast_slice(&words));
    }

    /// Takes this frame's programs (`None`: the table-driven one alone):
    /// starts the compiles of those not asked for yet, gives each compiled
    /// one a bin while there are bins, and writes the program-to-bin table
    /// through `write` (buffer, offset, bytes) when it changed.
    pub fn set_programs(
        &mut self,
        device: &wgpu::Device,
        programs: Option<&RayPrograms>,
        write: impl FnOnce(&wgpu::Buffer, u64, &[u8]),
    ) {
        let list = programs.map_or(&[RayProgram::TABLE][..], RayPrograms::programs);
        let mut rows = vec![0u32; list.len().max(1)];
        self.bins.truncate(1);
        self.seen = programs.map_or(0, |p| p.programs().len());
        self.waiting = 0;
        for (index, program) in list.iter().enumerate() {
            let Some(pipelines) = self.programs.ready(device, *program) else {
                self.waiting += usize::from(self.programs.waits_for_graph(*program));
                continue;
            };
            if self.bins.len() < MAX_HIT_PROGRAMS {
                rows[index] = self.bins.len() as u32;
                self.bins.push(pipelines);
            }
        }
        if rows != self.remap_rows {
            if self.remap.size() < (rows.len() * 4) as u64 {
                self.remap = remap_buffer(device, rows.len());
            }
            write(&self.remap, 0, bytemuck::cast_slice(&rows));
            self.remap_rows = rows;
        }
    }

    /// The bins this frame (1: every hit takes the table-driven program,
    /// in one dispatch with no binning).
    pub fn bin_count(&self) -> usize {
        self.bins.len()
    }

    /// Records the surfaces of `stream`'s hits that are this tracer's:
    /// nothing without a `scene` or the stream's surfaces. `materials`: the
    /// scene's material rows and texture table, if published (the
    /// stand-in's otherwise).
    pub fn record(
        &mut self,
        device: &wgpu::Device,
        cmds: &mut helio_core::CommandRecorder<'_>,
        stream: StreamBindings<'_>,
        scene: Option<MeshHitScene<'_>>,
        materials: Option<(&wgpu::Buffer, MaterialTextureBindings<'_>)>,
    ) {
        let (Some(scene), Some(surfaces)) = (scene, stream.surfaces) else {
            return;
        };
        let capacity = (stream.rays.size() / 32).max(1);
        if self.binning.as_ref().is_none_or(|(c, _, _)| *c != capacity) {
            let buffer = |label| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: capacity * 4,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                })
            };
            self.binning = Some((capacity, buffer("Mesh hit ranks"), buffer("Mesh hit list")));
            self.inputs = None;
        }
        let (_, ranks, list) = self.binning.as_ref().expect("made above");
        let bins = &self.bin_args;
        let programs = &self.programs;
        let (proxy_vertices, proxy_indices) = scene
            .proxies
            .unwrap_or((&programs.no_pool, &programs.no_pool));
        let skinned = scene.skinned.unwrap_or(&programs.no_pool);
        let buffers = [
            stream.header,
            stream.rays,
            stream.hits,
            surfaces,
            scene.instances,
            scene.transforms,
            scene.vertices,
            scene.indices,
            proxy_vertices,
            proxy_indices,
            skinned,
            bins,
            &self.remap,
            list,
        ];
        if !self.inputs.as_ref().is_some_and(|(key, _, _)| {
            key.len() == buffers.len() && key.iter().zip(&buffers).all(|(a, b)| a == *b)
        }) {
            let mut entries: Vec<wgpu::BindGroupEntry<'_>> = buffers[..12]
                .iter()
                .enumerate()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect();
            entries.push(wgpu::BindGroupEntry {
                binding: 12,
                resource: self.frame.as_entire_binding(),
            });
            entries.push(wgpu::BindGroupEntry {
                binding: 13,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &programs.bin_table,
                    offset: 0,
                    size: wgpu::BufferSize::new(16),
                }),
            });
            entries.push(wgpu::BindGroupEntry {
                binding: 14,
                resource: list.as_entire_binding(),
            });
            let surface_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Mesh hit surfaces"),
                layout: &programs.surface_layout,
                entries: &entries,
            });
            let bins_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Mesh hit bins"),
                layout: &programs.bins_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: stream.header.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: stream.hits.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: scene.instances.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.remap.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: bins.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: ranks.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: self.frame.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: self.counters.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: list.as_entire_binding(),
                    },
                ],
            });
            self.inputs = Some((
                buffers.iter().map(|b| (*b).clone()).collect(),
                surface_group,
                bins_group,
            ));
        }
        let (rows, textures) =
            materials.unwrap_or((programs.stand_in.rows(), programs.stand_in.textures()));
        let key = (
            rows.clone(),
            textures.material_textures.clone(),
            textures.version,
        );
        if self.materials.as_ref().is_none_or(|(k, _)| *k != key) {
            let mut entries = vec![
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: rows.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: textures.material_textures.as_entire_binding(),
                },
            ];
            programs.binding.append_bind_group_entries(
                &mut entries,
                2,
                textures.texture_views,
                textures.sampler,
            );
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Mesh hit surfaces materials"),
                layout: &programs.materials_layout,
                entries: &entries,
            });
            self.materials = Some((key, group));
        }
        let (Some((_, surface_group, bins_group)), Some((_, materials))) =
            (&self.inputs, &self.materials)
        else {
            return;
        };
        let mut pass = cmds.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Mesh hit surfaces"),
            timestamp_writes: None,
        });
        pass.set_bind_group(1, Some(materials), &[]);
        if self.bins.len() == 1 {
            // Every hit's program is the table-driven one: no binning.
            pass.set_pipeline(self.bins[0].of(self.detail));
            pass.set_bind_group(
                0,
                Some(surface_group),
                &[(MAX_HIT_PROGRAMS as u64 * BIN_STRIDE) as u32],
            );
            pass.dispatch_workgroups_indirect(stream.header, 0);
            return;
        }
        pass.set_bind_group(0, Some(bins_group), &[]);
        pass.set_pipeline(&programs.count);
        pass.dispatch_workgroups_indirect(stream.header, 0);
        pass.set_pipeline(&programs.offsets);
        pass.dispatch_workgroups(1, 1, 1);
        pass.set_pipeline(&programs.scatter);
        pass.dispatch_workgroups_indirect(stream.header, 0);
        pass.set_bind_group(1, Some(materials), &[]);
        for (k, program) in self.bins.iter().enumerate() {
            pass.set_pipeline(program.of(self.detail));
            pass.set_bind_group(0, Some(surface_group), &[(k as u64 * BIN_STRIDE) as u32]);
            pass.dispatch_workgroups_indirect(bins, k as u64 * BIN_WORDS * 4);
        }
    }
}

fn remap_buffer(device: &wgpu::Device, rows: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Mesh hit program bins"),
        size: (rows.max(1).next_power_of_two() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// The meshes' surface passes of one nearest-hit stream in the graph
/// (after every tracer traced it, `stream::trace`).
pub struct MeshHitSurfacePass {
    stream: Stream,
    surfaces: MeshHitSurfaces,
    name: &'static str,
    reads: &'static [&'static str],
    writes: &'static [&'static str],
}

impl MeshHitSurfacePass {
    pub fn new(stream: Stream, surfaces: MeshHitSurfaces) -> Self {
        let reads = helio_frame_api::static_names(&[
            stream.header,
            stream.rays,
            stream.hits,
            helio_mesh_api::RAY_INSTANCES,
            helio_mesh_api::RAY_INSTANCE_TRANSFORMS,
            helio_mesh_api::RAY_PROXY_VERTICES,
            helio_mesh_api::RAY_PROXY_INDICES,
            helio_mesh_api::RAY_SKINNED_VERTICES,
            helio_mats::MATERIAL_TEXTURES,
        ]);
        Self {
            name: helio_frame_api::static_name(&format!("MeshHitSurfaces {}", stream.name)),
            reads,
            writes: helio_frame_api::static_names(&stream.surfaces.into_iter().collect::<Vec<_>>()),
            stream,
            surfaces,
        }
    }
}

impl RenderPass for MeshHitSurfacePass {
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
        self.surfaces.write_frame(ctx.frame_num, |buffer, bytes| {
            ctx.write_buffer(buffer, 0, bytes)
        });
        let programs = ctx.scene_buffers.payload(&helio_mesh_api::RAY_PROGRAMS);
        self.surfaces
            .set_programs(ctx.device, programs.as_deref(), |buffer, offset, bytes| {
                ctx.write_buffer(buffer, offset, bytes)
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
        let buffer = |name: &'static str| ctx.registry.get::<&wgpu::Buffer>(ResourceKey::new(name));
        let pool = |key: helio_core::BufferKey| ctx.scene_buffers.get(key).map(|h| &h.buffer);
        let scene = match (
            buffer(helio_mesh_api::RAY_INSTANCES),
            buffer(helio_mesh_api::RAY_INSTANCE_TRANSFORMS),
            pool(helio_mesh_api::MESH_VERTICES.key()),
            pool(helio_mesh_api::MESH_INDICES.key()),
        ) {
            (Some(instances), Some(transforms), Some(vertices), Some(indices)) => {
                Some(MeshHitScene {
                    instances,
                    transforms,
                    vertices,
                    indices,
                    proxies: buffer(helio_mesh_api::RAY_PROXY_VERTICES)
                        .zip(buffer(helio_mesh_api::RAY_PROXY_INDICES)),
                    skinned: buffer(helio_mesh_api::RAY_SKINNED_VERTICES),
                })
            }
            _ => None,
        };
        let materials = pool(helio_mats::MATERIALS.key()).zip(
            ctx.registry
                .get::<MaterialTextureBindings<'_>>(helio_mats::material_textures()),
        );
        if let Some(stats) = ctx.stats() {
            let (pending, failed) = self.surfaces.programs.compile_states();
            stats.add_named(
                helio_mesh_api::stats::HIT_BINS,
                self.surfaces.bin_count() as u64,
            );
            stats.add_named(helio_mesh_api::stats::HIT_PROGRAMS_PENDING, pending as u64);
            stats.add_named(helio_mesh_api::stats::HIT_PROGRAMS_FAILED, failed as u64);
            stats.add_named(
                helio_mesh_api::stats::HIT_PROGRAMS_SEEN,
                self.surfaces.seen as u64,
            );
            stats.add_named(
                helio_mesh_api::stats::HIT_PROGRAMS_WAITING,
                self.surfaces.waiting as u64,
            );
        }
        let mut cmds = ctx.graphics_cmds();
        self.surfaces.record(
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
    use helio_mats::MaterialBindingMode;

    fn validate(name: &str, text: &str) -> wgpu::naga::Module {
        let module = wgpu::naga::front::wgsl::parse_str(text)
            .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(text)));
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(text)));
        module
    }

    /// Every built-in material class (and the opaque custom templates)
    /// composes into the hit program, in both binding modes, with no
    /// implicit derivative left (a compute stage has none).
    #[test]
    fn every_material_class_composes_into_the_hit_program() {
        let mut registry = helio_mats::RadiantTemplateRegistry::new();
        for (name, source) in [
            ("opal", helio_mats::templates::OPAL),
            ("glass", helio_mats::templates::GLASS),
            ("water", helio_mats::templates::WATER),
        ] {
            registry.register_str(name, source.to_string());
        }
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
            for class in registry.keys() {
                let template = registry.get(class).unwrap();
                let text = program_source(template, "", binding);
                for implicit in ["textureSample(", "dpdx(", "dpdy(", "fwidth("] {
                    assert!(
                        !text.contains(implicit),
                        "{}: {implicit} left in the hit program",
                        template.name
                    );
                }
                let module = validate(&template.name, &text);
                assert!(module
                    .entry_points
                    .iter()
                    .any(|e| e.name == "mesh_hit_surface"));
                assert!(module
                    .overrides
                    .iter()
                    .any(|(_, o)| o.name.as_deref() == Some("MATERIAL_SURFACE_DIFFUSE")));
            }
        }
    }

    #[test]
    fn the_binning_shader_validates() {
        let text = helio_core::shader::resolve_with(BINS_SHADER, &BINS_SNIPPETS);
        let module = validate("bins", &text);
        for entry in ["mesh_hit_count", "mesh_hit_offsets", "mesh_hit_scatter"] {
            assert!(
                module.entry_points.iter().any(|e| e.name == entry),
                "no {entry}"
            );
        }
        // The shaders' bin layout is this file's.
        assert!(text.contains(&format!("const HIT_BINS: u32 = {MAX_HIT_PROGRAMS}u;")));

        assert!(HIT_PROGRAM.contains(&format!("const HIT_BINS: u32 = {MAX_HIT_PROGRAMS}u;")));
    }
}
