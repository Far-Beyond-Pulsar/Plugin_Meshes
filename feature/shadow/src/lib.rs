//! Packed, budgeted shadow atlases with persistent static and dynamic caches.
//! CPU scheduling scans a fixed resident pool. GPU dirty bits gate cached draws.
//! Which faces to visit comes from the dirty bits themselves, read back
//! without waiting (`DIRTY_LINGER_FRAMES`) beside what the CPU knows (a face
//! never drawn, a light or a static caster that changed): a face nothing has
//! dirtied costs no commands, and a frame with none opens no pass. The visited
//! faces' pages are cleared in one draw per layer (`depth_clear.wgsl`), their
//! casters drawn into the tile (a viewport, the face's constants and one
//! multi-draw each), and their matrices published and dirty state cleared in
//! one dispatch (`retire.wgsl`), only after the matching depth commands.
//! Pending dirty bits survive deferral.

pub mod feature;
pub use feature::ShadowDrawers;

use helio_core::graph::ResourceBuilder;
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

mod transmittance;
pub use transmittance::{TRANSMITTANCE_FORMAT, TRANSMITTANCE_KEY};

// ── Constants ─────────────────────────────────────────────────────────────────

/// Maximum logical faces in the bounded residency pool.
const MAX_SHADOW_FACES: usize = helio_light_api::MAX_SHADOW_FACES;

/// Byte stride between consecutive face-index entries in `face_idx_buf`.
///
/// Must satisfy `device.limits().min_uniform_buffer_offset_alignment`, which is
/// guaranteed to be ≤ 256 on every wgpu backend (Metal, Vulkan, DX12, WebGPU).
const FACE_BUF_STRIDE: u64 = 256;
/// Faces serviced each frame outside the shadow update budget: those whose
/// cached static layer is current, so only the GPU's dirty flags decide
/// whether they draw anything.
const MAX_UNMETERED_FACES_PER_FRAME: usize = 192;
/// Frames a face stays in the serviced set after the GPU's dirty bits last
/// showed it pending. Which faces are dirty is decided on the GPU
/// (`ShadowDirty`) and reaches the CPU through a readback that trails by about
/// two frames (the copy, then the map), so a face is first visited that late.
/// A dirty bit stays set until its face is serviced, so nothing is lost by
/// the delay; a face that keeps being dirtied (a moving caster, a sun cascade
/// following the camera) stays in the set and is drawn the frame the GPU
/// dirties it. Twice the readback's latency, so one missed sample does not
/// drop a face that is still moving.
const DIRTY_LINGER_FRAMES: u8 = 4;

/// Regions of the face-index buffer, each `MAX_SHADOW_FACES` slots. The slot's
/// second word is the `ShadowDirty` bit its layer draws only when set (see
/// `FACE_REGION_GATES`); `0` draws unconditionally.
const FACE_REGIONS: usize = 3;
/// Pages per face (an 8x8 grid): the instances of the depth-clear draw.
const PAGES_PER_FACE: u32 = 64;
/// Threads of one workgroup of the retire dispatch (`retire.wgsl`).
const RETIRE_WORKGROUP: u32 = 64;
/// Region 0: ungated (the face's cached static layer is being rebuilt).
/// Region 1: gated on the dynamic bit. Region 2: gated on the static bit.
const FACE_REGION_GATES: [u32; FACE_REGIONS] = [0, 1, 2];

/// Dynamic offset into the face-index buffer for one layer of `face`. A face
/// whose static layer is being rebuilt draws both layers unconditionally;
/// otherwise each layer draws only if `ShadowDirty` marked that layer, so a
/// moving movable caster redraws the dynamic layer and leaves the cached
/// static layer (and glass, which follows it) alone.
fn face_idx_offset(face: usize, static_dirty: bool, static_layer: bool) -> u32 {
    let region = if static_dirty {
        0
    } else if static_layer {
        2
    } else {
        1
    };
    ((region * MAX_SHADOW_FACES + face) as u64 * FACE_BUF_STRIDE) as u32
}

/// Regions of `ShadowDirty`'s per-face draw counts (`face_geom_count_buf`),
/// each `MAX_SHADOW_FACES` `u32`s.
const MOVABLE_COUNTS: usize = 0;
const STATIC_COUNTS: usize = 1;
const TRANSMISSIVE_COUNTS: usize = 2;

/// Byte offset of `face`'s count in a region of `face_geom_count_buf`.
fn face_count_offset(region: usize, face: usize) -> u64 {
    (region * MAX_SHADOW_FACES + face) as u64 * 4
}

/// Bytes of one job of the depth-clear draw: face, gate, tile origin, tile size.
const CLEAR_JOB_BYTES: u64 = 16;
/// Jobs of the clear draw: both layers of every face, after a header entry.
const CLEAR_JOB_CAPACITY: u64 = 1 + 2 * MAX_SHADOW_FACES as u64;
/// Words of the retire list: the count, the region size, then every face.
const RETIRE_WORD_CAPACITY: u64 = 2 + MAX_SHADOW_FACES as u64;

/// A copy of `face_dirty_buf` on its way to the CPU: the buffer, whether its
/// map finished, and whether the map was started (it is, the frame after the
/// copy was submitted).
struct DirtyReadback {
    buffer: helio_core::gpu_cache::PooledBuffer,
    ready: Arc<AtomicBool>,
    mapping: bool,
}

/// The lists a frame builds, owned by the pass so they are allocated once.
#[derive(Default)]
struct Scratch {
    /// (face, rebuilds the static layer, urgency), then ranked.
    candidates: Vec<(usize, bool, f32)>,
    /// The faces serviced this frame and whether each rebuilds its static layer.
    selected: Vec<(usize, bool)>,
    /// The depth-clear draw's jobs, as written to `clear_jobs`.
    clear: Vec<[u32; 4]>,
    /// The retire dispatch's words, as written to `retire_jobs`.
    retire: Vec<u32>,
}

// ── Pass struct ───────────────────────────────────────────────────────────────

/// The caster program's geometry (`shadow.wgsl`) with the lights' shadow
/// views (`//!use shadow_atlas`: `LightMatrix`) resolved, ahead of the depth
/// materials (`DepthMaterialBindings::shader_source`).
pub fn caster_geometry() -> String {
    helio_core::shader::resolve_with(
        include_str!("../shaders/shadow.wgsl"),
        &[
            helio_light_api::SHADOW_ATLAS_SNIPPET,
            helio_mesh_api::DRAW_ROWS_SNIPPET,
            helio_mesh_api::SKINNING_SNIPPET,
            helio_frame_api::rays::stream::SNIPPET,
        ],
    )
    .into_owned()
}

pub struct ShadowPass {
    depth_materials: helio_pass_gbuffer::depth_material::DepthMaterialBindings,
    /// Shadow geometry pipeline (depth-only, front-face culled, depth-bias = 2.0).
    pipeline: wgpu::RenderPipeline,
    face_static_gen: Vec<u64>,
    face_light_gen: Vec<u64>,
    face_last_update: Vec<u64>,
    face_strength: Vec<f32>,
    /// The strength last written to the GPU for each face, and the caster
    /// generation it was written under: a fade is the only reason to write
    /// it again, but a residency change rewrites it behind our back.
    face_strength_written: Vec<(f32, u64)>,
    face_ownership: Vec<(u32, helio_light_api::ShadowTile)>,
    schedule_frame: u64,
    last_work: (u32, u32),
    /// Faces serviced last frame: (metered by the budget, unmetered GPU-gated).
    last_visits: (usize, usize),

    /// Depth-clear pipeline: one draw per layer clears the pages of every
    /// serviced face to z = 1.0 (`DepthCompare::Always`) before geometry.
    depth_clear_pipeline: wgpu::RenderPipeline,
    /// The clear draw's jobs (`depth_clear.wgsl`), rewritten each frame, and
    /// the group that binds them with the dirty state it gates on.
    clear_jobs: wgpu::Buffer,
    clear_group: wgpu::BindGroup,
    /// Retires the serviced faces in one dispatch (`retire.wgsl`): the faces,
    /// rewritten each frame, and the group over them and the matrix buffers
    /// (rebuilt when either grows).
    retire_pipeline: wgpu::ComputePipeline,
    retire_layout: wgpu::BindGroupLayout,
    retire_jobs: wgpu::Buffer,
    retire_group: Option<wgpu::BindGroup>,
    retire_group_key: Option<[wgpu::Buffer; 2]>,
    /// The GPU's pending-dirty bits, read back without waiting: the faces that
    /// may need servicing are the ones this shows (`recently_dirty`).
    readbacks: helio_core::gpu_cache::ReadbackPool,
    dirty_readback: Option<DirtyReadback>,
    /// Frames each face stays in the serviced set (`DIRTY_LINGER_FRAMES`).
    recently_dirty: Vec<u8>,
    /// Each face's pending dirty bits as the last readback showed them (the
    /// bits `ShadowDirty` set and no service has cleared yet): whether any
    /// serviced face has static work, which decides if the static layer's pass
    /// opens at all.
    pending_bits: Vec<u32>,
    /// The frame's lists, kept between frames so a frame allocates nothing.
    scratch: Scratch,
    /// Edge of both atlases in texels.
    atlas_size: u32,

    #[allow(dead_code)]
    bgl_0: wgpu::BindGroupLayout,

    /// Per-face face-index values, written once at construction and never touched again.
    face_idx_buf: wgpu::Buffer,

    // ── Dynamic shadow atlas (Movable objects only) ───────────────────────────
    face_views: Box<[wgpu::TextureView]>,
    bg_0: Option<wgpu::BindGroup>,
    bg_0_key: Option<[wgpu::Buffer; 5]>,

    // ── Static shadow atlas (Static/Stationary objects only) ─────────────────
    static_face_views: Box<[wgpu::TextureView]>,
    /// Whole-array view of the static atlas, for the transmittance depth test.
    static_array_view: Option<wgpu::TextureView>,
    dynamic_array_view: wgpu::TextureView,
    /// Coloured transmittance of translucent static casters.
    transmittance: transmittance::Transmittance,
    /// Translucent casters were drawn: the layer is published only then, so
    /// receivers skip RGB visibility entirely in scenes without glass.
    has_glass: bool,

    pub compare_sampler: wgpu::Sampler,

    // ── GPU dirty buffers (shared with ShadowDirtyPass) ───────────────────────
    /// `array<u32, 256>` — 0 = clean, 1 = dirty (written by ShadowDirtyPass).
    /// Used as indirect draw count for the depth-clear triangle (0 = no clear, 1 = clear).
    face_dirty_buf: Arc<wgpu::Buffer>,
    /// Per-face draw counts written by ShadowDirtyPass, in movable, static and
    /// transmissive regions: 0 = clean, the list's live count = dirty. Used as
    /// the indirect draw count (`multi_draw_indexed_indirect_count`) for faces
    /// not known to be dirty on the CPU.
    #[allow(dead_code)]
    face_geom_count_buf: Arc<wgpu::Buffer>,
    /// Per-face 8x8 page masks of the dynamic layer written by ShadowDirtyPass
    /// (two words per face); a serviced face's words are cleared.
    face_pages_buf: Arc<wgpu::Buffer>,

    /// True when the device supports MULTI_DRAW_INDIRECT_COUNT (Vulkan 1.2+, DX12 tier2).
    /// False on macOS Metal, WASM, and older Vulkan/DX12.  When false the ObjectDirty path
    /// falls back to a full LoadOp::Clear + multi_draw_indexed_indirect (no per-face GPU culling).
    supports_multi_draw_count: bool,
    draw_slots: helio_mesh_api::DrawSlots,
}

impl ShadowPass {
    /// Reserved atlas operations and texels in the last frame; GPU dirty gates may reduce actual work.
    pub fn last_update_work(&self) -> (u32, u32) {
        self.last_work
    }

    /// Faces serviced in the last frame: (metered by the update budget,
    /// unmetered because only the GPU's dirty flags decide their work).
    pub fn last_face_visits(&self) -> (usize, usize) {
        self.last_visits
    }

    /// Allocate persistent atlas resources.
    ///
    /// `face_dirty_buf` and `face_geom_count_buf` are shared with `ShadowDirtyPass`
    /// which writes them each frame; they arrive via `Arc`.
    /// Compatibility constructor. Physical storage is now one packed layer.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        face_dirty_buf: Arc<wgpu::Buffer>,
        face_geom_count_buf: Arc<wgpu::Buffer>,
        face_pages_buf: Arc<wgpu::Buffer>,
        _face_cull_indirect: Arc<wgpu::Buffer>,
        _face_cull_counts: Arc<wgpu::Buffer>,
        atlas_size: u32,
        _atlas_layers: u32,
    ) -> Self {
        Self::new_tiled(
            device,
            queue,
            face_dirty_buf,
            face_geom_count_buf,
            face_pages_buf,
            atlas_size,
        )
    }
    pub fn new_tiled(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        face_dirty_buf: Arc<wgpu::Buffer>,
        face_geom_count_buf: Arc<wgpu::Buffer>,
        face_pages_buf: Arc<wgpu::Buffer>,
        atlas_size: u32,
    ) -> Self {
        let atlas_layers = 1;
        // ── Shader ────────────────────────────────────────────────────────────
        let depth_materials =
            helio_pass_gbuffer::depth_material::DepthMaterialBindings::new(device);
        let source = depth_materials.shader_source(&caster_geometry());
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shadow cutout"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });

        let clear_shader = helio_core::shader::module(
            device,
            "Shadow/DepthClear",
            helio_core::include_wgsl!("../shaders/depth_clear.wgsl"),
        );

        // ── Bind Group Layout 0 ───────────────────────────────────────────────
        let bgl_0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Shadow BGL 0"),
            entries: &[
                // binding 0: shadow_matrices — array of mat4x4 light-space transforms
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: instances — per-instance world transforms
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 2: face index — 16-byte uniform, dynamic offset selects face
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 3: coordinate_spaces — current-frame per-space transforms
                // (sublevels/portals), slot 0 = identity. See gbuffer.wgsl for the
                // full mechanism; shadows only need the current-frame copy.
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 7: per-face dirty page masks of the dynamic layer
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // ── Pipeline ──────────────────────────────────────────────────────────
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Shadow PL"),
            bind_group_layouts: &[Some(&bgl_0), Some(&depth_materials.layout)],
            immediate_size: 0,
        });

        let pipeline = helio_core::shader::create_render_pipeline(
            &device,
            wgpu::RenderPipelineDescriptor {
                label: Some("Shadow Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    // Shared mesh vertex buffer layout (stride = 40 bytes, matches GBuffer pass).
                    // Only position (Float32x3 at offset 0) is needed for depth projection.
                    buffers: &[
                        Some(wgpu::VertexBufferLayout {
                            array_stride: 40,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: &[
                                wgpu::VertexAttribute {
                                    format: wgpu::VertexFormat::Float32x3,
                                    offset: 0,
                                    shader_location: 0,
                                },
                                wgpu::VertexAttribute {
                                    format: wgpu::VertexFormat::Float32x2,
                                    offset: 16,
                                    shader_location: 1,
                                },
                            ],
                        }),
                        Some(helio_mesh_api::DRAW_SLOT_LAYOUT),
                    ],
                },
                // Depth-only: no colour outputs, no fragment shader.
                // The GPU writes depth from the vertex clip position automatically.
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    // Front-face culling: light "looks into" the scene; culling the faces
                    // visible to the light prevents writing depth for lit-surface geometry
                    // directly, eliminating shadow acne.  Identical convention to UE4/Unity.
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: wgpu::StencilState::default(),
                    // slope_scale compensates for FP depth precision on surfaces at
                    // grazing angles to the light.  Without it the shadow map depth for
                    // a surface can be equal-to or less-than the depth reconstructed in
                    // the lighting shader for that same surface, causing self-shadowing
                    // on every light independently (making each light appear to inherit
                    // every other light's shadow geometry).
                    // constant is left at 0 — that was the source of the visible offset.
                    bias: wgpu::DepthBiasState {
                        constant: 0,
                        slope_scale: 2.0,
                        clamp: 0.0,
                    },
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            },
        );

        // ── Depth-clear pipeline ───────────────────────────────────────────────
        // GPU-clear the serviced faces' pages: one draw per layer of quads at
        // depth = 1.0 (far plane) using DepthCompare::Always to overwrite
        // existing depth values, placed in the atlas from the jobs the pass
        // writes each frame. No vertex buffer, no fragment shader, no bias.
        let storage_entry = |binding, read_only, visibility| wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let clear_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Shadow/DepthClear BGL"),
            entries: &[
                storage_entry(0, true, wgpu::ShaderStages::VERTEX),
                storage_entry(1, true, wgpu::ShaderStages::VERTEX),
                storage_entry(2, true, wgpu::ShaderStages::VERTEX),
            ],
        });
        let depth_clear_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Shadow/DepthClear PL"),
                bind_group_layouts: &[Some(&clear_layout)],
                immediate_size: 0,
            });
        let clear_jobs = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shadow/DepthClear jobs"),
            size: CLEAR_JOB_CAPACITY * CLEAR_JOB_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let clear_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shadow/DepthClear group"),
            layout: &clear_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: clear_jobs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: face_dirty_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: face_pages_buf.as_entire_binding(),
                },
            ],
        });

        let depth_clear_pipeline = helio_core::shader::create_render_pipeline(
            &device,
            wgpu::RenderPipelineDescriptor {
                label: Some("Shadow/DepthClear Pipeline"),
                layout: Some(&depth_clear_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &clear_shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: None,
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Always),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            },
        );

        // ── Retire dispatch ────────────────────────────────────────────────────
        let retire_shader = helio_core::shader::module(
            device,
            "Shadow/Retire",
            helio_core::include_wgsl!("../shaders/retire.wgsl"),
        );
        let compute = wgpu::ShaderStages::COMPUTE;
        let retire_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Shadow/Retire BGL"),
            entries: &[
                storage_entry(0, true, compute),
                storage_entry(1, true, compute),
                storage_entry(2, false, compute),
                storage_entry(3, false, compute),
                storage_entry(4, false, compute),
                storage_entry(5, false, compute),
            ],
        });
        let retire_pipeline = helio_core::shader::create_compute_pipeline(
            device,
            wgpu::ComputePipelineDescriptor {
                label: Some("Shadow/Retire Pipeline"),
                layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Shadow/Retire PL"),
                    bind_group_layouts: &[Some(&retire_layout)],
                    immediate_size: 0,
                })),
                module: &retire_shader,
                entry_point: Some("retire"),
                compilation_options: Default::default(),
                cache: None,
            },
        );
        let retire_jobs = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shadow/Retire jobs"),
            size: RETIRE_WORD_CAPACITY * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // One u32 per face at FACE_BUF_STRIDE byte intervals.
        // The CPU never touches this buffer after construction.
        let mut face_idx_data =
            vec![0u8; FACE_REGIONS * MAX_SHADOW_FACES * FACE_BUF_STRIDE as usize];
        for i in 0..FACE_REGIONS * MAX_SHADOW_FACES {
            let offset = i * FACE_BUF_STRIDE as usize;
            face_idx_data[offset..offset + 4]
                .copy_from_slice(&((i % MAX_SHADOW_FACES) as u32).to_ne_bytes());
            face_idx_data[offset + 4..offset + 8]
                .copy_from_slice(&FACE_REGION_GATES[i / MAX_SHADOW_FACES].to_ne_bytes());
        }
        let face_idx_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shadow/FaceIdx"),
            size: (FACE_REGIONS * MAX_SHADOW_FACES) as u64 * FACE_BUF_STRIDE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&face_idx_buf, 0, &face_idx_data);

        // ── Face views (lazily initialized from graph-owned textures) ──────────

        // Comparison sampler for PCF shadow lookups in the lighting pass.
        let compare_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Shadow/Compare"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        let transmittance =
            transmittance::Transmittance::new(device, queue, &bgl_0, atlas_size, atlas_layers);

        let make_atlas = |label| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: atlas_size,
                    height: atlas_size,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        };
        let dynamic_atlas = make_atlas("Dynamic shadow atlas");
        let static_atlas = make_atlas("Static shadow atlas");
        let array_view = |tex: &wgpu::Texture| {
            tex.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            })
        };
        let face_views = Self::create_face_views(&dynamic_atlas, "Dynamic tiles", 1);
        let static_face_views = Self::create_face_views(&static_atlas, "Static tiles", 1);
        Self {
            depth_materials,
            last_work: (0, 0),
            last_visits: (0, 0),
            dynamic_array_view: array_view(&dynamic_atlas),
            face_ownership: vec![(0, Default::default()); MAX_SHADOW_FACES],
            pipeline,
            face_static_gen: vec![u64::MAX; MAX_SHADOW_FACES],
            face_light_gen: vec![u64::MAX; MAX_SHADOW_FACES],
            face_last_update: vec![0; MAX_SHADOW_FACES],
            face_strength: vec![0.0; MAX_SHADOW_FACES],
            face_strength_written: vec![(f32::NAN, u64::MAX); MAX_SHADOW_FACES],
            schedule_frame: 0,
            depth_clear_pipeline,
            clear_jobs,
            clear_group,
            retire_pipeline,
            retire_layout,
            retire_jobs,
            retire_group: None,
            retire_group_key: None,
            readbacks: helio_core::gpu_cache::ReadbackPool::new("Shadow dirty readback"),
            dirty_readback: None,
            recently_dirty: vec![0; MAX_SHADOW_FACES],
            pending_bits: vec![0; MAX_SHADOW_FACES],
            scratch: Scratch::default(),
            atlas_size,
            static_array_view: Some(array_view(&static_atlas)),
            transmittance,
            has_glass: false,
            bgl_0,
            bg_0: None,
            bg_0_key: None,
            face_idx_buf,
            face_views,
            static_face_views,
            compare_sampler,
            face_dirty_buf,
            face_geom_count_buf,
            face_pages_buf,
            supports_multi_draw_count: device
                .features()
                .contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT),
            draw_slots: Default::default(),
        }
    }

    fn create_face_views(
        texture: &wgpu::Texture,
        label: &str,
        layer_count: u32,
    ) -> Box<[wgpu::TextureView]> {
        (0..layer_count)
            .map(|i| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some(label),
                    format: Some(wgpu::TextureFormat::Depth32Float),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: i,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect()
    }
}

// ── RenderPass impl ───────────────────────────────────────────────────────────

impl RenderPass for ShadowPass {
    fn scene_buffers(&self) -> &'static [&'static str] {
        const LIST: &[&str] = &[
            helio_mesh_api::MESH_INDICES.name(),
            helio_mesh_api::MESH_VERTICES.name(),
            helio_mats::MATERIALS.name(),
            helio_mesh_api::SKIN_CACHE_ROWS.name(),
            helio_mesh_api::SKIN_CACHE.name(),
        ];
        LIST
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a helio_core::ResourceRegistry<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        // Pass-owned one-layer array views, preserving the sampling ABI at one physical layer.
        builder.write_buffer(helio_light_api::SHADOW_ATLAS);
        builder.write_buffer(helio_light_api::STATIC_SHADOW_ATLAS);
        builder.read("object_batch");
        builder.read(helio_light_api::SHADOW_MATRICES);
        builder.read("shadow_dirty");
        builder.read(helio_mats::MATERIAL_TEXTURES);
        // Pass-owned (it caches with the static atlas): declared for ordering,
        // routed in `publish`.
        builder.write_buffer(TRANSMITTANCE_KEY);
    }

    fn name(&self) -> &'static str {
        "Shadow"
    }

    fn writes(&self) -> &'static [&'static str] {
        &[
            helio_light_api::SHADOW_ATLAS,
            helio_light_api::SHADOW_SAMPLER,
            helio_light_api::STATIC_SHADOW_ATLAS,
            TRANSMITTANCE_KEY,
        ]
    }

    fn publish<'a>(&self, frame: &mut helio_core::ResourceRegistry<'a>) {
        frame.route_named_texture(
            helio_light_api::SHADOW_ATLAS,
            &self.dynamic_array_view,
            self.name(),
        );
        if let Some(view) = self.static_array_view.as_ref() {
            frame.route_named_texture(helio_light_api::STATIC_SHADOW_ATLAS, view, self.name());
        }
        if self.has_glass {
            frame.route_named_texture(TRANSMITTANCE_KEY, &self.transmittance.view, self.name());
        }
    }

    fn prepare(&mut self, _ctx: &PrepareContext) -> HelioResult<()> {
        helio_core::cpu_scope!("Shadow: prepare");
        // Last frames' pending bits: map the copy once it has been submitted,
        // and read it once mapped. A face with a bit set stays serviced for a
        // few frames (`DIRTY_LINGER_FRAMES`); the rest are not visited.
        if let Some(readback) = self.dirty_readback.as_mut() {
            if !readback.mapping {
                readback.buffer.map_read(Arc::clone(&readback.ready));
                readback.mapping = true;
            } else if readback.ready.load(Ordering::Acquire) {
                {
                    let view = readback
                        .buffer
                        .slice(..MAX_SHADOW_FACES as u64 * 4)
                        .get_mapped_range();
                    if let Ok(bytes) = view.map(|bytes| bytes.to_vec()) {
                        for (face, word) in bytes.chunks_exact(4).enumerate() {
                            let bits = u32::from_le_bytes([word[0], word[1], word[2], word[3]]);
                            self.pending_bits[face] = bits;
                            if bits != 0 {
                                self.recently_dirty[face] = DIRTY_LINGER_FRAMES;
                            }
                        }
                    }
                }
                readback.buffer.unmap();
                self.dirty_readback = None;
            }
        }
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        helio_core::cpu_scope!("Shadow: execute");
        self.last_work = (0, 0);
        self.last_visits = (0, 0);
        if self.depth_materials.update(ctx) {
            self.face_static_gen.fill(u64::MAX);
            self.face_light_gen.fill(u64::MAX);
        }
        if self.depth_materials.group.is_none() {
            return Ok(());
        }
        let Some(batch) = ctx
            .registry
            .get::<helio_mesh_api::ObjectBatchFrameData<'_>>(helio_core::ResourceKey::new(
                "object_batch",
            ))
        else {
            return Ok(());
        };
        let Some(data) =
            ctx.registry
                .get::<helio_light_api::ShadowMatricesFrameData<'_>>(
                    helio_light_api::shadow_matrices(),
                )
        else {
            return Ok(());
        };
        let Some(residency) = data.residency else {
            return Ok(());
        };
        let Some(coords) = ctx
            .registry
            .get::<helio_core::CoordinateSpacesFrameData<'_>>(
                helio_core::resource_keys::coordinate_spaces(),
            )
        else {
            return Ok(());
        };
        let Some(vertices) = ctx.scene_buffers.get(helio_mesh_api::MESH_VERTICES.key()) else {
            return Ok(());
        };
        let Some(indices) = ctx.scene_buffers.get(helio_mesh_api::MESH_INDICES.key()) else {
            return Ok(());
        };
        let desired = data.desired_matrices.unwrap_or(data.shadow_matrices);
        let skin = ctx.scene_buffers.get(helio_mesh_api::SKIN_CACHE.key());
        let palette = ctx.scene_buffers.get(helio_mesh_api::SKIN_CACHE_ROWS.key());
        let key = [
            desired.clone(),
            batch.instances.clone(),
            coords.coordinate_spaces.clone(),
            skin.map_or_else(|| batch.instances.clone(), |h| h.buffer.clone()),
            palette.map_or_else(|| batch.instances.clone(), |h| h.buffer.clone()),
        ];
        if self.bg_0_key.as_ref() != Some(&key) {
            helio_core::cpu_scope!("Shadow: rebuild bind group");
            self.bg_0 = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Shadow tiles"),
                layout: &self.bgl_0,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: desired.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: batch.instances.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.face_idx_buf,
                            offset: 0,
                            size: std::num::NonZeroU64::new(16),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: coords.coordinate_spaces.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.face_dirty_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: key[3].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: key[4].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: self.face_pages_buf.as_entire_binding(),
                    },
                ],
            }));
            self.bg_0_key = Some(key);
        }
        helio_core::cpu_scope!("Shadow: gather candidates");
        self.schedule_frame += 1;
        for frames in &mut self.recently_dirty {
            *frames = frames.saturating_sub(1);
        }
        self.has_glass = batch.readback_shadow_transmissive_draw_count > 0;
        let mut candidates = std::mem::take(&mut self.scratch.candidates);
        candidates.clear();
        let fade_step = 1.0 / data.budget.fade_frames.max(1) as f32;
        for (slot, r) in residency.residents.iter().enumerate() {
            for (f, t) in r.tiles.iter().enumerate() {
                let face = slot * 6 + f;
                if r.owner == 0 || t.size == 0 {
                    self.face_strength[face] = 0.0;
                    continue;
                }
                if self.face_ownership[face] != (r.owner, *t) {
                    self.face_strength[face] = 0.0;
                    self.face_last_update[face] = 0;
                    self.face_ownership[face] = (r.owner, *t);
                }
                let changed = self.face_light_gen[face] != data.per_caster_dirty_gen[slot];

                let target = if self.face_last_update[face] == 0 {
                    0.0
                } else {
                    r.strength
                };
                self.face_strength[face] +=
                    (target - self.face_strength[face]).clamp(-fade_step, fade_step);
                let written = (self.face_strength[face], data.per_caster_dirty_gen[slot]);
                if self.face_strength_written[face] != written {
                    ctx.write_buffer(
                        data.shadow_matrices,
                        face as u64 * 96 + 76,
                        bytemuck::bytes_of(&self.face_strength[face]),
                    );
                    self.face_strength_written[face] = written;
                }
                let static_dirty =
                    changed || self.face_static_gen[face] != batch.shadow_static_generation;
                // A face nothing has dirtied is not visited: its cached layers
                // are current and its tile costs no commands.
                if !static_dirty && self.recently_dirty[face] == 0 {
                    continue;
                }
                // Aging guarantees low-priority pending faces are eventually serviced.
                let age = self
                    .schedule_frame
                    .saturating_sub(self.face_last_update[face]) as f32;
                let urgency = if changed { 1.0e6 } else { r.score + age * 0.01 };
                candidates.push((face, static_dirty, urgency));
            }
        }
        {
            helio_core::cpu_scope!("Shadow: rank candidates");
            candidates.sort_by(|a, b| b.2.total_cmp(&a.2).then(a.0.cmp(&b.0)));
        }
        let mut updates = 0u32;
        let mut texels = 0u32;
        let mut free_visits = 0usize;
        let slots = self
            .draw_slots
            .buffer(
                &ctx.device,
                helio_mesh_api::DrawSlots::len_of(
                    batch.instances.size(),
                    helio_mesh_api::draw_slots::INSTANCE_BYTES,
                ),
            )
            .clone();
        let bg = self.bg_0.as_ref().unwrap();

        // ── Select the faces to service this frame ───────────────────────────
        helio_core::cpu_scope!("Shadow: select faces");
        let mut selected = std::mem::take(&mut self.scratch.selected);
        selected.clear();
        for &(face, static_dirty, _) in &candidates {
            let r = &residency.residents[face / 6];
            let tile = r.tiles[face % 6];
            let layers = 3;
            let cost = tile.size * tile.size * layers;
            // Without GPU-side draw counts a face that is not rebuilding its
            // static layer still draws the whole caster lists (the vertex
            // shader discards them), so it is real work to the budget.
            if static_dirty || !self.supports_multi_draw_count {
                // A rebuild of the cached static layer: real work known to the
                // CPU, so it is metered.
                if updates + layers > data.budget.updates_per_frame
                    || texels + cost > data.budget.update_texels_per_frame
                {
                    continue;
                }
                updates += layers;
                texels += cost;
            } else {
                // Anything else is driven by the GPU's own dirty flags
                // (`ShadowDirty`: a movable caster moved, a light changed): a
                // clean face runs no vertices, a dirty one redraws only its
                // dynamic layer. The CPU cannot tell which is which, so
                // charging these to the budget let clean faces starve the
                // dirty ones and a moving object's shadow updated in steps.
                // They are visited every frame, up to a safety cap.
                if free_visits >= MAX_UNMETERED_FACES_PER_FRAME {
                    continue;
                }
                free_visits += 1;
            }
            selected.push((face, static_dirty));
        }
        self.last_visits = (selected.len() - free_visits, free_visits);

        self.scratch.candidates = candidates;

        // ── The depth-clear draw's jobs: both layers of every serviced face ──
        // (header first), written once so each layer clears all its faces in
        // one draw with no per-face state.
        let serviced = selected.len() as u32;
        let mut clear_jobs = std::mem::take(&mut self.scratch.clear);
        clear_jobs.clear();
        clear_jobs.push([self.atlas_size, 0, 0, 0]);
        for is_static in [true, false] {
            for &(face, static_dirty) in &selected {
                let tile = residency.residents[face / 6].tiles[face % 6];
                // The dirty bit the layer is gated on (1 dynamic, 2 static);
                // 0 clears it unconditionally (its static layer is rebuilt).
                let gate = if static_dirty {
                    0
                } else if is_static {
                    2
                } else {
                    1
                };
                clear_jobs.push([face as u32, gate, tile.x | (tile.y << 16), tile.size]);
            }
        }
        if serviced > 0 {
            ctx.write_buffer(&self.clear_jobs, 0, bytemuck::cast_slice(&clear_jobs));
        }
        self.scratch.clear = clear_jobs;

        // ── Both depth layers: one render pass each. The pages of every
        // serviced face are cleared in one draw, then each face's casters are
        // drawn into its tile (a viewport, the face's constants and one
        // multi-draw), so a clean face costs three commands and no pass
        // boundary. The layers are separate textures, so they are timed apart.
        let mut cmds = ctx.graphics_cmds();
        // The pending bits as they stand now, after `ShadowDirty` has marked
        // this frame's changes and before the faces serviced below clear
        // theirs: what the next frames' selection reads (one copy in flight).
        if self.dirty_readback.is_none() {
            let buffer = self
                .readbacks
                .get(ctx.device, MAX_SHADOW_FACES as u64 * 4);
            cmds.copy_buffer_to_buffer(
                &self.face_dirty_buf,
                0,
                &buffer,
                0,
                MAX_SHADOW_FACES as u64 * 4,
            );
            self.dirty_readback = Some(DirtyReadback {
                buffer,
                ready: Arc::new(AtomicBool::new(false)),
                mapping: false,
            });
        }
        // No face to service: no pass at all (a pass costs wgpu's validation of
        // everything it binds, whatever it draws).
        // The static layer's pass is opened only when a serviced face has
        // static work: rebuilding its cached layer (known here), or a pending
        // static bit the readback showed. A visit that is only a movable
        // caster's (the dynamic layer) must not pay for the pass: it cost
        // 1.1 ms of GPU time with nothing drawn, in every frame that
        // a character walked through a cascade.
        let static_bit = FACE_REGION_GATES[2];
        let static_work = !self.supports_multi_draw_count
            || selected
                .iter()
                .any(|&(face, rebuilding)| rebuilding || self.pending_bits[face] & static_bit != 0);
        for is_static in [true, false]
            .into_iter()
            .filter(|&is_static| serviced > 0 && (!is_static || static_work))
        {
            let scope = if is_static {
                "Shadow: static layer"
            } else {
                "Shadow: dynamic layer"
            };
            helio_core::cpu_scope!(scope);
            ctx.begin_gpu_scope(&mut cmds, scope);
            {
                let view = if is_static {
                    &self.static_face_views[0]
                } else {
                    &self.face_views[0]
                };
                let mut pass = cmds.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some(scope),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                if serviced > 0 {
                    let first = if is_static { 0 } else { serviced * PAGES_PER_FACE };
                    pass.set_pipeline(&self.depth_clear_pipeline);
                    pass.set_bind_group(0, &self.clear_group, &[]);
                    pass.draw(0..6, first..first + serviced * PAGES_PER_FACE);
                    let (max_draws, count) = if is_static {
                        batch.shadow_static_draws()
                    } else {
                        batch.shadow_movable_draws()
                    };
                    // Nothing to draw anywhere: the clear above is all a
                    // layer without casters needs.
                    if max_draws > 0 {
                        let indirect = if is_static {
                            batch.shadow_static_indirect
                        } else {
                            batch.shadow_movable_indirect
                        };
                        let counts_region = if is_static { STATIC_COUNTS } else { MOVABLE_COUNTS };
                        pass.set_pipeline(&self.pipeline);
                        pass.set_bind_group(1, self.depth_materials.group.as_ref().unwrap(), &[]);
                        pass.set_vertex_buffer(0, vertices.buffer.slice(..));
                        pass.set_vertex_buffer(1, slots.slice(..));
                        pass.set_index_buffer(indices.buffer.slice(..), wgpu::IndexFormat::Uint32);
                        for &(face, static_dirty) in &selected {
                            let r = &residency.residents[face / 6];
                            if r.flags & if is_static { 16 } else { 32 } != 0 {
                                continue;
                            }
                            let tile = r.tiles[face % 6];
                            pass.set_viewport(
                                tile.x as f32,
                                tile.y as f32,
                                tile.size as f32,
                                tile.size as f32,
                                0.0,
                                1.0,
                            );
                            pass.set_bind_group(
                                0,
                                bg,
                                &[face_idx_offset(face, static_dirty, is_static)],
                            );
                            // With GPU counts, every list is bounded by its
                            // capacity and drawn to this frame's live count,
                            // so a caster spawned or despawned this frame is
                            // already (or no longer) in it. A face not known
                            // to be dirty draws the list only if `ShadowDirty`
                            // dirtied it on the GPU (a light's matrix changed
                            // or a movable caster moved): its per-face count
                            // is the live count then, else 0.
                            let count = if self.supports_multi_draw_count && !static_dirty {
                                Some(helio_mesh_api::GpuDrawCount {
                                    buffer: &self.face_geom_count_buf,
                                    offset: face_count_offset(counts_region, face),
                                })
                            } else {
                                count
                            };
                            helio_mesh_api::multi_draw_indexed_indirect(
                                &mut pass, indirect, 0, max_draws, count,
                            );
                        }
                    }
                }
            }
            ctx.end_gpu_scope(&mut cmds, scope);
        }
        // ── Glass transmittance (follows the static layer) ───────────────────
        if let Some(depth) = self.static_array_view.as_ref() {
            helio_core::cpu_scope!("Shadow: transmittance");
            ctx.begin_gpu_scope(&mut cmds, "Shadow: transmittance");
            let materials = ctx.scene_buffers.get(helio_mats::MATERIALS.key());
            for &(face, static_dirty) in &selected {
                // Glass is cached with the static layer: with none in the
                // scene, only a rebuild has a stale tile to clear.
                if !static_dirty && !self.has_glass {
                    continue;
                }
                let r = &residency.residents[face / 6];
                let tile = r.tiles[face % 6];
                let face_count = |region| helio_mesh_api::GpuDrawCount {
                    buffer: &self.face_geom_count_buf,
                    offset: face_count_offset(region, face),
                };
                self.transmittance.render_face(
                    ctx.device,
                    &mut cmds,
                    materials.map(|m| &m.buffer),
                    bg,
                    0,
                    face_idx_offset(face, static_dirty, true),
                    depth,
                    batch.shadow_transmissive_indirect,
                    // Static layer: whether there is glass at all
                    // follows the readback, like the rest of it.
                    if r.flags & 16 == 0 && self.has_glass {
                        batch.shadow_transmissive_draws().0
                    } else {
                        0
                    },
                    // As the static layer: a clean face draws no glass.
                    if self.supports_multi_draw_count && !static_dirty {
                        Some(face_count(TRANSMISSIVE_COUNTS))
                    } else {
                        batch.shadow_transmissive_draws().1
                    },
                    &vertices.buffer,
                    &slots,
                    &indices.buffer,
                    Some([tile.x, tile.y, tile.size]),
                );
            }
            ctx.end_gpu_scope(&mut cmds, "Shadow: transmittance");
        }

        // ── Retire each serviced face ─────────────────────────────────────────
        // Only after its matching depth has been rendered: publish its matrix
        // and clear its dirty state, in one dispatch for all of them.
        helio_core::cpu_scope!("Shadow: retire faces");
        if serviced > 0 {
            let mut retire = std::mem::take(&mut self.scratch.retire);
            retire.clear();
            retire.push(serviced);
            retire.push(MAX_SHADOW_FACES as u32);
            retire.extend(selected.iter().map(|&(face, _)| face as u32));
            ctx.write_buffer(&self.retire_jobs, 0, bytemuck::cast_slice(&retire));
            self.scratch.retire = retire;
            let key = [desired.clone(), data.shadow_matrices.clone()];
            if self.retire_group_key.as_ref() != Some(&key) {
                self.retire_group = Some(ctx.device.create_bind_group(
                    &wgpu::BindGroupDescriptor {
                        label: Some("Shadow/Retire group"),
                        layout: &self.retire_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: self.retire_jobs.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: desired.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: data.shadow_matrices.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: self.face_dirty_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: self.face_pages_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: self.face_geom_count_buf.as_entire_binding(),
                            },
                        ],
                    },
                ));
                self.retire_group_key = Some(key);
            }
            ctx.begin_gpu_scope(&mut cmds, "Shadow: activate faces");
            {
                let mut pass = cmds.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Shadow/Retire"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.retire_pipeline);
                pass.set_bind_group(0, self.retire_group.as_ref().unwrap(), &[]);
                pass.dispatch_workgroups(serviced.div_ceil(RETIRE_WORKGROUP), 1, 1);
            }
            ctx.end_gpu_scope(&mut cmds, "Shadow: activate faces");
        }
        for &(face, static_dirty) in &selected {
            // The "ready" marker only changes when the face was (re)assigned:
            // a rebuild of its static layer, or its first service.
            if static_dirty || self.face_last_update[face] == 0 {
                ctx.write_buffer(
                    data.shadow_matrices,
                    face as u64 * 96 + 92,
                    bytemuck::bytes_of(&2u32),
                );
            }
            self.face_static_gen[face] = batch.shadow_static_generation;
            self.face_light_gen[face] = data.per_caster_dirty_gen[face / 6];
            self.face_last_update[face] = self.schedule_frame;
        }
        self.scratch.selected = selected;
        self.last_work = (updates, texels);
        Ok(())
    }
}
