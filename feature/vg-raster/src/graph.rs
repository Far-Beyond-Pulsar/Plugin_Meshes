//! Render-graph wrappers of the raster: [`VgRasterEarlyPass`] (phase 0),
//! [`VgHizLatePass`] (Hi-Z of the phase-0 depth, published as `vg_hiz_late`
//! for the late cull) and [`VgRasterLatePass`] (phase 1 delta + tile
//! classification). The three share one [`VgRasterShared`].
//!
//! Graph order (see `helio-presets`):
//!
//! ```text
//! VgCullEarly -> VgRasterEarly -> VgHizLate -> VgCullLate -> VgRasterLate
//! ```
//!
//! A pass whose cull inputs are not published this frame (virtual geometry
//! idle) records nothing.

use std::sync::{Arc, Mutex, MutexGuard};

use helio_core::graph::{ResourceBuilder, ResourceFormat, ResourceSize};
use helio_core::{PassContext, PrepareContext, RenderPass, ResourceKey, Result as HelioResult};
use helio_pass_vg_cull::{
    hw_masked_draw_args_offset, state, sw_dispatch_args_offset, sw_masked_dispatch_args_offset,
    VgCullFrameData, VgCullInputs, ARGS_BYTE_OFFSET, ARGS_PHASE_BYTES, BIN_COUNT, BIN_SW,
    BIN_SW_MASKED,
};
use helio_vg_format::VgRun;
use wgpu::util::DeviceExt;

use crate::{VgChainClip, VgMaskedInputs, VgRaster, VgRasterConfig, VgRasterInputs, VgTargets};

/// [`VgRasterFrameData`] after phase 0, of the frame's own run
/// ([`VgRun::DIRECT`]`.raster_early`; every run names its own).
pub const KEY_RASTER_EARLY: &str = VgRun::DIRECT.raster_early;
/// [`VgRasterFrameData`] after phase 1 and tile classification (final), of
/// the frame's own run ([`VgRun::DIRECT`]`.raster`).
pub const KEY_RASTER: &str = VgRun::DIRECT.raster;

/// What the dense tier and the shading read. `Copy` so it lives in the frame
/// registry.
#[derive(Clone, Copy)]
pub struct VgRasterFrameData<'a> {
    /// `R32Uint` Tier 1 ids (`abi::VIS_EMPTY` where nothing was drawn).
    pub vis: &'a wgpu::TextureView,
    /// Tier 1 `Depth32Float` (reversed-Z).
    pub depth: &'a wgpu::TextureView,
    /// Tier 2 keys, `width * height` of `vec2<u32>{id, depth bits}`.
    pub sw_keys: &'a wgpu::Buffer,
    /// Dense-tile mask classified this frame (valid after the late pass).
    pub dense_mask: &'a wgpu::Buffer,
    /// Per-tile software-raster counters, `vec2<u32>{triangles, pixel hits}`
    /// (row-major, [`abi::TILE_SIZE`] pixels square); valid after the late pass.
    pub tile_counters: &'a wgpu::Buffer,
    pub width: u32,
    pub height: u32,
    pub tiles: [u32; 2],
}

struct Shared {
    raster: VgRaster,
    targets: Option<VgTargets>,
}

/// The raster and its targets, shared by the three passes (and by the
/// integration that feeds the cull last frame's dense mask).
#[derive(Clone)]
pub struct VgRasterShared(Arc<Mutex<Shared>>);

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

impl VgRasterShared {
    pub fn new(device: &wgpu::Device, config: VgRasterConfig) -> Self {
        Self(Arc::new(Mutex::new(Shared {
            raster: VgRaster::new(device, config),
            targets: None,
        })))
    }

    /// The dense-tile classifier's sensitivity (`VirtualGeometrySettings`): both
    /// thresholds are divided by it.
    pub fn set_dense_sensitivity(&self, sensitivity: f32) {
        lock(&self.0).raster.config.dense_sensitivity = sensitivity;
    }

    /// (Re)create the targets for `width x height`.
    pub fn ensure_targets(&self, device: &wgpu::Device, width: u32, height: u32) {
        let mut shared = lock(&self.0);
        let (w, h) = (width.max(1), height.max(1));
        if shared
            .targets
            .as_ref()
            .is_some_and(|t| t.width == w && t.height == h)
        {
            return;
        }
        let targets = VgTargets::new(device, w, h, &shared.raster);
        shared.targets = Some(targets);
    }

    /// The mask classified in the previous frame (this frame's cull routing
    /// input) and the tile grid, once targets exist.
    pub fn dense_mask_last_frame(&self) -> Option<(wgpu::Buffer, [u32; 2])> {
        lock(&self.0)
            .targets
            .as_ref()
            .map(|t| (t.dense_mask_last_frame().clone(), [t.tiles_x, t.tiles_y]))
    }

    pub fn uses_tile_fallback(&self) -> bool {
        lock(&self.0).raster.uses_tile_fallback()
    }

    /// Diagnostics (capture / calibration tools only: it BLOCKS on the GPU).
    /// The per-tile Tier 2 counters of the last recorded frame as
    /// `[triangles, depth-passing hits]` per tile, row-major, plus the tile grid.
    pub fn tile_counters_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Option<(Vec<[u32; 2]>, [u32; 2])> {
        let (buffer, tiles) = {
            let shared = lock(&self.0);
            let t = shared.targets.as_ref()?;
            (t.tile_counters.clone(), [t.tiles_x, t.tiles_y])
        };
        let bytes = tiles[0] as u64 * tiles[1] as u64 * 8;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VG tile counters readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&buffer, 0, &staging, 0, bytes);
        queue.submit(Some(enc.finish()));
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map tile counters"));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let data: Vec<u32> = bytemuck::cast_slice(&slice.get_mapped_range().ok()?).to_vec();
        staging.unmap();
        Some((data.chunks_exact(2).map(|c| [c[0], c[1]]).collect(), tiles))
    }
}

/// Cloned handles of the targets, kept so `publish` can lend them.
struct Published {
    vis: wgpu::TextureView,
    depth: wgpu::TextureView,
    sw_keys: wgpu::Buffer,
    dense_mask: wgpu::Buffer,
    tile_counters: wgpu::Buffer,
    width: u32,
    height: u32,
    tiles: [u32; 2],
}

impl Published {
    fn of(t: &VgTargets) -> Self {
        Self {
            vis: t.vis_view.clone(),
            depth: t.depth_view.clone(),
            sw_keys: t.sw_keys.clone(),
            dense_mask: t.dense_mask_last_frame().clone(),
            tile_counters: t.tile_counters.clone(),
            width: t.width,
            height: t.height,
            tiles: [t.tiles_x, t.tiles_y],
        }
    }

    fn data<'a>(&self) -> VgRasterFrameData<'a> {
        // SAFETY: the handles live in `self`, which the graph keeps alive for
        // longer than the frame registry (same bridge as the other passes that
        // publish pass-owned resources); they are only replaced in `execute`.
        unsafe {
            VgRasterFrameData {
                vis: std::mem::transmute(&self.vis),
                depth: std::mem::transmute(&self.depth),
                sw_keys: std::mem::transmute(&self.sw_keys),
                dense_mask: std::mem::transmute(&self.dense_mask),
                tile_counters: std::mem::transmute(&self.tile_counters),
                width: self.width,
                height: self.height,
                tiles: self.tiles,
            }
        }
    }
}

fn cull_inputs<'a>(
    registry: &'a helio_core::ResourceRegistry<'a>,
    run: &VgRun,
) -> Option<VgCullInputs<'a>> {
    registry.get::<VgCullInputs<'a>>(ResourceKey::new(run.inputs))
}

fn record_phase(
    shared: &VgRasterShared,
    ctx: &mut PassContext,
    run: &VgRun,
    cull_key: &'static str,
    phase: u32,
) -> Option<Published> {
    let inputs = cull_inputs(ctx.registry, run)?;
    let cull = ctx
        .registry
        .get::<VgCullFrameData<'_>>(ResourceKey::new(cull_key))?;
    let mut guard = lock(&shared.0);
    let Shared { raster, targets } = &mut *guard;
    let targets = targets.as_mut()?;
    let hw_offset = ARGS_BYTE_OFFSET + u64::from(phase) * ARGS_PHASE_BYTES;
    // Alpha-masked clusters need the material table; without it the cull emits
    // no masked bins either (it gets the same `materials` buffer).
    let masked = match (
        ctx.scene_buffers.get(helio_mats::MATERIALS.key()),
        ctx.registry
            .get::<helio_mats::MaterialTextureBindings<'_>>(helio_mats::material_textures()),
    ) {
        (Some(materials), Some(material_textures)) => Some(VgMaskedInputs {
            materials: &materials.buffer,
            material_textures,
            hw_indirect_offset: hw_masked_draw_args_offset(phase),
            sw_meta_offset_words: state::RANGES + phase * BIN_COUNT * 2 + BIN_SW_MASKED * 2,
            sw_dispatch_offset: sw_masked_dispatch_args_offset(phase),
        }),
        _ => None,
    };
    // A run drawn through portal chains: the chains it clips to, as the
    // portals published them this frame.
    let clip = inputs.remaps.and_then(|remaps| {
        let reg = ctx.registry;
        Some(VgChainClip {
            remaps,
            chains: reg.buffer(ResourceKey::new(helio_portal_api::PORTAL_CHAINS))?,
            stages: reg.buffer(ResourceKey::new(helio_portal_api::PORTAL_STAGES))?,
            mask: ctx.resource_pool.get_view(helio_portal_api::PORTAL_MASK)?,
            view_remaps: reg.buffer(ResourceKey::new(helio_frame_api::lighting::VIEW_REMAPS))?,
        })
    });
    let raster_inputs = VgRasterInputs {
        camera: ctx.camera,
        view_index: 0,
        pages: inputs.stream.page_pool,
        asset_quant: inputs.stream.asset_quant,
        visible: cull.visible_clusters,
        vg_instances: inputs.instances,
        objects: inputs.objects,
        hw_indirect: cull.state,
        hw_indirect_offset: hw_offset,
        sw_meta: cull.state,
        sw_meta_offset_words: state::RANGES + phase * BIN_COUNT * 2 + BIN_SW * 2,
        sw_dispatch: cull.state,
        sw_dispatch_offset: sw_dispatch_args_offset(phase),
        masked,
        clip,
    };
    let mut rec = ctx.graphics_cmds();
    raster.encode_phase(
        ctx.device,
        ctx.queue,
        &mut rec,
        targets,
        &raster_inputs,
        phase,
    );
    if phase == 1 {
        raster.encode_classify(
            ctx.device,
            ctx.queue,
            &mut rec,
            targets,
            Some(cull.tile_estimate),
        );
    }
    Some(Published::of(targets))
}

/// Phase 0: draw what the early cull selected.
pub struct VgRasterEarlyPass {
    shared: VgRasterShared,
    published: Option<Published>,
    run: &'static VgRun,
    gate: Option<helio_vg_format::VgRunGate>,
}

impl VgRasterEarlyPass {
    pub fn shared(&self) -> &VgRasterShared {
        &self.shared
    }

    pub fn new(shared: VgRasterShared, run: &'static VgRun) -> Self {
        Self {
            shared,
            published: None,
            run,
            gate: None,
        }
    }

    /// Schedule this pass only while `gate` is open (a run whose content
    /// another pass decides, [`helio_vg_format::VgRunGate`]).
    pub fn with_gate(mut self, gate: helio_vg_format::VgRunGate) -> Self {
        self.gate = Some(gate);
        self
    }
}

impl RenderPass for VgRasterEarlyPass {
    /// A gated run's pass is a member only while its gate is open.
    fn is_member(&mut self, _scene: &mut helio_core::SceneContent<'_>) -> bool {
        self.gate.as_ref().is_none_or(|gate| gate.is_open())
    }

    fn name(&self) -> &'static str {
        self.run.names.raster_early
    }

    fn reads(&self) -> &'static [&'static str] {
        &[helio_mats::MATERIAL_TEXTURES]
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.read(self.run.inputs);
        builder.read(self.run.cull_early);
        builder.write_buffer(self.run.raster_early);
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        if cull_inputs(ctx.registry, self.run).is_some() {
            self.shared
                .ensure_targets(ctx.device, ctx.width, ctx.height);
        }
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        self.published = record_phase(&self.shared, ctx, self.run, self.run.cull_early, 0);
        Ok(())
    }

    fn publish<'a>(&self, frame: &mut helio_core::ResourceRegistry<'a>) {
        if let Some(p) = &self.published {
            frame.write(
                ResourceKey::new(self.run.raster_early),
                p.data(),
                self.run.names.raster_early,
            );
        }
    }
}

/// Phase 1: draw the late cull's delta, then classify dense tiles.
pub struct VgRasterLatePass {
    shared: VgRasterShared,
    published: Option<Published>,
    run: &'static VgRun,
    gate: Option<helio_vg_format::VgRunGate>,
}

impl VgRasterLatePass {
    pub fn new(shared: VgRasterShared, run: &'static VgRun) -> Self {
        Self {
            shared,
            published: None,
            run,
            gate: None,
        }
    }

    /// Schedule this pass only while `gate` is open (a run whose content
    /// another pass decides, [`helio_vg_format::VgRunGate`]).
    pub fn with_gate(mut self, gate: helio_vg_format::VgRunGate) -> Self {
        self.gate = Some(gate);
        self
    }
}

impl RenderPass for VgRasterLatePass {
    /// A gated run's pass is a member only while its gate is open.
    fn is_member(&mut self, _scene: &mut helio_core::SceneContent<'_>) -> bool {
        self.gate.as_ref().is_none_or(|gate| gate.is_open())
    }

    fn name(&self) -> &'static str {
        self.run.names.raster_late
    }

    fn reads(&self) -> &'static [&'static str] {
        &[helio_mats::MATERIAL_TEXTURES]
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.read(self.run.raster_early);
        builder.read(self.run.cull);
        builder.write_buffer(self.run.raster);
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        self.published = None;
        if ctx
            .registry
            .get::<VgRasterFrameData<'_>>(ResourceKey::new(self.run.raster_early))
            .is_none()
        {
            return Ok(());
        }
        self.published = record_phase(&self.shared, ctx, self.run, self.run.cull, 1);
        Ok(())
    }

    fn publish<'a>(&self, frame: &mut helio_core::ResourceRegistry<'a>) {
        if let Some(p) = &self.published {
            frame.write(
                ResourceKey::new(self.run.raster),
                p.data(),
                self.run.names.raster_late,
            );
        }
    }
}

// ── Hi-Z of the phase-0 depth ───────────────────────────────────────────────

const HIZ_MIP0_WGSL: &str = r#"
struct P { dst: vec2<u32>, pad: vec2<u32> }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var src_depth: texture_depth_2d;
@group(0) @binding(2) var dst: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn mip0(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid.xy >= p.dst)) { return; }
    let d = textureLoad(src_depth, vec2<i32>(gid.xy), 0);
    textureStore(dst, vec2<i32>(gid.xy), vec4<f32>(d, 0.0, 0.0, 1.0));
}
"#;

// Farthest depth of the 2x2 footprint. The engine's depth is reversed
// (`helio_core::depth`), so the farthest value is the smallest.
const HIZ_MIPN_WGSL: &str = r#"
struct P { dst: vec2<u32>, src: vec2<u32> }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var src_tex: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn mipn(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (any(gid.xy >= p.dst)) { return; }
    let last = p.src - vec2<u32>(1u, 1u);
    let s = gid.xy * 2u;
    let a = textureLoad(src_tex, vec2<i32>(min(s, last)), 0).r;
    let b = textureLoad(src_tex, vec2<i32>(min(s + vec2<u32>(1u, 0u), last)), 0).r;
    let c = textureLoad(src_tex, vec2<i32>(min(s + vec2<u32>(0u, 1u), last)), 0).r;
    let d = textureLoad(src_tex, vec2<i32>(min(s + vec2<u32>(1u, 1u), last)), 0).r;
    textureStore(dst, vec2<i32>(gid.xy), vec4<f32>(min(min(a, b), min(c, d)), 0.0, 0.0, 1.0));
}
"#;

struct HizLevel {
    group: wgpu::BindGroup,
    groups: (u32, u32),
}

/// What a [`VgHizPass`] builds its pyramid from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VgHizSource {
    /// The run's own phase-0 depth, into its `hiz_late` (the late cull's).
    PhaseZero,
    /// The frame's depth as the drawers before the run left it, into the run's
    /// `hiz_early` (a run drawn after other drawers, `VgRun::builds_hiz_early`).
    FrameDepth,
}

/// Builds a Hi-Z pyramid ([`VgHizSource`]) into a graph texture the run's cull
/// samples.
pub struct VgHizPass {
    run: &'static VgRun,
    source: VgHizSource,
    gate: Option<helio_vg_format::VgRunGate>,
    mip0: wgpu::ComputePipeline,
    mip0_bgl: wgpu::BindGroupLayout,
    mipn: wgpu::ComputePipeline,
    mipn_bgl: wgpu::BindGroupLayout,
    levels: Vec<HizLevel>,
    /// Source depth view the level 0 group was built for.
    built_for: Option<(wgpu::TextureView, u32, u32)>,
    mip_views: Vec<wgpu::TextureView>,
}

impl VgHizPass {
    /// Schedule this pass only while `gate` is open (a run whose content
    /// another pass decides, [`helio_vg_format::VgRunGate`]).
    pub fn with_gate(mut self, gate: helio_vg_format::VgRunGate) -> Self {
        self.gate = Some(gate);
        self
    }

    /// The late cull's Hi-Z of the run's phase-0 depth.
    pub fn late(device: &wgpu::Device, run: &'static VgRun) -> Self {
        Self::new(device, run, VgHizSource::PhaseZero)
    }

    pub fn new(device: &wgpu::Device, run: &'static VgRun, source: VgHizSource) -> Self {
        let cs = wgpu::ShaderStages::COMPUTE;
        let uniform = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: cs,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage_tex = wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: cs,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: wgpu::TextureFormat::R32Float,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        };
        let sampled = |sample_type| wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: cs,
            ty: wgpu::BindingType::Texture {
                sample_type,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let bgl = |label, entries: &[wgpu::BindGroupLayoutEntry]| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries,
            })
        };
        let mip0_bgl = bgl(
            "VG Hiz mip0",
            &[
                uniform(0),
                sampled(wgpu::TextureSampleType::Depth),
                storage_tex,
            ],
        );
        let mipn_bgl = bgl(
            "VG Hiz mipN",
            &[
                uniform(0),
                sampled(wgpu::TextureSampleType::Float { filterable: false }),
                storage_tex,
            ],
        );
        let pipeline = |label: &str, source: &str, entry: &str, bgl: &wgpu::BindGroupLayout| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(bgl)],
                immediate_size: 0,
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Self {
            run,
            source,
            gate: None,
            mip0: pipeline("VG Hiz mip0", HIZ_MIP0_WGSL, "mip0", &mip0_bgl),
            mipn: pipeline("VG Hiz mipN", HIZ_MIPN_WGSL, "mipn", &mipn_bgl),
            mip0_bgl,
            mipn_bgl,
            levels: Vec::new(),
            built_for: None,
            mip_views: Vec::new(),
        }
    }

    fn rebuild(
        &mut self,
        device: &wgpu::Device,
        texture: &wgpu::Texture,
        depth: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) {
        self.mip_views = (0..texture.mip_level_count())
            .map(|mip| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("VG Hiz mip"),
                    base_mip_level: mip,
                    mip_level_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        self.levels.clear();
        let mut size = (width.max(1), height.max(1));
        for (mip, dst_view) in self.mip_views.iter().enumerate() {
            let (layout, source, params): (_, wgpu::BindingResource<'_>, [u32; 4]) = if mip == 0 {
                (
                    &self.mip0_bgl,
                    wgpu::BindingResource::TextureView(depth),
                    [size.0, size.1, 0, 0],
                )
            } else {
                (
                    &self.mipn_bgl,
                    wgpu::BindingResource::TextureView(&self.mip_views[mip - 1]),
                    [size.0, size.1, 0, 0],
                )
            };
            // Source size of mip N is the extent of mip N-1.
            let mut params = params;
            if mip > 0 {
                let prev = ((width.max(1)) >> (mip - 1)).max(1);
                let prev_h = ((height.max(1)) >> (mip - 1)).max(1);
                params[2] = prev;
                params[3] = prev_h;
            }
            let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("VG Hiz params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("VG Hiz level"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: source,
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(dst_view),
                    },
                ],
            });
            self.levels.push(HizLevel {
                group,
                groups: (size.0.div_ceil(8), size.1.div_ceil(8)),
            });
            size = ((size.0 >> 1).max(1), (size.1 >> 1).max(1));
        }
        self.built_for = Some((depth.clone(), width, height));
    }
}

impl VgHizPass {
    fn output(&self) -> &'static str {
        match self.source {
            VgHizSource::PhaseZero => self.run.hiz_late,
            VgHizSource::FrameDepth => self.run.hiz_early,
        }
    }
}

impl RenderPass for VgHizPass {
    /// A gated run's pass is a member only while its gate is open.
    fn is_member(&mut self, _scene: &mut helio_core::SceneContent<'_>) -> bool {
        self.gate.as_ref().is_none_or(|gate| gate.is_open())
    }

    fn name(&self) -> &'static str {
        match self.source {
            VgHizSource::PhaseZero => self.run.names.hiz_late,
            VgHizSource::FrameDepth => self.run.names.hiz_early,
        }
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        match self.source {
            VgHizSource::PhaseZero => builder.read(self.run.raster_early),
            // After the drawers before the run: their depth.
            VgHizSource::FrameDepth => builder.read(helio_core::resource_keys::DEPTH),
        }
        // Full mip chain is allocated for R32Float by the graph (like `hiz`).
        builder.write_color(
            self.output(),
            ResourceFormat::R32Float,
            ResourceSize::MatchSurface,
        );
        builder.with_extra_usage(wgpu::TextureUsages::STORAGE_BINDING);
    }

    fn on_resize(&mut self, _device: &wgpu::Device, _size: helio_core::RenderSize) {
        self.levels.clear();
        self.mip_views.clear();
        self.built_for = None;
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        // Nothing to cull this frame: no pyramid.
        if ctx
            .registry
            .get::<VgCullInputs<'_>>(ResourceKey::new(self.run.inputs))
            .is_none()
        {
            return Ok(());
        }
        let (depth, width, height) = match self.source {
            VgHizSource::PhaseZero => {
                let Some(early) = ctx
                    .registry
                    .get::<VgRasterFrameData<'_>>(ResourceKey::new(self.run.raster_early))
                else {
                    return Ok(());
                };
                (early.depth, early.width, early.height)
            }
            VgHizSource::FrameDepth => (ctx.depth, ctx.width, ctx.height),
        };
        let Some(texture) = ctx.resource_pool.get_texture(self.output()) else {
            return Ok(());
        };
        let stale = self
            .built_for
            .as_ref()
            .is_none_or(|(view, w, h)| view != depth || *w != width || *h != height);
        if stale {
            self.rebuild(ctx.device, texture, depth, width, height);
        }
        let mut cmds = ctx.graphics_cmds();
        let mut pass = cmds.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(self.name()),
            timestamp_writes: None,
        });
        for (mip, level) in self.levels.iter().enumerate() {
            pass.set_pipeline(if mip == 0 { &self.mip0 } else { &self.mipn });
            pass.set_bind_group(0, &level.group, &[]);
            pass.dispatch_workgroups(level.groups.0, level.groups.1, 1);
        }
        Ok(())
    }
}

/// The three passes of `run`, sharing one [`VgRasterShared`].
pub struct VgRasterPasses {
    pub shared: VgRasterShared,
    pub early: VgRasterEarlyPass,
    pub hiz_late: VgHizPass,
    pub late: VgRasterLatePass,
}

impl VgRasterPasses {
    pub fn new(device: &wgpu::Device, config: VgRasterConfig, run: &'static VgRun) -> Self {
        let shared = VgRasterShared::new(device, config);
        Self {
            early: VgRasterEarlyPass::new(shared.clone(), run),
            hiz_late: VgHizPass::late(device, run),
            late: VgRasterLatePass::new(shared.clone(), run),
            shared,
        }
    }
}
