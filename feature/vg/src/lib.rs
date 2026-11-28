//! Graph glue for virtual geometry (plan `.agents/plans/virtual-geometry.md`
//! section 9, T10): the pass set, its order, and the two small passes that tie
//! the streaming store to the graph.
//!
//! The chain is added by [`add_virtual_geometry_passes`] (the
//! `virtual_geometry` node, [`VgFeature`]) at the end of the geometry stage
//! (after the classic G-buffer, portals and foliage), only when the
//! [`helio_vg_stream::VgSettings`] setting names a live [`VgRuntime`] and the
//! device supports it ([`virtual_geometry_supported`]); otherwise the graph is
//! exactly the classic one.
//!
//! ```text
//! GBuffer (classic; VG-active rows are skipped by the main-view cull)
//!  VgStream        service the store, start last frame's readback, stream pages
//!  VgBvh           instance BVH over `vg_instances`
//!  VgInputs        publishes `VgCullInputs` as `vg_cull_inputs`
//!  VgCullEarly     phase 0 against last frame's Hi-Z
//!  VgRasterEarly   Tier 1 HW + Tier 2 SW of phase 0
//!  VgHizLate       Hi-Z of the phase-0 depth -> `vg_hiz_late`
//!  VgCullLate      phase 1 against that Hi-Z
//!  VgRasterLate    delta of phase 1, tile classification
//!  VgStreamEnd     records the page-request readback
//!  VgDense         Tier 3 over dense tiles (3-NonRT depth peel, or 3-RT ray queries when ready)
//!  VgShade         vis resolve, material bin, shade -> G-buffer, final depth
//! portal mask, classic occurrences (`helio_portal_api`)
//!  VgPortalInstances  per live chain, the instances it may show, placed
//!  VgInputsSecondary ... VgShadeSecondary  the second run over them (no tree:
//!                  its cull seeds each instance)
//! foliage / decals / deferred light: unchanged
//! ```
//!
//! The second run ([`VgRun::SECONDARY`], [`add_portal_virtual_geometry_passes`])
//! draws virtual geometry through portal chains: its instances are the ones
//! [`VgPortalInstancesPass`] places per chain, its early cull tests one exact
//! phase against a Hi-Z of the frame's depth after the mask and the classic
//! occurrences (its own, never the direct run's), every pixel is clipped to
//! its chain, and its surfaces carry the chain's remap. It shares the one
//! cluster store: its page requests join the frame's feedback, read back with
//! the next frame's. Its passes are members only while the frame has live
//! chains and the scene VG instances.
//!
//! Every pass of the chain records nothing in a frame where `VgInputs` did not
//! publish `vg_cull_inputs` (virtual geometry off, no asset ready, or the
//! scene buffers not published yet), so an idle chain is nearly free.

use std::sync::Arc;

use helio_core::graph::ResourceBuilder;
use helio_core::{
    BufferKey, PassContext, PrepareContext, RenderGraph, RenderPass, ResourceKey,
    Result as HelioResult,
};
use helio_pass_vg_bvh::{keys as bvh_keys, VgBvhPass};
use helio_pass_vg_cull::{
    VgCullConfig, VgCullDemand, VgCullDynamic, VgCullEarlyPass, VgCullInputs, KEY_CULL, KEY_INPUTS,
};
use helio_pass_vg_dense::VgDensePass;
use helio_pass_vg_raster::{
    VgHizPass, VgHizSource, VgRasterConfig, VgRasterPasses, VgRasterShared, KEY_RASTER,
};
use helio_pass_vg_shade::VgShadePass;
use helio_vg_format::{VgInstanceSource, VgMode, VgRun, VgRunGate};
use helio_vg_stream::{VgRuntime, VgStreamBuffers};

/// Ordering marker written by [`VgStreamBeginPass`].
pub const KEY_STREAM: &str = "vg_stream";

/// Whether `device` can run the virtual-geometry passes: the cull's bind
/// group needs 19 storage buffers and a sampled texture (plan section 6: the
/// pass set is simply absent on devices below that, and classic rendering is
/// unchanged), the shade pass 8 storage textures, and the vertex-pulling
/// raster `INDIRECT_FIRST_INSTANCE`.
pub fn virtual_geometry_supported(device: &wgpu::Device) -> bool {
    let limits = device.limits();
    limits.max_storage_buffers_per_shader_stage >= 20
        && limits.max_storage_textures_per_shader_stage >= 8
        && limits.max_bind_groups >= 4
        && device
            .features()
            .contains(wgpu::Features::INDIRECT_FIRST_INSTANCE)
}

fn lock_free_stream<'a>(buffers: &'a StreamSnapshot) -> VgStreamBuffers<'a> {
    VgStreamBuffers {
        page_pool: &buffers.stream[0],
        residency: &buffers.stream[1],
        nodes: &buffers.stream[2],
        group_refs: &buffers.stream[3],
        asset_headers: &buffers.stream[4],
        asset_bases: &buffers.stream[5],
        asset_quant: &buffers.stream[6],
        feedback: &buffers.stream[7],
    }
}

struct StreamSnapshot {
    stream: [wgpu::Buffer; 8],
}

/// Services the store and starts the streaming step before any VG pass
/// records.
pub struct VgStreamBeginPass {
    runtime: VgRuntime,
}

impl RenderPass for VgStreamBeginPass {
    fn name(&self) -> &'static str {
        "VgStream"
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.write_buffer(KEY_STREAM);
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        self.runtime.service(ctx.device, ctx.queue);
        self.runtime.with_store(|store| {
            // The previous frame's readback was recorded and submitted: map it,
            // then collect whatever finished and schedule uploads.
            store.after_submit();
            store.update();
        });
        Ok(())
    }

    fn execute(&mut self, _ctx: &mut PassContext) -> HelioResult<()> {
        Ok(())
    }
}

struct InputsSnapshot {
    instances: wgpu::Buffer,
    objects: wgpu::Buffer,
    remaps: Option<wgpu::Buffer>,
    stream: StreamSnapshot,
    /// The run's instance BVH (nodes, header, items), if it has one.
    tree: Option<[wgpu::Buffer; 3]>,
    dense_mask: Option<wgpu::Buffer>,
    materials: Option<wgpu::Buffer>,
    instance_count: u32,
    demand: VgCullDemand,
    dense_tiles: [u32; 2],
}

/// Publishes [`VgCullInputs`] (`vg_cull_inputs`) when virtual geometry has
/// something to draw this frame.
pub struct VgInputsPass {
    runtime: VgRuntime,
    raster: VgRasterShared,
    dense_enabled: bool,
    snapshot: Option<InputsSnapshot>,
    run: &'static VgRun,
    gate: Option<helio_vg_format::VgRunGate>,
}

impl RenderPass for VgInputsPass {
    /// A gated run's pass is a member only while its gate is open.
    fn is_member(&mut self, _scene: &mut helio_core::SceneContent<'_>) -> bool {
        self.gate.as_ref().is_none_or(|gate| gate.is_open())
    }

    fn name(&self) -> &'static str {
        self.run.names.inputs
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        // The frame's run services the stream first; a later run reads what
        // it streamed.
        if !self.run.remapped() {
            builder.read(KEY_STREAM);
        }
        if let Some(tree) = self.run.tree {
            builder.read(tree.nodes);
            builder.read(tree.header);
            builder.read(tree.items);
        }
        builder.write_buffer(self.run.inputs);
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        self.snapshot = None;
        let settings = self.runtime.settings();
        let debug = std::env::var_os("VG_DEBUG").is_some();
        if settings.mode == VgMode::Off || !self.runtime.any_ready() {
            if debug {
                eprintln!(
                    "VgInputs: idle (mode {:?}, ready {})",
                    settings.mode,
                    self.runtime.any_ready()
                );
            }
            return Ok(());
        }
        let reg = ctx.registry;
        let (instances, objects, remaps) = match self.run.instances {
            VgInstanceSource::Scene => {
                let (Some(instances), Some(objects)) = (
                    ctx.scene_buffers.get(BufferKey::of(bvh_keys::INSTANCES)),
                    ctx.scene_buffers
                        .get(BufferKey::of(bvh_keys::STATIC_OBJECTS)),
                ) else {
                    if debug {
                        eprintln!("VgInputs: vg_instances / static_objects not published");
                    }
                    return Ok(());
                };
                (&instances.buffer, &objects.buffer, None)
            }
            VgInstanceSource::Registry {
                instances,
                objects,
                remaps,
            } => {
                let (Some(instances), Some(objects), Some(remaps)) = (
                    reg.buffer(ResourceKey::new(instances)),
                    reg.buffer(ResourceKey::new(objects)),
                    reg.buffer(ResourceKey::new(remaps)),
                ) else {
                    return Ok(());
                };
                (instances, objects, Some(remaps))
            }
        };
        let tree = match self.run.tree {
            Some(keys) => {
                let (Some(nodes), Some(header), Some(items)) = (
                    reg.buffer(ResourceKey::new(keys.nodes)),
                    reg.buffer(ResourceKey::new(keys.header)),
                    reg.buffer(ResourceKey::new(keys.items)),
                ) else {
                    return Ok(());
                };
                Some([nodes.clone(), header.clone(), items.clone()])
            }
            None => None,
        };
        let Some((stream, largest)) = self.runtime.with_store(|store| {
            let b = store.stream_buffers();
            let stream = StreamSnapshot {
                stream: b.as_array().map(|buffer| buffer.clone()),
            };
            (stream, store.largest_asset())
        }) else {
            return Ok(());
        };
        self.raster
            .ensure_targets(ctx.device, ctx.width, ctx.height);
        self.raster
            .set_dense_sensitivity(settings.dense_sensitivity);
        let (dense_mask, dense_tiles) = match self.raster.dense_mask_last_frame() {
            Some((mask, tiles))
                if self.dense_enabled && settings.tier3 && self.runtime.tier3_stats().engaged =>
            {
                (Some(mask), tiles)
            }
            Some((_, tiles)) => (None, tiles),
            None => (None, [1, 1]),
        };
        if debug {
            eprintln!(
                "{}: ACTIVE, instance rows {}",
                self.run.names.inputs,
                instances.size() / 16
            );
        }
        let instance_count = (instances.size()
            / std::mem::size_of::<helio_vg_format::GpuVgInstance>() as u64)
            as u32;
        self.snapshot = Some(InputsSnapshot {
            instance_count,
            // Every row: how many are live is known only to the GPU.
            demand: VgCullDemand::for_instances(instance_count, largest),
            instances: instances.clone(),
            objects: objects.clone(),
            remaps: remaps.cloned(),
            stream,
            tree,
            dense_mask,
            materials: ctx
                .scene_buffers
                .get(helio_mats::MATERIALS.key())
                .map(|m| m.buffer.clone()),
            dense_tiles,
        });
        Ok(())
    }

    fn execute(&mut self, _ctx: &mut PassContext) -> HelioResult<()> {
        Ok(())
    }

    fn publish<'a>(&self, frame: &mut helio_core::ResourceRegistry<'a>) {
        let Some(s) = &self.snapshot else { return };
        // SAFETY: the buffers are clones owned by `self`, which the graph keeps
        // alive for longer than the frame registry; they are only replaced in
        // `prepare`, before this frame's publish.
        let inputs: VgCullInputs<'a> = unsafe {
            std::mem::transmute::<VgCullInputs<'_>, VgCullInputs<'a>>(VgCullInputs {
                instances: &s.instances,
                objects: &s.objects,
                stream: lock_free_stream(&s.stream),
                tree: s.tree.as_ref().map(|[nodes, header, items]| {
                    helio_pass_vg_cull::VgCullTree {
                        nodes,
                        header,
                        items,
                    }
                }),
                dense_mask: s.dense_mask.as_ref(),
                materials: s.materials.as_ref(),
                instance_count: s.instance_count,
                demand: s.demand,
                dense_tiles: s.dense_tiles,
                remaps: s.remaps.as_ref(),
            })
        };
        frame.write(
            ResourceKey::new(self.run.inputs),
            inputs,
            self.run.names.inputs,
        );
    }
}

/// Records the page-request readback after the last pass that appends
/// requests, so the next frame's `VgStream` can decode it.
pub struct VgStreamEndPass {
    runtime: VgRuntime,
}

impl RenderPass for VgStreamEndPass {
    fn name(&self) -> &'static str {
        "VgStreamEnd"
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.read(KEY_CULL);
        builder.read(KEY_RASTER);
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        if ctx
            .registry
            .get::<VgCullInputs<'_>>(ResourceKey::new(KEY_INPUTS))
            .is_none()
        {
            return Ok(());
        }
        let Some(plan) = self
            .runtime
            .with_store(|store| store.begin_feedback_readback())
            .flatten()
        else {
            return Ok(());
        };
        let mut rec = ctx.graphics_cmds();
        rec.copy_buffer_to_buffer(&plan.src, 0, &plan.staging, 0, plan.bytes);
        for (offset, size) in plan.clears {
            rec.clear_buffer(&plan.src, offset, Some(size));
        }
        Ok(())
    }
}

/// The settings the cull re-reads every frame.
fn cull_settings(runtime: &VgRuntime) -> VgCullDynamic {
    let s = runtime.settings();
    VgCullDynamic {
        tau_px: s.tau_px,
        prefetch_factor: s.prefetch_factor,
        tile_density_limit: s.tile_density_limit,
        mode: s.mode,
        two_phase_occlusion: s.two_phase_occlusion,
        max_visible_clusters: s.max_visible_clusters,
    }
}

/// Adds the virtual-geometry chain to `graph` if `settings` name a runtime
/// and the device supports it, for `view` (`helio_core::view`). Returns
/// whether it was added.
///
/// Streaming is the frame's: a secondary view (a reflection, a portal's)
/// culls, rasterizes and shades the clusters the frame's own view streamed,
/// and adds no stream begin or end of its own (two would service the store
/// and read back feedback twice a frame).
pub fn add_virtual_geometry_passes(
    graph: &mut RenderGraph,
    device: &Arc<wgpu::Device>,
    settings: &helio_vg_stream::VgSettings,
    view: helio_core::ViewDescription,
) -> bool {
    let Some(id) = settings.runtime else {
        return false;
    };
    let Some(runtime) = VgRuntime::from_handle_id(id) else {
        log::warn!("virtual geometry: runtime {id} is gone; building the classic graph");
        return false;
    };
    // The host keeps instances' full classic geometry where the renderer
    // cannot draw their clusters, and releases it where it can.
    let supported = virtual_geometry_supported(device);
    runtime.set_renderer_support(supported);
    if !supported {
        log::info!(
            "virtual geometry: device lacks {} storage buffers per stage or INDIRECT_FIRST_INSTANCE; \
             classic rendering only",
            20
        );
        return false;
    }

    add_run(graph, device, runtime, view, &VgRun::DIRECT, None);
    true
}

/// Adds the second virtual-geometry run ([`VgRun::SECONDARY`]): virtual
/// geometry through the frame's portal chains, after the mask and the classic
/// occurrences. Returns whether it was added (as [`add_virtual_geometry_passes`]).
pub fn add_portal_virtual_geometry_passes(
    graph: &mut RenderGraph,
    device: &Arc<wgpu::Device>,
    settings: &helio_vg_stream::VgSettings,
    view: helio_core::ViewDescription,
) -> bool {
    // A view other than the frame's draws no portal chains.
    if !view.is_primary() {
        return false;
    }
    let Some(runtime) = settings.runtime.and_then(VgRuntime::from_handle_id) else {
        return false;
    };
    if !virtual_geometry_supported(device) {
        return false;
    }
    let gate = VgRunGate::default();
    graph.add_pass(Box::new(VgPortalInstancesPass::new(device, gate.clone())));
    add_run(graph, device, runtime, view, &VgRun::SECONDARY, Some(gate));
    true
}

/// The passes of `run`: the frame's own run streams (on the frame's view);
/// a run over published instances ([`VgRun::remapped`]) culls one exact phase
/// against the depth it is drawn over, clips to its chains, has no Tier 3,
/// and is scheduled only while `gate` is open.
fn add_run(
    graph: &mut RenderGraph,
    device: &Arc<wgpu::Device>,
    runtime: VgRuntime,
    view: helio_core::ViewDescription,
    run: &'static VgRun,
    gate: Option<VgRunGate>,
) {
    let secondary = run.remapped();
    let raster = VgRasterPasses::new(
        device,
        VgRasterConfig {
            clip: view.clipped,
            chain_clip: secondary,
            ..VgRasterConfig::default()
        },
        run,
    );
    let raster_shared = raster.shared.clone();
    let tile_fallback = raster.shared.uses_tile_fallback();
    if !secondary {
        runtime.update_tier3_stats(|s| s.tier2_tile_fallback = tile_fallback);
    }

    let source_runtime = runtime.clone();
    let cull_config = VgCullConfig {
        sw_masked: !raster.shared.uses_tile_fallback(),
        current_hiz: run.builds_hiz_early,
        seeds_instances: run.tree.is_none(),
        ..VgCullConfig::default()
    };
    let (cull_early, cull_late) = VgCullEarlyPass::pair_for(device, cull_config, run);
    let cull_early =
        cull_early.with_settings_source(Arc::new(move || cull_settings(&source_runtime)));

    let mode_runtime = runtime.clone();
    // Tier 3 is opt-in (`VgRuntimeSettings::tier3`): with it off the dense pass
    // sees `Off` and records nothing, and the cull never routes clusters to it.
    // A later run leaves dense tiles to Tier 1/2.
    let dense = if secondary {
        None
    } else {
        match VgDensePass::new(
            device,
            Box::new(move || {
                let settings = mode_runtime.settings();
                if settings.tier3 {
                    settings.mode
                } else {
                    VgMode::Off
                }
            }),
        ) {
            // The runtime gives Tier 3 the stream store (proxy sources, page
            // residency) so it can use the ray-traced backend where supported.
            Ok(dense) => Some(dense.with_runtime(runtime.clone())),
            Err(error) => {
                log::warn!(
                    "virtual geometry: Tier 3 unavailable ({error}); dense tiles stay on Tier 1/2"
                );
                None
            }
        }
    };

    let streams = !secondary && view.is_primary();
    if streams {
        graph.add_pass(Box::new(VgStreamBeginPass {
            runtime: runtime.clone(),
        }));
    }
    let inputs = VgInputsPass {
        runtime: runtime.clone(),
        raster: raster_shared,
        dense_enabled: dense.is_some(),
        snapshot: None,
        run,
        gate: gate.clone(),
    };
    let tau_runtime = runtime.clone();
    let shade = VgShadePass::for_run(device, dense.is_some(), run)
        .with_tau_source(Arc::new(move || tau_runtime.settings().tau_px));
    match gate {
        None => {
            graph.add_pass(Box::new(VgBvhPass::new(device)));
            graph.add_pass(Box::new(inputs));
            graph.add_pass(Box::new(cull_early));
            graph.add_pass(Box::new(raster.early));
            graph.add_pass(Box::new(raster.hiz_late));
            graph.add_pass(Box::new(cull_late));
            graph.add_pass(Box::new(raster.late));
            if streams {
                graph.add_pass(Box::new(VgStreamEndPass { runtime }));
            }
            if let Some(dense) = dense {
                graph.add_pass(Box::new(dense));
            }
            graph.add_pass(Box::new(shade));
        }
        Some(gate) => {
            // Its instances are what the producer's spatial query found: no
            // tree (the cull seeds each), one exact phase against this
            // frame's depth, no phase-0 Hi-Z.
            graph.add_pass(Box::new(inputs));
            if run.builds_hiz_early {
                graph.add_pass(Box::new(
                    VgHizPass::new(device, run, VgHizSource::FrameDepth).with_gate(gate.clone()),
                ));
            }
            graph.add_pass(Box::new(cull_early.with_gate(gate.clone())));
            graph.add_pass(Box::new(raster.early.with_gate(gate.clone())));
            graph.add_pass(Box::new(cull_late.with_gate(gate.clone())));
            graph.add_pass(Box::new(raster.late.with_gate(gate.clone())));
            graph.add_pass(Box::new(shade.with_gate(gate)));
        }
    }
}

mod feature;
mod portal_instances;
pub use feature::VgFeature;
pub use portal_instances::VgPortalInstancesPass;

/// `config` building the virtual-geometry passes for `runtime` (when the
/// device supports them). `VG_DEBUG_VIEW` selects a virtual-geometry debug
/// view at creation (tests, headless captures); an explicit `debug_mode` is
/// left alone.
pub fn with_virtual_geometry(
    mut config: helio::RendererConfig,
    runtime: &helio_vg_stream::VgRuntime,
) -> helio::RendererConfig {
    config
        .features
        .insert(helio_vg_stream::VgSettings::for_runtime(runtime));
    if config.debug_mode == 0 {
        if let Some(view) = helio_vg_format::debug_view::from_env() {
            config.debug_mode = view.id();
        }
    }
    config
}
