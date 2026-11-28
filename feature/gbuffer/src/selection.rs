//! The meshes in the editor's selection mask
//! (`helio_frame_api::Phase::SelectionMask`), for the selection outline.
//!
//! Two passes, sharing their buffers ([`SelectionDraws`]):
//!
//! - [`SelectionSelectPass`], compute before the phase: of this frame's
//!   culled draws (the G-buffer's), each instance whose object is selected
//!   becomes a draw of its own. An instance finds its object through
//!   [`helio_mesh_api::STATIC_OBJECT_OWNERS`] (structure, written by the
//!   scene join) and reads the host's selection rows
//!   ([`helio_frame_api::selection::OBJECT_SELECTED`]), so a selection change
//!   writes one row per object and no join or batch work follows it.
//! - [`SelectionMaskPass`], the phase's drawer: those draws, depth only and
//!   unjittered, placed as the G-buffer places them.
//!
//! The selection costs one invocation per visible draw, and drawing it costs
//! what is selected. Outside the editor neither pass records anything.

use std::sync::{Arc, Mutex};

use helio_core::graph::{ResourceBuilder, ResourceSize};
use helio_core::{PassContext, PrepareContext, RenderPass, ResourceKey, Result as HelioResult};
use helio_frame_api::selection::{MASK_DEPTH, OBJECT_SELECTED};
use helio_mesh_api::{
    multi_draw_indexed_indirect, CulledBatchFrameData, DrawSlots, GpuDrawCount,
    ObjectBatchFrameData, DRAW_SLOT_LAYOUT, STATIC_OBJECT_OWNERS,
};

/// Bytes of one indirect draw (`DrawIndexedIndirectArgs`).
const DRAW_BYTES: u64 = std::mem::size_of::<helio_mesh_api::DrawIndexedIndirectArgs>() as u64;
const WORKGROUP: u32 = 64;
/// The graph resource that orders the drawer after the selection.
const SELECTION_DRAWS: &str = "mesh_selection_draws";

/// The selected draws, written by [`SelectionSelectPass`] and drawn by
/// [`SelectionMaskPass`] (the two passes' shared state, one per graph).
#[derive(Default)]
pub struct SelectionDraws {
    /// One `DrawIndexedIndirectArgs` per selected visible instance.
    draws: Option<wgpu::Buffer>,
    /// `[count, ..]`: how many of `draws` were written (may exceed `capacity`).
    count: Option<wgpu::Buffer>,
    /// Rows `draws` holds.
    capacity: u32,
    /// Whether this frame's selection ran (the editor, with selection rows and
    /// mesh owners present).
    active: bool,
}

/// The shared selection state for a graph (`BuildEnv::shared`).
pub struct SharedSelectionDraws(pub Arc<Mutex<SelectionDraws>>);

impl SelectionDraws {
    fn ensure(&mut self, device: &wgpu::Device, capacity: u32) {
        let capacity = capacity.max(1);
        if self.draws.is_some() && capacity <= self.capacity {
            return;
        }
        let capacity = capacity.next_power_of_two();
        let indirect = wgpu::BufferUsages::INDIRECT
            | wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST;
        self.draws = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Mesh Selection Draws"),
            size: DRAW_BYTES * u64::from(capacity),
            usage: indirect,
            mapped_at_creation: false,
        }));
        self.count = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Mesh Selection Draw Count"),
            size: 16,
            usage: indirect,
            mapped_at_creation: false,
        }));
        self.capacity = capacity;
    }
}

fn lock(shared: &Mutex<SelectionDraws>) -> std::sync::MutexGuard<'_, SelectionDraws> {
    shared
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SelectParams {
    draw_capacity: u32,
    out_capacity: u32,
    gpu_count: u32,
    count_word: u32,
}

/// Turns this frame's culled draws of selected objects into
/// [`SelectionDraws`]. Compute before the selection mask phase.
pub struct SelectionSelectPass {
    editor: bool,
    shared: Arc<Mutex<SelectionDraws>>,
    pipeline: helio_core::shader::DeferredCompute,
    params: wgpu::Buffer,
    bind: Option<(Vec<wgpu::Buffer>, wgpu::BindGroup)>,
}

impl SelectionSelectPass {
    pub fn new(device: &wgpu::Device, shared: Arc<Mutex<SelectionDraws>>) -> Self {
        let shader = helio_core::shader::module_with(
            device,
            "Mesh Selection Select",
            helio_core::include_wgsl!("../shaders/selection_select.wgsl"),
            &[helio_mesh_api::DRAW_ROWS_SNIPPET],
        );
        let pipeline = helio_core::shader::create_compute_pipeline_deferred(
            device,
            wgpu::ComputePipelineDescriptor {
                label: Some("Mesh Selection Select"),
                layout: None,
                module: &shader,
                entry_point: Some("cs_select"),
                compilation_options: Default::default(),
                cache: None,
            },
        );
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Mesh Selection Params"),
            size: std::mem::size_of::<SelectParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            editor: false,
            shared,
            pipeline,
            params,
            bind: None,
        }
    }
}

impl RenderPass for SelectionSelectPass {
    fn name(&self) -> &'static str {
        "MeshSelectionSelect"
    }

    fn scene_buffers(&self) -> &'static [&'static str] {
        const KEYS: &[&str] = &[STATIC_OBJECT_OWNERS.name(), OBJECT_SELECTED.name()];
        KEYS
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.read("object_batch");
        builder.read("culled_batch");
        builder.write_buffer(SELECTION_DRAWS);
    }

    fn publish<'a>(&self, frame: &mut helio_core::ResourceRegistry<'a>) {
        frame.write(ResourceKey::new(SELECTION_DRAWS), (), "MeshSelectionSelect");
    }

    fn set_editor_mode(&mut self, enabled: bool) {
        self.editor = enabled;
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let capacity = ctx
            .registry
            .get::<ObjectBatchFrameData<'_>>(ResourceKey::new("object_batch"))
            .map_or(0, |batch| batch.group_capacity);
        lock(&self.shared).ensure(ctx.device, capacity);
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let mut shared = lock(&self.shared);
        shared.active = false;
        if !self.editor {
            return Ok(());
        }
        let Some(batch) = ctx
            .registry
            .get::<ObjectBatchFrameData<'_>>(ResourceKey::new("object_batch"))
        else {
            return Ok(());
        };
        let Some(culled) = ctx
            .registry
            .get::<CulledBatchFrameData<'_>>(ResourceKey::new("culled_batch"))
        else {
            return Ok(());
        };
        let (Some(owners), Some(selected)) = (
            ctx.scene_buffers.get(STATIC_OBJECT_OWNERS.key()),
            ctx.scene_buffers.get(OBJECT_SELECTED.key()),
        ) else {
            return Ok(());
        };
        let (Some(draws), Some(count)) = (shared.draws.clone(), shared.count.clone()) else {
            return Ok(());
        };
        let (max_draws, gpu_count) = batch.all_draws();
        let gpu_count = gpu_count.filter(|count| count.offset % 4 == 0);
        ctx.write_buffer(
            &self.params,
            0,
            bytemuck::bytes_of(&SelectParams {
                draw_capacity: max_draws,
                out_capacity: shared.capacity,
                gpu_count: u32::from(gpu_count.is_some()),
                count_word: gpu_count.map_or(0, |count| (count.offset / 4) as u32),
            }),
        );
        let bound = vec![
            self.params.clone(),
            culled.indirect.clone(),
            gpu_count.map_or_else(|| count.clone(), |count| count.buffer.clone()),
            culled.compacted_indices.clone(),
            batch.instances.clone(),
            owners.buffer.clone(),
            selected.buffer.clone(),
            draws.clone(),
            count.clone(),
        ];
        if self
            .bind
            .as_ref()
            .is_none_or(|(cached, _)| *cached != bound)
        {
            let layout = self.pipeline.get_bind_group_layout(0);
            let entries: Vec<wgpu::BindGroupEntry<'_>> = bound
                .iter()
                .enumerate()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect();
            let group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Mesh Selection Select"),
                layout: &layout,
                entries: &entries,
            });
            self.bind = Some((bound, group));
        }
        let mut cmds = ctx.graphics_cmds();
        // Unused rows must read as empty draws where the device draws a fixed
        // number of them (no GPU count).
        cmds.clear_buffer(&draws, 0, None);
        cmds.clear_buffer(&count, 0, None);
        if max_draws > 0 {
            let mut pass = cmds.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Mesh Selection Select"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind.as_ref().expect("bound above").1, &[]);
            pass.dispatch_workgroups(max_draws.div_ceil(WORKGROUP).min(65_535), 1, 1);
        }
        shared.active = true;
        Ok(())
    }
}

/// Draws [`SelectionDraws`] into the selection mask, depth only and
/// unjittered: the meshes' drawer of `Phase::SelectionMask`.
pub struct SelectionMaskPass {
    editor: bool,
    shared: Arc<Mutex<SelectionDraws>>,
    pipeline: helio_core::shader::DeferredRender,
    draw_slots: DrawSlots,
    /// Bound for an absent optional input (no skinned meshes, no coordinate
    /// spaces): one identity matrix.
    identity: wgpu::Buffer,
    bind: Option<(Vec<wgpu::Buffer>, wgpu::BindGroup)>,
}

impl SelectionMaskPass {
    pub fn new(device: &wgpu::Device, shared: Arc<Mutex<SelectionDraws>>) -> Self {
        let shader = helio_core::shader::module_with(
            device,
            "Mesh Selection Mask",
            helio_core::include_wgsl!("../shaders/selection_mask.wgsl"),
            &[
                helio_mesh_api::DRAW_ROWS_SNIPPET,
                helio_mesh_api::SKINNING_SNIPPET,
                helio_frame_api::rays::stream::SNIPPET,
            ],
        );
        let pipeline = helio_core::shader::create_render_pipeline_deferred(
            device,
            wgpu::RenderPipelineDescriptor {
                label: Some("Mesh Selection Mask"),
                layout: None,
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[
                        Some(wgpu::VertexBufferLayout {
                            array_stride: helio_mesh_api::PackedVertex::STRIDE,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: &[helio_mesh_api::PackedVertex::POSITION],
                        }),
                        Some(DRAW_SLOT_LAYOUT),
                    ],
                },
                fragment: None,
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    // The silhouette, whatever the winding (a mirrored or
                    // double-sided mesh outlines the same).
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(helio_core::depth::NEARER_EQUAL),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            },
        );
        let identity = wgpu::util::DeviceExt::create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("Mesh Selection Identity"),
                contents: bytemuck::bytes_of(&glam::Mat4::IDENTITY.to_cols_array()),
                usage: wgpu::BufferUsages::STORAGE,
            },
        );
        Self {
            editor: false,
            shared,
            pipeline,
            draw_slots: DrawSlots::default(),
            identity,
            bind: None,
        }
    }
}

impl RenderPass for SelectionMaskPass {
    fn name(&self) -> &'static str {
        "MeshSelectionMask"
    }

    fn scene_buffers(&self) -> &'static [&'static str] {
        const KEYS: &[&str] = &[
            helio_mesh_api::MESH_VERTICES.name(),
            helio_mesh_api::MESH_INDICES.name(),
            helio_mesh_api::SKIN_CACHE.name(),
            helio_mesh_api::SKIN_CACHE_ROWS.name(),
        ];
        KEYS
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.read("object_batch");
        builder.read("culled_batch");
        builder.read(SELECTION_DRAWS);
        builder.write_depth(MASK_DEPTH, ResourceSize::MatchSurface);
    }

    fn set_editor_mode(&mut self, enabled: bool) {
        self.editor = enabled;
    }

    fn render_pass_descriptor_with_storage<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        resources: &'a helio_core::ResourceRegistry<'a>,
        _storage: &'a mut helio_core::RenderFrameStorage,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        if !self.editor {
            return None;
        }
        let mask =
            resources.read_texture_view(ResourceKey::new(MASK_DEPTH), "MeshSelectionMask")?;
        Some(wgpu::RenderPassDescriptor {
            label: Some("Mesh Selection Mask"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: mask,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        if !self.editor {
            return Ok(());
        }
        let (draws, count, capacity) = {
            let shared = lock(&self.shared);
            match (shared.active, shared.draws.clone(), shared.count.clone()) {
                (true, Some(draws), Some(count)) => (draws, count, shared.capacity),
                _ => return Ok(()),
            }
        };
        let Some(batch) = ctx
            .registry
            .get::<ObjectBatchFrameData<'_>>(ResourceKey::new("object_batch"))
        else {
            return Ok(());
        };
        let Some(culled) = ctx
            .registry
            .get::<CulledBatchFrameData<'_>>(ResourceKey::new("culled_batch"))
        else {
            return Ok(());
        };
        let (Some(vertices), Some(indices)) = (
            ctx.scene_buffers.get(helio_mesh_api::MESH_VERTICES.key()),
            ctx.scene_buffers.get(helio_mesh_api::MESH_INDICES.key()),
        ) else {
            return Ok(());
        };
        let spaces = ctx
            .registry
            .get::<helio_core::CoordinateSpacesFrameData<'_>>(
                helio_core::resource_keys::coordinate_spaces(),
            )
            .map_or(&self.identity, |spaces| spaces.coordinate_spaces);
        let skin = ctx
            .scene_buffers
            .get(helio_mesh_api::SKIN_CACHE.key())
            .map_or(&self.identity, |handle| &handle.buffer);
        let palette = ctx
            .scene_buffers
            .get(helio_mesh_api::SKIN_CACHE_ROWS.key())
            .map_or(&self.identity, |handle| &handle.buffer);
        let bound = vec![
            ctx.camera.clone(),
            batch.instances.clone(),
            culled.compacted_indices.clone(),
            spaces.clone(),
            skin.clone(),
            palette.clone(),
        ];
        if self
            .bind
            .as_ref()
            .is_none_or(|(cached, _)| *cached != bound)
        {
            let layout = self.pipeline.get_bind_group_layout(0);
            let entries: Vec<wgpu::BindGroupEntry<'_>> = bound
                .iter()
                .enumerate()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect();
            let group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Mesh Selection Mask"),
                layout: &layout,
                entries: &entries,
            });
            self.bind = Some((bound, group));
        }
        // The slot indexes `compacted_indices`.
        let slots = self
            .draw_slots
            .buffer(
                &ctx.device,
                DrawSlots::len_of(culled.compacted_indices.size(), 4),
            )
            .clone();
        let gpu_count = ctx
            .device
            .features()
            .contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT)
            .then_some(GpuDrawCount {
                buffer: &count,
                offset: 0,
            });
        let Some(mut pass) = ctx.render_cmds() else {
            return Ok(());
        };
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind.as_ref().expect("bound above").1, &[]);
        pass.set_vertex_buffer(0, vertices.buffer.slice(..));
        pass.set_vertex_buffer(1, slots.slice(..));
        pass.set_index_buffer(indices.buffer.slice(..), wgpu::IndexFormat::Uint32);
        multi_draw_indexed_indirect(&mut pass, &draws, 0, capacity, gpu_count);
        Ok(())
    }

    fn on_resize(&mut self, _device: &wgpu::Device, _size: helio_core::RenderSize) {}
}
