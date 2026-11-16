//! The meshes' occurrences through portal chains (`helio_portal_api`).
//!
//! While the frame has live chains, [`PortalOccurrencePass`] finds, per
//! draw group, the instances each chain may show (beyond every opening of
//! the chain, inside the frustum once mapped through it), packs them per
//! group without a reserved region (count, scan, write) and compacts the
//! groups' records into the occlusion cull's material segments, with the
//! same range compaction. The G-buffer's occurrence draw reads the result
//! ([`helio_mesh_api::PortalOccurrenceFrameData`]) and draws each segment
//! with the material's pipeline, as the direct draw does.
//!
//! The output buffers grow from the total the GPU found (read back a frame
//! later); a frame that finds more occurrences than they hold draws the
//! ones that fit.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};
use helio_mesh_api::{
    CulledBatchFrameData, DrawSegments, ObjectBatchFrameData, PortalOccurrenceFrameData,
    PORTAL_OCCURRENCES,
};

use crate::{buffer_entry, storage_layout_entry, RangeCompaction, RangeCompactParams};

/// `portal_occurrences.wgsl`'s `OccurrenceParams`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct OccurrenceParams {
    frustum_planes: [[f32; 4]; 6],
    draw_capacity: u32,
    chain_count: u32,
    capacity: u32,
    groups_per_row: u32,
}

/// The fewest occurrences the buffers hold.
const MIN_OCCURRENCES: u32 = 1024;
/// Workgroups along x before the dispatch folds into z.
const GROUPS_PER_ROW: u32 = 65_535;

pub struct PortalOccurrencePass {
    count_pipeline: wgpu::ComputePipeline,
    scan_pipeline: wgpu::ComputePipeline,
    write_pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    bind_group: helio_core::CachedBindGroup,
    params: wgpu::Buffer,
    compaction: RangeCompaction,
    compaction_group: helio_core::CachedBindGroup,
    /// Per group: occurrences found, and the write cursor.
    group_counts: wgpu::Buffer,
    group_cursors: wgpu::Buffer,
    /// Per group: its indirect record (offset and kept count).
    indirect: wgpu::Buffer,
    /// The same records, compacted per range (the compaction's scratch).
    compacted: wgpu::Buffer,
    /// The compaction's per-range counts (its scratch: the batch's stay).
    draw_counts: wgpu::Buffer,
    /// Per occurrence: the instance slot and the chain.
    instances: wgpu::Buffer,
    chains: wgpu::Buffer,
    /// The occurrences' records per segment, and the counts per segment.
    segment_indirect: wgpu::Buffer,
    segment_counts: wgpu::Buffer,
    totals: wgpu::Buffer,
    capacity: u32,
    group_capacity: u32,
    /// The total copied for reading back, waiting to be mapped or read.
    readback: Option<(helio_core::gpu_cache::PooledBuffer, Arc<AtomicBool>, bool)>,
    readbacks: helio_core::gpu_cache::ReadbackPool,
    /// This frame's live chains (0: nothing to do).
    chain_count: u32,
    draw_capacity: u32,
    /// Whether the frame publishes chains (a graph member only then).
    live: bool,
}

fn storage_buffer(device: &wgpu::Device, label: &str, size: u64, indirect: bool) -> wgpu::Buffer {
    let mut usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
    if indirect {
        usage |= wgpu::BufferUsages::INDIRECT;
    }
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(16),
        usage: usage | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

impl PortalOccurrencePass {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = helio_core::shader::module_with(
            device,
            "PortalOccurrences Shader",
            helio_core::include_wgsl!("../shaders/portal_occurrences.wgsl"),
            &[
                helio_mesh_api::DRAW_ROWS_SNIPPET,
                helio_portal_api::PORTAL_CHAIN_SNIPPET,
            ],
        );
        let uniform = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PortalOccurrences BGL"),
            entries: &[
                uniform,
                storage_layout_entry(1, true),
                storage_layout_entry(2, true),
                storage_layout_entry(3, true),
                storage_layout_entry(4, true),
                storage_layout_entry(5, true),
                storage_layout_entry(6, true),
                storage_layout_entry(7, false),
                storage_layout_entry(8, false),
                storage_layout_entry(9, false),
                storage_layout_entry(10, false),
                storage_layout_entry(11, false),
                storage_layout_entry(12, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PortalOccurrences PL"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |entry: &'static str| {
            helio_core::shader::create_compute_pipeline(
                device,
                wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                },
            )
        };
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PortalOccurrences Params"),
            size: std::mem::size_of::<OccurrenceParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let group_capacity = 1;
        let capacity = MIN_OCCURRENCES;
        Self {
            count_pipeline: pipeline("count_occurrences"),
            scan_pipeline: pipeline("scan_occurrences"),
            write_pipeline: pipeline("write_occurrences"),
            layout,
            bind_group: helio_core::CachedBindGroup::default(),
            params,
            compaction: RangeCompaction::new(device, "PortalOccurrences"),
            compaction_group: helio_core::CachedBindGroup::default(),
            group_counts: storage_buffer(device, "PortalOccurrences Group Counts", 4, false),
            group_cursors: storage_buffer(device, "PortalOccurrences Group Cursors", 4, false),
            indirect: storage_buffer(device, "PortalOccurrences Indirect", 20, true),
            compacted: storage_buffer(device, "PortalOccurrences Compacted", 20, true),
            draw_counts: storage_buffer(device, "PortalOccurrences Range Counts", 16, true),
            instances: storage_buffer(device, "PortalOccurrences Instances", capacity as u64 * 4, false),
            chains: storage_buffer(device, "PortalOccurrences Chains", capacity as u64 * 4, false),
            segment_indirect: storage_buffer(device, "PortalOccurrences Segment Indirect", 20, true),
            segment_counts: storage_buffer(device, "PortalOccurrences Segment Counts", 4, true),
            totals: storage_buffer(device, "PortalOccurrences Totals", 16, false),
            capacity,
            group_capacity,
            readback: None,
            readbacks: helio_core::gpu_cache::ReadbackPool::new("PortalOccurrences Readback"),
            chain_count: 0,
            draw_capacity: 0,
            live: false,
        }
    }

    /// Sizes the per-group buffers for `groups` and the segment buffers as
    /// the occlusion cull's.
    fn fit(&mut self, device: &wgpu::Device, groups: u32, culled: &CulledBatchFrameData<'_>, batch: &ObjectBatchFrameData<'_>) {
        if groups > self.group_capacity {
            self.group_capacity = groups.next_power_of_two();
            let n = self.group_capacity as u64;
            self.group_counts = storage_buffer(device, "PortalOccurrences Group Counts", n * 4, false);
            self.group_cursors = storage_buffer(device, "PortalOccurrences Group Cursors", n * 4, false);
            self.indirect = storage_buffer(device, "PortalOccurrences Indirect", n * 20, true);
            self.compacted = storage_buffer(device, "PortalOccurrences Compacted", n * 20, true);
        }
        let fit = |buffer: &mut wgpu::Buffer, size: u64, label: &str| {
            if buffer.size() < size {
                *buffer = storage_buffer(device, label, size, true);
            }
        };
        fit(&mut self.draw_counts, batch.draw_counts_gpu.size(), "PortalOccurrences Range Counts");
        fit(&mut self.segment_indirect, culled.segments.indirect.size(), "PortalOccurrences Segment Indirect");
        let counts = culled
            .segments
            .counts
            .map_or((culled.segments.segments.len().max(1) * 4) as u64, |b| b.size());
        fit(&mut self.segment_counts, counts, "PortalOccurrences Segment Counts");
    }

    /// Grows the occurrence buffers to hold `total`.
    fn grow(&mut self, device: &wgpu::Device, total: u32) {
        if total <= self.capacity {
            return;
        }
        self.capacity = total.next_power_of_two().max(MIN_OCCURRENCES);
        let bytes = self.capacity as u64 * 4;
        self.instances = storage_buffer(device, "PortalOccurrences Instances", bytes, false);
        self.chains = storage_buffer(device, "PortalOccurrences Chains", bytes, false);
    }
}

impl RenderPass for PortalOccurrencePass {
    fn name(&self) -> &'static str {
        "PortalOccurrences"
    }

    /// A member while the frame publishes portal chains.
    fn is_member(&mut self, scene: &mut helio_core::SceneContent<'_>) -> bool {
        if let Some(buffers) = scene.scene_buffers() {
            self.live = helio_portal_api::portal_chain_set(buffers).is_some();
        }
        self.live
    }

    fn reads(&self) -> &'static [&'static str] {
        &[
            "object_batch",
            "culled_batch",
            helio_portal_api::PORTAL_CHAINS,
            helio_portal_api::PORTAL_STAGES,
        ]
    }

    fn declare_resources(&self, builder: &mut helio_core::graph::ResourceBuilder) {
        builder.read("object_batch");
        builder.read("culled_batch");
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a helio_core::ResourceRegistry<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        // Last frame's total: map it once its copy has been submitted, read
        // it once mapped, and grow to it.
        if let Some((buffer, ready, mapping)) = self.readback.as_mut() {
            if !*mapping {
                buffer.map_read(Arc::clone(ready));
                *mapping = true;
            } else if ready.load(Ordering::Acquire) {
                let total = {
                    let view = buffer.slice(..4).get_mapped_range();
                    view.map(|bytes| u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                        .unwrap_or(0)
                };
                buffer.unmap();
                self.readback = None;
                self.grow(ctx.device, total);
            }
        }
        self.chain_count = ctx
            .registry
            .get(helio_portal_api::PORTAL_CHAIN_COUNT)
            .unwrap_or(0);
        let batch = ctx
            .registry
            .get::<ObjectBatchFrameData<'_>>(helio_core::ResourceKey::new("object_batch"));
        self.draw_capacity = batch.map_or(0, |batch| batch.group_capacity);
        let params = OccurrenceParams {
            frustum_planes: helio_core::depth::frustum_planes(ctx.camera_data.view_proj),
            draw_capacity: self.draw_capacity,
            chain_count: self.chain_count,
            capacity: self.capacity,
            groups_per_row: GROUPS_PER_ROW,
        };
        ctx.write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
        if let Some(batch) = batch.as_ref() {
            let segment_count = ctx
                .registry
                .get::<CulledBatchFrameData<'_>>(helio_core::ResourceKey::new("culled_batch"))
                .map_or(0, |culled| culled.segments.segments.len() as u32);
            ctx.write_buffer(
                &self.compaction.params,
                0,
                bytemuck::bytes_of(&RangeCompactParams {
                    slots: batch.range_slot_capacity,
                    bucket: 0,
                    segment_count,
                    _pad: 0,
                }),
            );
        }
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        if self.chain_count == 0 || self.draw_capacity == 0 {
            return Ok(());
        }
        let Some(batch) = ctx
            .registry
            .get::<ObjectBatchFrameData<'_>>(helio_core::ResourceKey::new("object_batch"))
        else {
            return Ok(());
        };
        let Some(culled) = ctx
            .registry
            .get::<CulledBatchFrameData<'_>>(helio_core::ResourceKey::new("culled_batch"))
        else {
            return Ok(());
        };
        let Some(spaces) = ctx
            .registry
            .get::<helio_core::CoordinateSpacesFrameData<'_>>(
                helio_core::resource_keys::coordinate_spaces(),
            )
        else {
            return Ok(());
        };
        let (Some(chains), Some(stages)) = (
            ctx.registry.buffer_binding(helio_portal_api::PORTAL_CHAINS),
            ctx.registry.buffer_binding(helio_portal_api::PORTAL_STAGES),
        ) else {
            return Ok(());
        };
        self.fit(&ctx.device, self.draw_capacity, &culled, &batch);
        let group = self.bind_group.get_or_create(
            &ctx.device,
            &wgpu::BindGroupDescriptor {
                label: Some("PortalOccurrences BG"),
                layout: &self.layout,
                entries: &[
                    buffer_entry(0, &self.params),
                    buffer_entry(1, batch.instances),
                    buffer_entry(2, batch.draw_calls),
                    buffer_entry(3, spaces.coordinate_spaces),
                    buffer_entry(4, chains),
                    buffer_entry(5, stages),
                    buffer_entry(6, batch.draw_counts_gpu),
                    buffer_entry(7, &self.group_counts),
                    buffer_entry(8, &self.group_cursors),
                    buffer_entry(9, &self.instances),
                    buffer_entry(10, &self.chains),
                    buffer_entry(11, &self.indirect),
                    buffer_entry(12, &self.totals),
                ],
            },
        ).clone();
        let rows = self.draw_capacity.div_ceil(GROUPS_PER_ROW);
        let across = self.draw_capacity.min(GROUPS_PER_ROW);
        {
            let mut cmds = ctx.graphics_cmds();
            cmds.clear_buffer(&self.group_counts, 0, None);
            cmds.clear_buffer(&self.group_cursors, 0, None);
            cmds.clear_buffer(&self.segment_counts, 0, None);
            cmds.clear_buffer(&self.segment_indirect, 0, None);
            {
                let mut pass = cmds.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("PortalOccurrences"),
                    timestamp_writes: None,
                });
                pass.set_bind_group(0, &group, &[]);
                pass.set_pipeline(&self.count_pipeline);
                pass.dispatch_workgroups(across, self.chain_count, rows);
                pass.set_pipeline(&self.scan_pipeline);
                pass.dispatch_workgroups(1, 1, 1);
                pass.set_pipeline(&self.write_pipeline);
                pass.dispatch_workgroups(across, self.chain_count, rows);
            }
            // The total, read back next frame to grow the buffers.
            if self.readback.is_none() {
                let buffer = self.readbacks.get(&ctx.device, 4);
                cmds.copy_buffer_to_buffer(&self.totals, 0, &buffer, 0, 4);
                self.readback = Some((buffer, Arc::new(AtomicBool::new(false)), false));
            }
        }
        // The groups' records into the occlusion cull's segments (the
        // opaque bucket: the G-buffer draws it).
        if batch.range_slot_capacity != 0 && !culled.segments.is_empty() {
            let compaction_group = self
                .compaction
                .bind_group(
                    &ctx.device,
                    &mut self.compaction_group,
                    [
                        &self.indirect,
                        &self.compacted,
                        batch.range_counts_gpu,
                        batch.opaque_ranges_gpu,
                        batch.transparent_ranges_gpu,
                        batch.forward_ranges_gpu,
                        &self.draw_counts,
                        culled.segments.table,
                        &self.segment_indirect,
                        &self.segment_counts,
                    ],
                )
                .clone();
            let mut cmds = ctx.graphics_cmds();
            let mut pass = cmds.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("PortalOccurrences RangeCompaction"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.compaction.pipeline);
            pass.set_bind_group(0, &compaction_group, &[0]);
            pass.dispatch_workgroups_indirect(batch.range_counts_gpu, 16);
        }
        Ok(())
    }

    fn publish<'a>(&self, frame: &mut helio_core::ResourceRegistry<'a>) {
        if self.chain_count == 0 {
            return;
        }
        let Some(culled) =
            frame.get::<CulledBatchFrameData<'a>>(helio_core::ResourceKey::new("culled_batch"))
        else {
            return;
        };
        // SAFETY: the buffers are this pass's, owned by the graph past the
        // frame's registry, as the occlusion cull's publication.
        let own = |buffer: &wgpu::Buffer| -> &'a wgpu::Buffer { unsafe { std::mem::transmute(buffer) } };
        frame.write(
            helio_core::ResourceKey::new(PORTAL_OCCURRENCES),
            PortalOccurrenceFrameData {
                indirect: own(&self.indirect),
                instances: own(&self.instances),
                chains: own(&self.chains),
                segments: DrawSegments {
                    indirect: own(&self.segment_indirect),
                    counts: culled.segments.counts.map(|_| own(&self.segment_counts)),
                    segments: culled.segments.segments,
                    table: culled.segments.table,
                },
            },
            "PortalOccurrences",
        );
    }
}
