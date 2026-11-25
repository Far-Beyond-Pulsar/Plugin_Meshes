//! Virtual geometry through portal chains (`helio_portal_api`): the instances
//! of the frame's second VG run ([`VgRun::SECONDARY`]).
//!
//! While the frame has live chains and the scene holds VG instances,
//! [`VgPortalInstancesPass`] walks the frame's instance BVH once per chain
//! against the chain's volume (`helio_portal_api::PORTAL_CHAIN_VOLUMES`: the
//! frustum mapped into the content, clipped to the chain's deepest opening),
//! and appends each instance it may show as a virtual instance: its object
//! rows copied with their transforms placed where the chain draws them (this
//! frame's and last frame's map, so motion vectors are the chain's), and the
//! chain's remap. The second run culls, rasterizes and shades those against
//! the depth the frame holds after the mask and the classic occurrences,
//! clipping every pixel to its chain, in the one cluster store (its page
//! requests join the frame's feedback).
//!
//! The work follows what the openings show: a chain visits only the BVH
//! nodes its volume overlaps. The buffers grow from the totals the GPU found
//! (read back a frame or two later); a frame that finds more than they hold
//! draws the ones that fit.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use helio_core::graph::ResourceBuilder;
use helio_core::{PassContext, PrepareContext, RenderPass, ResourceKey, Result as HelioResult};
use helio_pass_vg_bvh::keys as bvh_keys;
use helio_vg_format::{GpuVgInstance, VgInstanceSource, VgRun, VgRunGate};

/// `vg_portal_instances.wgsl`'s `Params`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    chain_count: u32,
    instance_capacity: u32,
    object_capacity: u32,
    _pad: u32,
}

/// The fewest virtual instances, and object rows, the buffers hold.
const MIN_INSTANCES: u32 = 256;
const MIN_OBJECTS: u32 = 1024;
/// `helio_mesh_api::StaticObjectRow`.
const OBJECT_ROW_BYTES: u64 = std::mem::size_of::<helio_mesh_api::StaticObjectRow>() as u64;
const INSTANCE_BYTES: u64 = std::mem::size_of::<GpuVgInstance>() as u64;

/// The registry keys the run reads its instances from.
const fn secondary_keys() -> (&'static str, &'static str, &'static str) {
    match VgRun::SECONDARY.instances {
        VgInstanceSource::Registry {
            instances,
            objects,
            remaps,
        } => (instances, objects, remaps),
        VgInstanceSource::Scene => panic!("the secondary run reads published instances"),
    }
}

fn storage_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(16),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
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

fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

/// Publishes [`VgRun::SECONDARY`]'s instances: the frame's VG instances each
/// live portal chain may show, placed where it draws them.
pub struct VgPortalInstancesPass {
    pipeline: helio_core::shader::DeferredCompute,
    layout: wgpu::BindGroupLayout,
    bind_group: helio_core::CachedBindGroup,
    params: wgpu::Buffer,
    instances: wgpu::Buffer,
    objects: wgpu::Buffer,
    remaps: wgpu::Buffer,
    /// [0] instances, [1] object rows found this frame.
    counts: wgpu::Buffer,
    instance_capacity: u32,
    object_capacity: u32,
    readback: Option<(helio_core::gpu_cache::PooledBuffer, Arc<AtomicBool>, bool)>,
    readbacks: helio_core::gpu_cache::ReadbackPool,
    chain_count: u32,
    /// Whether the frame has live chains and the scene VG instances (a graph
    /// member only then; the run's passes follow through [`Self::gate`]).
    live: bool,
    gate: VgRunGate,
}

impl VgPortalInstancesPass {
    /// The snippets [`Self::source`] uses.
    pub const SNIPPETS: [helio_core::shader::ShaderSnippet; 3] = [
        helio_portal_api::PORTAL_CHAIN_SNIPPET,
        helio_mesh_api::OBJECT_ROW_SNIPPET,
        helio_mesh_api::DRAW_ROWS_SNIPPET,
    ];

    /// The kernel's WGSL before its snippets resolve (shader validation
    /// without a device).
    pub fn source() -> String {
        format!(
            "{}\n{}\n{}",
            helio_vg_format::WGSL,
            helio_pass_vg_bvh::NODES_WGSL,
            include_str!("../shaders/vg_portal_instances.wgsl"),
        )
    }

    /// The pass, opening `gate` (the second run's passes) while it is a member.
    pub fn new(device: &wgpu::Device, gate: VgRunGate) -> Self {
        let module = helio_core::shader::module_with(
            device,
            "VgPortalInstances",
            Self::source().as_str(),
            &Self::SNIPPETS,
        );
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("VgPortalInstances"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, true),
                storage_entry(5, true),
                storage_entry(6, true),
                storage_entry(7, true),
                storage_entry(8, true),
                storage_entry(9, true),
                storage_entry(10, false),
                storage_entry(11, false),
                storage_entry(12, false),
                storage_entry(13, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("VgPortalInstances"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = helio_core::shader::create_compute_pipeline_deferred(
            device,
            wgpu::ComputePipelineDescriptor {
                label: Some("VgPortalInstances"),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("cs_portal_instances"),
                compilation_options: Default::default(),
                cache: None,
            },
        );
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VgPortalInstances Params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut pass = Self {
            pipeline,
            layout,
            bind_group: helio_core::CachedBindGroup::default(),
            params,
            instances: storage_buffer(device, "VgPortalInstances Instances", 16),
            objects: storage_buffer(device, "VgPortalInstances Objects", 16),
            remaps: storage_buffer(device, "VgPortalInstances Remaps", 16),
            counts: storage_buffer(device, "VgPortalInstances Counts", 8),
            instance_capacity: 0,
            object_capacity: 0,
            readback: None,
            readbacks: helio_core::gpu_cache::ReadbackPool::new("VgPortalInstances Readback"),
            chain_count: 0,
            live: false,
            gate,
        };
        pass.grow(device, MIN_INSTANCES, MIN_OBJECTS);
        pass
    }

    /// Grows the buffers to hold `instances` virtual instances and `objects`
    /// object rows.
    fn grow(&mut self, device: &wgpu::Device, instances: u32, objects: u32) {
        if instances > self.instance_capacity {
            self.instance_capacity = instances.next_power_of_two().max(MIN_INSTANCES);
            let n = self.instance_capacity as u64;
            self.instances =
                storage_buffer(device, "VgPortalInstances Instances", n * INSTANCE_BYTES);
            self.remaps = storage_buffer(device, "VgPortalInstances Remaps", n * 4);
        }
        if objects > self.object_capacity {
            self.object_capacity = objects.next_power_of_two().max(MIN_OBJECTS);
            self.objects = storage_buffer(
                device,
                "VgPortalInstances Objects",
                self.object_capacity as u64 * OBJECT_ROW_BYTES,
            );
        }
    }
}

impl RenderPass for VgPortalInstancesPass {
    fn name(&self) -> &'static str {
        "VgPortalInstances"
    }

    /// A member while the frame has live chains and the scene VG instances;
    /// the second run's passes are members exactly then ([`VgRunGate`]: the
    /// graph asks this pass before them, in build order).
    fn is_member(&mut self, scene: &mut helio_core::SceneContent<'_>) -> bool {
        if let Some(buffers) = scene.scene_buffers() {
            let chains = helio_portal_api::portal_chain_set(buffers).is_some();
            self.live = chains && scene.holds_rows(bvh_keys::INSTANCES);
        }
        self.gate.set(self.live);
        self.live
    }

    fn on_leave(&mut self, _device: &wgpu::Device) {
        self.gate.set(false);
    }

    fn reads(&self) -> &'static [&'static str] {
        &[
            helio_portal_api::PORTAL_CHAINS,
            helio_portal_api::PORTAL_CHAIN_VOLUMES,
        ]
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.read(bvh_keys::NODES);
        builder.read(bvh_keys::HEADER);
        builder.read(bvh_keys::ITEMS);
        let (instances, objects, remaps) = secondary_keys();
        builder.write_buffer(instances);
        builder.write_buffer(objects);
        builder.write_buffer(remaps);
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
        // Last frame's totals: map them once their copy has been submitted,
        // read them once mapped, and grow to them.
        if let Some((buffer, ready, mapping)) = self.readback.as_mut() {
            if !*mapping {
                buffer.map_read(Arc::clone(ready));
                *mapping = true;
            } else if ready.load(Ordering::Acquire) {
                let totals = {
                    let view = buffer.slice(..8).get_mapped_range();
                    view.map(|bytes| {
                        let word = |i: usize| {
                            u32::from_le_bytes([
                                bytes[i * 4],
                                bytes[i * 4 + 1],
                                bytes[i * 4 + 2],
                                bytes[i * 4 + 3],
                            ])
                        };
                        [word(0), word(1)]
                    })
                    .unwrap_or([0, 0])
                };
                buffer.unmap();
                self.readback = None;
                self.grow(ctx.device, totals[0], totals[1]);
            }
        }
        self.chain_count = ctx
            .registry
            .get(helio_portal_api::PORTAL_CHAIN_COUNT)
            .unwrap_or(0);
        ctx.write_buffer(
            &self.params,
            0,
            bytemuck::bytes_of(&Params {
                chain_count: self.chain_count,
                instance_capacity: self.instance_capacity,
                object_capacity: self.object_capacity,
                _pad: 0,
            }),
        );
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        // Every slot starts disabled, so a frame that finds fewer instances
        // (or cannot look) leaves none of last frame's.
        {
            let mut cmds = ctx.graphics_cmds();
            cmds.clear_buffer(&self.instances, 0, None);
            cmds.clear_buffer(&self.counts, 0, None);
        }
        if self.chain_count == 0 {
            return Ok(());
        }
        let reg = ctx.registry;
        let (Some(nodes), Some(header), Some(items)) = (
            reg.buffer(ResourceKey::new(bvh_keys::NODES)),
            reg.buffer(ResourceKey::new(bvh_keys::HEADER)),
            reg.buffer(ResourceKey::new(bvh_keys::ITEMS)),
        ) else {
            return Ok(());
        };
        let (Some(instances), Some(objects)) = (
            ctx.scene_buffers
                .get(helio_core::BufferKey::of(bvh_keys::INSTANCES)),
            ctx.scene_buffers
                .get(helio_core::BufferKey::of(bvh_keys::STATIC_OBJECTS)),
        ) else {
            return Ok(());
        };
        let Some(spaces) = reg.get::<helio_core::CoordinateSpacesFrameData<'_>>(
            helio_core::resource_keys::coordinate_spaces(),
        ) else {
            return Ok(());
        };
        let (Some(chains), Some(volumes)) = (
            reg.buffer_binding(helio_portal_api::PORTAL_CHAINS),
            reg.buffer_binding(helio_portal_api::PORTAL_CHAIN_VOLUMES),
        ) else {
            return Ok(());
        };
        let group = self
            .bind_group
            .get_or_create(
                ctx.device,
                &wgpu::BindGroupDescriptor {
                    label: Some("VgPortalInstances"),
                    layout: &self.layout,
                    entries: &[
                        entry(0, &self.params),
                        entry(1, nodes),
                        entry(2, header),
                        entry(3, items),
                        entry(4, &instances.buffer),
                        entry(5, &objects.buffer),
                        entry(6, spaces.coordinate_spaces),
                        entry(7, spaces.coordinate_spaces_prev),
                        entry(8, chains),
                        entry(9, volumes),
                        entry(10, &self.instances),
                        entry(11, &self.objects),
                        entry(12, &self.remaps),
                        entry(13, &self.counts),
                    ],
                },
            )
            .clone();
        let mut cmds = ctx.graphics_cmds();
        {
            let mut pass = cmds.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("VgPortalInstances"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(self.chain_count, 1, 1);
        }
        // The totals, read back to grow the buffers.
        if self.readback.is_none() {
            let buffer = self.readbacks.get(ctx.device, 8);
            cmds.copy_buffer_to_buffer(&self.counts, 0, &buffer, 0, 8);
            self.readback = Some((buffer, Arc::new(AtomicBool::new(false)), false));
        }
        Ok(())
    }

    fn publish<'a>(&self, frame: &mut helio_core::ResourceRegistry<'a>) {
        // SAFETY: the buffers are this pass's, owned by the graph past the
        // frame's registry, as every pass's publication.
        let own =
            |buffer: &wgpu::Buffer| -> &'a wgpu::Buffer { unsafe { std::mem::transmute(buffer) } };
        let (instances, objects, remaps) = secondary_keys();
        for (key, buffer) in [
            (instances, &self.instances),
            (objects, &self.objects),
            (remaps, &self.remaps),
        ] {
            frame.write_buffer(ResourceKey::new(key), own(buffer), "VgPortalInstances");
        }
    }
}
