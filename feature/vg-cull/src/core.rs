//! GPU resources and command recording of the cull. Independent of the render
//! graph so tests and offline tools can drive it with a plain encoder; the
//! passes in `pass.rs` are thin wrappers.

use std::num::NonZeroU64;
use std::sync::{Arc, Mutex};

use bytemuck::{Pod, Zeroable};
use helio_core::CommandRecorder;
use helio_vg_format::VgMode;

use crate::{
    mode_bits, state, stats, FLAG_CURRENT_HIZ, FLAG_DENSE_MASK, FLAG_MASKED, FLAG_NO_ROOT_SHORTCUT,
    FLAG_STATS, FLAG_SW_MASKED, FLAG_TWO_PHASE, MAX_BIN_CAPACITY,
};

/// Stride of the per-phase uniform blocks (dynamic-offset alignment).
const ROUND_STRIDE: u64 = 256;
/// Job queues: the early phase's, and the deferred work the late phase starts
/// from (see the crate docs, "Traversal").
const QUEUES: u64 = 2;
/// Entries a bin or queue starts with.
const INITIAL_CAPACITY: u32 = 1 << 12;
/// The most a queue grows ahead of a frame. The bound on a frame's jobs is far
/// above what its live frontier reaches, so a queue grows past this only once
/// the GPU reports it full.
pub const QUEUE_DEMAND_CEILING: u32 = 1 << 20;
/// The most a queue ever grows.
const MAX_QUEUE_CAPACITY: u32 = 1 << 24;

/// The most one frame of a run can ask of the cull, as far as the CPU knows it
/// before recording: what the bins and queues grow to ahead of the frame, so
/// they never overflow below their ceilings. Zero means unknown: the buffers
/// then grow only after the GPU reports them filling.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VgCullDemand {
    /// Visible clusters, all bins together; any one bin may take all of them.
    pub clusters: u64,
    /// Traversal jobs.
    pub jobs: u64,
}

impl VgCullDemand {
    /// The bound for `instances` rows of assets that have at most `largest`.
    /// A frame emits each cluster of an instance at most once (over both
    /// phases), and queues each of its nodes, groups and clusters at most once,
    /// besides the instance itself and its share of the instance BVH.
    pub fn for_instances(instances: u32, largest: helio_vg_format::GpuAssetHeader) -> Self {
        let rows = u64::from(instances);
        let clusters = u64::from(largest.cluster_count);
        let jobs = clusters + u64::from(largest.node_count) + u64::from(largest.group_count);
        if clusters == 0 {
            return Self::default();
        }
        Self {
            clusters: rows.saturating_mul(clusters),
            jobs: rows.saturating_mul(jobs.saturating_add(3)),
        }
    }
}

/// `capacity` grown to hold `demand` entries, in powers of two, no further than
/// `ceiling`. Never smaller than it was.
fn grown_to(capacity: u32, demand: u64, ceiling: u32) -> u32 {
    let wanted = (demand.min(u64::from(ceiling)) as u32)
        .next_power_of_two()
        .min(ceiling);
    capacity.max(wanted)
}

/// Tunables. Defaults match the plan (tau 1 px, SW at <= 2 px edges).
#[derive(Clone, Debug)]
pub struct VgCullConfig {
    /// Screen-space error threshold in pixels.
    pub tau_px: f32,
    /// Projected average triangle edge at or below which a cluster is SW.
    pub sw_edge_px: f32,
    /// Overdraw valve: triangles per covered pixel one 16x16 tile may hold before
    /// the groups covering it are coarsened (0 turns the valve off). Per tile, never
    /// screen-wide; see the crate docs.
    pub tile_density_limit: f32,
    /// Prefetch band: child pages are requested while the projected error exceeds
    /// `tau / prefetch_factor` (1 = request only when the cut needs the page).
    pub prefetch_factor: f32,
    pub mode: VgMode,
    /// Last-frame-visible-first occlusion with a Hi-Z hook between phases.
    pub two_phase_occlusion: bool,
    /// Persistent workgroups of a phase's traversal dispatch. Any value is
    /// correct (workgroups that do not fit on the GPU at once start late and
    /// find the work done); more than fit only adds idle polling.
    pub max_groups: u32,
    /// Entries per bin before anything is drawn (8 bytes each). The bins grow
    /// ahead of each frame to what it can emit ([`VgCullDemand`]), up to
    /// [`Self::max_bin_capacity`], so a run with little or nothing to draw
    /// stays this small. A bin that overflows drops clusters for that frame,
    /// which is why growth goes by that bound and not by what the GPU reports
    /// afterwards: a read-back arrives frames late and reads as flicker.
    pub initial_bin_capacity: u32,
    /// Entries per job queue ([`QUEUES`] queues, 8 bytes each) before anything
    /// is drawn. Jobs past a queue are lost work, so the queues grow ahead of
    /// the frame like the bins, up to [`QUEUE_DEMAND_CEILING`]; past that, only
    /// a queue the GPU reports full grows further.
    pub initial_queue_capacity: u32,
    /// Bits of the visible-last-frame hash set (power of two).
    pub history_bits: u32,
    /// Maintain the stats counters (small atomic cost).
    pub stats: bool,
    /// Upper bound on entries per bin (a ceiling, never an allocation): the
    /// visible-cluster ceiling of the settings divided over the bins.
    pub max_bin_capacity: u32,
    /// Alpha-tested clusters of the software class may be routed to the software
    /// masked bin (the raster must have the 64-bit atomic kernel). Default false:
    /// they use the hardware masked bin.
    pub sw_masked: bool,
    /// Draw the root clusters of an instance whose whole-asset root error projects
    /// under tau directly, without walking its hierarchy (exactly the same cut).
    /// `HELIO_VG_NO_ROOT_SHORTCUT=1` turns it off for A/B measurements.
    pub root_shortcut: bool,
    /// The early Hi-Z is this frame's depth so far, not last frame's (a run drawn
    /// after other drawers, `helio_vg_format::VgRun::builds_hiz_early`): one
    /// exact occlusion phase, no deferral, no late phase ([`crate::FLAG_CURRENT_HIZ`]).
    pub current_hiz: bool,
    /// The run's instances are the result of a spatial query (few, every one
    /// a candidate, `helio_vg_format::VgRun::tree` `None`): the traversal
    /// starts from every enabled instance instead of an instance BVH's root.
    pub seeds_instances: bool,
}

impl Default for VgCullConfig {
    fn default() -> Self {
        Self {
            tau_px: 1.0,
            sw_edge_px: 2.0,
            tile_density_limit: helio_vg_format::DEFAULT_TILE_DENSITY_LIMIT,
            prefetch_factor: 2.0,
            mode: VgMode::Auto,
            two_phase_occlusion: true,
            max_groups: 256,
            initial_bin_capacity: INITIAL_CAPACITY,
            initial_queue_capacity: INITIAL_CAPACITY,
            history_bits: 1 << 22,
            stats: true,
            max_bin_capacity: MAX_BIN_CAPACITY,
            sw_masked: false,
            root_shortcut: std::env::var_os("HELIO_VG_NO_ROOT_SHORTCUT").is_none(),
            current_hiz: false,
            seeds_instances: false,
        }
    }
}

/// Everything the cull reads that other crates own. Cheap to copy; buffers are
/// borrowed for the frame. Binding contract (all `read-only storage` except
/// the stream's feedback):
///
/// | field | element |
/// |---|---|
/// | `instances` (`"vg_instances"`) | [`helio_vg_format::GpuVgInstance`] |
/// | `objects` (`"static_objects"`) | 240-byte (60-word) scene-join row; `transform` is read via `GpuVgInstance::object_row` |
/// | `stream` | the eight `helio-vg-stream` buffers (pool, residency, nodes, group refs, headers, bases, quant, feedback); the cull appends page requests / touches straight into `feedback` |
/// | `tree` | `helio-pass-vg-bvh` outputs (`vg_gcbvh_*`); `None` for a run that seeds its instances ([`VgCullConfig::seeds_instances`]) |
/// | `dense_mask` | one `u32` per 16x16 tile, row-major, nonzero = dense last frame |
#[derive(Clone, Copy)]
pub struct VgCullInputs<'a> {
    pub instances: &'a wgpu::Buffer,
    pub objects: &'a wgpu::Buffer,
    pub stream: helio_vg_stream::VgStreamBuffers<'a>,
    pub tree: Option<VgCullTree<'a>>,
    pub dense_mask: Option<&'a wgpu::Buffer>,
    /// The scene's `materials` table (`GpuMaterial` rows, 24 words). `Some` turns on
    /// alpha-mask routing; `None` is the opaque-only path with no material loads.
    pub materials: Option<&'a wgpu::Buffer>,
    /// Rows of `instances` (the traversal ignores BVH leaves beyond it).
    pub instance_count: u32,
    /// Dense-mask grid `[tiles_x, tiles_y]`.
    pub dense_tiles: [u32; 2],
    /// The most this frame can ask of the bins and queues.
    pub demand: VgCullDemand,
    /// Per `instances` row, the surface remap its pixels carry
    /// (`helio_vg_format::VgInstanceSource::Registry::remaps`): a run drawn
    /// through another space. The raster clips to it and the shading tags
    /// it; `None` for a run drawn where its instances are.
    pub remaps: Option<&'a wgpu::Buffer>,
}

/// A run's instance BVH as the cull reads it (`helio-pass-vg-bvh`).
#[derive(Clone, Copy)]
pub struct VgCullTree<'a> {
    pub nodes: &'a wgpu::Buffer,
    pub header: &'a wgpu::Buffer,
    pub items: &'a wgpu::Buffer,
}

/// Per-frame values the graph knows.
#[derive(Clone, Copy, Debug)]
pub struct VgCullFrame {
    pub width: u32,
    pub height: u32,
}

/// Buffers sized for the current capacities. Replaced (new `Arc`) when one of
/// them grows; pass wrappers and the passes after them re-read it each frame
/// and keep no bind group across frames.
pub struct VgCullBuffers {
    pub visible: wgpu::Buffer,
    /// Every word the cull writes with atomics: the counters, ranges and args
    /// ([`crate::state`], words `0..state::WORDS`), then from
    /// [`state::HISTORY`] this frame's visibility/hysteresis marks
    /// (`history_words`) and the tile block at [`Self::tile_base`]
    /// ([`crate::TILE_WORDS`] words per tile, see the crate docs).
    pub state: wgpu::Buffer,
    pub queues: wgpu::Buffer,
    /// What the traversal only reads, copied from `state` at the start of the
    /// early phase: last frame's marks (`history_words`), then the tile block
    /// with this frame's valve scales (at word `history_words`). Bound
    /// read-only, so the per-job lookups are cached loads, not L2 atomics.
    pub history_ro: wgpu::Buffer,
    pub bin_capacity: u32,
    pub queue_capacity: u32,
    /// `u32` words of the visibility/hysteresis mark set.
    pub history_words: u32,
    /// The 16x16 tile grid of the screen the tile block is sized for.
    pub tiles: [u32; 2],
}

impl VgCullBuffers {
    /// Word offset of the tile block in `state`.
    pub fn tile_base(&self) -> u32 {
        state::HISTORY + self.history_words
    }

    fn tile_block_words(tiles: [u32; 2]) -> u64 {
        u64::from(crate::TILE_WORDS) * u64::from(tiles[0]) * u64::from(tiles[1])
    }

    fn new(
        device: &wgpu::Device,
        bin_capacity: u32,
        queue_capacity: u32,
        history_words: u32,
        tiles: [u32; 2],
    ) -> Self {
        let (state, history_ro) = Self::history_buffers(device, history_words, tiles);
        Self {
            visible: Self::visible_buffer(device, bin_capacity),
            state,
            queues: Self::queue_buffer(device, queue_capacity),
            history_ro,
            bin_capacity,
            queue_capacity,
            history_words,
            tiles,
        }
    }

    fn visible_buffer(device: &wgpu::Device, bin_capacity: u32) -> wgpu::Buffer {
        let bytes = u64::from(crate::BIN_COUNT) * u64::from(bin_capacity) * 8;
        Self::entry_buffer(device, "VgCull Visible", bytes)
    }

    fn queue_buffer(device: &wgpu::Device, queue_capacity: u32) -> wgpu::Buffer {
        let bytes = QUEUES * u64::from(queue_capacity) * 8;
        Self::entry_buffer(device, "VgCull Queues", bytes)
    }

    fn entry_buffer(device: &wgpu::Device, label: &str, bytes: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// These buffers with other bin and queue capacities: only the one that
    /// changed is replaced. Both hold one frame's entries, so nothing is
    /// copied, and the visibility history is kept.
    fn with_capacities(
        &self,
        device: &wgpu::Device,
        bin_capacity: u32,
        queue_capacity: u32,
    ) -> Self {
        Self {
            visible: if bin_capacity == self.bin_capacity {
                self.visible.clone()
            } else {
                Self::visible_buffer(device, bin_capacity)
            },
            state: self.state.clone(),
            queues: if queue_capacity == self.queue_capacity {
                self.queues.clone()
            } else {
                Self::queue_buffer(device, queue_capacity)
            },
            history_ro: self.history_ro.clone(),
            bin_capacity,
            queue_capacity,
            history_words: self.history_words,
            tiles: self.tiles,
        }
    }

    /// `state` and `history_ro`, sized for the mark set and the tile grid.
    fn history_buffers(
        device: &wgpu::Device,
        history_words: u32,
        tiles: [u32; 2],
    ) -> (wgpu::Buffer, wgpu::Buffer) {
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let tile_words = Self::tile_block_words(tiles);
        let state_words = u64::from(state::HISTORY) + u64::from(history_words) + tile_words;
        let ro_words = u64::from(history_words) + tile_words;
        let state = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VgCull State"),
            size: state_words * 4,
            usage: usage | wgpu::BufferUsages::INDIRECT,
            mapped_at_creation: false,
        });
        let history_ro = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VgCull History (read-only copy)"),
            size: (ro_words * 4).max(16),
            usage,
            mapped_at_creation: false,
        });
        (state, history_ro)
    }

    /// These buffers with the tile block resized for a new screen grid: only
    /// `state` and its read-only copy are replaced (a window resize must not
    /// reallocate the bins).
    fn with_tiles(&self, device: &wgpu::Device, tiles: [u32; 2]) -> Self {
        let (state, history_ro) = Self::history_buffers(device, self.history_words, tiles);
        Self {
            visible: self.visible.clone(),
            state,
            queues: self.queues.clone(),
            history_ro,
            bin_capacity: self.bin_capacity,
            queue_capacity: self.queue_capacity,
            history_words: self.history_words,
            tiles,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParams {
    screen_width: u32,
    screen_height: u32,
    hiz_mip_count: u32,
    mode: u32,
    tau_px: f32,
    sw_edge_px: f32,
    instance_count: u32,
    bin_capacity: u32,
    queue_capacity: u32,
    history_bits_mask: u32,
    hist_base: u32,
    ro_tile_base: u32,
    flags: u32,
    dense_tiles_x: u32,
    dense_tiles_y: u32,
    max_groups: u32,
    tile_limit: f32,
    prefetch_k: f32,
    tiles_x: u32,
    tiles_y: u32,
    tile_base: u32,
    _pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuPhase {
    phase: u32,
    hiz_valid: u32,
    _pad: [u32; 2],
}

struct Readback {
    staging: wgpu::Buffer,
    ready: Arc<Mutex<Option<Vec<u32>>>>,
    copied: bool,
    in_flight: bool,
    revision: u64,
}

/// Staging buffers of the tile-valve readback: enough for the frames a host keeps
/// in flight plus the one being mapped, so watching never waits for the GPU.
const TILE_VALVE_SLOTS: usize = 8;

enum ValveSlotState {
    Free,
    /// The copy is recorded in frame `frame`'s commands (maybe not submitted yet).
    Copied {
        frame: u64,
        tiles: [u32; 2],
    },
    /// Submitted and mapping; the callback fills `ready`.
    Mapping {
        frame: u64,
        tiles: [u32; 2],
        ready: Arc<Mutex<Option<bool>>>,
    },
}

struct ValveSlot {
    buffer: wgpu::Buffer,
    state: ValveSlotState,
}

/// The owned, asynchronous readback of the tile block ([`VgCullCore::watch_tile_valve`]).
#[derive(Default)]
struct TileValveReadback {
    slots: Vec<ValveSlot>,
    /// Frames still to copy.
    remaining: u32,
    samples: std::collections::VecDeque<crate::TileValveSample>,
}

impl TileValveReadback {
    /// Starts mapping every copied slot. Only once its frame is submitted: the
    /// core calls this at the next `begin_frame` (or from the blocking read).
    fn map_copied(&mut self) {
        for slot in &mut self.slots {
            if let ValveSlotState::Copied { frame, tiles } = slot.state {
                let ready = Arc::new(Mutex::new(None));
                let done = ready.clone();
                slot.buffer
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, move |r| {
                        *done.lock().unwrap_or_else(|p| p.into_inner()) = Some(r.is_ok());
                    });
                slot.state = ValveSlotState::Mapping {
                    frame,
                    tiles,
                    ready,
                };
            }
        }
    }

    /// Moves every mapped slot's block into `samples`, oldest frame first.
    fn collect(&mut self) {
        let mut got = Vec::new();
        for slot in &mut self.slots {
            let ValveSlotState::Mapping {
                frame,
                tiles,
                ref ready,
            } = slot.state
            else {
                continue;
            };
            let Some(ok) = *ready.lock().unwrap_or_else(|p| p.into_inner()) else {
                continue;
            };
            if ok {
                let words = (crate::TILE_WORDS * tiles[0] * tiles[1]) as usize;
                if let Ok(view) = slot.buffer.slice(..).get_mapped_range() {
                    let all = bytemuck::cast_slice::<u8, u32>(&view);
                    got.push(crate::TileValveSample::decode(frame, tiles, &all[..words]));
                }
                slot.buffer.unmap();
            }
            slot.state = ValveSlotState::Free;
        }
        got.sort_by_key(|s| s.frame);
        self.samples.extend(got);
    }

    fn busy(&self) -> bool {
        self.slots
            .iter()
            .any(|s| !matches!(s.state, ValveSlotState::Free))
    }
}

struct Pipelines {
    tiles: helio_core::shader::DeferredCompute,
    seed: helio_core::shader::DeferredCompute,
    seed_instances: helio_core::shader::DeferredCompute,
    phase_init: helio_core::shader::DeferredCompute,
    traverse: helio_core::shader::DeferredCompute,
    finalize: helio_core::shader::DeferredCompute,
}

type BindCache = Option<(
    Vec<wgpu::Buffer>,
    Option<wgpu::TextureView>,
    wgpu::BindGroup,
)>;

pub struct VgCullCore {
    pub config: VgCullConfig,
    pipelines: Pipelines,
    bgl0: wgpu::BindGroupLayout,
    bg1: wgpu::BindGroup,
    params_buf: wgpu::Buffer,
    round_buf: wgpu::Buffer,
    sampler: wgpu::Sampler,
    dummy_view: wgpu::TextureView,
    dummy_buf: wgpu::Buffer,
    /// Bound as the tree of a run without one (the shader declares the
    /// header's 128 bytes).
    dummy_tree: wgpu::Buffer,
    buffers: Arc<VgCullBuffers>,
    bind_cache: [BindCache; 2],
    frame: u64,
    readback: Readback,
    valve_readback: TileValveReadback,
    last_stats: Option<Vec<u32>>,
    pending_capacity_stats: Option<Vec<u32>>,
    size: (u32, u32),
    prepared: bool,
    /// An early phase deferred work that no late phase has taken yet.
    late_owed: bool,
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

pub fn mip_levels(w: u32, h: u32) -> u32 {
    (u32::BITS - w.max(h).max(1).leading_zeros()).max(1)
}

impl VgCullCore {
    pub fn new(device: &wgpu::Device, config: VgCullConfig) -> Self {
        let source = crate::shader_source();
        let module = helio_core::shader::module(device, "VgCull", source.as_str());

        let mut entries0 = vec![
            storage_entry(0, true),
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ];
        // 2..=3 instances/objects, 4..=10 stream (read-only), 11 stream feedback.
        for b in 2..=10 {
            entries0.push(storage_entry(b, true));
        }
        entries0.push(storage_entry(11, false));
        // 12..=15 gcbvh nodes/header/items, dense mask.
        for b in 12..=15 {
            entries0.push(storage_entry(b, true));
        }
        entries0.push(wgpu::BindGroupLayoutEntry {
            binding: 16,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        });
        entries0.push(wgpu::BindGroupLayoutEntry {
            binding: 17,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
            count: None,
        });
        // 18..=20 state, queues, visible; 21 the read-only history copy; 22
        // materials.
        for b in 18..=20 {
            entries0.push(storage_entry(b, false));
        }
        entries0.push(storage_entry(21, true));
        entries0.push(storage_entry(22, true));
        let bgl0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("VgCull BGL0"),
            entries: &entries0,
        });
        let bgl1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("VgCull BGL1 (phase)"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: NonZeroU64::new(std::mem::size_of::<GpuPhase>() as u64),
                },
                count: None,
            }],
        });

        let pl2 = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("VgCull PL"),
            bind_group_layouts: &[Some(&bgl0), Some(&bgl1)],
            immediate_size: 0,
        });
        let pipeline = |label: &str, layout: &wgpu::PipelineLayout, entry: &str| {
            helio_core::shader::create_compute_pipeline_deferred(
                device,
                wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(layout),
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                },
            )
        };
        let pipelines = Pipelines {
            tiles: pipeline("VgCull Tiles", &pl2, "cs_tiles"),
            seed: pipeline("VgCull Seed", &pl2, "cs_seed"),
            seed_instances: pipeline("VgCull Seed Instances", &pl2, "cs_seed_instances"),
            phase_init: pipeline("VgCull PhaseInit", &pl2, "cs_phase_init"),
            traverse: pipeline("VgCull Traverse", &pl2, "cs_traverse"),
            finalize: pipeline("VgCull Finalize", &pl2, "cs_finalize"),
        };

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VgCull Params"),
            size: std::mem::size_of::<GpuParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let round_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VgCull Phases"),
            size: 2 * ROUND_STRIDE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("VgCull BG1"),
            layout: &bgl1,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &round_buf,
                    offset: 0,
                    size: NonZeroU64::new(std::mem::size_of::<GpuPhase>() as u64),
                }),
            }],
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("VgCull HiZ Sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let dummy_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("VgCull Dummy HiZ"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let dummy_view = dummy_tex.create_view(&Default::default());
        let dummy_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VgCull Dummy Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let history_words = (config.history_bits.next_power_of_two().max(1024)) / 32;
        let buffers = Arc::new(VgCullBuffers::new(
            device,
            config.initial_bin_capacity.min(MAX_BIN_CAPACITY).max(16),
            config.initial_queue_capacity.max(256),
            history_words,
            [1, 1],
        ));
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VgCull Stats Readback"),
            size: state::WORDS as u64 * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            config,
            pipelines,
            bgl0,
            bg1,
            params_buf,
            round_buf,
            sampler,
            dummy_view,
            dummy_buf,
            dummy_tree: {
                let tree = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("VgCull Empty Tree"),
                    size: 128,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: true,
                });
                tree.unmap();
                tree
            },
            buffers,
            bind_cache: [None, None],
            frame: 0,
            readback: Readback {
                staging,
                ready: Arc::new(Mutex::new(None)),
                copied: false,
                in_flight: false,
                revision: 0,
            },
            valve_readback: TileValveReadback::default(),
            last_stats: None,
            pending_capacity_stats: None,
            size: (1, 1),
            prepared: false,
            late_owed: false,
        }
    }

    pub fn buffers(&self) -> &Arc<VgCullBuffers> {
        &self.buffers
    }

    /// Frames begun so far (phase-0 hi-z is trusted from the second).
    pub fn frame_index(&self) -> u64 {
        self.frame
    }

    /// The state words (bins, stats, ...) of the most recent completed
    /// read-back, one to a few frames old.
    pub fn last_state_words(&self) -> Option<&[u32]> {
        self.last_stats.as_deref()
    }

    /// Increments when a new asynchronous GPU stats sample becomes available.
    pub fn stats_revision(&self) -> u64 {
        self.readback.revision
    }

    /// Whether `begin_frame` ran this frame (the late pass skips otherwise).
    pub fn prepared(&self) -> bool {
        self.prepared
    }

    /// Reads back the tile block (the overdraw valve's scales and the estimates
    /// they were stepped from) at the end of each of the next `frames` frames,
    /// asynchronously: the frame records one copy into a staging buffer the core
    /// owns and nothing waits. Collect them with [`Self::take_tile_valve_samples`].
    /// A frame that finds every staging buffer still in flight is skipped (its
    /// sample is missing, so frame numbers show the gap). Replaces any earlier
    /// request; 0 stops.
    pub fn watch_tile_valve(&mut self, frames: u32) {
        self.valve_readback.remaining = frames;
    }

    /// The tile-valve samples whose readback has completed, oldest first. Never
    /// waits for the GPU.
    pub fn take_tile_valve_samples(&mut self) -> Vec<crate::TileValveSample> {
        self.valve_readback.collect();
        self.valve_readback.samples.drain(..).collect()
    }

    /// Diagnostics (capture and tests only: it BLOCKS on the GPU). Every watched
    /// frame's sample, waiting for those still in flight. Call it after the
    /// frames' commands are submitted.
    pub fn tile_valve_samples_blocking(
        &mut self,
        device: &wgpu::Device,
    ) -> Vec<crate::TileValveSample> {
        self.valve_readback.map_copied();
        while self.valve_readback.busy() {
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            self.valve_readback.collect();
        }
        self.take_tile_valve_samples()
    }

    /// Records the copy of the tile block for a watched frame (after the frame's
    /// last phase: the estimate is complete and the scale is the one it used).
    fn record_tile_valve_copy(&mut self, rec: &mut CommandRecorder<'_>, device: &wgpu::Device) {
        let rb = &mut self.valve_readback;
        if rb.remaining == 0 {
            return;
        }
        rb.remaining -= 1;
        let b = &self.buffers;
        let bytes = VgCullBuffers::tile_block_words(b.tiles) * 4;
        let free = rb
            .slots
            .iter()
            .position(|s| matches!(s.state, ValveSlotState::Free) && s.buffer.size() >= bytes);
        let slot = match free {
            Some(i) => i,
            None => {
                // A slot too small for the grid is replaced; otherwise grow up to
                // the cap, and past it the frame is skipped.
                let small = rb.slots.iter().position(|s| {
                    matches!(s.state, ValveSlotState::Free) && s.buffer.size() < bytes
                });
                if small.is_none() && rb.slots.len() >= TILE_VALVE_SLOTS {
                    return;
                }
                let slot = ValveSlot {
                    buffer: device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("VgCull Tile Valve Readback"),
                        size: bytes.max(16),
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }),
                    state: ValveSlotState::Free,
                };
                match small {
                    Some(i) => {
                        rb.slots[i] = slot;
                        i
                    }
                    None => {
                        rb.slots.push(slot);
                        rb.slots.len() - 1
                    }
                }
            }
        };
        rec.copy_buffer_to_buffer(
            &b.state,
            u64::from(b.tile_base()) * 4,
            &rb.slots[slot].buffer,
            0,
            bytes,
        );
        rb.slots[slot].state = ValveSlotState::Copied {
            frame: self.frame,
            tiles: b.tiles,
        };
    }

    /// Start of frame: finish any stats read-back (growing capacities), size
    /// the tile block for the screen and upload the frame constants.
    pub fn begin_frame(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: &VgCullFrame,
        inputs: &VgCullInputs<'_>,
    ) {
        self.poll_readback();
        let b = &self.buffers;
        let bin_ceiling = self.config.max_bin_capacity.clamp(16, MAX_BIN_CAPACITY);
        // Ahead of this frame, to the most it can emit.
        let mut bin_cap = grown_to(b.bin_capacity, inputs.demand.clusters, bin_ceiling);
        let mut queue_cap = grown_to(b.queue_capacity, inputs.demand.jobs, QUEUE_DEMAND_CEILING);
        // And after an earlier one, by what the GPU saw.
        if let Some(words) = self.pending_capacity_stats.take() {
            let st = |i: u32| words[(state::STATS + i) as usize];
            let max_bin = (0..crate::BIN_COUNT)
                .map(|i| words[i as usize])
                .max()
                .unwrap_or(0);
            if st(stats::VISIBLE_OVERFLOW) > 0 || max_bin > bin_cap / 4 * 3 {
                let seen = (max_bin.max(b.bin_capacity) * 2).next_power_of_two();
                bin_cap = bin_cap.max(seen.min(bin_ceiling));
            }
            if st(stats::JOB_OVERFLOW) > 0 || st(stats::PEAK_QUEUE) > b.queue_capacity / 4 * 3 {
                queue_cap = queue_cap.max((b.queue_capacity * 2).min(MAX_QUEUE_CAPACITY));
            }
            if std::env::var_os("VG_DEBUG").is_some() {
                eprintln!(
                    "VgCull stats: bins {:?} jobs {} tested {} emitted {} deferred {} culled(frustum {} occl {} cone {} sse {}) page_req {} overflow(job {} vis {})",
                    &words[..crate::BIN_COUNT as usize],
                    words[(state::STATS + stats::JOBS) as usize],
                    words[(state::STATS + stats::CLUSTERS_TESTED) as usize],
                    words[(state::STATS + stats::EMITTED) as usize],
                    words[(state::STATS + stats::DEFERRED) as usize],
                    words[(state::STATS + stats::CULLED_FRUSTUM) as usize],
                    words[(state::STATS + stats::CULLED_OCCLUSION) as usize],
                    words[(state::STATS + stats::CULLED_CONE) as usize],
                    words[(state::STATS + stats::CULLED_SSE) as usize],
                    words[(state::STATS + stats::PAGE_REQUESTS) as usize],
                    words[(state::STATS + stats::JOB_OVERFLOW) as usize],
                    words[(state::STATS + stats::VISIBLE_OVERFLOW) as usize],
                );
            }
            self.last_stats = Some(words);
        }
        if bin_cap != b.bin_capacity || queue_cap != b.queue_capacity {
            self.buffers = Arc::new(b.with_capacities(device, bin_cap, queue_cap));
            self.bind_cache = [None, None];
        }
        let tiles = [
            frame.width.max(1).div_ceil(crate::DENSE_TILE),
            frame.height.max(1).div_ceil(crate::DENSE_TILE),
        ];
        if tiles != self.buffers.tiles {
            self.buffers = Arc::new(self.buffers.with_tiles(device, tiles));
            self.bind_cache = [None, None];
        }
        let b = &self.buffers;
        self.frame += 1;
        self.size = (frame.width.max(1), frame.height.max(1));
        let mut flags = 0;
        if self.config.two_phase_occlusion {
            flags |= FLAG_TWO_PHASE;
        }
        if inputs.dense_mask.is_some() {
            flags |= FLAG_DENSE_MASK;
        }
        if self.config.stats {
            flags |= FLAG_STATS;
        }
        if !self.config.root_shortcut {
            flags |= FLAG_NO_ROOT_SHORTCUT;
        }
        if self.config.seeds_instances {
            flags |= crate::FLAG_EXPLICIT_INSTANCES;
        }
        if self.config.current_hiz {
            flags |= FLAG_CURRENT_HIZ;
        }
        if inputs.materials.is_some() {
            flags |= FLAG_MASKED;
            if self.config.sw_masked {
                flags |= FLAG_SW_MASKED;
            }
        }
        let params = GpuParams {
            screen_width: self.size.0,
            screen_height: self.size.1,
            hiz_mip_count: mip_levels(self.size.0, self.size.1),
            mode: mode_bits(self.config.mode),
            tau_px: self.config.tau_px,
            sw_edge_px: self.config.sw_edge_px,
            instance_count: inputs.instance_count,
            bin_capacity: b.bin_capacity,
            queue_capacity: b.queue_capacity,
            history_bits_mask: b.history_words * 32 - 1,
            hist_base: state::HISTORY,
            ro_tile_base: b.history_words,
            flags,
            dense_tiles_x: inputs.dense_tiles[0].max(1),
            dense_tiles_y: inputs.dense_tiles[1].max(1),
            max_groups: self.config.max_groups.max(1),
            tile_limit: self.config.tile_density_limit.max(0.0),
            prefetch_k: self.config.prefetch_factor.max(1.0),
            tiles_x: b.tiles[0],
            tiles_y: b.tiles[1],
            tile_base: b.tile_base(),
            _pad: [0; 3],
        };
        helio_core::upload::write_buffer(queue, &self.params_buf, 0, bytemuck::bytes_of(&params));
        self.prepared = true;
    }

    fn poll_readback(&mut self) {
        // Last frame's tile-valve copy is submitted by now.
        self.valve_readback.collect();
        self.valve_readback.map_copied();
        let rb = &mut self.readback;
        if rb.copied && !rb.in_flight {
            let ready = rb.ready.clone();
            let buf = rb.staging.clone();
            rb.staging
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |r| {
                    if r.is_ok() {
                        let words: Vec<u32> = {
                            let Ok(view) = buf.slice(..).get_mapped_range() else {
                                return;
                            };
                            bytemuck::cast_slice::<u8, u32>(&view).to_vec()
                        };
                        *ready.lock().unwrap() = Some(words);
                    }
                });
            rb.in_flight = true;
            rb.copied = false;
        }
        if rb.in_flight {
            if let Some(words) = rb.ready.lock().unwrap().take() {
                rb.staging.unmap();
                rb.in_flight = false;
                rb.revision = rb.revision.wrapping_add(1);
                self.pending_capacity_stats = Some(words.clone());
                self.last_stats = Some(words);
            }
        }
    }

    /// Records one phase (0 = early, 1 = late). `hiz` is that phase's Hi-Z
    /// (previous frame for 0, the phase-0 build for 1); `None` skips occlusion
    /// for the phase.
    pub fn record_phase(
        &mut self,
        rec: &mut CommandRecorder<'_>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        camera: &wgpu::Buffer,
        inputs: &VgCullInputs<'_>,
        hiz: Option<&wgpu::TextureView>,
        phase: u32,
    ) {
        let b = self.buffers.clone();
        let marks_bytes = u64::from(state::HISTORY) * 4;
        let hist_bytes = u64::from(b.history_words) * 4;
        if phase == 0 {
            // Last frame's marks become this frame's read-only set, then the
            // counters and the marks start over. Nothing writes the read set
            // during the frame, so both phases read the same snapshot.
            rec.copy_buffer_to_buffer(&b.state, marks_bytes, &b.history_ro, 0, hist_bytes);
            rec.clear_buffer(&b.state, 0, Some(marks_bytes + hist_bytes));
            // The queues are all zero between phases (every reserved slot is
            // taken and zeroed), except the deferred queue when the early
            // phase's deferrals were never taken by a late phase.
            if self.late_owed {
                let q = u64::from(b.queue_capacity) * 8;
                rec.clear_buffer(&b.queues, q, Some(q));
            }
            self.late_owed = self.config.two_phase_occlusion;
        } else {
            self.late_owed = false;
        }
        // Phase 0 trusts last frame's Hi-Z only from the second frame on
        // (frame 1 has no previous depth). Stale-but-valid is still safe.
        // This frame's Hi-Z (`current_hiz`) is valid whenever it is there.
        let hiz_valid = hiz.is_some()
            && (self.config.current_hiz
                || (self.config.two_phase_occlusion && (phase == 1 || self.frame >= 2)));

        let block = u64::from(phase) * ROUND_STRIDE;
        queue.write_buffer(
            &self.round_buf,
            block,
            bytemuck::bytes_of(&GpuPhase {
                phase,
                hiz_valid: hiz_valid as u32,
                _pad: [0; 2],
            }),
        );

        let bg0 = self.bind_group0(device, camera, inputs, hiz, phase as usize, &b);
        if phase == 0 {
            // The valve scales of this frame, then their read-only copy (the
            // tile words the traversal accumulates into are copied too and
            // never read from the copy).
            let mut pass = rec.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("VgCull Tiles"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &bg0, &[]);
            pass.set_bind_group(1, &self.bg1, &[0]);
            pass.set_pipeline(&self.pipelines.tiles);
            pass.dispatch_workgroups((b.tiles[0] * b.tiles[1]).div_ceil(64), 1, 1);
            drop(pass);
            rec.copy_buffer_to_buffer(
                &b.state,
                u64::from(b.tile_base()) * 4,
                &b.history_ro,
                hist_bytes,
                VgCullBuffers::tile_block_words(b.tiles) * 4,
            );
        }
        let mut pass = rec.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(if phase == 0 {
                "VgCull Early"
            } else {
                "VgCull Late"
            }),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, &bg0, &[]);
        pass.set_bind_group(1, &self.bg1, &[block as u32]);
        // The phase's queued work: the root (early), the early phase's
        // deferrals (late).
        if phase == 0 && self.config.seeds_instances {
            pass.set_pipeline(&self.pipelines.seed_instances);
            pass.dispatch_workgroups(inputs.instance_count.div_ceil(64).max(1), 1, 1);
        } else {
            pass.set_pipeline(if phase == 0 {
                &self.pipelines.seed
            } else {
                &self.pipelines.phase_init
            });
            pass.dispatch_workgroups(1, 1, 1);
        }
        // The whole traversal, one persistent dispatch (see the crate docs).
        pass.set_pipeline(&self.pipelines.traverse);
        pass.dispatch_workgroups(self.config.max_groups.max(1), 1, 1);
        pass.set_pipeline(&self.pipelines.finalize);
        pass.dispatch_workgroups(1, 1, 1);
        drop(pass);

        let last = if self.config.two_phase_occlusion {
            1
        } else {
            0
        };
        if phase == last && !self.readback.copied && !self.readback.in_flight {
            rec.copy_buffer_to_buffer(
                &b.state,
                0,
                &self.readback.staging,
                0,
                state::WORDS as u64 * 4,
            );
            self.readback.copied = true;
        }
        if phase == last {
            self.record_tile_valve_copy(rec, device);
        }
    }

    fn bind_group0(
        &mut self,
        device: &wgpu::Device,
        camera: &wgpu::Buffer,
        i: &VgCullInputs<'_>,
        hiz: Option<&wgpu::TextureView>,
        slot: usize,
        b: &VgCullBuffers,
    ) -> wgpu::BindGroup {
        let dense = i.dense_mask.unwrap_or(&self.dummy_buf);
        let materials = i.materials.unwrap_or(&self.dummy_buf);
        let mut bufs: Vec<&wgpu::Buffer> = vec![camera, &self.params_buf, i.instances, i.objects];
        bufs.extend(i.stream.as_array());
        let tree = i.tree.unwrap_or(VgCullTree {
            nodes: &self.dummy_tree,
            header: &self.dummy_tree,
            items: &self.dummy_tree,
        });
        bufs.extend([
            tree.nodes,
            tree.header,
            tree.items,
            dense,
            &b.state,
            &b.queues,
            &b.visible,
            &b.history_ro,
        ]);
        bufs.push(materials);
        let key: Vec<wgpu::Buffer> = bufs.iter().map(|b| (*b).clone()).collect();
        if let Some((k, v, bg)) = &self.bind_cache[slot] {
            if *k == key && v.as_ref() == hiz {
                return bg.clone();
            }
        }
        let view = hiz.unwrap_or(&self.dummy_view);
        let mut entries: Vec<wgpu::BindGroupEntry<'_>> = Vec::new();
        // bufs[0..12] = bindings 0..=11, bufs[12..16] = 12..=15, then texture (16),
        // sampler (17), then bufs[16..20] = 18..=21.
        for (n, buf) in bufs.iter().enumerate().take(16) {
            entries.push(wgpu::BindGroupEntry {
                binding: n as u32,
                resource: buf.as_entire_binding(),
            });
        }
        entries.push(wgpu::BindGroupEntry {
            binding: 16,
            resource: wgpu::BindingResource::TextureView(view),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: 17,
            resource: wgpu::BindingResource::Sampler(&self.sampler),
        });
        for (n, buf) in bufs.iter().enumerate().skip(16).take(4) {
            entries.push(wgpu::BindGroupEntry {
                binding: n as u32 + 2,
                resource: buf.as_entire_binding(),
            });
        }
        entries.push(wgpu::BindGroupEntry {
            binding: 22,
            resource: bufs[20].as_entire_binding(),
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("VgCull BG0"),
            layout: &self.bgl0,
            entries: &entries,
        });
        self.bind_cache[slot] = Some((key, hiz.cloned(), bg.clone()));
        bg
    }
}
