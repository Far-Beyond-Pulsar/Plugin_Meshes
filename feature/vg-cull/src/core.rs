//! GPU resources and command recording of the cull. Independent of the render
//! graph so tests and offline tools can drive it with a plain encoder; the
//! passes in `pass.rs` are thin wrappers.

use std::num::NonZeroU64;
use std::sync::{Arc, Mutex};

use bytemuck::{Pod, Zeroable};
use helio_core::CommandRecorder;
use helio_vg_format::VgMode;

use crate::{
    mode_bits, state, stats, FLAG_DENSE_MASK, FLAG_MASKED, FLAG_NO_ROOT_SHORTCUT, FLAG_STATS,
    FLAG_SW_MASKED, FLAG_TWO_PHASE, MAX_BIN_CAPACITY,
};

/// Stride of the per-phase uniform blocks (dynamic-offset alignment).
const ROUND_STRIDE: u64 = 256;
/// Job queues: the early phase's, and the deferred work the late phase starts
/// from (see the crate docs, "Traversal").
const QUEUES: u64 = 2;

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
    /// Initial entries per bin. The default is the ceiling [`MAX_BIN_CAPACITY`]:
    /// a bin that overflows drops clusters for that frame, and growing only after
    /// an asynchronous stats readback means several frames of silently missing
    /// geometry (which changes with the camera: it reads as flicker) plus a
    /// rebuild of every cull buffer that resets the visibility history. 8 bins of
    /// 1M entries cost 64 MiB, so the ceiling is allocated up front. Tests lower it
    /// to exercise the growth path.
    pub initial_bin_capacity: u32,
    /// Initial entries per job queue (3 queues, 8 bytes per entry). Jobs past a
    /// queue are lost work, so this also starts large (24 MiB); it still grows
    /// like the bins if a frame ever fills it.
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
}

impl Default for VgCullConfig {
    fn default() -> Self {
        Self {
            tau_px: 1.0,
            sw_edge_px: 2.0,
            tile_density_limit: 8.0,
            prefetch_factor: 2.0,
            mode: VgMode::Auto,
            two_phase_occlusion: true,
            max_groups: 256,
            initial_bin_capacity: MAX_BIN_CAPACITY,
            initial_queue_capacity: 1 << 20,
            history_bits: 1 << 22,
            stats: true,
            max_bin_capacity: MAX_BIN_CAPACITY,
            sw_masked: false,
            root_shortcut: std::env::var_os("HELIO_VG_NO_ROOT_SHORTCUT").is_none(),
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
/// | `objects` (`"static_objects"`) | 252-byte (63-word) scene-join row; `transform` is read via `GpuVgInstance::object_row` |
/// | `stream` | the eight `helio-vg-stream` buffers (pool, residency, nodes, group refs, headers, bases, quant, feedback); the cull appends page requests / touches straight into `feedback` |
/// | `gcbvh_nodes`/`gcbvh_header`/`gcbvh_items` | `helio-pass-vg-bvh` outputs (`vg_gcbvh_*`) |
/// | `dense_mask` | one `u32` per 16x16 tile, row-major, nonzero = dense last frame |
#[derive(Clone, Copy)]
pub struct VgCullInputs<'a> {
    pub instances: &'a wgpu::Buffer,
    pub objects: &'a wgpu::Buffer,
    pub stream: helio_vg_stream::VgStreamBuffers<'a>,
    pub gcbvh_nodes: &'a wgpu::Buffer,
    pub gcbvh_header: &'a wgpu::Buffer,
    pub gcbvh_items: &'a wgpu::Buffer,
    pub dense_mask: Option<&'a wgpu::Buffer>,
    /// The scene's `materials` table (`GpuMaterial` rows, 24 words). `Some` turns on
    /// alpha-mask routing; `None` is the opaque-only path with no material loads.
    pub materials: Option<&'a wgpu::Buffer>,
    /// Rows of `instances` (the traversal ignores BVH leaves beyond it).
    pub instance_count: u32,
    /// Dense-mask grid `[tiles_x, tiles_y]`.
    pub dense_tiles: [u32; 2],
}

/// Per-frame values the graph knows.
#[derive(Clone, Copy, Debug)]
pub struct VgCullFrame {
    pub width: u32,
    pub height: u32,
}

/// Buffers sized for the current capacities. Replaced (new `Arc`) when they
/// grow; pass wrappers re-read it each frame.
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
        let storage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let mk = |label: &str, bytes: u64, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(16),
                usage,
                mapped_at_creation: false,
            })
        };
        let (state, history_ro) = Self::history_buffers(device, history_words, tiles);
        Self {
            visible: mk(
                "VgCull Visible",
                crate::BIN_COUNT as u64 * bin_capacity as u64 * 8,
                storage,
            ),
            state,
            queues: mk("VgCull Queues", QUEUES * queue_capacity as u64 * 8, storage),
            history_ro,
            bin_capacity,
            queue_capacity,
            history_words,
            tiles,
        }
    }

    /// `state` and `history_ro`, sized for the mark set and the tile grid.
    fn history_buffers(
        device: &wgpu::Device,
        history_words: u32,
        tiles: [u32; 2],
    ) -> (wgpu::Buffer, wgpu::Buffer) {
        let usage =
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
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

struct Pipelines {
    tiles: wgpu::ComputePipeline,
    seed: wgpu::ComputePipeline,
    phase_init: wgpu::ComputePipeline,
    traverse: wgpu::ComputePipeline,
    finalize: wgpu::ComputePipeline,
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
    buffers: Arc<VgCullBuffers>,
    bind_cache: [BindCache; 2],
    frame: u64,
    readback: Readback,
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
            helio_core::shader::create_compute_pipeline(
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
        let mut bin_cap = b.bin_capacity;
        let mut queue_cap = b.queue_capacity;
        if let Some(words) = self.pending_capacity_stats.take() {
            let st = |i: u32| words[(state::STATS + i) as usize];
            let max_bin = (0..crate::BIN_COUNT)
                .map(|i| words[i as usize])
                .max()
                .unwrap_or(0);
            if st(stats::VISIBLE_OVERFLOW) > 0 || max_bin > bin_cap / 4 * 3 {
                bin_cap = (max_bin.max(bin_cap) * 2)
                    .next_power_of_two()
                    .min(self.config.max_bin_capacity.clamp(16, MAX_BIN_CAPACITY));
            }
            if st(stats::JOB_OVERFLOW) > 0 || st(stats::PEAK_QUEUE) > queue_cap / 4 * 3 {
                queue_cap = (queue_cap * 2).min(1 << 24);
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
            self.buffers = Arc::new(VgCullBuffers::new(
                device,
                bin_cap,
                queue_cap,
                b.history_words,
                b.tiles,
            ));
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
        let hiz_valid =
            self.config.two_phase_occlusion && hiz.is_some() && (phase == 1 || self.frame >= 2);

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
        pass.set_pipeline(if phase == 0 {
            &self.pipelines.seed
        } else {
            &self.pipelines.phase_init
        });
        pass.dispatch_workgroups(1, 1, 1);
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
        bufs.extend([
            i.gcbvh_nodes,
            i.gcbvh_header,
            i.gcbvh_items,
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
