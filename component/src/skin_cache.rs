//! The skin cache: every skinned instance's posed vertices, written once a
//! frame by one compute pass (`skin_cache.wgsl`) for the instances whose pose
//! changed, and read by everything that draws or traces a skinned mesh: the
//! G-buffer, shadow, forward, transparent and selection drawers
//! (`helio_mesh_api::SKINNING_SNIPPET`) and each skinned instance's deforming
//! BLAS (`crate::ray_scene`). What is rasterised is what is traced, and the
//! palette is blended in one place.
//!
//! - **What it holds.** A range of [`SkinnedVertex`]es for each skinned
//!   instance whose mesh is loaded and mirrored (`resident`), in one growable
//!   buffer, plus an instance table by entity row ([`SkinCacheRow`], the
//!   instance rows' `mesh_id`) the drawers find a range by.
//! - **What a frame costs.** The journal of [`SkeletalMeshComponent`] says
//!   which instances changed; one whose palette is bitwise the one it was
//!   posed with costs nothing (an animator showing the same pose each frame).
//!   A posed instance is skinned once: its current palette and the palette
//!   shown before it (both mirrored by the component) are blended in the same
//!   pass, so previous positions for motion vectors cost no read of the cache
//!   (that read was a third of the pass, which is memory-bound), and
//!   on the frame after its last pose its row is marked settled (previous =
//!   current, a row write and no pass over its vertices, so a
//!   stopped pose stops moving in the velocity buffer). A frame with no
//!   change records nothing; no buffer or bind group is made unless the
//!   cache or a pool grows.
//! - **What it scales with.** Memory: 32 bytes a vertex of every resident
//!   skinned instance (an instance deforms by itself, so its vertices are its
//!   own). Compute: the vertices of the instances posed this frame. CPU: the
//!   instances whose component changed.
//!
//! How others do it: Unreal's GPU skin cache (required by its ray tracing)
//! writes each skinned primitive's positions and tangents once a frame in
//! compute and keeps the previous frame's positions for velocity; the
//! raster and the ray-tracing geometry both read it. Frostbite and Decima
//! pre-skin in compute as well. Unreal sizes its cache with a memory budget
//! and falls back to vertex-shader skinning past it (a second skinning path);
//! this cache holds every resident instance, so there is one path, and the
//! lever on its memory is the skinned meshes' level of detail (open).

use std::collections::{HashMap, HashSet};
use std::hash::Hasher;
use std::sync::{Arc, Mutex};

use helio_core::{BufferHandle, BufferKey};
use helio_mesh_api::{SkinCacheRow, SkinnedVertex};
use pulsar_scenedb::gpu::SceneGpuStore;
use pulsar_scenedb::{
    ChangeCursor, ChangeRead, ComponentChange, ComponentChangeKind, Entity, World,
};

use crate::components::SkeletalMeshComponent;

/// Lanes of a workgroup of the skin pass (`LANES` in `skin_cache.wgsl`).
const LANES: u32 = 64;
/// Workgroups a dispatch row holds (the device limit every backend has).
const GROUPS_X: u32 = 65_535;
/// Vertices the cache holds before its first growth, and the least it
/// shrinks to.
const INITIAL_VERTICES: u32 = 1 << 14;
/// Frames an instance stays resident while nothing wants it (off-screen and
/// outside the traced range): a character that turns out of view and back
/// keeps its range; one left behind gives it back.
const EVICT_FRAMES: u32 = 300;
const VERTEX_BYTES: u64 = std::mem::size_of::<SkinnedVertex>() as u64;
const ROW_BYTES: u64 = std::mem::size_of::<SkinCacheRow>() as u64;

/// A job of the skin pass (`SkinJob` in `skin_cache.wgsl`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct SkinJob {
    mesh_vertex: u32,
    skin_vertex: u32,
    palette: u32,
    joints: u32,
    cache: u32,
    count: u32,
    mode: u32,
    first_group: u32,
}

/// The skin pass's frame (`SkinFrame` in `skin_cache.wgsl`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct SkinFrame {
    jobs: u32,
    groups: u32,
    groups_x: u32,
    _pad: u32,
}

/// `SKIN_FRESH`, `SKIN_POSE` in `skin_cache.wgsl`.
const FRESH: u32 = 0;
const POSE: u32 = 1;

/// A resident skinned instance as the ray-traced scene reads it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SkinnedInstance {
    /// Its first vertex in the cache, and their count.
    pub cache_vertex: u32,
    pub vertex_count: u32,
    /// Its first vertex in the mesh pool (UVs; the tracer's rows).
    pub mesh_vertex: u32,
    /// Changes whenever the skin pass wrote new positions (a BLAS refits).
    pub revision: u64,
}

struct Entry {
    cache: u32,
    count: u32,
    mesh_vertex: u32,
    skin_vertex: u32,
    palette: (u32, u32),
    palette_hash: u64,
    revision: u64,
    /// The frame it was last posed in.
    posed: u64,
    /// Its pose stopped: its row says previous = current.
    settled: bool,
}

/// First-fit ranges of the cache's vertices.
#[derive(Default)]
struct Ranges {
    /// Free `(start, len)`, sorted, never adjacent.
    free: Vec<(u32, u32)>,
}

impl Ranges {
    fn take(&mut self, len: u32) -> Option<u32> {
        let at = self.free.iter().position(|&(_, l)| l >= len)?;
        let (start, l) = self.free[at];
        if l == len {
            self.free.remove(at);
        } else {
            self.free[at] = (start + len, l - len);
        }
        Some(start)
    }

    fn give(&mut self, start: u32, len: u32) {
        let at = self.free.partition_point(|&(s, _)| s < start);
        self.free.insert(at, (start, len));
        // Merge with the next, then the previous.
        if at + 1 < self.free.len() && start + len == self.free[at + 1].0 {
            self.free[at].1 += self.free[at + 1].1;
            self.free.remove(at + 1);
        }
        if at > 0 && self.free[at - 1].0 + self.free[at - 1].1 == start {
            self.free[at - 1].1 += self.free[at].1;
            self.free.remove(at);
        }
    }
}

/// What the frame sees: the cache skins only the instances it wants (in the
/// camera's frustum, or within the ray-traced scene's culling criterion).
#[derive(Clone, Copy, Debug)]
pub struct SkinView {
    /// The eye, in world space.
    pub eye: glam::DVec3,
    /// The camera's view-projection relative to the eye
    /// (`helio::Camera::view_proj_in(eye)`).
    pub view_proj: glam::Mat4,
    /// The ray-traced scene's culling criterion.
    pub rays: helio_pass_mesh_rays::RayTracingSettings,
}

impl SkinView {
    /// Whether a skinned instance whose world bounding sphere is `center`,
    /// `radius` is drawn or traced this frame: inside the frustum's four side
    /// planes (a sphere behind the eye fails one), or traced by the
    /// ray-tracing culling (which covers the shadow casters near the eye).
    pub fn wants(&self, center: glam::DVec3, radius: f64) -> bool {
        let distance = center.distance(self.eye);
        if self.rays.traces(distance, radius, 1.0) {
            return true;
        }
        let c = (center - self.eye).as_vec3().extend(1.0);
        let m = self.view_proj;
        let row = |i: usize| glam::Vec4::new(m.x_axis[i], m.y_axis[i], m.z_axis[i], m.w_axis[i]);
        let (x, y, w) = (row(0), row(1), row(3));
        [w + x, w - x, w + y, w - y].iter().all(|plane| {
            let n = plane.truncate().length();
            plane.dot(c) >= -(radius as f32) * n
        })
    }
}

/// A skinned instance the cache knows, resident or not.
#[derive(Default)]
struct Known {
    /// Its component changed and it is not posed from it yet (it was not
    /// wanted, or its mesh was not mirrored).
    pending: bool,
    /// Its pose changed while nothing wanted it: posed fresh (no motion)
    /// when it is wanted, its previous pose being stale.
    missed: bool,
    /// Frames nothing has wanted it.
    unwanted: u32,
    /// It has no mesh to skin (until its component changes).
    idle: bool,
}

/// What the cache did this frame, for the ray-traced scene.
#[derive(Default)]
pub struct SkinCacheFrame {
    /// Instances whose cache range or mesh range changed (their ray rows and
    /// BLAS must follow), or that left the cache.
    pub moved: Vec<Entity>,
}

/// The cache's buffers as the frame's scene buffers
/// ([`helio_mesh_api::SKIN_CACHE`], [`helio_mesh_api::SKIN_CACHE_ROWS`]),
/// shared with the derivation that publishes them ([`SkinCacheBuffers`]).
type Published = Arc<Mutex<Vec<(BufferKey, BufferHandle)>>>;

/// The skin cache of one renderer; see the module docs.
pub struct SkinCache {
    device: Arc<wgpu::Device>,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    frame_uniform: wgpu::Buffer,
    jobs_buffer: wgpu::Buffer,
    cache: wgpu::Buffer,
    capacity: u32,
    used: u32,
    ranges: Ranges,
    rows: wgpu::Buffer,
    rows_capacity: u32,
    cache_usage: wgpu::BufferUsages,
    entries: HashMap<Entity, Entry>,
    cursor: Option<ChangeCursor>,
    changes: Vec<ComponentChange>,
    /// Instances posed last frame.
    posed_last: Vec<Entity>,
    /// Every skinned instance the journal has shown, resident or not.
    known: HashMap<Entity, Known>,
    /// Instances skinned this frame and not wanted (zero with a view).
    jobs: Vec<SkinJob>,
    bind_group: Option<([wgpu::Buffer; 5], wgpu::BindGroup)>,
    frame: u64,
    epoch: u64,
    generation: u64,
    published: Published,
    // This frame's counts.
    updated: u64,
    vertices: u64,
}

impl SkinCache {
    pub fn new(device: Arc<wgpu::Device>) -> Self {
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let read = wgpu::BufferBindingType::Storage { read_only: true };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Skin cache"),
            entries: &[
                entry(0, wgpu::BufferBindingType::Uniform),
                entry(1, read),
                entry(2, read),
                entry(3, read),
                entry(4, read),
                entry(5, wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        });
        let module = helio_core::shader::module_with(
            &device,
            "Skin cache",
            helio_core::include_wgsl!("skin_cache.wgsl"),
            &[
                helio_mesh_api::DRAW_ROWS_SNIPPET,
                helio_mesh_api::SKINNING_SNIPPET,
                helio_frame_api::rays::stream::SNIPPET,
            ],
        );
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Skin cache"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Skin cache"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("skin"),
            compilation_options: Default::default(),
            cache: None,
        });
        // The deforming BLASes build from the cache directly.
        let mut cache_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        if device
            .features()
            .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
        {
            cache_usage |= wgpu::BufferUsages::BLAS_INPUT;
        }
        let buffer = |label, size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let cache = buffer(
            "Skin cache",
            u64::from(INITIAL_VERTICES) * VERTEX_BYTES,
            cache_usage,
        );
        let rows = buffer(
            "Skin cache rows",
            64 * ROW_BYTES,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );
        let frame_uniform = buffer(
            "Skin cache frame",
            std::mem::size_of::<SkinFrame>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let jobs_buffer = buffer(
            "Skin cache jobs",
            64 * std::mem::size_of::<SkinJob>() as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let this = Self {
            device,
            pipeline,
            layout,
            frame_uniform,
            jobs_buffer,
            cache,
            capacity: INITIAL_VERTICES,
            used: 0,
            ranges: Ranges {
                free: vec![(0, INITIAL_VERTICES)],
            },
            rows,
            rows_capacity: 64,
            cache_usage,
            entries: HashMap::new(),
            cursor: None,
            changes: Vec::new(),
            posed_last: Vec::new(),
            known: HashMap::new(),
            jobs: Vec::new(),
            bind_group: None,
            frame: 0,
            epoch: 0,
            generation: 0,
            published: Arc::new(Mutex::new(Vec::new())),
            updated: 0,
            vertices: 0,
        };
        this.publish();
        this
    }

    /// The derivation that publishes the cache's buffers into each frame's
    /// scene buffers (where the drawers resolve them by key).
    pub fn derivation(&self) -> SkinCacheBuffers {
        SkinCacheBuffers(self.published.clone())
    }

    /// The cache's vertices (what deforming BLASes build from and the
    /// meshes' tracer reads skinned triangles from).
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.cache
    }

    /// The instance table ([`SkinCacheRow`] by entity row).
    pub fn rows(&self) -> &wgpu::Buffer {
        &self.rows
    }

    /// `entity`'s range, if the cache holds it.
    pub fn instance(&self, entity: Entity) -> Option<SkinnedInstance> {
        self.entries.get(&entity).map(|e| SkinnedInstance {
            cache_vertex: e.cache,
            vertex_count: e.count,
            mesh_vertex: e.mesh_vertex,
            revision: e.revision,
        })
    }

    /// The instances the cache holds.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Bring the cache up to date with `world` (after the GPU mirror's
    /// flush): skin the instances whose pose changed into `encoder`, settle
    /// those posed last frame and not this one. With a `view`, only the
    /// instances it wants are skinned ([`SkinView::wants`]): one posed while
    /// unwanted is skinned the frame it is wanted again (with no motion that
    /// frame: its previous pose is stale), and one unwanted for
    /// [`EVICT_FRAMES`] gives its range back. Without one every instance is.
    pub fn update(
        &mut self,
        world: &World,
        view: Option<&SkinView>,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
    ) -> SkinCacheFrame {
        self.frame += 1;
        self.updated = 0;
        self.vertices = 0;
        self.jobs.clear();
        let mut out = SkinCacheFrame::default();
        let Some(mirror) = world.gpu_mirror() else {
            return out;
        };
        let store = mirror.store();
        let mut dirty = HashSet::new();
        let complete = match self.cursor.as_mut() {
            Some(cursor) => {
                self.changes.clear();
                world.read_changes(cursor, &mut self.changes) == ChangeRead::Complete
            }
            None => false,
        };
        if complete {
            let changes = std::mem::take(&mut self.changes);
            for change in &changes {
                match change.kind {
                    ComponentChangeKind::Removed => {
                        self.known.remove(&change.entity);
                        if self.free(change.entity, queue) {
                            out.moved.push(change.entity);
                        }
                    }
                    _ => {
                        dirty.insert(change.entity);
                    }
                }
            }
            self.changes = changes;
        } else {
            // First frame or a journal overflow: every instance, from a scan.
            self.cursor = Some(world.open_change_cursor::<SkeletalMeshComponent>());
            let alive: HashSet<Entity> = world
                .query::<&SkeletalMeshComponent>()
                .map(|(entity, _)| entity)
                .collect();
            let gone: Vec<Entity> = self
                .entries
                .keys()
                .copied()
                .filter(|e| !alive.contains(e))
                .collect();
            for entity in gone {
                self.free(entity, queue);
                out.moved.push(entity);
            }
            self.known.retain(|entity, _| alive.contains(entity));
            dirty.extend(alive);
        }
        for entity in &dirty {
            let known = self.known.entry(*entity).or_default();
            known.pending = true;
            known.idle = false;
        }
        // Which instances the frame wants: every one it knows, a sphere test
        // each (a skinned instance costs its animation's CPU a frame anyway).
        let mut wanted = Vec::new();
        let mut evict = Vec::new();
        for (entity, known) in &mut self.known {
            let wants = view.is_none_or(|view| {
                world_sphere(world, *entity).is_none_or(|(c, r)| view.wants(c, r))
            });
            let resident = self.entries.contains_key(entity);
            if wants {
                known.unwanted = 0;
                if known.pending || (!known.idle && !resident) {
                    wanted.push((*entity, known.missed && resident));
                    known.pending = false;
                    known.missed = false;
                }
            } else {
                known.missed |= known.pending;
                known.unwanted = known.unwanted.saturating_add(1);
                if known.unwanted > EVICT_FRAMES && self.entries.contains_key(entity) {
                    evict.push(*entity);
                }
            }
        }
        for entity in evict {
            if self.free(entity, queue) {
                out.moved.push(entity);
            }
        }
        let mut posed = Vec::new();
        for (entity, missed) in wanted {
            // A pose that changed while unwanted has no previous pose to move
            // from: it is posed fresh.
            match self.pose(world, store, entity, encoder, queue, missed) {
                Pose::Unchanged => {}
                Pose::Posed { moved } => {
                    posed.push(entity);
                    if moved {
                        out.moved.push(entity);
                    }
                }
                Pose::Waiting => {
                    if let Some(known) = self.known.get_mut(&entity) {
                        known.pending = true;
                    }
                }
                Pose::Gone => {
                    if let Some(known) = self.known.get_mut(&entity) {
                        known.idle = true;
                    }
                    if self.free(entity, queue) {
                        out.moved.push(entity);
                    }
                }
            }
        }
        // Posed last frame and not this one: settled (its row says the
        // previous position is the current one; no pass over its vertices).
        let posed_now: HashSet<Entity> = posed.iter().copied().collect();
        for entity in std::mem::take(&mut self.posed_last) {
            if posed_now.contains(&entity) {
                continue;
            }
            if let Some(e) = self.entries.get_mut(&entity) {
                e.settled = true;
                let row = SkinCacheRow {
                    offset: e.cache.wrapping_sub(e.mesh_vertex),
                    resident: 1,
                    pose: e.revision as u32,
                    settled: 1,
                };
                self.write_row(queue, encoder, entity.index(), row);
            }
        }
        self.posed_last = posed;
        self.shrink(queue, encoder, &mut out);
        self.dispatch(store, queue, encoder);
        self.publish();
        out
    }

    /// Pose `entity` from its component, if its palette or mesh changed.
    fn pose(
        &mut self,
        world: &World,
        store: &SceneGpuStore,
        entity: Entity,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        fresh: bool,
    ) -> Pose {
        let Some(mesh) = world.get::<SkeletalMeshComponent>(entity) else {
            return Pose::Gone;
        };
        if mesh.vertices.is_empty() || mesh.bone_palette.is_empty() {
            return Pose::Gone;
        }
        let row = entity.index();
        let (Some(vertices), Some(skin), Some(palette)) = (
            SkeletalMeshComponent::vertices_gpu_handle(store, row),
            SkeletalMeshComponent::skin_vertices_gpu_handle(store, row),
            SkeletalMeshComponent::gpu_bone_palettes_gpu_handle(store, row),
        ) else {
            return Pose::Waiting;
        };
        let count = mesh.vertices.len() as u32;
        if vertices.count != count
            || skin.count != count
            // The current palette then the previous one.
            || palette.count as usize != 2 * mesh.bone_palette.len()
        {
            return Pose::Waiting;
        }
        let hash = {
            let mut h = twox_hash::XxHash3_64::default();
            h.write(bytemuck::cast_slice(&mesh.bone_palette));
            h.finish()
        };
        let (moved, mode) = match self.entries.get(&entity) {
            Some(e)
                if e.count == count
                    && e.mesh_vertex == vertices.offset
                    && e.skin_vertex == skin.offset
                    && e.palette == (palette.offset, palette.count)
                    && e.palette_hash == hash =>
            {
                return Pose::Unchanged;
            }
            Some(e) if e.count == count => (
                e.mesh_vertex != vertices.offset,
                if fresh { FRESH } else { POSE },
            ),
            Some(_) => {
                self.free(entity, queue);
                (true, FRESH)
            }
            None => (true, FRESH),
        };
        if !self.entries.contains_key(&entity) {
            let Some(cache) = self.allocate(count, encoder) else {
                tracing::warn!("Meshes: the skin cache cannot hold {count} more vertices");
                return Pose::Waiting;
            };
            self.entries.insert(
                entity,
                Entry {
                    cache,
                    count,
                    mesh_vertex: vertices.offset,
                    skin_vertex: skin.offset,
                    palette: (palette.offset, palette.count),
                    palette_hash: hash,
                    revision: 0,
                    posed: 0,
                    settled: false,
                },
            );
        }
        let frame = self.frame;
        let e = self.entries.get_mut(&entity).expect("held");
        e.mesh_vertex = vertices.offset;
        e.skin_vertex = skin.offset;
        e.palette = (palette.offset, palette.count);
        e.palette_hash = hash;
        e.revision += 1;
        e.posed = frame;
        e.settled = false;
        let (cache, count, mesh_vertex) = (e.cache, e.count, e.mesh_vertex);
        self.jobs.push(SkinJob {
            mesh_vertex,
            skin_vertex: skin.offset,
            palette: palette.offset,
            joints: palette.count / 2,
            cache,
            count,
            mode,
            first_group: 0,
        });
        self.updated += 1;
        self.vertices += u64::from(count);
        // The row on every pose: its `pose` is what caches of rendered
        // results (shadow pages) key a skinned caster on.
        let pose = self.entries[&entity].revision as u32;
        self.write_row(
            queue,
            encoder,
            row,
            SkinCacheRow {
                offset: cache.wrapping_sub(mesh_vertex),
                resident: 1,
                pose,
                settled: 0,
            },
        );
        Pose::Posed { moved }
    }

    /// Repack the cache into a smaller buffer once a quarter of it is used
    /// (never below [`INITIAL_VERTICES`]): every resident range moves (its
    /// rows are rewritten and the ray-traced scene follows, `out.moved`).
    fn shrink(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        out: &mut SkinCacheFrame,
    ) {
        if self.capacity <= INITIAL_VERTICES || self.used.saturating_mul(4) >= self.capacity {
            return;
        }
        let capacity = (self.used.max(1) * 2)
            .next_power_of_two()
            .max(INITIAL_VERTICES);
        if capacity >= self.capacity {
            return;
        }
        let packed = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Skin cache"),
            size: u64::from(capacity) * VERTEX_BYTES,
            usage: self.cache_usage,
            mapped_at_creation: false,
        });
        let mut order: Vec<Entity> = self.entries.keys().copied().collect();
        order.sort_by_key(|e| self.entries[e].cache);
        let mut at = 0u32;
        let mut rows = Vec::new();
        for entity in order {
            let e = self.entries.get_mut(&entity).expect("listed");
            encoder.copy_buffer_to_buffer(
                &self.cache,
                u64::from(e.cache) * VERTEX_BYTES,
                &packed,
                u64::from(at) * VERTEX_BYTES,
                u64::from(e.count) * VERTEX_BYTES,
            );
            e.cache = at;
            e.revision += 1;
            rows.push((
                entity.index(),
                SkinCacheRow {
                    offset: at.wrapping_sub(e.mesh_vertex),
                    resident: 1,
                    pose: e.revision as u32,
                    settled: u32::from(e.settled),
                },
            ));
            at += e.count;
            out.moved.push(entity);
        }
        for (row, value) in rows {
            self.write_row(queue, encoder, row, value);
        }
        self.cache = packed;
        self.capacity = capacity;
        self.ranges = Ranges {
            free: (at < capacity)
                .then_some((at, capacity - at))
                .into_iter()
                .collect(),
        };
        self.epoch += 1;
    }

    /// Drop `entity`'s range; whether it held one. Its row is cleared now
    /// (queue writes land in call order: a later instance at the same row
    /// writes over it).
    fn free(&mut self, entity: Entity, queue: &wgpu::Queue) -> bool {
        let Some(e) = self.entries.remove(&entity) else {
            return false;
        };
        self.ranges.give(e.cache, e.count);
        self.used -= e.count;
        let row = entity.index();
        if row < self.rows_capacity {
            queue.write_buffer(
                &self.rows,
                u64::from(row) * ROW_BYTES,
                bytemuck::bytes_of(&SkinCacheRow::default()),
            );
        }
        true
    }

    /// A range of `count` vertices, growing the cache (its contents copied
    /// in `encoder`) when none is free.
    fn allocate(&mut self, count: u32, encoder: &mut wgpu::CommandEncoder) -> Option<u32> {
        if let Some(at) = self.ranges.take(count) {
            self.used += count;
            return Some(at);
        }
        let needed = u64::from(self.used) + u64::from(count);
        let limit = self.device.limits().max_storage_buffer_binding_size / VERTEX_BYTES;
        if needed > limit {
            return None;
        }
        let capacity =
            (needed.next_power_of_two().max(u64::from(self.capacity) * 2)).min(limit) as u32;
        let grown = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Skin cache"),
            size: u64::from(capacity) * VERTEX_BYTES,
            usage: self.cache_usage,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&self.cache, 0, &grown, 0, self.cache.size());
        self.ranges.give(self.capacity, capacity - self.capacity);
        self.capacity = capacity;
        self.cache = grown;
        self.epoch += 1;
        let at = self.ranges.take(count)?;
        self.used += count;
        Some(at)
    }

    /// Write instance table row `row`, growing the table (its rows copied in
    /// `encoder`) to hold it.
    fn write_row(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        row: u32,
        value: SkinCacheRow,
    ) {
        if row >= self.rows_capacity {
            let capacity = (row + 1).next_power_of_two().max(self.rows_capacity * 2);
            let grown = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Skin cache rows"),
                size: u64::from(capacity) * ROW_BYTES,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            encoder.copy_buffer_to_buffer(&self.rows, 0, &grown, 0, self.rows.size());
            self.rows = grown;
            self.rows_capacity = capacity;
            self.epoch += 1;
        }
        queue.write_buffer(
            &self.rows,
            u64::from(row) * ROW_BYTES,
            bytemuck::bytes_of(&value),
        );
    }

    /// Record this frame's jobs.
    fn dispatch(
        &mut self,
        store: &SceneGpuStore,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        if self.jobs.is_empty() {
            return;
        }
        let pool = |key: BufferKey| store.resolve_buffer_handle(key).map(|h| h.buffer);
        let (Some(vertices), Some(skin), Some(palette)) = (
            pool(helio_mesh_api::MESH_VERTICES.key()),
            pool(helio_mesh_api::SKIN_VERTICES.key()),
            pool(helio_mesh_api::BONE_PALETTE.key()),
        ) else {
            return;
        };
        let mut groups = 0u32;
        for job in &mut self.jobs {
            job.first_group = groups;
            groups += job.count.div_ceil(LANES);
        }
        let jobs_bytes = (self.jobs.len() * std::mem::size_of::<SkinJob>()) as u64;
        if self.jobs_buffer.size() < jobs_bytes {
            self.jobs_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Skin cache jobs"),
                size: jobs_bytes.next_power_of_two(),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        queue.write_buffer(&self.jobs_buffer, 0, bytemuck::cast_slice(&self.jobs));
        let groups_x = groups.min(GROUPS_X);
        queue.write_buffer(
            &self.frame_uniform,
            0,
            bytemuck::bytes_of(&SkinFrame {
                jobs: self.jobs.len() as u32,
                groups,
                groups_x,
                _pad: 0,
            }),
        );
        let key = [
            self.jobs_buffer.clone(),
            vertices,
            skin,
            palette,
            self.cache.clone(),
        ];
        if self.bind_group.as_ref().is_none_or(|(k, _)| *k != key) {
            let entries: Vec<_> = std::iter::once(&self.frame_uniform)
                .chain(key.iter())
                .enumerate()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect();
            let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Skin cache"),
                layout: &self.layout,
                entries: &entries,
            });
            self.bind_group = Some((key, group));
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("SkinCache"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group.as_ref().expect("made above").1, &[]);
        pass.dispatch_workgroups(groups_x, groups.div_ceil(GROUPS_X), 1);
        drop(pass);
        self.generation += 1;
    }

    fn publish(&self) {
        let handle = |buffer: &wgpu::Buffer, row_bytes| BufferHandle {
            buffer: buffer.clone(),
            epoch: self.epoch,
            row_bytes,
            content_generation: self.generation,
        };
        *self.published.lock().expect("skin cache buffers") = vec![
            (
                helio_mesh_api::SKIN_CACHE.key(),
                handle(&self.cache, VERTEX_BYTES),
            ),
            (
                helio_mesh_api::SKIN_CACHE_ROWS.key(),
                handle(&self.rows, ROW_BYTES),
            ),
        ];
    }

    /// Bytes of the cache's buffers.
    pub fn bytes(&self) -> u64 {
        self.cache.size() + self.rows.size()
    }

    /// Add this frame's cache stats (`helio_mesh_api::stats`) to `stats`.
    pub fn report(&self, stats: &helio_core::FrameStats) {
        use helio_mesh_api::stats;
        stats.add_named(stats::SKIN_CACHE_BYTES, self.bytes());
        stats.add_named(stats::SKIN_CACHE_INSTANCES, self.entries.len() as u64);
        stats.add_named(stats::SKIN_CACHE_UPDATED, self.updated);
        stats.add_named(stats::SKIN_CACHE_VERTICES, self.vertices);
    }

    /// Instances posed this frame, and vertices the pass wrote.
    pub fn frame_counts(&self) -> (u64, u64) {
        (self.updated, self.vertices)
    }

    /// Vertices the cache holds room for, and how many are in use.
    pub fn capacity(&self) -> (u32, u32) {
        (self.capacity, self.used)
    }
}

/// `entity`'s world bounding sphere: its mesh's local sphere through its
/// owner's world matrix (`None` without a mesh; its local sphere without an
/// owner).
fn world_sphere(world: &World, entity: Entity) -> Option<(glam::DVec3, f64)> {
    let mesh = world.get::<SkeletalMeshComponent>(entity)?;
    let [x, y, z, r] = mesh.bounds_local.map(f64::from);
    let matrix = world
        .get::<pulsar_scene_model::attachments::ComponentOwner>(entity)
        .and_then(|link| {
            pulsar_scene_model::hierarchy_transform::world_matrix_of(world, link.entity())
        })
        .unwrap_or(glam::DMat4::IDENTITY);
    let scale = matrix
        .x_axis
        .truncate()
        .length()
        .max(matrix.y_axis.truncate().length())
        .max(matrix.z_axis.truncate().length());
    Some((
        matrix.transform_point3(glam::DVec3::new(x, y, z)),
        r * scale,
    ))
}

enum Pose {
    Unchanged,
    Posed { moved: bool },
    Waiting,
    Gone,
}

/// Publishes the skin cache's buffers into each frame's scene buffers, so
/// the drawers resolve [`helio_mesh_api::SKIN_CACHE`] and
/// [`helio_mesh_api::SKIN_CACHE_ROWS`] by key. The cache itself is updated
/// before the frame (the meshes' host part), with the ray-traced scene that
/// refits from it.
pub struct SkinCacheBuffers(Published);

impl helio_core::SceneDerivation for SkinCacheBuffers {
    fn name(&self) -> &'static str {
        "meshes.skin_cache"
    }

    fn derive(
        &mut self,
        _ctx: &helio_core::SceneDerivationContext<'_>,
        _encoder: &mut wgpu::CommandEncoder,
    ) -> helio_core::SceneDerivationOutput {
        helio_core::SceneDerivationOutput {
            buffers: self.0.lock().expect("skin cache buffers").clone(),
            flushed: Vec::new(),
            recorded: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ranges are taken first fit and merge with their neighbours when given
    /// back.
    #[test]
    fn ranges_take_first_fit_and_merge() {
        let mut r = Ranges {
            free: vec![(0, 100)],
        };
        assert_eq!(r.take(30), Some(0));
        assert_eq!(r.take(30), Some(30));
        assert_eq!(r.take(50), None);
        r.give(0, 30);
        assert_eq!(r.free, vec![(0, 30), (60, 40)]);
        r.give(30, 30);
        assert_eq!(r.free, vec![(0, 100)]);
    }

    /// The pass's rows are the WGSL's.
    #[test]
    fn the_skin_pass_validates() {
        let source = helio_core::shader::resolve_with(
            include_str!("skin_cache.wgsl"),
            &[
                helio_mesh_api::DRAW_ROWS_SNIPPET,
                helio_mesh_api::SKINNING_SNIPPET,
                helio_frame_api::rays::stream::SNIPPET,
            ],
        );
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{e:?}"));
        let size = |name: &str| {
            module
                .types
                .iter()
                .find_map(|(_, ty)| match ty.inner {
                    naga::TypeInner::Struct { span, .. } if ty.name.as_deref() == Some(name) => {
                        Some(span as usize)
                    }
                    _ => None,
                })
                .unwrap()
        };
        assert_eq!(size("SkinJob"), std::mem::size_of::<SkinJob>());
        assert_eq!(size("SkinFrame"), std::mem::size_of::<SkinFrame>());
        assert_eq!(
            size("SkinVertex"),
            std::mem::size_of::<helio_mesh_api::SkinVertex>()
        );
        assert_eq!(helio_mesh_api::PackedVertex::STRIDE, 32);
    }
}
