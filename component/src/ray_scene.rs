//! The engine world's meshes as the frame's ray-traced scene
//! (`helio_frame_api::rays::SCENE_RAYS`): the TLAS, plus a `RayInstance`
//! surface per traced section (`ray_instances`), that the GI probes' mesh
//! tracer and ray-query shadows read.
//!
//! `helio_mesh_host::ray_tracing` projects Helio's standalone rows
//! (`StaticObjectComponent`/`MeshComponent`), which an engine world does not
//! hold: here a mesh is a [`StaticMeshComponent`] on a component entity, placed
//! by its owner object's transform hierarchy, its sections drawing with the
//! material instances of its [`MeshMaterials`]. This is that world's
//! projection.
//!
//! - **One TLAS instance per mesh, one BLAS per loaded mesh.** A mesh's
//!   traced sections are the segments of its BLAS (each with its own
//!   opacity), shared by every instance of the mesh; a hit's surface row is
//!   its instance's custom data (the mesh's first row) plus the hit's
//!   `geometry_index`. Virtual-geometry meshes are traced through their
//!   assets' coarse proxies ([`helio_mesh_host::ray_proxies`]), one segment
//!   drawn with the mesh's largest section's material.
//! - **Incremental.** The world's change journals say what changed: a mesh,
//!   its materials or its enabled state re-evaluates that mesh; a transform
//!   or parent change re-places the meshes below that entity (an index of
//!   each entity's meshes, from their ancestor chains); a material change
//!   re-evaluates its users. A frame with no changes does no per-mesh work.
//!   The TLAS rewrites only the instances that changed
//!   (`TlasManager::build_masked`); the surface rows are rewritten only when
//!   the traced set or a surface changed, not when something moves. An
//!   origin shift re-places every instance from its cached world matrix.
//! - **Spread out.** New BLASes are built under a per-frame triangle budget
//!   (a mesh joins the TLAS once its BLAS is built), and mesh evaluations
//!   under a per-frame count, so a level loading does not stall one frame on
//!   thousands of builds. Built BLASes are compacted a few a frame
//!   (`BlasManager::maintain`): a level's BLASes shrink to their final size
//!   over the frames after it loads.
//!
//! - **Skinned meshes** ([`SkeletalMeshComponent`]) are traced from the skin
//!   cache ([`crate::skin_cache`]), the positions every drawer draws: a BLAS
//!   per instance (they deform independently), `Movability::Dynamic`, refit
//!   when the cache posed it (`positions_revision` is the cache's revision),
//!   rebuilt on the acceleration policy's schedule, nothing when its pose
//!   is unchanged. Refits are budgeted in triangles a frame
//!   (`helio_mesh_api::RAY_TRACING`'s `skinned_refit_triangles`), nearest
//!   first and the longest-waiting ahead of nearer ones; an instance over the
//!   budget keeps last frame's BLAS (a frame stale, not wrong). Their rows
//!   (`RAY_INSTANCE_SKINNED`) point the tracer at the cache, so a hit shades
//!   from the skinned normal.
//! - **Culled by distance and angle** (`RAY_TRACING`'s `culling_radius` and
//!   `culling_angle`, Unreal's ray-tracing culling): an instance past the
//!   radius whose bounding sphere spans less than the angle from the eye
//!   leaves the TLAS (with a tenth of slack, so one at a threshold does not
//!   flicker). The eye's moves re-test the traced instances only after it
//!   has moved a twentieth of the radius.
//!
//! What it traces: every enabled, visible static or skinned mesh section
//! whose material is opaque or a cut-out, within the culling criterion. Cut-outs are non-opaque segments flagged
//! `RAY_INSTANCE_MASKED`, so a ray query tests their alpha as the raster
//! does. Blended sections (glass) are left out: probe rays pass them. A
//! material graph is traced as its resolved row (its fallback surface and
//! its first texture's mean colour): rays cannot run a graph, so a hit's
//! colour is that coarse surface, as with every material
//! (`RayInstance::base_color`).
//!
//! How others do it: Unreal's ray-tracing scene keeps one instance per
//! primitive with a BLAS per mesh LOD whose segments are its sections (hit
//! groups per segment), updated from primitive changes, BLAS builds
//! budgeted and compacted, and Nanite meshes traced through fallback
//! meshes; Lumen's hit lighting uses a surface cache instead of the material.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use glam::{DMat4, DVec3};
use helio_core::{BlasGeometry, BlasManager, BlasSegment, TlasInstanceInput, TlasManager};
use helio_frame_api::rays::stream::{MASK_MOVABLE, MASK_STATIC};
use helio_mats::instance::{MaterialInstanceComponent, row_of};
use helio_mesh_api::{
    RAY_INSTANCE_MASKED, RAY_INSTANCE_PROXY, RAY_INSTANCE_SKINNED, RAY_INSTANCE_TWO_SIDED,
    RayInstance, RayTracingSettings,
};
use helio_mesh_host::ray_proxies::{Lookup, ProxyRange, VgRayProxies};
use helio_mesh_host::texture_means::TextureMeans;
use pulsar_scene_model::attachments::ComponentOwner;
use pulsar_scene_model::{Parent, Transform, Visibility};
use pulsar_scenedb::gpu::{GpuMirrorHandle, SceneGpuStore, VarLenHandle};
use pulsar_scenedb::{ChangeCursor, ChangeRead, ComponentChange, Entity, World};

use crate::components::{
    MeshGeometryKey, SkeletalMeshComponent, StaticMeshComponent, StaticMeshDraw,
};
use crate::mesh_materials::MeshMaterials;
use crate::skin_cache::{SkinCache, SkinCacheFrame};

/// Triangles of new BLASes built a frame (at least one BLAS is).
const BUILD_TRIANGLES_PER_FRAME: u64 = 1 << 20;
/// Meshes (re-)evaluated a frame.
const EVALUATIONS_PER_FRAME: usize = 256;
/// Ancestors followed from a mesh's owner (as the transform hierarchy).
const MAX_DEPTH: usize = 64;
/// How far past the culling thresholds a traced instance stays traced.
const CULLING_SLACK: f64 = 1.1;
/// The share of the culling radius the eye moves before the traced
/// instances are tested again.
const CULLING_RETEST: f64 = 0.05;

/// A BLAS: what geometry, and each segment's opacity.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum BlasKey {
    /// A loaded mesh's traced sections: first index (mesh-relative), index
    /// count and opacity each.
    Mesh {
        geometry: u128,
        sections: Vec<(u32, u32, bool)>,
    },
    /// A virtual-geometry asset's proxy.
    Proxy { key: u64, opaque: bool },
    /// A skinned instance's own deforming BLAS (its entity's bits), its
    /// mesh and traced sections as a mesh's.
    Skinned {
        entity: u64,
        geometry: u128,
        sections: Vec<(u32, u32, bool)>,
    },
}

/// Where a BLAS's triangles are, for its build.
#[derive(Clone, Copy, Debug)]
enum Source {
    /// A mesh's ranges of the mirror's vertex and index pools.
    Mesh {
        vertices: VarLenHandle,
        first_index: u32,
    },
    Proxy(ProxyRange),
    /// A skinned instance's vertices in the skin cache (by its entity), its
    /// indices in the mesh pool from `first_index`.
    Skinned {
        first_index: u32,
    },
}

struct Blas {
    id: u64,
    /// Meshes tracing it.
    users: u32,
    state: BlasState,
    source: Source,
    triangles: u64,
    /// A skinned BLAS: the skin cache revision it was last built from, and
    /// the frames its refit has waited.
    revision: u64,
    waited: u32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BlasState {
    Queued,
    Built,
    Failed,
}

/// What a ray sees of a material instance.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Surface {
    /// Blended (glass): not traced.
    blended: bool,
    /// `RAY_INSTANCE_MASKED` and `RAY_INSTANCE_TWO_SIDED` bits.
    flags: u32,
    base_color: [f32; 4],
    emissive: [f32; 4],
}

/// Why a mesh is not traced. The first four are retried every frame (they
/// clear without a journaled change of the mesh); the others wait for one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Skip {
    Unloaded,
    Unmirrored,
    NoMaterials,
    ProxyPending,
    Disabled,
    Hidden,
    Glass,
}

impl Skip {
    fn retried(self) -> bool {
        matches!(
            self,
            Self::Unloaded | Self::Unmirrored | Self::NoMaterials | Self::ProxyPending
        )
    }

    fn label(self) -> &'static str {
        match self {
            Self::Unloaded => "not loaded",
            Self::Unmirrored => "not on the GPU yet",
            Self::NoMaterials => "without materials",
            Self::ProxyPending => "proxy building",
            Self::Disabled => "disabled",
            Self::Hidden => "hidden",
            Self::Glass => "only glass",
        }
    }
}

/// A traced mesh.
struct Traced {
    key: BlasKey,
    /// Its rows, a segment each, and the material instance of each.
    rows: Vec<RayInstance>,
    materials: Vec<Entity>,
    world: DMat4,
    movable: bool,
    proxy: Option<u64>,
    /// Its TLAS instance, once its BLAS is built.
    slot: Option<usize>,
    /// Its BLAS's triangles and their count, for the build.
    source_hint: (Source, u64),
    /// Its world bounding sphere (centre, radius), for the culling.
    center: DVec3,
    radius: f64,
    /// Its mesh-space bounding sphere.
    bounds_local: [f32; 4],
    /// Left out by the culling.
    culled: bool,
    skinned: bool,
}

/// A mesh the projection knows: its owner's ancestor chain (owner first),
/// and what it traces or why it does not.
struct Entry {
    chain: Vec<Entity>,
    state: Result<Traced, Skip>,
}

/// The journals the projection follows.
struct Cursors {
    meshes: ChangeCursor,
    skeletal: ChangeCursor,
    mesh_materials: ChangeCursor,
    owners: ChangeCursor,
    materials: ChangeCursor,
    visibility: ChangeCursor,
    parents: ChangeCursor,
    transforms: ChangeCursor,
}

impl Cursors {
    fn open(world: &World) -> Self {
        Self {
            meshes: world.open_change_cursor::<StaticMeshComponent>(),
            skeletal: world.open_change_cursor::<SkeletalMeshComponent>(),
            mesh_materials: world.open_change_cursor::<MeshMaterials>(),
            owners: world.open_change_cursor::<ComponentOwner>(),
            materials: world.open_change_cursor::<MaterialInstanceComponent>(),
            visibility: world.open_change_cursor::<Visibility>(),
            parents: world.open_change_cursor::<Parent>(),
            transforms: world.open_change_cursor::<Transform>(),
        }
    }
}

/// What the projection holds, for its log line.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Stats {
    meshes: usize,
    traced: usize,
    rows: usize,
    blases: usize,
    queued: usize,
    failed: usize,
    skipped: Vec<(&'static str, usize)>,
}

/// The ray-traced scene of an engine world's meshes; see the module doc.
pub struct WorldRayScene {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    blas: BlasManager,
    tlas: TlasManager,
    blases: HashMap<BlasKey, Blas>,
    build_queue: VecDeque<BlasKey>,
    next_id: u64,
    entries: HashMap<Entity, Entry>,
    /// Each entity's meshes: the meshes whose chain holds it.
    below: HashMap<Entity, HashSet<Entity>>,
    /// Meshes whose skip clears without a journaled change.
    waiting: HashSet<Entity>,
    /// Traced meshes whose BLAS is not built yet.
    awaiting_blas: HashSet<Entity>,
    /// TLAS order: the mesh of each instance, its input and shadow mask.
    slots: Vec<Entity>,
    instances: Vec<TlasInstanceInput>,
    masks: Vec<u8>,
    rows: Vec<RayInstance>,
    rows_dirty: bool,
    instances_buffer: Option<wgpu::Buffer>,
    origin: DVec3,
    cursors: Option<Cursors>,
    changes: Vec<ComponentChange>,
    /// Material instances' surfaces: their rows with the texture means
    /// applied; cleared when means arrive.
    surfaces: HashMap<Entity, Option<Surface>>,
    /// Meshes to re-evaluate past last frame's budget.
    pending: HashSet<Entity>,
    /// Base colour textures' means: a hit's colour is its material's factor
    /// times its texture's mean.
    texture_means: TextureMeans,
    vg: Option<VgRayProxies>,
    tlas_ok: bool,
    logged: Option<Stats>,
    warned: bool,
    /// Traced skinned instances the culling keeps.
    skinned: HashSet<Entity>,
    /// The skin cache's vertices, published beside the TLAS.
    skinned_vertices: Option<wgpu::Buffer>,
    settings: RayTracingSettings,
    /// The eye this frame, and where the traced instances were last tested.
    eye: DVec3,
    tested_at: Option<DVec3>,
    culled: usize,
}

impl WorldRayScene {
    /// `None` without hardware ray queries on `device`. `virtual_geometry`:
    /// the runtime whose assets are traced through proxies.
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        virtual_geometry: Option<helio_vg_stream::VgRuntime>,
    ) -> Option<Self> {
        let tlas = TlasManager::new(device.clone(), 1);
        if !tlas.is_rt_available() {
            return None;
        }
        Some(Self {
            texture_means: TextureMeans::new(&device),
            blas: BlasManager::new(device.clone()),
            tlas,
            device,
            queue,
            blases: HashMap::new(),
            build_queue: VecDeque::new(),
            next_id: 0,
            entries: HashMap::new(),
            below: HashMap::new(),
            waiting: HashSet::new(),
            awaiting_blas: HashSet::new(),
            slots: Vec::new(),
            instances: Vec::new(),
            masks: Vec::new(),
            rows: Vec::new(),
            rows_dirty: true,
            instances_buffer: None,
            origin: DVec3::ZERO,
            cursors: None,
            changes: Vec::new(),
            surfaces: HashMap::new(),
            pending: HashSet::new(),
            vg: virtual_geometry.map(VgRayProxies::new),
            tlas_ok: false,
            logged: None,
            warned: false,
            skinned: HashSet::new(),
            skinned_vertices: None,
            settings: RayTracingSettings::default(),
            eye: DVec3::ZERO,
            tested_at: None,
            culled: 0,
        })
    }

    /// Bring the TLAS up to date with `world`, in the frame whose world
    /// origin is `origin` and whose eye is `eye` (world space), recording
    /// its builds into `encoder` after the skin cache's pass (`skins`, which
    /// did `skin_frame` this frame). Call after the world's GPU mirror is
    /// flushed (the BLASes read the mirrored geometry); submit `encoder`,
    /// then call [`Self::after_submit`]. Adds the acceleration structures'
    /// stats to `stats`.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        &mut self,
        world: &World,
        origin: DVec3,
        eye: DVec3,
        settings: RayTracingSettings,
        skins: &SkinCache,
        skin_frame: &SkinCacheFrame,
        encoder: &mut wgpu::CommandEncoder,
        stats: &helio_core::FrameStats,
    ) {
        let Some(mirror) = world.gpu_mirror() else {
            return;
        };
        let store = mirror.store();
        // Compact the BLASes built in earlier frames (a few a frame).
        self.blas.maintain(&self.queue);
        // Proxies that came or went: re-evaluate everything.
        if self
            .vg
            .as_mut()
            .is_some_and(|vg| vg.poll(&self.device, &self.queue))
        {
            self.cursors = None;
        }
        // A new criterion re-tests every traced instance.
        if settings != self.settings {
            self.settings = settings;
            self.tested_at = None;
        }
        self.eye = eye;
        self.skinned_vertices = Some(skins.buffer().clone());
        let mut dirty = std::mem::take(&mut self.pending);
        let mut moved = HashSet::new();
        self.read_changes(world, &mut dirty, &mut moved);
        dirty.extend(self.waiting.drain());
        // Skinned instances the cache moved or dropped: their rows follow.
        dirty.extend(skin_frame.moved.iter().copied());
        // Texture means that arrived recolour the rows (from the cached
        // material rows: nothing is resolved again).
        if self.texture_means.collect() {
            self.surfaces.clear();
            self.recolour(world, mirror);
        }
        // A budget of meshes a frame; the rest wait for the next frames (a
        // level loading spreads over them).
        let mut evaluated = 0;
        for entity in dirty {
            if evaluated >= EVALUATIONS_PER_FRAME {
                self.pending.insert(entity);
                continue;
            }
            evaluated += 1;
            moved.remove(&entity);
            self.evaluate(world, mirror, store, skins, entity);
        }
        for entity in moved {
            if !self.pending.contains(&entity) {
                self.place(world, entity);
            }
        }
        // The eye moved far enough: test every traced instance again.
        let retest = (self.settings.culling_radius * CULLING_RETEST).max(1.0);
        if self.tested_at.is_none_or(|at| at.distance(eye) > retest) {
            self.tested_at = Some(eye);
            let traced: Vec<Entity> = self
                .entries
                .iter()
                .filter(|(_, e)| e.state.is_ok())
                .map(|(entity, _)| *entity)
                .collect();
            for entity in traced {
                self.cull(entity);
            }
        }
        if origin != self.origin {
            self.origin = origin;
            for (slot, entity) in self.slots.iter().enumerate() {
                if let Some(Ok(traced)) = self.entries.get(entity).map(|e| &e.state) {
                    self.instances[slot].transform = tlas_transform(traced.world, origin);
                }
            }
        }
        self.build_blases(store, encoder);
        self.refit_skinned(store, skins, encoder);
        self.admit_built();
        if self.rows_dirty {
            self.write_rows();
        }
        match self
            .tlas
            .build_masked(encoder, &self.instances, &self.blas, &self.masks)
        {
            Ok(()) => {
                self.tlas_ok = true;
                self.warned = false;
            }
            Err(error) => {
                self.tlas_ok = false;
                if !self.warned {
                    self.warned = true;
                    tracing::warn!(
                        "Meshes: the ray-traced scene's TLAS failed ({error}); rays see no meshes"
                    );
                }
            }
        }
        self.blas.report(stats);
        self.tlas.report(stats);
        stats.add_named(
            helio_mesh_api::stats::RAY_INSTANCES,
            self.slots.len() as u64,
        );
        stats.add_named(
            helio_mesh_api::stats::RAY_INSTANCES_CULLED,
            self.culled as u64,
        );
        self.log_stats();
    }

    /// After the encoder [`Self::prepare`] recorded into is submitted.
    pub fn after_submit(&mut self) {
        self.texture_means.dispatch(&self.device, &self.queue);
    }

    /// The BLAS manager's counters (builds, refits, compactions).
    pub fn blas_counters(&self) -> helio_core::BlasCounters {
        self.blas.counters()
    }

    /// The TLAS manager's counters.
    pub fn tlas_counters(&self) -> helio_core::TlasCounters {
        self.tlas.counters()
    }

    /// How many instances the culling leaves out.
    pub fn culled_count(&self) -> usize {
        self.culled
    }

    /// The frame's ray-traced scene, after [`Self::prepare`].
    pub fn scene_rays(&self) -> Option<helio_frame_api::rays::SceneRays> {
        if !self.tlas_ok {
            return None;
        }
        let mut rays = helio_frame_api::rays::SceneRays::new(self.tlas.tlas()?.clone());
        if let Some(instances) = &self.instances_buffer {
            rays = rays.with(helio_mesh_api::RAY_INSTANCES, instances.clone());
        }
        if let Some((vertices, indices)) = self.vg.as_ref().and_then(VgRayProxies::buffers) {
            rays = rays
                .with(helio_mesh_api::RAY_PROXY_VERTICES, vertices.clone())
                .with(helio_mesh_api::RAY_PROXY_INDICES, indices.clone());
        }
        if let Some(skinned) = self
            .skinned_vertices
            .as_ref()
            .filter(|_| !self.skinned.is_empty())
        {
            rays = rays.with(helio_mesh_api::RAY_SKINNED_VERTICES, skinned.clone());
        }
        Some(rays)
    }

    /// How many meshes the TLAS holds.
    pub fn instance_count(&self) -> usize {
        self.slots.len()
    }

    /// The journals' changes since last frame, as meshes to re-evaluate
    /// (`dirty`) and to re-place (`moved`); everything after an overflow.
    fn read_changes(
        &mut self,
        world: &World,
        dirty: &mut HashSet<Entity>,
        moved: &mut HashSet<Entity>,
    ) {
        let complete = match self.cursors.as_mut() {
            None => false,
            Some(cursors) => {
                let mut complete = true;
                let changes = &mut self.changes;
                let mut read = |cursor: &mut ChangeCursor| {
                    changes.clear();
                    complete &= world.read_changes(cursor, changes) == ChangeRead::Complete;
                    changes.iter().map(|c| c.entity).collect::<Vec<_>>()
                };
                let own = [
                    read(&mut cursors.meshes),
                    read(&mut cursors.mesh_materials),
                    read(&mut cursors.owners),
                ];
                // A skinned mesh's pose changes reach it through the skin
                // cache; its journal still says when it was added, removed
                // or re-pointed at another mesh (re-evaluated: cheap).
                let skeletal = read(&mut cursors.skeletal);
                let materials = read(&mut cursors.materials);
                let placed = [read(&mut cursors.visibility), read(&mut cursors.parents)];
                let transforms = read(&mut cursors.transforms);
                if complete {
                    dirty.extend(own.into_iter().flatten());
                    for entity in skeletal {
                        // A pose alone changes nothing here: only a new mesh
                        // (or one not traced yet) re-evaluates.
                        let geometry = world
                            .get::<SkeletalMeshComponent>(entity)
                            .map(|mesh| mesh.geometry_key.0);
                        let traced = match self.entries.get(&entity).map(|e| &e.state) {
                            Some(Ok(Traced {
                                key: BlasKey::Skinned { geometry: g, .. },
                                ..
                            })) => Some(*g),
                            _ => None,
                        };
                        if geometry.is_none() || traced != geometry {
                            dirty.insert(entity);
                        }
                    }
                    for entity in placed.into_iter().flatten() {
                        dirty.extend(self.below.get(&entity).into_iter().flatten());
                    }
                    for entity in transforms {
                        moved.extend(self.below.get(&entity).into_iter().flatten());
                    }
                    if !materials.is_empty() {
                        let changed: HashSet<_> = materials.into_iter().collect();
                        for entity in &changed {
                            self.surfaces.remove(entity);
                        }
                        for (mesh, entry) in &self.entries {
                            if let Ok(traced) = &entry.state {
                                if traced.materials.iter().any(|m| changed.contains(m)) {
                                    dirty.insert(*mesh);
                                }
                            }
                        }
                    }
                }
                complete
            }
        };
        if !complete {
            // A first frame or a journal overflow: every mesh, from a scan.
            self.cursors = Some(Cursors::open(world));
            self.surfaces.clear();
            dirty.extend(self.entries.keys().copied());
            dirty.extend(
                world
                    .query::<&StaticMeshComponent>()
                    .map(|(entity, _)| entity),
            );
            dirty.extend(
                world
                    .query::<&SkeletalMeshComponent>()
                    .map(|(entity, _)| entity),
            );
        }
    }

    /// Re-evaluate the mesh `entity` from the world: drop what it traced,
    /// trace what it now draws.
    fn evaluate(
        &mut self,
        world: &World,
        mirror: &GpuMirrorHandle,
        store: &SceneGpuStore,
        skins: &SkinCache,
        entity: Entity,
    ) {
        if let Some(old) = self.entries.remove(&entity) {
            self.forget(entity, old);
        }
        let Some(mut entry) = self.entry(world, mirror, store, skins, entity) else {
            return;
        };
        if let Ok(traced) = &mut entry.state {
            traced.culled =
                !self
                    .settings
                    .traces(traced.center.distance(self.eye), traced.radius, 1.0);
            if traced.culled {
                self.culled += 1;
            }
        }
        for ancestor in &entry.chain {
            self.below.entry(*ancestor).or_default().insert(entity);
        }
        match &entry.state {
            Err(skip) if skip.retried() => {
                self.waiting.insert(entity);
            }
            Err(_) => {}
            Ok(traced) => {
                let source = traced.source_hint;
                let skinned = traced.skinned;
                let blas = self.blases.entry(traced.key.clone()).or_insert_with(|| {
                    let id = self.next_id;
                    self.next_id += 1;
                    // Skinned BLASes are built by the refit budget.
                    if !skinned {
                        self.build_queue.push_back(traced.key.clone());
                    }
                    Blas {
                        id,
                        users: 0,
                        state: BlasState::Queued,
                        source: source.0,
                        triangles: source.1,
                        revision: 0,
                        waited: 0,
                    }
                });
                blas.users += 1;
                if !traced.culled {
                    self.awaiting_blas.insert(entity);
                    if skinned {
                        self.skinned.insert(entity);
                    }
                }
            }
        }
        self.entries.insert(entity, entry);
    }

    /// Drop what `entity` traced.
    fn forget(&mut self, entity: Entity, old: Entry) {
        for ancestor in &old.chain {
            if let Some(meshes) = self.below.get_mut(ancestor) {
                meshes.remove(&entity);
                if meshes.is_empty() {
                    self.below.remove(ancestor);
                }
            }
        }
        self.waiting.remove(&entity);
        self.awaiting_blas.remove(&entity);
        self.skinned.remove(&entity);
        let Ok(traced) = old.state else {
            return;
        };
        if traced.culled {
            self.culled -= 1;
        }
        if let Some(slot) = traced.slot {
            self.leave_slot(slot);
        }
        if let Some(blas) = self.blases.get_mut(&traced.key) {
            blas.users -= 1;
            if blas.users == 0 {
                self.blas.remove_blas(blas.id);
                self.blases.remove(&traced.key);
            }
        }
    }

    /// Take TLAS slot `slot`'s instance out (the last slot's moves in).
    fn leave_slot(&mut self, slot: usize) {
        self.slots.swap_remove(slot);
        self.instances.swap_remove(slot);
        self.masks.swap_remove(slot);
        if let Some(&moved) = self.slots.get(slot) {
            if let Some(Ok(other)) = self.entries.get_mut(&moved).map(|e| &mut e.state) {
                other.slot = Some(slot);
            }
        }
        self.rows_dirty = true;
    }

    /// Test the traced instance `entity` against the culling criterion at
    /// this frame's eye: it leaves the TLAS when culled (a skinned one's own
    /// BLAS is dropped), and waits for its BLAS to join again when not.
    fn cull(&mut self, entity: Entity) {
        let eye = self.eye;
        let settings = self.settings;
        let Some(Ok(traced)) = self.entries.get_mut(&entity).map(|e| &mut e.state) else {
            return;
        };
        let slack = if traced.culled { 1.0 } else { CULLING_SLACK };
        let culled = !settings.traces(traced.center.distance(eye), traced.radius, slack);
        if culled == traced.culled {
            return;
        }
        traced.culled = culled;
        let (slot, skinned, key) = (traced.slot.take(), traced.skinned, traced.key.clone());
        if culled {
            self.culled += 1;
            if let Some(slot) = slot {
                self.leave_slot(slot);
            }
            self.awaiting_blas.remove(&entity);
            self.skinned.remove(&entity);
            if skinned {
                if let Some(blas) = self.blases.get_mut(&key) {
                    self.blas.remove_blas(blas.id);
                    blas.state = BlasState::Queued;
                }
            }
        } else {
            self.culled -= 1;
            self.awaiting_blas.insert(entity);
            if skinned {
                self.skinned.insert(entity);
            }
        }
    }

    /// What the mesh `entity` traces now (`None`: it is no mesh).
    fn entry(
        &mut self,
        world: &World,
        mirror: &GpuMirrorHandle,
        store: &SceneGpuStore,
        skins: &SkinCache,
        entity: Entity,
    ) -> Option<Entry> {
        let link = world.get::<ComponentOwner>(entity)?;
        let owner = link.entity();
        let state = if let Some(mesh) = world.get::<StaticMeshComponent>(entity) {
            self.trace(world, mirror, store, entity, mesh, link, owner)
        } else {
            let mesh = world.get::<SkeletalMeshComponent>(entity)?;
            self.trace_skinned(world, mirror, store, skins, entity, mesh, link, owner)
        };
        let chain = chain_of(world, owner);
        Some(Entry { chain, state })
    }

    /// Where `owner` places a mesh, if it draws: enabled, alive, visible.
    fn placed(world: &World, link: &ComponentOwner, owner: Entity) -> Result<DMat4, Skip> {
        if !link.is_enabled() || !world.is_alive(owner) {
            return Err(Skip::Disabled);
        }
        if world
            .get::<Visibility>(owner)
            .is_some_and(|visibility| !visibility.visible)
        {
            return Err(Skip::Hidden);
        }
        pulsar_scene_model::hierarchy_transform::world_matrix_of(world, owner).ok_or(Skip::Disabled)
    }

    /// What the skinned mesh `entity` traces now: its traced sections, from
    /// the skin cache (every section draws with the mesh's first material,
    /// as its draw rows do).
    #[allow(clippy::too_many_arguments)]
    fn trace_skinned(
        &mut self,
        world: &World,
        mirror: &GpuMirrorHandle,
        store: &SceneGpuStore,
        skins: &SkinCache,
        entity: Entity,
        mesh: &SkeletalMeshComponent,
        link: &ComponentOwner,
        owner: Entity,
    ) -> Result<Traced, Skip> {
        let world_matrix = Self::placed(world, link, owner)?;
        if mesh.geometry_key == MeshGeometryKey::NONE || mesh.indices.is_empty() {
            return Err(Skip::Unloaded);
        }
        let Some(indices) = SkeletalMeshComponent::indices_gpu_handle(store, entity.index()) else {
            return Err(Skip::Unmirrored);
        };
        if indices.count as usize != mesh.indices.len() {
            return Err(Skip::Unmirrored);
        }
        let Some(skin) = skins.instance(entity) else {
            return Err(Skip::Unmirrored);
        };
        let Some(instances) = world.get::<MeshMaterials>(entity) else {
            return Err(Skip::NoMaterials);
        };
        let material = *instances.0.first().ok_or(Skip::NoMaterials)?;
        let surface = self
            .surface(world, mirror, material)
            .ok_or(Skip::NoMaterials)?;
        if surface.blended {
            return Err(Skip::Glass);
        }
        let sections: Vec<_> = mesh
            .mesh_sections
            .iter()
            .filter(|s| {
                s.index_count > 0
                    && u64::from(s.first_index) + u64::from(s.index_count)
                        <= u64::from(indices.count)
            })
            .collect();
        if sections.is_empty() {
            return Err(Skip::Unloaded);
        }
        let opaque = surface.flags & RAY_INSTANCE_MASKED == 0;
        let (center, radius) = world_sphere(world_matrix, mesh.bounds_local);
        Ok(Traced {
            key: BlasKey::Skinned {
                entity: entity_bits(entity),
                geometry: mesh.geometry_key.0,
                sections: sections
                    .iter()
                    .map(|s| (s.first_index, s.index_count, opaque))
                    .collect(),
            },
            rows: sections
                .iter()
                .map(|s| RayInstance {
                    first_index: indices.offset + s.first_index,
                    base_vertex: skin.mesh_vertex,
                    flags: RAY_INSTANCE_SKINNED | surface.flags,
                    material: row_of(material),
                    base_color: surface.base_color,
                    emissive: surface.emissive,
                    transmission: [0.0; 3],
                    skinned_vertex: skin.cache_vertex,
                })
                .collect(),
            materials: vec![material; sections.len()],
            world: world_matrix,
            movable: true,
            proxy: None,
            slot: None,
            source_hint: (
                Source::Skinned {
                    first_index: indices.offset,
                },
                sections.iter().map(|s| u64::from(s.index_count / 3)).sum(),
            ),
            center,
            radius,
            bounds_local: mesh.bounds_local,
            culled: false,
            skinned: true,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn trace(
        &mut self,
        world: &World,
        mirror: &GpuMirrorHandle,
        store: &SceneGpuStore,
        entity: Entity,
        mesh: &StaticMeshComponent,
        link: &ComponentOwner,
        owner: Entity,
    ) -> Result<Traced, Skip> {
        let world_matrix = Self::placed(world, link, owner)?;
        if mesh.geometry_key == MeshGeometryKey::NONE || mesh.indices.is_empty() {
            return Err(Skip::Unloaded);
        }
        let (Some(vertices), Some(indices)) = (
            StaticMeshComponent::vertices_gpu_handle(store, entity.index()),
            StaticMeshComponent::indices_gpu_handle(store, entity.index()),
        ) else {
            return Err(Skip::Unmirrored);
        };
        if vertices.count as usize != mesh.vertices.len()
            || indices.count as usize != mesh.indices.len()
        {
            return Err(Skip::Unmirrored);
        }
        let Some(instances) = world.get::<MeshMaterials>(entity) else {
            return Err(Skip::NoMaterials);
        };
        // The traced sections: each with its material instance and surface.
        let mut sections = Vec::new();
        for section in StaticMeshDraw::of(mesh).sections {
            if section.index_count == 0
                || section.first_index as u64 + section.index_count as u64 > indices.count as u64
            {
                continue;
            }
            let material = *instances
                .0
                .get(section.material_slot as usize)
                .ok_or(Skip::NoMaterials)?;
            let surface = self
                .surface(world, mirror, material)
                .ok_or(Skip::NoMaterials)?;
            if !surface.blended {
                sections.push((section, material, surface));
            }
        }
        if sections.is_empty() {
            return Err(Skip::Glass);
        }
        let row =
            |first_index, base_vertex, flags, material: Entity, surface: &Surface| RayInstance {
                first_index,
                base_vertex,
                flags: flags | surface.flags,
                material: row_of(material),
                base_color: surface.base_color,
                emissive: surface.emissive,
                transmission: [0.0; 3],
                skinned_vertex: 0,
            };
        let lookup = match self.vg.as_mut() {
            Some(vg) => vg.lookup((vertices.offset, vertices.count)),
            None => Lookup::Classic,
        };
        let movable = helio::Movability::from(mesh.movability).can_move();
        let (key, rows, materials, proxy, source_hint) = match lookup {
            Lookup::Pending => return Err(Skip::ProxyPending),
            Lookup::Proxy(key, range) => {
                // One segment: the proxy, drawn as the largest section.
                let (_, material, surface) = sections
                    .iter()
                    .max_by_key(|(section, _, _)| section.index_count)
                    .expect("a traced section");
                (
                    BlasKey::Proxy {
                        key,
                        opaque: surface.flags & RAY_INSTANCE_MASKED == 0,
                    },
                    vec![row(
                        range.first_index,
                        range.first_vertex,
                        RAY_INSTANCE_PROXY,
                        *material,
                        surface,
                    )],
                    vec![*material],
                    Some(key),
                    (Source::Proxy(range), u64::from(range.index_count / 3)),
                )
            }
            Lookup::Classic => (
                BlasKey::Mesh {
                    geometry: mesh.geometry_key.0,
                    sections: sections
                        .iter()
                        .map(|(s, _, surface)| {
                            (
                                s.first_index,
                                s.index_count,
                                surface.flags & RAY_INSTANCE_MASKED == 0,
                            )
                        })
                        .collect(),
                },
                sections
                    .iter()
                    .map(|(s, material, surface)| {
                        row(
                            indices.offset + s.first_index,
                            vertices.offset,
                            0,
                            *material,
                            surface,
                        )
                    })
                    .collect(),
                sections.iter().map(|(_, material, _)| *material).collect(),
                None,
                (
                    Source::Mesh {
                        vertices,
                        first_index: indices.offset,
                    },
                    sections
                        .iter()
                        .map(|(s, _, _)| u64::from(s.index_count / 3))
                        .sum(),
                ),
            ),
        };
        let (center, radius) = world_sphere(world_matrix, mesh.bounds_local);
        Ok(Traced {
            key,
            rows,
            materials,
            world: world_matrix,
            movable,
            proxy,
            slot: None,
            source_hint,
            center,
            radius,
            bounds_local: mesh.bounds_local,
            culled: false,
            skinned: false,
        })
    }

    /// Re-place the traced mesh `entity` after a transform change above it.
    fn place(&mut self, world: &World, entity: Entity) {
        let Some(Entry {
            chain,
            state: Ok(traced),
        }) = self.entries.get_mut(&entity)
        else {
            return;
        };
        let Some(matrix) = chain.first().and_then(|owner| {
            pulsar_scene_model::hierarchy_transform::world_matrix_of(world, *owner)
        }) else {
            return;
        };
        traced.world = matrix;
        (traced.center, traced.radius) = world_sphere(matrix, traced.bounds_local);
        if let Some(slot) = traced.slot {
            self.instances[slot].transform = tlas_transform(matrix, self.origin);
        }
        self.cull(entity);
    }

    /// The traced rows' colours from the surfaces (texture means arrived).
    fn recolour(&mut self, world: &World, mirror: &GpuMirrorHandle) {
        let meshes: Vec<Entity> = self.entries.keys().copied().collect();
        for entity in meshes {
            let Some(materials) = (match self.entries.get(&entity).map(|e| &e.state) {
                Some(Ok(traced)) => Some(traced.materials.clone()),
                _ => None,
            }) else {
                continue;
            };
            let surfaces: Vec<_> = materials
                .iter()
                .map(|material| self.surface(world, mirror, *material))
                .collect();
            if let Some(Ok(traced)) = self.entries.get_mut(&entity).map(|e| &mut e.state) {
                for (row, surface) in traced.rows.iter_mut().zip(surfaces) {
                    if let Some(surface) = surface {
                        row.base_color = surface.base_color;
                        row.emissive = surface.emissive;
                    }
                }
            }
        }
        self.rows_dirty = true;
    }

    /// Build queued BLASes, up to the frame's triangle budget.
    fn build_blases(&mut self, store: &SceneGpuStore, encoder: &mut wgpu::CommandEncoder) {
        let vertex_pool = store.resolve_buffer_handle(helio_mesh_api::MESH_VERTICES.key());
        let index_pool = store.resolve_buffer_handle(helio_mesh_api::MESH_INDICES.key());
        let proxies = self.vg.as_ref().and_then(VgRayProxies::buffers);
        let mut budget = BUILD_TRIANGLES_PER_FRAME;
        let mut built_any = false;
        while let Some(key) = self.build_queue.front() {
            let Some(blas) = self.blases.get(key) else {
                self.build_queue.pop_front();
                continue;
            };
            if blas.state != BlasState::Queued {
                self.build_queue.pop_front();
                continue;
            }
            if built_any && blas.triangles > budget {
                break;
            }
            let key = self.build_queue.pop_front().expect("front");
            let blas = self.blases.get_mut(&key).expect("present");
            let result = match (&key, blas.source) {
                (
                    BlasKey::Mesh { sections, .. },
                    Source::Mesh {
                        vertices,
                        first_index,
                    },
                ) => match (&vertex_pool, &index_pool) {
                    (Some(vertex_pool), Some(index_pool)) => {
                        let segments =
                            sections
                                .iter()
                                .map(|&(first, count, opaque)| BlasSegment {
                                    geometry: BlasGeometry {
                                        positions_revision: 0,
                                        topology_revision: 0,
                                        vertices: &vertex_pool.buffer,
                                        first_vertex: vertices.offset,
                                        vertex_count: vertices.count,
                                        vertex_stride: std::mem::size_of::<
                                            helio_mesh_api::PackedVertex,
                                        >()
                                            as u64,
                                        indices: Some(&index_pool.buffer),
                                        first_index: first_index + first,
                                        index_count: count,
                                    },
                                    opaque,
                                })
                                .collect();
                        self.blas
                            .build(blas.id, encoder, segments, helio_core::Movability::Static)
                            .map_err(|e| e.to_string())
                    }
                    _ => Err("the mesh pools are not registered".into()),
                },
                (BlasKey::Proxy { opaque, .. }, Source::Proxy(range)) => match proxies {
                    Some((vertices, indices)) => self
                        .blas
                        .build(
                            blas.id,
                            encoder,
                            vec![BlasSegment {
                                geometry: BlasGeometry {
                                    positions_revision: 0,
                                    topology_revision: 0,
                                    vertices,
                                    first_vertex: range.first_vertex,
                                    vertex_count: range.vertex_count,
                                    vertex_stride: helio_mesh_api::RAY_PROXY_VERTEX_BYTES,
                                    indices: Some(indices),
                                    first_index: range.first_index,
                                    index_count: range.index_count,
                                },
                                opaque: *opaque,
                            }],
                            helio_core::Movability::Static,
                        )
                        .map_err(|e| e.to_string()),
                    None => Err("the proxy pool is missing".into()),
                },
                _ => Err("source does not match its key".into()),
            };
            match result {
                Ok(_) => blas.state = BlasState::Built,
                Err(error) => {
                    blas.state = BlasState::Failed;
                    tracing::warn!(
                        "Meshes: a ray-tracing BLAS could not be built ({error}); its meshes are not traced"
                    );
                }
            }
            budget = budget.saturating_sub(blas.triangles);
            built_any = true;
        }
    }

    /// Build or refit the skinned instances' BLASes whose pose the skin cache
    /// changed, up to the frame's triangle budget: nearest first, an instance
    /// that waited counting as nearer by its frames waited (none waits for
    /// ever). The rest keep last frame's BLAS. An unchanged pose costs
    /// nothing.
    fn refit_skinned(
        &mut self,
        store: &SceneGpuStore,
        skins: &SkinCache,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        if self.skinned.is_empty() {
            return;
        }
        let Some(index_pool) = store.resolve_buffer_handle(helio_mesh_api::MESH_INDICES.key())
        else {
            return;
        };
        let eye = self.eye;
        let mut due: Vec<(f64, Entity, BlasKey)> = Vec::new();
        for entity in &self.skinned {
            let (Some(Ok(traced)), Some(skin)) = (
                self.entries.get(entity).map(|e| &e.state),
                skins.instance(*entity),
            ) else {
                continue;
            };
            let Some(blas) = self.blases.get(&traced.key) else {
                continue;
            };
            let current = blas.state == BlasState::Built && blas.revision == skin.revision;
            if current || blas.state == BlasState::Failed {
                continue;
            }
            let priority = traced.center.distance(eye) / f64::from(1 + blas.waited);
            due.push((priority, *entity, traced.key.clone()));
        }
        due.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut budget = self.settings.skinned_refit_triangles;
        let mut built_any = false;
        for (_, entity, key) in due {
            let skin = skins.instance(entity).expect("held above");
            let Some(blas) = self.blases.get_mut(&key) else {
                continue;
            };
            if built_any && blas.triangles > budget {
                blas.waited = blas.waited.saturating_add(1);
                continue;
            }
            let (BlasKey::Skinned { sections, .. }, Source::Skinned { first_index }) =
                (&key, blas.source)
            else {
                continue;
            };
            let segments = sections
                .iter()
                .map(|&(first, count, opaque)| BlasSegment {
                    geometry: BlasGeometry {
                        positions_revision: skin.revision,
                        topology_revision: 0,
                        vertices: skins.buffer(),
                        first_vertex: skin.cache_vertex,
                        vertex_count: skin.vertex_count,
                        vertex_stride: std::mem::size_of::<helio_mesh_api::SkinnedVertex>() as u64,
                        indices: Some(&index_pool.buffer),
                        first_index: first_index + first,
                        index_count: count,
                    },
                    opaque,
                })
                .collect();
            match self
                .blas
                .build(blas.id, encoder, segments, helio_core::Movability::Dynamic)
            {
                Ok(_) => {
                    blas.state = BlasState::Built;
                    blas.revision = skin.revision;
                    blas.waited = 0;
                }
                Err(error) => {
                    blas.state = BlasState::Failed;
                    tracing::warn!(
                        "Meshes: a skinned mesh's BLAS could not be built ({error}); it is not traced"
                    );
                }
            }
            budget = budget.saturating_sub(blas.triangles);
            built_any = true;
        }
    }

    /// Meshes whose BLAS is built join the TLAS.
    fn admit_built(&mut self) {
        let ready: Vec<Entity> = self
            .awaiting_blas
            .iter()
            .copied()
            .filter(|entity| {
                matches!(
                    self.entries.get(entity).map(|e| &e.state),
                    Some(Ok(traced)) if self.blases.get(&traced.key).is_some_and(|b| b.state == BlasState::Built)
                )
            })
            .collect();
        for entity in ready {
            self.awaiting_blas.remove(&entity);
            let Some(Ok(traced)) = self.entries.get_mut(&entity).map(|e| &mut e.state) else {
                continue;
            };
            let id = self.blases[&traced.key].id;
            traced.slot = Some(self.slots.len());
            self.slots.push(entity);
            self.instances.push(TlasInstanceInput {
                geometry_id: id,
                transform: tlas_transform(traced.world, self.origin),
                custom_data: 0,
            });
            // The casters' masks rays select by (the frame's ray masks).
            self.masks.push(if traced.movable {
                MASK_MOVABLE
            } else {
                MASK_STATIC
            });
            self.rows_dirty = true;
        }
    }

    /// Lay the traced meshes' rows out in TLAS order (each instance's custom
    /// data its first row) and upload them; keep the proxies they use.
    fn write_rows(&mut self) {
        self.rows_dirty = false;
        self.rows.clear();
        let mut proxies = HashSet::new();
        for (slot, entity) in self.slots.iter().enumerate() {
            let Some(Ok(traced)) = self.entries.get(entity).map(|e| &e.state) else {
                continue;
            };
            self.instances[slot].custom_data = self.rows.len() as u32;
            self.rows.extend_from_slice(&traced.rows);
            proxies.extend(traced.proxy);
        }
        for entity in &self.awaiting_blas {
            if let Some(Ok(traced)) = self.entries.get(entity).map(|e| &e.state) {
                proxies.extend(traced.proxy);
            }
        }
        if let Some(vg) = self.vg.as_mut() {
            vg.retain(&proxies);
        }
        let size = (self.rows.len().max(1) * std::mem::size_of::<RayInstance>()) as u64;
        if self
            .instances_buffer
            .as_ref()
            .is_none_or(|buffer| buffer.size() < size)
        {
            self.instances_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("World RT surfaces"),
                size: size.next_power_of_two(),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        if !self.rows.is_empty() {
            self.queue.write_buffer(
                self.instances_buffer.as_ref().expect("made above"),
                0,
                bytemuck::cast_slice(&self.rows),
            );
        }
    }

    /// What a ray sees of the material instance `entity` (cached until it
    /// changes).
    fn surface(
        &mut self,
        world: &World,
        mirror: &GpuMirrorHandle,
        entity: Entity,
    ) -> Option<Surface> {
        if let Some(surface) = self.surfaces.get(&entity) {
            return *surface;
        }
        // The row the materials plugin resolved when the instance bound, as
        // the GPU mirror's CPU shadow holds it. Never resolved here:
        // resolving a graph registers and uploads its textures (locks, cooks)
        // and must not run under the scene lock this projection holds.
        let row = world
            .is_alive(entity)
            .then(|| {
                mirror.store().read_dirty_tracked_row_bytes(
                    helio_mats::host_rows::MaterialComponent::packed_gpu_component_id(),
                    row_of(entity),
                )
            })
            .flatten()
            .and_then(|bytes| {
                bytemuck::try_pod_read_unaligned::<helio_mats::host_rows::MaterialComponent>(&bytes)
                    .ok()
            })
            .map(helio_mats::GpuMaterial::from);
        let surface = row.map(|row| {
            let texture = (row.tex_base_color != helio_mats::GpuMaterial::NO_TEXTURE)
                .then(|| mirror.texture_store())
                .flatten()
                .and_then(|store| store.read().ok()?.texture(row.tex_base_color).cloned());
            let mean = texture
                .and_then(|texture| self.texture_means.mean(&texture))
                .unwrap_or([1.0; 3]);
            let [r, g, b, _] = row.base_color;
            let [er, eg, eb, strength] = row.emissive;
            let masked = row.flags & helio_mats::FLAG_ALPHA_TEST != 0;
            let two_sided = row.flags & helio_mats::FLAG_DOUBLE_SIDED != 0;
            Surface {
                blended: !masked && row.flags & helio_mats::FLAG_ALPHA_BLEND != 0,
                flags: if masked { RAY_INSTANCE_MASKED } else { 0 }
                    | if two_sided { RAY_INSTANCE_TWO_SIDED } else { 0 },
                base_color: [r * mean[0], g * mean[1], b * mean[2], 1.0],
                emissive: [er * strength, eg * strength, eb * strength, 0.0],
            }
        });
        self.surfaces.insert(entity, surface);
        surface
    }

    fn stats(&self) -> Stats {
        let mut skipped: HashMap<Skip, usize> = HashMap::new();
        for entry in self.entries.values() {
            if let Err(skip) = entry.state {
                *skipped.entry(skip).or_default() += 1;
            }
        }
        let mut skipped: Vec<_> = skipped
            .into_iter()
            .map(|(skip, n)| (skip.label(), n))
            .collect();
        skipped.sort();
        Stats {
            meshes: self.entries.len(),
            traced: self.slots.len(),
            rows: self.rows.len(),
            blases: self.blases.len(),
            // Skinned BLASes of culled instances stay queued.
            queued: self
                .blases
                .iter()
                .filter(|(key, b)| {
                    b.state == BlasState::Queued && !matches!(key, BlasKey::Skinned { .. })
                })
                .count(),
            failed: self
                .blases
                .values()
                .filter(|b| b.state == BlasState::Failed)
                .count(),
            skipped,
        }
    }

    /// One line when what the scene traces settles on something new.
    fn log_stats(&mut self) {
        let stats = self.stats();
        if stats.queued > 0 || self.logged.as_ref() == Some(&stats) {
            return;
        }
        let skipped = stats
            .skipped
            .iter()
            .map(|(label, n)| format!("{n} {label}"))
            .collect::<Vec<_>>()
            .join(", ");
        tracing::info!(
            "Meshes: ray-traced scene: {} of {} meshes traced ({} sections, {} BLASes, {} failed){}",
            stats.traced,
            stats.meshes,
            stats.rows,
            stats.blases,
            stats.failed,
            if skipped.is_empty() {
                String::new()
            } else {
                format!("; not traced: {skipped}")
            }
        );
        self.logged = Some(stats);
    }
}

/// `owner` and its ancestors.
fn chain_of(world: &World, owner: Entity) -> Vec<Entity> {
    let mut chain = vec![owner];
    let mut at = world.get::<Parent>(owner).map(|parent| parent.0);
    while let Some(parent) = at {
        if chain.len() > MAX_DEPTH || chain.contains(&parent) {
            break;
        }
        chain.push(parent);
        at = world.get::<Parent>(parent).map(|p| p.0);
    }
    chain
}

/// A mesh's world bounding sphere: its local sphere (`[x, y, z, r]`) through
/// `matrix`, the radius by the largest axis scale.
fn world_sphere(matrix: DMat4, local: [f32; 4]) -> (DVec3, f64) {
    let [x, y, z, r] = local.map(f64::from);
    let center = matrix.transform_point3(DVec3::new(x, y, z));
    let scale = matrix
        .x_axis
        .truncate()
        .length()
        .max(matrix.y_axis.truncate().length())
        .max(matrix.z_axis.truncate().length());
    (center, r.max(0.0) * scale)
}

/// An entity as a key's bits (index and generation).
fn entity_bits(entity: Entity) -> u64 {
    (u64::from(entity.generation()) << 32) | u64::from(entity.index())
}

/// Row-major 3x4 TLAS transform of `matrix`, relative to `origin`.
fn tlas_transform(matrix: DMat4, origin: DVec3) -> [f32; 12] {
    let mut matrix = matrix;
    matrix.w_axis -= origin.extend(0.0);
    let m = matrix.as_mat4().to_cols_array();
    [
        m[0], m[4], m[8], m[12], m[1], m[5], m[9], m[13], m[2], m[6], m[10], m[14],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The TLAS transform is the world matrix moved into the frame whose
    /// origin is `origin`, row-major.
    #[test]
    fn transforms_are_relative_to_the_origin() {
        let matrix = DMat4::from_scale_rotation_translation(
            DVec3::new(2.0, 2.0, 2.0),
            glam::DQuat::IDENTITY,
            DVec3::new(1000.0, 5.0, -3.0),
        );
        let t = tlas_transform(matrix, DVec3::new(1000.0, 0.0, 0.0));
        assert_eq!(
            t,
            [2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 5.0, 0.0, 0.0, 2.0, -3.0]
        );
    }

    /// A mesh's chain is its owner then each ancestor, and a parent cycle
    /// ends it.
    #[test]
    fn chains_follow_parents_and_stop_at_cycles() {
        let mut world = World::new();
        let root = world.spawn();
        let mid = world.spawn();
        let owner = world.spawn();
        world.insert(mid, Parent(root));
        world.insert(owner, Parent(mid));
        assert_eq!(chain_of(&world, owner), vec![owner, mid, root]);
        world.insert(root, Parent(owner));
        assert_eq!(chain_of(&world, owner), vec![owner, mid, root]);
    }
}
