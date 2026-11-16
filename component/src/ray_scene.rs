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
//!   thousands of builds. BLASes are not compacted: wgpu-core's
//!   `Queue::compact_blas` takes `pending_writes` before
//!   `command_indices`, the reverse of `Queue::present`, and deadlocks
//!   against the editor UI presenting on the same device from its thread.
//!
//! What it traces: every enabled, visible static mesh section whose
//! material is opaque or a cut-out. Cut-outs are non-opaque segments flagged
//! `RAY_INSTANCE_MASKED`, so a ray query tests their alpha as the raster
//! does. Blended sections (glass) are left out: probe rays pass them. A
//! material graph is traced as its resolved row (its fallback surface and
//! its first texture's mean colour): rays cannot run a graph, so a hit's
//! colour is that coarse surface, as with every material
//! (`RayInstance::base_color`). Skinned meshes are not traced.
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
use helio_mats::instance::{MaterialInstanceComponent, row_of};
use helio_mesh_api::{RAY_INSTANCE_MASKED, RAY_INSTANCE_PROXY, RAY_INSTANCE_TWO_SIDED, RayInstance};
use helio_mesh_host::ray_proxies::{Lookup, ProxyRange, VgRayProxies};
use helio_mesh_host::texture_means::TextureMeans;
use pulsar_scene_model::attachments::ComponentOwner;
use pulsar_scene_model::{Parent, Transform, Visibility};
use pulsar_scenedb::gpu::{GpuMirrorHandle, SceneGpuStore, VarLenHandle};
use pulsar_scenedb::{ChangeCursor, ChangeRead, ComponentChange, Entity, World};

use crate::components::{MeshGeometryKey, StaticMeshComponent, StaticMeshDraw};
use crate::mesh_materials::MeshMaterials;

/// Triangles of new BLASes built a frame (at least one BLAS is).
const BUILD_TRIANGLES_PER_FRAME: u64 = 1 << 20;
/// Meshes (re-)evaluated a frame.
const EVALUATIONS_PER_FRAME: usize = 256;
/// Ancestors followed from a mesh's owner (as the transform hierarchy).
const MAX_DEPTH: usize = 64;

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
}

struct Blas {
    id: u64,
    /// Meshes tracing it.
    users: u32,
    state: BlasState,
    source: Source,
    triangles: u64,
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
        })
    }

    /// Bring the TLAS up to date with `world`, in the frame whose world
    /// origin is `origin`. Call after the world's GPU mirror is flushed (the
    /// BLASes read the mirrored geometry).
    pub fn prepare(&mut self, world: &World, origin: DVec3) {
        let Some(mirror) = world.gpu_mirror() else {
            return;
        };
        let store = mirror.store();
        // Proxies that came or went: re-evaluate everything.
        if self
            .vg
            .as_mut()
            .is_some_and(|vg| vg.poll(&self.device, &self.queue))
        {
            self.cursors = None;
        }
        let mut dirty = std::mem::take(&mut self.pending);
        let mut moved = HashSet::new();
        self.read_changes(world, &mut dirty, &mut moved);
        dirty.extend(self.waiting.drain());
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
            self.evaluate(world, mirror, store, entity);
        }
        for entity in moved {
            if !self.pending.contains(&entity) {
                self.place(world, entity);
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
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("World RT acceleration"),
            });
        self.build_blases(store, &mut encoder);
        self.admit_built();
        if self.rows_dirty {
            self.write_rows();
        }
        match self
            .tlas
            .build_masked(&mut encoder, &self.instances, &self.blas, &self.masks)
        {
            Ok(()) => {
                self.tlas_ok = true;
                self.warned = false;
            }
            Err(error) => {
                self.tlas_ok = false;
                if !self.warned {
                    self.warned = true;
                    tracing::warn!("Meshes: the ray-traced scene's TLAS failed ({error}); rays see no meshes");
                }
            }
        }
        self.queue.submit([encoder.finish()]);
        self.texture_means.dispatch(&self.device, &self.queue);
        self.log_stats();
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
        Some(rays)
    }

    /// How many meshes the TLAS holds.
    pub fn instance_count(&self) -> usize {
        self.slots.len()
    }

    /// The journals' changes since last frame, as meshes to re-evaluate
    /// (`dirty`) and to re-place (`moved`); everything after an overflow.
    fn read_changes(&mut self, world: &World, dirty: &mut HashSet<Entity>, moved: &mut HashSet<Entity>) {
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
                let materials = read(&mut cursors.materials);
                let placed = [read(&mut cursors.visibility), read(&mut cursors.parents)];
                let transforms = read(&mut cursors.transforms);
                if complete {
                    dirty.extend(own.into_iter().flatten());
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
            dirty.extend(world.query::<&StaticMeshComponent>().map(|(entity, _)| entity));
        }
    }

    /// Re-evaluate the mesh `entity` from the world: drop what it traced,
    /// trace what it now draws.
    fn evaluate(&mut self, world: &World, mirror: &GpuMirrorHandle, store: &SceneGpuStore, entity: Entity) {
        if let Some(old) = self.entries.remove(&entity) {
            self.forget(entity, old);
        }
        let Some(entry) = self.entry(world, mirror, store, entity) else {
            return;
        };
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
                let blas = self.blases.entry(traced.key.clone()).or_insert_with(|| {
                    let id = self.next_id;
                    self.next_id += 1;
                    self.build_queue.push_back(traced.key.clone());
                    Blas {
                        id,
                        users: 0,
                        state: BlasState::Queued,
                        source: source.0,
                        triangles: source.1,
                    }
                });
                blas.users += 1;
                self.awaiting_blas.insert(entity);
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
        let Ok(traced) = old.state else {
            return;
        };
        if let Some(slot) = traced.slot {
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
        if let Some(blas) = self.blases.get_mut(&traced.key) {
            blas.users -= 1;
            if blas.users == 0 {
                self.blas.remove_blas(blas.id);
                self.blases.remove(&traced.key);
            }
        }
    }

    /// What the mesh `entity` traces now (`None`: it is no mesh).
    fn entry(&mut self, world: &World, mirror: &GpuMirrorHandle, store: &SceneGpuStore, entity: Entity) -> Option<Entry> {
        let mesh = world.get::<StaticMeshComponent>(entity)?;
        let link = world.get::<ComponentOwner>(entity)?;
        let owner = link.entity();
        let chain = chain_of(world, owner);
        let state = self.trace(world, mirror, store, entity, mesh, link, owner);
        Some(Entry { chain, state })
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
        if !link.is_enabled() || !world.is_alive(owner) {
            return Err(Skip::Disabled);
        }
        if world
            .get::<Visibility>(owner)
            .is_some_and(|visibility| !visibility.visible)
        {
            return Err(Skip::Hidden);
        }
        if mesh.geometry_key == MeshGeometryKey::NONE || mesh.indices.is_empty() {
            return Err(Skip::Unloaded);
        }
        let (Some(vertices), Some(indices)) = (
            StaticMeshComponent::vertices_gpu_handle(store, entity.index()),
            StaticMeshComponent::indices_gpu_handle(store, entity.index()),
        ) else {
            return Err(Skip::Unmirrored);
        };
        if vertices.count as usize != mesh.vertices.len() || indices.count as usize != mesh.indices.len() {
            return Err(Skip::Unmirrored);
        }
        let Some(instances) = world.get::<MeshMaterials>(entity) else {
            return Err(Skip::NoMaterials);
        };
        let world_matrix =
            pulsar_scene_model::hierarchy_transform::world_matrix_of(world, owner).ok_or(Skip::Disabled)?;
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
            let surface = self.surface(world, mirror, material).ok_or(Skip::NoMaterials)?;
            if !surface.blended {
                sections.push((section, material, surface));
            }
        }
        if sections.is_empty() {
            return Err(Skip::Glass);
        }
        let row = |first_index, base_vertex, flags, material: Entity, surface: &Surface| RayInstance {
            first_index,
            base_vertex,
            flags: flags | surface.flags,
            material: row_of(material),
            base_color: surface.base_color,
            emissive: surface.emissive,
            transmission: [0.0; 4],
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
                    vec![row(range.first_index, range.first_vertex, RAY_INSTANCE_PROXY, *material, surface)],
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
                            (s.first_index, s.index_count, surface.flags & RAY_INSTANCE_MASKED == 0)
                        })
                        .collect(),
                },
                sections
                    .iter()
                    .map(|(s, material, surface)| {
                        row(indices.offset + s.first_index, vertices.offset, 0, *material, surface)
                    })
                    .collect(),
                sections.iter().map(|(_, material, _)| *material).collect(),
                None,
                (
                    Source::Mesh {
                        vertices,
                        first_index: indices.offset,
                    },
                    sections.iter().map(|(s, _, _)| u64::from(s.index_count / 3)).sum(),
                ),
            ),
        };
        Ok(Traced {
            key,
            rows,
            materials,
            world: world_matrix,
            movable,
            proxy,
            slot: None,
            source_hint,
        })
    }

    /// Re-place the traced mesh `entity` after a transform change above it.
    fn place(&mut self, world: &World, entity: Entity) {
        let Some(Entry { chain, state: Ok(traced) }) = self.entries.get_mut(&entity) else {
            return;
        };
        let Some(matrix) = chain
            .first()
            .and_then(|owner| pulsar_scene_model::hierarchy_transform::world_matrix_of(world, *owner))
        else {
            return;
        };
        traced.world = matrix;
        if let Some(slot) = traced.slot {
            self.instances[slot].transform = tlas_transform(matrix, self.origin);
        }
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
                (BlasKey::Mesh { sections, .. }, Source::Mesh { vertices, first_index }) => {
                    match (&vertex_pool, &index_pool) {
                        (Some(vertex_pool), Some(index_pool)) => {
                            let segments = sections
                                .iter()
                                .map(|&(first, count, opaque)| BlasSegment {
                                    geometry: BlasGeometry {
                                        revision: 0,
                                        vertices: &vertex_pool.buffer,
                                        first_vertex: vertices.offset,
                                        vertex_count: vertices.count,
                                        vertex_stride: std::mem::size_of::<helio_mesh_api::PackedVertex>()
                                            as u64,
                                        indices: Some(&index_pool.buffer),
                                        first_index: first_index + first,
                                        index_count: count,
                                    },
                                    opaque,
                                })
                                .collect();
                            self.blas
                                .build_segments(blas.id, encoder, segments)
                                .map_err(|e| e.to_string())
                        }
                        _ => Err("the mesh pools are not registered".into()),
                    }
                }
                (BlasKey::Proxy { opaque, .. }, Source::Proxy(range)) => match proxies {
                    Some((vertices, indices)) => self
                        .blas
                        .build_segments(
                            blas.id,
                            encoder,
                            vec![BlasSegment {
                                geometry: BlasGeometry {
                                    revision: 0,
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
                    tracing::warn!("Meshes: a ray-tracing BLAS could not be built ({error}); its meshes are not traced");
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
            // The shadow casters' masks: static 1, movable 2.
            self.masks.push(if traced.movable { 2 } else { 1 });
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
    fn surface(&mut self, world: &World, mirror: &GpuMirrorHandle, entity: Entity) -> Option<Surface> {
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
                bytemuck::try_pod_read_unaligned::<helio_mats::host_rows::MaterialComponent>(&bytes).ok()
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
        let mut skipped: Vec<_> = skipped.into_iter().map(|(skip, n)| (skip.label(), n)).collect();
        skipped.sort();
        Stats {
            meshes: self.entries.len(),
            traced: self.slots.len(),
            rows: self.rows.len(),
            blases: self.blases.len(),
            queued: self
                .blases
                .values()
                .filter(|b| b.state == BlasState::Queued)
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
        assert_eq!(t, [2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 5.0, 0.0, 0.0, 2.0, -3.0]);
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
