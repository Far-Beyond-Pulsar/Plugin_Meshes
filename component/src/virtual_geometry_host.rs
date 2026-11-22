//! The CPU side of virtual geometry for a host (editor, game): which meshes
//! need a `.vgeo`, building and caching it as a background task, handing it
//! to the streaming runtime, cutting each mesh's error-bounded fallback and
//! swapping it in for the full classic geometry, and publishing which assets
//! are ready to the GPU.
//!
//! ```text
//! host frame          VirtualGeometryHost::sync(world, mirror, settings, fallback error)
//!                       - settings -> VgRuntime, VgInstanceControl
//!                       - StaticMeshComponent change journal -> the instances
//!                         with `virtual_geometry` set, by content, and the
//!                         instances showing a fallback
//!                       - per wanted content: load or build its `.vgeo`
//!                         (a `helio_tasks` job), then
//!                         VgRuntime::request_asset(content), and cut its
//!                         fallback (another job)
//!                       - the instances whose geometry must change
//!                     VirtualGeometryHost::apply(world)
//!                       - swap the fallback in, or the full geometry back
//! scene derivations   VgAssetReadyPublisher  -> `vg_asset_ready` (rows from the runtime)
//!                     VirtualGeometryInstances -> `vg_instances`, `lod_suppressed`,
//!                                                 `classic_suppressed`
//! graph passes        stream pages, cull, raster, shade (helio-presets)
//! ```
//!
//! **Incremental.** The instances are followed through SceneDB's change
//! journal for `StaticMeshComponent` (one full scan when the cursor opens or
//! the journal overflows, as on a world swap), so a frame's work scales with
//! what changed, never with every mesh in the world: an instance is looked at
//! again when it changes, when its content's fallback arrives, or when the
//! renderer starts or stops drawing virtual geometry. The binding of a mesh's
//! geometry is a journaled write, so an instance is seen again once its mesh
//! has loaded.
//!
//! **Where the cluster data lives.** A mesh with a native `.mesh` file keeps
//! its `.vgeo` as the file's embedded chunk (`mesh_cache::write_embedded_vgeo`),
//! checked against the mesh's content id and the builder's cook revision, so
//! it is built once and ships with the asset. Anything else (generated
//! geometry, a mesh read straight from a source model) is derived data in the
//! project's `.pulsar/cache/vgeo/`, keyed by content id and format version.
//!
//! **Builds are tasks.** Each load-or-build is a `helio_tasks` job named
//! after its mesh: the editor shows it in the Tasks window with its phases
//! and log, a failure is an error in that log, and the content is retried the
//! next time an instance asks for it after none did (turning the toggle off
//! and on again). So is each fallback cut.
//!
//! **Only the fallback stays on the GPU.** A mesh's fallback is one cut of its
//! cluster DAG at an explicit error bound (`helio_vg_stream::ProxyOptions::fallback`:
//! the mesh asset's `MeshAssetSettings::vg_fallback_error`, else the
//! project's `fallback_relative_error`), with every vertex's attributes and
//! the mesh's material sections. Once it is cut, every instance that uses
//! virtual geometry shows it in place of the full geometry
//! ([`StaticMeshComponent::show_fallback`]): the full geometry's interned GPU
//! ranges are released with their last reference, and its CPU copy is kept
//! once, shared ([`crate::components::FullGeometry`]: the world's asset
//! payload, or for generated geometry one copy for all its instances). What
//! draws classic geometry draws the fallback: the instance itself until its
//! clusters are resident (it never draws nothing), shadows, the world's
//! ray-traced scene (`crate::ray_scene`). Turning `virtual_geometry` off on an
//! instance shows its full geometry again (the property hook), as does the
//! project's mode `Off`, a renderer whose device cannot draw virtual geometry
//! (`VgRuntime::draws_virtual_geometry`), or a failed build. The fallback is
//! written into the world, which every renderer of it draws, so it is shown
//! only while all of them draw virtual geometry (`WorldRenderers`): a
//! second viewport with virtual geometry off keeps the full geometry for
//! both. A changed error
//! bound cuts a new fallback; instances keep the old one until it arrives.
//!
//! Until an asset's root pages are resident its instances are not VG-active:
//! they render classically and their authored LOD applies (plan section 3,
//! item 7). Only static meshes take part; skinned meshes, voxels and foliage
//! are other components and stay classic, as do deforming (`Dynamic`)
//! static meshes: virtual geometry is for rigid geometry.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::{Arc, Weak};

use helio_core::{SceneDerivation, SceneDerivationContext, SceneDerivationOutput};
use helio_mesh_api::PackedVertex;
use helio_vg_format::{FORMAT_VERSION, VgAssetReadyRow};
use helio_vg_stream::{
    ProxyOptions, VgAssetReport, VgRuntime, VgRuntimeSettings, VgeoBytes, VgeoSource,
    content_words,
};
use pulsar_scenedb::gpu::GpuMirrorHandle;
use pulsar_scenedb::handle_ledger::HandleId;
use pulsar_scenedb::{ChangeCursor, ChangeRead, ComponentChange, ComponentChangeKind, Entity, World};

use crate::components::{
    FullGeometry, LoadedMesh, MeshAssetPath, MeshGeometry, MeshGeometryKey, StaticMeshComponent,
};
use crate::mesh_cache::MeshSection;
use crate::virtual_geometry_instances::{
    Output, VG_ASSET_READY_KEY, VG_ASSET_READY_ROW_BYTES, VgInstanceControl, VgInstanceKeys,
    VirtualGeometryInstances,
};

enum HostAsset {
    Building(mpsc::Receiver<Result<Arc<VgeoBytes>, String>>),
    Ready(Arc<VgeoBytes>),
    Failed,
}

/// What a fallback cut job reports.
enum CutResult {
    /// The asset's error bound is still `error`: the fallback in place stays.
    Unchanged,
    Cut(Arc<MeshGeometry>, f32),
    Failed,
}

/// One content's fallback.
#[derive(Default)]
struct Fallback {
    /// The one in place, and the error bound it was cut at.
    current: Option<(Arc<MeshGeometry>, f32)>,
    /// A cut running.
    job: Option<mpsc::Receiver<CutResult>>,
    /// The project error and settings revision the last cut was asked at.
    asked: Option<(u32, u64)>,
    failed: bool,
}

/// A change [`VirtualGeometryHost::apply`] makes to an instance.
enum Swap {
    /// Show this fallback of the instance's content.
    Fallback(u128, Arc<MeshGeometry>),
    /// Show its full geometry again.
    Full,
}

/// What the host knows of one instance, from its component.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Instance {
    /// The content id of its full geometry (`StaticMeshComponent::source_key`).
    content: u128,
    /// It asks for virtual geometry and may have it.
    wanted: bool,
    /// The key of the geometry it shows (its fallback's while one shows).
    shows: u128,
    /// Its component holds a fallback.
    showing_fallback: bool,
}

/// GPU bytes of the full geometry and of the fallback of the contents in
/// use, as the host last measured them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VgGeometryBytes {
    pub classic: u64,
    pub fallback: u64,
}

/// Whether each renderer of a world draws virtual geometry, by world (its
/// address) and host. The fallback replaces the full geometry in the world
/// itself, which every renderer of the world draws, so it is shown only when
/// all of them draw virtual geometry; one that does not (its mode `Off`, a
/// device without the features) draws the full geometry, and so do the
/// others' instances until their clusters are resident. Without the
/// agreement two renderers that disagree would swap the same meshes back and
/// forth every frame.
struct WorldRenderers {
    votes: std::sync::Mutex<HashMap<usize, HashMap<u64, bool>>>,
    next_host: std::sync::atomic::AtomicU64,
}

fn world_renderers() -> &'static WorldRenderers {
    static RENDERERS: std::sync::OnceLock<WorldRenderers> = std::sync::OnceLock::new();
    RENDERERS.get_or_init(|| WorldRenderers {
        votes: std::sync::Mutex::new(HashMap::new()),
        next_host: std::sync::atomic::AtomicU64::new(1),
    })
}

/// A host's place in [`WorldRenderers`]: the world it last drew; its vote
/// leaves with it.
struct WorldVote {
    host: u64,
    world: Option<usize>,
}

impl WorldVote {
    fn new() -> Self {
        Self {
            host: world_renderers()
                .next_host
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            world: None,
        }
    }

    /// Record whether this host's renderer `draws` virtual geometry of
    /// `world`; whether every renderer of that world does.
    fn vote(&mut self, world: &World, draws: bool) -> bool {
        let key = std::ptr::from_ref(world) as usize;
        let mut votes = world_renderers()
            .votes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(old) = self.world.filter(|old| *old != key) {
            Self::leave(&mut votes, old, self.host);
        }
        self.world = Some(key);
        let world_votes = votes.entry(key).or_default();
        world_votes.insert(self.host, draws);
        world_votes.values().all(|draws| *draws)
    }

    fn leave(votes: &mut HashMap<usize, HashMap<u64, bool>>, world: usize, host: u64) {
        if let Some(world_votes) = votes.get_mut(&world) {
            world_votes.remove(&host);
            if world_votes.is_empty() {
                votes.remove(&world);
            }
        }
    }
}

impl Drop for WorldVote {
    fn drop(&mut self) {
        if let Some(world) = self.world {
            let mut votes = world_renderers()
                .votes
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            Self::leave(&mut votes, world, self.host);
        }
    }
}

/// See the module doc. Owned by whoever owns the renderer.
pub struct VirtualGeometryHost {
    runtime: VgRuntime,
    /// This renderer's say in whether its world shows fallbacks.
    vote: WorldVote,
    control: Option<VgInstanceControl>,
    /// Loaded, building or failed `.vgeo`s, by mesh content id.
    assets: HashMap<u128, HostAsset>,
    /// Each content's fallback.
    fallbacks: HashMap<u128, Fallback>,
    /// The instances with virtual geometry wanted or a fallback shown.
    instances: HashMap<Entity, Instance>,
    /// Those instances by content.
    by_content: HashMap<u128, HashSet<Entity>>,
    /// Instances to look at this frame.
    dirty: HashSet<Entity>,
    /// Changes for [`Self::apply`].
    swaps: Vec<(Entity, Swap)>,
    /// Generated full geometry kept while its instances show their fallback,
    /// shared by them.
    owned: HashMap<u128, Weak<MeshGeometry>>,
    /// Whether the last sync showed fallbacks (every renderer of the world
    /// drew virtual geometry).
    drawn: bool,
    /// The GPU bytes of the contents in use ([`Self::geometry_bytes`]).
    bytes: VgGeometryBytes,
    /// `StaticMeshComponent`'s change journal (`None` until the first scan).
    cursor: Option<ChangeCursor>,
    changes: Vec<ComponentChange>,
}

impl Default for VirtualGeometryHost {
    fn default() -> Self {
        Self::new()
    }
}

impl VirtualGeometryHost {
    pub fn new() -> Self {
        Self {
            runtime: VgRuntime::new(VgRuntimeSettings::default()),
            vote: WorldVote::new(),
            control: None,
            assets: HashMap::new(),
            fallbacks: HashMap::new(),
            instances: HashMap::new(),
            by_content: HashMap::new(),
            dirty: HashSet::new(),
            swaps: Vec::new(),
            owned: HashMap::new(),
            drawn: false,
            bytes: VgGeometryBytes::default(),
            cursor: None,
            changes: Vec::new(),
        }
    }

    /// The streaming runtime, for `helio_pass_vg::with_virtual_geometry`.
    pub fn runtime(&self) -> &VgRuntime {
        &self.runtime
    }

    /// The GPU bytes of the full classic geometry and of the fallbacks of
    /// the contents in use, as last measured.
    pub fn geometry_bytes(&self) -> VgGeometryBytes {
        self.bytes
    }

    /// The `vg_asset_ready` publisher. Install it after the scene join and
    /// before [`Self::instances_derivation`].
    pub fn asset_ready_derivation(&self, device: &wgpu::Device) -> Box<VgAssetReadyPublisher> {
        Box::new(VgAssetReadyPublisher::new(device, self.runtime.clone()))
    }

    /// The instance derivation. Install it after [`Self::asset_ready_derivation`]
    /// and before the authored-LOD selection.
    pub fn instances_derivation(&mut self, device: &wgpu::Device) -> Box<VirtualGeometryInstances> {
        let derivation = Box::new(VirtualGeometryInstances::new(
            device,
            VgInstanceKeys::standard(),
        ));
        let control = derivation.control();
        control.set_enabled(self.runtime.settings().mode != helio_vg_format::VgMode::Off);
        self.control = Some(control);
        derivation
    }

    /// Per-frame (or per-tick) service: pushes `settings`, follows the
    /// instances' changes, collects finished builds and cuts, starts builds
    /// for contents that need one, offers the ready ones to the runtime and
    /// decides which instances show their fallback (`fallback_error`: the
    /// project's error bound, a fraction of a mesh's bounding radius). Cheap
    /// when nothing changed: its work is what changed. The geometry swaps
    /// it decides are made by [`Self::apply`] ([`Self::has_swaps`]).
    pub fn sync(
        &mut self,
        world: &World,
        mirror: &GpuMirrorHandle,
        settings: VgRuntimeSettings,
        fallback_error: f32,
    ) {
        self.runtime.set_settings(settings);
        let on = settings.mode != helio_vg_format::VgMode::Off;
        if let Some(control) = &self.control {
            control.set_enabled(on);
        }
        self.follow(world);
        self.collect_builds();
        self.collect_cuts();

        // Fallbacks shown (this renderer and every other renderer of the
        // world draw virtual geometry) started or stopped: every instance may
        // swap.
        let drawn = self.vote.vote(world, self.runtime.draws_virtual_geometry());
        if drawn != self.drawn {
            self.drawn = drawn;
            self.dirty.extend(self.instances.keys().copied());
        }

        // The contents wanted, and one instance of each (the one a build
        // reads).
        let mut wanted: HashMap<u128, Entity> = HashMap::new();
        for (&entity, instance) in &self.instances {
            if instance.wanted {
                wanted.entry(instance.content).or_insert(entity);
            }
        }
        // A failed content nobody asks for any more is forgotten, so asking
        // again retries it; so is the fallback of a content nobody uses.
        self.assets.retain(|content, asset| {
            !matches!(asset, HostAsset::Failed) || wanted.contains_key(content)
        });
        let used: HashSet<u128> = self.by_content.keys().copied().collect();
        self.fallbacks.retain(|content, _| used.contains(content));
        self.owned.retain(|_, geometry| geometry.strong_count() > 0);

        let revision = crate::mesh_cache::asset_settings_revision();
        for (&content, &entity) in &wanted {
            match self.assets.get(&content) {
                None if on => {
                    if let Some(mesh) = world.get::<StaticMeshComponent>(entity) {
                        self.assets
                            .insert(content, HostAsset::Building(spawn_build(content, mesh)));
                    }
                }
                Some(HostAsset::Ready(bytes)) => {
                    let source: Arc<dyn VgeoSource> = bytes.clone();
                    self.runtime.request_asset(content, source);
                    if drawn {
                        let bytes = bytes.clone();
                        self.ask_cut(world, content, entity, bytes, fallback_error, revision);
                    }
                }
                _ => {}
            }
        }

        // The instances to look at: what they should show now.
        let dirty: Vec<Entity> = self.dirty.drain().collect();
        for entity in dirty {
            let Some(instance) = self.instances.get(&entity).copied() else {
                continue;
            };
            let failed = matches!(self.assets.get(&instance.content), Some(HostAsset::Failed));
            if instance.wanted && drawn && !failed {
                if let Some((fallback, _)) = self
                    .fallbacks
                    .get(&instance.content)
                    .and_then(|f| f.current.as_ref())
                {
                    if instance.shows != fallback.key.0 {
                        self.swaps
                            .push((entity, Swap::Fallback(instance.content, fallback.clone())));
                    }
                }
                // No fallback yet: it keeps what it shows (its full geometry,
                // or the previous fallback while a new one is cut).
            } else if instance.showing_fallback {
                self.swaps.push((entity, Swap::Full));
            }
        }
        self.measure(world, mirror);
    }

    /// Whether [`Self::apply`] has geometry to swap.
    pub fn has_swaps(&self) -> bool {
        !self.swaps.is_empty()
    }

    /// Make the swaps the last [`Self::sync`] decided: each instance shows its
    /// fallback (its full geometry kept on the CPU, released from the GPU) or
    /// its full geometry again. An instance that changed meanwhile is left
    /// for the next sync, which sees its change.
    pub fn apply(&mut self, world: &mut World) {
        for (entity, swap) in std::mem::take(&mut self.swaps) {
            match swap {
                Swap::Full => {
                    if let Some(mut mesh) = world.get_mut::<StaticMeshComponent>(entity) {
                        mesh.show_full_geometry();
                    }
                }
                Swap::Fallback(content, fallback) => {
                    let Some(mesh) = world.get::<StaticMeshComponent>(entity) else {
                        continue;
                    };
                    if mesh.source_key().0 != content {
                        continue;
                    }
                    // The asset's payload when it is the geometry shown (the
                    // world keeps it while the instance references it).
                    let payload = world
                        .asset_payload::<LoadedMesh>(mesh.mesh)
                        .filter(|loaded| {
                            (loaded.content_id == content || loaded.content_id == 0)
                                && loaded.upload.geometry.vertices.len()
                                    == mesh.source_vertices().len()
                                && loaded.upload.geometry.indices.len()
                                    == mesh.source_indices().len()
                        });
                    let shared = self.owned.get(&content).and_then(Weak::upgrade);
                    let mut kept = None;
                    if let Some(mut mesh) = world.get_mut::<StaticMeshComponent>(entity) {
                        mesh.show_fallback(&fallback, |mesh| {
                            let key = MeshGeometryKey(content);
                            if let Some(loaded) = payload {
                                return FullGeometry::Asset { key, mesh: loaded };
                            }
                            let owned = shared.unwrap_or_else(|| {
                                Arc::new(MeshGeometry {
                                    key,
                                    vertices: std::mem::take(&mut mesh.vertices),
                                    indices: std::mem::take(&mut mesh.indices),
                                    sections: std::mem::take(&mut mesh.mesh_sections),
                                })
                            });
                            kept = Some(owned.clone());
                            FullGeometry::Owned(owned)
                        });
                    }
                    if let Some(owned) = kept {
                        self.owned.insert(content, Arc::downgrade(&owned));
                    }
                }
            }
        }
    }

    /// Start a fallback cut for `content` when it has none, or when its error
    /// bound may have changed (the project's, or a mesh asset's settings).
    fn ask_cut(
        &mut self,
        world: &World,
        content: u128,
        entity: Entity,
        bytes: Arc<VgeoBytes>,
        project_error: f32,
        revision: u64,
    ) {
        let fallback = self.fallbacks.entry(content).or_default();
        let asked = (project_error.to_bits(), revision);
        if fallback.job.is_some() || fallback.asked == Some(asked) {
            return;
        }
        let Some(mesh) = world.get::<StaticMeshComponent>(entity) else {
            return;
        };
        fallback.asked = Some(asked);
        fallback.failed = false;
        fallback.job = Some(spawn_cut(
            content,
            mesh,
            bytes,
            project_error,
            fallback.current.as_ref().map(|(_, error)| *error),
        ));
    }

    /// Take the results of finished fallback cuts: a new fallback dirties
    /// its content's instances.
    fn collect_cuts(&mut self) {
        for (content, fallback) in &mut self.fallbacks {
            let Some(job) = &fallback.job else {
                continue;
            };
            let result = match job.try_recv() {
                Ok(result) => result,
                Err(mpsc::TryRecvError::Empty) => continue,
                Err(mpsc::TryRecvError::Disconnected) => CutResult::Failed,
            };
            fallback.job = None;
            match result {
                CutResult::Unchanged => {}
                CutResult::Cut(geometry, error) => {
                    fallback.current = Some((geometry, error));
                    self.dirty
                        .extend(self.by_content.get(content).into_iter().flatten());
                }
                // The task logged why; its instances keep what they show.
                CutResult::Failed => fallback.failed = true,
            }
        }
    }

    /// Bring [`Self::instances`] up to date from the change journal, or by a
    /// full scan when there is no cursor yet or the journal cannot say what
    /// changed (overflow, another world). What changed is looked at again.
    fn follow(&mut self, world: &World) {
        let rescan = match self.cursor.as_mut() {
            None => true,
            Some(cursor) => {
                self.changes.clear();
                world.read_changes(cursor, &mut self.changes) == ChangeRead::Overflowed
            }
        };
        if rescan {
            // Open first: a change made during the scan is read next frame.
            self.cursor = Some(world.open_change_cursor::<StaticMeshComponent>());
            self.instances.clear();
            self.by_content.clear();
            for (entity, mesh) in world.query::<&StaticMeshComponent>() {
                self.track(entity, Some(mesh));
            }
            return;
        }
        let changes = std::mem::take(&mut self.changes);
        for change in &changes {
            let mesh = match change.kind {
                ComponentChangeKind::Removed => None,
                ComponentChangeKind::Inserted | ComponentChangeKind::Mutated => {
                    world.get::<StaticMeshComponent>(change.entity)
                }
            };
            self.track(change.entity, mesh);
        }
        self.changes = changes;
    }

    /// Record what `entity`'s component (`None`: it has none) is to the host.
    fn track(&mut self, entity: Entity, mesh: Option<&StaticMeshComponent>) {
        let instance = mesh.and_then(|mesh| {
            let content = mesh.source_key().0;
            let wanted = wanted_content(mesh).is_some();
            let showing_fallback = mesh.full_geometry.is_some();
            (wanted || showing_fallback).then_some(Instance {
                content,
                wanted,
                shows: mesh.geometry_key.0,
                showing_fallback,
            })
        });
        let old = match instance {
            Some(instance) => {
                self.dirty.insert(entity);
                self.instances.insert(entity, instance)
            }
            None => self.instances.remove(&entity),
        };
        if let Some(old) = old {
            if let Some(set) = self.by_content.get_mut(&old.content) {
                set.remove(&entity);
                if set.is_empty() {
                    self.by_content.remove(&old.content);
                }
            }
        }
        if let Some(instance) = instance {
            self.by_content
                .entry(instance.content)
                .or_default()
                .insert(entity);
        }
    }

    /// Take the results of finished builds.
    fn collect_builds(&mut self) {
        for (content, asset) in &mut self.assets {
            if let HostAsset::Building(rx) = asset {
                match rx.try_recv() {
                    Ok(Ok(bytes)) => *asset = HostAsset::Ready(bytes),
                    // The task logged why: its instances show their full
                    // geometry.
                    Ok(Err(_)) | Err(mpsc::TryRecvError::Disconnected) => {
                        *asset = HostAsset::Failed;
                        self.dirty
                            .extend(self.by_content.get(content).into_iter().flatten());
                    }
                    Err(mpsc::TryRecvError::Empty) => {}
                }
            }
        }
    }

    /// The GPU bytes of each content in use (its full classic geometry while
    /// any instance shows it, and its fallback), reported to the runtime for
    /// the Memory Explorer and summed for the frame stats. Hash lookups per
    /// content in use; no scan of the world.
    fn measure(&mut self, world: &World, mirror: &GpuMirrorHandle) {
        let store = mirror.store();
        let vertices =
            store.interned_var_len_pool::<PackedVertex>(helio_mesh_api::MESH_VERTICES.key());
        let indices = store.interned_var_len_pool::<u32>(helio_mesh_api::MESH_INDICES.key());
        let resident = |key: u128| -> u64 {
            let count = |count: Option<(u32, u32)>, stride: usize| {
                count.map_or(0, |(_, count)| u64::from(count) * stride as u64)
            };
            count(
                vertices.as_ref().and_then(|p| p.resolve(HandleId(key))),
                std::mem::size_of::<PackedVertex>(),
            ) + count(indices.as_ref().and_then(|p| p.resolve(HandleId(key))), 4)
        };
        let mut total = VgGeometryBytes::default();
        for (&content, entities) in &self.by_content {
            let fallback = self
                .fallbacks
                .get(&content)
                .and_then(|f| f.current.as_ref())
                .map_or(0, |(geometry, _)| resident(geometry.key.0));
            let classic = resident(content);
            total.classic += classic;
            total.fallback += fallback;
            let name = entities
                .iter()
                .find_map(|entity| world.get::<StaticMeshComponent>(*entity))
                .map_or_else(
                    || format!("{content:032x}"),
                    |mesh| display_name(&mesh.mesh_asset, content),
                );
            self.runtime.report_asset(
                content,
                VgAssetReport {
                    name,
                    classic_bytes: classic,
                    fallback_bytes: fallback,
                },
            );
        }
        self.bytes = total;
    }
}

/// The columns of [`memory_rows`].
pub const MEMORY_COLUMNS: &[&str] = &["Classic geometry", "Fallback", "Resident pages"];

/// Every virtual-geometry asset of every live runtime, for the Memory
/// Explorer: the GPU bytes of its full classic geometry (zero once only its
/// fallback is resident), of its fallback, and of its resident pages
/// (`VgRuntime::asset_memory`). Run at a tool's pace.
pub fn memory_rows() -> Vec<engine_backend::services::memory_owners::MemoryRow> {
    let mut rows = Vec::new();
    for runtime in VgRuntime::live() {
        for asset in runtime.asset_memory() {
            let label = if asset.report.name.is_empty() {
                format!("{:016x}{:016x}", asset.content[1], asset.content[0])
            } else {
                asset.report.name.clone()
            };
            rows.push(engine_backend::services::memory_owners::MemoryRow {
                label,
                bytes: vec![
                    asset.report.classic_bytes,
                    asset.report.fallback_bytes,
                    asset.page_bytes,
                ],
            });
        }
    }
    rows
}

/// The content a mesh asks virtual geometry for: its full geometry's key,
/// when `virtual_geometry` is set, its geometry is bound and it may be
/// virtual geometry at all.
fn wanted_content(mesh: &StaticMeshComponent) -> Option<u128> {
    let content = mesh.source_key().0;
    (mesh.virtual_geometry
        && content != 0
        && !mesh.source_vertices().is_empty()
        && mesh.source_indices().len() >= 3
        && is_vg_eligible(mesh))
    .then_some(content)
}

/// Whether a mesh may be virtual geometry. Blended (translucent) surfaces never
/// can: a mesh with a translucent surface slot stays classic. Alpha-TESTED
/// (masked) materials can: their rows carry `FLAG_ALPHA_TEST`, the cull routes
/// those clusters to the masked bins and the raster tiers evaluate the alpha
/// test (plan section 12, "Masked materials"). Masked-ness lives in the scene
/// material row, not in the surface, so nothing here needs to know about it.
/// A deforming (`Dynamic`) mesh cannot either: its clusters are built once
/// from geometry that would move on.
fn is_vg_eligible(mesh: &StaticMeshComponent) -> bool {
    !helio::Movability::from(mesh.movability).can_deform()
        && mesh.material_slots.slots.iter().all(|slot| {
            let surface = slot
                .surface_override
                .as_ref()
                .unwrap_or(&slot.imported_surface);
            surface.alpha >= 0.999
        })
}

/// The key a fallback of `content` cut at `error` is interned under: a
/// stable mix of both, never zero and never the content's own.
fn fallback_key(content: u128, error: f32) -> u128 {
    let mix = |mut z: u64| {
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let tag = u64::from(error.to_bits()) ^ 0x7667_6661_6c6c_6261; // "vgfallba"
    let low = mix(content as u64 ^ tag);
    let high = mix((content >> 64) as u64 ^ tag.rotate_left(17));
    let key = (u128::from(high) << 64) | u128::from(low);
    if key == 0 || key == content { key ^ 1 } else { key }
}

/// The fallback geometry of `lod` (a [`ProxyOptions::fallback`] cut) for a
/// mesh whose full sections are `full_sections`: classic vertices from the
/// cut's attribute records, its triangles by section in the full mesh's
/// section order (a section the cut has no triangles of is empty), so each
/// section keeps its material slot and its object row.
fn fallback_geometry(
    key: u128,
    lod: &helio_vg_stream::ProxyLodCpu,
    full_sections: &[MeshSection],
) -> MeshGeometry {
    let vertices = lod
        .positions
        .iter()
        .zip(&lod.records)
        .map(|(position, record)| {
            let attr = helio_vg_format::VertexAttr::decode(record).unwrap_or(
                helio_vg_format::VertexAttr {
                    normal: [0.0, 0.0, 1.0],
                    tangent: [1.0, 0.0, 0.0],
                    bitangent_sign: 1.0,
                    uv0: [0.0; 2],
                    uv1: [0.0; 2],
                },
            );
            let mut vertex = PackedVertex::from_components(
                *position,
                attr.normal,
                attr.uv0,
                attr.tangent,
                attr.bitangent_sign,
            );
            vertex.tex_coords1 = attr.uv1;
            vertex
        })
        .collect();
    // Cluster material slots are section ordinals (the build's
    // `tri_material`); a mesh without sections has one implicit section.
    let sections = full_sections
        .iter()
        .enumerate()
        .map(|(ordinal, section)| {
            let part = lod.sections.iter().find(|p| p.material == ordinal as u32);
            MeshSection {
                first_index: part.map_or(0, |p| p.first_index),
                index_count: part.map_or(0, |p| p.index_count),
                material_slot: section.material_slot,
            }
        })
        .collect();
    MeshGeometry {
        key: MeshGeometryKey(key),
        vertices,
        indices: lod.indices.clone(),
        sections,
    }
}

/// Cut `content`'s fallback as a background task: read the mesh asset's
/// error bound (its settings, else `project_error`), and unless it equals
/// `current` (the bound of the fallback in place), cut the `.vgeo` there.
fn spawn_cut(
    content: u128,
    mesh: &StaticMeshComponent,
    bytes: Arc<VgeoBytes>,
    project_error: f32,
    current: Option<f32>,
) -> mpsc::Receiver<CutResult> {
    let (tx, rx) = mpsc::channel();
    let name = display_name(&mesh.mesh_asset, content);
    let asset = mesh.mesh_asset.clone();
    let sections = mesh.source_sections().to_vec();
    helio_tasks::submit(
        helio_tasks::JobSpec::new(format!("Cut virtual geometry fallback: {name}"))
            .owner("meshes"),
        move |task| {
            task.set_phase("Reading the mesh's settings");
            let (lo, hi) = crate::vg_settings::VirtualGeometrySettings::FALLBACK_RELATIVE_ERROR_RANGE;
            let error = crate::components::mesh_asset_settings(&asset)
                .vg_fallback_error
                .filter(|e| e.is_finite())
                .unwrap_or(project_error)
                .clamp(lo, hi);
            if current == Some(error) {
                let _ = tx.send(CutResult::Unchanged);
                return;
            }
            task.set_phase("Cutting the fallback");
            let result = helio_vg_stream::build_proxy(bytes.as_ref(), &ProxyOptions::fallback(error), None)
                .and_then(|proxy| {
                    let lod = proxy
                        .lods
                        .first()
                        .ok_or_else(|| "the cut is empty".to_string())?;
                    tracing::info!(
                        "virtual geometry fallback for {name}: {} triangles of {} at error {:.4} ({:.2} % of its bounding radius {:.3})",
                        lod.triangle_count(),
                        proxy.source_triangles,
                        lod.error,
                        error * 100.0,
                        proxy.radius,
                    );
                    Ok(fallback_geometry(fallback_key(content, error), lod, &sections))
                });
            let _ = tx.send(match result {
                Ok(geometry) => CutResult::Cut(Arc::new(geometry), error),
                Err(message) => {
                    tracing::error!(
                        "the virtual geometry fallback of {name} could not be cut ({message}); its instances keep their full geometry"
                    );
                    CutResult::Failed
                }
            });
        },
    );
    rx
}


/// Where a mesh's `.vgeo` is kept: embedded in its native `.mesh`, or as
/// derived data in the project cache (module doc).
enum Store {
    Embedded(PathBuf),
    Cache(PathBuf),
    /// No project to keep it in: built each run.
    None,
}

fn store_for(mesh: &MeshAssetPath, content: u128) -> Store {
    let Some(root) = engine_state::get_project_path() else {
        return Store::None;
    };
    if mesh.generated_content_id().is_none() && !mesh.as_str().trim().is_empty() {
        let source = PathBuf::from(crate::components::mesh_source(&root, mesh.as_str()));
        if source.extension().and_then(|e| e.to_str()) == Some("mesh") {
            return Store::Embedded(source);
        }
    }
    Store::Cache(
        Path::new(&root)
            .join(".pulsar")
            .join("cache")
            .join("vgeo")
            .join(format!("{content:032x}.v{FORMAT_VERSION}.vgeo")),
    )
}

/// The name a build task shows for `mesh`.
fn display_name(mesh: &MeshAssetPath, content: u128) -> String {
    if mesh.generated_content_id().is_some() || mesh.as_str().trim().is_empty() {
        return format!("generated mesh {:08x}", (content >> 96) as u32);
    }
    Path::new(mesh.as_str())
        .file_name()
        .map_or_else(|| mesh.as_str().to_owned(), |n| n.to_string_lossy().into_owned())
}

/// The `.vgeo` attribute records of `vertices` (`helio_vg_format::attr_record`).
fn attribute_bytes(vertices: &[PackedVertex]) -> Vec<u8> {
    vertices
        .iter()
        .flat_map(|v| {
            helio_vg_format::attr_record(
                v.normal,
                v.tangent,
                v.bitangent_sign,
                v.tex_coords0,
                v.tex_coords1,
            )
        })
        .collect()
}

/// Load or build the mesh's `.vgeo` as a background task and keep it where
/// [`store_for`] says.
fn spawn_build(
    content: u128,
    mesh: &StaticMeshComponent,
) -> mpsc::Receiver<Result<Arc<VgeoBytes>, String>> {
    let (tx, rx) = mpsc::channel();
    let store = store_for(&mesh.mesh_asset, content);
    let name = display_name(&mesh.mesh_asset, content);
    let vertices = mesh.source_vertices().to_vec();
    let indices = mesh.source_indices().to_vec();
    let sections: Vec<(u32, u32)> = mesh
        .source_sections()
        .iter()
        .map(|s| (s.first_index, s.index_count))
        .collect();
    helio_tasks::submit(
        helio_tasks::JobSpec::new(format!("Build virtual geometry: {name}")).owner("meshes"),
        move |task| {
            let result = build_or_load(task, content, &store, &vertices, &indices, &sections);
            if let Err(error) = &result {
                tracing::error!("virtual geometry for {name} could not be built: {error}");
            }
            let _ = tx.send(result);
        },
    );
    rx
}

/// The stored `.vgeo` for `content`, if it is current (same content id and
/// cook revision).
fn load_stored(store: &Store, content: u128) -> Option<VgeoBytes> {
    let bytes = match store {
        Store::Embedded(path) => {
            let file = engine_fs::virtual_fs::read_file(path).ok()?;
            crate::mesh_cache::embedded_vgeo(&file)?.to_vec()
        }
        Store::Cache(path) => engine_fs::virtual_fs::read_file(path).ok()?,
        Store::None => return None,
    };
    let loaded = VgeoBytes::new(bytes).ok()?;
    (loaded.header().content_id == content_words(content)
        && helio_vg_build::is_current_cook(&loaded.header()))
    .then_some(loaded)
}

fn build_or_load(
    task: &helio_tasks::TaskContext,
    content: u128,
    store: &Store,
    vertices: &[PackedVertex],
    indices: &[u32],
    sections: &[(u32, u32)],
) -> Result<Arc<VgeoBytes>, String> {
    task.set_phase("Reading stored cluster data");
    if let Some(loaded) = load_stored(store, content) {
        tracing::info!("virtual geometry: using the stored cluster data");
        return Ok(Arc::new(loaded));
    }

    task.set_phase("Building the cluster hierarchy");
    let started = std::time::Instant::now();
    let positions: Vec<[f32; 3]> = vertices.iter().map(|v| v.position).collect();
    let attributes = attribute_bytes(vertices);
    // One material section per `mesh_sections` entry: the scene join emits one
    // `static_objects` row per section in this order.
    let mut tri_material = Vec::new();
    if !sections.is_empty() {
        tri_material = vec![0u32; indices.len() / 3];
        for (section, &(first, count)) in sections.iter().enumerate() {
            let first_tri = (first / 3) as usize;
            let tris = (count / 3) as usize;
            for tri in tri_material.iter_mut().skip(first_tri).take(tris) {
                *tri = section as u32;
            }
        }
    }
    let input = helio_vg_build::MeshInput {
        positions: &positions,
        attributes: &attributes,
        attribute_stride: helio_vg_format::ATTR_STRIDE as usize,
        indices: &indices[..indices.len() / 3 * 3],
        tri_material: &tri_material,
    };
    let options = helio_vg_build::BuildOptions {
        content_id: content_words(content),
        resident_levels: 2,
    };
    let bytes = helio_vg_build::build_vgeo(&input, &options).map_err(|e| e.to_string())?;
    tracing::info!(
        "virtual geometry: built {} triangles into {} KiB in {:.1} s",
        indices.len() / 3,
        bytes.len() / 1024,
        started.elapsed().as_secs_f64()
    );

    task.set_phase("Saving the cluster data");
    let saved = match store {
        // Keeps the classic geometry and material payload, replaces the file
        // atomically.
        Store::Embedded(path) => crate::mesh_cache::write_embedded_vgeo(path, content, &bytes)
            .map(|()| path.as_path()),
        // `write_file` creates the parent folders.
        Store::Cache(path) => engine_fs::virtual_fs::write_file(path, &bytes)
            .map(|()| path.as_path())
            .map_err(|e| e.to_string()),
        Store::None => Err("no project is open".into()),
    };
    match saved {
        Ok(path) => tracing::info!("virtual geometry: saved to {}", path.display()),
        // The data is still used this run (a read-only project, a mesh that
        // changed on disk meanwhile); it is built again next time.
        Err(error) => tracing::warn!("virtual geometry: not saved ({error}); built again next run"),
    }
    VgeoBytes::new(bytes).map(Arc::new)
}

/// Publishes `vg_asset_ready` from the runtime (sorted by interned vertex
/// offset, padded; see `virtual_geometry_instances`).
pub struct VgAssetReadyPublisher {
    runtime: VgRuntime,
    output: Output,
    last_version: Option<u64>,
}

impl VgAssetReadyPublisher {
    pub fn new(device: &wgpu::Device, runtime: VgRuntime) -> Self {
        Self {
            runtime,
            output: Output::new(device, "VG Asset Ready", VG_ASSET_READY_ROW_BYTES),
            last_version: None,
        }
    }
}

impl SceneDerivation for VgAssetReadyPublisher {
    fn name(&self) -> &'static str {
        "VgAssetReadyPublisher"
    }

    fn derive(
        &mut self,
        ctx: &SceneDerivationContext<'_>,
        _encoder: &mut wgpu::CommandEncoder,
    ) -> SceneDerivationOutput {
        if self.runtime.settings().mode != helio_vg_format::VgMode::Off {
            // Creates the store on first use and registers requested assets.
            self.runtime.service(ctx.device, ctx.queue);
        }
        let (rows, version) = self.runtime.asset_ready_rows(1);
        let grew = self.output.ensure(ctx.device, rows.len() as u32);
        if grew || self.last_version != Some(version) {
            self.last_version = Some(version);
            let mut padded: Vec<VgAssetReadyRow> = rows;
            padded.resize(self.output.rows as usize, VgAssetReadyRow::PADDING);
            ctx.queue
                .write_buffer(&self.output.buffer, 0, bytemuck::cast_slice(&padded));
            self.output.content_generation += 1;
        }
        SceneDerivationOutput {
            buffers: vec![(VG_ASSET_READY_KEY, self.output.handle())],
            flushed: Vec::new(),
            recorded: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vg_mesh(content: u128, on: bool) -> StaticMeshComponent {
        StaticMeshComponent {
            virtual_geometry: on,
            geometry_key: MeshGeometryKey(content),
            vertices: vec![PackedVertex::default(); 3],
            indices: vec![0, 1, 2],
            ..Default::default()
        }
    }

    /// The instances the host builds virtual geometry for, by content.
    fn wanted(host: &VirtualGeometryHost) -> HashMap<Entity, u128> {
        host.instances
            .iter()
            .filter(|(_, instance)| instance.wanted)
            .map(|(entity, instance)| (*entity, instance.content))
            .collect()
    }

    /// A fallback of `content` with `vertices` vertices in two sections.
    fn fallback(content: u128, vertices: usize) -> Arc<MeshGeometry> {
        Arc::new(MeshGeometry {
            key: MeshGeometryKey(fallback_key(content, 0.01)),
            vertices: vec![PackedVertex::default(); vertices],
            indices: vec![0, 1, 2],
            sections: vec![MeshSection {
                first_index: 0,
                index_count: 3,
                material_slot: 0,
            }],
        })
    }

    /// Fallbacks are shown only while every renderer of a world draws
    /// virtual geometry; a renderer's vote leaves with it, and votes of
    /// another world do not count.
    #[test]
    fn the_renderers_of_a_world_agree_on_fallbacks() {
        let world = World::new();
        let other_world = World::new();
        let mut vg = WorldVote::new();
        let mut classic = WorldVote::new();
        let mut elsewhere = WorldVote::new();
        assert!(vg.vote(&world, true), "a lone VG renderer shows fallbacks");
        assert!(!elsewhere.vote(&other_world, false));
        assert!(vg.vote(&world, true), "another world's renderer has no say");
        assert!(!classic.vote(&world, false));
        assert!(!vg.vote(&world, true), "a classic renderer of the world vetoes");
        drop(classic);
        assert!(vg.vote(&world, true), "the vote left with its renderer");
        // Drawing another world moves the vote.
        let mut moving = WorldVote::new();
        assert!(!moving.vote(&world, false));
        assert!(!moving.vote(&other_world, true), "the other world's classic renderer vetoes");
        assert!(vg.vote(&world, true));
    }

    /// The swap and its way back: a generated mesh's instances show the
    /// fallback (its key, so the full geometry's interned GPU ranges lose
    /// their references), share one CPU copy of the full geometry, and show
    /// it again, as it was, when they stop using virtual geometry. Their
    /// asset identity never changes.
    #[test]
    fn instances_swap_to_the_fallback_and_back() {
        let mut world = World::new();
        let full = StaticMeshComponent {
            vertices: vec![PackedVertex::default(); 9],
            indices: (0..9).collect(),
            mesh_sections: vec![MeshSection {
                first_index: 0,
                index_count: 9,
                material_slot: 0,
            }],
            ..vg_mesh(7, true)
        };
        let a = world.spawn();
        world.insert(a, full.clone());
        let b = world.spawn();
        world.insert(b, full.clone());
        let mut host = VirtualGeometryHost::new();
        host.follow(&world);
        let cut = fallback(7, 3);
        host.swaps = vec![
            (a, Swap::Fallback(7, cut.clone())),
            (b, Swap::Fallback(7, cut.clone())),
        ];
        host.apply(&mut world);
        for entity in [a, b] {
            let mesh = world.get::<StaticMeshComponent>(entity).unwrap();
            assert_eq!(mesh.geometry_key, cut.key);
            assert_eq!(mesh.vertices.len(), 3);
            assert_eq!(mesh.source_key(), MeshGeometryKey(7));
            assert_eq!(mesh.source_vertices().len(), 9);
            assert_eq!(
                crate::components::StaticMeshDraw::of(&mesh).vg_asset,
                [7, 0, 0, 0],
                "the asset identity is the full geometry's"
            );
        }
        let shared = |entity| match &world
            .get::<StaticMeshComponent>(entity)
            .unwrap()
            .full_geometry
        {
            Some(FullGeometry::Owned(geometry)) => Arc::as_ptr(geometry),
            other => panic!("{other:?}"),
        };
        assert_eq!(shared(a), shared(b), "one CPU copy of the full geometry");

        // The host sees its own writes as showing a fallback.
        host.follow(&world);
        assert!(host.instances[&a].showing_fallback);
        assert_eq!(host.instances[&a].content, 7);

        host.swaps = vec![(a, Swap::Full)];
        host.apply(&mut world);
        let mesh = world.get::<StaticMeshComponent>(a).unwrap();
        assert_eq!(mesh.geometry_key, MeshGeometryKey(7));
        assert_eq!(mesh.vertices.len(), 9);
        assert_eq!(mesh.mesh_sections, full.mesh_sections);
        assert!(mesh.full_geometry.is_none());
    }

    /// The fallback keeps the full mesh's sections in order (an ordinal the
    /// cut has no triangles of becomes an empty section), each with its own
    /// material slot, and its vertices take the records' attributes.
    #[test]
    fn a_fallback_keeps_every_section_and_its_material() {
        let record = helio_vg_format::VertexAttr {
            normal: [0.0, 1.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
            bitangent_sign: -1.0,
            uv0: [0.25, 0.5],
            uv1: [0.75, 1.0],
        }
        .encode();
        let lod = helio_vg_stream::ProxyLodCpu {
            error: 0.1,
            positions: vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            uvs: vec![[0.25, 0.5]; 3],
            records: vec![record; 3],
            indices: vec![0, 1, 2, 0, 2, 1],
            prim_map: vec![0, 1],
            clusters: vec![0],
            cluster_pages: vec![0],
            pages: vec![0],
            sections: vec![
                helio_vg_stream::ProxySection {
                    material: 0,
                    first_index: 0,
                    index_count: 3,
                },
                helio_vg_stream::ProxySection {
                    material: 2,
                    first_index: 3,
                    index_count: 3,
                },
            ],
        };
        let full = [5, 6, 7].map(|material_slot| MeshSection {
            first_index: 0,
            index_count: 30,
            material_slot,
        });
        let geometry = fallback_geometry(42, &lod, &full);
        assert_eq!(geometry.key, MeshGeometryKey(42));
        assert_eq!(
            geometry
                .sections
                .iter()
                .map(|s| (s.first_index, s.index_count, s.material_slot))
                .collect::<Vec<_>>(),
            [(0, 3, 5), (0, 0, 6), (3, 3, 7)]
        );
        let vertex = geometry.vertices[1];
        assert_eq!(vertex.position, [1.0, 0.0, 0.0]);
        assert_eq!(vertex.tex_coords0, [0.25, 0.5]);
        assert_eq!(vertex.tex_coords1, [0.75, 1.0]);
        assert_eq!(vertex.bitangent_sign, -1.0);
        assert_eq!((vertex.normal >> 8) & 0xff, 127, "normal +y");
    }

    /// A fallback's key depends on its content and error, is never the
    /// content's own and never zero.
    #[test]
    fn fallback_keys_are_distinct_per_content_and_error() {
        let a = fallback_key(7, 0.01);
        assert_eq!(a, fallback_key(7, 0.01));
        assert_ne!(a, fallback_key(7, 0.02));
        assert_ne!(a, fallback_key(8, 0.01));
        assert_ne!(a, 7);
        assert_ne!(fallback_key(0, 0.0), 0);
    }

    /// A deforming mesh stays classic.
    #[test]
    fn deforming_meshes_are_not_wanted() {
        let mut mesh = vg_mesh(7, true);
        mesh.movability = helio_component::components::ObjectMovability::Dynamic;
        assert_eq!(wanted_content(&mesh), None);
        mesh.movability = helio_component::components::ObjectMovability::Movable;
        assert_eq!(wanted_content(&mesh), Some(7));
    }

    /// The host follows the instances through the change journal: one scan,
    /// then only what changed (toggle off and on, a new instance, a removal,
    /// a mesh whose geometry is not bound yet).
    #[test]
    fn instances_follow_the_change_journal() {
        let mut world = World::new();
        let a = world.spawn();
        world.insert(a, vg_mesh(7, true));
        let classic = world.spawn();
        world.insert(classic, vg_mesh(8, false));

        let mut host = VirtualGeometryHost::new();
        host.follow(&world);
        assert_eq!(wanted(&host), HashMap::from([(a, 7)]));

        // Turned off, then a second instance of the same mesh and an
        // unbound one (no geometry yet) are placed.
        world.get_mut::<StaticMeshComponent>(a).unwrap().virtual_geometry = false;
        let b = world.spawn();
        world.insert(b, vg_mesh(7, true));
        let unbound = world.spawn();
        world.insert(unbound, vg_mesh(0, true));
        host.follow(&world);
        assert_eq!(wanted(&host), HashMap::from([(b, 7)]));

        // The unbound mesh's geometry arrives (a journaled write); `b` goes.
        {
            let mut mesh = world.get_mut::<StaticMeshComponent>(unbound).unwrap();
            mesh.geometry_key = MeshGeometryKey(9);
        }
        world.remove::<StaticMeshComponent>(b);
        host.follow(&world);
        assert_eq!(wanted(&host), HashMap::from([(unbound, 9)]));

        // Nothing changed: nothing to do.
        host.follow(&world);
        assert!(host.changes.is_empty());
    }

    /// A translucent slot keeps the mesh classic.
    #[test]
    fn translucent_meshes_are_not_wanted() {
        let mut mesh = vg_mesh(7, true);
        assert_eq!(wanted_content(&mesh), Some(7));
        let mut slot = crate::components::StaticMeshMaterialSlot::default();
        slot.imported_surface.alpha = 0.5;
        mesh.material_slots.slots.push(slot);
        assert_eq!(wanted_content(&mesh), None);
    }
}
