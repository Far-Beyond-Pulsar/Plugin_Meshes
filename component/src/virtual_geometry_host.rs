//! The CPU side of virtual geometry for a host (editor, game): which meshes
//! need a `.vgeo`, building and caching it as a background task, handing it
//! to the streaming runtime, and publishing which assets are ready to the GPU.
//!
//! ```text
//! host frame          VirtualGeometryHost::sync(world, mirror, settings)
//!                       - settings -> VgRuntime, VgInstanceControl
//!                       - StaticMeshComponent change journal -> the instances
//!                         with `virtual_geometry` set, by content
//!                       - per wanted content: load or build its `.vgeo`
//!                         (a `helio_tasks` job), then
//!                         VgRuntime::request_asset(content, interned vertex range)
//! scene derivations   VgAssetReadyPublisher  -> `vg_asset_ready` (rows from the runtime)
//!                     VirtualGeometryInstances -> `vg_instances`, `lod_suppressed`,
//!                                                 `classic_suppressed`
//! graph passes        stream pages, cull, raster, shade (helio-presets)
//! ```
//!
//! **Incremental.** The instances are followed through SceneDB's change
//! journal for `StaticMeshComponent` (one full scan when the cursor opens or
//! the journal overflows, as on a world swap), so a frame's work scales with
//! the virtual-geometry contents in use, never with every mesh in the world.
//! The binding of a mesh's geometry is a journaled write, so an instance is
//! seen again once its mesh has loaded.
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
//! and on again).
//!
//! Until an asset's root pages are resident its instances are not VG-active:
//! they render classically and their authored LOD applies (plan section 3,
//! item 7). Only static meshes take part; skinned meshes, voxels and foliage
//! are other components and stay classic. The mesh retains classic geometry
//! as a non-destructive fallback, and the world's ray-traced scene traces that
//! full geometry (`crate::ray_scene`).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc;

use helio_core::{BufferKey, SceneDerivation, SceneDerivationContext, SceneDerivationOutput};
use helio_mesh_api::PackedVertex;
use helio_vg_format::{FORMAT_VERSION, VgAssetReadyRow};
use helio_vg_stream::{VgRuntime, VgRuntimeSettings, VgeoBytes, VgeoSource, content_words};
use pulsar_scenedb::gpu::GpuMirrorHandle;
use pulsar_scenedb::handle_ledger::HandleId;
use pulsar_scenedb::{ChangeCursor, ChangeRead, ComponentChange, ComponentChangeKind, Entity, World};

use crate::components::{MeshAssetPath, StaticMeshComponent};
use crate::virtual_geometry_instances::{
    Output, VG_ASSET_READY_KEY, VG_ASSET_READY_ROW_BYTES, VgInstanceControl, VgInstanceKeys,
    VirtualGeometryInstances,
};

enum HostAsset {
    Building(mpsc::Receiver<Result<Arc<VgeoBytes>, String>>),
    Ready(Arc<VgeoBytes>),
    Failed,
}

/// See the module doc. Owned by whoever owns the renderer.
pub struct VirtualGeometryHost {
    runtime: VgRuntime,
    control: Option<VgInstanceControl>,
    /// Loaded, building or failed `.vgeo`s, by mesh content id.
    assets: HashMap<u128, HostAsset>,
    /// The instances with `virtual_geometry` set, and the content each shows.
    instances: HashMap<Entity, u128>,
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
            control: None,
            assets: HashMap::new(),
            instances: HashMap::new(),
            cursor: None,
            changes: Vec::new(),
        }
    }

    /// The streaming runtime, for `helio_pass_vg::with_virtual_geometry`.
    pub fn runtime(&self) -> &VgRuntime {
        &self.runtime
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
    /// instances' changes, collects finished builds, starts builds for
    /// contents that need one and offers the ready ones to the runtime.
    /// Cheap when nothing changed: its work is the contents in use.
    pub fn sync(&mut self, world: &World, mirror: &GpuMirrorHandle, settings: VgRuntimeSettings) {
        self.runtime.set_settings(settings);
        let on = settings.mode != helio_vg_format::VgMode::Off;
        if let Some(control) = &self.control {
            control.set_enabled(on);
        }
        self.follow(world);
        self.collect_builds();
        if !on {
            return;
        }

        // One instance per wanted content (the one a build reads).
        let mut wanted: HashMap<u128, Entity> = HashMap::new();
        for (&entity, &content) in &self.instances {
            wanted.entry(content).or_insert(entity);
        }
        // A failed content nobody asks for any more is forgotten, so asking
        // again retries it.
        self.assets.retain(|content, asset| {
            !matches!(asset, HostAsset::Failed) || wanted.contains_key(content)
        });

        let pool = mirror
            .store()
            .interned_var_len_pool::<PackedVertex>(BufferKey::of("builtin_mesh_vertex"));
        for (content, entity) in wanted {
            match self.assets.get(&content) {
                None => {
                    if let Some(mesh) = world.get::<StaticMeshComponent>(entity) {
                        self.assets
                            .insert(content, HostAsset::Building(spawn_build(content, mesh)));
                    }
                }
                // Ready assets are offered every frame: the interned vertex
                // range is what identifies the mesh on the GPU and can move.
                // The runtime ignores an unchanged offer.
                Some(HostAsset::Ready(bytes)) => {
                    if let Some((offset, count)) =
                        pool.as_ref().and_then(|pool| pool.resolve(HandleId(content)))
                    {
                        let source: Arc<dyn VgeoSource> = bytes.clone();
                        self.runtime.request_asset(content, offset, count, source);
                    }
                }
                Some(HostAsset::Building(_) | HostAsset::Failed) => {}
            }
        }
    }

    /// Bring [`Self::instances`] up to date from the change journal, or by a
    /// full scan when there is no cursor yet or the journal cannot say what
    /// changed (overflow, another world).
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
            for (entity, mesh) in world.query::<&StaticMeshComponent>() {
                if let Some(content) = wanted_content(mesh) {
                    self.instances.insert(entity, content);
                }
            }
            return;
        }
        for change in &self.changes {
            let content = match change.kind {
                ComponentChangeKind::Removed => None,
                ComponentChangeKind::Inserted | ComponentChangeKind::Mutated => world
                    .get::<StaticMeshComponent>(change.entity)
                    .and_then(wanted_content),
            };
            match content {
                Some(content) => {
                    self.instances.insert(change.entity, content);
                }
                None => {
                    self.instances.remove(&change.entity);
                }
            }
        }
    }

    /// Take the results of finished builds.
    fn collect_builds(&mut self) {
        for asset in self.assets.values_mut() {
            if let HostAsset::Building(rx) = asset {
                match rx.try_recv() {
                    Ok(Ok(bytes)) => *asset = HostAsset::Ready(bytes),
                    // The task logged why.
                    Ok(Err(_)) | Err(mpsc::TryRecvError::Disconnected) => {
                        *asset = HostAsset::Failed;
                    }
                    Err(mpsc::TryRecvError::Empty) => {}
                }
            }
        }
    }
}

/// The content a mesh asks virtual geometry for: its geometry key, when
/// `virtual_geometry` is set, its geometry is bound and it may be virtual
/// geometry at all.
fn wanted_content(mesh: &StaticMeshComponent) -> Option<u128> {
    let content = mesh.geometry_key.0;
    (mesh.virtual_geometry
        && content != 0
        && !mesh.vertices.is_empty()
        && mesh.indices.len() >= 3
        && is_vg_eligible(mesh))
    .then_some(content)
}

/// Whether a mesh may be virtual geometry. Blended (translucent) surfaces never
/// can: a mesh with a translucent surface slot stays classic. Alpha-TESTED
/// (masked) materials can: their rows carry `FLAG_ALPHA_TEST`, the cull routes
/// those clusters to the masked bins and the raster tiers evaluate the alpha
/// test (plan section 12, "Masked materials"). Masked-ness lives in the scene
/// material row, not in the surface, so nothing here needs to know about it.
fn is_vg_eligible(mesh: &StaticMeshComponent) -> bool {
    mesh.material_slots.slots.iter().all(|slot| {
        let surface = slot
            .surface_override
            .as_ref()
            .unwrap_or(&slot.imported_surface);
        surface.alpha >= 0.999
    })
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

fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x007f_ffff;
    if exponent == 0xff {
        return sign | 0x7c00 | if mantissa != 0 { 0x0200 } else { 0 };
    }
    let unbiased = exponent - 127 + 15;
    if unbiased >= 0x1f {
        return sign | 0x7c00;
    }
    if unbiased <= 0 {
        if unbiased < -10 {
            return sign;
        }
        let m = mantissa | 0x0080_0000;
        let shift = (14 - unbiased) as u32;
        let half = (m >> shift) as u16;
        let round = ((m >> (shift - 1)) & 1) as u16;
        return sign | (half + round);
    }
    let half = sign | ((unbiased as u16) << 10) | (mantissa >> 13) as u16;
    // Round to nearest.
    if mantissa & 0x1000 != 0 {
        half.wrapping_add(1)
    } else {
        half
    }
}

/// The 16-byte attribute record of the `.vgeo` (`helio_vg_format::ATTR_STRIDE`):
/// normal snorm8x4, tangent snorm8x4 (w = bitangent sign), uv0 f16x2, uv1 f16x2.
fn attribute_bytes(vertices: &[PackedVertex]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vertices.len() * 16);
    for v in vertices {
        let tangent = (v.tangent & 0x00ff_ffff)
            | (if v.bitangent_sign < 0.0 {
                0x81u32
            } else {
                0x7fu32
            } << 24);
        out.extend_from_slice(&v.normal.to_le_bytes());
        out.extend_from_slice(&tangent.to_le_bytes());
        for uv in [v.tex_coords0, v.tex_coords1] {
            out.extend_from_slice(&f32_to_f16_bits(uv[0]).to_le_bytes());
            out.extend_from_slice(&f32_to_f16_bits(uv[1]).to_le_bytes());
        }
    }
    out
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
    let vertices = mesh.vertices.clone();
    let indices = mesh.indices.clone();
    let sections: Vec<(u32, u32)> = mesh
        .mesh_sections
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
    use crate::components::MeshGeometryKey;

    fn vg_mesh(content: u128, on: bool) -> StaticMeshComponent {
        StaticMeshComponent {
            virtual_geometry: on,
            geometry_key: MeshGeometryKey(content),
            vertices: vec![PackedVertex::default(); 3],
            indices: vec![0, 1, 2],
            ..Default::default()
        }
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
        assert_eq!(host.instances, HashMap::from([(a, 7)]));

        // Turned off, then a second instance of the same mesh and an
        // unbound one (no geometry yet) are placed.
        world.get_mut::<StaticMeshComponent>(a).unwrap().virtual_geometry = false;
        let b = world.spawn();
        world.insert(b, vg_mesh(7, true));
        let unbound = world.spawn();
        world.insert(unbound, vg_mesh(0, true));
        host.follow(&world);
        assert_eq!(host.instances, HashMap::from([(b, 7)]));

        // The unbound mesh's geometry arrives (a journaled write); `b` goes.
        {
            let mut mesh = world.get_mut::<StaticMeshComponent>(unbound).unwrap();
            mesh.geometry_key = MeshGeometryKey(9);
        }
        world.remove::<StaticMeshComponent>(b);
        host.follow(&world);
        assert_eq!(host.instances, HashMap::from([(unbound, 9)]));

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
