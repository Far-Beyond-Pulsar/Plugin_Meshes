//! Meshes (`pulsar.meshes`): the component crate of the plugin that owns
//! meshes, their virtual geometry and their asset import.
//!
//! - The SceneDB components ([`components`]): `StaticMeshComponent` and its
//!   derived draw rows (`StaticMeshDraw`), `SkeletalMeshComponent`,
//!   `LODComponent`, `VirtualGeometryComponent`; mesh assets ([`mesh_cache`],
//!   [`mesh_files`]) and their slots' materials ([`mesh_materials`], through
//!   the materials' api only).
//! - What the engine runs for them, each through a generic registration
//!   (the engine names no mesh): the scene join and its source columns
//!   (`ObjectJoinRegistration`), the virtual-geometry host and the authored
//!   LOD selection ([`host_part`], a `RenderHostPartRegistration`), picking,
//!   framing and debug geometry (`ContentGeometryRegistration`), animated surfaces (`AnimatedContentRegistration`),
//!   movability, the skinned meshes an animator drives
//!   (`helio_component::skinning::SkinnedMeshAccess`), the headless report,
//!   the virtual-geometry project settings ([`vg_settings`]) and the
//!   mesh asset reloads.
//! - The plugin's Helio features: the object batch (first in the frame's
//!   prepare slot), its culling around the depth pyramid, the drawers (the
//!   G-buffer, shadow-caster, forward and transparent phases) and virtual
//!   geometry.
//!
//! This crate registers nothing itself: the plugin's root crate
//! (`meshes_plugin`) submits every registration above, in one place.

pub mod authored_lod;
pub mod components;
pub mod host_part;
pub mod import;
pub mod mesh_cache;
pub mod mesh_files;
pub mod mesh_materials;
pub mod mesh_spatial;
pub mod records;
pub mod vg_settings;
pub mod virtual_geometry_host;
pub mod virtual_geometry_instances;

pub use components::*;

use engine_backend::scene::ObjectJoinContext;
use helio_pass_object_batch::scene_join::{SceneJoin, SceneJoinKeys};
use pulsar_scenedb::gpu::BufferKey;
use pulsar_scenedb::{Entity, World};

/// The meshes' features.
pub fn render_features() -> helio_feature_api::FeatureSet {
    helio_feature_api::FeatureSet::new()
        .with(helio_pass_object_batch::ObjectBatchFeature)
        .with(helio_pass_occlusion_cull::MeshCulling)
        .with(helio_pass_gbuffer::GBufferDrawers)
        .with(helio_pass_shadow::ShadowDrawers)
        .with(helio_pass_forward_lit::ForwardDrawers)
        .with(helio_pass_transparent::TransparentDrawers)
        .with(helio_pass_vg::VgFeature)
}

/// The meshes' features read no project settings through the renderer
/// config: virtual geometry's come in with its runtime ([`host_part`]),
/// shadows' with the lights' `ShadowSettings`.
///
/// The one setting it adds is for the render graph pin probe: the virtual
/// geometry node builds its chain only for a runtime, so a config that has
/// none (the probe's, made from the plugins' settings alone) gets an idle
/// placeholder runtime, and the node's pins are those of the chain. A
/// renderer's config has the host part's runtime before this runs
/// ([`host_part::MeshesHostPart`] configures first), and a setting already
/// there is left alone.
pub fn configure_renderer(mut config: helio::RendererConfig) -> helio::RendererConfig {
    if config.features.get::<helio_vg_stream::VgSettings>().is_none() {
        config.features.insert(helio_vg_stream::VgSettings::for_runtime(
            probe_runtime(),
        ));
    }
    config
}

/// The idle runtime a config without one gets ([`configure_renderer`]). The
/// runtimes are found by id while one is alive, so one is kept.
fn probe_runtime() -> &'static helio_vg_stream::VgRuntime {
    static RUNTIME: std::sync::OnceLock<helio_vg_stream::VgRuntime> = std::sync::OnceLock::new();
    RUNTIME.get_or_init(helio_vg_stream::VgRuntime::default)
}

// ── The scene join ──────────────────────────────────────────────────────────

/// Where the meshes' authored rows live, for the scene join: the engine's
/// object rows (owner links, generations, visibility, transforms) and the
/// mesh geometry handles and derived draw rows this crate's components
/// register (`StaticMeshComponent`, `StaticMeshDraw`, the skeletal skin).
pub fn scene_join_keys(context: &ObjectJoinContext) -> SceneJoinKeys {
    // A var-len field's handle table is its pool key + `::handles`.
    SceneJoinKeys {
        owners: context.owners,
        generations: context.generations,
        hidden: context.hidden,
        transforms: context.transforms,
        vertex_handles: BufferKey::of("builtin_mesh_vertex::handles"),
        index_handles: BufferKey::of("builtin_mesh_index::handles"),
        mesh_bounds: BufferKey::of(MESH_BOUNDS_BUFFER),
        mesh_flags: BufferKey::of(MESH_FLAGS_BUFFER),
        section_handles: BufferKey::of("static_mesh_draw_sections::handles"),
        skin_handles: BufferKey::of(SKELETAL_SKIN_HANDLES_BUFFER),
        palette_handles: BufferKey::of(SKELETAL_PALETTE_HANDLES_BUFFER),
        mesh_sections: BufferKey::of(MESH_SECTIONS_BUFFER),
        material_row_handles: BufferKey::of("mesh_material_rows::handles"),
        material_rows: BufferKey::of(mesh_materials::MESH_MATERIAL_ROWS_BUFFER),
        material_keys: BufferKey::of(helio_mats::instance::MATERIAL_KEYS),
    }
}

/// Registers the SceneDB columns the scene join reads.
pub fn register_columns(
    store: &mut pulsar_scenedb::gpu::SceneGpuStore,
    device: &std::sync::Arc<wgpu::Device>,
) {
    // Capacities are initial capacities only; growable registration owns
    // GPU storage and deduplication.
    StaticMeshComponent::register_gpu_columns_growable(store, 4096, device);
    // Authored mesh LOD levels: source rows of the per-view LOD selection
    // (separate from virtual geometry's own LOD).
    LODComponent::register_gpu_columns_growable(store, 1024, device);
    // Per-instance virtual-geometry overrides, joined to mesh instances by
    // the VirtualGeometryInstances derivation (`vg_instance_overrides`).
    VirtualGeometryComponent::register_gpu_columns_growable(store, 256, device);
    SkeletalMeshComponent::register_gpu_columns_growable(store, 128, device);
    // The scene join's inputs; the object rows passes draw are the join's
    // output, not SceneDB's.
    StaticMeshDraw::register_gpu_columns_growable(store, 4096, device);
    // Each mesh's slots' rows of the material table, and the table itself:
    // the instances' rows.
    mesh_materials::MeshMaterialRows::register_gpu_columns_growable(store, 4096, device);
    helio_mats::instance::register_columns(store, device);
}

/// The meshes' scene join (`ObjectJoinRegistration`).
pub fn scene_join(
    device: &wgpu::Device,
    context: &ObjectJoinContext,
) -> Box<dyn helio_scene_join::SceneDerivation> {
    Box::new(SceneJoin::new(device, scene_join_keys(context)))
}

// ── Picking ─────────────────────────────────────────────────────────────────

/// The box around `object`'s enabled meshes' local bounding spheres.
pub fn mesh_local_bounds(world: &World, object: Entity) -> Option<(glam::Vec3, glam::Vec3)> {
    let meshes = pulsar_scene_model::attachments::enabled_components_of::<StaticMeshComponent>(
        world, object,
    );
    let mut local: Option<(glam::Vec3, glam::Vec3)> = None;
    for (_, mesh) in meshes {
        let [x, y, z, radius] = mesh.bounds_local;
        // No geometry loaded yet: nothing to frame.
        if !(radius > 0.0) {
            continue;
        }
        let (center, extent) = (glam::Vec3::new(x, y, z), glam::Vec3::splat(radius));
        let (min, max) = (center - extent, center + extent);
        local = Some(match local {
            Some((lo, hi)) => (lo.min(min), hi.max(max)),
            None => (min, max),
        });
    }
    local
}

/// A local ray against `object`'s enabled meshes' triangles.
pub fn mesh_hit(
    world: &World,
    object: Entity,
    origin: glam::Vec3,
    direction: glam::Vec3,
) -> Option<Option<f32>> {
    let meshes = pulsar_scene_model::attachments::enabled_components_of::<StaticMeshComponent>(
        world, object,
    );
    let has_geometry = meshes.iter().any(|(_, mesh)| {
        mesh.vertices.len() >= 3 && (mesh.indices.is_empty() || mesh.indices.len() >= 3)
    });
    if !has_geometry {
        return None;
    }
    Some(
        meshes
            .into_iter()
            .filter_map(|(_, mesh)| {
                if mesh.vertices.len() < 3 {
                    return None;
                }
                mesh_spatial::raycast_mesh(
                    mesh.geometry_key,
                    &mesh.vertices,
                    &mesh.indices,
                    origin,
                    direction,
                )
            })
            .filter(|t| *t >= 0.0 && t.is_finite())
            .min_by(f32::total_cmp),
    )
}

/// Every enabled mesh triangle in `world`, in its owner's local space.
pub fn mesh_triangles(world: &World, visit: &mut dyn FnMut(Entity, [glam::Vec3; 3]) -> bool) {
    for (_, owner, mesh) in
        pulsar_scene_model::attachments::enabled_components::<StaticMeshComponent>(world)
    {
        for triangle in mesh.indices.chunks_exact(3) {
            let corner = |index: u32| {
                mesh.vertices
                    .get(index as usize)
                    .map(|vertex| glam::Vec3::from_array(vertex.position))
            };
            let (Some(a), Some(b), Some(c)) = (
                corner(triangle[0]),
                corner(triangle[1]),
                corner(triangle[2]),
            ) else {
                continue;
            };
            if !visit(owner, [a, b, c]) {
                return;
            }
        }
    }
}

// ── What the root registers for the engine's hooks ──────────────────────────

/// Whether a mesh drawn with a shader graph that reads the frame clock is in
/// `world` (it animates). The answer changes when a mesh does or when an
/// instance binds its material.
/// What a headless run reports of the meshes: how many mesh components
/// there are and how many bound their geometry.
pub fn content_report(world: &World) -> serde_json::Value {
    let mut components = 0;
    let mut bound = 0;
    for (_, mesh) in world.query::<&StaticMeshComponent>() {
        components += 1;
        if mesh.geometry_key != MeshGeometryKey::NONE {
            bound += 1;
        }
    }
    serde_json::json!({ "components": components, "bound": bound })
}

pub fn meshes_read_time(world: &World) -> bool {
    world
        .query::<&StaticMeshComponent>()
        .filter(|(instance, _)| pulsar_scene_model::attachments::is_enabled(world, *instance))
        .any(|(entity, _)| {
            let instances = world.get::<mesh_materials::MeshMaterials>(entity);
            instances
                .into_iter()
                .flat_map(|held| held.0.iter())
                .any(|instance| {
                    world
                        .get::<helio_mats::instance::MaterialInstanceComponent>(*instance)
                        .is_some_and(|instance| instance.reads_time())
                })
        })
}

#[cfg(test)]
mod tests {
    /// The join reads these rows with fixed layouts; a drift here must fail
    /// a test, not draw garbage.
    #[test]
    fn the_rows_the_scene_join_reads_have_its_layouts() {
        use helio_pass_object_batch::scene_join as join;
        use pulsar_scene_model::attachments::ComponentOwner;
        use pulsar_scene_model::{ObjectHidden, TransformRow};
        use std::mem::size_of;
        assert_eq!(size_of::<ComponentOwner>() as u64, join::OWNER_ROW_BYTES);
        assert_eq!(size_of::<ObjectHidden>() as u64, join::HIDDEN_ROW_BYTES);
        assert_eq!(size_of::<TransformRow>() as u64, join::TRANSFORM_ROW_BYTES);
        assert_eq!(
            size_of::<pulsar_scenedb::gpu::VarLenHandle>() as u64,
            join::HANDLE_ROW_BYTES
        );
        assert_eq!(
            size_of::<super::MeshSectionDraw>() as u64,
            join::MESH_SECTION_ROW_BYTES
        );
        assert_eq!(size_of::<[f32; 4]>() as u64, join::OWNER_BOUNDS_ROW_BYTES);
        assert_eq!(size_of::<u32>() as u64, join::MESH_FLAGS_ROW_BYTES);
        assert_eq!(
            size_of::<helio_mats::instance::MaterialKeyComponent>() as u64,
            join::MATERIAL_KEY_ROW_BYTES
        );
    }
}
