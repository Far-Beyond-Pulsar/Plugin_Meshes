//! What the meshes plugin registers for a game (and the editor): every
//! runtime contribution, grouped by the engine seam it plugs into.

use engine_backend::scene::game_hooks::ContentReportRegistration;
use engine_backend::scene::ObjectJoinRegistration;
use engine_backend::scene::{ClassSpec, ComponentsBuilder};
use engine_backend::subsystems::render::helio_renderer::{
    ContentGeometryRegistration, RenderHostPartRegistration,
};
use engine_backend::subsystems::render::{AnimatedContentRegistration, RenderFeatureRegistration};
use helio_component::asset_component::{AssetComponentRegistration, AssetReloadRegistration};
use helio_component::components::MovabilityRegistration;
use helio_component::skinning::SkinnedMeshAccess;
use meshes_component::components::{self, SkeletalMeshComponent};
use meshes_component::mesh_materials::MeshMaterialInstances;
use meshes_component::{
    LODComponent, StaticMeshComponent, VirtualGeometryComponent, MESH_BOUNDS_BUFFER,
};
use plugin_editor_api::AssetKind;
use pulsar_scenedb::gpu::BufferKey;

/// Everything the plugin registers for a game (and the editor).
pub(crate) fn build(p: &mut pulsar_plugins::PluginBuilder) {
    // ── Component classes ───────────────────────────────────────────────────

    p.components()
        .class::<StaticMeshComponent>(ClassSpec::default())
        .class::<SkeletalMeshComponent>(ClassSpec::default())
        .class::<LODComponent>(ClassSpec::default())
        .class::<VirtualGeometryComponent>(ClassSpec::default());

    // ── Rendering ───────────────────────────────────────────────────────────

    // The object batch (first in the frame's prepare slot), its culling, the
    // G-buffer, shadow, forward and transparent drawers and virtual geometry.
    p.add(RenderFeatureRegistration {
        origin: p.origin(),
        features: meshes_component::render_features,
        configure: meshes_component::configure_renderer,
    });

    // The scene join: first, since the other joins and the row joins read the
    // object rows it writes.
    p.add(ObjectJoinRegistration {
        origin: p.origin(),
        name: "meshes",
        order: 0,
        owner_bounds: Some(BufferKey::of(MESH_BOUNDS_BUFFER)),
        register_columns: meshes_component::register_columns,
        derivation: meshes_component::scene_join,
    });

    // The virtual-geometry host and the authored LOD selection.
    p.add(RenderHostPartRegistration {
        origin: p.origin(),
        name: "meshes",
        new: || Box::new(meshes_component::host_part::MeshesHostPart::new()),
    });

    // Picking, framing and debug geometry over enabled meshes.
    p.add(ContentGeometryRegistration {
        origin: p.origin(),
        name: "meshes",
        local_bounds: meshes_component::mesh_local_bounds,
        hit: meshes_component::mesh_hit,
        triangles: meshes_component::mesh_triangles,
    });

    // A mesh drawn with a shader graph that reads the frame clock animates; the
    // answer changes when a mesh does or when an instance binds its material.
    p.add(AnimatedContentRegistration {
        origin: p.origin(),
        name: "meshes.mesh_changes",
        watch: |world| world.open_change_cursor::<StaticMeshComponent>(),
        animated: meshes_component::meshes_read_time,
    });

    p.add(AnimatedContentRegistration {
        origin: p.origin(),
        name: "meshes.material_instance_changes",
        watch: |world| {
            world.open_change_cursor::<helio_mats::instance::MaterialInstanceComponent>()
        },
        animated: meshes_component::meshes_read_time,
    });

    // ── Scene ───────────────────────────────────────────────────────────────

    // The material instances a mesh's slots hold.
    p.add(pulsar_world_registry::SceneSubsystemRegistration {
        origin: p.origin(),
        name: MeshMaterialInstances::NAME,
        install: |scene| scene.register_subsystem(MeshMaterialInstances::default()),
    });

    // What an object's mesh instances author fixes the object.
    p.add(MovabilityRegistration {
        origin: p.origin(),
        name: "meshes",
        of: |world, instance| {
            world
                .get::<StaticMeshComponent>(instance)
                .map(|mesh| mesh.movability)
        },
    });

    // An animator (the animation plugin's) finds the skinned mesh it drives and
    // shows its pose there (`helio_component::skinning`).
    p.add(SkinnedMeshAccess {
        origin: p.origin(),
        name: "meshes.skeletal",
        instance: |world, owner, ordinal| {
            pulsar_world_registry::instances::resolve_instance(
                world,
                owner,
                components::SKELETAL_MESH_COMPONENT,
                ordinal,
            )
        },
        binding: |world, instance| {
            world
                .get::<SkeletalMeshComponent>(instance)
                .map(|mesh| mesh.binding())
        },
        show_pose: |world, instance, bones, pose| {
            world
                .get_mut::<SkeletalMeshComponent>(instance)
                .map(|mut mesh| mesh.show_pose(bones, pose))
                .is_some()
        },
    });

    // A headless run reports the mesh components and how many bound geometry.
    p.add(ContentReportRegistration {
        origin: p.origin(),
        key: "meshes",
        report: meshes_component::content_report,
    });

    // The `project.virtual_geometry` settings.
    p.add(pulsar_settings::project::ProjectSettingsRegistration {
        origin: p.origin(),
        name: "meshes.virtual_geometry",
        register: meshes_component::vg_settings::register,
    });

    // ── Assets ──────────────────────────────────────────────────────────────

    // A mesh asset dropped in a level becomes a static mesh component.
    p.add(AssetComponentRegistration {
        origin: p.origin(),
        asset_kind: AssetKind::Mesh,
        class_name: "StaticMeshComponent",
        value_for: |path| Box::new(StaticMeshComponent::for_mesh_asset(path)),
    });

    // A changed mesh file reloads its mesh asset.
    p.add(AssetReloadRegistration {
        origin: p.origin(),
        name: "meshes",
        kind: AssetKind::Mesh,
        sources_for_file: |path| vec![meshes_component::mesh_source_for_file(path)],
    });

    // Models (FBX, glTF, OBJ, USD...) import as native mesh assets.
    p.add(asset_import::importer::ImporterRegistration {
        origin: p.origin(),
        importer: &meshes_component::import::ModelImporter,
    });

    // ── Level records ───────────────────────────────────────────────────────

    p.add(pulsar_class::records::RecordMigrationRegistration {
        origin: p.origin(),
        name: "meshes.fold_material_override",
        description: meshes_component::records::FOLDED,
        entries: Some(meshes_component::records::fold_material_override),
        from_props: None,
        stale_props: &[],
        retired_fields: &[],
    });

    p.add(pulsar_class::records::RecordMigrationRegistration {
        origin: p.origin(),
        name: "meshes.mesh_from_props",
        description: meshes_component::records::FROM_PROPS,
        entries: None,
        from_props: Some(meshes_component::records::mesh_from_props),
        stale_props: &["mesh_asset"],
        retired_fields: &[],
    });
}

#[cfg(test)]
mod tests {
    use engine_backend::subsystems::render::helio_renderer::render_features;

    #[test]
    fn registers_its_features() {
        let ids = render_features::plugin_features().ids();
        for id in [
            "helio.mesh.batch",
            "helio.mesh.cull",
            "helio.mesh.gbuffer",
            "helio.mesh.shadow",
        ] {
            assert!(ids.contains(&id), "{id} is missing: {ids:?}");
        }
    }

    /// Every registration of the plugin names this root, so it resolves to
    /// the plugin's id.
    #[test]
    fn every_registration_resolves_to_the_plugin() {
        let ours: Vec<_> = pulsar_plugins::contributions()
            .into_iter()
            .filter(|summary| summary.origin == pulsar_plugins::origin!())
            .collect();
        assert!(ours
            .iter()
            .any(|summary| summary.category == "component.class"));
        for summary in &ours {
            assert_eq!(
                pulsar_plugins::plugin_of(summary.origin).map(|plugin| plugin.id.as_str()),
                Some("pulsar.meshes"),
                "{summary:?}"
            );
        }
    }
}
