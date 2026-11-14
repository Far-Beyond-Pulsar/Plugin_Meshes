//! The meshes' SceneDB components and ray-tracing projection for standalone
//! Helio hosts (the plugin's examples, its tests, the web and Android demos):
//! what `helio-components` is for the kinds still in Helio. Engines register
//! their own components (Pulsar's mesh components) and run the derivations
//! the meshes' feature crates provide.
//!
//! - [`MeshComponent`], [`StaticObjectComponent`], [`RenderGroupComponent`]:
//!   the rows the meshes' passes read ([`register_columns`]).
//! - [`ray_tracing`]: the frame's TLAS from these rows, published as the
//!   frame's ray-traced scene (`helio_frame_api::rays::SCENE_RAYS`);
//!   [`ray_proxies`]: virtual-geometry instances traced through proxies.
//! - [`features`]: the meshes' Helio features, for a host's preset
//!   (`helio_presets::Preset::with_features`).
//! - Imported scenes' materials ([`upload_scene_materials`]).

pub mod ray_proxies;
pub mod ray_tracing;
pub mod texture_means;
/// The meshes' row components (helio-mesh-api's `host_rows`, where
/// their rows are defined).
pub use helio_mesh_api::host_rows::{MeshComponent, RenderGroupComponent, StaticObjectComponent};

/// The meshes' Helio features: the object batch, its culling, the drawers
/// (G-buffer, shadow casters, forward, transparent) and virtual geometry.
pub fn features() -> helio_feature_api::FeatureSet {
    helio_feature_api::FeatureSet::new()
        .with(helio_pass_object_batch::ObjectBatchFeature)
        .with(helio_pass_occlusion_cull::MeshCulling)
        .with(helio_pass_gbuffer::GBufferDrawers)
        .with(helio_pass_shadow::ShadowDrawers)
        .with(helio_pass_forward_lit::ForwardDrawers)
        .with(helio_pass_transparent::TransparentDrawers)
        .with(helio_pass_vg::VgFeature)
}

/// Registers the meshes' columns (meshes, objects, render groups) with a
/// store before its mirror is attached: registering up front, not on first
/// insert, so a column's first rows are never lost.
pub fn register_columns(
    store: &mut pulsar_scenedb::gpu::SceneGpuStore,
    device: &std::sync::Arc<wgpu::Device>,
) {
    MeshComponent::register_gpu_columns_growable(store, 4096, device);
    StaticObjectComponent::register_gpu_columns_growable(store, 4096, device);
}

/// Spawn a SceneDB material row for every material in a loaded
/// [`helio_asset_compat::ConvertedScene`], returning one [`pulsar_scenedb::Entity`] per
/// `scene.materials[i]`, in the same order.
///
/// This is the direct SceneDB replacement for the old `Scene`-based
/// `upload_scene_materials`: since a material is now just a
/// `MaterialComponent` row, spawning one *is* the whole
/// job. Its old job also included uploading each material's textures into
/// the (now-deleted) `Scene`'s bindless texture table and patching the
/// resulting indices into the `GpuMaterial` before inserting it -- bindless
/// texture upload has no SceneDB-authored home yet, so `ConvertedMaterial::
/// textures` is not consulted here. `convert_material`'s own `GpuMaterial`
/// already carries `NO_TEXTURE` in every `tex_*` slot (texture info is
/// returned separately, in `textures`), so every spawned material carries
/// its real base color/roughness/metallic/emissive factors but none of its
/// texture maps.
pub fn upload_scene_materials(
    world: &mut pulsar_scenedb::World,
    scene: &helio_asset_compat::ConvertedScene,
) -> Vec<pulsar_scenedb::Entity> {
    scene
        .materials
        .iter()
        .map(|converted| {
            let entity = world.spawn();
            world.insert(
                entity,
                helio_mats::host_rows::MaterialComponent::from(helio_mats::MaterialRow::from(
                    converted.gpu,
                )),
            );
            entity
        })
        .collect()
}

/// Helio's presets over the meshes' features ([`features`]), the
/// materials' (`helio_pass_materials::MaterialsFeature`: the texture table
/// the drawers bind) and the lights' (`helio_light_host::features`: a
/// standalone host draws its meshes lit, as an engine with these plugins
/// does): what a standalone host builds a renderer with to draw meshes
/// (`RendererBuilder::with_pass_build_context(Box::new(presets::deferred))`).
pub mod presets {
    use helio::PassBuildContext;
    use helio_core::RenderGraph;
    use helio_presets::{Preset, PresetGraph};

    /// The meshes', the materials' and the lights' features.
    fn lit() -> helio_feature_api::FeatureSet {
        super::features()
            .with(helio_pass_materials::MaterialsFeature)
            .extended(&helio_light_host::features())
    }

    /// `preset` over the meshes', the materials' and the lights' features.
    pub fn preset(preset: Preset) -> PresetGraph {
        preset.with_features(&lit())
    }

    /// `preset` over the meshes', the materials' and the lights' features and `extra`
    /// (another plugin's).
    pub fn with(preset: Preset, extra: &helio_feature_api::FeatureSet) -> PresetGraph {
        preset.with_features(&lit().extended(extra))
    }

    pub fn deferred(ctx: PassBuildContext<'_>) -> RenderGraph {
        preset(Preset::Deferred).build(ctx)
    }

    pub fn deferred_fxaa(ctx: PassBuildContext<'_>) -> RenderGraph {
        preset(Preset::DeferredFxaa).build(ctx)
    }

    pub fn hlfs(ctx: PassBuildContext<'_>) -> RenderGraph {
        preset(Preset::Model(helio_light_host::HLFS)).build(ctx)
    }

    pub fn hlfs_fxaa(ctx: PassBuildContext<'_>) -> RenderGraph {
        preset(Preset::Model(helio_light_host::HLFS_FXAA)).build(ctx)
    }

    pub fn forward(ctx: PassBuildContext<'_>) -> RenderGraph {
        preset(Preset::Forward).build(ctx)
    }
}
