//! The meshes' G-buffer drawers: the opener (it opens the frame's G-buffer
//! phase, clearing the layout, `helio_frame_api::gbuffer`, and draws the
//! culled opaque batches into it) and the same program drawing the meshes'
//! occurrences through the frame's portal chains (`helio_portal_api`).

use helio_feature_api::{pass_node, Feature, FeatureRegistrar};
use helio_frame_api::{FrameNode, Phase};

use crate::GBufferPass;

/// Meshes in the G-buffer ([`crate`]).
pub struct GBufferDrawers;

impl Feature for GBufferDrawers {
    fn id(&self) -> &'static str {
        "helio.mesh.gbuffer"
    }

    fn register(&self, registrar: &mut FeatureRegistrar<'_>) {
        registrar.consumes(helio_mesh_api::MESH_INDICES);
        registrar.consumes(helio_mesh_api::MESH_VERTICES);
        registrar.consumes(helio_mats::MATERIALS);
        registrar.consumes(helio_light_api::SCENE_LIGHTS);
        registrar.consumes(helio_mesh_api::BONE_PALETTE);
        registrar.consumes(helio_mesh_api::SKIN_VERTICES);
        registrar.node(
            pass_node(
                helio_frame_api::gbuffer::GROUP,
                "Geometry: GBuffer",
                "Passes: Geometry",
                "Opaque meshes into the GBuffer and the scene depth; opens the G-buffer phase.",
                |build| {
                    // Baked lighting asked for each object's lightmap UV as the
                    // G-buffer's surface payload.
                    let lightmap = helio_frame_api::gbuffer::payload(build.env)
                        == Some(helio_bake_api::LIGHTMAP_UV);
                    let mut pass = GBufferPass::new(build.env.device)
                        .with_lightmap_payload(lightmap)
                        .with_view(helio_frame_api::view::view_of(build.env));
                    if let Some(templates) = build.env.setting::<helio_mats::MaterialTemplates>() {
                        pass = pass.with_template_registry(templates.opaque.clone());
                    }
                    build.graph.add_pass(Box::new(pass));
                    Ok(())
                },
            )
            .opens(Phase::GBuffer, 0),
        );
        registrar.node(
            pass_node(
                "gbuffer_portal_occurrences",
                "Geometry: GBuffer Through Portals",
                "Passes: Geometry",
                "The meshes again through the frame's portal chains (helio_portal_api): each occurrence mapped by its chain, clipped to what the chain shows, with the G-buffer program's own occurrence entry points. A member only while the frame has portal chains.",
                |build| {
                    let mut pass = GBufferPass::new(build.env.device)
                        .with_view(helio_frame_api::view::view_of(build.env))
                        .portal_occurrences(build.env.device);
                    if let Some(templates) = build.env.setting::<helio_mats::MaterialTemplates>() {
                        pass = pass.with_template_registry(templates.opaque.clone());
                    }
                    build.graph.add_pass(Box::new(pass));
                    Ok(())
                },
            )
            .drawer(Phase::GBuffer, helio_portal_api::OCCURRENCE_ORDER),
        );
    }
}

/// The selected meshes in the editor's selection mask
/// ([`crate::selection`]): which culled draws are selected, then their
/// depth.
pub struct SelectionDrawers;

/// The two selection passes' shared draws, one per graph.
fn selection_draws(
    env: &mut helio_feature_api::BuildEnv<'_>,
) -> std::sync::Arc<std::sync::Mutex<crate::selection::SelectionDraws>> {
    env.shared(|| {
        crate::selection::SharedSelectionDraws(std::sync::Arc::default())
    })
    .0
    .clone()
}

impl Feature for SelectionDrawers {
    fn id(&self) -> &'static str {
        "helio.mesh.selection"
    }

    fn register(&self, registrar: &mut FeatureRegistrar<'_>) {
        registrar.consumes(helio_mesh_api::STATIC_OBJECT_OWNERS);
        registrar.consumes(helio_frame_api::selection::OBJECT_SELECTED);
        registrar.consumes(helio_mesh_api::MESH_INDICES);
        registrar.consumes(helio_mesh_api::MESH_VERTICES);
        registrar.consumes(helio_mesh_api::BONE_PALETTE);
        registrar.consumes(helio_mesh_api::SKIN_VERTICES);
        {
            use helio_core::graph::type_name_of as name;
            registrar.swap_group(&[
                name::<crate::selection::SelectionSelectPass>(),
                name::<crate::selection::SelectionMaskPass>(),
            ]);
        }
        registrar.node(
            pass_node(
                "mesh_selection_select",
                "Selection: Selected Meshes",
                "Passes: Output",
                "Picks this frame's visible draws of selected objects for the selection mask. Editor only.",
                |build| {
                    let shared = selection_draws(build.env);
                    build.graph.add_pass(Box::new(crate::selection::SelectionSelectPass::new(
                        build.env.device,
                        shared,
                    )));
                    Ok(())
                },
            )
            .before_phase(Phase::SelectionMask, 0),
        );
        registrar.node(
            pass_node(
                "mesh_selection_mask",
                "Selection: Mesh Mask",
                "Passes: Output",
                "Draws the selected meshes' depth into the selection mask, for the outline. Editor only.",
                |build| {
                    let shared = selection_draws(build.env);
                    build.graph.add_pass(Box::new(crate::selection::SelectionMaskPass::new(
                        build.env.device,
                        shared,
                    )));
                    Ok(())
                },
            )
            .drawer(Phase::SelectionMask, 0),
        );
    }
}
