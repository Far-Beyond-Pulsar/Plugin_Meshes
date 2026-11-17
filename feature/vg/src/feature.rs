//! Virtual geometry as a feature: the node that adds the chain at the end of
//! the classic geometry.

use helio_feature_api::{pass_node, Feature, FeatureRegistrar};
use helio_frame_api::{FrameNode, Phase};
use helio_vg_stream::VgSettings;

use crate::{add_portal_virtual_geometry_passes, add_virtual_geometry_passes};

/// Virtual geometry ([`crate`]). Its runtime is the [`VgSettings`] setting.
pub struct VgFeature;

impl Feature for VgFeature {
    fn id(&self) -> &'static str {
        "helio.vg"
    }

    fn register(&self, registrar: &mut FeatureRegistrar<'_>) {
        registrar.consumes(helio_mats::MATERIALS);
        registrar.consumes(helio_light_api::SCENE_LIGHTS);
        registrar.consumes(helio_mesh_api::VG_INSTANCE_ROWS);
        registrar.consumes(helio_mesh_api::STATIC_OBJECTS);
        registrar.consumes(helio_mesh_api::OBJECTS_STRUCTURE);
        // Virtual-geometry instances are traced only through their proxies
        // (a ray-tracing projection with a VG runtime).
        registrar.untraced_geometry(helio_mesh_api::VG_INSTANCES);
        // Last of the classic geometry: its resolve depth-tests against the
        // combined classic depth (portals and foliage included). Builds nothing
        // unless the host set up virtual geometry and the device supports it.
        registrar.node(
            pass_node(
                "virtual_geometry",
                "Geometry: Virtual Geometry",
                "Passes: Geometry",
                "Streams, culls, rasterises and shades virtual-geometry clusters into the GBuffer.",
                |build| {
                    let settings = build
                        .env
                        .setting::<VgSettings>()
                        .cloned()
                        .unwrap_or_default();
                    add_virtual_geometry_passes(
                        build.graph,
                        build.env.device,
                        &settings,
                        helio_frame_api::view::view_of(build.env),
                    );
                    Ok(())
                },
            )
            .with_marker_after()
            .drawer(Phase::GBuffer, 30),
        );
        // The same geometry through the frame's portal chains: a second run
        // over the instances each chain shows, culled against the depth the
        // mask and the classic occurrences left.
        registrar.node(
            pass_node(
                "virtual_geometry_portals",
                "Geometry: Virtual Geometry Through Portals",
                "Passes: Geometry",
                "Virtual geometry again through the frame's portal chains (helio_portal_api): the instances each chain shows, placed where it draws them, culled against the frame's depth after the classic occurrences and clipped to the chain. Members only while the frame has portal chains and the scene virtual-geometry instances.",
                |build| {
                    let settings = build
                        .env
                        .setting::<VgSettings>()
                        .cloned()
                        .unwrap_or_default();
                    add_portal_virtual_geometry_passes(
                        build.graph,
                        build.env.device,
                        &settings,
                        helio_frame_api::view::view_of(build.env),
                    );
                    Ok(())
                },
            )
            .drawer(Phase::GBuffer, helio_portal_api::OCCLUDED_OCCURRENCE_ORDER),
        );
    }
}
