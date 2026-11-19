//! The meshes' tracer as a feature: registered once, it answers every ray
//! stream a consumer's node traces (`helio_frame_api::rays::stream::trace`).
//! It adds no node of its own; without hardware ray queries it is not
//! offered, and the streams see no meshes.

use helio_feature_api::{Feature, FeatureRegistrar};
use helio_frame_api::rays::stream::RegisterRayTracer;

/// The meshes' tracer of ray streams ([`crate`]).
pub struct MeshRays;

impl Feature for MeshRays {
    fn id(&self) -> &'static str {
        "helio.mesh.rays"
    }

    fn register(&self, registrar: &mut FeatureRegistrar<'_>) {
        registrar.consumes(helio_mesh_api::MESH_VERTICES);
        registrar.consumes(helio_mesh_api::MESH_INDICES);
        registrar.consumes(helio_mats::MATERIALS);
        registrar.ray_tracer(crate::tracer());
    }
}
