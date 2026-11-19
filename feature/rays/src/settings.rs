//! The meshes' ray-traced scene's host settings ([`RAY_TRACING`], as
//! `helio_settings` describes the renderer's): which mesh instances the
//! frame's TLAS holds and how much skinned geometry is refit a frame. Read
//! back into [`RayTracingSettings`] by the projections.
//!
//! The group lives with the meshes' ray-tracing feature, as the other kinds'
//! settings live with their features (GI's, the voxels', water's): a group is
//! a `static` compared by address, which an api crate does not hold, and both
//! projections (the engine's `meshes_component::ray_scene` and the standalone
//! hosts' `helio_mesh_host::ray_tracing`) already link this crate.

use helio_settings::{integer, live, number, Group, Resolved};

/// The meshes' ray-traced scene's host settings: a host lists it with its
/// other settings groups.
pub static RAY_TRACING: Group = Group {
    key: "ray_tracing",
    title: "Ray Tracing",
    description: "Which meshes the ray-traced scene holds, and how much skinned geometry it refits a frame",
    settings: &[
        live(number(
            "culling_radius",
            "Ray Tracing Culling Radius",
            "Ray Tracing",
            (0.0, 1_000_000.0, 10.0),
            300.0,
            "Metres from the eye inside which every mesh is traced; past it a mesh is traced only while it looks larger than the culling angle",
        )),
        live(number(
            "culling_angle",
            "Ray Tracing Culling Angle",
            "Ray Tracing",
            (0.0, 90.0, 0.1),
            1.0,
            "Degrees a mesh past the culling radius must span, seen from the eye, to stay traced (0 traces every mesh)",
        )),
        live(integer(
            "skinned_refit_triangles",
            "Skinned Refit Budget (triangles)",
            "Ray Tracing",
            (1, 1 << 30),
            1 << 18,
            "Triangles of posed skinned meshes refit a frame, nearest first; the rest keep last frame's pose in rays",
        )),
    ],
};

/// [`RAY_TRACING`] as the projections read it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayTracingSettings {
    /// Metres from the eye inside which every instance is traced.
    pub culling_radius: f64,
    /// Radians an instance past `culling_radius` must span to stay traced
    /// (its bounding sphere's angular diameter); 0 traces every instance.
    pub culling_angle: f64,
    /// Triangles of skinned BLASes refit (or rebuilt) a frame, at least one
    /// BLAS's.
    pub skinned_refit_triangles: u64,
}

impl Default for RayTracingSettings {
    fn default() -> Self {
        Self::from_settings(&RAY_TRACING.defaults())
    }
}

impl RayTracingSettings {
    pub fn from_settings(settings: &Resolved) -> Self {
        debug_assert!(std::ptr::eq(settings.group(), &RAY_TRACING));
        Self {
            culling_radius: settings.number("culling_radius").max(0.0),
            culling_angle: settings
                .number("culling_angle")
                .clamp(0.0, 90.0)
                .to_radians(),
            skinned_refit_triangles: settings.number("skinned_refit_triangles").max(1.0) as u64,
        }
    }

    /// Whether an instance whose bounding sphere has `radius` and its centre
    /// `distance` from the eye is in the ray-traced scene: within the
    /// culling radius, or spanning at least the culling angle (Unreal's
    /// ray-tracing culling by radius and solid angle). `slack` > 1 widens
    /// both (an instance already traced leaves only past `slack` times the
    /// thresholds, so one at a threshold does not flicker in and out).
    pub fn traces(&self, distance: f64, radius: f64, slack: f64) -> bool {
        if distance <= self.culling_radius * slack || self.culling_angle <= 0.0 {
            return true;
        }
        let span = 2.0 * (radius.max(0.0) / distance.max(1e-9)).atan();
        span * slack >= self.culling_angle
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Inside the radius everything is traced; past it only what spans the
    /// angle; the slack keeps an instance at a threshold traced.
    #[test]
    fn instances_are_culled_by_distance_and_angle() {
        let s = RayTracingSettings::default();
        assert_eq!(s.culling_radius, 300.0);
        assert!((s.culling_angle - 1f64.to_radians()).abs() < 1e-12);
        assert!(s.traces(299.0, 0.01, 1.0), "inside the radius");
        assert!(
            !s.traces(301.0, 0.5, 1.0),
            "a 1 m pebble at 301 m spans 0.19 degrees"
        );
        // A 20 m tree spans 1 degree out to about 1146 m.
        assert!(s.traces(1100.0, 10.0, 1.0));
        assert!(!s.traces(1200.0, 10.0, 1.0));
        assert!(s.traces(1200.0, 10.0, 1.1), "the slack keeps it");
        assert!(s.traces(310.0, 0.01, 1.1), "and the radius's");
        let all = RayTracingSettings {
            culling_angle: 0.0,
            ..s
        };
        assert!(all.traces(1e7, 0.0, 1.0), "angle 0 traces every instance");
    }
}
