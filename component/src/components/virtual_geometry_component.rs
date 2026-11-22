//! Per-instance virtual-geometry tuning (plan:
//! `.agents/plans/virtual-geometry.md` section 3 items 2 and 7).
//!
//! Whether a mesh uses virtual geometry at all is the static mesh's own
//! `virtual_geometry` property (seeded from its mesh asset when placed), not
//! this component. A VG mesh with **no** `VirtualGeometryComponent` uses the
//! defaults (tier `Auto`, error scale 1, casts shadows); attaching the
//! component tunes those per instance.
//!
//! The component is its own component-instance entity linked to its owner
//! object through `ComponentOwner`, like `LODComponent`. Its authored values
//! reach the GPU as one [`VgOverrideGpu`] row in the variable-length buffer
//! `vg_instance_overrides` (handles in `vg_instance_overrides::handles`,
//! keyed by the component's entity row), derived from the value on every
//! insert, write, removal and replay ([`VirtualGeometryOverrideRow`]), so
//! however a value is written (a literal, a decode, a property edit) its row
//! is its authored values. The virtual-geometry scene
//! derivation ([`crate::virtual_geometry_instances`]) joins it to the
//! owner's static-mesh instances on the GPU.
//!
//! This component declares no `declare_unfinished_component!`: the issue link
//! that declaration requires does not exist for this feature. Until the
//! virtual-geometry passes are wired into the default graph (task T10) its
//! ledger row is `unverified` instead.

use engine_class_derive::{engine_class, world_component};
use helio_vg_format::{INST_FORCE_ROOT_ONLY, INST_SHADOW_CASTER, INST_TIER_SHIFT, TierLock};
use pulsar_reflection::Reflectable;
use pulsar_scenedb::gpu::GpuMirrorHandle;
use pulsar_scenedb_derive::SceneStore;
use serde::{Deserialize, Serialize};

/// Which raster tier an instance is locked to.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq, Reflectable)]
pub enum VgTierLock {
    /// Each cluster picks its tier from its projected triangle size.
    #[default]
    Auto,
    /// Every cluster of the instance is rasterised by the hardware tier.
    HwRaster,
    /// Every cluster of the instance is rasterised by the software
    /// micro-polygon tier.
    SwRaster,
    /// Every cluster of the instance goes through the dense-overlap tier.
    Dense,
}

impl From<VgTierLock> for TierLock {
    fn from(tier: VgTierLock) -> Self {
        match tier {
            VgTierLock::Auto => Self::Auto,
            VgTierLock::HwRaster => Self::HwRaster,
            VgTierLock::SwRaster => Self::SwRaster,
            VgTierLock::Dense => Self::Dense,
        }
    }
}

/// Smallest and largest per-instance multiplier of the screen-space error
/// threshold. Finer than 1/64 or coarser than 64x is never what an author
/// wants, and bounding it keeps a typo from collapsing or exploding the cut.
pub const LOD_ERROR_SCALE_RANGE: (f32, f32) = (1.0 / 64.0, 64.0);

/// Reflected per-instance tuning for virtual geometry. See the module doc.
#[engine_class(
    category = "Rendering",
    clone,
    debug,
    serialize,
    deserialize
)]
pub struct VirtualGeometryComponent {
    /// Pin the instance to one raster tier instead of choosing per cluster.
    #[property]
    #[serde(default)]
    pub tier_lock: VgTierLock,

    /// Multiplies the screen-space error threshold for this instance: above 1
    /// keeps coarser clusters (cheaper), below 1 keeps finer ones.
    #[property(min = 0.0625, max = 16.0, step = 0.05)]
    #[serde(default = "default_one")]
    pub lod_error_scale: f32,

    /// Draw only the coarsest cut of the cluster hierarchy (useful for
    /// debugging and for very distant fillers).
    #[property]
    #[serde(default)]
    pub force_root_only: bool,

    /// Whether this instance is drawn into shadow maps.
    #[property]
    #[serde(default = "default_true")]
    pub shadow_caster: bool,
}

fn default_true() -> bool {
    true
}

fn default_one() -> f32 {
    1.0
}

impl Default for VirtualGeometryComponent {
    fn default() -> Self {
        Self {
            tier_lock: VgTierLock::Auto,
            lod_error_scale: 1.0,
            force_root_only: false,
            shadow_caster: true,
        }
    }
}

/// One component's GPU row: `INST_*` flags (shadow caster, force root only,
/// tier lock in bits 8..10; `INST_ENABLED`, `INST_LOD_SUPPRESS` and
/// `INST_CONE_CULL` are derived per instance, never authored) and the
/// sanitised error scale.
/// 8 bytes, no padding. Mirrored by `struct Override` in
/// `virtual_geometry_instances.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VgOverrideGpu {
    pub flags: u32,
    pub lod_error_scale: f32,
}

// SAFETY: `repr(C)`, all fields are `Pod`, and the 8-byte layout has no
// implicit padding.
unsafe impl pulsar_scenedb::page::Pod for VgOverrideGpu {}

/// `INST_*` flags the authored values map to.
pub fn vg_override_flags(tier_lock: VgTierLock, force_root_only: bool, shadow_caster: bool) -> u32 {
    let mut flags = 0;
    if shadow_caster {
        flags |= INST_SHADOW_CASTER;
    }
    if force_root_only {
        flags |= INST_FORCE_ROOT_ONLY;
    }
    flags | ((TierLock::from(tier_lock) as u32) << INST_TIER_SHIFT)
}

/// A finite, in-range error scale; anything else is the neutral 1.
pub fn sanitize_lod_error_scale(scale: f32) -> f32 {
    if scale.is_finite() {
        scale.clamp(LOD_ERROR_SCALE_RANGE.0, LOD_ERROR_SCALE_RANGE.1)
    } else {
        1.0
    }
}

impl VirtualGeometryComponent {
    /// The GPU row for the current authored values.
    pub fn gpu_row(&self) -> VgOverrideGpu {
        VgOverrideGpu {
            flags: vg_override_flags(self.tier_lock, self.force_root_only, self.shadow_caster),
            lod_error_scale: sanitize_lod_error_scale(self.lod_error_scale),
        }
    }
}

/// What a [`VirtualGeometryComponent`] uploads: its one [`VgOverrideGpu`]
/// row, held as a variable-length range (the handle/pool scheme
/// `LODComponent` follows: a zero count means no component). Derived from
/// the authored value by SceneDB on every insert, guarded write, removal and
/// replay of the component (a second GPU registration on it, as
/// `StaticMeshDraw` is the static mesh's), never cached on the value, so no
/// write path can leave the row behind its properties.
#[derive(SceneStore, Clone, Debug)]
pub struct VirtualGeometryOverrideRow {
    #[gpu(buffer = "vg_instance_overrides")]
    pub rows: Vec<VgOverrideGpu>,
}

impl VirtualGeometryOverrideRow {
    pub fn of(component: &VirtualGeometryComponent) -> Self {
        Self {
            rows: vec![component.gpu_row()],
        }
    }
}

fn virtual_geometry_override_dispatch(
    mirror: &GpuMirrorHandle,
    row: u32,
    data: *const (),
    is_new_insert: bool,
) {
    // SAFETY: SceneDB reaches this only through `VirtualGeometryComponent`'s
    // own `ComponentId`, with a pointer to a live `VirtualGeometryComponent`.
    let component = unsafe { &*(data as *const VirtualGeometryComponent) };
    let derived = VirtualGeometryOverrideRow::of(component);
    pulsar_scenedb::gpu::write_derived_row(mirror, row, &derived, is_new_insert);
}

fn virtual_geometry_override_clear(mirror: &GpuMirrorHandle, row: u32) {
    pulsar_scenedb::gpu::clear_derived_row::<VirtualGeometryOverrideRow>(mirror, row);
}

pulsar_scenedb::pulsar_reflection::inventory::submit! {
    pulsar_scenedb::gpu::GpuMirrorRegistration {
        component_id: pulsar_scenedb::component_id::<VirtualGeometryComponent>,
        dispatch: virtual_geometry_override_dispatch,
    }
}

pulsar_scenedb::pulsar_reflection::inventory::submit! {
    pulsar_scenedb::gpu::GpuClearRegistration {
        component_id: pulsar_scenedb::component_id::<VirtualGeometryComponent>,
        clear: virtual_geometry_override_clear,
    }
}

#[world_component]
impl VirtualGeometryComponent {}


#[cfg(test)]
mod tests {
    use super::*;
    use helio_vg_format::{INST_CONE_CULL, INST_ENABLED, INST_LOD_SUPPRESS};
    use std::mem::size_of;

    #[test]
    fn row_layout_is_pinned() {
        assert_eq!(size_of::<VgOverrideGpu>(), 8);
    }

    #[test]
    fn default_component_is_auto_and_casts_shadows() {
        let component = VirtualGeometryComponent::default();
        assert!(component.shadow_caster && !component.force_root_only);
        assert_eq!(component.tier_lock, VgTierLock::Auto);
        assert_eq!(component.lod_error_scale, 1.0);
        let row = component.gpu_row();
        assert_eq!(row.flags, INST_SHADOW_CASTER);
        assert_eq!(row.lod_error_scale, 1.0);
    }

    #[test]
    fn flags_encode_every_authored_value_and_never_the_derived_bits() {
        let flags = vg_override_flags(VgTierLock::SwRaster, true, false);
        assert_eq!(
            flags,
            INST_FORCE_ROOT_ONLY | ((TierLock::SwRaster as u32) << INST_TIER_SHIFT)
        );
        assert_eq!(
            flags & (INST_ENABLED | INST_LOD_SUPPRESS | INST_CONE_CULL),
            0
        );
        assert_eq!(vg_override_flags(VgTierLock::Auto, false, false), 0);
        for tier in [
            VgTierLock::Auto,
            VgTierLock::HwRaster,
            VgTierLock::SwRaster,
            VgTierLock::Dense,
        ] {
            let row = vg_override_flags(tier, false, true);
            assert_eq!(
                TierLock::from_bits(row >> INST_TIER_SHIFT),
                TierLock::from(tier)
            );
        }
    }

    #[test]
    fn error_scale_is_sanitised() {
        assert_eq!(sanitize_lod_error_scale(f32::NAN), 1.0);
        assert_eq!(sanitize_lod_error_scale(f32::INFINITY), 1.0);
        assert_eq!(sanitize_lod_error_scale(0.0), LOD_ERROR_SCALE_RANGE.0);
        assert_eq!(sanitize_lod_error_scale(1000.0), LOD_ERROR_SCALE_RANGE.1);
        assert_eq!(sanitize_lod_error_scale(2.5), 2.5);
    }

    /// The uploaded row is derived from the value as written, however it was
    /// made: a literal built with struct update syntax from the defaults
    /// (what scripts and tests write) uploads its own authored values.
    #[test]
    fn the_uploaded_row_is_the_values_authored_values() {
        let component = VirtualGeometryComponent {
            force_root_only: true,
            shadow_caster: false,
            lod_error_scale: 16.0,
            ..VirtualGeometryComponent::default()
        };
        let derived = VirtualGeometryOverrideRow::of(&component);
        assert_eq!(derived.rows.len(), 1);
        assert_eq!(derived.rows[0].flags, INST_FORCE_ROOT_ONLY);
        assert_eq!(derived.rows[0].lod_error_scale, 16.0);
    }

    /// A component saved with fewer (or no) fields, or by a build that knew
    /// fewer, still loads with the defaults for the rest.
    #[test]
    fn serde_defaults_keep_old_and_partial_records_loading() {
        let empty: VirtualGeometryComponent = serde_json::from_str("{}").expect("empty record");
        assert!(empty.shadow_caster);
        assert_eq!(empty.lod_error_scale, 1.0);
        assert_eq!(empty.tier_lock, VgTierLock::Auto);

        // A record from before the on/off switch moved to the static mesh
        // still names `enabled`; it is ignored.
        let partial: VirtualGeometryComponent =
            serde_json::from_str(r#"{"enabled": false, "tier_lock": "Dense"}"#).expect("partial");
        assert!(partial.shadow_caster);
        assert_eq!(partial.tier_lock, VgTierLock::Dense);

        // The GPU row is derived, never persisted.
        let json = serde_json::to_value(VirtualGeometryComponent::default()).unwrap();
        assert!(json.get("gpu_override").is_none());
    }
}
