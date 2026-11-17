//! A run of the virtual-geometry pipeline in a graph ([`VgRun`]): the names of
//! its passes and every graph key they share, and where its instances come
//! from.
//!
//! A frame can run the pipeline more than once over different instance sets,
//! each run at its own place in the frame: the frame's own instances
//! ([`VgRun::DIRECT`]), and a second set another pass publishes
//! ([`VgRun::SECONDARY`], for instances drawn after other drawers). Every VG
//! pass takes the run it belongs to, so two runs never share a key, a target
//! or a Hi-Z.

/// Where a run's instances come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VgInstanceSource {
    /// The scene's: the VG derivation's `vg_instances` rows and the scene
    /// join's `static_objects` rows (scene buffers).
    Scene,
    /// Buffers a pass of the graph publishes (registry buffers): `instances`
    /// (`array<GpuVgInstance>`), `objects` (`array<StaticObjectRow>`, the rows
    /// the instances' `object_row` index) and `remaps` (`array<u32>` per
    /// instance row: the surface remap its pixels carry,
    /// `helio_frame_api::lighting::VIEW_REMAPS`; `NO_REMAP` for none).
    Registry {
        instances: &'static str,
        objects: &'static str,
        remaps: &'static str,
    },
}

/// A per-instance remap meaning "drawn where it is" (no remap).
pub const NO_REMAP: u32 = u32::MAX;

/// The names of a run's passes (timings, membership, debugging).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VgRunNames {
    pub inputs: &'static str,
    pub hiz_early: &'static str,
    pub cull_early: &'static str,
    pub raster_early: &'static str,
    pub hiz_late: &'static str,
    pub cull_late: &'static str,
    pub raster_late: &'static str,
    pub dense: &'static str,
    pub shade: &'static str,
}

/// The graph keys of a run's instance BVH (`helio-pass-vg-bvh`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VgRunTree {
    pub nodes: &'static str,
    pub header: &'static str,
    pub items: &'static str,
}

/// One run of the virtual-geometry pipeline ([`crate::run`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VgRun {
    pub names: VgRunNames,
    /// `VgCullInputs`, published by the run's inputs pass.
    pub inputs: &'static str,
    /// The Hi-Z the early cull tests against: the frame's (last frame's
    /// depth) for the direct run, or one the run builds itself
    /// ([`Self::builds_hiz_early`]).
    pub hiz_early: &'static str,
    /// Whether the run builds [`Self::hiz_early`] from the frame's depth as it
    /// stands where the run draws (after the drawers before it).
    pub builds_hiz_early: bool,
    /// The Hi-Z of the run's own phase-0 depth (the late cull's).
    pub hiz_late: &'static str,
    /// The cull's frame data after phase 0 and after phase 1.
    pub cull_early: &'static str,
    pub cull: &'static str,
    /// The raster's frame data after each phase.
    pub raster_early: &'static str,
    pub raster: &'static str,
    /// The run's instance BVH, which its cull walks from the root; `None`
    /// for a run whose instances are already the result of a spatial query
    /// (few, every one a candidate): its cull seeds each instance.
    pub tree: Option<VgRunTree>,
    /// Tier 3's frame data.
    pub dense: &'static str,
    pub instances: VgInstanceSource,
}

impl VgRun {
    /// The frame's own instances, drawn where the classic geometry is.
    pub const DIRECT: VgRun = VgRun {
        names: VgRunNames {
            inputs: "VgInputs",
            hiz_early: "VgHizEarly",
            cull_early: "VgCullEarly",
            raster_early: "VgRasterEarly",
            hiz_late: "VgHizLate",
            cull_late: "VgCullLate",
            raster_late: "VgRasterLate",
            dense: "VgDense",
            shade: "VgShade",
        },
        inputs: "vg_cull_inputs",
        hiz_early: "hiz",
        builds_hiz_early: false,
        hiz_late: "vg_hiz_late",
        cull_early: "vg_cull_early",
        cull: "vg_cull",
        raster_early: "vg_raster_early",
        raster: "vg_raster",
        tree: Some(VgRunTree {
            nodes: "vg_gcbvh_nodes",
            header: "vg_gcbvh_header",
            items: "vg_gcbvh_items",
        }),
        dense: "vg_dense",
        instances: VgInstanceSource::Scene,
    };

    /// A second instance set another pass publishes, drawn later in the
    /// frame (after the drawers before it, against the depth they left). Its
    /// instances are what a spatial query found (no tree: the cull seeds
    /// each).
    pub const SECONDARY: VgRun = VgRun {
        names: VgRunNames {
            inputs: "VgInputsSecondary",
            hiz_early: "VgHizEarlySecondary",
            cull_early: "VgCullEarlySecondary",
            raster_early: "VgRasterEarlySecondary",
            hiz_late: "VgHizLateSecondary",
            cull_late: "VgCullLateSecondary",
            raster_late: "VgRasterLateSecondary",
            dense: "VgDenseSecondary",
            shade: "VgShadeSecondary",
        },
        inputs: "vg_secondary_cull_inputs",
        hiz_early: "vg_secondary_hiz_early",
        builds_hiz_early: true,
        hiz_late: "vg_secondary_hiz_late",
        cull_early: "vg_secondary_cull_early",
        cull: "vg_secondary_cull",
        raster_early: "vg_secondary_raster_early",
        raster: "vg_secondary_raster",
        tree: None,
        dense: "vg_secondary_dense",
        instances: VgInstanceSource::Registry {
            instances: "vg_secondary_instances",
            objects: "vg_secondary_objects",
            remaps: "vg_secondary_remaps",
        },
    };

    /// Whether the run's instances carry remaps (drawn through another
    /// space: their surfaces are tagged and clipped).
    pub const fn remapped(&self) -> bool {
        matches!(self.instances, VgInstanceSource::Registry { .. })
    }

    /// [`Self::tree`] of a run that has one.
    pub const fn tree(&self) -> VgRunTree {
        match self.tree {
            Some(tree) => tree,
            None => panic!("the run seeds its instances: it has no tree"),
        }
    }
}

/// Whether a run's passes are members of this frame's graph: shared by every
/// pass of a run whose content another pass decides (a run over instances it
/// publishes: scheduled only while it publishes some). Open by default.
#[derive(Clone, Debug)]
pub struct VgRunGate(std::sync::Arc<std::sync::atomic::AtomicBool>);

impl Default for VgRunGate {
    fn default() -> Self {
        Self(std::sync::Arc::new(std::sync::atomic::AtomicBool::new(
            true,
        )))
    }
}

impl VgRunGate {
    pub fn set(&self, open: bool) {
        self.0.store(open, std::sync::atomic::Ordering::Release);
    }

    pub fn is_open(&self) -> bool {
        self.0.load(std::sync::atomic::Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_share_no_key_or_pass_name() {
        let keys = |r: &VgRun| {
            vec![
                r.inputs,
                r.hiz_late,
                r.cull_early,
                r.cull,
                r.raster_early,
                r.raster,
                r.dense,
                r.names.inputs,
                r.names.hiz_early,
                r.names.cull_early,
                r.names.raster_early,
                r.names.hiz_late,
                r.names.cull_late,
                r.names.raster_late,
                r.names.dense,
                r.names.shade,
            ]
        };
        let mut direct = keys(&VgRun::DIRECT);
        let tree = VgRun::DIRECT.tree();
        direct.extend([tree.nodes, tree.header, tree.items]);
        assert!(VgRun::SECONDARY.tree.is_none());
        for key in keys(&VgRun::SECONDARY) {
            assert!(!direct.contains(&key), "{key} is shared");
        }
        assert_ne!(VgRun::DIRECT.hiz_early, VgRun::SECONDARY.hiz_early);
    }
}
