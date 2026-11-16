//! The meshes' occurrences through portal chains (`helio_portal_api`): what
//! the occurrence cull (`helio-pass-occlusion-cull`'s
//! `PortalOccurrencePass`) found and the G-buffer's occurrence draw reads.
//! The meshes' contract, as [`crate::CulledBatchFrameData`].

/// Registry key of [`PortalOccurrenceFrameData`].
pub const PORTAL_OCCURRENCES: &str = "mesh_portal_occurrences";

#[derive(Clone, Copy)]
pub struct PortalOccurrenceFrameData<'a> {
    /// One indirect record per draw group (`first_instance` the group's
    /// offset in the two buffers below), for adapters drawing before the
    /// segments exist.
    pub indirect: &'a wgpu::Buffer,
    /// Per occurrence, the instance slot (index `instances` of
    /// [`crate::ObjectBatchFrameData`] through it).
    pub instances: &'a wgpu::Buffer,
    /// Per occurrence, the chain (`helio_portal_api::PORTAL_CHAINS` row).
    pub chains: &'a wgpu::Buffer,
    /// The same draws packed per material key, in the occlusion cull's
    /// segments (its table, this pass's records and counts).
    pub segments: crate::DrawSegments<'a>,
}
