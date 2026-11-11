//! The rows the meshes' components author (objects, render groups) and the
//! mesh pools they address. The object batch projects
//! them into transient instance and coordinate-space inputs.

/// Mesh vertices: a pool shared by every mesh. A [`StaticObjectRow`]'s
/// `vertex_offset` addresses it. Engines fill it (Helio's standalone hosts use
/// `helio_mesh_host::MeshComponent`).
pub const MESH_VERTICES: helio_core::PoolContract<crate::PackedVertex> =
    helio_core::PoolContract::new("builtin_mesh_vertex");

/// Mesh indices: a pool shared by every mesh, addressed by a
/// [`StaticObjectRow`]'s `first_index`/`index_count`.
pub const MESH_INDICES: helio_core::PoolContract<u32> =
    helio_core::PoolContract::new("builtin_mesh_index");

/// Stable group membership. `group_mask == 0` means always visible.
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct RenderGroupRow {
    pub group_mask: u64,
}

// SAFETY: a `#[repr(C)]` bytemuck `Pod` row: no padding bytes, any bit
// pattern valid. Lets an engine's SceneDB component embed it.
unsafe impl pulsar_scenedb::Pod for RenderGroupRow {}

/// The buffer the renderer reads [`RenderGroupRow`]s from. Engines produce it
/// (a SceneDB component stored under this key, or a scene derivation).
pub const RENDER_GROUPS: helio_core::BufferContract<RenderGroupRow> =
    helio_core::BufferContract::new("render_groups");

/// A single-mesh static-object placement: SceneDB-owned mesh/material row
/// indices, a world transform, a culling bound, and the static draw
/// parameters a GPU-driven instancing/culling pipeline needs to read this
/// row directly with no per-frame CPU involvement.
///
/// `SectionedObjectComponent` above covers the multi-material case, which
/// additionally needs the owning `MultiMeshId`/per-section materials tracked
/// elsewhere.
///
/// # Why the draw parameters are stored, not derived per frame
///
/// `index_count`/`first_index`/`vertex_offset` (the mesh's vertex/index
/// range), and `material_class`/`graph_hash` (the material's
/// pipeline-selection key) are authored alongside the row. They are not
/// resolved through a renderer API; the SceneDB asset/ingestion layer owns
/// that resolution before inserting this component.
///
/// `mesh_slot` and `material_slot` are indices into the keyed SceneDB asset
/// buffers. The draw pipeline consumes those plain indices; no renderer
/// handle type is stored or reconstructed here.
///
/// # What's NOT solved by this component alone
///
/// Building the *grouped* `draw_calls`/`instances`/`aabbs` arrays a
/// GPU-driven culling pass reads (batching same-mesh-same-material
/// instances into one indirect draw call, partitioned into contiguous
/// opaque/transparent/forward ranges by `material_class`) requires sorting
/// every live `StaticObjectRow` row by `(material_class, graph_hash,
/// mesh_id, material_id)` — a cross-entity operation, not a per-row one.
/// That GPU sort/group/dispatch pipeline is specified but not yet
/// implemented; see the Helio issue tracking it. Until it lands, a
/// consumer must still assemble those arrays itself from a `World` query
/// over this component (a per-frame CPU cost, same category as the
/// pre-existing shadow-atlas scoring — tracked, not silently hidden).
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct StaticObjectRow {
    pub mesh_slot: u32,
    pub mesh_generation: u32,
    pub material_slot: u32,
    pub material_generation: u32,
    pub transform: [[f32; 4]; 4],
    /// Previous frame's `transform`, for TAA/TSR motion vectors. Equal to
    /// `transform` (zero apparent velocity) the frame this row is first
    /// inserted or whenever a caller uses [`Self::with_transform`] without
    /// tracking real per-frame motion -- a quality simplification (a
    /// stationary-looking first frame), never a correctness one.
    pub prev_transform: [[f32; 4]; 4],
    pub normal_mat: [[f32; 4]; 3],
    pub bounds: [f32; 4],
    /// Resolved once at spawn time from `Renderer::mesh_slice` -- see the
    /// struct doc's "Why the draw parameters are stored" section.
    pub index_count: u32,
    pub first_index: u32,
    pub vertex_offset: i32,
    /// Resolved once at spawn time from `Renderer::material_batch_key`.
    pub material_class: u32,
    pub graph_hash_lo: u32,
    pub graph_hash_hi: u32,
    /// See `crate::GpuInstanceData::flags`.
    pub flags: u32,
    /// Base vertex in the parallel four-influence skin stream (zero for rigid meshes).
    pub skin_vertex_offset: u32,
    /// Base joint matrix in the shared palette pool (zero for rigid meshes).
    pub bone_palette_offset: u32,
    /// Number of valid joint matrices for this instance.
    pub bone_palette_count: u32,
    pub _skin_pad: u32,
}

// SAFETY: a `#[repr(C)]` bytemuck `Pod` row: no padding bytes, any bit
// pattern valid. Lets an engine's SceneDB component embed it.
unsafe impl pulsar_scenedb::Pod for StaticObjectRow {}

/// The skinned meshes' per-vertex skin influences (a variable-length pool
/// the skinned drawers read by instance range).
pub const SKELETAL_SKIN_VERTICES: &str = "skeletal_skin_vertices";
/// The skinned meshes' bone palettes: each instance's current palette
/// followed by its previous frame's, in one variable-length range.
pub const SKELETAL_BONE_PALETTE: &str = "skeletal_bone_palette";
/// The virtual-geometry instance rows (`GpuVgInstance`), one per VG-active
/// instance, which the VG host derives and the VG passes read.
pub const VG_INSTANCES: &str = "vg_instances";

/// Per-vertex four-influence skin data of [`SKELETAL_SKIN_VERTICES`]. Kept in
/// a storage buffer parallel to [`MESH_VERTICES`], so static geometry keeps
/// its established vertex ABI.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SkinVertex {
    pub joints: [u32; 4],
    pub weights: [f32; 4],
}
unsafe impl pulsar_scenedb::page::Pod for SkinVertex {}

/// [`SKELETAL_SKIN_VERTICES`] as a contract: a pool of [`SkinVertex`]es the
/// skinned drawers read by instance range.
pub const SKIN_VERTICES: helio_core::PoolContract<SkinVertex> =
    helio_core::PoolContract::new(SKELETAL_SKIN_VERTICES);

/// [`SKELETAL_BONE_PALETTE`] as a contract: a pool of joint matrices (the
/// current palette followed by the previous frame's, per instance).
pub const BONE_PALETTE: helio_core::PoolContract<[[f32; 4]; 4]> =
    helio_core::PoolContract::new(SKELETAL_BONE_PALETTE);

/// [`VG_INSTANCES`] as a contract: [`helio_vg_format::GpuVgInstance`] rows.
pub const VG_INSTANCE_ROWS: helio_core::BufferContract<helio_vg_format::GpuVgInstance> =
    helio_core::BufferContract::new(VG_INSTANCES);

/// The scene join's structure marker: a buffer whose `content_generation`
/// moves only when objects are added, removed or change what they are (mesh,
/// material, flags), not when they move. It carries no rows (handle
/// `row_bytes` 0), only the generation; consumers that sort and group on the
/// objects keep their work while it is unchanged.
pub const OBJECTS_STRUCTURE: helio_core::BufferContract<()> =
    helio_core::BufferContract::new("scene_objects_structure");

/// The object slots (rows of [`STATIC_OBJECTS`]) a frame that only moved
/// objects rewrote, as `[total, fresh, slot, ..]` words: this frame's moved
/// slots plus the previous frame's. Published only for a run whose list is
/// complete; like [`OBJECTS_STRUCTURE`] it has no row stride (handle
/// `row_bytes` 0).
pub const OBJECTS_MOVED: helio_core::BufferContract<()> =
    helio_core::BufferContract::new("scene_objects_moved");

/// One `u32` per [`STATIC_OBJECTS`] row: non-zero = the row belongs to a
/// virtual-geometry-active instance, so the main view's classic draw skips it
/// (shadows keep the classic geometry as the proxy).
pub const CLASSIC_SUPPRESSED: helio_core::BufferContract<u32> =
    helio_core::BufferContract::new("classic_suppressed");

/// The buffer the renderer reads [`StaticObjectRow`]s from. Engines produce it
/// (a SceneDB component stored under this key, or a scene derivation).
pub const STATIC_OBJECTS: helio_core::BufferContract<StaticObjectRow> =
    helio_core::BufferContract::new("static_objects");

impl StaticObjectRow {
    /// Construct an object row from already-resolved SceneDB asset metadata.
    ///
    /// `mesh_slot` and `material_slot` are plain SceneDB row indices. The
    /// renderer is intentionally absent: resolving those indices and the
    /// associated draw range belongs to the asset/component authoring side.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mesh_slot: u32,
        mesh_generation: u32,
        material_slot: u32,
        material_generation: u32,
        transform: glam::Mat4,
        bounds: [f32; 4],
        index_count: u32,
        first_index: u32,
        vertex_offset: i32,
        material_class: u32,
        graph_hash: u64,
        flags: u32,
    ) -> Self {
        Self {
            mesh_slot,
            mesh_generation,
            material_slot,
            material_generation,
            transform: transform.to_cols_array_2d(),
            prev_transform: transform.to_cols_array_2d(),
            normal_mat: normal_matrix_cols(transform),
            bounds,
            index_count,
            first_index,
            vertex_offset,
            material_class,
            graph_hash_lo: graph_hash as u32,
            graph_hash_hi: (graph_hash >> 32) as u32,
            flags,
            skin_vertex_offset: 0,
            bone_palette_offset: 0,
            bone_palette_count: 0,
            _skin_pad: 0,
        }
    }

    pub fn transform(&self) -> glam::Mat4 {
        glam::Mat4::from_cols_array_2d(&self.transform)
    }

    pub fn graph_hash(&self) -> u64 {
        (self.graph_hash_hi as u64) << 32 | self.graph_hash_lo as u64
    }

    /// Re-place this row at a new transform/bounds, recomputing `normal_mat`
    /// to match (it's derived from `transform`, so the two must never drift
    /// apart). `mesh`/`material` and their resolved draw parameters are
    /// static asset properties and are carried over unchanged -- moving an
    /// object never needs to re-resolve them.
    pub fn with_transform(mut self, transform: glam::Mat4, bounds: [f32; 4]) -> Self {
        self.prev_transform = self.transform;
        self.transform = transform.to_cols_array_2d();
        self.normal_mat = normal_matrix_cols(transform);
        self.bounds = bounds;
        self
    }
}

/// Inverse-transpose of `m`'s upper-left 3x3, as three padded columns
/// (matches `crate::GpuInstanceData::normal_mat`'s layout).
fn normal_matrix_cols(m: glam::Mat4) -> [[f32; 4]; 3] {
    let m3 = glam::Mat3::from_mat4(m);
    let inv_t = m3.inverse().transpose();
    let c0 = inv_t.x_axis;
    let c1 = inv_t.y_axis;
    let c2 = inv_t.z_axis;
    [
        [c0.x, c0.y, c0.z, 0.0],
        [c1.x, c1.y, c1.z, 0.0],
        [c2.x, c2.y, c2.z, 0.0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn component_layouts_are_stable() {
        assert_eq!(std::mem::size_of::<RenderGroupRow>(), 8);
    }
}
