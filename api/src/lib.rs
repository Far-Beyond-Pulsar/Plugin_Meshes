//! The meshes' public contract, the one place another kind reads a mesh
//! concept from:
//! - what their components author: the packed vertex layout, mesh uploads,
//!   the object, render-group and sublevel rows ([`rows`]), the mesh pools;
//! - what their batch publishes each frame for the passes that draw or cull
//!   mesh content: the instance and draw rows ([`draw`]), the object batch's
//!   and the culled batch's frame data, draw segments, draw slots and the
//!   multi-draw they are issued with; in WGSL, [`DRAW_ROWS_SNIPPET`];
//! - the buffers their ray-tracing projection publishes beside the frame's
//!   TLAS (`helio_frame_api::rays::SceneRays`): the instance rows
//!   ([`RAY_INSTANCES`]), each TLAS instance's transform
//!   ([`RAY_INSTANCE_TRANSFORMS`]), the coordinate spaces' mask groups
//!   ([`RAY_SPACE_GROUPS`]) and the proxy pool ([`RAY_PROXY_VERTICES`],
//!   [`RAY_PROXY_INDICES`]); in WGSL, [`RAY_ROWS_SNIPPET`] and
//!   [`RAY_SURFACE_SNIPPET`]; and, as a
//!   payload, the programs the rows' materials are evaluated by at hits
//!   ([`RAY_PROGRAMS`]).
//!
//! No passes: a kind that reads meshes depends on this crate, never on the
//! meshes' feature crates. With `host-rows`, the rows as SceneDB components
//! for standalone hosts ([`host_rows`]).

mod asset_types;
mod culled_batch_frame_data;
pub mod draw;
mod draw_segments;
pub mod draw_slots;
#[cfg(feature = "host-rows")]
pub mod host_rows;
mod indirect_draw;
mod object_batch_frame_data;
mod portal_occurrence_frame_data;
pub mod rows;
pub mod stats;

pub use asset_types::{MeshUpload, PackedVertex, SectionedMeshUpload};
pub use culled_batch_frame_data::CulledBatchFrameData;
pub use draw::*;
pub use draw_segments::{
    front_face, DrawSegment, DrawSegments, ShadingBucket, RASTER_FLAG_REVERSED_WINDING,
};
pub use draw_slots::{DrawSlots, DRAW_SLOT_LAYOUT, DRAW_SLOT_LOCATION};
pub use indirect_draw::{multi_draw_indexed_indirect, GpuDrawCount, DRAW_INDEXED_INDIRECT_STRIDE};
pub use object_batch_frame_data::{IndirectDispatchArgs, ObjectBatchFrameData};
pub use portal_occurrence_frame_data::{PortalOccurrenceFrameData, PORTAL_OCCURRENCES};
pub use rows::{
    RenderGroupRow, SkinCacheRow, SkinVertex, SkinnedVertex, StaticObjectRow, BONE_PALETTE,
    CLASSIC_SUPPRESSED, MESH_INDICES, MESH_VERTICES, OBJECTS_MOVED, OBJECTS_STRUCTURE,
    RENDER_GROUPS, SKELETAL_BONE_PALETTE, SKELETAL_SKIN_VERTICES, SKIN_CACHE, SKIN_CACHE_ROWS,
    SKIN_VERTICES, STATIC_OBJECTS, STATIC_OBJECT_OWNERS, VG_INSTANCES, VG_INSTANCE_ROWS,
};

/// The meshes' traced rows in WGSL (`//!use mesh_ray_rows`): the
/// [`RayInstance`] row and its flags, [`RayTransform`]
/// (`ray_transform_point`, `_vector`, `_normal`) and the space groups'
/// decode (`ray_space_cull`). No bindings: what reads the rows alone (a pass
/// binning hits by program) takes this one.
pub const RAY_ROWS_SNIPPET: helio_core::shader::ShaderSnippet =
    helio_core::wgsl_snippet!("//!use mesh_ray_rows", "shaders/ray_rows.wgsl");

/// The meshes' surfaces as rays see them, for a tracer's WGSL
/// (`//!use mesh_ray_surface`, with [`RAY_ROWS_SNIPPET`] and the frame's ray
/// stream snippet): the mesh pools' vertex decode, the proxy pool's,
/// `ray_triangle`, which fetches a hit's triangle by instance and primitive,
/// its outside (`ray_triangle_outside`, `mesh_ray_outside`), tangent frame
/// (`ray_triangle_tangent`) and the normals a hit is oriented by
/// (`mesh_ray_normals`). The reader declares
/// the storage it reads under these names: `instances: array<RayInstance>`
/// ([`RAY_INSTANCES`]), `vertices: array<u32>` and `indices: array<u32>`
/// (the mesh pools, [`MESH_VERTICES`] / [`MESH_INDICES`]),
/// `proxy_vertices: array<f32>` and `proxy_indices: array<u32>`
/// ([`RAY_PROXY_VERTICES`] / [`RAY_PROXY_INDICES`]) and
/// `skinned_vertices: array<u32>` (the skin cache, [`RAY_SKINNED_VERTICES`],
/// [`SkinnedVertex`] words).
pub const RAY_SURFACE_SNIPPET: helio_core::shader::ShaderSnippet =
    helio_core::wgsl_snippet!("//!use mesh_ray_surface", "shaders/ray_surface.wgsl");

/// The object batch's rows in WGSL (`//!use mesh_draw_rows`):
/// [`GpuInstanceData`], [`GpuDrawCall`], [`DrawIndexedIndirectArgs`] (as
/// `DrawIndexedIndirect`), the `INSTANCE_FLAG_*` bits and
/// `instance_coordinate_space`, for what reads the batch on the GPU.
pub const DRAW_ROWS_SNIPPET: helio_core::shader::ShaderSnippet =
    helio_core::wgsl_snippet!("//!use mesh_draw_rows", "shaders/draw_rows.wgsl");

/// The object row in WGSL (`//!use mesh_object_row`): [`StaticObjectRow`]
/// as `struct StaticObjectRow`, for what reads the `static_objects` rows on
/// the GPU.
pub const OBJECT_ROW_SNIPPET: helio_core::shader::ShaderSnippet =
    helio_core::wgsl_snippet!("//!use mesh_object_row", "shaders/object_row.wgsl");

/// The meshes' skinning in WGSL (`//!use mesh_skinning`): the skin cache's
/// rows ([`SkinnedVertex`], [`SkinCacheRow`]) and what a drawer reads,
/// `mesh_skinned_vertex` / `mesh_skinned_position` (a skinned instance's
/// vertex this frame and last, from the cache; the palette is blended only
/// by the meshes' skin pass, which writes the cache). A drawer
/// binds `skin_cache: array<SkinnedVertex>` ([`SKIN_CACHE`]) and
/// `skin_cache_rows: array<SkinCacheRow>` ([`SKIN_CACHE_ROWS`]) and declares
/// `GpuInstanceData` ([`DRAW_ROWS_SNIPPET`]).
pub const SKINNING_SNIPPET: helio_core::shader::ShaderSnippet =
    helio_core::wgsl_snippet!("//!use mesh_skinning", "shaders/skinning.wgsl");

/// Graph key of the rows that follow the frame's TLAS instance order: a
/// [`RayInstance`] a traced section, an instance's sections from its custom
/// data (its first row) on, a hit's row its custom data plus its
/// `geometry_index`.
pub const RAY_INSTANCES: &str = "ray_instances";

/// Graph key of each TLAS instance's transform, by TLAS instance index
/// ([`RayInstance::instance`]): a [`RayTransform`] an instance, the
/// object-to-frame matrix the TLAS holds (frame space: relative to the
/// frame's origin). What a pass shading hits without the ray query (a
/// surface pass over hits binned by material) takes a hit's normals and
/// tangents to the frame with. Both projections publish it beside the TLAS.
pub const RAY_INSTANCE_TRANSFORMS: &str = "ray_instance_transforms";

/// A TLAS instance's object-to-frame transform ([`RAY_INSTANCE_TRANSFORMS`]):
/// three rows of a 3x4 matrix (row-major, as `wgpu::TlasInstance` holds it),
/// 48 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RayTransform {
    pub rows: [[f32; 4]; 3],
}

impl RayTransform {
    /// The row-major 3x4 matrix `m` (a `TlasInstance::transform`).
    pub fn from_rows(m: [f32; 12]) -> Self {
        Self {
            rows: [
                [m[0], m[1], m[2], m[3]],
                [m[4], m[5], m[6], m[7]],
                [m[8], m[9], m[10], m[11]],
            ],
        }
    }
}

/// Graph key of the coordinate spaces' TLAS mask groups: 256 `u32`, one a
/// space slot (`RAY_SPACE_*` bits), so a ray restricted to one space
/// (`SceneRay::space`) is traced with the cull mask of that space's group
/// alone. An instance's TLAS mask is one bit, `1 << (2 * group + class)`,
/// class 0 for casters that never move and 1 for those that may (the frame's
/// `MASK_STATIC` / `MASK_MOVABLE` are group 0's bits): four groups of two
/// classes fill the 8 bits a mask has. Group 0 is the world's (slot 0)
/// alone; the other traced spaces take groups 1 to 3, one each while there
/// are at most three, else sharing them (`RAY_SPACE_SHARED`): a ray of a
/// shared group's space has its candidates tested against each row's space
/// ([`RayInstance::space`]), which costs it the any-hit loop, never the rays
/// of another group. Absent (a projection that traces only the world): the
/// world's group is 0 and no other space is traced.
pub const RAY_SPACE_GROUPS: &str = "ray_space_groups";
/// Space slots [`RAY_SPACE_GROUPS`] holds.
pub const RAY_SPACE_SLOTS: usize = 256;
/// [`RAY_SPACE_GROUPS`] entry: the group (bits 0..2).
pub const RAY_SPACE_GROUP: u32 = 3;
/// [`RAY_SPACE_GROUPS`] entry: the group holds other spaces too.
pub const RAY_SPACE_SHARED: u32 = 4;
/// [`RAY_SPACE_GROUPS`] entry: the space has traced instances.
pub const RAY_SPACE_PRESENT: u32 = 8;
/// Groups of the TLAS mask (two class bits each).
pub const RAY_SPACE_MASK_GROUPS: u32 = 4;

/// The TLAS mask of an instance of group `group` (`RAY_SPACE_GROUPS`), of
/// casters that may move (`movable`) or not.
pub const fn ray_instance_mask(group: u32, movable: bool) -> u8 {
    1 << (2 * group + movable as u32)
}

/// The space groups of the traced spaces `spaces` (slots, any order,
/// duplicates allowed): the [`RAY_SPACE_GROUPS`] table. The world (slot 0)
/// is group 0, present or not; the others take groups 1..=3 in slot order,
/// shared past three (by slot, round robin).
pub fn ray_space_groups(spaces: impl IntoIterator<Item = u32>) -> [u32; RAY_SPACE_SLOTS] {
    let mut present = [false; RAY_SPACE_SLOTS];
    for space in spaces {
        present[(space as usize).min(RAY_SPACE_SLOTS - 1)] = true;
    }
    let others: Vec<usize> = (1..RAY_SPACE_SLOTS).filter(|&s| present[s]).collect();
    let shared = others.len() > (RAY_SPACE_MASK_GROUPS - 1) as usize;
    let mut table = [0u32; RAY_SPACE_SLOTS];
    table[0] = if present[0] { RAY_SPACE_PRESENT } else { 0 };
    for (rank, &slot) in others.iter().enumerate() {
        let group = 1 + (rank as u32 % (RAY_SPACE_MASK_GROUPS - 1));
        table[slot] = group | RAY_SPACE_PRESENT | if shared { RAY_SPACE_SHARED } else { 0 };
    }
    table
}
/// Graph key of the proxy pool's vertices: the triangles of the instances
/// traced through a proxy ([`RAY_INSTANCE_PROXY`]), packed vertices of
/// [`RAY_PROXY_VERTEX_BYTES`] (a `Float32x3` position, then a `Float32x2`
/// UV), addressed as the mesh pools are (`first_index + 3p`, plus
/// `base_vertex`).
pub const RAY_PROXY_VERTICES: &str = "ray_proxy_vertices";
/// Graph key of the proxy pool's mesh-local `u32` indices.
pub const RAY_PROXY_INDICES: &str = "ray_proxy_indices";
/// Bytes of a proxy pool vertex: position, then UV.
pub const RAY_PROXY_VERTEX_BYTES: u64 = 20;

/// A traced section, for shading ray hits without a G-buffer
/// ([`RAY_INSTANCES`] rows, a hit's its instance's custom data plus its
/// `geometry_index`): where its triangles are in the scene's mesh pools, its
/// material and the program that evaluates it, its TLAS instance (whose
/// transform is [`RAY_INSTANCE_TRANSFORMS`]'s) and coordinate space.
/// Triangle `p` of the instance is pool indices `first_index + 3p ..`
/// (mesh-local), plus `base_vertex`, into `builtin_mesh_vertex`
/// (`PackedVertex`). A hit is shaded by its material, evaluated at the hit
/// as the G-buffer's drawers evaluate it (the meshes' tracer's surface
/// passes, by [`RayInstance::program`]). An instance with a nonzero
/// `transmission` is a thin sheet (glass) its BLAS marks non-opaque: rays
/// pass it, tinted.
///
/// An instance flagged [`RAY_INSTANCE_MASKED`] is a cut-out (an
/// alpha-tested material, row `material` of the scene's `materials`): its
/// BLAS is non-opaque too, and a ray query decides each candidate triangle
/// by the material's alpha at the hit's UV (`PackedVertex::tex_coords0`),
/// as the raster's alpha test does.
///
/// An instance flagged [`RAY_INSTANCE_PROXY`] is traced through a proxy of
/// its geometry (a coarse cut of a virtual-geometry asset): its triangles are
/// in the frame's proxy pool ([`RAY_PROXY_VERTICES`]) instead, positions and UVs
/// only; a hit's normal is its triangle's face (the winding says outside)
/// and its tangent its triangle's UV directions.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RayInstance {
    pub first_index: u32,
    pub base_vertex: u32,
    /// `RAY_INSTANCE_*` bits, and the coordinate space slot it is authored
    /// in at [`RAY_INSTANCE_SPACE_SHIFT`] (8 bits).
    pub flags: u32,
    /// The material's row in the scene's `materials` (and its texture
    /// metadata's row).
    pub material: u32,
    /// The linear RGB share of light it lets through (zero: opaque).
    pub transmission: [f32; 3],
    /// Its TLAS instance: the row of [`RAY_INSTANCE_TRANSFORMS`] holding
    /// its object-to-frame transform.
    pub instance: u32,
    /// For a [`RAY_INSTANCE_SKINNED`] instance, its first vertex in the skin
    /// cache ([`RAY_SKINNED_VERTICES`]): triangle corners, normals and
    /// tangents are read there (mesh-local index plus this), UVs and the
    /// bitangent's sign from the mesh pool as for any mesh. Zero otherwise.
    pub skinned_vertex: u32,
    /// The program that evaluates its material at a hit: an index of the
    /// frame's [`RayPrograms`] ([`RAY_PROGRAMS`]); 0 the table-driven
    /// evaluation ([`RayProgram::TABLE`]).
    pub program: u32,
    pub _pad: [u32; 2],
}

/// The program a material is evaluated by (the drawers' material key,
/// `helio_mats::RadiantShaderKey` without its raster flags): its material
/// class's template and, for a material graph, the graph's hash. A traced
/// section's [`RayInstance::program`] names one of the frame's
/// [`RayPrograms`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RayProgram {
    pub template: u32,
    pub graph_hash: u64,
}

impl RayProgram {
    /// The default material class with no graph: the table-driven
    /// evaluation every row's material falls back to (its row's factors and
    /// textures), always program 0.
    pub const TABLE: Self = Self {
        template: 0,
        graph_hash: 0,
    };
}

/// The programs the traced sections' materials are evaluated by, indexed by
/// [`RayInstance::program`]: entry 0 is always [`RayProgram::TABLE`]. The
/// projection that lays the rows out publishes it each frame with the TLAS
/// ([`RAY_PROGRAMS`]), so the tracer knows, without reading rows back, which
/// programs a frame's hits may need and compiles them before they do.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RayPrograms {
    programs: Vec<RayProgram>,
}

impl Default for RayPrograms {
    fn default() -> Self {
        Self {
            programs: vec![RayProgram::TABLE],
        }
    }
}

impl RayPrograms {
    /// `program`'s index, added if new.
    pub fn index(&mut self, program: RayProgram) -> u32 {
        match self.programs.iter().position(|p| *p == program) {
            Some(index) => index as u32,
            None => {
                self.programs.push(program);
                (self.programs.len() - 1) as u32
            }
        }
    }

    /// The programs, by index.
    pub fn programs(&self) -> &[RayProgram] {
        &self.programs
    }
}

/// The frame's [`RayPrograms`], published by the projection that writes
/// [`RAY_INSTANCES`] beside the TLAS.
pub const RAY_PROGRAMS: helio_core::PayloadContract<RayPrograms> =
    helio_core::PayloadContract::new("ray_programs");

/// [`RayInstance::flags`]: a skinned instance, its positions and normals
/// in the skin cache at [`RayInstance::skinned_vertex`] (the pose its BLAS
/// was last refit to, a frame stale at most when the refit budget defers it).
pub const RAY_INSTANCE_SKINNED: u32 = 8;

/// Graph key of the skin cache as the frame's ray-traced scene publishes it
/// beside its TLAS ([`SKIN_CACHE`]'s buffer): the vertices of the skinned
/// instances it traces.
pub const RAY_SKINNED_VERTICES: &str = "ray_skinned_vertices";

/// [`RayInstance::flags`]: the instance's triangles are in the frame's proxy
/// pool ([`RAY_PROXY_VERTICES`]), not the scene's mesh pools.
pub const RAY_INSTANCE_PROXY: u32 = 1;
/// [`RayInstance::flags`]: an alpha-tested cut-out (see [`RayInstance`]).
pub const RAY_INSTANCE_MASKED: u32 = 2;
/// [`RayInstance::flags`]: both faces are its outside (a double-sided
/// material): a ray meeting its back has met a surface, not an inside.
pub const RAY_INSTANCE_TWO_SIDED: u32 = 4;
/// [`RayInstance::flags`]: where its coordinate space slot is (bits 8..16).
pub const RAY_INSTANCE_SPACE_SHIFT: u32 = 8;

impl RayInstance {
    /// The coordinate space slot it is authored in.
    pub fn space(&self) -> u32 {
        (self.flags >> RAY_INSTANCE_SPACE_SHIFT) & 0xff
    }

    /// The row in space slot `space`.
    pub fn in_space(mut self, space: u32) -> Self {
        self.flags = (self.flags & !(0xff << RAY_INSTANCE_SPACE_SHIFT))
            | ((space & 0xff) << RAY_INSTANCE_SPACE_SHIFT);
        self
    }
}
