//! Virtual-geometry offline builder (plan: `.agents/plans/virtual-geometry.md`,
//! section 2.1 / task T2).
//!
//! Pure CPU: an indexed triangle mesh goes in, a `.vgeo` byte blob comes out
//! (cluster DAG + always-resident hierarchy + 128 KiB pages). No GPU
//! dependencies; the output is a deterministic function of the input and the
//! [`BuildOptions`].
//!
//! Pipeline: [`mesh::weld`] -> per-material [`cluster`] -> repeat
//! { [`partition`] (`meshopt_partitionClusters`), [`simplify`] toward half
//! the triangles with the group border locked, attribute-aware quadrics,
//! optimal vertex placement and an error limit (see [`dag`]), split,
//! [`deviation`] } -> [`pack::write_vgeo`].
//!
//! Error metric: every group's error is a rigorous two-sided Hausdorff bound
//! between its simplified surface and the original (level-0) surface it
//! stands for, with the normal / uv change converted to the same unit
//! (module `deviation`); parent error is `max(children, measured)`, never a
//! sum and never a safety factor. The simplifier's own estimate only steers
//! the collapses.

mod cluster;
mod dag;
mod deviation;
mod file;
mod math;
mod mesh;
mod pack;
mod partition;
mod reader;
mod simplify;
// meshopt's C++ allocator on wasm32 (no C++ standard library there).
#[cfg(target_arch = "wasm32")]
mod wasm_cpp_alloc;

pub use dag::{build_dag, ClusterNode, Dag, Group, NONE};
pub use file::{
    DecodedCluster, PageDirEntry, Projection, SelectedCluster, VgeoMeta, LATTICE_BITS, LATTICE_MAX,
    POS_BLOCK_HEADER_WORDS,
};
pub use math::sphere_contains;
pub use mesh::{BuildError, MeshInput, VertexStore};
pub use pack::{write_vgeo, BuildOptions, MAX_ROOT_PAGES};
pub use reader::{PageView, ParseError, VgeoReader};

/// Builder revision stored in the upper 16 bits of asset flags. The binary
/// format stays readable, but derived caches from older algorithms must cook
/// again. Revision 2 fixed accumulated QEM error and premature hierarchy stop;
/// revision 3 measures each group's error against the original surface
/// (`deviation`), reduces 40% per level and stops stalled levels. Revision 5 builds a real DAG for disconnected/island geometry (floating pieces, proximity merging, appearance error) and bounds the pinned root set to `MAX_ROOT_PAGES`.
/// Revision 7 restores revision 5 foliage simplification and coverage error.
/// Revision 6 produced distorted foliage and excessive refinement; recook it.
/// Revision 12 stores the true deviation (position, normal and uv, no baked density relaxation).
/// Revision 13 simplifies with attribute-aware quadrics and optimal vertex
/// placement under an error limit, partitions with meshoptimizer and stores a
/// rigorous two-sided deviation from the original surface.
pub const COOK_REVISION: u32 = 13;

pub fn is_current_cook(header: &helio_vg_format::VgeoHeader) -> bool {
    header.asset.flags >> 16 == COOK_REVISION
}

/// Build a `.vgeo` from a mesh. Same input and options give the same bytes
/// regardless of thread count.
pub fn build_vgeo(mesh: &MeshInput, opts: &BuildOptions) -> Result<Vec<u8>, BuildError> {
    let dag = build_dag(mesh)?;
    write_vgeo(&dag, mesh.attributes, mesh.attribute_stride, opts)
}
