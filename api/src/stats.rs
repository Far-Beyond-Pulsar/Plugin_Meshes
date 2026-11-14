//! The meshes' frame statistics (`helio_core::stats`): what the culling
//! passes count on the GPU, by name. They also add the main view's issued
//! draws, vertices and instances to Helio's shared `draw_calls`, `vertices`
//! and `instances`, which other kinds add to as well.

/// Draw-call groups (batches) culled this frame.
pub const BATCHES: &str = "meshes.batches";
/// Batches with no instance in the frustum.
pub const BATCHES_FRUSTUM_CULLED: &str = "meshes.batches_frustum_culled";
/// Batches whose visible instances were all smaller than a pixel.
pub const BATCHES_SUBPIXEL: &str = "meshes.batches_subpixel";
/// Batches the Hi-Z occlusion test emptied.
pub const BATCHES_OCCLUDED: &str = "meshes.batches_occluded";
/// Indirect draws the main view issued with at least one instance.
pub const DRAWS: &str = "meshes.draws";
/// Batches holding a shadow caster.
pub const SHADOW_BATCHES: &str = "meshes.shadow_batches";
/// Issued draws holding a shadow caster.
pub const SHADOW_DRAWS: &str = "meshes.shadow_draws";
