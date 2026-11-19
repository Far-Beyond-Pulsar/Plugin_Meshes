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

/// Bytes of the skin cache: its vertices (resident skinned instances'
/// posed vertices) and its instance table.
pub const SKIN_CACHE_BYTES: &str = "meshes.skin_cache_bytes";
/// Skinned instances the skin cache holds.
pub const SKIN_CACHE_INSTANCES: &str = "meshes.skin_cache_instances";
/// Skinned instances the skin pass posed this frame (their pose changed).
pub const SKIN_CACHE_UPDATED: &str = "meshes.skin_cache_updated";
/// Vertices the skin pass wrote this frame (posed and settled).
pub const SKIN_CACHE_VERTICES: &str = "meshes.skin_cache_vertices";
/// Instances the meshes' ray-traced scene holds this frame (its TLAS).
pub const RAY_INSTANCES: &str = "meshes.ray_instances";
/// Instances the ray-tracing culling left out this frame (past the culling
/// radius and smaller than the culling angle).
pub const RAY_INSTANCES_CULLED: &str = "meshes.ray_instances_culled";
/// Programs the ray-traced scene's rows name this frame
/// (`crate::RAY_PROGRAMS`), the table-driven one included.
pub const RAY_PROGRAMS: &str = "meshes.ray_programs";
/// Bins the meshes' hit surfaces ran in this frame, summed over the ray
/// streams (one a stream when every hit takes the table-driven program).
pub const HIT_BINS: &str = "meshes.hit_bins";
/// Hit programs compiling (off the frame) this frame.
pub const HIT_PROGRAMS_PENDING: &str = "meshes.hit_programs_pending";
/// Hit programs whose compile failed (their hits shade by the table).
pub const HIT_PROGRAMS_FAILED: &str = "meshes.hit_programs_failed";
/// Programs the meshes' hit surfaces were handed this frame (the
/// `crate::RAY_PROGRAMS` payload's), summed over the ray streams.
pub const HIT_PROGRAMS_SEEN: &str = "meshes.hit_programs_seen";
/// Hit programs waiting for their material graph's source to be registered
/// (their hits shade by the table meanwhile).
pub const HIT_PROGRAMS_WAITING: &str = "meshes.hit_programs_waiting";
