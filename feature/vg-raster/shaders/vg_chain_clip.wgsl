//!use portal_chain
// The clip of a run drawn through another space (`VgRasterConfig::chain_clip`,
// a run whose instances carry remaps: what a portal chain shows, its
// instances already mapped where they are drawn). A fragment of instance `i`
// keeps only what its chain shows: beyond the chain's outermost opening where
// the aperture mask holds it, beyond each deeper opening the camera sees
// through, and not behind a deeper chain's opening (`//!use portal_chain`,
// tested in the content's own world: the drawn position taken back through
// the chain's remap).
//
// Group 3 of every raster pipeline of such a run.

@group(3) @binding(0) var<storage, read> vg_remaps: array<u32>;
@group(3) @binding(1) var<storage, read> vg_portal_chains: array<GpuPortalChain>;
@group(3) @binding(2) var<storage, read> vg_portal_stages: array<GpuPortalStage>;
@group(3) @binding(3) var vg_portal_mask: texture_2d<u32>;
// `helio_frame_api::lighting::VIEW_REMAPS`: drawn space to the content's world.
@group(3) @binding(4) var<uniform> vg_view_remaps: array<mat4x4<f32>, 1024>;

fn vg_chain_clipped(instance: u32, world: vec3<f32>, pixel: vec2<i32>) -> bool {
    // A chain's remap is its index (`GpuPortalChain::remap`).
    let chain_index = vg_remaps[instance];
    if chain_index >= arrayLength(&vg_portal_chains) {
        return true;
    }
    let chain = vg_portal_chains[chain_index];
    let p = (vg_view_remaps[min(chain.remap, 1023u)] * vec4<f32>(world, 1.0)).xyz;
    let mask = textureLoad(vg_portal_mask, pixel, 0).r;
    if !portal_outer_shows(chain, vg_portal_stages[chain.first_stage], p, mask) {
        return true;
    }
    for (var s = 1u; s < chain.stage_count; s += 1u) {
        if !portal_stage_shows(vg_portal_stages[chain.first_stage + s], p) {
            return true;
        }
    }
    for (var h = 0u; h < chain.hole_count; h += 1u) {
        if portal_hole_covers(vg_portal_stages[chain.first_hole + h], p) {
            return true;
        }
    }
    return false;
}
