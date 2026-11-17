//! Every raster compute shader compiles under FXC (see `test-support/fxc.rs`). Slow.

#![cfg(windows)]

#[path = "../../../test-support/fxc.rs"]
mod fxc;

#[test]
fn raster_shaders_compile_under_fxc() {
    let Some(fxc_exe) = fxc::find_fxc() else {
        eprintln!("fxc not found; skipping");
        return;
    };
    let mut failures = Vec::new();
    for (name, src) in helio_pass_vg_raster::shaders::all_sources() {
        // 64-bit atomics need SM 6.6, so wgpu compiles these with DXC.
        if name.starts_with("vg_sw_raster") {
            continue;
        }
        // Resolved as the raster resolves it (the chain clip's snippet).
        let src = helio_core::shader::resolve_with(&src, &[helio_portal_api::PORTAL_CHAIN_SNIPPET]);
        failures.extend(fxc::fxc_failures(&fxc_exe, name, &src));
    }
    assert!(
        failures.is_empty(),
        "FXC rejected:\n{}",
        failures.join("\n")
    );
}
