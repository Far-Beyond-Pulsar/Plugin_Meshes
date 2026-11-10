//! Compute shaders compile under FXC (see `test-support/fxc.rs`): Windows on ARM has no DXC. Slow.

#![cfg(windows)]

#[path = "../../../test-support/fxc.rs"]
mod fxc;

#[test]
fn compute_shaders_compile_under_fxc() {
    let Some(fxc_exe) = fxc::find_fxc() else {
        eprintln!("fxc not found; skipping");
        return;
    };
    let sources: Vec<(&str, String)> = vec![
        ("scene_join_meshes", helio_pass_object_batch::scene_join::mesh_join_source()),
        ("object_batch", format!("{}
{}", include_str!("../shaders/object_batch.wgsl"), helio_mats::MATERIAL_ROW_SNIPPET.source)),
    ];
    let mut failures = Vec::new();
    for (name, src) in sources {
        failures.extend(fxc::fxc_failures(&fxc_exe, name, &helio_core::shader::resolve(&src)));
    }
    assert!(failures.is_empty(), "FXC rejected:\n{}", failures.join("\n"));
}
