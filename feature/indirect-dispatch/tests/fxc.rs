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
        ("indirect_dispatch", include_str!("../shaders/indirect_dispatch.wgsl").to_string()),
    ];
    let mut failures = Vec::new();
    for (name, src) in sources {
        failures.extend(fxc::fxc_failures(&fxc_exe, name, &helio_core::shader::resolve(&src)));
    }
    assert!(failures.is_empty(), "FXC rejected:\n{}", failures.join("\n"));
}
