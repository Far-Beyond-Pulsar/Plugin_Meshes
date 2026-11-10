//! Every dense shader compiles under FXC (see `test-support/fxc.rs`). Slow.

#![cfg(windows)]

#[path = "../../../test-support/fxc.rs"]
mod fxc;

use helio_pass_vg_dense::config::{validation_sources, DenseConfig};

#[test]
fn dense_shaders_compile_under_fxc() {
    let Some(fxc_exe) = fxc::find_fxc() else {
        eprintln!("fxc not found; skipping");
        return;
    };
    let cfg = DenseConfig::default();
    let mut failures = Vec::new();
    for (name, src) in validation_sources(&cfg) {
        // Ray-query shaders need SM 6.3+, so wgpu always compiles them with DXC.
        if name == "rt" {
            continue;
        }
        failures.extend(fxc::fxc_failures(&fxc_exe, name, &src));
    }
    assert!(failures.is_empty(), "FXC rejected:
{}", failures.join("
"));
}
