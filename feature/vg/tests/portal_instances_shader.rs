//! The portal instances kernel (`VgPortalInstancesPass`) parses and
//! validates as the pass compiles it: no device needed.

use helio_pass_vg::VgPortalInstancesPass;
use naga::valid::{Capabilities, ValidationFlags, Validator};

#[test]
fn the_portal_instances_kernel_validates() {
    let raw = VgPortalInstancesPass::source();
    let source = helio_core::shader::resolve_with(&raw, &VgPortalInstancesPass::SNIPPETS);
    let module = naga::front::wgsl::parse_str(&source)
        .unwrap_or_else(|e| panic!("vg_portal_instances.wgsl:\n{}", e.emit_to_string(&source)));
    Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .unwrap_or_else(|e| panic!("vg_portal_instances.wgsl: {e:?}"));
    assert!(module
        .entry_points
        .iter()
        .any(|e| e.name == "cs_portal_instances"));
}
