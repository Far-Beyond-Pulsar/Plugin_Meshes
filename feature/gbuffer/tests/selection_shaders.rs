//! The meshes' selection programs are valid WGSL once their snippets (and the
//! prelude) are resolved, validated without a device.

fn validate(name: &str, source: &str, snippets: &[helio_core::shader::ShaderSnippet]) {
    let resolved = helio_core::shader::resolve_with(source, snippets);
    let module = naga::front::wgsl::parse_str(&resolved)
        .unwrap_or_else(|error| panic!("{name}: {}", error.emit_to_string(&resolved)));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|error| panic!("{name}: {error:?}"));
}

#[test]
fn the_selection_select_program_validates() {
    validate(
        "selection select",
        include_str!("../shaders/selection_select.wgsl"),
        &[helio_mesh_api::DRAW_ROWS_SNIPPET],
    );
}

#[test]
fn the_selection_mask_program_validates() {
    validate(
        "selection mask",
        include_str!("../shaders/selection_mask.wgsl"),
        &[
            helio_mesh_api::DRAW_ROWS_SNIPPET,
            helio_mesh_api::SKINNING_SNIPPET,
        ],
    );
}

/// The selected draws are the hardware's indexed indirect args, as the
/// shader writes them.
#[test]
fn a_selected_draw_is_one_indexed_indirect_record() {
    assert_eq!(
        std::mem::size_of::<helio_mesh_api::DrawIndexedIndirectArgs>(),
        20
    );
}
