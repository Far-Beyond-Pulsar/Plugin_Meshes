//! What the meshes plugin registers for the editor (the `editor` feature):
//! its level editor entries and the asset browser's thumbnails.

use meshes_editor::{level_editor, thumbnails};
use ui_common::asset_thumbnails::ThumbnailRendererRegistration;
use ui_level_editor::extend::{CreateEntry, ObjectIconRegistration};

/// The Create menu's primitive meshes ([`level_editor::PRIMITIVES`]).
macro_rules! primitive_entries {
    ($p:expr; $($index:literal),*) => {$(
        $p.add(CreateEntry {
            origin: $p.origin(),
            id: level_editor::PRIMITIVES[$index].id,
            menu_label: Some(level_editor::PRIMITIVES[$index].menu_label),
            environment_label: None,
            quick_add: None,
            offered: None,
            order: level_editor::PRIMITIVES[$index].order,
            create: level_editor::PRIMITIVES[$index].create,
        });
    )*};
}

/// Source models, drawn by the meshes' snapshot renderer
/// ([`thumbnails::MODEL_EXTENSIONS`]).
macro_rules! model_thumbnails {
    ($p:expr; $($index:literal),*) => {$(
        $p.add(ThumbnailRendererRegistration {
            origin: $p.origin(),
            extension: thumbnails::MODEL_EXTENSIONS[$index],
            render: thumbnails::render_model_thumbnail,
        });
    )*};
}

/// Everything the plugin registers for the editor.
pub(crate) fn build(p: &mut pulsar_plugins::PluginBuilder) {
    // ── Level editor ────────────────────────────────────────────────────────

    // An object holding a mesh shows as one in the hierarchy.
    p.add(ObjectIconRegistration {
        origin: p.origin(),
        id: "meshes",
        order: 200,
        icon: level_editor::object_icon,
    });

    primitive_entries!(p; 0, 1, 2, 3);

    // ── Asset browser ───────────────────────────────────────────────────────

    // Native `.mesh` assets, rasterised on the CPU.
    p.add(ThumbnailRendererRegistration {
        origin: p.origin(),
        extension: "mesh",
        render: thumbnails::render_mesh_thumbnail,
    });

    model_thumbnails!(p; 0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
}
