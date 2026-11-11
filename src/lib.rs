//! Meshes (`pulsar.meshes`): the plugin's root crate, the one place the
//! plugin registers everything it contributes.
//!
//! The plugin's other crates hold the code: `helio-mesh-api` (the contracts
//! other kinds read), the feature crates (`helio-pass-*`: the object batch,
//! culling, the drawers, virtual geometry), `meshes_component` (the SceneDB
//! components, their methods, the scene join, import) and `meshes_editor`
//! (the level editor UI and thumbnails). None of them registers a
//! contribution; this crate imports them and registers them
//! through its [`build`], so
//! this file and the two modules below are the whole list of what the plugin
//! brings:
//!
//! - [`runtime`]: what a game links. The component classes, the Helio
//!   features, the scene join and its columns, the subsystems, picking and
//!   framing geometry, animated content, movability, skinned mesh access,
//!   the headless report, the project settings, the asset seams, the model
//!   importer and the level record migrations.
//! - `editor` (with the `editor` feature): what the editor adds. The
//!   hierarchy icon, the Create menu's primitive meshes and the asset
//!   browser's thumbnails.
//!
//! Every registration names this crate as its origin
//! (`p.origin()`), so the plugin gate skips all of them while
//! a project disables the plugin, and the plugin catalog lists them under
//! `pulsar.meshes`. Other plugins never depend on this crate (only on
//! `helio-mesh-api`); hosts reach it through the installed set or a
//! generated game, a game without the `editor` feature.

// The plugin's descriptor (`pulsar-plugin.toml`): linking this crate installs
// the plugin; a project enables it or not (`pulsar_plugins`).
pulsar_plugins::installed!("pulsar-plugin.toml", build);

/// Everything the plugin registers (runtime always, editor with the `editor`
/// feature).
pub fn build(p: &mut pulsar_plugins::PluginBuilder) {
    runtime::build(p);
    #[cfg(feature = "editor")]
    editor::build(p);
}

pub use helio_mesh_api;
pub use meshes_component;
#[cfg(feature = "editor")]
pub use meshes_editor;

pub mod runtime;

#[cfg(feature = "editor")]
pub mod editor;
