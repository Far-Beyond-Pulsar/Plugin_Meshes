//! Meshes (`pulsar.meshes`): the editor crate -- the level editor's mesh
//! entries and icon ([`level_editor`]) and the asset browser's thumbnails of
//! mesh files ([`thumbnails`]). It registers nothing itself: the plugin's
//! root crate (`meshes_plugin`, with its `editor` feature) submits these
//! through the level editor's and the asset browser's extension points.

// The level editor UI's text (`locales/`).
rust_i18n::i18n!("locales", fallback = "en");

pub mod level_editor;
pub mod thumbnails;
