//! The plugin's `examples/` folder: small source models (OBJ with its
//! material library, glTF) for the model importer the plugin registers,
//! served read-only in the plugin's virtual folder and embedded from
//! `examples_assets/`.

use std::sync::Arc;

use engine_fs::{FolderSource, InGame, MemoryFolder, PluginFolder};

/// The example files, embedded; each path is where it sits in the plugin's
/// folder.
pub const FILES: &[engine_fs::EmbeddedFile] = engine_fs::embedded_files!("examples_assets":
    "examples/README.md",
    "examples/models/cube.obj",
    "examples/models/pyramid.obj",
    "examples/models/pyramid.mtl",
    "examples/models/floor_tile.gltf",
);

fn open(folder: PluginFolder) -> Arc<dyn FolderSource> {
    Arc::new(MemoryFolder::with_files(folder, FILES))
}

/// The plugin's one file provider.
pub(crate) fn provider(origin: &'static str) -> engine_fs::FileProvider {
    engine_fs::FileProvider {
        origin,
        open,
        // A game does not ask the plugin: it serves the packager's snapshot.
        in_game: InGame::Snapshot,
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use asset_import::{importer_for, OptionValues, Previous};

    use super::*;

    /// Every model imports with the plugin's own importer into a native mesh
    /// that has geometry.
    #[test]
    fn every_model_imports_with_the_plugins_importer() {
        // The importer registers itself when the component crate is linked.
        use meshes_component as _;

        let dir = tempfile::tempdir().unwrap();
        // The whole folder, so the `.obj` finds its `.mtl` beside it.
        for file in FILES {
            let path = dir.path().join(file.path.trim_start_matches("examples/"));
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, file.bytes).unwrap();
        }
        let mut imported = 0;
        for file in FILES {
            let path = Path::new(file.path);
            let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
                continue;
            };
            let Some(importer) = importer_for(ext) else {
                // `.mtl` and the README are not models.
                assert!(matches!(ext, "mtl" | "md"), "{} has no importer", file.path);
                continue;
            };
            let source = dir.path().join(file.path.trim_start_matches("examples/"));
            let asset = source.with_extension("mesh");
            let result = importer
                .import(
                    &source,
                    &asset,
                    &OptionValues::new(),
                    &Previous {
                        project_root: dir.path(),
                        outputs: &[],
                    },
                )
                .unwrap_or_else(|e| panic!("{}: {e}", file.path));
            assert!(
                result.asset.is_file() && std::fs::metadata(&result.asset).unwrap().len() > 0,
                "{} imported to nothing",
                file.path
            );
            imported += 1;
        }
        assert_eq!(imported, 3);
    }

    #[test]
    fn the_readme_is_there() {
        assert!(FILES.iter().any(|f| f.path == "examples/README.md"));
    }
}
