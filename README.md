# Plugin_Meshes

Static, skeletal and virtual-geometry meshes: their batching, culling, drawers and asset import.

`pulsar.meshes` ("Meshes") is a plugin of
[Pulsar-Native](https://github.com/Far-Beyond-Pulsar/Pulsar-Native), the engine and editor that renders with
[Helio](https://github.com/Far-Beyond-Pulsar/Helio). Pulsar-Native vendors this repository as a git submodule
at `plugins/vendor/meshes`.

## What it owns

Every object kind is a plugin. A plugin owns its SceneDB components (their
data and scriptable methods; the editor calls the same methods scripts do),
the contracts other kinds read (its `api` crate: graph keys, row ABIs, WGSL
snippets), the Helio feature that renders it (passes, scene derivations,
settings), its file formats and its editor UI. Helio knows no object kind,
and other kinds meet this one only through frame contracts, extension points,
draw phases and api crates.

| Crate | Path | What |
|---|---|---|
| `helio-android-demos` | `android-demos/` | Helio demos compiled for Android — one feature per demo |
| `helio-mesh-api` | `api/` | The meshes' public contract: the rows their components author, the batch, draw and frame data their passes publish, the ray-instance rows their acceleration publishes, and their WGSL. The one crate another kind reads a mesh concept from. |
| `helio-asset-compat` | `asset-compat/` | SolidRS integration layer for Helio renderer - asset loading and conversion |
| `meshes_component` | `component/` | Meshes (pulsar.meshes): the SceneDB components (static and skeletal meshes, their draw rows, authored LOD, virtual-geometry overrides), mesh assets and their import, the scene join, LOD selection, the virtual-geometry host and picking, and the registrations of the plugin's Helio features. |
| `meshes_plugin` | `editor/` | Meshes (pulsar.meshes): the editor plugin -- its metadata, over the plugin's component crate |
| `meshes-examples` | `examples/` | The meshes plugin's standalone hosts on Helio: its demos and GPU tests |
| `helio-pass-forward-lit` | `feature/forward/` | Helio pass: the forward-lit mesh drawer (presets without a G-buffer) |
| `helio-pass-gbuffer` | `feature/gbuffer/` | Helio pass: the meshes' G-buffer drawer (opens the G-buffer phase) |
| `helio-pass-indirect-dispatch` | `feature/indirect-dispatch/` | Helio pass: builds the meshes' indirect draw arguments |
| `helio-pass-object-batch` | `feature/object-batch/` | GPU-driven sort/group/batch pipeline: turns SceneDB's StaticObjectComponent rows into instances/draw_calls/material_class_ranges with zero per-frame CPU involvement |
| `helio-pass-occlusion-cull` | `feature/occlusion-cull/` | Helio pass: Hi-Z occlusion culling of the mesh instances |
| `helio-pass-shadow` | `feature/shadow/` | Helio pass: the meshes' shadow-caster drawer (opens the shadow-caster phase) |
| `helio-pass-shadow-dirty` | `feature/shadow-dirty/` | Helio pass: which shadow views need redrawing |
| `helio-pass-transparent` | `feature/transparent/` | Helio pass: the meshes' transparent drawer |
| `helio-pass-vg` | `feature/vg/` | Virtual geometry's graph glue as a Helio feature: the pass chain, its order, and the passes that tie the streaming store to the graph. |
| `helio-pass-vg-bvh` | `feature/vg-bvh/` | Virtual geometry: GPU Morton LBVH (GCBVH top level) over VG instances, with batch leaves and refit-only updates |
| `helio-pass-vg-cull` | `feature/vg-cull/` | Virtual geometry GPU culling: instance BVH + cluster DAG cut, two-phase Hi-Z occlusion, tier/bin output |
| `helio-pass-vg-dense` | `feature/vg-dense/` | Virtual geometry Tier 3 (ultra-dense overlap): tiled depth-peel compute raster (3-NonRT) and ray-query primary visibility over proxy-LOD BLASes (3-RT) |
| `helio-pass-vg-raster` | `feature/vg-raster/` | Helio virtual geometry raster: Tier 1 hardware visbuffer raster, Tier 2 software micro-polygon raster, tile classification |
| `helio-pass-vg-shade` | `feature/vg-shade/` | Virtual geometry deferred shading: vis resolve, material binning, G-buffer attribute/material evaluation and final depth |
| `helio-mesh-host` | `host/` | The meshes' SceneDB components and ray-tracing projection for standalone Helio hosts (pulsar.meshes): the plugin's examples, tests, web and Android demos. Engines register their own components. |
| `helio-snapshot` | `snapshot/` | Headless snapshot renderer for Helio — loads a model file, renders one frame, saves a PNG |
| `helio-vg-build` | `vg/build/` | Virtual geometry cluster-DAG builder and .vgeo writer/reader (pure CPU, no GPU dependencies) |
| `helio-vg-format` | `vg/format/` | Virtual geometry cluster/page/hierarchy formats shared by the offline builder, streamer and GPU passes |
| `helio-vg-stream` | `vg/stream/` | Virtual geometry page streaming: GPU page pool, residency table, request readback, upload scheduler |
| `helio-web-demos` | `web-demos/` | The meshes plugin's scenes compiled to WASM, one feature per demo; `cargo run --bin web` builds every demo and serves them |

- **Runtime module** (what a game built for a project links): `meshes_component`.
- **Editor module** (what the editor links in addition): `meshes_plugin`.
- **Requires**: nothing. A project enables or disables the plugin in its
  `Pulsar.toml` (`[plugins]`); disabled, the kind is absent and the frame
  draws without it.
- **Reads other kinds through**: `helio-light-api, helio-pass-shadow-matrix` (lights); `helio-mats` (materials).
- **Its standalone hosts also compose**: `helio-light-api, helio-light-host, helio-pass-billboard, helio-pass-deferred-light, helio-pass-hlfs, helio-pass-light-cull` (lights); `helio-mats, helio-pass-materials` (materials) (demos and GPU tests
  draw a whole scene; the plugin itself does not depend on these).

## Building and testing

Like [Plugin_Physics](https://github.com/Far-Beyond-Pulsar/Plugin_Physics), this plugin's crates are members
of Pulsar-Native's workspace: their dependencies are `workspace = true`, and
Helio comes from Pulsar-Native's `crates/renderer/helio` submodule. It builds
only inside a Pulsar-Native checkout:

```bash
git clone --recursive https://github.com/Far-Beyond-Pulsar/Pulsar-Native
cd Pulsar-Native
cargo check -p helio-android-demos -p helio-mesh-api -p helio-asset-compat -p meshes_component -p meshes_plugin -p meshes-examples -p helio-pass-forward-lit -p helio-pass-gbuffer -p helio-pass-indirect-dispatch -p helio-pass-object-batch -p helio-pass-occlusion-cull -p helio-pass-shadow -p helio-pass-shadow-dirty -p helio-pass-transparent -p helio-pass-vg -p helio-pass-vg-bvh -p helio-pass-vg-cull -p helio-pass-vg-dense -p helio-pass-vg-raster -p helio-pass-vg-shade -p helio-mesh-host -p helio-snapshot -p helio-vg-build -p helio-vg-format -p helio-vg-stream -p helio-web-demos --tests
cargo test -p helio-mesh-api -p meshes_component -p helio-pass-forward-lit -p helio-pass-gbuffer -p helio-pass-indirect-dispatch -p helio-pass-object-batch -p helio-pass-occlusion-cull -p helio-pass-shadow -p helio-pass-shadow-dirty -p helio-pass-transparent -p helio-pass-vg -p helio-pass-vg-bvh -p helio-pass-vg-cull -p helio-pass-vg-dense -p helio-pass-vg-raster -p helio-pass-vg-shade
```

Suites that render real frames need a GPU; CI leaves them out by name
(Pulsar-Native's `.config/nextest.toml`, `ci` profile). This repository's CI
checks the plugin inside a Pulsar-Native checkout: rustfmt, clippy and the
`ci` profile's tests.

A change that spans this plugin and Pulsar-Native (or Helio) merges
dependency-first: merge here, re-pin `plugins/vendor/meshes` in Pulsar-Native to
this repository's `main`, then merge the Pulsar-Native PR (Pulsar-Native's
`.agents/CRATES.md`, "Submodule pins").

## Architecture

- [Every object kind is a plugin](https://github.com/Far-Beyond-Pulsar/Helio/blob/main/docs/architecture/plugin-migration.md): the
  rule, what Helio is, the plugin table and the "As built" notes of each
  phase.
- [The layered rewrite](https://github.com/Far-Beyond-Pulsar/Helio/blob/main/docs/architecture/layered-rewrite.md): layers, buffer
  contracts, extension points and why the renderer has this shape.
- [The `helio-architecture` skill](https://github.com/Far-Beyond-Pulsar/Helio/blob/main/.agents/skills/helio-architecture/SKILL.md): required reading for any change
  here (see `AGENTS.md`).
- Pulsar-Native's `.agents/PLUGIN_SYSTEM.md` and `.agents/CRATES.md`: plugin
  descriptors, the installed set and the submodule pins.

History before 2026-11-02 (when the plugin moved into Pulsar-Native) is in
Helio's repository; this repository's history starts with that move
(`git subtree split` of `plugins/vendor/meshes`).

## License

MIT, as the rest of Pulsar (`LICENSE.md`).
