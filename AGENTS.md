# Plugin_Meshes

`pulsar.meshes`: static, skeletal and virtual-geometry meshes: their batching, culling, drawers and asset import.
Pulsar-Native vendors this repository at `plugins/vendor/meshes`; it builds only
inside a Pulsar-Native checkout (see `README.md`).

## Required for every agent

Before reading, designing, editing, reviewing or briefing any change here,
load the **`helio-architecture`** skill. Its single source is Helio's
[`.agents/skills/helio-architecture/SKILL.md`](https://github.com/Far-Beyond-Pulsar/Helio/blob/main/.agents/skills/helio-architecture/SKILL.md); in a Pulsar-Native
checkout it is at `crates/renderer/helio/.agents/skills/helio-architecture/SKILL.md`.
That applies to bug fixes, perf work, tests and small changes too.

Follow the skill's steps in order and end with its report section (Design,
AAA comparison, Deleted, Verified). Anything you brief (a subagent, another
session or another agent) must load the skill too. The skill is the single
source of these rules: this file only points to it, so change the skill
rather than copying rules here.

## Docs

| File | What |
|---|---|
| [Helio `docs/architecture/plugin-migration.md`](https://github.com/Far-Beyond-Pulsar/Helio/blob/main/docs/architecture/plugin-migration.md) | Every object kind is a plugin; the plugin table; "As built" notes per phase |
| [Helio `docs/architecture/layered-rewrite.md`](https://github.com/Far-Beyond-Pulsar/Helio/blob/main/docs/architecture/layered-rewrite.md) | Layers, rules, buffer contracts, extension points |
| Pulsar-Native `AGENTS.md`, `.agents/PLUGIN_SYSTEM.md`, `.agents/CRATES.md` | Plugin descriptors, the installed set, submodule pins |

## Key commands (from the Pulsar-Native root)

```
cargo check -p helio-android-demos -p helio-mesh-api -p helio-asset-compat -p meshes_component -p meshes_plugin -p meshes-examples -p helio-pass-forward-lit -p helio-pass-gbuffer -p helio-pass-indirect-dispatch -p helio-pass-object-batch -p helio-pass-occlusion-cull -p helio-pass-shadow -p helio-pass-shadow-dirty -p helio-pass-transparent -p helio-pass-vg -p helio-pass-vg-bvh -p helio-pass-vg-cull -p helio-pass-vg-dense -p helio-pass-vg-raster -p helio-pass-vg-shade -p helio-mesh-host -p helio-snapshot -p helio-vg-build -p helio-vg-format -p helio-vg-stream -p helio-web-demos --tests
cargo test -p helio-mesh-api -p meshes_component -p helio-pass-forward-lit -p helio-pass-gbuffer -p helio-pass-indirect-dispatch -p helio-pass-object-batch -p helio-pass-occlusion-cull -p helio-pass-shadow -p helio-pass-shadow-dirty -p helio-pass-transparent -p helio-pass-vg -p helio-pass-vg-bvh -p helio-pass-vg-cull -p helio-pass-vg-dense -p helio-pass-vg-raster -p helio-pass-vg-shade
cargo test -p one_owner          every concept has one owner (scans this plugin)
cargo test -p installed_plugins  the installed set and absent plugins
```
