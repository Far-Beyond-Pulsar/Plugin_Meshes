# VG forest example

A second scene of the `vg_stress` binary: a procedural mountain valley, about
3 km across, covered with up to 30,000 Poly Haven trees drawn through virtual
geometry, under the sky plugin's atmosphere with GI. It shares the loader,
cooker, capture mode, debug views and controls with the museum
([VG_STRESS.md](VG_STRESS.md)); this file covers only what differs.

## Running it

From the Pulsar-Native repository root (PowerShell):

```powershell
cargo run -p meshes-examples --features vg-stress --bin vg_stress --release -- --scene forest
```

`VG_STRESS_SCENE=forest` does the same (`--scene` wins over the environment;
`museum` is the default). Other settings work as in the museum:

| Setting | CLI | Environment | Forest meaning |
|---|---|---|---|
| Scene | `--scene forest` | `VG_STRESS_SCENE` | `museum` (default) or `forest` |
| Density | `--density <v>` | `VG_STRESS_DENSITY` | tree count = 30,000 x density (`low` 0.4, `normal` 1, `high` 3, `insane` 8, or a number, clamped to 128) |
| Trees | `--trees <v>` / `--small-trees` | `VG_STRESS_TREES` | species to scatter: `full` (default), `small`, `none`, or a list of ids |
| Tau | | `VG_STRESS_TAU` | VG error tolerance in pixels (default 1; `-` / `=` change it live) |

Headless cook (provision and cook every cache, no window or GPU, then exit):

```powershell
cargo run -p meshes-examples --features vg-stress --bin vg_stress --release -- --scene forest --prepare
```

`HELIO_VG_STRESS_SKIP_PROVISION=1` stops all downloads (build time and run
time); the scene then opens with what is on disk.

Scripted capture of the four fixed shots (`forest-ground`, `forest-valley`,
`forest-overview`, `forest-peaks`; 1920x1080, 60 degree field of view, 300
settle frames, post on):

```powershell
$env:VG_STRESS_SCENE = 'forest'
$env:VG_STRESS_CAPTURE = 'forest-out'
$env:VG_STRESS_SHOTS = 'vg_stress_shots/forest.json'
$env:HELIO_VG_PROFILE = '1'
cargo run -p meshes-examples --features vg-stress --bin vg_stress --release
```

Outputs and the other capture settings (`VG_STRESS_CAPTURE_ONLY`,
`VG_STRESS_CAPTURE_RES`, ...) are described under "Capture mode" in
[VG_STRESS.md](VG_STRESS.md).

## What the scene is

Built in memory at startup; nothing is written as a `.level`. Everything in
`vg_stress_data/forest.rs` is pure data (no GPU, no files) and seeded, so the
same seed gives the same mountains and the same trees.

### Terrain

- **Heightfield** (`Terrain::generate`): 3,072 m square centred on the
  origin, 3 m cells (1,025 x 1,025 vertices), 520 m relief. Domain-warped fBm
  for rolling ground and a ridged multifractal for crests, blended by a radial
  mountain mask into a valley ringed by peaks.
- **Slope limit**: thermal relaxation (`relax`) moves height across any step
  steeper than 0.9 x `MAX_SLOPE_DEG` (52 degrees) until no step is more than
  2 cm over. The bound keeps the planar UV stretch (`1 / cos(slope)`, at most
  about 1.6x) bounded; both are unit tested.
- **Tiles** (`tiles`): 64 x 64 cells, so 192 m, which is 32 periods of the
  6 m ground texture (`GROUND_UV_PERIOD_M` in `vg_stress_prep/catalog.rs`):
  16 x 16 tiles. Each tile is split by face slope into a forest-floor mesh and
  a rock mesh (faces steeper than `ROCK_SLOPE_DEG`, 34 degrees); empty meshes
  are dropped. Vertex positions are tile-local, normals come from the shared
  heightfield and UVs are metres over the period with the tile origin on a
  whole period, so tiles have no seams.
- **Drawing**: the tiles are classic meshes (one object row per tile and
  surface, spawned by `AppState::spawn_terrain` in `vg_stress.rs`), culled and
  drawn by the object batch like the museum's architecture. They are not VG.

### Trees

- **Scatter** (`scatter`): candidates on a jittered grid (8 m, shrunk by 0.8
  until enough usable candidates exist, not below 2 m), 24 m in from the
  terrain edge. Each candidate's weight combines a noisy treeline (altitude
  fraction 0.50 to 0.64), slope (full up to 22 degrees, none above 36),
  glades (a low-frequency mask) and stands (a finer one). The heaviest
  `count` by Efraimidis-Spirakis weighted-random key are kept, so density
  follows the weights and the count is exact.
- **Species by altitude** (`species_weight`): the island trees in the valley,
  `tree_small_02` in the valley and lower slopes, `pine_tree_01` in the mixed
  and high band, `fir_tree_01` highest. The variant is random per tree.
- **Per tree**: size from the species' `forest_scale` range (`catalog.rs`),
  skewed to middle age and reduced by up to 40% near the treeline; horizontal
  squash 0.94 to 1.06; yaw 0 to 360 degrees; tilt up to 2.5 degrees on both
  horizontal axes; sunk 0.15 m into the ground.
- **Tint stands**: one of `TINT_BUCKETS` (6) tint buckets per tree, picked by
  low-frequency noise so neighbours share it, with a 40% chance of a one-step
  shift. Foliage buckets shift hue and value; bark buckets only value.
- **Count**: `DEFAULT_TREES` (30,000) x density, rounded.
- **One VG instance per tree**: when the parts of a tree variant share one
  node transform, the loader merges them into one VG asset with one section
  per material slot (`merge_sections` in `vg_stress_loading.rs`); each
  placement then takes one consecutive object row per section. This applies
  to the museum's trees too. The builder simplifies each material on its own,
  so a merged root is the union of every material's root: the pine variants
  exceed the pinned root budget (`RootTooLarge`, 82 to 130 pages against 16)
  and are cooked as separate parts instead.

### Materials (`vg_stress_data/materials.rs`)

| Material | Use | Texture | Flat fallback |
|---|---|---|---|
| `Mat::Terrain(0)` | forest floor | `forrest_ground_01` | brown, roughness 0.95 |
| `Mat::Terrain(1)` | rock faces | `aerial_rocks_02` | grey, roughness 0.9 |
| `Mat::TreeTint(tree, slot, bucket)` | every tree slot in the forest | the tree glTF's | the slot's fallback colour x tint |

`Mat::all` adds the terrain materials and the 6 tinted copies of every slot of
the selected trees only for the forest scene. A tinted material multiplies the
base-colour texture by its bucket's factor (`texture_tint`).

### Sky, light and camera (`vg_stress.rs`)

- `scene_preset`: Helio's deferred preset over the meshes', materials' and
  lights' features; for the forest also the sky's (`SkyFeature`) and GI's
  (`GiFeature`). The museum adds neither.
- `spawn_forest_sky`: one `AtmosphereComponent` with
  `AtmosphereRow::earth()`.
- Sun: one directional light at azimuth 140 degrees, elevation 34 degrees,
  colour (1.0, 0.93, 0.80), intensity 3, spawned right after the sky; the sky
  takes the scene's first directional light as its sun.
- Camera: starts 14 m above the valley centre (`forest::vantage`); near plane
  0.3 m, far plane 6,000 m, free-camera speed 40 m/s (the museum: 0.1 m,
  250 m, 6 m/s). Capture mode uses the scene's near plane and a far plane of
  at least 500 m.

## Assets and licences

Provisioned by the same pure-Rust code as the museum (`vg_stress_prep/`). The
forest scene needs no Stanford scans and no marble: it fetches the two ground
sets and the selected trees.

| Asset | Use | Source | Licence |
|---|---|---|---|
| `pine_tree_01`, `fir_tree_01`, `island_tree_01`, `island_tree_02`, `island_tree_03`, `tree_small_02` | trees (`catalog::TREES`) | `https://polyhaven.com/a/<id>` | CC0 |
| `forrest_ground_01` (Forest Ground 01, Rob Tuytel) | forest floor | https://polyhaven.com/a/forrest_ground_01 | CC0 |
| `aerial_rocks_02` (Aerial Rocks 02, Rob Tuytel) | steep terrain and cliffs | https://polyhaven.com/a/aerial_rocks_02 | CC0 |

Tree sizes are in the Poly Haven table of [VG_STRESS.md](VG_STRESS.md) (about
1.5 GB for the full set). The ground sets (`catalog::GROUNDS`) have the
marble package shape (diffuse, nor_gl, arm), prefer 4k and fall back to 2k
(about 32 MB and 27 MB at 4k), and are stored beside the marbles
(`assets/vg_stress/marble/<id>/`). `assets/vg_stress/CREDITS_POLYHAVEN.md` is
regenerated whenever something is downloaded. Everything downloaded is
git-ignored.

Status (2026-11-10): the two ground sets have not been downloaded on the
development machine yet (waiting on the user's go-ahead), so the terrain in
current captures renders with the flat fallback colours above.

## Tests

```powershell
cargo test --release -p meshes-examples --features vg-stress --bin vg_stress forest
```

Runs the tests in `forest.rs` on a 768 m crop of the terrain: determinism and
relief, the slope limit, the UV stretch bound, tiles covering every cell once
and facing up, tile origins on whole texture periods, the rock/floor split,
an exact and deterministic scatter below the treeline and slope limit,
variation in species, scale, tint and yaw, no trees without a tree set, tinted
placements sharing at most 16 assets, and the terrain materials resolving to
the ground sets. `every_section_row_of_a_visible_instance_leaves_the_classic_path`
(`vg_stress.rs`) covers sectioned instances.

The camera positions in `vg_stress_shots/forest.json` were placed with an
ignored helper that prints the terrain range, valley centre, highest tree,
species counts, the densest 100 m cell and sample heights and slopes. Rerun it
after changing the terrain or the scatter and update the shot list:

```powershell
cargo test --release -p meshes-examples --features vg-stress --bin vg_stress print_vantage_points -- --ignored --nocapture
```

## Performance

### Method

- Capture mode with `vg_stress_shots/forest.json`, density 1, tau 1, the same
  cooked caches and unchanged materials for every compared run.
- The capture host keeps up to three frames in flight: it records
  `Renderer::last_submission()` each frame and waits only for the oldest once
  three are queued, instead of draining the GPU after every frame. The final
  image readback still waits for completion.
- `gpu_ms` in each shot's stats JSON is the mean of the distinct asynchronous
  timestamp samples over the last 30 settle iterations, per pass and `TOTAL`.
- Run one GPU process at a time and compare A/B runs interleaved.
- `vg.overflow.visible` above 0 means clusters were dropped (bin ceiling): the
  frame is not coverage-complete and is not a quality reference.

### Results

To be filled with measured numbers (RTX Spark N1X, Vulkan, 1920x1080, tau 1,
density 1, pipelined capture).

| Shot | GPU total (ms) | VG cull early (ms) | VG raster (ms) | Selected triangles | Clusters tested | Jobs | Visible overflow |
|---|---|---|---|---|---|---|---|
| `forest-ground` | | | | | | | |
| `forest-valley` | | | | | | | |
| `forest-overview` | | | | | | | |
| `forest-peaks` | | | | | | | |

### Investigation log

RTX Spark N1X, Vulkan, 1920x1080, tau 1, `forest-ground` shot unless noted.
The 177 ms run was a serialized capture (the host drained the GPU every
frame); the log does not record which capture host the other runs used.

| Run | GPU total | Notes |
|---|---|---|
| Initial | 430 ms | `VgCullEarly` 383 ms; 58 M clusters tested, 37 M jobs, 160 M triangles selected, 558k visible-bin overflow |
| Density 0.1 (3,000 trees) | 108 ms | |
| Tau 4 | 123 ms | |
| Tau 16 | 43 ms | `forest-overview` shot |
| `tile_scale_min` node gating | 177 ms | 5 M jobs; early cull 116 ms, early raster 35 ms, deferred lighting 16 ms, GI 4.5 ms; 8.9 M clusters tested, 570k visible overflow |
| Valve at 32x scale, 8 triangles per pixel | 79 ms | cull 52 ms, raster 3.9 ms, 30 M triangles selected, no overflow. Now the defaults (`TILE_SCALE_MAX` 32, `tile_density_limit` 8) |
| Rounds 8, redistribute every 4 waves | 37 ms | cull 7.7 ms (47 ms with 4 rounds / 8 waves, which also dropped 59 to 970 jobs): the cost was load imbalance in the last round |
| Rounds 16, redistribute every 8 waves | 35 ms | cull 6.0 ms ground, 10.2 ms overview; deferred lighting 15.5 ms and GI 4.4 ms were now the largest passes |
| Lighting fix (sky table read through its binding) | | deferred lighting 16.6 to 0.4 ms, GI 4.2 to 0.2 ms, same frame (see Helio's layered-rewrite "No per-invocation table copies") |

Not yet re-measured: the persistent single-dispatch traversal (which replaces
the rounds) together with sectioned trees and the read-only snapshot and per-wave
emission. One interleaved run of sections plus the snapshot and emission changes,
before the persistent traversal, selected 175 M triangles with visible-bin
overflow instead of 30 M; whether the sections or the snapshot cause that is
open and must be settled by the next capture.

Where the cost was: the early cull's hierarchy walk (383 of 430 ms), with
160 M triangles selected for about 2 M pixels. The per-tile overdraw valve
(`tile_density_limit`, see "Triangle selection and tau" in VG_STRESS.md)
applied only to the group and cluster decisions, so the hierarchy nodes above
saturated tiles were still walked.

What the cull now does (`feature/vg-cull/shaders/vg_cull.wgsl` and
`vg/stream/shaders/vg_stream_feedback.wgsl`):

- **Node gating by the valve.** A hierarchy node stops the descent when its
  projected error is under `tau * TAU_HOLD * tile_scale_min`, the smallest
  valve scale over the tiles its sphere covers. Every cluster below lies
  inside the node's sphere, so its own scale is never smaller and the cut
  stays coverage-complete; a distant tree on one saturated tile is skipped
  whole.
- **No redundant atomics on monotone bits.** The page request, used and
  urgent bitsets (`helio-vg-stream`) and the cull's history and hysteresis
  marks only accumulate between ordered resets, so each helper reads the bit
  and skips the `atomicOr` when it is already set. Racing first requesters
  still use `atomicOr`'s old value, so exactly one appends the page to the
  request list.
- **Refinement below the hold threshold** (`group_refines`) returns coarse
  with no hash or history lookup; only decisions inside the hysteresis band
  read history.
- **Per-thread counters**: stats are summed per thread and added to the shared
  counters once when the thread leaves the traversal (`stats_flush`).
- **Cluster jobs**: a group's clusters are pushed onto the workgroup stack as
  separate jobs when the whole group fits (compare-exchange reservation), so
  the next wave tests them in parallel; a group that does not fit is tested
  inline as before.

Buffer contracts, page residency and eviction, the render graph and the
geometric error tolerance are unchanged. The GPU regression
`contended_feedback_preserves_requests_touches_and_urgency_after_reset`
(`vg/stream/tests/gpu.rs`) checks one request-list append per missing page
under contention, urgent upgrades and a second reset epoch. Per-change
timings of these steps are not in the log above.

AAA comparison: Nanite
([SIGGRAPH 2021](https://advances.realtimerendering.com/s2021/Karis_Nanite_SIGGRAPH_Advances_2021_final.pdf))
uses the same GPU hierarchy traversal with persistent threads; the changes
above remove synchronisation inside it without changing its geometric
decisions. Epic's
[content guidance](https://dev.epicgames.com/documentation/unreal-engine/working-with-naniteenabled-content)
names aggregate foliage and overdraw as hard cases for simplification; the
bookkeeping changes do not solve that representation cost (see Known limits).

## Known limits

- **No terrain blending.** The material system has no splat or triplanar
  blending, so the rock/floor boundary is a hard edge along triangle slope and
  the 6 m texture period repeats visibly.
- **Terrain is classic.** Tiled meshes through the object batch, not VG and
  not a clipmap or virtual-texture terrain.
- **No aggregate foliage tier.** Distant trees are still their own cluster
  hierarchies down to the roots; there is no impostor or aggregate
  representation (the Nanite Foliage equivalent is missing), which is why the
  valve and tau dominate the far-field cost.
- **No CLAS/PTLAS.** Cluster and partitioned TLAS acceleration structures
  ([DirectX preview](https://devblogs.microsoft.com/directx/clas-ptlas-preview/))
  are not available through wgpu yet.
- **Slow first DX12 start.** There is no persistent pipeline cache, so the
  first DX12 run compiles every pipeline.
