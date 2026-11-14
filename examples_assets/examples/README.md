# Meshes examples

The Meshes plugin imports source models (FBX, glTF/GLB, OBJ, USD) as native
`.mesh` assets; it registers no editable file type of its own. These small
source models show what the importer reads. Drop one into a project (or drag
it into the asset browser) to link or convert it, then place the resulting
`.mesh` in a level.

| File | Format | Shows |
|---|---|---|
| `models/cube.obj` | OBJ (text) | a unit cube with per-face normals and UVs, in one object |
| `models/pyramid.obj` + `models/pyramid.mtl` | OBJ and its material library | two materials (`usemtl`), which import as two material slots; keep the `.mtl` beside the `.obj` |
| `models/floor_tile.gltf` | glTF 2.0 (JSON) | a quad with positions, normals, UVs, indices and a PBR material, its buffer embedded as a data URI so it is one file |

Units are metres, Y up. The OBJ files carry their explanation as `#` comments.
A rigged, animated model imports as a skeleton, a skeletal mesh and animation
sequences instead of a `.mesh`; see the Animation plugin's examples for those
assets.
