//! The mountain forest scene of `vg_stress` (`--scene forest`): a procedural
//! heightfield, its terrain tiles, and the deterministic scatter of the trees.
//!
//! Everything here is pure data (no GPU, no files), so it is unit tested:
//! the same seed gives the same mountains and the same trees.
//!
//! * **Terrain.** Domain-warped fBm and a ridged multifractal, blended by a
//!   mountain mask into a valley ringed by peaks, then thermal relaxation so
//!   no slope exceeds [`MAX_SLOPE_DEG`]. That bound is what keeps the
//!   planar UV stretch (`1 / cos(slope)`) bounded (tested), which is what
//!   "consistent texel density" means for a heightfield without a
//!   parametrization pass.
//! * **Tiles.** The heightfield is cut into square tiles of whole texture
//!   periods; each tile is two meshes (forest floor, rock) split by face
//!   slope, so tiles are culled and draw as ordinary classic meshes. Vertex
//!   positions are tile-local (f32 precision at any world size), normals come
//!   from the shared heightfield (no seam between tiles) and UVs are the
//!   metre position over the texture period, with the tile origin on a whole
//!   period (no seam either).
//! * **Scatter.** Candidates on a jittered grid are weighted by altitude
//!   (treeline), slope, clearings and the meadow around the valley centre; the
//!   N heaviest by weighted-random key (Efraimidis-Spirakis) are kept, so the
//!   density follows the weights and the count is exact. Species follow
//!   altitude (broadleaf in the valley, conifers above), scale, yaw, tilt and
//!   tint bucket vary per tree, tint in stands.

use super::layout::{Placement, Rng};
use super::materials::{Mat, TINT_BUCKETS};
use crate::vg_stress_prep::catalog::{self, Config, GROUND_UV_PERIOD_M};
use helio_mesh_api::{MeshUpload, PackedVertex};

/// No terrain face is steeper than this after relaxation.
pub const MAX_SLOPE_DEG: f32 = 52.0;
/// Faces steeper than this take the rock material.
pub const ROCK_SLOPE_DEG: f32 = 34.0;
/// Trees grow on ground up to this steep, thinning out from [`TREE_FULL_SLOPE_DEG`].
pub const TREE_MAX_SLOPE_DEG: f32 = 36.0;
pub const TREE_FULL_SLOPE_DEG: f32 = 22.0;
/// Number of trees of the default (density 1) forest.
pub const DEFAULT_TREES: usize = 30_000;

#[derive(Clone, Copy, Debug)]
pub struct TerrainParams {
    pub seed: u64,
    /// Side of the square terrain in metres, centred on the origin.
    pub size_m: f32,
    /// Vertices per metre-step: cell size in metres.
    pub cell_m: f32,
    /// Height of the highest peaks above the valley floor.
    pub relief_m: f32,
}

impl Default for TerrainParams {
    fn default() -> Self {
        Self {
            seed: 0x70_7265_7374,
            size_m: 3072.0,
            cell_m: 3.0,
            relief_m: 520.0,
        }
    }
}

// ── noise ───────────────────────────────────────────────────────────────────

fn hash2(seed: u64, x: i32, y: i32) -> f32 {
    let mut z = seed
        ^ (x as u32 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (y as u32 as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((z ^ (z >> 31)) >> 40) as f32 / (1u64 << 24) as f32
}

/// Smooth value noise in [0, 1].
fn noise(seed: u64, x: f32, y: f32) -> f32 {
    let (xi, yi) = (x.floor(), y.floor());
    let (fx, fy) = (x - xi, y - yi);
    let (sx, sy) = (fx * fx * (3.0 - 2.0 * fx), fy * fy * (3.0 - 2.0 * fy));
    let (xi, yi) = (xi as i32, yi as i32);
    let a = hash2(seed, xi, yi);
    let b = hash2(seed, xi + 1, yi);
    let c = hash2(seed, xi, yi + 1);
    let d = hash2(seed, xi + 1, yi + 1);
    let top = a + (b - a) * sx;
    let bottom = c + (d - c) * sx;
    top + (bottom - top) * sy
}

fn fbm(seed: u64, mut x: f32, mut y: f32, octaves: u32) -> f32 {
    let (mut sum, mut amp, mut norm) = (0.0, 0.5, 0.0);
    for o in 0..octaves {
        sum += amp * noise(seed.wrapping_add(o as u64 * 7919), x, y);
        norm += amp;
        amp *= 0.5;
        x = x * 2.03 + 17.3;
        y = y * 2.03 - 9.1;
    }
    sum / norm
}

/// Ridged multifractal in [0, 1]: sharp crests, smooth valleys.
fn ridged(seed: u64, mut x: f32, mut y: f32, octaves: u32) -> f32 {
    let (mut sum, mut amp, mut norm, mut weight) = (0.0, 0.5, 0.0, 1.0);
    for o in 0..octaves {
        let n = 1.0 - (noise(seed.wrapping_add(o as u64 * 104_729), x, y) * 2.0 - 1.0).abs();
        let n = n * n * weight;
        weight = (n * 2.0).clamp(0.0, 1.0);
        sum += amp * n;
        norm += amp;
        amp *= 0.5;
        x = x * 2.1 + 5.7;
        y = y * 2.1 + 3.3;
    }
    sum / norm
}

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// ── the heightfield ─────────────────────────────────────────────────────────

pub struct Terrain {
    pub params: TerrainParams,
    /// Vertices per side.
    pub n: usize,
    /// Heights in metres, row-major `[z][x]`, x then z increasing.
    pub heights: Vec<f32>,
}

impl Terrain {
    pub fn generate(params: TerrainParams) -> Self {
        let cells = (params.size_m / params.cell_m).round() as usize;
        let n = cells + 1;
        let half = params.size_m * 0.5;
        let mut heights = vec![0.0f32; n * n];
        let s = params.seed;
        for j in 0..n {
            for i in 0..n {
                let x = i as f32 * params.cell_m - half;
                let z = j as f32 * params.cell_m - half;
                let (u, v) = (x / params.size_m, z / params.size_m);
                // Domain warp: the ranges do not run on a grid.
                // Feature sizes are in metres, not fractions of the terrain, so a
                // small terrain is a crop of a big one (and a test of it).
                let wx = (fbm(s ^ 1, x / 1024.0, z / 1024.0, 4) - 0.5) * 0.9;
                let wz = (fbm(s ^ 2, x / 1024.0 + 7.0, z / 1024.0 - 3.0, 4) - 0.5) * 0.9;
                let (px, pz) = (x / 768.0 + wx, z / 768.0 + wz);
                let rolling = fbm(s ^ 3, px * 1.5, pz * 1.5, 5);
                let crests = ridged(s ^ 4, px * 0.55, pz * 0.55, 5);
                // A valley around the middle, mountains to every side.
                let r = (u * u + v * v).sqrt() * 2.0;
                let mountain = smoothstep(0.22, 0.85, r + (fbm(s ^ 5, x / 614.4, z / 614.4, 3) - 0.5) * 0.35);
                let rim = smoothstep(0.85, 1.05, r);
                let h = rolling * 0.16 + mountain * (0.10 + crests * 0.70) + rim * 0.25;
                heights[j * n + i] = h * params.relief_m;
            }
        }
        let mut terrain = Terrain { params, n, heights };
        terrain.relax(MAX_SLOPE_DEG * 0.9, 1200);
        terrain
    }

    /// Thermal relaxation: material slides off any step steeper than
    /// `max_slope_deg` between neighbouring vertices (4- and 8-connected),
    /// sweeping until no step is more than 2 cm over the limit (or
    /// `max_passes`).
    fn relax(&mut self, max_slope_deg: f32, max_passes: usize) {
        let cell = self.params.cell_m;
        let talus = max_slope_deg.to_radians().tan();
        let n = self.n;
        for _ in 0..max_passes {
            let mut worst = 0.0f32;
            for j in 0..n {
                for i in 0..n {
                    for (di, dj, len) in [
                        (1i32, 0i32, 1.0f32),
                        (0, 1, 1.0),
                        (1, 1, std::f32::consts::SQRT_2),
                        (1, -1, std::f32::consts::SQRT_2),
                    ] {
                        let (ni, nj) = (i as i32 + di, j as i32 + dj);
                        if ni < 0 || nj < 0 || ni >= n as i32 || nj >= n as i32 {
                            continue;
                        }
                        let k = nj as usize * n + ni as usize;
                        let diff = self.heights[j * n + i] - self.heights[k];
                        let over = diff.abs() - talus * cell * len;
                        if over > 0.0 {
                            worst = worst.max(over);
                            // Split the excess: the high side drops, the low side rises.
                            let shift = over * 0.5 * diff.signum();
                            self.heights[j * n + i] -= shift;
                            self.heights[k] += shift;
                        }
                    }
                }
            }
            if worst < 0.02 {
                break;
            }
        }
    }

    pub fn half(&self) -> f32 {
        self.params.size_m * 0.5
    }

    pub fn height_at(&self, x: f32, z: f32) -> f32 {
        let c = self.params.cell_m;
        let fx = ((x + self.half()) / c).clamp(0.0, (self.n - 1) as f32 - 1e-3);
        let fz = ((z + self.half()) / c).clamp(0.0, (self.n - 1) as f32 - 1e-3);
        let (i, j) = (fx as usize, fz as usize);
        let (tx, tz) = (fx - i as f32, fz - j as f32);
        let h = |i: usize, j: usize| self.heights[j * self.n + i];
        let a = h(i, j) + (h(i + 1, j) - h(i, j)) * tx;
        let b = h(i, j + 1) + (h(i + 1, j + 1) - h(i, j + 1)) * tx;
        a + (b - a) * tz
    }

    fn vertex_height(&self, i: i64, j: i64) -> f32 {
        let c = |v: i64| v.clamp(0, self.n as i64 - 1) as usize;
        self.heights[c(j) * self.n + c(i)]
    }

    /// Unit surface normal at a vertex (central differences).
    pub fn vertex_normal(&self, i: i64, j: i64) -> glam::Vec3 {
        let c = self.params.cell_m;
        let hx = (self.vertex_height(i + 1, j) - self.vertex_height(i - 1, j)) / (2.0 * c);
        let hz = (self.vertex_height(i, j + 1) - self.vertex_height(i, j - 1)) / (2.0 * c);
        glam::Vec3::new(-hx, 1.0, -hz).normalize()
    }

    /// Slope in degrees at a world position.
    pub fn slope_deg_at(&self, x: f32, z: f32) -> f32 {
        let c = self.params.cell_m;
        let hx = (self.height_at(x + c, z) - self.height_at(x - c, z)) / (2.0 * c);
        let hz = (self.height_at(x, z + c) - self.height_at(x, z - c)) / (2.0 * c);
        (hx * hx + hz * hz).sqrt().atan().to_degrees()
    }

    pub fn max_height(&self) -> f32 {
        self.heights.iter().copied().fold(f32::MIN, f32::max)
    }

    pub fn min_height(&self) -> f32 {
        self.heights.iter().copied().fold(f32::MAX, f32::min)
    }
}

// ── tiles ───────────────────────────────────────────────────────────────────

/// Cells per tile side. With 3 m cells a tile is 192 m = 32 texture periods.
pub const TILE_CELLS: usize = 64;

/// One terrain mesh: a material and its geometry, placed at `origin`.
pub struct TileMesh {
    pub origin: [f32; 3],
    pub radius: f32,
    /// 0 forest floor, 1 rock (`Mat::Terrain`).
    pub surface: u8,
    pub mesh: MeshUpload,
}

/// Cuts `terrain` into tiles of [`TILE_CELLS`] cells, each into a forest-floor
/// and a rock mesh (empty ones are dropped).
pub fn tiles(terrain: &Terrain) -> Vec<TileMesh> {
    let cell = terrain.params.cell_m;
    let tile_m = TILE_CELLS as f32 * cell;
    // The UV origin of a tile must sit on a whole period or tiles would seam.
    debug_assert!(
        ((tile_m / GROUND_UV_PERIOD_M) - (tile_m / GROUND_UV_PERIOD_M).round()).abs() < 1e-4,
        "tile size must be a whole number of texture periods"
    );
    let per_side = (terrain.n - 1) / TILE_CELLS;
    let half = terrain.half();
    let mut out = Vec::new();
    for tj in 0..per_side {
        for ti in 0..per_side {
            let (i0, j0) = (ti * TILE_CELLS, tj * TILE_CELLS);
            let ox = i0 as f32 * cell - half;
            let oz = j0 as f32 * cell - half;
            let mut lo = f32::MAX;
            let mut hi = f32::MIN;
            for j in 0..=TILE_CELLS {
                for i in 0..=TILE_CELLS {
                    let h = terrain.heights[(j0 + j) * terrain.n + i0 + i];
                    lo = lo.min(h);
                    hi = hi.max(h);
                }
            }
            let mid = (lo + hi) * 0.5;
            let origin = [ox, mid, oz];
            for surface in [0u8, 1u8] {
                if let Some(mesh) = tile_mesh(terrain, i0, j0, origin, surface) {
                    let ext = tile_m * std::f32::consts::FRAC_1_SQRT_2;
                    let radius = (ext * ext + ((hi - lo) * 0.5).powi(2)).sqrt() + cell;
                    out.push(TileMesh {
                        // Object centre: the tile centre; vertices are relative to it.
                        origin: [ox + tile_m * 0.5, mid, oz + tile_m * 0.5],
                        radius,
                        surface,
                        mesh,
                    });
                }
            }
        }
    }
    out
}

fn tile_mesh(
    terrain: &Terrain,
    i0: usize,
    j0: usize,
    origin: [f32; 3],
    surface: u8,
) -> Option<MeshUpload> {
    let cell = terrain.params.cell_m;
    let tile_m = TILE_CELLS as f32 * cell;
    let centre = [origin[0] + tile_m * 0.5, origin[1], origin[2] + tile_m * 0.5];
    let rock_cos = ROCK_SLOPE_DEG.to_radians().cos();
    let mut remap: std::collections::HashMap<(usize, usize), u32> = Default::default();
    let mut vertices: Vec<PackedVertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut vertex = |i: usize, j: usize, vertices: &mut Vec<PackedVertex>| -> u32 {
        *remap.entry((i, j)).or_insert_with(|| {
            let (gi, gj) = (i0 + i, j0 + j);
            let local_x = i as f32 * cell - tile_m * 0.5;
            let local_z = j as f32 * cell - tile_m * 0.5;
            let h = terrain.heights[gj * terrain.n + gi];
            let normal = terrain.vertex_normal(gi as i64, gj as i64);
            // dP/du along +x on the surface, projected into the tangent plane.
            let tangent = (glam::Vec3::X - normal * normal.x).normalize();
            // u = x / period and v = z / period; the tile origin is on a whole
            // period, so tile-local metres give seamless UVs.
            let uv = [
                (i as f32 * cell) / GROUND_UV_PERIOD_M,
                (j as f32 * cell) / GROUND_UV_PERIOD_M,
            ];
            vertices.push(PackedVertex::from_components(
                [local_x, h - centre[1], local_z],
                normal.to_array(),
                uv,
                tangent.to_array(),
                1.0,
            ));
            vertices.len() as u32 - 1
        })
    };
    for j in 0..TILE_CELLS {
        for i in 0..TILE_CELLS {
            let h = |di: usize, dj: usize| terrain.heights[(j0 + j + dj) * terrain.n + i0 + i + di];
            let (a, b, c, d) = (h(0, 0), h(1, 0), h(0, 1), h(1, 1));
            // Two triangles (a, d, b) and (a, c, d): counter-clockwise from above.
            let tris = [
                ([(0, 0, a), (1, 1, d), (1, 0, b)]),
                ([(0, 0, a), (0, 1, c), (1, 1, d)]),
            ];
            for tri in tris {
                let p = |(di, dj, hh): (usize, usize, f32)| {
                    glam::Vec3::new(di as f32 * cell, hh, dj as f32 * cell)
                };
                let n = (p(tri[1]) - p(tri[0])).cross(p(tri[2]) - p(tri[0])).normalize();
                let is_rock = n.y < rock_cos;
                if is_rock != (surface == 1) {
                    continue;
                }
                for (di, dj, _) in tri {
                    let index = vertex(i + di, j + dj, &mut vertices);
                    indices.push(index);
                }
            }
        }
    }
    (!indices.is_empty()).then_some(MeshUpload { vertices, indices })
}

// ── trees ───────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct TreeSpot {
    pub position: [f32; 3],
    pub yaw_deg: f32,
    pub tilt_deg: [f32; 2],
    pub scale: [f32; 3],
    /// Index into `catalog::TREES`.
    pub species: usize,
    pub variant: usize,
    pub tint: u8,
}

/// Altitude as a fraction of the terrain's relief above its lowest point.
fn altitude_fraction(terrain: &Terrain, y: f32) -> f32 {
    (y - terrain.min_height()) / (terrain.max_height() - terrain.min_height()).max(1.0)
}

/// Treeline and conditions: the relative density of trees at `(x, z)`.
fn tree_weight(terrain: &Terrain, seed: u64, x: f32, z: f32) -> f32 {
    let h = terrain.height_at(x, z);
    let alt = altitude_fraction(terrain, h);
    let slope = terrain.slope_deg_at(x, z);
    let treeline = 1.0 - smoothstep(0.50, 0.64, alt + (fbm(seed ^ 11, x * 0.004, z * 0.004, 3) - 0.5) * 0.12);
    let slope_w = 1.0 - smoothstep(TREE_FULL_SLOPE_DEG, TREE_MAX_SLOPE_DEG, slope);
    // Glades: a low-frequency mask removes whole patches.
    let glade = smoothstep(0.30, 0.46, fbm(seed ^ 12, x * 0.006, z * 0.006, 4));
    // Stands: a second, finer mask makes clumps and gaps inside the forest.
    let stand = 0.35 + 0.65 * smoothstep(0.25, 0.7, fbm(seed ^ 13, x * 0.02, z * 0.02, 3));
    treeline * slope_w * glade * stand
}

/// Species weights at altitude fraction `alt` for the catalogue's trees:
/// broadleaf in the valley, conifers above, fir at the top.
fn species_weight(id: &str, alt: f32) -> f32 {
    let broad = 1.0 - smoothstep(0.12, 0.34, alt);
    let mixed = smoothstep(0.08, 0.28, alt) * (1.0 - smoothstep(0.40, 0.58, alt));
    let high = smoothstep(0.30, 0.52, alt);
    match id {
        "island_tree_01" | "island_tree_02" | "island_tree_03" => broad * 0.8,
        "tree_small_02" => broad * 0.5 + mixed * 0.15,
        "pine_tree_01" => mixed * 0.7 + high * 0.55 + broad * 0.05,
        "fir_tree_01" => mixed * 0.35 + high * 0.9,
        _ => 0.1,
    }
}

pub struct Scatter {
    pub spots: Vec<TreeSpot>,
    /// Candidate-grid cell size used, metres.
    pub cell_m: f32,
}

/// Places up to `count` trees on `terrain` for the trees `config` provides.
pub fn scatter(terrain: &Terrain, config: &Config, seed: u64, count: usize) -> Scatter {
    let trees: Vec<usize> = catalog::TREES
        .iter()
        .enumerate()
        .filter(|(_, t)| config.has_tree(t.id))
        .map(|(i, _)| i)
        .collect();
    if trees.is_empty() || count == 0 {
        return Scatter { spots: Vec::new(), cell_m: 0.0 };
    }
    let half = terrain.half() - 24.0;
    // Candidates on a jittered grid, fine enough that `count` heavy ones exist.
    let mut cell = 8.0f32;
    loop {
        let candidates = candidates(terrain, seed, cell, half);
        let usable = candidates.iter().filter(|c| c.weight > 0.02).count();
        if usable >= count || cell < 2.0 {
            return pick(terrain, config, &trees, seed, count, candidates, cell);
        }
        cell *= 0.8;
    }
}

struct Candidate {
    x: f32,
    z: f32,
    weight: f32,
}

fn candidates(terrain: &Terrain, seed: u64, cell: f32, half: f32) -> Vec<Candidate> {
    let per_side = (2.0 * half / cell) as i32;
    let mut out = Vec::with_capacity((per_side * per_side) as usize / 2);
    for j in 0..per_side {
        for i in 0..per_side {
            let h = |k: u64| hash2(seed ^ k, i, j);
            // Jitter within +-0.35 cell: neighbours stay at least 0.3 cell apart.
            let x = -half + (i as f32 + 0.5 + (h(21) - 0.5) * 0.7) * cell;
            let z = -half + (j as f32 + 0.5 + (h(22) - 0.5) * 0.7) * cell;
            let weight = tree_weight(terrain, seed, x, z);
            if weight > 0.0 {
                out.push(Candidate { x, z, weight });
            }
        }
    }
    out
}

fn pick(
    terrain: &Terrain,
    _config: &Config,
    trees: &[usize],
    seed: u64,
    count: usize,
    mut candidates: Vec<Candidate>,
    cell: f32,
) -> Scatter {
    let mut rng = Rng::new(seed ^ 0x5EED);
    // Efraimidis-Spirakis keys: u^(1/w); the largest `count` win.
    let mut keyed: Vec<(f32, usize)> = candidates
        .iter()
        .enumerate()
        .map(|(i, c)| (rng.unit().max(1e-6).powf(1.0 / c.weight.max(1e-4)), i))
        .collect();
    keyed.sort_by(|a, b| b.0.total_cmp(&a.0));
    keyed.truncate(count);
    // Stable order (by position) so placements do not depend on the sort.
    keyed.sort_by_key(|k| k.1);
    let mut spots = Vec::with_capacity(keyed.len());
    for (_, i) in keyed {
        let c = &mut candidates[i];
        let y = terrain.height_at(c.x, c.z);
        let alt = altitude_fraction(terrain, y);
        let weights: Vec<f32> = trees
            .iter()
            .map(|&t| species_weight(catalog::TREES[t].id, alt))
            .collect();
        let total: f32 = weights.iter().sum();
        let species = if total <= 1e-4 {
            trees[rng.below(trees.len())]
        } else {
            let mut pickv = rng.unit() * total;
            let mut chosen = trees[0];
            for (&t, w) in trees.iter().zip(&weights) {
                if pickv <= *w {
                    chosen = t;
                    break;
                }
                pickv -= w;
            }
            chosen
        };
        let def = &catalog::TREES[species];
        let variant = rng.below(def.variants.len());
        // Age: most trees middle-aged, a few giants; smaller near the treeline.
        let age = rng.unit() * rng.unit().max(0.35);
        let (lo, hi) = def.forest_scale;
        let size = (lo + (hi - lo) * (0.35 + 0.65 * age)) * (1.0 - 0.4 * smoothstep(0.42, 0.62, alt));
        let squash = rng.range(0.94, 1.06);
        // Tint in stands (low-frequency noise picks the bucket), with some per-tree scatter.
        let stand = fbm(seed ^ 31, c.x * 0.01, c.z * 0.01, 2);
        let mut tint = (stand * TINT_BUCKETS as f32) as i32;
        if rng.unit() < 0.4 {
            tint += rng.below(3) as i32 - 1;
        }
        let tint = tint.rem_euclid(TINT_BUCKETS as i32) as u8;
        spots.push(TreeSpot {
            position: [c.x, y - 0.15, c.z],
            yaw_deg: rng.range(0.0, 360.0),
            tilt_deg: [rng.range(-2.5, 2.5), rng.range(-2.5, 2.5)],
            scale: [size * squash, size, size * squash],
            species,
            variant,
            tint,
        });
    }
    Scatter { spots, cell_m: cell }
}

// ── layout ──────────────────────────────────────────────────────────────────

/// The forest's tree placements: one tinted `Placement::tree` per spot.
pub fn placements(spots: &[TreeSpot]) -> Vec<Placement> {
    spots
        .iter()
        .enumerate()
        .map(|(n, s)| {
            let def = &catalog::TREES[s.species];
            let mut p = Placement::tree(
                format!("Tree {n}"),
                def,
                s.variant,
                s.position,
                s.yaw_deg,
                1.0,
            );
            p.scale = s.scale;
            p.rotation = [s.tilt_deg[0], s.yaw_deg, s.tilt_deg[1]];
            for slot in &mut p.slots {
                if let Some(m) = Mat::tree_slot_tinted(def.id, &slot.0, s.tint) {
                    slot.1 = m;
                }
            }
            p
        })
        .collect()
}

/// Everything the scene builder needs from the forest layout.
pub struct Forest {
    pub terrain: Terrain,
    pub spots: Vec<TreeSpot>,
    pub placements: Vec<Placement>,
}

pub fn build(config: &Config) -> Forest {
    let terrain = Terrain::generate(TerrainParams::default());
    let count = ((DEFAULT_TREES as f32) * config.density).round() as usize;
    let scatter = scatter(&terrain, config, TerrainParams::default().seed, count);
    let placements = placements(&scatter.spots);
    Forest {
        terrain,
        spots: scatter.spots,
        placements,
    }
}

/// A walkable valley position with trees around it and the way to look: the
/// default camera of the scene. Returns `(position, yaw, pitch)` radians for
/// the museum camera convention (`forward = (sin yaw * cos pitch, sin pitch, -cos yaw * cos pitch)`).
pub fn vantage(terrain: &Terrain) -> glam::Vec3 {
    let h = terrain.height_at(0.0, 0.0);
    glam::Vec3::new(0.0, h + 14.0, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small() -> Terrain {
        Terrain::generate(TerrainParams {
            size_m: 768.0,
            ..TerrainParams::default()
        })
    }

    #[test]
    fn terrain_is_deterministic_and_has_relief() {
        let (a, b) = (small(), small());
        assert_eq!(a.heights, b.heights);
        assert!(a.max_height() - a.min_height() > 120.0, "flat terrain");
    }

    #[test]
    fn no_edge_is_steeper_than_the_slope_limit() {
        let t = small();
        let c = t.params.cell_m;
        let limit = MAX_SLOPE_DEG.to_radians().tan() * c * 1.05;
        for j in 0..t.n - 1 {
            for i in 0..t.n - 1 {
                let h = t.heights[j * t.n + i];
                for other in [t.heights[j * t.n + i + 1], t.heights[(j + 1) * t.n + i]] {
                    assert!((h - other).abs() <= limit, "step {} > {limit}", (h - other).abs());
                }
            }
        }
    }

    #[test]
    fn planar_uv_stretch_is_bounded_by_the_slope_limit() {
        // Texel density along the steepest direction is 1 / cos(slope) of the
        // flat one: bounded because slopes are.
        let t = small();
        let worst = 1.0 / MAX_SLOPE_DEG.to_radians().cos();
        let c = t.params.cell_m;
        let mut max_stretch = 1.0f32;
        for j in 0..t.n - 1 {
            for i in 0..t.n - 1 {
                let dx = t.heights[j * t.n + i + 1] - t.heights[j * t.n + i];
                let dz = t.heights[(j + 1) * t.n + i] - t.heights[j * t.n + i];
                max_stretch = max_stretch
                    .max(((c * c + dx * dx).sqrt() / c).max((c * c + dz * dz).sqrt() / c));
            }
        }
        assert!(max_stretch <= worst * 1.1, "stretch {max_stretch} > {worst}");
    }

    #[test]
    fn tiles_cover_every_cell_once_and_face_up() {
        let t = small();
        let tiles = tiles(&t);
        let triangles: usize = tiles.iter().map(|m| m.mesh.indices.len() / 3).sum();
        assert_eq!(triangles, (t.n - 1) * (t.n - 1) * 2);
        for tile in &tiles {
            for tri in tile.mesh.indices.chunks(3) {
                let p = |i: u32| glam::Vec3::from_array(tile.mesh.vertices[i as usize].position);
                let n = (p(tri[1]) - p(tri[0])).cross(p(tri[2]) - p(tri[0]));
                assert!(n.y > 0.0, "a terrain triangle faces down");
            }
        }
    }

    #[test]
    fn tile_origins_sit_on_whole_texture_periods() {
        let tile_m = TILE_CELLS as f32 * TerrainParams::default().cell_m;
        assert_eq!((tile_m / GROUND_UV_PERIOD_M).fract(), 0.0);
    }

    #[test]
    fn rock_takes_steep_faces_and_floor_the_rest() {
        let t = small();
        let rock_cos = ROCK_SLOPE_DEG.to_radians().cos();
        for tile in tiles(&t) {
            for tri in tile.mesh.indices.chunks(3) {
                let p = |i: u32| glam::Vec3::from_array(tile.mesh.vertices[i as usize].position);
                let n = (p(tri[1]) - p(tri[0])).cross(p(tri[2]) - p(tri[0])).normalize();
                assert_eq!(n.y < rock_cos, tile.surface == 1);
            }
        }
    }

    fn all_trees() -> Config {
        Config::resolve(Some("full"), None, None, None, Some("forest"))
    }

    #[test]
    fn scatter_is_exact_deterministic_and_respects_the_ground() {
        let t = small();
        let config = all_trees();
        let a = scatter(&t, &config, 7, 800);
        let b = scatter(&t, &config, 7, 800);
        assert_eq!(a.spots.len(), 800);
        for (x, y) in a.spots.iter().zip(&b.spots) {
            assert_eq!(x.position, y.position);
            assert_eq!((x.species, x.variant, x.tint), (y.species, y.variant, y.tint));
        }
        let treeline = t.min_height() + 0.66 * (t.max_height() - t.min_height());
        for s in &a.spots {
            assert!(t.slope_deg_at(s.position[0], s.position[2]) < TREE_MAX_SLOPE_DEG + 6.0);
            assert!(s.position[1] < treeline, "tree above the treeline");
            assert!(s.position[0].abs() < t.half() && s.position[2].abs() < t.half());
        }
    }

    #[test]
    fn trees_vary_in_species_scale_tint_and_rotation() {
        let t = small();
        let a = scatter(&t, &all_trees(), 3, 1_200);
        let species: std::collections::HashSet<_> = a.spots.iter().map(|s| s.species).collect();
        let tints: std::collections::HashSet<_> = a.spots.iter().map(|s| s.tint).collect();
        let (lo, hi) = a.spots.iter().fold((f32::MAX, f32::MIN), |(l, h), s| {
            (l.min(s.scale[1]), h.max(s.scale[1]))
        });
        assert!(species.len() >= 4, "species {species:?}");
        assert_eq!(tints.len(), TINT_BUCKETS as usize);
        assert!(hi / lo > 1.5, "scale range {lo}..{hi}");
        let yaw_spread = a.spots.iter().filter(|s| s.yaw_deg > 180.0).count();
        assert!(yaw_spread > 400 && yaw_spread < 800);
    }

    #[test]
    fn no_trees_are_scattered_without_a_tree_set() {
        let t = small();
        let config = Config::resolve(Some("none"), None, None, None, Some("forest"));
        assert!(scatter(&t, &config, 1, 100).spots.is_empty());
    }

    #[test]
    fn placements_use_tinted_materials_and_share_assets() {
        let t = small();
        let spots = scatter(&t, &all_trees(), 5, 500).spots;
        let placements = placements(&spots);
        assert_eq!(placements.len(), 500);
        assert!(placements.iter().all(|p| p.virtual_geometry));
        let assets: std::collections::HashSet<_> = placements.iter().map(|p| &p.asset).collect();
        assert!(assets.len() <= 16, "{} distinct assets", assets.len());
        assert!(placements
            .iter()
            .flat_map(|p| &p.slots)
            .all(|(_, m)| matches!(m, Mat::TreeTint(..))));
    }

    #[test]
    fn terrain_materials_resolve_to_the_ground_sets() {
        assert_eq!(Mat::Terrain(0).marble_set(), Some("forrest_ground_01"));
        assert_eq!(Mat::Terrain(1).marble_set(), Some("aerial_rocks_02"));
        let config = all_trees();
        let all = Mat::all(&config);
        assert!(all.contains(&Mat::Terrain(0)) && all.contains(&Mat::Terrain(1)));
        assert!(!Mat::all(&Config::default()).contains(&Mat::Terrain(0)));
    }

    /// Prints camera positions for `vg_stress_shots/forest.json` (not a test of
    /// anything: run it with `--ignored --nocapture` after changing the terrain
    /// or the scatter, and update the shot list from its output).
    #[test]
    #[ignore]
    fn print_vantage_points() {
        let config = all_trees();
        let forest = build(&config);
        let t = &forest.terrain;
        println!("terrain {:.1}..{:.1} m, {} trees", t.min_height(), t.max_height(), forest.spots.len());
        println!("valley centre {:?}", vantage(t));
        let mut by_height: Vec<&TreeSpot> = forest.spots.iter().collect();
        by_height.sort_by(|a, b| b.position[1].total_cmp(&a.position[1]));
        println!("highest tree {:?}", by_height[0].position);
        let mut per_species = std::collections::BTreeMap::new();
        for s in &forest.spots {
            *per_species.entry(catalog::TREES[s.species].id).or_insert(0usize) += 1;
        }
        println!("species {per_species:?}");
        // Densest 100 m cell.
        let mut grid = std::collections::HashMap::new();
        for s in &forest.spots {
            *grid.entry(((s.position[0] / 100.0) as i32, (s.position[2] / 100.0) as i32)).or_insert(0usize) += 1;
        }
        let best = grid.iter().max_by_key(|(_, n)| **n).unwrap();
        let (cx, cz) = ((best.0 .0 as f32 + 0.5) * 100.0, (best.0 .1 as f32 + 0.5) * 100.0);
        println!("densest cell centre ({cx}, {cz}) h {:.1} with {} trees", t.height_at(cx, cz), best.1);
        for (x, z) in [(0.0f32, 0.0f32), (300.0, 300.0), (-500.0, 200.0), (700.0, -600.0), (-900.0, -900.0)] {
            println!("height at ({x}, {z}) = {:.1}, slope {:.1}", t.height_at(x, z), t.slope_deg_at(x, z));
        }
    }
}
