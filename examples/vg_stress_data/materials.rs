//! Material catalogue of the museum: which surface each kit slot / statue /
//! tree part uses.
//!
//! The four marble variants are PBR texture sets from Poly Haven (diffuse,
//! normal, ARM); tree parts take their textures from the tree glTFs. Every
//! material also has flat scalar values, used until its textures have loaded
//! and as the fallback when a download failed so the scene still opens.

use crate::vg_stress_prep::catalog::{self, Config};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Mat {
    /// marble_01: clean white (columns, entablature, walls).
    MarbleWhite,
    /// grey_cartago_01: rich dark veined marble (floor, pedestals).
    MarbleCartago,
    /// grey_cartago_03: grey veined marble (Ionic columns, accents).
    MarbleGrey,
    /// marble_mosaic_tiles: floor inlay and plaza.
    MarbleMosaic,
    Basalt,
    BasaltMirror,
    Bronze,
    Gold,
    Terracotta,
    Limestone,
    ClothRed,
    ClothBlue,
    Flame,
    Foliage,
    Trunk,
    Ground,
    // statue stones (scalar only: scanned statues carry no UVs)
    StatueMarble,
    StatueGreenStone,
    StatueBronze,
    StatueLimestone,
    /// `(index in catalog::TREES, slot index)`; textured from the tree glTF.
    Tree(u8, u8),
    /// `(index in catalog::TREES, slot index, tint bucket)`: the same glTF
    /// material with its base colour multiplied by [`tint`], so one species
    /// does not read as copies. Forest scene only.
    TreeTint(u8, u8, u8),
    /// Terrain surface: 0 forest floor, 1 rock (`catalog::GROUNDS`).
    Terrain(u8),
}

/// Tint buckets per tree material in the forest scene.
pub const TINT_BUCKETS: u8 = 6;

/// Base-colour multiplier of tint bucket `bucket`. Leaf and needle cards
/// (`masked`, or fallback-green slots) shift hue and value (young yellow-green
/// to old blue-green, some autumn-dry); bark only varies in value, because
/// tinting a photographed bark hue looks painted.
pub fn tint(bucket: u8, foliage: bool) -> [f32; 3] {
    const FOLIAGE: [[f32; 3]; TINT_BUCKETS as usize] = [
        [1.00, 1.00, 1.00],
        [1.12, 1.08, 0.78],
        [0.86, 1.00, 0.92],
        [0.94, 0.90, 0.80],
        [1.20, 1.04, 0.70],
        [0.80, 0.92, 0.96],
    ];
    const BARK: [f32; TINT_BUCKETS as usize] = [1.0, 1.12, 0.88, 0.96, 1.2, 0.8];
    let b = (bucket % TINT_BUCKETS) as usize;
    if foliage {
        FOLIAGE[b]
    } else {
        [BARK[b]; 3]
    }
}

/// Scalar surface values of a material.
pub struct Surface {
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub emissive_color: [f32; 3],
    pub emissive_intensity: f32,
}

impl Mat {
    pub const SCALAR: [Mat; 20] = [
        Mat::MarbleWhite,
        Mat::MarbleCartago,
        Mat::MarbleGrey,
        Mat::MarbleMosaic,
        Mat::Basalt,
        Mat::BasaltMirror,
        Mat::Bronze,
        Mat::Gold,
        Mat::Terracotta,
        Mat::Limestone,
        Mat::ClothRed,
        Mat::ClothBlue,
        Mat::Flame,
        Mat::Foliage,
        Mat::Trunk,
        Mat::Ground,
        Mat::StatueMarble,
        Mat::StatueGreenStone,
        Mat::StatueBronze,
        Mat::StatueLimestone,
    ];

    /// Every material the scene needs for `config` (scalar ones plus the
    /// material slots of the selected trees).
    pub fn all(config: &Config) -> Vec<Mat> {
        let forest = config.scene == catalog::Scene::Forest;
        let mut v = Self::SCALAR.to_vec();
        if forest {
            v.extend((0..catalog::GROUNDS.len() as u8).map(Mat::Terrain));
        }
        for (ti, tree) in catalog::TREES.iter().enumerate() {
            if config.has_tree(tree.id) {
                v.extend((0..tree.slots.len()).map(|si| Mat::Tree(ti as u8, si as u8)));
                if forest {
                    for si in 0..tree.slots.len() {
                        v.extend((0..TINT_BUCKETS).map(|b| Mat::TreeTint(ti as u8, si as u8, b)));
                    }
                }
            }
        }
        v
    }

    /// The material of tree `tree_id`'s glTF slot `slot` in tint bucket `bucket`.
    pub fn tree_slot_tinted(tree_id: &str, slot: &str, bucket: u8) -> Option<Mat> {
        match Self::tree_slot(tree_id, slot)? {
            Mat::Tree(t, s) => Some(Mat::TreeTint(t, s, bucket % TINT_BUCKETS)),
            _ => None,
        }
    }

    /// The material of tree `tree_id`'s glTF slot `slot`.
    pub fn tree_slot(tree_id: &str, slot: &str) -> Option<Mat> {
        let ti = catalog::TREES.iter().position(|t| t.id == tree_id)?;
        let si = catalog::TREES[ti]
            .slots
            .iter()
            .position(|s| s.name == slot)?;
        Some(Mat::Tree(ti as u8, si as u8))
    }

    pub fn name(self) -> &'static str {
        match self {
            Mat::MarbleWhite => "marble_white",
            Mat::MarbleCartago => "marble_cartago",
            Mat::MarbleGrey => "marble_grey",
            Mat::MarbleMosaic => "marble_mosaic",
            Mat::Basalt => "basalt",
            Mat::BasaltMirror => "basalt_mirror",
            Mat::Bronze => "bronze",
            Mat::Gold => "gold",
            Mat::Terracotta => "terracotta",
            Mat::Limestone => "limestone",
            Mat::ClothRed => "cloth_red",
            Mat::ClothBlue => "cloth_blue",
            Mat::Flame => "flame",
            Mat::Foliage => "foliage",
            Mat::Trunk => "trunk",
            Mat::Ground => "ground",
            Mat::StatueMarble => "statue_marble",
            Mat::StatueGreenStone => "statue_green_stone",
            Mat::StatueBronze => "statue_bronze",
            Mat::StatueLimestone => "statue_limestone",
            Mat::Tree(t, s) | Mat::TreeTint(t, s, _) => {
                catalog::TREES[t as usize].slots[s as usize].name
            }
            Mat::Terrain(0) => "terrain_forest_floor",
            Mat::Terrain(_) => "terrain_rock",
        }
    }

    /// Poly Haven asset id of a textured marble variant.
    pub fn marble_set(self) -> Option<&'static str> {
        match self {
            Mat::MarbleWhite => Some("marble_01"),
            Mat::MarbleCartago => Some("grey_cartago_01"),
            Mat::MarbleGrey => Some("grey_cartago_03"),
            Mat::MarbleMosaic => Some("marble_mosaic_tiles"),
            Mat::Terrain(i) => catalog::GROUNDS.get(i as usize).map(|g| g.id),
            _ => None,
        }
    }

    /// `(tree, slot)` of a tree material.
    pub fn tree_part(self) -> Option<(&'static catalog::TreeDef, &'static catalog::TreeSlot)> {
        match self {
            Mat::Tree(t, s) | Mat::TreeTint(t, s, _) => {
                let tree = &catalog::TREES[t as usize];
                Some((tree, &tree.slots[s as usize]))
            }
            _ => None,
        }
    }

    pub fn surface(self) -> Surface {
        let s = |base: [f32; 4], metallic, roughness| Surface {
            base_color: base,
            metallic,
            roughness,
            emissive_color: [0.0; 3],
            emissive_intensity: 0.0,
        };
        match self {
            // Polished marble: glossy, so HLFS reflections read on the floor.
            Mat::MarbleWhite => s([0.86, 0.85, 0.81, 1.0], 0.0, 0.10),
            Mat::MarbleCartago => s([0.09, 0.09, 0.10, 1.0], 0.0, 0.08),
            Mat::MarbleGrey => s([0.50, 0.50, 0.52, 1.0], 0.0, 0.12),
            Mat::MarbleMosaic => s([0.62, 0.56, 0.46, 1.0], 0.0, 0.12),
            Mat::Basalt => s([0.035, 0.038, 0.042, 1.0], 0.0, 0.18),
            Mat::BasaltMirror => s([0.012, 0.016, 0.02, 1.0], 0.0, 0.02),
            Mat::Bronze => s([0.62, 0.38, 0.14, 1.0], 1.0, 0.28),
            Mat::Gold => s([0.92, 0.70, 0.24, 1.0], 1.0, 0.20),
            Mat::Terracotta => s([0.55, 0.26, 0.15, 1.0], 0.0, 0.62),
            Mat::Limestone => s([0.72, 0.66, 0.54, 1.0], 0.0, 0.72),
            Mat::ClothRed => s([0.42, 0.04, 0.05, 1.0], 0.0, 0.88),
            Mat::ClothBlue => s([0.05, 0.10, 0.38, 1.0], 0.0, 0.88),
            Mat::Flame => Surface {
                base_color: [1.0, 0.55, 0.15, 1.0],
                metallic: 0.0,
                roughness: 0.5,
                emissive_color: [1.0, 0.36, 0.05],
                emissive_intensity: 6.0,
            },
            Mat::Foliage => s([0.06, 0.15, 0.07, 1.0], 0.0, 0.8),
            Mat::Trunk => s([0.17, 0.10, 0.05, 1.0], 0.0, 0.9),
            Mat::Ground => s([0.27, 0.23, 0.16, 1.0], 0.0, 0.95),
            Mat::StatueMarble => s([0.88, 0.87, 0.83, 1.0], 0.0, 0.14),
            Mat::StatueGreenStone => s([0.06, 0.20, 0.12, 1.0], 0.0, 0.10),
            Mat::StatueBronze => s([0.50, 0.30, 0.12, 1.0], 1.0, 0.24),
            Mat::StatueLimestone => s([0.70, 0.64, 0.52, 1.0], 0.0, 0.55),
            Mat::Tree(..) | Mat::TreeTint(..) => {
                let (_, slot) = self.tree_part().unwrap();
                let rgb = match self {
                    // A glTF slot's texture carries the colour; the tint scales it.
                    Mat::TreeTint(_, _, b) => {
                        tint(b, slot.masked || slot.fallback[1] > slot.fallback[0])
                    }
                    _ => [1.0; 3],
                };
                s(
                    [
                        slot.fallback[0] * rgb[0],
                        slot.fallback[1] * rgb[1],
                        slot.fallback[2] * rgb[2],
                        1.0,
                    ],
                    0.0,
                    0.85,
                )
            }
            Mat::Terrain(0) => s([0.20, 0.16, 0.10, 1.0], 0.0, 0.95),
            Mat::Terrain(_) => s([0.30, 0.29, 0.27, 1.0], 0.0, 0.9),
        }
    }

    /// Base colour factor once the base-colour texture is bound: the texture
    /// carries the colour, a tinted material only scales it.
    pub fn texture_tint(self) -> [f32; 4] {
        match self {
            Mat::TreeTint(_, _, b) => {
                let (_, slot) = self.tree_part().unwrap();
                let [r, g, bl] = tint(b, slot.masked || slot.fallback[1] > slot.fallback[0]);
                [r, g, bl, 1.0]
            }
            _ => [1.0; 4],
        }
    }

    /// The material a kit slot of this name gets unless the placement overrides it.
    pub fn for_kit_slot(slot: &str) -> Mat {
        match slot {
            "marble" => Mat::MarbleWhite,
            "marble_alt" => Mat::MarbleGrey,
            "marble_green" => Mat::MarbleMosaic,
            "basalt" => Mat::Basalt,
            "water" => Mat::BasaltMirror,
            "bronze" => Mat::Bronze,
            "gold" => Mat::Gold,
            "terracotta" => Mat::Terracotta,
            "limestone" => Mat::Limestone,
            "cloth_red" => Mat::ClothRed,
            "cloth_blue" => Mat::ClothBlue,
            "flame" => Mat::Flame,
            "foliage" => Mat::Foliage,
            "trunk" => Mat::Trunk,
            "ground" => Mat::Ground,
            _ => Mat::MarbleWhite,
        }
    }
}

/// Material slot names of every kit FBX, in slot order (`kit_manifest.json`
/// records the same; a test compares them when the kit has been built).
pub const KIT_SLOTS: &[(&str, &[&str])] = &[
    ("doric_column", &["marble"]),
    ("ionic_column", &["marble"]),
    ("corinthian_column", &["marble"]),
    ("doric_entablature_mid", &["marble"]),
    ("doric_entablature_end_l", &["marble"]),
    ("doric_entablature_end_r", &["marble"]),
    ("ionic_entablature", &["marble"]),
    ("pediment", &["marble"]),
    ("acroterion", &["marble"]),
    ("acroterion_small", &["marble"]),
    ("antefix", &["marble"]),
    ("pedestal_small", &["marble"]),
    ("pedestal_medium", &["marble"]),
    ("pedestal_large", &["marble"]),
    ("pedestal_tall", &["marble"]),
    ("pedestal_round", &["marble"]),
    ("bench", &["marble"]),
    ("brazier", &["bronze", "basalt"]),
    ("flame", &["flame"]),
    ("banner", &["cloth_red", "bronze", "gold"]),
    ("urn", &["terracotta"]),
    ("cypress", &["foliage", "trunk"]),
    ("stele", &["marble", "bronze"]),
    ("crepidoma", &["marble", "marble_alt"]),
    (
        "floor",
        &["marble", "marble_green", "basalt", "bronze", "marble_alt"],
    ),
    ("cella_walls", &["marble", "marble_alt", "bronze", "basalt"]),
    ("ceilings", &["marble", "gold"]),
    ("roof", &["terracotta", "marble"]),
    ("impluvium", &["marble", "water"]),
    ("plaza", &["limestone", "marble", "basalt"]),
    ("ground", &["ground"]),
];

pub fn kit_slots(piece: &str) -> &'static [&'static str] {
    KIT_SLOTS
        .iter()
        .find(|(name, _)| *name == piece)
        .map(|(_, slots)| *slots)
        .unwrap_or_else(|| panic!("unknown kit piece {piece}"))
}
