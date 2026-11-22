//! PBR material mapping from SolidRS to Helio material assets.

use helio_mats::MaterialWorkflow;
use helio_mats::{
    GpuMaterial, MaterialTextureData, MaterialTextureSlot, TextureTransform, FLAG_ALPHA_TEST,
};
use solid_rs::scene::{AlphaMode, Material as SolidMaterial, TextureRef as SolidTextureRef};

use crate::texture_loader::TextureSemantic;
use crate::Result;

const MATERIAL_WORKFLOW_EPSILON: f32 = 1.0e-6;

#[derive(Debug, Clone, Copy)]
pub struct ConvertedTextureRef {
    pub texture_index: usize,
    pub uv_channel: u32,
    pub transform: TextureTransform,
}

#[derive(Debug, Clone, Default)]
pub struct ConvertedMaterialTextures {
    pub base_color: Option<ConvertedTextureRef>,
    pub normal: Option<ConvertedTextureRef>,
    pub roughness_metallic: Option<ConvertedTextureRef>,
    pub emissive: Option<ConvertedTextureRef>,
    pub occlusion: Option<ConvertedTextureRef>,
    pub specular_color: Option<ConvertedTextureRef>,
    pub specular_weight: Option<ConvertedTextureRef>,
    pub normal_scale: f32,
    pub occlusion_strength: f32,
    pub alpha_cutoff: f32,
}

#[derive(Debug, Clone)]
pub struct ConvertedMaterial {
    pub gpu: GpuMaterial,
    pub textures: ConvertedMaterialTextures,
}

impl ConvertedMaterial {
    /// The material's texture data ([`MaterialTextureData`]): each slot's
    /// texture as the table slot `slot_of` gives for its scene texture index
    /// (`ConvertedScene::textures`; `None`: not resident, the slot samples
    /// nothing), with its UV set and transform, and the material's normal
    /// scale, occlusion strength and alpha cutoff (0 unless alpha-tested).
    pub fn texture_data(
        &self,
        mut slot_of: impl FnMut(usize) -> Option<u32>,
    ) -> MaterialTextureData {
        let textures = &self.textures;
        let mut slot = |texture: &Option<ConvertedTextureRef>| {
            texture
                .as_ref()
                .and_then(|texture| {
                    slot_of(texture.texture_index).map(|index| {
                        MaterialTextureSlot::new(index, texture.uv_channel, &texture.transform)
                    })
                })
                .unwrap_or(MaterialTextureSlot::NONE)
        };
        let cutoff = if self.gpu.flags & FLAG_ALPHA_TEST != 0 {
            textures.alpha_cutoff
        } else {
            0.0
        };
        MaterialTextureData {
            base_color: slot(&textures.base_color),
            normal: slot(&textures.normal),
            roughness_metallic: slot(&textures.roughness_metallic),
            emissive: slot(&textures.emissive),
            occlusion: slot(&textures.occlusion),
            specular_color: slot(&textures.specular_color),
            specular_weight: slot(&textures.specular_weight),
            params: [
                textures.normal_scale,
                textures.occlusion_strength,
                cutoff,
                0.0,
            ],
        }
    }

    /// The material row and its texture data as a host uploads them (both
    /// at the material's row: `helio_mats::host_rows::material_rows`), the
    /// textures resolved by `slot_of` ([`Self::texture_data`]).
    pub fn rows(
        &self,
        slot_of: impl FnMut(usize) -> Option<u32>,
    ) -> (GpuMaterial, MaterialTextureData) {
        let data = self.texture_data(slot_of);
        (data.bind(self.gpu), data)
    }
}

fn uses_explicit_specular_ior_workflow(material: &SolidMaterial) -> bool {
    let specular_color_is_default = (material.specular_color.x - 1.0).abs()
        <= MATERIAL_WORKFLOW_EPSILON
        && (material.specular_color.y - 1.0).abs() <= MATERIAL_WORKFLOW_EPSILON
        && (material.specular_color.z - 1.0).abs() <= MATERIAL_WORKFLOW_EPSILON;
    let specular_weight_is_default =
        (material.specular_weight - 1.0).abs() <= MATERIAL_WORKFLOW_EPSILON;
    let ior_is_default = (material.ior - 1.5).abs() <= MATERIAL_WORKFLOW_EPSILON;

    !specular_color_is_default
        || material.specular_color_texture.is_some()
        || !specular_weight_is_default
        || material.specular_weight_texture.is_some()
        || !ior_is_default
}

fn convert_texture_ref(
    texture: &SolidTextureRef,
    semantic: TextureSemantic,
) -> ConvertedTextureRef {
    if texture.uv_channel > 1 {
        log::warn!(
            "Texture semantic {:?} uses UV channel {}, but the current Helio wrapper only carries UV0/UV1; falling back to UV1.",
            semantic,
            texture.uv_channel
        );
    }

    // SolidRS keeps a slot's `KHR_texture_transform` in glTF's convention,
    // which is the materials' own (`helio_mats::TextureTransform`): the
    // values carry over unchanged.
    let transform =
        texture
            .transform
            .as_ref()
            .map_or_else(TextureTransform::default, |transform| TextureTransform {
                offset: [transform.offset.x, transform.offset.y],
                scale: [transform.scale.x, transform.scale.y],
                rotation_radians: transform.rotation,
            });

    ConvertedTextureRef {
        texture_index: texture.texture_index,
        uv_channel: texture.uv_channel.min(1) as u32,
        transform,
    }
}

pub fn convert_material<F>(
    material: &SolidMaterial,
    mut resolve_texture: F,
) -> Result<ConvertedMaterial>
where
    F: FnMut(ConvertedTextureRef, TextureSemantic) -> Result<ConvertedTextureRef>,
{
    let mut workflow = MaterialWorkflow::Metallic as u32;
    let mut metallic = material.metallic_factor;
    let mut roughness = material.roughness_factor;
    let mut ior = 1.5;
    let mut specular = 0.5;

    if uses_explicit_specular_ior_workflow(material) {
        workflow = MaterialWorkflow::Specular as u32;
        metallic = material.specular_weight;
        roughness = material.roughness_factor;
        ior = material.ior;
        specular = material.specular_weight;
    }

    let mut flags = 0u32;
    match material.alpha_mode {
        AlphaMode::Opaque => {}
        AlphaMode::Mask => {
            flags |= 1 << 2;
        }
        AlphaMode::Blend => {
            flags |= 1 << 1;
        }
    }
    if material.double_sided {
        flags |= 1;
    }

    let convert_slot = |slot: &Option<SolidTextureRef>,
                        semantic: TextureSemantic,
                        resolve_texture: &mut F|
     -> Result<Option<ConvertedTextureRef>> {
        slot.as_ref()
            .map(|texture| resolve_texture(convert_texture_ref(texture, semantic), semantic))
            .transpose()
    };

    let textures = ConvertedMaterialTextures {
        base_color: convert_slot(
            &material.base_color_texture,
            TextureSemantic::BaseColor,
            &mut resolve_texture,
        )?,
        normal: convert_slot(
            &material.normal_texture,
            TextureSemantic::Normal,
            &mut resolve_texture,
        )?,
        roughness_metallic: convert_slot(
            &material.metallic_roughness_texture,
            TextureSemantic::MetallicRoughness,
            &mut resolve_texture,
        )?,
        emissive: convert_slot(
            &material.emissive_texture,
            TextureSemantic::Emissive,
            &mut resolve_texture,
        )?,
        occlusion: convert_slot(
            &material.occlusion_texture,
            TextureSemantic::Occlusion,
            &mut resolve_texture,
        )?,
        specular_color: convert_slot(
            &material.specular_color_texture,
            TextureSemantic::SpecularColor,
            &mut resolve_texture,
        )?,
        specular_weight: convert_slot(
            &material.specular_weight_texture,
            TextureSemantic::SpecularWeight,
            &mut resolve_texture,
        )?,
        normal_scale: material.normal_scale,
        occlusion_strength: material.occlusion_strength,
        alpha_cutoff: material.alpha_cutoff,
    };

    Ok(ConvertedMaterial {
        gpu: GpuMaterial {
            base_color: [
                material.base_color_factor.x,
                material.base_color_factor.y,
                material.base_color_factor.z,
                material.base_color_factor.w,
            ],
            emissive: [
                material.emissive_factor.x,
                material.emissive_factor.y,
                material.emissive_factor.z,
                1.0,
            ],
            roughness_metallic: [roughness, metallic, ior, specular],
            tex_base_color: GpuMaterial::NO_TEXTURE,
            tex_normal: GpuMaterial::NO_TEXTURE,
            tex_roughness: GpuMaterial::NO_TEXTURE,
            tex_emissive: GpuMaterial::NO_TEXTURE,
            tex_occlusion: GpuMaterial::NO_TEXTURE,
            workflow,
            flags,
            material_class: 0,
            class_params: [0.0; 4],
        },
        textures,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use solid_rs::scene::TextureTransform as SolidTextureTransform;

    /// The Poly Haven tree's branches: base colour, normal and ARM share
    /// `KHR_texture_transform` offset (0, -2.4), scale (15, 3.4).
    fn branches() -> SolidMaterial {
        let transformed = |texture_index| SolidTextureRef {
            texture_index,
            uv_channel: 0,
            transform: Some(SolidTextureTransform {
                offset: solid_rs::glam::Vec2::new(0.0, -2.4),
                rotation: 0.5,
                scale: solid_rs::glam::Vec2::new(15.0, 3.4),
            }),
        };
        let mut material = SolidMaterial::new("branches");
        material.base_color_texture = Some(transformed(0));
        material.normal_texture = Some(transformed(1));
        material.metallic_roughness_texture = Some(transformed(2));
        material.occlusion_texture = Some(SolidTextureRef::new(2));
        material.alpha_mode = AlphaMode::Mask;
        material.alpha_cutoff = 0.3;
        material.normal_scale = 0.8;
        material
    }

    /// A transformed slot reaches the row a host uploads: its table slot,
    /// its transform as glTF gives it, and the row reading its texture data.
    #[test]
    fn a_transformed_slot_reaches_the_uploaded_rows() {
        let converted = convert_material(&branches(), |texture, _| Ok(texture)).unwrap();
        // Scene textures 0, 1, 2 resident at table slots 10, 11, 12.
        let (row, data) = converted.rows(|index| Some(10 + index as u32));

        assert_ne!(row.flags & helio_mats::FLAG_TEXTURE_DATA, 0);
        assert_ne!(row.flags & helio_mats::FLAG_HAS_NORMAL_MAP, 0);
        assert_eq!(
            [
                row.tex_base_color,
                row.tex_normal,
                row.tex_roughness,
                row.tex_occlusion
            ],
            [10, 11, 12, 12]
        );

        let (sin, cos) = 0.5f32.sin_cos();
        for slot in [data.base_color, data.normal, data.roughness_metallic] {
            assert_eq!(slot.offset_scale, [0.0, -2.4, 15.0, 3.4]);
            assert_eq!(slot.rotation, [sin, cos, 0.0, 0.0]);
        }
        assert_eq!(data.base_color.texture_index, 10);
        // A slot without a transform samples with identity UVs.
        assert_eq!(data.occlusion.texture_index, 12);
        assert_eq!(data.occlusion.offset_scale, [0.0, 0.0, 1.0, 1.0]);
        assert_eq!(data.occlusion.rotation, [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(data.emissive, MaterialTextureSlot::NONE);
        assert_eq!(data.params, [0.8, 1.0, 0.3, 0.0]);

        // Where the shaders sample: the transform as glTF defines it.
        let t = data.base_color.transform();
        let uv = t.apply([0.1, 0.2]);
        let (u, v) = (0.1 * 15.0, 0.2 * 3.4);
        assert!((uv[0] - (cos * u + sin * v)).abs() < 1e-5);
        assert!((uv[1] - (-sin * u + cos * v - 2.4)).abs() < 1e-5);
    }

    /// A texture that is not resident samples nothing.
    #[test]
    fn a_texture_without_a_table_slot_samples_nothing() {
        let converted = convert_material(&branches(), |texture, _| Ok(texture)).unwrap();
        let (row, data) = converted.rows(|index| (index != 1).then_some(index as u32));
        assert_eq!(data.normal, MaterialTextureSlot::NONE);
        assert_eq!(row.tex_normal, GpuMaterial::NO_TEXTURE);
        assert_eq!(row.flags & helio_mats::FLAG_HAS_NORMAL_MAP, 0);
    }
}
