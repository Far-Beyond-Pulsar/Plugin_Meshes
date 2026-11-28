//! CPU-side asset payloads used to author SceneDB components.
//!
//! These types intentionally contain no GPU buffers, handles, arenas, or
//! renderer state. Frontends convert them into SceneDB component rows.
//!
//! Generic: a vertex-interchange layout (position/normal/tangent/uv packing)
//! any mesh-owning pass can use to author geometry, not a specific pass's
//! resource — the same tier as `GpuCameraUniforms`.

use bytemuck::{Pod, Zeroable};

/// A mesh pool vertex ([`crate::MESH_VERTICES`]), 32 bytes. The meshes own
/// both ends of it: their components write it and their passes read it.
///
/// Draws bind it through [`PackedVertex::LAYOUT`] (or one of the attributes
/// below); shaders that read the pool as storage decode it with the
/// `mesh_ray_surface` snippet. Saved `.mesh` files keep their own layout
/// (`meshes_component::mesh_cache`), so this one can change without them.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Pod, Zeroable)]
pub struct PackedVertex {
    pub position: [f32; 3],
    /// UV0: materials. It may tile far from zero, so it stays `f32`.
    pub tex_coords0: [f32; 2],
    /// UV1: the lightmap channel, unorm16 over `0..=1`
    /// ([`Self::lightmap_uv`]).
    pub tex_coords1: [u16; 2],
    /// snorm8x4: xyz; w unused.
    pub normal: u32,
    /// snorm8x4: xyz, and in w the bitangent's sign, +1 or -1
    /// ([`Self::bitangent_sign`]).
    pub tangent: u32,
}

unsafe impl pulsar_scenedb::Pod for PackedVertex {}

/// The bit of [`PackedVertex::tangent`] that is its w component's sign.
const TANGENT_W_NEGATIVE: u32 = 0x8000_0000;

impl PackedVertex {
    /// Bytes from one vertex to the next.
    pub const STRIDE: u64 = std::mem::size_of::<Self>() as u64;
    /// Shader location 0: `vec3<f32>`.
    pub const POSITION: wgpu::VertexAttribute = attribute(
        wgpu::VertexFormat::Float32x3,
        std::mem::offset_of!(Self, position),
        0,
    );
    /// Shader location 2: `vec2<f32>`.
    pub const UV0: wgpu::VertexAttribute = attribute(
        wgpu::VertexFormat::Float32x2,
        std::mem::offset_of!(Self, tex_coords0),
        2,
    );
    /// Shader location 5: `vec2<f32>`.
    pub const UV1: wgpu::VertexAttribute = attribute(
        wgpu::VertexFormat::Unorm16x2,
        std::mem::offset_of!(Self, tex_coords1),
        5,
    );
    /// Shader location 3: the packed `u32`.
    pub const NORMAL: wgpu::VertexAttribute = attribute(
        wgpu::VertexFormat::Uint32,
        std::mem::offset_of!(Self, normal),
        3,
    );
    /// Shader location 4: the packed `u32`, the bitangent's sign in its top
    /// byte.
    pub const TANGENT: wgpu::VertexAttribute = attribute(
        wgpu::VertexFormat::Uint32,
        std::mem::offset_of!(Self, tangent),
        4,
    );
    /// Every attribute, for a draw that reads the whole vertex.
    pub const ATTRIBUTES: [wgpu::VertexAttribute; 5] = [
        Self::POSITION,
        Self::UV0,
        Self::UV1,
        Self::NORMAL,
        Self::TANGENT,
    ];

    pub fn from_components(
        position: [f32; 3],
        normal: [f32; 3],
        tex_coords: [f32; 2],
        tangent: [f32; 3],
        bitangent_sign: f32,
    ) -> Self {
        let mut vertex = Self {
            position,
            tex_coords0: tex_coords,
            tex_coords1: [0, 0],
            normal: pack_snorm4x8([normal[0], normal[1], normal[2], 0.0]),
            tangent: pack_snorm4x8([tangent[0], tangent[1], tangent[2], 0.0]),
        };
        vertex.set_bitangent_sign(bitangent_sign);
        vertex
    }

    /// +1 or -1.
    pub fn bitangent_sign(&self) -> f32 {
        if self.tangent & TANGENT_W_NEGATIVE == 0 {
            1.0
        } else {
            -1.0
        }
    }

    /// Keeps only whether `sign` is negative.
    pub fn set_bitangent_sign(&mut self, sign: f32) {
        let w = if sign < 0.0 { -1.0 } else { 1.0 };
        self.tangent = (self.tangent & 0x00ff_ffff) | pack_snorm4x8([0.0, 0.0, 0.0, w]);
    }

    pub fn lightmap_uv(&self) -> [f32; 2] {
        self.tex_coords1.map(|c| f32::from(c) / f32::from(u16::MAX))
    }

    /// A coordinate outside `0..=1` is clamped to it: the channel is a
    /// lightmap unwrap.
    pub fn set_lightmap_uv(&mut self, uv: [f32; 2]) {
        self.tex_coords1 = uv.map(|c| (c.clamp(0.0, 1.0) * f32::from(u16::MAX)).round() as u16);
    }
}

const fn attribute(
    format: wgpu::VertexFormat,
    offset: usize,
    shader_location: u32,
) -> wgpu::VertexAttribute {
    wgpu::VertexAttribute {
        format,
        offset: offset as u64,
        shader_location,
    }
}

fn pack_snorm4x8(v: [f32; 4]) -> u32 {
    let to_i8 = |x: f32| -> u32 {
        let scaled = (x.clamp(-1.0, 1.0) * 127.0).round() as i8;
        scaled as u8 as u32
    };
    to_i8(v[0]) | (to_i8(v[1]) << 8) | (to_i8(v[2]) << 16) | (to_i8(v[3]) << 24)
}

#[derive(Debug, Clone)]
pub struct MeshUpload {
    pub vertices: Vec<PackedVertex>,
    pub indices: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct SectionedMeshUpload {
    pub vertices: Vec<PackedVertex>,
    pub sections: Vec<Vec<u32>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_vertex_is_32_bytes_and_keeps_the_sign_beside_the_tangent() {
        assert_eq!(PackedVertex::STRIDE, 32);
        for sign in [1.0, -1.0] {
            let v = PackedVertex::from_components(
                [1.0, 2.0, 3.0],
                [0.0, 1.0, 0.0],
                [7.25, -3.5],
                [1.0, 0.0, 0.0],
                sign,
            );
            assert_eq!(v.bitangent_sign(), sign);
            // The tangent's xyz is as it was packed.
            assert_eq!(v.tangent & 0x00ff_ffff, 0x0000_007f);
            assert_eq!(v.tex_coords0, [7.25, -3.5]);
        }
    }

    #[test]
    fn a_lightmap_uv_stays_within_half_a_step() {
        let mut v = PackedVertex::default();
        for uv in [[0.0, 1.0], [0.5, 0.125], [0.999_9, 0.000_3]] {
            v.set_lightmap_uv(uv);
            for (got, want) in v.lightmap_uv().into_iter().zip(uv) {
                assert!(
                    (got - want).abs() <= 0.5 / 65535.0 + f32::EPSILON,
                    "{got} {want}"
                );
            }
        }
        v.set_lightmap_uv([-1.0, 2.0]);
        assert_eq!(v.lightmap_uv(), [0.0, 1.0]);
    }
}
