/// Clears one face's tile of the transmittance layer to zero (zero is
/// unfiltered): one triangle over the viewport `ShadowPass` sets to the tile,
/// with `fs_clear` appended by `Transmittance::new`.
///
/// The face is gated on a dirty bit (`face.dynamic`: 2 for the static layer
/// the transmittance follows), so a face whose static layer is not being
/// rebuilt and was not dirtied on the GPU keeps its cached tile; 0 clears it
/// unconditionally.

struct Face { value:u32, dynamic:u32, pad0:u32, pad1:u32 }
@group(0) @binding(2) var<uniform> face:Face;
@group(0) @binding(4) var<storage,read> dirty:array<u32>;

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    if face.dynamic != 0u && (dirty[face.value] & face.dynamic) == 0u { return vec4f(2,2,2,1); }
    let corner = vec2f(f32((vid << 1u) & 2u), f32(vid & 2u));
    return vec4f(corner * 2.0 - vec2f(1.0), 1.0, 1.0);
}
