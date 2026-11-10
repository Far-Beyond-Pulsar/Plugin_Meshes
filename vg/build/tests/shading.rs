//! The stored group error also bounds normal and uv change (cook revision 12).
//!
//! A flat plane simplifies with zero position error, so any error above the
//! position floor comes from the shading attributes.

use helio_vg_build::{build_dag, Dag, MeshInput};
use helio_vg_format::ATTR_STRIDE;

const N: usize = 48;

fn half_bits(x: f32) -> u16 {
    // Round to nearest even half for the normal range used here (|x| < 64).
    if x == 0.0 {
        return 0;
    }
    let (sign, a) = (if x < 0.0 { 0x8000u16 } else { 0 }, x.abs());
    let e = a.log2().floor() as i32;
    let m = ((a / 2f32.powi(e) - 1.0) * 1024.0).round() as u32;
    let (e, m) = if m == 1024 { (e + 1, 0) } else { (e, m) };
    if e < -14 {
        return sign;
    }
    sign | (((e + 15) as u16) << 10) | m as u16
}

/// A flat N x N plane, `attr(x, y)` giving (normal, uv) per vertex.
fn plane(attr: impl Fn(f32, f32) -> ([f32; 3], [f32; 2])) -> (Vec<[f32; 3]>, Vec<u8>, Vec<u32>) {
    let mut pos = Vec::new();
    let mut bytes = Vec::new();
    for j in 0..=N {
        for i in 0..=N {
            let (x, y) = (i as f32 / N as f32, j as f32 / N as f32);
            pos.push([x, y, 0.0]);
            let (n, uv) = attr(x, y);
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            for c in n {
                bytes.push((c / len * 127.0).round() as i8 as u8);
            }
            bytes.push(0);
            bytes.extend([127, 0, 0, 127]);
            for c in [uv[0], uv[1], uv[0], uv[1]] {
                bytes.extend(half_bits(c).to_le_bytes());
            }
        }
    }
    let mut idx = Vec::new();
    for j in 0..N {
        for i in 0..N {
            let a = (j * (N + 1) + i) as u32;
            let b = a + 1;
            let (c, d) = (a + (N + 1) as u32, a + (N + 1) as u32 + 1);
            idx.extend([a, b, c, b, d, c]);
        }
    }
    (pos, bytes, idx)
}

/// Largest finite group error of a plane with these attributes. `stride` 16 is
/// the format's layout (shading measured); any other stride is opaque.
fn max_error(stride: usize, attr: impl Fn(f32, f32) -> ([f32; 3], [f32; 2])) -> f32 {
    let (pos, bytes, idx) = plane(attr);
    let (attributes, stride) = if stride == ATTR_STRIDE as usize {
        (bytes, stride)
    } else {
        // Same vertices, opaque 12-byte records: position-only measurement.
        let opaque: Vec<u8> = pos
            .iter()
            .flat_map(|p| p.iter().flat_map(|c| c.to_le_bytes()))
            .collect();
        (opaque, 12)
    };
    let dag: Dag = build_dag(&MeshInput {
        positions: &pos,
        attributes: &attributes,
        attribute_stride: stride,
        indices: &idx,
        tri_material: &[],
    })
    .unwrap();
    dag.groups
        .iter()
        .filter(|g| g.error.is_finite())
        .map(|g| g.error)
        .fold(0.0, f32::max)
}

#[test]
fn a_flat_plane_with_smooth_attributes_has_only_the_position_floor() {
    let plain = |x: f32, y: f32| ([0.0, 0.0, 1.0], [x, y]);
    let (shaded, opaque) = (max_error(16, plain), max_error(12, plain));
    assert!(shaded < 1e-3, "linear uv on a flat plane: {shaded}");
    assert!(opaque < 1e-3, "{opaque}");
}

#[test]
fn normals_the_simplified_surface_cannot_reproduce_raise_the_error() {
    // Position-wise the plane is exact; the normal field is not linear.
    let bumpy = |x: f32, y: f32| {
        (
            [0.6 * (12.0 * x).sin(), 0.6 * (12.0 * y).sin(), 1.0],
            [x, y],
        )
    };
    let (shaded, opaque) = (max_error(16, bumpy), max_error(12, bumpy));
    assert!(
        shaded > 20.0 * opaque.max(1e-5),
        "normal deviation must reach the stored error: {shaded} vs position-only {opaque}"
    );
}

#[test]
fn uv_slides_the_simplified_surface_cannot_reproduce_raise_the_error() {
    // A nonlinear uv warp: bilinear interpolation across a coarse triangle is
    // off by the curvature of the warp, in object-space length.
    let warped = |x: f32, y: f32| ([0.0, 0.0, 1.0], [x * x, y]);
    let (shaded, opaque) = (max_error(16, warped), max_error(12, warped));
    assert!(
        shaded > 20.0 * opaque.max(1e-5),
        "uv deviation must reach the stored error: {shaded} vs position-only {opaque}"
    );
}
