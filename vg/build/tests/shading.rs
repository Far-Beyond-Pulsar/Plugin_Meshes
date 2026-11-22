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
    // The simplifier weighs uv too, so it keeps the warp to about 1e-3 at the
    // coarsest finite level. The position-only error is a two-sided bound,
    // not a sampled zero: about 1e-4 here, where an original triangle owned by
    // one group lies a fraction of an edge beyond its border. The uv slide
    // must still raise the stored error by an order of magnitude.
    assert!(
        shaded > 10.0 * opaque.max(1e-5),
        "uv deviation must reach the stored error: {shaded} vs position-only {opaque}"
    );
}

/// A UV sphere (`rings x segments`, uv seam and poles), records in the
/// format's layout, as an imported mesh reaches the builder.
fn uv_sphere(rings: u32, segments: u32) -> (Vec<[f32; 3]>, Vec<u8>, Vec<u32>) {
    let (mut pos, mut bytes) = (Vec::new(), Vec::new());
    for ring in 0..=rings {
        let theta = std::f32::consts::PI * ring as f32 / rings as f32;
        for segment in 0..=segments {
            // The seam column repeats the first one bit for bit, as an exporter
            // writes it (sin(TAU) is not 0 in f32).
            let phi = std::f32::consts::TAU * (segment % segments) as f32 / segments as f32;
            // Poles are one point (sin(PI) is not 0 in f32 either).
            let s = if ring == 0 || ring == rings {
                0.0
            } else {
                theta.sin()
            };
            let n = [s * phi.cos(), theta.cos(), s * phi.sin()];
            pos.push(n);
            bytes.extend(
                helio_vg_format::VertexAttr {
                    normal: n,
                    tangent: [-phi.sin(), 0.0, phi.cos()],
                    bitangent_sign: 1.0,
                    uv0: [segment as f32 / segments as f32, ring as f32 / rings as f32],
                    uv1: [segment as f32 / segments as f32, ring as f32 / rings as f32],
                }
                .encode(),
            );
        }
    }
    let mut idx = Vec::new();
    let stride = segments + 1;
    for ring in 0..rings {
        for segment in 0..segments {
            let a = ring * stride + segment;
            let b = a + stride;
            idx.extend([a, b, a + 1, a + 1, b, b + 1]);
        }
    }
    (pos, bytes, idx)
}

#[test]
fn a_uv_sphere_reduces_to_a_coarse_root() {
    let (pos, bytes, idx) = uv_sphere(120, 160);
    let dag = build_dag(&MeshInput {
        positions: &pos,
        attributes: &bytes,
        attribute_stride: ATTR_STRIDE as usize,
        indices: &idx,
        tri_material: &[],
    })
    .unwrap();
    let root: usize = dag
        .groups
        .iter()
        .filter(|g| g.is_root)
        .flat_map(|g| &g.clusters)
        .map(|&c| dag.clusters[c as usize].tris.len())
        .sum();
    println!(
        "uv sphere: {} source triangles, root {root}",
        dag.source_tris
    );
    assert!(
        root * 50 < dag.source_tris,
        "root {root} of {}",
        dag.source_tris
    );
}
