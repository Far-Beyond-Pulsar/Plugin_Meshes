//! The api's WGSL snippets declare the rows as the Rust does: each struct's
//! size and each constant's value, so a reader including the snippet reads
//! the buffers the meshes write.

use helio_mesh_api::*;

/// The storage `mesh_ray_surface` reads, as a reader declares it.
const RAY_SURFACE_READER: &str = "//!use mesh_ray_surface
//!use frame_ray_stream
@group(0) @binding(0) var<storage, read> instances: array<RayInstance>;
@group(0) @binding(1) var<storage, read> vertices: array<u32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read> proxy_vertices: array<f32>;
@group(0) @binding(4) var<storage, read> proxy_indices: array<u32>;
@group(0) @binding(5) var<storage, read> skinned_vertices: array<u32>;
@group(0) @binding(6) var<storage, read> transforms: array<RayTransform>;
@group(0) @binding(7) var<storage, read> space_groups: array<u32>;
@compute @workgroup_size(1)
fn main() {
    let t = ray_triangle(instances[0], 0u);
    let n = ray_triangle_outside(t, vec3<f32>(1.0, 0.0, 0.0));
    let x = transforms[instances[0].instance];
    _ = ray_transform_point(x, n) + ray_transform_vector(x, n) + ray_transform_normal(x, n);
    _ = ray_space_cull(3u, ray_instance_space(instances[0]) + 1u, space_groups[0]);
}
";

const DRAW_ROWS_READER: &str = "//!use mesh_draw_rows
@group(0) @binding(0) var<storage, read> instances: array<GpuInstanceData>;
@group(0) @binding(1) var<storage, read> draws: array<GpuDrawCall>;
@group(0) @binding(2) var<storage, read_write> indirect: array<DrawIndexedIndirect>;
@compute @workgroup_size(1)
fn main() {
    _ = instance_coordinate_space(instances[0].flags);
    indirect[0] = DrawIndexedIndirect(draws[0].index_count, 0u, 0u, 0, 0u);
}
";

fn module(reader: &str, snippet: helio_core::shader::ShaderSnippet) -> naga::Module {
    modules(reader, &[snippet])
}

fn modules(reader: &str, snippets: &[helio_core::shader::ShaderSnippet]) -> naga::Module {
    let source = helio_core::shader::resolve_with(reader, snippets);
    let module = naga::front::wgsl::parse_str(&source)
        .unwrap_or_else(|error| panic!("{}", error.emit_to_string(&source)));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|error| panic!("{error:?}"));
    module
}

fn struct_size(module: &naga::Module, name: &str) -> usize {
    module
        .types
        .iter()
        .find_map(|(_, ty)| match ty.inner {
            naga::TypeInner::Struct { span, .. } if ty.name.as_deref() == Some(name) => {
                Some(span as usize)
            }
            _ => None,
        })
        .unwrap_or_else(|| panic!("no struct {name}"))
}

fn constant(module: &naga::Module, name: &str) -> u32 {
    let (_, c) = module
        .constants
        .iter()
        .find(|(_, c)| c.name.as_deref() == Some(name))
        .unwrap_or_else(|| panic!("no constant {name}"));
    match module.global_expressions[c.init] {
        naga::Expression::Literal(naga::Literal::U32(v)) => v,
        ref other => panic!("{name}: {other:?}"),
    }
}

#[test]
fn the_ray_surface_rows_match_the_rust() {
    let m = modules(
        RAY_SURFACE_READER,
        &[RAY_SURFACE_SNIPPET, helio_frame_api::rays::stream::SNIPPET],
    );
    assert_eq!(
        struct_size(&m, "RayInstance"),
        std::mem::size_of::<RayInstance>()
    );
    assert_eq!(
        struct_size(&m, "RayTransform"),
        std::mem::size_of::<RayTransform>()
    );
    assert_eq!(
        constant(&m, "RAY_INSTANCE_SPACE_SHIFT"),
        RAY_INSTANCE_SPACE_SHIFT
    );
    assert_eq!(constant(&m, "RAY_SPACE_GROUP"), RAY_SPACE_GROUP);
    assert_eq!(constant(&m, "RAY_SPACE_SHARED"), RAY_SPACE_SHARED);
    assert_eq!(constant(&m, "RAY_SPACE_PRESENT"), RAY_SPACE_PRESENT);
    assert_eq!(constant(&m, "RAY_INSTANCE_PROXY"), RAY_INSTANCE_PROXY);
    assert_eq!(constant(&m, "RAY_INSTANCE_MASKED"), RAY_INSTANCE_MASKED);
    assert_eq!(
        constant(&m, "RAY_INSTANCE_TWO_SIDED"),
        RAY_INSTANCE_TWO_SIDED
    );
    assert_eq!(
        constant(&m, "MESH_VERTEX_WORDS") as usize * 4,
        std::mem::size_of::<PackedVertex>()
    );
    assert_eq!(
        u64::from(constant(&m, "RAY_PROXY_FLOATS")) * 4,
        RAY_PROXY_VERTEX_BYTES
    );
    assert_eq!(constant(&m, "RAY_INSTANCE_SKINNED"), RAY_INSTANCE_SKINNED);
    assert_eq!(
        constant(&m, "SKINNED_VERTEX_WORDS") as usize * 4,
        std::mem::size_of::<SkinnedVertex>()
    );
}

/// The storage `mesh_skinning` reads, as a drawer declares it.
const SKINNING_READER: &str = "//!use mesh_draw_rows
//!use mesh_skinning
//!use frame_ray_stream
@group(0) @binding(0) var<storage, read> instances: array<GpuInstanceData>;
@group(0) @binding(1) var<storage, read> skin_cache: array<SkinnedVertex>;
@group(0) @binding(2) var<storage, read> skin_cache_rows: array<SkinCacheRow>;
@compute @workgroup_size(1)
fn main() {
    let v = mesh_skinned_vertex(instances[0], 0u, vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
    _ = v.previous + mesh_skinned_position(instances[0], vec3<f32>(0.0), 0u);
}
";

#[test]
fn the_skin_cache_rows_match_the_rust() {
    let m = modules(
        SKINNING_READER,
        &[
            DRAW_ROWS_SNIPPET,
            SKINNING_SNIPPET,
            helio_frame_api::rays::stream::SNIPPET,
        ],
    );
    assert_eq!(
        struct_size(&m, "SkinnedVertex"),
        std::mem::size_of::<SkinnedVertex>()
    );
    assert_eq!(
        struct_size(&m, "SkinCacheRow"),
        std::mem::size_of::<SkinCacheRow>()
    );
}

const OBJECT_ROW_READER: &str = "//!use mesh_object_row
@group(0) @binding(0) var<storage, read> objects: array<StaticObjectRow>;
@compute @workgroup_size(1)
fn main() {
    _ = objects[0].bounds[3];
}
";

#[test]
fn the_object_row_matches_the_rust() {
    let m = module(OBJECT_ROW_READER, OBJECT_ROW_SNIPPET);
    assert_eq!(
        struct_size(&m, "StaticObjectRow"),
        std::mem::size_of::<StaticObjectRow>()
    );
}

#[test]
fn the_draw_rows_match_the_rust() {
    let m = module(DRAW_ROWS_READER, DRAW_ROWS_SNIPPET);
    assert_eq!(
        struct_size(&m, "GpuInstanceData"),
        std::mem::size_of::<GpuInstanceData>()
    );
    assert_eq!(
        struct_size(&m, "GpuDrawCall"),
        std::mem::size_of::<GpuDrawCall>()
    );
    assert_eq!(
        struct_size(&m, "DrawIndexedIndirect"),
        std::mem::size_of::<DrawIndexedIndirectArgs>()
    );
    for (name, value) in [
        ("INSTANCE_FLAG_CASTS_SHADOW", INSTANCE_FLAG_CASTS_SHADOW),
        (
            "INSTANCE_FLAG_RECEIVES_SHADOW",
            INSTANCE_FLAG_RECEIVES_SHADOW,
        ),
        ("INSTANCE_FLAG_ALWAYS_VISIBLE", INSTANCE_FLAG_ALWAYS_VISIBLE),
        ("INSTANCE_FLAG_MOVABLE", INSTANCE_FLAG_MOVABLE),
        ("INSTANCE_FLAG_SKINNED", INSTANCE_FLAG_SKINNED),
        (
            "INSTANCE_FLAG_CLASSIC_SUPPRESSED",
            INSTANCE_FLAG_CLASSIC_SUPPRESSED,
        ),
        (
            "INSTANCE_COORDINATE_SPACE_SHIFT",
            INSTANCE_COORDINATE_SPACE_SHIFT,
        ),
    ] {
        assert_eq!(constant(&m, name), value, "{name}");
    }
}

/// The space groups: the world alone in group 0 (its masks the frame's
/// `MASK_STATIC` / `MASK_MOVABLE`), up to three other spaces a group each,
/// shared past that.
#[test]
fn spaces_take_mask_groups() {
    use helio_frame_api::rays::stream::{MASK_MOVABLE, MASK_STATIC};
    assert_eq!(ray_instance_mask(0, false), MASK_STATIC);
    assert_eq!(ray_instance_mask(0, true), MASK_MOVABLE);
    assert_eq!(ray_instance_mask(3, true), 0x80);
    let table = ray_space_groups([0, 5, 9, 5]);
    assert_eq!(table[0], RAY_SPACE_PRESENT);
    assert_eq!(table[5], 1 | RAY_SPACE_PRESENT);
    assert_eq!(table[9], 2 | RAY_SPACE_PRESENT);
    assert_eq!(table[7], 0, "a space with nothing traced");
    let table = ray_space_groups([1, 2, 3, 4]);
    assert_eq!(table[0], 0, "no world instance traced");
    for (slot, group) in [(1, 1), (2, 2), (3, 3), (4, 1)] {
        assert_eq!(
            table[slot],
            group | RAY_SPACE_PRESENT | RAY_SPACE_SHARED,
            "slot {slot}"
        );
    }
    // The row keeps its space beside its flags.
    let row = RayInstance {
        flags: RAY_INSTANCE_MASKED,
        ..Default::default()
    }
    .in_space(200);
    assert_eq!(row.space(), 200);
    assert_eq!(row.flags & 0xff, RAY_INSTANCE_MASKED);
}
