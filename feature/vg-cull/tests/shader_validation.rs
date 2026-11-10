//! naga parse + validate of the fully expanded cull shader, and the struct
//! sizes the Rust side uploads.

use helio_pass_vg_cull::{validation_sources, BIN_COUNT, MAX_BIN_CAPACITY};
use helio_vg_format::MAX_VISIBLE_CLUSTERS;

fn module() -> naga::Module {
    let (_, src) = validation_sources().into_iter().next().unwrap();
    naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| panic!("{}", e.emit_to_string(&src)))
}

#[test]
fn cull_shader_validates() {
    for (name, src) in validation_sources() {
        let m = naga::front::wgsl::parse_str(&src)
            .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(&src)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&m)
        .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        for entry in [
            "cs_tiles",
            "cs_seed",
            "cs_round_prepare",
            "cs_traverse",
            "cs_finalize",
        ] {
            assert!(
                m.entry_points.iter().any(|e| e.name == entry),
                "missing {entry}"
            );
        }
    }
}

#[test]
fn shader_struct_sizes_match_the_uploads() {
    let m = module();
    let mut layouter = naga::proc::Layouter::default();
    layouter.update(m.to_ctx()).unwrap();
    let size = |name: &str| {
        let (h, _) = m
            .types
            .iter()
            .find(|(_, t)| t.name.as_deref() == Some(name))
            .unwrap_or_else(|| panic!("missing {name}"));
        layouter[h].size
    };
    assert_eq!(size("CullParams"), 96);
    assert_eq!(size("VgBvhNode"), 32);
    assert_eq!(size("VgBvhHeader"), 128);
    assert_eq!(size("RoundParams"), 16);
    // Scene-join row: 63 words, only `transform` (word 4) is read.
    assert_eq!(size("VgObjectRow"), 252);
}

#[test]
fn vis_id_can_address_every_bin() {
    assert!(BIN_COUNT * MAX_BIN_CAPACITY <= MAX_VISIBLE_CLUSTERS);
}

/// Windows on ARM has no DXC, so wgpu compiles through FXC, which rejects a
/// barrier in flow control it cannot prove uniform (X4026). Compile every
/// entry point's HLSL with the SDK's `fxc` when one is installed.
#[cfg(windows)]
#[test]
fn cull_shader_compiles_under_fxc() {
    use std::process::Command;
    let Some(fxc) = find_fxc() else {
        eprintln!("fxc not found; skipping");
        return;
    };
    let (_, src) = validation_sources().into_iter().next().unwrap();
    let m = naga::front::wgsl::parse_str(&src).unwrap();
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&m)
    .unwrap();
    let options = naga::back::hlsl::Options {
        shader_model: naga::back::hlsl::ShaderModel::V5_1,
        fake_missing_bindings: true,
        ..Default::default()
    };
    let mut hlsl = String::new();
    let pipeline = Default::default();
    let mut writer = naga::back::hlsl::Writer::new(&mut hlsl, &options, &pipeline);
    writer.write(&m, &info, None).unwrap();
    let dir = std::env::temp_dir().join("vg_cull_fxc");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("vg_cull.hlsl");
    std::fs::write(&path, &hlsl).unwrap();
    let mut failures = Vec::new();
    for ep in &m.entry_points {
        let out = Command::new(&fxc)
            .args(["/nologo", "/T", "cs_5_1", "/E", &ep.name, "/Fo"])
            .arg(dir.join("out.bin"))
            .arg(&path)
            .output()
            .unwrap();
        if !out.status.success() {
            let err = String::from_utf8_lossy(&out.stderr);
            let errors: Vec<_> = err.lines().filter(|l| l.contains("error")).take(12).collect();
            failures.push(format!("{}:\n{}", ep.name, errors.join("\n")));
        }
    }
    assert!(failures.is_empty(), "FXC rejected:\n{}", failures.join("\n"));
}

#[cfg(windows)]
fn find_fxc() -> Option<std::path::PathBuf> {
    let root = std::path::Path::new(r"C:\Program Files (x86)\Windows Kits\10\bin");
    let mut versions: Vec<_> = std::fs::read_dir(root).ok()?.flatten().map(|e| e.path()).collect();
    versions.sort();
    versions.reverse();
    for v in versions {
        for arch in ["x64", "arm64"] {
            let p = v.join(arch).join("fxc.exe");
            if p.exists() {
                return Some(p);
            }
        }
    }
    None
}
