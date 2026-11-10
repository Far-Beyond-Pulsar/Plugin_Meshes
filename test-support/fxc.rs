//! Shared by the crates' `tests/fxc.rs`: compile a WGSL source's compute entry
//! points to HLSL (naga, SM 5.1) and then with the Windows SDK's `fxc`.
//!
//! Windows on ARM has no DXC, so wgpu compiles through FXC there, and FXC
//! rejects a barrier in flow control it cannot prove uniform (X4026/X3663).
//! Include with `#[path = "../../../test-support/fxc.rs"] mod fxc;`.

#![allow(dead_code)]

use std::process::Command;

pub fn find_fxc() -> Option<std::path::PathBuf> {
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

pub fn fxc_failures(fxc: &std::path::Path, name: &str, src: &str) -> Vec<String> {
    let m = match naga::front::wgsl::parse_str(src) {
        Ok(m) => m,
        Err(e) => return vec![format!("{name}: WGSL parse failed: {e:?}")],
    };
    let Ok(info) = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&m) else {
        return vec![format!("{name}: naga validation failed")];
    };
    let options = naga::back::hlsl::Options {
        shader_model: naga::back::hlsl::ShaderModel::V5_1,
        fake_missing_bindings: true,
        ..Default::default()
    };
    let mut hlsl = String::new();
    let pipeline = Default::default();
    let mut writer = naga::back::hlsl::Writer::new(&mut hlsl, &options, &pipeline);
    if let Err(e) = writer.write(&m, &info, None) {
        // Features HLSL SM 5.1 cannot express (64-bit atomics, ray queries) make
        // wgpu use DXC for that shader, so FXC never sees it.
        eprintln!("SKIP {name}: not expressible in SM 5.1 ({e:?})");
        return vec![];
    }
    let dir = std::env::temp_dir().join("helio_fxc");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{name}.hlsl"));
    std::fs::write(&path, &hlsl).unwrap();
    let mut failures = Vec::new();
    for ep in m.entry_points.iter().filter(|e| e.stage == naga::ShaderStage::Compute) {
        let out = Command::new(fxc)
            .args(["/nologo", "/T", "cs_5_1", "/E", &ep.name, "/Fo"])
            .arg(dir.join("out.bin"))
            .arg(&path)
            .output()
            .unwrap();
        if !out.status.success() {
            let err = String::from_utf8_lossy(&out.stderr);
            let errors: Vec<_> = err.lines().filter(|l| l.contains("error")).take(8).collect();
            failures.push(format!("{name}::{}:\n{}", ep.name, errors.join("\n")));
        }
    }
    failures
}
