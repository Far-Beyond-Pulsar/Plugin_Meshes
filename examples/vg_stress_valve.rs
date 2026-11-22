//! The overdraw valve on capture shots (#1642).
//!
//! Over the last [`WINDOW`] settle frames of every still single-image shot the
//! capture reads the cull's tile block back (`VgCullEarlyPass::watch_tile_valve`,
//! asynchronous, so the frames stay pipelined) and [`summarize`] writes the
//! `valve` section of the shot's stats JSON. `vg_stress --capture-check-valve
//! <dir>` ([`check_dir`]) reads those sections back, without a window or GPU,
//! and fails when the valve regresses:
//!
//! * a tile's scale exceeds `TILE_SCALE_MAX`;
//! * a still shot oscillates: a tile's scale reverses direction inside the
//!   window (the valve holds inside its dead band, so a reversal means the
//!   density crossed the whole band and came back);
//! * after the valve, a tile's density is over the band's upper edge
//!   (`limit * (1 + TILE_BAND)`) while its scale is below the cap: the valve
//!   should still be coarsening it, and on a settled still camera it has had
//!   every frame it needs (1x to the cap is `ln(TILE_SCALE_MAX) /
//!   ln(TILE_STEP_UP)` = 19 frames).
//!
//! The report also shows each shot's dropped clusters (`vg.overflow.visible`):
//! a frame with drops is not coverage-complete and its estimate varies with
//! which clusters were dropped.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use helio_pass_vg_cull::{TileValveSample, TILE_BAND, TILE_SCALE_MAX};

/// Settle frames at the end of a still shot whose tile block is read back.
/// Longer than one limit cycle of the valve (up 1.2x, down 0.95x per frame: a
/// few frames up, several down), so a flapping tile reverses inside it.
pub const WINDOW: u32 = 32;

/// Relative change below which two scales are the same (f32 rounding).
const SAME: f32 = 1e-5;

fn upper_band(limit: f32) -> f32 {
    limit * (1.0 + TILE_BAND)
}

fn below_cap(scale: f32) -> bool {
    scale < TILE_SCALE_MAX * (1.0 - SAME)
}

fn percentile(sorted: &[f32], p: f64) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    sorted[((sorted.len() as f64 - 1.0) * p).round() as usize]
}

/// The `valve` section of a shot's stats: `samples` are the watched frames
/// (oldest first), `limit` the shot's `tile_density_limit`, `still` whether the
/// camera and the scene stood still for the window.
pub fn summarize(samples: &[TileValveSample], limit: f32, still: bool) -> serde_json::Value {
    let consecutive = samples.windows(2).all(|w| w[1].frame == w[0].frame + 1)
        && samples.windows(2).all(|w| w[1].tiles == w[0].tiles);
    let Some(last) = samples.last() else {
        return serde_json::json!({
            "limit": limit, "band": TILE_BAND, "scale_cap": TILE_SCALE_MAX, "still": still,
            "window": WINDOW, "frames": 0, "consecutive": false,
        });
    };
    // The last frame: where the valve settled.
    let mut scales: Vec<f32> = last.valve.iter().map(|t| t.scale).collect();
    scales.sort_by(f32::total_cmp);
    let mut densities: Vec<f32> = last
        .valve
        .iter()
        .filter(|t| t.triangles > 0.0)
        .map(|t| t.density())
        .collect();
    densities.sort_by(f32::total_cmp);
    let tiles_scaled = last.valve.iter().filter(|t| t.scale > 1.0 + SAME).count();
    let tiles_at_cap = last.valve.iter().filter(|t| !below_cap(t.scale)).count();
    let tiles_parked = last.valve.iter().filter(|t| t.park > 0.0).count();
    let open = |t: &&helio_pass_vg_cull::TileValve| below_cap(t.scale);
    let worst_below_cap = last
        .valve
        .iter()
        .enumerate()
        .filter(|(_, t)| open(t))
        .max_by(|a, b| a.1.density().total_cmp(&b.1.density()));
    let over_band_below_cap = if limit > 0.0 {
        last.valve
            .iter()
            .filter(|t| open(t) && t.density() > upper_band(limit))
            .count()
    } else {
        0
    };
    // The window: how the scales moved.
    let scale_max_window = samples
        .iter()
        .flat_map(|s| s.valve.iter().map(|t| t.scale))
        .fold(1.0f32, f32::max);
    let tiles = last.valve.len();
    let (mut max_step, mut moving, mut reversing, mut max_reversals) = (1.0f32, 0, 0, 0u32);
    let mut flapping = 0;
    let mut worst_flapping = None;
    if consecutive {
        for t in 0..tiles {
            let (mut dir, mut reversals, mut moved) = (0i8, 0u32, false);
            for w in samples.windows(2) {
                let (a, b) = (w[0].valve[t].scale, w[1].valve[t].scale);
                max_step = max_step.max(b.max(a) / a.min(b));
                let d = if b > a * (1.0 + SAME) {
                    1
                } else if b < a * (1.0 - SAME) {
                    -1
                } else {
                    0
                };
                if d != 0 {
                    moved = true;
                    if dir != 0 && d != dir {
                        reversals += 1;
                    }
                    dir = d;
                }
            }
            moving += usize::from(moved);
            reversing += usize::from(reversals > 0);
            // A cycle reverses at least twice (up, down, up). One reversal is the
            // valve settling: an overshoot past a cliff and the park, or its
            // answer to a content change (a page arriving on a still camera).
            flapping += usize::from(reversals >= 2);
            if reversals > max_reversals {
                worst_flapping = Some(t);
            }
            max_reversals = max_reversals.max(reversals);
        }
    }
    // The tile that reversed most, frame by frame: what to look at when it fails.
    let worst_flapping = worst_flapping.map(|t| {
        let series = |pick: fn(&helio_pass_vg_cull::TileValve) -> serde_json::Value| {
            samples
                .iter()
                .map(|s| pick(&s.valve[t]))
                .collect::<Vec<_>>()
        };
        serde_json::json!({
            "tile": [t as u32 % last.tiles[0], t as u32 / last.tiles[0].max(1)],
            "scale": series(|v| v.scale.into()),
            "density": series(|v| v.density().into()),
            "triangles": series(|v| v.triangles.into()),
            "covered_px": series(|v| v.covered_px.into()),
        })
    });
    serde_json::json!({
        "limit": limit,
        "band": TILE_BAND,
        "scale_cap": TILE_SCALE_MAX,
        "still": still,
        "window": WINDOW,
        "frames": samples.len(),
        "consecutive": consecutive,
        "tiles": last.tiles,
        "last": {
            "frame": last.frame,
            "scale_max": scales.last().copied().unwrap_or(1.0),
            "scale_p99": percentile(&scales, 0.99),
            "tiles_scaled": tiles_scaled,
            "tiles_at_cap": tiles_at_cap,
            "tiles_covered": densities.len(),
            "tiles_parked": tiles_parked,
            "density_max": densities.last().copied().unwrap_or(0.0),
            "density_p99": percentile(&densities, 0.99),
            "density_max_below_cap": worst_below_cap.map_or(0.0, |(_, t)| t.density()),
            "density_max_below_cap_tile": worst_below_cap.map(|(i, t)| serde_json::json!({
                "tile": [i as u32 % last.tiles[0], i as u32 / last.tiles[0].max(1)],
                "scale": t.scale, "triangles": t.triangles, "covered_px": t.covered_px,
            })),
            "tiles_over_band_below_cap": over_band_below_cap,
        },
        "scale_max_window": scale_max_window,
        "max_step_ratio": max_step,
        "tiles_moving": moving,
        "tiles_reversing": reversing,
        "tiles_flapping": flapping,
        "max_reversals": max_reversals,
        "worst_flapping_tile": worst_flapping,
        // The whole frame: the estimate summed over the tiles, per watched frame.
        "estimated_triangles": samples
            .iter()
            .map(|s| s.valve.iter().map(|t| f64::from(t.triangles)).sum::<f64>().round())
            .collect::<Vec<_>>(),
    })
}

/// The ways a shot's valve section fails the check; empty when it passes.
pub fn failures(valve: &serde_json::Value) -> Vec<String> {
    let f = |p: &str| valve.pointer(p).and_then(serde_json::Value::as_f64);
    let u = |p: &str| {
        valve
            .pointer(p)
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0)
    };
    let mut out = Vec::new();
    let frames = u("/frames");
    if frames < 2 || valve["consecutive"] != serde_json::Value::Bool(true) {
        out.push(format!(
            "only {frames} tile-valve samples, or not of consecutive frames: nothing to judge"
        ));
        return out;
    }
    let cap = f64::from(TILE_SCALE_MAX) * (1.0 + f64::from(SAME));
    let scale_max = f("/scale_max_window").unwrap_or(0.0);
    if scale_max > cap {
        out.push(format!(
            "a tile scale reached {scale_max:.3}, over the cap {TILE_SCALE_MAX}"
        ));
    }
    if valve["still"] == serde_json::Value::Bool(true) && u("/tiles_flapping") > 0 {
        out.push(format!(
            "still camera, but {} tiles keep reversing their scale in the last {frames} frames (up to {} times, largest step x{:.3}): the valve oscillates",
            u("/tiles_flapping"),
            u("/max_reversals"),
            f("/max_step_ratio").unwrap_or(1.0),
        ));
    }
    let limit = f("/limit").unwrap_or(0.0);
    let over = u("/last/tiles_over_band_below_cap");
    if limit > 0.0 && over > 0 {
        out.push(format!(
            "{over} tiles stay over {:.1} triangles per pixel (limit {limit} plus the band) below the scale cap; worst {:.2} at scale {:.3}: the valve stopped coarsening them",
            upper_band(limit as f32),
            f("/last/density_max_below_cap").unwrap_or(0.0),
            f("/last/density_max_below_cap_tile/scale").unwrap_or(0.0),
        ));
    }
    out
}

/// `--capture-check-valve <dir>`: the capture directory to check.
pub fn check_requested(args: &[String]) -> Option<PathBuf> {
    let at = args.iter().position(|a| a == "--capture-check-valve")?;
    args.get(at + 1).map(PathBuf::from)
}

/// The report of every shot's valve, and whether all passed. Shots without a
/// valve section (camera paths, shots captured without it) are listed as not
/// checked; a capture where no shot has one fails.
pub fn check_stats(stats: &BTreeMap<String, serde_json::Value>) -> (String, bool) {
    use std::fmt::Write;
    let mut out = String::new();
    let _ = writeln!(
        out,
        "{:<30} {:>6} {:>9} {:>9} {:>7} {:>7} {:>7} {:>7} {:>9} {:>9} {:>6} {:>8}  verdict",
        "shot",
        "limit",
        "scale max",
        "scale p99",
        "scaled",
        "at cap",
        "parked",
        "covered",
        "dens max",
        "below cap",
        "flaps",
        "dropped"
    );
    let (mut checked, mut passed) = (0, true);
    for (name, s) in stats {
        let v = &s["valve"];
        if v.is_null() {
            let _ = writeln!(out, "{name:<30} not checked (no valve section)");
            continue;
        }
        checked += 1;
        let f = |p: &str| {
            v.pointer(p)
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0)
        };
        let u = |p: &str| {
            v.pointer(p)
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0)
        };
        let fails = failures(v);
        let _ =
            writeln!(
            out,
            "{name:<30} {:>6} {:>9.3} {:>9.3} {:>7} {:>7} {:>7} {:>7} {:>9.2} {:>9.2} {:>6} {:>8}  {}",
            f("/limit"),
            f("/scale_max_window"),
            f("/last/scale_p99"),
            u("/last/tiles_scaled"),
            u("/last/tiles_at_cap"),
            u("/last/tiles_parked"),
            u("/last/tiles_covered"),
            f("/last/density_max"),
            f("/last/density_max_below_cap"),
            u("/tiles_flapping"),
            s.pointer("/vg/overflow/visible").and_then(serde_json::Value::as_u64).unwrap_or(0),
            if fails.is_empty() { "ok" } else { "FAIL" },
        );
        for fail in &fails {
            let _ = writeln!(out, "    {name}: {fail}");
        }
        passed &= fails.is_empty();
    }
    if checked == 0 {
        let _ = writeln!(out, "no shot has a valve section: nothing was checked");
        passed = false;
    }
    (out, passed)
}

/// Check a capture directory's stats.
pub fn check_dir(dir: &Path) -> Result<(String, bool), String> {
    Ok(check_stats(&super::vg_stress_capture::read_stats(dir)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use helio_pass_vg_cull::TileValve;

    const LIMIT: f32 = 16.0;

    fn tile(scale: f32, density: f32) -> TileValve {
        TileValve {
            triangles: density * 64.0,
            covered_px: 64.0,
            scale,
            ..TileValve::default()
        }
    }

    /// `frames` samples of four tiles, consecutive from frame 100, from `at`.
    fn window(frames: u32, at: impl Fn(u32, usize) -> TileValve) -> Vec<TileValveSample> {
        (0..frames)
            .map(|f| TileValveSample {
                frame: 100 + u64::from(f),
                tiles: [2, 2],
                valve: (0..4).map(|t| at(f, t)).collect(),
            })
            .collect()
    }

    fn settled() -> Vec<TileValveSample> {
        // A cold tile, one held in the band, one at the cap over the limit, one
        // still relaxing (monotone) under it.
        window(WINDOW, |f, t| match t {
            0 => tile(1.0, 2.0),
            1 => tile(3.0, LIMIT * 1.1),
            2 => tile(TILE_SCALE_MAX, 40.0),
            _ => tile(8.0 * 0.95f32.powi(f as i32), 5.0),
        })
    }

    fn stats_of(valve: serde_json::Value) -> BTreeMap<String, serde_json::Value> {
        BTreeMap::from([("shot".to_string(), serde_json::json!({ "valve": valve }))])
    }

    #[test]
    fn a_settled_valve_passes() {
        let v = summarize(&settled(), LIMIT, true);
        assert_eq!(failures(&v), Vec::<String>::new(), "{v:#}");
        assert_eq!(v["last"]["tiles_at_cap"], 1);
        assert_eq!(v["last"]["tiles_scaled"], 3);
        assert_eq!(v["tiles_moving"], 1);
        assert_eq!(v["tiles_reversing"], 0);
        assert_eq!(v["last"]["density_max"], 40.0);
        let (report, ok) = check_stats(&stats_of(v));
        assert!(ok, "{report}");
    }

    #[test]
    fn a_scale_over_the_cap_fails() {
        let mut s = settled();
        for f in &mut s {
            f.valve[2].scale = TILE_SCALE_MAX * 1.2;
        }
        let fails = failures(&summarize(&s, LIMIT, true));
        assert!(
            fails.len() == 1 && fails[0].contains("over the cap"),
            "{fails:?}"
        );
    }

    #[test]
    fn a_flapping_tile_fails_on_a_still_camera_only() {
        // Up for 2 frames, down for 4: a limit cycle.
        let flap = |f: u32| {
            if f % 6 < 2 {
                1.2f32.powi((f % 6) as i32 + 1)
            } else {
                1.44 * 0.95f32.powi((f % 6) as i32 - 1)
            }
        };
        let s = window(WINDOW, |f, t| {
            if t == 1 {
                tile(2.0 * flap(f), LIMIT)
            } else {
                tile(1.0, 1.0)
            }
        });
        let v = summarize(&s, LIMIT, true);
        let fails = failures(&v);
        assert!(
            fails.len() == 1 && fails[0].contains("oscillates"),
            "{fails:?}"
        );
        // The report shows the flapping tile frame by frame.
        let worst = &v["worst_flapping_tile"];
        assert_eq!(worst["tile"], serde_json::json!([1, 0]));
        assert_eq!(
            worst["scale"].as_array().map(Vec::len),
            Some(WINDOW as usize)
        );
        // A moving scene may move the scales both ways.
        assert!(failures(&summarize(&s, LIMIT, false)).is_empty());
    }

    #[test]
    fn one_reversal_that_ends_in_a_hold_is_settling() {
        // Relaxing, over a cliff, one coarsening step back, then parked: the
        // valve settling, not cycling.
        let settle = |at: u32| {
            window(WINDOW, move |f, t| {
                let scale = if t != 1 {
                    1.0
                } else if f < at {
                    4.0 * 0.95f32.powi(f as i32)
                } else {
                    4.0 * 0.95f32.powi(at as i32 - 1) * 1.2
                };
                tile(scale, 11.0)
            })
        };
        let v = summarize(&settle(8), LIMIT, true);
        assert_eq!(v["tiles_reversing"], 1);
        assert_eq!(v["tiles_flapping"], 0);
        assert!(failures(&v).is_empty(), "{:?}", failures(&v));
        // So is the same reversal at the window's end.
        let v = summarize(&settle(WINDOW - 3), LIMIT, true);
        assert_eq!(v["tiles_flapping"], 0);
        // A second reversal is a cycle.
        let mut s = settle(8);
        for f in &mut s[20..] {
            f.valve[1].scale *= 0.95;
        }
        let v = summarize(&s, LIMIT, true);
        assert_eq!(v["tiles_flapping"], 1);
        assert!(!failures(&v).is_empty());
    }

    #[test]
    fn density_over_the_band_below_the_cap_fails() {
        let mut s = settled();
        let hot = TileValve {
            scale: 6.0,
            ..tile(6.0, LIMIT * 2.0)
        };
        for f in &mut s {
            f.valve[1] = hot;
        }
        let v = summarize(&s, LIMIT, true);
        let fails = failures(&v);
        assert!(
            fails.len() == 1 && fails[0].contains("stopped coarsening"),
            "{fails:?}"
        );
        assert_eq!(v["last"]["density_max_below_cap"], LIMIT * 2.0);
        assert_eq!(
            v["last"]["density_max_below_cap_tile"]["tile"],
            serde_json::json!([1, 0])
        );
        // Within the band it holds: fine.
        for f in &mut s {
            f.valve[1] = tile(6.0, LIMIT * (1.0 + TILE_BAND) * 0.99);
        }
        assert!(failures(&summarize(&s, LIMIT, true)).is_empty());
        // The valve off: nothing to hold.
        for f in &mut s {
            f.valve[1] = TileValve { scale: 1.0, ..hot };
        }
        assert!(failures(&summarize(&s, 0.0, true)).is_empty());
    }

    #[test]
    fn missing_or_gapped_samples_fail_and_an_unchecked_capture_fails() {
        assert!(!failures(&summarize(&[], LIMIT, true)).is_empty());
        let mut s = settled();
        s.remove(5);
        let fails = failures(&summarize(&s, LIMIT, true));
        assert!(
            fails.len() == 1 && fails[0].contains("consecutive"),
            "{fails:?}"
        );
        let (report, ok) = check_stats(&BTreeMap::from([(
            "path-shot".to_string(),
            serde_json::json!({ "valve": null }),
        )]));
        assert!(!ok && report.contains("nothing was checked"), "{report}");
    }

    #[test]
    fn the_check_argument_names_a_directory() {
        let args = ["--capture-check-valve", "out"].map(String::from);
        assert_eq!(check_requested(&args), Some(PathBuf::from("out")));
        assert_eq!(check_requested(&args[..1]), None);
    }
}
