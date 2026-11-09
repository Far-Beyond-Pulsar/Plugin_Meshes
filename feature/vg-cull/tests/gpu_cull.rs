//! Headless GPU test: the compute cull on the synthetic DAG must select the
//! same (instance, cluster, bin) set as the CPU reference, including page
//! requests, frustum culling, two-phase deferral and bin overflow handling.
//! The synthetic DAG is written in the REAL page format and read through the
//! real `helio-vg-stream` bindings. Skips when no adapter is available.

mod common;

use std::collections::BTreeSet;

use common::{build, SynthDag};
use helio_core::{CommandRecorder, GpuCameraUniforms};
use helio_pass_vg_bvh::gpu_types::{GpuBvhHeader, GpuBvhNode, NODE_LEAF_BIT};
use helio_pass_vg_cull::reference::{choose_bin, select_clusters, RefInstance, RefView};
use helio_pass_vg_cull::{
    state, stats, VgCullConfig, VgCullCore, VgCullFrame, VgCullInputs, BIN_COUNT,
};
use helio_vg_format::*;
use helio_vg_stream::feedback::{self, FeedbackLayout};
use helio_vg_stream::{VgStreamBuffers, PAGE_NOT_RESIDENT};

const W: u32 = 1920;
const H: u32 = 1080;
const FOV: f32 = std::f32::consts::FRAC_PI_3; // 60 degrees

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

fn gpu() -> Option<Gpu> {
    let instance = wgpu::Instance::new(helio_core::wgpu_instance_descriptor());
    let mut adapter = None;
    for force_fallback_adapter in [false, true] {
        if let Ok(a) = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter,
            apply_limit_buckets: false,
        })) {
            adapter = Some(a);
            break;
        }
    }
    let adapter = adapter?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("vg-cull test"),
        required_features: wgpu::Features::empty(),
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .ok()?;
    if device.limits().max_storage_buffers_per_shader_stage < 20 {
        eprintln!("skipping: adapter has fewer than 20 storage buffers per stage");
        return None;
    }
    device.on_uncaptured_error(std::sync::Arc::new(|e| {
        panic!("GPU validation error: {e:?}")
    }));
    Some(Gpu { device, queue })
}

fn buf<T: bytemuck::Pod>(g: &Gpu, label: &str, data: &[T]) -> wgpu::Buffer {
    let bytes: &[u8] = bytemuck::cast_slice(data);
    let b = g.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (bytes.len() as u64).max(16).next_multiple_of(4),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    if !bytes.is_empty() {
        g.queue.write_buffer(&b, 0, bytes);
    }
    b
}

fn read_words(g: &Gpu, src: &wgpu::Buffer, words: u64) -> Vec<u32> {
    let staging = g.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: words * 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = g.device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(src, 0, &staging, 0, words * 4);
    g.queue.submit([enc.finish()]);
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, |r| r.unwrap());
    g.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let v = {
        let view = staging.slice(..).get_mapped_range().expect("mapped");
        bytemuck::cast_slice::<u8, u32>(&view).to_vec()
    };
    staging.unmap();
    v
}

/// A `StaticObjectRow` (63 words): the cull reads `transform` (words 4..20), the
/// instance BVH `mesh_generation` (1) and `bounds` (48..52).
fn object_row(m: [f32; 16], bounds: [f32; 4]) -> [u32; 63] {
    let mut w = [0u32; 63];
    w[1] = 1; // mesh_generation: live row (the instance BVH skips dead rows)
    for i in 0..4 {
        w[48 + i] = bounds[i].to_bits();
    }
    for i in 0..16 {
        w[4 + i] = m[i].to_bits();
        w[20 + i] = m[i].to_bits();
    }
    w
}

struct Scene {
    dag: SynthDag,
    resident: Vec<bool>,
    instances: Vec<RefInstance>,
}

struct Resources {
    instances: wgpu::Buffer,
    objects: wgpu::Buffer,
    pool: wgpu::Buffer,
    residency: wgpu::Buffer,
    nodes: wgpu::Buffer,
    group_refs: wgpu::Buffer,
    headers: wgpu::Buffer,
    bases: wgpu::Buffer,
    quant: wgpu::Buffer,
    feedback: wgpu::Buffer,
    layout: FeedbackLayout,
    bvh_nodes: wgpu::Buffer,
    bvh_header: wgpu::Buffer,
    bvh_items: wgpu::Buffer,
    camera: wgpu::Buffer,
    page_count: u32,
    instance_count: u32,
    /// The scene's `materials` table; `Some` turns on alpha-mask routing.
    materials: Option<wgpu::Buffer>,
}

impl Resources {
    fn inputs(&self) -> VgCullInputs<'_> {
        VgCullInputs {
            instances: &self.instances,
            objects: &self.objects,
            stream: VgStreamBuffers {
                page_pool: &self.pool,
                residency: &self.residency,
                nodes: &self.nodes,
                group_refs: &self.group_refs,
                asset_headers: &self.headers,
                asset_bases: &self.bases,
                asset_quant: &self.quant,
                feedback: &self.feedback,
            },
            gcbvh_nodes: &self.bvh_nodes,
            gcbvh_header: &self.bvh_header,
            gcbvh_items: &self.bvh_items,
            dense_mask: None,
            materials: self.materials.as_ref(),
            instance_count: self.instance_count,
            dense_tiles: [1, 1],
        }
    }

    /// Clear the feedback buffer (a new readback epoch).
    fn reset_feedback(&self, g: &Gpu) {
        g.queue.write_buffer(
            &self.feedback,
            0,
            bytemuck::cast_slice(&self.layout.initial_words()),
        );
    }
}

/// The synthetic DAG as real pages: pool slot == page id, each page holds
/// a `PageHeader` and the group's `GpuCluster` records.
fn pool_bytes(dag: &SynthDag) -> Vec<u8> {
    let mut pool = vec![0u8; dag.page_count() * PAGE_BYTES as usize];
    for (p, r) in dag.group_refs.iter().enumerate() {
        let base = p * PAGE_BYTES as usize;
        let hdr = PageHeader {
            magic: PAGE_MAGIC,
            version: FORMAT_VERSION,
            page_id: p as u32,
            cluster_count: r.cluster_count(),
            clusters_offset: 32,
            ..Default::default()
        };
        pool[base..base + 32].copy_from_slice(bytemuck::bytes_of(&hdr));
        for i in 0..r.cluster_count() {
            let c = dag.clusters[(dag.page_first_slot[p] + i) as usize];
            let at = base + 32 + i as usize * 64;
            pool[at..at + 64].copy_from_slice(bytemuck::bytes_of(&c));
        }
    }
    pool
}

fn upload(g: &Gpu, s: &Scene) -> Resources {
    let n = s.instances.len() as u32;
    let inst: Vec<GpuVgInstance> = s.instances.iter().map(|i| i.inst).collect();
    let mut rows = Vec::new();
    for (i, r) in s.instances.iter().enumerate() {
        // Rows are addressed through `object_row`; keep them in order.
        assert_eq!(r.inst.object_row as usize, i);
        let m = r.matrix;
        rows.extend_from_slice(&object_row(m, [m[12], m[13], m[14], 300.0 * r.scale()]));
    }
    let header = GpuAssetHeader {
        root_node: s.dag.root_node,
        node_count: s.dag.nodes.len() as u32,
        group_count: s.dag.group_refs.len() as u32,
        page_count: s.dag.page_count() as u32,
        cluster_count: s.dag.clusters.len() as u32,
        root_page_count: 1,
        triangle_count: 0,
        flags: 0,
        ..Default::default()
    };
    // As the production store uploads it: root leaves in the unread header words.
    let header = helio_vg_format::gpu_header_with_root_leaves(header, &s.dag.nodes);
    let residency: Vec<u32> = (0..s.dag.page_count())
        .map(|p| {
            if s.resident[p] {
                p as u32
            } else {
                PAGE_NOT_RESIDENT
            }
        })
        .collect();
    // One batch leaf covering every instance with a generous AABB.
    let bvh_node = GpuBvhNode {
        min: [-1.0e4; 3],
        a: NODE_LEAF_BIT | n,
        max: [1.0e4; 3],
        b: 0,
    };
    let bvh_header = GpuBvhHeader {
        root_node: 0,
        node_count: 1,
        leaf_count: 1,
        instance_count: n,
        ..GpuBvhHeader::empty()
    };
    let items: Vec<u32> = (0..n).collect();
    let layout = FeedbackLayout::new(4096, s.dag.page_count() as u32);
    let feedback = buf(g, "feedback", &layout.initial_words());
    Resources {
        instances: buf(g, "instances", &inst),
        objects: buf(g, "objects", &rows),
        pool: buf(g, "pool", &pool_bytes(&s.dag)),
        residency: buf(g, "residency", &residency),
        nodes: buf(g, "nodes", &s.dag.nodes),
        group_refs: buf(g, "group_refs", &s.dag.group_refs),
        headers: buf(g, "headers", &[header]),
        bases: buf(g, "bases", &[[0u32, 0, 0, 1]]),
        quant: buf(
            g,
            "quant",
            &[GpuAssetQuant {
                origin: [0.0; 3],
                step: 1.0,
            }],
        ),
        feedback,
        layout,
        bvh_nodes: buf(g, "bvh_nodes", &[bvh_node]),
        bvh_header: buf(g, "bvh_header", &[bvh_header]),
        bvh_items: buf(g, "bvh_items", &items),
        camera: g.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera"),
            size: 2 * std::mem::size_of::<GpuCameraUniforms>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }),
        page_count: s.dag.page_count() as u32,
        instance_count: n,
        materials: None,
    }
}

fn set_camera(g: &Gpu, r: &Resources, eye: [f32; 3]) -> RefView {
    let eye_v = glam::Vec3::from(eye);
    let view = glam::camera::rh::view::look_at_mat4(
        eye_v,
        glam::Vec3::new(eye[0], 0.0, 0.0),
        glam::Vec3::Y,
    );
    let proj = helio_core::depth::perspective(FOV, W as f32 / H as f32, 0.1);
    let cam = GpuCameraUniforms::new(view, proj, eye_v, 0.1, 1.0e6, 0, [0.0, 0.0], proj * view);
    g.queue
        .write_buffer(&r.camera, 0, bytemuck::cast_slice(&[cam, cam]));
    RefView {
        view: view.to_cols_array(),
        proj_y: proj.y_axis.y,
        near: 0.1,
        screen_height: H as f32,
    }
}

type Item = (u32, u32, u32); // instance, cluster slot, bin

struct PhaseOut {
    items: BTreeSet<Item>,
    /// (first, count) per bin.
    ranges: Vec<(u32, u32)>,
    state: Vec<u32>,
}

fn collect(g: &Gpu, core: &VgCullCore, res: &Resources, phase: usize) -> (PhaseOut, Vec<u32>) {
    let b = core.buffers();
    let st = read_words(g, &b.state, state::WORDS as u64);
    let vis = read_words(g, &b.visible, BIN_COUNT as u64 * b.bin_capacity as u64 * 2);
    let mut items = BTreeSet::new();
    let mut ranges = Vec::new();
    for bin in 0..BIN_COUNT as usize {
        let o = state::RANGES as usize + phase * BIN_COUNT as usize * 2 + bin * 2;
        let (first, count) = (st[o], st[o + 1]);
        ranges.push((first, count));
        for k in 0..count {
            let e = (first + k) as usize * 2;
            assert!(
                items.insert((vis[e], vis[e + 1], bin as u32)),
                "duplicate (instance, cluster) in bin {bin}"
            );
        }
    }
    let fb = feedback::decode(
        &read_words(g, &res.feedback, res.layout.total_words() as u64),
        res.layout,
        res.page_count,
    );
    let req: Vec<u32> = fb.requests;
    (
        PhaseOut {
            items,
            ranges,
            state: st,
        },
        req,
    )
}

fn expected(s: &Scene, v: &RefView, tau: f32, mode: VgMode) -> (BTreeSet<Item>, BTreeSet<u32>) {
    let mut items = BTreeSet::new();
    let mut pages = BTreeSet::new();
    let slots = s.dag.slots(&s.resident);
    let lookup = s.dag.lookup();
    let asset = s.dag.asset(&slots, &lookup);
    for (i, inst) in s.instances.iter().enumerate() {
        if inst.matrix[12].abs() > 5000.0 {
            continue; // outside the frustum: culled on the GPU
        }
        let sel = select_clusters(&asset, v, tau, inst);
        for slot in sel.drawn {
            let bin = choose_bin(
                v,
                inst,
                &s.dag.clusters[s.dag.index_of(slot)],
                mode,
                2.0,
                false,
            );
            items.insert((i as u32, slot, bin));
        }
        pages.extend(sel.requested_pages);
    }
    (items, pages)
}

fn scene(resident_root_only: bool) -> Scene {
    let dag = build(64, true);
    let resident = if resident_root_only {
        dag.close_upward(
            &(0..dag.page_count())
                .map(|p| p == dag.root_group())
                .collect::<Vec<_>>(),
        )
    } else {
        vec![true; dag.page_count()]
    };
    let flags = INST_ENABLED;
    let mk = |t: [f32; 3], s: f32, lod: f32, row: u32| {
        RefInstance::translated_scaled(
            t,
            s,
            GpuVgInstance {
                asset_slot: 0,
                flags,
                lod_error_scale: lod,
                object_row: row,
            },
        )
    };
    Scene {
        dag,
        resident,
        instances: vec![
            mk([-150.0, 0.0, 0.0], 1.0, 1.0, 0),
            mk([0.0, 0.0, 0.0], 2.0, 0.5, 1),
            // Far outside the frustum: must contribute nothing.
            mk([20000.0, 0.0, 0.0], 1.0, 1.0, 2),
        ],
    }
}

fn run_phase(g: &Gpu, core: &mut VgCullCore, r: &Resources, phase: u32) {
    let mut enc = g.device.create_command_encoder(&Default::default());
    {
        let mut rec = CommandRecorder::from_encoder(&mut enc);
        core.record_phase(
            &mut rec,
            &g.device,
            &g.queue,
            &r.camera,
            &r.inputs(),
            None,
            phase,
        );
    }
    g.queue.submit([enc.finish()]);
}

fn begin(g: &Gpu, core: &mut VgCullCore, r: &Resources) {
    core.begin_frame(
        &g.device,
        &g.queue,
        &VgCullFrame {
            width: W,
            height: H,
        },
        &r.inputs(),
    );
}

/// Run one frame with the camera far away from everything, so the hysteresis
/// memory (last frame's refined groups, SW/HW classes) is empty for the next one:
/// the tests that compare against the stateless reference start from it.
fn flush_history(g: &Gpu, core: &mut VgCullCore, r: &Resources) {
    set_camera(g, r, [1.0e7, 0.0, 300.0]);
    begin(g, core, r);
    run_phase(g, core, r, 0);
    if core.config.two_phase_occlusion {
        run_phase(g, core, r, 1);
    }
}

fn check_args(out: &PhaseOut, phase: usize, bin_capacity: u32) {
    let a = state::ARGS as usize + phase * state::ARGS_PHASE_WORDS as usize;
    for b in 0..4usize {
        let (first, count) = out.ranges[b];
        assert_eq!(
            &out.state[a + b * 4..a + b * 4 + 4],
            &[96 * (b as u32 + 1), count, 0, first],
            "hw draw args bin {b}"
        );
        assert_eq!(
            first / bin_capacity,
            b as u32,
            "bin {b} range lies in its region"
        );
    }
    for (b, o) in [(4usize, a + 16), (5usize, a + 20)] {
        let n = out.ranges[b].1;
        let w = helio_pass_vg_cull::SW_GROUP_WIDTH;
        assert_eq!(
            &out.state[o..o + 3],
            &[n.min(w), n.div_ceil(w), 1],
            "dispatch args bin {b}"
        );
    }
}

fn triangle_count(s: &Scene, items: &BTreeSet<Item>) -> u32 {
    items
        .iter()
        .map(|&(_, slot, _)| s.dag.clusters[s.dag.index_of(slot)].counts_material & 0xff)
        .sum()
}

#[test]
fn distance_reduces_gpu_triangles_and_history_does_not_pin_fine_lod() {
    let Some(g) = gpu() else {
        return;
    };
    // Exercise both traversal paths, and move back toward the object after
    // retreating. Scaling object and camera together must preserve the cut.
    for two_phase in [false, true] {
        let mut baseline = Vec::new();
        for scale in [1.0, 0.01, 10.0] {
            let mut s = scene(false);
            s.instances = vec![RefInstance::translated_scaled(
                [-63.0 * scale, 0.0, 0.0],
                scale,
                GpuVgInstance {
                    asset_slot: 0,
                    flags: INST_ENABLED,
                    lod_error_scale: 1.0,
                    object_row: 0,
                },
            )];
            let r = upload(&g, &s);
            let mut core = VgCullCore::new(
                &g.device,
                VgCullConfig {
                    two_phase_occlusion: two_phase,
                    initial_bin_capacity: 4096,
                    tau_px: 1.0,
                    ..Default::default()
                },
            );
            let mut counts = Vec::new();
            for distance in [300.0, 1000.0, 3000.0, 10000.0, 100000.0, 300.0] {
                flush_history(&g, &mut core, &r);
                let v = set_camera(&g, &r, [0.0, 0.0, distance * scale]);
                begin(&g, &mut core, &r);
                run_phase(&g, &mut core, &r, 0);
                let (early, _) = collect(&g, &core, &r, 0);
                let mut items = early.items;
                let state = if two_phase {
                    run_phase(&g, &mut core, &r, 1);
                    let (late, _) = collect(&g, &core, &r, 1);
                    assert!(
                        items.is_disjoint(&late.items),
                        "phases must not double draw"
                    );
                    items.extend(late.items);
                    late.state
                } else {
                    early.state
                };
                let (want, _) = expected(&s, &v, 1.0, VgMode::Auto);
                assert_eq!(
                    items, want,
                    "distance={distance} scale={scale} two_phase={two_phase}"
                );
                let triangles = triangle_count(&s, &items);
                assert_eq!(
                    state[(state::STATS + stats::TRIANGLES) as usize],
                    triangles,
                    "title counter must equal emitted triangles across both phases"
                );
                assert_eq!(state[(state::STATS + stats::VISIBLE_OVERFLOW) as usize], 0);
                assert!(triangles > 0, "distance must simplify, not lose the object");
                counts.push(triangles);
            }
            assert!(counts[..5].windows(2).all(|w| w[1] <= w[0]), "{counts:?}");
            assert!(
                counts[4] * 8 < counts[0],
                "distant geometry must be substantially cheaper: {counts:?}"
            );
            assert_eq!(counts[0], counts[5], "returning near restores fine detail");
            if baseline.is_empty() {
                baseline = counts.clone();
            }
            assert_eq!(
                counts, baseline,
                "world units must not change projected LOD"
            );
            eprintln!("GPU distance sweep: two_phase={two_phase} scale={scale}: {counts:?}");

            // Move the camera sideways, preserving its direction, until every
            // object/cluster is outside the frustum. Counters must reset too.
            set_camera(&g, &r, [1.0e6 * scale, 0.0, 300.0 * scale]);
            begin(&g, &mut core, &r);
            run_phase(&g, &mut core, &r, 0);
            let (early, _) = collect(&g, &core, &r, 0);
            assert!(early.items.is_empty());
            if two_phase {
                run_phase(&g, &mut core, &r, 1);
            }
            let (last, _) = collect(&g, &core, &r, u32::from(two_phase) as usize);
            assert!(last.items.is_empty());
            assert_eq!(last.state[(state::STATS + stats::TRIANGLES) as usize], 0);
        }
    }
}

#[test]
fn single_phase_matches_the_reference_over_taus_and_residencies() {
    let Some(g) = gpu() else {
        eprintln!("skipping vg-cull GPU test: no adapter");
        return;
    };
    for root_only in [false, true] {
        let s = scene(root_only);
        let r = upload(&g, &s);
        let mut core = VgCullCore::new(
            &g.device,
            VgCullConfig {
                two_phase_occlusion: false,
                initial_bin_capacity: 4096,
                // The reference is the stateless cut: no overdraw valve.
                tile_density_limit: 0.0,
                ..Default::default()
            },
        );
        for dist in [300.0f32, 700.0] {
            let v = set_camera(&g, &r, [31.0, 0.0, dist]);
            for tau in [0.01f32, 0.5, 2.0, 6.0, 20.0, 100.0, 1.0e5] {
                core.config.tau_px = tau;
                flush_history(&g, &mut core, &r);
                set_camera(&g, &r, [31.0, 0.0, dist]);
                r.reset_feedback(&g);
                begin(&g, &mut core, &r);
                run_phase(&g, &mut core, &r, 0);
                let (out, req) = collect(&g, &core, &r, 0);
                let (want, want_pages) = expected(&s, &v, tau, VgMode::Auto);
                assert_eq!(
                    out.items, want,
                    "root_only={root_only} dist={dist} tau={tau}: GPU != reference"
                );
                check_args(&out, 0, core.buffers().bin_capacity);
                let got_pages: BTreeSet<u32> = req.iter().copied().collect();
                assert_eq!(got_pages.len(), req.len(), "requests are deduplicated");
                assert_eq!(got_pages, want_pages, "page requests, tau={tau}");
                let stat = |i: u32| out.state[(state::STATS + i) as usize];
                assert_eq!(stat(stats::JOB_OVERFLOW), 0);
                assert_eq!(stat(stats::VISIBLE_OVERFLOW), 0);
                assert_eq!(stat(stats::EMITTED) as usize, want.len());
            }
        }
    }
}

#[test]
fn modes_and_tier_locks_route_to_the_right_bins() {
    let Some(g) = gpu() else {
        return;
    };
    let s = scene(false);
    let r = upload(&g, &s);
    let mut core = VgCullCore::new(
        &g.device,
        VgCullConfig {
            two_phase_occlusion: false,
            initial_bin_capacity: 4096,
            ..Default::default()
        },
    );
    let v = set_camera(&g, &r, [31.0, 0.0, 400.0]);
    for mode in [
        VgMode::Auto,
        VgMode::ForceNonRtFallback,
        VgMode::ForceHwRaster,
    ] {
        core.config.mode = mode;
        core.config.tau_px = 2.0;
        begin(&g, &mut core, &r);
        run_phase(&g, &mut core, &r, 0);
        let (out, _) = collect(&g, &core, &r, 0);
        let (want, _) = expected(&s, &v, 2.0, mode);
        assert_eq!(out.items, want, "{mode:?}");
        if mode == VgMode::ForceHwRaster {
            assert!(
                out.items.iter().all(|i| i.2 < 4),
                "ForceHwRaster leaves only hw bins"
            );
        }
    }
    // Off: nothing at all.
    core.config.mode = VgMode::Off;
    begin(&g, &mut core, &r);
    run_phase(&g, &mut core, &r, 0);
    let (out, _) = collect(&g, &core, &r, 0);
    assert!(out.items.is_empty());
}

#[test]
fn two_phase_defers_new_work_then_replays_history() {
    let Some(g) = gpu() else {
        return;
    };
    let s = scene(false);
    let r = upload(&g, &s);
    let mut core = VgCullCore::new(
        &g.device,
        VgCullConfig {
            initial_bin_capacity: 4096,
            tau_px: 2.0,
            ..Default::default()
        },
    );
    let v = set_camera(&g, &r, [31.0, 0.0, 400.0]);
    let (want, _) = expected(&s, &v, 2.0, VgMode::Auto);
    assert!(!want.is_empty());

    // Frame 1: nothing was visible before, so phase 0 draws nothing and phase
    // 1 (no Hi-Z hook wired: no occlusion) draws everything.
    begin(&g, &mut core, &r);
    run_phase(&g, &mut core, &r, 0);
    let (p0, _) = collect(&g, &core, &r, 0);
    assert!(p0.items.is_empty(), "phase 0 of a cold frame draws nothing");
    run_phase(&g, &mut core, &r, 1);
    let (p1, _) = collect(&g, &core, &r, 1);
    assert_eq!(p1.items, want);
    check_args(&p1, 1, core.buffers().bin_capacity);
    assert!(p1.state[(state::STATS + stats::DEFERRED) as usize] > 0);

    // Frame 2: the same set is last frame's visible set, so phase 0 redraws
    // it all and phase 1 has nothing left.
    begin(&g, &mut core, &r);
    run_phase(&g, &mut core, &r, 0);
    let (p0, _) = collect(&g, &core, &r, 0);
    assert_eq!(p0.items, want, "phase 0 replays the previous visible set");
    run_phase(&g, &mut core, &r, 1);
    let (p1, _) = collect(&g, &core, &r, 1);
    assert!(p1.items.is_empty(), "phase 1 draws only the delta");
    // Phase-1 ranges start after phase 0's.
    for b in 0..BIN_COUNT as usize {
        assert_eq!(p1.ranges[b].0, p0.ranges[b].0 + p0.ranges[b].1);
    }
}

#[test]
fn bin_overflow_is_contained_and_capacity_grows() {
    let Some(g) = gpu() else {
        return;
    };
    let s = scene(false);
    let r = upload(&g, &s);
    let mut core = VgCullCore::new(
        &g.device,
        VgCullConfig {
            two_phase_occlusion: false,
            initial_bin_capacity: 16,
            tau_px: 0.01,
            ..Default::default()
        },
    );
    let v = set_camera(&g, &r, [31.0, 0.0, 400.0]);
    let (want, _) = expected(&s, &v, 0.01, VgMode::Auto);
    let start_cap = core.buffers().bin_capacity;
    assert_eq!(start_cap, 16);
    begin(&g, &mut core, &r);
    run_phase(&g, &mut core, &r, 0);
    let (out, _) = collect(&g, &core, &r, 0);
    assert!(out.state[(state::STATS + stats::VISIBLE_OVERFLOW) as usize] > 0);
    assert!(out.items.len() < want.len());
    assert!(out.ranges.iter().all(|r| r.1 <= start_cap));
    // The stats read-back (mapped on the next frame, delivered the one after)
    // grows the bins; then the full set fits.
    for _ in 0..3 {
        begin(&g, &mut core, &r);
        g.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        run_phase(&g, &mut core, &r, 0);
    }
    begin(&g, &mut core, &r);
    assert!(core.buffers().bin_capacity > start_cap, "bins grew lazily");
    run_phase(&g, &mut core, &r, 0);
    let (out, _) = collect(&g, &core, &r, 0);
    assert_eq!(out.items, want);
}

fn hiz_texture(g: &Gpu, value: f32) -> wgpu::TextureView {
    let mips = helio_pass_vg_cull_mips(W, H);
    let tex = g.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("test hiz"),
        size: wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
        mip_level_count: mips,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for m in 0..mips {
        let (w, h) = ((W >> m).max(1), (H >> m).max(1));
        let data = vec![value; (w * h) as usize];
        g.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: m,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
    }
    tex.create_view(&Default::default())
}

fn helio_pass_vg_cull_mips(w: u32, h: u32) -> u32 {
    (u32::BITS - w.max(h).leading_zeros()).max(1)
}

#[test]
fn late_phase_hiz_occlusion_drops_hidden_work_and_keeps_visible_work() {
    let Some(g) = gpu() else {
        return;
    };
    let s = scene(false);
    let r = upload(&g, &s);
    let v = set_camera(&g, &r, [31.0, 0.0, 400.0]);
    let (want, _) = expected(&s, &v, 2.0, VgMode::Auto);
    for (value, occluded) in [(1.0f32, true), (0.0, false)] {
        let hiz = hiz_texture(&g, value);
        let mut core = VgCullCore::new(
            &g.device,
            VgCullConfig {
                initial_bin_capacity: 4096,
                tau_px: 2.0,
                ..Default::default()
            },
        );
        begin(&g, &mut core, &r);
        run_phase(&g, &mut core, &r, 0);
        let mut enc = g.device.create_command_encoder(&Default::default());
        {
            let mut rec = CommandRecorder::from_encoder(&mut enc);
            core.record_phase(
                &mut rec,
                &g.device,
                &g.queue,
                &r.camera,
                &r.inputs(),
                Some(&hiz),
                1,
            );
        }
        g.queue.submit([enc.finish()]);
        let (p1, _) = collect(&g, &core, &r, 1);
        if occluded {
            assert!(
                p1.items.is_empty(),
                "everything is behind a near-plane occluder"
            );
            assert!(p1.state[(state::STATS + stats::CULLED_OCCLUSION) as usize] > 0);
        } else {
            assert_eq!(p1.items, want, "an empty Hi-Z occludes nothing");
        }
    }
}

#[test]
fn cone_cull_honours_the_instance_flag() {
    let Some(g) = gpu() else {
        return;
    };
    let mut s = scene(false);
    // Normals point away from the camera (-Z), 30 degree half-angle (cos convention).
    let cone = helio_vg_format::pack_cone([0.0, 0.0, -1.0], 0.866); // cos(30 deg)
    for c in &mut s.dag.clusters {
        c.cone = cone;
    }
    s.instances[0].inst.flags |= helio_vg_format::INST_CONE_CULL;
    let r = upload(&g, &s);
    let v = set_camera(&g, &r, [31.0, 0.0, 400.0]);
    let mut core = VgCullCore::new(
        &g.device,
        VgCullConfig {
            two_phase_occlusion: false,
            initial_bin_capacity: 4096,
            tau_px: 2.0,
            ..Default::default()
        },
    );
    begin(&g, &mut core, &r);
    run_phase(&g, &mut core, &r, 0);
    let (out, _) = collect(&g, &core, &r, 0);
    let (all, _) = expected(&s, &v, 2.0, VgMode::Auto);
    let want: BTreeSet<Item> = all.into_iter().filter(|i| i.0 != 0).collect();
    assert!(!want.is_empty());
    assert_eq!(
        out.items, want,
        "instance 0 is cone culled, instance 1 (no flag) is not"
    );
    assert!(out.state[(state::STATS + stats::CULLED_CONE) as usize] > 0);
}

#[test]
fn instances_through_the_real_gcbvh_match_the_reference() {
    use helio_pass_vg_bvh::{BvhMode, VgBvhPass};
    let Some(g) = gpu() else {
        return;
    };
    // Many instances so the tree has internal nodes: copies of the three
    // scene instances spread along x, plus the far-away one.
    let mut s = scene(false);
    let base = s.instances.clone();
    for k in 1..12 {
        for b in &base[..2] {
            let mut i = *b;
            i.matrix[12] += 400.0 * k as f32;
            i.inst.object_row = s.instances.len() as u32;
            s.instances.push(i);
        }
    }
    let mut r = upload(&g, &s);
    let mut bvh = VgBvhPass::new(&g.device);
    bvh.run_for_testing(
        &g.device,
        &g.queue,
        &r.instances,
        &r.objects,
        BvhMode::Rebuild,
    );
    r.bvh_nodes = bvh.nodes_buffer().clone();
    r.bvh_header = bvh.header_buffer().clone();
    r.bvh_items = bvh.items_buffer().clone();
    let hdr = read_words(&g, &r.bvh_header, 8);
    assert!(
        hdr[1] > 1,
        "expected an interior tree, node_count = {}",
        hdr[1]
    );
    assert_eq!(
        hdr[3],
        s.instances.len() as u32,
        "every instance is in the tree"
    );

    let mut core = VgCullCore::new(
        &g.device,
        VgCullConfig {
            two_phase_occlusion: false,
            initial_bin_capacity: 8192,
            tau_px: 2.0,
            ..Default::default()
        },
    );
    let v = set_camera(&g, &r, [31.0, 0.0, 400.0]);
    r.reset_feedback(&g);
    begin(&g, &mut core, &r);
    run_phase(&g, &mut core, &r, 0);
    let (out, _) = collect(&g, &core, &r, 0);
    let (want, _) = expected(&s, &v, 2.0, VgMode::Auto);
    // Instances far outside the frustum are culled on the GPU by their BVH
    // bounds; `expected` only drops those beyond 5000 units, so compare the
    // set restricted to instances wholly inside the frustum (x < 260).
    let near = |i: u32| s.instances[i as usize].matrix[12] < 260.0;
    let got: BTreeSet<Item> = out.items.iter().copied().filter(|i| near(i.0)).collect();
    let want: BTreeSet<Item> = want.into_iter().filter(|i| near(i.0)).collect();
    assert!(!want.is_empty());
    assert_eq!(got, want, "BVH-fed cull != reference");
    let far_emitted = out.items.iter().any(|i| !near(i.0));
    assert!(
        !far_emitted || s.instances.len() > 3,
        "far instances are culled by node bounds"
    );
}

#[test]
fn redistributed_frontier_matches_reference_without_lost_or_duplicate_clusters() {
    use helio_pass_vg_bvh::{BvhMode, VgBvhPass};
    let Some(g) = gpu() else {
        return;
    };
    let mut s = scene(false);
    let template = s.instances[0];
    s.instances = (0..128)
        .map(|i| {
            let mut instance = template;
            instance.matrix[12] = (i % 16) as f32 * 3.0 - 24.0;
            instance.matrix[13] = (i / 16) as f32 * 3.0 - 12.0;
            instance.inst.object_row = i;
            instance
        })
        .collect();
    let mut r = upload(&g, &s);
    let mut bvh = VgBvhPass::new(&g.device);
    bvh.run_for_testing(
        &g.device,
        &g.queue,
        &r.instances,
        &r.objects,
        BvhMode::Rebuild,
    );
    r.bvh_nodes = bvh.nodes_buffer().clone();
    r.bvh_header = bvh.header_buffer().clone();
    r.bvh_items = bvh.items_buffer().clone();
    let view = set_camera(&g, &r, [0.0, 0.0, 700.0]);
    let (want, _) = expected(&s, &view, 0.01, VgMode::Auto);
    assert!(want.len() > 4000);
    for rounds in [2, 4, 8] {
        for max_groups in [1, 256] {
            let mut core = VgCullCore::new(
                &g.device,
                VgCullConfig {
                    rounds,
                    max_groups,
                    two_phase_occlusion: false,
                    tau_px: 0.01,
                    initial_bin_capacity: 32768,
                    initial_queue_capacity: 65536,
                    ..Default::default()
                },
            );
            r.reset_feedback(&g);
            begin(&g, &mut core, &r);
            run_phase(&g, &mut core, &r, 0);
            let (out, _) = collect(&g, &core, &r, 0);
            assert_eq!(out.items, want, "rounds={rounds}, groups={max_groups}");
            assert_eq!(out.state[(state::STATS + stats::JOB_OVERFLOW) as usize], 0);
            assert_eq!(
                out.state[(state::STATS + stats::VISIBLE_OVERFLOW) as usize],
                0
            );
            assert!(
                out.state[(state::STATS + stats::SPILLED) as usize] > 0,
                "exercise actual redistribution"
            );
        }
    }
}

/// One `GpuMaterial` row (24 words) with the given flags.
fn material_table(g: &Gpu, flags: u32) -> wgpu::Buffer {
    let mut row = [0u32; 24];
    row[18] = flags;
    buf(g, "materials", &row)
}

fn masked_cull(
    g: &Gpu,
    s: &Scene,
    flags: u32,
    sw_masked: bool,
    cone: bool,
) -> (PhaseOut, BTreeSet<(u32, u32)>, Vec<u32>) {
    let mut r = upload(g, s);
    r.materials = Some(material_table(g, flags));
    let v = set_camera(g, &r, [31.0, 0.0, 400.0]);
    let mut core = VgCullCore::new(
        &g.device,
        VgCullConfig {
            two_phase_occlusion: false,
            initial_bin_capacity: 4096,
            tau_px: 2.0,
            sw_masked,
            ..Default::default()
        },
    );
    begin(g, &mut core, &r);
    run_phase(g, &mut core, &r, 0);
    let (out, _) = collect(g, &core, &r, 0);
    let (all, _) = expected(s, &v, 2.0, VgMode::Auto);
    let want: BTreeSet<(u32, u32)> = all
        .into_iter()
        .filter(|i| !cone || i.0 != 0)
        .map(|i| (i.0, i.1))
        .collect();
    let args = out.state.clone();
    (out, want, args)
}

#[test]
fn alpha_tested_material_routes_clusters_to_the_masked_bins() {
    use helio_pass_vg_cull::{BIN_HW_MASKED, BIN_SW_MASKED, HW_MASKED_VERTICES};
    let Some(g) = gpu() else {
        return;
    };
    let s = scene(false);
    let flags = helio_mats::FLAG_ALPHA_TEST;

    // Hardware masked only: every cluster, whatever its class, lands in bin 6.
    let (out, want, st) = masked_cull(&g, &s, flags, false, false);
    assert!(!want.is_empty());
    let got: BTreeSet<(u32, u32)> = out.items.iter().map(|i| (i.0, i.1)).collect();
    assert_eq!(
        got, want,
        "masked routing selects the same clusters, only the bin differs"
    );
    for item in &out.items {
        assert_eq!(item.2, BIN_HW_MASKED, "{item:?}");
    }
    for b in 0..BIN_COUNT as usize {
        let n = out.ranges[b].1;
        assert_eq!(n == 0, b != BIN_HW_MASKED as usize, "bin {b} holds {n}");
    }
    let a = state::ARGS as usize;
    let (first, count) = out.ranges[BIN_HW_MASKED as usize];
    assert_eq!(
        &st[a + 24..a + 28],
        &[HW_MASKED_VERTICES, count, 0, first],
        "masked draw args"
    );
    assert_eq!(
        helio_pass_vg_cull::hw_masked_draw_args_offset(0),
        (a as u64 + 24) * 4
    );

    // With the software masked kernel available, SW-class clusters go to bin 7.
    let (out, want_sw, st) = masked_cull(&g, &s, flags, true, false);
    let got: BTreeSet<(u32, u32)> = out.items.iter().map(|i| (i.0, i.1)).collect();
    assert_eq!(got, want_sw);
    let sw_n = out.ranges[BIN_SW_MASKED as usize].1;
    assert!(sw_n > 0, "the scene has software-class clusters");
    for item in &out.items {
        assert!(
            item.2 == BIN_HW_MASKED || item.2 == BIN_SW_MASKED,
            "{item:?}"
        );
    }
    assert_eq!(
        &st[a + 28..a + 31],
        &[
            sw_n.min(helio_pass_vg_cull::SW_GROUP_WIDTH),
            sw_n.div_ceil(helio_pass_vg_cull::SW_GROUP_WIDTH),
            1
        ]
    );
    for b in 0..6usize {
        assert_eq!(out.ranges[b].1, 0, "opaque bin {b} stays empty");
    }
}

#[test]
fn opaque_materials_leave_the_opaque_bins_untouched() {
    use helio_pass_vg_cull::{BIN_HW_MASKED, BIN_SW_MASKED};
    let Some(g) = gpu() else {
        return;
    };
    let s = scene(false);
    // The material table is bound, but the material is not alpha tested.
    let (out, want, _) = masked_cull(&g, &s, helio_mats::FLAG_DOUBLE_SIDED, true, false);
    let got: BTreeSet<(u32, u32)> = out.items.iter().map(|i| (i.0, i.1)).collect();
    assert_eq!(got, want);
    assert_eq!(
        out.ranges[BIN_HW_MASKED as usize].1 + out.ranges[BIN_SW_MASKED as usize].1,
        0
    );
}

#[test]
fn two_sided_masked_clusters_skip_the_normal_cone_test() {
    let Some(g) = gpu() else {
        return;
    };
    let mut s = scene(false);
    let cone = helio_vg_format::pack_cone([0.0, 0.0, -1.0], 0.866);
    for c in &mut s.dag.clusters {
        c.cone = cone;
    }
    s.instances[0].inst.flags |= helio_vg_format::INST_CONE_CULL;
    // Single-sided masked: instance 0 faces away and is cone culled.
    let (out, want, _) = masked_cull(&g, &s, helio_mats::FLAG_ALPHA_TEST, false, true);
    let got: BTreeSet<(u32, u32)> = out.items.iter().map(|i| (i.0, i.1)).collect();
    assert_eq!(got, want);
    assert!(out.state[(state::STATS + stats::CULLED_CONE) as usize] > 0);
    // Two-sided masked (foliage): back faces are visible, nothing is cone culled.
    let (out, _, _) = masked_cull(
        &g,
        &s,
        helio_mats::FLAG_ALPHA_TEST | helio_mats::FLAG_DOUBLE_SIDED,
        false,
        false,
    );
    assert_eq!(out.state[(state::STATS + stats::CULLED_CONE) as usize], 0);
    assert!(
        out.items.iter().any(|i| i.0 == 0),
        "instance 0 survives for two-sided foliage"
    );
}

/// The per-instance root early-out must not change the cut: identical visible
/// sets and page requests with and without it, over distances (including ones
/// where it fires for every instance), taus, scales and partial residency.
#[test]
fn root_shortcut_gives_the_same_cut_as_the_full_walk() {
    let Some(g) = gpu() else {
        eprintln!("skipping vg-cull GPU test: no adapter");
        return;
    };
    let mut fired_total = 0u32;
    for root_only in [false, true] {
        let s = scene(root_only);
        let r = upload(&g, &s);
        let mut core = VgCullCore::new(
            &g.device,
            VgCullConfig {
                two_phase_occlusion: false,
                initial_bin_capacity: 4096,
                ..Default::default()
            },
        );
        for dist in [300.0f32, 700.0, 4000.0, 30000.0] {
            let _v = set_camera(&g, &r, [31.0, 0.0, dist]);
            for tau in [0.01f32, 0.5, 2.0, 20.0, 1.0e3, 1.0e5] {
                core.config.tau_px = tau;
                let mut results = Vec::new();
                for shortcut in [false, true] {
                    core.config.root_shortcut = shortcut;
                    r.reset_feedback(&g);
                    begin(&g, &mut core, &r);
                    run_phase(&g, &mut core, &r, 0);
                    let (out, req) = collect(&g, &core, &r, 0);
                    let stat = |i: u32| out.state[(state::STATS + i) as usize];
                    if !shortcut {
                        assert_eq!(stat(stats::ROOT_SHORTCUT), 0);
                    }
                    results.push((
                        out.items.clone(),
                        req.iter().copied().collect::<BTreeSet<u32>>(),
                        stat(stats::ROOT_SHORTCUT),
                        stat(stats::JOBS),
                    ));
                }
                assert_eq!(
                    results[0].0, results[1].0,
                    "root_only={root_only} dist={dist} tau={tau}: cut differs"
                );
                assert_eq!(
                    results[0].1, results[1].1,
                    "root_only={root_only} dist={dist} tau={tau}: page requests differ"
                );
                fired_total += results[1].2;
                if results[1].2 > 0 {
                    assert!(
                        results[1].3 <= results[0].3,
                        "the early-out never adds jobs"
                    );
                }
            }
        }
    }
    assert!(
        fired_total > 0,
        "the early-out never fired: the test would prove nothing"
    );
}

// ── Stability: hysteresis, prefetch, the overdraw valve ────────────────────

/// One frame (single phase, no occlusion) at `eye`, from the current history.
fn frame_at(g: &Gpu, core: &mut VgCullCore, r: &Resources, eye: [f32; 3]) -> (PhaseOut, Vec<u32>) {
    set_camera(g, r, eye);
    r.reset_feedback(g);
    begin(g, core, r);
    run_phase(g, core, r, 0);
    collect(g, core, r, 0)
}

fn one_instance_scene(resident_root_only: bool) -> Scene {
    let mut s = scene(resident_root_only);
    s.instances.truncate(1);
    s
}

fn stateless_triangles(g: &Gpu, s: &Scene, r: &Resources, d: f32) -> u32 {
    let v = set_camera(g, r, [-87.0, 0.0, d]);
    triangle_count(s, &expected(s, &v, 1.0, VgMode::Auto).0)
}

/// A distance `d` where the cut (stateless) is finer than at `d * 1.004`: the
/// threshold of some group sits between them.
fn find_transition(g: &Gpu, s: &Scene, r: &Resources) -> (f32, f32) {
    let mut d = 200.0f32;
    let mut last = stateless_triangles(g, s, r, d);
    while d < 1.0e5 {
        let next = d * 1.004;
        let t = stateless_triangles(g, s, r, next);
        if t < last {
            return (d, next);
        }
        last = t;
        d = next;
    }
    panic!("the cut never changes over the scanned distances");
}

fn stability_config() -> VgCullConfig {
    VgCullConfig {
        two_phase_occlusion: false,
        initial_bin_capacity: 4096,
        tile_density_limit: 0.0,
        ..Default::default()
    }
}

#[test]
fn hysteresis_keeps_the_cut_still_under_jitter_and_releases_it_outside_the_band() {
    let Some(g) = gpu() else {
        return;
    };
    let s = one_instance_scene(false);
    let r = upload(&g, &s);
    let (d_fine, d_coarse) = find_transition(&g, &s, &r);
    let mut core = VgCullCore::new(&g.device, stability_config());
    let fine = {
        let v = set_camera(&g, &r, [-87.0, 0.0, d_fine]);
        expected(&s, &v, 1.0, VgMode::Auto).0
    };
    let coarse = {
        let v = set_camera(&g, &r, [-87.0, 0.0, d_coarse]);
        expected(&s, &v, 1.0, VgMode::Auto).0
    };
    assert!(
        triangle_count(&s, &fine) > triangle_count(&s, &coarse),
        "the test needs a threshold between {d_fine} and {d_coarse}"
    );

    // Jitter across the threshold every frame: a stateless cut flaps between
    // `fine` and `coarse`; the held cut stays on the finer one, which is never
    // coarser than tau requires.
    flush_history(&g, &mut core, &r);
    for frame in 0..16 {
        let d = if frame % 2 == 0 { d_fine } else { d_coarse };
        let (out, _) = frame_at(&g, &mut core, &r, [-87.0, 0.0, d]);
        assert_eq!(out.items, fine, "frame {frame} at {d}: the cut flapped");
    }
    // A still camera on the coarse side after the finer cut was chosen: held.
    for frame in 0..4 {
        let (out, _) = frame_at(&g, &mut core, &r, [-87.0, 0.0, d_coarse]);
        assert_eq!(out.items, fine, "still frame {frame} lost the held detail");
    }
    // Far outside the band (error well below tau / (1 + TAU_HYST)) it releases.
    let d_far = d_coarse * (1.0 + helio_pass_vg_cull::TAU_HYST) * 1.3;
    let far_want = {
        let v = set_camera(&g, &r, [-87.0, 0.0, d_far]);
        expected(&s, &v, 1.0, VgMode::Auto).0
    };
    let mut released = BTreeSet::new();
    for _ in 0..3 {
        released = frame_at(&g, &mut core, &r, [-87.0, 0.0, d_far]).0.items;
    }
    assert_eq!(released, far_want, "the hold never releases");
    assert!(triangle_count(&s, &far_want) < triangle_count(&s, &coarse));
}

#[test]
fn the_sw_hw_edge_test_has_hysteresis() {
    let Some(g) = gpu() else {
        return;
    };
    let s = one_instance_scene(false);
    let r = upload(&g, &s);
    let mut core = VgCullCore::new(&g.device, stability_config());
    // Software-class population of the stateless cut over distance; find the
    // distance where it first changes.
    let sw_count = |g: &Gpu, core: &mut VgCullCore, d: f32| {
        flush_history(g, core, &r);
        let (out, _) = frame_at(g, core, &r, [-87.0, 0.0, d]);
        out.items
            .iter()
            .filter(|i| i.2 == helio_pass_vg_cull::BIN_SW)
            .count()
    };
    let mut d = 100.0f32;
    let mut last = sw_count(&g, &mut core, d);
    let mut found = None;
    while d < 1.0e5 {
        let next = d * 1.004;
        let n = sw_count(&g, &mut core, next);
        if n != last {
            found = Some((d, next, last, n));
            break;
        }
        last = n;
        d = next;
    }
    let (d_a, d_b, n_a, n_b) =
        found.expect("the SW class never changes over the scanned distances");
    // Jitter across that threshold: after the first frame the classes stay put.
    flush_history(&g, &mut core, &r);
    let mut seen = Vec::new();
    for frame in 0..12 {
        let d = if frame % 2 == 0 { d_a } else { d_b };
        let (out, _) = frame_at(&g, &mut core, &r, [-87.0, 0.0, d]);
        seen.push(
            out.items
                .iter()
                .filter(|i| i.2 == helio_pass_vg_cull::BIN_SW)
                .count(),
        );
    }
    assert!(
        seen.windows(2).all(|w| w[0] == w[1]),
        "SW/HW classes flapped (stateless: {n_a} vs {n_b}): {seen:?}"
    );
}

/// Page requests and the urgent subset of one frame at distance `d`.
fn requests_at(
    g: &Gpu,
    core: &mut VgCullCore,
    r: &Resources,
    d: f32,
) -> (BTreeSet<u32>, BTreeSet<u32>) {
    flush_history(g, core, r);
    set_camera(g, r, [-87.0, 0.0, d]);
    r.reset_feedback(g);
    begin(g, core, r);
    run_phase(g, core, r, 0);
    let fb = feedback::decode(
        &read_words(g, &r.feedback, r.layout.total_words() as u64),
        r.layout,
        r.page_count,
    );
    (
        fb.requests.into_iter().collect(),
        fb.urgent.into_iter().collect(),
    )
}

#[test]
fn prefetch_requests_children_while_the_error_is_within_the_band() {
    use helio_pass_vg_cull::reference::select_clusters_prefetch;
    let Some(g) = gpu() else {
        return;
    };
    // Only the roots are resident: every child page is a request candidate.
    let s = one_instance_scene(true);
    let r = upload(&g, &s);
    let slots = s.dag.slots(&s.resident);
    let lookup = s.dag.lookup();
    let asset = s.dag.asset(&slots, &lookup);
    let reference = |d: f32, k: f32| {
        let v = set_camera(&g, &r, [-87.0, 0.0, d]);
        let sel = select_clusters_prefetch(&asset, &v, 1.0, &s.instances[0], false, k);
        (sel.requested_pages, sel.urgent_pages)
    };
    // A distance where the cut does not need the next level yet, but is within
    // 4x of needing it.
    let mut d = 100.0f32;
    let band = loop {
        let (need, _) = reference(d, 1.0);
        let (early, _) = reference(d, 4.0);
        if need.is_empty() && !early.is_empty() {
            break d;
        }
        d *= 1.01;
        assert!(d < 1.0e6, "no prefetch-only distance found");
    };
    let core = |k: f32| {
        VgCullCore::new(
            &g.device,
            VgCullConfig {
                prefetch_factor: k,
                ..stability_config()
            },
        )
    };
    let mut demand_only = core(1.0);
    let (requests, _) = requests_at(&g, &mut demand_only, &r, band);
    assert!(requests.is_empty(), "k=1 requested early: {requests:?}");
    drop(demand_only);
    let mut prefetching = core(4.0);
    let (requests, urgent) = requests_at(&g, &mut prefetching, &r, band);
    let (want, want_urgent) = reference(band, 4.0);
    assert_eq!(requests, want, "prefetch band requests at {band}");
    assert!(urgent.is_empty() && want_urgent.is_empty(), "nothing is urgent yet");
    // Close enough that the cut needs the pages: those are urgent, and the band
    // still holds the prefetches below them.
    let (requests, urgent) = requests_at(&g, &mut prefetching, &r, band / 3.0);
    let (want, want_urgent) = reference(band / 3.0, 4.0);
    assert_eq!(requests, want);
    assert_eq!(urgent, want_urgent);
    assert!(!urgent.is_empty() && urgent.is_subset(&requests));
}

fn tile_block(g: &Gpu, core: &VgCullCore) -> Vec<[u32; 3]> {
    let b = core.buffers();
    let words = u64::from(b.tile_base()) + u64::from(b.tiles[0] * b.tiles[1]) * 3;
    let all = read_words(g, &b.history, words);
    all[b.tile_base() as usize..]
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect()
}

#[test]
fn overdraw_valve_coarsens_only_the_hot_tiles_stably() {
    use helio_pass_vg_cull::{TILE_FIXED, TILE_SCALE_MAX};
    let Some(g) = gpu() else {
        return;
    };
    // 48 copies of the asset stacked at one place (a hot patch), and one lone
    // copy far to the side on screen (its tiles are below the limit).
    const STACK: usize = 48;
    let mut s = scene(false);
    let make = |x: f32, row: u32| {
        RefInstance::translated_scaled(
            [x, 0.0, 0.0],
            1.0,
            GpuVgInstance {
                asset_slot: 0,
                flags: INST_ENABLED,
                lod_error_scale: 1.0,
                object_row: row,
            },
        )
    };
    s.instances = (0..STACK).map(|i| make(-150.0, i as u32)).collect();
    s.instances.push(make(700.0, STACK as u32));
    let r = upload(&g, &s);
    let eye = [-87.0, 0.0, 1500.0];
    let lone = STACK as u32;
    let run = |limit: f32, frames: usize| -> (Vec<BTreeSet<Item>>, Vec<f32>, Vec<[u32; 3]>) {
        let mut core = VgCullCore::new(
            &g.device,
            VgCullConfig {
                initial_bin_capacity: 1 << 16,
                tile_density_limit: limit,
                ..stability_config()
            },
        );
        let mut sets = Vec::new();
        for _ in 0..frames {
            let (out, _) = frame_at(&g, &mut core, &r, eye);
            sets.push(out.items);
        }
        let tiles = tile_block(&g, &core);
        let scales = tiles
            .iter()
            .map(|t| f32::from_bits(t[2]).max(1.0))
            .collect();
        (sets, scales, tiles)
    };
    let (open, _, tiles) = run(0.0, 2);
    let density = |t: &[u32; 3]| t[0] as f32 / (t[1] as f32).clamp(1.0, 256.0 * TILE_FIXED);
    let hot = tiles.iter().map(density).fold(0.0f32, f32::max);
    assert!(hot > 25.0, "the stacked patch is not hot enough for the test: {hot}");
    let triangles = |items: &BTreeSet<Item>, lone_only: bool| -> u32 {
        triangle_count(
            &s,
            &items
                .iter()
                .copied()
                .filter(|i| (i.0 == lone) == lone_only)
                .collect(),
        )
    };
    let (valved, scales, _) = run(16.0, 40);
    let last = valved.last().unwrap();
    assert!(
        triangles(last, false) * 10 < triangles(&open[1], false) * 9,
        "the hot stack was not coarsened: {} vs {}",
        triangles(last, false),
        triangles(&open[1], false)
    );
    // The lone copy is not over the limit: its cut is exactly the unvalved one.
    let lone_open: BTreeSet<_> = open[1].iter().copied().filter(|i| i.0 == lone).collect();
    let lone_valved: BTreeSet<_> = last.iter().copied().filter(|i| i.0 == lone).collect();
    assert_eq!(lone_valved, lone_open, "the valve moved an unrelated instance");
    // Stable: the last frames are identical (no oscillation), within the ceiling.
    let tail = &valved[valved.len() - 8..];
    assert!(tail.windows(2).all(|w| w[0] == w[1]), "the valve oscillates");
    assert!(scales.iter().all(|&g| g <= TILE_SCALE_MAX + 1e-3));
    assert!(scales.iter().any(|&g| g > 1.5), "no tile coarsened");
}
