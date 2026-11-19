//! The skin cache (`meshes_component::skin_cache`) over a skinned mesh in an
//! engine world: a pose is blended once into the cache (four influences,
//! normalized), its previous positions are the cache's before the pose, the
//! frame after the last pose settles them, an unchanged palette costs
//! nothing, and the instance table points the drawers at the instance's
//! range. Needs a GPU adapter (lavapipe works); skips without one.

use std::sync::Arc;

use glam::{Mat4, Vec3};
use helio_mesh_api::{PackedVertex, SkinCacheRow, SkinVertex, SkinnedVertex};
use meshes_component::components::{MeshGeometryKey, SkeletalMeshComponent};
use meshes_component::mesh_cache::MeshSection;
use meshes_component::skin_cache::{SkinCache, SkinView};
use pulsar_scenedb::gpu::{
    EngineGpuContext, GpuMirrorHandle, RegionClassConfig, SceneGpuConfig, SceneGpuStore,
};
use pulsar_scenedb::{Entity, World};

/// A world mirrored to the GPU with the skinned mesh's columns, as the
/// meshes' columns registration makes them.
fn mirrored_world(device: &Arc<wgpu::Device>, queue: &Arc<wgpu::Queue>) -> World {
    let ctx = EngineGpuContext::new(device.clone(), queue.clone());
    let mut store = SceneGpuStore::new(
        &ctx,
        SceneGpuConfig {
            classes: vec![RegionClassConfig {
                capacity: 64,
                max_resident_cells: 1,
            }],
            tombstone_headroom: 8,
            max_cells_metadata: 16,
        },
    );
    SkeletalMeshComponent::register_gpu_columns_growable(&mut store, 16, device);
    let mut world = World::new();
    world.attach_gpu_mirror(GpuMirrorHandle::new(Arc::new(store), queue.clone()));
    world
}

fn device() -> Option<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> {
    let instance = wgpu::Instance::new(helio_core::wgpu_instance_descriptor());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).ok()?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: helio::required_wgpu_limits(adapter.limits()),
        ..Default::default()
    }))
    .ok()?;
    Some((Arc::new(device), Arc::new(queue)))
}

/// Three vertices: one on joint 0, one on joint 1, one half on each.
// The component has private fields: no struct literal from outside.
#[allow(clippy::field_reassign_with_default)]
fn skinned_mesh(palette: [Mat4; 2]) -> SkeletalMeshComponent {
    let vertex = |x: f32| {
        PackedVertex::from_components(
            [x, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0; 2],
            [1.0, 0.0, 0.0],
            1.0,
        )
    };
    let mut mesh = SkeletalMeshComponent::default();
    mesh.vertices = vec![vertex(0.0), vertex(1.0), vertex(2.0)];
    mesh.indices = vec![0, 1, 2];
    mesh.skin_vertices = vec![
        SkinVertex {
            joints: [0, 0, 0, 0],
            weights: [1.0, 0.0, 0.0, 0.0],
        },
        SkinVertex {
            joints: [1, 0, 0, 0],
            weights: [1.0, 0.0, 0.0, 0.0],
        },
        SkinVertex {
            joints: [0, 1, 0, 0],
            weights: [0.5, 0.5, 0.0, 0.0],
        },
    ];
    mesh.set_palette(palette.map(|m| m.to_cols_array_2d()).to_vec());
    mesh.mesh_sections = vec![MeshSection {
        first_index: 0,
        index_count: 3,
        material_slot: 0,
    }];
    mesh.bounds_local = [1.0, 0.0, 0.0, 1.5];
    mesh.geometry_key = MeshGeometryKey(7);
    mesh
}

struct Frame {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
}

impl Frame {
    /// Flush the scene and update the cache, as the meshes' host part does.
    fn update(&self, world: &World, cache: &mut SkinCache) {
        self.update_in(world, None, cache);
    }

    /// [`Self::update`] for the frame `view` sees.
    fn update_in(&self, world: &World, view: Option<&SkinView>, cache: &mut SkinCache) {
        world.flush_gpu_mirror(&self.queue);
        let mut encoder = self.device.create_command_encoder(&Default::default());
        cache.update(world, view, &self.queue, &mut encoder);
        self.queue.submit([encoder.finish()]);
    }

    fn read<T: bytemuck::Pod>(&self, buffer: &wgpu::Buffer) -> Vec<T> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
        self.queue.submit([encoder.finish()]);
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, |r| r.unwrap());
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        let out = bytemuck::cast_slice(&staging.slice(..).get_mapped_range().unwrap()).to_vec();
        staging.unmap();
        out
    }

    /// The instance's cached vertices.
    fn vertices(&self, cache: &SkinCache, instance: Entity) -> Vec<SkinnedVertex> {
        let range = cache.instance(instance).expect("resident");
        let all: Vec<SkinnedVertex> = self.read(cache.buffer());
        all[range.cache_vertex as usize..][..range.vertex_count as usize].to_vec()
    }
}

fn close(a: [f32; 3], b: Vec3) -> bool {
    (Vec3::from(a) - b).abs().max_element() < 1e-5
}

#[test]
fn the_skin_cache_poses_settles_and_skips_unchanged_poses() {
    let Some((device, queue)) = device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let frame = Frame {
        device: device.clone(),
        queue: queue.clone(),
    };
    let mut world = mirrored_world(&device, &queue);
    let up = Mat4::from_translation(Vec3::Z);
    let right = Mat4::from_translation(Vec3::X * 2.0);
    let instance = world.spawn();
    world.insert(instance, skinned_mesh([up, right]));
    let mut cache = SkinCache::new(device.clone());

    // Its first pose: blended, no motion.
    frame.update(&world, &mut cache);
    assert_eq!(cache.frame_counts(), (1, 3));
    let first = frame.vertices(&cache, instance);
    let want = [
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(3.0, 0.0, 0.0),
        Vec3::new(3.0, 0.0, 0.5),
    ];
    for (v, w) in first.iter().zip(want) {
        assert!(close(v.position, w), "{v:?} want {w}");
        assert_eq!(v.previous, v.position, "a first pose does not move");
        let normal = Vec3::from(helio_frame_api::rays::stream::unpack_unit(v.normal));
        assert!(
            normal.distance(Vec3::Z) < 1e-4,
            "+z normal, octahedral: {normal}"
        );
    }
    // The instance table points the drawers at its range.
    let rows: Vec<SkinCacheRow> = frame.read(cache.rows());
    let range = cache.instance(instance).unwrap();
    assert_eq!(
        rows[instance.index() as usize],
        SkinCacheRow {
            offset: range.cache_vertex.wrapping_sub(range.mesh_vertex),
            resident: 1,
            pose: range.revision as u32,
            settled: 0,
        }
    );

    // The same palette shown again: nothing posed, the last pose settles.
    world
        .get_mut::<SkeletalMeshComponent>(instance)
        .unwrap()
        .set_palette(vec![up.to_cols_array_2d(), right.to_cols_array_2d()]);
    frame.update(&world, &mut cache);
    assert_eq!(cache.frame_counts(), (0, 0), "a settle is a row, no pass");
    let rows: Vec<SkinCacheRow> = frame.read(cache.rows());
    assert_eq!(rows[instance.index() as usize].settled, 1, "settled");
    let revision = cache.instance(instance).unwrap().revision;

    // A new pose: blended, its previous positions the previous palette's
    // (the pose shown before it).
    let higher = Mat4::from_translation(Vec3::Z * 3.0);
    world
        .get_mut::<SkeletalMeshComponent>(instance)
        .unwrap()
        .set_palette(vec![higher.to_cols_array_2d(), right.to_cols_array_2d()]);
    frame.update(&world, &mut cache);
    assert_eq!(cache.frame_counts(), (1, 3));
    assert_eq!(cache.instance(instance).unwrap().revision, revision + 1);
    let posed = frame.vertices(&cache, instance);
    assert!(close(posed[0].position, Vec3::new(0.0, 0.0, 3.0)));
    assert!(close(posed[2].position, Vec3::new(3.0, 0.0, 1.5)));
    for (v, before) in posed.iter().zip(&first) {
        assert_eq!(v.previous, before.position, "the previous pose's position");
    }

    // No change: its row settles (previous = current), then nothing.
    frame.update(&world, &mut cache);
    assert_eq!(cache.frame_counts(), (0, 0));
    let rows: Vec<SkinCacheRow> = frame.read(cache.rows());
    assert_eq!(rows[instance.index() as usize].settled, 1, "settled");
    frame.update(&world, &mut cache);
    assert_eq!(
        cache.frame_counts(),
        (0, 0),
        "an unchanged pose costs nothing"
    );
    assert_eq!(cache.instance(instance).unwrap().revision, revision + 1);

    // Removed: it leaves the cache and its row.
    world.remove::<SkeletalMeshComponent>(instance);
    frame.update(&world, &mut cache);
    assert!(cache.instance(instance).is_none());
    let rows: Vec<SkinCacheRow> = frame.read(cache.rows());
    assert_eq!(rows[instance.index() as usize], SkinCacheRow::default());
}

/// With a view, an instance off-screen and past the traced range is not
/// skinned (and its pose is kept for when it is wanted); wanted again it is
/// skinned that frame with no motion; unwanted for long it leaves the
/// cache. A rotated pose's normal is the float blend's to 1e-4 (octahedral
/// 16-bit axes).
#[test]
fn the_skin_cache_skins_what_the_frame_sees() {
    let Some((device, queue)) = device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let frame = Frame {
        device: device.clone(),
        queue: queue.clone(),
    };
    let mut world = mirrored_world(&device, &queue);
    let instance = world.spawn();
    let turn = Mat4::from_rotation_y(0.7);
    let mut mesh = skinned_mesh([turn, turn]);
    // 5 km behind a view looking down -z from the origin.
    mesh.bounds_local = [0.0, 0.0, 5000.0, 2.0];
    world.insert(instance, mesh);
    let view = |eye: glam::DVec3, forward: Vec3| SkinView {
        eye,
        view_proj: helio_core::depth::perspective(1.0, 1.0, 0.1)
            * glam::camera::rh::view::look_to_mat4(Vec3::ZERO, forward, Vec3::Y),
        rays: helio_pass_mesh_rays::RayTracingSettings::default(),
    };
    let away = view(glam::DVec3::ZERO, -Vec3::Z);
    let mut cache = SkinCache::new(device.clone());
    frame.update_in(&world, Some(&away), &mut cache);
    assert_eq!(cache.frame_counts(), (0, 0), "not skinned");
    assert!(cache.instance(instance).is_none(), "and holds no range");

    // Turned to face it: skinned that frame.
    let facing = view(glam::DVec3::ZERO, Vec3::Z);
    frame.update_in(&world, Some(&facing), &mut cache);
    assert_eq!(cache.frame_counts(), (1, 3));
    let posed = frame.vertices(&cache, instance);
    let n = turn.transform_vector3(Vec3::Z).normalize();
    for v in &posed {
        assert_eq!(v.previous, v.position, "no motion on its first frame");
        let got = Vec3::from(helio_frame_api::rays::stream::unpack_unit(v.normal));
        assert!(got.angle_between(n) < 1e-4, "normal {got} want {n}");
    }

    // Posed while looking away: kept stale, skinned when seen again.
    world
        .get_mut::<SkeletalMeshComponent>(instance)
        .unwrap()
        .set_palette(vec![Mat4::IDENTITY.to_cols_array_2d(); 2]);
    frame.update_in(&world, Some(&away), &mut cache);
    assert_eq!(
        cache.frame_counts().0,
        0,
        "a pose nobody sees is not skinned"
    );
    frame.update_in(&world, Some(&facing), &mut cache);
    assert_eq!(cache.frame_counts().0, 1, "skinned once seen");
    for v in frame.vertices(&cache, instance) {
        assert_eq!(v.previous, v.position, "a stale pose does not move");
    }

    // Unwanted for long: its range goes back.
    for _ in 0..302 {
        frame.update_in(&world, Some(&away), &mut cache);
    }
    assert!(cache.instance(instance).is_none(), "evicted");
}

// ── Cost ────────────────────────────────────────────────────────────────────

/// A grid mesh of `side`² vertices on two joints (weights across x), for the
/// cost table.
fn grid_mesh(side: u32) -> SkeletalMeshComponent {
    let mut mesh = skinned_mesh([Mat4::IDENTITY; 2]);
    let n = side as f32 - 1.0;
    mesh.vertices.clear();
    mesh.skin_vertices.clear();
    mesh.indices.clear();
    for y in 0..side {
        for x in 0..side {
            let (u, v) = (x as f32 / n, y as f32 / n);
            mesh.vertices.push(PackedVertex::from_components(
                [u, v, 0.0],
                [0.0, 0.0, 1.0],
                [u, v],
                [1.0, 0.0, 0.0],
                1.0,
            ));
            mesh.skin_vertices.push(SkinVertex {
                joints: [0, 1, 0, 0],
                weights: [1.0 - u, u, 0.0, 0.0],
            });
        }
    }
    for y in 0..side - 1 {
        for x in 0..side - 1 {
            let i = y * side + x;
            mesh.indices
                .extend_from_slice(&[i, i + 1, i + side, i + 1, i + side + 1, i + side]);
        }
    }
    mesh.mesh_sections[0].index_count = mesh.indices.len() as u32;
    mesh.bounds_local = [0.5, 0.5, 0.0, 0.8];
    mesh
}

/// The skin pass, the deforming BLASes' refits and the TLAS build a frame,
/// in GPU time over pipelined frames (three in flight), for 1, 16 and 256
/// posed instances of a 10,000-vertex mesh, and for an unchanged pose.
/// Run by hand: `skin_cache_cost --ignored --nocapture`.
#[test]
#[ignore = "a cost table, run by hand"]
fn skin_cache_cost() {
    let instance = wgpu::Instance::new(helio_core::wgpu_instance_descriptor());
    let Ok(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else {
        return;
    };
    let needed = wgpu::Features::TIMESTAMP_QUERY
        | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
        | wgpu::Features::EXPERIMENTAL_RAY_QUERY;
    if !adapter.features().contains(needed) {
        eprintln!("no timestamps or ray queries: skipped");
        return;
    }
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: needed,
        required_limits: helio::required_wgpu_limits(adapter.limits()),
        // The deforming BLASes need wgpu's experimental ray tracing.
        experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
        ..Default::default()
    }))
    .unwrap();
    let (device, queue) = (Arc::new(device), Arc::new(queue));
    let period = f64::from(queue.get_timestamp_period());
    println!("| instances x vertices | skin pass ms | BLAS refits ms | TLAS ms | CPU ms |");
    println!("|---|---|---|---|---|");
    // Posed but off-screen and past the traced range: a view looking away
    // from instances 5 km behind it.
    let away = SkinView {
        eye: glam::DVec3::ZERO,
        view_proj: helio_core::depth::perspective(1.0, 16.0 / 9.0, 0.1)
            * glam::camera::rh::view::look_to_mat4(Vec3::ZERO, -Vec3::Z, Vec3::Y),
        rays: helio_pass_mesh_rays::RayTracingSettings::default(),
    };
    // (instances, grid side: 100 is 10,000 vertices, 173 about 30,000).
    for (count, side, posed, culled) in [
        (1usize, 100u32, true, false),
        (16, 100, true, false),
        (50, 173, true, false),
        (256, 100, true, false),
        (256, 100, false, false),
        (256, 100, true, true),
    ] {
        let mut world = mirrored_world(&device, &queue);
        let mut instances = Vec::new();
        for _ in 0..count {
            let instance = world.spawn();
            let mut mesh = grid_mesh(side);
            if culled {
                mesh.bounds_local = [0.0, 0.0, 5000.0, 0.8];
            }
            world.insert(instance, mesh);
            instances.push(instance);
        }
        let mut cache = SkinCache::new(device.clone());
        let mut blas = helio_core::BlasManager::new(device.clone());
        let mut tlas = helio_core::TlasManager::new(device.clone(), count as u32);
        let ring: Vec<_> = (0..3)
            .map(|_| {
                (
                    device.create_query_set(&wgpu::QuerySetDescriptor {
                        label: None,
                        ty: wgpu::QueryType::Timestamp,
                        count: 4,
                    }),
                    device.create_buffer(&wgpu::BufferDescriptor {
                        label: None,
                        size: 32,
                        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    }),
                    device.create_buffer(&wgpu::BufferDescriptor {
                        label: None,
                        size: 32,
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }),
                )
            })
            .collect();
        // Per frame: skin pass, refits, TLAS (GPU ms) and CPU ms.
        let mut samples: Vec<[f64; 4]> = Vec::new();
        for frame in 0..90usize {
            if posed || frame < 2 {
                let angle = frame as f32 * 0.01;
                for &instance in &instances {
                    world
                        .get_mut::<SkeletalMeshComponent>(instance)
                        .unwrap()
                        .set_palette(vec![
                            Mat4::IDENTITY.to_cols_array_2d(),
                            Mat4::from_rotation_y(angle).to_cols_array_2d(),
                        ]);
                }
            }
            world.flush_gpu_mirror(&queue);
            let (queries, resolve, readback) = &ring[frame % 3];
            if frame >= 3 {
                // The frame three back, its timestamps.
                let slice = readback.slice(..);
                slice.map_async(wgpu::MapMode::Read, |r| r.unwrap());
                let _ = device.poll(wgpu::PollType::wait_indefinitely());
                let t: Vec<u64> = bytemuck::cast_slice(&slice.get_mapped_range().unwrap()).to_vec();
                readback.unmap();
                let back = frame - 3;
                if back >= 10 {
                    let ms = |a: u64, b: u64| b.saturating_sub(a) as f64 * period / 1e6;
                    let s = &mut samples[back - 10];
                    s[0] = ms(t[0], t[1]);
                    s[1] = ms(t[1], t[2]);
                    s[2] = ms(t[2], t[3]);
                }
            }
            let start = std::time::Instant::now();
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.write_timestamp(queries, 0);
            cache.update(&world, culled.then_some(&away), &queue, &mut encoder);
            encoder.write_timestamp(queries, 1);
            let store = world.gpu_mirror().unwrap().store();
            let index_pool = store
                .resolve_buffer_handle(helio_mesh_api::MESH_INDICES.key())
                .unwrap();
            let mut inputs = Vec::new();
            // `SKIN_COST_NO_RAYS=1`: the skin pass alone (no BLAS or TLAS).
            let no_rays = std::env::var_os("SKIN_COST_NO_RAYS").is_some();
            for (k, &instance) in instances.iter().enumerate() {
                if no_rays {
                    break;
                }
                // A culled instance is not skinned, so has no BLAS to refit.
                let Some(skin) = cache.instance(instance) else {
                    continue;
                };
                let indices =
                    SkeletalMeshComponent::indices_gpu_handle(store, instance.index()).unwrap();
                blas.build(
                    k as u64,
                    &mut encoder,
                    vec![helio_core::BlasSegment {
                        geometry: helio_core::BlasGeometry {
                            positions_revision: skin.revision,
                            topology_revision: 0,
                            vertices: cache.buffer(),
                            first_vertex: skin.cache_vertex,
                            vertex_count: skin.vertex_count,
                            vertex_stride: 32,
                            indices: Some(&index_pool.buffer),
                            first_index: indices.offset,
                            index_count: indices.count,
                        },
                        opaque: true,
                    }],
                    helio_core::Movability::Dynamic,
                )
                .unwrap();
                inputs.push(helio_core::TlasInstanceInput {
                    geometry_id: k as u64,
                    custom_data: 0,
                    transform: [
                        1.0, 0.0, 0.0, k as f32, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                    ],
                });
            }
            encoder.write_timestamp(queries, 2);
            if !inputs.is_empty() {
                tlas.build(&mut encoder, &inputs, &blas).unwrap();
            }
            encoder.write_timestamp(queries, 3);
            encoder.resolve_query_set(queries, 0..4, resolve, 0);
            encoder.copy_buffer_to_buffer(resolve, 0, readback, 0, 32);
            let cpu = start.elapsed().as_secs_f64() * 1e3;
            queue.submit([encoder.finish()]);
            if frame >= 10 {
                samples.push([0.0, 0.0, 0.0, cpu]);
            }
        }
        // The last three frames' GPU times were never read.
        samples.truncate(samples.len() - 3);
        let median = |i: usize| {
            let mut v: Vec<f64> = samples.iter().map(|s| s[i]).collect();
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        println!(
            "| {count} x {}k {} | {:.3} | {:.3} | {:.3} | {:.3} |",
            side * side / 1000,
            match (posed, culled) {
                (_, true) => "posed, off-screen and untraced",
                (true, _) => "posed",
                _ => "unchanged",
            },
            median(0),
            median(1),
            median(2),
            median(3)
        );
    }
}
