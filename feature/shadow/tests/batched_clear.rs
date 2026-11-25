//! `depth_clear.wgsl` clears every serviced face's pages in one draw and
//! `retire.wgsl` retires the serviced faces in one dispatch. These tests run
//! the shipped shaders on the GPU and check what the per-face viewport and
//! buffer commands used to guarantee: the clear lands in the face's tile and
//! nowhere else, a gated face is cleared only when its dirty bit is set, the
//! dynamic layer clears only the pages in the face's mask, and retiring copies
//! the matrix and zeroes the dirty state of the serviced faces and of no other.
//! They return without checking when there is no GPU.

use wgpu::util::DeviceExt;

const CLEAR_WGSL: &str = include_str!("../shaders/depth_clear.wgsl");
const RETIRE_WGSL: &str = include_str!("../shaders/retire.wgsl");

const ATLAS: u32 = 128;
const BACKGROUND: f32 = 0.25;

fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(helio_core::wgpu_instance_descriptor());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await
            .ok()?;
        adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .ok()
    })
}

fn storage(device: &wgpu::Device, words: &[u32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(words),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    })
}

fn read_buffer(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer) -> Vec<u32> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: buffer.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
    queue.submit(Some(encoder.finish()));
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, |r| r.unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let mapped = staging.slice(..).get_mapped_range().unwrap();
    bytemuck::cast_slice(&mapped).to_vec()
}

fn storage_entry(
    binding: u32,
    read_only: bool,
    visibility: wgpu::ShaderStages,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Runs the clear shader over `jobs` (a header, then one job each) against an
/// atlas filled with `BACKGROUND`, and returns the atlas's depth.
fn run_clear(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    jobs: &[[u32; 4]],
    dirty: &[u32],
    pages: &[u32],
) -> Vec<f32> {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("depth_clear"),
        source: wgpu::ShaderSource::Wgsl(CLEAR_WGSL.into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            storage_entry(0, true, wgpu::ShaderStages::VERTEX),
            storage_entry(1, true, wgpu::ShaderStages::VERTEX),
            storage_entry(2, true, wgpu::ShaderStages::VERTEX),
        ],
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        })),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: None,
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Always),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });
    let flat: Vec<u32> = jobs.iter().flatten().copied().collect();
    let jobs_buf = storage(device, &flat);
    let dirty_buf = storage(device, dirty);
    let pages_buf = storage(device, pages);
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: jobs_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: dirty_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: pages_buf.as_entire_binding(),
            },
        ],
    });
    let atlas = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: ATLAS,
            height: ATLAS,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = atlas.create_view(&Default::default());
    let out = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (ATLAS * ATLAS * 4) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(BACKGROUND),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.draw(0..6, 0..64 * (jobs.len() as u32 - 1));
    }
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &atlas,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::DepthOnly,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &out,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(ATLAS * 4),
                rows_per_image: Some(ATLAS),
            },
        },
        wgpu::Extent3d {
            width: ATLAS,
            height: ATLAS,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));
    read_buffer(device, queue, &out)
        .into_iter()
        .map(f32::from_bits)
        .collect()
}

fn depth(atlas: &[f32], x: u32, y: u32) -> f32 {
    atlas[(y * ATLAS + x) as usize]
}

#[test]
fn the_batched_clear_lands_in_each_faces_tile_and_honours_the_gates() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    // Face 0: ungated, tile (0, 0) size 64. Face 1: the dynamic layer, tile
    // (64, 0) size 32, only page 9 (column 1, row 1 from the bottom) marked.
    // Face 2: gated on the static bit, which is clear, tile (0, 64) size 64.
    // Face 3: gated on the static bit, which is set, tile (64, 64) size 64.
    let tile = |x: u32, y: u32| x | (y << 16);
    let jobs = [
        [ATLAS, 0, 0, 0],
        [0, 0, tile(0, 0), 64],
        [1, 1, tile(64, 0), 32],
        [2, 2, tile(0, 64), 64],
        [3, 2, tile(64, 64), 64],
    ];
    let dirty = [0, 1, 0, 2];
    // Two words per face: rows 0-3 then rows 4-7, bit `row * 8 + column`.
    let mut pages = vec![0u32; 8];
    pages[2] = 1 << 9;
    let atlas = run_clear(&device, &queue, &jobs, &dirty, &pages);

    // Face 0: its whole tile, and not one texel beyond it.
    for (x, y) in [(0, 0), (63, 0), (0, 63), (63, 63), (31, 40)] {
        assert_eq!(depth(&atlas, x, y), 1.0, "face 0 texel ({x}, {y})");
    }
    // Face 1: only page 9. A page is 4 texels; rows count up from the bottom
    // of the tile, so row 1 of 8 spans tile rows 24..28 from the top.
    for (x, y) in [(68, 24), (71, 27), (68, 27), (71, 24)] {
        assert_eq!(depth(&atlas, x, y), 1.0, "face 1 page 9 texel ({x}, {y})");
    }
    for (x, y) in [(64, 0), (67, 24), (72, 24), (68, 23), (68, 28), (95, 31)] {
        assert_eq!(
            depth(&atlas, x, y),
            BACKGROUND,
            "face 1 outside page 9 ({x}, {y})"
        );
    }
    // Face 2 is not dirty; face 3 is.
    for (x, y) in [(0, 64), (63, 127), (30, 100)] {
        assert_eq!(
            depth(&atlas, x, y),
            BACKGROUND,
            "clean face 2 texel ({x}, {y})"
        );
    }
    for (x, y) in [(64, 64), (127, 127), (100, 90)] {
        assert_eq!(depth(&atlas, x, y), 1.0, "dirty face 3 texel ({x}, {y})");
    }
    // Between the tiles nothing is touched.
    for (x, y) in [(64, 40), (100, 40), (127, 63)] {
        assert_eq!(depth(&atlas, x, y), BACKGROUND, "gap texel ({x}, {y})");
    }
}

#[test]
fn retiring_publishes_the_matrix_and_clears_only_the_serviced_faces() {
    let Some((device, queue)) = device() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    const FACES: u32 = 4;
    // Faces per region of the count buffer.
    const REGION: u32 = 4;
    let serviced = [1u32, 3];
    let mut jobs = vec![serviced.len() as u32, REGION];
    jobs.extend(serviced);
    // `LightMatrix` is 24 words: 16 of matrix, then atlas and policy.
    let desired: Vec<u32> = (0..FACES * 24).map(|i| 1000 + i).collect();
    let published: Vec<u32> = (0..FACES * 24).map(|i| 5000 + i).collect();
    let dirty = vec![7u32; FACES as usize];
    let pages = vec![9u32; FACES as usize * 2];
    let counts = vec![11u32; 3 * REGION as usize];

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("retire"),
        source: wgpu::ShaderSource::Wgsl(RETIRE_WGSL.into()),
    });
    let compute = wgpu::ShaderStages::COMPUTE;
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            storage_entry(0, true, compute),
            storage_entry(1, true, compute),
            storage_entry(2, false, compute),
            storage_entry(3, false, compute),
            storage_entry(4, false, compute),
            storage_entry(5, false, compute),
        ],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(
            &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            }),
        ),
        module: &module,
        entry_point: Some("retire"),
        compilation_options: Default::default(),
        cache: None,
    });
    let buffers = [&jobs, &desired, &published, &dirty, &pages, &counts].map(|w| storage(&device, w));
    let entries: Vec<wgpu::BindGroupEntry> = buffers
        .iter()
        .enumerate()
        .map(|(i, buffer)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: buffer.as_entire_binding(),
        })
        .collect();
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &entries,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    queue.submit(Some(encoder.finish()));

    let published_after = read_buffer(&device, &queue, &buffers[2]);
    for face in 0..FACES as usize {
        let was_serviced = serviced.contains(&(face as u32));
        for word in 0..24 {
            let at = face * 24 + word;
            let expected = if was_serviced && word < 16 {
                desired[at]
            } else {
                published[at]
            };
            assert_eq!(published_after[at], expected, "face {face} word {word}");
        }
    }
    let dirty_after = read_buffer(&device, &queue, &buffers[3]);
    let pages_after = read_buffer(&device, &queue, &buffers[4]);
    let counts_after = read_buffer(&device, &queue, &buffers[5]);
    for face in 0..FACES as usize {
        let cleared = serviced.contains(&(face as u32));
        assert_eq!(dirty_after[face], if cleared { 0 } else { 7 }, "dirty {face}");
        for word in 0..2 {
            assert_eq!(
                pages_after[face * 2 + word],
                if cleared { 0 } else { 9 },
                "pages {face}"
            );
        }
        for region in 0..3 {
            let at = region * REGION as usize + face;
            assert_eq!(
                counts_after[at],
                if cleared { 0 } else { 11 },
                "count {region} {face}"
            );
        }
    }
}
