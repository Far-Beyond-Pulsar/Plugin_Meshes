//! The meshes' passes are nodes of render graphs: each registered by the
//! plugin's root through its render features, pins found by the probe from
//! what the passes declare, placed in the default deferred graph where the
//! pass order has always been, built from a graph asset, and absent, with
//! the reason, while the plugin is disabled.

// The plugin as a game links it: its root, with every registration.
use meshes_plugin as _;

use std::sync::{Arc, Mutex};

use engine_backend::subsystems::render::helio_renderer::{pipelines, render_features};
use helio::{DebugDrawState, PassBuildContext, RendererConfig};
use helio_graph_asset::{GraphAsset, NodeInstance, NodeRegistry, PinKind, validate};
use pulsar_scenedb::gpu::{EngineGpuContext, GpuMirrorHandle, SceneGpuConfig, SceneGpuStore};

/// Every mesh node: kind, the feature that registers it, the pass type its
/// build adds (`None`: built only with another plugin's settings or feature).
const NODES: &[(&str, &str, Option<&str>)] = &[
    (
        "helio.pass.object_batch",
        "helio.mesh.batch",
        Some("ObjectBatchPass"),
    ),
    (
        "helio.pass.indirect_dispatch",
        "helio.mesh.cull",
        Some("IndirectDispatchPass"),
    ),
    (
        "helio.pass.occlusion_cull",
        "helio.mesh.cull",
        Some("OcclusionCullPass"),
    ),
    ("helio.pass.shadow_dirty", "helio.mesh.shadow", None),
    ("helio.pass.shadow", "helio.mesh.shadow", None),
    (
        "helio.pass.gbuffer",
        "helio.mesh.gbuffer",
        Some("GBufferPass"),
    ),
    ("helio.pass.forward_lit", "helio.mesh.forward", None),
    (
        "helio.pass.transparent",
        "helio.mesh.transparent",
        Some("TransparentPass"),
    ),
    (
        "helio.pass.virtual_geometry",
        "helio.vg",
        Some("VgStreamBeginPass"),
    ),
];

/// The plugin gate is process-wide state and a wgpu device is one at a time
/// (Pulsar-Native#1647): every test takes this first.
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

fn device() -> Option<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(helio_core::wgpu_instance_descriptor());
        let adapter = instance.request_adapter(&Default::default()).await.ok()?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: helio::required_wgpu_features(adapter.features()),
                required_limits: helio::required_wgpu_limits(adapter.limits()),
                experimental_features: helio::required_experimental_features(adapter.features()),
                ..Default::default()
            })
            .await
            .ok()?;
        Some((Arc::new(device), Arc::new(queue)))
    })
}

fn buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::UNIFORM
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn configured() -> RendererConfig {
    // The renderer config with the plugins' settings, as a host makes it.
    render_features::configure(RendererConfig::new(
        320,
        180,
        wgpu::TextureFormat::Rgba8Unorm,
    ))
}

fn registry() -> NodeRegistry {
    helio_presets::graphs::registry_with(&render_features::plugin_features())
}

/// The default deferred graph with its sections expanded: every node in
/// execution order.
fn asset_of(registry: &NodeRegistry) -> GraphAsset {
    let sectioned = helio_presets::graphs::deferred_graph_asset(registry);
    helio_graph_asset::sections::expand(&sectioned).asset
}

/// Compiles `asset` and lists the type names of its passes (`None`: no GPU).
fn pass_types(asset: &GraphAsset) -> Option<Vec<String>> {
    let (device, queue) = device()?;
    let gpu = EngineGpuContext::new(Arc::clone(&device), Arc::clone(&queue));
    let store = SceneGpuStore::new(
        &gpu,
        SceneGpuConfig {
            classes: Vec::new(),
            tombstone_headroom: 0,
            max_cells_metadata: 0,
        },
    );
    let camera = buffer(&device, 512);
    let debug_camera = buffer(&device, 64);
    let cull_stats = buffer(&device, 64);
    let mut env = helio_graph_asset::build_env(&PassBuildContext {
        device: &device,
        queue: &queue,
        config: configured(),
        debug_state: Arc::new(Mutex::new(DebugDrawState::default())),
        camera_buffer: &camera,
        debug_camera_buffer: &debug_camera,
        cull_stats_buffer: &cull_stats,
        owns_device: false,
        scene_db: GpuMirrorHandle::new(Arc::new(store), Arc::clone(&queue)),
    });
    let graph = helio_graph_asset::compile(asset, &registry(), &mut env)
        .unwrap_or_else(|errors| panic!("the graph builds: {errors:?}"));
    Some(
        (0..graph.pass_count())
            .filter_map(|index| graph.pass_declaration(index))
            .map(|pass| pass.type_name.to_string())
            .collect(),
    )
}

#[test]
fn every_mesh_pass_is_a_registered_node() {
    let _serial = serial();
    pulsar_plugins::deactivate();
    let registry = registry();
    for (kind, feature, _) in NODES {
        let node = registry
            .get(kind)
            .unwrap_or_else(|| panic!("{kind} is not registered"));
        assert_eq!(registry.feature_of(kind), Some(*feature), "{kind}");
        assert!(
            node.generated,
            "{kind}: pins come from what the pass declares"
        );
        assert!(node.params.iter().any(|p| p.key == "enabled"), "{kind}");
        assert!(node.placement.is_some(), "{kind}: the preset places it");
        assert!(
            node.title.contains(": "),
            "{kind}: title reads `Group: Action`"
        );
    }
}

/// The meshes' nodes in the default deferred graph, in execution order: the
/// batches first, the dispatch arguments and culling around the depth
/// pyramid, the shadow faces, the G-buffer with virtual geometry after it,
/// transparency over the colour.
#[test]
fn the_default_graph_keeps_the_pass_order() {
    let _serial = serial();
    pulsar_plugins::deactivate();
    let asset = asset_of(&registry());
    let order: Vec<&str> = asset
        .nodes
        .iter()
        .map(|node| node.kind.as_str())
        .filter(|kind| NODES.iter().any(|(k, ..)| k == kind))
        .collect();
    eprintln!("MESH_NODE_ORDER {order:?}");
    let before = |a: &str, b: &str| {
        let position = |kind: &str| order.iter().position(|k| *k == kind);
        match (position(a), position(b)) {
            (Some(a), Some(b)) => a < b,
            _ => panic!("{a} and {b} are both in the default graph: {order:?}"),
        }
    };
    assert!(before(
        "helio.pass.object_batch",
        "helio.pass.indirect_dispatch"
    ));
    assert!(before(
        "helio.pass.indirect_dispatch",
        "helio.pass.occlusion_cull"
    ));
    assert!(before(
        "helio.pass.occlusion_cull",
        "helio.pass.shadow_dirty"
    ));
    assert!(before("helio.pass.shadow_dirty", "helio.pass.shadow"));
    assert!(before("helio.pass.shadow", "helio.pass.gbuffer"));
    assert!(before("helio.pass.gbuffer", "helio.pass.virtual_geometry"));
    assert!(before(
        "helio.pass.virtual_geometry",
        "helio.pass.transparent"
    ));
}

#[test]
fn the_graph_builds_the_nodes_and_enabled_off_bypasses_them() {
    let _serial = serial();
    pulsar_plugins::deactivate();
    let mut asset = asset_of(&registry());
    let Some(types) = pass_types(&asset) else {
        eprintln!("GPU_VALIDATION_SKIPPED_NO_ADAPTER: mesh graph nodes");
        return;
    };
    for (kind, _, pass) in NODES {
        let Some(pass) = pass else { continue };
        // Virtual geometry builds only where the device supports it.
        if *kind == "helio.pass.virtual_geometry" {
            continue;
        }
        assert!(
            types.iter().any(|t| t.contains(pass)),
            "{kind} builds {pass}: {types:?}"
        );
    }

    // The Enabled parameter bypasses a node: no pass.
    for node in asset
        .nodes
        .iter_mut()
        .filter(|n| n.kind == "helio.pass.object_batch")
    {
        node.params.insert("enabled".into(), false.into());
    }
    let after = pass_types(&asset).expect("the same device");
    assert!(!after.iter().any(|t| t.contains("ObjectBatchPass")));
}

#[test]
fn the_probe_gives_the_nodes_the_pins_their_passes_declare() {
    let _serial = serial();
    pulsar_plugins::deactivate();
    let registry = pipelines::graph_node_registry();
    if pipelines::graph_pin_probe().is_none() {
        eprintln!(
            "GPU_VALIDATION_SKIPPED_NO_ADAPTER: mesh node pins: {:?}",
            pipelines::graph_pin_probe_error()
        );
        return;
    }
    for (kind, _, pass) in NODES {
        let pins = registry
            .get(kind)
            .unwrap_or_else(|| panic!("{kind} is not registered"))
            .pins(&Default::default());
        // Built only with other plugins' settings (the lights'), or, for
        // virtual geometry, where the device supports it.
        if pass.is_none() || *kind == "helio.pass.virtual_geometry" {
            continue;
        }
        assert!(
            pins.inputs.iter().any(|p| p.ty.kind == PinKind::Flow),
            "{kind}: {pins:?}"
        );
        assert!(
            pins.outputs.iter().any(|p| p.ty.kind == PinKind::Flow),
            "{kind}: {pins:?}"
        );
    }
    let gbuffer = registry
        .get("helio.pass.gbuffer")
        .expect("registered")
        .pins(&Default::default());
    assert!(
        gbuffer.outputs.iter().any(|p| p.ty.kind != PinKind::Flow),
        "the G-buffer pass writes resources: {gbuffer:?}"
    );
    let vg = registry
        .get("helio.pass.virtual_geometry")
        .expect("registered")
        .pins(&Default::default());
    eprintln!("VG_NODE_PINS {vg:?}");
}

#[test]
fn a_disabled_plugin_has_no_nodes_and_a_graph_naming_one_says_so() {
    let _serial = serial();
    pulsar_plugins::deactivate();
    let with_plugin = registry();
    let mut asset = asset_of(&with_plugin);
    asset.nodes.push(NodeInstance {
        id: "batches".into(),
        kind: "helio.pass.object_batch".into(),
        params: Default::default(),
        position: [0.0; 2],
    });

    pulsar_plugins::activate(&pulsar_plugins::resolve(
        pulsar_plugins::installed(),
        &pulsar_plugins::ProjectPlugins::default().with(&["pulsar.meshes"], false),
    ));
    let registry = pipelines::graph_node_registry();
    pulsar_plugins::deactivate();

    for (kind, ..) in NODES {
        assert!(registry.get(kind).is_none(), "{kind} is unregistered");
        assert_eq!(
            registry.absence(kind),
            Some("plugin pulsar.meshes disabled"),
            "{kind}"
        );
    }
    let validation = validate(&asset, &registry);
    let messages: Vec<_> = validation.errors().map(|d| d.message.clone()).collect();
    assert!(
        messages.iter().any(|m| {
            m == "missing node `helio.pass.object_batch` (plugin pulsar.meshes disabled)"
        }),
        "{messages:?}"
    );
    // Enabled again, the registry follows.
    assert!(
        pipelines::graph_node_registry()
            .get("helio.pass.object_batch")
            .is_some()
    );
}
