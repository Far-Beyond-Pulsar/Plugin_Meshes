//! The shared, host-facing handle to virtual-geometry streaming: the
//! [`VgAssetStore`] plus the settings the passes read each frame and the
//! assets the host asked for.
//!
//! One [`VgRuntime`] is created by the host (editor / game) before the
//! renderer. The host
//!
//! * pushes the global settings ([`VgRuntime::set_settings`]),
//! * registers each mesh whose `.vgeo` is built by its content id, the one
//!   identity a mesh instance carries on the GPU for its virtual geometry
//!   ([`VgRuntime::request_asset`]; `vg_asset_ready` rows are keyed by it),
//!
//! and the renderer side (the `vg_asset_ready` scene derivation and the graph
//! passes) service it: the store is created lazily on the first
//! [`VgRuntime::service`] call with the renderer's device, pending assets are
//! registered, and [`VgRuntime::update`] runs the streaming step once per
//! frame before the passes record. The graph says whether the renderer it
//! builds for can draw virtual geometry at all
//! ([`VgRuntime::set_renderer_support`]): a host keeps classic geometry
//! where it cannot.
//!
//! Everything is behind one mutex; every call is short and none blocks on the
//! GPU.

use crate::core::{VgStreamConfig, VgStreamError};
use crate::source::VgeoSource;
use crate::store::{VgAssetHandle, VgAssetStore};
use helio_vg_format::{VgAssetReadyRow, VgMode};
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

/// Settings the passes read every frame (from `VirtualGeometrySettings`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VgRuntimeSettings {
    pub mode: VgMode,
    /// Screen-space error threshold in pixels: the one thing that drives LOD.
    pub tau_px: f32,
    /// Prefetch band: child pages are requested while the projected error exceeds
    /// `tau_px / prefetch_factor`, so fine pages are resident before the cut needs
    /// them (1 = request only when the cut needs the page).
    pub prefetch_factor: f32,
    /// Overdraw valve: triangles per covered pixel one 16x16 screen tile may hold
    /// before the groups covering it are coarsened (0 = off). Per tile, bounded
    /// (by at most `helio_pass_vg_cull::TILE_SCALE_MAX`), dense tiles included;
    /// it only engages on pathological overdraw.
    pub tile_density_limit: f32,
    /// Divides the dense-tile classifier's thresholds (64 triangles or 64 pixel
    /// hits per 16x16 tile): higher classifies a tile dense sooner.
    pub dense_sensitivity: f32,
    /// Page pool budget in MiB (applied when the store is created).
    pub page_budget_mb: u32,
    /// Ceiling on visible clusters per frame (a ceiling, not an allocation).
    pub max_visible_clusters: u32,
    pub two_phase_occlusion: bool,
    /// Tier 3 (dense tiles: ray tracing where ready, depth peel on the no-int64
    /// Tier 2 fallback). Off by default: enabled but idle it costs under 0.02 ms,
    /// but on an RTX 4090 it measured no gain over Tier 2 (VG_STRESS.md).
    pub tier3: bool,
}

impl Default for VgRuntimeSettings {
    fn default() -> Self {
        Self {
            mode: VgMode::Auto,
            tau_px: 1.0,
            prefetch_factor: 2.0,
            tile_density_limit: helio_vg_format::DEFAULT_TILE_DENSITY_LIMIT,
            dense_sensitivity: 1.0,
            page_budget_mb: 512,
            max_visible_clusters: 4_194_304,
            two_phase_occlusion: true,
            tier3: false,
        }
    }
}

struct AssetEntry {
    content: [u64; 2],
    source: Option<Arc<dyn VgeoSource>>,
    handle: Option<VgAssetHandle>,
    failed: bool,
}

/// One asset of [`VgRuntime::assets`].
#[derive(Clone)]
pub struct VgAssetInfo {
    pub content: [u64; 2],
    pub source: Arc<dyn VgeoSource>,
}

/// What a host reports of one of its assets for [`VgRuntime::asset_memory`]:
/// a name to show, and the bytes of the asset's classic geometry the host
/// keeps on the GPU beside its pages.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VgAssetReport {
    pub name: String,
    /// The full classic geometry (vertices and indices), while any instance
    /// draws it.
    pub classic_bytes: u64,
    /// The error-bounded classic stand-in instances draw while their virtual
    /// geometry is not drawn, and that shadows and rays use.
    pub fallback_bytes: u64,
}

/// Where one asset's memory is ([`VgRuntime::asset_memory`]).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VgAssetMemory {
    pub content: [u64; 2],
    pub report: VgAssetReport,
    /// Its resident pages' share of the page pool.
    pub page_bytes: u64,
}

#[derive(Default)]
struct State {
    store: Option<VgAssetStore>,
    assets: Vec<AssetEntry>,
    reports: HashMap<[u64; 2], VgAssetReport>,
}

/// Whether the renderer a runtime's passes are built for draws virtual
/// geometry ([`VgRuntime::set_renderer_support`]).
const SUPPORT_UNKNOWN: u8 = 0;
const SUPPORT_YES: u8 = 1;
const SUPPORT_NO: u8 = 2;

struct Inner {
    settings: Mutex<VgRuntimeSettings>,
    state: Mutex<State>,
    /// Bumped on every host-side change to the asset list or settings.
    host_version: AtomicU64,
    /// Bumped whenever [`VgRuntime::assets`] may change.
    assets_version: AtomicU64,
    support: AtomicU8,
    tier3: Mutex<VgTier3Stats>,
}

/// See the module doc. Cloning shares the same runtime.
#[derive(Clone)]
pub struct VgRuntime(Arc<Inner>);

impl Default for VgRuntime {
    fn default() -> Self {
        Self::new(VgRuntimeSettings::default())
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

/// Content id (`u128`) <-> the `[low, high]` words `.vgeo` headers carry.
pub fn content_words(content_id: u128) -> [u64; 2] {
    [content_id as u64, (content_id >> 64) as u64]
}

impl VgRuntime {
    pub fn new(settings: VgRuntimeSettings) -> Self {
        Self(Arc::new(Inner {
            settings: Mutex::new(settings),
            state: Mutex::new(State::default()),
            host_version: AtomicU64::new(1),
            assets_version: AtomicU64::new(1),
            support: AtomicU8::new(SUPPORT_UNKNOWN),
            tier3: Mutex::new(VgTier3Stats::default()),
        }))
    }

    /// Set by the graph builder: whether the renderer it builds the passes
    /// for can draw virtual geometry (its device has what the passes need).
    pub fn set_renderer_support(&self, supported: bool) {
        self.0.support.store(
            if supported { SUPPORT_YES } else { SUPPORT_NO },
            Ordering::Relaxed,
        );
    }

    /// Whether virtual geometry is drawn: the mode is not `Off` and the
    /// renderer's passes were built on a device that supports them. False
    /// until a graph was built for the runtime: until then, and wherever it
    /// stays false, instances draw classic geometry only.
    pub fn draws_virtual_geometry(&self) -> bool {
        self.settings().mode != VgMode::Off
            && self.0.support.load(Ordering::Relaxed) == SUPPORT_YES
    }

    pub fn settings(&self) -> VgRuntimeSettings {
        *lock(&self.0.settings)
    }

    pub fn set_settings(&self, settings: VgRuntimeSettings) {
        let mut current = lock(&self.0.settings);
        if *current != settings {
            *current = settings;
            self.0.host_version.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Ask for `content_id`'s `.vgeo` to be streamed. The content id is the
    /// asset's identity on the GPU: an instance names it, and the
    /// `vg_asset_ready` rows map it to the asset's slot. Idempotent.
    pub fn request_asset(&self, content_id: u128, source: Arc<dyn VgeoSource>) {
        let content = content_words(content_id);
        let mut state = lock(&self.0.state);
        if state.assets.iter().any(|e| e.content == content) {
            return;
        }
        state.assets.push(AssetEntry {
            content,
            source: Some(source),
            handle: None,
            failed: false,
        });
        self.0.host_version.fetch_add(1, Ordering::Relaxed);
        self.0.assets_version.fetch_add(1, Ordering::Relaxed);
    }

    /// The host's report of `content_id` ([`VgAssetReport`]), shown by
    /// [`Self::asset_memory`]. Cheap; an unchanged report changes nothing.
    pub fn report_asset(&self, content_id: u128, report: VgAssetReport) {
        let content = content_words(content_id);
        let mut state = lock(&self.0.state);
        if state.reports.get(&content) != Some(&report) {
            state.reports.insert(content, report);
        }
    }

    /// Every requested asset's memory: what its host reported and its
    /// resident pages, largest first. For tools (the Memory Explorer); it
    /// walks each asset's page range, so call it at a tool's pace, not per
    /// frame.
    pub fn asset_memory(&self) -> Vec<VgAssetMemory> {
        let state = lock(&self.0.state);
        let mut rows: Vec<VgAssetMemory> = state
            .assets
            .iter()
            .map(|e| VgAssetMemory {
                content: e.content,
                report: state.reports.get(&e.content).cloned().unwrap_or_default(),
                page_bytes: match (e.handle, state.store.as_ref()) {
                    (Some(handle), Some(store)) => store.resident_page_bytes(handle),
                    _ => 0,
                },
            })
            .collect();
        rows.sort_by_key(|row| {
            std::cmp::Reverse(row.page_bytes + row.report.classic_bytes + row.report.fallback_bytes)
        });
        rows
    }

    /// Whether `content_id` was requested (registered or pending).
    pub fn has_asset(&self, content_id: u128) -> bool {
        let content = content_words(content_id);
        lock(&self.0.state)
            .assets
            .iter()
            .any(|e| e.content == content)
    }

    /// Every requested asset that was not rejected, registered or pending:
    /// its content id and bytes. For consumers that read assets on the CPU,
    /// such as a scene TLAS's proxies.
    pub fn assets(&self) -> Vec<VgAssetInfo> {
        let state = lock(&self.0.state);
        state
            .assets
            .iter()
            .filter(|e| !e.failed)
            .filter_map(|e| {
                let source = match (&e.source, e.handle, state.store.as_ref()) {
                    (Some(source), _, _) => source.clone(),
                    (None, Some(handle), Some(store)) => {
                        store.core().asset_source(handle.slot())?.0
                    }
                    _ => return None,
                };
                Some(VgAssetInfo {
                    content: e.content,
                    source,
                })
            })
            .collect()
    }

    /// Changes whenever [`Self::assets`] may have (assets requested or
    /// released).
    pub fn assets_version(&self) -> u64 {
        self.0.assets_version.load(Ordering::Relaxed)
    }

    /// Stop streaming `content_id` and free its pages.
    pub fn release_asset(&self, content_id: u128) {
        let content = content_words(content_id);
        let mut state = lock(&self.0.state);
        let Some(index) = state.assets.iter().position(|e| e.content == content) else {
            return;
        };
        let entry = state.assets.remove(index);
        state.reports.remove(&content);
        if let (Some(handle), Some(store)) = (entry.handle, state.store.as_mut()) {
            store.release(handle);
        }
        self.0.host_version.fetch_add(1, Ordering::Relaxed);
        self.0.assets_version.fetch_add(1, Ordering::Relaxed);
    }

    /// Create the store on first use and register pending assets. Cheap when
    /// idle. Call from the renderer side (derivation or pass `prepare`).
    pub fn service(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let budget = self.settings().page_budget_mb;
        let mut state = lock(&self.0.state);
        if state.store.is_none() {
            let cfg = VgStreamConfig {
                budget_mb: budget,
                ..VgStreamConfig::default()
            };
            state.store = Some(VgAssetStore::new(device, queue, cfg));
        }
        let State { store, assets, .. } = &mut *state;
        let store = store.as_mut().expect("store created above");
        for entry in assets
            .iter_mut()
            .filter(|e| e.handle.is_none() && !e.failed)
        {
            let Some(source) = entry.source.clone() else {
                continue;
            };
            match store.register(source) {
                Ok(handle) => {
                    entry.handle = Some(handle);
                    // The CPU bytes live in the store now.
                    entry.source = None;
                }
                Err(error @ (VgStreamError::Source(_) | VgStreamError::Invalid(_))) => {
                    log::warn!("virtual geometry: asset rejected: {error}");
                    entry.failed = true;
                    self.0.assets_version.fetch_add(1, Ordering::Relaxed);
                }
                Err(error) => {
                    log::warn!("virtual geometry: asset not registered: {error}");
                    entry.failed = true;
                    self.0.assets_version.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }

    /// The per-frame streaming step (`VgAssetStore::update`). No-op before
    /// the store exists.
    pub fn update(&self) {
        if let Some(store) = lock(&self.0.state).store.as_mut() {
            store.update();
        }
    }

    /// Run `f` over the store, if it exists.
    pub fn with_store<R>(&self, f: impl FnOnce(&mut VgAssetStore) -> R) -> Option<R> {
        lock(&self.0.state).store.as_mut().map(f)
    }

    /// Whether the store exists.
    pub fn has_store(&self) -> bool {
        lock(&self.0.state).store.is_some()
    }

    /// `vg_asset_ready` rows for the scene derivation, and a version that
    /// changes whenever they may have (assets added/ready, settings).
    pub fn asset_ready_rows(&self, min_rows: usize) -> (Vec<VgAssetReadyRow>, u64) {
        let state = lock(&self.0.state);
        let host = self.0.host_version.load(Ordering::Relaxed);
        let Some(store) = state.store.as_ref() else {
            return (vec![VgAssetReadyRow::PADDING; min_rows.max(1)], host);
        };
        let contents: Vec<[u64; 2]> = state
            .assets
            .iter()
            .filter(|e| e.handle.is_some())
            .map(|e| e.content)
            .collect();
        (
            store.asset_ready_rows(&contents, min_rows),
            host.wrapping_mul(0x9e37_79b9)
                .wrapping_add(store.content_version()),
        )
    }
}

// ── Handle table ────────────────────────────────────────────────────────────
//
// `helio::RendererConfig` is `Copy`, so it carries a runtime as an opaque id
// (`RendererConfig::with_virtual_geometry`) that the graph builder resolves
// here. The table holds weak references: dropping the host's last
// [`VgRuntime`] ends the runtime.

use std::collections::HashMap;
use std::sync::{OnceLock, Weak};

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

fn table() -> &'static Mutex<HashMap<u64, Weak<Inner>>> {
    static TABLE: OnceLock<Mutex<HashMap<u64, Weak<Inner>>>> = OnceLock::new();
    TABLE.get_or_init(|| Mutex::new(HashMap::new()))
}

impl VgRuntime {
    /// An id that [`VgRuntime::from_handle_id`] resolves back to this runtime
    /// while any clone of it is alive. Stable per runtime.
    pub fn handle_id(&self) -> u64 {
        let mut table = lock(table());
        table.retain(|_, weak| weak.strong_count() > 0);
        if let Some((&id, _)) = table.iter().find(|(_, weak)| {
            weak.upgrade()
                .is_some_and(|inner| Arc::ptr_eq(&inner, &self.0))
        }) {
            return id;
        }
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        table.insert(id, Arc::downgrade(&self.0));
        id
    }

    /// The runtime registered under `id` by [`VgRuntime::handle_id`], if it
    /// is still alive.
    pub fn from_handle_id(id: u64) -> Option<Self> {
        lock(table()).get(&id).and_then(Weak::upgrade).map(Self)
    }

    /// Every live runtime that has a [`Self::handle_id`] (the ones a
    /// renderer was configured with), for tools that report across
    /// renderers.
    pub fn live() -> Vec<Self> {
        let mut table = lock(table());
        table.retain(|_, weak| weak.strong_count() > 0);
        let mut ids: Vec<u64> = table.keys().copied().collect();
        ids.sort_unstable();
        ids.iter()
            .filter_map(|id| table.get(id).and_then(Weak::upgrade).map(Self))
            .collect()
    }

    /// Whether any registered asset has its root pages resident.
    pub fn any_ready(&self) -> bool {
        let state = lock(&self.0.state);
        let Some(store) = state.store.as_ref() else {
            return false;
        };
        state
            .assets
            .iter()
            .any(|e| e.handle.is_some_and(|h| store.is_ready(h)))
    }
}

// ── Tier 3 statistics ───────────────────────────────────────────────────────

/// Counters of the Tier-3 (dense overlap) backend selection, written by the
/// `VgDense` pass and read by tools and tests. Cheap to copy.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VgTier3Stats {
    /// Frames Tier 3 ran with the depth-peel backend.
    pub frames_nonrt: u64,
    /// Frames Tier 3 ran with the ray-traced backend.
    pub frames_rt: u64,
    /// Frames the candidate/proxy machinery of the RT tier was active
    /// (zero under `ForceNonRtFallback` or without hardware support).
    pub rt_tier_frames: u64,
    /// The device supports ray queries (`rt_available`).
    pub rt_hardware: bool,
    /// Dense tiles of the most recent frame the CPU learned about (lagged).
    pub dense_tiles: u32,
    /// Distinct dense instances seen by the latest completed readback.
    pub candidates: u32,
    /// Instances in the current TLAS.
    pub tlas_instances: u32,
    /// TLAS builds recorded / skipped because nothing changed.
    pub tlas_rebuilds: u64,
    pub tlas_reuses: u64,
    /// BLAS builds recorded (one per (asset, proxy level) first use).
    pub blas_builds: u64,
    /// Frames the selector wanted RT but proxies/instances were incomplete.
    pub rt_incomplete_frames: u64,
    /// Readbacks that reported dense instances missing from the TLAS.
    pub uncovered_readbacks: u64,
    /// Assets with GPU-resident proxies and the bytes they hold.
    pub proxy_assets: u32,
    pub proxy_bytes: u64,
    pub proxy_jobs_started: u64,
    pub proxy_jobs_failed: u64,
    pub proxy_evictions: u64,
    /// Over-tau instances in the latest RT frame (proxy coarser than tau).
    pub over_tau_instances: u32,
    /// Candidate list overflowed (more dense instances than the readback holds).
    pub candidate_overflows: u64,
    /// Latest Tier-3 counters (read back a few frames late): tiles resolved,
    /// cluster-list entries and dense tiles dropped by the caps, clamped peels.
    pub dense_tiles_processed: u32,
    pub bin_overflow: u32,
    pub slot_overflow: u32,
    pub clamped_tiles: u32,
    pub clamped_tris: u32,
    /// Peel layers summed over the tiles of the latest counters readback.
    pub peel_layers: u32,
    /// Tier 3 decided to take clusters from Tier 2 next frame (the cull diverts
    /// them to the dense bin only while this holds; set by the `VgDense` pass).
    pub engaged: bool,
    /// Tier 2 runs the tiled no-int64 fallback (its per-tile lists can overflow):
    /// the depth peel is then useful on its own. Set when the graph is built.
    pub tier2_tile_fallback: bool,
    pub list_total: u32,
    pub list_max: u32,
    pub tris_total: u32,
    pub tris_max: u32,
    pub masked_tiles: u32,
    pub masked_tris: u32,
    pub masked_tris_max: u32,
    /// CPU microseconds `RtTier::prepare` took on the latest frame.
    pub rt_prepare_cpu_us: u32,
}

impl VgRuntime {
    /// The latest Tier-3 counters.
    pub fn tier3_stats(&self) -> VgTier3Stats {
        *lock(&self.0.tier3)
    }

    /// Update the Tier-3 counters (the `VgDense` pass).
    pub fn update_tier3_stats(&self, f: impl FnOnce(&mut VgTier3Stats)) {
        f(&mut lock(&self.0.tier3));
    }
}
