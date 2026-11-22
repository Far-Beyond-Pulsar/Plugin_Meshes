//! Virtual-geometry instances in the scene's TLAS ([`super::ray_tracing`]),
//! through one proxy per asset.
//!
//! A standalone host's VG instance is an ordinary static object whose mesh
//! entity carries a [`VirtualGeometryMesh`] naming the asset's content id
//! (the identity `VgRuntime::request_asset` registered it under); its
//! classic mesh is a fallback or a zero-area placeholder, never the shape
//! rays should see, and the asset's full geometry is far too big for a
//! BLAS. So each asset is traced through ONE cut of its cluster DAG at an
//! explicit error bound ([`ProxyOptions::trace`]: at most
//! [`VgRayProxySettings::relative_error`] of its bounding radius off the
//! surface), built on worker threads from the asset's bytes ([`ProxyCache`])
//! the first time an object uses it, shared by all its instances. Its
//! triangles are grouped by the asset's material sections
//! ([`VgRayProxies::sections`], `helio_vg_stream::ProxySection`), so a
//! projection makes each section a BLAS segment with its own row and
//! material: a ray hit on it is shaded as its section's material is
//! (`RayInstance` rows flagged [`helio_mesh_api::RAY_INSTANCE_PROXY`], the
//! face's winding as its normal, the tangent from its uv0); its vertices
//! keep their uv0, so an alpha-tested material is tested on it, and its
//! textures read, as on a classic mesh.
//!
//! **The error bound is authoritative; memory is budgeted scene-wide.** A
//! proxy is never coarsened to fit a triangle cap. Its vertices and indices
//! count against one scene budget ([`VgRayProxySettings::budget_bytes`]);
//! a finished proxy that does not fit what the others left is not admitted:
//! its instances stay out of the TLAS (rays pass them, as any object the
//! projection cannot represent yet), it is logged once with what it needs,
//! and it is counted in [`VgRayStats::over_budget`] (and the frame stats,
//! `helio_mesh_api::stats::VG_RAY_PROXIES_OVER_BUDGET`). When proxies leave
//! or the budget grows, the ones that now fit are built again and admitted.
//!
//! The proxies live in one pool of packed vertices (position, then uv0) and
//! indices (the proxy pool a frame publishes, `helio_mesh_api::RAY_PROXY_VERTICES`), used directly as BLAS input
//! and by shaders that read hit triangles. It only grows by appending; when full it
//! is reallocated at twice the live size and the live proxies are copied in,
//! packed (their BLASes rebuild: their input moved). An asset no object uses
//! any more is dropped at the next build.
//!
//! Until an asset's proxy is built its instances are left out (rays pass
//! them), as any object the projection cannot represent yet.

use helio_vg_stream::{proxy_key, ProxyBuild, ProxyCache, ProxyOptions, ProxyState, VgRuntime};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Bytes of a pool vertex: a `Float32x3` position, then a `Float32x2` uv0.
const VERTEX_BYTES: u64 = helio_mesh_api::RAY_PROXY_VERTEX_BYTES;

/// Marks a standalone host's mesh entity as the stand-in of a
/// virtual-geometry asset: objects drawing the mesh are traced through the
/// asset's proxy. `content` is the content id the asset was requested under
/// (`VgRuntime::request_asset`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VirtualGeometryMesh {
    pub content: u128,
}

/// How a projection's proxies are cut and how much memory they may take.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VgRayProxySettings {
    /// Error bound of every proxy, a fraction of its asset's bounding radius
    /// (`ProxyOptions::DEFAULT_RELATIVE_ERROR` by default).
    pub relative_error: f32,
    /// Bytes of proxy vertices and indices the scene may hold.
    pub budget_bytes: u64,
}

impl Default for VgRayProxySettings {
    fn default() -> Self {
        Self {
            relative_error: ProxyOptions::DEFAULT_RELATIVE_ERROR,
            budget_bytes: 256 << 20,
        }
    }
}

/// Where one asset's proxy is in the pool.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProxyRange {
    pub first_vertex: u32,
    pub vertex_count: u32,
    pub first_index: u32,
    pub index_count: u32,
}

impl ProxyRange {
    fn bytes(&self) -> u64 {
        u64::from(self.vertex_count) * VERTEX_BYTES + u64::from(self.index_count) * 4
    }
}

/// What a mesh is to the projection.
pub enum Lookup {
    /// Not a virtual-geometry asset: traced as its classic mesh.
    Classic,
    /// A VG asset whose proxy is not built (or failed): left out.
    Pending,
    /// A VG asset whose proxy does not fit the scene budget: left out.
    OverBudget,
    /// Traced through its proxy, by key.
    Proxy(u64, ProxyRange),
}

/// Counters of the projection's virtual geometry.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VgRayStats {
    /// Assets with a proxy in the pool, and their triangles.
    pub proxies: u32,
    pub triangles: u64,
    /// Triangles of those assets' original geometry.
    pub source_triangles: u64,
    /// Proxy builds running or queued, and those that failed.
    pub pending: u32,
    pub failed: u32,
    /// Live bytes of vertices and indices, and the pool's allocated bytes.
    pub live_bytes: u64,
    pub pool_bytes: u64,
    /// The scene budget for live bytes.
    pub budget_bytes: u64,
    /// Proxies built at their bound that the budget could not admit, and
    /// the bytes they need.
    pub over_budget: u32,
    pub over_budget_bytes: u64,
}

struct Pool {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    vertex_capacity: u32,
    index_capacity: u32,
    vertex_end: u32,
    index_end: u32,
}

impl Pool {
    fn new(device: &wgpu::Device, vertex_capacity: u32, index_capacity: u32) -> Self {
        let buffer = |label, bytes: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(16),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::BLAS_INPUT
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        Self {
            vertices: buffer(
                "VG ray proxy vertices",
                u64::from(vertex_capacity) * VERTEX_BYTES,
            ),
            indices: buffer("VG ray proxy indices", u64::from(index_capacity) * 4),
            vertex_capacity,
            index_capacity,
            vertex_end: 0,
            index_end: 0,
        }
    }

    fn bytes(&self) -> u64 {
        self.vertices.size() + self.indices.size()
    }
}

struct Proxy {
    range: ProxyRange,
    /// Its material sections, their ranges relative to `range.first_index`.
    sections: Vec<helio_vg_stream::ProxySection>,
    source_triangles: u32,
}

/// The scene projection's VG proxies (module docs).
pub struct VgRayProxies {
    runtime: VgRuntime,
    settings: VgRayProxySettings,
    cache: ProxyCache,
    /// Runtime assets by content id: their proxy key and bytes.
    assets: HashMap<u128, (u64, Arc<dyn helio_vg_stream::VgeoSource>)>,
    assets_version: Option<u64>,
    proxies: HashMap<u64, Proxy>,
    /// Proxies the budget did not admit, by key: the bytes each needs.
    over_budget: HashMap<u64, u64>,
    /// Keys the last build used: kept pinned in the cache.
    live: Vec<u64>,
    pool: Option<Pool>,
    compact_pool: bool,
    frame: u64,
}

impl VgRayProxies {
    pub fn new(runtime: VgRuntime, settings: VgRayProxySettings) -> Self {
        Self {
            runtime,
            settings,
            // The scene budget is this projection's (`upload`); the cache
            // only builds and hands over.
            cache: ProxyCache::new(usize::MAX, ProxyOptions::trace(settings.relative_error)),
            assets: HashMap::new(),
            assets_version: None,
            proxies: HashMap::new(),
            over_budget: HashMap::new(),
            live: Vec::new(),
            pool: None,
            compact_pool: false,
            frame: 0,
        }
    }

    pub fn settings(&self) -> VgRayProxySettings {
        self.settings
    }

    /// Takes effect at the next [`Self::poll`]: a new error bound rebuilds
    /// every proxy; a new budget re-admits what now fits (a smaller one
    /// keeps what is in until it leaves).
    pub fn set_settings(&mut self, settings: VgRayProxySettings) {
        if settings == self.settings {
            return;
        }
        if settings.relative_error != self.settings.relative_error {
            self.cache = ProxyCache::new(usize::MAX, ProxyOptions::trace(settings.relative_error));
            self.proxies.clear();
            self.over_budget.clear();
            self.compact_pool = true;
        }
        self.settings = settings;
    }

    fn live_bytes(&self) -> u64 {
        self.proxies.values().map(|p| p.range.bytes()).sum()
    }

    /// Per frame, before the projection builds: follows the runtime's
    /// assets, collects finished proxy builds and uploads the ones the
    /// budget admits. True when the projection must rebuild (an asset or a
    /// proxy came or went).
    pub fn poll(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> bool {
        self.frame += 1;
        let mut changed = false;
        let version = self.runtime.assets_version();
        if self.assets_version != Some(version) {
            self.assets_version = Some(version);
            self.assets = self
                .runtime
                .assets()
                .into_iter()
                .map(|a| {
                    let content = u128::from(a.content[0]) | (u128::from(a.content[1]) << 64);
                    (content, (proxy_key(a.content), a.source))
                })
                .collect();
            changed = true;
        }
        for &key in &self.live {
            self.cache.touch(key, self.frame, 2);
        }
        for key in self.cache.poll(self.frame) {
            changed |= self.proxies.remove(&key).is_some();
        }
        let ready = self.cache.take_ready();
        if !ready.is_empty() {
            self.upload(device, queue, ready);
            changed = true;
        }
        changed |= self.readmit();
        if self.compact_pool && self.pool_is_sparse() {
            self.grow(device, queue, 0, 0);
            changed = true;
        }
        self.compact_pool = false;
        changed
    }

    /// Over-budget proxies that fit what is free now are built again (the
    /// build data was not kept) and admitted when they arrive.
    fn readmit(&mut self) -> bool {
        let mut free = self.settings.budget_bytes.saturating_sub(self.live_bytes());
        let mut fitting: Vec<(u64, u64)> = self
            .over_budget
            .iter()
            .map(|(key, bytes)| (*key, *bytes))
            .collect();
        fitting.sort_unstable_by_key(|(_, bytes)| *bytes);
        let mut any = false;
        for (key, bytes) in fitting {
            if bytes > free {
                break;
            }
            free -= bytes;
            self.over_budget.remove(&key);
            self.cache.forget(key);
            any = true;
        }
        any
    }

    /// What the mesh entity carrying `asset` (its [`VirtualGeometryMesh`],
    /// if any) is to the projection; a VG asset met for the first time
    /// starts its build.
    pub fn lookup(&mut self, asset: Option<VirtualGeometryMesh>) -> Lookup {
        let Some((key, source)) = asset.and_then(|asset| self.assets.get(&asset.content)) else {
            return if asset.is_some() {
                // Named, but not (yet) an asset of the runtime.
                Lookup::Pending
            } else {
                Lookup::Classic
            };
        };
        if let Some(proxy) = self.proxies.get(key) {
            return Lookup::Proxy(*key, proxy.range);
        }
        if self.over_budget.contains_key(key) {
            return Lookup::OverBudget;
        }
        if self.cache.state(*key).is_none() {
            self.cache.request(*key, source.clone(), self.frame);
        }
        Lookup::Pending
    }

    /// The material sections of proxy `key` (none for a proxy not in the
    /// pool): each section's triangles from `first_index` relative to its
    /// [`ProxyRange::first_index`].
    pub fn sections(&self, key: u64) -> &[helio_vg_stream::ProxySection] {
        self.proxies
            .get(&key)
            .map_or(&[], |p| p.sections.as_slice())
    }

    /// After a build: the proxies its objects use. The others are dropped.
    pub fn retain(&mut self, used: &HashSet<u64>) {
        self.live = used.iter().copied().collect();
        let unused: Vec<u64> = self
            .proxies
            .keys()
            .filter(|k| !used.contains(k))
            .copied()
            .collect();
        let had_unused = !unused.is_empty();
        for key in unused {
            self.proxies.remove(&key);
            self.cache.forget(key);
        }
        self.compact_pool |= had_unused;
    }

    fn pool_is_sparse(&self) -> bool {
        let Some(pool) = &self.pool else {
            return false;
        };
        let live_vertices: u32 = self.proxies.values().map(|p| p.range.vertex_count).sum();
        let live_indices: u32 = self.proxies.values().map(|p| p.range.index_count).sum();
        let target_vertices = live_vertices
            .max(1024)
            .next_power_of_two()
            .saturating_mul(2);
        let target_indices = live_indices.max(4096).next_power_of_two().saturating_mul(2);
        pool.vertex_capacity > target_vertices.saturating_mul(4)
            || pool.index_capacity > target_indices.saturating_mul(4)
    }

    /// The pool's vertices and indices (after the first upload).
    pub fn buffers(&self) -> Option<(&wgpu::Buffer, &wgpu::Buffer)> {
        self.pool.as_ref().map(|p| (&p.vertices, &p.indices))
    }

    pub fn stats(&self) -> VgRayStats {
        let mut stats = VgRayStats {
            proxies: self.proxies.len() as u32,
            budget_bytes: self.settings.budget_bytes,
            over_budget: self.over_budget.len() as u32,
            over_budget_bytes: self.over_budget.values().sum(),
            ..Default::default()
        };
        for proxy in self.proxies.values() {
            stats.triangles += u64::from(proxy.range.index_count / 3);
            stats.source_triangles += u64::from(proxy.source_triangles);
            stats.live_bytes += proxy.range.bytes();
        }
        for (key, _) in self.assets.values() {
            match self.cache.state(*key) {
                Some(ProxyState::Building) => stats.pending += 1,
                Some(ProxyState::Failed(_)) => stats.failed += 1,
                _ => {}
            }
        }
        stats.pool_bytes = self.pool.as_ref().map_or(0, Pool::bytes);
        stats
    }

    /// Admits finished builds the budget holds, appending them to the pool
    /// (growing and packing it when they do not fit); the others are
    /// recorded over budget.
    fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        ready: Vec<(u64, Arc<ProxyBuild>)>,
    ) {
        let mut free = self.settings.budget_bytes.saturating_sub(self.live_bytes());
        let mut admitted = Vec::new();
        for (key, build) in ready {
            let Some(lod) = build.lods.first() else {
                continue;
            };
            let bytes = lod.positions.len() as u64 * VERTEX_BYTES + lod.indices.len() as u64 * 4;
            if bytes > free {
                log::warn!(
                    "VG ray proxy {key:#018x}: {} triangles at error {:.4} ({:.2} % of its bounding radius {:.3}) need {} KiB; \
                     the scene's proxy budget ({} MiB, {} KiB free) cannot hold them, so its instances are not traced. \
                     Raise the budget or the proxies' error bound.",
                    lod.triangle_count(),
                    lod.error,
                    self.settings.relative_error * 100.0,
                    build.radius,
                    bytes / 1024,
                    self.settings.budget_bytes >> 20,
                    free / 1024,
                );
                self.over_budget.insert(key, bytes);
                continue;
            }
            free -= bytes;
            admitted.push((key, build));
        }
        if admitted.is_empty() {
            return;
        }
        let (new_vertices, new_indices) = admitted.iter().fold((0u32, 0u32), |(v, i), (_, b)| {
            (
                v + b.lods[0].positions.len() as u32,
                i + b.lods[0].indices.len() as u32,
            )
        });
        let fits = self.pool.as_ref().is_some_and(|p| {
            p.vertex_end + new_vertices <= p.vertex_capacity
                && p.index_end + new_indices <= p.index_capacity
        });
        if !fits {
            self.grow(device, queue, new_vertices, new_indices);
        }
        let pool = self.pool.as_mut().expect("pool allocated above");
        for (key, build) in admitted {
            let lod = &build.lods[0];
            let range = ProxyRange {
                first_vertex: pool.vertex_end,
                vertex_count: lod.positions.len() as u32,
                first_index: pool.index_end,
                index_count: lod.indices.len() as u32,
            };
            let vertices: Vec<[f32; 5]> = lod
                .positions
                .iter()
                .enumerate()
                .map(|(v, p)| {
                    let uv = lod.uvs.get(v).copied().unwrap_or([0.0; 2]);
                    [p[0], p[1], p[2], uv[0], uv[1]]
                })
                .collect();
            queue.write_buffer(
                &pool.vertices,
                u64::from(range.first_vertex) * VERTEX_BYTES,
                bytemuck::cast_slice(&vertices),
            );
            queue.write_buffer(
                &pool.indices,
                u64::from(range.first_index) * 4,
                bytemuck::cast_slice(&lod.indices),
            );
            pool.vertex_end += range.vertex_count;
            pool.index_end += range.index_count;
            log::info!(
                "VG ray proxy {key:#018x}: {} triangles of {} (error {:.4}, bounding radius {:.3}), {} KiB of vertices and indices",
                lod.triangle_count(),
                build.source_triangles,
                lod.error,
                build.radius,
                range.bytes() / 1024,
            );
            self.proxies.insert(
                key,
                Proxy {
                    range,
                    sections: lod.sections.clone(),
                    source_triangles: build.source_triangles,
                },
            );
        }
    }

    /// Reallocates the pool at twice what the live proxies and `extra` need,
    /// copying the live proxies in, packed.
    fn grow(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        extra_vertices: u32,
        extra_indices: u32,
    ) {
        let live_vertices: u32 = self.proxies.values().map(|p| p.range.vertex_count).sum();
        let live_indices: u32 = self.proxies.values().map(|p| p.range.index_count).sum();
        let vertices = (live_vertices + extra_vertices)
            .max(1024)
            .next_power_of_two()
            * 2;
        let indices = (live_indices + extra_indices).max(4096).next_power_of_two() * 2;
        let mut pool = Pool::new(device, vertices, indices);
        if let Some(old) = self.pool.take() {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("VG ray proxy pool growth"),
            });
            let mut keys: Vec<u64> = self.proxies.keys().copied().collect();
            keys.sort_unstable_by_key(|k| self.proxies[k].range.first_vertex);
            for key in keys {
                let range = &mut self.proxies.get_mut(&key).expect("listed above").range;
                encoder.copy_buffer_to_buffer(
                    &old.vertices,
                    u64::from(range.first_vertex) * VERTEX_BYTES,
                    &pool.vertices,
                    u64::from(pool.vertex_end) * VERTEX_BYTES,
                    u64::from(range.vertex_count) * VERTEX_BYTES,
                );
                encoder.copy_buffer_to_buffer(
                    &old.indices,
                    u64::from(range.first_index) * 4,
                    &pool.indices,
                    u64::from(pool.index_end) * 4,
                    u64::from(range.index_count) * 4,
                );
                range.first_vertex = pool.vertex_end;
                range.first_index = pool.index_end;
                pool.vertex_end += range.vertex_count;
                pool.index_end += range.index_count;
            }
            queue.submit([encoder.finish()]);
        }
        self.pool = Some(pool);
    }
}
