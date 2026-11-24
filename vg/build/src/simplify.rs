//! Group simplification using meshoptimizer.
//!
//! Collapse ordering and vertex placement come from meshoptimizer; the error
//! stored for a group is not taken from it but measured by `deviation`.
//!
//! A group's triangles fall into two classes:
//! * **anchored** geometry (connected surfaces that touch other groups, or big
//!   open/non-manifold components): group-boundary and non-manifold edges stay
//!   locked (bit-exact), so neighbouring groups never crack. Everything else is
//!   simplified with attribute-aware quadrics (normal, uv0, uv1, weighted into
//!   object-space length, see [`AttrWeights`]) and every collapse places the
//!   surviving vertex at the quadric optimum, updating its position and
//!   attributes (`meshopt_simplifyWithUpdate`). Attribute seams are not
//!   locked: the simplifier moves seam vertices only along their seam and
//!   keeps all wedges of a vertex at one position. Simplification stops at the
//!   group's error limit (see `dag`), so a hard region keeps its triangles
//!   instead of being forced to the target.
//! * **floating** pieces (small connected components that lie entirely inside
//!   the group, e.g. needle/leaf cards, branch fragments): nothing outside the
//!   group references them, so nothing is locked. Card-like pieces merge into
//!   area-preserving quads, so canopies coarsen into fewer, larger cards;
//!   larger pieces are reduced by edge collapse with island pruning. (Island
//!   pruning under an unbounded error always reaches the target, so a
//!   proximity-merging fallback never ran on any fixture and was removed;
//!   permissive simplification only relaxes attribute seams, which pruning
//!   does not stall on either.)
//!
//! Material sections are always simplified independently. Output references
//! existing vertices where they did not move and [`NewVertex`]es where the
//! simplifier placed one.

use crate::deviation::{uv_scale, Surface};
use crate::math::{add, cross, dist, dot, edge_key, length, normalize, scale, sub, V3};
use crate::mesh::{Tri, VertexStore};
use helio_vg_format::{VertexAttr, ATTR_STRIDE};
use meshopt::SimplifyVertex;
use std::collections::{BTreeMap, HashMap};

/// A connected component with at most this many triangles that touches no
/// other group is "floating": the triangles of the smallest group the
/// partitioner typically forms (a third under its target), so a floating
/// piece usually lies inside one group.
pub const FLOATING_MAX_TRIS: usize =
    crate::partition::TARGET_GROUP * 3 / 4 * helio_vg_format::CLUSTER_MAX_TRIS as usize;
/// Appearance error of a floating-dominated group (see
/// [`SimplifyResult::coverage_tris`]): `COVERAGE_FACTOR` x the spacing of its
/// output triangles, `r * cbrt(4pi/3 / n)` for `n` triangles in a group sphere
/// of radius `r`. Below that scale the extra triangles of a canopy add detail
/// the viewer cannot resolve (a node smaller than ~1/COVERAGE_FACTOR spacings
/// per pixel).
pub const COVERAGE_FACTOR: f32 = 0.05;
/// ... and never below this many linear footprints (sqrt of mean triangle area)
/// of the output: two samples per feature (Nyquist), so the group is swapped in
/// only once its triangles are at most half a pixel at the error threshold.
pub const COVERAGE_FOOTPRINT: f32 = 2.0;
/// A floating piece with at most this many triangles is a "card" (needle, leaf
/// quad, small polygon): it is merged with its neighbours into area-preserving
/// quads instead of being collapsed or pruned (see [`merge_cards`]).
const CARD_MAX_TRIS: usize = 6;
/// A merge cell whose members spread over more than this multiple of the median
/// cell radius is split further.
const CELL_RADIUS_SPREAD: f32 = 2.0;

/// Attribute record of a [`NewVertex`].
#[derive(Clone, Copy, Debug)]
pub enum NewAttr {
    /// An existing record (index for [`VertexStore::attr`]).
    Record(u32),
    /// A record the simplifier computed.
    Bytes([u8; ATTR_STRIDE as usize]),
}

/// Position id of a [`NewVertex`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NewPid {
    /// The vertex sits exactly on an existing vertex's position.
    Existing(u32),
    /// A new position; new vertices with the same wedge number share it
    /// (wedges of one placed vertex). Numbered from 0 per group.
    Wedge(u32),
}

/// A vertex placed by this simplification.
#[derive(Clone, Copy, Debug)]
pub struct NewVertex {
    pub pos: V3,
    pub attr: NewAttr,
    pub pid: NewPid,
    pub floating: bool,
}

pub struct SimplifyResult {
    pub tris: Vec<([u32; 3], u32)>,
    /// Vertices placed by this simplification; output corners reference them
    /// as `vs.len() + index`.
    pub new_verts: Vec<NewVertex>,
    /// Number of distinct [`NewPid::Wedge`] positions in `new_verts`.
    pub wedges: u32,
    /// Diagnostics: locked / total vertices of the input group.
    pub locked: usize,
    pub verts: usize,
    /// Diagnostics: input triangles in floating pieces.
    pub floating: usize,
    /// Floating output triangles when the group is (>= 90%) floating pieces
    /// (needle/leaf cards, branch fragments); `None` for all other groups.
    /// For those the DAG builder replaces the Hausdorff distance (a vanished
    /// card leaves a hole its own size while its neighbours are further away
    /// than that, so the distance says little about a canopy) by the spacing
    /// error above.
    pub coverage_tris: Option<(usize, f32)>,
}

/// Edges of one DAG level whose global use count is not 2 (open, non-manifold),
/// used to tell a group-boundary edge (shared with another group) from a
/// genuinely open one.
pub struct GlobalEdges {
    special: Vec<(u64, u32)>,
}

impl GlobalEdges {
    pub fn build<'a>(vs: &VertexStore, tris: impl Iterator<Item = &'a [u32; 3]>) -> Self {
        use rayon::prelude::*;
        let mut keys: Vec<u64> = Vec::new();
        for t in tris {
            for e in 0..3 {
                keys.push(edge_key(
                    vs.pid[t[e] as usize],
                    vs.pid[t[(e + 1) % 3] as usize],
                ));
            }
        }
        keys.par_sort_unstable();
        let mut special = Vec::new();
        let mut i = 0;
        while i < keys.len() {
            let mut j = i + 1;
            while j < keys.len() && keys[j] == keys[i] {
                j += 1;
            }
            if j - i != 2 {
                special.push((keys[i], (j - i) as u32));
            }
            i = j;
        }
        Self { special }
    }

    fn count(&self, key: u64) -> u32 {
        match self.special.binary_search_by_key(&key, |e| e.0) {
            Ok(i) => self.special[i].1,
            Err(_) => 2,
        }
    }
}

// Include non-manifold multiplicities: locking vertices alone does not stop
// a simplifier from removing a duplicate face incident to a locked edge.
fn edge_counts(tris: &[[u32; 3]], pid: &dyn Fn(u32) -> u32) -> BTreeMap<u64, usize> {
    let mut edges = BTreeMap::new();
    for t in tris {
        for e in 0..3 {
            *edges
                .entry(edge_key(pid(t[e]), pid(t[(e + 1) % 3])))
                .or_insert(0) += 1;
        }
    }
    edges
}

fn boundary_signature(
    tris: &[[u32; 3]],
    pid: &dyn Fn(u32) -> u32,
    strict: bool,
) -> BTreeMap<u64, usize> {
    let mut edges = edge_counts(tris, pid);
    edges.retain(|_, n| *n == 1 || (strict && *n > 2));
    edges
}

/// Number of edges used by more than two triangles.
fn nonmanifold(tris: &[[u32; 3]], pid: &dyn Fn(u32) -> u32) -> usize {
    edge_counts(tris, pid).values().filter(|&&n| n > 2).count()
}

/// Weights that turn attribute differences into object-space length, the
/// unit of the position error, so the simplifier's combined error is the
/// same quantity `deviation` measures and the error limit bounds:
/// * normal: a tilt `t` (the chord `|n_a - n_b|`) acting over half an output
///   edge displaces the shaded surface by `t * e / 2`. Output edges of a
///   group simplified to half its triangles are `sqrt(2)` times the input's
///   mean edge `m`, so the weight is `m / sqrt(2)`.
/// * uv: object-space length per uv unit of the section
///   (`sqrt(area / uv area)`, per channel), i.e. how far the texture slides
///   along the surface.
///
/// Tangents carry no weight: they are re-derived from the normal when a
/// record is written ([`VertexAttr::encode`]).
struct AttrWeights([f32; ATTR_FLOATS]);

/// Floats per vertex handed to the simplifier: normal xyz, uv0, uv1.
const ATTR_FLOATS: usize = 7;

impl AttrWeights {
    fn of(vs: &VertexStore, section: &[[u32; 3]], surf: &Surface) -> Self {
        let mut sum = 0.0f64;
        for t in section {
            let p = t.map(|v| vs.pos[v as usize]);
            sum += f64::from(dist(p[0], p[1]) + dist(p[1], p[2]) + dist(p[2], p[0]));
        }
        let mean_edge = (sum / (3 * section.len().max(1)) as f64) as f32;
        let n = mean_edge * std::f32::consts::FRAC_1_SQRT_2;
        let [u0, u1] = uv_scale(section, surf);
        Self([n, n, n, u0, u0, u1, u1])
    }
}

fn attr_floats(a: &VertexAttr) -> [f32; ATTR_FLOATS] {
    [
        a.normal[0],
        a.normal[1],
        a.normal[2],
        a.uv0[0],
        a.uv0[1],
        a.uv1[0],
        a.uv1[1],
    ]
}

fn find(p: &mut [u32], mut x: u32) -> u32 {
    while p[x as usize] != x {
        p[x as usize] = p[p[x as usize] as usize];
        x = p[x as usize];
    }
    x
}

/// Per triangle: does it belong to a floating piece? `edges` = sorted edge
/// keys of the group (one entry per use).
fn floating_flags(
    vs: &VertexStore,
    tris: &[Tri],
    edges: &[u64],
    global: Option<&GlobalEdges>,
) -> Vec<bool> {
    let cross = |key: u64| -> bool {
        let Some(global) = global else { return false };
        let lo = edges.partition_point(|&e| e < key);
        let hi = edges.partition_point(|&e| e <= key);
        (hi - lo) as u32 != global.count(key)
    };
    let mut pids: Vec<u32> = tris
        .iter()
        .filter(|t| vs.floating[t.v[0] as usize])
        .flat_map(|t| t.v.map(|v| vs.pid[v as usize]))
        .collect();
    pids.sort_unstable();
    pids.dedup();
    let local = |v: u32| pids.binary_search(&vs.pid[v as usize]).unwrap() as u32;
    let mut parent: Vec<u32> = (0..pids.len() as u32).collect();
    for t in tris.iter().filter(|t| vs.floating[t.v[0] as usize]) {
        for k in 1..3 {
            let a = find(&mut parent, local(t.v[0]));
            let b = find(&mut parent, local(t.v[k]));
            if a != b {
                parent[b as usize] = a;
            }
        }
    }
    let mut touches = vec![false; pids.len()];
    let roots: Vec<Option<u32>> = tris
        .iter()
        .map(|t| vs.floating[t.v[0] as usize].then(|| find(&mut parent, local(t.v[0]))))
        .collect();
    for (t, r) in tris.iter().zip(&roots) {
        let Some(r) = r else { continue };
        for e in 0..3 {
            let key = edge_key(vs.pid[t.v[e] as usize], vs.pid[t.v[(e + 1) % 3] as usize]);
            if cross(key) {
                touches[*r as usize] = true;
            }
        }
    }
    roots
        .iter()
        .map(|r| r.is_some_and(|r| !touches[r as usize]))
        .collect()
}

/// Simplify a group toward `target` triangles with its boundary locked.
///
/// `limit` is the largest object-space error (position, and attributes in
/// the same unit, see [`AttrWeights`]) any collapse may introduce; the
/// simplifier stops there, so the result can keep more than `target`
/// triangles (the caller decides whether that is progress).
///
/// `whole` marks a single group covering every remaining cluster: there is no
/// neighbour to crack against, so only open (single-use) edges are protected
/// and non-manifold edges may simplify like any other.
///
/// `global` gives the level's open/non-manifold edge counts (needed to tell
/// floating pieces from group-boundary geometry); `None` when the group is
/// the whole level.
///
/// `attr` decodes a vertex's record when the mesh uses the format's attribute
/// layout (`None` for opaque layouts: positions are then placed optimally and
/// the record of the surviving vertex is kept).
pub fn simplify_group(
    vs: &VertexStore,
    tris: &[Tri],
    target: usize,
    limit: f32,
    whole: bool,
    global: Option<&GlobalEdges>,
    attr: Option<&(dyn Fn(u32) -> Option<VertexAttr> + Sync)>,
) -> SimplifyResult {
    let mut edges = Vec::with_capacity(tris.len() * 3);
    for t in tris {
        for e in 0..3 {
            edges.push(edge_key(
                vs.pid[t.v[e] as usize],
                vs.pid[t.v[(e + 1) % 3] as usize],
            ));
        }
    }
    edges.sort_unstable();
    // Edges another group also uses (or non-manifold ones) lock their
    // vertices: the neighbour must meet this group exactly. The mesh's own
    // open border is locked too, so separate meshes placed edge to edge (a
    // modular kit) keep meeting at every level of detail.
    let mut boundary = Vec::new();
    let mut i = 0;
    while i < edges.len() {
        let mut end = i + 1;
        while end < edges.len() && edges[end] == edges[i] {
            end += 1;
        }
        let n = end - i;
        let shared = global.is_some_and(|g| g.count(edges[i]) != n as u32);
        let ends = [(edges[i] >> 32) as u32, edges[i] as u32];
        if shared || n == 1 || (!whole && n > 2) {
            boundary.extend(ends);
        }
        i = end;
    }
    boundary.sort_unstable();
    boundary.dedup();

    // Without level-wide edge counts nothing can be proven isolated.
    let floating = if global.is_some() || whole {
        floating_flags(vs, tris, &edges, global)
    } else {
        vec![false; tris.len()]
    };
    let n_floating = floating.iter().filter(|&&f| f).count();

    // (material, floating) sections.
    let mut sections: BTreeMap<(u32, bool), Vec<[u32; 3]>> = BTreeMap::new();
    for (t, &f) in tris.iter().zip(&floating) {
        sections.entry((t.mat, f)).or_default().push(t.v);
    }
    let base0 = vs.len() as u32;
    let mut placed = Placed {
        base0,
        verts: Vec::new(),
        wedges: 0,
    };
    let mut output = Vec::new();
    let mut merged_cards = false;
    let (mut n_locked, mut n_verts) = (0usize, 0usize);
    for ((material, is_floating), section) in sections {
        let want = (target * section.len() / tris.len()).max(1);
        if is_floating {
            // Vertices on an edge shared with another material section stay put
            // (the sections are simplified independently and must still meet).
            let mut sec_edges: Vec<u64> = section
                .iter()
                .flat_map(|t| {
                    (0..3).map(move |e| {
                        edge_key(vs.pid[t[e] as usize], vs.pid[t[(e + 1) % 3] as usize])
                    })
                })
                .collect();
            sec_edges.sort_unstable();
            let mut pinned: Vec<u32> = Vec::new();
            let mut k = 0;
            while k < sec_edges.len() {
                let mut j = k + 1;
                while j < sec_edges.len() && sec_edges[j] == sec_edges[k] {
                    j += 1;
                }
                let all = edges.partition_point(|&e| e <= sec_edges[k])
                    - edges.partition_point(|&e| e < sec_edges[k]);
                if all != j - k {
                    pinned.push((sec_edges[k] >> 32) as u32);
                    pinned.push(sec_edges[k] as u32);
                }
                k = j;
            }
            pinned.sort_unstable();
            pinned.dedup();
            let (reduced, nv, merged) = simplify_floating(vs, &section, want, &pinned, &mut placed);
            merged_cards |= merged;
            n_verts += nv;
            output.extend(reduced.into_iter().map(|t| (t, material)));
            continue;
        }
        let (reduced, locked, verts) = simplify_anchored(
            vs,
            &section,
            &edges,
            want,
            &boundary,
            whole,
            limit,
            attr,
            &mut placed,
        );
        n_locked += locked;
        n_verts += verts;
        output.extend(reduced.into_iter().map(|t| (t, material)));
    }
    // Dominance is by source pieces (small components of the input mesh), also when
    // some of them currently straddle a group border and stay anchored.
    let src_floating = tris.iter().filter(|t| vs.floating[t.v[0] as usize]).count();
    // Merged cards keep the measured deviation (their error is a real distance,
    // not the spacing proxy used for pieces that could only be thinned).
    let coverage = (!merged_cards && src_floating * 10 >= tris.len() * 9).then(|| {
        let float_out: Vec<[u32; 3]> = output
            .iter()
            .filter(|(t, _)| placed.floating(vs, t[0]))
            .map(|(t, _)| *t)
            .collect();
        (!float_out.is_empty()).then(|| (float_out.len(), placed.mean_footprint(vs, &float_out)))
    });
    SimplifyResult {
        tris: output,
        new_verts: placed.verts,
        wedges: placed.wedges,
        locked: n_locked,
        verts: n_verts,
        floating: n_floating,
        coverage_tris: coverage.flatten(),
    }
}

/// Vertices one group's simplification placed so far.
struct Placed {
    base0: u32,
    verts: Vec<NewVertex>,
    wedges: u32,
}

impl Placed {
    fn pos(&self, vs: &VertexStore, v: u32) -> V3 {
        if v >= self.base0 {
            self.verts[(v - self.base0) as usize].pos
        } else {
            vs.pos[v as usize]
        }
    }

    fn floating(&self, vs: &VertexStore, v: u32) -> bool {
        if v >= self.base0 {
            self.verts[(v - self.base0) as usize].floating
        } else {
            vs.floating[v as usize]
        }
    }

    /// Position id for topology checks; placed positions get ids past every
    /// existing one.
    fn pid(&self, vs: &VertexStore, v: u32) -> u32 {
        if v < self.base0 {
            return vs.pid[v as usize];
        }
        match self.verts[(v - self.base0) as usize].pid {
            NewPid::Existing(p) => p,
            NewPid::Wedge(w) => vs.pid_count() + w,
        }
    }

    /// Square root of the mean triangle area: the linear footprint of one
    /// output triangle (thin needles count as thin, not as long).
    fn mean_footprint(&self, vs: &VertexStore, tris: &[[u32; 3]]) -> f32 {
        let mut s = 0.0f64;
        for t in tris {
            let p = t.map(|v| self.pos(vs, v));
            s += 0.5 * f64::from(length(cross(sub(p[1], p[0]), sub(p[2], p[0]))));
        }
        (s / tris.len().max(1) as f64).sqrt() as f32
    }
}

/// One anchored section: attribute-aware simplification with optimal vertex
/// placement, verified crack-free. `boundary` are the position ids that must
/// not move (shared with other groups, the open border, or non-manifold).
/// Edges the section shares with the group's other material
/// sections are locked too. Returns (triangles, locked vertices, vertices).
#[allow(clippy::too_many_arguments)]
fn simplify_anchored(
    vs: &VertexStore,
    section: &[[u32; 3]],
    group_edges: &[u64],
    want: usize,
    boundary: &[u32],
    whole: bool,
    limit: f32,
    attr: Option<&(dyn Fn(u32) -> Option<VertexAttr> + Sync)>,
    placed: &mut Placed,
) -> (Vec<[u32; 3]>, usize, usize) {
    let mut ids: Vec<u32> = section.iter().flatten().copied().collect();
    ids.sort_unstable();
    ids.dedup();
    let local = |g: &u32| ids.binary_search(g).unwrap() as u32;
    let positions: Vec<V3> = ids.iter().map(|&g| vs.pos[g as usize]).collect();
    let records: Option<Vec<VertexAttr>> =
        attr.and_then(|attr| ids.iter().map(|&g| attr(g)).collect());
    let (floats, weights): (Vec<f32>, Vec<f32>) = match (&records, attr) {
        (Some(r), Some(attr)) => (
            r.iter().flat_map(attr_floats).collect(),
            AttrWeights::of(
                vs,
                section,
                &Surface {
                    pos: &vs.pos,
                    attr,
                    seam: &vs.seam,
                },
            )
            .0
            .to_vec(),
        ),
        _ => (Vec::new(), Vec::new()),
    };
    let stride = weights.len() * 4;
    // Locked position ids: the group's boundary plus edges shared with other
    // material sections of the group (simplified independently, they must
    // still meet).
    let orig_pid = |v: u32| vs.pid[v as usize];
    let mut fixed: Vec<u32> = boundary.to_vec();
    for (&key, &n) in &edge_counts(section, &orig_pid) {
        let all =
            group_edges.partition_point(|&e| e <= key) - group_edges.partition_point(|&e| e < key);
        if all > n {
            fixed.push((key >> 32) as u32);
            fixed.push(key as u32);
        }
    }
    fixed.sort_unstable();
    fixed.dedup();
    let is_fixed = |p: u32| fixed.binary_search(&p).is_ok();
    let mut locks: Vec<SimplifyVertex> = ids
        .iter()
        .map(|&g| {
            if is_fixed(vs.pid[g as usize]) {
                SimplifyVertex::Lock
            } else {
                SimplifyVertex::None
            }
        })
        .collect();
    let n_locked = locks
        .iter()
        .filter(|l| l.contains(SimplifyVertex::Lock))
        .count();
    let indices: Vec<u32> = section.iter().flatten().map(local).collect();
    // The locked edges must survive exactly: open (or non-manifold) edges
    // between two locked vertices.
    let protected = |tris: &[[u32; 3]], pid: &dyn Fn(u32) -> u32| {
        let mut e = boundary_signature(tris, pid, !whole);
        e.retain(|&k, _| is_fixed((k >> 32) as u32) && is_fixed(k as u32));
        e
    };
    let expected = protected(section, &orig_pid);
    let nm_in = if whole {
        nonmanifold(section, &orig_pid)
    } else {
        0
    };
    // Existing positions of the section, so a placed vertex landing exactly
    // on one shares its position id.
    let existing: HashMap<[u32; 3], u32> = positions
        .iter()
        .zip(&ids)
        .map(|(p, &g)| (p.map(f32::to_bits), vs.pid[g as usize]))
        .collect();
    let (mark_verts, mark_wedges) = (placed.verts.len(), placed.wedges);
    for _ in 0..16 {
        let mut idx = indices.clone();
        let mut pos = positions.clone();
        let mut att = floats.clone();
        let n = meshopt::simplify_with_update(
            &mut idx,
            &mut pos,
            &mut att,
            &weights,
            stride,
            Some(&locks),
            want * 3,
            limit,
            meshopt::SimplifyOptions::ErrorAbsolute,
            None,
        );
        idx.truncate(n);
        // Vertices whose position or attributes changed become new vertices;
        // the rest keep their ids (and locked ones are bit-exact by contract).
        let mut remap: HashMap<u32, u32> = HashMap::new();
        let mut wedge_of: HashMap<[u32; 3], u32> = HashMap::new();
        let mut out: Vec<[u32; 3]> = Vec::with_capacity(n / 3);
        for t in idx.as_chunks::<3>().0 {
            let mut tri = [0u32; 3];
            for (k, &l) in t.iter().enumerate() {
                let li = l as usize;
                tri[k] = *remap.entry(l).or_insert_with(|| {
                    let g = ids[li];
                    let a = &att[li * weights.len()..(li + 1) * weights.len()];
                    let a0 = &floats[li * weights.len()..(li + 1) * weights.len()];
                    let moved = pos[li].map(f32::to_bits) != positions[li].map(f32::to_bits);
                    let reshaded = a.iter().zip(a0).any(|(x, y)| x.to_bits() != y.to_bits());
                    if locks[li].contains(SimplifyVertex::Lock) || (!moved && !reshaded) {
                        return g;
                    }
                    let key = pos[li].map(f32::to_bits);
                    let pid = match existing.get(&key) {
                        Some(&p) => NewPid::Existing(p),
                        None => {
                            let next = placed.wedges;
                            let w = *wedge_of.entry(key).or_insert(next);
                            if w == next {
                                placed.wedges += 1;
                            }
                            NewPid::Wedge(w)
                        }
                    };
                    let rec = match &records {
                        Some(r) if reshaded => {
                            let base = r[li];
                            NewAttr::Bytes(
                                VertexAttr {
                                    normal: [a[0], a[1], a[2]],
                                    uv0: [a[3], a[4]],
                                    uv1: [a[5], a[6]],
                                    ..base
                                }
                                .encode(),
                            )
                        }
                        _ => NewAttr::Record(vs.attr[g as usize]),
                    };
                    placed.verts.push(NewVertex {
                        pos: pos[li],
                        attr: rec,
                        pid,
                        floating: vs.floating[g as usize],
                    });
                    placed.base0 + placed.verts.len() as u32 - 1
                });
            }
            out.push(tri);
        }
        let pid = |v: u32| placed.pid(vs, v);
        let actual = protected(&out, &pid);
        // Any other open edge is a hole.
        let border_ok = |v: u32| v < placed.base0 && is_fixed(vs.pid[v as usize]);
        let open_edges = edge_counts(&out, &pid);
        // Source vertex of every output vertex (placed ones: the slot they
        // came from).
        let source: HashMap<u32, u32> = remap.iter().map(|(&l, &o)| (o, ids[l as usize])).collect();
        let mut hole_pids: Vec<u32> = Vec::new();
        for t in &out {
            for e in 0..3 {
                let (a, b) = (t[e], t[(e + 1) % 3]);
                if open_edges[&edge_key(pid(a), pid(b))] == 1 && !(border_ok(a) && border_ok(b)) {
                    hole_pids.extend([a, b].map(|v| vs.pid[source[&v] as usize]));
                }
            }
        }
        let holes = !hole_pids.is_empty();
        if expected == actual
            && !holes
            && !out.is_empty()
            && (!whole || nonmanifold(&out, &pid) <= nm_in)
        {
            return (out, n_locked, ids.len());
        }
        placed.verts.truncate(mark_verts);
        placed.wedges = mark_wedges;
        // Freeze the one-ring around changed topology (around a hole, the
        // one-ring of every triangle the hole touches), then retry the original
        // section. Preserve as much safe reduction as possible.
        let mut affected = std::collections::BTreeSet::new();
        for edge in expected.keys().chain(actual.keys()) {
            if expected.get(edge) != actual.get(edge) {
                affected.insert((edge >> 32) as u32);
                affected.insert(*edge as u32);
            }
        }
        affected.extend(hole_pids);
        let mut changed = false;
        for t in section {
            if t.iter().any(|&w| affected.contains(&vs.pid[w as usize])) {
                for v in t {
                    let l = &mut locks[local(v) as usize];
                    changed |= !l.contains(SimplifyVertex::Lock);
                    *l = SimplifyVertex::Lock;
                }
            }
        }
        if !changed {
            break;
        }
    }
    // Malformed/non-manifold input must retain its geometry instead of
    // opening a crack between independently selected groups.
    (section.to_vec(), n_locked, ids.len())
}

/// Floating pieces of one material section, reduced to about `want` triangles.
/// Card-like pieces (needles, leaf quads) are merged into area-preserving
/// quads; larger pieces are thinned by [`collapse_floating`]. Returns
/// (triangles, vertex count, whether cards were merged).
fn simplify_floating(
    vs: &VertexStore,
    section: &[[u32; 3]],
    want: usize,
    pinned: &[u32],
    placed: &mut Placed,
) -> (Vec<[u32; 3]>, usize, bool) {
    // Connected pieces (by canonical position id).
    let mut pids: Vec<u32> = section
        .iter()
        .flatten()
        .map(|&v| vs.pid[v as usize])
        .collect();
    pids.sort_unstable();
    pids.dedup();
    let local = |v: u32| pids.binary_search(&vs.pid[v as usize]).unwrap() as u32;
    let mut parent: Vec<u32> = (0..pids.len() as u32).collect();
    for t in section {
        for k in 1..3 {
            let (a, b) = (
                find(&mut parent, local(t[0])),
                find(&mut parent, local(t[k])),
            );
            if a != b {
                parent[a.max(b) as usize] = a.min(b);
            }
        }
    }
    let mut pieces: BTreeMap<u32, Vec<[u32; 3]>> = BTreeMap::new();
    for t in section {
        pieces
            .entry(find(&mut parent, local(t[0])))
            .or_default()
            .push(*t);
    }
    let mut cards: Vec<Vec<[u32; 3]>> = Vec::new();
    let mut rest: Vec<[u32; 3]> = Vec::new();
    for (_, p) in pieces {
        let pinned_piece = p
            .iter()
            .flatten()
            .any(|&v| pinned.binary_search(&vs.pid[v as usize]).is_ok());
        if p.len() <= CARD_MAX_TRIS && !pinned_piece {
            cards.push(p);
        } else {
            rest.extend(p);
        }
    }
    let card_tris: usize = cards.iter().map(Vec::len).sum();
    let want_cards = (want * card_tris / section.len().max(1)).max(2) / 2;
    if cards.len() <= want_cards.max(1) {
        // Nothing to merge: the section is thinned as a whole.
        return {
            let (t, n) = collapse_floating(vs, section, want, pinned);
            (t, n, false)
        };
    }
    let mut out = Vec::new();
    let mut verts = 0;
    if !rest.is_empty() {
        let want_rest = (want * rest.len() / section.len()).max(1);
        let (t, n) = collapse_floating(vs, &rest, want_rest, pinned);
        verts += n;
        out.extend(t);
    }
    let merged = merge_cards(vs, &cards, want_cards, placed);
    verts += merged.len() * 2;
    out.extend(merged);
    (out, verts, true)
}

/// Symmetric 3x3 eigen decomposition (cyclic Jacobi, fixed sweeps, f64 so the
/// result does not depend on the order of anything). Returns eigenvalues
/// descending with matching unit eigenvectors.
fn eigen_sym3(m: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut a = m;
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..24 {
        let off = a[0][1].abs() + a[0][2].abs() + a[1][2].abs();
        if off < 1e-18 {
            break;
        }
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            if a[p][q].abs() < 1e-30 {
                continue;
            }
            let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let (c, s) = (1.0 / (t * t + 1.0).sqrt(), t / (t * t + 1.0).sqrt());
            for k in 0..3 {
                let (akp, akq) = (a[k][p], a[k][q]);
                a[k][p] = c * akp - s * akq;
                a[k][q] = s * akp + c * akq;
            }
            for k in 0..3 {
                let (apk, aqk) = (a[p][k], a[q][k]);
                a[p][k] = c * apk - s * aqk;
                a[q][k] = s * apk + c * aqk;
            }
            for k in 0..3 {
                let (vkp, vkq) = (v[k][p], v[k][q]);
                v[k][p] = c * vkp - s * vkq;
                v[k][q] = s * vkp + c * vkq;
            }
        }
    }
    let mut order = [0usize, 1, 2];
    order.sort_by(|&i, &j| a[j][j].total_cmp(&a[i][i]));
    let vals = order.map(|i| a[i][i]);
    let vecs = order.map(|i| [v[0][i], v[1][i], v[2][i]]);
    (vals, vecs)
}

fn to_f32(v: [f64; 3]) -> V3 {
    [v[0] as f32, v[1] as f32, v[2] as f32]
}

/// One card-like piece reduced to the quad that stands in for it.
struct Card {
    centre: V3,
    axis: V3,
    normal: V3,
    area: f32,
    length: f32,
    verts: Vec<u32>,
}

fn describe_card(vs: &VertexStore, tris: &[[u32; 3]]) -> Option<Card> {
    let mut verts: Vec<u32> = tris.iter().flatten().copied().collect();
    verts.sort_unstable_by_key(|&v| (vs.pid[v as usize], v));
    verts.dedup_by_key(|v| vs.pid[*v as usize]);
    let mut area = 0.0f64;
    let mut centre = [0.0f64; 3];
    let mut n_sum = [0.0f64; 3];
    for t in tris {
        let p = t.map(|v| vs.pos[v as usize]);
        let n = cross(sub(p[1], p[0]), sub(p[2], p[0]));
        let a = 0.5 * length(n) as f64;
        area += a;
        for k in 0..3 {
            centre[k] += a * ((p[0][k] + p[1][k] + p[2][k]) / 3.0) as f64;
            n_sum[k] += n[k] as f64;
        }
    }
    if area <= 0.0 {
        return None;
    }
    for c in &mut centre {
        *c /= area;
    }
    let mut cov = [[0.0f64; 3]; 3];
    for &v in &verts {
        let d = [0, 1, 2].map(|k| vs.pos[v as usize][k] as f64 - centre[k]);
        for i in 0..3 {
            for j in 0..3 {
                cov[i][j] += d[i] * d[j];
            }
        }
    }
    let (_, e) = eigen_sym3(cov);
    let axis = to_f32(e[0]);
    let mut normal = to_f32(e[2]);
    if dot(normal, to_f32(n_sum)) < 0.0 {
        normal = scale(normal, -1.0);
    }
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for &v in &verts {
        let d = dot(sub(vs.pos[v as usize], to_f32(centre)), axis);
        lo = lo.min(d);
        hi = hi.max(d);
    }
    Some(Card {
        centre: to_f32(centre),
        axis,
        normal,
        area: area as f32,
        length: (hi - lo).max(1e-9),
        verts,
    })
}

/// Merge card-like pieces into `want_cards` quads (two triangles each).
///
/// Pieces are split by recursive median bisection along the widest spatial
/// axis into `want_cards` equally populated cells; each cell becomes one
/// quad whose area is the summed area of its members (coverage is preserved)
/// and whose length is the members' mean length (scale is preserved) until
/// the quad would be wider than it is long, after which it grows evenly. The
/// quad's width is the only thing that changes: the deviation that
/// introduces, and the spread of the members inside the cell, are what the
/// DAG builder measures as this group's error, so a level is only selected at
/// distances where that error is below a pixel.
fn merge_cards(
    vs: &VertexStore,
    pieces: &[Vec<[u32; 3]>],
    want_cards: usize,
    placed: &mut Placed,
) -> Vec<[u32; 3]> {
    let cards: Vec<Card> = pieces.iter().filter_map(|p| describe_card(vs, p)).collect();
    let mut cells: Vec<Vec<usize>> = Vec::new();
    fn split(cards: &[Card], idx: Vec<usize>, k: usize, out: &mut Vec<Vec<usize>>) {
        if k <= 1 || idx.len() <= 1 {
            out.push(idx);
            return;
        }
        let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
        for &i in &idx {
            for a in 0..3 {
                lo[a] = lo[a].min(cards[i].centre[a]);
                hi[a] = hi[a].max(cards[i].centre[a]);
            }
        }
        let axis = (0..3)
            .max_by(|&a, &b| (hi[a] - lo[a]).total_cmp(&(hi[b] - lo[b])))
            .unwrap();
        let mut idx = idx;
        idx.sort_by(|&a, &b| {
            cards[a].centre[axis]
                .total_cmp(&cards[b].centre[axis])
                .then(a.cmp(&b))
        });
        let k1 = k / 2;
        let cut = (idx.len() * k1 / k).clamp(1, idx.len() - 1);
        let right = idx.split_off(cut);
        split(cards, idx, k1, out);
        split(cards, right, k - k1, out);
    }
    split(
        &cards,
        (0..cards.len()).collect(),
        want_cards.max(1),
        &mut cells,
    );
    // A cell that straddles a gap (stray pieces left at the edge of a group)
    // would put its card where no piece is: split such cells until compact, at
    // the price of fewer merges there.
    let radius_of = |cell: &[usize]| -> f32 {
        let mut c = [0.0f32; 3];
        for &i in cell {
            for a in 0..3 {
                c[a] += cards[i].centre[a];
            }
        }
        let c = scale(c, 1.0 / cell.len() as f32);
        cell.iter()
            .map(|&i| dist(cards[i].centre, c))
            .fold(0.0f32, f32::max)
    };
    let mut radii: Vec<f32> = cells
        .iter()
        .filter(|c| c.len() > 1)
        .map(|c| radius_of(c))
        .collect();
    if !radii.is_empty() {
        radii.sort_by(f32::total_cmp);
        let limit = CELL_RADIUS_SPREAD * radii[radii.len() / 2];
        fn compact(
            cards: &[Card],
            cell: Vec<usize>,
            limit: f32,
            radius_of: &dyn Fn(&[usize]) -> f32,
            out: &mut Vec<Vec<usize>>,
        ) {
            if cell.len() <= 1 || radius_of(&cell) <= limit {
                out.push(cell);
                return;
            }
            let mut halves = Vec::new();
            split(cards, cell, 2, &mut halves);
            for h in halves {
                compact(cards, h, limit, radius_of, out);
            }
        }
        let mut refined = Vec::with_capacity(cells.len());
        for cell in std::mem::take(&mut cells) {
            compact(&cards, cell, limit, &radius_of, &mut refined);
        }
        cells = refined;
    }

    let mut out = Vec::with_capacity(cells.len() * 2);
    for cell in cells {
        let members: Vec<&Card> = cell.iter().map(|&i| &cards[i]).collect();
        let area: f32 = members.iter().map(|c| c.area).sum();
        let reference = members
            .iter()
            .max_by(|a, b| a.area.total_cmp(&b.area))
            .unwrap();
        // Orientation-agnostic dominant axis and consistent plane normal.
        let mut axis_cov = [[0.0f64; 3]; 3];
        let mut centre = [0.0f64; 3];
        let mut n_sum = [0.0f64; 3];
        let mut mean_len = 0.0f64;
        for c in &members {
            let w = c.area as f64;
            for i in 0..3 {
                centre[i] += w * c.centre[i] as f64;
                for j in 0..3 {
                    axis_cov[i][j] += w * (c.axis[i] * c.axis[j]) as f64;
                }
            }
            let s = if dot(c.normal, reference.normal) < 0.0 {
                -1.0
            } else {
                1.0
            };
            for i in 0..3 {
                n_sum[i] += w * (s * c.normal[i]) as f64;
            }
            mean_len += w * c.length as f64;
        }
        let a_total = area.max(1e-30) as f64;
        let centre = to_f32([
            centre[0] / a_total,
            centre[1] / a_total,
            centre[2] / a_total,
        ]);
        let mean_len = (mean_len / a_total) as f32;
        let normal = {
            let n = normalize(to_f32(n_sum));
            if length(n) > 0.0 {
                n
            } else {
                reference.normal
            }
        };
        let (_, ev) = eigen_sym3(axis_cov);
        let mut axis = sub(to_f32(ev[0]), scale(normal, dot(to_f32(ev[0]), normal)));
        if length(axis) < 1e-6 {
            axis = sub(reference.axis, scale(normal, dot(reference.axis, normal)));
        }
        let axis = normalize(axis);
        let side = normalize(cross(normal, axis));
        let len = mean_len.max(area.sqrt());
        let wid = area / len;
        let (a, b) = (scale(axis, len * 0.5), scale(side, wid * 0.5));
        let corners = [
            sub(sub(centre, a), b),
            sub(add(centre, a), b),
            add(add(centre, a), b),
            add(sub(centre, a), b),
        ];
        let first = placed.base0 + placed.verts.len() as u32;
        let signs = [(-1.0f32, -1.0f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];
        for (p, (sa, sb)) in corners.into_iter().zip(signs) {
            // Attribute bytes (normal, uv, ...) of all four corners come from ONE
            // member, the largest: the vertex of that member that lies furthest
            // toward this corner of the card. Its texture coordinates stay a
            // coherent quad (the member's whole leaf or needle texture is
            // stretched over the card). Taking each corner from whichever member
            // is nearest mixes unrelated atlas islands across the card, which
            // sampled as dark shards.
            let mut best = (f32::NEG_INFINITY, reference.verts[0]);
            for &v in &reference.verts {
                let d = sub(vs.pos[v as usize], reference.centre);
                let score = sa * dot(d, axis) + sb * dot(d, side);
                if score > best.0 {
                    best = (score, v);
                }
            }
            placed.verts.push(NewVertex {
                pos: p,
                attr: NewAttr::Record(vs.attr[best.1 as usize]),
                pid: NewPid::Wedge(placed.wedges),
                floating: true,
            });
            placed.wedges += 1;
        }
        out.push([first, first + 1, first + 2]);
        out.push([first, first + 2, first + 3]);
    }
    out
}

/// Reduce floating pieces of one material section to about `want` triangles:
/// unlocked edge collapse with island pruning. Returns (triangles, vertex
/// count).
fn collapse_floating(
    vs: &VertexStore,
    section: &[[u32; 3]],
    want: usize,
    pinned: &[u32],
) -> (Vec<[u32; 3]>, usize) {
    let mut ids: Vec<u32> = section.iter().flatten().copied().collect();
    ids.sort_unstable();
    ids.dedup();
    let positions: Vec<_> = ids.iter().map(|&g| vs.pos[g as usize]).collect();
    let locks: Vec<bool> = ids
        .iter()
        .map(|&g| pinned.binary_search(&vs.pid[g as usize]).is_ok())
        .collect();
    let indices: Vec<u32> = section
        .iter()
        .flatten()
        .map(|g| ids.binary_search(g).unwrap() as u32)
        .collect();
    let vertices = meshopt::VertexDataAdapter::new(bytemuck::cast_slice(&positions), 12, 0)
        .expect("packed float3 positions");
    let result = meshopt::simplify_with_locks(
        &indices,
        &vertices,
        &locks,
        want * 3,
        f32::MAX,
        meshopt::SimplifyOptions::ErrorAbsolute | meshopt::SimplifyOptions::Prune,
        None,
    );
    let out: Vec<[u32; 3]> = result
        .as_chunks::<3>()
        .0
        .iter()
        .map(|t| [ids[t[0] as usize], ids[t[1] as usize], ids[t[2] as usize]])
        .filter(|t| {
            let p = t.map(|v| vs.pid[v as usize]);
            p[0] != p[1] && p[1] != p[2] && p[0] != p[2]
        })
        .collect();
    if out.is_empty() {
        // Never drop a whole section: keep it (cannot happen with a sane target).
        return (section.to_vec(), ids.len());
    }
    (out, ids.len())
}
