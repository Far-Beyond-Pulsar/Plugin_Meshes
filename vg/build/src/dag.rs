//! Cluster DAG construction (plan section 2.1).
//!
//! Level 0 = clusters of the full-resolution mesh (per material section).
//! Each round partitions the current level into groups (`partition`),
//! simplifies each group toward half its triangles with its boundary locked
//! and an error limit (`simplify`), measures the result against the original
//! surface (`deviation`) and splits it into new clusters; this repeats until a
//! single cluster remains or nothing simplifies any more.
//!
//! Error limit. Halving a group's triangles lengthens its edges by `sqrt(2)`;
//! on a smooth surface the deviation of a chord grows with the square of its
//! length (sagitta `e^2 / 8R`), so a level is expected to double the error:
//! [`GROWTH`] `= 1 / RATIO`. A group may therefore move its surface by at most
//! `GROWTH` times its reference error: the largest error of its children, and
//! never less than half the mean input edge (features narrower than the
//! sampling interval of the input are not represented by it anyway; this is
//! the only reference level-0 groups have). A group that cannot reach
//! [`ACCEPT`] of its triangles within its limit keeps them: its clusters pass
//! to the next level unchanged and are regrouped there with neighbours whose
//! errors (and so the limit) have grown, under a new partition with different
//! locked borders. When no group of a level makes progress, the level's limits
//! double (as if a level had elapsed) until a group does, or until every
//! limit exceeds its group's diameter, past which no error limit constrains
//! anything and the remaining clusters become roots.
//!
//! Terminology used across the crate:
//! * *producer group*: the group whose simplification created a cluster.
//! * *consumer group* ("hgroup"): the group that consumed a cluster as input.
//!   All clusters sharing a consumer share `parent_error`/`parent_sphere`;
//!   they are exactly the clusters the hierarchy leaves enumerate.

use std::collections::BTreeMap;

use rayon::prelude::*;

use helio_vg_format::{VertexAttr, PAGE_BYTES};

use crate::cluster::cluster_tris;
use crate::deviation::{measure, Measured, Region, Surface};
use crate::math::*;
use crate::mesh::{mark_floating, weld, BuildError, MeshInput, Tri, VertexStore};
use crate::partition::partition;
use crate::simplify::{
    simplify_group, GlobalEdges, NewAttr, NewPid, SimplifyResult, COVERAGE_FACTOR,
    COVERAGE_FOOTPRINT,
};

pub const NONE: u32 = u32::MAX;
/// A triangle with its material section.
type MatTri = ([u32; 3], u32);
/// One produced cluster's triangles with its material section.
type Piece = (Vec<[u32; 3]>, u32);
/// Triangles a group is simplified toward, as a fraction of its input.
const RATIO: (usize, usize) = (1, 2);
/// Expected error growth of one level (see the module docs): `1 / RATIO`.
const GROWTH: f32 = 2.0;
/// A group's result counts as progress when it keeps at most this fraction of
/// its triangles: half of the intended reduction. Accepting less would at
/// least double the levels (and the per-level selection and streaming cost)
/// needed to reach the root.
const ACCEPT: (usize, usize) = (3, 4);
/// Fraction of the mean input edge below which a level-0 group's reference
/// error never falls (half the sampling interval, see the module docs).
const SAMPLING_FRACTION: f32 = 0.5;
/// Roots are chunked into hierarchy leaves of at most this many clusters.
const ROOT_CHUNK: usize = 8;
/// Upper bound on clusters per simplification group (also bounded by the page).
const MAX_GROUP_CLUSTERS: usize = helio_vg_format::GROUP_MAX_CLUSTERS_LARGE as usize;

pub struct ClusterNode {
    /// Triangles as global unified vertex ids.
    pub tris: Vec<[u32; 3]>,
    pub mat: u32,
    /// Culling + self-error sphere. For non-leaf clusters this is the sphere
    /// of the producing group, shared by all siblings, so the per-cluster
    /// screen-error test agrees across the whole group.
    pub sphere: [f32; 4],
    pub parent_sphere: [f32; 4],
    pub self_error: f32,
    pub parent_error: f32,
    pub level: u32,
    /// Group that produced this cluster (`NONE` for leaves).
    pub producer: u32,
    /// Group that consumed this cluster (root groups for the final level).
    pub consumer: u32,
    /// Build-time only: the original (level-0) triangles this cluster stands
    /// for, as indices into the build's original triangle list. Every
    /// original triangle belongs to exactly one cluster of the current level;
    /// emptied once the consuming group has measured them.
    pub(crate) refs: Vec<u32>,
    /// Build-time only: further original triangles near this cluster's
    /// surface (see `deviation::Region::claimed`).
    pub(crate) claimed: Vec<u32>,
}

/// A set of clusters sharing one parent: either the inputs of one
/// simplification group, or a chunk of root clusters.
pub struct Group {
    pub clusters: Vec<u32>,
    pub level: u32,
    /// Sphere (== `parent_sphere` of its clusters).
    pub sphere: [f32; 4],
    /// Error (== `parent_error` of its clusters); infinite for roots.
    pub error: f32,
    pub is_root: bool,
    /// Clusters this group's simplification produced.
    pub produced: Vec<u32>,
}

pub struct Dag {
    pub vs: VertexStore,
    pub clusters: Vec<ClusterNode>,
    pub groups: Vec<Group>,
    /// Triangles that survived welding/degenerate removal.
    pub source_tris: usize,
}

fn split_into_clusters(vs: &VertexStore, tris: &[MatTri]) -> Vec<Piece> {
    let mut by_mat: BTreeMap<u32, Vec<[u32; 3]>> = BTreeMap::new();
    for (t, m) in tris {
        by_mat.entry(*m).or_default().push(*t);
    }
    let mut out = Vec::new();
    for (mat, ts) in by_mat {
        for cl in cluster_tris(vs, &ts) {
            out.push((cl.iter().map(|&i| ts[i as usize]).collect(), mat));
        }
    }
    out
}

pub fn build_dag(mesh: &MeshInput) -> Result<Dag, BuildError> {
    mesh.validate()?;
    let (mut vs, tris) = weld(mesh);
    vs.release_lookup();
    mark_floating(&mut vs, &tris, crate::simplify::FLOATING_MAX_TRIS);
    if tris.is_empty() {
        return Err(BuildError::EmptyMesh);
    }
    let mut dag = Dag {
        vs,
        clusters: Vec::new(),
        groups: Vec::new(),
        source_tris: tris.len(),
    };

    // Vertex attribute records in the format's layout can be measured and
    // simplified with (normal, uv); any other stride is opaque and only
    // positions are.
    let stride = mesh.attribute_stride;
    let format_layout = stride == helio_vg_format::ATTR_STRIDE as usize;
    let attr_of = |vs: &VertexStore, v: u32| -> Option<VertexAttr> {
        format_layout
            .then(|| VertexAttr::decode(vs.record(v, mesh.attributes)))
            .flatten()
    };

    // ── level 0 ────────────────────────────────────────────────────────────
    let mut by_mat: BTreeMap<u32, Vec<[u32; 3]>> = BTreeMap::new();
    for t in &tris {
        by_mat.entry(t.mat).or_default().push(t.v);
    }
    drop(tris);
    let per_mat: Vec<(u32, Vec<[u32; 3]>, Vec<Vec<u32>>)> = by_mat
        .into_par_iter()
        .map(|(mat, ts)| {
            let parts = cluster_tris(&dag.vs, &ts);
            (mat, ts, parts)
        })
        .collect();
    // The original surface every level is measured against.
    let mut original: Vec<[u32; 3]> = Vec::new();
    let mut cur: Vec<u32> = Vec::new();
    for (mat, ts, parts) in per_mat {
        for cl in parts {
            let tris: Vec<[u32; 3]> = cl.iter().map(|&i| ts[i as usize]).collect();
            let sphere = bounding_sphere(
                tris.iter()
                    .flat_map(|t| t.iter().map(|&v| dag.vs.pos[v as usize])),
            );
            let first = original.len() as u32;
            original.extend_from_slice(&tris);
            cur.push(dag.clusters.len() as u32);
            dag.clusters.push(ClusterNode {
                refs: (first..original.len() as u32).collect(),
                claimed: Vec::new(),
                tris,
                mat,
                sphere,
                parent_sphere: sphere,
                self_error: 0.0,
                parent_error: f32::INFINITY,
                level: 0,
                producer: NONE,
                consumer: NONE,
            });
        }
    }

    // ── levels ─────────────────────────────────────────────────────────────
    // Largest group whose streams are guaranteed to fit one page (worst case:
    // 128 verts/cluster, 12 B position, 3x u16 indices per triangle).
    let group_cap = {
        let per_cluster = 64 + 128 * 12 + 128 * mesh.attribute_stride + 128 * 6;
        ((PAGE_BYTES as usize - 1024) * 9 / 10 / per_cluster).clamp(8, MAX_GROUP_CLUSTERS)
    };
    let mut level = 0u32;
    while cur.len() > 1 {
        let groups = partition(&dag.vs, &cur, &dag.clusters, group_cap);
        // A group that covers everything left has no interior boundary to
        // protect.
        let whole = groups.len() == 1;
        let global = (!whole).then(|| {
            GlobalEdges::build(
                &dag.vs,
                cur.iter()
                    .flat_map(|&c| dag.clusters[c as usize].tris.iter()),
            )
        });
        let inputs: Vec<Vec<Tri>> = groups
            .iter()
            .map(|g| {
                g.iter()
                    .flat_map(|&c| {
                        let cn = &dag.clusters[c as usize];
                        cn.tris.iter().map(move |t| Tri { v: *t, mat: cn.mat })
                    })
                    .collect()
            })
            .collect();
        // Reference error and diameter of each group (see the module docs).
        let reference: Vec<(f32, f32)> = groups
            .par_iter()
            .zip(inputs.par_iter())
            .map(|(g, tris)| {
                let child = g
                    .iter()
                    .map(|&c| dag.clusters[c as usize].self_error)
                    .fold(0.0f32, f32::max);
                let mut sum = 0.0f64;
                for t in tris {
                    let p = t.v.map(|v| dag.vs.pos[v as usize]);
                    sum += f64::from(dist(p[0], p[1]) + dist(p[1], p[2]) + dist(p[2], p[0]));
                }
                let mean_edge = (sum / (3 * tris.len().max(1)) as f64) as f32;
                let sphere = g.iter().fold(dag.clusters[g[0] as usize].sphere, |s, &c| {
                    sphere_union(s, dag.clusters[c as usize].sphere)
                });
                (child.max(SAMPLING_FRACTION * mean_edge), 2.0 * sphere[3])
            })
            .collect();
        let mut relax = 1.0f32;
        let (results, accepted) = loop {
            let attr = |v: u32| attr_of(&dag.vs, v);
            let attr_dyn: Option<&(dyn Fn(u32) -> Option<VertexAttr> + Sync)> =
                format_layout.then_some(&attr as _);
            let results: Vec<SimplifyResult> = inputs
                .par_iter()
                .zip(reference.par_iter())
                .map(|(tris, &(r, _))| {
                    simplify_group(
                        &dag.vs,
                        tris,
                        tris.len() * RATIO.0 / RATIO.1,
                        GROWTH * r * relax,
                        whole,
                        global.as_ref(),
                        attr_dyn,
                    )
                })
                .collect();
            let accepted: Vec<bool> = results
                .iter()
                .zip(&inputs)
                .map(|(r, i)| r.tris.len() * ACCEPT.1 <= i.len() * ACCEPT.0)
                .collect();
            let exhausted = reference
                .iter()
                .all(|&(r, diameter)| GROWTH * r * relax >= diameter);
            if accepted.iter().any(|&a| a) || exhausted {
                break (results, accepted);
            }
            relax *= 2.0;
        };
        if std::env::var("VG_DBG").is_ok() {
            let (l, v, fl) = results.iter().fold((0usize, 0usize, 0usize), |a, r| {
                (a.0 + r.locked, a.1 + r.verts, a.2 + r.floating)
            });
            let old: usize = inputs.iter().map(Vec::len).sum();
            let new: usize = results
                .iter()
                .zip(&accepted)
                .zip(&inputs)
                .map(|((r, &a), i)| if a { r.tris.len() } else { i.len() })
                .sum();
            eprintln!(
                "DBG level {level} clusters {} groups {} accepted {} relax {relax} old {old} new {new} locked {:.1}% floating {:.1}%",
                cur.len(),
                groups.len(),
                accepted.iter().filter(|&&a| a).count(),
                100.0 * l as f64 / v.max(1) as f64,
                100.0 * fl as f64 / old.max(1) as f64
            );
        }
        if !accepted.iter().any(|&a| a) {
            // Nothing simplifies at any error: the current clusters are the
            // roots.
            break;
        }

        // Serial: allocate the placed vertices (ids must not depend on thread
        // scheduling) and translate output corners to global ids.
        let base0 = dag.vs.len() as u32;
        let mut staged: Vec<Option<Vec<MatTri>>> = Vec::with_capacity(results.len());
        let mut coverage: Vec<Option<(usize, f32)>> = Vec::with_capacity(results.len());
        for (res, &ok) in results.iter().zip(&accepted) {
            coverage.push(res.coverage_tris);
            if !ok {
                staged.push(None);
                continue;
            }
            let wedge_pids: Vec<u32> = (0..res.wedges).map(|_| dag.vs.new_pid()).collect();
            let mut per_pid: BTreeMap<u32, u32> = BTreeMap::new();
            let pids: Vec<u32> = res
                .new_verts
                .iter()
                .map(|nv| match nv.pid {
                    NewPid::Existing(p) => p,
                    NewPid::Wedge(w) => wedge_pids[w as usize],
                })
                .collect();
            for (nv, &p) in res.new_verts.iter().zip(&pids) {
                // A position shared with an existing vertex is a seam either way.
                *per_pid.entry(p).or_default() += if matches!(nv.pid, NewPid::Existing(_)) {
                    2
                } else {
                    1
                };
            }
            let ids: Vec<u32> = res
                .new_verts
                .iter()
                .zip(&pids)
                .map(|(nv, &pid)| {
                    let record = match nv.attr {
                        NewAttr::Record(r) => r,
                        NewAttr::Bytes(b) => dag.vs.add_record(&b),
                    };
                    dag.vs
                        .push_placed(nv.pos, record, pid, per_pid[&pid] > 1, nv.floating)
                })
                .collect();
            staged.push(Some(
                res.tris
                    .iter()
                    .map(|&(c, m)| {
                        let f = |x: u32| {
                            if x >= base0 {
                                ids[(x - base0) as usize]
                            } else {
                                x
                            }
                        };
                        ([f(c[0]), f(c[1]), f(c[2])], m)
                    })
                    .collect(),
            ));
        }
        drop(results);
        let parts: Vec<Option<Vec<Piece>>> = staged
            .par_iter()
            .map(|s| s.as_ref().map(|t| split_into_clusters(&dag.vs, t)))
            .collect();

        // Measure the deviation of every simplified group from the original
        // surface it stands for (both directions, see `deviation`).
        let attr = |v: u32| attr_of(&dag.vs, v);
        let surface = Surface {
            pos: &dag.vs.pos,
            attr: &attr,
            seam: &dag.vs.seam,
        };
        // Level triangles by position id, to find each group's neighbours.
        let by_pid: Vec<(u32, u32, u32)> =
            {
                let mut v: Vec<(u32, u32, u32)> =
                    cur.iter()
                        .flat_map(|&c| {
                            let vs = &dag.vs;
                            dag.clusters[c as usize].tris.iter().enumerate().flat_map(
                                move |(ti, t)| t.map(|x| (vs.pid[x as usize], c, ti as u32)),
                            )
                        })
                        .collect();
                v.par_sort_unstable();
                v.dedup();
                v
            };
        let measured: Vec<Option<Measured>> = groups
            .par_iter()
            .zip(parts.par_iter())
            .zip(inputs.par_iter())
            .map(|((members, pieces), input)| {
                let pieces = pieces.as_ref()?;
                let mut owned: Vec<u32> = members
                    .iter()
                    .flat_map(|&c| dag.clusters[c as usize].refs.iter().copied())
                    .collect();
                owned.sort_unstable();
                let mut claimed: Vec<u32> = members
                    .iter()
                    .flat_map(|&c| dag.clusters[c as usize].claimed.iter().copied())
                    .collect();
                claimed.sort_unstable();
                claimed.dedup();
                claimed.retain(|i| owned.binary_search(i).is_err());
                // Triangles of other clusters sharing a position with the group.
                let mut sorted_members = members.clone();
                sorted_members.sort_unstable();
                let mut pids: Vec<u32> = input
                    .iter()
                    .flat_map(|t| t.v.map(|v| dag.vs.pid[v as usize]))
                    .collect();
                pids.sort_unstable();
                pids.dedup();
                let mut near: Vec<(u32, u32)> = Vec::new();
                for p in pids {
                    let lo = by_pid.partition_point(|e| e.0 < p);
                    for &(q, c, t) in &by_pid[lo..] {
                        if q != p {
                            break;
                        }
                        if sorted_members.binary_search(&c).is_err() {
                            near.push((c, t));
                        }
                    }
                }
                near.sort_unstable();
                near.dedup();
                // Originals across the border belong to the bordering clusters: the
                // result near the border is measured against them too.
                let mut bordering: Vec<u32> = near.iter().map(|&(c, _)| c).collect();
                bordering.dedup();
                for c in bordering {
                    claimed.extend_from_slice(&dag.clusters[c as usize].refs);
                }
                claimed.sort_unstable();
                claimed.dedup();
                claimed.retain(|i| owned.binary_search(i).is_err());
                let neighbours: Vec<[V3; 3]> = near
                    .iter()
                    .map(|&(c, t)| {
                        dag.clusters[c as usize].tris[t as usize].map(|v| dag.vs.pos[v as usize])
                    })
                    .collect();
                let piece_tris: Vec<&[[u32; 3]]> = pieces.iter().map(|p| p.0.as_slice()).collect();
                Some(measure(
                    &original,
                    Region {
                        owned: &owned,
                        claimed: &claimed,
                    },
                    &piece_tris,
                    &neighbours,
                    &surface,
                ))
            })
            .collect();

        let mut next: Vec<u32> = Vec::with_capacity(cur.len());
        for ((((members, pieces), meas), cov), st) in groups
            .into_iter()
            .zip(parts)
            .zip(measured)
            .zip(coverage)
            .zip(staged)
        {
            let (Some(pieces), Some(meas), Some(st)) = (pieces, meas, st) else {
                // Kept its triangles: the clusters wait for the next level.
                next.extend_from_slice(&members);
                continue;
            };
            // Group sphere contains every child sphere (so containment is
            // monotone up the DAG) and every vertex the simplifier produced.
            let mut sphere = dag.clusters[members[0] as usize].sphere;
            let mut child_err = 0.0f32;
            for &c in &members {
                sphere = sphere_union(sphere, dag.clusters[c as usize].sphere);
                child_err = child_err.max(dag.clusters[c as usize].self_error);
            }
            for (t, _) in &st {
                for &v in t {
                    let p = dag.vs.pos[v as usize];
                    sphere = sphere_union(sphere, [p[0], p[1], p[2], 0.0]);
                }
            }
            let floor = 1e-5 * sphere[3].max(1e-6);
            let mut error = match cov {
                Some((n, foot)) => {
                    let spacing = sphere[3] * (4.18879 / n.max(1) as f32).cbrt();
                    child_err
                        .max(COVERAGE_FOOTPRINT * foot)
                        .max(COVERAGE_FACTOR * spacing)
                        .max(floor)
                }
                // The measured two-sided deviation from the original surface
                // (position, and normal/uv in the same unit) is the stored
                // error: a bound on what swapping this group in changes, with
                // nothing relaxed for density (the runtime controls overdraw
                // per tile instead). It already contains everything earlier
                // levels changed, so it is combined with the children's error
                // by max, only to keep the DAG monotone.
                None => child_err.max(meas.position).max(meas.shading).max(floor),
            };
            if error <= child_err {
                error = f32::from_bits(child_err.to_bits() + 1);
            }

            let gid = dag.groups.len() as u32;
            for &c in &members {
                let cn = &mut dag.clusters[c as usize];
                cn.parent_error = error;
                cn.parent_sphere = sphere;
                cn.consumer = gid;
                cn.refs = Vec::new();
                cn.claimed = Vec::new();
            }
            let mut produced = Vec::new();
            for (((tris, mat), refs), claimed) in pieces
                .into_iter()
                .zip(meas.piece_refs)
                .zip(meas.piece_claimed)
            {
                produced.push(dag.clusters.len() as u32);
                next.push(dag.clusters.len() as u32);
                dag.clusters.push(ClusterNode {
                    tris,
                    mat,
                    sphere,
                    parent_sphere: sphere,
                    self_error: error,
                    parent_error: f32::INFINITY,
                    level: level + 1,
                    producer: gid,
                    consumer: NONE,
                    refs,
                    claimed,
                });
            }
            dag.groups.push(Group {
                clusters: members,
                level,
                sphere,
                error,
                is_root: false,
                produced,
            });
        }
        next.sort_unstable();
        cur = next;
        level += 1;
    }

    // ── roots ──────────────────────────────────────────────────────────────
    for c in &cur {
        dag.clusters[*c as usize].refs = Vec::new();
        dag.clusters[*c as usize].claimed = Vec::new();
    }
    for chunk in cur.chunks(ROOT_CHUNK) {
        let gid = dag.groups.len() as u32;
        let mut sphere = dag.clusters[chunk[0] as usize].sphere;
        for &c in chunk {
            sphere = sphere_union(sphere, dag.clusters[c as usize].sphere);
        }
        for &c in chunk {
            let cn = &mut dag.clusters[c as usize];
            cn.parent_error = f32::INFINITY;
            cn.parent_sphere = cn.sphere;
            cn.consumer = gid;
        }
        dag.groups.push(Group {
            clusters: chunk.to_vec(),
            level,
            sphere,
            error: f32::INFINITY,
            is_root: true,
            produced: Vec::new(),
        });
    }
    Ok(dag)
}
