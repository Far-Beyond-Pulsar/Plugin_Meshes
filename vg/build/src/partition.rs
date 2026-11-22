//! Partition one DAG level's clusters into simplification groups.
//!
//! A group's locked border (the vertices it shares with other groups) is
//! what limits how far it can simplify, so partitions should be compact
//! and have short borders. `meshopt_partitionClusters` does this on the
//! shared-vertex graph (by position, so attribute seams do not cut it),
//! weighted by spatial proximity so disconnected pieces still group with their
//! neighbours.
//!
//! Group size: groups target [`TARGET_GROUP`] clusters (the partitioner may
//! exceed a target by up to a third), capped so a group's streams fit one
//! page. Larger groups lock fewer vertices per cluster (border length grows
//! with the square root of a group's area) but make each level-of-detail
//! switch coarser in area; 16 keeps the locked fraction of an interior group
//! near a quarter of its vertices at 128-triangle clusters. While the whole
//! level fits one group it becomes one group, with no interior border at all.

use crate::dag::ClusterNode;
use crate::mesh::VertexStore;

/// Clusters per group the partitioner aims for.
const TARGET_GROUP: usize = 16;

/// Returns groups of ids taken from `cur` (each id indexes `clusters`).
/// `cap` is the largest group a page holds.
pub fn partition(
    vs: &VertexStore,
    cur: &[u32],
    clusters: &[ClusterNode],
    cap: usize,
) -> Vec<Vec<u32>> {
    if cur.len() <= cap {
        return vec![cur.to_vec()];
    }
    // Position ids of the level, compacted.
    let mut pids: Vec<u32> = cur
        .iter()
        .flat_map(|&c| clusters[c as usize].tris.iter().flatten())
        .map(|&v| vs.pid[v as usize])
        .collect();
    pids.sort_unstable();
    pids.dedup();
    let mut positions = vec![[0.0f32; 3]; pids.len()];
    let mut indices = Vec::new();
    let mut counts = Vec::with_capacity(cur.len());
    for &c in cur {
        let tris = &clusters[c as usize].tris;
        for &v in tris.iter().flatten() {
            let l = pids.binary_search(&vs.pid[v as usize]).unwrap();
            positions[l] = vs.pos[v as usize];
            indices.push(l as u32);
        }
        counts.push(tris.len() as u32 * 3);
    }
    let vertices = meshopt::VertexDataAdapter::new(bytemuck::cast_slice(&positions), 12, 0)
        .expect("packed float3 positions");
    // The partitioner may exceed its target by a third.
    let target = TARGET_GROUP.min(cap * 3 / 4).max(1);
    let mut part = vec![0u32; cur.len()];
    let n =
        meshopt::partition_clusters_with_positions(&mut part, &indices, &counts, &vertices, target);
    let mut groups: Vec<Vec<u32>> = vec![Vec::new(); n];
    for (&c, &p) in cur.iter().zip(&part) {
        groups[p as usize].push(c);
    }
    groups.retain(|g| !g.is_empty());
    // The page bound is hard: halve any group that still exceeds it
    // (spatially, so the halves stay compact).
    let mut out = Vec::with_capacity(groups.len());
    while let Some(mut g) = groups.pop() {
        if g.len() <= cap {
            out.push(g);
            continue;
        }
        let centre = |c: u32| {
            let s = clusters[c as usize].sphere;
            [s[0], s[1], s[2]]
        };
        let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
        for &c in &g {
            let p = centre(c);
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        let axis = (0..3)
            .max_by(|&a, &b| (hi[a] - lo[a]).total_cmp(&(hi[b] - lo[b])))
            .unwrap();
        g.sort_by(|&a, &b| centre(a)[axis].total_cmp(&centre(b)[axis]).then(a.cmp(&b)));
        let right = g.split_off(g.len() / 2);
        groups.push(g);
        groups.push(right);
    }
    out.sort_by_key(|g| g[0]);
    out
}
