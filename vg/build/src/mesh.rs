//! Input mesh, vertex welding and the builder's global vertex store.
//!
//! The builder works on *unified* vertices: one per distinct
//! (position, attribute bytes) pair. Vertices that share a position but not
//! attributes (UV seams, hard edges) keep separate ids but share a *position
//! id* (`pid`). Topology (edge adjacency, crack checks) is decided on `pid`
//! so a seam is still a closed surface, while simplification treats every
//! seam vertex as locked so attribute discontinuities are never smeared.

use std::collections::HashMap;
use std::fmt;

use rayon::prelude::*;

use crate::math::{pos_key, V3};

/// Indexed triangle mesh handed to the builder.
pub struct MeshInput<'a> {
    pub positions: &'a [[f32; 3]],
    /// Opaque per-vertex attribute bytes (normals, uvs, tangents ... packed
    /// however the caller likes): `positions.len() * attribute_stride` bytes.
    /// Copied verbatim into the page attribute stream.
    pub attributes: &'a [u8],
    pub attribute_stride: usize,
    pub indices: &'a [u32],
    /// Material section per triangle; empty means "all section 0".
    pub tri_material: &'a [u32],
}

#[derive(Debug, PartialEq, Eq)]
pub enum BuildError {
    EmptyMesh,
    BadIndexCount,
    IndexOutOfRange,
    BadMaterialArray,
    BadAttributeArray,
    MaterialTooLarge,
    NonFinitePosition,
    /// A cluster group's streams do not fit in one 128 KiB page (attribute
    /// stride too large).
    GroupTooLarge,
    FileTooLarge,
    /// The DAG could not be reduced below [`crate::MAX_ROOT_PAGES`] root pages
    /// (geometry that does not simplify): the asset must not be pinned.
    RootTooLarge {
        pages: u32,
    },
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for BuildError {}

impl MeshInput<'_> {
    pub fn validate(&self) -> Result<(), BuildError> {
        if self.positions.iter().flatten().any(|x| !x.is_finite()) {
            return Err(BuildError::NonFinitePosition);
        }
        if self.indices.len() % 3 != 0 {
            return Err(BuildError::BadIndexCount);
        }
        if self
            .indices
            .iter()
            .any(|&i| i as usize >= self.positions.len())
        {
            return Err(BuildError::IndexOutOfRange);
        }
        let tris = self.indices.len() / 3;
        if !self.tri_material.is_empty() && self.tri_material.len() != tris {
            return Err(BuildError::BadMaterialArray);
        }
        if self.tri_material.iter().any(|&m| m > 0xffff) {
            return Err(BuildError::MaterialTooLarge);
        }
        if self.positions.len().checked_mul(self.attribute_stride) != Some(self.attributes.len()) {
            return Err(BuildError::BadAttributeArray);
        }
        Ok(())
    }
}

/// A triangle over unified vertex ids with its material section.
#[derive(Clone, Copy, Debug)]
pub struct Tri {
    pub v: [u32; 3],
    pub mat: u32,
}

/// Growing vertex table. Simplification appends the vertices it places; ids
/// of existing vertices never change, so child clusters keep referring to the
/// geometry they were built from.
pub struct VertexStore {
    pub pos: Vec<V3>,
    /// Attribute record of this vertex: below [`Self::source_records`] the
    /// index of a source vertex (its bytes in [`MeshInput::attributes`]),
    /// from there on a record the builder wrote (see [`Self::add_record`]).
    pub attr: Vec<u32>,
    /// Number of source vertices (records in the input attribute array).
    pub source_records: u32,
    /// Records written by the builder for vertices whose attributes
    /// simplification updated, `stride` bytes each.
    built: Vec<u8>,
    stride: usize,
    /// Canonical id of the (bit-exact) position.
    pub pid: Vec<u32>,
    /// True if another vertex shares this position (attribute seam).
    pub seam: Vec<bool>,
    /// Vertex belongs to a small connected piece of the source mesh (see
    /// [`mark_floating`]); the flag follows the vertex up the DAG.
    pub floating: Vec<bool>,
    pid_map: HashMap<[u32; 3], u32>,
    pid_first: Vec<u32>,
}

impl VertexStore {
    fn new(source_records: u32, stride: usize) -> Self {
        Self {
            pos: Vec::new(),
            attr: Vec::new(),
            source_records,
            built: Vec::new(),
            stride,
            pid: Vec::new(),
            seam: Vec::new(),
            floating: Vec::new(),
            pid_map: HashMap::new(),
            pid_first: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.pos.len()
    }
    pub fn is_empty(&self) -> bool {
        self.pos.is_empty()
    }

    /// Free the position->pid lookup once welding is done (the map is the
    /// largest transient of a 14M-vertex mesh). The DAG adds vertices only
    /// through [`Self::push_placed`] afterwards, with its own position ids.
    pub fn release_lookup(&mut self) {
        self.pid_map = HashMap::new();
    }

    /// Attribute bytes of vertex `v` (`source` is the input's attribute array).
    pub fn record<'a>(&'a self, v: u32, source: &'a [u8]) -> &'a [u8] {
        let r = self.attr[v as usize];
        if r < self.source_records {
            let at = r as usize * self.stride;
            &source[at..at + self.stride]
        } else {
            let at = (r - self.source_records) as usize * self.stride;
            &self.built[at..at + self.stride]
        }
    }

    /// Store a builder-written attribute record (`stride` bytes); returns its
    /// index for [`Self::attr`].
    pub fn add_record(&mut self, bytes: &[u8]) -> u32 {
        debug_assert_eq!(bytes.len(), self.stride);
        let id = self.source_records + (self.built.len() / self.stride.max(1)) as u32;
        self.built.extend_from_slice(bytes);
        id
    }

    /// A fresh position id, for the wedges (vertices at one position with
    /// different attributes) of one vertex the builder placed.
    pub fn new_pid(&mut self) -> u32 {
        let pid = self.pid_first.len() as u32;
        self.pid_first.push(self.pos.len() as u32);
        pid
    }

    /// Append a vertex the builder placed, at position id `pid` (from
    /// [`Self::new_pid`]); `seam` when the position has several wedges.
    pub fn push_placed(&mut self, pos: V3, attr: u32, pid: u32, seam: bool, floating: bool) -> u32 {
        let id = self.pos.len() as u32;
        self.pos.push(pos);
        self.attr.push(attr);
        self.pid.push(pid);
        self.seam.push(seam);
        self.floating.push(floating);
        id
    }

    /// Number of position ids handed out so far.
    pub fn pid_count(&self) -> u32 {
        self.pid_first.len() as u32
    }

    pub fn push(&mut self, pos: V3, attr: u32) -> u32 {
        let id = self.pos.len() as u32;
        let next = self.pid_first.len() as u32;
        let mut seam = false;
        let pid = match self.pid_map.get(&pos_key(pos)) {
            Some(&p) => {
                // Two distinct vertices at one position: both are seams from
                // now on so neither is ever moved.
                seam = true;
                self.seam[self.pid_first[p as usize] as usize] = true;
                p
            }
            None => {
                self.pid_map.insert(pos_key(pos), next);
                self.pid_first.push(id);
                next
            }
        };
        self.pos.push(pos);
        self.attr.push(attr);
        self.pid.push(pid);
        self.seam.push(seam);
        self.floating.push(false);
        id
    }
}

/// Weld identical (position, attribute) vertices, drop triangles that are
/// degenerate in position, and return the unified store + triangle list.
pub fn weld(mesh: &MeshInput) -> (VertexStore, Vec<Tri>) {
    let n = mesh.positions.len();
    let stride = mesh.attribute_stride;
    let attr = |i: u32| &mesh.attributes[i as usize * stride..(i as usize + 1) * stride];

    let mut order: Vec<u32> = (0..n as u32).collect();
    // Index is the final tiebreak, so the sort is total and deterministic.
    order.par_sort_unstable_by(|&a, &b| {
        pos_key(mesh.positions[a as usize])
            .cmp(&pos_key(mesh.positions[b as usize]))
            .then_with(|| attr(a).cmp(attr(b)))
            .then(a.cmp(&b))
    });
    // rep[i] = smallest source index with identical (pos, attr).
    let mut rep = vec![0u32; n];
    let mut run_start = 0;
    for k in 0..n {
        let i = order[k];
        let same = k > 0 && {
            let p = order[run_start];
            pos_key(mesh.positions[p as usize]) == pos_key(mesh.positions[i as usize])
                && attr(p) == attr(i)
        };
        if !same {
            run_start = k;
        }
        rep[i as usize] = order[run_start];
    }
    // Unified ids in ascending representative order (keeps source locality).
    let mut store = VertexStore::new(n as u32, stride);
    let mut unified = vec![u32::MAX; n];
    for i in 0..n {
        if rep[i] as usize == i {
            unified[i] = store.push(mesh.positions[i], i as u32);
        }
    }
    for i in 0..n {
        unified[i] = unified[rep[i] as usize];
    }

    let mut tris = Vec::with_capacity(mesh.indices.len() / 3);
    for (t, c) in mesh.indices.chunks_exact(3).enumerate() {
        let v = [
            unified[c[0] as usize],
            unified[c[1] as usize],
            unified[c[2] as usize],
        ];
        let p = [
            store.pid[v[0] as usize],
            store.pid[v[1] as usize],
            store.pid[v[2] as usize],
        ];
        if p[0] == p[1] || p[1] == p[2] || p[0] == p[2] {
            continue;
        }
        let mat = if mesh.tri_material.is_empty() {
            0
        } else {
            mesh.tri_material[t]
        };
        tris.push(Tri { v, mat });
    }
    (store, tris)
}

/// Source connected components (by position id) with at most
/// `max_tris` triangles are "floating pieces" (needle/leaf cards, branch
/// fragments): flag their vertices. Decided once on the source mesh so that a
/// coarse level's small remainder of a big surface is never mistaken for one.
pub fn mark_floating(vs: &mut VertexStore, tris: &[Tri], max_tris: usize) {
    let np = vs.pid.iter().copied().max().map_or(0, |m| m as usize + 1);
    let mut parent: Vec<u32> = (0..np as u32).collect();
    fn find(p: &mut [u32], mut x: u32) -> u32 {
        while p[x as usize] != x {
            p[x as usize] = p[p[x as usize] as usize];
            x = p[x as usize];
        }
        x
    }
    for t in tris {
        let a = find(&mut parent, vs.pid[t.v[0] as usize]);
        for k in 1..3 {
            let a = find(&mut parent, a);
            let b = find(&mut parent, vs.pid[t.v[k] as usize]);
            if a != b {
                parent[b as usize] = a;
            }
        }
    }
    let mut size = vec![0u32; np];
    for t in tris {
        let r = find(&mut parent, vs.pid[t.v[0] as usize]);
        size[r as usize] += 1;
    }
    for i in 0..vs.len() {
        let r = find(&mut parent, vs.pid[i]);
        vs.floating[i] = size[r as usize] as usize <= max_tris && size[r as usize] > 0;
    }
}
