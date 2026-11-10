//! Measured object-space deviation of a simplified group.
//!
//! Every cluster carries *reference points*: samples of the original
//! (level-0) surface that it stands in for. When a group is simplified, each
//! reference point of its input clusters is measured against the simplified
//! surface; the maximum distance is the group's error. Because the points
//! are original geometry (not the previous level's vertices), the measured
//! value already contains everything accumulated by earlier levels, so errors
//! neither need to be summed nor scaled by a safety factor: parent error is
//! just `max(children, measured)` (plus the strict-ordering epsilon).
//!
//! Points then follow their nearest simplified triangle into the produced
//! cluster, so the next level keeps measuring against the original surface.
//!
//! Position is not the only thing a swap can change. A reference point also
//! carries the shading attributes of the original surface (normal, uv0, uv1)
//! when the vertex records use the format's layout (`VertexAttr`). The
//! simplified surface interpolates the attributes of the vertices it kept, so
//! at the point's nearest simplified triangle the interpolated attributes are
//! compared with the original ones and the difference is converted to the same
//! unit as the position error, an object-space length (`Measured::shading`):
//!
//! * normal: the chord `|n_orig - n_interp|` is the tilt in radians; a tilt
//!   `t` acting over half an output edge `e/2` displaces the shaded surface by
//!   `t * e / 2`;
//! * uv: the uv distance times the object-space length per uv unit of the
//!   group's input triangles (`sqrt(sum area / sum uv-area)`), i.e. how far the
//!   texture slides along the surface.
//!
//! The group error is the maximum of the three, so it stays a bound on what a
//! swap can change on screen. Points on attribute seams (vertices shared by
//! several attribute records) carry no attributes: their nearest triangle may
//! lie on the other side of the seam, where the attributes legitimately differ.

use helio_vg_format::VertexAttr;

use crate::math::*;

/// Maximum reference points a cluster passes on to its consumer group.
pub const MAX_REFS_PER_CLUSTER: usize = 256;

/// Closest point of triangle `abc` to `p` (Ericson, RTCD 5.1.5): squared
/// distance and the barycentric weights of `a`, `b` and `c` at that point.
pub fn closest_on_tri(p: V3, a: V3, b: V3, c: V3) -> (f32, [f32; 3]) {
    let (ab, ac, ap) = (sub(b, a), sub(c, a), sub(p, a));
    let q = |q: V3, w: [f32; 3]| {
        let d = sub(p, q);
        (dot(d, d), w)
    };
    let (d1, d2) = (dot(ab, ap), dot(ac, ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return q(a, [1.0, 0.0, 0.0]);
    }
    let bp = sub(p, b);
    let (d3, d4) = (dot(ab, bp), dot(ac, bp));
    if d3 >= 0.0 && d4 <= d3 {
        return q(b, [0.0, 1.0, 0.0]);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let t = d1 / (d1 - d3);
        return q(add(a, scale(ab, t)), [1.0 - t, t, 0.0]);
    }
    let cp = sub(p, c);
    let (d5, d6) = (dot(ab, cp), dot(ac, cp));
    if d6 >= 0.0 && d5 <= d6 {
        return q(c, [0.0, 0.0, 1.0]);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let t = d2 / (d2 - d6);
        return q(add(a, scale(ac, t)), [1.0 - t, 0.0, t]);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return q(add(b, scale(sub(c, b), w)), [0.0, 1.0 - w, w]);
    }
    let denom = 1.0 / (va + vb + vc);
    let (v, w) = (vb * denom, vc * denom);
    q(add(add(a, scale(ab, v)), scale(ac, w)), [1.0 - v - w, v, w])
}

/// Squared distance from `p` to triangle `abc`.
pub fn point_tri_dist2(p: V3, a: V3, b: V3, c: V3) -> f32 {
    closest_on_tri(p, a, b, c).0
}

/// Shading attributes of a point of the original surface.
#[derive(Clone, Copy, Debug)]
pub struct Shading {
    /// Unit normal.
    pub normal: V3,
    pub uv: [[f32; 2]; 2],
}

impl Shading {
    pub fn of(a: &VertexAttr) -> Option<Self> {
        let len = length(a.normal);
        (len > 1e-3 && a.uv0.iter().chain(&a.uv1).all(|x| x.is_finite())).then(|| Self {
            normal: scale(a.normal, 1.0 / len),
            uv: [a.uv0, a.uv1],
        })
    }

    /// Weighted blend of three corners (weights sum to 1).
    pub fn blend(c: [&Shading; 3], w: [f32; 3]) -> Option<Self> {
        let mut n = [0.0; 3];
        let mut uv = [[0.0; 2]; 2];
        for k in 0..3 {
            n = add(n, scale(c[k].normal, w[k]));
            for (ch, out) in uv.iter_mut().enumerate() {
                for (i, o) in out.iter_mut().enumerate() {
                    *o += c[k].uv[ch][i] * w[k];
                }
            }
        }
        let len = length(n);
        (len > 1e-3).then(|| Self {
            normal: scale(n, 1.0 / len),
            uv,
        })
    }
}

/// A sample of the original surface the DAG stands in for.
#[derive(Clone, Copy, Debug)]
pub struct RefPoint {
    pub p: V3,
    /// Attributes at `p`; `None` when unknown or not comparable (see the
    /// module docs).
    pub shading: Option<Shading>,
}

/// What `measure` needs to know about the vertices of one build.
pub struct Surface<'a> {
    pub pos: &'a [V3],
    /// Attributes of a unified vertex, `None` when the vertex records do not
    /// use the format layout (position-only measurement).
    pub attr: &'a (dyn Fn(u32) -> Option<VertexAttr> + Sync),
}

/// Small uniform grid over the triangles of one simplified group.
struct TriGrid {
    tris: Vec<[V3; 3]>,
    lo: V3,
    cell: f32,
    dims: [i32; 3],
    start: Vec<u32>,
    items: Vec<u32>,
}

impl TriGrid {
    fn new(tris: Vec<[V3; 3]>) -> Self {
        let mut lo = [f32::INFINITY; 3];
        let mut hi = [f32::NEG_INFINITY; 3];
        let mut area = 0.0f64;
        for t in &tris {
            for v in t {
                for k in 0..3 {
                    lo[k] = lo[k].min(v[k]);
                    hi[k] = hi[k].max(v[k]);
                }
            }
            area += 0.5 * length(cross(sub(t[1], t[0]), sub(t[2], t[0]))) as f64;
        }
        let ext = (0..3).map(|k| hi[k] - lo[k]).fold(0.0f32, f32::max);
        let by_area = (2.0 * area / tris.len().max(1) as f64).sqrt() as f32;
        let cell = by_area.max(ext / 24.0).max(1e-12);
        let dims: [i32; 3] = std::array::from_fn(|k| (((hi[k] - lo[k]) / cell) as i32 + 1).max(1));
        let ncell = (dims[0] * dims[1] * dims[2]) as usize;
        let mut g = Self {
            tris,
            lo,
            cell,
            dims,
            start: vec![0; ncell + 1],
            items: Vec::new(),
        };
        let ranges: Vec<([i32; 3], [i32; 3])> = g
            .tris
            .iter()
            .map(|t| {
                let mut a = [f32::INFINITY; 3];
                let mut b = [f32::NEG_INFINITY; 3];
                for v in t {
                    for k in 0..3 {
                        a[k] = a[k].min(v[k]);
                        b[k] = b[k].max(v[k]);
                    }
                }
                (g.key(a), g.key(b))
            })
            .collect();
        for (a, b) in &ranges {
            for x in a[0]..=b[0] {
                for y in a[1]..=b[1] {
                    for z in a[2]..=b[2] {
                        let ci = g.cell_index(x, y, z);
                        g.start[ci + 1] += 1;
                    }
                }
            }
        }
        for i in 0..ncell {
            g.start[i + 1] += g.start[i];
        }
        g.items = vec![0; g.start[ncell] as usize];
        let mut fill = g.start.clone();
        for (i, (a, b)) in ranges.iter().enumerate() {
            for x in a[0]..=b[0] {
                for y in a[1]..=b[1] {
                    for z in a[2]..=b[2] {
                        let c = g.cell_index(x, y, z);
                        g.items[fill[c] as usize] = i as u32;
                        fill[c] += 1;
                    }
                }
            }
        }
        g
    }

    fn cell_index(&self, x: i32, y: i32, z: i32) -> usize {
        ((z * self.dims[1] + y) * self.dims[0] + x) as usize
    }

    fn key(&self, p: V3) -> [i32; 3] {
        std::array::from_fn(|k| {
            (((p[k] - self.lo[k]) / self.cell).floor() as i32).clamp(0, self.dims[k] - 1)
        })
    }

    /// Nearest triangle to `p`: (distance, index).
    fn nearest(&self, p: V3) -> (f32, u32) {
        let c = self.key(p);
        let max_r = self.dims[0].max(self.dims[1]).max(self.dims[2]);
        let mut best = (f32::INFINITY, u32::MAX);
        for r in 0..=max_r {
            if r > 0 && best.0 <= (r - 1) as f32 * self.cell {
                break;
            }
            for z in (c[2] - r).max(0)..=(c[2] + r).min(self.dims[2] - 1) {
                for y in (c[1] - r).max(0)..=(c[1] + r).min(self.dims[1] - 1) {
                    let shell = (z - c[2]).abs() == r || (y - c[1]).abs() == r;
                    let step = if shell { 1 } else { (2 * r).max(1) };
                    let mut x = c[0] - r;
                    while x <= c[0] + r {
                        if x >= 0 && x < self.dims[0] {
                            let cell = self.cell_index(x, y, z);
                            for &i in &self.items
                                [self.start[cell] as usize..self.start[cell + 1] as usize]
                            {
                                let t = &self.tris[i as usize];
                                let d = point_tri_dist2(p, t[0], t[1], t[2]).sqrt();
                                if d < best.0 {
                                    best = (d, i);
                                }
                            }
                        }
                        x += step;
                    }
                }
            }
        }
        best
    }
}

pub struct Measured {
    /// Max distance of simplified-surface samples (edge midpoints, centroids)
    /// to the group's input surface: catches long triangles bridging detail
    /// that no reference point sits under.
    pub forward: f32,
    /// Max distance of any reference point to the simplified surface.
    pub error: f32,
    /// Max normal / uv deviation of any reference point, as an object-space
    /// length (see the module docs); 0 without attributes.
    pub shading: f32,
    /// Reference points assigned to each output piece (nearest triangle).
    pub piece_refs: Vec<Vec<RefPoint>>,
}

/// Object-space length per uv unit of the triangles `ids`, per uv channel:
/// `sqrt(surface area / uv area)`, 0 when the channel has no area.
fn uv_scale(ids: &[[u32; 3]], surf: &Surface) -> [f32; 2] {
    let mut area = 0.0f64;
    let mut uv_area = [0.0f64; 2];
    for t in ids {
        let attrs = t.map(|v| (surf.attr)(v));
        let (Some(a0), Some(a1), Some(a2)) = (attrs[0], attrs[1], attrs[2]) else {
            continue;
        };
        let p = t.map(|v| surf.pos[v as usize]);
        area += 0.5 * f64::from(length(cross(sub(p[1], p[0]), sub(p[2], p[0]))));
        for (ch, ua) in uv_area.iter_mut().enumerate() {
            let uv = |a: &VertexAttr| if ch == 0 { a.uv0 } else { a.uv1 };
            let (u0, u1, u2) = (uv(&a0), uv(&a1), uv(&a2));
            let e1 = [u1[0] - u0[0], u1[1] - u0[1]];
            let e2 = [u2[0] - u0[0], u2[1] - u0[1]];
            *ua += 0.5 * f64::from((e1[0] * e2[1] - e1[1] * e2[0]).abs());
        }
    }
    uv_area.map(|u| {
        if u > 1e-12 && area > 0.0 {
            (area / u).sqrt() as f32
        } else {
            0.0
        }
    })
}

/// Measure `refs` against the simplified `pieces` (triangles per produced
/// cluster) and distribute the points (capped) to the pieces. `input` are the
/// group's input triangles.
pub fn measure(
    refs: &[RefPoint],
    input: &[[u32; 3]],
    pieces: &[&[[u32; 3]]],
    surf: &Surface,
) -> Measured {
    let pos = surf.pos;
    let mut tris = Vec::new();
    let mut ids = Vec::new();
    let mut owner = Vec::new();
    for (pi, p) in pieces.iter().enumerate() {
        for t in p.iter() {
            tris.push(t.map(|v| pos[v as usize]));
            ids.push(*t);
            owner.push(pi as u32);
        }
    }
    let mut per_piece: Vec<Vec<(f32, RefPoint)>> = vec![Vec::new(); pieces.len()];
    let mut error = 0.0f32;
    let mut shading = 0.0f32;
    let mut forward = 0.0f32;
    if !input.is_empty() {
        let g = TriGrid::new(input.iter().map(|t| t.map(|v| pos[v as usize])).collect());
        for t in &tris {
            let m = |a: V3, b: V3| scale(add(a, b), 0.5);
            let c = scale(add(add(t[0], t[1]), t[2]), 1.0 / 3.0);
            for p in [c, m(t[0], t[1]), m(t[1], t[2]), m(t[2], t[0])] {
                forward = forward.max(g.nearest(p).0);
            }
        }
    }
    if !tris.is_empty() {
        // Units of the attribute deviations: half the mean output edge for a
        // normal tilt, object-space length per uv unit for each uv channel.
        let mean_edge = {
            let sum: f64 = tris
                .iter()
                .map(|t| f64::from(dist(t[0], t[1]) + dist(t[1], t[2]) + dist(t[2], t[0])))
                .sum();
            (sum / (3 * tris.len()) as f64) as f32
        };
        let uv_len = uv_scale(input, surf);
        let grid = TriGrid::new(tris);
        for &r in refs {
            let (d, t) = grid.nearest(r.p);
            let mut e = d;
            error = error.max(d);
            if let Some(want) = r.shading {
                let corners = ids[t as usize].map(|v| (surf.attr)(v).and_then(|a| Shading::of(&a)));
                if let (Some(c0), Some(c1), Some(c2)) = (corners[0], corners[1], corners[2]) {
                    let tp = grid.tris[t as usize];
                    let (_, w) = closest_on_tri(r.p, tp[0], tp[1], tp[2]);
                    if let Some(got) = Shading::blend([&c0, &c1, &c2], w) {
                        let tilt = length(sub(want.normal, got.normal));
                        let mut len = 0.5 * tilt * mean_edge;
                        for ch in 0..2 {
                            let du = [
                                want.uv[ch][0] - got.uv[ch][0],
                                want.uv[ch][1] - got.uv[ch][1],
                            ];
                            len = len.max((du[0] * du[0] + du[1] * du[1]).sqrt() * uv_len[ch]);
                        }
                        shading = shading.max(len);
                        e = e.max(len);
                    }
                }
            }
            per_piece[owner[t as usize] as usize].push((e, r));
        }
    }
    let piece_refs = per_piece.into_iter().map(cap_refs).collect();
    Measured {
        forward,
        error,
        shading,
        piece_refs,
    }
}

/// Keep at most [`MAX_REFS_PER_CLUSTER`] points: the worst half by measured
/// deviation (the ones that define the error) plus an even stride over the
/// rest (so the whole surface stays covered). Deterministic.
fn cap_refs(mut pts: Vec<(f32, RefPoint)>) -> Vec<RefPoint> {
    if pts.len() <= MAX_REFS_PER_CLUSTER {
        return pts.into_iter().map(|p| p.1).collect();
    }
    let keep_worst = MAX_REFS_PER_CLUSTER / 2;
    let mut idx: Vec<u32> = (0..pts.len() as u32).collect();
    idx.sort_unstable_by(|&a, &b| {
        pts[b as usize]
            .0
            .total_cmp(&pts[a as usize].0)
            .then(a.cmp(&b))
    });
    let mut taken = vec![false; pts.len()];
    for &i in &idx[..keep_worst] {
        taken[i as usize] = true;
    }
    let rest: Vec<usize> = (0..pts.len()).filter(|&i| !taken[i]).collect();
    let want = MAX_REFS_PER_CLUSTER - keep_worst;
    for k in 0..want {
        taken[rest[k * rest.len() / want]] = true;
    }
    let mut out = Vec::with_capacity(MAX_REFS_PER_CLUSTER);
    for (i, p) in pts.drain(..).enumerate() {
        if taken[i] {
            out.push(p.1);
        }
    }
    out
}

/// Distance of every point to the nearest of `tris`.
pub fn point_deviations(points: &[V3], tris: &[[V3; 3]]) -> Vec<f32> {
    if tris.is_empty() {
        return vec![f32::INFINITY; points.len()];
    }
    let grid = TriGrid::new(tris.to_vec());
    points.iter().map(|&p| grid.nearest(p).0).collect()
}
