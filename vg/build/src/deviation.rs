//! Measured object-space deviation of a simplified group from the original
//! surface.
//!
//! Every cluster of the current level stands for a set of *original*
//! (level-0) triangles; each original triangle belongs to exactly one of
//! them. When a group is simplified, its result `S` is compared with the
//! original triangles `O` its input clusters stand for, in both directions,
//! as a two-sided Hausdorff distance:
//!
//! * `O -> S`: how far any point of the original surface is from `S` (detail
//!   that was removed);
//! * `S -> O`: how far any point of `S` is from the original surface
//!   (triangles bridging gaps or cutting corners).
//!
//! Each is a rigorous upper bound, not a sample maximum. The distance to one
//! triangle is a convex function of the query point, so over a triangle `T`
//! its maximum lies at a corner of `T`: for any candidate triangle `s` of the
//! other surface, `max_{x in T} d(x, S) <= max_{x in T} d(x, s) = max over the
//! corners of T of d(corner, s)`. The candidates are the nearest triangles to
//! `T`'s corners and centroid; the smallest such corner maximum bounds `T`.
//! Where the bound is loose (`T` straddles several candidates) and could
//! decide the group's error, `T` is split at its edge midpoints and the halves
//! bounded again (branch and bound). The bound is exact on smooth regions,
//! where one candidate covers the whole triangle, and never underestimates.
//!
//! Because `O` is the original surface (not the previous level), the measured
//! value already contains everything earlier levels changed: the parent error
//! is `max(children, measured)`, kept only for monotonicity, never a sum and
//! never a safety factor.
//!
//! Original triangles then follow their nearest simplified triangle into the
//! produced cluster, so the next level measures against the original surface
//! again.
//!
//! Position is not the only thing a swap can change. When the vertex records
//! use the format's layout (`VertexAttr`), the original surface's shading
//! attributes (normal, uv0, uv1) at the corners and centroid of every original
//! triangle are compared with the attributes `S` interpolates at the nearest
//! point, and the difference is converted to the unit of the position error,
//! an object-space length (`Measured::shading`):
//!
//! * normal: the chord `|n_orig - n_interp|` is the tilt in radians; a tilt
//!   `t` acting over half an output edge `e/2` displaces the shaded surface by
//!   `t * e / 2`;
//! * uv: the uv distance times the object-space length per uv unit of the
//!   group's input triangles (`sqrt(sum area / sum uv-area)`), i.e. how far the
//!   texture slides along the surface.
//!
//! Shading is a sampled appearance measure (attributes have no distance
//! bound). Corners on attribute seams (vertices shared by several attribute
//! records) are not compared: their nearest triangle may lie on the other
//! side of the seam, where the attributes legitimately differ.

use helio_vg_format::VertexAttr;

use crate::math::*;

/// A triangle's distance bound is refined only while it exceeds the group's
/// sampled maximum by more than this fraction. The result is a rigorous bound
/// whatever the value; it trades build time against how tight the bound is.
const REFINE_SLACK: f32 = 1.0 / 16.0;
/// Bounds are not tightened below this fraction of the simplified surface's
/// mean edge. A group whose error is that small is swapped in once its
/// triangles are `1 / REFINE_FLOOR` pixels across at the error threshold, so
/// a tighter bound could only coarsen surfaces already drawn with triangles
/// that large (a flat wall filling a 2000 pixel screen keeps a few thousand
/// triangles instead of two), while proving it costs subdivision everywhere
/// on exactly flat surfaces, where the true distance is zero.
const REFINE_FLOOR: f32 = 1.0 / 4096.0;
/// Triangles refined together (in parallel) against one proven bound.
const REFINE_BATCH: usize = 256;
/// `1 / sqrt(3)`: a triangle's points lie within this fraction of its longest
/// edge of one of its corners.
const FRAC_1_SQRT_3: f32 = 0.577_350_3;
/// Midpoint subdivisions of one triangle at most when tightening its bound
/// (4^depth pieces).
const REFINE_DEPTH: u32 = 8;

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

/// Small uniform grid over a set of triangles.
struct TriGrid {
    tris: Vec<[V3; 3]>,
    /// Triangles around each vertex position (the fan, see [`fan_bound`]).
    fans: std::collections::HashMap<[u32; 3], Vec<u32>>,
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
        // Cells about the size of a triangle (so a query scans few), but never
        // more cells than a small multiple of the triangles (memory).
        let by_area = (2.0 * area / tris.len().max(1) as f64).sqrt() as f32;
        let span: Vec<f32> = (0..3).map(|k| (hi[k] - lo[k]).max(by_area)).collect();
        let by_count =
            ((span[0] * span[1] * span[2]) as f64 / (8 * tris.len().max(1)) as f64).cbrt() as f32;
        let cell = by_area.max(by_count).max(1e-12);
        let dims: [i32; 3] = std::array::from_fn(|k| (((hi[k] - lo[k]) / cell) as i32 + 1).max(1));
        let ncell = (dims[0] * dims[1] * dims[2]) as usize;
        let mut fans: std::collections::HashMap<[u32; 3], Vec<u32>> =
            std::collections::HashMap::new();
        for (i, t) in tris.iter().enumerate() {
            for p in t {
                fans.entry(p.map(f32::to_bits)).or_default().push(i as u32);
            }
        }
        let mut g = Self {
            tris,
            fans,
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
/// Object-space length per uv unit of the triangles `ids`, per uv channel:
/// `sqrt(surface area / uv area)`, 0 when the channel has no area.
pub(crate) fn uv_scale(ids: &[[u32; 3]], surf: &Surface) -> [f32; 2] {
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

/// What `measure` needs to know about the vertices of one build.
pub struct Surface<'a> {
    pub pos: &'a [V3],
    /// Attributes of a unified vertex, `None` when the vertex records do not
    /// use the format layout (position-only measurement).
    pub attr: &'a (dyn Fn(u32) -> Option<VertexAttr> + Sync),
    /// Vertex is one of several records at its position (attribute seam).
    pub seam: &'a [bool],
}

pub struct Measured {
    /// Two-sided Hausdorff distance bound between the simplified surface and
    /// the original surface (see the module docs).
    pub position: f32,
    /// Max normal / uv deviation, as an object-space length (see the module
    /// docs); 0 without attributes.
    pub shading: f32,
    /// Original triangles (indices into the original list) each output piece
    /// owns (nearest simplified triangle).
    pub piece_refs: Vec<Vec<u32>>,
    /// Further originals each piece claims (see [`Region::claimed`]).
    pub piece_claimed: Vec<Vec<u32>>,
}

/// Sampled and rigorous distance bound of triangle `t` to the surface of
/// `grid`, plus the nearest triangles of its corners and centroid (last).
///
/// The bound is the smallest of: for one candidate `s`, the largest corner
/// distance to `s` (convexity, see the module docs); for two candidates, the
/// same over the two parts of `t` on either side of a plane separating them
/// ([`split_bound`]); and the corners' distances plus `t`'s covering radius.
/// Any plane gives a valid bound; the one between the
/// candidates' regions makes it tight where `t` straddles their shared edge.
fn tri_bound(grid: &TriGrid, t: [V3; 3]) -> (f32, f32, [u32; 4]) {
    let c = scale(add(add(t[0], t[1]), t[2]), 1.0 / 3.0);
    let q = [t[0], t[1], t[2], c].map(|p| grid.nearest(p));
    let lower = q.iter().fold(0.0f32, |m, x| m.max(x.0));
    let mut cand: Vec<u32> = q.iter().map(|x| x.1).collect();
    cand.sort_unstable();
    cand.dedup();
    let mut upper = f32::INFINITY;
    for (i, &a) in cand.iter().enumerate() {
        let sa = &grid.tris[a as usize];
        let single = t
            .iter()
            .map(|&p| point_tri_dist2(p, sa[0], sa[1], sa[2]))
            .fold(0.0f32, f32::max);
        upper = upper.min(single);
        for &b in &cand[i + 1..] {
            upper = upper.min(split_bound(t, sa, &grid.tris[b as usize]));
        }
    }
    // Fans around the candidates' vertices near `t`, while still loose.
    let longest = dist(t[0], t[1]).max(dist(t[1], t[2])).max(dist(t[2], t[0]));
    if upper.sqrt() > lower * (1.0 + REFINE_SLACK) {
        let mut tried: Vec<[u32; 3]> = Vec::new();
        for &a in &cand {
            for v in grid.tris[a as usize] {
                let key = v.map(f32::to_bits);
                if tried.contains(&key) || t.iter().all(|&p| dist(p, v) > longest) {
                    continue;
                }
                tried.push(key);
                let fan: Vec<[V3; 3]> = grid.fans[&key]
                    .iter()
                    .map(|&f| grid.tris[f as usize])
                    .collect();
                if let Some(b) = fan_bound(t, v, &fan) {
                    upper = upper.min(b);
                }
            }
        }
    }
    // Distance is 1-Lipschitz and every point of a triangle lies within
    // longest edge / sqrt(3) of a corner, so the corners' distances bound it too;
    // this is what closes small pieces around the other surface's vertices,
    // where several candidates meet.
    let corners = q[..3].iter().fold(0.0f32, |m, x| m.max(x.0));
    let lipschitz = corners + longest * FRAC_1_SQRT_3;
    (
        lower,
        upper.sqrt().min(lipschitz).max(lower),
        q.map(|x| x.1),
    )
}

/// Squared distance bound of `t` against `s1` on one side of a plane
/// separating `s1` and `s2`, and against `s2` on the other. Each part of `t` is
/// convex, so its maximum distance to its triangle is at the part's corners:
/// `t`'s own corners on that side and the points where `t`'s edges cross the
/// plane (which belong to both parts).
fn split_bound(t: [V3; 3], s1: &[V3; 3], s2: &[V3; 3]) -> f32 {
    let (n, d) = separating_plane(s1, s2);
    let side = t.map(|p| dot(n, p) - d);
    let d1 = |p: V3| point_tri_dist2(p, s1[0], s1[1], s1[2]);
    let d2 = |p: V3| point_tri_dist2(p, s2[0], s2[1], s2[2]);
    let mut worst = 0.0f32;
    for k in 0..3 {
        worst = worst.max(if side[k] >= 0.0 { d1(t[k]) } else { d2(t[k]) });
        let j = (k + 1) % 3;
        if (side[k] < 0.0) != (side[j] < 0.0) {
            let s = side[k] / (side[k] - side[j]);
            let x = add(t[k], scale(sub(t[j], t[k]), s));
            worst = worst.max(d1(x)).max(d2(x));
        }
    }
    worst
}

/// A plane (unit normal `n`, offset `d`; `s1` on the positive side) between
/// two triangles: through their shared edge, perpendicular to their mean
/// surface, when they share one; else the bisector of their centroids.
fn separating_plane(s1: &[V3; 3], s2: &[V3; 3]) -> (V3, f32) {
    let c1 = scale(add(add(s1[0], s1[1]), s1[2]), 1.0 / 3.0);
    let c2 = scale(add(add(s2[0], s2[1]), s2[2]), 1.0 / 3.0);
    let shared: Vec<V3> = s1.iter().copied().filter(|p| s2.contains(p)).collect();
    if let [e0, e1] = shared[..] {
        let n1 = cross(sub(s1[1], s1[0]), sub(s1[2], s1[0]));
        let n2 = cross(sub(s2[1], s2[0]), sub(s2[2], s2[0]));
        let m = add(normalize(n1), normalize(n2));
        let n = normalize(cross(sub(e1, e0), if length(m) > 1e-6 { m } else { n1 }));
        if length(n) > 0.5 {
            let n = if dot(n, sub(c1, e0)) < 0.0 {
                scale(n, -1.0)
            } else {
                n
            };
            return (n, dot(n, e0));
        }
    }
    let n = normalize(sub(c1, c2));
    (n, dot(n, scale(add(c1, c2), 0.5)))
}

/// Squared distance bound of `t` against the fan of triangles `fan` around
/// the vertex `v` they share. Seen along the fan's mean normal `n`, every
/// fan triangle covers an angular wedge around `v`; when the wedges cover the
/// full circle, every point of `t` lies in the wedge (a prism along `n`) of
/// some fan triangle, and its distance to the surface is at most its distance
/// to that triangle. Each wedge cuts a convex part out of `t`, whose maximum
/// distance to its triangle is at the part's corners (convexity). Exact on a
/// flat fan, where one candidate (or two, [`split_bound`]) leaves the points
/// beside it in-plane distances that only deep subdivision would remove.
/// `None` when the wedges leave a gap (an open fan, or a folded one).
fn fan_bound(t: [V3; 3], v: V3, fan: &[[V3; 3]]) -> Option<f32> {
    let mut n = [0.0f32; 3];
    for s in fan {
        n = add(n, cross(sub(s[1], s[0]), sub(s[2], s[0])));
    }
    let n = normalize(n);
    if length(n) < 0.5 {
        return None;
    }
    let u = {
        let a = if n[0].abs() < 0.9 {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        normalize(sub(a, scale(n, dot(a, n))))
    };
    let w = cross(n, u);
    let angle = |p: V3| {
        let d = sub(p, v);
        dot(d, w).atan2(dot(d, u))
    };
    use std::f32::consts::{PI, TAU};
    // (start, span) of each fan triangle's wedge.
    let mut wedges: Vec<(f32, f32, usize)> = Vec::with_capacity(fan.len());
    for (i, s) in fan.iter().enumerate() {
        let others: Vec<V3> = s.iter().copied().filter(|p| *p != v).collect();
        let [a, b] = others[..] else { return None };
        let (ta, tb) = (angle(a), angle(b));
        let mut span = (tb - ta).rem_euclid(TAU);
        let start = if span <= PI {
            ta
        } else {
            span = TAU - span;
            tb
        };
        if span > 0.0 {
            wedges.push((start.rem_euclid(TAU), span, i));
        }
    }
    // The wedges must cover every direction from `v` that `t` reaches: the
    // arc of `t`'s corners when `v` lies outside `t` (seen along `n`), else
    // the full circle. An open fan (the mesh border) can still cover a
    // triangle that lies inside the mesh.
    let rel: Vec<f32> = t
        .iter()
        .filter(|&&p| length(sub(sub(p, v), scale(n, dot(sub(p, v), n)))) > 0.0)
        .map(|&p| angle(p))
        .collect();
    let (arc_start, arc_span) = {
        let mut a = rel.clone();
        a.sort_by(f32::total_cmp);
        // Largest gap between consecutive corner directions: the arc is its
        // complement, when that is less than half a turn.
        let mut best = (0.0f32, 0.0f32);
        for k in 0..a.len() {
            let next = if k + 1 < a.len() {
                a[k + 1]
            } else {
                a[0] + TAU
            };
            if next - a[k] > best.0 {
                best = (next - a[k], next);
            }
        }
        if a.is_empty() || TAU - best.0 >= PI {
            (0.0, TAU)
        } else {
            (best.1.rem_euclid(TAU), TAU - best.0)
        }
    };
    let mut order: Vec<(f32, f32)> = Vec::with_capacity(wedges.len() * 2);
    for &(start, span, _) in &wedges {
        let s = (start - arc_start).rem_euclid(TAU);
        order.push((s, span));
        if s + span > TAU {
            order.push((s - TAU, span));
        }
    }
    order.sort_by(|x, y| x.0.total_cmp(&y.0));
    let tol = 1e-4;
    let mut reach = 0.0f32;
    for &(start, span) in &order {
        if reach >= arc_span - tol {
            break;
        }
        if start > reach + tol {
            return None;
        }
        reach = reach.max(start + span);
    }
    if reach < arc_span - tol {
        return None;
    }
    let mut worst = 0.0f32;
    for &(start, span, i) in &wedges {
        let dir = |a: f32| add(scale(u, a.cos()), scale(w, a.sin()));
        // Counter-clockwise of the start edge, clockwise of the end edge.
        let lo = cross(n, dir(start));
        let hi = scale(cross(n, dir(start + span)), -1.0);
        let part = clip(&clip(&t, v, lo), v, hi);
        let s = &fan[i];
        for p in part {
            worst = worst.max(point_tri_dist2(p, s[0], s[1], s[2]));
        }
    }
    Some(worst)
}

/// The part of the convex polygon `poly` where `dot(p - o, m) >= 0`.
fn clip(poly: &[V3], o: V3, m: V3) -> Vec<V3> {
    let mut out = Vec::with_capacity(poly.len() + 1);
    for (k, &p) in poly.iter().enumerate() {
        let q = poly[(k + 1) % poly.len()];
        let (dp, dq) = (dot(sub(p, o), m), dot(sub(q, o), m));
        if dp >= 0.0 {
            out.push(p);
        }
        if (dp >= 0.0) != (dq >= 0.0) {
            out.push(add(p, scale(sub(q, p), dp / (dp - dq))));
        }
    }
    out
}

/// Tighten the bound of `t` by midpoint subdivision until it is below `goal`
/// or the depth runs out; always an upper bound.
fn refine(grid: &TriGrid, t: [V3; 3], depth: u32, goal: f32) -> f32 {
    let (_, upper, _) = tri_bound(grid, t);
    if upper <= goal || depth == 0 {
        return upper;
    }
    let m = |a: V3, b: V3| scale(add(a, b), 0.5);
    let (ab, bc, ca) = (m(t[0], t[1]), m(t[1], t[2]), m(t[2], t[0]));
    [[t[0], ab, ca], [ab, t[1], bc], [ca, bc, t[2]], [ab, bc, ca]]
        .into_iter()
        .map(|c| refine(grid, c, depth - 1, goal))
        .fold(0.0f32, f32::max)
}

/// Two-sided distance bound: from the triangles `a` to the surface of
/// `b_grid` and from the triangles `b` to the surface of `a_grid`, with the
/// nearest triangles (indices into each grid) of every triangle's corners and
/// centroid.
#[allow(clippy::type_complexity)]
fn hausdorff(
    a: &[[V3; 3]],
    b_grid: &TriGrid,
    b: &[[V3; 3]],
    a_grid: &TriGrid,
) -> (f32, Vec<[u32; 4]>, Vec<[u32; 4]>) {
    use rayon::prelude::*;
    let fwd: Vec<(f32, f32, [u32; 4])> = a.par_iter().map(|&t| tri_bound(b_grid, t)).collect();
    let back: Vec<(f32, f32, [u32; 4])> = b.par_iter().map(|&t| tri_bound(a_grid, t)).collect();
    let lower = fwd.iter().chain(&back).fold(0.0f32, |m, x| m.max(x.0));
    // A bound tighter than a small fraction of the simplified triangles' own
    // size is never worth proving (see `REFINE_FLOOR`).
    let mean_edge = {
        let sum: f64 = b
            .iter()
            .map(|t| f64::from(dist(t[0], t[1]) + dist(t[1], t[2]) + dist(t[2], t[0])))
            .sum();
        (sum / (3 * b.len().max(1)) as f64) as f32
    };
    let noise = REFINE_FLOOR * mean_edge;
    // Branch and bound: refine the loosest triangles first. Whatever they
    // prove becomes the bound every later triangle only has to stay under, so
    // most never need refining. Triangles are refined in fixed batches, each
    // in parallel against the bound proven before it, so the result does not
    // depend on thread scheduling.
    let mut open: Vec<(f32, bool, u32)> = fwd
        .iter()
        .enumerate()
        .map(|(i, x)| (x.1, false, i as u32))
        .chain(back.iter().enumerate().map(|(i, x)| (x.1, true, i as u32)))
        .collect();
    open.sort_by(|x, y| y.0.total_cmp(&x.0).then(x.1.cmp(&y.1)).then(x.2.cmp(&y.2)));
    let mut bound = (lower * (1.0 + REFINE_SLACK)).max(noise);
    // Every triangle's bound counts: refined ones by what refinement proved,
    // the rest by their own (smaller) initial bound.
    let mut proven = lower;
    let mut done = 0;
    for batch in open.chunks(REFINE_BATCH) {
        let goal = bound;
        let todo: Vec<&(f32, bool, u32)> = batch.iter().take_while(|x| x.0 > goal).collect();
        if todo.is_empty() {
            break;
        }
        done += todo.len();
        let r = todo
            .par_iter()
            .map(|&&(_, is_back, i)| {
                let (t, grid) = if is_back {
                    (b[i as usize], a_grid)
                } else {
                    (a[i as usize], b_grid)
                };
                refine(grid, t, REFINE_DEPTH, goal)
            })
            .reduce(|| 0.0f32, f32::max);
        proven = proven.max(r);
        bound = bound.max(r);
    }
    if let Some(rest) = open.get(done) {
        proven = proven.max(rest.0);
    }
    (
        proven,
        fwd.iter().map(|x| x.2).collect(),
        back.iter().map(|x| x.2).collect(),
    )
}

/// The original surface a group stands for, as indices into the build's
/// original triangle list.
pub struct Region<'a> {
    /// Originals the group owns: every original triangle is owned by exactly
    /// one cluster of a level. These are measured against the result.
    pub owned: &'a [u32],
    /// Further originals near the group's surface (claimed by its input
    /// clusters' triangles, see `measure`, or owned by the clusters it
    /// borders): targets for the result's distance to the original surface,
    /// never measured themselves.
    pub claimed: &'a [u32],
}

/// Measure the simplified `pieces` (triangles per produced cluster) against
/// the original surface `region` the group's input stands for, and hand that
/// surface on to the pieces. `input` are the group's input triangles (they
/// give the uv scale); `neighbours` are the triangles of the level's other
/// clusters that touch the group's border.
///
/// An owned original triangle can straddle the group border; its part beyond
/// the border is rendered by the neighbour, which shares the border exactly
/// and accounts for its own changes, so originals are measured against the
/// result together with the neighbours' surface.
pub fn measure(
    original: &[[u32; 3]],
    region: Region,
    input: &[[u32; 3]],
    pieces: &[&[[u32; 3]]],
    neighbours: &[[V3; 3]],
    surf: &Surface,
) -> Measured {
    let pos = surf.pos;
    let mut s_ids = Vec::new();
    let mut owner = Vec::new();
    for (pi, p) in pieces.iter().enumerate() {
        for t in p.iter() {
            s_ids.push(*t);
            owner.push(pi as u32);
        }
    }
    let mut piece_refs: Vec<Vec<u32>> = vec![Vec::new(); pieces.len()];
    let mut piece_claimed: Vec<Vec<u32>> = vec![Vec::new(); pieces.len()];
    if s_ids.is_empty() || region.owned.is_empty() {
        return Measured {
            position: 0.0,
            shading: 0.0,
            piece_refs,
            piece_claimed,
        };
    }
    let t0 = std::time::Instant::now();
    let tri_of = |i: u32| original[i as usize].map(|v| pos[v as usize]);
    let s_tris: Vec<[V3; 3]> = s_ids.iter().map(|t| t.map(|v| pos[v as usize])).collect();
    let o_tris: Vec<[V3; 3]> = region.owned.iter().map(|&i| tri_of(i)).collect();
    let targets: Vec<u32> = region.owned.iter().chain(region.claimed).copied().collect();
    let s_grid = TriGrid::new(s_tris.clone());
    let sn_grid = TriGrid::new(s_tris.iter().chain(neighbours).copied().collect());
    let t_grid = TriGrid::new(targets.iter().map(|&i| tri_of(i)).collect());
    let t1 = t0.elapsed();
    let (position, o_near, s_near) = hausdorff(&o_tris, &sn_grid, &s_tris, &t_grid);
    let t2 = t0.elapsed();
    // Every owned original goes to the piece of the simplified triangle
    // nearest its centroid. Every simplified triangle also claims the
    // originals nearest its corners and centroid: a small simplified triangle
    // that is nearest to no original centroid must still have the surface it
    // replaced near it, or the next level would measure it against an
    // original surface that does not reach it.
    let s_count = s_tris.len() as u32;
    for ((&i, n), t) in region.owned.iter().zip(&o_near).zip(&o_tris) {
        let s = if n[3] < s_count {
            n[3]
        } else {
            s_grid
                .nearest(scale(add(add(t[0], t[1]), t[2]), 1.0 / 3.0))
                .1
        };
        piece_refs[owner[s as usize] as usize].push(i);
    }
    for (s, n) in s_near.iter().enumerate() {
        for &o in n {
            piece_claimed[owner[s] as usize].push(targets[o as usize]);
        }
    }
    for (owned, claimed) in piece_refs.iter_mut().zip(&mut piece_claimed) {
        owned.sort_unstable();
        claimed.sort_unstable();
        claimed.dedup();
        claimed.retain(|i| owned.binary_search(i).is_err());
    }

    // Shading at the corners and centroid of every original triangle.
    let mean_edge = {
        let sum: f64 = s_tris
            .iter()
            .map(|t| f64::from(dist(t[0], t[1]) + dist(t[1], t[2]) + dist(t[2], t[0])))
            .sum();
        (sum / (3 * s_tris.len()) as f64) as f32
    };
    let uv_len = uv_scale(input, surf);
    let shade = |a: Option<VertexAttr>| a.as_ref().and_then(Shading::of);
    let mut shading = 0.0f32;
    for &i in region.owned {
        let t = original[i as usize];
        let corners = t.map(|v| shade((surf.attr)(v)));
        let p = t.map(|v| pos[v as usize]);
        let mut samples: [(V3, Option<Shading>); 4] = [(p[0], None); 4];
        for k in 0..3 {
            samples[k] = (p[k], corners[k].filter(|_| !surf.seam[t[k] as usize]));
        }
        samples[3] = (
            scale(add(add(p[0], p[1]), p[2]), 1.0 / 3.0),
            match (&corners[0], &corners[1], &corners[2]) {
                (Some(a), Some(b), Some(c)) => Shading::blend([a, b, c], [1.0 / 3.0; 3]),
                _ => None,
            },
        );
        for (q, want) in samples {
            let Some(want) = want else { continue };
            let (_, s) = s_grid.nearest(q);
            let sv = s_ids[s as usize];
            let got = sv.map(|v| shade((surf.attr)(v)));
            let (Some(c0), Some(c1), Some(c2)) = (&got[0], &got[1], &got[2]) else {
                continue;
            };
            let st = &s_tris[s as usize];
            let (_, w) = closest_on_tri(q, st[0], st[1], st[2]);
            let Some(got) = Shading::blend([c0, c1, c2], w) else {
                continue;
            };
            let tilt = length(sub(want.normal, got.normal));
            let mut len = 0.5 * tilt * mean_edge;
            for (ch, &per_uv) in uv_len.iter().enumerate() {
                let du = [
                    want.uv[ch][0] - got.uv[ch][0],
                    want.uv[ch][1] - got.uv[ch][1],
                ];
                len = len.max((du[0] * du[0] + du[1] * du[1]).sqrt() * per_uv);
            }
            shading = shading.max(len);
        }
    }
    if std::env::var("VG_TIME").is_ok() {
        eprintln!(
            "M grids {t1:?} hausdorff {:?} total {:?} owned {} targets {} s {}",
            t2 - t1,
            t0.elapsed(),
            region.owned.len(),
            targets.len(),
            s_tris.len()
        );
    }
    Measured {
        position,
        shading,
        piece_refs,
        piece_claimed,
    }
}
