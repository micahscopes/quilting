//! Deterministic exact nearest-triangle queries for offline remeshing work.
//!
//! This is deliberately Euclidean and backend-neutral. It accelerates source
//! correspondence in normalized chart coordinates; it is not the conformal
//! round index used for runtime visibility.

use crate::geometry;

const LEAF_TRIANGLES: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct IndexedTriangle {
    pub stable_index: usize,
    pub positions: [[f64; 3]; 3],
    pub orientation: [f64; 3],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct NearestTriangle {
    pub stable_index: usize,
    pub barycentric: [f64; 3],
    pub squared_distance: f64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct SearchCounters {
    pub candidate_tests: usize,
    pub node_visits: usize,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct SearchScratch {
    stack: Vec<(usize, f64)>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SearchError {
    CountOverflow,
    CandidateBudgetExceeded { attempted: usize, maximum: usize },
}

#[derive(Clone, Copy, Debug)]
struct Bounds {
    minimum: [f64; 3],
    maximum: [f64; 3],
}

impl Bounds {
    fn empty() -> Self {
        Self {
            minimum: [f64::INFINITY; 3],
            maximum: [f64::NEG_INFINITY; 3],
        }
    }

    fn include_point(&mut self, point: [f64; 3]) {
        for axis in 0..3 {
            self.minimum[axis] = self.minimum[axis].min(point[axis]);
            self.maximum[axis] = self.maximum[axis].max(point[axis]);
        }
    }

    fn include_triangle(&mut self, triangle: &IndexedTriangle) {
        for point in triangle.positions {
            self.include_point(point);
        }
    }

    fn squared_distance(self, point: [f64; 3]) -> f64 {
        let mut result = 0.0;
        for axis in 0..3 {
            let delta = if point[axis] < self.minimum[axis] {
                self.minimum[axis] - point[axis]
            } else if point[axis] > self.maximum[axis] {
                point[axis] - self.maximum[axis]
            } else {
                0.0
            };
            result += delta * delta;
        }
        result
    }
}

#[derive(Clone, Copy, Debug)]
enum NodeKind {
    Leaf { start: usize, end: usize },
    Branch { left: usize, right: usize },
}

#[derive(Clone, Copy, Debug)]
struct Node {
    bounds: Bounds,
    kind: NodeKind,
}

#[derive(Clone, Debug)]
pub(crate) struct TriangleBvh {
    triangles: Vec<IndexedTriangle>,
    nodes: Vec<Node>,
    root: Option<usize>,
}

impl TriangleBvh {
    pub fn new(mut triangles: Vec<IndexedTriangle>) -> Self {
        let mut nodes = Vec::new();
        let root = (!triangles.is_empty()).then(|| {
            let length = triangles.len();
            build_node(&mut triangles, &mut nodes, 0, length)
        });
        Self {
            triangles,
            nodes,
            root,
        }
    }

    pub fn nearest_orientation_compatible(
        &self,
        point: [f64; 3],
        source_orientation: [f64; 3],
        scratch: &mut SearchScratch,
        counters: &mut SearchCounters,
        maximum_candidate_tests: usize,
    ) -> Result<Option<NearestTriangle>, SearchError> {
        let Some(root) = self.root else {
            return Ok(None);
        };
        let mut best = None;
        scratch.stack.clear();
        scratch
            .stack
            .push((root, self.nodes[root].bounds.squared_distance(point)));
        while let Some((node_index, lower_bound)) = scratch.stack.pop() {
            counters.node_visits = counters
                .node_visits
                .checked_add(1)
                .ok_or(SearchError::CountOverflow)?;
            if best.as_ref().is_some_and(|hit: &NearestTriangle| {
                definitely_farther(lower_bound, hit.squared_distance)
            }) {
                continue;
            }
            match self.nodes[node_index].kind {
                NodeKind::Leaf { start, end } => {
                    for triangle in &self.triangles[start..end] {
                        let attempted = counters
                            .candidate_tests
                            .checked_add(1)
                            .ok_or(SearchError::CountOverflow)?;
                        if attempted > maximum_candidate_tests {
                            return Err(SearchError::CandidateBudgetExceeded {
                                attempted,
                                maximum: maximum_candidate_tests,
                            });
                        }
                        counters.candidate_tests = attempted;
                        if geometry::vec3_dot(source_orientation, triangle.orientation) <= 0.0 {
                            continue;
                        }
                        let barycentric = closest_barycentric(point, triangle.positions);
                        let closest = barycentric_point(triangle.positions, barycentric);
                        let delta = geometry::vec3_sub(point, closest);
                        let squared_distance = geometry::vec3_dot(delta, delta);
                        let replace = best.as_ref().is_none_or(|hit: &NearestTriangle| {
                            squared_distance < hit.squared_distance
                                || (squared_distance == hit.squared_distance
                                    && triangle.stable_index < hit.stable_index)
                        });
                        if replace {
                            best = Some(NearestTriangle {
                                stable_index: triangle.stable_index,
                                barycentric,
                                squared_distance,
                            });
                        }
                    }
                }
                NodeKind::Branch { left, right } => {
                    let left_distance = self.nodes[left].bounds.squared_distance(point);
                    let right_distance = self.nodes[right].bounds.squared_distance(point);
                    let (near, far) = if left_distance < right_distance
                        || (left_distance == right_distance && left < right)
                    {
                        ((left, left_distance), (right, right_distance))
                    } else {
                        ((right, right_distance), (left, left_distance))
                    };
                    if best
                        .as_ref()
                        .is_none_or(|hit| !definitely_farther(far.1, hit.squared_distance))
                    {
                        scratch.stack.push(far);
                    }
                    if best
                        .as_ref()
                        .is_none_or(|hit| !definitely_farther(near.1, hit.squared_distance))
                    {
                        scratch.stack.push(near);
                    }
                }
            }
        }
        Ok(best)
    }
}

fn definitely_farther(lower_bound: f64, best: f64) -> bool {
    // AABB distance is a mathematical lower bound, but both it and the exact
    // triangle distance are rounded independently. Keep near-ULP cases alive
    // so pruning cannot turn a floating-point tie into an approximate query.
    let scale = lower_bound.abs().max(best.abs()).max(f64::MIN_POSITIVE);
    lower_bound - best > scale * (32.0 * f64::EPSILON) + f64::MIN_POSITIVE
}

fn build_node(
    triangles: &mut [IndexedTriangle],
    nodes: &mut Vec<Node>,
    start: usize,
    end: usize,
) -> usize {
    let mut bounds = Bounds::empty();
    let mut centroid_bounds = Bounds::empty();
    for triangle in &triangles[start..end] {
        bounds.include_triangle(triangle);
        centroid_bounds.include_point(centroid(triangle.positions));
    }
    let index = nodes.len();
    nodes.push(Node {
        bounds,
        kind: NodeKind::Leaf { start, end },
    });
    if end - start <= LEAF_TRIANGLES {
        return index;
    }
    let axis = longest_axis(centroid_bounds);
    triangles[start..end].sort_by(|left, right| {
        centroid(left.positions)[axis]
            .total_cmp(&centroid(right.positions)[axis])
            .then_with(|| left.stable_index.cmp(&right.stable_index))
    });
    let middle = start + (end - start) / 2;
    let left = build_node(triangles, nodes, start, middle);
    let right = build_node(triangles, nodes, middle, end);
    nodes[index].kind = NodeKind::Branch { left, right };
    index
}

fn centroid(triangle: [[f64; 3]; 3]) -> [f64; 3] {
    [
        (triangle[0][0] + triangle[1][0] + triangle[2][0]) / 3.0,
        (triangle[0][1] + triangle[1][1] + triangle[2][1]) / 3.0,
        (triangle[0][2] + triangle[1][2] + triangle[2][2]) / 3.0,
    ]
}

fn longest_axis(bounds: Bounds) -> usize {
    let extents = [
        bounds.maximum[0] - bounds.minimum[0],
        bounds.maximum[1] - bounds.minimum[1],
        bounds.maximum[2] - bounds.minimum[2],
    ];
    if extents[1] > extents[0] && extents[1] >= extents[2] {
        1
    } else if extents[2] > extents[0] && extents[2] > extents[1] {
        2
    } else {
        0
    }
}

fn barycentric_point(triangle: [[f64; 3]; 3], barycentric: [f64; 3]) -> [f64; 3] {
    [
        triangle[0][0] * barycentric[0]
            + triangle[1][0] * barycentric[1]
            + triangle[2][0] * barycentric[2],
        triangle[0][1] * barycentric[0]
            + triangle[1][1] * barycentric[1]
            + triangle[2][1] * barycentric[2],
        triangle[0][2] * barycentric[0]
            + triangle[1][2] * barycentric[1]
            + triangle[2][2] * barycentric[2],
    ]
}

fn closest_barycentric(point: [f64; 3], triangle: [[f64; 3]; 3]) -> [f64; 3] {
    let a = triangle[0];
    let b = triangle[1];
    let c = triangle[2];
    let ab = geometry::vec3_sub(b, a);
    let ac = geometry::vec3_sub(c, a);
    let ap = geometry::vec3_sub(point, a);
    let d1 = geometry::vec3_dot(ab, ap);
    let d2 = geometry::vec3_dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return [1.0, 0.0, 0.0];
    }
    let bp = geometry::vec3_sub(point, b);
    let d3 = geometry::vec3_dot(ab, bp);
    let d4 = geometry::vec3_dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return [0.0, 1.0, 0.0];
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let parameter = d1 / (d1 - d3);
        return [1.0 - parameter, parameter, 0.0];
    }
    let cp = geometry::vec3_sub(point, c);
    let d5 = geometry::vec3_dot(ab, cp);
    let d6 = geometry::vec3_dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return [0.0, 0.0, 1.0];
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let parameter = d2 / (d2 - d6);
        return [1.0 - parameter, 0.0, parameter];
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        let parameter = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return [0.0, 1.0 - parameter, parameter];
    }
    let inverse = 1.0 / (va + vb + vc);
    let second = vb * inverse;
    let third = vc * inverse;
    [1.0 - second - third, second, third]
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    fn triangle(stable_index: usize, x: f64) -> IndexedTriangle {
        IndexedTriangle {
            stable_index,
            positions: [[x, 0.0, 0.0], [x + 0.75, 0.0, 0.0], [x, 0.75, 0.0]],
            orientation: [0.0, 0.0, 1.0],
        }
    }

    fn brute_force(
        triangles: &[IndexedTriangle],
        point: [f64; 3],
        orientation: [f64; 3],
    ) -> Option<NearestTriangle> {
        let mut best = None;
        for triangle in triangles {
            if geometry::vec3_dot(orientation, triangle.orientation) <= 0.0 {
                continue;
            }
            let barycentric = closest_barycentric(point, triangle.positions);
            let closest = barycentric_point(triangle.positions, barycentric);
            let delta = geometry::vec3_sub(point, closest);
            let squared_distance = geometry::vec3_dot(delta, delta);
            if best.as_ref().is_none_or(|hit: &NearestTriangle| {
                squared_distance < hit.squared_distance
                    || (squared_distance == hit.squared_distance
                        && triangle.stable_index < hit.stable_index)
            }) {
                best = Some(NearestTriangle {
                    stable_index: triangle.stable_index,
                    barycentric,
                    squared_distance,
                });
            }
        }
        best
    }

    #[test]
    fn indexed_search_matches_brute_force_exactly() {
        let mut triangles = (0..64)
            .map(|index| triangle(1000 - index, index as f64 * 1.25))
            .collect::<Vec<_>>();
        triangles.push(IndexedTriangle {
            stable_index: 2000,
            positions: [[3.0, 0.0, 0.0], [3.0, 0.75, 0.0], [3.75, 0.0, 0.0]],
            orientation: [0.0, 0.0, -1.0],
        });
        let bvh = TriangleBvh::new(triangles.clone());
        let mut scratch = SearchScratch::default();
        for point in [
            [-2.0, 0.2, 0.1],
            [0.2, 0.2, 0.0],
            [3.1, 0.1, 2.0],
            [37.7, -4.0, -0.5],
            [100.0, 100.0, 100.0],
        ] {
            for orientation in [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]] {
                let expected = brute_force(&triangles, point, orientation);
                let mut counters = SearchCounters::default();
                let actual = bvh
                    .nearest_orientation_compatible(
                        point,
                        orientation,
                        &mut scratch,
                        &mut counters,
                        usize::MAX,
                    )
                    .unwrap();
                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    fn indexed_search_matches_brute_force_on_seeded_3d_cases() {
        let mut rng = rand_pcg::Pcg64::seed_from_u64(0x5142_4256_485f_4558);
        let mut triangles = Vec::new();
        while triangles.len() < 257 {
            let origin = [
                rng.gen_range(-10.0..10.0),
                rng.gen_range(-10.0..10.0),
                rng.gen_range(-10.0..10.0),
            ];
            let first = [
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
            ];
            let second = [
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
            ];
            let orientation = geometry::vec3_cross(first, second);
            if geometry::vec3_len(orientation) < 0.05 {
                continue;
            }
            triangles.push(IndexedTriangle {
                stable_index: 10_000 - triangles.len() * 17,
                positions: [
                    origin,
                    geometry::vec3_add(origin, first),
                    geometry::vec3_add(origin, second),
                ],
                orientation,
            });
        }
        let bvh = TriangleBvh::new(triangles.clone());
        let mut scratch = SearchScratch::default();
        for _ in 0..256 {
            let point = [
                rng.gen_range(-12.0..12.0),
                rng.gen_range(-12.0..12.0),
                rng.gen_range(-12.0..12.0),
            ];
            let orientation = [
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
            ];
            let expected = brute_force(&triangles, point, orientation);
            let mut counters = SearchCounters::default();
            let actual = bvh
                .nearest_orientation_compatible(
                    point,
                    orientation,
                    &mut scratch,
                    &mut counters,
                    usize::MAX,
                )
                .unwrap();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn equal_distance_ties_use_stable_index() {
        let left = triangle(9, 0.0);
        let mut right = left;
        right.stable_index = 3;
        let bvh = TriangleBvh::new(vec![left, right]);
        let mut scratch = SearchScratch::default();
        let mut counters = SearchCounters::default();
        let hit = bvh
            .nearest_orientation_compatible(
                [0.2, 0.2, 0.0],
                [0.0, 0.0, 1.0],
                &mut scratch,
                &mut counters,
                usize::MAX,
            )
            .unwrap()
            .unwrap();
        assert_eq!(hit.stable_index, 3);
    }

    #[test]
    fn candidate_budget_is_a_hard_runtime_guard() {
        let bvh = TriangleBvh::new((0..32).map(|index| triangle(index, index as f64)).collect());
        let mut scratch = SearchScratch::default();
        let mut counters = SearchCounters::default();
        let error = bvh
            .nearest_orientation_compatible(
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                &mut scratch,
                &mut counters,
                0,
            )
            .unwrap_err();
        assert_eq!(
            error,
            SearchError::CandidateBudgetExceeded {
                attempted: 1,
                maximum: 0,
            }
        );
    }
}
