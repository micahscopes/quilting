use crate::{
    enclose_negative_spheres, ContainmentCertificate, NodeId, NodeSpec, PredicateConfig,
    QueryResult, RoundIndex, RoundIndexError, RoundQuery, RoundSide, RoundSideAutomorphism,
    TopologyKey, TransformError,
};
use quilting_core::{Quat, RoundSideOrientation, RoundWallGeometry};
use serde::{Deserialize, Serialize};

const SOURCE_BOUND_PADDING: f64 = 1.0e-6;
const DENOMINATOR_GUARD: f64 = 1.0e-12;

/// Three source controls for one rational quaternionic-Bezier patch.
/// Positions are ordinary `(x,y,z)` points; weights are quaternion `(w,x,y,z)`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PatchControl {
    pub face: u32,
    pub positions: [[f64; 3]; 3],
    pub weights: [[f64; 4]; 3],
}

/// A static/rest-pose patch hierarchy plus an explicit lane for patches whose
/// complete source image cannot be enclosed by a finite sphere.
#[derive(Debug, Clone)]
pub struct StaticPatchIndex {
    index: Option<RoundIndex<Option<u32>>>,
    always_candidates: Vec<u32>,
    report: PatchIndexBuildReport,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PatchIndexBuildReport {
    pub patches: usize,
    pub bounded_patches: usize,
    pub always_candidate_patches: usize,
    pub hierarchy_nodes: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PatchQueryResult {
    pub candidate_faces: Vec<u32>,
    pub visited_nodes: usize,
    pub pruned_nodes: usize,
    pub always_candidate_faces: usize,
}

impl StaticPatchIndex {
    pub fn build(
        topology: TopologyKey,
        patches: &[PatchControl],
        predicates: PredicateConfig,
    ) -> Result<Self, RoundIndexError> {
        let mut faces = std::collections::BTreeSet::new();
        for patch in patches {
            if !faces.insert(patch.face) {
                return Err(RoundIndexError::DuplicatePatchFace(patch.face));
            }
        }

        let mut leaves = Vec::with_capacity(patches.len());
        let mut always_candidates = Vec::new();
        for patch in patches {
            match conservative_patch_source_bound(patch) {
                Some(bound) => leaves.push(BuildNode::leaf(bound, patch.face)),
                None => always_candidates.push(patch.face),
            }
        }
        always_candidates.sort_unstable();
        always_candidates.dedup();

        let index = if leaves.is_empty() {
            None
        } else {
            let root = build_hierarchy(leaves)?;
            let mut specs = Vec::new();
            let mut next_id = 0;
            flatten_hierarchy(&root, None, &mut next_id, &mut specs);
            Some(RoundIndex::build_for(topology, specs, predicates)?)
        };
        let hierarchy_nodes = index.as_ref().map_or(0, |index| index.nodes().count());
        let report = PatchIndexBuildReport {
            patches: patches.len(),
            bounded_patches: patches.len() - always_candidates.len(),
            always_candidate_patches: always_candidates.len(),
            hierarchy_nodes,
        };
        Ok(Self {
            index,
            always_candidates,
            report,
        })
    }

    pub fn report(&self) -> PatchIndexBuildReport {
        self.report
    }

    pub fn query(&self, pulled_back_query: &RoundQuery) -> PatchQueryResult {
        self.finish_query(
            self.index
                .as_ref()
                .map(|index| (index, index.query(pulled_back_query))),
        )
    }

    pub fn query_output_chart<A: RoundSideAutomorphism>(
        &self,
        output_chart_query: &RoundQuery,
        source_to_output: &A,
    ) -> Result<PatchQueryResult, TransformError> {
        let indexed = self.index.as_ref().map(|index| {
            index
                .query_output_chart(output_chart_query, source_to_output)
                .map(|result| (index, result))
        });
        match indexed {
            Some(Ok(indexed)) => Ok(self.finish_query(Some(indexed))),
            Some(Err(error)) => Err(error),
            None => Ok(self.finish_query(None)),
        }
    }

    fn finish_query(
        &self,
        indexed: Option<(&RoundIndex<Option<u32>>, QueryResult)>,
    ) -> PatchQueryResult {
        let mut candidate_faces = self.always_candidates.clone();
        let (visited_nodes, pruned_nodes) = indexed.map_or((0, 0), |(index, result)| {
            candidate_faces.extend(
                index
                    .candidate_payloads(&result)
                    .filter_map(|(_, face)| *face),
            );
            (result.visited_nodes, result.pruned_nodes)
        });
        candidate_faces.sort_unstable();
        candidate_faces.dedup();
        PatchQueryResult {
            candidate_faces,
            visited_nodes,
            pruned_nodes,
            always_candidate_faces: self.always_candidates.len(),
        }
    }
}

/// Finite source-space carrier for a complete patch. `None` means the patch
/// must remain an unconditional candidate (invalid data or a possible rational
/// denominator zero); it never means the patch may be discarded.
pub fn conservative_patch_source_bound(patch: &PatchControl) -> Option<RoundSide> {
    if patch
        .positions
        .iter()
        .flatten()
        .chain(patch.weights.iter().flatten())
        .any(|value| !value.is_finite())
    {
        return None;
    }
    let common_weight = quaternion_norm(patch.weights[0]) > DENOMINATOR_GUARD
        && patch.weights[1] == patch.weights[0]
        && patch.weights[2] == patch.weights[0];
    let (center, radius) = if common_weight {
        let center = std::array::from_fn(|axis| {
            (patch.positions[0][axis] + patch.positions[1][axis] + patch.positions[2][axis]) / 3.0
        });
        let radius = patch
            .positions
            .iter()
            .map(|point| distance(*point, center))
            .fold(0.0_f64, f64::max);
        (center, radius)
    } else {
        let denominators = patch.weights;
        let denominator_distance = origin_to_quaternion_triangle(denominators);
        if !denominator_distance.is_finite() || denominator_distance <= DENOMINATOR_GUARD {
            return None;
        }
        let numerator_radius = patch
            .positions
            .iter()
            .zip(patch.weights)
            .map(|(&point, weight)| {
                let point = Quat::from_point(point[0], point[1], point[2]);
                let weight = Quat::new(weight[0], weight[1], weight[2], weight[3]);
                (point * weight).norm()
            })
            .fold(0.0_f64, f64::max);
        ([0.0; 3], numerator_radius / denominator_distance)
    };
    if !radius.is_finite() {
        return None;
    }
    RoundSide::sphere(
        center,
        radius * (1.0 + SOURCE_BOUND_PADDING) + SOURCE_BOUND_PADDING,
        RoundSideOrientation::Negative,
    )
    .ok()
}

#[derive(Debug)]
struct BuildNode {
    bound: RoundSide,
    face: Option<u32>,
    minimum_face: u32,
    children: Vec<BuildNode>,
}

impl BuildNode {
    fn leaf(bound: RoundSide, face: u32) -> Self {
        Self {
            bound,
            face: Some(face),
            minimum_face: face,
            children: Vec::new(),
        }
    }

    fn center(&self) -> [f64; 3] {
        match self.bound.geometry() {
            RoundWallGeometry::Sphere { center, .. } => center,
            RoundWallGeometry::Plane { .. } => unreachable!("patch bounds are finite spheres"),
        }
    }
}

fn build_hierarchy(mut nodes: Vec<BuildNode>) -> Result<BuildNode, RoundIndexError> {
    if nodes.len() == 1 {
        return Ok(nodes.pop().expect("one node"));
    }
    let axis = widest_center_axis(&nodes);
    nodes.sort_unstable_by(|left, right| {
        left.center()[axis]
            .total_cmp(&right.center()[axis])
            .then_with(|| left.minimum_face.cmp(&right.minimum_face))
    });
    let right = nodes.split_off(nodes.len() / 2);
    let left = build_hierarchy(nodes)?;
    let right = build_hierarchy(right)?;
    let bound = enclose_negative_spheres(
        &[(NodeId(0), left.bound), (NodeId(1), right.bound)],
        SOURCE_BOUND_PADDING,
    )?;
    let minimum_face = left.minimum_face.min(right.minimum_face);
    Ok(BuildNode {
        bound,
        face: None,
        minimum_face,
        children: vec![left, right],
    })
}

fn flatten_hierarchy(
    node: &BuildNode,
    parent: Option<NodeId>,
    next_id: &mut u64,
    specs: &mut Vec<NodeSpec<Option<u32>>>,
) {
    let id = NodeId(*next_id);
    *next_id += 1;
    specs.push(NodeSpec {
        id,
        parent,
        parent_containment: ContainmentCertificate::Computed,
        bound: node.bound,
        payload: node.face,
    });
    for child in &node.children {
        flatten_hierarchy(child, Some(id), next_id, specs);
    }
}

fn widest_center_axis(nodes: &[BuildNode]) -> usize {
    let mut minimum = [f64::INFINITY; 3];
    let mut maximum = [f64::NEG_INFINITY; 3];
    for node in nodes {
        for (axis, value) in node.center().into_iter().enumerate() {
            minimum[axis] = minimum[axis].min(value);
            maximum[axis] = maximum[axis].max(value);
        }
    }
    (0..3)
        .max_by(|&left, &right| {
            (maximum[left] - minimum[left]).total_cmp(&(maximum[right] - minimum[right]))
        })
        .unwrap_or(0)
}

fn origin_to_quaternion_triangle(values: [[f64; 4]; 3]) -> f64 {
    let [a, b, c] = values;
    let mut distance_squared = dot4(a, a).min(dot4(b, b)).min(dot4(c, c));
    for (start, end) in [(a, b), (a, c), (b, c)] {
        let edge = sub4(end, start);
        let length_squared = dot4(edge, edge);
        if length_squared > 1.0e-24 {
            let t = (-dot4(start, edge) / length_squared).clamp(0.0, 1.0);
            let closest = add4(start, scale4(edge, t));
            distance_squared = distance_squared.min(dot4(closest, closest));
        }
    }
    let ab = sub4(b, a);
    let ac = sub4(c, a);
    let g00 = dot4(ab, ab);
    let g01 = dot4(ab, ac);
    let g11 = dot4(ac, ac);
    let determinant = g00 * g11 - g01 * g01;
    if determinant > 1.0e-24 {
        let rhs0 = -dot4(a, ab);
        let rhs1 = -dot4(a, ac);
        let s = (rhs0 * g11 - g01 * rhs1) / determinant;
        let t = (g00 * rhs1 - g01 * rhs0) / determinant;
        if s >= 0.0 && t >= 0.0 && s + t <= 1.0 {
            let closest = add4(a, add4(scale4(ab, s), scale4(ac, t)));
            distance_squared = distance_squared.min(dot4(closest, closest));
        }
    }
    distance_squared.max(0.0).sqrt()
}

fn quaternion_norm(value: [f64; 4]) -> f64 {
    dot4(value, value).sqrt()
}

fn dot4(left: [f64; 4], right: [f64; 4]) -> f64 {
    left.into_iter().zip(right).map(|(a, b)| a * b).sum()
}

fn sub4(left: [f64; 4], right: [f64; 4]) -> [f64; 4] {
    std::array::from_fn(|axis| left[axis] - right[axis])
}

fn add4(left: [f64; 4], right: [f64; 4]) -> [f64; 4] {
    std::array::from_fn(|axis| left[axis] + right[axis])
}

fn scale4(value: [f64; 4], factor: f64) -> [f64; 4] {
    value.map(|coordinate| coordinate * factor)
}

fn distance(left: [f64; 3], right: [f64; 3]) -> f64 {
    left.into_iter()
        .zip(right)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use quilting_core::ConformalTransformChain;

    const IDENTITY_VIEW_PROJECTION: [f64; 16] = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];

    fn flat(face: u32, x: f64) -> PatchControl {
        PatchControl {
            face,
            positions: [[x - 0.1, -0.1, 0.0], [x + 0.1, -0.1, 0.0], [x, 0.1, 0.0]],
            weights: [[1.0, 0.0, 0.0, 0.0]; 3],
        }
    }

    #[test]
    fn flat_patch_bound_contains_its_triangle() {
        let patch = flat(7, 2.0);
        let bound = conservative_patch_source_bound(&patch).unwrap();
        for point in patch.positions {
            assert!(bound.contains(point).unwrap());
        }
    }

    #[test]
    fn rational_denominator_zero_enters_always_candidate_lane() {
        let patch = PatchControl {
            weights: [
                [1.0, 0.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            ..flat(9, 10.0)
        };
        assert!(conservative_patch_source_bound(&patch).is_none());
        let index = StaticPatchIndex::build(
            TopologyKey::default(),
            &[flat(1, 0.0), patch],
            PredicateConfig::default(),
        )
        .unwrap();
        let query = RoundQuery::from_view_projection(&IDENTITY_VIEW_PROJECTION).unwrap();
        assert_eq!(index.query(&query).candidate_faces, vec![1, 9]);
    }

    #[test]
    fn hierarchy_prunes_separated_patch_subtrees() {
        let patches = (-64..64)
            .enumerate()
            .map(|(face, x)| flat(face as u32, f64::from(x)))
            .collect::<Vec<_>>();
        let index =
            StaticPatchIndex::build(TopologyKey::default(), &patches, PredicateConfig::default())
                .unwrap();
        let query = RoundQuery::from_view_projection(&IDENTITY_VIEW_PROJECTION).unwrap();
        let result = index
            .query_output_chart(&query, &ConformalTransformChain::identity())
            .unwrap();
        assert_eq!(result.candidate_faces, vec![63, 64, 65]);
        assert!(result.visited_nodes < index.report().hierarchy_nodes / 2);
    }

    #[test]
    fn duplicate_face_ids_are_rejected() {
        assert!(matches!(
            StaticPatchIndex::build(
                TopologyKey::default(),
                &[flat(7, 0.0), flat(7, 1.0)],
                PredicateConfig::default(),
            ),
            Err(RoundIndexError::DuplicatePatchFace(7))
        ));
    }
}
