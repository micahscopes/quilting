//! Conservative round-side spatial queries under conformal generator words.
//!
//! This crate is the executable counterpart of
//! `formal/ConformalMereology/RoundSideIndex.lean`. The Lean development says
//! carrier containment, disjointness, and pulled-back queries are preserved
//! by a certified `RoundSideAutomorphism`. This crate supplies a finite
//! runtime representation and a deliberately partial disjointness predicate.
//!
//! [`Separation::Disjoint`] is the only result that permits pruning. Every
//! borderline or unsupported case returns [`Separation::IntersectsOrUnknown`].
//! A [`RoundIndex`] additionally assumes every payload's actual geometry lies
//! in its leaf carrier; that application-level certificate is outside this
//! crate.
//!
//! Topology and [`NodeId`] values are immutable after construction. Animation
//! changes only the bounds layer through [`RoundIndex::refit`]. Cache animated
//! bounds by `(TopologyKey, PoseKey, NodeId)`: asset/topology revisions own
//! cluster membership, while a clip/sample key owns one posed snapshot. The
//! caller owns pose evaluation and QB bulge bounds. When runtime containment
//! cannot prove such a bound, [`ContainmentCertificate::Trusted`] is the
//! explicit external-proof boundary; a false trusted certificate can cause
//! false-negative queries.
//!
//! The index is metric-agnostic. Persistent patch addresses and animated
//! carriers live in posed source coordinates, while walking, physics, frusta,
//! and proximity distances live in the ordinary Euclidean output chart. Use
//! [`RoundIndex::query_output_chart`] to pull that output query back; do not
//! interpret source-coordinate distances as physical distances after a
//! non-affine conformal map.

use quilting_core::mereology::MereologyError;
use quilting_core::{
    ConformalGenerator, ConformalTransformChain, Quat, RoundSideOrientation, RoundWallGeometry,
};
use serde::de::Error as DeError;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

mod patch;

pub use patch::{
    conservative_patch_source_bound, PatchControl, PatchIndexBuildReport, PatchQueryResult,
    StaticPatchIndex,
};

const DEFAULT_CLEARANCE: f64 = 1.0e-12;

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(v: [f64; 3]) -> f64 {
    v[0].hypot(v[1]).hypot(v[2])
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(v: [f64; 3], factor: f64) -> [f64; 3] {
    [v[0] * factor, v[1] * factor, v[2] * factor]
}

fn finite3(v: [f64; 3]) -> bool {
    v.into_iter().all(f64::is_finite)
}

fn numerical_guard(clearance: f64, values: impl IntoIterator<Item = f64>) -> f64 {
    let scale = values
        .into_iter()
        .fold(1.0_f64, |largest, value| largest.max(value.abs()));
    clearance + 128.0 * f64::EPSILON * scale
}

/// An oriented open side with value-local geometry.
///
/// Unlike `quilting_core::OpenRoundSide`, this does not refer through a
/// `WallId`/`RoundWallSet`; query-local frustum and proximity sides can be
/// constructed and transformed directly.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct RoundSide {
    geometry: RoundWallGeometry,
    orientation: RoundSideOrientation,
}

impl RoundSide {
    pub fn new(
        mut geometry: RoundWallGeometry,
        mut orientation: RoundSideOrientation,
    ) -> Result<Self, RoundIndexError> {
        geometry.validate()?;
        if let RoundWallGeometry::Plane {
            unit_normal,
            offset,
        } = geometry
        {
            // A plane has a sign gauge: `(n, o, Negative)` and
            // `(-n, -o, Positive)` are the same carrier. Fix the first nonzero
            // normal component positive so transforms and cache snapshots have
            // one stable value representation.
            let first_nonzero = unit_normal.into_iter().find(|component| *component != 0.0);
            if first_nonzero.is_some_and(|component| component < 0.0) {
                geometry = RoundWallGeometry::Plane {
                    unit_normal: scale(unit_normal, -1.0),
                    offset: -offset,
                };
                orientation = orientation.complement();
            }
        }
        Ok(Self {
            geometry,
            orientation,
        })
    }

    pub fn sphere(
        center: [f64; 3],
        radius: f64,
        orientation: RoundSideOrientation,
    ) -> Result<Self, RoundIndexError> {
        Self::new(RoundWallGeometry::sphere(center, radius)?, orientation)
    }

    pub fn plane(
        normal: [f64; 3],
        offset: f64,
        orientation: RoundSideOrientation,
    ) -> Result<Self, RoundIndexError> {
        Self::new(RoundWallGeometry::plane(normal, offset)?, orientation)
    }

    pub fn validate(&self) -> Result<(), RoundIndexError> {
        self.geometry.validate()?;
        Ok(())
    }

    pub fn geometry(&self) -> RoundWallGeometry {
        self.geometry
    }

    pub fn orientation(&self) -> RoundSideOrientation {
        self.orientation
    }

    pub fn complement(self) -> Self {
        Self {
            orientation: self.orientation.complement(),
            ..self
        }
    }

    /// Strict membership in this ordinary Euclidean chart.
    pub fn contains(&self, point: [f64; 3]) -> Result<bool, RoundIndexError> {
        let value = self.geometry.signed_value(point)?;
        Ok(match self.orientation {
            RoundSideOrientation::Negative => value < 0.0,
            RoundSideOrientation::Positive => value > 0.0,
        })
    }

    fn positive_halfspace(&self) -> Option<([f64; 3], f64)> {
        let RoundWallGeometry::Plane {
            unit_normal,
            offset,
        } = self.geometry
        else {
            return None;
        };
        let sign = match self.orientation {
            RoundSideOrientation::Negative => -1.0,
            RoundSideOrientation::Positive => 1.0,
        };
        // The carrier is `dot(normal, x) + constant > 0`.
        Some((scale(unit_normal, sign), -offset * sign))
    }
}

impl<'de> Deserialize<'de> for RoundSide {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct RawRoundSide {
            geometry: RoundWallGeometry,
            orientation: RoundSideOrientation,
        }

        let raw = RawRoundSide::deserialize(deserializer)?;
        Self::new(raw.geometry, raw.orientation).map_err(D::Error::custom)
    }
}

/// Finite intersection of oriented open round sides.
///
/// An empty side list denotes the whole compactified space. A conventional
/// frustum has six sides; a proximity ball has one negative sphere side.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RoundQuery {
    sides: Vec<RoundSide>,
}

impl RoundQuery {
    pub fn new(sides: Vec<RoundSide>) -> Result<Self, RoundIndexError> {
        for side in &sides {
            side.validate()?;
        }
        Ok(Self { sides })
    }

    pub fn whole_space() -> Self {
        Self::default()
    }

    pub fn sides(&self) -> &[RoundSide] {
        &self.sides
    }

    pub fn contains(&self, point: [f64; 3]) -> Result<bool, RoundIndexError> {
        self.sides
            .iter()
            .try_fold(true, |inside, side| Ok(inside && side.contains(point)?))
    }

    /// Construct the six open half-spaces of a WebGL-style clip frustum.
    ///
    /// `view_projection` is column-major and maps ordinary output-chart points
    /// to the clip inequalities `-w < x,y,z < w`. The returned query therefore
    /// lives after the conformal map and is suitable for
    /// [`RoundIndex::query_output_chart`]. Degenerate/non-finite matrices are
    /// rejected rather than producing a side that might prune geometry.
    pub fn from_view_projection(view_projection: &[f64; 16]) -> Result<Self, RoundIndexError> {
        if view_projection.iter().any(|value| !value.is_finite()) {
            return Err(RoundIndexError::InvalidViewProjection);
        }
        let row = |index: usize| {
            [
                view_projection[index],
                view_projection[4 + index],
                view_projection[8 + index],
                view_projection[12 + index],
            ]
        };
        let r0 = row(0);
        let r1 = row(1);
        let r2 = row(2);
        let r3 = row(3);
        let mut sides = Vec::with_capacity(6);
        for (axis, sign) in [
            (r0, 1.0),
            (r0, -1.0),
            (r1, 1.0),
            (r1, -1.0),
            (r2, 1.0),
            (r2, -1.0),
        ] {
            let coefficients = [
                r3[0] + sign * axis[0],
                r3[1] + sign * axis[1],
                r3[2] + sign * axis[2],
            ];
            let normal_length = norm(coefficients);
            if !normal_length.is_finite() || normal_length == 0.0 {
                return Err(RoundIndexError::InvalidViewProjection);
            }
            sides.push(RoundSide::plane(
                scale(coefficients, normal_length.recip()),
                -(r3[3] + sign * axis[3]) / normal_length,
                RoundSideOrientation::Positive,
            )?);
        }
        Self::new(sides)
    }

    /// Pull a destination-space query into an automorphism's source chart.
    pub fn pullback<A: RoundSideAutomorphism>(
        &self,
        automorphism: &A,
    ) -> Result<Self, TransformError> {
        self.sides
            .iter()
            .map(|side| automorphism.pull_side(side))
            .collect::<Result<Vec<_>, _>>()
            .map(|sides| Self { sides })
    }

    pub fn pushforward<A: RoundSideAutomorphism>(
        &self,
        automorphism: &A,
    ) -> Result<Self, TransformError> {
        self.sides
            .iter()
            .map(|side| automorphism.push_side(side))
            .collect::<Result<Vec<_>, _>>()
            .map(|sides| Self { sides })
    }
}

impl From<RoundSide> for RoundQuery {
    fn from(side: RoundSide) -> Self {
        Self { sides: vec![side] }
    }
}

/// The only conservative result that permits pruning is `Disjoint`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Separation {
    Disjoint,
    IntersectsOrUnknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Containment {
    Contained,
    NotContained,
    Unknown,
}

/// Numerical policy for conservative analytic predicates.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct PredicateConfig {
    /// Absolute chart-space margin. Tangent/nearly tangent cases become
    /// unknown. A scale-relative floating-point guard is added internally;
    /// applications may increase this field for fitting/quantization error.
    clearance: f64,
}

impl Default for PredicateConfig {
    fn default() -> Self {
        Self {
            clearance: DEFAULT_CLEARANCE,
        }
    }
}

impl PredicateConfig {
    pub fn new(clearance: f64) -> Result<Self, RoundIndexError> {
        if !clearance.is_finite() || clearance < 0.0 {
            return Err(RoundIndexError::InvalidClearance);
        }
        Ok(Self { clearance })
    }

    pub fn clearance(self) -> f64 {
        self.clearance
    }
}

impl<'de> Deserialize<'de> for PredicateConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct RawPredicateConfig {
            clearance: f64,
        }

        let raw = RawPredicateConfig::deserialize(deserializer)?;
        Self::new(raw.clearance).map_err(D::Error::custom)
    }
}

/// Classify whether two open round-side carriers are certainly disjoint.
///
/// This implements clearance-separated sphere/sphere, sphere/plane, and
/// exactly parallel plane/plane cases for both orientations. A tolerance-based
/// parallel decision is deliberately not used: nearly parallel halfspaces
/// eventually intersect.
pub fn classify_separation(
    first: &RoundSide,
    second: &RoundSide,
    config: PredicateConfig,
) -> Separation {
    use RoundSideOrientation::{Negative, Positive};
    use RoundWallGeometry::{Plane, Sphere};

    match (first.geometry, second.geometry) {
        (
            Sphere {
                center: a,
                radius: r,
            },
            Sphere {
                center: b,
                radius: s,
            },
        ) => {
            let d = norm(sub(a, b));
            let e = numerical_guard(config.clearance, a.into_iter().chain(b).chain([r, s, d]));
            let disjoint = match (first.orientation, second.orientation) {
                (Negative, Negative) => d > r + s + e,
                (Negative, Positive) => d + r < s - e,
                (Positive, Negative) => d + s < r - e,
                (Positive, Positive) => false,
            };
            if disjoint {
                Separation::Disjoint
            } else {
                Separation::IntersectsOrUnknown
            }
        }
        (Sphere { center, radius }, Plane { .. }) => {
            sphere_plane_separation(center, radius, first.orientation, second, config.clearance)
        }
        (Plane { .. }, Sphere { center, radius }) => {
            sphere_plane_separation(center, radius, second.orientation, first, config.clearance)
        }
        (Plane { .. }, Plane { .. }) => plane_plane_separation(first, second, config.clearance),
    }
}

fn sphere_plane_separation(
    center: [f64; 3],
    radius: f64,
    sphere_orientation: RoundSideOrientation,
    plane: &RoundSide,
    clearance: f64,
) -> Separation {
    if sphere_orientation == RoundSideOrientation::Positive {
        return Separation::IntersectsOrUnknown;
    }
    let Some((normal, constant)) = plane.positive_halfspace() else {
        return Separation::IntersectsOrUnknown;
    };
    let maximum_over_ball = dot(normal, center) + constant + radius;
    let guard = numerical_guard(
        clearance,
        center
            .into_iter()
            .chain(normal)
            .chain([constant, radius, maximum_over_ball]),
    );
    if maximum_over_ball < -guard {
        Separation::Disjoint
    } else {
        Separation::IntersectsOrUnknown
    }
}

fn plane_plane_separation(first: &RoundSide, second: &RoundSide, clearance: f64) -> Separation {
    let (a, b) = first.positive_halfspace().expect("plane side");
    let (c, d) = second.positive_halfspace().expect("plane side");
    let guard = numerical_guard(clearance, a.into_iter().chain(c).chain([b, d]));
    if c == scale(a, -1.0) && -b > d + guard {
        Separation::Disjoint
    } else {
        Separation::IntersectsOrUnknown
    }
}

/// Classify whether `child` is certainly contained in `parent`.
pub fn classify_containment(
    child: &RoundSide,
    parent: &RoundSide,
    config: PredicateConfig,
) -> Containment {
    use RoundSideOrientation::{Negative, Positive};
    use RoundWallGeometry::{Plane, Sphere};

    if child == parent {
        return Containment::Contained;
    }
    match (child.geometry, parent.geometry) {
        (
            Sphere {
                center: a,
                radius: r,
            },
            Sphere {
                center: b,
                radius: s,
            },
        ) => {
            let d = norm(sub(a, b));
            let e = numerical_guard(config.clearance, a.into_iter().chain(b).chain([r, s, d]));
            match (child.orientation, parent.orientation) {
                (Negative, Negative) if d + r < s - e => Containment::Contained,
                (Negative, Negative) if d + r > s + e => Containment::NotContained,
                (Negative, Positive) if d > r + s + e => Containment::Contained,
                (Positive, Positive) if d + s < r - e => Containment::Contained,
                (Positive, Negative) => Containment::NotContained,
                _ => Containment::Unknown,
            }
        }
        (Sphere { center, radius }, Plane { .. }) => {
            if child.orientation == Positive {
                return Containment::NotContained;
            }
            let (normal, constant) = parent.positive_halfspace().expect("plane side");
            let minimum = dot(normal, center) + constant - radius;
            let e = numerical_guard(
                config.clearance,
                center
                    .into_iter()
                    .chain(normal)
                    .chain([constant, radius, minimum]),
            );
            if minimum > e {
                Containment::Contained
            } else if minimum < -e {
                Containment::NotContained
            } else {
                Containment::Unknown
            }
        }
        (Plane { .. }, Sphere { .. }) => match parent.orientation {
            Negative => Containment::NotContained,
            Positive => {
                let sphere_interior = RoundSide {
                    geometry: parent.geometry,
                    orientation: Negative,
                };
                match classify_separation(child, &sphere_interior, config) {
                    Separation::Disjoint => Containment::Contained,
                    Separation::IntersectsOrUnknown => Containment::Unknown,
                }
            }
        },
        (Plane { .. }, Plane { .. }) => {
            plane_halfspace_containment(child, parent, config.clearance)
        }
    }
}

fn plane_halfspace_containment(
    child: &RoundSide,
    parent: &RoundSide,
    clearance: f64,
) -> Containment {
    let (a, b) = child.positive_halfspace().expect("plane side");
    let (c, d) = parent.positive_halfspace().expect("plane side");
    if c != a {
        return Containment::Unknown;
    }
    // Child: a.x > -b. Parent: a.x > -d.
    let guard = numerical_guard(clearance, a.into_iter().chain(c).chain([b, d]));
    if -b > -d + guard {
        Containment::Contained
    } else if -b < -d - guard {
        Containment::NotContained
    } else {
        Containment::Unknown
    }
}

/// Stable application-chosen hierarchy node identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct NodeId(pub u64);

/// Revisions that own immutable cluster membership and hierarchy topology.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TopologyKey {
    pub asset_revision: u64,
    pub topology_revision: u64,
}

/// Application-defined identity for one animated source-space bound snapshot.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PoseKey {
    pub clip_revision: u64,
    pub sample: u64,
}

/// How a non-root node's carrier containment is certified.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContainmentCertificate {
    /// Require [`classify_containment`] to prove the relation.
    Computed,
    /// An external builder or animated QB bounder promises exact containment.
    /// Incorrect use can make traversal return false negatives.
    Trusted,
}

/// Input record for building an immutable topology and its initial bounds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeSpec<P> {
    pub id: NodeId,
    pub parent: Option<NodeId>,
    pub parent_containment: ContainmentCertificate,
    pub bound: RoundSide,
    /// Stable leaf or cluster identity. Traversal returns payloads only for
    /// leaves, but internal cluster identities are useful for refit caches.
    pub payload: P,
}

impl<P> NodeSpec<P> {
    pub fn root(id: NodeId, bound: RoundSide, payload: P) -> Self {
        Self {
            id,
            parent: None,
            parent_containment: ContainmentCertificate::Computed,
            bound,
            payload,
        }
    }

    pub fn child(
        id: NodeId,
        parent: NodeId,
        bound: RoundSide,
        payload: P,
        parent_containment: ContainmentCertificate,
    ) -> Self {
        Self {
            id,
            parent: Some(parent),
            parent_containment,
            bound,
            payload,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RoundIndexNode<P> {
    id: NodeId,
    parent: Option<NodeId>,
    children: Vec<NodeId>,
    bound: RoundSide,
    payload: P,
}

impl<P> RoundIndexNode<P> {
    pub fn id(&self) -> NodeId {
        self.id
    }

    pub fn parent(&self) -> Option<NodeId> {
        self.parent
    }

    pub fn children(&self) -> &[NodeId] {
        &self.children
    }

    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    pub fn bound(&self) -> &RoundSide {
        &self.bound
    }

    pub fn payload(&self) -> &P {
        &self.payload
    }
}

/// Immutable topology with a mutable, pose-stamped conservative bounds layer.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RoundIndex<P> {
    topology: TopologyKey,
    current_pose: Option<PoseKey>,
    roots: Vec<NodeId>,
    nodes: BTreeMap<NodeId, RoundIndexNode<P>>,
    predicates: PredicateConfig,
}

impl<P> RoundIndex<P> {
    pub fn build(specs: Vec<NodeSpec<P>>) -> Result<Self, RoundIndexError> {
        Self::build_for(TopologyKey::default(), specs, PredicateConfig::default())
    }

    pub fn build_for(
        topology: TopologyKey,
        specs: Vec<NodeSpec<P>>,
        predicates: PredicateConfig,
    ) -> Result<Self, RoundIndexError> {
        PredicateConfig::new(predicates.clearance)?;
        let mut nodes = BTreeMap::new();
        let mut certificates = BTreeMap::new();

        for spec in specs {
            spec.bound.validate()?;
            let id = spec.id;
            if nodes.contains_key(&id) {
                return Err(RoundIndexError::DuplicateNode(id));
            }
            certificates.insert(id, spec.parent_containment);
            nodes.insert(
                id,
                RoundIndexNode {
                    id,
                    parent: spec.parent,
                    children: Vec::new(),
                    bound: spec.bound,
                    payload: spec.payload,
                },
            );
        }

        let parent_links = nodes
            .iter()
            .filter_map(|(&id, node)| node.parent.map(|parent| (id, parent)))
            .collect::<Vec<_>>();
        for &(child, parent) in &parent_links {
            let parent_node = nodes
                .get_mut(&parent)
                .ok_or(RoundIndexError::UnknownParent { child, parent })?;
            parent_node.children.push(child);
        }
        for node in nodes.values_mut() {
            node.children.sort_unstable();
        }

        let roots = nodes
            .iter()
            .filter_map(|(&id, node)| node.parent.is_none().then_some(id))
            .collect::<Vec<_>>();
        validate_acyclic(&nodes)?;

        for (child_id, parent_id) in parent_links {
            certify_containment(
                child_id,
                nodes[&child_id].bound,
                parent_id,
                nodes[&parent_id].bound,
                certificates[&child_id],
                predicates,
            )?;
        }

        Ok(Self {
            topology,
            current_pose: None,
            roots,
            nodes,
            predicates,
        })
    }

    pub fn topology_key(&self) -> TopologyKey {
        self.topology
    }

    pub fn current_pose(&self) -> Option<PoseKey> {
        self.current_pose
    }

    pub fn roots(&self) -> &[NodeId] {
        &self.roots
    }

    pub fn nodes(&self) -> impl Iterator<Item = &RoundIndexNode<P>> {
        self.nodes.values()
    }

    pub fn node(&self, id: NodeId) -> Option<&RoundIndexNode<P>> {
        self.nodes.get(&id)
    }

    pub fn predicates(&self) -> PredicateConfig {
        self.predicates
    }

    /// Traverse source carriers against an already pulled-back query.
    ///
    /// Euclidean frusta and proximity balls should be authored in the active
    /// output chart (where walking/physics happens), then pulled back with
    /// [`RoundQuery::pullback`]. This method has no source-space distance or
    /// nearest-neighbour semantics.
    pub fn query(&self, pulled_back_query: &RoundQuery) -> QueryResult {
        let mut result = QueryResult::default();
        let mut stack = self.roots.clone();
        while let Some(id) = stack.pop() {
            result.visited_nodes += 1;
            let node = &self.nodes[&id];
            let disjoint = pulled_back_query.sides.iter().any(|query_side| {
                classify_separation(&node.bound, query_side, self.predicates)
                    == Separation::Disjoint
            });
            if disjoint {
                result.pruned_nodes += 1;
            } else if node.children.is_empty() {
                result.candidate_leaves.push(id);
            } else {
                stack.extend(node.children.iter().rev().copied());
            }
        }
        result
    }

    /// Pull an active-output-chart query back and traverse in one operation.
    ///
    /// This is the preferred entry point for rendered frusta, physical
    /// proximity balls, and sticky-walker recovery regions. The returned IDs
    /// remain persistent source-space addresses; the query metric remains the
    /// ordinary Euclidean metric of the post-Möbius chart.
    pub fn query_output_chart<A: RoundSideAutomorphism>(
        &self,
        output_chart_query: &RoundQuery,
        source_to_output: &A,
    ) -> Result<QueryResult, TransformError> {
        let pulled_back = output_chart_query.pullback(source_to_output)?;
        Ok(self.query(&pulled_back))
    }

    pub fn candidate_payloads<'a>(
        &'a self,
        result: &'a QueryResult,
    ) -> impl Iterator<Item = (NodeId, &'a P)> + 'a {
        result
            .candidate_leaves
            .iter()
            .map(|&id| (id, &self.nodes[&id].payload))
    }

    /// Atomically update animated leaf bounds and refit every affected parent.
    ///
    /// Topology, node IDs, and payloads do not change. The refitter is called
    /// deepest-first with the already-updated child bounds. It must return one
    /// parent carrier plus either a computable or explicit external
    /// containment certificate. On any error, no bounds or pose stamp change.
    pub fn refit<F>(
        &mut self,
        pose: PoseKey,
        leaf_updates: &[(NodeId, RoundSide)],
        mut refit_parent: F,
    ) -> Result<RefitReport, RoundIndexError>
    where
        F: FnMut(NodeId, &P, &[(NodeId, RoundSide)]) -> Result<RefitBound, RoundIndexError>,
    {
        let mut staged = self
            .nodes
            .iter()
            .map(|(&id, node)| (id, node.bound))
            .collect::<BTreeMap<_, _>>();
        let mut changed_leaves = BTreeSet::new();
        let mut dirty_parents = BTreeSet::new();

        for &(id, bound) in leaf_updates {
            bound.validate()?;
            let node = self
                .nodes
                .get(&id)
                .ok_or(RoundIndexError::UnknownNode(id))?;
            if !node.is_leaf() {
                return Err(RoundIndexError::NonLeafUpdate(id));
            }
            if !changed_leaves.insert(id) {
                return Err(RoundIndexError::DuplicateBoundUpdate(id));
            }
            staged.insert(id, bound);
            let mut cursor = node.parent;
            while let Some(parent) = cursor {
                dirty_parents.insert(parent);
                cursor = self.nodes[&parent].parent;
            }
        }

        let mut dirty = dirty_parents.into_iter().collect::<Vec<_>>();
        dirty.sort_by_key(|id| std::cmp::Reverse(node_depth(&self.nodes, *id)));
        for id in &dirty {
            let node = &self.nodes[id];
            let children = node
                .children
                .iter()
                .map(|&child| (child, staged[&child]))
                .collect::<Vec<_>>();
            let refit = refit_parent(*id, &node.payload, &children)?;
            refit.bound.validate()?;
            for &(child, child_bound) in &children {
                certify_containment(
                    child,
                    child_bound,
                    *id,
                    refit.bound,
                    refit.child_containment,
                    self.predicates,
                )?;
            }
            staged.insert(*id, refit.bound);
        }

        for (id, bound) in staged {
            self.nodes.get_mut(&id).expect("staged node exists").bound = bound;
        }
        self.current_pose = Some(pose);
        Ok(RefitReport {
            pose,
            updated_leaves: changed_leaves.len(),
            refit_internal_nodes: dirty.len(),
        })
    }
}

fn validate_acyclic<P>(nodes: &BTreeMap<NodeId, RoundIndexNode<P>>) -> Result<(), RoundIndexError> {
    for &start in nodes.keys() {
        let mut seen = BTreeSet::new();
        let mut cursor = Some(start);
        while let Some(id) = cursor {
            if !seen.insert(id) {
                return Err(RoundIndexError::Cycle(id));
            }
            cursor = nodes[&id].parent;
        }
    }
    Ok(())
}

fn node_depth<P>(nodes: &BTreeMap<NodeId, RoundIndexNode<P>>, id: NodeId) -> usize {
    let mut depth = 0;
    let mut cursor = nodes[&id].parent;
    while let Some(parent) = cursor {
        depth += 1;
        cursor = nodes[&parent].parent;
    }
    depth
}

fn certify_containment(
    child_id: NodeId,
    child: RoundSide,
    parent_id: NodeId,
    parent: RoundSide,
    certificate: ContainmentCertificate,
    predicates: PredicateConfig,
) -> Result<(), RoundIndexError> {
    if certificate == ContainmentCertificate::Trusted {
        return Ok(());
    }
    match classify_containment(&child, &parent, predicates) {
        Containment::Contained => Ok(()),
        Containment::NotContained => Err(RoundIndexError::ContainmentNotSatisfied {
            child: child_id,
            parent: parent_id,
        }),
        Containment::Unknown => Err(RoundIndexError::ContainmentUnproven {
            child: child_id,
            parent: parent_id,
        }),
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RefitBound {
    pub bound: RoundSide,
    pub child_containment: ContainmentCertificate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RefitReport {
    pub pose: PoseKey,
    pub updated_leaves: usize,
    pub refit_internal_nodes: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct QueryResult {
    pub candidate_leaves: Vec<NodeId>,
    pub visited_nodes: usize,
    pub pruned_nodes: usize,
}

/// Deterministically enclose bounded negative sphere sides for a parent refit.
///
/// `padding` should cover the application's floating-point and animation
/// fitting error. Unsupported planes or exterior carriers return an error
/// instead of inventing a finite Euclidean bound.
pub fn enclose_negative_spheres(
    children: &[(NodeId, RoundSide)],
    padding: f64,
) -> Result<RoundSide, RoundIndexError> {
    if children.is_empty() || !padding.is_finite() || padding < 0.0 {
        return Err(RoundIndexError::UnsupportedBoundUnion);
    }
    let mut center = [0.0; 3];
    let mut radius = 0.0;
    for (index, (_, side)) in children.iter().enumerate() {
        let RoundWallGeometry::Sphere {
            center: next_center,
            radius: next_radius,
        } = side.geometry
        else {
            return Err(RoundIndexError::UnsupportedBoundUnion);
        };
        if side.orientation != RoundSideOrientation::Negative {
            return Err(RoundIndexError::UnsupportedBoundUnion);
        }
        if index == 0 {
            center = next_center;
            radius = next_radius;
            continue;
        }
        let delta = sub(next_center, center);
        let distance = norm(delta);
        if distance + next_radius <= radius {
            continue;
        }
        if distance + radius <= next_radius {
            center = next_center;
            radius = next_radius;
            continue;
        }
        let expanded_radius = (distance + radius + next_radius) * 0.5;
        center = add(center, scale(delta, (expanded_radius - radius) / distance));
        radius = expanded_radius;
    }
    RoundSide::sphere(
        center,
        radius + padding.max(DEFAULT_CLEARANCE * radius.max(1.0)),
        RoundSideOrientation::Negative,
    )
}

/// Runtime seam corresponding to Lean's `RoundSideAutomorphism`.
///
/// Implementations must be bijections on the conformal compactification and
/// preserve the exact represented open side. This crate implements structured
/// generator words from `quilting-core`; it intentionally does not pretend an
/// arbitrary collapsed quaternionic `Mobius` has a recovered formula.
pub trait RoundSideAutomorphism {
    fn push_side(&self, side: &RoundSide) -> Result<RoundSide, TransformError>;
    fn pull_side(&self, side: &RoundSide) -> Result<RoundSide, TransformError>;
}

impl RoundSideAutomorphism for ConformalGenerator {
    fn push_side(&self, side: &RoundSide) -> Result<RoundSide, TransformError> {
        self.validate()
            .map_err(|error| TransformError::InvalidGenerator(error.to_string()))?;
        transform_generator(side, self)
    }

    fn pull_side(&self, side: &RoundSide) -> Result<RoundSide, TransformError> {
        let inverse = self
            .inverse()
            .map_err(|error| TransformError::InvalidGenerator(error.to_string()))?;
        transform_generator(side, &inverse)
    }
}

impl RoundSideAutomorphism for ConformalTransformChain {
    fn push_side(&self, side: &RoundSide) -> Result<RoundSide, TransformError> {
        self.validate()
            .map_err(|error| TransformError::InvalidGenerator(error.to_string()))?;
        self.generators
            .iter()
            .try_fold(*side, |current, generator| generator.push_side(&current))
    }

    fn pull_side(&self, side: &RoundSide) -> Result<RoundSide, TransformError> {
        self.validate()
            .map_err(|error| TransformError::InvalidGenerator(error.to_string()))?;
        self.generators
            .iter()
            .rev()
            .try_fold(*side, |current, generator| generator.pull_side(&current))
    }
}

#[derive(Debug, Clone, Copy)]
struct QuadraticSide {
    alpha: f64,
    beta: [f64; 3],
    gamma: f64,
    orientation: RoundSideOrientation,
}

impl From<&RoundSide> for QuadraticSide {
    fn from(side: &RoundSide) -> Self {
        match side.geometry {
            RoundWallGeometry::Sphere { center, radius } => Self {
                alpha: 1.0,
                beta: scale(center, -2.0),
                gamma: dot(center, center) - radius * radius,
                orientation: side.orientation,
            },
            RoundWallGeometry::Plane {
                unit_normal,
                offset,
            } => Self {
                alpha: 0.0,
                beta: unit_normal,
                gamma: -offset,
                orientation: side.orientation,
            },
        }
    }
}

impl QuadraticSide {
    fn into_round_side(self) -> Result<RoundSide, TransformError> {
        if !self.alpha.is_finite() || !finite3(self.beta) || !self.gamma.is_finite() {
            return Err(TransformError::NonFiniteResult);
        }
        if self.alpha == 0.0 {
            let length = norm(self.beta);
            if !length.is_finite() || length == 0.0 {
                return Err(TransformError::DegenerateResult);
            }
            return RoundSide::plane(
                scale(self.beta, length.recip()),
                -self.gamma / length,
                self.orientation,
            )
            .map_err(TransformError::from);
        }

        let center = scale(self.beta, -0.5 / self.alpha);
        let radius_squared = dot(center, center) - self.gamma / self.alpha;
        if !radius_squared.is_finite() || radius_squared <= 0.0 {
            return Err(TransformError::DegenerateResult);
        }
        let orientation = if self.alpha < 0.0 {
            self.orientation.complement()
        } else {
            self.orientation
        };
        RoundSide::sphere(center, radius_squared.sqrt(), orientation).map_err(TransformError::from)
    }
}

fn transform_generator(
    side: &RoundSide,
    generator: &ConformalGenerator,
) -> Result<RoundSide, TransformError> {
    let q = QuadraticSide::from(side);
    let transformed = match *generator {
        ConformalGenerator::Translation { offset } => QuadraticSide {
            alpha: q.alpha,
            beta: sub(q.beta, scale(offset, 2.0 * q.alpha)),
            gamma: q.alpha * dot(offset, offset) - dot(q.beta, offset) + q.gamma,
            orientation: q.orientation,
        },
        ConformalGenerator::UniformScale { factor } => QuadraticSide {
            // q(y/factor) * factor²; the multiplier is strictly positive.
            alpha: q.alpha,
            beta: scale(q.beta, factor),
            gamma: q.gamma * factor * factor,
            orientation: q.orientation,
        },
        ConformalGenerator::Rotation { quaternion_wxyz } => {
            let [w, x, y, z] = quaternion_wxyz;
            let rotation = Quat::new(w, x, y, z).normalize();
            let beta =
                rotation * Quat::from_point(q.beta[0], q.beta[1], q.beta[2]) * rotation.conj();
            QuadraticSide {
                alpha: q.alpha,
                beta: beta.to_point(),
                gamma: q.gamma,
                orientation: q.orientation,
            }
        }
        ConformalGenerator::SphereReflection { center, radius } => {
            // The reflection is its own inverse. Substitute
            // x = c + r²(y-c)/|y-c|² and multiply by |y-c|² > 0.
            let radius_squared = radius * radius;
            let value_at_center = q.alpha * dot(center, center) + dot(q.beta, center) + q.gamma;
            let beta_about_center =
                scale(add(scale(center, 2.0 * q.alpha), q.beta), radius_squared);
            QuadraticSide {
                alpha: value_at_center,
                beta: sub(beta_about_center, scale(center, 2.0 * value_at_center)),
                gamma: value_at_center * dot(center, center) - dot(beta_about_center, center)
                    + q.alpha * radius_squared * radius_squared,
                orientation: q.orientation,
            }
        }
    };
    transformed.into_round_side()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransformError {
    InvalidGenerator(String),
    InvalidSide(String),
    NonFiniteResult,
    DegenerateResult,
    /// Reserved for downstream automorphisms lacking an exact round-side map.
    Unsupported(&'static str),
}

impl From<RoundIndexError> for TransformError {
    fn from(value: RoundIndexError) -> Self {
        Self::InvalidSide(value.to_string())
    }
}

impl fmt::Display for TransformError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGenerator(message) => write!(f, "invalid conformal generator: {message}"),
            Self::InvalidSide(message) => write!(f, "invalid round side: {message}"),
            Self::NonFiniteResult => write!(f, "round-side transform produced non-finite values"),
            Self::DegenerateResult => write!(f, "round-side transform produced a degenerate wall"),
            Self::Unsupported(message) => write!(f, "unsupported round-side transform: {message}"),
        }
    }
}

impl Error for TransformError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RoundIndexError {
    InvalidSide(String),
    InvalidClearance,
    InvalidViewProjection,
    DuplicatePatchFace(u32),
    DuplicateNode(NodeId),
    UnknownNode(NodeId),
    UnknownParent { child: NodeId, parent: NodeId },
    Cycle(NodeId),
    NonLeafUpdate(NodeId),
    DuplicateBoundUpdate(NodeId),
    ContainmentNotSatisfied { child: NodeId, parent: NodeId },
    ContainmentUnproven { child: NodeId, parent: NodeId },
    UnsupportedBoundUnion,
    RefitFailed(String),
}

impl From<MereologyError> for RoundIndexError {
    fn from(value: MereologyError) -> Self {
        Self::InvalidSide(value.to_string())
    }
}

impl fmt::Display for RoundIndexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSide(message) => write!(f, "invalid round side: {message}"),
            Self::InvalidClearance => write!(f, "predicate clearance must be finite and nonnegative"),
            Self::InvalidViewProjection => {
                write!(f, "view-projection matrix must define six finite clip planes")
            }
            Self::DuplicatePatchFace(face) => {
                write!(f, "duplicate patch face {face}")
            }
            Self::DuplicateNode(id) => write!(f, "duplicate round-index node {}", id.0),
            Self::UnknownNode(id) => write!(f, "unknown round-index node {}", id.0),
            Self::UnknownParent { child, parent } => {
                write!(f, "node {} refers to unknown parent {}", child.0, parent.0)
            }
            Self::Cycle(id) => write!(f, "round-index parent cycle at node {}", id.0),
            Self::NonLeafUpdate(id) => write!(f, "node {} is not a leaf", id.0),
            Self::DuplicateBoundUpdate(id) => write!(f, "duplicate bound update for node {}", id.0),
            Self::ContainmentNotSatisfied { child, parent } => write!(
                f,
                "node {} is not contained in parent {}",
                child.0, parent.0
            ),
            Self::ContainmentUnproven { child, parent } => write!(
                f,
                "node {} containment in parent {} is unknown; provide an external certificate or a more conservative bound",
                child.0, parent.0
            ),
            Self::UnsupportedBoundUnion => write!(
                f,
                "cannot form a finite enclosing sphere for these carrier sides"
            ),
            Self::RefitFailed(message) => write!(f, "parent-bound refit failed: {message}"),
        }
    }
}

impl Error for RoundIndexError {}
