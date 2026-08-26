//! Provenance-bearing source topology for shared QB coarse complexes.
//!
//! Mesh simplification is deliberately downstream of this module. Quilting
//! first validates authored identity and oriented topology, derives cut edges
//! from boundaries/domains/explicit locks, closes dangling interior cuts with
//! explicit induced edges, and splits *face corners* into independent fans.
//! A backend may simplify the resulting charts, but it cannot infer, weld, or
//! discard these ownership boundaries.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::geometry;

/// Stable authored vertex identity. It is never inferred from position.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SourceVertexId(pub u64);

/// Stable authored face identity.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SourceFaceId(pub u64);

/// Source geometry and ownership required by a provenance-bearing optimizer.
#[derive(Clone, Copy, Debug)]
pub struct CoarseComplexInput<'a> {
    pub positions: &'a [[f64; 3]],
    pub triangles: &'a [[usize; 3]],
    /// Required, unique, and one-for-one with `positions`.
    pub source_vertex_ids: &'a [SourceVertexId],
    /// Required, unique, and one-for-one with `triangles`.
    pub source_face_ids: &'a [SourceFaceId],
    /// Required and one-for-one with `triangles`. A domain represents material,
    /// attribute, animation, or other ownership that simplification may not
    /// cross.
    pub face_domains: &'a [u32],
    /// Authored geometric edges that must survive as chart boundaries.
    /// Endpoints are source vertex-buffer indices and may be supplied in either
    /// order.
    pub locked_edges: &'a [[usize; 2]],
}

/// Reorder-stable authoritative identity for one connected cut chart.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ChartKey {
    pub domain: u32,
    pub source_faces: Vec<SourceFaceId>,
}

/// Reorder-stable identity for one cut copy of an authored vertex.
///
/// `source_face_fan` distinguishes the copies created when a cut edge splits
/// the incident face-corner fan at the same source vertex.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct CutVertexKey {
    pub source_vertex: SourceVertexId,
    pub source_face_fan: Vec<SourceFaceId>,
}

#[derive(Clone, Debug)]
pub struct CutVertex {
    pub key: CutVertexKey,
    pub source_vertex: usize,
    pub position: [f64; 3],
    /// Mesh backends must lock this vertex and preserve the cut border.
    pub boundary_locked: bool,
}

/// One simplification chart with stable face order and corner-cut local indices.
#[derive(Clone, Debug)]
pub struct CutChart {
    pub key: ChartKey,
    pub vertices: Vec<CutVertex>,
    pub triangles: Vec<[usize; 3]>,
    /// Source face-buffer index corresponding one-for-one with `triangles`.
    pub source_faces: Vec<usize>,
    /// Local unoriented edges with one incident chart face, sorted.
    pub boundary_edges: Vec<[usize; 2]>,
}

/// Why a source edge is owned by the cut topology instead of a simplifier.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CutEdgeReason {
    pub source_boundary: bool,
    pub explicit_lock: bool,
    pub domain_boundary: bool,
    /// Conservatively extended from an authored hard edge so an open cut cannot
    /// terminate at an interior manifold vertex and create a bow-tie boundary.
    pub induced_cut: bool,
}

/// Stable cut-edge record. Incident IDs/domains are sorted and authoritative.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CutEdge {
    pub source_vertices: [usize; 2],
    pub source_vertex_ids: [SourceVertexId; 2],
    pub incident_face_ids: Vec<SourceFaceId>,
    pub incident_domains: Vec<u32>,
    pub reason: CutEdgeReason,
}

#[derive(Clone, Debug)]
pub struct CutSourceTopology {
    pub charts: Vec<CutChart>,
    pub cut_edges: Vec<CutEdge>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CoarseComplexError {
    EmptyPositions,
    EmptyTriangles,
    VertexIdCount { expected: usize, actual: usize },
    FaceIdCount { expected: usize, actual: usize },
    DomainCount { expected: usize, actual: usize },
    DuplicateVertexId(SourceVertexId),
    DuplicateFaceId(SourceFaceId),
    NonFinitePosition { vertex: usize },
    InvalidFaceVertex { face: usize, vertex: usize },
    DegenerateFace { face: usize },
    InvalidLockedVertex { edge: usize, vertex: usize },
    LockedEdgeAbsent { edge: [usize; 2] },
    NonManifoldEdge { edge: [usize; 2], incident: usize },
    NonManifoldVertex { vertex: usize },
    InconsistentWinding { edge: [usize; 2] },
    InvalidCutChart(&'static str),
    CountOverflow,
}

impl std::fmt::Display for CoarseComplexError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyPositions => write!(formatter, "coarse-complex source has no vertices"),
            Self::EmptyTriangles => write!(formatter, "coarse-complex source has no faces"),
            Self::VertexIdCount { expected, actual } => write!(
                formatter,
                "source-vertex ID count {actual} does not match vertex count {expected}",
            ),
            Self::FaceIdCount { expected, actual } => write!(
                formatter,
                "source-face ID count {actual} does not match face count {expected}",
            ),
            Self::DomainCount { expected, actual } => write!(
                formatter,
                "face-domain count {actual} does not match face count {expected}",
            ),
            Self::DuplicateVertexId(id) => write!(formatter, "duplicate source vertex ID {id:?}"),
            Self::DuplicateFaceId(id) => write!(formatter, "duplicate source face ID {id:?}"),
            Self::NonFinitePosition { vertex } => {
                write!(formatter, "source vertex {vertex} is non-finite")
            }
            Self::InvalidFaceVertex { face, vertex } => {
                write!(
                    formatter,
                    "source face {face} references missing vertex {vertex}"
                )
            }
            Self::DegenerateFace { face } => write!(formatter, "source face {face} is degenerate"),
            Self::InvalidLockedVertex { edge, vertex } => write!(
                formatter,
                "locked edge {edge} references missing vertex {vertex}",
            ),
            Self::LockedEdgeAbsent { edge } => write!(
                formatter,
                "locked edge {}–{} is absent from the source topology",
                edge[0], edge[1],
            ),
            Self::NonManifoldEdge { edge, incident } => write!(
                formatter,
                "source edge {}–{} has {incident} incident faces",
                edge[0], edge[1],
            ),
            Self::NonManifoldVertex { vertex } => {
                write!(
                    formatter,
                    "source vertex {vertex} has disconnected face fans"
                )
            }
            Self::InconsistentWinding { edge } => write!(
                formatter,
                "source edge {}–{} has inconsistent face winding",
                edge[0], edge[1],
            ),
            Self::InvalidCutChart(message) => write!(formatter, "invalid cut chart: {message}"),
            Self::CountOverflow => write!(formatter, "coarse-complex dimensions overflow usize"),
        }
    }
}

impl std::error::Error for CoarseComplexError {}

type Edge = (usize, usize);

#[derive(Clone, Copy, Debug)]
struct EdgeIncident {
    face: usize,
    from: usize,
    to: usize,
}

#[derive(Debug)]
struct DisjointSet {
    parent: Vec<usize>,
}

impl DisjointSet {
    fn new(count: usize) -> Self {
        Self {
            parent: (0..count).collect(),
        }
    }

    fn root(&mut self, mut value: usize) -> usize {
        while self.parent[value] != value {
            self.parent[value] = self.parent[self.parent[value]];
            value = self.parent[value];
        }
        value
    }

    fn join(&mut self, left: usize, right: usize) {
        let left = self.root(left);
        let right = self.root(right);
        if left != right {
            let minimum = left.min(right);
            let maximum = left.max(right);
            self.parent[maximum] = minimum;
        }
    }
}

fn edge(left: usize, right: usize) -> Edge {
    (left.min(right), left.max(right))
}

fn stable_edge_key(input: &CoarseComplexInput<'_>, source_edge: Edge) -> [SourceVertexId; 2] {
    let mut key = [
        input.source_vertex_ids[source_edge.0],
        input.source_vertex_ids[source_edge.1],
    ];
    key.sort_unstable();
    key
}

fn corner_for_vertex(triangle: [usize; 3], vertex: usize) -> usize {
    triangle
        .iter()
        .position(|candidate| *candidate == vertex)
        .expect("validated edge endpoint belongs to incident triangle")
}

fn validate_input(input: &CoarseComplexInput<'_>) -> Result<(), CoarseComplexError> {
    if input.positions.is_empty() {
        return Err(CoarseComplexError::EmptyPositions);
    }
    if input.triangles.is_empty() {
        return Err(CoarseComplexError::EmptyTriangles);
    }
    if input.source_vertex_ids.len() != input.positions.len() {
        return Err(CoarseComplexError::VertexIdCount {
            expected: input.positions.len(),
            actual: input.source_vertex_ids.len(),
        });
    }
    if input.source_face_ids.len() != input.triangles.len() {
        return Err(CoarseComplexError::FaceIdCount {
            expected: input.triangles.len(),
            actual: input.source_face_ids.len(),
        });
    }
    if input.face_domains.len() != input.triangles.len() {
        return Err(CoarseComplexError::DomainCount {
            expected: input.triangles.len(),
            actual: input.face_domains.len(),
        });
    }

    let mut vertex_ids = HashSet::with_capacity(input.source_vertex_ids.len());
    for (vertex, (position, id)) in input
        .positions
        .iter()
        .zip(input.source_vertex_ids)
        .enumerate()
    {
        if position.iter().any(|component| !component.is_finite()) {
            return Err(CoarseComplexError::NonFinitePosition { vertex });
        }
        if !vertex_ids.insert(*id) {
            return Err(CoarseComplexError::DuplicateVertexId(*id));
        }
    }
    let mut face_ids = HashSet::with_capacity(input.source_face_ids.len());
    for (face, (triangle, id)) in input
        .triangles
        .iter()
        .zip(input.source_face_ids)
        .enumerate()
    {
        if !face_ids.insert(*id) {
            return Err(CoarseComplexError::DuplicateFaceId(*id));
        }
        for &vertex in triangle {
            if vertex >= input.positions.len() {
                return Err(CoarseComplexError::InvalidFaceVertex { face, vertex });
            }
        }
        let normal_length = geometry::vec3_len(geometry::face_normal(input.positions, *triangle));
        if triangle[0] == triangle[1]
            || triangle[1] == triangle[2]
            || triangle[2] == triangle[0]
            || !normal_length.is_finite()
            || normal_length <= f64::MIN_POSITIVE
        {
            return Err(CoarseComplexError::DegenerateFace { face });
        }
    }
    for (edge_index, endpoints) in input.locked_edges.iter().enumerate() {
        for &vertex in endpoints {
            if vertex >= input.positions.len() {
                return Err(CoarseComplexError::InvalidLockedVertex {
                    edge: edge_index,
                    vertex,
                });
            }
        }
    }
    Ok(())
}

fn source_edges(input: &CoarseComplexInput<'_>) -> BTreeMap<Edge, Vec<EdgeIncident>> {
    let mut edges = BTreeMap::<Edge, Vec<EdgeIncident>>::new();
    for (face, triangle) in input.triangles.iter().enumerate() {
        for local in 0..3 {
            let from = triangle[local];
            let to = triangle[(local + 1) % 3];
            edges
                .entry(edge(from, to))
                .or_default()
                .push(EdgeIncident { face, from, to });
        }
    }
    edges
}

fn non_manifold_vertex(
    vertex_count: usize,
    triangles: &[[usize; 3]],
    edges: &BTreeMap<Edge, Vec<EdgeIncident>>,
) -> Option<usize> {
    let mut vertex_faces = vec![BTreeSet::new(); vertex_count];
    let mut vertex_edges = vec![Vec::new(); vertex_count];
    for (face, triangle) in triangles.iter().enumerate() {
        for &vertex in triangle {
            vertex_faces[vertex].insert(face);
        }
    }
    for (&source_edge, incident) in edges {
        vertex_edges[source_edge.0].push((source_edge, incident));
        vertex_edges[source_edge.1].push((source_edge, incident));
    }
    for vertex in 0..vertex_count {
        let faces = &vertex_faces[vertex];
        if faces.is_empty() {
            continue;
        }
        let mut neighbors = BTreeMap::<usize, Vec<usize>>::new();
        let mut boundary_edges = 0usize;
        for &(_, incident) in &vertex_edges[vertex] {
            match incident.as_slice() {
                [only] => {
                    debug_assert!(faces.contains(&only.face));
                    boundary_edges += 1;
                }
                [left, right] => {
                    neighbors.entry(left.face).or_default().push(right.face);
                    neighbors.entry(right.face).or_default().push(left.face);
                }
                _ => return Some(vertex),
            }
        }
        let mut reached = BTreeSet::new();
        let mut frontier = vec![*faces.iter().next().expect("nonempty face fan")];
        while let Some(face) = frontier.pop() {
            if !reached.insert(face) {
                continue;
            }
            frontier.extend(neighbors.get(&face).into_iter().flatten().copied());
        }
        let degrees = faces
            .iter()
            .map(|face| neighbors.get(face).map_or(0, Vec::len))
            .collect::<Vec<_>>();
        let valid_link = if boundary_edges == 0 {
            degrees.iter().all(|degree| *degree == 2)
        } else if faces.len() == 1 {
            boundary_edges == 2 && degrees == [0]
        } else {
            boundary_edges == 2
                && degrees.iter().filter(|degree| **degree == 1).count() == 2
                && degrees.iter().all(|degree| *degree == 1 || *degree == 2)
        };
        if reached != *faces || !valid_link {
            return Some(vertex);
        }
    }
    None
}

fn close_cut_edges(
    input: &CoarseComplexInput<'_>,
    edges: &BTreeMap<Edge, Vec<EdgeIncident>>,
    mut cuts: BTreeMap<Edge, CutEdgeReason>,
) -> BTreeMap<Edge, CutEdgeReason> {
    let mut incident_edges = vec![Vec::new(); input.positions.len()];
    for &source_edge in edges.keys() {
        incident_edges[source_edge.0].push(source_edge);
        incident_edges[source_edge.1].push(source_edge);
    }
    for list in &mut incident_edges {
        list.sort_by_key(|source_edge| stable_edge_key(input, *source_edge));
    }
    let mut vertex_order = (0..input.positions.len()).collect::<Vec<_>>();
    vertex_order.sort_by_key(|vertex| input.source_vertex_ids[*vertex]);

    loop {
        let mut promotion = None;
        for &vertex in &vertex_order {
            let cut_degree = incident_edges[vertex]
                .iter()
                .filter(|source_edge| cuts.contains_key(source_edge))
                .count();
            if cut_degree != 1 {
                continue;
            }
            promotion = incident_edges[vertex]
                .iter()
                .copied()
                .find(|source_edge| !cuts.contains_key(source_edge));
            if promotion.is_some() {
                break;
            }
        }
        let Some(source_edge) = promotion else {
            break;
        };
        cuts.insert(
            source_edge,
            CutEdgeReason {
                source_boundary: false,
                explicit_lock: false,
                domain_boundary: false,
                induced_cut: true,
            },
        );
    }
    cuts
}

pub(crate) fn validate_cut_chart(
    vertices: &mut [CutVertex],
    triangles: &[[usize; 3]],
) -> Result<Vec<[usize; 2]>, CoarseComplexError> {
    if triangles.is_empty() {
        return Err(CoarseComplexError::InvalidCutChart("it has no faces"));
    }
    let positions = vertices
        .iter()
        .map(|vertex| vertex.position)
        .collect::<Vec<_>>();
    let mut edges = BTreeMap::<Edge, Vec<EdgeIncident>>::new();
    for (face, triangle) in triangles.iter().enumerate() {
        if triangle.iter().any(|vertex| *vertex >= vertices.len()) {
            return Err(CoarseComplexError::InvalidCutChart(
                "a face references a missing vertex",
            ));
        }
        let normal_length = geometry::vec3_len(geometry::face_normal(&positions, *triangle));
        if triangle[0] == triangle[1]
            || triangle[1] == triangle[2]
            || triangle[2] == triangle[0]
            || !normal_length.is_finite()
            || normal_length <= f64::MIN_POSITIVE
        {
            return Err(CoarseComplexError::InvalidCutChart("a face is degenerate"));
        }
        for local in 0..3 {
            let from = triangle[local];
            let to = triangle[(local + 1) % 3];
            edges
                .entry(edge(from, to))
                .or_default()
                .push(EdgeIncident { face, from, to });
        }
    }
    for incident in edges.values() {
        if incident.is_empty() || incident.len() > 2 {
            return Err(CoarseComplexError::InvalidCutChart(
                "an edge is not manifold",
            ));
        }
        if incident.len() == 2
            && (incident[0].from != incident[1].to || incident[0].to != incident[1].from)
        {
            return Err(CoarseComplexError::InvalidCutChart(
                "an interior edge has inconsistent winding",
            ));
        }
    }
    if non_manifold_vertex(vertices.len(), triangles, &edges).is_some() {
        return Err(CoarseComplexError::InvalidCutChart(
            "a vertex link is not one path or cycle",
        ));
    }
    let mut used = vec![false; vertices.len()];
    for triangle in triangles {
        for &vertex in triangle {
            used[vertex] = true;
        }
    }
    if used.iter().any(|value| !*value) {
        return Err(CoarseComplexError::InvalidCutChart(
            "the compact topology contains an unused vertex",
        ));
    }

    let mut face_neighbors = vec![Vec::new(); triangles.len()];
    let mut boundary_degree = vec![0usize; vertices.len()];
    let mut boundary_edges = Vec::new();
    for (&local_edge, incident) in &edges {
        if incident.len() == 2 {
            face_neighbors[incident[0].face].push(incident[1].face);
            face_neighbors[incident[1].face].push(incident[0].face);
        } else {
            boundary_degree[local_edge.0] += 1;
            boundary_degree[local_edge.1] += 1;
            boundary_edges.push([local_edge.0, local_edge.1]);
        }
    }
    if boundary_degree
        .iter()
        .any(|degree| *degree != 0 && *degree != 2)
    {
        return Err(CoarseComplexError::InvalidCutChart(
            "a boundary vertex does not have degree two",
        ));
    }
    for (vertex, degree) in vertices.iter_mut().zip(boundary_degree) {
        vertex.boundary_locked = degree == 2;
    }
    let mut reached = BTreeSet::new();
    let mut frontier = vec![0usize];
    while let Some(face) = frontier.pop() {
        if reached.insert(face) {
            frontier.extend(face_neighbors[face].iter().copied());
        }
    }
    if reached.len() != triangles.len() {
        return Err(CoarseComplexError::InvalidCutChart(
            "the emitted face topology is disconnected",
        ));
    }
    Ok(boundary_edges)
}

/// Validate and cut source face-corner connectivity across every owned edge.
pub fn cut_source_topology(
    input: &CoarseComplexInput<'_>,
) -> Result<CutSourceTopology, CoarseComplexError> {
    validate_input(input)?;
    let edges = source_edges(input);
    let locked = input
        .locked_edges
        .iter()
        .map(|pair| edge(pair[0], pair[1]))
        .collect::<BTreeSet<_>>();
    for &locked_edge in &locked {
        if !edges.contains_key(&locked_edge) {
            return Err(CoarseComplexError::LockedEdgeAbsent {
                edge: [locked_edge.0, locked_edge.1],
            });
        }
    }

    let mut cuts = BTreeMap::<Edge, CutEdgeReason>::new();
    for (&source_edge, incident) in &edges {
        if incident.len() > 2 {
            return Err(CoarseComplexError::NonManifoldEdge {
                edge: [source_edge.0, source_edge.1],
                incident: incident.len(),
            });
        }
        if incident.len() == 2
            && (incident[0].from != incident[1].to || incident[0].to != incident[1].from)
        {
            return Err(CoarseComplexError::InconsistentWinding {
                edge: [source_edge.0, source_edge.1],
            });
        }
        let source_boundary = incident.len() == 1;
        let explicit_lock = locked.contains(&source_edge);
        let domain_boundary = incident.len() == 2
            && input.face_domains[incident[0].face] != input.face_domains[incident[1].face];
        if source_boundary || explicit_lock || domain_boundary {
            cuts.insert(
                source_edge,
                CutEdgeReason {
                    source_boundary,
                    explicit_lock,
                    domain_boundary,
                    induced_cut: false,
                },
            );
        }
    }
    if let Some(vertex) = non_manifold_vertex(input.positions.len(), input.triangles, &edges) {
        return Err(CoarseComplexError::NonManifoldVertex { vertex });
    }
    let cuts = close_cut_edges(input, &edges, cuts);

    let corner_count = input
        .triangles
        .len()
        .checked_mul(3)
        .ok_or(CoarseComplexError::CountOverflow)?;
    let mut face_sets = DisjointSet::new(input.triangles.len());
    let mut corner_sets = DisjointSet::new(corner_count);
    let mut ordered_edges = edges.iter().collect::<Vec<_>>();
    ordered_edges.sort_by_key(|(source_edge, _)| stable_edge_key(input, **source_edge));
    for (&source_edge, incident) in &ordered_edges {
        if incident.len() != 2 || cuts.contains_key(&source_edge) {
            continue;
        }
        let left = incident[0].face;
        let right = incident[1].face;
        face_sets.join(left, right);
        for vertex in [source_edge.0, source_edge.1] {
            let left_corner = corner_for_vertex(input.triangles[left], vertex);
            let right_corner = corner_for_vertex(input.triangles[right], vertex);
            corner_sets.join(left * 3 + left_corner, right * 3 + right_corner);
        }
    }

    let mut chart_faces = BTreeMap::<usize, Vec<usize>>::new();
    for face in 0..input.triangles.len() {
        chart_faces
            .entry(face_sets.root(face))
            .or_default()
            .push(face);
    }

    let mut charts = Vec::with_capacity(chart_faces.len());
    for source_faces in chart_faces.values_mut() {
        source_faces.sort_by_key(|face| input.source_face_ids[*face]);
        let domain = input.face_domains[source_faces[0]];
        debug_assert!(source_faces
            .iter()
            .all(|face| input.face_domains[*face] == domain));
        let key = ChartKey {
            domain,
            source_faces: source_faces
                .iter()
                .map(|face| input.source_face_ids[*face])
                .collect(),
        };

        let mut root_corners = BTreeMap::<usize, Vec<(usize, usize)>>::new();
        for &face in source_faces.iter() {
            for local in 0..3 {
                let corner = face * 3 + local;
                root_corners
                    .entry(corner_sets.root(corner))
                    .or_default()
                    .push((face, local));
            }
        }

        let mut cut_vertices = root_corners
            .into_values()
            .map(|mut corners| {
                corners.sort_by_key(|(face, local)| (input.source_face_ids[*face], *local));
                let (first_face, first_local) = corners[0];
                let source_vertex = input.triangles[first_face][first_local];
                debug_assert!(corners
                    .iter()
                    .all(|(face, local)| input.triangles[*face][*local] == source_vertex));
                let source_face_fan = corners
                    .iter()
                    .map(|(face, _)| input.source_face_ids[*face])
                    .collect::<Vec<_>>();
                (
                    CutVertexKey {
                        source_vertex: input.source_vertex_ids[source_vertex],
                        source_face_fan,
                    },
                    source_vertex,
                    input.positions[source_vertex],
                    corners,
                )
            })
            .collect::<Vec<_>>();
        cut_vertices.sort_by(|left, right| left.0.cmp(&right.0));

        let mut corner_to_local = HashMap::new();
        let mut vertices = Vec::with_capacity(cut_vertices.len());
        for (local_vertex, (key, source_vertex, position, corners)) in
            cut_vertices.into_iter().enumerate()
        {
            for corner in corners {
                corner_to_local.insert(corner, local_vertex);
            }
            vertices.push(CutVertex {
                key,
                source_vertex,
                position,
                boundary_locked: false,
            });
        }
        let triangles = source_faces
            .iter()
            .map(|face| {
                [
                    corner_to_local[&(*face, 0)],
                    corner_to_local[&(*face, 1)],
                    corner_to_local[&(*face, 2)],
                ]
            })
            .collect::<Vec<_>>();
        let boundary_edges = validate_cut_chart(&mut vertices, &triangles)?;
        charts.push(CutChart {
            key,
            vertices,
            triangles,
            source_faces: source_faces.clone(),
            boundary_edges,
        });
    }
    charts.sort_by(|left, right| left.key.cmp(&right.key));

    let mut cut_edges: Vec<CutEdge> = cuts
        .into_iter()
        .map(|(source_edge, reason)| {
            let incident = &edges[&source_edge];
            let mut incident_face_ids = incident
                .iter()
                .map(|entry| input.source_face_ids[entry.face])
                .collect::<Vec<_>>();
            incident_face_ids.sort_unstable();
            let mut incident_domains = incident
                .iter()
                .map(|entry| input.face_domains[entry.face])
                .collect::<Vec<_>>();
            incident_domains.sort_unstable();
            incident_domains.dedup();
            let mut endpoints = [
                (input.source_vertex_ids[source_edge.0], source_edge.0),
                (input.source_vertex_ids[source_edge.1], source_edge.1),
            ];
            endpoints.sort_unstable_by_key(|endpoint| endpoint.0);
            CutEdge {
                source_vertices: [endpoints[0].1, endpoints[1].1],
                source_vertex_ids: [endpoints[0].0, endpoints[1].0],
                incident_face_ids,
                incident_domains,
                reason,
            }
        })
        .collect();
    cut_edges.sort_by_key(|edge| edge.source_vertex_ids);
    Ok(CutSourceTopology { charts, cut_edges })
}

#[cfg(test)]
mod tests {
    use super::*;

    type ChartSignature = (ChartKey, Vec<CutVertexKey>, Vec<[CutVertexKey; 3]>);
    type CutSignature = (
        [SourceVertexId; 2],
        Vec<SourceFaceId>,
        Vec<u32>,
        CutEdgeReason,
    );

    fn ids(vertex_count: usize, face_count: usize) -> (Vec<SourceVertexId>, Vec<SourceFaceId>) {
        (
            (0..vertex_count)
                .map(|vertex| SourceVertexId(100 + vertex as u64 * 7))
                .collect(),
            (0..face_count)
                .map(|face| SourceFaceId(1_000 + face as u64 * 11))
                .collect(),
        )
    }

    fn chart_signature(topology: &CutSourceTopology) -> Vec<ChartSignature> {
        topology
            .charts
            .iter()
            .map(|chart| {
                (
                    chart.key.clone(),
                    chart
                        .vertices
                        .iter()
                        .map(|vertex| vertex.key.clone())
                        .collect(),
                    chart
                        .triangles
                        .iter()
                        .map(|triangle| triangle.map(|vertex| chart.vertices[vertex].key.clone()))
                        .collect(),
                )
            })
            .collect()
    }

    fn cut_signature(topology: &CutSourceTopology) -> Vec<CutSignature> {
        topology
            .cut_edges
            .iter()
            .map(|edge| {
                (
                    edge.source_vertex_ids,
                    edge.incident_face_ids.clone(),
                    edge.incident_domains.clone(),
                    edge.reason,
                )
            })
            .collect()
    }

    fn assert_closed_chart_boundaries(topology: &CutSourceTopology) {
        let mut copies = BTreeMap::<[SourceVertexId; 2], usize>::new();
        for chart in &topology.charts {
            let mut degree = vec![0usize; chart.vertices.len()];
            for &boundary in &chart.boundary_edges {
                assert!(chart.vertices[boundary[0]].boundary_locked);
                assert!(chart.vertices[boundary[1]].boundary_locked);
                degree[boundary[0]] += 1;
                degree[boundary[1]] += 1;
                let mut source_edge = [
                    chart.vertices[boundary[0]].key.source_vertex,
                    chart.vertices[boundary[1]].key.source_vertex,
                ];
                source_edge.sort_unstable();
                *copies.entry(source_edge).or_default() += 1;
            }
            for (vertex, degree) in chart.vertices.iter().zip(degree) {
                assert_eq!(vertex.boundary_locked, degree == 2);
            }
        }
        for cut in &topology.cut_edges {
            assert_eq!(
                copies.get(&cut.source_vertex_ids).copied().unwrap_or(0),
                if cut.reason.source_boundary { 1 } else { 2 },
                "wrong local-border multiplicity for {:?}",
                cut.source_vertex_ids,
            );
        }
    }

    #[test]
    fn locked_diagonal_splits_both_endpoint_corner_fans() {
        let positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let triangles = [[0, 1, 2], [0, 2, 3]];
        let (vertex_ids, face_ids) = ids(positions.len(), triangles.len());
        let topology = cut_source_topology(&CoarseComplexInput {
            positions: &positions,
            triangles: &triangles,
            source_vertex_ids: &vertex_ids,
            source_face_ids: &face_ids,
            face_domains: &[0, 0],
            locked_edges: &[[2, 0]],
        })
        .unwrap();
        assert_eq!(topology.charts.len(), 2);
        assert_eq!(
            topology
                .charts
                .iter()
                .map(|chart| chart.vertices.len())
                .sum::<usize>(),
            6
        );
        let diagonal = topology
            .cut_edges
            .iter()
            .find(|edge| edge.reason.explicit_lock)
            .unwrap();
        assert_eq!(diagonal.source_vertex_ids, [vertex_ids[0], vertex_ids[2]]);
        assert_eq!(diagonal.incident_face_ids, face_ids);
    }

    #[test]
    fn open_locked_seam_splits_corner_fans_even_when_faces_reconnect_elsewhere() {
        // A lone edge cannot terminate inside a closed manifold. Complete it
        // to a deterministic whole-edge loop before cutting the corner fans.
        let positions = [
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
        ];
        let triangles = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
        let (vertex_ids, face_ids) = ids(positions.len(), triangles.len());
        let topology = cut_source_topology(&CoarseComplexInput {
            positions: &positions,
            triangles: &triangles,
            source_vertex_ids: &vertex_ids,
            source_face_ids: &face_ids,
            face_domains: &[0; 4],
            locked_edges: &[[0, 1]],
        })
        .unwrap();
        assert_eq!(topology.charts.len(), 2);
        assert_eq!(
            topology
                .charts
                .iter()
                .flat_map(|chart| &chart.vertices)
                .filter(|vertex| vertex.key.source_vertex == vertex_ids[0])
                .count(),
            2,
        );
        assert_eq!(
            topology
                .charts
                .iter()
                .flat_map(|chart| &chart.vertices)
                .filter(|vertex| vertex.key.source_vertex == vertex_ids[1])
                .count(),
            2,
        );
        assert!(topology
            .cut_edges
            .iter()
            .any(|edge| edge.reason.induced_cut));
        assert_closed_chart_boundaries(&topology);

        let reordered = cut_source_topology(&CoarseComplexInput {
            positions: &[positions[2], positions[0], positions[3], positions[1]],
            triangles: &[[3, 0, 2], [1, 2, 0], [1, 3, 2], [1, 0, 3]],
            source_vertex_ids: &[vertex_ids[2], vertex_ids[0], vertex_ids[3], vertex_ids[1]],
            source_face_ids: &[face_ids[3], face_ids[2], face_ids[1], face_ids[0]],
            face_domains: &[0; 4],
            locked_edges: &[[1, 3]],
        })
        .unwrap();
        assert_eq!(chart_signature(&topology), chart_signature(&reordered));
        assert_eq!(cut_signature(&topology), cut_signature(&reordered));
        assert_closed_chart_boundaries(&reordered);
    }

    #[test]
    fn domain_boundary_is_hard_and_charts_are_reorder_stable() {
        let positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let triangles = [[0, 1, 2], [0, 2, 3]];
        let (vertex_ids, face_ids) = ids(positions.len(), triangles.len());
        let original = cut_source_topology(&CoarseComplexInput {
            positions: &positions,
            triangles: &triangles,
            source_vertex_ids: &vertex_ids,
            source_face_ids: &face_ids,
            face_domains: &[4, 9],
            locked_edges: &[],
        })
        .unwrap();
        let reordered = cut_source_topology(&CoarseComplexInput {
            positions: &[positions[2], positions[0], positions[3], positions[1]],
            triangles: &[[1, 0, 2], [1, 3, 0]],
            source_vertex_ids: &[vertex_ids[2], vertex_ids[0], vertex_ids[3], vertex_ids[1]],
            source_face_ids: &[face_ids[1], face_ids[0]],
            face_domains: &[9, 4],
            locked_edges: &[],
        })
        .unwrap();
        assert_eq!(chart_signature(&original), chart_signature(&reordered));
        assert_eq!(
            original
                .charts
                .iter()
                .map(|chart| chart.key.clone())
                .collect::<Vec<_>>(),
            reordered
                .charts
                .iter()
                .map(|chart| chart.key.clone())
                .collect::<Vec<_>>(),
        );
        assert_eq!(cut_signature(&original), cut_signature(&reordered),);
        assert_eq!(original.cut_edges.len(), 5);
        assert!(original
            .cut_edges
            .iter()
            .any(|edge| edge.reason.domain_boundary));
        assert_closed_chart_boundaries(&original);
        assert_closed_chart_boundaries(&reordered);
    }

    #[test]
    fn malformed_identity_and_oriented_topology_fail_closed() {
        let positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        let triangles = [[0, 1, 2], [0, 1, 3]];
        let (vertex_ids, face_ids) = ids(positions.len(), triangles.len());
        assert!(matches!(
            cut_source_topology(&CoarseComplexInput {
                positions: &positions,
                triangles: &triangles,
                source_vertex_ids: &vertex_ids,
                source_face_ids: &face_ids,
                face_domains: &[0, 0],
                locked_edges: &[],
            }),
            Err(CoarseComplexError::InconsistentWinding { edge: [0, 1] })
        ));
        assert!(matches!(
            cut_source_topology(&CoarseComplexInput {
                positions: &positions,
                triangles: &[[0, 1, 2]],
                source_vertex_ids: &[
                    SourceVertexId(1),
                    SourceVertexId(1),
                    SourceVertexId(3),
                    SourceVertexId(4),
                ],
                source_face_ids: &[face_ids[0]],
                face_domains: &[0],
                locked_edges: &[],
            }),
            Err(CoarseComplexError::DuplicateVertexId(SourceVertexId(1)))
        ));
    }

    #[test]
    fn rejects_a_source_vertex_with_two_disconnected_face_fans() {
        let positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ];
        let triangles = [[0, 1, 2], [0, 3, 4]];
        let (vertex_ids, face_ids) = ids(positions.len(), triangles.len());
        assert!(matches!(
            cut_source_topology(&CoarseComplexInput {
                positions: &positions,
                triangles: &triangles,
                source_vertex_ids: &vertex_ids,
                source_face_ids: &face_ids,
                face_domains: &[0, 0],
                locked_edges: &[],
            }),
            Err(CoarseComplexError::NonManifoldVertex { vertex: 0 })
        ));
    }

    #[test]
    fn open_locked_seam_is_extended_to_an_existing_source_boundary() {
        let positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],
        ];
        let triangles = [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]];
        let (vertex_ids, face_ids) = ids(positions.len(), triangles.len());
        let topology = cut_source_topology(&CoarseComplexInput {
            positions: &positions,
            triangles: &triangles,
            source_vertex_ids: &vertex_ids,
            source_face_ids: &face_ids,
            face_domains: &[0; 4],
            locked_edges: &[[0, 4]],
        })
        .unwrap();

        let authored = topology
            .cut_edges
            .iter()
            .find(|edge| edge.reason.explicit_lock)
            .unwrap();
        assert_eq!(authored.source_vertex_ids, [vertex_ids[0], vertex_ids[4]]);
        assert_eq!(
            topology
                .cut_edges
                .iter()
                .filter(|edge| edge.reason.induced_cut)
                .count(),
            1,
        );
        assert_closed_chart_boundaries(&topology);
    }

    #[test]
    fn an_already_closed_locked_loop_needs_no_induced_cut() {
        let positions = [
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
        ];
        let triangles = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
        let (vertex_ids, face_ids) = ids(positions.len(), triangles.len());
        let topology = cut_source_topology(&CoarseComplexInput {
            positions: &positions,
            triangles: &triangles,
            source_vertex_ids: &vertex_ids,
            source_face_ids: &face_ids,
            face_domains: &[0; 4],
            locked_edges: &[[0, 1], [1, 2], [2, 0]],
        })
        .unwrap();

        assert_eq!(topology.charts.len(), 2);
        assert_eq!(topology.cut_edges.len(), 3);
        assert!(topology
            .cut_edges
            .iter()
            .all(|edge| edge.reason.explicit_lock && !edge.reason.induced_cut));
        assert_closed_chart_boundaries(&topology);
    }

    #[test]
    fn multiple_domain_and_locked_edges_can_meet_at_one_vertex() {
        let positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],
        ];
        let triangles = [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]];
        let (vertex_ids, face_ids) = ids(positions.len(), triangles.len());
        let topology = cut_source_topology(&CoarseComplexInput {
            positions: &positions,
            triangles: &triangles,
            source_vertex_ids: &vertex_ids,
            source_face_ids: &face_ids,
            face_domains: &[0, 1, 2, 0],
            locked_edges: &[[0, 4]],
        })
        .unwrap();

        assert_eq!(topology.charts.len(), 4);
        assert_eq!(
            topology
                .cut_edges
                .iter()
                .filter(|edge| edge.reason.domain_boundary)
                .count(),
            3,
        );
        assert_eq!(
            topology
                .cut_edges
                .iter()
                .filter(|edge| edge.reason.explicit_lock)
                .count(),
            1,
        );
        assert!(topology
            .cut_edges
            .iter()
            .all(|edge| !edge.reason.induced_cut));
        assert_closed_chart_boundaries(&topology);
    }
}
