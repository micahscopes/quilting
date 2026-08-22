//! Experimental meshlet-style clustering and conformal/QB fit diagnostics.
//!
//! This module is deliberately offline-only. It supplies two pieces that were
//! previously conflated in the remeshing experiments:
//!
//! - deterministic, constraint-aware connected source-face clusters; and
//! - a score for an already fitted shared QB patch complex.
//!
//! Building the shared coarse triangular complex between those pieces remains
//! future work. See `../CONFORMAL_OPTIMIZER.md` for the proposed seam.

use std::collections::{BTreeMap, BTreeSet, HashSet};

use quilting_core::patch::QBTriPatch;
use quilting_core::quaternion::{Mobius, Quat};

use crate::geometry;

/// Stable identity for a source-face cluster.
///
/// The digest is computed from sorted stable source-face IDs, rather than face
/// buffer offsets. The membership remains the authoritative identity because a
/// 64-bit digest can theoretically collide.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ClusterId(pub u64);

/// Source mesh and discontinuity information used for cluster formation.
#[derive(Debug, Clone, Copy)]
pub struct ClusterInput<'a> {
    pub positions: &'a [[f64; 3]],
    pub triangles: &'a [[usize; 3]],
    /// Stable IDs corresponding one-for-one with `triangles`. Empty means face
    /// buffer indices are used; production callers should always provide IDs.
    pub source_face_ids: &'a [u64],
    /// Attribute/ownership domain per face. Empty means one domain.
    pub face_domains: &'a [u32],
    /// Edges that may be present on a cluster boundary but never in its
    /// interior. Endpoints may be supplied in either order.
    pub locked_edges: &'a [[usize; 2]],
}

/// Tuning for meshlet-style connected growth.
#[derive(Debug, Clone, Copy)]
pub struct ClusterConfig {
    pub max_triangles: usize,
    pub max_vertices: usize,
    /// Cost per newly introduced vertex. Larger values favor vertex reuse.
    pub vertex_reuse_weight: f64,
    /// Cost for disagreement with the cluster's area-weighted normal.
    pub normal_weight: f64,
    /// Cost for distance from the cluster centroid, normalized by mesh extent.
    pub spatial_weight: f64,
    /// Meshoptimizer normal-cone tradeoff when its optional seed backend is on.
    pub meshopt_cone_weight: f32,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            max_triangles: 64,
            max_vertices: 64,
            vertex_reuse_weight: 1.0,
            normal_weight: 2.0,
            spatial_weight: 0.25,
            meshopt_cone_weight: 0.25,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PrototypeCluster {
    pub id: ClusterId,
    /// Face buffer offsets, sorted by stable source-face ID.
    pub source_faces: Vec<usize>,
    /// Authoritative stable membership, sorted.
    pub source_face_ids: Vec<u64>,
    pub vertices: Vec<usize>,
    pub boundary_edges: Vec<[usize; 2]>,
    pub adjacent_clusters: Vec<ClusterId>,
    pub domain: u32,
}

#[derive(Debug, Clone)]
pub struct ClusterSet {
    pub clusters: Vec<PrototypeCluster>,
    pub face_cluster_ids: Vec<ClusterId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptimizerError {
    EmptyMesh,
    InvalidConfig(&'static str),
    InvalidVertex { face: usize, vertex: usize },
    FaceIdCount { expected: usize, actual: usize },
    DomainCount { expected: usize, actual: usize },
    DuplicateStableFaceId(u64),
    InvalidSamplePatch { sample: usize, patch: usize },
    PatchFaceCount { patches: usize, faces: usize },
    Meshopt(String),
}

impl std::fmt::Display for OptimizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyMesh => write!(f, "mesh has no positions or triangles"),
            Self::InvalidConfig(message) => write!(f, "invalid cluster config: {message}"),
            Self::InvalidVertex { face, vertex } => {
                write!(f, "face {face} references missing vertex {vertex}")
            }
            Self::FaceIdCount { expected, actual } => write!(
                f,
                "source-face ID count {actual} does not match face count {expected}",
            ),
            Self::DomainCount { expected, actual } => write!(
                f,
                "face-domain count {actual} does not match face count {expected}",
            ),
            Self::DuplicateStableFaceId(id) => write!(f, "duplicate stable source-face ID {id}"),
            Self::InvalidSamplePatch { sample, patch } => {
                write!(f, "sample {sample} references missing patch {patch}")
            }
            Self::PatchFaceCount { patches, faces } => {
                write!(
                    f,
                    "patch count {patches} does not match coarse face count {faces}"
                )
            }
            Self::Meshopt(message) => write!(f, "meshopt seed generation failed: {message}"),
        }
    }
}

impl std::error::Error for OptimizerError {}

type Edge = (usize, usize);

fn edge(a: usize, b: usize) -> Edge {
    (a.min(b), a.max(b))
}

fn face_id(input: &ClusterInput<'_>, face: usize) -> u64 {
    if input.source_face_ids.is_empty() {
        face as u64
    } else {
        input.source_face_ids[face]
    }
}

fn face_domain(input: &ClusterInput<'_>, face: usize) -> u32 {
    if input.face_domains.is_empty() {
        0
    } else {
        input.face_domains[face]
    }
}

#[derive(Debug)]
struct Topology {
    edge_faces: BTreeMap<Edge, Vec<usize>>,
    neighbors: Vec<Vec<usize>>,
    forbidden_pairs: HashSet<(usize, usize)>,
}

fn validate_cluster_input(
    input: &ClusterInput<'_>,
    config: &ClusterConfig,
) -> Result<(), OptimizerError> {
    if input.positions.is_empty() || input.triangles.is_empty() {
        return Err(OptimizerError::EmptyMesh);
    }
    if config.max_triangles == 0 {
        return Err(OptimizerError::InvalidConfig(
            "max_triangles must be positive",
        ));
    }
    if config.max_vertices < 3 {
        return Err(OptimizerError::InvalidConfig(
            "max_vertices must be at least three",
        ));
    }
    if !input.source_face_ids.is_empty() && input.source_face_ids.len() != input.triangles.len() {
        return Err(OptimizerError::FaceIdCount {
            expected: input.triangles.len(),
            actual: input.source_face_ids.len(),
        });
    }
    if !input.face_domains.is_empty() && input.face_domains.len() != input.triangles.len() {
        return Err(OptimizerError::DomainCount {
            expected: input.triangles.len(),
            actual: input.face_domains.len(),
        });
    }
    let mut ids = HashSet::with_capacity(input.triangles.len());
    for face in 0..input.triangles.len() {
        let id = face_id(input, face);
        if !ids.insert(id) {
            return Err(OptimizerError::DuplicateStableFaceId(id));
        }
        for &vertex in &input.triangles[face] {
            if vertex >= input.positions.len() {
                return Err(OptimizerError::InvalidVertex { face, vertex });
            }
        }
    }
    Ok(())
}

fn build_topology(input: &ClusterInput<'_>) -> Topology {
    let mut edge_faces: BTreeMap<Edge, Vec<usize>> = BTreeMap::new();
    for (face_index, triangle) in input.triangles.iter().enumerate() {
        for local in 0..3 {
            edge_faces
                .entry(edge(triangle[local], triangle[(local + 1) % 3]))
                .or_default()
                .push(face_index);
        }
    }

    let locked: HashSet<Edge> = input
        .locked_edges
        .iter()
        .map(|pair| edge(pair[0], pair[1]))
        .collect();
    let mut neighbors = vec![Vec::new(); input.triangles.len()];
    let mut forbidden_pairs = HashSet::new();

    for (shared_edge, incident) in &edge_faces {
        if incident.len() == 2 {
            let a = incident[0];
            let b = incident[1];
            let pair = (a.min(b), a.max(b));
            let crosses_domain = face_domain(input, a) != face_domain(input, b);
            if locked.contains(shared_edge) || crosses_domain {
                forbidden_pairs.insert(pair);
            } else {
                neighbors[a].push(b);
                neighbors[b].push(a);
            }
        } else if incident.len() > 2 {
            // A non-manifold edge is always a hard cluster boundary, even if
            // its incident faces can reach each other by another route.
            for i in 0..incident.len() {
                for j in (i + 1)..incident.len() {
                    let a = incident[i].min(incident[j]);
                    let b = incident[i].max(incident[j]);
                    forbidden_pairs.insert((a, b));
                }
            }
        }
    }
    for list in &mut neighbors {
        list.sort_unstable();
        list.dedup();
    }
    Topology {
        edge_faces,
        neighbors,
        forbidden_pairs,
    }
}

#[derive(Debug, Clone)]
struct FaceGeometry {
    normal: [f64; 3],
    area: f64,
    centroid: [f64; 3],
}

fn precompute_face_geometry(input: &ClusterInput<'_>) -> Vec<FaceGeometry> {
    input
        .triangles
        .iter()
        .map(|triangle| {
            let points = [
                input.positions[triangle[0]],
                input.positions[triangle[1]],
                input.positions[triangle[2]],
            ];
            FaceGeometry {
                normal: geometry::face_normal_normalized(&points, [0, 1, 2]),
                area: geometry::face_area(&points, [0, 1, 2]),
                centroid: [
                    (points[0][0] + points[1][0] + points[2][0]) / 3.0,
                    (points[0][1] + points[1][1] + points[2][1]) / 3.0,
                    (points[0][2] + points[1][2] + points[2][2]) / 3.0,
                ],
            }
        })
        .collect()
}

fn mesh_diagonal(positions: &[[f64; 3]]) -> f64 {
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for point in positions {
        for axis in 0..3 {
            min[axis] = min[axis].min(point[axis]);
            max[axis] = max[axis].max(point[axis]);
        }
    }
    geometry::vec3_dist(min, max).max(1e-12)
}

fn stable_cluster_id(sorted_ids: &[u64]) -> ClusterId {
    // FNV-1a with an explicit count prefix. Membership is retained alongside
    // the digest and is authoritative if a collision is ever observed.
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in (sorted_ids.len() as u64).to_le_bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    for id in sorted_ids {
        for byte in id.to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    ClusterId(hash)
}

fn candidate_crosses_barrier(
    candidate: usize,
    cluster_faces: &[usize],
    forbidden_pairs: &HashSet<(usize, usize)>,
) -> bool {
    cluster_faces
        .iter()
        .any(|&other| forbidden_pairs.contains(&(candidate.min(other), candidate.max(other))))
}

fn cluster_connected_in_order(
    input: &ClusterInput<'_>,
    config: &ClusterConfig,
    seed_order: &[usize],
) -> Result<ClusterSet, OptimizerError> {
    validate_cluster_input(input, config)?;
    let topology = build_topology(input);
    let face_geometry = precompute_face_geometry(input);
    let extent = mesh_diagonal(input.positions);
    let mut rank = vec![usize::MAX; input.triangles.len()];
    for (order, &face) in seed_order.iter().enumerate() {
        if face < rank.len() {
            rank[face] = rank[face].min(order);
        }
    }
    let mut fallback: Vec<usize> = (0..input.triangles.len()).collect();
    fallback.sort_by_key(|&face| (face_id(input, face), face));
    let mut next_rank = seed_order.len();
    for face in fallback {
        if rank[face] == usize::MAX {
            rank[face] = next_rank;
            next_rank += 1;
        }
    }
    let mut seeds: Vec<usize> = (0..input.triangles.len()).collect();
    seeds.sort_by_key(|&face| (rank[face], face_id(input, face), face));

    let mut assigned = vec![false; input.triangles.len()];
    let mut raw_clusters: Vec<Vec<usize>> = Vec::new();
    for seed in seeds {
        if assigned[seed] {
            continue;
        }
        let domain = face_domain(input, seed);
        let mut faces = vec![seed];
        assigned[seed] = true;
        let mut vertices: BTreeSet<usize> = input.triangles[seed].into_iter().collect();
        let mut normal_sum = geometry::vec3_scale(
            face_geometry[seed].normal,
            face_geometry[seed].area.max(1e-20),
        );
        let mut centroid_sum = geometry::vec3_scale(
            face_geometry[seed].centroid,
            face_geometry[seed].area.max(1e-20),
        );
        let mut area_sum = face_geometry[seed].area.max(1e-20);

        while faces.len() < config.max_triangles {
            let mut frontier = BTreeSet::new();
            for &face in &faces {
                for &neighbor in &topology.neighbors[face] {
                    if !assigned[neighbor] {
                        frontier.insert(neighbor);
                    }
                }
            }
            let cluster_normal = geometry::vec3_normalize(normal_sum);
            let cluster_centroid = geometry::vec3_scale(centroid_sum, 1.0 / area_sum);
            let mut best: Option<(usize, f64)> = None;
            for candidate in frontier {
                if face_domain(input, candidate) != domain
                    || candidate_crosses_barrier(candidate, &faces, &topology.forbidden_pairs)
                {
                    continue;
                }
                let new_vertices = input.triangles[candidate]
                    .iter()
                    .filter(|vertex| !vertices.contains(vertex))
                    .count();
                if vertices.len() + new_vertices > config.max_vertices {
                    continue;
                }
                let normal_disagreement = 1.0
                    - geometry::vec3_dot(cluster_normal, face_geometry[candidate].normal)
                        .clamp(-1.0, 1.0);
                let spatial =
                    geometry::vec3_dist(cluster_centroid, face_geometry[candidate].centroid)
                        / extent;
                let cost = config.vertex_reuse_weight * new_vertices as f64
                    + config.normal_weight * normal_disagreement
                    + config.spatial_weight * spatial;
                match best {
                    None => best = Some((candidate, cost)),
                    Some((previous, previous_cost)) => {
                        let tie = (rank[candidate], face_id(input, candidate), candidate)
                            < (rank[previous], face_id(input, previous), previous);
                        if cost < previous_cost - 1e-14
                            || ((cost - previous_cost).abs() <= 1e-14 && tie)
                        {
                            best = Some((candidate, cost));
                        }
                    }
                }
            }
            let Some((chosen, _)) = best else { break };
            assigned[chosen] = true;
            faces.push(chosen);
            vertices.extend(input.triangles[chosen]);
            let weight = face_geometry[chosen].area.max(1e-20);
            normal_sum = geometry::vec3_add(
                normal_sum,
                geometry::vec3_scale(face_geometry[chosen].normal, weight),
            );
            centroid_sum = geometry::vec3_add(
                centroid_sum,
                geometry::vec3_scale(face_geometry[chosen].centroid, weight),
            );
            area_sum += weight;
        }
        faces.sort_by_key(|&face| (face_id(input, face), face));
        raw_clusters.push(faces);
    }

    let mut face_raw_cluster = vec![usize::MAX; input.triangles.len()];
    for (cluster, faces) in raw_clusters.iter().enumerate() {
        for &face in faces {
            face_raw_cluster[face] = cluster;
        }
    }

    let mut clusters = Vec::with_capacity(raw_clusters.len());
    for (raw_index, source_faces) in raw_clusters.into_iter().enumerate() {
        let domain = face_domain(input, source_faces[0]);
        let source_face_ids: Vec<u64> = source_faces
            .iter()
            .map(|&face| face_id(input, face))
            .collect();
        let id = stable_cluster_id(&source_face_ids);
        let mut vertices = BTreeSet::new();
        let mut boundary_edges = BTreeSet::new();
        let mut adjacent_raw = BTreeSet::new();
        for &face in &source_faces {
            let triangle = input.triangles[face];
            vertices.extend(triangle);
            for local in 0..3 {
                let key = edge(triangle[local], triangle[(local + 1) % 3]);
                let mut interior = false;
                if let Some(incident) = topology.edge_faces.get(&key) {
                    for &other in incident {
                        if other == face {
                            continue;
                        }
                        let pair = (face.min(other), face.max(other));
                        if face_raw_cluster[other] == raw_index
                            && !topology.forbidden_pairs.contains(&pair)
                        {
                            interior = true;
                        } else if face_raw_cluster[other] != raw_index {
                            adjacent_raw.insert(face_raw_cluster[other]);
                        }
                    }
                }
                if !interior {
                    boundary_edges.insert(key);
                }
            }
        }
        clusters.push(PrototypeCluster {
            id,
            source_faces,
            source_face_ids,
            vertices: vertices.into_iter().collect(),
            boundary_edges: boundary_edges.into_iter().map(|(a, b)| [a, b]).collect(),
            adjacent_clusters: adjacent_raw
                .into_iter()
                .map(|index| ClusterId(index as u64))
                .collect(),
            domain,
        });
    }

    // Sort by stable identity and replace temporary adjacency indices with IDs.
    let raw_ids: Vec<ClusterId> = clusters.iter().map(|cluster| cluster.id).collect();
    for cluster in &mut clusters {
        for adjacent in &mut cluster.adjacent_clusters {
            *adjacent = raw_ids[adjacent.0 as usize];
        }
        cluster.adjacent_clusters.sort_unstable();
        cluster.adjacent_clusters.dedup();
    }
    clusters.sort_by_key(|cluster| cluster.id);

    let mut face_cluster_ids = vec![ClusterId(0); input.triangles.len()];
    for cluster in &clusters {
        for &face in &cluster.source_faces {
            face_cluster_ids[face] = cluster.id;
        }
    }
    Ok(ClusterSet {
        clusters,
        face_cluster_ids,
    })
}

/// Form deterministic connected clusters without an external backend.
pub fn cluster_connected(
    input: &ClusterInput<'_>,
    config: &ClusterConfig,
) -> Result<ClusterSet, OptimizerError> {
    let mut order: Vec<usize> = (0..input.triangles.len()).collect();
    order.sort_by_key(|&face| (face_id(input, face), face));
    cluster_connected_in_order(input, config, &order)
}

/// Use meshoptimizer meshlets as a candidate ordering, while retaining
/// Quilting's adjacency and discontinuity constraints as authoritative.
///
/// This is feature-gated because the Rust crate builds vendored C++ and follows
/// an older upstream release. It is a comparative seed backend, not the QB
/// objective and not a renderer dependency.
#[cfg(feature = "meshopt-prototype")]
pub fn cluster_meshopt_seeded(
    input: &ClusterInput<'_>,
    config: &ClusterConfig,
) -> Result<ClusterSet, OptimizerError> {
    validate_cluster_input(input, config)?;
    if input.positions.len() > u32::MAX as usize {
        return Err(OptimizerError::Meshopt("vertex count exceeds u32".into()));
    }
    let positions_f32: Vec<[f32; 3]> = input
        .positions
        .iter()
        .map(|point| [point[0] as f32, point[1] as f32, point[2] as f32])
        .collect();
    let mut indices = Vec::with_capacity(input.triangles.len() * 3);
    for triangle in input.triangles {
        indices.extend(triangle.iter().map(|&vertex| vertex as u32));
    }
    let bytes = meshopt::typed_to_bytes(&positions_f32);
    let vertices = meshopt::VertexDataAdapter::new(bytes, std::mem::size_of::<[f32; 3]>(), 0)
        .map_err(|error| OptimizerError::Meshopt(error.to_string()))?;
    let max_vertices = config.max_vertices.clamp(3, 256);
    let max_triangles = config.max_triangles.clamp(4, 512).div_ceil(4) * 4;
    let meshlets = meshopt::build_meshlets(
        &indices,
        &vertices,
        max_vertices,
        max_triangles,
        config.meshopt_cone_weight.clamp(0.0, 1.0),
    );

    let mut faces_by_triangle: BTreeMap<[u32; 3], Vec<usize>> = BTreeMap::new();
    for (face, triangle) in input.triangles.iter().enumerate() {
        let mut key = [triangle[0] as u32, triangle[1] as u32, triangle[2] as u32];
        key.sort_unstable();
        faces_by_triangle.entry(key).or_default().push(face);
    }
    for faces in faces_by_triangle.values_mut() {
        faces.sort_by_key(|&face| (face_id(input, face), face));
        faces.reverse();
    }
    let mut order = Vec::with_capacity(input.triangles.len());
    for meshlet in meshlets.iter() {
        for triangle in meshlet.triangles.chunks(3) {
            let mut key = [
                meshlet.vertices[triangle[0] as usize],
                meshlet.vertices[triangle[1] as usize],
                meshlet.vertices[triangle[2] as usize],
            ];
            key.sort_unstable();
            let face = faces_by_triangle
                .get_mut(&key)
                .and_then(Vec::pop)
                .ok_or_else(|| {
                    OptimizerError::Meshopt(format!(
                        "meshlet triangle {key:?} is absent from the source"
                    ))
                })?;
            order.push(face);
        }
    }
    cluster_connected_in_order(input, config, &order)
}

/// A source-surface sample with a known parameter location on a candidate
/// coarse patch.
#[derive(Debug, Clone, Copy)]
pub struct FitSample {
    pub patch_index: usize,
    pub barycentric: [f64; 3],
    pub target_position: [f64; 3],
    pub target_normal: [f64; 3],
}

/// A representative authored transform in the conformal robustness envelope.
#[derive(Debug, Clone)]
pub struct ConformalProbe {
    pub name: String,
    pub transform: Mobius,
}

#[derive(Debug, Clone, Copy)]
pub struct FitScoreConfig {
    pub edge_steps: usize,
    pub denominator_grid: usize,
    /// `|c*x+d|` below this value is reported as pole-near.
    pub pole_epsilon: f64,
}

impl Default for FitScoreConfig {
    fn default() -> Self {
        Self {
            edge_steps: 16,
            denominator_grid: 16,
            pole_epsilon: 1e-5,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct EuclideanError {
    pub sample_count: usize,
    pub non_finite_samples: usize,
    pub position_rms: f64,
    pub position_max: f64,
    pub position_relative_rms: f64,
    pub normal_rms_degrees: f64,
    pub normal_max_degrees: f64,
}

#[derive(Debug, Clone)]
pub struct ProbeError {
    pub name: String,
    /// Euclidean error in this active/output Möbius chart. The raw position
    /// values, rather than the source-material values, model perceived geometry
    /// and runtime walking/physics tolerances.
    pub error: EuclideanError,
    pub boundary: BoundaryAgreement,
    pub weights: WeightConditioning,
    pub peak_local_dilation: f64,
    pub pole_near_samples: usize,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct BoundaryAgreement {
    pub shared_edge_count: usize,
    pub sampled_pair_count: usize,
    pub rms_gap: f64,
    pub max_gap: f64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct WeightConditioning {
    pub min_relative_denominator: f64,
    pub max_denominator_ratio: f64,
    pub near_singular_patches: usize,
}

#[derive(Debug, Clone)]
pub struct FitScore {
    pub source: EuclideanError,
    pub conformal_probes: Vec<ProbeError>,
    pub boundary: BoundaryAgreement,
    pub weights: WeightConditioning,
}

/// Weights for turning the diagnostic vector into a candidate-ordering scalar.
/// Keeping this explicit prevents the conformal envelope from masquerading as
/// a universally invariant metric.
#[derive(Debug, Clone, Copy)]
pub struct ObjectiveWeights {
    pub source_relative_position: f64,
    pub source_normal_radians: f64,
    pub conformal_relative_envelope: f64,
    pub boundary_gap: f64,
    pub denominator_penalty: f64,
}

impl Default for ObjectiveWeights {
    fn default() -> Self {
        Self {
            source_relative_position: 1.0,
            source_normal_radians: 0.1,
            conformal_relative_envelope: 1.0,
            boundary_gap: 10.0,
            denominator_penalty: 0.01,
        }
    }
}

impl FitScore {
    pub fn scalar_objective(&self, weights: &ObjectiveWeights) -> f64 {
        let conformal_envelope = self
            .conformal_probes
            .iter()
            .map(|probe| probe.error.position_relative_rms)
            .fold(0.0, f64::max);
        let worst_boundary = self
            .conformal_probes
            .iter()
            .map(|probe| probe.boundary.max_gap)
            .fold(self.boundary.max_gap, f64::max);
        let worst_denominator = self
            .conformal_probes
            .iter()
            .map(|probe| probe.weights.min_relative_denominator)
            .fold(self.weights.min_relative_denominator, f64::min);
        let denominator_penalty = if worst_denominator > 0.0 {
            (-worst_denominator.ln()).max(0.0)
        } else {
            f64::INFINITY
        };
        weights.source_relative_position * self.source.position_relative_rms
            + weights.source_normal_radians * self.source.normal_rms_degrees.to_radians()
            + weights.conformal_relative_envelope * conformal_envelope
            + weights.boundary_gap * worst_boundary
            + weights.denominator_penalty * denominator_penalty
    }
}

fn normalize(vector: [f64; 3]) -> [f64; 3] {
    geometry::vec3_normalize(vector)
}

fn transformed_normal(transform: &Mobius, point: [f64; 3], normal: [f64; 3]) -> [f64; 3] {
    let n = normalize(normal);
    let reference = if n[0].abs() <= n[1].abs() && n[0].abs() <= n[2].abs() {
        [1.0, 0.0, 0.0]
    } else if n[1].abs() <= n[2].abs() {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    let tangent_a = normalize(geometry::vec3_cross(n, reference));
    let tangent_b = normalize(geometry::vec3_cross(n, tangent_a));
    let source = Quat::from_point(point[0], point[1], point[2]);
    let epsilon = 1e-6 * point.iter().map(|value| value.abs()).fold(1.0, f64::max);
    let mapped = transform.apply(source).to_point();
    let map_offset = |tangent: [f64; 3]| {
        let displaced = geometry::vec3_add(point, geometry::vec3_scale(tangent, epsilon));
        geometry::vec3_sub(
            transform
                .apply(Quat::from_point(displaced[0], displaced[1], displaced[2]))
                .to_point(),
            mapped,
        )
    };
    normalize(geometry::vec3_cross(
        map_offset(tangent_a),
        map_offset(tangent_b),
    ))
}

fn target_extent(points: &[[f64; 3]]) -> f64 {
    if points.is_empty() {
        return 1.0;
    }
    mesh_diagonal(points).max(1e-12)
}

fn measure_error(
    patches: &[QBTriPatch],
    samples: &[FitSample],
    transform: Option<&Mobius>,
) -> EuclideanError {
    let transformed_patches: Vec<QBTriPatch> = match transform {
        Some(map) => patches.iter().map(|patch| patch.transform(map)).collect(),
        None => patches.to_vec(),
    };
    let targets: Vec<[f64; 3]> = samples
        .iter()
        .map(|sample| match transform {
            Some(map) => map
                .apply(Quat::from_point(
                    sample.target_position[0],
                    sample.target_position[1],
                    sample.target_position[2],
                ))
                .to_point(),
            None => sample.target_position,
        })
        .collect();
    let extent = target_extent(&targets);
    let mut position_sum_sq = 0.0;
    let mut position_max = 0.0_f64;
    let mut normal_sum_sq = 0.0;
    let mut normal_max = 0.0_f64;
    let mut finite_count = 0usize;
    let mut non_finite_samples = 0usize;
    for (sample_index, sample) in samples.iter().enumerate() {
        let bary = sample.barycentric;
        let surface = transformed_patches[sample.patch_index].eval_with_normal(bary[1], bary[2]);
        let target = targets[sample_index];
        if !surface.position.iter().all(|value| value.is_finite())
            || !target.iter().all(|value| value.is_finite())
        {
            non_finite_samples += 1;
            continue;
        }
        let distance = geometry::vec3_dist(surface.position, target);
        position_sum_sq += distance * distance;
        position_max = position_max.max(distance);
        let target_normal = match transform {
            Some(map) => transformed_normal(map, sample.target_position, sample.target_normal),
            None => normalize(sample.target_normal),
        };
        let dot = geometry::vec3_dot(normalize(surface.normal), target_normal)
            .abs()
            .clamp(0.0, 1.0);
        let angle = dot.acos().to_degrees();
        normal_sum_sq += angle * angle;
        normal_max = normal_max.max(angle);
        finite_count += 1;
    }
    let denominator = finite_count.max(1) as f64;
    let position_rms = (position_sum_sq / denominator).sqrt();
    EuclideanError {
        sample_count: samples.len(),
        non_finite_samples,
        position_rms,
        position_max,
        position_relative_rms: position_rms / extent,
        normal_rms_degrees: (normal_sum_sq / denominator).sqrt(),
        normal_max_degrees: normal_max,
    }
}

fn patch_edge_point(
    patch: &QBTriPatch,
    face: [usize; 3],
    from: usize,
    to: usize,
    t: f64,
) -> [f64; 3] {
    let mut bary = [0.0; 3];
    for local in 0..3 {
        if face[local] == from {
            bary[local] = 1.0 - t;
        } else if face[local] == to {
            bary[local] = t;
        }
    }
    patch.eval(bary[1], bary[2]).to_point()
}

fn measure_boundaries(
    patches: &[QBTriPatch],
    coarse_faces: &[[usize; 3]],
    steps: usize,
) -> BoundaryAgreement {
    let mut incident: BTreeMap<Edge, Vec<usize>> = BTreeMap::new();
    for (patch_index, face) in coarse_faces.iter().enumerate() {
        for local in 0..3 {
            incident
                .entry(edge(face[local], face[(local + 1) % 3]))
                .or_default()
                .push(patch_index);
        }
    }
    let mut shared_edge_count = 0;
    let mut sampled_pair_count = 0;
    let mut sum_sq = 0.0;
    let mut max_gap = 0.0_f64;
    let steps = steps.max(1);
    for ((from, to), patches_on_edge) in incident {
        if patches_on_edge.len() != 2 {
            continue;
        }
        shared_edge_count += 1;
        let a = patches_on_edge[0];
        let b = patches_on_edge[1];
        for sample in 0..=steps {
            let t = sample as f64 / steps as f64;
            let pa = patch_edge_point(&patches[a], coarse_faces[a], from, to, t);
            let pb = patch_edge_point(&patches[b], coarse_faces[b], from, to, t);
            let gap = geometry::vec3_dist(pa, pb);
            sum_sq += gap * gap;
            max_gap = max_gap.max(gap);
            sampled_pair_count += 1;
        }
    }
    BoundaryAgreement {
        shared_edge_count,
        sampled_pair_count,
        rms_gap: (sum_sq / sampled_pair_count.max(1) as f64).sqrt(),
        max_gap,
    }
}

fn measure_weight_conditioning(
    patches: &[QBTriPatch],
    grid: usize,
    near_singular_threshold: f64,
) -> WeightConditioning {
    let grid = grid.max(1);
    let mut min_relative = f64::INFINITY;
    let mut max_ratio = 0.0_f64;
    let mut near_singular_patches = 0;
    for patch in patches {
        let weight_scale = patch
            .weights
            .iter()
            .map(|weight| weight.norm())
            .fold(0.0, f64::max)
            .max(1e-300);
        let mut patch_min = f64::INFINITY;
        let mut patch_max = 0.0_f64;
        for i in 0..=grid {
            for j in 0..=(grid - i) {
                let u = i as f64 / grid as f64;
                let v = j as f64 / grid as f64;
                let w = 1.0 - u - v;
                let denominator =
                    (w * patch.weights[0] + u * patch.weights[1] + v * patch.weights[2]).norm();
                patch_min = patch_min.min(denominator);
                patch_max = patch_max.max(denominator);
            }
        }
        let relative = patch_min / weight_scale;
        min_relative = min_relative.min(relative);
        if patch_min > 0.0 {
            max_ratio = max_ratio.max(patch_max / patch_min);
        } else {
            max_ratio = f64::INFINITY;
        }
        if relative < near_singular_threshold {
            near_singular_patches += 1;
        }
    }
    if patches.is_empty() {
        min_relative = 0.0;
    }
    WeightConditioning {
        min_relative_denominator: min_relative,
        max_denominator_ratio: max_ratio,
        near_singular_patches,
    }
}

/// Score an existing first-order QB patch complex.
///
/// `coarse_faces` provides shared global vertex identities for boundary-crack
/// measurement. Transform probes are an empirical authored envelope; the
/// returned values are not claimed to be conformally invariant.
pub fn score_patch_complex(
    patches: &[QBTriPatch],
    coarse_faces: &[[usize; 3]],
    samples: &[FitSample],
    probes: &[ConformalProbe],
    config: &FitScoreConfig,
) -> Result<FitScore, OptimizerError> {
    if patches.len() != coarse_faces.len() {
        return Err(OptimizerError::PatchFaceCount {
            patches: patches.len(),
            faces: coarse_faces.len(),
        });
    }
    for (sample_index, sample) in samples.iter().enumerate() {
        if sample.patch_index >= patches.len() {
            return Err(OptimizerError::InvalidSamplePatch {
                sample: sample_index,
                patch: sample.patch_index,
            });
        }
    }
    let source = measure_error(patches, samples, None);
    let conformal_probes = probes
        .iter()
        .map(|probe| {
            let transformed_patches: Vec<QBTriPatch> = patches
                .iter()
                .map(|patch| patch.transform(&probe.transform))
                .collect();
            let mut peak_local_dilation = 0.0_f64;
            let mut pole_near_samples = 0;
            for sample in samples {
                let point = Quat::from_point(
                    sample.target_position[0],
                    sample.target_position[1],
                    sample.target_position[2],
                );
                let denominator = probe.transform.c * point + probe.transform.d;
                if denominator.norm() < config.pole_epsilon {
                    pole_near_samples += 1;
                }
                peak_local_dilation =
                    peak_local_dilation.max(probe.transform.conformal_scale_at(point));
            }
            ProbeError {
                name: probe.name.clone(),
                error: measure_error(patches, samples, Some(&probe.transform)),
                boundary: measure_boundaries(&transformed_patches, coarse_faces, config.edge_steps),
                weights: measure_weight_conditioning(
                    &transformed_patches,
                    config.denominator_grid,
                    config.pole_epsilon,
                ),
                peak_local_dilation,
                pole_near_samples,
            }
        })
        .collect();
    Ok(FitScore {
        source,
        conformal_probes,
        boundary: measure_boundaries(patches, coarse_faces, config.edge_steps),
        weights: measure_weight_conditioning(patches, config.denominator_grid, config.pole_epsilon),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::roundtrip::{sphere_patches, tessellate_patch};
    use crate::test_shapes;

    fn input_from<'a>(
        positions: &'a [[f64; 3]],
        triangles: &'a [[usize; 3]],
        ids: &'a [u64],
        domains: &'a [u32],
    ) -> ClusterInput<'a> {
        ClusterInput {
            positions,
            triangles,
            source_face_ids: ids,
            face_domains: domains,
            locked_edges: &[],
        }
    }

    fn cluster_memberships(set: &ClusterSet) -> Vec<Vec<u64>> {
        let mut memberships: Vec<Vec<u64>> = set
            .clusters
            .iter()
            .map(|cluster| cluster.source_face_ids.clone())
            .collect();
        memberships.sort();
        memberships
    }

    fn assert_face_connected(cluster: &PrototypeCluster, triangles: &[[usize; 3]]) {
        let members: BTreeSet<usize> = cluster.source_faces.iter().copied().collect();
        let mut reached = BTreeSet::new();
        let mut frontier = vec![cluster.source_faces[0]];
        while let Some(face) = frontier.pop() {
            if !reached.insert(face) {
                continue;
            }
            let face_edges: BTreeSet<Edge> = (0..3)
                .map(|local| edge(triangles[face][local], triangles[face][(local + 1) % 3]))
                .collect();
            for &candidate in &members {
                if !reached.contains(&candidate)
                    && (0..3).any(|local| {
                        face_edges.contains(&edge(
                            triangles[candidate][local],
                            triangles[candidate][(local + 1) % 3],
                        ))
                    })
                {
                    frontier.push(candidate);
                }
            }
        }
        assert_eq!(reached, members, "cluster {:?} is disconnected", cluster.id);
    }

    #[test]
    fn sphere_clusters_are_bounded_connected_and_reorder_stable() {
        let (positions, triangles) = test_shapes::sphere(2);
        let ids: Vec<u64> = (0..triangles.len())
            .map(|face| 10_000 + face as u64 * 17)
            .collect();
        let config = ClusterConfig {
            max_triangles: 24,
            max_vertices: 24,
            ..Default::default()
        };
        let original =
            cluster_connected(&input_from(&positions, &triangles, &ids, &[]), &config).unwrap();
        assert!(original.clusters.len() > 1);
        assert!(original.clusters.iter().all(|cluster| {
            cluster.source_faces.len() <= config.max_triangles
                && cluster.vertices.len() <= config.max_vertices
        }));
        for cluster in &original.clusters {
            assert_face_connected(cluster, &triangles);
        }

        // Reverse the source face buffer while retaining stable IDs.
        let mut reordered_triangles = triangles.clone();
        let mut reordered_ids = ids.clone();
        reordered_triangles.reverse();
        reordered_ids.reverse();
        let reordered = cluster_connected(
            &input_from(&positions, &reordered_triangles, &reordered_ids, &[]),
            &config,
        )
        .unwrap();
        assert_eq!(
            cluster_memberships(&original),
            cluster_memberships(&reordered)
        );
    }

    #[test]
    fn locked_edge_never_becomes_a_cluster_interior() {
        let positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let triangles = [[0, 1, 2], [0, 2, 3]];
        let input = ClusterInput {
            positions: &positions,
            triangles: &triangles,
            source_face_ids: &[10, 20],
            face_domains: &[],
            locked_edges: &[[0, 2]],
        };
        let result = cluster_connected(
            &input,
            &ClusterConfig {
                max_triangles: 2,
                max_vertices: 4,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(result.clusters.len(), 2);
        assert!(result
            .clusters
            .iter()
            .all(|cluster| cluster.boundary_edges.contains(&[0, 2])));
    }

    #[test]
    fn cylinder_domains_do_not_mix_caps_and_side() {
        let segments = 24;
        let rings = 8;
        let (positions, triangles) = test_shapes::cylinder(segments, rings, 2.0, 1.0);
        let side_faces = rings * segments * 2;
        let mut domains = vec![0; triangles.len()];
        domains[side_faces..side_faces + segments].fill(1);
        domains[side_faces + segments..].fill(2);
        let ids: Vec<u64> = (0..triangles.len()).map(|face| face as u64).collect();
        let config = ClusterConfig {
            max_triangles: 32,
            max_vertices: 32,
            ..Default::default()
        };
        let result =
            cluster_connected(&input_from(&positions, &triangles, &ids, &domains), &config)
                .unwrap();
        for cluster in &result.clusters {
            assert!(cluster
                .source_faces
                .iter()
                .all(|&face| domains[face] == cluster.domain));
        }
    }

    fn sphere_fixture() -> (Vec<QBTriPatch>, Vec<[usize; 3]>, Vec<FitSample>) {
        let patches = sphere_patches(2.5);
        let faces = vec![
            [0, 2, 4],
            [2, 1, 4],
            [1, 3, 4],
            [3, 0, 4],
            [2, 0, 5],
            [1, 2, 5],
            [3, 1, 5],
            [0, 3, 5],
        ];
        let mut samples = Vec::new();
        for (patch_index, patch) in patches.iter().enumerate() {
            let tessellation = tessellate_patch(patch, 6);
            for sample_index in 0..tessellation.positions.len() {
                samples.push(FitSample {
                    patch_index,
                    barycentric: tessellation.bary[sample_index],
                    target_position: tessellation.positions[sample_index],
                    target_normal: tessellation.normals[sample_index],
                });
            }
        }
        (patches, faces, samples)
    }

    #[test]
    fn conformal_envelope_distinguishes_exact_qb_from_flat_baseline() {
        let (truth, faces, samples) = sphere_fixture();
        let flat: Vec<QBTriPatch> = truth
            .iter()
            .map(|patch| {
                QBTriPatch::flat(
                    patch.positions[0].to_point(),
                    patch.positions[1].to_point(),
                    patch.positions[2].to_point(),
                )
            })
            .collect();
        let probes = vec![
            ConformalProbe {
                name: "scale-8".into(),
                transform: Mobius::scale(8.0),
            },
            ConformalProbe {
                name: "off-surface reflection".into(),
                transform: Mobius::sphere_reflection(Quat::from_point(2.0, -1.0, 3.0), 1.5),
            },
        ];
        let config = FitScoreConfig::default();
        let exact = score_patch_complex(&truth, &faces, &samples, &probes, &config).unwrap();
        let baseline = score_patch_complex(&flat, &faces, &samples, &probes, &config).unwrap();
        assert!(exact.source.position_rms < 1e-10, "{:?}", exact.source);
        assert!(baseline.source.position_rms > 1e-3, "{:?}", baseline.source);
        assert!(exact.boundary.max_gap < 1e-10, "{:?}", exact.boundary);
        assert!(exact
            .conformal_probes
            .iter()
            .all(|probe| probe.error.position_relative_rms < 1e-9));
        assert!(
            baseline.scalar_objective(&Default::default())
                > exact.scalar_objective(&Default::default())
        );
    }

    #[test]
    fn boundary_metric_detects_independent_weight_ownership() {
        let (mut patches, faces, samples) = sphere_fixture();
        let exact =
            score_patch_complex(&patches, &faces, &samples, &[], &Default::default()).unwrap();
        patches[0].weights[0].x += 0.2;
        let probes = [ConformalProbe {
            name: "scale-8".into(),
            transform: Mobius::scale(8.0),
        }];
        let cracked =
            score_patch_complex(&patches, &faces, &samples, &probes, &Default::default()).unwrap();
        assert!(exact.boundary.max_gap < 1e-10);
        assert!(cracked.boundary.max_gap > 1e-4);
        assert!(cracked.conformal_probes[0].boundary.max_gap > 7.9 * cracked.boundary.max_gap);
    }

    #[cfg(feature = "meshopt-prototype")]
    #[test]
    fn meshopt_seed_backend_respects_quilting_constraints() {
        let (positions, triangles) = test_shapes::sphere(1);
        let ids: Vec<u64> = (0..triangles.len()).map(|face| face as u64).collect();
        let config = ClusterConfig {
            max_triangles: 16,
            max_vertices: 20,
            ..Default::default()
        };
        let result =
            cluster_meshopt_seeded(&input_from(&positions, &triangles, &ids, &[]), &config)
                .unwrap();
        assert_eq!(result.face_cluster_ids.len(), triangles.len());
        assert!(result.clusters.iter().all(|cluster| {
            cluster.source_faces.len() <= config.max_triangles
                && cluster.vertices.len() <= config.max_vertices
        }));
    }
}
