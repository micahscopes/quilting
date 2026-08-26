//! Stable global assembly and source correspondence for reduced QB topology.
//!
//! Chart-local copies exist so independent simplification cannot corrupt a
//! seam. This module deliberately welds those copies back by authoritative
//! [`SourceVertexId`] before fitting: both sides then share one position and,
//! eventually, one quaternion weight. It also emits source-area quadrature
//! weights rather than source-vertex counts, so the next weighted-fit stage can
//! prevent densely triangulated bevels from dominating the objective.

use std::collections::{BTreeMap, BTreeSet};

use crate::coarse_complex::{
    cut_source_topology, ChartKey, CoarseComplexError, CoarseComplexInput, CutEdgeReason,
    CutVertexKey, SourceFaceId, SourceVertexId,
};
use crate::coarse_reduction::{
    reduce_source_charts, CoarseReductionConfig, CoarseReductionError, ReducedChart,
    ReducedSourceCharts, RejectedReductionCandidate,
};
use crate::geometry;
use crate::triangle_bvh::{
    IndexedTriangle, SearchCounters, SearchError, SearchScratch, TriangleBvh,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoarsePatchConfig {
    pub reduction: CoarseReductionConfig,
    /// Each source triangle contributes `n²` deterministic subtriangle-centroid
    /// samples. The samples divide that face's normalized area weight equally.
    pub correspondence_subdivisions: usize,
    /// Maximum closest-point distance divided by the source chart AABB extent.
    pub max_correspondence_distance_ratio: f64,
    /// Hard allocation guard until correspondence gains an area-stratified
    /// sampler.
    pub maximum_correspondence_samples: usize,
    /// Hard indexed-search work guard. One candidate test visits one coarse
    /// triangle from a BVH leaf for one source sample.
    pub maximum_candidate_tests: usize,
}

impl Default for CoarsePatchConfig {
    fn default() -> Self {
        Self {
            reduction: CoarseReductionConfig::default(),
            correspondence_subdivisions: 2,
            max_correspondence_distance_ratio: 0.1,
            maximum_correspondence_samples: 1_000_000,
            maximum_candidate_tests: 50_000_000,
        }
    }
}

/// One global fit vertex. `contributors` retains every chart-local identity
/// welded into the stable source identity.
#[derive(Clone, Debug, PartialEq)]
pub struct CoarseVertex {
    pub source_vertex: SourceVertexId,
    pub position: [f64; 3],
    pub contributors: Vec<CutVertexKey>,
    /// True when any chart-local copy belonged to an owned cut boundary.
    pub constrained: bool,
}

/// Stable authoritative source position retained for candidate-independent
/// score normalization and provenance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SourceReferenceVertex {
    pub source_vertex: SourceVertexId,
    pub position: [f64; 3],
}

/// Stable within a [`CoarsePatchComplex`]. Chart reports are sorted by their
/// authoritative [`ChartKey`], so the chart index is reorder-stable.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct CoarseFaceKey {
    pub chart: usize,
    /// Oriented cyclic corner IDs, rotated to put the smallest ID first.
    pub corners: [SourceVertexId; 3],
}

#[derive(Clone, Debug, PartialEq)]
pub struct CoarseFace {
    pub key: CoarseFaceKey,
    pub vertices: [usize; 3],
}

#[derive(Clone, Debug, PartialEq)]
pub struct ChartReductionReport {
    pub key: ChartKey,
    pub requested_triangles: usize,
    pub selected_target_triangles: usize,
    pub achieved_triangles: usize,
    pub used_source_fallback: bool,
    pub backend_attempts: usize,
    pub rejected_candidates: Vec<RejectedReductionCandidate>,
    pub backend_result_error: Option<f32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StableCutEdge {
    pub source_vertices: [SourceVertexId; 2],
    pub incident_face_ids: Vec<SourceFaceId>,
    pub incident_domains: Vec<u32>,
    pub reason: CutEdgeReason,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct SourceSampleKey {
    pub face: SourceFaceId,
    pub ordinal: u32,
}

/// Deterministic area-measure correspondence from one source quadrature point
/// to one coarse triangle.
#[derive(Clone, Debug, PartialEq)]
pub struct CorrespondenceSample {
    pub key: SourceSampleKey,
    /// Coordinates in the orientation-preserving source-corner order rotated
    /// to place the smallest stable source-vertex ID first.
    pub source_barycentric: [f64; 3],
    pub target: [f64; 3],
    /// Oriented unit normal of the owning source triangle.
    pub target_normal: [f64; 3],
    pub coarse_face: usize,
    pub coarse_face_key: CoarseFaceKey,
    pub coarse_barycentric: [f64; 3],
    /// Dimensionless normalized surface-area measure. All samples sum to one.
    pub surface_weight: f64,
    /// Closest-point distance divided by the owning source chart AABB extent.
    pub distance_ratio: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CorrespondenceDiagnostics {
    pub sample_count: usize,
    /// Exact candidate count the previous chart-local exhaustive search would
    /// have visited for the same samples and coarse topology.
    pub exhaustive_candidate_tests: u128,
    /// Coarse triangles actually visited in BVH leaves.
    pub candidate_tests: usize,
    /// BVH nodes visited, including nodes rejected by their lower bound.
    pub bvh_node_visits: usize,
    pub weighted_rms_distance_ratio: f64,
    pub maximum_distance_ratio: f64,
    pub total_surface_weight: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CoarsePatchComplex {
    pub source_vertices: Vec<SourceReferenceVertex>,
    pub vertices: Vec<CoarseVertex>,
    pub faces: Vec<CoarseFace>,
    pub charts: Vec<ChartReductionReport>,
    pub cut_edges: Vec<StableCutEdge>,
    pub correspondence: Vec<CorrespondenceSample>,
    pub correspondence_diagnostics: CorrespondenceDiagnostics,
}

impl CoarsePatchComplex {
    pub fn positions(&self) -> Vec<[f64; 3]> {
        self.vertices.iter().map(|vertex| vertex.position).collect()
    }

    pub fn triangles(&self) -> Vec<[usize; 3]> {
        self.faces.iter().map(|face| face.vertices).collect()
    }

    pub fn weighted_fit_samples(&self) -> Vec<crate::linear_fit::WeightedSample> {
        self.correspondence
            .iter()
            .map(|sample| crate::linear_fit::WeightedSample {
                face_index: sample.coarse_face,
                bary: sample.coarse_barycentric,
                target: sample.target,
                surface_weight: sample.surface_weight,
            })
            .collect()
    }

    pub fn weighted_score_samples(&self) -> Vec<crate::conformal_optimizer::WeightedFitSample> {
        self.correspondence
            .iter()
            .map(|sample| crate::conformal_optimizer::WeightedFitSample {
                patch_index: sample.coarse_face,
                barycentric: sample.coarse_barycentric,
                target_position: sample.target,
                target_normal: sample.target_normal,
                surface_weight: sample.surface_weight,
            })
            .collect()
    }

    pub fn fit_score_context(
        &self,
        probes: &[crate::conformal_optimizer::ConformalProbe],
    ) -> Result<
        crate::conformal_optimizer::FitScoreContext,
        crate::conformal_optimizer::OptimizerError,
    > {
        let positions = self
            .source_vertices
            .iter()
            .map(|vertex| vertex.position)
            .collect::<Vec<_>>();
        crate::conformal_optimizer::fit_score_context_from_source(&positions, probes)
    }

    pub fn fit_shared_qb(
        &self,
        config: &crate::linear_fit::LinearFitConfig,
    ) -> Result<crate::linear_fit::LinearFitResult, crate::linear_fit::LinearFitError> {
        crate::linear_fit::linear_global_fit_weighted_full(
            &self.positions(),
            &self.triangles(),
            &self.weighted_fit_samples(),
            config,
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum CoarsePatchError {
    Reduction(CoarseReductionError),
    InvalidSubdivisions(usize),
    InvalidMaximumDistance(f64),
    InvalidSampleBudget(usize),
    InvalidCandidateTestBudget(usize),
    SampleBudgetExceeded {
        requested: usize,
        maximum: usize,
    },
    CandidateTestBudgetExceeded {
        requested: usize,
        maximum: usize,
    },
    CountOverflow,
    ReducedInputMismatch(&'static str),
    DuplicateFace(CoarseFaceKey),
    AssemblyTopology(CoarseComplexError),
    CutEdgeChanged([SourceVertexId; 2]),
    InvalidSourceNormal(SourceFaceId),
    NoCompatibleFace(SourceSampleKey),
    CorrespondenceTooFar {
        sample: SourceSampleKey,
        distance_ratio: f64,
        maximum: f64,
    },
    NonFiniteCorrespondence(SourceSampleKey),
}

impl std::fmt::Display for CoarsePatchError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Reduction(error) => write!(formatter, "coarse reduction failed: {error}"),
            Self::InvalidSubdivisions(value) => write!(
                formatter,
                "correspondence subdivisions {value} are not in 1..=64",
            ),
            Self::InvalidMaximumDistance(value) => write!(
                formatter,
                "maximum correspondence distance ratio {value} is not finite and nonnegative",
            ),
            Self::InvalidSampleBudget(value) => {
                write!(formatter, "maximum correspondence sample budget {value} is zero")
            }
            Self::InvalidCandidateTestBudget(value) => {
                write!(formatter, "maximum correspondence candidate-test budget {value} is zero")
            }
            Self::SampleBudgetExceeded { requested, maximum } => write!(
                formatter,
                "correspondence requests {requested} samples, exceeding budget {maximum}",
            ),
            Self::CandidateTestBudgetExceeded { requested, maximum } => write!(
                formatter,
                "indexed correspondence reached candidate test {requested}, exceeding budget {maximum}",
            ),
            Self::CountOverflow => write!(formatter, "coarse patch dimensions overflow usize"),
            Self::ReducedInputMismatch(reason) => {
                write!(formatter, "reduced charts do not match their source: {reason}")
            }
            Self::DuplicateFace(key) => write!(formatter, "duplicate coarse face {key:?}"),
            Self::AssemblyTopology(error) => {
                write!(formatter, "assembled coarse topology failed: {error}")
            }
            Self::CutEdgeChanged(edge) => {
                write!(formatter, "assembled cut edge {edge:?} has wrong incidence")
            }
            Self::InvalidSourceNormal(face) => {
                write!(formatter, "source face {face:?} has no finite unit normal")
            }
            Self::NoCompatibleFace(sample) => write!(
                formatter,
                "source sample {sample:?} has no orientation-compatible coarse face",
            ),
            Self::CorrespondenceTooFar {
                sample,
                distance_ratio,
                maximum,
            } => write!(
                formatter,
                "source sample {sample:?} projects {distance_ratio} chart extents (limit {maximum})",
            ),
            Self::NonFiniteCorrespondence(sample) => {
                write!(formatter, "source sample {sample:?} correspondence is non-finite")
            }
        }
    }
}

impl std::error::Error for CoarsePatchError {}

impl From<CoarseReductionError> for CoarsePatchError {
    fn from(error: CoarseReductionError) -> Self {
        Self::Reduction(error)
    }
}

#[derive(Clone, Copy)]
struct ChartFrame {
    center: [f64; 3],
    extent: f64,
}

impl ChartFrame {
    fn point(self, point: [f64; 3]) -> [f64; 3] {
        [
            (point[0] - self.center[0]) / self.extent,
            (point[1] - self.center[1]) / self.extent,
            (point[2] - self.center[2]) / self.extent,
        ]
    }
}

fn validate_config(config: &CoarsePatchConfig) -> Result<(), CoarsePatchError> {
    if !(1..=64).contains(&config.correspondence_subdivisions) {
        return Err(CoarsePatchError::InvalidSubdivisions(
            config.correspondence_subdivisions,
        ));
    }
    if !config.max_correspondence_distance_ratio.is_finite()
        || config.max_correspondence_distance_ratio < 0.0
    {
        return Err(CoarsePatchError::InvalidMaximumDistance(
            config.max_correspondence_distance_ratio,
        ));
    }
    if config.maximum_correspondence_samples == 0 {
        return Err(CoarsePatchError::InvalidSampleBudget(0));
    }
    if config.maximum_candidate_tests == 0 {
        return Err(CoarsePatchError::InvalidCandidateTestBudget(0));
    }
    Ok(())
}

fn rotate_face(mut vertices: [usize; 3]) -> [usize; 3] {
    let minimum = vertices
        .iter()
        .enumerate()
        .min_by_key(|(_, vertex)| **vertex)
        .map(|(index, _)| index)
        .expect("face has three vertices");
    vertices.rotate_left(minimum);
    vertices
}

fn chart_frame(
    input: &CoarseComplexInput<'_>,
    source_faces: &[usize],
) -> Result<ChartFrame, CoarsePatchError> {
    let mut minimum = [f64::INFINITY; 3];
    let mut maximum = [f64::NEG_INFINITY; 3];
    for &face in source_faces {
        let triangle = input
            .triangles
            .get(face)
            .ok_or(CoarsePatchError::ReducedInputMismatch(
                "a chart references a missing source face",
            ))?;
        for &vertex in triangle {
            let point =
                input
                    .positions
                    .get(vertex)
                    .ok_or(CoarsePatchError::ReducedInputMismatch(
                        "a source face references a missing vertex",
                    ))?;
            for axis in 0..3 {
                minimum[axis] = minimum[axis].min(point[axis]);
                maximum[axis] = maximum[axis].max(point[axis]);
            }
        }
    }
    let center = [
        minimum[0] * 0.5 + maximum[0] * 0.5,
        minimum[1] * 0.5 + maximum[1] * 0.5,
        minimum[2] * 0.5 + maximum[2] * 0.5,
    ];
    let extent = (0..3)
        .map(|axis| maximum[axis] - minimum[axis])
        .fold(0.0f64, f64::max);
    if !extent.is_finite() || extent <= 0.0 || center.iter().any(|value| !value.is_finite()) {
        return Err(CoarsePatchError::ReducedInputMismatch(
            "a source chart has an invalid normalization frame",
        ));
    }
    Ok(ChartFrame { center, extent })
}

fn face_key(chart: usize, vertices: [usize; 3], ids: &[SourceVertexId]) -> CoarseFaceKey {
    CoarseFaceKey {
        chart,
        corners: vertices.map(|vertex| ids[vertex]),
    }
}

struct Assembled {
    vertices: Vec<CoarseVertex>,
    faces: Vec<CoarseFace>,
    charts: Vec<ChartReductionReport>,
    cut_edges: Vec<StableCutEdge>,
}

fn assemble(
    input: &CoarseComplexInput<'_>,
    reduced: &ReducedSourceCharts,
) -> Result<Assembled, CoarsePatchError> {
    let mut vertex_builders =
        BTreeMap::<SourceVertexId, ([f64; 3], BTreeSet<CutVertexKey>, bool)>::new();
    for chart in &reduced.charts {
        let stable_faces = chart
            .source_faces
            .iter()
            .map(|face| {
                input.source_face_ids.get(*face).copied().ok_or(
                    CoarsePatchError::ReducedInputMismatch(
                        "a chart references a missing source-face ID",
                    ),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        if stable_faces != chart.key.source_faces {
            return Err(CoarsePatchError::ReducedInputMismatch(
                "a chart key does not match its source faces",
            ));
        }
        for vertex in &chart.vertices {
            if input.source_vertex_ids.get(vertex.source_vertex) != Some(&vertex.key.source_vertex)
                || input.positions.get(vertex.source_vertex) != Some(&vertex.position)
            {
                return Err(CoarsePatchError::ReducedInputMismatch(
                    "a retained vertex does not match its stable source identity",
                ));
            }
            let entry = vertex_builders.entry(vertex.key.source_vertex).or_insert((
                vertex.position,
                BTreeSet::new(),
                false,
            ));
            if entry.0 != vertex.position {
                return Err(CoarsePatchError::ReducedInputMismatch(
                    "chart copies of one stable vertex disagree on position",
                ));
            }
            entry.1.insert(vertex.key.clone());
            entry.2 |= vertex.boundary_locked;
        }
    }

    let mut ids = Vec::with_capacity(vertex_builders.len());
    let mut vertices = Vec::with_capacity(vertex_builders.len());
    for (source_vertex, (position, contributors, constrained)) in vertex_builders {
        ids.push(source_vertex);
        vertices.push(CoarseVertex {
            source_vertex,
            position,
            contributors: contributors.into_iter().collect(),
            constrained,
        });
    }
    let vertex_indices = ids
        .iter()
        .enumerate()
        .map(|(index, id)| (*id, index))
        .collect::<BTreeMap<_, _>>();

    let mut faces = Vec::new();
    let mut charts = Vec::with_capacity(reduced.charts.len());
    for (chart_index, chart) in reduced.charts.iter().enumerate() {
        charts.push(ChartReductionReport {
            key: chart.key.clone(),
            requested_triangles: chart.requested_triangles,
            selected_target_triangles: chart.selected_target_triangles,
            achieved_triangles: chart.triangles.len(),
            used_source_fallback: chart.used_source_fallback,
            backend_attempts: chart.backend_attempts,
            rejected_candidates: chart.rejected_candidates.clone(),
            backend_result_error: chart.backend_result_error,
        });
        for triangle in &chart.triangles {
            let global = rotate_face(
                triangle.map(|local| vertex_indices[&chart.vertices[local].key.source_vertex]),
            );
            let key = face_key(chart_index, global, &ids);
            faces.push(CoarseFace {
                key,
                vertices: global,
            });
        }
    }
    faces.sort_by_key(|face| face.key);
    for pair in faces.windows(2) {
        if pair[0].key == pair[1].key {
            return Err(CoarsePatchError::DuplicateFace(pair[0].key));
        }
    }

    let positions = vertices
        .iter()
        .map(|vertex| vertex.position)
        .collect::<Vec<_>>();
    let triangles = faces.iter().map(|face| face.vertices).collect::<Vec<_>>();
    let face_ids = (0..faces.len())
        .map(|face| {
            u64::try_from(face)
                .map(SourceFaceId)
                .map_err(|_| CoarsePatchError::CountOverflow)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let face_domains = vec![0; faces.len()];
    cut_source_topology(&CoarseComplexInput {
        positions: &positions,
        triangles: &triangles,
        source_vertex_ids: &ids,
        source_face_ids: &face_ids,
        face_domains: &face_domains,
        locked_edges: &[],
    })
    .map_err(CoarsePatchError::AssemblyTopology)?;

    let mut incidence =
        BTreeMap::<[SourceVertexId; 2], Vec<(SourceVertexId, SourceVertexId)>>::new();
    for face in &faces {
        let corners = face.vertices.map(|vertex| ids[vertex]);
        for local in 0..3 {
            let from = corners[local];
            let to = corners[(local + 1) % 3];
            let mut stable = [from, to];
            stable.sort_unstable();
            incidence.entry(stable).or_default().push((from, to));
        }
    }
    for cut in &reduced.cut_edges {
        let incident = incidence
            .get(&cut.source_vertex_ids)
            .map_or(&[][..], Vec::as_slice);
        let valid = if cut.reason.source_boundary {
            incident.len() == 1
        } else {
            incident.len() == 2 && incident[0].0 == incident[1].1 && incident[0].1 == incident[1].0
        };
        if !valid {
            return Err(CoarsePatchError::CutEdgeChanged(cut.source_vertex_ids));
        }
    }

    Ok(Assembled {
        vertices,
        faces,
        charts,
        cut_edges: reduced
            .cut_edges
            .iter()
            .map(|edge| StableCutEdge {
                source_vertices: edge.source_vertex_ids,
                incident_face_ids: edge.incident_face_ids.clone(),
                incident_domains: edge.incident_domains.clone(),
                reason: edge.reason,
            })
            .collect(),
    })
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

fn normalize_direction(vector: [f64; 3]) -> Option<[f64; 3]> {
    let scale = vector
        .iter()
        .map(|component| component.abs())
        .fold(0.0_f64, f64::max);
    if !scale.is_finite() || scale == 0.0 {
        return None;
    }
    let scaled = geometry::vec3_scale(vector, scale.recip());
    let length = geometry::vec3_len(scaled);
    (length.is_finite() && length > 0.0).then(|| geometry::vec3_scale(scaled, length.recip()))
}

fn lattice_centroids(subdivisions: usize) -> Result<Vec<[f64; 3]>, CoarsePatchError> {
    let count = subdivisions
        .checked_mul(subdivisions)
        .ok_or(CoarsePatchError::CountOverflow)?;
    let mut result = Vec::with_capacity(count);
    let denominator = 3.0 * subdivisions as f64;
    for first in 0..subdivisions {
        for second in 0..subdivisions - first {
            let u = (3 * first + 1) as f64 / denominator;
            let v = (3 * second + 1) as f64 / denominator;
            result.push([1.0 - u - v, u, v]);
            if first + second + 1 < subdivisions {
                let u = (3 * first + 2) as f64 / denominator;
                let v = (3 * second + 2) as f64 / denominator;
                result.push([1.0 - u - v, u, v]);
            }
        }
    }
    debug_assert_eq!(result.len(), count);
    Ok(result)
}

fn canonical_source_triangle(input: &CoarseComplexInput<'_>, face: usize) -> [usize; 3] {
    let mut triangle = input.triangles[face];
    let minimum = triangle
        .iter()
        .enumerate()
        .min_by_key(|(_, vertex)| input.source_vertex_ids[**vertex])
        .map(|(index, _)| index)
        .expect("face has three vertices");
    triangle.rotate_left(minimum);
    triangle
}

fn source_face_weights(input: &CoarseComplexInput<'_>) -> Result<Vec<f64>, CoarsePatchError> {
    let mut keyed_areas = (0..input.triangles.len())
        .map(|face| {
            (
                input.source_face_ids[face],
                face,
                geometry::vec3_len(geometry::face_normal(
                    input.positions,
                    canonical_source_triangle(input, face),
                )),
            )
        })
        .collect::<Vec<_>>();
    keyed_areas.sort_by_key(|entry| entry.0);
    let maximum = keyed_areas
        .iter()
        .map(|entry| entry.2)
        .fold(0.0f64, f64::max);
    let sum = keyed_areas
        .iter()
        .map(|entry| entry.2 / maximum)
        .sum::<f64>();
    if !maximum.is_finite() || maximum <= 0.0 || !sum.is_finite() || sum <= 0.0 {
        return Err(CoarsePatchError::ReducedInputMismatch(
            "source surface area cannot be normalized",
        ));
    }
    let mut result = vec![0.0; keyed_areas.len()];
    for (_, face, area) in keyed_areas {
        result[face] = area / maximum / sum;
    }
    Ok(result)
}

fn correspondence(
    input: &CoarseComplexInput<'_>,
    reduced: &[ReducedChart],
    assembled: &Assembled,
    config: &CoarsePatchConfig,
) -> Result<(Vec<CorrespondenceSample>, CorrespondenceDiagnostics), CoarsePatchError> {
    let lattice = lattice_centroids(config.correspondence_subdivisions)?;
    let face_weights = source_face_weights(input)?;
    let per_face_samples = lattice.len() as f64;
    let positions = assembled
        .vertices
        .iter()
        .map(|vertex| vertex.position)
        .collect::<Vec<_>>();
    let mut chart_faces = vec![Vec::new(); reduced.len()];
    for (face, coarse) in assembled.faces.iter().enumerate() {
        chart_faces[coarse.key.chart].push(face);
    }

    let sample_capacity = input
        .triangles
        .len()
        .checked_mul(lattice.len())
        .ok_or(CoarsePatchError::CountOverflow)?;
    let exhaustive_candidate_tests =
        reduced
            .iter()
            .enumerate()
            .try_fold(0u128, |total, (chart_index, chart)| {
                let chart_tests = (chart.source_faces.len() as u128)
                    .checked_mul(lattice.len() as u128)
                    .and_then(|samples| samples.checked_mul(chart_faces[chart_index].len() as u128))
                    .ok_or(CoarsePatchError::CountOverflow)?;
                total
                    .checked_add(chart_tests)
                    .ok_or(CoarsePatchError::CountOverflow)
            })?;
    let mut samples = Vec::with_capacity(sample_capacity);
    let mut search_counters = SearchCounters::default();
    let mut search_scratch = SearchScratch::default();
    let mut weighted_squared = 0.0;
    let mut maximum_distance = 0.0f64;
    for (chart_index, chart) in reduced.iter().enumerate() {
        let frame = chart_frame(input, &chart.source_faces)?;
        let coarse_index = TriangleBvh::new(
            chart_faces[chart_index]
                .iter()
                .map(|&coarse_face| {
                    let face = &assembled.faces[coarse_face];
                    let triangle = face.vertices.map(|vertex| frame.point(positions[vertex]));
                    let normal = geometry::vec3_cross(
                        geometry::vec3_sub(triangle[1], triangle[0]),
                        geometry::vec3_sub(triangle[2], triangle[0]),
                    );
                    IndexedTriangle {
                        stable_index: coarse_face,
                        positions: triangle,
                        orientation: normal,
                    }
                })
                .collect::<Vec<_>>(),
        );
        for &source_face in &chart.source_faces {
            let triangle_indices = canonical_source_triangle(input, source_face);
            let source_triangle = triangle_indices.map(|vertex| input.positions[vertex]);
            let source_normalized = source_triangle.map(|point| frame.point(point));
            let source_normal = geometry::vec3_cross(
                geometry::vec3_sub(source_normalized[1], source_normalized[0]),
                geometry::vec3_sub(source_normalized[2], source_normalized[0]),
            );
            let target_normal = normalize_direction(source_normal).ok_or(
                CoarsePatchError::InvalidSourceNormal(input.source_face_ids[source_face]),
            )?;
            for (ordinal, &source_barycentric) in lattice.iter().enumerate() {
                let key = SourceSampleKey {
                    face: input.source_face_ids[source_face],
                    ordinal: u32::try_from(ordinal).map_err(|_| CoarsePatchError::CountOverflow)?,
                };
                let target = barycentric_point(source_triangle, source_barycentric);
                let target_normalized = barycentric_point(source_normalized, source_barycentric);
                let best = coarse_index
                    .nearest_orientation_compatible(
                        target_normalized,
                        source_normal,
                        &mut search_scratch,
                        &mut search_counters,
                        config.maximum_candidate_tests,
                    )
                    .map_err(|error| match error {
                        SearchError::CountOverflow => CoarsePatchError::CountOverflow,
                        SearchError::CandidateBudgetExceeded { attempted, maximum } => {
                            CoarsePatchError::CandidateTestBudgetExceeded {
                                requested: attempted,
                                maximum,
                            }
                        }
                    })?;
                let Some(best) = best else {
                    return Err(CoarsePatchError::NoCompatibleFace(key));
                };
                let coarse_face = best.stable_index;
                let coarse_barycentric = best.barycentric;
                let squared_distance = best.squared_distance;
                let distance_ratio = squared_distance.sqrt();
                let surface_weight = face_weights[source_face] / per_face_samples;
                if !distance_ratio.is_finite()
                    || !surface_weight.is_finite()
                    || surface_weight <= 0.0
                    || coarse_barycentric.iter().any(|value| !value.is_finite())
                {
                    return Err(CoarsePatchError::NonFiniteCorrespondence(key));
                }
                if distance_ratio > config.max_correspondence_distance_ratio {
                    return Err(CoarsePatchError::CorrespondenceTooFar {
                        sample: key,
                        distance_ratio,
                        maximum: config.max_correspondence_distance_ratio,
                    });
                }
                weighted_squared += surface_weight * squared_distance;
                maximum_distance = maximum_distance.max(distance_ratio);
                samples.push(CorrespondenceSample {
                    key,
                    source_barycentric,
                    target,
                    target_normal,
                    coarse_face,
                    coarse_face_key: assembled.faces[coarse_face].key,
                    coarse_barycentric,
                    surface_weight,
                    distance_ratio,
                });
            }
        }
    }
    samples.sort_by_key(|sample| sample.key);
    let total_surface_weight = samples.iter().map(|sample| sample.surface_weight).sum();
    let sample_count = samples.len();
    Ok((
        samples,
        CorrespondenceDiagnostics {
            sample_count,
            exhaustive_candidate_tests,
            candidate_tests: search_counters.candidate_tests,
            bvh_node_visits: search_counters.node_visits,
            weighted_rms_distance_ratio: weighted_squared.sqrt(),
            maximum_distance_ratio: maximum_distance,
            total_surface_weight,
        },
    ))
}

/// Build the first complete, provenance-bearing coarse patch complex. This
/// stops before QB fitting but emits exactly the global positions, triangles,
/// and area-aware correspondences that a weighted shared fit requires.
pub fn build_coarse_patch_complex(
    input: &CoarseComplexInput<'_>,
    config: &CoarsePatchConfig,
) -> Result<CoarsePatchComplex, CoarsePatchError> {
    validate_config(config)?;
    let sample_count = input
        .triangles
        .len()
        .checked_mul(
            config
                .correspondence_subdivisions
                .checked_mul(config.correspondence_subdivisions)
                .ok_or(CoarsePatchError::CountOverflow)?,
        )
        .ok_or(CoarsePatchError::CountOverflow)?;
    if sample_count > config.maximum_correspondence_samples {
        return Err(CoarsePatchError::SampleBudgetExceeded {
            requested: sample_count,
            maximum: config.maximum_correspondence_samples,
        });
    }
    let reduced = reduce_source_charts(input, &config.reduction)?;
    let assembled = assemble(input, &reduced)?;
    let (correspondence, correspondence_diagnostics) =
        correspondence(input, &reduced.charts, &assembled, config)?;
    let referenced_source_vertices = input
        .triangles
        .iter()
        .flat_map(|triangle| triangle.iter().copied())
        .collect::<BTreeSet<_>>();
    let mut source_vertices = referenced_source_vertices
        .into_iter()
        .map(|vertex| SourceReferenceVertex {
            source_vertex: input.source_vertex_ids[vertex],
            position: input.positions[vertex],
        })
        .collect::<Vec<_>>();
    source_vertices.sort_by_key(|vertex| vertex.source_vertex);
    Ok(CoarsePatchComplex {
        source_vertices,
        vertices: assembled.vertices,
        faces: assembled.faces,
        charts: assembled.charts,
        cut_edges: assembled.cut_edges,
        correspondence,
        correspondence_diagnostics,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Grid {
        side: usize,
        positions: Vec<[f64; 3]>,
        triangles: Vec<[usize; 3]>,
        vertex_ids: Vec<SourceVertexId>,
        face_ids: Vec<SourceFaceId>,
        domains: Vec<u32>,
    }

    fn grid(side: usize) -> Grid {
        let mut positions = Vec::new();
        for y in 0..side {
            for x in 0..side {
                positions.push([
                    x as f64 / (side - 1) as f64,
                    y as f64 / (side - 1) as f64,
                    0.0,
                ]);
            }
        }
        let mut triangles = Vec::new();
        for y in 0..side - 1 {
            for x in 0..side - 1 {
                let a = y * side + x;
                let b = a + 1;
                let d = (y + 1) * side + x;
                let c = d + 1;
                triangles.push([a, b, c]);
                triangles.push([a, c, d]);
            }
        }
        Grid {
            side,
            vertex_ids: (0..positions.len())
                .map(|index| SourceVertexId(1_000 + index as u64 * 17))
                .collect(),
            face_ids: (0..triangles.len())
                .map(|index| SourceFaceId(10_000 + index as u64 * 23))
                .collect(),
            domains: vec![0; triangles.len()],
            positions,
            triangles,
        }
    }

    fn input(grid: &Grid) -> CoarseComplexInput<'_> {
        CoarseComplexInput {
            positions: &grid.positions,
            triangles: &grid.triangles,
            source_vertex_ids: &grid.vertex_ids,
            source_face_ids: &grid.face_ids,
            face_domains: &grid.domains,
            locked_edges: &[],
        }
    }

    fn config() -> CoarsePatchConfig {
        CoarsePatchConfig {
            reduction: CoarseReductionConfig {
                target_ratio: 0.25,
                target_error: 1.0,
            },
            correspondence_subdivisions: 2,
            max_correspondence_distance_ratio: 0.1,
            maximum_correspondence_samples: 10_000,
            maximum_candidate_tests: 1_000_000,
        }
    }

    fn reversed_buffers(source: &Grid) -> Grid {
        let mut old_to_new = vec![0usize; source.positions.len()];
        let vertex_order = (0..source.positions.len()).rev().collect::<Vec<_>>();
        for (new, &old) in vertex_order.iter().enumerate() {
            old_to_new[old] = new;
        }
        let face_order = (0..source.triangles.len()).rev().collect::<Vec<_>>();
        Grid {
            side: source.side,
            positions: vertex_order
                .iter()
                .map(|old| source.positions[*old])
                .collect(),
            vertex_ids: vertex_order
                .iter()
                .map(|old| source.vertex_ids[*old])
                .collect(),
            triangles: face_order
                .iter()
                .map(|old| source.triangles[*old].map(|vertex| old_to_new[vertex]))
                .collect(),
            face_ids: face_order.iter().map(|old| source.face_ids[*old]).collect(),
            domains: face_order.iter().map(|old| source.domains[*old]).collect(),
        }
    }

    #[test]
    fn planar_grid_welds_and_corresponds_independently_of_buffer_order() {
        let source = grid(5);
        let complex = build_coarse_patch_complex(&input(&source), &config()).unwrap();
        assert_eq!(complex.charts.len(), 1);
        assert_eq!(complex.charts[0].requested_triangles, 8);
        assert!(complex.charts[0].achieved_triangles >= 14);
        assert!(complex.charts[0].achieved_triangles < 32);
        assert_eq!(complex.correspondence.len(), 32 * 4);
        assert_eq!(
            complex.correspondence_diagnostics.sample_count,
            complex.correspondence.len(),
        );
        assert!(complex.correspondence_diagnostics.maximum_distance_ratio < 1.0e-12);
        assert!((complex.correspondence_diagnostics.total_surface_weight - 1.0).abs() < 1.0e-12);
        for sample in &complex.correspondence {
            assert!((sample.coarse_barycentric.iter().sum::<f64>() - 1.0).abs() < 1.0e-12);
            assert!(sample
                .coarse_barycentric
                .iter()
                .all(|coordinate| *coordinate >= -1.0e-12));
            assert_eq!(
                sample.coarse_face_key,
                complex.faces[sample.coarse_face].key
            );
            assert!((sample.target_normal[2] - 1.0).abs() < 1.0e-12);
        }
        let fit = complex
            .fit_shared_qb(&crate::linear_fit::LinearFitConfig::default())
            .unwrap();
        assert_eq!(fit.patches.len(), complex.faces.len());
        let score = crate::conformal_optimizer::score_patch_complex_weighted(
            &fit.patches,
            &complex.triangles(),
            &complex.weighted_score_samples(),
            &complex.fit_score_context(&[]).unwrap(),
            &Default::default(),
        )
        .unwrap();
        assert!(score.source.position_rms < 1.0e-12);
        assert!(score.source.normal_rms_degrees < 1.0e-8);

        let reordered = reversed_buffers(&source);
        let reordered_complex = build_coarse_patch_complex(&input(&reordered), &config()).unwrap();
        assert_eq!(complex, reordered_complex);

        let mut cyclic = grid(5);
        for triangle in &mut cyclic.triangles {
            triangle.rotate_left(1);
        }
        let cyclic_complex = build_coarse_patch_complex(&input(&cyclic), &config()).unwrap();
        assert_eq!(complex, cyclic_complex);
    }

    #[test]
    fn domain_chart_copies_weld_back_to_one_global_source_vertex() {
        let mut source = grid(3);
        for (face, domain) in source.domains.iter_mut().enumerate() {
            let cell_x = (face / 2) % (source.side - 1);
            *domain = cell_x as u32;
        }
        let complex = build_coarse_patch_complex(
            &input(&source),
            &CoarsePatchConfig {
                reduction: CoarseReductionConfig {
                    target_ratio: 1.0,
                    target_error: 0.0,
                },
                correspondence_subdivisions: 1,
                max_correspondence_distance_ratio: 0.1,
                maximum_correspondence_samples: 10_000,
                maximum_candidate_tests: 1_000_000,
            },
        )
        .unwrap();

        assert_eq!(complex.charts.len(), 2);
        assert_eq!(complex.vertices.len(), source.positions.len());
        let seam_ids = [
            source.vertex_ids[1],
            source.vertex_ids[4],
            source.vertex_ids[7],
        ];
        for id in seam_ids {
            let vertex = complex
                .vertices
                .iter()
                .find(|vertex| vertex.source_vertex == id)
                .unwrap();
            assert_eq!(vertex.contributors.len(), 2);
            assert!(vertex.constrained);
        }
        assert!(complex
            .cut_edges
            .iter()
            .any(|edge| edge.reason.domain_boundary));
    }

    #[test]
    fn correspondence_uses_surface_area_instead_of_face_count() {
        let positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.01, 0.0],
        ];
        let triangles = [[0, 1, 2], [0, 2, 3]];
        let vertex_ids = (0..4)
            .map(|index| SourceVertexId(100 + index))
            .collect::<Vec<_>>();
        let face_ids = [SourceFaceId(1_000), SourceFaceId(2_000)];
        let complex = build_coarse_patch_complex(
            &CoarseComplexInput {
                positions: &positions,
                triangles: &triangles,
                source_vertex_ids: &vertex_ids,
                source_face_ids: &face_ids,
                face_domains: &[0, 0],
                locked_edges: &[],
            },
            &CoarsePatchConfig {
                reduction: CoarseReductionConfig {
                    target_ratio: 1.0,
                    target_error: 0.0,
                },
                correspondence_subdivisions: 1,
                max_correspondence_distance_ratio: 0.1,
                maximum_correspondence_samples: 10_000,
                maximum_candidate_tests: 1_000_000,
            },
        )
        .unwrap();

        assert_eq!(complex.correspondence.len(), 2);
        let large = complex
            .correspondence
            .iter()
            .find(|sample| sample.key.face == face_ids[0])
            .unwrap();
        let small = complex
            .correspondence
            .iter()
            .find(|sample| sample.key.face == face_ids[1])
            .unwrap();
        assert!((large.surface_weight / small.surface_weight - 100.0).abs() < 1.0e-10);
        assert!((large.surface_weight + small.surface_weight - 1.0).abs() < 1.0e-12);

        let reordered_positions = [positions[3], positions[2], positions[1], positions[0]];
        let reordered_vertex_ids = [vertex_ids[3], vertex_ids[2], vertex_ids[1], vertex_ids[0]];
        let old_to_new = [3, 2, 1, 0];
        let reordered_triangles = [
            triangles[1].map(|vertex| old_to_new[vertex]),
            triangles[0].map(|vertex| old_to_new[vertex]),
        ];
        let reordered_face_ids = [face_ids[1], face_ids[0]];
        let reordered = build_coarse_patch_complex(
            &CoarseComplexInput {
                positions: &reordered_positions,
                triangles: &reordered_triangles,
                source_vertex_ids: &reordered_vertex_ids,
                source_face_ids: &reordered_face_ids,
                face_domains: &[0, 0],
                locked_edges: &[],
            },
            &CoarsePatchConfig {
                reduction: CoarseReductionConfig {
                    target_ratio: 1.0,
                    target_error: 0.0,
                },
                correspondence_subdivisions: 1,
                max_correspondence_distance_ratio: 0.1,
                maximum_correspondence_samples: 10_000,
                maximum_candidate_tests: 1_000_000,
            },
        )
        .unwrap();
        assert_eq!(complex, reordered);
    }

    #[test]
    fn invalid_correspondence_policy_and_work_budgets_fail_closed() {
        let source = grid(2);
        assert!(matches!(
            build_coarse_patch_complex(
                &input(&source),
                &CoarsePatchConfig {
                    correspondence_subdivisions: 0,
                    ..CoarsePatchConfig::default()
                },
            ),
            Err(CoarsePatchError::InvalidSubdivisions(0))
        ));
        assert!(matches!(
            build_coarse_patch_complex(
                &input(&source),
                &CoarsePatchConfig {
                    max_correspondence_distance_ratio: f64::NAN,
                    ..CoarsePatchConfig::default()
                },
            ),
            Err(CoarsePatchError::InvalidMaximumDistance(value)) if value.is_nan()
        ));
        assert!(matches!(
            build_coarse_patch_complex(
                &input(&source),
                &CoarsePatchConfig {
                    maximum_correspondence_samples: 0,
                    ..CoarsePatchConfig::default()
                },
            ),
            Err(CoarsePatchError::InvalidSampleBudget(0))
        ));
        assert!(matches!(
            build_coarse_patch_complex(
                &input(&source),
                &CoarsePatchConfig {
                    maximum_correspondence_samples: 1,
                    ..CoarsePatchConfig::default()
                },
            ),
            Err(CoarsePatchError::SampleBudgetExceeded {
                requested: 8,
                maximum: 1,
            })
        ));
        assert!(matches!(
            build_coarse_patch_complex(
                &input(&source),
                &CoarsePatchConfig {
                    maximum_candidate_tests: 0,
                    ..CoarsePatchConfig::default()
                },
            ),
            Err(CoarsePatchError::InvalidCandidateTestBudget(0))
        ));
        assert!(matches!(
            build_coarse_patch_complex(
                &input(&source),
                &CoarsePatchConfig {
                    maximum_candidate_tests: 1,
                    ..CoarsePatchConfig::default()
                },
            ),
            Err(CoarsePatchError::CandidateTestBudgetExceeded {
                requested: 2,
                maximum: 1,
            })
        ));
    }
}
