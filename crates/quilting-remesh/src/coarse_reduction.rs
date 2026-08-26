//! Constrained meshoptimizer reduction inside provenance-safe source charts.
//!
//! [`crate::coarse_complex`] owns topology and seam decisions. This module lets
//! meshoptimizer remove chart-interior triangles, then independently validates
//! that every owned border is byte-for-byte the same stable edge set and that
//! the result remains an oriented connected 2-manifold. The output still has
//! chart-local boundary copies; welding and source correspondence belong to the
//! next, Quilting-owned assembly stage.

use std::collections::{BTreeMap, BTreeSet};

use crate::coarse_complex::{
    cut_source_topology, validate_cut_chart, ChartKey, CoarseComplexError, CoarseComplexInput,
    CutChart, CutEdge, CutVertex, CutVertexKey, SourceVertexId,
};

/// Per-chart reduction policy. Targets are requests: locked chart boundaries
/// can impose a strictly larger topological minimum.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoarseReductionConfig {
    /// Requested output/input triangle ratio in `(0, 1]`, applied per chart.
    pub target_ratio: f64,
    /// Maximum meshoptimizer error relative to each normalized chart extent,
    /// in `[0, 1]`.
    pub target_error: f32,
}

impl Default for CoarseReductionConfig {
    fn default() -> Self {
        Self {
            target_ratio: 0.25,
            target_error: 0.01,
        }
    }
}

/// One validated chart result. Vertices are a stable-key-sorted subset of the
/// source chart; `source_faces` retains the complete source ownership domain.
#[derive(Clone, Debug)]
pub struct ReducedChart {
    pub key: ChartKey,
    pub vertices: Vec<CutVertex>,
    pub triangles: Vec<[usize; 3]>,
    pub boundary_edges: Vec<[usize; 2]>,
    pub source_faces: Vec<usize>,
    pub requested_triangles: usize,
    /// Target selected by the deterministic backoff search after complete
    /// output validation. This can exceed `requested_triangles`; the exact
    /// source count is the fail-closed upper bound. Candidate validity is not
    /// assumed to be monotone, so this is not claimed to be the globally
    /// smallest valid target.
    pub selected_target_triangles: usize,
    /// True when every attempted backend candidate failed and the independently
    /// validated exact source chart was selected instead.
    pub used_source_fallback: bool,
    /// Number of meshoptimizer candidates evaluated for this chart.
    pub backend_attempts: usize,
    /// Candidates rejected by geometry, topology, or boundary validation,
    /// retained in deterministic attempt order.
    pub rejected_candidates: Vec<RejectedReductionCandidate>,
    /// Relative meshoptimizer error in normalized chart coordinates. `None`
    /// means exact source topology was selected without a backend result.
    pub backend_result_error: Option<f32>,
}

/// Compact, actionable evidence for one rejected meshoptimizer candidate.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RejectedReductionCandidate {
    pub target_triangles: usize,
    pub category: ReductionRejectionCategory,
    pub reason: &'static str,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ReductionRejectionCategory {
    CountOverflow,
    InvalidMeshoptOutput,
    InvalidReducedTopology,
    BoundaryChanged,
    Unexpected,
}

impl std::fmt::Display for ReductionRejectionCategory {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::CountOverflow => "count-overflow",
            Self::InvalidMeshoptOutput => "invalid-meshopt-output",
            Self::InvalidReducedTopology => "invalid-reduced-topology",
            Self::BoundaryChanged => "boundary-changed",
            Self::Unexpected => "unexpected-candidate-error",
        };
        formatter.write_str(name)
    }
}

/// Validated chart-local reduction plus the authoritative source cut records.
#[derive(Clone, Debug)]
pub struct ReducedSourceCharts {
    pub charts: Vec<ReducedChart>,
    pub cut_edges: Vec<CutEdge>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum CoarseReductionError {
    Topology(CoarseComplexError),
    InvalidTargetRatio(f64),
    InvalidTargetError(f32),
    CountOverflow,
    IndexOverflow {
        chart: ChartKey,
    },
    PositionCollision {
        chart: ChartKey,
        left: SourceVertexId,
        right: SourceVertexId,
    },
    InvalidBackendGeometry {
        chart: ChartKey,
        reason: &'static str,
    },
    InvalidMeshoptOutput {
        chart: ChartKey,
        reason: &'static str,
    },
    InvalidReducedChart {
        chart: ChartKey,
        reason: &'static str,
    },
    BoundaryChanged {
        chart: ChartKey,
    },
}

impl std::fmt::Display for CoarseReductionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Topology(error) => write!(formatter, "source topology failed: {error}"),
            Self::InvalidTargetRatio(value) => {
                write!(formatter, "coarse reduction ratio {value} is not in (0, 1]")
            }
            Self::InvalidTargetError(value) => {
                write!(formatter, "coarse reduction error {value} is not in [0, 1]")
            }
            Self::CountOverflow => write!(formatter, "coarse reduction dimensions overflow usize"),
            Self::IndexOverflow { chart } => write!(
                formatter,
                "chart {:?} has more vertices than meshoptimizer's u32 index space",
                chart.source_faces,
            ),
            Self::PositionCollision { chart, left, right } => write!(
                formatter,
                "chart {:?} vertices {left:?} and {right:?} collide after f32 normalization",
                chart.source_faces,
            ),
            Self::InvalidBackendGeometry { chart, reason } => write!(
                formatter,
                "chart {:?} cannot be represented safely for meshoptimizer: {reason}",
                chart.source_faces,
            ),
            Self::InvalidMeshoptOutput { chart, reason } => write!(
                formatter,
                "meshoptimizer returned invalid chart {:?}: {reason}",
                chart.source_faces,
            ),
            Self::InvalidReducedChart { chart, reason } => write!(
                formatter,
                "reduced chart {:?} is invalid: {reason}",
                chart.source_faces,
            ),
            Self::BoundaryChanged { chart } => write!(
                formatter,
                "reduction changed an owned boundary in chart {:?}",
                chart.source_faces,
            ),
        }
    }
}

impl std::error::Error for CoarseReductionError {}

impl From<CoarseComplexError> for CoarseReductionError {
    fn from(error: CoarseComplexError) -> Self {
        Self::Topology(error)
    }
}

type Edge = (usize, usize);
type CompactChart = (Vec<CutVertex>, Vec<[usize; 3]>, Vec<[usize; 2]>);

struct NormalizedPositions {
    exact: Vec<[f64; 3]>,
    quantized: Vec<[f32; 3]>,
}

fn edge(left: usize, right: usize) -> Edge {
    (left.min(right), left.max(right))
}

fn stable_edge(vertices: &[CutVertex], endpoints: [usize; 2]) -> [CutVertexKey; 2] {
    let mut result = [
        vertices[endpoints[0]].key.clone(),
        vertices[endpoints[1]].key.clone(),
    ];
    result.sort_unstable();
    result
}

fn validate_config(config: &CoarseReductionConfig) -> Result<(), CoarseReductionError> {
    if !config.target_ratio.is_finite() || config.target_ratio <= 0.0 || config.target_ratio > 1.0 {
        return Err(CoarseReductionError::InvalidTargetRatio(
            config.target_ratio,
        ));
    }
    if !config.target_error.is_finite() || config.target_error < 0.0 || config.target_error > 1.0 {
        return Err(CoarseReductionError::InvalidTargetError(
            config.target_error,
        ));
    }
    Ok(())
}

fn normalized_positions(chart: &CutChart) -> Result<NormalizedPositions, CoarseReductionError> {
    let mut minimum = [f64::INFINITY; 3];
    let mut maximum = [f64::NEG_INFINITY; 3];
    for vertex in &chart.vertices {
        for axis in 0..3 {
            minimum[axis] = minimum[axis].min(vertex.position[axis]);
            maximum[axis] = maximum[axis].max(vertex.position[axis]);
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
        return Err(CoarseReductionError::InvalidBackendGeometry {
            chart: chart.key.clone(),
            reason: "its normalization extent is non-finite or non-positive",
        });
    }
    let exact = chart
        .vertices
        .iter()
        .map(|vertex| {
            [
                (vertex.position[0] - center[0]) / extent,
                (vertex.position[1] - center[1]) / extent,
                (vertex.position[2] - center[2]) / extent,
            ]
        })
        .collect::<Vec<_>>();
    let quantized = exact
        .iter()
        .map(|position| position.map(|component| component as f32))
        .collect();
    Ok(NormalizedPositions { exact, quantized })
}

fn requested_triangles(input_triangles: usize, ratio: f64) -> Result<usize, CoarseReductionError> {
    let requested = (input_triangles as f64 * ratio).ceil();
    if !requested.is_finite() || requested > usize::MAX as f64 {
        return Err(CoarseReductionError::CountOverflow);
    }
    Ok((requested as usize).clamp(1, input_triangles))
}

fn backoff_to_valid<T, E>(
    requested: usize,
    source_target: usize,
    source_value: T,
    mut evaluate: impl FnMut(usize) -> Result<T, E>,
) -> (usize, T, usize, Vec<(usize, E)>) {
    debug_assert!(requested <= source_target);
    if requested == source_target {
        return (source_target, source_value, 0, Vec::new());
    }
    let mut accepted_target = source_target;
    let mut accepted = source_value;
    let mut rejected_target = requested.saturating_sub(1);
    let mut attempts = 0usize;
    let mut rejections = Vec::new();
    // This is a bounded deterministic probe schedule, not a proof that
    // candidate validity is monotone. It first tries the requested target and
    // then bisects the remaining interval toward the independently validated
    // exact source fallback. The selected result is the most aggressive valid
    // candidate this schedule actually attempted, not a global minimum.
    while accepted_target.saturating_sub(rejected_target) > 1 {
        let target = if rejected_target + 1 == requested {
            requested
        } else {
            rejected_target + (accepted_target - rejected_target) / 2
        };
        attempts += 1;
        match evaluate(target) {
            Ok(value) => {
                accepted_target = target;
                accepted = value;
            }
            Err(error) => {
                rejections.push((target, error));
                rejected_target = target;
            }
        }
    }
    (accepted_target, accepted, attempts, rejections)
}

fn rejection_evidence(
    target_triangles: usize,
    error: CoarseReductionError,
) -> RejectedReductionCandidate {
    let (category, reason) = match error {
        CoarseReductionError::CountOverflow => (
            ReductionRejectionCategory::CountOverflow,
            "target index count overflows usize",
        ),
        CoarseReductionError::InvalidMeshoptOutput { reason, .. } => {
            (ReductionRejectionCategory::InvalidMeshoptOutput, reason)
        }
        CoarseReductionError::InvalidReducedChart { reason, .. } => {
            (ReductionRejectionCategory::InvalidReducedTopology, reason)
        }
        CoarseReductionError::BoundaryChanged { .. } => (
            ReductionRejectionCategory::BoundaryChanged,
            "stable chart boundary changed",
        ),
        _ => (
            ReductionRejectionCategory::Unexpected,
            "candidate evaluation returned an unexpected error category",
        ),
    };
    RejectedReductionCandidate {
        target_triangles,
        category,
        reason,
    }
}

fn rotate_triangle(mut triangle: [usize; 3]) -> [usize; 3] {
    let minimum = triangle
        .iter()
        .enumerate()
        .min_by_key(|(_, vertex)| **vertex)
        .map(|(index, _)| index)
        .expect("triangle has three vertices");
    triangle.rotate_left(minimum);
    triangle
}

fn topology_invariants(vertex_count: usize, triangles: &[[usize; 3]]) -> (i128, usize) {
    let mut incidence = BTreeMap::<Edge, usize>::new();
    for triangle in triangles {
        for local in 0..3 {
            *incidence
                .entry(edge(triangle[local], triangle[(local + 1) % 3]))
                .or_default() += 1;
        }
    }
    let euler = vertex_count as i128 - incidence.len() as i128 + triangles.len() as i128;
    let mut boundary_neighbors = vec![Vec::new(); vertex_count];
    for (&boundary, count) in &incidence {
        if *count == 1 {
            boundary_neighbors[boundary.0].push(boundary.1);
            boundary_neighbors[boundary.1].push(boundary.0);
        }
    }
    let mut reached = BTreeSet::new();
    let mut components = 0usize;
    for vertex in 0..vertex_count {
        if boundary_neighbors[vertex].is_empty() || reached.contains(&vertex) {
            continue;
        }
        components += 1;
        let mut frontier = vec![vertex];
        while let Some(current) = frontier.pop() {
            if reached.insert(current) {
                frontier.extend(boundary_neighbors[current].iter().copied());
            }
        }
    }
    (euler, components)
}

fn validate_backend_geometry(
    source: &CutChart,
    positions: &NormalizedPositions,
) -> Result<(), CoarseReductionError> {
    let mut owners = BTreeMap::<[u32; 3], (SourceVertexId, [f64; 3])>::new();
    for ((vertex, position), exact) in source
        .vertices
        .iter()
        .zip(&positions.quantized)
        .zip(&positions.exact)
    {
        if position.iter().any(|component| !component.is_finite()) {
            return Err(CoarseReductionError::InvalidBackendGeometry {
                chart: source.key.clone(),
                reason: "a normalized position is non-finite",
            });
        }
        let bits = position.map(|component| {
            if component == 0.0 {
                0
            } else {
                component.to_bits()
            }
        });
        if let Some((owner, owner_exact)) = owners.insert(bits, (vertex.key.source_vertex, *exact))
        {
            // meshoptimizer deliberately represents exact position-sharing
            // vertices as attribute wedges. Only reject a collision introduced
            // by our f64 -> f32 backend conversion; rejecting exact wedges makes
            // ordinary glTF UV/normal seams impossible to simplify.
            if owner_exact != *exact {
                return Err(CoarseReductionError::PositionCollision {
                    chart: source.key.clone(),
                    left: owner,
                    right: vertex.key.source_vertex,
                });
            }
        }
    }
    for triangle in &source.triangles {
        let a = positions.quantized[triangle[0]];
        let b = positions.quantized[triangle[1]];
        let c = positions.quantized[triangle[2]];
        let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let quantized_normal = [
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        ];
        if quantized_normal.iter().all(|component| *component == 0.0)
            || quantized_normal
                .iter()
                .any(|component| !component.is_finite())
        {
            return Err(CoarseReductionError::InvalidBackendGeometry {
                chart: source.key.clone(),
                reason: "a source triangle becomes degenerate after f32 normalization",
            });
        }
        let original_normal = crate::geometry::vec3_cross(
            crate::geometry::vec3_sub(positions.exact[triangle[1]], positions.exact[triangle[0]]),
            crate::geometry::vec3_sub(positions.exact[triangle[2]], positions.exact[triangle[0]]),
        );
        let quantized_normal = [
            f64::from(quantized_normal[0]),
            f64::from(quantized_normal[1]),
            f64::from(quantized_normal[2]),
        ];
        let orientation = crate::geometry::vec3_dot(original_normal, quantized_normal);
        if !orientation.is_finite() || orientation <= 0.0 {
            return Err(CoarseReductionError::InvalidBackendGeometry {
                chart: source.key.clone(),
                reason: "a source triangle flips after f32 normalization",
            });
        }
    }
    Ok(())
}

fn validate_meshopt_output_geometry(
    source: &CutChart,
    positions: &NormalizedPositions,
    indices: &[u32],
) -> Result<(), CoarseReductionError> {
    if indices.is_empty() || !indices.len().is_multiple_of(3) {
        return Err(CoarseReductionError::InvalidMeshoptOutput {
            chart: source.key.clone(),
            reason: "the index count is empty or not divisible by three",
        });
    }
    for triangle in indices.chunks_exact(3) {
        let vertices = [
            triangle[0] as usize,
            triangle[1] as usize,
            triangle[2] as usize,
        ];
        if vertices
            .iter()
            .any(|vertex| *vertex >= source.vertices.len())
        {
            return Err(CoarseReductionError::InvalidMeshoptOutput {
                chart: source.key.clone(),
                reason: "an index references a missing chart vertex",
            });
        }
        let a = positions.quantized[vertices[0]];
        let b = positions.quantized[vertices[1]];
        let c = positions.quantized[vertices[2]];
        let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let quantized_normal = [
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        ];
        if quantized_normal.iter().all(|component| *component == 0.0)
            || quantized_normal
                .iter()
                .any(|component| !component.is_finite())
        {
            return Err(CoarseReductionError::InvalidMeshoptOutput {
                chart: source.key.clone(),
                reason: "an output triangle becomes degenerate after f32 normalization",
            });
        }
        let exact_normal = crate::geometry::vec3_cross(
            crate::geometry::vec3_sub(positions.exact[vertices[1]], positions.exact[vertices[0]]),
            crate::geometry::vec3_sub(positions.exact[vertices[2]], positions.exact[vertices[0]]),
        );
        let quantized_normal = quantized_normal.map(f64::from);
        let orientation = crate::geometry::vec3_dot(exact_normal, quantized_normal);
        if !orientation.is_finite() || orientation <= 0.0 {
            return Err(CoarseReductionError::InvalidMeshoptOutput {
                chart: source.key.clone(),
                reason: "an output triangle flips after f32 normalization",
            });
        }
    }
    Ok(())
}

fn compact_and_validate(
    source: &CutChart,
    indices: &[u32],
) -> Result<CompactChart, CoarseReductionError> {
    if indices.is_empty() || !indices.len().is_multiple_of(3) {
        return Err(CoarseReductionError::InvalidMeshoptOutput {
            chart: source.key.clone(),
            reason: "the index count is empty or not divisible by three",
        });
    }
    let mut used = BTreeSet::new();
    for &index in indices {
        let vertex = index as usize;
        if vertex >= source.vertices.len() {
            return Err(CoarseReductionError::InvalidMeshoptOutput {
                chart: source.key.clone(),
                reason: "an index references a missing chart vertex",
            });
        }
        used.insert(vertex);
    }
    let mut old_to_new = vec![usize::MAX; source.vertices.len()];
    let mut vertices = Vec::with_capacity(used.len());
    for old in used {
        old_to_new[old] = vertices.len();
        vertices.push(source.vertices[old].clone());
    }
    let mut triangles = Vec::with_capacity(indices.len() / 3);
    for face in 0..indices.len() / 3 {
        let offset = face * 3;
        triangles.push(rotate_triangle([
            old_to_new[indices[offset] as usize],
            old_to_new[indices[offset + 1] as usize],
            old_to_new[indices[offset + 2] as usize],
        ]));
    }
    triangles.sort_unstable();

    let expected_invariants = topology_invariants(source.vertices.len(), &source.triangles);
    let actual_boundary = validate_cut_chart(&mut vertices, &triangles).map_err(|error| {
        let reason = match error {
            CoarseComplexError::InvalidCutChart(reason) => reason,
            _ => "shared chart validation failed",
        };
        CoarseReductionError::InvalidReducedChart {
            chart: source.key.clone(),
            reason,
        }
    })?;
    if topology_invariants(vertices.len(), &triangles) != expected_invariants {
        return Err(CoarseReductionError::InvalidReducedChart {
            chart: source.key.clone(),
            reason: "Euler characteristic or boundary-component count changed",
        });
    }
    let actual_stable = actual_boundary
        .iter()
        .map(|boundary| stable_edge(&vertices, *boundary))
        .collect::<BTreeSet<_>>();
    let expected_stable = source
        .boundary_edges
        .iter()
        .map(|boundary| stable_edge(&source.vertices, *boundary))
        .collect::<BTreeSet<_>>();
    if actual_stable != expected_stable {
        return Err(CoarseReductionError::BoundaryChanged {
            chart: source.key.clone(),
        });
    }
    Ok((vertices, triangles, actual_boundary))
}

fn reduce_chart(
    source: &CutChart,
    config: &CoarseReductionConfig,
) -> Result<ReducedChart, CoarseReductionError> {
    if source.vertices.len() > u32::MAX as usize {
        return Err(CoarseReductionError::IndexOverflow {
            chart: source.key.clone(),
        });
    }
    let mut indices = Vec::with_capacity(
        source
            .triangles
            .len()
            .checked_mul(3)
            .ok_or(CoarseReductionError::CountOverflow)?,
    );
    for triangle in &source.triangles {
        for &vertex in triangle {
            indices.push(u32::try_from(vertex).map_err(|_| {
                CoarseReductionError::IndexOverflow {
                    chart: source.key.clone(),
                }
            })?);
        }
    }
    let source_triangles = source.triangles.len();
    let input_index_count = indices.len();
    let requested = requested_triangles(source_triangles, config.target_ratio)?;
    let (source_vertices, source_faces, source_boundary) = compact_and_validate(source, &indices)?;
    let source_value = (source_vertices, source_faces, source_boundary, None);
    let (selected_target_triangles, selected, backend_attempts, rejected_candidates) = if requested
        < source_triangles
    {
        let positions = normalized_positions(source)?;
        validate_backend_geometry(source, &positions)?;
        let locks = source
            .vertices
            .iter()
            .map(|vertex| vertex.boundary_locked)
            .collect::<Vec<_>>();
        backoff_to_valid(requested, source_triangles, source_value, |target| {
            let target_indices = target
                .checked_mul(3)
                .ok_or(CoarseReductionError::CountOverflow)?;
            let mut result_error = 0.0f32;
            let result = meshopt::simplify_with_locks_decoder(
                &indices,
                &positions.quantized,
                &locks,
                target_indices,
                config.target_error,
                meshopt::SimplifyOptions::LockBorder,
                Some(&mut result_error),
            );
            if !result_error.is_finite() || result_error < 0.0 || result.len() > input_index_count {
                return Err(CoarseReductionError::InvalidMeshoptOutput {
                    chart: source.key.clone(),
                    reason: "the reported error or output size is invalid",
                });
            }
            if result_error > config.target_error {
                return Err(CoarseReductionError::InvalidMeshoptOutput {
                    chart: source.key.clone(),
                    reason: "the reported error exceeds the configured target error",
                });
            }
            validate_meshopt_output_geometry(source, &positions, &result)?;
            let (vertices, triangles, boundary_edges) = compact_and_validate(source, &result)?;
            Ok((vertices, triangles, boundary_edges, Some(result_error)))
        })
    } else {
        (source_triangles, source_value, 0, Vec::new())
    };
    let rejected_candidates = rejected_candidates
        .into_iter()
        .map(|(target, error)| rejection_evidence(target, error))
        .collect::<Vec<_>>();
    let (vertices, triangles, boundary_edges, backend_result_error) = selected;
    let used_source_fallback = requested < source_triangles && backend_result_error.is_none();
    Ok(ReducedChart {
        key: source.key.clone(),
        vertices,
        triangles,
        boundary_edges,
        source_faces: source.source_faces.clone(),
        requested_triangles: requested,
        selected_target_triangles,
        used_source_fallback,
        backend_attempts,
        rejected_candidates,
        backend_result_error,
    })
}

/// Cut authored topology, reduce each chart interior with meshoptimizer, and
/// fail closed unless every stable border and manifold invariant survives.
pub fn reduce_source_charts(
    input: &CoarseComplexInput<'_>,
    config: &CoarseReductionConfig,
) -> Result<ReducedSourceCharts, CoarseReductionError> {
    validate_config(config)?;
    let topology = cut_source_topology(input)?;
    let charts = topology
        .charts
        .iter()
        .map(|chart| reduce_chart(chart, config))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ReducedSourceCharts {
        charts,
        cut_edges: topology.cut_edges,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coarse_complex::{SourceFaceId, SourceVertexId};

    #[test]
    fn deterministic_backoff_finds_the_boundary_for_a_monotone_fixture() {
        let mut attempted = Vec::new();
        let (target, value, attempts, rejections) = backoff_to_valid(15, 60, 60, |candidate| {
            attempted.push(candidate);
            if candidate >= 57 {
                Ok(candidate)
            } else {
                Err(())
            }
        });
        assert_eq!(target, 57);
        assert_eq!(value, 57);
        assert_eq!(attempts, attempted.len());
        assert_eq!(rejections.len(), attempted.len() - 1);
        assert_eq!(rejections.first(), Some(&(15, ())));
        assert_eq!(attempted.first(), Some(&15));
        assert!(attempted.contains(&56));

        let (target, value, attempts, rejections) =
            backoff_to_valid(15, 60, 60, |_| Err::<usize, ()>(()));
        assert_eq!((target, value), (60, 60));
        assert_eq!(attempts, rejections.len());
    }

    #[test]
    fn deterministic_backoff_does_not_claim_an_unprobed_global_minimum() {
        let mut attempted = Vec::new();
        let (target, value, _, _) = backoff_to_valid(15, 60, 60, |candidate| {
            attempted.push(candidate);
            if candidate == 20 || candidate >= 57 {
                Ok(candidate)
            } else {
                Err(())
            }
        });
        assert_eq!((target, value), (57, 57));
        assert!(!attempted.contains(&20));
        assert!(attempted.contains(&57));
    }

    type ReductionSignature = (
        ChartKey,
        Vec<CutVertexKey>,
        Vec<[CutVertexKey; 3]>,
        Vec<[CutVertexKey; 2]>,
        usize,
        usize,
        bool,
        usize,
        Vec<(usize, ReductionRejectionCategory, &'static str)>,
        Option<u32>,
    );

    struct Grid {
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
            vertex_ids: (0..positions.len())
                .map(|index| SourceVertexId(1000 + index as u64 * 17))
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

    fn signature(topology: &ReducedSourceCharts) -> Vec<ReductionSignature> {
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
                    chart
                        .boundary_edges
                        .iter()
                        .map(|boundary| stable_edge(&chart.vertices, *boundary))
                        .collect(),
                    chart.requested_triangles,
                    chart.selected_target_triangles,
                    chart.used_source_fallback,
                    chart.backend_attempts,
                    chart
                        .rejected_candidates
                        .iter()
                        .map(|candidate| {
                            (
                                candidate.target_triangles,
                                candidate.category,
                                candidate.reason,
                            )
                        })
                        .collect(),
                    chart.backend_result_error.map(f32::to_bits),
                )
            })
            .collect()
    }

    fn assert_cut_multiplicity(topology: &ReducedSourceCharts) {
        let mut copies = BTreeMap::<[SourceVertexId; 2], usize>::new();
        for chart in &topology.charts {
            for boundary in &chart.boundary_edges {
                let mut source_edge = [
                    chart.vertices[boundary[0]].key.source_vertex,
                    chart.vertices[boundary[1]].key.source_vertex,
                ];
                source_edge.sort_unstable();
                *copies.entry(source_edge).or_default() += 1;
            }
        }
        for cut in &topology.cut_edges {
            assert_eq!(
                copies.get(&cut.source_vertex_ids).copied().unwrap_or(0),
                if cut.reason.source_boundary { 1 } else { 2 },
            );
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
    fn planar_chart_reduces_without_changing_its_owned_boundary() {
        let source = grid(5);
        let reduced = reduce_source_charts(
            &input(&source),
            &CoarseReductionConfig {
                target_ratio: 0.25,
                target_error: 1.0,
            },
        )
        .unwrap();
        assert_eq!(reduced.charts.len(), 1);
        let chart = &reduced.charts[0];
        assert_eq!(chart.source_faces.len(), 32);
        assert_eq!(chart.requested_triangles, 8);
        assert!(chart.triangles.len() < chart.source_faces.len());
        assert!(
            chart.triangles.len() >= 14,
            "the 16-edge border has a floor"
        );
        assert_eq!(chart.boundary_edges.len(), 16);
        assert_cut_multiplicity(&reduced);
        assert!(chart.vertices.iter().all(|vertex| {
            !vertex.boundary_locked
                || chart.boundary_edges.iter().any(|boundary| {
                    chart.vertices[boundary[0]].key == vertex.key
                        || chart.vertices[boundary[1]].key == vertex.key
                })
        }));

        let reordered_source = reversed_buffers(&source);
        let reordered = reduce_source_charts(
            &input(&reordered_source),
            &CoarseReductionConfig {
                target_ratio: 0.25,
                target_error: 1.0,
            },
        )
        .unwrap();
        assert_eq!(signature(&reduced), signature(&reordered));
        assert_cut_multiplicity(&reordered);

        let mut tiny = grid(5);
        for position in &mut tiny.positions {
            *position = position.map(|component| component * 1.0e-8);
        }
        let tiny_reduced = reduce_source_charts(
            &input(&tiny),
            &CoarseReductionConfig {
                target_ratio: 0.25,
                target_error: 1.0,
            },
        )
        .unwrap();
        assert_eq!(signature(&reduced), signature(&tiny_reduced));
    }

    #[test]
    fn induced_cut_survives_reduction_with_two_complete_border_copies() {
        let positions = [
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
        ];
        let triangles = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
        let vertex_ids = (0..4)
            .map(|index| SourceVertexId(100 + index * 7))
            .collect::<Vec<_>>();
        let face_ids = (0..4)
            .map(|index| SourceFaceId(1_000 + index * 11))
            .collect::<Vec<_>>();
        let reduced = reduce_source_charts(
            &CoarseComplexInput {
                positions: &positions,
                triangles: &triangles,
                source_vertex_ids: &vertex_ids,
                source_face_ids: &face_ids,
                face_domains: &[0; 4],
                locked_edges: &[[0, 1]],
            },
            &CoarseReductionConfig {
                target_ratio: 0.5,
                target_error: 1.0,
            },
        )
        .unwrap();

        assert!(reduced.cut_edges.iter().any(|edge| edge.reason.induced_cut));
        assert_cut_multiplicity(&reduced);
    }

    #[test]
    fn post_validation_rejects_changed_boundary_and_winding() {
        let source = grid(2);
        let topology = cut_source_topology(&input(&source)).unwrap();
        let chart = &topology.charts[0];
        let first_face = chart.triangles[0].map(|vertex| vertex as u32);
        assert!(matches!(
            compact_and_validate(chart, &first_face),
            Err(CoarseReductionError::BoundaryChanged { .. })
        ));

        let mut flipped = chart
            .triangles
            .iter()
            .flat_map(|triangle| triangle.map(|vertex| vertex as u32))
            .collect::<Vec<_>>();
        flipped.swap(1, 2);
        assert!(matches!(
            compact_and_validate(chart, &flipped),
            Err(CoarseReductionError::InvalidReducedChart {
                reason: "an interior edge has inconsistent winding",
                ..
            })
        ));
    }

    #[test]
    fn distinct_stable_vertices_may_not_alias_in_meshopt_f32_space() {
        let mut source = grid(5);
        source.positions[12] = [0.6, 0.5, 0.0];
        source.positions[13] = [0.600_000_000_1, 0.5, 0.0];
        assert!(matches!(
            reduce_source_charts(
                &input(&source),
                &CoarseReductionConfig {
                    target_ratio: 0.5,
                    target_error: 1.0,
                },
            ),
            Err(CoarseReductionError::PositionCollision { left, right, .. })
                if left != right
        ));
    }

    #[test]
    fn exact_position_wedges_are_valid_meshopt_input() {
        let source = grid(2);
        let mut chart = cut_source_topology(&input(&source))
            .unwrap()
            .charts
            .remove(0);
        let mut wedge = chart.vertices[0].clone();
        wedge.key.source_vertex = SourceVertexId(999_999);
        chart.vertices.push(wedge);
        let wedge_index = chart.vertices.len() - 1;
        let replaced = chart.triangles[1]
            .iter_mut()
            .find(|vertex| **vertex == 0)
            .expect("the second grid triangle uses vertex zero");
        *replaced = wedge_index;

        let positions = normalized_positions(&chart).unwrap();
        validate_backend_geometry(&chart, &positions).unwrap();
    }

    #[test]
    fn triangle_may_not_become_collinear_in_meshopt_f32_space() {
        let positions = [
            [0.0, 0.0, 0.0],
            [0.4, 1.0e-50, 0.0],
            [1.0, 0.0, 0.0],
            [0.6, -1.0e-50, 0.0],
        ];
        let triangles = [[0, 2, 1], [0, 3, 2]];
        let vertex_ids = (0..4)
            .map(|index| SourceVertexId(50 + index))
            .collect::<Vec<_>>();
        let face_ids = (0..2)
            .map(|index| SourceFaceId(500 + index))
            .collect::<Vec<_>>();
        assert!(matches!(
            reduce_source_charts(
                &CoarseComplexInput {
                    positions: &positions,
                    triangles: &triangles,
                    source_vertex_ids: &vertex_ids,
                    source_face_ids: &face_ids,
                    face_domains: &[0; 2],
                    locked_edges: &[],
                },
                &CoarseReductionConfig {
                    target_ratio: 0.5,
                    target_error: 1.0,
                },
            ),
            Err(CoarseReductionError::InvalidBackendGeometry {
                reason: "a source triangle becomes degenerate after f32 normalization",
                ..
            })
        ));
    }

    #[test]
    fn meshopt_output_triangle_may_not_become_collinear_in_f32_space() {
        let source = grid(2);
        let mut chart = cut_source_topology(&input(&source))
            .unwrap()
            .charts
            .remove(0);
        chart.vertices[0].position = [0.0, 0.0, 0.0];
        chart.vertices[1].position = [0.4, 1.0e-50, 0.0];
        chart.vertices[2].position = [1.0, 0.0, 0.0];
        chart.vertices[3].position = [1.0, 1.0, 0.0];
        let positions = normalized_positions(&chart).unwrap();
        validate_backend_geometry(&chart, &positions).unwrap();
        assert!(matches!(
            validate_meshopt_output_geometry(&chart, &positions, &[0, 1, 2]),
            Err(CoarseReductionError::InvalidMeshoptOutput {
                reason: "an output triangle becomes degenerate after f32 normalization",
                ..
            })
        ));
    }

    #[test]
    fn non_finite_normalization_extent_fails_without_panicking() {
        let source = grid(2);
        let mut chart = cut_source_topology(&input(&source))
            .unwrap()
            .charts
            .remove(0);
        chart.vertices[0].position[0] = -f64::MAX;
        chart.vertices[1].position[0] = f64::MAX;
        assert!(matches!(
            normalized_positions(&chart),
            Err(CoarseReductionError::InvalidBackendGeometry {
                reason: "its normalization extent is non-finite or non-positive",
                ..
            })
        ));
    }

    #[test]
    fn invalid_policy_fails_before_meshopt() {
        let source = grid(2);
        assert!(matches!(
            reduce_source_charts(
                &input(&source),
                &CoarseReductionConfig {
                    target_ratio: 0.0,
                    ..CoarseReductionConfig::default()
                },
            ),
            Err(CoarseReductionError::InvalidTargetRatio(0.0))
        ));
        assert!(matches!(
            reduce_source_charts(
                &input(&source),
                &CoarseReductionConfig {
                    target_error: f32::NAN,
                    ..CoarseReductionConfig::default()
                },
            ),
            Err(CoarseReductionError::InvalidTargetError(value)) if value.is_nan()
        ));
    }
}
