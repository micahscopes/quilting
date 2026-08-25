//! Backend-neutral attachment and walking on animated conformal surfaces.
//!
//! Persistent addresses stay in source topology as `(entity, face,
//! barycentric)`. A [`SurfaceField`] evaluates that address in the displayed
//! Euclidean output chart, where walking speed, surface velocity, eye height,
//! and conditioning have ordinary metric meaning. The round-side index can
//! implement [`SurfaceField::recover`] without becoming the local traversal
//! path; ordinary source-face adjacency remains the fast, stable path.

use crate::StableEntityId;
use std::collections::BTreeMap;

const FINITE_EPSILON: f64 = 1.0e-12;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceAddress {
    pub entity: StableEntityId,
    pub face: u32,
    pub barycentric: [f64; 3],
}

impl SurfaceAddress {
    pub fn new(
        entity: StableEntityId,
        face: u32,
        barycentric: [f64; 3],
    ) -> Result<Self, SurfaceAddressError> {
        if barycentric.into_iter().any(|value| !value.is_finite()) {
            return Err(SurfaceAddressError::NonFinite);
        }
        let sum = barycentric.into_iter().sum::<f64>();
        if sum <= FINITE_EPSILON || barycentric.into_iter().any(|value| value < -FINITE_EPSILON) {
            return Err(SurfaceAddressError::OutsideFace);
        }
        let mut normalized = barycentric.map(|value| (value / sum).max(0.0));
        let normalized_sum = normalized.into_iter().sum::<f64>();
        if normalized_sum <= FINITE_EPSILON {
            return Err(SurfaceAddressError::OutsideFace);
        }
        normalized = normalized.map(|value| value / normalized_sum);
        Ok(Self {
            entity,
            face,
            barycentric: normalized,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceAddressError {
    NonFinite,
    OutsideFace,
    InvalidNormalSign,
}

/// A posed surface sample in the displayed Euclidean output chart.
///
/// `tangent_u` differentiates along `barycentric[1]` and `tangent_v` along
/// `barycentric[2]`; `barycentric[0] = 1 - u - v`. `surface_velocity` is the
/// output-chart velocity caused by animation and conformal-frame motion while
/// the source address remains fixed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceSample {
    pub output_position: [f64; 3],
    pub tangent_u: [f64; 3],
    pub tangent_v: [f64; 3],
    pub surface_velocity: [f64; 3],
}

impl SurfaceSample {
    pub fn normal(self) -> Option<[f64; 3]> {
        normalize(cross(self.tangent_u, self.tangent_v))
    }

    fn is_finite(self) -> bool {
        [
            self.output_position,
            self.tangent_u,
            self.tangent_v,
            self.surface_velocity,
        ]
        .into_iter()
        .flatten()
        .all(f64::is_finite)
    }
}

/// Geometry/topology adapter used by the deterministic walker.
///
/// A renderer, animation evaluator, or test fixture supplies posed samples.
/// `cross_edge` receives an address exactly on the edge opposite
/// `opposite_corner` and maps its shared-vertex weights to the adjacent face.
/// `recover` is deliberately optional: a round-index-backed broad phase may
/// offer nearby candidates, followed by an exact closest-point solve.
pub trait SurfaceField {
    fn sample(&mut self, address: SurfaceAddress) -> Option<SurfaceSample>;

    fn cross_edge(
        &mut self,
        address_on_edge: SurfaceAddress,
        opposite_corner: usize,
    ) -> Option<SurfaceAddress>;

    fn recover(
        &mut self,
        _entity: StableEntityId,
        _output_position: [f64; 3],
        _maximum_distance: f64,
    ) -> Option<SurfaceAddress> {
        None
    }
}

/// Stable source-topology attachment shared by stationary followers and
/// walkers. Camera height, locomotion, and filtering are deliberately owned by
/// navigation policy rather than this identity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceAttachment {
    pub address: SurfaceAddress,
    /// Which side of the currently sampled oriented surface is retained.
    /// This is always `+1` or `-1` and is explicitly flipped by a runtime when
    /// its output coordinate frame reverses orientation.
    pub normal_sign: i8,
}

impl SurfaceAttachment {
    pub fn new(address: SurfaceAddress) -> Result<Self, SurfaceAddressError> {
        let address = SurfaceAddress::new(address.entity, address.face, address.barycentric)?;
        Ok(Self {
            address,
            normal_sign: 1,
        })
    }

    pub fn with_normal_sign(
        address: SurfaceAddress,
        normal_sign: i8,
    ) -> Result<Self, SurfaceAddressError> {
        let mut attachment = Self::new(address)?;
        if !matches!(normal_sign, -1 | 1) {
            return Err(SurfaceAddressError::InvalidNormalSign);
        }
        attachment.normal_sign = normal_sign;
        Ok(attachment)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceContact {
    pub address: SurfaceAddress,
    pub output_position: [f64; 3],
    pub output_normal: [f64; 3],
    /// Output-chart velocity of the attached material point due to animation
    /// and conformal-frame motion. Locomotion adds its relative tangent intent
    /// to this value before advancing through the surface metric.
    pub surface_velocity: [f64; 3],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceDetachReason {
    Manual,
    InvalidInput,
    SampleUnavailable,
    IllConditioned,
    Boundary,
    IterationLimit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceWalkerStatus {
    Attached,
    Detached(SurfaceDetachReason),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceAdvance {
    pub status: SurfaceWalkerStatus,
    pub contact: Option<SurfaceContact>,
    pub projected_output_velocity: [f64; 3],
    pub condition_number: f64,
    pub substeps: u32,
    pub edge_crossings: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceWalkerConfig {
    /// Bounds integration error when the output Jacobian changes rapidly.
    pub maximum_substep_seconds: f64,
    /// Prevents malformed topology or seam oscillation from hanging a frame.
    pub maximum_iterations: u32,
    pub maximum_edge_crossings: u32,
    /// Largest accepted eigenvalue ratio of `JᵀJ`.
    pub maximum_condition_number: f64,
    /// Scale-relative determinant threshold for `JᵀJ`.
    pub minimum_metric_determinant: f64,
    pub barycentric_epsilon: f64,
}

impl Default for SurfaceWalkerConfig {
    fn default() -> Self {
        Self {
            maximum_substep_seconds: 1.0 / 120.0,
            maximum_iterations: 128,
            maximum_edge_crossings: 16,
            maximum_condition_number: 1.0e8,
            minimum_metric_determinant: 1.0e-14,
            barycentric_epsilon: 1.0e-10,
        }
    }
}

impl SurfaceWalkerConfig {
    pub fn validate(self) -> bool {
        self.maximum_substep_seconds.is_finite()
            && self.maximum_substep_seconds > 0.0
            && self.maximum_iterations > 0
            && self.maximum_edge_crossings > 0
            && self.maximum_condition_number.is_finite()
            && self.maximum_condition_number >= 1.0
            && self.minimum_metric_determinant.is_finite()
            && self.minimum_metric_determinant > 0.0
            && self.barycentric_epsilon.is_finite()
            && self.barycentric_epsilon > 0.0
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SurfaceWalker {
    pub config: SurfaceWalkerConfig,
    attachment: Option<SurfaceAttachment>,
    last_detach_reason: Option<SurfaceDetachReason>,
}

impl SurfaceWalker {
    pub fn attachment(&self) -> Option<SurfaceAttachment> {
        self.attachment
    }

    pub fn last_detach_reason(&self) -> Option<SurfaceDetachReason> {
        self.last_detach_reason
    }

    pub fn attach(&mut self, attachment: SurfaceAttachment) {
        self.attachment = Some(attachment);
        self.last_detach_reason = None;
    }

    /// Preserve the physical attachment side when an output chart changes
    /// orientation (for example when sphere inversion is toggled).
    pub fn flip_normal_side(&mut self) {
        if let Some(attachment) = self.attachment.as_mut() {
            attachment.normal_sign = -attachment.normal_sign;
        }
    }

    pub fn detach(&mut self, reason: SurfaceDetachReason) {
        self.attachment = None;
        self.last_detach_reason = Some(reason);
    }

    /// Ask the provider (eventually the conservative round index plus an exact
    /// projection) for a nearby stable source address.
    pub fn recover<F: SurfaceField>(
        &mut self,
        entity: StableEntityId,
        output_position: [f64; 3],
        maximum_distance: f64,
        field: &mut F,
    ) -> bool {
        if !finite3(output_position)
            || !maximum_distance.is_finite()
            || maximum_distance <= 0.0
        {
            return false;
        }
        let Some(address) = field.recover(entity, output_position, maximum_distance) else {
            return false;
        };
        if SurfaceAddress::new(address.entity, address.face, address.barycentric).is_err() {
            return false;
        }
        if field
            .sample(address)
            .and_then(SurfaceSample::normal)
            .is_none()
        {
            return false;
        }
        self.attach(SurfaceAttachment {
            address,
            normal_sign: 1,
        });
        true
    }

    /// Advance using an absolute velocity in the displayed Euclidean chart.
    /// The sampled surface velocity is subtracted before the Jacobian
    /// pseudoinverse, so callers can stand on or move relative to animation.
    pub fn advance<F: SurfaceField>(
        &mut self,
        delta_seconds: f64,
        desired_output_velocity: [f64; 3],
        field: &mut F,
    ) -> SurfaceAdvance {
        let Some(mut attachment) = self.attachment else {
            return detached_advance(
                self.last_detach_reason
                    .unwrap_or(SurfaceDetachReason::SampleUnavailable),
                0,
                0,
            );
        };
        if !self.config.validate()
            || !delta_seconds.is_finite()
            || delta_seconds < 0.0
            || !finite3(desired_output_velocity)
        {
            self.detach(SurfaceDetachReason::InvalidInput);
            return detached_advance(SurfaceDetachReason::InvalidInput, 0, 0);
        }

        let mut remaining = delta_seconds;
        let mut iterations = 0;
        let mut substeps = 0;
        let mut edge_crossings = 0;
        let mut projected_output_velocity = [0.0; 3];
        let mut condition_number = 1.0;
        let mut integration_velocity = desired_output_velocity;

        while remaining > FINITE_EPSILON {
            iterations += 1;
            if iterations > self.config.maximum_iterations {
                self.detach(SurfaceDetachReason::IterationLimit);
                return detached_advance(
                    SurfaceDetachReason::IterationLimit,
                    substeps,
                    edge_crossings,
                );
            }
            let Some(sample) = field.sample(attachment.address) else {
                self.detach(SurfaceDetachReason::SampleUnavailable);
                return detached_advance(
                    SurfaceDetachReason::SampleUnavailable,
                    substeps,
                    edge_crossings,
                );
            };
            let Some(solution) = solve_output_velocity(sample, integration_velocity, self.config)
            else {
                self.detach(SurfaceDetachReason::IllConditioned);
                return detached_advance(
                    SurfaceDetachReason::IllConditioned,
                    substeps,
                    edge_crossings,
                );
            };
            condition_number = solution.condition_number;
            projected_output_velocity = solution.projected_output_velocity;

            let segment = remaining.min(self.config.maximum_substep_seconds);
            let barycentric_rate = [
                -solution.parameter_velocity[0] - solution.parameter_velocity[1],
                solution.parameter_velocity[0],
                solution.parameter_velocity[1],
            ];
            let crossing = first_edge_crossing(
                attachment.address.barycentric,
                barycentric_rate,
                segment,
                self.config.barycentric_epsilon,
            );
            let elapsed = crossing.map_or(segment, |(_, time)| time.max(0.0));
            attachment.address.barycentric = normalized_barycentric(
                add_scaled(attachment.address.barycentric, barycentric_rate, elapsed),
                self.config.barycentric_epsilon,
            );
            remaining = (remaining - elapsed).max(0.0);
            substeps += u32::from(elapsed > FINITE_EPSILON);

            let Some((opposite_corner, _)) = crossing else {
                continue;
            };
            edge_crossings += 1;
            if edge_crossings > self.config.maximum_edge_crossings {
                self.detach(SurfaceDetachReason::IterationLimit);
                return detached_advance(
                    SurfaceDetachReason::IterationLimit,
                    substeps,
                    edge_crossings,
                );
            }
            attachment.address.barycentric[opposite_corner] = 0.0;
            attachment.address.barycentric = normalized_barycentric(
                attachment.address.barycentric,
                self.config.barycentric_epsilon,
            );
            let Some(next) = field.cross_edge(attachment.address, opposite_corner) else {
                self.detach(SurfaceDetachReason::Boundary);
                return detached_advance(SurfaceDetachReason::Boundary, substeps, edge_crossings);
            };
            if SurfaceAddress::new(next.entity, next.face, next.barycentric).is_err() {
                self.detach(SurfaceDetachReason::SampleUnavailable);
                return detached_advance(
                    SurfaceDetachReason::SampleUnavailable,
                    substeps,
                    edge_crossings,
                );
            }
            let Some(source_edge_sample) = field.sample(attachment.address) else {
                self.detach(SurfaceDetachReason::SampleUnavailable);
                return detached_advance(
                    SurfaceDetachReason::SampleUnavailable,
                    substeps,
                    edge_crossings,
                );
            };
            let Some(source_edge_solution) =
                solve_output_velocity(source_edge_sample, integration_velocity, self.config)
            else {
                self.detach(SurfaceDetachReason::IllConditioned);
                return detached_advance(
                    SurfaceDetachReason::IllConditioned,
                    substeps,
                    edge_crossings,
                );
            };
            let Some(target_edge_sample) = field.sample(next) else {
                self.detach(SurfaceDetachReason::SampleUnavailable);
                return detached_advance(
                    SurfaceDetachReason::SampleUnavailable,
                    substeps,
                    edge_crossings,
                );
            };
            let source_relative_velocity = sub(
                source_edge_solution.projected_output_velocity,
                source_edge_sample.surface_velocity,
            );
            let Some(target_relative_velocity) = transport_tangent_velocity(
                source_relative_velocity,
                source_edge_sample.normal(),
                target_edge_sample.normal(),
            ) else {
                self.detach(SurfaceDetachReason::IllConditioned);
                return detached_advance(
                    SurfaceDetachReason::IllConditioned,
                    substeps,
                    edge_crossings,
                );
            };
            integration_velocity = add(
                target_relative_velocity,
                target_edge_sample.surface_velocity,
            );
            attachment.address = next;
        }

        self.attachment = Some(attachment);
        let contact = field
            .sample(attachment.address)
            .and_then(|sample| contact_from_sample(attachment, sample));
        if contact.is_none() {
            self.detach(SurfaceDetachReason::SampleUnavailable);
            return detached_advance(
                SurfaceDetachReason::SampleUnavailable,
                substeps,
                edge_crossings,
            );
        }
        SurfaceAdvance {
            status: SurfaceWalkerStatus::Attached,
            contact,
            projected_output_velocity,
            condition_number,
            substeps,
            edge_crossings,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct VelocitySolution {
    parameter_velocity: [f64; 2],
    projected_output_velocity: [f64; 3],
    condition_number: f64,
}

fn solve_output_velocity(
    sample: SurfaceSample,
    desired_output_velocity: [f64; 3],
    config: SurfaceWalkerConfig,
) -> Option<VelocitySolution> {
    if !sample.is_finite() {
        return None;
    }
    let a = dot(sample.tangent_u, sample.tangent_u);
    let b = dot(sample.tangent_u, sample.tangent_v);
    let c = dot(sample.tangent_v, sample.tangent_v);
    let trace = a + c;
    let discriminant = ((a - c) * (a - c) + 4.0 * b * b).sqrt();
    let largest = 0.5 * (trace + discriminant);
    let smallest = 0.5 * (trace - discriminant);
    let determinant = a * c - b * b;
    if !largest.is_finite()
        || !smallest.is_finite()
        || smallest <= FINITE_EPSILON
        || determinant <= config.minimum_metric_determinant * trace.mul_add(trace, 1.0)
    {
        return None;
    }
    let condition_number = largest / smallest;
    if !condition_number.is_finite() || condition_number > config.maximum_condition_number {
        return None;
    }

    let relative_velocity = sub(desired_output_velocity, sample.surface_velocity);
    let rhs_u = dot(sample.tangent_u, relative_velocity);
    let rhs_v = dot(sample.tangent_v, relative_velocity);
    let du = (c * rhs_u - b * rhs_v) / determinant;
    let dv = (a * rhs_v - b * rhs_u) / determinant;
    if !du.is_finite() || !dv.is_finite() {
        return None;
    }
    let relative_projected = add(scale(sample.tangent_u, du), scale(sample.tangent_v, dv));
    Some(VelocitySolution {
        parameter_velocity: [du, dv],
        projected_output_velocity: add(relative_projected, sample.surface_velocity),
        condition_number,
    })
}

/// Parallel-transport a tangent direction through the shortest rotation that
/// aligns the source and target normals. For a folded triangle pair this is
/// exactly the intrinsic "unfold around the shared edge" continuation; it also
/// prevents a global output-space direction from bouncing at zero elapsed time
/// between the same two faces.
fn transport_tangent_velocity(
    velocity: [f64; 3],
    source_normal: Option<[f64; 3]>,
    target_normal: Option<[f64; 3]>,
) -> Option<[f64; 3]> {
    let source_normal = source_normal?;
    let target_normal = target_normal?;
    let cosine = dot(source_normal, target_normal).clamp(-1.0, 1.0);
    let axis = cross(source_normal, target_normal);
    let sine = dot(axis, axis).sqrt();
    let rotated = if sine > FINITE_EPSILON {
        let axis = scale(axis, 1.0 / sine);
        add(
            add(scale(velocity, cosine), scale(cross(axis, velocity), sine)),
            scale(axis, dot(axis, velocity) * (1.0 - cosine)),
        )
    } else if cosine > 0.0 {
        velocity
    } else {
        // Antiparallel normals do not define a unique shortest rotation.
        return None;
    };
    let tangent = sub(rotated, scale(target_normal, dot(rotated, target_normal)));
    finite3(tangent).then_some(tangent)
}

fn first_edge_crossing(
    barycentric: [f64; 3],
    rate: [f64; 3],
    duration: f64,
    epsilon: f64,
) -> Option<(usize, f64)> {
    let mut first: Option<(usize, f64)> = None;
    for corner in 0..3 {
        if rate[corner] >= -epsilon || barycentric[corner] + rate[corner] * duration >= -epsilon {
            continue;
        }
        let time = (barycentric[corner] / -rate[corner]).max(0.0);
        if time <= duration + epsilon && first.is_none_or(|(_, prior)| time < prior) {
            first = Some((corner, time));
        }
    }
    first
}

fn contact_from_sample(
    attachment: SurfaceAttachment,
    sample: SurfaceSample,
) -> Option<SurfaceContact> {
    let output_normal = scale(sample.normal()?, attachment.normal_sign as f64);
    Some(SurfaceContact {
        address: attachment.address,
        output_position: sample.output_position,
        output_normal,
        surface_velocity: sample.surface_velocity,
    })
}

fn detached_advance(
    reason: SurfaceDetachReason,
    substeps: u32,
    edge_crossings: u32,
) -> SurfaceAdvance {
    SurfaceAdvance {
        status: SurfaceWalkerStatus::Detached(reason),
        contact: None,
        projected_output_velocity: [0.0; 3],
        condition_number: f64::INFINITY,
        substeps,
        edge_crossings,
    }
}

/// Immutable source-triangle adjacency with deterministic shared-weight
/// transfer. Boundary and non-manifold edges deliberately have no neighbor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TriangleAdjacency {
    faces: Vec<[u64; 3]>,
    neighbors: Vec<[Option<u32>; 3]>,
    report: TriangleAdjacencyReport,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TriangleAdjacencyReport {
    pub faces: usize,
    pub manifold_edges: usize,
    pub boundary_edges: usize,
    pub non_manifold_edges: usize,
}

impl TriangleAdjacency {
    pub fn new(faces: Vec<[u64; 3]>) -> Self {
        let mut incidence = BTreeMap::<(u64, u64), Vec<(usize, usize)>>::new();
        for (face, vertices) in faces.iter().enumerate() {
            for opposite in 0..3 {
                let mut edge = [vertices[(opposite + 1) % 3], vertices[(opposite + 2) % 3]];
                edge.sort_unstable();
                incidence
                    .entry((edge[0], edge[1]))
                    .or_default()
                    .push((face, opposite));
            }
        }
        let mut neighbors = vec![[None; 3]; faces.len()];
        let mut report = TriangleAdjacencyReport {
            faces: faces.len(),
            ..TriangleAdjacencyReport::default()
        };
        for incidents in incidence.values() {
            match incidents.as_slice() {
                [(left_face, left_corner), (right_face, right_corner)] => {
                    neighbors[*left_face][*left_corner] = Some(*right_face as u32);
                    neighbors[*right_face][*right_corner] = Some(*left_face as u32);
                    report.manifold_edges += 1;
                }
                [_] => report.boundary_edges += 1,
                _ => report.non_manifold_edges += 1,
            }
        }
        Self {
            faces,
            neighbors,
            report,
        }
    }

    pub fn report(&self) -> TriangleAdjacencyReport {
        self.report
    }

    pub fn cross_edge(
        &self,
        address_on_edge: SurfaceAddress,
        opposite_corner: usize,
    ) -> Option<SurfaceAddress> {
        let source_face = *self.faces.get(address_on_edge.face as usize)?;
        let target_face_index = self
            .neighbors
            .get(address_on_edge.face as usize)?
            .get(opposite_corner)
            .copied()
            .flatten()?;
        let target_face = *self.faces.get(target_face_index as usize)?;
        let mut target_barycentric = [0.0; 3];
        for (source_corner, vertex) in source_face.into_iter().enumerate() {
            if source_corner == opposite_corner {
                continue;
            }
            let target_corner = target_face
                .iter()
                .position(|candidate| *candidate == vertex)?;
            target_barycentric[target_corner] = address_on_edge.barycentric[source_corner];
        }
        SurfaceAddress::new(
            address_on_edge.entity,
            target_face_index,
            target_barycentric,
        )
        .ok()
    }
}

fn finite3(value: [f64; 3]) -> bool {
    value.into_iter().all(f64::is_finite)
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn normalize(value: [f64; 3]) -> Option<[f64; 3]> {
    let length = dot(value, value).sqrt();
    (length > FINITE_EPSILON && length.is_finite()).then(|| scale(value, 1.0 / length))
}

fn add(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] + right[0], left[1] + right[1], left[2] + right[2]]
}

fn sub(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn scale(value: [f64; 3], factor: f64) -> [f64; 3] {
    [value[0] * factor, value[1] * factor, value[2] * factor]
}

fn add_scaled(value: [f64; 3], rate: [f64; 3], time: f64) -> [f64; 3] {
    [
        value[0] + rate[0] * time,
        value[1] + rate[1] * time,
        value[2] + rate[2] * time,
    ]
}

fn normalized_barycentric(mut value: [f64; 3], epsilon: f64) -> [f64; 3] {
    for coordinate in &mut value {
        if coordinate.abs() <= epsilon {
            *coordinate = 0.0;
        }
    }
    let sum = value.into_iter().sum::<f64>();
    if sum.abs() > FINITE_EPSILON {
        value.map(|coordinate| coordinate / sum)
    } else {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    struct PlanarField {
        positions: Vec<[f64; 3]>,
        faces: Vec<[usize; 3]>,
        adjacency: TriangleAdjacency,
        surface_velocity: [f64; 3],
    }

    impl PlanarField {
        fn sample_face(&self, address: SurfaceAddress) -> Option<SurfaceSample> {
            let face = *self.faces.get(address.face as usize)?;
            let p0 = self.positions[face[0]];
            let p1 = self.positions[face[1]];
            let p2 = self.positions[face[2]];
            Some(SurfaceSample {
                output_position: add(
                    add(
                        scale(p0, address.barycentric[0]),
                        scale(p1, address.barycentric[1]),
                    ),
                    scale(p2, address.barycentric[2]),
                ),
                tangent_u: sub(p1, p0),
                tangent_v: sub(p2, p0),
                surface_velocity: self.surface_velocity,
            })
        }
    }

    impl SurfaceField for PlanarField {
        fn sample(&mut self, address: SurfaceAddress) -> Option<SurfaceSample> {
            self.sample_face(address)
        }

        fn cross_edge(
            &mut self,
            address_on_edge: SurfaceAddress,
            opposite_corner: usize,
        ) -> Option<SurfaceAddress> {
            self.adjacency.cross_edge(address_on_edge, opposite_corner)
        }
    }

    fn entity() -> StableEntityId {
        StableEntityId(Uuid::nil())
    }

    fn field(scale_factor: f64) -> PlanarField {
        let positions = vec![
            [0.0, 0.0, 0.0],
            [scale_factor, 0.0, 0.0],
            [0.0, scale_factor, 0.0],
            [scale_factor, scale_factor, 0.0],
        ];
        let faces = vec![[0, 1, 2], [1, 3, 2]];
        PlanarField {
            positions,
            faces,
            adjacency: TriangleAdjacency::new(vec![[0, 1, 2], [1, 3, 2]]),
            surface_velocity: [0.0; 3],
        }
    }

    fn walker_at(barycentric: [f64; 3]) -> SurfaceWalker {
        let address = SurfaceAddress::new(entity(), 0, barycentric).unwrap();
        let mut walker = SurfaceWalker {
            config: SurfaceWalkerConfig {
                maximum_substep_seconds: 1.0,
                ..SurfaceWalkerConfig::default()
            },
            ..SurfaceWalker::default()
        };
        walker.attach(SurfaceAttachment::new(address).unwrap());
        walker
    }

    #[test]
    fn output_velocity_integrates_source_barycentrics() {
        let mut field = field(1.0);
        let mut walker = walker_at([0.6, 0.2, 0.2]);
        let step = walker.advance(1.0, [0.1, 0.0, 0.0], &mut field);
        assert_eq!(step.status, SurfaceWalkerStatus::Attached);
        let contact = step.contact.unwrap();
        assert_eq!(contact.address.face, 0);
        assert!((contact.address.barycentric[1] - 0.3).abs() < 1.0e-12);
        assert!((contact.output_position[0] - 0.3).abs() < 1.0e-12);
    }

    #[test]
    fn attachment_normal_sign_selects_and_flips_the_surface_side() {
        let address = SurfaceAddress::new(entity(), 0, [0.6, 0.2, 0.2]).unwrap();
        let mut walker = SurfaceWalker {
            config: SurfaceWalkerConfig {
                maximum_substep_seconds: 1.0,
                ..SurfaceWalkerConfig::default()
            },
            ..SurfaceWalker::default()
        };
        walker.attach(SurfaceAttachment::with_normal_sign(address, -1).unwrap());
        let mut field = field(1.0);
        let below = walker.advance(0.0, [0.0; 3], &mut field).contact.unwrap();
        assert!(below.output_normal[2] < -0.999);

        walker.flip_normal_side();
        let above = walker.advance(0.0, [0.0; 3], &mut field).contact.unwrap();
        assert!(above.output_normal[2] > 0.999);
    }

    #[test]
    fn manifold_edge_crossing_preserves_shared_source_weights() {
        let mut field = field(1.0);
        let mut walker = walker_at([0.1, 0.8, 0.1]);
        let step = walker.advance(1.0, [0.15, 0.0, 0.0], &mut field);
        assert_eq!(step.status, SurfaceWalkerStatus::Attached);
        assert_eq!(step.edge_crossings, 1);
        let contact = step.contact.unwrap();
        assert_eq!(contact.address.face, 1);
        assert!((contact.output_position[0] - 0.95).abs() < 1.0e-10);
        assert!((contact.output_position[1] - 0.1).abs() < 1.0e-10);
    }

    #[test]
    fn edge_crossing_unfolds_velocity_onto_a_sharply_folded_neighbor() {
        let mut field = field(1.0);
        // Fold the neighboring triangle almost all the way back over the
        // source triangle. Reusing the source-chart velocity on the target
        // would project immediately back across the same edge forever.
        field.positions[3] = [0.0, 0.0, 0.1];
        let mut walker = walker_at([0.1, 0.8, 0.1]);
        let step = walker.advance(1.0, [0.15, 0.0, 0.0], &mut field);
        assert_eq!(step.status, SurfaceWalkerStatus::Attached);
        assert_eq!(step.edge_crossings, 1);
        let contact = step.contact.unwrap();
        assert_eq!(contact.address.face, 1);
        assert!(contact.address.barycentric[1] > 0.0);
    }

    #[test]
    fn animation_velocity_is_removed_before_parameter_integration() {
        let mut field = field(1.0);
        field.surface_velocity = [0.4, -0.2, 0.0];
        let mut walker = walker_at([0.6, 0.2, 0.2]);
        let before = walker.attachment().unwrap().address.barycentric;
        let step = walker.advance(0.5, field.surface_velocity, &mut field);
        assert_eq!(step.status, SurfaceWalkerStatus::Attached);
        assert_eq!(walker.attachment().unwrap().address.barycentric, before);
        assert_eq!(step.projected_output_velocity, field.surface_velocity);
    }

    #[test]
    fn output_chart_scale_changes_parameter_rate_not_physical_speed() {
        let mut field = field(10.0);
        let mut walker = walker_at([0.6, 0.2, 0.2]);
        let step = walker.advance(1.0, [1.0, 0.0, 0.0], &mut field);
        let contact = step.contact.unwrap();
        assert!((contact.address.barycentric[1] - 0.3).abs() < 1.0e-12);
        assert!((contact.output_position[0] - 3.0).abs() < 1.0e-12);
        assert!((step.projected_output_velocity[0] - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn boundary_exit_detaches_instead_of_teleporting() {
        let mut field = field(1.0);
        field.faces.truncate(1);
        field.adjacency = TriangleAdjacency::new(vec![[0, 1, 2]]);
        let mut walker = walker_at([0.1, 0.8, 0.1]);
        let step = walker.advance(1.0, [0.5, 0.0, 0.0], &mut field);
        assert_eq!(
            step.status,
            SurfaceWalkerStatus::Detached(SurfaceDetachReason::Boundary)
        );
        assert!(walker.attachment().is_none());
    }

    #[test]
    fn singular_surface_detaches_explicitly() {
        let mut field = field(1.0);
        field.positions[2] = [2.0, 0.0, 0.0];
        let mut walker = walker_at([0.6, 0.2, 0.2]);
        let step = walker.advance(0.1, [0.1, 0.0, 0.0], &mut field);
        assert_eq!(
            step.status,
            SurfaceWalkerStatus::Detached(SurfaceDetachReason::IllConditioned)
        );
    }

    #[test]
    fn non_manifold_edges_are_not_crossable() {
        let adjacency = TriangleAdjacency::new(vec![[0, 1, 2], [1, 0, 3], [0, 1, 4]]);
        assert_eq!(adjacency.report().non_manifold_edges, 1);
        let address = SurfaceAddress::new(entity(), 0, [0.0, 0.5, 0.5]).unwrap();
        // Face 0's corner 2 is opposite edge (0,1).
        assert!(adjacency.cross_edge(address, 2).is_none());
    }
}
