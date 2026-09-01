//! Backend-neutral pointer interaction above renderer-specific picking.
//!
//! A browser, renderer, XR adapter, or replay supplies validated hits. This
//! module owns hover/press semantics and emits activations; selection remains
//! authoritative in [`FocusNavigation`](crate::FocusNavigation) through the
//! existing [`NavigationAction`](crate::NavigationAction) queue.

use crate::{
    FocusNavigation, FocusSphere, HyperscapeDiagnostics, NavigationAction, NavigationActionQueue,
    TransitionEasing,
};
use bevy_ecs::prelude::{Res, ResMut, Resource};
use bevy_time::{Time, Virtual};
use hyperscape_protocol::AssetEntityId;
use quilting_core::render_evidence::{RenderPickEvidenceError, RenderPickEvidenceReport};
use serde::Serialize;
use std::collections::{BTreeMap, VecDeque};
use std::error::Error;
use std::fmt;

const BARYCENTRIC_EPSILON: f64 = 1.0e-9;

/// Optional source-topology detail retained by an interaction hit.
///
/// The asset/entity identity is inherited from [`InteractionHit`], preventing
/// a face address from silently naming a different object.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InteractionSurfacePoint {
    pub face: u32,
    pub barycentric: [f64; 3],
}

impl InteractionSurfacePoint {
    pub fn new(face: u32, barycentric: [f64; 3]) -> Result<Self, &'static str> {
        if barycentric.into_iter().any(|value| !value.is_finite()) {
            return Err("interaction barycentric coordinates must be finite");
        }
        let sum = barycentric.into_iter().sum::<f64>();
        if sum <= BARYCENTRIC_EPSILON
            || barycentric
                .into_iter()
                .any(|value| value < -BARYCENTRIC_EPSILON)
        {
            return Err("interaction barycentric coordinates lie outside the face");
        }
        let mut normalized = barycentric.map(|value| (value / sum).max(0.0));
        let normalized_sum = normalized.into_iter().sum::<f64>();
        normalized = normalized.map(|value| value / normalized_sum);
        Ok(Self {
            face,
            barycentric: normalized,
        })
    }

    pub fn validate(self) -> Result<(), &'static str> {
        if self.barycentric.into_iter().any(|value| !value.is_finite())
            || self
                .barycentric
                .into_iter()
                .any(|value| value < -BARYCENTRIC_EPSILON)
            || (self.barycentric.into_iter().sum::<f64>() - 1.0).abs() > BARYCENTRIC_EPSILON
        {
            return Err("interaction surface point is not normalized inside its face");
        }
        Ok(())
    }
}

/// One exact renderer/query result in both source and displayed charts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InteractionHit {
    pub identity: AssetEntityId,
    pub source_bound: FocusSphere,
    pub source_pivot: [f64; 3],
    /// Camera-to-hit distance in the displayed Euclidean output chart.
    pub output_distance: f64,
    pub surface: Option<InteractionSurfacePoint>,
}

impl InteractionHit {
    pub fn new(
        identity: AssetEntityId,
        source_bound: FocusSphere,
        source_pivot: [f64; 3],
        output_distance: f64,
    ) -> Result<Self, &'static str> {
        let hit = Self {
            identity,
            source_bound,
            source_pivot,
            output_distance,
            surface: None,
        };
        hit.validate()?;
        Ok(hit)
    }

    pub fn with_surface(mut self, face: u32, barycentric: [f64; 3]) -> Result<Self, &'static str> {
        self.surface = Some(InteractionSurfacePoint::new(face, barycentric)?);
        Ok(self)
    }

    pub fn validate(self) -> Result<(), &'static str> {
        self.identity
            .validate()
            .map_err(|_| "interaction hit must have non-nil asset and entity identities")?;
        FocusSphere::new(self.source_bound.center, self.source_bound.radius)?;
        if self
            .source_pivot
            .into_iter()
            .any(|value| !value.is_finite())
        {
            return Err("interaction source pivot must be finite");
        }
        if !self.output_distance.is_finite() || self.output_distance < 0.0 {
            return Err("interaction output distance must be finite and nonnegative");
        }
        if let Some(surface) = self.surface {
            surface.validate()?;
        }
        Ok(())
    }
}

/// One renderer-resident node joined to source geometry and optional stable
/// identity. `packed_node` is process-local and must never escape as semantic
/// selection identity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InteractionTarget {
    pub packed_node: u32,
    pub identity: Option<AssetEntityId>,
    pub source_bound: FocusSphere,
}

impl InteractionTarget {
    pub fn new(
        packed_node: u32,
        identity: Option<AssetEntityId>,
        source_bound: FocusSphere,
    ) -> Result<Self, InteractionTargetError> {
        if identity.is_some_and(|identity| identity.validate().is_err()) {
            return Err(InteractionTargetError::InvalidTarget {
                packed_node,
                reason: "interaction target identity must be non-nil",
            });
        }
        FocusSphere::new(source_bound.center, source_bound.radius).map_err(|reason| {
            InteractionTargetError::InvalidTarget {
                packed_node,
                reason,
            }
        })?;
        Ok(Self {
            packed_node,
            identity,
            source_bound,
        })
    }
}

/// Backend-local result after a renderer has resolved its ray/depth query.
/// Stable identity and source bounds are deliberately absent; the current
/// residency table supplies them atomically.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InteractionTargetSample {
    pub target_epoch: u32,
    pub packed_node: u32,
    pub source_pivot: [f64; 3],
    pub output_distance: f64,
    pub surface: Option<InteractionSurfacePoint>,
}

impl InteractionTargetSample {
    pub fn new(
        packed_node: u32,
        source_pivot: [f64; 3],
        output_distance: f64,
    ) -> Result<Self, InteractionTargetError> {
        Self::new_for_epoch(0, packed_node, source_pivot, output_distance)
    }

    pub fn new_for_epoch(
        target_epoch: u32,
        packed_node: u32,
        source_pivot: [f64; 3],
        output_distance: f64,
    ) -> Result<Self, InteractionTargetError> {
        let sample = Self {
            target_epoch,
            packed_node,
            source_pivot,
            output_distance,
            surface: None,
        };
        sample.validate()?;
        Ok(sample)
    }

    pub fn with_surface(
        mut self,
        face: u32,
        barycentric: [f64; 3],
    ) -> Result<Self, InteractionTargetError> {
        self.surface = Some(
            InteractionSurfacePoint::new(face, barycentric)
                .map_err(InteractionTargetError::InvalidSample)?,
        );
        Ok(self)
    }

    fn validate(self) -> Result<(), InteractionTargetError> {
        if self
            .source_pivot
            .into_iter()
            .any(|coordinate| !coordinate.is_finite())
        {
            return Err(InteractionTargetError::InvalidSample(
                "interaction source pivot must be finite",
            ));
        }
        if !self.output_distance.is_finite() || self.output_distance < 0.0 {
            return Err(InteractionTargetError::InvalidSample(
                "interaction output distance must be finite and nonnegative",
            ));
        }
        if let Some(surface) = self.surface {
            surface
                .validate()
                .map_err(InteractionTargetError::InvalidSample)?;
        }
        Ok(())
    }
}

/// Atomic backend-neutral join from transient renderer handles to semantic
/// interaction targets. Replacing the table is a residency operation, not an
/// authored/application mutation.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct InteractionTargetTable {
    epoch: u32,
    targets: BTreeMap<u32, InteractionTarget>,
}

impl InteractionTargetTable {
    pub fn try_from_targets(
        targets: impl IntoIterator<Item = InteractionTarget>,
    ) -> Result<Self, InteractionTargetError> {
        Self::try_from_epoch(0, targets)
    }

    pub fn try_from_epoch(
        epoch: u32,
        targets: impl IntoIterator<Item = InteractionTarget>,
    ) -> Result<Self, InteractionTargetError> {
        let mut resolved = BTreeMap::new();
        for target in targets {
            let target = InteractionTarget::new(
                target.packed_node,
                target.identity,
                target.source_bound,
            )?;
            if resolved.insert(target.packed_node, target).is_some() {
                return Err(InteractionTargetError::DuplicatePackedNode(
                    target.packed_node,
                ));
            }
        }
        Ok(Self {
            epoch,
            targets: resolved,
        })
    }

    pub fn epoch(&self) -> u32 {
        self.epoch
    }

    pub fn len(&self) -> usize {
        self.targets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }

    pub fn resolve(
        &self,
        sample: InteractionTargetSample,
    ) -> Result<InteractionHit, InteractionTargetError> {
        if sample.target_epoch != self.epoch {
            return Err(InteractionTargetError::StaleTargetEpoch {
                expected: self.epoch,
                actual: sample.target_epoch,
            });
        }
        sample.validate()?;
        let target = self
            .targets
            .get(&sample.packed_node)
            .ok_or(InteractionTargetError::UnknownPackedNode(sample.packed_node))?;
        let identity = target
            .identity
            .ok_or(InteractionTargetError::UnmappedPackedNode(sample.packed_node))?;
        let mut hit = InteractionHit::new(
            identity,
            target.source_bound,
            sample.source_pivot,
            sample.output_distance,
        )
        .map_err(InteractionTargetError::InvalidSample)?;
        if let Some(surface) = sample.surface {
            hit = hit
                .with_surface(surface.face, surface.barycentric)
                .map_err(InteractionTargetError::InvalidSample)?;
        }
        Ok(hit)
    }
}

/// Identity of one asynchronous renderer pick. The target epoch joins packed
/// renderer handles to the exact semantic table observed when the request was
/// staged; the monotonic request ID rejects out-of-order pointer completions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InteractionPickRequest {
    pub request_id: u64,
    pub target_epoch: u32,
}

/// Current process-local state of the authoritative asynchronous pick lane.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum InteractionPickAuthorityState {
    #[default]
    Idle,
    Staging,
    Reading,
    Accepted,
    StageRejected,
    StaleTargetEpoch,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InteractionPickAuthorityDisposition {
    Current,
    IgnoredSuperseded,
    IgnoredStaleTargetEpoch,
}

/// Bounded lifecycle telemetry. Individual hits remain in the semantic
/// interaction reducer or the caller; this observer retains no per-click
/// payload and cannot grow with session duration.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InteractionPickAuthorityDiagnostics {
    pub state: InteractionPickAuthorityState,
    pub requests: u64,
    pub staged: u64,
    pub stage_rejects: u64,
    pub readbacks: u64,
    pub accepted: u64,
    pub superseded_requests: u64,
    pub stale_completions: u64,
    pub stale_target_epochs: u64,
    pub errors: u64,
    pub latest_request_id: Option<u64>,
    pub latest_target_epoch: Option<u32>,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct InteractionPickAuthority {
    next_request_id: u64,
    latest: Option<InteractionPickRequest>,
    diagnostics: InteractionPickAuthorityDiagnostics,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InteractionPickAuthorityError {
    RequestIdExhausted,
}

impl fmt::Display for InteractionPickAuthorityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RequestIdExhausted => {
                formatter.write_str("interaction pick request ID exhausted")
            }
        }
    }
}

impl Error for InteractionPickAuthorityError {}

impl InteractionPickAuthority {
    pub fn snapshot(&self) -> InteractionPickAuthorityDiagnostics {
        self.diagnostics.clone()
    }

    pub fn begin(
        &mut self,
        target_epoch: u32,
    ) -> Result<InteractionPickRequest, InteractionPickAuthorityError> {
        let request_id = self
            .next_request_id
            .checked_add(1)
            .ok_or(InteractionPickAuthorityError::RequestIdExhausted)?;
        self.next_request_id = request_id;
        if self.latest.is_some() {
            self.diagnostics.superseded_requests =
                self.diagnostics.superseded_requests.saturating_add(1);
        }
        let request = InteractionPickRequest {
            request_id,
            target_epoch,
        };
        self.latest = Some(request);
        self.diagnostics.requests = self.diagnostics.requests.saturating_add(1);
        self.diagnostics.state = InteractionPickAuthorityState::Staging;
        self.diagnostics.latest_request_id = Some(request_id);
        self.diagnostics.latest_target_epoch = Some(target_epoch);
        self.diagnostics.last_error = None;
        Ok(request)
    }

    pub fn record_stage(
        &mut self,
        request: InteractionPickRequest,
        rejection: Option<String>,
    ) -> InteractionPickAuthorityDisposition {
        if self.latest != Some(request) {
            self.record_stale_completion();
            return InteractionPickAuthorityDisposition::IgnoredSuperseded;
        }
        if let Some(error) = rejection {
            self.diagnostics.stage_rejects = self.diagnostics.stage_rejects.saturating_add(1);
            self.diagnostics.state = InteractionPickAuthorityState::StageRejected;
            self.diagnostics.last_error = Some(error);
            self.clear_latest();
        } else {
            self.diagnostics.staged = self.diagnostics.staged.saturating_add(1);
            self.diagnostics.state = InteractionPickAuthorityState::Reading;
        }
        InteractionPickAuthorityDisposition::Current
    }

    pub fn complete(
        &mut self,
        request: InteractionPickRequest,
        current_target_epoch: u32,
    ) -> InteractionPickAuthorityDisposition {
        if self.latest != Some(request) {
            self.record_stale_completion();
            return InteractionPickAuthorityDisposition::IgnoredSuperseded;
        }
        self.diagnostics.readbacks = self.diagnostics.readbacks.saturating_add(1);
        self.diagnostics.last_error = None;
        if request.target_epoch != current_target_epoch {
            self.diagnostics.stale_target_epochs =
                self.diagnostics.stale_target_epochs.saturating_add(1);
            self.diagnostics.state = InteractionPickAuthorityState::StaleTargetEpoch;
            self.clear_latest();
            return InteractionPickAuthorityDisposition::IgnoredStaleTargetEpoch;
        }
        self.diagnostics.accepted = self.diagnostics.accepted.saturating_add(1);
        self.diagnostics.state = InteractionPickAuthorityState::Accepted;
        self.clear_latest();
        InteractionPickAuthorityDisposition::Current
    }

    pub fn record_error(
        &mut self,
        request: InteractionPickRequest,
        error: impl Into<String>,
    ) -> InteractionPickAuthorityDisposition {
        if self.latest != Some(request) {
            self.record_stale_completion();
            return InteractionPickAuthorityDisposition::IgnoredSuperseded;
        }
        self.diagnostics.errors = self.diagnostics.errors.saturating_add(1);
        self.diagnostics.state = InteractionPickAuthorityState::Error;
        self.diagnostics.last_error = Some(error.into());
        self.clear_latest();
        InteractionPickAuthorityDisposition::Current
    }

    fn record_stale_completion(&mut self) {
        self.diagnostics.stale_completions = self.diagnostics.stale_completions.saturating_add(1);
    }

    fn clear_latest(&mut self) {
        self.latest = None;
        self.diagnostics.latest_request_id = None;
        self.diagnostics.latest_target_epoch = None;
    }
}

/// Current state of the opt-in retained-renderer pick comparison lane. This is
/// process-local evidence, not authored scene or interaction state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum InteractionPickEvidenceState {
    #[default]
    AwaitingRetainedFrame,
    Reading,
    StageRejected,
    Shadowing,
    StaleTargetEpoch,
    Mismatch,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InteractionPickEvidenceDisposition {
    Recorded,
    IgnoredStale,
}

/// Bounded scalar telemetry for backend pick parity. Only the last report and
/// error are retained; the observer never accumulates per-click samples.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InteractionPickEvidenceDiagnostics {
    pub state: InteractionPickEvidenceState,
    pub requests: u64,
    pub staged: u64,
    pub stage_rejects: u64,
    pub readbacks: u64,
    pub stale_results: u64,
    pub coverage_mismatches: u64,
    pub identity_mismatches: u64,
    pub maximum_barycentric_error: f32,
    pub maximum_source_position_error: f32,
    pub maximum_output_distance_error: f32,
    pub maximum_staging_ms: f64,
    pub maximum_readback_ms: f64,
    pub maximum_total_ms: f64,
    pub errors: u64,
    pub last_error: Option<String>,
    pub last_report: Option<RenderPickEvidenceReport>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct InteractionPickEvidenceObserver {
    diagnostics: InteractionPickEvidenceDiagnostics,
}

impl InteractionPickEvidenceObserver {
    pub fn snapshot(&self) -> InteractionPickEvidenceDiagnostics {
        self.diagnostics.clone()
    }

    pub fn record_stage(&mut self, staged: bool, rejection: Option<String>) {
        self.diagnostics.requests = self.diagnostics.requests.saturating_add(1);
        if staged {
            self.diagnostics.staged = self.diagnostics.staged.saturating_add(1);
            self.diagnostics.state = InteractionPickEvidenceState::Reading;
            self.diagnostics.last_error = None;
        } else {
            self.diagnostics.stage_rejects =
                self.diagnostics.stage_rejects.saturating_add(1);
            self.diagnostics.state = InteractionPickEvidenceState::StageRejected;
            self.diagnostics.last_error = rejection;
        }
    }

    pub fn record_error(&mut self, error: impl Into<String>) {
        self.diagnostics.errors = self.diagnostics.errors.saturating_add(1);
        self.diagnostics.state = InteractionPickEvidenceState::Error;
        self.diagnostics.last_error = Some(error.into());
    }

    pub fn record_report(
        &mut self,
        targets: &InteractionTargetTable,
        report: RenderPickEvidenceReport,
    ) -> Result<InteractionPickEvidenceDisposition, RenderPickEvidenceError> {
        report.validate()?;
        self.diagnostics.readbacks = self.diagnostics.readbacks.saturating_add(1);
        self.diagnostics.last_report = Some(report);
        self.diagnostics.last_error = None;
        if report.target_epoch != targets.epoch() {
            self.diagnostics.stale_results = self.diagnostics.stale_results.saturating_add(1);
            self.diagnostics.state = InteractionPickEvidenceState::StaleTargetEpoch;
            return Ok(InteractionPickEvidenceDisposition::IgnoredStale);
        }

        let comparison = report.comparison;
        if !comparison.coverage_matches {
            self.diagnostics.coverage_mismatches =
                self.diagnostics.coverage_mismatches.saturating_add(1);
        }
        if !comparison.identity_matches {
            self.diagnostics.identity_mismatches =
                self.diagnostics.identity_mismatches.saturating_add(1);
        }
        if let Some(error) = comparison.maximum_barycentric_error {
            self.diagnostics.maximum_barycentric_error =
                self.diagnostics.maximum_barycentric_error.max(error);
        }
        if let Some(error) = comparison.maximum_source_position_error {
            self.diagnostics.maximum_source_position_error =
                self.diagnostics.maximum_source_position_error.max(error);
        }
        if let Some(error) = comparison.output_distance_error {
            self.diagnostics.maximum_output_distance_error =
                self.diagnostics.maximum_output_distance_error.max(error);
        }
        self.diagnostics.maximum_staging_ms =
            self.diagnostics.maximum_staging_ms.max(report.staging_ms);
        self.diagnostics.maximum_readback_ms =
            self.diagnostics.maximum_readback_ms.max(report.readback_ms);
        self.diagnostics.maximum_total_ms =
            self.diagnostics.maximum_total_ms.max(report.total_ms);
        self.diagnostics.state = if comparison.topology_matches() {
            InteractionPickEvidenceState::Shadowing
        } else {
            InteractionPickEvidenceState::Mismatch
        };
        Ok(InteractionPickEvidenceDisposition::Recorded)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InteractionTargetError {
    DuplicatePackedNode(u32),
    InvalidTarget {
        packed_node: u32,
        reason: &'static str,
    },
    UnknownPackedNode(u32),
    UnmappedPackedNode(u32),
    StaleTargetEpoch {
        expected: u32,
        actual: u32,
    },
    InvalidSample(&'static str),
}

impl fmt::Display for InteractionTargetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicatePackedNode(node) => {
                write!(formatter, "interaction target table repeats packed node {node}")
            }
            Self::InvalidTarget {
                packed_node,
                reason,
            } => write!(
                formatter,
                "interaction target for packed node {packed_node} is invalid: {reason}",
            ),
            Self::UnknownPackedNode(node) => {
                write!(formatter, "interaction query references unknown packed node {node}")
            }
            Self::UnmappedPackedNode(node) => write!(
                formatter,
                "interaction query references packed node {node} without stable identity",
            ),
            Self::StaleTargetEpoch { expected, actual } => write!(
                formatter,
                "interaction query target epoch {actual} is stale; current epoch is {expected}",
            ),
            Self::InvalidSample(reason) => {
                write!(formatter, "interaction query sample is invalid: {reason}")
            }
        }
    }
}

impl Error for InteractionTargetError {}

/// Scale-aware interaction and selection policy.
#[derive(Resource, Debug, Clone, Copy, PartialEq)]
pub struct InteractionPolicy {
    /// Prevent a tiny detached focus sphere from making interaction impossible.
    pub minimum_output_reach: f64,
    /// Reach measured in radii of the shared focus/inversion sphere.
    pub focus_radius_reach: f64,
    pub selection_margin: f64,
    pub selection_duration_seconds: f64,
    pub selection_easing: TransitionEasing,
}

impl Default for InteractionPolicy {
    fn default() -> Self {
        Self {
            minimum_output_reach: 0.25,
            focus_radius_reach: 8.0,
            selection_margin: 1.1,
            selection_duration_seconds: 0.25,
            selection_easing: TransitionEasing::SmootherStep,
        }
    }
}

impl InteractionPolicy {
    pub fn validate(self) -> Result<(), &'static str> {
        if !self.minimum_output_reach.is_finite() || self.minimum_output_reach <= 0.0 {
            return Err("minimum interaction reach must be finite and positive");
        }
        if !self.focus_radius_reach.is_finite() || self.focus_radius_reach <= 0.0 {
            return Err("focus-relative interaction reach must be finite and positive");
        }
        if !self.selection_margin.is_finite()
            || self.selection_margin < FocusNavigation::MIN_ANCHORED_MARGIN
            || self.selection_margin > FocusNavigation::MAX_ANCHORED_MARGIN
        {
            return Err("interaction selection margin lies outside the anchored-focus bounds");
        }
        if !self.selection_duration_seconds.is_finite() || self.selection_duration_seconds < 0.0 {
            return Err("interaction selection duration must be finite and nonnegative");
        }
        Ok(())
    }

    pub fn output_reach(self, focus: &FocusNavigation) -> Result<f64, &'static str> {
        self.validate()?;
        FocusSphere::new(focus.sphere.center, focus.sphere.radius)?;
        Ok(self
            .minimum_output_reach
            .max(focus.sphere.radius * self.focus_radius_reach))
    }

    pub fn admits(
        self,
        focus: &FocusNavigation,
        hit: InteractionHit,
    ) -> Result<bool, &'static str> {
        hit.validate()?;
        Ok(hit.output_distance <= self.output_reach(focus)?)
    }
}

/// Device-neutral pointer semantics. Raw coordinates, rays, and DOM button
/// events are adapter concerns and must resolve to this vocabulary first.
#[derive(Debug, Clone, PartialEq)]
pub enum InteractionAction {
    /// Accept a hit already resolved by an exact screen ray, XR ray, or other
    /// query whose own geometry defines its reach.
    SetHover(Option<InteractionHit>),
    /// Accept a nearby hit only when it lies within the scale-aware focus
    /// reach. Game proximity queries use this instead of weakening ray picks.
    SetProximityHover(Option<InteractionHit>),
    PressPrimary,
    ReleasePrimary,
    CancelPrimary,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScheduledInteractionAction {
    pub sequence: u64,
    pub at_seconds: f64,
    pub action: InteractionAction,
}

#[derive(Resource, Debug, Clone, Default, PartialEq)]
pub struct InteractionActionQueue {
    next_sequence: u64,
    pending: VecDeque<ScheduledInteractionAction>,
}

impl InteractionActionQueue {
    pub fn push(
        &mut self,
        at_seconds: f64,
        action: InteractionAction,
    ) -> Result<u64, &'static str> {
        let sequence = self.next_sequence;
        self.insert(ScheduledInteractionAction {
            sequence,
            at_seconds,
            action,
        })?;
        Ok(sequence)
    }

    pub fn insert(&mut self, scheduled: ScheduledInteractionAction) -> Result<(), &'static str> {
        if !scheduled.at_seconds.is_finite() || scheduled.at_seconds < 0.0 {
            return Err("interaction action time must be finite and nonnegative");
        }
        if self
            .pending
            .iter()
            .any(|queued| queued.sequence == scheduled.sequence)
        {
            return Err("interaction action sequence is already queued");
        }
        let position = self
            .pending
            .iter()
            .position(|queued| queued.sequence > scheduled.sequence)
            .unwrap_or(self.pending.len());
        if position
            .checked_sub(1)
            .and_then(|previous| self.pending.get(previous))
            .is_some_and(|previous| previous.at_seconds > scheduled.at_seconds)
            || self
                .pending
                .get(position)
                .is_some_and(|next| next.at_seconds < scheduled.at_seconds)
        {
            return Err("interaction action times must be nondecreasing by sequence");
        }
        self.next_sequence = self.next_sequence.max(scheduled.sequence.saturating_add(1));
        self.pending.insert(position, scheduled);
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.pending.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    fn pop_due(&mut self, elapsed_seconds: f64) -> Option<ScheduledInteractionAction> {
        (self.pending.front()?.at_seconds <= elapsed_seconds)
            .then(|| self.pending.pop_front())
            .flatten()
    }
}

/// Ephemeral interaction state. Selected identity is deliberately absent and
/// is derived from [`FocusNavigation::anchor`] in [`InteractionSnapshot`].
#[derive(Resource, Debug, Clone, Default, PartialEq)]
pub struct InteractionState {
    pub revision: u64,
    pub integrated_until_seconds: f64,
    pub last_applied_sequence: Option<u64>,
    pub hovered: Option<InteractionHit>,
    pub active: Option<InteractionHit>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InteractionActivation {
    pub sequence: u64,
    pub at_seconds: f64,
    pub hit: InteractionHit,
}

impl InteractionActivation {
    pub fn navigation_action(self, policy: InteractionPolicy) -> NavigationAction {
        NavigationAction::AnchorFocus {
            identity: self.hit.identity,
            source_bound: self.hit.source_bound,
            source_pivot: self.hit.source_pivot,
            margin: policy.selection_margin,
            duration_seconds: policy.selection_duration_seconds,
            easing: policy.selection_easing,
        }
    }
}

/// Current-frame interaction effects. They are cleared by the reducer before
/// each update and are not durable application state.
#[derive(Resource, Debug, Clone, Default, PartialEq)]
pub struct InteractionActivations(pub Vec<InteractionActivation>);

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct InteractionSnapshot {
    pub revision: u64,
    pub integrated_until_seconds: f64,
    pub last_applied_sequence: Option<u64>,
    pub hovered: Option<InteractionHit>,
    pub active: Option<InteractionHit>,
    pub selected: Option<AssetEntityId>,
}

impl InteractionSnapshot {
    pub fn from_state(state: &InteractionState, focus: &FocusNavigation) -> Self {
        Self {
            revision: state.revision,
            integrated_until_seconds: state.integrated_until_seconds,
            last_applied_sequence: state.last_applied_sequence,
            hovered: state.hovered,
            active: state.active,
            selected: focus.anchor.map(|anchor| anchor.identity),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct InteractionController {
    pub queue: InteractionActionQueue,
    pub state: InteractionState,
    pub policy: InteractionPolicy,
    pub diagnostics: HyperscapeDiagnostics,
}

impl InteractionController {
    pub fn push(&mut self, action: InteractionAction) -> Result<u64, &'static str> {
        self.queue.push(self.state.integrated_until_seconds, action)
    }

    pub fn schedule(
        &mut self,
        at_seconds: f64,
        action: InteractionAction,
    ) -> Result<u64, &'static str> {
        self.queue.push(at_seconds, action)
    }

    pub fn advance_to(
        &mut self,
        elapsed_seconds: f64,
        focus: &FocusNavigation,
    ) -> Result<Vec<InteractionActivation>, &'static str> {
        integrate_interactions_to(
            elapsed_seconds,
            &mut self.queue,
            &mut self.state,
            self.policy,
            focus,
            &mut self.diagnostics,
        )
    }
}

fn integrate_interactions_to(
    elapsed_seconds: f64,
    queue: &mut InteractionActionQueue,
    state: &mut InteractionState,
    policy: InteractionPolicy,
    focus: &FocusNavigation,
    diagnostics: &mut HyperscapeDiagnostics,
) -> Result<Vec<InteractionActivation>, &'static str> {
    if !elapsed_seconds.is_finite() || elapsed_seconds < state.integrated_until_seconds {
        return Err("interaction time must be finite and monotonic");
    }
    policy.validate()?;
    let mut activations = Vec::new();
    while let Some(scheduled) = queue.pop_due(elapsed_seconds) {
        match apply_interaction_action(state, policy, focus, &scheduled) {
            Ok(Some(activation)) => activations.push(activation),
            Ok(None) => {}
            Err(error) => diagnostics.0.push(format!(
                "interaction action {} rejected: {error}",
                scheduled.sequence
            )),
        }
    }
    state.integrated_until_seconds = elapsed_seconds;
    Ok(activations)
}

fn apply_interaction_action(
    state: &mut InteractionState,
    policy: InteractionPolicy,
    focus: &FocusNavigation,
    scheduled: &ScheduledInteractionAction,
) -> Result<Option<InteractionActivation>, &'static str> {
    let previous_hover = state.hovered;
    let previous_active = state.active;
    let activation = match scheduled.action {
        InteractionAction::SetHover(hit) => {
            if let Some(hit) = hit {
                hit.validate()?;
            }
            state.hovered = hit;
            None
        }
        InteractionAction::SetProximityHover(hit) => {
            state.hovered = match hit {
                Some(hit) if policy.admits(focus, hit)? => Some(hit),
                Some(_) | None => None,
            };
            None
        }
        InteractionAction::PressPrimary => {
            state.active = state.hovered;
            None
        }
        InteractionAction::ReleasePrimary => {
            let activation = state
                .active
                .zip(state.hovered)
                .filter(|(active, hovered)| active.identity == hovered.identity)
                .map(|(_, hit)| InteractionActivation {
                    sequence: scheduled.sequence,
                    at_seconds: scheduled.at_seconds,
                    hit,
                });
            state.active = None;
            activation
        }
        InteractionAction::CancelPrimary => {
            state.active = None;
            None
        }
    };
    if state.hovered != previous_hover || state.active != previous_active || activation.is_some() {
        state.revision = state.revision.saturating_add(1);
    }
    state.last_applied_sequence = Some(scheduled.sequence);
    Ok(activation)
}

pub(crate) fn apply_interaction_actions(
    time: Res<Time<Virtual>>,
    mut queue: ResMut<InteractionActionQueue>,
    mut state: ResMut<InteractionState>,
    policy: Res<InteractionPolicy>,
    focus: Res<FocusNavigation>,
    mut activations: ResMut<InteractionActivations>,
    mut diagnostics: ResMut<HyperscapeDiagnostics>,
) {
    activations.0.clear();
    match integrate_interactions_to(
        time.elapsed_secs_f64(),
        &mut queue,
        &mut state,
        *policy,
        &focus,
        &mut diagnostics,
    ) {
        Ok(produced) => activations.0 = produced,
        Err(error) => diagnostics
            .0
            .push(format!("interaction integration rejected: {error}")),
    }
}

pub(crate) fn route_interaction_activations(
    activations: Res<InteractionActivations>,
    policy: Res<InteractionPolicy>,
    mut navigation: ResMut<NavigationActionQueue>,
    mut diagnostics: ResMut<HyperscapeDiagnostics>,
) {
    for activation in &activations.0 {
        if let Err(error) =
            navigation.push(activation.at_seconds, activation.navigation_action(*policy))
        {
            diagnostics.0.push(format!(
                "interaction activation {} could not enter navigation: {error}",
                activation.sequence
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HyperscapePlugin;
    use bevy_app::App;
    use hyperscape_protocol::{AssetId, EntityId};
    use quilting_core::render_evidence::{RenderPickComparison, RenderPickHit};
    use uuid::Uuid;

    fn identity(asset: u128, entity: u128) -> AssetEntityId {
        AssetEntityId::new(
            AssetId::new(Uuid::from_u128(asset)).unwrap(),
            EntityId::new(Uuid::from_u128(entity)).unwrap(),
        )
        .unwrap()
    }

    fn hit(asset: u128, entity: u128, distance: f64) -> InteractionHit {
        InteractionHit::new(
            identity(asset, entity),
            FocusSphere::new([1.0, 2.0, 3.0], 2.0).unwrap(),
            [1.25, 2.0, 3.0],
            distance,
        )
        .unwrap()
        .with_surface(7, [2.0, 1.0, 1.0])
        .unwrap()
    }

    fn pick_report(
        target_epoch: u32,
        expected: Option<RenderPickHit>,
        actual: Option<RenderPickHit>,
    ) -> RenderPickEvidenceReport {
        RenderPickEvidenceReport {
            webgl_render_call: 17,
            webgpu_frame_revision: 9,
            viewport: [1600, 900],
            pixel: [812, 417],
            target_epoch,
            comparison: RenderPickComparison::between(expected, actual).unwrap(),
            staging_ms: 0.2,
            readback_ms: 0.8,
            total_ms: 1.1,
        }
    }

    fn render_hit(node: u32, face: u32, barycentric: [f32; 3]) -> RenderPickHit {
        RenderPickHit::new(node, face, barycentric, [1.0, -2.0, 0.5], 4.0).unwrap()
    }

    #[test]
    fn surface_points_normalize_and_reject_invalid_coordinates() {
        assert_eq!(
            InteractionSurfacePoint::new(4, [2.0, 1.0, 1.0])
                .unwrap()
                .barycentric,
            [0.5, 0.25, 0.25],
        );
        assert!(InteractionSurfacePoint::new(4, [f64::NAN, 0.0, 1.0]).is_err());
        assert!(InteractionSurfacePoint::new(4, [-1.0, 0.0, 1.0]).is_err());
    }

    #[test]
    fn target_table_joins_transient_nodes_to_asset_scoped_identity() {
        let shared_entity = 0x33;
        let table = InteractionTargetTable::try_from_targets([
            InteractionTarget::new(
                7,
                Some(identity(0x11, shared_entity)),
                FocusSphere::new([1.0, 2.0, 3.0], 2.0).unwrap(),
            )
            .unwrap(),
            InteractionTarget::new(
                8,
                Some(identity(0x22, shared_entity)),
                FocusSphere::new([-1.0, 0.0, 1.0], 0.5).unwrap(),
            )
            .unwrap(),
        ])
        .unwrap();
        let resolved = table
            .resolve(
                InteractionTargetSample::new(8, [-0.75, 0.0, 1.0], 4.0)
                    .unwrap()
                    .with_surface(19, [2.0, 1.0, 1.0])
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(resolved.identity, identity(0x22, shared_entity));
        assert_eq!(resolved.source_bound.radius, 0.5);
        assert_eq!(resolved.source_pivot, [-0.75, 0.0, 1.0]);
        assert_eq!(resolved.surface.unwrap().face, 19);
        assert_eq!(resolved.surface.unwrap().barycentric, [0.5, 0.25, 0.25]);
    }

    #[test]
    fn target_table_rejects_duplicate_unknown_unmapped_and_invalid_samples() {
        let bound = FocusSphere::new([0.0; 3], 1.0).unwrap();
        let duplicate = InteractionTarget::new(3, Some(identity(1, 2)), bound).unwrap();
        assert_eq!(
            InteractionTargetTable::try_from_targets([duplicate, duplicate]),
            Err(InteractionTargetError::DuplicatePackedNode(3)),
        );

        let table = InteractionTargetTable::try_from_targets([
            InteractionTarget::new(3, None, bound).unwrap(),
        ])
        .unwrap();
        assert_eq!(
            table.resolve(InteractionTargetSample::new(4, [0.0; 3], 1.0).unwrap()),
            Err(InteractionTargetError::UnknownPackedNode(4)),
        );
        assert_eq!(
            table.resolve(InteractionTargetSample::new(3, [0.0; 3], 1.0).unwrap()),
            Err(InteractionTargetError::UnmappedPackedNode(3)),
        );
        assert_eq!(
            InteractionTargetSample::new(3, [f64::NAN, 0.0, 0.0], 1.0),
            Err(InteractionTargetError::InvalidSample(
                "interaction source pivot must be finite"
            )),
        );

        let current = InteractionTargetTable::try_from_epoch(
            9,
            [InteractionTarget::new(3, Some(identity(1, 2)), bound).unwrap()],
        )
        .unwrap();
        assert_eq!(
            current.resolve(
                InteractionTargetSample::new_for_epoch(8, 3, [0.0; 3], 1.0).unwrap(),
            ),
            Err(InteractionTargetError::StaleTargetEpoch {
                expected: 9,
                actual: 8,
            }),
        );
    }

    #[test]
    fn pick_authority_accepts_only_the_newest_request() {
        let mut authority = InteractionPickAuthority::default();
        let first = authority.begin(9).unwrap();
        assert_eq!(
            authority.record_stage(first, None),
            InteractionPickAuthorityDisposition::Current,
        );
        let second = authority.begin(9).unwrap();
        authority.record_stage(second, None);

        assert_eq!(
            authority.complete(first, 9),
            InteractionPickAuthorityDisposition::IgnoredSuperseded,
        );
        let reading = authority.snapshot();
        assert_eq!(reading.state, InteractionPickAuthorityState::Reading);
        assert_eq!(reading.superseded_requests, 1);
        assert_eq!(reading.stale_completions, 1);
        assert_eq!(reading.latest_request_id, Some(second.request_id));
        assert_eq!(reading.readbacks, 0);

        assert_eq!(
            authority.complete(second, 9),
            InteractionPickAuthorityDisposition::Current,
        );
        let accepted = authority.snapshot();
        assert_eq!(accepted.state, InteractionPickAuthorityState::Accepted);
        assert_eq!(accepted.requests, 2);
        assert_eq!(accepted.staged, 2);
        assert_eq!(accepted.readbacks, 1);
        assert_eq!(accepted.accepted, 1);
        assert_eq!(accepted.latest_request_id, None);
    }

    #[test]
    fn pick_authority_rejects_a_replaced_target_epoch() {
        let mut authority = InteractionPickAuthority::default();
        let request = authority.begin(12).unwrap();
        authority.record_stage(request, None);
        assert_eq!(
            authority.complete(request, 13),
            InteractionPickAuthorityDisposition::IgnoredStaleTargetEpoch,
        );
        let stale = authority.snapshot();
        assert_eq!(stale.state, InteractionPickAuthorityState::StaleTargetEpoch);
        assert_eq!(stale.readbacks, 1);
        assert_eq!(stale.accepted, 0);
        assert_eq!(stale.stale_target_epochs, 1);
        assert_eq!(stale.latest_request_id, None);
    }

    #[test]
    fn stale_pick_failures_do_not_clobber_a_newer_request() {
        let mut authority = InteractionPickAuthority::default();
        let first = authority.begin(3).unwrap();
        authority.record_stage(first, None);
        let second = authority.begin(3).unwrap();
        authority.record_stage(second, None);
        assert_eq!(
            authority.record_error(first, "old readback failed"),
            InteractionPickAuthorityDisposition::IgnoredSuperseded,
        );
        assert_eq!(
            authority.snapshot().state,
            InteractionPickAuthorityState::Reading
        );
        assert_eq!(authority.snapshot().errors, 0);

        assert_eq!(
            authority.record_error(second, "device lost"),
            InteractionPickAuthorityDisposition::Current,
        );
        let failed = authority.snapshot();
        assert_eq!(failed.state, InteractionPickAuthorityState::Error);
        assert_eq!(failed.errors, 1);
        assert_eq!(failed.stale_completions, 1);
        assert_eq!(failed.last_error.as_deref(), Some("device lost"));

        let rejected = authority.begin(4).unwrap();
        assert_eq!(
            authority.record_stage(rejected, Some("backend warming".to_string())),
            InteractionPickAuthorityDisposition::Current,
        );
        let rejected = authority.snapshot();
        assert_eq!(rejected.state, InteractionPickAuthorityState::StageRejected);
        assert_eq!(rejected.stage_rejects, 1);
        assert_eq!(rejected.latest_request_id, None);
    }

    #[test]
    fn pick_request_exhaustion_does_not_mutate_diagnostics() {
        let mut authority = InteractionPickAuthority {
            next_request_id: u64::MAX,
            ..InteractionPickAuthority::default()
        };
        let before = authority.snapshot();
        assert_eq!(
            authority.begin(1),
            Err(InteractionPickAuthorityError::RequestIdExhausted),
        );
        assert_eq!(authority.snapshot(), before);
    }

    #[test]
    fn pick_evidence_observer_rejects_stale_target_epochs_before_parity_metrics() {
        let targets = InteractionTargetTable::try_from_epoch(
            9,
            std::iter::empty::<InteractionTarget>(),
        )
        .unwrap();
        let expected = render_hit(7, 3, [0.2, 0.3, 0.5]);
        let wrong_identity = render_hit(8, 4, [0.2, 0.3, 0.5]);
        let mut observer = InteractionPickEvidenceObserver::default();
        observer.record_stage(true, None);
        assert_eq!(observer.snapshot().state, InteractionPickEvidenceState::Reading);

        assert_eq!(
            observer
                .record_report(&targets, pick_report(8, Some(expected), Some(wrong_identity)))
                .unwrap(),
            InteractionPickEvidenceDisposition::IgnoredStale,
        );
        let stale = observer.snapshot();
        assert_eq!(stale.requests, 1);
        assert_eq!(stale.staged, 1);
        assert_eq!(stale.readbacks, 1);
        assert_eq!(stale.stale_results, 1);
        assert_eq!(stale.coverage_mismatches, 0);
        assert_eq!(stale.identity_mismatches, 0);
        assert_eq!(stale.state, InteractionPickEvidenceState::StaleTargetEpoch);

        assert_eq!(
            observer
                .record_report(&targets, pick_report(9, Some(expected), Some(wrong_identity)))
                .unwrap(),
            InteractionPickEvidenceDisposition::Recorded,
        );
        let mismatch = observer.snapshot();
        assert_eq!(mismatch.readbacks, 2);
        assert_eq!(mismatch.identity_mismatches, 1);
        assert_eq!(mismatch.state, InteractionPickEvidenceState::Mismatch);
    }

    #[test]
    fn pick_evidence_observer_is_bounded_and_validates_reports_atomically() {
        let targets = InteractionTargetTable::try_from_epoch(
            9,
            std::iter::empty::<InteractionTarget>(),
        )
        .unwrap();
        let expected = render_hit(7, 3, [0.2, 0.3, 0.5]);
        let actual = RenderPickHit::new(
            7,
            3,
            [0.201, 0.297, 0.502],
            [1.004, -2.0, 0.499],
            4.006,
        )
        .unwrap();
        let mut observer = InteractionPickEvidenceObserver::default();
        observer.record_stage(false, Some("backend warming".to_string()));
        observer.record_error("readback: device lost");
        observer
            .record_report(&targets, pick_report(9, Some(expected), Some(actual)))
            .unwrap();
        let measured = observer.snapshot();
        assert_eq!(measured.requests, 1);
        assert_eq!(measured.stage_rejects, 1);
        assert_eq!(measured.errors, 1);
        assert_eq!(measured.state, InteractionPickEvidenceState::Shadowing);
        assert!((measured.maximum_barycentric_error - 0.003).abs() < 1.0e-6);
        assert!((measured.maximum_source_position_error - 0.004).abs() < 1.0e-6);
        assert!((measured.maximum_output_distance_error - 0.006).abs() < 1.0e-6);
        assert_eq!(measured.last_report.unwrap().pixel, [812, 417]);
        assert_eq!(measured.last_error, None);

        let mut invalid = pick_report(9, Some(expected), Some(expected));
        invalid.total_ms = -1.0;
        assert!(observer.record_report(&targets, invalid).is_err());
        assert_eq!(observer.snapshot(), measured);
    }

    #[test]
    fn scale_aware_reach_filters_hover_without_rejecting_the_input() {
        let focus = FocusNavigation {
            sphere: FocusSphere::new([0.0; 3], 0.1).unwrap(),
            ..FocusNavigation::default()
        };
        let mut controller = InteractionController::default();
        controller.policy.minimum_output_reach = 0.25;
        controller.policy.focus_radius_reach = 2.0;
        controller
            .push(InteractionAction::SetProximityHover(Some(hit(1, 2, 0.24))))
            .unwrap();
        controller.advance_to(0.0, &focus).unwrap();
        assert!(controller.state.hovered.is_some());
        controller
            .push(InteractionAction::SetProximityHover(Some(hit(1, 2, 0.26))))
            .unwrap();
        controller.advance_to(0.0, &focus).unwrap();
        assert_eq!(controller.state.hovered, None);
        assert!(controller.diagnostics.0.is_empty());

        controller
            .push(InteractionAction::SetHover(Some(hit(1, 2, 100.0))))
            .unwrap();
        controller.advance_to(0.0, &focus).unwrap();
        assert!(controller.state.hovered.is_some());
    }

    #[test]
    fn activation_routes_selection_through_navigation_without_duplicate_state() {
        let selected = hit(0x100, 0x200, 1.0);
        let mut app = App::new();
        app.add_plugins(HyperscapePlugin);
        {
            let mut queue = app.world_mut().resource_mut::<InteractionActionQueue>();
            queue
                .push(0.0, InteractionAction::SetHover(Some(selected)))
                .unwrap();
            queue.push(0.0, InteractionAction::PressPrimary).unwrap();
            queue.push(0.0, InteractionAction::ReleasePrimary).unwrap();
        }
        app.update();

        let state = app.world().resource::<InteractionState>();
        let focus = app.world().resource::<FocusNavigation>();
        let snapshot = InteractionSnapshot::from_state(state, focus);
        assert_eq!(snapshot.hovered, Some(selected));
        assert_eq!(snapshot.active, None);
        assert_eq!(snapshot.selected, Some(selected.identity));
        assert_eq!(focus.anchor.unwrap().source_pivot, selected.source_pivot);
        assert_eq!(app.world().resource::<InteractionActivations>().0.len(), 1,);
    }

    #[test]
    fn releasing_over_another_entity_cancels_activation() {
        let focus = FocusNavigation::default();
        let mut controller = InteractionController::default();
        controller
            .schedule(0.0, InteractionAction::SetHover(Some(hit(1, 2, 1.0))))
            .unwrap();
        controller
            .schedule(0.0, InteractionAction::PressPrimary)
            .unwrap();
        controller
            .schedule(0.0, InteractionAction::SetHover(Some(hit(1, 3, 1.0))))
            .unwrap();
        controller
            .schedule(0.0, InteractionAction::ReleasePrimary)
            .unwrap();
        assert!(controller.advance_to(0.0, &focus).unwrap().is_empty());
        assert_eq!(controller.state.active, None);
        assert_eq!(controller.state.hovered.unwrap().identity, identity(1, 3));
    }

    #[test]
    fn invalid_hits_are_consumed_without_partial_state() {
        let focus = FocusNavigation::default();
        let mut controller = InteractionController::default();
        let valid = hit(1, 2, 1.0);
        controller
            .push(InteractionAction::SetHover(Some(valid)))
            .unwrap();
        controller.advance_to(0.0, &focus).unwrap();
        let before = controller.state.clone();
        let mut invalid = valid;
        invalid.source_pivot[0] = f64::NAN;
        controller
            .push(InteractionAction::SetHover(Some(invalid)))
            .unwrap();
        controller.advance_to(0.0, &focus).unwrap();
        assert_eq!(controller.state, before);
        assert_eq!(controller.queue.len(), 0);
        assert_eq!(controller.diagnostics.0.len(), 1);
    }

    #[test]
    fn virtual_time_and_sequence_are_cadence_invariant() {
        let focus = FocusNavigation::default();
        let actions = [
            (0.1, InteractionAction::SetHover(Some(hit(1, 2, 1.0)))),
            (0.2, InteractionAction::PressPrimary),
            (0.3, InteractionAction::ReleasePrimary),
        ];
        let mut single = InteractionController::default();
        let mut stepped = InteractionController::default();
        for (time, action) in actions {
            single.schedule(time, action.clone()).unwrap();
            stepped.schedule(time, action).unwrap();
        }
        let single_activations = single.advance_to(0.3, &focus).unwrap();
        let mut stepped_activations = Vec::new();
        for time in [0.1, 0.2, 0.3] {
            stepped_activations.extend(stepped.advance_to(time, &focus).unwrap());
        }
        assert_eq!(single.state, stepped.state);
        assert_eq!(single_activations, stepped_activations);
        assert_eq!(single.state.last_applied_sequence, Some(2));
    }
}
