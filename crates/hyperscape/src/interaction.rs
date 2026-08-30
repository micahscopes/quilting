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
use std::collections::VecDeque;

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
