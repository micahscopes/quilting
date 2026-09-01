//! Hyperscape is the deterministic ECS/game layer above Quilting's conformal
//! math and below the Hyperscope renderer.
//!
//! The crate deliberately depends on Bevy's app, ECS, and time modules only.
//! A conformal frame is not an affine `Transform`, and rendering remains the
//! responsibility of Hyperscope/Quilting.

use bevy_app::{App, Plugin, Update};
use bevy_ecs::prelude::*;
use bevy_time::{Time, Virtual};
use hyperscape_protocol::{EntityId, WireError};
use quilting_core::{
    AnchorState, ConformalFrameForest, FrameId, OpenRoundSide, RoundSideOrientation,
    RoundWallRelation, RoundWallSet, WallId,
};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use uuid::Uuid;

pub mod interchange;
pub mod interaction;
pub mod navigation;
pub mod packed_scene;
#[cfg(not(target_arch = "wasm32"))]
pub mod preflight;
pub mod presentation;
pub mod surface;
pub mod surface_frame_pin;

pub use navigation::{
    compose_surface_relative_forward, decompose_surface_relative_forward, framed_sphere_distance,
    map_pointer_turntable, map_space_mouse_axes, map_space_mouse_camera, scale_relative_near_plane,
    AnimatedSurfaceAnchor, CameraBasis, CameraError, CameraRig, CameraTransition, FocusAnchor,
    FocusNavigation, FocusSphere, FocusSphereTransition, MappedSpaceMouseFrame, NavigationAction,
    NavigationActionQueue, NavigationAxes, NavigationController, NavigationFrame,
    NavigationPreset, NavigationRuntime, PerspectiveLens, PointerTurntableGesture,
    PointerTurntableInput, ReflectionTransport, ScheduledNavigationAction, SpaceMouseCameraInput,
    SpaceMouseInputError, SpaceMouseMapping, SphereReflectionState, SurfaceAnchorError,
    SurfaceAnchoredCameraFrame, SurfaceAnchorTarget, SurfaceAnchorTransition,
    SurfaceRelativeCamera, SurfaceRelativeView, SurfaceTangentFrame, SurfaceWalkAttachRequest,
    SurfaceWalkContactFrame, SurfaceWalkController, SurfaceWalkControls, SurfaceWalkError,
    SurfaceWalkFrame, SurfaceWalkInput, SurfaceWalkMetrics, SurfaceWalkMotion,
    SurfaceWalkRecoveryRequest, SurfaceWalkReflectionTransport, SurfaceWalkRuntime,
    SurfaceWalkRuntimeError, SurfaceWalkStepRequest, SurfaceWalkUpdate, TransitionEasing,
    TurntableFrame,
};
pub use interaction::{
    InteractionAction, InteractionActionQueue, InteractionActivation, InteractionActivations,
    InteractionController, InteractionHit, InteractionPickAuthority,
    InteractionPickAuthorityDiagnostics, InteractionPickAuthorityDisposition,
    InteractionPickAuthorityError, InteractionPickAuthorityState,
    InteractionPickEvidenceDiagnostics, InteractionPickEvidenceDisposition,
    InteractionPickEvidenceObserver, InteractionPickEvidenceState, InteractionPickRequest,
    InteractionPolicy, InteractionSnapshot, InteractionState,
    InteractionSurfacePoint, InteractionTarget, InteractionTargetError, InteractionTargetSample,
    InteractionTargetTable, ScheduledInteractionAction,
};
pub use packed_scene::{
    extract_packed_presentation_scene, extract_packed_scene, PackedAssetInstance,
    PackedNodeSource, PackedNodeTransform, PackedNodeTransformSource,
    PackedPresentationLayerBinding, PackedPresentationNode, PackedPresentationSceneError,
    PackedPresentationSceneExtraction, PackedSceneError, PackedSceneExtraction,
};
#[cfg(not(target_arch = "wasm32"))]
pub use preflight::{
    hyperscope_build_receipt, run_offline_preflight, write_hyperscope_build_receipt,
    BundleFileCheck, DistributionPolicy, HyperscopeBuildReceipt, OfflinePreflightOptions,
    OfflinePreflightReport,
};
pub use presentation::{
    presentation_render_style, AssetLoadPolicy, AuthoredCamera, AuthoredFocus, CueAnimation,
    CueText, LayerTransform, Presentation, PresentationAsset, PresentationCue, PresentationError,
    PresentationLayer, PresentationLayerState, PresentationOverlay, PresentationRuntime,
    PresentationScene, PresentationSnapshot, PresentationTessellation, PresentationTransition,
    PresentationView, RenderStyle, ViewLayerOverride, HACKER_NIGHT_PRESENTATION_JSON,
    PRESENTATION_VERSION,
};
pub use surface::{
    SurfaceAddress, SurfaceAddressError, SurfaceAdvance, SurfaceAttachment, SurfaceContact,
    SurfaceDetachReason, SurfaceField, SurfaceSample, SurfaceWalker, SurfaceWalkerConfig,
    SurfaceWalkerStatus, TriangleAdjacency, TriangleAdjacencyReport,
};
pub use surface_frame_pin::{
    ResolvedSurfaceFramePin, SurfaceFrameOrientation, SurfaceFramePin, SurfaceFramePinBinding,
    SurfaceFramePinError, SurfaceFramePinSamples, SurfaceFramePinSet,
};

/// Shared conformal scene topology. Ordinary entity parenting remains in the
/// caller's entity graph; this resource contains only coordinate-frame and
/// round-wall structure.
#[derive(Resource, Debug, Clone, Default)]
pub struct ConformalScene {
    pub frames: ConformalFrameForest,
    pub walls: RoundWallSet,
}

/// An entity's point coordinates are expressed in this conformal frame.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub struct EntityFrame(pub FrameId);

/// Durable authored identity. Bevy [`Entity`] values and glTF node indices are
/// runtime/container handles and must never be written into HHHS history or a
/// presentation manifest as identity.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StableEntityId(pub Uuid);

/// Wire and ECS identities carry the same UUID. The protocol wrapper rejects
/// nil values at interchange boundaries; the runtime wrapper remains backward
/// compatible with existing internal fixtures while conversion is explicit.
impl From<EntityId> for StableEntityId {
    fn from(value: EntityId) -> Self {
        Self(value.as_uuid())
    }
}

impl TryFrom<StableEntityId> for EntityId {
    type Error = WireError;

    fn try_from(value: StableEntityId) -> Result<Self, Self::Error> {
        Self::new(value.0)
    }
}

/// Editable coordinates in [`EntityFrame`].
#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct LocalCoordinates(pub [f64; 3]);

/// Derived coordinates in the ambient Euclidean chart. Keeping both values
/// makes frame entry, exit, and re-anchoring inspectable.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct EuclideanCoordinates(pub [f64; 3]);

/// Ordinary glTF/node-local affine model matrix, kept distinct from the
/// conformal frame chain. Column-major, matching glTF and Hyperscope.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct EuclideanModelMatrix(pub [f32; 16]);

impl Default for EuclideanModelMatrix {
    fn default() -> Self {
        Self([
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ])
    }
}

/// Marks geometry that should receive a per-view Hyperscope transform.
#[derive(Component, Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RenderSubject;

/// A conformal view/chart consumed by render extraction.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProjectionCamera {
    pub frame: FrameId,
}

/// A path control point in an entity's local conformal frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PathKeyframe {
    pub time_seconds: f64,
    pub point: [f64; 3],
}

/// Piecewise-linear path animation. Authoring surfaces may retain richer
/// curves and bake deterministic keyframes for this first runtime slice.
#[derive(Component, Debug, Clone, PartialEq)]
pub struct ConformalPath {
    pub keyframes: Vec<PathKeyframe>,
    pub looping: bool,
}

impl ConformalPath {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.keyframes.is_empty() {
            return Err("a conformal path needs at least one keyframe");
        }
        if self.keyframes.iter().any(|key| {
            !key.time_seconds.is_finite()
                || key.time_seconds < 0.0
                || key.point.into_iter().any(|value| !value.is_finite())
        }) {
            return Err("path times and points must be finite and times nonnegative");
        }
        if self
            .keyframes
            .windows(2)
            .any(|pair| pair[0].time_seconds >= pair[1].time_seconds)
        {
            return Err("path keyframe times must be strictly increasing");
        }
        Ok(())
    }

    pub fn sample(&self, elapsed_seconds: f64) -> Result<[f64; 3], &'static str> {
        self.validate()?;
        let first = self.keyframes[0];
        let last = *self.keyframes.last().expect("validated nonempty path");
        let time = self.sample_time(elapsed_seconds)?;
        if time <= first.time_seconds {
            return Ok(first.point);
        }
        for pair in self.keyframes.windows(2) {
            let [left, right] = [pair[0], pair[1]];
            if time <= right.time_seconds {
                let t = (time - left.time_seconds) / (right.time_seconds - left.time_seconds);
                return Ok([
                    left.point[0] + t * (right.point[0] - left.point[0]),
                    left.point[1] + t * (right.point[1] - left.point[1]),
                    left.point[2] + t * (right.point[2] - left.point[2]),
                ]);
            }
        }
        Ok(last.point)
    }

    pub fn sample_time(&self, elapsed_seconds: f64) -> Result<f64, &'static str> {
        self.validate()?;
        let first = self.keyframes[0];
        let last = *self.keyframes.last().expect("validated nonempty path");
        Ok(if self.looping && last.time_seconds > 0.0 {
            elapsed_seconds.rem_euclid(last.time_seconds)
        } else {
            elapsed_seconds.clamp(first.time_seconds, last.time_seconds)
        })
    }
}

/// A discrete chart/anchor state change on a path timeline. Path geometry is
/// sampled in one stable coordinate frame, then converted to this active
/// frame, so the ambient trajectory stays continuous at the transition.
#[derive(Debug, Clone, PartialEq)]
pub struct PathTransition {
    pub time_seconds: f64,
    pub frame: FrameId,
    pub anchor: AnchorState,
}

/// Reference chart and deterministic chart/anchor timeline for a path.
#[derive(Component, Debug, Clone, PartialEq)]
pub struct ConformalPathTimeline {
    pub coordinate_frame: FrameId,
    pub initial_frame: FrameId,
    pub initial_anchor: AnchorState,
    pub transitions: Vec<PathTransition>,
}

/// Track another entity while expressing the result in the tracker's own
/// frame. This is the basic cross-frame projection/camera constraint.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct CrossFrameTarget {
    pub target: Entity,
    pub local_offset: [f64; 3],
}

/// Derived target coordinates in the tracking entity's frame.
#[derive(Component, Debug, Clone, Copy, PartialEq)]
pub struct TrackedCoordinates(pub [f64; 3]);

/// Anchor-dependent side selection. The wall skeleton itself remains in
/// [`ConformalScene`] and is not mutated by re-anchoring.
#[derive(Component, Debug, Clone, PartialEq, Eq)]
pub struct ActiveAnchor(pub AnchorState);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ChamberSide {
    Negative,
    OnWall,
    Positive,
}

/// Derived sign vector for all round walls, oriented by an optional anchor.
#[derive(Component, Debug, Clone, Default, PartialEq, Eq)]
pub struct ChamberSignature(pub BTreeMap<WallId, ChamberSide>);

/// Contact data is known directly only when both wall equations are expressed
/// in the same frame. A cross-frame pair remains explicit instead of being
/// incorrectly classified without a round-wall transport implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContactClassification {
    Known(RoundWallRelation),
    RequiresCommonChart,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContactRecord {
    pub first: WallId,
    pub second: WallId,
    pub classification: ContactClassification,
}

#[derive(Resource, Debug, Clone, Default, PartialEq, Eq)]
pub struct ContactState(pub Vec<ContactRecord>);

pub type ChamberKey = Vec<(WallId, ChamberSide)>;

/// Sparse aggregate maintenance and measurement for chamber-dependent
/// visibility/LOD hints. Geometric classification is still authoritative;
/// these counts only avoid rebuilding aggregate state when no sign changed.
#[derive(Resource, Debug, Clone, Default)]
pub struct ChamberAggregateState {
    pub epoch: u64,
    pub counts: BTreeMap<ChamberKey, usize>,
    pub changed_entities: Vec<Entity>,
    pub changed_walls: BTreeSet<WallId>,
    pub contact_frontier: BTreeSet<WallId>,
    pub classifications_last_tick: usize,
    pub aggregate_updates_last_tick: usize,
    memberships: BTreeMap<Entity, ChamberKey>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TransformHistorySample {
    pub elapsed_seconds: f64,
    pub frame: FrameId,
    pub local: [f64; 3],
    pub euclidean: [f64; 3],
    pub anchor_frame: Option<FrameId>,
    pub flipped_walls: Vec<WallId>,
}

/// Bounded, change-only coordinate history for browser/editor diagnostics.
#[derive(Resource, Debug, Clone)]
pub struct TransformHistory {
    pub capacity_per_entity: usize,
    pub epoch: u64,
    pub samples: BTreeMap<Entity, VecDeque<TransformHistorySample>>,
}

impl Default for TransformHistory {
    fn default() -> Self {
        Self {
            capacity_per_entity: 32,
            epoch: 0,
            samples: BTreeMap::new(),
        }
    }
}

/// Request queue for discontinuity-free frame parenting changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameReparentRequest {
    pub frame: FrameId,
    pub new_parent: Option<FrameId>,
}

#[derive(Resource, Debug, Clone, Default)]
pub struct FrameReparentRequests(pub Vec<FrameReparentRequest>);

/// Sparse XOR changes to one anchor's orientation bits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnchorFlipRequest {
    pub anchor: Entity,
    pub walls: BTreeSet<WallId>,
}

#[derive(Resource, Debug, Clone, Default)]
pub struct AnchorFlipRequests(pub Vec<AnchorFlipRequest>);

/// One entity/view transform ready for the existing Hyperscope uniform path.
#[derive(Debug, Clone, PartialEq)]
pub struct HyperscopePacket {
    pub subject: Entity,
    pub camera: Entity,
    /// Quaternion coefficients `[a, b, c, d]`, each in `(w,x,y,z)` order.
    pub mobius: [f32; 16],
    pub orientation_sign: i8,
    /// Ordinary affine transform applied before the conformal frame map.
    pub euclidean_model: [f32; 16],
    /// Ordinary projection-camera eye in the camera frame.
    pub camera_eye: [f32; 3],
    /// Optional cross-frame target, expressed in the camera frame.
    pub camera_target: Option<[f32; 3]>,
    /// Squared Möbius denominator at the affine model origin. Values below
    /// `POLE_PROXIMITY_NORM_SQ` are a diagnostic/maximum-LOD warning, not a
    /// proof that the entire mesh intersects the pole.
    pub origin_pole_denominator_norm_sq: f64,
}

#[derive(Resource, Debug, Clone, Default, PartialEq)]
pub struct HyperscopeExtraction(pub Vec<HyperscopePacket>);

/// Recoverable runtime diagnostics. A malformed authored relation does not
/// panic or partially mutate the rest of the scene tick.
#[derive(Resource, Debug, Clone, Default, PartialEq, Eq)]
pub struct HyperscapeDiagnostics(pub Vec<String>);

#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HyperscapeSet {
    Interaction,
    FrameMutation,
    Animation,
    Coordinates,
    Constraints,
    Mereology,
    Extraction,
}

#[derive(Default)]
pub struct HyperscapePlugin;

impl Plugin for HyperscapePlugin {
    fn build(&self, app: &mut App) {
        // Hyperscape owns a deterministic virtual clock. Embedders advance it
        // explicitly before `App::update`; this avoids coupling authored time
        // to a native wall clock and is equally valid in native and wasm apps.
        app.init_resource::<Time<Virtual>>()
            .init_resource::<ConformalScene>()
            .init_resource::<FrameReparentRequests>()
            .init_resource::<AnchorFlipRequests>()
            .init_resource::<ContactState>()
            .init_resource::<ChamberAggregateState>()
            .init_resource::<TransformHistory>()
            .init_resource::<FocusNavigation>()
            .init_resource::<InteractionPolicy>()
            .init_resource::<InteractionActionQueue>()
            .init_resource::<InteractionState>()
            .init_resource::<InteractionActivations>()
            .init_resource::<CameraRig>()
            .init_resource::<NavigationActionQueue>()
            .init_resource::<NavigationRuntime>()
            .init_resource::<SurfaceWalkRuntime>()
            .init_resource::<SurfaceFramePinSet>()
            .init_resource::<SurfaceFramePinSamples>()
            .init_resource::<HyperscopeExtraction>()
            .init_resource::<HyperscapeDiagnostics>()
            .configure_sets(
                Update,
                (
                    HyperscapeSet::Interaction,
                    HyperscapeSet::FrameMutation,
                    HyperscapeSet::Animation,
                    HyperscapeSet::Coordinates,
                    HyperscapeSet::Constraints,
                    HyperscapeSet::Mereology,
                    HyperscapeSet::Extraction,
                )
                    .chain(),
            )
            .add_systems(
                Update,
                (
                    interaction::apply_interaction_actions,
                    interaction::route_interaction_activations,
                    navigation::action::apply_navigation_actions,
                )
                    .chain()
                    .in_set(HyperscapeSet::Interaction),
            )
            .add_systems(
                Update,
                (
                    apply_frame_reparents,
                    surface_frame_pin::apply_surface_frame_pin_samples,
                )
                    .chain()
                    .in_set(HyperscapeSet::FrameMutation),
            )
            .add_systems(
                Update,
                (apply_path_timelines, animate_paths)
                    .chain()
                    .in_set(HyperscapeSet::Animation),
            )
            .add_systems(
                Update,
                resolve_euclidean_coordinates.in_set(HyperscapeSet::Coordinates),
            )
            .add_systems(
                Update,
                (track_across_frames, sync_authored_model_translations)
                    .chain()
                    .in_set(HyperscapeSet::Constraints),
            )
            .add_systems(
                Update,
                (apply_anchor_flips, classify_contacts, classify_chambers)
                    .chain()
                    .in_set(HyperscapeSet::Mereology),
            )
            .add_systems(
                Update,
                (record_transform_history, extract_hyperscope_packets)
                    .chain()
                    .in_set(HyperscapeSet::Extraction),
            );
    }
}

fn apply_frame_reparents(
    mut scene: ResMut<ConformalScene>,
    mut requests: ResMut<FrameReparentRequests>,
    mut diagnostics: ResMut<HyperscapeDiagnostics>,
) {
    for request in requests.0.drain(..) {
        if let Err(error) = scene
            .frames
            .reparent_preserve_world(request.frame, request.new_parent)
        {
            diagnostics.0.push(format!(
                "could not reparent conformal frame {}: {error}",
                request.frame.0
            ));
        }
    }
}

fn animate_paths(
    time: Res<Time<Virtual>>,
    scene: Res<ConformalScene>,
    mut diagnostics: ResMut<HyperscapeDiagnostics>,
    mut paths: Query<(
        Entity,
        &ConformalPath,
        Option<&ConformalPathTimeline>,
        &EntityFrame,
        &mut LocalCoordinates,
    )>,
) {
    let elapsed = time.elapsed_secs_f64();
    for (entity, path, timeline, active_frame, mut coordinates) in &mut paths {
        let sampled = path.sample(elapsed).and_then(|point| {
            timeline.map_or(Ok(point), |timeline| {
                scene
                    .frames
                    .convert_point(point, timeline.coordinate_frame, active_frame.0)
                    .map_err(|_| "path coordinate conversion failed")
            })
        });
        match sampled {
            Ok(point) => coordinates.0 = point,
            Err(error) => diagnostics
                .0
                .push(format!("invalid conformal path on {entity:?}: {error}")),
        }
    }
}

fn apply_path_timelines(
    time: Res<Time<Virtual>>,
    mut diagnostics: ResMut<HyperscapeDiagnostics>,
    mut paths: Query<(
        Entity,
        &ConformalPath,
        &ConformalPathTimeline,
        &mut EntityFrame,
        &mut ActiveAnchor,
    )>,
) {
    let elapsed = time.elapsed_secs_f64();
    for (entity, path, timeline, mut frame, mut anchor) in &mut paths {
        let Ok(sample_time) = path.sample_time(elapsed) else {
            diagnostics
                .0
                .push(format!("invalid conformal path timeline on {entity:?}"));
            continue;
        };
        let mut selected_frame = timeline.initial_frame;
        let mut selected_anchor = &timeline.initial_anchor;
        for transition in &timeline.transitions {
            if transition.time_seconds > sample_time {
                break;
            }
            selected_frame = transition.frame;
            selected_anchor = &transition.anchor;
        }
        frame.0 = selected_frame;
        anchor.0 = selected_anchor.clone();
    }
}

fn resolve_euclidean_coordinates(
    scene: Res<ConformalScene>,
    mut diagnostics: ResMut<HyperscapeDiagnostics>,
    mut points: Query<(
        Entity,
        &EntityFrame,
        &LocalCoordinates,
        &mut EuclideanCoordinates,
    )>,
) {
    for (entity, frame, local, mut euclidean) in &mut points {
        match scene
            .frames
            .world_chain(frame.0)
            .and_then(|chain| chain.apply_point(local.0))
        {
            Ok(point) => euclidean.0 = point,
            Err(error) => diagnostics.0.push(format!(
                "could not resolve Euclidean coordinates for {entity:?}: {error}"
            )),
        }
    }
}

fn track_across_frames(
    scene: Res<ConformalScene>,
    mut diagnostics: ResMut<HyperscapeDiagnostics>,
    targets: Query<(&EntityFrame, &LocalCoordinates)>,
    mut trackers: Query<(
        Entity,
        &EntityFrame,
        &CrossFrameTarget,
        &mut TrackedCoordinates,
    )>,
) {
    for (entity, tracker_frame, constraint, mut tracked) in &mut trackers {
        let Ok((target_frame, target_point)) = targets.get(constraint.target) else {
            diagnostics.0.push(format!(
                "cross-frame target {:?} for {entity:?} has no frame/local coordinates",
                constraint.target
            ));
            continue;
        };
        match scene
            .frames
            .convert_point(target_point.0, target_frame.0, tracker_frame.0)
        {
            Ok(point) => {
                tracked.0 = [
                    point[0] + constraint.local_offset[0],
                    point[1] + constraint.local_offset[1],
                    point[2] + constraint.local_offset[2],
                ];
            }
            Err(error) => diagnostics
                .0
                .push(format!("could not track target for {entity:?}: {error}")),
        }
    }
}

/// Apply authored path coordinates to the ordinary affine transform that acts
/// before the entity's conformal frame chain. Cross-frame tracking remains a
/// target/aim constraint; it must not teleport the tracking entity. Rotation
/// and scale remain ordinary glTF data; only translation is driven here.
fn sync_authored_model_translations(
    mut models: Query<(
        Option<&ConformalPath>,
        &LocalCoordinates,
        &mut EuclideanModelMatrix,
    )>,
) {
    for (path, local, mut model) in &mut models {
        if path.is_none() {
            continue;
        }
        model.0[12] = local.0[0] as f32;
        model.0[13] = local.0[1] as f32;
        model.0[14] = local.0[2] as f32;
    }
}

fn apply_anchor_flips(
    mut requests: ResMut<AnchorFlipRequests>,
    mut anchors: Query<&mut ActiveAnchor>,
    mut diagnostics: ResMut<HyperscapeDiagnostics>,
) {
    for request in requests.0.drain(..) {
        match anchors.get_mut(request.anchor) {
            Ok(mut anchor) => anchor.0.apply_flip_set(&request.walls),
            Err(_) => diagnostics
                .0
                .push(format!("anchor entity {:?} does not exist", request.anchor)),
        }
    }
}

fn classify_contacts(
    scene: Res<ConformalScene>,
    mut contacts: ResMut<ContactState>,
    mut diagnostics: ResMut<HyperscapeDiagnostics>,
) {
    contacts.0.clear();
    let walls = scene.walls.walls();
    for first in 0..walls.len() {
        for second in first + 1..walls.len() {
            let classification = if walls[first].frame != walls[second].frame {
                ContactClassification::RequiresCommonChart
            } else {
                match walls[first]
                    .geometry
                    .relation(&walls[second].geometry, 1.0e-9)
                {
                    Ok(relation) => ContactClassification::Known(relation),
                    Err(error) => {
                        diagnostics.0.push(format!(
                            "could not classify walls {first} and {second}: {error}"
                        ));
                        continue;
                    }
                }
            };
            contacts.0.push(ContactRecord {
                first: WallId(first),
                second: WallId(second),
                classification,
            });
        }
    }
}

fn classify_chambers(
    scene: Res<ConformalScene>,
    contacts: Res<ContactState>,
    mut diagnostics: ResMut<HyperscapeDiagnostics>,
    mut aggregates: ResMut<ChamberAggregateState>,
    mut points: Query<(
        Entity,
        &EntityFrame,
        &LocalCoordinates,
        Option<&ActiveAnchor>,
        &mut ChamberSignature,
    )>,
) {
    aggregates.changed_entities.clear();
    aggregates.changed_walls.clear();
    aggregates.contact_frontier.clear();
    aggregates.classifications_last_tick = 0;
    aggregates.aggregate_updates_last_tick = 0;
    let mut seen = BTreeSet::new();
    for (entity, frame, point, anchor, mut signature) in &mut points {
        seen.insert(entity);
        let mut next_signature = BTreeMap::new();
        for (index, _) in scene.walls.walls().iter().enumerate() {
            aggregates.classifications_last_tick += 1;
            let wall = WallId(index);
            let negative = OpenRoundSide {
                wall,
                orientation: RoundSideOrientation::Negative,
            };
            let positive = negative.complement();
            let side = match (
                scene
                    .walls
                    .contains(&scene.frames, negative, point.0, frame.0),
                scene
                    .walls
                    .contains(&scene.frames, positive, point.0, frame.0),
            ) {
                (Ok(true), Ok(false)) => ChamberSide::Negative,
                (Ok(false), Ok(true)) => ChamberSide::Positive,
                (Ok(false), Ok(false)) => ChamberSide::OnWall,
                (Ok(true), Ok(true)) => {
                    diagnostics.0.push(format!(
                        "wall {} classified both complementary sides for {entity:?}",
                        wall.0
                    ));
                    continue;
                }
                (Err(error), _) | (_, Err(error)) => {
                    diagnostics.0.push(format!(
                        "could not classify wall {} for {entity:?}: {error}",
                        wall.0
                    ));
                    continue;
                }
            };
            let oriented = match (side, anchor) {
                (ChamberSide::OnWall, _) => ChamberSide::OnWall,
                (ChamberSide::Negative, Some(anchor))
                    if anchor.0.orient(wall, RoundSideOrientation::Negative)
                        == RoundSideOrientation::Positive =>
                {
                    ChamberSide::Positive
                }
                (ChamberSide::Positive, Some(anchor))
                    if anchor.0.orient(wall, RoundSideOrientation::Positive)
                        == RoundSideOrientation::Negative =>
                {
                    ChamberSide::Negative
                }
                _ => side,
            };
            next_signature.insert(wall, oriented);
        }

        let next_key = next_signature
            .iter()
            .map(|(&wall, &side)| (wall, side))
            .collect::<ChamberKey>();
        let old_key = aggregates.memberships.get(&entity).cloned();
        if old_key.as_ref() != Some(&next_key) {
            if let Some(old_key) = &old_key {
                let remove = if let Some(count) = aggregates.counts.get_mut(old_key) {
                    *count -= 1;
                    *count == 0
                } else {
                    false
                };
                aggregates.aggregate_updates_last_tick += 1;
                if remove {
                    aggregates.counts.remove(old_key);
                }
            }
            *aggregates.counts.entry(next_key.clone()).or_default() += 1;
            aggregates.aggregate_updates_last_tick += 1;
            for wall in 0..scene.walls.walls().len() {
                let wall = WallId(wall);
                let old_side = old_key
                    .as_ref()
                    .and_then(|key| key.iter().find(|(candidate, _)| *candidate == wall))
                    .map(|(_, side)| *side);
                let next_side = next_signature.get(&wall).copied();
                if old_side != next_side {
                    aggregates.changed_walls.insert(wall);
                }
            }
            aggregates.memberships.insert(entity, next_key);
            aggregates.changed_entities.push(entity);
        }
        signature.0 = next_signature;
    }

    let removed = aggregates
        .memberships
        .keys()
        .copied()
        .filter(|entity| !seen.contains(entity))
        .collect::<Vec<_>>();
    for entity in removed {
        if let Some(old_key) = aggregates.memberships.remove(&entity) {
            let remove = if let Some(count) = aggregates.counts.get_mut(&old_key) {
                *count -= 1;
                *count == 0
            } else {
                false
            };
            aggregates.aggregate_updates_last_tick += 1;
            if remove {
                aggregates.counts.remove(&old_key);
            }
            aggregates.changed_entities.push(entity);
        }
    }
    if !aggregates.changed_entities.is_empty() {
        aggregates.epoch = aggregates.epoch.wrapping_add(1);
    }
    aggregates.contact_frontier = aggregates.changed_walls.clone();
    for contact in &contacts.0 {
        if aggregates.changed_walls.contains(&contact.first)
            || aggregates.changed_walls.contains(&contact.second)
        {
            aggregates.contact_frontier.insert(contact.first);
            aggregates.contact_frontier.insert(contact.second);
        }
    }
}

fn record_transform_history(
    time: Res<Time<Virtual>>,
    mut history: ResMut<TransformHistory>,
    points: Query<(
        Entity,
        &EntityFrame,
        &LocalCoordinates,
        &EuclideanCoordinates,
        Option<&ActiveAnchor>,
    )>,
) {
    let elapsed_seconds = time.elapsed_secs_f64();
    let mut seen = BTreeSet::new();
    for (entity, frame, local, euclidean, anchor) in &points {
        seen.insert(entity);
        let sample = TransformHistorySample {
            elapsed_seconds,
            frame: frame.0,
            local: local.0,
            euclidean: euclidean.0,
            anchor_frame: anchor.map(|anchor| anchor.0.frame),
            flipped_walls: anchor
                .map(|anchor| anchor.0.flipped_walls().iter().copied().collect())
                .unwrap_or_default(),
        };
        let changed = history
            .samples
            .get(&entity)
            .and_then(|samples| samples.back())
            .is_none_or(|previous| {
                previous.frame != sample.frame
                    || previous.local != sample.local
                    || previous.euclidean != sample.euclidean
                    || previous.anchor_frame != sample.anchor_frame
                    || previous.flipped_walls != sample.flipped_walls
            });
        if !changed {
            continue;
        }
        let capacity = history.capacity_per_entity.max(1);
        let samples = history.samples.entry(entity).or_default();
        samples.push_back(sample);
        while samples.len() > capacity {
            samples.pop_front();
        }
        history.epoch = history.epoch.wrapping_add(1);
    }
    let retained = history.samples.len();
    history.samples.retain(|entity, _| seen.contains(entity));
    if history.samples.len() != retained {
        history.epoch = history.epoch.wrapping_add(1);
    }
}

fn extract_hyperscope_packets(
    scene: Res<ConformalScene>,
    mut extraction: ResMut<HyperscopeExtraction>,
    mut diagnostics: ResMut<HyperscapeDiagnostics>,
    subjects: Query<(Entity, &EntityFrame, Option<&EuclideanModelMatrix>), With<RenderSubject>>,
    cameras: Query<(
        Entity,
        &ProjectionCamera,
        Option<&EuclideanModelMatrix>,
        Option<&TrackedCoordinates>,
    )>,
) {
    extraction.0.clear();
    for (subject, subject_frame, model) in &subjects {
        for (camera_entity, camera, camera_model, camera_target) in &cameras {
            match scene.frames.relative_chain(subject_frame.0, camera.frame) {
                Ok(chain) => match chain.to_mobius() {
                    Ok(mobius) => {
                        let euclidean_model = model.copied().unwrap_or_default().0;
                        let origin = quilting_core::Quat::from_point(
                            euclidean_model[12] as f64,
                            euclidean_model[13] as f64,
                            euclidean_model[14] as f64,
                        );
                        let denominator = mobius.c * origin + mobius.d;
                        extraction.0.push(HyperscopePacket {
                            subject,
                            camera: camera_entity,
                            mobius: mobius.coefficients_f32(),
                            orientation_sign: chain.orientation_sign(),
                            euclidean_model,
                            camera_eye: camera_model
                                .map(|model| [model.0[12], model.0[13], model.0[14]])
                                .unwrap_or([0.0; 3]),
                            camera_target: camera_target.map(|target| {
                                [target.0[0] as f32, target.0[1] as f32, target.0[2] as f32]
                            }),
                            origin_pole_denominator_norm_sq: denominator.norm_sq(),
                        });
                    }
                    Err(error) => diagnostics.0.push(format!(
                        "could not collapse render chain for {subject:?}: {error}"
                    )),
                },
                Err(error) => diagnostics.0.push(format!(
                    "could not extract {subject:?} for {camera_entity:?}: {error}"
                )),
            }
        }
    }
    extraction
        .0
        .sort_by_key(|packet| (packet.camera.to_bits(), packet.subject.to_bits()));
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyperscape_protocol::{AssetEntityId, AssetId};
    use quilting_core::{
        ConformalGenerator, ConformalTransformChain, Quat, RoundWall, RoundWallGeometry,
    };
    use std::time::Duration;

    #[test]
    fn runtime_identity_round_trips_the_validated_wire_identity() {
        let wire = EntityId::from_u128(42).unwrap();
        let runtime = StableEntityId::from(wire);
        assert_eq!(EntityId::try_from(runtime).unwrap(), wire);
        assert!(EntityId::try_from(StableEntityId(Uuid::nil())).is_err());
    }

    const EPS: f64 = 1.0e-7;

    fn selection_identity(entity: u128) -> AssetEntityId {
        AssetEntityId::new(
            AssetId::from_u128(0x6000).unwrap(),
            EntityId::from_u128(entity).unwrap(),
        )
        .unwrap()
    }

    fn chain(generators: Vec<ConformalGenerator>) -> ConformalTransformChain {
        ConformalTransformChain::new(generators).unwrap()
    }

    fn assert_point_close(actual: [f64; 3], expected: [f64; 3]) {
        for axis in 0..3 {
            assert!(
                (actual[axis] - expected[axis]).abs() < EPS,
                "axis {axis}: actual={actual:?}, expected={expected:?}"
            );
        }
    }

    fn test_app(scene: ConformalScene) -> App {
        let mut app = App::new();
        app.add_plugins(HyperscapePlugin)
            .insert_resource(scene)
            .insert_resource(Time::<Virtual>::from_max_delta(Duration::MAX));
        app
    }

    fn tick(app: &mut App, delta: Duration) {
        app.world_mut()
            .resource_mut::<Time<Virtual>>()
            .advance_by(delta);
        app.update();
    }

    #[test]
    fn focus_navigation_anchors_animates_and_detaches_without_losing_the_sphere() {
        let identity = selection_identity(7);
        let bound = FocusSphere::new([1.0, 2.0, 3.0], 2.0).unwrap();
        let mut focus = FocusNavigation::default();
        focus.anchor_to(identity, bound, 1.1, 1.0).unwrap();
        assert_eq!(focus.anchor.unwrap().identity, identity);
        assert!(focus.focus_enabled);

        assert!(focus.advance(0.5));
        assert_eq!(focus.sphere.center, [0.75, 1.0, 1.5]);
        assert!((focus.sphere.radius - (4.4_f64).sqrt()).abs() < EPS);
        assert!(focus.advance(0.5));
        assert_eq!(
            focus.sphere,
            FocusSphere {
                center: [1.0, 2.0, 3.0],
                radius: 2.2
            }
        );

        focus.detach();
        let detached = focus.sphere;
        assert!(focus.translate_free([1.0, 0.0, -1.0]));
        assert_eq!(focus.sphere.center, [2.0, 2.0, 2.0]);
        assert_eq!(focus.sphere.radius, detached.radius);
        assert!(focus.focus_enabled);
    }

    #[test]
    fn anchored_focus_edits_only_a_bounded_margin() {
        let identity = selection_identity(11);
        let bound = FocusSphere::new([3.0, 4.0, 5.0], 2.0).unwrap();
        let mut focus = FocusNavigation::default();
        let initial = focus.clone();
        let invalid_identity: AssetEntityId = serde_json::from_value(serde_json::json!({
            "asset": AssetId::from_u128(0x6000).unwrap(),
            "entity": Uuid::nil(),
        }))
        .unwrap();
        assert_eq!(
            focus.anchor_to(invalid_identity, bound, 1.1, 0.0),
            Err("focus anchor must have non-nil asset and entity identities")
        );
        assert_eq!(focus, initial);
        focus.anchor_to(identity, bound, 1.1, 0.0).unwrap();
        assert!(!focus.translate_free([1.0, 0.0, 0.0]));
        assert_eq!(focus.sphere.center, bound.center);
        assert!(focus.scale_radius(100.0));
        assert_eq!(focus.sphere.radius, 8.0);
        assert!(focus.scale_radius(0.001));
        assert_eq!(focus.sphere.radius, 2.0);
    }

    #[test]
    fn spheroidal_focus_uses_inversion_symmetric_geodesic_radius() {
        let sphere = FocusSphere::new([1.0, 0.0, 0.0], 2.0).unwrap();
        assert_eq!(
            sphere
                .compactified_radial_coordinate([1.0, 0.0, 0.0])
                .unwrap(),
            0.0
        );
        assert_eq!(
            sphere
                .compactified_radial_coordinate([3.0, 0.0, 0.0])
                .unwrap(),
            0.5
        );

        let outside = sphere
            .compactified_radial_coordinate([9.0, 0.0, 0.0])
            .unwrap();
        let reflected = sphere
            .compactified_radial_coordinate([1.5, 0.0, 0.0])
            .unwrap();
        assert!((outside + reflected - 1.0).abs() < EPS);

        let focus = FocusNavigation {
            sphere,
            focus_enabled: true,
            ..FocusNavigation::default()
        };
        assert_eq!(focus.defocus_at([3.0, 0.0, 0.0]).unwrap(), 0.0);
        let nearer = focus.defocus_at([1.0, 0.0, 0.0]).unwrap();
        let farther = focus.defocus_at([f64::INFINITY, 0.0, 0.0]);
        assert!(nearer > 0.0);
        assert!(farther.is_err());
    }

    #[test]
    fn deterministic_tick_animates_resolves_tracks_and_extracts() {
        let mut frames = ConformalFrameForest::new();
        let world = frames
            .add_frame("world", None, ConformalTransformChain::identity())
            .unwrap();
        let object_frame = frames
            .add_frame(
                "object",
                Some(world),
                chain(vec![
                    ConformalGenerator::uniform_scale(2.0),
                    ConformalGenerator::translation([10.0, 0.0, 0.0]),
                ]),
            )
            .unwrap();
        let view_frame = frames
            .add_frame(
                "view",
                Some(world),
                chain(vec![ConformalGenerator::translation([4.0, 0.0, 0.0])]),
            )
            .unwrap();
        let expected_relative = frames.relative_chain(object_frame, view_frame).unwrap();
        let mut app = test_app(ConformalScene {
            frames,
            walls: RoundWallSet::new(),
        });
        let target = app
            .world_mut()
            .spawn((
                EntityFrame(object_frame),
                LocalCoordinates([0.0; 3]),
                EuclideanCoordinates([0.0; 3]),
                ConformalPath {
                    keyframes: vec![
                        PathKeyframe {
                            time_seconds: 0.0,
                            point: [0.0; 3],
                        },
                        PathKeyframe {
                            time_seconds: 2.0,
                            point: [2.0, 0.0, 0.0],
                        },
                    ],
                    looping: false,
                },
                RenderSubject,
            ))
            .id();
        let camera = app
            .world_mut()
            .spawn((
                EntityFrame(view_frame),
                LocalCoordinates([0.0; 3]),
                EuclideanCoordinates([0.0; 3]),
                ProjectionCamera { frame: view_frame },
                CrossFrameTarget {
                    target,
                    local_offset: [0.0, 1.0, 0.0],
                },
                TrackedCoordinates([0.0; 3]),
            ))
            .id();

        // Hyperscape's authored clock is advanced explicitly, independently
        // of wall-clock time.
        app.update();
        tick(&mut app, Duration::from_secs(1));

        assert_point_close(
            app.world().get::<LocalCoordinates>(target).unwrap().0,
            [1.0, 0.0, 0.0],
        );
        assert_point_close(
            app.world().get::<EuclideanCoordinates>(target).unwrap().0,
            [12.0, 0.0, 0.0],
        );
        assert_point_close(
            app.world().get::<TrackedCoordinates>(camera).unwrap().0,
            [8.0, 1.0, 0.0],
        );
        let extraction = app.world().resource::<HyperscopeExtraction>();
        assert_eq!(extraction.0.len(), 1);
        assert_eq!(extraction.0[0].subject, target);
        assert_eq!(extraction.0[0].camera, camera);
        let packet = extraction.0[0].mobius;
        let mobius = quilting_core::Mobius::new(
            Quat::new(
                packet[0] as f64,
                packet[1] as f64,
                packet[2] as f64,
                packet[3] as f64,
            ),
            Quat::new(
                packet[4] as f64,
                packet[5] as f64,
                packet[6] as f64,
                packet[7] as f64,
            ),
            Quat::new(
                packet[8] as f64,
                packet[9] as f64,
                packet[10] as f64,
                packet[11] as f64,
            ),
            Quat::new(
                packet[12] as f64,
                packet[13] as f64,
                packet[14] as f64,
                packet[15] as f64,
            ),
        );
        assert_point_close(
            mobius.apply(Quat::from_point(1.0, 0.0, 0.0)).to_point(),
            expected_relative.apply_point([1.0, 0.0, 0.0]).unwrap(),
        );
        assert!(app.world().resource::<HyperscapeDiagnostics>().0.is_empty());
    }

    #[test]
    fn path_timeline_enters_reanchors_and_exits_without_ambient_jump() {
        let mut frames = ConformalFrameForest::new();
        let world = frames
            .add_frame("world", None, ConformalTransformChain::identity())
            .unwrap();
        let room = frames
            .add_frame(
                "room",
                Some(world),
                chain(vec![ConformalGenerator::translation([10.0, 0.0, 0.0])]),
            )
            .unwrap();
        let mut room_anchor = AnchorState::new(room);
        room_anchor.flip(WallId(0));
        let mut app = test_app(ConformalScene {
            frames,
            walls: RoundWallSet::new(),
        });
        let subject = app
            .world_mut()
            .spawn((
                EntityFrame(world),
                LocalCoordinates([0.0; 3]),
                EuclideanCoordinates([0.0; 3]),
                ActiveAnchor(AnchorState::new(world)),
                ChamberSignature::default(),
                ConformalPath {
                    keyframes: vec![
                        PathKeyframe {
                            time_seconds: 0.0,
                            point: [0.0; 3],
                        },
                        PathKeyframe {
                            time_seconds: 3.0,
                            point: [3.0, 0.0, 0.0],
                        },
                    ],
                    looping: false,
                },
                ConformalPathTimeline {
                    coordinate_frame: world,
                    initial_frame: world,
                    initial_anchor: AnchorState::new(world),
                    transitions: vec![
                        PathTransition {
                            time_seconds: 1.0,
                            frame: room,
                            anchor: room_anchor.clone(),
                        },
                        PathTransition {
                            time_seconds: 2.0,
                            frame: world,
                            anchor: AnchorState::new(world),
                        },
                    ],
                },
            ))
            .id();

        app.update();
        tick(&mut app, Duration::from_secs(1));
        assert_eq!(
            app.world().get::<EntityFrame>(subject),
            Some(&EntityFrame(room))
        );
        assert_point_close(
            app.world().get::<LocalCoordinates>(subject).unwrap().0,
            [-9.0, 0.0, 0.0],
        );
        assert_point_close(
            app.world().get::<EuclideanCoordinates>(subject).unwrap().0,
            [1.0, 0.0, 0.0],
        );
        assert_eq!(
            app.world().get::<ActiveAnchor>(subject).unwrap().0,
            room_anchor
        );

        tick(&mut app, Duration::from_secs(1));
        assert_eq!(
            app.world().get::<EntityFrame>(subject),
            Some(&EntityFrame(world))
        );
        assert_point_close(
            app.world().get::<LocalCoordinates>(subject).unwrap().0,
            [2.0, 0.0, 0.0],
        );
        assert_point_close(
            app.world().get::<EuclideanCoordinates>(subject).unwrap().0,
            [2.0, 0.0, 0.0],
        );
        assert!(app.world().resource::<HyperscapeDiagnostics>().0.is_empty());
    }

    #[test]
    fn queued_reparent_preserves_world_mapping() {
        let mut frames = ConformalFrameForest::new();
        let left = frames
            .add_frame(
                "left",
                None,
                chain(vec![ConformalGenerator::translation([10.0, 0.0, 0.0])]),
            )
            .unwrap();
        let right = frames
            .add_frame(
                "right",
                None,
                chain(vec![ConformalGenerator::translation([-5.0, 0.0, 0.0])]),
            )
            .unwrap();
        let moving = frames
            .add_frame(
                "moving",
                Some(left),
                chain(vec![ConformalGenerator::uniform_scale(2.0)]),
            )
            .unwrap();
        let before = frames
            .world_chain(moving)
            .unwrap()
            .apply_point([0.5, 0.0, 0.0])
            .unwrap();
        let mut app = test_app(ConformalScene {
            frames,
            walls: RoundWallSet::new(),
        });
        app.world_mut()
            .resource_mut::<FrameReparentRequests>()
            .0
            .push(FrameReparentRequest {
                frame: moving,
                new_parent: Some(right),
            });
        app.update();
        let scene = app.world().resource::<ConformalScene>();
        let after = scene
            .frames
            .world_chain(moving)
            .unwrap()
            .apply_point([0.5, 0.0, 0.0])
            .unwrap();
        assert_point_close(after, before);
        assert_eq!(scene.frames.frame(moving).unwrap().parent, Some(right));
    }

    #[test]
    fn anchor_xor_reorients_chamber_without_changing_wall() {
        let mut frames = ConformalFrameForest::new();
        let world = frames
            .add_frame("world", None, ConformalTransformChain::identity())
            .unwrap();
        let mut walls = RoundWallSet::new();
        let wall = walls
            .add_wall(
                &frames,
                RoundWall {
                    name: "unit sphere".into(),
                    frame: world,
                    geometry: RoundWallGeometry::sphere([0.0; 3], 1.0).unwrap(),
                },
            )
            .unwrap();
        let crossing_wall = walls
            .add_wall(
                &frames,
                RoundWall {
                    name: "crossing sphere".into(),
                    frame: world,
                    geometry: RoundWallGeometry::sphere([1.0, 0.0, 0.0], 1.0).unwrap(),
                },
            )
            .unwrap();
        let mut app = test_app(ConformalScene { frames, walls });
        let anchor = app
            .world_mut()
            .spawn((
                EntityFrame(world),
                LocalCoordinates([0.0; 3]),
                EuclideanCoordinates([0.0; 3]),
                ActiveAnchor(AnchorState::new(world)),
                ChamberSignature::default(),
            ))
            .id();
        app.update();
        assert_eq!(
            app.world().get::<ChamberSignature>(anchor).unwrap().0[&wall],
            ChamberSide::Negative
        );
        let initial_epoch = app.world().resource::<ChamberAggregateState>().epoch;
        assert_eq!(initial_epoch, 1);
        assert_eq!(
            app.world()
                .resource::<ChamberAggregateState>()
                .aggregate_updates_last_tick,
            1
        );
        assert_eq!(
            app.world().resource::<TransformHistory>().samples[&anchor].len(),
            1
        );

        app.update();
        let aggregates = app.world().resource::<ChamberAggregateState>();
        assert_eq!(aggregates.epoch, initial_epoch);
        assert_eq!(aggregates.aggregate_updates_last_tick, 0);
        assert!(aggregates.changed_entities.is_empty());
        assert_eq!(
            app.world().resource::<TransformHistory>().samples[&anchor].len(),
            1
        );
        app.world_mut()
            .resource_mut::<AnchorFlipRequests>()
            .0
            .push(AnchorFlipRequest {
                anchor,
                walls: BTreeSet::from([wall]),
            });
        app.update();
        assert_eq!(
            app.world().get::<ChamberSignature>(anchor).unwrap().0[&wall],
            ChamberSide::Positive
        );
        let aggregates = app.world().resource::<ChamberAggregateState>();
        assert_eq!(aggregates.epoch, initial_epoch + 1);
        assert_eq!(aggregates.aggregate_updates_last_tick, 2);
        assert_eq!(aggregates.changed_walls, BTreeSet::from([wall]));
        assert_eq!(
            aggregates.contact_frontier,
            BTreeSet::from([wall, crossing_wall])
        );
        assert_eq!(aggregates.counts.values().sum::<usize>(), 1);
        assert_eq!(
            app.world().resource::<TransformHistory>().samples[&anchor].len(),
            2
        );
        assert_eq!(
            app.world().resource::<ConformalScene>().walls.walls()[0]
                .geometry
                .signed_value([0.0; 3])
                .unwrap(),
            -1.0
        );

        let epoch_after_flip = aggregates.epoch;
        app.world_mut().despawn(anchor);
        app.update();
        let aggregates = app.world().resource::<ChamberAggregateState>();
        assert_eq!(aggregates.epoch, epoch_after_flip + 1);
        assert!(aggregates.counts.is_empty());
        assert_eq!(aggregates.aggregate_updates_last_tick, 1);
        assert!(!app
            .world()
            .resource::<TransformHistory>()
            .samples
            .contains_key(&anchor));
    }

    #[test]
    fn cross_frame_contacts_remain_explicitly_unclassified() {
        let mut frames = ConformalFrameForest::new();
        let first_frame = frames
            .add_frame("first", None, ConformalTransformChain::identity())
            .unwrap();
        let second_frame = frames
            .add_frame("second", None, ConformalTransformChain::identity())
            .unwrap();
        let mut walls = RoundWallSet::new();
        for (name, frame) in [("first", first_frame), ("second", second_frame)] {
            walls
                .add_wall(
                    &frames,
                    RoundWall {
                        name: name.into(),
                        frame,
                        geometry: RoundWallGeometry::sphere([0.0; 3], 1.0).unwrap(),
                    },
                )
                .unwrap();
        }
        let mut app = test_app(ConformalScene { frames, walls });
        app.update();
        assert_eq!(
            app.world().resource::<ContactState>().0[0].classification,
            ContactClassification::RequiresCommonChart
        );
    }
}
