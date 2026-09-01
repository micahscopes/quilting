use crate::navigation::{
    asset_entity_id, optional_vector3, parse_easing, parse_preset, perspective_lens, preset_name,
    synchronized_navigation_state, vector3, SelectedFocusJsSnapshot,
};
use hyperscape::{
    extract_packed_scene, map_pointer_turntable, map_space_mouse_camera, CameraBasis, CameraRig,
    FocusSphere, InteractionAction, InteractionHit, InteractionPickActivationGate,
    InteractionPickAuthority, InteractionPickAuthorityDisposition, InteractionPickEvidenceObserver,
    InteractionTarget, InteractionTargetSample, InteractionTargetTable, LayerTransform,
    MappedSpaceMouseFrame, NavigationAction, NavigationFrame, NavigationPreset,
    PackedAssetInstance, PackedNodeSource, PackedNodeTransformSource,
    PackedPresentationLayerBinding, PointerTurntableGesture, PointerTurntableInput, Presentation,
    PresentationAsset, PresentationSnapshot, SpaceMouseCameraInput, SpaceMouseMapping,
    SurfaceAnchorTarget, SurfaceWalkControls, TurntableFrame,
};
use hyperscape_protocol::{
    AssetDescriptor, AssetId, AuthoredEnvelope, CameraPresence, EntityId, EphemeralPresence,
    FocusPresence, LocalPeerEnvelope, MessageHeader, MessageId, PeerId, PresenceEnvelope,
    RequestId, CURRENT_PROTOCOL_VERSION,
};
use hyperscope_app::{
    session_node_identity, AnimationAction, AnimationClipCompletionDispatch,
    AnimationClipDescriptor, AnimationClipJobEffect, AnimationClipSelectionCompletion,
    AnimationClipSelectionOutcome, AnimationClipSelectionReadModel, AnimationClock,
    AnimationPoseRequestDisposition, AnimationPoseScheduler, AnimationPoseStamp, AppCommit,
    AppEffect, AppEvent, AppFrameSnapshot, AppStore, AssetFetchJob, AssetJobIdentity,
    AssetLoadCompletion, AssetLoadCompletionDispatch, AssetLoadOutcome, AssetLoadRequest,
    AssetLoadScope, AssetMetadata, AssetReadModel, AssetStatus, AuthoredRevision,
    AuthoringLeaseStatus, CommitDisposition,
    FocusDiagnosticView, FocusPostprocessMode, FocusPostprocessSettings, FrameTick,
    GraphicsPresentationDecision,
    InstalledPrimarySceneReadModel, LocalAuthoringLeaseController, LocalPeerDisposition,
    LocalPeerIngress, LocalPeerLane, LocalPeerReceipt, LocalPresenceAuthoringReadModel,
    NavigationSettings,
    NavigationSettingsSynchronizationDisposition, NavigationSynchronization, PatchLabCompletion,
    PatchLabCompletionDispatch, PatchLabControls, PatchLabEffect, PatchLabEffects,
    PatchLabFailure, PatchLabField, PatchLabGeometryCompletion, PatchLabGeometryOutcome,
    PatchLabHistogramBin, PatchLabLodCompletion, PatchLabLodOutcome, PatchLabLodSummary,
    PatchLabReadModel, PatchLabSessionDispatch, PatchLabSessionIntent, PatchLabShape,
    PresentationAction, PresentationAnimationResidencyBinding,
    PresentationAnimationResidencyDispatch, PrimarySceneInstallCompletion,
    PrimarySceneInstallCompletionDispatch, PrimarySceneInstallMetadata,
    PrimarySceneInstallOutcome, RenderSettings,
    RenderSettingsSynchronizationDisposition, SemanticAction, Timed,
    WebGpuLodAuthority, WebGpuLodAuthorityDisposition, WebGpuLodAuthorityPhase,
    WebGpuLodAuthorityReason, WebGpuLodAuthoritySnapshot, WebGpuPresentationEvidence,
};
use quilting_core::{render::RenderStyle, render_evidence::RenderPickEvidenceReport};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::BTreeMap;
use uuid::Uuid;
use wasm_bindgen::prelude::*;

const WEBGPU_LOD_COMPLETE_SCENE: u8 = 1 << 0;
const WEBGPU_LOD_DEVICE_AUTHORITY: u8 = 1 << 1;
const WEBGPU_LOD_INCUMBENT_REQUIRED: u8 = 1 << 2;
const WEBGPU_LOD_INCUMBENT_RESET_PENDING: u8 = 1 << 3;
const WEBGPU_LOD_STALE_COMPLETION: u8 = 1 << 4;
const WEBGPU_LOD_INCUMBENT_RECOVERED: u8 = 1 << 5;

fn webgpu_lod_authority_flags(
    snapshot: WebGpuLodAuthoritySnapshot,
    disposition: WebGpuLodAuthorityDisposition,
    complete_scene: bool,
) -> u8 {
    let mut flags = 0;
    if complete_scene {
        flags |= WEBGPU_LOD_COMPLETE_SCENE;
    }
    if snapshot.suppress_incumbent() {
        flags |= WEBGPU_LOD_DEVICE_AUTHORITY;
    }
    if disposition == WebGpuLodAuthorityDisposition::IncumbentRequired {
        flags |= WEBGPU_LOD_INCUMBENT_REQUIRED;
    }
    if snapshot.incumbent_reset_pending {
        flags |= WEBGPU_LOD_INCUMBENT_RESET_PENDING;
    }
    if disposition == WebGpuLodAuthorityDisposition::IgnoredStale {
        flags |= WEBGPU_LOD_STALE_COMPLETION;
    }
    if disposition == WebGpuLodAuthorityDisposition::IncumbentRecovered {
        flags |= WEBGPU_LOD_INCUMBENT_RECOVERED;
    }
    flags
}

fn webgpu_lod_phase_name(phase: WebGpuLodAuthorityPhase) -> &'static str {
    match phase {
        WebGpuLodAuthorityPhase::AwaitingPresentation => "awaiting-presentation",
        WebGpuLodAuthorityPhase::AwaitingDeviceEpoch => "awaiting-device-epoch",
        WebGpuLodAuthorityPhase::DeviceResident => "device-resident",
        WebGpuLodAuthorityPhase::IncumbentRecovery => "incumbent-recovery",
        WebGpuLodAuthorityPhase::Incumbent => "incumbent",
    }
}

fn webgpu_lod_reason_name(reason: WebGpuLodAuthorityReason) -> &'static str {
    match reason {
        WebGpuLodAuthorityReason::PresentationObserved => "presentation-observed",
        WebGpuLodAuthorityReason::DeviceEpochAccepted => "device-epoch-accepted",
        WebGpuLodAuthorityReason::DevicePrefixAccepted => "device-prefix-accepted",
        WebGpuLodAuthorityReason::PresentationRetired => "presentation-retired",
        WebGpuLodAuthorityReason::DeviceDispatchRejected => "device-dispatch-rejected",
        WebGpuLodAuthorityReason::IncumbentRecovered => "incumbent-recovered",
        WebGpuLodAuthorityReason::SparseIncumbentRejected => "sparse-incumbent-rejected",
        WebGpuLodAuthorityReason::StaleDeviceCompletion => "stale-device-completion",
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowWebGpuLodAuthority {
    revision: String,
    phase: &'static str,
    presentation_authoritative: bool,
    active: bool,
    incumbent_reset_pending: bool,
    pending_dispatch_token: Option<String>,
    pending_dispatch_complete_scene: bool,
    activations: String,
    dispatches: String,
    full_scene_dispatches: String,
    fallback_transitions: String,
    incumbent_recoveries: String,
    stale_device_completions: String,
    sparse_incumbent_rejections: String,
    last_reason: Option<&'static str>,
}

impl From<WebGpuLodAuthoritySnapshot> for ShadowWebGpuLodAuthority {
    fn from(snapshot: WebGpuLodAuthoritySnapshot) -> Self {
        Self {
            revision: snapshot.revision.to_string(),
            phase: webgpu_lod_phase_name(snapshot.phase),
            presentation_authoritative: snapshot.presentation_authoritative,
            active: snapshot.active,
            incumbent_reset_pending: snapshot.incumbent_reset_pending,
            pending_dispatch_token: snapshot
                .pending_dispatch
                .map(|dispatch| dispatch.token.to_string()),
            pending_dispatch_complete_scene: snapshot
                .pending_dispatch
                .is_some_and(|dispatch| dispatch.complete_scene),
            activations: snapshot.activations.to_string(),
            dispatches: snapshot.dispatches.to_string(),
            full_scene_dispatches: snapshot.full_scene_dispatches.to_string(),
            fallback_transitions: snapshot.fallback_transitions.to_string(),
            incumbent_recoveries: snapshot.incumbent_recoveries.to_string(),
            stale_device_completions: snapshot.stale_device_completions.to_string(),
            sparse_incumbent_rejections: snapshot.sparse_incumbent_rejections.to_string(),
            last_reason: snapshot.last_reason.map(webgpu_lod_reason_name),
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowGraphicsPresentationDecision {
    phase: &'static str,
    supports_requested_style: bool,
    failed: bool,
    present_webgpu: bool,
    device_lod_recovery_eligible: bool,
    device_lod_authority_eligible: bool,
}

impl From<GraphicsPresentationDecision> for ShadowGraphicsPresentationDecision {
    fn from(decision: GraphicsPresentationDecision) -> Self {
        Self {
            phase: decision.phase.wire_name(),
            supports_requested_style: decision.supports_requested_style,
            failed: decision.failed,
            present_webgpu: decision.present_webgpu,
            device_lod_recovery_eligible: decision.device_lod_recovery_eligible,
            device_lod_authority_eligible: decision.device_lod_authority_eligible,
        }
    }
}

fn browser_render_style(name: &str) -> Option<RenderStyle> {
    RenderStyle::from_wire_name(match name {
        "both" => "matcap_wire",
        name => name,
    })
}

/// Pure generated-WASM oracle for the normalized SpaceMouse camera boundary.
/// This does not queue an action, advance virtual time, or mutate app state.
#[wasm_bindgen(js_name = mapSpaceMouseCameraFrame)]
#[allow(clippy::too_many_arguments)]
pub fn map_space_mouse_camera_frame(
    normalized_axes: &[f32],
    preset: &str,
    swap_yz: bool,
    invert_pan: f64,
    invert_rotate: f64,
    delta_seconds: f64,
    registered_linear_speed: f64,
    move_gain: f64,
    rotate_gain: f64,
    horizon_lock_requested: bool,
) -> Result<JsValue, JsValue> {
    let mapped = space_mouse_camera_input(
        normalized_axes,
        preset,
        swap_yz,
        invert_pan,
        invert_rotate,
        delta_seconds,
        registered_linear_speed,
        move_gain,
        rotate_gain,
        horizon_lock_requested,
    )?;
    to_js(&ShadowMappedSpaceMouseFrame::from(mapped))
}

/// Pure generated-WASM oracle for the incumbent pointer turntable response.
/// Gesture 0 is orbit, 1 is shift-pan, and 2 is wheel dolly.
#[wasm_bindgen(js_name = mapPointerTurntableFrame)]
pub fn map_pointer_turntable_frame(
    delta_x: f64,
    delta_y: f64,
    gesture: u8,
    control_distance: f64,
) -> Result<JsValue, JsValue> {
    let frame = pointer_turntable_input(delta_x, delta_y, gesture, control_distance)?;
    to_js(&ShadowTurntableFrame::from(frame))
}

/// Encode one short-lived browser viewport sample through the canonical Rust
/// protocol model. The browser still supplies its incumbent semantic camera
/// until navigation authority moves into [`HyperscopeAppShadow`]; this
/// boundary owns UUID/u64 parsing, presence-lane validation, and exact JSON.
/// It never dispatches an application event or makes the sample durable.
#[wasm_bindgen(js_name = encodeLocalPresenceEnvelope)]
#[allow(clippy::too_many_arguments)]
pub fn encode_local_presence_envelope(
    message_id: &str,
    sender: &str,
    sequence: &str,
    ttl_millis: u32,
    eye: &[f64],
    forward: &[f64],
    up: &[f64],
    selection_json: &str,
    include_focus: bool,
    focus_center: &[f64],
    focus_radius: f64,
    inversion_enabled: bool,
    active_cue: &str,
    animation_seconds: f64,
) -> Result<String, JsValue> {
    let selection = serde_json::from_str::<Vec<EntityId>>(selection_json)
        .map_err(|error| js_error(format!("presence selection is invalid JSON: {error}")))?;
    let active_cue = if active_cue.is_empty() {
        None
    } else {
        Some(MessageId::new(parse_uuid(active_cue, "active cue ID")?).map_err(js_error)?)
    };
    let animation_seconds = if animation_seconds < 0.0 {
        None
    } else {
        Some(animation_seconds)
    };
    let focus = if include_focus {
        Some(FocusPresence {
            center: vector3(focus_center, "presence focus center")?,
            radius: focus_radius,
            inversion_enabled,
        })
    } else {
        None
    };
    let envelope = PresenceEnvelope {
        header: MessageHeader {
            version: CURRENT_PROTOCOL_VERSION,
            message_id: MessageId::new(parse_uuid(message_id, "presence message ID")?)
                .map_err(js_error)?,
            sender: PeerId::new(parse_uuid(sender, "presence sender ID")?)
                .map_err(js_error)?,
            sequence: sequence.parse::<u64>().map_err(|error| {
                js_error(format!("presence sequence is invalid decimal u64: {error}"))
            })?,
        },
        presence: EphemeralPresence {
            ttl_millis,
            camera: Some(CameraPresence {
                eye: vector3(eye, "presence camera eye")?,
                forward: vector3(forward, "presence camera forward")?,
                up: vector3(up, "presence camera up")?,
            }),
            selection,
            authoring_leases: Vec::new(),
            focus,
            active_cue,
            animation_seconds,
        },
    };
    envelope.validate().map_err(js_error)?;
    serde_json::to_string(&envelope)
        .map_err(|error| js_error(format!("presence envelope could not be encoded: {error}")))
}

/// Opt-in WASM adapter for comparing browser asset jobs with the Rust app
/// reducer. It observes fetch/file acquisition only and never loads a model,
/// mutates the renderer, or chooses a browser transport.
#[wasm_bindgen]
pub struct HyperscopeAppShadow {
    store: AppStore,
    /// Process-local renderer authority. It owns no GPU handles or durable
    /// state; adapters report only presentation/dispatch/recovery evidence.
    webgpu_lod_authority: RefCell<WebGpuLodAuthority>,
    peer_ingress: RefCell<LocalPeerIngress>,
    local_authoring_leases: RefCell<LocalAuthoringLeaseController>,
    animation_pose_scheduler: RefCell<AnimationPoseScheduler>,
    /// Renderer residency joins transient packed-node handles to stable
    /// semantic identity without making either JavaScript or AppState own the
    /// backend-local handles.
    interaction_targets: RefCell<InteractionTargetTable>,
    /// Bounded process-local parity telemetry for retained renderer queries.
    /// It observes the interaction target table's epoch but cannot dispatch an
    /// interaction or mutate authored/application state.
    backend_pick_evidence: RefCell<InteractionPickEvidenceObserver>,
    /// Newest-request gate for the asynchronous WebGPU interaction lane. GPU
    /// resources stay in the renderer; this owns only process-local authority.
    backend_pick_authority: RefCell<InteractionPickAuthority>,
    /// At most one already validated semantic hit may cross the asynchronous
    /// query/activation boundary. It is invalidated by a newer query or target
    /// residency and consumed by an exact primary activation.
    backend_pick_activation: RefCell<InteractionPickActivationGate>,
    /// Effects generated by the allocation-sensitive frame lane. Capacity is
    /// retained across drains so the ordinary no-effect frame stays free of
    /// JavaScript objects and heap churn.
    pending_adapter_effects: RefCell<Vec<AppEffect>>,
}

#[wasm_bindgen]
impl HyperscopeAppShadow {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            store: AppStore::default(),
            webgpu_lod_authority: RefCell::new(WebGpuLodAuthority::default()),
            peer_ingress: RefCell::new(LocalPeerIngress::default()),
            local_authoring_leases: RefCell::new(LocalAuthoringLeaseController::default()),
            animation_pose_scheduler: RefCell::new(AnimationPoseScheduler::default()),
            interaction_targets: RefCell::new(InteractionTargetTable::default()),
            backend_pick_evidence: RefCell::new(InteractionPickEvidenceObserver::default()),
            backend_pick_authority: RefCell::new(InteractionPickAuthority::default()),
            backend_pick_activation: RefCell::new(InteractionPickActivationGate::default()),
            pending_adapter_effects: RefCell::new(Vec::new()),
        }
    }

    /// Interpret renderer residency into one backend-neutral presentation
    /// decision. The browser remains responsible only for observing platform
    /// facts and applying the returned decision to its canvases.
    #[wasm_bindgen(js_name = resolveGraphicsBackendPresentation)]
    #[allow(clippy::too_many_arguments)]
    pub fn resolve_graphics_backend_presentation(
        &self,
        live_presentation_requested: bool,
        requested_style: &str,
        presentation_armed: bool,
        backend_state: &str,
        surface_ready: bool,
        pbr_presentation_ready: bool,
        focus_postprocess_requested: bool,
        focus_presentation_ready: bool,
        frame_admitted: bool,
        has_presented_frame: bool,
        surface_lost: bool,
        presented_style: &str,
    ) -> Result<JsValue, JsValue> {
        let decision = hyperscope_app::resolve_graphics_presentation(
            WebGpuPresentationEvidence {
                live_presentation_requested,
                requested_style: browser_render_style(requested_style),
                presentation_armed,
                backend_ready: backend_state == "ready",
                backend_failed: backend_state == "failed" || backend_state == "lost",
                surface_ready,
                pbr_presentation_ready,
                focus_postprocess_requested,
                focus_presentation_ready,
                frame_admitted,
                has_presented_frame,
                surface_lost,
                presented_style: browser_render_style(presented_style),
            },
        );
        to_js(&ShadowGraphicsPresentationDecision::from(decision))
    }

    /// Observe whether WebGPU is the actual visible presenter. The compact
    /// result is a bit packet decoded by the browser adapter; no JS object is
    /// allocated on this transition path.
    #[wasm_bindgen(js_name = observeWebGpuLodPresentation)]
    pub fn observe_webgpu_lod_presentation(&self, authoritative: bool) -> Result<u8, JsValue> {
        let transition = self
            .webgpu_lod_authority
            .borrow_mut()
            .observe_presentation(authoritative)
            .map_err(js_error)?;
        Ok(webgpu_lod_authority_flags(
            transition.snapshot,
            transition.disposition,
            false,
        ))
    }

    /// Start one synchronous device dispatch decision. Bit zero says the
    /// adapter must classify the complete composed scene; the remaining bits
    /// describe the current authority and rollback obligations.
    #[wasm_bindgen(js_name = beginWebGpuLodDispatch)]
    pub fn begin_webgpu_lod_dispatch(
        &self,
        presentation_authoritative: bool,
        complete_scene_required: bool,
    ) -> Result<u8, JsValue> {
        let mut authority = self.webgpu_lod_authority.borrow_mut();
        let observed = authority
            .observe_presentation(presentation_authoritative)
            .map_err(js_error)?;
        let begun = authority
            .begin_dispatch(complete_scene_required)
            .map_err(js_error)?;
        let complete_scene = begun
            .dispatch
            .is_some_and(|dispatch| dispatch.complete_scene);
        let mut flags =
            webgpu_lod_authority_flags(begun.snapshot, begun.disposition, complete_scene);
        if observed.disposition == WebGpuLodAuthorityDisposition::IncumbentRequired {
            flags |= WEBGPU_LOD_INCUMBENT_REQUIRED;
        }
        Ok(flags)
    }

    /// Settle the one synchronous device dispatch begun above. A presentation
    /// change between calls makes the completion stale inside the Rust reducer
    /// instead of allowing an old device epoch to seize authority.
    #[wasm_bindgen(js_name = completeWebGpuLodDispatch)]
    pub fn complete_webgpu_lod_dispatch(&self, accepted: bool) -> Result<u8, JsValue> {
        let mut authority = self.webgpu_lod_authority.borrow_mut();
        let token = authority
            .snapshot()
            .pending_dispatch
            .ok_or_else(|| JsValue::from_str("no WebGPU LOD dispatch is pending"))?
            .token;
        let transition = authority
            .complete_dispatch(token, accepted)
            .map_err(js_error)?;
        Ok(webgpu_lod_authority_flags(
            transition.snapshot,
            transition.disposition,
            false,
        ))
    }

    /// Admit an incumbent recovery only when it is a complete coherent
    /// snapshot. Sparse/delta publications cannot clear a device-authority
    /// rollback obligation.
    #[wasm_bindgen(js_name = completeIncumbentLodRecovery)]
    pub fn complete_incumbent_lod_recovery(&self, full_snapshot: bool) -> Result<u8, JsValue> {
        let transition = self
            .webgpu_lod_authority
            .borrow_mut()
            .complete_incumbent_recovery(full_snapshot)
            .map_err(js_error)?;
        Ok(webgpu_lod_authority_flags(
            transition.snapshot,
            transition.disposition,
            false,
        ))
    }

    /// Low-rate exact diagnostics for parity checks and the inspector. The
    /// high-rate decision methods above remain allocation-free scalar calls.
    #[wasm_bindgen(js_name = webGpuLodAuthorityDiagnostics)]
    pub fn webgpu_lod_authority_diagnostics(&self) -> Result<JsValue, JsValue> {
        to_js(&ShadowWebGpuLodAuthority::from(
            self.webgpu_lod_authority.borrow().snapshot(),
        ))
    }

    /// Allocate one exact animation-pose stamp. Returns 1 when the adapter
    /// should dispatch it immediately and 2 when it replaced the latest
    /// coalesced follow-up behind an existing physical job.
    #[wasm_bindgen(js_name = writeAnimationPoseRequest)]
    pub fn write_animation_pose_request(
        &self,
        clip_time_seconds: f64,
        sample_time_seconds: f64,
        output: &mut [f64],
    ) -> Result<u8, JsValue> {
        validate_animation_pose_output(output)?;
        let request = self
            .animation_pose_scheduler
            .borrow_mut()
            .request(clip_time_seconds, sample_time_seconds)
            .map_err(js_error)?;
        write_animation_pose_stamp(output, request.stamp)?;
        Ok(match request.disposition {
            AnimationPoseRequestDisposition::Dispatch => 1,
            AnimationPoseRequestDisposition::Coalesced => 2,
        })
    }

    /// Allocate an animation-pose stamp from the application frame clock.
    /// Rust-authority adapters supply only renderer clip time; the explicitly
    /// sampled method above remains the JS/shadow parity oracle.
    #[wasm_bindgen(js_name = writeAnimationPoseRequestFromFrame)]
    pub fn write_animation_pose_request_from_frame(
        &self,
        clip_time_seconds: f64,
        output: &mut [f64],
    ) -> Result<u8, JsValue> {
        let sample_time_seconds = self.store.animation_pose_sample_time_seconds();
        self.write_animation_pose_request(clip_time_seconds, sample_time_seconds, output)
    }

    /// Settle the exact physical animation worker job. Returns 0 for a stale
    /// completion, 1 for a matched completion with no follow-up, and 2 after
    /// writing the newest coalesced request that should dispatch next.
    #[wasm_bindgen(js_name = completeAnimationPoseRequest)]
    #[allow(clippy::too_many_arguments)]
    pub fn complete_animation_pose_request(
        &self,
        clip_time_seconds: f64,
        sample_time_seconds: f64,
        revision: u32,
        continuity_epoch: u32,
        evaluation_available: bool,
        output: &mut [f64],
    ) -> Result<u8, JsValue> {
        validate_animation_pose_output(output)?;
        let completion = self
            .animation_pose_scheduler
            .borrow_mut()
            .complete(
                AnimationPoseStamp {
                    clip_time_seconds,
                    sample_time_seconds,
                    revision,
                    continuity_epoch,
                },
                evaluation_available,
            )
            .map_err(js_error)?;
        if !completion.matched_in_flight {
            output.fill(f64::NAN);
            return Ok(0);
        }
        let Some(next) = completion.next else {
            output.fill(f64::NAN);
            return Ok(1);
        };
        write_animation_pose_stamp(output, next)?;
        Ok(2)
    }

    /// Retire the current pose continuity epoch without starting a second
    /// physical worker job. A still-running job settles through the ordinary
    /// completion path before the newest current-epoch sample dispatches.
    #[wasm_bindgen(js_name = rebaseAnimationPoseSchedule)]
    pub fn rebase_animation_pose_schedule(&self) -> Result<u32, JsValue> {
        self.animation_pose_scheduler
            .borrow_mut()
            .rebase()
            .map_err(js_error)
    }

    /// Mount the opt-in Leptos asset-credit island over this controller's
    /// committed AppStore projections. The browser passes only the host
    /// element; Rust retains the reactive subscription and rendering logic.
    #[cfg(feature = "leptos-ui")]
    #[wasm_bindgen(js_name = mountAssetCredits)]
    pub fn mount_asset_credits(&self, parent: web_sys::HtmlElement) {
        hyperscope_web::asset_credits::mount_asset_credits(parent, self.store.clone());
    }

    /// Mount the opt-in Leptos animation control over the committed summary
    /// signal. The view dispatches through AppStore directly; callbacks expose
    /// only committed renderer adaptation and rejection effects.
    #[cfg(feature = "leptos-ui")]
    #[wasm_bindgen(js_name = mountAnimationControl)]
    pub fn mount_animation_control(
        &self,
        parent: web_sys::HtmlElement,
        on_commit: js_sys::Function,
        on_error: js_sys::Function,
    ) {
        hyperscope_web::animation_control::mount_animation_control(
            parent,
            self.store.clone(),
            on_commit,
            on_error,
        );
    }

    /// Mount the Rust-authoritative animation timeline over its compact,
    /// explicitly throttled FRP projection.
    #[cfg(feature = "leptos-ui")]
    #[wasm_bindgen(js_name = mountAnimationTimeline)]
    pub fn mount_animation_timeline(
        &self,
        parent: web_sys::HtmlElement,
        on_commit: js_sys::Function,
        on_error: js_sys::Function,
    ) {
        hyperscope_web::animation_control::mount_animation_timeline(
            parent,
            self.store.clone(),
            on_commit,
            on_error,
        );
    }

    /// Mount the explicit Rust-authority installed-animation selector. The
    /// view dispatches through AppStore; the browser receives only the exact
    /// committed selection/cancellation effects needed to update resources.
    #[cfg(feature = "leptos-ui")]
    #[wasm_bindgen(js_name = mountAnimationClipControl)]
    pub fn mount_animation_clip_control(
        &self,
        parent: web_sys::HtmlElement,
        on_commit: js_sys::Function,
        on_error: js_sys::Function,
    ) {
        hyperscope_web::animation_control::mount_animation_clip_control(
            parent,
            self.store.clone(),
            on_commit,
            on_error,
        );
    }

    /// Mount the read-only Leptos navigation/focus status over the AppStore's
    /// throttled navigation projection. Renderer frames remain on the direct
    /// snapshot lane and never wait for this view.
    #[cfg(feature = "leptos-ui")]
    #[wasm_bindgen(js_name = mountNavigationStatus)]
    pub fn mount_navigation_status(&self, parent: web_sys::HtmlElement) {
        hyperscope_web::navigation_status::mount_navigation_status(parent, self.store.clone());
    }

    /// Mount the read-only stable selection projection. Transient platform
    /// notices remain a browser concern, but selected identity, source bound,
    /// pivot, and exact face come only from committed AppStore state.
    #[cfg(feature = "leptos-ui")]
    #[wasm_bindgen(js_name = mountInteractionStatus)]
    pub fn mount_interaction_status(&self, parent: web_sys::HtmlElement) {
        hyperscope_web::interaction_status::mount_interaction_status(parent, self.store.clone());
    }

    /// Mount the explicit Rust-authority navigation preference controls over
    /// the committed low-rate AppStore projection. The callback receives only
    /// the complete committed packet needed by the browser adapters.
    #[cfg(feature = "leptos-ui")]
    #[wasm_bindgen(js_name = mountNavigationControls)]
    pub fn mount_navigation_controls(
        &self,
        parent: web_sys::HtmlElement,
        on_commit: js_sys::Function,
        on_error: js_sys::Function,
    ) {
        hyperscope_web::navigation_controls::mount_navigation_controls(
            parent,
            self.store.clone(),
            on_commit,
            on_error,
        );
    }

    /// Mount the camera-lens control in the explicit Rust navigation lane.
    /// The view queues a typed navigation action; the browser callback merely
    /// advances the shared application clock and applies its Rust projection.
    #[cfg(feature = "leptos-ui")]
    #[wasm_bindgen(js_name = mountCameraLensControl)]
    pub fn mount_camera_lens_control(
        &self,
        parent: web_sys::HtmlElement,
        on_queue: js_sys::Function,
        on_error: js_sys::Function,
    ) {
        hyperscope_web::camera_controls::mount_camera_lens_control(
            parent,
            self.store.clone(),
            on_queue,
            on_error,
        );
    }

    /// Mount the opt-in Leptos presentation card over the committed
    /// presentation signal. A platform callback synchronizes incumbent input
    /// state, then the view dispatches through AppStore and exposes only
    /// committed renderer adaptation or rejection effects.
    #[cfg(feature = "leptos-ui")]
    #[wasm_bindgen(js_name = mountPresentationCard)]
    pub fn mount_presentation_card(
        &self,
        parent: web_sys::HtmlElement,
        on_prepare: js_sys::Function,
        on_commit: js_sys::Function,
        on_error: js_sys::Function,
    ) {
        hyperscope_web::presentation_card::mount_presentation_card(
            parent,
            self.store.clone(),
            on_prepare,
            on_commit,
            on_error,
        );
    }

    /// Mount the explicit Rust-authority render controls over the AppStore's
    /// committed render projection. The view dispatches directly through the
    /// reducer; the callback receives only the committed value needed by the
    /// incumbent renderer adapter.
    #[cfg(feature = "leptos-ui")]
    #[wasm_bindgen(js_name = mountRenderControls)]
    pub fn mount_render_controls(
        &self,
        parent: web_sys::HtmlElement,
        on_commit: js_sys::Function,
        on_error: js_sys::Function,
    ) {
        hyperscope_web::render_controls::mount_render_controls(
            parent,
            self.store.clone(),
            on_commit,
            on_error,
        );
    }

    /// Mount the focus-composition portion of the same Rust render policy in
    /// its established sidebar section. This view shares AppStore and reducer
    /// authority with `mountRenderControls`.
    #[cfg(feature = "leptos-ui")]
    #[wasm_bindgen(js_name = mountFocusPostprocessControls)]
    pub fn mount_focus_postprocess_controls(
        &self,
        parent: web_sys::HtmlElement,
        on_commit: js_sys::Function,
        on_error: js_sys::Function,
    ) {
        hyperscope_web::render_controls::mount_focus_postprocess_controls(
            parent,
            self.store.clone(),
            on_commit,
            on_error,
        );
    }

    /// Mount the Rust-authoritative Patch Lab control island. User edits
    /// dispatch directly through AppStore; the host receives only committed
    /// backend-neutral jobs and the few explicitly platform-owned actions.
    #[cfg(feature = "leptos-ui")]
    #[wasm_bindgen(js_name = mountPatchLabControls)]
    pub fn mount_patch_lab_controls(
        &self,
        parent: web_sys::HtmlElement,
        on_commit: js_sys::Function,
        on_platform_action: js_sys::Function,
        on_error: js_sys::Function,
    ) {
        hyperscope_web::patch_lab::mount_patch_lab_controls(
            parent,
            self.store.clone(),
            on_commit,
            on_platform_action,
            on_error,
        );
    }

    #[wasm_bindgen(js_name = requestAsset)]
    #[allow(clippy::too_many_arguments)]
    pub fn request_asset(
        &self,
        sequence: u32,
        at_seconds: f64,
        request_id: &str,
        asset_id: &str,
        uri: &str,
        media_type: &str,
    ) -> Result<JsValue, JsValue> {
        self.request_asset_scoped(
            sequence,
            at_seconds,
            request_id,
            asset_id,
            uri,
            media_type,
            AssetLoadScope::Asset,
        )
    }

    /// Request the one browser asset job allowed to replace the renderer's
    /// primary scene. A later call cancels the preceding primary request even
    /// when it names a different asset; presentation-layer requests remain
    /// independent through `requestAsset`.
    #[wasm_bindgen(js_name = requestPrimaryAsset)]
    #[allow(clippy::too_many_arguments)]
    pub fn request_primary_asset(
        &self,
        sequence: u32,
        at_seconds: f64,
        request_id: &str,
        asset_id: &str,
        uri: &str,
        media_type: &str,
    ) -> Result<JsValue, JsValue> {
        self.request_asset_scoped(
            sequence,
            at_seconds,
            request_id,
            asset_id,
            uri,
            media_type,
            AssetLoadScope::PrimaryScene,
        )
    }

    /// Request one platform acquisition through AppStore's local sequence and
    /// typed asset-job authority. Explicitly sequenced request methods remain
    /// available for replay and rollback adapters.
    #[wasm_bindgen(js_name = requestAssetLoad)]
    #[allow(clippy::too_many_arguments)]
    pub fn request_asset_load(
        &self,
        request_id: &str,
        asset_id: &str,
        uri: &str,
        media_type: &str,
        scope: &str,
    ) -> Result<JsValue, JsValue> {
        let scope = match scope {
            "asset" => AssetLoadScope::Asset,
            "primary_scene" => AssetLoadScope::PrimaryScene,
            _ => {
                return Err(JsValue::from_str(
                    "asset scope must be asset or primary_scene",
                ))
            }
        };
        let request = self
            .store
            .request_asset_load(
                request_id_from_str(request_id)?,
                AssetDescriptor {
                    id: asset_id_from_str(asset_id)?,
                    uri: uri.to_owned(),
                    media_type: (!media_type.is_empty()).then(|| media_type.to_owned()),
                    content_digest: None,
                },
                scope,
            )
            .map_err(js_error)?;
        asset_load_request_to_js(request)
    }

    /// Allocate browser-session correlation identities in Rust. An explicit
    /// asset ID preserves authored/presentation identity; an empty value asks
    /// the AppStore to memoize a process-local ID by exact URI.
    #[wasm_bindgen(js_name = requestSessionAssetLoad)]
    #[allow(clippy::too_many_arguments)]
    pub fn request_session_asset_load(
        &self,
        explicit_asset_id: &str,
        uri: &str,
        media_type: &str,
        scope: &str,
    ) -> Result<JsValue, JsValue> {
        let scope = match scope {
            "asset" => AssetLoadScope::Asset,
            "primary_scene" => AssetLoadScope::PrimaryScene,
            _ => {
                return Err(JsValue::from_str(
                    "asset scope must be asset or primary_scene",
                ))
            }
        };
        let explicit_asset_id = (!explicit_asset_id.is_empty())
            .then(|| asset_id_from_str(explicit_asset_id))
            .transpose()?;
        let request = self
            .store
            .request_session_asset_load(
                uri.to_owned(),
                (!media_type.is_empty()).then(|| media_type.to_owned()),
                scope,
                explicit_asset_id,
            )
            .map_err(js_error)?;
        asset_load_request_to_js(request)
    }

    #[wasm_bindgen(js_name = cancelAsset)]
    pub fn cancel_asset(
        &self,
        sequence: u32,
        at_seconds: f64,
        asset_id: &str,
    ) -> Result<JsValue, JsValue> {
        let commit = self
            .store
            .dispatch(AppEvent::Input(Timed {
                sequence: u64::from(sequence),
                at_seconds,
                value: SemanticAction::CancelAsset(asset_id_from_str(asset_id)?),
            }))
            .map_err(js_error)?;
        commit_to_js(&commit)
    }

    #[wasm_bindgen(js_name = completeAssetLoaded)]
    pub fn complete_asset_loaded(
        &self,
        request_id: &str,
        asset_id: &str,
        byte_length: u32,
    ) -> Result<JsValue, JsValue> {
        let dispatch = self.complete_asset(
            request_id,
            asset_id,
            AssetLoadOutcome::Loaded {
                byte_length: byte_length as usize,
                content_digest: None,
                metadata: AssetMetadata::default(),
            },
        )?;
        commit_to_js(&dispatch.commit)
    }

    #[wasm_bindgen(js_name = completeAssetLoadedWithMetadata)]
    pub fn complete_asset_loaded_with_metadata(
        &self,
        request_id: &str,
        asset_id: &str,
        byte_length: u32,
        metadata: JsValue,
    ) -> Result<JsValue, JsValue> {
        let metadata = serde_wasm_bindgen::from_value::<AssetMetadata>(metadata)
            .map_err(|error| js_error(format!("asset metadata is invalid: {error}")))?;
        let dispatch = self.complete_asset(
            request_id,
            asset_id,
            AssetLoadOutcome::Loaded {
                byte_length: byte_length as usize,
                content_digest: None,
                metadata,
            },
        )?;
        commit_to_js(&dispatch.commit)
    }

    #[wasm_bindgen(js_name = completeAssetFailed)]
    pub fn complete_asset_failed(
        &self,
        request_id: &str,
        asset_id: &str,
        code: &str,
        message: &str,
        retryable: bool,
    ) -> Result<JsValue, JsValue> {
        let dispatch = self.complete_asset(
            request_id,
            asset_id,
            AssetLoadOutcome::Failed {
                code: code.to_owned(),
                message: message.to_owned(),
                retryable,
            },
        )?;
        commit_to_js(&dispatch.commit)
    }

    /// Complete one successful acquisition and return the typed primary
    /// install job, if the reducer admitted this as the current primary load.
    #[wasm_bindgen(js_name = finishAssetLoaded)]
    pub fn finish_asset_loaded(
        &self,
        request_id: &str,
        asset_id: &str,
        byte_length: u32,
    ) -> Result<JsValue, JsValue> {
        asset_load_completion_to_js(self.complete_asset(
            request_id,
            asset_id,
            AssetLoadOutcome::Loaded {
                byte_length: byte_length as usize,
                content_digest: None,
                metadata: AssetMetadata::default(),
            },
        )?)
    }

    #[wasm_bindgen(js_name = finishAssetLoadedWithMetadata)]
    pub fn finish_asset_loaded_with_metadata(
        &self,
        request_id: &str,
        asset_id: &str,
        byte_length: u32,
        metadata: JsValue,
    ) -> Result<JsValue, JsValue> {
        let metadata = serde_wasm_bindgen::from_value::<AssetMetadata>(metadata)
            .map_err(|error| js_error(format!("asset metadata is invalid: {error}")))?;
        asset_load_completion_to_js(self.complete_asset(
            request_id,
            asset_id,
            AssetLoadOutcome::Loaded {
                byte_length: byte_length as usize,
                content_digest: None,
                metadata,
            },
        )?)
    }

    #[wasm_bindgen(js_name = finishAssetFailed)]
    pub fn finish_asset_failed(
        &self,
        request_id: &str,
        asset_id: &str,
        code: &str,
        message: &str,
        retryable: bool,
    ) -> Result<JsValue, JsValue> {
        asset_load_completion_to_js(self.complete_asset(
            request_id,
            asset_id,
            AssetLoadOutcome::Failed {
                code: code.to_owned(),
                message: message.to_owned(),
                retryable,
            },
        )?)
    }

    /// Complete the distinct renderer-install job emitted after a primary
    /// asset has decoded. The browser supplies backend-neutral facts only
    /// after upload and activation have both succeeded.
    #[wasm_bindgen(js_name = completePrimarySceneInstalled)]
    pub fn complete_primary_scene_installed(
        &self,
        request_id: &str,
        asset_id: &str,
        num_vertices: u32,
        num_faces: u32,
        animation_clips_json: &str,
    ) -> Result<JsValue, JsValue> {
        let dispatch = self.complete_primary_scene_install_dispatch(
            request_id,
            asset_id,
            PrimarySceneInstallOutcome::Installed(PrimarySceneInstallMetadata {
                num_vertices,
                num_faces,
                animation_clips: decode_animation_clips(animation_clips_json)?,
            }),
        )?;
        commit_to_js(&dispatch.commit)
    }

    /// Complete a renderer-side primary install through the typed application
    /// port, including any obsolete animation clip jobs invalidated by scene
    /// replacement.
    #[wasm_bindgen(js_name = finishPrimarySceneInstalled)]
    pub fn finish_primary_scene_installed(
        &self,
        request_id: &str,
        asset_id: &str,
        num_vertices: u32,
        num_faces: u32,
        animation_clips_json: &str,
    ) -> Result<JsValue, JsValue> {
        primary_scene_install_completion_to_js(self.complete_primary_scene_install_dispatch(
            request_id,
            asset_id,
            PrimarySceneInstallOutcome::Installed(PrimarySceneInstallMetadata {
                num_vertices,
                num_faces,
                animation_clips: decode_animation_clips(animation_clips_json)?,
            }),
        )?)
    }

    #[wasm_bindgen(js_name = completePrimarySceneInstallFailed)]
    pub fn complete_primary_scene_install_failed(
        &self,
        request_id: &str,
        asset_id: &str,
        code: &str,
        message: &str,
        retryable: bool,
    ) -> Result<JsValue, JsValue> {
        let dispatch = self.complete_primary_scene_install_dispatch(
            request_id,
            asset_id,
            PrimarySceneInstallOutcome::Failed {
                code: code.to_owned(),
                message: message.to_owned(),
                retryable,
            },
        )?;
        commit_to_js(&dispatch.commit)
    }

    #[wasm_bindgen(js_name = finishPrimarySceneInstallFailed)]
    pub fn finish_primary_scene_install_failed(
        &self,
        request_id: &str,
        asset_id: &str,
        code: &str,
        message: &str,
        retryable: bool,
    ) -> Result<JsValue, JsValue> {
        primary_scene_install_completion_to_js(self.complete_primary_scene_install_dispatch(
            request_id,
            asset_id,
            PrimarySceneInstallOutcome::Failed {
                code: code.to_owned(),
                message: message.to_owned(),
                retryable,
            },
        )?)
    }

    /// Resolve renderer-local glTF nodes to application-local selection IDs
    /// after Rust has committed the corresponding session asset as ready.
    /// Authored/durable UUIDs use the separate loader metadata path.
    #[wasm_bindgen(js_name = sessionNodeIdentities)]
    pub fn session_node_identities(
        &self,
        asset_id: &str,
        source_nodes: &[i32],
    ) -> Result<JsValue, JsValue> {
        let asset = asset_id_from_str(asset_id)?;
        let ready = self.store.asset_snapshot().into_iter().any(|candidate| {
            candidate.descriptor.id == asset
                && matches!(candidate.status, AssetStatus::Ready { .. })
        });
        if !ready {
            return Err(JsValue::from_str(
                "session selection identity requires a ready AppStore asset",
            ));
        }

        let mut nodes = source_nodes
            .iter()
            .copied()
            .map(|node| {
                u32::try_from(node)
                    .map_err(|_| JsValue::from_str("session selection node must be nonnegative"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        nodes.sort_unstable();
        nodes.dedup();
        let identities = nodes
            .into_iter()
            .map(|source_node| {
                let identity = session_node_identity(asset, source_node);
                SessionNodeIdentity {
                    asset_id: identity.asset.to_string(),
                    entity_id: identity.entity.to_string(),
                    source_node,
                    durable: false,
                }
            })
            .collect::<Vec<_>>();
        to_js(&identities)
    }

    #[wasm_bindgen(js_name = advanceFrame)]
    pub fn advance_frame(
        &self,
        elapsed_seconds: f64,
        delta_seconds: f64,
    ) -> Result<JsValue, JsValue> {
        let commit = self
            .store
            .dispatch(AppEvent::Frame(FrameTick {
                elapsed_seconds,
                delta_seconds,
            }))
            .map_err(js_error)?;
        commit_to_js(&commit)
    }

    /// Advance the high-rate frame lane without serializing an `AppCommit`.
    /// Effects are retained for `drainAdapterEffects`; the returned count lets
    /// an adapter skip that allocation on ordinary frames. Ignoring the count
    /// remains backward compatible, but an adapter that activates a semantic
    /// frame job such as animated Patch Lab LOD must drain it.
    #[wasm_bindgen(js_name = advanceFrameQuiet)]
    pub fn advance_frame_quiet(
        &self,
        elapsed_seconds: f64,
        delta_seconds: f64,
    ) -> Result<u32, JsValue> {
        let commit = self
            .store
            .dispatch(AppEvent::Frame(FrameTick {
                elapsed_seconds,
                delta_seconds,
            }))
            .map_err(js_error)?;
        self.retain_quiet_frame_effects(commit)
    }

    /// Advance from a platform delta while Rust retains the monotonic elapsed
    /// epoch. The absolute-time port above remains available for replay and
    /// js|shadow comparison, but the Rust-authority RAF lane cannot round-trip
    /// or accidentally rewind application time.
    #[wasm_bindgen(js_name = advanceFrameDeltaQuiet)]
    pub fn advance_frame_delta_quiet(&self, delta_seconds: f64) -> Result<u32, JsValue> {
        let commit = self
            .store
            .dispatch_frame_delta(delta_seconds)
            .map_err(js_error)?;
        self.retain_quiet_frame_effects(commit)
    }

    /// Sample the committed monotonic application epoch for low-rate platform
    /// effects. Ordinary RAF adaptation uses `advanceFrameDeltaQuiet` and does
    /// not read or mirror this value.
    #[wasm_bindgen(js_name = frameElapsedSeconds)]
    pub fn frame_elapsed_seconds(&self) -> f64 {
        self.store.frame_elapsed_seconds()
    }

    /// Project one protocol-valid local viewport sample. Transport identity,
    /// sequencing, retry, and delivery remain platform responsibilities.
    #[wasm_bindgen(js_name = localPresenceSample)]
    pub fn local_presence_sample(&self, ttl_millis: u32) -> Result<JsValue, JsValue> {
        let presence = self
            .store
            .local_presence_authoring_snapshot(ttl_millis)
            .map_err(js_error)?;
        to_js(&ShadowLocalPresenceSample::from(presence))
    }

    /// Encode the Rust-authoritative local presence sample and retain advisory
    /// lease IDs across refreshes. The platform supplies only sender/message
    /// ordering and delivers the validated JSON on the presence lane.
    #[wasm_bindgen(js_name = encodeAuthoritativeLocalPresenceEnvelope)]
    pub fn encode_authoritative_local_presence_envelope(
        &self,
        message_id: &str,
        sender: &str,
        sequence: &str,
        ttl_millis: u32,
    ) -> Result<String, JsValue> {
        let message_id = MessageId::new(parse_uuid(message_id, "presence message ID")?)
            .map_err(js_error)?;
        let sender = PeerId::new(parse_uuid(sender, "presence sender ID")?)
            .map_err(js_error)?;
        let sequence = sequence.parse::<u64>().map_err(|error| {
            js_error(format!("presence sequence is invalid decimal u64: {error}"))
        })?;
        let sample = self
            .store
            .local_presence_authoring_snapshot(ttl_millis)
            .map_err(js_error)?;
        let mut presence = sample.presence;
        presence.authoring_leases = self
            .local_authoring_leases
            .try_borrow_mut()
            .map_err(|_| JsValue::from_str("local authoring leases are already active"))?
            .synchronize(sender, sample.authoring_targets, message_id)
            .map_err(js_error)?;
        let envelope = PresenceEnvelope {
            header: MessageHeader {
                version: CURRENT_PROTOCOL_VERSION,
                message_id,
                sender,
                sequence,
            },
            presence,
        };
        envelope.validate().map_err(js_error)?;
        serde_json::to_string(&envelope)
            .map_err(|error| js_error(format!("presence envelope could not be encoded: {error}")))
    }

    /// Resolve stable primary/secondary presentation identity before the
    /// browser allocates decoded buffers or renderer-local packed offsets.
    #[wasm_bindgen(js_name = presentationCompositionPlan)]
    pub fn presentation_composition_plan(&self) -> Result<JsValue, JsValue> {
        let plan = self
            .store
            .presentation_composition_plan()
            .map_err(js_error)?;
        to_js(&ShadowPresentationCompositionPlan {
            revision: plan.revision.to_string(),
            primary: plan.primary,
            secondary: plan.secondary,
        })
    }

    /// Resolve exact manifest URI identity after the platform has selected a
    /// concrete byte source. A missing URI serializes as null/undefined; an
    /// ambiguous manifest is rejected instead of taking first-match order.
    #[wasm_bindgen(js_name = presentationAssetForExactUri)]
    pub fn presentation_asset_for_exact_uri(&self, uri: &str) -> Result<JsValue, JsValue> {
        let asset = self
            .store
            .presentation_asset_for_exact_uri(uri)
            .map_err(js_error)?;
        to_js(&asset)
    }

    /// Serialize and atomically clear effects retained by quiet frame
    /// dispatches. A second drain is empty; job IDs and cancellation order are
    /// preserved exactly as committed by the reducer.
    #[wasm_bindgen(js_name = drainAdapterEffects)]
    pub fn drain_adapter_effects(&self) -> Result<JsValue, JsValue> {
        let effects = std::mem::take(&mut *self.pending_adapter_effects.borrow_mut());
        let effects = effects.iter().map(shadow_effect).collect::<Vec<_>>();
        to_js(&effects)
    }

    /// Drain the frame lane as a typed Patch Lab job list. The generic drain
    /// remains a rollback seam; ordinary frame adaptation rejects and restores
    /// the queue if a future reducer adds a different high-rate effect kind.
    #[wasm_bindgen(js_name = drainPatchLabEffects)]
    pub fn drain_patch_lab_effects(&self) -> Result<JsValue, JsValue> {
        let effects = std::mem::take(&mut *self.pending_adapter_effects.borrow_mut());
        let patch_lab_effects = PatchLabEffects::from_effects(&effects);
        if patch_lab_effects.len() != effects.len() {
            self.pending_adapter_effects.borrow_mut().extend(effects);
            return Err(JsValue::from_str(
                "quiet frame queue contains a non-Patch-Lab adapter effect",
            ));
        }
        to_js(&shadow_patch_lab_effects(&patch_lab_effects))
    }

    #[wasm_bindgen(js_name = pendingAdapterEffectCount)]
    pub fn pending_adapter_effect_count(&self) -> u32 {
        u32::try_from(self.pending_adapter_effects.borrow().len()).unwrap_or(u32::MAX)
    }

    /// Publish all throttled FRP read models at an adapter-selected low-rate
    /// boundary. The summary revision is the final commit fence.
    #[wasm_bindgen(js_name = flushReadModels)]
    pub fn flush_read_models(&self) {
        self.store.flush_read_models();
    }

    /// Publish only the low-rate animation-control projection. This is the
    /// browser RAF throttle boundary, not a renderer or application tick.
    #[wasm_bindgen(js_name = flushAnimationReadModel)]
    pub fn flush_animation_read_model(&self) -> u64 {
        self.store.flush_animation_read_model()
    }

    /// Replace settled camera/focus state before a low-rate authored
    /// transition. The app reducer validates and commits the replacement.
    #[wasm_bindgen(js_name = synchronizeNavigation)]
    #[allow(clippy::too_many_arguments)]
    pub fn synchronize_navigation(
        &self,
        eye: &[f64],
        forward: &[f64],
        up: &[f64],
        control_distance: f64,
        semantic_target: &[f64],
        vertical_fov_radians: f64,
        near: f64,
        far: f64,
        focus_center: &[f64],
        focus_radius: f64,
        focus_enabled: bool,
        inversion_enabled: bool,
        focus_coordinate: f64,
        angular_aperture: f64,
    ) -> Result<JsValue, JsValue> {
        let (camera, focus) = synchronized_navigation_state(
            eye,
            forward,
            up,
            control_distance,
            semantic_target,
            vertical_fov_radians,
            near,
            far,
            focus_center,
            focus_radius,
            focus_enabled,
            inversion_enabled,
            focus_coordinate,
            angular_aperture,
        )?;
        let commit = self
            .store
            .dispatch(AppEvent::NavigationSynchronized(
                NavigationSynchronization { camera, focus },
            ))
            .map_err(js_error)?;
        commit_to_js(&commit)
    }

    #[wasm_bindgen(js_name = setPreset)]
    pub fn set_preset(&self, preset: &str) -> Result<u64, JsValue> {
        self.dispatch_navigation(NavigationAction::SetPreset(parse_preset(preset)?))
    }

    #[wasm_bindgen(js_name = setPerspectiveLens)]
    pub fn set_perspective_lens(
        &self,
        vertical_fov_radians: f64,
        near: f64,
        far: f64,
    ) -> Result<u64, JsValue> {
        self.dispatch_navigation(NavigationAction::SetPerspectiveLens(perspective_lens(
            vertical_fov_radians,
            near,
            far,
        )?))
    }

    #[wasm_bindgen(js_name = setSemanticTargetEnabled)]
    pub fn set_semantic_target_enabled(&self, enabled: bool) -> Result<u64, JsValue> {
        self.dispatch_navigation(NavigationAction::SetSemanticTargetEnabled(enabled))
    }

    #[wasm_bindgen(js_name = applyFrame)]
    pub fn apply_frame(
        &self,
        translation: &[f64],
        rotation: &[f64],
        dolly_log: f64,
        horizon_locked: bool,
    ) -> Result<u64, JsValue> {
        self.dispatch_navigation(NavigationAction::ApplyFrame(NavigationFrame {
            translation: vector3(translation, "camera translation")?,
            rotation: vector3(rotation, "camera rotation")?,
            dolly_log,
            horizon_locked,
        }))
    }

    /// Convert one browser-filtered SpaceMouse sample into semantic navigation
    /// actions using Rust-owned mapping and response policy. WebHID acquisition,
    /// report decoding, response shaping, and gesture-speed registration remain
    /// outside this boundary. This queues actions but never advances time.
    #[wasm_bindgen(js_name = queueSpaceMouseCamera)]
    #[allow(clippy::too_many_arguments)]
    pub fn queue_space_mouse_camera(
        &self,
        normalized_axes: &[f32],
        preset: &str,
        swap_yz: bool,
        invert_pan: f64,
        invert_rotate: f64,
        delta_seconds: f64,
        registered_linear_speed: f64,
        move_gain: f64,
        rotate_gain: f64,
        horizon_lock_requested: bool,
    ) -> Result<JsValue, JsValue> {
        let mapped = space_mouse_camera_input(
            normalized_axes,
            preset,
            swap_yz,
            invert_pan,
            invert_rotate,
            delta_seconds,
            registered_linear_speed,
            move_gain,
            rotate_gain,
            horizon_lock_requested,
        )?;
        let preset_sequence =
            self.dispatch_navigation(NavigationAction::SetPreset(mapped.preset))?;
        let frame_sequence =
            self.dispatch_navigation(NavigationAction::ApplyFrame(mapped.frame))?;
        to_js(&ShadowSpaceMouseDispatch {
            preset_sequence: preset_sequence.to_string(),
            frame_sequence: frame_sequence.to_string(),
            preset: preset_name(mapped.preset),
            frame: ShadowNavigationFrame::from(mapped.frame),
        })
    }

    /// Map and integrate one browser-filtered SpaceMouse sample as a single
    /// device-neutral camera intent, then copy the committed camera into a
    /// retained numeric packet. This avoids allocating a serialized snapshot
    /// on every report while preset, target policy, transition cancellation,
    /// and quaternion integration remain one reducer transaction.
    ///
    /// Packet layout: eye[3], right[3], up[3], forward[3], control distance,
    /// semantic-target-present, semantic target[3].
    #[wasm_bindgen(js_name = stepSpaceMouseCamera)]
    #[allow(clippy::too_many_arguments)]
    pub fn step_space_mouse_camera(
        &self,
        normalized_axes: &[f32],
        preset: &str,
        swap_yz: bool,
        invert_pan: f64,
        invert_rotate: f64,
        delta_seconds: f64,
        registered_linear_speed: f64,
        move_gain: f64,
        rotate_gain: f64,
        horizon_lock_requested: bool,
        output: &mut [f64],
    ) -> Result<(), JsValue> {
        let mapped = space_mouse_camera_input(
            normalized_axes,
            preset,
            swap_yz,
            invert_pan,
            invert_rotate,
            delta_seconds,
            registered_linear_speed,
            move_gain,
            rotate_gain,
            horizon_lock_requested,
        )?;
        self.step_camera_action(
            NavigationAction::ApplyCameraIntent {
                preset: mapped.preset,
                semantic_target_enabled: mapped.preset == NavigationPreset::Object,
                frame: mapped.frame,
            },
            "SpaceMouse",
            output,
        )
    }

    /// Map and integrate one mouse/trackpad gesture through the same retained
    /// camera packet used by SpaceMouse authority. Gesture 0 is orbit, 1 is
    /// shift-pan, and 2 is wheel dolly. The browser supplies only platform
    /// deltas and the current semantic-target policy.
    #[wasm_bindgen(js_name = stepPointerCamera)]
    pub fn step_pointer_camera(
        &self,
        delta_x: f64,
        delta_y: f64,
        gesture: u8,
        semantic_target_enabled: bool,
        output: &mut [f64],
    ) -> Result<(), JsValue> {
        let control_distance = self.store.frame_snapshot().camera.control_distance;
        let frame = pointer_turntable_input(delta_x, delta_y, gesture, control_distance)?;
        self.step_camera_action(
            NavigationAction::ApplyTurntableIntent {
                semantic_target_enabled,
                frame,
            },
            "pointer",
            output,
        )
    }

    #[wasm_bindgen(js_name = transitionCamera)]
    #[allow(clippy::too_many_arguments)]
    pub fn transition_camera(
        &self,
        eye: &[f64],
        forward: &[f64],
        up: &[f64],
        control_distance: f64,
        semantic_target: &[f64],
        duration_seconds: f64,
        easing: &str,
    ) -> Result<u64, JsValue> {
        let target = CameraRig::new(
            vector3(eye, "camera eye")?,
            CameraBasis::from_forward_up(
                vector3(forward, "camera forward")?,
                vector3(up, "camera up")?,
            )
            .map_err(js_error)?,
            control_distance,
            optional_vector3(semantic_target, "camera target")?,
            self.store.frame_snapshot().camera.lens,
        )
        .map_err(js_error)?;
        self.dispatch_navigation(NavigationAction::TransitionCamera {
            target,
            duration_seconds,
            easing: parse_easing(easing)?,
        })
    }

    /// Move the finite control target to the current selected output-chart
    /// pivot through the application reducer and shared virtual clock.
    #[wasm_bindgen(js_name = aimAtSelection)]
    pub fn aim_at_selection(
        &self,
        duration_seconds: f64,
        easing: &str,
    ) -> Result<u64, JsValue> {
        self.dispatch_navigation(NavigationAction::AimAtSelection {
            duration_seconds,
            easing: parse_easing(easing)?,
        })
    }

    /// Frame the currently selected source-chart sphere in the active output
    /// chart through the same atomic application reducer used by replay.
    #[wasm_bindgen(js_name = reframeSelection)]
    pub fn reframe_selection(
        &self,
        viewport_aspect: f64,
        margin: f64,
        duration_seconds: f64,
        easing: &str,
    ) -> Result<u64, JsValue> {
        self.dispatch_navigation(NavigationAction::ReframeSelection {
            viewport_aspect,
            margin,
            duration_seconds,
            easing: parse_easing(easing)?,
        })
    }

    #[wasm_bindgen(js_name = beginSurfaceAnchorTransition)]
    #[allow(clippy::too_many_arguments)]
    pub fn begin_surface_anchor_transition(
        &self,
        eye: &[f64],
        forward: &[f64],
        up: &[f64],
        control_distance: f64,
        normal: &[f64],
        scene_radius: f64,
        duration_seconds: f64,
        easing: &str,
    ) -> Result<u64, JsValue> {
        let target = self.surface_anchor_target(eye, forward, up, control_distance, normal)?;
        self.dispatch_navigation(NavigationAction::BeginSurfaceAnchorTransition {
            target,
            scene_radius,
            duration_seconds,
            easing: parse_easing(easing)?,
        })
    }

    #[wasm_bindgen(js_name = updateSurfaceAnchorTarget)]
    pub fn update_surface_anchor_target(
        &self,
        eye: &[f64],
        forward: &[f64],
        up: &[f64],
        control_distance: f64,
        normal: &[f64],
    ) -> Result<u64, JsValue> {
        let target = self.surface_anchor_target(eye, forward, up, control_distance, normal)?;
        self.dispatch_navigation(NavigationAction::UpdateSurfaceAnchorTarget(target))
    }

    #[wasm_bindgen(js_name = cancelSurfaceAnchorTransition)]
    pub fn cancel_surface_anchor_transition(&self) -> Result<u64, JsValue> {
        self.dispatch_navigation(NavigationAction::CancelSurfaceAnchorTransition)
    }

    #[wasm_bindgen(js_name = setFreeFocusSphere)]
    pub fn set_free_focus_sphere(&self, center: &[f64], radius: f64) -> Result<u64, JsValue> {
        let sphere =
            FocusSphere::new(vector3(center, "focus center")?, radius).map_err(js_error)?;
        self.dispatch_navigation(NavigationAction::SetFreeFocusSphere(sphere))
    }

    /// Queue one absolute device-independent sphere edit. Rust resolves the
    /// edit against the current application focus under the store lock so a
    /// selected anchor becomes a radius-only margin edit, while a detached
    /// request replaces the complete free sphere.
    #[wasm_bindgen(js_name = editFocusSphere)]
    pub fn edit_focus_sphere(
        &self,
        center: &[f64],
        radius: f64,
        preserve_anchor: bool,
    ) -> Result<u64, JsValue> {
        let target =
            FocusSphere::new(vector3(center, "focus center")?, radius).map_err(js_error)?;
        let (sequence, _) = self
            .store
            .dispatch_focus_sphere_edit(target, preserve_anchor)
            .map_err(js_error)?;
        Ok(sequence)
    }

    #[wasm_bindgen(js_name = anchorFocus)]
    #[allow(clippy::too_many_arguments)]
    pub fn anchor_focus(
        &self,
        asset: &str,
        entity: &str,
        source_bound_center: &[f64],
        source_bound_radius: f64,
        source_pivot: &[f64],
        margin: f64,
        duration_seconds: f64,
        easing: &str,
    ) -> Result<u64, JsValue> {
        self.dispatch_navigation(NavigationAction::AnchorFocus {
            identity: asset_entity_id(asset, entity)?,
            source_bound: FocusSphere::new(
                vector3(source_bound_center, "focus source-bound center")?,
                source_bound_radius,
            )
            .map_err(js_error)?,
            source_pivot: vector3(source_pivot, "focus source pivot")?,
            margin,
            duration_seconds,
            easing: parse_easing(easing)?,
        })
    }

    #[wasm_bindgen(js_name = detachFocus)]
    pub fn detach_focus(&self) -> Result<u64, JsValue> {
        self.dispatch_navigation(NavigationAction::DetachFocus)
    }

    /// Clear pointer hover/press and detach selected focus in one AppStore
    /// transaction integrated at the current virtual frame time. The sphere,
    /// focus enablement, and inversion remain unchanged.
    #[wasm_bindgen(js_name = clearSelection)]
    pub fn clear_selection(&self) -> Result<JsValue, JsValue> {
        let mut activation = self
            .backend_pick_activation
            .try_borrow_mut()
            .map_err(|_| js_error("backend pick activation is already borrowed"))?;
        let dispatch = self.store.dispatch_selection_clear().map_err(js_error)?;
        activation.clear();
        drop(activation);
        self.retain_quiet_frame_effects(dispatch.commit)?;
        navigation_to_js(
            self.store.frame_snapshot(),
            self.store.navigation_diagnostics_snapshot(),
        )
    }

    #[wasm_bindgen(js_name = translateFocus)]
    pub fn translate_focus(&self, delta: &[f64]) -> Result<u64, JsValue> {
        self.dispatch_navigation(NavigationAction::TranslateFocus(vector3(
            delta,
            "focus translation",
        )?))
    }

    #[wasm_bindgen(js_name = scaleFocusLog)]
    pub fn scale_focus_log(&self, log_delta: f64) -> Result<u64, JsValue> {
        self.dispatch_navigation(NavigationAction::ScaleFocusLog(log_delta))
    }

    #[wasm_bindgen(js_name = setFocusEnabled)]
    pub fn set_focus_enabled(&self, enabled: bool) -> Result<u64, JsValue> {
        self.dispatch_navigation(NavigationAction::SetFocusEnabled(enabled))
    }

    #[wasm_bindgen(js_name = setFocusField)]
    pub fn set_focus_field(&self, coordinate: f64, angular_aperture: f64) -> Result<u64, JsValue> {
        self.dispatch_navigation(NavigationAction::SetFocusField {
            enabled: None,
            coordinate,
            angular_aperture,
        })
    }

    /// Commit the complete spheroidal focus field as one semantic action.
    #[wasm_bindgen(js_name = setFocusFieldState)]
    pub fn set_focus_field_state(
        &self,
        enabled: bool,
        coordinate: f64,
        angular_aperture: f64,
    ) -> Result<u64, JsValue> {
        self.dispatch_navigation(NavigationAction::SetFocusField {
            enabled: Some(enabled),
            coordinate,
            angular_aperture,
        })
    }

    #[wasm_bindgen(js_name = setInversionEnabled)]
    pub fn set_inversion_enabled(&self, enabled: bool) -> Result<u64, JsValue> {
        self.dispatch_navigation(NavigationAction::SetInversionEnabled(enabled))
    }

    #[wasm_bindgen(js_name = toggleInversion)]
    pub fn toggle_inversion(&self) -> Result<u64, JsValue> {
        self.dispatch_navigation(NavigationAction::ToggleInversion)
    }

    /// Admit the selected-object inversion gesture as one application action.
    /// Rust restarts an existing anchored fit, toggles the chart, and performs
    /// camera/surface transport transactionally before exposing a snapshot.
    #[wasm_bindgen(js_name = refitFocusAndToggleInversion)]
    pub fn refit_focus_and_toggle_inversion(
        &self,
        duration_seconds: f64,
        easing: &str,
    ) -> Result<u64, JsValue> {
        self.dispatch_navigation(NavigationAction::RefitFocusAndToggleInversion {
            duration_seconds,
            easing: parse_easing(easing)?,
        })
    }

    #[wasm_bindgen(js_name = tickNavigation)]
    pub fn tick_navigation(&self, delta_seconds: f64) -> Result<JsValue, JsValue> {
        self.advance_frame_delta_quiet(delta_seconds)?;
        navigation_to_js(
            self.store.frame_snapshot(),
            self.store.navigation_diagnostics_snapshot(),
        )
    }

    #[wasm_bindgen(js_name = navigationSnapshot)]
    pub fn navigation_snapshot(&self) -> Result<JsValue, JsValue> {
        navigation_to_js(
            self.store.frame_snapshot(),
            self.store.navigation_diagnostics_snapshot(),
        )
    }

    /// Atomically replace the renderer-resident interaction join. Packed nodes
    /// are transient WebGL2/WebGPU handles; optional stable identity and source
    /// bounds are validated before the previous table is replaced.
    #[wasm_bindgen(js_name = replaceInteractionTargets)]
    pub fn replace_interaction_targets(&self, targets_json: &str) -> Result<u32, JsValue> {
        let inputs = serde_json::from_str::<Vec<ShadowInteractionTargetInput>>(targets_json)
            .map_err(|error| {
                js_error(format!(
                    "interaction target input is invalid JSON: {error}"
                ))
            })?;
        let targets = inputs
            .into_iter()
            .map(|input| {
                let identity = input
                    .identity
                    .map(|identity| asset_entity_id(&identity.asset_id, &identity.entity_id))
                    .transpose()?;
                let source_bound = FocusSphere::new(
                    input.source_bound.center,
                    input.source_bound.radius,
                )
                .map_err(js_error)?;
                InteractionTarget::new(input.packed_node, identity, source_bound).map_err(js_error)
            })
            .collect::<Result<Vec<_>, JsValue>>()?;
        let epoch = self
            .interaction_targets
            .try_borrow()
            .map_err(|_| JsValue::from_str("interaction targets are already borrowed"))?
            .epoch()
            .checked_add(1)
            .ok_or_else(|| JsValue::from_str("interaction target epoch is exhausted"))?;
        let table = InteractionTargetTable::try_from_epoch(epoch, targets).map_err(js_error)?;
        let mut target_slot = self
            .interaction_targets
            .try_borrow_mut()
            .map_err(|_| JsValue::from_str("interaction targets are already borrowed"))?;
        let mut activation = self
            .backend_pick_activation
            .try_borrow_mut()
            .map_err(|_| JsValue::from_str("backend pick activation is already borrowed"))?;
        *target_slot = table;
        activation.clear();
        Ok(epoch)
    }

    /// Resolve one WebGL2/WebGPU packed-node sample through the current Rust
    /// residency join before it enters semantic interaction state.
    #[wasm_bindgen(js_name = setPackedInteractionHover)]
    #[allow(clippy::too_many_arguments)]
    pub fn set_packed_interaction_hover(
        &self,
        target_epoch: u32,
        packed_node: u32,
        source_pivot: &[f64],
        output_distance: f64,
        face: i32,
        barycentric: &[f64],
    ) -> Result<u64, JsValue> {
        let mut sample = InteractionTargetSample::new_for_epoch(
            target_epoch,
            packed_node,
            vector3(source_pivot, "interaction source pivot")?,
            output_distance,
        )
        .map_err(js_error)?;
        if let Some((face, barycentric)) = interaction_surface(face, barycentric)? {
            sample = sample.with_surface(face, barycentric).map_err(js_error)?;
        }
        let hit = self
            .interaction_targets
            .try_borrow()
            .map_err(|_| JsValue::from_str("interaction targets are being replaced"))?
            .resolve(sample)
            .map_err(js_error)?;
        self.dispatch_interaction(InteractionAction::SetHover(Some(hit)))
    }

    /// Resolve one renderer-local hit through the current semantic target
    /// table and activate it atomically. This is the ordinary WebGL/JS picker
    /// counterpart to the token-checked asynchronous WebGPU boundary.
    #[wasm_bindgen(js_name = activatePackedInteraction)]
    #[allow(clippy::too_many_arguments)]
    pub fn activate_packed_interaction(
        &self,
        target_epoch: u32,
        packed_node: u32,
        source_pivot: &[f64],
        output_distance: f64,
        face: i32,
        barycentric: &[f64],
    ) -> Result<JsValue, JsValue> {
        let mut sample = InteractionTargetSample::new_for_epoch(
            target_epoch,
            packed_node,
            vector3(source_pivot, "interaction source pivot")?,
            output_distance,
        )
        .map_err(js_error)?;
        if let Some((face, barycentric)) = interaction_surface(face, barycentric)? {
            sample = sample.with_surface(face, barycentric).map_err(js_error)?;
        }
        let hit = self
            .interaction_targets
            .try_borrow()
            .map_err(|_| JsValue::from_str("interaction targets are being replaced"))?
            .resolve(sample)
            .map_err(js_error)?;
        self.backend_pick_activation
            .try_borrow_mut()
            .map_err(|_| js_error("backend pick activation is already borrowed"))?
            .clear();
        self.activate_interaction_hit(hit)
    }

    /// Resolve one backend-local pick into the shared interaction vocabulary.
    /// A negative-one face with an empty barycentric slice means the adapter
    /// has entity-level identity only; a nonnegative face requires exactly
    /// three barycentric coordinates.
    #[wasm_bindgen(js_name = setInteractionHover)]
    #[allow(clippy::too_many_arguments)]
    pub fn set_interaction_hover(
        &self,
        asset: &str,
        entity: &str,
        source_bound_center: &[f64],
        source_bound_radius: f64,
        source_pivot: &[f64],
        output_distance: f64,
        face: i32,
        barycentric: &[f64],
    ) -> Result<u64, JsValue> {
        let mut hit = InteractionHit::new(
            asset_entity_id(asset, entity)?,
            FocusSphere::new(
                vector3(source_bound_center, "interaction source-bound center")?,
                source_bound_radius,
            )
            .map_err(js_error)?,
            vector3(source_pivot, "interaction source pivot")?,
            output_distance,
        )
        .map_err(js_error)?;
        if let Some((face, barycentric)) = interaction_surface(face, barycentric)? {
            hit = hit.with_surface(face, barycentric).map_err(js_error)?;
        }
        self.dispatch_interaction(InteractionAction::SetHover(Some(hit)))
    }

    #[wasm_bindgen(js_name = clearInteractionHover)]
    pub fn clear_interaction_hover(&self) -> Result<u64, JsValue> {
        self.dispatch_interaction(InteractionAction::SetHover(None))
    }

    #[wasm_bindgen(js_name = pressInteractionPrimary)]
    pub fn press_interaction_primary(&self) -> Result<u64, JsValue> {
        self.dispatch_interaction(InteractionAction::PressPrimary)
    }

    #[wasm_bindgen(js_name = releaseInteractionPrimary)]
    pub fn release_interaction_primary(&self) -> Result<u64, JsValue> {
        self.dispatch_interaction(InteractionAction::ReleasePrimary)
    }

    #[wasm_bindgen(js_name = cancelInteractionPrimary)]
    pub fn cancel_interaction_primary(&self) -> Result<u64, JsValue> {
        self.dispatch_interaction(InteractionAction::CancelPrimary)
    }

    #[wasm_bindgen(js_name = interactionSnapshot)]
    pub fn interaction_snapshot(&self) -> Result<JsValue, JsValue> {
        interaction_to_js(
            self.store.frame_snapshot(),
            self.store.interaction_diagnostics_snapshot(),
        )
    }

    /// Record the synchronous stage disposition for one opt-in retained pick
    /// comparison. This is telemetry only and never changes selection.
    #[wasm_bindgen(js_name = recordBackendPickStage)]
    pub fn record_backend_pick_stage(
        &self,
        staged: bool,
        rejection: &str,
    ) -> Result<JsValue, JsValue> {
        let rejection = (!staged && !rejection.is_empty()).then(|| rejection.to_string());
        let snapshot = {
            let mut observer = self
                .backend_pick_evidence
                .try_borrow_mut()
                .map_err(|_| JsValue::from_str("backend pick diagnostics are already borrowed"))?;
            observer.record_stage(staged, rejection);
            observer.snapshot()
        };
        to_js(&snapshot)
    }

    /// Validate and record one asynchronous renderer comparison against the
    /// current interaction residency epoch. Stale evidence is retained only as
    /// a bounded diagnostic and cannot enter semantic interaction state.
    #[wasm_bindgen(js_name = recordBackendPickEvidence)]
    pub fn record_backend_pick_evidence(&self, report: JsValue) -> Result<JsValue, JsValue> {
        let report = serde_wasm_bindgen::from_value::<RenderPickEvidenceReport>(report)
            .map_err(|error| js_error(format!("backend pick evidence is invalid: {error}")))?;
        let snapshot = (|| -> Result<_, String> {
            let targets = self
                .interaction_targets
                .try_borrow()
                .map_err(|_| "interaction targets are being replaced".to_string())?;
            let mut observer = self
                .backend_pick_evidence
                .try_borrow_mut()
                .map_err(|_| "backend pick diagnostics are already borrowed".to_string())?;
            observer
                .record_report(&targets, report)
                .map_err(|error| error.to_string())?;
            Ok(observer.snapshot())
        })();
        let snapshot = match snapshot {
            Ok(snapshot) => snapshot,
            Err(error) => {
                if let Ok(mut observer) = self.backend_pick_evidence.try_borrow_mut() {
                    observer.record_error(format!("observe: {error}"));
                }
                return Err(js_error(error));
            }
        };
        to_js(&snapshot)
    }

    #[wasm_bindgen(js_name = recordBackendPickError)]
    pub fn record_backend_pick_error(
        &self,
        phase: &str,
        error: &str,
    ) -> Result<JsValue, JsValue> {
        let message = if phase.is_empty() {
            error.to_string()
        } else {
            format!("{phase}: {error}")
        };
        let snapshot = {
            let mut observer = self
                .backend_pick_evidence
                .try_borrow_mut()
                .map_err(|_| JsValue::from_str("backend pick diagnostics are already borrowed"))?;
            observer.record_error(message);
            observer.snapshot()
        };
        to_js(&snapshot)
    }

    #[wasm_bindgen(js_name = backendPickDiagnostics)]
    pub fn backend_pick_diagnostics(&self) -> Result<JsValue, JsValue> {
        let snapshot = self
            .backend_pick_evidence
            .try_borrow()
            .map_err(|_| JsValue::from_str("backend pick diagnostics are already borrowed"))?
            .snapshot();
        to_js(&snapshot)
    }

    /// Stage one renderer comparison using the interaction table's current
    /// epoch directly. The epoch never crosses the browser adapter.
    #[cfg(feature = "webgpu-backend")]
    #[wasm_bindgen(js_name = stageBackendPickEvidence)]
    pub fn stage_backend_pick_evidence(
        &self,
        mvp: &[f32],
        mv: &[f32],
        camera_pos: &[f32],
        x: i32,
        y: i32,
    ) -> Result<JsValue, JsValue> {
        let target_epoch = self
            .interaction_targets
            .try_borrow()
            .map_err(|_| JsValue::from_str("interaction targets are being replaced"))?
            .epoch();
        let receipt = crate::main_renderer::stage_backend_pick_evidence(
            mvp,
            mv,
            camera_pos,
            x,
            y,
            target_epoch,
        );
        if let Ok(mut observer) = self.backend_pick_evidence.try_borrow_mut() {
            observer.record_stage(receipt.staged(), receipt.error().map(str::to_string));
        }
        Ok(receipt.into_js())
    }

    /// Complete the one staged renderer comparison and record it against the
    /// current target table without serializing the report through JavaScript.
    #[cfg(feature = "webgpu-backend")]
    #[wasm_bindgen(js_name = readBackendPickEvidence)]
    pub async fn read_backend_pick_evidence(&self) -> Result<JsValue, JsValue> {
        let report = match crate::main_renderer::read_backend_pick_evidence().await {
            Ok(report) => report,
            Err(error) => {
                if let Ok(mut observer) = self.backend_pick_evidence.try_borrow_mut() {
                    observer.record_error(format!("readback: {error}"));
                }
                return Err(js_error(error));
            }
        };
        let targets = self
            .interaction_targets
            .try_borrow()
            .map_err(|_| JsValue::from_str("interaction targets are being replaced"))?;
        let snapshot = {
            let mut observer = self
                .backend_pick_evidence
                .try_borrow_mut()
                .map_err(|_| JsValue::from_str("backend pick diagnostics are already borrowed"))?;
            observer
                .record_report(&targets, report)
                .map_err(js_error)?;
            observer.snapshot()
        };
        to_js(&snapshot)
    }

    /// Resolve one current retained WebGPU pixel without executing the WebGL
    /// picker. Multiple browser calls may overlap: the Rust authority gate
    /// accepts only the newest request and only while its packed-node table
    /// epoch remains current.
    #[cfg(feature = "webgpu-backend")]
    #[wasm_bindgen(js_name = pickBackendSurface)]
    pub async fn pick_backend_surface(
        &self,
        x: i32,
        y: i32,
        retain_for_activation: bool,
    ) -> Result<JsValue, JsValue> {
        let pixel = [
            u32::try_from(x).map_err(|_| js_error("pick x is outside the viewport"))?,
            u32::try_from(y).map_err(|_| js_error("pick y is outside the viewport"))?,
        ];
        let target_epoch = self
            .interaction_targets
            .try_borrow()
            .map_err(|_| js_error("interaction targets are being replaced"))?
            .epoch();
        let mut activation = self
            .backend_pick_activation
            .try_borrow_mut()
            .map_err(|_| js_error("backend pick activation is already borrowed"))?;
        let request = self
            .backend_pick_authority
            .try_borrow_mut()
            .map_err(|_| js_error("backend pick authority is already borrowed"))?
            .begin(target_epoch)
            .map_err(js_error)?;
        activation.clear();
        drop(activation);
        let capture = match crate::main_renderer::stage_backend_pick_authority(
            pixel,
            target_epoch,
        ) {
            Ok(capture) => capture,
            Err(error) => {
                self.backend_pick_authority.borrow_mut().record_stage(
                    request,
                    Some(error.clone()),
                );
                return Err(js_error(error));
            }
        };
        self.backend_pick_authority
            .borrow_mut()
            .record_stage(request, None);
        let webgpu_frame_revision = capture.webgpu_frame_revision();
        let source_render_call = capture.source_render_call();
        let viewport = capture.viewport();
        let result = |disposition, activation_ready, hit, surface| ShadowBackendPickAuthorityResult {
            disposition: pick_authority_disposition_name(disposition),
            request_id: request.request_id.to_string(),
            target_epoch: request.target_epoch,
            activation_ready,
            webgpu_frame_revision: Some(webgpu_frame_revision.to_string()),
            source_render_call: Some(source_render_call.to_string()),
            viewport: Some(viewport),
            hit,
            surface,
        };
        let readback = match capture.read().await {
            Ok(readback) => readback,
            Err(error) => {
                let disposition = self
                    .backend_pick_authority
                    .borrow_mut()
                    .record_error(request, error.clone());
                if disposition == InteractionPickAuthorityDisposition::Current {
                    return Err(js_error(error));
                }
                return to_js(&result(disposition, false, None, None));
            }
        };
        let current_target_epoch = match self.interaction_targets.try_borrow() {
            Ok(targets) => targets.epoch(),
            Err(_) => {
                let error = "interaction targets are being replaced";
                self.backend_pick_authority
                    .borrow_mut()
                    .record_error(request, error);
                return Err(js_error(error));
            }
        };
        let disposition = self
            .backend_pick_authority
            .borrow_mut()
            .observe_readback(request, current_target_epoch);
        if disposition != InteractionPickAuthorityDisposition::Current {
            return to_js(&result(disposition, false, None, None));
        }

        let resolved = match readback.hit {
            None => Ok((None, None)),
            Some(raw) => (|| {
                let sample = InteractionTargetSample::new_for_epoch(
                    raw.target_epoch,
                    raw.packed_node,
                    raw.source_position.map(f64::from),
                    f64::from(raw.output_distance),
                )
                .map_err(|error| error.to_string())?
                .with_surface(raw.source_face, raw.source_barycentric.map(f64::from))
                .map_err(|error| error.to_string())?;
                let hit = self
                    .interaction_targets
                    .try_borrow()
                    .map_err(|_| "interaction targets are being replaced".to_string())?
                    .resolve(sample)
                    .map_err(|error| error.to_string())?;
                let surface = crate::main_renderer::resolve_backend_pick_surface(raw)?;
                Ok::<_, String>((
                    Some(hit),
                    Some(surface),
                ))
            })(),
        };
        let (hit, surface) = match resolved {
            Ok(resolved) => resolved,
            Err(error) => {
                self.backend_pick_authority
                    .borrow_mut()
                    .record_error(request, error.to_string());
                return Err(js_error(error));
            }
        };
        let disposition = self
            .backend_pick_authority
            .borrow_mut()
            .accept(request);
        let activation_ready = disposition == InteractionPickAuthorityDisposition::Current
            && retain_for_activation
            && hit.is_some();
        if activation_ready {
            self.backend_pick_activation
                .try_borrow_mut()
                .map_err(|_| js_error("backend pick activation is already borrowed"))?
                .publish(request, hit.expect("activation readiness requires a hit"))
                .map_err(js_error)?;
        }
        to_js(&result(
            disposition,
            activation_ready,
            hit.map(ShadowInteractionHit::from),
            surface,
        ))
    }

    /// Consume one accepted semantic hit as a single primary interaction and
    /// apply its routed focus/navigation transition at the current virtual
    /// application time. No renderer-local payload is accepted here.
    #[cfg(feature = "webgpu-backend")]
    #[wasm_bindgen(js_name = activateBackendPick)]
    pub fn activate_backend_pick(&self, request_id: &str) -> Result<JsValue, JsValue> {
        let request_id = request_id.parse::<u64>().map_err(|error| {
            js_error(format!("backend pick activation request ID is invalid: {error}"))
        })?;
        let target_epoch = self
            .interaction_targets
            .try_borrow()
            .map_err(|_| js_error("interaction targets are being replaced"))?
            .epoch();
        let hit = self
            .backend_pick_activation
            .try_borrow_mut()
            .map_err(|_| js_error("backend pick activation is already borrowed"))?
            .take(request_id, target_epoch)
            .map_err(js_error)?;
        self.activate_interaction_hit(hit)
    }

    /// Explicitly retire a platform-stale accepted pick, for example when the
    /// camera changes while its asynchronous pixel readback is pending.
    #[cfg(feature = "webgpu-backend")]
    #[wasm_bindgen(js_name = discardBackendPickActivation)]
    pub fn discard_backend_pick_activation(&self, request_id: &str) -> Result<bool, JsValue> {
        let request_id = request_id.parse::<u64>().map_err(|error| {
            js_error(format!("backend pick activation request ID is invalid: {error}"))
        })?;
        Ok(self
            .backend_pick_activation
            .try_borrow_mut()
            .map_err(|_| js_error("backend pick activation is already borrowed"))?
            .discard(request_id))
    }

    #[wasm_bindgen(js_name = backendPickAuthorityDiagnostics)]
    pub fn backend_pick_authority_diagnostics(&self) -> Result<JsValue, JsValue> {
        to_js(&self.backend_pick_authority.borrow().snapshot())
    }

    /// Apply one identity-checked AppStore focus/selection packet directly to
    /// the resident renderer. The packed node remains a backend-local handle;
    /// the application `(asset, entity)` pair must match the selected Rust state
    /// before it can be joined to that handle.
    ///
    /// Passing `selected_node = -1` and empty IDs applies a detached focus
    /// sphere and clears renderer selection. `false` means no renderer is
    /// resident; malformed or mismatched identity is an error and changes
    /// nothing.
    #[wasm_bindgen(js_name = applyFocusToRenderer)]
    pub fn apply_focus_to_renderer(
        &self,
        selected_node: i32,
        asset: &str,
        entity: &str,
    ) -> Result<bool, JsValue> {
        if selected_node < -1 {
            return Err(JsValue::from_str(
                "selected renderer node must be -1 or nonnegative",
            ));
        }
        let frame = self.store.frame_snapshot();
        match (selected_node, frame.selected_focus) {
            (-1, None) if asset.is_empty() && entity.is_empty() => {}
            (-1, _) => {
                return Err(JsValue::from_str(
                    "detached renderer focus requires empty IDs and no selected AppStore focus",
                ));
            }
            (_, Some(selected)) => {
                let expected = asset_entity_id(asset, entity)?;
                if selected.identity != expected {
                    return Err(JsValue::from_str(
                        "renderer focus identity does not match selected AppStore focus",
                    ));
                }
            }
            (_, None) => {
                return Err(JsValue::from_str(
                    "renderer node requires a selected AppStore focus",
                ));
            }
        }

        let center = frame.focus.sphere.center.map(|component| component as f32);
        let radius = frame.focus.sphere.radius as f32;
        if !center.iter().all(|component| component.is_finite())
            || !radius.is_finite()
            || radius <= 0.0
        {
            return Err(JsValue::from_str(
                "AppStore focus sphere is not representable by the f32 renderer",
            ));
        }
        Ok(crate::main_renderer::apply_focus_packet(
            [center[0], center[1], center[2], radius],
            frame.focus.focus_enabled,
            selected_node,
        ))
    }

    /// Advance the app-owned presentation clock by the same delta as the
    /// incumbent controller and return a compact pose/focus parity snapshot.
    #[wasm_bindgen(js_name = tickPresentation)]
    pub fn tick_presentation(&self, delta_seconds: f64) -> Result<JsValue, JsValue> {
        self.tick_navigation(delta_seconds)
    }

    /// Admit a validated presentation document without activating a cue or
    /// changing navigation state.
    #[wasm_bindgen(js_name = loadPresentation)]
    pub fn load_presentation(&self, json: &str) -> Result<JsValue, JsValue> {
        let presentation = Presentation::from_json(json).map_err(js_error)?;
        let commit = self
            .store
            .dispatch(AppEvent::PresentationLoaded(presentation))
            .map_err(js_error)?;
        commit_to_js(&commit)
    }

    /// Bind one authored presentation asset to the exact process-local
    /// primary scene currently resident in the renderer. This is platform
    /// resource evidence, never authored identity or durable HHHS state.
    #[wasm_bindgen(js_name = bindPresentationAnimationResidency)]
    pub fn bind_presentation_animation_residency(
        &self,
        presentation_asset_id: &str,
        scene_request_id: &str,
        resident_asset_id: &str,
    ) -> Result<JsValue, JsValue> {
        let commit = self
            .store
            .dispatch(AppEvent::PresentationAnimationResidencyChanged(Some(
                PresentationAnimationResidencyBinding {
                    presentation_asset_id: asset_id_from_str(presentation_asset_id)?,
                    scene_request_id: request_id_from_str(scene_request_id)?,
                    resident_asset_id: asset_id_from_str(resident_asset_id)?,
                },
            )))
            .map_err(js_error)?;
        commit_to_js(&commit)
    }

    /// Admit the renderer-local binding through the typed application port
    /// and return any clip selection/cancellation jobs chosen with it.
    #[wasm_bindgen(js_name = setPresentationAnimationResidency)]
    pub fn set_presentation_animation_residency(
        &self,
        presentation_asset_id: &str,
        scene_request_id: &str,
        resident_asset_id: &str,
    ) -> Result<JsValue, JsValue> {
        presentation_animation_residency_to_js(
            self.store
                .set_presentation_animation_residency(Some(PresentationAnimationResidencyBinding {
                    presentation_asset_id: asset_id_from_str(presentation_asset_id)?,
                    scene_request_id: request_id_from_str(scene_request_id)?,
                    resident_asset_id: asset_id_from_str(resident_asset_id)?,
                }))
                .map_err(js_error)?,
        )
    }

    /// Bind presentation animation to the AppStore-owned installed scene
    /// without exposing its process-local identity to the browser.
    #[wasm_bindgen(js_name = bindInstalledPresentationAnimationResidency)]
    pub fn bind_installed_presentation_animation_residency(
        &self,
        presentation_asset_id: &str,
    ) -> Result<JsValue, JsValue> {
        presentation_animation_residency_to_js(
            self.store
                .bind_presentation_animation_to_installed_scene(asset_id_from_str(
                    presentation_asset_id,
                )?)
                .map_err(js_error)?,
        )
    }

    #[wasm_bindgen(js_name = clearPresentationAnimationResidency)]
    pub fn clear_presentation_animation_residency(&self) -> Result<JsValue, JsValue> {
        let commit = self
            .store
            .dispatch(AppEvent::PresentationAnimationResidencyChanged(None))
            .map_err(js_error)?;
        commit_to_js(&commit)
    }

    #[wasm_bindgen(js_name = unsetPresentationAnimationResidency)]
    pub fn unset_presentation_animation_residency(&self) -> Result<JsValue, JsValue> {
        presentation_animation_residency_to_js(
            self.store
                .set_presentation_animation_residency(None)
                .map_err(js_error)?,
        )
    }

    /// Mirror low-rate cue intent. This shadow compares resolved desired state;
    /// the existing navigation controller remains frame/camera authority until
    /// a separate pose-parity gate is enabled.
    #[wasm_bindgen(js_name = present)]
    pub fn present(&self, sequence: u32, action: &str, cue_id: &str) -> Result<JsValue, JsValue> {
        let action = presentation_action_from_wire(action, cue_id)?;
        let commit = self
            .store
            .dispatch(AppEvent::Input(Timed {
                sequence: u64::from(sequence),
                at_seconds: self.store.frame_snapshot().elapsed_seconds,
                value: SemanticAction::Present(action),
            }))
            .map_err(js_error)?;
        commit_to_js(&commit)
    }

    /// Commit local presentation intent through AppStore's sequence authority.
    /// The explicitly sequenced `present` method remains available to replay
    /// and shadow adapters; ordinary Rust-authority browser input uses this
    /// boundary so JavaScript cannot race or reuse application input sequence.
    #[wasm_bindgen(js_name = dispatchPresentation)]
    pub fn dispatch_presentation(&self, action: &str, cue_id: &str) -> Result<JsValue, JsValue> {
        let (sequence, commit) = self
            .store
            .dispatch_semantic(SemanticAction::Present(presentation_action_from_wire(
                action, cue_id,
            )?))
            .map_err(js_error)?;
        to_js(&ShadowDirectSemanticReceipt {
            sequence: sequence.to_string(),
            commit: shadow_commit(&commit),
        })
    }

    /// Commit one cue action and expose its animation renderer jobs as typed
    /// fields. The generic dispatch method remains the replay/rollback seam.
    #[wasm_bindgen(js_name = requestPresentation)]
    pub fn request_presentation(&self, action: &str, cue_id: &str) -> Result<JsValue, JsValue> {
        let dispatch = self
            .store
            .dispatch_presentation(presentation_action_from_wire(action, cue_id)?)
            .map_err(js_error)?;
        to_js(&ShadowPresentationDispatch {
            sequence: dispatch.sequence.to_string(),
            commit: shadow_commit(&dispatch.commit),
            active: dispatch.active,
            selection: dispatch
                .selection
                .as_ref()
                .map(ShadowAnimationClipJobEffect::selection),
            cancellations: dispatch
                .cancellations
                .iter()
                .map(ShadowAnimationClipJobEffect::cancellation)
                .collect(),
        })
    }

    /// Commit explicit playback intent from a browser control or restored URL.
    #[wasm_bindgen(js_name = setAnimationPlaying)]
    pub fn set_animation_playing(
        &self,
        sequence: u32,
        playing: bool,
    ) -> Result<JsValue, JsValue> {
        self.dispatch_animation(sequence, AnimationAction::SetPlaying(playing))
    }

    /// Toggle playback atomically in the reducer so multiple input adapters do
    /// not race a browser-side read/modify/write sequence.
    #[wasm_bindgen(js_name = toggleAnimationPlaying)]
    pub fn toggle_animation_playing(&self, sequence: u32) -> Result<JsValue, JsValue> {
        self.dispatch_animation(sequence, AnimationAction::TogglePlaying)
    }

    /// Commit local playback intent through AppStore's sequence authority.
    /// Explicitly sequenced playback methods remain available for replay and
    /// shadow adapters, but ordinary browser controls do not allocate Rust
    /// application sequence numbers.
    #[wasm_bindgen(js_name = dispatchAnimationPlaying)]
    pub fn dispatch_animation_playing(&self, playing: bool) -> Result<JsValue, JsValue> {
        self.dispatch_animation_semantic(AnimationAction::SetPlaying(playing))
    }

    /// Atomically toggle playback through AppStore's sequence authority.
    #[wasm_bindgen(js_name = dispatchAnimationToggle)]
    pub fn dispatch_animation_toggle(&self) -> Result<JsValue, JsValue> {
        self.dispatch_animation_semantic(AnimationAction::TogglePlaying)
    }

    /// Seek the primary unwrapped scene clock without changing playback or
    /// speed. Clip wrapping remains a renderer concern.
    #[wasm_bindgen(js_name = seekAnimation)]
    pub fn seek_animation(&self, sequence: u32, time_seconds: f64) -> Result<JsValue, JsValue> {
        self.dispatch_animation(sequence, AnimationAction::Seek(time_seconds))
    }

    /// Seek local Rust-authority animation time with a store-allocated input
    /// sequence. The explicitly sequenced method remains the shadow oracle.
    #[wasm_bindgen(js_name = dispatchAnimationSeek)]
    pub fn dispatch_animation_seek(&self, time_seconds: f64) -> Result<JsValue, JsValue> {
        self.dispatch_animation_semantic(AnimationAction::Seek(time_seconds))
    }

    /// Change primary animation speed without changing time or playing state.
    #[wasm_bindgen(js_name = setAnimationSpeed)]
    pub fn set_animation_speed(&self, sequence: u32, speed: f64) -> Result<JsValue, JsValue> {
        self.dispatch_animation(sequence, AnimationAction::SetSpeed(speed))
    }

    /// Restore primary animation transport as one reducer action.
    #[wasm_bindgen(js_name = setAnimationClock)]
    pub fn set_animation_clock(
        &self,
        sequence: u32,
        playing: bool,
        time_seconds: f64,
        speed: f64,
    ) -> Result<JsValue, JsValue> {
        self.dispatch_animation(
            sequence,
            AnimationAction::SetClock(AnimationClock {
                playing,
                time_seconds,
                speed,
            }),
        )
    }

    /// Restore local Rust-authority animation transport atomically with a
    /// store-allocated input sequence.
    #[wasm_bindgen(js_name = dispatchAnimationClock)]
    pub fn dispatch_animation_clock(
        &self,
        playing: bool,
        time_seconds: f64,
        speed: f64,
    ) -> Result<JsValue, JsValue> {
        self.dispatch_animation_semantic(AnimationAction::SetClock(AnimationClock {
            playing,
            time_seconds,
            speed,
        }))
    }

    /// Request a renderer clip switch against the installed scene catalog.
    /// Rust allocates the job ID and emits the exact asynchronous effect; a
    /// browser adapter returns one of the completion methods below.
    #[wasm_bindgen(js_name = dispatchAnimationClip)]
    pub fn dispatch_animation_clip(&self, index: u32) -> Result<JsValue, JsValue> {
        let (sequence, commit) = self
            .store
            .dispatch_semantic(SemanticAction::Animate(AnimationAction::SelectClip(index)))
            .map_err(js_error)?;
        to_js(&ShadowDirectSemanticReceipt {
            sequence: sequence.to_string(),
            commit: shadow_commit(&commit),
        })
    }

    /// Request one renderer clip job through the typed application port.
    /// The generic dispatch method remains a compatibility/evidence seam;
    /// ordinary browser code must not rediscover selection and cancellation
    /// jobs by filtering its generic effect list.
    #[wasm_bindgen(js_name = requestAnimationClip)]
    pub fn request_animation_clip(&self, index: u32) -> Result<JsValue, JsValue> {
        let request = self.store.request_animation_clip(index).map_err(js_error)?;
        to_js(&ShadowAnimationClipRequest {
            sequence: request.sequence.to_string(),
            commit: shadow_commit(&request.commit),
            requested_index: request.requested_index,
            selection: request
                .selection
                .as_ref()
                .map(ShadowAnimationClipJobEffect::selection),
            cancellations: request
                .cancellations
                .iter()
                .map(ShadowAnimationClipJobEffect::cancellation)
                .collect(),
            state: request.state.into(),
            matches_request: request.matches_request,
        })
    }

    #[wasm_bindgen(js_name = completeAnimationClipSelected)]
    pub fn complete_animation_clip_selected(
        &self,
        job_id: &str,
        scene_request_id: &str,
        asset_id: &str,
        clip_index: u32,
    ) -> Result<JsValue, JsValue> {
        let dispatch = self.complete_animation_clip_selection_dispatch(
            job_id,
            scene_request_id,
            asset_id,
            clip_index,
            AnimationClipSelectionOutcome::Selected,
        )?;
        commit_to_js(&dispatch.commit)
    }

    #[wasm_bindgen(js_name = finishAnimationClipSelected)]
    pub fn finish_animation_clip_selected(
        &self,
        job_id: &str,
        scene_request_id: &str,
        asset_id: &str,
        clip_index: u32,
    ) -> Result<JsValue, JsValue> {
        animation_clip_completion_to_js(self.complete_animation_clip_selection_dispatch(
            job_id,
            scene_request_id,
            asset_id,
            clip_index,
            AnimationClipSelectionOutcome::Selected,
        )?)
    }

    #[wasm_bindgen(js_name = completeAnimationClipSelectionFailed)]
    #[allow(clippy::too_many_arguments)]
    pub fn complete_animation_clip_selection_failed(
        &self,
        job_id: &str,
        scene_request_id: &str,
        asset_id: &str,
        clip_index: u32,
        code: &str,
        message: &str,
        retryable: bool,
    ) -> Result<JsValue, JsValue> {
        let dispatch = self.complete_animation_clip_selection_dispatch(
            job_id,
            scene_request_id,
            asset_id,
            clip_index,
            AnimationClipSelectionOutcome::Failed {
                code: code.to_owned(),
                message: message.to_owned(),
                retryable,
            },
        )?;
        commit_to_js(&dispatch.commit)
    }

    #[wasm_bindgen(js_name = finishAnimationClipSelectionFailed)]
    #[allow(clippy::too_many_arguments)]
    pub fn finish_animation_clip_selection_failed(
        &self,
        job_id: &str,
        scene_request_id: &str,
        asset_id: &str,
        clip_index: u32,
        code: &str,
        message: &str,
        retryable: bool,
    ) -> Result<JsValue, JsValue> {
        animation_clip_completion_to_js(self.complete_animation_clip_selection_dispatch(
            job_id,
            scene_request_id,
            asset_id,
            clip_index,
            AnimationClipSelectionOutcome::Failed {
                code: code.to_owned(),
                message: message.to_owned(),
                retryable,
            },
        )?)
    }

    /// Replace the complete semantic render policy as one application event.
    /// Browser controls may shadow this boundary before consuming its value;
    /// backend resources and environment selection remain outside it.
    #[wasm_bindgen(js_name = setRenderSettings)]
    #[allow(clippy::too_many_arguments)]
    pub fn set_render_settings(
        &self,
        sequence: u32,
        style: &str,
        resolution_level: u8,
        density: f64,
        screen_attenuation: bool,
        min_pixels_per_subdivision: f64,
        atlas_exponent: u8,
        max_face_edge_ratio: u8,
        focus_enabled: bool,
        focus_mode: u8,
        blur_radius_pixels: u16,
        blur_strength: f64,
        focus_coordinate: f64,
        focus_bandwidth: f64,
        normalize_focus_range: bool,
        gaussian_passes: u8,
        kawase_passes: u8,
        kawase_offset: f64,
    ) -> Result<JsValue, JsValue> {
        let settings = ShadowRenderSettingsInput {
            style: style.to_owned(),
            resolution_level,
            density,
            screen_attenuation,
            min_pixels_per_subdivision,
            atlas_exponent,
            max_face_edge_ratio,
            focus_postprocess: ShadowFocusPostprocessInput {
                enabled: focus_enabled,
                mode: focus_mode,
                diagnostic_view: FocusDiagnosticView::Composite.wire_index(),
                blur_radius_pixels,
                blur_strength,
                focus_coordinate,
                bandwidth: focus_bandwidth,
                normalize_range: normalize_focus_range,
                gaussian_passes,
                kawase_passes,
                kawase_offset,
            },
        }
        .into_settings()
        .map_err(js_error)?;
        let commit = self
            .store
            .dispatch(AppEvent::Input(Timed {
                sequence: u64::from(sequence),
                at_seconds: self.store.frame_snapshot().elapsed_seconds,
                value: SemanticAction::SetRenderSettings(settings),
            }))
            .map_err(js_error)?;
        to_js(&ShadowRenderSettingsReceipt {
            commit: shadow_commit(&commit),
            render: self.store.render_snapshot().into(),
        })
    }

    /// Idempotently reconcile one complete backend-neutral render packet.
    /// Rust owns decoding, equality, sequence allocation, and every Patch Lab
    /// effect; JavaScript forwards a value and executes returned effects.
    #[wasm_bindgen(js_name = synchronizeRenderSettings)]
    pub fn synchronize_render_settings(&self, input: JsValue) -> Result<JsValue, JsValue> {
        let input: ShadowRenderSettingsInput =
            serde_wasm_bindgen::from_value(input).map_err(js_error)?;
        let settings = input.into_settings().map_err(js_error)?;
        let synchronization = self
            .store
            .synchronize_render_settings(settings)
            .map_err(js_error)?;
        let matches_input = synchronization.snapshot.settings == settings;
        let patch_lab_effects = synchronization
            .commit
            .as_ref()
            .map(PatchLabEffects::from_commit)
            .unwrap_or_default();
        to_js(&ShadowRenderSettingsSynchronizationReceipt {
            disposition: match synchronization.disposition {
                RenderSettingsSynchronizationDisposition::Unchanged => "unchanged",
                RenderSettingsSynchronizationDisposition::Committed => "committed",
            },
            sequence: synchronization.sequence.map(|sequence| sequence.to_string()),
            commit: synchronization.commit.as_ref().map(shadow_commit),
            patch_lab_effects: shadow_patch_lab_effects(&patch_lab_effects),
            matches_input,
            render: synchronization.snapshot.into(),
        })
    }

    /// Replace the device-independent navigation preference packet. Raw HID
    /// mappings and browser focus policy deliberately do not cross this
    /// boundary; transition and walk semantics do.
    #[wasm_bindgen(js_name = setNavigationSettings)]
    #[allow(clippy::too_many_arguments)]
    pub fn set_navigation_settings(
        &self,
        sequence: u32,
        transition_seconds: f64,
        smoothing_seconds: f64,
        tangent_pull_fraction: f64,
        speed_octave_steps: f64,
        body_scale_octave_steps: f64,
        eye_height_octave_steps: f64,
    ) -> Result<JsValue, JsValue> {
        let current = self.store.navigation_settings_snapshot().settings;
        let settings = NavigationSettings {
            transition_seconds,
            surface_walk: SurfaceWalkControls {
                smoothing_seconds,
                tangent_pull_fraction,
                speed_octave_steps,
                body_scale_octave_steps,
                eye_height_octave_steps,
                ..current.surface_walk
            },
        };
        let commit = self
            .store
            .dispatch(AppEvent::Input(Timed {
                sequence: u64::from(sequence),
                at_seconds: self.store.frame_snapshot().elapsed_seconds,
                value: SemanticAction::SetNavigationSettings(settings),
            }))
            .map_err(js_error)?;
        to_js(&ShadowNavigationSettingsReceipt {
            commit: shadow_commit(&commit),
            navigation: self.store.navigation_settings_snapshot().into(),
        })
    }

    /// Idempotently reconcile the browser's complete navigation-settings
    /// projection. Rust owns validation, equality, local input allocation, and
    /// the revision fence; JavaScript only supplies platform signal values and
    /// mirrors the returned committed projection.
    #[wasm_bindgen(js_name = synchronizeNavigationSettings)]
    #[allow(clippy::too_many_arguments)]
    pub fn synchronize_navigation_settings(
        &self,
        transition_seconds: f64,
        smoothing_seconds: f64,
        tangent_pull_fraction: f64,
        speed_octave_steps: f64,
        body_scale_octave_steps: f64,
        eye_height_octave_steps: f64,
    ) -> Result<JsValue, JsValue> {
        let current = self.store.navigation_settings_snapshot().settings;
        let settings = NavigationSettings {
            transition_seconds,
            surface_walk: SurfaceWalkControls {
                smoothing_seconds,
                tangent_pull_fraction,
                speed_octave_steps,
                body_scale_octave_steps,
                eye_height_octave_steps,
                ..current.surface_walk
            },
        };
        let synchronization = self
            .store
            .synchronize_navigation_settings(settings)
            .map_err(js_error)?;
        let matches_input = synchronization.snapshot.settings == settings;
        to_js(&ShadowNavigationSettingsSynchronizationReceipt {
            disposition: match synchronization.disposition {
                NavigationSettingsSynchronizationDisposition::Unchanged => "unchanged",
                NavigationSettingsSynchronizationDisposition::Committed => "committed",
            },
            sequence: synchronization.sequence.map(|sequence| sequence.to_string()),
            commit: synchronization.commit.as_ref().map(shadow_commit),
            matches_input,
            navigation: synchronization.snapshot.into(),
        })
    }

    /// Replace the complete educational Patch Lab session through the same
    /// reducer/effect boundary used by native, replay, and future WebGPU
    /// adapters. Rust allocates job IDs and coalesces LOD work; JavaScript may
    /// execute the emitted effects but does not own job lifetime.
    #[wasm_bindgen(js_name = dispatchPatchLab)]
    pub fn dispatch_patch_lab(&self, input: JsValue) -> Result<JsValue, JsValue> {
        let input: ShadowPatchLabIntentInput =
            serde_wasm_bindgen::from_value(input).map_err(js_error)?;
        let intent = input.into_intent()?;
        let (sequence, commit) = self
            .store
            .dispatch_semantic(SemanticAction::SetPatchLab(intent))
            .map_err(js_error)?;
        to_js(&ShadowDirectSemanticReceipt {
            sequence: sequence.to_string(),
            commit: shadow_commit(&commit),
        })
    }

    /// Commit a Patch Lab edit through AppStore's sequence authority and
    /// return its backend-neutral renderer jobs as a typed list.
    #[wasm_bindgen(js_name = requestPatchLab)]
    pub fn request_patch_lab(&self, input: JsValue) -> Result<JsValue, JsValue> {
        let input: ShadowPatchLabIntentInput =
            serde_wasm_bindgen::from_value(input).map_err(js_error)?;
        let dispatch = self
            .store
            .set_patch_lab_session(input.into_intent()?)
            .map_err(js_error)?;
        patch_lab_session_to_js(dispatch)
    }

    /// Complete one Rust-issued Patch Lab geometry job. The decoded geometry
    /// remains adapter-owned until renderer installation; this completion
    /// records only the backend-neutral evidence needed to schedule LOD work.
    #[wasm_bindgen(js_name = completePatchLabGeometry)]
    pub fn complete_patch_lab_geometry(
        &self,
        job_id: &str,
        vertex_count: u32,
        face_count: u32,
    ) -> Result<JsValue, JsValue> {
        let dispatch = self.complete_patch_lab_dispatch(PatchLabCompletion::Geometry(
            PatchLabGeometryCompletion {
                job_id: parse_patch_lab_job_id(job_id)?,
                outcome: PatchLabGeometryOutcome::Built {
                    vertex_count,
                    face_count,
                },
            },
        ))?;
        commit_to_js(&dispatch.commit)
    }

    #[wasm_bindgen(js_name = finishPatchLabGeometry)]
    pub fn finish_patch_lab_geometry(
        &self,
        job_id: &str,
        vertex_count: u32,
        face_count: u32,
    ) -> Result<JsValue, JsValue> {
        patch_lab_completion_to_js(self.complete_patch_lab_dispatch(
            PatchLabCompletion::Geometry(PatchLabGeometryCompletion {
                job_id: parse_patch_lab_job_id(job_id)?,
                outcome: PatchLabGeometryOutcome::Built {
                    vertex_count,
                    face_count,
                },
            }),
        )?)
    }

    #[wasm_bindgen(js_name = failPatchLabGeometry)]
    pub fn fail_patch_lab_geometry(
        &self,
        job_id: &str,
        code: &str,
        message: &str,
        retryable: bool,
    ) -> Result<JsValue, JsValue> {
        let dispatch = self.complete_patch_lab_dispatch(PatchLabCompletion::Geometry(
            PatchLabGeometryCompletion {
                job_id: parse_patch_lab_job_id(job_id)?,
                outcome: PatchLabGeometryOutcome::Failed(PatchLabFailure {
                    code: code.to_owned(),
                    message: message.to_owned(),
                    retryable,
                }),
            },
        ))?;
        commit_to_js(&dispatch.commit)
    }

    #[wasm_bindgen(js_name = finishPatchLabGeometryFailed)]
    pub fn finish_patch_lab_geometry_failed(
        &self,
        job_id: &str,
        code: &str,
        message: &str,
        retryable: bool,
    ) -> Result<JsValue, JsValue> {
        patch_lab_completion_to_js(self.complete_patch_lab_dispatch(
            PatchLabCompletion::Geometry(PatchLabGeometryCompletion {
                job_id: parse_patch_lab_job_id(job_id)?,
                outcome: PatchLabGeometryOutcome::Failed(PatchLabFailure {
                    code: code.to_owned(),
                    message: message.to_owned(),
                    retryable,
                }),
            }),
        )?)
    }

    /// Complete one Rust-issued LOD evaluation with a compact semantic
    /// summary. Renderer buffers stay out of application state and can be
    /// installed independently by WebGL2 or WebGPU adapters.
    #[wasm_bindgen(js_name = completePatchLabLod)]
    pub fn complete_patch_lab_lod(
        &self,
        job_id: &str,
        geometry_job_id: &str,
        summary: JsValue,
    ) -> Result<JsValue, JsValue> {
        let summary: ShadowPatchLabLodSummaryInput =
            serde_wasm_bindgen::from_value(summary).map_err(js_error)?;
        let dispatch = self.complete_patch_lab_dispatch(PatchLabCompletion::Lod(
            PatchLabLodCompletion {
                job_id: parse_patch_lab_job_id(job_id)?,
                geometry_job_id: parse_patch_lab_job_id(geometry_job_id)?,
                outcome: PatchLabLodOutcome::Evaluated(summary.try_into_summary()?),
            },
        ))?;
        commit_to_js(&dispatch.commit)
    }

    #[wasm_bindgen(js_name = finishPatchLabLod)]
    pub fn finish_patch_lab_lod(
        &self,
        job_id: &str,
        geometry_job_id: &str,
        summary: JsValue,
    ) -> Result<JsValue, JsValue> {
        let summary: ShadowPatchLabLodSummaryInput =
            serde_wasm_bindgen::from_value(summary).map_err(js_error)?;
        patch_lab_completion_to_js(self.complete_patch_lab_dispatch(PatchLabCompletion::Lod(
            PatchLabLodCompletion {
                job_id: parse_patch_lab_job_id(job_id)?,
                geometry_job_id: parse_patch_lab_job_id(geometry_job_id)?,
                outcome: PatchLabLodOutcome::Evaluated(summary.try_into_summary()?),
            },
        ))?)
    }

    #[wasm_bindgen(js_name = failPatchLabLod)]
    pub fn fail_patch_lab_lod(
        &self,
        job_id: &str,
        geometry_job_id: &str,
        code: &str,
        message: &str,
        retryable: bool,
    ) -> Result<JsValue, JsValue> {
        let dispatch = self.complete_patch_lab_dispatch(PatchLabCompletion::Lod(
            PatchLabLodCompletion {
                job_id: parse_patch_lab_job_id(job_id)?,
                geometry_job_id: parse_patch_lab_job_id(geometry_job_id)?,
                outcome: PatchLabLodOutcome::Failed(PatchLabFailure {
                    code: code.to_owned(),
                    message: message.to_owned(),
                    retryable,
                }),
            },
        ))?;
        commit_to_js(&dispatch.commit)
    }

    #[wasm_bindgen(js_name = finishPatchLabLodFailed)]
    pub fn finish_patch_lab_lod_failed(
        &self,
        job_id: &str,
        geometry_job_id: &str,
        code: &str,
        message: &str,
        retryable: bool,
    ) -> Result<JsValue, JsValue> {
        patch_lab_completion_to_js(self.complete_patch_lab_dispatch(PatchLabCompletion::Lod(
            PatchLabLodCompletion {
                job_id: parse_patch_lab_job_id(job_id)?,
                geometry_job_id: parse_patch_lab_job_id(geometry_job_id)?,
                outcome: PatchLabLodOutcome::Failed(PatchLabFailure {
                    code: code.to_owned(),
                    message: message.to_owned(),
                    retryable,
                }),
            },
        ))?)
    }

    /// Read the committed Patch Lab projection without exposing reducer
    /// internals or renderer-owned buffers.
    #[wasm_bindgen(js_name = patchLabSnapshot)]
    pub fn patch_lab_snapshot(&self) -> Result<JsValue, JsValue> {
        to_js(&ShadowPatchLabReadModel::from(
            self.store.patch_lab_snapshot(),
        ))
    }

    /// Write `[playing, unwrapped_time_seconds, speed]` without allocating a
    /// per-frame JavaScript object. The browser can shadow or consume this
    /// packet after `advanceFrameQuiet`.
    #[wasm_bindgen(js_name = writeAnimationClock)]
    pub fn write_animation_clock(&self, output: &mut [f64]) -> Result<(), JsValue> {
        if output.len() != 3 {
            return Err(JsValue::from_str(
                "animation clock output must contain exactly 3 numbers",
            ));
        }
        let animation = self.store.frame_snapshot().animation;
        output.copy_from_slice(&[
            if animation.playing { 1.0 } else { 0.0 },
            animation.time_seconds,
            animation.speed,
        ]);
        Ok(())
    }

    /// Write `[playing, wrapped_clip_time, speed]` for one resident clip. Rust
    /// owns reverse/forward loop semantics; the browser supplies only the
    /// renderer-local clip range.
    #[wasm_bindgen(js_name = writeAnimationSample)]
    pub fn write_animation_sample(
        &self,
        time_min: f64,
        duration: f64,
        output: &mut [f64],
    ) -> Result<(), JsValue> {
        if output.len() != 3 {
            return Err(JsValue::from_str(
                "animation sample output must contain exactly 3 numbers",
            ));
        }
        let animation = self.store.frame_snapshot().animation;
        let time = animation
            .clip_time(time_min, duration)
            .ok_or_else(|| JsValue::from_str("animation clip range must be finite and positive"))?;
        output.copy_from_slice(&[
            if animation.playing { 1.0 } else { 0.0 },
            time,
            animation.speed,
        ]);
        Ok(())
    }

    /// Write `[playing, clip_index, wrapped_clip_time, speed]` from the
    /// renderer-installed catalog owned by AppStore. Unlike
    /// `writeAnimationSample`, this boundary accepts no browser-authored clip
    /// range and therefore cannot sample a stale or different clip silently.
    #[wasm_bindgen(js_name = writeInstalledAnimationSample)]
    pub fn write_installed_animation_sample(&self, output: &mut [f64]) -> Result<(), JsValue> {
        if output.len() != 4 {
            return Err(JsValue::from_str(
                "installed animation sample output must contain exactly 4 numbers",
            ));
        }
        let frame = self.store.frame_snapshot();
        let clip = frame.active_animation_clip.ok_or_else(|| {
            JsValue::from_str("no renderer-installed active animation clip is available")
        })?;
        output.copy_from_slice(&[
            if frame.animation.playing { 1.0 } else { 0.0 },
            f64::from(clip.clip_index),
            clip.sample_time_seconds,
            frame.animation.speed,
        ]);
        Ok(())
    }

    /// Return only the coherent low-rate state needed by animation adapters.
    /// High-rate sampling remains allocation-free through
    /// `writeInstalledAnimationSample`.
    #[wasm_bindgen(js_name = animationRuntimeState)]
    pub fn animation_runtime_state(&self) -> Result<JsValue, JsValue> {
        let state = self.store.animation_runtime_snapshot();
        to_js(&ShadowAnimationRuntimeState {
            revision: state.revision.to_string(),
            residency: state.residency.map(Into::into),
            clip_state: state.clip_state.into(),
            playing: state.clock.playing,
            time_seconds: state.clock.time_seconds,
            speed: state.clock.speed,
        })
    }

    /// Atomically admit one transport-neutral authored checkpoint. The
    /// revision travels as decimal text so JavaScript cannot truncate a u64;
    /// commands retain the canonical protocol JSON shape shared with Blender.
    /// Parsing or validation failure leaves the preceding AppStore revision
    /// and materialized authored scene unchanged.
    #[wasm_bindgen(js_name = applyAuthoredRevision)]
    pub fn apply_authored_revision(
        &self,
        projection_revision: &str,
        commands_json: &str,
    ) -> Result<JsValue, JsValue> {
        let projection_revision = projection_revision.parse::<u64>().map_err(|error| {
            js_error(format!("authored projection revision is invalid: {error}"))
        })?;
        let commands =
            serde_json::from_str::<Vec<AuthoredEnvelope>>(commands_json).map_err(|error| {
                js_error(format!("authored command batch is invalid JSON: {error}"))
            })?;
        let commit = self
            .store
            .dispatch(AppEvent::AuthoredRevision(AuthoredRevision {
                projection_revision,
                commands,
            }))
            .map_err(js_error)?;
        commit_to_js(&commit)
    }

    /// Admit one canonical local-peer frame through the Rust application
    /// boundary. This method selects no carrier and gives no authority to
    /// WebSocket arrival order: it is the direct-demo single-writer adapter.
    #[wasm_bindgen(js_name = receiveLocalPeerEnvelope)]
    pub fn receive_local_peer_envelope(
        &self,
        received_at_seconds: f64,
        frame_json: &str,
    ) -> Result<JsValue, JsValue> {
        let envelope = serde_json::from_str::<LocalPeerEnvelope>(frame_json)
            .map_err(|error| js_error(format!("local peer frame is invalid JSON: {error}")))?;
        let receipt = self
            .peer_ingress
            .try_borrow_mut()
            .map_err(|_| JsValue::from_str("local peer ingress is already active"))?
            .accept(&self.store, envelope, received_at_seconds)
            .map_err(js_error)?;
        peer_receipt_to_js(&receipt)
    }

    /// Mark one already-applied local authored envelope before a carrier sends
    /// it. A subsequent relay echo is consumed without a second reducer event.
    #[wasm_bindgen(js_name = recordLocalAuthoredEnvelope)]
    pub fn record_local_authored_envelope(&self, envelope_json: &str) -> Result<(), JsValue> {
        let envelope =
            serde_json::from_str::<AuthoredEnvelope>(envelope_json).map_err(|error| {
                js_error(format!("local authored envelope is invalid JSON: {error}"))
            })?;
        self.peer_ingress
            .try_borrow_mut()
            .map_err(|_| JsValue::from_str("local peer ingress is already active"))?
            .record_local_authored(&envelope)
            .map_err(js_error)
    }

    /// Validate one locally generated presence envelope and remember its
    /// delivery echo without publishing this process as its own remote peer.
    #[wasm_bindgen(js_name = recordLocalPresenceEnvelope)]
    pub fn record_local_presence_envelope(&self, envelope_json: &str) -> Result<(), JsValue> {
        let envelope = serde_json::from_str::<PresenceEnvelope>(envelope_json).map_err(|error| {
            js_error(format!("local presence envelope is invalid JSON: {error}"))
        })?;
        self.peer_ingress
            .try_borrow_mut()
            .map_err(|_| JsValue::from_str("local peer ingress is already active"))?
            .record_local_presence(&envelope)
            .map_err(js_error)
    }

    /// Sample ephemeral peers independently from the throttled UI read-model
    /// commit fence. Each peer carries its sender sequence and local expiry.
    #[wasm_bindgen(js_name = peerPresenceSnapshot)]
    pub fn peer_presence_snapshot(&self) -> Result<JsValue, JsValue> {
        let elapsed_seconds = self.store.frame_snapshot().elapsed_seconds;
        let peers = self
            .store
            .presence_snapshot()
            .into_iter()
            .map(|peer| ShadowPeerPresence {
                peer_id: peer.peer.to_string(),
                sequence: peer.sequence.to_string(),
                expires_at_seconds: peer.expires_at_seconds,
                presence: peer.presence,
            })
            .collect();
        to_js(&ShadowPeerPresenceSnapshot {
            elapsed_seconds,
            peers,
        })
    }

    /// Resolve advisory asset-scoped authoring claims from the currently live
    /// presence set. This is a read-only coordination projection: it neither
    /// grants authority nor mutates authored/HHHS state.
    #[wasm_bindgen(js_name = authoringLeaseSnapshot)]
    pub fn authoring_lease_snapshot(&self) -> Result<JsValue, JsValue> {
        let leases = self
            .store
            .authoring_lease_snapshot()
            .into_iter()
            .map(|lease| {
                let (status, holders) = match lease.status {
                    AuthoringLeaseStatus::Held(holder) => ("held", vec![holder]),
                    AuthoringLeaseStatus::Contended(holders) => ("contended", holders),
                };
                ShadowAuthoringLease {
                    asset_id: lease.target.asset.to_string(),
                    entity_id: lease.target.entity.to_string(),
                    status,
                    holders: holders
                        .into_iter()
                        .map(|holder| ShadowAuthoringLeaseHolder {
                            peer_id: holder.peer.to_string(),
                            lease_id: holder.lease_id.to_string(),
                        })
                        .collect(),
                }
            })
            .collect();
        to_js(&ShadowAuthoringLeaseSnapshot { leases })
    }

    /// Resolve low-rate ordinary scene transforms through the same authored
    /// materialization observed by the application. The JSON input contains
    /// layer/asset/node packing metadata only; output is backend-neutral and
    /// this method never mutates a renderer.
    #[wasm_bindgen(js_name = extractPackedScene)]
    pub fn extract_packed_scene(&self, instances_json: &str) -> Result<JsValue, JsValue> {
        let inputs = serde_json::from_str::<Vec<ShadowPackedAssetInput>>(instances_json)
            .map_err(|error| js_error(format!("packed scene input is invalid JSON: {error}")))?;
        let instances = inputs
            .into_iter()
            .map(|input| {
                Ok(PackedAssetInstance {
                    layer: parse_uuid(&input.layer, "packed layer ID")?,
                    asset: asset_id_from_str(&input.asset)?,
                    layer_transform: input.layer_transform,
                    nodes: input
                        .nodes
                        .into_iter()
                        .map(packed_node_source)
                        .collect::<Result<Vec<_>, JsValue>>()?,
                })
            })
            .collect::<Result<Vec<_>, JsValue>>()?;
        let summary = self.store.summary_snapshot();
        let authored = self.store.authored_scene_snapshot();
        let authored_transforms = authored
            .entities
            .into_iter()
            .map(|entity| (entity.entity, entity.transform))
            .collect::<BTreeMap<_, _>>();
        let extraction =
            extract_packed_scene(&instances, &authored_transforms).map_err(js_error)?;
        to_js(&ShadowPackedSceneExtraction {
            app_revision: summary.revision.to_string(),
            authored_projection_revision: authored
                .projection_revision
                .map(|revision| revision.to_string()),
            nodes: extraction
                .nodes
                .into_iter()
                .map(|node| ShadowPackedNodeTransform {
                    layer: node.layer.to_string(),
                    asset: node.asset.to_string(),
                    packed_node: node.packed_node,
                    source_node: node.source_node,
                    entity_id: node.identity.map(|identity| identity.entity.to_string()),
                    source: match node.source {
                        PackedNodeTransformSource::GltfWorld => "gltf_world",
                        PackedNodeTransformSource::AuthoredAbsolute => "authored_absolute",
                    },
                    matrix: node.matrix,
                })
                .collect(),
            unmatched_authored_entities: extraction
                .unmatched_authored_entities
                .into_iter()
                .map(|entity| entity.to_string())
                .collect(),
        })
    }

    /// Join renderer-resident handles to the application-owned active cue.
    /// Bindings carry no layer transform, visibility, or opacity; those values
    /// and authored Blender overrides are sampled atomically from AppStore.
    #[wasm_bindgen(js_name = extractActivePresentationScene)]
    pub fn extract_active_presentation_scene(
        &self,
        bindings_json: &str,
    ) -> Result<JsValue, JsValue> {
        let inputs =
            serde_json::from_str::<Vec<ShadowPresentationLayerBindingInput>>(bindings_json)
                .map_err(|error| {
                    js_error(format!(
                        "active presentation binding input is invalid JSON: {error}"
                    ))
                })?;
        let bindings = inputs
            .into_iter()
            .map(|input| {
                Ok(PackedPresentationLayerBinding {
                    layer: parse_uuid(&input.layer, "presentation layer ID")?,
                    asset: asset_id_from_str(&input.asset)?,
                    nodes: input
                        .nodes
                        .into_iter()
                        .map(packed_node_source)
                        .collect::<Result<Vec<_>, JsValue>>()?,
                })
            })
            .collect::<Result<Vec<_>, JsValue>>()?;
        let extraction = self
            .store
            .extract_active_presentation_scene(&bindings)
            .map_err(js_error)?;
        to_js(&ShadowActivePresentationSceneExtraction {
            app_revision: extraction.revision.to_string(),
            authored_projection_revision: extraction
                .authored_projection_revision
                .map(|revision| revision.to_string()),
            cue_id: extraction.cue_id.to_string(),
            scene_id: extraction.scene_id.to_string(),
            nodes: extraction
                .nodes
                .into_iter()
                .map(|node| ShadowPackedPresentationNode {
                    layer: node.layer.to_string(),
                    asset: node.asset.to_string(),
                    packed_node: node.packed_node,
                    source_node: node.source_node,
                    entity_id: node.identity.map(|identity| identity.entity.to_string()),
                    source: match node.source {
                        PackedNodeTransformSource::GltfWorld => "gltf_world",
                        PackedNodeTransformSource::AuthoredAbsolute => "authored_absolute",
                    },
                    matrix: node.matrix,
                    visible: node.visible,
                    opacity: node.opacity,
                })
                .collect(),
            unmatched_authored_entities: extraction
                .unmatched_authored_entities
                .into_iter()
                .map(|entity| entity.to_string())
                .collect(),
        })
    }

    /// A bounded, UI-shaped projection. The app store publishes runtime asset,
    /// authored asset/entity, and diagnostic vectors before its summary
    /// revision commit fence.
    pub fn snapshot(&self) -> Result<JsValue, JsValue> {
        let summary = self.store.summary_snapshot();
        let navigation_settings = self.store.navigation_settings_snapshot();
        let render = self.store.render_snapshot();
        let ready_primary_asset = self.store.primary_asset_snapshot().map(|asset| {
            ShadowReadyPrimaryAsset {
                request_id: asset.request_id.to_string(),
                asset_id: asset.descriptor.id.to_string(),
                uri: asset.descriptor.uri,
                media_type: asset.descriptor.media_type,
                byte_length: asset.byte_length,
                content_digest: asset.content_digest,
                metadata: asset.metadata,
            }
        });
        let installed_primary_scene = self
            .store
            .installed_primary_scene_snapshot()
            .map(ShadowInstalledPrimaryScene::from);
        let animation_clip_selection: ShadowAnimationClipSelection =
            self.store.animation_clip_selection_snapshot().into();
        let authored = self.store.authored_scene_snapshot();
        let assets = self
            .store
            .asset_snapshot()
            .into_iter()
            .map(ShadowAsset::from)
            .collect();
        let diagnostics = self
            .store
            .diagnostic_snapshot()
            .into_iter()
            .map(|diagnostic| ShadowDiagnostic {
                revision: diagnostic.revision.to_string(),
                code: diagnostic.code,
                message: diagnostic.message,
            })
            .collect();
        let presentation = self.store.presentation_snapshot().map(ShadowPresentation::from);
        to_js(&ShadowSnapshot {
            revision: summary.revision.to_string(),
            animation_playing: summary.animation_playing,
            animation_time_seconds: summary.animation_time_seconds,
            animation_speed: summary.animation_speed,
            navigation_settings: navigation_settings.into(),
            render_settings: render.into(),
            patch_lab: self.store.patch_lab_snapshot().into(),
            assets,
            loading_assets: summary.loading_assets,
            loading_primary_scene_asset: summary
                .loading_primary_scene_asset
                .map(|asset| asset.to_string()),
            loading_primary_scene_request: summary
                .loading_primary_scene_request
                .map(|request| request.to_string()),
            ready_primary_asset,
            installing_primary_scene_asset: summary
                .installing_primary_scene_asset
                .map(|asset| asset.to_string()),
            installing_primary_scene_request: summary
                .installing_primary_scene_request
                .map(|request| request.to_string()),
            installed_primary_scene,
            animation_clip_selection,
            authored_projection_revision: authored
                .projection_revision
                .map(|revision| revision.to_string()),
            authored_assets: authored
                .assets
                .into_iter()
                .map(|asset| ShadowAuthoredAsset {
                    id: asset.id.to_string(),
                    uri: asset.uri,
                    media_type: asset.media_type,
                    content_digest: asset.content_digest,
                })
                .collect(),
            authored_entities: authored
                .entities
                .into_iter()
                .map(|entity| ShadowAuthoredEntity {
                    entity_id: entity.entity.to_string(),
                    translation: entity.transform.translation,
                    rotation_wxyz: entity.transform.rotation_wxyz,
                    scale: entity.transform.scale,
                })
                .collect(),
            diagnostics,
            presentation,
        })
    }
}

impl Default for HyperscopeAppShadow {
    fn default() -> Self {
        Self::new()
    }
}

impl HyperscopeAppShadow {
    fn retain_quiet_frame_effects(&self, commit: AppCommit) -> Result<u32, JsValue> {
        let effect_count = u32::try_from(commit.effects.len())
            .map_err(|_| JsValue::from_str("frame emitted too many adapter effects"))?;
        if effect_count != 0 {
            self.pending_adapter_effects
                .borrow_mut()
                .extend(commit.effects);
        }
        Ok(effect_count)
    }

    fn step_camera_action(
        &self,
        action: NavigationAction,
        input_name: &str,
        output: &mut [f64],
    ) -> Result<(), JsValue> {
        const CAMERA_PACKET_LEN: usize = 17;
        if output.len() != CAMERA_PACKET_LEN {
            return Err(JsValue::from_str(&format!(
                "{input_name} camera output must contain exactly 17 numbers"
            )));
        }
        let diagnostic_count = self.store.navigation_diagnostic_count();
        self.dispatch_navigation(action)?;
        self.advance_frame_delta_quiet(0.0)?;
        if self.store.navigation_diagnostic_count() != diagnostic_count {
            return Err(JsValue::from_str(
                &self
                    .store
                    .last_navigation_diagnostic()
                    .unwrap_or_else(|| format!("{input_name} camera intent was rejected")),
            ));
        }

        let frame = self.store.frame_snapshot();
        let basis = frame.camera.basis();
        let mut packet = [0.0; CAMERA_PACKET_LEN];
        packet[0..3].copy_from_slice(&frame.camera.eye);
        packet[3..6].copy_from_slice(&basis.right);
        packet[6..9].copy_from_slice(&basis.up);
        packet[9..12].copy_from_slice(&basis.forward);
        packet[12] = frame.camera.control_distance;
        if let Some(target) = frame.camera.semantic_target {
            packet[13] = 1.0;
            packet[14..17].copy_from_slice(&target);
        }
        output.copy_from_slice(&packet);
        Ok(())
    }

    fn dispatch_animation(
        &self,
        sequence: u32,
        action: AnimationAction,
    ) -> Result<JsValue, JsValue> {
        let commit = self
            .store
            .dispatch(AppEvent::Input(Timed {
                sequence: u64::from(sequence),
                at_seconds: self.store.frame_snapshot().elapsed_seconds,
                value: SemanticAction::Animate(action),
            }))
            .map_err(js_error)?;
        let animation = self.store.frame_snapshot().animation;
        to_js(&ShadowAnimationPlaybackReceipt {
            commit: shadow_commit(&commit),
            playing: animation.playing,
            time_seconds: animation.time_seconds,
            speed: animation.speed,
        })
    }

    fn dispatch_animation_semantic(&self, action: AnimationAction) -> Result<JsValue, JsValue> {
        let (sequence, commit) = self
            .store
            .dispatch_semantic(SemanticAction::Animate(action))
            .map_err(js_error)?;
        let animation = self.store.frame_snapshot().animation;
        to_js(&ShadowDirectAnimationPlaybackReceipt {
            sequence: sequence.to_string(),
            commit: shadow_commit(&commit),
            playing: animation.playing,
            time_seconds: animation.time_seconds,
            speed: animation.speed,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn request_asset_scoped(
        &self,
        sequence: u32,
        at_seconds: f64,
        request_id: &str,
        asset_id: &str,
        uri: &str,
        media_type: &str,
        scope: AssetLoadScope,
    ) -> Result<JsValue, JsValue> {
        let request_id = request_id_from_str(request_id)?;
        let asset = AssetDescriptor {
            id: asset_id_from_str(asset_id)?,
            uri: uri.to_owned(),
            media_type: (!media_type.is_empty()).then(|| media_type.to_owned()),
            content_digest: None,
        };
        let commit = self
            .store
            .dispatch(AppEvent::Input(Timed {
                sequence: u64::from(sequence),
                at_seconds,
                value: SemanticAction::RequestAsset {
                    request_id,
                    asset,
                    scope,
                },
            }))
            .map_err(js_error)?;
        commit_to_js(&commit)
    }

    fn dispatch_navigation(&self, action: NavigationAction) -> Result<u64, JsValue> {
        let (sequence, _) = self.store.dispatch_navigation(action).map_err(js_error)?;
        Ok(sequence)
    }

    fn dispatch_interaction(&self, action: InteractionAction) -> Result<u64, JsValue> {
        let (sequence, _) = self
            .store
            .dispatch_semantic(SemanticAction::Interact(action))
            .map_err(js_error)?;
        Ok(sequence)
    }

    fn activate_interaction_hit(&self, hit: InteractionHit) -> Result<JsValue, JsValue> {
        self.dispatch_interaction(InteractionAction::ActivatePrimary(hit))?;
        self.advance_frame_delta_quiet(0.0)?;
        let frame = self.store.frame_snapshot();
        if frame.interaction.hovered != Some(hit)
            || frame.interaction.active.is_some()
            || frame.interaction.selected != Some(hit.identity)
        {
            return Err(js_error(
                "interaction activation did not commit the resolved semantic hit",
            ));
        }
        navigation_to_js(
            frame,
            self.store.navigation_diagnostics_snapshot(),
        )
    }

    fn surface_anchor_target(
        &self,
        eye: &[f64],
        forward: &[f64],
        up: &[f64],
        control_distance: f64,
        normal: &[f64],
    ) -> Result<SurfaceAnchorTarget, JsValue> {
        let camera = CameraRig::new(
            vector3(eye, "surface anchor eye")?,
            CameraBasis::from_forward_up(
                vector3(forward, "surface anchor forward")?,
                vector3(up, "surface anchor up")?,
            )
            .map_err(js_error)?,
            control_distance,
            None,
            self.store.frame_snapshot().camera.lens,
        )
        .map_err(js_error)?;
        SurfaceAnchorTarget::new(camera, vector3(normal, "surface anchor normal")?)
            .map_err(js_error)
    }

    fn complete_asset(
        &self,
        request_id: &str,
        asset_id: &str,
        outcome: AssetLoadOutcome,
    ) -> Result<AssetLoadCompletionDispatch, JsValue> {
        self.store
            .complete_asset_load(AssetLoadCompletion {
                request_id: request_id_from_str(request_id)?,
                asset_id: asset_id_from_str(asset_id)?,
                outcome,
            })
            .map_err(js_error)
    }

    fn complete_primary_scene_install_dispatch(
        &self,
        request_id: &str,
        asset_id: &str,
        outcome: PrimarySceneInstallOutcome,
    ) -> Result<PrimarySceneInstallCompletionDispatch, JsValue> {
        self.store
            .complete_primary_scene_install(PrimarySceneInstallCompletion {
                request_id: request_id_from_str(request_id)?,
                asset_id: asset_id_from_str(asset_id)?,
                outcome,
            })
            .map_err(js_error)
    }

    fn complete_animation_clip_selection_dispatch(
        &self,
        job_id: &str,
        scene_request_id: &str,
        asset_id: &str,
        clip_index: u32,
        outcome: AnimationClipSelectionOutcome,
    ) -> Result<AnimationClipCompletionDispatch, JsValue> {
        let job_id = job_id
            .parse::<u64>()
            .map_err(|error| js_error(format!("animation clip job ID is invalid: {error}")))?;
        self.store
            .complete_animation_clip_selection(AnimationClipSelectionCompletion {
                job_id,
                scene_request_id: request_id_from_str(scene_request_id)?,
                asset_id: asset_id_from_str(asset_id)?,
                clip_index,
                outcome,
            })
            .map_err(js_error)
    }

    fn complete_patch_lab_dispatch(
        &self,
        completion: PatchLabCompletion,
    ) -> Result<PatchLabCompletionDispatch, JsValue> {
        self.store.complete_patch_lab(completion).map_err(js_error)
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ShadowPatchLabIntentInput {
    active: bool,
    shape: String,
    field: String,
    manual_edge_exponents: [u8; 3],
    min_exponent: u8,
    max_exponent: u8,
    phase_radians: f64,
    bend_percent: u8,
    grid: u8,
    animate: bool,
}

impl ShadowPatchLabIntentInput {
    fn into_intent(self) -> Result<PatchLabSessionIntent, JsValue> {
        let shape = PatchLabShape::from_wire_name(&self.shape)
            .ok_or_else(|| JsValue::from_str("unknown Patch Lab shape"))?;
        let field = match self.field.as_str() {
            // The URL and incumbent browser label predate the core enum's
            // unambiguous wire name. Admit both at this platform boundary.
            "edges" => PatchLabField::ManualEdges,
            value => PatchLabField::from_wire_name(value)
                .ok_or_else(|| JsValue::from_str("unknown Patch Lab LOD field"))?,
        };
        if !self.phase_radians.is_finite() {
            return Err(JsValue::from_str("Patch Lab phase must be finite"));
        }
        let phase_microradians = ((self
            .phase_radians
            .rem_euclid(std::f64::consts::TAU)
            * 1_000_000.0)
            .round() as u32)
            % hyperscope_app::PATCH_LAB_PHASE_TURN_MICRORADIANS;
        Ok(PatchLabSessionIntent {
            active: self.active,
            controls: PatchLabControls {
                shape,
                field,
                manual_edge_exponents: self.manual_edge_exponents,
                min_exponent: self.min_exponent,
                max_exponent: self.max_exponent,
                phase_microradians,
                bend_percent: self.bend_percent,
                grid: self.grid,
                animate: self.animate,
            },
        })
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ShadowPatchLabLodSummaryInput {
    requested_first_face: Option<[u32; 3]>,
    resident_first_face: Option<[u32; 3]>,
    promoted_faces: u32,
    promoted_edges: u32,
    shared_edges: u32,
    shared_edge_mismatches: u32,
    max_face_edge_ratio: u32,
    /// Browser renderer accounting arrives as a JavaScript Number. Admission
    /// below requires an exact safe integer before it enters Rust's `u64`
    /// semantic model.
    rendered_triangles: f64,
    #[serde(default)]
    histogram: Vec<ShadowPatchLabHistogramBin>,
}

impl ShadowPatchLabLodSummaryInput {
    fn try_into_summary(self) -> Result<PatchLabLodSummary, JsValue> {
        const MAX_SAFE_INTEGER: f64 = 9_007_199_254_740_991.0;
        if !self.rendered_triangles.is_finite()
            || self.rendered_triangles < 0.0
            || self.rendered_triangles.fract() != 0.0
            || self.rendered_triangles > MAX_SAFE_INTEGER
        {
            return Err(JsValue::from_str(
                "Patch Lab rendered triangle count must be an exact nonnegative JavaScript integer",
            ));
        }
        Ok(PatchLabLodSummary {
            requested_first_face: self.requested_first_face,
            resident_first_face: self.resident_first_face,
            promoted_faces: self.promoted_faces,
            promoted_edges: self.promoted_edges,
            shared_edges: self.shared_edges,
            shared_edge_mismatches: self.shared_edge_mismatches,
            max_face_edge_ratio: self.max_face_edge_ratio,
            rendered_triangles: self.rendered_triangles as u64,
            histogram: self
                .histogram
                .into_iter()
                .map(PatchLabHistogramBin::from)
                .collect(),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ShadowPatchLabHistogramBin {
    edge_subdivisions: [u32; 3],
    face_count: u32,
}

impl From<ShadowPatchLabHistogramBin> for PatchLabHistogramBin {
    fn from(bin: ShadowPatchLabHistogramBin) -> Self {
        Self {
            edge_subdivisions: bin.edge_subdivisions,
            face_count: bin.face_count,
        }
    }
}

impl From<PatchLabHistogramBin> for ShadowPatchLabHistogramBin {
    fn from(bin: PatchLabHistogramBin) -> Self {
        Self {
            edge_subdivisions: bin.edge_subdivisions,
            face_count: bin.face_count,
        }
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ShadowAnimationClipInput {
    index: u32,
    #[serde(default)]
    name: String,
    time_min_seconds: f64,
    time_max_seconds: f64,
}

fn decode_animation_clips(value: &str) -> Result<Vec<AnimationClipDescriptor>, JsValue> {
    serde_json::from_str::<Vec<ShadowAnimationClipInput>>(value)
        .map_err(|error| {
            js_error(format!(
                "primary scene animation clips are invalid JSON: {error}"
            ))
        })
        .map(|clips| {
            clips
                .into_iter()
                .map(|clip| AnimationClipDescriptor {
                    index: clip.index,
                    name: clip.name,
                    time_min_seconds: clip.time_min_seconds,
                    time_max_seconds: clip.time_max_seconds,
                })
                .collect()
        })
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowCommit {
    revision: String,
    disposition: &'static str,
    published_ui: bool,
    effects: Vec<ShadowEffect>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowDirectSemanticReceipt {
    sequence: String,
    commit: ShadowCommit,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowAssetLoadRequest {
    sequence: String,
    commit: ShadowCommit,
    fetch: ShadowAssetFetchJob,
    load_cancellations: Vec<ShadowAssetJobIdentity>,
    install_cancellations: Vec<ShadowAssetJobIdentity>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowAssetLoadCompletion {
    commit: ShadowCommit,
    install: Option<ShadowAssetJobIdentity>,
    asset: Option<ShadowAsset>,
}

fn asset_load_request_to_js(request: AssetLoadRequest) -> Result<JsValue, JsValue> {
    to_js(&ShadowAssetLoadRequest {
        sequence: request.sequence.to_string(),
        commit: shadow_commit(&request.commit),
        fetch: ShadowAssetFetchJob::from(request.fetch),
        load_cancellations: request
            .load_cancellations
            .into_iter()
            .map(ShadowAssetJobIdentity::from)
            .collect(),
        install_cancellations: request
            .install_cancellations
            .into_iter()
            .map(ShadowAssetJobIdentity::from)
            .collect(),
    })
}

fn asset_load_completion_to_js(dispatch: AssetLoadCompletionDispatch) -> Result<JsValue, JsValue> {
    to_js(&ShadowAssetLoadCompletion {
        commit: shadow_commit(&dispatch.commit),
        install: dispatch.install.map(ShadowAssetJobIdentity::from),
        asset: dispatch.asset.map(ShadowAsset::from),
    })
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowAssetFetchJob {
    request_id: String,
    asset_id: String,
    uri: String,
    media_type: Option<String>,
    content_digest: Option<[u8; 32]>,
}

impl From<AssetFetchJob> for ShadowAssetFetchJob {
    fn from(job: AssetFetchJob) -> Self {
        Self {
            request_id: job.request_id.to_string(),
            asset_id: job.asset.id.to_string(),
            uri: job.asset.uri,
            media_type: job.asset.media_type,
            content_digest: job.asset.content_digest,
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowAssetJobIdentity {
    request_id: String,
    asset_id: String,
}

impl From<AssetJobIdentity> for ShadowAssetJobIdentity {
    fn from(job: AssetJobIdentity) -> Self {
        Self {
            request_id: job.request_id.to_string(),
            asset_id: job.asset_id.to_string(),
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowLocalPeerReceipt {
    lane: &'static str,
    disposition: &'static str,
    projection_revision: Option<String>,
    commit: Option<ShadowCommit>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowPeerPresenceSnapshot {
    elapsed_seconds: f64,
    peers: Vec<ShadowPeerPresence>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowAuthoringLeaseSnapshot {
    leases: Vec<ShadowAuthoringLease>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowAuthoringLease {
    asset_id: String,
    entity_id: String,
    status: &'static str,
    holders: Vec<ShadowAuthoringLeaseHolder>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowAuthoringLeaseHolder {
    peer_id: String,
    lease_id: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowLocalPresenceSample {
    eye: [f64; 3],
    forward: [f64; 3],
    up: [f64; 3],
    selection: Vec<String>,
    authoring_targets: Vec<ShadowPresenceAuthoringTarget>,
    include_focus: bool,
    focus_center: [f64; 3],
    focus_radius: f64,
    inversion_enabled: bool,
    active_cue: String,
    animation_seconds: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowPresenceAuthoringTarget {
    asset_id: String,
    entity_id: String,
}

impl From<LocalPresenceAuthoringReadModel> for ShadowLocalPresenceSample {
    fn from(sample: LocalPresenceAuthoringReadModel) -> Self {
        let presence = sample.presence;
        let camera = presence
            .camera
            .expect("local application presence always carries a camera");
        let focus = presence.focus;
        Self {
            eye: camera.eye,
            forward: camera.forward,
            up: camera.up,
            selection: presence
                .selection
                .into_iter()
                .map(|entity| entity.to_string())
                .collect(),
            authoring_targets: sample
                .authoring_targets
                .into_iter()
                .map(|target| ShadowPresenceAuthoringTarget {
                    asset_id: target.asset.to_string(),
                    entity_id: target.entity.to_string(),
                })
                .collect(),
            include_focus: focus.is_some(),
            focus_center: focus.map_or([0.0; 3], |focus| focus.center),
            focus_radius: focus.map_or(1.0, |focus| focus.radius),
            inversion_enabled: focus.is_some_and(|focus| focus.inversion_enabled),
            active_cue: presence
                .active_cue
                .map_or_else(String::new, |cue| cue.to_string()),
            animation_seconds: presence.animation_seconds.unwrap_or(-1.0),
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowPeerPresence {
    peer_id: String,
    sequence: String,
    expires_at_seconds: f64,
    presence: EphemeralPresence,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SessionNodeIdentity {
    asset_id: String,
    entity_id: String,
    source_node: u32,
    durable: bool,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ShadowEffect {
    FetchAsset {
        request_id: String,
        asset_id: String,
        uri: String,
    },
    CancelAssetLoad {
        request_id: String,
        asset_id: String,
    },
    InstallPrimaryScene {
        request_id: String,
        asset_id: String,
    },
    CancelPrimarySceneInstall {
        request_id: String,
        asset_id: String,
    },
    SelectAnimationClip {
        job_id: String,
        scene_request_id: String,
        asset_id: String,
        clip_index: u32,
    },
    CancelAnimationClipSelection {
        job_id: String,
        scene_request_id: String,
        asset_id: String,
        clip_index: u32,
    },
    PatchLab {
        effect: ShadowPatchLabEffect,
    },
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ShadowPatchLabEffect {
    BuildGeometry {
        job_id: String,
        shape: &'static str,
        grid: u8,
        bend_percent: u8,
    },
    CancelGeometry {
        job_id: String,
    },
    DiscardGeometry {
        geometry_job_id: String,
    },
    EvaluateLod {
        job_id: String,
        geometry_job_id: String,
        field: &'static str,
        phase_microradians: u32,
        min_exponent: u8,
        max_exponent: u8,
        manual_edge_exponents: [u8; 3],
        atlas_exponent: u8,
        max_face_edge_ratio: u8,
    },
    CancelLod {
        job_id: String,
        geometry_job_id: String,
    },
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowPatchLabSessionDispatch {
    sequence: String,
    commit: ShadowCommit,
    patch_lab: ShadowPatchLabReadModel,
    effects: Vec<ShadowPatchLabEffect>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowPatchLabCompletionDispatch {
    commit: ShadowCommit,
    patch_lab: ShadowPatchLabReadModel,
    effects: Vec<ShadowPatchLabEffect>,
}

fn shadow_patch_lab_effects(effects: &PatchLabEffects) -> Vec<ShadowPatchLabEffect> {
    effects
        .as_slice()
        .iter()
        .map(shadow_patch_lab_effect)
        .collect()
}

fn patch_lab_session_to_js(dispatch: PatchLabSessionDispatch) -> Result<JsValue, JsValue> {
    to_js(&ShadowPatchLabSessionDispatch {
        sequence: dispatch.sequence.to_string(),
        commit: shadow_commit(&dispatch.commit),
        patch_lab: dispatch.state.into(),
        effects: shadow_patch_lab_effects(&dispatch.effects),
    })
}

fn patch_lab_completion_to_js(dispatch: PatchLabCompletionDispatch) -> Result<JsValue, JsValue> {
    to_js(&ShadowPatchLabCompletionDispatch {
        commit: shadow_commit(&dispatch.commit),
        patch_lab: dispatch.state.into(),
        effects: shadow_patch_lab_effects(&dispatch.effects),
    })
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowSnapshot {
    revision: String,
    animation_playing: bool,
    animation_time_seconds: f64,
    animation_speed: f64,
    navigation_settings: ShadowNavigationSettings,
    render_settings: ShadowRenderSettings,
    patch_lab: ShadowPatchLabReadModel,
    assets: Vec<ShadowAsset>,
    loading_assets: usize,
    loading_primary_scene_asset: Option<String>,
    loading_primary_scene_request: Option<String>,
    ready_primary_asset: Option<ShadowReadyPrimaryAsset>,
    installing_primary_scene_asset: Option<String>,
    installing_primary_scene_request: Option<String>,
    installed_primary_scene: Option<ShadowInstalledPrimaryScene>,
    animation_clip_selection: ShadowAnimationClipSelection,
    authored_projection_revision: Option<String>,
    authored_assets: Vec<ShadowAuthoredAsset>,
    authored_entities: Vec<ShadowAuthoredEntity>,
    diagnostics: Vec<ShadowDiagnostic>,
    presentation: Option<ShadowPresentation>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowPatchLabReadModel {
    active: bool,
    controls: ShadowPatchLabControls,
    pending_geometry_job: Option<String>,
    installed_geometry: Option<ShadowPatchLabGeometryReadModel>,
    pending_lod_job: Option<String>,
    lod_dirty: bool,
    latest_lod: Option<ShadowPatchLabLodSummary>,
    last_error: Option<ShadowPatchLabFailure>,
}

impl From<PatchLabReadModel> for ShadowPatchLabReadModel {
    fn from(model: PatchLabReadModel) -> Self {
        Self {
            active: model.active,
            controls: model.controls.into(),
            pending_geometry_job: model.pending_geometry_job.map(|job| job.to_string()),
            installed_geometry: model.installed_geometry.map(Into::into),
            pending_lod_job: model.pending_lod_job.map(|job| job.to_string()),
            lod_dirty: model.lod_dirty,
            latest_lod: model.latest_lod.map(Into::into),
            last_error: model.last_error.map(Into::into),
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowPatchLabControls {
    shape: &'static str,
    field: &'static str,
    manual_edge_exponents: [u8; 3],
    min_exponent: u8,
    max_exponent: u8,
    phase_microradians: u32,
    phase_radians: f64,
    bend_percent: u8,
    grid: u8,
    animate: bool,
}

impl From<PatchLabControls> for ShadowPatchLabControls {
    fn from(controls: PatchLabControls) -> Self {
        Self {
            shape: controls.shape.wire_name(),
            field: controls.field.wire_name(),
            manual_edge_exponents: controls.manual_edge_exponents,
            min_exponent: controls.min_exponent,
            max_exponent: controls.max_exponent,
            phase_microradians: controls.phase_microradians,
            phase_radians: controls.phase_radians(),
            bend_percent: controls.bend_percent,
            grid: controls.grid,
            animate: controls.animate,
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowPatchLabGeometryReadModel {
    job_id: String,
    shape: &'static str,
    grid: u8,
    bend_percent: u8,
    vertex_count: u32,
    face_count: u32,
}

impl From<hyperscope_app::PatchLabGeometryReadModel> for ShadowPatchLabGeometryReadModel {
    fn from(model: hyperscope_app::PatchLabGeometryReadModel) -> Self {
        Self {
            job_id: model.job_id.to_string(),
            shape: model.geometry.shape.wire_name(),
            grid: model.geometry.grid,
            bend_percent: model.geometry.bend_percent,
            vertex_count: model.vertex_count,
            face_count: model.face_count,
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowPatchLabLodSummary {
    requested_first_face: Option<[u32; 3]>,
    resident_first_face: Option<[u32; 3]>,
    promoted_faces: u32,
    promoted_edges: u32,
    shared_edges: u32,
    shared_edge_mismatches: u32,
    max_face_edge_ratio: u32,
    rendered_triangles: u64,
    histogram: Vec<ShadowPatchLabHistogramBin>,
}

impl From<PatchLabLodSummary> for ShadowPatchLabLodSummary {
    fn from(summary: PatchLabLodSummary) -> Self {
        Self {
            requested_first_face: summary.requested_first_face,
            resident_first_face: summary.resident_first_face,
            promoted_faces: summary.promoted_faces,
            promoted_edges: summary.promoted_edges,
            shared_edges: summary.shared_edges,
            shared_edge_mismatches: summary.shared_edge_mismatches,
            max_face_edge_ratio: summary.max_face_edge_ratio,
            rendered_triangles: summary.rendered_triangles,
            histogram: summary.histogram.into_iter().map(Into::into).collect(),
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowPatchLabFailure {
    code: String,
    message: String,
    retryable: bool,
}

impl From<PatchLabFailure> for ShadowPatchLabFailure {
    fn from(failure: PatchLabFailure) -> Self {
        Self {
            code: failure.code,
            message: failure.message,
            retryable: failure.retryable,
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowReadyPrimaryAsset {
    request_id: String,
    asset_id: String,
    uri: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    media_type: Option<String>,
    byte_length: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    content_digest: Option<[u8; 32]>,
    #[serde(skip_serializing_if = "AssetMetadata::is_empty")]
    metadata: AssetMetadata,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowInstalledPrimaryScene {
    asset: ShadowReadyPrimaryAsset,
    num_vertices: u32,
    num_faces: u32,
    animation_clips: Vec<ShadowAnimationClip>,
}

impl From<InstalledPrimarySceneReadModel> for ShadowInstalledPrimaryScene {
    fn from(scene: InstalledPrimarySceneReadModel) -> Self {
        Self {
            asset: ShadowReadyPrimaryAsset {
                request_id: scene.asset.request_id.to_string(),
                asset_id: scene.asset.descriptor.id.to_string(),
                uri: scene.asset.descriptor.uri,
                media_type: scene.asset.descriptor.media_type,
                byte_length: scene.asset.byte_length,
                content_digest: scene.asset.content_digest,
                metadata: scene.asset.metadata,
            },
            num_vertices: scene.install.num_vertices,
            num_faces: scene.install.num_faces,
            animation_clips: scene
                .install
                .animation_clips
                .into_iter()
                .map(ShadowAnimationClip::from)
                .collect(),
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowAnimationClip {
    index: u32,
    name: String,
    time_min_seconds: f64,
    time_max_seconds: f64,
}

impl From<AnimationClipDescriptor> for ShadowAnimationClip {
    fn from(clip: AnimationClipDescriptor) -> Self {
        Self {
            index: clip.index,
            name: clip.name,
            time_min_seconds: clip.time_min_seconds,
            time_max_seconds: clip.time_max_seconds,
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowAnimationClipSelection {
    active: Option<ShadowActiveAnimationClip>,
    pending: Option<ShadowPendingAnimationClip>,
}

impl From<AnimationClipSelectionReadModel> for ShadowAnimationClipSelection {
    fn from(selection: AnimationClipSelectionReadModel) -> Self {
        Self {
            active: selection.active.map(|active| ShadowActiveAnimationClip {
                scene_request_id: active.scene_request_id.to_string(),
                asset_id: active.asset_id.to_string(),
                clip: active.clip.into(),
            }),
            pending: selection.pending.map(|pending| ShadowPendingAnimationClip {
                job_id: pending.job_id.to_string(),
                scene_request_id: pending.scene_request_id.to_string(),
                asset_id: pending.asset_id.to_string(),
                clip: pending.clip.into(),
            }),
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowAnimationClipRequest {
    sequence: String,
    commit: ShadowCommit,
    requested_index: u32,
    selection: Option<ShadowAnimationClipJobEffect>,
    cancellations: Vec<ShadowAnimationClipJobEffect>,
    state: ShadowAnimationClipSelection,
    matches_request: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowAnimationClipCompletionDispatch {
    commit: ShadowCommit,
    selection: ShadowAnimationClipSelection,
}

fn animation_clip_completion_to_js(
    dispatch: AnimationClipCompletionDispatch,
) -> Result<JsValue, JsValue> {
    to_js(&ShadowAnimationClipCompletionDispatch {
        commit: shadow_commit(&dispatch.commit),
        selection: dispatch.state.into(),
    })
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowPrimarySceneInstallCompletionDispatch {
    commit: ShadowCommit,
    clip_cancellations: Vec<ShadowAnimationClipJobEffect>,
    installed_scene: Option<ShadowInstalledPrimaryScene>,
    clip_state: ShadowAnimationClipSelection,
}

fn primary_scene_install_completion_to_js(
    dispatch: PrimarySceneInstallCompletionDispatch,
) -> Result<JsValue, JsValue> {
    to_js(&ShadowPrimarySceneInstallCompletionDispatch {
        commit: shadow_commit(&dispatch.commit),
        clip_cancellations: dispatch
            .clip_cancellations
            .iter()
            .map(ShadowAnimationClipJobEffect::cancellation)
            .collect(),
        installed_scene: dispatch.installed_scene.map(Into::into),
        clip_state: dispatch.clip_state.into(),
    })
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowPresentationDispatch {
    sequence: String,
    commit: ShadowCommit,
    active: Option<PresentationSnapshot>,
    selection: Option<ShadowAnimationClipJobEffect>,
    cancellations: Vec<ShadowAnimationClipJobEffect>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowPresentationAnimationResidencyDispatch {
    commit: ShadowCommit,
    residency: Option<ShadowPresentationAnimationResidency>,
    active: Option<PresentationSnapshot>,
    selection: Option<ShadowAnimationClipJobEffect>,
    cancellations: Vec<ShadowAnimationClipJobEffect>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowAnimationRuntimeState {
    revision: String,
    residency: Option<ShadowPresentationAnimationResidency>,
    clip_state: ShadowAnimationClipSelection,
    playing: bool,
    time_seconds: f64,
    speed: f64,
}

fn presentation_animation_residency_to_js(
    dispatch: PresentationAnimationResidencyDispatch,
) -> Result<JsValue, JsValue> {
    to_js(&ShadowPresentationAnimationResidencyDispatch {
        commit: shadow_commit(&dispatch.commit),
        residency: dispatch.residency.map(Into::into),
        active: dispatch.active,
        selection: dispatch
            .selection
            .as_ref()
            .map(ShadowAnimationClipJobEffect::selection),
        cancellations: dispatch
            .cancellations
            .iter()
            .map(ShadowAnimationClipJobEffect::cancellation)
            .collect(),
    })
}

#[derive(Serialize)]
struct ShadowAnimationClipJobEffect {
    #[serde(rename = "type")]
    effect_type: &'static str,
    job_id: String,
    scene_request_id: String,
    asset_id: String,
    clip_index: u32,
}

impl ShadowAnimationClipJobEffect {
    fn selection(effect: &AnimationClipJobEffect) -> Self {
        Self::new("select_animation_clip", effect)
    }

    fn cancellation(effect: &AnimationClipJobEffect) -> Self {
        Self::new("cancel_animation_clip_selection", effect)
    }

    fn new(effect_type: &'static str, effect: &AnimationClipJobEffect) -> Self {
        Self {
            effect_type,
            job_id: effect.job_id.to_string(),
            scene_request_id: effect.scene_request_id.to_string(),
            asset_id: effect.asset_id.to_string(),
            clip_index: effect.clip_index,
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowActiveAnimationClip {
    scene_request_id: String,
    asset_id: String,
    clip: ShadowAnimationClip,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowPendingAnimationClip {
    job_id: String,
    scene_request_id: String,
    asset_id: String,
    clip: ShadowAnimationClip,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowRenderSettingsReceipt {
    commit: ShadowCommit,
    render: ShadowRenderSettings,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowRenderSettingsSynchronizationReceipt {
    disposition: &'static str,
    sequence: Option<String>,
    commit: Option<ShadowCommit>,
    patch_lab_effects: Vec<ShadowPatchLabEffect>,
    matches_input: bool,
    render: ShadowRenderSettings,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ShadowRenderSettingsInput {
    style: String,
    resolution_level: u8,
    density: f64,
    screen_attenuation: bool,
    min_pixels_per_subdivision: f64,
    atlas_exponent: u8,
    max_face_edge_ratio: u8,
    focus_postprocess: ShadowFocusPostprocessInput,
}

impl ShadowRenderSettingsInput {
    fn into_settings(self) -> Result<RenderSettings, &'static str> {
        RenderSettings::from_wire_values(
            &self.style,
            self.resolution_level,
            self.density,
            self.screen_attenuation,
            self.min_pixels_per_subdivision,
            self.atlas_exponent,
            self.max_face_edge_ratio,
        )
        .and_then(|settings| {
            settings.with_focus_postprocess(FocusPostprocessSettings {
                enabled: self.focus_postprocess.enabled,
                mode: FocusPostprocessMode::from_wire_index(self.focus_postprocess.mode)
                    .ok_or("unknown focus postprocess mode")?,
                diagnostic_view: FocusDiagnosticView::from_wire_index(
                    self.focus_postprocess.diagnostic_view,
                )
                .ok_or("unknown focus diagnostic view")?,
                blur_radius_pixels: self.focus_postprocess.blur_radius_pixels,
                blur_strength: self.focus_postprocess.blur_strength,
                focus_coordinate: self.focus_postprocess.focus_coordinate,
                bandwidth: self.focus_postprocess.bandwidth,
                normalize_range: self.focus_postprocess.normalize_range,
                gaussian_passes: self.focus_postprocess.gaussian_passes,
                kawase_passes: self.focus_postprocess.kawase_passes,
                kawase_offset: self.focus_postprocess.kawase_offset,
            })
        })
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ShadowFocusPostprocessInput {
    enabled: bool,
    mode: u8,
    diagnostic_view: u8,
    blur_radius_pixels: u16,
    blur_strength: f64,
    focus_coordinate: f64,
    bandwidth: f64,
    normalize_range: bool,
    gaussian_passes: u8,
    kawase_passes: u8,
    kawase_offset: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowNavigationSettingsReceipt {
    commit: ShadowCommit,
    navigation: ShadowNavigationSettings,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowNavigationSettingsSynchronizationReceipt {
    disposition: &'static str,
    sequence: Option<String>,
    commit: Option<ShadowCommit>,
    matches_input: bool,
    navigation: ShadowNavigationSettings,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowNavigationSettings {
    revision: String,
    transition_seconds: f64,
    smoothing_seconds: f64,
    tangent_pull_fraction: f64,
    speed_octave_steps: f64,
    body_scale_octave_steps: f64,
    eye_height_octave_steps: f64,
}

impl From<hyperscope_app::AppNavigationSettingsSnapshot> for ShadowNavigationSettings {
    fn from(snapshot: hyperscope_app::AppNavigationSettingsSnapshot) -> Self {
        let walk = snapshot.settings.surface_walk;
        Self {
            revision: snapshot.revision.to_string(),
            transition_seconds: snapshot.settings.transition_seconds,
            smoothing_seconds: walk.smoothing_seconds,
            tangent_pull_fraction: walk.tangent_pull_fraction,
            speed_octave_steps: walk.speed_octave_steps,
            body_scale_octave_steps: walk.body_scale_octave_steps,
            eye_height_octave_steps: walk.eye_height_octave_steps,
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowRenderSettings {
    revision: String,
    style: &'static str,
    resolution_level: u8,
    density: f64,
    screen_attenuation: bool,
    min_pixels_per_subdivision: f64,
    atlas_exponent: u8,
    max_face_edge_ratio: u8,
    focus_postprocess: FocusPostprocessShadow,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct FocusPostprocessShadow {
    enabled: bool,
    mode: u8,
    diagnostic_view: u8,
    blur_radius_pixels: u16,
    blur_strength: f64,
    focus_coordinate: f64,
    bandwidth: f64,
    normalize_range: bool,
    gaussian_passes: u8,
    kawase_passes: u8,
    kawase_offset: f64,
}

impl From<FocusPostprocessSettings> for FocusPostprocessShadow {
    fn from(settings: FocusPostprocessSettings) -> Self {
        Self {
            enabled: settings.enabled,
            mode: settings.mode.wire_index(),
            diagnostic_view: settings.diagnostic_view.wire_index(),
            blur_radius_pixels: settings.blur_radius_pixels,
            blur_strength: settings.blur_strength,
            focus_coordinate: settings.focus_coordinate,
            bandwidth: settings.bandwidth,
            normalize_range: settings.normalize_range,
            gaussian_passes: settings.gaussian_passes,
            kawase_passes: settings.kawase_passes,
            kawase_offset: settings.kawase_offset,
        }
    }
}

impl From<hyperscope_app::AppRenderSnapshot> for ShadowRenderSettings {
    fn from(snapshot: hyperscope_app::AppRenderSnapshot) -> Self {
        Self {
            revision: snapshot.revision.to_string(),
            style: snapshot.settings.style.wire_name(),
            resolution_level: snapshot.settings.resolution_level,
            density: snapshot.settings.tessellation.density,
            screen_attenuation: snapshot.settings.tessellation.screen_attenuation,
            min_pixels_per_subdivision: snapshot.settings.tessellation.min_pixels_per_subdivision,
            atlas_exponent: snapshot.settings.atlas_exponent,
            max_face_edge_ratio: snapshot.settings.max_face_edge_ratio,
            focus_postprocess: snapshot.settings.focus_postprocess.into(),
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowAnimationPlaybackReceipt {
    commit: ShadowCommit,
    playing: bool,
    time_seconds: f64,
    speed: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowDirectAnimationPlaybackReceipt {
    sequence: String,
    commit: ShadowCommit,
    playing: bool,
    time_seconds: f64,
    speed: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowAuthoredAsset {
    id: String,
    uri: String,
    media_type: Option<String>,
    content_digest: Option<[u8; 32]>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowAuthoredEntity {
    entity_id: String,
    translation: [f64; 3],
    rotation_wxyz: [f64; 4],
    scale: [f64; 3],
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ShadowPackedAssetInput {
    layer: String,
    asset: String,
    layer_transform: LayerTransform,
    nodes: Vec<ShadowPackedNodeInput>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ShadowPresentationLayerBindingInput {
    layer: String,
    asset: String,
    nodes: Vec<ShadowPackedNodeInput>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ShadowPackedNodeInput {
    packed_node: u32,
    source_node: u32,
    entity_id: Option<String>,
    source_world: [f32; 16],
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ShadowInteractionTargetInput {
    packed_node: u32,
    identity: Option<ShadowInteractionTargetIdentityInput>,
    source_bound: ShadowInteractionSourceBoundInput,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ShadowInteractionTargetIdentityInput {
    asset_id: String,
    entity_id: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ShadowInteractionSourceBoundInput {
    center: [f64; 3],
    radius: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowPackedSceneExtraction {
    app_revision: String,
    authored_projection_revision: Option<String>,
    nodes: Vec<ShadowPackedNodeTransform>,
    unmatched_authored_entities: Vec<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowActivePresentationSceneExtraction {
    app_revision: String,
    authored_projection_revision: Option<String>,
    cue_id: String,
    scene_id: String,
    nodes: Vec<ShadowPackedPresentationNode>,
    unmatched_authored_entities: Vec<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowPackedNodeTransform {
    layer: String,
    asset: String,
    packed_node: u32,
    source_node: u32,
    entity_id: Option<String>,
    source: &'static str,
    matrix: [f32; 16],
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowPackedPresentationNode {
    layer: String,
    asset: String,
    packed_node: u32,
    source_node: u32,
    entity_id: Option<String>,
    source: &'static str,
    matrix: [f32; 16],
    visible: bool,
    opacity: f32,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowPresentation {
    id: String,
    title: String,
    cue_count: usize,
    assets: Vec<PresentationAsset>,
    active: Option<PresentationSnapshot>,
    animation_residency: Option<ShadowPresentationAnimationResidency>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowPresentationCompositionPlan {
    revision: String,
    primary: PresentationAsset,
    secondary: Vec<PresentationAsset>,
}

impl From<hyperscope_app::PresentationReadModel> for ShadowPresentation {
    fn from(presentation: hyperscope_app::PresentationReadModel) -> Self {
        Self {
            id: presentation.presentation_id.to_string(),
            title: presentation.title,
            cue_count: presentation.cue_count,
            assets: presentation.assets,
            active: presentation.active,
            animation_residency: presentation.animation_residency.map(Into::into),
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowPresentationAnimationResidency {
    presentation_asset_id: String,
    scene_request_id: String,
    resident_asset_id: String,
}

impl From<PresentationAnimationResidencyBinding> for ShadowPresentationAnimationResidency {
    fn from(binding: PresentationAnimationResidencyBinding) -> Self {
        Self {
            presentation_asset_id: binding.presentation_asset_id.to_string(),
            scene_request_id: binding.scene_request_id.to_string(),
            resident_asset_id: binding.resident_asset_id.to_string(),
        }
    }
}

#[derive(Serialize)]
struct ShadowNavigationSnapshot {
    elapsed_seconds: f64,
    preset: &'static str,
    pending_actions: usize,
    last_applied_sequence: Option<u64>,
    reflection: &'static str,
    reflection_mobius: [f32; 16],
    camera: ShadowCameraSnapshot,
    focus: ShadowFocusSnapshot,
    selected_focus: Option<SelectedFocusJsSnapshot>,
    diagnostics: Vec<String>,
}

#[derive(Serialize)]
struct ShadowInteractionSnapshot {
    revision: String,
    integrated_until_seconds: f64,
    last_applied_sequence: Option<String>,
    hovered: Option<ShadowInteractionHit>,
    active: Option<ShadowInteractionHit>,
    selected: Option<ShadowInteractionIdentity>,
    diagnostics: Vec<String>,
}

#[derive(Serialize)]
struct ShadowInteractionIdentity {
    asset_id: String,
    entity_id: String,
}

impl From<hyperscape_protocol::AssetEntityId> for ShadowInteractionIdentity {
    fn from(identity: hyperscape_protocol::AssetEntityId) -> Self {
        Self {
            asset_id: identity.asset.to_string(),
            entity_id: identity.entity.to_string(),
        }
    }
}

#[derive(Serialize)]
struct ShadowInteractionHit {
    identity: ShadowInteractionIdentity,
    source_bound_center: [f64; 3],
    source_bound_radius: f64,
    source_pivot: [f64; 3],
    output_distance: f64,
    face: Option<u32>,
    barycentric: Option<[f64; 3]>,
}

impl From<InteractionHit> for ShadowInteractionHit {
    fn from(hit: InteractionHit) -> Self {
        Self {
            identity: hit.identity.into(),
            source_bound_center: hit.source_bound.center,
            source_bound_radius: hit.source_bound.radius,
            source_pivot: hit.source_pivot,
            output_distance: hit.output_distance,
            face: hit.surface.map(|surface| surface.face),
            barycentric: hit.surface.map(|surface| surface.barycentric),
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowBackendPickAuthorityResult {
    disposition: &'static str,
    request_id: String,
    target_epoch: u32,
    activation_ready: bool,
    webgpu_frame_revision: Option<String>,
    source_render_call: Option<String>,
    viewport: Option<[u32; 2]>,
    hit: Option<ShadowInteractionHit>,
    surface: Option<crate::main_renderer::ResolvedSurfacePick>,
}

fn pick_authority_disposition_name(
    disposition: InteractionPickAuthorityDisposition,
) -> &'static str {
    match disposition {
        InteractionPickAuthorityDisposition::Current => "accepted",
        InteractionPickAuthorityDisposition::IgnoredSuperseded => "ignored-superseded",
        InteractionPickAuthorityDisposition::IgnoredStaleTargetEpoch => {
            "ignored-stale-target-epoch"
        }
    }
}

#[derive(Serialize)]
struct ShadowSpaceMouseDispatch {
    preset_sequence: String,
    frame_sequence: String,
    preset: &'static str,
    frame: ShadowNavigationFrame,
}

#[derive(Serialize)]
struct ShadowMappedSpaceMouseFrame {
    preset: &'static str,
    frame: ShadowNavigationFrame,
}

impl From<MappedSpaceMouseFrame> for ShadowMappedSpaceMouseFrame {
    fn from(mapped: MappedSpaceMouseFrame) -> Self {
        Self {
            preset: preset_name(mapped.preset),
            frame: mapped.frame.into(),
        }
    }
}

#[derive(Serialize)]
struct ShadowNavigationFrame {
    translation: [f64; 3],
    rotation: [f64; 3],
    dolly_log: f64,
    horizon_locked: bool,
}

#[derive(Serialize)]
struct ShadowTurntableFrame {
    pan: [f64; 2],
    pitch: f64,
    yaw: f64,
    dolly_log: f64,
}

impl From<TurntableFrame> for ShadowTurntableFrame {
    fn from(frame: TurntableFrame) -> Self {
        Self {
            pan: frame.pan,
            pitch: frame.pitch,
            yaw: frame.yaw,
            dolly_log: frame.dolly_log,
        }
    }
}

impl From<NavigationFrame> for ShadowNavigationFrame {
    fn from(frame: NavigationFrame) -> Self {
        Self {
            translation: frame.translation,
            rotation: frame.rotation,
            dolly_log: frame.dolly_log,
            horizon_locked: frame.horizon_locked,
        }
    }
}

#[derive(Serialize)]
struct ShadowCameraSnapshot {
    eye: [f64; 3],
    orientation: [f64; 4],
    right: [f64; 3],
    up: [f64; 3],
    forward: [f64; 3],
    control_distance: f64,
    semantic_target: Option<[f64; 3]>,
    vertical_fov_radians: f64,
    near: f64,
    far: f64,
    camera_transition_remaining: Option<f64>,
    surface_anchor_transition_remaining: Option<f64>,
    surface_anchor_hop_height: Option<f64>,
}

#[derive(Serialize)]
struct ShadowFocusSnapshot {
    center: [f64; 3],
    radius: f64,
    anchored: bool,
    focus_enabled: bool,
    inversion_enabled: bool,
    focus_coordinate: f64,
    angular_aperture: f64,
    focus_transition_remaining: Option<f64>,
}

#[derive(Serialize)]
struct ShadowAsset {
    id: String,
    uri: String,
    status: ShadowAssetStatus,
}

impl From<AssetReadModel> for ShadowAsset {
    fn from(asset: AssetReadModel) -> Self {
        Self {
            id: asset.descriptor.id.to_string(),
            uri: asset.descriptor.uri,
            status: asset.status.into(),
        }
    }
}

#[derive(Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
enum ShadowAssetStatus {
    Loading {
        request_id: String,
    },
    Ready {
        byte_length: usize,
        #[serde(skip_serializing_if = "AssetMetadata::is_empty")]
        metadata: AssetMetadata,
    },
    Failed {
        code: String,
        message: String,
        retryable: bool,
    },
    Cancelled,
}

impl From<AssetStatus> for ShadowAssetStatus {
    fn from(status: AssetStatus) -> Self {
        match status {
            AssetStatus::Loading { request_id } => Self::Loading {
                request_id: request_id.to_string(),
            },
            AssetStatus::Ready {
                byte_length,
                metadata,
                ..
            } => Self::Ready {
                byte_length,
                metadata,
            },
            AssetStatus::Failed {
                code,
                message,
                retryable,
            } => Self::Failed {
                code,
                message,
                retryable,
            },
            AssetStatus::Cancelled => Self::Cancelled,
        }
    }
}

#[derive(Serialize)]
struct ShadowDiagnostic {
    revision: String,
    code: &'static str,
    message: String,
}

fn navigation_to_js(
    frame: AppFrameSnapshot,
    navigation_diagnostics: Vec<String>,
) -> Result<JsValue, JsValue> {
    let basis = frame.camera.basis();
    let reflection_mobius = frame.reflection.mobius().coefficients_f32();
    let focus_transition_remaining = frame
        .focus
        .transition
        .map(|transition| (transition.duration_seconds - transition.elapsed_seconds).max(0.0));
    to_js(&ShadowNavigationSnapshot {
        elapsed_seconds: frame.elapsed_seconds,
        preset: preset_name(frame.navigation_preset),
        pending_actions: frame.pending_navigation_actions,
        last_applied_sequence: frame.last_applied_navigation_sequence,
        reflection: match frame.reflection {
            hyperscape::SphereReflectionState::Identity => "identity",
            hyperscape::SphereReflectionState::Sphere(_) => "sphere_reflection",
        },
        reflection_mobius,
        camera: ShadowCameraSnapshot {
            eye: frame.camera.eye,
            orientation: [
                frame.camera.orientation.w,
                frame.camera.orientation.x,
                frame.camera.orientation.y,
                frame.camera.orientation.z,
            ],
            right: basis.right,
            up: basis.up,
            forward: basis.forward,
            control_distance: frame.camera.control_distance,
            semantic_target: frame.camera.semantic_target,
            vertical_fov_radians: frame.camera.lens.vertical_fov_radians,
            near: frame.camera.lens.near,
            far: frame.camera.lens.far,
            camera_transition_remaining: frame.camera_transition_remaining,
            surface_anchor_transition_remaining: frame.surface_anchor_transition_remaining,
            surface_anchor_hop_height: frame.surface_anchor_hop_height,
        },
        focus: ShadowFocusSnapshot {
            center: frame.focus.sphere.center,
            radius: frame.focus.sphere.radius,
            anchored: frame.focus.anchor.is_some(),
            focus_enabled: frame.focus.focus_enabled,
            inversion_enabled: frame.focus.inversion_enabled,
            focus_coordinate: frame.focus.focus_coordinate,
            angular_aperture: frame.focus.angular_aperture,
            focus_transition_remaining,
        },
        selected_focus: frame.selected_focus.map(SelectedFocusJsSnapshot::from),
        diagnostics: navigation_diagnostics,
    })
}

fn interaction_to_js(
    frame: AppFrameSnapshot,
    diagnostics: Vec<String>,
) -> Result<JsValue, JsValue> {
    to_js(&ShadowInteractionSnapshot {
        revision: frame.interaction.revision.to_string(),
        integrated_until_seconds: frame.interaction.integrated_until_seconds,
        last_applied_sequence: frame
            .interaction
            .last_applied_sequence
            .map(|sequence| sequence.to_string()),
        hovered: frame.interaction.hovered.map(ShadowInteractionHit::from),
        active: frame.interaction.active.map(ShadowInteractionHit::from),
        selected: frame.interaction.selected.map(ShadowInteractionIdentity::from),
        diagnostics,
    })
}

fn commit_to_js(commit: &AppCommit) -> Result<JsValue, JsValue> {
    to_js(&shadow_commit(commit))
}

fn shadow_commit(commit: &AppCommit) -> ShadowCommit {
    let effects = commit.effects.iter().map(shadow_effect).collect();
    ShadowCommit {
        revision: commit.revision.to_string(),
        disposition: match commit.disposition {
            CommitDisposition::Applied => "applied",
            CommitDisposition::IgnoredStale => "ignored_stale",
        },
        published_ui: commit.published_ui,
        effects,
    }
}

fn shadow_effect(effect: &AppEffect) -> ShadowEffect {
    match effect {
        AppEffect::FetchAsset { request_id, asset } => ShadowEffect::FetchAsset {
            request_id: request_id.to_string(),
            asset_id: asset.id.to_string(),
            uri: asset.uri.clone(),
        },
        AppEffect::CancelAssetLoad {
            request_id,
            asset_id,
        } => ShadowEffect::CancelAssetLoad {
            request_id: request_id.to_string(),
            asset_id: asset_id.to_string(),
        },
        AppEffect::InstallPrimaryScene {
            request_id,
            asset_id,
        } => ShadowEffect::InstallPrimaryScene {
            request_id: request_id.to_string(),
            asset_id: asset_id.to_string(),
        },
        AppEffect::CancelPrimarySceneInstall {
            request_id,
            asset_id,
        } => ShadowEffect::CancelPrimarySceneInstall {
            request_id: request_id.to_string(),
            asset_id: asset_id.to_string(),
        },
        AppEffect::SelectAnimationClip {
            job_id,
            scene_request_id,
            asset_id,
            clip_index,
        } => ShadowEffect::SelectAnimationClip {
            job_id: job_id.to_string(),
            scene_request_id: scene_request_id.to_string(),
            asset_id: asset_id.to_string(),
            clip_index: *clip_index,
        },
        AppEffect::CancelAnimationClipSelection {
            job_id,
            scene_request_id,
            asset_id,
            clip_index,
        } => ShadowEffect::CancelAnimationClipSelection {
            job_id: job_id.to_string(),
            scene_request_id: scene_request_id.to_string(),
            asset_id: asset_id.to_string(),
            clip_index: *clip_index,
        },
        AppEffect::PatchLab(effect) => ShadowEffect::PatchLab {
            effect: shadow_patch_lab_effect(effect),
        },
    }
}

fn shadow_patch_lab_effect(effect: &PatchLabEffect) -> ShadowPatchLabEffect {
    match effect {
        PatchLabEffect::BuildGeometry { job_id, geometry } => {
            ShadowPatchLabEffect::BuildGeometry {
                job_id: job_id.to_string(),
                shape: geometry.shape.wire_name(),
                grid: geometry.grid,
                bend_percent: geometry.bend_percent,
            }
        }
        PatchLabEffect::CancelGeometry { job_id } => ShadowPatchLabEffect::CancelGeometry {
            job_id: job_id.to_string(),
        },
        PatchLabEffect::DiscardGeometry { geometry_job_id } => {
            ShadowPatchLabEffect::DiscardGeometry {
                geometry_job_id: geometry_job_id.to_string(),
            }
        }
        PatchLabEffect::EvaluateLod {
            job_id,
            geometry_job_id,
            parameters,
        } => ShadowPatchLabEffect::EvaluateLod {
            job_id: job_id.to_string(),
            geometry_job_id: geometry_job_id.to_string(),
            field: parameters.field.wire_name(),
            phase_microradians: parameters.phase_microradians,
            min_exponent: parameters.min_exponent,
            max_exponent: parameters.max_exponent,
            manual_edge_exponents: parameters.manual_edge_exponents,
            atlas_exponent: parameters.atlas_exponent,
            max_face_edge_ratio: parameters.max_face_edge_ratio,
        },
        PatchLabEffect::CancelLod {
            job_id,
            geometry_job_id,
        } => ShadowPatchLabEffect::CancelLod {
            job_id: job_id.to_string(),
            geometry_job_id: geometry_job_id.to_string(),
        },
    }
}

fn peer_receipt_to_js(receipt: &LocalPeerReceipt) -> Result<JsValue, JsValue> {
    to_js(&ShadowLocalPeerReceipt {
        lane: match receipt.lane {
            LocalPeerLane::Authored => "authored",
            LocalPeerLane::Presence => "presence",
        },
        disposition: match receipt.disposition {
            LocalPeerDisposition::Applied => "applied",
            LocalPeerDisposition::IgnoredStale => "ignored_stale",
            LocalPeerDisposition::IgnoredDuplicate => "ignored_duplicate",
            LocalPeerDisposition::IgnoredEcho => "ignored_echo",
        },
        projection_revision: receipt
            .projection_revision
            .map(|revision| revision.to_string()),
        commit: receipt.commit.as_ref().map(shadow_commit),
    })
}

fn request_id_from_str(value: &str) -> Result<RequestId, JsValue> {
    RequestId::new(parse_uuid(value, "request ID")?).map_err(js_error)
}

fn asset_id_from_str(value: &str) -> Result<AssetId, JsValue> {
    AssetId::new(parse_uuid(value, "asset ID")?).map_err(js_error)
}

fn entity_id_from_str(value: &str) -> Result<EntityId, JsValue> {
    EntityId::new(parse_uuid(value, "entity ID")?).map_err(js_error)
}

fn interaction_surface(
    face: i32,
    barycentric: &[f64],
) -> Result<Option<(u32, [f64; 3])>, JsValue> {
    match face {
        -1 if barycentric.is_empty() => Ok(None),
        -1 => Err(JsValue::from_str(
            "entity-level interaction hover must omit barycentric coordinates",
        )),
        face if face >= 0 => Ok(Some((
            face as u32,
            vector3(barycentric, "interaction barycentric coordinates")?,
        ))),
        _ => Err(JsValue::from_str(
            "interaction face must be -1 or nonnegative",
        )),
    }
}

fn presentation_action_from_wire(
    action: &str,
    cue_id: &str,
) -> Result<PresentationAction, JsValue> {
    match action {
        "start" => Ok(PresentationAction::Start),
        "advance" => Ok(PresentationAction::Advance),
        "reverse" => Ok(PresentationAction::Reverse),
        "jump" => Ok(PresentationAction::JumpToCue(parse_uuid(cue_id, "cue ID")?)),
        "clear" => Ok(PresentationAction::Clear),
        _ => Err(JsValue::from_str("unknown presentation action")),
    }
}

fn packed_node_source(node: ShadowPackedNodeInput) -> Result<PackedNodeSource, JsValue> {
    Ok(PackedNodeSource {
        packed_node: node.packed_node,
        source_node: node.source_node,
        entity: node
            .entity_id
            .as_deref()
            .map(entity_id_from_str)
            .transpose()?,
        source_world: node.source_world,
    })
}

fn parse_uuid(value: &str, context: &str) -> Result<Uuid, JsValue> {
    Uuid::parse_str(value)
        .map_err(|error| JsValue::from_str(&format!("{context} must be a UUID: {error}")))
}

const ANIMATION_POSE_PACKET_LEN: usize = 4;

fn validate_animation_pose_output(output: &[f64]) -> Result<(), JsValue> {
    if output.len() == ANIMATION_POSE_PACKET_LEN {
        Ok(())
    } else {
        Err(JsValue::from_str(
            "animation pose output must contain exactly four f64 values",
        ))
    }
}

fn write_animation_pose_stamp(
    output: &mut [f64],
    stamp: AnimationPoseStamp,
) -> Result<(), JsValue> {
    validate_animation_pose_output(output)?;
    output.copy_from_slice(&[
        stamp.clip_time_seconds,
        stamp.sample_time_seconds,
        f64::from(stamp.revision),
        f64::from(stamp.continuity_epoch),
    ]);
    Ok(())
}

fn to_js(value: &impl Serialize) -> Result<JsValue, JsValue> {
    serde_wasm_bindgen::to_value(value).map_err(js_error)
}

fn js_error(error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&error.to_string())
}

fn parse_patch_lab_job_id(value: &str) -> Result<u64, JsValue> {
    value
        .parse::<u64>()
        .map_err(|error| js_error(format!("Patch Lab job ID is invalid: {error}")))
}

#[allow(clippy::too_many_arguments)]
fn space_mouse_camera_input(
    normalized_axes: &[f32],
    preset: &str,
    swap_yz: bool,
    invert_pan: f64,
    invert_rotate: f64,
    delta_seconds: f64,
    registered_linear_speed: f64,
    move_gain: f64,
    rotate_gain: f64,
    horizon_lock_requested: bool,
) -> Result<MappedSpaceMouseFrame, JsValue> {
    let normalized_axes: [f32; 6] = normalized_axes.try_into().map_err(|_| {
        JsValue::from_str("SpaceMouse input must contain exactly six normalized axes")
    })?;
    map_space_mouse_camera(SpaceMouseCameraInput {
        normalized_axes: normalized_axes.map(f64::from),
        mapping: SpaceMouseMapping {
            preset: parse_preset(preset)?,
            swap_yz,
            invert_pan: parse_space_mouse_mask(invert_pan)?,
            invert_rotate: parse_space_mouse_mask(invert_rotate)?,
        },
        delta_seconds,
        registered_linear_speed,
        move_gain,
        rotate_gain,
        horizon_lock_requested,
    })
    .map_err(js_error)
}

fn pointer_turntable_input(
    delta_x: f64,
    delta_y: f64,
    gesture: u8,
    control_distance: f64,
) -> Result<TurntableFrame, JsValue> {
    let gesture = match gesture {
        0 => PointerTurntableGesture::Orbit,
        1 => PointerTurntableGesture::Pan,
        2 => PointerTurntableGesture::Wheel,
        _ => {
            return Err(JsValue::from_str(
                "pointer gesture must be 0 (orbit), 1 (pan), or 2 (wheel)",
            ))
        }
    };
    map_pointer_turntable(PointerTurntableInput {
        delta: [delta_x, delta_y],
        gesture,
        control_distance,
    })
    .map_err(js_error)
}

fn parse_space_mouse_mask(value: f64) -> Result<u8, JsValue> {
    if !value.is_finite() || value.fract() != 0.0 || !(0.0..=7.0).contains(&value) {
        return Err(JsValue::from_str(
            "SpaceMouse inversion masks must be finite integers from 0 through 7",
        ));
    }
    Ok(value as u8)
}
