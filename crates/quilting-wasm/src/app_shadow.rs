use crate::navigation::{
    asset_entity_id, optional_vector3, parse_easing, parse_preset, perspective_lens, preset_name,
    synchronized_navigation_state, vector3, SelectedFocusJsSnapshot,
};
use hyperscape::{
    extract_packed_scene, map_space_mouse_camera, CameraBasis, CameraRig, FocusSphere,
    LayerTransform, MappedSpaceMouseFrame, NavigationAction, NavigationFrame, PackedAssetInstance,
    PackedNodeSource, PackedNodeTransformSource, Presentation, PresentationSnapshot,
    SpaceMouseCameraInput, SpaceMouseMapping, SurfaceAnchorTarget,
};
use hyperscape_protocol::{AssetDescriptor, AssetId, AuthoredEnvelope, EntityId, RequestId};
use hyperscope_app::{
    session_node_identity, AppCommit, AppEffect, AppEvent, AppFrameSnapshot, AppStore,
    AssetLoadCompletion, AssetLoadOutcome, AssetStatus, AuthoredRevision, CommitDisposition,
    EffectCompletion, FrameTick, NavigationSynchronization, PresentationAction, SemanticAction,
    Timed,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use uuid::Uuid;
use wasm_bindgen::prelude::*;

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

/// Opt-in WASM adapter for comparing browser asset jobs with the Rust app
/// reducer. It observes fetch/file acquisition only and never loads a model,
/// mutates the renderer, or chooses a browser transport.
#[wasm_bindgen]
pub struct HyperscopeAppShadow {
    store: AppStore,
}

#[wasm_bindgen]
impl HyperscopeAppShadow {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            store: AppStore::default(),
        }
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
                value: SemanticAction::RequestAsset { request_id, asset },
            }))
            .map_err(js_error)?;
        commit_to_js(&commit)
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
        self.complete_asset(
            request_id,
            asset_id,
            AssetLoadOutcome::Loaded {
                byte_length: byte_length as usize,
                content_digest: None,
            },
        )
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
        self.complete_asset(
            request_id,
            asset_id,
            AssetLoadOutcome::Failed {
                code: code.to_owned(),
                message: message.to_owned(),
                retryable,
            },
        )
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
    /// The ordinary `advanceFrame` API remains available to orchestration
    /// adapters that need the commit disposition and effects.
    #[wasm_bindgen(js_name = advanceFrameQuiet)]
    pub fn advance_frame_quiet(
        &self,
        elapsed_seconds: f64,
        delta_seconds: f64,
    ) -> Result<(), JsValue> {
        self.store
            .dispatch(AppEvent::Frame(FrameTick {
                elapsed_seconds,
                delta_seconds,
            }))
            .map_err(js_error)?;
        Ok(())
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

    #[wasm_bindgen(js_name = tickNavigation)]
    pub fn tick_navigation(&self, delta_seconds: f64) -> Result<JsValue, JsValue> {
        let current = self.store.frame_snapshot();
        self.advance_frame_quiet(current.elapsed_seconds + delta_seconds, delta_seconds)?;
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

    /// Mirror low-rate cue intent. This shadow compares resolved desired state;
    /// the existing navigation controller remains frame/camera authority until
    /// a separate pose-parity gate is enabled.
    #[wasm_bindgen(js_name = present)]
    pub fn present(&self, sequence: u32, action: &str, cue_id: &str) -> Result<JsValue, JsValue> {
        let action = match action {
            "start" => PresentationAction::Start,
            "advance" => PresentationAction::Advance,
            "reverse" => PresentationAction::Reverse,
            "jump" => PresentationAction::JumpToCue(parse_uuid(cue_id, "cue ID")?),
            "clear" => PresentationAction::Clear,
            _ => return Err(JsValue::from_str("unknown presentation action")),
        };
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
                        .map(|node| {
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
                        })
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

    /// A bounded, UI-shaped projection. The app store publishes runtime asset,
    /// authored asset/entity, and diagnostic vectors before its summary
    /// revision commit fence.
    pub fn snapshot(&self) -> Result<JsValue, JsValue> {
        let summary = self.store.summary_snapshot();
        let authored = self.store.authored_scene_snapshot();
        let assets = self
            .store
            .asset_snapshot()
            .into_iter()
            .map(|asset| ShadowAsset {
                id: asset.descriptor.id.to_string(),
                uri: asset.descriptor.uri,
                status: ShadowAssetStatus::from(asset.status),
            })
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
        let presentation =
            self.store
                .presentation_snapshot()
                .map(|presentation| ShadowPresentation {
                    id: presentation.presentation_id.to_string(),
                    title: presentation.title,
                    cue_count: presentation.cue_count,
                    active: presentation.active,
                });
        to_js(&ShadowSnapshot {
            revision: summary.revision.to_string(),
            assets,
            loading_assets: summary.loading_assets,
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
    fn dispatch_navigation(&self, action: NavigationAction) -> Result<u64, JsValue> {
        let (sequence, _) = self.store.dispatch_navigation(action).map_err(js_error)?;
        Ok(sequence)
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
    ) -> Result<JsValue, JsValue> {
        let commit = self
            .store
            .dispatch(AppEvent::EffectCompleted(EffectCompletion::AssetLoad(
                AssetLoadCompletion {
                    request_id: request_id_from_str(request_id)?,
                    asset_id: asset_id_from_str(asset_id)?,
                    outcome,
                },
            )))
            .map_err(js_error)?;
        commit_to_js(&commit)
    }
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
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShadowSnapshot {
    revision: String,
    assets: Vec<ShadowAsset>,
    loading_assets: usize,
    authored_projection_revision: Option<String>,
    authored_assets: Vec<ShadowAuthoredAsset>,
    authored_entities: Vec<ShadowAuthoredEntity>,
    diagnostics: Vec<ShadowDiagnostic>,
    presentation: Option<ShadowPresentation>,
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
#[serde(rename_all = "camelCase")]
struct ShadowPackedNodeInput {
    packed_node: u32,
    source_node: u32,
    entity_id: Option<String>,
    source_world: [f32; 16],
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
struct ShadowPresentation {
    id: String,
    title: String,
    cue_count: usize,
    active: Option<PresentationSnapshot>,
}

#[derive(Serialize)]
struct ShadowNavigationSnapshot {
    elapsed_seconds: f64,
    preset: &'static str,
    pending_actions: usize,
    last_applied_sequence: Option<u64>,
    reflection: &'static str,
    camera: ShadowCameraSnapshot,
    focus: ShadowFocusSnapshot,
    selected_focus: Option<SelectedFocusJsSnapshot>,
    diagnostics: Vec<String>,
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

#[derive(Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
enum ShadowAssetStatus {
    Loading {
        request_id: String,
    },
    Ready {
        byte_length: usize,
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
            AssetStatus::Ready { byte_length, .. } => Self::Ready { byte_length },
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

fn commit_to_js(commit: &AppCommit) -> Result<JsValue, JsValue> {
    let effects = commit
        .effects
        .iter()
        .map(|effect| match effect {
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
        })
        .collect();
    to_js(&ShadowCommit {
        revision: commit.revision.to_string(),
        disposition: match commit.disposition {
            CommitDisposition::Applied => "applied",
            CommitDisposition::IgnoredStale => "ignored_stale",
        },
        published_ui: commit.published_ui,
        effects,
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

fn parse_uuid(value: &str, context: &str) -> Result<Uuid, JsValue> {
    Uuid::parse_str(value)
        .map_err(|error| JsValue::from_str(&format!("{context} must be a UUID: {error}")))
}

fn to_js(value: &impl Serialize) -> Result<JsValue, JsValue> {
    serde_wasm_bindgen::to_value(value).map_err(js_error)
}

fn js_error(error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&error.to_string())
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

fn parse_space_mouse_mask(value: f64) -> Result<u8, JsValue> {
    if !value.is_finite() || value.fract() != 0.0 || !(0.0..=7.0).contains(&value) {
        return Err(JsValue::from_str(
            "SpaceMouse inversion masks must be finite integers from 0 through 7",
        ));
    }
    Ok(value as u8)
}
