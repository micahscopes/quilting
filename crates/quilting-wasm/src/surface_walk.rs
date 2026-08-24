use hyperscape::{
    CameraBasis, CameraRig, PerspectiveLens, SurfaceWalkContactFrame, SurfaceWalkController,
    SurfaceWalkControls, SurfaceWalkFrame, SurfaceWalkInput, SurfaceWalkMetrics, SurfaceWalkMotion,
};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

#[wasm_bindgen(typescript_custom_section)]
const SURFACE_WALK_TYPES: &'static str = r#"
export type SurfaceWalkVec3 = readonly [number, number, number] | Float32Array | Float64Array;
export interface SurfaceWalkCameraInput {
  eye: SurfaceWalkVec3;
  forward: SurfaceWalkVec3;
  up: SurfaceWalkVec3;
  controlDistance: number;
  verticalFovRadians: number;
  near: number;
  far: number;
}
export interface SurfaceWalkControlsInput {
  baseRadiiPerSecond: number;
  baseEyeHeight: number;
  speedOctaveSteps: number;
  bodyScaleOctaveSteps: number;
  eyeHeightOctaveSteps: number;
  smoothingSeconds: number;
  tangentPullFraction: number;
  fastMultiplier: number;
  defaultNear: number;
  minimumNear: number;
  nearEyeFraction: number;
}
export interface SurfaceWalkMotionRequest {
  camera: SurfaceWalkCameraInput;
  outputNormal: SurfaceWalkVec3;
  sceneRadius: number;
  controls: SurfaceWalkControlsInput;
  input: { forwardAxis: number; rightAxis: number; fast: boolean };
}
export interface SurfaceWalkFollowRequest {
  camera: SurfaceWalkCameraInput;
  outputPosition: SurfaceWalkVec3;
  outputNormal: SurfaceWalkVec3;
  sceneRadius: number;
  controls: SurfaceWalkControlsInput;
  deltaSeconds: number;
  orient: boolean;
  captureRelativeView: boolean;
}
export interface SurfaceWalkMetricsSnapshot {
  bodyScale: number;
  radiiPerSecond: number;
  speed: number;
  eyeHeight: number;
  near: number;
}
export interface SurfaceWalkMotionSnapshot {
  desiredOutputVelocity: readonly [number, number, number];
  tangentForward: readonly [number, number, number];
  tangentRight: readonly [number, number, number];
  metrics: SurfaceWalkMetricsSnapshot;
}
export interface SurfaceWalkCameraSnapshot {
  eye: readonly [number, number, number];
  right: readonly [number, number, number];
  up: readonly [number, number, number];
  forward: readonly [number, number, number];
  controlDistance: number;
  verticalFovRadians: number;
  near: number;
  far: number;
}
export interface SurfaceWalkFrameSnapshot {
  camera: SurfaceWalkCameraSnapshot;
  filteredPosition: readonly [number, number, number];
  filteredNormal: readonly [number, number, number];
  tangentForward?: readonly [number, number, number];
  relativePitchRadians?: number;
  metrics: SurfaceWalkMetricsSnapshot;
}
export interface ComposedSurfaceWalkCameraSnapshot {
  eye: readonly [number, number, number];
  right: readonly [number, number, number];
  up: readonly [number, number, number];
  forward: readonly [number, number, number];
  control_distance: number;
  vertical_fov_radians: number;
  near: number;
  far: number;
}
export interface ComposedSurfaceWalkMetricsSnapshot {
  body_scale: number;
  radii_per_second: number;
  speed: number;
  eye_height: number;
  near: number;
}
export interface SurfacePoseSampleSnapshot {
  clip_time_seconds: number;
  sample_time_seconds: number;
  revision: number;
  continuity_epoch: number;
  continuous: boolean;
  sample_delta_seconds?: number | null;
  velocity_rebased: boolean;
}
export interface ComposedSurfaceWalkSnapshot {
  status: 'attached' | 'detached';
  phase: 'anchoring' | 'walking' | 'detached';
  detach_reason?: string | null;
  node?: number | null;
  face?: number | null;
  barycentric?: readonly [number, number, number] | null;
  output_position?: readonly [number, number, number] | null;
  output_normal?: readonly [number, number, number] | null;
  surface_velocity?: readonly [number, number, number] | null;
  projected_output_velocity: readonly [number, number, number];
  desired_output_velocity?: readonly [number, number, number] | null;
  condition_number?: number | null;
  substeps: number;
  edge_crossings: number;
  camera: ComposedSurfaceWalkCameraSnapshot;
  target_camera?: ComposedSurfaceWalkCameraSnapshot | null;
  filtered_position?: readonly [number, number, number] | null;
  filtered_normal?: readonly [number, number, number] | null;
  tangent_forward?: readonly [number, number, number] | null;
  relative_pitch_radians?: number | null;
  metrics?: ComposedSurfaceWalkMetricsSnapshot | null;
  anchor_transition_remaining_seconds?: number | null;
  pose_sample?: SurfacePoseSampleSnapshot | null;
}
export interface ComposedSurfaceWalkErrorSnapshot {
  status: 'error';
  error: string;
}
export type ComposedSurfaceWalkResult =
  | ComposedSurfaceWalkSnapshot
  | ComposedSurfaceWalkErrorSnapshot
  | null;
export interface SurfaceWalkReflectionTransportSnapshot {
  legacy_attached: boolean;
  composed_attached: boolean;
  composed_follower_transported: boolean;
  normal_side_flipped: boolean;
  anchor_transition_cancelled: boolean;
  legacy_previous_position_transported: boolean;
  composed_previous_position_transported: boolean;
}
export type SurfaceWalkReflectionTransportResult =
  | SurfaceWalkReflectionTransportSnapshot
  | null;
"#;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(typescript_type = "SurfaceWalkMotionRequest")]
    pub type SurfaceWalkMotionRequestJs;
    #[wasm_bindgen(typescript_type = "SurfaceWalkFollowRequest")]
    pub type SurfaceWalkFollowRequestJs;
    #[wasm_bindgen(typescript_type = "SurfaceWalkMotionSnapshot")]
    pub type SurfaceWalkMotionSnapshotJs;
    #[wasm_bindgen(typescript_type = "SurfaceWalkFrameSnapshot")]
    pub type SurfaceWalkFrameSnapshotJs;
    #[wasm_bindgen(typescript_type = "ComposedSurfaceWalkResult")]
    pub type ComposedSurfaceWalkResultJs;
    #[wasm_bindgen(typescript_type = "SurfaceWalkReflectionTransportResult")]
    pub type SurfaceWalkReflectionTransportResultJs;
}

/// Offline-capable WASM facade for the Rust surface-walk authority.
///
/// It deliberately has no renderer, DOM, HID, or worker dependency, allowing
/// generated-WASM parity traces to run in Node before browser authority moves.
/// Rejected view frames preserve the preceding oracle state; this facade does
/// not own topology attachment and therefore never infers a detach by itself.
#[wasm_bindgen]
pub struct HyperscopeSurfaceWalk {
    controller: SurfaceWalkController,
}

#[wasm_bindgen]
impl HyperscopeSurfaceWalk {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            controller: SurfaceWalkController::default(),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn active(&self) -> bool {
        self.controller.is_active()
    }

    pub fn reset(&mut self) {
        self.controller.reset();
    }

    #[wasm_bindgen(js_name = planMotion)]
    pub fn plan_motion(
        &self,
        request: SurfaceWalkMotionRequestJs,
    ) -> Result<SurfaceWalkMotionSnapshotJs, JsValue> {
        let request: JsValue = request.into();
        validate_motion_request(&request)?;
        let request: MotionRequest = serde_wasm_bindgen::from_value(request).map_err(js_error)?;
        let motion = self
            .controller
            .plan_motion(
                &request.camera.rig()?,
                vector3(&request.output_normal, "outputNormal")?,
                request.scene_radius,
                request.controls.into(),
                request.input.into(),
            )
            .map_err(js_error)?;
        Ok(to_value(&MotionSnapshot::from(motion))?.unchecked_into())
    }

    #[wasm_bindgen(js_name = followFrame)]
    pub fn follow_frame(
        &mut self,
        request: SurfaceWalkFollowRequestJs,
    ) -> Result<SurfaceWalkFrameSnapshotJs, JsValue> {
        let request: JsValue = request.into();
        validate_follow_request(&request)?;
        let request: FollowRequest = serde_wasm_bindgen::from_value(request).map_err(js_error)?;
        let frame = self
            .controller
            .follow_frame(
                &request.camera.rig()?,
                SurfaceWalkContactFrame {
                    output_position: vector3(&request.output_position, "outputPosition")?,
                    output_normal: vector3(&request.output_normal, "outputNormal")?,
                },
                request.scene_radius,
                request.controls.into(),
                request.delta_seconds,
                request.orient,
                request.capture_relative_view,
            )
            .map_err(js_error)?;
        Ok(to_value(&FrameSnapshot::from(frame))?.unchecked_into())
    }
}

impl Default for HyperscopeSurfaceWalk {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct MotionRequest {
    camera: CameraRequest,
    output_normal: Vec<f64>,
    scene_radius: f64,
    controls: ControlsRequest,
    input: InputRequest,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct FollowRequest {
    camera: CameraRequest,
    output_position: Vec<f64>,
    output_normal: Vec<f64>,
    scene_radius: f64,
    controls: ControlsRequest,
    delta_seconds: f64,
    orient: bool,
    capture_relative_view: bool,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CameraRequest {
    eye: Vec<f64>,
    forward: Vec<f64>,
    up: Vec<f64>,
    control_distance: f64,
    vertical_fov_radians: f64,
    near: f64,
    far: f64,
}

impl CameraRequest {
    fn rig(&self) -> Result<CameraRig, JsValue> {
        CameraRig::new(
            vector3(&self.eye, "camera.eye")?,
            CameraBasis::from_forward_up(
                vector3(&self.forward, "camera.forward")?,
                vector3(&self.up, "camera.up")?,
            )
            .map_err(js_error)?,
            self.control_distance,
            None,
            PerspectiveLens {
                vertical_fov_radians: self.vertical_fov_radians,
                near: self.near,
                far: self.far,
            },
        )
        .map_err(js_error)
    }
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ControlsRequest {
    base_radii_per_second: f64,
    base_eye_height: f64,
    speed_octave_steps: f64,
    body_scale_octave_steps: f64,
    eye_height_octave_steps: f64,
    smoothing_seconds: f64,
    tangent_pull_fraction: f64,
    fast_multiplier: f64,
    default_near: f64,
    minimum_near: f64,
    near_eye_fraction: f64,
}

impl From<ControlsRequest> for SurfaceWalkControls {
    fn from(value: ControlsRequest) -> Self {
        Self {
            base_radii_per_second: value.base_radii_per_second,
            base_eye_height: value.base_eye_height,
            speed_octave_steps: value.speed_octave_steps,
            body_scale_octave_steps: value.body_scale_octave_steps,
            eye_height_octave_steps: value.eye_height_octave_steps,
            smoothing_seconds: value.smoothing_seconds,
            tangent_pull_fraction: value.tangent_pull_fraction,
            fast_multiplier: value.fast_multiplier,
            default_near: value.default_near,
            minimum_near: value.minimum_near,
            near_eye_fraction: value.near_eye_fraction,
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct InputRequest {
    forward_axis: f64,
    right_axis: f64,
    fast: bool,
}

impl From<InputRequest> for SurfaceWalkInput {
    fn from(value: InputRequest) -> Self {
        Self {
            forward_axis: value.forward_axis,
            right_axis: value.right_axis,
            fast: value.fast,
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct MetricsSnapshot {
    body_scale: f64,
    radii_per_second: f64,
    speed: f64,
    eye_height: f64,
    near: f64,
}

impl From<SurfaceWalkMetrics> for MetricsSnapshot {
    fn from(value: SurfaceWalkMetrics) -> Self {
        Self {
            body_scale: value.body_scale,
            radii_per_second: value.radii_per_second,
            speed: value.speed,
            eye_height: value.eye_height,
            near: value.near,
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct MotionSnapshot {
    desired_output_velocity: [f64; 3],
    tangent_forward: [f64; 3],
    tangent_right: [f64; 3],
    metrics: MetricsSnapshot,
}

impl From<SurfaceWalkMotion> for MotionSnapshot {
    fn from(value: SurfaceWalkMotion) -> Self {
        Self {
            desired_output_velocity: value.desired_output_velocity,
            tangent_forward: value.tangent_forward,
            tangent_right: value.tangent_right,
            metrics: value.metrics.into(),
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct FrameSnapshot {
    camera: CameraSnapshot,
    filtered_position: [f64; 3],
    filtered_normal: [f64; 3],
    tangent_forward: Option<[f64; 3]>,
    relative_pitch_radians: Option<f64>,
    metrics: MetricsSnapshot,
}

impl From<SurfaceWalkFrame> for FrameSnapshot {
    fn from(value: SurfaceWalkFrame) -> Self {
        Self {
            camera: value.camera.into(),
            filtered_position: value.filtered_position,
            filtered_normal: value.filtered_normal,
            tangent_forward: value.tangent_forward,
            relative_pitch_radians: value.relative_pitch_radians,
            metrics: value.metrics.into(),
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CameraSnapshot {
    eye: [f64; 3],
    right: [f64; 3],
    up: [f64; 3],
    forward: [f64; 3],
    control_distance: f64,
    vertical_fov_radians: f64,
    near: f64,
    far: f64,
}

impl From<CameraRig> for CameraSnapshot {
    fn from(value: CameraRig) -> Self {
        let basis = value.basis();
        Self {
            eye: value.eye,
            right: basis.right,
            up: basis.up,
            forward: basis.forward,
            control_distance: value.control_distance,
            vertical_fov_radians: value.lens.vertical_fov_radians,
            near: value.lens.near,
            far: value.lens.far,
        }
    }
}

fn to_value(value: &impl Serialize) -> Result<JsValue, JsValue> {
    serde_wasm_bindgen::to_value(value).map_err(js_error)
}

fn vector3(values: &[f64], label: &str) -> Result<[f64; 3], JsValue> {
    values
        .try_into()
        .map_err(|_| JsValue::from_str(&format!("{label} must contain exactly three numbers")))
}

fn validate_motion_request(value: &JsValue) -> Result<(), JsValue> {
    validate_object_keys(
        value,
        "surface-walk motion request",
        &["camera", "outputNormal", "sceneRadius", "controls", "input"],
    )?;
    validate_common_nested_objects(value)?;
    validate_object_keys(
        &field(value, "input")?,
        "input",
        &["forwardAxis", "rightAxis", "fast"],
    )
}

fn validate_follow_request(value: &JsValue) -> Result<(), JsValue> {
    validate_object_keys(
        value,
        "surface-walk follow request",
        &[
            "camera",
            "outputPosition",
            "outputNormal",
            "sceneRadius",
            "controls",
            "deltaSeconds",
            "orient",
            "captureRelativeView",
        ],
    )?;
    validate_common_nested_objects(value)
}

fn validate_common_nested_objects(value: &JsValue) -> Result<(), JsValue> {
    validate_object_keys(
        &field(value, "camera")?,
        "camera",
        &[
            "eye",
            "forward",
            "up",
            "controlDistance",
            "verticalFovRadians",
            "near",
            "far",
        ],
    )?;
    validate_object_keys(
        &field(value, "controls")?,
        "controls",
        &[
            "baseRadiiPerSecond",
            "baseEyeHeight",
            "speedOctaveSteps",
            "bodyScaleOctaveSteps",
            "eyeHeightOctaveSteps",
            "smoothingSeconds",
            "tangentPullFraction",
            "fastMultiplier",
            "defaultNear",
            "minimumNear",
            "nearEyeFraction",
        ],
    )
}

fn validate_object_keys(value: &JsValue, label: &str, allowed: &[&str]) -> Result<(), JsValue> {
    if !value.is_object() || value.is_null() {
        return Err(JsValue::from_str(&format!("{label} must be an object")));
    }
    let object: &js_sys::Object = value.unchecked_ref();
    for key in js_sys::Object::keys(object).iter() {
        let Some(key) = key.as_string() else {
            return Err(JsValue::from_str(&format!("{label} has a non-string key")));
        };
        if !allowed.contains(&key.as_str()) {
            return Err(JsValue::from_str(&format!(
                "{label} contains unknown field '{key}'"
            )));
        }
    }
    Ok(())
}

fn field(value: &JsValue, name: &str) -> Result<JsValue, JsValue> {
    js_sys::Reflect::get(value, &JsValue::from_str(name))
        .map_err(|_| JsValue::from_str(&format!("could not read field '{name}'")))
}

fn js_error(error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&error.to_string())
}
