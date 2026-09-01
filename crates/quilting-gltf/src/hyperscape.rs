//! Hyperscape's provisional glTF 2.0 interchange in `extras.hyperscape`.
//!
//! The data model is deliberately independent of the container. A future
//! registered vendor extension can reuse it, but version 0.1 does not claim a
//! reserved Khronos or multi-vendor prefix.

use hyperscape_protocol::{AssetId, ConformalFrameId, EntityId, SurfaceFrameOrientation};
use quilting_core::{
    AnchorState, ConformalFrameForest, ConformalGenerator, ConformalTransformChain, FrameId,
    RoundWall, RoundWallGeometry, RoundWallSet, WallId,
};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
#[cfg(test)]
use uuid::Uuid;

pub const HYPERSCAPE_INTERCHANGE_VERSION: &str = "0.1";
const GLB_JSON_CHUNK: u32 = 0x4e4f_534a;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HyperscapePayload {
    pub version: String,
    /// Durable identity of this authored asset. Legacy 0.1 payloads may omit
    /// it; collaborative authoring requires it before addressing entities.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub asset_id: Option<AssetId>,
    #[serde(default)]
    pub frames: Vec<HyperscapeFrame>,
    #[serde(default)]
    pub walls: Vec<HyperscapeWall>,
    #[serde(default)]
    pub anchors: Vec<HyperscapeAnchor>,
    #[serde(default)]
    pub paths: Vec<HyperscapePath>,
    #[serde(default)]
    pub constraints: Vec<HyperscapeConstraint>,
}

impl Default for HyperscapePayload {
    fn default() -> Self {
        Self {
            version: HYPERSCAPE_INTERCHANGE_VERSION.into(),
            asset_id: None,
            frames: Vec::new(),
            walls: Vec::new(),
            anchors: Vec::new(),
            paths: Vec::new(),
            constraints: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HyperscapeFrame {
    /// Durable authored identity. Array position is retained only as a compact
    /// glTF/runtime handle and must not be used by editors or replicated
    /// authored operations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stable_id: Option<ConformalFrameId>,
    pub name: String,
    pub parent: Option<usize>,
    #[serde(default)]
    pub generators: Vec<ConformalGenerator>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HyperscapeWall {
    pub name: String,
    pub frame: usize,
    pub geometry: RoundWallGeometry,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HyperscapeAnchor {
    pub name: String,
    pub frame: usize,
    #[serde(default)]
    pub flipped_walls: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HyperscapePathKeyframe {
    pub time_seconds: f64,
    pub point: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HyperscapePath {
    pub name: String,
    pub node: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coordinate_frame: Option<usize>,
    #[serde(default)]
    pub looping: bool,
    pub keyframes: Vec<HyperscapePathKeyframe>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub transitions: Vec<HyperscapePathTransition>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HyperscapePathTransition {
    pub time_seconds: f64,
    pub frame: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anchor: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum HyperscapeConstraint {
    Track {
        node: usize,
        target_node: usize,
        #[serde(default)]
        local_offset: [f64; 3],
    },
    ProjectionCamera {
        node: usize,
        frame: usize,
    },
    /// Drive one conformal frame from a stable material point on a posed QB
    /// surface. The target entity is durable; its glTF node index is only a
    /// container lookup.
    SurfacePin {
        frame: usize,
        target_entity: EntityId,
        face: u32,
        barycentric: [f64; 3],
        #[serde(default = "default_normal_sign")]
        normal_sign: i8,
        #[serde(default)]
        heading_radians: f64,
        #[serde(default = "default_uniform_scale")]
        uniform_scale: f64,
        #[serde(default)]
        orientation: SurfaceFrameOrientation,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        local_offset: Vec<ConformalGenerator>,
    },
}

const fn default_normal_sign() -> i8 {
    1
}

const fn default_uniform_scale() -> f64 {
    1.0
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HyperscapeNodeBinding {
    /// Durable identity shared by Blender, Hyperscape, presentations, and
    /// replicated authored operations. Array position remains a container
    /// detail and must not become a persistent reference.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stable_id: Option<EntityId>,
    pub frame: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anchor: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<usize>,
}

/// Authored payload plus the binding extracted from each ordinary glTF node.
#[derive(Debug, Clone, PartialEq)]
pub struct HyperscapeAsset {
    pub payload: HyperscapePayload,
    /// Indexed exactly like the ordinary glTF node array.
    pub node_bindings: Vec<Option<HyperscapeNodeBinding>>,
}

/// Validated runtime structures. The original [`HyperscapeAsset`] remains the
/// serializable source of paths and constraints.
#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeHyperscapeScene {
    pub frames: ConformalFrameForest,
    /// Dense runtime handle to durable authored identity. Legacy 0.1 assets
    /// may omit an identity; newly authored frames should always provide one.
    pub frame_stable_ids: Vec<Option<ConformalFrameId>>,
    /// Durable authored identity to dense runtime handle.
    pub frames_by_stable_id: BTreeMap<ConformalFrameId, FrameId>,
    pub walls: RoundWallSet,
    pub anchors: Vec<AnchorState>,
}

impl HyperscapeAsset {
    pub fn validate(&self) -> Result<RuntimeHyperscapeScene, HyperscapeGltfError> {
        let node_count = self.node_bindings.len();
        if self.payload.version != HYPERSCAPE_INTERCHANGE_VERSION {
            return Err(HyperscapeGltfError::Validation(format!(
                "unsupported version {:?}; expected {:?}",
                self.payload.version, HYPERSCAPE_INTERCHANGE_VERSION
            )));
        }
        if let Some(asset_id) = self.payload.asset_id {
            asset_id
                .validate()
                .map_err(|error| validation(format!("asset_id: {error}")))?;
        }
        let mut nodes_by_stable_id = BTreeMap::new();
        for (node, binding) in self.node_bindings.iter().enumerate() {
            let Some(stable_id) = binding.as_ref().and_then(|binding| binding.stable_id) else {
                continue;
            };
            stable_id
                .validate()
                .map_err(|error| validation(format!("node {node}: {error}")))?;
            if nodes_by_stable_id.insert(stable_id, node).is_some() {
                return Err(validation(format!(
                    "node {node} repeats stable entity ID {stable_id}"
                )));
            }
        }

        let mut frames = ConformalFrameForest::new();
        let mut frame_stable_ids = Vec::with_capacity(self.payload.frames.len());
        let mut frames_by_stable_id = BTreeMap::new();
        for (index, frame) in self.payload.frames.iter().enumerate() {
            if frame.parent.is_some_and(|parent| parent >= index) {
                return Err(HyperscapeGltfError::Validation(format!(
                    "frame {index} parent must precede its child"
                )));
            }
            let chain = ConformalTransformChain::new(frame.generators.clone())
                .map_err(|error| validation(format!("frame {index}: {error}")))?;
            let id = frames
                .add_frame(frame.name.clone(), frame.parent.map(FrameId), chain)
                .map_err(|error| validation(format!("frame {index}: {error}")))?;
            debug_assert_eq!(id, FrameId(index));
            if let Some(stable_id) = frame.stable_id {
                stable_id
                    .validate()
                    .map_err(|error| validation(format!("frame {index}: {error}")))?;
                if frames_by_stable_id.insert(stable_id, id).is_some() {
                    return Err(validation(format!(
                        "frame {index} repeats stable conformal frame ID {stable_id}"
                    )));
                }
            }
            frame_stable_ids.push(frame.stable_id);
        }
        frames
            .validate()
            .map_err(|error| validation(format!("frame forest: {error}")))?;

        let mut walls = RoundWallSet::new();
        for (index, wall) in self.payload.walls.iter().enumerate() {
            let id = walls
                .add_wall(
                    &frames,
                    RoundWall {
                        name: wall.name.clone(),
                        frame: FrameId(wall.frame),
                        geometry: wall.geometry,
                    },
                )
                .map_err(|error| validation(format!("wall {index}: {error}")))?;
            debug_assert_eq!(id, WallId(index));
        }

        let mut anchors = Vec::with_capacity(self.payload.anchors.len());
        for (index, authored) in self.payload.anchors.iter().enumerate() {
            frames.frame(FrameId(authored.frame)).map_err(|error| {
                validation(format!("anchor {index} has invalid frame: {error}"))
            })?;
            let mut anchor = AnchorState::new(FrameId(authored.frame));
            for &wall in &authored.flipped_walls {
                walls.wall(WallId(wall)).map_err(|error| {
                    validation(format!("anchor {index} has invalid wall: {error}"))
                })?;
                if anchor.flipped_walls().contains(&WallId(wall)) {
                    return Err(validation(format!(
                        "anchor {index} repeats flipped wall {wall}"
                    )));
                }
                anchor.flip(WallId(wall));
            }
            anchors.push(anchor);
        }

        for (node, binding) in self.node_bindings.iter().enumerate() {
            let Some(binding) = binding else { continue };
            frames
                .frame(FrameId(binding.frame))
                .map_err(|error| validation(format!("node {node} has invalid frame: {error}")))?;
            if binding.anchor.is_some_and(|anchor| anchor >= anchors.len()) {
                return Err(validation(format!("node {node} has invalid anchor index")));
            }
            if let Some(anchor) = binding.anchor {
                if self.payload.anchors[anchor].frame != binding.frame {
                    return Err(validation(format!(
                        "node {node} anchor frame does not match its entity frame"
                    )));
                }
            }
            if binding
                .path
                .is_some_and(|path| path >= self.payload.paths.len())
            {
                return Err(validation(format!("node {node} has invalid path index")));
            }
        }

        for (index, path) in self.payload.paths.iter().enumerate() {
            require_node(path.node, node_count, &format!("path {index}"))?;
            validate_keyframes(index, &path.keyframes)?;
            let binding = self.node_bindings[path.node].as_ref().ok_or_else(|| {
                validation(format!("path {index} node has no Hyperscape binding"))
            })?;
            if binding.path != Some(index) {
                return Err(validation(format!(
                    "path {index} must be referenced by its authored node binding"
                )));
            }
            if self
                .node_bindings
                .iter()
                .enumerate()
                .any(|(node, candidate)| {
                    node != path.node && candidate.as_ref().and_then(|b| b.path) == Some(index)
                })
            {
                return Err(validation(format!(
                    "path {index} is referenced by more than its authored node"
                )));
            }
            if let Some(frame) = path.coordinate_frame {
                frames.frame(FrameId(frame)).map_err(|error| {
                    validation(format!(
                        "path {index} has invalid coordinate frame: {error}"
                    ))
                })?;
            }
            let mut previous_transition = None;
            let last_time = path
                .keyframes
                .last()
                .expect("keyframes validated nonempty")
                .time_seconds;
            for (transition_index, transition) in path.transitions.iter().enumerate() {
                if !transition.time_seconds.is_finite()
                    || transition.time_seconds < 0.0
                    || transition.time_seconds > last_time
                    || previous_transition.is_some_and(|time| time >= transition.time_seconds)
                {
                    return Err(validation(format!(
                        "path {index} transition {transition_index} needs a finite, in-range, strictly increasing time"
                    )));
                }
                frames.frame(FrameId(transition.frame)).map_err(|error| {
                    validation(format!(
                        "path {index} transition {transition_index} has invalid frame: {error}"
                    ))
                })?;
                if let Some(anchor) = transition.anchor {
                    let authored = self.payload.anchors.get(anchor).ok_or_else(|| {
                        validation(format!(
                            "path {index} transition {transition_index} has invalid anchor"
                        ))
                    })?;
                    if authored.frame != transition.frame {
                        return Err(validation(format!(
                            "path {index} transition {transition_index} anchor frame does not match"
                        )));
                    }
                }
                previous_transition = Some(transition.time_seconds);
            }
        }
        let mut pinned_frames = BTreeSet::new();
        for (index, constraint) in self.payload.constraints.iter().enumerate() {
            match constraint {
                HyperscapeConstraint::Track {
                    node,
                    target_node,
                    local_offset,
                } => {
                    require_node(*node, node_count, &format!("constraint {index}"))?;
                    require_node(*target_node, node_count, &format!("constraint {index}"))?;
                    if local_offset.iter().any(|x| !x.is_finite()) {
                        return Err(validation(format!(
                            "constraint {index} offset must be finite"
                        )));
                    }
                }
                HyperscapeConstraint::ProjectionCamera { node, frame } => {
                    require_node(*node, node_count, &format!("constraint {index}"))?;
                    frames.frame(FrameId(*frame)).map_err(|error| {
                        validation(format!("constraint {index} has invalid frame: {error}"))
                    })?;
                }
                HyperscapeConstraint::SurfacePin {
                    frame,
                    target_entity,
                    barycentric,
                    normal_sign,
                    heading_radians,
                    uniform_scale,
                    local_offset,
                    ..
                } => {
                    let pinned_frame = frames.frame(FrameId(*frame)).map_err(|error| {
                        validation(format!("constraint {index} has invalid frame: {error}"))
                    })?;
                    if self.payload.frames[*frame].stable_id.is_none() {
                        return Err(validation(format!(
                            "constraint {index} pinned frame needs a stable ID"
                        )));
                    }
                    if !pinned_frames.insert(*frame) {
                        return Err(validation(format!(
                            "constraint {index} repeats a surface pin for frame {frame}"
                        )));
                    }
                    target_entity
                        .validate()
                        .map_err(|error| validation(format!("constraint {index}: {error}")))?;
                    let target_node = nodes_by_stable_id.get(target_entity).ok_or_else(|| {
                        validation(format!(
                            "constraint {index} targets unknown stable entity {target_entity}"
                        ))
                    })?;
                    let target_binding = self.node_bindings[*target_node]
                        .as_ref()
                        .expect("stable IDs exist only on bindings");
                    if pinned_frame.parent != Some(FrameId(target_binding.frame)) {
                        return Err(validation(format!(
                            "constraint {index} pinned frame parent must be target entity frame {}",
                            target_binding.frame
                        )));
                    }
                    validate_barycentric(*barycentric).map_err(|message| {
                        validation(format!("constraint {index} barycentric address {message}"))
                    })?;
                    if !matches!(normal_sign, -1 | 1) {
                        return Err(validation(format!(
                            "constraint {index} normal sign must be +1 or -1"
                        )));
                    }
                    if !heading_radians.is_finite() {
                        return Err(validation(format!(
                            "constraint {index} heading must be finite"
                        )));
                    }
                    if !uniform_scale.is_finite() || *uniform_scale <= 0.0 {
                        return Err(validation(format!(
                            "constraint {index} scale must be finite and positive"
                        )));
                    }
                    ConformalTransformChain::new(local_offset.clone()).map_err(|error| {
                        validation(format!("constraint {index} local offset: {error}"))
                    })?;
                }
            }
        }

        Ok(RuntimeHyperscapeScene {
            frames,
            frame_stable_ids,
            frames_by_stable_id,
            walls,
            anchors,
        })
    }
}

impl RuntimeHyperscapeScene {
    /// Resolve durable authored identity to the compact handle used by the
    /// conformal evaluator.
    pub fn frame_id(&self, stable_id: ConformalFrameId) -> Option<FrameId> {
        self.frames_by_stable_id.get(&stable_id).copied()
    }

    /// Resolve a compact evaluator handle back to durable authored identity.
    /// Legacy interchange may legitimately return `None`.
    pub fn stable_frame_id(&self, frame: FrameId) -> Option<ConformalFrameId> {
        self.frame_stable_ids.get(frame.0).copied().flatten()
    }
}

fn require_node(node: usize, node_count: usize, context: &str) -> Result<(), HyperscapeGltfError> {
    if node < node_count {
        Ok(())
    } else {
        Err(validation(format!(
            "{context} references node {node}, but there are {node_count} nodes"
        )))
    }
}

fn validate_barycentric(barycentric: [f64; 3]) -> Result<(), &'static str> {
    const EPSILON: f64 = 1.0e-12;
    if barycentric.into_iter().any(|value| !value.is_finite()) {
        return Err("must be finite");
    }
    if barycentric.into_iter().any(|value| value < -EPSILON)
        || barycentric.into_iter().sum::<f64>() <= EPSILON
    {
        return Err("must lie on the source face");
    }
    Ok(())
}

fn validate_keyframes(
    path: usize,
    keys: &[HyperscapePathKeyframe],
) -> Result<(), HyperscapeGltfError> {
    if keys.is_empty() {
        return Err(validation(format!("path {path} has no keyframes")));
    }
    for key in keys {
        if !key.time_seconds.is_finite()
            || key.time_seconds < 0.0
            || key.point.into_iter().any(|x| !x.is_finite())
        {
            return Err(validation(format!(
                "path {path} keyframes must be finite with nonnegative times"
            )));
        }
    }
    if keys
        .windows(2)
        .any(|pair| pair[0].time_seconds >= pair[1].time_seconds)
    {
        return Err(validation(format!(
            "path {path} keyframe times must be strictly increasing"
        )));
    }
    Ok(())
}

fn validation(message: String) -> HyperscapeGltfError {
    HyperscapeGltfError::Validation(message)
}

fn parse_extras<T: DeserializeOwned>(
    extras: &gltf::json::Extras,
    context: &str,
) -> Result<Option<T>, HyperscapeGltfError> {
    let Some(raw) = extras.as_ref() else {
        return Ok(None);
    };
    let extras: Value = serde_json::from_str(raw.get())
        .map_err(|error| HyperscapeGltfError::InvalidJson(format!("{context}: {error}")))?;
    let Some(hyperscape) = extras
        .as_object()
        .and_then(|object| object.get("hyperscape"))
    else {
        return Ok(None);
    };
    serde_json::from_value(hyperscape.clone())
        .map(Some)
        .map_err(|error| HyperscapeGltfError::InvalidJson(format!("{context}: {error}")))
}

/// Extract and fully validate root and node extras from a parsed glTF document.
pub fn extract_asset(
    document: &gltf::Document,
) -> Result<Option<HyperscapeAsset>, HyperscapeGltfError> {
    let root = parse_extras::<HyperscapePayload>(&document.as_json().extras, "root extras")?;
    let node_bindings = document
        .nodes()
        .map(|node| {
            parse_extras::<HyperscapeNodeBinding>(
                node.extras(),
                &format!("node {} extras", node.index()),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    match root {
        Some(payload) => {
            let asset = HyperscapeAsset {
                payload,
                node_bindings,
            };
            asset.validate()?;
            Ok(Some(asset))
        }
        None if node_bindings.iter().any(Option::is_some) => Err(validation(
            "node Hyperscape bindings require root extras.hyperscape".into(),
        )),
        None => Ok(None),
    }
}

/// Inject a validated payload into a parsed glTF root without disturbing
/// ordinary nodes or unrelated extras fields.
pub fn inject_into_json(
    root: &mut Value,
    asset: &HyperscapeAsset,
) -> Result<(), HyperscapeGltfError> {
    asset.validate()?;
    let object = root
        .as_object_mut()
        .ok_or_else(|| validation("glTF root must be a JSON object".into()))?;
    let node_count = object
        .get("nodes")
        .and_then(Value::as_array)
        .map(Vec::len)
        .ok_or_else(|| validation("glTF root needs a nodes array".into()))?;
    if node_count != asset.node_bindings.len() {
        return Err(validation(format!(
            "binding count {} does not match node count {}",
            asset.node_bindings.len(),
            node_count
        )));
    }

    insert_extras_value(
        object,
        serde_json::to_value(&asset.payload)
            .map_err(|error| HyperscapeGltfError::InvalidJson(error.to_string()))?,
    )?;
    let nodes = object
        .get_mut("nodes")
        .and_then(Value::as_array_mut)
        .expect("nodes array checked above");
    for (index, binding) in asset.node_bindings.iter().enumerate() {
        let node = nodes[index]
            .as_object_mut()
            .ok_or_else(|| validation(format!("node {index} must be an object")))?;
        let Some(binding) = binding else {
            if let Some(extras) = node.get_mut("extras").and_then(Value::as_object_mut) {
                extras.remove("hyperscape");
            }
            continue;
        };
        insert_extras_value(
            node,
            serde_json::to_value(binding)
                .map_err(|error| HyperscapeGltfError::InvalidJson(error.to_string()))?,
        )?;
    }
    Ok(())
}

fn insert_extras_value(
    owner: &mut Map<String, Value>,
    hyperscape: Value,
) -> Result<(), HyperscapeGltfError> {
    let extras = owner
        .entry("extras")
        .or_insert_with(|| Value::Object(Map::new()));
    let extras = extras.as_object_mut().ok_or_else(|| {
        validation("cannot add Hyperscape data to a non-object extras value".into())
    })?;
    extras.insert("hyperscape".into(), hyperscape);
    Ok(())
}

/// Inject Hyperscape extras into either JSON `.gltf` bytes or a GLB while
/// preserving all non-JSON GLB chunks byte-for-byte.
pub fn inject_into_gltf_bytes(
    bytes: &[u8],
    asset: &HyperscapeAsset,
) -> Result<Vec<u8>, HyperscapeGltfError> {
    if bytes.starts_with(b"glTF") {
        inject_into_glb(bytes, asset)
    } else {
        let mut root: Value = serde_json::from_slice(bytes)
            .map_err(|error| HyperscapeGltfError::InvalidJson(error.to_string()))?;
        inject_into_json(&mut root, asset)?;
        serde_json::to_vec_pretty(&root)
            .map_err(|error| HyperscapeGltfError::InvalidJson(error.to_string()))
    }
}

fn inject_into_glb(bytes: &[u8], asset: &HyperscapeAsset) -> Result<Vec<u8>, HyperscapeGltfError> {
    if bytes.len() < 20 {
        return Err(HyperscapeGltfError::InvalidGlb(
            "header is truncated".into(),
        ));
    }
    let version = read_u32(bytes, 4)?;
    let declared_length = read_u32(bytes, 8)? as usize;
    if version != 2 || declared_length != bytes.len() {
        return Err(HyperscapeGltfError::InvalidGlb(
            "expected a complete GLB version 2 buffer".into(),
        ));
    }

    let mut chunks = Vec::<(u32, Vec<u8>)>::new();
    let mut cursor = 12;
    while cursor < bytes.len() {
        if cursor + 8 > bytes.len() {
            return Err(HyperscapeGltfError::InvalidGlb(
                "chunk header is truncated".into(),
            ));
        }
        let length = read_u32(bytes, cursor)? as usize;
        let kind = read_u32(bytes, cursor + 4)?;
        cursor += 8;
        let end = cursor
            .checked_add(length)
            .filter(|end| *end <= bytes.len())
            .ok_or_else(|| HyperscapeGltfError::InvalidGlb("chunk is truncated".into()))?;
        chunks.push((kind, bytes[cursor..end].to_vec()));
        cursor = end;
    }
    let Some((_, json_bytes)) = chunks
        .first_mut()
        .filter(|(kind, _)| *kind == GLB_JSON_CHUNK)
    else {
        return Err(HyperscapeGltfError::InvalidGlb(
            "first GLB chunk must be JSON".into(),
        ));
    };
    let mut root: Value = serde_json::from_slice(json_bytes)
        .map_err(|error| HyperscapeGltfError::InvalidJson(error.to_string()))?;
    inject_into_json(&mut root, asset)?;
    *json_bytes = serde_json::to_vec(&root)
        .map_err(|error| HyperscapeGltfError::InvalidJson(error.to_string()))?;
    while json_bytes.len() % 4 != 0 {
        json_bytes.push(b' ');
    }

    let total_length = 12usize
        + chunks
            .iter()
            .map(|(_, chunk)| 8usize + chunk.len())
            .sum::<usize>();
    let total_u32 = u32::try_from(total_length)
        .map_err(|_| HyperscapeGltfError::InvalidGlb("output is too large".into()))?;
    let mut output = Vec::with_capacity(total_length);
    output.extend_from_slice(b"glTF");
    output.extend_from_slice(&2u32.to_le_bytes());
    output.extend_from_slice(&total_u32.to_le_bytes());
    for (kind, chunk) in chunks {
        let length = u32::try_from(chunk.len())
            .map_err(|_| HyperscapeGltfError::InvalidGlb("chunk is too large".into()))?;
        output.extend_from_slice(&length.to_le_bytes());
        output.extend_from_slice(&kind.to_le_bytes());
        output.extend_from_slice(&chunk);
    }
    Ok(output)
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, HyperscapeGltfError> {
    let raw = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| HyperscapeGltfError::InvalidGlb("integer is truncated".into()))?;
    Ok(u32::from_le_bytes(raw.try_into().expect("four-byte slice")))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HyperscapeGltfError {
    InvalidJson(String),
    InvalidGlb(String),
    Validation(String),
}

impl fmt::Display for HyperscapeGltfError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidJson(message) => write!(formatter, "invalid JSON: {message}"),
            Self::InvalidGlb(message) => write!(formatter, "invalid GLB: {message}"),
            Self::Validation(message) => write!(formatter, "invalid payload: {message}"),
        }
    }
}

impl Error for HyperscapeGltfError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_asset() -> HyperscapeAsset {
        HyperscapeAsset {
            payload: HyperscapePayload {
                version: HYPERSCAPE_INTERCHANGE_VERSION.into(),
                asset_id: Some(AssetId::from_u128(3).unwrap()),
                frames: vec![
                    HyperscapeFrame {
                        stable_id: Some(ConformalFrameId::from_u128(10).unwrap()),
                        name: "world".into(),
                        parent: None,
                        generators: Vec::new(),
                    },
                    HyperscapeFrame {
                        stable_id: Some(ConformalFrameId::from_u128(11).unwrap()),
                        name: "reflection room".into(),
                        parent: Some(0),
                        generators: vec![ConformalGenerator::SphereReflection {
                            center: [0.0; 3],
                            radius: 3.0,
                        }],
                    },
                ],
                walls: vec![HyperscapeWall {
                    name: "room wall".into(),
                    frame: 1,
                    geometry: RoundWallGeometry::Sphere {
                        center: [0.0; 3],
                        radius: 3.0,
                    },
                }],
                anchors: vec![HyperscapeAnchor {
                    name: "inside out".into(),
                    frame: 1,
                    flipped_walls: vec![0],
                }],
                paths: vec![HyperscapePath {
                    name: "approach".into(),
                    node: 0,
                    coordinate_frame: None,
                    looping: false,
                    keyframes: vec![
                        HyperscapePathKeyframe {
                            time_seconds: 0.0,
                            point: [0.0; 3],
                        },
                        HyperscapePathKeyframe {
                            time_seconds: 2.0,
                            point: [2.0, 0.0, 0.0],
                        },
                    ],
                    transitions: Vec::new(),
                }],
                constraints: vec![HyperscapeConstraint::Track {
                    node: 1,
                    target_node: 0,
                    local_offset: [0.0, 1.0, 0.0],
                }],
            },
            node_bindings: vec![
                Some(HyperscapeNodeBinding {
                    stable_id: Some(EntityId::from_u128(1).unwrap()),
                    frame: 1,
                    anchor: Some(0),
                    path: Some(0),
                }),
                Some(HyperscapeNodeBinding {
                    stable_id: Some(EntityId::from_u128(2).unwrap()),
                    frame: 0,
                    anchor: None,
                    path: None,
                }),
            ],
        }
    }

    fn minimal_json() -> Value {
        serde_json::json!({
            "asset": { "version": "2.0" },
            "nodes": [
                { "name": "horse", "extras": { "author": "kept" } },
                { "name": "camera" }
            ],
            "scenes": [{ "nodes": [0, 1] }],
            "scene": 0,
            "extras": { "application": "also kept" }
        })
    }

    fn make_glb(json: &Value, bin: &[u8]) -> Vec<u8> {
        let mut json = serde_json::to_vec(json).unwrap();
        while json.len() % 4 != 0 {
            json.push(b' ');
        }
        let mut bin = bin.to_vec();
        while bin.len() % 4 != 0 {
            bin.push(0);
        }
        let total = 12 + 8 + json.len() + 8 + bin.len();
        let mut output = Vec::new();
        output.extend_from_slice(b"glTF");
        output.extend_from_slice(&2u32.to_le_bytes());
        output.extend_from_slice(&(total as u32).to_le_bytes());
        output.extend_from_slice(&(json.len() as u32).to_le_bytes());
        output.extend_from_slice(&GLB_JSON_CHUNK.to_le_bytes());
        output.extend_from_slice(&json);
        output.extend_from_slice(&(bin.len() as u32).to_le_bytes());
        output.extend_from_slice(&0x004e_4942u32.to_le_bytes());
        output.extend_from_slice(&bin);
        output
    }

    #[test]
    fn json_injection_preserves_fallback_and_unrelated_extras() {
        let asset = sample_asset();
        let encoded =
            inject_into_gltf_bytes(&serde_json::to_vec(&minimal_json()).unwrap(), &asset).unwrap();
        let value: Value = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(value["nodes"][0]["name"], "horse");
        assert_eq!(value["nodes"][0]["extras"]["author"], "kept");
        assert_eq!(
            value["nodes"][0]["extras"]["hyperscape"]["stable_id"],
            EntityId::from_u128(1).unwrap().to_string()
        );
        assert_eq!(value["extras"]["application"], "also kept");
        let document = gltf::Gltf::from_slice(&encoded).unwrap().document;
        let recovered = extract_asset(&document).unwrap().unwrap();
        assert_eq!(recovered, asset);
        let runtime = recovered.validate().unwrap();
        assert_eq!(runtime.frames.frames().len(), 2);
        assert_eq!(
            runtime.frame_id(ConformalFrameId::from_u128(11).unwrap()),
            Some(FrameId(1))
        );
        assert_eq!(
            runtime.stable_frame_id(FrameId(0)),
            Some(ConformalFrameId::from_u128(10).unwrap())
        );
        assert_eq!(runtime.stable_frame_id(FrameId(99)), None);
        assert_eq!(runtime.walls.walls().len(), 1);
        assert_eq!(
            runtime.anchors[0].flipped_walls(),
            &std::collections::BTreeSet::from([WallId(0)])
        );
    }

    #[test]
    fn json_injection_removes_stale_bindings_from_unbound_nodes() {
        let mut asset = sample_asset();
        asset.node_bindings[1] = None;
        let mut source = minimal_json();
        source["nodes"][1]["extras"] = serde_json::json!({
            "hyperscape": { "frame": 99 },
            "ordinary": "kept"
        });

        inject_into_json(&mut source, &asset).unwrap();

        assert!(source["nodes"][1]["extras"].get("hyperscape").is_none());
        assert_eq!(source["nodes"][1]["extras"]["ordinary"], "kept");
    }

    #[test]
    fn glb_injection_preserves_binary_chunk() {
        let asset = sample_asset();
        let original_bin = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let glb = make_glb(&minimal_json(), &original_bin);
        let rewritten = inject_into_gltf_bytes(&glb, &asset).unwrap();
        let parsed = gltf::Gltf::from_slice(&rewritten).unwrap();
        assert_eq!(parsed.blob.as_deref(), Some(original_bin.as_slice()));
        assert_eq!(extract_asset(&parsed.document).unwrap().unwrap(), asset);
    }

    #[test]
    fn invalid_references_and_non_object_extras_are_rejected() {
        let mut asset = sample_asset();
        asset.payload.walls[0].frame = 99;
        assert!(asset.validate().is_err());

        let mut asset = sample_asset();
        let mut root = minimal_json();
        root["extras"] = Value::String("opaque".into());
        assert!(inject_into_json(&mut root, &asset).is_err());
        asset.node_bindings.pop();
        assert!(inject_into_json(&mut minimal_json(), &asset).is_err());
    }

    #[test]
    fn stable_node_ids_must_be_non_nil_and_unique() {
        let mut duplicate = sample_asset();
        duplicate.node_bindings[1].as_mut().unwrap().stable_id =
            Some(EntityId::from_u128(1).unwrap());
        assert!(duplicate.validate().is_err());

        let mut nil = sample_asset();
        nil.node_bindings[0].as_mut().unwrap().stable_id =
            Some(serde_json::from_value(Value::String(Uuid::nil().to_string())).unwrap());
        assert!(nil.validate().is_err());
    }

    #[test]
    fn asset_identity_is_optional_stable_and_non_nil() {
        let asset = sample_asset();
        let encoded = serde_json::to_value(&asset.payload).unwrap();
        assert_eq!(
            encoded["asset_id"],
            AssetId::from_u128(3).unwrap().to_string()
        );

        let mut legacy = asset.clone();
        legacy.payload.asset_id = None;
        legacy.validate().unwrap();
        assert!(serde_json::to_value(&legacy.payload)
            .unwrap()
            .get("asset_id")
            .is_none());

        let mut nil = asset;
        nil.payload.asset_id =
            Some(serde_json::from_value(Value::String(Uuid::nil().to_string())).unwrap());
        assert!(nil.validate().is_err());
    }

    #[test]
    fn stable_conformal_frame_ids_must_be_non_nil_and_unique() {
        let mut duplicate = sample_asset();
        duplicate.payload.frames[1].stable_id = duplicate.payload.frames[0].stable_id;
        assert!(duplicate.validate().is_err());

        let mut nil = sample_asset();
        nil.payload.frames[0].stable_id =
            Some(serde_json::from_value(Value::String(Uuid::nil().to_string())).unwrap());
        assert!(nil.validate().is_err());
    }

    #[test]
    fn legacy_frames_without_stable_ids_remain_valid() {
        let mut asset = sample_asset();
        for frame in &mut asset.payload.frames {
            frame.stable_id = None;
        }
        let runtime = asset.validate().unwrap();
        assert_eq!(runtime.frame_stable_ids, vec![None, None]);
        assert!(runtime.frames_by_stable_id.is_empty());
    }

    fn surface_pin_constraint() -> HyperscapeConstraint {
        HyperscapeConstraint::SurfacePin {
            frame: 1,
            target_entity: EntityId::from_u128(2).unwrap(),
            face: 17,
            barycentric: [0.5, 0.25, 0.25],
            normal_sign: -1,
            heading_radians: 0.4,
            uniform_scale: 1.5,
            orientation: SurfaceFrameOrientation::RightSideIn,
            local_offset: vec![ConformalGenerator::sphere_reflection([0.0; 3], 2.0)],
        }
    }

    #[test]
    fn surface_pins_use_stable_targets_and_a_single_spatial_parent() {
        let mut asset = sample_asset();
        asset.payload.constraints.push(surface_pin_constraint());
        asset.validate().unwrap();
        let encoded = serde_json::to_value(&asset.payload).unwrap();
        assert_eq!(
            encoded["constraints"][1]["target_entity"],
            EntityId::from_u128(2).unwrap().to_string()
        );
        assert_eq!(encoded["constraints"][1]["orientation"], "right_side_in");

        let mut duplicate = asset.clone();
        duplicate.payload.constraints.push(surface_pin_constraint());
        assert!(duplicate.validate().is_err());

        let mut unknown_target = asset.clone();
        let HyperscapeConstraint::SurfacePin { target_entity, .. } =
            &mut unknown_target.payload.constraints[1]
        else {
            unreachable!()
        };
        *target_entity = EntityId::from_u128(99).unwrap();
        assert!(unknown_target.validate().is_err());

        let mut wrong_parent = asset.clone();
        let HyperscapeConstraint::SurfacePin { target_entity, .. } =
            &mut wrong_parent.payload.constraints[1]
        else {
            unreachable!()
        };
        *target_entity = EntityId::from_u128(1).unwrap();
        assert!(wrong_parent.validate().is_err());

        let mut invalid_address = asset;
        let HyperscapeConstraint::SurfacePin { barycentric, .. } =
            &mut invalid_address.payload.constraints[1]
        else {
            unreachable!()
        };
        *barycentric = [1.0, -0.5, 0.5];
        assert!(invalid_address.validate().is_err());
    }

    #[test]
    fn path_transitions_validate_frames_anchors_and_time_order() {
        let mut asset = sample_asset();
        asset.payload.paths[0].coordinate_frame = Some(0);
        asset.payload.paths[0].transitions = vec![HyperscapePathTransition {
            time_seconds: 1.0,
            frame: 1,
            anchor: Some(0),
        }];
        asset.validate().unwrap();

        let mut mismatch = asset.clone();
        mismatch.payload.paths[0].transitions[0].frame = 0;
        assert!(mismatch.validate().is_err());

        let mut unordered = asset;
        unordered.payload.paths[0]
            .transitions
            .push(HyperscapePathTransition {
                time_seconds: 0.5,
                frame: 0,
                anchor: None,
            });
        assert!(unordered.validate().is_err());
    }

    #[test]
    fn rootless_node_binding_is_not_silently_accepted() {
        let json = serde_json::json!({
            "asset": { "version": "2.0" },
            "nodes": [{ "extras": { "hyperscape": { "frame": 0 } } }]
        });
        let document = gltf::Gltf::from_slice(&serde_json::to_vec(&json).unwrap())
            .unwrap()
            .document;
        assert!(extract_asset(&document).is_err());
    }

    #[test]
    fn unrelated_non_object_extras_remain_valid_fallback_data() {
        let json = serde_json::json!({
            "asset": { "version": "2.0" },
            "nodes": [{ "extras": "opaque application value" }],
            "extras": 42
        });
        let document = gltf::Gltf::from_slice(&serde_json::to_vec(&json).unwrap())
            .unwrap()
            .document;
        assert_eq!(extract_asset(&document).unwrap(), None);
    }

    #[test]
    fn checked_in_example_is_valid_and_keeps_ordinary_nodes() {
        let bytes = crate::HYPERSCAPE_TRACK_GLTF;
        let parsed = gltf::Gltf::from_slice(bytes).unwrap();
        assert_eq!(parsed.document.nodes().count(), 3);
        let asset = extract_asset(&parsed.document).unwrap().unwrap();
        let runtime = asset.validate().unwrap();
        assert_eq!(
            asset.payload.asset_id,
            Some(AssetId::from_u128(0xaaaa_aaaa_aaaa_4aaa_8aaa_aaaa_aaaa_aaaa).unwrap())
        );
        assert_eq!(runtime.frames.frames().len(), 2);
        assert_eq!(runtime.walls.walls().len(), 2);
        assert_eq!(asset.payload.paths.len(), 1);
        assert_eq!(asset.payload.constraints.len(), 2);

        let (nodes, graph_asset) = crate::load_hyperscape_graph(bytes).unwrap();
        assert_eq!(nodes.len(), 3);
        assert_eq!(graph_asset.unwrap(), asset);
    }
}
