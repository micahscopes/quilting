//! GPU compute via transform feedback (WebGL2 GPGPU).
//!
//! Two-pass pipeline — one "vertex" per face:
//!   Pass 1: animated positions → conservative image bound + raw LOD exponents
//!   Pass 2: edge coherence (via adjacency texture) + canonical sort + atlas LUT
//!
//! Pass 1 renders directly to a texture for pass 2 to read.
//! Pass 2 emits one lossless packed `u32` per face. Rust authority admits the
//! typed fields directly; shadow/worker parity can still expand them to the
//! historical six-field classification
//! (canon_a, canon_b, canon_c, perm_index, parity, atlas_index).

use glow::HasContext;
use quilting_core::{batch, instance_layout};
use quilting_core::batch::RenderBatchLayer;
use quilting_core::quaternion::{Mobius, Quat};
use quilting_core::render::{
    RenderGeometry, RenderSceneSnapshot, VisibilityCompactionPlan,
};
use quilting_mesh::HalfEdgeMesh;
use std::collections::HashMap;

#[cfg(target_arch = "wasm32")]
fn log_info(msg: &str) {
    web_sys::console::info_1(&msg.into());
}
#[cfg(not(target_arch = "wasm32"))]
fn log_info(msg: &str) {
    eprintln!("{}", msg);
}

/// Pass 1 FBO payload: raw LOD exponents plus visibility/adaptation priority.
pub const FLOATS_PER_FACE_PASS1: usize = 4;

/// Retained CPU output: (canon_a, canon_b, canon_c, perm_index, parity, atlas_index).
pub const FLOATS_PER_FACE_OUTPUT: usize = 6;

/// Lossless GPU readback ABI for one classifier record.
///
/// The three four-bit exponent fields, three-bit S3 permutation, one-bit
/// visibility flag, eight-bit atlas index, and eight-bit adaptive priority fit
/// in one word. Parity is derived exactly from the permutation. The priority
/// is selection metadata, not topology, and is omitted from the historical
/// six-float batch projection.
pub const PACKED_LOD_OUTPUT_BYTES_PER_FACE: usize = std::mem::size_of::<u32>();

const PACKED_LOD_EXPONENT_MASK: u32 = 0x0f;
const PACKED_LOD_PERMUTATION_MASK: u32 = 0x07;
const PACKED_LOD_VISIBLE_BIT: u32 = 1 << 15;
const PACKED_LOD_ATLAS_SHIFT: u32 = 16;
const PACKED_LOD_TOPOLOGY_MASK: u32 = 0x00ff_ffff;

/// Validated semantic fields carried by one packed classifier word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackedLodClassification {
    pub canonical: [u32; 3],
    pub permutation: u8,
    pub parity_bucket: u8,
    pub atlas_index: Option<u8>,
    pub adaptation_priority: u8,
}

impl PackedLodClassification {
    pub const fn visible(self) -> bool {
        self.atlas_index.is_some()
    }

    pub fn into_face_lod_classification(self) -> batch::FaceLodClassification {
        batch::FaceLodClassification {
            requested: batch::ResidentLod {
                canonical: self.canonical,
                perm_index: self.permutation as usize,
                parity_bucket: self.parity_bucket as usize,
            },
            visible: self.visible(),
        }
    }
}

/// Encode one shader-facing classifier record into the shared 32-bit ABI.
/// Topology occupies the low 24 bits and adaptation priority the high byte.
/// Exponents are stored rather than their exact power-of-two `f32` values.
pub fn pack_lod_classification(
    exponents: [u32; 3],
    permutation: u32,
    atlas_index: Option<u32>,
    adaptation_priority: u8,
) -> Result<u32, String> {
    if exponents.iter().any(|&exponent| exponent > 9) {
        return Err("packed LOD exponent exceeds the atlas ABI".to_string());
    }
    if permutation > 5 {
        return Err("packed LOD permutation exceeds S3".to_string());
    }
    if atlas_index.is_some_and(|atlas_index| atlas_index > u8::MAX as u32) {
        return Err("packed LOD atlas index exceeds u8".to_string());
    }
    let mut packed = exponents[0]
        | (exponents[1] << 4)
        | (exponents[2] << 8)
        | (permutation << 12)
        | (u32::from(adaptation_priority) << 24);
    if let Some(atlas_index) = atlas_index {
        packed |= PACKED_LOD_VISIBLE_BIT | (atlas_index << PACKED_LOD_ATLAS_SHIFT);
    }
    Ok(packed)
}

/// Validate and decode one packed GPU record without expanding its ABI.
pub fn unpack_lod_classification_fields(
    packed: u32,
) -> Result<PackedLodClassification, String> {
    let exponents = [
        packed & PACKED_LOD_EXPONENT_MASK,
        (packed >> 4) & PACKED_LOD_EXPONENT_MASK,
        (packed >> 8) & PACKED_LOD_EXPONENT_MASK,
    ];
    if exponents.iter().any(|&exponent| exponent > 9) {
        return Err("packed LOD record contains an invalid exponent".to_string());
    }
    let permutation = (packed >> 12) & PACKED_LOD_PERMUTATION_MASK;
    if permutation > 5 {
        return Err("packed LOD record contains an invalid permutation".to_string());
    }
    let atlas_index = if packed & PACKED_LOD_VISIBLE_BIT == 0 {
        None
    } else {
        Some(((packed >> PACKED_LOD_ATLAS_SHIFT) & u8::MAX as u32) as u8)
    };
    Ok(PackedLodClassification {
        canonical: [
            1u32 << exponents[0],
            1u32 << exponents[1],
            1u32 << exponents[2],
        ],
        permutation: permutation as u8,
        parity_bucket: u8::from(matches!(permutation, 1 | 2 | 5)),
        atlas_index,
        adaptation_priority: (packed >> 24) as u8,
    })
}

/// Decode one packed GPU record to the historical six-float shadow ABI.
pub fn unpack_lod_classification(
    packed: u32,
) -> Result<[f32; FLOATS_PER_FACE_OUTPUT], String> {
    let fields = unpack_lod_classification_fields(packed)?;
    Ok([
        fields.canonical[0] as f32,
        fields.canonical[1] as f32,
        fields.canonical[2] as f32,
        fields.permutation as f32,
        if fields.parity_bucket == 0 { 1.0 } else { -1.0 },
        fields.atlas_index.map_or(-1.0, f32::from),
    ])
}

fn validate_packed_lod_classifications(packed: &[u32]) -> Result<(), String> {
    for &record in packed {
        unpack_lod_classification_fields(record)?;
    }
    Ok(())
}

/// Shape and workload summary for one retained packed-classifier comparison.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PackedLodDeltaSummary {
    pub full_snapshot: bool,
    pub changed_records: usize,
}

impl PackedLodDeltaSummary {
    pub const fn is_unchanged(self) -> bool {
        !self.full_snapshot && self.changed_records == 0
    }
}

/// Compare a complete packed publication with the previous publication and
/// retain only changed words and their monotonically increasing face IDs.
///
/// A shape change is an explicit full snapshot; in that case the sparse output
/// remains empty and the caller publishes `current` directly. Scratch vectors
/// are always cleared and retain their allocations between calls.
pub fn diff_packed_lod_classifications(
    current: &[u32],
    previous: &[u32],
    changed_faces: &mut Vec<u32>,
    changed_words: &mut Vec<u32>,
) -> PackedLodDeltaSummary {
    changed_faces.clear();
    changed_words.clear();
    if current.len() != previous.len() {
        return PackedLodDeltaSummary {
            full_snapshot: true,
            changed_records: current.len(),
        };
    }
    debug_assert!(current.len() <= u32::MAX as usize);
    for (face, (&old, &new)) in previous.iter().zip(current).enumerate() {
        if old & PACKED_LOD_TOPOLOGY_MASK != new & PACKED_LOD_TOPOLOGY_MASK {
            changed_faces.push(face as u32);
            changed_words.push(new);
        }
    }
    PackedLodDeltaSummary {
        full_snapshot: false,
        changed_records: changed_faces.len(),
    }
}

fn decode_packed_lod_classifications(
    packed: &[u32],
    decoded: &mut Vec<f32>,
) -> Result<(), String> {
    decoded.clear();
    decoded.reserve(packed.len().saturating_mul(FLOATS_PER_FACE_OUTPUT));
    for &record in packed {
        decoded.extend_from_slice(&unpack_lod_classification(record)?);
    }
    Ok(())
}

/// Backend-neutral source payload consumed by one LOD classifier residency.
///
/// WebGL2 uploads these arrays into textures; a WebGPU backend can upload the
/// same validated data into storage buffers. The payload deliberately contains
/// no context handles, atlas ownership, or per-frame pose.
#[derive(Clone, Debug, PartialEq)]
pub struct LodModelData {
    pub positions: Vec<f32>,
    pub faces: Vec<[u32; 3]>,
    pub joint_indices: Vec<[u16; 4]>,
    pub joint_weights: Vec<[f32; 4]>,
    pub morph_deltas: Vec<f32>,
    pub num_morph_targets: usize,
    pub face_nodes: Vec<usize>,
}

/// Validated, backend-neutral model payload prepared for classifier residency.
///
/// WebGL2 consumes `face_indices` and `adjacency` directly. A future WebGPU
/// backend can instead upload the same source model and topology to storage
/// buffers without rebuilding worker-specific metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedLodModel {
    pub model: LodModelData,
    pub face_indices: Vec<f32>,
    pub adjacency: Vec<f32>,
    pub residency: LodModelResidency,
}

/// Exact, backend-neutral identity for one prepared classifier model.
///
/// This deliberately hashes the uploaded bit patterns rather than semantic
/// floating-point values. A worker WebGL2 context, renderer WebGL2 context,
/// and future WebGPU backend must receive the same immutable model before
/// per-frame output parity is meaningful.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PreparedLodModelFingerprint {
    pub positions: (usize, u64),
    pub faces: (usize, u64),
    pub joint_indices: (usize, u64),
    pub joint_weights: (usize, u64),
    pub morph_deltas: (usize, u64),
    pub face_nodes: (usize, u64),
    pub face_indices: (usize, u64),
    pub adjacency: (usize, u64),
    pub num_morph_targets: usize,
    pub mesh_radius_bits: u32,
}

impl PreparedLodModelFingerprint {
    /// Stable textual ABI suitable for diagnostics crossing a JS number
    /// boundary without truncating 64-bit hashes.
    pub fn stable_text(self) -> String {
        format!(
            "lod-model-v1:p{}:{:016x};f{}:{:016x};ji{}:{:016x};jw{}:{:016x};m{}:{:016x};n{}:{:016x};fi{}:{:016x};a{}:{:016x};mt{};r{:08x}",
            self.positions.0,
            self.positions.1,
            self.faces.0,
            self.faces.1,
            self.joint_indices.0,
            self.joint_indices.1,
            self.joint_weights.0,
            self.joint_weights.1,
            self.morph_deltas.0,
            self.morph_deltas.1,
            self.face_nodes.0,
            self.face_nodes.1,
            self.face_indices.0,
            self.face_indices.1,
            self.adjacency.0,
            self.adjacency.1,
            self.num_morph_targets,
            self.mesh_radius_bits,
        )
    }
}

fn exact_word_fingerprint(words: impl IntoIterator<Item = u64>) -> (usize, u64) {
    let mut len = 0usize;
    let mut hash = 0xcbf29ce484222325u64;
    for word in words {
        len = len.saturating_add(1);
        for byte in word.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    (len, hash)
}

/// Exact bit fingerprint for a classifier payload. This is intentionally the
/// same primitive used by prepared-model diagnostics, so a worker can stamp a
/// full GPU result before sparse encoding without copying it across contexts.
pub fn exact_f32_slice_fingerprint(values: &[f32]) -> (usize, u64) {
    exact_word_fingerprint(values.iter().map(|value| u64::from(value.to_bits())))
}

/// Fingerprint every immutable array and scalar that reaches classifier GPU
/// residency. Component hashes make any divergence immediately localizable.
pub fn prepared_lod_model_fingerprint(
    prepared: &PreparedLodModel,
) -> PreparedLodModelFingerprint {
    let model = &prepared.model;
    PreparedLodModelFingerprint {
        positions: exact_f32_slice_fingerprint(&model.positions),
        faces: exact_word_fingerprint(
            model.faces.iter().flatten().map(|&value| u64::from(value)),
        ),
        joint_indices: exact_word_fingerprint(
            model
                .joint_indices
                .iter()
                .flatten()
                .map(|&value| u64::from(value)),
        ),
        joint_weights: exact_word_fingerprint(
            model
                .joint_weights
                .iter()
                .flatten()
                .map(|value| u64::from(value.to_bits())),
        ),
        morph_deltas: exact_f32_slice_fingerprint(&model.morph_deltas),
        face_nodes: exact_word_fingerprint(
            model.face_nodes.iter().map(|&value| value as u64),
        ),
        face_indices: exact_f32_slice_fingerprint(&prepared.face_indices),
        adjacency: exact_f32_slice_fingerprint(&prepared.adjacency),
        num_morph_targets: model.num_morph_targets,
        mesh_radius_bits: prepared.residency.mesh_radius.to_bits(),
    }
}

/// Small retained identity for one uploaded classifier model.
#[derive(Clone, Debug, PartialEq)]
pub struct LodModelResidency {
    pub num_faces: usize,
    pub num_vertices: u32,
    pub node_first_faces: HashMap<usize, usize>,
    pub mesh_radius: f32,
}

/// Canonical exponent lookup shared by WebGL2 and future WebGPU classifiers.
#[derive(Clone, Debug, PartialEq)]
pub struct LodAtlasLookup {
    pub lut: Vec<u8>,
    pub keys: Vec<[u32; 3]>,
    pub max_lod: f32,
}

/// Validate canonical atlas keys and build the compact shader lookup.
pub fn prepare_lod_atlas_lookup(
    keys: impl IntoIterator<Item = [u32; 3]>,
) -> Result<LodAtlasLookup, String> {
    const LUT_SIZE: usize = 1_200;
    const MISSING: u8 = u8::MAX;
    let mut keys: Vec<[u32; 3]> = keys.into_iter().collect();
    if keys.is_empty() {
        return Err("LOD atlas contains no canonical patches".to_string());
    }
    keys.sort_unstable();
    if keys.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err("LOD atlas contains duplicate canonical patches".to_string());
    }
    if keys.len() > MISSING as usize {
        return Err("LOD atlas exceeds the u8 classifier lookup".to_string());
    }

    let mut lut = vec![MISSING; LUT_SIZE];
    let mut max_lod = 1u32;
    for (index, key) in keys.iter().copied().enumerate() {
        if key[0] > key[1] || key[1] > key[2] {
            return Err(format!("LOD atlas key {key:?} is not canonical"));
        }
        let mut exponents = [0usize; 3];
        for (edge, exponent) in key.into_iter().zip(&mut exponents) {
            if !edge.is_power_of_two() || edge.trailing_zeros() > 9 {
                return Err(format!("LOD atlas edge {edge} is outside the shader lookup"));
            }
            *exponent = edge.trailing_zeros() as usize;
            max_lod = max_lod.max(edge);
        }
        let lut_key = exponents[0] + exponents[1] * 10 + exponents[2] * 100;
        debug_assert!(lut_key < LUT_SIZE);
        lut[lut_key] = index as u8;
    }
    Ok(LodAtlasLookup {
        lut,
        keys,
        max_lod: max_lod as f32,
    })
}

/// Validate and derive the topology shared by every LOD compute backend.
pub fn prepare_lod_model(model: LodModelData) -> Result<PreparedLodModel, String> {
    let num_vertices = model.positions.len() / 3;
    if model.positions.len() != num_vertices * 3 || num_vertices == 0 {
        return Err("LOD position payload is malformed".to_string());
    }
    let num_vertices_u32 = u32::try_from(num_vertices)
        .map_err(|_| "LOD vertex count exceeds the classifier ABI".to_string())?;
    if model.faces.is_empty() || model.faces.len() != model.face_nodes.len() {
        return Err("LOD face ownership does not match topology".to_string());
    }
    if model.joint_indices.len() != num_vertices
        || model.joint_weights.len() != num_vertices
    {
        return Err("LOD skinning payload does not match vertex count".to_string());
    }
    let expected_morph_scalars = model
        .num_morph_targets
        .checked_mul(num_vertices)
        .and_then(|count| count.checked_mul(3))
        .ok_or_else(|| "LOD morph payload size overflow".to_string())?;
    if model.morph_deltas.len() != expected_morph_scalars {
        return Err("LOD morph payload does not match model shape".to_string());
    }
    if !model.positions.iter().all(|value| value.is_finite())
        || !model.joint_weights.iter().flatten().all(|value| value.is_finite())
        || !model.morph_deltas.iter().all(|value| value.is_finite())
    {
        return Err("LOD model payload contains non-finite values".to_string());
    }

    let (position_rows, position_remainder) = model.positions.as_chunks::<3>();
    debug_assert!(position_remainder.is_empty());
    let topology_positions: Vec<[f64; 3]> = position_rows
        .iter()
        .map(|position| [position[0] as f64, position[1] as f64, position[2] as f64])
        .collect();
    let adjacency = build_scoped_lod_adjacency(
        &topology_positions,
        &model.faces,
        &model.face_nodes,
    )?;
    let mesh_radius = lod_mesh_radius(&model.positions, &model.faces)? as f32;
    let face_indices = model
        .faces
        .iter()
        .flat_map(|face| face.map(|vertex| vertex as f32))
        .collect();
    let node_first_faces = model.face_nodes.iter().enumerate().fold(
        HashMap::new(),
        |mut first_faces, (face, &node)| {
            first_faces.entry(node).or_insert(face);
            first_faces
        },
    );
    let residency = LodModelResidency {
        num_faces: model.faces.len(),
        num_vertices: num_vertices_u32,
        node_first_faces,
        mesh_radius,
    };
    Ok(PreparedLodModel {
        model,
        face_indices,
        adjacency,
        residency,
    })
}

fn lod_mesh_radius(positions: &[f32], faces: &[[u32; 3]]) -> Result<f64, String> {
    let num_vertices = positions.len() / 3;
    let mut used = vec![false; num_vertices];
    for face in faces {
        for &vertex in face {
            let vertex = vertex as usize;
            if vertex >= num_vertices {
                return Err(format!("LOD face references missing vertex {vertex}"));
            }
            used[vertex] = true;
        }
    }
    let used_count = used.iter().filter(|&&is_used| is_used).count();
    if used_count == 0 {
        return Err("LOD model has no referenced vertices".to_string());
    }
    let mut center = [0.0f64; 3];
    for (vertex, &is_used) in used.iter().enumerate() {
        if is_used {
            for axis in 0..3 {
                center[axis] += positions[vertex * 3 + axis] as f64;
            }
        }
    }
    for coordinate in &mut center {
        *coordinate /= used_count as f64;
    }
    let mut radius = 0.0f64;
    for (vertex, &is_used) in used.iter().enumerate() {
        if is_used {
            let delta = [
                positions[vertex * 3] as f64 - center[0],
                positions[vertex * 3 + 1] as f64 - center[1],
                positions[vertex * 3 + 2] as f64 - center[2],
            ];
            radius = radius.max(
                (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt(),
            );
        }
    }
    Ok(radius.max(1e-6))
}

/// Primary-asset animation data embedded into a packed multi-asset LOD model.
/// Static secondary vertices receive zero joint weights and zero morph deltas.
#[derive(Clone, Copy, Debug)]
pub struct LodAnimationSource<'a> {
    pub primary_vertices: usize,
    pub joint_indices: Option<&'a [[u16; 4]]>,
    pub joint_weights: Option<&'a [[f32; 4]]>,
    /// Target-major `xyz` deltas for exactly `primary_vertices` per target.
    pub morph_deltas: &'a [f32],
    pub num_morph_targets: usize,
}

/// Reconstruct one face-indexed LOD model from the renderer's immutable source
/// instances. This is the shared worker/main-renderer/WebGPU packing boundary.
pub fn build_composed_lod_model(
    instances: &[f32],
    face_nodes: &[usize],
    total_vertices: usize,
    primary_faces: usize,
    animation: LodAnimationSource<'_>,
) -> Result<LodModelData, String> {
    let num_faces = instances.len() / instance_layout::STRIDE;
    if instances.len() != num_faces * instance_layout::STRIDE || num_faces == 0 {
        return Err("composed LOD instances are malformed".to_string());
    }
    if face_nodes.len() != num_faces {
        return Err("composed LOD face ownership does not match instances".to_string());
    }
    if primary_faces > num_faces {
        return Err("composed LOD primary face boundary exceeds resident topology".to_string());
    }
    if animation.primary_vertices > total_vertices {
        return Err("composed LOD primary vertex boundary exceeds resident vertices".to_string());
    }

    let position_scalars = total_vertices
        .checked_mul(3)
        .ok_or_else(|| "composed LOD position payload size overflow".to_string())?;
    let mut positions = vec![0.0f32; position_scalars];
    let mut position_seen = vec![false; total_vertices];
    let mut faces = Vec::with_capacity(num_faces);
    for face in 0..num_faces {
        let instance = &instances[
            face * instance_layout::STRIDE..(face + 1) * instance_layout::STRIDE
        ];
        let mut vertices = [0u32; 3];
        for (corner, vertex_slot) in vertices.iter_mut().enumerate() {
            let offset = instance_layout::offset::POSITIONS + corner * 4;
            let encoded = instance[offset];
            if !encoded.is_finite() || encoded < 0.0 || encoded.fract() != 0.0
                || encoded as usize >= total_vertices
            {
                return Err(format!("composed LOD face {face} has invalid vertex index"));
            }
            let vertex = encoded as usize;
            if vertex > u32::MAX as usize {
                return Err(format!("composed LOD face {face} has an unrepresentable vertex index"));
            }
            let position = [instance[offset + 1], instance[offset + 2], instance[offset + 3]];
            if !position.iter().all(|coordinate| coordinate.is_finite()) {
                return Err(format!("composed LOD vertex {vertex} has a non-finite position"));
            }
            if position_seen[vertex] {
                let resident = &positions[vertex * 3..vertex * 3 + 3];
                if resident != position.as_slice() {
                    return Err(format!(
                        "composed LOD vertex {vertex} has inconsistent source positions"
                    ));
                }
            } else {
                positions[vertex * 3..vertex * 3 + 3].copy_from_slice(&position);
                position_seen[vertex] = true;
            }
            *vertex_slot = vertex as u32;
        }
        faces.push(vertices);
    }

    let mut joint_indices = vec![[0u16; 4]; total_vertices];
    let mut joint_weights = vec![[0.0f32; 4]; total_vertices];
    match (animation.joint_indices, animation.joint_weights) {
        (Some(indices), Some(weights))
            if indices.len() == animation.primary_vertices
                && weights.len() == animation.primary_vertices =>
        {
            if !weights.iter().flatten().all(|weight| weight.is_finite()) {
                return Err("composed LOD primary skinning payload is non-finite".to_string());
            }
            joint_indices[..animation.primary_vertices].copy_from_slice(indices);
            joint_weights[..animation.primary_vertices].copy_from_slice(weights);
        }
        (None, None) => {}
        _ => return Err("composed LOD primary skinning payload is incomplete".to_string()),
    }

    let expected_primary_morphs = animation.num_morph_targets
        .checked_mul(animation.primary_vertices)
        .and_then(|count| count.checked_mul(3))
        .ok_or_else(|| "composed LOD morph payload size overflow".to_string())?;
    if animation.morph_deltas.len() != expected_primary_morphs {
        return Err("composed LOD primary morph payload has the wrong length".to_string());
    }
    if !animation.morph_deltas.iter().all(|delta| delta.is_finite()) {
        return Err("composed LOD primary morph payload is non-finite".to_string());
    }
    let total_morphs = animation.num_morph_targets
        .checked_mul(total_vertices)
        .and_then(|count| count.checked_mul(3))
        .ok_or_else(|| "composed LOD resident morph payload size overflow".to_string())?;
    let mut morph_deltas = vec![0.0f32; total_morphs];
    for target in 0..animation.num_morph_targets {
        let source_start = target * animation.primary_vertices * 3;
        let destination_start = target * total_vertices * 3;
        morph_deltas[destination_start..destination_start + animation.primary_vertices * 3]
            .copy_from_slice(&animation.morph_deltas[
                source_start..source_start + animation.primary_vertices * 3
            ]);
    }

    Ok(LodModelData {
        positions,
        faces,
        joint_indices,
        joint_weights,
        morph_deltas,
        num_morph_targets: animation.num_morph_targets,
        face_nodes: face_nodes.to_vec(),
    })
}

/// Per-node state consumed by the composed-scene LOD pass. The four Möbius
/// quaternions and four model-matrix columns are packed without transposition.
#[derive(Clone, Debug, PartialEq)]
pub struct LodSubjectState {
    pub node: usize,
    pub mobius: [f32; 16],
    pub model: [f32; 16],
    pub pole: [f32; 4],
    pub mobius_power: f32,
    pub c_norm_sq: f32,
    pub has_pole: f32,
}

/// Exact transform selection for one backend-neutral classifier dispatch.
#[derive(Clone, Debug, PartialEq)]
pub struct LodDispatchState {
    pub subjects: Vec<LodSubjectState>,
    pub baseline_mobius: [f32; 16],
    pub baseline_model: [f32; 16],
    pub pole: [f32; 4],
    pub mobius_power: f32,
    pub c_norm_sq: f32,
    pub has_pole: f32,
}

/// Linear little-endian word records for WebGPU classifier residency.
///
/// These are deliberately plain integer arrays: callers can upload their byte
/// representation without depending on Rust struct padding, while tests can
/// compare the exact words against the WGSL storage contract.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WgslLodModelWords {
    pub faces: Vec<[u32; 4]>,
    pub positions: Vec<[u32; 4]>,
    pub skinning: Vec<[u32; 8]>,
    pub morph_deltas: Vec<[u32; 4]>,
    pub adjacency: Vec<[u32; 4]>,
}

/// Immutable scene-side words consumed by the WebGPU visibility compactor.
///
/// Batch records and eligibility change only when retained render extraction
/// changes. Current-pose visibility remains a separate one-word-per-source
/// stream so a WebGPU classifier can feed compaction without a CPU readback.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WgslVisibilityCompactionSceneWords {
    /// Exact 16-byte `VisibilityCompactionUniforms` block.
    pub uniform: [u32; 4],
    /// One 16-byte `VisibilityBatchRecord` per canonical render batch.
    pub batches: Vec<[u32; 4]>,
    /// One validated 0/1 word per source instance.
    pub source_eligibility: Vec<u32>,
}

/// Exact output words produced by the deterministic CPU compaction oracle.
/// These arrays are fixtures for native/browser WebGPU execution; they are
/// never required by the live same-device path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WgslVisibilityCompactionOracleWords {
    pub compacted_source_instances: Vec<u32>,
    pub compacted_ranges: Vec<[u32; 5]>,
    pub indirect_arguments: Vec<[u32; 5]>,
}

/// Pack retained batch shape, draw geometry, and static eligibility into the
/// exact WGSL storage ABI. Suppressed roots and disabled batches stay resident
/// but carry zero eligibility, preserving stable source-instance identities.
pub fn pack_wgsl_visibility_compaction_scene_words(
    scene: &RenderSceneSnapshot,
    geometry: RenderGeometry,
) -> Result<WgslVisibilityCompactionSceneWords, String> {
    scene.validate().map_err(|error| error.to_string())?;
    let batch_count = u32::try_from(scene.batches.len())
        .map_err(|_| "visibility batch count exceeds the WGSL ABI".to_string())?;
    let source_count = scene.batches.iter().try_fold(0u32, |total, batch| {
        let count = u32::try_from(batch.members.len())
            .map_err(|_| "visibility source count exceeds the WGSL ABI".to_string())?;
        total
            .checked_add(count)
            .ok_or_else(|| "visibility source count exceeds the WGSL ABI".to_string())
    })?;

    let mut batches = Vec::with_capacity(scene.batches.len());
    let mut source_eligibility = Vec::with_capacity(source_count as usize);
    let mut source_first_instance = 0u32;
    for batch in &scene.batches {
        let source_instance_count = u32::try_from(batch.members.len())
            .map_err(|_| "visibility source count exceeds the WGSL ABI".to_string())?;
        let index_count = match geometry {
            RenderGeometry::Triangles => batch.triangle_index_count,
            RenderGeometry::Lines => batch.line_index_count,
        };
        batches.push([
            source_first_instance,
            source_instance_count,
            index_count,
            0,
        ]);
        for member in &batch.members {
            let suppressed_root = batch.id.layer == RenderBatchLayer::RetainedRoot
                && scene
                    .suppressed_root_faces
                    .binary_search(&member.face_index)
                    .is_ok();
            source_eligibility.push(u32::from(batch.enabled && !suppressed_root));
        }
        source_first_instance = source_first_instance
            .checked_add(source_instance_count)
            .ok_or_else(|| "visibility source count exceeds the WGSL ABI".to_string())?;
    }
    debug_assert_eq!(source_first_instance, source_count);
    Ok(WgslVisibilityCompactionSceneWords {
        uniform: [batch_count, source_count, 0, 0],
        batches,
        source_eligibility,
    })
}

/// Validate and widen one CPU visibility fixture to the WGSL `u32` stream.
/// Live WebGPU execution can instead bind the classifier's resident output.
pub fn pack_wgsl_source_visibility_words(
    source_visibility: &[u8],
    expected_source_count: u32,
) -> Result<Vec<u32>, String> {
    if source_visibility.len() != expected_source_count as usize {
        return Err(format!(
            "visibility stream has {} records; expected {}",
            source_visibility.len(), expected_source_count,
        ));
    }
    source_visibility
        .iter()
        .enumerate()
        .map(|(source_instance, &visibility)| {
            if visibility > 1 {
                Err(format!(
                    "visibility source {source_instance} has invalid value {visibility}",
                ))
            } else {
                Ok(u32::from(visibility))
            }
        })
        .collect()
}

/// Freeze the exact compaction/range/indirect words expected from WebGPU.
pub fn wgsl_visibility_compaction_oracle_words(
    scene: &RenderSceneSnapshot,
    source_visibility: &[u8],
    geometry: RenderGeometry,
) -> Result<WgslVisibilityCompactionOracleWords, String> {
    let plan = VisibilityCompactionPlan::build(scene, source_visibility)
        .map_err(|error| error.to_string())?;
    let indirect = plan
        .indexed_indirect_arguments(scene, geometry)
        .map_err(|error| error.to_string())?;
    Ok(WgslVisibilityCompactionOracleWords {
        compacted_source_instances: plan.compacted_source_instances,
        compacted_ranges: plan
            .batches
            .into_iter()
            .map(|range| {
                [
                    range.batch_index,
                    range.source_first_instance,
                    range.source_instance_count,
                    range.compacted_first_instance,
                    range.compacted_instance_count,
                ]
            })
            .collect(),
        indirect_arguments: indirect
            .into_iter()
            .map(|arguments| {
                [
                    arguments.index_count,
                    arguments.instance_count,
                    arguments.first_index,
                    arguments.base_vertex as u32,
                    arguments.first_instance,
                ]
            })
            .collect(),
    })
}

/// Per-dispatch values not already carried by `LodDispatchState` or the
/// immutable prepared model. Matrices retain the column-major renderer ABI.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WgslLodDispatchMetrics {
    pub view_projection: [f32; 16],
    pub density: f32,
    pub pixel_floor: f32,
    pub max_lod: f32,
    pub viewport: [f32; 2],
    pub num_joints: u32,
}

fn wgsl_lod_subject_rows(
    prepared: &PreparedLodModel,
) -> Result<std::collections::BTreeMap<usize, u32>, String> {
    let mut rows = std::collections::BTreeMap::new();
    for &node in &prepared.model.face_nodes {
        if !rows.contains_key(&node) {
            let row = u32::try_from(rows.len())
                .map_err(|_| "LOD subject table exceeds the WGSL ABI".to_string())?;
            rows.insert(node, row);
        }
    }
    Ok(rows)
}

/// Pack immutable classifier residency into the exact WGSL record strides.
pub fn pack_wgsl_lod_model_words(
    prepared: &PreparedLodModel,
) -> Result<WgslLodModelWords, String> {
    let model = &prepared.model;
    let subject_rows = wgsl_lod_subject_rows(prepared)?;
    let mut faces = Vec::with_capacity(model.faces.len());
    for (&vertices, &node) in model.faces.iter().zip(&model.face_nodes) {
        let subject = subject_rows[&node];
        faces.push([vertices[0], vertices[1], vertices[2], subject]);
    }

    let mut positions = Vec::with_capacity(prepared.residency.num_vertices as usize);
    for position in model.positions.chunks_exact(3) {
        positions.push([
            position[0].to_bits(),
            position[1].to_bits(),
            position[2].to_bits(),
            0,
        ]);
    }

    let mut skinning = Vec::with_capacity(model.joint_indices.len());
    for (&indices, &weights) in model.joint_indices.iter().zip(&model.joint_weights) {
        skinning.push([
            u32::from(indices[0]),
            u32::from(indices[1]),
            u32::from(indices[2]),
            u32::from(indices[3]),
            weights[0].to_bits(),
            weights[1].to_bits(),
            weights[2].to_bits(),
            weights[3].to_bits(),
        ]);
    }

    let mut morph_deltas = Vec::with_capacity(
        model
            .num_morph_targets
            .saturating_mul(prepared.residency.num_vertices as usize),
    );
    for delta in model.morph_deltas.chunks_exact(3) {
        morph_deltas.push([
            delta[0].to_bits(),
            delta[1].to_bits(),
            delta[2].to_bits(),
            0,
        ]);
    }

    let mut adjacency = Vec::with_capacity(model.faces.len().saturating_mul(3));
    for edge in prepared.adjacency.chunks_exact(4) {
        let neighbor = edge[0];
        let neighbor_edge = edge[1];
        if !neighbor.is_finite()
            || neighbor.fract() != 0.0
            || neighbor < -1.0
            || neighbor > i32::MAX as f32
            || !neighbor_edge.is_finite()
            || neighbor_edge.fract() != 0.0
            || !(0.0..=2.0).contains(&neighbor_edge)
        {
            return Err("LOD adjacency cannot be represented by the WGSL ABI".to_string());
        }
        adjacency.push([
            (neighbor as i32) as u32,
            neighbor_edge as u32,
            0,
            0,
        ]);
    }

    Ok(WgslLodModelWords {
        faces,
        positions,
        skinning,
        morph_deltas,
        adjacency,
    })
}

/// Pack authored nodes into deterministic dense 160-byte WGSL rows. Missing
/// dispatch states stay invalid (`conformal.w == 0`), while arbitrarily sparse
/// scene-node IDs do not inflate GPU residency.
pub fn pack_wgsl_lod_subject_words(
    prepared: &PreparedLodModel,
    dispatch: &LodDispatchState,
) -> Result<Vec<[u32; 40]>, String> {
    let subject_rows = wgsl_lod_subject_rows(prepared)?;
    let mut packed = vec![[0; 40]; subject_rows.len().max(1)];
    for state in &dispatch.subjects {
        let row_index = subject_rows
            .get(&state.node)
            .ok_or_else(|| "LOD subject state has no resident face row".to_string())?;
        let row = &mut packed[*row_index as usize];
        for (destination, value) in row[..16].iter_mut().zip(state.mobius) {
            *destination = value.to_bits();
        }
        for (destination, value) in row[16..32].iter_mut().zip(state.model) {
            *destination = value.to_bits();
        }
        for (destination, value) in row[32..36].iter_mut().zip(state.pole) {
            *destination = value.to_bits();
        }
        row[36] = state.mobius_power.to_bits();
        row[37] = state.c_norm_sq.to_bits();
        row[38] = state.has_pole.to_bits();
        row[39] = 1.0f32.to_bits();
    }
    Ok(packed)
}

/// Pack the exact 272-byte `LodDispatchUniforms` block declared in WGSL.
pub fn pack_wgsl_lod_dispatch_words(
    prepared: &PreparedLodModel,
    dispatch: &LodDispatchState,
    metrics: WgslLodDispatchMetrics,
) -> Result<[u32; 68], String> {
    let float_inputs = dispatch
        .baseline_mobius
        .iter()
        .chain(&dispatch.baseline_model)
        .chain(&dispatch.pole)
        .chain(&metrics.view_projection)
        .copied()
        .chain([
            dispatch.mobius_power,
            dispatch.c_norm_sq,
            dispatch.has_pole,
            metrics.density,
            metrics.pixel_floor,
            metrics.max_lod,
            metrics.viewport[0],
            metrics.viewport[1],
            prepared.residency.mesh_radius,
        ]);
    if !float_inputs.clone().all(f32::is_finite)
        || metrics.density <= 0.0
        || metrics.max_lod < 1.0
        || metrics.viewport.iter().any(|&extent| extent <= 0.0)
    {
        return Err("LOD dispatch cannot be represented by the WGSL ABI".to_string());
    }

    let mut words = [0u32; 68];
    for (destination, value) in words[..16].iter_mut().zip(dispatch.baseline_mobius) {
        *destination = value.to_bits();
    }
    for (destination, value) in words[16..32].iter_mut().zip(dispatch.baseline_model) {
        *destination = value.to_bits();
    }
    for (destination, value) in words[32..36].iter_mut().zip(dispatch.pole) {
        *destination = value.to_bits();
    }
    words[36] = dispatch.mobius_power.to_bits();
    words[37] = dispatch.c_norm_sq.to_bits();
    words[38] = dispatch.has_pole.to_bits();
    words[39] = f32::from(!dispatch.subjects.is_empty()).to_bits();
    for (destination, value) in words[40..56].iter_mut().zip(metrics.view_projection) {
        *destination = value.to_bits();
    }
    words[56] = metrics.density.to_bits();
    words[57] = prepared.residency.mesh_radius.to_bits();
    words[58] = metrics.pixel_floor.to_bits();
    words[59] = metrics.max_lod.to_bits();
    words[60] = metrics.viewport[0].to_bits();
    words[61] = metrics.viewport[1].to_bits();
    words[64] = u32::try_from(prepared.residency.num_faces)
        .map_err(|_| "LOD face count exceeds the WGSL ABI".to_string())?;
    words[65] = prepared.residency.num_vertices;
    words[66] = metrics.num_joints;
    words[67] = u32::try_from(prepared.model.num_morph_targets)
        .map_err(|_| "LOD morph target count exceeds the WGSL ABI".to_string())?;
    Ok(words)
}

/// Expand the compact resident atlas lookup to the u32 storage words consumed
/// by WGSL pass two. Missing entries retain the historical 255 sentinel.
pub fn pack_wgsl_lod_atlas_words(lookup: &LodAtlasLookup) -> Vec<u32> {
    lookup.lut.iter().copied().map(u32::from).collect()
}

fn canonicalize_lod_exponents(exponents: [u32; 3]) -> ([u32; 3], u32) {
    let [a, b, c] = exponents;
    if a <= b && b <= c {
        ([a, b, c], 0)
    } else if a <= c && c <= b {
        ([a, c, b], 1)
    } else if b <= a && a <= c {
        ([b, a, c], 2)
    } else if b <= c && c <= a {
        ([b, c, a], 4)
    } else if c <= a && a <= b {
        ([c, a, b], 3)
    } else {
        ([c, b, a], 5)
    }
}

/// CPU oracle for the WGSL coherence/packing pass.
///
/// This is intentionally not used by the live WebGL2 path. It freezes exact
/// neighbor visibility, S3 canonicalization, atlas lookup, and packed-word
/// semantics for device parity tests when a WebGPU executor is attached.
pub fn reconcile_and_pack_wgsl_lod_pass2(
    pass1: &[[f32; 4]],
    adjacency: &[[u32; 4]],
    atlas_lut: &[u32],
) -> Result<Vec<u32>, String> {
    if adjacency.len() != pass1.len().saturating_mul(3) {
        return Err("WGSL pass-two adjacency shape does not match faces".to_string());
    }
    if atlas_lut.len() < 1_200 || atlas_lut.iter().any(|&index| index > u8::MAX as u32) {
        return Err("WGSL pass-two atlas lookup is malformed".to_string());
    }
    for record in pass1 {
        if record.iter().any(|value| !value.is_finite())
            || record[..3]
                .iter()
                .any(|&exponent| exponent < 0.0 || exponent > 9.0 || exponent.fract() != 0.0)
            || record[3] < 0.0
            || record[3] > 256.0
            || record[3].fract() != 0.0
        {
            return Err("WGSL pass-two intermediate record is malformed".to_string());
        }
    }

    let mut packed = Vec::with_capacity(pass1.len());
    for (face_index, face) in pass1.iter().enumerate() {
        let visible = face[3] >= 0.5;
        if !visible {
            packed.push(pack_lod_classification(
                [face[0] as u32, face[1] as u32, face[2] as u32],
                0,
                None,
                0,
            )?);
            continue;
        }

        let mut reconciled = [face[0], face[1], face[2]];
        for edge in 0..3 {
            let neighbor = adjacency[face_index * 3 + edge];
            let neighbor_face = neighbor[0] as i32;
            if neighbor_face < 0 {
                continue;
            }
            let neighbor_face = neighbor_face as usize;
            let neighbor_edge = neighbor[1] as usize;
            if neighbor_face >= pass1.len() || neighbor_edge >= 3 {
                return Err("WGSL pass-two adjacency record is out of range".to_string());
            }
            let neighbor_record = pass1[neighbor_face];
            if neighbor_record[3] >= 0.5 {
                reconciled[edge] = reconciled[edge].max(neighbor_record[neighbor_edge]);
            }
        }
        let exponents = reconciled.map(|exponent| exponent as u32);
        let (canonical, permutation) = canonicalize_lod_exponents(exponents);
        let key = canonical[0] as usize
            + canonical[1] as usize * 10
            + canonical[2] as usize * 100;
        let priority = (face[3] - 1.0) as u8;
        packed.push(pack_lod_classification(
            canonical,
            permutation,
            Some(atlas_lut[key]),
            priority,
        )?);
    }
    Ok(packed)
}

/// One exact mismatch between two backend LOD classification records.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LodClassificationMismatch {
    pub face: usize,
    pub field: usize,
    pub expected_bits: u32,
    pub actual_bits: u32,
}

/// Exact parity report for two complete classifier prefixes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LodClassificationParity {
    pub compared_faces: usize,
    pub mismatched_faces: usize,
    pub mismatched_fields: usize,
    pub examples: Vec<LodClassificationMismatch>,
}

/// Apply one already-sequenced worker publication to a retained full snapshot.
///
/// A full publication may cover only the classified primary prefix of a
/// composed scene. Sparse indices are likewise relative to that prefix. The
/// caller remains responsible for admitting the delta sequence before this
/// payload boundary; this function validates shape and applies it atomically.
pub fn apply_lod_classification_publication(
    resident: &mut Vec<f32>,
    lods: &[f32],
    indices: &[u32],
    full_snapshot: bool,
    classified_faces: usize,
    resident_faces: usize,
) -> Result<(), String> {
    if classified_faces == 0 || resident_faces == 0 || classified_faces > resident_faces {
        return Err("LOD publication has an invalid classified face domain".to_string());
    }
    let resident_fields = resident_faces
        .checked_mul(FLOATS_PER_FACE_OUTPUT)
        .ok_or_else(|| "LOD resident publication size overflow".to_string())?;
    let classified_fields = classified_faces
        .checked_mul(FLOATS_PER_FACE_OUTPUT)
        .ok_or_else(|| "LOD classified publication size overflow".to_string())?;
    if !lods.iter().all(|value| value.is_finite()) {
        return Err("LOD publication contains non-finite fields".to_string());
    }

    if full_snapshot {
        if !indices.is_empty() || lods.len() != classified_fields {
            return Err("full LOD publication has inconsistent payload shape".to_string());
        }
        if resident.is_empty() {
            if classified_faces != resident_faces {
                return Err(
                    "partial full LOD publication has no resident scene baseline".to_string(),
                );
            }
        } else if resident.len() != resident_fields {
            return Err("LOD resident snapshot has the wrong scene shape".to_string());
        }

        if resident.is_empty() {
            resident.resize(resident_fields, 0.0);
        }
        resident[..classified_fields].copy_from_slice(lods);
        return Ok(());
    }

    if resident.len() != resident_fields {
        return Err("sparse LOD publication has no matching resident baseline".to_string());
    }
    if lods.len() != indices.len().saturating_mul(FLOATS_PER_FACE_OUTPUT) {
        return Err("sparse LOD publication has inconsistent payload shape".to_string());
    }
    let mut previous = None;
    for &face in indices {
        let face = face as usize;
        if face >= classified_faces {
            return Err("sparse LOD publication references an unclassified face".to_string());
        }
        if previous.is_some_and(|previous| face <= previous) {
            return Err("sparse LOD publication indices are not strictly increasing".to_string());
        }
        previous = Some(face);
    }

    for (record, &face) in lods.chunks_exact(FLOATS_PER_FACE_OUTPUT).zip(indices) {
        let offset = face as usize * FLOATS_PER_FACE_OUTPUT;
        resident[offset..offset + FLOATS_PER_FACE_OUTPUT].copy_from_slice(record);
    }
    Ok(())
}

/// Compare two exact GPU classifier prefixes without float tolerances.
///
/// Classifier fields are powers of two, small integer permutation/parity
/// values, and an integer atlas index represented as `f32`; bit identity is
/// therefore the appropriate WebGL2/WebGPU promotion gate.
pub fn compare_lod_classifications(
    expected: &[f32],
    actual: &[f32],
) -> Result<LodClassificationParity, String> {
    if expected.len() != actual.len()
        || expected.len() % FLOATS_PER_FACE_OUTPUT != 0
    {
        return Err("LOD parity payloads have different or partial shapes".to_string());
    }
    let mut mismatched_faces = 0usize;
    let mut mismatched_fields = 0usize;
    let mut examples = Vec::new();
    for (face, (expected, actual)) in expected
        .chunks_exact(FLOATS_PER_FACE_OUTPUT)
        .zip(actual.chunks_exact(FLOATS_PER_FACE_OUTPUT))
        .enumerate()
    {
        let mut face_mismatched = false;
        for field in 0..FLOATS_PER_FACE_OUTPUT {
            let expected_bits = expected[field].to_bits();
            let actual_bits = actual[field].to_bits();
            if expected_bits == actual_bits {
                continue;
            }
            face_mismatched = true;
            mismatched_fields += 1;
            if examples.len() < 8 {
                examples.push(LodClassificationMismatch {
                    face,
                    field,
                    expected_bits,
                    actual_bits,
                });
            }
        }
        mismatched_faces += usize::from(face_mismatched);
    }
    Ok(LodClassificationParity {
        compared_faces: expected.len() / FLOATS_PER_FACE_OUTPUT,
        mismatched_faces,
        mismatched_fields,
        examples,
    })
}

/// Resolve legacy and composed-scene transforms once for every backend.
pub fn prepare_lod_dispatch_state(
    packed_subjects: &[f32],
    residency: &LodModelResidency,
    classified_faces: usize,
    legacy_mobius: [f32; 16],
) -> LodDispatchState {
    let subjects = build_lod_subject_states(
        packed_subjects,
        &residency.node_first_faces,
        classified_faces,
    );
    let baseline_mobius = if subjects.is_empty() {
        legacy_mobius
    } else {
        [
            1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0,
        ]
    };
    let baseline_model = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];
    let q = |offset: usize| {
        Quat::new(
            baseline_mobius[offset] as f64,
            baseline_mobius[offset + 1] as f64,
            baseline_mobius[offset + 2] as f64,
            baseline_mobius[offset + 3] as f64,
        )
    };
    let mobius = Mobius::new(q(0), q(4), q(8), q(12));
    let (pole, mobius_power, c_norm_sq, has_pole) = match mobius.pole() {
        Some(pole) => (
            [pole.w as f32, pole.x as f32, pole.y as f32, pole.z as f32],
            mobius.power() as f32,
            mobius.c.norm_sq() as f32,
            1.0,
        ),
        None => ([0.0; 4], 0.0, mobius.c.norm_sq() as f32, 0.0),
    };
    LodDispatchState {
        subjects,
        baseline_mobius,
        baseline_model,
        pole,
        mobius_power,
        c_norm_sq,
        has_pole,
    }
}

const SUBJECT_STATE_TEXELS: usize = 10;
pub const SUBJECT_STATE_STRIDE: usize = 33;

/// Decode the browser/WASM subject ABI into the backend-neutral node table.
/// Only nodes owning a face inside the classified prefix are retained; a
/// duplicate record deterministically resolves to its last value.
pub fn build_lod_subject_states(
    packed: &[f32],
    node_first_faces: &HashMap<usize, usize>,
    classified_faces: usize,
) -> Vec<LodSubjectState> {
    let mut states = HashMap::<usize, LodSubjectState>::new();
    for record_index in 0..packed.len() / SUBJECT_STATE_STRIDE {
        let offset = record_index * SUBJECT_STATE_STRIDE;
        let record = &packed[offset..offset + SUBJECT_STATE_STRIDE];
        if !record[0].is_finite() || record[0] < 0.0 {
            continue;
        }
        let node = record[0] as usize;
        if node_first_faces.get(&node).is_none_or(|&face| face >= classified_faces) {
            continue;
        }
        let mut transform = [0.0; 16];
        let mut model = [0.0; 16];
        transform.copy_from_slice(&record[1..17]);
        model.copy_from_slice(&record[17..33]);
        let q = |offset: usize| {
            Quat::new(
                transform[offset] as f64,
                transform[offset + 1] as f64,
                transform[offset + 2] as f64,
                transform[offset + 3] as f64,
            )
        };
        let mobius = Mobius::new(q(0), q(4), q(8), q(12));
        let (pole, power, c_norm_sq, has_pole) = match mobius.pole() {
            Some(h) => (
                [h.w as f32, h.x as f32, h.y as f32, h.z as f32],
                mobius.power() as f32,
                mobius.c.norm_sq() as f32,
                1.0,
            ),
            None => ([0.0; 4], 0.0, mobius.c.norm_sq() as f32, 0.0),
        };
        states.insert(
            node,
            LodSubjectState {
                node,
                mobius: transform,
                model,
                pole,
                mobius_power: power,
                c_norm_sq,
                has_pole,
            },
        );
    }
    let mut states: Vec<_> = states.into_values().collect();
    states.sort_by_key(|state| state.node);
    states
}

/// Build the pass-two edge-negotiation texture while preserving authored
/// ownership boundaries. Exact duplicate render vertices are welded inside
/// each node/domain, but coincident objects remain independent.
pub fn build_scoped_lod_adjacency(
    positions: &[[f64; 3]],
    faces: &[[u32; 3]],
    face_domains: &[usize],
) -> Result<Vec<f32>, String> {
    let num_faces = faces.len();
    if num_faces == 0 || face_domains.len() != num_faces {
        return Err("LOD topology ownership must match every face".to_string());
    }

    let mut adjacency = vec![0.0f32; num_faces * 3 * 4];
    for face in 0..num_faces {
        for edge_lod in 0..3 {
            adjacency[(face * 3 + edge_lod) * 4] = -1.0;
        }
    }

    let mut domains = std::collections::BTreeMap::<usize, Vec<(usize, [u32; 3])>>::new();
    for (face, (&domain, &vertices)) in face_domains.iter().zip(faces).enumerate() {
        domains.entry(domain).or_default().push((face, vertices));
    }
    for domain_faces in domains.values() {
        let local_faces: Vec<[u32; 3]> =
            domain_faces.iter().map(|(_, vertices)| *vertices).collect();
        let mesh = HalfEdgeMesh::from_triangles_welded_exact(positions, &local_faces);
        for local_face in 0..domain_faces.len() {
            let global_face = domain_faces[local_face].0;
            let half_edges = mesh.face_half_edges(local_face as u32);
            for (half_edge_index, &half_edge) in half_edges.iter().enumerate() {
                let edge_lod = (half_edge_index + 2) % 3;
                let base = (global_face * 3 + edge_lod) * 4;
                let Some(twin) =
                    quilting_mesh::unpack_twin(mesh.half_edges[half_edge as usize].twin)
                else {
                    continue;
                };
                let adjacent_local_face = mesh.half_edges[twin as usize].face as usize;
                let adjacent_half_edges = mesh.face_half_edges(adjacent_local_face as u32);
                let adjacent_lod = adjacent_half_edges
                    .iter()
                    .position(|&candidate| candidate == twin)
                    .map(|index| (index + 2) % 3)
                    .unwrap_or(0);
                adjacency[base] = domain_faces[adjacent_local_face].0 as f32;
                adjacency[base + 1] = adjacent_lod as f32;
            }
        }
    }
    Ok(adjacency)
}

/// GPU-side copy of a completed packed LOD transform-feedback output. The worker can
/// fence after staging one or more runs, poll without blocking, then read only
/// once the shared fence is signaled.
pub struct StagedLodReadback {
    buffer: ReadbackBuffer,
    num_faces: usize,
}

impl StagedLodReadback {
    pub fn byte_len(&self) -> usize {
        self.num_faces.saturating_mul(PACKED_LOD_OUTPUT_BYTES_PER_FACE)
    }
}

struct ReadbackBuffer {
    handle: glow::Buffer,
    capacity_bytes: usize,
}

const LOD_COMPUTE_VS: &str = include_str!("../shaders/lod_compute.vert.glsl");
const LOD_COMPUTE_FS: &str = include_str!("../shaders/lod_compute.frag.glsl");
const LOD_COHERENCE_VS: &str = include_str!("../shaders/lod_coherence.vert.glsl");
const DUMMY_FS: &str = include_str!("../shaders/lod_dummy.frag.glsl");

/// Owns one handle only while a multi-step GL construction is incomplete.
/// Taking the handle transfers it to the completed owner; every early return
/// instead releases it exactly once.
struct StagedHandle<'a, C, T: Copy> {
    context: &'a C,
    handle: Option<T>,
    delete: fn(&C, T),
}

impl<'a, C, T: Copy> StagedHandle<'a, C, T> {
    fn new(context: &'a C, handle: T, delete: fn(&C, T)) -> Self {
        Self { context, handle: Some(handle), delete }
    }

    fn get(&self) -> T {
        self.handle.expect("staged handle has not been transferred")
    }

    fn into_inner(mut self) -> T {
        self.handle.take().expect("staged handle has not been transferred")
    }
}

impl<C, T: Copy> Drop for StagedHandle<'_, C, T> {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            (self.delete)(self.context, handle);
        }
    }
}

fn delete_program(gl: &glow::Context, program: glow::Program) {
    unsafe { gl.delete_program(program); }
}

fn delete_shader(gl: &glow::Context, shader: glow::Shader) {
    unsafe { gl.delete_shader(shader); }
}

fn delete_vertex_array(gl: &glow::Context, vao: glow::VertexArray) {
    unsafe { gl.delete_vertex_array(vao); }
}

fn delete_buffer(gl: &glow::Context, buffer: glow::Buffer) {
    unsafe { gl.delete_buffer(buffer); }
}

fn delete_framebuffer(gl: &glow::Context, framebuffer: glow::Framebuffer) {
    unsafe { gl.delete_framebuffer(framebuffer); }
}

fn delete_transform_feedback(gl: &glow::Context, feedback: glow::TransformFeedback) {
    unsafe { gl.delete_transform_feedback(feedback); }
}

/// Two-pass LOD compute pipeline.
/// Pass 1: FBO render (LOD exponents → RGBA32F texture, one pixel per face)
/// Pass 2: Transform feedback (edge coherence + canonicalize → packed readback buffer)
pub struct LodCompute {
    // --- Pass 1: FBO render for LOD exponents ---
    program1: glow::Program,
    vao1: glow::VertexArray,
    input_buf: glow::Buffer,        // face indices (3 floats per face)
    pass1_fbo: glow::Framebuffer,   // renders LOD exponents to texture
    pass1_texture: Option<glow::Texture>,  // RGBA32F, one pixel per face
    pass1_tex_w: i32,
    pass1_tex_h: i32,

    // Textures for pass 1 (animation + geometry)
    pos_texture: Option<glow::Texture>,
    skinning_texture: Option<glow::Texture>,
    joints_texture: Option<glow::Texture>,
    joints_texture_capacity: usize,
    morph_texture: Option<glow::Texture>,
    morph_wt_texture: Option<glow::Texture>,
    morph_wt_texture_capacity: usize,
    face_subject_texture: Option<glow::Texture>,
    subject_state_texture: Option<glow::Texture>,
    subject_state_rows: usize,
    subject_state_scratch: Vec<f32>,

    // Pass 1 uniform locations
    pos_loc: Option<glow::UniformLocation>,
    skinning_loc: Option<glow::UniformLocation>,
    joints_loc: Option<glow::UniformLocation>,
    morph_deltas_loc: Option<glow::UniformLocation>,
    morph_wt_loc: Option<glow::UniformLocation>,
    face_subjects_loc: Option<glow::UniformLocation>,
    subject_states_loc: Option<glow::UniformLocation>,
    use_subject_states_loc: Option<glow::UniformLocation>,
    mob_a_loc: glow::UniformLocation,
    mob_b_loc: glow::UniformLocation,
    mob_c_loc: glow::UniformLocation,
    mob_d_loc: glow::UniformLocation,
    u_pole_loc: Option<glow::UniformLocation>,
    u_mob_k_loc: Option<glow::UniformLocation>,
    u_c_norm_sq_loc: Option<glow::UniformLocation>,
    u_has_pole_loc: Option<glow::UniformLocation>,
    model_matrix_loc: glow::UniformLocation,
    density_loc: glow::UniformLocation,
    mesh_radius_loc: glow::UniformLocation,
    min_px_loc: glow::UniformLocation,
    max_lod_loc: glow::UniformLocation,
    vp_matrix_loc: glow::UniformLocation,
    vp_width_loc: glow::UniformLocation,
    vp_height_loc: glow::UniformLocation,
    num_verts_loc: Option<glow::UniformLocation>,
    num_joints_loc: Option<glow::UniformLocation>,
    num_morph_loc: Option<glow::UniformLocation>,
    fbo_width_loc: Option<glow::UniformLocation>,
    fbo_height_loc: Option<glow::UniformLocation>,

    // --- Pass 2: TF edge coherence + canonicalize ---
    program2: glow::Program,
    vao2: glow::VertexArray,
    output_buf2: glow::Buffer,      // TF output: one packed u32 per face (final)
    tf2: glow::TransformFeedback,

    // Static adjacency texture (uploaded once per model)
    adjacency_texture: Option<glow::Texture>,

    // Atlas LUT texture (shared by pass 2)
    lut_texture: Option<glow::Texture>,

    // Pass 2 uniform locations
    p2_lods_loc: Option<glow::UniformLocation>,
    p2_adj_loc: Option<glow::UniformLocation>,
    p2_lut_loc: Option<glow::UniformLocation>,
    p2_num_faces_loc: Option<glow::UniformLocation>,

    max_faces: usize,
    bound1: bool,

    // A completed LOD job is always consumed or canceled before the next one
    // is accepted. Recycle its GPU staging buffers and CPU readback vectors
    // instead of allocating mesh-sized resources every classification.
    readback_buffers: Vec<ReadbackBuffer>,
    readback_vectors: Vec<Vec<u32>>,
    decoded_vectors: Vec<Vec<f32>>,
    readback_buffer_creations: u64,
    readback_buffer_reallocations: u64,
    readback_vector_creations: u64,
    decoded_vector_creations: u64,
}

impl LodCompute {
    pub fn new(gl: &glow::Context, max_faces: usize) -> Result<Self, String> {
        unsafe {
            // --- Pass 1: regular program (renders to FBO, not TF) ---
            let program1 = StagedHandle::new(
                gl,
                build_program(gl, LOD_COMPUTE_VS, LOD_COMPUTE_FS, "LOD pass 1")?,
                delete_program,
            );

            let req = |prog, name: &str| -> Result<glow::UniformLocation, String> {
                gl.get_uniform_location(prog, name)
                    .ok_or_else(|| format!("{name} uniform not found in pass 1"))
            };

            let mob_a_loc = req(program1.get(), "mob_a")?;
            let mob_b_loc = req(program1.get(), "mob_b")?;
            let mob_c_loc = req(program1.get(), "mob_c")?;
            let mob_d_loc = req(program1.get(), "mob_d")?;
            let u_pole_loc = gl.get_uniform_location(program1.get(), "u_pole");
            let u_mob_k_loc = gl.get_uniform_location(program1.get(), "u_mob_k");
            let u_c_norm_sq_loc = gl.get_uniform_location(program1.get(), "u_c_norm_sq");
            let u_has_pole_loc = gl.get_uniform_location(program1.get(), "u_has_pole");
            let model_matrix_loc = req(program1.get(), "model_matrix")?;
            let density_loc = req(program1.get(), "density")?;
            let mesh_radius_loc = req(program1.get(), "mesh_radius")?;
            let min_px_loc = req(program1.get(), "min_px")?;
            let max_lod_loc = req(program1.get(), "max_lod")?;
            let vp_matrix_loc = req(program1.get(), "vp_matrix")?;
            let vp_width_loc = req(program1.get(), "vp_width")?;
            let vp_height_loc = req(program1.get(), "vp_height")?;
            let num_verts_loc = gl.get_uniform_location(program1.get(), "u_num_vertices");
            let num_joints_loc = gl.get_uniform_location(program1.get(), "u_num_joints");
            let num_morph_loc = gl.get_uniform_location(program1.get(), "u_num_morph_targets");
            let fbo_width_loc = gl.get_uniform_location(program1.get(), "u_fbo_width");
            let fbo_height_loc = gl.get_uniform_location(program1.get(), "u_fbo_height");

            // Pass 1 VAO + input buffer
            let vao1 = StagedHandle::new(
                gl,
                gl.create_vertex_array().map_err(|e| format!("{e}"))?,
                delete_vertex_array,
            );
            gl.bind_vertex_array(Some(vao1.get()));

            let input_buf = StagedHandle::new(
                gl,
                gl.create_buffer().map_err(|e| format!("{e}"))?,
                delete_buffer,
            );
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(input_buf.get()));
            gl.buffer_data_size(glow::ARRAY_BUFFER,
                (max_faces * 3 * 4) as i32, glow::STATIC_DRAW);
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(0, 3, glow::FLOAT, false, 12, 0);
            gl.bind_vertex_array(None);

            // Pass 1 FBO (texture attached in upload_adjacency when we know num_faces)
            let pass1_fbo = StagedHandle::new(
                gl,
                gl.create_framebuffer().map_err(|e| format!("{e}"))?,
                delete_framebuffer,
            );

            // --- Pass 2: TF program ---
            let program2 = StagedHandle::new(
                gl,
                build_tf_program(
                    gl,
                    LOD_COHERENCE_VS,
                    DUMMY_FS,
                    &["out_packed"],
                    "LOD pass 2",
                )?,
                delete_program,
            );

            let vao2 = StagedHandle::new(
                gl,
                gl.create_vertex_array().map_err(|e| format!("{e}"))?,
                delete_vertex_array,
            );

            let output_buf2 = StagedHandle::new(
                gl,
                gl.create_buffer().map_err(|e| format!("{e}"))?,
                delete_buffer,
            );
            gl.bind_buffer(glow::TRANSFORM_FEEDBACK_BUFFER, Some(output_buf2.get()));
            gl.buffer_data_size(glow::TRANSFORM_FEEDBACK_BUFFER,
                (max_faces * PACKED_LOD_OUTPUT_BYTES_PER_FACE) as i32,
                // Transform feedback writes this buffer every classification;
                // staging then copies it GPU-to-GPU before the CPU-visible
                // STREAM_READ buffer is fenced. READ here makes Chromium
                // maintain and repeatedly discard an unnecessary shadow copy.
                glow::DYNAMIC_COPY);

            let tf2 = StagedHandle::new(
                gl,
                gl.create_transform_feedback().map_err(|e| format!("{e}"))?,
                delete_transform_feedback,
            );

            let pos_loc = gl.get_uniform_location(program1.get(), "u_positions");
            let skinning_loc = gl.get_uniform_location(program1.get(), "u_skinning");
            let joints_loc = gl.get_uniform_location(program1.get(), "u_joints");
            let morph_deltas_loc = gl.get_uniform_location(program1.get(), "u_morph_deltas");
            let morph_wt_loc = gl.get_uniform_location(program1.get(), "u_morph_wt");
            let face_subjects_loc = gl.get_uniform_location(program1.get(), "u_face_subjects");
            let subject_states_loc = gl.get_uniform_location(program1.get(), "u_subject_states");
            let use_subject_states_loc =
                gl.get_uniform_location(program1.get(), "u_use_subject_states");
            let p2_lods_loc = gl.get_uniform_location(program2.get(), "u_pass1_lods");
            let p2_adj_loc = gl.get_uniform_location(program2.get(), "u_adjacency");
            let p2_lut_loc = gl.get_uniform_location(program2.get(), "u_atlas_lut");
            let p2_num_faces_loc = gl.get_uniform_location(program2.get(), "u_num_faces");

            Ok(Self {
                program1: program1.into_inner(),
                vao1: vao1.into_inner(),
                input_buf: input_buf.into_inner(),
                pass1_fbo: pass1_fbo.into_inner(),
                pass1_texture: None, pass1_tex_w: 0, pass1_tex_h: 0,
                pos_texture: None, skinning_texture: None, joints_texture: None,
                joints_texture_capacity: 0,
                morph_texture: None, morph_wt_texture: None,
                morph_wt_texture_capacity: 0,
                face_subject_texture: None,
                subject_state_texture: None,
                subject_state_rows: 0,
                subject_state_scratch: Vec::new(),
                pos_loc,
                skinning_loc,
                joints_loc,
                morph_deltas_loc,
                morph_wt_loc,
                face_subjects_loc,
                subject_states_loc,
                use_subject_states_loc,
                mob_a_loc, mob_b_loc, mob_c_loc, mob_d_loc,
                u_pole_loc, u_mob_k_loc, u_c_norm_sq_loc, u_has_pole_loc,
                model_matrix_loc,
                density_loc, mesh_radius_loc,
                min_px_loc, max_lod_loc, vp_matrix_loc, vp_width_loc, vp_height_loc,
                num_verts_loc, num_joints_loc, num_morph_loc,
                fbo_width_loc, fbo_height_loc,

                program2: program2.into_inner(),
                vao2: vao2.into_inner(),
                output_buf2: output_buf2.into_inner(),
                tf2: tf2.into_inner(),
                adjacency_texture: None,
                lut_texture: None,
                p2_lods_loc,
                p2_adj_loc,
                p2_lut_loc,
                p2_num_faces_loc,

                max_faces,
                bound1: false,
                readback_buffers: Vec::new(),
                readback_vectors: Vec::new(),
                decoded_vectors: Vec::new(),
                readback_buffer_creations: 0,
                readback_buffer_reallocations: 0,
                readback_vector_creations: 0,
                decoded_vector_creations: 0,
            })
        }
    }

    pub fn has_pass1_texture(&self) -> bool { self.pass1_texture.is_some() }
    pub fn has_adjacency_texture(&self) -> bool { self.adjacency_texture.is_some() }

    pub fn readback_pool_stats(&self) -> (usize, usize, usize, u64, u64, u64, u64) {
        (
            self.readback_buffers.len(),
            self.readback_vectors.len(),
            self.decoded_vectors.len(),
            self.readback_buffer_creations,
            self.readback_buffer_reallocations,
            self.readback_vector_creations,
            self.decoded_vector_creations,
        )
    }

    /// Upload one fully prepared model into this context-local classifier.
    /// The returned residency contains no GL handles and can therefore stamp
    /// worker, main-context shadow, and future WebGPU jobs identically.
    pub fn upload_model(
        &mut self,
        gl: &glow::Context,
        prepared: &PreparedLodModel,
        atlas_lut: &[u8],
    ) -> LodModelResidency {
        let model = &prepared.model;
        let residency = &prepared.residency;
        self.upload_positions_texture(gl, &model.positions, residency.num_vertices as usize);
        self.upload_face_indices(gl, &prepared.face_indices);
        self.upload_face_subjects(gl, &model.face_nodes);
        self.upload_skinning_texture(gl, &model.joint_indices, &model.joint_weights);
        if model.num_morph_targets > 0 {
            self.upload_morph_deltas(
                gl,
                &model.morph_deltas,
                residency.num_vertices as usize,
                model.num_morph_targets,
            );
        } else {
            self.upload_morph_deltas(gl, &[0.0; 3], 1, 1);
        }
        self.upload_atlas_lut(gl, atlas_lut);
        self.upload_adjacency(gl, &prepared.adjacency, residency.num_faces);
        residency.clone()
    }

    // --- Static data uploads (called once on model load) ---

    /// Upload atlas LUT: exponent triples → atlas indices.
    pub fn upload_atlas_lut(&mut self, gl: &glow::Context, lut: &[u8]) {
        unsafe {
            if let Some(old) = self.lut_texture { gl.delete_texture(old); }
            let tex = gl.create_texture().unwrap();
            gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            let mut data = vec![255u8; 1200];
            for (i, &v) in lut.iter().take(1200).enumerate() { data[i] = v; }
            gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::R8 as i32,
                40, 30, 0, glow::RED, glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(&data)));
            set_nearest(gl);
            self.lut_texture = Some(tex);
            self.bound1 = false;
        }
    }

    /// Upload face indices (3 floats per face — vertex indices as floats).
    pub fn upload_face_indices(&self, gl: &glow::Context, indices: &[f32]) {
        let max_floats = self.max_faces * 3;
        let clamped = if indices.len() > max_floats { &indices[..max_floats] } else { indices };
        unsafe {
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.input_buf));
            gl.buffer_data_u8_slice(glow::ARRAY_BUFFER,
                bytemuck_cast_slice(clamped), glow::STATIC_DRAW);
        }
    }

    /// Upload the immutable face → authored-node relation and allocate the
    /// compact per-node state table used by composed-scene classifications.
    pub fn upload_face_subjects(&mut self, gl: &glow::Context, subjects: &[usize]) {
        let width = 4096usize;
        let height = subjects.len().div_ceil(width).max(1);
        let mut packed = vec![0.0f32; width * height];
        for (destination, &subject) in packed.iter_mut().zip(subjects) {
            *destination = subject as f32;
        }
        self.subject_state_rows = subjects
            .iter()
            .copied()
            .max()
            .map_or(1, |node| node.saturating_add(1));
        self.subject_state_scratch
            .resize(self.subject_state_rows * SUBJECT_STATE_TEXELS * 4, 0.0);

        unsafe {
            if let Some(old) = self.face_subject_texture { gl.delete_texture(old); }
            let face_texture = gl.create_texture().unwrap();
            gl.bind_texture(glow::TEXTURE_2D, Some(face_texture));
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::R32F as i32,
                width as i32,
                height as i32,
                0,
                glow::RED,
                glow::FLOAT,
                glow::PixelUnpackData::Slice(Some(bytemuck_cast_slice(&packed))),
            );
            set_nearest(gl);
            self.face_subject_texture = Some(face_texture);

            if let Some(old) = self.subject_state_texture { gl.delete_texture(old); }
            let state_texture = gl.create_texture().unwrap();
            gl.bind_texture(glow::TEXTURE_2D, Some(state_texture));
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGBA32F as i32,
                SUBJECT_STATE_TEXELS as i32,
                self.subject_state_rows as i32,
                0,
                glow::RGBA,
                glow::FLOAT,
                glow::PixelUnpackData::Slice(Some(bytemuck_cast_slice(
                    &self.subject_state_scratch,
                ))),
            );
            set_nearest(gl);
            self.subject_state_texture = Some(state_texture);
            self.bound1 = false;
        }
    }

    fn upload_subject_states(
        &mut self,
        gl: &glow::Context,
        states: &[LodSubjectState],
    ) -> Result<(), String> {
        if self.face_subject_texture.is_none() || self.subject_state_texture.is_none() {
            return Err("LOD subject textures are not resident".to_string());
        }
        if states.is_empty() {
            return Ok(());
        }
        self.subject_state_scratch.fill(0.0);
        for state in states {
            if state.node >= self.subject_state_rows {
                continue;
            }
            let row = state.node * SUBJECT_STATE_TEXELS * 4;
            self.subject_state_scratch[row..row + 16].copy_from_slice(&state.mobius);
            self.subject_state_scratch[row + 16..row + 32].copy_from_slice(&state.model);
            self.subject_state_scratch[row + 32..row + 36].copy_from_slice(&state.pole);
            self.subject_state_scratch[row + 36] = state.mobius_power;
            self.subject_state_scratch[row + 37] = state.c_norm_sq;
            self.subject_state_scratch[row + 38] = state.has_pole;
            self.subject_state_scratch[row + 39] = 1.0;
        }
        unsafe {
            gl.bind_texture(glow::TEXTURE_2D, self.subject_state_texture);
            gl.tex_sub_image_2d(
                glow::TEXTURE_2D,
                0,
                0,
                0,
                SUBJECT_STATE_TEXELS as i32,
                self.subject_state_rows as i32,
                glow::RGBA,
                glow::FLOAT,
                glow::PixelUnpackData::Slice(Some(bytemuck_cast_slice(
                    &self.subject_state_scratch,
                ))),
            );
            self.bound1 = false;
        }
        Ok(())
    }

    /// Upload rest-pose positions as a float texture.
    pub fn upload_positions_texture(&mut self, gl: &glow::Context, positions: &[f32], num_vertices: usize) {
        unsafe {
            let mut rgba = vec![0.0f32; num_vertices * 4];
            for i in 0..num_vertices {
                rgba[i*4]   = positions[i*3];
                rgba[i*4+1] = positions[i*3+1];
                rgba[i*4+2] = positions[i*3+2];
            }

            let width = 4096;
            let height = (num_vertices + width - 1) / width;
            rgba.resize(width * height * 4, 0.0);

            if let Some(old) = self.pos_texture { gl.delete_texture(old); }
            let tex = gl.create_texture().unwrap();
            gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::RGBA32F as i32,
                width as i32, height as i32, 0,
                glow::RGBA, glow::FLOAT,
                glow::PixelUnpackData::Slice(Some(bytemuck_cast_slice(&rgba))));
            set_nearest(gl);
            self.pos_texture = Some(tex);
            self.bound1 = false;
        }
    }

    /// Upload per-vertex skinning data (joint indices + weights).
    pub fn upload_skinning_texture(
        &mut self, gl: &glow::Context,
        joint_indices: &[[u16; 4]], joint_weights: &[[f32; 4]],
    ) {
        let nv = joint_indices.len();
        if nv == 0 { return; }
        let width = nv.min(4096);
        let height = ((nv + width - 1) / width) * 2;
        let mut data = vec![0.0f32; width * height * 4];
        for (i, (ji, jw)) in joint_indices.iter().zip(joint_weights.iter()).enumerate() {
            let chunk = i / width;
            let col = i % width;
            let idx_row = chunk * 2;
            let wt_row = chunk * 2 + 1;
            let idx_off = (idx_row * width + col) * 4;
            data[idx_off]     = ji[0] as f32;
            data[idx_off + 1] = ji[1] as f32;
            data[idx_off + 2] = ji[2] as f32;
            data[idx_off + 3] = ji[3] as f32;
            let wt_off = (wt_row * width + col) * 4;
            data[wt_off]     = jw[0];
            data[wt_off + 1] = jw[1];
            data[wt_off + 2] = jw[2];
            data[wt_off + 3] = jw[3];
        }
        unsafe {
            if let Some(old) = self.skinning_texture { gl.delete_texture(old); }
            let tex = gl.create_texture().unwrap();
            gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::RGBA32F as i32,
                width as i32, height as i32, 0, glow::RGBA, glow::FLOAT,
                glow::PixelUnpackData::Slice(Some(bytemuck_cast_slice(&data))));
            set_nearest(gl);
            self.skinning_texture = Some(tex);
            self.bound1 = false;
        }
    }

    /// Upload morph target delta texture (static).
    pub fn upload_morph_deltas(
        &mut self, gl: &glow::Context,
        deltas: &[f32], num_vertices: usize, num_targets: usize,
    ) {
        let mut rgba = vec![0.0f32; num_vertices * num_targets * 4];
        for t in 0..num_targets {
            for v in 0..num_vertices {
                let src = (t * num_vertices + v) * 3;
                let dst = (t * num_vertices + v) * 4;
                if src + 2 < deltas.len() {
                    rgba[dst]     = deltas[src];
                    rgba[dst + 1] = deltas[src + 1];
                    rgba[dst + 2] = deltas[src + 2];
                }
            }
        }
        unsafe {
            if let Some(old) = self.morph_texture { gl.delete_texture(old); }
            let tex = gl.create_texture().unwrap();
            gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::RGBA32F as i32,
                num_vertices as i32, num_targets as i32, 0,
                glow::RGBA, glow::FLOAT,
                glow::PixelUnpackData::Slice(Some(bytemuck_cast_slice(&rgba))));
            set_nearest(gl);
            self.morph_texture = Some(tex);
            self.bound1 = false;
        }
    }

    /// Upload adjacency data for edge coherence (called once per model).
    ///
    /// `data`: flat f32 array, 4 floats per entry, 3 entries per face.
    /// Entry for face `fi`, edge `e`: data[(fi*3+e)*4 .. +4] = (neighbor_face, neighbor_lod_idx, 0, 0).
    /// neighbor_face < 0 means boundary (no neighbor).
    ///
    /// Stored as RGBA32F texture, tiled 4096-wide.
    pub fn upload_adjacency(&mut self, gl: &glow::Context, data: &[f32], num_faces: usize) {
        let total_texels = num_faces * 3;
        let width = 4096;
        let height = (total_texels + width - 1) / width;
        let mut padded = vec![0.0f32; width * height * 4];
        let copy_len = data.len().min(padded.len());
        padded[..copy_len].copy_from_slice(&data[..copy_len]);

        unsafe {
            if let Some(old) = self.adjacency_texture { gl.delete_texture(old); }
            let tex = gl.create_texture().unwrap();
            gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::RGBA32F as i32,
                width as i32, height as i32, 0,
                glow::RGBA, glow::FLOAT,
                glow::PixelUnpackData::Slice(Some(bytemuck_cast_slice(&padded))));
            set_nearest(gl);
            self.adjacency_texture = Some(tex);
        }

        // Allocate pass 1 FBO texture (RGBA32F, one pixel per face)
        let p1_w = 4096i32;
        let p1_h = ((num_faces + 4095) / 4096) as i32;
        unsafe {
            if let Some(old) = self.pass1_texture { gl.delete_texture(old); }
            let tex = gl.create_texture().unwrap();
            gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::RGBA32F as i32,
                p1_w, p1_h, 0,
                glow::RGBA, glow::FLOAT,
                glow::PixelUnpackData::Slice(None));
            set_nearest(gl);
            self.pass1_texture = Some(tex);
            self.pass1_tex_w = p1_w;
            self.pass1_tex_h = p1_h;

            // Attach texture to FBO
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.pass1_fbo));
            gl.framebuffer_texture_2d(glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D, Some(tex), 0);

            let status = gl.check_framebuffer_status(glow::FRAMEBUFFER);
            if status != glow::FRAMEBUFFER_COMPLETE {
                log_info(&format!("Pass 1 FBO incomplete: 0x{:x}", status));
            }
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        }
    }

    // --- Per-frame animation uploads ---

    /// Upload joint matrices for the current frame.
    pub fn upload_joint_matrices(&mut self, gl: &glow::Context, matrices: &[f32]) {
        let num_joints = matrices.len() / 16;
        if num_joints == 0 { return; }
        unsafe {
            let needs_allocation = self.joints_texture.is_none()
                || self.joints_texture_capacity < num_joints;
            if needs_allocation {
                if let Some(old) = self.joints_texture { gl.delete_texture(old); }
                self.joints_texture = Some(gl.create_texture().unwrap());
                self.joints_texture_capacity = num_joints;
            }
            let tex = self.joints_texture.unwrap();
            gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            if needs_allocation {
                gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::RGBA32F as i32,
                    4, self.joints_texture_capacity as i32, 0,
                    glow::RGBA, glow::FLOAT,
                    glow::PixelUnpackData::Slice(None));
                set_nearest(gl);
                self.bound1 = false;
            }
            gl.tex_sub_image_2d(glow::TEXTURE_2D, 0, 0, 0,
                4, num_joints as i32, glow::RGBA, glow::FLOAT,
                glow::PixelUnpackData::Slice(Some(bytemuck_cast_slice(
                    &matrices[..num_joints * 16]
                ))));
        }
    }

    /// Upload morph weights for the current frame.
    pub fn upload_morph_weights(&mut self, gl: &glow::Context, weights: &[f32]) {
        if weights.is_empty() { return; }
        unsafe {
            let needs_allocation = self.morph_wt_texture.is_none()
                || self.morph_wt_texture_capacity < weights.len();
            if needs_allocation {
                if let Some(old) = self.morph_wt_texture { gl.delete_texture(old); }
                self.morph_wt_texture = Some(gl.create_texture().unwrap());
                self.morph_wt_texture_capacity = weights.len();
            }
            let tex = self.morph_wt_texture.unwrap();
            gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            if needs_allocation {
                gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::R32F as i32,
                    self.morph_wt_texture_capacity as i32, 1, 0,
                    glow::RED, glow::FLOAT,
                    glow::PixelUnpackData::Slice(None));
                set_nearest(gl);
                self.bound1 = false;
            }
            gl.tex_sub_image_2d(glow::TEXTURE_2D, 0, 0, 0,
                weights.len() as i32, 1, glow::RED, glow::FLOAT,
                glow::PixelUnpackData::Slice(Some(bytemuck_cast_slice(weights))));
        }
    }

    // --- Two-pass compute ---

    /// Run both LOD compute passes. Returns number of faces processed.
    ///
    /// Composed-scene subject transforms are selected per face from a compact
    /// node table, so every classification remains one two-pass GPU job.
    pub fn compute_lods(
        &mut self,
        gl: &glow::Context,
        num_faces: usize,
        num_vertices: u32,
        num_joints: u32,
        num_morph_targets: u32,
        subject_states: &[LodSubjectState],
        mobius: [f32; 16],
        model_matrix: [f32; 16],
        pole: [f32; 4],
        mob_k: f32,
        c_norm_sq: f32,
        has_pole: f32,
        density: f32,
        mesh_radius: f32,
        min_px: f32,
        max_lod: f32,
        vp_matrix: &[f32; 16],
        vp_width: f32,
        vp_height: f32,
    ) -> Result<usize, String> {
        let n = num_faces.min(self.max_faces);
        if self.pass1_texture.is_none() || self.adjacency_texture.is_none() {
            log_info(&format!("LOD compute skipped: pass1_tex={} adj_tex={}",
                self.pass1_texture.is_some(), self.adjacency_texture.is_some()));
            return Ok(0);
        }
        self.upload_subject_states(gl, subject_states)?;
        unsafe {
            // === Pass 1: LOD exponent computation (FBO render) ===
            gl.use_program(Some(self.program1));

            if !self.bound1 {
                let mut unit = 0u32;
                let bind_tex = |gl: &glow::Context, unit: &mut u32, tex: Option<glow::Texture>, loc: &Option<glow::UniformLocation>| {
                    gl.active_texture(glow::TEXTURE0 + *unit);
                    if let Some(t) = tex {
                        gl.bind_texture(glow::TEXTURE_2D, Some(t));
                    }
                    if let Some(ref l) = loc {
                        gl.uniform_1_i32(Some(l), *unit as i32);
                    }
                    *unit += 1;
                };

                bind_tex(gl, &mut unit, self.pos_texture, &self.pos_loc);           // unit 0
                bind_tex(gl, &mut unit, self.skinning_texture, &self.skinning_loc);  // unit 1
                bind_tex(gl, &mut unit, self.joints_texture, &self.joints_loc);      // unit 2
                bind_tex(gl, &mut unit, self.morph_texture, &self.morph_deltas_loc); // unit 3
                bind_tex(gl, &mut unit, self.morph_wt_texture, &self.morph_wt_loc);  // unit 4
                bind_tex(gl, &mut unit, self.face_subject_texture, &self.face_subjects_loc); // unit 5
                bind_tex(gl, &mut unit, self.subject_state_texture, &self.subject_states_loc); // unit 6

                self.bound1 = true;
            }

            // Bind pass 1 VAO and FBO
            gl.bind_vertex_array(Some(self.vao1));
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.pass1_fbo));
            gl.viewport(0, 0, self.pass1_tex_w, self.pass1_tex_h);

            // Per-frame uniforms
            if let Some(ref loc) = self.num_verts_loc {
                gl.uniform_1_i32(Some(loc), num_vertices as i32);
            }
            if let Some(ref loc) = self.num_joints_loc {
                gl.uniform_1_i32(Some(loc), num_joints as i32);
            }
            if let Some(ref loc) = self.num_morph_loc {
                gl.uniform_1_i32(Some(loc), num_morph_targets as i32);
            }
            if let Some(ref loc) = self.fbo_width_loc {
                gl.uniform_1_i32(Some(loc), self.pass1_tex_w);
            }
            if let Some(ref loc) = self.fbo_height_loc {
                gl.uniform_1_i32(Some(loc), self.pass1_tex_h);
            }

            // Re-bind per-frame textures
            if let Some(tex) = self.joints_texture {
                gl.active_texture(glow::TEXTURE2);
                gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            }
            if let Some(tex) = self.morph_wt_texture {
                gl.active_texture(glow::TEXTURE4);
                gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            }
            if let Some(tex) = self.subject_state_texture {
                gl.active_texture(glow::TEXTURE6);
                gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            }
            if let Some(ref loc) = self.use_subject_states_loc {
                gl.uniform_1_i32(Some(loc), i32::from(!subject_states.is_empty()));
            }

            gl.uniform_4_f32(Some(&self.mob_a_loc), mobius[0], mobius[1], mobius[2], mobius[3]);
            gl.uniform_4_f32(Some(&self.mob_b_loc), mobius[4], mobius[5], mobius[6], mobius[7]);
            gl.uniform_4_f32(Some(&self.mob_c_loc), mobius[8], mobius[9], mobius[10], mobius[11]);
            gl.uniform_4_f32(Some(&self.mob_d_loc), mobius[12], mobius[13], mobius[14], mobius[15]);
            if let Some(l) = &self.u_pole_loc { gl.uniform_4_f32(Some(l), pole[0], pole[1], pole[2], pole[3]); }
            if let Some(l) = &self.u_mob_k_loc { gl.uniform_1_f32(Some(l), mob_k); }
            if let Some(l) = &self.u_c_norm_sq_loc { gl.uniform_1_f32(Some(l), c_norm_sq); }
            if let Some(l) = &self.u_has_pole_loc { gl.uniform_1_f32(Some(l), has_pole); }
            gl.uniform_matrix_4_f32_slice(Some(&self.model_matrix_loc), false, &model_matrix);
            gl.uniform_1_f32(Some(&self.density_loc), density);
            gl.uniform_1_f32(Some(&self.mesh_radius_loc), mesh_radius);
            gl.uniform_1_f32(Some(&self.min_px_loc), min_px);
            gl.uniform_1_f32(Some(&self.max_lod_loc), max_lod);
            gl.uniform_matrix_4_f32_slice(Some(&self.vp_matrix_loc), false, vp_matrix);
            gl.uniform_1_f32(Some(&self.vp_width_loc), vp_width);
            gl.uniform_1_f32(Some(&self.vp_height_loc), vp_height);

            // Clear FBO and render pass 1 (one point per face → one pixel of LOD exponents)
            // Use a sentinel clear value that would produce obviously wrong LOD if read
            // The main renderer normally leaves alpha blending enabled. An
            // invisible classifier record has alpha zero but must still write
            // its bounded standby exponents, so this pass owns blend state
            // explicitly instead of inheriting context history.
            gl.disable(glow::BLEND);
            gl.clear_color(-1.0, -1.0, -1.0, 0.0);
            gl.clear(glow::COLOR_BUFFER_BIT);

            // Ensure depth test doesn't cull our points
            gl.disable(glow::DEPTH_TEST);

            gl.draw_arrays(glow::POINTS, 0, n as i32);

            // Unbind FBO — pass1_texture now contains LOD exponents
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);

            // === Pass 2: edge coherence + canonicalize (TF) ===
            gl.enable(glow::RASTERIZER_DISCARD);
            gl.use_program(Some(self.program2));

            // Bind textures for pass 2
            gl.active_texture(glow::TEXTURE0);
            if let Some(tex) = self.pass1_texture {
                gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            }
            if let Some(ref loc) = self.p2_lods_loc {
                gl.uniform_1_i32(Some(loc), 0);
            }

            gl.active_texture(glow::TEXTURE1);
            if let Some(tex) = self.adjacency_texture {
                gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            }
            if let Some(ref loc) = self.p2_adj_loc {
                gl.uniform_1_i32(Some(loc), 1);
            }

            gl.active_texture(glow::TEXTURE2);
            if let Some(tex) = self.lut_texture {
                gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            }
            if let Some(ref loc) = self.p2_lut_loc {
                gl.uniform_1_i32(Some(loc), 2);
            }

            if let Some(ref loc) = self.p2_num_faces_loc {
                gl.uniform_1_i32(Some(loc), n as i32);
            }

            gl.bind_vertex_array(Some(self.vao2));
            gl.bind_transform_feedback(glow::TRANSFORM_FEEDBACK, Some(self.tf2));
            gl.bind_buffer_base(glow::TRANSFORM_FEEDBACK_BUFFER, 0, Some(self.output_buf2));

            gl.begin_transform_feedback(glow::POINTS);
            gl.draw_arrays(glow::POINTS, 0, n as i32);
            gl.end_transform_feedback();

            gl.disable(glow::RASTERIZER_DISCARD);
            gl.bind_transform_feedback(glow::TRANSFORM_FEEDBACK, None);
            gl.bind_vertex_array(None);
            gl.use_program(None);

            // Pass 2 clobbers texture units — force rebind on next frame
            self.bound1 = false;
        }
        Ok(n)
    }

    /// Copy the latest transform-feedback result into an independent GPU
    /// staging buffer. This is GPU-to-GPU and does not wait for completion.
    pub fn stage_readback(
        &mut self,
        gl: &glow::Context,
        num_faces: usize,
    ) -> Result<StagedLodReadback, String> {
        let n = num_faces.min(self.max_faces);
        let byte_size = n
            .checked_mul(PACKED_LOD_OUTPUT_BYTES_PER_FACE)
            .ok_or_else(|| "LOD staging buffer size overflow".to_string())?;
        let byte_size_i32 = i32::try_from(byte_size)
            .map_err(|_| "LOD staging buffer exceeds WebGL2 limits".to_string())?;
        unsafe {
            let mut buffer = match self.readback_buffers.pop() {
                Some(buffer) => buffer,
                None => {
                    let handle = gl.create_buffer()
                        .map_err(|error| format!("LOD staging buffer: {error}"))?;
                    gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(handle));
                    gl.buffer_data_size(glow::COPY_WRITE_BUFFER, byte_size_i32, glow::STREAM_READ);
                    self.readback_buffer_creations += 1;
                    ReadbackBuffer { handle, capacity_bytes: byte_size }
                }
            };
            gl.bind_buffer(glow::COPY_READ_BUFFER, Some(self.output_buf2));
            gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(buffer.handle));
            if buffer.capacity_bytes < byte_size {
                gl.buffer_data_size(glow::COPY_WRITE_BUFFER, byte_size_i32, glow::STREAM_READ);
                buffer.capacity_bytes = byte_size;
                self.readback_buffer_reallocations += 1;
            }
            gl.copy_buffer_sub_data(
                glow::COPY_READ_BUFFER,
                glow::COPY_WRITE_BUFFER,
                0,
                0,
                byte_size_i32,
            );
            gl.bind_buffer(glow::COPY_READ_BUFFER, None);
            gl.bind_buffer(glow::COPY_WRITE_BUFFER, None);
            Ok(StagedLodReadback { buffer, num_faces: n })
        }
    }

    /// Read and validate a staging buffer after its fence has signaled.
    ///
    /// The packed words remain the authoritative CPU representation. Call
    /// [`Self::decode_readback_vector`] only for legacy/shadow comparison.
    pub fn finish_staged_readback(
        &mut self,
        gl: &glow::Context,
        staged: StagedLodReadback,
    ) -> Result<Vec<u32>, String> {
        let mut result = match self.readback_vectors.pop() {
            Some(result) => result,
            None => {
                self.readback_vector_creations += 1;
                Vec::new()
            }
        };
        result.resize(staged.num_faces, 0);
        unsafe {
            gl.bind_buffer(glow::COPY_READ_BUFFER, Some(staged.buffer.handle));
            gl.get_buffer_sub_data(
                glow::COPY_READ_BUFFER,
                0,
                u32_slice_as_bytes_mut(&mut result),
            );
            gl.bind_buffer(glow::COPY_READ_BUFFER, None);
        }
        self.readback_buffers.push(staged.buffer);
        if let Err(error) = validate_packed_lod_classifications(&result) {
            self.readback_vectors.push(result);
            return Err(error);
        }
        Ok(result)
    }

    /// Expand packed records only for the retained six-float worker-parity
    /// oracle. Renderer authority never needs this allocation or write pass.
    pub fn decode_readback_vector(&mut self, packed: &[u32]) -> Result<Vec<f32>, String> {
        let mut decoded = match self.decoded_vectors.pop() {
            Some(decoded) => decoded,
            None => {
                self.decoded_vector_creations += 1;
                Vec::new()
            }
        };
        if let Err(error) = decode_packed_lod_classifications(packed, &mut decoded) {
            self.decoded_vectors.push(decoded);
            return Err(error);
        }
        Ok(decoded)
    }

    /// Return a staged result that will not be consumed to the retained pool.
    pub fn discard_staged_readback(&mut self, _gl: &glow::Context, staged: StagedLodReadback) {
        self.readback_buffers.push(staged.buffer);
    }

    /// Return a completed packed CPU payload to the retained pool.
    pub fn recycle_readback_vector(&mut self, mut result: Vec<u32>) {
        result.clear();
        self.readback_vectors.push(result);
    }

    /// Return a legacy shadow decode to its separate retained pool.
    pub fn recycle_decoded_vector(&mut self, mut result: Vec<f32>) {
        result.clear();
        self.decoded_vectors.push(result);
    }

    pub fn destroy(&self, gl: &glow::Context) {
        unsafe {
            gl.delete_program(self.program1);
            gl.delete_vertex_array(self.vao1);
            gl.delete_buffer(self.input_buf);
            gl.delete_framebuffer(self.pass1_fbo);

            gl.delete_program(self.program2);
            gl.delete_vertex_array(self.vao2);
            gl.delete_buffer(self.output_buf2);
            gl.delete_transform_feedback(self.tf2);
            for buffer in &self.readback_buffers {
                gl.delete_buffer(buffer.handle);
            }

            if let Some(t) = self.pos_texture { gl.delete_texture(t); }
            if let Some(t) = self.lut_texture { gl.delete_texture(t); }
            if let Some(t) = self.skinning_texture { gl.delete_texture(t); }
            if let Some(t) = self.joints_texture { gl.delete_texture(t); }
            if let Some(t) = self.morph_texture { gl.delete_texture(t); }
            if let Some(t) = self.morph_wt_texture { gl.delete_texture(t); }
            if let Some(t) = self.face_subject_texture { gl.delete_texture(t); }
            if let Some(t) = self.subject_state_texture { gl.delete_texture(t); }
            if let Some(t) = self.pass1_texture { gl.delete_texture(t); }
            if let Some(t) = self.adjacency_texture { gl.delete_texture(t); }
        }
    }
}

/// Build a regular (non-TF) program from vertex + fragment source.
fn build_program(
    gl: &glow::Context,
    vs_src: &str,
    fs_src: &str,
    label: &str,
) -> Result<glow::Program, String> {
    unsafe {
        let vs = StagedHandle::new(
            gl,
            compile_shader(gl, glow::VERTEX_SHADER, vs_src)?,
            delete_shader,
        );
        let fs = StagedHandle::new(
            gl,
            compile_shader(gl, glow::FRAGMENT_SHADER, fs_src)?,
            delete_shader,
        );

        let program = StagedHandle::new(
            gl,
            gl.create_program().map_err(|e| format!("{e}"))?,
            delete_program,
        );
        gl.attach_shader(program.get(), vs.get());
        gl.attach_shader(program.get(), fs.get());

        gl.link_program(program.get());
        if !gl.get_program_link_status(program.get()) {
            let log = gl.get_program_info_log(program.get());
            return Err(format!("{label} link: {log}"));
        }
        gl.detach_shader(program.get(), vs.get());
        gl.detach_shader(program.get(), fs.get());
        Ok(program.into_inner())
    }
}

/// Build a TF program from vertex + fragment source with specified varyings.
fn build_tf_program(
    gl: &glow::Context,
    vs_src: &str,
    fs_src: &str,
    varyings: &[&str],
    label: &str,
) -> Result<glow::Program, String> {
    crate::shader::create_transform_feedback_program(gl, vs_src, fs_src, varyings)
        .map_err(|error| format!("{label}: {error}"))
}

unsafe fn set_nearest(gl: &glow::Context) {
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
}

fn compile_shader(gl: &glow::Context, shader_type: u32, source: &str) -> Result<glow::Shader, String> {
    unsafe {
        let shader = gl.create_shader(shader_type).map_err(|e| format!("{e}"))?;
        gl.shader_source(shader, source);
        gl.compile_shader(shader);
        if !gl.get_shader_compile_status(shader) {
            let log = gl.get_shader_info_log(shader);
            gl.delete_shader(shader);
            return Err(format!("compute shader: {log}"));
        }
        Ok(shader)
    }
}

fn bytemuck_cast_slice<T>(data: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * std::mem::size_of::<T>()) }
}

fn u32_slice_as_bytes_mut(data: &mut [u32]) -> &mut [u8] {
    unsafe {
        std::slice::from_raw_parts_mut(
            data.as_mut_ptr() as *mut u8,
            std::mem::size_of_val(data),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quilting_core::batch::{RenderBatchId, RenderBatchKey, RenderBatchMember};
    use quilting_core::instance_layout::InstanceWriter;
    use quilting_core::render::{PbrDrawClass, RenderBatchSnapshot, RenderEntityTransform};
    use quilting_core::screen_partition::ScreenPatchLeafId;
    use std::cell::RefCell;

    const IDENTITY: [f32; 16] = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];
    const IDENTITY_MOBIUS: [f32; 16] = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0,
    ];

    fn visibility_member(face_index: u32, leaf_id: ScreenPatchLeafId) -> RenderBatchMember {
        RenderBatchMember {
            face_index,
            leaf_id,
            node_index: 0,
            edge_lods: [2; 3],
            permutation_index: 0,
            vertex_lods: [2; 3],
        }
    }

    fn visibility_batch(
        material_index: usize,
        face_index: u32,
        enabled: bool,
    ) -> RenderBatchSnapshot {
        RenderBatchSnapshot {
            id: RenderBatchId::complete(RenderBatchKey {
                lod: [2; 3],
                parity_bucket: 0,
                material_index,
                render_node_index: 0,
            }),
            members: vec![visibility_member(face_index, ScreenPatchLeafId::ROOT)],
            triangle_index_count: 6 * (material_index as u32 + 1),
            line_index_count: 8 * (material_index as u32 + 1),
            transform: RenderEntityTransform {
                mobius: IDENTITY_MOBIUS,
                orientation_sign: 1,
                euclidean_model: IDENTITY,
                euclidean_normal: IDENTITY,
            },
            enabled,
            pbr_class: PbrDrawClass::Opaque,
        }
    }

    #[test]
    fn wgsl_visibility_words_match_the_stable_cpu_oracle() {
        let mut first = visibility_batch(0, 0, true);
        first
            .members
            .push(visibility_member(3, ScreenPatchLeafId::ROOT));
        let scene = RenderSceneSnapshot {
            revision: 41,
            suppressed_root_faces: Vec::new(),
            batches: vec![
                first,
                visibility_batch(1, 1, false),
                visibility_batch(2, 2, true),
            ],
        };

        let words = pack_wgsl_visibility_compaction_scene_words(
            &scene,
            RenderGeometry::Triangles,
        ).unwrap();
        assert_eq!(words.uniform, [3, 4, 0, 0]);
        assert_eq!(
            words.batches,
            [[0, 2, 6, 0], [2, 1, 12, 0], [3, 1, 18, 0]],
        );
        assert_eq!(words.source_eligibility, [1, 1, 0, 1]);
        assert_eq!(std::mem::size_of_val(&words.uniform), 16);
        assert_eq!(std::mem::size_of_val(&words.batches[0]), 16);

        let source_visibility = pack_wgsl_source_visibility_words(&[1, 0, 1, 1], 4).unwrap();
        assert_eq!(source_visibility, [1, 0, 1, 1]);
        let oracle = wgsl_visibility_compaction_oracle_words(
            &scene,
            &[1, 0, 1, 1],
            RenderGeometry::Triangles,
        ).unwrap();
        assert_eq!(oracle.compacted_source_instances, [0, 3]);
        assert_eq!(
            oracle.compacted_ranges,
            [
                [0, 0, 2, 0, 1],
                [1, 2, 1, 1, 0],
                [2, 3, 1, 1, 1],
            ],
        );
        assert_eq!(
            oracle.indirect_arguments,
            [[6, 1, 0, 0, 0], [12, 0, 0, 0, 1], [18, 1, 0, 0, 1]],
        );
        assert_eq!(std::mem::size_of_val(&oracle.compacted_ranges[0]), 20);
        assert_eq!(std::mem::size_of_val(&oracle.indirect_arguments[0]), 20);
    }

    #[test]
    fn wgsl_visibility_eligibility_replaces_suppressed_roots() {
        let mut roots = visibility_batch(0, 0, true);
        roots.id.layer = RenderBatchLayer::RetainedRoot;
        roots
            .members
            .push(visibility_member(1, ScreenPatchLeafId::ROOT));
        let mut overlay = visibility_batch(0, 0, true);
        overlay.id.layer = RenderBatchLayer::AdaptiveOverlay;
        overlay.members[0].leaf_id = ScreenPatchLeafId::ROOT.child(0).unwrap();
        let scene = RenderSceneSnapshot {
            revision: 42,
            suppressed_root_faces: vec![0],
            batches: vec![roots, overlay],
        };

        let words = pack_wgsl_visibility_compaction_scene_words(&scene, RenderGeometry::Lines)
            .unwrap();
        assert_eq!(words.uniform, [2, 3, 0, 0]);
        assert_eq!(words.batches, [[0, 2, 8, 0], [2, 1, 8, 0]]);
        assert_eq!(words.source_eligibility, [0, 1, 1]);
        let oracle = wgsl_visibility_compaction_oracle_words(
            &scene,
            &[1, 1, 1],
            RenderGeometry::Lines,
        ).unwrap();
        assert_eq!(oracle.compacted_source_instances, [1, 2]);
        assert_eq!(
            oracle.indirect_arguments,
            [[8, 1, 0, 0, 0], [8, 1, 0, 0, 1]],
        );
    }

    #[test]
    fn wgsl_visibility_fixture_rejects_bad_shape_and_values() {
        assert_eq!(
            pack_wgsl_source_visibility_words(&[1, 0], 3).unwrap_err(),
            "visibility stream has 2 records; expected 3",
        );
        assert_eq!(
            pack_wgsl_source_visibility_words(&[1, 2, 0], 3).unwrap_err(),
            "visibility source 1 has invalid value 2",
        );
    }

    fn subject_record(node: f32, marker: f32) -> Vec<f32> {
        let mut record = vec![0.0; SUBJECT_STATE_STRIDE];
        record[0] = node;
        record[1] = 1.0;
        record[13] = 1.0;
        record[17] = marker;
        record[22] = 1.0;
        record[27] = 1.0;
        record[32] = 1.0;
        record
    }

    #[test]
    fn staged_handle_releases_an_untransferred_resource_exactly_once() {
        let deleted = RefCell::new(Vec::new());
        {
            let _staged = StagedHandle::new(&deleted, 17_u32, |deleted, handle| {
                deleted.borrow_mut().push(handle);
            });
        }
        assert_eq!(*deleted.borrow(), vec![17]);
    }

    #[test]
    fn staged_handle_does_not_release_a_transferred_resource() {
        let deleted = RefCell::new(Vec::new());
        let handle = StagedHandle::new(&deleted, 23_u32, |deleted, handle| {
            deleted.borrow_mut().push(handle);
        }).into_inner();
        assert_eq!(handle, 23);
        assert!(deleted.borrow().is_empty());
    }

    #[test]
    fn staged_handles_release_in_reverse_construction_order() {
        let deleted = RefCell::new(Vec::new());
        {
            let _shader = StagedHandle::new(&deleted, "shader", |deleted, handle| {
                deleted.borrow_mut().push(handle);
            });
            let _program = StagedHandle::new(&deleted, "program", |deleted, handle| {
                deleted.borrow_mut().push(handle);
            });
        }
        assert_eq!(*deleted.borrow(), vec!["program", "shader"]);
    }

    #[test]
    fn visibility_payload_survives_both_gpu_passes() {
        assert_eq!(FLOATS_PER_FACE_PASS1, 4);
        assert!(LOD_COMPUTE_VS.contains("flat out vec4 v_lods"));
        assert!(LOD_COMPUTE_FS.contains("frag_color = v_lods"));
        assert!(LOD_COHERENCE_VS.contains("face.w < 0.5"));
        assert!(LOD_COMPUTE_VS.contains("min_px > 0.0 ? min(2.0, max_lod) : max_lod"));
        assert!(LOD_COHERENCE_VS.contains("int(face.x + 0.5)"));
        assert!(LOD_COHERENCE_VS.contains("out_packed = pack_lod"));
        assert!(LOD_COHERENCE_VS.contains("false,"));
    }

    #[test]
    fn adaptive_priority_survives_without_expanding_the_readback() {
        assert!(LOD_COMPUTE_VS.contains("float metric_variation = max("));
        assert!(LOD_COMPUTE_VS.contains("float pole_octaves = 0.0"));
        assert!(LOD_COMPUTE_VS.contains("1.0 + adaptive_priority"));
        assert!(LOD_COHERENCE_VS.contains("int adaptive_priority = clamp("));
        assert!(LOD_COHERENCE_VS.contains("uint(adaptive_priority) << 24u"));
        assert_eq!(PACKED_LOD_OUTPUT_BYTES_PER_FACE, 4);
    }

    #[test]
    fn gpu_lod_pass_allows_the_source_triangle_level() {
        assert!(LOD_COMPUTE_VS.contains(
            "lod_a = min(lod_a, clamp(floor_pow2"
        ));
        assert!(LOD_COMPUTE_VS.contains(
            "lod_b = min(lod_b, clamp(floor_pow2"
        ));
        assert!(LOD_COMPUTE_VS.contains(
            "lod_c = min(lod_c, clamp(floor_pow2"
        ));
        assert!(!LOD_COMPUTE_VS.contains(", 2.0, max_lod)"));
    }

    #[test]
    fn screen_attenuation_caps_instead_of_driving_lod() {
        assert!(LOD_COMPUTE_VS.contains("density/curvature demand above"));
        assert_eq!(LOD_COMPUTE_VS.matches("= min(lod_").count(), 3);
        assert_eq!(LOD_COMPUTE_VS.matches("floor_pow2(min(max(px_").count(), 3);
        assert!(LOD_COMPUTE_VS.contains("float max_screen_extent = length(vec2(vp_width, vp_height))"));
    }

    #[test]
    fn pole_demand_is_bounded_by_screen_capacity() {
        let demand = LOD_COMPUTE_VS.find("world_demand_a = max_lod").unwrap();
        let attenuation = LOD_COMPUTE_VS.find("if (min_px > 0.0)").unwrap();
        assert!(demand < attenuation);
        assert!(!LOD_COMPUTE_VS[attenuation..].contains("lod_a = max_lod"));
    }

    #[test]
    fn gpu_lod_keeps_skinny_face_demands_per_edge() {
        assert!(LOD_COMPUTE_VS.contains("float source_l_a = distance(p1, p2)"));
        assert!(LOD_COMPUTE_VS.contains("intrinsic_peak * source_l_b"));
        assert!(LOD_COMPUTE_VS.contains("snap_pow2(world_demand_c)"));
        assert!(LOD_COMPUTE_VS.contains("projected_peak * source_l_a"));
    }

    #[test]
    fn gpu_intrinsic_lod_removes_global_mobius_similarity() {
        assert!(LOD_COMPUTE_VS.contains(
            "float intrinsic_similarity = active_has_pole > 0.5"
        ));
        assert!(LOD_COMPUTE_VS.contains("max(active_mob_k, 1e-12)"));
        assert!(LOD_COMPUTE_VS.contains("target_size * intrinsic_similarity"));
        assert!(LOD_COMPUTE_VS.contains("lambda_star / (intrinsic_similarity * target_size)"));
    }

    #[test]
    fn composed_subject_state_is_selected_in_the_single_face_pass() {
        assert!(LOD_COMPUTE_VS.contains("uniform highp sampler2D u_face_subjects"));
        assert!(LOD_COMPUTE_VS.contains("uniform highp sampler2D u_subject_states"));
        assert!(LOD_COMPUTE_VS.contains("int subject = int(texelFetch"));
        assert!(LOD_COMPUTE_VS.contains("active_model_matrix = mat4("));
    }

    #[test]
    fn wgsl_classifier_payload_words_have_stable_layout_and_offsets() {
        let prepared = prepare_lod_model(LodModelData {
            positions: vec![1.0, -2.0, 3.0, 4.0, 5.0, 6.0, -7.0, 8.0, 9.0],
            faces: vec![[0, 1, 2]],
            joint_indices: vec![[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            joint_weights: vec![
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.25, 0.125, 0.125],
                [1.0, 0.0, 0.0, 0.0],
            ],
            morph_deltas: vec![
                0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
            ],
            num_morph_targets: 1,
            face_nodes: vec![2],
        })
        .unwrap();
        let model = pack_wgsl_lod_model_words(&prepared).unwrap();
        assert_eq!(model.faces, vec![[0, 1, 2, 0]]);
        assert_eq!(model.positions[0], [1.0f32.to_bits(), (-2.0f32).to_bits(), 3.0f32.to_bits(), 0]);
        assert_eq!(
            model.skinning[0],
            [
                1,
                2,
                3,
                4,
                0.1f32.to_bits(),
                0.2f32.to_bits(),
                0.3f32.to_bits(),
                0.4f32.to_bits(),
            ],
        );
        assert_eq!(
            model.morph_deltas[2],
            [0.07f32.to_bits(), 0.08f32.to_bits(), 0.09f32.to_bits(), 0],
        );
        assert_eq!(model.adjacency, vec![[u32::MAX, 0, 0, 0]; 3]);
        assert_eq!(std::mem::size_of_val(&model.faces[0]), 16);
        assert_eq!(std::mem::size_of_val(&model.positions[0]), 16);
        assert_eq!(std::mem::size_of_val(&model.skinning[0]), 32);
        assert_eq!(std::mem::size_of_val(&model.morph_deltas[0]), 16);
        assert_eq!(std::mem::size_of_val(&model.adjacency[0]), 16);

        let dispatch = LodDispatchState {
            subjects: vec![LodSubjectState {
                node: 2,
                mobius: [2.0; 16],
                model: [3.0; 16],
                pole: [4.0; 4],
                mobius_power: 5.0,
                c_norm_sq: 6.0,
                has_pole: 1.0,
            }],
            baseline_mobius: [7.0; 16],
            baseline_model: [8.0; 16],
            pole: [9.0; 4],
            mobius_power: 10.0,
            c_norm_sq: 11.0,
            has_pole: 1.0,
        };
        let subjects = pack_wgsl_lod_subject_words(&prepared, &dispatch).unwrap();
        assert_eq!(subjects.len(), 1);
        assert_eq!(subjects[0][..16], [2.0f32.to_bits(); 16]);
        assert_eq!(subjects[0][16..32], [3.0f32.to_bits(); 16]);
        assert_eq!(subjects[0][32..36], [4.0f32.to_bits(); 4]);
        assert_eq!(
            &subjects[0][36..40],
            &[
                5.0f32.to_bits(),
                6.0f32.to_bits(),
                1.0f32.to_bits(),
                1.0f32.to_bits(),
            ],
        );
        assert_eq!(std::mem::size_of_val(&subjects[0]), 160);

        let view_projection = std::array::from_fn(|index| 20.0 + index as f32);
        let uniform = pack_wgsl_lod_dispatch_words(
            &prepared,
            &dispatch,
            WgslLodDispatchMetrics {
                view_projection,
                density: 12.0,
                pixel_floor: 13.0,
                max_lod: 64.0,
                viewport: [1920.0, 1080.0],
                num_joints: 14,
            },
        )
        .unwrap();
        assert_eq!(std::mem::size_of_val(&uniform), 272);
        assert_eq!(uniform[..16], [7.0f32.to_bits(); 16]);
        assert_eq!(uniform[16..32], [8.0f32.to_bits(); 16]);
        assert_eq!(uniform[32..36], [9.0f32.to_bits(); 4]);
        assert_eq!(
            &uniform[36..40],
            &[
                10.0f32.to_bits(),
                11.0f32.to_bits(),
                1.0f32.to_bits(),
                1.0f32.to_bits(),
            ],
        );
        assert_eq!(
            &uniform[40..56],
            &view_projection.map(f32::to_bits),
        );
        assert_eq!(uniform[56], 12.0f32.to_bits());
        assert_eq!(uniform[57], prepared.residency.mesh_radius.to_bits());
        assert_eq!(uniform[58], 13.0f32.to_bits());
        assert_eq!(uniform[59], 64.0f32.to_bits());
        assert_eq!(uniform[60], 1920.0f32.to_bits());
        assert_eq!(uniform[61], 1080.0f32.to_bits());
        assert_eq!(&uniform[62..64], &[0, 0]);
        assert_eq!(&uniform[64..68], &[1, 3, 14, 1]);
    }

    #[test]
    fn wgsl_pass_two_oracle_matches_every_s3_packed_word() {
        let pass1 = [
            [1.0, 2.0, 3.0, 11.0],
            [1.0, 3.0, 2.0, 12.0],
            [2.0, 1.0, 3.0, 13.0],
            [3.0, 1.0, 2.0, 14.0],
            [2.0, 3.0, 1.0, 15.0],
            [3.0, 2.0, 1.0, 16.0],
            [4.0, 5.0, 6.0, 0.0],
        ];
        let adjacency = vec![[u32::MAX, 0, 0, 0]; pass1.len() * 3];
        let mut atlas = vec![u8::MAX as u32; 1_200];
        atlas[321] = 77;
        let actual = reconcile_and_pack_wgsl_lod_pass2(&pass1, &adjacency, &atlas).unwrap();
        let permutations = [0, 1, 2, 4, 3, 5];
        for (face, permutation) in permutations.into_iter().enumerate() {
            assert_eq!(
                actual[face],
                pack_lod_classification(
                    [1, 2, 3],
                    permutation,
                    Some(77),
                    10 + face as u8,
                )
                .unwrap(),
            );
        }
        assert_eq!(
            actual[6],
            pack_lod_classification([4, 5, 6], 0, None, 0).unwrap(),
        );
    }

    #[test]
    fn wgsl_pass_two_oracle_reconciles_only_visible_neighbors() {
        let pass1 = [
            [1.0, 1.0, 1.0, 1.0],
            [3.0, 2.0, 1.0, 2.0],
            [8.0, 8.0, 8.0, 0.0],
        ];
        let mut adjacency = vec![[u32::MAX, 0, 0, 0]; pass1.len() * 3];
        adjacency[0] = [1, 0, 0, 0];
        adjacency[1] = [2, 0, 0, 0];
        let mut atlas = vec![u8::MAX as u32; 1_200];
        atlas[311] = 31;
        atlas[321] = 32;
        let actual = reconcile_and_pack_wgsl_lod_pass2(&pass1, &adjacency, &atlas).unwrap();
        assert_eq!(
            actual[0],
            pack_lod_classification([1, 1, 3], 4, Some(31), 0).unwrap(),
        );
        assert_eq!(
            actual[1],
            pack_lod_classification([1, 2, 3], 5, Some(32), 1).unwrap(),
        );
        assert_eq!(
            actual[2],
            pack_lod_classification([8, 8, 8], 0, None, 0).unwrap(),
        );
    }

    #[test]
    fn subject_table_covers_only_the_classified_face_domain() {
        let packed = [
            subject_record(1.0, 11.0),
            subject_record(2.0, 22.0),
            subject_record(3.0, 33.0),
        ]
        .concat();
        let first_faces = HashMap::from([(1, 0), (2, 1), (3, 3)]);

        let scene = build_lod_subject_states(&packed, &first_faces, 3);
        assert_eq!(scene.iter().map(|state| state.node).collect::<Vec<_>>(), [1, 2]);

        let primary = build_lod_subject_states(&packed, &first_faces, 1);
        assert_eq!(primary.iter().map(|state| state.node).collect::<Vec<_>>(), [1]);
    }

    #[test]
    fn last_duplicate_subject_record_wins_deterministically() {
        let packed = [subject_record(2.0, 20.0), subject_record(2.0, 29.0)].concat();
        let states = build_lod_subject_states(&packed, &HashMap::from([(2, 0)]), 1);
        assert_eq!(states.len(), 1);
        assert_eq!(states[0].model[0], 29.0);
    }

    #[test]
    fn dispatch_state_uses_legacy_transform_without_composed_subjects() {
        let legacy = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            1.0, 2.0, 3.0, 4.0,
        ];
        let residency = LodModelResidency {
            num_faces: 1,
            num_vertices: 3,
            node_first_faces: HashMap::from([(7, 0)]),
            mesh_radius: 1.0,
        };
        let state = prepare_lod_dispatch_state(&[], &residency, 1, legacy);
        assert!(state.subjects.is_empty());
        assert_eq!(state.baseline_mobius, legacy);
        assert_eq!(state.baseline_model[0], 1.0);
        assert_eq!(state.baseline_model[15], 1.0);
    }

    #[test]
    fn dispatch_state_uses_identity_baseline_for_composed_subjects() {
        let residency = LodModelResidency {
            num_faces: 2,
            num_vertices: 4,
            node_first_faces: HashMap::from([(3, 0), (8, 1)]),
            mesh_radius: 1.0,
        };
        let packed = [subject_record(3.0, 30.0), subject_record(8.0, 80.0)].concat();
        let state = prepare_lod_dispatch_state(&packed, &residency, 1, [9.0; 16]);
        assert_eq!(state.subjects.len(), 1);
        assert_eq!(state.subjects[0].node, 3);
        assert_eq!(state.baseline_mobius[0], 1.0);
        assert_eq!(state.baseline_mobius[12], 1.0);
        assert_eq!(state.has_pole, 0.0);
    }

    #[test]
    fn lod_publications_reconstruct_full_and_sparse_composed_prefixes() {
        let scene = [
            1.0, 1.0, 1.0, 0.0, 1.0, 3.0,
            2.0, 2.0, 2.0, 1.0, 1.0, 4.0,
            4.0, 4.0, 4.0, 2.0, 1.0, 5.0,
        ];
        let mut resident = Vec::new();
        apply_lod_classification_publication(&mut resident, &scene, &[], true, 3, 3)
            .unwrap();
        assert_eq!(resident, scene);

        let primary = [8.0, 8.0, 8.0, 3.0, 1.0, 6.0];
        apply_lod_classification_publication(&mut resident, &primary, &[], true, 1, 3)
            .unwrap();
        assert_eq!(&resident[..6], &primary);
        assert_eq!(&resident[6..], &scene[6..]);

        let sparse = [16.0, 16.0, 16.0, 4.0, 1.0, 7.0];
        apply_lod_classification_publication(
            &mut resident,
            &sparse,
            &[1],
            false,
            2,
            3,
        )
        .unwrap();
        assert_eq!(&resident[6..12], &sparse);
        assert_eq!(&resident[12..], &scene[12..]);
    }

    #[test]
    fn lod_publications_fail_atomically_without_a_scene_baseline() {
        let mut resident = Vec::new();
        let error = apply_lod_classification_publication(
            &mut resident,
            &[1.0; FLOATS_PER_FACE_OUTPUT],
            &[],
            true,
            1,
            2,
        )
        .unwrap_err();
        assert!(error.contains("no resident scene baseline"));
        assert!(resident.is_empty());

        let mut resident = vec![2.0; FLOATS_PER_FACE_OUTPUT * 2];
        let before = resident.clone();
        let error = apply_lod_classification_publication(
            &mut resident,
            &[4.0; FLOATS_PER_FACE_OUTPUT * 2],
            &[1, 1],
            false,
            2,
            2,
        )
        .unwrap_err();
        assert!(error.contains("strictly increasing"));
        assert_eq!(resident, before);
    }

    #[test]
    fn lod_parity_is_bit_exact_and_reports_bounded_examples() {
        let expected = vec![1.0; FLOATS_PER_FACE_OUTPUT * 3];
        let mut actual = expected.clone();
        actual[1] = 2.0;
        actual[FLOATS_PER_FACE_OUTPUT + 2] = 4.0;
        actual[FLOATS_PER_FACE_OUTPUT + 3] = 3.0;

        let parity = compare_lod_classifications(&expected, &actual).unwrap();
        assert_eq!(parity.compared_faces, 3);
        assert_eq!(parity.mismatched_faces, 2);
        assert_eq!(parity.mismatched_fields, 3);
        assert_eq!(parity.examples.len(), 3);
        assert_eq!(parity.examples[0].face, 0);
        assert_eq!(parity.examples[1].face, 1);
        assert!(compare_lod_classifications(&expected, &actual[..6]).is_err());
    }

    #[test]
    fn packed_lod_readback_roundtrips_every_field_domain() {
        for a in 0..=9 {
            for b in 0..=9 {
                for c in 0..=9 {
                    for permutation in 0..=5 {
                        for atlas_index in [None, Some(0), Some(1), Some(254), Some(255)] {
                            for adaptation_priority in [0, 1, 127, 255] {
                                let packed = pack_lod_classification(
                                    [a, b, c],
                                    permutation,
                                    atlas_index,
                                    adaptation_priority,
                                )
                                .unwrap();
                                let fields = unpack_lod_classification_fields(packed).unwrap();
                                assert_eq!(
                                    fields.canonical,
                                    [1u32 << a, 1u32 << b, 1u32 << c],
                                );
                                assert_eq!(fields.permutation, permutation as u8);
                                assert_eq!(
                                    fields.parity_bucket,
                                    u8::from(matches!(permutation, 1 | 2 | 5)),
                                );
                                assert_eq!(
                                    fields.atlas_index,
                                    atlas_index.map(|index| index as u8),
                                );
                                assert_eq!(fields.adaptation_priority, adaptation_priority);
                                let semantic = fields.into_face_lod_classification();
                                assert_eq!(semantic.requested.canonical, fields.canonical);
                                assert_eq!(semantic.requested.perm_index, permutation as usize);
                                assert_eq!(
                                    semantic.requested.parity_bucket,
                                    fields.parity_bucket as usize,
                                );
                                assert_eq!(semantic.visible, atlas_index.is_some());
                                let decoded = unpack_lod_classification(packed).unwrap();
                                assert_eq!(decoded[0], (1u32 << a) as f32);
                                assert_eq!(decoded[1], (1u32 << b) as f32);
                                assert_eq!(decoded[2], (1u32 << c) as f32);
                                assert_eq!(decoded[3], permutation as f32);
                                assert_eq!(
                                    decoded[4],
                                    if matches!(permutation, 1 | 2 | 5) {
                                        -1.0
                                    } else {
                                        1.0
                                    },
                                );
                                assert_eq!(
                                    decoded[5],
                                    atlas_index.map_or(-1.0, |index| index as f32),
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn packed_lod_readback_has_a_stable_32_bit_layout() {
        let packed = pack_lod_classification([1, 5, 9], 4, Some(254), 173).unwrap();
        assert_eq!(
            packed,
            1 | (5 << 4) | (9 << 8) | (4 << 12) | (1 << 15) | (254 << 16) | (173 << 24),
        );
        assert_eq!(PACKED_LOD_OUTPUT_BYTES_PER_FACE, 4);
        assert!(pack_lod_classification([10, 0, 0], 0, None, 0).is_err());
        assert!(pack_lod_classification([0, 0, 0], 6, None, 0).is_err());
        assert!(pack_lod_classification([0, 0, 0], 0, Some(256), 0).is_err());
        assert_eq!(
            unpack_lod_classification_fields(1 << 24)
                .unwrap()
                .adaptation_priority,
            1,
        );
        assert!(unpack_lod_classification(10).is_err());
        assert!(unpack_lod_classification(6 << 12).is_err());
    }

    #[test]
    fn packed_lod_delta_distinguishes_full_sparse_and_noop_publications() {
        let mut faces = vec![91];
        let mut words = vec![92];
        let full = diff_packed_lod_classifications(
            &[1, 2, 3],
            &[],
            &mut faces,
            &mut words,
        );
        assert_eq!(
            full,
            PackedLodDeltaSummary {
                full_snapshot: true,
                changed_records: 3,
            },
        );
        assert!(faces.is_empty());
        assert!(words.is_empty());

        let sparse = diff_packed_lod_classifications(
            &[1, 20, 3, 40],
            &[1, 2, 3, 4],
            &mut faces,
            &mut words,
        );
        assert_eq!(
            sparse,
            PackedLodDeltaSummary {
                full_snapshot: false,
                changed_records: 2,
            },
        );
        assert_eq!(faces, vec![1, 3]);
        assert_eq!(words, vec![20, 40]);

        let priority_only = diff_packed_lod_classifications(
            &[1 | (9 << 24), 20 | (7 << 24), 3 | (5 << 24), 40 | (3 << 24)],
            &[1, 20, 3, 40],
            &mut faces,
            &mut words,
        );
        assert!(priority_only.is_unchanged());
        assert!(faces.is_empty());
        assert!(words.is_empty());

        let unchanged = diff_packed_lod_classifications(
            &[1, 20, 3, 40],
            &[1, 20, 3, 40],
            &mut faces,
            &mut words,
        );
        assert!(unchanged.is_unchanged());
        assert!(faces.is_empty());
        assert!(words.is_empty());
    }

    fn coincident_square() -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
        (
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            vec![[0, 1, 2], [3, 4, 5]],
        )
    }

    fn adjacent_edges(adjacency: &[f32]) -> usize {
        (0..adjacency.len())
            .step_by(4)
            .filter(|&offset| adjacency[offset] >= 0.0)
            .count()
    }

    #[test]
    fn exact_render_seams_negotiate_inside_one_node() {
        let (positions, faces) = coincident_square();
        let adjacency = build_scoped_lod_adjacency(&positions, &faces, &[0, 0]).unwrap();
        assert_eq!(adjacent_edges(&adjacency), 2);
    }

    #[test]
    fn coincident_edges_do_not_join_distinct_nodes() {
        let (positions, faces) = coincident_square();
        let adjacency = build_scoped_lod_adjacency(&positions, &faces, &[0, 1]).unwrap();
        assert_eq!(adjacent_edges(&adjacency), 0);
    }

    #[test]
    fn topology_ownership_must_cover_every_face() {
        let (positions, faces) = coincident_square();
        assert!(build_scoped_lod_adjacency(&positions, &faces, &[0]).is_err());
        assert!(build_scoped_lod_adjacency(&positions, &faces, &[0, 1, 2]).is_err());
    }

    fn two_face_instances() -> Vec<f32> {
        let positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        let mut instances = vec![0.0; 2 * instance_layout::STRIDE];
        for (face, vertices) in [[0, 1, 2], [1, 3, 2]].into_iter().enumerate() {
            let mut writer = InstanceWriter::new(&mut instances, face);
            for (corner, vertex) in vertices.into_iter().enumerate() {
                writer.set_position(corner, vertex, positions[vertex as usize]);
            }
        }
        instances
    }

    #[test]
    fn composed_model_extends_primary_animation_with_static_secondary_vertices() {
        let instances = two_face_instances();
        let joint_indices = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]];
        let joint_weights = [
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
            [0.25, 0.25, 0.25, 0.25],
        ];
        let morph_deltas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let model = build_composed_lod_model(
            &instances,
            &[4, 9],
            4,
            1,
            LodAnimationSource {
                primary_vertices: 3,
                joint_indices: Some(&joint_indices),
                joint_weights: Some(&joint_weights),
                morph_deltas: &morph_deltas,
                num_morph_targets: 1,
            },
        ).unwrap();

        assert_eq!(model.faces, [[0, 1, 2], [1, 3, 2]]);
        assert_eq!(model.face_nodes, [4, 9]);
        assert_eq!(model.positions[9..12], [1.0, 1.0, 0.0]);
        assert_eq!(&model.joint_indices[..3], &joint_indices);
        assert_eq!(model.joint_indices[3], [0; 4]);
        assert_eq!(&model.joint_weights[..3], &joint_weights);
        assert_eq!(model.joint_weights[3], [0.0; 4]);
        assert_eq!(&model.morph_deltas[..9], &morph_deltas);
        assert_eq!(model.morph_deltas[9..12], [0.0; 3]);
    }

    #[test]
    fn prepared_model_freezes_backend_neutral_residency_metadata() {
        let instances = two_face_instances();
        let model = build_composed_lod_model(
            &instances,
            &[4, 9],
            4,
            1,
            LodAnimationSource {
                primary_vertices: 0,
                joint_indices: None,
                joint_weights: None,
                morph_deltas: &[],
                num_morph_targets: 0,
            },
        )
        .unwrap();
        let prepared = prepare_lod_model(model).unwrap();

        assert_eq!(prepared.residency.num_faces, 2);
        assert_eq!(prepared.residency.num_vertices, 4);
        assert_eq!(
            prepared.residency.node_first_faces,
            HashMap::from([(4, 0), (9, 1)]),
        );
        assert!((prepared.residency.mesh_radius - 0.5_f32.sqrt()).abs() < 1.0e-6);
        assert_eq!(
            prepared.face_indices,
            [0.0, 1.0, 2.0, 1.0, 3.0, 2.0],
        );
        assert_eq!(prepared.adjacency.len(), 2 * 3 * 4);
        assert_eq!(adjacent_edges(&prepared.adjacency), 0);
    }

    #[test]
    fn prepared_model_fingerprint_is_exact_and_component_local() {
        let model = build_composed_lod_model(
            &two_face_instances(),
            &[4, 9],
            4,
            1,
            LodAnimationSource {
                primary_vertices: 0,
                joint_indices: None,
                joint_weights: None,
                morph_deltas: &[],
                num_morph_targets: 0,
            },
        )
        .unwrap();
        let prepared = prepare_lod_model(model).unwrap();
        let fingerprint = prepared_lod_model_fingerprint(&prepared);
        assert_eq!(
            fingerprint,
            prepared_lod_model_fingerprint(&prepared.clone()),
        );
        assert!(fingerprint.stable_text().starts_with("lod-model-v1:"));

        let mut changed = prepared.clone();
        changed.model.joint_weights[0][0] = -0.0;
        let changed = prepared_lod_model_fingerprint(&changed);
        assert_eq!(changed.positions, fingerprint.positions);
        assert_eq!(changed.faces, fingerprint.faces);
        assert_ne!(changed.joint_weights, fingerprint.joint_weights);
    }

    #[test]
    fn prepared_model_rejects_morph_shape_even_when_target_count_is_zero() {
        let mut model = build_composed_lod_model(
            &two_face_instances(),
            &[0, 0],
            4,
            2,
            LodAnimationSource {
                primary_vertices: 0,
                joint_indices: None,
                joint_weights: None,
                morph_deltas: &[],
                num_morph_targets: 0,
            },
        )
        .unwrap();
        model.morph_deltas.push(1.0);
        assert_eq!(
            prepare_lod_model(model).unwrap_err(),
            "LOD morph payload does not match model shape",
        );
    }

    #[test]
    fn atlas_lookup_sorts_and_indexes_canonical_power_of_two_keys() {
        let lookup = prepare_lod_atlas_lookup([[2, 4, 4], [1, 1, 1], [2, 2, 4]])
            .unwrap();
        assert_eq!(lookup.keys, [[1, 1, 1], [2, 2, 4], [2, 4, 4]]);
        assert_eq!(lookup.max_lod, 4.0);
        assert_eq!(lookup.lut[0], 0);
        assert_eq!(lookup.lut[211], 1);
        assert_eq!(lookup.lut[221], 2);
        assert_eq!(lookup.lut[999], u8::MAX);
    }

    #[test]
    fn atlas_lookup_rejects_ambiguous_or_unrepresentable_keys() {
        for (keys, message) in [
            (vec![], "LOD atlas contains no canonical patches"),
            (
                vec![[1, 1, 1], [1, 1, 1]],
                "LOD atlas contains duplicate canonical patches",
            ),
            (vec![[2, 1, 2]], "LOD atlas key [2, 1, 2] is not canonical"),
            (
                vec![[1, 1, 3]],
                "LOD atlas edge 3 is outside the shader lookup",
            ),
            (
                vec![[1, 1, 1_024]],
                "LOD atlas edge 1024 is outside the shader lookup",
            ),
        ] {
            assert_eq!(prepare_lod_atlas_lookup(keys).unwrap_err(), message);
        }
    }

    #[test]
    fn composed_model_rejects_inconsistent_shared_vertex_positions() {
        let mut instances = two_face_instances();
        let second_face_vertex_one = instance_layout::STRIDE + instance_layout::offset::POSITIONS;
        instances[second_face_vertex_one + 1] = 2.0;
        let error = build_composed_lod_model(
            &instances,
            &[0, 0],
            4,
            1,
            LodAnimationSource {
                primary_vertices: 0,
                joint_indices: None,
                joint_weights: None,
                morph_deltas: &[],
                num_morph_targets: 0,
            },
        ).unwrap_err();
        assert_eq!(error, "composed LOD vertex 1 has inconsistent source positions");
    }

    #[test]
    fn composed_model_rejects_partial_animation_payloads() {
        let instances = two_face_instances();
        let joint_indices = [[0; 4]; 2];
        let joint_weights = [[0.0; 4]; 3];
        let error = build_composed_lod_model(
            &instances,
            &[0, 0],
            4,
            1,
            LodAnimationSource {
                primary_vertices: 3,
                joint_indices: Some(&joint_indices),
                joint_weights: Some(&joint_weights),
                morph_deltas: &[],
                num_morph_targets: 0,
            },
        ).unwrap_err();
        assert_eq!(error, "composed LOD primary skinning payload is incomplete");
    }

}
