//! GPU compute via transform feedback (WebGL2 GPGPU).
//!
//! Two-pass pipeline — one "vertex" per face:
//!   Pass 1: animated positions → conservative image bound + raw LOD exponents
//!   Pass 2: edge coherence (via adjacency texture) + canonical sort + atlas LUT
//!
//! Pass 1 renders directly to a texture for pass 2 to read.
//! Final output: 6 floats per face (canon_a, canon_b, canon_c, perm_index, parity, atlas_index),
//! directly consumable by group_into_batches.

use glow::HasContext;
use quilting_core::instance_layout;
use quilting_core::quaternion::{Mobius, Quat};
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

/// Pass 1 FBO payload: raw LOD exponents plus conservative visibility.
pub const FLOATS_PER_FACE_PASS1: usize = 4;

/// Pass 2 final output: (canon_a, canon_b, canon_c, perm_index, parity, atlas_index) = 6 floats per face.
pub const FLOATS_PER_FACE_OUTPUT: usize = 6;

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

/// GPU-side copy of a completed LOD transform-feedback output. The worker can
/// fence after staging one or more runs, poll without blocking, then read only
/// once the shared fence is signaled.
pub struct StagedLodReadback {
    buffer: ReadbackBuffer,
    num_faces: usize,
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
/// Pass 2: Transform feedback (edge coherence + canonicalize → readback buffer)
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
    output_buf2: glow::Buffer,      // TF output: 6 floats per face (final)
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
    readback_vectors: Vec<Vec<f32>>,
    readback_buffer_creations: u64,
    readback_buffer_reallocations: u64,
    readback_vector_creations: u64,
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
                build_tf_program(gl, LOD_COHERENCE_VS, DUMMY_FS,
                    &["out_canon_a", "out_canon_b", "out_canon_c", "out_perm_index", "out_parity", "out_atlas_index"],
                    "LOD pass 2")?,
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
                (max_faces * FLOATS_PER_FACE_OUTPUT * 4) as i32,
                glow::DYNAMIC_READ);

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
                readback_buffer_creations: 0,
                readback_buffer_reallocations: 0,
                readback_vector_creations: 0,
            })
        }
    }

    pub fn has_pass1_texture(&self) -> bool { self.pass1_texture.is_some() }
    pub fn has_adjacency_texture(&self) -> bool { self.adjacency_texture.is_some() }

    pub fn readback_pool_stats(&self) -> (usize, usize, u64, u64, u64) {
        (
            self.readback_buffers.len(),
            self.readback_vectors.len(),
            self.readback_buffer_creations,
            self.readback_buffer_reallocations,
            self.readback_vector_creations,
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
            .checked_mul(FLOATS_PER_FACE_OUTPUT)
            .and_then(|size| size.checked_mul(std::mem::size_of::<f32>()))
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

    /// Read and destroy a staging buffer after its fence has signaled.
    pub fn finish_staged_readback(
        &mut self,
        gl: &glow::Context,
        staged: StagedLodReadback,
    ) -> Vec<f32> {
        let mut result = match self.readback_vectors.pop() {
            Some(result) => result,
            None => {
                self.readback_vector_creations += 1;
                Vec::new()
            }
        };
        result.resize(staged.num_faces * FLOATS_PER_FACE_OUTPUT, 0.0);
        unsafe {
            gl.bind_buffer(glow::COPY_READ_BUFFER, Some(staged.buffer.handle));
            gl.get_buffer_sub_data(
                glow::COPY_READ_BUFFER,
                0,
                bytemuck_cast_slice_mut(&mut result),
            );
            gl.bind_buffer(glow::COPY_READ_BUFFER, None);
        }
        self.readback_buffers.push(staged.buffer);
        result
    }

    /// Return a staged result that will not be consumed to the retained pool.
    pub fn discard_staged_readback(&mut self, _gl: &glow::Context, staged: StagedLodReadback) {
        self.readback_buffers.push(staged.buffer);
    }

    /// Return a completed CPU payload after its JS transfer has been copied.
    pub fn recycle_readback_vector(&mut self, mut result: Vec<f32>) {
        result.clear();
        self.readback_vectors.push(result);
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

fn bytemuck_cast_slice_mut(data: &mut [f32]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, data.len() * 4) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quilting_core::instance_layout::InstanceWriter;
    use std::cell::RefCell;

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
        assert!(LOD_COHERENCE_VS.contains("out_canon_a = exp2(face.x)"));
        assert!(LOD_COHERENCE_VS.contains("out_atlas_index = -1.0"));
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
