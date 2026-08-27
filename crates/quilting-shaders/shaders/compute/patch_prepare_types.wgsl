#define_import_path quilting::compute::patch_prepare_types

// WebGPU storage form of quilting_core::instance_layout's ten-float topology
// record. The first 40 bytes retain the canonical WebGL2 order; one dense
// subject row and one padding word extend the storage stride to 48 bytes.
struct PatchTopologyRecord {
    lod_info: vec4<f32>,
    face_info: vec4<f32>,
    leaf_meta: vec2<f32>,
    subject_index: u32,
    _padding: u32,
}

// Ordinary affine state needed while turning source controls into the exact
// current-pose prepared record. Conformal state is intentionally absent: it is
// evaluated later by the surface render stage and does not invalidate pose
// preparation.
struct PatchSubjectState {
    model: mat4x4<f32>,
    normal_model: mat4x4<f32>,
}

struct PatchPrepareDispatch {
    // x = patch instances, y = source vertices, z = joints, w = morph targets.
    counts: vec4<u32>,
}
