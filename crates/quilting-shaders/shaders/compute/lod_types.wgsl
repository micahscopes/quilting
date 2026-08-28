#define_import_path quilting::compute::lod_types

// Backend-neutral storage ABI for the two-pass LOD classifier. Every record
// uses an explicit 16-byte stride except the final packed word. This avoids
// texture tiling and float-encoded indices in the future WebGPU path while
// retaining the exact semantics of the current WebGL2 classifier.

struct LodFaceRecord {
    vertex_indices: vec3<u32>,
    subject_index: u32,
}

struct LodSkinningRecord {
    joint_indices: vec4<u32>,
    joint_weights: vec4<f32>,
}

struct LodAdjacencyRecord {
    neighbor_face: i32,
    neighbor_edge: u32,
    _padding: vec2<u32>,
}

struct LodPass1Record {
    exponents: vec3<f32>,
    // Zero means culled. Visible records store 1 + an integer priority in
    // [0, 255], exactly matching the WebGL2 RGBA32F intermediate payload.
    visibility_priority: f32,
}

struct LodSubjectState {
    mob_a: vec4<f32>,
    mob_b: vec4<f32>,
    mob_c: vec4<f32>,
    mob_d: vec4<f32>,
    model: mat4x4<f32>,
    pole: vec4<f32>,
    // x = Mobius similarity power, y = |c|^2, z = finite-pole flag,
    // w = valid subject-state flag.
    conformal: vec4<f32>,
}

struct LodDispatchUniforms {
    baseline_mob_a: vec4<f32>,
    baseline_mob_b: vec4<f32>,
    baseline_mob_c: vec4<f32>,
    baseline_mob_d: vec4<f32>,
    baseline_model: mat4x4<f32>,
    baseline_pole: vec4<f32>,
    // x = Mobius similarity power, y = |c|^2, z = finite-pole flag,
    // w = use subject-state table.
    baseline_conformal: vec4<f32>,
    view_projection: mat4x4<f32>,
    // x = density, y = mesh radius, z = pixel/subtriangle floor,
    // w = atlas maximum LOD.
    density_metrics: vec4<f32>,
    // x = viewport width, y = viewport height. z/w are reserved.
    viewport: vec4<f32>,
    // x = faces, y = vertices, z = joints, w = morph targets.
    counts: vec4<u32>,
}

fn pack_lod_classification(
    exponents: vec3<u32>,
    permutation: u32,
    visible: bool,
    atlas_index: u32,
    adaptive_priority: u32,
) -> u32 {
    var packed = exponents.x
        | (exponents.y << 4u)
        | (exponents.z << 8u)
        | (permutation << 12u)
        | (adaptive_priority << 24u);
    if visible {
        packed = packed | (1u << 15u) | (atlas_index << 16u);
    }
    return packed;
}

fn unpack_lod_permutation(packed: u32) -> u32 {
    return (packed >> 12u) & 7u;
}

fn unpack_lod_canonical_counts(packed: u32) -> vec3<u32> {
    return vec3<u32>(
        1u << (packed & 15u),
        1u << ((packed >> 4u) & 15u),
        1u << ((packed >> 8u) & 15u),
    );
}

// Recover the authored face-edge order from the sorted atlas order. This is
// the WGSL equivalent of ResidentLod::edge_lods and is shared by preparation,
// diagnostic corner maxima, and future material-aware draw construction.
fn unpack_lod_edge_counts(packed: u32) -> vec3<u32> {
    let canonical = unpack_lod_canonical_counts(packed);
    switch unpack_lod_permutation(packed) {
        case 0u: { return canonical.xyz; }
        case 1u: { return canonical.xzy; }
        case 2u: { return canonical.yxz; }
        case 3u: { return canonical.yzx; }
        case 4u: { return canonical.zxy; }
        case 5u: { return canonical.zyx; }
        default: { return canonical.xyz; }
    }
}
