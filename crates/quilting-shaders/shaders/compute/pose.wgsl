#define_import_path quilting::compute::pose

#import quilting::compute::lod_types::LodSkinningRecord

// Shared dynamic-pose bindings. Keeping these at the established LOD binding
// numbers lets both the LOD classifier and patch-preparation compute pass use
// one retained animation residency instead of duplicating joint/morph data.
@group(0) @binding(3) var<storage, read> pose_skinning: array<LodSkinningRecord>;
@group(0) @binding(4) var<storage, read> pose_joint_matrices: array<mat4x4<f32>>;
@group(0) @binding(5) var<storage, read> pose_morph_deltas: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> pose_morph_weights: array<f32>;

fn animated_position(
    source_position: vec3<f32>,
    vertex: u32,
    num_vertices: u32,
    num_joints: u32,
    num_morph_targets: u32,
) -> vec3<f32> {
    var position = source_position;
    for (var morph_target = 0u; morph_target < 64u; morph_target++) {
        if morph_target >= num_morph_targets {
            break;
        }
        let weight = pose_morph_weights[morph_target];
        if abs(weight) >= 1e-6 {
            position += weight
                * pose_morph_deltas[morph_target * num_vertices + vertex].xyz;
        }
    }

    if num_joints == 0u {
        return position;
    }
    let influences = pose_skinning[vertex];
    var skinned = vec3<f32>(0.0);
    var applied_weight = 0.0;
    let homogeneous = vec4<f32>(position, 1.0);
    for (var influence = 0u; influence < 4u; influence++) {
        let weight = influences.joint_weights[influence];
        let joint = influences.joint_indices[influence];
        if weight >= 1e-6 && joint < num_joints {
            applied_weight += weight;
            skinned += weight * (pose_joint_matrices[joint] * homogeneous).xyz;
        }
    }
    if applied_weight <= 1e-6 {
        return position;
    }
    return skinned;
}

// Match the WebGL vertex path: blend joint cofactors, then normalize once.
// The cofactor is det(M) * inverse_transpose(M), so it remains defined for
// non-uniform scaling without performing a matrix inverse in the shader.
fn animated_normal(
    source_normal: vec3<f32>,
    vertex: u32,
    num_joints: u32,
) -> vec3<f32> {
    if num_joints == 0u {
        return source_normal;
    }
    let influences = pose_skinning[vertex];
    var skinned = vec3<f32>(0.0);
    for (var influence = 0u; influence < 4u; influence++) {
        let weight = influences.joint_weights[influence];
        let joint = influences.joint_indices[influence];
        if weight >= 1e-6 && joint < num_joints {
            let matrix = pose_joint_matrices[joint];
            let linear = mat3x3<f32>(matrix[0].xyz, matrix[1].xyz, matrix[2].xyz);
            let cofactor = mat3x3<f32>(
                cross(linear[1], linear[2]),
                cross(linear[2], linear[0]),
                cross(linear[0], linear[1]),
            );
            skinned += weight * (cofactor * source_normal);
        }
    }
    let magnitude = length(skinned);
    if magnitude > 1e-8 {
        return skinned / magnitude;
    }
    return source_normal;
}
