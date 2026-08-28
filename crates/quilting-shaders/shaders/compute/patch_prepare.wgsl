#import quilting::surface::patch_prepare::{PreparedPatchRecord, PosedPatchControls, prepare_patch_record}
#import quilting::compute::patch_prepare_types::{PatchTopologyRecord, PatchSubjectState, PatchPrepareDispatch}
#import quilting::compute::pose::{animated_position, animated_normal}

@group(0) @binding(0) var<uniform> dispatch: PatchPrepareDispatch;
@group(0) @binding(1) var<storage, read> topology_records: array<PatchTopologyRecord>;
@group(0) @binding(2) var<storage, read> source_face_records: array<PreparedPatchRecord>;
// Bindings 3..6 are the shared dynamic-pose residency imported above.
@group(0) @binding(7) var<storage, read> subject_states: array<PatchSubjectState>;
@group(0) @binding(8) var<storage, read_write> prepared_records: array<PreparedPatchRecord>;

fn source_vertex_index(encoded: f32) -> u32 {
    return u32(max(round(encoded), 0.0));
}

@compute @workgroup_size(64)
fn prepare_patch_instances(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let patch_index = invocation.x;
    if patch_index >= dispatch.counts.x {
        return;
    }
    let topology = topology_records[patch_index];
    let face_index = u32(max(round(topology.face_info.x), 0.0));
    let source = source_face_records[face_index];
    let subject = subject_states[topology.subject_index];
    let vertex_a = source_vertex_index(source.record_position_a.x);
    let vertex_b = source_vertex_index(source.record_position_b.x);
    let vertex_c = source_vertex_index(source.record_position_c.x);
    let posed_a = (subject.model * vec4<f32>(animated_position(
        source.record_position_a.yzw,
        vertex_a,
        dispatch.counts.y,
        dispatch.counts.z,
        dispatch.counts.w,
    ), 1.0)).xyz;
    let posed_b = (subject.model * vec4<f32>(animated_position(
        source.record_position_b.yzw,
        vertex_b,
        dispatch.counts.y,
        dispatch.counts.z,
        dispatch.counts.w,
    ), 1.0)).xyz;
    let posed_c = (subject.model * vec4<f32>(animated_position(
        source.record_position_c.yzw,
        vertex_c,
        dispatch.counts.y,
        dispatch.counts.z,
        dispatch.counts.w,
    ), 1.0)).xyz;

    // A prepared record is fully posed: do normal skinning and the affine
    // normal transform once per patch control instead of once per generated
    // tessellation vertex. Root controls retain their source vertex IDs for
    // picking and diagnostics, but their normals are no longer source-space.
    let normal_a = (subject.normal_model * vec4<f32>(animated_normal(
        source.record_normal_a.xyz, vertex_a, dispatch.counts.z,
    ), 0.0)).xyz;
    let normal_b = (subject.normal_model * vec4<f32>(animated_normal(
        source.record_normal_b.xyz, vertex_b, dispatch.counts.z,
    ), 0.0)).xyz;
    let normal_c = (subject.normal_model * vec4<f32>(animated_normal(
        source.record_normal_c.xyz, vertex_c, dispatch.counts.z,
    ), 0.0)).xyz;

    prepared_records[patch_index] = prepare_patch_record(
        source,
        topology.lod_info,
        topology.face_info,
        topology.leaf_meta,
        PosedPatchControls(
            posed_a, posed_b, posed_c,
            normal_a, normal_b, normal_c,
        ),
    );
}
