#import quilting::compute::patch_prepare_types::{PatchTopologyRecord}

struct LodVisibilityDispatch {
    // x = source patch instances, y = source faces.
    counts: vec4<u32>,
}

@group(0) @binding(0) var<uniform> dispatch: LodVisibilityDispatch;
@group(0) @binding(1) var<storage, read> topology: array<PatchTopologyRecord>;
@group(0) @binding(2) var<storage, read> resident_lod: array<u32>;
@group(0) @binding(3) var<storage, read_write> source_visibility: array<u32>;

// Project the resident classifier visibility bit through current patch order.
// Adaptive leaves of one source face intentionally share that face's result.
@compute @workgroup_size(64)
fn expand_resident_lod_visibility(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let source_index = invocation.x;
    if source_index >= dispatch.counts.x {
        return;
    }
    let encoded_face = topology[source_index].face_info.x;
    let face = u32(encoded_face);
    if face >= dispatch.counts.y {
        source_visibility[source_index] = 0u;
        return;
    }
    source_visibility[source_index] = (resident_lod[face] >> 15u) & 1u;
}
