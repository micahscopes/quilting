#import quilting::compute::patch_prepare_types::{PatchTopologyRecord}

struct FaceVisibilityDispatch {
    // x = source patch instances, y = source faces, z = packed bit words.
    counts: vec4<u32>,
}

@group(0) @binding(0) var<uniform> dispatch: FaceVisibilityDispatch;
@group(0) @binding(1) var<storage, read> topology: array<PatchTopologyRecord>;
@group(0) @binding(2) var<storage, read> face_visibility_bits: array<u32>;
@group(0) @binding(3) var<storage, read_write> source_visibility: array<u32>;

// Expand one compact face bit into every current flattened patch instance.
// Topology can reorder instances every LOD epoch without invalidating the
// face-indexed input buffer.
@compute @workgroup_size(64)
fn expand_face_visibility(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let source_index = invocation.x;
    if source_index >= dispatch.counts.x {
        return;
    }
    let encoded_face = topology[source_index].face_info.x;
    // CPU packing validates exact non-negative integral face IDs before this
    // buffer becomes resident.
    let face = u32(encoded_face);
    if face >= dispatch.counts.y {
        source_visibility[source_index] = 0u;
        return;
    }
    let word = face >> 5u;
    if word >= dispatch.counts.z {
        source_visibility[source_index] = 0u;
        return;
    }
    source_visibility[source_index] =
        (face_visibility_bits[word] >> (face & 31u)) & 1u;
}
