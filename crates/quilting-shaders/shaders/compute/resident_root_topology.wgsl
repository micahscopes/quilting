#import quilting::compute::lod_types::{LodFaceRecord, unpack_lod_edge_counts, unpack_lod_permutation}
#import quilting::compute::patch_prepare_types::PatchTopologyRecord
#import quilting::compute::resident_root_topology_types::ResidentRootTopologyDispatch

@group(0) @binding(0) var<uniform> dispatch: ResidentRootTopologyDispatch;
@group(0) @binding(1) var<storage, read> resident_lod: array<u32>;
@group(0) @binding(2) var<storage, read> faces: array<LodFaceRecord>;
@group(0) @binding(3) var<storage, read> face_subject_rows: array<u32>;
@group(0) @binding(4) var<storage, read_write> vertex_lod_max: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> root_topology: array<PatchTopologyRecord>;

@compute @workgroup_size(64)
fn clear_resident_root_vertex_lods(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let vertex = invocation.x;
    if vertex >= dispatch.counts.y {
        return;
    }
    atomicStore(&vertex_lod_max[vertex], 1u);
}

// Source-edge ordering follows HalfEdgeMesh: edge a is opposite corner zero,
// edge b opposite corner one, and edge c opposite corner two. Each compact
// vertex retains the greatest incident edge density for a continuous LOD-color
// field, including the drawable standby topology of currently invisible faces.
@compute @workgroup_size(64)
fn accumulate_resident_root_vertex_lods(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let face = invocation.x;
    if face >= dispatch.counts.x {
        return;
    }
    let face_record = faces[face];
    let edge_lods = unpack_lod_edge_counts(resident_lod[face]);
    atomicMax(&vertex_lod_max[face_record.vertex_indices.x], max(edge_lods.y, edge_lods.z));
    atomicMax(&vertex_lod_max[face_record.vertex_indices.y], max(edge_lods.x, edge_lods.z));
    atomicMax(&vertex_lod_max[face_record.vertex_indices.z], max(edge_lods.x, edge_lods.y));
}

@compute @workgroup_size(64)
fn emit_resident_root_topology(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let face = invocation.x;
    if face >= dispatch.counts.x {
        return;
    }
    let packed = resident_lod[face];
    let face_record = faces[face];
    let edge_lods = unpack_lod_edge_counts(packed);
    root_topology[face] = PatchTopologyRecord(
        vec4<f32>(vec3<f32>(edge_lods), f32(unpack_lod_permutation(packed))),
        vec4<f32>(
            f32(face),
            f32(atomicLoad(&vertex_lod_max[face_record.vertex_indices.x])),
            f32(atomicLoad(&vertex_lod_max[face_record.vertex_indices.y])),
            f32(atomicLoad(&vertex_lod_max[face_record.vertex_indices.z])),
        ),
        vec2<f32>(0.0),
        face_subject_rows[face],
        0u,
    );
}
