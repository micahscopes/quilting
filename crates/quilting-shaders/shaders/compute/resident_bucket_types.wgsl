#define_import_path quilting::compute::resident_bucket_types

const RESIDENT_BUCKET_WORKGROUP_SIZE: u32 = 64u;
const MAX_RESIDENT_BUCKETS: u32 = 510u;
const INVALID_RESIDENT_BUCKET: u32 = 0xffffffffu;

struct ResidentBucketUniforms {
    // x = source faces, y = atlas/parity buckets, z = 64-face chunks,
    // w = sorted atlas entries.
    counts: vec4<u32>,
}

struct ResidentAtlasDrawRecord {
    triangle_first_index: u32,
    triangle_index_count: u32,
    line_first_index: u32,
    line_index_count: u32,
}

struct ResidentBucketRangeRecord {
    bucket_index: u32,
    atlas_index: u32,
    parity_bucket: u32,
    compacted_first_face: u32,
    compacted_face_count: u32,
}

struct ResidentIndexedIndirectArguments {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

fn resident_geometry_bucket(packed: u32, atlas_count: u32) -> u32 {
    if (packed & (1u << 15u)) == 0u {
        return INVALID_RESIDENT_BUCKET;
    }
    let atlas_index = (packed >> 16u) & 255u;
    let permutation = (packed >> 12u) & 7u;
    if atlas_index >= atlas_count || permutation >= 6u {
        return INVALID_RESIDENT_BUCKET;
    }
    let odd = permutation == 1u || permutation == 2u || permutation == 5u;
    return atlas_index * 2u + select(0u, 1u, odd);
}
