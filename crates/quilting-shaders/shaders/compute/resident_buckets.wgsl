#import quilting::compute::resident_bucket_types::{ResidentBucketUniforms, ResidentAtlasDrawRecord, ResidentBucketRangeRecord, ResidentIndexedIndirectArguments, ResidentDrawDomainRecord, RESIDENT_BUCKET_WORKGROUP_SIZE, MAX_RESIDENT_BUCKETS, INVALID_RESIDENT_BUCKET, resident_geometry_bucket}

@group(0) @binding(0) var<uniform> dispatch: ResidentBucketUniforms;
@group(0) @binding(1) var<storage, read> resident_lod: array<u32>;
@group(0) @binding(2) var<storage, read> root_eligibility: array<u32>;
@group(0) @binding(3) var<storage, read> atlas_draws: array<ResidentAtlasDrawRecord>;
@group(0) @binding(4) var<storage, read_write> chunk_counts: array<u32>;
@group(0) @binding(5) var<storage, read_write> chunk_offsets: array<u32>;
@group(0) @binding(6) var<storage, read_write> bucket_counts: array<u32>;
@group(0) @binding(7) var<storage, read_write> bucket_ranges: array<ResidentBucketRangeRecord>;
@group(0) @binding(8) var<storage, read_write> indirect_arguments: array<ResidentIndexedIndirectArguments>;
@group(0) @binding(9) var<storage, read_write> compacted_faces: array<u32>;
@group(0) @binding(10) var<storage, read> face_domain_rows: array<u32>;
@group(0) @binding(11) var<storage, read> draw_domains: array<ResidentDrawDomainRecord>;

var<workgroup> histogram: array<atomic<u32>, 510>;
var<workgroup> local_buckets: array<u32, 64>;

// One workgroup owns one 64-face chunk. Atomics are confined to the local
// commutative histogram; stable ordering is established by later passes.
@compute @workgroup_size(64)
fn histogram_resident_geometry_buckets(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    let chunk = group_id.x;
    if chunk >= dispatch.counts.z {
        return;
    }
    for (
        var bucket = local_index;
        bucket < dispatch.counts.y;
        bucket += RESIDENT_BUCKET_WORKGROUP_SIZE
    ) {
        atomicStore(&histogram[bucket], 0u);
    }
    workgroupBarrier();

    let face = chunk * RESIDENT_BUCKET_WORKGROUP_SIZE + local_index;
    let eligible = face < dispatch.counts.x
        && (root_eligibility[face / 32u] & (1u << (face % 32u))) != 0u;
    if eligible {
        let domain = draw_domains[face_domain_rows[face]];
        let bucket = resident_geometry_bucket(resident_lod[face], dispatch.counts.w, domain);
        if bucket != INVALID_RESIDENT_BUCKET && bucket < MAX_RESIDENT_BUCKETS {
            atomicAdd(&histogram[bucket], 1u);
        }
    }
    workgroupBarrier();

    for (
        var bucket = local_index;
        bucket < dispatch.counts.y;
        bucket += RESIDENT_BUCKET_WORKGROUP_SIZE
    ) {
        chunk_counts[chunk * dispatch.counts.y + bucket] = atomicLoad(&histogram[bucket]);
    }
}

// Buckets scan independently across source-order chunks. The resulting local
// prefixes plus the global bucket prefix define a deterministic destination.
@compute @workgroup_size(64)
fn prefix_resident_geometry_chunks(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let bucket = invocation.x;
    if bucket >= dispatch.counts.y {
        return;
    }
    var prefix = 0u;
    for (var chunk = 0u; chunk < dispatch.counts.z; chunk++) {
        let index = chunk * dispatch.counts.y + bucket;
        chunk_offsets[index] = prefix;
        prefix += chunk_counts[index];
    }
    bucket_counts[bucket] = prefix;
}

// The atlas/parity domain is bounded by 255 * 2. One deterministic scan keeps
// the exact range and indirect tables portable without a counter readback.
@compute @workgroup_size(1)
fn scan_resident_geometry_buckets(@builtin(global_invocation_id) invocation: vec3<u32>) {
    if invocation.x != 0u {
        return;
    }
    var compacted_first = 0u;
    for (var bucket = 0u; bucket < dispatch.counts.y; bucket++) {
        let atlas_index = bucket / 2u;
        let parity = bucket % 2u;
        let count = bucket_counts[bucket];
        let draw = atlas_draws[atlas_index];
        bucket_ranges[bucket] = ResidentBucketRangeRecord(
            bucket,
            atlas_index,
            parity,
            compacted_first,
            count,
        );
        indirect_arguments[bucket] = ResidentIndexedIndirectArguments(
            draw.triangle_index_count,
            count,
            draw.triangle_first_index,
            0,
            0u,
        );
        compacted_first += count;
    }
}

// One workgroup revisits one source-order chunk. A lane's rank is the exact
// number of preceding lanes with the same bucket, avoiding nondeterministic
// global atomics while keeping work bounded by the fixed 64-face chunk.
@compute @workgroup_size(64)
fn scatter_resident_geometry_faces(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    let chunk = group_id.x;
    if chunk >= dispatch.counts.z {
        return;
    }
    let face = chunk * RESIDENT_BUCKET_WORKGROUP_SIZE + local_index;
    var bucket = INVALID_RESIDENT_BUCKET;
    let eligible = face < dispatch.counts.x
        && (root_eligibility[face / 32u] & (1u << (face % 32u))) != 0u;
    if eligible {
        let domain = draw_domains[face_domain_rows[face]];
        bucket = resident_geometry_bucket(resident_lod[face], dispatch.counts.w, domain);
    }
    local_buckets[local_index] = bucket;
    workgroupBarrier();

    if bucket == INVALID_RESIDENT_BUCKET || bucket >= dispatch.counts.y {
        return;
    }
    var local_rank = 0u;
    for (var lane = 0u; lane < local_index; lane++) {
        local_rank += u32(local_buckets[lane] == bucket);
    }
    let chunk_first = chunk_offsets[chunk * dispatch.counts.y + bucket];
    let destination = bucket_ranges[bucket].compacted_first_face + chunk_first + local_rank;
    compacted_faces[destination] = face;
}
