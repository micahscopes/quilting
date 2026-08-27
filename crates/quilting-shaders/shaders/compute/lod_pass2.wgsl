#import quilting::compute::lod_types::{LodAdjacencyRecord, LodPass1Record, LodDispatchUniforms, pack_lod_classification}

// WebGPU form of the WebGL2 transform-feedback coherence pass. One invocation
// reconciles a face with its visible neighbors, canonicalizes the S3 edge
// ordering, looks up the resident atlas entry, and writes the same packed u32.

@group(0) @binding(0) var<uniform> dispatch: LodDispatchUniforms;
@group(0) @binding(1) var<storage, read> pass1_records: array<LodPass1Record>;
@group(0) @binding(2) var<storage, read> adjacency: array<LodAdjacencyRecord>;
@group(0) @binding(3) var<storage, read> atlas_lut: array<u32>;
@group(0) @binding(4) var<storage, read_write> packed_records: array<u32>;

struct CanonicalLod {
    exponents: vec3<u32>,
    permutation: u32,
}

fn canonicalize_lod(exponents: vec3<u32>) -> CanonicalLod {
    let a = exponents.x;
    let b = exponents.y;
    let c = exponents.z;
    if a <= b && b <= c {
        return CanonicalLod(vec3<u32>(a, b, c), 0u);
    } else if a <= c && c <= b {
        return CanonicalLod(vec3<u32>(a, c, b), 1u);
    } else if b <= a && a <= c {
        return CanonicalLod(vec3<u32>(b, a, c), 2u);
    } else if b <= c && c <= a {
        return CanonicalLod(vec3<u32>(b, c, a), 4u);
    } else if c <= a && a <= b {
        return CanonicalLod(vec3<u32>(c, a, b), 3u);
    }
    return CanonicalLod(vec3<u32>(c, b, a), 5u);
}

@compute @workgroup_size(64)
fn classify_lod_pass2(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let face_index = invocation.x;
    if face_index >= dispatch.counts.x {
        return;
    }
    let face = pass1_records[face_index];
    if face.visibility_priority < 0.5 {
        packed_records[face_index] = pack_lod_classification(
            vec3<u32>(round(face.exponents)),
            0u,
            false,
            0u,
            0u,
        );
        return;
    }

    var reconciled = face.exponents;
    for (var edge = 0u; edge < 3u; edge++) {
        let neighbor = adjacency[face_index * 3u + edge];
        if neighbor.neighbor_face < 0 {
            continue;
        }
        let neighbor_record = pass1_records[u32(neighbor.neighbor_face)];
        if neighbor_record.visibility_priority < 0.5 {
            continue;
        }
        reconciled[edge] = max(
            reconciled[edge],
            neighbor_record.exponents[neighbor.neighbor_edge],
        );
    }

    let canonical = canonicalize_lod(vec3<u32>(round(reconciled)));
    let key = canonical.exponents.x
        + canonical.exponents.y * 10u
        + canonical.exponents.z * 100u;
    let adaptive_priority = u32(clamp(round(face.visibility_priority - 1.0), 0.0, 255.0));
    packed_records[face_index] = pack_lod_classification(
        canonical.exponents,
        canonical.permutation,
        true,
        atlas_lut[key],
        adaptive_priority,
    );
}
