#import quilting::compute::lod_types::{LodAdjacencyRecord, LodDispatchUniforms, pack_lod_classification}

// Device-resident closure over the classifier's packed face requests. The
// exponent lattice is [0, 9]. A face-to-face propagation step must cross one
// within-face grading edge and therefore loses at least one exponent at 2:1
// (two at 4:1); ten Jacobi passes cover the complete nonzero influence radius.

@group(0) @binding(0) var<uniform> dispatch: LodDispatchUniforms;
@group(0) @binding(1) var<storage, read> requested_records: array<u32>;
@group(0) @binding(2) var<storage, read> adjacency: array<LodAdjacencyRecord>;
@group(0) @binding(3) var<storage, read> atlas_lut: array<u32>;
@group(0) @binding(4) var<storage, read_write> resident_input: array<u32>;
@group(0) @binding(5) var<storage, read_write> resident_output: array<u32>;
@group(0) @binding(6) var<storage, read_write> resident_records: array<u32>;

struct CanonicalResidentLod {
    exponents: vec3<u32>,
    permutation: u32,
}

fn unpack_local_exponents(packed: u32) -> vec3<u32> {
    let canonical = vec3<u32>(packed & 15u, (packed >> 4u) & 15u, (packed >> 8u) & 15u);
    let permutation = (packed >> 12u) & 7u;
    switch permutation {
        case 0u: { return canonical.xyz; }
        case 1u: { return canonical.xzy; }
        case 2u: { return canonical.yxz; }
        case 3u: { return canonical.yzx; }
        case 4u: { return canonical.zxy; }
        case 5u: { return canonical.zyx; }
        default: { return canonical.xyz; }
    }
}

fn pack_local_exponents(exponents: vec3<u32>) -> u32 {
    return exponents.x | (exponents.y << 4u) | (exponents.z << 8u);
}

fn unpack_resident_exponents(packed: u32) -> vec3<u32> {
    return vec3<u32>(packed & 15u, (packed >> 4u) & 15u, (packed >> 8u) & 15u);
}

fn canonicalize_resident_lod(exponents: vec3<u32>) -> CanonicalResidentLod {
    let a = exponents.x;
    let b = exponents.y;
    let c = exponents.z;
    if a <= b && b <= c {
        return CanonicalResidentLod(vec3<u32>(a, b, c), 0u);
    } else if a <= c && c <= b {
        return CanonicalResidentLod(vec3<u32>(a, c, b), 1u);
    } else if b <= a && a <= c {
        return CanonicalResidentLod(vec3<u32>(b, a, c), 2u);
    } else if b <= c && c <= a {
        return CanonicalResidentLod(vec3<u32>(b, c, a), 4u);
    } else if c <= a && a <= b {
        return CanonicalResidentLod(vec3<u32>(c, a, b), 3u);
    }
    return CanonicalResidentLod(vec3<u32>(c, b, a), 5u);
}

fn reconcile_resident_face(face_index: u32, grading_exponent: u32) {
    var reconciled = unpack_resident_exponents(resident_input[face_index]);
    for (var edge = 0u; edge < 3u; edge++) {
        let neighbor = adjacency[face_index * 3u + edge];
        if neighbor.neighbor_face < 0 {
            continue;
        }
        let neighbor_exponents = unpack_resident_exponents(
            resident_input[u32(neighbor.neighbor_face)],
        );
        reconciled[edge] = max(reconciled[edge], neighbor_exponents[neighbor.neighbor_edge]);
    }

    let maximum = max(reconciled.x, max(reconciled.y, reconciled.z));
    let minimum = maximum - min(maximum, grading_exponent);
    reconciled = max(reconciled, vec3<u32>(minimum));
    resident_output[face_index] = pack_local_exponents(reconciled);
}

@compute @workgroup_size(64)
fn seed_resident_lod(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let face_index = invocation.x;
    if face_index >= dispatch.counts.x {
        return;
    }
    resident_input[face_index] = pack_local_exponents(
        unpack_local_exponents(requested_records[face_index]),
    );
}

@compute @workgroup_size(64)
fn reconcile_resident_lod_2_to_1(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let face_index = invocation.x;
    if face_index >= dispatch.counts.x {
        return;
    }
    reconcile_resident_face(face_index, 1u);
}

@compute @workgroup_size(64)
fn reconcile_resident_lod_4_to_1(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let face_index = invocation.x;
    if face_index >= dispatch.counts.x {
        return;
    }
    reconcile_resident_face(face_index, 2u);
}

@compute @workgroup_size(64)
fn pack_resident_lod(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let face_index = invocation.x;
    if face_index >= dispatch.counts.x {
        return;
    }
    let requested = requested_records[face_index];
    let visible = (requested & (1u << 15u)) != 0u;
    let priority = requested >> 24u;
    let canonical = canonicalize_resident_lod(
        unpack_resident_exponents(resident_input[face_index]),
    );
    let key = canonical.exponents.x
        + canonical.exponents.y * 10u
        + canonical.exponents.z * 100u;
    resident_records[face_index] = pack_lod_classification(
        canonical.exponents,
        canonical.permutation,
        visible,
        atlas_lut[key],
        priority,
    );
}
