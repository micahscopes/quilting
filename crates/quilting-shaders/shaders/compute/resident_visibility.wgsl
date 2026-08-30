#import quilting::surface::patch_prepare::PreparedPatchRecord
#import quilting::surface::patch_render::PatchRenderTransform
#import quilting::surface::patch_visibility::prepared_patch_outside_frustum
#import quilting::render::patch_vertex::PatchRenderFrame
#import quilting::compute::resident_bucket_types::{ResidentBucketUniforms, ResidentDrawDomainRecord}

@group(0) @binding(0) var<uniform> dispatch: ResidentBucketUniforms;
@group(0) @binding(1) var<storage, read> frames: array<PatchRenderFrame>;
@group(0) @binding(2) var<storage, read> prepared_records: array<PreparedPatchRecord>;
@group(0) @binding(3) var<storage, read> root_eligibility: array<u32>;
@group(0) @binding(4) var<storage, read> face_domain_rows: array<u32>;
@group(0) @binding(5) var<storage, read> draw_domains: array<ResidentDrawDomainRecord>;
@group(0) @binding(6) var<storage, read_write> root_visibility: array<u32>;

fn visibility_transform(frame: PatchRenderFrame) -> PatchRenderTransform {
    return PatchRenderTransform(
        frame.mvp,
        frame.mv,
        frame.modes.x,
        frame.mob_a,
        frame.mob_b,
        frame.mob_c,
        frame.mob_d,
        frame.camera_pos,
    );
}

// One invocation owns one 32-face word, avoiding atomics and a separate clear
// pass. The result remains on-device and is consumed directly by resident
// bucket compaction in the same command encoder.
@compute @workgroup_size(64)
fn classify_resident_root_visibility(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let word = invocation.x;
    let word_count = (dispatch.counts.x + 31u) / 32u;
    if word >= word_count {
        return;
    }
    var visible_bits = 0u;
    for (var bit = 0u; bit < 32u; bit++) {
        let face = word * 32u + bit;
        if face >= dispatch.counts.x {
            break;
        }
        if (root_eligibility[word] & (1u << bit)) == 0u {
            continue;
        }
        let domain_row = face_domain_rows[face];
        let domain = draw_domains[domain_row];
        if (domain.flags & 1u) == 0u {
            continue;
        }
        let frame = frames[domain_row];
        let record = prepared_records[face];
        let outside = prepared_patch_outside_frustum(
            visibility_transform(frame),
            record.record_position_a.yzw,
            record.record_position_b.yzw,
            record.record_position_c.yzw,
            record.record_weight_a,
            record.record_weight_b,
            record.record_weight_c,
        );
        if !outside {
            visible_bits |= 1u << bit;
        }
    }
    root_visibility[word] = visible_bits;
}
