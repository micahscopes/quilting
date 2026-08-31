#import quilting::render::patch_vertex::{PatchPbrMaterial, PatchRenderDomain, PatchRenderGlobal, PatchVertexOutput, evaluate_prepared_patch_vertex}
#import quilting::render::patch_pick_packet::{PatchPickOutput, PatchPickViewport, encode_patch_pick, remap_patch_pick_clip}
#import quilting::surface::patch_prepare::PreparedPatchRecord
#import quilting::compute::resident_bucket_types::{ResidentBucketRangeRecord, ResidentDrawDomainRecord}

struct DrawRootBucketIndex {
    bucket_index: u32,
    _padding_a: u32,
    _padding_b: u32,
    _padding_c: u32,
}

// Group zero deliberately matches the resident-root render family, including
// its PBR-material slot. The query therefore consumes the current source-face
// buckets without publishing a second root scene.
@group(0) @binding(0) var<storage, read> global_frame: array<PatchRenderGlobal>;
@group(0) @binding(1) var<storage, read> render_domains: array<PatchRenderDomain>;
@group(0) @binding(2) var<storage, read> prepared_records: array<PreparedPatchRecord>;
@group(0) @binding(3) var<storage, read> compacted_faces: array<u32>;
@group(0) @binding(4) var<storage, read> bucket_ranges: array<ResidentBucketRangeRecord>;
@group(0) @binding(5) var<uniform> draw_bucket: DrawRootBucketIndex;
@group(0) @binding(6) var<storage, read> face_domain_rows: array<u32>;
@group(0) @binding(7) var<storage, read> draw_domains: array<ResidentDrawDomainRecord>;
@group(0) @binding(8) var<storage, read> _pbr_materials: array<PatchPbrMaterial>;

@group(1) @binding(0) var<uniform> pick_viewport: PatchPickViewport;

struct ResidentRootVertexInput {
    @location(0) bary: vec3<f32>,
}

@vertex
fn render_resident_root_pick_vertex(
    input: ResidentRootVertexInput,
    @builtin(instance_index) local_instance: u32,
) -> PatchVertexOutput {
    let range = bucket_ranges[draw_bucket.bucket_index];
    let compacted_index = range.compacted_first_face + local_instance;
    let source_face = compacted_faces[compacted_index];
    let domain_row = face_domain_rows[source_face];
    let domain = draw_domains[domain_row];
    var output = evaluate_prepared_patch_vertex(
        global_frame[0],
        render_domains[domain_row],
        input.bary,
        prepared_records[source_face],
    );
    if (domain.flags & 1u) == 0u {
        output.clip_pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        output.fade = 0.0;
    }
    return remap_patch_pick_clip(output, pick_viewport);
}

@fragment
fn render_resident_root_pick(input: PatchVertexOutput) -> PatchPickOutput {
    if input.fade < 0.001 {
        discard;
    }
    return encode_patch_pick(input);
}
