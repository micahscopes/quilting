#import quilting::render::patch_vertex::{PatchRenderFrame, PatchVertexOutput, evaluate_prepared_patch_vertex, shade_patch_lod, shade_patch_normals}
#import quilting::surface::patch_prepare::PreparedPatchRecord
#import quilting::compute::resident_bucket_types::{ResidentBucketRangeRecord, ResidentDrawDomainRecord}

struct DrawRootBucketIndex {
    bucket_index: u32,
    _padding_a: u32,
    _padding_b: u32,
    _padding_c: u32,
}

@group(0) @binding(0) var<storage, read> frames: array<PatchRenderFrame>;
@group(0) @binding(1) var<storage, read> prepared_records: array<PreparedPatchRecord>;
@group(0) @binding(2) var<storage, read> compacted_faces: array<u32>;
@group(0) @binding(3) var<storage, read> bucket_ranges: array<ResidentBucketRangeRecord>;
@group(0) @binding(4) var<uniform> draw_bucket: DrawRootBucketIndex;
@group(0) @binding(5) var<storage, read> face_domain_rows: array<u32>;
@group(0) @binding(6) var<storage, read> draw_domains: array<ResidentDrawDomainRecord>;

struct ResidentRootVertexInput {
    @location(0) bary: vec3<f32>,
}

@vertex
fn render_resident_root_vertex(
    input: ResidentRootVertexInput,
    @builtin(instance_index) local_instance: u32,
) -> PatchVertexOutput {
    let range = bucket_ranges[draw_bucket.bucket_index];
    let compacted_index = range.compacted_first_face + local_instance;
    let source_face = compacted_faces[compacted_index];
    let domain_row = face_domain_rows[source_face];
    let domain = draw_domains[domain_row];
    var output = evaluate_prepared_patch_vertex(
        frames[domain_row],
        input.bary,
        prepared_records[source_face],
    );
    // Bucketing already excludes disabled domains. Keep the vertex path
    // fail-closed if a future producer violates that invariant.
    if (domain.flags & 1u) == 0u {
        output.clip_pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        output.fade = 0.0;
    }
    return output;
}

@fragment
fn render_resident_root_normals(
    @builtin(front_facing) front_facing: bool,
    input: PatchVertexOutput,
) -> @location(0) vec4<f32> {
    return shade_patch_normals(front_facing, input);
}

@fragment
fn render_resident_root_lod(input: PatchVertexOutput) -> @location(0) vec4<f32> {
    return shade_patch_lod(input);
}
