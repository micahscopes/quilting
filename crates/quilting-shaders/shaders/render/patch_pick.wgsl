#import quilting::render::patch_vertex::{PatchPbrMaterial, PatchRenderFrame, PatchVertexOutput, evaluate_prepared_patch_vertex}
#import quilting::render::patch_pick_packet::{PatchPickOutput, PatchPickViewport, encode_patch_pick, remap_patch_pick_clip}
#import quilting::surface::patch_prepare::PreparedPatchRecord
#import quilting::compute::visibility_compaction_types::CompactedBatchRangeRecord

struct DrawBatchIndex {
    batch_index: u32,
    _padding_a: u32,
    _padding_b: u32,
    _padding_c: u32,
}

// The root layout deliberately matches the ordinary prepared-patch pipeline.
// Its bind group can therefore be reused without publishing parallel scene
// residency for picking. The PBR material table is retained in the layout even
// though this entry point does not read it.
@group(0) @binding(0) var<storage, read> frames: array<PatchRenderFrame>;
@group(0) @binding(1) var<storage, read> prepared_records: array<PreparedPatchRecord>;
@group(0) @binding(2) var<storage, read> compacted_sources: array<u32>;
@group(0) @binding(3) var<storage, read> compacted_ranges: array<CompactedBatchRangeRecord>;
@group(0) @binding(4) var<uniform> draw_batch: DrawBatchIndex;
@group(0) @binding(5) var<storage, read> _pbr_materials: array<PatchPbrMaterial>;

@group(1) @binding(0) var<uniform> pick_viewport: PatchPickViewport;

struct PatchVertexInput {
    @location(0) bary: vec3<f32>,
}

@vertex
fn render_patch_pick_vertex(
    input: PatchVertexInput,
    @builtin(instance_index) local_instance: u32,
) -> PatchVertexOutput {
    let range = compacted_ranges[draw_batch.batch_index];
    let compacted_index = range.compacted_first_instance + local_instance;
    let source_instance = compacted_sources[compacted_index];
    let output = evaluate_prepared_patch_vertex(
        frames[draw_batch.batch_index],
        input.bary,
        prepared_records[source_instance],
    );
    return remap_patch_pick_clip(output, pick_viewport);
}

@fragment
fn render_patch_pick(input: PatchVertexOutput) -> PatchPickOutput {
    if input.fade < 0.001 {
        discard;
    }
    return encode_patch_pick(input);
}
