#import quilting::render::patch_vertex::{PatchPbrMaterial, PatchRenderFrame, PatchVertexOutput, evaluate_prepared_patch_vertex}
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

// x/y = full viewport extent; z/w = queried top-left pixel coordinate.
struct PatchPickViewport {
    extent_and_pixel: vec4<f32>,
}

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
    var output = evaluate_prepared_patch_vertex(
        frames[draw_batch.batch_index],
        input.bary,
        prepared_records[source_instance],
    );

    // Remap the queried full-viewport pixel to a one-pixel attachment. This
    // preserves the incumbent raster/depth result without retaining an entire
    // second full-resolution ID framebuffer.
    let extent = pick_viewport.extent_and_pixel.xy;
    let pixel = pick_viewport.extent_and_pixel.zw;
    output.clip_pos = vec4<f32>(
        extent.x * output.clip_pos.x
            + (extent.x - 2.0 * pixel.x - 1.0) * output.clip_pos.w,
        extent.y * output.clip_pos.y
            + (1.0 - extent.y + 2.0 * pixel.y) * output.clip_pos.w,
        output.clip_pos.z,
        output.clip_pos.w,
    );
    return output;
}

struct PatchPickOutput {
    // Zero is the cleared no-hit sentinel. Authored indices are encoded +1.
    // z/w retain exact f32 barycentric bits without format conversion.
    @location(0) identity_and_bary: vec4<u32>,
    // xyz is the source-chart surface point; w is displayed camera distance.
    @location(1) surface_and_distance: vec4<f32>,
}

@fragment
fn render_patch_pick(input: PatchVertexOutput) -> PatchPickOutput {
    if input.fade < 0.001 {
        discard;
    }
    let face = u32(max(round(input.instance_id), 0.0));
    let node = u32(max(round(input.node_id), 0.0));
    return PatchPickOutput(
        vec4<u32>(
            face + 1u,
            node + 1u,
            bitcast<u32>(input.tess_bary.x),
            bitcast<u32>(input.tess_bary.y),
        ),
        vec4<f32>(input.source_position_ws, length(input.position_vs)),
    );
}
