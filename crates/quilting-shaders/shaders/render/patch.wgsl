#import quilting::render::patch_vertex::{PatchPbrMaterial, PatchRenderDomain, PatchRenderGlobal, PatchVertexOutput, evaluate_prepared_patch_vertex, patch_focus_raw_field, pull_patch_wire_overlay_forward, shade_patch_highlight, shade_patch_lod, shade_patch_matcap, shade_patch_normals, shade_patch_stretch, shade_patch_wire}
#import quilting::render::patch_pbr::shade_textured_patch_pbr
#import quilting::surface::patch_prepare::PreparedPatchRecord
#import quilting::compute::visibility_compaction_types::CompactedBatchRangeRecord

struct DrawBatchIndex {
    batch_index: u32,
    _padding_a: u32,
    _padding_b: u32,
    _padding_c: u32,
}

// A real extracted scene may assign a distinct conformal map to every batch.
// Keep those immutable-for-the-frame records in one device table and select
// them with the same portable batch index used for compacted ranges. A single
// uniform here would make queue writes race all draws in one submission.
@group(0) @binding(0) var<storage, read> global_frame: array<PatchRenderGlobal>;
@group(0) @binding(1) var<storage, read> domains: array<PatchRenderDomain>;
@group(0) @binding(2) var<storage, read> prepared_records: array<PreparedPatchRecord>;
@group(0) @binding(3) var<storage, read> compacted_sources: array<u32>;
@group(0) @binding(4) var<storage, read> compacted_ranges: array<CompactedBatchRangeRecord>;
@group(0) @binding(5) var<uniform> draw_batch: DrawBatchIndex;
@group(0) @binding(6) var<storage, read> pbr_materials: array<PatchPbrMaterial>;

struct PatchVertexInput {
    @location(0) bary: vec3<f32>,
}

fn evaluate_patch_draw_vertex(
    input: PatchVertexInput,
    local_instance: u32,
) -> PatchVertexOutput {
    let range = compacted_ranges[draw_batch.batch_index];
    let compacted_index = range.compacted_first_instance + local_instance;
    let source_instance = compacted_sources[compacted_index];
    return evaluate_prepared_patch_vertex(
        global_frame[0],
        domains[draw_batch.batch_index],
        input.bary,
        prepared_records[source_instance],
    );
}

@vertex
fn render_patch_vertex(
    input: PatchVertexInput,
    @builtin(instance_index) local_instance: u32,
) -> PatchVertexOutput {
    return evaluate_patch_draw_vertex(input, local_instance);
}

@vertex
fn render_patch_wire_vertex(
    input: PatchVertexInput,
    @builtin(instance_index) local_instance: u32,
) -> PatchVertexOutput {
    let output = evaluate_patch_draw_vertex(input, local_instance);
    // Wire is also a semantic coplanar overlay after filled matcap. Pull it a
    // tiny, deterministic amount toward the camera so visibility does not
    // depend on line-versus-triangle depth interpolation details.
    return pull_patch_wire_overlay_forward(output);
}

@fragment
fn render_patch_normals(
    @builtin(front_facing) front_facing: bool,
    input: PatchVertexOutput,
) -> @location(0) vec4<f32> {
    return shade_patch_normals(front_facing, input);
}

@fragment
fn render_patch_lod(input: PatchVertexOutput) -> @location(0) vec4<f32> {
    return shade_patch_lod(input);
}

@fragment
fn render_patch_stretch(input: PatchVertexOutput) -> @location(0) vec4<f32> {
    return shade_patch_stretch(input);
}

@fragment
fn render_patch_matcap(input: PatchVertexOutput) -> @location(0) vec4<f32> {
    return shade_patch_matcap(input);
}

@fragment
fn render_patch_wire(input: PatchVertexOutput) -> @location(0) vec4<f32> {
    return shade_patch_wire(input);
}

@fragment
fn render_patch_highlight(input: PatchVertexOutput) -> @location(0) vec4<f32> {
    return shade_patch_highlight(
        input,
        bitcast<u32>(global_frame[0].modes.w),
    );
}

@fragment
fn render_patch_pbr(
    @builtin(front_facing) front_facing: bool,
    input: PatchVertexOutput,
) -> @location(0) vec4<f32> {
    let material_index = u32(max(domains[draw_batch.batch_index].modes.x, 0));
    let material = pbr_materials[material_index];
    return shade_textured_patch_pbr(
        front_facing,
        input,
        material,
        global_frame[0].modes.z,
    );
}

struct PatchPbrFocusOutput {
    @location(0) color: vec4<f32>,
    @location(1) raw_field: vec4<f32>,
}

@fragment
fn render_patch_pbr_focus(
    @builtin(front_facing) front_facing: bool,
    input: PatchVertexOutput,
) -> PatchPbrFocusOutput {
    let global = global_frame[0];
    let domain = domains[draw_batch.batch_index];
    let material_index = u32(max(domain.modes.x, 0));
    let material = pbr_materials[material_index];
    return PatchPbrFocusOutput(
        shade_textured_patch_pbr(front_facing, input, material, global.modes.z),
        patch_focus_raw_field(input, global),
    );
}
