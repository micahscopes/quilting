#import quilting::render::patch_vertex::{PatchPbrMaterial, PatchRenderDomain, PatchRenderGlobal, PatchVertexOutput, evaluate_prepared_patch_vertex, patch_focus_raw_field, pull_patch_wire_overlay_forward, shade_patch_highlight, shade_patch_lod, shade_patch_matcap, shade_patch_normals, shade_patch_stretch, shade_patch_wire}
#import quilting::render::patch_pbr_portable::shade_portable_patch_pbr
#import quilting::surface::patch_prepare::PreparedPatchRecord
#import quilting::compute::resident_bucket_types::{ResidentBucketRangeRecord, ResidentDrawDomainRecord}

struct DrawRootBucketIndex {
    bucket_index: u32,
    _padding_a: u32,
    _padding_b: u32,
    _padding_c: u32,
}

@group(0) @binding(0) var<storage, read> global_frame: array<PatchRenderGlobal>;
@group(0) @binding(1) var<storage, read> render_domains: array<PatchRenderDomain>;
@group(0) @binding(2) var<storage, read> prepared_records: array<PreparedPatchRecord>;
@group(0) @binding(3) var<storage, read> compacted_faces: array<u32>;
@group(0) @binding(4) var<storage, read> bucket_ranges: array<ResidentBucketRangeRecord>;
@group(0) @binding(5) var<uniform> draw_bucket: DrawRootBucketIndex;
@group(0) @binding(6) var<storage, read> face_domain_rows: array<u32>;
@group(0) @binding(7) var<storage, read> draw_domains: array<ResidentDrawDomainRecord>;
@group(0) @binding(8) var<storage, read> pbr_materials: array<PatchPbrMaterial>;

struct ResidentRootVertexInput {
    @location(0) bary: vec3<f32>,
}

fn evaluate_resident_root_draw_vertex(
    input: ResidentRootVertexInput,
    local_instance: u32,
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
    // Bucketing already excludes disabled domains. Keep the vertex path
    // fail-closed if a future producer violates that invariant.
    if (domain.flags & 1u) == 0u {
        output.clip_pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        output.fade = 0.0;
    }
    return output;
}

@vertex
fn render_resident_root_vertex(
    input: ResidentRootVertexInput,
    @builtin(instance_index) local_instance: u32,
) -> PatchVertexOutput {
    return evaluate_resident_root_draw_vertex(input, local_instance);
}

@vertex
fn render_resident_root_wire_vertex(
    input: ResidentRootVertexInput,
    @builtin(instance_index) local_instance: u32,
) -> PatchVertexOutput {
    let output = evaluate_resident_root_draw_vertex(input, local_instance);
    return pull_patch_wire_overlay_forward(output);
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

@fragment
fn render_resident_root_matcap(input: PatchVertexOutput) -> @location(0) vec4<f32> {
    return shade_patch_matcap(input);
}

@fragment
fn render_resident_root_stretch(input: PatchVertexOutput) -> @location(0) vec4<f32> {
    return shade_patch_stretch(input);
}

@fragment
fn render_resident_root_wire(input: PatchVertexOutput) -> @location(0) vec4<f32> {
    return shade_patch_wire(input);
}

@fragment
fn render_resident_root_highlight(input: PatchVertexOutput) -> @location(0) vec4<f32> {
    return shade_patch_highlight(input, bitcast<u32>(global_frame[0].modes.w));
}

@fragment
fn render_resident_root_pbr(
    @builtin(front_facing) front_facing: bool,
    input: PatchVertexOutput,
) -> @location(0) vec4<f32> {
    let source_face = u32(max(input.instance_id, 0.0));
    let domain_row = face_domain_rows[source_face];
    let domain = draw_domains[domain_row];
    let material_count = arrayLength(&pbr_materials);
    let material_index = select(
        0u,
        domain.material_index,
        domain.material_index < material_count,
    );
    return shade_portable_patch_pbr(
        front_facing,
        input,
        pbr_materials[material_index],
        material_index,
        global_frame[0].modes.z,
    );
}

struct ResidentRootPbrFocusOutput {
    @location(0) color: vec4<f32>,
    @location(1) raw_field: vec4<f32>,
}

@fragment
fn render_resident_root_pbr_focus(
    @builtin(front_facing) front_facing: bool,
    input: PatchVertexOutput,
) -> ResidentRootPbrFocusOutput {
    let source_face = u32(max(input.instance_id, 0.0));
    let domain_row = face_domain_rows[source_face];
    let domain = draw_domains[domain_row];
    let material_count = arrayLength(&pbr_materials);
    let material_index = select(
        0u,
        domain.material_index,
        domain.material_index < material_count,
    );
    let global = global_frame[0];
    return ResidentRootPbrFocusOutput(
        shade_portable_patch_pbr(
            front_facing,
            input,
            pbr_materials[material_index],
            material_index,
            global.modes.z,
        ),
        patch_focus_raw_field(input, global),
    );
}
