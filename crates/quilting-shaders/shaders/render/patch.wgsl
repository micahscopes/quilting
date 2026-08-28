#import quilting::surface::patch_prepare::PreparedPatchRecord
#import quilting::surface::patch_render::{PatchRenderTransform, PatchSurfaceInput, evaluate_patch_surface}
#import quilting::compute::visibility_compaction_types::CompactedBatchRangeRecord

// Compact WebGPU frame block. The pure evaluator deliberately accepts a
// function-local transform value, so WebGL2 retains its established UBO ABI
// while WebGPU pays for no fallback-only model/skinning fields.
struct PatchRenderFrame {
    mvp: mat4x4<f32>,
    mv: mat4x4<f32>,
    // x = use rational QB; y/z/w reserved.
    modes: vec4<i32>,
    mob_a: vec4<f32>,
    mob_b: vec4<f32>,
    mob_c: vec4<f32>,
    mob_d: vec4<f32>,
    camera_pos: vec4<f32>,
}

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
@group(0) @binding(0) var<storage, read> frames: array<PatchRenderFrame>;
@group(0) @binding(1) var<storage, read> prepared_records: array<PreparedPatchRecord>;
@group(0) @binding(2) var<storage, read> compacted_sources: array<u32>;
@group(0) @binding(3) var<storage, read> compacted_ranges: array<CompactedBatchRangeRecord>;
@group(0) @binding(4) var<uniform> draw_batch: DrawBatchIndex;

struct PatchVertexInput {
    @location(0) bary: vec3<f32>,
}

struct PatchVertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) normal_vs: vec3<f32>,
    @location(1) density: f32,
    @location(2) tex_uv: vec2<f32>,
    @location(3) position_vs: vec3<f32>,
    @location(4) tangent_vs: vec3<f32>,
    @location(5) bitangent_vs: vec3<f32>,
    @location(6) normal_ws: vec3<f32>,
    @location(7) position_ws: vec3<f32>,
    @location(8) camera_pos_ws: vec3<f32>,
    @location(9) fade: f32,
    @location(10) tess_bary: vec3<f32>,
    @location(11) instance_id: f32,
    @location(12) mobius_stretch: f32,
    @location(13) source_position_ws: vec3<f32>,
    @location(14) @interpolate(flat) node_id: f32,
}

@vertex
fn render_patch_vertex(
    input: PatchVertexInput,
    @builtin(instance_index) local_instance: u32,
) -> PatchVertexOutput {
    let frame = frames[draw_batch.batch_index];
    let range = compacted_ranges[draw_batch.batch_index];
    let compacted_index = range.compacted_first_instance + local_instance;
    let source_instance = compacted_sources[compacted_index];
    let record = prepared_records[source_instance];
    let surface = evaluate_patch_surface(
        PatchRenderTransform(
            frame.mvp,
            frame.mv,
            frame.modes.x,
            frame.mob_a,
            frame.mob_b,
            frame.mob_c,
            frame.mob_d,
            frame.camera_pos,
        ),
        PatchSurfaceInput(
            input.bary,
            record.record_position_a,
            record.record_position_b,
            record.record_position_c,
            record.record_weight_a,
            record.record_weight_b,
            record.record_weight_c,
            record.record_lod_info,
            record.record_vertex_lod,
            record.record_uv_ab,
            record.record_uv_c_prepare,
            record.record_normal_a,
            record.record_normal_b,
            record.record_normal_c,
            1u,
        ),
    );
    return PatchVertexOutput(
        surface.clip_pos,
        surface.normal_vs,
        surface.density,
        surface.tex_uv,
        surface.position_vs,
        surface.tangent_vs,
        surface.bitangent_vs,
        surface.normal_ws,
        surface.position_ws,
        surface.camera_pos_ws,
        surface.fade,
        surface.tess_bary,
        surface.instance_id,
        surface.mobius_stretch,
        surface.source_position_ws,
        surface.node_id,
    );
}

// First production diagnostic fragment. Material-specific fragments will use
// this exact vertex interface; normals make winding and smooth-normal failures
// immediately observable during backend parity bring-up.
@fragment
fn render_patch_normals(
    @builtin(front_facing) front_facing: bool,
    input: PatchVertexOutput,
) -> @location(0) vec4<f32> {
    if input.fade < 0.001 {
        discard;
    }
    let normal = normalize(input.normal_vs);
    let rgb = normal * 0.5 + 0.5;
    if !front_facing {
        return vec4<f32>(rgb.r * 0.3 + 0.7, rgb.g * 0.3, rgb.b * 0.3, input.fade);
    }
    return vec4<f32>(rgb, input.fade);
}
