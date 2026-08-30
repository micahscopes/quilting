#import quilting::surface::patch_prepare::PreparedPatchRecord
#import quilting::surface::patch_render::PatchRenderTransform
#import quilting::surface::patch_visibility::prepared_patch_outside_frustum
#import quilting::render::patch_vertex::PatchRenderFrame
#import quilting::compute::patch_prepare_types::PatchPrepareDispatch

@group(0) @binding(0) var<uniform> dispatch: PatchPrepareDispatch;
@group(0) @binding(1) var<storage, read> frames: array<PatchRenderFrame>;
@group(0) @binding(2) var<storage, read> prepared_records: array<PreparedPatchRecord>;
@group(0) @binding(3) var<storage, read> patch_frame_rows: array<u32>;
@group(0) @binding(4) var<storage, read_write> patch_visibility: array<u32>;

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

// Prepared adaptive leaves have already been posed and rationally restricted
// to their exact dyadic domain. Classifying those controls avoids both a
// source-face false positive and any CPU visibility expansion.
@compute @workgroup_size(64)
fn classify_prepared_patch_visibility(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let patch_index = invocation.x;
    if patch_index >= dispatch.counts.x {
        return;
    }
    let frame = frames[patch_frame_rows[patch_index]];
    let record = prepared_records[patch_index];
    let outside = prepared_patch_outside_frustum(
        visibility_transform(frame),
        record.record_position_a.yzw,
        record.record_position_b.yzw,
        record.record_position_c.yzw,
        record.record_weight_a,
        record.record_weight_b,
        record.record_weight_c,
    );
    patch_visibility[patch_index] = select(1u, 0u, outside);
}
