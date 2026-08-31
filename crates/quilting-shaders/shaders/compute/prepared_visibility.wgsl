#import quilting::surface::patch_prepare::PreparedPatchRecord
#import quilting::surface::patch_render::PatchRenderTransform
#import quilting::surface::patch_visibility::prepared_patch_outside_frustum
#import quilting::render::patch_vertex::{PatchRenderDomain, PatchRenderGlobal}
#import quilting::compute::patch_prepare_types::PatchPrepareDispatch

@group(0) @binding(0) var<uniform> dispatch: PatchPrepareDispatch;
@group(0) @binding(1) var<storage, read> global_frame: array<PatchRenderGlobal>;
@group(0) @binding(2) var<storage, read> domains: array<PatchRenderDomain>;
@group(0) @binding(3) var<storage, read> prepared_records: array<PreparedPatchRecord>;
@group(0) @binding(4) var<storage, read> patch_domain_rows: array<u32>;
@group(0) @binding(5) var<storage, read_write> patch_visibility: array<u32>;

fn visibility_transform(
    global: PatchRenderGlobal,
    domain: PatchRenderDomain,
) -> PatchRenderTransform {
    return PatchRenderTransform(
        global.mvp,
        global.mv,
        global.modes.x,
        domain.mob_a,
        domain.mob_b,
        domain.mob_c,
        domain.mob_d,
        global.camera_pos_focus,
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
    let global = global_frame[0];
    let domain = domains[patch_domain_rows[patch_index]];
    let record = prepared_records[patch_index];
    let outside = prepared_patch_outside_frustum(
        visibility_transform(global, domain),
        record.record_position_a.yzw,
        record.record_position_b.yzw,
        record.record_position_c.yzw,
        record.record_weight_a,
        record.record_weight_b,
        record.record_weight_c,
    );
    patch_visibility[patch_index] = select(1u, 0u, outside);
}
