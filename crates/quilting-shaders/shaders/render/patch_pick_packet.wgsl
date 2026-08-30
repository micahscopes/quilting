#define_import_path quilting::render::patch_pick_packet

#import quilting::render::patch_vertex::PatchVertexOutput

// x/y = full viewport extent; z/w = queried top-left pixel coordinate.
struct PatchPickViewport {
    extent_and_pixel: vec4<f32>,
}

struct PatchPickOutput {
    // Zero is the cleared no-hit sentinel. Authored indices are encoded +1.
    // z/w retain exact f32 barycentric bits without format conversion.
    @location(0) identity_and_bary: vec4<u32>,
    // xyz is the source-chart surface point; w is displayed camera distance.
    @location(1) surface_and_distance: vec4<f32>,
}

// Remap one queried full-viewport pixel to a one-pixel attachment. This
// preserves the ordinary raster/depth result without retaining a second
// full-resolution ID framebuffer.
fn remap_patch_pick_clip(
    input: PatchVertexOutput,
    viewport: PatchPickViewport,
) -> PatchVertexOutput {
    var output = input;
    let extent = viewport.extent_and_pixel.xy;
    let pixel = viewport.extent_and_pixel.zw;
    output.clip_pos = vec4<f32>(
        extent.x * input.clip_pos.x
            + (extent.x - 2.0 * pixel.x - 1.0) * input.clip_pos.w,
        extent.y * input.clip_pos.y
            + (1.0 - extent.y + 2.0 * pixel.y) * input.clip_pos.w,
        input.clip_pos.z,
        input.clip_pos.w,
    );
    return output;
}

fn encode_patch_pick(input: PatchVertexOutput) -> PatchPickOutput {
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
