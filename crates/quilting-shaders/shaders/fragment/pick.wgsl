// Pick buffer fragment shader.
// Encodes global face ID + tessellation bary coords for mouse picking.
// R,G = global face ID (low, high byte), B = bary.x, A = bary.y

struct FragInput {
    @location(9) fade: f32,
    @location(10) tess_bary: vec3<f32>,
    @location(11) instance_id: f32,
}

@fragment
fn fs_pick(in: FragInput) -> @location(0) vec4<f32> {
    if in.fade < 0.001 { discard; }

    let id = i32(in.instance_id + 0.5);
    let r = f32(id & 255) / 255.0;
    let g = f32((id >> 8) & 255) / 255.0;
    let b = in.tess_bary.x;
    let a = in.tess_bary.y;

    return vec4<f32>(r, g, b, a);
}
