// Pick buffer fragment shader.
// Encodes global face ID (24-bit) for mouse picking.
// Attachment 0: R,G,B = face ID (low, mid, high byte), A = 1.0.
// Attachment 1: source-face barycentric coordinates for stable surface
// attachment. RGBA8 precision seeds the Rust walker; subsequent integration
// stays in f64 source coordinates.

struct FragInput {
    @location(9) fade: f32,
    @location(10) tess_bary: vec3<f32>,
    @location(11) instance_id: f32,
}

struct PickOutput {
    @location(0) face_id: vec4<f32>,
    @location(1) barycentric: vec4<f32>,
}

@fragment
fn fs_pick(in: FragInput) -> PickOutput {
    if in.fade < 0.001 { discard; }

    let id = i32(in.instance_id + 0.5);
    let r = f32(id & 255) / 255.0;
    let g = f32((id >> 8) & 255) / 255.0;
    let b = f32((id >> 16) & 255) / 255.0;

    return PickOutput(
        vec4<f32>(r, g, b, 1.0),
        vec4<f32>(in.tess_bary, 1.0),
    );
}
