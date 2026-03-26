// Normal visualization fragment shader: map normal XYZ from [-1,1] to [0,1] as RGB.
// Back-facing fragments are tinted red to reveal winding issues.

struct FragInput {
    @location(0) normal_vs: vec3<f32>,
    @location(1) density: f32,
    @location(9) fade: f32,
}

@fragment
fn fs_normals(@builtin(front_facing) front_facing: bool, in: FragInput) -> @location(0) vec4<f32> {
    if in.fade < 0.001 { discard; }
    let n = normalize(in.normal_vs);
    let rgb = n * 0.5 + 0.5;
    if !front_facing {
        // Back-face: tint red to make winding issues obvious
        return vec4<f32>(rgb.r * 0.3 + 0.7, rgb.g * 0.3, rgb.b * 0.3, in.fade);
    }
    return vec4<f32>(rgb, in.fade);
}
