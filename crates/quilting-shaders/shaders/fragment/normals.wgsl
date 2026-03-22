// Normal visualization fragment shader: map normal XYZ from [-1,1] to [0,1] as RGB.

struct FragInput {
    @location(0) normal_vs: vec3<f32>,
    @location(1) density: f32,
    @location(9) fade: f32,
}

@fragment
fn fs_normals(in: FragInput) -> @location(0) vec4<f32> {
    if in.fade < 0.001 { discard; }
    let n = normalize(in.normal_vs);
    return vec4<f32>(n * 0.5 + 0.5, in.fade);
}
