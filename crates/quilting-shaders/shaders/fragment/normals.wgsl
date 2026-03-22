// Normal visualization fragment shader: map normal XYZ from [-1,1] to [0,1] as RGB.

struct FragInput {
    @builtin(front_facing) front_facing: bool,
    @location(0) normal_vs: vec3<f32>,
    @location(1) density: f32,
    @location(9) fade: f32,
    @location(10) @interpolate(flat) perm_sign: f32,
}

@fragment
fn fs_normals(in: FragInput) -> @location(0) vec4<f32> {
    if in.fade < 0.001 { discard; }
    var n = normalize(in.normal_vs);
    // Two-sided: face winding corrected for permutation parity
    var face_back = in.front_facing;
    if in.perm_sign < 0.0 {
        face_back = !face_back;
    }
    if face_back {
        n = -n;
    }
    return vec4<f32>(n * 0.5 + 0.5, in.fade);
}
