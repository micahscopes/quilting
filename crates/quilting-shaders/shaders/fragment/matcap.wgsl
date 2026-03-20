// Matcap fragment shader: density heatmap + matcap-style lighting.

#import quilting::lighting::matcap::matcap_shade

struct FragInput {
    @location(0) normal_vs: vec3<f32>,
    @location(1) density: f32,
}

// Heatmap colormap: blue -> cyan -> green -> yellow -> red
fn heatmap(t_in: f32) -> vec3<f32> {
    let t = clamp(t_in, 0.0, 1.0);
    let c0 = vec3<f32>(0.0, 0.0, 0.5);
    let c1 = vec3<f32>(0.0, 0.5, 1.0);
    let c2 = vec3<f32>(0.0, 1.0, 0.5);
    let c3 = vec3<f32>(1.0, 1.0, 0.0);
    let c4 = vec3<f32>(1.0, 0.0, 0.0);
    if t < 0.25 { return mix(c0, c1, t * 4.0); }
    if t < 0.5  { return mix(c1, c2, (t - 0.25) * 4.0); }
    if t < 0.75 { return mix(c2, c3, (t - 0.5) * 4.0); }
    return mix(c3, c4, (t - 0.75) * 4.0);
}

@fragment
fn fs_matcap(in: FragInput) -> @location(0) vec4<f32> {
    var n = normalize(in.normal_vs);
    // Two-sided lighting: flip if facing away from camera
    if n.z < 0.0 {
        n = -n;
    }

    let base = heatmap(in.density);
    let col = matcap_shade(n, base);
    return vec4<f32>(col, 1.0);
}
