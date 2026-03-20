#define_import_path quilting::lighting::matcap

// Procedural matcap shading from view-space normal.
// No texture needed — computes lighting analytically.

fn matcap_shade(normal_vs: vec3<f32>, base_color: vec3<f32>) -> vec3<f32> {
    let n = normal_vs;
    let n_dot_v = max(n.z, 0.0);

    let light_dir = normalize(vec3<f32>(0.4, 0.7, 0.9));
    let n_dot_l = dot(n, light_dir);
    let diff = n_dot_l * 0.5 + 0.5; // half-lambert wrap

    let h = normalize(light_dir + vec3<f32>(0.0, 0.0, 1.0));
    let spec = pow(max(dot(n, h), 0.0), 40.0);
    let rim = pow(1.0 - n_dot_v, 3.0);

    let shadow = base_color * 0.15 + vec3<f32>(0.02, 0.02, 0.06);
    let lit = base_color * 1.2;
    var col = mix(shadow, lit, diff);
    col += vec3<f32>(1.0, 0.97, 0.92) * spec * 0.6;
    col += base_color * rim * 0.3;
    col += vec3<f32>(0.04, 0.03, 0.08);

    return col;
}
