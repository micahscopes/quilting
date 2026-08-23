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

// Texture-free character matcaps. Keeping these profiles analytic makes the
// selected look reproducible in WebGL and a future WebGPU backend without
// shipping an otherwise unrelated image asset or sampling a texture per pixel.
// style: 0 = aqua, 1 = citric acid, 2 = golden soft, 3 = soft studio.
fn procedural_matcap(normal_vs: vec3<f32>, style: f32) -> vec3<f32> {
    let n = normalize(normal_vs);
    let view = vec3<f32>(0.0, 0.0, 1.0);
    let key = normalize(vec3<f32>(-0.42, 0.66, 0.72));
    let half_vector = normalize(key + view);
    let diffuse = smoothstep(-0.55, 0.92, dot(n, key));
    let specular = pow(max(dot(n, half_vector), 0.0), 34.0);
    let broad_specular = pow(max(dot(n, half_vector), 0.0), 7.0);
    let rim = pow(1.0 - max(n.z, 0.0), 2.4);
    let upper = smoothstep(-0.8, 0.9, n.y);

    var shadow = vec3<f32>(0.012, 0.035, 0.055);
    var body = vec3<f32>(0.055, 0.48, 0.50);
    var highlight = vec3<f32>(0.62, 1.0, 0.90);
    var rim_color = vec3<f32>(0.20, 0.12, 0.46);
    var accent = vec3<f32>(0.10, 0.70, 0.62);

    if style > 0.5 && style < 1.5 {
        shadow = vec3<f32>(0.105, 0.015, 0.13);
        body = vec3<f32>(0.70, 0.20, 0.39);
        highlight = vec3<f32>(1.0, 0.95, 0.24);
        rim_color = vec3<f32>(0.66, 0.07, 0.47);
        accent = vec3<f32>(0.98, 0.58, 0.16);
    } else if style > 1.5 && style < 2.5 {
        shadow = vec3<f32>(0.035, 0.020, 0.006);
        body = vec3<f32>(0.56, 0.29, 0.035);
        highlight = vec3<f32>(1.0, 0.84, 0.36);
        rim_color = vec3<f32>(0.27, 0.11, 0.018);
        accent = vec3<f32>(0.82, 0.52, 0.12);
    } else if style > 2.5 {
        shadow = vec3<f32>(0.045, 0.065, 0.105);
        body = vec3<f32>(0.50, 0.47, 0.46);
        highlight = vec3<f32>(0.96, 0.78, 0.61);
        rim_color = vec3<f32>(0.17, 0.37, 0.58);
        accent = vec3<f32>(0.63, 0.52, 0.48);
    }

    var color = mix(shadow, body, diffuse);
    color = mix(color, accent, upper * 0.16);
    color += highlight * (specular * 0.72 + broad_specular * 0.10);
    color += rim_color * rim * 0.48;
    return color;
}
