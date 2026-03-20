#define_import_path quilting::lighting::pbr

// Cook-Torrance PBR with analytical IBL (no texture dependencies).
// Implements the metallic-roughness model from glTF 2.0.

const PI: f32 = 3.14159265359;

// GGX/Trowbridge-Reitz normal distribution
fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

// Schlick-GGX geometry function
fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return n_dot_v / (n_dot_v * (1.0 - k) + k);
}

// Smith's method — combine for both view and light directions
fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    return geometry_schlick_ggx(n_dot_v, roughness) * geometry_schlick_ggx(n_dot_l, roughness);
}

// Schlick Fresnel approximation
fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// Fresnel with roughness (for IBL)
fn fresnel_schlick_roughness(cos_theta: f32, f0: vec3<f32>, roughness: f32) -> vec3<f32> {
    let max_reflect = max(vec3<f32>(1.0 - roughness), f0);
    return f0 + (max_reflect - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// Analytical DFG approximation (Narkowicz) — replaces BRDF LUT texture
fn env_dfg(specular_color: vec3<f32>, gloss: f32, n_dot_v: f32) -> vec3<f32> {
    let x = gloss;
    let y = n_dot_v;
    let bias = clamp(
        min(-0.1688 * x + 1.895 * x * x,
            0.9903 - 4.853 * y + 8.404 * y * y - 5.069 * y * y * y),
        0.0, 1.0
    );
    var delta = clamp(
        0.6045 + 1.699 * x - 0.5228 * y - 3.603 * x * x
        + 1.404 * x * y + 0.1939 * y * y + 2.661 * x * x * x,
        0.0, 1.0
    );
    let scale = delta - bias;
    let adjusted_bias = bias * clamp(50.0 * specular_color.y, 0.0, 1.0);
    return specular_color * scale + adjusted_bias;
}

struct PBRInput {
    base_color: vec3<f32>,
    metallic: f32,
    roughness: f32,
    normal: vec3<f32>,       // world-space
    view_dir: vec3<f32>,     // toward camera
    light_dir: vec3<f32>,    // toward light
    light_color: vec3<f32>,
}

struct PBROutput {
    color: vec3<f32>,
}

// Direct lighting PBR
fn pbr_direct(input: PBRInput) -> PBROutput {
    let n = input.normal;
    let v = input.view_dir;
    let l = input.light_dir;
    let h = normalize(v + l);

    let n_dot_v = max(dot(n, v), 0.001);
    let n_dot_l = max(dot(n, l), 0.0);
    let n_dot_h = max(dot(n, h), 0.0);
    let v_dot_h = max(dot(v, h), 0.0);

    // Dielectric F0 = 0.04, metallic F0 = base_color
    let f0 = mix(vec3<f32>(0.04), input.base_color, input.metallic);

    let D = distribution_ggx(n_dot_h, input.roughness);
    let G = geometry_smith(n_dot_v, n_dot_l, input.roughness);
    let F = fresnel_schlick(v_dot_h, f0);

    let specular = (D * G * F) / max(4.0 * n_dot_v * n_dot_l, 0.001);

    let k_d = (1.0 - F) * (1.0 - input.metallic);
    let diffuse = k_d * input.base_color / PI;

    let color = (diffuse + specular) * input.light_color * n_dot_l;
    return PBROutput(color);
}

// Ambient / IBL approximation using analytical DFG + simple environment
fn pbr_ambient(
    base_color: vec3<f32>,
    metallic: f32,
    roughness: f32,
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    ambient_color: vec3<f32>,
) -> vec3<f32> {
    let n_dot_v = max(dot(normal, view_dir), 0.001);
    let f0 = mix(vec3<f32>(0.04), base_color, metallic);

    let F = fresnel_schlick_roughness(n_dot_v, f0, roughness);
    let k_d = (1.0 - F) * (1.0 - metallic);

    let diffuse = k_d * base_color * ambient_color;
    let specular = env_dfg(f0, 1.0 - roughness, n_dot_v) * ambient_color;

    return diffuse + specular;
}
