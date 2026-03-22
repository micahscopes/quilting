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

// --- Cubemap IBL for diffuse + specular environment lighting ---
// When cubemaps are bound, sample them directly.
// Fallback to analytical SH + gradient when cubemaps are unavailable.

// Analytical fallback: SH irradiance (L0 + L1)
fn sh_irradiance_fallback(normal: vec3<f32>) -> vec3<f32> {
    let l0 = vec3<f32>(0.30, 0.28, 0.32);
    let l1_x = vec3<f32>(0.05, 0.04, 0.02);
    let l1_y = vec3<f32>(0.15, 0.18, 0.25);
    let l1_z = vec3<f32>(0.02, 0.02, 0.03);
    return max(l0 + l1_x * normal.x + l1_y * normal.y + l1_z * normal.z, vec3<f32>(0.0));
}

// Analytical fallback: sky/ground gradient specular
fn env_specular_fallback(reflect_dir: vec3<f32>, roughness: f32) -> vec3<f32> {
    let sky = vec3<f32>(0.4, 0.45, 0.55);
    let horizon = vec3<f32>(0.35, 0.35, 0.38);
    let ground = vec3<f32>(0.12, 0.10, 0.08);
    let blur = roughness * roughness;
    let y = reflect_dir.y;
    let sharp = mix(ground, mix(horizon, sky, clamp(y * 2.0 + 0.5, 0.0, 1.0)), clamp(y + 0.5, 0.0, 1.0));
    let average = vec3<f32>(0.25, 0.25, 0.28);
    return mix(sharp, average, blur);
}

struct PBRInput {
    base_color: vec3<f32>,
    metallic: f32,
    roughness: f32,
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    light_dir: vec3<f32>,
    light_color: vec3<f32>,
    f0_override: vec3<f32>,  // custom F0 (from KHR_materials_specular)
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

    // Use f0_override (already incorporates specularColorFactor)
    let f0 = input.f0_override;

    let D = distribution_ggx(n_dot_h, input.roughness);
    let G = geometry_smith(n_dot_v, n_dot_l, input.roughness);
    let F = fresnel_schlick(v_dot_h, f0);

    let specular = (D * G * F) / max(4.0 * n_dot_v * n_dot_l, 0.001);

    let k_d = (1.0 - F) * (1.0 - input.metallic);
    let diffuse = k_d * input.base_color / PI;

    let color = (diffuse + specular) * input.light_color * n_dot_l;
    return PBROutput(color);
}

// Ambient / IBL: takes pre-sampled irradiance and specular environment colors.
// The caller is responsible for sampling cubemaps (or using the analytical fallbacks).
fn pbr_ambient(
    base_color: vec3<f32>,
    metallic: f32,
    roughness: f32,
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    irradiance: vec3<f32>,
    env_color: vec3<f32>,
    f0_in: vec3<f32>,
) -> vec3<f32> {
    let n_dot_v = max(dot(normal, view_dir), 0.001);
    let f0 = f0_in;

    // Use f0 magnitude as the f90 (specularWeight in the Khronos spec).
    // When specularColorFactor=[0,0,0], f90=0 → no grazing reflections.
    let f90 = vec3<f32>(max(f0.x, max(f0.y, f0.z)));
    let F = f0 + (max(f90 * (1.0 - roughness), f0) - f0) * pow(clamp(1.0 - n_dot_v, 0.0, 1.0), 5.0);
    let k_d = (1.0 - F) * (1.0 - metallic);

    let diffuse = k_d * base_color * irradiance;
    let specular = env_dfg(f0, 1.0 - roughness, n_dot_v) * env_color;

    return diffuse + specular;
}
