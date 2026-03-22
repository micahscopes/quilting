#define_import_path quilting::fragment::pbr

#import quilting::lighting::pbr::{pbr_direct, pbr_ambient, PBRInput, env_dfg, fresnel_schlick_roughness, sh_irradiance_fallback, env_specular_fallback}

struct PbrUniforms {
    base_color: vec4<f32>,
    metallic: f32,
    roughness: f32,
    has_base_color_tex: f32,
    has_metallic_roughness_tex: f32,
    emissive_factor: vec4<f32>,          // rgb in xyz, w unused
    normal_scale: f32,
    has_normal_tex: f32,
    has_emissive_tex: f32,
    has_occlusion_tex: f32,
    occlusion_strength: f32,
    alpha_cutoff: f32,
    alpha_mode: f32,                     // 0=opaque, 1=mask, 2=blend
    unlit: f32,                          // >0.5 = KHR_materials_unlit (base color only)
    has_env_map: f32,                    // >0.5 = cubemap IBL available
    env_mip_count: f32,                  // number of mip levels in prefiltered env map
    _pbr_pad0: f32,
    _pbr_pad1: f32,
}

@group(0) @binding(1)
var<uniform> pbr: PbrUniforms;

@group(0) @binding(2)
var base_color_tex: texture_2d<f32>;
@group(0) @binding(3)
var base_color_sampler: sampler;

@group(0) @binding(4)
var metallic_roughness_tex: texture_2d<f32>;
@group(0) @binding(5)
var metallic_roughness_sampler: sampler;

@group(0) @binding(6)
var normal_tex: texture_2d<f32>;
@group(0) @binding(7)
var normal_sampler: sampler;

@group(0) @binding(8)
var emissive_tex: texture_2d<f32>;
@group(0) @binding(9)
var emissive_sampler: sampler;

@group(0) @binding(10)
var occlusion_tex: texture_2d<f32>;
@group(0) @binding(11)
var occlusion_sampler: sampler;

// Environment cubemaps for IBL
@group(0) @binding(12)
var env_prefiltered: texture_cube<f32>;
@group(0) @binding(13)
var env_prefiltered_sampler: sampler;

@group(0) @binding(14)
var env_irradiance: texture_cube<f32>;
@group(0) @binding(15)
var env_irradiance_sampler: sampler;

struct FragInput {
    @location(0) normal_vs: vec3<f32>,
    @location(1) density: f32,
    @location(2) tex_uv: vec2<f32>,
    @location(3) position_vs: vec3<f32>,
    @location(4) tangent_vs: vec3<f32>,
    @location(5) bitangent_vs: vec3<f32>,
    @location(6) normal_ws: vec3<f32>,
    @location(7) position_ws: vec3<f32>,
}

@fragment
fn fs_pbr(in: FragInput) -> @location(0) vec4<f32> {
    var n = normalize(in.normal_vs);
    let view_dir = vec3<f32>(0.0, 0.0, 1.0); // view space: camera at +Z

    // Two-sided lighting: flip normal if it points away from the camera.
    if dot(n, view_dir) < 0.0 {
        n = -n;
    }

    // --- Base color ---
    var base = pbr.base_color;
    if pbr.has_base_color_tex > 0.5 {
        let tex_color = textureSample(base_color_tex, base_color_sampler, in.tex_uv);
        let linear_rgb = pow(tex_color.rgb, vec3<f32>(2.2));
        base = vec4<f32>(linear_rgb * pbr.base_color.rgb, tex_color.a * pbr.base_color.a);
    }

    // --- Alpha ---
    var alpha = base.a;
    if pbr.alpha_mode > 0.5 && pbr.alpha_mode < 1.5 {
        // MASK mode
        if alpha < pbr.alpha_cutoff {
            discard;
        }
    }
    if pbr.alpha_mode < 0.5 {
        // OPAQUE mode
        alpha = 1.0;
    }

    // --- Unlit: just output base color with gamma correction ---
    if pbr.unlit > 0.5 {
        var unlit_color = base.rgb;
        // Gamma correction (base is already linear from sRGB conversion above)
        unlit_color = pow(unlit_color, vec3<f32>(1.0 / 2.2));
        return vec4<f32>(unlit_color, alpha);
    }

    // --- Metallic / Roughness ---
    var metallic = pbr.metallic;
    var roughness = pbr.roughness;
    if pbr.has_metallic_roughness_tex > 0.5 {
        let mr = textureSample(metallic_roughness_tex, metallic_roughness_sampler, in.tex_uv);
        // glTF: B = metallic, G = roughness (linear, no sRGB conversion)
        metallic = mr.b * pbr.metallic;
        roughness = mr.g * pbr.roughness;
    }
    roughness = clamp(roughness, 0.04, 1.0);

    // --- Normal mapping (vertex-computed tangent frame) ---
    if pbr.has_normal_tex > 0.5 {
        let raw_t = in.tangent_vs;
        // Guard against degenerate tangents (NaN/zero from stretched Möbius faces)
        if dot(raw_t, raw_t) > 1e-6 {
            var t = normalize(raw_t);
            t = normalize(t - n * dot(t, n));  // Gram-Schmidt
            if dot(t, t) > 0.5 {               // guard post-projection
                let b = cross(n, t);
                let tbn = mat3x3<f32>(t, b, n);

                let nm = textureSample(normal_tex, normal_sampler, in.tex_uv).xyz;
                var tangent_n = nm * 2.0 - vec3<f32>(1.0);
                tangent_n.x = tangent_n.x * pbr.normal_scale;
                tangent_n.y = tangent_n.y * pbr.normal_scale;
                n = normalize(tbn * tangent_n);
            }
        }
    }

    // --- Lighting ---
    // Key light
    let light_dir = normalize(vec3<f32>(0.5, 0.8, 0.6));
    let light_color = vec3<f32>(3.0, 2.9, 2.7);

    let input = PBRInput(
        base.rgb,
        metallic,
        roughness,
        n,
        view_dir,
        light_dir,
        light_color,
    );
    let direct = pbr_direct(input);

    // Fill light
    let fill_input = PBRInput(
        base.rgb,
        metallic,
        roughness,
        n,
        view_dir,
        normalize(vec3<f32>(-0.4, -0.3, 0.5)),
        vec3<f32>(0.3, 0.35, 0.5),
    );
    let fill = pbr_direct(fill_input);

    // --- IBL ambient ---
    // Use world-space normal for cubemap lookups
    let n_ws = normalize(in.normal_ws);
    let view_dir_ws = normalize(-in.position_ws); // camera at origin in world space
    let reflect_ws = reflect(-view_dir_ws, n_ws);

    var irradiance: vec3<f32>;
    var env_color: vec3<f32>;
    if pbr.has_env_map > 0.5 {
        // Cubemap IBL: sample irradiance + prefiltered specular
        irradiance = textureSample(env_irradiance, env_irradiance_sampler, n_ws).rgb;
        let lod = roughness * pbr.env_mip_count;
        env_color = textureSampleLevel(env_prefiltered, env_prefiltered_sampler, reflect_ws, lod).rgb;
    } else {
        // Analytical fallback
        irradiance = sh_irradiance_fallback(n_ws);
        env_color = env_specular_fallback(reflect_ws, roughness);
    }

    var ambient = pbr_ambient(
        base.rgb,
        metallic,
        roughness,
        n,
        view_dir,
        irradiance,
        env_color,
    );

    // --- Occlusion (ambient only) ---
    if pbr.has_occlusion_tex > 0.5 {
        let ao = textureSample(occlusion_tex, occlusion_sampler, in.tex_uv).r;
        ambient = ambient * mix(1.0, ao, pbr.occlusion_strength);
    }

    var color = direct.color + fill.color + ambient;

    // --- Emissive ---
    var emissive = pbr.emissive_factor.rgb;
    if pbr.has_emissive_tex > 0.5 {
        let em_tex = textureSample(emissive_tex, emissive_sampler, in.tex_uv);
        let em_linear = pow(em_tex.rgb, vec3<f32>(2.2));
        emissive = emissive * em_linear;
    }
    color = color + emissive;

    // Tone mapping (Reinhard)
    color = color / (color + vec3<f32>(1.0));
    // Gamma correction
    color = pow(color, vec3<f32>(1.0 / 2.2));

    return vec4<f32>(color, alpha);
}
