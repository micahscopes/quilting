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
    // KHR_materials_sheen
    sheen_color: vec4<f32>,              // rgb in xyz, w = has_sheen (>0.5)
    sheen_roughness: f32,
    // KHR_materials_specular
    specular_color: vec4<f32>,           // rgb in xyz, w = has_specular (>0.5)
    // KHR_texture_transform for normal map
    normal_uv_transform: vec4<f32>,     // xy = scale, zw = offset
    normal_uv_rotation: f32,            // rotation in radians
    _pbr_pad2: f32,
    _pbr_pad3: f32,
    _pbr_pad4: f32,
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
    @location(8) camera_pos_ws: vec3<f32>,
    @location(9) fade: f32,
}

@fragment
fn fs_pbr(in: FragInput) -> @location(0) vec4<f32> {
    if in.fade < 0.001 { discard; }

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
        let linear_rgb = pow(tex_color.rgb, vec3<f32>(2.2)); // sRGB → linear
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
    // Apply KHR_texture_transform to normal map UVs if present.
    var normal_uv = in.tex_uv;
    if pbr.normal_uv_transform.x > 0.0 {
        let s = pbr.normal_uv_transform.xy; // scale
        let o = pbr.normal_uv_transform.zw; // offset
        let r = pbr.normal_uv_rotation;
        let cr = cos(r); let sr = sin(r);
        let uv = in.tex_uv;
        // rotate then scale then offset (per KHR_texture_transform spec)
        normal_uv = vec2<f32>(
            (uv.x * cr - uv.y * sr) * s.x + o.x,
            (uv.x * sr + uv.y * cr) * s.y + o.y,
        );
    }
    if pbr.has_normal_tex > 0.5 {
        let raw_t = in.tangent_vs;
        // Guard against degenerate tangents (NaN/zero from stretched Möbius faces)
        if dot(raw_t, raw_t) > 1e-6 {
            var t = normalize(raw_t);
            t = normalize(t - n * dot(t, n));  // Gram-Schmidt
            if dot(t, t) > 0.5 {               // guard post-projection
                let b = cross(n, t);
                let tbn = mat3x3<f32>(t, b, n);

                let nm = textureSample(normal_tex, normal_sampler, normal_uv).xyz;
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
    let view_dir_ws = normalize(in.camera_pos_ws - in.position_ws);
    let reflect_ws = reflect(-view_dir_ws, n_ws);

    // Sample environment cubemaps (HDR values when loaded from .hdr files)
    let irradiance = textureSampleLevel(env_irradiance, env_irradiance_sampler, n_ws, 0.0).rgb;
    let lod = roughness * max(pbr.env_mip_count, 1.0);
    let env_color = textureSampleLevel(env_prefiltered, env_prefiltered_sampler, reflect_ws, lod).rgb;

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

    // --- KHR_materials_sheen (velvet/fabric) ---
    // Charlie distribution BRDF: peaks at grazing angles (sin-based, not cos-based).
    // Evaluated as a full BRDF lobe per light, not just a rim effect.
    if pbr.sheen_color.w > 0.5 {
        let sheen_col = pbr.sheen_color.rgb;
        let sheen_r = max(pbr.sheen_roughness, 0.07);
        let n_dot_v_s = max(dot(n, view_dir), 0.001);
        let inv_r = 1.0 / sheen_r;

        // Charlie D: (2 + 1/r) / (2π) * sin(θ_h)^(1/r)
        // V: Ashikhmin visibility: 1 / (4 * (NdotL + NdotV - NdotL*NdotV))
        // Sheen on key light
        let l_key = normalize(vec3<f32>(0.5, 0.8, 0.6));
        let h_key = normalize(view_dir + l_key);
        let n_dot_h_key = max(dot(n, h_key), 0.0);
        let n_dot_l_key = max(dot(n, l_key), 0.0);
        let sin2_key = 1.0 - n_dot_h_key * n_dot_h_key;
        let D_key = (2.0 + inv_r) / (2.0 * 3.14159) * pow(max(sin2_key, 1e-6), 0.5 * inv_r);
        let V_key = 1.0 / (4.0 * (n_dot_l_key + n_dot_v_s - n_dot_l_key * n_dot_v_s) + 0.001);
        color += sheen_col * D_key * V_key * n_dot_l_key * vec3<f32>(3.0, 2.9, 2.7);

        // Sheen on fill light
        let l_fill = normalize(vec3<f32>(-0.4, -0.3, 0.5));
        let h_fill = normalize(view_dir + l_fill);
        let n_dot_h_fill = max(dot(n, h_fill), 0.0);
        let n_dot_l_fill = max(dot(n, l_fill), 0.0);
        let sin2_fill = 1.0 - n_dot_h_fill * n_dot_h_fill;
        let D_fill = (2.0 + inv_r) / (2.0 * 3.14159) * pow(max(sin2_fill, 1e-6), 0.5 * inv_r);
        let V_fill = 1.0 / (4.0 * (n_dot_l_fill + n_dot_v_s - n_dot_l_fill * n_dot_v_s) + 0.001);
        color += sheen_col * D_fill * V_fill * n_dot_l_fill * vec3<f32>(0.3, 0.35, 0.5);

        // Sheen on environment: approximate with irradiance * sheen color
        // Charlie distribution at high roughness converges to diffuse
        color += sheen_col * irradiance * mix(0.3, 1.0, sheen_r);
    }

    // --- KHR_materials_specular (modulate F0) ---
    // specularColorFactor modulates the dielectric F0. For [0,0,0] this
    // completely suppresses specular (pure fabric). For [0.1,0.34,1] it
    // tints specular blue. Applied as a post-multiplier on the specular
    // components of both direct and ambient lighting.
    if pbr.specular_color.w > 0.5 {
        let spec_mod = pbr.specular_color.rgb;
        // Scale down the direct + ambient specular by the specular color factor.
        // The diffuse components are unaffected.
        let total_before = direct.color + fill.color + ambient;
        let diffuse_only = base.rgb * irradiance * (1.0 - metallic);
        let specular_part = total_before - diffuse_only;
        color = diffuse_only + specular_part * spec_mod;
    }

    // --- Emissive ---
    var emissive = pbr.emissive_factor.rgb;
    if pbr.has_emissive_tex > 0.5 {
        let em_tex = textureSample(emissive_tex, emissive_sampler, in.tex_uv);
        let em_linear = pow(em_tex.rgb, vec3<f32>(2.2)); // sRGB → linear
        emissive = emissive * em_linear;
    }
    color = color + emissive;

    // Tone mapping (Reinhard)
    color = color / (color + vec3<f32>(1.0));
    // Gamma correction
    color = pow(color, vec3<f32>(1.0 / 2.2));

    return vec4<f32>(color, alpha * in.fade);
}
