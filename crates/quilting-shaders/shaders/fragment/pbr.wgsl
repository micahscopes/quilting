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
    double_sided: f32,                   // >0.5 = flip normal for back-facing fragments
    debug_output: f32,                   // >0.5 = output normals as RGB instead of lit color
    // KHR_materials_sheen
    sheen_color: vec4<f32>,              // rgb in xyz, w = has_sheen (>0.5)
    sheen_roughness: f32,
    // KHR_materials_specular
    specular_color: vec4<f32>,           // rgb in xyz, w = has_specular (>0.5)
    // KHR_texture_transform for normal map
    normal_uv_transform: vec4<f32>,     // xy = scale, zw = offset
    normal_uv_rotation: f32,
    // KHR_texture_transform for base color
    base_uv_scale_x: f32,
    base_uv_scale_y: f32,
    base_uv_rotation: f32,
    // KHR_materials_ior + transmission + volume
    ior: f32,
    transmission_factor: f32,
    thickness_factor: f32,
    has_transmission_tex: f32,
    attenuation_color: vec3<f32>,
    attenuation_distance: f32,
    selection_tint: vec4<f32>,           // rgb + subtle post-lighting blend amount
    focus_sphere: vec4<f32>,             // source-space center xyz + radius
    focus_field_params: vec4<f32>,       // x=enabled; yzw reserved
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

// Sheen E LUT (directional albedo of Charlie distribution)
@group(0) @binding(16)
var sheen_e_lut: texture_2d<f32>;
@group(0) @binding(17)
var sheen_e_sampler: sampler;

// Environment cubemaps for IBL
@group(0) @binding(12)
var env_prefiltered: texture_cube<f32>;
@group(0) @binding(13)
var env_prefiltered_sampler: sampler;

@group(0) @binding(14)
var env_irradiance: texture_cube<f32>;
@group(0) @binding(15)
var env_irradiance_sampler: sampler;

// Screen-space scene color for transmission refraction
@group(0) @binding(18)
var scene_color_tex: texture_2d<f32>;
@group(0) @binding(19)
var scene_color_sampler: sampler;

// Gaussian-blurred scene color for rough transmission
@group(0) @binding(20)
var scene_color_blurred: texture_2d<f32>;
@group(0) @binding(21)
var scene_color_blurred_sampler: sampler;

// Per-pixel transmission modulator texture (R channel)
@group(0) @binding(22)
var transmission_tex: texture_2d<f32>;
@group(0) @binding(23)
var transmission_tex_sampler: sampler;

struct FragInput {
    @builtin(position) frag_coord: vec4<f32>,
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
    @location(12) mobius_stretch: f32,
    @location(13) source_position_ws: vec3<f32>,
}

struct PbrOutput {
    @location(0) color: vec4<f32>,
    @location(1) weight: vec4<f32>,
}

@fragment
fn fs_pbr(@builtin(front_facing) front_facing: bool, in: FragInput) -> PbrOutput {
    if in.fade < 0.001 { discard; }

    // DoF depth: log-encoded view-space distance.
    // position_vs = mv * post_mobius_pos, so length(position_vs) is the apparent distance
    // from the viewer to wherever the geometry ended up after Möbius warp.
    // Log encoding handles the wide distance range under Möbius (0.03 to 32 units → [0,1]).
    // 0.5 = 1 unit distance. Matches the log-space of stretch encoding.
    let dof_dist = max(length(in.position_vs), 0.001);
    let dof_depth = clamp(log2(dof_dist) / 10.0 + 0.5, 0.0, 1.0);
    let focus_radius = max(pbr.focus_sphere.w, 1e-4);
    let focus_radius_ratio = distance(in.source_position_ws, pbr.focus_sphere.xyz)
        / focus_radius;
    // Exact normalized geodesic polar coordinate of the round S3
    // compactification under stereographic projection:
    // origin=0, inversion sphere=1/2, infinity=1. Sphere inversion sends
    // this coordinate to 1-u.
    let focus_geodesic = 0.6366197723675814 * atan(focus_radius_ratio);
    let focus_field = select(
        0.0,
        focus_geodesic,
        pbr.focus_field_params.x > 0.5,
    );

    var n = normalize(in.normal_vs);
    // Double-sided materials: flip normal for back-facing fragments so they light correctly.
    // Only when the material explicitly requests it — QB perm_parity handles orientation otherwise.
    if pbr.double_sided > 0.5 && !front_facing { n = -n; }
    let view_dir = vec3<f32>(0.0, 0.0, 1.0); // view space: camera looks along +Z axis

    // --- Base color ---
    // Apply KHR_texture_transform to base color UVs
    var base_uv = in.tex_uv;
    if pbr.base_uv_scale_x != 1.0 || pbr.base_uv_scale_y != 1.0 || pbr.base_uv_rotation != 0.0 {
        let s = vec2<f32>(pbr.base_uv_scale_x, pbr.base_uv_scale_y);
        let r = pbr.base_uv_rotation;
        let cr = cos(r); let sr_val = sin(r);
        base_uv = vec2<f32>(
            (in.tex_uv.x * cr - in.tex_uv.y * sr_val) * s.x,
            (in.tex_uv.x * sr_val + in.tex_uv.y * cr) * s.y,
        );
    }
    var base = pbr.base_color;
    if pbr.has_base_color_tex > 0.5 {
        let tex_color = textureSample(base_color_tex, base_color_sampler, base_uv);
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
        unlit_color = mix(unlit_color, pbr.selection_tint.rgb, pbr.selection_tint.a);
        return PbrOutput(vec4<f32>(unlit_color, alpha), vec4<f32>(in.mobius_stretch, dof_depth, focus_field, 1.0));
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
        let raw_b = in.bitangent_vs;
        // Guard against degenerate tangents (NaN/zero from stretched Möbius faces)
        if dot(raw_t, raw_t) > 1e-6 {
            var t = normalize(raw_t);
            t = normalize(t - n * dot(t, n));  // Gram-Schmidt against N
            if dot(t, t) > 0.5 {               // guard post-projection
                // Use vertex-computed bitangent to preserve UV handedness.
                // cross(n, t) always produces a right-handed frame, but UV mirroring
                // (e.g. V-only flip) requires the bitangent to match the actual UV layout.
                var b = cross(n, t); // fallback
                if dot(raw_b, raw_b) > 1e-6 {
                    // Project vertex bitangent onto plane perpendicular to N and T
                    var bv = normalize(raw_b);
                    bv = bv - n * dot(bv, n);
                    bv = bv - t * dot(bv, t);
                    if dot(bv, bv) > 0.01 {
                        b = normalize(bv);
                    }
                }
                let tbn = mat3x3<f32>(t, b, n);

                let nm = textureSample(normal_tex, normal_sampler, normal_uv).xyz;
                var tangent_n = nm * 2.0 - vec3<f32>(1.0);
                tangent_n.x = tangent_n.x * pbr.normal_scale;
                tangent_n.y = tangent_n.y * pbr.normal_scale;
                n = normalize(tbn * tangent_n);
            }
        }
    }

    // --- Debug: output final computed normal as RGB ---
    if pbr.debug_output > 0.5 {
        let dbg = n * 0.5 + 0.5;
        // Tint back-faces slightly red so winding issues are visible
        if !front_facing {
            return PbrOutput(vec4<f32>(dbg.r * 0.3 + 0.7, dbg.g * 0.3, dbg.b * 0.3, in.fade), vec4<f32>(in.mobius_stretch, dof_depth, focus_field, 1.0));
        }
        return PbrOutput(vec4<f32>(dbg, in.fade), vec4<f32>(in.mobius_stretch, dof_depth, focus_field, 1.0));
    }

    // --- KHR_materials_specular: modify F0 before BRDF ---
    // specularColorFactor multiplies the dielectric F0 (default 0.04).
    // For [0,0,0] this suppresses all specular. For [0.1,0.34,1] it tints blue.
    var specular_weight = 1.0;
    var f0_mod = vec3<f32>(1.0);
    if pbr.specular_color.w > 0.5 {
        f0_mod = pbr.specular_color.rgb;
        specular_weight = max(f0_mod.x, max(f0_mod.y, f0_mod.z));
    }

    // Compute F0: IOR-based for dielectrics, albedo for metals
    let ior_f0 = pow((pbr.ior - 1.0) / (pbr.ior + 1.0), 2.0);
    let f0_base = mix(vec3<f32>(ior_f0) * f0_mod, base.rgb, metallic);

    // --- Lighting ---
    let light_dir = normalize(vec3<f32>(0.5, 0.8, 0.6));
    let light_color = vec3<f32>(3.0, 2.9, 2.7);
    let n_dot_v = max(dot(n, view_dir), 0.001);

    let input = PBRInput(base.rgb, metallic, roughness, n, view_dir, light_dir, light_color, f0_base);
    let direct = pbr_direct(input);

    let fill_input = PBRInput(base.rgb, metallic, roughness, n, view_dir,
        normalize(vec3<f32>(-0.4, -0.3, 0.5)), vec3<f32>(0.3, 0.35, 0.5), f0_base);
    let fill = pbr_direct(fill_input);

    // --- IBL ambient ---
    var ambient = vec3<f32>(0.0);
    var irradiance = vec3<f32>(0.0);
    if pbr.has_env_map > 0.5 {
        var n_ws = normalize(in.normal_ws);
        let view_dir_ws = normalize(in.camera_pos_ws - in.position_ws);
        if dot(n_ws, view_dir_ws) < 0.0 {
            n_ws = -n_ws;
        }
        let reflect_ws = reflect(-view_dir_ws, n_ws);

        irradiance = textureSampleLevel(env_irradiance, env_irradiance_sampler, n_ws, 0.0).rgb;
        let lod = roughness * max(pbr.env_mip_count, 1.0);
        let env_color = textureSampleLevel(env_prefiltered, env_prefiltered_sampler, reflect_ws, lod).rgb;

        ambient = pbr_ambient(base.rgb, metallic, roughness, n, view_dir, irradiance, env_color, f0_base);

        if pbr.has_occlusion_tex > 0.5 {
            let ao = textureSample(occlusion_tex, occlusion_sampler, in.tex_uv).r;
            ambient = ambient * mix(1.0, ao, pbr.occlusion_strength);
        }
    } else {
        // Analytical ambient fallback (no IBL) — hemisphere light
        let sky = vec3<f32>(0.25, 0.28, 0.40);
        let ground = vec3<f32>(0.10, 0.08, 0.06);
        let hemisphere = mix(ground, sky, dot(n, vec3<f32>(0.0, 1.0, 0.0)) * 0.5 + 0.5);
        ambient = base.rgb * hemisphere * (1.0 - metallic);
        irradiance = hemisphere;
    }

    var color = direct.color + fill.color + ambient;

    // Sheen materials: the specular F0 is already modified by specularColorFactor
    // (which is [0,0,0] for most fabric variants), so pbr_direct/ambient naturally
    // produce near-zero specular. No additional suppression needed.

    // --- KHR_materials_sheen (velvet/fabric) ---
    // Charlie distribution, composited with energy conservation:
    // color = f_sheen + base_layer * (1 - max(sheenColor) * E_sheen)
    if pbr.sheen_color.w > 0.5 {
        let sheen_col = pbr.sheen_color.rgb;
        let sheen_r = max(pbr.sheen_roughness, 0.000001);
        let alpha_g = sheen_r * sheen_r;
        let inv_alpha = 1.0 / alpha_g;

        // lambdaSheen visibility function (Khronos reference numerical fit).
        // Much more accurate than Ashikhmin approximation — prevents over-bright sheen.
        let one_minus_alpha_sq = (1.0 - alpha_g) * (1.0 - alpha_g);

        // V_Sheen = 1 / ((1 + lambda(NdotV) + lambda(NdotL)) * 4 * NdotV * NdotL)
        // lambda(x) = exp(a/(1+b*x^c) + d*x + e) with coefficients interpolated by roughness
        let ls_a = mix(21.5473, 25.3245, one_minus_alpha_sq);
        let ls_b = mix(3.82987, 3.32435, one_minus_alpha_sq);
        let ls_c = mix(0.19823, 0.16801, one_minus_alpha_sq);
        let ls_d = mix(-1.97760, -1.27393, one_minus_alpha_sq);
        let ls_e = mix(-4.32054, -4.85967, one_minus_alpha_sq);

        // Charlie D function
        // Key light sheen
        let h_key = normalize(view_dir + light_dir);
        let n_dot_h_key = max(dot(n, h_key), 0.0);
        let n_dot_l_key = max(dot(n, light_dir), 0.001);
        let sin2_key = 1.0 - n_dot_h_key * n_dot_h_key;
        let D_key = (2.0 + inv_alpha) / (2.0 * 3.14159) * pow(max(sin2_key, 1e-6), 0.5 * inv_alpha);

        // lambdaSheen for V and L
        let lv = exp(ls_a / (1.0 + ls_b * pow(n_dot_v, ls_c)) + ls_d * n_dot_v + ls_e);
        let ll_key = exp(ls_a / (1.0 + ls_b * pow(n_dot_l_key, ls_c)) + ls_d * n_dot_l_key + ls_e);
        let V_key = clamp(1.0 / ((1.0 + lv + ll_key) * 4.0 * n_dot_v * n_dot_l_key), 0.0, 1.0);
        let f_sheen_key = sheen_col * D_key * V_key * n_dot_l_key * light_color;

        // Fill light sheen
        let l_fill = normalize(vec3<f32>(-0.4, -0.3, 0.5));
        let h_fill = normalize(view_dir + l_fill);
        let n_dot_h_fill = max(dot(n, h_fill), 0.0);
        let n_dot_l_fill = max(dot(n, l_fill), 0.001);
        let sin2_fill = 1.0 - n_dot_h_fill * n_dot_h_fill;
        let D_fill = (2.0 + inv_alpha) / (2.0 * 3.14159) * pow(max(sin2_fill, 1e-6), 0.5 * inv_alpha);
        let ll_fill = exp(ls_a / (1.0 + ls_b * pow(n_dot_l_fill, ls_c)) + ls_d * n_dot_l_fill + ls_e);
        let V_fill = clamp(1.0 / ((1.0 + lv + ll_fill) * 4.0 * n_dot_v * n_dot_l_fill), 0.0, 1.0);
        let f_sheen_fill = sheen_col * D_fill * V_fill * n_dot_l_fill * vec3<f32>(0.3, 0.35, 0.5);

        // Environment sheen: very subtle — fabric doesn't reflect much ambient
        let f_sheen_env = sheen_col * irradiance * sheen_r * 0.1;

        let f_sheen = f_sheen_key + f_sheen_fill + f_sheen_env;

        // Energy conservation: base layer scaling per Khronos spec
        let max_sheen = max(sheen_col.x, max(sheen_col.y, sheen_col.z));
        // Approximate E_sheen from the lambdaSheen at the view angle
        let e_v = 1.0 / (1.0 + lv);
        let albedo_sheen_scaling = 1.0 - max_sheen * e_v;

        color = f_sheen + color * albedo_sheen_scaling;
    }

    // --- Emissive ---
    var emissive = pbr.emissive_factor.rgb;
    if pbr.has_emissive_tex > 0.5 {
        let em_tex = textureSample(emissive_tex, emissive_sampler, in.tex_uv);
        let em_linear = pow(em_tex.rgb, vec3<f32>(2.2)); // sRGB → linear
        emissive = emissive * em_linear;
    }
    color = color + emissive;

    // --- KHR_materials_transmission + volume ---
    if pbr.transmission_factor > 0.0 {
        // Volume absorption (Beer-Lambert law)
        var volume_atten = vec3<f32>(1.0);
        if pbr.thickness_factor > 0.0 && pbr.attenuation_distance > 0.0 {
            let optical_depth = -pbr.thickness_factor / pbr.attenuation_distance;
            volume_atten = exp(vec3<f32>(optical_depth) * log(max(pbr.attenuation_color, vec3<f32>(0.001))));
        }

        // Screen-space transmission
        let screen_size = vec2<f32>(textureDimensions(scene_color_tex));
        let screen_uv = in.frag_coord.xy / screen_size;

        // Refraction offset only when volume thickness is specified
        var refract_uv = screen_uv;
        if pbr.thickness_factor > 0.0 {
            let ior_offset = (pbr.ior - 1.0) * 0.02 * pbr.thickness_factor;
            refract_uv = clamp(screen_uv + n.xy * ior_offset, vec2<f32>(0.001), vec2<f32>(0.999));
        }

        // Variable blur via Gaussian mip pyramid: each mip = blurrier
        // Cap at LOD 4 to prevent background bleeding into the blur
        let blur_lod = roughness * roughness * 4.0;
        let scene_behind = textureSampleLevel(scene_color_tex, scene_color_sampler, refract_uv, blur_lod).rgb;

        // Convert scene from sRGB (tone-mapped output) back to linear for correct blending
        let scene_linear = pow(max(scene_behind, vec3<f32>(0.0)), vec3<f32>(2.2));
        // Tint by base color + volume absorption in linear space
        let transmitted = scene_linear * base.rgb * volume_atten;

        // Fresnel controls reflection vs transmission balance
        let n_dot_v_t = max(dot(n, view_dir), 0.001);
        let ior_f0_t = pow((pbr.ior - 1.0) / (pbr.ior + 1.0), 2.0);
        let fresnel_t = ior_f0_t + (1.0 - ior_f0_t) * pow(1.0 - n_dot_v_t, 5.0);
        var t_factor = pbr.transmission_factor * (1.0 - metallic);
        // Per-pixel transmission texture (creates patterns like cross-hatch)
        if pbr.has_transmission_tex > 0.5 {
            t_factor = t_factor * textureSample(transmission_tex, transmission_tex_sampler, in.tex_uv).r;
        }

        // Specular reflection (Fresnel) always on top.
        // Transmission replaces diffuse body; opaque keeps it.
        let reflection = color * fresnel_t;
        let body = mix(color * (1.0 - fresnel_t), transmitted, t_factor);
        color = reflection + body;
    }

    // Tone mapping: ACES filmic with slight exposure boost for deeper contrast
    let exposed = color;
    let a = exposed * 2.51 + vec3<f32>(0.03);
    let b = exposed * 2.43 + vec3<f32>(0.59);
    color = clamp((exposed * a) / (exposed * b + vec3<f32>(0.14)), vec3<f32>(0.0), vec3<f32>(1.0));
    // Gamma correction
    color = pow(color, vec3<f32>(1.0 / 2.2));
    color = mix(color, pbr.selection_tint.rgb, pbr.selection_tint.a);

    return PbrOutput(vec4<f32>(color, alpha * in.fade), vec4<f32>(in.mobius_stretch, dof_depth, focus_field, 1.0));
}
