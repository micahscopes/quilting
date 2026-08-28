#define_import_path quilting::render::patch_vertex

#import quilting::surface::patch_prepare::PreparedPatchRecord
#import quilting::surface::patch_render::{PatchRenderTransform, PatchSurfaceInput, evaluate_patch_surface}
#import quilting::lighting::matcap::{matcap_shade, procedural_matcap}
#import quilting::lighting::pbr::{PBRInput, pbr_direct}

struct PatchRenderFrame {
    mvp: mat4x4<f32>,
    mv: mat4x4<f32>,
    // x = use rational QB; y = procedural matcap style; z/w reserved.
    modes: vec4<i32>,
    mob_a: vec4<f32>,
    mob_b: vec4<f32>,
    mob_c: vec4<f32>,
    mob_d: vec4<f32>,
    camera_pos: vec4<f32>,
}

struct PatchVertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) normal_vs: vec3<f32>,
    @location(1) density: f32,
    @location(2) tex_uv: vec2<f32>,
    @location(3) position_vs: vec3<f32>,
    @location(4) tangent_vs: vec3<f32>,
    @location(5) bitangent_vs: vec3<f32>,
    @location(6) normal_ws: vec3<f32>,
    @location(7) position_ws: vec3<f32>,
    // xyz = world-space camera; w = procedural matcap style. Packing the
    // frame-global scalar here preserves the WebGPU minimum inter-stage
    // variable limit without allocating another binding.
    @location(8) camera_pos_ws_and_style: vec4<f32>,
    @location(9) fade: f32,
    @location(10) tess_bary: vec3<f32>,
    @location(11) instance_id: f32,
    @location(12) mobius_stretch: f32,
    @location(13) source_position_ws: vec3<f32>,
    @location(14) @interpolate(flat) node_id: f32,
}

// Texture-free authored PBR values. Texture references and environment maps
// are backend resources and deliberately live outside this stable record.
struct PatchPbrMaterial {
    base_color: vec4<f32>,
    emissive_metallic: vec4<f32>,
    // x=roughness, y=alpha cutoff, z=alpha mode, w=IOR.
    surface: vec4<f32>,
    // x=unlit, y=double-sided, z=has-specular, w=has-sheen.
    flags: vec4<u32>,
    specular_color: vec4<f32>,
    // xyz=sheen color, w=sheen roughness.
    sheen_color_roughness: vec4<f32>,
}

fn evaluate_prepared_patch_vertex(
    frame: PatchRenderFrame,
    bary: vec3<f32>,
    record: PreparedPatchRecord,
) -> PatchVertexOutput {
    let surface = evaluate_patch_surface(
        PatchRenderTransform(
            frame.mvp,
            frame.mv,
            frame.modes.x,
            frame.mob_a,
            frame.mob_b,
            frame.mob_c,
            frame.mob_d,
            frame.camera_pos,
        ),
        PatchSurfaceInput(
            bary,
            record.record_position_a,
            record.record_position_b,
            record.record_position_c,
            record.record_weight_a,
            record.record_weight_b,
            record.record_weight_c,
            record.record_lod_info,
            record.record_vertex_lod,
            record.record_uv_ab,
            record.record_uv_c_prepare,
            record.record_normal_a,
            record.record_normal_b,
            record.record_normal_c,
            1u,
        ),
    );
    return PatchVertexOutput(
        surface.clip_pos,
        surface.normal_vs,
        surface.density,
        surface.tex_uv,
        surface.position_vs,
        surface.tangent_vs,
        surface.bitangent_vs,
        surface.normal_ws,
        surface.position_ws,
        vec4<f32>(surface.camera_pos_ws, f32(frame.modes.y)),
        surface.fade,
        surface.tess_bary,
        surface.instance_id,
        surface.mobius_stretch,
        surface.source_position_ws,
        surface.node_id,
    );
}

fn shade_patch_normals(front_facing: bool, input: PatchVertexOutput) -> vec4<f32> {
    if input.fade < 0.001 {
        discard;
    }
    let normal = normalize(input.normal_vs);
    let rgb = normal * 0.5 + 0.5;
    if !front_facing {
        return vec4<f32>(rgb.r * 0.3 + 0.7, rgb.g * 0.3, rgb.b * 0.3, input.fade);
    }
    return vec4<f32>(rgb, input.fade);
}

fn patch_lod_heatmap(t_in: f32) -> vec3<f32> {
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

fn shade_patch_lod(input: PatchVertexOutput) -> vec4<f32> {
    if input.fade < 0.001 {
        discard;
    }
    var normal = normalize(input.normal_vs);
    if normal.z < 0.0 {
        normal = -normal;
    }
    return vec4<f32>(matcap_shade(normal, patch_lod_heatmap(input.density)), input.fade);
}

fn shade_patch_wire(input: PatchVertexOutput) -> vec4<f32> {
    if input.fade < 0.001 {
        discard;
    }
    return vec4<f32>(patch_lod_heatmap(input.density), input.fade);
}

fn shade_patch_stretch(input: PatchVertexOutput) -> vec4<f32> {
    if input.fade < 0.001 {
        discard;
    }
    let stretch = (input.mobius_stretch - 0.5) * 2.0;
    let expand = max(stretch, 0.0);
    let squash = max(-stretch, 0.0);
    let extreme = max(expand, squash);
    return vec4<f32>(
        0.25 + expand * 0.75,
        0.25 * (1.0 - extreme * 0.7),
        0.25 + squash * 0.75,
        input.fade,
    );
}

fn shade_patch_matcap(input: PatchVertexOutput) -> vec4<f32> {
    if input.fade < 0.001 {
        discard;
    }
    var normal = normalize(input.normal_vs);
    if normal.z < 0.0 {
        normal = -normal;
    }
    return vec4<f32>(procedural_matcap(normal, input.camera_pos_ws_and_style.w), input.fade);
}

fn shade_patch_pbr(
    front_facing: bool,
    input: PatchVertexOutput,
    material: PatchPbrMaterial,
    selected_node: i32,
) -> vec4<f32> {
    if input.fade < 0.001 {
        discard;
    }
    let double_sided = material.flags.y != 0u;
    if !double_sided && !front_facing {
        discard;
    }

    var normal = normalize(input.normal_vs);
    if double_sided && !front_facing {
        normal = -normal;
    }
    let base = material.base_color;
    let alpha_mode = material.surface.z;
    if alpha_mode > 0.5 && alpha_mode < 1.5 && base.a < material.surface.y {
        discard;
    }
    var alpha = base.a;
    if alpha_mode < 0.5 {
        alpha = 1.0;
    }

    let selection_amount = select(
        0.0,
        0.13,
        selected_node >= 0 && abs(input.node_id - f32(selected_node)) < 0.5,
    );
    let selection_tint = vec3<f32>(0.16, 0.78, 1.0);
    if material.flags.x != 0u {
        var unlit = pow(max(base.rgb, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));
        unlit = mix(unlit, selection_tint, selection_amount);
        return vec4<f32>(unlit, alpha * input.fade);
    }

    let metallic = material.emissive_metallic.w;
    let roughness = clamp(material.surface.x, 0.04, 1.0);
    var f0_mod = vec3<f32>(1.0);
    if material.flags.z != 0u {
        f0_mod = material.specular_color.rgb;
    }
    let ior = material.surface.w;
    let ior_f0 = pow((ior - 1.0) / (ior + 1.0), 2.0);
    let f0 = mix(vec3<f32>(ior_f0) * f0_mod, base.rgb, metallic);
    let view_dir = vec3<f32>(0.0, 0.0, 1.0);
    let key_dir = normalize(vec3<f32>(0.5, 0.8, 0.6));
    let key_color = vec3<f32>(3.0, 2.9, 2.7);
    let key = pbr_direct(PBRInput(
        base.rgb, metallic, roughness, normal, view_dir, key_dir, key_color, f0,
    ));
    let fill = pbr_direct(PBRInput(
        base.rgb,
        metallic,
        roughness,
        normal,
        view_dir,
        normalize(vec3<f32>(-0.4, -0.3, 0.5)),
        vec3<f32>(0.3, 0.35, 0.5),
        f0,
    ));
    let sky = vec3<f32>(0.25, 0.28, 0.40);
    let ground = vec3<f32>(0.10, 0.08, 0.06);
    let hemisphere = mix(
        ground,
        sky,
        dot(normal, vec3<f32>(0.0, 1.0, 0.0)) * 0.5 + 0.5,
    );
    var color = key.color + fill.color
        + base.rgb * hemisphere * (1.0 - metallic)
        + material.emissive_metallic.rgb;
    let aces_a = color * 2.51 + vec3<f32>(0.03);
    let aces_b = color * 2.43 + vec3<f32>(0.59);
    color = clamp((color * aces_a) / (color * aces_b + vec3<f32>(0.14)), vec3<f32>(0.0), vec3<f32>(1.0));
    color = pow(color, vec3<f32>(1.0 / 2.2));
    color = mix(color, selection_tint, selection_amount);
    return vec4<f32>(color, alpha * input.fade);
}
