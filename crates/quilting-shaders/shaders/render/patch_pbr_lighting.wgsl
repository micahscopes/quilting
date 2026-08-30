#define_import_path quilting::render::patch_pbr_lighting

#import quilting::render::patch_vertex::{PatchPbrMaterial, PatchVertexOutput, shade_patch_pbr}
#import quilting::lighting::pbr::pbr_apply_world_tangent_normal

struct PbrEnvironmentUniform {
    resident: u32,
    prefiltered_mip_count: u32,
    _padding_a: u32,
    _padding_b: u32,
}

@group(2) @binding(0) var<uniform> pbr_environment: PbrEnvironmentUniform;
@group(2) @binding(1) var pbr_prefiltered_environment: texture_cube<f32>;
@group(2) @binding(2) var pbr_irradiance_environment: texture_cube<f32>;
@group(2) @binding(3) var pbr_environment_sampler: sampler;

fn pbr_rotated_uv(uv: vec2<f32>, scale: vec2<f32>, rotation: f32) -> vec2<f32> {
    let cosine = cos(rotation);
    let sine = sin(rotation);
    return vec2<f32>(
        (uv.x * cosine - uv.y * sine) * scale.x,
        (uv.x * sine + uv.y * cosine) * scale.y,
    );
}

fn shade_sampled_patch_pbr(
    front_facing: bool,
    input: PatchVertexOutput,
    material: PatchPbrMaterial,
    selected_node: i32,
    normal_uv: vec2<f32>,
    base_texel: vec4<f32>,
    metallic_roughness_texel: vec4<f32>,
    normal_texel: vec4<f32>,
    emissive_texel: vec4<f32>,
    occlusion_texel: vec4<f32>,
    has_normal_texture: bool,
) -> vec4<f32> {
    let roughness = clamp(material.surface.x * metallic_roughness_texel.g, 0.04, 1.0);
    var normal_ws = normalize(input.normal_ws);
    if material.flags.y != 0u && !front_facing {
        normal_ws = -normal_ws;
    }
    let camera_delta_ws = input.camera_pos_ws_and_style.xyz - input.position_ws;
    let camera_distance_ws = length(camera_delta_ws);
    let view_dir_ws = select(
        vec3<f32>(0.0, 0.0, 1.0),
        camera_delta_ws / max(camera_distance_ws, 1e-7),
        camera_distance_ws > 1e-7,
    );
    if dot(normal_ws, view_dir_ws) < 0.0 {
        normal_ws = -normal_ws;
    }
    normal_ws = pbr_apply_world_tangent_normal(
        normal_ws,
        input.position_ws,
        normal_uv,
        normal_texel,
        material.specular_color_normal_scale.w,
        has_normal_texture,
    );
    if dot(normal_ws, view_dir_ws) < 0.0 {
        normal_ws = -normal_ws;
    }
    let reflection_dir_ws = reflect(-view_dir_ws, normal_ws);
    var irradiance = vec3<f32>(0.0);
    var environment_color = vec3<f32>(0.0);
    if pbr_environment.resident != 0u {
        irradiance = textureSample(
            pbr_irradiance_environment,
            pbr_environment_sampler,
            normal_ws,
        ).rgb;
        let maximum_lod = f32(max(pbr_environment.prefiltered_mip_count, 1u) - 1u);
        environment_color = textureSampleLevel(
            pbr_prefiltered_environment,
            pbr_environment_sampler,
            reflection_dir_ws,
            roughness * maximum_lod,
        ).rgb;
    }
    return shade_patch_pbr(
        front_facing,
        input,
        material,
        selected_node,
        base_texel,
        metallic_roughness_texel,
        normal_texel,
        emissive_texel,
        occlusion_texel,
        has_normal_texture,
        pbr_environment.resident != 0u,
        irradiance,
        environment_color,
    );
}
