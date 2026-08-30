#define_import_path quilting::render::patch_pbr

#import quilting::render::patch_vertex::{PatchPbrMaterial, PatchVertexOutput}
#import quilting::render::patch_pbr_lighting::{pbr_rotated_uv, shade_sampled_patch_pbr}

@group(1) @binding(0) var pbr_base_color_texture: texture_2d<f32>;
@group(1) @binding(1) var pbr_base_color_sampler: sampler;
@group(1) @binding(2) var pbr_metallic_roughness_texture: texture_2d<f32>;
@group(1) @binding(3) var pbr_metallic_roughness_sampler: sampler;
@group(1) @binding(4) var pbr_normal_texture: texture_2d<f32>;
@group(1) @binding(5) var pbr_normal_sampler: sampler;
@group(1) @binding(6) var pbr_emissive_texture: texture_2d<f32>;
@group(1) @binding(7) var pbr_emissive_sampler: sampler;
@group(1) @binding(8) var pbr_occlusion_texture: texture_2d<f32>;
@group(1) @binding(9) var pbr_occlusion_sampler: sampler;
@group(1) @binding(10) var pbr_transmission_texture: texture_2d<f32>;
@group(1) @binding(11) var pbr_transmission_sampler: sampler;

const PBR_BASE_COLOR_TEXTURE_BIT: u32 = 1u << 0u;
const PBR_METALLIC_ROUGHNESS_TEXTURE_BIT: u32 = 1u << 1u;
const PBR_NORMAL_TEXTURE_BIT: u32 = 1u << 2u;
const PBR_EMISSIVE_TEXTURE_BIT: u32 = 1u << 3u;
const PBR_OCCLUSION_TEXTURE_BIT: u32 = 1u << 4u;

fn shade_textured_patch_pbr(
    front_facing: bool,
    input: PatchVertexOutput,
    material: PatchPbrMaterial,
    selected_node: i32,
) -> vec4<f32> {
    let texture_mask = material.flags.w >> 1u;
    let base_uv = pbr_rotated_uv(
        input.tex_uv,
        material.normal_occlusion_base_scale.zw,
        material.base_transmission_volume.x,
    );
    let normal_uv = pbr_rotated_uv(
        input.tex_uv,
        material.normal_uv_scale_offset.xy,
        material.normal_occlusion_base_scale.x,
    ) + material.normal_uv_scale_offset.zw;
    var base_texel = vec4<f32>(1.0);
    var metallic_roughness_texel = vec4<f32>(1.0);
    var normal_texel = vec4<f32>(0.5, 0.5, 1.0, 1.0);
    var emissive_texel = vec4<f32>(1.0);
    var occlusion_texel = vec4<f32>(1.0);
    if (texture_mask & PBR_BASE_COLOR_TEXTURE_BIT) != 0u {
        base_texel = textureSample(pbr_base_color_texture, pbr_base_color_sampler, base_uv);
    }
    if (texture_mask & PBR_METALLIC_ROUGHNESS_TEXTURE_BIT) != 0u {
        metallic_roughness_texel = textureSample(
            pbr_metallic_roughness_texture,
            pbr_metallic_roughness_sampler,
            input.tex_uv,
        );
    }
    let has_normal_texture = (texture_mask & PBR_NORMAL_TEXTURE_BIT) != 0u;
    if has_normal_texture {
        normal_texel = textureSample(pbr_normal_texture, pbr_normal_sampler, normal_uv);
    }
    if (texture_mask & PBR_EMISSIVE_TEXTURE_BIT) != 0u {
        emissive_texel = textureSample(pbr_emissive_texture, pbr_emissive_sampler, input.tex_uv);
    }
    if (texture_mask & PBR_OCCLUSION_TEXTURE_BIT) != 0u {
        occlusion_texel = textureSample(pbr_occlusion_texture, pbr_occlusion_sampler, input.tex_uv);
    }
    return shade_sampled_patch_pbr(
        front_facing,
        input,
        material,
        selected_node,
        normal_uv,
        base_texel,
        metallic_roughness_texel,
        normal_texel,
        emissive_texel,
        occlusion_texel,
        has_normal_texture,
    );
}
