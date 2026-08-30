#define_import_path quilting::render::patch_pbr_portable

#import quilting::render::patch_vertex::{PatchPbrMaterial, PatchVertexOutput}
#import quilting::render::patch_pbr_lighting::{pbr_rotated_uv, shade_sampled_patch_pbr}

struct PbrPortableTextureRecord {
    // xy = atlas origin, zw = exact source dimensions.
    origin_size: vec4<u32>,
    // x = array layer, y/z = S/T wrap mode, w = resident.
    layer_wrap_resident: vec4<u32>,
}

struct PbrPortableMaterialTextures {
    // base color, metallic/roughness, normal, emissive.
    slots_a: vec4<u32>,
    // occlusion, transmission, reserved, reserved.
    slots_b: vec4<u32>,
}

@group(1) @binding(0) var pbr_portable_atlas: texture_2d_array<f32>;
@group(1) @binding(1) var<storage, read> pbr_portable_textures: array<PbrPortableTextureRecord>;
@group(1) @binding(2) var<storage, read> pbr_portable_materials: array<PbrPortableMaterialTextures>;

const PBR_BASE_COLOR_TEXTURE_BIT: u32 = 1u << 0u;
const PBR_METALLIC_ROUGHNESS_TEXTURE_BIT: u32 = 1u << 1u;
const PBR_NORMAL_TEXTURE_BIT: u32 = 1u << 2u;
const PBR_EMISSIVE_TEXTURE_BIT: u32 = 1u << 3u;
const PBR_OCCLUSION_TEXTURE_BIT: u32 = 1u << 4u;

fn pbr_positive_mod(value: i32, modulus: i32) -> i32 {
    let remainder = value % modulus;
    return select(remainder, remainder + modulus, remainder < 0);
}

fn pbr_wrap_texel(value: i32, size: u32, mode: u32) -> u32 {
    let extent = i32(size);
    if mode == 0u {
        return u32(clamp(value, 0, extent - 1));
    }
    if mode == 1u {
        let period = extent * 2;
        let mirrored = pbr_positive_mod(value, period);
        return u32(select(mirrored, period - 1 - mirrored, mirrored >= extent));
    }
    return u32(pbr_positive_mod(value, extent));
}

fn pbr_load_portable_texel(
    record: PbrPortableTextureRecord,
    coordinate: vec2<i32>,
) -> vec4<f32> {
    let local = vec2<u32>(
        pbr_wrap_texel(coordinate.x, record.origin_size.z, record.layer_wrap_resident.y),
        pbr_wrap_texel(coordinate.y, record.origin_size.w, record.layer_wrap_resident.z),
    );
    return textureLoad(
        pbr_portable_atlas,
        vec2<i32>(record.origin_size.xy + local),
        i32(record.layer_wrap_resident.x),
        0,
    );
}

fn pbr_sample_portable_texture(
    slot: u32,
    uv: vec2<f32>,
    fallback: vec4<f32>,
) -> vec4<f32> {
    if slot >= arrayLength(&pbr_portable_textures) {
        return fallback;
    }
    let record = pbr_portable_textures[slot];
    if record.layer_wrap_resident.w == 0u
        || record.origin_size.z == 0u
        || record.origin_size.w == 0u
    {
        return fallback;
    }
    let texel = uv * vec2<f32>(record.origin_size.zw) - vec2<f32>(0.5);
    let lower = vec2<i32>(floor(texel));
    let blend = fract(texel);
    let lower_left = pbr_load_portable_texel(record, lower);
    let lower_right = pbr_load_portable_texel(record, lower + vec2<i32>(1, 0));
    let upper_left = pbr_load_portable_texel(record, lower + vec2<i32>(0, 1));
    let upper_right = pbr_load_portable_texel(record, lower + vec2<i32>(1, 1));
    return mix(
        mix(lower_left, lower_right, blend.x),
        mix(upper_left, upper_right, blend.x),
        blend.y,
    );
}

fn shade_portable_patch_pbr(
    front_facing: bool,
    input: PatchVertexOutput,
    material: PatchPbrMaterial,
    material_index: u32,
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
    let slots = pbr_portable_materials[material_index];
    var base_texel = vec4<f32>(1.0);
    var metallic_roughness_texel = vec4<f32>(1.0);
    var normal_texel = vec4<f32>(0.5, 0.5, 1.0, 1.0);
    var emissive_texel = vec4<f32>(1.0);
    var occlusion_texel = vec4<f32>(1.0);
    if (texture_mask & PBR_BASE_COLOR_TEXTURE_BIT) != 0u {
        base_texel = pbr_sample_portable_texture(slots.slots_a.x, base_uv, base_texel);
    }
    if (texture_mask & PBR_METALLIC_ROUGHNESS_TEXTURE_BIT) != 0u {
        metallic_roughness_texel = pbr_sample_portable_texture(
            slots.slots_a.y,
            input.tex_uv,
            metallic_roughness_texel,
        );
    }
    let has_normal_texture = (texture_mask & PBR_NORMAL_TEXTURE_BIT) != 0u;
    if has_normal_texture {
        normal_texel = pbr_sample_portable_texture(slots.slots_a.z, normal_uv, normal_texel);
    }
    if (texture_mask & PBR_EMISSIVE_TEXTURE_BIT) != 0u {
        emissive_texel = pbr_sample_portable_texture(
            slots.slots_a.w,
            input.tex_uv,
            emissive_texel,
        );
    }
    if (texture_mask & PBR_OCCLUSION_TEXTURE_BIT) != 0u {
        occlusion_texel = pbr_sample_portable_texture(
            slots.slots_b.x,
            input.tex_uv,
            occlusion_texel,
        );
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
