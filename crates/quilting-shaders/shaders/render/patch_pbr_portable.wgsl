#define_import_path quilting::render::patch_pbr_portable

#import quilting::render::patch_vertex::{PatchPbrMaterial, PatchVertexOutput}
#import quilting::render::patch_pbr_lighting::{pbr_rotated_uv, shade_sampled_patch_pbr}

struct PbrPortableTextureRecord {
    // xy = exact source dimensions, zw = S/T wrap mode.
    size_wrap: vec4<u32>,
    // x = mip count, y = placement offset, z = resident, w = reserved.
    mips_offset_resident: vec4<u32>,
}

struct PbrPortableMipPlacement {
    // xy = independently packed origin, z = array layer, w = stable slot.
    origin_layer_slot: vec4<u32>,
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
@group(1) @binding(3) var<storage, read> pbr_portable_mip_placements: array<PbrPortableMipPlacement>;

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
    placement: PbrPortableMipPlacement,
    coordinate: vec2<i32>,
    mip_level: u32,
) -> vec4<f32> {
    let mip_size = max(record.size_wrap.xy >> vec2<u32>(mip_level), vec2<u32>(1u));
    let local = vec2<u32>(
        pbr_wrap_texel(coordinate.x, mip_size.x, record.size_wrap.z),
        pbr_wrap_texel(coordinate.y, mip_size.y, record.size_wrap.w),
    );
    return textureLoad(
        pbr_portable_atlas,
        vec2<i32>(placement.origin_layer_slot.xy + local),
        i32(placement.origin_layer_slot.z),
        i32(mip_level),
    );
}

fn pbr_sample_portable_mip(
    record: PbrPortableTextureRecord,
    uv: vec2<f32>,
    mip_level: u32,
) -> vec4<f32> {
    let placement_index = record.mips_offset_resident.y + mip_level;
    let placement = pbr_portable_mip_placements[placement_index];
    let mip_size = max(record.size_wrap.xy >> vec2<u32>(mip_level), vec2<u32>(1u));
    let texel = uv * vec2<f32>(mip_size) - vec2<f32>(0.5);
    let lower = vec2<i32>(floor(texel));
    let blend = fract(texel);
    let lower_left = pbr_load_portable_texel(record, placement, lower, mip_level);
    let lower_right = pbr_load_portable_texel(
        record,
        placement,
        lower + vec2<i32>(1, 0),
        mip_level,
    );
    let upper_left = pbr_load_portable_texel(
        record,
        placement,
        lower + vec2<i32>(0, 1),
        mip_level,
    );
    let upper_right = pbr_load_portable_texel(
        record,
        placement,
        lower + vec2<i32>(1, 1),
        mip_level,
    );
    return mix(
        mix(lower_left, lower_right, blend.x),
        mix(upper_left, upper_right, blend.x),
        blend.y,
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
    if record.mips_offset_resident.z == 0u
        || record.mips_offset_resident.x == 0u
        || record.size_wrap.x == 0u
        || record.size_wrap.y == 0u
    {
        return fallback;
    }
    let placement_end = record.mips_offset_resident.y + record.mips_offset_resident.x;
    if placement_end > arrayLength(&pbr_portable_mip_placements) {
        return fallback;
    }
    let scaled_uv = uv * vec2<f32>(record.size_wrap.xy);
    let derivative_x = dpdx(scaled_uv);
    let derivative_y = dpdy(scaled_uv);
    let footprint = max(length(derivative_x), length(derivative_y));
    let maximum_lod = f32(record.mips_offset_resident.x - 1u);
    let lod = clamp(log2(max(footprint, 1.0)), 0.0, maximum_lod);
    let lower_mip = u32(floor(lod));
    let upper_mip = min(lower_mip + 1u, record.mips_offset_resident.x - 1u);
    let lower_sample = pbr_sample_portable_mip(record, uv, lower_mip);
    let upper_sample = pbr_sample_portable_mip(record, uv, upper_mip);
    return mix(lower_sample, upper_sample, fract(lod));
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
