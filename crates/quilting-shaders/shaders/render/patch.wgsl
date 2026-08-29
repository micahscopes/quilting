#import quilting::render::patch_vertex::{PatchPbrMaterial, PatchRenderFrame, PatchVertexOutput, evaluate_prepared_patch_vertex, shade_patch_lod, shade_patch_matcap, shade_patch_normals, shade_patch_pbr, shade_patch_stretch, shade_patch_wire}
#import quilting::surface::patch_prepare::PreparedPatchRecord
#import quilting::compute::visibility_compaction_types::CompactedBatchRangeRecord

struct DrawBatchIndex {
    batch_index: u32,
    _padding_a: u32,
    _padding_b: u32,
    _padding_c: u32,
}

// A real extracted scene may assign a distinct conformal map to every batch.
// Keep those immutable-for-the-frame records in one device table and select
// them with the same portable batch index used for compacted ranges. A single
// uniform here would make queue writes race all draws in one submission.
@group(0) @binding(0) var<storage, read> frames: array<PatchRenderFrame>;
@group(0) @binding(1) var<storage, read> prepared_records: array<PreparedPatchRecord>;
@group(0) @binding(2) var<storage, read> compacted_sources: array<u32>;
@group(0) @binding(3) var<storage, read> compacted_ranges: array<CompactedBatchRangeRecord>;
@group(0) @binding(4) var<uniform> draw_batch: DrawBatchIndex;
@group(0) @binding(5) var<storage, read> pbr_materials: array<PatchPbrMaterial>;

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

const PBR_BASE_COLOR_TEXTURE_BIT: u32 = 1u << 0u;
const PBR_METALLIC_ROUGHNESS_TEXTURE_BIT: u32 = 1u << 1u;
const PBR_NORMAL_TEXTURE_BIT: u32 = 1u << 2u;
const PBR_EMISSIVE_TEXTURE_BIT: u32 = 1u << 3u;
const PBR_OCCLUSION_TEXTURE_BIT: u32 = 1u << 4u;

fn pbr_rotated_uv(uv: vec2<f32>, scale: vec2<f32>, rotation: f32) -> vec2<f32> {
    let cosine = cos(rotation);
    let sine = sin(rotation);
    return vec2<f32>(
        (uv.x * cosine - uv.y * sine) * scale.x,
        (uv.x * sine + uv.y * cosine) * scale.y,
    );
}

// Reconstruct a world-space cotangent frame from the actually rasterized
// conformal surface. This keeps image-based lighting aligned with a normal map
// without spending two additional inter-stage locations on world tangents.
fn pbr_mapped_world_normal(
    geometric_normal: vec3<f32>,
    position_ws: vec3<f32>,
    uv: vec2<f32>,
    normal_texel: vec4<f32>,
    normal_scale: f32,
    enabled: bool,
) -> vec3<f32> {
    let normal = normalize(geometric_normal);
    let position_dx = dpdx(position_ws);
    let position_dy = dpdy(position_ws);
    let uv_dx = dpdx(uv);
    let uv_dy = dpdy(uv);
    let position_dy_perp = cross(position_dy, normal);
    let position_dx_perp = cross(normal, position_dx);
    let tangent = position_dy_perp * uv_dx.x + position_dx_perp * uv_dy.x;
    let bitangent = position_dy_perp * uv_dx.y + position_dx_perp * uv_dy.y;
    let basis_scale_squared = max(dot(tangent, tangent), dot(bitangent, bitangent));
    if !enabled || basis_scale_squared <= 1e-14 {
        return normal;
    }
    let inverse_basis_scale = inverseSqrt(basis_scale_squared);
    var tangent_normal = normal_texel.xyz * 2.0 - vec3<f32>(1.0);
    tangent_normal.x = tangent_normal.x * normal_scale;
    tangent_normal.y = tangent_normal.y * normal_scale;
    return normalize(
        tangent * (tangent_normal.x * inverse_basis_scale)
        + bitangent * (tangent_normal.y * inverse_basis_scale)
        + normal * tangent_normal.z,
    );
}

struct PatchVertexInput {
    @location(0) bary: vec3<f32>,
}

@vertex
fn render_patch_vertex(
    input: PatchVertexInput,
    @builtin(instance_index) local_instance: u32,
) -> PatchVertexOutput {
    let range = compacted_ranges[draw_batch.batch_index];
    let compacted_index = range.compacted_first_instance + local_instance;
    let source_instance = compacted_sources[compacted_index];
    return evaluate_prepared_patch_vertex(
        frames[draw_batch.batch_index],
        input.bary,
        prepared_records[source_instance],
    );
}

@fragment
fn render_patch_normals(
    @builtin(front_facing) front_facing: bool,
    input: PatchVertexOutput,
) -> @location(0) vec4<f32> {
    return shade_patch_normals(front_facing, input);
}

@fragment
fn render_patch_lod(input: PatchVertexOutput) -> @location(0) vec4<f32> {
    return shade_patch_lod(input);
}

@fragment
fn render_patch_stretch(input: PatchVertexOutput) -> @location(0) vec4<f32> {
    return shade_patch_stretch(input);
}

@fragment
fn render_patch_matcap(input: PatchVertexOutput) -> @location(0) vec4<f32> {
    return shade_patch_matcap(input);
}

@fragment
fn render_patch_wire(input: PatchVertexOutput) -> @location(0) vec4<f32> {
    return shade_patch_wire(input);
}

@fragment
fn render_patch_pbr(
    @builtin(front_facing) front_facing: bool,
    input: PatchVertexOutput,
) -> @location(0) vec4<f32> {
    let material_index = u32(max(frames[draw_batch.batch_index].modes.z, 0));
    let material = pbr_materials[material_index];
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
    let roughness = clamp(material.surface.x * metallic_roughness_texel.g, 0.04, 1.0);
    var normal_ws = normalize(input.normal_ws);
    if material.flags.y != 0u && !front_facing {
        normal_ws = -normal_ws;
    }
    normal_ws = pbr_mapped_world_normal(
        normal_ws,
        input.position_ws,
        normal_uv,
        normal_texel,
        material.specular_color_normal_scale.w,
        has_normal_texture,
    );
    let camera_delta_ws = input.camera_pos_ws_and_style.xyz - input.position_ws;
    let camera_distance_ws = length(camera_delta_ws);
    let view_dir_ws = select(
        vec3<f32>(0.0, 0.0, 1.0),
        camera_delta_ws / max(camera_distance_ws, 1e-7),
        camera_distance_ws > 1e-7,
    );
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
        frames[draw_batch.batch_index].modes.w,
        base_texel,
        metallic_roughness_texel,
        normal_texel,
        emissive_texel,
        occlusion_texel,
        has_normal_texture,
        pbr_environment.resident != 0u,
        irradiance,
        environment_color,
        normal_ws,
        view_dir_ws,
    );
}
