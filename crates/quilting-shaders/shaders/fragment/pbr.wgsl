#define_import_path quilting::fragment::pbr

#import quilting::lighting::pbr::{pbr_direct, pbr_ambient, PBRInput}

struct PbrUniforms {
    base_color: vec4<f32>,
    metallic: f32,
    roughness: f32,
    has_base_color_tex: f32,
    _pad1: f32,
}

@group(0) @binding(1)
var<uniform> pbr: PbrUniforms;

@group(0) @binding(2)
var base_color_tex: texture_2d<f32>;
@group(0) @binding(3)
var base_color_sampler: sampler;

struct FragInput {
    @location(0) normal_vs: vec3<f32>,
    @location(1) density: f32,
    @location(2) tex_uv: vec2<f32>,
}

@fragment
fn fs_pbr(in: FragInput) -> @location(0) vec4<f32> {
    var n = normalize(in.normal_vs);
    let view_dir = vec3<f32>(0.0, 0.0, 1.0); // view space: camera at +Z

    // Two-sided lighting: flip normal if it points away from the camera.
    // This handles inconsistent winding from glTF models and QB patches.
    if dot(n, view_dir) < 0.0 {
        n = -n;
    }

    // Sample base color: texture * factor when texture is present
    var base = pbr.base_color;
    if pbr.has_base_color_tex > 0.5 {
        let tex_color = textureSample(base_color_tex, base_color_sampler, in.tex_uv);
        // glTF spec: base color texture is sRGB, convert to linear for PBR
        let linear_rgb = pow(tex_color.rgb, vec3<f32>(2.2));
        base = vec4<f32>(linear_rgb * pbr.base_color.rgb, tex_color.a * pbr.base_color.a);
    }

    // Key light — bright, from upper-right
    let light_dir = normalize(vec3<f32>(0.5, 0.8, 0.6));
    let light_color = vec3<f32>(3.0, 2.9, 2.7);

    let input = PBRInput(
        base.rgb,
        pbr.metallic,
        pbr.roughness,
        n,
        view_dir,
        light_dir,
        light_color,
    );
    let direct = pbr_direct(input);

    // Fill light (softer, from below-left)
    let fill_input = PBRInput(
        base.rgb,
        pbr.metallic,
        pbr.roughness,
        n,
        view_dir,
        normalize(vec3<f32>(-0.4, -0.3, 0.5)),
        vec3<f32>(0.3, 0.35, 0.5),
    );
    let fill = pbr_direct(fill_input);

    // Ambient
    let ambient = pbr_ambient(
        base.rgb,
        pbr.metallic,
        pbr.roughness,
        n,
        view_dir,
        vec3<f32>(0.15, 0.12, 0.18),
    );

    var color = direct.color + fill.color + ambient;

    // Tone mapping (Reinhard)
    color = color / (color + vec3<f32>(1.0));
    // Gamma correction
    color = pow(color, vec3<f32>(1.0 / 2.2));

    return vec4<f32>(color, base.a);
}
