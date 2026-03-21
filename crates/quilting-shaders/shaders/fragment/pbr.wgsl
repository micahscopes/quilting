#define_import_path quilting::fragment::pbr

#import quilting::lighting::pbr::{pbr_direct, pbr_ambient, PBRInput}

struct PbrUniforms {
    base_color: vec4<f32>,
    metallic: f32,
    roughness: f32,
    _pad0: f32,
    _pad1: f32,
}

@group(0) @binding(1)
var<uniform> pbr: PbrUniforms;

struct FragInput {
    @location(0) normal_vs: vec3<f32>,
    @location(1) density: f32,
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

    // Key light — bright, from upper-right
    let light_dir = normalize(vec3<f32>(0.5, 0.8, 0.6));
    let light_color = vec3<f32>(3.0, 2.9, 2.7);

    let input = PBRInput(
        pbr.base_color.rgb,
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
        pbr.base_color.rgb,
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
        pbr.base_color.rgb,
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

    return vec4<f32>(color, pbr.base_color.a);
}
