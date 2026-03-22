// Matcap fragment shader: samples matcap texture using view-space normal.
// Falls back to procedural matcap when no texture is bound.

#import quilting::lighting::matcap::matcap_shade

struct MatcapUniforms {
    has_matcap_tex: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(1)
var<uniform> matcap_u: MatcapUniforms;

@group(0) @binding(2)
var matcap_tex: texture_2d<f32>;
@group(0) @binding(3)
var matcap_sampler: sampler;

struct FragInput {
    @location(0) normal_vs: vec3<f32>,
    @location(1) density: f32,
}

// Heatmap colormap: blue -> cyan -> green -> yellow -> red
fn heatmap(t_in: f32) -> vec3<f32> {
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

@fragment
fn fs_matcap(in: FragInput) -> @location(0) vec4<f32> {
    var n = normalize(in.normal_vs);
    if n.z < 0.0 { n = -n; }

    // Matcap UV: map view-space normal to texture coordinates
    let uv = n.xy * 0.48 + 0.5;

    if matcap_u.has_matcap_tex > 0.5 {
        let col = textureSample(matcap_tex, matcap_sampler, uv);
        return vec4<f32>(col.rgb, 1.0);
    } else {
        let base = heatmap(in.density);
        let col = matcap_shade(n, base);
        return vec4<f32>(col, 1.0);
    }
}
