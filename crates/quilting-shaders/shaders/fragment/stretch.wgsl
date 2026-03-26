// Möbius stretch heatmap: visualize conformal distortion as a color gradient.
// Blue = no stretch (1.0), Green = moderate, Red = high stretch.

struct FragInput {
    @location(0) normal_vs: vec3<f32>,
    @location(1) density: f32,
    @location(9) fade: f32,
    @location(12) mobius_stretch: f32,
}

@fragment
fn fs_stretch(in: FragInput) -> @location(0) vec4<f32> {
    if in.fade < 0.001 { discard; }
    let s = clamp(in.mobius_stretch, 0.0, 1.0);
    // Heatmap: blue (0) → green (0.5) → red (1.0)
    let r = smoothstep(0.4, 0.8, s);
    let g = 1.0 - abs(s - 0.5) * 2.0;
    let b = 1.0 - smoothstep(0.0, 0.4, s);
    return vec4<f32>(r, g, b, in.fade);
}
