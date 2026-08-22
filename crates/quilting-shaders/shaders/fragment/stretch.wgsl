// Möbius stretch heatmap: red shift (expanded) vs blue shift (squashed).
// mobius_stretch encoded as [0,1]: 0.5 = neutral, <0.5 = squash, >0.5 = expand.

struct FragInput {
    @location(0) normal_vs: vec3<f32>,
    @location(1) density: f32,
    @location(9) fade: f32,
    @location(12) mobius_stretch: f32,
}

@fragment
fn fs_stretch(in: FragInput) -> @location(0) vec4<f32> {
    if in.fade < 0.001 { discard; }
    // Decode to signed: negative = squash, positive = expand
    let s = (in.mobius_stretch - 0.5) * 2.0;
    // Red shift for expansion, blue shift for compression
    let expand = max(s, 0.0);    // 0 to 1
    let squash = max(-s, 0.0);   // 0 to 1
    // Keep neutral geometry legible against the viewer background while the
    // two signed extremes approach saturated red and blue.
    let r = 0.25 + expand * 0.75;
    let g = 0.25 * (1.0 - max(expand, squash) * 0.7);
    let b = 0.25 + squash * 0.75;
    return vec4<f32>(r, g, b, in.fade);
}
