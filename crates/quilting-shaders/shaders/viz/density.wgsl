#define_import_path quilting::viz::density

// Edge-based density interpolation and viridis colormap.

// Product-of-bary-coords weighting with a geometric mean in log space.
// Matches Rust's tri_edge_density formula without an avoidable exp/log pair
// when the caller wants the logarithmic visualization value.
fn edge_log2_density(bary: vec3<f32>, res: vec3<f32>) -> f32 {
    let x = bary.x;
    let y = bary.y;
    let z = bary.z;
    let e0 = y * z; // edge A (between verts y,z)
    let e1 = x * z; // edge B
    let e2 = x * y; // edge C
    let sum = e0 + e1 + e2;
    if sum < 1e-10 {
        return log2(max(res.x, max(res.y, res.z)));
    }
    return (e0 * log2(max(res.x, 1.0)) + e1 * log2(max(res.y, 1.0)) + e2 * log2(max(res.z, 1.0))) / sum;
}

fn edge_density(bary: vec3<f32>, res: vec3<f32>) -> f32 {
    return exp2(edge_log2_density(bary, res));
}

// Viridis-like colormap: dark purple → blue → teal → green → yellow
fn viridis(t_in: f32) -> vec3<f32> {
    let t = clamp(t_in, 0.0, 1.0);
    let c0 = vec3<f32>(0.27, 0.00, 0.33);
    let c1 = vec3<f32>(0.13, 0.37, 0.73);
    let c2 = vec3<f32>(0.12, 0.68, 0.55);
    let c3 = vec3<f32>(0.56, 0.85, 0.27);
    let c4 = vec3<f32>(0.99, 0.91, 0.15);
    if t < 0.25 { return mix(c0, c1, t * 4.0); }
    if t < 0.5  { return mix(c1, c2, (t - 0.25) * 4.0); }
    if t < 0.75 { return mix(c2, c3, (t - 0.5) * 4.0); }
    return mix(c3, c4, (t - 0.75) * 4.0);
}

// Magma colormap: black → purple → red → orange → yellow
fn magma(t_in: f32) -> vec3<f32> {
    let t = clamp(t_in, 0.0, 1.0);
    let c0 = vec3<f32>(0.00, 0.00, 0.02);
    let c1 = vec3<f32>(0.27, 0.01, 0.49);
    let c2 = vec3<f32>(0.72, 0.11, 0.40);
    let c3 = vec3<f32>(0.99, 0.45, 0.23);
    let c4 = vec3<f32>(0.99, 0.99, 0.75);
    if t < 0.25 { return mix(c0, c1, t * 4.0); }
    if t < 0.5  { return mix(c1, c2, (t - 0.25) * 4.0); }
    if t < 0.75 { return mix(c2, c3, (t - 0.5) * 4.0); }
    return mix(c3, c4, (t - 0.75) * 4.0);
}
