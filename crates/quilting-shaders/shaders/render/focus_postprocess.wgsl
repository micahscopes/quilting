// WebGPU focus-composition passes. Rust owns the exact pass schedule; this
// module contains only one fullscreen vertex and the backend-local fragment
// implementations selected by that schedule.

struct FocusPassUniform {
    // xy = active source extent in pixels; z = JFA step; w = Kawase offset.
    extent_step_offset: vec4<f32>,
    // x = maximum distance; y = focal coordinate; z = bandwidth; w = mode.
    focus: vec4<f32>,
    // x = blur radius; y = per-subpass strength; z = final-pass flag;
    // w = normalize-range flag.
    blur: vec4<f32>,
    // xy = retained stretch minimum/maximum; zw reserved.
    stretch_range: vec4<f32>,
}

@group(0) @binding(0) var focus_source_a: texture_2d<f32>;
@group(0) @binding(1) var focus_sampler: sampler;
@group(0) @binding(2) var focus_source_b: texture_2d<f32>;
@group(0) @binding(3) var<uniform> focus_pass: FocusPassUniform;

struct FocusFullscreenOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn focus_fullscreen_vertex(@builtin(vertex_index) vertex_index: u32) -> FocusFullscreenOutput {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    let position = positions[vertex_index];
    return FocusFullscreenOutput(
        vec4<f32>(position, 0.0, 1.0),
        position * 0.5 + vec2<f32>(0.5),
    );
}

// Convert the raw PBR MRT payload (stretch, depth, spheroidal field) to the
// scalar consumed by JFA/firmness. CPU-provided stretch bounds avoid a
// readback when normalization is requested and a useful range is resident.
@fragment
fn focus_select_weight(input: FocusFullscreenOutput) -> @location(0) vec4<f32> {
    let data = textureSampleLevel(focus_source_a, focus_sampler, input.uv, 0.0);
    let mode = focus_pass.focus.w;
    var value = data.r;
    if mode < 0.5 {
        value = data.g;
    } else if mode < 1.5 {
        value = data.r;
    } else if mode < 2.5 {
        value = max(data.r, data.g);
    } else {
        value = data.b;
    }
    if focus_pass.blur.w > 0.5 && mode < 2.5 {
        let minimum = focus_pass.stretch_range.x;
        let maximum = focus_pass.stretch_range.y;
        value = (value - minimum) / max(maximum - minimum, 0.001);
    }
    return vec4<f32>(clamp(value, 0.0, 1.0), 0.0, 0.0, 1.0);
}

@fragment
fn focus_jfa_init(input: FocusFullscreenOutput) -> @location(0) vec4<f32> {
    let weight = textureSampleLevel(focus_source_a, focus_sampler, input.uv, 0.0).r;
    if weight > 0.001 {
        return vec4<f32>(input.uv, weight, 0.0);
    }
    return vec4<f32>(input.uv, 0.0, 0.0);
}

@fragment
fn focus_jfa_step(input: FocusFullscreenOutput) -> @location(0) vec4<f32> {
    let dimensions = vec2<i32>(textureDimensions(focus_source_a));
    let coordinates = clamp(
        vec2<i32>(input.uv * vec2<f32>(dimensions)),
        vec2<i32>(0),
        dimensions - vec2<i32>(1),
    );
    let pixel_uv = (vec2<f32>(coordinates) + vec2<f32>(0.5)) / vec2<f32>(dimensions);
    let step_size = max(i32(round(focus_pass.extent_step_offset.z)), 1);
    var best = textureLoad(focus_source_a, coordinates, 0);
    var best_distance = select(
        1e9,
        distance(pixel_uv * vec2<f32>(dimensions), best.xy * vec2<f32>(dimensions)),
        best.z > 0.0,
    );
    for (var dy = -1; dy <= 1; dy += 1) {
        for (var dx = -1; dx <= 1; dx += 1) {
            if dx == 0 && dy == 0 {
                continue;
            }
            let neighbor_coordinates = coordinates + vec2<i32>(dx, dy) * step_size;
            if any(neighbor_coordinates < vec2<i32>(0))
                || any(neighbor_coordinates >= dimensions) {
                continue;
            }
            let neighbor = textureLoad(focus_source_a, neighbor_coordinates, 0);
            if neighbor.z <= 0.0 {
                continue;
            }
            let neighbor_distance = distance(
                pixel_uv * vec2<f32>(dimensions),
                neighbor.xy * vec2<f32>(dimensions),
            );
            if neighbor_distance < best_distance - 0.5
                || (abs(neighbor_distance - best_distance) <= 0.5 && neighbor.z > best.z) {
                best = neighbor;
                best_distance = neighbor_distance;
            }
        }
    }
    return best;
}

@fragment
fn focus_firmness(input: FocusFullscreenOutput) -> @location(0) vec4<f32> {
    let mode = focus_pass.focus.w;
    if mode > 2.5 {
        let radial_coordinate = textureSampleLevel(
            focus_source_b,
            focus_sampler,
            input.uv,
            0.0,
        ).r;
        let angular_distance = abs(radial_coordinate - focus_pass.focus.y);
        let aperture = max(focus_pass.focus.z, 0.001);
        let circle_of_confusion = angular_distance / aperture;
        let defocus = circle_of_confusion / (1.0 + circle_of_confusion);
        return vec4<f32>(input.uv, defocus, 0.0);
    }

    let dimensions = max(focus_pass.extent_step_offset.xy, vec2<f32>(1.0));
    let texel = vec2<f32>(1.0) / dimensions;
    let sample_position = input.uv * dimensions - vec2<f32>(0.5);
    let base = floor(sample_position) * texel + texel * 0.5;
    let fraction = fract(sample_position);
    let j00 = textureSampleLevel(focus_source_a, focus_sampler, base, 0.0);
    let j10 = textureSampleLevel(focus_source_a, focus_sampler, base + vec2<f32>(texel.x, 0.0), 0.0);
    let j01 = textureSampleLevel(focus_source_a, focus_sampler, base + vec2<f32>(0.0, texel.y), 0.0);
    let j11 = textureSampleLevel(focus_source_a, focus_sampler, base + texel, 0.0);
    let jfa = mix(mix(j00, j10, fraction.x), mix(j01, j11, fraction.x), fraction.y);
    if jfa.z <= 0.0 {
        return vec4<f32>(0.0);
    }

    let difference = jfa.z - focus_pass.focus.y;
    let bandwidth = max(focus_pass.focus.z, 0.01);
    let sharpness = exp(-(difference * difference) / (2.0 * bandwidth * bandwidth));
    let focus_weight = 1.0 - sharpness;
    let pixel_distance = distance(input.uv * dimensions, jfa.xy * dimensions);
    let effective_maximum = focus_pass.focus.x * focus_weight;
    let ratio = clamp(pixel_distance / max(effective_maximum, 1.0), 0.0, 1.0);
    let falloff = 1.0 - smoothstep(0.0, 1.0, ratio);
    return vec4<f32>(jfa.xy, focus_weight * falloff, ratio);
}

@fragment
fn focus_kawase(input: FocusFullscreenOutput) -> @location(0) vec4<f32> {
    let texel = vec2<f32>(1.0) / vec2<f32>(textureDimensions(focus_source_a));
    let offset = focus_pass.extent_step_offset.w;
    let center = textureSampleLevel(focus_source_a, focus_sampler, input.uv, 0.0);
    let top_left = textureSampleLevel(
        focus_source_a,
        focus_sampler,
        input.uv + vec2<f32>(-offset, -offset) * texel,
        0.0,
    );
    let top_right = textureSampleLevel(
        focus_source_a,
        focus_sampler,
        input.uv + vec2<f32>(offset, -offset) * texel,
        0.0,
    );
    let bottom_left = textureSampleLevel(
        focus_source_a,
        focus_sampler,
        input.uv + vec2<f32>(-offset, offset) * texel,
        0.0,
    );
    let bottom_right = textureSampleLevel(
        focus_source_a,
        focus_sampler,
        input.uv + vec2<f32>(offset, offset) * texel,
        0.0,
    );
    let average_weight =
        (center.z + top_left.z + top_right.z + bottom_left.z + bottom_right.z) / 5.0;
    let average_distance =
        (center.w + top_left.w + top_right.w + bottom_left.w + bottom_right.w) / 5.0;
    return vec4<f32>(center.xy, average_weight, average_distance);
}

@fragment
fn focus_directional_blur(input: FocusFullscreenOutput) -> @location(0) vec4<f32> {
    let mask = textureSampleLevel(focus_source_b, focus_sampler, input.uv, 0.0);
    let blur_weight = clamp(mask.z * (1.0 - clamp(mask.w, 0.0, 1.0)), 0.0, 1.0);
    let original = textureSampleLevel(focus_source_a, focus_sampler, input.uv, 0.0);
    let effective_radius = focus_pass.blur.x * blur_weight * focus_pass.blur.y;
    let sigma = max(effective_radius / 2.0, 0.001);
    let radius = min(max(i32(ceil(effective_radius)), 1), 48);
    let texel = vec2<f32>(1.0) / vec2<f32>(textureDimensions(focus_source_a));
    let direction = focus_pass.extent_step_offset.xy * texel;
    var color_sum = vec4<f32>(0.0);
    var weight_sum = 0.0;
    for (var index = -48; index <= 48; index += 1) {
        if index < -radius || index > radius {
            continue;
        }
        let distance_from_center = f32(abs(index));
        let gaussian_weight = exp(
            -(distance_from_center * distance_from_center) / (2.0 * sigma * sigma),
        );
        let uv = clamp(
            input.uv + f32(index) * direction,
            vec2<f32>(0.0),
            vec2<f32>(1.0),
        );
        color_sum += textureSampleLevel(focus_source_a, focus_sampler, uv, 0.0)
            * gaussian_weight;
        weight_sum += gaussian_weight;
    }
    let blurred = color_sum / max(weight_sum, 0.001);
    if focus_pass.blur.z > 0.5 {
        return mix(original, blurred, blur_weight);
    }
    return blurred;
}
