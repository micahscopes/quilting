#define_import_path quilting::surface::patch_render

#import quilting::math::quaternion::{qmul, qconj, qinv, q_to_point}
#import quilting::surface::qb_eval::eval_qb
#import quilting::surface::patch_prepare::dyadic_leaf_domain
#import quilting::viz::density::edge_log2_density

// Pure subset of a backend's frame uniforms required for prepared-surface
// evaluation. Concrete binding blocks retain their established names/layouts
// and construct this value at the entry-point boundary.
struct PatchRenderTransform {
    mvp: mat4x4<f32>,
    mv: mat4x4<f32>,
    use_qb: i32,
    mob_a: vec4<f32>,
    mob_b: vec4<f32>,
    mob_c: vec4<f32>,
    mob_d: vec4<f32>,
    camera_pos: vec4<f32>,
}

// Hard ceiling on evaluated positions, in normalized model units. It is far
// beyond distinguishable scene scale while preventing a Möbius pole from
// destroying rasterizer and depth precision.
const POSITION_CLAMP: f32 = 1e4;

// One already-posed logical patch plus one canonical atlas vertex. Position
// scalar lanes retain root vertex IDs or adaptive depth/path tags.
struct PatchSurfaceInput {
    atlas_bary: vec3<f32>,
    control_a: vec4<f32>,
    control_b: vec4<f32>,
    control_c: vec4<f32>,
    weight_a: vec4<f32>,
    weight_b: vec4<f32>,
    weight_c: vec4<f32>,
    lod_info: vec4<f32>,
    vertex_lod: vec4<f32>,
    uv_ab: vec4<f32>,
    uv_c_prepare: vec4<f32>,
    normal_a: vec4<f32>,
    normal_b: vec4<f32>,
    normal_c: vec4<f32>,
    is_prepared: u32,
}

// Backend-neutral vertex result. Concrete entry points only attach shader
// locations/builtins, so WebGL2 and WebGPU cannot silently diverge in surface
// evaluation while retaining different input mechanisms.
struct PatchSurfaceResult {
    clip_pos: vec4<f32>,
    normal_vs: vec3<f32>,
    density: f32,
    tex_uv: vec2<f32>,
    position_vs: vec3<f32>,
    tangent_vs: vec3<f32>,
    bitangent_vs: vec3<f32>,
    normal_ws: vec3<f32>,
    position_ws: vec3<f32>,
    camera_pos_ws: vec3<f32>,
    fade: f32,
    tess_bary: vec3<f32>,
    instance_id: f32,
    mobius_stretch: f32,
    source_position_ws: vec3<f32>,
    node_id: f32,
}

// S3 permutation remapping lets all six face orientations share one canonical
// tessellation buffer.
fn permute_patch_barycentric(bary: vec3<f32>, permutation: i32) -> vec3<f32> {
    switch permutation {
        case 1: { return vec3<f32>(bary.x, bary.z, bary.y); }
        case 2: { return vec3<f32>(bary.y, bary.x, bary.z); }
        case 3: { return vec3<f32>(bary.y, bary.z, bary.x); }
        case 4: { return vec3<f32>(bary.z, bary.x, bary.y); }
        case 5: { return vec3<f32>(bary.z, bary.y, bary.x); }
        default: { return bary; }
    }
}

fn evaluate_flat_patch(
    bary: vec3<f32>,
    control_a: vec4<f32>,
    control_b: vec4<f32>,
    control_c: vec4<f32>,
) -> vec3<f32> {
    return bary.x * control_a.yzw
        + bary.y * control_b.yzw
        + bary.z * control_c.yzw;
}

struct MobiusPatchResult {
    position: vec3<f32>,
    normal: vec3<f32>,
    fade: f32,
}

// Fused Möbius + rational QB evaluation. Both quotient-rule tangents are
// right-multiplied by conj(bot), omitting a common positive scalar and keeping
// intermediates finite close to a pole.
fn evaluate_mobius_qb_patch(
    uniforms: PatchRenderTransform,
    bary: vec3<f32>,
    control_a: vec4<f32>,
    control_b: vec4<f32>,
    control_c: vec4<f32>,
    weight_a: vec4<f32>,
    weight_b: vec4<f32>,
    weight_c: vec4<f32>,
) -> MobiusPatchResult {
    let numerator_a = qmul(qmul(uniforms.mob_a, control_a) + uniforms.mob_b, weight_a);
    let numerator_b = qmul(qmul(uniforms.mob_a, control_b) + uniforms.mob_b, weight_b);
    let numerator_c = qmul(qmul(uniforms.mob_a, control_c) + uniforms.mob_b, weight_c);
    let denominator_a = qmul(qmul(uniforms.mob_c, control_a) + uniforms.mob_d, weight_a);
    let denominator_b = qmul(qmul(uniforms.mob_c, control_b) + uniforms.mob_d, weight_b);
    let denominator_c = qmul(qmul(uniforms.mob_c, control_c) + uniforms.mob_d, weight_c);

    let numerator = bary.x * numerator_a + bary.y * numerator_b + bary.z * numerator_c;
    let denominator = bary.x * denominator_a
        + bary.y * denominator_b
        + bary.z * denominator_c;
    let inverse_denominator = qinv(denominator);
    let value = qmul(numerator, inverse_denominator);
    let fade = smoothstep(0.0001, 0.001, dot(denominator, denominator));

    let derivative_numerator_u = numerator_b - numerator_a;
    let derivative_denominator_u = denominator_b - denominator_a;
    let derivative_numerator_v = numerator_c - numerator_a;
    let derivative_denominator_v = denominator_c - denominator_a;
    let conjugate_denominator = qconj(denominator);
    let tangent_u = qmul(
        derivative_numerator_u - qmul(value, derivative_denominator_u),
        conjugate_denominator,
    );
    let tangent_v = qmul(
        derivative_numerator_v - qmul(value, derivative_denominator_v),
        conjugate_denominator,
    );
    var normal = cross(tangent_u.yzw, tangent_v.yzw);
    let normal_length = length(normal);
    if normal_length > 1e-10 {
        normal /= normal_length;
    } else {
        normal = vec3<f32>(0.0, 0.0, 1.0);
    }
    return MobiusPatchResult(q_to_point(value), normal, fade);
}

fn transform_patch_normal(
    uniforms: PatchRenderTransform,
    control: vec4<f32>,
    normal: vec3<f32>,
) -> vec3<f32> {
    let denominator = qmul(uniforms.mob_c, control) + uniforms.mob_d;
    let mapped = qmul(
        qmul(uniforms.mob_a, control) + uniforms.mob_b,
        qinv(denominator),
    );
    let differential = uniforms.mob_a - qmul(mapped, uniforms.mob_c);
    return qmul(
        qmul(differential, vec4<f32>(0.0, normal)),
        qinv(denominator),
    ).yzw;
}

fn patch_conformal_scale(
    uniforms: PatchRenderTransform,
    control: vec4<f32>,
) -> f32 {
    let denominator = qmul(uniforms.mob_c, control) + uniforms.mob_d;
    let mapped = qmul(
        qmul(uniforms.mob_a, control) + uniforms.mob_b,
        qinv(denominator),
    );
    let differential = uniforms.mob_a - qmul(mapped, uniforms.mob_c);
    return sqrt(
        max(dot(differential, differential), 1e-20)
            / max(dot(denominator, denominator), 1e-20),
    );
}

fn evaluate_patch_surface(
    uniforms: PatchRenderTransform,
    input: PatchSurfaceInput,
) -> PatchSurfaceResult {
    let permutation = clamp(i32(round(input.lod_info.w)), 0, 5);
    let bary = permute_patch_barycentric(input.atlas_bary, permutation);
    let adaptive = input.is_prepared != 0u && input.control_a.x < -0.5;
    var source_bary = bary;
    var leaf_lod_scale = 1.0;
    if adaptive {
        let leaf_depth = -input.control_a.x - 1.0;
        leaf_lod_scale = exp2(leaf_depth);
        let domain = dyadic_leaf_domain(vec2<f32>(leaf_depth, input.control_b.x));
        source_bary = bary.x * domain.domain_corner_a
            + bary.y * domain.domain_corner_b
            + bary.z * domain.domain_corner_c;
    }

    let control_a = vec4<f32>(0.0, input.control_a.yzw);
    let control_b = vec4<f32>(0.0, input.control_b.yzw);
    let control_c = vec4<f32>(0.0, input.control_c.yzw);
    var source_position: vec3<f32>;
    if uniforms.use_qb == 1 {
        source_position = eval_qb(
            bary,
            control_a, control_b, control_c,
            input.weight_a, input.weight_b, input.weight_c,
        );
    } else {
        source_position = evaluate_flat_patch(bary, control_a, control_b, control_c);
    }

    var position: vec3<f32>;
    var normal: vec3<f32>;
    var fade = 1.0;
    if uniforms.use_qb == 1 {
        let evaluated = evaluate_mobius_qb_patch(
            uniforms,
            bary,
            control_a, control_b, control_c,
            input.weight_a, input.weight_b, input.weight_c,
        );
        position = evaluated.position;
        normal = evaluated.normal;
        fade = evaluated.fade;
    } else {
        let mapped_a = qmul(
            qmul(uniforms.mob_a, control_a) + uniforms.mob_b,
            qinv(qmul(uniforms.mob_c, control_a) + uniforms.mob_d),
        );
        let mapped_b = qmul(
            qmul(uniforms.mob_a, control_b) + uniforms.mob_b,
            qinv(qmul(uniforms.mob_c, control_b) + uniforms.mob_d),
        );
        let mapped_c = qmul(
            qmul(uniforms.mob_a, control_c) + uniforms.mob_b,
            qinv(qmul(uniforms.mob_c, control_c) + uniforms.mob_d),
        );
        position = evaluate_flat_patch(bary, mapped_a, mapped_b, mapped_c);
        let edge_a = normalize(mapped_b.yzw - mapped_a.yzw);
        let edge_b = normalize(mapped_c.yzw - mapped_a.yzw);
        let crossed = cross(edge_a, edge_b);
        let crossed_length = length(crossed);
        if crossed_length > 1e-10 {
            normal = crossed / crossed_length;
        } else {
            normal = vec3<f32>(0.0, 0.0, 1.0);
        }
    }

    let position_radius = length(position);
    if position_radius > POSITION_CLAMP {
        position *= POSITION_CLAMP / position_radius;
    }

    let smooth_a = input.normal_a.xyz;
    let smooth_b = input.normal_b.xyz;
    let smooth_c = input.normal_c.xyz;
    let has_smooth = dot(smooth_a, smooth_a)
        + dot(smooth_b, smooth_b)
        + dot(smooth_c, smooth_c) > 0.01;
    if has_smooth {
        let transformed_a = transform_patch_normal(uniforms, control_a, smooth_a);
        let transformed_b = transform_patch_normal(uniforms, control_b, smooth_b);
        let transformed_c = transform_patch_normal(uniforms, control_c, smooth_c);
        let blended = bary.x * transformed_a
            + bary.y * transformed_b
            + bary.z * transformed_c;
        let largest_component = max(max(abs(blended.x), abs(blended.y)), abs(blended.z));
        if largest_component > 1e-20 {
            normal = normalize(blended / largest_component);
        }
    }

    let absolute_edge_lods = max(input.lod_info.xyz * leaf_lod_scale, vec3<f32>(1.0));
    var visual_log_density = edge_log2_density(bary, absolute_edge_lods);
    if bary.x > 0.999999 {
        visual_log_density = log2(max(input.vertex_lod.x, 1.0));
    } else if bary.y > 0.999999 {
        visual_log_density = log2(max(input.vertex_lod.y, 1.0));
    } else if bary.z > 0.999999 {
        visual_log_density = log2(max(input.vertex_lod.z, 1.0));
    }

    let uv_a = input.uv_ab.xy;
    let uv_b = input.uv_ab.zw;
    let uv_c = input.uv_c_prepare.xy;
    let tex_uv = bary.x * uv_a + bary.y * uv_b + bary.z * uv_c;
    let delta_uv_ab = uv_b - uv_a;
    let delta_uv_ac = uv_c - uv_a;
    let determinant = delta_uv_ab.x * delta_uv_ac.y
        - delta_uv_ab.y * delta_uv_ac.x;
    var tangent = vec3<f32>(1.0, 0.0, 0.0);
    var bitangent = vec3<f32>(0.0, 1.0, 0.0);
    if abs(determinant) > 1e-6 {
        let edge_ab = control_b.yzw - control_a.yzw;
        let edge_ac = control_c.yzw - control_a.yzw;
        let inverse_determinant = 1.0 / determinant;
        tangent = normalize(
            (edge_ab * delta_uv_ac.y - edge_ac * delta_uv_ab.y)
                * inverse_determinant,
        );
        bitangent = normalize(
            (edge_ac * delta_uv_ab.x - edge_ab * delta_uv_ac.x)
                * inverse_determinant,
        );
    }

    let scale_a = patch_conformal_scale(uniforms, control_a);
    let scale_b = patch_conformal_scale(uniforms, control_b);
    let scale_c = patch_conformal_scale(uniforms, control_c);
    let stretch = bary.x * scale_a + bary.y * scale_b + bary.z * scale_c;
    let log_stretch = log2(max(stretch, 1e-20));

    return PatchSurfaceResult(
        uniforms.mvp * vec4<f32>(position, 1.0),
        normalize((uniforms.mv * vec4<f32>(normal, 0.0)).xyz),
        max(visual_log_density, 0.0) / 10.0,
        tex_uv,
        (uniforms.mv * vec4<f32>(position, 1.0)).xyz,
        normalize((uniforms.mv * vec4<f32>(tangent, 0.0)).xyz),
        normalize((uniforms.mv * vec4<f32>(bitangent, 0.0)).xyz),
        normal,
        position,
        uniforms.camera_pos.xyz,
        fade,
        source_bary,
        input.vertex_lod.w,
        1.0 / (1.0 + exp(-log_stretch)),
        source_position,
        input.normal_a.w,
    );
}
