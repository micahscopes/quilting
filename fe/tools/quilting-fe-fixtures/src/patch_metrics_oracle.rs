//! Independent f64 checks for the small pullback telemetry used by Fe.

#[derive(Clone, Copy)]
struct Metric2 {
    uu: f64,
    uv: f64,
    vv: f64,
    defined: bool,
}

#[derive(Clone, Copy, Debug)]
struct Spectrum2 {
    minimum_stretch: f64,
    maximum_stretch: f64,
    area_scale: f64,
    anisotropy: f64,
    defined: bool,
}

fn spectrum(metric: Metric2, minimum_eigenvalue: f64) -> Spectrum2 {
    let trace = metric.uu + metric.vv;
    let difference = metric.uu - metric.vv;
    let discriminant = (difference * difference + 4.0 * metric.uv * metric.uv)
        .max(0.0)
        .sqrt();
    let minimum_eigen = 0.5 * (trace - discriminant);
    let maximum_eigen = 0.5 * (trace + discriminant);
    let determinant = metric.uu * metric.vv - metric.uv * metric.uv;
    let defined = metric.defined
        && minimum_eigen >= minimum_eigenvalue
        && maximum_eigen >= minimum_eigen
        && determinant > 0.0;
    if !defined {
        return Spectrum2 {
            minimum_stretch: 0.0,
            maximum_stretch: 0.0,
            area_scale: 0.0,
            anisotropy: 0.0,
            defined: false,
        };
    }
    let minimum_stretch = minimum_eigen.sqrt();
    let maximum_stretch = maximum_eigen.sqrt();
    Spectrum2 {
        minimum_stretch,
        maximum_stretch,
        area_scale: determinant.sqrt(),
        anisotropy: maximum_stretch / minimum_stretch,
        defined: true,
    }
}

fn shape_distance(left: Metric2, right: Metric2) -> f64 {
    let left_trace = left.uu + left.vv;
    let right_trace = right.uu + right.vv;
    let delta_uu = left.uu / left_trace - right.uu / right_trace;
    let delta_uv = left.uv / left_trace - right.uv / right_trace;
    let delta_vv = left.vv / left_trace - right.vv / right_trace;
    (delta_uu * delta_uu + 2.0 * delta_uv * delta_uv + delta_vv * delta_vv).sqrt()
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1.0e-12,
        "expected {expected}, got {actual}"
    );
}

#[test]
fn pullback_spectrum_recovers_principal_stretches() {
    let identity = spectrum(
        Metric2 {
            uu: 1.0,
            uv: 0.0,
            vv: 1.0,
            defined: true,
        },
        1.0e-12,
    );
    assert!(identity.defined);
    assert_close(identity.minimum_stretch, 1.0);
    assert_close(identity.maximum_stretch, 1.0);
    assert_close(identity.area_scale, 1.0);
    assert_close(identity.anisotropy, 1.0);

    // Singular values 1 and 2, with their axes rotated by 45 degrees.
    let rotated = spectrum(
        Metric2 {
            uu: 2.5,
            uv: 1.5,
            vv: 2.5,
            defined: true,
        },
        1.0e-12,
    );
    assert!(rotated.defined);
    assert_close(rotated.minimum_stretch, 1.0);
    assert_close(rotated.maximum_stretch, 2.0);
    assert_close(rotated.area_scale, 2.0);
    assert_close(rotated.anisotropy, 2.0);
}

#[test]
fn normalized_metric_shape_ignores_uniform_scale_but_detects_shear() {
    let base = Metric2 {
        uu: 4.0,
        uv: 0.0,
        vv: 1.0,
        defined: true,
    };
    let scaled = Metric2 {
        uu: 52.0,
        uv: 0.0,
        vv: 13.0,
        defined: true,
    };
    let sheared = Metric2 {
        uu: 4.0,
        uv: 1.0,
        vv: 1.0,
        defined: true,
    };
    assert_close(shape_distance(base, scaled), 0.0);
    assert!(shape_distance(base, sheared) > 0.25);
}

#[test]
fn degenerate_or_unadmitted_metrics_fail_closed() {
    let collapsed = spectrum(
        Metric2 {
            uu: 1.0,
            uv: 1.0,
            vv: 1.0,
            defined: true,
        },
        1.0e-12,
    );
    let rejected = spectrum(
        Metric2 {
            uu: 1.0,
            uv: 0.0,
            vv: 1.0,
            defined: false,
        },
        1.0e-12,
    );
    assert!(!collapsed.defined);
    assert!(!rejected.defined);
}

#[test]
fn mixed_frame_difference_detects_a_bilinear_saddle() {
    // P(u,v) = (u, v, k*u*v), so du changes along v and dv changes along u.
    let k: f64 = 7.0;
    let du_bottom: [f64; 3] = [1.0, 0.0, 0.0];
    let du_top: [f64; 3] = [1.0, 0.0, k];
    let dv_left: [f64; 3] = [0.0, 1.0, 0.0];
    let dv_right: [f64; 3] = [0.0, 1.0, k];
    let tangent_u_change = [
        du_top[0] - du_bottom[0],
        du_top[1] - du_bottom[1],
        du_top[2] - du_bottom[2],
    ];
    let tangent_v_change = [
        dv_right[0] - dv_left[0],
        dv_right[1] - dv_left[1],
        dv_right[2] - dv_left[2],
    ];
    let norm =
        |value: [f64; 3]| (value[0] * value[0] + value[1] * value[1] + value[2] * value[2]).sqrt();
    let center_trace = 2.0 + 0.5 * k * k;
    let mixed_turn = (norm(tangent_u_change) + norm(tangent_v_change)) / center_trace.sqrt();
    assert!(mixed_turn > 2.0);

    let planar_turn = (norm([0.0, 0.0, 0.0]) + norm([0.0, 0.0, 0.0])) / 2.0_f64.sqrt();
    assert_close(planar_turn, 0.0);
}
