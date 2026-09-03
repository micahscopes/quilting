use std::path::Path;
use std::sync::OnceLock;

use common::InputDb;
use driver::DriverDataBase;
use fe_codegen::{layout_for, BackendKind, OptLevel};
use hir::hir_def::HirIngot;
use quilting_core::patch::QBTriPatch;
use quilting_core::permutation::{perm_sign, S3_PERMUTATIONS};
use quilting_core::quaternion::Quat;
use salsa::Setter;
use url::Url;
use wasmtime::{Instance, Store, TypedFunc};

static ORACLE_WASM: OnceLock<Vec<u8>> = OnceLock::new();

fn compile_oracle_gate() -> &'static [u8] {
    ORACLE_WASM.get_or_init(|| {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../ingots/validation/classic_quilting_oracle");
        let url = Url::from_directory_path(path.canonicalize().unwrap()).unwrap();
        let mut db = DriverDataBase::default();
        db.compilation_settings()
            .set_profile(&mut db)
            .to("release".into());
        assert!(
            !driver::init_ingot(&mut db, &url),
            "classic Quilting oracle ingot initialization diagnostics"
        );
        let ingot = db
            .workspace()
            .containing_ingot(&db, url)
            .expect("classic Quilting oracle ingot");
        let top_mod = ingot.root_mod(&db);
        let diagnostics = db.run_on_top_mod(top_mod).format_diags(&db);
        assert!(
            diagnostics.is_empty(),
            "unexpected classic Quilting diagnostics:\n{diagnostics}"
        );
        let wasm = BackendKind::Wasm
            .create()
            .compile(&db, top_mod, layout_for(BackendKind::Wasm), OptLevel::O2)
            .expect("classic Quilting oracle should compile to Wasm")
            .into_bytecode()
            .expect("Wasm output should be bytecode");
        wasmparser::validate(&wasm).expect("classic Quilting oracle Wasm should validate");
        wasm
    })
}

fn instantiate() -> (Store<()>, Instance) {
    let engine = wasmtime::Engine::default();
    let module = wasmtime::Module::new(&engine, compile_oracle_gate())
        .expect("load classic Quilting oracle Wasm");
    assert!(
        module.imports().next().is_none(),
        "pure M1 Fe oracle must be self-contained"
    );
    let mut store = Store::new(&engine, ());
    let instance = Instance::new(&mut store, &module, &[]).expect("instantiate M1 oracle");
    (store, instance)
}

fn function<P, R>(store: &mut Store<()>, instance: &Instance, name: &str) -> TypedFunc<P, R>
where
    P: wasmtime::WasmParams,
    R: wasmtime::WasmResults,
{
    instance
        .get_typed_func::<P, R>(store, name)
        .unwrap_or_else(|error| panic!("missing {name}: {error}"))
}

fn call2(store: &mut Store<()>, instance: &Instance, name: &str, a: f32, b: f32) -> f32 {
    function::<(f32, f32), f32>(store, instance, name)
        .call(store, (a, b))
        .unwrap()
}

fn call3(store: &mut Store<()>, instance: &Instance, name: &str, values: [f32; 3]) -> f32 {
    let [a, b, c] = values;
    function::<(f32, f32, f32), f32>(store, instance, name)
        .call(store, (a, b, c))
        .unwrap()
}

fn call4_f32(store: &mut Store<()>, instance: &Instance, name: &str, values: [f32; 4]) -> f32 {
    let [a, b, c, d] = values;
    function::<(f32, f32, f32, f32), f32>(store, instance, name)
        .call(store, (a, b, c, d))
        .unwrap()
}

fn call4_i32(store: &mut Store<()>, instance: &Instance, name: &str, values: [f32; 4]) -> i32 {
    let [a, b, c, d] = values;
    function::<(f32, f32, f32, f32), i32>(store, instance, name)
        .call(store, (a, b, c, d))
        .unwrap()
}

#[allow(clippy::many_single_char_names)]
fn call5_f32(store: &mut Store<()>, instance: &Instance, name: &str, values: [f32; 5]) -> f32 {
    let [a, b, c, d, e] = values;
    function::<(f32, f32, f32, f32, f32), f32>(store, instance, name)
        .call(store, (a, b, c, d, e))
        .unwrap()
}

#[allow(clippy::many_single_char_names)]
fn call5_i32(store: &mut Store<()>, instance: &Instance, name: &str, values: [f32; 5]) -> i32 {
    let [a, b, c, d, e] = values;
    function::<(f32, f32, f32, f32, f32), i32>(store, instance, name)
        .call(store, (a, b, c, d, e))
        .unwrap()
}

#[allow(clippy::many_single_char_names)]
fn call8(store: &mut Store<()>, instance: &Instance, name: &str, values: [f32; 8]) -> f32 {
    let [a, b, c, d, e, f, g, h] = values;
    function::<(f32, f32, f32, f32, f32, f32, f32, f32), f32>(store, instance, name)
        .call(store, (a, b, c, d, e, f, g, h))
        .unwrap()
}

#[allow(clippy::many_single_char_names)]
fn call7_f32(store: &mut Store<()>, instance: &Instance, name: &str, values: [f32; 7]) -> f32 {
    let [a, b, c, d, e, f, g] = values;
    function::<(f32, f32, f32, f32, f32, f32, f32), f32>(store, instance, name)
        .call(store, (a, b, c, d, e, f, g))
        .unwrap()
}

#[allow(clippy::many_single_char_names)]
fn call7_i32(store: &mut Store<()>, instance: &Instance, name: &str, values: [f32; 7]) -> i32 {
    let [a, b, c, d, e, f, g] = values;
    function::<(f32, f32, f32, f32, f32, f32, f32), i32>(store, instance, name)
        .call(store, (a, b, c, d, e, f, g))
        .unwrap()
}

#[allow(clippy::many_single_char_names)]
fn call10_f32(store: &mut Store<()>, instance: &Instance, name: &str, values: [f32; 10]) -> f32 {
    let [a, b, c, d, e, f, g, h, i, j] = values;
    function::<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32), f32>(store, instance, name)
        .call(store, (a, b, c, d, e, f, g, h, i, j))
        .unwrap()
}

fn assert_close(actual: f32, expected: f32, tolerance: f32, context: &str) {
    assert!(
        actual.is_finite(),
        "{context}: nonfinite Fe output {actual}"
    );
    assert!(
        (actual - expected).abs() <= tolerance,
        "{context}: Fe={actual:?}, oracle={expected:?}, tolerance={tolerance:?}"
    );
}

#[test]
fn sparse_clifford_patch_wasm_matches_the_independent_dense_oracle() {
    let (mut store, instance) = instantiate();
    let position_exports = [
        "clifford_position_x",
        "clifford_position_y",
        "clifford_position_z",
    ];

    for s_step in 0_u16..=8 {
        for t_step in 0_u16..=8 {
            let s = f32::from(s_step) / 8.0;
            let t = f32::from(t_step) / 8.0;
            let (expected_position, expected_residual) =
                crate::clifford_oracle::paper_sample(f64::from(s), f64::from(t));
            for lane in 0..3 {
                assert_close(
                    call2(&mut store, &instance, position_exports[lane], s, t),
                    oracle_f32(expected_position[lane]),
                    2.0e-5,
                    &format!("Clifford patch ({s},{t}) lane {lane}"),
                );
            }
            assert_close(
                call2(&mut store, &instance, "clifford_trivector_residual", s, t),
                oracle_f32(expected_residual),
                2.0e-5,
                &format!("Clifford patch ({s},{t}) trivector residual"),
            );
            assert_eq!(
                function::<(f32, f32), i32>(&mut store, &instance, "clifford_patch_defined",)
                    .call(&mut store, (s, t))
                    .unwrap(),
                1,
                "paper patch must remain a defined Euclidean Patch at ({s},{t})",
            );
        }
    }

    assert_close(
        function::<(), f32>(&mut store, &instance, "clifford_reconciliation_scale")
            .call(&mut store, ())
            .unwrap(),
        oracle_f32(crate::clifford_oracle::paper_reconciliation_scale()),
        f32::EPSILON,
        "paper fourth-weight reconciliation",
    );
    assert_eq!(
        function::<(), i32>(&mut store, &instance, "clifford_reconciliation_conditioned",)
            .call(&mut store, ())
            .unwrap(),
        1,
    );
}

#[test]
fn sparse_clifford_differential_wasm_matches_finite_differences_of_the_dense_oracle() {
    let (mut store, instance) = instantiate();
    let tangent_exports = [
        [
            "clifford_tangent_s_x",
            "clifford_tangent_s_y",
            "clifford_tangent_s_z",
        ],
        [
            "clifford_tangent_t_x",
            "clifford_tangent_t_y",
            "clifford_tangent_t_z",
        ],
    ];
    let normal_exports = [
        "clifford_normal_x",
        "clifford_normal_y",
        "clifford_normal_z",
    ];
    let h = 1.0e-5_f64;

    for s_step in 1_u16..8 {
        for t_step in 1_u16..8 {
            let s = f32::from(s_step) / 8.0;
            let t = f32::from(t_step) / 8.0;
            let s64 = f64::from(s);
            let t64 = f64::from(t);
            let (s_lower, _) = crate::clifford_oracle::paper_sample(s64 - h, t64);
            let (s_upper, _) = crate::clifford_oracle::paper_sample(s64 + h, t64);
            let (t_lower, _) = crate::clifford_oracle::paper_sample(s64, t64 - h);
            let (t_upper, _) = crate::clifford_oracle::paper_sample(s64, t64 + h);
            let tangent_s: [f64; 3] =
                std::array::from_fn(|lane| (s_upper[lane] - s_lower[lane]) / (2.0 * h));
            let tangent_t: [f64; 3] =
                std::array::from_fn(|lane| (t_upper[lane] - t_lower[lane]) / (2.0 * h));
            let cross = [
                tangent_s[1] * tangent_t[2] - tangent_s[2] * tangent_t[1],
                tangent_s[2] * tangent_t[0] - tangent_s[0] * tangent_t[2],
                tangent_s[0] * tangent_t[1] - tangent_s[1] * tangent_t[0],
            ];
            let cross_norm = cross
                .iter()
                .map(|component| component * component)
                .sum::<f64>()
                .sqrt();
            assert!(
                cross_norm > 1.0e-8,
                "oracle differential degenerated at ({s},{t})"
            );

            for lane in 0..3 {
                assert_close(
                    call2(&mut store, &instance, tangent_exports[0][lane], s, t),
                    oracle_f32(tangent_s[lane]),
                    7.5e-4,
                    &format!("Clifford tangent-s ({s},{t}) lane {lane}"),
                );
                assert_close(
                    call2(&mut store, &instance, tangent_exports[1][lane], s, t),
                    oracle_f32(tangent_t[lane]),
                    7.5e-4,
                    &format!("Clifford tangent-t ({s},{t}) lane {lane}"),
                );
                assert_close(
                    call2(&mut store, &instance, normal_exports[lane], s, t),
                    oracle_f32(cross[lane] / cross_norm),
                    2.0e-4,
                    &format!("Clifford normal ({s},{t}) lane {lane}"),
                );
            }
        }
    }
}

#[test]
fn sparse_cga_sphere_map_wasm_matches_the_independent_dense_cl41_oracle() {
    let (mut store, instance) = instantiate();
    let position_exports = [
        "cga_reflect_position_x",
        "cga_reflect_position_y",
        "cga_reflect_position_z",
    ];
    let tangent_exports = [
        "cga_reflect_tangent_x",
        "cga_reflect_tangent_y",
        "cga_reflect_tangent_z",
    ];
    let normal_exports = [
        "cga_reflect_xy_normal_x",
        "cga_reflect_xy_normal_y",
        "cga_reflect_xy_normal_z",
    ];
    let cases = [
        ([1.0, 0.5, -0.25], [0.0, 0.0, 0.0], 0.5),
        ([-0.8, 1.4, 2.0], [0.35, -0.7, 1.1], 1.25),
        ([3.0, -2.0, 0.7], [-1.3, 0.2, 0.4], 2.0),
    ];
    let tangents = [[1.0, 0.0, 0.0], [0.3, -0.8, 0.5], [0.0, 0.0, 1.0]];

    for (point, center, radius) in cases {
        let values = [
            point[0], point[1], point[2], center[0], center[1], center[2], radius,
        ];
        let expected = crate::cga_oracle::sphere_reflection(
            point.map(f64::from),
            center.map(f64::from),
            f64::from(radius),
        );
        for lane in 0..3 {
            assert_close(
                call7_f32(&mut store, &instance, position_exports[lane], values),
                oracle_f32(expected.position[lane]),
                4.0e-5,
                &format!("CGA reflected position {point:?} lane {lane}"),
            );
        }
        assert_close(
            call7_f32(&mut store, &instance, "cga_reflect_weight", values),
            oracle_f32(expected.weight),
            4.0e-5,
            &format!("CGA projective weight {point:?}"),
        );
        assert_close(
            call7_f32(&mut store, &instance, "cga_reflect_null_residual", values),
            oracle_f32(expected.null_residual),
            3.0e-5,
            &format!("CGA null residual {point:?}"),
        );
        assert_eq!(
            call7_i32(&mut store, &instance, "cga_reflect_conditioned", values),
            1,
        );

        for tangent in tangents {
            let tangent_values = [
                point[0], point[1], point[2], tangent[0], tangent[1], tangent[2], center[0],
                center[1], center[2], radius,
            ];
            let expected_tangent = crate::cga_oracle::finite_difference_tangent(
                point.map(f64::from),
                tangent.map(f64::from),
                center.map(f64::from),
                f64::from(radius),
            );
            for lane in 0..3 {
                assert_close(
                    call10_f32(&mut store, &instance, tangent_exports[lane], tangent_values),
                    oracle_f32(expected_tangent[lane]),
                    6.0e-4,
                    &format!("CGA reflected tangent {point:?}/{tangent:?} lane {lane}"),
                );
            }
        }

        let expected_normal = crate::cga_oracle::xy_normal(
            point.map(f64::from),
            center.map(f64::from),
            f64::from(radius),
        );
        for lane in 0..3 {
            assert_close(
                call7_f32(&mut store, &instance, normal_exports[lane], values),
                oracle_f32(expected_normal[lane]),
                3.0e-4,
                &format!("CGA reflected xy normal {point:?} lane {lane}"),
            );
        }
    }

    let center = [0.4_f32, -0.2, 0.7];
    let radius = 1.75_f32;
    for direction in [[1.0_f32, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]] {
        let point = std::array::from_fn(|lane| center[lane] + radius * direction[lane]);
        let values = [
            point[0], point[1], point[2], center[0], center[1], center[2], radius,
        ];
        assert_close(
            call7_f32(&mut store, &instance, "cga_sphere_incidence", values),
            oracle_f32(crate::cga_oracle::sphere_incidence(
                point.map(f64::from),
                center.map(f64::from),
                f64::from(radius),
            )),
            2.0e-6,
            "CGA point-on-sphere incidence",
        );
    }
}

#[test]
fn quilting_domain_wasm_matches_the_frozen_m0_barycentrics() {
    const MATRIX: &[u8] =
        include_bytes!("../../../fixtures/classic-quilting/v1/direct-seed42-matrix.cqa");
    let artifact = crate::decode(MATRIX).expect("frozen M0 matrix");
    let (mut store, instance) = instantiate();

    for (index, vertex) in artifact.vertices.iter().enumerate() {
        let [a, b, c] = vertex.barycentric;
        let expected_x = 0.866_025_4_f32 * (c - b);
        let expected_y = (3.0 * a - 1.0) * 0.5;
        let actual_x = call3(&mut store, &instance, "domain_cartesian_x", [a, b, c]);
        let actual_y = call3(&mut store, &instance, "domain_cartesian_y", [a, b, c]);
        assert_close(
            actual_x,
            expected_x,
            f32::EPSILON,
            &format!("vertex {index} x"),
        );
        assert_close(
            actual_y,
            expected_y,
            f32::EPSILON,
            &format!("vertex {index} y"),
        );

        let round_trip = [
            call2(&mut store, &instance, "domain_bary_a", actual_x, actual_y),
            call2(&mut store, &instance, "domain_bary_b", actual_x, actual_y),
            call2(&mut store, &instance, "domain_bary_c", actual_x, actual_y),
        ];
        for (lane, (&actual, &expected)) in
            round_trip.iter().zip(vertex.barycentric.iter()).enumerate()
        {
            assert_close(
                actual,
                expected,
                3.0e-7,
                &format!("vertex {index} bary lane {lane}"),
            );
        }
        assert_eq!(
            call4_i32(&mut store, &instance, "domain_contains", [a, b, c, 2.0e-6],),
            1,
            "frozen vertex {index} must remain admitted"
        );
        for edge in 0..3 {
            if vertex.barycentric[edge].to_bits() == 0.0_f32.to_bits() {
                let expected_parameter = match edge {
                    0 => c,
                    1 => a,
                    2 => b,
                    _ => unreachable!(),
                };
                let edge_u32 = u32::try_from(edge).unwrap();
                let actual_parameter = function::<(u32, f32, f32, f32), f32>(
                    &mut store,
                    &instance,
                    "domain_edge_parameter",
                )
                .call(&mut store, (edge_u32, a, b, c))
                .unwrap();
                assert_eq!(actual_parameter.to_bits(), expected_parameter.to_bits());
            }
        }
    }

    let near_boundary = [1.0e-8_f32, 0.25, 0.75 - 1.0e-8, 1.0e-6];
    let admitted = [
        call4_f32(&mut store, &instance, "domain_admit_a", near_boundary),
        call4_f32(&mut store, &instance, "domain_admit_b", near_boundary),
        call4_f32(&mut store, &instance, "domain_admit_c", near_boundary),
    ];
    assert_eq!(admitted[0].to_bits(), 0.0_f32.to_bits());
    assert_close(admitted.iter().sum(), 1.0, f32::EPSILON, "admitted sum");
    assert_eq!(
        call4_i32(&mut store, &instance, "domain_admit_valid", near_boundary),
        1
    );
    assert_eq!(
        call4_i32(
            &mut store,
            &instance,
            "domain_admit_valid",
            [-1.0, -2.0, -3.0, 1.0e-6],
        ),
        0
    );
}

fn multiply_f32(left: [f32; 4], right: [f32; 4]) -> [f32; 4] {
    let [aw, ax, ay, az] = left;
    let [bw, bx, by, bz] = right;
    [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]
}

#[test]
fn quilting_quaternion_wasm_matches_independent_f32_vectors_and_fails_closed() {
    let (mut store, instance) = instantiate();
    let cases = [
        ([0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]),
        ([1.0, 2.0, -3.0, 0.5], [-0.25, 4.0, 0.75, -2.0]),
        ([0.9, -0.15, 0.25, 0.1], [1.1, 0.1, 0.05, -0.2]),
    ];
    let exports = [
        "quaternion_multiply_w",
        "quaternion_multiply_x",
        "quaternion_multiply_y",
        "quaternion_multiply_z",
    ];
    for (case_index, (left, right)) in cases.into_iter().enumerate() {
        let expected = multiply_f32(left, right);
        let arguments = [
            left[0], left[1], left[2], left[3], right[0], right[1], right[2], right[3],
        ];
        for lane in 0..4 {
            let actual = call8(&mut store, &instance, exports[lane], arguments);
            assert_close(
                actual,
                expected[lane],
                2.0 * f32::EPSILON,
                &format!("quaternion case {case_index} lane {lane}"),
            );
        }
    }

    let value = [1.0_f32, -2.0, 0.5, 3.0];
    let minimum = 1.0e-20;
    let norm_squared = value.iter().map(|lane| lane * lane).sum::<f32>();
    let expected = [
        value[0] / norm_squared,
        -value[1] / norm_squared,
        -value[2] / norm_squared,
        -value[3] / norm_squared,
    ];
    let inverse_exports = [
        "quaternion_inverse_w",
        "quaternion_inverse_x",
        "quaternion_inverse_y",
        "quaternion_inverse_z",
    ];
    for lane in 0..4 {
        assert_close(
            call5_f32(
                &mut store,
                &instance,
                inverse_exports[lane],
                [value[0], value[1], value[2], value[3], minimum],
            ),
            expected[lane],
            2.0 * f32::EPSILON,
            &format!("inverse lane {lane}"),
        );
    }
    assert_eq!(
        call5_i32(
            &mut store,
            &instance,
            "quaternion_inverse_valid",
            [value[0], value[1], value[2], value[3], minimum],
        ),
        1
    );

    let pole = [1.0e-12_f32, 0.0, 0.0, 0.0, minimum];
    assert_eq!(
        call5_i32(&mut store, &instance, "quaternion_inverse_valid", pole,),
        0
    );
    for export in inverse_exports {
        let lane = call5_f32(&mut store, &instance, export, pole);
        assert_eq!(lane.to_bits(), 0.0_f32.to_bits());
    }
}

fn curved_patch() -> QBTriPatch {
    QBTriPatch::new(
        [
            Quat::from_point(-0.75, -0.25, 0.1),
            Quat::from_point(0.8, -0.15, -0.2),
            Quat::from_point(0.05, 0.9, 0.35),
        ],
        [
            Quat::new(1.0, 0.2, -0.1, 0.05),
            Quat::new(0.9, -0.15, 0.25, 0.1),
            Quat::new(1.1, 0.1, 0.05, -0.2),
        ],
    )
}

fn normal_from_tangents(tangent_u: [f64; 3], tangent_v: [f64; 3]) -> [f64; 3] {
    let cross = [
        tangent_u[1] * tangent_v[2] - tangent_u[2] * tangent_v[1],
        tangent_u[2] * tangent_v[0] - tangent_u[0] * tangent_v[2],
        tangent_u[0] * tangent_v[1] - tangent_u[1] * tangent_v[0],
    ];
    let length = cross.iter().map(|value| value * value).sum::<f64>().sqrt();
    cross.map(|value| value / length)
}

fn oracle_f32(value: f64) -> f32 {
    assert!(value.is_finite());
    assert!(value >= f64::from(f32::MIN) && value <= f64::from(f32::MAX));
    #[allow(clippy::cast_possible_truncation)]
    {
        value as f32
    }
}

fn assert_curved_patch(store: &mut Store<()>, instance: &Instance) {
    let patch = curved_patch();
    let position_exports = ["qb_position_x", "qb_position_y", "qb_position_z"];
    let first_tangent_exports = ["qb_tangent_u_x", "qb_tangent_u_y", "qb_tangent_u_z"];
    let second_tangent_exports = ["qb_tangent_v_x", "qb_tangent_v_y", "qb_tangent_v_z"];
    let normal_exports = ["qb_normal_x", "qb_normal_y", "qb_normal_z"];

    for denominator in 1_u16..=4 {
        for u_step in 0..=denominator {
            for v_step in 0..=denominator - u_step {
                let u = f32::from(u_step) / f32::from(denominator);
                let v = f32::from(v_step) / f32::from(denominator);
                let expected = patch.eval_differential(f64::from(u), f64::from(v));
                let expected_normal = normal_from_tangents(expected.tangent_u, expected.tangent_v);
                for lane in 0..3 {
                    assert_close(
                        call2(store, instance, position_exports[lane], u, v),
                        oracle_f32(expected.position[lane]),
                        2.0e-6,
                        &format!("QB position ({u},{v}) lane {lane}"),
                    );
                    assert_close(
                        call2(store, instance, first_tangent_exports[lane], u, v),
                        oracle_f32(expected.tangent_u[lane]),
                        4.0e-6,
                        &format!("QB tangent u ({u},{v}) lane {lane}"),
                    );
                    assert_close(
                        call2(store, instance, second_tangent_exports[lane], u, v),
                        oracle_f32(expected.tangent_v[lane]),
                        4.0e-6,
                        &format!("QB tangent v ({u},{v}) lane {lane}"),
                    );
                    assert_close(
                        call2(store, instance, normal_exports[lane], u, v),
                        oracle_f32(expected_normal[lane]),
                        4.0e-6,
                        &format!("QB normal ({u},{v}) lane {lane}"),
                    );
                }
            }
        }
    }
}

#[test]
fn patch_qb_adapter_is_identical_to_the_family_evaluator() {
    let (mut store, instance) = instantiate();
    let direct_exports = ["qb_position_x", "qb_position_y", "qb_position_z"];
    let patch_exports = [
        "patch_qb_position_x",
        "patch_qb_position_y",
        "patch_qb_position_z",
    ];

    for denominator in 1_u16..=4 {
        for u_step in 0..=denominator {
            for v_step in 0..=denominator - u_step {
                let u = f32::from(u_step) / f32::from(denominator);
                let v = f32::from(v_step) / f32::from(denominator);
                assert_eq!(
                    function::<(f32, f32), i32>(&mut store, &instance, "patch_qb_defined",)
                        .call(&mut store, (u, v))
                        .unwrap(),
                    1,
                    "Patch QB should be defined at ({u},{v})",
                );
                for lane in 0..3 {
                    let direct = call2(&mut store, &instance, direct_exports[lane], u, v);
                    let generic = call2(&mut store, &instance, patch_exports[lane], u, v);
                    assert_eq!(
                        generic.to_bits(),
                        direct.to_bits(),
                        "Patch/QB mismatch at ({u},{v}) lane {lane}",
                    );
                }
            }
        }
    }

    assert_eq!(
        function::<(f32, f32), i32>(&mut store, &instance, "patch_zero_weight_defined",)
            .call(&mut store, (0.25, 0.5))
            .unwrap(),
        0,
        "Patch QB must preserve the family evaluator's explicit conditioning failure",
    );
}

fn assert_flat_patch(store: &mut Store<()>, instance: &Instance) {
    for (u, v) in [(0.0_f32, 0.0_f32), (1.0, 0.0), (0.0, 1.0), (0.25, 0.5)] {
        assert_close(
            call2(store, instance, "qb_flat_position_x", u, v),
            u,
            f32::EPSILON,
            "flat x",
        );
        assert_close(
            call2(store, instance, "qb_flat_position_y", u, v),
            v,
            f32::EPSILON,
            "flat y",
        );
        assert_eq!(
            call2(store, instance, "qb_flat_position_z", u, v).to_bits(),
            0.0_f32.to_bits()
        );
        assert_close(
            call2(store, instance, "qb_flat_normal_z", u, v),
            1.0,
            f32::EPSILON,
            "flat normal",
        );
    }
}

fn assert_pole_fails_closed(store: &mut Store<()>, instance: &Instance) {
    assert_eq!(
        function::<(f32, f32), i32>(store, instance, "qb_zero_weight_conditioned")
            .call(&mut *store, (1.0 / 3.0, 1.0 / 3.0))
            .unwrap(),
        0
    );
    assert_eq!(
        call2(
            store,
            instance,
            "qb_zero_weight_position_x",
            1.0 / 3.0,
            1.0 / 3.0,
        )
        .to_bits(),
        0.0_f32.to_bits()
    );
}

fn assert_s3_remaps(store: &mut Store<()>, instance: &Instance) {
    let bary = [0.2_f32, 0.3, 0.5];
    let remap_exports = ["qb_remap_a", "qb_remap_b", "qb_remap_c"];
    for (permutation, indices) in S3_PERMUTATIONS.into_iter().enumerate() {
        let permutation_u32 = u32::try_from(permutation).unwrap();
        for lane in 0..3 {
            let actual =
                function::<(u32, f32, f32, f32), f32>(store, instance, remap_exports[lane])
                    .call(&mut *store, (permutation_u32, bary[0], bary[1], bary[2]))
                    .unwrap();
            assert_eq!(actual.to_bits(), bary[indices[lane]].to_bits());
        }
        let parity = function::<u32, f32>(store, instance, "qb_permutation_parity")
            .call(&mut *store, permutation_u32)
            .unwrap();
        let expected_parity = if perm_sign(permutation) == 1 {
            1.0
        } else {
            -1.0
        };
        assert_eq!(parity, expected_parity);
        let normal_z = call2(store, instance, "qb_normal_z", 0.25, 0.25);
        let permuted_z = function::<(u32, f32, f32), f32>(store, instance, "qb_permuted_normal_z")
            .call(&mut *store, (permutation_u32, 0.25, 0.25))
            .unwrap();
        assert_close(
            permuted_z,
            normal_z * parity,
            f32::EPSILON,
            "permuted normal parity",
        );
    }
}

#[test]
fn quilting_qb_wasm_matches_rust_differentials_flat_patch_and_s3() {
    let (mut store, instance) = instantiate();
    assert_curved_patch(&mut store, &instance);
    assert_flat_patch(&mut store, &instance);
    assert_pole_fails_closed(&mut store, &instance);
    assert_s3_remaps(&mut store, &instance);
}
