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
fn radial_atlas_fan_wasm_preserves_seams_and_triangle_orientation() {
    let (mut store, instance) = instantiate();
    let warp = function::<(f32,f32,f32,f32,u32),f32>(&mut store,&instance,"radial_warp_lane");
    let fan = function::<(f32,f32,f32,f32,f32,f32,f32,u32,u32),f32>(
        &mut store,&instance,"radial_fan_lane");
    let strengths = [0.0625_f32,0.25,1.0,4.0,16.0];
    let projective = function::<(f32,f32,f32,f32,f32,f32,u32),f32>(
        &mut store,&instance,"projective_triangle_lane");
    // Two triangles may have unrelated opposite vertices/weights. Their edge
    // sampling depends only on the shared endpoints, even in reversed order.
    for wb in [0.25_f32,1.0,4.0] { for wc in [0.25_f32,1.0,4.0] {
        for step in 0..=256 {
            let t = step as f32/256.0;
            let left_b = projective.call(&mut store,(0.0,t,1.0-t,16.0,wb,wc,1)).unwrap();
            let left_c = projective.call(&mut store,(0.0,t,1.0-t,16.0,wb,wc,2)).unwrap();
            let right_b = projective.call(&mut store,(1.0-t,t,0.0,wc,wb,0.0625,1)).unwrap();
            let right_c = projective.call(&mut store,(1.0-t,t,0.0,wc,wb,0.0625,0)).unwrap();
            assert_eq!(left_b.to_bits(),right_b.to_bits(),"shared outer edge endpoint B");
            assert_eq!(left_c.to_bits(),right_c.to_bits(),"shared outer edge endpoint C");
        }
    }}
    for strength in strengths {
        for step in 0..=256 {
            let t = step as f32 / 256.0;
            let outer = [0.0,t,1.0-t];
            for lane in 0..3 {
                let value = warp.call(&mut store,(outer[0],outer[1],outer[2],strength,lane)).unwrap();
                assert_eq!(value.to_bits(),outer[lane as usize].to_bits(),"outer boundary unchanged");
            }
            // Both incident children evaluate the same radial edge, with its
            // outer endpoint in different local lanes. Require bitwise equality.
            for focus in [[0.2_f32,0.3,0.5],[0.001,0.009,0.99]] {
                for child in 0..3_u32 { for lane in 0..3_u32 {
                    let left = fan.call(&mut store,(t,0.0,1.0-t,strength,focus[0],focus[1],focus[2],child,lane)).unwrap();
                    let right = fan.call(&mut store,(t,1.0-t,0.0,strength,focus[0],focus[1],focus[2],(child+1)%3,lane)).unwrap();
                    assert_eq!(left.to_bits(),right.to_bits(),"shared radial edge");
                }}
            }
        }
    }

    let artifact = crate::decode(include_bytes!("../../../fixtures/classic-quilting/v1/direct-seed42-matrix.cqa")).unwrap();
    let signed_area = |p: [[f32;3];3]| {
        let [a,b,c] = p.map(|v| v.map(f64::from));
        (b[1]-a[1])*(c[2]-a[2])-(b[2]-a[2])*(c[1]-a[1])
    };
    for permutation in S3_PERMUTATIONS { for strength in strengths {
        let mapped: Vec<[f32;3]> = artifact.vertices.iter().map(|vertex| {
            let p = permutation.map(|i| vertex.barycentric[i]);
            let result = std::array::from_fn(|lane| warp.call(&mut store,(p[0],p[1],p[2],strength,lane as u32)).unwrap());
            assert!(result.iter().all(|v| *v >= 0.0 && v.is_finite()));
            assert!((result.iter().sum::<f32>()-1.0).abs()<3.0e-7);
            for lane in 0..3 {
                let inverse = warp.call(&mut store,(result[0],result[1],result[2],1.0/strength,lane as u32)).unwrap();
                assert_close(inverse,p[lane],3.0e-7,"inverse radial map");
            }
            result
        }).collect();
        for patch in &artifact.patches {
            let mut covered = 0.0_f64;
            for triangle in &artifact.triangles[patch.first_triangle as usize..(patch.first_triangle+patch.triangle_count) as usize] {
                let original = triangle.indices.map(|i| permutation.map(|lane| artifact.vertices[i as usize].barycentric[lane]));
                let result = triangle.indices.map(|i| mapped[i as usize]);
                let before = signed_area(original);
                let after = signed_area(result);
                assert!(before*after>0.0,"radial map preserves orientation, including odd permutations");
                covered += after.abs();
            }
            assert!((covered-1.0).abs()<2.0e-6,"fixed outer triangle is covered once: {covered}");
        }
    }}
}

#[test]
fn separable_corner_weights_wasm_resolve_symmetrically() {
    let (mut store, instance) = instantiate();
    let resolve = function::<(f32, f32, f32, f32, i32), f32>(
        &mut store, &instance, "resolved_corner_scale",
    );
    let values = [0.02_f32, 0.1, 0.4, 1.0, 3.0, 8.0];
    for a in values { for b in values { for c in values { for d in values {
        let inputs = [a, b, c, d];
        let mut output = [0.0_f32; 4];
        let logs = inputs.map(|x| f64::from(x).ln());
        let residual = (logs[0] - logs[1] - logs[2] + logs[3]) / 4.0;
        for i in 0..4 {
            output[i] = resolve.call(&mut store, (a, b, c, d, i as i32)).unwrap();
            let expected = (logs[i] + if i == 0 || i == 3 { -residual } else { residual }).exp();
            assert!((f64::from(output[i]) / expected - 1.0).abs() < 4.0e-7);
        }
        assert!((output[0] * output[3] / (output[1] * output[2]) - 1.0).abs() < 8.0e-7);
        let low = output.iter().copied().fold(f32::INFINITY, f32::min);
        let high = output.iter().copied().fold(0.0, f32::max);
        assert!(high / low <= 400.001, "a common scale must fit the slider range");
        for i in 0..4 {
            let again = resolve.call(&mut store, (output[0], output[1], output[2], output[3], i as i32)).unwrap();
            assert!((again / output[i] - 1.0).abs() < 4.0e-7, "idempotent correction");
        }
    }}}}
    for (i, expected) in [0.02_f32, 0.4, 0.4, 8.0].into_iter().enumerate() {
        let actual = resolve.call(&mut store, (0.02, 0.4, 0.4, 8.0, i as i32)).unwrap();
        assert!((actual / expected - 1.0).abs() < 4.0e-7, "retain compatible extreme weights");
    }
}

#[test]
fn tangent_triangle_wasm_detects_flattening_and_preserves_scale() {
    let (mut store, instance) = instantiate();
    type Tangents = (f32, f32, f32, f32, f32, f32);
    let quality = function::<Tangents, f32>(&mut store, &instance, "tangent_triangle_quality");
    let area = function::<Tangents, f32>(&mut store, &instance, "tangent_triangle_area");
    let defined = function::<Tangents, i32>(&mut store, &instance, "tangent_triangle_defined");

    // The prior shortest/longest rule scores (1,0,0),(2,epsilon,0) near 1/2
    // even as area vanishes. Verify the production Fe operation reaches zero.
    for height in [1.0_f32, 0.01, 0.0001, 0.000001] {
        let input = (1.0, 0.0, 0.0, 2.0, height, 0.0);
        let expected = 2.0 * 3.0_f64.sqrt() * f64::from(height)
            / (6.0 + 2.0 * f64::from(height).powi(2));
        let actual = quality.call(&mut store, input).unwrap();
        assert_close(actual, expected as f32, 1.0e-7, "near-collinear quality");
        assert_eq!(defined.call(&mut store, input).unwrap(), 1);
    }
    // A rigid axis permutation and 24 orders of uniform scale must not
    // change quality. Area must retain its ordinary square-scale law.
    for scale in [1.0e-12_f32, 1.0, 1.0e12] {
        let height = scale * (3.0_f32.sqrt() * 0.5);
        let input = (0.0, scale, 0.0, 0.0, 0.5 * scale, height);
        assert_close(quality.call(&mut store, input).unwrap(), 1.0, 3.0e-7, "equilateral quality");
        let actual_area = f64::from(area.call(&mut store, input).unwrap());
        let expected_area = 0.5 * f64::from(scale) * f64::from(height);
        assert!((actual_area / expected_area - 1.0).abs() < 3.0e-7);
        assert_eq!(defined.call(&mut store, input).unwrap(), 1);
    }
    for input in [
        (1.0, 0.0, 0.0, 2.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (f32::INFINITY, 0.0, 0.0, 0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0, 0.0, f32::NAN, 0.0),
        (1.0e30, 0.0, 0.0, 0.0, 1.0e30, 0.0),
    ] {
        assert_eq!(defined.call(&mut store, input).unwrap(), 0);
        assert_eq!(quality.call(&mut store, input).unwrap(), 0.0);
        assert_eq!(area.call(&mut store, input).unwrap(), 0.0);
    }
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
