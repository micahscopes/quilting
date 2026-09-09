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
        compile_ingot(&path)
    })
}

/// Shared compiler/validation setup for focused Fe-to-Wasm oracle ingots.
pub(crate) fn compile_ingot(path: &Path) -> Vec<u8> {
    compile_ingot_at_level(path, OptLevel::O2)
}

pub(crate) fn compile_ingot_at_level(path: &Path, opt: OptLevel) -> Vec<u8> {
    let url = Url::from_directory_path(path.canonicalize().unwrap()).unwrap();
    let mut db = DriverDataBase::default();
    db.compilation_settings()
        .set_profile(&mut db)
        .to("release".into());
    assert!(
        !driver::init_ingot(&mut db, &url),
        "oracle ingot initialization diagnostics: {}", path.display()
    );
    let ingot = db.workspace().containing_ingot(&db, url).expect("oracle ingot");
    let top_mod = ingot.root_mod(&db);
    let diagnostics = db.run_on_top_mod(top_mod).format_diags(&db);
    assert!(diagnostics.is_empty(), "unexpected oracle diagnostics:\n{diagnostics}");
    let wasm = BackendKind::Wasm
        .create()
        .compile(&db, top_mod, layout_for(BackendKind::Wasm), opt)
        .unwrap_or_else(|error| panic!("{} should compile to Wasm: {error}", path.display()))
        .into_bytecode()
        .expect("Wasm output should be bytecode");
    wasmparser::validate(&wasm).expect("oracle Wasm should validate");
    wasm
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

#[test]
fn display_rgba_alpha_preserves_rgb_and_all_alpha_bytes() {
    let (mut store, instance) = instantiate();
    let pack = function::<(f32, f32, f32, f32), i32>(&mut store, &instance, "display_rgba_with_alpha");
    for a in 0..=255u32 {
        let actual = pack.call(&mut store, (0.2, 0.4, 0.6, a as f32 / 255.0)).unwrap() as u32;
        assert_eq!(actual, 51 | (102 << 8) | (153 << 16) | (a << 24));
    }
    // Sweep every lane independently, including the signed high-bit boundary.
    // The expected word is assembled from byte identities, not a copy of the
    // authored float packing implementation.
    for lane in 0..4 {
        for value in 0..=255u8 {
            let mut bytes = [17u8, 83, 149, 211];
            bytes[lane] = value;
            let channels = bytes.map(|byte| byte as f32 / 255.0);
            let actual = pack.call(&mut store, (channels[0], channels[1], channels[2], channels[3])).unwrap();
            assert_eq!(actual, i32::from_le_bytes(bytes), "lane={lane} value={value}");
        }
    }
    assert_eq!(pack.call(&mut store, (1.0, 0.0, 0.0, 0.5)).unwrap() as u32, 0x800000ff);
    assert_eq!(pack.call(&mut store, (1.0, 0.0, 0.0, -1.0)).unwrap() as u32, 0x000000ff);
    assert_eq!(pack.call(&mut store, (1.0, 0.0, 0.0, 2.0)).unwrap() as u32, 0xff0000ff);
}

#[test]
fn common_pole_tri_and_quad_match_independent_inversion_and_shared_edges() {
    let (mut store, instance) = instantiate();
    let sample = function::<(i32,f32,f32,f32,f32,f32,i32,i32),f32>(
        &mut store,&instance,"pole_patch_sample");
    let points = [[-1.0_f64,-0.5,0.0],[1.0,-0.5,0.2],[-0.7,1.0,0.4],[0.8,1.1,-0.3]];
    let mut maximum_error = 0.0_f64;
    for pole in [[0.1_f32,0.2,1.3],[2.0,-1.0,0.5],[0.0,0.0,0.01],[-1.001,-0.5,0.0]] {
        for quad in [0,1] {
            for i in 0..=16 {
                for j in 0..=16 {
                    if quad==0 && i+j>16 {continue;}
                    let u=i as f32/16.0;
                    let v=j as f32/16.0;
                    let basis=if quad==0 {[1.0-u-v,u,v,0.0]} else {[(1.0-u)*(1.0-v),u*(1.0-v),(1.0-u)*v,u*v]};
                    // Independently invert the points, interpolate in that
                    // Euclidean chart, then invert back. No Clifford product
                    // implementation or constructed Fe weights is reused here.
                    let mut s=[0.0_f64;3];
                    for (point,b) in points.iter().zip(basis) {
                        let d=std::array::from_fn::<_,3,_>(|k|point[k]-f64::from(pole[k]));
                        let squared=d.iter().map(|x|x*x).sum::<f64>();
                        for k in 0..3 {s[k]+=f64::from(b)*d[k]/squared;}
                    }
                    let squared=s.iter().map(|x|x*x).sum::<f64>();
                    assert!(squared>1e-12,"fixture must avoid the interior pole");
                    let expected=std::array::from_fn::<_,3,_>(|k|f64::from(pole[k])+s[k]/squared);
                    for gauge in [0,1] {
                        assert_eq!(sample.call(&mut store,(quad,u,v,pole[0],pole[1],pole[2],gauge,4)).unwrap(),1.0);
                        for lane in 0..4 {
                            let actual=f64::from(sample.call(&mut store,(quad,u,v,pole[0],pole[1],pole[2],gauge,lane)).unwrap());
                            let target=if lane<3 {expected[lane as usize]} else {0.0};
                            let error=(actual-target).abs()/(1.0+target.abs());
                            maximum_error=maximum_error.max(error);
                            assert!(error<1e-5,"quad={quad} pole={pole:?} uv={u},{v} gauge={gauge} lane={lane}: {actual} != {target}");
                        }
                    }
                }
            }
        }
        for i in 0..=64 {
            let u=i as f32/64.0;
            for lane in 0..3 {
                let tri=sample.call(&mut store,(0,u,0.0,pole[0],pole[1],pole[2],0,lane)).unwrap();
                let quad=sample.call(&mut store,(1,u,0.0,pole[0],pole[1],pole[2],0,lane)).unwrap();
                assert!((tri-quad).abs()<1e-5*(1.0+tri.abs()),"shared boundary");
            }
        }
    }
    for pole in [[-1.0,-0.5,0.0],[1.0,-0.5,0.2],[f32::NAN,0.0,0.0],[f32::INFINITY,0.0,0.0]] {
        assert_eq!(sample.call(&mut store,(1,0.2,0.3,pole[0],pole[1],pole[2],0,4)).unwrap(),-1.0,"reject a corner at the pole or a nonfinite pole");
    }
    eprintln!("common-pole max normalized coordinate/residual error: {maximum_error}");
}

#[test]
fn edge_authored_patch_preserves_midpoint_corners_and_reversal() {
    let (mut store,instance)=instantiate();
    let sample=function::<(i32,f32,f32,i32,i32,f32,f32,f32,i32),f32>(
        &mut store,&instance,"edge_patch_sample");
    let corners=[[-1.0_f32,-0.5,0.0],[1.0,-0.5,0.2],[-0.7,1.0,0.4],[0.8,1.1,-0.3]];
    for quad in [0,1] {
        let edges=if quad==0 {vec![(0,1,[0.5,0.0]),(0,2,[0.0,0.5]),(1,2,[0.5,0.5])]}
            else {vec![(0,1,[0.5,0.0]),(0,2,[0.0,0.5]),(1,3,[1.0,0.5]),(2,3,[0.5,1.0])]};
        for (first,last,uv) in edges {
            for offset in [[0.1_f32,0.4,0.7],[-0.2,0.3,-0.6]] {
                let middle=std::array::from_fn::<_,3,_>(|k|(corners[first][k]+corners[last][k])*0.5+offset[k]);
                for reverse in [false,true] {
                    let (a,b)=if reverse {(last,first)} else {(first,last)};
                    let mut value=|u:f32,v:f32,lane:i32|sample.call(&mut store,(quad,u,v,a as i32,b as i32,middle[0],middle[1],middle[2],lane)).unwrap();
                    assert_eq!(value(uv[0],uv[1],4),1.0,"admitted edge midpoint");
                    for lane in 0..3 {
                        assert!((value(uv[0],uv[1],lane)-middle[lane as usize]).abs()<2e-5,
                            "quad={quad} edge={a},{b} midpoint={middle:?} lane={lane}");
                    }
                    let coordinates=[[0.0_f32,0.0],[1.0,0.0],[0.0,1.0],[1.0,1.0]];
                    for step in 0..=16 {
                        let t=step as f32/16.0;
                        let u=coordinates[a][0]*(1.0-t)+coordinates[b][0]*t;
                        let v=coordinates[a][1]*(1.0-t)+coordinates[b][1]*t;
                        let expected=crate::clifford_oracle::authored_arc_between_reference(
                            [1.0;3],corners[a].map(f64::from),middle.map(f64::from),
                            corners[b].map(f64::from),false,f64::from(t)).unwrap();
                        assert_eq!(value(u,v,4),1.0);
                        for lane in 0..3 {
                            assert!((f64::from(value(u,v,lane))-expected[lane as usize]).abs()
                                <2e-5*(1.0+expected[lane as usize].abs()),
                                "whole boundary arc: quad={quad} edge={a},{b} t={t}");
                        }
                    }
                    for (index,[u,v]) in [[0.0,0.0],[1.0,0.0],[0.0,1.0],[1.0,1.0]].into_iter().enumerate() {
                        if quad==0 && index==3 {continue;}
                        assert_eq!(value(u,v,4),1.0);
                        for lane in 0..3 {assert!((value(u,v,lane)-corners[index][lane as usize]).abs()<2e-5,"corners stay fixed");}
                    }
                }
            }
        }
    }
    // An affine edge has its parameter-infinity limit at infinity. This finite
    // pole family rejects that limit instead of inventing a finite substitute.
    assert_eq!(sample.call(&mut store,(1,0.5,0.0,0,1,0.0,-0.5,0.1,4)).unwrap(),-1.0);
    for (a,b) in [(0,4),(4,0),(0,0)] {
        assert_eq!(sample.call(&mut store,(1,0.5,0.0,a,b,0.0,0.4,0.7,4)).unwrap(),-1.0);
    }
}

#[test]
fn authored_arc_weights_wasm_match_dense_algebras_and_geometric_handles() {
    let (mut store,instance)=instantiate();
    let sample=function::<(i32,i32,f32,f32,f32,f32,i32),f32>(&mut store,&instance,"authored_arc_sample");
    let oblique=function::<(i32,f32,f32,f32,f32,i32),f32>(&mut store,&instance,"authored_oblique_arc_sample");
    let interval=function::<(i32,f32,f32),i32>(&mut store,&instance,"arc_segment_regular");
    assert_eq!(interval.call(&mut store,(1,0.2,0.3)).unwrap(),0,"never bridge the pole at 1/4");
    assert_eq!(interval.call(&mut store,(1,0.0,0.2)).unwrap(),1,"retain first branch");
    assert_eq!(interval.call(&mut store,(1,0.3,1.0)).unwrap(),1,"retain second branch");
    for step in 0..64 {
        assert_eq!(interval.call(&mut store,(0,step as f32/64.0,(step+1) as f32/64.0)).unwrap(),1,"finite semicircle segments");
    }
    let handles=[[0.0_f32,1.0,0.0],[0.3,0.4,0.7],[-0.75,0.1,-1.3],[0.0,0.0,0.0]];
    for (model,metric) in [[1.0,1.0,1.0],[1.0,1.0,0.0]].into_iter().enumerate() {
        for h in handles {
            for kind in 0..=1 {
                if kind==1 && h==[0.0;3] {continue;}
                for step in 0..=64 {
                    let t=step as f32/64.0;
                    let expected=crate::clifford_oracle::authored_arc_reference(metric,h.map(f64::from),kind==1,f64::from(t));
                    let Some(expected)=expected else {continue;};
                    let defined=sample.call(&mut store,(model as i32,kind,h[0],h[1],h[2],t,4)).unwrap();
                    assert_eq!(defined,1.0,"accepted finite arc, model={model} kind={kind} handle={h:?} t={t}");
                    for lane in 0..4 {
                        let value=sample.call(&mut store,(model as i32,kind,h[0],h[1],h[2],t,lane)).unwrap();
                        let tolerance=2.0e-5*(1.0+expected[lane as usize].abs());
                        assert!((f64::from(value)-expected[lane as usize]).abs()<tolerance,
                            "model={model} kind={kind} handle={h:?} t={t} lane={lane}: {value} vs {expected:?}");
                        if kind==0 && (step==0 || step==32 || step==64) && lane<3 {
                            let point=if step==0 {[-1.0,0.0,0.0]} else if step==64 {[1.0,0.0,0.0]} else {h};
                            assert!((value-point[lane as usize]).abs()<3.0e-6,"handle interpolation");
                        }
                    }
                }
            }
        }
        for (kind,h) in [(0,[-1.0,0.0,0.0]),(0,[1.0,0.0,0.0]),(1,[0.0,0.0,0.0]),
            (0,[f32::NAN,0.0,0.0]),(0,[0.0,f32::INFINITY,0.0])] {
            assert_eq!(sample.call(&mut store,(model as i32,kind,h[0],h[1],h[2],0.5,4)).unwrap(),-999.0,"invalid construction");
        }
    }
    // A collinear exterior handle selects a projective arc crossing infinity.
    // The construction is meaningful, but its pole is not a drawable point.
    assert_eq!(sample.call(&mut store,(0,0,2.0,0.0,0.0,0.25,4)).unwrap(),0.0);
    // A purely null displacement is rejected only in the isotropic metric.
    assert_eq!(sample.call(&mut store,(1,1,0.0,0.0,1.0,0.5,4)).unwrap(),-999.0);
    assert_eq!(sample.call(&mut store,(0,1,0.0,0.0,1.0,0.5,4)).unwrap(),1.0);
    // Non-axis-aligned endpoints exercise all three bivector coefficients.
    let a=[-0.8_f32,0.3,0.4];
    let b=[0.7_f32,-0.6,1.1];
    for (model,metric) in [[1.0,1.0,1.0],[1.0,1.0,0.0]].into_iter().enumerate() {
        for h in handles { for step in 0..=64 {
            let t=step as f32/64.0;
            let expected=crate::clifford_oracle::authored_arc_between_reference(metric,a.map(f64::from),h.map(f64::from),b.map(f64::from),false,f64::from(t)).unwrap();
            for lane in 0..4 {
                let actual=oblique.call(&mut store,(model as i32,h[0],h[1],h[2],t,lane)).unwrap();
                assert!((f64::from(actual)-expected[lane as usize]).abs()<2.0e-5*(1.0+expected[lane as usize].abs()),
                    "oblique model={model} handle={h:?} t={t} lane={lane}: {actual} vs {expected:?}");
            }
        }}
    }
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
fn projective_control_pullback_wasm_matches_exact_surface_and_normals() {
    let (mut store,instance) = instantiate();
    let evaluate = function::<(f32,f32,f32,u32),f32>(&mut store,&instance,"reparameterized_patch_lane");
    for strength in [0.0625_f32,0.25,1.0,4.0,16.0] {
        for i in 1..16 { for j in 1..16-i {
            let b = i as f32/16.0;
            let c = j as f32/16.0;
            let denominator = 1.0 + (f64::from(strength)-1.0)*f64::from(c);
            let wb = f64::from(b)/denominator;
            let wc = f64::from(c)*f64::from(strength)/denominator;
            let u = wb + f64::from(0.43_f32)*wc;
            let v = f64::from(0.58_f32)*wc;
            let (expected,_) = crate::clifford_oracle::paper_sample(u,v);
            for lane in 0..3 {
                let actual = evaluate.call(&mut store,(b,c,strength,lane as u32)).unwrap();
                assert_close(actual,expected[lane] as f32,0.00003*(1.0+expected[lane].abs()) as f32,"control pullback vs direct analytic evaluation");
            }
            let h = 1.0e-6;
            let (pu,_) = crate::clifford_oracle::paper_sample(u+h,v);
            let (mu,_) = crate::clifford_oracle::paper_sample(u-h,v);
            let (pv,_) = crate::clifford_oracle::paper_sample(u,v+h);
            let (mv,_) = crate::clifford_oracle::paper_sample(u,v-h);
            let du: [f64;3] = std::array::from_fn(|k| pu[k]-mu[k]);
            let dv: [f64;3] = std::array::from_fn(|k| pv[k]-mv[k]);
            let cross = [du[1]*dv[2]-du[2]*dv[1],du[2]*dv[0]-du[0]*dv[2],du[0]*dv[1]-du[1]*dv[0]];
            let length = cross.iter().map(|x| x*x).sum::<f64>().sqrt();
            for lane in 0..3 {
                let actual = evaluate.call(&mut store,(b,c,strength,(lane+3) as u32)).unwrap();
                assert_close(actual,(cross[lane]/length) as f32,0.0005,"pullback analytic normal vs independent f64 finite difference");
            }
        }}
    }
}

#[test]
fn nested_triangle_fans_wasm_preserve_domain_surface_and_diagonal() {
    let (mut store, instance) = instantiate();
    let sample = function::<(u32,u32,f32,f32,f32,f32,f32,u32),f32>(
        &mut store,&instance,"nested_fan_sample_lane");
    for (fb,fc,strength) in [(1.0,1.0,1.0),(0.01,100.0,0.0625),(100.0,0.01,16.0),(0.25,4.0,4.0)] {
        let mut area = 0.0_f64;
        for half in 0..2 { for child in 0..3 {
            let mut uv = [[0.0_f64;2];3];
            for (i,(a,b)) in [(1.0,0.0),(0.0,1.0),(0.0,0.0)].into_iter().enumerate() {
                for lane in 0..2 {
                    uv[i][lane] = sample.call(&mut store,(half,child,fb,fc,strength,a,b,lane as u32)).unwrap() as f64;
                }
            }
            let signed = (uv[1][0]-uv[0][0])*(uv[2][1]-uv[0][1])-(uv[1][1]-uv[0][1])*(uv[2][0]-uv[0][0]);
            assert!(signed>0.0,"each child retains orientation");
            area += signed*0.5;
            for (a,b) in [(0.25,0.25),(0.5,0.25),(0.125,0.625)] {
                let u=sample.call(&mut store,(half,child,fb,fc,strength,a,b,0)).unwrap();
                let v=sample.call(&mut store,(half,child,fb,fc,strength,a,b,1)).unwrap();
                let (expected,_) = crate::clifford_oracle::paper_sample(u as f64,v as f64);
                for lane in 0..3 {
                    let actual=sample.call(&mut store,(half,child,fb,fc,strength,a,b,(lane+2) as u32)).unwrap();
                    assert_close(actual,expected[lane] as f32,0.0001*(1.0+expected[lane].abs()) as f32,"nested exact child vs original surface");
                }
            }
        }}
        assert!((area-1.0).abs()<0.000002,"six children partition the square");
    }
    let mut child_normal_disagreement = 0.0_f32;
    for i in 0..=256 {
        let t=i as f32/256.0;
        for lane in 0..14 {
            // First half's diagonal is C->A, second half's is A->B.
            // Independent interior positions and concentrations must not move it.
            let left=sample.call(&mut store,(0,1,0.01,100.0,16.0,0.0,t,lane)).unwrap();
            let right=sample.call(&mut store,(1,2,100.0,0.01,0.0625,0.0,1.0-t,lane)).unwrap();
            if (5..8).contains(&lane) {
                // Thin children amplify f32 cancellation in their reconstructed
                // differential. Display uses the original differential instead.
                child_normal_disagreement=child_normal_disagreement.max((left-right).abs());
                assert!(left.is_finite() && right.is_finite());
            } else {
                assert_close(left,right,0.0001,"independent fans share diagonal; root evaluation avoids child differential cancellation");
            }
            if lane<2 || lane>=8 {assert_eq!(left.to_bits(),right.to_bits(),"shared root evaluation matches bitwise");}
        }
    }
    eprintln!("thin child-net maximum normal-component disagreement: {child_normal_disagreement}");
}

#[test]
fn sampling_warps_wasm_preserve_independent_boundaries_and_interior_domain() {
    let (mut store,instance)=instantiate();
    let map=function::<(u32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,u32),f32>(
        &mut store,&instance,"sampling_warp_lane");
    let mut sample=|domain:u32,p:[f32;2],s:[f32;8]| -> [f32;2] {
        std::array::from_fn(|lane|map.call(&mut store,
            (domain,p[0],p[1],s[0],s[1],s[2],s[3],s[4],s[5],s[6],s[7],lane as u32)).unwrap())
    };
    let identity=[1.0,1.0,1.0,1.0,0.0,1.0,1.0,1.0];
    for domain in 0..2 {
        for x in 0..=16 {for y in 0..=16 {
            if domain==0 && x+y>16 {continue;}
            let p=[x as f32/16.0,y as f32/16.0];
            assert_eq!(sample(domain,p,identity),p,"neutral map is exact identity");
        }}
        for edge in 0..if domain==0 {3} else {4} {
            for k in [0.0625_f32,0.25,1.0,4.0,16.0] {
                let mut settings=[4.0,0.25,16.0,0.0625,0.0,1.0,1.0,1.0];
                settings[edge]=k;
                for i in 0..=256 {
                    let t=i as f32/256.0;
                    let p=if domain==0 {
                        match edge {0=>[1.0-t,t],1=>[0.0,1.0-t],_=>[t,0.0]}
                    } else {
                        match edge {0=>[t,0.0],1=>[1.0,t],2=>[t,1.0],_=>[0.0,t]}
                    };
                    let f=(k as f64*t as f64/(1.0+(k as f64-1.0)*t as f64)) as f32;
                    let expected=if domain==0 {
                        match edge {0=>[1.0-f,f],1=>[0.0,1.0-f],_=>[f,0.0]}
                    } else {
                        match edge {0=>[f,0.0],1=>[1.0,f],2=>[f,1.0],_=>[0.0,f]}
                    };
                    let baseline=sample(domain,p,settings);
                    for lane in 0..2 {assert_close(baseline[lane],expected[lane],0.000003,"only the edge's own skew applies");}
                    for interior in [[6.0,0.02,50.0,16.0],[-6.0,50.0,0.02,0.0625]] {
                        let mut changed=settings;
                        changed[4..8].copy_from_slice(&interior);
                        assert_eq!(sample(domain,p,changed),baseline,"twist/focus/concentration leave every boundary sample unchanged");
                    }
                }
            }
        }
        for settings in [identity,[0.0625,16.0,4.0,0.25,6.0,0.02,50.0,16.0],
            [4.0,0.25,16.0,0.0625,-6.0,50.0,0.02,0.0625]] {
            for x in 0..=24 {for y in 0..=24 {
                if domain==0 && x+y>24 {continue;}
                let q=sample(domain,[x as f32/24.0,y as f32/24.0],settings);
                assert!(q.iter().all(|v|v.is_finite() && *v>=-0.000003 && *v<=1.000003),"finite in-domain point: {q:?}");
                if domain==0 {assert!(q[0]+q[1]<=1.000003,"inside triangle");}
            }}
        }
        for (fx,fy) in [(1.0,1.0),(0.02,50.0),(50.0,0.02)] {
            let p=if domain==0 {[fx/(1.0+fx+fy),fy/(1.0+fx+fy)]} else {[fx/(1.0+fx),fy/(1.0+fy)]};
            let q=sample(domain,p,[1.0,1.0,1.0,1.0,6.0,fx,fy,16.0]);
            for lane in 0..2 {assert_close(q[lane],p[lane],0.000003,"interior focus remains fixed");}
        }
    }
    // A monotone continuous warp need not preserve orientation of a coarse
    // straight-edge display mesh. Record rather than conceal this distinction.
    let settings=[1.0,1.0,1.0,1.0,6.0,1.0,1.0,4.0];
    let mut folded=0;
    for x in 0..8 {for y in 0..8 {
        let p=[[x as f32/8.0,y as f32/8.0],[(x+1) as f32/8.0,y as f32/8.0],
            [(x+1) as f32/8.0,(y+1) as f32/8.0],[x as f32/8.0,(y+1) as f32/8.0]].map(|p|sample(1,p,settings));
        for [a,b,c] in [[0,1,2],[0,2,3]] {
            let area=(p[b][0]-p[a][0])*(p[c][1]-p[a][1])-(p[b][1]-p[a][1])*(p[c][0]-p[a][0]);
            folded+=usize::from(area<=0.0);
        }
    }}
    eprintln!("coarse twisted square: {folded}/128 straight triangles inverted or collapsed");
}

#[test]
fn projective_interval_inverse_uniformizes_a_rational_line() {
    let (mut store,instance)=instantiate();
    let bias=function::<(f32,f32),f32>(&mut store,&instance,"sampling_interval_bias");
    for k in [0.0625_f32,0.25,1.0,4.0,16.0] {
        let mut previous=-1.0_f32;
        for i in 0..=256 {
            let fraction=i as f32/256.0;
            let parameter=bias.call(&mut store,(fraction,1.0/k)).unwrap();
            assert!(parameter>previous,"inverse map is strictly monotone");
            let distance=bias.call(&mut store,(parameter,k)).unwrap();
            assert_close(distance,fraction,0.000003,"analytic inverse gives uniformly spaced points on this rational straight edge");
            previous=parameter;
        }
    }
}

#[test]
fn quad_atlas_wasm_canonicalizes_d4_and_preserves_exact_boundary_rings() {
    let (mut store,instance)=instantiate();
    let canonical=function::<(u32,u32,u32,u32,u32),u32>(&mut store,&instance,"quad_key_lane");
    let transformed=function::<(u32,u32,u32,u32,u32),u32>(&mut store,&instance,"quad_transformed_key");
    let point=function::<(u32,u32,u32,u32,u32),u32>(&mut store,&instance,"quad_transformed_point_lane");
    let boundary=function::<(u32,u32,u32,u32,u32,u32),u32>(&mut store,&instance,"quad_boundary_lane");
    let decode=|code:u32| [code/729,(code/81)%9,(code/9)%9,code%9];
    let encode=|k:[u32;4]| ((k[0]*9+k[1])*9+k[2])*9+k[3];
    let mut classes=std::collections::BTreeSet::new();
    for code in 0..6561 {
        let k=decode(code);
        let [a,b,c,d]=k;
        // Independent dihedral action on a circular sequence. Reflection
        // reverses the sequence; rotations select each possible starting edge.
        let expected=(0..4).flat_map(|r| [std::array::from_fn(|i|k[(r+i)%4]),
            std::array::from_fn(|i|k[(r+4-i)%4])]).map(encode).min().unwrap();
        let actual=canonical.call(&mut store,(a,b,c,d,0)).unwrap();
        assert_eq!(actual,expected,"D4 key class {k:?}");
        let witness=canonical.call(&mut store,(a,b,c,d,1)).unwrap();
        assert_eq!(transformed.call(&mut store,(a,b,c,d,witness)).unwrap(),actual);
        classes.insert(actual);
    }
    assert_eq!(classes.len(),1035);
    assert_ne!(canonical.call(&mut store,(0,0,1,1,0)).unwrap(),canonical.call(&mut store,(0,1,0,1,0)).unwrap(),
        "adjacent high edges and opposite high edges are different classes");
    let mut boundary_samples=0;
    for code in classes {
        let [a,b,c,d]=decode(code);
        let counts=[1<<a,1<<b,1<<c,1<<d];
        let total:u32=counts.iter().sum();
        assert_eq!(canonical.call(&mut store,(a,b,c,d,2)).unwrap(),total);
        let mut ordinal=0;
        let mut ring=Vec::new();
        for (edge,count) in counts.into_iter().enumerate() {
            for step in 0..count {
                let t=step*(16384/count);
                let expected=match edge {0=>[t,0],1=>[16384,t],2=>[16384-t,16384],_=>[0,16384-t]};
                let p=std::array::from_fn(|lane|boundary.call(&mut store,(a,b,c,d,ordinal,lane as u32)).unwrap());
                assert_eq!(p,expected,"canonical ring {code}, edge {edge}, step {step}");
                ring.push(p);
                ordinal+=1;
            }
        }
        assert_eq!(ring.iter().copied().collect::<std::collections::BTreeSet<_>>().len(),total as usize);
        let area:i64=(0..ring.len()).map(|i| {
            let p=ring[i].map(i64::from);
            let q=ring[(i+1)%ring.len()].map(i64::from);
            p[0]*q[1]-p[1]*q[0]
        }).sum();
        assert_eq!(area,2*16384_i64.pow(2),"positive full-square boundary area");
        assert_eq!(boundary.call(&mut store,(a,b,c,d,total,4)).unwrap(),0,"no out-of-range seed");
        boundary_samples+=total;
    }
    for symmetry in 0..8 {
        for x in [0,1,257,8192,16384] {for y in [0,1,257,8192,16384] {
            let qx=point.call(&mut store,(x,y,symmetry,0,0)).unwrap();
            let qy=point.call(&mut store,(x,y,symmetry,0,1)).unwrap();
            assert_eq!(point.call(&mut store,(qx,qy,symmetry,1,0)).unwrap(),x);
            assert_eq!(point.call(&mut store,(qx,qy,symmetry,1,1)).unwrap(),y);
        }}
    }
    assert_eq!(canonical.call(&mut store,(0,0,0,9,3)).unwrap(),0,"invalid LoD rejected");
    eprintln!("6,561 requested quad keys; 1,035 canonical rings; {boundary_samples} exact boundary samples");
}

#[test]
fn sampled_length_inverse_wasm_is_monotone_and_lod_tracks_curved_length() {
    let (mut store,instance)=instantiate();
    let inverse=function::<(u32,f32),f32>(&mut store,&instance,"inverse_length_test");
    let lod=function::<(f32,f32,u32),u32>(&mut store,&instance,"measured_edge_lod");
    for kind in 0..4 {
        let mut previous=0.0;
        for step in 0..=4096 {
            let t=step as f32/4096.0;
            let x=inverse.call(&mut store,(kind,t)).unwrap();
            assert!(x>=previous && x<=1.0,"monotone inverse, kind {kind}, step {step}");
            previous=x;
            if step==0 || step==4096 {assert_eq!(x,t,"exact endpoints");}
            if kind==0 || kind==3 {assert_eq!(x,t,"linear CDF / explicitly unmeasured fallback");}
            if kind==1 {assert!((x*x-t).abs()<=0.000004,"piecewise-linear length approximation error");}
            if kind==2 && step>0 {assert!((2.0*x-1.0-t).abs()<=0.000001,"zero-length prefix is skipped");}
        }
    }
    for maximum in 0..=8 {
        for length in [0.01_f32,0.1,1.0,3.7,12.0,100.0] {
            for spacing in [0.01_f32,0.18,1.0,2.0] {
                let actual=lod.call(&mut store,(length,spacing,maximum)).unwrap();
                let expected=(0..=maximum).find(|&l|spacing*(1_u32<<l) as f32>=length).unwrap_or(maximum);
                assert_eq!(actual,expected,"LoD by arc length, with explicit cap");
            }
        }
    }
}

#[test]
fn planar_metric_incircle_wasm_matches_independent_i128() {
    let (mut store,instance)=instantiate();
    let predicate=function::<(u32,u32,u32,u32,u32,u32,u32,u32,u32),i32>(&mut store,&instance,"planar_incircle_lane");
    let expected=|p:[[u32;2];4],metric:u32|->i32 {
        let d: [[i128;2];3]=std::array::from_fn(|i|std::array::from_fn(|j|p[i][j] as i128-p[3][j] as i128));
        let lift=|q:[i128;2]| q[0]*q[0]+q[1]*q[1]+if metric==0 {q[0]*q[1]} else {0};
        let cross=|a:[i128;2],b:[i128;2]| a[0]*b[1]-a[1]*b[0];
        let determinant=lift(d[0])*cross(d[1],d[2])+lift(d[1])*cross(d[2],d[0])+lift(d[2])*cross(d[0],d[1]);
        let u=[p[1][0] as i128-p[0][0] as i128,p[1][1] as i128-p[0][1] as i128];
        let v=[p[2][0] as i128-p[0][0] as i128,p[2][1] as i128-p[0][1] as i128];
        (determinant.signum()*cross(u,v).signum()) as i32
    };
    let mut state=0x921cd61_u64;
    for index in 0..10000 {
        let p=if index==0 {[[0,0],[16384,0],[16384,16384],[0,16384]]}
            else if index==1 {[[0,0],[8192,8192],[16384,16384],[8192,1]]}
            else {std::array::from_fn(|_|std::array::from_fn(|_| {
                state=state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((state>>32)%16385) as u32
            }))};
        for metric in 0..2 {
            let actual=predicate.call(&mut store,(p[0][0],p[0][1],p[1][0],p[1][1],p[2][0],p[2][1],p[3][0],p[3][1],metric)).unwrap();
            assert_eq!(actual,expected(p,metric),"metric {metric}, {p:?}");
            if index==0 && metric==1 {assert_eq!(actual,0,"square corners are cocircular in the square metric");}
            if index==0 && metric==0 {assert_ne!(actual,0,"equilateral metric must not leak into square triangulation");}
        }
    }
}

#[test]
fn quad_sampling_wasm_preserves_edge_density_and_square_symmetry() {
    let (mut store,instance)=instantiate();
    let density=function::<(u32,u32,u32,u32,u32,u32,u32),u32>(&mut store,&instance,"quad_density_lane");
    let transform=function::<(u32,u32,u32,u32,u32),u32>(&mut store,&instance,"quad_transformed_key");
    let point=function::<(u32,u32,u32,u32,u32),u32>(&mut store,&instance,"quad_transformed_point_lane");
    let decode=|code:u32| [code/729,(code/81)%9,(code/9)%9,code%9];
    let mut checks=0;
    for code in 0..6561_u32 {
        let [a,b,c,d]=decode(code);
        for t in [1,64,8192,16320,16383] {
            for (x,y,edge) in [(t,0,a),(16384,t,b),(t,16384,c),(0,t,d)] {
                assert_eq!(density.call(&mut store,(a,b,c,d,x,y,0)).unwrap(),edge*256,
                    "open edge owns density, key {code} at {x},{y}");
                assert_eq!(density.call(&mut store,(a,b,c,d,x,y,1)).unwrap(),1_u32<<(28-2*edge));
                checks+=1;
            }
        }
        for (x,y,expected) in [(0,0,a.max(d)),(16384,0,a.max(b)),
            (16384,16384,b.max(c)),(0,16384,c.max(d))] {
            assert_eq!(density.call(&mut store,(a,b,c,d,x,y,0)).unwrap(),expected*256);
        }
        assert_eq!(density.call(&mut store,(a,b,c,d,8192,8192,0)).unwrap(),(a+b+c+d)*64);
        let x=(code*7187+1)%16385;
        let y=(code*3559+3)%16385;
        let reference=density.call(&mut store,(a,b,c,d,x,y,0)).unwrap();
        assert!((a.min(b).min(c).min(d)*256..=a.max(b).max(c).max(d)*256).contains(&reference));
        for symmetry in 0..8 {
            let [e,f,g,h]=decode(transform.call(&mut store,(a,b,c,d,symmetry)).unwrap());
            let qx=point.call(&mut store,(x,y,symmetry,0,0)).unwrap();
            let qy=point.call(&mut store,(x,y,symmetry,0,1)).unwrap();
            assert_eq!(density.call(&mut store,(e,f,g,h,qx,qy,0)).unwrap(),reference,
                "density equivariant under D4, key {code}, symmetry {symmetry}");
        }
    }
    assert_eq!(density.call(&mut store,(9,0,0,0,8192,8192,2)).unwrap(),0);
    eprintln!("all 6,561 quad keys: {checks} exact open-edge densities, corner/center rules, D4 covariance");
}

#[test]
fn planar_topology_wasm_preserves_incidence_orientation_and_area() {
    let (mut store,instance)=instantiate();
    let location=function::<(u32,u32,u32,u32,u32,u32,u32,u32,u32,u32),u32>(&mut store,&instance,"planar_location_lane");
    let cross=|a:[u32;2],b:[u32;2],c:[u32;2]| {
        let a=a.map(i64::from); let b=b.map(i64::from); let c=c.map(i64::from);
        (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    };
    for x in (0..=16384_u32).step_by(1024) {
        for y in (0..=16384_u32).step_by(1024) {
            let expected=if y>x {0} else if [[0,0],[16384,0],[16384,16384]].contains(&[x,y]) {3}
                else if y==0 || x==16384 || y==x {2} else {1};
            assert_eq!(location.call(&mut store,(0,0,16384,0,16384,16384,x,y,0,0)).unwrap(),expected);
            for mode in [2,3] {
                let count=if mode==2 {3} else {2};
                let expected_valid=if mode==2 {expected==1} else {y==0 && x>0 && x<16384};
                let mut area=0;
                for child in 0..count {
                    let triangle:[u32;4]=std::array::from_fn(|lane|location.call(&mut store,
                        (0,0,16384,0,16384,16384,x,y,mode,child*4+lane as u32)).unwrap());
                    assert_eq!(triangle[3],u32::from(expected_valid));
                    if expected_valid {
                        let points=[[0,0],[16384,0],[16384,16384],[x,y]];
                        let signed=cross(points[triangle[0] as usize],points[triangle[1] as usize],points[triangle[2] as usize]);
                        assert!(signed>0,"every inserted triangle is nondegenerate CCW");
                        area+=signed;
                    }
                }
                if expected_valid {assert_eq!(area,16384_i64.pow(2),"split preserves exact area");}
            }
        }
    }
    // A collinear point beyond the segment is not an edge insertion.
    assert_eq!(location.call(&mut store,(0,0,8192,0,8192,8192,12288,0,3,3)).unwrap(),0);
    let normalized:[u32;4]=std::array::from_fn(|lane|location.call(&mut store,
        (0,0,16384,16384,16384,0,0,0,1,lane as u32)).unwrap());
    assert_eq!(normalized,[0,2,1,1]);
}

#[test]
fn planar_topology_wasm_flips_match_exact_metric_and_preserve_area() {
    let (mut store,instance)=instantiate();
    let flip=function::<(u32,u32,u32,u32,u32,u32,u32,u32,u32,u32),i32>(&mut store,&instance,"planar_flip_lane");
    let cross=|a:[u32;2],b:[u32;2],c:[u32;2]| {
        let a=a.map(i128::from); let b=b.map(i128::from); let c=c.map(i128::from);
        (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    };
    let mut state=0x738190ab_u64;
    let mut flipped=0;
    for index in 0..4000 {
        let [u,v,a,b]=if index==0 {[[0,0],[16384,16384],[0,16384],[16384,0]]}
            else {std::array::from_fn(|_| std::array::from_fn(|_| {
                state=state.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((state>>32)%16385) as u32
            }))};
        let convex=cross(u,v,a)>0 && cross(v,u,b)>0 && cross(a,u,b)>0 && cross(b,v,a)>0;
        for metric in 0..2 {
            let d:[[_;2];3]=[u,v,a].map(|p| [p[0] as i128-b[0] as i128,p[1] as i128-b[1] as i128]);
            let lift=|p:[i128;2]| p[0]*p[0]+p[1]*p[1]+if metric==0 {p[0]*p[1]} else {0};
            let wedge=|p:[i128;2],q:[i128;2]| p[0]*q[1]-p[1]*q[0];
            let determinant=lift(d[0])*wedge(d[1],d[2])+lift(d[1])*wedge(d[2],d[0])+lift(d[2])*wedge(d[0],d[1]);
            let circle=if convex {determinant.signum() as i32} else {0};
            // On a tie, candidate diagonal (0,1) precedes existing (2,3).
            let should_flip=convex && circle>=0;
            let mut lane=|n| flip.call(&mut store,(u[0],u[1],v[0],v[1],a[0],a[1],b[0],b[1],metric,n)).unwrap();
            assert_eq!(lane(0),circle);
            assert_eq!(lane(1),i32::from(convex));
            assert_eq!(lane(2),i32::from(should_flip));
            let first:[i32;4]=std::array::from_fn(|i|lane(3+i as u32));
            let second:[i32;4]=std::array::from_fn(|i|lane(7+i as u32));
            assert_eq!(first[3],i32::from(should_flip));
            assert_eq!(second[3],i32::from(should_flip));
            if should_flip {
                let p=[a,b,u,v];
                let area1=cross(p[first[0] as usize],p[first[1] as usize],p[first[2] as usize]);
                let area2=cross(p[second[0] as usize],p[second[1] as usize],p[second[2] as usize]);
                assert!(area1>0 && area2>0);
                assert_eq!(area1+area2,cross(u,v,a)+cross(v,u,b));
                flipped+=1;
            }
        }
    }
    assert!(flipped>50,"fixtures must exercise actual topology mutations");
    eprintln!("{flipped} exact metric-qualified flips preserve area and orientation");
}

#[test]
fn atlas_topology_wasm_locks_triangle_and_quad_boundary_chains() {
    let (mut store,instance)=instantiate();
    let locked=function::<(u32,u32,u32,u32,u32,u32,u32),u32>(&mut store,&instance,"atlas_boundary_segment");
    for a in 0..=8_u32 {for b in a..=8 {for c in b..=8 {
        let ra=1<<a; let rb=1<<b; let rc=1<<c;
        let count=ra+rb+rc;
        // Independent circular order from the triangle sampler's AB/AC/BC storage.
        let ring:Vec<u32>=(0..=rc).chain((rc+rb+1)..count).chain(std::iter::once(rc+rb))
            .chain(((rc+1)..(rc+rb)).rev()).collect();
        assert_eq!(ring.len(),count as usize);
        for i in 0..ring.len() {
            let u=ring[i]; let v=ring[(i+1)%ring.len()];
            assert_eq!(locked.call(&mut store,(a,b,c,0,u,v,0)).unwrap(),1);
            assert_eq!(locked.call(&mut store,(a,b,c,0,v,u,0)).unwrap(),1);
            assert_eq!(locked.call(&mut store,(a,b,c,0,u,u,0)).unwrap(),0);
            assert_eq!(locked.call(&mut store,(a,b,c,0,u,count,0)).unwrap(),0);
            if count>3 {
                assert_eq!(locked.call(&mut store,(a,b,c,0,u,ring[(i+2)%ring.len()],0)).unwrap(),0);
            }
        }
    }}}
    for code in 0..6561_u32 {
        let [a,b,c,d]=[code/729,(code/81)%9,(code/9)%9,code%9];
        let count=(1<<a)+(1<<b)+(1<<c)+(1<<d);
        for u in [0,1,count/2,count-1] {
            let v=(u+1)%count;
            assert_eq!(locked.call(&mut store,(a,b,c,d,u,v,1)).unwrap(),1);
            assert_eq!(locked.call(&mut store,(a,b,c,d,v,u,1)).unwrap(),1);
            assert_eq!(locked.call(&mut store,(a,b,c,d,u,(u+2)%count,1)).unwrap(),0);
            assert_eq!(locked.call(&mut store,(a,b,c,d,u,u,1)).unwrap(),0);
            assert_eq!(locked.call(&mut store,(a,b,c,d,u,count,1)).unwrap(),0);
        }
    }
    assert_eq!(locked.call(&mut store,(9,0,0,0,0,1,1)).unwrap(),0);
}

#[test]
fn quad_sampling_wasm_candidates_match_counter_reference_and_boundary_exclusion() {
    let (mut store,instance)=instantiate();
    let candidate=function::<(u32,u32,u32,u32,u32,u32,u32),u32>(&mut store,&instance,"quad_candidate_lane");
    let density=function::<(u32,u32,u32,u32,u32,u32,u32),u32>(&mut store,&instance,"quad_density_lane");
    let boundary=function::<(u32,u32,u32,u32,u32,u32),u32>(&mut store,&instance,"quad_boundary_lane");
    let mix=|mut x:u32| { x^=x>>16; x=x.wrapping_mul(0x7feb352d); x^=x>>15; x=x.wrapping_mul(0x846ca68b); x^(x>>16) };
    let mut checked=0;
    for [a,b,c,d] in [[0,0,0,0],[2,1,4,0],[8,8,8,8]] {
        let code=((a*9+b)*9+c)*9+d;
        let side=2_u32<<a.max(b).max(c).max(d);
        let count=side*side*2;
        let width=16384/side;
        assert_eq!(density.call(&mut store,(a,b,c,d,0,0,2)).unwrap(),count);
        for slot in 0..count {
            let hash=mix(1337 ^ code.wrapping_mul(0x9e3779b9) ^ slot.wrapping_mul(0x85ebca6b));
            let expected=[(slot/2%side)*width+hash%width,
                (slot/2/side)*width+mix(hash^0xa511e9b3)%width];
            let p:[u32;2]=std::array::from_fn(|lane|candidate.call(&mut store,(a,b,c,d,1337,slot,lane as u32)).unwrap());
            assert_eq!(p,expected);
            assert_eq!(candidate.call(&mut store,(a,b,c,d,1337,slot,3)).unwrap(),hash);
            assert_eq!(candidate.call(&mut store,(a,b,c,d,1337,slot,4)).unwrap(),u32::from(p[0]>0 && p[1]>0));
            assert!(p[0]<16384 && p[1]<16384);
            checked+=1;
            // Check the potentially expensive full boundary relation only at
            // a spread of slots; production will need an indexed provider.
            if slot%(count/16).max(1)!=0 {continue;}
            let radius=candidate.call(&mut store,(a,b,c,d,1337,slot,2)).unwrap();
            let boundary_count=(1<<a)+(1<<b)+(1<<c)+(1<<d);
            let mut expected_conflict=false;
            for ordinal in 0..boundary_count {
                let q:[u32;2]=std::array::from_fn(|lane|boundary.call(&mut store,(a,b,c,d,ordinal,lane as u32)).unwrap());
                let qr=density.call(&mut store,(a,b,c,d,q[0],q[1],1)).unwrap();
                let dx=i64::from(q[0])-i64::from(p[0]);
                let dy=i64::from(q[1])-i64::from(p[1]);
                expected_conflict|=dx*dx+dy*dy<i64::from(radius.max(qr));
            }
            expected_conflict&=p[0]>0 && p[1]>0;
            assert_eq!(candidate.call(&mut store,(a,b,c,d,1337,slot,5)).unwrap(),u32::from(expected_conflict));
        }
        assert_eq!(candidate.call(&mut store,(a,b,c,d,1337,count,4)).unwrap(),0);
    }
    assert_eq!(candidate.call(&mut store,(9,0,0,0,1337,0,4)).unwrap(),0);
    eprintln!("{checked} deterministic square candidate slots through LoD 8; boundary exclusion checked independently");
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
fn tensor_projective_chart_wasm_matches_independent_fractional_map() {
    let (mut store,instance)=instantiate();
    let map=function::<(f32,f32,f32,f32,f32,i32,i32),f32>(&mut store,&instance,"tensor_chart_lane");
    for a in [0.02_f32,0.4,1.0,8.0] { for b in [0.02_f32,0.4,1.0,8.0] { for c in [0.02_f32,0.4,1.0,8.0] {
        for step in 0..=64 {
            let u=step as f32/64.0;
            let v=1.0-u;
            for inverse in [false,true] {for lane in 0..2 {
                let ratio=f64::from(if lane==0 {b}else{c})/f64::from(a);
                let ratio=if inverse {1.0/ratio}else{ratio};
                let t=f64::from(if lane==0 {u}else{v});
                let expected=ratio*t/((1.0-t)+ratio*t);
                let actual=map.call(&mut store,(a,b,c,u,v,i32::from(inverse),lane)).unwrap();
                assert!((f64::from(actual)-expected).abs()<2.0e-7);
                if t==0.0 || t==1.0 {assert_eq!(f64::from(actual),t,"fixed endpoints");}
            }}
            let x=map.call(&mut store,(a,b,c,u,v,0,0)).unwrap();
            let y=map.call(&mut store,(a,b,c,u,v,0,1)).unwrap();
            let p=map.call(&mut store,(a,b,c,x,y,1,0)).unwrap();
            let q=map.call(&mut store,(a,b,c,x,y,1,1)).unwrap();
            // f32 chart compression loses information; exact algebra is not
            // bit-exact numerical inversion at extreme ratios.
            assert!((p-u).abs()<4.0e-5 && (q-v).abs()<4.0e-5);
        }
    }}}
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
fn finite_triangle_wasm_measures_actual_vertices_and_rejects_invalid_samples() {
    let (mut store, instance) = instantiate();
    type Vertices = (f32, f32, f32, f32, f32, f32, f32, f32, f32);
    let measure = function::<Vertices, (f32, f32, i32)>(
        &mut store, &instance, "finite_triangle_measure",
    );
    // Translation, rotation (axis permutation), winding and uniform scale.
    for scale in [0.001_f32, 1.0, 1000.0] {
        for reversed in [false, true] {
            let a = [2.0 * scale, -3.0 * scale, 5.0 * scale];
            let b = [a[0], a[1] + scale, a[2]];
            let c = [a[0], a[1], a[2] + 2.0 * scale];
            let (b, c) = if reversed { (c, b) } else { (b, c) };
            let (area, quality, defined) = measure.call(&mut store,
                (a[0],a[1],a[2],b[0],b[1],b[2],c[0],c[1],c[2])).unwrap();
            assert_eq!(defined, 1);
            assert!((f64::from(area) / f64::from(scale).powi(2) - 1.0).abs() < 2e-6);
            assert_close(quality, (2.0 * 3.0_f64.sqrt() / 5.0) as f32, 2e-6,
                "finite triangle shape");
        }
    }
    for input in [
        (0.,0.,0.,1.,0.,0.,2.,0.,0.),
        (f32::NAN,0.,0.,1.,0.,0.,0.,1.,0.),
        (0.,0.,0.,f32::INFINITY,0.,0.,0.,1.,0.),
    ] {
        assert_eq!(measure.call(&mut store, input).unwrap(), (0.,0.,0));
    }
}

#[test]
fn midpoint_square_has_exact_coverage_and_protected_master_intervals() {
    let (mut store,instance)=instantiate();
    let vertex=function::<i32,(i32,i32)>(&mut store,&instance,"midpoint_square_vertex");
    let child=function::<i32,(i32,i32,i32)>(&mut store,&instance,"midpoint_square_child");
    let sample=function::<(i32,i32,i32),i32>(&mut store,&instance,"midpoint_square_sample");
    let outer=function::<(i32,i32,i32,i32),(i32,i32,i32)>(&mut store,&instance,"midpoint_outer_sample");
    let primary=function::<i32,(i32,i32,i32,i32)>(&mut store,&instance,"midpoint_square_primary");
    let remap_a=function::<(i32,f32,f32,f32),f32>(&mut store,&instance,"qb_remap_a");
    let remap_b=function::<(i32,f32,f32,f32),f32>(&mut store,&instance,"qb_remap_b");
    let remap_c=function::<(i32,f32,f32,f32),f32>(&mut store,&instance,"qb_remap_c");
    let mut area=0;
    let mut edges=std::collections::BTreeMap::new();
    for i in 0..8 {
        let (a,b,c)=child.call(&mut store,i).unwrap();
        let pa=vertex.call(&mut store,a).unwrap();
        let pb=vertex.call(&mut store,b).unwrap();
        let pc=vertex.call(&mut store,c).unwrap();
        let signed=(pb.0-pa.0)*(pc.1-pa.1)-(pb.1-pa.1)*(pc.0-pa.0);
        assert_eq!(signed,1,"all eight triangles are CCW and have equal area");
        area+=signed;
        for (a,b) in [(a,b),(b,c),(c,a)] {
            let entry=edges.entry((a.min(b),a.max(b))).or_insert((0,0));
            entry.0+=1;
            entry.1+=if a<b {1} else {-1};
        }
    }
    assert_eq!(area,8); // Twice the area of the exact [0,2]^2 domain.
    assert_eq!(edges.len(),16);
    assert_eq!(edges.values().filter(|e|e.0==1).count(),8);
    for (_, (uses,orientation)) in &edges {
        assert!(*uses==1 || (*uses==2 && *orientation==0));
    }
    assert_eq!(child.call(&mut store,8).unwrap(),(99,99,99));
    for level in [0,9] {assert_eq!(sample.call(&mut store,(level,0,0)).unwrap(),999);}
    for level in 1..=8 {
        let (a,b,c,permutation)=primary.call(&mut store,level).unwrap();
        assert_eq!((a,b,c),(level-1,level-1,level));
        // The canonical edge counts must return to the authored opposite-vertex
        // order: diamond opposite A, half-master edges opposite B and C.
        let args=(permutation,a as f32,b as f32,c as f32);
        assert_eq!(remap_a.call(&mut store,args).unwrap(),level as f32);
        assert_eq!(remap_b.call(&mut store,args).unwrap(),(level-1) as f32);
        assert_eq!(remap_c.call(&mut store,args).unwrap(),(level-1) as f32);
        let half=1<<(level-1);
        for i in 0..=half {
            assert_eq!(sample.call(&mut store,(level,0,i)).unwrap(),i);
            assert_eq!(sample.call(&mut store,(level,1,i)).unwrap(),half+i);
        }
        assert_eq!(sample.call(&mut store,(level,0,half+1)).unwrap(),999);
        for side in 0..4 {
            let a=side;
            let b=(side+1)%4;
            let middle=side+4;
            let full=2*half;
            for (from,to,start,end) in [(a,middle,0,half),(middle,b,half,full)] {
                for i in 0..=half {
                    let expected=if a<b {start+i} else {full-start-i};
                    let forward=outer.call(&mut store,(level,from,to,i)).unwrap();
                    let reverse=outer.call(&mut store,(level,to,from,half-i)).unwrap();
                    assert_eq!(forward,(a.min(b),a.max(b),expected));
                    assert_eq!(forward,reverse,"integer reversal preserves the master sample");
                }
                assert_eq!(end-start,half);
                assert_eq!(outer.call(&mut store,(level,from,to,half+1)).unwrap(),(99,99,999));
            }
        }
        for (&(a,b),&(uses,_)) in &edges {
            if uses==2 {assert_eq!(outer.call(&mut store,(level,a,b,0)).unwrap(),(99,99,999));}
        }
    }
}

#[test]
fn configured_patch_variations_move_the_actual_surface() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../ingots/validation/composition_oracle");
    let wasm = compile_ingot(&path);
    let engine = wasmtime::Engine::default();
    let module = wasmtime::Module::new(&engine, &wasm).unwrap();
    let mut store = Store::new(&engine, ());
    let instance = Instance::new(&mut store, &module, &[]).unwrap();
    let sample = function::<(i32,f32,f32,i32,f32,f32,f32,f32),f32>(
        &mut store, &instance, "configured_surface_at");
    for kind in [0, 1] {
        for edge in [0.0_f32, 1.0] {
            for bulge in [0.6_f32, 2.4] {
                for (dx,dz) in [(0.6_f32,0.0_f32),(0.0,0.9),(-0.5,0.7)] {
                    // The changed corner is an exact positional witness. Also
                    // require an interior witness; a changed control alone is
                    // not evidence that the evaluator used it.
                    let corner = if kind == 0 { (0.0,1.0) } else { (1.0,1.0) };
                    for (u,v) in [corner, (0.23,0.31)] {
                        let mut displacement = 0.0_f64;
                        for lane in 0..3 {
                            let base = sample.call(&mut store,
                                (kind,u,v,lane,bulge,0.0,0.0,edge)).unwrap();
                            let moved = sample.call(&mut store,
                                (kind,u,v,lane,bulge,dx,dz,edge)).unwrap();
                            assert!(base.is_finite() && moved.is_finite()
                                && base > -900.0 && moved > -900.0,
                                "rejected witness: kind={kind} edge={edge} bulge={bulge} dx={dx} dz={dz}");
                            let delta = f64::from(moved)-f64::from(base);
                            displacement += delta*delta;
                            if (u,v) == corner {
                                let expected = if lane == 0 { dx } else if lane == 2 { dz } else { 0.0 };
                                assert_close(delta as f32, expected, 2e-5, "active corner displacement");
                            }
                        }
                        assert!(displacement > 1e-8,
                            "ineffective variation: kind={kind} edge={edge} bulge={bulge} dx={dx} dz={dz} at ({u},{v})");
                    }
                }
            }
        }
    }
}

#[test]
fn curved_atlas_mesh_quality_baseline() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../ingots/validation/composition_oracle");
    let wasm = compile_ingot(&path);
    let engine = wasmtime::Engine::default();
    let module = wasmtime::Module::new(&engine, &wasm).unwrap();
    let mut store = Store::new(&engine, ());
    let instance = Instance::new(&mut store, &module, &[]).unwrap();
    let audit = function::<(i32,f32,f32,i32,f32,f32,f32),
        (i32,i32,i32,i32,i32,f32,f32,f32,f32,i32)>(
        &mut store, &instance, "curved_mesh_quality");
    for (kind,split) in [(0,0.0_f32),(1,0.0),(1,1.0),(1,2.0)] {
        for (bulge,dx,dz) in [(0.6_f32,0.0_f32,0.0_f32),(2.4,-0.5,0.7)] {
            for measured in [0,1] {
                let start = std::time::Instant::now();
                let (status,count,invalid,folds,poor,min,mean,cv,area,evals) = audit.call(
                    &mut store,(kind,split,3.0,measured,bulge,dx,dz)).unwrap();
                eprintln!("CURVED_MESH kind={kind} split={split} measured={measured} bulge={bulge} dx={dx} dz={dz} status={status} count={count} invalid={invalid} uv_folds={folds} poor={poor} min={min:.6} mean={mean:.6} area_cv={cv:.6} area={area:.6} vertex_evals={evals} elapsed_ms={:.1}", start.elapsed().as_secs_f64()*1000.0);
                assert_eq!(status, 0, "fixture/build failure, not a quality result");
                assert!(count > 0 && invalid >= 0 && invalid <= count);
                assert!(folds >= 0 && folds <= count && poor >= 0 && poor <= count-invalid);
                assert_eq!(evals, 3*count);
                assert!(min.is_finite() && mean.is_finite() && cv.is_finite() && area.is_finite());
                assert!(min >= 0.0 && mean >= min && mean <= 1.000001 && cv >= 0.0 && area > 0.0);
                // Pathology quality is observed, not given a permissive "pass"
                // threshold. These assertions only check measurement integrity.
            }
        }
    }
}

#[test]
fn shape_histogram_rank_wasm() {
    let (mut store,instance)=instantiate();
    let rank=function::<(f32,i32),f32>(&mut store,&instance,"histogram_rank");
    assert_eq!(rank.call(&mut store,(0.52,1)).unwrap(),0.5);
    // Invalid ranks must trap, not produce a plausible but fabricated percentile.
    assert!(rank.call(&mut store,(0.52,0)).is_err());
    assert!(rank.call(&mut store,(0.52,2)).is_err());
    assert!(rank.call(&mut store,(f32::NAN,1)).is_err());
}

#[test]
fn curved_mesh_distribution_and_approximation_probe() {
    let path=Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../ingots/validation/composition_oracle");
    let wasm=compile_ingot(&path);
    let engine=wasmtime::Engine::default();
    let module=wasmtime::Module::new(&engine,&wasm).unwrap();
    let mut store=Store::new(&engine,());
    let instance=Instance::new(&mut store,&module,&[]).unwrap();
    let laws=function::<(),i32>(&mut store,&instance,"shape_histogram_laws");
    assert_eq!(laws.call(&mut store,()).unwrap(),1);
    let error_laws=function::<(),i32>(&mut store,&instance,"centroid_error_laws");
    assert_eq!(error_laws.call(&mut store,()).unwrap(),1);
    let details=function::<(i32,i32,f32,f32,f32),
        (i32,i32,i32,i32,f32,f32,i32,i32,f32,f32,f32,f32,i32)>(
        &mut store,&instance,"uniform_quad_mesh_details");
    let summary=function::<(i32,i32,f32,f32,f32),
        (i32,i32,i32,i32,i32,f32,f32,f32,f32,i32)>(
        &mut store,&instance,"uniform_quad_layout_quality");
    for (bulge,dx,dz) in [(0.6,0.0,0.0),(2.4,-0.5,0.7),(1.2,0.7,-0.5)] {
        for level in [3,4] {
            for layout in 0..4 {
                let args=(layout,level,bulge,dx,dz);
                let r=details.call(&mut store,args).unwrap();
                let s=summary.call(&mut store,args).unwrap();
                assert_eq!((r.0,r.1,r.2,r.3),(s.0,s.1,s.1-s.2,s.2));
                assert_eq!(r.0,0);
                assert_eq!(r.6+r.7,r.1);
                assert_eq!(r.7,0);
                assert_eq!(r.12,4*r.1);
                assert!(r.4>=0.0 && r.5>=r.4 && r.5<=0.95);
                assert!(r.4+0.050001>=s.5);
                assert!(r.8.is_finite() && r.9.is_finite() && r.9>=0.0 && r.9<=r.8+1e-6);
                assert!((0.0..=1.0).contains(&r.10) && (0.0..=1.0).contains(&r.11));
                eprintln!("MESH_DETAILS layout={layout} level={level} bulge={bulge} dx={dx} dz={dz} count={} p05_lower={} p10_lower={} valid_error={} invalid_error={} max_error={} rms_error={} worst_uv=({}, {}) evals={}",r.1,r.4,r.5,r.6,r.7,r.8,r.9,r.10,r.11,r.12);
            }
        }
    }
}

#[test]
fn bounded_composite_draw_ranges() {
    let path=Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../ingots/validation/composition_oracle");
    let wasm=compile_ingot(&path);
    let engine=wasmtime::Engine::default();
    let module=wasmtime::Module::new(&engine,&wasm).unwrap();
    let mut store=Store::new(&engine,());
    let instance=Instance::new(&mut store,&module,&[]).unwrap();
    for name in ["bounded_draw_range_laws","coherent_pending_selection"] {
        let test=function::<(),i32>(&mut store,&instance,name);
        assert_eq!(test.call(&mut store,()).unwrap(),1,"{name}");
    }
}

#[test]
fn frozen_topology_count_allocation_probe() {
    let path=Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../ingots/validation/composition_oracle");
    let wasm=compile_ingot(&path);
    let engine=wasmtime::Engine::default();
    let module=wasmtime::Module::new(&engine,&wasm).unwrap();
    let mut store=Store::new(&engine,());
    let instance=Instance::new(&mut store,&module,&[]).unwrap();
    let plan=function::<(i32,f32,f32,f32,i32,f32),(i32,i32,i32,i32,i32,i32,i32,i32)>(
        &mut store,&instance,"quad_count_plan");
    let mesh=function::<(i32,f32,f32,f32,f32),
        (i32,i32,i32,i32,i32,f32,f32,f32,f32,i32)>(
        &mut store,&instance,"quad_count_mesh");
    let reference=function::<(i32,i32,f32,f32,f32),
        (i32,i32,i32,i32,i32,f32,f32,f32,f32,i32)>(
        &mut store,&instance,"uniform_quad_layout_quality");
    for segments in [0,3,65] {
        assert_eq!(plan.call(&mut store,(0,0.6,0.0,0.0,segments,1.0)).unwrap().0,1);
    }
    for scale in [0.0,f32::NAN,4.1] {
        assert_eq!(plan.call(&mut store,(0,0.6,0.0,0.0,16,scale)).unwrap().0,1);
    }
    for (bulge,dx,dz) in [(0.6,0.0,0.0),(2.4,-0.5,0.7),(1.2,0.7,-0.5)] {
        for layout in 0..3 {
            let baseline=reference.call(&mut store,(layout,3,bulge,dx,dz)).unwrap();
            for scale in [0.5,1.0,1.5] {
                let start=std::time::Instant::now();
                let a=plan.call(&mut store,(layout,bulge,dx,dz,16,scale)).unwrap();
                let b=plan.call(&mut store,(layout,bulge,dx,dz,32,scale)).unwrap();
                let plan_ms=start.elapsed().as_secs_f64()*1000.0;
                assert_eq!((a.0,a.2,a.7),(0,0,1));
                assert_eq!((b.0,b.2,b.7),(0,0,1));
                assert_eq!(a.1,if layout==2 {8*17} else {5*17});
                assert_eq!(b.1,if layout==2 {8*33} else {5*33});
                // Doubling quadrature is observed, not assumed to preserve rounding.
                let actual=mesh.call(&mut store,(layout,bulge,dx,dz,scale)).unwrap();
                assert_eq!((actual.0,actual.2,actual.3),(0,0,0));
                assert_eq!(actual.9,3*actual.1);
                eprintln!("COUNT_PROBE layout={layout} bulge={bulge} dx={dx} dz={dz} scale={scale} plan16={a:?} plan32={b:?} reference={baseline:?} actual={actual:?} two_plan_ms={plan_ms:.4}");
            }
        }
    }
}

#[test]
fn bounded_quad_candidate_ranking_probe() {
    let path=Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../ingots/validation/composition_oracle");
    let wasm=compile_ingot(&path);
    let engine=wasmtime::Engine::default();
    let module=wasmtime::Module::new(&engine,&wasm).unwrap();
    let mut store=Store::new(&engine,());
    let instance=Instance::new(&mut store,&module,&[]).unwrap();
    let score=function::<(i32,f32,f32,f32,f32),(i32,f32,f32,i32)>(
        &mut store,&instance,"quad_candidate_score");
    let mesh=function::<(i32,i32,f32,f32,f32),
        (i32,i32,i32,i32,i32,f32,f32,f32,f32,i32)>(
        &mut store,&instance,"quad_candidate_mesh");
    for (bulge,dx,dz) in [(0.6,0.0,0.0),(2.4,-0.5,0.7),(1.2,0.7,-0.5)] {
        for id in 0..11 {
            let start=std::time::Instant::now();
            let a=score.call(&mut store,(id,bulge,dx,dz,1.0/1024.0)).unwrap();
            let b=score.call(&mut store,(id,bulge,dx,dz,1.0/2048.0)).unwrap();
            let score_ms=start.elapsed().as_secs_f64()*1000.0;
            assert_eq!(a.0,0);
            assert_eq!(b.0,0);
            assert_eq!(a.3,if id<2 {32} else {64});
            assert!(a.1.is_finite() && a.2.is_finite());
            let actual=mesh.call(&mut store,(id,3,bulge,dx,dz)).unwrap();
            assert_eq!((actual.0,actual.2,actual.3),(0,0,0));
            eprintln!("CANDIDATE id={id} bulge={bulge} dx={dx} dz={dz} predicted={a:?} half_step={b:?} actual={actual:?} two_score_ms={score_ms:.3}");
            // This test observes ranking failures; it does not assert that
            // a coarse differential score predicts the actual fine minimum.
        }
    }
}

#[test]
fn uniform_quad_layout_quality_cost_probe() {
    let path=Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../ingots/validation/composition_oracle");
    let wasm=compile_ingot(&path);
    let engine=wasmtime::Engine::default();
    let module=wasmtime::Module::new(&engine,&wasm).unwrap();
    let mut store=Store::new(&engine,());
    let instance=Instance::new(&mut store,&module,&[]).unwrap();
    let audit=function::<(i32,i32,f32,f32,f32),
        (i32,i32,i32,i32,i32,f32,f32,f32,f32,i32)>(
        &mut store,&instance,"uniform_quad_layout_quality");
    for (bulge,dx,dz) in [(0.6,0.0,0.0),(2.4,-0.5,0.7),(1.2,0.7,-0.5)] {
        for level in [3,4] {
            for layout in 0..4 {
                let start=std::time::Instant::now();
                let r=audit.call(&mut store,(layout,level,bulge,dx,dz)).unwrap();
                eprintln!("UNIFORM_LAYOUT layout={layout} level={level} bulge={bulge} dx={dx} dz={dz} report={r:?} elapsed_ms={:.2}",start.elapsed().as_secs_f64()*1000.0);
                assert_eq!(r.0,0);
                assert!(r.1>0);
                assert_eq!(r.2,0,"invalid evaluated vertex");
                assert_eq!(r.3,0,"UV fold");
                assert_eq!(r.9,3*r.1);
                assert!(r.5.is_finite() && r.6.is_finite() && r.7.is_finite() && r.8>0.0);
                assert!(r.5>=0.0 && r.6>=r.5 && r.6<=1.000001);
                // A quality-cost observation, not an assertion of improvement.
            }
        }
    }
}

#[test]
fn curved_interior_comparison_preserves_boundaries_and_counts() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../ingots/validation/composition_oracle");
    let wasm = compile_ingot(&path);
    let engine = wasmtime::Engine::default();
    let module = wasmtime::Module::new(&engine, &wasm).unwrap();
    let mut store = Store::new(&engine, ());
    let instance = Instance::new(&mut store, &module, &[]).unwrap();
    let boundaries = function::<(i32,f32,f32),i32>(&mut store,&instance,"interior_comparison_boundaries");
    let audit = function::<(i32,f32,f32,i32,f32,f32,f32),
        (i32,i32,i32,i32,i32,f32,f32,f32,f32,i32)>(
        &mut store,&instance,"curved_mesh_quality");
    for (kind,split) in [(0,0.0_f32),(1,0.0),(1,1.0),(1,2.0)] {
        for bulge in [0.6_f32,2.4] {
            assert_eq!(boundaries.call(&mut store,(kind,split,bulge)).unwrap(),1);
            let mut count=None;
            for mode in [1,2] {
                let start=std::time::Instant::now();
                let result=audit.call(&mut store,(kind,split,3.0,mode,bulge,-0.5,0.7)).unwrap();
                assert_eq!(result.0,0);
                if let Some(n)=count {assert_eq!(result.1,n);} else {count=Some(result.1);}
                eprintln!("INTERIOR_AB kind={kind} split={split} bulge={bulge} mode={mode} report={result:?} elapsed_ms={:.2}",start.elapsed().as_secs_f64()*1000.0);
            }
        }
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

#[test]
fn atlas_candidate_windows_cover_conflicts_and_bound_uniform_work_through_lod8() {
    use std::collections::HashSet;
    let (mut store, instance) = instantiate();
    let query = function::<(i32, i32, i32, i32, i32, i32), i32>(
        &mut store, &instance, "candidate_window_lane",
    );
    let s = 16384i32;
    for maximum in 0..=8 {
        let side = 2 << maximum;
        let width = s / side;
        for triangular in 0..=1 {
            for minimum in [0, maximum / 2, maximum] {
                for (row, col) in [(0, 0), (s / 3, s / 5), (s - 1, 0)] {
                    let count = query.call(&mut store, (maximum, minimum, row, col, triangular, -1)).unwrap();
                    if minimum == maximum {
                        assert!(count <= 50, "uniform query grew at LoD {maximum}: {count}");
                    }
                    let mut actual = HashSet::new();
                    for i in 0..count {
                        let slot = query.call(&mut store, (maximum, minimum, row, col, triangular, i)).unwrap();
                        if slot >= 0 { assert!(actual.insert(slot), "duplicate candidate slot"); }
                    }
                    assert_eq!(query.call(&mut store, (maximum, minimum, row, col, triangular, count)).unwrap(), -1);
                    let radius = s >> minimum;
                    // Independent closed cell/AABB intersection: stronger
                    // than disk conflict, so no metric-specific false negative
                    // can hide inside an excluded cell. Probe every cell.
                    for r in 0..side {
                        for c in 0..side {
                            if triangular != 0 && r + c >= side { continue; }
                            let first = if triangular != 0 { r * (2 * side - r + 1) / 2 } else { r * side };
                            let slot = (first + c) * 2;
                            let dx = (r * width - row).max(row - ((r + 1) * width - 1)).max(0);
                            let dy = (c * width - col).max(col - ((c + 1) * width - 1)).max(0);
                            if dx <= radius && dy <= radius {
                                assert!(actual.contains(&slot) && actual.contains(&(slot + 1)),
                                    "missed cell ({r},{c}) for max={maximum} min={minimum} tri={triangular}");
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn atlas_nearest_boundary_query_matches_exhaustive_relation_for_every_key() {
    let (mut store, instance) = instantiate();
    let query = function::<(i32, i32, i32, i32, i32, i32, i32, i32), i32>(
        &mut store, &instance, "boundary_query_comparison",
    );
    let mut checked = 0;
    for domain in 0..=1 {
        for a in 0..=8 {
            for b in 0..=8 {
                for c in 0..=8 {
                    if domain == 0 && !(a <= b && b <= c) { continue; }
                    for d in 0..=if domain == 0 { 0 } else { 8 } {
                        // Include corners, exact sample positions, half-step
                        // ties, and deterministic interior points. Candidate
                        // radius is independent of boundary radius here.
                        let mut points = vec![(0,0), (16384,0), (0,16384),
                            (8192,8192), (32,32), (64,64), (8192,1),
                            (16383,1), (1,8192), (4001,7351)];
                        if domain == 1 { points.extend([(16384,16384), (16352,8192), (8192,16352)]); }
                        let code: u32 = (((a * 9 + b) * 9 + c) * 9 + d) as u32;
                        let mut state = code.wrapping_add(42);
                        for _ in 0..8 {
                            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                            let mut x = (state % 16385) as i32;
                            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                            let mut y = (state % 16385) as i32;
                            if domain == 0 && x+y > 16384 { x=16384-x; y=16384-y; }
                            points.push((x,y));
                        }
                        for (x,y) in points {
                            for radius in [0, 4096, 4097, 65536, 268435456] {
                                let result = query.call(&mut store,(a,b,c,d,x,y,radius,domain)).unwrap();
                                assert!(result == 0 || result == 3,
                                    "boundary mismatch domain={domain} key={a},{b},{c},{d} point={x},{y} r2={radius} bits={result}");
                                checked += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    eprintln!("matched {checked} exhaustive/bounded boundary queries across every LoD 0–8 key");
}

#[test]
fn insertion_child_face_search_is_bounded_disjoint_and_fails_closed() {
    let (mut store, instance) = instantiate();
    let query = function::<(i32,i32,i32,i32,i32,i32),i32>(&mut store,&instance,"insertion_child_face_lane");
    for a in 0..12 {
        for an in 0..5 {
            for b in 0..12 {
                for bn in 0..5 {
                    let valid=(1..=3).contains(&an) && a+an<=12 &&
                        (bn==0 || (bn<=3 && b+bn<=12 && (a+an<=b || b+bn<=a)));
                    let expected:Vec<i32>=if valid {(a..a+an).chain(b..b+bn).collect()} else {vec![]};
                    let count=query.call(&mut store,(a,an,b,bn,12,-1)).unwrap();
                    assert_eq!(count,expected.len() as i32);
                    assert!(count<=6);
                    for i in 0..=count {
                        assert_eq!(query.call(&mut store,(a,an,b,bn,12,i)).unwrap(),
                            expected.get(i as usize).copied().unwrap_or(-1));
                    }
                }
            }
        }
    }
    for (a,an,b,bn,n) in [(-1,1,0,0,12),(0,1,-1,1,12),(0,4,4,1,12),(0,1,0,1,12)] {
        assert_eq!(query.call(&mut store,(a,an,b,bn,n,-1)).unwrap(),0);
    }
}
