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
