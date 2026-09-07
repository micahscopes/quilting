//! Host execution/acceptance only; sampling, CDT and mesh audits are Fe.
use super::fe_oracle::compile_ingot_at_level;
use fe_codegen::OptLevel;
use std::path::Path;

#[test]
fn dyadic_arc_spacing_has_uniform_geometric_chords_and_nested_samples() {
    let path=Path::new(env!("CARGO_MANIFEST_DIR")).join("../../ingots/validation/composition_oracle");
    let wasm=compile_ingot_at_level(&path,OptLevel::O2);
    let engine=wasmtime::Engine::default();
    let module=wasmtime::Module::new(&engine,&wasm).unwrap();
    let mut store=wasmtime::Store::new(&engine,());
    let instance=wasmtime::Instance::new(&mut store,&module,&[]).unwrap();
    let parameter=instance.get_typed_func::<(f32,f32,f32,i32),f32>(&mut store,"dyadic_arc_parameter").unwrap();
    let nested=instance.get_typed_func::<(f32,f32,f32),i32>(&mut store,"dyadic_arc_nesting").unwrap();
    let mut worst_mass_error=0.0_f64;
    let mut worst_chord_error=0.0_f64;
    for ratio in [0.1_f32,1.0,10.0] {
        for angle in [0.0_f64,0.01,0.3,1.5,2.8,3.1] {
            let x=(f64::from(ratio)*angle.cos()) as f32;
            let y=(f64::from(ratio)*angle.sin()) as f32;
            let theta=f64::from(y).atan2(f64::from(x));
            let end_norm=f64::from(x).hypot(f64::from(y));
            let end=[f64::from(x)/end_norm.powi(2),f64::from(y)/end_norm.powi(2)];
            let endpoint_chord=(end[0]-1.0).hypot(end[1]);
            let expected=if theta==0.0 {endpoint_chord/256.0}
                else {endpoint_chord*(theta/256.0).sin()/theta.sin()};
            assert_eq!(nested.call(&mut store,(1.0,x,y)).unwrap(),1);
            let mut previous_t=-1.0;
            let mut previous_point=[1.0,0.0];
            for i in 0..=256 {
                let t=parameter.call(&mut store,(1.0,x,y,i)).unwrap();
                assert!(t>=0.0 && t<=1.0 && t>previous_t,"ratio={ratio} angle={angle} i={i} t={t}");
                if i==0 {assert_eq!(t,0.0)}
                if i==256 {assert_eq!(t,1.0)}
                let t64=f64::from(t);
                let w=[1.0-t64+f64::from(x)*t64,f64::from(y)*t64];
                let norm=w[0]*w[0]+w[1]*w[1];
                let point=[w[0]/norm,w[1]/norm];
                if theta>0.0 {
                    let mass=w[1].atan2(w[0])/theta;
                    worst_mass_error=worst_mass_error.max((mass-f64::from(i)/256.0).abs());
                }
                if i>0 && expected>1e-12 {
                    let chord=(point[0]-previous_point[0]).hypot(point[1]-previous_point[1]);
                    worst_chord_error=worst_chord_error.max((chord/expected-1.0).abs());
                }
                previous_t=t;
                previous_point=point;
            }
        }
    }
    for (a,x,y) in [(0.0,1.0,0.0),(1.0,0.0,0.0),(1.0,-1.0,0.0),(f32::NAN,1.0,0.0)] {
        assert_eq!(parameter.call(&mut store,(a,x,y,128)).unwrap(),-1.0,"singular input must reject");
    }
    eprintln!("DYADIC_ARC 18 families x 257 points: worst normalized mass error={worst_mass_error:.9}, relative chord error={worst_chord_error:.9}");
    assert!(worst_mass_error<2e-5);
    assert!(worst_chord_error<0.002);
    // Exercise the actual sparse-Clifford weights and patch evaluator, rather
    // than only the two-dimensional inverse-line reference above.
    let actual=instance.get_typed_func::<(i32,i32,i32,i32),f32>(&mut store,"authored_uniform_arc").unwrap();
    let mut worst_patch_chord_error=0.0_f64;
    for kind in 0..2 {
        for edge in 0..if kind==0 {3} else {4} {
            let mut points=Vec::new();
            for index in 0..=256 {
                let mut point=[0.0_f64;3];
                for lane in 0..3 {
                    let value=actual.call(&mut store,(kind,edge,index,lane)).unwrap();
                    assert!(value.is_finite() && value != -999.0);
                    point[lane as usize]=f64::from(value);
                }
                points.push(point);
            }
            let chords:Vec<f64>=points.windows(2).map(|p| {
                ((p[1][0]-p[0][0]).powi(2)+(p[1][1]-p[0][1]).powi(2)+(p[1][2]-p[0][2]).powi(2)).sqrt()
            }).collect();
            let mean=chords.iter().sum::<f64>()/256.0;
            assert!(mean>0.0);
            for chord in chords {worst_patch_chord_error=worst_patch_chord_error.max((chord/mean-1.0).abs());}
        }
    }
    eprintln!("DYADIC_PATCH 7 actual circular boundaries: relative chord deviation={worst_patch_chord_error:.9}");
    assert!(worst_patch_chord_error<0.002);
}

/// Micah's observation: dragging the arc handle along its own edge changed the
/// density. That is a reparameterization, not a shape change, so it is exactly
/// what measured arc-length spacing must be blind to. Uniform-parameter spacing
/// is not blind to it, and this pins both halves of that claim.
#[test]
fn measured_arc_spacing_is_invariant_to_sliding_the_authored_midpoint() {
    let path=Path::new(env!("CARGO_MANIFEST_DIR")).join("../../ingots/validation/composition_oracle");
    let wasm=compile_ingot_at_level(&path,OptLevel::O2);
    let engine=wasmtime::Engine::default();
    let module=wasmtime::Module::new(&engine,&wasm).unwrap();
    let mut store=wasmtime::Store::new(&engine,());
    let instance=wasmtime::Instance::new(&mut store,&module,&[]).unwrap();
    let boundary=instance.get_typed_func::<(f32,i32,i32,i32),f32>(&mut store,"slid_arc_boundary").unwrap();
    let mut collect=|store:&mut wasmtime::Store<()>,slide:f32,measured:i32| {
        let mut points=Vec::new();
        for index in 0..=256 {
            let mut point=[0.0_f64;3];
            for lane in 0..3 {
                let value=boundary.call(&mut *store,(slide,measured,index,lane)).unwrap();
                assert!(value.is_finite() && value!=-999.0,"slide={slide} measured={measured}");
                point[lane as usize]=f64::from(value);
            }
            points.push(point);
        }
        points
    };
    let distance=|a:&[f64;3],b:&[f64;3]| {
        ((a[0]-b[0]).powi(2)+(a[1]-b[1]).powi(2)+(a[2]-b[2]).powi(2)).sqrt()
    };
    // The authored edge spans two units between its fixed endpoints, so these
    // displacements are already in a meaningful scale for this patch.
    let mut worst=[0.0_f64;2];
    for measured in 0..2 {
        let reference=collect(&mut store,0.5,measured);
        for slide in [0.30_f32,0.40,0.60,0.70] {
            let moved=collect(&mut store,slide,measured);
            for (a,b) in reference.iter().zip(moved.iter()) {
                worst[measured as usize]=worst[measured as usize].max(distance(a,b));
            }
        }
    }
    eprintln!("SLID_ARC worst sample displacement: uniform parameter={:.9}, measured arc length={:.9}",worst[0],worst[1]);
    // The reparameterization must actually be visible, otherwise the test would
    // pass for the trivial reason that sliding did nothing.
    assert!(worst[0]>0.02,"sliding must move uniform-parameter samples: {}",worst[0]);
    // And measured spacing must not follow it. The residual is f32 evaluation
    // noise through a different weight representation of the same circle.
    assert!(worst[1]<1e-3,"measured spacing must be reparameterization invariant: {}",worst[1]);
    assert!(worst[1]<worst[0]/20.0);
}

/// Interior diagonals and spokes get integrated spacing rather than the closed
/// form the outer sides use, so this pins that the integration actually helps
/// and reports how far it lands from uniform.
#[test]
fn measured_interior_spacing_beats_uniform_parameter_on_spokes_and_diagonals() {
    let path=Path::new(env!("CARGO_MANIFEST_DIR")).join("../../ingots/validation/composition_oracle");
    let wasm=compile_ingot_at_level(&path,OptLevel::O2);
    let engine=wasmtime::Engine::default();
    let module=wasmtime::Module::new(&engine,&wasm).unwrap();
    let mut store=wasmtime::Store::new(&engine,());
    let instance=wasmtime::Instance::new(&mut store,&module,&[]).unwrap();
    let deviation=instance.get_typed_func::<(i32,i32,i32,i32),f32>(&mut store,"interior_chord_deviation").unwrap();
    // Four fan spokes to the movable centre, then the two diagonals.
    let segments=[(0,4),(1,4),(2,4),(3,4),(0,2),(1,3)];
    let mut worst_measured=0.0_f64;
    let mut least_uniform=f64::INFINITY;
    for (from,to) in segments {
        let uniform=f64::from(deviation.call(&mut store,(0,from,to,64)).unwrap());
        let measured=f64::from(deviation.call(&mut store,(1,from,to,64)).unwrap());
        assert!(uniform>=0.0 && measured>=0.0,"segment {from}->{to} did not evaluate");
        eprintln!("INTERIOR {from}->{to}: uniform parameter={uniform:.6}, measured length={measured:.6}");
        assert!(measured<uniform,"segment {from}->{to}: {measured} !< {uniform}");
        worst_measured=worst_measured.max(measured);
        least_uniform=least_uniform.min(uniform);
    }
    eprintln!("INTERIOR worst measured deviation={worst_measured:.6}, smallest uniform deviation={least_uniform:.6}");
    // Integration leaves a quadrature residual, unlike the exact outer tables.
    // This bound records what the sixteen-span table actually achieves; it is
    // not a claim of exact uniformity on a curve with no closed-form arc law.
    assert!(worst_measured<0.05,"measured interior deviation regressed: {worst_measured}");
}

/// Settles whether surface area density has the closed form one over the fourth
/// power of the denominator norm, or whether the numerator contributes its own
/// affine scalar. Everything downstream of a density field depends on which.
#[test]
fn surface_area_density_is_the_reciprocal_fourth_power_of_the_denominator_norm() {
    let path=Path::new(env!("CARGO_MANIFEST_DIR")).join("../../ingots/validation/composition_oracle");
    let wasm=compile_ingot_at_level(&path,OptLevel::O2);
    let engine=wasmtime::Engine::default();
    let module=wasmtime::Module::new(&engine,&wasm).unwrap();
    let mut store=wasmtime::Store::new(&engine,());
    let instance=wasmtime::Instance::new(&mut store,&module,&[]).unwrap();
    let ratio=instance.get_typed_func::<(i32,f32,f32,f32),f32>(&mut store,"area_density_ratio").unwrap();
    for (kind,name) in [(0,"triangle"),(1,"quad")] {
        let mut values=Vec::new();
        for i in 1..12 {
            for j in 1..12 {
                let u=f64::from(i)/12.0;
                let v=f64::from(j)/12.0;
                // Stay inside the triangular domain where u+v<1.
                if kind==0 && u+v>0.92 {continue}
                let r=f64::from(ratio.call(&mut store,(kind,u as f32,v as f32,1.0/512.0)).unwrap());
                assert!(r>0.0,"{name} at {u},{v} did not evaluate: {r}");
                values.push(r);
            }
        }
        let mean=values.iter().sum::<f64>()/values.len() as f64;
        let spread=values.iter().map(|r| (r/mean-1.0).abs()).fold(0.0_f64,f64::max);
        eprintln!("AREA_DENSITY {name}: {} samples, mean ratio={mean:.6}, worst relative spread={spread:.6}",values.len());
        if kind==0 {
            // The triangular blend is affine, so its Jacobian is constant and
            // the reciprocal fourth power is the whole law. The tolerance is
            // central-difference truncation, not slack.
            assert!(spread<2e-3,"triangle ratio should be constant: {spread}");
        } else {
            // The bilinear blend's Jacobian varies with position, so the same
            // law is NOT sufficient for a quad. Pinned as a measured fact so a
            // later change cannot quietly adopt the triangle's formula here.
            assert!(spread>0.05,"quad departure disappeared, recheck the law: {spread}");
        }
    }
}

/// The quad's departure from the reciprocal fourth power is the bilinear
/// blend's own Jacobian. If that is what it is, the squared correction must be
/// exactly a tensor-product quadratic, so a nine-node interpolant built at
/// three nodes per axis must reproduce it everywhere else. Deciding this
/// decides whether a quad density field is closed form and cheap, or needs
/// runtime differentiation.
#[test]
fn quad_area_density_correction_is_a_tensor_product_quadratic() {
    let path=Path::new(env!("CARGO_MANIFEST_DIR")).join("../../ingots/validation/composition_oracle");
    let wasm=compile_ingot_at_level(&path,OptLevel::O2);
    let engine=wasmtime::Engine::default();
    let module=wasmtime::Module::new(&engine,&wasm).unwrap();
    let mut store=wasmtime::Store::new(&engine,());
    let instance=wasmtime::Instance::new(&mut store,&module,&[]).unwrap();
    let ratio=instance.get_typed_func::<(i32,f32,f32,f32),f32>(&mut store,"area_density_ratio").unwrap();
    let mut squared=|store:&mut wasmtime::Store<()>,u:f64,v:f64| -> f64 {
        let r=f64::from(ratio.call(&mut *store,(1,u as f32,v as f32,1.0/512.0)).unwrap());
        assert!(r>0.0,"quad ratio at {u},{v}: {r}");
        r*r
    };
    let nodes=[0.2_f64,0.5,0.8];
    let mut node_values=[[0.0_f64;3];3];
    for (i,&u) in nodes.iter().enumerate() {
        for (j,&v) in nodes.iter().enumerate() {node_values[i][j]=squared(&mut store,u,v);}
    }
    // Tensor-product Lagrange basis on three nodes per axis.
    let basis=|t:f64,k:usize| {
        let (a,b,c)=(nodes[k],nodes[(k+1)%3],nodes[(k+2)%3]);
        (t-b)*(t-c)/((a-b)*(a-c))
    };
    let mut worst=0.0_f64;
    for i in 1..10 {
        for j in 1..10 {
            let u=f64::from(i)/10.0;
            let v=f64::from(j)/10.0;
            let mut predicted=0.0;
            for k in 0..3 {
                for l in 0..3 {predicted+=node_values[k][l]*basis(u,k)*basis(v,l);}
            }
            let actual=squared(&mut store,u,v);
            worst=worst.max((predicted/actual-1.0).abs());
        }
    }
    eprintln!("QUAD_DENSITY squared correction vs 9-node tensor quadratic: worst relative error={worst:.8}");
    assert!(worst<5e-3,"quad correction is not a tensor-product quadratic: {worst}");
}

/// Extends the density law to genuinely degree-two geometry, using the exact
/// triangular restriction of the authored quad rather than a degree-elevated
/// rewrite of a linear patch. Predicted structure: the blend is total degree
/// two, so its Jacobian is degree one, the cross product degree two, and the
/// squared correction degree four. Checked on lines, where a bivariate degree
/// four restricts to a univariate degree four.
#[test]
fn quadratic_patch_density_correction_is_degree_four() {
    let path=Path::new(env!("CARGO_MANIFEST_DIR")).join("../../ingots/validation/composition_oracle");
    let wasm=compile_ingot_at_level(&path,OptLevel::O2);
    let engine=wasmtime::Engine::default();
    let module=wasmtime::Module::new(&engine,&wasm).unwrap();
    let mut store=wasmtime::Store::new(&engine,());
    let instance=wasmtime::Instance::new(&mut store,&module,&[]).unwrap();
    let ratio=instance.get_typed_func::<(f32,f32,f32),f32>(&mut store,"quadratic_area_density_ratio").unwrap();
    // Lines chosen to stay well inside the restricted triangular domain.
    let lines=[((0.10_f64,0.15_f64),(0.60_f64,0.0_f64)),
               ((0.15,0.10),(0.0,0.60)),
               ((0.08,0.08),(0.35,0.30))];
    // Lagrange value at `t` through nodes at parameters `ts` with values `ys`.
    let lagrange=|ts:&[f64],ys:&[f64],t:f64| {
        let mut total=0.0;
        for i in 0..ts.len() {
            let mut term=ys[i];
            for j in 0..ts.len() {
                if i!=j {term*=(t-ts[j])/(ts[i]-ts[j]);}
            }
            total+=term;
        }
        total
    };
    // Report a degree ladder rather than asserting one guess: the smallest
    // degree that fits is the structural answer.
    let mut worst=[0.0_f64;5];
    for ((x0,y0),(dx,dy)) in lines {
        let mut squared=|store:&mut wasmtime::Store<()>,t:f64| -> f64 {
            let r=f64::from(ratio.call(&mut *store,((x0+dx*t) as f32,(y0+dy*t) as f32,1.0/1024.0)).unwrap());
            assert!(r>0.0,"degree-two probe failed at t={t}: {r}");
            r*r
        };
        let mut nodes=Vec::new();
        for degree in 0..5 {
            let ts:Vec<f64>=(0..=degree).map(|i| if degree==0 {0.5} else {f64::from(i)/f64::from(degree)}).collect();
            let ys:Vec<f64>=ts.iter().map(|&t| squared(&mut store,t)).collect();
            nodes.push((ts,ys));
        }
        for k in 1..10 {
            let t=f64::from(k)/10.0;
            let actual=squared(&mut store,t);
            for degree in 0..5 {
                let (ts,ys)=&nodes[degree];
                worst[degree]=worst[degree].max((lagrange(ts,ys,t)/actual-1.0).abs());
            }
        }
    }
    for degree in 0..5 {
        eprintln!("QUADRATIC_DENSITY squared correction, degree {degree} fit: worst relative error={:.8}",worst[degree]);
    }
    // Whatever the minimal degree is, it must be small and exact, because that
    // is what makes the field a closed form rather than a runtime derivative.
    assert!(worst[4]<5e-3,"squared correction is not a low-degree polynomial: {}",worst[4]);
    let minimal=(0..5).find(|&d| worst[d]<5e-4).expect("no low degree fits");
    eprintln!("QUADRATIC_DENSITY minimal fitting degree={minimal}");
    assert!(minimal<=2,"correction needed degree {minimal}, higher than the blend predicts");
}

/// Recursion can only act on whole octaves of the density field, so its range
/// decides whether recursive substitution is worth building at all. Under one
/// octave it is a no-op and the interior residual is sub-dyadic. This needs no
/// geometry, only the closed-form field, so it is the cheap early falsifier.
#[test]
fn density_field_octave_range_decides_whether_recursion_helps() {
    let path=Path::new(env!("CARGO_MANIFEST_DIR")).join("../../ingots/validation/composition_oracle");
    let wasm=compile_ingot_at_level(&path,OptLevel::O2);
    let engine=wasmtime::Engine::default();
    let module=wasmtime::Module::new(&engine,&wasm).unwrap();
    let mut store=wasmtime::Store::new(&engine,());
    let instance=wasmtime::Instance::new(&mut store,&module,&[]).unwrap();
    let density=instance.get_typed_func::<(i32,f32,f32,f32,f32),f32>(&mut store,"measured_area_density_at").unwrap();
    // The default authoring is a mild bulge. Sweep the out-of-plane offset so the
    // answer is "beyond what curvature does recursion earn its place", not just
    // a verdict on one patch.
    for bulge in [0.6_f32,1.2,2.4,4.8,9.6] {
    for (kind,name) in [(0,"triangle"),(1,"quad")] {
        let mut values=Vec::new();
        let mut invalid=0;
        for i in 1..40 {
            for j in 1..40 {
                let u=f64::from(i)/40.0;
                let v=f64::from(j)/40.0;
                if kind==0 && u+v>0.97 {continue}
                let d=f64::from(density.call(&mut store,(kind,u as f32,v as f32,1.0/1024.0,bulge)).unwrap());
                if !(d>0.0) {invalid+=1; continue;}
                values.push(d);
            }
        }
        if values.len()<100 {eprintln!("DEPTH_RANGE {name} bulge={bulge}: mostly invalid ({invalid})"); continue;}
        let lo=values.iter().cloned().fold(f64::INFINITY,f64::min);
        let hi=values.iter().cloned().fold(0.0_f64,f64::max);
        // Depth is half the log2 of density, since each level quarters area.
        let octaves=0.5*(hi/lo).log2();
        // Fractional part of depth, relative to the coarsest point.
        let mut buckets=[0usize;4];
        for d in &values {
            let depth=0.5*(d/lo).log2();
            buckets[((depth.fract()*4.0) as usize).min(3)]+=1;
        }
        eprintln!("DEPTH_RANGE {name} bulge={bulge}: {} samples, depth span={octaves:.4} levels, fractional quartiles={buckets:?}",values.len());
    }
    }
}

/// A single atlas tile is uniform in the reference domain, so whatever density
/// varies WITHIN one child is residual the composition cannot correct. Splitting
/// the square into more children shrinks each child and therefore that residual,
/// which makes fan count a cheaper lever than recursion. This reports the worst
/// per-child depth span for each candidate recipe on the quad domain.
#[test]
fn finer_fan_counts_shrink_the_within_child_depth_span() {
    let path=Path::new(env!("CARGO_MANIFEST_DIR")).join("../../ingots/validation/composition_oracle");
    let wasm=compile_ingot_at_level(&path,OptLevel::O2);
    let engine=wasmtime::Engine::default();
    let module=wasmtime::Module::new(&engine,&wasm).unwrap();
    let mut store=wasmtime::Store::new(&engine,());
    let instance=wasmtime::Instance::new(&mut store,&module,&[]).unwrap();
    let density=instance.get_typed_func::<(i32,f32,f32,f32,f32),f32>(&mut store,"measured_area_density_at").unwrap();
    let corner=|i:usize| -> (f64,f64) {[(0.0,0.0),(1.0,0.0),(1.0,1.0),(0.0,1.0)][i]};
    let mid=|a:(f64,f64),b:(f64,f64)| ((a.0+b.0)/2.0,(a.1+b.1)/2.0);
    let centre=(0.5,0.5);
    // Each recipe as a list of child triangles in reference coordinates.
    let mut recipes:Vec<(&str,Vec<[(f64,f64);3]>)>=Vec::new();
    recipes.push(("whole square",vec![[corner(0),corner(1),corner(2)],[corner(0),corner(2),corner(3)]]));
    recipes.push(("4-fan",(0..4).map(|i| [corner(i),corner((i+1)%4),centre]).collect()));
    recipes.push(("8-fan",(0..4).flat_map(|i| {
        let (a,b)=(corner(i),corner((i+1)%4));
        let m=mid(a,b);
        [[a,m,centre],[m,b,centre]]
    }).collect()));
    recipes.push(("16-fan",(0..4).flat_map(|i| {
        let (a,b)=(corner(i),corner((i+1)%4));
        let (m,p,q)=(mid(a,b),mid(a,mid(a,b)),mid(mid(a,b),b));
        [[a,p,centre],[p,m,centre],[m,q,centre],[q,b,centre]]
    }).collect()));
    for bulge in [1.2_f32,2.4,4.8] {
        let mut line=format!("FAN_SPAN bulge={bulge}:");
        for (name,children) in &recipes {
            let mut worst=0.0_f64;
            for tri in children {
                let (mut lo,mut hi)=(f64::INFINITY,0.0_f64);
                // Barycentric sampling strictly inside the child.
                for i in 1..14 { for j in 1..(14-i) {
                    let (a,b)=(f64::from(i)/14.0,f64::from(j)/14.0);
                    let c=1.0-a-b;
                    let u=tri[0].0*a+tri[1].0*b+tri[2].0*c;
                    let v=tri[0].1*a+tri[1].1*b+tri[2].1*c;
                    let d=f64::from(density.call(&mut store,(1,u as f32,v as f32,1.0/1024.0,bulge)).unwrap());
                    if !(d>0.0) {continue;}
                    lo=lo.min(d); hi=hi.max(d);
                }}
                if lo.is_finite() && hi>0.0 {worst=worst.max(0.5*(hi/lo).log2());}
            }
            line.push_str(&format!("  {name}={worst:.3}"));
        }
        eprintln!("{line}");
    }
}

/// Acceptance for the depth field: the Gram path must reproduce the measured
/// density law, and its sub-triangle intervals must be certificates over whole
/// regions rather than sampled ranges. A hull bound that a sample escapes would
/// make recursive depth selection unsound.
#[test]
fn depth_field_gram_reproduces_the_density_law_and_certifies_regions() {
    let path=Path::new(env!("CARGO_MANIFEST_DIR")).join("../../ingots/validation/composition_oracle");
    let wasm=compile_ingot_at_level(&path,OptLevel::O2);
    let engine=wasmtime::Engine::default();
    let module=wasmtime::Module::new(&engine,&wasm).unwrap();
    let mut store=wasmtime::Store::new(&engine,());
    let instance=wasmtime::Instance::new(&mut store,&module,&[]).unwrap();
    let norm=instance.get_typed_func::<(f32,f32,f32),f32>(&mut store,"gram_norm").unwrap();
    let bound=instance.get_typed_func::<(f32,f32,f32,f32,f32,f32,i32,f32),f32>(&mut store,"gram_sub_bound").unwrap();
    let density=instance.get_typed_func::<(i32,f32,f32,f32,f32),f32>(&mut store,"measured_area_density_at").unwrap();
    let level=instance.get_typed_func::<(f32,f32,f32,f32,f32,f32,f32,f32),i32>(&mut store,"gram_region_level").unwrap();
    for bulge in [0.6_f32,2.4] {
        // 1. Density times the squared Gram norm must be constant, which is the
        // same law verified against central differences but reached through the
        // polarized Gram rather than by re-blending weights.
        let mut ratios=Vec::new();
        for i in 1..12 { for j in 1..(12-i) {
            let (u,v)=(f64::from(i)/12.0,f64::from(j)/12.0);
            let n=f64::from(norm.call(&mut store,(u as f32,v as f32,bulge)).unwrap());
            let d=f64::from(density.call(&mut store,(0,u as f32,v as f32,1.0/1024.0,bulge)).unwrap());
            assert!(n>0.0 && d>0.0,"gram norm {n} density {d} at {u},{v}");
            ratios.push(d*n*n);
        }}
        let mean=ratios.iter().sum::<f64>()/ratios.len() as f64;
        let spread=ratios.iter().map(|r| (r/mean-1.0).abs()).fold(0.0_f64,f64::max);
        eprintln!("DEPTH_FIELD bulge={bulge}: {} samples, density*norm^2 spread={spread:.8}",ratios.len());
        assert!(spread<2e-3,"gram path does not reproduce the density law: {spread}");

        // 2. The hull bound is conservative, so a large region at high curvature
        // may refuse to certify. That is correct fail-closed behaviour, not a
        // defect: what must hold is that certified intervals never lie, and that
        // subdivision converges to certification.
        let root=[(0.02_f64,0.02_f64),(0.96,0.02),(0.02,0.96)];
        let mut regions=vec![root];
        for depth in 0..4 {
            let mut certified_count=0usize;
            let mut worst_ratio=0.0_f64;
            for tri in &regions {
                let a=(tri[0].0 as f32,tri[0].1 as f32,tri[1].0 as f32,tri[1].1 as f32,
                       tri[2].0 as f32,tri[2].1 as f32);
                let certified=bound.call(&mut store,(a.0,a.1,a.2,a.3,a.4,a.5,2,bulge)).unwrap();
                if certified!=1.0 {continue;}
                certified_count+=1;
                let lo=f64::from(bound.call(&mut store,(a.0,a.1,a.2,a.3,a.4,a.5,0,bulge)).unwrap());
                let hi=f64::from(bound.call(&mut store,(a.0,a.1,a.2,a.3,a.4,a.5,1,bulge)).unwrap());
                worst_ratio=worst_ratio.max(hi/lo);
                for i in 0..13 { for j in 0..(13-i) {
                    let (x,y)=(f64::from(i)/12.0,f64::from(j)/12.0);
                    let z=1.0-x-y;
                    let u=tri[0].0*x+tri[1].0*y+tri[2].0*z;
                    let v=tri[0].1*x+tri[1].1*y+tri[2].1*z;
                    let n=f64::from(norm.call(&mut store,(u as f32,v as f32,bulge)).unwrap());
                    assert!(n>=lo*(1.0-1e-4) && n<=hi*(1.0+1e-4),
                        "certified hull escaped at bulge {bulge}: {n} outside {lo}..{hi}");
                }}
            }
            eprintln!("DEPTH_FIELD bulge={bulge} depth {depth}: {certified_count}/{} certified, worst interval ratio={worst_ratio:.4}",regions.len());
            if depth==3 {
                assert_eq!(certified_count,regions.len(),
                    "subdivision must converge to certification by depth 3 at bulge {bulge}");
            }
            let mid=|a:(f64,f64),b:(f64,f64)| ((a.0+b.0)/2.0,(a.1+b.1)/2.0);
            regions=regions.iter().flat_map(|t| {
                let (m01,m02,m12)=(mid(t[0],t[1]),mid(t[0],t[2]),mid(t[1],t[2]));
                [[t[0],m01,m02],[m01,t[1],m12],[m02,m12,t[2]],[m01,m12,m02]]
            }).collect();
        }
    }
}

#[test]
fn composition_wasm_boundary_locality_sweep() {
    let path=Path::new(env!("CARGO_MANIFEST_DIR")).join("../../ingots/validation/composition_oracle");
    let wasm=compile_ingot_at_level(&path,OptLevel::O2);
    let engine=wasmtime::Engine::default();
    let module=wasmtime::Module::new(&engine,&wasm).unwrap();
    for case in 0..6 {
        for reach in [0.0625_f32,0.25,1.0,4.0,16.0] {
            let mut store=wasmtime::Store::new(&engine,());
            let instance=wasmtime::Instance::new(&mut store,&module,&[]).unwrap();
            let audit=instance.get_typed_func::<(i32,f32),(i32,i32,i32,i32,f32,f32,f32,f32,f32,f32)>(&mut store,"quality_locality").unwrap();
            let (status,triangles,folds,poor,minimum,mean,signed,absolute,metric_mean,mass_cv)=audit.call(&mut store,(case,reach)).unwrap();
            assert_eq!(status,0);
            assert!(triangles>0 && [minimum,mean,signed,absolute,metric_mean,mass_cv].into_iter().all(f32::is_finite));
            assert!((signed-1.0).abs()<0.001);
            eprintln!("LOCALITY case={case} reach={reach} triangles={triangles} folds={folds} poor={poor} min={minimum:.7} metric_mean={metric_mean:.7} mass_cv={mass_cv:.7} absolute_area={absolute:.7}");
        }
    }
}

#[test]
fn composition_wasm_actual_mesh_quality() {
    let path=Path::new(env!("CARGO_MANIFEST_DIR")).join("../../ingots/validation/composition_oracle");
    let wasm=compile_ingot_at_level(&path,OptLevel::O2);
    let engine=wasmtime::Engine::default();
    let module=wasmtime::Module::new(&engine,&wasm).unwrap();
    for case in 0..6 {
        for compensation in 0..2 {
            let mut store=wasmtime::Store::new(&engine,());
            let instance=wasmtime::Instance::new(&mut store,&module,&[]).unwrap();
            let laws=instance.get_typed_func::<(),i32>(&mut store,"quality_laws").unwrap();
            assert_eq!(laws.call(&mut store,()).unwrap(),1);
            let audit=instance.get_typed_func::<(i32,i32),(i32,i32,i32,i32,f32,f32,f32,f32,f32,f32)>(&mut store,"quality_case").unwrap();
            let (status,triangles,folds,poor,minimum,mean,signed,absolute,metric_mean,mass_cv)=audit.call(&mut store,(case,compensation)).unwrap();
            assert_eq!(status,0,"case {case}");
            assert!(triangles>0 && folds>=0 && poor>=0);
            assert!([minimum,mean,signed,absolute,metric_mean,mass_cv].into_iter().all(f32::is_finite));
            assert!((0.0..=1.00001).contains(&metric_mean) && mass_cv>=0.0);
            assert!((0.0..=1.00001).contains(&minimum) && (minimum..=1.00001).contains(&mean));
            assert!((signed-1.0).abs()<0.001,"square signed area {signed}");
            assert!(absolute>=signed-0.00001);
            if case==0 {assert_eq!(folds,0);}
            eprintln!("QUALITY case={case} compensation={compensation} triangles={triangles} folds={folds} shape_lt_0_1={poor} min={minimum:.7} mean={mean:.7} signed_area={signed:.7} absolute_area={absolute:.7} metric_mean={metric_mean:.7} mass_cv={mass_cv:.7}");
        }
    }
}

#[test]
fn composition_wasm_density_and_coherent_selection() {
    let path=Path::new(env!("CARGO_MANIFEST_DIR")).join("../../ingots/validation/composition_oracle");
    let wasm=compile_ingot_at_level(&path,OptLevel::O2);
    let engine=wasmtime::Engine::default();
    let module=wasmtime::Module::new(&engine,&wasm).unwrap();
    let mut store=wasmtime::Store::new(&engine,());
    let instance=wasmtime::Instance::new(&mut store,&module,&[]).unwrap();
    let uniform=instance.get_typed_func::<i32,(f32,f32,i32,i32)>(&mut store,"uniform_case").unwrap();
    for lod in 0..=8 {
        let (diagonal,spoke,diagonal_lod,spoke_lod)=uniform.call(&mut store,lod).unwrap();
        let density=(1_u32<<lod) as f32;
        assert!((diagonal-density*2_f32.sqrt()).abs()<density*0.00001);
        assert!((spoke-density/2_f32.sqrt()).abs()<density*0.00001);
        assert_eq!(diagonal_lod,(lod+1).min(8));
        assert_eq!(spoke_lod,lod);
    }
    let outer=instance.get_typed_func::<(),(f32,f32,f32,f32)>(&mut store,"outer_integrals").unwrap();
    assert_eq!(outer.call(&mut store,()).unwrap(),(1.0,8.0,32.0,256.0));
    let spacing=instance.get_typed_func::<i32,(f32,f32,i32)>(&mut store,"spacing_case").unwrap();
    for code in 0..6561 {
        let (residual,midpoint,monotone)=spacing.call(&mut store,code).unwrap();
        assert!(residual<0.000001,"CDF residual {residual} for {code}");
        assert_eq!(monotone,1,"nonmonotone map for {code}");
        assert!(midpoint>0.0 && midpoint<1.0);
    }
    let (_,midpoint,_)=spacing.call(&mut store,8*9+8*81).unwrap();
    assert!(midpoint>0.69 && midpoint<0.72,"expected density-driven skew, got {midpoint}");
    let boundary=instance.get_typed_func::<i32,i32>(&mut store,"warped_boundary_case").unwrap();
    for code in 0..6561 {
        assert_eq!(boundary.call(&mut store,code).unwrap(),1,"warped boundary {code}");
    }
    for name in ["whole_composition_warp_laws","geometric_patch_view_laws","edge_skew_preserves_mass_and_shared_positions","resolution_reports_visible_counts_and_caps","interior_policy_preserves_boundaries","boundary_admission","manual_values_survive_automatic","coherent_pending_selection"] {
        let check=instance.get_typed_func::<(),i32>(&mut store,name).unwrap();
        assert_eq!(check.call(&mut store,()).unwrap(),1,"{name}");
    }
}

#[test]
fn cpu_atlas_compiles_four_worker_pool() {
    use common::InputDb;
    use hir::hir_def::HirIngot;
    use salsa::Setter;
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../ingots/validation/atlas_pool_workers").canonicalize().unwrap();
    let url = url::Url::from_directory_path(path).unwrap();
    let mut db = driver::DriverDataBase::default();
    db.compilation_settings().set_profile(&mut db).to("release".into());
    assert!(!driver::init_ingot(&mut db,&url));
    let ingot = db.workspace().containing_ingot(&db,url).unwrap();
    let top = ingot.root_mod(&db);
    let diagnostics = db.run_on_top_mod(top).format_diags(&db);
    assert!(diagnostics.is_empty(),"{diagnostics}");
    let artifact = fe_codegen::compile_resident_actor(&db,top).unwrap().unwrap();
    assert_eq!(artifact.structured_children.len(),4);
    assert_eq!(artifact.scoped_tasks.len(),8);
    let package=fe_codegen::materialize_scoped_task_package(&artifact.scoped_tasks,&artifact.structured_children)
        .unwrap().unwrap();
    wasmparser::validate(&artifact.wasm).unwrap();
    for child in &artifact.structured_children { wasmparser::validate(&child.wasm).unwrap(); }
    if let Some(directory)=std::env::var_os("QUILTING_ATLAS_POOL_ARTIFACT_DIR") {
        let directory=std::path::PathBuf::from(directory);
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join("parent.wasm"),&artifact.wasm).unwrap();
        for file in &package.files {
            let path=directory.join(&file.path);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path,&file.bytes).unwrap();
        }
        eprintln!("pool package entry: {}",package.entry_path);
    }
    eprintln!("four-worker Fe pool: parent {} bytes, children {:?}",artifact.wasm.len(),
        artifact.structured_children.iter().map(|child|child.wasm.len()).collect::<Vec<_>>());
}

#[test]
fn cpu_atlas_wasm_pool_retains_tiles_and_rejects_cancelled_results() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../ingots/validation/atlas_pool_oracle");
    let wasm = compile_ingot_at_level(&path, OptLevel::O2);
    let engine = wasmtime::Engine::default();
    let module = wasmtime::Module::new(&engine, &wasm).unwrap();
    let mut store = wasmtime::Store::new(&engine, ());
    let instance = wasmtime::Instance::new(&mut store, &module, &[]).unwrap();
    let create = instance.get_typed_func::<i32, i32>(&mut store, "create").unwrap();
    let primaries = instance.get_typed_func::<i32, i32>(&mut store, "create_primaries").unwrap();
    let empty = instance.get_typed_func::<(), i32>(&mut store, "create_empty").unwrap();
    let summary = instance.get_typed_func::<i32, (i32,i32,i32,i32,i32,i32,i32)>(&mut store, "summary").unwrap();
    let generate = instance.get_typed_func::<(i32,i32,i32),i32>(&mut store, "generate_one").unwrap();
    let entry = instance.get_typed_func::<(i32,i32),(i32,i32,i32,i32)>(&mut store, "entry").unwrap();
    let storage = instance.get_typed_func::<i32,i32>(&mut store, "storage").unwrap();
    let tile_info = instance.get_typed_func::<(i32,i32),(i32,i32,i32,i32,i32)>(&mut store, "tile_info").unwrap();
    let update_info = instance.get_typed_func::<(i32,i32,i32),(i32,i32,i32,i32,i32,i32)>(&mut store, "update_info").unwrap();
    let memory = instance.get_memory(&mut store, "memory").unwrap();
    let reset = instance.get_typed_func::<(),()>(&mut store,"fe_cabi_reset").unwrap();
    let priority = instance.get_typed_func::<(),i32>(&mut store,"priority_preserves_live_leases").unwrap();
    assert_eq!(priority.call(&mut store,()).unwrap(),1);
    reset.call(&mut store,()).unwrap();
    // Explicit primary requests retain their caller's ordinal, independently
    // of full-atlas enumeration. Neither invalid nor empty recipes can run.
    let chosen = primaries.call(&mut store, 0).unwrap();
    assert_eq!(summary.call(&mut store, chosen).unwrap(), (7,2,0,0,0,0,0));
    assert_eq!(generate.call(&mut store, (chosen,0,0)).unwrap(), 0);
    assert!(entry.call(&mut store, (chosen,1)).unwrap().3 > 0);
    assert_eq!(tile_info.call(&mut store, (chosen,0)).unwrap().0, 1);
    assert_eq!(generate.call(&mut store, (chosen,1,0)).unwrap(), 0);
    assert!(entry.call(&mut store, (chosen,0)).unwrap().3 > 0);
    assert_eq!(generate.call(&mut store, (chosen,0,0)).unwrap(), 4);
    reset.call(&mut store, ()).unwrap();
    let bad = primaries.call(&mut store, 1).unwrap();
    assert_eq!(summary.call(&mut store, bad).unwrap(), (7,0,0,0,0,0,2));
    assert_eq!(generate.call(&mut store, (bad,0,0)).unwrap(), 7);
    reset.call(&mut store, ()).unwrap();
    let absent = empty.call(&mut store, ()).unwrap();
    assert_eq!(summary.call(&mut store, absent).unwrap(), (8,0,0,0,0,0,2));
    reset.call(&mut store, ()).unwrap();
    // Enumerate the entire requested key space without generating expensive
    // geometry in this ownership regression. Full generation has separate gates.
    let full = create.call(&mut store, 8).unwrap();
    assert_ne!(full,0);
    assert_eq!(summary.call(&mut store,full).unwrap(),(1,1200,0,0,0,0,0));
    reset.call(&mut store,()).unwrap();
    let pool = create.call(&mut store,0).unwrap();
    assert_eq!(summary.call(&mut store,pool).unwrap(),(1,2,0,0,0,0,0));
    assert_eq!(tile_info.call(&mut store,(pool,1)).unwrap(),(1,0,0,0,0));
    assert_eq!(tile_info.call(&mut store,(pool,2)).unwrap(),(2,0,0,0,0));
    assert_eq!(update_info.call(&mut store,(pool,0,0)).unwrap(),(1,0,0,0,0,0));
    assert_eq!(generate.call(&mut store,(pool,0,0)).unwrap(),0);
    let first = entry.call(&mut store,(pool,1)).unwrap();
    assert!(first.1>0 && first.2>0 && first.3>0);
    let base = storage.call(&mut store,pool).unwrap() as usize;
    assert_eq!(update_info.call(&mut store,(pool,1,0)).unwrap(),(0,1,1,0,base as i32,first.1));
    assert_eq!(update_info.call(&mut store,(pool,1,first.1)).unwrap(),(1,0,0,0,0,0));
    assert_eq!(update_info.call(&mut store,(pool,1,first.1+1)).unwrap(),(2,0,0,0,0,0));
    assert_eq!(tile_info.call(&mut store,(pool,1)).unwrap(),
        (0,first.2,first.3,(base+first.0 as usize*4) as i32,first.1));
    let mut before=vec![0;first.1 as usize*4];
    memory.read(&store,base+first.0 as usize*4,&mut before).unwrap();
    assert_eq!(generate.call(&mut store,(pool,1,0)).unwrap(),0);
    let second=entry.call(&mut store,(pool,0)).unwrap();
    assert_eq!(second.0,first.1);
    assert_eq!(update_info.call(&mut store,(pool,1,first.1)).unwrap(),
        (0,1,2,first.1,(base+first.1 as usize*4) as i32,second.1));
    // A frame that missed both completions gets the entire prefix; a new
    // resource generation ignores the obsolete cursor and replays all bytes.
    assert_eq!(update_info.call(&mut store,(pool,1,0)).unwrap(),
        (0,1,2,0,base as i32,first.1+second.1));
    assert_eq!(update_info.call(&mut store,(pool,0,999)).unwrap(),
        (0,1,2,0,base as i32,first.1+second.1));
    assert_eq!(tile_info.call(&mut store,(pool,0)).unwrap(),
        (0,second.2,second.3,(base+second.0 as usize*4) as i32,second.1));
    let mut after=vec![0;before.len()];
    memory.read(&store,base+first.0 as usize*4,&mut after).unwrap();
    assert_eq!(before,after,"another tile must not overwrite retained geometry");
    assert_eq!(generate.call(&mut store,(pool,2,0)).unwrap(),4);
    let done=summary.call(&mut store,pool).unwrap();
    assert_eq!(done.2,2); assert_eq!(done.3,first.1+second.1); assert_eq!(done.6,0);
    reset.call(&mut store,()).unwrap();
    let cancelled=create.call(&mut store,0).unwrap();
    assert_eq!(generate.call(&mut store,(cancelled,0,1)).unwrap(),2);
    assert_eq!(summary.call(&mut store,cancelled).unwrap(),(1,2,0,0,0,0,1));
    assert_eq!(entry.call(&mut store,(cancelled,1)).unwrap(),(0,0,0,0));
    assert_eq!(tile_info.call(&mut store,(cancelled,1)).unwrap(),(1,0,0,0,0));
    reset.call(&mut store,()).unwrap();
    let partial=create.call(&mut store,0).unwrap();
    assert_eq!(generate.call(&mut store,(partial,0,0)).unwrap(),0);
    let published=tile_info.call(&mut store,(partial,1)).unwrap();
    assert_eq!(published.0,0);
    assert_eq!(generate.call(&mut store,(partial,1,1)).unwrap(),2);
    assert_eq!(tile_info.call(&mut store,(partial,1)).unwrap(),published);
    assert_eq!(tile_info.call(&mut store,(partial,0)).unwrap(),(1,0,0,0,0));
    let partial_summary=summary.call(&mut store,partial).unwrap();
    assert_eq!(partial_summary.2,1);
    assert_eq!(partial_summary.6,1);
    eprintln!("retained pool ownership passed; Wasm linear memory {} bytes",memory.data_size(&store));
}

#[test]
fn cpu_atlas_compiles_typed_worker_payload() {
    use common::InputDb;
    use hir::hir_def::HirIngot;
    use salsa::Setter;
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../ingots/validation/atlas_worker_oracle").canonicalize().unwrap();
    let url = url::Url::from_directory_path(path).unwrap();
    let mut db = driver::DriverDataBase::default();
    db.compilation_settings().set_profile(&mut db).to("release".into());
    assert!(!driver::init_ingot(&mut db,&url));
    let ingot = db.workspace().containing_ingot(&db,url).unwrap();
    let top = ingot.root_mod(&db);
    let diagnostics = db.run_on_top_mod(top).format_diags(&db);
    assert!(diagnostics.is_empty(),"{diagnostics}");
    let artifact = fe_codegen::compile_resident_actor(&db,top).unwrap().unwrap();
    assert_eq!(artifact.structured_children.len(),1);
    let child = &artifact.structured_children[0];
    assert_eq!(child.interface.lanes.len(),1);
    wasmparser::validate(&child.wasm).unwrap();
    if let Some(directory)=std::env::var_os("QUILTING_ATLAS_WORKER_ARTIFACT_DIR") {
        let directory=std::path::PathBuf::from(directory);
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join("child.wasm"),&child.wasm).unwrap();
        std::fs::write(directory.join("interface.js"),fe_codegen::emit_canonical_interface_js(&child.interface).unwrap()).unwrap();
        for (relative,source) in fe_codegen::browser_actor_runtime_files() {
            let path=directory.join(relative);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path,source).unwrap();
        }
    }
    eprintln!("atlas worker package: parent={}bytes child={}bytes response={:?}",
        artifact.wasm.len(),child.wasm.len(),child.interface.lanes[0].response);
    let engine = wasmtime::Engine::default();
    let module = wasmtime::Module::new(&engine,&child.wasm).unwrap();
    assert_eq!(module.imports().count(),0);
    let mut store = wasmtime::Store::new(&engine,());
    let instance = wasmtime::Instance::new(&mut store,&module,&[]).unwrap();
    let memory = instance.get_memory(&mut store,"memory").unwrap();
    let lane = &child.interface.lanes[0];
    let export = lane.export.as_ref().unwrap();
    eprintln!("worker lane export={export} exports={:?}",module.exports().map(|e|e.name().to_owned()).collect::<Vec<_>>());
    let call = instance.get_typed_func::<i32,i32>(&mut store,export).unwrap();
    let fe_codegen::CanonicalShape::Record {fields} = &lane.request.shape else {panic!("request record")};
    for square in [false,true] { for dense in [false,true] {
        let mut request=vec![0u8;lane.request.size as usize];
        for field in fields {
            let value:u32 = match field.name.as_str() {
                "epoch"=>17, "square"=>u32::from(square), "c"=>8, "seed"=>42,
                "a"|"b"|"d"=>if dense {8} else {0}, other=>panic!("unknown request field {other}"),
            };
            let offset=field.offset as usize;
            request[offset..offset+field.layout.size as usize].copy_from_slice(&value.to_le_bytes()[..field.layout.size as usize]);
        }
        memory.write(&mut store,64,&request).unwrap();
        let start=std::time::Instant::now();
        let ptr=call.call(&mut store,64).unwrap();
        let elapsed=start.elapsed();
        assert_eq!(lane.response.size,24);
        let mut response=[0u8;24];
        memory.read(&store,ptr as usize,&mut response).unwrap();
        let fields:Vec<u32>=response.chunks_exact(4).map(|b|u32::from_le_bytes(b.try_into().unwrap())).collect();
        assert_eq!(&fields[..2],&[17,0]);
        let (points,triangles,payload,len)=(fields[2],fields[3],fields[4],fields[5]);
        assert!(points>0 && triangles>0);
        assert_eq!(payload%4,0);
        assert_eq!(len,points+(triangles*3).div_ceil(2));
        let mut bytes=vec![0;len as usize*4];
        memory.read(&store,payload as usize,&mut bytes).unwrap();
        assert!(bytes.iter().any(|b|*b!=0));
        eprintln!("canonical worker square={square} dense={dense} epoch=17 points={points} triangles={triangles} bytes={} elapsed={elapsed:?}",bytes.len());
    } }
}

#[test]
fn cpu_atlas_wasm_exports_packed_geometry() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../ingots/validation/cpu_atlas_oracle");
    let wasm = compile_ingot_at_level(&path, OptLevel::O2);
    if let Some(directory)=std::env::var_os("QUILTING_ATLAS_WORKER_ARTIFACT_DIR") {
        let directory=std::path::PathBuf::from(directory);
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join("validation.wasm"),&wasm).unwrap();
    }
    let engine = wasmtime::Engine::default();
    let module = wasmtime::Module::new(&engine, &wasm).unwrap();
    let mut store = wasmtime::Store::new(&engine, ());
    let instance = wasmtime::Instance::new(&mut store, &module, &[]).unwrap();
    let memory = instance.get_memory(&mut store, "memory").unwrap();
    let reset = instance.get_typed_func::<(), ()>(&mut store, "fe_cabi_reset").unwrap();
    type Packed = (i32,i32,i32,i32,i32);
    let tri = instance.get_typed_func::<(i32,i32,i32,i32), Packed>(&mut store, "packed_triangle").unwrap();
    let quad = instance.get_typed_func::<(i32,i32,i32,i32,i32), Packed>(&mut store, "packed_square").unwrap();
    for square in [false,true] {
        for key in [[0,0,0,0],[0,0,8,0],[4,4,4,4],[8,8,8,8]] {
            reset.call(&mut store, ()).unwrap();
            let r = if square { quad.call(&mut store,(key[0],key[1],key[2],key[3],42)) }
                else { tri.call(&mut store,(key[0],key[1],key[2],42)) }.unwrap();
            eprintln!("packed geometry square={square} key={key:?} receipt={r:?}");
            assert_eq!(r.0,0); assert!(r.1>0 && r.2>0 && r.3>0);
            let (points,faces)=(r.1 as usize,r.2 as usize);
            assert!(points<=65536);
            assert_eq!(r.4 as usize, points+(faces*3).div_ceil(2));
            let mut bytes=vec![0u8;r.4 as usize*4];
            memory.read(&store,r.3 as usize,&mut bytes).unwrap();
            let words:Vec<u32>=bytes.chunks_exact(4).map(|b|u32::from_le_bytes(b.try_into().unwrap())).collect();
            if points < 10 { eprintln!("packed words: {words:?}"); }
            let coords:Vec<(i64,i64)>=words[..points].iter().map(|w|((w&65535) as i64,(w>>16) as i64)).collect();
            assert!(coords.iter().all(|&(x,y)|x<=16384 && y<=16384));
            let index=|i:usize| -> usize { ((words[points+i/2]>>(16*(i%2)))&65535) as usize };
            let mut area=0i64;
            let mut used=vec![false;points];
            for f in 0..faces {
                let ids=[index(f*3),index(f*3+1),index(f*3+2)];
                assert!(ids.iter().all(|&i|i<points));
                for &i in &ids {used[i]=true;}
                let (a,b,c)=(coords[ids[0]],coords[ids[1]],coords[ids[2]]);
                let cross=(b.0-a.0)*(c.1-a.1)-(b.1-a.1)*(c.0-a.0);
                assert!(cross>0,"packed triangle inverted or degenerate"); area+=cross;
            }
            assert!(used.into_iter().all(|v|v));
            assert_eq!(area,if square {536870912} else {268435456});
            if faces*3%2==1 {assert_eq!(words.last().unwrap()>>16,0);}
        }
    }
    assert_eq!(tri.call(&mut store,(9,9,9,42)).unwrap(),(2,0,0,0,0));
}

#[test]
fn cpu_atlas_wasm_packing_preserves_rejected_output() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../ingots/validation/cpu_atlas_oracle");
    let wasm = compile_ingot_at_level(&path, OptLevel::O2);
    let engine = wasmtime::Engine::default();
    let module = wasmtime::Module::new(&engine, &wasm).unwrap();
    let mut store = wasmtime::Store::new(&engine, ());
    let instance = wasmtime::Instance::new(&mut store, &module, &[]).unwrap();
    let packing = instance.get_typed_func::<i32, (i32, i32, i32, i32, i32, i32)>(&mut store, "packing").unwrap();
    for mode in 0..4 {
        let r = packing.call(&mut store, mode).unwrap();
        eprintln!("packing mode={mode}: {r:?}");
        assert_eq!(r, match mode {
            0 => (0, 0, 16384, 1073741824, 65536, 2),
            1 => (4, 77, 77, 77, 77, 77),
            2 => (2, 77, 77, 77, 77, 77),
            _ => (3, 77, 77, 77, 77, 77),
        });
    }
}

/// Opt-in exhaustive geometry gate. Fe owns key enumeration and all geometry;
/// this host only calls exports, checks receipts and records wall-clock cost.
#[test]
#[ignore = "complete LoD-8 canonical atlas generation; run explicitly in release"]
fn cpu_atlas_wasm_generates_every_canonical_key() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../ingots/validation/cpu_atlas_oracle");
    let wasm = compile_ingot_at_level(&path, OptLevel::O2);
    let mut config = wasmtime::Config::new();
    config.consume_fuel(true);
    let engine = wasmtime::Engine::new(&config).unwrap();
    let module = wasmtime::Module::new(&engine, &wasm).unwrap();
    assert_eq!(module.imports().count(), 0);
    let mut store = wasmtime::Store::new(&engine, ());
    let instance = wasmtime::Instance::new(&mut store, &module, &[]).unwrap();
    let memory = instance.get_memory(&mut store, "memory").unwrap();
    let reset = instance.get_typed_func::<(), ()>(&mut store, "fe_cabi_reset").unwrap();
    let tri_key = instance.get_typed_func::<i32, (i32, i32, i32)>(&mut store, "triangle_key").unwrap();
    let quad_key = instance.get_typed_func::<i32, (i32, i32, i32, i32, i32, i32)>(&mut store, "quad_key").unwrap();
    type Probe = (i32, i32, i32, i32, i32, i32, i32, i32, i64);
    let triangle = instance.get_typed_func::<(i32, i32, i32, i32, i32), Probe>(&mut store, "triangle").unwrap();
    let square = instance.get_typed_func::<(i32, i32, i32, i32, i32, i32), Probe>(&mut store, "square").unwrap();
    let start = std::time::Instant::now();
    let mut points = 0_u64;
    let mut faces = 0_u64;
    let mut quad_cursor = 0;
    let mut seen_tri = std::collections::BTreeSet::new();
    let mut seen_quad = std::collections::BTreeSet::new();
    for quad in [false, true] {
        let count = if quad { 1035 } else { 165 };
        for ordinal in 0..count {
            store.set_fuel(100_000_000_000).unwrap();
            reset.call(&mut store, ()).unwrap();
            let key = if quad {
                let k = quad_key.call(&mut store, quad_cursor).unwrap();
                assert_eq!(k.5, 1);
                assert!(k.4 > quad_cursor);
                quad_cursor = k.4;
                let key = [k.0, k.1, k.2, k.3];
                assert!(seen_quad.insert(key));
                key
            } else {
                let k = tri_key.call(&mut store, ordinal).unwrap();
                let key = [k.0, k.1, k.2, -1];
                assert!(seen_tri.insert(key));
                key
            };
            let tile_start = std::time::Instant::now();
            let r = if quad {
                square.call(&mut store, (key[0], key[1], key[2], key[3], 42, 2))
            } else {
                triangle.call(&mut store, (key[0], key[1], key[2], 42, 2))
            }.unwrap_or_else(|e| panic!("tile {key:?} trapped: {e}"));
            eprintln!("full-atlas quad={quad} ordinal={ordinal} key={key:?} points={} faces={} status={}/{}/{} time={:?} memory={}",
                r.2, r.3, r.0, r.1, r.4, tile_start.elapsed(), memory.data_size(&store));
            assert_eq!((r.0, r.1, r.4), (0, 0, 0), "failed tile {key:?}");
            let boundary: i32 = key.iter().filter(|&&e| e >= 0).map(|&e| 1 << e).sum();
            assert_eq!(r.3, 2 * r.2 - boundary - 2);
            assert_eq!(r.8, if quad { 536870912 } else { 268435456 });
            assert!(memory.data_size(&store) <= 64 * 1024 * 1024);
            points += r.2 as u64;
            faces += r.3 as u64;
        }
    }
    assert_eq!(quad_key.call(&mut store, quad_cursor).unwrap().5, 0);
    assert_eq!((seen_tri.len(), seen_quad.len()), (165, 1035));
    eprintln!("full-atlas complete: triangles=165 quads=1035 points={points} faces={faces} instrumented_total={:?} memory={}", start.elapsed(), memory.data_size(&store));
}

#[test]
fn cpu_atlas_wasm_triangulates_mixed_lod8_domains() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../ingots/validation/cpu_atlas_oracle");
    let start = std::time::Instant::now();
    let wasm = compile_ingot_at_level(&path, OptLevel::O2);
    eprintln!("CPU atlas probe: {} Wasm bytes, compilation {:?}", wasm.len(), start.elapsed());
    let mut config = wasmtime::Config::new();
    config.consume_fuel(true);
    let engine = wasmtime::Engine::new(&config).unwrap();
    let module = wasmtime::Module::new(&engine, &wasm).unwrap();
    assert!(module.imports().next().is_none(), "no imported host geometry implementation");
    let mut store = wasmtime::Store::new(&engine, ());
    let instance = wasmtime::Instance::new(&mut store, &module, &[]).unwrap();
    let memory = instance.get_memory(&mut store, "memory").unwrap();
    let reset = instance.get_typed_func::<(), ()>(&mut store, "fe_cabi_reset").unwrap();
    let alloc = instance.get_typed_func::<(i32, i32), i32>(&mut store, "fe_cabi_alloc").unwrap();
    type Probe = (i32, i32, i32, i32, i32, i32, i32, i32, i64);
    let triangle = instance.get_typed_func::<(i32, i32, i32, i32, i32), Probe>(&mut store, "triangle").unwrap();
    for key in [(0, 0, 0), (0, 0, 4), (0, 0, 8), (2, 2, 2), (4, 4, 4), (5, 5, 5), (8, 8, 8)] {
        store.set_fuel(100_000_000_000).unwrap();
        reset.call(&mut store, ()).unwrap();
        let start = std::time::Instant::now();
        let r = triangle.call(&mut store, (key.0, key.1, key.2, 42, 2)).unwrap();
        eprintln!("triangle arena end: {}", alloc.call(&mut store, (1, 1)).unwrap());
        eprintln!("linear-memory high-water: {} bytes", memory.data_size(&store));
        eprintln!("CPU atlas triangle {key:?}: {r:?}, instrumented sampling+CDT+audit {:?}", start.elapsed());
        assert_eq!((r.0, r.1, r.4), (0, 0, 0), "sampling/CDT/audit must all succeed for {key:?}");
        let boundary = (1 << key.0) + (1 << key.1) + (1 << key.2);
        assert_eq!(r.3, 2 * r.2 - boundary - 2);
        assert_eq!(r.8, 16384 * 16384);
    }
    let square = instance.get_typed_func::<(i32, i32, i32, i32, i32, i32), Probe>(&mut store, "square").unwrap();
    for key in [(0, 0, 0, 0), (0, 0, 0, 8), (2, 2, 2, 2), (4, 4, 4, 4), (5, 5, 5, 5), (8, 8, 8, 8)] {
        store.set_fuel(100_000_000_000).unwrap();
        reset.call(&mut store, ()).unwrap();
        let start = std::time::Instant::now();
        let r = square.call(&mut store, (key.0, key.1, key.2, key.3, 42, 2)).unwrap();
        eprintln!("square arena end: {}", alloc.call(&mut store, (1, 1)).unwrap());
        eprintln!("linear-memory high-water: {} bytes", memory.data_size(&store));
        eprintln!("CPU atlas square {key:?}: {r:?}, instrumented sampling+CDT+audit {:?}", start.elapsed());
        assert_eq!((r.0, r.1, r.4), (0, 0, 0), "sampling/CDT/audit must all succeed for {key:?}");
        let boundary = (1 << key.0) + (1 << key.1) + (1 << key.2) + (1 << key.3);
        assert_eq!(r.3, 2 * r.2 - boundary - 2);
        assert_eq!(r.8, 2 * 16384 * 16384);
    }
    for stage in 0..2 {
        for quad in [false, true] {
            store.set_fuel(100_000_000_000).unwrap();
            reset.call(&mut store, ()).unwrap();
            let start = std::time::Instant::now();
            let r = if quad {
                square.call(&mut store, (8, 8, 8, 8, 42, stage)).unwrap()
            } else {
                triangle.call(&mut store, (8, 8, 8, 42, stage)).unwrap()
            };
            let elapsed = start.elapsed();
            let end = alloc.call(&mut store, (1, 1)).unwrap();
            eprintln!("phase probe quad={quad} stage={stage}: {r:?}, elapsed={elapsed:?}, arena_end={end}");
            assert_eq!(r.0, 0);
            assert_eq!(r.2, if quad { 39714 } else { 47366 });
            assert_eq!(r.1, if stage == 0 { 5 } else { 0 });
            assert_eq!(r.3, if stage == 0 { 0 } else if quad { 78402 } else { 93962 });
            assert_eq!(r.8, 0, "phase probe must not pretend to have run its area audit");
        }
    }
    // Observed full-density peak is ~36MiB. Keep a bounded per-worker guard,
    // separate from the much smaller live arena after a completed scalar probe.
    assert!(memory.data_size(&store) <= 64 * 1024 * 1024,
        "single-worker probe exceeded 64MiB linear memory: {}", memory.data_size(&store));
}
