//! Host execution/acceptance only; sampling, CDT and mesh audits are Fe.
use super::fe_oracle::compile_ingot_at_level;
use fe_codegen::OptLevel;
use std::path::Path;

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
    for name in ["edge_skew_preserves_mass_and_shared_positions","resolution_reports_visible_counts_and_caps","interior_policy_preserves_boundaries","boundary_admission","manual_values_survive_automatic","coherent_pending_selection"] {
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
