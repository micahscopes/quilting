//! Host execution/acceptance only; sampling, CDT and mesh audits are Fe.
use super::fe_oracle::compile_ingot_at_level;
use fe_codegen::OptLevel;
use std::path::Path;

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
    for square in [false,true] {
        let mut request=vec![0u8;lane.request.size as usize];
        for field in fields {
            let value:u32 = match field.name.as_str() {
                "epoch"=>17, "square"=>u32::from(square), "c"=>8, "seed"=>42,
                "a"|"b"|"d"=>0, other=>panic!("unknown request field {other}"),
            };
            let offset=field.offset as usize;
            request[offset..offset+field.layout.size as usize].copy_from_slice(&value.to_le_bytes()[..field.layout.size as usize]);
        }
        memory.write(&mut store,64,&request).unwrap();
        let ptr=call.call(&mut store,64).unwrap();
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
        eprintln!("canonical worker square={square} epoch=17 points={points} triangles={triangles} bytes={}",bytes.len());
    }
}

#[test]
fn cpu_atlas_wasm_exports_packed_geometry() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../ingots/validation/cpu_atlas_oracle");
    let wasm = compile_ingot_at_level(&path, OptLevel::O2);
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
