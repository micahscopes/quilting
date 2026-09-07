//! Host execution/acceptance only; sampling, CDT and mesh audits are Fe.
use super::fe_oracle::compile_ingot_at_level;
use fe_codegen::OptLevel;
use std::path::Path;

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
