//! Host execution/acceptance only; sampling, CDT and mesh audits are Fe.
use super::fe_oracle::compile_ingot_at_level;
use fe_codegen::OptLevel;
use std::path::Path;

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
