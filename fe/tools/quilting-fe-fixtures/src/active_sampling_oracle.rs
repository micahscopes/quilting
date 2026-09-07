//! Executes authored Fe sampling through Wasmtime. No Rust sampling substitute.
use super::fe_oracle::compile_ingot_at_level;
use fe_codegen::OptLevel;
use std::path::Path;

#[test]
fn active_index_wasm_matches_exhaustive_streams() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../ingots/validation/active_sampling_oracle");
    let wasm = compile_ingot_at_level(&path, OptLevel::O2);
    eprintln!("index comparison module: {} Wasm bytes", wasm.len());
    let mut config = wasmtime::Config::new();
    config.consume_fuel(true);
    let engine = wasmtime::Engine::new(&config).unwrap();
    let module = wasmtime::Module::new(&engine, &wasm).unwrap();
    assert!(module.imports().next().is_none());
    let mut store = wasmtime::Store::new(&engine, ());
    let instance = wasmtime::Instance::new(&mut store, &module, &[]).unwrap();
    type Comparison = (i32, i32, i64, i64, i64, i32, i32);
    let triangle = instance.get_typed_func::<(i32, i32, i32, i32), Comparison>(&mut store, "compare_triangle").unwrap();
    for key in [(0, 0, 8, 42), (0, 0, 8, 7), (4, 4, 4, 42), (0, 0, 4, 42)] {
        store.set_fuel(20_000_000_000).unwrap();
        let r = triangle.call(&mut store, key).unwrap();
        eprintln!("triangle index {key:?}: {r:?}");
        assert_eq!(r.5, 0, "complete point streams must match");
        assert_eq!(r.6, 2, "both samplers must complete, not exhaust capacity/budget");
    }
    let square = instance.get_typed_func::<(i32, i32, i32, i32, i32), Comparison>(&mut store, "compare_square").unwrap();
    for key in [(0, 0, 0, 8, 42), (0, 0, 0, 8, 7), (4, 4, 4, 4, 42)] {
        store.set_fuel(20_000_000_000).unwrap();
        let r = square.call(&mut store, key).unwrap();
        eprintln!("square index {key:?}: {r:?}");
        assert_eq!(r.5, 0);
        assert_eq!(r.6, 2);
    }
    // Separate uninstrumented realization after bounded correctness gates.
    // Includes Fe scratch initialization, excludes compilation/validation/IO.
    let timing_engine = wasmtime::Engine::default();
    let timing_module = wasmtime::Module::new(&timing_engine, &wasm).unwrap();
    let mut timing_store = wasmtime::Store::new(&timing_engine, ());
    let timing_instance = wasmtime::Instance::new(&mut timing_store, &timing_module, &[]).unwrap();
    type Work = (i32, i32, i64, i64, i32);
    let tri = timing_instance.get_typed_func::<(i32, i32, i32, i32), Work>(&mut timing_store, "measure_triangle").unwrap();
    let quad = timing_instance.get_typed_func::<(i32, i32, i32, i32, i32), Work>(&mut timing_store, "measure_square").unwrap();
    for indexed in [0, 1] {
        for key in [(0, 0, 8), (4, 4, 4)] {
            let args = (key.0, key.1, key.2, indexed);
            let expected = tri.call(&mut timing_store, args).unwrap();
            assert_eq!(expected.4, 1);
            let mut durations = Vec::new();
            for _ in 0..5 {
                let start = std::time::Instant::now();
                assert_eq!(tri.call(&mut timing_store, args).unwrap(), expected);
                durations.push(start.elapsed());
            }
            durations.sort();
            eprintln!("SAMPLING triangle={key:?} indexed={indexed} median={:?} work={expected:?}", durations[2]);
        }
        for key in [(0, 0, 0, 8), (4, 4, 4, 4)] {
            let args = (key.0, key.1, key.2, key.3, indexed);
            let expected = quad.call(&mut timing_store, args).unwrap();
            assert_eq!(expected.4, 1);
            let mut durations = Vec::new();
            for _ in 0..5 {
                let start = std::time::Instant::now();
                assert_eq!(quad.call(&mut timing_store, args).unwrap(), expected);
                durations.push(start.elapsed());
            }
            durations.sort();
            eprintln!("SAMPLING square={key:?} indexed={indexed} median={:?} work={expected:?}", durations[2]);
        }
    }
}

#[test]
fn active_sampling_wasm_executes_mixed_lod8_domains() {
    run_probe(OptLevel::O2);
}

#[test]
fn active_sampling_wasm_unoptimized_diagnostic() {
    run_probe(OptLevel::O0);
}

fn run_probe(opt: OptLevel) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../ingots/validation/active_sampling_oracle");
    let start = std::time::Instant::now();
    let wasm = compile_ingot_at_level(&path, opt);
    eprintln!("Fe sampling probe: {} Wasm bytes, compilation {:?}", wasm.len(), start.elapsed());
    // A compiler/runtime regression must fail with a bounded receipt, not
    // monopolize the host. These instrumented times are not benchmarks.
    let mut config = wasmtime::Config::new();
    config.consume_fuel(true);
    let engine = wasmtime::Engine::new(&config).unwrap();
    let module = wasmtime::Module::new(&engine, wasm).unwrap();
    assert!(module.imports().next().is_none(), "sampler must execute wholly in Fe/Wasm");
    let mut store = wasmtime::Store::new(&engine, ());
    store.set_fuel(1_000_000_000).unwrap();
    let instance = wasmtime::Instance::new(&mut store, &module, &[]).unwrap();
    type Receipt = (i32, i32, i32, i32, i64, i32, i32);
    let bounded = instance.get_typed_func::<(i32, i32, i32, i32, i32, i32), Receipt>(&mut store, "triangle_budget").unwrap();
    for budget in [0, 1, 30, 300, 1000, 3000, 10000, 100000] {
        store.set_fuel(1_000_000_000).unwrap();
        let r = bounded.call(&mut store, (0, 0, 4, 42, 256, budget)).unwrap();
        eprintln!("triangle [0,0,4] budget={budget}: {r:?}, fuel={}", 1_000_000_000 - store.get_fuel().unwrap());
        assert!(r.2 <= budget);
        assert_eq!(r.6, 0);
    }
    for budget in [0, 30, 300, 3000, 10000] {
        store.set_fuel(10_000_000_000).unwrap();
        let r = bounded.call(&mut store, (0, 0, 8, 42, 256, budget)).unwrap();
        eprintln!("triangle [0,0,8] budget={budget}: {r:?}, fuel={}", 10_000_000_000 - store.get_fuel().unwrap());
        assert!(r.2 <= budget);
        assert_eq!(r.6, 0);
    }
    store.set_fuel(1_000_000_000).unwrap();
    let triangle = instance.get_typed_func::<(i32, i32, i32, i32, i32), Receipt>(&mut store, "triangle").unwrap();
    for key in [(0, 0, 4), (0, 0, 8), (2, 2, 2)] {
        store.set_fuel(10_000_000_000).unwrap();
        let start = std::time::Instant::now();
        let r = triangle.call(&mut store, (key.0, key.1, key.2, 42, 256)).unwrap();
        eprintln!("triangle {key:?}: {r:?}, sampling+validation {:?}", start.elapsed());
        assert_eq!(r.5, 1, "must finish exploration, not exhaust storage/budget");
        assert_eq!(r.6, 0, "boundary and all-pairs spacing violations");
        assert!(r.0 > r.1, "must reach interior, not merely retain boundary seeds");
        store.set_fuel(10_000_000_000).unwrap();
        assert_eq!(r, triangle.call(&mut store, (key.0, key.1, key.2, 42, 256)).unwrap(), "deterministic replay");
    }
    let square = instance.get_typed_func::<(i32, i32, i32, i32, i32, i32), Receipt>(&mut store, "square").unwrap();
    for key in [(0, 0, 0, 8), (2, 2, 2, 2)] {
        store.set_fuel(10_000_000_000).unwrap();
        let r = square.call(&mut store, (key.0, key.1, key.2, key.3, 42, 256)).unwrap();
        eprintln!("square {key:?}: {r:?}");
        assert_eq!(r.5, 1);
        assert_eq!(r.6, 0);
        assert!(r.0 > r.1);
    }
}
