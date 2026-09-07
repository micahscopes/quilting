use super::fe_oracle::compile_ingot_at_level;
use fe_codegen::OptLevel;
use std::path::Path;

/// Algorithm baseline, NOT a language shootout: native Rust/f64 and Fe/Wasm
/// fixed-point currently have different sampling policies and output counts.
/// No audits, packing, workers, upload, or compilation inside timed regions.
#[cfg(feature = "quilting-export")]
#[test]
#[ignore = "release full triangle corpus timing; run explicitly"]
fn cpu_atlas_triangle_rust_fe_baseline() {
    use quilting_core::{sampling::{tri_patch, PatchConfig}, delaunay::triangulate_2d_constrained};
    use std::time::{Duration, Instant};
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../ingots/validation/cpu_atlas_oracle");
    let wasm = compile_ingot_at_level(&path, OptLevel::O2);
    // Default Wasmtime configuration does not instrument fuel consumption.
    let engine = wasmtime::Engine::default();
    let module = wasmtime::Module::new(&engine, &wasm).unwrap();
    assert_eq!(module.imports().count(), 0);
    let mut store = wasmtime::Store::new(&engine, ());
    let instance = wasmtime::Instance::new(&mut store, &module, &[]).unwrap();
    let reset = instance.get_typed_func::<(), ()>(&mut store, "fe_cabi_reset").unwrap();
    type Probe = (i32, i32, i32, i32, i32, i32, i32, i32, i64);
    let triangle = instance.get_typed_func::<(i32,i32,i32,i32,i32), Probe>(&mut store, "triangle").unwrap();
    let memory = instance.get_memory(&mut store, "memory").unwrap();
    let config = PatchConfig { k_candidates: 30, seed: 42 };
    let mut keys = Vec::new();
    for a in 0..=8 { for b in a..=8 { for c in b..=8 { keys.push([a,b,c]); } } }
    assert_eq!(keys.len(), 165);
    // Warm initialization outside measurement; both implementations run serially.
    triangle.call(&mut store, (2,2,2,42,1)).unwrap();
    let warm = tri_patch([4.0;3], &config);
    let _ = triangulate_2d_constrained(&warm.positions, &warm.bary);
    for round in 0..3 {
        let mut rust_sampling = Duration::ZERO;
        let mut rust_cdt = Duration::ZERO;
        let mut fe_sampling = Duration::ZERO;
        let mut fe_combined = Duration::ZERO;
        let (mut rp, mut rf, mut fp, mut ff) = (0usize,0usize,0usize,0usize);
        for &[a,b,c] in &keys {
            let start = Instant::now();
            let sample = tri_patch([(1u32<<a) as f64,(1u32<<b) as f64,(1u32<<c) as f64], &config);
            let rs = start.elapsed();
            let start = Instant::now();
            let mesh = triangulate_2d_constrained(&sample.positions, &sample.bary);
            let rt = start.elapsed();
            rust_sampling += rs;
            rust_cdt += rt;
            rp += sample.positions.len(); rf += mesh.triangles.len();
            reset.call(&mut store, ()).unwrap();
            let start = Instant::now();
            let s = triangle.call(&mut store, (a,b,c,42,0)).unwrap();
            let fs = start.elapsed();
            reset.call(&mut store, ()).unwrap();
            let start = Instant::now();
            let t = triangle.call(&mut store, (a,b,c,42,1)).unwrap();
            let ft = start.elapsed();
            assert_eq!(s.0, 0); assert_eq!((t.0,t.1), (0,0));
            assert_eq!(s.2, t.2);
            fe_sampling += fs; fe_combined += ft;
            fp += t.2 as usize; ff += t.3 as usize;
            eprintln!("paired-tile round={round} key={a},{b},{c} rust_points={} rust_faces={} rust_sample_us={} rust_cdt_us={} fe_points={} fe_faces={} fe_sample_us={} fe_combined_us={}",
                sample.positions.len(),mesh.triangles.len(),rs.as_micros(),rt.as_micros(),t.2,t.3,fs.as_micros(),ft.as_micros());
        }
        eprintln!("paired-summary round={round} keys=165 rust_points={rp} rust_faces={rf} rust_sampling={rust_sampling:?} rust_cdt={rust_cdt:?} rust_total={:?} fe_points={fp} fe_faces={ff} fe_sampling={fe_sampling:?} fe_combined={fe_combined:?} fe_memory={}", rust_sampling+rust_cdt,memory.data_size(&store));
    }
}

