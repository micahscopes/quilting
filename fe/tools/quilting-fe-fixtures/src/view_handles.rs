//! Host execution only; the reusable picking algorithm is authored in Fe.
#[test]
fn shared_handle_selection_obeys_screen_radius_depth_and_identity() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../ingots/validation/view_handle_oracle");
    let wasm = super::fe_oracle::compile_ingot(&path);
    let engine = wasmtime::Engine::default();
    let module = wasmtime::Module::new(&engine, wasm).unwrap();
    let mut store = wasmtime::Store::new(&engine, ());
    let instance = wasmtime::Instance::new(&mut store, &module, &[]).unwrap();
    let single = instance
        .get_typed_func::<(f32, f32, f32, f32), i32>(&mut store, "single")
        .unwrap();
    for (input, expected) in [
        ((0.0, 0.0, 0.5, 1.0), 7),
        ((0.1, 0.0, 0.5, 1.0), 7),
        ((0.0, 0.2, 0.5, 1.0), 7),
        ((0.2, 0.0, 0.5, 1.0), -1),
        ((0.0, 0.4, 0.5, 1.0), -1),
        ((0.0, 0.0, -0.1, 1.0), -1),
        ((0.0, 0.0, 1.1, 1.0), -1),
        ((0.0, 0.0, 0.0, 0.0), -1),
        ((0.0, 0.0, 0.0, -1.0), -1),
        ((0.0, 0.0, 0.5, f32::INFINITY), -1),
        ((f32::INFINITY, 0.0, 0.5, 1.0), -1),
        ((f32::NAN, 0.0, 0.5, 1.0), -1),
    ] {
        assert_eq!(
            single.call(&mut store, input).unwrap(),
            expected,
            "{input:?}"
        );
    }
    for (name, expected) in [("ordering", 2), ("identity_tie", 3)] {
        let check = instance
            .get_typed_func::<i32, i32>(&mut store, name)
            .unwrap();
        for reversed in [0, 1] {
            assert_eq!(check.call(&mut store, reversed).unwrap(), expected);
        }
    }
    let glyph=instance.get_typed_func::<(i32,f32,f32,i32),(f32,f32,f32,f32)>(&mut store,"glyph").unwrap();
    for corner in 0..6 {
        let (x,y,z,w)=glyph.call(&mut store,(corner,1.0,2.0,1)).unwrap();
        assert!((x.abs()/w*100.0-5.0).abs()<1e-5);
        assert!((y.abs()/w*50.0-5.0).abs()<1e-5);
        assert_eq!(z,0.0);
    }
    for (z,w,visible) in [(0.0,1.0,0),(-1.0,1.0,1),(2.0,1.0,1),(0.0,-1.0,1),(0.0,f32::INFINITY,1)] {
        assert_eq!(glyph.call(&mut store,(0,z,w,visible)).unwrap(),(2.0,2.0,0.0,1.0));
    }
}
