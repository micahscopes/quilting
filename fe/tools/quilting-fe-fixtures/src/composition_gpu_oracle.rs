//! Execute authored Fe unchanged on Wasm and GPU. Rust only hosts and compares.
use common::InputDb;
use fe_codegen::{WasmCompileOptions, WebBuildOptions, WebBundle};
use hir::hir_def::HirIngot;
use salsa::Setter;

#[test]
fn composition_gpu_boundary_positions_match_wasm() {
    let (adapter, device, queue) =
        super::raster_oracle::device().expect("this agreement gate requires a real GPU execution");
    eprintln!("composition adapter: {}", adapter.get_info().name);
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../ingots/validation/composition_gpu_oracle")
        .canonicalize()
        .unwrap();
    let url = url::Url::from_directory_path(path).unwrap();
    let mut db = driver::DriverDataBase::default();
    db.compilation_settings()
        .set_profile(&mut db)
        .to("release".into());
    assert!(!driver::init_ingot(&mut db, &url));
    let top = db
        .workspace()
        .containing_ingot(&db, url)
        .unwrap()
        .root_mod(&db);
    let diagnostics = db.run_on_top_mod(top).format_diags(&db);
    assert!(diagnostics.is_empty(), "{diagnostics}");
    let package = mir::build_wasm_runtime_package_for_entries(
        &db,
        top,
        &["table_word".into(), "expected".into()],
    )
    .unwrap();
    let wasm = fe_codegen::compile_runtime_package_wasm_with_options(
        &db,
        &package,
        WasmCompileOptions::default().with_optimization(),
    )
    .unwrap()
    .bytes;
    let bundle = WebBundle::compile(&db, top, WebBuildOptions::compute("probe", None)).unwrap();
    assert_eq!(bundle.manifest.passes.len(), 1);
    let pass = &bundle.manifest.passes[0];
    let shader = &bundle
        .pass_wgsl
        .iter()
        .find(|s| s.path == pass.shader)
        .unwrap()
        .source;
    eprintln!(
        "agreement probe: Wasm {} bytes, WGSL {} bytes",
        wasm.len(),
        shader.len()
    );
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("authored Fe boundary probe"),
        source: wgpu::ShaderSource::Wgsl(shader.clone().into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Fe boundary agreement"),
        layout: None,
        module: &module,
        entry_point: Some(&pass.layout.entry_point),
        compilation_options: Default::default(),
        cache: None,
    });
    let buffers: Vec<_> = pass
        .layout
        .bindings
        .iter()
        .map(|binding| {
            assert_eq!(binding.group, 0);
            let resource = bundle
                .manifest
                .resources
                .iter()
                .find(|r| r.group == binding.group && r.binding == binding.binding);
            let bytes = resource
                .map_or(binding.span, |r| r.length.checked_mul(r.stride).unwrap())
                .max(binding.span)
                .max(4);
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&binding.name),
                size: u64::from(bytes),
                mapped_at_creation: false,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            })
        })
        .collect();
    let index = |name: &str| {
        let resource = bundle
            .manifest
            .resources
            .iter()
            .find(|r| r.name == name)
            .unwrap();
        pass.layout
            .bindings
            .iter()
            .position(|b| b.group == resource.group && b.binding == resource.binding)
            .unwrap()
    };
    let maps = index("maps");
    let captured = index("captured");
    let entries: Vec<_> = pass
        .layout
        .bindings
        .iter()
        .zip(&buffers)
        .map(|(b, buffer)| wgpu::BindGroupEntry {
            binding: b.binding,
            resource: buffer.as_entire_binding(),
        })
        .collect();
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Fe boundary probe resources"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("boundary readback"),
        size: buffers[captured].size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let engine = wasmtime::Engine::default();
    let module = wasmtime::Module::new(&engine, &wasm).unwrap();
    let mut store = wasmtime::Store::new(&engine, ());
    let instance = wasmtime::Instance::new(&mut store, &module, &[]).unwrap();
    let table = instance
        .get_typed_func::<(i32, i32, i32), f32>(&mut store, "table_word")
        .unwrap();
    let expected = instance
        .get_typed_func::<(i32, i32), (f32, f32, f32, f32)>(&mut store, "expected")
        .unwrap();
    let cases = [
        [0, 0, 0, 0],
        [8, 8, 8, 8],
        [1, 6, 6, 1],
        [0, 8, 8, 0],
        [8, 0, 0, 8],
        [0, 0, 8, 8],
        [8, 8, 0, 0],
        [0, 8, 0, 8],
        [8, 0, 8, 0],
        [0, 1, 7, 8],
        [8, 7, 1, 0],
        [0, 0, 0, 8],
        [3, 7, 2, 6],
    ];
    let mut maximum = 0.0_f32;
    for levels in cases {
        let code = levels[0] + 9 * levels[1] + 81 * levels[2] + 729 * levels[3];
        let mut bytes = Vec::new();
        for slot in 0..10 {
            for sample in 0..17 {
                bytes.extend_from_slice(
                    &table
                        .call(&mut store, (code, slot, sample))
                        .unwrap()
                        .to_le_bytes(),
                );
            }
        }
        queue.write_buffer(&buffers[maps], 0, &bytes);
        let sentinel: Vec<_> = (0..buffers[captured].size() / 4)
            .flat_map(|_| f32::NAN.to_le_bytes())
            .collect();
        queue.write_buffer(&buffers[captured], 0, &sentinel);
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut compute = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            compute.set_pipeline(&pipeline);
            compute.set_bind_group(0, &group, &[]);
            let [x, y, z] = pass.dispatch.unwrap();
            compute.dispatch_workgroups(x, y, z);
        }
        encoder.copy_buffer_to_buffer(&buffers[captured], 0, &staging, 0, staging.size());
        queue.submit(Some(encoder.finish()));
        let bytes = super::raster_oracle::readback(&device, &staging);
        let values: Vec<_> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(values.len(), 10 * 2 * 257 * 4);
        for query in 0..5140 {
            let (a, b, c, d) = expected.call(&mut store, (code, query)).unwrap();
            for (lane, want) in [a, b, c, d].into_iter().enumerate() {
                let got = values[query as usize * 4 + lane];
                let error = (got - want).abs();
                assert!(
                    got.is_finite() && error <= 2e-6,
                    "{levels:?} query {query} lane {lane}: GPU {got}, Wasm {want}"
                );
                maximum = maximum.max(error);
            }
        }
        for slot in 0..10 {
            for sample in 0..257 {
                for lane in [0, 2, 3] {
                    let forward = (slot * 514 + sample) * 4 + lane;
                    let backward = (slot * 514 + 257 + 256 - sample) * 4 + lane;
                    assert_eq!(
                        values[forward].to_bits(),
                        values[backward].to_bits(),
                        "GPU shared placement {levels:?} slot {slot} sample {sample}"
                    );
                }
            }
        }
    }
    eprintln!(
        "{} boundary queries; max GPU/Wasm error {maximum}; reversed GPU positions bit-exact",
        cases.len() * 5140
    );
}
