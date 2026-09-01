use std::path::Path;
use std::sync::OnceLock;
use std::time::Duration;

use common::InputDb;
use driver::DriverDataBase;
use fe_codegen::{
    compile_runtime_package_wasm_with_options, WasmCompileOptions, WebBuildOptions, WebBundle,
};
use hir::hir_def::HirIngot;
use quilting_core::patch::QBTriPatch;
use quilting_core::quaternion::Quat;
use url::Url;
use wasmtime::{Instance, Store};

const FIXTURE: &[u8] =
    include_bytes!("../../../fixtures/classic-quilting/v1/direct-seed42-k1-1-1.cqa");
const VECTOR_LANES: usize = 10;

struct CompiledRaster {
    bundle: WebBundle,
    wasm: Vec<u8>,
}

static COMPILED: OnceLock<CompiledRaster> = OnceLock::new();

fn compiled_raster() -> &'static CompiledRaster {
    COMPILED.get_or_init(|| {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../ingots/demos/classic_quilting_fixed_raster");
        let url = Url::from_directory_path(path.canonicalize().unwrap()).unwrap();
        let mut db = DriverDataBase::default();
        assert!(
            !driver::init_ingot(&mut db, &url),
            "fixed raster ingot initialization diagnostics"
        );
        let top_mod = db
            .workspace()
            .containing_ingot(&db, url)
            .expect("fixed raster ingot")
            .root_mod(&db);
        let diagnostics = db.run_on_top_mod(top_mod).format_diags(&db);
        assert!(
            diagnostics.is_empty(),
            "unexpected fixed raster diagnostics:\n{diagnostics}"
        );
        let package = mir::build_wasm_runtime_package_for_entries(
            &db,
            top_mod,
            &["vertices".to_owned(), "shade".to_owned()],
        )
        .expect("build fixed raster Wasm runtime package");
        let wasm = compile_runtime_package_wasm_with_options(
            &db,
            &package,
            WasmCompileOptions::default().with_optimization(),
        )
        .expect("compile fixed raster Wasm")
        .bytes;
        wasmparser::validate(&wasm).expect("fixed raster Wasm validates");
        let bundle = WebBundle::compile(
            &db,
            top_mod,
            WebBuildOptions::render("shade", Some("classic-quilting-fixed-raster".to_owned())),
        )
        .expect("compile fixed authored raster bundle");
        CompiledRaster { bundle, wasm }
    })
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

fn expected_vectors() -> [[f32; VECTOR_LANES]; 3] {
    let artifact = crate::decode(FIXTURE).expect("checked smallest M0 fixture");
    let patch = curved_patch();
    let triangle = artifact.triangles[0];
    triangle.indices.map(|index| {
        let vertex = artifact.vertices[usize::try_from(index).unwrap()];
        let [_, u, v] = vertex.barycentric;
        let differential = patch.eval_differential(f64::from(u), f64::from(v));
        let normal = normal_from_tangents(differential.tangent_u, differential.tangent_v);
        [
            oracle_f32(differential.position[0]),
            oracle_f32(differential.position[1]),
            oracle_f32(differential.position[2] * 0.25),
            1.0,
            oracle_f32(differential.position[0]),
            oracle_f32(differential.position[1]),
            oracle_f32(differential.position[2]),
            oracle_f32(normal[0]),
            oracle_f32(normal[1]),
            oracle_f32(normal[2]),
        ]
    })
}

fn assert_vector_close(actual: &[f32], expected: &[f32], context: &str) {
    assert_eq!(actual.len(), expected.len());
    for (lane, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(actual.is_finite(), "{context} lane {lane} is nonfinite");
        assert!(
            (actual - expected).abs() <= 4.0e-6,
            "{context} lane {lane}: actual={actual:?}, expected={expected:?}"
        );
    }
}

#[test]
fn authored_raster_bundle_has_the_generated_draw_and_vector_interface() {
    let bundle = &compiled_raster().bundle;
    assert_eq!(bundle.manifest.passes.len(), 1);
    let pass = &bundle.manifest.passes[0];
    assert_eq!(pass.draw_vertices, Some(3));
    assert_eq!(pass.layout.bindings.as_slice(), []);
    assert_eq!(pass.layout.vertex_entry.as_deref(), Some("vertices"));
    assert_eq!(pass.layout.fragment_entry.as_deref(), Some("shade"));
    assert_eq!(bundle.pass_wgsl.len(), 1);
    assert_eq!(bundle.wgsl, bundle.pass_wgsl[0].source);

    let module = naga::front::wgsl::parse_str(&bundle.wgsl).expect("authored raster WGSL parses");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::default(),
    )
    .validate(&module)
    .expect("authored raster WGSL validates with browser capabilities");
    assert!(bundle.wgsl.contains("@vertex\nfn vertices"));
    assert!(bundle.wgsl.contains("@fragment\nfn shade"));
    for location in 0..6 {
        assert!(
            bundle.wgsl.contains(&format!("@location({location})")),
            "missing typed position/normal varying lane {location}"
        );
    }
}

fn instantiate_wasm() -> (Store<()>, Instance) {
    let engine = wasmtime::Engine::default();
    let module =
        wasmtime::Module::new(&engine, &compiled_raster().wasm).expect("load fixed raster Wasm");
    assert!(module.imports().next().is_none());
    let mut store = Store::new(&engine, ());
    let instance = Instance::new(&mut store, &module, &[]).expect("instantiate fixed raster Wasm");
    (store, instance)
}

#[test]
fn authored_raster_wasm_vectors_match_the_frozen_rust_oracle() {
    type RasterVector = (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32);

    let (mut store, instance) = instantiate_wasm();
    let vertices = instance
        .get_typed_func::<i32, RasterVector>(&mut store, "vertices")
        .expect("fixed raster vertices export");
    for (index, expected) in expected_vectors().into_iter().enumerate() {
        let actual = vertices
            .call(&mut store, i32::try_from(index).unwrap())
            .expect("execute fixed raster vertex");
        assert_vector_close(
            &[
                actual.0, actual.1, actual.2, actual.3, actual.4, actual.5, actual.6, actual.7,
                actual.8, actual.9,
            ],
            &expected,
            &format!("Wasm vertex {index}"),
        );
    }
}

fn replace_once(source: &mut String, from: &str, to: &str) {
    assert_eq!(source.matches(from).count(), 1, "expected one `{from}`");
    *source = source.replacen(from, to, 1);
}

fn wgsl_capture_shader() -> String {
    let raster = &compiled_raster().bundle.wgsl;
    let fragment = raster
        .find("\n@fragment\n")
        .expect("generated raster fragment boundary");
    let mut shader = raster[..fragment].to_owned();
    replace_once(
        &mut shader,
        "    @builtin(position) position",
        "    position",
    );
    for location in 0..6 {
        replace_once(
            &mut shader,
            &format!("    @location({location}) v{location}_"),
            &format!("    v{location}_"),
        );
    }
    replace_once(
        &mut shader,
        "@vertex\nfn vertices(@builtin(vertex_index) vertex_index: u32)",
        "fn vertices(vertex_index: u32)",
    );
    shader.push_str(
        r"

@group(0) @binding(0)
var<storage, read_write> captured: array<f32>;

@compute @workgroup_size(1)
fn capture(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let vector = vertices(invocation.x);
    let base = invocation.x * 10u;
    captured[base + 0u] = vector.position.x;
    captured[base + 1u] = vector.position.y;
    captured[base + 2u] = vector.position.z;
    captured[base + 3u] = vector.position.w;
    captured[base + 4u] = vector.v0_;
    captured[base + 5u] = vector.v1_;
    captured[base + 6u] = vector.v2_;
    captured[base + 7u] = vector.v3_;
    captured[base + 8u] = vector.v4_;
    captured[base + 9u] = vector.v5_;
}
",
    );
    shader
}

fn device() -> Option<(wgpu::Adapter, wgpu::Device, wgpu::Queue)> {
    let allow_skip = std::env::var_os("QUILTING_FE_ALLOW_GPU_SKIP").is_some();
    let instance = wgpu::Instance::default();
    let adapter = match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        force_fallback_adapter: false,
        ..Default::default()
    })) {
        Ok(adapter) => adapter,
        Err(error) if allow_skip => {
            eprintln!(
                "fixed raster WGSL execution skipped (QUILTING_FE_ALLOW_GPU_SKIP): {error:?}"
            );
            return None;
        }
        Err(error) => panic!("fixed raster WGSL execution has no adapter: {error:?}"),
    };
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: wgpu::Features::empty(),
        ..Default::default()
    }))
    .expect("request fixed raster WGSL device");
    Some((adapter, device, queue))
}

fn readback(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<u8> {
    let slice = buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(30)),
        })
        .expect("poll fixed raster WGSL readback");
    receiver
        .recv()
        .expect("receive fixed raster map result")
        .expect("map fixed raster result");
    let bytes = slice.get_mapped_range().to_vec();
    buffer.unmap();
    bytes
}

#[test]
fn authored_raster_wgsl_vectors_match_the_frozen_rust_oracle() {
    let Some((adapter, device, queue)) = device() else {
        return;
    };
    eprintln!("fixed raster WGSL adapter: {}", adapter.get_info().name);
    let shader = wgsl_capture_shader();
    let parsed = naga::front::wgsl::parse_str(&shader).expect("WGSL capture shader parses");
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::default(),
    )
    .validate(&parsed)
    .expect("WGSL capture shader validates");

    let byte_count = u64::try_from(3 * VECTOR_LANES * size_of::<f32>()).unwrap();
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fixed raster WGSL vectors"),
        size: byte_count,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fixed raster WGSL vector readback"),
        size: byte_count,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("fixed raster WGSL capture layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("fixed raster WGSL capture group"),
        layout: &layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: output.as_entire_binding(),
        }],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("fixed raster WGSL capture pipeline layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Fe-derived fixed raster WGSL capture"),
        source: wgpu::ShaderSource::Wgsl(shader.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Fe-derived fixed raster WGSL capture"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("capture"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("fixed raster WGSL capture"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fixed raster WGSL capture"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(3, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, byte_count);
    queue.submit(Some(encoder.finish()));
    let bytes = readback(&device, &staging);
    let (words, remainder) = bytes.as_chunks::<{ size_of::<f32>() }>();
    assert_eq!(remainder, &[0_u8; 0]);
    let actual = words
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect::<Vec<_>>();
    assert_eq!(actual.len(), 3 * VECTOR_LANES);
    for (index, expected) in expected_vectors().iter().enumerate() {
        let start = index * VECTOR_LANES;
        assert_vector_close(
            &actual[start..start + VECTOR_LANES],
            expected,
            &format!("WGSL vertex {index}"),
        );
    }
}
