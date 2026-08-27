pub use naga_oil::compose::ShaderDefValue;
use naga_oil::compose::{ComposableModuleDescriptor, Composer, NagaModuleDescriptor};
use std::collections::HashMap;
use std::fmt::Write;
use std::sync::{Arc, OnceLock};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryPointStage {
    Vertex,
    Fragment,
}

/// Entry names produced by the pinned Naga WGSL writer. Naga reserves the
/// source spelling while flattening and deterministically appends `_`.
pub const LOD_PASS1_DEVICE_ENTRY_POINT: &str = "classify_lod_pass1_";
pub const LOD_PASS2_DEVICE_ENTRY_POINT: &str = "classify_lod_pass2_";
pub const VISIBILITY_COUNT_DEVICE_ENTRY_POINT: &str = "count_visible_instances";
pub const VISIBILITY_SCAN_DEVICE_ENTRY_POINT: &str = "scan_visible_batches";
pub const VISIBILITY_SCATTER_DEVICE_ENTRY_POINT: &str = "scatter_visible_instances";
pub const PATCH_PREPARE_DEVICE_ENTRY_POINT: &str = "prepare_patch_instances";

/// All WGSL shader module sources, embedded at compile time.
pub mod sources {
    // Library modules (imported by other shaders)
    pub const QUATERNION: &str = include_str!("../shaders/math/quaternion.wgsl");
    pub const QB_EVAL: &str = include_str!("../shaders/surface/qb_eval.wgsl");
    pub const PATCH_PREPARE: &str = include_str!("../shaders/surface/patch_prepare.wgsl");
    pub const PBR: &str = include_str!("../shaders/lighting/pbr.wgsl");
    pub const MATCAP: &str = include_str!("../shaders/lighting/matcap.wgsl");
    pub const DENSITY: &str = include_str!("../shaders/viz/density.wgsl");
    pub const LOD_TYPES: &str = include_str!("../shaders/compute/lod_types.wgsl");
    pub const POSE: &str = include_str!("../shaders/compute/pose.wgsl");
    pub const PATCH_PREPARE_TYPES: &str =
        include_str!("../shaders/compute/patch_prepare_types.wgsl");
    pub const PATCH_PREPARE_COMPUTE: &str = include_str!("../shaders/compute/patch_prepare.wgsl");
    pub const LOD_PASS1: &str = include_str!("../shaders/compute/lod_pass1.wgsl");
    pub const LOD_PASS2: &str = include_str!("../shaders/compute/lod_pass2.wgsl");
    pub const VISIBILITY_COMPACTION_TYPES: &str =
        include_str!("../shaders/compute/visibility_compaction_types.wgsl");
    pub const VISIBILITY_COUNT: &str = include_str!("../shaders/compute/visibility_count.wgsl");
    pub const VISIBILITY_SCAN: &str = include_str!("../shaders/compute/visibility_scan.wgsl");
    pub const VISIBILITY_SCATTER: &str = include_str!("../shaders/compute/visibility_scatter.wgsl");

    // Entry-point shaders (compiled to GLSL for WebGL2)
    pub const VERTEX_MAIN: &str = include_str!("../shaders/vertex/main.wgsl");
    pub const FRAG_MATCAP: &str = include_str!("../shaders/fragment/matcap.wgsl");
    pub const FRAG_WIRE: &str = include_str!("../shaders/fragment/wire.wgsl");
    pub const FRAG_NORMALS: &str = include_str!("../shaders/fragment/normals.wgsl");
    pub const FRAG_PBR: &str = include_str!("../shaders/fragment/pbr.wgsl");
    pub const FRAG_STRETCH: &str = include_str!("../shaders/fragment/stretch.wgsl");
    pub const FRAG_PICK: &str = include_str!("../shaders/fragment/pick.wgsl");
}

/// Exact identity of the compiler configuration and composable WGSL catalog.
///
/// Pipeline descriptor caches retain this value beside an entry module's own
/// source. Including the complete imported sources makes a catalog edit a
/// cache miss without relying on a manually bumped, collision-prone digest.
/// The dependency versions mirror this crate's resolved compiler API; changing
/// either compiler dependency must change this prefix as part of the upgrade.
pub fn compiler_catalog_revision() -> Arc<str> {
    static REVISION: OnceLock<Arc<str>> = OnceLock::new();
    Arc::clone(REVISION.get_or_init(build_compiler_catalog_revision))
}

fn build_compiler_catalog_revision() -> Arc<str> {
    const COMPILER_CONFIGURATION: &str = "quilting-shaders/catalog-v1;naga=28.0.0;naga-oil=0.21.0";
    let modules = [
        ("quilting::math::quaternion", sources::QUATERNION),
        ("quilting::surface::qb_eval", sources::QB_EVAL),
        ("quilting::surface::patch_prepare", sources::PATCH_PREPARE),
        ("quilting::lighting::pbr", sources::PBR),
        ("quilting::lighting::matcap", sources::MATCAP),
        ("quilting::viz::density", sources::DENSITY),
        ("quilting::compute::lod_types", sources::LOD_TYPES),
        ("quilting::compute::pose", sources::POSE),
        (
            "quilting::compute::patch_prepare_types",
            sources::PATCH_PREPARE_TYPES,
        ),
        (
            "quilting::compute::visibility_compaction_types",
            sources::VISIBILITY_COMPACTION_TYPES,
        ),
    ];
    let capacity = COMPILER_CONFIGURATION.len()
        + modules
            .iter()
            .map(|(path, source)| path.len() + source.len() + 48)
            .sum::<usize>();
    let mut revision = String::with_capacity(capacity);
    revision.push_str(COMPILER_CONFIGURATION);
    for (path, source) in modules {
        // Length framing keeps the identity unambiguous even when a source
        // happens to contain one of the textual separators.
        write!(revision, "\0{}:{path}\0{}:", path.len(), source.len())
            .expect("writing to a String cannot fail");
        revision.push_str(source);
    }
    revision.into()
}

/// Build a naga-oil Composer preloaded with all quilting shader modules.
pub fn create_composer() -> Result<Composer, Box<dyn std::error::Error>> {
    let mut composer = Composer::default();

    let modules = [
        ("quilting::math::quaternion", sources::QUATERNION),
        ("quilting::surface::qb_eval", sources::QB_EVAL),
        ("quilting::surface::patch_prepare", sources::PATCH_PREPARE),
        ("quilting::lighting::pbr", sources::PBR),
        ("quilting::lighting::matcap", sources::MATCAP),
        ("quilting::viz::density", sources::DENSITY),
        ("quilting::compute::lod_types", sources::LOD_TYPES),
        ("quilting::compute::pose", sources::POSE),
        (
            "quilting::compute::patch_prepare_types",
            sources::PATCH_PREPARE_TYPES,
        ),
        (
            "quilting::compute::visibility_compaction_types",
            sources::VISIBILITY_COMPACTION_TYPES,
        ),
    ];

    for (path, source) in modules {
        composer.add_composable_module(ComposableModuleDescriptor {
            source,
            file_path: path,
            ..Default::default()
        })?;
    }

    Ok(composer)
}

/// Compile a WGSL shader that imports quilting modules.
pub fn compile_shader(
    source: &str,
    shader_defs: HashMap<String, ShaderDefValue>,
) -> Result<naga::Module, Box<dyn std::error::Error>> {
    let mut composer = create_composer()?;

    let module = composer.make_naga_module(NagaModuleDescriptor {
        source,
        shader_defs,
        ..Default::default()
    })?;

    Ok(module)
}

/// Emit a naga Module as GLSL ES 300 (for WebGL2).
///
/// By default, naga emits a Y-flip + Z-remap for WebGPU coordinate conventions.
/// Pass `adjust_coordinate_space: false` to disable this for native OpenGL/WebGL
/// rendering where you manage coordinates yourself.
pub fn emit_glsl(
    module: &naga::Module,
    stage: naga::ShaderStage,
    entry_point: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    emit_glsl_with_options(module, stage, entry_point, true)
}

/// Emit a validated Naga module back to standalone WGSL.
///
/// This flattens naga-oil imports into source accepted directly by WebGPU's
/// `createShaderModule`; no compositor directives remain at the device edge.
pub fn emit_wgsl(module: &naga::Module) -> Result<String, Box<dyn std::error::Error>> {
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(module)?;
    Ok(naga::back::wgsl::write_string(
        module,
        &info,
        naga::back::wgsl::WriterFlags::empty(),
    )?)
}

/// Emit GLSL for direct OpenGL/WebGL use (no coordinate space adjustment).
///
/// Naga's default ADJUST_COORDINATE_SPACE flips Y and remaps Z for WebGPU conventions.
/// This function disables that, producing GLSL suitable for glow-based rendering
/// where gl_Position is already in standard OpenGL clip space.
pub fn emit_glsl_native(
    module: &naga::Module,
    stage: naga::ShaderStage,
    entry_point: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    emit_glsl_with_options(module, stage, entry_point, false)
}

/// Emit one descriptor-selected graphics entry point without exposing naga's
/// stage type through renderer-facing APIs.
pub fn emit_graphics_entry_glsl(
    module: &naga::Module,
    stage: EntryPointStage,
    entry_point: &str,
    adjust_coordinate_space: bool,
) -> Result<String, Box<dyn std::error::Error>> {
    let stage = match stage {
        EntryPointStage::Vertex => naga::ShaderStage::Vertex,
        EntryPointStage::Fragment => naga::ShaderStage::Fragment,
    };
    emit_glsl_with_options(module, stage, entry_point, adjust_coordinate_space)
}

fn emit_glsl_with_options(
    module: &naga::Module,
    stage: naga::ShaderStage,
    entry_point: &str,
    adjust_coordinate_space: bool,
) -> Result<String, Box<dyn std::error::Error>> {
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(module)?;

    let mut writer_flags = naga::back::glsl::WriterFlags::empty();
    if adjust_coordinate_space {
        writer_flags |= naga::back::glsl::WriterFlags::ADJUST_COORDINATE_SPACE;
    }

    let options = naga::back::glsl::Options {
        version: naga::back::glsl::Version::Embedded {
            version: 300,
            is_webgl: true,
        },
        writer_flags,
        ..Default::default()
    };

    let pipeline = naga::back::glsl::PipelineOptions {
        shader_stage: stage,
        entry_point: entry_point.to_string(),
        multiview: None,
    };

    let mut output = String::new();
    let mut writer = naga::back::glsl::Writer::new(
        &mut output,
        module,
        &info,
        &options,
        &pipeline,
        naga::proc::BoundsCheckPolicies::default(),
    )?;

    writer.write()?;

    Ok(output)
}

/// Compile the main vertex shader to GLSL ES 300.
pub fn compile_vertex_glsl() -> Result<String, Box<dyn std::error::Error>> {
    let module = compile_shader(sources::VERTEX_MAIN, HashMap::new())?;
    emit_glsl(&module, naga::ShaderStage::Vertex, "vs_main")
}

/// Compile a fragment shader to GLSL ES 300 by render mode name.
/// Supported modes: "matcap", "wire", "normals"
pub fn compile_fragment_glsl(mode: &str) -> Result<String, Box<dyn std::error::Error>> {
    let (source, entry) = match mode {
        "matcap" => (sources::FRAG_MATCAP, "fs_matcap"),
        "wire" => (sources::FRAG_WIRE, "fs_wire"),
        "normals" => (sources::FRAG_NORMALS, "fs_normals"),
        "pbr" => (sources::FRAG_PBR, "fs_pbr"),
        "stretch" => (sources::FRAG_STRETCH, "fs_stretch"),
        "pick" => (sources::FRAG_PICK, "fs_pick"),
        _ => return Err(format!("unknown fragment mode: {}", mode).into()),
    };
    let module = compile_shader(source, HashMap::new())?;
    emit_glsl(&module, naga::ShaderStage::Fragment, entry)
}

/// Compile the main vertex shader to GLSL ES 300 for native OpenGL/WebGL
/// (no coordinate space adjustment -- no Y-flip or Z-remap).
pub fn compile_vertex_glsl_native() -> Result<String, Box<dyn std::error::Error>> {
    let module = compile_shader(sources::VERTEX_MAIN, HashMap::new())?;
    emit_glsl_native(&module, naga::ShaderStage::Vertex, "vs_main")
}

/// Compile the backend-neutral prepared-patch entry point for WebGL2 transform
/// feedback.
pub fn compile_patch_prepare_glsl_native() -> Result<String, Box<dyn std::error::Error>> {
    let module = compile_shader(sources::VERTEX_MAIN, HashMap::new())?;
    emit_glsl_native(&module, naga::ShaderStage::Vertex, "prepare_patches")
}

/// Compile the one-float camera-dependent visibility entry point.
pub fn compile_patch_visibility_glsl_native() -> Result<String, Box<dyn std::error::Error>> {
    let module = compile_shader(sources::VERTEX_MAIN, HashMap::new())?;
    emit_glsl_native(
        &module,
        naga::ShaderStage::Vertex,
        "classify_patch_visibility",
    )
}

/// Compile and validate the backend-neutral first LOD classification pass.
/// The returned module is ready for a future WebGPU compute pipeline; the
/// current WebGL2 runtime continues to use its established GLSL program.
pub fn compile_lod_pass1_module() -> Result<naga::Module, Box<dyn std::error::Error>> {
    let module = compile_shader(sources::LOD_PASS1, HashMap::new())?;
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)?;
    Ok(module)
}

/// Compile and validate the backend-neutral coherence/packing LOD pass.
pub fn compile_lod_pass2_module() -> Result<naga::Module, Box<dyn std::error::Error>> {
    let module = compile_shader(sources::LOD_PASS2, HashMap::new())?;
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)?;
    Ok(module)
}

/// Standalone device WGSL for LOD pass one.
pub fn compile_lod_pass1_wgsl() -> Result<String, Box<dyn std::error::Error>> {
    emit_wgsl(&compile_lod_pass1_module()?)
}

/// Standalone device WGSL for LOD pass two.
pub fn compile_lod_pass2_wgsl() -> Result<String, Box<dyn std::error::Error>> {
    emit_wgsl(&compile_lod_pass2_module()?)
}

/// Compile and validate current-pose patch preparation for WebGPU storage.
pub fn compile_patch_prepare_compute_module() -> Result<naga::Module, Box<dyn std::error::Error>> {
    compile_validated_compute_module(sources::PATCH_PREPARE_COMPUTE)
}

/// Standalone device WGSL for current-pose patch preparation.
pub fn compile_patch_prepare_compute_wgsl() -> Result<String, Box<dyn std::error::Error>> {
    emit_wgsl(&compile_patch_prepare_compute_module()?)
}

fn compile_validated_compute_module(
    source: &str,
) -> Result<naga::Module, Box<dyn std::error::Error>> {
    let module = compile_shader(source, HashMap::new())?;
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)?;
    Ok(module)
}

/// Compile the per-batch visible-instance counting pass.
pub fn compile_visibility_count_module() -> Result<naga::Module, Box<dyn std::error::Error>> {
    compile_validated_compute_module(sources::VISIBILITY_COUNT)
}

/// Compile the deterministic batch-prefix and indirect-argument pass.
pub fn compile_visibility_scan_module() -> Result<naga::Module, Box<dyn std::error::Error>> {
    compile_validated_compute_module(sources::VISIBILITY_SCAN)
}

/// Compile the stable per-batch survivor scatter pass.
pub fn compile_visibility_scatter_module() -> Result<naga::Module, Box<dyn std::error::Error>> {
    compile_validated_compute_module(sources::VISIBILITY_SCATTER)
}

pub fn compile_visibility_count_wgsl() -> Result<String, Box<dyn std::error::Error>> {
    emit_wgsl(&compile_visibility_count_module()?)
}

pub fn compile_visibility_scan_wgsl() -> Result<String, Box<dyn std::error::Error>> {
    emit_wgsl(&compile_visibility_scan_module()?)
}

pub fn compile_visibility_scatter_wgsl() -> Result<String, Box<dyn std::error::Error>> {
    emit_wgsl(&compile_visibility_scatter_module()?)
}

/// Compile a fragment shader to GLSL ES 300 for native OpenGL/WebGL
/// (no coordinate space adjustment).
pub fn compile_fragment_glsl_native(mode: &str) -> Result<String, Box<dyn std::error::Error>> {
    let (source, entry) = match mode {
        "matcap" => (sources::FRAG_MATCAP, "fs_matcap"),
        "wire" => (sources::FRAG_WIRE, "fs_wire"),
        "normals" => (sources::FRAG_NORMALS, "fs_normals"),
        "pbr" => (sources::FRAG_PBR, "fs_pbr"),
        "stretch" => (sources::FRAG_STRETCH, "fs_stretch"),
        "pick" => (sources::FRAG_PICK, "fs_pick"),
        _ => return Err(format!("unknown fragment mode: {}", mode).into()),
    };
    let module = compile_shader(source, HashMap::new())?;
    emit_glsl_native(&module, naga::ShaderStage::Fragment, entry)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiler_catalog_identity_is_deterministic_and_exact() {
        let first = compiler_catalog_revision();
        let second = compiler_catalog_revision();
        assert_eq!(first, second);
        assert!(first.starts_with("quilting-shaders/catalog-v1;naga=28.0.0;naga-oil=0.21.0"));
        for (path, source) in [
            ("quilting::math::quaternion", sources::QUATERNION),
            ("quilting::surface::qb_eval", sources::QB_EVAL),
            ("quilting::surface::patch_prepare", sources::PATCH_PREPARE),
            ("quilting::lighting::pbr", sources::PBR),
            ("quilting::lighting::matcap", sources::MATCAP),
            ("quilting::viz::density", sources::DENSITY),
            ("quilting::compute::lod_types", sources::LOD_TYPES),
            ("quilting::compute::pose", sources::POSE),
            (
                "quilting::compute::patch_prepare_types",
                sources::PATCH_PREPARE_TYPES,
            ),
            (
                "quilting::compute::visibility_compaction_types",
                sources::VISIBILITY_COMPACTION_TYPES,
            ),
        ] {
            assert!(first.contains(path));
            assert!(first.contains(source));
        }
    }

    #[test]
    fn composer_loads_all_modules() {
        let composer = create_composer();
        assert!(composer.is_ok(), "Failed: {:?}", composer.err());
    }

    #[test]
    fn prepared_patch_storage_contract_is_thirteen_contiguous_vec4s() {
        const PROBE: &str = r#"
#import quilting::surface::patch_prepare::PreparedPatchRecord

@group(0) @binding(0) var<storage, read> source_records: array<PreparedPatchRecord>;
@group(0) @binding(1) var<storage, read_write> prepared_records: array<PreparedPatchRecord>;

@compute @workgroup_size(1)
fn probe(@builtin(global_invocation_id) invocation: vec3<u32>) {
    prepared_records[invocation.x] = source_records[invocation.x];
}
"#;

        let module = compile_shader(PROBE, HashMap::new()).expect("prepared-patch ABI compiles");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .expect("prepared-patch ABI validates");
        let mut layouter = naga::proc::Layouter::default();
        layouter
            .update(module.to_ctx())
            .expect("prepared-patch ABI lays out");
        let (handle, ty) = module
            .types
            .iter()
            .find(|(_, ty)| {
                ty.name.as_deref().is_some_and(|candidate| {
                    candidate == "PreparedPatchRecord"
                        || candidate.starts_with("PreparedPatchRecordX_naga_oil_mod_")
                })
            })
            .expect("prepared-patch record type");
        let layout = layouter[handle];
        assert_eq!(layout.size, 13 * 16);
        assert_eq!(layout.to_stride(), 13 * 16);
        let naga::TypeInner::Struct { members, span } = &ty.inner else {
            panic!("prepared-patch record is not a struct");
        };
        assert_eq!(*span, 13 * 16);
        assert_eq!(members.len(), 13);
        assert_eq!(
            members
                .iter()
                .map(|member| member.offset)
                .collect::<Vec<_>>(),
            (0..13).map(|slot| slot * 16).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn patch_prepare_compute_storage_contract_is_exact() {
        const PROBE: &str = r#"
#import quilting::compute::patch_prepare_types::{PatchTopologyRecord, PatchSubjectState, PatchPrepareDispatch}

@group(0) @binding(0) var<uniform> dispatch: PatchPrepareDispatch;
@group(0) @binding(1) var<storage, read> topology: array<PatchTopologyRecord>;
@group(0) @binding(2) var<storage, read> subjects: array<PatchSubjectState>;
@group(0) @binding(3) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(1)
fn probe(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let record = topology[invocation.x];
    output[invocation.x] = dispatch.counts.x
        + record.subject_index
        + u32(subjects[record.subject_index].model[0].x);
}
"#;
        let module = compile_shader(PROBE, HashMap::new()).expect("patch compute ABI compiles");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .expect("patch compute ABI validates");
        let mut layouter = naga::proc::Layouter::default();
        layouter
            .update(module.to_ctx())
            .expect("patch compute ABI lays out");
        for (name, expected_size, expected_offsets) in [
            ("PatchTopologyRecord", 48, &[0, 16, 32, 40, 44][..]),
            ("PatchSubjectState", 128, &[0, 64][..]),
            ("PatchPrepareDispatch", 16, &[0][..]),
        ] {
            let (handle, ty) = module
                .types
                .iter()
                .find(|(_, ty)| {
                    ty.name.as_deref().is_some_and(|candidate| {
                        candidate == name
                            || candidate.starts_with(&format!("{name}X_naga_oil_mod_"))
                    })
                })
                .unwrap_or_else(|| panic!("missing patch compute type {name}"));
            let layout = layouter[handle];
            assert_eq!(layout.size, expected_size, "{name} size");
            assert_eq!(layout.to_stride(), expected_size, "{name} stride");
            let naga::TypeInner::Struct { members, span } = &ty.inner else {
                panic!("{name} is not a struct");
            };
            assert_eq!(*span, expected_size, "{name} declared span");
            assert_eq!(
                members
                    .iter()
                    .map(|member| member.offset)
                    .collect::<Vec<_>>(),
                expected_offsets,
                "{name} member offsets",
            );
        }
    }

    #[test]
    fn lod_classifier_storage_contract_is_valid_and_exact() {
        const PROBE: &str = r#"
#import quilting::compute::lod_types::{LodFaceRecord, LodSkinningRecord, LodAdjacencyRecord, LodPass1Record, LodSubjectState, LodDispatchUniforms, pack_lod_classification}

@group(0) @binding(0) var<storage, read> faces: array<LodFaceRecord>;
@group(0) @binding(1) var<storage, read> skinning: array<LodSkinningRecord>;
@group(0) @binding(2) var<storage, read> adjacency: array<LodAdjacencyRecord>;
@group(0) @binding(3) var<storage, read_write> pass1: array<LodPass1Record>;
@group(0) @binding(4) var<storage, read> subjects: array<LodSubjectState>;
@group(0) @binding(5) var<uniform> dispatch: LodDispatchUniforms;
@group(0) @binding(6) var<storage, read_write> packed: array<u32>;

@compute @workgroup_size(1)
fn probe(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let face = faces[invocation.x];
    let skin = skinning[face.vertex_indices.x];
    let edge = adjacency[invocation.x * 3u];
    let subject = subjects[0u];
    let exponent = u32(pass1[invocation.x].exponents.x);
    let retained = dispatch.counts.x + u32(subject.conformal.w);
    packed[invocation.x] = pack_lod_classification(
        vec3<u32>(exponent, edge.neighbor_edge, skin.joint_indices.x),
        retained & 0u,
        false,
        0u,
        0u,
    );
}
"#;

        let module = compile_shader(PROBE, HashMap::new()).expect("LOD ABI probe compiles");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .expect("LOD ABI probe validates");

        let mut layouter = naga::proc::Layouter::default();
        layouter.update(module.to_ctx()).expect("LOD ABI lays out");
        for (name, expected_size, expected_stride) in [
            ("LodFaceRecord", 16, 16),
            ("LodSkinningRecord", 32, 32),
            ("LodAdjacencyRecord", 16, 16),
            ("LodPass1Record", 16, 16),
            ("LodSubjectState", 160, 160),
            ("LodDispatchUniforms", 272, 272),
        ] {
            let (handle, ty) = module
                .types
                .iter()
                .find(|(_, ty)| {
                    ty.name.as_deref().is_some_and(|candidate| {
                        candidate == name
                            || candidate.starts_with(&format!("{name}X_naga_oil_mod_"))
                    })
                })
                .unwrap_or_else(|| {
                    let names = module
                        .types
                        .iter()
                        .filter_map(|(_, ty)| ty.name.as_deref())
                        .collect::<Vec<_>>();
                    panic!("missing {name}; module types: {names:?}")
                });
            let layout = layouter[handle];
            assert_eq!(layout.size, expected_size, "{name} size");
            assert_eq!(layout.to_stride(), expected_stride, "{name} stride");
            if let naga::TypeInner::Struct { span, .. } = ty.inner {
                assert_eq!(span, expected_size, "{name} declared span");
            } else {
                panic!("{name} is not a struct");
            }
        }
    }

    #[test]
    fn compile_lod_pass1_compute_shader() {
        let module = compile_lod_pass1_module().expect("LOD pass one compiles and validates");
        let entry = module
            .entry_points
            .iter()
            .find(|entry| entry.name == "classify_lod_pass1")
            .expect("LOD pass one entry point");
        assert_eq!(entry.stage, naga::ShaderStage::Compute);
        assert_eq!(entry.workgroup_size, [64, 1, 1]);
        assert_eq!(
            module
                .global_variables
                .iter()
                .filter(|(_, variable)| variable.binding.is_some())
                .count(),
            9,
            "one uniform plus eight storage buffers stay inside WebGPU minimum limits",
        );
    }

    #[test]
    fn compile_lod_pass2_compute_shader() {
        let module = compile_lod_pass2_module().expect("LOD pass two compiles and validates");
        let entry = module
            .entry_points
            .iter()
            .find(|entry| entry.name == "classify_lod_pass2")
            .expect("LOD pass two entry point");
        assert_eq!(entry.stage, naga::ShaderStage::Compute);
        assert_eq!(entry.workgroup_size, [64, 1, 1]);
        assert_eq!(
            module
                .global_variables
                .iter()
                .filter(|(_, variable)| variable.binding.is_some())
                .count(),
            5,
        );
    }

    #[test]
    fn compile_patch_prepare_compute_shader() {
        let module = compile_patch_prepare_compute_module()
            .expect("patch preparation compute shader compiles and validates");
        let entry = module
            .entry_points
            .iter()
            .find(|entry| entry.name == "prepare_patch_instances")
            .expect("patch preparation compute entry point");
        assert_eq!(entry.stage, naga::ShaderStage::Compute);
        assert_eq!(entry.workgroup_size, [64, 1, 1]);
        assert_eq!(
            module
                .global_variables
                .iter()
                .filter(|(_, variable)| variable.binding.is_some())
                .count(),
            9,
            "one uniform plus eight storage buffers stay inside WebGPU minimum limits",
        );
    }

    #[test]
    fn visibility_compaction_storage_contract_is_valid_and_exact() {
        const PROBE: &str = r#"
#import quilting::compute::visibility_compaction_types::{VisibilityCompactionUniforms, VisibilityBatchRecord, CompactedBatchRangeRecord, IndexedIndirectArguments}

@group(0) @binding(0) var<uniform> dispatch: VisibilityCompactionUniforms;
@group(0) @binding(1) var<storage, read> batches: array<VisibilityBatchRecord>;
@group(0) @binding(2) var<storage, read_write> ranges: array<CompactedBatchRangeRecord>;
@group(0) @binding(3) var<storage, read_write> arguments: array<IndexedIndirectArguments>;

@compute @workgroup_size(1)
fn probe(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let batch = batches[invocation.x];
    ranges[invocation.x] = CompactedBatchRangeRecord(
        invocation.x,
        batch.source_first_instance,
        batch.source_instance_count,
        dispatch.counts.y,
        0u,
    );
    arguments[invocation.x] = IndexedIndirectArguments(
        batch.index_count,
        0u,
        0u,
        0,
        dispatch.counts.x,
    );
}
"#;
        let module = compile_shader(PROBE, HashMap::new()).expect("compaction ABI compiles");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .expect("compaction ABI validates");
        let mut layouter = naga::proc::Layouter::default();
        layouter
            .update(module.to_ctx())
            .expect("compaction ABI lays out");
        for (name, expected_size) in [
            ("VisibilityCompactionUniforms", 16),
            ("VisibilityBatchRecord", 16),
            ("CompactedBatchRangeRecord", 20),
            ("IndexedIndirectArguments", 20),
        ] {
            let (handle, ty) = module
                .types
                .iter()
                .find(|(_, ty)| {
                    ty.name.as_deref().is_some_and(|candidate| {
                        candidate == name
                            || candidate.starts_with(&format!("{name}X_naga_oil_mod_"))
                    })
                })
                .unwrap_or_else(|| panic!("missing compaction type {name}"));
            let layout = layouter[handle];
            assert_eq!(layout.size, expected_size, "{name} size");
            assert_eq!(layout.to_stride(), expected_size, "{name} stride");
            if let naga::TypeInner::Struct { span, .. } = ty.inner {
                assert_eq!(span, expected_size, "{name} declared span");
            } else {
                panic!("{name} is not a struct");
            }
        }
    }

    #[test]
    fn compile_visibility_compaction_compute_shaders() {
        for (source_entry, expected_workgroup, expected_bindings, module) in [
            (
                "count_visible_instances",
                [64, 1, 1],
                5,
                compile_visibility_count_module().unwrap(),
            ),
            (
                "scan_visible_batches",
                [1, 1, 1],
                5,
                compile_visibility_scan_module().unwrap(),
            ),
            (
                "scatter_visible_instances",
                [64, 1, 1],
                6,
                compile_visibility_scatter_module().unwrap(),
            ),
        ] {
            let entry = module
                .entry_points
                .iter()
                .find(|entry| entry.name == source_entry)
                .unwrap_or_else(|| panic!("missing {source_entry}"));
            assert_eq!(entry.stage, naga::ShaderStage::Compute);
            assert_eq!(entry.workgroup_size, expected_workgroup);
            assert_eq!(
                module
                    .global_variables
                    .iter()
                    .filter(|(_, variable)| variable.binding.is_some())
                    .count(),
                expected_bindings,
            );
        }
    }

    #[test]
    fn flattened_lod_compute_wgsl_is_standalone_and_reparseable() {
        for (source_entry, device_entry, source) in [
            (
                "classify_lod_pass1",
                LOD_PASS1_DEVICE_ENTRY_POINT,
                compile_lod_pass1_wgsl().unwrap(),
            ),
            (
                "classify_lod_pass2",
                LOD_PASS2_DEVICE_ENTRY_POINT,
                compile_lod_pass2_wgsl().unwrap(),
            ),
            (
                "prepare_patch_instances",
                PATCH_PREPARE_DEVICE_ENTRY_POINT,
                compile_patch_prepare_compute_wgsl().unwrap(),
            ),
        ] {
            assert!(!source.contains("#import"));
            assert!(!source.contains("#define_import_path"));
            assert!(source.contains(source_entry));
            let module = naga::front::wgsl::parse_str(&source).unwrap();
            assert_eq!(
                module
                    .entry_points
                    .iter()
                    .map(|entry| entry.name.as_str())
                    .collect::<Vec<_>>(),
                [device_entry],
            );
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::empty(),
            )
            .validate(&module)
            .unwrap();
        }
    }

    #[test]
    fn flattened_visibility_compaction_wgsl_is_standalone_and_reparseable() {
        for (source_entry, device_entry, source) in [
            (
                "count_visible_instances",
                VISIBILITY_COUNT_DEVICE_ENTRY_POINT,
                compile_visibility_count_wgsl().unwrap(),
            ),
            (
                "scan_visible_batches",
                VISIBILITY_SCAN_DEVICE_ENTRY_POINT,
                compile_visibility_scan_wgsl().unwrap(),
            ),
            (
                "scatter_visible_instances",
                VISIBILITY_SCATTER_DEVICE_ENTRY_POINT,
                compile_visibility_scatter_wgsl().unwrap(),
            ),
        ] {
            assert!(!source.contains("#import"));
            assert!(!source.contains("#define_import_path"));
            assert!(source.contains(source_entry));
            let module = naga::front::wgsl::parse_str(&source).unwrap();
            assert_eq!(
                module
                    .entry_points
                    .iter()
                    .map(|entry| entry.name.as_str())
                    .collect::<Vec<_>>(),
                [device_entry],
            );
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::empty(),
            )
            .validate(&module)
            .unwrap();
        }
    }

    #[test]
    fn compile_vertex_shader_to_glsl() {
        let glsl = compile_vertex_glsl();
        assert!(glsl.is_ok(), "vertex shader failed: {:?}", glsl.err());
        let code = glsl.unwrap();
        assert!(
            code.contains("#version 300 es"),
            "should target GLSL ES 300"
        );
        assert!(code.contains("void main()"), "should have main()");
    }

    #[test]
    fn compile_patch_prepare_shader_to_glsl() {
        let glsl = compile_patch_prepare_glsl_native();
        assert!(glsl.is_ok(), "patch preparation failed: {:?}", glsl.err());
        let code = glsl.unwrap();
        assert!(code.contains("#version 300 es"));
        assert!(code.contains("void main()"));
        for location in 0..10 {
            assert!(
                code.contains(&format!("_vs2fs_location{location}")),
                "prepared record must expose transform-feedback location {location}",
            );
        }
        assert!(
            code.contains("_group_0_binding_4_vs"),
            "patch preparation must fetch immutable source-face data",
        );
    }

    #[test]
    fn compile_patch_visibility_shader_to_glsl() {
        let glsl = compile_patch_visibility_glsl_native();
        assert!(glsl.is_ok(), "patch visibility failed: {:?}", glsl.err());
        let code = glsl.unwrap();
        assert!(code.contains("#version 300 es"));
        assert!(code.contains("void main()"));
        assert!(code.contains("_vs2fs_location0"));
        assert!(code.contains("Uniforms_block_0Vertex"));
    }

    #[test]
    fn compile_fragment_matcap_to_glsl() {
        let glsl = compile_fragment_glsl("matcap");
        assert!(glsl.is_ok(), "matcap fragment failed: {:?}", glsl.err());
        let code = glsl.unwrap();
        assert!(
            code.contains("#version 300 es"),
            "should target GLSL ES 300"
        );
    }

    #[test]
    fn compile_fragment_wire_to_glsl() {
        let glsl = compile_fragment_glsl("wire");
        assert!(glsl.is_ok(), "wire fragment failed: {:?}", glsl.err());
    }

    #[test]
    fn compile_fragment_normals_to_glsl() {
        let glsl = compile_fragment_glsl("normals");
        assert!(glsl.is_ok(), "normals fragment failed: {:?}", glsl.err());
    }

    #[test]
    fn compile_fragment_pbr_to_glsl() {
        let glsl = compile_fragment_glsl("pbr");
        assert!(glsl.is_ok(), "pbr fragment failed: {:?}", glsl.err());
        let code = glsl.unwrap();
        assert!(
            code.contains("#version 300 es"),
            "should target GLSL ES 300"
        );
        // Verify all 5 PBR texture samplers are present
        assert!(
            code.contains("sampler2D"),
            "PBR should have sampler2D uniforms"
        );
        // Verify the PBR UBO block is present
        assert!(
            code.contains("PbrUniforms"),
            "PBR should have PbrUniforms UBO"
        );
    }

    #[test]
    fn unknown_mode_returns_error() {
        let result = compile_fragment_glsl("nonexistent");
        assert!(result.is_err());
    }

    /// `qinv`'s pole guard must stay in step with `SINGULARITY_NORM_SQ` /
    /// `SINGULARITY_SENTINEL` in quilting-core's `quaternion.rs`. The CPU
    /// computes LODs and smooth normals for the geometry this shader draws, so
    /// if the two disagree about where a Möbius pole starts, the mismatch shows
    /// up exactly where the surface stretches to infinity.
    ///
    /// This crate can't import quilting-core (that would be a dependency
    /// cycle through the renderer), so the constants are asserted literally.
    /// If you change one side, change both.
    #[test]
    fn qinv_pole_guard_matches_the_cpu_constants() {
        let src = sources::QUATERNION;
        assert!(
            src.contains("d < 1e-20"),
            "qinv threshold changed; update SINGULARITY_NORM_SQ in quilting-core too"
        );
        assert!(
            src.contains("vec4<f32>(1e10, 0.0, 0.0, 0.0)"),
            "qinv sentinel changed; update SINGULARITY_SENTINEL in quilting-core too"
        );
    }
}
