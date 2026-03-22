use naga_oil::compose::{
    ComposableModuleDescriptor, Composer, NagaModuleDescriptor, ShaderDefValue,
};
use std::collections::HashMap;

/// All WGSL shader module sources, embedded at compile time.
pub mod sources {
    // Library modules (imported by other shaders)
    pub const QUATERNION: &str = include_str!("../shaders/math/quaternion.wgsl");
    pub const QB_EVAL: &str = include_str!("../shaders/surface/qb_eval.wgsl");
    pub const PBR: &str = include_str!("../shaders/lighting/pbr.wgsl");
    pub const MATCAP: &str = include_str!("../shaders/lighting/matcap.wgsl");
    pub const DENSITY: &str = include_str!("../shaders/viz/density.wgsl");

    // Entry-point shaders (compiled to GLSL for WebGL2)
    pub const VERTEX_MAIN: &str = include_str!("../shaders/vertex/main.wgsl");
    pub const FRAG_MATCAP: &str = include_str!("../shaders/fragment/matcap.wgsl");
    pub const FRAG_WIRE: &str = include_str!("../shaders/fragment/wire.wgsl");
    pub const FRAG_NORMALS: &str = include_str!("../shaders/fragment/normals.wgsl");
    pub const FRAG_PBR: &str = include_str!("../shaders/fragment/pbr.wgsl");
    pub const FRAG_PICK: &str = include_str!("../shaders/fragment/pick.wgsl");
}

/// Build a naga-oil Composer preloaded with all quilting shader modules.
pub fn create_composer() -> Result<Composer, Box<dyn std::error::Error>> {
    let mut composer = Composer::default();

    let modules = [
        ("quilting::math::quaternion", sources::QUATERNION),
        ("quilting::surface::qb_eval", sources::QB_EVAL),
        ("quilting::lighting::pbr", sources::PBR),
        ("quilting::lighting::matcap", sources::MATCAP),
        ("quilting::viz::density", sources::DENSITY),
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

/// Compile a fragment shader to GLSL ES 300 for native OpenGL/WebGL
/// (no coordinate space adjustment).
/// Supported modes: "matcap", "wire", "normals"
pub fn compile_fragment_glsl_native(mode: &str) -> Result<String, Box<dyn std::error::Error>> {
    let (source, entry) = match mode {
        "matcap" => (sources::FRAG_MATCAP, "fs_matcap"),
        "wire" => (sources::FRAG_WIRE, "fs_wire"),
        "normals" => (sources::FRAG_NORMALS, "fs_normals"),
        "pbr" => (sources::FRAG_PBR, "fs_pbr"),
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
    fn composer_loads_all_modules() {
        let composer = create_composer();
        assert!(composer.is_ok(), "Failed: {:?}", composer.err());
    }

    #[test]
    fn compile_vertex_shader_to_glsl() {
        let glsl = compile_vertex_glsl();
        assert!(glsl.is_ok(), "vertex shader failed: {:?}", glsl.err());
        let code = glsl.unwrap();
        assert!(code.contains("#version 300 es"), "should target GLSL ES 300");
        assert!(code.contains("void main()"), "should have main()");
    }

    #[test]
    fn compile_fragment_matcap_to_glsl() {
        let glsl = compile_fragment_glsl("matcap");
        assert!(glsl.is_ok(), "matcap fragment failed: {:?}", glsl.err());
        let code = glsl.unwrap();
        assert!(code.contains("#version 300 es"), "should target GLSL ES 300");
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
        assert!(code.contains("#version 300 es"), "should target GLSL ES 300");
        // Verify all 5 PBR texture samplers are present
        assert!(code.contains("sampler2D"), "PBR should have sampler2D uniforms");
        // Verify the PBR UBO block is present
        assert!(code.contains("PbrUniforms"), "PBR should have PbrUniforms UBO");
    }

    #[test]
    fn unknown_mode_returns_error() {
        let result = compile_fragment_glsl("nonexistent");
        assert!(result.is_err());
    }
}
