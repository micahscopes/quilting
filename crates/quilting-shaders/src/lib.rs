use naga_oil::compose::{
    ComposableModuleDescriptor, Composer, NagaModuleDescriptor, ShaderDefValue,
};
use std::collections::HashMap;

/// All WGSL shader module sources, embedded at compile time.
pub mod sources {
    pub const QUATERNION: &str = include_str!("../shaders/math/quaternion.wgsl");
    pub const QB_EVAL: &str = include_str!("../shaders/surface/qb_eval.wgsl");
    pub const PBR: &str = include_str!("../shaders/lighting/pbr.wgsl");
    pub const MATCAP: &str = include_str!("../shaders/lighting/matcap.wgsl");
    pub const DENSITY: &str = include_str!("../shaders/viz/density.wgsl");
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
pub fn emit_glsl(
    module: &naga::Module,
    stage: naga::ShaderStage,
    entry_point: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(module)?;

    let options = naga::back::glsl::Options {
        version: naga::back::glsl::Version::Embedded {
            version: 300,
            is_webgl: true,
        },
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn composer_loads_all_modules() {
        let composer = create_composer();
        assert!(composer.is_ok(), "Failed: {:?}", composer.err());
    }
}
