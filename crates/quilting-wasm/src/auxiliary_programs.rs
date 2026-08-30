//! Pure descriptors for lazily-created fullscreen WebGL auxiliary programs.

use quilting_core::render_pipeline::{
    GraphicsProgramDescriptor, ShaderModuleDescriptor, ShaderStage, ShaderTarget,
};
use quilting_renderer::shader::{
    WebGlBindingPlan, WebGlBindingSite, WebGlOpaqueBindingKind, WebGlProgramKey,
    WebGlSamplerBinding, WebGlUniformBlockBinding,
};
use std::sync::Arc;

pub(crate) const POST_PROCESS_UNIFORMS_BINDING: u32 = 3;

const FULLSCREEN_VERTEX_WGSL: &str = r#"
struct FullscreenOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn fullscreen_vertex(@builtin(vertex_index) vertex_index: u32) -> FullscreenOutput {
    let x = f32((vertex_index & 1u) << 2u) - 1.0;
    let y = f32((vertex_index & 2u) << 1u) - 1.0;
    return FullscreenOutput(
        vec4<f32>(x, y, 0.0, 1.0),
        vec2<f32>(x, y) * 0.5 + vec2<f32>(0.5),
    );
}
"#;

const BLUR_FRAGMENT_WGSL: &str = r#"
struct PostProcessUniforms {
    value: vec4<f32>,
}

@group(0) @binding(0)
var<uniform> post_process: PostProcessUniforms;
@group(0) @binding(1)
var source_texture: texture_2d<f32>;
@group(0) @binding(2)
var source_sampler: sampler;

struct FullscreenInput {
    @location(0) uv: vec2<f32>,
}

@fragment
fn blur_fragment(in: FullscreenInput) -> @location(0) vec4<f32> {
    let direction = post_process.value.xy;
    var color = textureSampleLevel(source_texture, source_sampler, in.uv, 0.0).rgb * 0.227027;
    color += textureSampleLevel(source_texture, source_sampler, in.uv + direction * 1.384615, 0.0).rgb * 0.316216;
    color += textureSampleLevel(source_texture, source_sampler, in.uv - direction * 1.384615, 0.0).rgb * 0.316216;
    color += textureSampleLevel(source_texture, source_sampler, in.uv + direction * 3.230769, 0.0).rgb * 0.070270;
    color += textureSampleLevel(source_texture, source_sampler, in.uv - direction * 3.230769, 0.0).rgb * 0.070270;
    return vec4<f32>(color, 1.0);
}
"#;

const HIGHLIGHT_FRAGMENT_WGSL: &str = r#"
struct PostProcessUniforms {
    value: vec4<f32>,
}

@group(0) @binding(0)
var<uniform> post_process: PostProcessUniforms;
@group(0) @binding(1)
var source_texture: texture_2d<f32>;
@group(0) @binding(2)
var source_sampler: sampler;

struct FullscreenInput {
    @location(0) uv: vec2<f32>,
}

@fragment
fn highlight_fragment(in: FullscreenInput) -> @location(0) vec4<f32> {
    let pixel = textureSample(source_texture, source_sampler, in.uv);
    if pixel.a < 0.5 {
        discard;
    }
    let target_color = post_process.value.xyz;
    if abs(pixel.r - target_color.r) > 0.003
        || abs(pixel.g - target_color.g) > 0.003
        || abs(pixel.b - target_color.b) > 0.003 {
        discard;
    }
    return vec4<f32>(0.0, 1.0, 1.0, 0.5);
}
"#;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AuxiliaryProgram {
    Blur,
    Highlight,
}

pub(crate) fn auxiliary_program_descriptor(
    kind: AuxiliaryProgram,
) -> Result<WebGlProgramKey, String> {
    let compiler_catalog_revision = quilting_shaders::compiler_catalog_revision();
    let vertex = ShaderModuleDescriptor::new(
        "hyperscope fullscreen vertex",
        FULLSCREEN_VERTEX_WGSL,
        Arc::clone(&compiler_catalog_revision),
        ShaderStage::Vertex,
        "fullscreen_vertex",
        ShaderTarget::GlslEs300 {
            adjust_coordinate_space: false,
        },
        Vec::new(),
    )
    .map_err(|error| error.to_string())?;
    let (label, source, entry_point) = match kind {
        AuxiliaryProgram::Blur => (
            "hyperscope transmission blur fragment",
            BLUR_FRAGMENT_WGSL,
            "blur_fragment",
        ),
        AuxiliaryProgram::Highlight => (
            "hyperscope selection highlight fragment",
            HIGHLIGHT_FRAGMENT_WGSL,
            "highlight_fragment",
        ),
    };
    let fragment = ShaderModuleDescriptor::new(
        label,
        source,
        compiler_catalog_revision,
        ShaderStage::Fragment,
        entry_point,
        ShaderTarget::GlslEs300 {
            adjust_coordinate_space: false,
        },
        Vec::new(),
    )
    .map_err(|error| error.to_string())?;
    let program = GraphicsProgramDescriptor::new(vertex, Some(fragment), Vec::new())
        .map_err(|error| error.to_string())?;
    let bindings = WebGlBindingPlan::new(
        vec![WebGlUniformBlockBinding {
            name: "PostProcessUniforms_block_0Fragment".into(),
            binding_point: POST_PROCESS_UNIFORMS_BINDING,
            source_name: "post_process".into(),
            source: WebGlBindingSite::new(0, 0, ShaderStage::Fragment),
        }],
        vec![
            WebGlSamplerBinding {
                name: "_group_0_binding_1_fs".into(),
                texture_unit: 0,
                source_name: "source_texture".into(),
                source: WebGlBindingSite::new(0, 1, ShaderStage::Fragment),
                source_kind: WebGlOpaqueBindingKind::SampledTexture,
            },
            WebGlSamplerBinding {
                name: "_group_0_binding_2_fs".into(),
                texture_unit: 0,
                source_name: "source_sampler".into(),
                source: WebGlBindingSite::new(0, 2, ShaderStage::Fragment),
                source_kind: WebGlOpaqueBindingKind::Sampler,
            },
        ],
    )?;
    Ok(WebGlProgramKey::new(program, bindings))
}

#[cfg(test)]
mod tests {
    use super::*;
    use quilting_renderer::prepare::patch_prepare_program_descriptor;
    use quilting_renderer::shader::primary_program_descriptors;
    use std::collections::{HashMap, HashSet};

    fn lower(descriptor: &ShaderModuleDescriptor) -> String {
        let module = quilting_shaders::compile_shader(descriptor.source(), HashMap::new()).unwrap();
        let stage = match descriptor.stage() {
            ShaderStage::Vertex => quilting_shaders::EntryPointStage::Vertex,
            ShaderStage::Fragment => quilting_shaders::EntryPointStage::Fragment,
            ShaderStage::Compute => panic!("auxiliary programs cannot contain compute shaders"),
        };
        quilting_shaders::emit_graphics_entry_glsl(&module, stage, descriptor.entry_point(), false)
            .unwrap()
    }

    #[test]
    fn auxiliary_descriptors_share_one_exact_fullscreen_vertex() {
        let blur = auxiliary_program_descriptor(AuxiliaryProgram::Blur).unwrap();
        let highlight = auxiliary_program_descriptor(AuxiliaryProgram::Highlight).unwrap();
        assert_eq!(blur.program().vertex(), highlight.program().vertex());
        assert_eq!(blur.program().vertex().source(), FULLSCREEN_VERTEX_WGSL);
        assert_eq!(blur.program().vertex().entry_point(), "fullscreen_vertex");
        assert_ne!(blur.program().fragment(), highlight.program().fragment());
        assert!(blur.program().transform_feedback_varyings().is_empty());
        assert_eq!(blur.bindings(), highlight.bindings());
        assert_eq!(
            blur.bindings()
                .uniform_blocks()
                .iter()
                .map(|binding| (binding.name.as_ref(), binding.binding_point))
                .collect::<Vec<_>>(),
            vec![(
                "PostProcessUniforms_block_0Fragment",
                POST_PROCESS_UNIFORMS_BINDING
            )]
        );
        assert_eq!(
            blur.bindings()
                .samplers()
                .iter()
                .map(|binding| (binding.name.as_ref(), binding.texture_unit))
                .collect::<Vec<_>>(),
            vec![("_group_0_binding_1_fs", 0), ("_group_0_binding_2_fs", 0)]
        );
    }

    #[test]
    fn emitted_auxiliary_interfaces_are_covered_by_exact_binding_plans() {
        for kind in [AuxiliaryProgram::Blur, AuxiliaryProgram::Highlight] {
            let key = auxiliary_program_descriptor(kind).unwrap();
            let planned_blocks = key
                .bindings()
                .uniform_blocks()
                .iter()
                .map(|binding| binding.name.as_ref())
                .collect::<HashSet<_>>();
            let planned_samplers = key
                .bindings()
                .samplers()
                .iter()
                .map(|binding| binding.name.as_ref())
                .collect::<HashSet<_>>();
            for descriptor in [key.program().vertex(), key.program().fragment().unwrap()] {
                let source = lower(descriptor);
                assert!(source.contains("#version 300 es"));
                assert!(source.contains("void main()"));
                let module = quilting_shaders::compile_shader(
                    descriptor.source(),
                    HashMap::new(),
                )
                .unwrap();
                let stage = match descriptor.stage() {
                    ShaderStage::Vertex => quilting_shaders::EntryPointStage::Vertex,
                    ShaderStage::Fragment => quilting_shaders::EntryPointStage::Fragment,
                    ShaderStage::Compute => unreachable!(),
                };
                let reflected = quilting_shaders::reflect_graphics_entry_bindings(
                    &module,
                    stage,
                    descriptor.entry_point(),
                )
                .unwrap();
                assert_eq!(
                    key.bindings()
                        .source_bindings_for_stage(descriptor.stage()),
                    reflected,
                    "{kind:?} {:?}",
                    descriptor.stage()
                );
                for line in source.lines() {
                    if let Some(after_uniform) = line.split("uniform ").nth(1) {
                        if let Some(block) = after_uniform
                            .split_whitespace()
                            .next()
                            .filter(|name| name.contains("_block_"))
                        {
                            assert!(
                                planned_blocks.contains(block),
                                "{kind:?} has unplanned uniform block {block}"
                            );
                        }
                    }
                    if line.contains("sampler") {
                        for token in line.split_whitespace() {
                            let sampler = token.trim_end_matches(';');
                            if sampler.starts_with("_group_") {
                                assert!(
                                    planned_samplers.contains(sampler),
                                    "{kind:?} has unplanned sampler {sampler}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn wgsl_preserves_fullscreen_blur_and_highlight_behavior() {
        assert!(FULLSCREEN_VERTEX_WGSL.contains("(vertex_index & 1u) << 2u"));
        assert!(FULLSCREEN_VERTEX_WGSL.contains("(vertex_index & 2u) << 1u"));
        for (offset, weight) in [
            ("0.0", "0.227027"),
            ("1.384615", "0.316216"),
            ("3.230769", "0.070270"),
        ] {
            assert!(BLUR_FRAGMENT_WGSL.contains(offset));
            assert!(BLUR_FRAGMENT_WGSL.contains(weight));
        }
        assert_eq!(BLUR_FRAGMENT_WGSL.matches("textureSampleLevel").count(), 5);
        assert!(HIGHLIGHT_FRAGMENT_WGSL.contains("pixel.a < 0.5"));
        assert_eq!(HIGHLIGHT_FRAGMENT_WGSL.matches("> 0.003").count(), 3);
        assert!(HIGHLIGHT_FRAGMENT_WGSL.contains("vec4<f32>(0.0, 1.0, 1.0, 0.5)"));
        assert_eq!(HIGHLIGHT_FRAGMENT_WGSL.matches("discard;").count(), 2);
    }

    #[test]
    fn complete_cold_catalog_has_expected_memo_shape() {
        let mut programs = primary_program_descriptors()
            .unwrap()
            .into_iter()
            .map(|(_, key)| key)
            .collect::<Vec<_>>();
        programs.push(patch_prepare_program_descriptor().unwrap());
        programs.push(auxiliary_program_descriptor(AuxiliaryProgram::Blur).unwrap());
        programs.push(auxiliary_program_descriptor(AuxiliaryProgram::Highlight).unwrap());
        let shader_requests = programs
            .iter()
            .map(|key| 1 + usize::from(key.program().fragment().is_some()))
            .sum::<usize>();
        let shaders = programs
            .iter()
            .flat_map(|key| std::iter::once(key.program().vertex()).chain(key.program().fragment()))
            .cloned()
            .collect::<HashSet<_>>();
        let unique_programs = programs.into_iter().collect::<HashSet<_>>();
        assert_eq!(shader_requests, 18);
        assert_eq!(shaders.len(), 12);
        assert_eq!(shader_requests - shaders.len(), 6);
        assert_eq!(unique_programs.len(), 9);
    }
}
