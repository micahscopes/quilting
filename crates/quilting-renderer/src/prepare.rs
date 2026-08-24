//! Backend-neutral patch preparation implemented with WebGL2 transform feedback.
//!
//! One point invocation consumes a compact topology record, fetches immutable
//! source-face data by ID, and emits one prepared record with current-pose
//! control points and conservative visibility. A future WebGPU backend can
//! populate the same record from a compute pass.

use glow::HasContext;
use quilting_core::instance_layout;
use quilting_core::render_pipeline::{
    GraphicsProgramDescriptor, ShaderModuleDescriptor, ShaderStage, ShaderTarget,
};
use std::sync::Arc;

use crate::shader::{
    shared_vertex_binding_entries, WebGlBindingPlan, WebGlProgramKey, WebGlProgramMemo,
};

const PATCH_PREPARE_FRAGMENT_WGSL: &str = r#"
@fragment
fn patch_prepare_fragment() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0);
}
"#;

const PREPARED_VARYINGS: [&str; 13] = [
    "_vs2fs_location0",
    "_vs2fs_location1",
    "_vs2fs_location2",
    "_vs2fs_location3",
    "_vs2fs_location4",
    "_vs2fs_location5",
    "_vs2fs_location6",
    "_vs2fs_location7",
    "_vs2fs_location8",
    "_vs2fs_location9",
    "_vs2fs_location10",
    "_vs2fs_location11",
    "_vs2fs_location12",
];

/// Reusable WebGL2 preparation pipeline. Batch VAOs and destination buffers
/// remain externally owned so rebuilding LOD groups does not rebuild shaders.
pub struct PatchPreparer {
    /// Non-owning handle retained by `WebGlProgramMemo` for the context epoch.
    program: glow::Program,
    transform_feedback: glow::TransformFeedback,
}

/// Pure shader, transform-feedback interface, and WebGL binding identity for
/// the patch-preparation pass.
pub fn patch_prepare_program_descriptor() -> Result<WebGlProgramKey, String> {
    let compiler_catalog_revision = quilting_shaders::compiler_catalog_revision();
    let vertex = ShaderModuleDescriptor::new(
        "quilting patch preparation vertex",
        quilting_shaders::sources::VERTEX_MAIN,
        Arc::clone(&compiler_catalog_revision),
        ShaderStage::Vertex,
        "prepare_patches",
        ShaderTarget::GlslEs300 { adjust_coordinate_space: false },
        Vec::new(),
    )
    .map_err(|error| error.to_string())?;
    let fragment = ShaderModuleDescriptor::new(
        "quilting patch preparation fragment",
        PATCH_PREPARE_FRAGMENT_WGSL,
        compiler_catalog_revision,
        ShaderStage::Fragment,
        "patch_prepare_fragment",
        ShaderTarget::GlslEs300 { adjust_coordinate_space: false },
        Vec::new(),
    )
    .map_err(|error| error.to_string())?;
    let program = GraphicsProgramDescriptor::new(
        vertex,
        Some(fragment),
        PREPARED_VARYINGS.iter().map(|varying| (*varying).into()).collect(),
    )
    .map_err(|error| error.to_string())?;
    let (uniform_blocks, samplers) = shared_vertex_binding_entries();
    let bindings = WebGlBindingPlan::new(uniform_blocks, samplers)?;
    Ok(WebGlProgramKey::new(program, bindings))
}

impl PatchPreparer {
    pub fn new(gl: &glow::Context, memo: &mut WebGlProgramMemo) -> Result<Self, String> {
        let descriptor = patch_prepare_program_descriptor()
            .map_err(|error| format!("patch preparation descriptor: {error}"))?;
        let program = memo.get_or_create(gl, descriptor)
            .map_err(|error| format!("patch preparation program: {error}"))?;

        let transform_feedback = unsafe {
            gl.create_transform_feedback()
                .map_err(|error| format!("patch transform feedback: {error}"))?
        };
        Ok(Self { program, transform_feedback })
    }

    /// Prepare a contiguous batch into the corresponding destination range.
    /// GPU command ordering makes the result visible to subsequent render
    /// draws without a CPU fence or readback.
    pub fn prepare_range(
        &self,
        gl: &glow::Context,
        source_vao: glow::VertexArray,
        destination: glow::Buffer,
        byte_offset: i32,
        num_patches: i32,
    ) {
        if num_patches <= 0 {
            return;
        }
        let byte_size = num_patches * instance_layout::STRIDE_BYTES as i32;
        unsafe {
            gl.enable(glow::RASTERIZER_DISCARD);
            gl.use_program(Some(self.program));
            gl.bind_vertex_array(Some(source_vao));
            // WebGL2 rejects beginTransformFeedback when its destination is
            // still present at the generic ARRAY_BUFFER binding, even though
            // the current VAO reads a distinct source buffer. VAO setup and
            // streaming uploads are both allowed to leave that binding stale.
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
            gl.bind_transform_feedback(glow::TRANSFORM_FEEDBACK, Some(self.transform_feedback));
            gl.bind_buffer_range(
                glow::TRANSFORM_FEEDBACK_BUFFER,
                0,
                Some(destination),
                byte_offset,
                byte_size,
            );
            gl.begin_transform_feedback(glow::POINTS);
            gl.draw_arrays(glow::POINTS, 0, num_patches);
            gl.end_transform_feedback();

            // Clear the binding on our object before unbinding it. Clearing
            // afterward would only mutate the default transform-feedback
            // object and keep a retired prepared buffer alive across reloads.
            gl.bind_buffer_base(glow::TRANSFORM_FEEDBACK_BUFFER, 0, None);
            gl.bind_buffer(glow::TRANSFORM_FEEDBACK_BUFFER, None);
            gl.bind_transform_feedback(glow::TRANSFORM_FEEDBACK, None);
            gl.bind_vertex_array(None);
            gl.use_program(None);
            gl.disable(glow::RASTERIZER_DISCARD);
        }
    }

    pub fn destroy(&self, gl: &glow::Context) {
        unsafe {
            gl.delete_transform_feedback(self.transform_feedback);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shader::{
        FACE_DATA_TEX_UNIT, JOINT_MATRICES_BINDING, MORPH_TEX_UNIT,
        SKINNING_TEX_UNIT, VERTEX_UNIFORMS_BINDING,
    };
    use std::collections::HashSet;

    #[test]
    fn transform_feedback_record_matches_instance_stride() {
        assert_eq!(PREPARED_VARYINGS.len() * 4, instance_layout::STRIDE);
    }

    #[test]
    fn descriptor_preserves_exact_patch_interface_and_bindings() {
        let key = patch_prepare_program_descriptor().unwrap();
        assert_eq!(key.program().vertex().source(), quilting_shaders::sources::VERTEX_MAIN);
        assert_eq!(key.program().vertex().entry_point(), "prepare_patches");
        assert_eq!(key.program().fragment().unwrap().source(), PATCH_PREPARE_FRAGMENT_WGSL);
        assert_eq!(key.program().fragment().unwrap().entry_point(), "patch_prepare_fragment");
        assert_eq!(
            key.program().transform_feedback_varyings()
                .iter().map(AsRef::as_ref).collect::<Vec<_>>(),
            PREPARED_VARYINGS
        );
        assert_eq!(
            key.bindings().uniform_blocks().iter()
                .map(|binding| (binding.name.as_ref(), binding.binding_point))
                .collect::<Vec<_>>(),
            vec![
                ("JointMatrices_block_1Vertex", JOINT_MATRICES_BINDING),
                ("Uniforms_block_0Vertex", VERTEX_UNIFORMS_BINDING),
            ]
        );
        assert_eq!(
            key.bindings().samplers().iter()
                .map(|binding| (binding.name.as_ref(), binding.texture_unit))
                .collect::<Vec<_>>(),
            vec![
                ("_group_0_binding_2_vs", SKINNING_TEX_UNIT),
                ("_group_0_binding_3_vs", MORPH_TEX_UNIT),
                ("_group_0_binding_4_vs", FACE_DATA_TEX_UNIT),
            ]
        );
    }

    #[test]
    fn emitted_vertex_exposes_every_ordered_prepared_varying() {
        let source = quilting_shaders::compile_patch_prepare_glsl_native().unwrap();
        let key = patch_prepare_program_descriptor().unwrap();
        let planned_blocks = key.bindings().uniform_blocks().iter()
            .map(|binding| binding.name.as_ref()).collect::<HashSet<_>>();
        let planned_samplers = key.bindings().samplers().iter()
            .map(|binding| binding.name.as_ref()).collect::<HashSet<_>>();
        for varying in PREPARED_VARYINGS {
            assert!(source.contains(varying), "missing transform-feedback varying {varying}");
        }
        for line in source.lines() {
            if let Some(after_uniform) = line.split("uniform ").nth(1) {
                if let Some(block) = after_uniform.split_whitespace().next()
                    .filter(|name| name.contains("_block_"))
                {
                    assert!(planned_blocks.contains(block),
                        "patch preparation has unplanned uniform block {block}");
                }
            }
            if line.contains("sampler") {
                for token in line.split_whitespace() {
                    let sampler = token.trim_end_matches(';');
                    if sampler.starts_with("_group_") {
                        assert!(planned_samplers.contains(sampler),
                            "patch preparation has unplanned sampler {sampler}");
                    }
                }
            }
        }
    }
}
