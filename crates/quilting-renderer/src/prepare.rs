//! Backend-neutral patch preparation implemented with WebGL2 transform feedback.
//!
//! One point invocation consumes one source instance record and emits one
//! prepared record with current-pose control points and conservative visibility.
//! A future WebGPU backend can populate the same record from a compute pass.

use glow::HasContext;
use quilting_core::instance_layout;

use crate::shader;

const DUMMY_FRAGMENT: &str = r#"#version 300 es
precision highp float;
out vec4 color;
void main() { color = vec4(0.0); }
"#;

const PREPARED_VARYINGS: [&str; 10] = [
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
];

/// Reusable WebGL2 preparation pipeline. Batch VAOs and destination buffers
/// remain externally owned so rebuilding LOD groups does not rebuild shaders.
pub struct PatchPreparer {
    program: glow::Program,
    transform_feedback: glow::TransformFeedback,
}

impl PatchPreparer {
    pub fn new(gl: &glow::Context) -> Result<Self, String> {
        let vertex = quilting_shaders::compile_patch_prepare_glsl_native()
            .map_err(|error| format!("patch preparation GLSL: {error}"))?;
        let program = shader::create_transform_feedback_program(
            gl,
            &vertex,
            DUMMY_FRAGMENT,
            &PREPARED_VARYINGS,
        )
        .map_err(|error| format!("patch preparation program: {error}"))?;
        shader::bind_uniform_blocks(gl, program);

        let transform_feedback = unsafe {
            gl.create_transform_feedback().map_err(|error| {
                gl.delete_program(program);
                format!("patch transform feedback: {error}")
            })?
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
            gl.delete_program(self.program);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transform_feedback_record_matches_instance_stride() {
        assert_eq!(PREPARED_VARYINGS.len() * 4, instance_layout::STRIDE);
    }
}
