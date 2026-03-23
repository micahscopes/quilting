//! quilting-renderer: glow-based WebGL2/OpenGL renderer for QB surfaces.
//!
//! Compiles WGSL shaders from quilting-shaders via naga -> GLSL ES 300,
//! manages GPU resources (VAOs, UBOs, textures), and provides a high-level
//! rendering API for the quilting pipeline.
//!
//! Works on both native OpenGL and WASM WebGL2 targets via glow.

pub mod shader;
pub mod buffer;
pub mod pass;
pub mod compute;

use glow::HasContext;

use buffer::{VertexUniformBuf, WireUniformBuf};
use shader::Programs;

/// High-level renderer for the quilting pipeline.
///
/// Owns the glow context, compiled shader programs, and shared uniform buffers.
pub struct Renderer {
    gl: glow::Context,
    programs: Programs,
    vtx_ubo: VertexUniformBuf,
    wire_ubo: WireUniformBuf,
    width: i32,
    height: i32,
}

impl Renderer {
    /// Initialize the renderer: compile all shaders and allocate GPU resources.
    ///
    /// The glow::Context must already be current (e.g. from a WebGL2 canvas
    /// or a native OpenGL window).
    pub fn new(gl: glow::Context) -> Result<Self, String> {
        let programs = shader::compile_programs(&gl)?;
        let vtx_ubo = VertexUniformBuf::new(&gl)?;
        let wire_ubo = WireUniformBuf::new(&gl)?;

        Ok(Renderer {
            gl,
            programs,
            vtx_ubo,
            wire_ubo,
            width: 0,
            height: 0,
        })
    }

    /// Handle viewport resize.
    pub fn resize(&mut self, width: i32, height: i32) {
        self.width = width;
        self.height = height;
        unsafe {
            self.gl.viewport(0, 0, width, height);
        }
    }

    /// Clear the framebuffer and set up GL state for a new frame.
    pub fn begin_frame(&self) {
        unsafe {
            self.gl.clear_color(0.06, 0.06, 0.14, 1.0);
            self.gl.clear(glow::COLOR_BUFFER_BIT | glow::DEPTH_BUFFER_BIT);
            self.gl.enable(glow::DEPTH_TEST);
            self.gl.enable(glow::BLEND);
            self.gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);
        }
    }

    /// Finalize the frame (currently a no-op, but reserved for future use).
    pub fn end_frame(&self) {
        // Placeholder for any end-of-frame work (e.g. swap buffers on native)
    }

    /// Render a full frame with the given mode, camera, and batches.
    pub fn render(
        &self,
        mode: pass::RenderMode,
        camera: &pass::Camera,
        batches: &[pass::RenderBatch],
    ) {
        pass::render_frame(
            &self.gl,
            &self.programs,
            mode,
            camera,
            batches,
            &self.vtx_ubo,
            &self.wire_ubo,
        );
    }

    /// Render the original mesh wireframe overlay.
    pub fn render_original_wireframe(
        &self,
        camera: &pass::Camera,
        mesh: &buffer::MeshBuffers,
    ) {
        pass::render_original_wireframe(
            &self.gl,
            &self.programs,
            camera,
            mesh,
            &self.vtx_ubo,
            &self.wire_ubo,
        );
    }

    /// Access the GL context (for creating buffers externally, etc).
    pub fn gl(&self) -> &glow::Context {
        &self.gl
    }

    /// Access compiled programs (for advanced uniform management).
    pub fn programs(&self) -> &Programs {
        &self.programs
    }

    /// Current viewport dimensions.
    pub fn viewport_size(&self) -> (i32, i32) {
        (self.width, self.height)
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        self.programs.destroy(&self.gl);
        self.vtx_ubo.destroy(&self.gl);
        self.wire_ubo.destroy(&self.gl);
    }
}

/// Convenience: compile all WGSL shaders to GLSL and return the source strings.
/// Useful for debugging or for platforms that need the GLSL source directly.
pub fn compiled_glsl_sources() -> Result<Vec<(&'static str, String, String)>, String> {
    let glsl = shader::compile_all_glsl()?;
    Ok(glsl
        .into_iter()
        .map(|(name, compiled)| (name, compiled.vertex, compiled.fragment))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glsl_compilation_succeeds() {
        // Verify that WGSL -> GLSL compilation works without a GL context.
        let sources = compiled_glsl_sources().unwrap();
        assert_eq!(sources.len(), 4, "should have 4 programs (matcap, wire, normals, pbr)");

        for (name, vertex, fragment) in &sources {
            assert!(
                vertex.contains("#version 300 es"),
                "{name} vertex should target GLSL ES 300"
            );
            assert!(
                fragment.contains("#version 300 es"),
                "{name} fragment should target GLSL ES 300"
            );
            assert!(
                vertex.contains("void main()"),
                "{name} vertex should have main()"
            );
            assert!(
                fragment.contains("void main()"),
                "{name} fragment should have main()"
            );
        }
    }

    #[test]
    fn vertex_shader_has_no_coordinate_flip() {
        // Verify that emit_glsl_native doesn't add the Y-flip/Z-remap.
        let sources = compiled_glsl_sources().unwrap();
        let (_name, vertex, _frag) = &sources[0];
        assert!(
            !vertex.contains("gl_Position.yz"),
            "vertex shader should NOT have naga coordinate space adjustment"
        );
    }

    #[test]
    fn vertex_shader_has_ubo() {
        let sources = compiled_glsl_sources().unwrap();
        let (_name, vertex, _frag) = &sources[0];
        assert!(
            vertex.contains("Uniforms_block_0Vertex"),
            "vertex shader should have the Uniforms UBO block"
        );
    }

    #[test]
    fn wire_fragment_has_ubo() {
        let sources = compiled_glsl_sources().unwrap();
        let wire = sources.iter().find(|(n, _, _)| *n == "wire").unwrap();
        assert!(
            wire.2.contains("WireUniforms_block_0Fragment"),
            "wire fragment should have the WireUniforms UBO block"
        );
    }
}
