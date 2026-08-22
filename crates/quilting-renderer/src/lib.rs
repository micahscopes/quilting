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
pub mod prepare;
pub mod texture;

use glow::HasContext;

use buffer::{
    VertexUniformBuf, WireUniformBuf, PbrUniformBuf, MatcapUniformBuf, JointMatricesBuf,
    SkinningTexture, MorphTargetTexture,
};
use shader::Programs;
use prepare::PatchPreparer;

/// High-level renderer for the quilting pipeline.
///
/// Owns the glow context, compiled shader programs, and all shared uniform buffers.
pub struct Renderer {
    gl: glow::Context,
    programs: Programs,
    vtx_ubo: VertexUniformBuf,
    wire_ubo: WireUniformBuf,
    pbr_ubo: PbrUniformBuf,
    matcap_ubo: MatcapUniformBuf,
    joint_ubo: JointMatricesBuf,
    patch_preparer: PatchPreparer,
    skinning_texture: Option<SkinningTexture>,
    morph_texture: Option<MorphTargetTexture>,
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
        let pbr_ubo = PbrUniformBuf::new(&gl)?;
        let matcap_ubo = MatcapUniformBuf::new(&gl)?;
        let joint_ubo = JointMatricesBuf::new(&gl)?;
        let patch_preparer = PatchPreparer::new(&gl)?;

        Ok(Renderer {
            gl,
            programs,
            vtx_ubo,
            wire_ubo,
            pbr_ubo,
            matcap_ubo,
            joint_ubo,
            patch_preparer,
            skinning_texture: None,
            morph_texture: None,
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
            self.gl.clear_color(0.2, 0.2, 0.3, 1.0);
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

    /// Prepare one resident patch batch from the current animation pose and
    /// entity transform. The destination is consumed directly by later draws.
    pub fn prepare_patch_batch(
        &self,
        camera: &pass::Camera,
        batch: &pass::RenderBatch<'_>,
        source_vao: glow::VertexArray,
        destination: glow::Buffer,
        byte_offset: i32,
    ) {
        let batch_camera = pass::camera_for_batch(camera, batch);
        pass::upload_batch_ubo(
            &self.gl,
            &self.vtx_ubo,
            &batch_camera,
            1,
            &batch.euclidean_model,
            &batch.euclidean_normal,
        );
        self.patch_preparer.prepare_range(
            &self.gl,
            source_vao,
            destination,
            byte_offset,
            batch.mesh.num_instances,
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

    /// Access the vertex uniform buffer (for per-batch uploads).
    pub fn vtx_ubo(&self) -> &VertexUniformBuf {
        &self.vtx_ubo
    }

    /// Access the PBR uniform buffer (for external PBR pass management).
    pub fn pbr_ubo(&self) -> &PbrUniformBuf {
        &self.pbr_ubo
    }

    /// Access the matcap uniform buffer.
    pub fn matcap_ubo(&self) -> &MatcapUniformBuf {
        &self.matcap_ubo
    }

    /// Access the joint matrices uniform buffer.
    pub fn joint_ubo(&self) -> &JointMatricesBuf {
        &self.joint_ubo
    }

    /// Replace the renderer's persistent per-vertex skinning resource.
    pub fn upload_skinning_texture(
        &mut self,
        joint_indices: &[[u16; 4]],
        joint_weights: &[[f32; 4]],
    ) -> Result<(), String> {
        let texture = SkinningTexture::new(&self.gl, joint_indices, joint_weights)?;
        texture.bind(&self.gl, shader::SKINNING_TEX_UNIT);
        if let Some(previous) = self.skinning_texture.replace(texture) {
            previous.destroy(&self.gl);
        }
        Ok(())
    }

    /// Replace the renderer's persistent morph-target delta resource.
    pub fn upload_morph_texture(
        &mut self,
        deltas: &[f32],
        num_vertices: usize,
        num_targets: usize,
    ) -> Result<(), String> {
        let texture = MorphTargetTexture::new(&self.gl, deltas, num_vertices, num_targets)?;
        texture.bind(&self.gl, shader::MORPH_TEX_UNIT);
        if let Some(previous) = self.morph_texture.replace(texture) {
            previous.destroy(&self.gl);
        }
        Ok(())
    }

    /// Re-establish animation sampler bindings after arbitrary render passes.
    pub fn bind_animation_textures(&self) {
        unsafe {
            self.gl.active_texture(glow::TEXTURE0 + shader::SKINNING_TEX_UNIT);
            self.gl.bind_texture(
                glow::TEXTURE_2D,
                self.skinning_texture.as_ref().map(|texture| texture.texture),
            );
            self.gl.active_texture(glow::TEXTURE0 + shader::MORPH_TEX_UNIT);
            self.gl.bind_texture(
                glow::TEXTURE_2D,
                self.morph_texture.as_ref().map(|texture| texture.texture),
            );
        }
    }

    /// Delete animation textures and leave their reserved sampler units empty.
    pub fn clear_animation_textures(&mut self) {
        if let Some(texture) = self.skinning_texture.take() {
            texture.destroy(&self.gl);
        }
        if let Some(texture) = self.morph_texture.take() {
            texture.destroy(&self.gl);
        }
        self.bind_animation_textures();
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        self.programs.destroy(&self.gl);
        self.vtx_ubo.destroy(&self.gl);
        self.wire_ubo.destroy(&self.gl);
        self.pbr_ubo.destroy(&self.gl);
        self.matcap_ubo.destroy(&self.gl);
        self.joint_ubo.destroy(&self.gl);
        self.patch_preparer.destroy(&self.gl);
        if let Some(texture) = self.skinning_texture.take() {
            texture.destroy(&self.gl);
        }
        if let Some(texture) = self.morph_texture.take() {
            texture.destroy(&self.gl);
        }
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
        // Compare against the mode list itself rather than a hardcoded count, so
        // adding a fragment mode doesn't strand this assertion the way `pick` did.
        let mut got: Vec<&str> = sources.iter().map(|&(name, _, _)| name).collect();
        let mut want = shader::FRAGMENT_MODES.to_vec();
        got.sort_unstable();
        want.sort_unstable();
        assert_eq!(got, want, "compiled programs should match FRAGMENT_MODES");

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
    fn vertex_shader_has_joint_ubo() {
        let sources = compiled_glsl_sources().unwrap();
        let (_name, vertex, _frag) = &sources[0];
        assert!(
            vertex.contains("JointMatrices_block_1Vertex"),
            "vertex shader should have the JointMatrices UBO block"
        );
        assert!(
            vertex.contains("_group_0_binding_2_vs"),
            "vertex shader should have the skinning texture sampler"
        );
        assert!(
            vertex.contains("_group_0_binding_3_vs"),
            "vertex shader should have the morph texture sampler"
        );
    }

    #[test]
    fn affine_conformal_maps_still_transform_normals_and_stretch() {
        let source = include_str!("../../quilting-shaders/shaders/vertex/main.wgsl");
        assert!(
            !source.contains("let is_mobius"),
            "c=0 includes rotations and signed scales, so it cannot bypass the differential"
        );
        assert!(
            source.contains("let ma0 = u.mob_a - qmul(mm0, u.mob_c)"),
            "stretch must use the full fractional-linear differential"
        );
    }

    #[test]
    fn current_pose_culling_precedes_surface_evaluation() {
        let source = include_str!("../../quilting-shaders/shaders/vertex/main.wgsl");
        let cull = source.find("if patch_outside_frustum(sp0.yzw, sp1.yzw, sp2.yzw)")
            .expect("vertex shader must cull from current posed control points");
        let evaluate = source[cull..].find("eval_mobius_qb(")
            .expect("QB evaluation must follow current-pose culling");
        assert!(evaluate > 0);
        assert_eq!(
            quilting_core::instance_layout::CONSTANT_WEIGHT_LOCATIONS,
            [4, 5, 6],
            "the flat-patch image bound must be upgraded when curved QB weights become resident",
        );
    }

    #[test]
    fn affine_model_normal_and_orientation_are_explicit() {
        let model = [
            2.0, 0.0, 0.0, 0.0,
            0.0, 3.0, 0.0, 0.0,
            0.0, 0.0, -4.0, 0.0,
            5.0, 6.0, 7.0, 1.0,
        ];
        let normal = pass::affine_normal_matrix(&model);
        assert!((normal[0] - 0.5).abs() < 1.0e-6);
        assert!((normal[5] - 1.0 / 3.0).abs() < 1.0e-6);
        assert!((normal[10] + 0.25).abs() < 1.0e-6);
        assert_eq!(pass::affine_orientation_sign(&model), -1);
    }

    #[test]
    fn canonical_atlas_permutation_controls_winding() {
        assert_eq!(pass::batch_orientation_sign(1, 1.0), 1);
        assert_eq!(pass::batch_orientation_sign(1, -1.0), -1);
        assert_eq!(pass::batch_orientation_sign(-1, 1.0), -1);
        assert_eq!(pass::batch_orientation_sign(-1, -1.0), 1);

        let source = include_str!("../../quilting-shaders/shaders/vertex/main.wgsl");
        assert!(
            source.contains("let bary = perm_bary(in.bary, perm_index)"),
            "canonical atlas barycentrics must be permuted per instance",
        );
        assert!(
            !source.contains("normal * perm_parity")
                && !source.contains("nrm * u.perm_parity")
                && !source.contains("n = n * perm_parity"),
            "permutation parity belongs in raster winding, not authored surface normals",
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
