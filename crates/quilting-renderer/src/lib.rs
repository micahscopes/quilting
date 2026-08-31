//! quilting-renderer: glow-based WebGL2/OpenGL renderer for QB surfaces.
//!
//! Compiles WGSL shaders from quilting-shaders via naga -> GLSL ES 300,
//! manages GPU resources (VAOs, UBOs, textures), and provides a high-level
//! rendering API for the quilting pipeline.
//!
//! Works on both native OpenGL and WASM WebGL2 targets via glow.

pub mod shader;
pub mod memo;
pub mod buffer;
pub mod pass;
pub mod compute;
pub mod prepare;
pub mod texture;

use glow::HasContext;
use quilting_core::render::DEFAULT_RENDER_CLEAR_COLOR;

use buffer::{
    VertexUniformBuf, WireUniformBuf, PbrUniformBuf, MatcapUniformBuf, JointMatricesBuf,
    FaceDataTexture, SkinningTexture, MorphTargetTexture, SuppressedFaceTexture,
};
use shader::{Programs, WebGlProgramKey, WebGlProgramMemo, WebGlProgramMemoDiagnostics};
use prepare::{PatchPreparer, PatchVisibilityClassifier};

/// High-level renderer for the quilting pipeline.
///
/// Owns the glow context, compiled shader programs, and all shared uniform buffers.
pub struct Renderer {
    gl: glow::Context,
    /// Sole owner of descriptor-lowered shaders and programs for this epoch.
    program_memo: WebGlProgramMemo,
    /// Non-owning convenience view into `program_memo`.
    programs: Programs,
    vtx_ubo: VertexUniformBuf,
    wire_ubo: WireUniformBuf,
    pbr_ubo: PbrUniformBuf,
    matcap_ubo: MatcapUniformBuf,
    joint_ubo: JointMatricesBuf,
    patch_preparer: PatchPreparer,
    patch_visibility: PatchVisibilityClassifier,
    face_data_texture: Option<FaceDataTexture>,
    suppressed_face_texture: SuppressedFaceTexture,
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
        // One Renderer is one immutable WebGL context epoch. Context restore
        // rebuilds the complete Renderer so no non-owning handle view can
        // survive an epoch change.
        let mut program_memo = WebGlProgramMemo::new(0);
        let mut programs: Option<Programs> = None;
        let mut vtx_ubo: Option<VertexUniformBuf> = None;
        let mut wire_ubo: Option<WireUniformBuf> = None;
        let mut pbr_ubo: Option<PbrUniformBuf> = None;
        let mut matcap_ubo: Option<MatcapUniformBuf> = None;
        let mut joint_ubo: Option<JointMatricesBuf> = None;
        let mut patch_preparer: Option<PatchPreparer> = None;
        let mut patch_visibility: Option<PatchVisibilityClassifier> = None;
        let mut suppressed_face_texture: Option<SuppressedFaceTexture> = None;

        let construction = (|| -> Result<(), String> {
            programs = Some(shader::compile_programs(&gl, &mut program_memo)?);
            vtx_ubo = Some(VertexUniformBuf::new(&gl)?);
            wire_ubo = Some(WireUniformBuf::new(&gl)?);
            pbr_ubo = Some(PbrUniformBuf::new(&gl)?);
            matcap_ubo = Some(MatcapUniformBuf::new(&gl)?);
            joint_ubo = Some(JointMatricesBuf::new(&gl)?);
            patch_preparer = Some(PatchPreparer::new(&gl, &mut program_memo)?);
            patch_visibility = Some(PatchVisibilityClassifier::new(&gl, &mut program_memo)?);
            suppressed_face_texture = Some(SuppressedFaceTexture::new(&gl, 0)?);
            Ok(())
        })();

        if let Err(error) = construction {
            if let Some(resource) = suppressed_face_texture {
                resource.destroy(&gl);
            }
            if let Some(resource) = patch_visibility {
                resource.destroy(&gl);
            }
            if let Some(resource) = patch_preparer {
                resource.destroy(&gl);
            }
            if let Some(resource) = joint_ubo {
                resource.destroy(&gl);
            }
            if let Some(resource) = matcap_ubo {
                resource.destroy(&gl);
            }
            if let Some(resource) = pbr_ubo {
                resource.destroy(&gl);
            }
            if let Some(resource) = wire_ubo {
                resource.destroy(&gl);
            }
            if let Some(resource) = vtx_ubo {
                resource.destroy(&gl);
            }
            program_memo.destroy(&gl);
            return Err(error);
        }

        Ok(Renderer {
            gl,
            program_memo,
            programs: programs.expect("successful construction resolved primary programs"),
            vtx_ubo: vtx_ubo.expect("successful construction allocated vertex UBO"),
            wire_ubo: wire_ubo.expect("successful construction allocated wire UBO"),
            pbr_ubo: pbr_ubo.expect("successful construction allocated PBR UBO"),
            matcap_ubo: matcap_ubo.expect("successful construction allocated matcap UBO"),
            joint_ubo: joint_ubo.expect("successful construction allocated joint UBO"),
            patch_preparer: patch_preparer
                .expect("successful construction allocated patch preparer"),
            patch_visibility: patch_visibility
                .expect("successful construction allocated patch visibility classifier"),
            face_data_texture: None,
            suppressed_face_texture: suppressed_face_texture
                .expect("successful construction allocated suppressed-face texture"),
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
        let [red, green, blue, alpha] = DEFAULT_RENDER_CLEAR_COLOR;
        unsafe {
            self.gl.clear_color(red, green, blue, alpha);
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
        style: quilting_core::render::RenderStyle,
        camera: &pass::Camera,
        batches: &[pass::RenderBatch],
    ) -> quilting_core::render::RenderSubmissionStats {
        pass::render_frame(
            &self.gl,
            &self.programs,
            style,
            camera,
            batches,
            &self.vtx_ubo,
            &self.wire_ubo,
        )
    }

    /// Render a validated backend-neutral diagnostic command stream against
    /// the retained WebGL resources. The legacy [`Self::render`] path remains
    /// available for atomic fallback until browser image parity is sustained.
    pub fn render_diagnostic_execution(
        &self,
        execution: quilting_core::render::RenderExecution<'_, '_>,
        camera: &pass::Camera,
        batches: &[pass::RenderBatch],
    ) -> Result<quilting_core::render::RenderSubmissionStats, pass::WebGlRenderExecutionError> {
        pass::render_diagnostic_execution(
            &self.gl,
            &self.programs,
            execution,
            camera,
            batches,
            &self.vtx_ubo,
            &self.wire_ubo,
        )
    }

    /// Prepare one resident patch batch from the current animation pose and
    /// entity transform. The destination is consumed directly by later draws.
    pub fn prepare_patch_batch(
        &self,
        camera: &pass::Camera,
        batch: &pass::RenderBatch,
        source_vao: glow::VertexArray,
        destination: glow::Buffer,
        byte_offset: i32,
    ) {
        let batch_camera = pass::camera_for_batch(camera, batch);
        pass::upload_batch_ubo(
            &self.gl,
            &self.vtx_ubo,
            &batch_camera,
            false,
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

    /// Classify one already-posed batch against the current conformal map and
    /// camera, writing one visibility float per patch.
    pub fn classify_patch_batch(
        &self,
        camera: &pass::Camera,
        batch: &pass::RenderBatch,
        source_vao: glow::VertexArray,
        destination: glow::Buffer,
        byte_offset: i32,
    ) {
        let batch_camera = pass::camera_for_batch(camera, batch);
        pass::upload_batch_ubo(
            &self.gl,
            &self.vtx_ubo,
            &batch_camera,
            batch.suppress_source_roots,
            1,
            &batch.euclidean_model,
            &batch.euclidean_normal,
        );
        self.patch_visibility.classify_range(
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

    /// Resolve an application-owned WebGL program descriptor in this
    /// renderer's context epoch. The returned handle is non-owning.
    pub fn resolve_program(&mut self, key: WebGlProgramKey) -> Result<glow::Program, String> {
        self.program_memo.get_or_create(&self.gl, key)
    }

    /// CPU-only cache counters; querying these never synchronizes with the GPU.
    pub fn program_memo_diagnostics(&self) -> WebGlProgramMemoDiagnostics {
        self.program_memo.diagnostics()
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

    /// Replace the immutable source-face resource used to expand compact LOD
    /// topology records during the current-pose preparation pass.
    pub fn upload_face_data_texture(
        &mut self,
        instances: &[f32],
        num_faces: usize,
    ) -> Result<(), String> {
        let texture = FaceDataTexture::new(&self.gl, instances, num_faces)?;
        texture.bind(&self.gl, shader::FACE_DATA_TEX_UNIT);
        if let Some(previous) = self.face_data_texture.replace(texture) {
            previous.destroy(&self.gl);
        }
        Ok(())
    }

    /// Replace the exact source-root suppression mask. Reallocation is needed
    /// only when model identity changes; ordinary adaptive updates touch sparse
    /// row-contiguous bytes in the retained texture.
    pub fn set_suppressed_source_faces(
        &mut self,
        num_faces: usize,
        faces: &[u32],
    ) -> Result<usize, String> {
        if self.suppressed_face_texture.num_faces != num_faces {
            let mut replacement = SuppressedFaceTexture::new(&self.gl, num_faces)?;
            let changed = match replacement.replace(&self.gl, faces) {
                Ok(changed) => changed,
                Err(error) => {
                    replacement.destroy(&self.gl);
                    return Err(error);
                }
            };
            replacement.bind(&self.gl, shader::SUPPRESSED_FACE_TEX_UNIT);
            let previous = std::mem::replace(&mut self.suppressed_face_texture, replacement);
            previous.destroy(&self.gl);
            return Ok(changed);
        }
        let changed = self.suppressed_face_texture.replace(&self.gl, faces)?;
        self.suppressed_face_texture
            .bind(&self.gl, shader::SUPPRESSED_FACE_TEX_UNIT);
        Ok(changed)
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

    /// Re-establish vertex-stage sampler bindings after arbitrary render passes.
    pub fn bind_vertex_textures(&self) {
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
            self.gl.active_texture(glow::TEXTURE0 + shader::FACE_DATA_TEX_UNIT);
            self.gl.bind_texture(
                glow::TEXTURE_2D,
                self.face_data_texture.as_ref().map(|texture| texture.texture),
            );
            self.gl.active_texture(glow::TEXTURE0 + shader::SUPPRESSED_FACE_TEX_UNIT);
            self.gl.bind_texture(
                glow::TEXTURE_2D,
                Some(self.suppressed_face_texture.texture),
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
        self.bind_vertex_textures();
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        // Transform-feedback objects release their bindings before the memo
        // deletes every linked program and then every shared shader module.
        self.patch_visibility.destroy(&self.gl);
        self.patch_preparer.destroy(&self.gl);
        self.program_memo.destroy(&self.gl);
        self.vtx_ubo.destroy(&self.gl);
        self.wire_ubo.destroy(&self.gl);
        self.pbr_ubo.destroy(&self.gl);
        self.matcap_ubo.destroy(&self.gl);
        self.joint_ubo.destroy(&self.gl);
        if let Some(texture) = self.face_data_texture.take() {
            texture.destroy(&self.gl);
        }
        self.suppressed_face_texture.destroy(&self.gl);
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
            vertex.contains("_group_0_binding_1_vs"),
            "vertex shader should have the skinning texture sampler"
        );
        assert!(
            vertex.contains("_group_0_binding_2_vs"),
            "vertex shader should have the morph texture sampler"
        );
    }

    #[test]
    fn affine_conformal_maps_still_transform_normals_and_stretch() {
        let source = quilting_shaders::sources::PATCH_RENDER;
        assert!(
            !source.contains("let is_mobius"),
            "c=0 includes rotations and signed scales, so it cannot bypass the differential"
        );
        assert!(
            source.contains(
                "let differential = uniforms.mob_a - qmul(mapped, uniforms.mob_c)"
            ),
            "stretch must use the full fractional-linear differential"
        );
    }

    #[test]
    fn current_pose_culling_precedes_surface_evaluation() {
        let source = quilting_shaders::sources::VERTEX_MAIN;
        let main = source.find("fn vs_main").expect("main vertex entry point");
        let cull = source[main..]
            .find("if prepared_patch_outside_frustum(")
            .expect("vertex shader must cull from current posed control points");
        let evaluate = source[main + cull..].find("evaluate_patch_surface(")
            .expect("QB evaluation must follow current-pose culling");
        assert!(evaluate > 0);
        assert_eq!(
            &quilting_core::instance_layout::ATTR_MAP[3..6],
            &[(4, 48), (5, 64), (6, 80)],
            "rational QB weights must be resident patch attributes",
        );
        let visibility = quilting_shaders::sources::PATCH_VISIBILITY;
        assert!(visibility.contains("origin_to_quaternion_triangle"));
        assert!(visibility.contains("rational_patch_outside_frustum"));
        assert!(quilting_shaders::sources::PATCH_RENDER.contains(
            "fn evaluate_mobius_qb_patch("
        ));
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

        let source = quilting_shaders::sources::PATCH_RENDER;
        assert!(
            source.contains(
                "let bary = permute_patch_barycentric(input.atlas_bary, permutation)"
            ),
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
