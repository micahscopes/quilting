//! GPU compute via transform feedback (WebGL2 GPGPU).
//!
//! Two-pass pipeline — one "vertex" per face:
//!   Pass 1: animated positions → conservative image bound + raw LOD exponents
//!   Pass 2: edge coherence (via adjacency texture) + canonical sort + atlas LUT
//!
//! Pass 1 renders directly to a texture for pass 2 to read.
//! Final output: 6 floats per face (canon_a, canon_b, canon_c, perm_index, parity, atlas_index),
//! directly consumable by group_into_batches.

use glow::HasContext;

#[cfg(target_arch = "wasm32")]
fn log_info(msg: &str) {
    web_sys::console::info_1(&msg.into());
}
#[cfg(not(target_arch = "wasm32"))]
fn log_info(msg: &str) {
    eprintln!("{}", msg);
}

/// Pass 1 FBO payload: raw LOD exponents plus conservative visibility.
pub const FLOATS_PER_FACE_PASS1: usize = 4;

/// Pass 2 final output: (canon_a, canon_b, canon_c, perm_index, parity, atlas_index) = 6 floats per face.
pub const FLOATS_PER_FACE_OUTPUT: usize = 6;

/// GPU-side copy of a completed LOD transform-feedback output. The worker can
/// fence after staging one or more runs, poll without blocking, then read only
/// once the shared fence is signaled.
pub struct StagedLodReadback {
    buffer: glow::Buffer,
    num_faces: usize,
}

const LOD_COMPUTE_VS: &str = include_str!("../shaders/lod_compute.vert.glsl");
const LOD_COMPUTE_FS: &str = include_str!("../shaders/lod_compute.frag.glsl");
const LOD_COHERENCE_VS: &str = include_str!("../shaders/lod_coherence.vert.glsl");
const DUMMY_FS: &str = include_str!("../shaders/lod_dummy.frag.glsl");

/// Two-pass LOD compute pipeline.
/// Pass 1: FBO render (LOD exponents → RGBA32F texture, one pixel per face)
/// Pass 2: Transform feedback (edge coherence + canonicalize → readback buffer)
pub struct LodCompute {
    // --- Pass 1: FBO render for LOD exponents ---
    program1: glow::Program,
    vao1: glow::VertexArray,
    input_buf: glow::Buffer,        // face indices (3 floats per face)
    pass1_fbo: glow::Framebuffer,   // renders LOD exponents to texture
    pass1_texture: Option<glow::Texture>,  // RGBA32F, one pixel per face
    pass1_tex_w: i32,
    pass1_tex_h: i32,

    // Textures for pass 1 (animation + geometry)
    pos_texture: Option<glow::Texture>,
    skinning_texture: Option<glow::Texture>,
    joints_texture: Option<glow::Texture>,
    joints_texture_capacity: usize,
    morph_texture: Option<glow::Texture>,
    morph_wt_texture: Option<glow::Texture>,
    morph_wt_texture_capacity: usize,

    // Pass 1 uniform locations
    pos_loc: Option<glow::UniformLocation>,
    skinning_loc: Option<glow::UniformLocation>,
    joints_loc: Option<glow::UniformLocation>,
    morph_deltas_loc: Option<glow::UniformLocation>,
    morph_wt_loc: Option<glow::UniformLocation>,
    mob_a_loc: glow::UniformLocation,
    mob_b_loc: glow::UniformLocation,
    mob_c_loc: glow::UniformLocation,
    mob_d_loc: glow::UniformLocation,
    u_pole_loc: Option<glow::UniformLocation>,
    u_mob_k_loc: Option<glow::UniformLocation>,
    u_c_norm_sq_loc: Option<glow::UniformLocation>,
    u_has_pole_loc: Option<glow::UniformLocation>,
    model_matrix_loc: glow::UniformLocation,
    density_loc: glow::UniformLocation,
    mesh_radius_loc: glow::UniformLocation,
    min_px_loc: glow::UniformLocation,
    max_lod_loc: glow::UniformLocation,
    vp_matrix_loc: glow::UniformLocation,
    vp_width_loc: glow::UniformLocation,
    vp_height_loc: glow::UniformLocation,
    num_verts_loc: Option<glow::UniformLocation>,
    num_joints_loc: Option<glow::UniformLocation>,
    num_morph_loc: Option<glow::UniformLocation>,
    fbo_width_loc: Option<glow::UniformLocation>,
    fbo_height_loc: Option<glow::UniformLocation>,

    // --- Pass 2: TF edge coherence + canonicalize ---
    program2: glow::Program,
    vao2: glow::VertexArray,
    output_buf2: glow::Buffer,      // TF output: 6 floats per face (final)
    tf2: glow::TransformFeedback,

    // Static adjacency texture (uploaded once per model)
    adjacency_texture: Option<glow::Texture>,

    // Atlas LUT texture (shared by pass 2)
    lut_texture: Option<glow::Texture>,

    // Pass 2 uniform locations
    p2_lods_loc: Option<glow::UniformLocation>,
    p2_adj_loc: Option<glow::UniformLocation>,
    p2_lut_loc: Option<glow::UniformLocation>,
    p2_num_faces_loc: Option<glow::UniformLocation>,

    max_faces: usize,
    bound1: bool,
}

impl LodCompute {
    pub fn new(gl: &glow::Context, max_faces: usize) -> Result<Self, String> {
        unsafe {
            // --- Pass 1: regular program (renders to FBO, not TF) ---
            let program1 = build_program(gl, LOD_COMPUTE_VS, LOD_COMPUTE_FS, "LOD pass 1")?;

            let req = |prog, name: &str| -> Result<glow::UniformLocation, String> {
                gl.get_uniform_location(prog, name)
                    .ok_or_else(|| format!("{name} uniform not found in pass 1"))
            };

            let mob_a_loc = req(program1, "mob_a")?;
            let mob_b_loc = req(program1, "mob_b")?;
            let mob_c_loc = req(program1, "mob_c")?;
            let mob_d_loc = req(program1, "mob_d")?;
            let u_pole_loc = gl.get_uniform_location(program1, "u_pole");
            let u_mob_k_loc = gl.get_uniform_location(program1, "u_mob_k");
            let u_c_norm_sq_loc = gl.get_uniform_location(program1, "u_c_norm_sq");
            let u_has_pole_loc = gl.get_uniform_location(program1, "u_has_pole");
            let model_matrix_loc = req(program1, "model_matrix")?;
            let density_loc = req(program1, "density")?;
            let mesh_radius_loc = req(program1, "mesh_radius")?;
            let min_px_loc = req(program1, "min_px")?;
            let max_lod_loc = req(program1, "max_lod")?;
            let vp_matrix_loc = req(program1, "vp_matrix")?;
            let vp_width_loc = req(program1, "vp_width")?;
            let vp_height_loc = req(program1, "vp_height")?;
            let num_verts_loc = gl.get_uniform_location(program1, "u_num_vertices");
            let num_joints_loc = gl.get_uniform_location(program1, "u_num_joints");
            let num_morph_loc = gl.get_uniform_location(program1, "u_num_morph_targets");
            let fbo_width_loc = gl.get_uniform_location(program1, "u_fbo_width");
            let fbo_height_loc = gl.get_uniform_location(program1, "u_fbo_height");

            // Pass 1 VAO + input buffer
            let vao1 = gl.create_vertex_array().map_err(|e| format!("{e}"))?;
            gl.bind_vertex_array(Some(vao1));

            let input_buf = gl.create_buffer().map_err(|e| format!("{e}"))?;
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(input_buf));
            gl.buffer_data_size(glow::ARRAY_BUFFER,
                (max_faces * 3 * 4) as i32, glow::STATIC_DRAW);
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(0, 3, glow::FLOAT, false, 12, 0);
            gl.bind_vertex_array(None);

            // Pass 1 FBO (texture attached in upload_adjacency when we know num_faces)
            let pass1_fbo = gl.create_framebuffer().map_err(|e| format!("{e}"))?;

            // --- Pass 2: TF program ---
            let program2 = build_tf_program(gl, LOD_COHERENCE_VS, DUMMY_FS,
                &["out_canon_a", "out_canon_b", "out_canon_c", "out_perm_index", "out_parity", "out_atlas_index"],
                "LOD pass 2")?;

            let vao2 = gl.create_vertex_array().map_err(|e| format!("{e}"))?;

            let output_buf2 = gl.create_buffer().map_err(|e| format!("{e}"))?;
            gl.bind_buffer(glow::TRANSFORM_FEEDBACK_BUFFER, Some(output_buf2));
            gl.buffer_data_size(glow::TRANSFORM_FEEDBACK_BUFFER,
                (max_faces * FLOATS_PER_FACE_OUTPUT * 4) as i32,
                glow::DYNAMIC_READ);

            let tf2 = gl.create_transform_feedback().map_err(|e| format!("{e}"))?;

            Ok(Self {
                program1, vao1, input_buf, pass1_fbo,
                pass1_texture: None, pass1_tex_w: 0, pass1_tex_h: 0,
                pos_texture: None, skinning_texture: None, joints_texture: None,
                joints_texture_capacity: 0,
                morph_texture: None, morph_wt_texture: None,
                morph_wt_texture_capacity: 0,
                pos_loc: gl.get_uniform_location(program1, "u_positions"),
                skinning_loc: gl.get_uniform_location(program1, "u_skinning"),
                joints_loc: gl.get_uniform_location(program1, "u_joints"),
                morph_deltas_loc: gl.get_uniform_location(program1, "u_morph_deltas"),
                morph_wt_loc: gl.get_uniform_location(program1, "u_morph_wt"),
                mob_a_loc, mob_b_loc, mob_c_loc, mob_d_loc,
                u_pole_loc, u_mob_k_loc, u_c_norm_sq_loc, u_has_pole_loc,
                model_matrix_loc,
                density_loc, mesh_radius_loc,
                min_px_loc, max_lod_loc, vp_matrix_loc, vp_width_loc, vp_height_loc,
                num_verts_loc, num_joints_loc, num_morph_loc,
                fbo_width_loc, fbo_height_loc,

                program2, vao2, output_buf2, tf2,
                adjacency_texture: None,
                lut_texture: None,
                p2_lods_loc: gl.get_uniform_location(program2, "u_pass1_lods"),
                p2_adj_loc: gl.get_uniform_location(program2, "u_adjacency"),
                p2_lut_loc: gl.get_uniform_location(program2, "u_atlas_lut"),
                p2_num_faces_loc: gl.get_uniform_location(program2, "u_num_faces"),

                max_faces,
                bound1: false,
            })
        }
    }

    pub fn has_pass1_texture(&self) -> bool { self.pass1_texture.is_some() }
    pub fn has_adjacency_texture(&self) -> bool { self.adjacency_texture.is_some() }

    // --- Static data uploads (called once on model load) ---

    /// Upload atlas LUT: exponent triples → atlas indices.
    pub fn upload_atlas_lut(&mut self, gl: &glow::Context, lut: &[u8]) {
        unsafe {
            if let Some(old) = self.lut_texture { gl.delete_texture(old); }
            let tex = gl.create_texture().unwrap();
            gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            let mut data = vec![255u8; 1200];
            for (i, &v) in lut.iter().take(1200).enumerate() { data[i] = v; }
            gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::R8 as i32,
                40, 30, 0, glow::RED, glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(&data)));
            set_nearest(gl);
            self.lut_texture = Some(tex);
            self.bound1 = false;
        }
    }

    /// Upload face indices (3 floats per face — vertex indices as floats).
    pub fn upload_face_indices(&self, gl: &glow::Context, indices: &[f32]) {
        let max_floats = self.max_faces * 3;
        let clamped = if indices.len() > max_floats { &indices[..max_floats] } else { indices };
        unsafe {
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.input_buf));
            gl.buffer_data_u8_slice(glow::ARRAY_BUFFER,
                bytemuck_cast_slice(clamped), glow::STATIC_DRAW);
        }
    }

    /// Upload rest-pose positions as a float texture.
    pub fn upload_positions_texture(&mut self, gl: &glow::Context, positions: &[f32], num_vertices: usize) {
        unsafe {
            let mut rgba = vec![0.0f32; num_vertices * 4];
            for i in 0..num_vertices {
                rgba[i*4]   = positions[i*3];
                rgba[i*4+1] = positions[i*3+1];
                rgba[i*4+2] = positions[i*3+2];
            }

            let width = 4096;
            let height = (num_vertices + width - 1) / width;
            rgba.resize(width * height * 4, 0.0);

            if let Some(old) = self.pos_texture { gl.delete_texture(old); }
            let tex = gl.create_texture().unwrap();
            gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::RGBA32F as i32,
                width as i32, height as i32, 0,
                glow::RGBA, glow::FLOAT,
                glow::PixelUnpackData::Slice(Some(bytemuck_cast_slice(&rgba))));
            set_nearest(gl);
            self.pos_texture = Some(tex);
            self.bound1 = false;
        }
    }

    /// Upload per-vertex skinning data (joint indices + weights).
    pub fn upload_skinning_texture(
        &mut self, gl: &glow::Context,
        joint_indices: &[[u16; 4]], joint_weights: &[[f32; 4]],
    ) {
        let nv = joint_indices.len();
        if nv == 0 { return; }
        let width = nv.min(4096);
        let height = ((nv + width - 1) / width) * 2;
        let mut data = vec![0.0f32; width * height * 4];
        for (i, (ji, jw)) in joint_indices.iter().zip(joint_weights.iter()).enumerate() {
            let chunk = i / width;
            let col = i % width;
            let idx_row = chunk * 2;
            let wt_row = chunk * 2 + 1;
            let idx_off = (idx_row * width + col) * 4;
            data[idx_off]     = ji[0] as f32;
            data[idx_off + 1] = ji[1] as f32;
            data[idx_off + 2] = ji[2] as f32;
            data[idx_off + 3] = ji[3] as f32;
            let wt_off = (wt_row * width + col) * 4;
            data[wt_off]     = jw[0];
            data[wt_off + 1] = jw[1];
            data[wt_off + 2] = jw[2];
            data[wt_off + 3] = jw[3];
        }
        unsafe {
            if let Some(old) = self.skinning_texture { gl.delete_texture(old); }
            let tex = gl.create_texture().unwrap();
            gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::RGBA32F as i32,
                width as i32, height as i32, 0, glow::RGBA, glow::FLOAT,
                glow::PixelUnpackData::Slice(Some(bytemuck_cast_slice(&data))));
            set_nearest(gl);
            self.skinning_texture = Some(tex);
            self.bound1 = false;
        }
    }

    /// Upload morph target delta texture (static).
    pub fn upload_morph_deltas(
        &mut self, gl: &glow::Context,
        deltas: &[f32], num_vertices: usize, num_targets: usize,
    ) {
        let mut rgba = vec![0.0f32; num_vertices * num_targets * 4];
        for t in 0..num_targets {
            for v in 0..num_vertices {
                let src = (t * num_vertices + v) * 3;
                let dst = (t * num_vertices + v) * 4;
                if src + 2 < deltas.len() {
                    rgba[dst]     = deltas[src];
                    rgba[dst + 1] = deltas[src + 1];
                    rgba[dst + 2] = deltas[src + 2];
                }
            }
        }
        unsafe {
            if let Some(old) = self.morph_texture { gl.delete_texture(old); }
            let tex = gl.create_texture().unwrap();
            gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::RGBA32F as i32,
                num_vertices as i32, num_targets as i32, 0,
                glow::RGBA, glow::FLOAT,
                glow::PixelUnpackData::Slice(Some(bytemuck_cast_slice(&rgba))));
            set_nearest(gl);
            self.morph_texture = Some(tex);
            self.bound1 = false;
        }
    }

    /// Upload adjacency data for edge coherence (called once per model).
    ///
    /// `data`: flat f32 array, 4 floats per entry, 3 entries per face.
    /// Entry for face `fi`, edge `e`: data[(fi*3+e)*4 .. +4] = (neighbor_face, neighbor_lod_idx, 0, 0).
    /// neighbor_face < 0 means boundary (no neighbor).
    ///
    /// Stored as RGBA32F texture, tiled 4096-wide.
    pub fn upload_adjacency(&mut self, gl: &glow::Context, data: &[f32], num_faces: usize) {
        let total_texels = num_faces * 3;
        let width = 4096;
        let height = (total_texels + width - 1) / width;
        let mut padded = vec![0.0f32; width * height * 4];
        let copy_len = data.len().min(padded.len());
        padded[..copy_len].copy_from_slice(&data[..copy_len]);

        unsafe {
            if let Some(old) = self.adjacency_texture { gl.delete_texture(old); }
            let tex = gl.create_texture().unwrap();
            gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::RGBA32F as i32,
                width as i32, height as i32, 0,
                glow::RGBA, glow::FLOAT,
                glow::PixelUnpackData::Slice(Some(bytemuck_cast_slice(&padded))));
            set_nearest(gl);
            self.adjacency_texture = Some(tex);
        }

        // Allocate pass 1 FBO texture (RGBA32F, one pixel per face)
        let p1_w = 4096i32;
        let p1_h = ((num_faces + 4095) / 4096) as i32;
        unsafe {
            if let Some(old) = self.pass1_texture { gl.delete_texture(old); }
            let tex = gl.create_texture().unwrap();
            gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::RGBA32F as i32,
                p1_w, p1_h, 0,
                glow::RGBA, glow::FLOAT,
                glow::PixelUnpackData::Slice(None));
            set_nearest(gl);
            self.pass1_texture = Some(tex);
            self.pass1_tex_w = p1_w;
            self.pass1_tex_h = p1_h;

            // Attach texture to FBO
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.pass1_fbo));
            gl.framebuffer_texture_2d(glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D, Some(tex), 0);

            let status = gl.check_framebuffer_status(glow::FRAMEBUFFER);
            if status != glow::FRAMEBUFFER_COMPLETE {
                log_info(&format!("Pass 1 FBO incomplete: 0x{:x}", status));
            }
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        }
    }

    // --- Per-frame animation uploads ---

    /// Upload joint matrices for the current frame.
    pub fn upload_joint_matrices(&mut self, gl: &glow::Context, matrices: &[f32]) {
        let num_joints = matrices.len() / 16;
        if num_joints == 0 { return; }
        unsafe {
            let needs_allocation = self.joints_texture.is_none()
                || self.joints_texture_capacity < num_joints;
            if needs_allocation {
                if let Some(old) = self.joints_texture { gl.delete_texture(old); }
                self.joints_texture = Some(gl.create_texture().unwrap());
                self.joints_texture_capacity = num_joints;
            }
            let tex = self.joints_texture.unwrap();
            gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            if needs_allocation {
                gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::RGBA32F as i32,
                    4, self.joints_texture_capacity as i32, 0,
                    glow::RGBA, glow::FLOAT,
                    glow::PixelUnpackData::Slice(None));
                set_nearest(gl);
                self.bound1 = false;
            }
            gl.tex_sub_image_2d(glow::TEXTURE_2D, 0, 0, 0,
                4, num_joints as i32, glow::RGBA, glow::FLOAT,
                glow::PixelUnpackData::Slice(Some(bytemuck_cast_slice(
                    &matrices[..num_joints * 16]
                ))));
        }
    }

    /// Upload morph weights for the current frame.
    pub fn upload_morph_weights(&mut self, gl: &glow::Context, weights: &[f32]) {
        if weights.is_empty() { return; }
        unsafe {
            let needs_allocation = self.morph_wt_texture.is_none()
                || self.morph_wt_texture_capacity < weights.len();
            if needs_allocation {
                if let Some(old) = self.morph_wt_texture { gl.delete_texture(old); }
                self.morph_wt_texture = Some(gl.create_texture().unwrap());
                self.morph_wt_texture_capacity = weights.len();
            }
            let tex = self.morph_wt_texture.unwrap();
            gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            if needs_allocation {
                gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::R32F as i32,
                    self.morph_wt_texture_capacity as i32, 1, 0,
                    glow::RED, glow::FLOAT,
                    glow::PixelUnpackData::Slice(None));
                set_nearest(gl);
                self.bound1 = false;
            }
            gl.tex_sub_image_2d(glow::TEXTURE_2D, 0, 0, 0,
                weights.len() as i32, 1, glow::RED, glow::FLOAT,
                glow::PixelUnpackData::Slice(Some(bytemuck_cast_slice(weights))));
        }
    }

    // --- Two-pass compute ---

    /// Run both LOD compute passes. Returns number of faces processed.
    pub fn compute_lods(
        &mut self,
        gl: &glow::Context,
        num_faces: usize,
        num_vertices: u32,
        num_joints: u32,
        num_morph_targets: u32,
        mobius: [f32; 16],
        model_matrix: [f32; 16],
        pole: [f32; 4],
        mob_k: f32,
        c_norm_sq: f32,
        has_pole: f32,
        density: f32,
        mesh_radius: f32,
        min_px: f32,
        max_lod: f32,
        vp_matrix: &[f32; 16],
        vp_width: f32,
        vp_height: f32,
    ) -> usize {
        let n = num_faces.min(self.max_faces);
        if self.pass1_texture.is_none() || self.adjacency_texture.is_none() {
            log_info(&format!("LOD compute skipped: pass1_tex={} adj_tex={}",
                self.pass1_texture.is_some(), self.adjacency_texture.is_some()));
            return 0;
        }
        unsafe {
            // === Pass 1: LOD exponent computation (FBO render) ===
            gl.use_program(Some(self.program1));

            if !self.bound1 {
                let mut unit = 0u32;
                let bind_tex = |gl: &glow::Context, unit: &mut u32, tex: Option<glow::Texture>, loc: &Option<glow::UniformLocation>| {
                    gl.active_texture(glow::TEXTURE0 + *unit);
                    if let Some(t) = tex {
                        gl.bind_texture(glow::TEXTURE_2D, Some(t));
                    }
                    if let Some(ref l) = loc {
                        gl.uniform_1_i32(Some(l), *unit as i32);
                    }
                    *unit += 1;
                };

                bind_tex(gl, &mut unit, self.pos_texture, &self.pos_loc);           // unit 0
                bind_tex(gl, &mut unit, self.skinning_texture, &self.skinning_loc);  // unit 1
                bind_tex(gl, &mut unit, self.joints_texture, &self.joints_loc);      // unit 2
                bind_tex(gl, &mut unit, self.morph_texture, &self.morph_deltas_loc); // unit 3
                bind_tex(gl, &mut unit, self.morph_wt_texture, &self.morph_wt_loc);  // unit 4

                self.bound1 = true;
            }

            // Bind pass 1 VAO and FBO
            gl.bind_vertex_array(Some(self.vao1));
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.pass1_fbo));
            gl.viewport(0, 0, self.pass1_tex_w, self.pass1_tex_h);

            // Per-frame uniforms
            if let Some(ref loc) = self.num_verts_loc {
                gl.uniform_1_i32(Some(loc), num_vertices as i32);
            }
            if let Some(ref loc) = self.num_joints_loc {
                gl.uniform_1_i32(Some(loc), num_joints as i32);
            }
            if let Some(ref loc) = self.num_morph_loc {
                gl.uniform_1_i32(Some(loc), num_morph_targets as i32);
            }
            if let Some(ref loc) = self.fbo_width_loc {
                gl.uniform_1_i32(Some(loc), self.pass1_tex_w);
            }
            if let Some(ref loc) = self.fbo_height_loc {
                gl.uniform_1_i32(Some(loc), self.pass1_tex_h);
            }

            // Re-bind per-frame textures
            if let Some(tex) = self.joints_texture {
                gl.active_texture(glow::TEXTURE2);
                gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            }
            if let Some(tex) = self.morph_wt_texture {
                gl.active_texture(glow::TEXTURE4);
                gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            }

            gl.uniform_4_f32(Some(&self.mob_a_loc), mobius[0], mobius[1], mobius[2], mobius[3]);
            gl.uniform_4_f32(Some(&self.mob_b_loc), mobius[4], mobius[5], mobius[6], mobius[7]);
            gl.uniform_4_f32(Some(&self.mob_c_loc), mobius[8], mobius[9], mobius[10], mobius[11]);
            gl.uniform_4_f32(Some(&self.mob_d_loc), mobius[12], mobius[13], mobius[14], mobius[15]);
            if let Some(l) = &self.u_pole_loc { gl.uniform_4_f32(Some(l), pole[0], pole[1], pole[2], pole[3]); }
            if let Some(l) = &self.u_mob_k_loc { gl.uniform_1_f32(Some(l), mob_k); }
            if let Some(l) = &self.u_c_norm_sq_loc { gl.uniform_1_f32(Some(l), c_norm_sq); }
            if let Some(l) = &self.u_has_pole_loc { gl.uniform_1_f32(Some(l), has_pole); }
            gl.uniform_matrix_4_f32_slice(Some(&self.model_matrix_loc), false, &model_matrix);
            gl.uniform_1_f32(Some(&self.density_loc), density);
            gl.uniform_1_f32(Some(&self.mesh_radius_loc), mesh_radius);
            gl.uniform_1_f32(Some(&self.min_px_loc), min_px);
            gl.uniform_1_f32(Some(&self.max_lod_loc), max_lod);
            gl.uniform_matrix_4_f32_slice(Some(&self.vp_matrix_loc), false, vp_matrix);
            gl.uniform_1_f32(Some(&self.vp_width_loc), vp_width);
            gl.uniform_1_f32(Some(&self.vp_height_loc), vp_height);

            // Clear FBO and render pass 1 (one point per face → one pixel of LOD exponents)
            // Use a sentinel clear value that would produce obviously wrong LOD if read
            gl.clear_color(-1.0, -1.0, -1.0, 0.0);
            gl.clear(glow::COLOR_BUFFER_BIT);

            // Ensure depth test doesn't cull our points
            gl.disable(glow::DEPTH_TEST);

            gl.draw_arrays(glow::POINTS, 0, n as i32);

            // Unbind FBO — pass1_texture now contains LOD exponents
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);

            // === Pass 2: edge coherence + canonicalize (TF) ===
            gl.enable(glow::RASTERIZER_DISCARD);
            gl.use_program(Some(self.program2));

            // Bind textures for pass 2
            gl.active_texture(glow::TEXTURE0);
            if let Some(tex) = self.pass1_texture {
                gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            }
            if let Some(ref loc) = self.p2_lods_loc {
                gl.uniform_1_i32(Some(loc), 0);
            }

            gl.active_texture(glow::TEXTURE1);
            if let Some(tex) = self.adjacency_texture {
                gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            }
            if let Some(ref loc) = self.p2_adj_loc {
                gl.uniform_1_i32(Some(loc), 1);
            }

            gl.active_texture(glow::TEXTURE2);
            if let Some(tex) = self.lut_texture {
                gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            }
            if let Some(ref loc) = self.p2_lut_loc {
                gl.uniform_1_i32(Some(loc), 2);
            }

            if let Some(ref loc) = self.p2_num_faces_loc {
                gl.uniform_1_i32(Some(loc), n as i32);
            }

            gl.bind_vertex_array(Some(self.vao2));
            gl.bind_transform_feedback(glow::TRANSFORM_FEEDBACK, Some(self.tf2));
            gl.bind_buffer_base(glow::TRANSFORM_FEEDBACK_BUFFER, 0, Some(self.output_buf2));

            gl.begin_transform_feedback(glow::POINTS);
            gl.draw_arrays(glow::POINTS, 0, n as i32);
            gl.end_transform_feedback();

            gl.disable(glow::RASTERIZER_DISCARD);
            gl.bind_transform_feedback(glow::TRANSFORM_FEEDBACK, None);
            gl.bind_vertex_array(None);
            gl.use_program(None);
            gl.flush();

            // Pass 2 clobbers texture units — force rebind on next frame
            self.bound1 = false;
        }
        n
    }

    /// Copy the latest transform-feedback result into an independent GPU
    /// staging buffer. This is GPU-to-GPU and does not wait for completion.
    pub fn stage_readback(
        &self,
        gl: &glow::Context,
        num_faces: usize,
    ) -> Result<StagedLodReadback, String> {
        let n = num_faces.min(self.max_faces);
        let byte_size = n
            .checked_mul(FLOATS_PER_FACE_OUTPUT)
            .and_then(|size| size.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| "LOD staging buffer size overflow".to_string())?;
        let byte_size = i32::try_from(byte_size)
            .map_err(|_| "LOD staging buffer exceeds WebGL2 limits".to_string())?;
        unsafe {
            let buffer = gl.create_buffer().map_err(|error| format!("LOD staging buffer: {error}"))?;
            gl.bind_buffer(glow::COPY_READ_BUFFER, Some(self.output_buf2));
            gl.bind_buffer(glow::COPY_WRITE_BUFFER, Some(buffer));
            gl.buffer_data_size(glow::COPY_WRITE_BUFFER, byte_size, glow::STREAM_READ);
            gl.copy_buffer_sub_data(
                glow::COPY_READ_BUFFER,
                glow::COPY_WRITE_BUFFER,
                0,
                0,
                byte_size,
            );
            gl.bind_buffer(glow::COPY_READ_BUFFER, None);
            gl.bind_buffer(glow::COPY_WRITE_BUFFER, None);
            Ok(StagedLodReadback { buffer, num_faces: n })
        }
    }

    /// Read and destroy a staging buffer after its fence has signaled.
    pub fn finish_staged_readback(
        &self,
        gl: &glow::Context,
        staged: StagedLodReadback,
    ) -> Vec<f32> {
        let mut result = vec![0.0f32; staged.num_faces * FLOATS_PER_FACE_OUTPUT];
        unsafe {
            gl.bind_buffer(glow::COPY_READ_BUFFER, Some(staged.buffer));
            gl.get_buffer_sub_data(
                glow::COPY_READ_BUFFER,
                0,
                bytemuck_cast_slice_mut(&mut result),
            );
            gl.bind_buffer(glow::COPY_READ_BUFFER, None);
            gl.delete_buffer(staged.buffer);
        }
        result
    }

    /// Destroy a staged result that will not be consumed.
    pub fn discard_staged_readback(&self, gl: &glow::Context, staged: StagedLodReadback) {
        unsafe { gl.delete_buffer(staged.buffer); }
    }

    pub fn destroy(&self, gl: &glow::Context) {
        unsafe {
            gl.delete_program(self.program1);
            gl.delete_vertex_array(self.vao1);
            gl.delete_buffer(self.input_buf);
            gl.delete_framebuffer(self.pass1_fbo);

            gl.delete_program(self.program2);
            gl.delete_vertex_array(self.vao2);
            gl.delete_buffer(self.output_buf2);
            gl.delete_transform_feedback(self.tf2);

            if let Some(t) = self.pos_texture { gl.delete_texture(t); }
            if let Some(t) = self.lut_texture { gl.delete_texture(t); }
            if let Some(t) = self.skinning_texture { gl.delete_texture(t); }
            if let Some(t) = self.joints_texture { gl.delete_texture(t); }
            if let Some(t) = self.morph_texture { gl.delete_texture(t); }
            if let Some(t) = self.morph_wt_texture { gl.delete_texture(t); }
            if let Some(t) = self.pass1_texture { gl.delete_texture(t); }
            if let Some(t) = self.adjacency_texture { gl.delete_texture(t); }
        }
    }
}

/// Build a regular (non-TF) program from vertex + fragment source.
fn build_program(
    gl: &glow::Context,
    vs_src: &str,
    fs_src: &str,
    label: &str,
) -> Result<glow::Program, String> {
    unsafe {
        let vs = compile_shader(gl, glow::VERTEX_SHADER, vs_src)?;
        let fs = compile_shader(gl, glow::FRAGMENT_SHADER, fs_src)?;

        let program = gl.create_program().map_err(|e| format!("{e}"))?;
        gl.attach_shader(program, vs);
        gl.attach_shader(program, fs);

        gl.link_program(program);
        if !gl.get_program_link_status(program) {
            let log = gl.get_program_info_log(program);
            return Err(format!("{label} link: {log}"));
        }
        gl.detach_shader(program, vs);
        gl.detach_shader(program, fs);
        gl.delete_shader(vs);
        gl.delete_shader(fs);
        Ok(program)
    }
}

/// Build a TF program from vertex + fragment source with specified varyings.
fn build_tf_program(
    gl: &glow::Context,
    vs_src: &str,
    fs_src: &str,
    varyings: &[&str],
    label: &str,
) -> Result<glow::Program, String> {
    crate::shader::create_transform_feedback_program(gl, vs_src, fs_src, varyings)
        .map_err(|error| format!("{label}: {error}"))
}

unsafe fn set_nearest(gl: &glow::Context) {
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
}

fn compile_shader(gl: &glow::Context, shader_type: u32, source: &str) -> Result<glow::Shader, String> {
    unsafe {
        let shader = gl.create_shader(shader_type).map_err(|e| format!("{e}"))?;
        gl.shader_source(shader, source);
        gl.compile_shader(shader);
        if !gl.get_shader_compile_status(shader) {
            let log = gl.get_shader_info_log(shader);
            gl.delete_shader(shader);
            return Err(format!("compute shader: {log}"));
        }
        Ok(shader)
    }
}

fn bytemuck_cast_slice<T>(data: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * std::mem::size_of::<T>()) }
}

fn bytemuck_cast_slice_mut(data: &mut [f32]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, data.len() * 4) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visibility_payload_survives_both_gpu_passes() {
        assert_eq!(FLOATS_PER_FACE_PASS1, 4);
        assert!(LOD_COMPUTE_VS.contains("flat out vec4 v_lods"));
        assert!(LOD_COMPUTE_FS.contains("frag_color = v_lods"));
        assert!(LOD_COHERENCE_VS.contains("face.w < 0.5"));
        assert!(LOD_COHERENCE_VS.contains("out_atlas_index = -1.0"));
    }

    #[test]
    fn gpu_lod_pass_allows_the_source_triangle_level() {
        assert!(LOD_COMPUTE_VS.contains(
            "lod_a = min(lod_a, clamp(floor_pow2"
        ));
        assert!(LOD_COMPUTE_VS.contains(
            "lod_b = min(lod_b, clamp(floor_pow2"
        ));
        assert!(LOD_COMPUTE_VS.contains(
            "lod_c = min(lod_c, clamp(floor_pow2"
        ));
        assert!(!LOD_COMPUTE_VS.contains(", 2.0, max_lod)"));
    }

    #[test]
    fn screen_attenuation_caps_instead_of_driving_lod() {
        assert!(LOD_COMPUTE_VS.contains("density/curvature demand above"));
        assert_eq!(LOD_COMPUTE_VS.matches("= min(lod_").count(), 3);
        assert_eq!(LOD_COMPUTE_VS.matches("floor_pow2(max(px_").count(), 3);
    }

}
