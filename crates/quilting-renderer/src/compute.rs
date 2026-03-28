//! GPU compute via transform feedback (WebGL2 GPGPU).
//!
//! Runs a vertex shader as a compute kernel — one "vertex" per face.
//! Transform feedback captures the output into a GPU buffer.
//! Used for per-frame LOD computation: fetch rest-pose positions, apply
//! morph targets + skeletal animation on the GPU, then Möbius-transform
//! edge midpoints → deformed medians → LOD per edge.

use glow::HasContext;

/// Per-face output: atlas_index + perm_index = 2 floats.
pub const FLOATS_PER_FACE_OUTPUT: usize = 2;

/// Transform feedback compute pipeline for per-frame LOD calculation.
///
/// Supports three animation modes (all on the GPU):
/// - **Static**: reads rest-pose positions directly
/// - **Morph targets**: applies weighted morph deltas to rest-pose
/// - **Skeletal**: applies joint matrix skinning after morph
///
/// The shader mirrors the rendering vertex shader's animation pipeline:
/// rest-pose → morph → skin → Möbius → LOD.
pub struct LodCompute {
    program: glow::Program,
    vao: glow::VertexArray,
    input_buf: glow::Buffer,
    output_buf: glow::Buffer,
    tf: glow::TransformFeedback,

    // Textures — all RGBA32F unless noted
    pos_texture: Option<glow::Texture>,      // rest-pose positions (static)
    lut_texture: Option<glow::Texture>,      // atlas LUT (R8, static)
    skinning_texture: Option<glow::Texture>, // joint indices + weights (static)
    joints_texture: Option<glow::Texture>,   // joint matrices (per-frame)
    morph_texture: Option<glow::Texture>,    // morph deltas (static)
    morph_wt_texture: Option<glow::Texture>, // morph weights (per-frame)

    // Sampler uniform locations
    pos_loc: Option<glow::UniformLocation>,
    lut_loc: Option<glow::UniformLocation>,
    skinning_loc: Option<glow::UniformLocation>,
    joints_loc: Option<glow::UniformLocation>,
    morph_deltas_loc: Option<glow::UniformLocation>,
    morph_wt_loc: Option<glow::UniformLocation>,

    // Scalar uniforms
    mob_a_loc: glow::UniformLocation,
    mob_b_loc: glow::UniformLocation,
    mob_c_loc: glow::UniformLocation,
    mob_d_loc: glow::UniformLocation,
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

    max_faces: usize,
    bound: bool,
}

const LOD_COMPUTE_VS: &str = include_str!("../shaders/lod_compute.vert.glsl");
const LOD_COMPUTE_FS: &str = include_str!("../shaders/lod_compute.frag.glsl");

impl LodCompute {
    pub fn new(gl: &glow::Context, max_faces: usize) -> Result<Self, String> {
        unsafe {
            let vs = compile_shader(gl, glow::VERTEX_SHADER, LOD_COMPUTE_VS)?;
            let fs = compile_shader(gl, glow::FRAGMENT_SHADER, LOD_COMPUTE_FS)?;

            let program = gl.create_program().map_err(|e| format!("{e}"))?;
            gl.attach_shader(program, vs);
            gl.attach_shader(program, fs);

            gl.transform_feedback_varyings(
                program,
                &["out_atlas_index", "out_perm_index"],
                glow::INTERLEAVED_ATTRIBS,
            );

            gl.link_program(program);
            if !gl.get_program_link_status(program) {
                let log = gl.get_program_info_log(program);
                return Err(format!("LOD compute link: {log}"));
            }
            gl.detach_shader(program, vs);
            gl.detach_shader(program, fs);
            gl.delete_shader(vs);
            gl.delete_shader(fs);

            let req = |name: &str| -> Result<glow::UniformLocation, String> {
                gl.get_uniform_location(program, name)
                    .ok_or_else(|| format!("{name} uniform not found"))
            };

            let mob_a_loc = req("mob_a")?;
            let mob_b_loc = req("mob_b")?;
            let mob_c_loc = req("mob_c")?;
            let mob_d_loc = req("mob_d")?;
            let density_loc = req("density")?;
            let mesh_radius_loc = req("mesh_radius")?;
            let min_px_loc = req("min_px")?;
            let max_lod_loc = req("max_lod")?;
            let vp_matrix_loc = req("vp_matrix")?;
            let vp_width_loc = req("vp_width")?;
            let vp_height_loc = req("vp_height")?;
            // These may be optimized out by the GLSL compiler when unused
            let num_verts_loc = gl.get_uniform_location(program, "u_num_vertices");
            let num_joints_loc = gl.get_uniform_location(program, "u_num_joints");
            let num_morph_loc = gl.get_uniform_location(program, "u_num_morph_targets");

            let vao = gl.create_vertex_array().map_err(|e| format!("{e}"))?;
            gl.bind_vertex_array(Some(vao));

            let input_buf = gl.create_buffer().map_err(|e| format!("{e}"))?;
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(input_buf));
            gl.buffer_data_size(glow::ARRAY_BUFFER,
                (max_faces * 3 * 4) as i32, glow::STATIC_DRAW);
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(0, 3, glow::FLOAT, false, 12, 0);

            let output_buf = gl.create_buffer().map_err(|e| format!("{e}"))?;
            gl.bind_buffer(glow::TRANSFORM_FEEDBACK_BUFFER, Some(output_buf));
            gl.buffer_data_size(glow::TRANSFORM_FEEDBACK_BUFFER,
                (max_faces * FLOATS_PER_FACE_OUTPUT * 4) as i32,
                glow::DYNAMIC_READ);

            let tf = gl.create_transform_feedback().map_err(|e| format!("{e}"))?;
            gl.bind_vertex_array(None);

            Ok(Self {
                program, vao, input_buf, output_buf, tf,
                pos_texture: None, lut_texture: None,
                skinning_texture: None, joints_texture: None,
                morph_texture: None, morph_wt_texture: None,
                pos_loc: gl.get_uniform_location(program, "u_positions"),
                lut_loc: gl.get_uniform_location(program, "u_atlas_lut"),
                skinning_loc: gl.get_uniform_location(program, "u_skinning"),
                joints_loc: gl.get_uniform_location(program, "u_joints"),
                morph_deltas_loc: gl.get_uniform_location(program, "u_morph_deltas"),
                morph_wt_loc: gl.get_uniform_location(program, "u_morph_wt"),
                mob_a_loc, mob_b_loc, mob_c_loc, mob_d_loc,
                density_loc, mesh_radius_loc,
                min_px_loc, max_lod_loc, vp_matrix_loc, vp_width_loc, vp_height_loc,
                num_verts_loc, num_joints_loc, num_morph_loc,
                max_faces,
                bound: false,
            })
        }
    }

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
            self.bound = false;
        }
    }

    /// Upload face indices (3 floats per face — vertex indices as floats).
    /// Clamps to max_faces to prevent GPU buffer overflow.
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
    /// `positions`: flat [x,y,z, ...] for all vertices (single frame).
    /// Packed into a 4096-wide RGBA32F texture.
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
            self.bound = false;
        }
    }

    /// Upload per-vertex skinning data (joint indices + weights).
    /// Tiled layout: RGBA32F, width = min(num_vertices, 4096), rows alternate
    /// (indices, weights) per chunk. Matches the rendering vertex shader's tiled lookup.
    pub fn upload_skinning_texture(
        &mut self, gl: &glow::Context,
        joint_indices: &[[u16; 4]], joint_weights: &[[f32; 4]],
    ) {
        let nv = joint_indices.len();
        if nv == 0 { return; }
        let width = nv.min(4096);
        let height = ((nv + width - 1) / width) * 2; // 2 rows per chunk
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
            self.bound = false;
        }
    }

    /// Upload morph target delta texture (static — deltas don't change).
    /// `deltas`: flat [dx, dy, dz, ...] for all vertices × all targets.
    /// Layout: RGBA32F, width = num_vertices, height = num_targets.
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
            self.bound = false;
        }
    }

    // --- Per-frame animation uploads ---

    /// Upload joint matrices for the current frame.
    /// `matrices`: flat column-major f32, num_joints × 16 floats.
    /// Stored as RGBA32F texture: width = 4, height = num_joints.
    /// Each row = one joint, 4 texels = 4 matrix columns.
    pub fn upload_joint_matrices(&mut self, gl: &glow::Context, matrices: &[f32]) {
        let num_joints = matrices.len() / 16;
        if num_joints == 0 { return; }

        // Already in the right layout: 4 vec4s per joint = width 4, height num_joints
        unsafe {
            if let Some(old) = self.joints_texture { gl.delete_texture(old); }
            let tex = gl.create_texture().unwrap();
            gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::RGBA32F as i32,
                4, num_joints as i32, 0,
                glow::RGBA, glow::FLOAT,
                glow::PixelUnpackData::Slice(Some(bytemuck_cast_slice(matrices))));
            set_nearest(gl);
            self.joints_texture = Some(tex);
            self.bound = false;
        }
    }

    /// Upload morph weights for the current frame.
    /// `weights`: one f32 per morph target.
    /// Stored as RGBA32F texture: width = num_targets, height = 1.
    pub fn upload_morph_weights(&mut self, gl: &glow::Context, weights: &[f32]) {
        if weights.is_empty() { return; }

        // Pack as RGBA (only .r used per texel, but RGBA32F is the reliable format)
        let mut rgba = vec![0.0f32; weights.len() * 4];
        for (i, &w) in weights.iter().enumerate() {
            rgba[i * 4] = w;
        }
        unsafe {
            if let Some(old) = self.morph_wt_texture { gl.delete_texture(old); }
            let tex = gl.create_texture().unwrap();
            gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::RGBA32F as i32,
                weights.len() as i32, 1, 0,
                glow::RGBA, glow::FLOAT,
                glow::PixelUnpackData::Slice(Some(bytemuck_cast_slice(&rgba))));
            set_nearest(gl);
            self.morph_wt_texture = Some(tex);
            self.bound = false;
        }
    }

    // --- Compute pass ---

    /// Run the LOD compute pass.
    ///
    /// Reads rest-pose positions, applies animation (morph + skeletal) on the GPU,
    /// then Möbius-transforms and computes per-face LOD classification.
    ///
    /// `num_joints` / `num_morph_targets`: set to 0 to disable that animation type.
    /// Animation textures must have been uploaded before calling this.
    pub fn compute_lods(
        &mut self,
        gl: &glow::Context,
        num_faces: usize,
        num_vertices: u32,
        num_joints: u32,
        num_morph_targets: u32,
        mobius: [f32; 16],
        density: f32,
        mesh_radius: f32,
        min_px: f32,
        max_lod: f32,
        vp_matrix: &[f32; 16],
        vp_width: f32,
        vp_height: f32,
    ) -> usize {
        let n = num_faces.min(self.max_faces);
        unsafe {
            // Always re-activate program — uniforms require it active,
            // and we can't guarantee nothing else touched GL state between calls.
            gl.use_program(Some(self.program));

            if !self.bound {
                // Bind all textures to fixed texture units
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

                bind_tex(gl, &mut unit, self.pos_texture, &self.pos_loc);       // unit 0
                bind_tex(gl, &mut unit, self.lut_texture, &self.lut_loc);        // unit 1
                bind_tex(gl, &mut unit, self.skinning_texture, &self.skinning_loc); // unit 2
                bind_tex(gl, &mut unit, self.joints_texture, &self.joints_loc);  // unit 3
                bind_tex(gl, &mut unit, self.morph_texture, &self.morph_deltas_loc); // unit 4
                bind_tex(gl, &mut unit, self.morph_wt_texture, &self.morph_wt_loc); // unit 5

                gl.bind_vertex_array(Some(self.vao));
                gl.bind_transform_feedback(glow::TRANSFORM_FEEDBACK, Some(self.tf));
                gl.bind_buffer_base(glow::TRANSFORM_FEEDBACK_BUFFER, 0, Some(self.output_buf));
                gl.enable(glow::RASTERIZER_DISCARD);

                self.bound = true;
            }

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

            // Re-bind per-frame textures (joints + morph weights change each frame)
            if let Some(tex) = self.joints_texture {
                gl.active_texture(glow::TEXTURE3);
                gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            }
            if let Some(tex) = self.morph_wt_texture {
                gl.active_texture(glow::TEXTURE5);
                gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            }

            gl.uniform_4_f32(Some(&self.mob_a_loc), mobius[0], mobius[1], mobius[2], mobius[3]);
            gl.uniform_4_f32(Some(&self.mob_b_loc), mobius[4], mobius[5], mobius[6], mobius[7]);
            gl.uniform_4_f32(Some(&self.mob_c_loc), mobius[8], mobius[9], mobius[10], mobius[11]);
            gl.uniform_4_f32(Some(&self.mob_d_loc), mobius[12], mobius[13], mobius[14], mobius[15]);
            gl.uniform_1_f32(Some(&self.density_loc), density);
            gl.uniform_1_f32(Some(&self.mesh_radius_loc), mesh_radius);
            gl.uniform_1_f32(Some(&self.min_px_loc), min_px);
            gl.uniform_1_f32(Some(&self.max_lod_loc), max_lod);
            gl.uniform_matrix_4_f32_slice(Some(&self.vp_matrix_loc), false, vp_matrix);
            gl.uniform_1_f32(Some(&self.vp_width_loc), vp_width);
            gl.uniform_1_f32(Some(&self.vp_height_loc), vp_height);

            gl.begin_transform_feedback(glow::POINTS);
            gl.draw_arrays(glow::POINTS, 0, n as i32);
            gl.end_transform_feedback();

            gl.flush();
        }
        n
    }

    /// Read back LOD results after compute. Clamps to max_faces.
    pub fn read_back(&self, gl: &glow::Context, num_faces: usize) -> Vec<f32> {
        let size = num_faces.min(self.max_faces) * FLOATS_PER_FACE_OUTPUT;
        let mut result = vec![0.0f32; size];
        unsafe {
            gl.bind_buffer(glow::TRANSFORM_FEEDBACK_BUFFER, Some(self.output_buf));
            gl.get_buffer_sub_data(glow::TRANSFORM_FEEDBACK_BUFFER, 0,
                bytemuck_cast_slice_mut(&mut result));
        }
        result
    }

    pub fn destroy(&self, gl: &glow::Context) {
        unsafe {
            gl.delete_program(self.program);
            gl.delete_vertex_array(self.vao);
            gl.delete_buffer(self.input_buf);
            gl.delete_buffer(self.output_buf);
            gl.delete_transform_feedback(self.tf);
            if let Some(t) = self.pos_texture { gl.delete_texture(t); }
            if let Some(t) = self.lut_texture { gl.delete_texture(t); }
            if let Some(t) = self.skinning_texture { gl.delete_texture(t); }
            if let Some(t) = self.joints_texture { gl.delete_texture(t); }
            if let Some(t) = self.morph_texture { gl.delete_texture(t); }
            if let Some(t) = self.morph_wt_texture { gl.delete_texture(t); }
        }
    }
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
