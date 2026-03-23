//! GPU compute via transform feedback (WebGL2 GPGPU).
//!
//! Runs a vertex shader as a compute kernel — one "vertex" per face.
//! Transform feedback captures the output into a GPU buffer.
//! Used for LOD computation: Möbius-transform edge midpoints → deformed
//! medians → LOD per edge. Result stays on GPU or is read back (~20KB).

use glow::HasContext;

/// Per-face input: 3 control point positions (quaternions w,x,y,z)
/// = 12 floats per face.
pub const FLOATS_PER_FACE_INPUT: usize = 12;

/// Pre-baked animation sequence stored on GPU.
/// All frames' vertex positions packed into one buffer.
/// GPU indexes by frame_index * num_vertices + vertex_index.
pub struct AnimationSequence {
    pub num_frames: usize,
    pub num_vertices: usize,
    pub num_faces: usize,
    /// Flat f32 buffer: [frame0_vert0_x, y, z, frame0_vert1_x, ...]
    /// Size: num_frames * num_vertices * 3
    pub positions: Vec<f32>,
    /// Face indices: [face0_v0, v1, v2, face1_v0, ...] — 3 ints per face
    pub face_indices: Vec<u32>,
}

/// Per-face output: atlas_index + perm_index = 2 floats.
pub const FLOATS_PER_FACE_OUTPUT: usize = 2;

/// Transform feedback compute pipeline for LOD calculation.
pub struct LodCompute {
    program: glow::Program,
    vao: glow::VertexArray,
    input_buf: glow::Buffer,  // face indices (3 floats per face)
    output_buf: glow::Buffer,
    tf: glow::TransformFeedback,
    pos_texture: Option<glow::Texture>, // prebaked positions
    lut_texture: Option<glow::Texture>, // exponent triple → atlas index
    lut_loc: Option<glow::UniformLocation>,
    mob_a_loc: glow::UniformLocation,
    mob_b_loc: glow::UniformLocation,
    mob_c_loc: glow::UniformLocation,
    mob_d_loc: glow::UniformLocation,
    density_loc: glow::UniformLocation,
    mesh_radius_loc: glow::UniformLocation,
    min_px_loc: glow::UniformLocation,
    vp_matrix_loc: glow::UniformLocation,
    vp_width_loc: glow::UniformLocation,
    vp_height_loc: glow::UniformLocation,
    num_verts_loc: glow::UniformLocation,
    frame_loc: glow::UniformLocation,
    max_faces: usize,
}

const LOD_COMPUTE_VS: &str = r#"#version 300 es
precision highp float;

// Per-face input: 3 vertex indices (packed as ivec3 via float attrib)
layout(location = 0) in vec3 face_indices;

// All vertex positions for all frames: sampled via texelFetch
// Layout: texel (vertex_index + frame * num_vertices) = vec4(x, y, z, 0)
uniform highp sampler2D u_positions;
uniform int u_num_vertices;
uniform int u_frame;

// Möbius transform: M(q) = (a*q + b) * (c*q + d)^{-1}
// Stored as 4 quaternions
uniform vec4 mob_a;
uniform vec4 mob_b;
uniform vec4 mob_c;
uniform vec4 mob_d;

uniform float density;
uniform float mesh_radius;
uniform float min_px;        // minimum pixels per subdivision (0 = no attenuation)
uniform mat4 vp_matrix;      // view-projection matrix for screen-space projection
uniform float vp_width;      // viewport width in pixels
uniform float vp_height;     // viewport height in pixels
uniform highp sampler2D u_atlas_lut; // exponent triple → atlas index (32×32 R8)

// Transform feedback outputs — just classification, no LODs
out float out_atlas_index;
out float out_perm_index;
out float out_lod_b;
out float out_lod_c;
out float out_median_a;
out float out_median_b;
out float out_median_c;

// Quaternion multiply
vec4 qmul(vec4 a, vec4 b) {
    return vec4(
        a.x*b.x - a.y*b.y - a.z*b.z - a.w*b.w,
        a.x*b.y + a.y*b.x + a.z*b.w - a.w*b.z,
        a.x*b.z - a.y*b.w + a.z*b.x + a.w*b.y,
        a.x*b.w + a.y*b.z - a.z*b.y + a.w*b.x
    );
}

// Quaternion inverse: conj(q) / |q|^2
vec4 qinv(vec4 q) {
    float n = dot(q, q);
    return vec4(q.x, -q.yzw) / max(n, 1e-20);
}

// Möbius transform of a pure quaternion point
vec3 mobius(vec3 p) {
    vec4 q = vec4(0.0, p);
    vec4 top = qmul(mob_a, q) + mob_b;  // a*q + b  (but a,b are quat not vec4... need proper encoding)
    vec4 bot = qmul(mob_c, q) + mob_d;  // c*q + d
    vec4 result = qmul(top, qinv(bot));
    return result.yzw;
}

// Snap to nearest power of 2 (simplified — no hysteresis)
float snap_pow2(float v) {
    return exp2(round(log2(max(v, 1.0))));
}

vec3 fetch_pos(int vertex_id) {
    int idx = vertex_id + u_frame * u_num_vertices;
    // 2D texture: pack linearly, width = 4096
    int tx = idx % 4096;
    int ty = idx / 4096;
    return texelFetch(u_positions, ivec2(tx, ty), 0).xyz;
}

void main() {
    // Fetch vertex positions from prebaked texture
    vec3 p0 = fetch_pos(int(face_indices.x));
    vec3 p1 = fetch_pos(int(face_indices.y));
    vec3 p2 = fetch_pos(int(face_indices.z));

    // Edge midpoints in world space
    vec3 mid_a = (p1 + p2) * 0.5;  // midpoint of edge BC (opposite v0)
    vec3 mid_b = (p0 + p2) * 0.5;  // midpoint of edge AC (opposite v1)
    vec3 mid_c = (p0 + p1) * 0.5;  // midpoint of edge AB (opposite v2)

    // Deformed positions
    vec3 d0 = mobius(p0);
    vec3 d1 = mobius(p1);
    vec3 d2 = mobius(p2);
    vec3 dm_a = mobius(mid_a);
    vec3 dm_b = mobius(mid_b);
    vec3 dm_c = mobius(mid_c);

    // Deformed medians
    out_median_a = distance(d0, dm_a);
    out_median_b = distance(d1, dm_b);
    out_median_c = distance(d2, dm_c);

    float target_size = mesh_radius / density;

    // LODs from medians (average of two adjacent medians per edge)
    float raw_a = (out_median_b + out_median_c) * 0.5 / target_size;
    float raw_b = (out_median_a + out_median_c) * 0.5 / target_size;
    float raw_c = (out_median_a + out_median_b) * 0.5 / target_size;

    out_lod_a = clamp(snap_pow2(raw_a), 2.0, 512.0);
    out_lod_b = clamp(snap_pow2(raw_b), 2.0, 512.0);
    out_lod_c = clamp(snap_pow2(raw_c), 2.0, 512.0);

    // Screen-space attenuation: project deformed vertices to screen,
    // measure per-edge pixel length, reduce LOD when sub-pixel.
    if (min_px > 0.0) {
        // Project deformed positions to screen pixels
        vec4 c0 = vp_matrix * vec4(d0, 1.0);
        vec4 c1 = vp_matrix * vec4(d1, 1.0);
        vec4 c2 = vp_matrix * vec4(d2, 1.0);
        vec2 s0 = (c0.xy / max(abs(c0.w), 0.001)) * vec2(vp_width, vp_height) * 0.5;
        vec2 s1 = (c1.xy / max(abs(c1.w), 0.001)) * vec2(vp_width, vp_height) * 0.5;
        vec2 s2 = (c2.xy / max(abs(c2.w), 0.001)) * vec2(vp_width, vp_height) * 0.5;

        // Screen-space edge lengths
        float px_a = distance(s1, s2); // edge BC
        float px_b = distance(s0, s2); // edge AC
        float px_c = distance(s0, s1); // edge AB

        // Attenuate: if pixels/LOD < min_px, reduce LOD
        if (px_a / out_lod_a < min_px) out_lod_a = clamp(snap_pow2(px_a / min_px), 2.0, 512.0);
        if (px_b / out_lod_b < min_px) out_lod_b = clamp(snap_pow2(px_b / min_px), 2.0, 512.0);
        if (px_c / out_lod_c < min_px) out_lod_c = clamp(snap_pow2(px_c / min_px), 2.0, 512.0);
    }

    // Compute atlas index from LOD exponents
    int ea = int(log2(out_lod_a));
    int eb = int(log2(out_lod_b));
    int ec = int(log2(out_lod_c));

    // Sort to canonical form + determine S3 permutation
    int sa, sb, sc;
    int perm;
    if (ea <= eb && eb <= ec)      { sa=ea; sb=eb; sc=ec; perm=0; }
    else if (ea <= ec && ec <= eb)  { sa=ea; sb=ec; sc=eb; perm=1; }
    else if (eb <= ea && ea <= ec)  { sa=eb; sb=ea; sc=ec; perm=2; }
    else if (eb <= ec && ec <= ea)  { sa=eb; sb=ec; sc=ea; perm=3; }
    else if (ec <= ea && ea <= eb)  { sa=ec; sb=ea; sc=eb; perm=4; }
    else                            { sa=ec; sb=eb; sc=ea; perm=5; }

    // LUT lookup: key = sa + sb*10 + sc*100, stored in 32×32 texture
    int key = sa + sb * 10 + sc * 100;
    int lut_x = key % 32;
    int lut_y = key / 32;
    out_atlas_index = texelFetch(u_atlas_lut, ivec2(lut_x, lut_y), 0).r * 255.0;
    out_perm_index = float(perm);

    gl_Position = vec4(0.0);
}
"#;

// Minimal fragment shader (required by WebGL2 even with transform feedback)
const LOD_COMPUTE_FS: &str = r#"#version 300 es
precision lowp float;
out vec4 dummy;
void main() { dummy = vec4(0.0); }
"#;

impl LodCompute {
    /// Create the LOD compute pipeline.
    pub fn new(gl: &glow::Context, max_faces: usize) -> Result<Self, String> {
        unsafe {
            // Compile shaders
            let vs = compile_shader(gl, glow::VERTEX_SHADER, LOD_COMPUTE_VS)?;
            let fs = compile_shader(gl, glow::FRAGMENT_SHADER, LOD_COMPUTE_FS)?;

            let program = gl.create_program().map_err(|e| format!("{e}"))?;
            gl.attach_shader(program, vs);
            gl.attach_shader(program, fs);

            // Set transform feedback varyings BEFORE linking
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

            // Get uniform locations
            let mob_a_loc = gl.get_uniform_location(program, "mob_a")
                .ok_or("mob_a uniform not found")?;
            let mob_b_loc = gl.get_uniform_location(program, "mob_b")
                .ok_or("mob_b uniform not found")?;
            let mob_c_loc = gl.get_uniform_location(program, "mob_c")
                .ok_or("mob_c uniform not found")?;
            let mob_d_loc = gl.get_uniform_location(program, "mob_d")
                .ok_or("mob_d uniform not found")?;
            let density_loc = gl.get_uniform_location(program, "density")
                .ok_or("density uniform not found")?;
            let mesh_radius_loc = gl.get_uniform_location(program, "mesh_radius")
                .ok_or("mesh_radius uniform not found")?;
            let min_px_loc = gl.get_uniform_location(program, "min_px")
                .ok_or("min_px uniform not found")?;
            let vp_matrix_loc = gl.get_uniform_location(program, "vp_matrix")
                .ok_or("vp_matrix uniform not found")?;
            let vp_width_loc = gl.get_uniform_location(program, "vp_width")
                .ok_or("vp_width uniform not found")?;
            let vp_height_loc = gl.get_uniform_location(program, "vp_height")
                .ok_or("vp_height uniform not found")?;
            let num_verts_loc = gl.get_uniform_location(program, "u_num_vertices")
                .ok_or("u_num_vertices uniform not found")?;
            let frame_loc = gl.get_uniform_location(program, "u_frame")
                .ok_or("u_frame uniform not found")?;

            // Create VAO + buffers
            let vao = gl.create_vertex_array().map_err(|e| format!("{e}"))?;
            gl.bind_vertex_array(Some(vao));

            let input_buf = gl.create_buffer().map_err(|e| format!("{e}"))?;
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(input_buf));
            gl.buffer_data_size(glow::ARRAY_BUFFER,
                (max_faces * 3 * 4) as i32, // 3 floats (face indices) per face
                glow::STATIC_DRAW);

            // 1 vec3 attribute: face_indices
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
                pos_texture: None,
                lut_texture: None,
                lut_loc: gl.get_uniform_location(program, "u_atlas_lut"),
                mob_a_loc, mob_b_loc, mob_c_loc, mob_d_loc,
                density_loc, mesh_radius_loc,
                min_px_loc, vp_matrix_loc, vp_width_loc, vp_height_loc,
                num_verts_loc, frame_loc,
                max_faces,
            })
        }
    }

    /// Upload atlas LUT: maps exponent triples to atlas indices.
    /// `lut`: 1024 u8 values. key = exp_a + exp_b*10 + exp_c*100.
    /// Stored as 32×32 R8 texture.
    pub fn upload_atlas_lut(&mut self, gl: &glow::Context, lut: &[u8]) {
        unsafe {
            if let Some(old) = self.lut_texture { gl.delete_texture(old); }
            let tex = gl.create_texture().unwrap();
            gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            // Pad to 32×32 = 1024
            let mut data = vec![255u8; 1024];
            for (i, &v) in lut.iter().take(1024).enumerate() { data[i] = v; }
            gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::R8 as i32,
                32, 32, 0, glow::RED, glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(&data)));
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);
            self.lut_texture = Some(tex);
        }
    }

    /// Upload face indices (3 floats per face — vertex indices as floats for attrib).
    pub fn upload_face_indices(&self, gl: &glow::Context, indices: &[f32]) {
        unsafe {
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.input_buf));
            gl.buffer_data_u8_slice(glow::ARRAY_BUFFER,
                bytemuck_cast_slice(indices), glow::STATIC_DRAW);
        }
    }

    /// Upload prebaked positions as a float texture.
    /// `positions`: flat [x,y,z, x,y,z, ...] for all vertices × all frames.
    /// Packed into a 4096-wide RGBA32F texture.
    pub fn upload_positions_texture(&mut self, gl: &glow::Context, positions: &[f32], num_vertices: usize, num_frames: usize) {
        unsafe {
            // Pack xyz into RGBA (w=0) for texelFetch
            let total = num_vertices * num_frames;
            let mut rgba = vec![0.0f32; total * 4];
            for i in 0..total {
                rgba[i*4]   = positions[i*3];
                rgba[i*4+1] = positions[i*3+1];
                rgba[i*4+2] = positions[i*3+2];
                rgba[i*4+3] = 0.0;
            }

            let width = 4096;
            let height = (total + width - 1) / width;

            // Pad to full texture size
            rgba.resize(width * height * 4, 0.0);

            if let Some(old) = self.pos_texture { gl.delete_texture(old); }
            let tex = gl.create_texture().unwrap();
            gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::RGBA32F as i32,
                width as i32, height as i32, 0,
                glow::RGBA, glow::FLOAT,
                glow::PixelUnpackData::Slice(Some(bytemuck_cast_slice(&rgba))));
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
            self.pos_texture = Some(tex);
        }
    }

    /// Upload per-face control points (3 quaternions × 4 floats = 12 floats per face).
    /// DEPRECATED: use upload_face_indices + upload_positions_texture instead.
    pub fn upload_control_points(&self, gl: &glow::Context, data: &[f32]) {
        unsafe {
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.input_buf));
            gl.buffer_sub_data_u8_slice(glow::ARRAY_BUFFER, 0,
                bytemuck_cast_slice(data));
        }
    }

    /// Run the compute pass (legacy — uses uploaded control points directly).
    pub fn compute(
        &self, gl: &glow::Context, num_faces: usize,
        mobius: [f32; 16], density: f32, mesh_radius: f32,
        min_px: f32, vp_matrix: &[f32; 16], vp_width: f32, vp_height: f32,
    ) -> usize {
        self.compute_with_texture(gl, num_faces, 0, 0, mobius, density, mesh_radius, min_px, vp_matrix, vp_width, vp_height)
    }

    /// Run the compute pass with prebaked position texture.
    /// `frame`: which animation frame to read from the texture.
    /// `num_vertices`: vertices per frame (for texture indexing).
    pub fn compute_with_texture(
        &self,
        gl: &glow::Context,
        num_faces: usize,
        frame: u32,
        num_vertices: u32,
        mobius: [f32; 16],
        density: f32,
        mesh_radius: f32,
        min_px: f32,
        vp_matrix: &[f32; 16],
        vp_width: f32,
        vp_height: f32,
    ) -> usize {
        let n = num_faces.min(self.max_faces);
        unsafe {
            gl.use_program(Some(self.program));

            if let Some(tex) = self.pos_texture {
                gl.active_texture(glow::TEXTURE0);
                gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            }
            if let Some(tex) = self.lut_texture {
                gl.active_texture(glow::TEXTURE1);
                gl.bind_texture(glow::TEXTURE_2D, Some(tex));
                if let Some(ref loc) = self.lut_loc {
                    gl.uniform_1_i32(Some(loc), 1); // texture unit 1
                }
            }

            gl.uniform_1_i32(Some(&self.frame_loc), frame as i32);
            gl.uniform_1_i32(Some(&self.num_verts_loc), num_vertices as i32);
            gl.uniform_4_f32(Some(&self.mob_a_loc), mobius[0], mobius[1], mobius[2], mobius[3]);
            gl.uniform_4_f32(Some(&self.mob_b_loc), mobius[4], mobius[5], mobius[6], mobius[7]);
            gl.uniform_4_f32(Some(&self.mob_c_loc), mobius[8], mobius[9], mobius[10], mobius[11]);
            gl.uniform_4_f32(Some(&self.mob_d_loc), mobius[12], mobius[13], mobius[14], mobius[15]);

            gl.uniform_1_f32(Some(&self.density_loc), density);
            gl.uniform_1_f32(Some(&self.mesh_radius_loc), mesh_radius);
            gl.uniform_1_f32(Some(&self.min_px_loc), min_px);
            gl.uniform_matrix_4_f32_slice(Some(&self.vp_matrix_loc), false, vp_matrix);
            gl.uniform_1_f32(Some(&self.vp_width_loc), vp_width);
            gl.uniform_1_f32(Some(&self.vp_height_loc), vp_height);

            gl.bind_vertex_array(Some(self.vao));
            gl.bind_transform_feedback(glow::TRANSFORM_FEEDBACK, Some(self.tf));
            gl.bind_buffer_base(glow::TRANSFORM_FEEDBACK_BUFFER, 0, Some(self.output_buf));
            gl.enable(glow::RASTERIZER_DISCARD);

            gl.begin_transform_feedback(glow::POINTS);
            gl.draw_arrays(glow::POINTS, 0, n as i32);
            gl.end_transform_feedback();

            gl.disable(glow::RASTERIZER_DISCARD);
            gl.bind_transform_feedback(glow::TRANSFORM_FEEDBACK, None);
            gl.bind_vertex_array(None);

            // Insert fence — GPU can work while CPU does other things.
            // Call read_back() later (after instance packing) to minimize stall.
            gl.flush();
        }
        n
    }

    /// Wait for GPU compute to finish, then read back results.
    /// Call this as LATE as possible — do CPU work between compute() and read_back().
    pub fn read_back(&self, gl: &glow::Context, num_faces: usize) -> Vec<f32> {
        let size = num_faces * FLOATS_PER_FACE_OUTPUT;
        let mut result = vec![0.0f32; size];
        unsafe {
            // Fence sync — wait for transform feedback to complete.
            // flush() in compute() kicked off the GPU work, this waits for it.
            let fence = gl.fence_sync(glow::SYNC_GPU_COMMANDS_COMPLETE, 0).unwrap();
            gl.client_wait_sync(fence, glow::SYNC_FLUSH_COMMANDS_BIT, 1_000_000_000); // 1s timeout
            gl.delete_sync(fence);

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
        }
    }
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

fn bytemuck_cast_slice(data: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) }
}

fn bytemuck_cast_slice_mut(data: &mut [f32]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, data.len() * 4) }
}
