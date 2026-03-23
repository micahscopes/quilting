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

/// Per-face output: 3 edge LODs + 3 deformed median lengths
/// = 6 floats per face.
pub const FLOATS_PER_FACE_OUTPUT: usize = 6;

/// Transform feedback compute pipeline for LOD calculation.
pub struct LodCompute {
    program: glow::Program,
    vao: glow::VertexArray,
    input_buf: glow::Buffer,
    output_buf: glow::Buffer,
    tf: glow::TransformFeedback,
    mobius_loc: glow::UniformLocation,
    density_loc: glow::UniformLocation,
    mesh_radius_loc: glow::UniformLocation,
    max_faces: usize,
}

const LOD_COMPUTE_VS: &str = r#"#version 300 es
precision highp float;

// Per-face input: 3 control points as quaternions (w,x,y,z)
layout(location = 0) in vec4 cp0;
layout(location = 1) in vec4 cp1;
layout(location = 2) in vec4 cp2;

// Möbius transform: M(q) = (a*q + b) * (c*q + d)^{-1}
// Stored as 4 quaternions
uniform vec4 mob_a;
uniform vec4 mob_b;
uniform vec4 mob_c;
uniform vec4 mob_d;

uniform float density;
uniform float mesh_radius;

// Transform feedback outputs
out float out_lod_a;
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

void main() {
    // Control point positions (xyz from quaternion wxyz)
    vec3 p0 = cp0.yzw;
    vec3 p1 = cp1.yzw;
    vec3 p2 = cp2.yzw;

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

    gl_Position = vec4(0.0); // not rendering, just computing
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
                &["out_lod_a", "out_lod_b", "out_lod_c",
                  "out_median_a", "out_median_b", "out_median_c"],
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
            let mobius_loc = gl.get_uniform_location(program, "mob_a")
                .ok_or("mob_a uniform not found")?;
            let density_loc = gl.get_uniform_location(program, "density")
                .ok_or("density uniform not found")?;
            let mesh_radius_loc = gl.get_uniform_location(program, "mesh_radius")
                .ok_or("mesh_radius uniform not found")?;

            // Create VAO + buffers
            let vao = gl.create_vertex_array().map_err(|e| format!("{e}"))?;
            gl.bind_vertex_array(Some(vao));

            let input_buf = gl.create_buffer().map_err(|e| format!("{e}"))?;
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(input_buf));
            gl.buffer_data_size(glow::ARRAY_BUFFER,
                (max_faces * FLOATS_PER_FACE_INPUT * 4) as i32,
                glow::DYNAMIC_DRAW);

            // 3 vec4 attributes: cp0, cp1, cp2
            for i in 0..3u32 {
                gl.enable_vertex_attrib_array(i);
                gl.vertex_attrib_pointer_f32(i, 4, glow::FLOAT, false,
                    (FLOATS_PER_FACE_INPUT * 4) as i32, (i * 16) as i32);
            }

            let output_buf = gl.create_buffer().map_err(|e| format!("{e}"))?;
            gl.bind_buffer(glow::TRANSFORM_FEEDBACK_BUFFER, Some(output_buf));
            gl.buffer_data_size(glow::TRANSFORM_FEEDBACK_BUFFER,
                (max_faces * FLOATS_PER_FACE_OUTPUT * 4) as i32,
                glow::DYNAMIC_READ);

            let tf = gl.create_transform_feedback().map_err(|e| format!("{e}"))?;

            gl.bind_vertex_array(None);

            Ok(Self {
                program, vao, input_buf, output_buf, tf,
                mobius_loc, density_loc, mesh_radius_loc,
                max_faces,
            })
        }
    }

    /// Upload per-face control points (3 quaternions × 4 floats = 12 floats per face).
    pub fn upload_control_points(&self, gl: &glow::Context, data: &[f32]) {
        unsafe {
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.input_buf));
            gl.buffer_sub_data_u8_slice(glow::ARRAY_BUFFER, 0,
                bytemuck_cast_slice(data));
        }
    }

    /// Run the compute pass. Returns the number of faces processed.
    pub fn compute(
        &self,
        gl: &glow::Context,
        num_faces: usize,
        mobius: [f32; 16], // a, b, c, d quaternions packed
        density: f32,
        mesh_radius: f32,
    ) -> usize {
        let n = num_faces.min(self.max_faces);
        unsafe {
            gl.use_program(Some(self.program));

            // Set Möbius uniforms
            gl.uniform_4_f32(Some(&self.mobius_loc),
                mobius[0], mobius[1], mobius[2], mobius[3]); // mob_a
            // TODO: set mob_b, mob_c, mob_d uniforms

            gl.uniform_1_f32(Some(&self.density_loc), density);
            gl.uniform_1_f32(Some(&self.mesh_radius_loc), mesh_radius);

            gl.bind_vertex_array(Some(self.vao));

            gl.bind_transform_feedback(glow::TRANSFORM_FEEDBACK, Some(self.tf));
            gl.bind_buffer_base(glow::TRANSFORM_FEEDBACK_BUFFER, 0, Some(self.output_buf));

            gl.enable(glow::RASTERIZER_DISCARD); // no rendering, just compute
            gl.begin_transform_feedback(glow::POINTS);
            gl.draw_arrays(glow::POINTS, 0, n as i32);
            gl.end_transform_feedback();
            gl.disable(glow::RASTERIZER_DISCARD);

            gl.bind_transform_feedback(glow::TRANSFORM_FEEDBACK, None);
            gl.bind_vertex_array(None);
        }
        n
    }

    /// Read back the LOD results from GPU. Returns [lod_a, lod_b, lod_c, med_a, med_b, med_c] per face.
    pub fn read_back(&self, gl: &glow::Context, num_faces: usize) -> Vec<f32> {
        let size = num_faces * FLOATS_PER_FACE_OUTPUT;
        let mut result = vec![0.0f32; size];
        unsafe {
            gl.bind_buffer(glow::TRANSFORM_FEEDBACK_BUFFER, Some(self.output_buf));
            let data = gl.map_buffer_range(
                glow::TRANSFORM_FEEDBACK_BUFFER,
                0,
                (size * 4) as i32,
                glow::MAP_READ_BIT,
            );
            if !data.is_null() {
                std::ptr::copy_nonoverlapping(
                    data as *const f32,
                    result.as_mut_ptr(),
                    size,
                );
                gl.unmap_buffer(glow::TRANSFORM_FEEDBACK_BUFFER);
            }
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

// Safe cast for &[f32] → &[u8]
fn bytemuck_cast_slice(data: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
    }
}
