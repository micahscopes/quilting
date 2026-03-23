//! GPU buffer management: VAO, VBO, IBO, instance buffers, and UBOs.
//!
//! Mirrors the index.html buffer setup:
//! - Vertex attribute 0: bary coords (vec3, per-vertex)
//! - Attributes 1-13: instance data (13 x vec4 = 52 floats per instance)
//!   [p0, p1, p2, w0, w1, w2, edge_lods, vertex_lods, uv01, uv2_pad, n0, n1, n2]

use glow::HasContext;

/// Per-batch tessellation geometry (bary coords + indices).
/// Cached by LOD triple -- these never change, only instance data changes per frame.
pub struct TessBuffers {
    pub bary_buf: glow::Buffer,
    pub tri_index_buf: glow::Buffer,
    pub line_index_buf: glow::Buffer,
    pub num_tri_indices: i32,
    pub num_line_indices: i32,
}

impl TessBuffers {
    /// Upload tessellation geometry from flat arrays.
    pub fn new(
        gl: &glow::Context,
        bary_data: &[f32],
        tri_indices: &[u32],
        line_indices: &[u32],
    ) -> Result<Self, String> {
        unsafe {
            let bary_buf = gl.create_buffer().map_err(|e| format!("bary buf: {e}"))?;
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(bary_buf));
            gl.buffer_data_u8_slice(
                glow::ARRAY_BUFFER,
                bytemuck_cast_slice(bary_data),
                glow::STATIC_DRAW,
            );

            let tri_index_buf = gl.create_buffer().map_err(|e| format!("tri idx buf: {e}"))?;
            gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(tri_index_buf));
            gl.buffer_data_u8_slice(
                glow::ELEMENT_ARRAY_BUFFER,
                bytemuck_cast_slice(tri_indices),
                glow::STATIC_DRAW,
            );

            let line_index_buf = gl.create_buffer().map_err(|e| format!("line idx buf: {e}"))?;
            gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(line_index_buf));
            gl.buffer_data_u8_slice(
                glow::ELEMENT_ARRAY_BUFFER,
                bytemuck_cast_slice(line_indices),
                glow::STATIC_DRAW,
            );

            Ok(TessBuffers {
                bary_buf,
                tri_index_buf,
                line_index_buf,
                num_tri_indices: tri_indices.len() as i32,
                num_line_indices: line_indices.len() as i32,
            })
        }
    }

    pub fn destroy(&self, gl: &glow::Context) {
        unsafe {
            gl.delete_buffer(self.bary_buf);
            gl.delete_buffer(self.tri_index_buf);
            gl.delete_buffer(self.line_index_buf);
        }
    }
}

/// A renderable batch: VAO with tessellation geometry + instanced data.
pub struct MeshBuffers {
    pub tri_vao: glow::VertexArray,
    pub line_vao: glow::VertexArray,
    pub instance_buf: glow::Buffer,
    pub instance_buf_capacity: usize,
    pub num_tri_indices: i32,
    pub num_line_indices: i32,
    pub num_instances: i32,
}

impl MeshBuffers {
    /// Create VAOs for triangle and line rendering, sharing tessellation + instance buffers.
    ///
    /// Vertex layout matches index.html:
    /// - location 0: vec3 bary (per-vertex from tess buffer)
    /// - locations 1-13: vec4 x 13 (per-instance, 208 bytes stride, divisor=1)
    pub fn new(
        gl: &glow::Context,
        tess: &TessBuffers,
        instance_data: &[f32],
        num_instances: i32,
    ) -> Result<Self, String> {
        unsafe {
            // Create instance buffer
            let instance_buf = gl.create_buffer().map_err(|e| format!("instance buf: {e}"))?;
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(instance_buf));
            gl.buffer_data_u8_slice(
                glow::ARRAY_BUFFER,
                bytemuck_cast_slice(instance_data),
                glow::DYNAMIC_DRAW,
            );
            let capacity = instance_data.len();

            // Triangle VAO
            let tri_vao = gl.create_vertex_array().map_err(|e| format!("tri vao: {e}"))?;
            setup_vao(gl, tri_vao, &tess.bary_buf, &tess.tri_index_buf, &instance_buf);

            // Line VAO
            let line_vao = gl.create_vertex_array().map_err(|e| format!("line vao: {e}"))?;
            setup_vao(gl, line_vao, &tess.bary_buf, &tess.line_index_buf, &instance_buf);

            gl.bind_vertex_array(None);

            Ok(MeshBuffers {
                tri_vao,
                line_vao,
                instance_buf,
                instance_buf_capacity: capacity,
                num_tri_indices: tess.num_tri_indices,
                num_line_indices: tess.num_line_indices,
                num_instances,
            })
        }
    }

    /// Re-upload instance data without recreating VAOs.
    pub fn update_instances(
        &mut self,
        gl: &glow::Context,
        instance_data: &[f32],
        num_instances: i32,
    ) {
        unsafe {
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.instance_buf));
            if instance_data.len() <= self.instance_buf_capacity {
                gl.buffer_sub_data_u8_slice(
                    glow::ARRAY_BUFFER,
                    0,
                    bytemuck_cast_slice(instance_data),
                );
            } else {
                gl.buffer_data_u8_slice(
                    glow::ARRAY_BUFFER,
                    bytemuck_cast_slice(instance_data),
                    glow::DYNAMIC_DRAW,
                );
                self.instance_buf_capacity = instance_data.len();
            }
            self.num_instances = num_instances;
        }
    }

    pub fn destroy(&self, gl: &glow::Context) {
        unsafe {
            gl.delete_vertex_array(self.tri_vao);
            gl.delete_vertex_array(self.line_vao);
            gl.delete_buffer(self.instance_buf);
        }
    }
}

/// Configure a VAO with the quilting vertex layout.
unsafe fn setup_vao(
    gl: &glow::Context,
    vao: glow::VertexArray,
    bary_buf: &glow::Buffer,
    index_buf: &glow::Buffer,
    instance_buf: &glow::Buffer,
) {
    gl.bind_vertex_array(Some(vao));

    // Attribute 0: bary coords (vec3, per-vertex)
    gl.bind_buffer(glow::ARRAY_BUFFER, Some(*bary_buf));
    gl.enable_vertex_attrib_array(0);
    gl.vertex_attrib_pointer_f32(0, 3, glow::FLOAT, false, 0, 0);

    // Attributes 1-13: instance data (13 x vec4, 208 bytes stride)
    gl.bind_buffer(glow::ARRAY_BUFFER, Some(*instance_buf));
    for i in 0..13u32 {
        let loc = 1 + i;
        gl.enable_vertex_attrib_array(loc);
        gl.vertex_attrib_pointer_f32(loc, 4, glow::FLOAT, false, 208, (i * 16) as i32);
        gl.vertex_attrib_divisor(loc, 1);
    }

    // Index buffer
    gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(*index_buf));

    gl.bind_vertex_array(None);
}

/// UBO for vertex shader uniforms (matches naga's Uniforms struct).
///
/// Layout (std140):
///   mat4x4 mvp       (offset 0, 64 bytes)
///   mat4x4 mv        (offset 64, 64 bytes)
///   float perm_parity (offset 128, 4 bytes)
///   int perm_index    (offset 132, 4 bytes)
///   int use_qb        (offset 136, 4 bytes)
///   float _pad        (offset 140, 4 bytes)
///   Total: 144 bytes
pub struct VertexUniformBuf {
    pub ubo: glow::Buffer,
}

impl VertexUniformBuf {
    pub fn new(gl: &glow::Context) -> Result<Self, String> {
        unsafe {
            let ubo = gl.create_buffer().map_err(|e| format!("vtx ubo: {e}"))?;
            gl.bind_buffer(glow::UNIFORM_BUFFER, Some(ubo));
            // Allocate 144 bytes
            gl.buffer_data_size(glow::UNIFORM_BUFFER, 144, glow::DYNAMIC_DRAW);
            Ok(VertexUniformBuf { ubo })
        }
    }

    /// Upload uniform data. `mvp` and `mv` are column-major 4x4 matrices.
    pub fn upload(
        &self,
        gl: &glow::Context,
        mvp: &[f32; 16],
        mv: &[f32; 16],
        perm_parity: f32,
        perm_index: i32,
        use_qb: i32,
    ) {
        // Pack into 144 bytes (36 f32s)
        let mut data = [0u8; 144];
        // mvp: 64 bytes at offset 0
        data[0..64].copy_from_slice(bytemuck_cast_slice(mvp));
        // mv: 64 bytes at offset 64
        data[64..128].copy_from_slice(bytemuck_cast_slice(mv));
        // perm_parity: f32 at offset 128
        data[128..132].copy_from_slice(&perm_parity.to_le_bytes());
        // perm_index: i32 at offset 132
        data[132..136].copy_from_slice(&perm_index.to_le_bytes());
        // use_qb: i32 at offset 136
        data[136..140].copy_from_slice(&use_qb.to_le_bytes());
        // _pad: f32 at offset 140 (already zeroed)

        unsafe {
            gl.bind_buffer(glow::UNIFORM_BUFFER, Some(self.ubo));
            gl.buffer_sub_data_u8_slice(glow::UNIFORM_BUFFER, 0, &data);
        }
    }

    /// Bind this UBO to the vertex uniforms binding point.
    pub fn bind(&self, gl: &glow::Context) {
        unsafe {
            gl.bind_buffer_base(
                glow::UNIFORM_BUFFER,
                crate::shader::VERTEX_UNIFORMS_BINDING,
                Some(self.ubo),
            );
        }
    }

    pub fn destroy(&self, gl: &glow::Context) {
        unsafe {
            gl.delete_buffer(self.ubo);
        }
    }
}

/// UBO for wire fragment uniforms (matches naga's WireUniforms struct).
///
/// Layout (std140):
///   vec3 color        (offset 0, 12 bytes)
///   int show_density  (offset 12, 4 bytes)
///   Total: 16 bytes
pub struct WireUniformBuf {
    pub ubo: glow::Buffer,
}

impl WireUniformBuf {
    pub fn new(gl: &glow::Context) -> Result<Self, String> {
        unsafe {
            let ubo = gl.create_buffer().map_err(|e| format!("wire ubo: {e}"))?;
            gl.bind_buffer(glow::UNIFORM_BUFFER, Some(ubo));
            gl.buffer_data_size(glow::UNIFORM_BUFFER, 16, glow::DYNAMIC_DRAW);
            Ok(WireUniformBuf { ubo })
        }
    }

    pub fn upload(&self, gl: &glow::Context, color: [f32; 3], show_density: bool) {
        let mut data = [0u8; 16];
        data[0..4].copy_from_slice(&color[0].to_le_bytes());
        data[4..8].copy_from_slice(&color[1].to_le_bytes());
        data[8..12].copy_from_slice(&color[2].to_le_bytes());
        let density_i32: i32 = if show_density { 1 } else { 0 };
        data[12..16].copy_from_slice(&density_i32.to_le_bytes());

        unsafe {
            gl.bind_buffer(glow::UNIFORM_BUFFER, Some(self.ubo));
            gl.buffer_sub_data_u8_slice(glow::UNIFORM_BUFFER, 0, &data);
        }
    }

    pub fn bind(&self, gl: &glow::Context) {
        unsafe {
            gl.bind_buffer_base(
                glow::UNIFORM_BUFFER,
                crate::shader::WIRE_UNIFORMS_BINDING,
                Some(self.ubo),
            );
        }
    }

    pub fn destroy(&self, gl: &glow::Context) {
        unsafe {
            gl.delete_buffer(self.ubo);
        }
    }
}

/// UBO for skeletal animation joint matrices.
///
/// Stores up to MAX_JOINTS skin matrices (world * inverse_bind), uploaded per frame
/// from AnimationEvaluator. Each mat4 is 64 bytes (std140).
///
/// Layout (std140):
///   i32  num_joints  (offset 0, 4 bytes)
///   pad              (offset 4, 12 bytes — std140 alignment to vec4)
///   mat4 joints[N]   (offset 16, 64 bytes each)
pub struct JointMatricesBuf {
    pub ubo: glow::Buffer,
    /// Current allocation size in bytes.
    capacity: usize,
}

/// Maximum joints supported. 128 × 64 bytes = 8KB, well within WebGL2's 16KB UBO limit.
pub const MAX_JOINTS: usize = 128;

/// Total UBO size: 16 bytes header + MAX_JOINTS × 64 bytes + 16 × morph weight vec4s.
const JOINT_UBO_SIZE: usize = 16 + MAX_JOINTS * 64 + 16 * 16;

impl JointMatricesBuf {
    pub fn new(gl: &glow::Context) -> Result<Self, String> {
        unsafe {
            let ubo = gl.create_buffer().map_err(|e| format!("joint ubo: {e}"))?;
            gl.bind_buffer(glow::UNIFORM_BUFFER, Some(ubo));
            gl.buffer_data_size(glow::UNIFORM_BUFFER, JOINT_UBO_SIZE as i32, glow::DYNAMIC_DRAW);
            // Zero-init: num_joints = 0
            let zeros = vec![0u8; JOINT_UBO_SIZE];
            gl.buffer_sub_data_u8_slice(glow::UNIFORM_BUFFER, 0, &zeros);
            Ok(JointMatricesBuf { ubo, capacity: JOINT_UBO_SIZE })
        }
    }

    /// Upload joint matrices from a flat f32 slice (num_joints × 16 floats).
    /// The matrices are skin matrices (joint_world × inverse_bind), column-major.
    pub fn upload(&self, gl: &glow::Context, matrices: &[f32]) {
        let num_joints = matrices.len() / 16;
        if num_joints == 0 { return; }
        let clamped = num_joints.min(MAX_JOINTS);

        // Pack header (16 bytes) + matrices
        let data_size = 16 + clamped * 64;
        let mut data = vec![0u8; data_size];
        // num_joints at offset 0
        data[0..4].copy_from_slice(&(clamped as i32).to_le_bytes());
        // Matrices at offset 16 (std140: mat4 already 64-byte aligned)
        let mat_bytes = &matrices[..clamped * 16];
        data[16..16 + clamped * 64].copy_from_slice(bytemuck_cast_slice(mat_bytes));

        unsafe {
            gl.bind_buffer(glow::UNIFORM_BUFFER, Some(self.ubo));
            gl.buffer_sub_data_u8_slice(glow::UNIFORM_BUFFER, 0, &data);
        }
    }

    /// Clear joint data (set num_joints = 0). Call when switching to a static model.
    pub fn clear(&self, gl: &glow::Context) {
        let zero = 0i32;
        unsafe {
            gl.bind_buffer(glow::UNIFORM_BUFFER, Some(self.ubo));
            gl.buffer_sub_data_u8_slice(glow::UNIFORM_BUFFER, 0, &zero.to_le_bytes());
        }
    }

    pub fn bind(&self, gl: &glow::Context) {
        unsafe {
            gl.bind_buffer_base(
                glow::UNIFORM_BUFFER,
                crate::shader::JOINT_MATRICES_BINDING,
                Some(self.ubo),
            );
        }
    }

    pub fn destroy(&self, gl: &glow::Context) {
        unsafe {
            gl.delete_buffer(self.ubo);
        }
    }
}

/// GPU texture storing per-vertex skinning data (joint indices + weights).
///
/// Layout: RGBA32F texture, width = num_vertices, height = 2.
///   Row 0: joint indices as f32 (r=j0, g=j1, b=j2, a=j3)
///   Row 1: joint weights (r=w0, g=w1, b=w2, a=w3)
pub struct SkinningTexture {
    pub texture: glow::Texture,
    pub num_vertices: usize,
}

impl SkinningTexture {
    /// Upload per-vertex skinning data.
    ///
    /// `joint_indices`: flat [j0,j1,j2,j3] × num_vertices (as f32)
    /// `joint_weights`: flat [w0,w1,w2,w3] × num_vertices
    pub fn new(
        gl: &glow::Context,
        joint_indices: &[[u16; 4]],
        joint_weights: &[[f32; 4]],
    ) -> Result<Self, String> {
        let num_vertices = joint_indices.len();
        assert_eq!(num_vertices, joint_weights.len());

        unsafe {
            let texture = gl.create_texture().map_err(|e| format!("skinning tex: {e}"))?;
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));

            // Pack into RGBA32F: width = num_vertices, height = 2
            // Row 0: indices, Row 1: weights
            let mut data = vec![0.0f32; num_vertices * 4 * 2];
            for (i, (ji, jw)) in joint_indices.iter().zip(joint_weights.iter()).enumerate() {
                // Row 0: indices
                data[i * 4 + 0] = ji[0] as f32;
                data[i * 4 + 1] = ji[1] as f32;
                data[i * 4 + 2] = ji[2] as f32;
                data[i * 4 + 3] = ji[3] as f32;
                // Row 1: weights
                let row1 = num_vertices * 4;
                data[row1 + i * 4 + 0] = jw[0];
                data[row1 + i * 4 + 1] = jw[1];
                data[row1 + i * 4 + 2] = jw[2];
                data[row1 + i * 4 + 3] = jw[3];
            }

            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGBA32F as i32,
                num_vertices as i32,
                2,
                0,
                glow::RGBA,
                glow::FLOAT,
                glow::PixelUnpackData::Slice(Some(bytemuck_cast_slice(&data))),
            );
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);

            Ok(SkinningTexture { texture, num_vertices })
        }
    }

    /// Bind to a specific texture unit.
    pub fn bind(&self, gl: &glow::Context, unit: u32) {
        unsafe {
            gl.active_texture(glow::TEXTURE0 + unit);
            gl.bind_texture(glow::TEXTURE_2D, Some(self.texture));
        }
    }

    pub fn destroy(&self, gl: &glow::Context) {
        unsafe {
            gl.delete_texture(self.texture);
        }
    }
}

/// Safe cast from &[T] to &[u8] without pulling in the bytemuck dependency.
fn bytemuck_cast_slice<T>(data: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<T>(),
        )
    }
}
