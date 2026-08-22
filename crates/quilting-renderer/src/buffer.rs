//! GPU buffer management: VAO, VBO, IBO, instance buffers, and UBOs.
//!
//! - Vertex attribute 0: bary coords (vec3, per-vertex)
//! - Attributes 1-3 and 7-13: per-instance vec4s read from the instance buffer
//! - Attributes 4-6: constant weight quaternions, not backed by the buffer
//!
//! The instance stride and per-attribute offsets come from
//! [`quilting_core::instance_layout`] — never hardcode them here.

use glow::HasContext;
use quilting_core::instance_layout;

/// A non-owning slice of the packed canonical tessellation atlas.
/// Cached by LOD triple -- these never change, only instance data changes per frame.
#[derive(Clone, Copy)]
pub struct TessBuffers {
    pub bary_buf: glow::Buffer,
    pub tri_index_buf: glow::Buffer,
    pub line_index_buf: glow::Buffer,
    pub num_tri_indices: i32,
    pub num_line_indices: i32,
    /// Byte offsets into the shared element buffers.
    pub tri_index_offset: i32,
    pub line_index_offset: i32,
}

/// Owns the three immutable GPU buffers shared by every canonical LOD patch.
pub struct TessAtlasBuffers {
    bary_buf: glow::Buffer,
    tri_index_buf: glow::Buffer,
    line_index_buf: glow::Buffer,
}

impl TessAtlasBuffers {
    /// Upload one packed canonical atlas.
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

            Ok(Self {
                bary_buf,
                tri_index_buf,
                line_index_buf,
            })
        }
    }

    /// Create a lightweight view into this atlas. Offsets and counts are in
    /// index elements; WebGL draw calls consume byte offsets.
    pub fn patch(
        &self,
        tri_start: u32,
        tri_count: u32,
        line_start: u32,
        line_count: u32,
    ) -> TessBuffers {
        TessBuffers {
            bary_buf: self.bary_buf,
            tri_index_buf: self.tri_index_buf,
            line_index_buf: self.line_index_buf,
            num_tri_indices: tri_count as i32,
            num_line_indices: line_count as i32,
            tri_index_offset: (tri_start * 4) as i32,
            line_index_offset: (line_start * 4) as i32,
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
    pub tri_index_offset: i32,
    pub line_index_offset: i32,
    pub num_instances: i32,
}

impl MeshBuffers {
    /// Create VAOs for triangle and line rendering, sharing tessellation + instance buffers.
    ///
    /// Compact vertex layout:
    /// - location 0: vec3 bary (per-vertex from tess buffer)
    /// - locations 1-3, 7-13: vec4 (per-instance, 160-byte stride, divisor=1)
    /// - locations 4-6: constant [1,0,0,0] (unused weight slots)
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
                tri_index_offset: tess.tri_index_offset,
                line_index_offset: tess.line_index_offset,
                num_instances,
            })
        }
    }

    /// Create VAOs that reference a shared (persistent) instance buffer at a byte offset.
    /// The shared buffer is NOT owned by this MeshBuffers — `destroy` won't delete it.
    pub fn from_shared(
        gl: &glow::Context,
        tess: &TessBuffers,
        shared_buf: &glow::Buffer,
        byte_offset: i32,
        num_instances: i32,
    ) -> Result<Self, String> {
        unsafe {
            let tri_vao = gl.create_vertex_array().map_err(|e| format!("tri vao: {e}"))?;
            setup_vao_offset(gl, tri_vao, &tess.bary_buf, &tess.tri_index_buf, shared_buf, byte_offset);

            let line_vao = gl.create_vertex_array().map_err(|e| format!("line vao: {e}"))?;
            setup_vao_offset(gl, line_vao, &tess.bary_buf, &tess.line_index_buf, shared_buf, byte_offset);

            gl.bind_vertex_array(None);

            Ok(MeshBuffers {
                tri_vao,
                line_vao,
                instance_buf: *shared_buf, // NOT owned — just a reference
                instance_buf_capacity: 0,  // 0 signals "shared, don't delete"
                num_tri_indices: tess.num_tri_indices,
                num_line_indices: tess.num_line_indices,
                tri_index_offset: tess.tri_index_offset,
                line_index_offset: tess.line_index_offset,
                num_instances,
            })
        }
    }

    /// Destroy VAOs. Skips deleting instance_buf if it's shared (capacity == 0).
    pub fn destroy_vaos_only(&self, gl: &glow::Context) {
        unsafe {
            gl.delete_vertex_array(self.tri_vao);
            gl.delete_vertex_array(self.line_vao);
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

const INSTANCE_STRIDE_BYTES: i32 = instance_layout::STRIDE_BYTES as i32;

/// Identity quaternion fed to the constant weight attributes.
const IDENTITY_WEIGHT: [f32; 4] = [1.0, 0.0, 0.0, 0.0];

/// Point the weight-quaternion attributes at a constant instead of the buffer.
unsafe fn set_constant_weights(gl: &glow::Context) {
    for loc in instance_layout::CONSTANT_WEIGHT_LOCATIONS {
        gl.vertex_attrib_4_f32(
            loc,
            IDENTITY_WEIGHT[0], IDENTITY_WEIGHT[1], IDENTITY_WEIGHT[2], IDENTITY_WEIGHT[3],
        );
    }
}

/// Configure a VAO with the quilting compact vertex layout.
///
/// Attribute 0: vec3 bary coords (per-vertex from tess buffer)
/// Instanced attributes: one vec4 per `instance_layout::ATTR_MAP` entry
/// Weight attributes: constant [1,0,0,0], see `CONSTANT_WEIGHT_LOCATIONS`
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

    // Instanced attributes from the compact stride.
    gl.bind_buffer(glow::ARRAY_BUFFER, Some(*instance_buf));
    for &(loc, offset) in &COMPACT_MAP {
        gl.enable_vertex_attrib_array(loc);
        gl.vertex_attrib_pointer_f32(loc, 4, glow::FLOAT, false, INSTANCE_STRIDE_BYTES, offset);
        gl.vertex_attrib_divisor(loc, 1);
    }

    set_constant_weights(gl);

    // Index buffer
    gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(*index_buf));

    gl.bind_vertex_array(None);
    gl.bind_buffer(glow::ARRAY_BUFFER, None);
}

/// Configure a VAO like `setup_vao` but with instance attribs starting at `byte_offset`.
unsafe fn setup_vao_offset(
    gl: &glow::Context,
    vao: glow::VertexArray,
    bary_buf: &glow::Buffer,
    index_buf: &glow::Buffer,
    instance_buf: &glow::Buffer,
    byte_offset: i32,
) {
    gl.bind_vertex_array(Some(vao));

    gl.bind_buffer(glow::ARRAY_BUFFER, Some(*bary_buf));
    gl.enable_vertex_attrib_array(0);
    gl.vertex_attrib_pointer_f32(0, 3, glow::FLOAT, false, 0, 0);

    gl.bind_buffer(glow::ARRAY_BUFFER, Some(*instance_buf));
    for &(loc, attr_offset) in &COMPACT_MAP {
        gl.enable_vertex_attrib_array(loc);
        gl.vertex_attrib_pointer_f32(loc, 4, glow::FLOAT, false, INSTANCE_STRIDE_BYTES, byte_offset + attr_offset);
        gl.vertex_attrib_divisor(loc, 1);
    }

    set_constant_weights(gl);

    gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(*index_buf));
    gl.bind_vertex_array(None);
    gl.bind_buffer(glow::ARRAY_BUFFER, None);
}

/// Create a VAO binding tessellation geometry + persistent instance buffer.
/// The instance attrib pointers start at byte offset 0 — call `bind_at_offset`
/// before each batch draw to adjust.
pub fn create_shared_vao(
    gl: &glow::Context,
    bary_buf: &glow::Buffer,
    index_buf: &glow::Buffer,
    instance_buf: &glow::Buffer,
) -> Result<glow::VertexArray, String> {
    unsafe {
        let vao = gl.create_vertex_array().map_err(|e| format!("{e}"))?;
        setup_vao(gl, vao, bary_buf, index_buf, instance_buf);
        Ok(vao)
    }
}

/// Instance attribute locations and byte offsets within the instance stride.
pub use instance_layout::ATTR_MAP as COMPACT_MAP;

/// Create a VAO that reads one complete instance record per point vertex.
/// Unlike a render VAO, these attributes have divisor zero: the preparation
/// pass dispatches one ordinary point per source patch.
pub fn create_patch_input_vao(
    gl: &glow::Context,
    instance_buf: &glow::Buffer,
    byte_offset: i32,
) -> Result<glow::VertexArray, String> {
    unsafe {
        let vao = gl.create_vertex_array().map_err(|e| format!("patch input vao: {e}"))?;
        gl.bind_vertex_array(Some(vao));
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(*instance_buf));
        for &(loc, attr_offset) in &COMPACT_MAP {
            gl.enable_vertex_attrib_array(loc);
            gl.vertex_attrib_pointer_f32(
                loc,
                4,
                glow::FLOAT,
                false,
                INSTANCE_STRIDE_BYTES,
                byte_offset + attr_offset,
            );
            gl.vertex_attrib_divisor(loc, 0);
        }
        gl.bind_vertex_array(None);
        gl.bind_buffer(glow::ARRAY_BUFFER, None);
        Ok(vao)
    }
}

/// Persistent source and prepared GPU records for every resident patch.
/// Batches are contiguous ranges in both buffers. The source is updated only
/// when batching changes; the GPU rewrites the prepared buffer every frame.
pub struct PersistentInstances {
    pub buf: glow::Buffer,
    pub prepared_buf: glow::Buffer,
    pub capacity: usize, // in bytes
}

impl PersistentInstances {
    pub fn new(gl: &glow::Context, data: &[f32]) -> Result<Self, String> {
        unsafe {
            let buf = gl.create_buffer().map_err(|e| format!("{e}"))?;
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(buf));
            let bytes = bytemuck_cast_slice(data);
            gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, bytes, glow::DYNAMIC_DRAW);

            let prepared_buf = gl.create_buffer().map_err(|error| {
                gl.delete_buffer(buf);
                format!("prepared instance buffer: {error}")
            })?;
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(prepared_buf));
            // Seed with source-format records so the render shader's explicit
            // unprepared fallback remains valid before the first GPU pass.
            gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, bytes, glow::DYNAMIC_COPY);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
            Ok(Self { buf, prepared_buf, capacity: bytes.len() })
        }
    }

    /// Re-upload the entire sorted buffer.
    pub fn upload(&self, gl: &glow::Context, data: &[f32]) {
        unsafe {
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.buf));
            let bytes = bytemuck_cast_slice(data);
            if bytes.len() <= self.capacity {
                gl.buffer_sub_data_u8_slice(glow::ARRAY_BUFFER, 0, bytes);
            } else {
                gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, bytes, glow::DYNAMIC_DRAW);
            }
        }
    }

    /// Bind instance attributes at a byte offset into this buffer.
    /// Call before each batch's draw call.
    pub fn bind_at_offset(&self, gl: &glow::Context, byte_offset: i32) {
        unsafe {
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.buf));
            for &(loc, attr_offset) in &COMPACT_MAP {
                gl.vertex_attrib_pointer_f32(
                    loc, 4, glow::FLOAT, false,
                    INSTANCE_STRIDE_BYTES,
                    byte_offset + attr_offset,
                );
            }
        }
    }

    pub fn destroy(&self, gl: &glow::Context) {
        unsafe {
            gl.delete_buffer(self.buf);
            gl.delete_buffer(self.prepared_buf);
        }
    }
}

/// UBO for vertex shader uniforms (matches WGSL Uniforms struct).
///
/// Layout (std140, 352 bytes):
///   mat4x4 mvp        (offset 0, 64 bytes)
///   mat4x4 mv         (offset 64, 64 bytes)
///   float reserved      (offset 128, 4 bytes)
///   int reserved        (offset 132, 4 bytes)
///   int use_qb         (offset 136, 4 bytes)
///   float reserved     (offset 140, 4 bytes)
///   vec4 mob_a         (offset 144, 16 bytes)
///   vec4 mob_b         (offset 160, 16 bytes)
///   vec4 mob_c         (offset 176, 16 bytes)
///   vec4 mob_d         (offset 192, 16 bytes)
///   vec4 camera_pos    (offset 208, 16 bytes)
///   mat4x4 model       (offset 224, 64 bytes)
///   mat4x4 normal_model (offset 288, 64 bytes)
pub struct VertexUniformBuf {
    pub ubo: glow::Buffer,
}

impl VertexUniformBuf {
    pub fn new(gl: &glow::Context) -> Result<Self, String> {
        unsafe {
            let ubo = gl.create_buffer().map_err(|e| format!("vtx ubo: {e}"))?;
            gl.bind_buffer(glow::UNIFORM_BUFFER, Some(ubo));
            gl.buffer_data_size(glow::UNIFORM_BUFFER, 352, glow::DYNAMIC_DRAW);
            Ok(VertexUniformBuf { ubo })
        }
    }

    /// Upload uniform data.
    ///
    /// `mvp` and `mv` are column-major 4x4 matrices.
    /// `mobius` is [a.w,a.x,a.y,a.z, b.w,b.x,b.y,b.z, c.w,..., d.w,...] (16 floats).
    /// `camera_pos` is world-space camera position (3 floats, w unused).
    pub fn upload(
        &self,
        gl: &glow::Context,
        mvp: &[f32; 16],
        mv: &[f32; 16],
        use_qb: i32,
        mobius: &[f32; 16],
        camera_pos: &[f32; 3],
        model: &[f32; 16],
        normal_model: &[f32; 16],
    ) {
        let mut data = [0u8; 352];
        // mvp: offset 0
        data[0..64].copy_from_slice(bytemuck_cast_slice(mvp));
        // mv: offset 64
        data[64..128].copy_from_slice(bytemuck_cast_slice(mv));
        // Offsets 128..136 are reserved. Permutations are per-instance now.
        // use_qb: offset 136
        data[136..140].copy_from_slice(&use_qb.to_le_bytes());
        // reserved: offset 140 (zeroed)
        // mob_a/b/c/d: offset 144-207
        data[144..208].copy_from_slice(bytemuck_cast_slice(mobius));
        // camera_pos: offset 208 (vec4, w=0)
        data[208..212].copy_from_slice(&camera_pos[0].to_le_bytes());
        data[212..216].copy_from_slice(&camera_pos[1].to_le_bytes());
        data[216..220].copy_from_slice(&camera_pos[2].to_le_bytes());
        data[224..288].copy_from_slice(bytemuck_cast_slice(model));
        data[288..352].copy_from_slice(bytemuck_cast_slice(normal_model));

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

/// UBO for skeletal animation joint matrices + morph weights.
///
/// Layout (std140, 8448 bytes):
///   i32  num_joints       (offset 0)
///   i32  skin_tex_w       (offset 4)
///   i32  num_morph_targets (offset 8)
///   i32  _pad             (offset 12)
///   mat4 joints[128]      (offset 16, 64 bytes each = 8192 bytes)
///   vec4 morph_weights[16] (offset 8208, 256 bytes)
pub struct JointMatricesBuf {
    pub ubo: glow::Buffer,
}

pub const MAX_JOINTS: usize = 128;
const MORPH_WEIGHTS_OFFSET: usize = 16 + MAX_JOINTS * 64; // 8208
const MAX_MORPH_VEC4S: usize = 16;
const JOINT_UBO_SIZE: usize = MORPH_WEIGHTS_OFFSET + MAX_MORPH_VEC4S * 16; // 8464

impl JointMatricesBuf {
    pub fn new(gl: &glow::Context) -> Result<Self, String> {
        unsafe {
            let ubo = gl.create_buffer().map_err(|e| format!("joint ubo: {e}"))?;
            gl.bind_buffer(glow::UNIFORM_BUFFER, Some(ubo));
            gl.buffer_data_size(glow::UNIFORM_BUFFER, JOINT_UBO_SIZE as i32, glow::DYNAMIC_DRAW);
            let zeros = vec![0u8; JOINT_UBO_SIZE];
            gl.buffer_sub_data_u8_slice(glow::UNIFORM_BUFFER, 0, &zeros);
            Ok(JointMatricesBuf { ubo })
        }
    }

    /// Upload joint matrices, morph weights, and header fields.
    pub fn upload(
        &self,
        gl: &glow::Context,
        matrices: &[f32],
        morph_weights: &[f32],
        skin_tex_w: i32,
    ) {
        let num_joints = (matrices.len() / 16).min(MAX_JOINTS);
        let num_morph = morph_weights.len().min(MAX_MORPH_VEC4S * 4);

        // Header (16 bytes)
        let mut header = [0u8; 16];
        header[0..4].copy_from_slice(&(num_joints as i32).to_le_bytes());
        header[4..8].copy_from_slice(&skin_tex_w.to_le_bytes());
        header[8..12].copy_from_slice(&(num_morph as i32).to_le_bytes());

        unsafe {
            gl.bind_buffer(glow::UNIFORM_BUFFER, Some(self.ubo));
            gl.buffer_sub_data_u8_slice(glow::UNIFORM_BUFFER, 0, &header);

            // Matrices at offset 16
            if num_joints > 0 {
                let mat_bytes = bytemuck_cast_slice(&matrices[..num_joints * 16]);
                gl.buffer_sub_data_u8_slice(glow::UNIFORM_BUFFER, 16, mat_bytes);
            }

            // Morph weights at offset 8208
            if num_morph > 0 {
                let morph_bytes = bytemuck_cast_slice(&morph_weights[..num_morph]);
                gl.buffer_sub_data_u8_slice(
                    glow::UNIFORM_BUFFER,
                    MORPH_WEIGHTS_OFFSET as i32,
                    morph_bytes,
                );
            }
        }
    }

    /// Clear all animation state at a model boundary.
    ///
    /// Both counts are load-bearing: clearing only `num_joints` leaves a
    /// previous model's morph-target count active, which makes the next static
    /// model sample stale animation textures.
    pub fn clear(&self, gl: &glow::Context) {
        unsafe {
            gl.bind_buffer(glow::UNIFORM_BUFFER, Some(self.ubo));
            gl.buffer_sub_data_u8_slice(glow::UNIFORM_BUFFER, 0, &[0u8; 16]);
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

/// UBO for matcap fragment uniforms (binding 1, shared with wire UBO).
///
/// Layout (std140, 16 bytes):
///   float has_matcap_tex  (offset 0)
///   pad                   (offset 4, 12 bytes)
pub struct MatcapUniformBuf {
    pub ubo: glow::Buffer,
}

impl MatcapUniformBuf {
    pub fn new(gl: &glow::Context) -> Result<Self, String> {
        unsafe {
            let ubo = gl.create_buffer().map_err(|e| format!("matcap ubo: {e}"))?;
            gl.bind_buffer(glow::UNIFORM_BUFFER, Some(ubo));
            gl.buffer_data_size(glow::UNIFORM_BUFFER, 16, glow::DYNAMIC_DRAW);
            Ok(MatcapUniformBuf { ubo })
        }
    }

    /// mode: 0.0 = LOD heatmap, 1.0 = texture matcap, 2.0 = procedural matcap
    pub fn upload(&self, gl: &glow::Context, mode: f32) {
        let mut data = [0u8; 16];
        let val: f32 = mode;
        data[0..4].copy_from_slice(&val.to_le_bytes());
        unsafe {
            gl.bind_buffer(glow::UNIFORM_BUFFER, Some(self.ubo));
            gl.buffer_sub_data_u8_slice(glow::UNIFORM_BUFFER, 0, &data);
        }
    }

    pub fn bind(&self, gl: &glow::Context) {
        unsafe {
            // Binding 1: shared with wire UBO — each fragment shader uses only one
            gl.bind_buffer_base(glow::UNIFORM_BUFFER, 1, Some(self.ubo));
        }
    }

    pub fn destroy(&self, gl: &glow::Context) {
        unsafe {
            gl.delete_buffer(self.ubo);
        }
    }
}

/// CPU-side PBR material parameters matching the WGSL PbrUniforms layout.
/// Used to upload per-material data to the PBR UBO.
#[derive(Debug, Clone)]
pub struct PbrParams {
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub has_base_color_tex: bool,
    pub has_metallic_roughness_tex: bool,
    pub emissive_factor: [f32; 3],
    pub normal_scale: f32,
    pub has_normal_tex: bool,
    pub has_emissive_tex: bool,
    pub has_occlusion_tex: bool,
    pub occlusion_strength: f32,
    pub alpha_cutoff: f32,
    pub alpha_mode: f32, // 0=opaque, 1=mask, 2=blend
    pub unlit: bool,
    pub has_env_map: bool,
    pub env_mip_count: f32,
    pub double_sided: bool,
    pub debug_output: bool,
    pub sheen_color: [f32; 3],
    pub has_sheen: bool,
    pub sheen_roughness: f32,
    pub specular_color: [f32; 3],
    pub has_specular: bool,
    pub normal_uv_scale: [f32; 2],
    pub normal_uv_offset: [f32; 2],
    pub normal_uv_rotation: f32,
    pub base_uv_scale: [f32; 2],
    pub base_uv_rotation: f32,
    // KHR_materials_ior / transmission / volume
    pub ior: f32,
    pub transmission_factor: f32,
    pub thickness_factor: f32,
    pub attenuation_color: [f32; 3],
    pub attenuation_distance: f32,
    // Image indices for texture binding (-1 = none)
    pub base_color_tex_idx: i32,
    pub metallic_roughness_tex_idx: i32,
    pub normal_tex_idx: i32,
    pub emissive_tex_idx: i32,
    pub occlusion_tex_idx: i32,
    pub transmission_tex_idx: i32,
}

impl Default for PbrParams {
    fn default() -> Self {
        PbrParams {
            base_color: [1.0, 1.0, 1.0, 1.0],
            metallic: 0.0,
            roughness: 1.0,
            has_base_color_tex: false,
            has_metallic_roughness_tex: false,
            emissive_factor: [0.0; 3],
            normal_scale: 1.0,
            has_normal_tex: false,
            has_emissive_tex: false,
            has_occlusion_tex: false,
            occlusion_strength: 1.0,
            alpha_cutoff: 0.5,
            alpha_mode: 0.0,
            unlit: false,
            has_env_map: false,
            env_mip_count: 1.0,
            double_sided: false,
            debug_output: false,
            sheen_color: [0.0; 3],
            has_sheen: false,
            sheen_roughness: 0.0,
            specular_color: [1.0; 3],
            has_specular: false,
            normal_uv_scale: [1.0, 1.0],
            normal_uv_offset: [0.0, 0.0],
            normal_uv_rotation: 0.0,
            base_uv_scale: [1.0, 1.0],
            base_uv_rotation: 0.0,
            ior: 1.5,
            transmission_factor: 0.0,
            thickness_factor: 0.0,
            attenuation_color: [1.0; 3],
            attenuation_distance: f32::INFINITY,
            base_color_tex_idx: -1,
            metallic_roughness_tex_idx: -1,
            normal_tex_idx: -1,
            emissive_tex_idx: -1,
            occlusion_tex_idx: -1,
            transmission_tex_idx: -1,
        }
    }
}

/// UBO for PBR material uniforms (binding 2, 224 bytes).
///
/// Matches WGSL PbrUniforms struct in fragment/pbr.wgsl.
pub struct PbrUniformBuf {
    pub ubo: glow::Buffer,
}

impl PbrUniformBuf {
    pub fn new(gl: &glow::Context) -> Result<Self, String> {
        unsafe {
            let ubo = gl.create_buffer().map_err(|e| format!("pbr ubo: {e}"))?;
            gl.bind_buffer(glow::UNIFORM_BUFFER, Some(ubo));
            gl.buffer_data_size(glow::UNIFORM_BUFFER, 224, glow::DYNAMIC_DRAW);
            Ok(PbrUniformBuf { ubo })
        }
    }

    pub fn upload(&self, gl: &glow::Context, p: &PbrParams) {
        let b = |v: bool| -> f32 { if v { 1.0 } else { 0.0 } };
        let mut d = [0u8; 224];
        let mut f = |off: usize, v: f32| { d[off..off+4].copy_from_slice(&v.to_le_bytes()); };

        // base_color vec4 at offset 0
        f(0, p.base_color[0]); f(4, p.base_color[1]); f(8, p.base_color[2]); f(12, p.base_color[3]);
        // metallic, roughness, has_base_color_tex, has_mr_tex at offset 16
        f(16, p.metallic); f(20, p.roughness); f(24, b(p.has_base_color_tex)); f(28, b(p.has_metallic_roughness_tex));
        // emissive_factor vec4 at offset 32
        f(32, p.emissive_factor[0]); f(36, p.emissive_factor[1]); f(40, p.emissive_factor[2]);
        // normal_scale, has_normal_tex, has_emissive_tex, has_occlusion_tex at offset 48
        f(48, p.normal_scale); f(52, b(p.has_normal_tex)); f(56, b(p.has_emissive_tex)); f(60, b(p.has_occlusion_tex));
        // occlusion_strength, alpha_cutoff, alpha_mode, unlit at offset 64
        f(64, p.occlusion_strength); f(68, p.alpha_cutoff); f(72, p.alpha_mode); f(76, b(p.unlit));
        // has_env_map, env_mip_count, double_sided, debug_output at offset 80
        f(80, b(p.has_env_map)); f(84, p.env_mip_count); f(88, b(p.double_sided)); f(92, b(p.debug_output));
        // sheen_color vec4 at offset 96 (w = has_sheen)
        f(96, p.sheen_color[0]); f(100, p.sheen_color[1]); f(104, p.sheen_color[2]); f(108, b(p.has_sheen));
        // sheen_roughness at offset 112
        f(112, p.sheen_roughness);
        // specular_color vec4 at offset 128 (w = has_specular)
        f(128, p.specular_color[0]); f(132, p.specular_color[1]); f(136, p.specular_color[2]); f(140, b(p.has_specular));
        // normal_uv_transform vec4 at offset 144 (xy=scale, zw=offset)
        f(144, p.normal_uv_scale[0]); f(148, p.normal_uv_scale[1]); f(152, p.normal_uv_offset[0]); f(156, p.normal_uv_offset[1]);
        // normal_uv_rotation at offset 160
        f(160, p.normal_uv_rotation);
        // base_uv_scale, base_uv_rotation at offset 164
        f(164, p.base_uv_scale[0]); f(168, p.base_uv_scale[1]); f(172, p.base_uv_rotation);
        // ior, transmission_factor, thickness_factor, has_transmission_tex at offset 176
        f(176, p.ior); f(180, p.transmission_factor); f(184, p.thickness_factor);
        f(188, if p.transmission_tex_idx >= 0 { 1.0 } else { 0.0 });
        // attenuation_color vec4 at offset 192 (w = attenuation_distance)
        f(192, p.attenuation_color[0]); f(196, p.attenuation_color[1]); f(200, p.attenuation_color[2]);
        f(204, p.attenuation_distance);

        unsafe {
            gl.bind_buffer(glow::UNIFORM_BUFFER, Some(self.ubo));
            gl.buffer_sub_data_u8_slice(glow::UNIFORM_BUFFER, 0, &d);
        }
    }

    pub fn bind(&self, gl: &glow::Context) {
        unsafe {
            gl.bind_buffer_base(
                glow::UNIFORM_BUFFER,
                crate::shader::PBR_UNIFORMS_BINDING,
                Some(self.ubo),
            );
        }
    }

    pub fn destroy(&self, gl: &glow::Context) {
        unsafe { gl.delete_buffer(self.ubo); }
    }
}

/// Material texture handles for a single PBR material.
#[derive(Debug, Clone, Default)]
pub struct MaterialTextures {
    pub base_color: Option<glow::Texture>,
    pub metallic_roughness: Option<glow::Texture>,
    pub normal: Option<glow::Texture>,
    pub emissive: Option<glow::Texture>,
    pub occlusion: Option<glow::Texture>,
}

/// Environment map textures (shared across all materials).
pub struct EnvironmentMaps {
    pub prefiltered: Option<glow::Texture>,
    pub irradiance: Option<glow::Texture>,
    pub sheen_lut: Option<glow::Texture>,
    pub mip_count: f32,
}

impl Default for EnvironmentMaps {
    fn default() -> Self {
        EnvironmentMaps {
            prefiltered: None,
            irradiance: None,
            sheen_lut: None,
            mip_count: 1.0,
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
        if num_vertices == 0 {
            return Err("skinning texture requires at least one vertex".into());
        }
        if num_vertices != joint_weights.len() {
            return Err(format!(
                "skinning texture has {num_vertices} joint-index rows but {} weight rows",
                joint_weights.len()
            ));
        }

        unsafe {
            let texture = gl.create_texture().map_err(|e| format!("skinning tex: {e}"))?;
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));

            // Tiled layout: width = min(num_vertices, 4096), rows alternate per chunk
            let width = num_vertices.min(4096);
            let height = ((num_vertices + width - 1) / width) * 2;
            let mut data = vec![0.0f32; width * height * 4];
            for (i, (ji, jw)) in joint_indices.iter().zip(joint_weights.iter()).enumerate() {
                let chunk = i / width;
                let col = i % width;
                let idx_off = (chunk * 2 * width + col) * 4;
                data[idx_off]     = ji[0] as f32;
                data[idx_off + 1] = ji[1] as f32;
                data[idx_off + 2] = ji[2] as f32;
                data[idx_off + 3] = ji[3] as f32;
                let wt_off = ((chunk * 2 + 1) * width + col) * 4;
                data[wt_off]     = jw[0];
                data[wt_off + 1] = jw[1];
                data[wt_off + 2] = jw[2];
                data[wt_off + 3] = jw[3];
            }

            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGBA32F as i32,
                width as i32,
                height as i32,
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

/// GPU texture storing morph-target position deltas.
///
/// Layout: RGBA32F texture, width = num_vertices, height = num_targets.
/// Each texel stores one XYZ delta with zero in the unused alpha channel.
pub struct MorphTargetTexture {
    pub texture: glow::Texture,
    pub num_vertices: usize,
    pub num_targets: usize,
}

impl MorphTargetTexture {
    /// Upload target-major `[dx, dy, dz]` deltas.
    pub fn new(
        gl: &glow::Context,
        deltas: &[f32],
        num_vertices: usize,
        num_targets: usize,
    ) -> Result<Self, String> {
        if num_vertices == 0 || num_targets == 0 {
            return Err("morph texture requires at least one vertex and target".into());
        }
        let max_texture_size = unsafe { gl.get_parameter_i32(glow::MAX_TEXTURE_SIZE).max(0) as usize };
        if num_vertices > max_texture_size || num_targets > max_texture_size {
            return Err(format!(
                "morph texture {num_vertices}x{num_targets} exceeds GL limit {max_texture_size}"
            ));
        }
        let rgba = pack_morph_target_deltas(deltas, num_vertices, num_targets)?;

        unsafe {
            let texture = gl.create_texture()
                .map_err(|e| format!("morph-target tex: {e}"))?;
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            gl.tex_image_2d(
                glow::TEXTURE_2D, 0, glow::RGBA32F as i32,
                num_vertices as i32, num_targets as i32, 0,
                glow::RGBA, glow::FLOAT,
                glow::PixelUnpackData::Slice(Some(bytemuck_cast_slice(&rgba))),
            );
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);

            Ok(Self { texture, num_vertices, num_targets })
        }
    }

    pub fn bind(&self, gl: &glow::Context, unit: u32) {
        unsafe {
            gl.active_texture(glow::TEXTURE0 + unit);
            gl.bind_texture(glow::TEXTURE_2D, Some(self.texture));
        }
    }

    pub fn destroy(&self, gl: &glow::Context) {
        unsafe { gl.delete_texture(self.texture); }
    }
}

fn pack_morph_target_deltas(
    deltas: &[f32],
    num_vertices: usize,
    num_targets: usize,
) -> Result<Vec<f32>, String> {
    if num_vertices == 0 || num_targets == 0 {
        return Err("morph texture requires at least one vertex and target".into());
    }
    let texel_count = num_vertices.checked_mul(num_targets)
        .ok_or_else(|| "morph texture dimensions overflow".to_string())?;
    let delta_count = texel_count.checked_mul(3)
        .ok_or_else(|| "morph delta count overflows".to_string())?;
    if deltas.len() < delta_count {
        return Err(format!(
            "morph texture needs {delta_count} delta components, received {}",
            deltas.len()
        ));
    }

    let mut rgba = vec![0.0; texel_count * 4];
    for (source, target) in deltas[..delta_count].chunks_exact(3).zip(rgba.chunks_exact_mut(4)) {
        target[..3].copy_from_slice(source);
    }
    Ok(rgba)
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

#[cfg(test)]
mod tests {
    use super::pack_morph_target_deltas;

    #[test]
    fn morph_deltas_are_packed_target_major_with_zero_alpha() {
        let packed = pack_morph_target_deltas(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // target 0
                7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // target 1
            ],
            2,
            2,
        ).unwrap();
        assert_eq!(
            packed,
            vec![
                1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0, 0.0,
                7.0, 8.0, 9.0, 0.0, 10.0, 11.0, 12.0, 0.0,
            ]
        );
    }

    #[test]
    fn morph_delta_dimensions_are_validated() {
        assert!(pack_morph_target_deltas(&[], 0, 1).is_err());
        assert!(pack_morph_target_deltas(&[1.0, 2.0], 1, 1).is_err());
    }
}
