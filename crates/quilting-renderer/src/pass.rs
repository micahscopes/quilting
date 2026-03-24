//! Render passes: configure GL state and issue draw calls per render mode.

use glow::HasContext;

use crate::buffer::{MeshBuffers, VertexUniformBuf, WireUniformBuf};
use crate::shader::Programs;

/// Rendering modes supported by the quilting pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderMode {
    /// Matcap-style shading with density heatmap.
    Matcap,
    /// Wireframe overlay (with optional density coloring).
    Wire,
    /// Normal visualization (RGB-encoded normals).
    Normals,
    /// Combined: solid matcap + wireframe overlay.
    Both,
    /// Full PBR rendering (per-material texture binding).
    Pbr,
    /// LOD heatmap visualization.
    Lod,
}

/// Camera, projection, and scene-level state needed for rendering.
pub struct Camera {
    /// Model-view-projection matrix (column-major).
    pub mvp: [f32; 16],
    /// Model-view matrix (column-major).
    pub mv: [f32; 16],
    /// Möbius transform quaternions [a.w,a.x,a.y,a.z, b..., c..., d...].
    pub mobius: [f32; 16],
    /// World-space camera position.
    pub camera_pos: [f32; 3],
}

/// A single draw batch with per-batch state.
pub struct RenderBatch<'a> {
    pub mesh: &'a MeshBuffers,
    /// Permutation parity (+1 or -1) for normal flipping.
    pub perm_parity: f32,
    /// S3 permutation index (0-5) for bary remapping.
    pub perm_index: i32,
    /// Wire color for this batch [r, g, b].
    pub wire_color: [f32; 3],
    /// Material index (for PBR rendering).
    pub material_index: usize,
}

/// Upload vertex UBO for a batch and bind it.
fn upload_batch_ubo(
    gl: &glow::Context,
    vtx_ubo: &VertexUniformBuf,
    camera: &Camera,
    perm_parity: f32,
    perm_index: i32,
    use_qb: i32,
) {
    vtx_ubo.upload(
        gl,
        &camera.mvp,
        &camera.mv,
        perm_parity,
        perm_index,
        use_qb,
        &camera.mobius,
        &camera.camera_pos,
    );
    vtx_ubo.bind(gl);
}

/// Render a frame with the given mode, camera, and batches.
pub fn render_frame(
    gl: &glow::Context,
    programs: &Programs,
    mode: RenderMode,
    camera: &Camera,
    batches: &[RenderBatch],
    vtx_ubo: &VertexUniformBuf,
    wire_ubo: &WireUniformBuf,
) {
    vtx_ubo.bind(gl);

    // PBR falls back to matcap until PBR pass is fully wired
    let draw_matcap = matches!(mode, RenderMode::Matcap | RenderMode::Both | RenderMode::Lod | RenderMode::Pbr);
    let draw_wire = matches!(mode, RenderMode::Wire | RenderMode::Both);
    let draw_normals = mode == RenderMode::Normals;

    // Matcap/LOD pass (filled triangles)
    if draw_matcap {
        unsafe { gl.use_program(Some(programs.matcap)); }

        for batch in batches {
            upload_batch_ubo(gl, vtx_ubo, camera, batch.perm_parity, batch.perm_index, 1);

            unsafe {
                gl.bind_vertex_array(Some(batch.mesh.tri_vao));
                gl.draw_elements_instanced(
                    glow::TRIANGLES,
                    batch.mesh.num_tri_indices,
                    glow::UNSIGNED_INT,
                    0,
                    batch.mesh.num_instances,
                );
            }
        }
    }

    // Wire pass (lines)
    if draw_wire {
        unsafe { gl.use_program(Some(programs.wire)); }
        wire_ubo.bind(gl);

        for batch in batches {
            upload_batch_ubo(gl, vtx_ubo, camera, batch.perm_parity, batch.perm_index, 1);

            wire_ubo.upload(gl, batch.wire_color, true);
            wire_ubo.bind(gl);

            unsafe {
                gl.bind_vertex_array(Some(batch.mesh.line_vao));
                gl.draw_elements_instanced(
                    glow::LINES,
                    batch.mesh.num_line_indices,
                    glow::UNSIGNED_INT,
                    0,
                    batch.mesh.num_instances,
                );
            }
        }
    }

    // Normals pass (filled triangles)
    if draw_normals {
        unsafe { gl.use_program(Some(programs.normals)); }

        for batch in batches {
            upload_batch_ubo(gl, vtx_ubo, camera, batch.perm_parity, batch.perm_index, 1);

            unsafe {
                gl.bind_vertex_array(Some(batch.mesh.tri_vao));
                gl.draw_elements_instanced(
                    glow::TRIANGLES,
                    batch.mesh.num_tri_indices,
                    glow::UNSIGNED_INT,
                    0,
                    batch.mesh.num_instances,
                );
            }
        }
    }

    // PBR pass placeholder — will be implemented in Phase 3
    // For now, falls through to matcap if PBR mode is selected

    unsafe { gl.bind_vertex_array(None); }
}

/// Identity Möbius transform: a=1, b=0, c=0, d=1 (as quaternions).
const IDENTITY_MOBIUS: [f32; 16] = [
    1.0, 0.0, 0.0, 0.0,  // a
    0.0, 0.0, 0.0, 0.0,  // b
    0.0, 0.0, 0.0, 0.0,  // c
    1.0, 0.0, 0.0, 0.0,  // d
];

/// Render the original (untransformed) mesh wireframe.
/// Uses the wire program with use_qb=0 and identity Möbius.
pub fn render_original_wireframe(
    gl: &glow::Context,
    programs: &Programs,
    camera: &Camera,
    mesh: &MeshBuffers,
    vtx_ubo: &VertexUniformBuf,
    wire_ubo: &WireUniformBuf,
) {
    unsafe { gl.use_program(Some(programs.wire)); }

    vtx_ubo.upload(
        gl,
        &camera.mvp,
        &camera.mv,
        1.0,
        0,
        0, // use_qb=0 for original mesh
        &IDENTITY_MOBIUS,
        &camera.camera_pos,
    );
    vtx_ubo.bind(gl);

    wire_ubo.upload(gl, [0.25, 0.25, 0.35], false);
    wire_ubo.bind(gl);

    unsafe {
        gl.bind_vertex_array(Some(mesh.line_vao));
        gl.draw_elements_instanced(
            glow::LINES,
            mesh.num_line_indices,
            glow::UNSIGNED_INT,
            0,
            mesh.num_instances,
        );
        gl.bind_vertex_array(None);
    }
}
