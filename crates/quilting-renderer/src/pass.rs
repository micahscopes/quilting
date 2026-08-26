//! Render passes: configure GL state and issue draw calls per render mode.

use glow::HasContext;
use quilting_core::render::{RenderGeometry, RenderSubmissionStats};

use crate::buffer::{MeshBuffers, MeshDraw, VertexUniformBuf, WireUniformBuf};
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
    /// Möbius stretch heatmap (conformal distortion debug).
    Stretch,
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
#[derive(Clone, Copy)]
pub struct RenderBatch {
    pub mesh: MeshDraw,
    /// Permutation parity (+1 or -1) for raster winding.
    pub perm_parity: f32,
    /// Material index (for PBR rendering).
    pub material_index: usize,
    /// Representative node for the render transform shared by this draw.
    /// Per-instance semantic node identity remains in the prepared patch data.
    pub render_node_index: usize,
    /// Per-entity conformal transform selected during render extraction.
    pub mobius: [f32; 16],
    /// Explicit orientation parity of the authored generator word.
    pub orientation_sign: i8,
    /// Ordinary affine transform applied before the conformal map.
    pub euclidean_model: [f32; 16],
    /// Inverse-transpose linear part of `euclidean_model`.
    pub euclidean_normal: [f32; 16],
}

pub const IDENTITY_MATRIX: [f32; 16] = [
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
];

fn affine_determinant(model: &[f32; 16]) -> f32 {
    let (a00, a01, a02) = (model[0], model[4], model[8]);
    let (a10, a11, a12) = (model[1], model[5], model[9]);
    let (a20, a21, a22) = (model[2], model[6], model[10]);
    a00 * (a11 * a22 - a12 * a21)
        - a01 * (a10 * a22 - a12 * a20)
        + a02 * (a10 * a21 - a11 * a20)
}

/// Orientation of the ordinary affine layer. Degenerate matrices are treated
/// as even because they have no well-defined inverse normal transform.
pub fn affine_orientation_sign(model: &[f32; 16]) -> i8 {
    if affine_determinant(model) < 0.0 { -1 } else { 1 }
}

/// Inverse-transpose of an affine matrix's 3×3 linear part, embedded in a
/// column-major 4×4 matrix for the shader. Singular inputs fall back to the
/// identity; their geometry is already degenerate and cannot define normals.
pub fn affine_normal_matrix(model: &[f32; 16]) -> [f32; 16] {
    let (a00, a01, a02) = (model[0], model[4], model[8]);
    let (a10, a11, a12) = (model[1], model[5], model[9]);
    let (a20, a21, a22) = (model[2], model[6], model[10]);
    let c00 = a11 * a22 - a12 * a21;
    let c01 = a12 * a20 - a10 * a22;
    let c02 = a10 * a21 - a11 * a20;
    let c10 = a02 * a21 - a01 * a22;
    let c11 = a00 * a22 - a02 * a20;
    let c12 = a01 * a20 - a00 * a21;
    let c20 = a01 * a12 - a02 * a11;
    let c21 = a02 * a10 - a00 * a12;
    let c22 = a00 * a11 - a01 * a10;
    let determinant = a00 * c00 + a01 * c01 + a02 * c02;
    if !determinant.is_finite() || determinant.abs() <= 1.0e-12 {
        return IDENTITY_MATRIX;
    }
    let inverse = determinant.recip();
    [
        c00 * inverse, c10 * inverse, c20 * inverse, 0.0,
        c01 * inverse, c11 * inverse, c21 * inverse, 0.0,
        c02 * inverse, c12 * inverse, c22 * inverse, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]
}

/// Copy view state while selecting this entity batch's conformal map.
pub fn camera_for_batch(camera: &Camera, batch: &RenderBatch) -> Camera {
    Camera {
        mvp: camera.mvp,
        mv: camera.mv,
        mobius: batch.mobius,
        camera_pos: camera.camera_pos,
    }
}

/// Whether two commands consume identical per-batch vertex uniforms. Camera
/// matrices and position are frame-global; winding and material state are
/// handled separately, so neither belongs in this comparison.
pub fn same_vertex_uniform_state(a: &RenderBatch, b: &RenderBatch) -> bool {
    a.mobius == b.mobius
        && a.euclidean_model == b.euclidean_model
        && a.euclidean_normal == b.euclidean_normal
}

/// Combine authored/affine orientation with the canonical atlas permutation.
/// Odd S3 permutations reflect barycentric space and therefore reverse winding.
pub fn batch_orientation_sign(orientation_sign: i8, perm_parity: f32) -> i8 {
    orientation_sign * if perm_parity < 0.0 { -1 } else { 1 }
}

/// Set winding from semantic orientation and permutation parity. Matrix shape
/// (`c != 0`) is not an orientation test: an even composition can be proper
/// with nonzero c.
pub fn apply_batch_winding(gl: &glow::Context, orientation_sign: i8, perm_parity: f32) {
    let sign = batch_orientation_sign(orientation_sign, perm_parity);
    unsafe {
        gl.front_face(if sign < 0 {
            glow::CW
        } else {
            glow::CCW
        });
    }
}

/// Upload vertex UBO for a batch and bind it.
pub fn upload_batch_ubo(
    gl: &glow::Context,
    vtx_ubo: &VertexUniformBuf,
    camera: &Camera,
    use_qb: i32,
    euclidean_model: &[f32; 16],
    euclidean_normal: &[f32; 16],
) {
    vtx_ubo.upload(
        gl,
        &camera.mvp,
        &camera.mv,
        use_qb,
        &camera.mobius,
        &camera.camera_pos,
        euclidean_model,
        euclidean_normal,
    );
    vtx_ubo.bind(gl);
}

fn upload_batch_ubo_if_changed<'a>(
    gl: &glow::Context,
    vtx_ubo: &VertexUniformBuf,
    camera: &Camera,
    previous: &mut Option<&'a RenderBatch>,
    batch: &'a RenderBatch,
) {
    if previous
        .is_some_and(|previous| same_vertex_uniform_state(previous, batch))
    {
        return;
    }
    let batch_camera = camera_for_batch(camera, batch);
    upload_batch_ubo(
        gl,
        vtx_ubo,
        &batch_camera,
        1,
        &batch.euclidean_model,
        &batch.euclidean_normal,
    );
    *previous = Some(batch);
}

fn draw_batches(
    gl: &glow::Context,
    camera: &Camera,
    batches: &[RenderBatch],
    vtx_ubo: &VertexUniformBuf,
    geometry: RenderGeometry,
) -> RenderSubmissionStats {
    let mut stats = RenderSubmissionStats::default();
    let mut vertex_state = None;
    for batch in batches {
        apply_batch_winding(gl, batch.orientation_sign, batch.perm_parity);
        upload_batch_ubo_if_changed(gl, vtx_ubo, camera, &mut vertex_state, batch);

        let (vertex_array, primitive, count, offset) = match geometry {
            RenderGeometry::Triangles => (
                batch.mesh.tri_vao,
                glow::TRIANGLES,
                batch.mesh.num_tri_indices,
                batch.mesh.tri_index_offset,
            ),
            RenderGeometry::Lines => (
                batch.mesh.line_vao,
                glow::LINES,
                batch.mesh.num_line_indices,
                batch.mesh.line_index_offset,
            ),
        };
        unsafe {
            gl.bind_vertex_array(Some(vertex_array));
            gl.draw_elements_instanced(
                primitive,
                count,
                glow::UNSIGNED_INT,
                offset,
                batch.mesh.num_instances,
            );
        }
        record_indexed_submission(&mut stats, geometry, count, batch.mesh.num_instances);
    }
    stats
}

/// Account for one indexed backend submission while preserving invalid signed
/// GL counts as diagnostics instead of reinterpreting them as large unsigned
/// workloads. The GL call itself remains adjacent to this recorder at each
/// submission site.
pub fn record_indexed_submission(
    stats: &mut RenderSubmissionStats,
    geometry: RenderGeometry,
    index_count: i32,
    instance_count: i32,
) {
    match (u32::try_from(index_count), u32::try_from(instance_count)) {
        (Ok(index_count), Ok(instance_count)) => {
            stats.record_indexed_draw(geometry, index_count, instance_count);
        }
        _ => stats.record_invalid_draw(),
    }
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
) -> RenderSubmissionStats {
    let mut stats = RenderSubmissionStats::default();
    vtx_ubo.bind(gl);

    let draw_pbr = mode == RenderMode::Pbr;
    let draw_matcap = matches!(
        mode,
        RenderMode::Matcap | RenderMode::Both | RenderMode::Lod
    );
    let draw_wire = matches!(mode, RenderMode::Wire | RenderMode::Both);
    let draw_normals = mode == RenderMode::Normals;

    // PBR pass (filled triangles with PBR shader)
    if draw_pbr {
        unsafe {
            gl.use_program(Some(programs.pbr));
        }
        stats.merge(draw_batches(
            gl,
            camera,
            batches,
            vtx_ubo,
            RenderGeometry::Triangles,
        ));
    }

    // Matcap/LOD pass (filled triangles)
    if draw_matcap {
        unsafe {
            gl.use_program(Some(programs.matcap));
        }

        stats.merge(draw_batches(
            gl,
            camera,
            batches,
            vtx_ubo,
            RenderGeometry::Triangles,
        ));
    }

    // Wire pass (lines)
    if draw_wire {
        unsafe {
            gl.use_program(Some(programs.wire));
        }
        // All adaptive wire draws use the density heatmap; the fallback color
        // is ignored by the shader and therefore frame-global state.
        wire_ubo.upload(gl, [0.0; 3], true);
        wire_ubo.bind(gl);

        stats.merge(draw_batches(
            gl,
            camera,
            batches,
            vtx_ubo,
            RenderGeometry::Lines,
        ));
    }

    // Normals pass (filled triangles)
    if draw_normals {
        unsafe {
            gl.use_program(Some(programs.normals));
        }

        stats.merge(draw_batches(
            gl,
            camera,
            batches,
            vtx_ubo,
            RenderGeometry::Triangles,
        ));
    }

    // Stretch heatmap pass (filled triangles)
    if mode == RenderMode::Stretch {
        unsafe {
            gl.use_program(Some(programs.stretch));
        }

        stats.merge(draw_batches(
            gl,
            camera,
            batches,
            vtx_ubo,
            RenderGeometry::Triangles,
        ));
    }

    unsafe {
        gl.bind_vertex_array(None);
    }
    stats
}

/// Identity Möbius transform: a=1, b=0, c=0, d=1 (as quaternions).
const IDENTITY_MOBIUS: [f32; 16] = [
    1.0, 0.0, 0.0, 0.0, // a
    0.0, 0.0, 0.0, 0.0, // b
    0.0, 0.0, 0.0, 0.0, // c
    1.0, 0.0, 0.0, 0.0, // d
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
    unsafe {
        gl.use_program(Some(programs.wire));
    }

    vtx_ubo.upload(
        gl,
        &camera.mvp,
        &camera.mv,
        0, // use_qb=0 for original mesh
        &IDENTITY_MOBIUS,
        &camera.camera_pos,
        &IDENTITY_MATRIX,
        &IDENTITY_MATRIX,
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
            mesh.line_index_offset,
            mesh.num_instances,
        );
        gl.bind_vertex_array(None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_submission_counts_are_validated_before_accounting() {
        let mut stats = RenderSubmissionStats::default();
        record_indexed_submission(&mut stats, RenderGeometry::Triangles, 12, 2);
        record_indexed_submission(&mut stats, RenderGeometry::Lines, 8, 0);
        record_indexed_submission(&mut stats, RenderGeometry::Triangles, -1, 2);
        record_indexed_submission(&mut stats, RenderGeometry::Triangles, 12, -1);

        assert_eq!(stats.draw_calls, 4);
        assert_eq!(stats.zero_instance_draw_calls, 1);
        assert_eq!(stats.invalid_draw_calls, 2);
        assert_eq!(stats.submitted_instances, 2);
        assert_eq!(stats.triangles, 8);
        assert_eq!(stats.lines, 0);
    }
}
