//! Render passes: configure GL state and issue draw calls per render mode.

use glow::HasContext;
use quilting_core::batch::RenderBatchLayer;
use quilting_core::render::{
    render_draw_passes, PbrDrawClass, RenderBatchSelection, RenderBatchSnapshot, RenderExecution,
    RenderGeometry, RenderPass, RenderStyle, RenderSubmissionStats, ResolvedRenderCommand,
};

use crate::buffer::{MeshBuffers, MeshDraw, VertexUniformBuf, WireUniformBuf};
use crate::shader::Programs;

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
    /// Apply the renderer-owned source-root suppression mask during the
    /// camera-dependent visibility pass.
    pub suppress_source_roots: bool,
    /// Permutation parity (+1 or -1) for raster winding.
    pub perm_parity: f32,
    /// Material index (for PBR rendering).
    pub material_index: usize,
    /// Shared semantic draw class used by the backend-neutral pass plan.
    pub pbr_class: PbrDrawClass,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebGlBatchField {
    TriangleIndexCount,
    LineIndexCount,
    InstanceCount,
    SourceRootSuppression,
    PermutationParity,
    Material,
    DrawClass,
    RenderNode,
    ConformalTransform,
    Orientation,
    EuclideanModel,
    EuclideanNormal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebGlRenderExecutionError {
    BatchCount {
        expected: usize,
        actual: usize,
    },
    BatchMismatch {
        batch_index: u32,
        field: WebGlBatchField,
    },
    UnsupportedCommand(&'static str),
}

impl std::fmt::Display for WebGlRenderExecutionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BatchCount { expected, actual } => write!(
                formatter,
                "WebGL has {actual} retained batches; resolved frame requires {expected}",
            ),
            Self::BatchMismatch { batch_index, field } => write!(
                formatter,
                "WebGL batch {batch_index} does not match resolved {field:?}",
            ),
            Self::UnsupportedCommand(command) => {
                write!(formatter, "WebGL diagnostic executor does not support {command}")
            }
        }
    }
}

impl std::error::Error for WebGlRenderExecutionError {}

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
    a.suppress_source_roots == b.suppress_source_roots
        && a.mobius == b.mobius
        && a.euclidean_model == b.euclidean_model
        && a.euclidean_normal == b.euclidean_normal
}

fn validate_batch_residency(
    batch_index: u32,
    expected: &RenderBatchSnapshot,
    actual: &RenderBatch,
) -> Result<(), WebGlRenderExecutionError> {
    let mismatch = |field| WebGlRenderExecutionError::BatchMismatch { batch_index, field };
    if u32::try_from(actual.mesh.num_tri_indices).ok() != Some(expected.triangle_index_count) {
        return Err(mismatch(WebGlBatchField::TriangleIndexCount));
    }
    if u32::try_from(actual.mesh.num_line_indices).ok() != Some(expected.line_index_count) {
        return Err(mismatch(WebGlBatchField::LineIndexCount));
    }
    if u32::try_from(actual.mesh.num_instances).ok() != expected.active_instance_count().ok() {
        return Err(mismatch(WebGlBatchField::InstanceCount));
    }
    if actual.suppress_source_roots != (expected.id.layer == RenderBatchLayer::RetainedRoot) {
        return Err(mismatch(WebGlBatchField::SourceRootSuppression));
    }
    if actual.perm_parity != expected.id.key.parity() {
        return Err(mismatch(WebGlBatchField::PermutationParity));
    }
    if actual.material_index != expected.id.key.material_index {
        return Err(mismatch(WebGlBatchField::Material));
    }
    if actual.pbr_class != expected.pbr_class {
        return Err(mismatch(WebGlBatchField::DrawClass));
    }
    if actual.render_node_index != expected.id.key.render_node_index {
        return Err(mismatch(WebGlBatchField::RenderNode));
    }
    if actual.mobius != expected.transform.mobius {
        return Err(mismatch(WebGlBatchField::ConformalTransform));
    }
    if actual.orientation_sign != expected.transform.orientation_sign {
        return Err(mismatch(WebGlBatchField::Orientation));
    }
    if actual.euclidean_model != expected.transform.euclidean_model {
        return Err(mismatch(WebGlBatchField::EuclideanModel));
    }
    if actual.euclidean_normal != expected.transform.euclidean_normal {
        return Err(mismatch(WebGlBatchField::EuclideanNormal));
    }
    Ok(())
}

/// Prove that a validated semantic frame addresses the exact retained WebGL
/// allocations before the first device call. PBR composition is admitted by a
/// separate executor because its transmission and focus commands interleave
/// framebuffer work with patch draws.
fn validate_execution_residency(
    execution: RenderExecution<'_, '_>,
    batches: &[RenderBatch],
) -> Result<(), WebGlRenderExecutionError> {
    if execution.batches().len() != batches.len() {
        return Err(WebGlRenderExecutionError::BatchCount {
            expected: execution.batches().len(),
            actual: batches.len(),
        });
    }
    for (batch_index, (expected, actual)) in execution
        .batches()
        .iter()
        .zip(batches)
        .enumerate()
    {
        validate_batch_residency(batch_index as u32, expected, actual)?;
    }
    Ok(())
}

pub fn validate_diagnostic_execution(
    execution: RenderExecution<'_, '_>,
    batches: &[RenderBatch],
) -> Result<(), WebGlRenderExecutionError> {
    validate_execution_residency(execution, batches)?;
    for command in execution {
        match command {
            ResolvedRenderCommand::PreparePatches { .. }
            | ResolvedRenderCommand::ResolveVisibility { .. }
            | ResolvedRenderCommand::HighlightFace { .. } => {}
            ResolvedRenderCommand::DrawPatches { pass, .. }
                if !matches!(pass, RenderPass::PbrOpaque | RenderPass::PbrTransparent) => {}
            ResolvedRenderCommand::DrawPatches { .. } => {
                return Err(WebGlRenderExecutionError::UnsupportedCommand("PBR draw"));
            }
            ResolvedRenderCommand::BuildTransmissionPyramid => {
                return Err(WebGlRenderExecutionError::UnsupportedCommand(
                    "transmission pyramid",
                ));
            }
            ResolvedRenderCommand::FocusPostProcess => {
                return Err(WebGlRenderExecutionError::UnsupportedCommand(
                    "focus postprocess",
                ));
            }
        }
    }
    Ok(())
}

/// Preflight the interleaved PBR command subset without issuing device calls.
/// The WASM compositor still owns framebuffer and material resources while
/// this gate moves draw/pass selection onto the shared command authority.
pub fn validate_pbr_execution(
    execution: RenderExecution<'_, '_>,
    batches: &[RenderBatch],
) -> Result<(), WebGlRenderExecutionError> {
    validate_execution_residency(execution, batches)?;
    for command in execution {
        match command {
            ResolvedRenderCommand::PreparePatches { .. }
            | ResolvedRenderCommand::ResolveVisibility { .. }
            | ResolvedRenderCommand::BuildTransmissionPyramid
            | ResolvedRenderCommand::FocusPostProcess => {}
            ResolvedRenderCommand::DrawPatches { pass, .. }
                if matches!(pass, RenderPass::PbrOpaque | RenderPass::PbrTransparent) => {}
            ResolvedRenderCommand::DrawPatches { .. } => {
                return Err(WebGlRenderExecutionError::UnsupportedCommand(
                    "diagnostic draw",
                ));
            }
            ResolvedRenderCommand::HighlightFace { .. } => {
                return Err(WebGlRenderExecutionError::UnsupportedCommand(
                    "diagnostic highlight",
                ));
            }
        }
    }
    Ok(())
}

/// Owned lowering of one validated PBR command stream for the incumbent
/// compositor. Owning only batch indices releases the scene borrow before the
/// WASM adapter mutates browser framebuffer resources. It is built only by the
/// opt-in shadow route; the default dispatcher allocates nothing here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WebGlPbrCommandPlan {
    opaque_batches: Vec<usize>,
    transparent_batches: Vec<usize>,
    build_transmission_pyramid: bool,
    focus_postprocess: bool,
}

impl WebGlPbrCommandPlan {
    pub fn build(
        execution: RenderExecution<'_, '_>,
        batches: &[RenderBatch],
    ) -> Result<Self, WebGlRenderExecutionError> {
        validate_pbr_execution(execution, batches)?;
        let mut plan = Self {
            opaque_batches: Vec::new(),
            transparent_batches: Vec::new(),
            build_transmission_pyramid: false,
            focus_postprocess: false,
        };
        for command in execution {
            match command {
                ResolvedRenderCommand::DrawPatches {
                    batch_index,
                    pass: RenderPass::PbrOpaque,
                    ..
                } => plan.opaque_batches.push(batch_index as usize),
                ResolvedRenderCommand::DrawPatches {
                    batch_index,
                    pass: RenderPass::PbrTransparent,
                    ..
                } => plan.transparent_batches.push(batch_index as usize),
                ResolvedRenderCommand::BuildTransmissionPyramid => {
                    plan.build_transmission_pyramid = true;
                }
                ResolvedRenderCommand::FocusPostProcess => plan.focus_postprocess = true,
                _ => {}
            }
        }
        Ok(plan)
    }

    pub fn batches(&self, pass: RenderPass) -> &[usize] {
        match pass {
            RenderPass::PbrOpaque => &self.opaque_batches,
            RenderPass::PbrTransparent => &self.transparent_batches,
            _ => &[],
        }
    }

    pub fn builds_transmission_pyramid(&self) -> bool {
        self.build_transmission_pyramid
    }

    pub fn runs_focus_postprocess(&self) -> bool {
        self.focus_postprocess
    }
}

pub enum WebGlPbrDraws<'a> {
    Resolved {
        indices: std::slice::Iter<'a, usize>,
        batches: &'a [RenderBatch],
    },
    Legacy {
        batches: std::iter::Enumerate<std::slice::Iter<'a, RenderBatch>>,
        selection: RenderBatchSelection,
    },
}

impl<'a> Iterator for WebGlPbrDraws<'a> {
    type Item = (usize, &'a RenderBatch);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Resolved { indices, batches } => {
                let batch_index = *indices.next()?;
                Some((batch_index, &batches[batch_index]))
            }
            Self::Legacy { batches, selection } => batches
                .find(|(_, batch)| selection.includes(batch.pbr_class)),
        }
    }
}

/// Iterate PBR draws from the resolved command plan when present, or from the
/// incumbent semantic pass selection without allocating in the default path.
pub fn webgl_pbr_draws<'a>(
    plan: Option<&'a WebGlPbrCommandPlan>,
    batches: &'a [RenderBatch],
    pass: RenderPass,
) -> WebGlPbrDraws<'a> {
    if let Some(plan) = plan {
        WebGlPbrDraws::Resolved {
            indices: plan.batches(pass).iter(),
            batches,
        }
    } else {
        let selection = render_draw_passes(RenderStyle::Pbr)
            .iter()
            .find(|draw_pass| draw_pass.pass == pass)
            .expect("PBR draw pass is canonical")
            .batches;
        WebGlPbrDraws::Legacy {
            batches: batches.iter().enumerate(),
            selection,
        }
    }
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
    suppress_source_roots: bool,
    use_qb: i32,
    euclidean_model: &[f32; 16],
    euclidean_normal: &[f32; 16],
) {
    vtx_ubo.upload(
        gl,
        &camera.mvp,
        &camera.mv,
        suppress_source_roots,
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
        batch.suppress_source_roots,
        1,
        &batch.euclidean_model,
        &batch.euclidean_normal,
    );
    *previous = Some(batch);
}

#[allow(clippy::too_many_arguments)]
fn draw_batch<'a>(
    gl: &glow::Context,
    camera: &Camera,
    vtx_ubo: &VertexUniformBuf,
    vertex_state: &mut Option<&'a RenderBatch>,
    batch_index: usize,
    batch: &'a RenderBatch,
    pass: RenderPass,
    geometry: RenderGeometry,
) -> RenderSubmissionStats {
    apply_batch_winding(gl, batch.orientation_sign, batch.perm_parity);
    upload_batch_ubo_if_changed(gl, vtx_ubo, camera, vertex_state, batch);

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
    let mut stats = RenderSubmissionStats::default();
    record_indexed_submission(
        &mut stats,
        batch_index,
        pass,
        geometry,
        count,
        batch.mesh.num_instances,
    );
    stats
}

fn draw_batches(
    gl: &glow::Context,
    camera: &Camera,
    batches: &[RenderBatch],
    vtx_ubo: &VertexUniformBuf,
    pass: RenderPass,
    geometry: RenderGeometry,
    selection: RenderBatchSelection,
) -> RenderSubmissionStats {
    let mut stats = RenderSubmissionStats::default();
    let mut vertex_state = None;
    for (batch_index, batch) in batches.iter().enumerate() {
        if selection.includes(batch.pbr_class) {
            stats.merge(draw_batch(
                gl,
                camera,
                vtx_ubo,
                &mut vertex_state,
                batch_index,
                batch,
                pass,
                geometry,
            ));
        }
    }
    stats
}

/// Account for one indexed backend submission while preserving invalid signed
/// GL counts as diagnostics instead of reinterpreting them as large unsigned
/// workloads. The GL call itself remains adjacent to this recorder at each
/// submission site.
pub fn record_indexed_submission(
    stats: &mut RenderSubmissionStats,
    batch_index: usize,
    pass: RenderPass,
    geometry: RenderGeometry,
    index_count: i32,
    instance_count: i32,
) {
    match (
        u32::try_from(batch_index),
        u32::try_from(index_count),
        u32::try_from(instance_count),
    ) {
        (Ok(batch_index), Ok(index_count), Ok(instance_count)) => {
            stats.record_patch_draw(batch_index, pass, geometry, index_count, instance_count);
        }
        _ => stats.record_invalid_draw(),
    }
}

/// Render a frame using the backend-neutral ordered draw-pass plan.
fn program_for_pass(programs: &Programs, pass: RenderPass) -> glow::Program {
    match pass {
        RenderPass::PbrOpaque | RenderPass::PbrTransparent => programs.pbr,
        RenderPass::Matcap | RenderPass::Lod => programs.matcap,
        RenderPass::Wire => programs.wire,
        RenderPass::Normals => programs.normals,
        RenderPass::Stretch => programs.stretch,
    }
}

/// Legacy rollback dispatcher. New diagnostic paths should consume
/// [`RenderExecution`] through [`render_diagnostic_execution`].
pub fn render_frame(
    gl: &glow::Context,
    programs: &Programs,
    style: RenderStyle,
    camera: &Camera,
    batches: &[RenderBatch],
    vtx_ubo: &VertexUniformBuf,
    wire_ubo: &WireUniformBuf,
) -> RenderSubmissionStats {
    let mut stats = RenderSubmissionStats::default();
    vtx_ubo.bind(gl);

    for draw_pass in render_draw_passes(style) {
        let program = program_for_pass(programs, draw_pass.pass);
        unsafe {
            gl.use_program(Some(program));
        }
        if draw_pass.pass == RenderPass::Wire {
            // All adaptive wire draws use the density heatmap; the fallback
            // color is ignored by the shader and therefore frame-global state.
            wire_ubo.upload(gl, [0.0; 3], true);
            wire_ubo.bind(gl);
        }
        stats.merge(draw_batches(
            gl,
            camera,
            batches,
            vtx_ubo,
            draw_pass.pass,
            draw_pass.geometry,
            draw_pass.batches,
        ));
    }

    unsafe {
        gl.bind_vertex_array(None);
    }
    stats
}

/// Execute canonical diagnostic patch draws against exact retained WebGL
/// resources. The complete semantic/resource preflight runs before any GL
/// state mutation, so callers can fall back to [`render_frame`] atomically.
#[allow(clippy::too_many_arguments)]
pub fn render_diagnostic_execution(
    gl: &glow::Context,
    programs: &Programs,
    execution: RenderExecution<'_, '_>,
    camera: &Camera,
    batches: &[RenderBatch],
    vtx_ubo: &VertexUniformBuf,
    wire_ubo: &WireUniformBuf,
) -> Result<RenderSubmissionStats, WebGlRenderExecutionError> {
    validate_diagnostic_execution(execution, batches)?;

    let mut stats = RenderSubmissionStats::default();
    let mut active_pass = None;
    let mut vertex_state = None;
    vtx_ubo.bind(gl);
    for command in execution {
        let ResolvedRenderCommand::DrawPatches {
            batch_index,
            pass,
            geometry,
            ..
        } = command
        else {
            continue;
        };
        if active_pass != Some(pass) {
            unsafe {
                gl.use_program(Some(program_for_pass(programs, pass)));
            }
            if pass == RenderPass::Wire {
                wire_ubo.upload(gl, [0.0; 3], true);
                wire_ubo.bind(gl);
            }
            active_pass = Some(pass);
            vertex_state = None;
        }
        let batch = &batches[batch_index as usize];
        stats.merge(draw_batch(
            gl,
            camera,
            vtx_ubo,
            &mut vertex_state,
            batch_index as usize,
            batch,
            pass,
            geometry,
        ));
    }
    unsafe {
        gl.bind_vertex_array(None);
    }
    Ok(stats)
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
        false,
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
    use quilting_core::batch::{RenderBatchId, RenderBatchKey, RenderBatchMember};
    use quilting_core::render::{
        FocusFieldPacket, RenderEntityTransform, RenderFrame, RenderFrameOptions,
        RenderPoseIdentity, RenderSceneSnapshot, RenderView,
    };
    use quilting_core::screen_partition::ScreenPatchLeafId;

    #[cfg(not(target_arch = "wasm32"))]
    fn resident_fixture(enabled: bool) -> (RenderSceneSnapshot, RenderBatch) {
        let key = RenderBatchKey {
            lod: [1; 3],
            parity_bucket: 0,
            material_index: 0,
            render_node_index: 0,
        };
        let snapshot = RenderBatchSnapshot {
            id: RenderBatchId::complete(key),
            members: vec![RenderBatchMember {
                face_index: 0,
                leaf_id: ScreenPatchLeafId::ROOT,
                node_index: 0,
                edge_lods: [1; 3],
                permutation_index: 0,
                vertex_lods: [1; 3],
            }],
            triangle_index_count: 6,
            line_index_count: 6,
            transform: RenderEntityTransform {
                mobius: IDENTITY_MOBIUS,
                orientation_sign: 1,
                euclidean_model: IDENTITY_MATRIX,
                euclidean_normal: IDENTITY_MATRIX,
            },
            enabled,
            pbr_class: PbrDrawClass::Opaque,
        };
        let vao = glow::NativeVertexArray(std::num::NonZeroU32::new(1).unwrap());
        let batch = RenderBatch {
            mesh: MeshDraw {
                tri_vao: vao,
                line_vao: vao,
                num_tri_indices: 6,
                num_line_indices: 6,
                tri_index_offset: 0,
                line_index_offset: 0,
                num_instances: i32::from(enabled),
            },
            suppress_source_roots: false,
            perm_parity: 1.0,
            material_index: 0,
            pbr_class: PbrDrawClass::Opaque,
            render_node_index: 0,
            mobius: IDENTITY_MOBIUS,
            orientation_sign: 1,
            euclidean_model: IDENTITY_MATRIX,
            euclidean_normal: IDENTITY_MATRIX,
        };
        (
            RenderSceneSnapshot {
                revision: 4,
                materials: Vec::new(),
                suppressed_root_faces: Vec::new(),
                batches: vec![snapshot],
            },
            batch,
        )
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn frame(scene: &RenderSceneSnapshot, style: RenderStyle) -> RenderFrame {
        RenderFrame::build(
            7,
            RenderPoseIdentity {
                asset_revision: 2,
                pose_revision: 3,
            },
            style,
            RenderView {
                viewport: [640, 480],
                mvp: IDENTITY_MATRIX,
                model_view: IDENTITY_MATRIX,
                camera_position: [0.0, 0.0, 3.0],
                selected_node: None,
                focus: FocusFieldPacket {
                    sphere: [0.0, 0.0, 0.0, 1.0],
                    enabled: false,
                },
            },
            RenderFrameOptions::default(),
            scene,
        )
        .unwrap()
    }

    #[test]
    fn signed_submission_counts_are_validated_before_accounting() {
        let mut stats = RenderSubmissionStats::default();
        record_indexed_submission(
            &mut stats,
            0,
            RenderPass::Matcap,
            RenderGeometry::Triangles,
            12,
            2,
        );
        record_indexed_submission(
            &mut stats,
            1,
            RenderPass::Wire,
            RenderGeometry::Lines,
            8,
            0,
        );
        record_indexed_submission(
            &mut stats,
            2,
            RenderPass::Normals,
            RenderGeometry::Triangles,
            -1,
            2,
        );
        record_indexed_submission(
            &mut stats,
            3,
            RenderPass::Stretch,
            RenderGeometry::Triangles,
            12,
            -1,
        );

        assert_eq!(stats.draw_calls, 4);
        assert_eq!(stats.zero_instance_draw_calls, 1);
        assert_eq!(stats.invalid_draw_calls, 2);
        assert_eq!(stats.submitted_instances, 2);
        assert_eq!(stats.triangles, 8);
        assert_eq!(stats.lines, 0);
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn diagnostic_execution_fails_closed_on_stale_webgl_residency() {
        let (scene, batch) = resident_fixture(true);
        let frame = frame(&scene, RenderStyle::Matcap);
        validate_diagnostic_execution(frame.execution(&scene).unwrap(), &[batch]).unwrap();

        let mut stale = batch;
        stale.mesh.num_instances = 0;
        assert_eq!(
            validate_diagnostic_execution(frame.execution(&scene).unwrap(), &[stale]),
            Err(WebGlRenderExecutionError::BatchMismatch {
                batch_index: 0,
                field: WebGlBatchField::InstanceCount,
            })
        );
        stale = batch;
        stale.mesh.num_tri_indices = 3;
        assert_eq!(
            validate_diagnostic_execution(frame.execution(&scene).unwrap(), &[stale]),
            Err(WebGlRenderExecutionError::BatchMismatch {
                batch_index: 0,
                field: WebGlBatchField::TriangleIndexCount,
            })
        );
        assert_eq!(
            validate_diagnostic_execution(frame.execution(&scene).unwrap(), &[]),
            Err(WebGlRenderExecutionError::BatchCount {
                expected: 1,
                actual: 0,
            })
        );
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn diagnostic_execution_preserves_zero_work_and_rejects_pbr() {
        let (hidden_scene, hidden_batch) = resident_fixture(false);
        let hidden_frame = frame(&hidden_scene, RenderStyle::Wire);
        let execution = hidden_frame.execution(&hidden_scene).unwrap();
        validate_diagnostic_execution(execution, &[hidden_batch]).unwrap();
        let stats = execution.submission_stats();
        assert_eq!(stats.draw_calls, 1);
        assert_eq!(stats.zero_instance_draw_calls, 1);
        assert_eq!(stats.submitted_instances, 0);

        let (scene, batch) = resident_fixture(true);
        let pbr = frame(&scene, RenderStyle::Pbr);
        validate_pbr_execution(pbr.execution(&scene).unwrap(), &[batch]).unwrap();
        let plan = WebGlPbrCommandPlan::build(pbr.execution(&scene).unwrap(), &[batch]).unwrap();
        assert_eq!(plan.batches(RenderPass::PbrOpaque), &[0]);
        assert!(plan.batches(RenderPass::PbrTransparent).is_empty());
        assert!(!plan.builds_transmission_pyramid());
        assert!(!plan.runs_focus_postprocess());
        assert_eq!(
            webgl_pbr_draws(Some(&plan), &[batch], RenderPass::PbrOpaque)
                .map(|(batch_index, _)| batch_index)
                .collect::<Vec<_>>(),
            vec![0],
        );
        assert_eq!(
            webgl_pbr_draws(None, &[batch], RenderPass::PbrOpaque)
                .map(|(batch_index, _)| batch_index)
                .collect::<Vec<_>>(),
            vec![0],
        );
        assert_eq!(
            validate_diagnostic_execution(pbr.execution(&scene).unwrap(), &[batch]),
            Err(WebGlRenderExecutionError::UnsupportedCommand("PBR draw"))
        );

        let (mut scene, mut batch) = resident_fixture(true);
        scene.batches[0].pbr_class = PbrDrawClass::Transmission;
        batch.pbr_class = PbrDrawClass::Transmission;
        let pbr = frame(&scene, RenderStyle::Pbr);
        let plan = WebGlPbrCommandPlan::build(pbr.execution(&scene).unwrap(), &[batch]).unwrap();
        assert!(plan.batches(RenderPass::PbrOpaque).is_empty());
        assert_eq!(plan.batches(RenderPass::PbrTransparent), &[0]);
        assert!(plan.builds_transmission_pyramid());
    }
}
