mod app_shadow;
mod adaptive_screen;
mod auxiliary_programs;
pub mod main_renderer;
mod navigation;
mod render_shadow;
mod round_shadow;
mod route_shadow;
mod surface_walk;
mod surface_runtime;

pub use app_shadow::{
    encode_local_presence_envelope, map_space_mouse_camera_frame, HyperscopeAppShadow,
};
pub use route_shadow::{canonicalize_hyperscope_route, hyperscope_control_specs};
pub use surface_walk::HyperscopeSurfaceWalk;

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use quilting_core::atlas::{TessellationAtlas, BuildMode};
use quilting_core::batch;
use quilting_renderer::compute::{
    build_composed_lod_model, prepare_lod_atlas_lookup, prepare_lod_dispatch_state,
    exact_f32_slice_fingerprint, prepare_lod_model, prepared_lod_model_fingerprint,
    LodAnimationSource, LodAtlasLookup, LodCompute, LodModelData, LodModelResidency,
    PreparedLodModelFingerprint, StagedLodReadback,
};
use quilting_core::instance_layout::{self, InstanceWriter};
use quilting_core::quaternion::{Quat, Mobius};
use quilting_core::sampling::PatchConfig;
use quilting_core::triangle;
use std::cell::{Ref, RefCell};
use std::collections::{HashMap, HashSet};
use glow::HasContext;

/// Performance.mark/measure helper for profiling in Chrome DevTools.
/// Works in both Window and Worker contexts.
pub fn perf_mark(name: &str) {
    let global = js_sys::global();
    if let Ok(perf) = js_sys::Reflect::get(&global, &"performance".into()) {
        if !perf.is_undefined() {
            let _ = js_sys::Reflect::apply(
                &js_sys::Reflect::get(&perf, &"mark".into()).unwrap().into(),
                &perf,
                &js_sys::Array::of1(&name.into()),
            );
        }
    }
}

pub fn perf_measure(name: &str, start: &str, end: &str) {
    let global = js_sys::global();
    if let Ok(perf) = js_sys::Reflect::get(&global, &"performance".into()) {
        if !perf.is_undefined() {
            let _ = js_sys::Reflect::apply(
                &js_sys::Reflect::get(&perf, &"measure".into()).unwrap().into(),
                &perf,
                &js_sys::Array::of3(&name.into(), &start.into(), &end.into()),
            );
        }
    }
}

#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
    let mut config = wasm_tracing::WasmLayerConfig::new();
    config.set_max_level(tracing::Level::INFO);
    let _ = wasm_tracing::set_as_global_default_with_config(config);
}

/// Stored glTF data for animation switching without re-parsing.
struct StoredGltfData {
    animations: Vec<quilting_gltf::animation::Animation>,
    skins: Vec<quilting_gltf::animation::Skin>,
    nodes: Vec<quilting_gltf::scene::Node>,
    combined: quilting_gltf::mesh::Primitive,
    face_material_indices: Vec<Option<usize>>,
    face_node_indices: Vec<usize>,
    primary_skin_idx: Option<usize>,
    active_animation: usize,
    /// Per-frame evaluator for GPU skinning (replaces prebake).
    evaluator: Option<quilting_gltf::evaluator::AnimationEvaluator>,
    /// Normalization: center and 1/extent for rest-pose bounding box.
    norm_center: [f64; 3],
    norm_scale: f64,
    /// The main renderer pose upload and worker-side LOD dispatch normally ask
    /// for the same animation time in adjacent messages. Retain the normalized
    /// result so joint evaluation and matrix normalization happen once.
    cached_pose: RefCell<Option<CachedAnimationPose>>,
}

struct CachedAnimationPose {
    time_bits: u64,
    pose: quilting_gltf::evaluator::AnimationPose,
}

fn mat4_mul_f32(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    for col in 0..4 {
        for row in 0..4 {
            out[col * 4 + row] = a[row] * b[col * 4]
                + a[4 + row] * b[col * 4 + 1]
                + a[8 + row] * b[col * 4 + 2]
                + a[12 + row] * b[col * 4 + 3];
        }
    }
    out
}

fn normalize_animation_pose(
    pose: &mut quilting_gltf::evaluator::AnimationPose,
    center: [f64; 3],
    scale: f64,
) {
    if pose.joint_matrices.is_empty() {
        return;
    }

    let inverse_scale = if scale.abs() > 1e-10 { 1.0 / scale } else { 1.0 };
    let sf = scale as f32;
    let isf = inverse_scale as f32;
    let cx = center[0] as f32;
    let cy = center[1] as f32;
    let cz = center[2] as f32;
    let norm = [
        sf, 0.0, 0.0, 0.0, 0.0, sf, 0.0, 0.0,
        0.0, 0.0, sf, 0.0, -cx * sf, -cy * sf, -cz * sf, 1.0,
    ];
    let unnorm = [
        isf, 0.0, 0.0, 0.0, 0.0, isf, 0.0, 0.0,
        0.0, 0.0, isf, 0.0, cx, cy, cz, 1.0,
    ];

    for matrix in pose.joint_matrices.chunks_exact_mut(16) {
        let source: [f32; 16] = matrix.try_into().expect("16-float joint matrix");
        let normalized = mat4_mul_f32(&norm, &mat4_mul_f32(&source, &unnorm));
        matrix.copy_from_slice(&normalized);
    }
}

fn normalized_animation_pose(
    data: &StoredGltfData,
    t: f64,
) -> Option<Ref<'_, quilting_gltf::evaluator::AnimationPose>> {
    let evaluator = data.evaluator.as_ref()?;
    let time_bits = t.to_bits();
    {
        let mut cached = data.cached_pose.borrow_mut();
        if cached.as_ref().map(|entry| entry.time_bits) != Some(time_bits) {
            let mut pose = evaluator.evaluate(t);
            normalize_animation_pose(&mut pose, data.norm_center, data.norm_scale);
            *cached = Some(CachedAnimationPose { time_bits, pose });
        }
    }

    Some(Ref::map(data.cached_pose.borrow(), |cached| {
        &cached.as_ref().expect("animation pose was cached").pose
    }))
}

struct PendingAnimatedLods {
    fence: glow::Fence,
    readback: StagedLodReadback,
    classified_faces: usize,
    resident_faces: usize,
    subject_records: usize,
    pose_stamp: Option<PendingLodPoseStamp>,
    pose: Option<PendingLodPose>,
}

#[derive(Clone, Copy)]
struct PendingLodPoseStamp {
    clip_time: f64,
    sample_time: f64,
    revision: u32,
    continuity_epoch: u32,
}

struct PendingLodPose {
    joint_matrices: Vec<f32>,
    morph_weights: Vec<f32>,
}

thread_local! {
    static ATLAS: RefCell<Option<TessellationAtlas>> = RefCell::new(None);
    static GPU_COMPUTE: RefCell<Option<(glow::Context, LodCompute)>> = RefCell::new(None);
    static PENDING_ANIMATED_LODS: RefCell<Option<PendingAnimatedLods>> = RefCell::new(None);
    static ANIMATED_LOD_DELTA: RefCell<batch::FaceLodDeltaEncoder> =
        RefCell::new(batch::FaceLodDeltaEncoder::default());
    static LOD_COMPUTE_MODEL: RefCell<Option<LodModelResidency>> = RefCell::new(None);
    static LOD_COMPUTE_MODEL_FINGERPRINT: RefCell<Option<PreparedLodModelFingerprint>> =
        RefCell::new(None);
    /// Track which (canonical_lod, perm_parity) tessellation keys have been sent to JS.
    /// JS caches the GPU buffers, so we skip re-sending the bary/triangle data.
    static SENT_TESS: RefCell<std::collections::HashSet<String>> = RefCell::new(std::collections::HashSet::new());
}

/// Build the tessellation atlas client-side. Call once at init.
/// max_lod_exp: build LODs from 2^0 to 2^max_lod_exp (e.g., 8 → up to 256)
/// mode: "direct" or "hierarchical"
#[wasm_bindgen]
pub fn build_atlas(max_lod_exp: u32, mode: &str) -> f64 {
    let config = PatchConfig { k_candidates: 30, seed: 42 };
    let lods: Vec<u32> = (0..=max_lod_exp).map(|n| 1u32 << n).collect();
    let build_mode = match mode {
        "hierarchical" => BuildMode::Hierarchical,
        _ => BuildMode::Direct,
    };

    let start = js_sys::Date::now();
    let atlas = TessellationAtlas::build_with_mode(&lods, &config, build_mode);
    let elapsed = js_sys::Date::now() - start;

    ATLAS.with(|a| *a.borrow_mut() = Some(atlas));
    // New atlas means new tessellation data — clear the sent cache
    SENT_TESS.with(|s| s.borrow_mut().clear());

    elapsed
}

/// Build only the canonical patches reachable under the selected validated
/// within-face LOD policy. This keeps the entire build in one WASM call so
/// browser startup does not schedule one worker message per patch. Returns a
/// negative duration for an unsupported ratio without replacing the atlas.
#[wasm_bindgen]
pub fn build_required_atlas(max_lod_exp: u32, max_face_edge_ratio: u32) -> f64 {
    let Some(grading) = batch::FaceLodGrading::from_ratio(max_face_edge_ratio) else {
        return -1.0;
    };
    let config = PatchConfig { k_candidates: 30, seed: 42 };
    let lods: Vec<u32> = (0..=max_lod_exp.min(30))
        .map(|exponent| 1u32 << exponent)
        .collect();
    let keys = quilting_core::atlas::ratio_bounded_canonical_triples(
        &lods,
        grading.ratio(),
    );

    let start = js_sys::Date::now();
    let atlas = TessellationAtlas::build_hierarchical_for_keys(&lods, &keys, &config);
    let elapsed = js_sys::Date::now() - start;
    ATLAS.with(|atlas_cell| *atlas_cell.borrow_mut() = Some(atlas));
    SENT_TESS.with(|sent| sent.borrow_mut().clear());
    elapsed
}

fn replace_retained_state<T, P>(
    current: &mut Option<T>,
    pending: &mut Option<P>,
    replacement: T,
    release: impl FnOnce(T, Option<P>),
) {
    match current.take() {
        Some(previous) => release(previous, pending.take()),
        None => {
            debug_assert!(pending.is_none(), "pending work exists without retained state");
            pending.take();
        }
    }
    *current = Some(replacement);
}

#[cfg(test)]
mod gpu_compute_lifecycle_tests {
    use super::replace_retained_state;

    #[test]
    fn replacement_releases_pending_work_with_the_previous_state() {
        let mut current = Some("old context");
        let mut pending = Some("old fence");
        let mut released = None;

        replace_retained_state(
            &mut current,
            &mut pending,
            "new context",
            |old, work| released = Some((old, work)),
        );

        assert_eq!(released, Some(("old context", Some("old fence"))));
        assert_eq!(current, Some("new context"));
        assert_eq!(pending, None);
    }

    #[test]
    fn first_install_does_not_run_retirement_cleanup() {
        let mut current = None;
        let mut pending: Option<&str> = None;
        let mut released = false;

        replace_retained_state(
            &mut current,
            &mut pending,
            "first context",
            |_, _| released = true,
        );

        assert!(!released);
        assert_eq!(current, Some("first context"));
    }

}

fn release_pending_lod_job(
    gl: &glow::Context,
    compute: &mut LodCompute,
    job: PendingAnimatedLods,
) {
    unsafe { gl.delete_sync(job.fence); }
    compute.discard_staged_readback(gl, job.readback);
}

fn install_gpu_compute(gl: glow::Context, compute: LodCompute) {
    PENDING_ANIMATED_LODS.with(|pending| {
        GPU_COMPUTE.with(|gpu_compute| {
            replace_retained_state(
                &mut gpu_compute.borrow_mut(),
                &mut pending.borrow_mut(),
                (gl, compute),
                |(old_gl, mut old_compute), pending_job| {
                    if let Some(job) = pending_job {
                        release_pending_lod_job(&old_gl, &mut old_compute, job);
                    }
                    old_compute.destroy(&old_gl);
                },
            );
        });
    });
}

/// Initialize GPU compute context using OffscreenCanvas.
/// Call once per worker — creates a WebGL2 context for transform feedback LOD computation.
#[wasm_bindgen]
pub fn init_gpu_compute(max_faces: u32) -> bool {
    // Size must accommodate pass 1 FBO viewport (4096 × ceil(num_faces/4096))
    let canvas = web_sys::OffscreenCanvas::new(4096, 4096);
    let canvas = match canvas {
        Ok(c) => c,
        Err(_) => return false,
    };

    let gl_ctx = canvas.get_context("webgl2");
    let gl_ctx = match gl_ctx {
        Ok(Some(ctx)) => ctx.dyn_into::<web_sys::WebGl2RenderingContext>().ok(),
        _ => None,
    };
    let gl_ctx = match gl_ctx {
        Some(c) => c,
        None => return false,
    };

    // Enable float FBO rendering for pass 1 (LOD exponents → RGBA32F texture)
    gl_ctx.get_extension("EXT_color_buffer_float").ok();

    let gl = glow::Context::from_webgl2_context(gl_ctx);

    match LodCompute::new(&gl, max_faces as usize) {
        Ok(compute) => {
            install_gpu_compute(gl, compute);
            true
        }
        Err(e) => {
            web_sys::console::warn_1(&format!("GPU compute init failed: {e}").into());
            false
        }
    }
}

thread_local! {
    /// Sorted atlas keys — maps atlas_index back to canonical LOD triples.
    static LOD_ATLAS_KEYS: RefCell<Vec<[u32; 3]>> = RefCell::new(Vec::new());
    /// Maximum LOD the atlas supports — clamp here instead of falling to LOD 2.
    static LOD_MAX: RefCell<f32> = RefCell::new(512.0);
}

fn upload_lod_compute_model(upload: LodModelData) -> Result<bool, String> {
    let prepared = prepare_lod_model(upload)?;
    let fingerprint = prepared_lod_model_fingerprint(&prepared);
    let atlas = lod_atlas_snapshot()?;

    let residency = GPU_COMPUTE.with(|gpu_compute| {
        let mut gpu_compute = gpu_compute.borrow_mut();
        let Some((gl, compute)) = gpu_compute.as_mut() else {
            return None;
        };
        Some(compute.upload_model(gl, &prepared, &atlas.lut))
    });
    let Some(residency) = residency else {
        return Ok(false);
    };
    LOD_COMPUTE_MODEL.with(|current| *current.borrow_mut() = Some(residency));
    LOD_COMPUTE_MODEL_FINGERPRINT.with(|current| {
        *current.borrow_mut() = Some(fingerprint);
    });
    install_lod_atlas_snapshot(atlas);
    Ok(true)
}

/// Exact immutable model identity retained by the worker classifier. The
/// textual form preserves 64-bit hashes across the JavaScript number boundary.
#[wasm_bindgen]
pub fn lod_compute_model_fingerprint() -> String {
    LOD_COMPUTE_MODEL_FINGERPRINT.with(|fingerprint| {
        fingerprint
            .borrow()
            .as_ref()
            .copied()
            .map(PreparedLodModelFingerprint::stable_text)
            .unwrap_or_default()
    })
}

fn lod_atlas_snapshot() -> Result<LodAtlasLookup, String> {
    ATLAS.with(|atlas_cell| {
        let atlas_cell = atlas_cell.borrow();
        let atlas = atlas_cell
            .as_ref()
            .ok_or_else(|| "LOD atlas is not resident".to_string())?;
        prepare_lod_atlas_lookup(atlas.patches.keys().copied())
    })
}

fn install_lod_atlas_snapshot(atlas: LodAtlasLookup) {
    LOD_MAX.with(|max| *max.borrow_mut() = atlas.max_lod);
    LOD_ATLAS_KEYS.with(|keys| *keys.borrow_mut() = atlas.keys);
    reset_animated_lod_delta();
}

/// Refresh only the atlas lookup used by the retained GPU classifier.
/// Geometry, animation textures, and adjacency remain resident. Callers must
/// cancel an in-flight classification before replacing this texture.
#[wasm_bindgen]
pub fn refresh_lod_compute_atlas() -> Result<bool, JsValue> {
    let atlas = lod_atlas_snapshot().map_err(|error| JsValue::from_str(&error))?;
    let uploaded = GPU_COMPUTE.with(|gpu_compute| {
        let mut gpu_compute = gpu_compute.borrow_mut();
        let Some((gl, compute)) = gpu_compute.as_mut() else {
            return false;
        };
        compute.upload_atlas_lut(gl, &atlas.lut);
        true
    });
    if uploaded {
        install_lod_atlas_snapshot(atlas);
    }
    Ok(uploaded)
}

fn stored_gltf_lod_upload(data: &StoredGltfData) -> Result<LodModelData, String> {
    let combined = &data.combined;
    let num_vertices = combined.positions.len();
    let mut positions = vec![0.0f32; num_vertices * 3];
    for (vertex, position) in combined.positions.iter().enumerate() {
        for axis in 0..3 {
            positions[vertex * 3 + axis] =
                ((position[axis] - data.norm_center[axis]) * data.norm_scale) as f32;
        }
    }
    let faces: Vec<[u32; 3]> = combined.triangles.iter()
        .map(|face| face.map(|vertex| vertex as u32))
        .collect();
    let (joint_indices, joint_weights) = match (
        combined.joint_indices.as_ref(),
        combined.joint_weights.as_ref(),
    ) {
        (Some(indices), Some(weights)) if indices.len() == num_vertices
            && weights.len() == num_vertices => (indices.clone(), weights.clone()),
        (None, None) => (vec![[0; 4]; num_vertices], vec![[0.0; 4]; num_vertices]),
        _ => return Err("stored glTF has incomplete skinning data".to_string()),
    };
    let num_morph_targets = combined.morph_targets.len();
    let mut morph_deltas = Vec::with_capacity(num_morph_targets * num_vertices * 3);
    for target in &combined.morph_targets {
        for vertex in 0..num_vertices {
            let delta = target.get(vertex).copied().unwrap_or([0.0; 3]);
            morph_deltas.extend(delta.map(|coordinate| coordinate as f32 * data.norm_scale as f32));
        }
    }
    if data.face_node_indices.len() != faces.len() {
        return Err("stored glTF has incomplete face ownership".to_string());
    }
    Ok(LodModelData {
        positions,
        faces,
        joint_indices,
        joint_weights,
        morph_deltas,
        num_morph_targets,
        face_nodes: data.face_node_indices.clone(),
    })
}

fn composed_lod_upload(
    data: &StoredGltfData,
    instances: &[f32],
    face_nodes: &[i32],
    total_vertices: u32,
    primary_faces: u32,
) -> Result<LodModelData, String> {
    if primary_faces as usize != data.combined.triangles.len() {
        return Err("composed LOD primary face boundary does not match the retained model".to_string());
    }
    let num_vertices = total_vertices as usize;
    let face_nodes: Vec<usize> = face_nodes.iter().enumerate()
        .map(|(face, &node)| {
            usize::try_from(node)
                .map_err(|_| format!("composed LOD face {face} has a negative node"))
        })
        .collect::<Result<_, _>>()?;
    let primary_vertices = data.combined.positions.len();
    let skinning = match (
        data.combined.joint_indices.as_ref(),
        data.combined.joint_weights.as_ref(),
    ) {
        (Some(indices), Some(weights)) if indices.len() == primary_vertices
            && weights.len() == primary_vertices => Some((indices.as_slice(), weights.as_slice())),
        (None, None) => None,
        _ => return Err("retained primary model has incomplete skinning data".to_string()),
    };
    let num_morph_targets = data.combined.morph_targets.len();
    let mut primary_morph_deltas = vec![0.0f32; num_morph_targets * primary_vertices * 3];
    for (target_index, target) in data.combined.morph_targets.iter().enumerate() {
        for vertex in 0..primary_vertices {
            let delta = target.get(vertex).copied().unwrap_or([0.0; 3]);
            let offset = (target_index * primary_vertices + vertex) * 3;
            for axis in 0..3 {
                primary_morph_deltas[offset + axis] =
                    delta[axis] as f32 * data.norm_scale as f32;
            }
        }
    }
    build_composed_lod_model(
        instances,
        &face_nodes,
        num_vertices,
        primary_faces as usize,
        LodAnimationSource {
            primary_vertices,
            joint_indices: skinning.map(|(indices, _)| indices),
            joint_weights: skinning.map(|(_, weights)| weights),
            morph_deltas: &primary_morph_deltas,
            num_morph_targets,
        },
    )
}

/// Upload static animation data to the GPU compute context for per-frame LOD.
/// Call once after loading a model. Uploads:
/// - Rest-pose positions as a texture
/// - Face indices for the LOD compute shader
/// - Skinning texture (joint indices + weights) if skeletal animation
/// - Morph delta texture if morph target animation
/// - Atlas LUT for LOD → atlas index mapping
/// Returns true if upload succeeded (GPU compute context available).
#[wasm_bindgen]
pub fn upload_model_to_compute() -> bool {
    GLTF_DATA.with(|gd| {
        let data = gd.borrow();
        let Some(data) = data.as_ref() else { return false };
        match stored_gltf_lod_upload(data).and_then(upload_lod_compute_model) {
            Ok(uploaded) => uploaded,
            Err(error) => {
                web_sys::console::warn_1(&format!("LOD model upload failed: {error}").into());
                false
            }
        }
    })
}

/// Replace the primary-only LOD topology with the renderer's packed scene.
/// Authored nodes retain separate topology domains while sharing the primary
/// worker's GPU context, animation pose, camera, and conformal state.
#[wasm_bindgen]
pub fn upload_composed_model_to_compute(
    instances: &[f32],
    face_nodes: &[i32],
    total_vertices: u32,
    primary_faces: u32,
) -> Result<bool, JsValue> {
    GLTF_DATA.with(|stored| {
        let stored = stored.borrow();
        let data = stored.as_ref()
            .ok_or_else(|| JsValue::from_str("primary glTF is not resident"))?;
        let upload = composed_lod_upload(
            data,
            instances,
            face_nodes,
            total_vertices,
            primary_faces,
        ).map_err(|error| JsValue::from_str(&error))?;
        upload_lod_compute_model(upload).map_err(|error| JsValue::from_str(&error))
    })
}

/// Sample Möbius stretch at the normalized centroid of each stored source face.
/// Returns [min_stretch, max_stretch] as sigmoid-mapped values matching the vertex shader.
/// Runs on CPU, async-safe — call from a worker without blocking rendering.
#[wasm_bindgen]
pub fn sample_stretch_range(mobius: &[f32]) -> Vec<f32> {
    if mobius.len() < 16 { return vec![0.5, 0.5]; }
    let transform = Mobius::new(
        Quat::new(mobius[0] as f64, mobius[1] as f64, mobius[2] as f64, mobius[3] as f64),
        Quat::new(mobius[4] as f64, mobius[5] as f64, mobius[6] as f64, mobius[7] as f64),
        Quat::new(mobius[8] as f64, mobius[9] as f64, mobius[10] as f64, mobius[11] as f64),
        Quat::new(mobius[12] as f64, mobius[13] as f64, mobius[14] as f64, mobius[15] as f64),
    );

    GLTF_DATA.with(|stored| {
        let stored = stored.borrow();
        let Some(data) = stored.as_ref() else { return vec![0.5, 0.5] };
        let center = data.norm_center;
        let scale = data.norm_scale;
        let mut min_s = f32::INFINITY;
        let mut max_s = f32::NEG_INFINITY;

        for face in &data.combined.triangles {
            let p0 = data.combined.positions[face[0]];
            let p1 = data.combined.positions[face[1]];
            let p2 = data.combined.positions[face[2]];
            let point = Quat::from_point(
                ((p0[0] + p1[0] + p2[0]) / 3.0 - center[0]) * scale,
                ((p0[1] + p1[1] + p2[1]) / 3.0 - center[1]) * scale,
                ((p0[2] + p1[2] + p2[2]) / 3.0 - center[2]) * scale,
            );
            let stretch = transform.conformal_scale_at(point) as f32;
            let log_s = stretch.log2();
            let sig = 1.0 / (1.0 + (-log_s * 0.25_f32).exp());
            min_s = min_s.min(sig);
            max_s = max_s.max(sig);
        }

        if min_s.is_infinite() { vec![0.5, 0.5] } else { vec![min_s, max_s] }
    })
}

#[wasm_bindgen]
pub fn debug_gpu_compute_state() -> String {
    GPU_COMPUTE.with(|gc| {
        let gc = gc.borrow();
        match gc.as_ref() {
            None => "GPU_COMPUTE is None".to_string(),
            Some((_gl, compute)) => {
                let (
                    gpu_pool,
                    packed_pool,
                    decoded_pool,
                    gpu_created,
                    gpu_resized,
                    packed_created,
                    decoded_created,
                ) = compute.readback_pool_stats();
                format!(
                "pass1_tex={} adj_tex={} readback_pool=gpu:{} packed:{} decoded:{} created_gpu:{} resized_gpu:{} created_packed:{} created_decoded:{}",
                compute.has_pass1_texture(),
                compute.has_adjacency_texture(),
                gpu_pool, packed_pool, decoded_pool, gpu_created, gpu_resized,
                packed_created, decoded_created,
            )},
        }
    })
}

/// Dispatch per-frame GPU LOD computation without waiting for readback.
///
/// 1. Evaluates animation (morph + skeletal) at time t
/// 2. Uploads joint matrices + morph weights to worker GPU
/// 3. Runs transform feedback LOD compute (Möbius + medians + snap)
/// 4. Classifies every face once, selecting its authored-node state on GPU
/// 5. Copies one coherent result into GPU staging and inserts one fence
///
/// `subject_states` is a packed sequence of
/// `[node_index, mobius(16), euclidean_model(16)]` records. When present, each
/// node's faces are classified with their own extracted subject/view state.
///
/// Returns true when a job was dispatched. Call [`poll_animated_lods`] until
/// it returns a completed sparse/full classification object. Uses mesh_radius
/// computed at upload time (not hardcoded).
#[allow(clippy::too_many_arguments)] // Flat arrays form the worker/WASM ABI.
#[wasm_bindgen]
pub fn dispatch_animated_lods(
    t: f64,
    pose_sample_time: f64,
    pose_revision: u32,
    pose_continuity_epoch: u32,
    mobius: &[f32],
    subject_states: &[f32],
    face_limit: u32,
    density: f32,
    min_px: f32,
    vp_matrix: &[f32],
    vp_width: f32,
    vp_height: f32,
    capture_pose: bool,
) -> bool {
    if PENDING_ANIMATED_LODS.with(|pending| pending.borrow().is_some()) {
        return false;
    }
    GLTF_DATA.with(|gd| {
        let data = match gd.try_borrow() {
            Ok(d) => d,
            Err(_) => return false, // concurrent borrow during model load
        };
        let data = match data.as_ref() {
            Some(d) => d,
            None => return false,
        };

        let Some(compute_model) = LOD_COMPUTE_MODEL.with(|model| model.borrow().clone()) else {
            return false;
        };
        let resident_faces = compute_model.num_faces;
        let num_faces = if face_limit == 0 {
            resident_faces
        } else {
            resident_faces.min(face_limit as usize)
        };
        let num_vertices = compute_model.num_vertices;
        let mesh_radius = compute_model.mesh_radius;
        let mut legacy_mobius = [0.0f32; 16];
        for (destination, &source) in legacy_mobius.iter_mut().zip(mobius) {
            *destination = source;
        }
        let dispatch_state = prepare_lod_dispatch_state(
            subject_states,
            &compute_model,
            num_faces,
            legacy_mobius,
        );

        perf_mark("lod-wasm-start");

        // 1. Evaluate animation pose at time t
        // t < 0 signals "skip animation" for a static or not-yet-posed model.
        // Paused animated models still provide their last accepted pose stamp.
        perf_mark("lod-anim-eval-start");
        let use_anim = t >= 0.0;
        let pose_stamp = if use_anim {
            if let Err(error) = surface_runtime::validate_pose_stamp(
                t,
                pose_sample_time,
                pose_revision,
                pose_continuity_epoch,
            ) {
                web_sys::console::warn_1(&format!("LOD pose stamp rejected: {error}").into());
                return false;
            }
            Some(PendingLodPoseStamp {
                clip_time: t,
                sample_time: pose_sample_time,
                revision: pose_revision,
                continuity_epoch: pose_continuity_epoch,
            })
        } else {
            None
        };
        let pose = if use_anim {
            normalized_animation_pose(data, t.max(0.0))
        } else {
            None
        };
        let (joint_matrices, morph_weights, num_joints, num_morph) = match pose.as_deref() {
            Some(pose) => (
                pose.joint_matrices.as_slice(),
                pose.morph_weights.as_slice(),
                pose.num_joints as u32,
                pose.morph_weights.len() as u32,
            ),
            None => (&[][..], &[][..], 0, 0),
        };
        // Shadow validation needs the exact small pose payload that produced
        // this asynchronous GPU classification. Capture it only on request;
        // the normal renderer path keeps its existing zero-clone behavior.
        let captured_pose = capture_pose.then(|| PendingLodPose {
            joint_matrices: joint_matrices.to_vec(),
            morph_weights: morph_weights.to_vec(),
        });

        perf_mark("lod-anim-eval-end");
        perf_measure("lod-anim-eval", "lod-anim-eval-start", "lod-anim-eval-end");

        // 2-5. GPU compute, staging copies, and fence insertion.
        perf_mark("lod-gpu-compute-start");
        let pending_job = GPU_COMPUTE.with(|gc| {
            let mut gc = gc.borrow_mut();
            let (gl, compute) = match gc.as_mut() {
                Some(pair) => pair,
                None => return None,
            };

            perf_mark("lod-gpu-pose-upload-start");
            if !joint_matrices.is_empty() {
                compute.upload_joint_matrices(gl, joint_matrices);
            }
            if !morph_weights.is_empty() {
                compute.upload_morph_weights(gl, morph_weights);
            }
            perf_mark("lod-gpu-pose-upload-end");
            perf_measure(
                "lod-gpu-pose-upload",
                "lod-gpu-pose-upload-start",
                "lod-gpu-pose-upload-end",
            );

            let mut vp = [0.0f32; 16];
            for (i, &v) in vp_matrix.iter().take(16).enumerate() { vp[i] = v; }

            let max_lod = LOD_MAX.with(|m| *m.borrow());
            perf_mark("lod-gpu-dispatch-start");
            let dispatched = compute.compute_lods(
                gl, num_faces, num_vertices,
                num_joints, num_morph,
                &dispatch_state.subjects,
                dispatch_state.baseline_mobius,
                dispatch_state.baseline_model,
                dispatch_state.pole,
                dispatch_state.mobius_power,
                dispatch_state.c_norm_sq,
                dispatch_state.has_pole,
                density, mesh_radius, min_px, max_lod,
                &vp, vp_width, vp_height,
            );
            perf_mark("lod-gpu-dispatch-end");
            perf_measure("lod-gpu-dispatch", "lod-gpu-dispatch-start", "lod-gpu-dispatch-end");
            let n = match dispatched {
                Ok(n) if n > 0 => n,
                Ok(_) => {
                    web_sys::console::warn_1(&"LOD dispatch returned no faces".into());
                    return None;
                }
                Err(error) => {
                    web_sys::console::warn_1(&format!("LOD dispatch failed: {error}").into());
                    return None;
                }
            };
            perf_mark("lod-gpu-stage-start");
            let readback = match compute.stage_readback(gl, n) {
                Ok(readback) => readback,
                Err(error) => {
                    web_sys::console::warn_1(&format!("LOD staging failed: {error}").into());
                    return None;
                }
            };
            perf_mark("lod-gpu-stage-end");
            perf_measure("lod-gpu-stage", "lod-gpu-stage-start", "lod-gpu-stage-end");

            let fence = match unsafe { gl.fence_sync(glow::SYNC_GPU_COMMANDS_COMPLETE, 0) } {
                Ok(fence) => fence,
                Err(error) => {
                    web_sys::console::warn_1(&format!("LOD fence creation failed: {error}").into());
                    compute.discard_staged_readback(gl, readback);
                    return None;
                }
            };
            perf_mark("lod-gpu-wait-start");
            unsafe { gl.flush(); }
            Some(PendingAnimatedLods {
                fence,
                readback,
                classified_faces: num_faces,
                resident_faces,
                subject_records: dispatch_state.subjects.len(),
                pose_stamp,
                pose: captured_pose,
            })
        });
        perf_mark("lod-gpu-compute-end");
        perf_measure("lod-gpu-compute", "lod-gpu-compute-start", "lod-gpu-compute-end");

        let Some(pending_job) = pending_job else {
            web_sys::console::warn_1(&format!(
                "LOD dispatch returned empty: faces={} verts={}",
                num_faces, num_vertices
            ).into());
            return false;
        };

        PENDING_ANIMATED_LODS.with(|pending| *pending.borrow_mut() = Some(pending_job));
        true
    })
}

/// Poll the current LOD job without blocking.
///
/// Returns `undefined` while the fence is unsignaled, `null` on failure or
/// when no job exists, and `{ lods, indices?, changed_faces, full_snapshot }`
/// when ready. `lods` contains all records only for a full snapshot; otherwise
/// it contains one six-float record per entry in `indices`.
#[wasm_bindgen]
pub fn poll_animated_lods() -> JsValue {
    let fence = match PENDING_ANIMATED_LODS.with(|pending| {
        pending.borrow().as_ref().map(|job| job.fence)
    }) {
        Some(fence) => fence,
        None => return JsValue::NULL,
    };

    GPU_COMPUTE.with(|gc| {
        let mut gc = gc.borrow_mut();
        let (gl, compute) = match gc.as_mut() {
            Some(pair) => pair,
            None => return JsValue::NULL,
        };
        let status = unsafe { gl.client_wait_sync(fence, 0, 0) };
        if status == glow::TIMEOUT_EXPIRED
            || (status != glow::ALREADY_SIGNALED
                && status != glow::CONDITION_SATISFIED
                && status != glow::WAIT_FAILED)
        {
            return JsValue::UNDEFINED;
        }

        let Some(job) = PENDING_ANIMATED_LODS.with(|pending| pending.borrow_mut().take()) else {
            return JsValue::NULL;
        };
        let PendingAnimatedLods {
            fence,
            readback,
            classified_faces,
            resident_faces,
            subject_records,
            pose_stamp,
            pose,
        } = job;
        unsafe { gl.delete_sync(fence); }
        if status == glow::WAIT_FAILED {
            compute.discard_staged_readback(gl, readback);
            web_sys::console::warn_1(&"LOD fence wait failed".into());
            return JsValue::NULL;
        }

        perf_mark("lod-gpu-wait-end");
        perf_measure("lod-gpu-wait", "lod-gpu-wait-start", "lod-gpu-wait-end");
        perf_mark("lod-gpu-readback-start");
        let gpu_readback_bytes = readback.byte_len();
        let packed_class = match compute.finish_staged_readback(gl, readback) {
            Ok(classification) => classification,
            Err(error) => {
                web_sys::console::warn_1(&format!("LOD packed readback failed: {error}").into());
                return JsValue::NULL;
            }
        };
        let gpu_class = match compute.decode_readback_vector(&packed_class) {
            Ok(classification) => classification,
            Err(error) => {
                compute.recycle_readback_vector(packed_class);
                web_sys::console::warn_1(&format!("LOD packed decode failed: {error}").into());
                return JsValue::NULL;
            }
        };
        let full_fingerprint = exact_f32_slice_fingerprint(&gpu_class);
        let full_fingerprint = format!(
            "{}:{:016x}",
            full_fingerprint.0,
            full_fingerprint.1,
        );
        perf_mark("lod-gpu-readback-end");
        perf_measure("lod-gpu-readback", "lod-gpu-readback-start", "lod-gpu-readback-end");
        let result = ANIMATED_LOD_DELTA.with(|delta| {
            let mut delta = delta.borrow_mut();
            let encoded = match delta.encode(&gpu_class) {
                Ok(encoded) => encoded,
                Err(error) => {
                    let result = js_sys::Object::new();
                    js_sys::Reflect::set(
                        &result,
                        &"error".into(),
                        &JsValue::from_str(&error.to_string()),
                    ).ok();
                    return result;
                }
            };
            let lods = js_sys::Float32Array::new_with_length(encoded.lods.len() as u32);
            lods.copy_from(encoded.lods);
            let indices = if encoded.full_snapshot {
                None
            } else {
                let indices = js_sys::Uint32Array::new_with_length(
                    encoded.changed_faces.len() as u32,
                );
                indices.copy_from(encoded.changed_faces);
                Some(indices)
            };
            let result = js_sys::Object::new();
            js_sys::Reflect::set(&result, &"lods".into(), &lods).ok();
            if let Some(indices) = indices {
                js_sys::Reflect::set(&result, &"indices".into(), &indices).ok();
            }
            js_sys::Reflect::set(
                &result,
                &"changed_faces".into(),
                &JsValue::from_f64(if encoded.full_snapshot {
                    gpu_class.len() as f64 / batch::FACE_LOD_STRIDE as f64
                } else {
                    encoded.changed_faces.len() as f64
                }),
            ).ok();
            js_sys::Reflect::set(
                &result,
                &"full_snapshot".into(),
                &JsValue::from_bool(encoded.full_snapshot),
            ).ok();
            js_sys::Reflect::set(
                &result,
                &"delta_epoch".into(),
                &JsValue::from_f64(encoded.sequence.epoch as f64),
            ).ok();
            js_sys::Reflect::set(
                &result,
                &"delta_base_revision".into(),
                &JsValue::from_f64(encoded.sequence.base_revision as f64),
            ).ok();
            js_sys::Reflect::set(
                &result,
                &"delta_revision".into(),
                &JsValue::from_f64(encoded.sequence.revision as f64),
            ).ok();
            js_sys::Reflect::set(
                &result,
                &"classified_faces".into(),
                &JsValue::from_f64(classified_faces as f64),
            ).ok();
            js_sys::Reflect::set(
                &result,
                &"resident_faces".into(),
                &JsValue::from_f64(resident_faces as f64),
            ).ok();
            js_sys::Reflect::set(
                &result,
                &"subject_records".into(),
                &JsValue::from_f64(subject_records as f64),
            ).ok();
            js_sys::Reflect::set(
                &result,
                &"gpu_passes".into(),
                &JsValue::from_f64(1.0),
            ).ok();
            js_sys::Reflect::set(
                &result,
                &"gpu_readback_bytes".into(),
                &JsValue::from_f64(gpu_readback_bytes as f64),
            ).ok();
            js_sys::Reflect::set(
                &result,
                &"full_fingerprint".into(),
                &JsValue::from_str(&full_fingerprint),
            ).ok();
            if let Some(stamp) = pose_stamp {
                js_sys::Reflect::set(
                    &result,
                    &"pose_time".into(),
                    &JsValue::from_f64(stamp.clip_time),
                )
                .ok();
                js_sys::Reflect::set(
                    &result,
                    &"pose_sample_time".into(),
                    &JsValue::from_f64(stamp.sample_time),
                ).ok();
                js_sys::Reflect::set(
                    &result,
                    &"pose_revision".into(),
                    &JsValue::from_f64(stamp.revision as f64),
                ).ok();
                js_sys::Reflect::set(
                    &result,
                    &"pose_continuity_epoch".into(),
                    &JsValue::from_f64(stamp.continuity_epoch as f64),
                ).ok();
            }
            if let Some(pose) = pose {
                let matrices = js_sys::Float32Array::new_with_length(
                    pose.joint_matrices.len() as u32,
                );
                matrices.copy_from(&pose.joint_matrices);
                js_sys::Reflect::set(&result, &"pose_matrices".into(), &matrices).ok();
                let morph_weights =
                    js_sys::Float32Array::new_with_length(pose.morph_weights.len() as u32);
                morph_weights.copy_from(&pose.morph_weights);
                js_sys::Reflect::set(
                    &result,
                    &"pose_morph_weights".into(),
                    &morph_weights,
                )
                .ok();
            }
            result
        });
        compute.recycle_decoded_vector(gpu_class);
        compute.recycle_readback_vector(packed_class);
        perf_mark("lod-wasm-end");
        perf_measure("lod-wasm-total", "lod-wasm-start", "lod-wasm-end");
        result.into()
    })
}

/// Reset the retained classification snapshot at a model/topology boundary.
#[wasm_bindgen]
pub fn reset_animated_lod_delta() {
    ANIMATED_LOD_DELTA.with(|delta| delta.borrow_mut().reset());
}

/// Cancel and release a pending asynchronous LOD job, if any.
#[wasm_bindgen]
pub fn cancel_animated_lods() -> bool {
    let Some(job) = PENDING_ANIMATED_LODS.with(|pending| pending.borrow_mut().take()) else {
        return false;
    };
    GPU_COMPUTE.with(|gc| {
        if let Some((gl, compute)) = gc.borrow_mut().as_mut() {
            release_pending_lod_job(gl, compute, job);
        }
    });
    true
}

/// Set tessellation parameters.
/// density: target mesh_radius/density = triangle size in deformed world units (default 100)
/// screen_atten: whether to attenuate LOD for distant/small screen faces
#[wasm_bindgen]
pub fn set_tess_params(density: f64, screen_atten: bool) {
    quilting_core::evaluate::set_tess_params(density, screen_atten);
}

/// Set minimum pixels per subdivision for screen attenuation.
#[wasm_bindgen]
pub fn set_min_px_per_sub(px: f64) {
    quilting_core::evaluate::set_min_px_per_sub(px);
}

/// Return canonical atlas triples reachable after the selected validated
/// within-face LOD grading. Unsupported ratios return an empty array.
#[wasm_bindgen]
pub fn required_tessellation_atlas_triples(
    max_lod_exp: u32,
    max_face_edge_ratio: u32,
) -> Vec<u32> {
    let Some(grading) = batch::FaceLodGrading::from_ratio(max_face_edge_ratio) else {
        return Vec::new();
    };
    let lods: Vec<u32> = (0..=max_lod_exp.min(30))
        .map(|exponent| 1u32 << exponent)
        .collect();
    quilting_core::atlas::ratio_bounded_canonical_triples(
        &lods,
        grading.ratio(),
    )
    .into_iter()
    .flatten()
    .collect()
}

/// Export the runtime atlas as three packed canonical GPU buffers plus metadata.
///
/// Each patch contributes seven u32 metadata values:
/// `[lod_a, lod_b, lod_c, tri_start, tri_count, line_start, line_count]`.
/// Indices are global into the packed barycentric buffer, so every LOD and all
/// six S3 permutations share the same three WebGL buffers.
#[wasm_bindgen]
pub fn export_all_patches() -> JsValue {
    ATLAS.with(|atlas_cell| {
        let atlas = atlas_cell.borrow();
        let atlas = match atlas.as_ref() {
            Some(a) => a,
            None => return JsValue::NULL,
        };

        let mut keys: Vec<[u32; 3]> = atlas.patches.keys().copied().collect();
        keys.sort_unstable();

        let mut bary = Vec::<f32>::new();
        let mut tris = Vec::<u32>::new();
        let mut lines = Vec::<u32>::new();
        let mut patches = Vec::<u32>::with_capacity(keys.len() * 7);

        for key in keys {
            let Some(entry) = atlas.patches.get(&key) else { continue };
            let base_vertex = (bary.len() / 3) as u32;

            for position in &atlas.positions
                [entry.base_vertex..entry.base_vertex + entry.vertex_count]
            {
                let mut b = triangle::cartesian_to_bary(position[0], position[1]);
                for component in &mut b {
                    if component.abs() < 1e-10 { *component = 0.0; }
                }
                let sum = b[0] + b[1] + b[2];
                if sum > 0.0 {
                    b[0] /= sum;
                    b[1] /= sum;
                    b[2] /= sum;
                }
                bary.extend_from_slice(&[b[0] as f32, b[1] as f32, b[2] as f32]);
            }

            let tri_start = tris.len() as u32;
            let line_start = lines.len() as u32;
            for triangle in &atlas.triangles
                [entry.base_triangle..entry.base_triangle + entry.triangle_count]
            {
                let local = [
                    triangle[0] - entry.base_vertex,
                    triangle[1] - entry.base_vertex,
                    triangle[2] - entry.base_vertex,
                ];
                let index = [
                    base_vertex + local[0] as u32,
                    base_vertex + local[1] as u32,
                    base_vertex + local[2] as u32,
                ];
                tris.extend_from_slice(&index);
                lines.extend_from_slice(&[
                    index[0], index[1], index[1], index[2], index[2], index[0],
                ]);
            }

            patches.extend_from_slice(&[
                key[0], key[1], key[2],
                tri_start, tris.len() as u32 - tri_start,
                line_start, lines.len() as u32 - line_start,
            ]);
        }

        let result = js_sys::Object::new();
        let set = |name: &str, value: JsValue| {
            js_sys::Reflect::set(&result, &name.into(), &value).ok();
        };
        set("patches", js_sys::Uint32Array::from(&patches[..]).into());
        set("bary", js_sys::Float32Array::from(&bary[..]).into());
        set("tris", js_sys::Uint32Array::from(&tris[..]).into());
        set("lines", js_sys::Uint32Array::from(&lines[..]).into());
        result.into()
    })
}

/// Build a subset of the atlas (for parallel construction).
#[wasm_bindgen]
pub fn build_atlas_subset(max_lod_exp: u32, mode: &str, worker_index: u32, num_workers: u32) -> f64 {
    let config = PatchConfig { k_candidates: 30, seed: 42 };
    let lods: Vec<u32> = (0..=max_lod_exp).map(|n| 1u32 << n).collect();
    let build_mode = match mode {
        "hierarchical" => BuildMode::Hierarchical,
        _ => BuildMode::Direct,
    };
    let start = js_sys::Date::now();
    let atlas = TessellationAtlas::build_subset(&lods, &config, build_mode, worker_index as usize, num_workers as usize);
    let elapsed = js_sys::Date::now() - start;
    ATLAS.with(|a| *a.borrow_mut() = Some(atlas));
    SENT_TESS.with(|s| s.borrow_mut().clear());
    elapsed
}

/// Merge another atlas (from bytes) into the current one.
#[wasm_bindgen]
pub fn merge_atlas_bytes(bytes: &[u8]) -> bool {
    match TessellationAtlas::from_bytes(bytes) {
        Ok(other) => {
            ATLAS.with(|a| {
                if let Some(atlas) = a.borrow_mut().as_mut() {
                    atlas.merge_from(&other);
                }
            });
            SENT_TESS.with(|s| s.borrow_mut().clear());
            true
        }
        Err(_) => false,
    }
}

/// Export the atlas as bytes for sharing with other workers.
#[wasm_bindgen]
pub fn export_atlas_bytes() -> Vec<u8> {
    ATLAS.with(|a| {
        a.borrow().as_ref().map(|atlas| atlas.to_bytes()).unwrap_or_default()
    })
}

/// Import an atlas from bytes (built by another worker).
#[wasm_bindgen]
pub fn import_atlas_bytes(bytes: &[u8]) -> bool {
    match TessellationAtlas::from_bytes(bytes) {
        Ok(atlas) => {
            ATLAS.with(|a| *a.borrow_mut() = Some(atlas));
            SENT_TESS.with(|s| s.borrow_mut().clear());
            true
        }
        Err(_) => false,
    }
}

/// Extend the atlas to include a new LOD level by subdividing from existing parents.
/// Much faster than rebuilding — each new patch is one subdivision step.
#[wasm_bindgen]
pub fn extend_atlas(new_lod: u32) -> f64 {
    let start = js_sys::Date::now();
    ATLAS.with(|atlas_cell| {
        let mut atlas_opt = atlas_cell.borrow_mut();
        if let Some(atlas) = atlas_opt.as_mut() {
            // Collect all LOD levels currently in the atlas + the new one
            let mut levels: Vec<u32> = atlas.lod_levels.clone();
            if !levels.contains(&new_lod) {
                levels.push(new_lod);
                levels.sort();
            }

            // Generate all canonical triples that include new_lod
            let mut new_triples = Vec::new();
            for &a in &levels {
                for &b in &levels {
                    for &c in &levels {
                        if a > b || b > c { continue; }
                        if a != new_lod && b != new_lod && c != new_lod { continue; }
                        let key = [a, b, c];
                        if !atlas.patches.contains_key(&key) {
                            new_triples.push(key);
                        }
                    }
                }
            }

            let config = PatchConfig { k_candidates: 30, seed: 42 };
            for key in &new_triples {
                atlas.ensure_hierarchical_patch(*key, &config);
            }

            atlas.lod_levels = levels;
        }
    });
    SENT_TESS.with(|s| s.borrow_mut().clear());
    js_sys::Date::now() - start
}

/// Generate a single tessellation patch via hierarchical subdivision and store
/// it in the atlas. Recursively generates ancestor patches if needed (e.g.,
/// [4,256,512] needs parent [2,128,256] which needs grandparent [1,64,128]).
/// Falls back to direct Poisson generation for base triples that can't be halved.
#[wasm_bindgen]
pub fn generate_and_store_patch(res_a: u32, res_b: u32, res_c: u32) {
    ATLAS.with(|atlas_cell| {
        let mut atlas_opt = atlas_cell.borrow_mut();
        // Initialize empty atlas if none exists
        let atlas = atlas_opt.get_or_insert_with(|| TessellationAtlas {
            positions: Vec::new(),
            triangles: Vec::new(),
            patches: HashMap::default(),
            lod_levels: Vec::new(),
        });
        let mut key = [res_a, res_b, res_c];
        key.sort();
        atlas.ensure_hierarchical_patch(key, &PatchConfig {
            k_candidates: 30,
            seed: 42,
        });
    });
}


// --- Shader compilation via quilting-shaders ---

/// Compile the vertex shader for native OpenGL/WebGL rendering
/// (no Y-flip or Z-remap -- suitable for direct use with WebGL2).
#[wasm_bindgen]
pub fn get_vertex_glsl() -> String {
    match quilting_shaders::compile_vertex_glsl_native() {
        Ok(glsl) => glsl,
        Err(e) => format!("// ERROR: {}", e),
    }
}

/// Compile a fragment shader for native OpenGL/WebGL rendering.
/// mode: "matcap", "wire", or "normals"
#[wasm_bindgen]
pub fn get_fragment_glsl(mode: &str) -> String {
    match quilting_shaders::compile_fragment_glsl_native(mode) {
        Ok(glsl) => glsl,
        Err(e) => format!("// ERROR: {}", e),
    }
}

thread_local! {
    /// Per-face material indices for the current loaded model.
    /// Index i -> material index for face i (None = default material).
    static FACE_MATERIALS: RefCell<Vec<Option<usize>>> = RefCell::new(Vec::new());
    /// Stored glTF data for animation switching without re-parsing.
    static GLTF_DATA: RefCell<Option<StoredGltfData>> = RefCell::new(None);
}

/// Set per-face material indices (for KHR_materials_variants switching).
#[wasm_bindgen]
pub fn set_face_materials(materials: &[i32]) {
    FACE_MATERIALS.with(|fm| {
        *fm.borrow_mut() = materials.iter().map(|&m| {
            if m >= 0 { Some(m as usize) } else { None }
        }).collect();
    });
}

/// List available animations for the current glTF model.
/// Returns a JS array of { index, name, duration, t_min, t_max }.
#[wasm_bindgen]
pub fn list_animations() -> JsValue {
    GLTF_DATA.with(|gd| {
        let data = gd.borrow();
        let data = match data.as_ref() {
            Some(d) => d,
            None => return JsValue::NULL,
        };
        let info = quilting_gltf::evaluator::list_animations(&data.animations);
        let arr = js_sys::Array::new();
        for i in &info {
            let obj = js_sys::Object::new();
            js_sys::Reflect::set(&obj, &"index".into(), &JsValue::from_f64(i.index as f64)).unwrap();
            let fallback = format!("Animation {}", i.index);
            let name = i.name.as_deref().unwrap_or(&fallback);
            js_sys::Reflect::set(&obj, &"name".into(), &JsValue::from_str(name)).unwrap();
            js_sys::Reflect::set(&obj, &"duration".into(), &JsValue::from_f64(i.duration)).unwrap();
            js_sys::Reflect::set(&obj, &"t_min".into(), &JsValue::from_f64(i.t_min)).unwrap();
            js_sys::Reflect::set(&obj, &"t_max".into(), &JsValue::from_f64(i.t_max)).unwrap();
            arr.push(&obj);
        }
        arr.into()
    })
}

/// Switch to a different animation by index and rebuild the retained evaluator.
/// Returns a JS object with { time_min, time_max, num_vertices, num_faces }
/// or null on failure.
#[wasm_bindgen]
pub fn set_active_animation(index: u32) -> JsValue {
    let index = index as usize;

    GLTF_DATA.with(|gd| {
        let mut data = gd.borrow_mut();
        let data = match data.as_mut() {
            Some(d) => d,
            None => return JsValue::NULL,
        };

        if index >= data.animations.len() {
            web_sys::console::warn_1(&format!(
                "set_active_animation: index {} out of range ({})", index, data.animations.len()
            ).into());
            return JsValue::NULL;
        }

        data.active_animation = index;
        *data.cached_pose.borrow_mut() = None;

        // Skinning is optional: a morph-only model still needs the evaluator
        // rebuilt or selecting clip N continues to play clip 0's channels.
        let skin = data.primary_skin_idx
            .and_then(|skin_index| data.skins.get(skin_index).cloned());
        let num_morph = data.combined.morph_targets.len();
        data.evaluator = Some(quilting_gltf::evaluator::AnimationEvaluator::new(
            data.animations[index].clone(),
            skin,
            data.nodes.clone(),
            num_morph,
        ));

        // Get animation time range from evaluator
        let anim_info = quilting_gltf::evaluator::list_animations(&data.animations);
        let (time_min, time_max) = anim_info.get(index)
            .map(|a| (a.t_min, a.t_max))
            .unwrap_or((0.0, 0.0));
        let n_verts = data.combined.positions.len();
        let n_faces = data.combined.triangles.len();

        FACE_MATERIALS.with(|fm| *fm.borrow_mut() = data.face_material_indices.clone());
        SENT_TESS.with(|s| s.borrow_mut().clear());

        web_sys::console::log_1(&format!(
            "set_active_animation: switched to animation {} — {} verts, {} faces, time [{:.3}, {:.3}]",
            index, n_verts, n_faces, time_min, time_max
        ).into());

        let result = js_sys::Object::new();
        js_sys::Reflect::set(&result, &"time_min".into(), &JsValue::from_f64(time_min)).unwrap();
        js_sys::Reflect::set(&result, &"time_max".into(), &JsValue::from_f64(time_max)).unwrap();
        js_sys::Reflect::set(&result, &"num_vertices".into(), &JsValue::from_f64(n_verts as f64)).unwrap();
        js_sys::Reflect::set(&result, &"num_faces".into(), &JsValue::from_f64(n_faces as f64)).unwrap();
        result.into()
    })
}

/// Evaluate skeletal animation at time t and return joint matrices as a flat f32 array.
/// Returns null if no evaluator is available.
/// The matrices are skin matrices (joint_world × inverse_bind), column-major, ready for UBO upload.
#[wasm_bindgen]
pub fn evaluate_animation_frame(t: f64) -> JsValue {
    GLTF_DATA.with(|gd| {
        let data = gd.borrow();
        let data = match data.as_ref() {
            Some(d) => d,
            None => return JsValue::NULL,
        };
        let pose = match normalized_animation_pose(data, t) {
            Some(pose) => pose,
            None => return JsValue::NULL,
        };

        let result = js_sys::Object::new();

        // Joint matrices (skeletal skinning)
        if !pose.joint_matrices.is_empty() {
            let arr = js_sys::Float32Array::new_with_length(pose.joint_matrices.len() as u32);
            arr.copy_from(&pose.joint_matrices);
            js_sys::Reflect::set(&result, &"matrices".into(), &arr).unwrap();
        }

        // Morph weights (morph target animation)
        if !pose.morph_weights.is_empty() {
            let arr = js_sys::Float32Array::new_with_length(pose.morph_weights.len() as u32);
            arr.copy_from(&pose.morph_weights);
            js_sys::Reflect::set(&result, &"morph_weights".into(), &arr).unwrap();
        }

        result.into()
    })
}

/// Get per-vertex animation data (skinning + morph targets) for the current model.
/// Returns a JS object with:
///   { joint_indices?, joint_weights?, num_vertices, num_joints, num_morph_targets,
///     morph_deltas?, t_min, t_max, duration }
/// Returns null if no animated model is loaded.
#[wasm_bindgen]
pub fn get_skinning_data() -> JsValue {
    GLTF_DATA.with(|gd| {
        let data = gd.borrow();
        let data = match data.as_ref() {
            Some(d) if d.evaluator.is_some() => d,
            _ => return JsValue::NULL,
        };

        let nv = data.combined.positions.len();
        let has_skin = data.combined.joint_indices.is_some();
        let num_morph = data.combined.morph_targets.len();

        let result = js_sys::Object::new();
        js_sys::Reflect::set(&result, &"num_vertices".into(), &JsValue::from_f64(nv as f64)).unwrap();
        js_sys::Reflect::set(&result, &"num_morph_targets".into(), &JsValue::from_f64(num_morph as f64)).unwrap();

        // Joint data (optional — only for skeletal skinning)
        if has_skin {
            let ji = data.combined.joint_indices.as_ref().unwrap();
            let jw = data.combined.joint_weights.as_ref().unwrap();
            let mut indices_f32 = Vec::with_capacity(nv * 4);
            for idx in ji {
                indices_f32.extend_from_slice(&[idx[0] as f32, idx[1] as f32, idx[2] as f32, idx[3] as f32]);
            }
            let mut weights_f32 = Vec::with_capacity(nv * 4);
            for w in jw {
                weights_f32.extend_from_slice(&[w[0], w[1], w[2], w[3]]);
            }
            let ji_arr = js_sys::Float32Array::new_with_length(indices_f32.len() as u32);
            ji_arr.copy_from(&indices_f32);
            let jw_arr = js_sys::Float32Array::new_with_length(weights_f32.len() as u32);
            jw_arr.copy_from(&weights_f32);
            js_sys::Reflect::set(&result, &"joint_indices".into(), &ji_arr).unwrap();
            js_sys::Reflect::set(&result, &"joint_weights".into(), &jw_arr).unwrap();
        }

        // Morph target deltas (optional — only for morph target animation)
        // Deltas are scaled by norm_scale to match normalized position space.
        if num_morph > 0 {
            let ns = data.norm_scale as f32;
            let mut deltas = Vec::with_capacity(num_morph * nv * 3);
            for target in &data.combined.morph_targets {
                for vi in 0..nv {
                    if vi < target.len() {
                        deltas.push(target[vi][0] as f32 * ns);
                        deltas.push(target[vi][1] as f32 * ns);
                        deltas.push(target[vi][2] as f32 * ns);
                    } else {
                        deltas.extend_from_slice(&[0.0, 0.0, 0.0]);
                    }
                }
            }
            let delta_arr = js_sys::Float32Array::new_with_length(deltas.len() as u32);
            delta_arr.copy_from(&deltas);
            js_sys::Reflect::set(&result, &"morph_deltas".into(), &delta_arr).unwrap();
        }

        // Evaluator metadata
        if let Some(ref eval) = data.evaluator {
            let (t_min, t_max) = eval.time_range();
            js_sys::Reflect::set(&result, &"num_joints".into(), &JsValue::from_f64(eval.num_joints() as f64)).unwrap();
            js_sys::Reflect::set(&result, &"t_min".into(), &JsValue::from_f64(t_min)).unwrap();
            js_sys::Reflect::set(&result, &"t_max".into(), &JsValue::from_f64(t_max)).unwrap();
            js_sys::Reflect::set(&result, &"duration".into(), &JsValue::from_f64(eval.duration())).unwrap();
        }
        result.into()
    })
}

/// Build rest-pose instance data for GPU skinning (static, uploaded once).
/// LODs are computed from skinned positions at time `lod_time` for accuracy.
/// Returns a Float32Array with 52 floats per face (canonical patch stride).
/// Vertex indices are packed in p0.x/p1.x/p2.x for skinning texture lookup.
/// Returns null if no skinned model is loaded.
#[wasm_bindgen]
pub fn get_rest_pose_instances() -> JsValue {
    GLTF_DATA.with(|gd| {
        let data = gd.borrow();
        let data = match data.as_ref() {
            Some(d) => d,
            _ => return JsValue::NULL,
        };

        let combined = &data.combined;
        let nf = combined.triangles.len();
        let n_verts = combined.positions.len();

        let mut instances = vec![0.0f32; nf * instance_layout::STRIDE];

        // Normalize positions
        let center = data.norm_center;
        let norm_scale = data.norm_scale;

        let norm_positions: Vec<[f64; 3]> = combined.positions.iter().map(|v| {
            [(v[0]-center[0])*norm_scale, (v[1]-center[1])*norm_scale, (v[2]-center[2])*norm_scale]
        }).collect();

        // Use the REAL LOD computation from evaluate.rs — half-edge coherent,
        // Möbius-deformed medians, screen attenuation, proper snap to power of 2.
        // Identity Möbius for initial load; async recompute handles Möbius-dependent LOD.
        let vertex_uvs_f32: Option<Vec<[f32; 2]>> = combined.uvs.as_ref().map(|uvs| {
            uvs.iter().map(|uv| [uv[0] as f32, uv[1] as f32]).collect()
        });
        let vertex_normals_f32: Vec<[f32; 3]> = if let Some(normals) = &combined.normals {
            normals.iter()
                .map(|normal| [normal[0] as f32, normal[1] as f32, normal[2] as f32])
                .collect()
        } else {
            // Only synthesize smooth normals when the asset omitted them. The
            // former unconditional pass needlessly revisited every triangle
            // in normal-bearing meshes such as the 95k-face chess scene.
            let mut vertex_normals = vec![[0.0f64; 3]; combined.positions.len()];
            for face in &combined.triangles {
                let v0 = combined.positions[face[0]];
                let v1 = combined.positions[face[1]];
                let v2 = combined.positions[face[2]];
                let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
                let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
                let face_normal = [
                    e1[1] * e2[2] - e1[2] * e2[1],
                    e1[2] * e2[0] - e1[0] * e2[2],
                    e1[0] * e2[1] - e1[1] * e2[0],
                ];
                for &vertex in face {
                    vertex_normals[vertex][0] += face_normal[0];
                    vertex_normals[vertex][1] += face_normal[1];
                    vertex_normals[vertex][2] += face_normal[2];
                }
            }
            vertex_normals.iter_mut().for_each(|normal| {
                let length = (normal[0] * normal[0]
                    + normal[1] * normal[1]
                    + normal[2] * normal[2])
                    .sqrt();
                if length > 1e-10 {
                    normal[0] /= length;
                    normal[1] /= length;
                    normal[2] /= length;
                }
            });
            vertex_normals.into_iter()
                .map(|normal| [normal[0] as f32, normal[1] as f32, normal[2] as f32])
                .collect()
        };

        let lod_instances = quilting_core::evaluate::compute_instances_with_uvs(
            &norm_positions,
            &combined.triangles,
            &Mobius::identity(),
            None, // no screen info for initial load
            None, // will build half-edge mesh internally
            vertex_uvs_f32.as_deref(),
            Some(&vertex_normals_f32),
        );

        // Pack into the compact instance format for the GPU animation path
        for (fi, face) in combined.triangles.iter().enumerate() {
            let inst = &lod_instances[fi];
            let mut w = InstanceWriter::new(&mut instances, fi);

            for (vi, &vert_idx) in face.iter().enumerate() {
                let p = norm_positions[vert_idx];
                w.set_position(vi, vert_idx as u32, [p[0] as f32, p[1] as f32, p[2] as f32]);
            }
            // Edge LODs from compute_instances (half-edge coherent)
            w.set_edge_lods([
                inst.edge_lods[0] as f32,
                inst.edge_lods[1] as f32,
                inst.edge_lods[2] as f32,
            ]);
            w.set_vertex_lods([
                inst.vertex_lods[0] as f32,
                inst.vertex_lods[1] as f32,
                inst.vertex_lods[2] as f32,
            ]);
            w.set_uvs(inst.uvs);
            for vi in 0..3 {
                w.set_normal(vi, inst.normals[vi]);
            }
        }

        let arr = js_sys::Float32Array::new_with_length(instances.len() as u32);
        arr.copy_from(&instances);

        // Build per-face LOD classification for batch grouping
        // Each face: [canonical_a, canonical_b, canonical_c, perm_index, parity, atlas_index]
        let atlas_keys: Vec<[u32; 3]> = LOD_ATLAS_KEYS.with(|ak| ak.borrow().clone());
        let mut face_lods = Vec::with_capacity(nf * 6);
        for fi in 0..nf {
            let e = fi * instance_layout::STRIDE + instance_layout::offset::EDGE_LODS;
            let lods = [instances[e] as u32, instances[e + 1] as u32, instances[e + 2] as u32];
            let ck = quilting_core::permutation::canonical_form(lods);
            let canonical = ck.res;
            let perm_idx = ck.perm_index;
            let parity = quilting_core::permutation::perm_sign(perm_idx);
            let atlas_idx = atlas_keys.binary_search(&canonical).unwrap_or(0);
            face_lods.push(canonical[0] as f32);
            face_lods.push(canonical[1] as f32);
            face_lods.push(canonical[2] as f32);
            face_lods.push(perm_idx as f32);
            face_lods.push(parity as f32);
            face_lods.push(atlas_idx as f32);
        }
        let lod_arr = js_sys::Float32Array::new_with_length(face_lods.len() as u32);
        lod_arr.copy_from(&face_lods);

        // Compute bounding box for camera framing
        let mut bb_min = [f64::INFINITY; 3];
        let mut bb_max = [f64::NEG_INFINITY; 3];
        for pos in &combined.positions {
            for i in 0..3 {
                bb_min[i] = bb_min[i].min(pos[i]);
                bb_max[i] = bb_max[i].max(pos[i]);
            }
        }
        let extent = (bb_max[0]-bb_min[0]).max(bb_max[1]-bb_min[1]).max(bb_max[2]-bb_min[2]);

        let result = js_sys::Object::new();
        js_sys::Reflect::set(&result, &"instances".into(), &arr).unwrap();
        js_sys::Reflect::set(&result, &"num_faces".into(), &JsValue::from_f64(nf as f64)).unwrap();
        js_sys::Reflect::set(&result, &"num_vertices".into(), &JsValue::from_f64(n_verts as f64)).unwrap();
        js_sys::Reflect::set(&result, &"stride".into(), &JsValue::from_f64(instance_layout::STRIDE as f64)).unwrap();
        js_sys::Reflect::set(&result, &"extent".into(), &JsValue::from_f64(extent)).unwrap();
        js_sys::Reflect::set(&result, &"face_lods".into(), &lod_arr).unwrap();
        result.into()
    })
}

/// Load a glTF/GLB file from raw bytes, bake animation into a HyperMesh,
/// and store it for slicing.  Accepts both GLB (binary) and glTF with
/// embedded base64 buffers — `gltf::import_slice` handles both.
///
/// Walks ALL scene nodes, applies world transforms, merges all meshes.
/// Returns a JS object with:
///   { asset_metadata, time_min, time_max, num_vertices, num_faces, materials, textures,
///     face_node_indices, node_stable_entity_ids, node_world_transforms,
///     base_color, metallic, roughness }
/// Built with js_sys to avoid serde overhead on large data.
#[wasm_bindgen]
pub fn load_gltf_data(data: &[u8]) -> JsValue {
    let t0 = js_sys::Date::now();

    let scene = match quilting_gltf::load_gltf_raw(data) {
        Ok(s) => s,
        Err(e) => {
            web_sys::console::warn_1(&format!("load_gltf_data: could not load: {e}").into());
            return JsValue::NULL;
        }
    };

    let t_parse = js_sys::Date::now();
    web_sys::console::log_1(&format!(
        "load_gltf_data: parsed in {:.0}ms — {} meshes, {} materials, {} animations, {} skins, {} nodes, {} raw textures",
        t_parse - t0, scene.meshes.len(), scene.materials.len(),
        scene.animations.len(), scene.skins.len(), scene.nodes.len(), scene.raw_textures.len()
    ).into());

    // Determine the active scene
    let active_scene = if let Some(si) = scene.default_scene {
        scene.scenes.get(si)
    } else {
        scene.scenes.first()
    };

    let node_world_transforms = active_scene
        .map(|active| quilting_gltf::scene::compute_world_transforms(&scene.nodes, active))
        .unwrap_or_default();

    // Collect all (mesh_idx, skin_idx, world_transform) from the scene graph
    let mut mesh_nodes: Vec<MeshNodeRef> = Vec::new();

    if active_scene.is_some() {
        debug_assert_eq!(node_world_transforms.len(), scene.nodes.len());
        for (node_idx, node) in scene.nodes.iter().enumerate() {
            if let Some(mi) = node.mesh {
                mesh_nodes.push(MeshNodeRef {
                    node_idx,
                    mesh_idx: mi,
                    skin_idx: node.skin,
                    world_transform: node_world_transforms[node_idx],
                });
            }
        }
    } else {
        // No scene graph — just use all meshes at identity
        let identity = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        for (mi, _) in scene.meshes.iter().enumerate() {
            let node_idx = scene
                .nodes
                .iter()
                .position(|node| node.mesh == Some(mi))
                .unwrap_or(0);
            mesh_nodes.push(MeshNodeRef {
                node_idx,
                mesh_idx: mi,
                skin_idx: None,
                world_transform: identity,
            });
        }
    }

    if mesh_nodes.is_empty() && !scene.meshes.is_empty() {
        let identity = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        mesh_nodes.push(MeshNodeRef {
            node_idx: scene
                .nodes
                .iter()
                .position(|node| node.mesh == Some(0))
                .unwrap_or(0),
            mesh_idx: 0,
            skin_idx: None,
            world_transform: identity,
        });
    }

    if mesh_nodes.is_empty() {
        web_sys::console::error_1(&"load_gltf_data: no meshes found".into());
        return JsValue::NULL;
    }

    web_sys::console::log_1(&format!(
        "load_gltf_data: {} mesh nodes in scene", mesh_nodes.len()
    ).into());

    // Find the primary skinned node for animation baking
    let primary_skinned = mesh_nodes.iter()
        .find(|mn| mn.skin_idx.is_some())
        .or(mesh_nodes.first());

    let _primary_mesh_idx = primary_skinned.map(|mn| mn.mesh_idx).unwrap_or(0);
    let primary_skin_idx = primary_skinned.and_then(|mn| mn.skin_idx);

    // Merge all meshes with world transforms, tracking per-triangle material index.
    // For animation baking we use the primary mesh; for static we merge all.
    let has_animation = !scene.animations.is_empty() && (
        primary_skin_idx.is_some() || scene.animations[0].channels.iter()
            .any(|c| c.property == quilting_gltf::animation::AnimationProperty::MorphTargetWeights)
    );
    // Hyperscape keeps node transforms as an explicit per-entity affine layer
    // before the conformal map. Legacy assets retain the viewer's historical
    // bake-and-normalize behavior.
    let authored_coordinates = scene.hyperscape.is_some();
    let authored_nodes: HashSet<usize> = scene
        .hyperscape
        .as_ref()
        .map(|asset| {
            asset
                .node_bindings
                .iter()
                .enumerate()
                .filter_map(|(node, binding)| binding.as_ref().map(|_| node))
                .collect()
        })
        .unwrap_or_default();

    // Track stable ordinary glTF node identity alongside each merged triangle.
    // The main renderer uses this to select the extracted subject/view chain.
    let mut face_material_indices: Vec<Option<usize>> = Vec::new();
    let mut face_node_indices: Vec<usize> = Vec::new();

    let combined = if has_animation {
        // For animated meshes, merge ALL mesh nodes sharing the primary skin.
        // Many glTF models split a skinned character into multiple meshes
        // (body, fur, wings, etc.) that all reference the same skeleton.
        let target_skin = primary_skin_idx;
        let skinned_meshes: Vec<(usize, usize)> = mesh_nodes.iter()
            .filter(|mn| mn.skin_idx == target_skin)
            .map(|mn| (mn.node_idx, mn.mesh_idx))
            .collect();
        web_sys::console::log_1(&format!(
            "load_gltf_data: merging {} skinned meshes for bake", skinned_meshes.len()
        ).into());
        flatten_multi_mesh_for_bake(
            &scene.meshes,
            &skinned_meshes,
            &mut face_material_indices,
            &mut face_node_indices,
        )
    } else {
        // For static models, merge ALL mesh nodes with world transforms
        merge_all_mesh_nodes(
            &scene,
            &mesh_nodes,
            &mut face_material_indices,
            &mut face_node_indices,
            &authored_nodes,
        )
    };

    let n_verts = combined.positions.len();
    let n_tris = combined.triangles.len();
    web_sys::console::log_1(&format!(
        "load_gltf_data: merged — {n_verts} vertices, {n_tris} triangles, {} materials",
        scene.materials.len()
    ).into());

    // Get time range from animation if available, otherwise default
    let (time_min, time_max) = if !scene.animations.is_empty() {
        let info = quilting_gltf::evaluator::list_animations(&scene.animations);
        if let Some(a) = info.first() { (a.t_min, a.t_max) } else { (0.0, 0.0) }
    } else {
        (0.0, 0.0)
    };

    // Build materials array using js_sys
    let js_materials = js_sys::Array::new();
    for mat in &scene.materials {
        let obj = js_sys::Object::new();

        let bc = js_sys::Array::new();
        for &c in &mat.base_color_factor {
            bc.push(&JsValue::from_f64(c));
        }
        js_sys::Reflect::set(&obj, &"base_color".into(), &bc).unwrap();
        js_sys::Reflect::set(&obj, &"metallic".into(), &JsValue::from_f64(mat.metallic_factor)).unwrap();
        js_sys::Reflect::set(&obj, &"roughness".into(), &JsValue::from_f64(mat.roughness_factor)).unwrap();

        // Texture index (resolved: texture -> image)
        let tex_idx = mat.base_color_texture.as_ref().and_then(|tex_ref| {
            scene.texture_to_image.get(tex_ref.index).copied()
        });
        if let Some(idx) = tex_idx {
            js_sys::Reflect::set(&obj, &"base_color_texture_index".into(), &JsValue::from_f64(idx as f64)).unwrap();
        } else {
            js_sys::Reflect::set(&obj, &"base_color_texture_index".into(), &JsValue::NULL).unwrap();
        }

        // Metallic-roughness texture
        let mr_idx = mat.metallic_roughness_texture.as_ref().and_then(|tex_ref| {
            scene.texture_to_image.get(tex_ref.index).copied()
        });
        if let Some(idx) = mr_idx {
            js_sys::Reflect::set(&obj, &"metallic_roughness_texture_index".into(), &JsValue::from_f64(idx as f64)).unwrap();
        } else {
            js_sys::Reflect::set(&obj, &"metallic_roughness_texture_index".into(), &JsValue::NULL).unwrap();
        }

        // Normal texture + scale
        let normal_idx = mat.normal_texture.as_ref().and_then(|tex_ref| {
            scene.texture_to_image.get(tex_ref.index).copied()
        });
        if let Some(idx) = normal_idx {
            js_sys::Reflect::set(&obj, &"normal_texture_index".into(), &JsValue::from_f64(idx as f64)).unwrap();
        } else {
            js_sys::Reflect::set(&obj, &"normal_texture_index".into(), &JsValue::NULL).unwrap();
        }
        js_sys::Reflect::set(&obj, &"normal_scale".into(), &JsValue::from_f64(mat.normal_scale)).unwrap();

        // Emissive
        let emissive = js_sys::Array::new();
        for &c in &mat.emissive_factor {
            emissive.push(&JsValue::from_f64(c));
        }
        js_sys::Reflect::set(&obj, &"emissive_factor".into(), &emissive).unwrap();
        let emissive_idx = mat.emissive_texture.as_ref().and_then(|tex_ref| {
            scene.texture_to_image.get(tex_ref.index).copied()
        });
        if let Some(idx) = emissive_idx {
            js_sys::Reflect::set(&obj, &"emissive_texture_index".into(), &JsValue::from_f64(idx as f64)).unwrap();
        } else {
            js_sys::Reflect::set(&obj, &"emissive_texture_index".into(), &JsValue::NULL).unwrap();
        }

        // Occlusion
        let occ_idx = mat.occlusion_texture.as_ref().and_then(|tex_ref| {
            scene.texture_to_image.get(tex_ref.index).copied()
        });
        if let Some(idx) = occ_idx {
            js_sys::Reflect::set(&obj, &"occlusion_texture_index".into(), &JsValue::from_f64(idx as f64)).unwrap();
        } else {
            js_sys::Reflect::set(&obj, &"occlusion_texture_index".into(), &JsValue::NULL).unwrap();
        }
        js_sys::Reflect::set(&obj, &"occlusion_strength".into(), &JsValue::from_f64(mat.occlusion_strength)).unwrap();

        // Alpha
        let alpha_mode_str = match mat.alpha_mode {
            quilting_gltf::material::AlphaMode::Opaque => "OPAQUE",
            quilting_gltf::material::AlphaMode::Mask => "MASK",
            quilting_gltf::material::AlphaMode::Blend => "BLEND",
        };
        js_sys::Reflect::set(&obj, &"alpha_mode".into(), &JsValue::from_str(alpha_mode_str)).unwrap();
        js_sys::Reflect::set(&obj, &"alpha_cutoff".into(), &JsValue::from_f64(mat.alpha_cutoff)).unwrap();
        js_sys::Reflect::set(&obj, &"double_sided".into(), &JsValue::from_bool(mat.double_sided)).unwrap();
        js_sys::Reflect::set(&obj, &"unlit".into(), &JsValue::from_bool(mat.unlit)).unwrap();

        // KHR_materials_sheen
        let sheen = js_sys::Array::new();
        for &c in &mat.sheen_color_factor { sheen.push(&JsValue::from_f64(c)); }
        js_sys::Reflect::set(&obj, &"sheen_color".into(), &sheen).unwrap();
        js_sys::Reflect::set(&obj, &"sheen_roughness".into(), &JsValue::from_f64(mat.sheen_roughness_factor)).unwrap();

        // KHR_materials_specular
        let spec = js_sys::Array::new();
        for &c in &mat.specular_color_factor { spec.push(&JsValue::from_f64(c)); }
        js_sys::Reflect::set(&obj, &"specular_color".into(), &spec).unwrap();

        // KHR_materials_ior / transmission / volume
        js_sys::Reflect::set(&obj, &"ior".into(), &JsValue::from_f64(mat.ior)).unwrap();
        js_sys::Reflect::set(&obj, &"transmission_factor".into(), &JsValue::from_f64(mat.transmission_factor)).unwrap();
        let trans_tex_idx = mat.transmission_texture.as_ref().and_then(|tex_ref| {
            scene.texture_to_image.get(tex_ref.index).copied()
        });
        if let Some(idx) = trans_tex_idx {
            js_sys::Reflect::set(&obj, &"transmission_texture_index".into(), &JsValue::from_f64(idx as f64)).unwrap();
        } else {
            js_sys::Reflect::set(&obj, &"transmission_texture_index".into(), &JsValue::NULL).unwrap();
        }
        js_sys::Reflect::set(&obj, &"thickness_factor".into(), &JsValue::from_f64(mat.thickness_factor)).unwrap();
        let attn = js_sys::Array::new();
        for &c in &mat.attenuation_color { attn.push(&JsValue::from_f64(c)); }
        js_sys::Reflect::set(&obj, &"attenuation_color".into(), &attn).unwrap();
        js_sys::Reflect::set(&obj, &"attenuation_distance".into(), &JsValue::from_f64(mat.attenuation_distance)).unwrap();

        // KHR_texture_transform on normal map
        let nuv_s = js_sys::Array::new();
        nuv_s.push(&JsValue::from_f64(mat.normal_uv_scale[0]));
        nuv_s.push(&JsValue::from_f64(mat.normal_uv_scale[1]));
        js_sys::Reflect::set(&obj, &"normal_uv_scale".into(), &nuv_s).unwrap();
        let nuv_o = js_sys::Array::new();
        nuv_o.push(&JsValue::from_f64(mat.normal_uv_offset[0]));
        nuv_o.push(&JsValue::from_f64(mat.normal_uv_offset[1]));
        js_sys::Reflect::set(&obj, &"normal_uv_offset".into(), &nuv_o).unwrap();
        js_sys::Reflect::set(&obj, &"normal_uv_rotation".into(), &JsValue::from_f64(mat.normal_uv_rotation)).unwrap();

        // KHR_texture_transform on base color texture
        let buv_s = js_sys::Array::new();
        buv_s.push(&JsValue::from_f64(mat.base_uv_scale[0]));
        buv_s.push(&JsValue::from_f64(mat.base_uv_scale[1]));
        js_sys::Reflect::set(&obj, &"base_uv_scale".into(), &buv_s).unwrap();
        let buv_o = js_sys::Array::new();
        buv_o.push(&JsValue::from_f64(mat.base_uv_offset[0]));
        buv_o.push(&JsValue::from_f64(mat.base_uv_offset[1]));
        js_sys::Reflect::set(&obj, &"base_uv_offset".into(), &buv_o).unwrap();
        js_sys::Reflect::set(&obj, &"base_uv_rotation".into(), &JsValue::from_f64(mat.base_uv_rotation)).unwrap();

        js_materials.push(&obj);
    }

    // Build textures array — send raw image blobs for browser-native decoding
    let js_textures = js_sys::Array::new();
    for tex_info in &scene.raw_textures {
        let obj = js_sys::Object::new();
        // Send raw encoded bytes (PNG/JPEG) for browser-native decoding
        let raw = js_sys::Uint8Array::new_with_length(tex_info.blob.data.len() as u32);
        raw.copy_from(&tex_info.blob.data);
        js_sys::Reflect::set(&obj, &"raw_data".into(), &raw).unwrap();
        js_sys::Reflect::set(&obj, &"mime_type".into(), &JsValue::from_str(&tex_info.blob.mime_type)).unwrap();
        // Sampler wrap modes
        js_sys::Reflect::set(&obj, &"wrap_s".into(), &JsValue::from_f64(tex_info.wrap_s as f64)).unwrap();
        js_sys::Reflect::set(&obj, &"wrap_t".into(), &JsValue::from_f64(tex_info.wrap_t as f64)).unwrap();

        js_textures.push(&obj);
    }

    // Build per-face material index array (Int32Array, -1 for no material)
    let js_face_materials = js_sys::Int32Array::new_with_length(face_material_indices.len() as u32);
    let mat_indices: Vec<i32> = face_material_indices.iter()
        .map(|mi| mi.map(|i| i as i32).unwrap_or(-1))
        .collect();
    js_face_materials.copy_from(&mat_indices);

    let js_face_nodes = js_sys::Int32Array::new_with_length(face_node_indices.len() as u32);
    let node_indices: Vec<i32> = face_node_indices
        .iter()
        .map(|&node| i32::try_from(node).unwrap_or(i32::MAX))
        .collect();
    js_face_nodes.copy_from(&node_indices);

    // Dense authored identity table indexed exactly like the source glTF node
    // array. Null entries are deliberate. Ordinary assets return an empty
    // table rather than allocating/cloning one null per renderer-only node.
    let authored_node_count = scene
        .hyperscape
        .as_ref()
        .map(|asset| asset.node_bindings.len())
        .unwrap_or(0);
    let js_node_stable_entity_ids = js_sys::Array::new_with_length(authored_node_count as u32);
    for node in 0..authored_node_count {
        let stable_id = scene
            .hyperscape
            .as_ref()
            .and_then(|asset| asset.node_bindings.get(node))
            .and_then(Option::as_ref)
            .and_then(|binding| binding.stable_id);
        js_node_stable_entity_ids.set(
            node as u32,
            stable_id
                .map(|stable_id| JsValue::from_str(&stable_id.to_string()))
                .unwrap_or(JsValue::NULL),
        );
    }

    // Authored Hyperscape geometry deliberately remains node-local so the
    // renderer can compose source node transforms with presentation layers and
    // future live edits. Ordinary assets retain baked coordinates and therefore
    // return an empty table instead of cloning one matrix per source node.
    let js_node_world_transforms = if authored_coordinates {
        let values: Vec<f32> = node_world_transforms
            .iter()
            .flat_map(|matrix| matrix.iter().map(|&value| value as f32))
            .collect();
        let transforms = js_sys::Float32Array::new_with_length(values.len() as u32);
        transforms.copy_from(&values);
        transforms
    } else {
        js_sys::Float32Array::new_with_length(0)
    };

    // Extract first material as default (backward compat).
    // With raw image blobs (browser-native decode), we use base_color_factor directly.
    let (base_color, metallic, roughness) = if !scene.materials.is_empty() {
        let mat = &scene.materials[0];
        let bc = [
            mat.base_color_factor[0] as f32, mat.base_color_factor[1] as f32,
            mat.base_color_factor[2] as f32, mat.base_color_factor[3] as f32,
        ];
        (bc, mat.metallic_factor as f32, mat.roughness_factor as f32)
    } else {
        ([0.9, 0.75, 0.6, 1.0], 0.0, 0.4)
    };

    // Build AnimationEvaluator for GPU animation (skeletal and/or morph targets)
    let evaluator = if !scene.animations.is_empty() {
        let skin = primary_skin_idx.and_then(|si| scene.skins.get(si).cloned());
        let num_morph = combined.morph_targets.len();
        Some(quilting_gltf::evaluator::AnimationEvaluator::new(
            scene.animations[0].clone(),
            skin,
            scene.nodes.clone(),
            num_morph,
        ))
    } else { None };

    // Compute rest-pose bounding box for normalization
    let (norm_center, norm_scale) = if authored_coordinates {
        ([0.0; 3], 1.0)
    } else {
        let mut bb_min = [f64::INFINITY; 3];
        let mut bb_max = [f64::NEG_INFINITY; 3];
        for pos in &combined.positions {
            for i in 0..3 {
                bb_min[i] = bb_min[i].min(pos[i]);
                bb_max[i] = bb_max[i].max(pos[i]);
            }
        }
        let center = [(bb_min[0]+bb_max[0])*0.5, (bb_min[1]+bb_max[1])*0.5, (bb_min[2]+bb_max[2])*0.5];
        let extent = ((bb_max[0]-bb_min[0]).max(bb_max[1]-bb_min[1]).max(bb_max[2]-bb_min[2])) * 0.5;
        let scale = if extent > 1e-10 { 1.0 / extent } else { 1.0 };
        (center, scale)
    };

    // Store glTF data for animation switching
    GLTF_DATA.with(|gd| {
        *gd.borrow_mut() = Some(StoredGltfData {
            animations: scene.animations.clone(),
            skins: scene.skins.clone(),
            nodes: scene.nodes.clone(),
            combined,
            face_material_indices: face_material_indices.clone(),
            face_node_indices: face_node_indices.clone(),
            primary_skin_idx,
            active_animation: 0,
            evaluator,
            norm_center,
            norm_scale,
            cached_pose: RefCell::new(None),
        });
    });
    // The uploaded GPU topology belongs to the previous retained model until
    // upload_model_to_compute installs this model's validated shape.
    LOD_COMPUTE_MODEL.with(|model| *model.borrow_mut() = None);
    LOD_COMPUTE_MODEL_FINGERPRINT.with(|fingerprint| {
        *fingerprint.borrow_mut() = None;
    });
    reset_animated_lod_delta();

    FACE_MATERIALS.with(|fm| *fm.borrow_mut() = face_material_indices);
    SENT_TESS.with(|s| s.borrow_mut().clear());
    // A previously loaded test shape takes priority in remesh_current_model —
    // drop it, or remeshing would silently operate on the stale shape.
    REMESH_SOURCE.with(|rs| *rs.borrow_mut() = None);
    REMESH_DATA.with(|rd| *rd.borrow_mut() = None);

    let t_end = js_sys::Date::now();
    web_sys::console::log_1(&format!(
        "load_gltf_data: total {:.0}ms — {} verts, {} faces, {} materials, {} textures, time [{:.3}, {:.3}]",
        t_end - t0, n_verts, n_tris, scene.materials.len(), scene.raw_textures.len(), time_min, time_max
    ).into());

    // Build result object using js_sys (not serde) for speed
    let result = js_sys::Object::new();
    let js_asset_metadata =
        serde_wasm_bindgen::to_value(&scene.asset_metadata).unwrap_or(JsValue::NULL);
    js_sys::Reflect::set(
        &result,
        &"asset_metadata".into(),
        &js_asset_metadata,
    ).unwrap();
    js_sys::Reflect::set(&result, &"time_min".into(), &JsValue::from_f64(time_min)).unwrap();
    js_sys::Reflect::set(&result, &"time_max".into(), &JsValue::from_f64(time_max)).unwrap();
    js_sys::Reflect::set(&result, &"num_vertices".into(), &JsValue::from_f64(n_verts as f64)).unwrap();
    js_sys::Reflect::set(&result, &"num_faces".into(), &JsValue::from_f64(n_tris as f64)).unwrap();
    js_sys::Reflect::set(&result, &"materials".into(), &js_materials).unwrap();
    js_sys::Reflect::set(&result, &"textures".into(), &js_textures).unwrap();
    js_sys::Reflect::set(
        &result,
        &"has_hyperscape".into(),
        &JsValue::from_bool(authored_coordinates),
    ).unwrap();
    js_sys::Reflect::set(&result, &"face_material_indices".into(), &js_face_materials).unwrap();
    js_sys::Reflect::set(&result, &"face_node_indices".into(), &js_face_nodes).unwrap();
    js_sys::Reflect::set(
        &result,
        &"node_stable_entity_ids".into(),
        &js_node_stable_entity_ids,
    ).unwrap();
    js_sys::Reflect::set(
        &result,
        &"node_world_transforms".into(),
        &js_node_world_transforms,
    ).unwrap();

    // Backward-compat scalar fields
    let bc_arr = js_sys::Array::new();
    for &c in &base_color { bc_arr.push(&JsValue::from_f64(c as f64)); }
    js_sys::Reflect::set(&result, &"base_color".into(), &bc_arr).unwrap();
    js_sys::Reflect::set(&result, &"metallic".into(), &JsValue::from_f64(metallic as f64)).unwrap();
    js_sys::Reflect::set(&result, &"roughness".into(), &JsValue::from_f64(roughness as f64)).unwrap();

    // KHR_materials_variants: extract from raw glTF JSON
    // Parse the GLB header to get the JSON chunk and extract variant data.
    let js_variants = js_sys::Array::new();
    let js_variant_mappings = js_sys::Array::new();
    if data.len() > 12 {
        // Try to parse as GLB and extract variant info
        if let Ok(gltf_json) = {
            let mut pos = 12usize; // skip GLB header
            if pos + 8 <= data.len() {
                let chunk_len = u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]) as usize;
                pos += 8; // skip chunk header
                if pos + chunk_len <= data.len() {
                    serde_json::from_slice::<serde_json::Value>(&data[pos..pos+chunk_len])
                } else {
                    Err(serde_json::Error::io(std::io::Error::new(std::io::ErrorKind::Other, "bad chunk")))
                }
            } else {
                Err(serde_json::Error::io(std::io::Error::new(std::io::ErrorKind::Other, "bad glb")))
            }
        } {
            // Extract variant names
            if let Some(vars) = gltf_json.pointer("/extensions/KHR_materials_variants/variants")
                .and_then(|v| v.as_array())
            {
                for v in vars {
                    let name = v.get("name").and_then(|n| n.as_str()).unwrap_or("unnamed");
                    js_variants.push(&JsValue::from_str(name));
                }
            }
            // Extract per-primitive variant mappings
            if let Some(meshes) = gltf_json.get("meshes").and_then(|v| v.as_array()) {
                for mesh in meshes {
                    if let Some(prims) = mesh.get("primitives").and_then(|v| v.as_array()) {
                        for prim in prims {
                            if let Some(mappings) = prim.pointer("/extensions/KHR_materials_variants/mappings")
                                .and_then(|v| v.as_array())
                            {
                                for m in mappings {
                                    let obj = js_sys::Object::new();
                                    let mat = m.get("material").and_then(|v| v.as_u64()).unwrap_or(0);
                                    js_sys::Reflect::set(&obj, &"material".into(), &JsValue::from_f64(mat as f64)).unwrap();
                                    let vi = js_sys::Array::new();
                                    if let Some(vars) = m.get("variants").and_then(|v| v.as_array()) {
                                        for v in vars { vi.push(&JsValue::from_f64(v.as_u64().unwrap_or(0) as f64)); }
                                    }
                                    js_sys::Reflect::set(&obj, &"variants".into(), &vi).unwrap();
                                    js_variant_mappings.push(&obj);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    js_sys::Reflect::set(&result, &"variant_names".into(), &js_variants).unwrap();
    js_sys::Reflect::set(&result, &"variant_mappings".into(), &js_variant_mappings).unwrap();

    // Animation list for selector UI
    let anim_info = quilting_gltf::evaluator::list_animations(&scene.animations);
    let js_animations = js_sys::Array::new();
    for info in &anim_info {
        let obj = js_sys::Object::new();
        js_sys::Reflect::set(&obj, &"index".into(), &JsValue::from_f64(info.index as f64)).unwrap();
        let fallback = format!("Animation {}", info.index);
        let name = info.name.as_deref().unwrap_or(&fallback);
        js_sys::Reflect::set(&obj, &"name".into(), &JsValue::from_str(name)).unwrap();
        js_sys::Reflect::set(&obj, &"duration".into(), &JsValue::from_f64(info.duration)).unwrap();
        js_sys::Reflect::set(&obj, &"t_min".into(), &JsValue::from_f64(info.t_min)).unwrap();
        js_sys::Reflect::set(&obj, &"t_max".into(), &JsValue::from_f64(info.t_max)).unwrap();
        js_animations.push(&obj);
    }
    js_sys::Reflect::set(&result, &"animations".into(), &js_animations).unwrap();

    result.into()
}

/// Flatten all primitives in a mesh into a single Primitive suitable for baking.
/// Merges positions, triangles, joint data with proper index offsets.
/// Tracks per-triangle material indices in the output vec.
/// Merge multiple meshes (by index) into one Primitive for skinned animation baking.
/// All meshes should share the same skin/skeleton.
fn flatten_multi_mesh_for_bake(
    meshes: &[quilting_gltf::mesh::Mesh],
    mesh_nodes: &[(usize, usize)],
    face_materials: &mut Vec<Option<usize>>,
    face_nodes: &mut Vec<usize>,
) -> quilting_gltf::mesh::Primitive {
    let mut positions = Vec::new();
    let mut normals_all: Option<Vec<[f64; 3]>> = None;
    let mut uvs_all: Option<Vec<[f64; 2]>> = None;
    let mut triangles = Vec::new();
    let mut joint_indices_all: Option<Vec<[u16; 4]>> = None;
    let mut joint_weights_all: Option<Vec<[f32; 4]>> = None;

    for &(node_idx, mi) in mesh_nodes {
        if mi >= meshes.len() { continue; }
        let mesh = &meshes[mi];
        for prim in &mesh.primitives {
            let offset = positions.len();
            positions.extend_from_slice(&prim.positions);
            let new_tris: Vec<[usize; 3]> = prim.triangles.iter()
                .map(|t| [t[0] + offset, t[1] + offset, t[2] + offset])
                .collect();
            for _ in &new_tris {
                face_materials.push(prim.material_index);
                face_nodes.push(node_idx);
            }
            triangles.extend(new_tris);

            if let Some(ref n) = prim.normals {
                normals_all.get_or_insert_with(Vec::new).extend_from_slice(n);
            }
            if let Some(ref uv) = prim.uvs {
                uvs_all.get_or_insert_with(Vec::new).extend_from_slice(uv);
            }
            if let Some(ref ji) = prim.joint_indices {
                joint_indices_all.get_or_insert_with(Vec::new).extend_from_slice(ji);
            }
            if let Some(ref jw) = prim.joint_weights {
                joint_weights_all.get_or_insert_with(Vec::new).extend_from_slice(jw);
            }
        }
    }

    // Use morph targets from the first mesh's first primitive (if any)
    let morph_targets = mesh_nodes.first()
        .and_then(|&(_, mi)| meshes.get(mi))
        .and_then(|m| m.primitives.first())
        .map(|p| p.morph_targets.clone())
        .unwrap_or_default();

    quilting_gltf::mesh::Primitive {
        positions,
        normals: normals_all,
        uvs: uvs_all,
        triangles,
        material_index: None,
        joint_indices: joint_indices_all,
        joint_weights: joint_weights_all,
        morph_targets,
        tangents: None,
    }
}

/// Merge all mesh nodes in the scene into one combined Primitive,
/// applying world transforms to positions and normals.
fn merge_all_mesh_nodes(
    scene: &quilting_gltf::GltfSceneRaw,
    mesh_nodes: &[MeshNodeRef],
    face_materials: &mut Vec<Option<usize>>,
    face_nodes: &mut Vec<usize>,
    authored_nodes: &HashSet<usize>,
) -> quilting_gltf::mesh::Primitive {
    let mut positions = Vec::new();
    let mut normals_all: Option<Vec<[f64; 3]>> = None;
    let mut uvs_all: Option<Vec<[f64; 2]>> = None;
    let mut triangles = Vec::new();

    for mn in mesh_nodes {
        let mesh = &scene.meshes[mn.mesh_idx];
        let identity = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let m = if authored_nodes.contains(&mn.node_idx) {
            &identity
        } else {
            &mn.world_transform
        };

        // Extract 3x3 normal matrix (inverse transpose of upper-left 3x3)
        // For uniform/no-shear transforms, the upper-left 3x3 works directly
        let nm = [
            m[0], m[1], m[2],
            m[4], m[5], m[6],
            m[8], m[9], m[10],
        ];

        for prim in &mesh.primitives {
            let offset = positions.len();

            // Transform positions by world matrix
            for &pos in &prim.positions {
                let x = m[0]*pos[0] + m[4]*pos[1] + m[8]*pos[2]  + m[12];
                let y = m[1]*pos[0] + m[5]*pos[1] + m[9]*pos[2]  + m[13];
                let z = m[2]*pos[0] + m[6]*pos[1] + m[10]*pos[2] + m[14];
                positions.push([x, y, z]);
            }

            // Transform normals — pad with defaults if primitive lacks them
            {
                let out = normals_all.get_or_insert_with(Vec::new);
                if let Some(ref normals) = prim.normals {
                    for &n in normals {
                        let nx = nm[0]*n[0] + nm[3]*n[1] + nm[6]*n[2];
                        let ny = nm[1]*n[0] + nm[4]*n[1] + nm[7]*n[2];
                        let nz = nm[2]*n[0] + nm[5]*n[1] + nm[8]*n[2];
                        let len = (nx*nx + ny*ny + nz*nz).sqrt();
                        if len > 1e-10 {
                            out.push([nx/len, ny/len, nz/len]);
                        } else {
                            out.push([0.0, 1.0, 0.0]);
                        }
                    }
                } else {
                    // Pad with up vector for vertices without normals
                    out.resize(out.len() + prim.positions.len(), [0.0, 1.0, 0.0]);
                }
            }

            // UVs — pad with zeros if primitive lacks them
            {
                let out = uvs_all.get_or_insert_with(Vec::new);
                if let Some(ref uv) = prim.uvs {
                    out.extend_from_slice(uv);
                } else {
                    out.resize(out.len() + prim.positions.len(), [0.0, 0.0]);
                }
            }

            let new_tris: Vec<[usize; 3]> = prim.triangles.iter()
                .map(|t| [t[0] + offset, t[1] + offset, t[2] + offset])
                .collect();
            for _ in &new_tris {
                face_materials.push(prim.material_index);
                face_nodes.push(mn.node_idx);
            }
            triangles.extend(new_tris);
        }
    }

    quilting_gltf::mesh::Primitive {
        positions,
        normals: normals_all,
        uvs: uvs_all,
        triangles,
        material_index: None,
        joint_indices: None,
        joint_weights: None,
        morph_targets: vec![],
        tangents: None,
    }
}

// ============================================================
// Remeshing integration
// ============================================================

thread_local! {
    static REMESH_DATA: RefCell<Option<RemeshedModel>> = RefCell::new(None);
}

struct RemeshedModel {
    patches: Vec<quilting_core::patch::QBTriPatch>,
    patch_uvs: Vec<[[f32; 2]; 3]>,
    patch_normals: Vec<[[f32; 3]; 3]>,
}

// ============================================================
// Remesh Lab — clean API for test harness
// ============================================================

/// Generate a test mesh. Returns { positions: [[x,y,z],...], faces: [[a,b,c],...], num_vertices, num_faces }.
#[wasm_bindgen]
pub fn generate_test_mesh(shape: &str, subdivisions: u32) -> JsValue {
    let (positions, faces) = match shape {
        "sphere" => quilting_remesh::test_shapes::sphere(subdivisions),
        "cylinder" => quilting_remesh::test_shapes::cylinder(
            (8 * 2_usize.pow(subdivisions.min(4))) .min(64),
            (4 * 2_usize.pow(subdivisions.min(4))).min(32),
            1.0, 0.3,
        ),
        "torus" => generate_torus(subdivisions),
        _ => return JsValue::NULL,
    };

    let result = js_sys::Object::new();

    // Positions as nested array
    let js_pos = js_sys::Array::new();
    for p in &positions {
        let arr = js_sys::Array::of3(
            &JsValue::from_f64(p[0]),
            &JsValue::from_f64(p[1]),
            &JsValue::from_f64(p[2]),
        );
        js_pos.push(&arr);
    }
    js_sys::Reflect::set(&result, &"positions".into(), &js_pos).unwrap();

    let js_faces = js_sys::Array::new();
    for f in &faces {
        let arr = js_sys::Array::of3(
            &JsValue::from_f64(f[0] as f64),
            &JsValue::from_f64(f[1] as f64),
            &JsValue::from_f64(f[2] as f64),
        );
        js_faces.push(&arr);
    }
    js_sys::Reflect::set(&result, &"faces".into(), &js_faces).unwrap();
    js_sys::Reflect::set(&result, &"num_vertices".into(), &JsValue::from_f64(positions.len() as f64)).unwrap();
    js_sys::Reflect::set(&result, &"num_faces".into(), &JsValue::from_f64(faces.len() as f64)).unwrap();

    result.into()
}

/// Simplify a mesh via QEM edge collapse.
/// positions: flat f64 array [x0,y0,z0, x1,y1,z1, ...]
/// faces: flat u32 array [a0,b0,c0, a1,b1,c1, ...]
/// Returns { positions, faces, max_error }.
#[wasm_bindgen]
pub fn simplify_mesh(positions: &[f64], faces: &[u32], target: u32) -> JsValue {
    let pos: Vec<[f64; 3]> = positions.chunks(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    let tris: Vec<[usize; 3]> = faces.chunks(3)
        .map(|c| [c[0] as usize, c[1] as usize, c[2] as usize])
        .collect();

    let (simp_pos, simp_faces) = quilting_remesh::simplify::simplify(&pos, &tris, target as usize);

    // Compute max error
    let mut max_err = 0.0_f64;
    for p in &pos {
        let mut min_d = f64::MAX;
        for f in &simp_faces {
            let d = point_plane_dist(*p, simp_pos[f[0]], simp_pos[f[1]], simp_pos[f[2]]);
            min_d = min_d.min(d);
        }
        max_err = max_err.max(min_d);
    }

    let result = js_sys::Object::new();

    let js_pos = js_sys::Array::new();
    for p in &simp_pos {
        let arr = js_sys::Array::of3(
            &JsValue::from_f64(p[0]),
            &JsValue::from_f64(p[1]),
            &JsValue::from_f64(p[2]),
        );
        js_pos.push(&arr);
    }
    js_sys::Reflect::set(&result, &"positions".into(), &js_pos).unwrap();

    let js_faces = js_sys::Array::new();
    for f in &simp_faces {
        let arr = js_sys::Array::of3(
            &JsValue::from_f64(f[0] as f64),
            &JsValue::from_f64(f[1] as f64),
            &JsValue::from_f64(f[2] as f64),
        );
        js_faces.push(&arr);
    }
    js_sys::Reflect::set(&result, &"faces".into(), &js_faces).unwrap();
    js_sys::Reflect::set(&result, &"max_error".into(), &JsValue::from_f64(max_err)).unwrap();

    result.into()
}

/// Simplify mesh with curved QB patch fitting.
/// Returns tessellated curved patches (sampled from the QB surface, not flat triangles).
#[wasm_bindgen]
pub fn simplify_mesh_curved(positions: &[f64], faces: &[u32], target: u32, tess_res: u32) -> JsValue {
    let pos: Vec<[f64; 3]> = positions.chunks(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    let tris: Vec<[usize; 3]> = faces.chunks(3)
        .map(|c| [c[0] as usize, c[1] as usize, c[2] as usize])
        .collect();

    let res = tess_res.max(1).min(10) as usize;

    match quilting_remesh::remesh_simplified_curved(&pos, &tris, target as usize) {
        Ok(remesh_result) => {
            // Tessellate each curved QB patch at the requested resolution
            let mut all_positions: Vec<[f64; 3]> = Vec::new();
            let mut all_faces: Vec<[usize; 3]> = Vec::new();

            for patch in &remesh_result.patches {
                let tess = quilting_remesh::roundtrip::tessellate_patch(patch, res);
                let offset = all_positions.len();
                all_positions.extend_from_slice(&tess.positions);
                for f in &tess.faces {
                    all_faces.push([f[0] + offset, f[1] + offset, f[2] + offset]);
                }
            }

            let result = js_sys::Object::new();

            let js_pos = js_sys::Array::new();
            for p in &all_positions {
                let arr = js_sys::Array::of3(
                    &JsValue::from_f64(p[0]),
                    &JsValue::from_f64(p[1]),
                    &JsValue::from_f64(p[2]),
                );
                js_pos.push(&arr);
            }
            js_sys::Reflect::set(&result, &"positions".into(), &js_pos).unwrap();

            let js_faces = js_sys::Array::new();
            for f in &all_faces {
                let arr = js_sys::Array::of3(
                    &JsValue::from_f64(f[0] as f64),
                    &JsValue::from_f64(f[1] as f64),
                    &JsValue::from_f64(f[2] as f64),
                );
                js_faces.push(&arr);
            }
            js_sys::Reflect::set(&result, &"faces".into(), &js_faces).unwrap();
            js_sys::Reflect::set(&result, &"num_patches".into(),
                &JsValue::from_f64(remesh_result.patches.len() as f64)).unwrap();
            js_sys::Reflect::set(&result, &"max_error".into(),
                &JsValue::from_f64(remesh_result.stats.max_position_error)).unwrap();

            result.into()
        }
        Err(_) => JsValue::NULL,
    }
}

/// Quadric VSA segmentation — clusters by fitted quadric surface instead of planar normal.
/// Returns the same format as simplify_mesh (positions, faces, max_error) plus
/// a `surface_types` array with per-cluster classification.
#[wasm_bindgen]
pub fn quadric_vsa_segment(positions: &[f64], faces: &[u32], target: u32) -> JsValue {
    let pos: Vec<[f64; 3]> = positions.chunks(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    let tris: Vec<[usize; 3]> = faces.chunks(3)
        .map(|c| [c[0] as usize, c[1] as usize, c[2] as usize])
        .collect();

    if tris.len() < 4 {
        return JsValue::NULL;
    }

    // Build half-edge mesh for VSA
    let faces_u32: Vec<[u32; 3]> = tris.iter()
        .map(|f| [f[0] as u32, f[1] as u32, f[2] as u32])
        .collect();
    let he_mesh = quilting_mesh::HalfEdgeMesh::from_triangles(pos.len() as u32, &faces_u32);

    let config = quilting_remesh::quadric_vsa::QuadricVsaConfig {
        target_clusters: target as usize,
        max_iterations: 20,
        sharp_edge_threshold: 40.0_f64.to_radians(),
        quadric_weight: 0.5,
    };
    let result = quilting_remesh::quadric_vsa::segment(&pos, &tris, &he_mesh, &config);

    // Build response object
    let obj = js_sys::Object::new();

    // Face labels (per-face cluster ID)
    let js_labels = js_sys::Array::new();
    for &l in &result.face_labels {
        js_labels.push(&JsValue::from_f64(l as f64));
    }
    js_sys::Reflect::set(&obj, &"labels".into(), &js_labels).unwrap();
    js_sys::Reflect::set(&obj, &"num_clusters".into(), &JsValue::from_f64(result.num_clusters as f64)).unwrap();

    // Surface type per cluster
    let js_types = js_sys::Array::new();
    for proxy in &result.proxies {
        let t = match &proxy.surface_type {
            quilting_remesh::quadric_vsa::SurfaceType::Plane => "plane",
            quilting_remesh::quadric_vsa::SurfaceType::Sphere { .. } => "sphere",
            quilting_remesh::quadric_vsa::SurfaceType::Cylinder { .. } => "cylinder",
            quilting_remesh::quadric_vsa::SurfaceType::General => "general",
        };
        js_types.push(&JsValue::from_str(t));
    }
    js_sys::Reflect::set(&obj, &"surface_types".into(), &js_types).unwrap();

    // Proxy centroids (for visualization)
    let js_centroids = js_sys::Array::new();
    for proxy in &result.proxies {
        let arr = js_sys::Array::of3(
            &JsValue::from_f64(proxy.centroid[0]),
            &JsValue::from_f64(proxy.centroid[1]),
            &JsValue::from_f64(proxy.centroid[2]),
        );
        js_centroids.push(&arr);
    }
    js_sys::Reflect::set(&obj, &"centroids".into(), &js_centroids).unwrap();

    obj.into()
}

fn point_plane_dist(p: [f64; 3], a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    let ab = [b[0]-a[0], b[1]-a[1], b[2]-a[2]];
    let ac = [c[0]-a[0], c[1]-a[1], c[2]-a[2]];
    let n = [ab[1]*ac[2]-ab[2]*ac[1], ab[2]*ac[0]-ab[0]*ac[2], ab[0]*ac[1]-ab[1]*ac[0]];
    let len = (n[0]*n[0] + n[1]*n[1] + n[2]*n[2]).sqrt();
    if len < 1e-15 { return f64::MAX; }
    let ap = [p[0]-a[0], p[1]-a[1], p[2]-a[2]];
    (ap[0]*n[0] + ap[1]*n[1] + ap[2]*n[2]).abs() / len
}

fn generate_torus(subdivisions: u32) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let seg = (8 * 2_usize.pow(subdivisions.min(3))).min(48);
    let ring = seg;
    let r_major = 0.5;
    let r_minor = 0.2;

    let mut positions = Vec::new();
    let mut faces = Vec::new();

    for i in 0..seg {
        let theta = 2.0 * std::f64::consts::PI * (i as f64 / seg as f64);
        for j in 0..ring {
            let phi = 2.0 * std::f64::consts::PI * (j as f64 / ring as f64);
            let x = (r_major + r_minor * phi.cos()) * theta.cos();
            let y = r_minor * phi.sin();
            let z = (r_major + r_minor * phi.cos()) * theta.sin();
            positions.push([x, y, z]);
        }
    }

    for i in 0..seg {
        let ni = (i + 1) % seg;
        for j in 0..ring {
            let nj = (j + 1) % ring;
            let a = i * ring + j;
            let b = i * ring + nj;
            let c = ni * ring + nj;
            let d = ni * ring + j;
            faces.push([a, b, c]);
            faces.push([a, c, d]);
        }
    }

    (positions, faces)
}

/// Load a test shape (sphere or cylinder) as the current model for remeshing experiments.
/// shape: "sphere" or "cylinder"
/// param1: subdivisions (sphere) or segments (cylinder)
/// param2: unused (sphere) or rings (cylinder)
#[wasm_bindgen]
pub fn load_test_shape(shape: &str, param1: u32, param2: u32) -> JsValue {
    let (positions, faces) = match shape {
        "sphere" => quilting_remesh::test_shapes::sphere(param1),
        "cylinder" => quilting_remesh::test_shapes::cylinder(param1 as usize, param2 as usize, 1.0, 0.3),
        _ => return JsValue::NULL,
    };

    let normals = quilting_remesh::geometry::compute_vertex_normals(&positions, &faces);

    let num_faces = faces.len();
    let num_verts = positions.len();

    // Test shapes bypass StoredGltfData entirely: the geometry goes into
    // REMESH_SOURCE (below) and the render path gets instances built here.

    // Pack instances in the compact format for rendering
    let mut instances = vec![0.0f32; num_faces * instance_layout::STRIDE];

    for (fi, face) in faces.iter().enumerate() {
        let mut w = InstanceWriter::new(&mut instances, fi);
        for vi in 0..3 {
            let p = positions[face[vi]];
            w.set_position(vi, face[vi] as u32, [p[0] as f32, p[1] as f32, p[2] as f32]);
            let n = normals[face[vi]];
            w.set_normal(vi, [n[0] as f32, n[1] as f32, n[2] as f32]);
        }
        w.set_edge_lods([4.0; 3]);
        w.set_vertex_lods([4.0; 3]);
    }

    // Also store positions/faces for remeshing
    REMESH_SOURCE.with(|rs| {
        *rs.borrow_mut() = Some(RemeshSource {
            positions: positions.clone(),
            faces: faces.clone(),
        });
    });

    // Build face LODs
    let mut face_lods = Vec::with_capacity(num_faces * 6);
    let canon = quilting_core::permutation::canonical_form([4, 4, 4]);
    for _ in 0..num_faces {
        face_lods.push(canon.res[0] as f32);
        face_lods.push(canon.res[1] as f32);
        face_lods.push(canon.res[2] as f32);
        face_lods.push(canon.perm_index as f32);
        face_lods.push(quilting_core::permutation::perm_sign(canon.perm_index) as f32);
        face_lods.push(0.0f32); // atlas_idx
    }

    let result = js_sys::Object::new();
    let js_instances = js_sys::Float32Array::new_with_length(instances.len() as u32);
    js_instances.copy_from(&instances);
    js_sys::Reflect::set(&result, &"instances".into(), &js_instances).unwrap();
    js_sys::Reflect::set(&result, &"num_faces".into(), &JsValue::from_f64(num_faces as f64)).unwrap();
    js_sys::Reflect::set(&result, &"num_vertices".into(), &JsValue::from_f64(num_verts as f64)).unwrap();

    let js_lods = js_sys::Float32Array::new_with_length(face_lods.len() as u32);
    js_lods.copy_from(&face_lods);
    js_sys::Reflect::set(&result, &"face_lods".into(), &js_lods).unwrap();

    let js_materials = js_sys::Int32Array::new_with_length(num_faces as u32);
    js_sys::Reflect::set(&result, &"face_materials".into(), &js_materials).unwrap();

    web_sys::console::log_1(&format!(
        "load_test_shape: {} — {} verts, {} faces", shape, num_verts, num_faces
    ).into());

    result.into()
}

/// Build a tiny Rust-authored scene for interactively explaining QB patches
/// and adaptive tessellation. LOD fields are updated separately so animation
/// transfers only the compact six-float classification, not source instances.
#[wasm_bindgen]
pub fn load_patch_lab(shape: &str, grid: u32, bend: f32) -> JsValue {
    use quilting_core::educational::{PatchLabMesh, PatchLabShape};

    let shape = match shape {
        "triangle" => PatchLabShape::Triangle,
        "plane" => PatchLabShape::Plane,
        "cube" => PatchLabShape::Cube,
        _ => return JsValue::NULL,
    };
    let mesh = PatchLabMesh::new(shape, grid, bend as f64);
    let num_faces = mesh.faces.len();
    let mut instances = vec![0.0f32; num_faces * instance_layout::STRIDE];
    for (face_index, face) in mesh.faces.iter().enumerate() {
        let mut writer = InstanceWriter::new(&mut instances, face_index);
        for corner in 0..3 {
            let vertex = face[corner];
            let point = mesh.positions[vertex as usize];
            let weight = mesh.face_weights[face_index][corner];
            writer.set_position(
                corner,
                vertex,
                [point[0] as f32, point[1] as f32, point[2] as f32],
            );
            writer.set_weight(corner, [
                weight[0] as f32,
                weight[1] as f32,
                weight[2] as f32,
                weight[3] as f32,
            ]);
        }
        writer.set_edge_lods([4.0; 3]);
        writer.set_vertex_lods([4.0; 3]);
        writer.set_face_id(face_index as u32);
        writer.set_uvs([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
        // Zero smooth normals intentionally select analytic QB normals in the
        // vertex shader. This keeps the curved triangle honest and the cube
        // faceted without duplicating its corner vertices.
    }

    REMESH_SOURCE.with(|source| {
        *source.borrow_mut() = Some(RemeshSource {
            positions: mesh.positions.clone(),
            faces: mesh.faces.iter()
                .map(|face| face.map(|vertex| vertex as usize))
                .collect(),
        });
    });
    PATCH_LAB_SOURCE.with(|source| *source.borrow_mut() = Some(mesh.clone()));

    let result = js_sys::Object::new();
    let js_instances = js_sys::Float32Array::new_with_length(instances.len() as u32);
    js_instances.copy_from(&instances);
    js_sys::Reflect::set(&result, &"instances".into(), &js_instances).unwrap();
    js_sys::Reflect::set(&result, &"num_faces".into(),
        &JsValue::from_f64(num_faces as f64)).unwrap();
    js_sys::Reflect::set(&result, &"num_vertices".into(),
        &JsValue::from_f64(mesh.positions.len() as f64)).unwrap();
    let materials = js_sys::Int32Array::new_with_length(num_faces as u32);
    js_sys::Reflect::set(&result, &"face_materials".into(), &materials).unwrap();
    result.into()
}

/// Re-sample and reconcile the current patch-laboratory LOD field.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn update_patch_lab_lods(
    field: &str,
    phase: f32,
    min_exp: u32,
    max_exp: u32,
    edge_a_exp: u32,
    edge_b_exp: u32,
    edge_c_exp: u32,
    max_face_edge_ratio: u32,
) -> JsValue {
    use quilting_core::educational::{PatchLabField, PatchLabLodConfig};

    let field = match field {
        "uniform" => PatchLabField::Uniform,
        "wave" => PatchLabField::Wave,
        "radial" => PatchLabField::Radial,
        "sweep" => PatchLabField::Sweep,
        "edges" => PatchLabField::ManualEdges,
        _ => return JsValue::NULL,
    };
    let Some(grading) = batch::FaceLodGrading::from_ratio(max_face_edge_ratio) else {
        return JsValue::NULL;
    };
    PATCH_LAB_SOURCE.with(|source| {
        let source = source.borrow();
        let Some(mesh) = source.as_ref() else { return JsValue::NULL };
        let lods = mesh.lods_with_grading(
            PatchLabLodConfig {
                field,
                phase: phase as f64,
                min_exp,
                max_exp,
                manual_edge_exp: [edge_a_exp, edge_b_exp, edge_c_exp],
            },
            grading,
        );
        let mut face_lods = Vec::with_capacity(lods.residents.len() * batch::FACE_LOD_STRIDE);
        let mut requested = Vec::with_capacity(lods.residents.len() * 3);
        let mut actual = Vec::with_capacity(lods.residents.len() * 3);
        for (wanted, resident) in lods.requested.iter().zip(&lods.residents) {
            requested.extend_from_slice(wanted);
            actual.extend_from_slice(&resident.edge_lods());
            face_lods.extend_from_slice(&[
                resident.canonical[0] as f32,
                resident.canonical[1] as f32,
                resident.canonical[2] as f32,
                resident.perm_index as f32,
                quilting_core::permutation::perm_sign(resident.perm_index) as f32,
                0.0,
            ]);
        }
        let histogram = lods.histogram.iter()
            .map(|(lod, count)| format!("{}/{}/{}×{}", lod[0], lod[1], lod[2], count))
            .collect::<Vec<_>>()
            .join(", ");

        let result = js_sys::Object::new();
        let js_lods = js_sys::Float32Array::new_with_length(face_lods.len() as u32);
        js_lods.copy_from(&face_lods);
        js_sys::Reflect::set(&result, &"face_lods".into(), &js_lods).unwrap();
        let js_requested = js_sys::Uint32Array::new_with_length(requested.len() as u32);
        js_requested.copy_from(&requested);
        js_sys::Reflect::set(&result, &"requested".into(), &js_requested).unwrap();
        let js_actual = js_sys::Uint32Array::new_with_length(actual.len() as u32);
        js_actual.copy_from(&actual);
        js_sys::Reflect::set(&result, &"actual".into(), &js_actual).unwrap();
        for (key, value) in [
            ("promoted_faces", lods.promoted_faces),
            ("promoted_edges", lods.promoted_edges),
            ("shared_edges", lods.shared_edges),
            ("shared_edge_mismatches", lods.shared_edge_mismatches),
            ("max_face_edge_ratio", lods.max_face_edge_ratio as usize),
            ("policy_face_edge_ratio", grading.ratio() as usize),
        ] {
            js_sys::Reflect::set(&result, &key.into(),
                &JsValue::from_f64(value as f64)).unwrap();
        }
        js_sys::Reflect::set(&result, &"histogram".into(), &histogram.into()).unwrap();
        result.into()
    })
}

/// Source mesh data for remeshing (separate from GLTF_DATA so test shapes work).
struct RemeshSource {
    positions: Vec<[f64; 3]>,
    faces: Vec<[usize; 3]>,
}

thread_local! {
    static REMESH_SOURCE: RefCell<Option<RemeshSource>> = RefCell::new(None);
    static PATCH_LAB_SOURCE: RefCell<Option<quilting_core::educational::PatchLabMesh>> = const { RefCell::new(None) };
}

/// Remesh the currently loaded glTF model into QB patches.
/// Returns a JS object with stats: { num_patches, original_faces, reduction_ratio, time_ms,
///   rms_position_error, max_position_error, rms_normal_error }.
#[wasm_bindgen]
pub fn remesh_current_model(target_patches: u32) -> JsValue {
    // Try REMESH_SOURCE first (test shapes), then GLTF_DATA
    let (positions_owned, faces_owned) = REMESH_SOURCE.with(|rs| {
        let src = rs.borrow();
        if let Some(s) = src.as_ref() {
            return Some((s.positions.clone(), s.faces.clone()));
        }
        None
    }).or_else(|| {
        GLTF_DATA.with(|gd| {
            let data = gd.borrow();
            let data = data.as_ref()?;
            let center = data.norm_center;
            let scale = data.norm_scale;
            let norm_positions: Vec<[f64; 3]> = data.combined.positions.iter().map(|v| {
                [(v[0]-center[0])*scale, (v[1]-center[1])*scale, (v[2]-center[2])*scale]
            }).collect();
            Some((norm_positions, data.combined.triangles.clone()))
        })
    }).unwrap_or_else(|| {
        web_sys::console::error_1(&"remesh_current_model: no model loaded".into());
        return (vec![], vec![]);
    });

    if faces_owned.is_empty() { return JsValue::NULL; }

    {
        let positions = &positions_owned;
        let faces = &faces_owned;

        web_sys::console::log_1(&format!(
            "remesh: starting — {} verts, {} faces, target {} patches",
            positions.len(), faces.len(), target_patches
        ).into());

        let t0 = js_sys::Date::now();

        let result = match quilting_remesh::remesh_simplified(positions, faces, target_patches as usize) {
            Ok(r) => r,
            Err(e) => {
                web_sys::console::error_1(&format!("remesh failed: {}", e).into());
                return JsValue::NULL;
            }
        };

        let elapsed = js_sys::Date::now() - t0;

        web_sys::console::log_1(&format!(
            "remesh: done in {:.0}ms — {} patches from {} faces ({:.1}x reduction)",
            elapsed, result.stats.num_patches, result.stats.original_faces, result.stats.reduction_ratio
        ).into());

        // Store the remeshed data
        REMESH_DATA.with(|rd| {
            *rd.borrow_mut() = Some(RemeshedModel {
                patches: result.patches.clone(),
                patch_uvs: result.patch_uvs.clone(),
                patch_normals: result.patch_normals.clone(),
            });
        });

        // Return stats
        let obj = js_sys::Object::new();
        let s = &result.stats;
        js_sys::Reflect::set(&obj, &"num_patches".into(), &JsValue::from_f64(s.num_patches as f64)).unwrap();
        js_sys::Reflect::set(&obj, &"original_faces".into(), &JsValue::from_f64(s.original_faces as f64)).unwrap();
        js_sys::Reflect::set(&obj, &"reduction_ratio".into(), &JsValue::from_f64(s.reduction_ratio)).unwrap();
        js_sys::Reflect::set(&obj, &"time_ms".into(), &JsValue::from_f64(elapsed)).unwrap();
        js_sys::Reflect::set(&obj, &"rms_position_error".into(), &JsValue::from_f64(s.avg_position_error)).unwrap();
        js_sys::Reflect::set(&obj, &"max_position_error".into(), &JsValue::from_f64(s.max_position_error)).unwrap();
        js_sys::Reflect::set(&obj, &"rms_normal_error".into(), &JsValue::from_f64(s.avg_normal_error_degrees)).unwrap();
        js_sys::Reflect::set(&obj, &"num_flipped".into(), &JsValue::from_f64(s.num_flipped as f64)).unwrap();
        obj.into()
    }
}

/// Compute instances from remeshed QB patches.
/// Returns the same shape as `get_rest_pose_instances`: packed f32 instance data,
/// per-face LOD data, and per-face material indices.
#[wasm_bindgen]
pub fn compute_remeshed_instances(
    lod: u32,
) -> JsValue {
    REMESH_DATA.with(|rd| {
        let data = rd.borrow();
        let data = match data.as_ref() {
            Some(d) => d,
            None => return JsValue::NULL,
        };

        // Preserve fitted patches in base space. The render shader combines
        // their source weights with the current Möbius uniforms exactly once.
        let instances = quilting_core::evaluate::compute_instances_from_patches(
            &data.patches,
            &data.patch_uvs,
            &data.patch_normals,
            &Mobius::identity(),
            lod,
        );

        let num_faces = instances.len();

        // Same compact layout as get_rest_pose_instances. Remeshed patches are
        // never GPU-skinned, so the vertex index slot stays 0.
        let mut all_instances = vec![0.0f32; num_faces * instance_layout::STRIDE];
        for (fi, inst) in instances.iter().enumerate() {
            let mut w = InstanceWriter::new(&mut all_instances, fi);
            for vi in 0..3 {
                let p = inst.positions[vi].to_point();
                w.set_position(vi, 0, [p[0] as f32, p[1] as f32, p[2] as f32]);
                let weight = inst.weights[vi];
                w.set_weight(vi, [
                    weight.w as f32,
                    weight.x as f32,
                    weight.y as f32,
                    weight.z as f32,
                ]);
                w.set_normal(vi, inst.normals[vi]);
            }
            w.set_edge_lods([
                inst.edge_lods[0] as f32,
                inst.edge_lods[1] as f32,
                inst.edge_lods[2] as f32,
            ]);
            w.set_vertex_lods([
                inst.vertex_lods[0] as f32,
                inst.vertex_lods[1] as f32,
                inst.vertex_lods[2] as f32,
            ]);
            w.set_uvs(inst.uvs);
        }

        // Face LOD data: all uniform since remeshed patches don't use adaptive LOD
        let effective_lod = if lod > 0 { lod } else { 4 };
        let canon = quilting_core::permutation::canonical_form([effective_lod, effective_lod, effective_lod]);
        let atlas_idx = ATLAS.with(|a| {
            a.borrow().as_ref().and_then(|atlas| {
                atlas.get_patch([effective_lod, effective_lod, effective_lod]).map(|_| 0u32)
            }).unwrap_or(0)
        });

        let mut face_lods = Vec::with_capacity(num_faces * 6);
        for _ in 0..num_faces {
            face_lods.push(canon.res[0] as f32);
            face_lods.push(canon.res[1] as f32);
            face_lods.push(canon.res[2] as f32);
            face_lods.push(canon.perm_index as f32);
            face_lods.push(quilting_core::permutation::perm_sign(canon.perm_index) as f32);
            face_lods.push(atlas_idx as f32);
        }

        // Build result object
        let result = js_sys::Object::new();
        let js_instances = js_sys::Float32Array::new_with_length(all_instances.len() as u32);
        js_instances.copy_from(&all_instances);
        js_sys::Reflect::set(&result, &"instances".into(), &js_instances).unwrap();
        js_sys::Reflect::set(&result, &"num_faces".into(), &JsValue::from_f64(num_faces as f64)).unwrap();

        let js_lods = js_sys::Float32Array::new_with_length(face_lods.len() as u32);
        js_lods.copy_from(&face_lods);
        js_sys::Reflect::set(&result, &"face_lods".into(), &js_lods).unwrap();

        // Face materials: all default (index 0)
        let js_materials = js_sys::Int32Array::new_with_length(num_faces as u32);
        js_sys::Reflect::set(&result, &"face_materials".into(), &js_materials).unwrap();

        result.into()
    })
}

/// Clear remeshed data and the remesh source, going back to the loaded glTF.
#[wasm_bindgen]
pub fn clear_remeshed_data() {
    REMESH_DATA.with(|rd| *rd.borrow_mut() = None);
    REMESH_SOURCE.with(|rs| *rs.borrow_mut() = None);
}

/// Helper struct to pass mesh node info to merge_all_mesh_nodes.
struct MeshNodeRef {
    node_idx: usize,
    mesh_idx: usize,
    #[allow(dead_code)]
    skin_idx: Option<usize>,
    world_transform: [f64; 16],
}
