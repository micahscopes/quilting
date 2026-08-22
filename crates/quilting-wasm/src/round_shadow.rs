//! Opt-in correctness observer for conservative round-side patch queries.
//!
//! This module never owns visibility or mutates draw membership. It builds a
//! rest-pose hierarchy from the renderer's immutable source records and
//! compares candidates with a completed authoritative GPU classification.

use quilting_core::conformal::{ConformalGenerator, ConformalTransformChain};
use quilting_core::instance_layout;
use quilting_core::Quat;
use quilting_round_index::{
    PatchControl, PatchIndexBuildReport, PredicateConfig, RoundQuery, StaticPatchIndex, TopologyKey,
};
use serde::Serialize;
use wasm_bindgen::JsValue;

#[derive(Default)]
pub(crate) struct RoundShadowObserver {
    enabled: bool,
    topology_revision: u64,
    index: Option<StaticPatchIndex>,
    patches: Vec<PatchControl>,
    build: BuildSnapshot,
    comparisons: u64,
    last: Option<ComparisonSnapshot>,
    authoritative_complete: bool,
}

#[derive(Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
struct BuildSnapshot {
    status: String,
    reason: Option<String>,
    topology_revision: u64,
    build_ms: f64,
    patches: usize,
    bounded_patches: usize,
    always_candidate_patches: usize,
    hierarchy_nodes: usize,
}

#[derive(Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
struct ComparisonSnapshot {
    status: String,
    reason: Option<String>,
    transform_kind: String,
    animated: bool,
    authored_scene: bool,
    query_ms: f64,
    faces: usize,
    candidate_faces: usize,
    gpu_survivor_faces: usize,
    gpu_only_survivor_faces: usize,
    index_only_candidate_faces: usize,
    false_negative_faces: usize,
    false_negative_examples: Vec<u32>,
    visited_nodes: usize,
    pruned_nodes: usize,
    always_candidate_faces: usize,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ObserverSnapshot {
    enabled: bool,
    comparisons: u64,
    build: BuildSnapshot,
    last: Option<ComparisonSnapshot>,
}

impl RoundShadowObserver {
    pub(crate) fn set_enabled(
        &mut self,
        enabled: bool,
        instances: &[f32],
        num_faces: usize,
    ) -> JsValue {
        self.enabled = enabled;
        self.rebuild(instances, num_faces);
        self.to_js()
    }

    pub(crate) fn rebuild(&mut self, instances: &[f32], num_faces: usize) {
        self.index = None;
        self.patches.clear();
        self.last = None;
        self.comparisons = 0;
        self.authoritative_complete = false;
        self.topology_revision = self.topology_revision.saturating_add(1);
        let revision = self.topology_revision;

        if !self.enabled {
            self.build = BuildSnapshot {
                status: "disabled".to_string(),
                topology_revision: revision,
                ..BuildSnapshot::default()
            };
            return;
        }
        if num_faces == 0 || instances.is_empty() {
            self.build = BuildSnapshot {
                status: "waiting".to_string(),
                reason: Some("no source patch buffer has been uploaded".to_string()),
                topology_revision: revision,
                ..BuildSnapshot::default()
            };
            return;
        }

        let started = browser_now_ms();
        let built = patch_controls_from_instances(instances, num_faces).and_then(|patches| {
            let index = StaticPatchIndex::build(
                TopologyKey {
                    asset_revision: revision,
                    topology_revision: revision,
                },
                &patches,
                PredicateConfig::default(),
            )
            .map_err(|error| error.to_string())?;
            Ok((index, patches))
        });
        let build_ms = browser_now_ms() - started;
        match built {
            Ok((index, patch_controls)) => {
                let PatchIndexBuildReport {
                    patches,
                    bounded_patches,
                    always_candidate_patches,
                    hierarchy_nodes,
                } = index.report();
                self.build = BuildSnapshot {
                    status: "ready".to_string(),
                    reason: None,
                    topology_revision: revision,
                    build_ms,
                    patches,
                    bounded_patches,
                    always_candidate_patches,
                    hierarchy_nodes,
                };
                self.index = Some(index);
                self.patches = patch_controls;
            }
            Err(reason) => {
                self.build = BuildSnapshot {
                    status: "error".to_string(),
                    reason: Some(reason),
                    topology_revision: revision,
                    build_ms,
                    patches: num_faces,
                    ..BuildSnapshot::default()
                };
            }
        }
    }

    pub(crate) fn to_js(&self) -> JsValue {
        serde_wasm_bindgen::to_value(&ObserverSnapshot {
            enabled: self.enabled,
            comparisons: self.comparisons,
            build: self.build.clone(),
            last: self.last.clone(),
        })
        .unwrap_or(JsValue::NULL)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn compare(
        &mut self,
        view_projection: &[f32],
        transform_kind: &str,
        transform_center: &[f32],
        transform_radius: f32,
        animated: bool,
        authored_scene: bool,
        authoritative_full_snapshot: bool,
        classified_visibility: &[bool],
    ) -> JsValue {
        self.authoritative_complete |= authoritative_full_snapshot;
        if !self.enabled {
            return self.skip(
                "disabled",
                "round shadow is not enabled",
                transform_kind,
                animated,
                authored_scene,
                classified_visibility,
            );
        }
        if animated {
            return self.skip(
                "unsupported",
                "active animation has no conservative pose envelope yet",
                transform_kind,
                animated,
                authored_scene,
                classified_visibility,
            );
        }
        if authored_scene {
            return self.skip(
                "unsupported",
                "authored per-node conformal or Euclidean transforms require structured frame chains",
                transform_kind,
                animated,
                authored_scene,
                classified_visibility,
            );
        }
        if !self.authoritative_complete {
            return self.skip(
                "waiting",
                "no complete authoritative visibility snapshot is available",
                transform_kind,
                animated,
                authored_scene,
                classified_visibility,
            );
        }
        if classified_visibility.len() != self.build.patches {
            return self.skip(
                "error",
                "authoritative visibility does not match the indexed topology",
                transform_kind,
                animated,
                authored_scene,
                classified_visibility,
            );
        }
        if self.index.is_none() {
            let reason = self
                .build
                .reason
                .clone()
                .unwrap_or_else(|| "round patch hierarchy is unavailable".to_string());
            return self.skip_owned(
                "unavailable",
                reason,
                transform_kind,
                animated,
                authored_scene,
                classified_visibility,
            );
        }
        if view_projection.len() < 16 {
            return self.skip(
                "error",
                "view-projection matrix has fewer than 16 values",
                transform_kind,
                animated,
                authored_scene,
                classified_visibility,
            );
        }

        let transform =
            match structured_transform(transform_kind, transform_center, transform_radius) {
                Ok(transform) => transform,
                Err((status, reason)) => {
                    return self.skip_owned(
                        status,
                        reason,
                        transform_kind,
                        animated,
                        authored_scene,
                        classified_visibility,
                    );
                }
            };
        let mut matrix = [0.0_f64; 16];
        for (destination, source) in matrix.iter_mut().zip(view_projection.iter().take(16)) {
            *destination = f64::from(*source);
        }
        let query = match RoundQuery::from_view_projection(&matrix) {
            Ok(query) => query,
            Err(error) => {
                return self.skip_owned(
                    "error",
                    error.to_string(),
                    transform_kind,
                    animated,
                    authored_scene,
                    classified_visibility,
                );
            }
        };

        let started = browser_now_ms();
        let query_result = match self
            .index
            .as_ref()
            .expect("checked round shadow index")
            .query_output_chart(&query, &transform)
        {
            Ok(result) => result,
            Err(error) => {
                return self.skip_owned(
                    "error",
                    error.to_string(),
                    transform_kind,
                    animated,
                    authored_scene,
                    classified_visibility,
                );
            }
        };
        let query_ms = browser_now_ms() - started;
        let mut candidate_mask = vec![false; classified_visibility.len()];
        for &face in &query_result.candidate_faces {
            if let Some(candidate) = candidate_mask.get_mut(face as usize) {
                *candidate = true;
            }
        }
        let mut gpu_survivor_faces = 0;
        let mut gpu_only_survivor_faces = 0;
        let mut index_only_candidate_faces = 0;
        let mut false_negative_faces = 0;
        let mut false_negative_examples = Vec::new();
        for (face, (&visible, &candidate)) in classified_visibility
            .iter()
            .zip(&candidate_mask)
            .enumerate()
        {
            gpu_survivor_faces += usize::from(visible);
            gpu_only_survivor_faces += usize::from(visible && !candidate);
            index_only_candidate_faces += usize::from(candidate && !visible);
            if visible && !candidate {
                let patch = &self.patches[face];
                if patch_has_strictly_visible_sample(patch, &transform, &matrix) {
                    false_negative_faces += 1;
                    if false_negative_examples.len() < 16 {
                        false_negative_examples.push(face as u32);
                    }
                }
            }
        }

        self.finish(ComparisonSnapshot {
            status: "ok".to_string(),
            reason: None,
            transform_kind: transform_kind.to_string(),
            animated,
            authored_scene,
            query_ms,
            faces: classified_visibility.len(),
            candidate_faces: query_result.candidate_faces.len(),
            gpu_survivor_faces,
            gpu_only_survivor_faces,
            index_only_candidate_faces,
            false_negative_faces,
            false_negative_examples,
            visited_nodes: query_result.visited_nodes,
            pruned_nodes: query_result.pruned_nodes,
            always_candidate_faces: query_result.always_candidate_faces,
        })
    }

    fn skip(
        &mut self,
        status: &str,
        reason: &str,
        transform_kind: &str,
        animated: bool,
        authored_scene: bool,
        classified_visibility: &[bool],
    ) -> JsValue {
        self.skip_owned(
            status,
            reason.to_string(),
            transform_kind,
            animated,
            authored_scene,
            classified_visibility,
        )
    }

    fn skip_owned(
        &mut self,
        status: &str,
        reason: String,
        transform_kind: &str,
        animated: bool,
        authored_scene: bool,
        classified_visibility: &[bool],
    ) -> JsValue {
        self.finish(ComparisonSnapshot {
            status: status.to_string(),
            reason: Some(reason),
            transform_kind: transform_kind.to_string(),
            animated,
            authored_scene,
            faces: classified_visibility.len(),
            gpu_survivor_faces: classified_visibility
                .iter()
                .filter(|&&visible| visible)
                .count(),
            ..ComparisonSnapshot::default()
        })
    }

    fn finish(&mut self, comparison: ComparisonSnapshot) -> JsValue {
        self.comparisons = self.comparisons.saturating_add(1);
        self.last = Some(comparison);
        self.to_js()
    }
}

fn structured_transform(
    transform_kind: &str,
    transform_center: &[f32],
    transform_radius: f32,
) -> Result<ConformalTransformChain, (&'static str, String)> {
    match transform_kind {
        "identity" => Ok(ConformalTransformChain::identity()),
        "sphere_reflection" if transform_center.len() >= 3 => {
            ConformalTransformChain::new(vec![ConformalGenerator::sphere_reflection(
                [
                    f64::from(transform_center[0]),
                    f64::from(transform_center[1]),
                    f64::from(transform_center[2]),
                ],
                f64::from(transform_radius),
            )])
            .map_err(|error| ("error", error.to_string()))
        }
        "sphere_reflection" => Err((
            "error",
            "sphere-reflection center has fewer than three values".to_string(),
        )),
        _ => Err((
            "unsupported",
            format!("transform kind '{transform_kind}' has no structured round-side adapter"),
        )),
    }
}

/// Independent red-alert check for hierarchy rejections. The authoritative
/// GPU bit means "not proved outside," so disagreement with it measures
/// relative conservatism rather than a geometric false negative. A rejected
/// patch is only called a false negative here when an evaluated QB sample lies
/// strictly inside all six clip inequalities.
fn patch_has_strictly_visible_sample(
    patch: &PatchControl,
    transform: &ConformalTransformChain,
    view_projection: &[f64; 16],
) -> bool {
    const SAMPLES: [[f64; 3]; 7] = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [1.0 / 3.0; 3],
    ];
    SAMPLES.into_iter().any(|barycentric| {
        evaluate_source_patch(patch, barycentric)
            .and_then(|point| transform.apply_point(point).ok())
            .is_some_and(|point| strictly_inside_clip(point, view_projection))
    })
}

fn evaluate_source_patch(patch: &PatchControl, barycentric: [f64; 3]) -> Option<[f64; 3]> {
    let mut numerator = Quat::ZERO;
    let mut denominator = Quat::ZERO;
    for (corner, coefficient) in barycentric.into_iter().enumerate() {
        let [x, y, z] = patch.positions[corner];
        let [w, i, j, k] = patch.weights[corner];
        let weight = Quat::new(w, i, j, k);
        numerator = numerator + (Quat::from_point(x, y, z) * weight) * coefficient;
        denominator = denominator + weight * coefficient;
    }
    if denominator.norm_sq() <= quilting_core::quaternion::SINGULARITY_NORM_SQ {
        return None;
    }
    let point = (numerator * denominator.inv()).to_point();
    point.into_iter().all(f64::is_finite).then_some(point)
}

fn strictly_inside_clip(point: [f64; 3], matrix: &[f64; 16]) -> bool {
    let [x, y, z] = point;
    let clip = [
        matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12],
        matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13],
        matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14],
        matrix[3] * x + matrix[7] * y + matrix[11] * z + matrix[15],
    ];
    if !clip.into_iter().all(f64::is_finite) {
        return false;
    }
    let margin = 1.0e-6 * clip[3].abs().max(1.0);
    clip[3] > margin
        && clip[0].abs() < clip[3] - margin
        && clip[1].abs() < clip[3] - margin
        && clip[2].abs() < clip[3] - margin
}

fn patch_controls_from_instances(
    instances: &[f32],
    num_faces: usize,
) -> Result<Vec<PatchControl>, String> {
    let required = num_faces
        .checked_mul(instance_layout::STRIDE)
        .ok_or_else(|| "source patch buffer size overflow".to_string())?;
    if instances.len() < required {
        return Err(format!(
            "source patch buffer has {} floats; expected at least {required}",
            instances.len(),
        ));
    }

    let mut patches = Vec::with_capacity(num_faces);
    for face in 0..num_faces {
        let record =
            &instances[face * instance_layout::STRIDE..(face + 1) * instance_layout::STRIDE];
        let positions = std::array::from_fn(|corner| {
            let offset = instance_layout::offset::POSITIONS + corner * 4;
            [
                f64::from(record[offset + 1]),
                f64::from(record[offset + 2]),
                f64::from(record[offset + 3]),
            ]
        });
        let weights = std::array::from_fn(|corner| {
            let offset = instance_layout::offset::WEIGHTS + corner * 4;
            [
                f64::from(record[offset]),
                f64::from(record[offset + 1]),
                f64::from(record[offset + 2]),
                f64::from(record[offset + 3]),
            ]
        });
        patches.push(PatchControl {
            face: u32::try_from(face)
                .map_err(|_| "source patch face index exceeds u32".to_string())?,
            positions,
            weights,
        });
    }
    Ok(patches)
}

fn browser_now_ms() -> f64 {
    web_sys::window()
        .and_then(|window| window.performance())
        .map_or(0.0, |performance| performance.now())
}
