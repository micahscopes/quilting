//! Reusable, fail-closed construction of a fitted coarse QB patch complex.
//!
//! Linear chart reduction cannot see defects introduced by the shared rational
//! QB fit. This module therefore owns the complete deterministic retry policy:
//! build a provenance-safe coarse complex, fit shared weights, score the fitted
//! surface, and force the chart owning the worst source-normal sample back to
//! exact source topology. If the same chart remains worst, every remaining
//! chart is forced exact in one explicit escalation. A failure after that is
//! returned with the complete fallback trace rather than silently accepting a
//! poor fit.

use std::collections::BTreeSet;
use std::time::{Duration, Instant};

use crate::coarse_complex::{ChartKey, CoarseComplexInput, SourceFaceId};
use crate::coarse_patch_complex::{
    build_coarse_patch_complex_with_fallbacks, CoarsePatchComplex, CoarsePatchConfig,
    CoarsePatchError,
};
use crate::conformal_optimizer::{
    score_patch_complex_weighted, ConformalProbe, FitScore, FitScoreConfig, ObjectiveWeights,
    OptimizerError,
};
use crate::linear_fit::{LinearFitConfig, LinearFitError, LinearFitResult};

/// Complete policy for [`fit_coarse_patch_complex_with_backoff`].
#[derive(Clone, Copy, Debug)]
pub struct FittedCoarsePatchConfig {
    pub patch: CoarsePatchConfig,
    pub fit: LinearFitConfig,
    pub score: FitScoreConfig,
    pub objective_weights: ObjectiveWeights,
    /// Final fitted source-normal limit. This is intentionally independent of
    /// the linear reduction's sampled-normal gate.
    pub maximum_fitted_normal_deviation_degrees: f64,
}

impl Default for FittedCoarsePatchConfig {
    fn default() -> Self {
        Self {
            patch: CoarsePatchConfig::default(),
            fit: LinearFitConfig::default(),
            score: FitScoreConfig::default(),
            objective_weights: ObjectiveWeights::default(),
            maximum_fitted_normal_deviation_degrees: 60.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FittedQualityBackoffAction {
    ForceChartExact,
    ForceAllRemainingExact,
}

impl std::fmt::Display for FittedQualityBackoffAction {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::ForceChartExact => "force-chart-exact",
            Self::ForceAllRemainingExact => "force-all-remaining-exact",
        })
    }
}

/// One deterministic fitted-quality escalation.
#[derive(Clone, Debug, PartialEq)]
pub struct FittedQualityFallback {
    /// One-based fit attempt that produced this fallback.
    pub attempt: usize,
    pub action: FittedQualityBackoffAction,
    pub chart: usize,
    pub key: ChartKey,
    pub source_face: SourceFaceId,
    pub source_sample_ordinal: u32,
    pub measured_degrees: f64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FittedCoarsePatchTimings {
    pub build: Duration,
    pub fit: Duration,
    pub context: Duration,
    pub score: Duration,
}

/// Accepted fitted complex plus cumulative retry evidence.
#[derive(Debug)]
pub struct FittedCoarsePatchResult {
    pub complex: CoarsePatchComplex,
    pub fit: LinearFitResult,
    pub score: FitScore,
    pub objective: f64,
    pub attempts: usize,
    pub total_backend_attempts: usize,
    pub total_rejected_backend_attempts: usize,
    pub fitted_quality_fallbacks: Vec<FittedQualityFallback>,
    pub timings: FittedCoarsePatchTimings,
}

#[derive(Clone, Debug, PartialEq)]
pub enum FittedCoarsePatchError {
    InvalidMaximumFittedNormalDeviation(f64),
    Build(CoarsePatchError),
    Fit(LinearFitError),
    ScoreContext(OptimizerError),
    Score(OptimizerError),
    CounterOverflow(&'static str),
    NonFiniteObjective {
        source_non_finite: usize,
        source_degenerate_normals: usize,
        source_near_singular: usize,
        source_invalid_patches: usize,
        probe_pole_near: usize,
    },
    MissingMaximumNormalSample {
        measured_degrees: f64,
    },
    MissingCorrespondenceSample {
        sample: usize,
    },
    MissingWorstChart {
        chart: usize,
        chart_count: usize,
    },
    ChartSetChanged {
        expected: Vec<ChartKey>,
        actual: Vec<ChartKey>,
    },
    AllExactSourceQualityExceeded {
        chart: usize,
        key: ChartKey,
        source_face: SourceFaceId,
        source_sample_ordinal: u32,
        measured_degrees: f64,
        fallbacks: Vec<FittedQualityFallback>,
    },
}

impl std::fmt::Display for FittedCoarsePatchError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidMaximumFittedNormalDeviation(value) => write!(
                formatter,
                "maximum fitted normal deviation {value} is not in [0, 90] degrees",
            ),
            Self::Build(error) => write!(formatter, "coarse-complex: {error}"),
            Self::Fit(error) => write!(formatter, "shared-fit: {error}"),
            Self::ScoreContext(error) => write!(formatter, "score-context: {error}"),
            Self::Score(error) => write!(formatter, "score: {error}"),
            Self::CounterOverflow(counter) => write!(formatter, "{counter} overflowed"),
            Self::NonFiniteObjective {
                source_non_finite,
                source_degenerate_normals,
                source_near_singular,
                source_invalid_patches,
                probe_pole_near,
            } => write!(
                formatter,
                "score-gate: non-finite objective (source_non_finite={source_non_finite} source_degenerate_normals={source_degenerate_normals} source_near_singular={source_near_singular} source_invalid_patches={source_invalid_patches} probe_pole_near={probe_pole_near})",
            ),
            Self::MissingMaximumNormalSample { measured_degrees } => write!(
                formatter,
                "fit-quality gate measured {measured_degrees:.3}° without a worst source sample",
            ),
            Self::MissingCorrespondenceSample { sample } => write!(
                formatter,
                "fit-quality gate references missing correspondence sample {sample}",
            ),
            Self::MissingWorstChart { chart, chart_count } => write!(
                formatter,
                "fit-quality gate references chart {chart}, but only {chart_count} charts exist",
            ),
            Self::ChartSetChanged { .. } => write!(
                formatter,
                "coarse chart identities changed across fitted-quality retries",
            ),
            Self::AllExactSourceQualityExceeded {
                chart,
                source_face,
                source_sample_ordinal,
                measured_degrees,
                ..
            } => write!(
                formatter,
                "fit-quality gate still measures {measured_degrees:.3}° at chart {chart} source face {source_face:?} sample {source_sample_ordinal} with every chart on exact source topology",
            ),
        }
    }
}

impl std::error::Error for FittedCoarsePatchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Build(error) => Some(error),
            Self::Fit(error) => Some(error),
            Self::ScoreContext(error) | Self::Score(error) => Some(error),
            _ => None,
        }
    }
}

struct FittedQualityWorst {
    chart: usize,
    source_face: SourceFaceId,
    source_sample_ordinal: u32,
    measured_degrees: f64,
}

struct RetryAttempt<T> {
    value: T,
    chart_keys: Vec<ChartKey>,
    worst: Option<FittedQualityWorst>,
}

#[derive(Debug)]
struct RetryResult<T> {
    value: T,
    attempts: usize,
    fallbacks: Vec<FittedQualityFallback>,
}

fn retry_fitted_quality<T>(
    mut evaluate: impl FnMut(&BTreeSet<ChartKey>) -> Result<RetryAttempt<T>, FittedCoarsePatchError>,
) -> Result<RetryResult<T>, FittedCoarsePatchError> {
    let mut source_fallbacks = BTreeSet::new();
    let mut fallbacks = Vec::new();
    let mut attempts = 0usize;
    let mut expected_chart_keys: Option<Vec<ChartKey>> = None;
    loop {
        attempts = attempts
            .checked_add(1)
            .ok_or(FittedCoarsePatchError::CounterOverflow(
                "fitted-quality attempt count",
            ))?;
        let attempt = evaluate(&source_fallbacks)?;
        if let Some(expected) = &expected_chart_keys {
            if expected != &attempt.chart_keys {
                return Err(FittedCoarsePatchError::ChartSetChanged {
                    expected: expected.clone(),
                    actual: attempt.chart_keys,
                });
            }
        } else {
            expected_chart_keys = Some(attempt.chart_keys.clone());
        }
        let Some(worst) = attempt.worst else {
            return Ok(RetryResult {
                value: attempt.value,
                attempts,
                fallbacks,
            });
        };
        let key = attempt.chart_keys.get(worst.chart).cloned().ok_or(
            FittedCoarsePatchError::MissingWorstChart {
                chart: worst.chart,
                chart_count: attempt.chart_keys.len(),
            },
        )?;
        let action = if source_fallbacks.insert(key.clone()) {
            FittedQualityBackoffAction::ForceChartExact
        } else {
            let previous_count = source_fallbacks.len();
            source_fallbacks.extend(attempt.chart_keys.iter().cloned());
            if source_fallbacks.len() == previous_count {
                return Err(FittedCoarsePatchError::AllExactSourceQualityExceeded {
                    chart: worst.chart,
                    key,
                    source_face: worst.source_face,
                    source_sample_ordinal: worst.source_sample_ordinal,
                    measured_degrees: worst.measured_degrees,
                    fallbacks,
                });
            }
            FittedQualityBackoffAction::ForceAllRemainingExact
        };
        fallbacks.push(FittedQualityFallback {
            attempt: attempts,
            action,
            chart: worst.chart,
            key,
            source_face: worst.source_face,
            source_sample_ordinal: worst.source_sample_ordinal,
            measured_degrees: worst.measured_degrees,
        });
    }
}

struct PipelineAttempt {
    complex: CoarsePatchComplex,
    fit: LinearFitResult,
    score: FitScore,
    objective: f64,
}

/// Build, fit, and score a coarse QB complex, forcing unsafe fitted charts back
/// to exact source topology until the configured source-normal gate passes.
pub fn fit_coarse_patch_complex_with_backoff(
    input: &CoarseComplexInput<'_>,
    config: &FittedCoarsePatchConfig,
    probes: &[ConformalProbe],
) -> Result<FittedCoarsePatchResult, FittedCoarsePatchError> {
    if !config.maximum_fitted_normal_deviation_degrees.is_finite()
        || !(0.0..=90.0).contains(&config.maximum_fitted_normal_deviation_degrees)
    {
        return Err(FittedCoarsePatchError::InvalidMaximumFittedNormalDeviation(
            config.maximum_fitted_normal_deviation_degrees,
        ));
    }

    let mut timings = FittedCoarsePatchTimings::default();
    let mut total_backend_attempts = 0usize;
    let mut total_rejected_backend_attempts = 0usize;
    let retried = retry_fitted_quality(|source_fallbacks| {
        let build_start = Instant::now();
        let complex =
            build_coarse_patch_complex_with_fallbacks(input, &config.patch, source_fallbacks)
                .map_err(FittedCoarsePatchError::Build)?;
        timings.build = timings.build.saturating_add(build_start.elapsed());

        let build_backend_attempts = complex
            .charts
            .iter()
            .try_fold(0usize, |total, chart| {
                total.checked_add(chart.backend_attempts)
            })
            .ok_or(FittedCoarsePatchError::CounterOverflow(
                "per-build backend-attempt count",
            ))?;
        let build_rejected_backend_attempts = complex
            .charts
            .iter()
            .try_fold(0usize, |total, chart| {
                total.checked_add(chart.rejected_candidates.len())
            })
            .ok_or(FittedCoarsePatchError::CounterOverflow(
                "per-build backend-rejection count",
            ))?;
        total_backend_attempts = total_backend_attempts
            .checked_add(build_backend_attempts)
            .ok_or(FittedCoarsePatchError::CounterOverflow(
                "cumulative backend-attempt count",
            ))?;
        total_rejected_backend_attempts = total_rejected_backend_attempts
            .checked_add(build_rejected_backend_attempts)
            .ok_or(FittedCoarsePatchError::CounterOverflow(
                "cumulative backend-rejection count",
            ))?;

        let fit_start = Instant::now();
        let fit = complex
            .fit_shared_qb(&config.fit)
            .map_err(FittedCoarsePatchError::Fit)?;
        timings.fit = timings.fit.saturating_add(fit_start.elapsed());

        let context_start = Instant::now();
        let context = complex
            .fit_score_context(probes)
            .map_err(FittedCoarsePatchError::ScoreContext)?;
        timings.context = timings.context.saturating_add(context_start.elapsed());

        let score_start = Instant::now();
        let score = score_patch_complex_weighted(
            &fit.patches,
            &complex.triangles(),
            &complex.weighted_score_samples(),
            &context,
            &config.score,
        )
        .map_err(FittedCoarsePatchError::Score)?;
        timings.score = timings.score.saturating_add(score_start.elapsed());
        let objective = score.scalar_objective(&config.objective_weights);
        if !objective.is_finite() {
            return Err(FittedCoarsePatchError::NonFiniteObjective {
                source_non_finite: score.source.non_finite_samples,
                source_degenerate_normals: score.source.degenerate_normal_samples,
                source_near_singular: score.weights.near_singular_patches,
                source_invalid_patches: score.weights.invalid_patches,
                probe_pole_near: score
                    .conformal_probes
                    .iter()
                    .map(|probe| probe.pole_near_samples)
                    .sum(),
            });
        }

        let chart_keys = complex
            .charts
            .iter()
            .map(|chart| chart.key.clone())
            .collect::<Vec<_>>();
        let worst =
            if score.source.normal_max_degrees <= config.maximum_fitted_normal_deviation_degrees {
                None
            } else {
                let sample_index = score.source.normal_max_sample.ok_or(
                    FittedCoarsePatchError::MissingMaximumNormalSample {
                        measured_degrees: score.source.normal_max_degrees,
                    },
                )?;
                let sample = complex.correspondence.get(sample_index).ok_or(
                    FittedCoarsePatchError::MissingCorrespondenceSample {
                        sample: sample_index,
                    },
                )?;
                Some(FittedQualityWorst {
                    chart: sample.coarse_face_key.chart,
                    source_face: sample.key.face,
                    source_sample_ordinal: sample.key.ordinal,
                    measured_degrees: score.source.normal_max_degrees,
                })
            };
        Ok(RetryAttempt {
            value: PipelineAttempt {
                complex,
                fit,
                score,
                objective,
            },
            chart_keys,
            worst,
        })
    })?;
    Ok(FittedCoarsePatchResult {
        complex: retried.value.complex,
        fit: retried.value.fit,
        score: retried.value.score,
        objective: retried.value.objective,
        attempts: retried.attempts,
        total_backend_attempts,
        total_rejected_backend_attempts,
        fitted_quality_fallbacks: retried.fallbacks,
        timings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coarse_complex::{SourceFaceId, SourceVertexId};

    fn chart(domain: u32, face: u64) -> ChartKey {
        ChartKey {
            domain,
            source_faces: vec![SourceFaceId(face)],
        }
    }

    #[test]
    fn retry_loop_records_new_repeated_force_all_and_all_exact_failure() {
        let keys = vec![chart(0, 10), chart(0, 20), chart(0, 30)];
        let mut observed = Vec::new();
        let mut scripted = [0usize, 1, 1, 2].into_iter();
        let error = retry_fitted_quality(|fallbacks| {
            observed.push(fallbacks.clone());
            let chart = scripted
                .next()
                .expect("the retry loop made an extra attempt");
            Ok(RetryAttempt {
                value: (),
                chart_keys: keys.clone(),
                worst: Some(FittedQualityWorst {
                    chart,
                    source_face: SourceFaceId(100 + chart as u64),
                    source_sample_ordinal: chart as u32,
                    measured_degrees: 70.0 + chart as f64,
                }),
            })
        })
        .unwrap_err();

        assert_eq!(
            observed,
            vec![
                BTreeSet::new(),
                BTreeSet::from([keys[0].clone()]),
                BTreeSet::from([keys[0].clone(), keys[1].clone()]),
                BTreeSet::from_iter(keys.iter().cloned()),
            ]
        );
        let FittedCoarsePatchError::AllExactSourceQualityExceeded {
            chart,
            key,
            source_face,
            source_sample_ordinal,
            measured_degrees,
            fallbacks,
        } = error
        else {
            panic!("unexpected retry error: {error}");
        };
        assert_eq!(chart, 2);
        assert_eq!(key, keys[2]);
        assert_eq!(source_face, SourceFaceId(102));
        assert_eq!(source_sample_ordinal, 2);
        assert_eq!(measured_degrees, 72.0);
        assert_eq!(fallbacks.len(), 3);
        assert_eq!(fallbacks[0].attempt, 1);
        assert_eq!(
            fallbacks[0].action,
            FittedQualityBackoffAction::ForceChartExact
        );
        assert_eq!(fallbacks[0].key, keys[0]);
        assert_eq!(fallbacks[1].attempt, 2);
        assert_eq!(
            fallbacks[1].action,
            FittedQualityBackoffAction::ForceChartExact
        );
        assert_eq!(fallbacks[1].key, keys[1]);
        assert_eq!(fallbacks[2].attempt, 3);
        assert_eq!(
            fallbacks[2].action,
            FittedQualityBackoffAction::ForceAllRemainingExact
        );
        assert_eq!(fallbacks[2].key, keys[1]);
    }

    #[test]
    fn retry_loop_rejects_changing_chart_identity() {
        let keys = vec![chart(0, 10), chart(0, 20)];
        let mut attempt = 0usize;
        let error = retry_fitted_quality(|_| {
            attempt += 1;
            Ok(RetryAttempt {
                value: (),
                chart_keys: if attempt == 1 {
                    keys.clone()
                } else {
                    vec![keys[1].clone(), keys[0].clone()]
                },
                worst: (attempt == 1).then_some(FittedQualityWorst {
                    chart: 0,
                    source_face: SourceFaceId(10),
                    source_sample_ordinal: 0,
                    measured_degrees: 70.0,
                }),
            })
        })
        .unwrap_err();

        assert_eq!(
            error,
            FittedCoarsePatchError::ChartSetChanged {
                expected: keys.clone(),
                actual: vec![keys[1].clone(), keys[0].clone()],
            }
        );
    }

    #[test]
    fn planar_grid_passes_the_reusable_pipeline_without_a_fallback() {
        let side = 5usize;
        let positions = (0..side)
            .flat_map(|y| {
                (0..side).map(move |x| {
                    [
                        x as f64 / (side - 1) as f64,
                        y as f64 / (side - 1) as f64,
                        0.0,
                    ]
                })
            })
            .collect::<Vec<_>>();
        let mut triangles = Vec::new();
        for y in 0..side - 1 {
            for x in 0..side - 1 {
                let a = y * side + x;
                let b = a + 1;
                let d = (y + 1) * side + x;
                let c = d + 1;
                triangles.push([a, b, c]);
                triangles.push([a, c, d]);
            }
        }
        let vertex_ids = (0..positions.len())
            .map(|index| SourceVertexId(index as u64))
            .collect::<Vec<_>>();
        let face_ids = (0..triangles.len())
            .map(|index| SourceFaceId(index as u64))
            .collect::<Vec<_>>();
        let domains = vec![0; triangles.len()];
        let result = fit_coarse_patch_complex_with_backoff(
            &CoarseComplexInput {
                positions: &positions,
                triangles: &triangles,
                source_vertex_ids: &vertex_ids,
                source_face_ids: &face_ids,
                face_domains: &domains,
                locked_edges: &[],
            },
            &FittedCoarsePatchConfig::default(),
            &[],
        )
        .unwrap();

        assert_eq!(result.attempts, 1);
        assert!(result.fitted_quality_fallbacks.is_empty());
        assert!(result.objective.is_finite());
        assert!(result.score.source.normal_max_degrees < 1.0e-8);
        assert_eq!(result.fit.patches.len(), result.complex.faces.len());
    }

    #[test]
    fn invalid_fitted_quality_limit_fails_before_building() {
        let config = FittedCoarsePatchConfig {
            maximum_fitted_normal_deviation_degrees: 90.1,
            ..FittedCoarsePatchConfig::default()
        };
        let error = fit_coarse_patch_complex_with_backoff(
            &CoarseComplexInput {
                positions: &[],
                triangles: &[],
                source_vertex_ids: &[],
                source_face_ids: &[],
                face_domains: &[],
                locked_edges: &[],
            },
            &config,
            &[],
        )
        .unwrap_err();
        assert_eq!(
            error,
            FittedCoarsePatchError::InvalidMaximumFittedNormalDeviation(90.1)
        );
    }
}
