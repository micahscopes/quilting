//! Deterministic, fail-closed report for the provenance-safe QB candidate path.
//!
//! Unlike `remesh_glb`, this harness visits every extracted triangle primitive
//! in a static active glTF scene, applies its node's world transform, and keeps
//! each primitive as an independent ownership domain. Boundary evidence is
//! therefore intra-primitive; the harness never position-welds authored seams.
//! It is evidence tooling: failures remain in the report and produce a failing
//! exit status instead of silently falling back to the legacy remesher.
//!
//! ```text
//! cargo run -p quilting-remesh --release --features meshopt-prototype \
//!   --example coarse_qb_report -- local-glbs/classic_chessboard.glb
//! ```

use std::collections::BTreeSet;
use std::env;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use quilting_core::quaternion::{Mobius, Quat};
use quilting_remesh::coarse_complex::{
    split_disconnected_vertex_fans, CoarseComplexInput, SourceFaceId, SourceVertexId,
};
use quilting_remesh::coarse_patch_complex::{
    ChartReductionReport, CoarsePatchComplex, CoarsePatchConfig, CorrespondenceSample,
};
use quilting_remesh::coarse_reduction::CoarseReductionConfig;
use quilting_remesh::conformal_optimizer::{
    ConformalProbe, FitScore, FitScoreConfig, ObjectiveWeights,
};
use quilting_remesh::fitted_coarse_patch::{
    fit_coarse_patch_complex_with_backoff, FittedCoarsePatchConfig, FittedCoarsePatchResult,
    FittedQualityFallback,
};
use quilting_remesh::geometry;
use quilting_remesh::linear_fit::{LinearFitConfig, LinearFitResult};

const IDENTITY: [f64; 16] = [
    1.0, 0.0, 0.0, 0.0, //
    0.0, 1.0, 0.0, 0.0, //
    0.0, 0.0, 1.0, 0.0, //
    0.0, 0.0, 0.0, 1.0,
];

#[derive(Debug)]
struct Config {
    path: PathBuf,
    target_ratio: f64,
    target_error: f32,
    maximum_normal_deviation_degrees: f64,
    subdivisions: usize,
    maximum_samples: usize,
    maximum_quality_candidate_tests: usize,
    maximum_candidate_tests: usize,
    probe_sphere: Option<([f64; 3], f64)>,
    only_part: Option<usize>,
    list_only: bool,
}

impl Config {
    fn parse() -> Result<Self, String> {
        let mut args = env::args().skip(1);
        let path = args.next().ok_or_else(usage)?;
        if path == "-h" || path == "--help" {
            return Err(usage());
        }
        let mut config = Self {
            path: PathBuf::from(path),
            target_ratio: 0.25,
            target_error: 0.01,
            maximum_normal_deviation_degrees: 60.0,
            subdivisions: 2,
            maximum_samples: 1_000_000,
            maximum_quality_candidate_tests: 50_000_000,
            maximum_candidate_tests: 50_000_000,
            probe_sphere: None,
            only_part: None,
            list_only: false,
        };
        while let Some(flag) = args.next() {
            match flag.as_str() {
                "--ratio" => {
                    config.target_ratio = next_value(&mut args, &flag)?.parse().map_err(|_| {
                        "--ratio must be a floating-point value in (0, 1]".to_string()
                    })?;
                }
                "--error" => {
                    config.target_error = next_value(&mut args, &flag)?.parse().map_err(|_| {
                        "--error must be a floating-point value in [0, 1]".to_string()
                    })?;
                }
                "--max-normal-deviation" => {
                    config.maximum_normal_deviation_degrees =
                        next_value(&mut args, &flag)?.parse().map_err(|_| {
                            "--max-normal-deviation must be a floating-point value in [0, 90]"
                                .to_string()
                        })?;
                }
                "--subdivisions" => {
                    config.subdivisions = next_value(&mut args, &flag)?
                        .parse()
                        .map_err(|_| "--subdivisions must be an integer in 1..=64".to_string())?;
                }
                "--maximum-samples" => {
                    config.maximum_samples = next_value(&mut args, &flag)?
                        .parse()
                        .map_err(|_| "--maximum-samples must be a positive integer".to_string())?;
                }
                "--maximum-quality-candidate-tests" => {
                    config.maximum_quality_candidate_tests =
                        next_value(&mut args, &flag)?.parse().map_err(|_| {
                            "--maximum-quality-candidate-tests must be a positive integer"
                                .to_string()
                        })?;
                }
                "--maximum-candidate-tests" => {
                    config.maximum_candidate_tests =
                        next_value(&mut args, &flag)?.parse().map_err(|_| {
                            "--maximum-candidate-tests must be a positive integer".to_string()
                        })?;
                }
                "--probe-sphere" => {
                    let mut values = [0.0_f64; 4];
                    for value in &mut values {
                        *value = next_value(&mut args, &flag)?.parse().map_err(|_| {
                            "--probe-sphere requires four finite numbers: X Y Z R".to_string()
                        })?;
                    }
                    if values.iter().any(|value| !value.is_finite()) || values[3] <= 0.0 {
                        return Err(
                            "--probe-sphere requires finite X Y Z and a positive finite R"
                                .to_string(),
                        );
                    }
                    config.probe_sphere = Some(([values[0], values[1], values[2]], values[3]));
                }
                "--part" => {
                    config.only_part = Some(
                        next_value(&mut args, &flag)?
                            .parse()
                            .map_err(|_| "--part must be a zero-based integer".to_string())?,
                    );
                }
                "--list-only" => config.list_only = true,
                "-h" | "--help" => return Err(usage()),
                _ => return Err(format!("unknown option {flag}\n\n{}", usage())),
            }
        }
        if !config.target_ratio.is_finite()
            || config.target_ratio <= 0.0
            || config.target_ratio > 1.0
        {
            return Err("--ratio must be finite and in (0, 1]".to_string());
        }
        if !config.target_error.is_finite()
            || config.target_error < 0.0
            || config.target_error > 1.0
        {
            return Err("--error must be finite and in [0, 1]".to_string());
        }
        if !config.maximum_normal_deviation_degrees.is_finite()
            || !(0.0..=90.0).contains(&config.maximum_normal_deviation_degrees)
        {
            return Err("--max-normal-deviation must be finite and in [0, 90]".to_string());
        }
        if !(1..=64).contains(&config.subdivisions) {
            return Err("--subdivisions must be in 1..=64".to_string());
        }
        if config.maximum_samples == 0
            || config.maximum_quality_candidate_tests == 0
            || config.maximum_candidate_tests == 0
        {
            return Err("sample and candidate-test budgets must be positive".to_string());
        }
        Ok(config)
    }
}

fn next_value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("missing value after {flag}"))
}

fn usage() -> String {
    "usage: coarse_qb_report <asset.glb> [--ratio 0.25] [--error 0.01] \
     [--max-normal-deviation 60] [--subdivisions 2] [--maximum-samples N] \
     [--maximum-quality-candidate-tests N] \
     [--maximum-candidate-tests N] \
     [--probe-sphere X Y Z R] [--part N] [--list-only]"
        .to_string()
}

#[derive(Debug)]
struct ScenePart {
    ordinal: usize,
    node: Option<usize>,
    node_name: Option<String>,
    mesh: usize,
    mesh_name: Option<String>,
    primitive: usize,
    material: Option<usize>,
    positions: Vec<[f64; 3]>,
    triangles: Vec<[usize; 3]>,
}

impl ScenePart {
    fn label(&self) -> String {
        let node = self
            .node_name
            .as_deref()
            .or(self.mesh_name.as_deref())
            .unwrap_or("unnamed");
        format!(
            "part {} node={:?} mesh={} primitive={} material={:?} ({node})",
            self.ordinal, self.node, self.mesh, self.primitive, self.material,
        )
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct ShapeStats {
    faces: usize,
    degenerate_faces: usize,
    total_area: f64,
    skinny_10_faces: usize,
    skinny_10_area: f64,
    skinny_50_faces: usize,
    skinny_50_area: f64,
    skinny_100_faces: usize,
    skinny_100_area: f64,
    maximum_aspect: f64,
}

impl ShapeStats {
    fn include(&mut self, other: Self) {
        self.faces += other.faces;
        self.degenerate_faces += other.degenerate_faces;
        self.total_area += other.total_area;
        self.skinny_10_faces += other.skinny_10_faces;
        self.skinny_10_area += other.skinny_10_area;
        self.skinny_50_faces += other.skinny_50_faces;
        self.skinny_50_area += other.skinny_50_area;
        self.skinny_100_faces += other.skinny_100_faces;
        self.skinny_100_area += other.skinny_100_area;
        self.maximum_aspect = self.maximum_aspect.max(other.maximum_aspect);
    }
}

#[derive(Debug)]
struct Success {
    source_faces: usize,
    coarse_faces: usize,
    charts: usize,
    backed_off_charts: usize,
    source_fallback_charts: usize,
    requested_chart_triangles: usize,
    selected_target_triangles: usize,
    total_backend_attempts: usize,
    total_rejected_backend_attempts: usize,
    fitted_attempts: usize,
    chart_reports: Vec<ChartReductionReport>,
    fitted_quality_fallbacks: Vec<FittedQualityFallback>,
    split_vertices: usize,
    constrained_vertices: usize,
    correspondence_rms: f64,
    correspondence_max: f64,
    correspondence_exhaustive_candidate_tests: u128,
    correspondence_candidate_tests: usize,
    correspondence_bvh_node_visits: usize,
    source_position_max_sample: Option<CorrespondenceSample>,
    source_normal_max_sample: Option<CorrespondenceSample>,
    build_time: Duration,
    fit_time: Duration,
    context_time: Duration,
    score_time: Duration,
    objective: f64,
    fit: LinearFitResult,
    score: FitScore,
}

fn main() {
    let config = match Config::parse() {
        Ok(config) => config,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(2);
        }
    };
    if let Err(error) = run(&config) {
        eprintln!("fatal: {error}");
        std::process::exit(1);
    }
}

fn run(config: &Config) -> Result<(), String> {
    let load_start = Instant::now();
    let bytes = std::fs::read(&config.path)
        .map_err(|error| format!("could not read {}: {error}", config.path.display()))?;
    let byte_count = bytes.len();
    let scene = quilting_gltf::load_gltf_raw(&bytes)
        .map_err(|error| format!("could not parse {}: {error}", config.path.display()))?;
    let load_time = load_start.elapsed();
    let parts = active_scene_parts(&scene)?;
    let excluded_mesh_nodes = excluded_mesh_node_count(&scene);
    let active_scene_index = scene
        .default_scene
        .filter(|index| *index < scene.scenes.len())
        .or_else(|| (!scene.scenes.is_empty()).then_some(0));
    let active_scene_name = active_scene_index
        .and_then(|index| scene.scenes[index].name.as_deref())
        .unwrap_or("unnamed");

    println!("QB COARSE-COMPLEX REPORT");
    println!("asset: {}", config.path.display());
    println!("bytes: {byte_count}");
    println!("load: {:.3}s", load_time.as_secs_f64());
    println!(
        "scene: selected={active_scene_index:?} name={active_scene_name:?} meshes={} nodes={} active_parts={} excluded_mesh_nodes={excluded_mesh_nodes} animations={} skins={}",
        scene.meshes.len(),
        scene.nodes.len(),
        parts.len(),
        scene.animations.len(),
        scene.skins.len(),
    );
    println!(
        "policy: ratio={:.6} error={:.6} sampled_normal_limit={:.3}deg subdivisions={} sample_budget={} quality_candidate_budget={} correspondence_candidate_budget={}",
        config.target_ratio,
        config.target_error,
        config.maximum_normal_deviation_degrees,
        config.subdivisions,
        config.maximum_samples,
        config.maximum_quality_candidate_tests,
        config.maximum_candidate_tests,
    );

    let mut aggregate_shape = ShapeStats::default();
    for part in &parts {
        let shape = shape_stats(&part.positions, &part.triangles);
        aggregate_shape.include(shape);
        println!(
            "{} vertices={} faces={} skinny>=10={} skinny>=50={} skinny>=100={} max_aspect={:.3}",
            part.label(),
            part.positions.len(),
            part.triangles.len(),
            shape.skinny_10_faces,
            shape.skinny_50_faces,
            shape.skinny_100_faces,
            shape.maximum_aspect,
        );
    }
    print_shape_summary("scene source", aggregate_shape);
    if config.list_only {
        return Ok(());
    }

    let (probes, probe_center, probe_radius, probe_source) = probes_for(config, &parts)?;
    println!(
        "probe: scene-offset-sphere-reflection source={probe_source} center=({:.9e},{:.9e},{:.9e}) radius={:.9e}",
        probe_center[0], probe_center[1], probe_center[2], probe_radius,
    );
    println!("ownership: primitives are isolated; boundary diagnostics are intra-primitive only");

    let patch_config = CoarsePatchConfig {
        reduction: CoarseReductionConfig {
            target_ratio: config.target_ratio,
            target_error: config.target_error,
            maximum_normal_deviation_degrees: config.maximum_normal_deviation_degrees,
            maximum_quality_candidate_tests: config.maximum_quality_candidate_tests,
        },
        correspondence_subdivisions: config.subdivisions,
        maximum_correspondence_samples: config.maximum_samples,
        maximum_candidate_tests: config.maximum_candidate_tests,
        ..CoarsePatchConfig::default()
    };
    let fit_config = LinearFitConfig::default();
    let score_config = FitScoreConfig::default();
    let selected = parts.iter().filter(|part| {
        config
            .only_part
            .is_none_or(|ordinal| part.ordinal == ordinal)
    });
    let mut attempted = 0usize;
    let mut succeeded = 0usize;
    let mut source_faces = 0usize;
    let mut coarse_faces = 0usize;
    let mut failures = Vec::new();
    let benchmark_start = Instant::now();

    for part in selected {
        attempted += 1;
        println!("RUN {}", part.label());
        match process_part(part, &patch_config, &fit_config, &score_config, &probes) {
            Ok(success) => {
                succeeded += 1;
                source_faces += success.source_faces;
                coarse_faces += success.coarse_faces;
                print_success(&success);
            }
            Err(error) => {
                println!("  FAIL {error}");
                failures.push((part.ordinal, error));
            }
        }
    }
    if attempted == 0 {
        return Err(format!(
            "part {:?} is not present; use --list-only to inspect ordinals",
            config.only_part
        ));
    }
    println!("SUMMARY");
    println!(
        "parts: attempted={attempted} succeeded={succeeded} failed={}",
        failures.len()
    );
    println!(
        "successful faces: source={source_faces} coarse={coarse_faces} reduction={:.3}x",
        source_faces as f64 / coarse_faces.max(1) as f64,
    );
    println!("benchmark: {:.3}s", benchmark_start.elapsed().as_secs_f64());
    for (ordinal, error) in &failures {
        println!("failure part={ordinal}: {error}");
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{} of {attempted} attempted part(s) failed; report is not accepted",
            failures.len(),
        ))
    }
}

fn process_part(
    part: &ScenePart,
    patch_config: &CoarsePatchConfig,
    fit_config: &LinearFitConfig,
    score_config: &FitScoreConfig,
    probes: &[ConformalProbe],
) -> Result<Success, String> {
    let vertex_ids = (0..part.positions.len())
        .map(|index| SourceVertexId(index as u64))
        .collect::<Vec<_>>();
    let face_ids = (0..part.triangles.len())
        .map(|index| SourceFaceId(index as u64))
        .collect::<Vec<_>>();
    let domains = vec![0; part.triangles.len()];
    let input = CoarseComplexInput {
        positions: &part.positions,
        triangles: &part.triangles,
        source_vertex_ids: &vertex_ids,
        source_face_ids: &face_ids,
        face_domains: &domains,
        locked_edges: &[],
    };
    let normalized = split_disconnected_vertex_fans(&input)
        .map_err(|error| format!("source-normalization: {error}"))?;
    let split_vertices = normalized.split_vertex_count();

    let fitted = fit_coarse_patch_complex_with_backoff(
        &normalized.input(),
        &FittedCoarsePatchConfig {
            patch: *patch_config,
            fit: *fit_config,
            score: *score_config,
            objective_weights: ObjectiveWeights::default(),
            maximum_fitted_normal_deviation_degrees: patch_config
                .reduction
                .maximum_normal_deviation_degrees,
        },
        probes,
    )
    .map_err(|error| error.to_string())?;
    let FittedCoarsePatchResult {
        complex,
        fit,
        score,
        objective,
        attempts,
        total_backend_attempts,
        total_rejected_backend_attempts,
        fitted_quality_fallbacks,
        timings,
    } = fitted;
    Ok(success(
        part,
        &complex,
        fit,
        score,
        objective,
        fitted_quality_fallbacks,
        total_backend_attempts,
        total_rejected_backend_attempts,
        attempts,
        split_vertices,
        timings.build,
        timings.fit,
        timings.context,
        timings.score,
    ))
}

fn success(
    part: &ScenePart,
    complex: &CoarsePatchComplex,
    fit: LinearFitResult,
    score: FitScore,
    objective: f64,
    fitted_quality_fallbacks: Vec<FittedQualityFallback>,
    total_backend_attempts: usize,
    total_rejected_backend_attempts: usize,
    fitted_attempts: usize,
    split_vertices: usize,
    build_time: Duration,
    fit_time: Duration,
    context_time: Duration,
    score_time: Duration,
) -> Success {
    Success {
        source_faces: part.triangles.len(),
        coarse_faces: complex.faces.len(),
        charts: complex.charts.len(),
        backed_off_charts: complex
            .charts
            .iter()
            .filter(|chart| chart.selected_target_triangles > chart.requested_triangles)
            .count(),
        source_fallback_charts: complex
            .charts
            .iter()
            .filter(|chart| chart.used_source_fallback)
            .count(),
        requested_chart_triangles: complex
            .charts
            .iter()
            .map(|chart| chart.requested_triangles)
            .sum(),
        selected_target_triangles: complex
            .charts
            .iter()
            .map(|chart| chart.selected_target_triangles)
            .sum(),
        total_backend_attempts,
        total_rejected_backend_attempts,
        fitted_attempts,
        chart_reports: complex.charts.clone(),
        fitted_quality_fallbacks,
        split_vertices,
        constrained_vertices: complex
            .vertices
            .iter()
            .filter(|vertex| vertex.constrained)
            .count(),
        correspondence_rms: complex
            .correspondence_diagnostics
            .weighted_rms_distance_ratio,
        correspondence_max: complex.correspondence_diagnostics.maximum_distance_ratio,
        correspondence_exhaustive_candidate_tests: complex
            .correspondence_diagnostics
            .exhaustive_candidate_tests,
        correspondence_candidate_tests: complex.correspondence_diagnostics.candidate_tests,
        correspondence_bvh_node_visits: complex.correspondence_diagnostics.bvh_node_visits,
        source_position_max_sample: score
            .source
            .position_max_sample
            .and_then(|sample| complex.correspondence.get(sample))
            .cloned(),
        source_normal_max_sample: score
            .source
            .normal_max_sample
            .and_then(|sample| complex.correspondence.get(sample))
            .cloned(),
        build_time,
        fit_time,
        context_time,
        score_time,
        objective,
        fit,
        score,
    }
}

fn print_success(success: &Success) {
    println!(
        "  OK source_faces={} coarse_faces={} reduction={:.3}x charts={} backed_off_charts={} source_fallback_charts={} requested_chart_triangles={} selected_target_triangles={} fitted_attempts={} total_backend_attempts={} total_rejected_backend_attempts={} split_vertices={} constrained_vertices={}",
        success.source_faces,
        success.coarse_faces,
        success.source_faces as f64 / success.coarse_faces.max(1) as f64,
        success.charts,
        success.backed_off_charts,
        success.source_fallback_charts,
        success.requested_chart_triangles,
        success.selected_target_triangles,
        success.fitted_attempts,
        success.total_backend_attempts,
        success.total_rejected_backend_attempts,
        success.split_vertices,
        success.constrained_vertices,
    );
    for (chart_index, chart) in success.chart_reports.iter().enumerate() {
        println!(
            "  final_chart={} source_faces={} requested={} selected={} achieved={} backend_attempts={} source_fallback={} backend_error={} sampled_normal_max={:.3}deg quality_samples={} quality_candidate_tests={} quality_bvh_node_visits={}",
            chart_index,
            chart.key.source_faces.len(),
            chart.requested_triangles,
            chart.selected_target_triangles,
            chart.achieved_triangles,
            chart.backend_attempts,
            chart.used_source_fallback,
            chart
                .backend_result_error
                .map(|error| format!("{error:.6e}"))
                .unwrap_or_else(|| "source".to_string()),
            chart.maximum_normal_deviation_degrees,
            chart.quality_sample_count,
            chart.quality_candidate_tests,
            chart.quality_bvh_node_visits,
        );
        for rejection in &chart.rejected_candidates {
            println!(
                "    rejected target={} category={} reason={} quality_candidate_tests={} quality_bvh_node_visits={}",
                rejection.target_triangles,
                rejection.category,
                rejection.reason,
                rejection.quality_candidate_tests,
                rejection.quality_bvh_node_visits,
            );
            if let Some(quality) = rejection.quality {
                println!(
                    "      quality source_face={:?} sample={} coarse_face={} measured={:.3}deg maximum={:.3}deg normalized_squared_distance={:.6e}",
                    quality.source_face,
                    quality.source_sample_ordinal,
                    quality.matched_coarse_face,
                    quality.measured_millidegrees as f64 / 1_000.0,
                    quality.maximum_millidegrees as f64 / 1_000.0,
                    quality.normalized_squared_distance,
                );
            }
        }
    }
    for fallback in &success.fitted_quality_fallbacks {
        println!(
            "  fitted-quality fallback attempt={} action={} chart={} chart_source_faces={} source_face={:?} sample={} measured={:.3}deg",
            fallback.attempt,
            fallback.action,
            fallback.chart,
            fallback.key.source_faces.len(),
            fallback.source_face,
            fallback.source_sample_ordinal,
            fallback.measured_degrees,
        );
    }
    println!(
        "  correspondence relative_rms={:.6e} relative_max={:.6e} exhaustive_candidate_tests={} candidate_tests={} candidate_reduction={:.3}x bvh_node_visits={}",
        success.correspondence_rms,
        success.correspondence_max,
        success.correspondence_exhaustive_candidate_tests,
        success.correspondence_candidate_tests,
        success.correspondence_exhaustive_candidate_tests as f64
            / success.correspondence_candidate_tests.max(1) as f64,
        success.correspondence_bvh_node_visits,
    );
    println!(
        "  fit iterations={} algebraic_rms={:.6e} max_weight_dev={:.6e} min_relative_denominator={:.6e}",
        success.fit.solver_iterations,
        success.fit.algebraic_residual_rms,
        success.fit.max_weight_dev,
        success.fit.min_relative_denominator,
    );
    println!(
        "  source_error position_rms={:.6e} position_max={:.6e} relative_position_rms={:.6e} normal_rms_degrees={:.6} normal_max_degrees={:.6} objective={:.6e}",
        success.score.source.position_rms,
        success.score.source.position_max,
        success.score.source.position_relative_rms,
        success.score.source.normal_rms_degrees,
        success.score.source.normal_max_degrees,
        success.objective,
    );
    print_worst_sample(
        "source_position_max",
        success.source_position_max_sample.as_ref(),
        &success.fit,
    );
    print_worst_sample(
        "source_normal_max",
        success.source_normal_max_sample.as_ref(),
        &success.fit,
    );
    println!(
        "  source_validity non_finite={} degenerate_normals={} boundary_invalid={} boundary_max={:.6e} near_singular={} invalid_patches={} min_relative_denominator={:.6e}",
        success.score.source.non_finite_samples,
        success.score.source.degenerate_normal_samples,
        success.score.boundary.invalid_pair_count,
        success.score.boundary.max_gap,
        success.score.weights.near_singular_patches,
        success.score.weights.invalid_patches,
        success.score.weights.min_relative_denominator,
    );
    for probe in &success.score.conformal_probes {
        println!(
            "  probe={} position_rms={:.6e} position_max={:.6e} relative_position_rms={:.6e} normal_rms_degrees={:.6} normal_max_degrees={:.6} boundary_max={:.6e} dilation={:.6e} pole_near={} non_finite={} degenerate_normals={} boundary_invalid={} near_singular={} invalid_patches={} min_relative_denominator={:.6e}",
            probe.name,
            probe.error.position_rms,
            probe.error.position_max,
            probe.error.position_relative_rms,
            probe.error.normal_rms_degrees,
            probe.error.normal_max_degrees,
            probe.boundary.max_gap,
            probe.peak_local_dilation,
            probe.pole_near_samples,
            probe.error.non_finite_samples,
            probe.error.degenerate_normal_samples,
            probe.boundary.invalid_pair_count,
            probe.weights.near_singular_patches,
            probe.weights.invalid_patches,
            probe.weights.min_relative_denominator,
        );
    }
    println!(
        "  time build={:.3}s fit={:.3}s context={:.3}s score={:.3}s",
        success.build_time.as_secs_f64(),
        success.fit_time.as_secs_f64(),
        success.context_time.as_secs_f64(),
        success.score_time.as_secs_f64(),
    );
}

fn print_worst_sample(label: &str, sample: Option<&CorrespondenceSample>, fit: &LinearFitResult) {
    let Some(sample) = sample else {
        println!("  {label}_sample=unavailable");
        return;
    };
    println!(
        "  {label}_sample source_face={} ordinal={} coarse_face={} chart={} source_bary=({:.6},{:.6},{:.6}) coarse_bary=({:.6},{:.6},{:.6}) correspondence_distance={:.6e}",
        sample.key.face.0,
        sample.key.ordinal,
        sample.coarse_face,
        sample.coarse_face_key.chart,
        sample.source_barycentric[0],
        sample.source_barycentric[1],
        sample.source_barycentric[2],
        sample.coarse_barycentric[0],
        sample.coarse_barycentric[1],
        sample.coarse_barycentric[2],
        sample.distance_ratio,
    );
    let patch = &fit.patches[sample.coarse_face];
    let differential =
        patch.eval_differential(sample.coarse_barycentric[1], sample.coarse_barycentric[2]);
    let tangent_cross = geometry::vec3_cross(differential.tangent_u, differential.tangent_v);
    let patch_normal = unit(tangent_cross);
    let linear_positions = patch.positions.map(|position| position.to_point());
    let linear_normal = unit(geometry::vec3_cross(
        geometry::vec3_sub(linear_positions[1], linear_positions[0]),
        geometry::vec3_sub(linear_positions[2], linear_positions[0]),
    ));
    let target_normal = unit(sample.target_normal);
    println!(
        "  {label}_frame linear_target_angle={:.6} fitted_target_angle={:.6} fitted_linear_angle={:.6} tangent_area={:.6e}",
        angle_degrees(linear_normal, target_normal),
        angle_degrees(patch_normal, target_normal),
        angle_degrees(patch_normal, linear_normal),
        geometry::vec3_len(tangent_cross),
    );
}

fn unit(vector: [f64; 3]) -> [f64; 3] {
    geometry::vec3_scale(vector, geometry::vec3_len(vector).recip())
}

fn angle_degrees(left: [f64; 3], right: [f64; 3]) -> f64 {
    geometry::vec3_dot(left, right)
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees()
}

fn probes_for(
    config: &Config,
    parts: &[ScenePart],
) -> Result<(Vec<ConformalProbe>, [f64; 3], f64, &'static str), String> {
    let scene_positions = parts
        .iter()
        .flat_map(|part| part.positions.iter().copied())
        .collect::<Vec<_>>();
    let (minimum, maximum) =
        bounds(&scene_positions).ok_or("active scene has no finite positive extent")?;
    let center = [
        (minimum[0] + maximum[0]) * 0.5,
        (minimum[1] + maximum[1]) * 0.5,
        (minimum[2] + maximum[2]) * 0.5,
    ];
    let diagonal = (maximum[0] - minimum[0])
        .hypot(maximum[1] - minimum[1])
        .hypot(maximum[2] - minimum[2]);
    let (pole, radius, source) = config.probe_sphere.map_or_else(
        || {
            (
                [
                    center[0] + 1.75 * diagonal,
                    center[1] + 0.35 * diagonal,
                    center[2] - 0.25 * diagonal,
                ],
                0.9 * diagonal,
                "derived-scene-bounds",
            )
        },
        |(pole, radius)| (pole, radius, "explicit-cli"),
    );
    Ok((
        vec![ConformalProbe {
            name: "scene-offset-sphere-reflection".to_string(),
            transform: Mobius::sphere_reflection(
                Quat::from_point(pole[0], pole[1], pole[2]),
                radius,
            ),
        }],
        pole,
        radius,
        source,
    ))
}

fn validate_primitive(
    mesh: usize,
    primitive: usize,
    geometry: &quilting_gltf::mesh::Primitive,
) -> Result<(), String> {
    for (vertex, position) in geometry.positions.iter().enumerate() {
        if position.iter().any(|component| !component.is_finite()) {
            return Err(format!(
                "mesh {mesh} primitive {primitive} vertex {vertex} is non-finite"
            ));
        }
    }
    for (face, triangle) in geometry.triangles.iter().enumerate() {
        for &vertex in triangle {
            if vertex >= geometry.positions.len() {
                return Err(format!(
                    "mesh {mesh} primitive {primitive} face {face} references missing vertex {vertex}"
                ));
            }
        }
    }
    Ok(())
}

fn active_scene_parts(scene: &quilting_gltf::GltfSceneRaw) -> Result<Vec<ScenePart>, String> {
    if !scene.animations.is_empty() {
        return Err(format!(
            "report is static-only but the asset contains {} animation(s)",
            scene.animations.len(),
        ));
    }
    let mut instances = Vec::<(Option<usize>, usize, [f64; 16])>::new();
    if let Some(active) = scene
        .default_scene
        .and_then(|index| scene.scenes.get(index))
        .or_else(|| scene.scenes.first())
    {
        let reachable = reachable_nodes(&scene.nodes, &active.root_nodes)?;
        let world = quilting_gltf::scene::compute_world_transforms(&scene.nodes, active);
        for node in reachable {
            if let Some(mesh) = scene.nodes[node].mesh {
                if scene.nodes[node].skin.is_some() {
                    return Err(format!(
                        "report is static-only but active node {node} uses a skin"
                    ));
                }
                instances.push((Some(node), mesh, world[node]));
            }
        }
    } else {
        instances.extend(
            scene
                .meshes
                .iter()
                .enumerate()
                .map(|(mesh, _)| (None, mesh, IDENTITY)),
        );
    }

    let mut parts = Vec::new();
    for (node, mesh_index, transform) in instances {
        let mesh = scene
            .meshes
            .get(mesh_index)
            .ok_or_else(|| format!("node {node:?} references missing mesh {mesh_index}"))?;
        for (primitive_index, primitive) in mesh.primitives.iter().enumerate() {
            if primitive.positions.is_empty() || primitive.triangles.is_empty() {
                continue;
            }
            if !primitive.morph_targets.is_empty() {
                return Err(format!(
                    "report is static-only but mesh {mesh_index} primitive {primitive_index} has morph targets"
                ));
            }
            validate_primitive(mesh_index, primitive_index, primitive)?;
            parts.push(ScenePart {
                ordinal: parts.len(),
                node,
                node_name: node.and_then(|index| scene.nodes[index].name.clone()),
                mesh: mesh_index,
                mesh_name: mesh.name.clone(),
                primitive: primitive_index,
                material: primitive.material_index,
                positions: primitive
                    .positions
                    .iter()
                    .map(|position| transform_point(&transform, *position))
                    .collect(),
                triangles: primitive.triangles.clone(),
            });
        }
    }
    if parts.is_empty() {
        return Err("active scene contains no triangle mesh primitives".to_string());
    }
    Ok(parts)
}

fn excluded_mesh_node_count(scene: &quilting_gltf::GltfSceneRaw) -> usize {
    let Some(active) = scene
        .default_scene
        .and_then(|index| scene.scenes.get(index))
        .or_else(|| scene.scenes.first())
    else {
        return 0;
    };
    let reachable = reachable_nodes(&scene.nodes, &active.root_nodes).unwrap_or_default();
    let reachable = reachable.into_iter().collect::<BTreeSet<_>>();
    scene
        .nodes
        .iter()
        .enumerate()
        .filter(|(index, node)| node.mesh.is_some() && !reachable.contains(index))
        .count()
}

fn reachable_nodes(
    nodes: &[quilting_gltf::scene::Node],
    roots: &[usize],
) -> Result<Vec<usize>, String> {
    let mut pending = roots.to_vec();
    let mut reachable = BTreeSet::new();
    while let Some(node) = pending.pop() {
        if node >= nodes.len() {
            return Err(format!("scene references missing node {node}"));
        }
        if reachable.insert(node) {
            pending.extend(nodes[node].children.iter().copied());
        }
    }
    Ok(reachable.into_iter().collect())
}

fn transform_point(matrix: &[f64; 16], point: [f64; 3]) -> [f64; 3] {
    [
        matrix[0] * point[0] + matrix[4] * point[1] + matrix[8] * point[2] + matrix[12],
        matrix[1] * point[0] + matrix[5] * point[1] + matrix[9] * point[2] + matrix[13],
        matrix[2] * point[0] + matrix[6] * point[1] + matrix[10] * point[2] + matrix[14],
    ]
}

fn shape_stats(positions: &[[f64; 3]], triangles: &[[usize; 3]]) -> ShapeStats {
    let mut stats = ShapeStats::default();
    for triangle in triangles {
        stats.faces += 1;
        let [a, b, c] = triangle.map(|index| positions[index]);
        let ab = subtract(b, a);
        let ac = subtract(c, a);
        let bc = subtract(c, b);
        let longest_squared = norm_squared(ab).max(norm_squared(ac)).max(norm_squared(bc));
        let double_area = norm(cross(ab, ac));
        let area = double_area * 0.5;
        stats.total_area += area;
        let aspect = longest_squared / double_area;
        if !aspect.is_finite() {
            stats.degenerate_faces += 1;
            stats.maximum_aspect = f64::INFINITY;
            continue;
        }
        stats.maximum_aspect = stats.maximum_aspect.max(aspect);
        if aspect >= 10.0 {
            stats.skinny_10_faces += 1;
            stats.skinny_10_area += area;
        }
        if aspect >= 50.0 {
            stats.skinny_50_faces += 1;
            stats.skinny_50_area += area;
        }
        if aspect >= 100.0 {
            stats.skinny_100_faces += 1;
            stats.skinny_100_area += area;
        }
    }
    stats
}

fn print_shape_summary(label: &str, stats: ShapeStats) {
    let face_percent = |count| 100.0 * count as f64 / stats.faces.max(1) as f64;
    let area_percent = |area| 100.0 * area / stats.total_area.max(f64::MIN_POSITIVE);
    println!("SHAPE {label}");
    println!(
        "faces={} degenerate={} total_area={:.9e} max_aspect={:.6e}",
        stats.faces, stats.degenerate_faces, stats.total_area, stats.maximum_aspect,
    );
    println!(
        "aspect>=10 faces={} ({:.3}%) area={:.3}%",
        stats.skinny_10_faces,
        face_percent(stats.skinny_10_faces),
        area_percent(stats.skinny_10_area),
    );
    println!(
        "aspect>=50 faces={} ({:.3}%) area={:.3}%",
        stats.skinny_50_faces,
        face_percent(stats.skinny_50_faces),
        area_percent(stats.skinny_50_area),
    );
    println!(
        "aspect>=100 faces={} ({:.3}%) area={:.3}%",
        stats.skinny_100_faces,
        face_percent(stats.skinny_100_faces),
        area_percent(stats.skinny_100_area),
    );
}

fn bounds(points: &[[f64; 3]]) -> Option<([f64; 3], [f64; 3])> {
    let mut minimum = [f64::INFINITY; 3];
    let mut maximum = [f64::NEG_INFINITY; 3];
    for point in points {
        for axis in 0..3 {
            minimum[axis] = minimum[axis].min(point[axis]);
            maximum[axis] = maximum[axis].max(point[axis]);
        }
    }
    let positive = (0..3).any(|axis| maximum[axis] > minimum[axis]);
    (positive
        && minimum.iter().all(|value| value.is_finite())
        && maximum.iter().all(|value| value.is_finite()))
    .then_some((minimum, maximum))
}

fn subtract(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn norm_squared(vector: [f64; 3]) -> f64 {
    vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]
}

fn norm(vector: [f64; 3]) -> f64 {
    norm_squared(vector).sqrt()
}
