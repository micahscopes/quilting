use std::collections::BTreeSet;
use std::env;
use std::process;
use std::time::{Duration, Instant};

use quilting_core::atlas::{ratio_bounded_canonical_triples, BuildMode, TessellationAtlas};
use quilting_core::batch::{balance_resident_lods_with_ratio, ResidentLod};
use quilting_core::sampling::PatchConfig;
use quilting_mesh::HalfEdgeMesh;

struct Options {
    min_exponent: Option<u32>,
    max_exponent: u32,
    ratios: Vec<u32>,
    modes: Vec<BuildMode>,
    rounds: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            min_exponent: None,
            max_exponent: 9,
            ratios: vec![2, 4],
            modes: vec![BuildMode::Hierarchical],
            rounds: 5,
        }
    }
}

fn usage() -> &'static str {
    "usage: cargo run -p quilting-core --release --example bench_runtime_atlas -- \
     [--min-exp N] [--max-exp N] [--ratios 2,4] \
     [--modes hierarchical,direct] [--rounds N]"
}

fn parse_u32(value: Option<String>, flag: &str) -> u32 {
    value
        .and_then(|value| value.parse().ok())
        .unwrap_or_else(|| panic!("{flag} requires an unsigned integer"))
}

fn parse_list(value: Option<String>, flag: &str) -> Vec<u32> {
    let value = value.unwrap_or_else(|| panic!("{flag} requires a comma-separated list"));
    let parsed: Vec<u32> = value
        .split(',')
        .map(|part| {
            part.parse::<u32>()
                .unwrap_or_else(|_| panic!("invalid {flag} value {part:?}"))
        })
        .collect();
    assert!(
        !parsed.is_empty() && parsed.iter().all(|ratio| matches!(ratio, 2 | 4)),
        "{flag} currently supports 2 and 4"
    );
    parsed
}

fn parse_options() -> Options {
    let mut options = Options::default();
    let mut arguments = env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--min-exp" => {
                options.min_exponent = Some(parse_u32(arguments.next(), "--min-exp"));
            }
            "--max-exp" => {
                options.max_exponent = parse_u32(arguments.next(), "--max-exp");
            }
            "--ratios" => options.ratios = parse_list(arguments.next(), "--ratios"),
            "--modes" => {
                let value = arguments
                    .next()
                    .unwrap_or_else(|| panic!("--modes requires a comma-separated list"));
                options.modes = value
                    .split(',')
                    .map(|mode| match mode {
                        "direct" => BuildMode::Direct,
                        "hierarchical" => BuildMode::Hierarchical,
                        _ => panic!("unknown atlas mode {mode:?}"),
                    })
                    .collect();
                assert!(!options.modes.is_empty(), "--modes cannot be empty");
            }
            "--rounds" => {
                options.rounds = parse_u32(arguments.next(), "--rounds") as usize;
            }
            "-h" | "--help" => {
                println!("{}", usage());
                process::exit(0);
            }
            _ => panic!("unknown argument {argument:?}\n{}", usage()),
        }
    }
    assert!(options.max_exponent <= 30, "--max-exp must be at most 30");
    assert!(options.rounds > 0, "--rounds must be positive");
    let min_exponent = options
        .min_exponent
        .unwrap_or(options.max_exponent.saturating_sub(2));
    assert!(
        min_exponent <= options.max_exponent,
        "--min-exp exceeds --max-exp"
    );
    options.min_exponent = Some(min_exponent);
    options
}

fn mode_name(mode: BuildMode) -> &'static str {
    match mode {
        BuildMode::Direct => "direct-blue-noise",
        BuildMode::Hierarchical => "hierarchical",
    }
}

fn independent_topology_count(keys: &[[u32; 3]], mode: BuildMode) -> usize {
    if mode == BuildMode::Direct {
        return keys.len();
    }
    keys.iter()
        .copied()
        .map(|mut key| {
            while key.iter().all(|resolution| resolution % 2 == 0) {
                key = key.map(|resolution| resolution / 2);
            }
            key
        })
        .collect::<BTreeSet<_>>()
        .len()
}

fn boundary_mismatches(atlas: &TessellationAtlas) -> usize {
    let mut mismatches = 0;
    for (key, entry) in &atlas.patches {
        let mut counts = [0_usize; 3];
        for &[x, y] in &atlas.positions[entry.base_vertex..entry.base_vertex + entry.vertex_count] {
            let bary = quilting_core::triangle::cartesian_to_bary(x, y);
            for (edge, coordinate) in bary.into_iter().enumerate() {
                counts[edge] += usize::from(coordinate.abs() <= 1.0e-9);
            }
        }
        mismatches += counts
            .into_iter()
            .zip(*key)
            .filter(|(count, resolution)| *count != *resolution as usize + 1)
            .count();
    }
    mismatches
}

fn triangle_quality_percentiles(atlas: &TessellationAtlas) -> (f64, f64) {
    let mut qualities = Vec::with_capacity(atlas.triangles.len());
    for &[a, b, c] in &atlas.triangles {
        let [ax, ay] = atlas.positions[a];
        let [bx, by] = atlas.positions[b];
        let [cx, cy] = atlas.positions[c];
        let ab2 = (ax - bx).powi(2) + (ay - by).powi(2);
        let bc2 = (bx - cx).powi(2) + (by - cy).powi(2);
        let ca2 = (cx - ax).powi(2) + (cy - ay).powi(2);
        let twice_area = ((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)).abs();
        let denominator = ab2 + bc2 + ca2;
        let quality = if denominator > 0.0 {
            2.0 * 3.0_f64.sqrt() * twice_area / denominator
        } else {
            0.0
        };
        qualities.push(quality);
    }
    qualities.sort_unstable_by(f64::total_cmp);
    if qualities.is_empty() {
        return (0.0, 0.0);
    }
    let percentile_index = (qualities.len() / 100).min(qualities.len() - 1);
    (qualities[0], qualities[percentile_index])
}

fn promotion_halo(
    atlas: &TessellationAtlas,
    maximum_lod: u32,
    ratio: u32,
) -> (usize, usize, usize, usize) {
    const SIDE: u32 = 24;
    let mut faces = Vec::with_capacity((SIDE * SIDE * 2) as usize);
    for y in 0..SIDE {
        for x in 0..SIDE {
            let row = SIDE + 1;
            let a = y * row + x;
            let b = a + 1;
            let c = a + row;
            let d = c + 1;
            faces.push([a, b, c]);
            faces.push([b, d, c]);
        }
    }
    let topology = HalfEdgeMesh::from_triangles((SIDE + 1) * (SIDE + 1), &faces);
    let mut residents = vec![Some(ResidentLod::uniform(1)); faces.len()];
    let peak_face = ((SIDE / 2 * SIDE + SIDE / 2) * 2) as usize;
    residents[peak_face] = Some(ResidentLod::uniform(maximum_lod));

    let unit_triangles = atlas.get_patch([1; 3]).expect("unit patch").triangles.len();
    let peak_triangles = atlas
        .get_patch([maximum_lod; 3])
        .expect("peak patch")
        .triangles
        .len();
    let requested_triangles = unit_triangles * (faces.len() - 1) + peak_triangles;
    let promoted_faces = match ratio {
        2 => balance_resident_lods_with_ratio::<2>(&mut residents, &topology),
        4 => balance_resident_lods_with_ratio::<4>(&mut residents, &topology),
        _ => unreachable!("ratio validation admits only benchmarked policies"),
    };
    let halo_faces = residents
        .iter()
        .filter(|resident| **resident != Some(ResidentLod::uniform(1)))
        .count()
        .saturating_sub(1);
    let resident_triangles = residents
        .iter()
        .map(|resident| {
            atlas
                .get_patch(resident.expect("complete synthetic residency").edge_lods())
                .expect("balanced key must exist in policy atlas")
                .triangles
                .len()
        })
        .sum();
    (
        promoted_faces,
        halo_faces,
        requested_triangles,
        resident_triangles,
    )
}

fn median(mut timings: Vec<Duration>) -> Duration {
    timings.sort_unstable();
    let middle = timings.len() / 2;
    if timings.len().is_multiple_of(2) {
        (timings[middle - 1] + timings[middle]) / 2
    } else {
        timings[middle]
    }
}

fn main() {
    let options = parse_options();
    let config = PatchConfig::default();
    let min_exponent = options.min_exponent.unwrap();

    for exponent in min_exponent..=options.max_exponent {
        let levels: Vec<u32> = (0..=exponent).map(|level| 1 << level).collect();
        for &ratio in &options.ratios {
            let keys = ratio_bounded_canonical_triples(&levels, ratio);
            let maximum = 64_u32.min(1 << exponent);
            let request = [2_u32.min(maximum), maximum, maximum];
            let minimum_allowed = (maximum / ratio).max(1);
            let resident = request.map(|resolution| resolution.max(minimum_allowed));
            println!(
                "policy max=2^{exponent} ratio={ratio}: request={request:?} -> resident={resident:?}"
            );

            for &mode in &options.modes {
                let mut timings = Vec::with_capacity(options.rounds);
                let mut final_atlas = None;
                for round in 0..=options.rounds {
                    let started = Instant::now();
                    let atlas = TessellationAtlas::build_for_keys(&levels, &keys, &config, mode);
                    if round > 0 {
                        timings.push(started.elapsed());
                    }
                    final_atlas = Some(atlas);
                }
                let atlas = final_atlas.unwrap();
                let (minimum_quality, percentile_quality) = triangle_quality_percentiles(&atlas);
                let resident_patch = atlas
                    .get_patch(resident)
                    .expect("balanced example must be present in the policy atlas");
                let (promoted_faces, halo_faces, requested_triangles, resident_triangles) =
                    promotion_halo(&atlas, 1 << exponent, ratio);
                println!(
                    "  mode={} keys={} independent={} vertices={} triangles={} bytes={} \
                     example_vertices={} example_triangles={} boundary_mismatches={} \
                     halo_promoted={} halo_faces={} halo_requested_triangles={} \
                     halo_resident_triangles={} \
                     q_min={minimum_quality:.6} q_p01={percentile_quality:.6} \
                     median_ms={:.3}",
                    mode_name(mode),
                    keys.len(),
                    independent_topology_count(&keys, mode),
                    atlas.positions.len(),
                    atlas.triangles.len(),
                    atlas.to_bytes().len(),
                    resident_patch.positions.len(),
                    resident_patch.triangles.len(),
                    boundary_mismatches(&atlas),
                    promoted_faces,
                    halo_faces,
                    requested_triangles,
                    resident_triangles,
                    median(timings).as_secs_f64() * 1000.0,
                );
            }
        }
    }
}
