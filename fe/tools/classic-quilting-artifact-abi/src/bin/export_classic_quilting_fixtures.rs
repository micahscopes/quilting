use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use classic_quilting_artifact_abi::quilting_export::{
    build_direct_fixture_artifact, build_direct_fixture_matrix, checked_triangulate_fixture,
    near_degenerate_fixture, DIRECT_ALGORITHM_VERSION, FIXTURE_KEYS, FIXTURE_K_CANDIDATES,
    FIXTURE_MASTER_SEED, VERIFIED_POOL_WIDTHS,
};
use classic_quilting_artifact_abi::{decode, encode, Artifact, AtlasKey};
use serde::Serialize;

const DEFAULT_OUTPUT: &str = "fixtures/classic-quilting/v1";
const GENERATOR_COMMAND_PREFIX: &str = "cargo run --locked --release --manifest-path tools/classic-quilting-artifact-abi/Cargo.toml --features quilting-export --bin export-classic-quilting-fixtures -- --output fixtures/classic-quilting/v1 --quilting-commit";

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Manifest<'a> {
    schema_version: u32,
    algorithm_version: u32,
    algorithm: &'a str,
    generator_version: &'a str,
    generator_command: String,
    quilting_commit: &'a str,
    license: &'a str,
    master_seed: u64,
    k_candidates: usize,
    pool_widths_verified: [usize; 3],
    combined: ArtifactRecord,
    fixtures: Vec<ArtifactRecord>,
    rejection: RejectionRecord,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ArtifactRecord {
    file: String,
    keys: Vec<[u32; 3]>,
    vertex_count: usize,
    triangle_count: usize,
    payload_sha256: String,
    boundary_parameter_f32_bits: Vec<[Vec<String>; 3]>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RejectionRecord {
    file: &'static str,
    domain: &'static str,
    code: &'static str,
    first_point: usize,
    second_point: usize,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RejectionFixture {
    name: &'static str,
    cartesian_f64_bits: Vec<[String; 2]>,
    barycentric_f64_bits: Vec<[String; 3]>,
    result: RejectionResult,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RejectionResult {
    domain: &'static str,
    code: &'static str,
    first_point: usize,
    second_point: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let (output, quilting_commit) = parse_args()?;
    fs::create_dir_all(&output)?;

    let mut pool_timings = Vec::new();
    let mut canonical_bytes = None;
    for width in VERIFIED_POOL_WIDTHS {
        let started = Instant::now();
        let artifact = build_direct_fixture_artifact(&FIXTURE_KEYS, width)?;
        let bytes = encode(&artifact)?;
        if let Some(expected) = &canonical_bytes {
            if *expected != bytes {
                return Err(format!("pool width {width} changed canonical bytes").into());
            }
        } else {
            canonical_bytes = Some(bytes);
        }
        pool_timings.push((width, started.elapsed()));
    }
    let combined_bytes = canonical_bytes.expect("verified nonempty pool width list");
    let combined_artifact = decode(&combined_bytes)?;
    let combined_file = "direct-seed42-matrix.cqa";
    fs::write(output.join(combined_file), &combined_bytes)?;

    let mut fixture_records = Vec::new();
    for (key, artifact) in build_direct_fixture_matrix(1)? {
        let bytes = encode(&artifact)?;
        let file = format!("direct-seed42-k{}-{}-{}.cqa", key.a, key.b, key.c);
        fs::write(output.join(&file), &bytes)?;
        fixture_records.push(record_for_artifact(file, &artifact, &bytes));
    }

    let rejection_file = "near-degenerate-rejection.json";
    let rejection = write_rejection_fixture(&output.join(rejection_file))?;
    let manifest = Manifest {
        schema_version: classic_quilting_artifact_abi::SCHEMA_VERSION,
        algorithm_version: DIRECT_ALGORITHM_VERSION,
        algorithm: "quilting-core direct Bridson sampling plus constrained CDT",
        generator_version: env!("CARGO_PKG_VERSION"),
        generator_command: format!("{GENERATOR_COMMAND_PREFIX} {quilting_commit}"),
        quilting_commit: &quilting_commit,
        license: "MIT OR Apache-2.0",
        master_seed: FIXTURE_MASTER_SEED,
        k_candidates: FIXTURE_K_CANDIDATES,
        pool_widths_verified: VERIFIED_POOL_WIDTHS,
        combined: record_for_artifact(
            combined_file.to_owned(),
            &combined_artifact,
            &combined_bytes,
        ),
        fixtures: fixture_records,
        rejection,
    };
    fs::write(
        output.join("manifest.json"),
        format!("{}\n", serde_json::to_string_pretty(&manifest)?),
    )?;

    for (width, elapsed) in pool_timings {
        println!("pool_width={width} elapsed_us={}", elapsed.as_micros());
    }
    println!("wrote {}", output.display());
    Ok(())
}

fn parse_args() -> Result<(PathBuf, String), Box<dyn Error>> {
    let mut output = PathBuf::from(DEFAULT_OUTPUT);
    let mut quilting_commit = None;
    let mut args = env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--output" => {
                output = PathBuf::from(args.next().ok_or("--output requires a path")?);
            }
            "--quilting-commit" => {
                quilting_commit = Some(args.next().ok_or("--quilting-commit requires a hash")?);
            }
            _ => return Err(format!("unknown argument: {argument}").into()),
        }
    }
    let quilting_commit = quilting_commit.ok_or("--quilting-commit is required")?;
    if quilting_commit.len() != 40 || !quilting_commit.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err("--quilting-commit must be a full 40-character hex commit".into());
    }
    Ok((output, quilting_commit.to_ascii_lowercase()))
}

fn record_for_artifact(file: String, artifact: &Artifact, bytes: &[u8]) -> ArtifactRecord {
    ArtifactRecord {
        file,
        keys: artifact
            .patches
            .iter()
            .map(|patch| [patch.key.a, patch.key.b, patch.key.c])
            .collect(),
        vertex_count: artifact.vertices.len(),
        triangle_count: artifact.triangles.len(),
        payload_sha256: hex(&bytes[80..112]),
        boundary_parameter_f32_bits: artifact
            .patches
            .iter()
            .map(|patch| {
                boundary_parameter_bits(artifact, patch.key, patch.first_vertex, patch.vertex_count)
            })
            .collect(),
    }
}

fn boundary_parameter_bits(
    artifact: &Artifact,
    _key: AtlasKey,
    first_vertex: u32,
    vertex_count: u32,
) -> [Vec<String>; 3] {
    let start = first_vertex as usize;
    let end = start + vertex_count as usize;
    std::array::from_fn(|edge| {
        let mut parameters: Vec<f32> = artifact.vertices[start..end]
            .iter()
            .filter_map(|vertex| {
                let bary = vertex.barycentric;
                (bary[edge] == 0.0).then(|| match edge {
                    0 => bary[2],
                    1 => bary[0],
                    2 => bary[1],
                    _ => unreachable!(),
                })
            })
            .collect();
        parameters.sort_by(f32::total_cmp);
        parameters
            .into_iter()
            .map(|value| format!("0x{:08x}", value.to_bits()))
            .collect()
    })
}

fn write_rejection_fixture(path: &Path) -> Result<RejectionRecord, Box<dyn Error>> {
    let (positions, barycentrics) = near_degenerate_fixture();
    let Err(error) = checked_triangulate_fixture(&positions, &barycentrics) else {
        return Err("the frozen near-degenerate fixture must reject".into());
    };
    let classic_quilting_artifact_abi::quilting_export::FixtureExportError::NearDuplicatePoints {
        first,
        second,
    } = error
    else {
        return Err(format!("unexpected near-degenerate error: {error}").into());
    };
    let fixture = RejectionFixture {
        name: "near-collinear AB boundary pair",
        cartesian_f64_bits: positions
            .iter()
            .map(|point| point.map(|value| format!("0x{:016x}", value.to_bits())))
            .collect(),
        barycentric_f64_bits: barycentrics
            .iter()
            .map(|point| point.map(|value| format!("0x{:016x}", value.to_bits())))
            .collect(),
        result: RejectionResult {
            domain: "triangulation",
            code: "near_duplicate_points",
            first_point: first,
            second_point: second,
        },
    };
    fs::write(
        path,
        format!("{}\n", serde_json::to_string_pretty(&fixture)?),
    )?;
    Ok(RejectionRecord {
        file: "near-degenerate-rejection.json",
        domain: "triangulation",
        code: "near_duplicate_points",
        first_point: first,
        second_point: second,
    })
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;

    bytes.iter().fold(String::new(), |mut output, byte| {
        write!(output, "{byte:02x}").expect("writing into String cannot fail");
        output
    })
}
