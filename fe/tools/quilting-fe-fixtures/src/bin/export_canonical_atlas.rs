//! Publish Quilting's canonical Rust atlas through the checked CQA boundary.
//!
//! Key enumeration and topology remain owned by `quilting-core`; this binary
//! only adapts the resulting canonical permutation orbits to the portable Fe
//! fixture format.

use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use quilting_core::atlas::canonical_triples;
use quilting_fe_fixtures::quilting_export::build_direct_fixture_artifact;
use quilting_fe_fixtures::{encode, AtlasKey};

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args().skip(1);
    let max_lod_exponent: u32 = arguments
        .next()
        .ok_or_else(usage)?
        .parse()
        .map_err(|_| usage())?;
    let output = PathBuf::from(arguments.next().ok_or_else(usage)?);
    if max_lod_exponent > 30 || arguments.next().is_some() {
        return Err(usage().into());
    }

    let levels: Vec<u32> = (0..=max_lod_exponent)
        .map(|exponent| 1_u32 << exponent)
        .collect();
    let keys: Vec<AtlasKey> = canonical_triples(&levels)
        .into_iter()
        .map(|[a, b, c]| AtlasKey::new(a, b, c))
        .collect();

    let started = Instant::now();
    let workers = std::thread::available_parallelism()?.get();
    let artifact = build_direct_fixture_artifact(&keys, workers)?;
    let bytes = encode(&artifact)?;
    fs::write(&output, &bytes)?;

    let max_patch_triangles = artifact
        .patches
        .iter()
        .map(|patch| patch.triangle_count)
        .max()
        .unwrap_or(0);
    println!(
        "levels={} canonical_keys={} vertices={} triangles={} max_patch_triangles={} bytes={} elapsed_ms={}",
        levels.len(),
        artifact.patches.len(),
        artifact.vertices.len(),
        artifact.triangles.len(),
        max_patch_triangles,
        bytes.len(),
        started.elapsed().as_millis(),
    );
    Ok(())
}

fn usage() -> String {
    "usage: export_canonical_atlas <max-lod-exponent> <output.cqa>".to_owned()
}
