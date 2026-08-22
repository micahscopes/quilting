use std::time::Instant;

use quilting_core::atlas::{ratio_bounded_canonical_triples, TessellationAtlas};
use quilting_core::sampling::PatchConfig;

fn main() {
    let config = PatchConfig::default();
    for exponent in [7, 8, 9] {
        let levels: Vec<u32> = (0..=exponent).map(|level| 1 << level).collect();
        let keys = ratio_bounded_canonical_triples(&levels, 2);
        let mut timings = Vec::new();
        let mut counts = (0, 0);

        for round in 0..11 {
            let started = Instant::now();
            let atlas = TessellationAtlas::build_hierarchical_for_keys(
                &levels,
                &keys,
                &config,
            );
            if round > 0 {
                timings.push(started.elapsed());
            }
            counts = (atlas.positions.len(), atlas.triangles.len());
        }

        timings.sort_unstable();
        let median = (timings[4] + timings[5]) / 2;
        println!(
            "2^0..2^{exponent}: {} patches, {} vertices, {} triangles, median {:.3}ms",
            keys.len(),
            counts.0,
            counts.1,
            median.as_secs_f64() * 1000.0,
        );
    }
}
