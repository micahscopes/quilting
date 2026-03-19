use std::time::Instant;
use quilting_core::atlas::TessellationAtlas;
use quilting_core::sampling::{tri_patch, PatchConfig};

fn main() {
    let config = PatchConfig { k_candidates: 30, seed: 42 };

    println!("=== Single patch sampling ===");
    for &res in &[16.0, 32.0, 64.0, 128.0, 256.0] {
        let t0 = Instant::now();
        let sample = tri_patch([res, res, res], &config);
        let elapsed = t0.elapsed();
        println!(
            "  res={:>4}: {:>5} points, {:>7.1}ms",
            res as u32, sample.positions.len(), elapsed.as_secs_f64() * 1000.0
        );
    }

    println!("\n=== Atlas build (sequential) ===");
    for max_lod in &[4u32, 5, 6, 7] {
        let lods: Vec<u32> = (0..=*max_lod).map(|n| 1u32 << n).collect();
        let n_triples = count_canonical_triples(&lods);
        let t0 = Instant::now();
        let atlas = TessellationAtlas::build(&lods, &config);
        let elapsed = t0.elapsed();
        println!(
            "  2^0..2^{}: {:>3} triples, {:>6} verts, {:>7.1}ms",
            max_lod, n_triples, atlas.positions.len(),
            elapsed.as_secs_f64() * 1000.0
        );
    }
}

fn count_canonical_triples(lods: &[u32]) -> usize {
    let mut triples = Vec::new();
    for &a in lods {
        for &b in lods {
            for &c in lods {
                let mut k = [a, b, c];
                k.sort();
                if !triples.contains(&k) { triples.push(k); }
            }
        }
    }
    triples.len()
}
