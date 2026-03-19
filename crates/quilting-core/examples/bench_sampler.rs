use std::time::Instant;
use quilting_core::atlas::{TessellationAtlas, BuildMode};
use quilting_core::sampling::PatchConfig;

fn main() {
    let config = PatchConfig { k_candidates: 30, seed: 42 };

    for &max_exp in &[7u32, 8, 9] {
        let lods: Vec<u32> = (0..=max_exp).map(|n| 1u32 << n).collect();
        let n = canonical_triple_count(&lods);

        let t0 = Instant::now();
        let direct = TessellationAtlas::build_with_mode(&lods, &config, BuildMode::Direct);
        let td = t0.elapsed();

        let t0 = Instant::now();
        let hier = TessellationAtlas::build_with_mode(&lods, &config, BuildMode::Hierarchical);
        let th = t0.elapsed();

        println!(
            "2^0..2^{}: {} triples | direct: {:>6} verts {:>7.0}ms | hier: {:>6} verts {:>7.0}ms | {:.1}x",
            max_exp, n,
            direct.positions.len(), td.as_secs_f64() * 1000.0,
            hier.positions.len(), th.as_secs_f64() * 1000.0,
            td.as_secs_f64() / th.as_secs_f64(),
        );
    }
}

fn canonical_triple_count(lods: &[u32]) -> usize {
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
