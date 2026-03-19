use std::time::Instant;
use quilting_core::sampling::{tri_patch, tri_patch_jittered, PatchConfig};

fn main() {
    let config = PatchConfig { k_candidates: 30, seed: 42 };

    println!("=== Bridson (Poisson disk) ===");
    for &res in &[16.0, 32.0, 64.0, 128.0, 256.0, 512.0] {
        let t0 = Instant::now();
        let sample = tri_patch([res, res, res], &config);
        let elapsed = t0.elapsed();
        println!(
            "  res={:>4}: {:>6} points, {:>8.1}ms",
            res as u32, sample.positions.len(), elapsed.as_secs_f64() * 1000.0
        );
    }

    println!("\n=== Jittered hex grid ===");
    for &res in &[16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0] {
        let t0 = Instant::now();
        let sample = tri_patch_jittered([res, res, res], &config);
        let elapsed = t0.elapsed();
        println!(
            "  res={:>4}: {:>6} points, {:>8.1}ms",
            res as u32, sample.positions.len(), elapsed.as_secs_f64() * 1000.0
        );
    }

    println!("\n=== Variable density comparison (2, 2, 256) ===");
    let res = [2.0, 2.0, 256.0];
    let t0 = Instant::now();
    let a = tri_patch(res, &config);
    let ta = t0.elapsed();
    let t0 = Instant::now();
    let b = tri_patch_jittered(res, &config);
    let tb = t0.elapsed();
    println!(
        "  Bridson:  {:>6} points, {:>8.1}ms",
        a.positions.len(), ta.as_secs_f64() * 1000.0
    );
    println!(
        "  Jittered: {:>6} points, {:>8.1}ms",
        b.positions.len(), tb.as_secs_f64() * 1000.0
    );
}
