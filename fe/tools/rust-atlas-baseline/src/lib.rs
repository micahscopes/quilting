//! Measurement-only exports of the EXISTING Rust implementation. Never linked
//! into the Fe atlas demo or its workers. Every run uses an explicit seed.
use quilting_core::{sampling::{tri_patch, PatchConfig}, delaunay::triangulate_2d_constrained};

#[cfg(target_arch = "wasm32")]
fn no_entropy(_: &mut [u8]) -> Result<(), getrandom::Error> {
    Err(getrandom::Error::UNSUPPORTED)
}
#[cfg(target_arch = "wasm32")]
getrandom::register_custom_getrandom!(no_entropy);

/// Low word: point count; high word: triangle count (zero in sampling-only mode).
/// Inputs are exponents. All geometry is discarded before the call returns.
#[no_mangle]
pub extern "C" fn triangle(a: u32, b: u32, c: u32, seed: u32, stage: u32) -> u64 {
    assert!(a <= b && b <= c && c <= 8 && stage <= 1);
    let config = PatchConfig { k_candidates: 30, seed: seed as u64 };
    let sample = tri_patch([(1u32 << a) as f64, (1u32 << b) as f64, (1u32 << c) as f64], &config);
    let points = sample.positions.len() as u64;
    let faces = if stage == 0 { 0 } else {
        triangulate_2d_constrained(&sample.positions, &sample.bary).triangles.len() as u64
    };
    points | (faces << 32)
}
