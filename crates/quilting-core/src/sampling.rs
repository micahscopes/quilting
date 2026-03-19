use crate::interpolation::tri_edge_weight;
use distressed_blue_noise::{Domain, PoissonSampler, SamplerConfig};

pub struct PatchConfig {
    pub k_candidates: usize,
    pub seed: u64,
}

impl Default for PatchConfig {
    fn default() -> Self {
        Self {
            k_candidates: 30,
            seed: 42,
        }
    }
}

pub struct PatchSample {
    pub positions: Vec<[f64; 2]>,
    pub bary: Vec<[f64; 3]>,
}

/// Generate sample points for a triangular patch.
///
/// Unit triangle: A=(0,0), B=(1,0), C=(0,1).
///
/// `res` = [res_a, res_b, res_c]:
///   - res_a = subdivisions on edge BC
///   - res_b = subdivisions on edge AC
///   - res_c = subdivisions on edge AB
pub fn tri_patch(res: [f64; 3], config: &PatchConfig) -> PatchSample {
    let mut seeds = Vec::new();

    // Edge AB (from A=(0,0) to B=(1,0)), resolution res_c
    let n_ab = res[2] as usize;
    for i in 0..=n_ab {
        let t = i as f64 / n_ab as f64;
        seeds.push([t, 0.0]);
    }

    // Edge AC (from A=(0,0) to C=(0,1)), resolution res_b
    let n_ac = res[1] as usize;
    for i in 1..=n_ac {
        // skip (0,0), already added
        let t = i as f64 / n_ac as f64;
        seeds.push([0.0, t]);
    }

    // Edge BC (from B=(1,0) to C=(0,1)), resolution res_a
    let n_bc = res[0] as usize;
    for i in 1..n_bc {
        // skip endpoints, already added
        let t = i as f64 / n_bc as f64;
        seeds.push([1.0 - t, t]);
    }

    let sampler = PoissonSampler::new(SamplerConfig {
        k_candidates: config.k_candidates,
        seed: config.seed,
        domain: Domain::UnitTriangle,
    })
    .with_seed_points(seeds);

    let density_fn = |p: [f64; 2]| -> f64 {
        let u = 1.0 - p[0] - p[1]; // bary coord for A
        let v = p[0]; // bary coord for B
        let w = p[1]; // bary coord for C
        tri_edge_weight([u, v, w], res)
    };

    let positions = sampler.sample(density_fn);
    let bary: Vec<[f64; 3]> = positions.iter().map(|&[x, y]| [1.0 - x - y, x, y]).collect();

    PatchSample { positions, bary }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tri_patch_produces_points() {
        let config = PatchConfig::default();
        let sample = tri_patch([4.0, 4.0, 4.0], &config);
        assert!(
            sample.positions.len() >= 3,
            "expected at least 3 points, got {}",
            sample.positions.len()
        );
        assert_eq!(sample.positions.len(), sample.bary.len());
    }

    #[test]
    fn tri_patch_all_inside_triangle() {
        let config = PatchConfig::default();
        let sample = tri_patch([8.0, 8.0, 8.0], &config);
        for (i, &[x, y]) in sample.positions.iter().enumerate() {
            assert!(
                x >= -1e-12 && y >= -1e-12 && x + y <= 1.0 + 1e-12,
                "point {} = [{}, {}] outside unit triangle",
                i,
                x,
                y
            );
        }
    }

    #[test]
    fn tri_patch_bary_consistent() {
        let config = PatchConfig::default();
        let sample = tri_patch([4.0, 6.0, 8.0], &config);
        for (i, (&[x, y], &[u, v, w])) in
            sample.positions.iter().zip(sample.bary.iter()).enumerate()
        {
            assert!(
                (u + v + w - 1.0).abs() < 1e-10,
                "barycentric coords don't sum to 1 at point {}: [{}, {}, {}]",
                i,
                u,
                v,
                w
            );
            assert!(
                (v - x).abs() < 1e-10 && (w - y).abs() < 1e-10,
                "bary mismatch at point {}",
                i
            );
        }
    }
}
