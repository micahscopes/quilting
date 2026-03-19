use crate::interpolation::tri_edge_weight;
use crate::triangle::{
    self, VERTEX_A, VERTEX_B, VERTEX_C,
};
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

/// Generate sample points for a triangular patch on the equilateral reference
/// triangle (vertices A=(0,1), B=(-√3/2,-1/2), C=(√3/2,-1/2)).
///
/// `res` = [res_a, res_b, res_c]:
///   - res_a = subdivisions on edge BC (opposite vertex A)
///   - res_b = subdivisions on edge AC (opposite vertex B)
///   - res_c = subdivisions on edge AB (opposite vertex C)
pub fn tri_patch(res: [f64; 3], config: &PatchConfig) -> PatchSample {
    let mut seeds = Vec::new();

    // Edge AB (from A to B), resolution res_c (opposite C)
    let n_ab = res[2] as usize;
    for i in 0..=n_ab {
        let t = i as f64 / n_ab as f64;
        seeds.push(triangle::lerp(VERTEX_A, VERTEX_B, t));
    }

    // Edge AC (from A to C), resolution res_b (opposite B)
    let n_ac = res[1] as usize;
    for i in 1..=n_ac {
        let t = i as f64 / n_ac as f64;
        seeds.push(triangle::lerp(VERTEX_A, VERTEX_C, t));
    }

    // Edge BC (from B to C), resolution res_a (opposite A)
    let n_bc = res[0] as usize;
    for i in 1..n_bc {
        let t = i as f64 / n_bc as f64;
        seeds.push(triangle::lerp(VERTEX_B, VERTEX_C, t));
    }

    let n_seeds = seeds.len();

    let sampler = PoissonSampler::new(SamplerConfig {
        k_candidates: config.k_candidates,
        seed: config.seed,
        domain: Domain::EquilateralTriangle,
    })
    .with_seed_points(seeds);

    let density_fn = |p: [f64; 2]| -> f64 {
        let bary = triangle::cartesian_to_bary(p[0], p[1]);
        tri_edge_weight(bary, res)
    };
    let raw_positions = sampler.sample(density_fn);

    // Filter: remove interior points too close to boundary edges.
    // A point's min barycentric coord is proportional to its distance
    // from the nearest edge. Reject if it's within half the local spacing
    // of any edge — these create degenerate slivers in the Delaunay.
    let min_res = res[0].min(res[1]).min(res[2]);
    let edge_margin = 0.4 / min_res; // ~40% of minimum edge spacing in bary

    let mut positions = Vec::with_capacity(raw_positions.len());
    let mut bary = Vec::with_capacity(raw_positions.len());

    for (i, &p) in raw_positions.iter().enumerate() {
        let b = triangle::cartesian_to_bary(p[0], p[1]);

        if i < n_seeds {
            // Seed points (boundary) — always keep
            positions.push(p);
            bary.push(b);
        } else {
            // Interior point — reject if too close to any edge
            let min_bary = b[0].min(b[1]).min(b[2]);
            if min_bary > edge_margin {
                positions.push(p);
                bary.push(b);
            }
        }
    }

    PatchSample { positions, bary }
}

/// Fast patch sampling. Uses O(N) jittered hex grid for uniform density
/// (all edges same resolution), falls back to Bridson for variable density.
pub fn tri_patch_jittered(res: [f64; 3], config: &PatchConfig) -> PatchSample {
    let is_uniform = res[0] == res[1] && res[1] == res[2];

    if !is_uniform {
        // Variable density: Bridson is the only correct approach
        return tri_patch(res, config);
    }

    // Uniform density: regular jittered hex grid, O(N)
    let spacing = 1.0 / res[0];

    let seeds = make_seeds(res);
    let n_seeds = seeds.len();

    let sampler = PoissonSampler::new(SamplerConfig {
        k_candidates: config.k_candidates,
        seed: config.seed,
        domain: Domain::EquilateralTriangle,
    })
    .with_seed_points(seeds);

    let raw_positions = sampler.sample_jittered(|_| spacing);

    let edge_margin = 0.4 / res[0];
    let mut positions = Vec::with_capacity(raw_positions.len());
    let mut bary = Vec::with_capacity(raw_positions.len());

    for (i, &p) in raw_positions.iter().enumerate() {
        let b = triangle::cartesian_to_bary(p[0], p[1]);
        if i < n_seeds {
            positions.push(p);
            bary.push(b);
        } else {
            let min_bary = b[0].min(b[1]).min(b[2]);
            if min_bary > edge_margin {
                positions.push(p);
                bary.push(b);
            }
        }
    }

    PatchSample { positions, bary }
}

fn make_seeds(res: [f64; 3]) -> Vec<[f64; 2]> {
    let mut seeds = Vec::new();
    let n_ab = res[2] as usize;
    for i in 0..=n_ab {
        seeds.push(triangle::lerp(VERTEX_A, VERTEX_B, i as f64 / n_ab as f64));
    }
    let n_ac = res[1] as usize;
    for i in 1..=n_ac {
        seeds.push(triangle::lerp(VERTEX_A, VERTEX_C, i as f64 / n_ac as f64));
    }
    let n_bc = res[0] as usize;
    for i in 1..n_bc {
        seeds.push(triangle::lerp(VERTEX_B, VERTEX_C, i as f64 / n_bc as f64));
    }
    seeds
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
            let [u, v, w] = triangle::cartesian_to_bary(x, y);
            assert!(
                u >= -1e-10 && v >= -1e-10 && w >= -1e-10,
                "point {} = [{}, {}] (bary [{}, {}, {}]) outside equilateral triangle",
                i, x, y, u, v, w
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
                i, u, v, w
            );
            let back = triangle::bary_to_cartesian([u, v, w]);
            assert!(
                (back[0] - x).abs() < 1e-10 && (back[1] - y).abs() < 1e-10,
                "bary roundtrip mismatch at point {}: [{},{}] vs [{},{}]",
                i, x, y, back[0], back[1]
            );
        }
    }
}
