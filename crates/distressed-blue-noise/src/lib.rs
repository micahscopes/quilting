use rand::Rng;
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use std::f64::consts::{PI, SQRT_2};

/// Domain over which to generate samples.
#[derive(Clone, Debug)]
pub enum Domain {
    /// Unit triangle with vertices (0,0), (1,0), (0,1).
    UnitTriangle,
    /// Axis-aligned rectangle from (0,0) to (width, height).
    Rectangle { width: f64, height: f64 },
}

impl Domain {
    fn width(&self) -> f64 {
        match self {
            Domain::UnitTriangle => 1.0,
            Domain::Rectangle { width, .. } => *width,
        }
    }

    fn height(&self) -> f64 {
        match self {
            Domain::UnitTriangle => 1.0,
            Domain::Rectangle { height, .. } => *height,
        }
    }

    /// Returns true if the point lies within the domain.
    #[inline]
    fn contains(&self, p: [f64; 2]) -> bool {
        let [x, y] = p;
        match self {
            Domain::UnitTriangle => x >= 0.0 && y >= 0.0 && (x + y) <= 1.0,
            Domain::Rectangle { width, height } => {
                x >= 0.0 && x <= *width && y >= 0.0 && y <= *height
            }
        }
    }
}

/// Configuration for the Poisson disk sampler.
#[derive(Clone, Debug)]
pub struct SamplerConfig {
    /// Number of candidate points generated per active point each round.
    pub k_candidates: usize,
    /// PRNG seed for deterministic output.
    pub seed: u64,
    /// Sampling domain.
    pub domain: Domain,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            k_candidates: 30,
            seed: 0,
            domain: Domain::Rectangle {
                width: 1.0,
                height: 1.0,
            },
        }
    }
}

/// Variable-density Poisson disk sampler using Bridson's algorithm variant.
pub struct PoissonSampler {
    config: SamplerConfig,
    seed_points: Vec<[f64; 2]>,
}

/// Flat background grid for spatial lookups. Stores point indices.
struct BackgroundGrid {
    cell_size: f64,
    cols: usize,
    rows: usize,
    /// Each cell holds either `usize::MAX` (empty) or a point index.
    cells: Vec<usize>,
}

impl BackgroundGrid {
    fn new(width: f64, height: f64, cell_size: f64) -> Self {
        let cols = ((width / cell_size).ceil() as usize).max(1);
        let rows = ((height / cell_size).ceil() as usize).max(1);
        Self {
            cell_size,
            cols,
            rows,
            cells: vec![usize::MAX; cols * rows],
        }
    }

    #[inline]
    fn cell_index(&self, p: [f64; 2]) -> (usize, usize) {
        let col = ((p[0] / self.cell_size) as usize).min(self.cols - 1);
        let row = ((p[1] / self.cell_size) as usize).min(self.rows - 1);
        (col, row)
    }

    #[inline]
    fn insert(&mut self, p: [f64; 2], idx: usize) {
        let (col, row) = self.cell_index(p);
        self.cells[row * self.cols + col] = idx;
    }

    /// Check if a candidate point conflicts with any existing neighbor.
    /// `density_fn` returns the minimum spacing at a given position.
    /// We reject if dist < max(density(candidate), density(neighbor)).
    #[inline]
    fn conflicts<F: Fn([f64; 2]) -> f64>(
        &self,
        candidate: [f64; 2],
        candidate_radius: f64,
        points: &[[f64; 2]],
        density_fn: &F,
    ) -> bool {
        let (cc, cr) = self.cell_index(candidate);

        // We need to search enough cells to cover the maximum possible radius.
        // Since cell_size = min_radius / sqrt(2), and candidate_radius >= min_radius,
        // we need to search ceil(candidate_radius / cell_size) + 1 cells in each direction.
        let search_radius = (candidate_radius / self.cell_size).ceil() as isize + 1;

        let col_min = (cc as isize - search_radius).max(0) as usize;
        let col_max = ((cc as isize + search_radius) as usize).min(self.cols - 1);
        let row_min = (cr as isize - search_radius).max(0) as usize;
        let row_max = ((cr as isize + search_radius) as usize).min(self.rows - 1);

        for row in row_min..=row_max {
            let row_off = row * self.cols;
            for col in col_min..=col_max {
                let idx = self.cells[row_off + col];
                if idx == usize::MAX {
                    continue;
                }
                let neighbor = points[idx];
                let dx = candidate[0] - neighbor[0];
                let dy = candidate[1] - neighbor[1];
                let dist_sq = dx * dx + dy * dy;
                let min_dist = candidate_radius.max(density_fn(neighbor));
                if dist_sq < min_dist * min_dist {
                    return true;
                }
            }
        }
        false
    }
}

impl PoissonSampler {
    pub fn new(config: SamplerConfig) -> Self {
        Self {
            config,
            seed_points: Vec::new(),
        }
    }

    /// Add pre-seeded boundary points (e.g. for edge stitching between patches).
    pub fn with_seed_points(mut self, points: Vec<[f64; 2]>) -> Self {
        self.seed_points = points;
        self
    }

    /// Run the sampling algorithm. `density_fn` maps a position to the desired
    /// minimum spacing at that point. Higher values produce sparser sampling.
    pub fn sample<F: Fn([f64; 2]) -> f64>(&self, density_fn: F) -> Vec<[f64; 2]> {
        let mut rng = Pcg64Mcg::seed_from_u64(self.config.seed);
        let domain = &self.config.domain;
        let k = self.config.k_candidates;

        // Determine minimum radius across the domain by sampling a grid of probe points
        // plus any seed points. This sets the background grid cell size.
        let min_radius = self.estimate_min_radius(&density_fn);
        let cell_size = min_radius / SQRT_2;

        let w = domain.width();
        let h = domain.height();
        let mut grid = BackgroundGrid::new(w, h, cell_size);

        let mut points: Vec<[f64; 2]> = Vec::new();
        let mut active: Vec<usize> = Vec::new();

        // Insert seed points.
        for &sp in &self.seed_points {
            if domain.contains(sp) {
                let idx = points.len();
                points.push(sp);
                active.push(idx);
                grid.insert(sp, idx);
            }
        }

        // If no seed points, generate an initial point inside the domain.
        if points.is_empty() {
            let p = self.random_initial_point(&mut rng);
            let idx = points.len();
            points.push(p);
            active.push(idx);
            grid.insert(p, idx);
        }

        while !active.is_empty() {
            // Pick a random active point.
            let active_idx = rng.gen_range(0..active.len());
            let point_idx = active[active_idx];
            let point = points[point_idx];
            let r = density_fn(point);

            let mut found = false;
            for _ in 0..k {
                // Generate candidate in annulus [r, 2r].
                let angle = rng.gen::<f64>() * 2.0 * PI;
                let dist = r + rng.gen::<f64>() * r;
                let candidate = [
                    point[0] + dist * angle.cos(),
                    point[1] + dist * angle.sin(),
                ];

                if !domain.contains(candidate) {
                    continue;
                }

                let candidate_r = density_fn(candidate);
                if grid.conflicts(candidate, candidate_r, &points, &density_fn) {
                    continue;
                }

                let idx = points.len();
                points.push(candidate);
                active.push(idx);
                grid.insert(candidate, idx);
                found = true;
                break;
            }

            if !found {
                active.swap_remove(active_idx);
            }
        }

        points
    }

    /// Estimate the minimum radius by probing the density function across the domain.
    fn estimate_min_radius<F: Fn([f64; 2]) -> f64>(&self, density_fn: &F) -> f64 {
        let mut min_r = f64::MAX;

        // Probe seed points.
        for &sp in &self.seed_points {
            min_r = min_r.min(density_fn(sp));
        }

        // Probe a coarse grid across the domain.
        let steps = 16;
        let w = self.config.domain.width();
        let h = self.config.domain.height();
        for iy in 0..=steps {
            for ix in 0..=steps {
                let p = [
                    w * (ix as f64) / (steps as f64),
                    h * (iy as f64) / (steps as f64),
                ];
                if self.config.domain.contains(p) {
                    min_r = min_r.min(density_fn(p));
                }
            }
        }

        assert!(
            min_r > 0.0 && min_r.is_finite(),
            "density function must return positive finite values"
        );
        min_r
    }

    fn random_initial_point(&self, rng: &mut Pcg64Mcg) -> [f64; 2] {
        match &self.config.domain {
            Domain::UnitTriangle => {
                // Uniform sampling in the unit triangle via folding.
                loop {
                    let u: f64 = rng.gen();
                    let v: f64 = rng.gen();
                    if u + v <= 1.0 {
                        return [u, v];
                    }
                }
            }
            Domain::Rectangle { width, height } => {
                [rng.gen::<f64>() * width, rng.gen::<f64>() * height]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_density_point_count() {
        let spacing = 0.05;
        let config = SamplerConfig {
            k_candidates: 30,
            seed: 42,
            domain: Domain::Rectangle {
                width: 1.0,
                height: 1.0,
            },
        };
        let sampler = PoissonSampler::new(config);
        let points = sampler.sample(|_| spacing);

        // Theoretical max packing for radius r in unit square is roughly 1/(pi*r^2 / 2)
        // but Poisson disk is less dense. Expect roughly 0.65-0.85 of hex packing density.
        // Hex packing: 2 / (sqrt(3) * r^2). For r=0.05 that's ~462 points.
        // We expect something in the range of 200-500.
        assert!(
            points.len() > 100,
            "too few points: {}",
            points.len()
        );
        assert!(
            points.len() < 600,
            "too many points: {}",
            points.len()
        );

        // Verify minimum distance constraint.
        for i in 0..points.len() {
            for j in (i + 1)..points.len() {
                let dx = points[i][0] - points[j][0];
                let dy = points[i][1] - points[j][1];
                let dist = (dx * dx + dy * dy).sqrt();
                assert!(
                    dist >= spacing * 0.999,
                    "points {} and {} too close: {:.6} < {:.6}",
                    i,
                    j,
                    dist,
                    spacing
                );
            }
        }
    }

    #[test]
    fn triangle_domain_rejects_exterior() {
        let config = SamplerConfig {
            k_candidates: 30,
            seed: 123,
            domain: Domain::UnitTriangle,
        };
        let sampler = PoissonSampler::new(config);
        let points = sampler.sample(|_| 0.05);

        for p in &points {
            assert!(
                p[0] >= 0.0 && p[1] >= 0.0 && p[0] + p[1] <= 1.0 + 1e-12,
                "point outside unit triangle: {:?}",
                p
            );
        }
        // Should still produce a decent number of points.
        assert!(points.len() > 30, "too few points in triangle: {}", points.len());
    }

    #[test]
    fn seed_points_preserved() {
        let seeds = vec![[0.2, 0.2], [0.5, 0.1], [0.8, 0.8]];
        let config = SamplerConfig {
            k_candidates: 30,
            seed: 99,
            domain: Domain::Rectangle {
                width: 1.0,
                height: 1.0,
            },
        };
        let sampler = PoissonSampler::new(config).with_seed_points(seeds.clone());
        let points = sampler.sample(|_| 0.1);

        for seed in &seeds {
            assert!(
                points.iter().any(|p| p[0] == seed[0] && p[1] == seed[1]),
                "seed point {:?} missing from output",
                seed
            );
        }
    }

    #[test]
    fn deterministic_output() {
        let config = SamplerConfig {
            k_candidates: 30,
            seed: 7,
            domain: Domain::Rectangle {
                width: 1.0,
                height: 1.0,
            },
        };
        let a = PoissonSampler::new(config.clone()).sample(|_| 0.08);
        let b = PoissonSampler::new(config).sample(|_| 0.08);
        assert_eq!(a.len(), b.len(), "different point counts");
        for (pa, pb) in a.iter().zip(b.iter()) {
            assert_eq!(pa, pb, "points differ");
        }
    }

    #[test]
    fn variable_density() {
        // Denser on the left (small spacing), sparser on the right (large spacing).
        let config = SamplerConfig {
            k_candidates: 30,
            seed: 55,
            domain: Domain::Rectangle {
                width: 1.0,
                height: 1.0,
            },
        };
        let sampler = PoissonSampler::new(config);
        let points = sampler.sample(|p| 0.03 + 0.12 * p[0]);

        let left_count = points.iter().filter(|p| p[0] < 0.5).count();
        let right_count = points.iter().filter(|p| p[0] >= 0.5).count();

        // Left side should have significantly more points.
        assert!(
            left_count > right_count,
            "expected more points on left ({}) than right ({})",
            left_count,
            right_count
        );
    }
}
