use rand::Rng;
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use std::f64::consts::SQRT_2;

const SQRT3_OVER_2: f64 = 0.866_025_403_784_438_6;
const ONE_OVER_SQRT3: f64 = 0.577_350_269_189_625_8;

/// Domain over which to generate samples.
#[derive(Clone, Debug)]
pub enum Domain {
    /// Equilateral triangle centered at origin, vertices on the unit circle:
    /// A=(0,1), B=(-√3/2,-1/2), C=(√3/2,-1/2). Side length √3.
    EquilateralTriangle,
    /// Axis-aligned rectangle from (0,0) to (width, height).
    Rectangle { width: f64, height: f64 },
}

impl Domain {
    fn x_min(&self) -> f64 {
        match self {
            Domain::EquilateralTriangle => -SQRT3_OVER_2,
            Domain::Rectangle { .. } => 0.0,
        }
    }

    fn y_min(&self) -> f64 {
        match self {
            Domain::EquilateralTriangle => -0.5,
            Domain::Rectangle { .. } => 0.0,
        }
    }

    fn width(&self) -> f64 {
        match self {
            Domain::EquilateralTriangle => SQRT3_OVER_2 * 2.0,
            Domain::Rectangle { width, .. } => *width,
        }
    }

    fn height(&self) -> f64 {
        match self {
            Domain::EquilateralTriangle => 1.5,
            Domain::Rectangle { height, .. } => *height,
        }
    }

    #[inline(always)]
    fn contains(&self, p: [f64; 2]) -> bool {
        let [x, y] = p;
        match self {
            Domain::EquilateralTriangle => {
                let u3 = 1.0 + 2.0 * y;
                if u3 < 0.0 { return false; }
                let base3 = 1.0 - y;
                let dx3 = x * (ONE_OVER_SQRT3 * 3.0);
                base3 >= dx3 && base3 >= -dx3
            }
            Domain::Rectangle { width, height } => {
                x >= 0.0 && x <= *width && y >= 0.0 && y <= *height
            }
        }
    }
}

/// Configuration for the Poisson disk sampler.
#[derive(Clone, Debug)]
pub struct SamplerConfig {
    pub k_candidates: usize,
    pub seed: u64,
    pub domain: Domain,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            k_candidates: 30,
            seed: 0,
            domain: Domain::Rectangle { width: 1.0, height: 1.0 },
        }
    }
}

/// Variable-density Poisson disk sampler using Bridson's algorithm variant.
pub struct PoissonSampler {
    config: SamplerConfig,
    seed_points: Vec<[f64; 2]>,
}

/// Flat single-entry background grid. Each cell stores one point index
/// (usize::MAX = empty). Radii are cached in a parallel array to avoid
/// recomputing the density function during conflict checks.
struct Grid {
    inv_cell_size: f64,
    x_offset: f64,
    y_offset: f64,
    cols: usize,
    rows: usize,
    cells: Vec<usize>,
}

impl Grid {
    fn new(x_min: f64, y_min: f64, width: f64, height: f64, cell_size: f64) -> Self {
        let cols = ((width / cell_size).ceil() as usize).max(1);
        let rows = ((height / cell_size).ceil() as usize).max(1);
        Self {
            inv_cell_size: 1.0 / cell_size,
            x_offset: x_min,
            y_offset: y_min,
            cols,
            rows,
            cells: vec![usize::MAX; cols * rows],
        }
    }

    #[inline(always)]
    fn cell_xy(&self, p: [f64; 2]) -> (usize, usize) {
        let col = (((p[0] - self.x_offset) * self.inv_cell_size) as usize).min(self.cols - 1);
        let row = (((p[1] - self.y_offset) * self.inv_cell_size) as usize).min(self.rows - 1);
        (col, row)
    }

    #[inline(always)]
    fn insert(&mut self, p: [f64; 2], idx: usize) {
        let (col, row) = self.cell_xy(p);
        self.cells[row * self.cols + col] = idx;
    }

    /// Conflict check using cached radii — no density_fn calls.
    #[inline]
    fn conflicts(
        &self,
        candidate: [f64; 2],
        candidate_radius_sq: f64,
        points: &[[f64; 2]],
        radii_sq: &[f64],
        search: isize,
    ) -> bool {
        let (cc, cr) = self.cell_xy(candidate);

        let col_min = (cc as isize - search).max(0) as usize;
        let col_max = ((cc as isize + search) as usize).min(self.cols - 1);
        let row_min = (cr as isize - search).max(0) as usize;
        let row_max = ((cr as isize + search) as usize).min(self.rows - 1);

        for row in row_min..=row_max {
            let base = row * self.cols;
            let slice = &self.cells[base + col_min..=base + col_max];
            for &idx in slice {
                if idx == usize::MAX { continue; }
                let neighbor = unsafe { *points.get_unchecked(idx) };
                let dx = candidate[0] - neighbor[0];
                let dy = candidate[1] - neighbor[1];
                let dist_sq = dx * dx + dy * dy;
                let min_dist_sq = if candidate_radius_sq > unsafe { *radii_sq.get_unchecked(idx) } {
                    candidate_radius_sq
                } else {
                    unsafe { *radii_sq.get_unchecked(idx) }
                };
                if dist_sq < min_dist_sq {
                    return true;
                }
            }
        }
        false
    }
}

#[inline]
fn random_direction(rng: &mut Pcg64Mcg) -> (f64, f64) {
    loop {
        let x = rng.gen::<f64>() * 2.0 - 1.0;
        let y = rng.gen::<f64>() * 2.0 - 1.0;
        let d2 = x * x + y * y;
        if d2 > 1e-10 && d2 <= 1.0 {
            let inv = 1.0 / d2.sqrt();
            return (x * inv, y * inv);
        }
    }
}

impl PoissonSampler {
    pub fn new(config: SamplerConfig) -> Self {
        Self { config, seed_points: Vec::new() }
    }

    pub fn with_seed_points(mut self, points: Vec<[f64; 2]>) -> Self {
        self.seed_points = points;
        self
    }

    pub fn sample<F: Fn([f64; 2]) -> f64 + Sync>(&self, density_fn: F) -> Vec<[f64; 2]> {
        let mut rng = Pcg64Mcg::seed_from_u64(self.config.seed);
        let domain = &self.config.domain;
        let k = self.config.k_candidates;

        let min_radius = self.estimate_min_radius(&density_fn);
        let cell_size = min_radius / SQRT_2;
        let inv_cell_size = 1.0 / cell_size;

        let mut grid = Grid::new(
            domain.x_min(), domain.y_min(),
            domain.width(), domain.height(),
            cell_size,
        );

        let mut points: Vec<[f64; 2]> = Vec::with_capacity(1024);
        let mut radii: Vec<f64> = Vec::with_capacity(1024);
        let mut radii_sq: Vec<f64> = Vec::with_capacity(1024);
        let mut active: Vec<usize> = Vec::with_capacity(512);

        for &sp in &self.seed_points {
            if domain.contains(sp) {
                let idx = points.len();
                let r = density_fn(sp);
                points.push(sp);
                radii.push(r);
                radii_sq.push(r * r);
                active.push(idx);
                grid.insert(sp, idx);
            }
        }

        if points.is_empty() {
            let p = self.random_initial_point(&mut rng);
            let r = density_fn(p);
            points.push(p);
            radii.push(r);
            radii_sq.push(r * r);
            active.push(0);
            grid.insert(p, 0);
        }

        let min_search = (min_radius * inv_cell_size).ceil() as isize + 1;

        while !active.is_empty() {
            let active_idx = rng.gen_range(0..active.len());
            let point_idx = active[active_idx];
            let point = points[point_idx];
            let r = radii[point_idx];

            let mut found = false;
            for _ in 0..k {
                let (dx, dy) = random_direction(&mut rng);
                let dist = r * (1.0 + rng.gen::<f64>());
                let candidate = [point[0] + dx * dist, point[1] + dy * dist];

                if !domain.contains(candidate) {
                    continue;
                }

                // Quick check with source radius first
                let source_r_sq = radii_sq[point_idx];
                if grid.conflicts(candidate, source_r_sq, &points, &radii_sq, min_search) {
                    continue;
                }

                // Full check with candidate's actual radius
                let candidate_r = density_fn(candidate);
                let candidate_r_sq = candidate_r * candidate_r;
                if candidate_r_sq > source_r_sq {
                    let search = (candidate_r * inv_cell_size).ceil() as isize + 1;
                    if grid.conflicts(candidate, candidate_r_sq, &points, &radii_sq, search) {
                        continue;
                    }
                }

                let idx = points.len();
                points.push(candidate);
                radii.push(candidate_r);
                radii_sq.push(candidate_r_sq);
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

    /// Fast O(N) sampling via jittered hexagonal grid with variable-density
    /// thinning. No spatial index or conflict checking — just a grid walk.
    ///
    /// Produces well-distributed points suitable for Delaunay tessellation.
    /// Much faster than Bridson for large point counts (>10k), but without
    /// the strict minimum-distance guarantee.
    pub fn sample_jittered<F: Fn([f64; 2]) -> f64 + Sync>(
        &self,
        density_fn: F,
    ) -> Vec<[f64; 2]> {
        let mut rng = Pcg64Mcg::seed_from_u64(self.config.seed);
        let domain = &self.config.domain;

        let min_radius = self.estimate_min_radius(&density_fn);

        // Hex grid spacing: rows offset by half a column.
        // Row height = min_radius * sqrt(3)/2 for tight packing.
        let col_step = min_radius;
        let row_step = min_radius * SQRT3_OVER_2;
        let jitter_amount = min_radius * 0.4; // 40% of spacing

        let x0 = domain.x_min();
        let y0 = domain.y_min();
        let x1 = x0 + domain.width();
        let y1 = y0 + domain.height();

        let n_rows = ((y1 - y0) / row_step).ceil() as usize + 1;
        let n_cols = ((x1 - x0) / col_step).ceil() as usize + 1;

        let mut points = Vec::with_capacity(n_rows * n_cols / 2);

        // Insert seed points first (these are not jittered)
        for &sp in &self.seed_points {
            if domain.contains(sp) {
                points.push(sp);
            }
        }

        let inv_min_radius_sq = 1.0 / (min_radius * min_radius);

        for iy in 0..n_rows {
            let y = y0 + iy as f64 * row_step;
            let x_offset = if iy % 2 == 0 { 0.0 } else { col_step * 0.5 };

            for ix in 0..n_cols {
                let x = x0 + ix as f64 * col_step + x_offset;

                if !domain.contains([x, y]) {
                    continue;
                }

                let local_spacing = density_fn([x, y]);

                // Thin: keep point with probability (min_radius / local_spacing)²
                // For uniform density this is always 1.0. For sparse areas, we skip.
                if local_spacing > min_radius {
                    let keep_prob = min_radius * min_radius * inv_min_radius_sq
                        / (local_spacing * local_spacing * inv_min_radius_sq);
                    if rng.gen::<f64>() > keep_prob {
                        continue;
                    }
                }

                // Jitter
                let jx = x + (rng.gen::<f64>() - 0.5) * jitter_amount;
                let jy = y + (rng.gen::<f64>() - 0.5) * jitter_amount;

                if domain.contains([jx, jy]) {
                    points.push([jx, jy]);
                }
            }
        }

        points
    }

    fn estimate_min_radius<F: Fn([f64; 2]) -> f64>(&self, density_fn: &F) -> f64 {
        let mut min_r = f64::MAX;
        for &sp in &self.seed_points {
            min_r = min_r.min(density_fn(sp));
        }
        let steps = 16;
        let domain = &self.config.domain;
        let x0 = domain.x_min();
        let y0 = domain.y_min();
        let w = domain.width();
        let h = domain.height();
        for iy in 0..=steps {
            for ix in 0..=steps {
                let p = [
                    x0 + w * (ix as f64) / (steps as f64),
                    y0 + h * (iy as f64) / (steps as f64),
                ];
                if domain.contains(p) {
                    min_r = min_r.min(density_fn(p));
                }
            }
        }
        assert!(min_r > 0.0 && min_r.is_finite(),
            "density function must return positive finite values");
        min_r
    }

    fn random_initial_point(&self, rng: &mut Pcg64Mcg) -> [f64; 2] {
        match &self.config.domain {
            Domain::EquilateralTriangle => loop {
                let u: f64 = rng.gen();
                let v: f64 = rng.gen();
                if u + v <= 1.0 {
                    let w = 1.0 - u - v;
                    return [
                        SQRT3_OVER_2 * (w - v),
                        (3.0 * u - 1.0) / 2.0,
                    ];
                }
            },
            Domain::Rectangle { width, height } => {
                [rng.gen::<f64>() * width, rng.gen::<f64>() * height]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn equilateral_contains(p: [f64; 2]) -> bool {
        let [x, y] = p;
        let u = (1.0 + 2.0 * y) / 3.0;
        let v = (1.0 - y) / 3.0 - x * ONE_OVER_SQRT3;
        let w = (1.0 - y) / 3.0 + x * ONE_OVER_SQRT3;
        u >= -1e-12 && v >= -1e-12 && w >= -1e-12
    }

    #[test]
    fn uniform_density_point_count() {
        let spacing = 0.05;
        let config = SamplerConfig {
            k_candidates: 30, seed: 42,
            domain: Domain::Rectangle { width: 1.0, height: 1.0 },
        };
        let points = PoissonSampler::new(config).sample(|_| spacing);
        assert!(points.len() > 100, "too few: {}", points.len());
        assert!(points.len() < 600, "too many: {}", points.len());

        for i in 0..points.len() {
            for j in (i + 1)..points.len() {
                let dx = points[i][0] - points[j][0];
                let dy = points[i][1] - points[j][1];
                assert!(dx*dx + dy*dy >= spacing * spacing * 0.998,
                    "points {} and {} too close", i, j);
            }
        }
    }

    #[test]
    fn triangle_domain_rejects_exterior() {
        let config = SamplerConfig {
            k_candidates: 30, seed: 123,
            domain: Domain::EquilateralTriangle,
        };
        let points = PoissonSampler::new(config).sample(|_| 0.05);
        for p in &points {
            assert!(equilateral_contains(*p), "outside triangle: {:?}", p);
        }
        assert!(points.len() > 30, "too few: {}", points.len());
    }

    #[test]
    fn seed_points_preserved() {
        let seeds = vec![[0.2, 0.2], [0.5, 0.1], [0.8, 0.8]];
        let config = SamplerConfig {
            k_candidates: 30, seed: 99,
            domain: Domain::Rectangle { width: 1.0, height: 1.0 },
        };
        let points = PoissonSampler::new(config).with_seed_points(seeds.clone()).sample(|_| 0.1);
        for seed in &seeds {
            assert!(points.iter().any(|p| p[0] == seed[0] && p[1] == seed[1]),
                "seed {:?} missing", seed);
        }
    }

    #[test]
    fn deterministic_output() {
        let config = SamplerConfig {
            k_candidates: 30, seed: 7,
            domain: Domain::Rectangle { width: 1.0, height: 1.0 },
        };
        let a = PoissonSampler::new(config.clone()).sample(|_| 0.08);
        let b = PoissonSampler::new(config).sample(|_| 0.08);
        assert_eq!(a.len(), b.len());
        for (pa, pb) in a.iter().zip(b.iter()) { assert_eq!(pa, pb); }
    }

    #[test]
    fn variable_density() {
        let config = SamplerConfig {
            k_candidates: 30, seed: 55,
            domain: Domain::Rectangle { width: 1.0, height: 1.0 },
        };
        let points = PoissonSampler::new(config).sample(|p| 0.03 + 0.12 * p[0]);
        let left = points.iter().filter(|p| p[0] < 0.5).count();
        let right = points.iter().filter(|p| p[0] >= 0.5).count();
        assert!(left > right, "left {} not > right {}", left, right);
    }

    #[test]
    fn high_density_triangle() {
        let config = SamplerConfig {
            k_candidates: 30, seed: 42,
            domain: Domain::EquilateralTriangle,
        };
        let points = PoissonSampler::new(config).sample(|_| 1.0 / 64.0);
        assert!(points.len() > 500, "too few: {}", points.len());
        for p in &points {
            assert!(equilateral_contains(*p), "outside: {:?}", p);
        }
    }
}
