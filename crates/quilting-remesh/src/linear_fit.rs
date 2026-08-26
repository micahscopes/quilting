//! Direct linear global fit of shared per-vertex QB weights.
//!
//! The other curved fitters in this crate ([`crate::c_estimator`],
//! [`crate::global_fit`]) treat weight fitting as a nonlinear least-squares
//! problem and lean on Gauss-Newton. This module rests on a sharper
//! observation: the *algebraic* patch residual is **linear** in the weights, so
//! one sparse linear least-squares solve recovers a globally-consistent,
//! C0-watertight weight per vertex — no nonlinear initialization sensitivity.
//!
//! ## The residual is linear in the weights
//!
//! A QB triangle patch evaluates to `X(λ) = Im[ top(λ) · bot(λ)⁻¹ ]` with
//! `top(λ) = Σ λᵢ pᵢ wᵢ` and `bot(λ) = Σ λᵢ wᵢ`. For a sample point `Xₖ` known
//! to sit at barycentric `λₖ` on the face `(a,b,c)`, define the algebraic
//! residual
//!
//! ```text
//!   rₖ = top(λₖ) − Xₖ · bot(λₖ)
//!      = Σ_{v∈{a,b,c}} λ_{k,v} (pᵥ − Xₖ) wᵥ .
//! ```
//!
//! `rₖ = 0` is *exactly* `X(λₖ) = Xₖ` (multiply on the right by `bot⁻¹`). Because
//! quaternion left-multiplication `q · w` is a linear map on the four real
//! components of `w` — the 4×4 matrix [`left_mul_matrix`] — `rₖ` is linear in the
//! stacked vector of all `wᵥ`. Stacking every sample gives an overdetermined
//! homogeneous system `A w = 0`.
//!
//! ## Killing the trivial and gauge nullspaces
//!
//! `w = 0` solves `A w = 0`, and so does any right-multiplication `wᵥ ↦ wᵥ·s`
//! (both `top` and `bot` pick up the same right factor `s`, which cancels in
//! `top·bot⁻¹`). Both are removed by **pinning** one vertex weight per connected
//! component to `Quat::ONE`: those contributions move to the right-hand side
//! `b`, leaving a `4·(V−C)` unknown system `A_free w_free = b` for `C`
//! components. A small Tikhonov pull of every free weight toward `ONE`
//! regularizes vertices that are starved of samples.
//! The augmented least-squares system is column-equilibrated and solved with
//! sparse LSQR. Each sample row touches at most twelve unknowns, so preserving
//! that structure matters once the coarse complex grows beyond a toy. We
//! deliberately do not form `AᵀA`: doing so squares the condition number
//! precisely where a rational denominator fit is most vulnerable.
//!
//! Sharing one weight per vertex is what makes the result watertight: two faces
//! meeting on an edge share that edge's two endpoint weights, and a QB edge curve
//! depends only on its two endpoints, so the boundary curves coincide to machine
//! precision (see [`crate::linear_fit`] tests and the `curved_vs_flat` example).

use quilting_core::patch::QBTriPatch;
use quilting_core::quaternion::Quat;
use quilting_core::screen_domain::denominator_norm_sq_range;

/// A target sample: a 3D point known to lie at barycentric `bary` on face
/// `face_index` of the coarse mesh.
#[derive(Debug, Clone, Copy)]
pub struct Sample {
    /// Index into the coarse `faces` array.
    pub face_index: usize,
    /// Affine barycentric coordinates `[λ₀, λ₁, λ₂]` matching the face's vertex
    /// order. They must sum to one; a correspondence sampler may use modest
    /// negative coordinates when projecting just outside a coarse boundary.
    pub bary: [f64; 3],
    /// Target 3D position the patch should pass through at `bary`.
    pub target: [f64; 3],
}

/// Tuning for [`linear_global_fit_full`].
#[derive(Debug, Clone, Copy)]
pub struct LinearFitConfig {
    /// Tikhonov weight pulling every free vertex weight toward `Quat::ONE`.
    /// Coordinates are normalized by the complete fit extent first, so this is
    /// dimensionless. Keep it tiny so it only breaks ties / stabilizes
    /// under-sampled vertices rather than biasing the fit toward flat.
    pub tikhonov: f64,
    /// Reject a fitted patch when its exact minimum affine-denominator norm,
    /// divided by the largest corner-weight norm, falls below this value.
    pub relative_denominator_epsilon: f64,
    /// Maximum sparse LSQR iterations.
    pub max_iterations: usize,
    /// Relative normal-residual tolerance used to accept LSQR convergence.
    pub solver_tolerance: f64,
}

impl Default for LinearFitConfig {
    fn default() -> Self {
        Self {
            tikhonov: 1e-8,
            relative_denominator_epsilon: 1e-5,
            max_iterations: 2_048,
            solver_tolerance: 1e-10,
        }
    }
}

/// Full result of the linear global fit, including diagnostics.
#[derive(Debug)]
pub struct LinearFitResult {
    /// One fitted QB patch per coarse face.
    pub patches: Vec<QBTriPatch>,
    /// One solved quaternion weight per coarse vertex. The lowest-index vertex
    /// in each connected component is pinned to `ONE`.
    pub vertex_weights: Vec<Quat>,
    /// `max_v |wᵥ − ONE|`. If ≈ 0 the solve collapsed to flat and the "curved"
    /// result is a lie; a healthy curved fit reports a clearly nonzero value.
    pub max_weight_dev: f64,
    /// Number of free unknowns (`4·(V−C)` for `C` connected components).
    pub dim: usize,
    /// Sparse LSQR iterations used by the accepted solution.
    pub solver_iterations: usize,
    /// `||Aᵀ(Ax-b)|| / ||Aᵀb||` in the equilibrated augmented system.
    pub relative_normal_residual: f64,
    /// Per-sample quaternion RMS algebraic residual in normalized coordinate
    /// units, excluding regularization. This is gauge-dependent and is not a
    /// geometric RMS distance.
    pub algebraic_residual_rms: f64,
    /// Worst exact relative affine-denominator minimum over output patches.
    pub min_relative_denominator: f64,
}

/// Structural or numerical failure in the shared global QB fit.
#[derive(Debug, Clone, PartialEq)]
pub enum LinearFitError {
    EmptyPositions,
    EmptyFaces,
    EmptySamples,
    NonFinitePosition {
        vertex: usize,
    },
    InvalidFaceVertex {
        face: usize,
        vertex: usize,
    },
    DegenerateFace {
        face: usize,
    },
    InvalidSampleFace {
        sample: usize,
        face: usize,
    },
    NonFiniteSample {
        sample: usize,
    },
    InvalidBarycentric {
        sample: usize,
    },
    InvalidRegularization(f64),
    InvalidDenominatorThreshold(f64),
    InvalidSolverTolerance(f64),
    InvalidMaxIterations,
    CountOverflow,
    SolverDidNotConverge {
        iterations: usize,
        relative_normal_residual: f64,
    },
    NonFiniteSolution,
    IllConditionedPatch {
        face: usize,
        min_relative_denominator: f64,
    },
}

impl std::fmt::Display for LinearFitError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyPositions => write!(formatter, "shared QB fit has no coarse vertices"),
            Self::EmptyFaces => write!(formatter, "shared QB fit has no coarse faces"),
            Self::EmptySamples => write!(formatter, "shared QB fit has no source samples"),
            Self::NonFinitePosition { vertex } => {
                write!(formatter, "coarse vertex {vertex} is non-finite")
            }
            Self::InvalidFaceVertex { face, vertex } => {
                write!(formatter, "coarse face {face} references missing vertex {vertex}")
            }
            Self::DegenerateFace { face } => {
                write!(formatter, "coarse face {face} repeats a vertex")
            }
            Self::InvalidSampleFace { sample, face } => {
                write!(formatter, "fit sample {sample} references missing face {face}")
            }
            Self::NonFiniteSample { sample } => {
                write!(formatter, "fit sample {sample} is non-finite")
            }
            Self::InvalidBarycentric { sample } => {
                write!(formatter, "fit sample {sample} barycentrics do not sum to one")
            }
            Self::InvalidRegularization(value) => write!(
                formatter,
                "shared QB regularization must be finite and positive, got {value}",
            ),
            Self::InvalidDenominatorThreshold(value) => write!(
                formatter,
                "shared QB denominator threshold must be finite and positive, got {value}",
            ),
            Self::InvalidSolverTolerance(value) => write!(
                formatter,
                "shared QB solver tolerance must be finite and positive, got {value}",
            ),
            Self::InvalidMaxIterations => {
                write!(formatter, "shared QB solver iteration limit must be positive")
            }
            Self::CountOverflow => write!(formatter, "shared QB fit dimensions overflow usize"),
            Self::SolverDidNotConverge {
                iterations,
                relative_normal_residual,
            } => write!(
                formatter,
                "shared QB fit did not converge in {iterations} iterations (relative normal residual {relative_normal_residual:e})",
            ),
            Self::NonFiniteSolution => {
                write!(formatter, "shared QB fit produced non-finite weights")
            }
            Self::IllConditionedPatch {
                face,
                min_relative_denominator,
            } => write!(
                formatter,
                "shared QB patch {face} has relative denominator minimum {min_relative_denominator:e}",
            ),
        }
    }
}

impl std::error::Error for LinearFitError {}

/// Convenience wrapper matching the assessment's signature: fit and return just
/// the patches. Uses [`LinearFitConfig::default`].
pub fn linear_global_fit(
    coarse_pos: &[[f64; 3]],
    coarse_faces: &[[usize; 3]],
    samples: &[Sample],
) -> Result<Vec<QBTriPatch>, LinearFitError> {
    Ok(linear_global_fit_full(
        coarse_pos,
        coarse_faces,
        samples,
        &LinearFitConfig::default(),
    )?
    .patches)
}

/// The full fit. See the module docs for the derivation.
pub fn linear_global_fit_full(
    coarse_pos: &[[f64; 3]],
    coarse_faces: &[[usize; 3]],
    samples: &[Sample],
    config: &LinearFitConfig,
) -> Result<LinearFitResult, LinearFitError> {
    if coarse_pos.is_empty() {
        return Err(LinearFitError::EmptyPositions);
    }
    if coarse_faces.is_empty() {
        return Err(LinearFitError::EmptyFaces);
    }
    if samples.is_empty() {
        return Err(LinearFitError::EmptySamples);
    }
    if !config.tikhonov.is_finite() || config.tikhonov <= 0.0 {
        return Err(LinearFitError::InvalidRegularization(config.tikhonov));
    }
    if !config.relative_denominator_epsilon.is_finite()
        || config.relative_denominator_epsilon <= 0.0
    {
        return Err(LinearFitError::InvalidDenominatorThreshold(
            config.relative_denominator_epsilon,
        ));
    }
    if !config.solver_tolerance.is_finite() || config.solver_tolerance <= 0.0 {
        return Err(LinearFitError::InvalidSolverTolerance(
            config.solver_tolerance,
        ));
    }
    if config.max_iterations == 0 {
        return Err(LinearFitError::InvalidMaxIterations);
    }
    for (vertex, position) in coarse_pos.iter().enumerate() {
        if position.iter().any(|component| !component.is_finite()) {
            return Err(LinearFitError::NonFinitePosition { vertex });
        }
    }
    for (face_index, face) in coarse_faces.iter().enumerate() {
        if face[0] == face[1] || face[1] == face[2] || face[2] == face[0] {
            return Err(LinearFitError::DegenerateFace { face: face_index });
        }
        for &vertex in face {
            if vertex >= coarse_pos.len() {
                return Err(LinearFitError::InvalidFaceVertex {
                    face: face_index,
                    vertex,
                });
            }
        }
    }
    for (sample_index, sample) in samples.iter().enumerate() {
        if sample.face_index >= coarse_faces.len() {
            return Err(LinearFitError::InvalidSampleFace {
                sample: sample_index,
                face: sample.face_index,
            });
        }
        if sample
            .bary
            .iter()
            .chain(sample.target.iter())
            .any(|component| !component.is_finite())
        {
            return Err(LinearFitError::NonFiniteSample {
                sample: sample_index,
            });
        }
        let barycentric_sum = sample.bary.iter().sum::<f64>();
        if (barycentric_sum - 1.0).abs() > 1e-9 {
            return Err(LinearFitError::InvalidBarycentric {
                sample: sample_index,
            });
        }
    }

    let v = coarse_pos.len();
    // Pin the lowest vertex in every connected component to ONE. Each component
    // has an independent right-quaternion gauge; pinning only global vertex 0
    // leaves disconnected components to the regularizer and biases their fit.
    let free_indices = free_vertex_indices(v, coarse_faces);
    let free = |vertex: usize| free_indices[vertex];
    let free_vertices = free_indices.iter().filter(|index| index.is_some()).count();
    let dim = free_vertices
        .checked_mul(4)
        .ok_or(LinearFitError::CountOverflow)?;
    let data_rows = samples
        .len()
        .checked_mul(4)
        .ok_or(LinearFitError::CountOverflow)?;
    let rows = data_rows
        .checked_add(dim)
        .ok_or(LinearFitError::CountOverflow)?;
    let data_nonzeros = samples
        .len()
        .checked_mul(4)
        .and_then(|value| value.checked_mul(12))
        .ok_or(LinearFitError::CountOverflow)?;
    let nonzero_capacity = data_nonzeros
        .checked_add(dim)
        .ok_or(LinearFitError::CountOverflow)?;
    let mut row_offsets =
        Vec::with_capacity(rows.checked_add(1).ok_or(LinearFitError::CountOverflow)?);
    let mut column_indices = Vec::with_capacity(nonzero_capacity);
    let mut values = Vec::with_capacity(nonzero_capacity);
    let mut rhs = Vec::with_capacity(rows);
    row_offsets.push(0);
    let inverse_coordinate_scale = fit_coordinate_scale(coarse_pos, samples).recip();

    for s in samples {
        let face = coarse_faces[s.face_index];
        let xk = Quat::from_point(s.target[0], s.target[1], s.target[2]);

        // Per-vertex 4×4 blocks Bᵥ = λ_{k,v} · L(pᵥ − Xₖ).
        let mut blocks: [[[f64; 4]; 4]; 3] = [[[0.0; 4]; 4]; 3];
        for i in 0..3 {
            let vtx = face[i];
            let p = Quat::from_point(coarse_pos[vtx][0], coarse_pos[vtx][1], coarse_pos[vtx][2]);
            let l = left_mul_matrix(p - xk);
            let lam = s.bary[i];
            for r in 0..4 {
                for c in 0..4 {
                    blocks[i][r][c] = lam * l[r][c];
                }
            }
        }

        // Pinned contribution → right-hand side. w₀ = vec(ONE) = [1,0,0,0], so
        // B₀·vec(ONE) is just the first column of B₀. The equation Σ Bᵥ wᵥ = 0
        // becomes Σ_{free} Bᵥ wᵥ = d with d = −(pinned column).
        let mut d = [0.0f64; 4];
        for i in 0..3 {
            if free(face[i]).is_none() {
                for r in 0..4 {
                    d[r] -= blocks[i][r][0];
                }
            }
        }

        // Write the four real residual equations directly. Each row touches at
        // most three quaternion weights (twelve scalar columns).
        for residual_component in 0..4 {
            rhs.push(d[residual_component] * inverse_coordinate_scale);
            for local_vertex in 0..3 {
                let Some(free_vertex) = free(face[local_vertex]) else {
                    continue;
                };
                for (weight_component, coefficient) in
                    blocks[local_vertex][residual_component].iter().enumerate()
                {
                    let column = free_vertex * 4 + weight_component;
                    let value = coefficient * inverse_coordinate_scale;
                    if value != 0.0 {
                        column_indices.push(column);
                        values.push(value);
                    }
                }
            }
            row_offsets.push(column_indices.len());
        }
    }

    // Tikhonov as extra least-squares rows: sqrt(τ)·x ≈ sqrt(τ)·ONE.
    // This is algebraically equivalent to adding τI, without forming AᵀA.
    let regularization = config.tikhonov.sqrt();
    for column in 0..dim {
        column_indices.push(column);
        values.push(regularization);
        rhs.push(if column % 4 == 0 { regularization } else { 0.0 });
        row_offsets.push(column_indices.len());
    }

    let matrix = SparseLeastSquares {
        rows,
        columns: dim,
        row_offsets,
        column_indices,
        values,
    };
    let solution =
        solve_least_squares_lsqr(matrix, &rhs, config.max_iterations, config.solver_tolerance)?;
    if solution.values.iter().any(|value| !value.is_finite()) {
        return Err(LinearFitError::NonFiniteSolution);
    }

    // Reconstruct per-vertex weights.
    let mut weights = vec![Quat::ONE; v];
    for (vertex, weight) in weights.iter_mut().enumerate() {
        if let Some(free_vertex) = free(vertex) {
            *weight = Quat::new(
                solution.values[4 * free_vertex],
                solution.values[4 * free_vertex + 1],
                solution.values[4 * free_vertex + 2],
                solution.values[4 * free_vertex + 3],
            );
        }
    }

    // Diagnostic: how far the weights actually moved from flat.
    let max_weight_dev = weights
        .iter()
        .map(|w| (*w - Quat::ONE).norm())
        .fold(0.0, f64::max);

    // Reconstruct patches.
    let patches = coarse_faces
        .iter()
        .map(|face| {
            let positions = [
                Quat::from_point(
                    coarse_pos[face[0]][0],
                    coarse_pos[face[0]][1],
                    coarse_pos[face[0]][2],
                ),
                Quat::from_point(
                    coarse_pos[face[1]][0],
                    coarse_pos[face[1]][1],
                    coarse_pos[face[1]][2],
                ),
                Quat::from_point(
                    coarse_pos[face[2]][0],
                    coarse_pos[face[2]][1],
                    coarse_pos[face[2]][2],
                ),
            ];
            QBTriPatch::new(
                positions,
                [weights[face[0]], weights[face[1]], weights[face[2]]],
            )
        })
        .collect::<Vec<_>>();

    let algebraic_residual_rms =
        algebraic_residual_rms(coarse_pos, coarse_faces, samples, &weights)
            * inverse_coordinate_scale;
    let min_relative_denominator =
        validate_patch_denominators(&patches, config.relative_denominator_epsilon)?;

    Ok(LinearFitResult {
        patches,
        vertex_weights: weights,
        max_weight_dev,
        dim,
        solver_iterations: solution.iterations,
        relative_normal_residual: solution.relative_normal_residual,
        algebraic_residual_rms,
        min_relative_denominator,
    })
}

fn fit_coordinate_scale(coarse_pos: &[[f64; 3]], samples: &[Sample]) -> f64 {
    let mut minimum = [f64::INFINITY; 3];
    let mut maximum = [f64::NEG_INFINITY; 3];
    for point in coarse_pos
        .iter()
        .chain(samples.iter().map(|sample| &sample.target))
    {
        for axis in 0..3 {
            minimum[axis] = minimum[axis].min(point[axis]);
            maximum[axis] = maximum[axis].max(point[axis]);
        }
    }
    let extent = (maximum[0] - minimum[0])
        .hypot(maximum[1] - minimum[1])
        .hypot(maximum[2] - minimum[2]);
    if extent.is_finite() && extent > f64::MIN_POSITIVE {
        extent
    } else {
        1.0
    }
}

fn free_vertex_indices(vertex_count: usize, faces: &[[usize; 3]]) -> Vec<Option<usize>> {
    fn root(parents: &mut [usize], mut vertex: usize) -> usize {
        while parents[vertex] != vertex {
            parents[vertex] = parents[parents[vertex]];
            vertex = parents[vertex];
        }
        vertex
    }

    let mut parents = (0..vertex_count).collect::<Vec<_>>();
    for face in faces {
        for [from, to] in [[face[0], face[1]], [face[1], face[2]]] {
            let from_root = root(&mut parents, from);
            let to_root = root(&mut parents, to);
            if from_root != to_root {
                let minimum = from_root.min(to_root);
                let maximum = from_root.max(to_root);
                parents[maximum] = minimum;
            }
        }
    }
    let roots = (0..vertex_count)
        .map(|vertex| root(&mut parents, vertex))
        .collect::<Vec<_>>();
    let mut pinned = vec![usize::MAX; vertex_count];
    for (vertex, &component) in roots.iter().enumerate() {
        pinned[component] = pinned[component].min(vertex);
    }
    let mut next_free = 0usize;
    roots
        .iter()
        .enumerate()
        .map(|(vertex, &component)| {
            if pinned[component] == vertex {
                None
            } else {
                let index = next_free;
                next_free += 1;
                Some(index)
            }
        })
        .collect()
}

fn validate_patch_denominators(
    patches: &[QBTriPatch],
    relative_epsilon: f64,
) -> Result<f64, LinearFitError> {
    let mut worst_relative = f64::INFINITY;
    for (face, patch) in patches.iter().enumerate() {
        let weight_scale = patch
            .weights
            .iter()
            .map(|weight| weight.norm())
            .fold(0.0f64, f64::max);
        let [minimum_squared, _] = denominator_norm_sq_range(patch);
        let relative = minimum_squared.sqrt() / weight_scale;
        if !relative.is_finite() || relative < relative_epsilon {
            return Err(LinearFitError::IllConditionedPatch {
                face,
                min_relative_denominator: relative,
            });
        }
        worst_relative = worst_relative.min(relative);
    }
    Ok(worst_relative)
}

struct LeastSquaresSolution {
    values: Vec<f64>,
    iterations: usize,
    relative_normal_residual: f64,
}

#[derive(Debug)]
struct SparseLeastSquares {
    rows: usize,
    columns: usize,
    row_offsets: Vec<usize>,
    column_indices: Vec<usize>,
    values: Vec<f64>,
}

impl SparseLeastSquares {
    fn multiply(&self, input: &[f64], output: &mut [f64]) {
        debug_assert_eq!(input.len(), self.columns);
        debug_assert_eq!(output.len(), self.rows);
        for (row, output_value) in output.iter_mut().enumerate() {
            *output_value = (self.row_offsets[row]..self.row_offsets[row + 1])
                .map(|entry| self.values[entry] * input[self.column_indices[entry]])
                .sum();
        }
    }

    fn multiply_transpose(&self, input: &[f64], output: &mut [f64]) {
        debug_assert_eq!(input.len(), self.rows);
        debug_assert_eq!(output.len(), self.columns);
        output.fill(0.0);
        for (row, &input_value) in input.iter().enumerate() {
            for entry in self.row_offsets[row]..self.row_offsets[row + 1] {
                output[self.column_indices[entry]] += self.values[entry] * input_value;
            }
        }
    }

    fn equilibrate_columns(&mut self) -> Result<Vec<f64>, LinearFitError> {
        let mut column_scales = vec![0.0f64; self.columns];
        for (&column, &value) in self.column_indices.iter().zip(&self.values) {
            if column >= self.columns || !value.is_finite() {
                return Err(LinearFitError::NonFiniteSolution);
            }
            column_scales[column] = column_scales[column].hypot(value);
        }
        if column_scales
            .iter()
            .any(|scale| !scale.is_finite() || *scale == 0.0)
        {
            return Err(LinearFitError::NonFiniteSolution);
        }
        for (entry, value) in self.values.iter_mut().enumerate() {
            *value /= column_scales[self.column_indices[entry]];
        }
        Ok(column_scales)
    }
}

/// Solve an overdetermined sparse system with column equilibration and LSQR.
///
/// The recurrence estimate alone is not used as an acceptance gate: every
/// eighth iteration (and at termination) we explicitly recompute
/// `||Aᵀ(Ax-b)|| / ||Aᵀb||` in the equilibrated augmented system.
fn solve_least_squares_lsqr(
    mut matrix: SparseLeastSquares,
    rhs: &[f64],
    max_iterations: usize,
    tolerance: f64,
) -> Result<LeastSquaresSolution, LinearFitError> {
    if matrix.rows < matrix.columns
        || matrix.row_offsets.len() != matrix.rows + 1
        || matrix.row_offsets.first() != Some(&0)
        || matrix.row_offsets.last() != Some(&matrix.values.len())
        || matrix.column_indices.len() != matrix.values.len()
        || rhs.len() != matrix.rows
        || max_iterations == 0
        || !tolerance.is_finite()
        || tolerance <= 0.0
    {
        return Err(LinearFitError::CountOverflow);
    }
    if rhs.iter().any(|value| !value.is_finite()) {
        return Err(LinearFitError::NonFiniteSolution);
    }
    let column_scales = matrix.equilibrate_columns()?;

    let mut u = rhs.to_vec();
    let mut beta = vector_norm(&u);
    if !beta.is_finite() || beta == 0.0 {
        return Err(LinearFitError::NonFiniteSolution);
    }
    scale_vector(&mut u, beta.recip());

    let mut normal_rhs = vec![0.0; matrix.columns];
    matrix.multiply_transpose(rhs, &mut normal_rhs);
    let normal_rhs_norm = vector_norm(&normal_rhs);
    if !normal_rhs_norm.is_finite() || normal_rhs_norm == 0.0 {
        return Err(LinearFitError::NonFiniteSolution);
    }

    let mut v = vec![0.0; matrix.columns];
    matrix.multiply_transpose(&u, &mut v);
    let mut alpha = vector_norm(&v);
    if !alpha.is_finite() || alpha == 0.0 {
        return Err(LinearFitError::NonFiniteSolution);
    }
    scale_vector(&mut v, alpha.recip());

    let mut solution = vec![0.0; matrix.columns];
    let mut direction = v.clone();
    let mut next_u = vec![0.0; matrix.rows];
    let mut next_v = vec![0.0; matrix.columns];
    let mut rho_bar = alpha;
    let mut phi_bar = beta;
    let mut relative_normal_residual = 1.0;
    let mut accepted_iteration = 0usize;
    let mut final_iteration = 0usize;

    for iteration in 1..=max_iterations {
        final_iteration = iteration;
        matrix.multiply(&v, &mut next_u);
        for (next, previous) in next_u.iter_mut().zip(&u) {
            *next -= alpha * previous;
        }
        beta = vector_norm(&next_u);
        if !beta.is_finite() {
            return Err(LinearFitError::NonFiniteSolution);
        }
        if beta > 0.0 {
            scale_vector(&mut next_u, beta.recip());
        }
        std::mem::swap(&mut u, &mut next_u);

        matrix.multiply_transpose(&u, &mut next_v);
        for (next, previous) in next_v.iter_mut().zip(&v) {
            *next -= beta * previous;
        }
        alpha = vector_norm(&next_v);
        if !alpha.is_finite() {
            return Err(LinearFitError::NonFiniteSolution);
        }
        if alpha > 0.0 {
            scale_vector(&mut next_v, alpha.recip());
        }
        std::mem::swap(&mut v, &mut next_v);

        let rho = rho_bar.hypot(beta);
        if !rho.is_finite() || rho == 0.0 {
            return Err(LinearFitError::NonFiniteSolution);
        }
        let cosine = rho_bar / rho;
        let sine = beta / rho;
        let theta = sine * alpha;
        rho_bar = -cosine * alpha;
        let phi = cosine * phi_bar;
        phi_bar *= sine;
        let solution_step = phi / rho;
        let direction_step = -theta / rho;
        for column in 0..matrix.columns {
            solution[column] += solution_step * direction[column];
            direction[column] = v[column] + direction_step * direction[column];
        }
        if solution.iter().any(|value| !value.is_finite()) {
            return Err(LinearFitError::NonFiniteSolution);
        }

        let recurrence_ended = alpha == 0.0 || beta == 0.0;
        if iteration == 1 || iteration % 8 == 0 || recurrence_ended || iteration == max_iterations {
            relative_normal_residual = exact_relative_normal_residual(
                &matrix,
                rhs,
                &solution,
                normal_rhs_norm,
                &mut next_u,
                &mut next_v,
            );
            if !relative_normal_residual.is_finite() {
                return Err(LinearFitError::NonFiniteSolution);
            }
            if relative_normal_residual <= tolerance {
                accepted_iteration = iteration;
                break;
            }
        }
        if recurrence_ended {
            break;
        }
    }

    if accepted_iteration == 0 {
        return Err(LinearFitError::SolverDidNotConverge {
            iterations: final_iteration,
            relative_normal_residual,
        });
    }
    for (value, scale) in solution.iter_mut().zip(column_scales) {
        *value /= scale;
    }
    Ok(LeastSquaresSolution {
        values: solution,
        iterations: accepted_iteration,
        relative_normal_residual,
    })
}

fn exact_relative_normal_residual(
    matrix: &SparseLeastSquares,
    rhs: &[f64],
    solution: &[f64],
    normal_rhs_norm: f64,
    residual: &mut [f64],
    gradient: &mut [f64],
) -> f64 {
    matrix.multiply(solution, residual);
    for (value, target) in residual.iter_mut().zip(rhs) {
        *value -= target;
    }
    matrix.multiply_transpose(residual, gradient);
    vector_norm(gradient) / normal_rhs_norm.max(f64::MIN_POSITIVE)
}

fn vector_norm(values: &[f64]) -> f64 {
    values.iter().fold(0.0f64, |norm, value| norm.hypot(*value))
}

fn scale_vector(values: &mut [f64], factor: f64) {
    for value in values {
        *value *= factor;
    }
}

fn algebraic_residual_rms(
    coarse_pos: &[[f64; 3]],
    coarse_faces: &[[usize; 3]],
    samples: &[Sample],
    weights: &[Quat],
) -> f64 {
    let sum_squared = samples
        .iter()
        .map(|sample| {
            let target = Quat::from_point(sample.target[0], sample.target[1], sample.target[2]);
            let face = coarse_faces[sample.face_index];
            let residual = (0..3).fold(Quat::ZERO, |residual, local| {
                let vertex = face[local];
                let position = Quat::from_point(
                    coarse_pos[vertex][0],
                    coarse_pos[vertex][1],
                    coarse_pos[vertex][2],
                );
                residual + ((position - target) * weights[vertex]) * sample.bary[local]
            });
            residual.norm_sq()
        })
        .sum::<f64>();
    (sum_squared / samples.len() as f64).sqrt()
}

/// The 4×4 real matrix `L(q)` of quaternion **left**-multiplication: for any
/// quaternion `w`, `L(q) · vec(w) = vec(q · w)` where `vec(w) = [w, x, y, z]`.
///
/// This is the linchpin of the whole module — it is what makes the algebraic
/// residual linear in the weights.
pub fn left_mul_matrix(q: Quat) -> [[f64; 4]; 4] {
    // Derived from q·w = (qw+qxi+qyj+qzk)(ww+wxi+wyj+wzk):
    [
        [q.w, -q.x, -q.y, -q.z],
        [q.x, q.w, -q.z, q.y],
        [q.y, q.z, q.w, -q.x],
        [q.z, -q.y, q.x, q.w],
    ]
}

/// Reuse the crate's mesh-projection sampler to build [`Sample`]s: project each
/// original vertex onto the coarse faces it falls in and record the barycentric
/// coordinates and the original position as the target. Thin wrapper over
/// [`crate::global_fit`]'s `collect_face_samples` — the same correspondence the
/// production curved fitter uses.
pub fn collect_samples(
    coarse_pos: &[[f64; 3]],
    coarse_faces: &[[usize; 3]],
    orig_pos: &[[f64; 3]],
    margin: f64,
) -> Vec<Sample> {
    let dummy_normals = vec![[0.0f64; 3]; orig_pos.len()];
    let mut out = Vec::new();
    for (fi, face) in coarse_faces.iter().enumerate() {
        let tri = [
            coarse_pos[face[0]],
            coarse_pos[face[1]],
            coarse_pos[face[2]],
        ];
        let fs = crate::global_fit::collect_face_samples(&tri, orig_pos, &dummy_normals, margin);
        for (k, bary) in fs.bary.iter().enumerate() {
            out.push(Sample {
                face_index: fi,
                bary: *bary,
                target: fs.positions[k],
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use quilting_core::quaternion::Mobius;

    /// The linearity claim the whole module rests on: `L(q)·vec(w) = vec(q·w)`.
    #[test]
    fn left_mul_matrix_matches_quaternion_product() {
        let cases = [
            (
                Quat::new(0.3, -1.2, 0.7, 2.1),
                Quat::new(-0.5, 0.9, 1.3, -0.2),
            ),
            (Quat::from_point(1.0, -2.0, 0.5), Quat::ONE),
            (
                Quat::new(2.0, 0.0, -1.0, 0.4),
                Quat::new(0.0, 0.0, 0.0, 1.0),
            ),
            (Quat::I, Quat::J),
        ];
        for (q, w) in cases {
            let l = left_mul_matrix(q);
            let wv = [w.w, w.x, w.y, w.z];
            let got = [
                l[0][0] * wv[0] + l[0][1] * wv[1] + l[0][2] * wv[2] + l[0][3] * wv[3],
                l[1][0] * wv[0] + l[1][1] * wv[1] + l[1][2] * wv[2] + l[1][3] * wv[3],
                l[2][0] * wv[0] + l[2][1] * wv[1] + l[2][2] * wv[2] + l[2][3] * wv[3],
                l[3][0] * wv[0] + l[3][1] * wv[1] + l[3][2] * wv[2] + l[3][3] * wv[3],
            ];
            let expect = q * w;
            let expect_v = [expect.w, expect.x, expect.y, expect.z];
            for i in 0..4 {
                assert!(
                    (got[i] - expect_v[i]).abs() < 1e-12,
                    "L(q)·vec(w) mismatch: {:?} vs {:?}",
                    got,
                    expect_v
                );
            }
        }
    }

    #[test]
    fn sparse_lsqr_solves_an_overdetermined_system() {
        let matrix = SparseLeastSquares {
            rows: 4,
            columns: 2,
            row_offsets: vec![0, 1, 2, 4, 6],
            column_indices: vec![0, 1, 0, 1, 0, 1],
            values: vec![1.0, 1.0, 1.0, 1.0, 2.0, -1.0],
        };
        let result = solve_least_squares_lsqr(matrix, &[2.0, -3.0, -1.0, 7.0], 32, 1e-12)
            .expect("full-rank exact system should converge");
        assert!((result.values[0] - 2.0).abs() < 1e-11);
        assert!((result.values[1] + 3.0).abs() < 1e-11);
        assert!(result.relative_normal_residual <= 1e-12);
    }

    #[test]
    fn sparse_lsqr_matches_the_analytic_inconsistent_solution() {
        // min ||[x, y, x+y] - [1, 2, 4]|| has normal-equation solution
        // (x, y) = (4/3, 7/3). The nonzero residual exercises the actual
        // least-squares path rather than only exact interpolation.
        let matrix = SparseLeastSquares {
            rows: 3,
            columns: 2,
            row_offsets: vec![0, 1, 2, 4],
            column_indices: vec![0, 1, 0, 1],
            values: vec![1.0, 1.0, 1.0, 1.0],
        };
        let result = solve_least_squares_lsqr(matrix, &[1.0, 2.0, 4.0], 32, 1e-12)
            .expect("full-rank inconsistent system should converge");
        assert!((result.values[0] - 4.0 / 3.0).abs() < 1e-11);
        assert!((result.values[1] - 7.0 / 3.0).abs() < 1e-11);
        assert!(result.relative_normal_residual <= 1e-12);
    }

    /// Build coarse mesh + samples from a known curved QB ground truth (a sphere
    /// obtained by inverting a flat octahedron), fit, and check we recover it to
    /// machine precision — this is only possible because a Möbius image of a
    /// plane admits an *exact* globally-consistent shared-weight assignment.
    fn octahedron_sphere() -> (Vec<[f64; 3]>, Vec<[usize; 3]>, Vec<Sample>, Vec<QBTriPatch>) {
        let z = 2.0;
        let verts = [
            [1.0, 0.0, z],
            [-1.0, 0.0, z],
            [0.0, 1.0, z],
            [0.0, -1.0, z],
            [0.0, 0.0, z + 1.0],
            [0.0, 0.0, z - 1.0],
        ];
        let faces_arr = [
            [0, 2, 4],
            [2, 1, 4],
            [1, 3, 4],
            [3, 0, 4],
            [2, 0, 5],
            [1, 2, 5],
            [3, 1, 5],
            [0, 3, 5],
        ];
        let inv = Mobius::inversion();
        // Coarse mesh = images of the octahedron vertices (shared under one map).
        let coarse_pos: Vec<[f64; 3]> = verts
            .iter()
            .map(|v| inv.apply(Quat::from_point(v[0], v[1], v[2])).to_point())
            .collect();
        let coarse_faces: Vec<[usize; 3]> = faces_arr.iter().map(|f| [f[0], f[1], f[2]]).collect();

        // Ground-truth patches and samples from tessellating them (exact bary).
        let mut truth = Vec::new();
        let mut samples = Vec::new();
        for (fi, f) in faces_arr.iter().enumerate() {
            let flat = QBTriPatch::flat(verts[f[0]], verts[f[1]], verts[f[2]]);
            let patch = flat.transform(&inv);
            truth.push(patch);
            let tess = crate::roundtrip::tessellate_patch(&patch, 4);
            for (k, bary) in tess.bary.iter().enumerate() {
                samples.push(Sample {
                    face_index: fi,
                    bary: *bary,
                    target: tess.positions[k],
                });
            }
        }
        (coarse_pos, coarse_faces, samples, truth)
    }

    #[test]
    fn sphere_ground_truth_recovered_near_exactly() {
        let (coarse_pos, coarse_faces, samples, _truth) = octahedron_sphere();
        let cfg = LinearFitConfig {
            tikhonov: 1e-12,
            ..LinearFitConfig::default()
        };
        let res = linear_global_fit_full(&coarse_pos, &coarse_faces, &samples, &cfg).unwrap();

        // Residual: patch at each sample bary should hit the target.
        let mut max_err = 0.0f64;
        for s in &samples {
            let p = res.patches[s.face_index]
                .eval(s.bary[1], s.bary[2])
                .to_point();
            let d = ((p[0] - s.target[0]).powi(2)
                + (p[1] - s.target[1]).powi(2)
                + (p[2] - s.target[2]).powi(2))
            .sqrt();
            max_err = max_err.max(d);
        }
        assert!(
            max_err < 1e-6,
            "sphere ground truth not recovered: max_err={max_err:e}"
        );
        // And the fit must be genuinely curved, not a collapse to flat.
        assert!(
            res.max_weight_dev > 0.1,
            "weights barely moved ({}), fit collapsed to flat",
            res.max_weight_dev
        );
        assert!(res.solver_iterations > 0);
        assert!(res.relative_normal_residual <= cfg.solver_tolerance);
        assert!(res.algebraic_residual_rms < 1e-6);
    }

    #[test]
    fn shared_fit_is_stable_across_extreme_scene_scales() {
        let (coarse_pos, coarse_faces, samples, _truth) = octahedron_sphere();
        for scale in [1e-9, 1e9] {
            let scaled_positions = coarse_pos
                .iter()
                .map(|position| position.map(|component| component * scale))
                .collect::<Vec<_>>();
            let scaled_samples = samples
                .iter()
                .map(|sample| Sample {
                    target: sample.target.map(|component| component * scale),
                    ..*sample
                })
                .collect::<Vec<_>>();
            let result = linear_global_fit_full(
                &scaled_positions,
                &coarse_faces,
                &scaled_samples,
                &LinearFitConfig {
                    tikhonov: 1e-12,
                    ..LinearFitConfig::default()
                },
            )
            .unwrap();
            let relative_rms = sample_rms(&result.patches, &coarse_faces, &scaled_samples) / scale;
            assert!(
                relative_rms < 1e-6,
                "scale {scale:e} produced relative RMS {relative_rms:e}",
            );
            assert!(result.solver_iterations > 0);
            assert!(result.relative_normal_residual <= 1e-10);
            assert!(result.algebraic_residual_rms < 1e-6);
            assert!(result.min_relative_denominator > 0.1);
        }
    }

    #[test]
    fn disconnected_components_receive_independent_stable_gauge_pins() {
        let positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
            [10.0, 1.0, 0.0],
        ];
        let faces = [[0, 1, 2], [3, 4, 5]];
        let samples = faces
            .iter()
            .enumerate()
            .flat_map(|(face_index, face)| {
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]].map(move |bary| Sample {
                    face_index,
                    bary,
                    target: positions[face[bary.iter().position(|value| *value == 1.0).unwrap()]],
                })
            })
            .collect::<Vec<_>>();
        let result =
            linear_global_fit_full(&positions, &faces, &samples, &LinearFitConfig::default())
                .unwrap();
        assert_eq!(result.dim, 16);
        assert!(result.solver_iterations > 0);
        assert!(result.relative_normal_residual <= 1e-10);
        assert_eq!(result.vertex_weights[0], Quat::ONE);
        assert_eq!(result.vertex_weights[3], Quat::ONE);
        assert!(result.algebraic_residual_rms < 1e-12);
    }

    #[test]
    fn exact_denominator_guard_rejects_an_interior_zero() {
        let patch = QBTriPatch::new(
            [
                Quat::from_point(0.0, 0.0, 0.0),
                Quat::from_point(1.0, 0.0, 0.0),
                Quat::from_point(0.0, 1.0, 0.0),
            ],
            [Quat::ONE, Quat::ONE, Quat::new(-1.0, 0.0, 0.0, 0.0)],
        );
        assert!(matches!(
            validate_patch_denominators(&[patch], 1e-5),
            Err(LinearFitError::IllConditionedPatch {
                face: 0,
                min_relative_denominator,
            }) if min_relative_denominator <= 1e-12
        ));
    }

    #[test]
    fn malformed_shared_fit_inputs_fail_before_matrix_construction() {
        let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let faces = [[0, 1, 2]];
        let samples = [Sample {
            face_index: 0,
            bary: [0.2, 0.3, 0.5],
            target: [0.3, 0.5, 0.0],
        }];
        assert_eq!(
            linear_global_fit_full(
                &positions,
                &[[0, 1, 3]],
                &samples,
                &LinearFitConfig::default(),
            )
            .unwrap_err(),
            LinearFitError::InvalidFaceVertex { face: 0, vertex: 3 },
        );
        assert_eq!(
            linear_global_fit_full(
                &positions,
                &faces,
                &[Sample {
                    bary: [0.2, 0.3, 0.6],
                    ..samples[0]
                }],
                &LinearFitConfig::default(),
            )
            .unwrap_err(),
            LinearFitError::InvalidBarycentric { sample: 0 },
        );
        assert_eq!(
            linear_global_fit_full(
                &positions,
                &faces,
                &samples,
                &LinearFitConfig {
                    tikhonov: 0.0,
                    ..LinearFitConfig::default()
                },
            )
            .unwrap_err(),
            LinearFitError::InvalidRegularization(0.0),
        );
        assert_eq!(
            linear_global_fit_full(
                &positions,
                &faces,
                &samples,
                &LinearFitConfig {
                    relative_denominator_epsilon: 0.0,
                    ..LinearFitConfig::default()
                },
            )
            .unwrap_err(),
            LinearFitError::InvalidDenominatorThreshold(0.0),
        );
        assert_eq!(
            linear_global_fit_full(
                &positions,
                &faces,
                &samples,
                &LinearFitConfig {
                    solver_tolerance: 0.0,
                    ..LinearFitConfig::default()
                },
            )
            .unwrap_err(),
            LinearFitError::InvalidSolverTolerance(0.0),
        );
        assert_eq!(
            linear_global_fit_full(
                &positions,
                &faces,
                &samples,
                &LinearFitConfig {
                    max_iterations: 0,
                    ..LinearFitConfig::default()
                },
            )
            .unwrap_err(),
            LinearFitError::InvalidMaxIterations,
        );
    }

    #[test]
    fn shared_fit_reports_nonconvergence_instead_of_returning_a_stale_iterate() {
        let (coarse_pos, coarse_faces, samples, _truth) = octahedron_sphere();
        let error = linear_global_fit_full(
            &coarse_pos,
            &coarse_faces,
            &samples,
            &LinearFitConfig {
                max_iterations: 1,
                solver_tolerance: 1e-30,
                ..LinearFitConfig::default()
            },
        )
        .unwrap_err();
        assert!(matches!(
            error,
            LinearFitError::SolverDidNotConverge {
                iterations: 1,
                relative_normal_residual,
            } if relative_normal_residual > 1e-30
        ));
    }

    #[test]
    fn sphere_curved_beats_flat_by_large_margin() {
        let (coarse_pos, coarse_faces, samples, _truth) = octahedron_sphere();
        let res = linear_global_fit_full(
            &coarse_pos,
            &coarse_faces,
            &samples,
            &LinearFitConfig::default(),
        )
        .unwrap();
        let curved = sample_rms(&res.patches, &coarse_faces, &samples);
        let flat_patches = flat_patches(&coarse_pos, &coarse_faces);
        let flat = sample_rms(&flat_patches, &coarse_faces, &samples);
        assert!(
            flat > curved * 50.0,
            "curved RMS {curved:e} should be ≫ better than flat RMS {flat:e}"
        );
    }

    #[test]
    fn c0_edge_gap_is_machine_epsilon() {
        let (coarse_pos, coarse_faces, samples, _truth) = octahedron_sphere();
        let res = linear_global_fit_full(
            &coarse_pos,
            &coarse_faces,
            &samples,
            &LinearFitConfig::default(),
        )
        .unwrap();
        let gap = max_c0_edge_gap(&res.patches, &coarse_faces, 12);
        assert!(
            gap < 1e-9,
            "shared-weight patches should be C0 watertight, gap={gap:e}"
        );
    }

    // --- small test helpers (mirrored, simply, in the example) ---

    fn flat_patches(pos: &[[f64; 3]], faces: &[[usize; 3]]) -> Vec<QBTriPatch> {
        faces
            .iter()
            .map(|f| QBTriPatch::flat(pos[f[0]], pos[f[1]], pos[f[2]]))
            .collect()
    }

    fn sample_rms(patches: &[QBTriPatch], _faces: &[[usize; 3]], samples: &[Sample]) -> f64 {
        let mut sum = 0.0;
        for s in samples {
            let p = patches[s.face_index].eval(s.bary[1], s.bary[2]).to_point();
            sum += (p[0] - s.target[0]).powi(2)
                + (p[1] - s.target[1]).powi(2)
                + (p[2] - s.target[2]).powi(2);
        }
        (sum / samples.len().max(1) as f64).sqrt()
    }

    fn max_c0_edge_gap(patches: &[QBTriPatch], faces: &[[usize; 3]], t_steps: usize) -> f64 {
        use std::collections::HashMap;
        let mut edge_faces: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        for (fi, f) in faces.iter().enumerate() {
            for i in 0..3 {
                let a = f[i];
                let b = f[(i + 1) % 3];
                edge_faces.entry((a.min(b), a.max(b))).or_default().push(fi);
            }
        }
        let mut max_gap = 0.0f64;
        for (&(g0, g1), fs) in &edge_faces {
            if fs.len() < 2 {
                continue;
            }
            let f0 = fs[0];
            let f1 = fs[1];
            for step in 0..=t_steps {
                let t = step as f64 / t_steps as f64;
                let p0 = eval_edge(&patches[f0], &faces[f0], g0, g1, t);
                let p1 = eval_edge(&patches[f1], &faces[f1], g0, g1, t);
                let d =
                    ((p0[0] - p1[0]).powi(2) + (p0[1] - p1[1]).powi(2) + (p0[2] - p1[2]).powi(2))
                        .sqrt();
                max_gap = max_gap.max(d);
            }
        }
        max_gap
    }

    fn eval_edge(patch: &QBTriPatch, face: &[usize; 3], g0: usize, g1: usize, t: f64) -> [f64; 3] {
        let l0 = face.iter().position(|&v| v == g0).unwrap();
        let l1 = face.iter().position(|&v| v == g1).unwrap();
        let mut bary = [0.0f64; 3];
        bary[l0] = 1.0 - t;
        bary[l1] = t;
        patch.eval(bary[1], bary[2]).to_point()
    }
}
