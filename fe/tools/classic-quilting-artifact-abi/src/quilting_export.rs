//! Deterministic adapter from current Rust Quilting topology to ABI v1.

use std::fmt;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::thread;

use quilting_core::atlas::{BuildMode, TessellationAtlas};
use quilting_core::delaunay::{triangulate_2d_constrained, Triangulation};
use quilting_core::sampling::PatchConfig;
use quilting_core::triangle;

use crate::{
    Artifact, AtlasKey, AtlasPatch, AtlasTriangle, AtlasVertex, Error as AbiError, SourceClass,
};

pub const DIRECT_ALGORITHM_VERSION: u32 = 1;
pub const FIXTURE_MASTER_SEED: u64 = 42;
pub const FIXTURE_K_CANDIDATES: usize = 30;
pub const FIXTURE_KEYS: [AtlasKey; 4] = [
    AtlasKey::new(1, 1, 1),
    AtlasKey::new(1, 1, 2),
    AtlasKey::new(1, 2, 2),
    AtlasKey::new(2, 4, 8),
];
pub const VERIFIED_POOL_WIDTHS: [usize; 3] = [1, 2, 4];

const BARYCENTRIC_SNAP_EPSILON: f64 = 1.0e-10;
const NEAR_DUPLICATE_DISTANCE_SQUARED: f64 = 1.0e-24;

#[derive(Debug)]
pub enum FixtureExportError {
    Abi(AbiError),
    EmptyKeySet,
    InvalidPoolWidth(usize),
    MissingPatch(AtlasKey),
    CountOverflow(&'static str),
    InvalidPatchIndex {
        key: AtlasKey,
        triangle: usize,
    },
    CoreGenerationPanicked,
    WorkerPanicked(usize),
    InputLengthMismatch {
        positions: usize,
        barycentrics: usize,
    },
    TooFewPoints(usize),
    NonfiniteInput {
        point: usize,
    },
    NearDuplicatePoints {
        first: usize,
        second: usize,
    },
    MissingBoundaryCorner(usize),
    CoreTriangulationPanicked,
    TriangulationIndexOutOfRange {
        triangle: usize,
    },
    DegenerateTriangle {
        triangle: usize,
    },
}

impl FixtureExportError {
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::Abi(_) => "artifact_validation_failed",
            Self::EmptyKeySet => "empty_key_set",
            Self::InvalidPoolWidth(_) => "invalid_pool_width",
            Self::MissingPatch(_) => "missing_patch",
            Self::CountOverflow(_) => "count_overflow",
            Self::InvalidPatchIndex { .. } => "invalid_patch_index",
            Self::CoreGenerationPanicked => "core_generation_panicked",
            Self::WorkerPanicked(_) => "worker_panicked",
            Self::InputLengthMismatch { .. } => "input_length_mismatch",
            Self::TooFewPoints(_) => "too_few_points",
            Self::NonfiniteInput { .. } => "nonfinite_input",
            Self::NearDuplicatePoints { .. } => "near_duplicate_points",
            Self::MissingBoundaryCorner(_) => "missing_boundary_corner",
            Self::CoreTriangulationPanicked => "core_triangulation_panicked",
            Self::TriangulationIndexOutOfRange { .. } => "triangulation_index_out_of_range",
            Self::DegenerateTriangle { .. } => "degenerate_triangle",
        }
    }
}

impl fmt::Display for FixtureExportError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {self:?}", self.code())
    }
}

impl std::error::Error for FixtureExportError {}

impl From<AbiError> for FixtureExportError {
    fn from(value: AbiError) -> Self {
        Self::Abi(value)
    }
}

#[derive(Clone, Debug)]
struct PatchFixture {
    key: AtlasKey,
    vertices: Vec<AtlasVertex>,
    triangles: Vec<[u32; 3]>,
}

/// Build the requested direct Quilting patches and merge them canonically.
///
/// `pool_width` changes only physical placement. Each patch owns the same seed
/// and the final merge is strictly key ordered, so widths 1, 2, and 4 must
/// encode to identical bytes.
///
/// # Errors
///
/// Returns a typed error for invalid admission, a Quilting panic, missing or
/// malformed core output, or a final ABI validation failure.
pub fn build_direct_fixture_artifact(
    keys: &[AtlasKey],
    pool_width: usize,
) -> Result<Artifact, FixtureExportError> {
    if keys.is_empty() {
        return Err(FixtureExportError::EmptyKeySet);
    }
    if pool_width == 0 {
        return Err(FixtureExportError::InvalidPoolWidth(pool_width));
    }

    let mut canonical_keys = keys.to_vec();
    for key in &mut canonical_keys {
        let mut values = [key.a, key.b, key.c];
        values.sort_unstable();
        *key = AtlasKey::new(values[0], values[1], values[2]);
    }
    canonical_keys.sort_unstable();
    canonical_keys.dedup();

    let admitted_width = pool_width.min(canonical_keys.len());
    let mut shards = vec![Vec::new(); admitted_width];
    for (index, key) in canonical_keys.into_iter().enumerate() {
        shards[index % admitted_width].push(key);
    }

    let mut patch_fixtures = thread::scope(|scope| {
        let handles: Vec<_> = shards
            .into_iter()
            .enumerate()
            .map(|(shard_index, shard)| {
                (shard_index, scope.spawn(move || build_direct_shard(&shard)))
            })
            .collect();
        let mut results = Vec::new();
        for (shard_index, handle) in handles {
            let mut shard = handle
                .join()
                .map_err(|_| FixtureExportError::WorkerPanicked(shard_index))??;
            results.append(&mut shard);
        }
        Ok::<_, FixtureExportError>(results)
    })?;

    patch_fixtures.sort_by_key(|patch| patch.key);
    assemble_artifact(&patch_fixtures)
}

/// Return one artifact per key while preserving the combined build's exact
/// patch-local topology and source order.
///
/// # Errors
///
/// Returns a typed generation or ABI validation error.
pub fn build_direct_fixture_matrix(
    pool_width: usize,
) -> Result<Vec<(AtlasKey, Artifact)>, FixtureExportError> {
    let combined = build_direct_fixture_artifact(&FIXTURE_KEYS, pool_width)?;
    let mut fixtures = Vec::with_capacity(combined.patches.len());
    for patch in &combined.patches {
        let vertex_start = patch.first_vertex as usize;
        let vertex_end = vertex_start + patch.vertex_count as usize;
        let triangle_start = patch.first_triangle as usize;
        let triangle_end = triangle_start + patch.triangle_count as usize;
        let triangles = combined.triangles[triangle_start..triangle_end]
            .iter()
            .map(|triangle| AtlasTriangle {
                indices: triangle.indices.map(|index| index - patch.first_vertex),
            })
            .collect();
        let artifact = Artifact {
            algorithm_version: combined.algorithm_version,
            master_seed: combined.master_seed,
            patches: vec![AtlasPatch {
                key: patch.key,
                first_vertex: 0,
                vertex_count: patch.vertex_count,
                first_triangle: 0,
                triangle_count: patch.triangle_count,
            }],
            vertices: combined.vertices[vertex_start..vertex_end].to_vec(),
            triangles,
        };
        crate::validate_artifact(&artifact)?;
        fixtures.push((patch.key, artifact));
    }
    Ok(fixtures)
}

fn build_direct_shard(keys: &[AtlasKey]) -> Result<Vec<PatchFixture>, FixtureExportError> {
    let core_keys: Vec<[u32; 3]> = keys.iter().map(|key| [key.a, key.b, key.c]).collect();
    let mut lod_levels: Vec<u32> = core_keys.iter().flatten().copied().collect();
    lod_levels.sort_unstable();
    lod_levels.dedup();
    let config = PatchConfig {
        k_candidates: FIXTURE_K_CANDIDATES,
        seed: FIXTURE_MASTER_SEED,
    };
    let atlas = catch_unwind(AssertUnwindSafe(|| {
        TessellationAtlas::build_for_keys(&lod_levels, &core_keys, &config, BuildMode::Direct)
    }))
    .map_err(|_| FixtureExportError::CoreGenerationPanicked)?;

    let mut fixtures = Vec::with_capacity(keys.len());
    for key in keys {
        fixtures.push(extract_patch(&atlas, *key)?);
    }
    Ok(fixtures)
}

fn extract_patch(
    atlas: &TessellationAtlas,
    key: AtlasKey,
) -> Result<PatchFixture, FixtureExportError> {
    let core_key = [key.a, key.b, key.c];
    let entry = atlas
        .patches
        .get(&core_key)
        .ok_or(FixtureExportError::MissingPatch(key))?;
    let positions = &atlas.positions[entry.base_vertex..entry.base_vertex + entry.vertex_count];
    let vertices: Vec<AtlasVertex> = positions
        .iter()
        .map(|position| atlas_vertex_from_cartesian(*position))
        .collect();
    let mut triangles = Vec::with_capacity(entry.triangle_count);
    for (triangle_index, triangle) in atlas.triangles
        [entry.base_triangle..entry.base_triangle + entry.triangle_count]
        .iter()
        .enumerate()
    {
        let local = triangle.map(|index| index.checked_sub(entry.base_vertex));
        let [Some(i0), Some(i1), Some(i2)] = local else {
            return Err(FixtureExportError::InvalidPatchIndex {
                key,
                triangle: triangle_index,
            });
        };
        if [i0, i1, i2]
            .into_iter()
            .any(|index| index >= vertices.len())
        {
            return Err(FixtureExportError::InvalidPatchIndex {
                key,
                triangle: triangle_index,
            });
        }
        let indices = [
            u32::try_from(i0).map_err(|_| FixtureExportError::CountOverflow("vertex index"))?,
            u32::try_from(i1).map_err(|_| FixtureExportError::CountOverflow("vertex index"))?,
            u32::try_from(i2).map_err(|_| FixtureExportError::CountOverflow("vertex index"))?,
        ];
        triangles.push(canonicalize_triangle(indices, &vertices));
    }
    Ok(PatchFixture {
        key,
        vertices,
        triangles,
    })
}

fn atlas_vertex_from_cartesian(position: [f64; 2]) -> AtlasVertex {
    let mut bary = triangle::cartesian_to_bary(position[0], position[1]);
    for component in &mut bary {
        if component.abs() < BARYCENTRIC_SNAP_EPSILON {
            *component = 0.0;
        }
    }
    let sum = bary.iter().sum::<f64>();
    for component in &mut bary {
        *component /= sum;
    }
    #[allow(clippy::cast_possible_truncation)]
    let barycentric = [bary[0] as f32, bary[1] as f32, bary[2] as f32];
    let zero_count = barycentric
        .iter()
        .filter(|component| **component == 0.0)
        .count();
    let source_class = match zero_count {
        0 => SourceClass::Interior,
        1 => SourceClass::Edge,
        2 => SourceClass::Corner,
        _ => unreachable!("a normalized barycentric point has at most two zero lanes"),
    };
    AtlasVertex {
        barycentric,
        source_class,
    }
}

fn canonicalize_triangle(mut indices: [u32; 3], vertices: &[AtlasVertex]) -> [u32; 3] {
    let bary = indices.map(|index| vertices[index as usize].barycentric);
    let orientation = (f64::from(bary[1][1]) - f64::from(bary[0][1]))
        * (f64::from(bary[2][2]) - f64::from(bary[0][2]))
        - (f64::from(bary[1][2]) - f64::from(bary[0][2]))
            * (f64::from(bary[2][1]) - f64::from(bary[0][1]));
    if orientation < 0.0 {
        indices.swap(1, 2);
    }
    let minimum_lane = indices
        .iter()
        .enumerate()
        .min_by_key(|(_, index)| *index)
        .map_or(0, |(lane, _)| lane);
    indices.rotate_left(minimum_lane);
    indices
}

fn assemble_artifact(patches: &[PatchFixture]) -> Result<Artifact, FixtureExportError> {
    let mut artifact = Artifact {
        algorithm_version: DIRECT_ALGORITHM_VERSION,
        master_seed: FIXTURE_MASTER_SEED,
        patches: Vec::with_capacity(patches.len()),
        vertices: Vec::new(),
        triangles: Vec::new(),
    };
    for patch in patches {
        let first_vertex = u32::try_from(artifact.vertices.len())
            .map_err(|_| FixtureExportError::CountOverflow("first_vertex"))?;
        let first_triangle = u32::try_from(artifact.triangles.len())
            .map_err(|_| FixtureExportError::CountOverflow("first_triangle"))?;
        let vertex_count = u32::try_from(patch.vertices.len())
            .map_err(|_| FixtureExportError::CountOverflow("vertex_count"))?;
        let triangle_count = u32::try_from(patch.triangles.len())
            .map_err(|_| FixtureExportError::CountOverflow("triangle_count"))?;
        artifact.patches.push(AtlasPatch {
            key: patch.key,
            first_vertex,
            vertex_count,
            first_triangle,
            triangle_count,
        });
        artifact.vertices.extend_from_slice(&patch.vertices);
        artifact
            .triangles
            .extend(patch.triangles.iter().map(|indices| AtlasTriangle {
                indices: indices.map(|index| index + first_vertex),
            }));
    }
    crate::validate_artifact(&artifact)?;
    Ok(artifact)
}

/// Fence the incumbent panicking CDT API behind deterministic typed admission.
///
/// # Errors
///
/// Returns a typed rejection for malformed/nonfinite/near-duplicate inputs,
/// missing reference corners, a caught core panic, or invalid output topology.
pub fn checked_triangulate_fixture(
    positions: &[[f64; 2]],
    barycentrics: &[[f64; 3]],
) -> Result<Triangulation, FixtureExportError> {
    preflight_triangulation(positions, barycentrics)?;
    let triangulation = catch_unwind(AssertUnwindSafe(|| {
        triangulate_2d_constrained(positions, barycentrics)
    }))
    .map_err(|_| FixtureExportError::CoreTriangulationPanicked)?;
    for (triangle_index, triangle) in triangulation.triangles.iter().enumerate() {
        if triangle.iter().any(|index| *index >= positions.len()) {
            return Err(FixtureExportError::TriangulationIndexOutOfRange {
                triangle: triangle_index,
            });
        }
        let [a, b, c] = triangle.map(|index| positions[index]);
        let area_twice = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
        if area_twice.abs() <= f64::EPSILON {
            return Err(FixtureExportError::DegenerateTriangle {
                triangle: triangle_index,
            });
        }
    }
    Ok(triangulation)
}

fn preflight_triangulation(
    positions: &[[f64; 2]],
    barycentrics: &[[f64; 3]],
) -> Result<(), FixtureExportError> {
    if positions.len() != barycentrics.len() {
        return Err(FixtureExportError::InputLengthMismatch {
            positions: positions.len(),
            barycentrics: barycentrics.len(),
        });
    }
    if positions.len() < 3 {
        return Err(FixtureExportError::TooFewPoints(positions.len()));
    }
    for (point, (position, barycentric)) in positions.iter().zip(barycentrics).enumerate() {
        if position
            .iter()
            .chain(barycentric)
            .any(|value| !value.is_finite())
        {
            return Err(FixtureExportError::NonfiniteInput { point });
        }
    }
    for first in 0..positions.len() {
        for second in first + 1..positions.len() {
            let dx = positions[first][0] - positions[second][0];
            let dy = positions[first][1] - positions[second][1];
            if dx.mul_add(dx, dy * dy) <= NEAR_DUPLICATE_DISTANCE_SQUARED {
                return Err(FixtureExportError::NearDuplicatePoints { first, second });
            }
        }
    }
    let corners: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for (corner_index, corner) in corners.into_iter().enumerate() {
        let has_exact_corner = barycentrics.iter().any(|value| {
            value
                .iter()
                .zip(corner)
                .all(|(actual, expected)| actual.to_bits() == expected.to_bits())
        });
        if !has_exact_corner {
            return Err(FixtureExportError::MissingBoundaryCorner(corner_index));
        }
    }
    Ok(())
}

/// Adversarial boundary fixture containing two distinct but numerically
/// inseparable points on AB.
#[must_use]
pub fn near_degenerate_fixture() -> (Vec<[f64; 2]>, Vec<[f64; 3]>) {
    let mut barycentrics = vec![
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0],
        [0.5 - f64::EPSILON, 0.5 + f64::EPSILON, 0.0],
    ];
    let positions = barycentrics
        .iter()
        .copied()
        .map(triangle::bary_to_cartesian)
        .collect();
    // Normalize the adversarial pair through the same explicit sum policy.
    for barycentric in &mut barycentrics {
        let sum = barycentric.iter().sum::<f64>();
        for component in barycentric {
            *component /= sum;
        }
    }
    (positions, barycentrics)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_fixture_matrix_is_admitted() {
        let fixtures = build_direct_fixture_matrix(1).expect("seed-42 fixture matrix");
        assert_eq!(fixtures.len(), FIXTURE_KEYS.len());
        for ((actual_key, artifact), expected_key) in fixtures.iter().zip(FIXTURE_KEYS) {
            assert_eq!(*actual_key, expected_key);
            crate::validate_artifact(artifact).expect("validated individual fixture");
            assert_eq!(
                crate::decode(&crate::encode(artifact).unwrap()).unwrap(),
                *artifact
            );
        }
    }

    #[test]
    fn pool_widths_are_byte_identical() {
        let expected = crate::encode(
            &build_direct_fixture_artifact(&FIXTURE_KEYS, 1).expect("width 1 fixture"),
        )
        .unwrap();
        for width in [2, 4] {
            let actual = crate::encode(
                &build_direct_fixture_artifact(&FIXTURE_KEYS, width)
                    .expect("parallel-width fixture"),
            )
            .unwrap();
            assert_eq!(
                actual, expected,
                "pool width {width} changed artifact bytes"
            );
        }
    }

    #[test]
    fn committed_fixture_bytes_match_the_read_only_quilting_adapter() {
        const COMMITTED_MATRIX: &[u8] =
            include_bytes!("../../../fixtures/classic-quilting/v1/direct-seed42-matrix.cqa");
        let artifact =
            build_direct_fixture_artifact(&FIXTURE_KEYS, 1).expect("width 1 fixture matrix");
        assert_eq!(crate::encode(&artifact).unwrap(), COMMITTED_MATRIX);

        let committed_individuals: [(&[u8], AtlasKey); 4] = [
            (
                include_bytes!("../../../fixtures/classic-quilting/v1/direct-seed42-k1-1-1.cqa"),
                AtlasKey::new(1, 1, 1),
            ),
            (
                include_bytes!("../../../fixtures/classic-quilting/v1/direct-seed42-k1-1-2.cqa"),
                AtlasKey::new(1, 1, 2),
            ),
            (
                include_bytes!("../../../fixtures/classic-quilting/v1/direct-seed42-k1-2-2.cqa"),
                AtlasKey::new(1, 2, 2),
            ),
            (
                include_bytes!("../../../fixtures/classic-quilting/v1/direct-seed42-k2-4-8.cqa"),
                AtlasKey::new(2, 4, 8),
            ),
        ];
        let generated = build_direct_fixture_matrix(1).expect("individual fixture matrix");
        for ((committed, expected_key), (actual_key, artifact)) in
            committed_individuals.into_iter().zip(generated)
        {
            assert_eq!(actual_key, expected_key);
            assert_eq!(crate::encode(&artifact).unwrap(), committed);
        }
    }

    #[test]
    fn near_degenerate_fixture_returns_stable_typed_rejection() {
        let (positions, barycentrics) = near_degenerate_fixture();
        let Err(error) = checked_triangulate_fixture(&positions, &barycentrics) else {
            panic!("near-duplicate boundary points must fail closed");
        };
        assert_eq!(error.code(), "near_duplicate_points");
        assert!(matches!(
            error,
            FixtureExportError::NearDuplicatePoints {
                first: 3,
                second: 4
            }
        ));
    }
}
