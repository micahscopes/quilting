//! Deterministic 3D projections of small regular four-dimensional polytopes.
//!
//! The returned triangle meshes are deliberately exploded by 3-cell. A 4D
//! polytope's complete 2-skeleton is non-manifold as an ordinary triangle
//! surface because more than two faces meet along many edges. Duplicating and
//! separating each closed 3-cell gives the renderer honest manifold shells,
//! preserves the polytope's cell incidence for explanation, and lets the
//! ordinary crack-free LOD machinery operate without inventing topology.

use std::f64::consts::FRAC_1_SQRT_2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectedPolytope4 {
    /// Five tetrahedral cells.
    Simplex,
    /// Eight cubical cells.
    Tesseract,
    /// Sixteen tetrahedral cells.
    CrossPolytope,
}

/// A triangle mesh consisting of separated, closed projections of every
/// 3-cell. Vertex sharing is retained inside each cell and intentionally not
/// retained between cells.
#[derive(Debug, Clone, PartialEq)]
pub struct ExplodedCellProjection {
    pub positions: Vec<[f64; 3]>,
    pub faces: Vec<[usize; 3]>,
    pub cell_ranges: Vec<std::ops::Range<usize>>,
}

pub fn exploded_cell_projection(polytope: ProjectedPolytope4) -> ExplodedCellProjection {
    let cells = match polytope {
        ProjectedPolytope4::Simplex => simplex_cells(),
        ProjectedPolytope4::Tesseract => tesseract_cells(),
        ProjectedPolytope4::CrossPolytope => cross_polytope_cells(),
    };
    project_cells(cells)
}

#[derive(Debug)]
struct Cell4 {
    vertices: Vec<[f64; 4]>,
    faces: Vec<[usize; 3]>,
}

fn simplex_cells() -> Vec<Cell4> {
    let inv_sqrt_five = 1.0 / 5.0_f64.sqrt();
    let vertices = [
        [1.0, 1.0, 1.0, -inv_sqrt_five],
        [1.0, -1.0, -1.0, -inv_sqrt_five],
        [-1.0, 1.0, -1.0, -inv_sqrt_five],
        [-1.0, -1.0, 1.0, -inv_sqrt_five],
        [0.0, 0.0, 0.0, 4.0 * inv_sqrt_five],
    ];
    (0..vertices.len())
        .map(|omitted| {
            let cell_vertices = vertices
                .iter()
                .enumerate()
                .filter(|(index, _)| *index != omitted)
                .map(|(_, vertex)| *vertex)
                .collect();
            Cell4 {
                vertices: cell_vertices,
                faces: tetrahedron_faces(),
            }
        })
        .collect()
}

fn tesseract_cells() -> Vec<Cell4> {
    let (cube_vertices, cube_faces) = crate::shapes::cube();
    let mut cells = Vec::with_capacity(8);
    for fixed_axis in 0..4 {
        for fixed_sign in [-1.0, 1.0] {
            let varying = (0..4)
                .filter(|axis| *axis != fixed_axis)
                .collect::<Vec<_>>();
            let vertices = cube_vertices
                .iter()
                .map(|point| {
                    let mut vertex = [0.0; 4];
                    vertex[fixed_axis] = fixed_sign;
                    for (component, axis) in varying.iter().enumerate() {
                        vertex[*axis] = point[component];
                    }
                    vertex
                })
                .collect();
            cells.push(Cell4 {
                vertices,
                faces: cube_faces.clone(),
            });
        }
    }
    cells
}

fn cross_polytope_cells() -> Vec<Cell4> {
    let mut cells = Vec::with_capacity(16);
    for sign_bits in 0..16 {
        let vertices = (0..4)
            .map(|axis| {
                let mut vertex = [0.0; 4];
                vertex[axis] = if sign_bits & (1 << axis) == 0 {
                    -1.0
                } else {
                    1.0
                };
                vertex
            })
            .collect();
        cells.push(Cell4 {
            vertices,
            faces: tetrahedron_faces(),
        });
    }
    cells
}

fn tetrahedron_faces() -> Vec<[usize; 3]> {
    vec![[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]]
}

fn project_cells(cells: Vec<Cell4>) -> ExplodedCellProjection {
    let mut positions = Vec::new();
    let mut faces = Vec::new();
    let mut cell_ranges = Vec::with_capacity(cells.len());
    for cell in cells {
        let projected = cell
            .vertices
            .iter()
            .copied()
            .map(rotate4)
            .map(perspective4)
            .collect::<Vec<_>>();
        let center = centroid3(&projected);
        let direction = normalized3(center).unwrap_or([0.0, 0.0, 0.0]);
        let exploded = projected
            .into_iter()
            .map(|point| add3(point, scale3(direction, 0.48)))
            .collect::<Vec<_>>();
        let cell_center = centroid3(&exploded);
        let base = positions.len();
        let range = base..base + exploded.len();
        positions.extend(exploded.iter().copied());
        for mut face in cell.faces {
            orient_outward(&exploded, cell_center, &mut face);
            faces.push(face.map(|index| base + index));
        }
        cell_ranges.push(range);
    }

    normalize_radius(&mut positions, 1.45);
    ExplodedCellProjection {
        positions,
        faces,
        cell_ranges,
    }
}

fn rotate4(mut point: [f64; 4]) -> [f64; 4] {
    rotate_plane(&mut point, 0, 3, 0.43);
    rotate_plane(&mut point, 1, 2, -0.31);
    rotate_plane(&mut point, 2, 3, 0.67);
    rotate_plane(&mut point, 0, 1, 0.26);
    point
}

fn rotate_plane(point: &mut [f64; 4], a: usize, b: usize, angle: f64) {
    let (sin, cos) = angle.sin_cos();
    let pa = point[a];
    let pb = point[b];
    point[a] = cos * pa - sin * pb;
    point[b] = sin * pa + cos * pb;
}

fn perspective4(point: [f64; 4]) -> [f64; 3] {
    let distance = 4.5;
    let scale = distance / (distance - point[3]);
    [point[0] * scale, point[1] * scale, point[2] * scale]
}

fn orient_outward(positions: &[[f64; 3]], center: [f64; 3], face: &mut [usize; 3]) {
    let a = positions[face[0]];
    let b = positions[face[1]];
    let c = positions[face[2]];
    let normal = cross3(sub3(b, a), sub3(c, a));
    let face_center = scale3(add3(add3(a, b), c), 1.0 / 3.0);
    if dot3(normal, sub3(face_center, center)) < 0.0 {
        face.swap(1, 2);
    }
}

fn normalize_radius(positions: &mut [[f64; 3]], radius: f64) {
    let center = centroid3(positions);
    let max_radius = positions
        .iter()
        .map(|point| length3(sub3(*point, center)))
        .fold(0.0, f64::max);
    let scale = if max_radius > f64::EPSILON {
        radius / max_radius
    } else {
        FRAC_1_SQRT_2
    };
    for point in positions {
        *point = scale3(sub3(*point, center), scale);
    }
}

fn centroid3(points: &[[f64; 3]]) -> [f64; 3] {
    if points.is_empty() {
        return [0.0; 3];
    }
    let sum = points.iter().fold([0.0; 3], |sum, point| add3(sum, *point));
    scale3(sum, 1.0 / points.len() as f64)
}

fn normalized3(vector: [f64; 3]) -> Option<[f64; 3]> {
    let length = length3(vector);
    (length > f64::EPSILON).then(|| scale3(vector, 1.0 / length))
}

fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale3(vector: [f64; 3], scale: f64) -> [f64; 3] {
    [vector[0] * scale, vector[1] * scale, vector[2] * scale]
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn length3(vector: [f64; 3]) -> f64 {
    dot3(vector, vector).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projected_cell_counts_match_regular_four_polytope_incidence() {
        let simplex = exploded_cell_projection(ProjectedPolytope4::Simplex);
        assert_eq!(
            (
                simplex.cell_ranges.len(),
                simplex.positions.len(),
                simplex.faces.len()
            ),
            (5, 20, 20)
        );

        let tesseract = exploded_cell_projection(ProjectedPolytope4::Tesseract);
        assert_eq!(
            (
                tesseract.cell_ranges.len(),
                tesseract.positions.len(),
                tesseract.faces.len()
            ),
            (8, 64, 96)
        );

        let cross = exploded_cell_projection(ProjectedPolytope4::CrossPolytope);
        assert_eq!(
            (
                cross.cell_ranges.len(),
                cross.positions.len(),
                cross.faces.len()
            ),
            (16, 64, 64)
        );
    }

    #[test]
    fn projections_are_finite_centered_bounded_closed_cell_shells() {
        for polytope in [
            ProjectedPolytope4::Simplex,
            ProjectedPolytope4::Tesseract,
            ProjectedPolytope4::CrossPolytope,
        ] {
            let mesh = exploded_cell_projection(polytope);
            assert!(mesh
                .positions
                .iter()
                .flatten()
                .all(|value| value.is_finite()));
            assert!(mesh
                .faces
                .iter()
                .flatten()
                .all(|index| *index < mesh.positions.len()));
            assert!(length3(centroid3(&mesh.positions)) < 1e-12);
            let max_radius = mesh
                .positions
                .iter()
                .map(|point| length3(*point))
                .fold(0.0, f64::max);
            assert!((max_radius - 1.45).abs() < 1e-12);

            for range in &mesh.cell_ranges {
                let local_faces = mesh
                    .faces
                    .iter()
                    .filter(|face| face.iter().all(|index| range.contains(index)));
                let mut undirected = std::collections::BTreeMap::new();
                for face in local_faces {
                    for edge in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])] {
                        *undirected
                            .entry([edge.0.min(edge.1), edge.0.max(edge.1)])
                            .or_insert(0) += 1;
                    }
                }
                assert!(undirected.values().all(|incidence| *incidence == 2));
            }
        }
    }
}
