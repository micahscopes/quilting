//! Small deterministic scenes for explaining quaternionic-Bézier patches and
//! crack-free adaptive tessellation.
//!
//! This module deliberately contains no browser or renderer state.  The same
//! mesh and LOD-field logic can drive the WebGL Hyperscope, a future WebGPU
//! frontend, or a native example without duplicating the educational model.

use crate::batch::{balance_resident_lods, ResidentLod};
use crate::quaternion::Quat;
use quilting_mesh::HalfEdgeMesh;
use std::collections::BTreeMap;

/// Geometry shown by the interactive patch laboratory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatchLabShape {
    /// One genuine rational QB triangle with an adjustable corner weight.
    Triangle,
    /// A shared-vertex triangulated plane.
    Plane,
    /// The minimal twelve-triangle cube from [`crate::shapes`].
    Cube,
}

/// Scalar field used to request edge tessellation levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatchLabField {
    Uniform,
    Wave,
    Radial,
    Sweep,
    /// Direct edge control for the single-triangle scene.
    ManualEdges,
}

/// Indexed source geometry plus per-face QB weights.
#[derive(Debug, Clone)]
pub struct PatchLabMesh {
    pub shape: PatchLabShape,
    pub positions: Vec<[f64; 3]>,
    pub faces: Vec<[u32; 3]>,
    /// Quaternion weights in `(w, x, y, z)` order, in face-corner order.
    pub face_weights: Vec<[[f64; 4]; 3]>,
}

/// Inputs to the deterministic LOD field.
#[derive(Debug, Clone, Copy)]
pub struct PatchLabLodConfig {
    pub field: PatchLabField,
    /// Field phase in radians.
    pub phase: f64,
    /// Inclusive power-of-two exponent range.
    pub min_exp: u32,
    pub max_exp: u32,
    /// Face-local edge exponents `[BC, CA, AB]` for [`PatchLabField::ManualEdges`].
    pub manual_edge_exp: [u32; 3],
}

impl Default for PatchLabLodConfig {
    fn default() -> Self {
        Self {
            field: PatchLabField::Wave,
            phase: 0.0,
            min_exp: 1,
            max_exp: 6,
            manual_edge_exp: [3, 4, 4],
        }
    }
}

/// Reconciled topology and enough accounting to explain promotions in the UI.
#[derive(Debug, Clone)]
pub struct PatchLabLodResult {
    pub requested: Vec<[u32; 3]>,
    pub residents: Vec<ResidentLod>,
    pub promoted_faces: usize,
    pub promoted_edges: usize,
    pub shared_edges: usize,
    pub shared_edge_mismatches: usize,
    pub max_face_edge_ratio: u32,
    pub histogram: BTreeMap<[u32; 3], usize>,
}

impl PatchLabMesh {
    /// Construct a laboratory mesh. `grid` is used only by the plane and is
    /// clamped so an accidental UI value cannot create an enormous demo.
    /// `bend` is clamped to `[0, 1]` and used only by the single QB triangle.
    pub fn new(shape: PatchLabShape, grid: u32, bend: f64) -> Self {
        match shape {
            PatchLabShape::Triangle => triangle(bend),
            PatchLabShape::Plane => plane(grid.clamp(1, 32)),
            PatchLabShape::Cube => {
                let (positions, faces) = crate::shapes::cube();
                let faces = faces
                    .into_iter()
                    .map(|face| face.map(|index| index as u32))
                    .collect::<Vec<_>>();
                Self::flat(PatchLabShape::Cube, positions, faces)
            }
        }
    }

    fn flat(shape: PatchLabShape, positions: Vec<[f64; 3]>, faces: Vec<[u32; 3]>) -> Self {
        let face_weights = vec![[quat_array(Quat::ONE); 3]; faces.len()];
        Self {
            shape,
            positions,
            faces,
            face_weights,
        }
    }

    pub fn topology(&self) -> HalfEdgeMesh {
        HalfEdgeMesh::from_triangles(self.positions.len() as u32, &self.faces)
    }

    /// Sample a field at every face-local edge midpoint, quantize to powers of
    /// two, then run the renderer's exact shared-edge and 2:1 face reconciliation.
    pub fn lods(&self, config: PatchLabLodConfig) -> PatchLabLodResult {
        let min_exp = config.min_exp.min(9);
        let max_exp = config.max_exp.clamp(min_exp, 9);
        let requested = self
            .faces
            .iter()
            .map(|face| {
                if self.shape == PatchLabShape::Triangle
                    && config.field == PatchLabField::ManualEdges
                {
                    return config
                        .manual_edge_exp
                        .map(|exp| 1u32 << exp.clamp(min_exp, max_exp));
                }

                // Face-local edge order is the edge opposite each corner:
                // [BC, CA, AB]. This is the order consumed by ResidentLod.
                let pairs = [(face[1], face[2]), (face[2], face[0]), (face[0], face[1])];
                pairs.map(|(a, b)| {
                    let pa = self.positions[a as usize];
                    let pb = self.positions[b as usize];
                    let midpoint = [
                        (pa[0] + pb[0]) * 0.5,
                        (pa[1] + pb[1]) * 0.5,
                        (pa[2] + pb[2]) * 0.5,
                    ];
                    let amount = field_amount(config.field, midpoint, config.phase);
                    let span = (max_exp - min_exp) as f64;
                    let exp = (min_exp as f64 + amount * span).round() as u32;
                    1u32 << exp.clamp(min_exp, max_exp)
                })
            })
            .collect::<Vec<_>>();

        let topology = self.topology();
        let mut optional = requested
            .iter()
            .copied()
            .map(ResidentLod::from_edge_lods)
            .map(Some)
            .collect::<Vec<_>>();
        balance_resident_lods(&mut optional, &topology);
        let residents = optional
            .into_iter()
            .map(|lod| lod.expect("all lab faces are resident"))
            .collect::<Vec<_>>();

        let mut promoted_faces = 0;
        let mut promoted_edges = 0;
        let mut max_face_edge_ratio = 1;
        let mut histogram = BTreeMap::new();
        for (wanted, actual) in requested.iter().zip(&residents) {
            let actual_edges = actual.edge_lods();
            let edge_promotions = wanted
                .iter()
                .zip(actual_edges)
                .filter(|(before, after)| **before != *after)
                .count();
            promoted_edges += edge_promotions;
            promoted_faces += usize::from(edge_promotions > 0);
            max_face_edge_ratio =
                max_face_edge_ratio.max(actual.canonical[2] / actual.canonical[0].max(1));
            *histogram.entry(actual.canonical).or_default() += 1;
        }

        let (shared_edges, shared_edge_mismatches) = shared_edge_stats(&topology, &residents);
        PatchLabLodResult {
            requested,
            residents,
            promoted_faces,
            promoted_edges,
            shared_edges,
            shared_edge_mismatches,
            max_face_edge_ratio,
            histogram,
        }
    }
}

fn triangle(bend: f64) -> PatchLabMesh {
    let positions = vec![[-1.15, -0.82, 0.0], [1.15, -0.82, 0.0], [0.0, 1.15, 0.0]];
    let faces = vec![[0, 1, 2]];
    let bend = bend.clamp(0.0, 1.0) * 0.82;
    // Interpolate one corner from a scalar identity weight toward k. The
    // denominator retains a positive scalar component throughout the patch,
    // so the pedagogical slider cannot introduce a pole.
    let apex = Quat::new(1.0 - bend, 0.0, 0.0, bend);
    PatchLabMesh {
        shape: PatchLabShape::Triangle,
        positions,
        faces,
        face_weights: vec![[
            quat_array(Quat::ONE),
            quat_array(Quat::ONE),
            quat_array(apex),
        ]],
    }
}

fn plane(grid: u32) -> PatchLabMesh {
    let side = grid + 1;
    let mut positions = Vec::with_capacity((side * side) as usize);
    for y in 0..=grid {
        for x in 0..=grid {
            let fx = x as f64 / grid as f64;
            let fy = y as f64 / grid as f64;
            positions.push([(fx - 0.5) * 2.8, (fy - 0.5) * 2.8, 0.0]);
        }
    }
    let mut faces = Vec::with_capacity((grid * grid * 2) as usize);
    for y in 0..grid {
        for x in 0..grid {
            let a = y * side + x;
            let b = a + 1;
            let d = (y + 1) * side + x;
            let c = d + 1;
            // Alternate diagonals so the field is not visually biased toward
            // one repeated slash direction.
            if (x + y) & 1 == 0 {
                faces.push([a, b, c]);
                faces.push([a, c, d]);
            } else {
                faces.push([a, b, d]);
                faces.push([b, c, d]);
            }
        }
    }
    PatchLabMesh::flat(PatchLabShape::Plane, positions, faces)
}

fn quat_array(q: Quat) -> [f64; 4] {
    [q.w, q.x, q.y, q.z]
}

fn field_amount(field: PatchLabField, point: [f64; 3], phase: f64) -> f64 {
    let raw = match field {
        PatchLabField::Uniform | PatchLabField::ManualEdges => 0.5,
        PatchLabField::Wave => {
            0.5 + 0.5 * (2.8 * point[0] + 1.35 * point[1] + 0.65 * point[2] - phase).sin()
        }
        PatchLabField::Radial => {
            let radius = (point[0] * point[0] + point[1] * point[1] + point[2] * point[2]).sqrt();
            0.5 + 0.5 * (4.2 * radius - phase).cos()
        }
        PatchLabField::Sweep => {
            let center = 1.55 * phase.sin();
            let distance = (point[0] - center).abs();
            (-distance * distance / (2.0 * 0.38 * 0.38)).exp()
        }
    };
    raw.clamp(0.0, 1.0)
}

fn shared_edge_stats(topology: &HalfEdgeMesh, residents: &[ResidentLod]) -> (usize, usize) {
    let mut shared = 0;
    let mut mismatched = 0;
    for half_edge in 0..topology.half_edges.len() as u32 {
        let Some(twin) = topology.twin(half_edge) else {
            continue;
        };
        if half_edge > twin {
            continue;
        }
        shared += 1;
        let face = topology.half_edges[half_edge as usize].face as usize;
        let twin_face = topology.half_edges[twin as usize].face as usize;
        let edge = (half_edge as usize % 3 + 2) % 3;
        let twin_edge = (twin as usize % 3 + 2) % 3;
        if residents[face].edge_lods()[edge] != residents[twin_face].edge_lods()[twin_edge] {
            mismatched += 1;
        }
    }
    (shared, mismatched)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn laboratory_mesh_counts_are_small_and_predictable() {
        let tri = PatchLabMesh::new(PatchLabShape::Triangle, 8, 0.5);
        assert_eq!((tri.positions.len(), tri.faces.len()), (3, 1));
        let plane = PatchLabMesh::new(PatchLabShape::Plane, 8, 0.0);
        assert_eq!((plane.positions.len(), plane.faces.len()), (81, 128));
        let cube = PatchLabMesh::new(PatchLabShape::Cube, 8, 0.0);
        assert_eq!((cube.positions.len(), cube.faces.len()), (8, 12));
    }

    #[test]
    fn animated_fields_are_deterministic_and_phase_sensitive() {
        let mesh = PatchLabMesh::new(PatchLabShape::Plane, 8, 0.0);
        let config = PatchLabLodConfig::default();
        let a = mesh.lods(config);
        let b = mesh.lods(config);
        assert_eq!(a.requested, b.requested);
        let moved = mesh.lods(PatchLabLodConfig {
            phase: 1.2,
            ..config
        });
        assert_ne!(a.requested, moved.requested);
    }

    #[test]
    fn all_fields_reconcile_to_crack_free_two_to_one_topology() {
        for shape in [PatchLabShape::Plane, PatchLabShape::Cube] {
            let mesh = PatchLabMesh::new(shape, 8, 0.0);
            for field in [
                PatchLabField::Uniform,
                PatchLabField::Wave,
                PatchLabField::Radial,
                PatchLabField::Sweep,
            ] {
                let result = mesh.lods(PatchLabLodConfig {
                    field,
                    phase: 0.73,
                    min_exp: 0,
                    max_exp: 7,
                    manual_edge_exp: [0; 3],
                });
                assert_eq!(result.shared_edge_mismatches, 0, "{shape:?} / {field:?}");
                assert!(result.max_face_edge_ratio <= 2, "{shape:?} / {field:?}");
                for resident in result.residents {
                    assert!(resident.edge_lods().into_iter().all(u32::is_power_of_two));
                    assert!(resident.canonical[0] >= 1 && resident.canonical[2] <= 128);
                }
            }
        }
    }

    #[test]
    fn direct_triangle_edges_make_reconciliation_visible() {
        let mesh = PatchLabMesh::new(PatchLabShape::Triangle, 1, 0.0);
        let result = mesh.lods(PatchLabLodConfig {
            field: PatchLabField::ManualEdges,
            min_exp: 0,
            max_exp: 7,
            manual_edge_exp: [0, 3, 7],
            ..PatchLabLodConfig::default()
        });
        assert_eq!(result.requested[0], [1, 8, 128]);
        assert_eq!(result.residents[0].canonical, [64, 64, 128]);
        assert_eq!(result.promoted_edges, 2);
    }

    #[test]
    fn bend_weight_stays_finite_across_the_triangle() {
        let mesh = PatchLabMesh::new(PatchLabShape::Triangle, 1, 1.0);
        let weights = mesh.face_weights[0];
        let patch = crate::QBTriPatch::new(
            [
                Quat::from_point(
                    mesh.positions[0][0],
                    mesh.positions[0][1],
                    mesh.positions[0][2],
                ),
                Quat::from_point(
                    mesh.positions[1][0],
                    mesh.positions[1][1],
                    mesh.positions[1][2],
                ),
                Quat::from_point(
                    mesh.positions[2][0],
                    mesh.positions[2][1],
                    mesh.positions[2][2],
                ),
            ],
            weights.map(|q| Quat::new(q[0], q[1], q[2], q[3])),
        );
        for row in 0..=20 {
            for column in 0..=20 - row {
                let u = row as f64 / 20.0;
                let v = column as f64 / 20.0;
                let point = patch.eval(u, v);
                assert!(
                    point.w.is_finite()
                        && point.x.is_finite()
                        && point.y.is_finite()
                        && point.z.is_finite()
                );
            }
        }
    }
}
