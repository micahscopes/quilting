/// 4D mesh: triangulated surface x time = prism complex.
///
/// Each face of the original 3D mesh sweeps through time, creating
/// a triangular prism between consecutive keyframes. The vertex
/// positions at any moment in time are given by their trajectories.

use crate::trajectory::VertexTrajectory;

/// A 4D mesh built from a triangle mesh whose vertices move over time.
#[derive(Debug, Clone)]
pub struct HyperMesh {
    /// Triangle connectivity (same as the underlying 3D mesh).
    pub faces: Vec<[u32; 3]>,
    /// Per-vertex trajectories through spacetime.
    pub trajectories: Vec<VertexTrajectory>,
    /// Number of vertices.
    pub num_vertices: u32,
    /// Original animation period (before loop padding).
    pub period: f64,
}

impl HyperMesh {
    /// Build from triangle connectivity + per-vertex trajectories.
    pub fn new(faces: Vec<[u32; 3]>, trajectories: Vec<VertexTrajectory>) -> Self {
        let num_vertices = trajectories.len() as u32;
        // Compute period from original trajectories before any padding
        let mut t_min = f64::INFINITY;
        let mut t_max = f64::NEG_INFINITY;
        for traj in &trajectories {
            if let Some(first) = traj.segments.first() { t_min = t_min.min(first.t_start); }
            if let Some(last) = traj.segments.last() { t_max = t_max.max(last.t_end); }
        }
        let period = if t_min.is_finite() && t_max.is_finite() { t_max - t_min } else { 1.0 };
        Self {
            faces,
            trajectories,
            num_vertices,
            period,
        }
    }

    /// Total time range of the animation (min start, max end across all trajectories).
    pub fn time_range(&self) -> (f64, f64) {
        let mut t_min = f64::INFINITY;
        let mut t_max = f64::NEG_INFINITY;

        for traj in &self.trajectories {
            if let Some(first) = traj.segments.first() {
                t_min = t_min.min(first.t_start);
            }
            if let Some(last) = traj.segments.last() {
                t_max = t_max.max(last.t_end);
            }
        }

        if t_min.is_infinite() {
            (0.0, 0.0)
        } else {
            (t_min, t_max)
        }
    }

    /// Evaluate all vertex positions at a given time.
    pub fn positions_at(&self, t: f64) -> Vec<[f64; 3]> {
        self.trajectories.iter().map(|traj| traj.eval(t)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trajectory::HermiteSegment;

    fn static_vertex(pos: [f64; 3], t0: f64, t1: f64) -> VertexTrajectory {
        VertexTrajectory {
            segments: vec![HermiteSegment {
                t_start: t0,
                t_end: t1,
                pos_start: pos,
                pos_end: pos,
                vel_start: [0.0; 3],
                vel_end: [0.0; 3],
            }],
        }
    }

    #[test]
    fn time_range_single_segment() {
        let mesh = HyperMesh::new(
            vec![[0, 1, 2]],
            vec![
                static_vertex([0.0, 0.0, 0.0], 0.0, 2.0),
                static_vertex([1.0, 0.0, 0.0], 0.5, 3.0),
                static_vertex([0.0, 1.0, 0.0], 0.0, 2.5),
            ],
        );
        let (t_min, t_max) = mesh.time_range();
        assert!((t_min - 0.0).abs() < 1e-12);
        assert!((t_max - 3.0).abs() < 1e-12);
    }

    #[test]
    fn positions_at_returns_correct_count() {
        let mesh = HyperMesh::new(
            vec![[0, 1, 2]],
            vec![
                static_vertex([0.0, 0.0, 0.0], 0.0, 1.0),
                static_vertex([1.0, 0.0, 0.0], 0.0, 1.0),
                static_vertex([0.0, 1.0, 0.0], 0.0, 1.0),
            ],
        );
        let positions = mesh.positions_at(0.5);
        assert_eq!(positions.len(), 3);
    }

    #[test]
    fn empty_mesh_time_range() {
        let mesh = HyperMesh::new(vec![], vec![]);
        let (t_min, t_max) = mesh.time_range();
        assert!((t_min - 0.0).abs() < 1e-12);
        assert!((t_max - 0.0).abs() < 1e-12);
    }
}
