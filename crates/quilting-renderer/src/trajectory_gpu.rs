//! GPU trajectory evaluation via texture lookup.
//!
//! Packs Hermite segment data into a float texture. The compute shader
//! searches for the right segment at time t and evaluates the cubic
//! Hermite interpolation.

/// Packed trajectory data for GPU upload.
/// All segments for all vertices packed sequentially.
/// Per-vertex: segment_offset (into the segment array) + num_segments.
pub struct TrajectoryGpuData {
    /// Per-vertex: [segment_offset, num_segments] = 2 ints per vertex
    pub vertex_info: Vec<i32>,
    /// All segments packed: 14 floats per segment
    /// [t_start, t_end, px0,py0,pz0, px1,py1,pz1, vx0,vy0,vz0, vx1,vy1,vz1]
    pub segments: Vec<f32>,
    pub num_vertices: usize,
    pub total_segments: usize,
}

impl TrajectoryGpuData {
    /// Pack trajectory data for GPU upload.
    pub fn from_trajectories(trajectories: &[crate::trajectory_types::TrajectorySegment]) -> Self {
        // This is called from the WASM side with pre-flattened data
        unimplemented!("Use from_raw instead")
    }

    /// Pack from raw Hermite segments.
    pub fn from_raw(
        vertex_segments: &[Vec<(f64, f64, [f64;3], [f64;3], [f64;3], [f64;3])>],
    ) -> Self {
        let num_vertices = vertex_segments.len();
        let mut vertex_info = Vec::with_capacity(num_vertices * 2);
        let mut segments = Vec::new();
        let mut offset = 0i32;

        for segs in vertex_segments {
            vertex_info.push(offset);
            vertex_info.push(segs.len() as i32);
            for &(t0, t1, p0, p1, v0, v1) in segs {
                segments.push(t0 as f32);
                segments.push(t1 as f32);
                segments.push(p0[0] as f32); segments.push(p0[1] as f32); segments.push(p0[2] as f32);
                segments.push(p1[0] as f32); segments.push(p1[1] as f32); segments.push(p1[2] as f32);
                segments.push(v0[0] as f32); segments.push(v0[1] as f32); segments.push(v0[2] as f32);
                segments.push(v1[0] as f32); segments.push(v1[1] as f32); segments.push(v1[2] as f32);
                offset += 1;
            }
        }

        Self {
            vertex_info,
            segments,
            num_vertices,
            total_segments: offset as usize,
        }
    }
}
