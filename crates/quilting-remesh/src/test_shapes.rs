/// Procedural test meshes for controlled remeshing experiments.

use crate::geometry;

/// Generate a unit icosphere with given subdivision level.
/// subdivision=0: 20 faces (icosahedron)
/// subdivision=1: 80 faces
/// subdivision=2: 320 faces
/// subdivision=3: 1280 faces
pub fn sphere(subdivisions: u32) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let (mut positions, mut faces) = quilting_core::shapes::icosahedron();
    for _ in 0..subdivisions {
        let (new_pos, new_faces) = subdivide_sphere(&positions, &faces);
        positions = new_pos;
        faces = new_faces;
    }
    for p in &mut positions {
        let len = geometry::vec3_len(*p);
        if len > 1e-10 { *p = geometry::vec3_scale(*p, 1.0 / len); }
    }
    (positions, faces)
}

/// Generate a closed cylinder along the Y axis.
pub fn cylinder(segments: usize, rings: usize, height: f64, radius: f64) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let mut positions = Vec::new();
    let mut faces = Vec::new();

    for ri in 0..=rings {
        let y = -height / 2.0 + height * (ri as f64 / rings as f64);
        for si in 0..segments {
            let angle = 2.0 * std::f64::consts::PI * (si as f64 / segments as f64);
            positions.push([radius * angle.cos(), y, radius * angle.sin()]);
        }
    }

    for ri in 0..rings {
        for si in 0..segments {
            let next_si = (si + 1) % segments;
            let a = ri * segments + si;
            let b = ri * segments + next_si;
            let c = (ri + 1) * segments + next_si;
            let d = (ri + 1) * segments + si;
            faces.push([a, b, c]);
            faces.push([a, c, d]);
        }
    }

    // Bottom cap
    let bottom_center = positions.len();
    positions.push([0.0, -height / 2.0, 0.0]);
    for si in 0..segments {
        let next_si = (si + 1) % segments;
        faces.push([bottom_center, next_si, si]);
    }

    // Top cap
    let top_center = positions.len();
    positions.push([0.0, height / 2.0, 0.0]);
    let top_ring_start = rings * segments;
    for si in 0..segments {
        let next_si = (si + 1) % segments;
        faces.push([top_center, top_ring_start + si, top_ring_start + next_si]);
    }

    (positions, faces)
}

fn subdivide_sphere(positions: &[[f64; 3]], faces: &[[usize; 3]]) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let mut new_positions = positions.to_vec();
    let mut new_faces = Vec::with_capacity(faces.len() * 4);
    let mut midpoint_cache: std::collections::HashMap<(usize, usize), usize> = std::collections::HashMap::new();

    let mut get_midpoint = |a: usize, b: usize, positions: &mut Vec<[f64; 3]>| -> usize {
        let key = (a.min(b), a.max(b));
        if let Some(&idx) = midpoint_cache.get(&key) {
            return idx;
        }
        let mid = geometry::vec3_scale(geometry::vec3_add(positions[a], positions[b]), 0.5);
        let idx = positions.len();
        positions.push(mid);
        midpoint_cache.insert(key, idx);
        idx
    };

    for face in faces {
        let a = face[0];
        let b = face[1];
        let c = face[2];
        let ab = get_midpoint(a, b, &mut new_positions);
        let bc = get_midpoint(b, c, &mut new_positions);
        let ca = get_midpoint(c, a, &mut new_positions);
        new_faces.push([a, ab, ca]);
        new_faces.push([b, bc, ab]);
        new_faces.push([c, ca, bc]);
        new_faces.push([ab, bc, ca]);
    }

    (new_positions, new_faces)
}
