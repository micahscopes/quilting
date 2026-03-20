use std::collections::HashMap;
use crate::quaternion::{Quat, Mobius};
use crate::patch::QBTriPatch;

/// Per-face instance data for instanced rendering.
#[derive(Debug, Clone)]
pub struct FaceInstance {
    pub positions: [Quat; 3],
    pub weights: [Quat; 3],
    /// Per-edge LOD levels [edge_a, edge_b, edge_c] where:
    /// edge_a = edge opposite vertex 0 (connecting verts 1,2)
    /// edge_b = edge opposite vertex 1 (connecting verts 0,2)
    /// edge_c = edge opposite vertex 2 (connecting verts 0,1)
    pub edge_lods: [u32; 3],
}

/// Compute per-face instance data with adaptive LOD.
pub fn compute_instances(
    vertices: &[[f64; 3]],
    faces: &[[usize; 3]],
    transform: &Mobius,
) -> Vec<FaceInstance> {
    // Pre-transform all vertices
    let transformed: Vec<(Quat, Quat)> = vertices.iter().map(|v| {
        let p = Quat::from_point(v[0], v[1], v[2]);
        let p_prime = transform.apply(p);
        let w_prime = transform.transform_weight(p, Quat::ONE);
        (p_prime, w_prime)
    }).collect();

    // Build face instances
    let instances: Vec<FaceInstance> = faces.iter().map(|face| {
        let (p0, w0) = transformed[face[0]];
        let (p1, w1) = transformed[face[1]];
        let (p2, w2) = transformed[face[2]];
        FaceInstance {
            positions: [p0, p1, p2],
            weights: [w0, w1, w2],
            edge_lods: [1, 1, 1], // placeholder, computed below
        }
    }).collect();

    // Estimate per-face LOD from curvature
    let face_lods: Vec<u32> = instances.iter().map(|inst| {
        estimate_face_lod(inst)
    }).collect();

    // Build edge adjacency: edge (min_vi, max_vi) → list of (face_idx, local_edge_idx)
    let mut edge_faces: HashMap<(usize, usize), Vec<(usize, usize)>> = HashMap::new();
    for (fi, face) in faces.iter().enumerate() {
        // 3 edges per face. local_edge_idx 0 = opposite vert 0 = edge(v1,v2)
        let edges = [
            (face[1], face[2], 0), // edge a: opposite v0
            (face[0], face[2], 1), // edge b: opposite v1
            (face[0], face[1], 2), // edge c: opposite v2
        ];
        for &(va, vb, local_idx) in &edges {
            let key = if va < vb { (va, vb) } else { (vb, va) };
            edge_faces.entry(key).or_default().push((fi, local_idx));
        }
    }

    // Assign per-edge LOD = max of adjacent face LODs, snapped to power of 2
    let mut result = instances;
    for (_edge, face_refs) in &edge_faces {
        let max_lod = face_refs.iter()
            .map(|&(fi, _)| face_lods[fi])
            .max()
            .unwrap_or(1);
        let snapped = snap_to_power_of_2(max_lod);
        for &(fi, local_idx) in face_refs {
            result[fi].edge_lods[local_idx] = snapped;
        }
    }

    // Boundary edges (only one face) — use that face's LOD
    for (fi, face) in faces.iter().enumerate() {
        let edges = [
            (face[1], face[2], 0),
            (face[0], face[2], 1),
            (face[0], face[1], 2),
        ];
        for &(va, vb, local_idx) in &edges {
            let key = if va < vb { (va, vb) } else { (vb, va) };
            if edge_faces.get(&key).map_or(true, |v| v.len() == 1) {
                result[fi].edge_lods[local_idx] = snap_to_power_of_2(face_lods[fi]);
            }
        }
    }

    result
}

/// Estimate face LOD from QB surface curvature.
///
/// From Karpavicius & Krasauskas: LOD = s * (H/L)^p where
/// L = patch size (longest edge), H = distance from QB midpoint
/// to flat midpoint (the "bulge").
fn estimate_face_lod(inst: &FaceInstance) -> u32 {
    let [p0, p1, p2] = inst.positions;
    let [w0, w1, w2] = inst.weights;

    // L: longest edge of the transformed triangle
    let e01 = (p0 - p1).norm();
    let e02 = (p0 - p2).norm();
    let e12 = (p1 - p2).norm();
    let l = e01.max(e02).max(e12);
    if l < 1e-12 { return 1; }

    // Flat midpoint: linear interpolation at centroid
    let flat_mid = (p0 + p1 + p2) * (1.0 / 3.0);

    // QB midpoint: evaluate the rational quaternion surface at centroid
    let patch = QBTriPatch::new([p0, p1, p2], [w0, w1, w2]);
    let qb_mid = patch.eval(1.0 / 3.0, 1.0 / 3.0);

    // H: distance between flat and QB midpoints
    let h = (qb_mid - Quat::new(flat_mid.w, flat_mid.x, flat_mid.y, flat_mid.z)).norm();

    // Also check edge midpoints for better sensitivity
    let edge_mids = [
        (0.5, 0.5, 0.0), // midpoint of edge v0-v1
        (0.5, 0.0, 0.5), // midpoint of edge v0-v2
        (0.0, 0.5, 0.5), // midpoint of edge v1-v2
    ];
    let mut max_h = h;
    for &(b0, b1, b2) in &edge_mids {
        let flat = p0 * b0 + p1 * b1 + p2 * b2;
        let qb = patch.eval(b1, b2);
        let d = (qb - Quat::new(flat.w, flat.x, flat.y, flat.z)).norm();
        max_h = max_h.max(d);
    }

    // LOD formula: higher H/L ratio → more subdivision needed
    let ratio = max_h / l;
    let scale = 64.0;
    let power = 1.0;
    let raw_lod = (scale * ratio.powf(power)).ceil() as u32;

    raw_lod.max(1).min(1024) // capped to atlas max
}

/// Snap to the nearest power of 2 (round up).
fn snap_to_power_of_2(v: u32) -> u32 {
    if v <= 1 { return 1; }
    let mut p = 1u32;
    while p < v { p *= 2; }
    p
}

impl FaceInstance {
    /// Pack as 28 f32s: [p0(4), p1(4), p2(4), w0(4), w1(4), w2(4), lod_a, lod_b, lod_c, pad]
    /// The 7th vec4 carries per-face edge LODs so the vertex shader can compute
    /// density from the correct face's LOD triple, not the batch uniform.
    pub fn to_f32_array(&self) -> [f32; 28] {
        let mut out = [0.0f32; 28];
        for (i, p) in self.positions.iter().enumerate() {
            out[i*4]   = p.w as f32;
            out[i*4+1] = p.x as f32;
            out[i*4+2] = p.y as f32;
            out[i*4+3] = p.z as f32;
        }
        for (i, w) in self.weights.iter().enumerate() {
            out[12+i*4]   = w.w as f32;
            out[12+i*4+1] = w.x as f32;
            out[12+i*4+2] = w.y as f32;
            out[12+i*4+3] = w.z as f32;
        }
        out[24] = self.edge_lods[0] as f32;
        out[25] = self.edge_lods[1] as f32;
        out[26] = self.edge_lods[2] as f32;
        out[27] = 0.0; // padding to vec4 alignment
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shapes;

    #[test]
    fn identity_uniform_lod() {
        let (verts, faces) = shapes::cube();
        let instances = compute_instances(&verts, &faces, &Mobius::identity());
        // Identity transform → no curvature → LOD should be 1 everywhere
        for inst in &instances {
            assert_eq!(inst.edge_lods, [1, 1, 1],
                "identity should have LOD=1, got {:?}", inst.edge_lods);
        }
    }

    #[test]
    fn sphere_reflection_higher_lod() {
        let (verts, faces) = shapes::cube();
        let m = Mobius::sphere_reflection(Quat::from_point(0.5, 0.0, 0.0), 2.0);
        let instances = compute_instances(&verts, &faces, &m);
        // Sphere reflection should produce higher LOD on some faces
        let max_lod = instances.iter()
            .flat_map(|i| i.edge_lods.iter())
            .copied()
            .max()
            .unwrap();
        assert!(max_lod > 1, "sphere reflection should increase LOD, got max={}", max_lod);
    }

    #[test]
    fn adjacent_edges_match() {
        let (verts, faces) = shapes::icosahedron();
        let m = Mobius::sphere_reflection(Quat::from_point(0.3, 0.0, 0.0), 1.5);
        let instances = compute_instances(&verts, &faces, &m);

        // Build edge → LOD map and verify consistency
        let mut edge_lods: HashMap<(usize, usize), Vec<u32>> = HashMap::new();
        for (fi, face) in faces.iter().enumerate() {
            let edges = [
                (face[1], face[2], 0),
                (face[0], face[2], 1),
                (face[0], face[1], 2),
            ];
            for &(va, vb, local_idx) in &edges {
                let key = if va < vb { (va, vb) } else { (vb, va) };
                edge_lods.entry(key).or_default().push(instances[fi].edge_lods[local_idx]);
            }
        }
        for (edge, lods) in &edge_lods {
            let first = lods[0];
            for &l in lods {
                assert_eq!(l, first,
                    "edge {:?} has inconsistent LODs: {:?}", edge, lods);
            }
        }
    }

    #[test]
    fn lods_are_powers_of_2() {
        let (verts, faces) = shapes::octahedron();
        let m = Mobius::sphere_reflection(Quat::from_point(0.2, 0.3, 0.0), 1.8);
        let instances = compute_instances(&verts, &faces, &m);
        for inst in &instances {
            for &l in &inst.edge_lods {
                assert!(l.is_power_of_two(), "LOD {} is not a power of 2", l);
            }
        }
    }
}
