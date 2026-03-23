use crate::quaternion::{Quat, Mobius};
use quilting_mesh::HalfEdgeMesh;
use std::cell::RefCell;

thread_local! {
    static TESS_DENSITY: RefCell<f64> = RefCell::new(20.0);
    static SCREEN_ATTEN: RefCell<bool> = RefCell::new(true);
    static MIN_PX_PER_SUB: RefCell<f64> = RefCell::new(2.0);
}

/// Get current tessellation density.
pub fn get_tess_density() -> f64 {
    TESS_DENSITY.with(|d| *d.borrow())
}

/// Set tessellation parameters.
pub fn set_tess_params(density: f64, screen_atten: bool) {
    TESS_DENSITY.with(|d| *d.borrow_mut() = density.max(1.0));
    SCREEN_ATTEN.with(|s| *s.borrow_mut() = screen_atten);
}

/// Set minimum pixels per subdivision for screen attenuation.
pub fn set_min_px_per_sub(px: f64) {
    MIN_PX_PER_SUB.with(|p| *p.borrow_mut() = px.max(0.1));
}

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
    /// Per-vertex LOD levels [v0, v1, v2] — max of all edges meeting at each vertex.
    /// Used for smooth density visualization that's continuous across face boundaries.
    pub vertex_lods: [u32; 3],
    /// Per-vertex texture coordinates [uv0, uv1, uv2].
    /// Sourced from glTF TEXCOORD_0; defaults to (0,0) when absent.
    pub uvs: [[f32; 2]; 3],
    /// Per-vertex smooth normals [n0, n1, n2].
    /// Sourced from glTF NORMAL attribute; defaults to face normal when absent.
    pub normals: [[f32; 3]; 3],
}

/// Compute per-face instance data with adaptive LOD.
/// Screen-space projection info for LOD computation.
pub struct ScreenInfo {
    pub vp_matrix: [f64; 16], // column-major view-projection
    pub width: f64,
    pub height: f64,
}

impl ScreenInfo {
    /// Project a 3D point to screen pixels. Returns None if behind camera.
    pub fn project(&self, p: [f64; 3]) -> Option<[f64; 2]> {
        let m = &self.vp_matrix;
        let x = m[0]*p[0] + m[4]*p[1] + m[8]*p[2] + m[12];
        let y = m[1]*p[0] + m[5]*p[1] + m[9]*p[2] + m[13];
        let w = m[3]*p[0] + m[7]*p[1] + m[11]*p[2] + m[15];
        if w.abs() < 1e-10 { return None; } // at infinity or behind camera
        let ndc_x = x / w;
        let ndc_y = y / w;
        Some([
            (ndc_x * 0.5 + 0.5) * self.width,
            (ndc_y * 0.5 + 0.5) * self.height,
        ])
    }

    /// Screen-space distance between two 3D points. Returns f64::MAX if either is behind camera.
    pub fn screen_distance(&self, a: [f64; 3], b: [f64; 3]) -> f64 {
        match (self.project(a), self.project(b)) {
            (Some(pa), Some(pb)) => {
                let dx = pa[0] - pb[0];
                let dy = pa[1] - pb[1];
                (dx*dx + dy*dy).sqrt()
            }
            _ => f64::MAX, // one or both behind camera → max LOD
        }
    }
}

/// Compute instance data without LOD — just copy vertex positions/weights.
/// Used for the untransformed original mesh display.
pub fn compute_instances_no_lod(
    vertices: &[[f64; 3]],
    faces: &[[usize; 3]],
) -> Vec<FaceInstance> {
    compute_instances_no_lod_with_uvs(vertices, faces, None, None)
}

/// Compute instance data without LOD, with optional per-vertex UVs and normals.
pub fn compute_instances_no_lod_with_uvs(
    vertices: &[[f64; 3]],
    faces: &[[usize; 3]],
    vertex_uvs: Option<&[[f32; 2]]>,
    vertex_normals: Option<&[[f32; 3]]>,
) -> Vec<FaceInstance> {
    faces.iter().map(|face| {
        let nv = vertices.len();
        let v = [
            vertices[face[0].min(nv - 1)],
            vertices[face[1].min(nv - 1)],
            vertices[face[2].min(nv - 1)],
        ];
        let p0 = Quat::from_point(v[0][0], v[0][1], v[0][2]);
        let p1 = Quat::from_point(v[1][0], v[1][1], v[1][2]);
        let p2 = Quat::from_point(v[2][0], v[2][1], v[2][2]);
        let uvs = match vertex_uvs {
            Some(uvs) => [
                uvs.get(face[0]).copied().unwrap_or([0.0, 0.0]),
                uvs.get(face[1]).copied().unwrap_or([0.0, 0.0]),
                uvs.get(face[2]).copied().unwrap_or([0.0, 0.0]),
            ],
            None => [[0.0, 0.0]; 3],
        };
        let normals = match vertex_normals {
            Some(n) => [
                n.get(face[0]).copied().unwrap_or([0.0, 1.0, 0.0]),
                n.get(face[1]).copied().unwrap_or([0.0, 1.0, 0.0]),
                n.get(face[2]).copied().unwrap_or([0.0, 1.0, 0.0]),
            ],
            None => face_normal_f32(&v),
        };
        FaceInstance {
            positions: [p0, p1, p2],
            weights: [Quat::ONE, Quat::ONE, Quat::ONE],
            edge_lods: [1, 1, 1],
            vertex_lods: [1, 1, 1],
            uvs,
            normals,
        }
    }).collect()
}

pub fn compute_instances(
    vertices: &[[f64; 3]],
    faces: &[[usize; 3]],
    transform: &Mobius,
    screen: Option<&ScreenInfo>,
    mesh: Option<&HalfEdgeMesh>,
) -> Vec<FaceInstance> {
    compute_instances_with_uvs(vertices, faces, transform, screen, mesh, None, None)
}

/// Compute Möbius-transformed instances WITHOUT LOD computation.
/// LODs will be filled in by GPU compute.
pub fn compute_instances_xform_only(
    vertices: &[[f64; 3]],
    faces: &[[usize; 3]],
    transform: &Mobius,
    vertex_uvs: Option<&[[f32; 2]]>,
    vertex_normals: Option<&[[f32; 3]]>,
) -> Vec<FaceInstance> {
    let transformed: Vec<(Quat, Quat)> = vertices.iter().map(|v| {
        let p = Quat::from_point(v[0], v[1], v[2]);
        (transform.apply(p), transform.transform_weight(p, Quat::ONE))
    }).collect();

    let mut instances: Vec<FaceInstance> = faces.iter().map(|face| {
        let v = [vertices[face[0]], vertices[face[1]], vertices[face[2]]];
        let (p0, w0) = transformed[face[0]];
        let (p1, w1) = transformed[face[1]];
        let (p2, w2) = transformed[face[2]];
        let uvs = match vertex_uvs {
            Some(uvs) => [uvs[face[0]], uvs[face[1]], uvs[face[2]]],
            None => [[0.0, 0.0]; 3],
        };
        let normals = match vertex_normals {
            Some(n) => [n[face[0]], n[face[1]], n[face[2]]],
            None => face_normal_f32(&v),
        };
        FaceInstance {
            positions: [p0, p1, p2],
            weights: [w0, w1, w2],
            edge_lods: [2, 2, 2], // placeholder, GPU fills in
            vertex_lods: [2, 2, 2],
            uvs,
            normals,
        }
    }).collect();

    // Vertex normals for non-affine transforms
    if !transform.is_affine() {
        let nv = vertices.len();
        let mut vertex_normals_acc = vec![[0.0f64; 3]; nv];
        for face in faces.iter() {
            let p0 = transformed[face[0]].0.to_point();
            let p1 = transformed[face[1]].0.to_point();
            let p2 = transformed[face[2]].0.to_point();
            let e01 = [p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]];
            let e02 = [p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2]];
            let fn_ = [
                e01[1]*e02[2] - e01[2]*e02[1],
                e01[2]*e02[0] - e01[0]*e02[2],
                e01[0]*e02[1] - e01[1]*e02[0],
            ];
            for &vi in face {
                vertex_normals_acc[vi][0] += fn_[0];
                vertex_normals_acc[vi][1] += fn_[1];
                vertex_normals_acc[vi][2] += fn_[2];
            }
        }
        let sign: f64 = -1.0;
        for (fi, face) in faces.iter().enumerate() {
            for vi in 0..3 {
                let n = &vertex_normals_acc[face[vi]];
                let len = (n[0]*n[0] + n[1]*n[1] + n[2]*n[2]).sqrt();
                if len > 1e-10 {
                    instances[fi].normals[vi] = [
                        (sign * n[0] / len) as f32,
                        (sign * n[1] / len) as f32,
                        (sign * n[2] / len) as f32,
                    ];
                }
            }
        }
    }

    instances
}

/// Compute instances with optional per-vertex UVs and normals.
pub fn compute_instances_with_uvs(
    vertices: &[[f64; 3]],
    faces: &[[usize; 3]],
    transform: &Mobius,
    screen: Option<&ScreenInfo>,
    mesh: Option<&HalfEdgeMesh>,
    vertex_uvs: Option<&[[f32; 2]]>,
    vertex_normals: Option<&[[f32; 3]]>,
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
        let v = [vertices[face[0]], vertices[face[1]], vertices[face[2]]];
        let (p0, w0) = transformed[face[0]];
        let (p1, w1) = transformed[face[1]];
        let (p2, w2) = transformed[face[2]];
        let uvs = match vertex_uvs {
            Some(uvs) => [uvs[face[0]], uvs[face[1]], uvs[face[2]]],
            None => [[0.0, 0.0]; 3],
        };
        let normals = match vertex_normals {
            Some(n) => [n[face[0]], n[face[1]], n[face[2]]],
            None => face_normal_f32(&v),
        };
        FaceInstance {
            positions: [p0, p1, p2],
            weights: [w0, w1, w2],
            edge_lods: [1, 1, 1],
            vertex_lods: [1, 1, 1],
            uvs,
            normals,
        }
    }).collect();

    // For affine transforms (identity, translation, rotation), skip LOD computation
    // entirely — the mesh is already tessellated, no QB curvature to capture.
    // Only non-affine (conformal) transforms need per-edge LOD.
    if transform.is_affine() {
        return instances;
    }

    // Per-edge LOD via canonical edge storage.
    // Each edge LOD is stored once per canonical edge index (min of
    // both half-edge indices). Both faces sharing an edge read from
    // the same slot → guaranteed matching. This is the v0.2.0 proven
    // approach that was working before the LOD refactors.

    const MAX_LOD: u32 = 512;
    let tess_density = TESS_DENSITY.with(|d| *d.borrow());
    let screen_atten_enabled = SCREEN_ATTEN.with(|s| *s.borrow());
    let min_px = MIN_PX_PER_SUB.with(|p| *p.borrow());

    let owned_mesh;
    let mesh = match mesh {
        Some(m) => m,
        None => {
            let faces_u32: Vec<[u32; 3]> = faces.iter()
                .map(|f| [f[0] as u32, f[1] as u32, f[2] as u32])
                .collect();
            owned_mesh = HalfEdgeMesh::from_triangles(vertices.len() as u32, &faces_u32);
            &owned_mesh
        }
    };
    let num_half_edges = mesh.half_edges.len();
    let nf = mesh.num_faces as usize;

    let canonical_edge = |he_idx: usize| -> usize {
        match mesh.half_edges[he_idx].twin {
            Some(nz) => he_idx.min((nz.get() - 1) as usize),
            None => he_idx,
        }
    };

    let screen_arc_len = |va: usize, vb: usize| -> f64 {
        match screen {
            Some(s) => {
                let n_samples = 9;
                let mut total = 0.0;
                let mut any_visible = false;
                let mut prev = s.project(transformed[va].0.to_point());
                for i in 1..=n_samples {
                    let t = i as f64 / n_samples as f64;
                    let orig = [
                        vertices[va][0]*(1.0-t) + vertices[vb][0]*t,
                        vertices[va][1]*(1.0-t) + vertices[vb][1]*t,
                        vertices[va][2]*(1.0-t) + vertices[vb][2]*t,
                    ];
                    let tp = transform.apply(Quat::from_point(orig[0], orig[1], orig[2])).to_point();
                    let curr = s.project(tp);
                    if let (Some(p1), Some(p2)) = (prev, curr) {
                        total += ((p2[0]-p1[0]).powi(2) + (p2[1]-p1[1]).powi(2)).sqrt();
                        any_visible = true;
                    }
                    // Skip unprojectable segments (behind camera) instead of
                    // returning MAX — offscreen edges should get min LOD, not max.
                    prev = curr;
                }
                if any_visible { total } else { 0.0 }
            }
            None => {
                let pa = transformed[va].0.to_point();
                let pb = transformed[vb].0.to_point();
                let dx = pa[0]-pb[0]; let dy = pa[1]-pb[1]; let dz = pa[2]-pb[2];
                (dx*dx + dy*dy + dz*dz).sqrt() * 100.0
            }
        }
    };

    let mut edge_lods: Vec<u32> = vec![0; num_half_edges];

    // Compute mesh scale: bounding sphere radius (for edge length normalization)
    let mesh_radius = {
        let (mut cx, mut cy, mut cz) = (0.0, 0.0, 0.0);
        for v in vertices { cx += v[0]; cy += v[1]; cz += v[2]; }
        let n = vertices.len() as f64;
        cx /= n; cy /= n; cz /= n;
        vertices.iter()
            .map(|v| ((v[0]-cx).powi(2) + (v[1]-cy).powi(2) + (v[2]-cz).powi(2)).sqrt())
            .fold(0.0f64, f64::max)
            .max(1e-6)
    };

    // Target triangle edge length in deformed world units.
    let target_size = mesh_radius / tess_density;

    let dist3 = |a: [f64;3], b: [f64;3]| -> f64 {
        ((a[0]-b[0]).powi(2) + (a[1]-b[1]).powi(2) + (a[2]-b[2]).powi(2)).sqrt()
    };

    // Per-face: transform edge midpoints through Möbius, compute deformed
    // medians to determine per-edge LODs. 3 Möbius evals per face.
    let mut edge_lods_world: Vec<u32> = vec![0; num_half_edges];

    for fi in 0..nf {
        let face = faces[fi];
        let v0 = vertices[face[0]];
        let v1 = vertices[face[1]];
        let v2 = vertices[face[2]];

        // Deformed vertex positions (already computed)
        let d0 = transformed[face[0]].0.to_point();
        let d1 = transformed[face[1]].0.to_point();
        let d2 = transformed[face[2]].0.to_point();

        // Transform edge midpoints through Möbius (3 evals per face).
        // These sample the INTERIOR — catching inflation that vertex
        // chord lengths miss when patches flip inside out.
        let xm = |a: [f64;3], b: [f64;3]| -> [f64; 3] {
            transform.apply(Quat::from_point(
                (a[0]+b[0])*0.5, (a[1]+b[1])*0.5, (a[2]+b[2])*0.5
            )).to_point()
        };
        let dm_a = xm(v1, v2); // deformed midpoint of edge BC
        let dm_b = xm(v0, v2); // deformed midpoint of edge AC
        let dm_c = xm(v0, v1); // deformed midpoint of edge AB

        // Deformed medians: vertex → opposite edge's deformed midpoint.
        // median_a (A → mid(BC)) captures scaling of edges AB and AC.
        // median_b (B → mid(AC)) captures scaling of edges AB and BC.
        // median_c (C → mid(AB)) captures scaling of edges AC and BC.
        let median_a = dist3(d0, dm_a);
        let median_b = dist3(d1, dm_b);
        let median_c = dist3(d2, dm_c);

        // Each edge's LOD is driven by the medians from its two endpoints.
        // Edge AB: median_a (from A across BC) + median_b (from B across AC)
        //   → both tell us how much AB's face extends, so average them.
        // Edge BC: median_b + median_c
        // Edge AC: median_a + median_c
        let lod_ab = ((median_a + median_b) * 0.5 / target_size).ceil() as u32;
        let lod_bc = ((median_b + median_c) * 0.5 / target_size).ceil() as u32;
        let lod_ac = ((median_a + median_c) * 0.5 / target_size).ceil() as u32;

        // Map to half-edge canonical edges, take max across adjacent faces
        let he_base = fi * 3;
        // edge_a (v1→v2 = BC), edge_b (v2→v0 = CA), edge_c (v0→v1 = AB)
        edge_lods_world[canonical_edge(he_base + 1)] =
            edge_lods_world[canonical_edge(he_base + 1)].max(lod_bc);
        edge_lods_world[canonical_edge(he_base + 2)] =
            edge_lods_world[canonical_edge(he_base + 2)].max(lod_ac);
        edge_lods_world[canonical_edge(he_base)] =
            edge_lods_world[canonical_edge(he_base)].max(lod_ab);
    }

    // Apply screen attenuation and snap to power of 2.
    // Skip screen_arc_len entirely when attenuation is off — it's the
    // most expensive part (9 Möbius evals per edge).
    for fi in 0..nf {
        for ei in 0..3u32 {
            let he_idx = fi * 3 + ei as usize;
            let canon = canonical_edge(he_idx);
            if edge_lods[canon] != 0 { continue; }

            let lod = if screen_atten_enabled {
                let (va, vb) = mesh.edge_vertices(he_idx as u32);
                let pixels = screen_arc_len(va as usize, vb as usize);
                if pixels > 0.0 {
                    let world = edge_lods_world[canon];
                    let px_per_sub = pixels / world.max(1) as f64;
                    if px_per_sub < min_px {
                        let reduced = (pixels / min_px).ceil() as u32;
                        world.min(reduced)
                    } else {
                        world
                    }
                } else {
                    edge_lods_world[canon]
                }
            } else {
                edge_lods_world[canon]
            };
            edge_lods[canon] = snap_to_power_of_2(lod).max(2).min(MAX_LOD);
        }
    }

    // Assign to faces — read directly from canonical edge storage.
    // Both faces sharing an edge read the same slot → matching guaranteed.
    let mut result = instances;
    for fi in 0..nf {
        let he_base = fi * 3;
        result[fi].edge_lods = [
            edge_lods[canonical_edge(he_base + 1)], // edge_a: v1→v2, opposite v0
            edge_lods[canonical_edge(he_base + 2)], // edge_b: v2→v0, opposite v1
            edge_lods[canonical_edge(he_base)],     // edge_c: v0→v1, opposite v2
        ];
    }

    // Compute per-vertex LOD = max of all edges meeting at each mesh vertex.
    // Vec indexed by vertex ID instead of HashMap.
    let mut vertex_max_lod: Vec<u32> = vec![1; vertices.len()];
    for fi in 0..nf {
        let face = faces[fi];
        let lods = result[fi].edge_lods;
        // edge_a (opposite v0) connects v1,v2 → contributes to v1 and v2
        // edge_b (opposite v1) connects v0,v2 → contributes to v0 and v2
        // edge_c (opposite v2) connects v0,v1 → contributes to v0 and v1
        for &vi in &[face[1], face[2]] {
            vertex_max_lod[vi] = vertex_max_lod[vi].max(lods[0]);
        }
        for &vi in &[face[0], face[2]] {
            vertex_max_lod[vi] = vertex_max_lod[vi].max(lods[1]);
        }
        for &vi in &[face[0], face[1]] {
            vertex_max_lod[vi] = vertex_max_lod[vi].max(lods[2]);
        }
    }

    // Write vertex LODs into each face instance
    for fi in 0..nf {
        let face = faces[fi];
        result[fi].vertex_lods = [
            vertex_max_lod[face[0]],
            vertex_max_lod[face[1]],
            vertex_max_lod[face[2]],
        ];
    }

    // Compute smooth normals from Möbius-deformed geometry.
    // Average face normals at each vertex using the deformed positions.
    if !transform.is_affine() {
        let nv = vertices.len();
        let mut vertex_normals_acc = vec![[0.0f64; 3]; nv];
        // Accumulate face normals at each vertex
        for face in faces.iter() {
            let p0 = transformed[face[0]].0.to_point();
            let p1 = transformed[face[1]].0.to_point();
            let p2 = transformed[face[2]].0.to_point();
            let e01 = [p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]];
            let e02 = [p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2]];
            let fn_ = [
                e01[1]*e02[2] - e01[2]*e02[1],
                e01[2]*e02[0] - e01[0]*e02[2],
                e01[0]*e02[1] - e01[1]*e02[0],
            ];
            for &vi in face {
                vertex_normals_acc[vi][0] += fn_[0];
                vertex_normals_acc[vi][1] += fn_[1];
                vertex_normals_acc[vi][2] += fn_[2];
            }
        }
        // Non-affine Möbius = sphere reflection = surface turns inside-out
        let sign: f64 = -1.0;

        // Normalize and assign to instances
        for fi in 0..nf {
            let face = faces[fi];
            for vi in 0..3 {
                let n = &vertex_normals_acc[face[vi]];
                let len = (n[0]*n[0] + n[1]*n[1] + n[2]*n[2]).sqrt();
                if len > 1e-10 {
                    result[fi].normals[vi] = [
                        (sign * n[0] / len) as f32,
                        (sign * n[1] / len) as f32,
                        (sign * n[2] / len) as f32,
                    ];
                } else {
                    result[fi].normals[vi] = [0.0; 3];
                }
            }
        }
    }

    result
}


/// Snap to the nearest power of 2 with hysteresis bias toward lower LOD.
/// Only snaps UP when the value exceeds 1.3x the lower power of 2.
/// This prevents oscillation at boundaries (e.g., 15.9 vs 16.1 pixels
/// alternating between LOD 16 and LOD 32).
fn snap_to_power_of_2(v: u32) -> u32 {
    if v <= 1 { return 1; }
    if v >= (1 << 30) { return 1 << 30; } // prevent overflow
    let mut p = 1u32;
    while p < v { p *= 2; }
    // p is now the next power of 2 >= v. p/2 is the one below.
    // Use the lower one unless v significantly exceeds it.
    let lower = p / 2;
    if lower > 0 && (v as f32) < (lower as f32) * 1.3 {
        lower
    } else {
        p
    }
}


/// Compute a face normal from 3 vertex positions, returned as [f32; 3] for all 3 corners.
fn face_normal_f32(v: &[[f64; 3]; 3]) -> [[f32; 3]; 3] {
    let e1 = [v[1][0] - v[0][0], v[1][1] - v[0][1], v[1][2] - v[0][2]];
    let e2 = [v[2][0] - v[0][0], v[2][1] - v[0][1], v[2][2] - v[0][2]];
    let nx = e1[1] * e2[2] - e1[2] * e2[1];
    let ny = e1[2] * e2[0] - e1[0] * e2[2];
    let nz = e1[0] * e2[1] - e1[1] * e2[0];
    let len = (nx * nx + ny * ny + nz * nz).sqrt();
    let n = if len > 1e-12 {
        [(nx / len) as f32, (ny / len) as f32, (nz / len) as f32]
    } else {
        [0.0, 1.0, 0.0]
    };
    [n, n, n]
}

impl FaceInstance {
    /// Pack as 52 f32s (13 vec4s = 208 bytes per instance):
    /// [p0(4), p1(4), p2(4), w0(4), w1(4), w2(4), edgeLods(3)+pad, vertexLods(3)+pad,
    ///  uv01(u0,v0,u1,v1), uv2(u2,v2,pad,pad), n0(3)+pad, n1(3)+pad, n2(3)+pad]
    pub fn to_f32_array(&self) -> [f32; 52] {
        let mut out = [0.0f32; 52];
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
        // vec4 #7: edge LODs
        out[24] = self.edge_lods[0] as f32;
        out[25] = self.edge_lods[1] as f32;
        out[26] = self.edge_lods[2] as f32;
        out[27] = 0.0;
        // vec4 #8: vertex LODs (for smooth density visualization)
        out[28] = self.vertex_lods[0] as f32;
        out[29] = self.vertex_lods[1] as f32;
        out[30] = self.vertex_lods[2] as f32;
        out[31] = 0.0;
        self.pack_uvs(&mut out, &self.uvs);
        self.pack_normals(&mut out, &self.normals);
        out
    }

    fn pack_uvs(&self, out: &mut [f32; 52], uvs: &[[f32; 2]; 3]) {
        out[32] = uvs[0][0]; out[33] = uvs[0][1];
        out[34] = uvs[1][0]; out[35] = uvs[1][1];
        out[36] = uvs[2][0]; out[37] = uvs[2][1];
        out[38] = 0.0; out[39] = 0.0;
    }

    fn pack_normals(&self, out: &mut [f32; 52], normals: &[[f32; 3]; 3]) {
        // vec4 #11-13: per-vertex smooth normals
        for i in 0..3 {
            out[40 + i*4]     = normals[i][0];
            out[40 + i*4 + 1] = normals[i][1];
            out[40 + i*4 + 2] = normals[i][2];
            out[40 + i*4 + 3] = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shapes;

    #[test]
    fn identity_lod_proportional_to_edge_length() {
        let (verts, faces) = shapes::cube();
        let instances = compute_instances(&verts, &faces, &Mobius::identity(), None, None);
        // All LODs should be power of 2 and within a reasonable range
        for inst in &instances {
            for &l in &inst.edge_lods {
                assert!(l.is_power_of_two(), "LOD {} not power of 2", l);
                assert!(l >= 1 && l <= 256, "LOD {} out of range", l);
            }
        }
    }

    #[test]
    fn sphere_reflection_higher_lod() {
        let (verts, faces) = shapes::icosahedron();
        // Place pole near a vertex to ensure high distortion on adjacent faces
        let m = Mobius::sphere_reflection(Quat::from_point(0.85, 0.0, 0.0), 0.3);
        let instances = compute_instances(&verts, &faces, &m, None, None);
        let max_lod = instances.iter()
            .flat_map(|i| i.edge_lods.iter())
            .copied()
            .max()
            .unwrap();
        assert!(max_lod > 1, "sphere reflection should increase LOD, got max={}", max_lod);
    }

    #[test]
    fn shared_edges_have_matching_lods() {
        let (verts, faces) = shapes::icosahedron();
        let m = Mobius::sphere_reflection(Quat::from_point(0.3, 0.0, 0.0), 1.5);
        let instances = compute_instances(&verts, &faces, &m, None, None);

        // Build a map from undirected edges to the LODs seen from each adjacent face.
        // Shared edges must have the same LOD from both sides (no T-junctions).
        use std::collections::HashMap;
        let mut edge_lod_map: HashMap<(usize, usize), Vec<(usize, u32)>> = HashMap::new();
        for (fi, face) in faces.iter().enumerate() {
            // edge_a (opposite v0) = v1-v2, edge_b (opposite v1) = v0-v2, edge_c (opposite v2) = v0-v1
            let edges = [
                (face[1].min(face[2]), face[1].max(face[2]), instances[fi].edge_lods[0]),
                (face[0].min(face[2]), face[0].max(face[2]), instances[fi].edge_lods[1]),
                (face[0].min(face[1]), face[0].max(face[1]), instances[fi].edge_lods[2]),
            ];
            for (va, vb, lod) in edges {
                edge_lod_map.entry((va, vb)).or_default().push((fi, lod));
            }
        }
        for ((va, vb), entries) in &edge_lod_map {
            if entries.len() == 2 {
                assert_eq!(entries[0].1, entries[1].1,
                    "edge ({},{}) has mismatched LODs: face {} has {}, face {} has {}",
                    va, vb, entries[0].0, entries[0].1, entries[1].0, entries[1].1);
            }
        }
    }

    #[test]
    fn lods_are_powers_of_2() {
        let (verts, faces) = shapes::octahedron();
        let m = Mobius::sphere_reflection(Quat::from_point(0.2, 0.3, 0.0), 1.8);
        let instances = compute_instances(&verts, &faces, &m, None, None);
        for inst in &instances {
            for &l in &inst.edge_lods {
                assert!(l.is_power_of_two(), "LOD {} is not a power of 2", l);
            }
        }
    }
}
