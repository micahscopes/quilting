/// Procedural hypermeshes for testing and experimentation.
///
/// Each function generates a 4D mesh (HyperMesh) with vertex trajectories
/// that trace interesting paths through spacetime.

use crate::hyper_mesh::HyperMesh;
use crate::trajectory::{HermiteSegment, VertexTrajectory};

/// Rotating cube: each vertex traces a circular helix in 4D.
///
/// The cube rotates around the Y axis at the given angular speed.
pub fn rotating_cube(duration: f64, angular_speed: f64, num_keyframes: u32) -> HyperMesh {
    let cube_verts: [[f64; 3]; 8] = [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ];

    let cube_faces: Vec<[u32; 3]> = vec![
        [0, 1, 2], [0, 2, 3], // front
        [5, 4, 7], [5, 7, 6], // back
        [4, 0, 3], [4, 3, 7], // left
        [1, 5, 6], [1, 6, 2], // right
        [3, 2, 6], [3, 6, 7], // top
        [4, 5, 1], [4, 1, 0], // bottom
    ];

    let nk = num_keyframes.max(2) as usize;
    let dt = duration / (nk - 1) as f64;

    let trajectories: Vec<VertexTrajectory> = cube_verts
        .iter()
        .map(|&v| {
            let segments: Vec<HermiteSegment> = (0..nk - 1)
                .map(|k| {
                    let t0 = k as f64 * dt;
                    let t1 = (k + 1) as f64 * dt;
                    let theta0 = angular_speed * t0;
                    let theta1 = angular_speed * t1;

                    let pos0 = rotate_y(v, theta0);
                    let pos1 = rotate_y(v, theta1);
                    let vel0 = rotate_y_velocity(v, theta0, angular_speed);
                    let vel1 = rotate_y_velocity(v, theta1, angular_speed);

                    HermiteSegment {
                        t_start: t0,
                        t_end: t1,
                        pos_start: pos0,
                        pos_end: pos1,
                        vel_start: vel0,
                        vel_end: vel1,
                    }
                })
                .collect();

            VertexTrajectory { segments }
        })
        .collect();

    HyperMesh::new(cube_faces, trajectories)
}

/// Breathing sphere: vertices oscillate radially over time.
///
/// Creates interesting fold structures when sliced at an angle.
pub fn breathing_sphere(
    duration: f64,
    frequency: f64,
    amplitude: f64,
    subdivisions: u32,
) -> HyperMesh {
    let (verts, faces) = uv_sphere(subdivisions);

    let nk = (duration * frequency * 4.0).max(8.0) as usize;
    let dt = duration / (nk - 1) as f64;

    let trajectories = verts
        .iter()
        .map(|&v| {
            let r0 = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            let dir = if r0 > 1e-10 {
                [v[0] / r0, v[1] / r0, v[2] / r0]
            } else {
                [0.0, 1.0, 0.0]
            };

            let segments: Vec<HermiteSegment> = (0..nk - 1)
                .map(|k| {
                    let t0 = k as f64 * dt;
                    let t1 = (k + 1) as f64 * dt;

                    let scale0 = 1.0 + amplitude * (std::f64::consts::TAU * frequency * t0).sin();
                    let scale1 = 1.0 + amplitude * (std::f64::consts::TAU * frequency * t1).sin();
                    let dscale0 = amplitude
                        * std::f64::consts::TAU
                        * frequency
                        * (std::f64::consts::TAU * frequency * t0).cos();
                    let dscale1 = amplitude
                        * std::f64::consts::TAU
                        * frequency
                        * (std::f64::consts::TAU * frequency * t1).cos();

                    let pos0 = [dir[0] * r0 * scale0, dir[1] * r0 * scale0, dir[2] * r0 * scale0];
                    let pos1 = [dir[0] * r0 * scale1, dir[1] * r0 * scale1, dir[2] * r0 * scale1];
                    let vel0 = [dir[0] * r0 * dscale0, dir[1] * r0 * dscale0, dir[2] * r0 * dscale0];
                    let vel1 = [dir[0] * r0 * dscale1, dir[1] * r0 * dscale1, dir[2] * r0 * dscale1];

                    HermiteSegment {
                        t_start: t0,
                        t_end: t1,
                        pos_start: pos0,
                        pos_end: pos1,
                        vel_start: vel0,
                        vel_end: vel1,
                    }
                })
                .collect();

            VertexTrajectory { segments }
        })
        .collect();

    HyperMesh::new(faces, trajectories)
}

/// Two spheres passing through each other along the X axis.
///
/// When sliced at an angle, the layers merge and split as the spheres overlap.
pub fn colliding_spheres(
    duration: f64,
    speed: f64,
    separation: f64,
    subdivisions: u32,
) -> HyperMesh {
    let (sphere_verts, sphere_faces) = uv_sphere(subdivisions);
    let n = sphere_verts.len();

    // Two copies of the sphere mesh
    let mut all_faces = sphere_faces.clone();
    let offset = n as u32;
    for face in &sphere_faces {
        all_faces.push([face[0] + offset, face[1] + offset, face[2] + offset]);
    }

    let nk = 16usize;
    let dt = duration / (nk - 1) as f64;

    let mut trajectories = Vec::with_capacity(n * 2);

    // Sphere A: starts at -separation/2, moves right
    for &v in &sphere_verts {
        let segments: Vec<HermiteSegment> = (0..nk - 1)
            .map(|k| {
                let t0 = k as f64 * dt;
                let t1 = (k + 1) as f64 * dt;
                let x_off0 = -separation / 2.0 + speed * t0;
                let x_off1 = -separation / 2.0 + speed * t1;

                HermiteSegment {
                    t_start: t0,
                    t_end: t1,
                    pos_start: [v[0] + x_off0, v[1], v[2]],
                    pos_end: [v[0] + x_off1, v[1], v[2]],
                    vel_start: [speed, 0.0, 0.0],
                    vel_end: [speed, 0.0, 0.0],
                }
            })
            .collect();
        trajectories.push(VertexTrajectory { segments });
    }

    // Sphere B: starts at +separation/2, moves left
    for &v in &sphere_verts {
        let segments: Vec<HermiteSegment> = (0..nk - 1)
            .map(|k| {
                let t0 = k as f64 * dt;
                let t1 = (k + 1) as f64 * dt;
                let x_off0 = separation / 2.0 - speed * t0;
                let x_off1 = separation / 2.0 - speed * t1;

                HermiteSegment {
                    t_start: t0,
                    t_end: t1,
                    pos_start: [v[0] + x_off0, v[1], v[2]],
                    pos_end: [v[0] + x_off1, v[1], v[2]],
                    vel_start: [-speed, 0.0, 0.0],
                    vel_end: [-speed, 0.0, 0.0],
                }
            })
            .collect();
        trajectories.push(VertexTrajectory { segments });
    }

    HyperMesh::new(all_faces, trajectories)
}

/// Twisting torus: the torus rotates around its tube axis over time.
///
/// Rich topology when sliced from different angles.
pub fn twisting_torus(
    duration: f64,
    twist_speed: f64,
    major_r: f64,
    minor_r: f64,
    segments_major: u32,
    segments_minor: u32,
) -> HyperMesh {
    let (verts, faces, angles) = torus_mesh(major_r, minor_r, segments_major, segments_minor);

    let nk = 16usize;
    let dt = duration / (nk - 1) as f64;

    let trajectories = verts
        .iter()
        .enumerate()
        .map(|(vi, &v)| {
            let (theta, _phi) = angles[vi];

            let segments: Vec<HermiteSegment> = (0..nk - 1)
                .map(|k| {
                    let t0 = k as f64 * dt;
                    let t1 = (k + 1) as f64 * dt;
                    let twist0 = twist_speed * t0;
                    let twist1 = twist_speed * t1;

                    let pos0 = torus_point(major_r, minor_r, theta, angles[vi].1 + twist0);
                    let pos1 = torus_point(major_r, minor_r, theta, angles[vi].1 + twist1);

                    // Velocity from derivative of torus_point w.r.t. twist
                    let vel0 = torus_twist_velocity(major_r, minor_r, theta, angles[vi].1 + twist0, twist_speed);
                    let vel1 = torus_twist_velocity(major_r, minor_r, theta, angles[vi].1 + twist1, twist_speed);

                    let _ = v; // suppress unused warning; we use angles directly

                    HermiteSegment {
                        t_start: t0,
                        t_end: t1,
                        pos_start: pos0,
                        pos_end: pos1,
                        vel_start: vel0,
                        vel_end: vel1,
                    }
                })
                .collect();

            VertexTrajectory { segments }
        })
        .collect();

    HyperMesh::new(faces, trajectories)
}

/// Morphing between two shapes.
///
/// Vertex trajectories interpolate smoothly between shape_a and shape_b
/// positions using cubic Hermite with zero endpoint velocities (ease in/out).
///
/// Both shapes must have the same number of vertices and faces.
pub fn morph(
    shape_a: (Vec<[f64; 3]>, Vec<[u32; 3]>),
    shape_b: (Vec<[f64; 3]>, Vec<[u32; 3]>),
    duration: f64,
    num_keyframes: u32,
) -> HyperMesh {
    let (verts_a, faces) = shape_a;
    let (verts_b, _) = shape_b;

    assert_eq!(
        verts_a.len(),
        verts_b.len(),
        "Morph requires same vertex count"
    );

    let nk = num_keyframes.max(2) as usize;
    let dt = duration / (nk - 1) as f64;

    let trajectories = verts_a
        .iter()
        .zip(verts_b.iter())
        .map(|(&a, &b)| {
            let segments: Vec<HermiteSegment> = (0..nk - 1)
                .map(|k| {
                    let t0 = k as f64 * dt;
                    let t1 = (k + 1) as f64 * dt;
                    let frac0 = k as f64 / (nk - 1) as f64;
                    let frac1 = (k + 1) as f64 / (nk - 1) as f64;

                    // Smooth interpolation fraction using smoothstep
                    let s0 = smoothstep(frac0);
                    let s1 = smoothstep(frac1);
                    let ds0 = smoothstep_deriv(frac0) / duration;
                    let ds1 = smoothstep_deriv(frac1) / duration;

                    let pos0 = lerp3(a, b, s0);
                    let pos1 = lerp3(a, b, s1);
                    let diff = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                    let vel0 = [diff[0] * ds0, diff[1] * ds0, diff[2] * ds0];
                    let vel1 = [diff[0] * ds1, diff[1] * ds1, diff[2] * ds1];

                    HermiteSegment {
                        t_start: t0,
                        t_end: t1,
                        pos_start: pos0,
                        pos_end: pos1,
                        vel_start: vel0,
                        vel_end: vel1,
                    }
                })
                .collect();

            VertexTrajectory { segments }
        })
        .collect();

    HyperMesh::new(faces, trajectories)
}

// --- Geometry helpers ---

fn rotate_y(p: [f64; 3], theta: f64) -> [f64; 3] {
    let c = theta.cos();
    let s = theta.sin();
    [p[0] * c + p[2] * s, p[1], -p[0] * s + p[2] * c]
}

fn rotate_y_velocity(p: [f64; 3], theta: f64, omega: f64) -> [f64; 3] {
    let c = theta.cos();
    let s = theta.sin();
    [
        omega * (-p[0] * s + p[2] * c),
        0.0,
        omega * (-p[0] * c - p[2] * s),
    ]
}

fn smoothstep(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn smoothstep_deriv(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    6.0 * t * (1.0 - t)
}

fn lerp3(a: [f64; 3], b: [f64; 3], t: f64) -> [f64; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

/// Generate a UV sphere with the given number of subdivisions.
/// Returns (positions, triangle_faces).
fn uv_sphere(subdivisions: u32) -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
    let stacks = (subdivisions * 4).max(4);
    let slices = (subdivisions * 4).max(4);

    let mut verts = Vec::new();
    let mut faces = Vec::new();

    // Top pole
    verts.push([0.0, 1.0, 0.0]);

    // Middle rows
    for i in 1..stacks {
        let phi = std::f64::consts::PI * i as f64 / stacks as f64;
        let sp = phi.sin();
        let cp = phi.cos();
        for j in 0..slices {
            let theta = std::f64::consts::TAU * j as f64 / slices as f64;
            verts.push([sp * theta.cos(), cp, sp * theta.sin()]);
        }
    }

    // Bottom pole
    let bottom = verts.len() as u32;
    verts.push([0.0, -1.0, 0.0]);

    // Top cap
    for j in 0..slices {
        let j_next = (j + 1) % slices;
        faces.push([0, 1 + j, 1 + j_next]);
    }

    // Middle bands
    for i in 0..(stacks - 2) {
        let row_start = 1 + i * slices;
        let next_row = 1 + (i + 1) * slices;
        for j in 0..slices {
            let j_next = (j + 1) % slices;
            faces.push([row_start + j, next_row + j, next_row + j_next]);
            faces.push([row_start + j, next_row + j_next, row_start + j_next]);
        }
    }

    // Bottom cap
    let last_row = 1 + (stacks - 2) * slices;
    for j in 0..slices {
        let j_next = (j + 1) % slices;
        faces.push([last_row + j, bottom, last_row + j_next]);
    }

    (verts, faces)
}

/// Generate a torus mesh. Returns (positions, faces, per_vertex (theta, phi) angles).
fn torus_mesh(
    major_r: f64,
    minor_r: f64,
    seg_major: u32,
    seg_minor: u32,
) -> (Vec<[f64; 3]>, Vec<[u32; 3]>, Vec<(f64, f64)>) {
    let mut verts = Vec::new();
    let mut angles = Vec::new();
    let mut faces = Vec::new();

    for i in 0..seg_major {
        let theta = std::f64::consts::TAU * i as f64 / seg_major as f64;
        for j in 0..seg_minor {
            let phi = std::f64::consts::TAU * j as f64 / seg_minor as f64;
            verts.push(torus_point(major_r, minor_r, theta, phi));
            angles.push((theta, phi));
        }
    }

    for i in 0..seg_major {
        let i_next = (i + 1) % seg_major;
        for j in 0..seg_minor {
            let j_next = (j + 1) % seg_minor;
            let v00 = i * seg_minor + j;
            let v10 = i_next * seg_minor + j;
            let v01 = i * seg_minor + j_next;
            let v11 = i_next * seg_minor + j_next;
            faces.push([v00, v10, v11]);
            faces.push([v00, v11, v01]);
        }
    }

    (verts, faces, angles)
}

fn torus_point(major_r: f64, minor_r: f64, theta: f64, phi: f64) -> [f64; 3] {
    let r = major_r + minor_r * phi.cos();
    [r * theta.cos(), minor_r * phi.sin(), r * theta.sin()]
}

fn torus_twist_velocity(
    _major_r: f64,
    minor_r: f64,
    theta: f64,
    phi: f64,
    twist_speed: f64,
) -> [f64; 3] {
    // d/dt of torus_point where phi += twist_speed * t
    // d/dphi of [(R + r*cos(phi))*cos(theta), r*sin(phi), (R + r*cos(phi))*sin(theta)]
    //   = [-r*sin(phi)*cos(theta), r*cos(phi), -r*sin(phi)*sin(theta)] * twist_speed
    let sp = phi.sin();
    let cp = phi.cos();
    [
        -minor_r * sp * theta.cos() * twist_speed,
        minor_r * cp * twist_speed,
        -minor_r * sp * theta.sin() * twist_speed,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotating_cube_structure() {
        let mesh = rotating_cube(1.0, 1.0, 8);
        assert_eq!(mesh.num_vertices, 8);
        assert_eq!(mesh.faces.len(), 12);
        assert_eq!(mesh.trajectories.len(), 8);

        let (t_min, t_max) = mesh.time_range();
        assert!((t_min - 0.0).abs() < 1e-12);
        assert!((t_max - 1.0).abs() < 1e-12);
    }

    #[test]
    fn rotating_cube_at_t0_matches_cube() {
        let mesh = rotating_cube(1.0, 1.0, 8);
        let positions = mesh.positions_at(0.0);

        let expected: [[f64; 3]; 8] = [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ];

        for (i, pos) in positions.iter().enumerate() {
            for j in 0..3 {
                assert!(
                    (pos[j] - expected[i][j]).abs() < 1e-10,
                    "Vertex {} component {} mismatch: {} vs {}",
                    i,
                    j,
                    pos[j],
                    expected[i][j]
                );
            }
        }
    }

    #[test]
    fn breathing_sphere_structure() {
        let mesh = breathing_sphere(2.0, 1.0, 0.3, 2);
        // UV sphere with subdivisions=2: 4 stacks, 8 slices
        // 2 poles + (4-1)*8 = 26 verts... but actually (stacks-1)*slices + 2
        // stacks=8, slices=8 -> 2 + 7*8 = 58 verts
        assert!(mesh.num_vertices > 0);
        assert!(!mesh.faces.is_empty());

        let (t_min, t_max) = mesh.time_range();
        assert!((t_min - 0.0).abs() < 1e-12);
        assert!((t_max - 2.0).abs() < 1e-6);
    }

    #[test]
    fn colliding_spheres_structure() {
        let mesh = colliding_spheres(1.0, 1.0, 4.0, 1);
        // Two copies of the sphere
        let (sphere_verts, sphere_faces) = uv_sphere(1);
        let expected_verts = sphere_verts.len() * 2;
        let expected_faces = sphere_faces.len() * 2;
        assert_eq!(mesh.num_vertices as usize, expected_verts);
        assert_eq!(mesh.faces.len(), expected_faces);
    }

    #[test]
    fn twisting_torus_structure() {
        let mesh = twisting_torus(1.0, 1.0, 2.0, 0.5, 8, 6);
        assert_eq!(mesh.num_vertices, 8 * 6);
        assert_eq!(mesh.faces.len() as u32, 8 * 6 * 2);

        let (t_min, t_max) = mesh.time_range();
        assert!((t_min - 0.0).abs() < 1e-12);
        assert!((t_max - 1.0).abs() < 1e-6);
    }

    #[test]
    fn morph_structure() {
        // Morph between two cubes (same topology, different positions)
        let cube_a: Vec<[f64; 3]> = vec![
            [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0],
        ];
        let cube_b: Vec<[f64; 3]> = vec![
            [-2.0, -2.0, -2.0], [2.0, -2.0, -2.0], [2.0, 2.0, -2.0], [-2.0, 2.0, -2.0],
            [-2.0, -2.0, 2.0], [2.0, -2.0, 2.0], [2.0, 2.0, 2.0], [-2.0, 2.0, 2.0],
        ];
        let faces: Vec<[u32; 3]> = vec![
            [0,1,2],[0,2,3],[5,4,7],[5,7,6],[4,0,3],[4,3,7],
            [1,5,6],[1,6,2],[3,2,6],[3,6,7],[4,5,1],[4,1,0],
        ];

        let mesh = morph(
            (cube_a.clone(), faces.clone()),
            (cube_b.clone(), faces),
            1.0,
            8,
        );

        assert_eq!(mesh.num_vertices, 8);
        assert_eq!(mesh.faces.len(), 12);

        // At t=0, should be shape A
        let pos0 = mesh.positions_at(0.0);
        for (i, p) in pos0.iter().enumerate() {
            for j in 0..3 {
                assert!(
                    (p[j] - cube_a[i][j]).abs() < 1e-10,
                    "Morph t=0: vertex {} mismatch",
                    i
                );
            }
        }

        // At t=1, should be shape B
        let pos1 = mesh.positions_at(1.0);
        for (i, p) in pos1.iter().enumerate() {
            for j in 0..3 {
                assert!(
                    (p[j] - cube_b[i][j]).abs() < 1e-10,
                    "Morph t=1: vertex {} mismatch",
                    i
                );
            }
        }
    }

    #[test]
    fn torus_at_t0_matches_static() {
        let mesh = twisting_torus(1.0, 1.0, 2.0, 0.5, 8, 6);
        let positions = mesh.positions_at(0.0);

        // Check a few vertices match the static torus
        let (static_verts, _, _) = torus_mesh(2.0, 0.5, 8, 6);
        for (i, pos) in positions.iter().enumerate() {
            for j in 0..3 {
                assert!(
                    (pos[j] - static_verts[i][j]).abs() < 1e-10,
                    "Torus vertex {} component {} at t=0: {} vs {}",
                    i,
                    j,
                    pos[j],
                    static_verts[i][j]
                );
            }
        }
    }
}
