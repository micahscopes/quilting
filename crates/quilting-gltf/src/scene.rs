//! Scene graph extraction from glTF.
//!
//! Builds a flat node array (indexed by glTF node index) with TRS transforms,
//! and provides world-space flattening.

/// A node in the scene graph.
#[derive(Debug, Clone)]
pub struct Node {
    pub name: Option<String>,
    /// Transform, stored as TRS or a raw 4x4 matrix (column-major).
    pub transform: Transform,
    /// Child node indices (into GltfScene::nodes).
    pub children: Vec<usize>,
    /// Mesh index in GltfScene::meshes, if this node carries geometry.
    pub mesh: Option<usize>,
    /// Skin index in GltfScene::skins, if this node is skinned.
    pub skin: Option<usize>,
}

/// Node transform — either decomposed TRS or a raw matrix.
#[derive(Debug, Clone)]
pub enum Transform {
    Trs {
        translation: [f64; 3],
        rotation: [f64; 4], // [x, y, z, w] quaternion
        scale: [f64; 3],
    },
    Matrix([f64; 16]), // column-major 4x4
}

impl Transform {
    /// Convert to a column-major 4x4 matrix.
    pub fn to_matrix(&self) -> [f64; 16] {
        match self {
            Transform::Matrix(m) => *m,
            Transform::Trs {
                translation,
                rotation,
                scale,
            } => {
                let [qx, qy, qz, qw] = *rotation;
                let [sx, sy, sz] = *scale;
                let [tx, ty, tz] = *translation;

                // Rotation matrix from quaternion, with scale baked in.
                let x2 = qx + qx;
                let y2 = qy + qy;
                let z2 = qz + qz;
                let xx = qx * x2;
                let xy = qx * y2;
                let xz = qx * z2;
                let yy = qy * y2;
                let yz = qy * z2;
                let zz = qz * z2;
                let wx = qw * x2;
                let wy = qw * y2;
                let wz = qw * z2;

                [
                    (1.0 - (yy + zz)) * sx, (xy + wz) * sx,         (xz - wy) * sx,         0.0,
                    (xy - wz) * sy,         (1.0 - (xx + zz)) * sy, (yz + wx) * sy,         0.0,
                    (xz + wy) * sz,         (yz - wx) * sz,         (1.0 - (xx + yy)) * sz, 0.0,
                    tx,                     ty,                     tz,                     1.0,
                ]
            }
        }
    }
}

/// A scene — just a list of root node indices.
#[derive(Debug, Clone)]
pub struct Scene {
    pub name: Option<String>,
    pub root_nodes: Vec<usize>,
}

/// Extract a node from the glTF document.
pub fn extract_node(node: &gltf::Node<'_>) -> Node {
    let transform = match node.transform() {
        gltf::scene::Transform::Decomposed {
            translation,
            rotation,
            scale,
        } => Transform::Trs {
            translation: [translation[0] as f64, translation[1] as f64, translation[2] as f64],
            rotation: [
                rotation[0] as f64,
                rotation[1] as f64,
                rotation[2] as f64,
                rotation[3] as f64,
            ],
            scale: [scale[0] as f64, scale[1] as f64, scale[2] as f64],
        },
        gltf::scene::Transform::Matrix { matrix } => {
            let mut m = [0.0f64; 16];
            for col in 0..4 {
                for row in 0..4 {
                    m[col * 4 + row] = matrix[col][row] as f64;
                }
            }
            Transform::Matrix(m)
        }
    };

    Node {
        name: node.name().map(|s| s.to_string()),
        transform,
        children: node.children().map(|c| c.index()).collect(),
        mesh: node.mesh().map(|m| m.index()),
        skin: node.skin().map(|s| s.index()),
    }
}

/// Extract a scene from the glTF document.
pub fn extract_scene(scene: &gltf::Scene<'_>) -> Scene {
    Scene {
        name: scene.name().map(|s| s.to_string()),
        root_nodes: scene.nodes().map(|n| n.index()).collect(),
    }
}

/// Multiply two column-major 4x4 matrices: result = a * b.
fn mat4_mul(a: &[f64; 16], b: &[f64; 16]) -> [f64; 16] {
    let mut out = [0.0f64; 16];
    for col in 0..4 {
        for row in 0..4 {
            let mut sum = 0.0;
            for k in 0..4 {
                sum += a[k * 4 + row] * b[col * 4 + k];
            }
            out[col * 4 + row] = sum;
        }
    }
    out
}

/// Flatten the scene graph, returning the world-space transform for every node.
///
/// The returned Vec is indexed by node index (same as GltfScene::nodes).
pub fn compute_world_transforms(nodes: &[Node], scene: &Scene) -> Vec<[f64; 16]> {
    let identity = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];

    let mut world_transforms = vec![identity; nodes.len()];

    fn visit(
        node_idx: usize,
        parent_world: &[f64; 16],
        nodes: &[Node],
        world_transforms: &mut Vec<[f64; 16]>,
    ) {
        let local = nodes[node_idx].transform.to_matrix();
        let world = mat4_mul(parent_world, &local);
        world_transforms[node_idx] = world;
        for &child in &nodes[node_idx].children {
            visit(child, &world, nodes, world_transforms);
        }
    }

    for &root in &scene.root_nodes {
        visit(root, &identity, nodes, &mut world_transforms);
    }

    world_transforms
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_trs_to_matrix() {
        let t = Transform::Trs {
            translation: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
        };
        let m = t.to_matrix();
        let identity = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        for i in 0..16 {
            assert!(
                (m[i] - identity[i]).abs() < 1e-10,
                "mismatch at index {i}: got {} expected {}",
                m[i],
                identity[i]
            );
        }
    }

    #[test]
    fn translation_trs_to_matrix() {
        let t = Transform::Trs {
            translation: [3.0, 4.0, 5.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
        };
        let m = t.to_matrix();
        // Column-major: translation is in m[12], m[13], m[14].
        assert!((m[12] - 3.0).abs() < 1e-10);
        assert!((m[13] - 4.0).abs() < 1e-10);
        assert!((m[14] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn world_transforms_parent_child() {
        let nodes = vec![
            Node {
                name: Some("parent".into()),
                transform: Transform::Trs {
                    translation: [10.0, 0.0, 0.0],
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: [1.0, 1.0, 1.0],
                },
                children: vec![1],
                mesh: None,
                skin: None,
            },
            Node {
                name: Some("child".into()),
                transform: Transform::Trs {
                    translation: [0.0, 5.0, 0.0],
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: [1.0, 1.0, 1.0],
                },
                children: vec![],
                mesh: Some(0),
                skin: None,
            },
        ];

        let scene = Scene {
            name: None,
            root_nodes: vec![0],
        };

        let world = compute_world_transforms(&nodes, &scene);

        // Parent at (10, 0, 0).
        assert!((world[0][12] - 10.0).abs() < 1e-10);
        assert!((world[0][13]).abs() < 1e-10);

        // Child world = parent * child = (10+0, 0+5, 0+0) = (10, 5, 0).
        assert!((world[1][12] - 10.0).abs() < 1e-10);
        assert!((world[1][13] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn scale_trs_to_matrix() {
        let t = Transform::Trs {
            translation: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [2.0, 3.0, 4.0],
        };
        let m = t.to_matrix();
        assert!((m[0] - 2.0).abs() < 1e-10);   // sx
        assert!((m[5] - 3.0).abs() < 1e-10);   // sy
        assert!((m[10] - 4.0).abs() < 1e-10);  // sz
    }
}
