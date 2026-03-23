//! Mesh extraction from glTF primitives.
//!
//! Converts glTF mesh data into the format quilting-core expects:
//! `Vec<[f64; 3]>` for positions and `Vec<[usize; 3]>` for triangle faces.

use gltf::buffer;
use gltf::mesh::Mode;
use crate::GltfError;

/// A single primitive within a mesh (one draw call's worth of geometry).
#[derive(Debug, Clone)]
pub struct Primitive {
    /// Vertex positions as f64 triples.
    pub positions: Vec<[f64; 3]>,
    /// Vertex normals, if present.
    pub normals: Option<Vec<[f64; 3]>>,
    /// Texture coordinates (first set), if present.
    pub uvs: Option<Vec<[f64; 2]>>,
    /// Triangle indices into the positions array.
    pub triangles: Vec<[usize; 3]>,
    /// Index of the material in GltfScene::materials, if assigned.
    pub material_index: Option<usize>,
    /// Per-vertex joint indices (up to 4 joints per vertex). For skinning.
    pub joint_indices: Option<Vec<[u16; 4]>>,
    /// Per-vertex joint weights (up to 4 weights per vertex). For skinning.
    pub joint_weights: Option<Vec<[f32; 4]>>,
    /// Morph target position deltas. morph_targets[target_idx][vertex_idx] = [dx, dy, dz].
    pub morph_targets: Vec<Vec<[f64; 3]>>,
    /// Per-vertex tangents from TANGENT attribute. [tx, ty, tz, sign] where
    /// sign is +1/-1 for bitangent handedness: B = sign * cross(N, T).
    pub tangents: Option<Vec<[f32; 4]>>,
}

/// A mesh is a collection of primitives.
#[derive(Debug, Clone)]
pub struct Mesh {
    pub name: Option<String>,
    pub primitives: Vec<Primitive>,
}

/// Extract all primitives from a glTF mesh.
pub fn extract_mesh(
    mesh: &gltf::Mesh<'_>,
    buffers: &[buffer::Data],
) -> Result<Mesh, GltfError> {
    let mut primitives = Vec::new();

    for prim in mesh.primitives() {
        if prim.mode() != Mode::Triangles {
            // Skip non-triangle primitives (lines, points, strips, fans).
            continue;
        }

        let reader = prim.reader(|buf| Some(&buffers[buf.index()]));

        // Positions (required by glTF spec for rendered primitives).
        let positions: Vec<[f64; 3]> = match reader.read_positions() {
            Some(iter) => iter.map(|p| [p[0] as f64, p[1] as f64, p[2] as f64]).collect(),
            None => continue, // Skip primitives without positions
        };

        // Normals (optional).
        let normals: Option<Vec<[f64; 3]>> = reader
            .read_normals()
            .map(|iter| iter.map(|n| [n[0] as f64, n[1] as f64, n[2] as f64]).collect());

        // Texture coordinates (first set, optional).
        let uvs: Option<Vec<[f64; 2]>> = reader
            .read_tex_coords(0)
            .map(|iter| {
                iter.into_f32()
                    .map(|uv| [uv[0] as f64, uv[1] as f64])
                    .collect()
            });

        // Tangents (optional) — vec4: xyz = tangent direction, w = bitangent sign.
        let tangents: Option<Vec<[f32; 4]>> = reader
            .read_tangents()
            .map(|iter| iter.collect());

        // Indices — handle both indexed and non-indexed meshes.
        let triangles = if let Some(indices_reader) = reader.read_indices() {
            let indices: Vec<usize> = indices_reader.into_u32().map(|i| i as usize).collect();
            if indices.len() % 3 != 0 {
                return Err(GltfError::MissingData(format!(
                    "index count {} not divisible by 3",
                    indices.len()
                )));
            }
            indices.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
        } else {
            // Non-indexed: every 3 vertices form a triangle.
            if positions.len() % 3 != 0 {
                return Err(GltfError::MissingData(format!(
                    "non-indexed vertex count {} not divisible by 3",
                    positions.len()
                )));
            }
            (0..positions.len())
                .step_by(3)
                .map(|i| [i, i + 1, i + 2])
                .collect()
        };

        // Joint indices (JOINTS_0) for skinning — up to 4 joints per vertex.
        let joint_indices: Option<Vec<[u16; 4]>> = reader
            .read_joints(0)
            .map(|iter| iter.into_u16().collect());

        // Joint weights (WEIGHTS_0) for skinning.
        let joint_weights: Option<Vec<[f32; 4]>> = reader
            .read_weights(0)
            .map(|iter| iter.into_f32().collect());

        // Morph targets — position deltas for each target.
        let morph_targets: Vec<Vec<[f64; 3]>> = {
            let morph_reader = prim.reader(|buf| Some(&buffers[buf.index()]));
            morph_reader.read_morph_targets()
                .map(|mt: (Option<gltf::mesh::util::ReadPositions<'_>>, _, _)| {
                    mt.0.map(|iter| iter.map(|p| [p[0] as f64, p[1] as f64, p[2] as f64]).collect())
                        .unwrap_or_default()
                })
                .collect()
        };

        let material_index = prim.material().index();

        primitives.push(Primitive {
            positions,
            normals,
            uvs,
            triangles,
            material_index,
            joint_indices,
            joint_weights,
            morph_targets,
            tangents,
        });
    }

    Ok(Mesh {
        name: mesh.name().map(|s| s.to_string()),
        primitives,
    })
}

/// Flatten a mesh into a single (positions, triangles) pair by merging all primitives.
///
/// This is the format `quilting_core::evaluate::compute_instances` expects.
/// Primitives are concatenated with index offsets adjusted accordingly.
pub fn flatten_mesh(mesh: &Mesh) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let mut positions = Vec::new();
    let mut triangles = Vec::new();

    for prim in &mesh.primitives {
        let offset = positions.len();
        positions.extend_from_slice(&prim.positions);
        triangles.extend(
            prim.triangles
                .iter()
                .map(|t| [t[0] + offset, t[1] + offset, t[2] + offset]),
        );
    }

    (positions, triangles)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flatten_merges_primitives() {
        let mesh = Mesh {
            name: Some("test".into()),
            primitives: vec![
                Primitive {
                    positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    normals: None,
                    uvs: None,
                    triangles: vec![[0, 1, 2]],
                    material_index: None,
                    joint_indices: None,
                    joint_weights: None,
                    morph_targets: vec![],
                    tangents: None,
                },
                Primitive {
                    positions: vec![[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 1.0, 0.0]],
                    normals: None,
                    uvs: None,
                    triangles: vec![[0, 1, 2]],
                    material_index: None,
                    joint_indices: None,
                    joint_weights: None,
                    morph_targets: vec![],
                    tangents: None,
                },
            ],
        };

        let (pos, tris) = flatten_mesh(&mesh);
        assert_eq!(pos.len(), 6);
        assert_eq!(tris.len(), 2);
        // Second primitive's indices should be offset by 3.
        assert_eq!(tris[1], [3, 4, 5]);
    }
}
