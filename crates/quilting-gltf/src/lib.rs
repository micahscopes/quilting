pub mod mesh;
pub mod material;
pub mod animation;
pub mod scene;
pub mod bake;

use std::fmt;

/// All data extracted from a glTF/GLB file.
#[derive(Debug)]
pub struct GltfScene {
    pub meshes: Vec<mesh::Mesh>,
    pub materials: Vec<material::PbrMaterial>,
    pub animations: Vec<animation::Animation>,
    pub skins: Vec<animation::Skin>,
    pub scenes: Vec<scene::Scene>,
    pub nodes: Vec<scene::Node>,
    /// Index of the default scene, if specified in the glTF.
    pub default_scene: Option<usize>,
}

/// Errors that can occur during glTF loading.
#[derive(Debug)]
pub enum GltfError {
    /// The gltf crate failed to parse the document.
    Parse(gltf::Error),
    /// A buffer or accessor referenced by the document is missing or invalid.
    MissingData(String),
    /// An unsupported feature was encountered.
    Unsupported(String),
}

impl fmt::Display for GltfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GltfError::Parse(e) => write!(f, "glTF parse error: {e}"),
            GltfError::MissingData(s) => write!(f, "missing data: {s}"),
            GltfError::Unsupported(s) => write!(f, "unsupported: {s}"),
        }
    }
}

impl std::error::Error for GltfError {}

impl From<gltf::Error> for GltfError {
    fn from(e: gltf::Error) -> Self {
        GltfError::Parse(e)
    }
}

/// Load a glTF or GLB file from raw bytes.
///
/// Uses `gltf::import_slice` to resolve embedded buffers and images.
/// Returns the full scene contents: meshes, materials, animations, scene graph.
pub fn load_gltf(data: &[u8]) -> Result<GltfScene, GltfError> {
    let (document, buffers, _images) = gltf::import_slice(data)?;

    let meshes = document
        .meshes()
        .map(|m| mesh::extract_mesh(&m, &buffers))
        .collect::<Result<Vec<_>, _>>()?;

    let materials = document
        .materials()
        .map(|m| material::extract_material(&m))
        .collect();

    let animations = document
        .animations()
        .map(|a| animation::extract_animation(&a, &buffers))
        .collect::<Result<Vec<_>, _>>()?;

    let skins = document
        .skins()
        .map(|s| animation::extract_skin(&s, &buffers))
        .collect::<Result<Vec<_>, _>>()?;

    let nodes: Vec<scene::Node> = document
        .nodes()
        .map(|n| scene::extract_node(&n))
        .collect();

    let scenes: Vec<scene::Scene> = document
        .scenes()
        .map(|s| scene::extract_scene(&s))
        .collect();

    let default_scene = document.default_scene().map(|s| s.index());

    Ok(GltfScene {
        meshes,
        materials,
        animations,
        skins,
        scenes,
        nodes,
        default_scene,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn module_structure_compiles() {
        // Verify that all public types are accessible.
        let _: Option<GltfScene> = None;
        let _: Option<GltfError> = None;
        let _: Option<mesh::Mesh> = None;
        let _: Option<material::PbrMaterial> = None;
        let _: Option<animation::Animation> = None;
        let _: Option<scene::Scene> = None;
    }

    #[test]
    fn load_minimal_glb() {
        // Build a minimal valid GLB in memory.
        // GLB = 12-byte header + JSON chunk + BIN chunk
        let json = serde_json::json!({
            "asset": { "version": "2.0" },
            "scene": 0,
            "scenes": [{ "nodes": [0] }],
            "nodes": [{ "mesh": 0 }],
            "meshes": [{
                "primitives": [{
                    "attributes": { "POSITION": 0 },
                    "indices": 1
                }]
            }],
            "accessors": [
                {
                    "bufferView": 0,
                    "componentType": 5126,
                    "count": 3,
                    "type": "VEC3",
                    "max": [1.0, 1.0, 0.0],
                    "min": [0.0, 0.0, 0.0]
                },
                {
                    "bufferView": 1,
                    "componentType": 5123,
                    "count": 3,
                    "type": "SCALAR"
                }
            ],
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0, "byteLength": 36 },
                { "buffer": 0, "byteOffset": 36, "byteLength": 6 }
            ],
            "buffers": [{ "byteLength": 44 }]
        });

        let json_bytes = serde_json::to_vec(&json).unwrap();
        // Pad JSON to 4-byte alignment
        let json_pad = (4 - (json_bytes.len() % 4)) % 4;
        let json_chunk_len = json_bytes.len() + json_pad;

        // BIN data: 3 vertices (3*3*4 = 36 bytes) + 3 u16 indices (3*2 = 6 bytes)
        // = 42 bytes, pad to 44 for 4-byte alignment
        let mut bin = Vec::new();
        // Triangle: (0,0,0), (1,0,0), (0,1,0)
        for &v in &[0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
            bin.extend_from_slice(&v.to_le_bytes());
        }
        // Indices: 0, 1, 2
        for &i in &[0u16, 1, 2] {
            bin.extend_from_slice(&i.to_le_bytes());
        }
        // Pad BIN to 4-byte alignment
        while bin.len() % 4 != 0 {
            bin.push(0);
        }
        let bin_chunk_len = bin.len();

        let total_len = 12 + 8 + json_chunk_len + 8 + bin_chunk_len;
        let mut glb = Vec::with_capacity(total_len);

        // GLB header
        glb.extend_from_slice(b"glTF");                         // magic
        glb.extend_from_slice(&2u32.to_le_bytes());             // version
        glb.extend_from_slice(&(total_len as u32).to_le_bytes()); // length

        // JSON chunk
        glb.extend_from_slice(&(json_chunk_len as u32).to_le_bytes());
        glb.extend_from_slice(&0x4E4F534Au32.to_le_bytes());   // "JSON"
        glb.extend_from_slice(&json_bytes);
        for _ in 0..json_pad {
            glb.push(b' ');
        }

        // BIN chunk
        glb.extend_from_slice(&(bin_chunk_len as u32).to_le_bytes());
        glb.extend_from_slice(&0x004E4942u32.to_le_bytes());   // "BIN\0"
        glb.extend_from_slice(&bin);

        let scene = load_gltf(&glb).expect("failed to load minimal GLB");

        assert_eq!(scene.meshes.len(), 1);
        assert_eq!(scene.meshes[0].primitives.len(), 1);

        let prim = &scene.meshes[0].primitives[0];
        assert_eq!(prim.positions.len(), 3);
        assert_eq!(prim.triangles.len(), 1);
        assert_eq!(prim.triangles[0], [0, 1, 2]);

        // Check vertex values
        let eps = 1e-6;
        assert!((prim.positions[0][0]).abs() < eps);
        assert!((prim.positions[1][0] - 1.0).abs() < eps);
        assert!((prim.positions[2][1] - 1.0).abs() < eps);

        assert_eq!(scene.scenes.len(), 1);
        assert_eq!(scene.nodes.len(), 1);
        assert_eq!(scene.default_scene, Some(0));
    }
}
