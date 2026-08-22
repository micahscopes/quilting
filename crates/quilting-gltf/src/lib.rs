pub mod mesh;
pub mod material;
pub mod animation;
pub mod scene;
pub mod evaluator;
pub mod hyperscape;

use std::fmt;

/// GL wrap mode constants matching WebGL/glow values.
pub const WRAP_REPEAT: u32 = 0x2901;
pub const WRAP_CLAMP_TO_EDGE: u32 = 0x812F;
pub const WRAP_MIRRORED_REPEAT: u32 = 0x8370;

/// Raw encoded image blob (PNG/JPEG bytes, not decoded).
#[derive(Debug, Clone, Default)]
pub struct RawImageBlob {
    pub data: Vec<u8>,
    pub mime_type: String,
}

/// Per-texture raw image + sampler info for browser-native decoding.
#[derive(Debug, Clone)]
pub struct RawTextureInfo {
    pub blob: RawImageBlob,
    pub wrap_s: u32,
    pub wrap_t: u32,
}

/// Decoded image data from a glTF file.
#[derive(Debug, Clone)]
pub struct ImageData {
    /// RGBA pixel data (always converted to RGBA8).
    pub pixels: Vec<u8>,
    pub width: u32,
    pub height: u32,
    /// GL wrap mode for S (U) axis. Default: REPEAT.
    pub wrap_s: u32,
    /// GL wrap mode for T (V) axis. Default: REPEAT.
    pub wrap_t: u32,
}

/// All data extracted from a glTF/GLB file.
#[derive(Debug)]
pub struct GltfScene {
    pub meshes: Vec<mesh::Mesh>,
    pub materials: Vec<material::PbrMaterial>,
    pub animations: Vec<animation::Animation>,
    pub skins: Vec<animation::Skin>,
    pub scenes: Vec<scene::Scene>,
    pub nodes: Vec<scene::Node>,
    /// Decoded images (RGBA8), indexed by glTF image index.
    pub images: Vec<ImageData>,
    /// Mapping from glTF texture index to image index.
    pub texture_to_image: Vec<usize>,
    /// Index of the default scene, if specified in the glTF.
    pub default_scene: Option<usize>,
    /// Optional conformal authoring data from `extras.hyperscape`.
    pub hyperscape: Option<hyperscape::HyperscapeAsset>,
}

/// Like GltfScene but with raw image blobs instead of decoded pixels.
/// Designed for browser-native image decoding via createImageBitmap.
#[derive(Debug)]
pub struct GltfSceneRaw {
    pub meshes: Vec<mesh::Mesh>,
    pub materials: Vec<material::PbrMaterial>,
    pub animations: Vec<animation::Animation>,
    pub skins: Vec<animation::Skin>,
    pub scenes: Vec<scene::Scene>,
    pub nodes: Vec<scene::Node>,
    /// Raw image blobs per texture (ready for browser-native decode).
    pub raw_textures: Vec<RawTextureInfo>,
    /// Mapping from glTF texture index to raw_textures index.
    pub texture_to_image: Vec<usize>,
    pub default_scene: Option<usize>,
    /// Optional conformal authoring data from `extras.hyperscape`.
    pub hyperscape: Option<hyperscape::HyperscapeAsset>,
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
    /// Hyperscape extras are present but malformed or inconsistent.
    Hyperscape(hyperscape::HyperscapeGltfError),
}

impl fmt::Display for GltfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GltfError::Parse(e) => write!(f, "glTF parse error: {e}"),
            GltfError::MissingData(s) => write!(f, "missing data: {s}"),
            GltfError::Unsupported(s) => write!(f, "unsupported: {s}"),
            GltfError::Hyperscape(e) => write!(f, "Hyperscape interchange error: {e}"),
        }
    }
}

impl std::error::Error for GltfError {}

impl From<gltf::Error> for GltfError {
    fn from(e: gltf::Error) -> Self {
        GltfError::Parse(e)
    }
}

impl From<hyperscape::HyperscapeGltfError> for GltfError {
    fn from(e: hyperscape::HyperscapeGltfError) -> Self {
        GltfError::Hyperscape(e)
    }
}

/// Parse only the ordinary node array and optional Hyperscape extras. This is
/// the inexpensive main-thread path; it does not decode images or accessors.
pub fn load_hyperscape_graph(
    data: &[u8],
) -> Result<(Vec<scene::Node>, Option<hyperscape::HyperscapeAsset>), GltfError> {
    let gltf = gltf::Gltf::from_slice(data)
        .or_else(|_| gltf::Gltf::from_slice_without_validation(data))?;
    let nodes = gltf
        .document
        .nodes()
        .map(|node| scene::extract_node(&node))
        .collect();
    let asset = hyperscape::extract_asset(&gltf.document)?;
    Ok((nodes, asset))
}

/// Load a glTF or GLB file from raw bytes.
///
/// Uses `gltf::import_slice` to resolve embedded buffers and images.
/// Returns the full scene contents: meshes, materials, animations, scene graph.
pub fn load_gltf(data: &[u8]) -> Result<GltfScene, GltfError> {
    // Try standard import first, fall back to validation-free parsing
    // to handle files with unsupported required extensions.
    let (document, buffers, raw_images) = match gltf::import_slice(data) {
        Ok(result) => result,
        Err(_) => {
            // Skip all validation (including extensionsRequired checks) so models
            // with KHR_materials_unlit, KHR_texture_basisu, etc. can still load.
            // Textures that need unsupported codecs will be empty — the avg_color
            // fallback in the renderer handles this gracefully.
            let gltf_obj = gltf::Gltf::from_slice_without_validation(data)?;
            let mut buffers = Vec::new();
            if let Some(blob) = gltf_obj.blob {
                buffers.push(gltf::buffer::Data(blob));
            }
            (gltf_obj.document, buffers, vec![])
        }
    };

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
    let hyperscape = hyperscape::extract_asset(&document)?;

    // Convert images to RGBA8
    let images: Vec<ImageData> = raw_images.iter().map(|img| {
        let pixels = match img.format {
            gltf::image::Format::R8G8B8A8 => img.pixels.clone(),
            gltf::image::Format::R8G8B8 => {
                let mut rgba = Vec::with_capacity(img.pixels.len() / 3 * 4);
                for chunk in img.pixels.chunks(3) {
                    rgba.extend_from_slice(chunk);
                    rgba.push(255);
                }
                rgba
            }
            gltf::image::Format::R8G8 => {
                let mut rgba = Vec::with_capacity(img.pixels.len() / 2 * 4);
                for chunk in img.pixels.chunks(2) {
                    rgba.push(chunk[0]);
                    rgba.push(chunk[1]);
                    rgba.push(0);
                    rgba.push(255);
                }
                rgba
            }
            gltf::image::Format::R8 => {
                let mut rgba = Vec::with_capacity(img.pixels.len() * 4);
                for &p in &img.pixels {
                    rgba.push(p);
                    rgba.push(p);
                    rgba.push(p);
                    rgba.push(255);
                }
                rgba
            }
            _ => {
                // For 16-bit or float formats, create a white placeholder
                vec![255u8; (img.width * img.height * 4) as usize]
            }
        };
        ImageData { pixels, width: img.width, height: img.height,
            wrap_s: WRAP_REPEAT, wrap_t: WRAP_REPEAT }
    }).collect();

    // Build per-texture image array: each glTF texture gets its own GL texture
    // with the correct sampler wrap modes. First texture claiming an image takes
    // ownership (no clone); subsequent textures sharing the same image clone.
    let to_gl_wrap = |w: gltf::texture::WrappingMode| -> u32 {
        match w {
            gltf::texture::WrappingMode::ClampToEdge => WRAP_CLAMP_TO_EDGE,
            gltf::texture::WrappingMode::MirroredRepeat => WRAP_MIRRORED_REPEAT,
            gltf::texture::WrappingMode::Repeat => WRAP_REPEAT,
        }
    };
    let mut images = images; // take ownership for move
    let mut first_tex_for_image: Vec<Option<usize>> = vec![None; images.len()];
    let mut tex_images: Vec<ImageData> = Vec::with_capacity(document.textures().len());
    for tex in document.textures() {
        let img_idx = tex.source().index();
        if img_idx >= images.len() { continue; }
        let sampler = tex.sampler();
        let ws = to_gl_wrap(sampler.wrap_s());
        let wt = to_gl_wrap(sampler.wrap_t());
        let pixels = if let Some(first) = first_tex_for_image[img_idx] {
            tex_images[first].pixels.clone()
        } else {
            first_tex_for_image[img_idx] = Some(tex_images.len());
            std::mem::take(&mut images[img_idx].pixels)
        };
        tex_images.push(ImageData {
            pixels, width: images[img_idx].width, height: images[img_idx].height,
            wrap_s: ws, wrap_t: wt,
        });
    }
    // texture_to_image is now identity — index directly into tex_images
    let texture_to_image: Vec<usize> = (0..tex_images.len()).collect();

    Ok(GltfScene {
        meshes,
        materials,
        animations,
        skins,
        scenes,
        nodes,
        images: tex_images,
        texture_to_image,
        default_scene,
        hyperscape,
    })
}

/// Load a glTF/GLB file, parsing structure but returning raw image blobs
/// instead of decoded pixels. Designed for browser-native decoding via
/// createImageBitmap (10-50× faster than WASM PNG decoding).
pub fn load_gltf_raw(data: &[u8]) -> Result<GltfSceneRaw, GltfError> {
    let gltf_obj = gltf::Gltf::from_slice(data)
        .or_else(|_| gltf::Gltf::from_slice_without_validation(data))?;
    let gltf::Gltf { document, blob } = gltf_obj;
    let mut buffers = Vec::new();
    if let Some(blob) = blob {
        // The parsed GLB already owns its BIN chunk. Move it into the accessor
        // backing store instead of cloning the entire embedded payload; large
        // texture-heavy assets can otherwise transiently duplicate tens of
        // megabytes before mesh extraction even begins.
        buffers.push(gltf::buffer::Data(blob));
    }

    let meshes = document.meshes()
        .map(|m| mesh::extract_mesh(&m, &buffers))
        .collect::<Result<Vec<_>, _>>()?;
    let materials = document.materials()
        .map(|m| material::extract_material(&m))
        .collect();
    let animations = document.animations()
        .map(|a| animation::extract_animation(&a, &buffers))
        .collect::<Result<Vec<_>, _>>()?;
    let skins = document.skins()
        .map(|s| animation::extract_skin(&s, &buffers))
        .collect::<Result<Vec<_>, _>>()?;
    let nodes: Vec<scene::Node> = document.nodes()
        .map(|n| scene::extract_node(&n))
        .collect();
    let scenes: Vec<scene::Scene> = document.scenes()
        .map(|s| scene::extract_scene(&s))
        .collect();
    let default_scene = document.default_scene().map(|s| s.index());
    let hyperscape = hyperscape::extract_asset(&document)?;

    // Extract raw image blobs from the GLB buffer
    let mut raw_images: Vec<RawImageBlob> = Vec::new();
    for image in document.images() {
        match image.source() {
            gltf::image::Source::View { view, mime_type } => {
                let buf_idx = view.buffer().index();
                if buf_idx < buffers.len() {
                    let begin = view.offset();
                    let end = begin + view.length();
                    raw_images.push(RawImageBlob {
                        data: buffers[buf_idx].0[begin..end].to_vec(),
                        mime_type: mime_type.to_string(),
                    });
                } else {
                    raw_images.push(RawImageBlob::default());
                }
            }
            gltf::image::Source::Uri { .. } => {
                raw_images.push(RawImageBlob::default());
            }
        }
    }

    // Build per-texture entries with sampler wrap modes
    let to_gl_wrap = |w: gltf::texture::WrappingMode| -> u32 {
        match w {
            gltf::texture::WrappingMode::ClampToEdge => WRAP_CLAMP_TO_EDGE,
            gltf::texture::WrappingMode::MirroredRepeat => WRAP_MIRRORED_REPEAT,
            gltf::texture::WrappingMode::Repeat => WRAP_REPEAT,
        }
    };
    let mut first_tex_for_image: Vec<Option<usize>> = vec![None; raw_images.len()];
    let mut raw_textures: Vec<RawTextureInfo> = Vec::with_capacity(document.textures().len());
    for tex in document.textures() {
        // Some textures use extensions (e.g. KHR_texture_basisu) with no standard source.
        // tex.source() panics on these — catch and skip.
        let source = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| tex.source()));
        let img_idx = match source {
            Ok(img) => img.index(),
            Err(_) => { raw_textures.push(RawTextureInfo { blob: RawImageBlob::default(), wrap_s: WRAP_REPEAT, wrap_t: WRAP_REPEAT }); continue; }
        };
        if img_idx >= raw_images.len() { continue; }
        let sampler = tex.sampler();
        let ws = to_gl_wrap(sampler.wrap_s());
        let wt = to_gl_wrap(sampler.wrap_t());
        let blob = if let Some(first) = first_tex_for_image[img_idx] {
            raw_textures[first].blob.clone()
        } else {
            first_tex_for_image[img_idx] = Some(raw_textures.len());
            std::mem::take(&mut raw_images[img_idx])
        };
        raw_textures.push(RawTextureInfo { blob, wrap_s: ws, wrap_t: wt });
    }
    let texture_to_image: Vec<usize> = (0..raw_textures.len()).collect();

    Ok(GltfSceneRaw {
        meshes, materials, animations, skins, scenes, nodes,
        raw_textures, texture_to_image, default_scene, hyperscape,
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
