//! PBR material extraction from glTF.
//!
//! Maps glTF's metallic-roughness material model to quilting's PBR uniforms.

/// Index of a texture in the glTF texture array.
pub type TextureIndex = usize;

/// Reference to a texture with its UV set.
#[derive(Debug, Clone, Copy)]
pub struct TextureRef {
    /// Index into GltfScene's texture array (from glTF document).
    pub index: TextureIndex,
    /// Which TEXCOORD_n attribute to use (usually 0).
    pub tex_coord: u32,
}

/// PBR metallic-roughness material, extracted from glTF.
#[derive(Debug, Clone)]
pub struct PbrMaterial {
    pub name: Option<String>,

    /// Base color factor (linear RGBA). Multiplied with the base color texture.
    pub base_color_factor: [f64; 4],
    /// Base color texture (sRGB + alpha).
    pub base_color_texture: Option<TextureRef>,

    /// Metallic factor [0..1].
    pub metallic_factor: f64,
    /// Roughness factor [0..1].
    pub roughness_factor: f64,
    /// Metallic-roughness texture (B = metallic, G = roughness).
    pub metallic_roughness_texture: Option<TextureRef>,

    /// Normal map texture.
    pub normal_texture: Option<TextureRef>,
    /// Normal map scale (how much to perturb normals).
    pub normal_scale: f64,

    /// Occlusion texture (R channel).
    pub occlusion_texture: Option<TextureRef>,
    /// Occlusion strength [0..1].
    pub occlusion_strength: f64,

    /// Emissive factor (linear RGB).
    pub emissive_factor: [f64; 3],
    /// Emissive texture (sRGB).
    pub emissive_texture: Option<TextureRef>,

    /// Alpha mode: "OPAQUE", "MASK", or "BLEND".
    pub alpha_mode: AlphaMode,
    /// Alpha cutoff for MASK mode.
    pub alpha_cutoff: f64,
    /// Whether to render both sides of faces.
    pub double_sided: bool,
    /// KHR_materials_unlit: render with base color only, no lighting.
    pub unlit: bool,

    /// KHR_materials_sheen: fabric/velvet appearance.
    pub sheen_color_factor: [f64; 3],
    pub sheen_roughness_factor: f64,

    /// KHR_materials_specular: custom specular color (overrides dielectric F0).
    pub specular_color_factor: [f64; 3],

    /// KHR_texture_transform on normal map
    pub normal_uv_scale: [f64; 2],
    pub normal_uv_offset: [f64; 2],
    pub normal_uv_rotation: f64,

    /// KHR_texture_transform on base color texture
    pub base_uv_scale: [f64; 2],
    pub base_uv_offset: [f64; 2],
    pub base_uv_rotation: f64,
}

/// glTF alpha rendering mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlphaMode {
    Opaque,
    Mask,
    Blend,
}

/// Extract a PBR material from a glTF material.
pub fn extract_material(mat: &gltf::Material<'_>) -> PbrMaterial {
    let pbr = mat.pbr_metallic_roughness();

    let base_color_factor = {
        let f = pbr.base_color_factor();
        [f[0] as f64, f[1] as f64, f[2] as f64, f[3] as f64]
    };

    let base_color_texture = pbr.base_color_texture().map(|info| TextureRef {
        index: info.texture().index(),
        tex_coord: info.tex_coord(),
    });

    let metallic_roughness_texture =
        pbr.metallic_roughness_texture().map(|info| TextureRef {
            index: info.texture().index(),
            tex_coord: info.tex_coord(),
        });

    let normal_texture = mat.normal_texture().map(|info| TextureRef {
        index: info.texture().index(),
        tex_coord: info.tex_coord(),
    });
    let normal_scale = mat
        .normal_texture()
        .map(|info| info.scale() as f64)
        .unwrap_or(1.0);

    // KHR_texture_transform on normal texture
    let mut normal_uv_scale = [1.0f64; 2];
    let mut normal_uv_offset = [0.0f64; 2];
    let mut normal_uv_rotation = 0.0f64;
    if let Some(info) = mat.normal_texture() {
        if let Some(t) = info.extension_value("KHR_texture_transform") {
            if let Some(s) = t.get("scale").and_then(|v| v.as_array()) {
                normal_uv_scale[0] = s.get(0).and_then(|v| v.as_f64()).unwrap_or(1.0);
                normal_uv_scale[1] = s.get(1).and_then(|v| v.as_f64()).unwrap_or(1.0);
            }
            if let Some(o) = t.get("offset").and_then(|v| v.as_array()) {
                normal_uv_offset[0] = o.get(0).and_then(|v| v.as_f64()).unwrap_or(0.0);
                normal_uv_offset[1] = o.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0);
            }
            normal_uv_rotation = t.get("rotation").and_then(|v| v.as_f64()).unwrap_or(0.0);
        }
    }

    // KHR_texture_transform on base color texture
    let mut base_uv_scale = [1.0f64; 2];
    let mut base_uv_offset = [0.0f64; 2];
    let mut base_uv_rotation = 0.0f64;
    if let Some(info) = pbr.base_color_texture() {
        if let Some(t) = info.extension_value("KHR_texture_transform") {
            if let Some(s) = t.get("scale").and_then(|v| v.as_array()) {
                base_uv_scale[0] = s.get(0).and_then(|v| v.as_f64()).unwrap_or(1.0);
                base_uv_scale[1] = s.get(1).and_then(|v| v.as_f64()).unwrap_or(1.0);
            }
            if let Some(o) = t.get("offset").and_then(|v| v.as_array()) {
                base_uv_offset[0] = o.get(0).and_then(|v| v.as_f64()).unwrap_or(0.0);
                base_uv_offset[1] = o.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0);
            }
            base_uv_rotation = t.get("rotation").and_then(|v| v.as_f64()).unwrap_or(0.0);
        }
    }

    let occlusion_texture = mat.occlusion_texture().map(|info| TextureRef {
        index: info.texture().index(),
        tex_coord: info.tex_coord(),
    });
    let occlusion_strength = mat
        .occlusion_texture()
        .map(|info| info.strength() as f64)
        .unwrap_or(1.0);

    let emissive_factor = {
        let e = mat.emissive_factor();
        [e[0] as f64, e[1] as f64, e[2] as f64]
    };

    let emissive_texture = mat.emissive_texture().map(|info| TextureRef {
        index: info.texture().index(),
        tex_coord: info.tex_coord(),
    });

    let alpha_mode = match mat.alpha_mode() {
        gltf::material::AlphaMode::Opaque => AlphaMode::Opaque,
        gltf::material::AlphaMode::Mask => AlphaMode::Mask,
        gltf::material::AlphaMode::Blend => AlphaMode::Blend,
    };

    let alpha_cutoff = mat.alpha_cutoff().unwrap_or(0.5) as f64;

    // KHR_materials_sheen
    let (sheen_color_factor, sheen_roughness_factor) = {
        let mut sc = [0.0; 3];
        let mut sr = 0.0;
        if let Some(ext) = mat.extensions() {
            if let Some(sheen) = ext.get("KHR_materials_sheen") {
                if let Some(c) = sheen.get("sheenColorFactor").and_then(|v| v.as_array()) {
                    for (i, v) in c.iter().take(3).enumerate() {
                        sc[i] = v.as_f64().unwrap_or(0.0);
                    }
                }
                sr = sheen.get("sheenRoughnessFactor").and_then(|v| v.as_f64()).unwrap_or(0.0);
            }
        }
        (sc, sr)
    };

    // KHR_materials_specular
    let specular_color_factor = {
        let mut sc = [1.0; 3]; // default white (no override)
        if let Some(ext) = mat.extensions() {
            if let Some(spec) = ext.get("KHR_materials_specular") {
                if let Some(c) = spec.get("specularColorFactor").and_then(|v| v.as_array()) {
                    for (i, v) in c.iter().take(3).enumerate() {
                        sc[i] = v.as_f64().unwrap_or(1.0);
                    }
                }
            }
        }
        sc
    };

    PbrMaterial {
        name: mat.name().map(|s| s.to_string()),
        base_color_factor,
        base_color_texture,
        metallic_factor: pbr.metallic_factor() as f64,
        roughness_factor: pbr.roughness_factor() as f64,
        metallic_roughness_texture,
        normal_texture,
        normal_scale,
        occlusion_texture,
        occlusion_strength,
        emissive_factor,
        emissive_texture,
        alpha_mode,
        alpha_cutoff,
        double_sided: mat.double_sided(),
        unlit: mat.unlit(),
        sheen_color_factor,
        sheen_roughness_factor,
        specular_color_factor,
        normal_uv_scale,
        normal_uv_offset,
        normal_uv_rotation,
        base_uv_scale,
        base_uv_offset,
        base_uv_rotation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_pbr_values() {
        // Verify the default material struct has sensible PBR defaults.
        let mat = PbrMaterial {
            name: None,
            base_color_factor: [1.0, 1.0, 1.0, 1.0],
            base_color_texture: None,
            metallic_factor: 1.0,
            roughness_factor: 1.0,
            metallic_roughness_texture: None,
            normal_texture: None,
            normal_scale: 1.0,
            occlusion_texture: None,
            occlusion_strength: 1.0,
            emissive_factor: [0.0, 0.0, 0.0],
            emissive_texture: None,
            alpha_mode: AlphaMode::Opaque,
            alpha_cutoff: 0.5,
            double_sided: false,
            unlit: false,
            sheen_color_factor: [0.0; 3],
            sheen_roughness_factor: 0.0,
            specular_color_factor: [1.0; 3],
            normal_uv_scale: [1.0; 2],
            normal_uv_offset: [0.0; 2],
            normal_uv_rotation: 0.0,
            base_uv_scale: [1.0; 2],
            base_uv_offset: [0.0; 2],
            base_uv_rotation: 0.0,
        };
        assert_eq!(mat.alpha_mode, AlphaMode::Opaque);
        assert!((mat.metallic_factor - 1.0).abs() < 1e-10);
        assert!((mat.roughness_factor - 1.0).abs() < 1e-10);
    }
}
