//! Backend-neutral authored material values.
//!
//! GPU textures, samplers, environment maps, framebuffers, and transient
//! focus/selection state deliberately do not live here. Render backends resolve
//! these stable asset references into resources owned by their device epoch.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PbrAlphaMode {
    #[default]
    Opaque,
    Mask,
    Blend,
}

impl PbrAlphaMode {
    pub const fn as_f32(self) -> f32 {
        match self {
            Self::Opaque => 0.0,
            Self::Mask => 1.0,
            Self::Blend => 2.0,
        }
    }

    pub fn from_wire_f32(value: f32) -> Option<Self> {
        match value {
            0.0 => Some(Self::Opaque),
            1.0 => Some(Self::Mask),
            2.0 => Some(Self::Blend),
            _ => None,
        }
    }
}

/// Stable image-table references from one authored glTF material. Missing
/// textures are represented structurally rather than with negative sentinels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PbrTextureReferences {
    pub base_color: Option<u32>,
    pub metallic_roughness: Option<u32>,
    pub normal: Option<u32>,
    pub emissive: Option<u32>,
    pub occlusion: Option<u32>,
    pub transmission: Option<u32>,
}

/// Authored metallic-roughness material state shared by WebGL2, WebGPU,
/// serialization, Blender exchange, and future HHHS scene commands.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PbrMaterial {
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub emissive_factor: [f32; 3],
    pub normal_scale: f32,
    pub occlusion_strength: f32,
    pub alpha_cutoff: f32,
    pub alpha_mode: PbrAlphaMode,
    pub unlit: bool,
    pub double_sided: bool,
    pub sheen_color: [f32; 3],
    pub has_sheen: bool,
    pub sheen_roughness: f32,
    pub specular_color: [f32; 3],
    pub has_specular: bool,
    pub normal_uv_scale: [f32; 2],
    pub normal_uv_offset: [f32; 2],
    pub normal_uv_rotation: f32,
    pub base_uv_scale: [f32; 2],
    pub base_uv_rotation: f32,
    pub ior: f32,
    pub transmission_factor: f32,
    pub thickness_factor: f32,
    pub attenuation_color: [f32; 3],
    /// `None` is glTF's infinite-distance default.
    pub attenuation_distance: Option<f32>,
    pub textures: PbrTextureReferences,
}

impl PbrMaterial {
    pub fn validate(&self) -> Result<(), &'static str> {
        let finite = self
            .base_color
            .into_iter()
            .chain([self.metallic, self.roughness])
            .chain(self.emissive_factor)
            .chain([
                self.normal_scale,
                self.occlusion_strength,
                self.alpha_cutoff,
            ])
            .chain(self.sheen_color)
            .chain([self.sheen_roughness])
            .chain(self.specular_color)
            .chain(self.normal_uv_scale)
            .chain(self.normal_uv_offset)
            .chain([self.normal_uv_rotation])
            .chain(self.base_uv_scale)
            .chain([self.base_uv_rotation, self.ior, self.transmission_factor])
            .chain([self.thickness_factor])
            .chain(self.attenuation_color)
            .all(f32::is_finite);
        if !finite {
            return Err("PBR material contains a non-finite authored value");
        }
        if self.ior <= 0.0 {
            return Err("PBR material IOR must be positive");
        }
        if self
            .attenuation_distance
            .is_some_and(|distance| !distance.is_finite() || distance <= 0.0)
        {
            return Err("finite PBR attenuation distance must be positive");
        }
        Ok(())
    }
}

impl Default for PbrMaterial {
    fn default() -> Self {
        Self {
            base_color: [1.0; 4],
            metallic: 0.0,
            roughness: 1.0,
            emissive_factor: [0.0; 3],
            normal_scale: 1.0,
            occlusion_strength: 1.0,
            alpha_cutoff: 0.5,
            alpha_mode: PbrAlphaMode::Opaque,
            unlit: false,
            double_sided: false,
            sheen_color: [0.0; 3],
            has_sheen: false,
            sheen_roughness: 0.0,
            specular_color: [1.0; 3],
            has_specular: false,
            normal_uv_scale: [1.0; 2],
            normal_uv_offset: [0.0; 2],
            normal_uv_rotation: 0.0,
            base_uv_scale: [1.0; 2],
            base_uv_rotation: 0.0,
            ior: 1.5,
            transmission_factor: 0.0,
            thickness_factor: 0.0,
            attenuation_color: [1.0; 3],
            attenuation_distance: None,
            textures: PbrTextureReferences::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alpha_mode_wire_values_are_exact() {
        assert_eq!(PbrAlphaMode::from_wire_f32(0.0), Some(PbrAlphaMode::Opaque));
        assert_eq!(PbrAlphaMode::from_wire_f32(1.0), Some(PbrAlphaMode::Mask));
        assert_eq!(PbrAlphaMode::from_wire_f32(2.0), Some(PbrAlphaMode::Blend));
        assert_eq!(PbrAlphaMode::from_wire_f32(1.0 + f32::EPSILON), None);
        assert_eq!(PbrAlphaMode::from_wire_f32(1.5), None);
        assert_eq!(PbrAlphaMode::from_wire_f32(f32::NAN), None);
    }

    #[test]
    fn default_material_is_valid_and_serializable() {
        let material = PbrMaterial::default();
        material.validate().unwrap();
        let encoded = serde_json::to_string(&material).unwrap();
        let decoded: PbrMaterial = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, material);
    }

    #[test]
    fn validation_distinguishes_unbounded_and_finite_attenuation() {
        let mut material = PbrMaterial::default();
        material.attenuation_distance = None;
        material.validate().unwrap();
        material.attenuation_distance = Some(2.0);
        material.validate().unwrap();
        material.attenuation_distance = Some(f32::INFINITY);
        assert!(material.validate().is_err());
    }
}
