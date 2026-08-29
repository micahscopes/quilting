//! Backend-neutral authored material values.
//!
//! GPU textures, samplers, environment-map resources, framebuffers, and
//! transient focus/selection state deliberately do not live here. Render
//! backends resolve these stable asset packets into resources owned by their
//! device epoch.

use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;

/// Backend-neutral texture addressing modes. The explicit WebGL conversion
/// keeps the current browser wire format at the adapter boundary; render and
/// asset code should carry this enum instead of GL constants.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextureWrapMode {
    ClampToEdge,
    MirroredRepeat,
    #[default]
    Repeat,
}

impl TextureWrapMode {
    pub const GL_CLAMP_TO_EDGE: u32 = 0x812f;
    pub const GL_MIRRORED_REPEAT: u32 = 0x8370;
    pub const GL_REPEAT: u32 = 0x2901;

    pub const fn from_gl_enum(value: u32) -> Option<Self> {
        match value {
            Self::GL_CLAMP_TO_EDGE => Some(Self::ClampToEdge),
            Self::GL_MIRRORED_REPEAT => Some(Self::MirroredRepeat),
            Self::GL_REPEAT => Some(Self::Repeat),
            _ => None,
        }
    }

    pub const fn as_gl_enum(self) -> u32 {
        match self {
            Self::ClampToEdge => Self::GL_CLAMP_TO_EDGE,
            Self::MirroredRepeat => Self::GL_MIRRORED_REPEAT,
            Self::Repeat => Self::GL_REPEAT,
        }
    }
}

/// Stable metadata for one decoded two-dimensional texture table entry.
/// Pixel ownership is deliberately separate so native byte uploads and
/// browser `ImageBitmap` uploads can share validation without forcing a large
/// copy through WASM memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextureAssetDescriptor {
    pub width: u32,
    pub height: u32,
    pub wrap_s: TextureWrapMode,
    pub wrap_t: TextureWrapMode,
}

impl TextureAssetDescriptor {
    pub fn validate(self) -> Result<(), TextureAssetError> {
        self.rgba8_byte_len().map(|_| ())
    }

    pub fn rgba8_byte_len(self) -> Result<usize, TextureAssetError> {
        if self.width == 0 || self.height == 0 {
            return Err(TextureAssetError::ZeroDimension {
                width: self.width,
                height: self.height,
            });
        }
        u64::from(self.width)
            .checked_mul(u64::from(self.height))
            .and_then(|pixels| pixels.checked_mul(4))
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or(TextureAssetError::Rgba8ByteLengthOverflow {
                width: self.width,
                height: self.height,
            })
    }
}

/// Borrowed native upload packet. Browser-native image handles use the same
/// [`TextureAssetDescriptor`] but remain platform adapter values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rgba8TextureAsset<'a> {
    pub descriptor: TextureAssetDescriptor,
    pub pixels: &'a [u8],
}

impl<'a> Rgba8TextureAsset<'a> {
    pub fn new(
        descriptor: TextureAssetDescriptor,
        pixels: &'a [u8],
    ) -> Result<Self, TextureAssetError> {
        let expected = descriptor.rgba8_byte_len()?;
        if pixels.len() != expected {
            return Err(TextureAssetError::InvalidRgba8ByteLength {
                expected,
                actual: pixels.len(),
            });
        }
        Ok(Self { descriptor, pixels })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextureAssetError {
    ZeroDimension { width: u32, height: u32 },
    Rgba8ByteLengthOverflow { width: u32, height: u32 },
    InvalidRgba8ByteLength { expected: usize, actual: usize },
}

impl fmt::Display for TextureAssetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension { width, height } => {
                write!(
                    formatter,
                    "texture dimensions must be nonzero, got {width}x{height}"
                )
            }
            Self::Rgba8ByteLengthOverflow { width, height } => write!(
                formatter,
                "RGBA8 texture byte length overflows the host for {width}x{height}",
            ),
            Self::InvalidRgba8ByteLength { expected, actual } => write!(
                formatter,
                "RGBA8 texture requires {expected} bytes, got {actual}",
            ),
        }
    }
}

impl Error for TextureAssetError {}

/// Shape of one filtered image-based-lighting asset. The prefiltered cube is a
/// complete power-of-two mip chain in mip-major, six-face-major RGBA order;
/// irradiance is one six-face RGBA level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EnvironmentMapDescriptor {
    pub prefiltered_face_size: u32,
    pub prefiltered_mip_count: u32,
    pub irradiance_face_size: u32,
}

impl EnvironmentMapDescriptor {
    pub fn validate(self) -> Result<(), EnvironmentMapAssetError> {
        if self.prefiltered_face_size == 0 || self.irradiance_face_size == 0 {
            return Err(EnvironmentMapAssetError::ZeroDimension {
                prefiltered_face_size: self.prefiltered_face_size,
                irradiance_face_size: self.irradiance_face_size,
            });
        }
        if !self.prefiltered_face_size.is_power_of_two()
            || !self.irradiance_face_size.is_power_of_two()
        {
            return Err(EnvironmentMapAssetError::NonPowerOfTwoDimension {
                prefiltered_face_size: self.prefiltered_face_size,
                irradiance_face_size: self.irradiance_face_size,
            });
        }
        let expected_mips = self.prefiltered_face_size.ilog2() + 1;
        if self.prefiltered_mip_count != expected_mips {
            return Err(EnvironmentMapAssetError::IncompleteMipChain {
                expected: expected_mips,
                actual: self.prefiltered_mip_count,
            });
        }
        self.prefiltered_rgba32f_len()?;
        self.irradiance_rgba32f_len()?;
        Ok(())
    }

    pub fn prefiltered_rgba32f_len(self) -> Result<usize, EnvironmentMapAssetError> {
        if self.prefiltered_face_size == 0 {
            return Err(EnvironmentMapAssetError::ZeroDimension {
                prefiltered_face_size: self.prefiltered_face_size,
                irradiance_face_size: self.irradiance_face_size,
            });
        }
        let expected_mips = self.prefiltered_face_size.ilog2() + 1;
        if self.prefiltered_mip_count != expected_mips {
            return Err(EnvironmentMapAssetError::IncompleteMipChain {
                expected: expected_mips,
                actual: self.prefiltered_mip_count,
            });
        }
        let texels = (0..self.prefiltered_mip_count).try_fold(0u64, |total, mip| {
            let size = u64::from((self.prefiltered_face_size >> mip).max(1));
            size.checked_mul(size)
                .and_then(|face_texels| face_texels.checked_mul(6))
                .and_then(|mip_texels| total.checked_add(mip_texels))
        });
        rgba32f_component_len(texels, "prefiltered")
    }

    pub fn irradiance_rgba32f_len(self) -> Result<usize, EnvironmentMapAssetError> {
        if self.irradiance_face_size == 0 {
            return Err(EnvironmentMapAssetError::ZeroDimension {
                prefiltered_face_size: self.prefiltered_face_size,
                irradiance_face_size: self.irradiance_face_size,
            });
        }
        let size = u64::from(self.irradiance_face_size);
        let texels = size
            .checked_mul(size)
            .and_then(|face_texels| face_texels.checked_mul(6));
        rgba32f_component_len(texels, "irradiance")
    }
}

/// Borrowed CPU packet generated by the browser HDR filter or a native asset
/// pipeline. Backends may convert RGBA32F to their filterable device format;
/// no backend handle or enum crosses this boundary.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EnvironmentMapAsset<'a> {
    pub descriptor: EnvironmentMapDescriptor,
    pub prefiltered_rgba32f: &'a [f32],
    pub irradiance_rgba32f: &'a [f32],
}

impl<'a> EnvironmentMapAsset<'a> {
    pub fn new(
        descriptor: EnvironmentMapDescriptor,
        prefiltered_rgba32f: &'a [f32],
        irradiance_rgba32f: &'a [f32],
    ) -> Result<Self, EnvironmentMapAssetError> {
        descriptor.validate()?;
        let expected_prefiltered = descriptor.prefiltered_rgba32f_len()?;
        if prefiltered_rgba32f.len() != expected_prefiltered {
            return Err(EnvironmentMapAssetError::InvalidComponentLength {
                field: "prefiltered",
                expected: expected_prefiltered,
                actual: prefiltered_rgba32f.len(),
            });
        }
        let expected_irradiance = descriptor.irradiance_rgba32f_len()?;
        if irradiance_rgba32f.len() != expected_irradiance {
            return Err(EnvironmentMapAssetError::InvalidComponentLength {
                field: "irradiance",
                expected: expected_irradiance,
                actual: irradiance_rgba32f.len(),
            });
        }
        if let Some((index, _)) = prefiltered_rgba32f
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(EnvironmentMapAssetError::NonFiniteComponent {
                field: "prefiltered",
                index,
            });
        }
        if let Some((index, _)) = irradiance_rgba32f
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(EnvironmentMapAssetError::NonFiniteComponent {
                field: "irradiance",
                index,
            });
        }
        Ok(Self {
            descriptor,
            prefiltered_rgba32f,
            irradiance_rgba32f,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnvironmentMapAssetError {
    ZeroDimension {
        prefiltered_face_size: u32,
        irradiance_face_size: u32,
    },
    NonPowerOfTwoDimension {
        prefiltered_face_size: u32,
        irradiance_face_size: u32,
    },
    IncompleteMipChain {
        expected: u32,
        actual: u32,
    },
    ComponentLengthOverflow {
        field: &'static str,
    },
    InvalidComponentLength {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    NonFiniteComponent {
        field: &'static str,
        index: usize,
    },
}

impl fmt::Display for EnvironmentMapAssetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension {
                prefiltered_face_size,
                irradiance_face_size,
            } => write!(
                formatter,
                "environment dimensions must be nonzero, got prefiltered {prefiltered_face_size} and irradiance {irradiance_face_size}",
            ),
            Self::NonPowerOfTwoDimension {
                prefiltered_face_size,
                irradiance_face_size,
            } => write!(
                formatter,
                "environment face dimensions must be powers of two, got prefiltered {prefiltered_face_size} and irradiance {irradiance_face_size}",
            ),
            Self::IncompleteMipChain { expected, actual } => write!(
                formatter,
                "prefiltered environment requires {expected} complete mips, got {actual}",
            ),
            Self::ComponentLengthOverflow { field } => {
                write!(formatter, "{field} environment component length overflowed")
            }
            Self::InvalidComponentLength {
                field,
                expected,
                actual,
            } => write!(
                formatter,
                "{field} environment payload requires {expected} floats, got {actual}",
            ),
            Self::NonFiniteComponent { field, index } => write!(
                formatter,
                "{field} environment component {index} is not finite",
            ),
        }
    }
}

impl Error for EnvironmentMapAssetError {}

fn rgba32f_component_len(
    texels: Option<u64>,
    field: &'static str,
) -> Result<usize, EnvironmentMapAssetError> {
    texels
        .and_then(|texels| texels.checked_mul(4))
        .and_then(|components| usize::try_from(components).ok())
        .ok_or(EnvironmentMapAssetError::ComponentLengthOverflow { field })
}

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

/// Resolve a glTF material-table index with the renderer's established
/// compatibility fallback: the first authored material, then the caller's
/// default when the table is empty. The returned slot is stable enough for
/// backend binding caches; `usize::MAX` denotes the synthetic default.
pub fn pbr_material_for_index<'a>(
    materials: &'a [PbrMaterial],
    default_material: &'a PbrMaterial,
    requested: usize,
) -> (usize, &'a PbrMaterial) {
    if let Some(material) = materials.get(requested) {
        (requested, material)
    } else if let Some(material) = materials.first() {
        (0, material)
    } else {
        (usize::MAX, default_material)
    }
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

    fn descriptor(width: u32, height: u32) -> TextureAssetDescriptor {
        TextureAssetDescriptor {
            width,
            height,
            wrap_s: TextureWrapMode::Repeat,
            wrap_t: TextureWrapMode::ClampToEdge,
        }
    }

    #[test]
    fn texture_wrap_modes_round_trip_the_legacy_gl_wire_values() {
        for mode in [
            TextureWrapMode::ClampToEdge,
            TextureWrapMode::MirroredRepeat,
            TextureWrapMode::Repeat,
        ] {
            assert_eq!(TextureWrapMode::from_gl_enum(mode.as_gl_enum()), Some(mode));
        }
        assert_eq!(TextureWrapMode::from_gl_enum(0), None);
    }

    #[test]
    fn texture_descriptors_are_small_serializable_asset_metadata() {
        let descriptor = descriptor(17, 9);
        descriptor.validate().unwrap();
        assert_eq!(descriptor.rgba8_byte_len().unwrap(), 17 * 9 * 4);
        let encoded = serde_json::to_string(&descriptor).unwrap();
        assert_eq!(
            serde_json::from_str::<TextureAssetDescriptor>(&encoded).unwrap(),
            descriptor,
        );
    }

    #[test]
    fn texture_asset_validation_rejects_bad_dimensions_and_payloads() {
        assert!(matches!(
            descriptor(0, 1).validate(),
            Err(TextureAssetError::ZeroDimension { .. })
        ));
        assert!(matches!(
            descriptor(u32::MAX, u32::MAX).validate(),
            Err(TextureAssetError::Rgba8ByteLengthOverflow { .. })
        ));
        assert_eq!(
            Rgba8TextureAsset::new(descriptor(2, 2), &[0; 15]),
            Err(TextureAssetError::InvalidRgba8ByteLength {
                expected: 16,
                actual: 15,
            }),
        );
        Rgba8TextureAsset::new(descriptor(2, 2), &[0; 16]).unwrap();
    }

    #[test]
    fn environment_descriptor_freezes_complete_cube_payload_shape() {
        let descriptor = EnvironmentMapDescriptor {
            prefiltered_face_size: 4,
            prefiltered_mip_count: 3,
            irradiance_face_size: 2,
        };
        descriptor.validate().unwrap();
        assert_eq!(descriptor.prefiltered_rgba32f_len().unwrap(), 504);
        assert_eq!(descriptor.irradiance_rgba32f_len().unwrap(), 96);
        let encoded = serde_json::to_string(&descriptor).unwrap();
        assert_eq!(
            serde_json::from_str::<EnvironmentMapDescriptor>(&encoded).unwrap(),
            descriptor,
        );
    }

    #[test]
    fn environment_asset_rejects_shape_length_and_finiteness_drift() {
        let descriptor = EnvironmentMapDescriptor {
            prefiltered_face_size: 2,
            prefiltered_mip_count: 2,
            irradiance_face_size: 1,
        };
        let prefiltered = vec![0.25; descriptor.prefiltered_rgba32f_len().unwrap()];
        let irradiance = vec![0.5; descriptor.irradiance_rgba32f_len().unwrap()];
        EnvironmentMapAsset::new(descriptor, &prefiltered, &irradiance).unwrap();

        let mut incomplete = descriptor;
        incomplete.prefiltered_mip_count = 1;
        assert!(matches!(
            incomplete.validate(),
            Err(EnvironmentMapAssetError::IncompleteMipChain { .. })
        ));
        let mut non_power_of_two = descriptor;
        non_power_of_two.irradiance_face_size = 3;
        assert!(matches!(
            non_power_of_two.validate(),
            Err(EnvironmentMapAssetError::NonPowerOfTwoDimension { .. })
        ));
        assert!(matches!(
            EnvironmentMapAsset::new(
                descriptor,
                &prefiltered[..prefiltered.len() - 1],
                &irradiance
            ),
            Err(EnvironmentMapAssetError::InvalidComponentLength {
                field: "prefiltered",
                ..
            })
        ));
        let mut nonfinite = irradiance.clone();
        nonfinite[7] = f32::INFINITY;
        assert_eq!(
            EnvironmentMapAsset::new(descriptor, &prefiltered, &nonfinite),
            Err(EnvironmentMapAssetError::NonFiniteComponent {
                field: "irradiance",
                index: 7,
            }),
        );
    }

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

    #[test]
    fn material_resolution_preserves_authored_and_default_fallbacks() {
        let default = PbrMaterial::default();
        let mut authored = PbrMaterial::default();
        authored.roughness = 0.25;
        let materials = vec![authored.clone()];
        assert_eq!(
            pbr_material_for_index(&materials, &default, 0),
            (0, &authored)
        );
        assert_eq!(
            pbr_material_for_index(&materials, &default, 17),
            (0, &authored)
        );
        assert_eq!(
            pbr_material_for_index(&[], &default, 17),
            (usize::MAX, &default),
        );
    }
}
