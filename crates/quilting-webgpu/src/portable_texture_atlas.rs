//! Deterministic baseline-WebGPU packing for dynamically addressed PBR images.
//!
//! Browser WebGPU has neither binding arrays nor GPU-counted multi-draw. A
//! paged `texture_2d_array` is the portable alternative: every page has one
//! fixed extent, while each source image retains its exact dimensions and wrap
//! metadata for manual shader filtering.

use quilting_core::material::TextureAssetDescriptor;
use std::error::Error;
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PortableTextureAtlasLimits {
    pub maximum_dimension: u32,
    pub maximum_layers: u32,
}

impl PortableTextureAtlasLimits {
    fn validate(self) -> Result<Self, PortableTextureAtlasPlanError> {
        if self.maximum_dimension == 0 {
            return Err(PortableTextureAtlasPlanError::ZeroMaximumDimension);
        }
        if self.maximum_layers == 0 {
            return Err(PortableTextureAtlasPlanError::ZeroMaximumLayers);
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PortableTextureAtlasPlacement {
    pub layer: u32,
    pub origin: [u32; 2],
    pub descriptor: TextureAssetDescriptor,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PortableTextureAtlasPlan {
    pub extent: [u32; 2],
    pub layer_count: u32,
    pub placements: Vec<Option<PortableTextureAtlasPlacement>>,
    pub source_texels: u64,
    pub allocated_texels: u64,
}

impl PortableTextureAtlasPlan {
    pub fn build(
        descriptors: &[Option<TextureAssetDescriptor>],
        limits: PortableTextureAtlasLimits,
    ) -> Result<Self, PortableTextureAtlasPlanError> {
        let limits = limits.validate()?;
        let mut occupied = Vec::new();
        let mut source_texels = 0u64;
        let mut maximum_width = 1u32;
        let mut maximum_height = 1u32;
        for (slot, descriptor) in descriptors.iter().copied().enumerate() {
            let Some(descriptor) = descriptor else {
                continue;
            };
            descriptor.validate().map_err(|error| {
                PortableTextureAtlasPlanError::InvalidDescriptor {
                    slot,
                    message: error.to_string(),
                }
            })?;
            if descriptor.width > limits.maximum_dimension
                || descriptor.height > limits.maximum_dimension
            {
                return Err(PortableTextureAtlasPlanError::DimensionLimit {
                    slot,
                    width: descriptor.width,
                    height: descriptor.height,
                    maximum: limits.maximum_dimension,
                });
            }
            source_texels = source_texels
                .checked_add(u64::from(descriptor.width) * u64::from(descriptor.height))
                .ok_or(PortableTextureAtlasPlanError::TexelCountOverflow)?;
            maximum_width = maximum_width.max(descriptor.width);
            maximum_height = maximum_height.max(descriptor.height);
            occupied.push((slot, descriptor));
        }

        if occupied.is_empty() {
            return Ok(Self {
                extent: [1, 1],
                layer_count: 1,
                placements: vec![None; descriptors.len()],
                source_texels: 0,
                allocated_texels: 1,
            });
        }

        occupied.sort_unstable_by(|left, right| {
            right
                .1
                .height
                .cmp(&left.1.height)
                .then_with(|| right.1.width.cmp(&left.1.width))
                .then_with(|| left.0.cmp(&right.0))
        });
        let ideal_side = ceil_integer_sqrt(source_texels);
        let required_side = u64::from(maximum_width.max(maximum_height)).max(ideal_side);
        let page_side = next_power_of_two_capped(required_side, limits.maximum_dimension);
        let mut pages = Vec::<AtlasPage>::new();
        let mut placements = vec![None; descriptors.len()];
        for (slot, descriptor) in occupied {
            let mut placement = None;
            for (layer, page) in pages.iter_mut().enumerate() {
                if let Some(origin) = page.place(descriptor.width, descriptor.height, page_side) {
                    placement = Some((layer as u32, origin));
                    break;
                }
            }
            if placement.is_none() {
                let next_layer = u32::try_from(pages.len())
                    .map_err(|_| PortableTextureAtlasPlanError::LayerCountOverflow)?;
                if next_layer >= limits.maximum_layers {
                    return Err(PortableTextureAtlasPlanError::LayerLimit {
                        required: next_layer.saturating_add(1),
                        maximum: limits.maximum_layers,
                    });
                }
                let mut page = AtlasPage::default();
                let origin = page
                    .place(descriptor.width, descriptor.height, page_side)
                    .expect("validated rectangle fits an empty atlas page");
                pages.push(page);
                placement = Some((next_layer, origin));
            }
            let (layer, origin) = placement.expect("placement or new page was created");
            placements[slot] = Some(PortableTextureAtlasPlacement {
                layer,
                origin,
                descriptor,
            });
        }
        let layer_count = u32::try_from(pages.len())
            .map_err(|_| PortableTextureAtlasPlanError::LayerCountOverflow)?;
        let allocated_texels = u64::from(page_side)
            .checked_mul(u64::from(page_side))
            .and_then(|texels| texels.checked_mul(u64::from(layer_count)))
            .ok_or(PortableTextureAtlasPlanError::TexelCountOverflow)?;
        Ok(Self {
            extent: [page_side, page_side],
            layer_count,
            placements,
            source_texels,
            allocated_texels,
        })
    }

    pub fn occupied_len(&self) -> usize {
        self.placements.iter().flatten().count()
    }

    pub fn utilization_millionths(&self) -> u32 {
        if self.allocated_texels == 0 {
            return 0;
        }
        ((u128::from(self.source_texels) * 1_000_000) / u128::from(self.allocated_texels)) as u32
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PortableTextureAtlasPlanError {
    ZeroMaximumDimension,
    ZeroMaximumLayers,
    InvalidDescriptor {
        slot: usize,
        message: String,
    },
    DimensionLimit {
        slot: usize,
        width: u32,
        height: u32,
        maximum: u32,
    },
    LayerLimit {
        required: u32,
        maximum: u32,
    },
    LayerCountOverflow,
    TexelCountOverflow,
}

impl fmt::Display for PortableTextureAtlasPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroMaximumDimension => formatter.write_str("texture atlas dimension is zero"),
            Self::ZeroMaximumLayers => formatter.write_str("texture atlas layer limit is zero"),
            Self::InvalidDescriptor { slot, message } => {
                write!(formatter, "texture atlas slot {slot} is invalid: {message}")
            }
            Self::DimensionLimit {
                slot,
                width,
                height,
                maximum,
            } => write!(
                formatter,
                "texture atlas slot {slot} is {width}x{height}, exceeding {maximum}",
            ),
            Self::LayerLimit { required, maximum } => write!(
                formatter,
                "texture atlas requires {required} layers, exceeding {maximum}",
            ),
            Self::LayerCountOverflow => formatter.write_str("texture atlas layers exceed u32"),
            Self::TexelCountOverflow => formatter.write_str("texture atlas texel count overflowed"),
        }
    }
}

impl Error for PortableTextureAtlasPlanError {}

#[derive(Default)]
struct AtlasPage {
    shelves: Vec<AtlasShelf>,
}

impl AtlasPage {
    fn place(&mut self, width: u32, height: u32, page_side: u32) -> Option<[u32; 2]> {
        let best = self
            .shelves
            .iter()
            .enumerate()
            .filter(|(_, shelf)| {
                shelf.height >= height && shelf.next_x.saturating_add(width) <= page_side
            })
            .min_by_key(|(_, shelf)| (shelf.height - height, page_side - shelf.next_x - width))
            .map(|(index, _)| index);
        if let Some(index) = best {
            let shelf = &mut self.shelves[index];
            let origin = [shelf.next_x, shelf.y];
            shelf.next_x += width;
            return Some(origin);
        }
        let y = self
            .shelves
            .last()
            .map_or(0, |shelf| shelf.y + shelf.height);
        if width > page_side || y.saturating_add(height) > page_side {
            return None;
        }
        self.shelves.push(AtlasShelf {
            y,
            height,
            next_x: width,
        });
        Some([0, y])
    }
}

struct AtlasShelf {
    y: u32,
    height: u32,
    next_x: u32,
}

fn ceil_integer_sqrt(value: u64) -> u64 {
    if value <= 1 {
        return value;
    }
    let mut low = 1u64;
    let mut high = value.min(u64::from(u32::MAX));
    while low < high {
        let middle = low + (high - low) / 2;
        if middle >= value.div_ceil(middle) {
            high = middle;
        } else {
            low = middle + 1;
        }
    }
    low
}

fn next_power_of_two_capped(value: u64, maximum: u32) -> u32 {
    let value = u32::try_from(value).unwrap_or(maximum).min(maximum);
    value
        .checked_next_power_of_two()
        .unwrap_or(maximum)
        .min(maximum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use quilting_core::material::TextureWrapMode;

    fn descriptor(width: u32, height: u32) -> TextureAssetDescriptor {
        TextureAssetDescriptor {
            width,
            height,
            wrap_s: TextureWrapMode::Repeat,
            wrap_t: TextureWrapMode::ClampToEdge,
        }
    }

    #[test]
    fn deterministic_plan_preserves_sparse_slots_without_overlap() {
        let descriptors = vec![
            Some(descriptor(8, 8)),
            None,
            Some(descriptor(4, 8)),
            Some(descriptor(4, 4)),
            Some(descriptor(4, 4)),
        ];
        let plan = PortableTextureAtlasPlan::build(
            &descriptors,
            PortableTextureAtlasLimits {
                maximum_dimension: 16,
                maximum_layers: 4,
            },
        )
        .unwrap();
        assert_eq!(plan.extent, [16, 16]);
        assert_eq!(plan.layer_count, 1);
        assert_eq!(plan.occupied_len(), 4);
        assert_eq!(plan.source_texels, 128);
        assert_eq!(plan.allocated_texels, 256);
        assert_eq!(plan.utilization_millionths(), 500_000);
        assert_eq!(plan.placements[1], None);
        assert_eq!(plan.placements[0].unwrap().origin, [0, 0]);
        assert_eq!(plan.placements[2].unwrap().origin, [8, 0]);
        assert_eq!(plan.placements[3].unwrap().origin, [12, 0]);
        assert_eq!(plan.placements[4].unwrap().origin, [0, 8]);

        for (left_index, left) in plan.placements.iter().enumerate() {
            let Some(left) = left else { continue };
            for right in plan.placements.iter().skip(left_index + 1).flatten() {
                if left.layer != right.layer {
                    continue;
                }
                let separated = left.origin[0] + left.descriptor.width <= right.origin[0]
                    || right.origin[0] + right.descriptor.width <= left.origin[0]
                    || left.origin[1] + left.descriptor.height <= right.origin[1]
                    || right.origin[1] + right.descriptor.height <= left.origin[1];
                assert!(
                    separated,
                    "atlas placements overlap: {left:?} and {right:?}"
                );
            }
        }
    }

    #[test]
    fn page_growth_is_bounded_and_empty_tables_stay_bindable() {
        let empty = PortableTextureAtlasPlan::build(
            &[None, None],
            PortableTextureAtlasLimits {
                maximum_dimension: 8,
                maximum_layers: 1,
            },
        )
        .unwrap();
        assert_eq!(empty.extent, [1, 1]);
        assert_eq!(empty.layer_count, 1);
        assert_eq!(empty.placements, vec![None, None]);

        let error = PortableTextureAtlasPlan::build(
            &[Some(descriptor(8, 8)), Some(descriptor(8, 8))],
            PortableTextureAtlasLimits {
                maximum_dimension: 8,
                maximum_layers: 1,
            },
        )
        .unwrap_err();
        assert_eq!(
            error,
            PortableTextureAtlasPlanError::LayerLimit {
                required: 2,
                maximum: 1,
            },
        );
    }

    #[test]
    fn invalid_limits_and_oversized_images_fail_before_packing() {
        assert_eq!(
            PortableTextureAtlasPlan::build(
                &[Some(descriptor(1, 1))],
                PortableTextureAtlasLimits {
                    maximum_dimension: 0,
                    maximum_layers: 1,
                },
            ),
            Err(PortableTextureAtlasPlanError::ZeroMaximumDimension),
        );
        assert_eq!(
            PortableTextureAtlasPlan::build(
                &[Some(descriptor(17, 1))],
                PortableTextureAtlasLimits {
                    maximum_dimension: 16,
                    maximum_layers: 1,
                },
            ),
            Err(PortableTextureAtlasPlanError::DimensionLimit {
                slot: 0,
                width: 17,
                height: 1,
                maximum: 16,
            }),
        );
    }
}
