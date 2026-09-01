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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PortableTextureAtlasMipPlacement {
    pub layer: u32,
    pub origin: [u32; 2],
    pub extent: [u32; 2],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PortableTextureAtlasPlan {
    pub extent: [u32; 2],
    pub layer_count: u32,
    pub mip_level_count: u32,
    pub placements: Vec<Option<PortableTextureAtlasPlacement>>,
    /// Independently packed placements indexed by `[mip][stable texture slot]`.
    /// Independent packing prevents cross-image filtering and avoids padding
    /// every non-power-of-two image to its largest mip alignment.
    pub mip_placements: Vec<Vec<Option<PortableTextureAtlasMipPlacement>>>,
    pub source_texels: u64,
    pub allocated_texels: u64,
    pub source_mip_texels: u64,
    pub allocated_mip_texels: u64,
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
                mip_level_count: 1,
                placements: vec![None; descriptors.len()],
                mip_placements: vec![vec![None; descriptors.len()]],
                source_texels: 0,
                allocated_texels: 1,
                source_mip_texels: 0,
                allocated_mip_texels: 1,
            });
        }

        let minimum_side = next_power_of_two_capped(
            u64::from(maximum_width.max(maximum_height)),
            limits.maximum_dimension,
        );
        let mut page_side = minimum_side;
        let mut candidates = Vec::new();
        let mut last_layer_error = None;
        loop {
            match pack_atlas_candidate(descriptors, page_side, limits.maximum_layers) {
                Ok(candidate) => candidates.push(candidate),
                Err(error @ PortableTextureAtlasPlanError::LayerLimit { .. }) => {
                    last_layer_error = Some(error);
                }
                Err(error) => return Err(error),
            }
            if page_side == limits.maximum_dimension {
                break;
            }
            let next = page_side
                .checked_mul(2)
                .unwrap_or(limits.maximum_dimension)
                .min(limits.maximum_dimension);
            if next == page_side {
                break;
            }
            page_side = next;
        }
        let candidate = candidates.into_iter().min_by_key(|candidate| {
            (
                candidate.allocated_mip_texels,
                candidate.allocated_texels,
                candidate.layer_count,
                candidate.page_side,
            )
        });
        let candidate = match candidate {
            Some(candidate) => candidate,
            None => {
                return Err(last_layer_error.unwrap_or(
                    PortableTextureAtlasPlanError::LayerLimit {
                        required: limits.maximum_layers.saturating_add(1),
                        maximum: limits.maximum_layers,
                    },
                ));
            }
        };
        let page_side = candidate.page_side;
        let layer_count = candidate.layer_count;
        let mip_level_count = candidate.mip_level_count;
        let mip_placements = candidate.mip_placements;
        let source_mip_texels = candidate.source_mip_texels;
        let allocated_texels = candidate.allocated_texels;
        let allocated_mip_texels = candidate.allocated_mip_texels;
        let placements = mip_placements[0]
            .iter()
            .zip(descriptors)
            .map(|(placement, descriptor)| {
                placement.zip(*descriptor).map(|(placement, descriptor)| {
                    PortableTextureAtlasPlacement {
                        layer: placement.layer,
                        origin: placement.origin,
                        descriptor,
                    }
                })
            })
            .collect();
        Ok(Self {
            extent: [page_side, page_side],
            layer_count,
            mip_level_count,
            placements,
            mip_placements,
            source_texels,
            allocated_texels,
            source_mip_texels,
            allocated_mip_texels,
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

    pub fn mip_utilization_millionths(&self) -> u32 {
        if self.allocated_mip_texels == 0 {
            return 0;
        }
        ((u128::from(self.source_mip_texels) * 1_000_000) / u128::from(self.allocated_mip_texels))
            as u32
    }
}

pub fn texture_mip_level_count(width: u32, height: u32) -> u32 {
    u32::BITS - width.max(height).leading_zeros()
}

struct PackedMipLevel {
    placements: Vec<Option<PortableTextureAtlasMipPlacement>>,
    layer_count: u32,
    source_texels: u64,
}

struct PackedAtlasCandidate {
    page_side: u32,
    layer_count: u32,
    mip_level_count: u32,
    mip_placements: Vec<Vec<Option<PortableTextureAtlasMipPlacement>>>,
    source_mip_texels: u64,
    allocated_texels: u64,
    allocated_mip_texels: u64,
}

fn pack_atlas_candidate(
    descriptors: &[Option<TextureAssetDescriptor>],
    page_side: u32,
    maximum_layers: u32,
) -> Result<PackedAtlasCandidate, PortableTextureAtlasPlanError> {
    let mip_level_count = texture_mip_level_count(page_side, page_side);
    let mut mip_placements = Vec::with_capacity(mip_level_count as usize);
    let mut layer_count = 1u32;
    let mut source_mip_texels = 0u64;
    for mip_level in 0..mip_level_count {
        let mip_side = (page_side >> mip_level).max(1);
        let packed = pack_mip_level(descriptors, mip_level, mip_side, maximum_layers)?;
        layer_count = layer_count.max(packed.layer_count);
        source_mip_texels = source_mip_texels
            .checked_add(packed.source_texels)
            .ok_or(PortableTextureAtlasPlanError::TexelCountOverflow)?;
        mip_placements.push(packed.placements);
    }
    let allocated_texels = u64::from(page_side)
        .checked_mul(u64::from(page_side))
        .and_then(|texels| texels.checked_mul(u64::from(layer_count)))
        .ok_or(PortableTextureAtlasPlanError::TexelCountOverflow)?;
    let allocated_mip_texels = (0..mip_level_count).try_fold(0u64, |total, mip_level| {
        let mip_side = u64::from((page_side >> mip_level).max(1));
        mip_side
            .checked_mul(mip_side)
            .and_then(|texels| texels.checked_mul(u64::from(layer_count)))
            .and_then(|texels| total.checked_add(texels))
            .ok_or(PortableTextureAtlasPlanError::TexelCountOverflow)
    })?;
    Ok(PackedAtlasCandidate {
        page_side,
        layer_count,
        mip_level_count,
        mip_placements,
        source_mip_texels,
        allocated_texels,
        allocated_mip_texels,
    })
}

fn pack_mip_level(
    descriptors: &[Option<TextureAssetDescriptor>],
    mip_level: u32,
    page_side: u32,
    maximum_layers: u32,
) -> Result<PackedMipLevel, PortableTextureAtlasPlanError> {
    let mut occupied = descriptors
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(slot, descriptor)| {
            let descriptor = descriptor?;
            (mip_level < texture_mip_level_count(descriptor.width, descriptor.height)).then(|| {
                (
                    slot,
                    [
                        (descriptor.width >> mip_level).max(1),
                        (descriptor.height >> mip_level).max(1),
                    ],
                )
            })
        })
        .collect::<Vec<_>>();
    occupied.sort_unstable_by(|left, right| {
        right.1[1]
            .cmp(&left.1[1])
            .then_with(|| right.1[0].cmp(&left.1[0]))
            .then_with(|| left.0.cmp(&right.0))
    });
    let mut pages = Vec::<AtlasPage>::new();
    let mut placements = vec![None; descriptors.len()];
    let mut source_texels = 0u64;
    for (slot, extent) in occupied {
        source_texels = source_texels
            .checked_add(u64::from(extent[0]) * u64::from(extent[1]))
            .ok_or(PortableTextureAtlasPlanError::TexelCountOverflow)?;
        let mut placement = None;
        for (layer, page) in pages.iter_mut().enumerate() {
            if let Some(origin) = page.place(extent[0], extent[1], page_side) {
                placement = Some((layer as u32, origin));
                break;
            }
        }
        if placement.is_none() {
            let next_layer = u32::try_from(pages.len())
                .map_err(|_| PortableTextureAtlasPlanError::LayerCountOverflow)?;
            if next_layer >= maximum_layers {
                return Err(PortableTextureAtlasPlanError::LayerLimit {
                    required: next_layer.saturating_add(1),
                    maximum: maximum_layers,
                });
            }
            let mut page = AtlasPage::default();
            let origin = page
                .place(extent[0], extent[1], page_side)
                .expect("validated mip rectangle fits an empty atlas page");
            pages.push(page);
            placement = Some((next_layer, origin));
        }
        let (layer, origin) = placement.expect("placement or new page was created");
        placements[slot] = Some(PortableTextureAtlasMipPlacement {
            layer,
            origin,
            extent,
        });
    }
    Ok(PackedMipLevel {
        placements,
        layer_count: u32::try_from(pages.len())
            .map_err(|_| PortableTextureAtlasPlanError::LayerCountOverflow)?,
        source_texels,
    })
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
        assert_eq!(plan.extent, [8, 8]);
        assert_eq!(plan.layer_count, 3);
        assert_eq!(plan.mip_level_count, 4);
        assert_eq!(plan.occupied_len(), 4);
        assert_eq!(plan.source_texels, 128);
        assert_eq!(plan.allocated_texels, 192);
        assert_eq!(plan.utilization_millionths(), 666_666);
        assert_eq!(plan.source_mip_texels, 170);
        assert_eq!(plan.allocated_mip_texels, 255);
        assert_eq!(plan.mip_utilization_millionths(), 666_666);
        assert_eq!(plan.placements[1], None);
        assert_eq!(plan.placements[0].unwrap().origin, [0, 0]);
        assert_eq!(plan.placements[2].unwrap().origin, [0, 0]);
        assert_eq!(plan.placements[2].unwrap().layer, 1);
        assert_eq!(plan.placements[3].unwrap().origin, [4, 0]);
        assert_eq!(plan.placements[3].unwrap().layer, 1);
        assert_eq!(plan.placements[4].unwrap().origin, [0, 0]);
        assert_eq!(plan.placements[4].unwrap().layer, 2);

        for (mip_level, placements) in plan.mip_placements.iter().enumerate() {
            let page_side = (plan.extent[0] >> mip_level).max(1);
            for (left_index, left) in placements.iter().enumerate() {
                let Some(left) = left else { continue };
                assert!(left.layer < plan.layer_count);
                assert!(left.origin[0] + left.extent[0] <= page_side);
                assert!(left.origin[1] + left.extent[1] <= page_side);
                for right in placements.iter().skip(left_index + 1).flatten() {
                    if left.layer != right.layer {
                        continue;
                    }
                    let separated = left.origin[0] + left.extent[0] <= right.origin[0]
                        || right.origin[0] + right.extent[0] <= left.origin[0]
                        || left.origin[1] + left.extent[1] <= right.origin[1]
                        || right.origin[1] + right.extent[1] <= left.origin[1];
                    assert!(
                        separated,
                        "atlas mip {mip_level} placements overlap: {left:?} and {right:?}"
                    );
                }
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
        assert_eq!(empty.mip_level_count, 1);
        assert_eq!(empty.placements, vec![None, None]);
        assert_eq!(empty.mip_placements, vec![vec![None, None]]);

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

    #[test]
    fn non_power_of_two_mips_pack_independently_without_alignment_padding() {
        let plan = PortableTextureAtlasPlan::build(
            &[Some(descriptor(7, 1)), Some(descriptor(1, 7))],
            PortableTextureAtlasLimits {
                maximum_dimension: 8,
                maximum_layers: 2,
            },
        )
        .unwrap();

        assert_eq!(plan.extent, [8, 8]);
        assert_eq!(plan.layer_count, 1);
        assert_eq!(plan.mip_level_count, 4);
        assert_eq!(plan.mip_placements[0][0].unwrap().extent, [7, 1]);
        assert_eq!(plan.mip_placements[0][1].unwrap().origin, [0, 0]);
        assert_eq!(plan.mip_placements[1][0].unwrap().extent, [3, 1]);
        assert_eq!(plan.mip_placements[1][1].unwrap().extent, [1, 3]);
        assert_eq!(plan.mip_placements[1][1].unwrap().origin, [0, 0]);
        assert_eq!(plan.mip_placements[2][0].unwrap().extent, [1, 1]);
        assert_eq!(plan.mip_placements[2][1].unwrap().extent, [1, 1]);
        assert_eq!(plan.mip_placements[3], vec![None, None]);
        assert_eq!(plan.source_texels, 14);
        assert_eq!(plan.allocated_texels, 64);
        assert_eq!(plan.source_mip_texels, 22);
        assert_eq!(plan.allocated_mip_texels, 85);
        assert_eq!(texture_mip_level_count(7, 1), 3);
        assert_eq!(texture_mip_level_count(1, 7), 3);
    }

    #[test]
    fn large_square_images_choose_layers_instead_of_a_wasteful_larger_page() {
        let descriptors = vec![Some(descriptor(4096, 4096)); 9];
        let plan = PortableTextureAtlasPlan::build(
            &descriptors,
            PortableTextureAtlasLimits {
                maximum_dimension: 8192,
                maximum_layers: 256,
            },
        )
        .unwrap();

        assert_eq!(plan.extent, [4096, 4096]);
        assert_eq!(plan.layer_count, 9);
        assert_eq!(plan.mip_level_count, 13);
        assert_eq!(plan.source_texels, 150_994_944);
        assert_eq!(plan.allocated_texels, plan.source_texels);
        assert_eq!(plan.source_mip_texels, 201_326_589);
        assert_eq!(plan.allocated_mip_texels, plan.source_mip_texels);
        assert_eq!(plan.mip_utilization_millionths(), 1_000_000);
    }
}
