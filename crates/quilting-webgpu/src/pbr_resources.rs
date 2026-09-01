//! Retained device resources for authored PBR images.
//!
//! Texture bytes and browser image handles are heavyweight asset payloads,
//! not semantic scene state. This table therefore lives beside the WebGPU
//! device epoch and is addressed by the stable indices in `PbrMaterial`.

use crate::{
    buffer_init_or_zero, texture_mip_level_count, LodClassifierDevice, LodWebGpuError,
    PatchRenderPipeline, PortableTextureAtlasLimits, PortableTextureAtlasPlan,
};
use futures_channel::oneshot;
use quilting_core::material::{
    PbrMaterial, PbrTextureReferences, Rgba8TextureAsset, TextureAssetDescriptor, TextureWrapMode,
};

pub(crate) const PBR_BASE_COLOR_TEXTURE_BIT: u32 = 1 << 0;
pub(crate) const PBR_METALLIC_ROUGHNESS_TEXTURE_BIT: u32 = 1 << 1;
pub(crate) const PBR_NORMAL_TEXTURE_BIT: u32 = 1 << 2;
pub(crate) const PBR_EMISSIVE_TEXTURE_BIT: u32 = 1 << 3;
pub(crate) const PBR_OCCLUSION_TEXTURE_BIT: u32 = 1 << 4;
pub(crate) const PBR_TRANSMISSION_TEXTURE_BIT: u32 = 1 << 5;
pub(crate) const PBR_TEXTURE_CHANNELS: usize = 6;
const PORTABLE_PBR_ATLAS_USES_MANUAL_BILINEAR_FILTERING: bool = true;
const PORTABLE_PBR_ATLAS_USES_MANUAL_TRILINEAR_FILTERING: bool = true;

pub(crate) struct PbrTextureResource {
    pub(crate) texture: wgpu::Texture,
    pub(crate) linear_view: wgpu::TextureView,
    pub(crate) srgb_view: wgpu::TextureView,
    pub(crate) sampler: wgpu::Sampler,
    mip_level_count: u32,
}

/// One retained, ordered table of decoded glTF texture resources. Every image
/// has both linear and sRGB views over the same allocation so a material can
/// interpret color and data channels correctly without duplicating pixels.
pub struct PbrTextureTable {
    descriptors: Vec<Option<TextureAssetDescriptor>>,
    resources: Vec<Option<PbrTextureResource>>,
    portable_atlas: PortablePbrTextureAtlas,
}

/// Baseline-WebGPU representation for non-uniform material texture access.
/// Individual resources remain resident during cutover so the established
/// material-batched path and the portable atlas can be compared exactly.
pub struct PortablePbrTextureAtlas {
    plan: PortableTextureAtlasPlan,
    mip_level_count: u32,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    descriptor_records: wgpu::Buffer,
    mip_placement_records: wgpu::Buffer,
}

/// Allocation and filtering facts for one retained PBR texture table. These
/// values deliberately describe the resources that were actually created,
/// rather than inferring capabilities from authored image descriptors.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PbrTextureTableDiagnostics {
    pub texture_slots: usize,
    pub occupied_images: usize,
    pub individual_min_mip_level_count: u32,
    pub individual_max_mip_level_count: u32,
    pub individual_allocated_mip_texels: u64,
    pub portable_atlas_extent: [u32; 2],
    pub portable_atlas_layers: u32,
    pub portable_atlas_source_texels: u64,
    pub portable_atlas_allocated_texels: u64,
    pub portable_atlas_utilization_millionths: u32,
    pub portable_atlas_source_mip_texels: u64,
    pub portable_atlas_allocated_mip_texels: u64,
    pub portable_atlas_mip_utilization_millionths: u32,
    pub portable_atlas_mip_level_count: u32,
    pub portable_atlas_uses_manual_bilinear_filtering: bool,
    pub portable_atlas_uses_manual_trilinear_filtering: bool,
    pub total_allocated_mip_texels: u64,
    pub total_allocated_bytes: u64,
}

pub(crate) struct PbrPortableAtlasBindings {
    pub(crate) bind_group: wgpu::BindGroup,
    residency: Vec<PbrMaterialTextureResidency>,
    _fallback_atlas: Option<PortablePbrTextureAtlas>,
    _material_texture_records: wgpu::Buffer,
}

impl PbrPortableAtlasBindings {
    pub(crate) fn residency(&self) -> &[PbrMaterialTextureResidency] {
        &self.residency
    }
}

/// Per-material evidence separating authored references from resources that
/// were actually decoded and retained. Referenced-but-nonresident channels
/// bind semantic placeholders without shifting any glTF texture index.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PbrMaterialTextureResidency {
    referenced_mask: u32,
    resident_mask: u32,
}

impl PbrMaterialTextureResidency {
    pub fn referenced_mask(self) -> u32 {
        self.referenced_mask
    }

    pub fn resident_mask(self) -> u32 {
        self.resident_mask
    }

    pub fn unresolved_mask(self) -> u32 {
        self.referenced_mask & !self.resident_mask
    }
}

struct PbrPlaceholderResources {
    white: PbrTextureResource,
    black: PbrTextureResource,
    flat_normal: PbrTextureResource,
}

/// One bind group per stable material-table slot. WebGPU's baseline limits do
/// not promise dynamically indexed sampled-texture arrays, while extracted
/// render batches are already grouped by material, so selecting one small bind
/// group per indirect draw is both portable and congruent with the scene ABI.
pub struct PbrMaterialTextureBindings {
    material_count: u32,
    residency: Vec<PbrMaterialTextureResidency>,
    bind_groups: Vec<wgpu::BindGroup>,
    _placeholders: PbrPlaceholderResources,
}

impl PbrMaterialTextureBindings {
    pub fn material_count(&self) -> u32 {
        self.material_count
    }

    pub fn residency(&self) -> &[PbrMaterialTextureResidency] {
        &self.residency
    }

    pub(crate) fn bind_group(&self, material_slot: u32) -> Option<&wgpu::BindGroup> {
        usize::try_from(material_slot)
            .ok()
            .and_then(|slot| self.bind_groups.get(slot))
    }
}

impl PbrTextureTable {
    pub fn is_empty(&self) -> bool {
        self.resources.is_empty()
    }

    pub fn len(&self) -> usize {
        self.resources.len()
    }

    pub fn occupied_len(&self) -> usize {
        self.resources.iter().flatten().count()
    }

    pub fn descriptors(&self) -> &[Option<TextureAssetDescriptor>] {
        &self.descriptors
    }

    pub fn portable_atlas_plan(&self) -> &PortableTextureAtlasPlan {
        &self.portable_atlas.plan
    }

    pub fn diagnostics(&self) -> PbrTextureTableDiagnostics {
        let mip_level_counts = self
            .resources
            .iter()
            .flatten()
            .map(|resource| resource.mip_level_count);
        let individual_min_mip_level_count = mip_level_counts.clone().min().unwrap_or(0);
        let individual_max_mip_level_count = mip_level_counts.max().unwrap_or(0);
        pbr_texture_table_diagnostics(
            self.len(),
            self.occupied_len(),
            individual_min_mip_level_count,
            individual_max_mip_level_count,
            &self.portable_atlas.plan,
            self.portable_atlas.mip_level_count,
        )
    }

    fn resource(&self, index: u32) -> Option<&PbrTextureResource> {
        usize::try_from(index)
            .ok()
            .and_then(|index| self.resources.get(index))
            .and_then(Option::as_ref)
    }

    pub fn linear_view(&self, index: u32) -> Option<&wgpu::TextureView> {
        self.resource(index).map(|resource| &resource.linear_view)
    }

    pub fn srgb_view(&self, index: u32) -> Option<&wgpu::TextureView> {
        self.resource(index).map(|resource| &resource.srgb_view)
    }

    pub fn sampler(&self, index: u32) -> Option<&wgpu::Sampler> {
        self.resource(index).map(|resource| &resource.sampler)
    }
}

fn pbr_texture_table_diagnostics(
    texture_slots: usize,
    occupied_images: usize,
    individual_min_mip_level_count: u32,
    individual_max_mip_level_count: u32,
    plan: &PortableTextureAtlasPlan,
    portable_atlas_mip_level_count: u32,
) -> PbrTextureTableDiagnostics {
    PbrTextureTableDiagnostics {
        texture_slots,
        occupied_images,
        individual_min_mip_level_count,
        individual_max_mip_level_count,
        individual_allocated_mip_texels: plan.source_mip_texels,
        portable_atlas_extent: plan.extent,
        portable_atlas_layers: plan.layer_count,
        portable_atlas_source_texels: plan.source_texels,
        portable_atlas_allocated_texels: plan.allocated_texels,
        portable_atlas_utilization_millionths: plan.utilization_millionths(),
        portable_atlas_source_mip_texels: plan.source_mip_texels,
        portable_atlas_allocated_mip_texels: plan.allocated_mip_texels,
        portable_atlas_mip_utilization_millionths: plan.mip_utilization_millionths(),
        portable_atlas_mip_level_count,
        portable_atlas_uses_manual_bilinear_filtering:
            PORTABLE_PBR_ATLAS_USES_MANUAL_BILINEAR_FILTERING,
        portable_atlas_uses_manual_trilinear_filtering:
            PORTABLE_PBR_ATLAS_USES_MANUAL_TRILINEAR_FILTERING,
        total_allocated_mip_texels: plan
            .source_mip_texels
            .saturating_add(plan.allocated_mip_texels),
        total_allocated_bytes: plan
            .source_mip_texels
            .saturating_add(plan.allocated_mip_texels)
            .saturating_mul(4),
    }
}

/// Outcome of an attempted allocation-preserving texture publication.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PbrTextureTableUpdate {
    Updated,
    ShapeChanged,
}

impl LodClassifierDevice {
    pub(crate) fn create_pbr_portable_atlas_bindings_for_layout(
        &self,
        layout: &wgpu::BindGroupLayout,
        materials: &[PbrMaterial],
        textures: Option<&PbrTextureTable>,
    ) -> Result<PbrPortableAtlasBindings, LodWebGpuError> {
        let fallback_atlas = textures
            .is_none()
            .then(|| create_portable_pbr_texture_atlas(&self.device, &[]))
            .transpose()?;
        let atlas = textures
            .map(|textures| &textures.portable_atlas)
            .or(fallback_atlas.as_ref())
            .expect("resident texture table or fallback atlas exists");
        let default_material = PbrMaterial::default();
        let material_count = materials.len().max(1);
        let mut residency = Vec::with_capacity(material_count);
        let mut material_records = Vec::with_capacity(material_count);
        for material_slot in 0..material_count {
            let material = materials.get(material_slot).unwrap_or(&default_material);
            material
                .validate()
                .map_err(|error| LodWebGpuError::Payload(error.to_string()))?;
            residency.push(portable_material_residency(material, textures));
            let references = material.textures;
            material_records.push([
                references.base_color.unwrap_or(u32::MAX),
                references.metallic_roughness.unwrap_or(u32::MAX),
                references.normal.unwrap_or(u32::MAX),
                references.emissive.unwrap_or(u32::MAX),
                references.occlusion.unwrap_or(u32::MAX),
                references.transmission.unwrap_or(u32::MAX),
                u32::MAX,
                u32::MAX,
            ]);
        }
        let material_texture_records = buffer_init_or_zero(
            &self.device,
            "quilting portable PBR material texture records",
            bytemuck::cast_slice(&material_records),
            wgpu::BufferUsages::STORAGE,
        );
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("quilting portable PBR atlas bindings"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&atlas.view),
                },
                crate::bind(1, &atlas.descriptor_records),
                crate::bind(2, &material_texture_records),
                crate::bind(3, &atlas.mip_placement_records),
            ],
        });
        Ok(PbrPortableAtlasBindings {
            bind_group,
            residency,
            _fallback_atlas: fallback_atlas,
            _material_texture_records: material_texture_records,
        })
    }

    /// Resolve every stable material slot to one portable sampled-texture bind
    /// group. Missing authored channels and unavailable decoded slots receive
    /// channel-correct placeholders; residency diagnostics retain the
    /// distinction so asset mismatches cannot become invisible.
    pub fn create_pbr_material_texture_bindings(
        &self,
        pipeline: &PatchRenderPipeline,
        materials: &[PbrMaterial],
        textures: Option<&PbrTextureTable>,
    ) -> Result<PbrMaterialTextureBindings, LodWebGpuError> {
        if pipeline.style() != Some(quilting_core::render::RenderStyle::Pbr) {
            return Err(LodWebGpuError::Payload(
                "PBR texture bindings require the PBR render pipeline".to_string(),
            ));
        }
        let layout = pipeline
            .pbr_texture_bind_group_layout
            .as_ref()
            .ok_or_else(|| {
                LodWebGpuError::Payload(
                    "PBR texture bindings require the PBR render pipeline".to_string(),
                )
            })?;
        self.create_pbr_material_texture_bindings_for_layout(layout, materials, textures)
    }

    pub(crate) fn create_pbr_material_texture_bindings_for_layout(
        &self,
        layout: &wgpu::BindGroupLayout,
        materials: &[PbrMaterial],
        textures: Option<&PbrTextureTable>,
    ) -> Result<PbrMaterialTextureBindings, LodWebGpuError> {
        let material_count = materials.len().max(1);
        let material_count_u32 = u32::try_from(material_count).map_err(|_| {
            LodWebGpuError::Payload("PBR material texture binding count exceeds u32".to_string())
        })?;
        for material in materials {
            material
                .validate()
                .map_err(|error| LodWebGpuError::Payload(error.to_string()))?;
        }

        let placeholders = create_placeholder_resources(&self.device, &self.queue)?;
        let default_material = PbrMaterial::default();
        let mut residency = Vec::with_capacity(material_count);
        let mut bind_groups = Vec::with_capacity(material_count);
        for material_slot in 0..material_count {
            let material = materials.get(material_slot).unwrap_or(&default_material);
            let channels = [
                resolve_material_texture(
                    textures,
                    material.textures.base_color,
                    TextureInterpretation::LegacyPow22,
                    &placeholders.white,
                ),
                resolve_material_texture(
                    textures,
                    material.textures.metallic_roughness,
                    TextureInterpretation::Linear,
                    &placeholders.white,
                ),
                resolve_material_texture(
                    textures,
                    material.textures.normal,
                    TextureInterpretation::Linear,
                    &placeholders.flat_normal,
                ),
                resolve_material_texture(
                    textures,
                    material.textures.emissive,
                    TextureInterpretation::LegacyPow22,
                    &placeholders.black,
                ),
                resolve_material_texture(
                    textures,
                    material.textures.occlusion,
                    TextureInterpretation::Linear,
                    &placeholders.white,
                ),
                resolve_material_texture(
                    textures,
                    material.textures.transmission,
                    TextureInterpretation::Linear,
                    &placeholders.white,
                ),
            ];
            let resident_mask = channels
                .iter()
                .enumerate()
                .fold(0u32, |mask, (channel, resolved)| {
                    mask | (u32::from(resolved.resident) << channel)
                });
            residency.push(PbrMaterialTextureResidency {
                referenced_mask: pbr_texture_reference_mask(material.textures),
                resident_mask,
            });

            let mut entries = Vec::with_capacity(PBR_TEXTURE_CHANNELS * 2);
            for (channel, resolved) in channels.iter().enumerate() {
                let texture_binding = u32::try_from(channel * 2).expect("six channels fit u32");
                entries.push(wgpu::BindGroupEntry {
                    binding: texture_binding,
                    resource: wgpu::BindingResource::TextureView(resolved.view),
                });
                entries.push(wgpu::BindGroupEntry {
                    binding: texture_binding + 1,
                    resource: wgpu::BindingResource::Sampler(resolved.sampler),
                });
            }
            bind_groups.push(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("quilting PBR material textures"),
                layout,
                entries: &entries,
            }));
        }
        Ok(PbrMaterialTextureBindings {
            material_count: material_count_u32,
            residency,
            bind_groups,
            _placeholders: placeholders,
        })
    }

    /// Construct a complete replacement table before returning it to the
    /// caller. All descriptors and payload sizes are validated before the
    /// first resource is allocated or queue write is issued.
    pub fn upload_pbr_texture_table(
        &self,
        assets: &[Rgba8TextureAsset<'_>],
    ) -> Result<PbrTextureTable, LodWebGpuError> {
        let slots = assets.iter().copied().map(Some).collect::<Vec<_>>();
        self.upload_pbr_texture_slot_table(&slots)
    }

    /// Upload a texture-index-preserving table where an unavailable decoded
    /// image remains an explicit empty slot rather than shifting later glTF
    /// indices. Material binding resolves those slots through placeholders.
    pub fn upload_pbr_texture_slot_table(
        &self,
        assets: &[Option<Rgba8TextureAsset<'_>>],
    ) -> Result<PbrTextureTable, LodWebGpuError> {
        validate_texture_asset_slots(&self.device, assets)?;
        let descriptors = assets
            .iter()
            .map(|asset| asset.map(|asset| asset.descriptor))
            .collect::<Vec<_>>();
        let resources = descriptors
            .iter()
            .map(|descriptor| {
                descriptor.map(|descriptor| create_texture_resource(&self.device, descriptor))
            })
            .collect::<Vec<_>>();
        let portable_atlas = create_portable_pbr_texture_atlas(&self.device, &descriptors)?;
        for (resource, asset) in resources.iter().zip(assets) {
            if let (Some(resource), Some(asset)) = (resource.as_ref(), *asset) {
                write_texture_asset(&self.queue, resource, asset)?;
            }
        }
        publish_pbr_texture_mips_and_atlas(&self.device, &self.queue, &resources, &portable_atlas)?;
        Ok(PbrTextureTable {
            descriptors,
            resources,
            portable_atlas,
        })
    }

    /// Upload browser-decoded images directly into WebGPU texture storage.
    /// The external-image copy captures each bitmap synchronously, so the web
    /// adapter may close its handles after this method returns.
    #[cfg(target_arch = "wasm32")]
    pub fn upload_pbr_image_bitmap_table(
        &self,
        assets: &[Option<(TextureAssetDescriptor, web_sys::ImageBitmap)>],
    ) -> Result<PbrTextureTable, LodWebGpuError> {
        validate_image_bitmap_slots(&self.device, assets)?;
        let descriptors = assets
            .iter()
            .map(|asset| asset.as_ref().map(|(descriptor, _)| *descriptor))
            .collect::<Vec<_>>();
        let resources = descriptors
            .iter()
            .map(|descriptor| {
                descriptor.map(|descriptor| create_texture_resource(&self.device, descriptor))
            })
            .collect::<Vec<_>>();
        let portable_atlas = create_portable_pbr_texture_atlas(&self.device, &descriptors)?;
        for (resource, asset) in resources.iter().zip(assets) {
            if let (Some(resource), Some((descriptor, bitmap))) = (resource.as_ref(), asset) {
                self.queue.copy_external_image_to_texture(
                    &wgpu::CopyExternalImageSourceInfo {
                        source: wgpu::ExternalImageSource::ImageBitmap(bitmap.clone()),
                        origin: wgpu::Origin2d::ZERO,
                        flip_y: false,
                    },
                    wgpu::CopyExternalImageDestInfo {
                        texture: &resource.texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                        color_space: wgpu::PredefinedColorSpace::Srgb,
                        premultiplied_alpha: false,
                    },
                    texture_extent(*descriptor),
                );
            }
        }
        publish_pbr_texture_mips_and_atlas(&self.device, &self.queue, &resources, &portable_atlas)?;
        Ok(PbrTextureTable {
            descriptors,
            resources,
            portable_atlas,
        })
    }

    /// Replace pixels in existing allocations only when the complete ordered
    /// descriptor table is unchanged. Validation and shape comparison happen
    /// before any queue write, so malformed or replacement-shaped candidates
    /// cannot partially update the incumbent table.
    pub fn update_pbr_texture_table_in_place(
        &self,
        retained: &mut PbrTextureTable,
        assets: &[Rgba8TextureAsset<'_>],
    ) -> Result<PbrTextureTableUpdate, LodWebGpuError> {
        validate_texture_assets(&self.device, assets)?;
        if !assets
            .iter()
            .map(|asset| Some(asset.descriptor))
            .eq(retained.descriptors.iter().copied())
        {
            return Ok(PbrTextureTableUpdate::ShapeChanged);
        }
        for (resource, asset) in retained.resources.iter().zip(assets) {
            let resource = resource.as_ref().ok_or_else(|| {
                LodWebGpuError::Payload(
                    "dense PBR texture update encountered an empty retained slot".to_string(),
                )
            })?;
            write_texture_asset(&self.queue, resource, *asset)?;
        }
        publish_pbr_texture_mips_and_atlas(
            &self.device,
            &self.queue,
            &retained.resources,
            &retained.portable_atlas,
        )?;
        Ok(PbrTextureTableUpdate::Updated)
    }

    /// Read one base mip only for conformance evidence. Warm rendering keeps
    /// texture data device-resident and never calls this method.
    pub async fn read_pbr_texture_rgba8_for_diagnostics(
        &self,
        table: &PbrTextureTable,
        index: u32,
    ) -> Result<Vec<u8>, LodWebGpuError> {
        self.read_pbr_texture_mip_rgba8_for_diagnostics(table, index, 0)
            .await
    }

    /// Read one generated source-image mip for explicit conformance evidence.
    /// Ordinary rendering never maps texture data back to the CPU.
    pub async fn read_pbr_texture_mip_rgba8_for_diagnostics(
        &self,
        table: &PbrTextureTable,
        index: u32,
        mip_level: u32,
    ) -> Result<Vec<u8>, LodWebGpuError> {
        let resource = table.resource(index).ok_or_else(|| {
            LodWebGpuError::Payload(format!("PBR texture index {index} is out of range"))
        })?;
        let descriptor = table.descriptors[index as usize].ok_or_else(|| {
            LodWebGpuError::Payload(format!("PBR texture index {index} is an empty slot"))
        })?;
        let descriptor = texture_mip_descriptor(descriptor, mip_level)?;
        self.read_rgba8_texture_region(
            &resource.texture,
            mip_level,
            wgpu::Origin3d::ZERO,
            descriptor,
            "quilting PBR texture evidence",
        )
        .await
    }

    /// Read the same stable texture slot from the portable array atlas. This
    /// is conformance-only evidence that packing and in-place publication did
    /// not change image bytes or slot identity.
    pub async fn read_pbr_portable_atlas_rgba8_for_diagnostics(
        &self,
        table: &PbrTextureTable,
        index: u32,
    ) -> Result<Vec<u8>, LodWebGpuError> {
        self.read_pbr_portable_atlas_mip_rgba8_for_diagnostics(table, index, 0)
            .await
    }

    /// Read one independently packed portable-atlas mip for conformance-only
    /// comparison with its source-image mip.
    pub async fn read_pbr_portable_atlas_mip_rgba8_for_diagnostics(
        &self,
        table: &PbrTextureTable,
        index: u32,
        mip_level: u32,
    ) -> Result<Vec<u8>, LodWebGpuError> {
        let descriptor = table
            .descriptors
            .get(index as usize)
            .copied()
            .flatten()
            .ok_or_else(|| {
                LodWebGpuError::Payload(format!(
                    "portable PBR texture index {index} is empty or out of range",
                ))
            })?;
        let descriptor = texture_mip_descriptor(descriptor, mip_level)?;
        let placement = table
            .portable_atlas
            .plan
            .mip_placements
            .get(mip_level as usize)
            .and_then(|placements| placements.get(index as usize))
            .copied()
            .flatten()
            .ok_or_else(|| {
                LodWebGpuError::Payload(format!(
                    "portable PBR texture index {index} mip {mip_level} has no placement",
                ))
            })?;
        self.read_rgba8_texture_region(
            &table.portable_atlas.texture,
            mip_level,
            wgpu::Origin3d {
                x: placement.origin[0],
                y: placement.origin[1],
                z: placement.layer,
            },
            descriptor,
            "quilting portable PBR atlas evidence",
        )
        .await
    }

    async fn read_rgba8_texture_region(
        &self,
        texture: &wgpu::Texture,
        mip_level: u32,
        origin: wgpu::Origin3d,
        descriptor: TextureAssetDescriptor,
        label: &'static str,
    ) -> Result<Vec<u8>, LodWebGpuError> {
        let unpadded_bytes_per_row = descriptor.width.checked_mul(4).ok_or_else(|| {
            LodWebGpuError::Payload("PBR texture row byte length overflowed u32".to_string())
        })?;
        let bytes_per_row = unpadded_bytes_per_row
            .div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            .checked_mul(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            .ok_or_else(|| {
                LodWebGpuError::Payload("PBR texture padded row size overflowed".to_string())
            })?;
        let byte_len = u64::from(bytes_per_row)
            .checked_mul(u64::from(descriptor.height))
            .ok_or_else(|| {
                LodWebGpuError::Payload("PBR texture readback size overflowed".to_string())
            })?;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: byte_len,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level,
                origin,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(descriptor.height),
                },
            },
            texture_extent(descriptor),
        );
        self.queue.submit([encoder.finish()]);

        let slice = buffer.slice(..byte_len);
        let (sender, receiver) = oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        #[cfg(not(target_arch = "wasm32"))]
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| LodWebGpuError::Poll(error.to_string()))?;
        receiver
            .await
            .map_err(|_| LodWebGpuError::Mapping("map callback was canceled".to_string()))?
            .map_err(|error| LodWebGpuError::Mapping(error.to_string()))?;
        let mapped = slice.get_mapped_range();
        let output_len = descriptor
            .rgba8_byte_len()
            .map_err(|error| LodWebGpuError::Payload(error.to_string()))?;
        let mut output = Vec::with_capacity(output_len);
        for row in mapped.chunks_exact(bytes_per_row as usize) {
            output.extend_from_slice(&row[..unpadded_bytes_per_row as usize]);
        }
        drop(mapped);
        buffer.unmap();
        Ok(output)
    }
}

pub(crate) fn create_pbr_texture_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let mut entries = Vec::with_capacity(PBR_TEXTURE_CHANNELS * 2);
    for channel in 0..PBR_TEXTURE_CHANNELS {
        let texture_binding = u32::try_from(channel * 2).expect("six channels fit u32");
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: texture_binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        });
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: texture_binding + 1,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        });
    }
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("quilting PBR material texture bindings"),
        entries: &entries,
    })
}

pub(crate) fn create_pbr_portable_atlas_bind_group_layout(
    device: &wgpu::Device,
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("quilting portable PBR atlas binding layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZeroU64::new(32),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZeroU64::new(32),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZeroU64::new(16),
                },
                count: None,
            },
        ],
    })
}

#[derive(Clone, Copy)]
enum TextureInterpretation {
    Linear,
    /// Bind raw UNORM so shared WGSL can reproduce the incumbent's historical
    /// `pow(2.2)` transfer on both APIs.
    LegacyPow22,
}

struct ResolvedMaterialTexture<'a> {
    view: &'a wgpu::TextureView,
    sampler: &'a wgpu::Sampler,
    resident: bool,
}

fn resolve_material_texture<'a>(
    table: Option<&'a PbrTextureTable>,
    index: Option<u32>,
    interpretation: TextureInterpretation,
    placeholder: &'a PbrTextureResource,
) -> ResolvedMaterialTexture<'a> {
    let resident_resource = index.and_then(|index| table.and_then(|table| table.resource(index)));
    let resident = resident_resource.is_some();
    let resource = resident_resource.unwrap_or(placeholder);
    let view = match interpretation {
        TextureInterpretation::Linear | TextureInterpretation::LegacyPow22 => &resource.linear_view,
    };
    ResolvedMaterialTexture {
        view,
        sampler: &resource.sampler,
        resident,
    }
}

pub(crate) fn pbr_texture_reference_mask(textures: PbrTextureReferences) -> u32 {
    [
        (textures.base_color, PBR_BASE_COLOR_TEXTURE_BIT),
        (
            textures.metallic_roughness,
            PBR_METALLIC_ROUGHNESS_TEXTURE_BIT,
        ),
        (textures.normal, PBR_NORMAL_TEXTURE_BIT),
        (textures.emissive, PBR_EMISSIVE_TEXTURE_BIT),
        (textures.occlusion, PBR_OCCLUSION_TEXTURE_BIT),
        (textures.transmission, PBR_TRANSMISSION_TEXTURE_BIT),
    ]
    .into_iter()
    .fold(0, |mask, (index, bit)| mask | index.map_or(0, |_| bit))
}

fn portable_material_residency(
    material: &PbrMaterial,
    textures: Option<&PbrTextureTable>,
) -> PbrMaterialTextureResidency {
    let references = [
        (material.textures.base_color, PBR_BASE_COLOR_TEXTURE_BIT),
        (
            material.textures.metallic_roughness,
            PBR_METALLIC_ROUGHNESS_TEXTURE_BIT,
        ),
        (material.textures.normal, PBR_NORMAL_TEXTURE_BIT),
        (material.textures.emissive, PBR_EMISSIVE_TEXTURE_BIT),
        (material.textures.occlusion, PBR_OCCLUSION_TEXTURE_BIT),
        (material.textures.transmission, PBR_TRANSMISSION_TEXTURE_BIT),
    ];
    let resident_mask = references.iter().fold(0, |mask, &(slot, bit)| {
        let resident =
            slot.is_some_and(|slot| textures.is_some_and(|table| table.resource(slot).is_some()));
        mask | if resident { bit } else { 0 }
    });
    PbrMaterialTextureResidency {
        referenced_mask: pbr_texture_reference_mask(material.textures),
        resident_mask,
    }
}

fn create_placeholder_resources(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<PbrPlaceholderResources, LodWebGpuError> {
    Ok(PbrPlaceholderResources {
        white: create_placeholder_resource(device, queue, [255, 255, 255, 255])?,
        black: create_placeholder_resource(device, queue, [0, 0, 0, 255])?,
        flat_normal: create_placeholder_resource(device, queue, [128, 128, 255, 255])?,
    })
}

fn create_placeholder_resource(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    rgba: [u8; 4],
) -> Result<PbrTextureResource, LodWebGpuError> {
    let descriptor = TextureAssetDescriptor {
        width: 1,
        height: 1,
        wrap_s: TextureWrapMode::ClampToEdge,
        wrap_t: TextureWrapMode::ClampToEdge,
    };
    let resource = create_texture_resource(device, descriptor);
    let asset = Rgba8TextureAsset::new(descriptor, &rgba)
        .map_err(|error| LodWebGpuError::Payload(error.to_string()))?;
    write_texture_asset(queue, &resource, asset)?;
    Ok(resource)
}

fn validate_texture_assets(
    device: &wgpu::Device,
    assets: &[Rgba8TextureAsset<'_>],
) -> Result<(), LodWebGpuError> {
    let slots = assets.iter().copied().map(Some).collect::<Vec<_>>();
    validate_texture_asset_slots(device, &slots)
}

fn validate_texture_asset_slots(
    device: &wgpu::Device,
    assets: &[Option<Rgba8TextureAsset<'_>>],
) -> Result<(), LodWebGpuError> {
    let maximum_dimension = device.limits().max_texture_dimension_2d;
    for (index, asset) in assets.iter().enumerate() {
        let Some(asset) = asset else {
            continue;
        };
        Rgba8TextureAsset::new(asset.descriptor, asset.pixels)
            .map_err(|error| LodWebGpuError::Payload(format!("PBR texture {index}: {error}")))?;
        if asset.descriptor.width > maximum_dimension || asset.descriptor.height > maximum_dimension
        {
            return Err(LodWebGpuError::Payload(format!(
                "PBR texture {index} is {}x{}, exceeding device limit {maximum_dimension}",
                asset.descriptor.width, asset.descriptor.height,
            )));
        }
    }
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn validate_image_bitmap_slots(
    device: &wgpu::Device,
    assets: &[Option<(TextureAssetDescriptor, web_sys::ImageBitmap)>],
) -> Result<(), LodWebGpuError> {
    let maximum_dimension = device.limits().max_texture_dimension_2d;
    for (index, asset) in assets.iter().enumerate() {
        let Some((descriptor, bitmap)) = asset else {
            continue;
        };
        descriptor
            .validate()
            .map_err(|error| LodWebGpuError::Payload(format!("PBR texture {index}: {error}")))?;
        if bitmap.width() != descriptor.width || bitmap.height() != descriptor.height {
            return Err(LodWebGpuError::Payload(format!(
                "PBR texture {index} descriptor is {}x{}, but its ImageBitmap is {}x{}",
                descriptor.width,
                descriptor.height,
                bitmap.width(),
                bitmap.height(),
            )));
        }
        if descriptor.width > maximum_dimension || descriptor.height > maximum_dimension {
            return Err(LodWebGpuError::Payload(format!(
                "PBR texture {index} is {}x{}, exceeding device limit {maximum_dimension}",
                descriptor.width, descriptor.height,
            )));
        }
    }
    Ok(())
}

fn create_portable_pbr_texture_atlas(
    device: &wgpu::Device,
    descriptors: &[Option<TextureAssetDescriptor>],
) -> Result<PortablePbrTextureAtlas, LodWebGpuError> {
    let limits = device.limits();
    let plan = PortableTextureAtlasPlan::build(
        descriptors,
        PortableTextureAtlasLimits {
            maximum_dimension: limits.max_texture_dimension_2d,
            maximum_layers: limits.max_texture_array_layers,
        },
    )
    .map_err(|error| LodWebGpuError::Payload(error.to_string()))?;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("quilting portable PBR texture atlas"),
        size: wgpu::Extent3d {
            width: plan.extent[0],
            height: plan.extent[1],
            depth_or_array_layers: plan.layer_count,
        },
        mip_level_count: plan.mip_level_count,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("quilting portable PBR texture atlas array view"),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        base_array_layer: 0,
        array_layer_count: Some(plan.layer_count),
        ..Default::default()
    });
    let mut descriptor_records = vec![[0u32; 8]; descriptors.len().max(1)];
    let placement_record_count = descriptors
        .len()
        .checked_mul(plan.mip_level_count as usize)
        .ok_or_else(|| {
            LodWebGpuError::Payload("portable PBR mip placement count overflowed usize".into())
        })?;
    let mut mip_placement_records = vec![[0u32; 4]; placement_record_count.max(1)];
    for (slot, (record, descriptor)) in descriptor_records.iter_mut().zip(descriptors).enumerate() {
        let Some(descriptor) = descriptor else {
            continue;
        };
        let mip_level_count = texture_mip_level_count(descriptor.width, descriptor.height);
        let placement_offset = slot
            .checked_mul(plan.mip_level_count as usize)
            .and_then(|offset| u32::try_from(offset).ok())
            .ok_or_else(|| {
                LodWebGpuError::Payload("portable PBR mip placement offset exceeded u32".into())
            })?;
        *record = [
            descriptor.width,
            descriptor.height,
            portable_wrap_mode(descriptor.wrap_s),
            portable_wrap_mode(descriptor.wrap_t),
            mip_level_count,
            placement_offset,
            1,
            0,
        ];
        for mip_level in 0..mip_level_count {
            let placement = plan.mip_placements[mip_level as usize][slot].ok_or_else(|| {
                LodWebGpuError::Payload(format!(
                    "portable PBR texture slot {slot} mip {mip_level} has no placement",
                ))
            })?;
            mip_placement_records[placement_offset as usize + mip_level as usize] =
                [placement.origin[0], placement.origin[1], placement.layer, 1];
        }
    }
    let descriptor_records = buffer_init_or_zero(
        device,
        "quilting portable PBR texture descriptor records",
        bytemuck::cast_slice(&descriptor_records),
        wgpu::BufferUsages::STORAGE,
    );
    let mip_placement_records = buffer_init_or_zero(
        device,
        "quilting portable PBR mip placement records",
        bytemuck::cast_slice(&mip_placement_records),
        wgpu::BufferUsages::STORAGE,
    );
    let mip_level_count = plan.mip_level_count;
    Ok(PortablePbrTextureAtlas {
        plan,
        mip_level_count,
        texture,
        view,
        descriptor_records,
        mip_placement_records,
    })
}

const PBR_MIPMAP_SHADER: &str = r#"
@group(0) @binding(0) var pbr_mip_source: texture_2d<f32>;

@vertex
fn pbr_mip_vertex(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(positions[vertex_index], 0.0, 1.0);
}

@fragment
fn pbr_mip_fragment(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let maximum = vec2<i32>(textureDimensions(pbr_mip_source)) - vec2<i32>(1);
    let source = vec2<i32>(position.xy) * 2;
    let a = textureLoad(pbr_mip_source, min(source, maximum), 0);
    let b = textureLoad(pbr_mip_source, min(source + vec2<i32>(1, 0), maximum), 0);
    let c = textureLoad(pbr_mip_source, min(source + vec2<i32>(0, 1), maximum), 0);
    let d = textureLoad(pbr_mip_source, min(source + vec2<i32>(1, 1), maximum), 0);
    return (a + b + c + d) * 0.25;
}
"#;

struct PbrMipmapGenerator {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
}

impl PbrMipmapGenerator {
    fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("quilting PBR mip source layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("quilting PBR mip pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("quilting PBR mip shader"),
            source: wgpu::ShaderSource::Wgsl(PBR_MIPMAP_SHADER.into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("quilting PBR mip pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("pbr_mip_vertex"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("pbr_mip_fragment"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        Self {
            bind_group_layout,
            pipeline,
        }
    }

    fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        resource: &PbrTextureResource,
    ) {
        for mip_level in 1..resource.mip_level_count {
            let source_view = resource.texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("quilting PBR mip source view"),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_mip_level: mip_level - 1,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: Some(1),
                ..Default::default()
            });
            let destination_view = resource.texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("quilting PBR mip destination view"),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_mip_level: mip_level,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: Some(1),
                ..Default::default()
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("quilting PBR mip source binding"),
                layout: &self.bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&source_view),
                }],
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("quilting PBR mip pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &destination_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}

fn publish_pbr_texture_mips_and_atlas(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    resources: &[Option<PbrTextureResource>],
    atlas: &PortablePbrTextureAtlas,
) -> Result<(), LodWebGpuError> {
    if resources.iter().all(Option::is_none) {
        return Ok(());
    }
    let generator = resources
        .iter()
        .flatten()
        .any(|resource| resource.mip_level_count > 1)
        .then(|| PbrMipmapGenerator::new(device));
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("quilting PBR mip and portable atlas publication"),
    });
    for (slot, resource) in resources.iter().enumerate() {
        let Some(resource) = resource else { continue };
        if let Some(generator) = generator.as_ref() {
            generator.encode(device, &mut encoder, resource);
        }
        for mip_level in 0..resource.mip_level_count {
            let placement =
                atlas.plan.mip_placements[mip_level as usize][slot].ok_or_else(|| {
                    LodWebGpuError::Payload(format!(
                    "portable PBR texture slot {slot} mip {mip_level} has no publication target",
                ))
                })?;
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &resource.texture,
                    mip_level,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &atlas.texture,
                    mip_level,
                    origin: wgpu::Origin3d {
                        x: placement.origin[0],
                        y: placement.origin[1],
                        z: placement.layer,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: placement.extent[0],
                    height: placement.extent[1],
                    depth_or_array_layers: 1,
                },
            );
        }
    }
    queue.submit(Some(encoder.finish()));
    Ok(())
}

fn portable_wrap_mode(mode: TextureWrapMode) -> u32 {
    match mode {
        TextureWrapMode::ClampToEdge => 0,
        TextureWrapMode::MirroredRepeat => 1,
        TextureWrapMode::Repeat => 2,
    }
}

fn create_texture_resource(
    device: &wgpu::Device,
    descriptor: TextureAssetDescriptor,
) -> PbrTextureResource {
    let mip_level_count = texture_mip_level_count(descriptor.width, descriptor.height);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("quilting PBR texture asset"),
        size: texture_extent(descriptor),
        mip_level_count,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[wgpu::TextureFormat::Rgba8UnormSrgb],
    });
    let linear_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let srgb_view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("quilting PBR texture asset sRGB view"),
        format: Some(wgpu::TextureFormat::Rgba8UnormSrgb),
        ..Default::default()
    });
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("quilting PBR texture sampler"),
        address_mode_u: address_mode(descriptor.wrap_s),
        address_mode_v: address_mode(descriptor.wrap_t),
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        ..Default::default()
    });
    PbrTextureResource {
        texture,
        linear_view,
        srgb_view,
        sampler,
        mip_level_count,
    }
}

fn write_texture_asset(
    queue: &wgpu::Queue,
    resource: &PbrTextureResource,
    asset: Rgba8TextureAsset<'_>,
) -> Result<(), LodWebGpuError> {
    let bytes_per_row = asset.descriptor.width.checked_mul(4).ok_or_else(|| {
        LodWebGpuError::Payload("PBR texture row byte length overflowed u32".to_string())
    })?;
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &resource.texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        asset.pixels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(bytes_per_row),
            rows_per_image: Some(asset.descriptor.height),
        },
        texture_extent(asset.descriptor),
    );
    Ok(())
}

fn texture_extent(descriptor: TextureAssetDescriptor) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width: descriptor.width,
        height: descriptor.height,
        depth_or_array_layers: 1,
    }
}

fn texture_mip_descriptor(
    descriptor: TextureAssetDescriptor,
    mip_level: u32,
) -> Result<TextureAssetDescriptor, LodWebGpuError> {
    let mip_level_count = texture_mip_level_count(descriptor.width, descriptor.height);
    if mip_level >= mip_level_count {
        return Err(LodWebGpuError::Payload(format!(
            "PBR texture mip {mip_level} is outside its {mip_level_count}-level chain",
        )));
    }
    Ok(TextureAssetDescriptor {
        width: (descriptor.width >> mip_level).max(1),
        height: (descriptor.height >> mip_level).max(1),
        ..descriptor
    })
}

fn address_mode(mode: TextureWrapMode) -> wgpu::AddressMode {
    match mode {
        TextureWrapMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
        TextureWrapMode::MirroredRepeat => wgpu::AddressMode::MirrorRepeat,
        TextureWrapMode::Repeat => wgpu::AddressMode::Repeat,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostics_report_the_created_complete_mip_atlas_contract() {
        let descriptors = [
            Some(TextureAssetDescriptor {
                width: 8,
                height: 4,
                wrap_s: TextureWrapMode::Repeat,
                wrap_t: TextureWrapMode::ClampToEdge,
            }),
            None,
        ];
        let plan = PortableTextureAtlasPlan::build(
            &descriptors,
            PortableTextureAtlasLimits {
                maximum_dimension: 16,
                maximum_layers: 2,
            },
        )
        .unwrap();
        let diagnostics = pbr_texture_table_diagnostics(2, 1, 4, 4, &plan, 4);

        assert_eq!(diagnostics.texture_slots, 2);
        assert_eq!(diagnostics.occupied_images, 1);
        assert_eq!(diagnostics.individual_min_mip_level_count, 4);
        assert_eq!(diagnostics.individual_max_mip_level_count, 4);
        assert_eq!(diagnostics.individual_allocated_mip_texels, 43);
        assert_eq!(diagnostics.portable_atlas_extent, [8, 8]);
        assert_eq!(diagnostics.portable_atlas_layers, 1);
        assert_eq!(diagnostics.portable_atlas_source_texels, 32);
        assert_eq!(diagnostics.portable_atlas_allocated_texels, 64);
        assert_eq!(diagnostics.portable_atlas_utilization_millionths, 500_000);
        assert_eq!(diagnostics.portable_atlas_source_mip_texels, 43);
        assert_eq!(diagnostics.portable_atlas_allocated_mip_texels, 85);
        assert_eq!(
            diagnostics.portable_atlas_mip_utilization_millionths,
            505_882
        );
        assert_eq!(diagnostics.portable_atlas_mip_level_count, 4);
        assert!(diagnostics.portable_atlas_uses_manual_bilinear_filtering);
        assert!(diagnostics.portable_atlas_uses_manual_trilinear_filtering);
        assert_eq!(diagnostics.total_allocated_mip_texels, 128);
        assert_eq!(diagnostics.total_allocated_bytes, 512);
        quilting_shaders::compile_shader(PBR_MIPMAP_SHADER, Default::default())
            .expect("PBR mipmap shader compiles");
    }
}
