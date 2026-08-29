//! Retained device resources for authored PBR images.
//!
//! Texture bytes and browser image handles are heavyweight asset payloads,
//! not semantic scene state. This table therefore lives beside the WebGPU
//! device epoch and is addressed by the stable indices in `PbrMaterial`.

use crate::{LodClassifierDevice, LodWebGpuError};
use futures_channel::oneshot;
use quilting_core::material::{Rgba8TextureAsset, TextureAssetDescriptor, TextureWrapMode};

pub(crate) struct PbrTextureResource {
    pub(crate) texture: wgpu::Texture,
    pub(crate) linear_view: wgpu::TextureView,
    pub(crate) srgb_view: wgpu::TextureView,
    pub(crate) sampler: wgpu::Sampler,
}

/// One retained, ordered table of decoded glTF texture resources. Every image
/// has both linear and sRGB views over the same allocation so a material can
/// interpret color and data channels correctly without duplicating pixels.
pub struct PbrTextureTable {
    descriptors: Vec<Option<TextureAssetDescriptor>>,
    resources: Vec<Option<PbrTextureResource>>,
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

/// Outcome of an attempted allocation-preserving texture publication.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PbrTextureTableUpdate {
    Updated,
    ShapeChanged,
}

impl LodClassifierDevice {
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
        for (resource, asset) in resources.iter().zip(assets) {
            if let (Some(resource), Some(asset)) = (resource.as_ref(), *asset) {
                write_texture_asset(&self.queue, resource, asset)?;
            }
        }
        Ok(PbrTextureTable {
            descriptors,
            resources,
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
        Ok(PbrTextureTable {
            descriptors,
            resources,
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
        for (resource, asset) in retained.resources.iter().flatten().zip(assets) {
            write_texture_asset(&self.queue, resource, *asset)?;
        }
        Ok(PbrTextureTableUpdate::Updated)
    }

    /// Read one base mip only for conformance evidence. Warm rendering keeps
    /// texture data device-resident and never calls this method.
    pub async fn read_pbr_texture_rgba8_for_diagnostics(
        &self,
        table: &PbrTextureTable,
        index: u32,
    ) -> Result<Vec<u8>, LodWebGpuError> {
        let resource = table.resource(index).ok_or_else(|| {
            LodWebGpuError::Payload(format!("PBR texture index {index} is out of range"))
        })?;
        let descriptor = table.descriptors[index as usize].ok_or_else(|| {
            LodWebGpuError::Payload(format!("PBR texture index {index} is an empty slot"))
        })?;
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
            label: Some("quilting PBR texture evidence readback"),
            size: byte_len,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("quilting PBR texture evidence copy"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &resource.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
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

fn create_texture_resource(
    device: &wgpu::Device,
    descriptor: TextureAssetDescriptor,
) -> PbrTextureResource {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("quilting PBR texture asset"),
        size: texture_extent(descriptor),
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::TEXTURE_BINDING,
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

fn address_mode(mode: TextureWrapMode) -> wgpu::AddressMode {
    match mode {
        TextureWrapMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
        TextureWrapMode::MirroredRepeat => wgpu::AddressMode::MirrorRepeat,
        TextureWrapMode::Repeat => wgpu::AddressMode::Repeat,
    }
}
