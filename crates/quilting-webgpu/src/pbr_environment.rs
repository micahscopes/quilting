//! Retained WebGPU image-based-lighting resources.
//!
//! The browser/native asset boundary supplies validated RGBA32F cube payloads.
//! WebGPU stores them as baseline-filterable RGBA16F, converting one face at a
//! time so upload does not duplicate the complete environment in host memory.

use crate::{buffer_init_or_zero, LodClassifierDevice, LodWebGpuError, PatchRenderPipeline};
use futures_channel::oneshot;
use quilting_core::material::{EnvironmentMapAsset, EnvironmentMapDescriptor};
use quilting_core::render::RenderStyle;

const PBR_ENVIRONMENT_UNIFORM_BYTES: u64 = 16;

/// One coherent prefiltered-specular plus diffuse-irradiance device epoch.
pub struct PbrEnvironmentMap {
    descriptor: EnvironmentMapDescriptor,
    prefiltered: wgpu::Texture,
    prefiltered_view: wgpu::TextureView,
    irradiance: wgpu::Texture,
    irradiance_view: wgpu::TextureView,
    sampler: wgpu::Sampler,
}

/// One complete PBR environment bind group. A nonresident binding contains a
/// valid black cube so the pipeline layout remains portable while the shader
/// deliberately follows its analytical fallback path.
pub struct PbrEnvironmentBindings {
    resident: bool,
    descriptor: Option<EnvironmentMapDescriptor>,
    bind_group: wgpu::BindGroup,
    _uniform: wgpu::Buffer,
    _placeholder: Option<wgpu::Texture>,
}

impl PbrEnvironmentBindings {
    pub fn is_resident(&self) -> bool {
        self.resident
    }

    pub fn descriptor(&self) -> Option<EnvironmentMapDescriptor> {
        self.descriptor
    }

    pub(crate) fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }
}

impl PbrEnvironmentMap {
    pub fn descriptor(&self) -> EnvironmentMapDescriptor {
        self.descriptor
    }

    pub fn prefiltered_mip_count(&self) -> u32 {
        self.descriptor.prefiltered_mip_count
    }

    pub fn prefiltered_view(&self) -> &wgpu::TextureView {
        &self.prefiltered_view
    }

    pub fn irradiance_view(&self) -> &wgpu::TextureView {
        &self.irradiance_view
    }

    pub fn sampler(&self) -> &wgpu::Sampler {
        &self.sampler
    }
}

impl LodClassifierDevice {
    /// Resolve one environment epoch to the fixed PBR group-two ABI. Missing
    /// IBL remains an explicit nonresident state rather than an absent group.
    pub fn create_pbr_environment_bindings(
        &self,
        pipeline: &PatchRenderPipeline,
        environment: Option<&PbrEnvironmentMap>,
    ) -> Result<PbrEnvironmentBindings, LodWebGpuError> {
        if pipeline.style() != Some(RenderStyle::Pbr) {
            return Err(LodWebGpuError::Payload(
                "PBR environment bindings require the PBR render pipeline".to_string(),
            ));
        }
        let layout = pipeline
            .pbr_environment_bind_group_layout
            .as_ref()
            .ok_or_else(|| {
                LodWebGpuError::Payload(
                    "PBR environment layout is not available on this pipeline".to_string(),
                )
            })?;
        self.create_pbr_environment_bindings_for_layout(layout, environment)
    }

    pub(crate) fn create_pbr_environment_bindings_for_layout(
        &self,
        layout: &wgpu::BindGroupLayout,
        environment: Option<&PbrEnvironmentMap>,
    ) -> Result<PbrEnvironmentBindings, LodWebGpuError> {
        let resident = environment.is_some();
        let descriptor = environment.map(PbrEnvironmentMap::descriptor);
        let placeholder = if environment.is_none() {
            let placeholder =
                create_cube_texture(&self.device, "quilting PBR environment placeholder", 1, 1);
            let zero_face = [0.0f32; 4];
            for face in 0..6 {
                write_cube_face_rgba16f(&self.queue, &placeholder, 0, face, 1, &zero_face)?;
            }
            Some(placeholder)
        } else {
            None
        };
        let placeholder_view = placeholder.as_ref().map(|placeholder| {
            cube_view(
                placeholder,
                "quilting PBR environment placeholder cube view",
            )
        });
        let placeholder_sampler = placeholder
            .as_ref()
            .map(|_| create_environment_sampler(&self.device));
        let prefiltered_view = environment
            .map(PbrEnvironmentMap::prefiltered_view)
            .or(placeholder_view.as_ref())
            .expect("resident or placeholder PBR environment view");
        let irradiance_view = environment
            .map(PbrEnvironmentMap::irradiance_view)
            .or(placeholder_view.as_ref())
            .expect("resident or placeholder PBR irradiance view");
        let sampler = environment
            .map(PbrEnvironmentMap::sampler)
            .or(placeholder_sampler.as_ref())
            .expect("resident or placeholder PBR environment sampler");
        let uniform_words = [
            u32::from(resident),
            descriptor.map_or(1, |descriptor| descriptor.prefiltered_mip_count),
            0,
            0,
        ];
        let uniform = buffer_init_or_zero(
            &self.device,
            "quilting PBR environment uniform",
            bytemuck::cast_slice(&uniform_words),
            wgpu::BufferUsages::UNIFORM,
        );
        debug_assert_eq!(
            std::mem::size_of_val(&uniform_words) as u64,
            PBR_ENVIRONMENT_UNIFORM_BYTES,
        );
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("quilting PBR environment bindings"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &uniform,
                        offset: 0,
                        size: std::num::NonZeroU64::new(PBR_ENVIRONMENT_UNIFORM_BYTES),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(prefiltered_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(irradiance_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });
        Ok(PbrEnvironmentBindings {
            resident,
            descriptor,
            bind_group,
            _uniform: uniform,
            _placeholder: placeholder,
        })
    }

    /// Validate and upload one complete environment before returning it for
    /// publication. No resource or queue write occurs until both CPU payloads
    /// and device limits have been checked.
    pub fn upload_pbr_environment_map(
        &self,
        asset: EnvironmentMapAsset<'_>,
    ) -> Result<PbrEnvironmentMap, LodWebGpuError> {
        let asset = EnvironmentMapAsset::new(
            asset.descriptor,
            asset.prefiltered_rgba32f,
            asset.irradiance_rgba32f,
        )
        .map_err(|error| LodWebGpuError::Payload(error.to_string()))?;
        validate_environment_for_device(&self.device, asset)?;

        let prefiltered = create_cube_texture(
            &self.device,
            "quilting PBR prefiltered environment",
            asset.descriptor.prefiltered_face_size,
            asset.descriptor.prefiltered_mip_count,
        );
        let irradiance = create_cube_texture(
            &self.device,
            "quilting PBR irradiance environment",
            asset.descriptor.irradiance_face_size,
            1,
        );

        let mut offset = 0usize;
        for mip in 0..asset.descriptor.prefiltered_mip_count {
            let face_size = (asset.descriptor.prefiltered_face_size >> mip).max(1);
            let face_components = rgba_face_component_len(face_size)?;
            for face in 0..6 {
                let end = offset.checked_add(face_components).ok_or_else(|| {
                    LodWebGpuError::Payload(
                        "prefiltered environment face offset overflowed".to_string(),
                    )
                })?;
                write_cube_face_rgba16f(
                    &self.queue,
                    &prefiltered,
                    mip,
                    face,
                    face_size,
                    &asset.prefiltered_rgba32f[offset..end],
                )?;
                offset = end;
            }
        }
        debug_assert_eq!(offset, asset.prefiltered_rgba32f.len());

        let irradiance_face_components =
            rgba_face_component_len(asset.descriptor.irradiance_face_size)?;
        for face in 0..6 {
            let offset = face as usize * irradiance_face_components;
            let end = offset + irradiance_face_components;
            write_cube_face_rgba16f(
                &self.queue,
                &irradiance,
                0,
                face,
                asset.descriptor.irradiance_face_size,
                &asset.irradiance_rgba32f[offset..end],
            )?;
        }

        let prefiltered_view = cube_view(
            &prefiltered,
            "quilting PBR prefiltered environment cube view",
        );
        let irradiance_view =
            cube_view(&irradiance, "quilting PBR irradiance environment cube view");
        let sampler = create_environment_sampler(&self.device);
        Ok(PbrEnvironmentMap {
            descriptor: asset.descriptor,
            prefiltered,
            prefiltered_view,
            irradiance,
            irradiance_view,
            sampler,
        })
    }

    /// Read one cube face only for upload conformance. Ordinary PBR rendering
    /// samples the retained cube directly and never stages it through the CPU.
    pub async fn read_pbr_environment_face_for_diagnostics(
        &self,
        environment: &PbrEnvironmentMap,
        prefiltered: bool,
        mip: u32,
        face: u32,
    ) -> Result<Vec<f32>, LodWebGpuError> {
        if face >= 6 {
            return Err(LodWebGpuError::Payload(
                "environment diagnostic face must be below six".to_string(),
            ));
        }
        let (texture, base_size, mip_count) = if prefiltered {
            (
                &environment.prefiltered,
                environment.descriptor.prefiltered_face_size,
                environment.descriptor.prefiltered_mip_count,
            )
        } else {
            (
                &environment.irradiance,
                environment.descriptor.irradiance_face_size,
                1,
            )
        };
        if mip >= mip_count {
            return Err(LodWebGpuError::Payload(format!(
                "environment diagnostic mip {mip} exceeds {mip_count} levels",
            )));
        }
        let face_size = (base_size >> mip).max(1);
        let unpadded_bytes_per_row = face_size.checked_mul(8).ok_or_else(|| {
            LodWebGpuError::Payload("environment diagnostic row size overflowed".to_string())
        })?;
        let bytes_per_row = unpadded_bytes_per_row
            .div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            .checked_mul(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            .ok_or_else(|| {
                LodWebGpuError::Payload(
                    "environment diagnostic padded row size overflowed".to_string(),
                )
            })?;
        let byte_len = u64::from(bytes_per_row)
            .checked_mul(u64::from(face_size))
            .ok_or_else(|| {
                LodWebGpuError::Payload("environment diagnostic buffer size overflowed".to_string())
            })?;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("quilting PBR environment evidence readback"),
            size: byte_len,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("quilting PBR environment evidence copy"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: mip,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: face,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(face_size),
                },
            },
            wgpu::Extent3d {
                width: face_size,
                height: face_size,
                depth_or_array_layers: 1,
            },
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
        let mut output = Vec::with_capacity(rgba_face_component_len(face_size)?);
        for row in mapped.chunks_exact(bytes_per_row as usize) {
            for half_bytes in row[..unpadded_bytes_per_row as usize].chunks_exact(2) {
                let bits = u16::from_le_bytes([half_bytes[0], half_bytes[1]]);
                output.push(half::f16::from_bits(bits).to_f32());
            }
        }
        drop(mapped);
        buffer.unmap();
        Ok(output)
    }
}

pub(crate) fn create_pbr_environment_bind_group_layout(
    device: &wgpu::Device,
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("quilting PBR environment binding layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZeroU64::new(PBR_ENVIRONMENT_UNIFORM_BYTES),
                },
                count: None,
            },
            cube_texture_layout(1),
            cube_texture_layout(2),
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

fn cube_texture_layout(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::Cube,
            multisampled: false,
        },
        count: None,
    }
}

fn create_environment_sampler(device: &wgpu::Device) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("quilting PBR environment sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        ..Default::default()
    })
}

fn validate_environment_for_device(
    device: &wgpu::Device,
    asset: EnvironmentMapAsset<'_>,
) -> Result<(), LodWebGpuError> {
    let maximum_dimension = device.limits().max_texture_dimension_2d;
    for (name, size) in [
        ("prefiltered", asset.descriptor.prefiltered_face_size),
        ("irradiance", asset.descriptor.irradiance_face_size),
    ] {
        if size > maximum_dimension {
            return Err(LodWebGpuError::Payload(format!(
                "{name} environment face size {size} exceeds device limit {maximum_dimension}",
            )));
        }
    }
    for (field, values) in [
        ("prefiltered", asset.prefiltered_rgba32f),
        ("irradiance", asset.irradiance_rgba32f),
    ] {
        if let Some((index, value)) = values
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| value.abs() > f32::from(half::f16::MAX))
        {
            return Err(LodWebGpuError::Payload(format!(
                "{field} environment component {index}={value} exceeds RGBA16F range",
            )));
        }
    }
    Ok(())
}

fn create_cube_texture(
    device: &wgpu::Device,
    label: &'static str,
    face_size: u32,
    mip_level_count: u32,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: face_size,
            height: face_size,
            depth_or_array_layers: 6,
        },
        mip_level_count,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}

fn cube_view(texture: &wgpu::Texture, label: &'static str) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some(label),
        dimension: Some(wgpu::TextureViewDimension::Cube),
        base_array_layer: 0,
        array_layer_count: Some(6),
        ..Default::default()
    })
}

fn write_cube_face_rgba16f(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    mip: u32,
    face: u32,
    face_size: u32,
    rgba32f: &[f32],
) -> Result<(), LodWebGpuError> {
    let expected = rgba_face_component_len(face_size)?;
    if rgba32f.len() != expected {
        return Err(LodWebGpuError::Payload(format!(
            "environment face requires {expected} floats, got {}",
            rgba32f.len(),
        )));
    }
    let rgba16f = rgba32f
        .iter()
        .copied()
        .map(|value| half::f16::from_f32(value).to_bits())
        .collect::<Vec<_>>();
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: mip,
            origin: wgpu::Origin3d {
                x: 0,
                y: 0,
                z: face,
            },
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&rgba16f),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(face_size * 8),
            rows_per_image: Some(face_size),
        },
        wgpu::Extent3d {
            width: face_size,
            height: face_size,
            depth_or_array_layers: 1,
        },
    );
    Ok(())
}

fn rgba_face_component_len(face_size: u32) -> Result<usize, LodWebGpuError> {
    u64::from(face_size)
        .checked_mul(u64::from(face_size))
        .and_then(|texels| texels.checked_mul(4))
        .and_then(|components| usize::try_from(components).ok())
        .ok_or_else(|| LodWebGpuError::Payload("environment face size overflowed".to_string()))
}
