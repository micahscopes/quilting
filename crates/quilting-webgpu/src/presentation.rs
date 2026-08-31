#[cfg(all(target_arch = "wasm32", feature = "browser-surface"))]
use crate::WebGpuAdapterSummary;
use crate::{LodClassifierDevice, LodWebGpuError, PatchRenderTarget};

const PRESENTATION_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth24Plus;

/// A nonfatal reason why no browser surface frame was presented.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PresentationSkipReason {
    ZeroSized,
    Timeout,
    Occluded,
}

/// Result of one attempt to acquire, encode, submit, and present a surface
/// frame. Surface loss is explicit because recreating it requires the browser
/// adapter to provide the canvas again; it is not silently treated as an
/// ordinary resize.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SurfacePresentation<T> {
    Presented(T),
    Skipped(PresentationSkipReason),
    RecreateRequired,
}

/// Bounded surface lifecycle facts suitable for application diagnostics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SurfacePresentationDiagnostics {
    pub size: [u32; 2],
    pub color_format: String,
    pub present_mode: String,
    pub alpha_mode: String,
    pub configured: bool,
    pub reconfigurations: u64,
    pub frames_presented: u64,
    pub frames_skipped: u64,
    pub surface_losses: u64,
}

/// Rust-owned WebGPU presentation attachments. Semantic scene and browser UI
/// state remain outside this object; a caller supplies an encoder closure that
/// receives the same `PatchRenderTarget` used by offscreen and native paths.
pub struct PatchPresentationSurface {
    surface: wgpu::Surface<'static>,
    configuration: wgpu::SurfaceConfiguration,
    depth: Option<wgpu::Texture>,
    depth_view: Option<wgpu::TextureView>,
    configured: bool,
    reconfigurations: u64,
    frames_presented: u64,
    frames_skipped: u64,
    surface_losses: u64,
}

impl PatchPresentationSurface {
    /// Configure a presentation surface created by the application on the
    /// same adapter/device pair used by the classifier executor.
    pub fn from_surface(
        surface: wgpu::Surface<'static>,
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        size: [u32; 2],
    ) -> Result<Self, LodWebGpuError> {
        let capabilities = surface.get_capabilities(adapter);
        let color_format = preferred_color_format(&capabilities.formats).ok_or_else(|| {
            LodWebGpuError::Payload(
                "WebGPU presentation surface has no supported color format".to_string(),
            )
        })?;
        let present_mode =
            preferred_present_mode(&capabilities.present_modes).ok_or_else(|| {
                LodWebGpuError::Payload(
                    "WebGPU presentation surface has no supported present mode".to_string(),
                )
            })?;
        let alpha_mode = preferred_alpha_mode(&capabilities.alpha_modes).ok_or_else(|| {
            LodWebGpuError::Payload(
                "WebGPU presentation surface has no supported alpha mode".to_string(),
            )
        })?;
        let mut target = Self {
            surface,
            configuration: wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: color_format,
                width: 1,
                height: 1,
                present_mode,
                desired_maximum_frame_latency: 2,
                alpha_mode,
                view_formats: Vec::new(),
            },
            depth: None,
            depth_view: None,
            configured: false,
            reconfigurations: 0,
            frames_presented: 0,
            frames_skipped: 0,
            surface_losses: 0,
        };
        target.resize(device, size)?;
        Ok(target)
    }

    pub fn color_format(&self) -> wgpu::TextureFormat {
        self.configuration.format
    }

    pub fn depth_format(&self) -> wgpu::TextureFormat {
        PRESENTATION_DEPTH_FORMAT
    }

    pub fn size(&self) -> [u32; 2] {
        if self.configured {
            [self.configuration.width, self.configuration.height]
        } else {
            [0, 0]
        }
    }

    pub fn diagnostics(&self) -> SurfacePresentationDiagnostics {
        SurfacePresentationDiagnostics {
            size: self.size(),
            color_format: format!("{:?}", self.configuration.format),
            present_mode: format!("{:?}", self.configuration.present_mode),
            alpha_mode: format!("{:?}", self.configuration.alpha_mode),
            configured: self.configured,
            reconfigurations: self.reconfigurations,
            frames_presented: self.frames_presented,
            frames_skipped: self.frames_skipped,
            surface_losses: self.surface_losses,
        }
    }

    /// Resize or suspend the surface. A zero-sized browser canvas is inert
    /// until a subsequent nonzero resize; `wgpu::Surface::configure` is never
    /// called with invalid dimensions.
    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        size: [u32; 2],
    ) -> Result<bool, LodWebGpuError> {
        if size[0] == 0 || size[1] == 0 {
            let changed = self.configured;
            self.configured = false;
            self.depth = None;
            self.depth_view = None;
            return Ok(changed);
        }
        let limit = device.limits().max_texture_dimension_2d;
        if size[0] > limit || size[1] > limit {
            return Err(LodWebGpuError::Payload(format!(
                "WebGPU presentation surface {}x{} exceeds device limit {limit}",
                size[0], size[1],
            )));
        }
        if self.configured
            && self.configuration.width == size[0]
            && self.configuration.height == size[1]
        {
            return Ok(false);
        }
        self.configuration.width = size[0];
        self.configuration.height = size[1];
        self.reconfigure(device);
        Ok(true)
    }

    /// Acquire and present exactly one frame. `Outdated` is reconfigured and
    /// retried once before reporting an error. Timeout/occlusion are ordinary
    /// skipped frames. `Lost` asks the browser adapter to recreate the surface.
    pub fn present_with<T>(
        &mut self,
        classifier: &LodClassifierDevice,
        label: &'static str,
        encode: impl FnOnce(
            &mut wgpu::CommandEncoder,
            PatchRenderTarget<'_>,
        ) -> Result<T, LodWebGpuError>,
    ) -> Result<SurfacePresentation<T>, LodWebGpuError> {
        if !self.configured {
            self.frames_skipped = self.frames_skipped.saturating_add(1);
            return Ok(SurfacePresentation::Skipped(
                PresentationSkipReason::ZeroSized,
            ));
        }
        let mut retried_outdated = false;
        let (surface_texture, suboptimal) = loop {
            match self.surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(texture) => break (texture, false),
                wgpu::CurrentSurfaceTexture::Suboptimal(texture) => break (texture, true),
                wgpu::CurrentSurfaceTexture::Timeout => {
                    self.frames_skipped = self.frames_skipped.saturating_add(1);
                    return Ok(SurfacePresentation::Skipped(
                        PresentationSkipReason::Timeout,
                    ));
                }
                wgpu::CurrentSurfaceTexture::Occluded => {
                    self.frames_skipped = self.frames_skipped.saturating_add(1);
                    return Ok(SurfacePresentation::Skipped(
                        PresentationSkipReason::Occluded,
                    ));
                }
                wgpu::CurrentSurfaceTexture::Outdated if !retried_outdated => {
                    self.reconfigure(classifier.device());
                    retried_outdated = true;
                }
                wgpu::CurrentSurfaceTexture::Outdated => {
                    return Err(LodWebGpuError::Payload(
                        "WebGPU presentation surface remained outdated after reconfiguration"
                            .to_string(),
                    ));
                }
                wgpu::CurrentSurfaceTexture::Lost => {
                    self.surface_losses = self.surface_losses.saturating_add(1);
                    return Ok(SurfacePresentation::RecreateRequired);
                }
                wgpu::CurrentSurfaceTexture::Validation => {
                    return Err(LodWebGpuError::Payload(
                        "WebGPU presentation surface acquisition failed validation".to_string(),
                    ));
                }
            }
        };
        let color_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let depth_view = self.depth_view.as_ref().ok_or_else(|| {
            LodWebGpuError::Payload(
                "configured WebGPU presentation surface has no depth attachment".to_string(),
            )
        })?;
        let mut encoder = classifier
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        let value = encode(
            &mut encoder,
            PatchRenderTarget {
                color_view: &color_view,
                resolve_target: None,
                depth_stencil_view: Some(depth_view),
                clear_color: None,
                clear_depth: None,
            },
        )?;
        classifier.queue().submit([encoder.finish()]);
        surface_texture.present();
        self.frames_presented = self.frames_presented.saturating_add(1);
        if suboptimal {
            self.reconfigure(classifier.device());
        }
        Ok(SurfacePresentation::Presented(value))
    }

    fn reconfigure(&mut self, device: &wgpu::Device) {
        self.surface.configure(device, &self.configuration);
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("quilting presentation depth"),
            size: wgpu::Extent3d {
                width: self.configuration.width,
                height: self.configuration.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: PRESENTATION_DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        self.depth_view = Some(depth.create_view(&wgpu::TextureViewDescriptor::default()));
        self.depth = Some(depth);
        self.configured = true;
        self.reconfigurations = self.reconfigurations.saturating_add(1);
    }
}

#[cfg(all(target_arch = "wasm32", feature = "browser-surface"))]
impl LodClassifierDevice {
    /// Request one surface-compatible browser device and claim a canvas that
    /// has not previously created a WebGL/WebGPU context.
    pub async fn request_canvas_presentation(
        canvas: web_sys::HtmlCanvasElement,
        size: [u32; 2],
        label: &str,
    ) -> Result<(Self, WebGpuAdapterSummary, PatchPresentationSurface), LodWebGpuError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
            .map_err(|error| {
                LodWebGpuError::Payload(format!(
                    "WebGPU browser presentation surface creation failed: {error}"
                ))
            })?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .map_err(|error| {
                LodWebGpuError::Payload(format!(
                    "WebGPU presentation adapter request failed: {error}"
                ))
            })?;
        let info = adapter.get_info();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some(label),
                ..Default::default()
            })
            .await
            .map_err(|error| {
                LodWebGpuError::Payload(format!(
                    "WebGPU presentation device request failed: {error}"
                ))
            })?;
        let classifier = Self::new(device, queue)?;
        let presentation =
            PatchPresentationSurface::from_surface(surface, &adapter, classifier.device(), size)?;
        let summary = WebGpuAdapterSummary {
            name: info.name,
            backend: format!("{:?}", info.backend),
            device_type: format!("{:?}", info.device_type),
        };
        Ok((classifier, summary, presentation))
    }
}

fn preferred_color_format(formats: &[wgpu::TextureFormat]) -> Option<wgpu::TextureFormat> {
    formats.first().copied()
}

fn preferred_present_mode(modes: &[wgpu::PresentMode]) -> Option<wgpu::PresentMode> {
    modes
        .contains(&wgpu::PresentMode::Fifo)
        .then_some(wgpu::PresentMode::Fifo)
        .or_else(|| modes.first().copied())
}

fn preferred_alpha_mode(modes: &[wgpu::CompositeAlphaMode]) -> Option<wgpu::CompositeAlphaMode> {
    modes
        .contains(&wgpu::CompositeAlphaMode::Opaque)
        .then_some(wgpu::CompositeAlphaMode::Opaque)
        .or_else(|| modes.first().copied())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presentation_preferences_are_deterministic_and_supported() {
        assert_eq!(
            preferred_color_format(&[
                wgpu::TextureFormat::Rgba16Float,
                wgpu::TextureFormat::Rgba8Unorm,
                wgpu::TextureFormat::Bgra8Unorm,
            ]),
            Some(wgpu::TextureFormat::Rgba16Float),
        );
        assert_eq!(
            preferred_color_format(&[wgpu::TextureFormat::Rgba16Float]),
            Some(wgpu::TextureFormat::Rgba16Float),
        );
        assert_eq!(
            preferred_present_mode(&[wgpu::PresentMode::Immediate, wgpu::PresentMode::Fifo,]),
            Some(wgpu::PresentMode::Fifo),
        );
        assert_eq!(
            preferred_alpha_mode(&[
                wgpu::CompositeAlphaMode::PreMultiplied,
                wgpu::CompositeAlphaMode::Opaque,
            ]),
            Some(wgpu::CompositeAlphaMode::Opaque),
        );
        assert_eq!(
            preferred_alpha_mode(&[wgpu::CompositeAlphaMode::PostMultiplied]),
            Some(wgpu::CompositeAlphaMode::PostMultiplied),
        );
        assert_eq!(preferred_color_format(&[]), None);
        assert_eq!(preferred_present_mode(&[]), None);
        assert_eq!(preferred_alpha_mode(&[]), None);
    }
}
