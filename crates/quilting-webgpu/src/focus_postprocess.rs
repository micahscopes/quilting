//! Retained WebGPU pipelines for the shared focus-composition schedule.
//!
//! The pipeline family, retained intermediate textures, bind groups, and pass
//! encoder live here. Scene rendering into the focus MRT remains a separate
//! integration boundary so ordinary rendering cannot accidentally claim focus
//! support before the composed output has evidence.

use crate::{LodClassifierDevice, LodWebGpuError};
use futures_channel::oneshot;
use quilting_core::focus_postprocess::{
    FocusBlurSurface, FocusPingPong, FocusPostprocessSchedule, FOCUS_JFA_DOWNSAMPLE,
};
use quilting_core::render::FocusPostprocessPacket;
use quilting_core::render_pipeline as functional;
use std::collections::BTreeMap;
use std::num::NonZeroU64;
use std::sync::Mutex;

const FOCUS_PASS_UNIFORM_BYTES: u64 = 64;
const FOCUS_PASS_CAPACITY: usize = 64;

#[derive(Clone)]
pub struct FocusPostprocessPipelines {
    bind_group_layout: wgpu::BindGroupLayout,
    select_weight: wgpu::RenderPipeline,
    jfa_init: wgpu::RenderPipeline,
    jfa_step: wgpu::RenderPipeline,
    firmness: wgpu::RenderPipeline,
    kawase: wgpu::RenderPipeline,
    directional_blur_intermediate: wgpu::RenderPipeline,
    directional_blur_output: wgpu::RenderPipeline,
    output_format: wgpu::TextureFormat,
}

/// Pure, backend-neutral graphics state for the seven retained focus passes.
/// Applications and FRP code may compare or memoize this value without owning
/// a WebGPU device; concrete handles remain inside `LodClassifierDevice`.
pub fn focus_postprocess_pipeline_descriptors(
    output_format: functional::TextureFormat,
) -> Result<Vec<functional::RenderPipelineDescriptor>, functional::RenderPipelineDescriptorError> {
    let shader = |stage, entry_point| {
        functional::ShaderModuleDescriptor::new(
            "quilting focus postprocess",
            quilting_shaders::sources::FOCUS_POSTPROCESS,
            quilting_shaders::compiler_catalog_revision(),
            stage,
            entry_point,
            functional::ShaderTarget::Wgsl,
            Vec::new(),
        )
    };
    let vertex = shader(
        functional::ShaderStage::Vertex,
        quilting_shaders::FOCUS_POSTPROCESS_VERTEX_ENTRY_POINT,
    )?;
    let layout = functional::PipelineLayoutDescriptor::new(vec![
        functional::BindGroupLayoutDescriptor::new(
            0,
            vec![
                functional::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: functional::ShaderVisibility::FRAGMENT,
                    kind: functional::BindingKind::Texture {
                        sample_kind: functional::TextureSampleKind::FloatFilterable,
                        view_dimension: functional::TextureViewDimension::D2,
                        multisampled: false,
                    },
                },
                functional::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: functional::ShaderVisibility::FRAGMENT,
                    kind: functional::BindingKind::Sampler(
                        functional::SamplerBindingKind::Filtering,
                    ),
                },
                functional::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: functional::ShaderVisibility::FRAGMENT,
                    kind: functional::BindingKind::Texture {
                        sample_kind: functional::TextureSampleKind::FloatFilterable,
                        view_dimension: functional::TextureViewDimension::D2,
                        multisampled: false,
                    },
                },
                functional::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: functional::ShaderVisibility::FRAGMENT,
                    kind: functional::BindingKind::UniformBuffer {
                        dynamic_offset: true,
                        minimum_size: FOCUS_PASS_UNIFORM_BYTES,
                    },
                },
            ],
        )?,
    ])?;
    let intermediate = functional::TextureFormat::Rgba16Float;
    let passes = [
        (
            quilting_shaders::FOCUS_SELECT_WEIGHT_ENTRY_POINT,
            intermediate,
        ),
        (quilting_shaders::FOCUS_JFA_INIT_ENTRY_POINT, intermediate),
        (quilting_shaders::FOCUS_JFA_STEP_ENTRY_POINT, intermediate),
        (quilting_shaders::FOCUS_FIRMNESS_ENTRY_POINT, intermediate),
        (quilting_shaders::FOCUS_KAWASE_ENTRY_POINT, intermediate),
        (
            quilting_shaders::FOCUS_DIRECTIONAL_BLUR_ENTRY_POINT,
            functional::TextureFormat::Rgba8Unorm,
        ),
        (
            quilting_shaders::FOCUS_DIRECTIONAL_BLUR_ENTRY_POINT,
            output_format,
        ),
    ];
    passes
        .into_iter()
        .map(|(fragment_entry_point, format)| {
            functional::RenderPipelineDescriptor::new(
                functional::GraphicsProgramDescriptor::new(
                    vertex.clone(),
                    Some(shader(
                        functional::ShaderStage::Fragment,
                        fragment_entry_point,
                    )?),
                    Vec::new(),
                )?,
                layout.clone(),
                Vec::new(),
                functional::PrimitiveStateDescriptor {
                    topology: functional::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: functional::FrontFace::CounterClockwise,
                    cull_mode: functional::CullMode::None,
                },
                None,
                vec![functional::ColorTargetStateDescriptor {
                    format,
                    blend: None,
                    write_mask: functional::ColorWriteMask::ALL,
                }],
                functional::MultisampleStateDescriptor::default(),
            )
        })
        .collect()
}

pub struct FocusPostprocessTarget {
    size: [u32; 2],
    output_format: wgpu::TextureFormat,
    raw_field: wgpu::Texture,
    raw_field_view: wgpu::TextureView,
    _selected_weight: wgpu::Texture,
    selected_weight_view: wgpu::TextureView,
    _jfa_ping: wgpu::Texture,
    jfa_ping_view: wgpu::TextureView,
    _jfa_pong: wgpu::Texture,
    jfa_pong_view: wgpu::TextureView,
    _firmness: wgpu::Texture,
    firmness_view: wgpu::TextureView,
    _kawase: wgpu::Texture,
    kawase_view: wgpu::TextureView,
    scene_color: wgpu::Texture,
    scene_color_view: wgpu::TextureView,
    _blur_ping: wgpu::Texture,
    blur_ping_view: wgpu::TextureView,
    _blur_pong: wgpu::Texture,
    blur_pong_view: wgpu::TextureView,
    uniform: wgpu::Buffer,
    _sampler: wgpu::Sampler,
    uniform_stride_bytes: u64,
    bind_groups: BTreeMap<(FocusTextureSlot, FocusTextureSlot), wgpu::BindGroup>,
    scratch: Mutex<FocusEncodingScratch>,
}

/// Explicit diagnostic copy of the raw PBR focus MRT. Production focus
/// composition never constructs this staging resource.
pub struct StagedFocusRawFieldReadback {
    #[cfg(not(target_arch = "wasm32"))]
    device: wgpu::Device,
    buffer: wgpu::Buffer,
    size: [u32; 2],
    bytes_per_row: usize,
    byte_len: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FocusRawFieldImage {
    size: [u32; 2],
    texels: Vec<[f32; 4]>,
}

impl FocusRawFieldImage {
    pub const fn size(&self) -> [u32; 2] {
        self.size
    }

    pub fn texels(&self) -> &[[f32; 4]] {
        &self.texels
    }

    pub fn covered_texels(&self) -> usize {
        self.texels.iter().filter(|texel| texel[3] > 0.5).count()
    }

    pub fn covered_channel_range(&self, channel: usize) -> Option<[f32; 2]> {
        if channel >= 4 {
            return None;
        }
        self.texels
            .iter()
            .filter(|texel| texel[3] > 0.5 && texel[channel].is_finite())
            .map(|texel| texel[channel])
            .fold(None, |range, value| {
                Some(range.map_or([value, value], |[minimum, maximum]| {
                    [minimum.min(value), maximum.max(value)]
                }))
            })
    }
}

impl StagedFocusRawFieldReadback {
    pub async fn read(self) -> Result<FocusRawFieldImage, LodWebGpuError> {
        let slice = self.buffer.slice(..self.byte_len);
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
        let image = decode_focus_raw_field(&mapped, self.size, self.bytes_per_row)?;
        drop(mapped);
        self.buffer.unmap();
        Ok(image)
    }
}

impl FocusPostprocessTarget {
    pub const fn scene_color_format(&self) -> wgpu::TextureFormat {
        wgpu::TextureFormat::Rgba8Unorm
    }

    pub const fn raw_field_format(&self) -> wgpu::TextureFormat {
        wgpu::TextureFormat::Rgba16Float
    }

    pub const fn size(&self) -> [u32; 2] {
        self.size
    }

    pub const fn output_format(&self) -> wgpu::TextureFormat {
        self.output_format
    }

    /// Attachment one of the focus-aware PBR MRT pass.
    pub fn raw_field_view(&self) -> &wgpu::TextureView {
        &self.raw_field_view
    }

    /// Attachment zero of the focus-aware PBR MRT pass.
    pub fn scene_color_view(&self) -> &wgpu::TextureView {
        &self.scene_color_view
    }

    pub fn scene_color_texture(&self) -> &wgpu::Texture {
        &self.scene_color
    }

    fn view(&self, slot: FocusTextureSlot) -> &wgpu::TextureView {
        match slot {
            FocusTextureSlot::RawField => &self.raw_field_view,
            FocusTextureSlot::SelectedWeight => &self.selected_weight_view,
            FocusTextureSlot::JfaPing => &self.jfa_ping_view,
            FocusTextureSlot::JfaPong => &self.jfa_pong_view,
            FocusTextureSlot::Firmness => &self.firmness_view,
            FocusTextureSlot::Kawase => &self.kawase_view,
            FocusTextureSlot::Scene => &self.scene_color_view,
            FocusTextureSlot::BlurPing => &self.blur_ping_view,
            FocusTextureSlot::BlurPong => &self.blur_pong_view,
        }
    }

    fn bind_group(
        &self,
        source_a: FocusTextureSlot,
        source_b: FocusTextureSlot,
    ) -> Result<&wgpu::BindGroup, LodWebGpuError> {
        self.bind_groups.get(&(source_a, source_b)).ok_or_else(|| {
            LodWebGpuError::Payload(format!(
                "focus target has no retained binding for {source_a:?}/{source_b:?}",
            ))
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FocusPostprocessEncoding {
    pub render_passes: u32,
    pub primary_jfa_passes: u32,
    pub cleanup_jfa_passes: u32,
    pub kawase_passes: u32,
    pub directional_blur_passes: u32,
    pub uniform_upload_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum FocusTextureSlot {
    RawField,
    SelectedWeight,
    JfaPing,
    JfaPong,
    Firmness,
    Kawase,
    Scene,
    BlurPing,
    BlurPong,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FocusDestination {
    Texture(FocusTextureSlot),
    Output,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FocusPipelineKind {
    SelectWeight,
    JfaInit,
    JfaStep,
    Firmness,
    Kawase,
    DirectionalBlur,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PlannedFocusPass {
    pipeline: FocusPipelineKind,
    source_a: FocusTextureSlot,
    source_b: FocusTextureSlot,
    destination: FocusDestination,
    uniform: [f32; 16],
    uniform_offset: u32,
    final_output: bool,
}

#[derive(Default)]
struct FocusEncodingScratch {
    passes: Vec<PlannedFocusPass>,
    uniform_words: Vec<f32>,
}

impl FocusPostprocessPipelines {
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    pub const fn output_format(&self) -> wgpu::TextureFormat {
        self.output_format
    }

    pub fn select_weight(&self) -> &wgpu::RenderPipeline {
        &self.select_weight
    }

    pub fn jfa_init(&self) -> &wgpu::RenderPipeline {
        &self.jfa_init
    }

    pub fn jfa_step(&self) -> &wgpu::RenderPipeline {
        &self.jfa_step
    }

    pub fn firmness(&self) -> &wgpu::RenderPipeline {
        &self.firmness
    }

    pub fn kawase(&self) -> &wgpu::RenderPipeline {
        &self.kawase
    }

    pub fn directional_blur(&self, final_output: bool) -> &wgpu::RenderPipeline {
        if final_output {
            &self.directional_blur_output
        } else {
            &self.directional_blur_intermediate
        }
    }
}

impl LodClassifierDevice {
    /// Create the fixed-format composer paired with the browser parity target
    /// without exposing backend texture enums to the WASM adapter.
    pub fn create_offscreen_focus_postprocess_pipelines(
        &self,
    ) -> Result<FocusPostprocessPipelines, LodWebGpuError> {
        self.create_focus_postprocess_pipelines(wgpu::TextureFormat::Rgba8Unorm)
    }

    pub fn create_focus_postprocess_pipelines(
        &self,
        output_format: wgpu::TextureFormat,
    ) -> Result<FocusPostprocessPipelines, LodWebGpuError> {
        let Some(functional_format) =
            crate::pipeline_lowering::functional_texture_format(output_format)
        else {
            // Preserve support for backend formats outside Quilting's current
            // portable descriptor subset; those uncommon formats deliberately
            // bypass memoization rather than receiving an incomplete key.
            return self.build_focus_postprocess_pipelines(output_format, None);
        };
        let descriptor = focus_postprocess_pipeline_descriptors(functional_format)
            .map_err(|error| LodWebGpuError::Payload(error.to_string()))?;
        let mut pipelines = self
            .focus_postprocess_render_pipelines
            .lock()
            .map_err(|_| {
                LodWebGpuError::Payload(
                    "WebGPU focus postprocess pipeline memo was poisoned".to_string(),
                )
            })?;
        let pipeline = pipelines.get_or_try_insert_with(descriptor, |descriptor| {
            self.build_focus_postprocess_pipelines(output_format, Some(descriptor.as_slice()))
        })?;
        Ok(pipeline.clone())
    }

    fn build_focus_postprocess_pipelines(
        &self,
        output_format: wgpu::TextureFormat,
        descriptors: Option<&[functional::RenderPipelineDescriptor]>,
    ) -> Result<FocusPostprocessPipelines, LodWebGpuError> {
        let module = self.memoized_render_shader_module(
            "quilting focus postprocess",
            quilting_shaders::sources::FOCUS_POSTPROCESS,
            quilting_shaders::FOCUS_POSTPROCESS_VERTEX_ENTRY_POINT,
            quilting_shaders::compile_focus_postprocess_wgsl,
        )?;
        let functional_layout = if let Some(descriptors) = descriptors {
            if descriptors.len() != 7 {
                return Err(LodWebGpuError::Payload(
                    "functional focus pipeline family must contain seven passes".to_string(),
                ));
            }
            let layout = descriptors[0].layout();
            if descriptors
                .iter()
                .skip(1)
                .any(|descriptor| descriptor.layout() != layout)
                || layout.groups().len() != 1
            {
                return Err(LodWebGpuError::Payload(
                    "functional focus pipeline family does not share one bind-group layout"
                        .to_string(),
                ));
            }
            Some(layout)
        } else {
            None
        };
        let bind_group_layout = if let Some(layout) = functional_layout {
            crate::pipeline_lowering::bind_group_layout(
                &self.device,
                "quilting focus postprocess bindings",
                &layout.groups()[0],
            )
        } else {
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("quilting focus postprocess bindings"),
                    entries: &[
                        sampled_texture_layout(0),
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        sampled_texture_layout(2),
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: true,
                                min_binding_size: NonZeroU64::new(FOCUS_PASS_UNIFORM_BYTES),
                            },
                            count: None,
                        },
                    ],
                })
        };
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("quilting focus postprocess pipeline layout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });
        let create_pipeline = |index: usize,
                               label: &'static str,
                               fragment_entry: &'static str,
                               format: wgpu::TextureFormat|
         -> Result<wgpu::RenderPipeline, LodWebGpuError> {
            if let Some(descriptors) = descriptors {
                let descriptor = &descriptors[index];
                let fragment = descriptor.program().fragment().ok_or_else(|| {
                    LodWebGpuError::Payload(
                        "functional focus pipeline is missing a fragment stage".to_string(),
                    )
                })?;
                if fragment.entry_point() != fragment_entry
                    || !descriptor.vertex_buffers().is_empty()
                    || descriptor.depth_stencil().is_some()
                    || descriptor.color_targets().len() != 1
                {
                    return Err(LodWebGpuError::Payload(
                        "functional focus pipeline uses inconsistent fixed state".to_string(),
                    ));
                }
                return crate::pipeline_lowering::render_pipeline(
                    &self.device,
                    label,
                    &pipeline_layout,
                    &module,
                    descriptor,
                );
            }
            let targets = [Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })];
            Ok(self
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &module,
                        entry_point: Some(quilting_shaders::FOCUS_POSTPROCESS_VERTEX_ENTRY_POINT),
                        compilation_options: Default::default(),
                        buffers: &[],
                    },
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    fragment: Some(wgpu::FragmentState {
                        module: &module,
                        entry_point: Some(fragment_entry),
                        compilation_options: Default::default(),
                        targets: &targets,
                    }),
                    multiview_mask: None,
                    cache: None,
                }))
        };
        let intermediate_format = wgpu::TextureFormat::Rgba16Float;
        Ok(FocusPostprocessPipelines {
            select_weight: create_pipeline(
                0,
                "quilting focus select weight",
                quilting_shaders::FOCUS_SELECT_WEIGHT_ENTRY_POINT,
                intermediate_format,
            )?,
            jfa_init: create_pipeline(
                1,
                "quilting focus JFA init",
                quilting_shaders::FOCUS_JFA_INIT_ENTRY_POINT,
                intermediate_format,
            )?,
            jfa_step: create_pipeline(
                2,
                "quilting focus JFA step",
                quilting_shaders::FOCUS_JFA_STEP_ENTRY_POINT,
                intermediate_format,
            )?,
            firmness: create_pipeline(
                3,
                "quilting focus firmness",
                quilting_shaders::FOCUS_FIRMNESS_ENTRY_POINT,
                intermediate_format,
            )?,
            kawase: create_pipeline(
                4,
                "quilting focus Kawase",
                quilting_shaders::FOCUS_KAWASE_ENTRY_POINT,
                intermediate_format,
            )?,
            directional_blur_intermediate: create_pipeline(
                5,
                "quilting focus directional blur intermediate",
                quilting_shaders::FOCUS_DIRECTIONAL_BLUR_ENTRY_POINT,
                wgpu::TextureFormat::Rgba8Unorm,
            )?,
            directional_blur_output: create_pipeline(
                6,
                "quilting focus directional blur output",
                quilting_shaders::FOCUS_DIRECTIONAL_BLUR_ENTRY_POINT,
                output_format,
            )?,
            bind_group_layout,
            output_format,
        })
    }

    pub fn create_focus_postprocess_target(
        &self,
        size: [u32; 2],
        pipelines: &FocusPostprocessPipelines,
    ) -> Result<FocusPostprocessTarget, LodWebGpuError> {
        if size[0] == 0 || size[1] == 0 {
            return Err(LodWebGpuError::Payload(
                "focus postprocess target dimensions must be nonzero".to_string(),
            ));
        }
        let maximum = self.device.limits().max_texture_dimension_2d;
        if size[0] > maximum || size[1] > maximum {
            return Err(LodWebGpuError::Payload(format!(
                "focus postprocess target {}x{} exceeds device limit {maximum}",
                size[0], size[1],
            )));
        }
        let jfa_size = [
            (size[0] / FOCUS_JFA_DOWNSAMPLE).max(1),
            (size[1] / FOCUS_JFA_DOWNSAMPLE).max(1),
        ];
        let (raw_field, raw_field_view) = focus_texture(
            &self.device,
            "quilting focus raw field",
            size,
            wgpu::TextureFormat::Rgba16Float,
        );
        let (selected_weight, selected_weight_view) = focus_texture(
            &self.device,
            "quilting focus selected weight",
            size,
            wgpu::TextureFormat::Rgba16Float,
        );
        let (jfa_ping, jfa_ping_view) = focus_texture(
            &self.device,
            "quilting focus JFA ping",
            jfa_size,
            wgpu::TextureFormat::Rgba16Float,
        );
        let (jfa_pong, jfa_pong_view) = focus_texture(
            &self.device,
            "quilting focus JFA pong",
            jfa_size,
            wgpu::TextureFormat::Rgba16Float,
        );
        let (firmness, firmness_view) = focus_texture(
            &self.device,
            "quilting focus firmness",
            size,
            wgpu::TextureFormat::Rgba16Float,
        );
        let (kawase, kawase_view) = focus_texture(
            &self.device,
            "quilting focus Kawase scratch",
            size,
            wgpu::TextureFormat::Rgba16Float,
        );
        let (scene_color, scene_color_view) = focus_texture(
            &self.device,
            "quilting focus scene color",
            size,
            wgpu::TextureFormat::Rgba8Unorm,
        );
        let (blur_ping, blur_ping_view) = focus_texture(
            &self.device,
            "quilting focus blur ping",
            size,
            wgpu::TextureFormat::Rgba8Unorm,
        );
        let (blur_pong, blur_pong_view) = focus_texture(
            &self.device,
            "quilting focus blur pong",
            size,
            wgpu::TextureFormat::Rgba8Unorm,
        );
        let uniform_stride_bytes = align_up(
            FOCUS_PASS_UNIFORM_BYTES,
            u64::from(self.device.limits().min_uniform_buffer_offset_alignment),
        );
        let uniform = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("quilting focus pass uniform table"),
            size: uniform_stride_bytes * FOCUS_PASS_CAPACITY as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("quilting focus linear clamp sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let mut target = FocusPostprocessTarget {
            size,
            output_format: pipelines.output_format,
            raw_field,
            raw_field_view,
            _selected_weight: selected_weight,
            selected_weight_view,
            _jfa_ping: jfa_ping,
            jfa_ping_view,
            _jfa_pong: jfa_pong,
            jfa_pong_view,
            _firmness: firmness,
            firmness_view,
            _kawase: kawase,
            kawase_view,
            scene_color,
            scene_color_view,
            _blur_ping: blur_ping,
            blur_ping_view,
            _blur_pong: blur_pong,
            blur_pong_view,
            uniform,
            _sampler: sampler.clone(),
            uniform_stride_bytes,
            bind_groups: BTreeMap::new(),
            scratch: Mutex::new(FocusEncodingScratch {
                passes: Vec::with_capacity(FOCUS_PASS_CAPACITY),
                uniform_words: Vec::with_capacity(
                    FOCUS_PASS_CAPACITY * (uniform_stride_bytes as usize / 4),
                ),
            }),
        };
        for pair in focus_binding_pairs() {
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("quilting retained focus source pair"),
                layout: pipelines.bind_group_layout(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(target.view(pair.0)),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(target.view(pair.1)),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &target.uniform,
                            offset: 0,
                            size: NonZeroU64::new(FOCUS_PASS_UNIFORM_BYTES),
                        }),
                    },
                ],
            });
            target.bind_groups.insert(pair, bind_group);
        }
        Ok(target)
    }

    /// Stage the most recently submitted raw focus MRT for parity evidence.
    /// This is an explicit copy/map boundary and is never called by the
    /// production frame encoder.
    pub fn stage_focus_raw_field_image(
        &self,
        target: &FocusPostprocessTarget,
    ) -> Result<StagedFocusRawFieldReadback, LodWebGpuError> {
        const BYTES_PER_TEXEL: u32 = 8;
        let unpadded_bytes_per_row =
            target.size[0].checked_mul(BYTES_PER_TEXEL).ok_or_else(|| {
                LodWebGpuError::Payload("focus raw-field row size overflowed".to_string())
            })?;
        let bytes_per_row = unpadded_bytes_per_row
            .div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            .checked_mul(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            .ok_or_else(|| {
                LodWebGpuError::Payload("focus raw-field padded row size overflowed".to_string())
            })?;
        let byte_len = u64::from(bytes_per_row)
            .checked_mul(u64::from(target.size[1]))
            .ok_or_else(|| {
                LodWebGpuError::Payload("focus raw-field readback size overflowed".to_string())
            })?;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("quilting focus raw-field evidence readback"),
            size: byte_len,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("quilting focus raw-field evidence copy"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target.raw_field,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(target.size[1]),
                },
            },
            wgpu::Extent3d {
                width: target.size[0],
                height: target.size[1],
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([encoder.finish()]);
        Ok(StagedFocusRawFieldReadback {
            #[cfg(not(target_arch = "wasm32"))]
            device: self.device.clone(),
            buffer,
            size: target.size,
            bytes_per_row: bytes_per_row as usize,
            byte_len,
        })
    }

    pub fn encode_focus_postprocess(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipelines: &FocusPostprocessPipelines,
        target: &FocusPostprocessTarget,
        output_view: &wgpu::TextureView,
        packet: FocusPostprocessPacket,
    ) -> Result<FocusPostprocessEncoding, LodWebGpuError> {
        if target.output_format != pipelines.output_format {
            return Err(LodWebGpuError::Payload(
                "focus target and pipeline output formats differ".to_string(),
            ));
        }
        let schedule = FocusPostprocessSchedule::build(target.size, packet)
            .map_err(|error| LodWebGpuError::Payload(error.to_string()))?;
        let mut scratch = target.scratch.lock().map_err(|_| {
            LodWebGpuError::Payload("focus encoding scratch lock was poisoned".to_string())
        })?;
        let FocusEncodingScratch {
            passes,
            uniform_words,
        } = &mut *scratch;
        build_focus_passes(schedule, target.uniform_stride_bytes, passes)?;
        if passes.len() > FOCUS_PASS_CAPACITY {
            return Err(LodWebGpuError::Payload(format!(
                "focus schedule requires {} passes; capacity is {FOCUS_PASS_CAPACITY}",
                passes.len(),
            )));
        }
        let stride_words = target.uniform_stride_bytes as usize / 4;
        let word_count = passes.len() * stride_words;
        uniform_words.clear();
        uniform_words.resize(word_count, 0.0);
        for (index, pass) in passes.iter().enumerate() {
            let start = index * stride_words;
            uniform_words[start..start + 16].copy_from_slice(&pass.uniform);
        }
        self.queue
            .write_buffer(&target.uniform, 0, bytemuck::cast_slice(uniform_words));

        let mut encoding = FocusPostprocessEncoding {
            render_passes: 0,
            primary_jfa_passes: schedule
                .jfa_plan()
                .map_or(0, |plan| plan.primary_step_count()),
            cleanup_jfa_passes: schedule
                .jfa_plan()
                .map_or(0, |plan| plan.cleanup_steps().len() as u32),
            kawase_passes: u32::from(packet.kawase_passes),
            directional_blur_passes: schedule.directional_blur_passes().len() as u32,
            uniform_upload_bytes: (passes.len() as u64) * target.uniform_stride_bytes,
        };
        for pass in passes {
            let destination = match pass.destination {
                FocusDestination::Texture(slot) => target.view(slot),
                FocusDestination::Output => output_view,
            };
            let pipeline = match pass.pipeline {
                FocusPipelineKind::SelectWeight => pipelines.select_weight(),
                FocusPipelineKind::JfaInit => pipelines.jfa_init(),
                FocusPipelineKind::JfaStep => pipelines.jfa_step(),
                FocusPipelineKind::Firmness => pipelines.firmness(),
                FocusPipelineKind::Kawase => pipelines.kawase(),
                FocusPipelineKind::DirectionalBlur => pipelines.directional_blur(pass.final_output),
            };
            let bind_group = target.bind_group(pass.source_a, pass.source_b)?;
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("quilting scheduled focus pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: destination,
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
            render_pass.set_pipeline(pipeline);
            render_pass.set_bind_group(0, bind_group, &[pass.uniform_offset]);
            render_pass.draw(0..3, 0..1);
            encoding.render_passes = encoding.render_passes.saturating_add(1);
        }
        Ok(encoding)
    }
}

fn sampled_texture_layout(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn focus_texture(
    device: &wgpu::Device,
    label: &'static str,
    size: [u32; 2],
    format: wgpu::TextureFormat,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn align_up(value: u64, alignment: u64) -> u64 {
    let alignment = alignment.max(1);
    value.div_ceil(alignment) * alignment
}

fn focus_binding_pairs() -> impl IntoIterator<Item = (FocusTextureSlot, FocusTextureSlot)> {
    use FocusTextureSlot::*;
    [
        (RawField, RawField),
        (SelectedWeight, SelectedWeight),
        (JfaPing, JfaPing),
        (JfaPong, JfaPong),
        (JfaPing, SelectedWeight),
        (JfaPong, SelectedWeight),
        (Firmness, Firmness),
        (Kawase, Kawase),
        (Scene, Firmness),
        (Scene, Kawase),
        (BlurPing, Firmness),
        (BlurPing, Kawase),
        (BlurPong, Firmness),
        (BlurPong, Kawase),
    ]
}

fn build_focus_passes(
    schedule: FocusPostprocessSchedule,
    uniform_stride_bytes: u64,
    output: &mut Vec<PlannedFocusPass>,
) -> Result<(), LodWebGpuError> {
    output.clear();
    let packet = schedule.packet();
    let full = schedule.full_extent();
    let jfa = schedule.jfa_extent();
    let base_uniform = |extent: [f32; 2]| {
        let mut uniform = [0.0; 16];
        uniform[0] = extent[0];
        uniform[1] = extent[1];
        uniform[4] = f32::from(packet.blur_radius_pixels);
        uniform[5] = packet.focus_coordinate;
        uniform[6] = packet.bandwidth;
        uniform[7] = f32::from(packet.mode.wire_index());
        uniform[8] = f32::from(packet.blur_radius_pixels);
        uniform[9] = schedule.per_subpass_blur_strength();
        uniform[11] = if packet.normalize_range { 1.0 } else { 0.0 };
        uniform[12] = packet.stretch_range[0];
        uniform[13] = packet.stretch_range[1];
        uniform
    };
    output.push(PlannedFocusPass {
        pipeline: FocusPipelineKind::SelectWeight,
        source_a: FocusTextureSlot::RawField,
        source_b: FocusTextureSlot::RawField,
        destination: FocusDestination::Texture(FocusTextureSlot::SelectedWeight),
        uniform: base_uniform([full.width as f32, full.height as f32]),
        uniform_offset: 0,
        final_output: false,
    });

    let firmness_source = if let Some(jfa_plan) = schedule.jfa_plan() {
        output.push(PlannedFocusPass {
            pipeline: FocusPipelineKind::JfaInit,
            source_a: FocusTextureSlot::SelectedWeight,
            source_b: FocusTextureSlot::SelectedWeight,
            destination: FocusDestination::Texture(FocusTextureSlot::JfaPing),
            uniform: base_uniform([jfa.width as f32, jfa.height as f32]),
            uniform_offset: 0,
            final_output: false,
        });
        for propagation in jfa_plan.primary_steps().chain(jfa_plan.cleanup_steps()) {
            let mut uniform = base_uniform([jfa.width as f32, jfa.height as f32]);
            uniform[2] = propagation.step as f32;
            output.push(PlannedFocusPass {
                pipeline: FocusPipelineKind::JfaStep,
                source_a: jfa_slot(propagation.source),
                source_b: jfa_slot(propagation.source),
                destination: FocusDestination::Texture(jfa_slot(propagation.destination)),
                uniform,
                uniform_offset: 0,
                final_output: false,
            });
        }
        jfa_slot(jfa_plan.final_buffer())
    } else {
        FocusTextureSlot::SelectedWeight
    };
    let mut firmness_uniform = base_uniform([jfa.width as f32, jfa.height as f32]);
    if schedule.uses_jfa() {
        firmness_uniform[4] /= FOCUS_JFA_DOWNSAMPLE as f32;
    }
    output.push(PlannedFocusPass {
        pipeline: FocusPipelineKind::Firmness,
        source_a: firmness_source,
        source_b: FocusTextureSlot::SelectedWeight,
        destination: FocusDestination::Texture(FocusTextureSlot::Firmness),
        uniform: firmness_uniform,
        uniform_offset: 0,
        final_output: false,
    });

    let mut mask = FocusTextureSlot::Firmness;
    for pass in 0..packet.kawase_passes {
        let destination = if mask == FocusTextureSlot::Firmness {
            FocusTextureSlot::Kawase
        } else {
            FocusTextureSlot::Firmness
        };
        let mut uniform = base_uniform([full.width as f32, full.height as f32]);
        uniform[3] = schedule
            .kawase_offset(pass)
            .expect("pass is bounded by packet Kawase count");
        output.push(PlannedFocusPass {
            pipeline: FocusPipelineKind::Kawase,
            source_a: mask,
            source_b: mask,
            destination: FocusDestination::Texture(destination),
            uniform,
            uniform_offset: 0,
            final_output: false,
        });
        mask = destination;
    }

    for blur in schedule.directional_blur_passes() {
        let source_a = blur_slot(blur.source)?;
        let destination = match blur.destination {
            FocusBlurSurface::Output => FocusDestination::Output,
            surface => FocusDestination::Texture(blur_slot(surface)?),
        };
        let mut uniform = base_uniform(blur.direction);
        uniform[10] = if blur.is_final { 1.0 } else { 0.0 };
        output.push(PlannedFocusPass {
            pipeline: FocusPipelineKind::DirectionalBlur,
            source_a,
            source_b: mask,
            destination,
            uniform,
            uniform_offset: 0,
            final_output: blur.is_final,
        });
    }
    for (index, pass) in output.iter_mut().enumerate() {
        let offset = (index as u64)
            .checked_mul(uniform_stride_bytes)
            .and_then(|offset| u32::try_from(offset).ok())
            .ok_or_else(|| {
                LodWebGpuError::Payload("focus uniform offset exceeds u32".to_string())
            })?;
        pass.uniform_offset = offset;
    }
    Ok(())
}

fn jfa_slot(buffer: FocusPingPong) -> FocusTextureSlot {
    match buffer {
        FocusPingPong::Ping => FocusTextureSlot::JfaPing,
        FocusPingPong::Pong => FocusTextureSlot::JfaPong,
    }
}

fn blur_slot(surface: FocusBlurSurface) -> Result<FocusTextureSlot, LodWebGpuError> {
    match surface {
        FocusBlurSurface::Scene => Ok(FocusTextureSlot::Scene),
        FocusBlurSurface::Ping => Ok(FocusTextureSlot::BlurPing),
        FocusBlurSurface::Pong => Ok(FocusTextureSlot::BlurPong),
        FocusBlurSurface::Output => Err(LodWebGpuError::Payload(
            "focus output surface cannot be sampled".to_string(),
        )),
    }
}

fn decode_focus_raw_field(
    bytes: &[u8],
    size: [u32; 2],
    bytes_per_row: usize,
) -> Result<FocusRawFieldImage, LodWebGpuError> {
    const BYTES_PER_TEXEL: usize = 8;
    let width = usize::try_from(size[0])
        .map_err(|_| LodWebGpuError::Mapping("focus raw-field width exceeds usize".to_string()))?;
    let height = usize::try_from(size[1])
        .map_err(|_| LodWebGpuError::Mapping("focus raw-field height exceeds usize".to_string()))?;
    let row_bytes = width.checked_mul(BYTES_PER_TEXEL).ok_or_else(|| {
        LodWebGpuError::Mapping("focus raw-field row length overflowed".to_string())
    })?;
    let required = bytes_per_row.checked_mul(height).ok_or_else(|| {
        LodWebGpuError::Mapping("focus raw-field mapped length overflowed".to_string())
    })?;
    if bytes_per_row < row_bytes || bytes.len() < required {
        return Err(LodWebGpuError::Mapping(format!(
            "focus raw-field mapping is malformed: {} bytes, stride {bytes_per_row}, payload row {row_bytes}, height {height}",
            bytes.len(),
        )));
    }
    let texel_count = width.checked_mul(height).ok_or_else(|| {
        LodWebGpuError::Mapping("focus raw-field texel count overflowed".to_string())
    })?;
    let mut texels = Vec::with_capacity(texel_count);
    for row in bytes.chunks_exact(bytes_per_row).take(height) {
        for texel in row[..row_bytes].chunks_exact(BYTES_PER_TEXEL) {
            texels.push(std::array::from_fn(|channel| {
                let offset = channel * 2;
                half::f16::from_bits(u16::from_le_bytes([texel[offset], texel[offset + 1]]))
                    .to_f32()
            }));
        }
    }
    Ok(FocusRawFieldImage { size, texels })
}

#[cfg(test)]
mod tests {
    use super::*;
    use quilting_core::render::FocusPostprocessMode;

    fn packet(mode: FocusPostprocessMode) -> FocusPostprocessPacket {
        FocusPostprocessPacket {
            mode,
            blur_radius_pixels: 11,
            blur_strength: 3.0,
            focus_coordinate: 0.62,
            bandwidth: 0.1,
            normalize_range: false,
            stretch_range: [0.5, 0.5],
            gaussian_passes: 1,
            kawase_passes: 3,
            kawase_offset: 1.5,
        }
    }

    #[test]
    fn functional_pipeline_family_names_every_pass_and_target_exactly() {
        let descriptors =
            focus_postprocess_pipeline_descriptors(functional::TextureFormat::Bgra8UnormSrgb)
                .unwrap();
        assert_eq!(descriptors.len(), 7);
        assert!(descriptors[..5].iter().all(|descriptor| {
            descriptor.color_targets()[0].format == functional::TextureFormat::Rgba16Float
        }));
        assert_eq!(
            descriptors[5].color_targets()[0].format,
            functional::TextureFormat::Rgba8Unorm,
        );
        assert_eq!(
            descriptors[6].color_targets()[0].format,
            functional::TextureFormat::Bgra8UnormSrgb,
        );
        assert_eq!(
            descriptors[5].program().fragment().unwrap().entry_point(),
            quilting_shaders::FOCUS_DIRECTIONAL_BLUR_ENTRY_POINT,
        );
        assert_eq!(
            descriptors[5].program().fragment(),
            descriptors[6].program().fragment(),
            "the intermediate and output blur differ only in attachment state",
        );
        let groups = descriptors[0].layout().groups();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].entries().len(), 4);
        assert_eq!(descriptors[0].multisample().count, 1);

        let rgba =
            focus_postprocess_pipeline_descriptors(functional::TextureFormat::Rgba8Unorm).unwrap();
        assert_ne!(descriptors, rgba);
        let mut memo = quilting_core::render_memo::DeviceMemo::new(0);
        memo.get_or_try_insert_with(descriptors.clone(), |_| Ok::<_, ()>(7))
            .unwrap();
        assert_eq!(
            *memo
                .get_or_try_insert_with(descriptors, |_| Ok::<_, ()>(8))
                .unwrap(),
            7,
        );
        assert_eq!(memo.diagnostics().misses, 1);
        assert_eq!(memo.diagnostics().hits, 1);
    }

    #[test]
    fn spheroidal_plan_bypasses_jfa_and_finishes_in_output() {
        let schedule =
            FocusPostprocessSchedule::build([1280, 720], packet(FocusPostprocessMode::Spheroidal))
                .unwrap();
        let mut passes = Vec::new();
        build_focus_passes(schedule, 256, &mut passes).unwrap();
        assert_eq!(passes.len(), 8);
        assert_eq!(passes[0].pipeline, FocusPipelineKind::SelectWeight);
        assert!(!passes.iter().any(|pass| matches!(
            pass.pipeline,
            FocusPipelineKind::JfaInit | FocusPipelineKind::JfaStep
        )));
        assert_eq!(passes[1].pipeline, FocusPipelineKind::Firmness);
        assert_eq!(passes[4].source_a, FocusTextureSlot::Firmness);
        assert_eq!(
            passes[4].destination,
            FocusDestination::Texture(FocusTextureSlot::Kawase)
        );
        assert_eq!(passes.last().unwrap().destination, FocusDestination::Output);
        assert_eq!(passes.last().unwrap().uniform_offset, 7 * 256);
    }

    #[test]
    fn conformal_plan_uses_exact_jfa_parity_and_reused_binding_pairs() {
        let schedule = FocusPostprocessSchedule::build(
            [1280, 720],
            packet(FocusPostprocessMode::ConformalStretch),
        )
        .unwrap();
        let primary = schedule.jfa_plan().unwrap().primary_step_count() as usize;
        let mut passes = Vec::new();
        build_focus_passes(schedule, 256, &mut passes).unwrap();
        assert_eq!(
            passes
                .iter()
                .filter(|pass| pass.pipeline == FocusPipelineKind::JfaStep)
                .count(),
            primary + 2,
        );
        assert!(passes.iter().all(|pass| focus_binding_pairs()
            .into_iter()
            .any(|pair| pair == (pass.source_a, pass.source_b))));
        assert_eq!(passes.last().unwrap().destination, FocusDestination::Output);
        assert!(passes.len() <= FOCUS_PASS_CAPACITY);
    }

    #[test]
    fn raw_field_decoder_discards_padding_and_reports_covered_ranges() {
        let encode = |texel: [f32; 4]| {
            texel
                .into_iter()
                .flat_map(|value| half::f16::from_f32(value).to_bits().to_le_bytes())
                .collect::<Vec<_>>()
        };
        let mut bytes = vec![0; 32];
        bytes[..8].copy_from_slice(&encode([0.25, 0.5, 0.75, 1.0]));
        bytes[8..16].copy_from_slice(&encode([0.0, 0.0, 0.0, 0.0]));
        let image = decode_focus_raw_field(&bytes, [2, 1], 32).unwrap();
        assert_eq!(image.size(), [2, 1]);
        assert_eq!(image.covered_texels(), 1);
        assert_eq!(image.covered_channel_range(0), Some([0.25, 0.25]));
        assert_eq!(image.covered_channel_range(2), Some([0.75, 0.75]));
        assert_eq!(image.covered_channel_range(4), None);
    }
}
