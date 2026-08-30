//! Retained WebGPU pipelines for the shared focus-composition schedule.
//!
//! The pipeline family, retained intermediate textures, bind groups, and pass
//! encoder live here. Scene rendering into the focus MRT remains a separate
//! integration boundary so ordinary rendering cannot accidentally claim focus
//! support before the composed output has evidence.

use crate::{LodClassifierDevice, LodWebGpuError};
use quilting_core::focus_postprocess::{
    FocusBlurSurface, FocusPingPong, FocusPostprocessSchedule, FOCUS_JFA_DOWNSAMPLE,
};
use quilting_core::render::FocusPostprocessPacket;
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::num::NonZeroU64;
use std::sync::Mutex;

const FOCUS_PASS_UNIFORM_BYTES: u64 = 64;
const FOCUS_PASS_CAPACITY: usize = 64;

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

pub struct FocusPostprocessTarget {
    size: [u32; 2],
    output_format: wgpu::TextureFormat,
    _raw_field: wgpu::Texture,
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

impl FocusPostprocessTarget {
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
    pub fn create_focus_postprocess_pipelines(
        &self,
        output_format: wgpu::TextureFormat,
    ) -> Result<FocusPostprocessPipelines, LodWebGpuError> {
        let source = quilting_shaders::compile_focus_postprocess_wgsl()
            .map_err(|error| LodWebGpuError::Shader(error.to_string()))?;
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("quilting focus postprocess"),
                source: wgpu::ShaderSource::Wgsl(Cow::Owned(source)),
            });
        let bind_group_layout =
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
                });
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("quilting focus postprocess pipeline layout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });
        let create_pipeline = |label: &'static str,
                               fragment_entry: &'static str,
                               format: wgpu::TextureFormat| {
            self.device
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
                        targets: &[Some(wgpu::ColorTargetState {
                            format,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    multiview_mask: None,
                    cache: None,
                })
        };
        let intermediate_format = wgpu::TextureFormat::Rgba16Float;
        Ok(FocusPostprocessPipelines {
            select_weight: create_pipeline(
                "quilting focus select weight",
                quilting_shaders::FOCUS_SELECT_WEIGHT_ENTRY_POINT,
                intermediate_format,
            ),
            jfa_init: create_pipeline(
                "quilting focus JFA init",
                quilting_shaders::FOCUS_JFA_INIT_ENTRY_POINT,
                intermediate_format,
            ),
            jfa_step: create_pipeline(
                "quilting focus JFA step",
                quilting_shaders::FOCUS_JFA_STEP_ENTRY_POINT,
                intermediate_format,
            ),
            firmness: create_pipeline(
                "quilting focus firmness",
                quilting_shaders::FOCUS_FIRMNESS_ENTRY_POINT,
                intermediate_format,
            ),
            kawase: create_pipeline(
                "quilting focus Kawase",
                quilting_shaders::FOCUS_KAWASE_ENTRY_POINT,
                intermediate_format,
            ),
            directional_blur_intermediate: create_pipeline(
                "quilting focus directional blur intermediate",
                quilting_shaders::FOCUS_DIRECTIONAL_BLUR_ENTRY_POINT,
                wgpu::TextureFormat::Rgba8Unorm,
            ),
            directional_blur_output: create_pipeline(
                "quilting focus directional blur output",
                quilting_shaders::FOCUS_DIRECTIONAL_BLUR_ENTRY_POINT,
                output_format,
            ),
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
            _raw_field: raw_field,
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
}
