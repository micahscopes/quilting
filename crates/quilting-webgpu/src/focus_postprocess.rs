//! Retained WebGPU pipelines for the shared focus-composition schedule.
//!
//! Texture residency and pass encoding are deliberately separate follow-up
//! cuts. Creating this family proves every WGSL entry against the actual wgpu
//! device and freezes one compatible bind-group layout for all passes.

use crate::{LodClassifierDevice, LodWebGpuError};
use std::borrow::Cow;
use std::num::NonZeroU64;

const FOCUS_PASS_UNIFORM_BYTES: u64 = 64;

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
