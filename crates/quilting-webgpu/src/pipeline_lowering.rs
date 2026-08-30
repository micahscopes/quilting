//! Mechanical lowering of Quilting's immutable functional pipeline values.
//!
//! Keep these conversions free of application state and GPU ownership. The
//! caller decides memoization and lifecycle; this module only translates the
//! common descriptor subset into `wgpu` state.

use crate::LodWebGpuError;
use quilting_core::render_pipeline as functional;
use std::num::NonZeroU64;

pub(crate) fn functional_texture_format(
    format: wgpu::TextureFormat,
) -> Option<functional::TextureFormat> {
    Some(match format {
        wgpu::TextureFormat::R8Unorm => functional::TextureFormat::R8Unorm,
        wgpu::TextureFormat::Rg8Unorm => functional::TextureFormat::Rg8Unorm,
        wgpu::TextureFormat::Rgba8Unorm => functional::TextureFormat::Rgba8Unorm,
        wgpu::TextureFormat::Rgba8UnormSrgb => functional::TextureFormat::Rgba8UnormSrgb,
        wgpu::TextureFormat::Bgra8Unorm => functional::TextureFormat::Bgra8Unorm,
        wgpu::TextureFormat::Bgra8UnormSrgb => functional::TextureFormat::Bgra8UnormSrgb,
        wgpu::TextureFormat::Rgb10a2Unorm => functional::TextureFormat::Rgb10a2Unorm,
        wgpu::TextureFormat::R16Float => functional::TextureFormat::R16Float,
        wgpu::TextureFormat::Rg16Float => functional::TextureFormat::Rg16Float,
        wgpu::TextureFormat::Rgba16Float => functional::TextureFormat::Rgba16Float,
        wgpu::TextureFormat::R32Float => functional::TextureFormat::R32Float,
        wgpu::TextureFormat::Rg32Float => functional::TextureFormat::Rg32Float,
        wgpu::TextureFormat::Rgba32Float => functional::TextureFormat::Rgba32Float,
        wgpu::TextureFormat::R32Uint => functional::TextureFormat::R32Uint,
        wgpu::TextureFormat::Rgba32Uint => functional::TextureFormat::Rgba32Uint,
        wgpu::TextureFormat::Depth24Plus => functional::TextureFormat::Depth24Plus,
        wgpu::TextureFormat::Depth24PlusStencil8 => functional::TextureFormat::Depth24PlusStencil8,
        wgpu::TextureFormat::Depth32Float => functional::TextureFormat::Depth32Float,
        _ => return None,
    })
}

pub(crate) fn texture_format(format: functional::TextureFormat) -> wgpu::TextureFormat {
    match format {
        functional::TextureFormat::R8Unorm => wgpu::TextureFormat::R8Unorm,
        functional::TextureFormat::Rg8Unorm => wgpu::TextureFormat::Rg8Unorm,
        functional::TextureFormat::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
        functional::TextureFormat::Rgba8UnormSrgb => wgpu::TextureFormat::Rgba8UnormSrgb,
        functional::TextureFormat::Bgra8Unorm => wgpu::TextureFormat::Bgra8Unorm,
        functional::TextureFormat::Bgra8UnormSrgb => wgpu::TextureFormat::Bgra8UnormSrgb,
        functional::TextureFormat::Rgb10a2Unorm => wgpu::TextureFormat::Rgb10a2Unorm,
        functional::TextureFormat::R16Float => wgpu::TextureFormat::R16Float,
        functional::TextureFormat::Rg16Float => wgpu::TextureFormat::Rg16Float,
        functional::TextureFormat::Rgba16Float => wgpu::TextureFormat::Rgba16Float,
        functional::TextureFormat::R32Float => wgpu::TextureFormat::R32Float,
        functional::TextureFormat::Rg32Float => wgpu::TextureFormat::Rg32Float,
        functional::TextureFormat::Rgba32Float => wgpu::TextureFormat::Rgba32Float,
        functional::TextureFormat::R32Uint => wgpu::TextureFormat::R32Uint,
        functional::TextureFormat::Rgba32Uint => wgpu::TextureFormat::Rgba32Uint,
        functional::TextureFormat::Depth24Plus => wgpu::TextureFormat::Depth24Plus,
        functional::TextureFormat::Depth24PlusStencil8 => wgpu::TextureFormat::Depth24PlusStencil8,
        functional::TextureFormat::Depth32Float => wgpu::TextureFormat::Depth32Float,
    }
}

fn shader_stages(visibility: functional::ShaderVisibility) -> wgpu::ShaderStages {
    let mut stages = wgpu::ShaderStages::empty();
    if visibility.contains(functional::ShaderStage::Vertex) {
        stages |= wgpu::ShaderStages::VERTEX;
    }
    if visibility.contains(functional::ShaderStage::Fragment) {
        stages |= wgpu::ShaderStages::FRAGMENT;
    }
    if visibility.contains(functional::ShaderStage::Compute) {
        stages |= wgpu::ShaderStages::COMPUTE;
    }
    stages
}

fn texture_view_dimension(
    dimension: functional::TextureViewDimension,
) -> wgpu::TextureViewDimension {
    match dimension {
        functional::TextureViewDimension::D1 => wgpu::TextureViewDimension::D1,
        functional::TextureViewDimension::D2 => wgpu::TextureViewDimension::D2,
        functional::TextureViewDimension::D2Array => wgpu::TextureViewDimension::D2Array,
        functional::TextureViewDimension::Cube => wgpu::TextureViewDimension::Cube,
        functional::TextureViewDimension::CubeArray => wgpu::TextureViewDimension::CubeArray,
        functional::TextureViewDimension::D3 => wgpu::TextureViewDimension::D3,
    }
}

fn binding_type(kind: functional::BindingKind) -> wgpu::BindingType {
    match kind {
        functional::BindingKind::UniformBuffer {
            dynamic_offset,
            minimum_size,
        } => wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: dynamic_offset,
            min_binding_size: NonZeroU64::new(minimum_size),
        },
        functional::BindingKind::StorageBuffer {
            read_only,
            dynamic_offset,
            minimum_size,
        } => wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: dynamic_offset,
            min_binding_size: NonZeroU64::new(minimum_size),
        },
        functional::BindingKind::Texture {
            sample_kind,
            view_dimension,
            multisampled,
        } => wgpu::BindingType::Texture {
            sample_type: match sample_kind {
                functional::TextureSampleKind::FloatFilterable => {
                    wgpu::TextureSampleType::Float { filterable: true }
                }
                functional::TextureSampleKind::FloatUnfilterable => {
                    wgpu::TextureSampleType::Float { filterable: false }
                }
                functional::TextureSampleKind::Sint => wgpu::TextureSampleType::Sint,
                functional::TextureSampleKind::Uint => wgpu::TextureSampleType::Uint,
                functional::TextureSampleKind::Depth => wgpu::TextureSampleType::Depth,
            },
            view_dimension: texture_view_dimension(view_dimension),
            multisampled,
        },
        functional::BindingKind::StorageTexture {
            access,
            format,
            view_dimension,
        } => wgpu::BindingType::StorageTexture {
            access: match access {
                functional::StorageTextureAccess::ReadOnly => wgpu::StorageTextureAccess::ReadOnly,
                functional::StorageTextureAccess::WriteOnly => {
                    wgpu::StorageTextureAccess::WriteOnly
                }
                functional::StorageTextureAccess::ReadWrite => {
                    wgpu::StorageTextureAccess::ReadWrite
                }
            },
            format: texture_format(format),
            view_dimension: texture_view_dimension(view_dimension),
        },
        functional::BindingKind::Sampler(kind) => wgpu::BindingType::Sampler(match kind {
            functional::SamplerBindingKind::Filtering => wgpu::SamplerBindingType::Filtering,
            functional::SamplerBindingKind::NonFiltering => wgpu::SamplerBindingType::NonFiltering,
            functional::SamplerBindingKind::Comparison => wgpu::SamplerBindingType::Comparison,
        }),
    }
}

pub(crate) fn bind_group_layout(
    device: &wgpu::Device,
    label: &str,
    descriptor: &functional::BindGroupLayoutDescriptor,
) -> wgpu::BindGroupLayout {
    let entries = descriptor
        .entries()
        .iter()
        .map(|entry| wgpu::BindGroupLayoutEntry {
            binding: entry.binding,
            visibility: shader_stages(entry.visibility),
            ty: binding_type(entry.kind),
            count: None,
        })
        .collect::<Vec<_>>();
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries,
    })
}

pub(crate) fn primitive_state(
    descriptor: functional::PrimitiveStateDescriptor,
) -> wgpu::PrimitiveState {
    wgpu::PrimitiveState {
        topology: match descriptor.topology {
            functional::PrimitiveTopology::PointList => wgpu::PrimitiveTopology::PointList,
            functional::PrimitiveTopology::LineList => wgpu::PrimitiveTopology::LineList,
            functional::PrimitiveTopology::LineStrip => wgpu::PrimitiveTopology::LineStrip,
            functional::PrimitiveTopology::TriangleList => wgpu::PrimitiveTopology::TriangleList,
            functional::PrimitiveTopology::TriangleStrip => wgpu::PrimitiveTopology::TriangleStrip,
        },
        strip_index_format: descriptor.strip_index_format.map(|format| match format {
            functional::IndexFormat::Uint16 => wgpu::IndexFormat::Uint16,
            functional::IndexFormat::Uint32 => wgpu::IndexFormat::Uint32,
        }),
        front_face: match descriptor.front_face {
            functional::FrontFace::CounterClockwise => wgpu::FrontFace::Ccw,
            functional::FrontFace::Clockwise => wgpu::FrontFace::Cw,
        },
        cull_mode: match descriptor.cull_mode {
            functional::CullMode::None => None,
            functional::CullMode::Front => Some(wgpu::Face::Front),
            functional::CullMode::Back => Some(wgpu::Face::Back),
        },
        ..Default::default()
    }
}

pub(crate) fn vertex_format(format: functional::VertexFormat) -> wgpu::VertexFormat {
    match format {
        functional::VertexFormat::Uint8x2 => wgpu::VertexFormat::Uint8x2,
        functional::VertexFormat::Uint8x4 => wgpu::VertexFormat::Uint8x4,
        functional::VertexFormat::Sint8x2 => wgpu::VertexFormat::Sint8x2,
        functional::VertexFormat::Sint8x4 => wgpu::VertexFormat::Sint8x4,
        functional::VertexFormat::Unorm8x2 => wgpu::VertexFormat::Unorm8x2,
        functional::VertexFormat::Unorm8x4 => wgpu::VertexFormat::Unorm8x4,
        functional::VertexFormat::Snorm8x2 => wgpu::VertexFormat::Snorm8x2,
        functional::VertexFormat::Snorm8x4 => wgpu::VertexFormat::Snorm8x4,
        functional::VertexFormat::Uint16x2 => wgpu::VertexFormat::Uint16x2,
        functional::VertexFormat::Uint16x4 => wgpu::VertexFormat::Uint16x4,
        functional::VertexFormat::Sint16x2 => wgpu::VertexFormat::Sint16x2,
        functional::VertexFormat::Sint16x4 => wgpu::VertexFormat::Sint16x4,
        functional::VertexFormat::Unorm16x2 => wgpu::VertexFormat::Unorm16x2,
        functional::VertexFormat::Unorm16x4 => wgpu::VertexFormat::Unorm16x4,
        functional::VertexFormat::Snorm16x2 => wgpu::VertexFormat::Snorm16x2,
        functional::VertexFormat::Snorm16x4 => wgpu::VertexFormat::Snorm16x4,
        functional::VertexFormat::Float16x2 => wgpu::VertexFormat::Float16x2,
        functional::VertexFormat::Float16x4 => wgpu::VertexFormat::Float16x4,
        functional::VertexFormat::Float32 => wgpu::VertexFormat::Float32,
        functional::VertexFormat::Float32x2 => wgpu::VertexFormat::Float32x2,
        functional::VertexFormat::Float32x3 => wgpu::VertexFormat::Float32x3,
        functional::VertexFormat::Float32x4 => wgpu::VertexFormat::Float32x4,
        functional::VertexFormat::Uint32 => wgpu::VertexFormat::Uint32,
        functional::VertexFormat::Uint32x2 => wgpu::VertexFormat::Uint32x2,
        functional::VertexFormat::Uint32x3 => wgpu::VertexFormat::Uint32x3,
        functional::VertexFormat::Uint32x4 => wgpu::VertexFormat::Uint32x4,
        functional::VertexFormat::Sint32 => wgpu::VertexFormat::Sint32,
        functional::VertexFormat::Sint32x2 => wgpu::VertexFormat::Sint32x2,
        functional::VertexFormat::Sint32x3 => wgpu::VertexFormat::Sint32x3,
        functional::VertexFormat::Sint32x4 => wgpu::VertexFormat::Sint32x4,
    }
}

pub(crate) fn vertex_step_mode(mode: functional::VertexStepMode) -> wgpu::VertexStepMode {
    match mode {
        functional::VertexStepMode::Vertex => wgpu::VertexStepMode::Vertex,
        functional::VertexStepMode::Instance => wgpu::VertexStepMode::Instance,
    }
}

fn compare_function(compare: functional::CompareFunction) -> wgpu::CompareFunction {
    match compare {
        functional::CompareFunction::Never => wgpu::CompareFunction::Never,
        functional::CompareFunction::Less => wgpu::CompareFunction::Less,
        functional::CompareFunction::Equal => wgpu::CompareFunction::Equal,
        functional::CompareFunction::LessEqual => wgpu::CompareFunction::LessEqual,
        functional::CompareFunction::Greater => wgpu::CompareFunction::Greater,
        functional::CompareFunction::NotEqual => wgpu::CompareFunction::NotEqual,
        functional::CompareFunction::GreaterEqual => wgpu::CompareFunction::GreaterEqual,
        functional::CompareFunction::Always => wgpu::CompareFunction::Always,
    }
}

fn stencil_operation(operation: functional::StencilOperation) -> wgpu::StencilOperation {
    match operation {
        functional::StencilOperation::Keep => wgpu::StencilOperation::Keep,
        functional::StencilOperation::Zero => wgpu::StencilOperation::Zero,
        functional::StencilOperation::Replace => wgpu::StencilOperation::Replace,
        functional::StencilOperation::Invert => wgpu::StencilOperation::Invert,
        functional::StencilOperation::IncrementClamp => wgpu::StencilOperation::IncrementClamp,
        functional::StencilOperation::DecrementClamp => wgpu::StencilOperation::DecrementClamp,
        functional::StencilOperation::IncrementWrap => wgpu::StencilOperation::IncrementWrap,
        functional::StencilOperation::DecrementWrap => wgpu::StencilOperation::DecrementWrap,
    }
}

fn stencil_face_state(
    descriptor: functional::StencilFaceStateDescriptor,
) -> wgpu::StencilFaceState {
    wgpu::StencilFaceState {
        compare: compare_function(descriptor.compare),
        fail_op: stencil_operation(descriptor.fail_op),
        depth_fail_op: stencil_operation(descriptor.depth_fail_op),
        pass_op: stencil_operation(descriptor.pass_op),
    }
}

pub(crate) fn depth_stencil_state(
    descriptor: functional::DepthStencilStateDescriptor,
) -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: texture_format(descriptor.format),
        depth_write_enabled: Some(descriptor.depth_write_enabled),
        depth_compare: Some(compare_function(descriptor.depth_compare)),
        stencil: wgpu::StencilState {
            front: stencil_face_state(descriptor.stencil_front),
            back: stencil_face_state(descriptor.stencil_back),
            read_mask: descriptor.stencil_read_mask,
            write_mask: descriptor.stencil_write_mask,
        },
        bias: wgpu::DepthBiasState {
            constant: descriptor.depth_bias_constant,
            slope_scale: descriptor.depth_bias_slope_scale.get(),
            clamp: descriptor.depth_bias_clamp.get(),
        },
    }
}

fn blend_factor(factor: functional::BlendFactor) -> wgpu::BlendFactor {
    match factor {
        functional::BlendFactor::Zero => wgpu::BlendFactor::Zero,
        functional::BlendFactor::One => wgpu::BlendFactor::One,
        functional::BlendFactor::Source => wgpu::BlendFactor::Src,
        functional::BlendFactor::OneMinusSource => wgpu::BlendFactor::OneMinusSrc,
        functional::BlendFactor::SourceAlpha => wgpu::BlendFactor::SrcAlpha,
        functional::BlendFactor::OneMinusSourceAlpha => wgpu::BlendFactor::OneMinusSrcAlpha,
        functional::BlendFactor::Destination => wgpu::BlendFactor::Dst,
        functional::BlendFactor::OneMinusDestination => wgpu::BlendFactor::OneMinusDst,
        functional::BlendFactor::DestinationAlpha => wgpu::BlendFactor::DstAlpha,
        functional::BlendFactor::OneMinusDestinationAlpha => wgpu::BlendFactor::OneMinusDstAlpha,
    }
}

fn blend_component(component: functional::BlendComponentDescriptor) -> wgpu::BlendComponent {
    wgpu::BlendComponent {
        src_factor: blend_factor(component.source_factor),
        dst_factor: blend_factor(component.destination_factor),
        operation: match component.operation {
            functional::BlendOperation::Add => wgpu::BlendOperation::Add,
            functional::BlendOperation::Subtract => wgpu::BlendOperation::Subtract,
            functional::BlendOperation::ReverseSubtract => wgpu::BlendOperation::ReverseSubtract,
            functional::BlendOperation::Min => wgpu::BlendOperation::Min,
            functional::BlendOperation::Max => wgpu::BlendOperation::Max,
        },
    }
}

pub(crate) fn color_target_state(
    descriptor: functional::ColorTargetStateDescriptor,
) -> Result<wgpu::ColorTargetState, LodWebGpuError> {
    let write_mask = wgpu::ColorWrites::from_bits(u32::from(descriptor.write_mask.bits()))
        .ok_or_else(|| {
            LodWebGpuError::Payload("functional color write mask is invalid".to_string())
        })?;
    Ok(wgpu::ColorTargetState {
        format: texture_format(descriptor.format),
        blend: descriptor.blend.map(|blend| wgpu::BlendState {
            color: blend_component(blend.color),
            alpha: blend_component(blend.alpha),
        }),
        write_mask,
    })
}

pub(crate) fn multisample_state(
    descriptor: functional::MultisampleStateDescriptor,
) -> wgpu::MultisampleState {
    wgpu::MultisampleState {
        count: descriptor.count,
        mask: descriptor.mask,
        alpha_to_coverage_enabled: descriptor.alpha_to_coverage_enabled,
    }
}

/// Fully lowered fixed state whose borrowed vertex attributes remain scoped
/// to one backend construction effect. Keeping this value callback-local
/// avoids self-referential storage while giving every pipeline family the
/// same mechanical descriptor lowering.
pub(crate) struct LoweredRenderPipelineState<'descriptor, 'attributes> {
    pub(crate) vertex_entry_point: &'descriptor str,
    pub(crate) fragment_entry_point: Option<&'descriptor str>,
    pub(crate) vertex_buffers: Vec<wgpu::VertexBufferLayout<'attributes>>,
    pub(crate) primitive: wgpu::PrimitiveState,
    pub(crate) depth_stencil: Option<wgpu::DepthStencilState>,
    pub(crate) multisample: wgpu::MultisampleState,
    pub(crate) targets: Vec<Option<wgpu::ColorTargetState>>,
}

pub(crate) fn with_render_pipeline_state<'descriptor, Output, Effect>(
    descriptor: &'descriptor functional::RenderPipelineDescriptor,
    effect: Effect,
) -> Result<Output, LodWebGpuError>
where
    Effect: for<'attributes> FnOnce(LoweredRenderPipelineState<'descriptor, 'attributes>) -> Output,
{
    let mut attribute_storage = Vec::with_capacity(descriptor.vertex_buffers().len());
    for (expected_slot, buffer) in descriptor.vertex_buffers().iter().enumerate() {
        if buffer.slot() != expected_slot as u32 {
            return Err(LodWebGpuError::Payload(
                "functional render-pipeline vertex slots are not contiguous".to_string(),
            ));
        }
        attribute_storage.push(
            buffer
                .attributes()
                .iter()
                .map(|attribute| wgpu::VertexAttribute {
                    format: vertex_format(attribute.format),
                    offset: attribute.offset,
                    shader_location: attribute.location,
                })
                .collect::<Vec<_>>(),
        );
    }
    let vertex_buffers = descriptor
        .vertex_buffers()
        .iter()
        .zip(attribute_storage.iter())
        .map(|(buffer, attributes)| wgpu::VertexBufferLayout {
            array_stride: buffer.stride(),
            step_mode: vertex_step_mode(buffer.step_mode()),
            attributes,
        })
        .collect::<Vec<_>>();
    let targets = descriptor
        .color_targets()
        .iter()
        .copied()
        .map(color_target_state)
        .map(|target| target.map(Some))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(effect(LoweredRenderPipelineState {
        vertex_entry_point: descriptor.program().vertex().entry_point(),
        fragment_entry_point: descriptor
            .program()
            .fragment()
            .map(functional::ShaderModuleDescriptor::entry_point),
        vertex_buffers,
        primitive: primitive_state(descriptor.primitive()),
        depth_stencil: descriptor.depth_stencil().map(depth_stencil_state),
        multisample: multisample_state(descriptor.multisample()),
        targets,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn portable_fixed_state_lowers_without_backend_defaults_changing_meaning() {
        assert_eq!(
            texture_format(functional::TextureFormat::Bgra8UnormSrgb),
            wgpu::TextureFormat::Bgra8UnormSrgb,
        );
        assert_eq!(
            functional_texture_format(wgpu::TextureFormat::Bgra8UnormSrgb),
            Some(functional::TextureFormat::Bgra8UnormSrgb),
        );
        assert_eq!(
            vertex_format(functional::VertexFormat::Float32x3),
            wgpu::VertexFormat::Float32x3,
        );
        let primitive = primitive_state(functional::PrimitiveStateDescriptor {
            topology: functional::PrimitiveTopology::TriangleStrip,
            strip_index_format: Some(functional::IndexFormat::Uint32),
            front_face: functional::FrontFace::Clockwise,
            cull_mode: functional::CullMode::Back,
        });
        assert_eq!(primitive.topology, wgpu::PrimitiveTopology::TriangleStrip);
        assert_eq!(
            primitive.strip_index_format,
            Some(wgpu::IndexFormat::Uint32)
        );
        assert_eq!(primitive.front_face, wgpu::FrontFace::Cw);
        assert_eq!(primitive.cull_mode, Some(wgpu::Face::Back));

        let target = color_target_state(functional::ColorTargetStateDescriptor {
            format: functional::TextureFormat::Rgba16Float,
            blend: None,
            write_mask: functional::ColorWriteMask::RED.union(functional::ColorWriteMask::ALPHA),
        })
        .unwrap();
        assert_eq!(target.format, wgpu::TextureFormat::Rgba16Float);
        assert_eq!(
            target.write_mask,
            wgpu::ColorWrites::RED | wgpu::ColorWrites::ALPHA,
        );

        let ignore = functional::StencilFaceStateDescriptor {
            compare: functional::CompareFunction::Always,
            fail_op: functional::StencilOperation::Keep,
            depth_fail_op: functional::StencilOperation::Keep,
            pass_op: functional::StencilOperation::Keep,
        };
        let depth = depth_stencil_state(functional::DepthStencilStateDescriptor {
            format: functional::TextureFormat::Depth24Plus,
            depth_write_enabled: false,
            depth_compare: functional::CompareFunction::LessEqual,
            stencil_front: ignore,
            stencil_back: ignore,
            stencil_read_mask: u32::MAX,
            stencil_write_mask: u32::MAX,
            depth_bias_constant: 0,
            depth_bias_slope_scale: functional::FiniteF32::new(0.0).unwrap(),
            depth_bias_clamp: functional::FiniteF32::new(0.0).unwrap(),
        });
        assert_eq!(depth.format, wgpu::TextureFormat::Depth24Plus);
        assert_eq!(depth.depth_write_enabled, Some(false));
        assert_eq!(depth.depth_compare, Some(wgpu::CompareFunction::LessEqual));
    }

    fn functional_pipeline(vertex_slot: u32) -> functional::RenderPipelineDescriptor {
        let shader = |stage, entry_point| {
            functional::ShaderModuleDescriptor::new(
                "lowering test",
                "synthetic shader source",
                "synthetic compiler revision",
                stage,
                entry_point,
                functional::ShaderTarget::Wgsl,
                Vec::new(),
            )
            .unwrap()
        };
        functional::RenderPipelineDescriptor::new(
            functional::GraphicsProgramDescriptor::new(
                shader(functional::ShaderStage::Vertex, "vertex_main"),
                Some(shader(functional::ShaderStage::Fragment, "fragment_main")),
                Vec::new(),
            )
            .unwrap(),
            functional::PipelineLayoutDescriptor::new(Vec::new()).unwrap(),
            vec![functional::VertexBufferLayoutDescriptor::new(
                vertex_slot,
                12,
                functional::VertexStepMode::Vertex,
                vec![functional::VertexAttributeDescriptor {
                    location: 3,
                    offset: 0,
                    format: functional::VertexFormat::Float32x3,
                }],
            )
            .unwrap()],
            functional::PrimitiveStateDescriptor {
                topology: functional::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: functional::FrontFace::Clockwise,
                cull_mode: functional::CullMode::Back,
            },
            None,
            vec![functional::ColorTargetStateDescriptor {
                format: functional::TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: functional::ColorWriteMask::ALL,
            }],
            functional::MultisampleStateDescriptor::default(),
        )
        .unwrap()
    }

    #[test]
    fn complete_pipeline_lowering_keeps_borrows_inside_one_effect() {
        let observation = with_render_pipeline_state(&functional_pipeline(0), |state| {
            (
                state.vertex_entry_point.to_string(),
                state.fragment_entry_point.map(str::to_string),
                state.vertex_buffers.len(),
                state.vertex_buffers[0].array_stride,
                state.vertex_buffers[0].attributes[0].shader_location,
                state.primitive.front_face,
                state.primitive.cull_mode,
                state.depth_stencil.is_none(),
                state.multisample.count,
                state.targets[0].as_ref().unwrap().format,
            )
        })
        .unwrap();
        assert_eq!(observation.0, "vertex_main");
        assert_eq!(observation.1.as_deref(), Some("fragment_main"));
        assert_eq!(observation.2, 1);
        assert_eq!(observation.3, 12);
        assert_eq!(observation.4, 3);
        assert_eq!(observation.5, wgpu::FrontFace::Cw);
        assert_eq!(observation.6, Some(wgpu::Face::Back));
        assert!(observation.7);
        assert_eq!(observation.8, 1);
        assert_eq!(observation.9, wgpu::TextureFormat::Rgba8Unorm);
    }

    #[test]
    fn complete_pipeline_lowering_rejects_noncontiguous_vertex_slots() {
        let error = with_render_pipeline_state(&functional_pipeline(2), |_| ()).unwrap_err();
        assert!(error.to_string().contains("not contiguous"));
    }
}
