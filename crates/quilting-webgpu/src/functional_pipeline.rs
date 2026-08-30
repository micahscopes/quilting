//! Shared pure constructors for WebGPU-facing functional pipeline families.
//!
//! These helpers create only `quilting_core::render_pipeline` values. They do
//! not inspect a device, allocate a handle, or own cache policy.

use quilting_core::render_pipeline as functional;

pub(crate) fn buffer_binding(
    binding: u32,
    visibility: functional::ShaderVisibility,
    uniform: bool,
    minimum_size: u64,
    dynamic_offset: bool,
) -> functional::BindGroupLayoutEntry {
    functional::BindGroupLayoutEntry {
        binding,
        visibility,
        kind: if uniform {
            functional::BindingKind::UniformBuffer {
                dynamic_offset,
                minimum_size,
            }
        } else {
            functional::BindingKind::StorageBuffer {
                read_only: true,
                dynamic_offset,
                minimum_size,
            }
        },
    }
}

pub(crate) fn pbr_environment_bindings(
    group: u32,
) -> Result<functional::BindGroupLayoutDescriptor, functional::RenderPipelineDescriptorError> {
    let fragment = functional::ShaderVisibility::FRAGMENT;
    functional::BindGroupLayoutDescriptor::new(
        group,
        vec![
            buffer_binding(0, fragment, true, 16, false),
            functional::BindGroupLayoutEntry {
                binding: 1,
                visibility: fragment,
                kind: functional::BindingKind::Texture {
                    sample_kind: functional::TextureSampleKind::FloatFilterable,
                    view_dimension: functional::TextureViewDimension::Cube,
                    multisampled: false,
                },
            },
            functional::BindGroupLayoutEntry {
                binding: 2,
                visibility: fragment,
                kind: functional::BindingKind::Texture {
                    sample_kind: functional::TextureSampleKind::FloatFilterable,
                    view_dimension: functional::TextureViewDimension::Cube,
                    multisampled: false,
                },
            },
            functional::BindGroupLayoutEntry {
                binding: 3,
                visibility: fragment,
                kind: functional::BindingKind::Sampler(functional::SamplerBindingKind::Filtering),
            },
        ],
    )
}

pub(crate) fn position_vertex_buffer(
) -> Result<functional::VertexBufferLayoutDescriptor, functional::RenderPipelineDescriptorError> {
    functional::VertexBufferLayoutDescriptor::new(
        0,
        12,
        functional::VertexStepMode::Vertex,
        vec![functional::VertexAttributeDescriptor {
            location: 0,
            offset: 0,
            format: functional::VertexFormat::Float32x3,
        }],
    )
}

pub(crate) const fn alpha_blending() -> functional::BlendStateDescriptor {
    functional::BlendStateDescriptor {
        color: functional::BlendComponentDescriptor {
            source_factor: functional::BlendFactor::SourceAlpha,
            destination_factor: functional::BlendFactor::OneMinusSourceAlpha,
            operation: functional::BlendOperation::Add,
        },
        alpha: functional::BlendComponentDescriptor {
            source_factor: functional::BlendFactor::One,
            destination_factor: functional::BlendFactor::OneMinusSourceAlpha,
            operation: functional::BlendOperation::Add,
        },
    }
}

pub(crate) fn depth_state(
    depth_format: Option<functional::TextureFormat>,
    write_enabled: bool,
) -> Result<
    Option<functional::DepthStencilStateDescriptor>,
    functional::RenderPipelineDescriptorError,
> {
    let stencil_ignore = functional::StencilFaceStateDescriptor {
        compare: functional::CompareFunction::Always,
        fail_op: functional::StencilOperation::Keep,
        depth_fail_op: functional::StencilOperation::Keep,
        pass_op: functional::StencilOperation::Keep,
    };
    depth_format
        .map(|format| {
            Ok(functional::DepthStencilStateDescriptor {
                format,
                depth_write_enabled: write_enabled,
                depth_compare: functional::CompareFunction::LessEqual,
                stencil_front: stencil_ignore,
                stencil_back: stencil_ignore,
                stencil_read_mask: u32::MAX,
                stencil_write_mask: u32::MAX,
                depth_bias_constant: 0,
                depth_bias_slope_scale: functional::FiniteF32::new(0.0)?,
                depth_bias_clamp: functional::FiniteF32::new(0.0)?,
            })
        })
        .transpose()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_pure_state_matches_the_portable_render_contract() {
        let environment = pbr_environment_bindings(2).unwrap();
        assert_eq!(environment.group(), 2);
        assert_eq!(environment.entries().len(), 4);
        let vertex = position_vertex_buffer().unwrap();
        assert_eq!(vertex.slot(), 0);
        assert_eq!(vertex.stride(), 12);
        assert_eq!(
            alpha_blending().color.source_factor,
            functional::BlendFactor::SourceAlpha,
        );
        assert!(
            depth_state(Some(functional::TextureFormat::Depth24Plus), true)
                .unwrap()
                .unwrap()
                .depth_write_enabled,
        );
        assert_eq!(depth_state(None, false).unwrap(), None);
    }
}
