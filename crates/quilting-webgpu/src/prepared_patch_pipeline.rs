//! Pure pipeline descriptions for the prepared/adaptive QB patch path.
//!
//! Runtime lowering remains device-owned, but family identity, resource
//! layout, fixed function state, and attachment shape are ordinary immutable
//! values shared with the renderer's other functional pipeline families.

use crate::functional_pipeline::{
    alpha_blending, buffer_binding, depth_state, pbr_environment_bindings, position_vertex_buffer,
};
use crate::pbr_resources::PBR_TEXTURE_CHANNELS;
use crate::{
    DRAW_BATCH_INDEX_BYTES, PACKED_RECORD_BYTES, PATCH_PBR_MATERIAL_BYTES,
    PATCH_RENDER_FRAME_BYTES, PREPARED_PATCH_RECORD_BYTES, VISIBILITY_RANGE_RECORD_BYTES,
};
use quilting_core::render::RenderStyle;
use quilting_core::render_pipeline as functional;
use std::error::Error;
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreparedPatchPipelineDescriptorError {
    UnsupportedStyle(RenderStyle),
    FocusRequiresExactlyPbr,
    Descriptor(functional::RenderPipelineDescriptorError),
}

impl fmt::Display for PreparedPatchPipelineDescriptorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedStyle(style) => write!(
                formatter,
                "WebGPU prepared-patch pipeline does not support {style:?}",
            ),
            Self::FocusRequiresExactlyPbr => {
                formatter.write_str("focus MRT pipeline creation requires exactly the PBR style")
            }
            Self::Descriptor(error) => error.fmt(formatter),
        }
    }
}

impl Error for PreparedPatchPipelineDescriptorError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Descriptor(error) => Some(error),
            _ => None,
        }
    }
}

impl From<functional::RenderPipelineDescriptorError> for PreparedPatchPipelineDescriptorError {
    fn from(error: functional::RenderPipelineDescriptorError) -> Self {
        Self::Descriptor(error)
    }
}

#[derive(Clone, Copy)]
struct PreparedPass {
    fragment_entry_point: &'static str,
    topology: functional::PrimitiveTopology,
    uses_pbr: bool,
    depth_write: bool,
}

fn style_pass(
    style: RenderStyle,
    focus: bool,
) -> Result<PreparedPass, PreparedPatchPipelineDescriptorError> {
    let (fragment_entry_point, topology, uses_pbr) = match style {
        RenderStyle::Pbr => (
            if focus {
                quilting_shaders::PATCH_RENDER_DEVICE_PBR_FOCUS_ENTRY_POINT
            } else {
                quilting_shaders::PATCH_RENDER_DEVICE_PBR_ENTRY_POINT
            },
            functional::PrimitiveTopology::TriangleList,
            true,
        ),
        RenderStyle::Normals => (
            quilting_shaders::PATCH_RENDER_DEVICE_NORMALS_ENTRY_POINT,
            functional::PrimitiveTopology::TriangleList,
            false,
        ),
        RenderStyle::Matcap => (
            quilting_shaders::PATCH_RENDER_DEVICE_MATCAP_ENTRY_POINT,
            functional::PrimitiveTopology::TriangleList,
            false,
        ),
        RenderStyle::Lod => (
            quilting_shaders::PATCH_RENDER_DEVICE_LOD_ENTRY_POINT,
            functional::PrimitiveTopology::TriangleList,
            false,
        ),
        RenderStyle::Stretch => (
            quilting_shaders::PATCH_RENDER_DEVICE_STRETCH_ENTRY_POINT,
            functional::PrimitiveTopology::TriangleList,
            false,
        ),
        RenderStyle::Wire => (
            quilting_shaders::PATCH_RENDER_DEVICE_WIRE_ENTRY_POINT,
            functional::PrimitiveTopology::LineList,
            false,
        ),
        unsupported => {
            return Err(PreparedPatchPipelineDescriptorError::UnsupportedStyle(
                unsupported,
            ));
        }
    };
    Ok(PreparedPass {
        fragment_entry_point,
        topology,
        uses_pbr,
        depth_write: true,
    })
}

fn prepared_bindings(
) -> Result<functional::BindGroupLayoutDescriptor, PreparedPatchPipelineDescriptorError> {
    let vertex = functional::ShaderVisibility::VERTEX;
    let fragment = functional::ShaderVisibility::FRAGMENT;
    let vertex_fragment = functional::ShaderVisibility::vertex_fragment();
    Ok(functional::BindGroupLayoutDescriptor::new(
        0,
        vec![
            buffer_binding(0, vertex_fragment, false, PATCH_RENDER_FRAME_BYTES, false),
            buffer_binding(1, vertex, false, PREPARED_PATCH_RECORD_BYTES, false),
            buffer_binding(2, vertex, false, PACKED_RECORD_BYTES, false),
            buffer_binding(3, vertex, false, VISIBILITY_RANGE_RECORD_BYTES, false),
            buffer_binding(4, vertex_fragment, true, DRAW_BATCH_INDEX_BYTES, true),
            buffer_binding(5, fragment, false, PATCH_PBR_MATERIAL_BYTES, false),
        ],
    )?)
}

fn pbr_texture_bindings(
) -> Result<functional::BindGroupLayoutDescriptor, PreparedPatchPipelineDescriptorError> {
    let fragment = functional::ShaderVisibility::FRAGMENT;
    let mut entries = Vec::with_capacity(PBR_TEXTURE_CHANNELS * 2);
    for channel in 0..PBR_TEXTURE_CHANNELS {
        let texture_binding = u32::try_from(channel * 2).expect("six channels fit u32");
        entries.push(functional::BindGroupLayoutEntry {
            binding: texture_binding,
            visibility: fragment,
            kind: functional::BindingKind::Texture {
                sample_kind: functional::TextureSampleKind::FloatFilterable,
                view_dimension: functional::TextureViewDimension::D2,
                multisampled: false,
            },
        });
        entries.push(functional::BindGroupLayoutEntry {
            binding: texture_binding + 1,
            visibility: fragment,
            kind: functional::BindingKind::Sampler(functional::SamplerBindingKind::Filtering),
        });
    }
    Ok(functional::BindGroupLayoutDescriptor::new(1, entries)?)
}

/// Describe exactly the prepared/adaptive patch pipeline family requested by
/// one render target. Triangle passes contribute counter-clockwise then
/// clockwise descriptors. The line pass contributes one descriptor because
/// the runtime deliberately reuses it for both winding handles.
#[allow(clippy::too_many_arguments)]
pub fn prepared_patch_pipeline_descriptors(
    styles: &[RenderStyle],
    color_format: functional::TextureFormat,
    depth_format: Option<functional::TextureFormat>,
    sample_count: u32,
    include_highlight: bool,
    pbr_raw_field_format: Option<functional::TextureFormat>,
) -> Result<Vec<functional::RenderPipelineDescriptor>, PreparedPatchPipelineDescriptorError> {
    if pbr_raw_field_format.is_some() && (include_highlight || styles != [RenderStyle::Pbr]) {
        return Err(PreparedPatchPipelineDescriptorError::FocusRequiresExactlyPbr);
    }

    let prepared_bindings = prepared_bindings()?;
    let diagnostic_layout =
        functional::PipelineLayoutDescriptor::new(vec![prepared_bindings.clone()])?;
    let pbr_layout = functional::PipelineLayoutDescriptor::new(vec![
        prepared_bindings,
        pbr_texture_bindings()?,
        pbr_environment_bindings(2)?,
    ])?;
    let shader = |stage, entry_point| {
        functional::ShaderModuleDescriptor::new(
            "quilting prepared QB render",
            quilting_shaders::sources::PATCH_RENDER_DEVICE,
            quilting_shaders::compiler_catalog_revision(),
            stage,
            entry_point,
            functional::ShaderTarget::Wgsl,
            Vec::new(),
        )
    };
    let vertex_shader = shader(
        functional::ShaderStage::Vertex,
        quilting_shaders::PATCH_RENDER_DEVICE_VERTEX_ENTRY_POINT,
    )?;
    let vertex_buffer = position_vertex_buffer()?;
    let alpha_blend = alpha_blending();
    let focus = pbr_raw_field_format.is_some();
    let mut passes = styles
        .iter()
        .copied()
        .map(|style| style_pass(style, focus))
        .collect::<Result<Vec<_>, _>>()?;
    if include_highlight {
        passes.push(PreparedPass {
            fragment_entry_point: quilting_shaders::PATCH_RENDER_DEVICE_HIGHLIGHT_ENTRY_POINT,
            topology: functional::PrimitiveTopology::TriangleList,
            uses_pbr: false,
            depth_write: false,
        });
    }

    let mut descriptors = Vec::with_capacity(passes.len() * 2);
    for pass in passes {
        let front_faces: &[_] = match pass.topology {
            functional::PrimitiveTopology::LineList => &[functional::FrontFace::CounterClockwise],
            _ => &[
                functional::FrontFace::CounterClockwise,
                functional::FrontFace::Clockwise,
            ],
        };
        for &front_face in front_faces {
            let mut color_targets = vec![functional::ColorTargetStateDescriptor {
                format: color_format,
                blend: Some(alpha_blend),
                write_mask: functional::ColorWriteMask::ALL,
            }];
            if let Some(format) = pbr_raw_field_format.filter(|_| pass.uses_pbr) {
                color_targets.push(functional::ColorTargetStateDescriptor {
                    format,
                    blend: None,
                    write_mask: functional::ColorWriteMask::ALL,
                });
            }
            descriptors.push(functional::RenderPipelineDescriptor::new(
                functional::GraphicsProgramDescriptor::new(
                    vertex_shader.clone(),
                    Some(shader(
                        functional::ShaderStage::Fragment,
                        pass.fragment_entry_point,
                    )?),
                    Vec::new(),
                )?,
                if pass.uses_pbr {
                    pbr_layout.clone()
                } else {
                    diagnostic_layout.clone()
                },
                vec![vertex_buffer.clone()],
                functional::PrimitiveStateDescriptor {
                    topology: pass.topology,
                    strip_index_format: None,
                    front_face,
                    cull_mode: functional::CullMode::None,
                },
                depth_state(depth_format, pass.depth_write)?,
                color_targets,
                functional::MultisampleStateDescriptor {
                    count: sample_count,
                    ..Default::default()
                },
            )?);
        }
    }
    Ok(descriptors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use quilting_core::render_memo::DeviceMemo;

    const STYLES: &[RenderStyle] = &[
        RenderStyle::Pbr,
        RenderStyle::Normals,
        RenderStyle::Matcap,
        RenderStyle::Lod,
        RenderStyle::Stretch,
        RenderStyle::Wire,
    ];

    fn descriptors(
        color_format: functional::TextureFormat,
    ) -> Vec<functional::RenderPipelineDescriptor> {
        prepared_patch_pipeline_descriptors(
            STYLES,
            color_format,
            Some(functional::TextureFormat::Depth24Plus),
            4,
            true,
            None,
        )
        .unwrap()
    }

    #[test]
    fn complete_family_preserves_layout_winding_and_attachment_contracts() {
        let descriptors = descriptors(functional::TextureFormat::Bgra8UnormSrgb);
        assert_eq!(descriptors.len(), 13);

        let pbr = &descriptors[0];
        assert_eq!(pbr.layout().groups().len(), 3);
        assert_eq!(pbr.layout().groups()[1].entries().len(), 12);
        assert_eq!(descriptors[2].layout().groups().len(), 1);
        assert_eq!(
            pbr.layout().groups()[0],
            descriptors[2].layout().groups()[0],
        );

        let wire = descriptors
            .iter()
            .filter(|descriptor| {
                descriptor.program().fragment().unwrap().entry_point()
                    == quilting_shaders::PATCH_RENDER_DEVICE_WIRE_ENTRY_POINT
            })
            .collect::<Vec<_>>();
        assert_eq!(wire.len(), 1);
        assert_eq!(
            wire[0].primitive().topology,
            functional::PrimitiveTopology::LineList,
        );

        let highlights = descriptors
            .iter()
            .filter(|descriptor| {
                descriptor.program().fragment().unwrap().entry_point()
                    == quilting_shaders::PATCH_RENDER_DEVICE_HIGHLIGHT_ENTRY_POINT
            })
            .collect::<Vec<_>>();
        assert_eq!(highlights.len(), 2);
        assert!(highlights
            .iter()
            .all(|descriptor| !descriptor.depth_stencil().unwrap().depth_write_enabled));
    }

    #[test]
    fn focus_family_is_only_pbr_and_names_both_targets() {
        let focus = prepared_patch_pipeline_descriptors(
            &[RenderStyle::Pbr],
            functional::TextureFormat::Rgba8Unorm,
            Some(functional::TextureFormat::Depth24Plus),
            1,
            false,
            Some(functional::TextureFormat::Rgba16Float),
        )
        .unwrap();
        assert_eq!(focus.len(), 2);
        assert!(focus
            .iter()
            .all(|descriptor| descriptor.color_targets().len() == 2));
        assert_eq!(
            prepared_patch_pipeline_descriptors(
                &[RenderStyle::Pbr, RenderStyle::Normals],
                functional::TextureFormat::Rgba8Unorm,
                None,
                1,
                false,
                Some(functional::TextureFormat::Rgba16Float),
            ),
            Err(PreparedPatchPipelineDescriptorError::FocusRequiresExactlyPbr),
        );
    }

    #[test]
    fn invalid_styles_and_multisampling_fail_before_backend_effects() {
        assert_eq!(
            prepared_patch_pipeline_descriptors(
                &[RenderStyle::MatcapWire],
                functional::TextureFormat::Rgba8Unorm,
                None,
                1,
                false,
                None,
            ),
            Err(PreparedPatchPipelineDescriptorError::UnsupportedStyle(
                RenderStyle::MatcapWire,
            )),
        );
        assert!(matches!(
            prepared_patch_pipeline_descriptors(
                &[RenderStyle::Normals],
                functional::TextureFormat::Rgba8Unorm,
                None,
                3,
                false,
                None,
            ),
            Err(PreparedPatchPipelineDescriptorError::Descriptor(
                functional::RenderPipelineDescriptorError::InvalidMultisampleCount,
            )),
        ));
    }

    #[test]
    fn exact_functional_families_are_device_epoch_memo_keys() {
        let rgba_descriptors = descriptors(functional::TextureFormat::Rgba8Unorm);
        let mut memo = DeviceMemo::new(4);
        memo.get_or_try_insert_with(rgba_descriptors.clone(), |_| Ok::<_, ()>(7))
            .unwrap();
        assert_eq!(
            *memo
                .get_or_try_insert_with(rgba_descriptors, |_| Ok::<_, ()>(8))
                .unwrap(),
            7,
        );
        memo.get_or_try_insert_with(
            descriptors(functional::TextureFormat::Bgra8UnormSrgb),
            |_| Ok::<_, ()>(9),
        )
        .unwrap();
        assert_eq!(memo.diagnostics().hits, 1);
        assert_eq!(memo.diagnostics().misses, 2);
        assert_eq!(memo.diagnostics().resident_entries, 2);
    }
}
