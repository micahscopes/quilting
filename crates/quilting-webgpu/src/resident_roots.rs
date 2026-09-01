//! Retained root-only geometry planning and direct source-face preparation.
//!
//! Adaptive dyadic leaves remain a separate sparse overlay because their edge
//! LOD and permutation cannot be reconstructed from one source-face record.

use super::*;
use quilting_core::render_pipeline as functional;

/// Diagnostic styles whose fragment inputs are unconditionally complete in
/// the direct source-root path. PBR has an additional resource-residency gate.
pub fn supports_resident_root_render_style(style: RenderStyle) -> bool {
    style != RenderStyle::Pbr
}

/// Whether one extracted scene can be rendered exactly from its source roots
/// without consulting the CPU-authored batch topology. Adaptive leaves remain
/// on their sparse overlay path until their partition is device-resident too.
pub fn supports_resident_root_render_scene(scene: &RenderSceneSnapshot, face_count: usize) -> bool {
    scene.suppressed_root_faces.is_empty()
        && !scene.batches.iter().any(|batch| {
            batch.id.layer == RenderBatchLayer::AdaptiveOverlay
                || batch.members.is_empty()
                || batch
                    .members
                    .iter()
                    .any(|member| member.leaf_id != ScreenPatchLeafId::ROOT)
        })
        && matches!(resident_root_render_domains(scene, face_count), Ok(Some(_)))
}

/// Extract the exact LOD-independent root state used by both root-only and
/// root-plus-adaptive rendering. Sparse adaptive members are validated with
/// the scene but do not create root domains. `None` remains a supported
/// fallback for future scene forms without a complete source-root layer.
pub fn resident_root_render_domains(
    scene: &RenderSceneSnapshot,
    face_count: usize,
) -> Result<Option<ResidentRootDrawDomains>, String> {
    if scene.batches.iter().any(|batch| batch.members.is_empty()) {
        return Ok(None);
    }
    ResidentRootDrawDomains::build(scene, face_count)
        .map(Some)
        .map_err(|error| error.to_string())
}

/// Retained root-only geometry buckets derived from packed resident LOD.
pub struct ResidentGeometryBucketScene {
    pub(super) model_identity: u64,
    pub(super) domain_identity: u64,
    pub(super) face_count: u32,
    pub(super) atlas_count: u32,
    pub(super) bucket_count: u32,
    pub(super) chunk_count: u32,
    pub(super) eligibility_word_count: u32,
    pub(super) uniform: wgpu::Buffer,
    pub(super) root_eligibility: wgpu::Buffer,
    pub(super) root_visibility: wgpu::Buffer,
    pub(super) suppressed_faces: Mutex<Vec<u32>>,
    pub(super) _chunk_counts: wgpu::Buffer,
    pub(super) _chunk_offsets: wgpu::Buffer,
    pub(super) _bucket_counts: wgpu::Buffer,
    pub(super) compacted_faces: wgpu::Buffer,
    pub(super) bucket_ranges: wgpu::Buffer,
    pub(super) triangle_indirect_arguments: wgpu::Buffer,
    pub(super) line_indirect_arguments: wgpu::Buffer,
    pub(super) histogram_bind_group: wgpu::BindGroup,
    pub(super) prefix_bind_group: wgpu::BindGroup,
    pub(super) scan_bind_group: wgpu::BindGroup,
    pub(super) scatter_bind_group: wgpu::BindGroup,
}

/// Diagnostic projection of the retained root geometry plan. Production draw
/// execution consumes the same buffers directly and never constructs this.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentGeometryBucketOutput {
    pub compacted_faces: Vec<u32>,
    pub bucket_ranges: Vec<[u32; 5]>,
    pub triangle_indirect_arguments: Vec<[u32; 5]>,
    pub line_indirect_arguments: Vec<[u32; 5]>,
}

/// Source-face-indexed root topology reconstructed from packed resident LOD.
/// The compacted bucket plan can therefore pull prepared records by face ID;
/// no second topology scatter or CPU batch expansion is required.
pub struct ResidentRootTopologyScene {
    pub(super) model_identity: u64,
    pub(super) face_count: u32,
    pub(super) vertex_count: u32,
    pub(super) subject_count: u32,
    pub(super) _uniform: wgpu::Buffer,
    pub(super) face_subject_rows: wgpu::Buffer,
    pub(super) _vertex_lod_max: wgpu::Buffer,
    pub(super) topology_records: wgpu::Buffer,
    pub(super) clear_bind_group: wgpu::BindGroup,
    pub(super) accumulate_bind_group: wgpu::BindGroup,
    pub(super) emit_bind_group: wgpu::BindGroup,
}

/// Device-resident material/render indirection for source roots. The domain
/// table contains only observed scene domains; current LOD never changes a
/// face's row.
pub struct ResidentRootDrawDomainScene {
    pub(super) model_identity: u64,
    pub(super) domain_identity: u64,
    pub(super) face_count: u32,
    pub(super) domain_count: u32,
    pub(super) domains: Vec<ResidentRootDrawDomain>,
    pub(super) face_domain_rows: wgpu::Buffer,
    pub(super) domain_records: wgpu::Buffer,
}

/// Diagnostic copy of the exact two storage tables retained for root draws.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentRootDrawDomainOutput {
    pub face_domain_rows: Vec<u32>,
    /// `(material, render_node, pbr_class, flags)`; flags bit zero is enabled
    /// and bit one marks an orientation-reversing entity transform.
    pub domain_records: Vec<[u32; 4]>,
}

/// Graphics pipelines that pull source-indexed prepared roots through the
/// device-generated atlas/parity bucket plan.
#[derive(Clone)]
pub struct ResidentRootRenderPipeline {
    pub(super) color_format: wgpu::TextureFormat,
    pub(super) sample_count: u32,
    pub(super) bind_group_layout: wgpu::BindGroupLayout,
    pbr_portable_atlas_bind_group_layout: wgpu::BindGroupLayout,
    pbr_environment_bind_group_layout: wgpu::BindGroupLayout,
    pbr: ResidentRootWindingPipelines,
    pbr_focus: ResidentRootWindingPipelines,
    matcap: ResidentRootWindingPipelines,
    normals: ResidentRootWindingPipelines,
    lod: ResidentRootWindingPipelines,
    stretch: ResidentRootWindingPipelines,
    wire: ResidentRootWindingPipelines,
    highlight: ResidentRootWindingPipelines,
}

#[derive(Clone)]
struct ResidentRootWindingPipelines {
    counter_clockwise: wgpu::RenderPipeline,
    clockwise: wgpu::RenderPipeline,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResidentRootPipelineKind {
    Pbr,
    PbrFocus,
    Matcap,
    Normals,
    Lod,
    Stretch,
    Wire,
    Highlight,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ResidentRootPipelinePass {
    kind: ResidentRootPipelineKind,
    label: &'static str,
    vertex_entry_point: &'static str,
    fragment_entry_point: &'static str,
    geometry: RenderGeometry,
    depth_write: bool,
    focus_mrt: bool,
    uses_pbr: bool,
}

impl ResidentRootPipelinePass {
    const fn topology(self) -> functional::PrimitiveTopology {
        match self.geometry {
            RenderGeometry::Triangles => functional::PrimitiveTopology::TriangleList,
            RenderGeometry::Lines => functional::PrimitiveTopology::LineList,
        }
    }
}

const RESIDENT_ROOT_PIPELINE_PASSES: [ResidentRootPipelinePass; 8] = [
    ResidentRootPipelinePass {
        kind: ResidentRootPipelineKind::Pbr,
        label: "quilting resident root PBR",
        vertex_entry_point: quilting_shaders::RESIDENT_ROOT_RENDER_DEVICE_VERTEX_ENTRY_POINT,
        fragment_entry_point: quilting_shaders::RESIDENT_ROOT_RENDER_DEVICE_PBR_ENTRY_POINT,
        geometry: RenderGeometry::Triangles,
        depth_write: true,
        focus_mrt: false,
        uses_pbr: true,
    },
    ResidentRootPipelinePass {
        kind: ResidentRootPipelineKind::PbrFocus,
        label: "quilting resident root focus PBR",
        vertex_entry_point: quilting_shaders::RESIDENT_ROOT_RENDER_DEVICE_VERTEX_ENTRY_POINT,
        fragment_entry_point: quilting_shaders::RESIDENT_ROOT_RENDER_DEVICE_PBR_FOCUS_ENTRY_POINT,
        geometry: RenderGeometry::Triangles,
        depth_write: true,
        focus_mrt: true,
        uses_pbr: true,
    },
    ResidentRootPipelinePass {
        kind: ResidentRootPipelineKind::Matcap,
        label: "quilting resident root matcap",
        vertex_entry_point: quilting_shaders::RESIDENT_ROOT_RENDER_DEVICE_VERTEX_ENTRY_POINT,
        fragment_entry_point: quilting_shaders::RESIDENT_ROOT_RENDER_DEVICE_MATCAP_ENTRY_POINT,
        geometry: RenderGeometry::Triangles,
        depth_write: true,
        focus_mrt: false,
        uses_pbr: false,
    },
    ResidentRootPipelinePass {
        kind: ResidentRootPipelineKind::Normals,
        label: "quilting resident root normals",
        vertex_entry_point: quilting_shaders::RESIDENT_ROOT_RENDER_DEVICE_VERTEX_ENTRY_POINT,
        fragment_entry_point: quilting_shaders::RESIDENT_ROOT_RENDER_DEVICE_NORMALS_ENTRY_POINT,
        geometry: RenderGeometry::Triangles,
        depth_write: true,
        focus_mrt: false,
        uses_pbr: false,
    },
    ResidentRootPipelinePass {
        kind: ResidentRootPipelineKind::Lod,
        label: "quilting resident root LOD",
        vertex_entry_point: quilting_shaders::RESIDENT_ROOT_RENDER_DEVICE_VERTEX_ENTRY_POINT,
        fragment_entry_point: quilting_shaders::RESIDENT_ROOT_RENDER_DEVICE_LOD_ENTRY_POINT,
        geometry: RenderGeometry::Triangles,
        depth_write: true,
        focus_mrt: false,
        uses_pbr: false,
    },
    ResidentRootPipelinePass {
        kind: ResidentRootPipelineKind::Stretch,
        label: "quilting resident root stretch",
        vertex_entry_point: quilting_shaders::RESIDENT_ROOT_RENDER_DEVICE_VERTEX_ENTRY_POINT,
        fragment_entry_point: quilting_shaders::RESIDENT_ROOT_RENDER_DEVICE_STRETCH_ENTRY_POINT,
        geometry: RenderGeometry::Triangles,
        depth_write: true,
        focus_mrt: false,
        uses_pbr: false,
    },
    ResidentRootPipelinePass {
        kind: ResidentRootPipelineKind::Wire,
        label: "quilting resident root wire",
        vertex_entry_point: quilting_shaders::RESIDENT_ROOT_RENDER_DEVICE_WIRE_VERTEX_ENTRY_POINT,
        fragment_entry_point: quilting_shaders::RESIDENT_ROOT_RENDER_DEVICE_WIRE_ENTRY_POINT,
        geometry: RenderGeometry::Lines,
        depth_write: true,
        focus_mrt: false,
        uses_pbr: false,
    },
    ResidentRootPipelinePass {
        kind: ResidentRootPipelineKind::Highlight,
        label: "quilting resident root highlight",
        vertex_entry_point: quilting_shaders::RESIDENT_ROOT_RENDER_DEVICE_VERTEX_ENTRY_POINT,
        fragment_entry_point: quilting_shaders::RESIDENT_ROOT_RENDER_DEVICE_HIGHLIGHT_ENTRY_POINT,
        geometry: RenderGeometry::Triangles,
        depth_write: false,
        focus_mrt: false,
        uses_pbr: false,
    },
];

/// Pure pipeline family for direct source-root rendering. The returned order
/// is stable: each semantic pass contributes counter-clockwise then clockwise
/// state, including line rendering where winding does not affect rasterization.
pub fn resident_root_pipeline_descriptors(
    color_format: functional::TextureFormat,
    depth_format: Option<functional::TextureFormat>,
    sample_count: u32,
) -> Result<Vec<functional::RenderPipelineDescriptor>, functional::RenderPipelineDescriptorError> {
    use crate::functional_pipeline::{
        alpha_blending, buffer_binding, depth_state, pbr_environment_bindings,
        position_vertex_buffer,
    };

    let vertex = functional::ShaderVisibility::VERTEX;
    let fragment = functional::ShaderVisibility::FRAGMENT;
    let vertex_fragment = functional::ShaderVisibility::vertex_fragment();
    let root_bindings = functional::BindGroupLayoutDescriptor::new(
        0,
        vec![
            buffer_binding(0, vertex_fragment, false, PATCH_RENDER_GLOBAL_BYTES, false),
            buffer_binding(1, vertex_fragment, false, PATCH_RENDER_DOMAIN_BYTES, false),
            buffer_binding(2, vertex, false, PREPARED_PATCH_RECORD_BYTES, false),
            buffer_binding(3, vertex, false, PACKED_RECORD_BYTES, false),
            buffer_binding(4, vertex, false, RESIDENT_BUCKET_RANGE_RECORD_BYTES, false),
            buffer_binding(5, vertex, true, DRAW_BATCH_INDEX_BYTES, true),
            buffer_binding(6, vertex_fragment, false, PACKED_RECORD_BYTES, false),
            buffer_binding(7, vertex_fragment, false, 16, false),
            buffer_binding(8, fragment, false, PATCH_PBR_MATERIAL_BYTES, false),
        ],
    )?;
    let portable_atlas_bindings = functional::BindGroupLayoutDescriptor::new(
        1,
        vec![
            functional::BindGroupLayoutEntry {
                binding: 0,
                visibility: fragment,
                kind: functional::BindingKind::Texture {
                    sample_kind: functional::TextureSampleKind::FloatUnfilterable,
                    view_dimension: functional::TextureViewDimension::D2Array,
                    multisampled: false,
                },
            },
            buffer_binding(1, fragment, false, 32, false),
            buffer_binding(2, fragment, false, 32, false),
            buffer_binding(3, fragment, false, 16, false),
        ],
    )?;
    let environment_bindings = pbr_environment_bindings(2)?;
    let diagnostic_layout = functional::PipelineLayoutDescriptor::new(vec![root_bindings.clone()])?;
    let pbr_layout = functional::PipelineLayoutDescriptor::new(vec![
        root_bindings,
        portable_atlas_bindings,
        environment_bindings,
    ])?;
    let shader = |stage, entry_point| {
        functional::ShaderModuleDescriptor::new(
            "quilting resident root render",
            quilting_shaders::sources::RESIDENT_ROOT_RENDER_DEVICE,
            quilting_shaders::compiler_catalog_revision(),
            stage,
            entry_point,
            functional::ShaderTarget::Wgsl,
            Vec::new(),
        )
    };
    let vertex_buffer = position_vertex_buffer()?;
    let alpha_blend = alpha_blending();
    let mut descriptors = Vec::with_capacity(RESIDENT_ROOT_PIPELINE_PASSES.len() * 2);
    for pass in RESIDENT_ROOT_PIPELINE_PASSES {
        for front_face in [
            functional::FrontFace::CounterClockwise,
            functional::FrontFace::Clockwise,
        ] {
            let mut color_targets = vec![functional::ColorTargetStateDescriptor {
                format: color_format,
                blend: Some(alpha_blend),
                write_mask: functional::ColorWriteMask::ALL,
            }];
            if pass.focus_mrt {
                color_targets.push(functional::ColorTargetStateDescriptor {
                    format: functional::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: functional::ColorWriteMask::ALL,
                });
            }
            descriptors.push(functional::RenderPipelineDescriptor::new(
                functional::GraphicsProgramDescriptor::new(
                    shader(
                        functional::ShaderStage::Vertex,
                        pass.vertex_entry_point,
                    )?,
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
                    topology: pass.topology(),
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
mod resident_pipeline_descriptor_tests {
    use super::*;

    #[test]
    fn one_semantic_pass_plan_drives_descriptor_and_runtime_families() {
        assert_eq!(
            RESIDENT_ROOT_PIPELINE_PASSES
                .iter()
                .map(|pass| pass.kind)
                .collect::<Vec<_>>(),
            vec![
                ResidentRootPipelineKind::Pbr,
                ResidentRootPipelineKind::PbrFocus,
                ResidentRootPipelineKind::Matcap,
                ResidentRootPipelineKind::Normals,
                ResidentRootPipelineKind::Lod,
                ResidentRootPipelineKind::Stretch,
                ResidentRootPipelineKind::Wire,
                ResidentRootPipelineKind::Highlight,
            ],
        );
        assert!(RESIDENT_ROOT_PIPELINE_PASSES[0].uses_pbr);
        assert!(RESIDENT_ROOT_PIPELINE_PASSES[1].uses_pbr);
        assert!(RESIDENT_ROOT_PIPELINE_PASSES[1].focus_mrt);
        assert_eq!(
            RESIDENT_ROOT_PIPELINE_PASSES[6].geometry,
            RenderGeometry::Lines,
        );
        assert_eq!(
            RESIDENT_ROOT_PIPELINE_PASSES[6].vertex_entry_point,
            quilting_shaders::RESIDENT_ROOT_RENDER_DEVICE_WIRE_VERTEX_ENTRY_POINT,
        );
        assert!(!RESIDENT_ROOT_PIPELINE_PASSES[7].depth_write);
        assert_eq!(RESIDENT_ROOT_PIPELINE_PASSES.len() * 2, 16);
    }

    #[test]
    fn functional_family_covers_every_pass_winding_layout_and_attachment() {
        let descriptors = resident_root_pipeline_descriptors(
            functional::TextureFormat::Bgra8UnormSrgb,
            Some(functional::TextureFormat::Depth24Plus),
            4,
        )
        .unwrap();
        assert_eq!(descriptors.len(), 16);
        for pair in descriptors.chunks_exact(2) {
            assert_eq!(
                pair[0].primitive().front_face,
                functional::FrontFace::CounterClockwise,
            );
            assert_eq!(
                pair[1].primitive().front_face,
                functional::FrontFace::Clockwise,
            );
            assert_eq!(pair[0].multisample().count, 4);
            assert_eq!(pair[1].multisample().count, 4);
        }
        assert_eq!(descriptors[0].layout().groups().len(), 3);
        assert_eq!(descriptors[4].layout().groups().len(), 1);
        let root_entries = descriptors[0].layout().groups()[0].entries();
        assert_eq!(root_entries.len(), 9);
        assert_eq!(
            root_entries[0].kind,
            functional::BindingKind::StorageBuffer {
                read_only: true,
                dynamic_offset: false,
                minimum_size: PATCH_RENDER_GLOBAL_BYTES,
            },
        );
        assert_eq!(
            root_entries[1].kind,
            functional::BindingKind::StorageBuffer {
                read_only: true,
                dynamic_offset: false,
                minimum_size: PATCH_RENDER_DOMAIN_BYTES,
            },
        );
        assert_eq!(
            descriptors[0].layout().groups()[0],
            descriptors[4].layout().groups()[0],
        );
        assert_eq!(descriptors[0].vertex_buffers().len(), 1);
        assert_eq!(descriptors[0].vertex_buffers()[0].stride(), 12);
        assert_eq!(descriptors[2].color_targets().len(), 2);
        assert_eq!(
            descriptors[2].color_targets()[1].format,
            functional::TextureFormat::Rgba16Float,
        );
        assert_eq!(descriptors[0].color_targets().len(), 1);
        assert_eq!(
            descriptors[12].primitive().topology,
            functional::PrimitiveTopology::LineList,
        );
        assert_eq!(
            descriptors[12].program().vertex().entry_point(),
            quilting_shaders::RESIDENT_ROOT_RENDER_DEVICE_WIRE_VERTEX_ENTRY_POINT,
        );
        assert!(descriptors[12].depth_stencil().unwrap().depth_write_enabled);
        assert!(!descriptors[14].depth_stencil().unwrap().depth_write_enabled);
        assert_eq!(
            descriptors[14].program().fragment().unwrap().entry_point(),
            quilting_shaders::RESIDENT_ROOT_RENDER_DEVICE_HIGHLIGHT_ENTRY_POINT,
        );

        let rgba = resident_root_pipeline_descriptors(
            functional::TextureFormat::Rgba8Unorm,
            Some(functional::TextureFormat::Depth24Plus),
            4,
        )
        .unwrap();
        assert_ne!(descriptors, rgba);
        let mut memo = quilting_core::render_memo::DeviceMemo::new(0);
        memo.get_or_try_insert_with(descriptors.clone(), |_| Ok::<_, ()>(3))
            .unwrap();
        assert_eq!(
            *memo
                .get_or_try_insert_with(descriptors, |_| Ok::<_, ()>(4))
                .unwrap(),
            3,
        );
        assert_eq!(memo.diagnostics().hits, 1);
        assert_eq!(memo.diagnostics().misses, 1);
        assert_eq!(
            resident_root_pipeline_descriptors(functional::TextureFormat::Rgba8Unorm, None, 3,),
            Err(functional::RenderPipelineDescriptorError::InvalidMultisampleCount),
        );
    }
}

/// Retained per-domain frames and source-face indirection for root rendering.
pub struct ResidentRootRenderBindings {
    pub(super) model_identity: u64,
    pub(super) domain_identity: u64,
    domain_count: u32,
    pub(super) bucket_count: u32,
    pub(super) bucket_index_uniform_stride: u32,
    pub(super) global_frame: Arc<PatchRenderGlobalResidency>,
    render_domains: wgpu::Buffer,
    _materials: wgpu::Buffer,
    portable_material_textures: Option<pbr_resources::PbrPortableAtlasBindings>,
    pbr_environment: Option<PbrEnvironmentBindings>,
    pbr_scene_supported: bool,
    render_domain_table: Mutex<RetainedFrameTable>,
    _bucket_index_uniform: wgpu::Buffer,
    visibility_bind_group: wgpu::BindGroup,
    pub(super) bind_group: wgpu::BindGroup,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidentRootFrameEncoding {
    pub indirect_draw_calls: u32,
    pub source_face_count: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FocusResidentRootFrameEncoding {
    pub roots: ResidentRootFrameEncoding,
    pub postprocess: FocusPostprocessEncoding,
}

/// Atomic retained aggregate for direct source-indexed root preparation.
/// Topology emission and the ordinary rational-QB preparation pass share the
/// same device buffer; no CPU topology upload exists in this path.
pub struct ResidentRootPreparationScene {
    pub(super) topology: ResidentRootTopologyScene,
    pub(super) patches: PatchPreparationScene,
    pub(super) draw_domains: ResidentRootDrawDomainScene,
}

impl ResidentRootRenderBindings {
    pub fn pbr_texture_residency(&self) -> Option<&[PbrMaterialTextureResidency]> {
        self.portable_material_textures
            .as_ref()
            .map(pbr_resources::PbrPortableAtlasBindings::residency)
    }

    pub fn pbr_environment_bindings(&self) -> Option<&PbrEnvironmentBindings> {
        self.pbr_environment.as_ref()
    }

    /// This narrower diagnostic distinguishes factor-only scenes from the
    /// general portable-atlas path admitted by [`Self::supports_resident_basic_pbr`].
    pub fn supports_resident_untextured_pbr(&self) -> bool {
        self.pbr_scene_supported
            && self.pbr_texture_residency().is_some_and(|residency| {
                residency
                    .iter()
                    .all(|material| material.referenced_mask() == 0)
            })
            && self
                .pbr_environment_bindings()
                .is_some_and(PbrEnvironmentBindings::is_resident)
    }

    pub fn supports_resident_basic_pbr(&self) -> bool {
        self.pbr_scene_supported
            && self.pbr_texture_residency().is_some_and(|residency| {
                residency
                    .iter()
                    .all(|material| material.unresolved_mask() == 0)
            })
            && self
                .pbr_environment_bindings()
                .is_some_and(PbrEnvironmentBindings::is_resident)
    }

    /// Decide whether this coherent binding epoch can execute one frame from
    /// device-resident roots without semantic lowering. Diagnostic styles can
    /// composite their sparse overlay; PBR additionally requires the caller
    /// to prove that its absent or retained adaptive layer has exact material
    /// and environment bindings.
    pub fn supports_resident_root_frame(
        &self,
        style: RenderStyle,
        adaptive_layer_supported: bool,
    ) -> bool {
        supports_resident_root_render_style(style)
            || (style == RenderStyle::Pbr
                && adaptive_layer_supported
                && self.supports_resident_basic_pbr())
    }
}

impl ResidentGeometryBucketScene {
    pub fn face_count(&self) -> u32 {
        self.face_count
    }

    pub fn atlas_count(&self) -> u32 {
        self.atlas_count
    }

    pub fn bucket_count(&self) -> u32 {
        self.bucket_count
    }

    /// Future GPU adaptive partitioning can update this packed inclusion field
    /// directly instead of publishing suppressed source faces through the CPU.
    pub fn root_eligibility_buffer(&self) -> &wgpu::Buffer {
        &self.root_eligibility
    }

    /// Current camera-dependent visibility words. The resident visibility
    /// compute pass overwrites this buffer before bucket compaction.
    pub fn root_visibility_buffer(&self) -> &wgpu::Buffer {
        &self.root_visibility
    }

    pub fn compacted_faces_buffer(&self) -> &wgpu::Buffer {
        &self.compacted_faces
    }

    pub fn bucket_ranges_buffer(&self) -> &wgpu::Buffer {
        &self.bucket_ranges
    }

    pub fn triangle_indirect_arguments_buffer(&self) -> &wgpu::Buffer {
        &self.triangle_indirect_arguments
    }

    pub fn line_indirect_arguments_buffer(&self) -> &wgpu::Buffer {
        &self.line_indirect_arguments
    }
}

impl ResidentRootTopologyScene {
    pub fn face_count(&self) -> u32 {
        self.face_count
    }

    pub fn subject_count(&self) -> u32 {
        self.subject_count
    }

    /// Exact 48-byte `PatchTopologyRecord`s in source-face order. The next
    /// stage binds this buffer directly to patch preparation.
    pub fn topology_records_buffer(&self) -> &wgpu::Buffer {
        &self.topology_records
    }
}

impl ResidentRootPreparationScene {
    pub fn topology(&self) -> &ResidentRootTopologyScene {
        &self.topology
    }

    pub fn patches(&self) -> &PatchPreparationScene {
        &self.patches
    }

    pub fn draw_domains(&self) -> &ResidentRootDrawDomainScene {
        &self.draw_domains
    }
}

impl ResidentRootDrawDomainScene {
    pub fn face_count(&self) -> u32 {
        self.face_count
    }

    pub fn domain_count(&self) -> u32 {
        self.domain_count
    }

    pub fn domains(&self) -> &[ResidentRootDrawDomain] {
        &self.domains
    }

    pub fn face_domain_rows_buffer(&self) -> &wgpu::Buffer {
        &self.face_domain_rows
    }

    pub fn domain_records_buffer(&self) -> &wgpu::Buffer {
        &self.domain_records
    }
}

impl ResidentRootRenderBindings {
    pub fn domain_count(&self) -> u32 {
        self.domain_count
    }

    pub fn bucket_count(&self) -> u32 {
        self.bucket_count
    }
}

impl ResidentRootRenderPipeline {
    fn for_pass(&self, pass: RenderPass) -> Option<&ResidentRootWindingPipelines> {
        match pass {
            RenderPass::PbrOpaque => Some(&self.pbr),
            RenderPass::Matcap => Some(&self.matcap),
            RenderPass::Normals => Some(&self.normals),
            RenderPass::Lod => Some(&self.lod),
            RenderPass::Stretch => Some(&self.stretch),
            RenderPass::Wire => Some(&self.wire),
            _ => None,
        }
    }
}

impl LodClassifierDevice {
    /// Create the fixed-format resident-root pipeline used by the headless
    /// browser parity target without exposing backend texture enums upstream.
    pub fn create_offscreen_resident_root_render_pipeline(
        &self,
    ) -> Result<ResidentRootRenderPipeline, LodWebGpuError> {
        self.create_resident_root_render_pipeline(
            wgpu::TextureFormat::Rgba8Unorm,
            Some(wgpu::TextureFormat::Depth24Plus),
            1,
        )
    }

    pub fn create_resident_root_render_pipeline(
        &self,
        color_format: wgpu::TextureFormat,
        depth_format: Option<wgpu::TextureFormat>,
        sample_count: u32,
    ) -> Result<ResidentRootRenderPipeline, LodWebGpuError> {
        if sample_count == 0 {
            return Err(LodWebGpuError::Payload(
                "resident root render sample count must be nonzero".to_string(),
            ));
        }
        let Some(functional_color_format) =
            crate::pipeline_lowering::functional_texture_format(color_format)
        else {
            return self.build_resident_root_render_pipeline(
                color_format,
                depth_format,
                sample_count,
                None,
            );
        };
        let functional_depth_format = match depth_format {
            Some(format) => {
                let Some(format) = crate::pipeline_lowering::functional_texture_format(format)
                else {
                    return self.build_resident_root_render_pipeline(
                        color_format,
                        depth_format,
                        sample_count,
                        None,
                    );
                };
                Some(format)
            }
            None => None,
        };
        let descriptors = resident_root_pipeline_descriptors(
            functional_color_format,
            functional_depth_format,
            sample_count,
        )
        .map_err(|error| LodWebGpuError::Payload(error.to_string()))?;
        let mut pipelines = self.resident_root_render_pipelines.lock().map_err(|_| {
            LodWebGpuError::Payload(
                "WebGPU resident-root render pipeline memo was poisoned".to_string(),
            )
        })?;
        let pipeline = pipelines.get_or_try_insert_with(descriptors, |descriptors| {
            self.build_resident_root_render_pipeline(
                color_format,
                depth_format,
                sample_count,
                Some(descriptors.as_slice()),
            )
        })?;
        Ok(pipeline.clone())
    }

    fn build_resident_root_render_pipeline(
        &self,
        color_format: wgpu::TextureFormat,
        depth_format: Option<wgpu::TextureFormat>,
        sample_count: u32,
        descriptors: Option<&[functional::RenderPipelineDescriptor]>,
    ) -> Result<ResidentRootRenderPipeline, LodWebGpuError> {
        let module = self.memoized_render_shader_module(
            "quilting resident root render",
            quilting_shaders::sources::RESIDENT_ROOT_RENDER_DEVICE,
            quilting_shaders::RESIDENT_ROOT_RENDER_DEVICE_VERTEX_ENTRY_POINT,
            quilting_shaders::compile_resident_root_render_device_wgsl,
        )?;
        let (
            bind_group_layout,
            pbr_portable_atlas_bind_group_layout,
            pbr_environment_bind_group_layout,
        ) = if let Some(descriptors) = descriptors {
            let expected_descriptor_count = RESIDENT_ROOT_PIPELINE_PASSES.len() * 2;
            if descriptors.len() != expected_descriptor_count {
                return Err(LodWebGpuError::Payload(
                    format!(
                        "functional resident-root pipeline family must contain {expected_descriptor_count} variants",
                    ),
                ));
            }
            let pbr_descriptor_count = RESIDENT_ROOT_PIPELINE_PASSES
                .iter()
                .take_while(|pass| pass.uses_pbr)
                .count()
                * 2;
            let pbr_layout = descriptors[0].layout();
            let diagnostic_layout = descriptors[pbr_descriptor_count].layout();
            if descriptors[..pbr_descriptor_count]
                .iter()
                .any(|descriptor| descriptor.layout() != pbr_layout)
                || descriptors[pbr_descriptor_count..]
                    .iter()
                    .any(|descriptor| descriptor.layout() != diagnostic_layout)
                || pbr_layout.groups().len() != 3
                || diagnostic_layout.groups().len() != 1
                || pbr_layout.groups()[0] != diagnostic_layout.groups()[0]
            {
                return Err(LodWebGpuError::Payload(
                    "functional resident-root pipeline layouts are inconsistent".to_string(),
                ));
            }
            (
                crate::pipeline_lowering::bind_group_layout(
                    &self.device,
                    "quilting resident root render bindings",
                    &pbr_layout.groups()[0],
                ),
                crate::pipeline_lowering::bind_group_layout(
                    &self.device,
                    "quilting portable PBR atlas binding layout",
                    &pbr_layout.groups()[1],
                ),
                crate::pipeline_lowering::bind_group_layout(
                    &self.device,
                    "quilting PBR environment binding layout",
                    &pbr_layout.groups()[2],
                ),
            )
        } else {
            (
                self.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("quilting resident root render bindings"),
                        entries: &[
                            render_buffer_layout_visible(
                                0,
                                wgpu::BufferBindingType::Storage { read_only: true },
                                PATCH_RENDER_GLOBAL_BYTES,
                                false,
                                wgpu::ShaderStages::VERTEX_FRAGMENT,
                            ),
                            render_buffer_layout_visible(
                                1,
                                wgpu::BufferBindingType::Storage { read_only: true },
                                PATCH_RENDER_DOMAIN_BYTES,
                                false,
                                wgpu::ShaderStages::VERTEX_FRAGMENT,
                            ),
                            render_buffer_layout(
                                2,
                                wgpu::BufferBindingType::Storage { read_only: true },
                                PREPARED_PATCH_RECORD_BYTES,
                                false,
                            ),
                            render_buffer_layout(
                                3,
                                wgpu::BufferBindingType::Storage { read_only: true },
                                PACKED_RECORD_BYTES,
                                false,
                            ),
                            render_buffer_layout(
                                4,
                                wgpu::BufferBindingType::Storage { read_only: true },
                                RESIDENT_BUCKET_RANGE_RECORD_BYTES,
                                false,
                            ),
                            render_buffer_layout(
                                5,
                                wgpu::BufferBindingType::Uniform,
                                DRAW_BATCH_INDEX_BYTES,
                                true,
                            ),
                            render_buffer_layout_visible(
                                6,
                                wgpu::BufferBindingType::Storage { read_only: true },
                                PACKED_RECORD_BYTES,
                                false,
                                wgpu::ShaderStages::VERTEX_FRAGMENT,
                            ),
                            render_buffer_layout_visible(
                                7,
                                wgpu::BufferBindingType::Storage { read_only: true },
                                16,
                                false,
                                wgpu::ShaderStages::VERTEX_FRAGMENT,
                            ),
                            render_buffer_layout_visible(
                                8,
                                wgpu::BufferBindingType::Storage { read_only: true },
                                PATCH_PBR_MATERIAL_BYTES,
                                false,
                                wgpu::ShaderStages::FRAGMENT,
                            ),
                        ],
                    }),
                pbr_resources::create_pbr_portable_atlas_bind_group_layout(&self.device),
                pbr_environment::create_pbr_environment_bind_group_layout(&self.device),
            )
        };
        let diagnostic_pipeline_layout =
            self.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("quilting resident root render pipeline layout"),
                    bind_group_layouts: &[Some(&bind_group_layout)],
                    immediate_size: 0,
                });
        let pbr_pipeline_layout =
            self.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("quilting resident root PBR pipeline layout"),
                    bind_group_layouts: &[
                        Some(&bind_group_layout),
                        Some(&pbr_portable_atlas_bind_group_layout),
                        Some(&pbr_environment_bind_group_layout),
                    ],
                    immediate_size: 0,
                });
        let depth_stencil = depth_format.map(|format| wgpu::DepthStencilState {
            format,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        });
        let attributes = wgpu::vertex_attr_array![0 => Float32x3];
        let create = |index: usize,
                      pass: ResidentRootPipelinePass,
                      front_face,
                      pipeline_layout: &wgpu::PipelineLayout|
         -> Result<wgpu::RenderPipeline, LodWebGpuError> {
            if let Some(descriptors) = descriptors {
                let descriptor = &descriptors[index];
                let fragment = descriptor.program().fragment().ok_or_else(|| {
                    LodWebGpuError::Payload(
                        "functional resident-root pipeline is missing a fragment stage".to_string(),
                    )
                })?;
                let expected_front_face = match front_face {
                    wgpu::FrontFace::Ccw => functional::FrontFace::CounterClockwise,
                    wgpu::FrontFace::Cw => functional::FrontFace::Clockwise,
                };
                let expected_depth_write = depth_format.map(|_| pass.depth_write);
                if fragment.entry_point() != pass.fragment_entry_point
                    || descriptor.program().vertex().entry_point()
                        != pass.vertex_entry_point
                    || descriptor.primitive().topology != pass.topology()
                    || descriptor.primitive().front_face != expected_front_face
                    || descriptor
                        .depth_stencil()
                        .map(|state| state.depth_write_enabled)
                        != expected_depth_write
                    || descriptor.color_targets().len() != 1 + usize::from(pass.focus_mrt)
                {
                    return Err(LodWebGpuError::Payload(
                        "functional resident-root pipeline order is inconsistent".to_string(),
                    ));
                }
                return crate::pipeline_lowering::render_pipeline(
                    &self.device,
                    pass.label,
                    pipeline_layout,
                    &module,
                    descriptor,
                );
            }
            let mut targets = vec![Some(wgpu::ColorTargetState {
                format: color_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })];
            if pass.focus_mrt {
                targets.push(Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }));
            }
            Ok(self
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(pass.label),
                    layout: Some(pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &module,
                        entry_point: Some(pass.vertex_entry_point),
                        compilation_options: Default::default(),
                        buffers: &[wgpu::VertexBufferLayout {
                            array_stride: 12,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: &attributes,
                        }],
                    },
                    primitive: wgpu::PrimitiveState {
                        topology: match pass.geometry {
                            RenderGeometry::Triangles => wgpu::PrimitiveTopology::TriangleList,
                            RenderGeometry::Lines => wgpu::PrimitiveTopology::LineList,
                        },
                        front_face,
                        cull_mode: None,
                        ..Default::default()
                    },
                    depth_stencil: depth_stencil.clone().map(|mut state| {
                        state.depth_write_enabled = Some(pass.depth_write);
                        state
                    }),
                    multisample: wgpu::MultisampleState {
                        count: sample_count,
                        ..Default::default()
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &module,
                        entry_point: Some(pass.fragment_entry_point),
                        compilation_options: Default::default(),
                        targets: &targets,
                    }),
                    multiview_mask: None,
                    cache: None,
                }))
        };
        let mut winding_pipelines = Vec::with_capacity(RESIDENT_ROOT_PIPELINE_PASSES.len());
        for (pass_index, pass) in RESIDENT_ROOT_PIPELINE_PASSES.into_iter().enumerate() {
            let base_index = pass_index * 2;
            let pipeline_layout = if pass.uses_pbr {
                &pbr_pipeline_layout
            } else {
                &diagnostic_pipeline_layout
            };
            winding_pipelines.push(ResidentRootWindingPipelines {
                counter_clockwise: create(base_index, pass, wgpu::FrontFace::Ccw, pipeline_layout)?,
                clockwise: create(base_index + 1, pass, wgpu::FrontFace::Cw, pipeline_layout)?,
            });
        }
        let winding_pipelines: [ResidentRootWindingPipelines; 8] =
            winding_pipelines.try_into().map_err(|_| {
                LodWebGpuError::Payload(
                    "resident-root pass plan did not produce eight pipeline families".to_string(),
                )
            })?;
        let [pbr, pbr_focus, matcap, normals, lod, stretch, wire, highlight] = winding_pipelines;
        Ok(ResidentRootRenderPipeline {
            color_format,
            sample_count,
            bind_group_layout,
            pbr,
            pbr_focus,
            matcap,
            normals,
            lod,
            stretch,
            wire,
            highlight,
            pbr_portable_atlas_bind_group_layout,
            pbr_environment_bind_group_layout,
        })
    }

    pub fn create_resident_root_render_bindings(
        &self,
        pipeline: &ResidentRootRenderPipeline,
        preparation: &ResidentRootPreparationScene,
        geometry: &ResidentGeometryBucketScene,
    ) -> Result<ResidentRootRenderBindings, LodWebGpuError> {
        self.create_resident_root_render_bindings_impl(
            pipeline,
            preparation,
            geometry,
            &[],
            None,
            None,
            false,
            false,
        )
    }

    pub fn create_resident_root_render_bindings_with_pbr(
        &self,
        pipeline: &ResidentRootRenderPipeline,
        preparation: &ResidentRootPreparationScene,
        geometry: &ResidentGeometryBucketScene,
        scene: &RenderSceneSnapshot,
        textures: Option<&PbrTextureTable>,
        environment: Option<&PbrEnvironmentMap>,
    ) -> Result<ResidentRootRenderBindings, LodWebGpuError> {
        self.create_resident_root_render_bindings_impl(
            pipeline,
            preparation,
            geometry,
            &scene.materials,
            textures,
            environment,
            true,
            supports_basic_pbr_frame(scene, RenderFrameOptions::default()),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn create_resident_root_render_bindings_impl(
        &self,
        pipeline: &ResidentRootRenderPipeline,
        preparation: &ResidentRootPreparationScene,
        geometry: &ResidentGeometryBucketScene,
        materials: &[PbrMaterial],
        textures: Option<&PbrTextureTable>,
        environment: Option<&PbrEnvironmentMap>,
        retain_pbr_resources: bool,
        pbr_scene_supported: bool,
    ) -> Result<ResidentRootRenderBindings, LodWebGpuError> {
        let domains = &preparation.draw_domains;
        if domains.model_identity != preparation.topology.model_identity
            || geometry.model_identity != preparation.topology.model_identity
            || geometry.domain_identity != domains.domain_identity
            || domains.face_count != preparation.topology.face_count
            || geometry.face_count != preparation.topology.face_count
            || domains.domain_count == 0
            || geometry.bucket_count == 0
        {
            return Err(LodWebGpuError::Payload(
                "resident root render bindings have incompatible retained domains".to_string(),
            ));
        }
        let render_domain_bytes = u64::from(domains.domain_count)
            .checked_mul(PATCH_RENDER_DOMAIN_BYTES)
            .ok_or_else(|| {
                LodWebGpuError::Payload(
                    "resident root render-domain table is too large".to_string(),
                )
            })?;
        let render_domain_word_count = usize::try_from(domains.domain_count)
            .ok()
            .and_then(|count| count.checked_mul(PATCH_RENDER_DOMAIN_WORDS))
            .ok_or_else(|| {
                LodWebGpuError::Payload(
                    "resident root render-domain staging table exceeds address space".to_string(),
                )
            })?;
        let global_frame = self.create_patch_render_global_residency("resident root global frame");
        let render_domains = gpu_buffer(
            &self.device,
            "resident root render-domain table",
            render_domain_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let uniform_alignment = self
            .device
            .limits()
            .min_uniform_buffer_offset_alignment
            .max(1);
        let bucket_index_uniform_stride = u32::try_from(
            DRAW_BATCH_INDEX_BYTES.div_ceil(u64::from(uniform_alignment))
                * u64::from(uniform_alignment),
        )
        .map_err(|_| LodWebGpuError::Payload("root bucket-index stride exceeds u32".into()))?;
        let bucket_index_bytes = usize::try_from(geometry.bucket_count)
            .ok()
            .and_then(|count| count.checked_mul(bucket_index_uniform_stride as usize))
            .ok_or_else(|| {
                LodWebGpuError::Payload("root bucket-index table is too large".to_string())
            })?;
        let mut bucket_index_words = vec![0u8; bucket_index_bytes];
        for bucket in 0..geometry.bucket_count {
            let offset = bucket as usize * bucket_index_uniform_stride as usize;
            bucket_index_words[offset..offset + 4].copy_from_slice(&bucket.to_le_bytes());
        }
        let bucket_index_uniform = buffer_init_or_zero(
            &self.device,
            "resident root draw bucket indices",
            &bucket_index_words,
            wgpu::BufferUsages::UNIFORM,
        );
        let material_words = patch_pbr_material_table_words(materials)?;
        let material_buffer = buffer_init_or_zero(
            &self.device,
            "resident root authored PBR material table",
            bytemuck::cast_slice(&material_words),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let portable_material_textures = retain_pbr_resources
            .then(|| {
                self.create_pbr_portable_atlas_bindings_for_layout(
                    &pipeline.pbr_portable_atlas_bind_group_layout,
                    materials,
                    textures,
                )
            })
            .transpose()?;
        let pbr_environment = retain_pbr_resources
            .then(|| {
                self.create_pbr_environment_bindings_for_layout(
                    &pipeline.pbr_environment_bind_group_layout,
                    environment,
                )
            })
            .transpose()?;
        let visibility_layout = self
            .resident_root_visibility_pipeline
            .get_bind_group_layout(0);
        let visibility_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("quilting resident root visibility bindings"),
            layout: &visibility_layout,
            entries: &[
                bind(0, &geometry.uniform),
                bind(1, &global_frame.buffer),
                bind(2, &render_domains),
                bind(3, &preparation.patches.prepared_records),
                bind(4, &geometry.root_eligibility),
                bind(5, &domains.face_domain_rows),
                bind(6, &domains.domain_records),
                bind(7, &geometry.root_visibility),
            ],
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("quilting resident root render bindings"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                bind(0, &global_frame.buffer),
                bind(1, &render_domains),
                bind(2, &preparation.patches.prepared_records),
                bind(3, &geometry.compacted_faces),
                bind(4, &geometry.bucket_ranges),
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &bucket_index_uniform,
                        offset: 0,
                        size: std::num::NonZeroU64::new(DRAW_BATCH_INDEX_BYTES),
                    }),
                },
                bind(6, &domains.face_domain_rows),
                bind(7, &domains.domain_records),
                bind(8, &material_buffer),
            ],
        });
        Ok(ResidentRootRenderBindings {
            model_identity: preparation.topology.model_identity,
            domain_identity: domains.domain_identity,
            domain_count: domains.domain_count,
            bucket_count: geometry.bucket_count,
            bucket_index_uniform_stride,
            global_frame,
            render_domains,
            _materials: material_buffer,
            portable_material_textures,
            pbr_environment,
            pbr_scene_supported,
            render_domain_table: Mutex::new(RetainedFrameTable::new(
                render_domain_word_count,
                PATCH_RENDER_DOMAIN_WORDS,
                render_domain_bytes,
            )),
            _bucket_index_uniform: bucket_index_uniform,
            visibility_bind_group,
            bind_group,
        })
    }

    pub fn write_resident_root_render_frames(
        &self,
        bindings: &ResidentRootRenderBindings,
        frame: &RenderFrame,
        domains: &ResidentRootDrawDomainScene,
        use_qb: bool,
    ) -> Result<(), LodWebGpuError> {
        if bindings.model_identity != domains.model_identity
            || bindings.domain_identity != domains.domain_identity
            || bindings.domain_count != domains.domain_count
        {
            return Err(LodWebGpuError::Payload(
                "resident root render frames belong to a different domain table".to_string(),
            ));
        }
        self.write_patch_render_global(
            &bindings.global_frame,
            PatchRenderGlobal::from_render_frame(frame, use_qb),
        )?;

        let mut domain_table = bindings.render_domain_table.lock().map_err(|_| {
            LodWebGpuError::Payload(
                "resident root render-domain staging lock was poisoned".to_string(),
            )
        })?;
        let mut domains_changed = domain_table.begin_update();
        for (row, domain) in domains.domains.iter().enumerate() {
            let words = match u32::try_from(domain.material_index)
                .map_err(|_| {
                    LodWebGpuError::Payload("resident root material slot exceeds u32".to_string())
                })
                .and_then(|material_slot| {
                    PatchRenderDomain::from_transform(domain.transform, material_slot).to_words()
                }) {
                Ok(words) => words,
                Err(error) => {
                    domain_table.invalidate();
                    return Err(error);
                }
            };
            domains_changed |= domain_table.replace_row(row, &words);
        }
        let domain_publication = domain_table.commit(domains_changed);
        if matches!(domain_publication, FrameTablePublication::Upload { .. }) {
            self.queue.write_buffer(
                &bindings.render_domains,
                0,
                bytemuck::cast_slice(domain_table.words.as_slice()),
            );
        }
        self.record_frame_table_publication(domain_publication);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_resident_roots<'resource>(
        &'resource self,
        encoder: &mut wgpu::CommandEncoder,
        frame: &RenderFrame,
        scene: &RenderSceneSnapshot,
        model: &LodClassifierModel,
        resident: &DeviceResidentLod<'_>,
        preparation: &'resource ResidentRootPreparationScene,
        geometry: &'resource ResidentGeometryBucketScene,
        pipeline: &'resource ResidentRootRenderPipeline,
        bindings: &'resource ResidentRootRenderBindings,
        atlas: &'resource PackedPatchAtlas,
        target: PatchRenderTarget<'resource>,
        pose: LodPose<'_>,
        num_joints: u32,
        pose_upload: PoseUploadPolicy,
        use_qb: bool,
    ) -> Result<ResidentRootFrameEncoding, LodWebGpuError> {
        self.encode_resident_roots_with_raw_field(
            encoder,
            frame,
            scene,
            model,
            resident,
            preparation,
            geometry,
            pipeline,
            bindings,
            atlas,
            target,
            None,
            pose,
            num_joints,
            pose_upload,
            use_qb,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_resident_roots_with_raw_field<'resource>(
        &'resource self,
        encoder: &mut wgpu::CommandEncoder,
        frame: &RenderFrame,
        scene: &RenderSceneSnapshot,
        model: &LodClassifierModel,
        resident: &DeviceResidentLod<'_>,
        preparation: &'resource ResidentRootPreparationScene,
        geometry: &'resource ResidentGeometryBucketScene,
        pipeline: &'resource ResidentRootRenderPipeline,
        bindings: &'resource ResidentRootRenderBindings,
        atlas: &'resource PackedPatchAtlas,
        target: PatchRenderTarget<'resource>,
        raw_field_target: Option<&'resource wgpu::TextureView>,
        pose: LodPose<'_>,
        num_joints: u32,
        pose_upload: PoseUploadPolicy,
        use_qb: bool,
    ) -> Result<ResidentRootFrameEncoding, LodWebGpuError> {
        frame
            .execution(scene)
            .map_err(|error| LodWebGpuError::Payload(format!("render frame contract: {error}")))?;
        let draw_passes = render_draw_passes(frame.style);
        let focus_frame = raw_field_target.is_some();
        if focus_frame && frame.style != RenderStyle::Pbr {
            return Err(LodWebGpuError::Payload(format!(
                "resident root focus target cannot render {:?}",
                frame.style,
            )));
        }
        if frame.style == RenderStyle::Pbr
            && (!bindings.supports_resident_basic_pbr()
                || frame.options.focus_postprocess.is_some() != focus_frame)
        {
            return Err(LodWebGpuError::Payload(
                "resident root PBR requires exact texture/IBL bindings and a focus target matching its postprocess packet"
                    .to_string(),
            ));
        }
        if focus_frame && frame.options.highlight_face.is_some() {
            return Err(LodWebGpuError::Payload(
                "resident root focus PBR does not yet compose source-face highlight".to_string(),
            ));
        }
        if draw_passes
            .iter()
            .filter(|draw| draw.pass != RenderPass::PbrTransparent)
            .any(|draw| pipeline.for_pass(draw.pass).is_none())
        {
            return Err(LodWebGpuError::Payload(format!(
                "resident root renderer does not support {:?}",
                frame.style,
            )));
        }
        if bindings.model_identity != model.identity
            || geometry.model_identity != model.identity
            || preparation.topology.model_identity != model.identity
            || bindings.domain_identity != preparation.draw_domains.domain_identity
            || geometry.domain_identity != preparation.draw_domains.domain_identity
            || bindings.bucket_count != geometry.bucket_count
            || geometry.atlas_count != atlas.keys.len() as u32
        {
            return Err(LodWebGpuError::Payload(
                "resident root render resources belong to different epochs".to_string(),
            ));
        }
        self.write_resident_root_render_frames(bindings, frame, &preparation.draw_domains, use_qb)?;
        if pose_upload.should_publish_dynamic() {
            self.write_dynamic_pose(model, pose, num_joints)?;
        }
        if pose_upload.should_publish_preparation() {
            self.write_patch_joint_count(&preparation.patches, num_joints);
        }
        self.encode_resident_root_preparation(preparation, resident, encoder)?;
        self.encode_resident_root_visibility(preparation, geometry, bindings, encoder)?;
        self.encode_resident_geometry_buckets(geometry, resident, encoder)?;

        let color_load = target
            .clear_color
            .map_or(wgpu::LoadOp::Load, wgpu::LoadOp::Clear);
        let depth_stencil_attachment =
            target
                .depth_stencil_view
                .map(|view| wgpu::RenderPassDepthStencilAttachment {
                    view,
                    depth_ops: Some(wgpu::Operations {
                        load: target
                            .clear_depth
                            .map_or(wgpu::LoadOp::Load, wgpu::LoadOp::Clear),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                });
        let primary_attachment = Some(wgpu::RenderPassColorAttachment {
            view: target.color_view,
            depth_slice: None,
            resolve_target: target.resolve_target,
            ops: wgpu::Operations {
                load: color_load,
                store: wgpu::StoreOp::Store,
            },
        });
        let raw_attachment = raw_field_target.map(|view| {
            Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(
                        crate::focus_postprocess::FOCUS_RAW_FIELD_CLEAR,
                    ),
                    store: wgpu::StoreOp::Store,
                },
            })
        });
        let color_attachments = [primary_attachment, raw_attachment.flatten()];
        let color_attachments = if focus_frame {
            &color_attachments[..]
        } else {
            &color_attachments[..1]
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("quilting resident root frame"),
            color_attachments,
            depth_stencil_attachment,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_vertex_buffer(0, atlas.barycentric_buffer.slice(..));
        let mut indirect_draw_calls = 0u32;
        for &draw in draw_passes {
            if draw.pass == RenderPass::PbrTransparent {
                continue;
            }
            let style_pipeline = if focus_frame && draw.pass == RenderPass::PbrOpaque {
                &pipeline.pbr_focus
            } else {
                pipeline
                    .for_pass(draw.pass)
                    .expect("resident root pass support was validated above")
            };
            if draw.pass == RenderPass::PbrOpaque {
                let portable_material_textures = bindings
                    .portable_material_textures
                    .as_ref()
                    .ok_or_else(|| {
                        LodWebGpuError::Payload(
                            "resident root PBR has no portable texture atlas binding".to_string(),
                        )
                    })?;
                pass.set_bind_group(1, &portable_material_textures.bind_group, &[]);
                let environment = bindings.pbr_environment.as_ref().ok_or_else(|| {
                    LodWebGpuError::Payload(
                        "resident root PBR has no environment binding".to_string(),
                    )
                })?;
                pass.set_bind_group(2, environment.bind_group(), &[]);
            }
            let indirect_arguments = match draw.geometry {
                RenderGeometry::Triangles => {
                    pass.set_index_buffer(
                        atlas.triangle_index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    &geometry.triangle_indirect_arguments
                }
                RenderGeometry::Lines => {
                    pass.set_index_buffer(
                        atlas.line_index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    &geometry.line_indirect_arguments
                }
            };
            for bucket in 0..geometry.bucket_count {
                let render_pipeline = if bucket % 2 == 0 {
                    &style_pipeline.counter_clockwise
                } else {
                    &style_pipeline.clockwise
                };
                pass.set_pipeline(render_pipeline);
                pass.set_bind_group(
                    0,
                    &bindings.bind_group,
                    &[bucket * bindings.bucket_index_uniform_stride],
                );
                pass.draw_indexed_indirect(
                    indirect_arguments,
                    u64::from(bucket) * INDEXED_INDIRECT_RECORD_BYTES,
                );
                indirect_draw_calls = indirect_draw_calls.saturating_add(1);
            }
        }
        if frame.options.highlight_face.is_some() {
            pass.set_index_buffer(
                atlas.triangle_index_buffer.slice(..),
                wgpu::IndexFormat::Uint32,
            );
            for bucket in 0..geometry.bucket_count {
                let render_pipeline = if bucket % 2 == 0 {
                    &pipeline.highlight.counter_clockwise
                } else {
                    &pipeline.highlight.clockwise
                };
                pass.set_pipeline(render_pipeline);
                pass.set_bind_group(
                    0,
                    &bindings.bind_group,
                    &[bucket * bindings.bucket_index_uniform_stride],
                );
                pass.draw_indexed_indirect(
                    &geometry.triangle_indirect_arguments,
                    u64::from(bucket) * INDEXED_INDIRECT_RECORD_BYTES,
                );
                indirect_draw_calls = indirect_draw_calls.saturating_add(1);
            }
        }
        drop(pass);
        Ok(ResidentRootFrameEncoding {
            indirect_draw_calls,
            source_face_count: geometry.face_count,
        })
    }

    /// Encode device-resident source roots into the retained focus MRT and
    /// immediately compose them through the shared Rust focus schedule.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_focus_resident_roots<'resource>(
        &'resource self,
        encoder: &mut wgpu::CommandEncoder,
        frame: &RenderFrame,
        scene: &RenderSceneSnapshot,
        model: &LodClassifierModel,
        resident: &DeviceResidentLod<'_>,
        preparation: &'resource ResidentRootPreparationScene,
        geometry: &'resource ResidentGeometryBucketScene,
        pipeline: &'resource ResidentRootRenderPipeline,
        bindings: &'resource ResidentRootRenderBindings,
        atlas: &'resource PackedPatchAtlas,
        focus_pipelines: &FocusPostprocessPipelines,
        focus_target: &'resource FocusPostprocessTarget,
        output_target: &'resource OffscreenPatchRenderTarget,
        pose: LodPose<'_>,
        num_joints: u32,
        pose_upload: PoseUploadPolicy,
        use_qb: bool,
    ) -> Result<FocusResidentRootFrameEncoding, LodWebGpuError> {
        if frame.view.viewport != focus_target.size() || frame.view.viewport != output_target.size {
            return Err(LodWebGpuError::Payload(format!(
                "resident focus/output targets {:?}/{:?} do not match viewport {:?}",
                focus_target.size(),
                output_target.size,
                frame.view.viewport,
            )));
        }
        if pipeline.color_format != focus_target.scene_color_format() || pipeline.sample_count != 1
        {
            return Err(LodWebGpuError::Payload(
                "resident focus path requires an RGBA8 single-sample root pipeline".to_string(),
            ));
        }
        let packet = frame.options.focus_postprocess.ok_or_else(|| {
            LodWebGpuError::Payload("resident focus frame has no postprocess packet".to_string())
        })?;
        let roots = self.encode_resident_roots_with_raw_field(
            encoder,
            frame,
            scene,
            model,
            resident,
            preparation,
            geometry,
            pipeline,
            bindings,
            atlas,
            PatchRenderTarget {
                color_view: focus_target.scene_color_view(),
                resolve_target: None,
                depth_stencil_view: Some(&output_target.depth_view),
                clear_color: Some(crate::focus_postprocess::FOCUS_SCENE_CLEAR),
                clear_depth: Some(1.0),
            },
            Some(focus_target.raw_field_view()),
            pose,
            num_joints,
            pose_upload,
            use_qb,
        )?;
        let postprocess = self.encode_focus_postprocess(
            encoder,
            focus_pipelines,
            focus_target,
            &output_target.color_view,
            packet,
        )?;
        Ok(FocusResidentRootFrameEncoding { roots, postprocess })
    }

    /// Submit one complete focus frame from the current device-resident root
    /// LOD epoch. No topology, visibility, or image data crosses the CPU.
    #[allow(clippy::too_many_arguments)]
    pub fn render_offscreen_focus_resident_roots(
        &self,
        frame: &RenderFrame,
        render_scene: &RenderSceneSnapshot,
        model: &LodClassifierModel,
        resident: &DeviceResidentLod<'_>,
        preparation: &ResidentRootPreparationScene,
        geometry: &ResidentGeometryBucketScene,
        pipeline: &ResidentRootRenderPipeline,
        bindings: &ResidentRootRenderBindings,
        atlas: &PackedPatchAtlas,
        focus_pipelines: &FocusPostprocessPipelines,
        focus_target: &FocusPostprocessTarget,
        output_target: &OffscreenPatchRenderTarget,
        pose: LodPose<'_>,
        num_joints: u32,
        pose_upload: PoseUploadPolicy,
        use_qb: bool,
    ) -> Result<FocusPatchFrameEncoding, LodWebGpuError> {
        let logical_submission = resident_root_frame_submission(frame, render_scene)?;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("quilting resident root focus frame"),
            });
        let encoding = self.encode_focus_resident_roots(
            &mut encoder,
            frame,
            render_scene,
            model,
            resident,
            preparation,
            geometry,
            pipeline,
            bindings,
            atlas,
            focus_pipelines,
            focus_target,
            output_target,
            pose,
            num_joints,
            pose_upload,
            use_qb,
        )?;
        self.queue.submit([encoder.finish()]);
        let scene = resident_root_frame_evidence(logical_submission, encoding.roots);
        Ok(FocusPatchFrameEncoding {
            scene,
            postprocess: encoding.postprocess,
        })
    }

    /// Submit one root-only supported frame to retained offscreen attachments.
    /// Classification, topology emission, preparation, visibility rejection,
    /// atlas bucketing, and indirect rendering stay ordered on the device.
    #[allow(clippy::too_many_arguments)]
    pub fn render_offscreen_resident_roots(
        &self,
        frame: &RenderFrame,
        render_scene: &RenderSceneSnapshot,
        model: &LodClassifierModel,
        resident: &DeviceResidentLod<'_>,
        preparation: &ResidentRootPreparationScene,
        geometry: &ResidentGeometryBucketScene,
        pipeline: &ResidentRootRenderPipeline,
        bindings: &ResidentRootRenderBindings,
        atlas: &PackedPatchAtlas,
        target: &OffscreenPatchRenderTarget,
        pose: LodPose<'_>,
        num_joints: u32,
        pose_upload: PoseUploadPolicy,
        use_qb: bool,
    ) -> Result<PatchFrameEncoding, LodWebGpuError> {
        let logical_submission = resident_root_frame_submission(frame, render_scene)?;
        if frame.view.viewport != target.size {
            return Err(LodWebGpuError::Payload(format!(
                "offscreen target {:?} does not match frame viewport {:?}",
                target.size, frame.view.viewport,
            )));
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("quilting resident root offscreen frame"),
            });
        let encoding = self.encode_resident_roots(
            &mut encoder,
            frame,
            render_scene,
            model,
            resident,
            preparation,
            geometry,
            pipeline,
            bindings,
            atlas,
            PatchRenderTarget {
                color_view: &target.color_view,
                resolve_target: None,
                depth_stencil_view: Some(&target.depth_view),
                clear_color: Some(DEFAULT_EVIDENCE_CLEAR),
                clear_depth: Some(1.0),
            },
            pose,
            num_joints,
            pose_upload,
            use_qb,
        )?;
        self.queue.submit([encoder.finish()]);
        Ok(resident_root_frame_evidence(logical_submission, encoding))
    }

    /// Present one root-only supported frame through the same direct encoder as
    /// the offscreen path. Surface acquisition and queue submission remain
    /// Rust-owned and no diagnostic copy or CPU map is introduced.
    #[allow(clippy::too_many_arguments)]
    pub fn present_resident_roots(
        &self,
        surface: &mut PatchPresentationSurface,
        frame: &RenderFrame,
        render_scene: &RenderSceneSnapshot,
        model: &LodClassifierModel,
        resident: &DeviceResidentLod<'_>,
        preparation: &ResidentRootPreparationScene,
        geometry: &ResidentGeometryBucketScene,
        pipeline: &ResidentRootRenderPipeline,
        bindings: &ResidentRootRenderBindings,
        atlas: &PackedPatchAtlas,
        pose: LodPose<'_>,
        num_joints: u32,
        pose_upload: PoseUploadPolicy,
        use_qb: bool,
    ) -> Result<SurfacePresentation<PatchFrameEncoding>, LodWebGpuError> {
        let logical_submission = resident_root_frame_submission(frame, render_scene)?;
        surface.present_with(
            self,
            "quilting resident root presentation frame",
            |encoder, mut target| {
                target.clear_color = Some(DEFAULT_FRAME_CLEAR);
                target.clear_depth = Some(1.0);
                let encoding = self.encode_resident_roots(
                    encoder,
                    frame,
                    render_scene,
                    model,
                    resident,
                    preparation,
                    geometry,
                    pipeline,
                    bindings,
                    atlas,
                    target,
                    pose,
                    num_joints,
                    pose_upload,
                    use_qb,
                )?;
                Ok(resident_root_frame_evidence(logical_submission, encoding))
            },
        )
    }
}

fn resident_root_frame_submission(
    frame: &RenderFrame,
    scene: &RenderSceneSnapshot,
) -> Result<RenderSubmissionStats, LodWebGpuError> {
    frame
        .expected_submission_stats(scene)
        .map_err(|error| LodWebGpuError::Payload(format!("render frame contract: {error}")))
}

fn resident_root_frame_evidence(
    logical_submission: RenderSubmissionStats,
    encoding: ResidentRootFrameEncoding,
) -> PatchFrameEncoding {
    PatchFrameEncoding {
        logical_submission,
        indirect_draw_calls: encoding.indirect_draw_calls,
        source_instance_count: encoding.source_face_count,
    }
}

impl LodClassifierDevice {
    /// Retain one dense row per observed material/render domain plus one row
    /// selector per source face. LOD changes never rebuild either table.
    pub fn upload_resident_root_draw_domain_scene(
        &self,
        model: &LodClassifierModel,
        scene: &RenderSceneSnapshot,
    ) -> Result<ResidentRootDrawDomainScene, LodWebGpuError> {
        let domains = ResidentRootDrawDomains::build(scene, model.prepared.residency.num_faces)
            .map_err(|error| LodWebGpuError::Payload(error.to_string()))?;
        self.upload_resident_root_draw_domains(model.identity, domains)
    }

    pub(super) fn upload_resident_root_draw_domains(
        &self,
        model_identity: u64,
        domains: ResidentRootDrawDomains,
    ) -> Result<ResidentRootDrawDomainScene, LodWebGpuError> {
        let face_count = u32::try_from(domains.face_domain_rows.len())
            .map_err(|_| LodWebGpuError::Payload("resident root face count exceeds u32".into()))?;
        let domain_count = u32::try_from(domains.domains.len()).map_err(|_| {
            LodWebGpuError::Payload("resident root draw-domain count exceeds u32".into())
        })?;
        if domains
            .face_domain_rows
            .iter()
            .any(|&row| row >= domain_count)
        {
            return Err(LodWebGpuError::Payload(
                "resident root face references a missing draw domain".to_string(),
            ));
        }
        let domain_records = domains
            .domains
            .iter()
            .map(|domain| {
                let material_index = u32::try_from(domain.material_index).map_err(|_| {
                    LodWebGpuError::Payload(
                        "resident root material index exceeds the WebGPU ABI".to_string(),
                    )
                })?;
                let render_node_index = u32::try_from(domain.render_node_index).map_err(|_| {
                    LodWebGpuError::Payload(
                        "resident root render-node index exceeds the WebGPU ABI".to_string(),
                    )
                })?;
                let pbr_class = match domain.pbr_class {
                    PbrDrawClass::Opaque => 0,
                    PbrDrawClass::Blend => 1,
                    PbrDrawClass::Transmission => 2,
                };
                let flags = u32::from(domain.enabled)
                    | (u32::from(domain.transform.orientation_sign < 0) << 1);
                Ok([material_index, render_node_index, pbr_class, flags])
            })
            .collect::<Result<Vec<_>, LodWebGpuError>>()?;
        let face_domain_rows = buffer_init_or_zero(
            &self.device,
            "resident root face draw-domain rows",
            bytemuck::cast_slice(&domains.face_domain_rows),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let domain_records_buffer = buffer_init_or_zero(
            &self.device,
            "resident root draw-domain records",
            bytemuck::cast_slice(&domain_records),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        Ok(ResidentRootDrawDomainScene {
            model_identity,
            domain_identity: self.allocate_model_identity()?,
            face_count,
            domain_count,
            domains: domains.domains,
            face_domain_rows,
            domain_records: domain_records_buffer,
        })
    }

    /// Diagnostic projection of immutable root draw-domain storage. The live
    /// render path binds these buffers directly.
    pub async fn resident_root_draw_domains_for_diagnostics(
        &self,
        scene: &ResidentRootDrawDomainScene,
    ) -> Result<ResidentRootDrawDomainOutput, LodWebGpuError> {
        if scene.face_count == 0 {
            return Ok(ResidentRootDrawDomainOutput {
                face_domain_rows: Vec::new(),
                domain_records: Vec::new(),
            });
        }
        let face_bytes = u64::from(scene.face_count) * PACKED_RECORD_BYTES;
        let domain_bytes = u64::from(scene.domain_count) * 16;
        let face_readback = gpu_buffer(
            &self.device,
            "resident root face-domain diagnostic readback",
            face_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let domain_readback = gpu_buffer(
            &self.device,
            "resident root domain-record diagnostic readback",
            domain_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("resident root draw-domain diagnostic encoder"),
            });
        encoder.copy_buffer_to_buffer(&scene.face_domain_rows, 0, &face_readback, 0, face_bytes);
        encoder.copy_buffer_to_buffer(&scene.domain_records, 0, &domain_readback, 0, domain_bytes);
        self.queue.submit([encoder.finish()]);
        let face_domain_rows = self.readback_words(&face_readback, face_bytes).await?;
        let words = self.readback_words(&domain_readback, domain_bytes).await?;
        if !words.len().is_multiple_of(4) {
            return Err(LodWebGpuError::Mapping(
                "resident root domain readback is not four-word aligned".to_string(),
            ));
        }
        let domain_records = words
            .chunks_exact(4)
            .map(|record| [record[0], record[1], record[2], record[3]])
            .collect();
        Ok(ResidentRootDrawDomainOutput {
            face_domain_rows,
            domain_records,
        })
    }

    /// Upload immutable source/affine extraction for direct resident-root
    /// preparation. The returned aggregate owns no CPU topology records: its
    /// preparation bind group reads the output of `ResidentRootTopologyScene`.
    pub fn upload_resident_root_preparation_scene(
        &self,
        model: &LodClassifierModel,
        scene: &RenderSceneSnapshot,
        source_instances: &[f32],
    ) -> Result<ResidentRootPreparationScene, LodWebGpuError> {
        let words = pack_wgsl_resident_root_preparation_scene_words(
            &model.prepared,
            scene,
            source_instances,
        )
        .map_err(LodWebGpuError::Payload)?;
        let draw_domains =
            ResidentRootDrawDomains::build(scene, model.prepared.residency.num_faces)
                .map_err(|error| LodWebGpuError::Payload(error.to_string()))?;
        self.upload_resident_root_preparation_words(model, words, draw_domains)
    }

    pub(super) fn upload_resident_root_preparation_words(
        &self,
        model: &LodClassifierModel,
        words: WgslResidentRootPreparationSceneWords,
        draw_domains: ResidentRootDrawDomains,
    ) -> Result<ResidentRootPreparationScene, LodWebGpuError> {
        let face_count = u32::try_from(model.prepared.residency.num_faces)
            .map_err(|_| LodWebGpuError::Payload("resident root faces exceed u32".into()))?;
        let subject_count = u32::try_from(words.subjects.len())
            .map_err(|_| LodWebGpuError::Payload("patch subject count exceeds u32".into()))?;
        let num_morph_targets = u32::try_from(model.prepared.model.num_morph_targets)
            .map_err(|_| LodWebGpuError::Payload("patch morph target count exceeds u32".into()))?;
        if words.uniform
            != [
                face_count,
                model.prepared.residency.num_vertices,
                0,
                num_morph_targets,
            ]
            || words.source_faces.len() != face_count as usize
            || words.face_subject_rows.len() != face_count as usize
            || words.subjects.is_empty()
        {
            return Err(LodWebGpuError::Payload(
                "resident root preparation scene shape is malformed".to_string(),
            ));
        }
        for (face_index, (source, expected_vertices)) in words
            .source_faces
            .iter()
            .zip(&model.prepared.model.faces)
            .enumerate()
        {
            for (corner, &expected_vertex) in expected_vertices.iter().enumerate() {
                let encoded_vertex = f32::from_bits(source[corner * 4]);
                if !encoded_vertex.is_finite()
                    || encoded_vertex < 0.0
                    || encoded_vertex.fract() != 0.0
                    || encoded_vertex as u32 != expected_vertex
                {
                    return Err(LodWebGpuError::Payload(format!(
                        "resident root source face {face_index} corner {corner} does not match the model",
                    )));
                }
            }
        }
        if words
            .source_faces
            .iter()
            .flatten()
            .chain(words.subjects.iter().flatten())
            .copied()
            .map(f32::from_bits)
            .any(|value| !value.is_finite())
        {
            return Err(LodWebGpuError::Payload(
                "resident root preparation source contains non-finite values".to_string(),
            ));
        }
        let topology = self.upload_resident_root_topology_scene(
            model,
            &words.face_subject_rows,
            subject_count,
        )?;
        let patches = self.allocate_patch_preparation_scene(
            model,
            words.uniform,
            topology.topology_records.clone(),
            &words.source_faces,
            &words.subjects,
        )?;
        let draw_domains = self.upload_resident_root_draw_domains(model.identity, draw_domains)?;
        Ok(ResidentRootPreparationScene {
            topology,
            patches,
            draw_domains,
        })
    }

    pub fn write_resident_root_preparation_pose(
        &self,
        model: &LodClassifierModel,
        scene: &ResidentRootPreparationScene,
        pose: LodPose<'_>,
        num_joints: u32,
    ) -> Result<(), LodWebGpuError> {
        if model.identity != scene.topology.model_identity {
            return Err(LodWebGpuError::Payload(
                "resident root preparation belongs to a different WebGPU model".to_string(),
            ));
        }
        self.write_patch_pose(model, &scene.patches, pose, num_joints)
    }

    /// Emit exact source-indexed topology, then prepare animated rational-QB
    /// controls in the same command encoder. The resulting records are pulled
    /// by the compacted source-face IDs produced by resident bucketing.
    pub fn encode_resident_root_preparation(
        &self,
        scene: &ResidentRootPreparationScene,
        resident: &DeviceResidentLod<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<(), LodWebGpuError> {
        self.encode_resident_root_topology(&scene.topology, resident, encoder)?;
        self.encode_patch_preparation(&scene.patches, encoder);
        Ok(())
    }

    /// Diagnostic-only exact readback of direct root preparation.
    pub async fn prepare_resident_roots_for_diagnostics(
        &self,
        model: &LodClassifierModel,
        scene: &ResidentRootPreparationScene,
        resident: &DeviceResidentLod<'_>,
        pose: LodPose<'_>,
        num_joints: u32,
    ) -> Result<Vec<[u32; PREPARED_PATCH_RECORD_WORDS]>, LodWebGpuError> {
        self.write_resident_root_preparation_pose(model, scene, pose, num_joints)?;
        let bytes = u64::from(scene.patches.patch_count) * PREPARED_PATCH_RECORD_BYTES;
        let readback = gpu_buffer(
            &self.device,
            "resident root preparation diagnostic readback",
            bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("resident root preparation diagnostic encoder"),
            });
        self.encode_resident_root_preparation(scene, resident, &mut encoder)?;
        encoder.copy_buffer_to_buffer(&scene.patches.prepared_records, 0, &readback, 0, bytes);
        self.queue.submit([encoder.finish()]);
        words_to_patch_records(self.readback_words(&readback, bytes).await?)
    }
}

impl LodClassifierDevice {
    /// Allocate the deterministic retained-root atlas/parity plan. The atlas
    /// keys must exactly match the sorted classifier lookup uploaded with the
    /// model; equal cardinality alone is not sufficient because packed records
    /// carry only an eight-bit atlas index.
    pub fn upload_resident_geometry_bucket_scene(
        &self,
        model: &LodClassifierModel,
        atlas: &PackedPatchAtlas,
        draw_domains: &ResidentRootDrawDomainScene,
    ) -> Result<ResidentGeometryBucketScene, LodWebGpuError> {
        let face_count = u32::try_from(model.prepared.residency.num_faces)
            .map_err(|_| LodWebGpuError::Payload("resident geometry faces exceed u32".into()))?;
        if draw_domains.model_identity != model.identity || draw_domains.face_count != face_count {
            return Err(LodWebGpuError::Payload(
                "resident geometry draw domains belong to a different WebGPU model".to_string(),
            ));
        }
        self.upload_resident_geometry_bucket_scene_for_records(
            model.identity,
            face_count,
            &model.atlas_keys,
            &model.resident.packed_records,
            atlas,
            draw_domains,
        )
    }

    fn upload_resident_geometry_bucket_scene_for_records(
        &self,
        model_identity: u64,
        face_count: u32,
        atlas_keys: &[[u32; 3]],
        resident_records: &wgpu::Buffer,
        atlas: &PackedPatchAtlas,
        draw_domains: &ResidentRootDrawDomainScene,
    ) -> Result<ResidentGeometryBucketScene, LodWebGpuError> {
        if draw_domains.model_identity != model_identity || draw_domains.face_count != face_count {
            return Err(LodWebGpuError::Payload(
                "resident geometry draw-domain shape does not match the classifier".to_string(),
            ));
        }
        if atlas_keys != atlas.keys {
            return Err(LodWebGpuError::Payload(
                "resident geometry classifier and packed atlas keys differ".to_string(),
            ));
        }
        let atlas_count = u32::try_from(atlas.keys.len())
            .map_err(|_| LodWebGpuError::Payload("resident geometry atlas exceeds u32".into()))?;
        let bucket_count = atlas_count.checked_mul(2).ok_or_else(|| {
            LodWebGpuError::Payload("resident geometry bucket count overflowed".to_string())
        })?;
        if bucket_count == 0 || bucket_count > MAX_RESIDENT_GEOMETRY_BUCKETS {
            return Err(LodWebGpuError::Payload(format!(
                "resident geometry needs 1..={MAX_RESIDENT_GEOMETRY_BUCKETS} buckets; got {bucket_count}",
            )));
        }
        let chunk_count = face_count.div_ceil(LOD_WORKGROUP_SIZE);
        let eligibility_word_count = face_count.div_ceil(32);
        let table_records = u64::from(chunk_count)
            .checked_mul(u64::from(bucket_count))
            .ok_or_else(|| {
                LodWebGpuError::Payload("resident geometry chunk table overflowed".to_string())
            })?;
        if table_records > u64::from(u32::MAX) {
            return Err(LodWebGpuError::Payload(
                "resident geometry chunk table exceeds WGSL u32 indexing".to_string(),
            ));
        }
        let storage_limit = self
            .device
            .limits()
            .max_buffer_size
            .min(self.device.limits().max_storage_buffer_binding_size);
        let storage_bytes = |records: u64, stride: u64, label: &str| {
            let bytes = records.checked_mul(stride).ok_or_else(|| {
                LodWebGpuError::Payload(format!("resident geometry {label} size overflowed"))
            })?;
            if bytes > storage_limit {
                return Err(LodWebGpuError::Payload(format!(
                    "resident geometry {label} needs {bytes} bytes; device storage limit is {storage_limit}",
                )));
            }
            Ok(bytes)
        };
        let table_bytes = storage_bytes(table_records, PACKED_RECORD_BYTES, "chunk table")?;
        storage_bytes(
            u64::from(atlas_count),
            RESIDENT_ATLAS_DRAW_RECORD_BYTES,
            "atlas draws",
        )?;
        let bucket_bytes = storage_bytes(
            u64::from(bucket_count),
            PACKED_RECORD_BYTES,
            "bucket counts",
        )?;
        let range_bytes = storage_bytes(
            u64::from(bucket_count),
            RESIDENT_BUCKET_RANGE_RECORD_BYTES,
            "bucket ranges",
        )?;
        let indirect_bytes = storage_bytes(
            u64::from(bucket_count),
            INDEXED_INDIRECT_RECORD_BYTES,
            "indirect arguments",
        )?;
        let face_bytes = storage_bytes(
            u64::from(face_count),
            PACKED_RECORD_BYTES,
            "compacted faces",
        )?;

        let uniform_words = [face_count, bucket_count, chunk_count, atlas_count];
        let uniform = buffer_init_or_zero(
            &self.device,
            "resident geometry bucket uniform",
            bytemuck::cast_slice(&uniform_words),
            wgpu::BufferUsages::UNIFORM,
        );
        let eligibility = pack_wgsl_root_eligibility_bits(face_count as usize, &[])
            .map_err(LodWebGpuError::Payload)?;
        let root_eligibility = buffer_init_or_zero(
            &self.device,
            "resident root eligibility bits",
            bytemuck::cast_slice(&eligibility),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let root_visibility = buffer_init_or_zero(
            &self.device,
            "resident root visibility bits",
            bytemuck::cast_slice(&eligibility),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );
        let atlas_draws = atlas
            .keys
            .iter()
            .map(|key| {
                let draw = atlas.entries.get(key).ok_or_else(|| {
                    LodWebGpuError::Payload(format!(
                        "resident geometry atlas has no draw for canonical key {key:?}",
                    ))
                })?;
                Ok([
                    draw.triangle_first_index,
                    draw.triangle_index_count,
                    draw.line_first_index,
                    draw.line_index_count,
                ])
            })
            .collect::<Result<Vec<_>, LodWebGpuError>>()?;
        let atlas_draw_buffer = buffer_init_or_zero(
            &self.device,
            "resident geometry atlas draws",
            bytemuck::cast_slice(&atlas_draws),
            wgpu::BufferUsages::STORAGE,
        );
        let chunk_counts = gpu_buffer(
            &self.device,
            "resident geometry chunk counts",
            table_bytes,
            wgpu::BufferUsages::STORAGE,
        );
        let chunk_offsets = gpu_buffer(
            &self.device,
            "resident geometry chunk offsets",
            table_bytes,
            wgpu::BufferUsages::STORAGE,
        );
        let bucket_counts = gpu_buffer(
            &self.device,
            "resident geometry bucket counts",
            bucket_bytes,
            wgpu::BufferUsages::STORAGE,
        );
        let compacted_faces = gpu_buffer(
            &self.device,
            "resident geometry compacted faces",
            face_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let bucket_ranges = gpu_buffer(
            &self.device,
            "resident geometry bucket ranges",
            range_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let triangle_indirect_arguments = gpu_buffer(
            &self.device,
            "resident geometry triangle indirect arguments",
            indirect_bytes,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_SRC,
        );
        let line_indirect_arguments = gpu_buffer(
            &self.device,
            "resident geometry line indirect arguments",
            indirect_bytes,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_SRC,
        );

        let histogram_layout = self
            .resident_bucket_histogram_pipeline
            .get_bind_group_layout(0);
        let histogram_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("resident geometry bucket histogram bindings"),
            layout: &histogram_layout,
            entries: &[
                bind(0, &uniform),
                bind(1, resident_records),
                bind(2, &root_visibility),
                bind(4, &chunk_counts),
                bind(11, &draw_domains.face_domain_rows),
                bind(12, &draw_domains.domain_records),
            ],
        });
        let prefix_layout = self
            .resident_bucket_prefix_pipeline
            .get_bind_group_layout(0);
        let prefix_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("resident geometry bucket prefix bindings"),
            layout: &prefix_layout,
            entries: &[
                bind(0, &uniform),
                bind(4, &chunk_counts),
                bind(5, &chunk_offsets),
                bind(6, &bucket_counts),
            ],
        });
        let scan_layout = self.resident_bucket_scan_pipeline.get_bind_group_layout(0);
        let scan_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("resident geometry bucket scan bindings"),
            layout: &scan_layout,
            entries: &[
                bind(0, &uniform),
                bind(3, &atlas_draw_buffer),
                bind(6, &bucket_counts),
                bind(7, &bucket_ranges),
                bind(8, &triangle_indirect_arguments),
                bind(9, &line_indirect_arguments),
            ],
        });
        let scatter_layout = self
            .resident_bucket_scatter_pipeline
            .get_bind_group_layout(0);
        let scatter_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("resident geometry bucket scatter bindings"),
            layout: &scatter_layout,
            entries: &[
                bind(0, &uniform),
                bind(1, resident_records),
                bind(2, &root_visibility),
                bind(5, &chunk_offsets),
                bind(7, &bucket_ranges),
                bind(10, &compacted_faces),
                bind(11, &draw_domains.face_domain_rows),
                bind(12, &draw_domains.domain_records),
            ],
        });
        Ok(ResidentGeometryBucketScene {
            model_identity,
            domain_identity: draw_domains.domain_identity,
            face_count,
            atlas_count,
            bucket_count,
            chunk_count,
            eligibility_word_count,
            uniform,
            root_eligibility,
            root_visibility,
            suppressed_faces: Mutex::new(Vec::new()),
            _chunk_counts: chunk_counts,
            _chunk_offsets: chunk_offsets,
            _bucket_counts: bucket_counts,
            compacted_faces,
            bucket_ranges,
            triangle_indirect_arguments,
            line_indirect_arguments,
            histogram_bind_group,
            prefix_bind_group,
            scan_bind_group,
            scatter_bind_group,
        })
    }

    pub fn write_resident_root_eligibility_bits(
        &self,
        scene: &ResidentGeometryBucketScene,
        words: &[u32],
    ) -> Result<(), LodWebGpuError> {
        if words.len() != scene.eligibility_word_count as usize {
            return Err(LodWebGpuError::Payload(format!(
                "resident root eligibility has {} words; expected {}",
                words.len(),
                scene.eligibility_word_count,
            )));
        }
        let tail = scene.face_count % 32;
        if tail != 0
            && words
                .last()
                .is_some_and(|word| word & !((1u32 << tail) - 1) != 0)
        {
            return Err(LodWebGpuError::Payload(
                "resident root eligibility has nonzero padding".to_string(),
            ));
        }
        let suppressed_faces = (0..scene.face_count)
            .filter(|&face| words[(face / 32) as usize] & (1u32 << (face % 32)) == 0)
            .collect::<Vec<_>>();
        self.write_resident_root_eligibility_state(scene, words, suppressed_faces)
    }

    /// Classify exact current-pose rational-QB root visibility into the
    /// device-resident bitset consumed by bucket compaction. One invocation
    /// owns a complete 32-face word, so no atomics, clear pass, CPU payload,
    /// or readback is required.
    pub fn encode_resident_root_visibility(
        &self,
        preparation: &ResidentRootPreparationScene,
        geometry: &ResidentGeometryBucketScene,
        bindings: &ResidentRootRenderBindings,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<(), LodWebGpuError> {
        if preparation.topology.model_identity != geometry.model_identity
            || bindings.model_identity != geometry.model_identity
            || preparation.draw_domains.domain_identity != geometry.domain_identity
            || bindings.domain_identity != geometry.domain_identity
            || preparation.topology.face_count != geometry.face_count
        {
            return Err(LodWebGpuError::Payload(
                "resident root visibility resources belong to different epochs".to_string(),
            ));
        }
        if geometry.eligibility_word_count == 0 {
            return Ok(());
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("quilting resident root visibility"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.resident_root_visibility_pipeline);
        pass.set_bind_group(0, &bindings.visibility_bind_group, &[]);
        pass.dispatch_workgroups(
            geometry.eligibility_word_count.div_ceil(LOD_WORKGROUP_SIZE),
            1,
            1,
        );
        Ok(())
    }

    /// Diagnostic-only exact readback of the production resident-root
    /// visibility pass. Ordinary frames consume the same buffer directly in
    /// bucket compaction and never call this method.
    #[allow(clippy::too_many_arguments)]
    pub async fn classify_resident_root_visibility_for_diagnostics(
        &self,
        frame: &RenderFrame,
        model: &LodClassifierModel,
        resident: &DeviceResidentLod<'_>,
        preparation: &ResidentRootPreparationScene,
        geometry: &ResidentGeometryBucketScene,
        bindings: &ResidentRootRenderBindings,
        pose: LodPose<'_>,
        num_joints: u32,
        use_qb: bool,
    ) -> Result<Vec<u32>, LodWebGpuError> {
        self.write_resident_root_render_frames(bindings, frame, &preparation.draw_domains, use_qb)?;
        self.write_resident_root_preparation_pose(model, preparation, pose, num_joints)?;
        let bytes = u64::from(geometry.eligibility_word_count)
            .checked_mul(PACKED_RECORD_BYTES)
            .ok_or_else(|| {
                LodWebGpuError::Payload(
                    "resident root visibility diagnostic is too large".to_string(),
                )
            })?;
        let readback = gpu_buffer(
            &self.device,
            "resident root visibility diagnostic readback",
            bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("resident root visibility diagnostic encoder"),
            });
        self.encode_resident_root_preparation(preparation, resident, &mut encoder)?;
        self.encode_resident_root_visibility(preparation, geometry, bindings, &mut encoder)?;
        encoder.copy_buffer_to_buffer(&geometry.root_visibility, 0, &readback, 0, bytes);
        self.queue.submit([encoder.finish()]);
        self.readback_words(&readback, bytes).await
    }

    fn write_resident_root_eligibility_state(
        &self,
        scene: &ResidentGeometryBucketScene,
        words: &[u32],
        suppressed_faces: Vec<u32>,
    ) -> Result<(), LodWebGpuError> {
        let mut retained_suppression = scene.suppressed_faces.lock().map_err(|_| {
            LodWebGpuError::Payload("resident root suppression lock was poisoned".to_string())
        })?;
        self.queue
            .write_buffer(&scene.root_eligibility, 0, bytemuck::cast_slice(words));
        self.queue
            .write_buffer(&scene.root_visibility, 0, bytemuck::cast_slice(words));
        *retained_suppression = suppressed_faces;
        Ok(())
    }

    pub fn write_resident_root_suppression(
        &self,
        scene: &ResidentGeometryBucketScene,
        suppressed_faces: &[u32],
    ) -> Result<(), LodWebGpuError> {
        let words = pack_wgsl_root_eligibility_bits(scene.face_count as usize, suppressed_faces)
            .map_err(LodWebGpuError::Payload)?;
        self.write_resident_root_eligibility_state(scene, &words, suppressed_faces.to_vec())
    }

    /// Append deterministic root histogram, chunk prefix, global bucket scan,
    /// and stable face scatter to an application-owned encoder.
    pub fn encode_resident_geometry_buckets(
        &self,
        scene: &ResidentGeometryBucketScene,
        resident: &DeviceResidentLod<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<(), LodWebGpuError> {
        if scene.model_identity != resident.model_identity {
            return Err(LodWebGpuError::Payload(
                "resident geometry buckets belong to a different WebGPU model".to_string(),
            ));
        }
        if scene.face_count != resident.face_count {
            return Err(LodWebGpuError::Payload(format!(
                "resident geometry has {} faces; classifier result has {}",
                scene.face_count, resident.face_count,
            )));
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("quilting resident geometry bucket histogram"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.resident_bucket_histogram_pipeline);
            pass.set_bind_group(0, &scene.histogram_bind_group, &[]);
            pass.dispatch_workgroups(scene.chunk_count, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("quilting resident geometry bucket chunk prefix"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.resident_bucket_prefix_pipeline);
            pass.set_bind_group(0, &scene.prefix_bind_group, &[]);
            pass.dispatch_workgroups(scene.bucket_count.div_ceil(LOD_WORKGROUP_SIZE), 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("quilting resident geometry bucket scan"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.resident_bucket_scan_pipeline);
            pass.set_bind_group(0, &scene.scan_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("quilting resident geometry bucket scatter"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.resident_bucket_scatter_pipeline);
            pass.set_bind_group(0, &scene.scatter_bind_group, &[]);
            pass.dispatch_workgroups(scene.chunk_count, 1, 1);
        }
        Ok(())
    }

    /// Diagnostic-only exact readback. The live path binds the compacted face,
    /// range, and indirect buffers directly to preparation/render execution.
    pub async fn resident_geometry_buckets_for_diagnostics(
        &self,
        scene: &ResidentGeometryBucketScene,
        resident: &DeviceResidentLod<'_>,
        suppressed_faces: &[u32],
    ) -> Result<ResidentGeometryBucketOutput, LodWebGpuError> {
        self.write_resident_root_suppression(scene, suppressed_faces)?;
        let face_bytes = u64::from(scene.face_count) * PACKED_RECORD_BYTES;
        let range_bytes = u64::from(scene.bucket_count) * RESIDENT_BUCKET_RANGE_RECORD_BYTES;
        let indirect_bytes = u64::from(scene.bucket_count) * INDEXED_INDIRECT_RECORD_BYTES;
        let face_readback = gpu_buffer(
            &self.device,
            "resident geometry face diagnostic readback",
            face_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let range_readback = gpu_buffer(
            &self.device,
            "resident geometry range diagnostic readback",
            range_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let triangle_indirect_readback = gpu_buffer(
            &self.device,
            "resident geometry triangle indirect diagnostic readback",
            indirect_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let line_indirect_readback = gpu_buffer(
            &self.device,
            "resident geometry line indirect diagnostic readback",
            indirect_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("resident geometry bucket diagnostic encoder"),
            });
        self.encode_resident_geometry_buckets(scene, resident, &mut encoder)?;
        encoder.copy_buffer_to_buffer(&scene.compacted_faces, 0, &face_readback, 0, face_bytes);
        encoder.copy_buffer_to_buffer(&scene.bucket_ranges, 0, &range_readback, 0, range_bytes);
        encoder.copy_buffer_to_buffer(
            &scene.triangle_indirect_arguments,
            0,
            &triangle_indirect_readback,
            0,
            indirect_bytes,
        );
        encoder.copy_buffer_to_buffer(
            &scene.line_indirect_arguments,
            0,
            &line_indirect_readback,
            0,
            indirect_bytes,
        );
        self.queue.submit([encoder.finish()]);
        let bucket_ranges = words_to_five_records(
            self.readback_words(&range_readback, range_bytes).await?,
            "resident geometry range",
        )?;
        let triangle_indirect_arguments = words_to_five_records(
            self.readback_words(&triangle_indirect_readback, indirect_bytes)
                .await?,
            "resident geometry triangle indirect",
        )?;
        let line_indirect_arguments = words_to_five_records(
            self.readback_words(&line_indirect_readback, indirect_bytes)
                .await?,
            "resident geometry line indirect",
        )?;
        let survivor_count = bucket_ranges
            .last()
            .map_or(0u32, |range| range[3].saturating_add(range[4]));
        let mut compacted_faces = self.readback_words(&face_readback, face_bytes).await?;
        compacted_faces.truncate(survivor_count as usize);
        Ok(ResidentGeometryBucketOutput {
            compacted_faces,
            bucket_ranges,
            triangle_indirect_arguments,
            line_indirect_arguments,
        })
    }

    /// Allocate the retained source-face root topology pass. Subject rows are
    /// an extraction concern and may be updated without replacing geometry;
    /// packed edge LOD, S3 permutation, and shared corner maxima remain device
    /// outputs owned by the current resident classifier epoch.
    pub fn upload_resident_root_topology_scene(
        &self,
        model: &LodClassifierModel,
        face_subject_rows: &[u32],
        subject_count: u32,
    ) -> Result<ResidentRootTopologyScene, LodWebGpuError> {
        let face_count = u32::try_from(model.prepared.residency.num_faces)
            .map_err(|_| LodWebGpuError::Payload("resident root faces exceed u32".into()))?;
        let vertex_count = model.prepared.residency.num_vertices;
        if face_subject_rows.len() != face_count as usize {
            return Err(LodWebGpuError::Payload(format!(
                "resident root topology has {} subject rows; expected {face_count}",
                face_subject_rows.len(),
            )));
        }
        if subject_count == 0 || face_subject_rows.iter().any(|&row| row >= subject_count) {
            return Err(LodWebGpuError::Payload(format!(
                "resident root topology subject rows exceed the {subject_count}-row domain",
            )));
        }
        if let Some(last_face) = face_count.checked_sub(1) {
            if last_face as f32 as u32 != last_face {
                return Err(LodWebGpuError::Payload(
                    "resident root face IDs exceed exact f32 encoding".to_string(),
                ));
            }
        }

        let uniform_words = [face_count, vertex_count, subject_count, 0];
        let uniform = buffer_init_or_zero(
            &self.device,
            "resident root topology uniform",
            bytemuck::cast_slice(&uniform_words),
            wgpu::BufferUsages::UNIFORM,
        );
        let face_subject_rows = buffer_init_or_zero(
            &self.device,
            "resident root face subject rows",
            bytemuck::cast_slice(face_subject_rows),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let vertex_lod_max = gpu_buffer(
            &self.device,
            "resident root vertex LOD maxima",
            u64::from(vertex_count) * PACKED_RECORD_BYTES,
            wgpu::BufferUsages::STORAGE,
        );
        let topology_records = gpu_buffer(
            &self.device,
            "resident root topology records",
            u64::from(face_count) * PATCH_TOPOLOGY_RECORD_BYTES,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        let clear_layout = self
            .resident_root_vertex_clear_pipeline
            .get_bind_group_layout(0);
        let clear_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("resident root vertex LOD clear bindings"),
            layout: &clear_layout,
            entries: &[bind(0, &uniform), bind(4, &vertex_lod_max)],
        });
        let accumulate_layout = self
            .resident_root_vertex_accumulate_pipeline
            .get_bind_group_layout(0);
        let accumulate_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("resident root vertex LOD accumulation bindings"),
            layout: &accumulate_layout,
            entries: &[
                bind(0, &uniform),
                bind(1, &model.resident.packed_records),
                bind(2, &model.faces),
                bind(4, &vertex_lod_max),
            ],
        });
        let emit_layout = self
            .resident_root_topology_pipeline
            .get_bind_group_layout(0);
        let emit_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("resident root topology emission bindings"),
            layout: &emit_layout,
            entries: &[
                bind(0, &uniform),
                bind(1, &model.resident.packed_records),
                bind(2, &model.faces),
                bind(3, &face_subject_rows),
                bind(4, &vertex_lod_max),
                bind(5, &topology_records),
            ],
        });
        Ok(ResidentRootTopologyScene {
            model_identity: model.identity,
            face_count,
            vertex_count,
            subject_count,
            _uniform: uniform,
            face_subject_rows,
            _vertex_lod_max: vertex_lod_max,
            topology_records,
            clear_bind_group,
            accumulate_bind_group,
            emit_bind_group,
        })
    }

    pub fn write_resident_root_subject_rows(
        &self,
        scene: &ResidentRootTopologyScene,
        face_subject_rows: &[u32],
    ) -> Result<(), LodWebGpuError> {
        if face_subject_rows.len() != scene.face_count as usize
            || face_subject_rows
                .iter()
                .any(|&row| row >= scene.subject_count)
        {
            return Err(LodWebGpuError::Payload(
                "resident root subject-row update changed the retained domain".to_string(),
            ));
        }
        self.queue.write_buffer(
            &scene.face_subject_rows,
            0,
            bytemuck::cast_slice(face_subject_rows),
        );
        Ok(())
    }

    /// Append vertex-maximum reconstruction and exact root topology emission
    /// to an application-owned encoder. The output remains indexed by source
    /// face and is suitable for direct patch preparation.
    pub fn encode_resident_root_topology(
        &self,
        scene: &ResidentRootTopologyScene,
        resident: &DeviceResidentLod<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<(), LodWebGpuError> {
        if scene.model_identity != resident.model_identity {
            return Err(LodWebGpuError::Payload(
                "resident root topology belongs to a different WebGPU model".to_string(),
            ));
        }
        if scene.face_count != resident.face_count {
            return Err(LodWebGpuError::Payload(format!(
                "resident root topology has {} faces; classifier result has {}",
                scene.face_count, resident.face_count,
            )));
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("quilting resident root vertex LOD clear"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.resident_root_vertex_clear_pipeline);
            pass.set_bind_group(0, &scene.clear_bind_group, &[]);
            pass.dispatch_workgroups(scene.vertex_count.div_ceil(LOD_WORKGROUP_SIZE), 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("quilting resident root vertex LOD accumulation"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.resident_root_vertex_accumulate_pipeline);
            pass.set_bind_group(0, &scene.accumulate_bind_group, &[]);
            pass.dispatch_workgroups(scene.face_count.div_ceil(LOD_WORKGROUP_SIZE), 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("quilting resident root topology emission"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.resident_root_topology_pipeline);
            pass.set_bind_group(0, &scene.emit_bind_group, &[]);
            pass.dispatch_workgroups(scene.face_count.div_ceil(LOD_WORKGROUP_SIZE), 1, 1);
        }
        Ok(())
    }

    /// Diagnostic-only projection of the same source-indexed topology buffer.
    pub async fn resident_root_topology_for_diagnostics(
        &self,
        scene: &ResidentRootTopologyScene,
        resident: &DeviceResidentLod<'_>,
    ) -> Result<Vec<[u32; 12]>, LodWebGpuError> {
        let bytes = u64::from(scene.face_count) * PATCH_TOPOLOGY_RECORD_BYTES;
        let readback = gpu_buffer(
            &self.device,
            "resident root topology diagnostic readback",
            bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("resident root topology diagnostic encoder"),
            });
        self.encode_resident_root_topology(scene, resident, &mut encoder)?;
        encoder.copy_buffer_to_buffer(&scene.topology_records, 0, &readback, 0, bytes);
        self.queue.submit([encoder.finish()]);
        words_to_twelve_records(self.readback_words(&readback, bytes).await?)
    }

    pub(super) async fn run_resident_geometry_bucket_conformance(
        &self,
    ) -> Result<usize, LodWebGpuError> {
        let atlas_keys = [[1, 1, 1], [1, 1, 2], [1, 2, 4]];
        let atlas_lookup =
            prepare_lod_atlas_lookup(atlas_keys).map_err(LodWebGpuError::Conformance)?;
        let atlas = self.upload_packed_patch_atlas(
            &[
                1, 2, 4, 6, 3, 0, 6, 1, 1, 1, 0, 3, 0, 6, 1, 1, 2, 3, 3, 0, 6,
            ],
            &[1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            &[0, 1, 2, 0, 1, 2, 0, 1, 2],
            &[0, 1, 1, 2, 2, 0],
        )?;
        let face_count = 137usize;
        let permutations = [0, 1, 2, 4, 3, 5];
        let exponents = [[0, 0, 0], [0, 0, 1], [0, 1, 2]];
        let packed = (0..face_count)
            .map(|face| {
                let atlas_index = face % atlas_keys.len();
                pack_lod_classification(
                    exponents[atlas_index],
                    permutations[face % permutations.len()],
                    (face % 11 != 0).then_some(atlas_index as u32),
                    face as u8,
                )
                .map_err(LodWebGpuError::Conformance)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let packed_buffer = buffer_init_or_zero(
            &self.device,
            "resident geometry bucket conformance records",
            bytemuck::cast_slice(&packed),
            wgpu::BufferUsages::STORAGE,
        );
        let identity = self.allocate_model_identity()?;
        let transform = |orientation_sign| RenderEntityTransform {
            mobius: identity_mobius(),
            orientation_sign,
            euclidean_model: identity_matrix(),
            euclidean_normal: identity_matrix(),
        };
        let draw_domain_words = ResidentRootDrawDomains {
            domains: vec![
                ResidentRootDrawDomain {
                    material_index: 3,
                    render_node_index: 1,
                    pbr_class: PbrDrawClass::Opaque,
                    transform: transform(1),
                    enabled: true,
                },
                ResidentRootDrawDomain {
                    material_index: 500_000,
                    render_node_index: 8,
                    pbr_class: PbrDrawClass::Blend,
                    transform: transform(-1),
                    enabled: true,
                },
                ResidentRootDrawDomain {
                    material_index: 900_000,
                    render_node_index: 13,
                    pbr_class: PbrDrawClass::Transmission,
                    transform: transform(1),
                    enabled: false,
                },
            ],
            face_domain_rows: (0..face_count).map(|face| (face % 3) as u32).collect(),
        };
        let draw_domains =
            self.upload_resident_root_draw_domains(identity, draw_domain_words.clone())?;
        let scene = self.upload_resident_geometry_bucket_scene_for_records(
            identity,
            face_count as u32,
            &atlas_lookup.keys,
            &packed_buffer,
            &atlas,
            &draw_domains,
        )?;
        let resident = DeviceResidentLod {
            packed_records: &packed_buffer,
            model_identity: identity,
            face_count: face_count as u32,
            classification_epoch: 1,
            grading: FaceLodGrading::TwoToOne,
        };
        let foreign_resident = DeviceResidentLod {
            packed_records: &packed_buffer,
            model_identity: self.allocate_model_identity()?,
            face_count: face_count as u32,
            classification_epoch: 1,
            grading: FaceLodGrading::TwoToOne,
        };
        let mut rejection_encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("foreign resident geometry bucket rejection"),
                });
        if self
            .encode_resident_geometry_buckets(&scene, &foreign_resident, &mut rejection_encoder)
            .is_ok()
        {
            return Err(LodWebGpuError::Conformance(
                "resident geometry accepted a foreign model result".to_string(),
            ));
        }
        let invalid_padding = vec![u32::MAX; scene.eligibility_word_count as usize];
        if self
            .write_resident_root_eligibility_bits(&scene, &invalid_padding)
            .is_ok()
        {
            return Err(LodWebGpuError::Conformance(
                "resident geometry accepted nonzero eligibility padding".to_string(),
            ));
        }
        let atlas_draws = [[0, 3, 0, 6], [3, 3, 0, 6], [6, 3, 0, 6]];
        let suppression_cases = [
            vec![2, 64, 65, 130],
            (0..face_count as u32)
                .filter(|face| face % 3 == 0)
                .collect::<Vec<_>>(),
        ];
        let mut compared_words = 0usize;
        for suppressed in suppression_cases {
            let eligibility = pack_wgsl_root_eligibility_bits(face_count, &suppressed)
                .map_err(LodWebGpuError::Conformance)?;
            let expected = wgsl_resident_geometry_bucket_oracle_words_with_domains(
                &packed,
                &atlas_draws,
                &eligibility,
                &draw_domain_words,
            )
            .map_err(LodWebGpuError::Conformance)?;
            let actual = self
                .resident_geometry_buckets_for_diagnostics(&scene, &resident, &suppressed)
                .await?;
            if actual.compacted_faces != expected.compacted_faces
                || actual.bucket_ranges != expected.bucket_ranges
                || actual.triangle_indirect_arguments != expected.triangle_indirect_arguments
                || actual.line_indirect_arguments != expected.line_indirect_arguments
            {
                return Err(LodWebGpuError::Conformance(format!(
                    "resident geometry bucket mismatch for suppression {suppressed:?}: expected {expected:?}, got {actual:?}",
                )));
            }
            compared_words += actual.compacted_faces.len()
                + actual.bucket_ranges.len() * 5
                + actual.triangle_indirect_arguments.len() * 5
                + actual.line_indirect_arguments.len() * 5;
        }
        Ok(compared_words)
    }
}
