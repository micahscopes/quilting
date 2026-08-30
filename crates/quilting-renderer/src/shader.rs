//! Shader compilation: WGSL -> naga -> GLSL ES 300 -> glow GL programs.
//!
//! Uses quilting-shaders to compile WGSL modules with naga-oil imports,
//! then creates OpenGL shader programs via glow.

use glow::HasContext;
use quilting_core::render_pipeline::{
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingKind, GraphicsProgramDescriptor,
    PipelineLayoutDescriptor, SamplerBindingKind, ShaderDefinitionValue, ShaderModuleDescriptor,
    ShaderStage, ShaderTarget, ShaderVisibility, TextureSampleKind, TextureViewDimension,
};

/// Canonical shader-group coordinates, re-exported beside the WebGL lowering.
pub use quilting_shaders::primary_graphics_bind_group::{
    ENTITY_BATCH as ENTITY_BATCH_BIND_GROUP, FRAME_POSE as FRAME_POSE_BIND_GROUP,
    MATERIAL_STYLE as MATERIAL_STYLE_BIND_GROUP, PASS_RESOURCE as PASS_RESOURCE_BIND_GROUP,
};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use crate::memo::{DeviceMemo, DeviceMemoDiagnostics};

#[cfg(target_arch = "wasm32")]
fn log_info(msg: &str) {
    web_sys::console::info_1(&msg.into());
}
#[cfg(not(target_arch = "wasm32"))]
fn log_info(msg: &str) {
    eprintln!("{}", msg);
}

/// All compiled shader programs for the quilting rendering pipeline.
pub struct Programs {
    pub matcap: glow::Program,
    pub wire: glow::Program,
    pub normals: glow::Program,
    pub pbr: glow::Program,
    pub stretch: glow::Program,
    pub pick: glow::Program,
}

/// Uniform block binding points.
/// Naga emits `layout(std140) uniform BlockName { ... }` blocks.
/// We bind these to UBO binding points.
pub const VERTEX_UNIFORMS_BINDING: u32 = 0;
pub const WIRE_UNIFORMS_BINDING: u32 = 1;
pub const PBR_UNIFORMS_BINDING: u32 = 2;
pub const JOINT_MATRICES_BINDING: u32 = 4;

/// Complete WebGPU-facing policies for opaque resources that WebGL represents
/// through post-link sampler uniforms.
const UNFILTERABLE_TEXTURE_2D: BindingKind = BindingKind::Texture {
    sample_kind: TextureSampleKind::FloatUnfilterable,
    view_dimension: TextureViewDimension::D2,
    multisampled: false,
};
const FILTERABLE_TEXTURE_2D: BindingKind = BindingKind::Texture {
    sample_kind: TextureSampleKind::FloatFilterable,
    view_dimension: TextureViewDimension::D2,
    multisampled: false,
};
const FILTERABLE_TEXTURE_CUBE: BindingKind = BindingKind::Texture {
    sample_kind: TextureSampleKind::FloatFilterable,
    view_dimension: TextureViewDimension::Cube,
    multisampled: false,
};
const FILTERING_SAMPLER: BindingKind = BindingKind::Sampler(SamplerBindingKind::Filtering);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WebGlUniformBlockBinding {
    pub name: Arc<str>,
    pub binding_point: u32,
    pub source_name: Arc<str>,
    pub source: WebGlBindingSite,
    pub minimum_size: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WebGlBindingSite {
    pub group: u32,
    pub binding: u32,
    pub stage: ShaderStage,
}

impl WebGlBindingSite {
    pub const fn new(group: u32, binding: u32, stage: ShaderStage) -> Self {
        Self {
            group,
            binding,
            stage,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WebGlSamplerBinding {
    pub name: Arc<str>,
    pub texture_unit: u32,
    pub source_name: Arc<str>,
    pub source: WebGlBindingSite,
    pub source_kind: BindingKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WebGlStageBindingConflict {
    pub group: u32,
    pub binding: u32,
    pub stages: Vec<ShaderStage>,
}

/// Immutable WebGL mutations applied immediately after a program links.
///
/// These assignments are part of program cache identity: two otherwise equal
/// shader programs configured with different UBO points or texture units must
/// never alias the same mutable `WebGlProgram`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WebGlBindingPlan {
    uniform_blocks: Vec<WebGlUniformBlockBinding>,
    samplers: Vec<WebGlSamplerBinding>,
}

impl WebGlBindingPlan {
    pub fn new(
        mut uniform_blocks: Vec<WebGlUniformBlockBinding>,
        mut samplers: Vec<WebGlSamplerBinding>,
    ) -> Result<Self, String> {
        if uniform_blocks
            .iter()
            .any(|binding| binding.name.is_empty() || binding.source_name.is_empty())
            || samplers
                .iter()
                .any(|binding| binding.name.is_empty() || binding.source_name.is_empty())
        {
            return Err("WebGL emitted and source binding names must not be empty".into());
        }
        if uniform_blocks
            .iter()
            .any(|binding| binding.minimum_size == 0)
        {
            return Err("WebGL uniform-block minimum sizes must be nonzero".into());
        }
        if samplers.iter().any(|binding| {
            !matches!(
                binding.source_kind,
                BindingKind::Texture { .. } | BindingKind::Sampler(_)
            )
        }) {
            return Err("WebGL opaque bindings must be sampled textures or samplers".into());
        }
        if uniform_blocks
            .iter()
            .any(|binding| binding.source.stage == ShaderStage::Compute)
            || samplers
                .iter()
                .any(|binding| binding.source.stage == ShaderStage::Compute)
        {
            return Err("WebGL graphics bindings cannot use the compute stage".into());
        }
        uniform_blocks.sort_by(|left, right| left.name.cmp(&right.name));
        samplers.sort_by(|left, right| left.name.cmp(&right.name));
        if uniform_blocks
            .windows(2)
            .any(|pair| pair[0].name == pair[1].name)
        {
            return Err("duplicate WebGL uniform-block binding".into());
        }
        if samplers.windows(2).any(|pair| pair[0].name == pair[1].name) {
            return Err("duplicate WebGL sampler binding".into());
        }
        let mut source_sites = BTreeSet::new();
        if !uniform_blocks
            .iter()
            .map(|binding| binding.source)
            .chain(samplers.iter().map(|binding| binding.source))
            .all(|source| source_sites.insert(source))
        {
            return Err("duplicate stage-specific WebGL source binding".into());
        }
        Ok(Self {
            uniform_blocks,
            samplers,
        })
    }

    pub fn uniform_blocks(&self) -> &[WebGlUniformBlockBinding] {
        &self.uniform_blocks
    }

    pub fn samplers(&self) -> &[WebGlSamplerBinding] {
        &self.samplers
    }

    /// Canonical compiler-input interface represented by this WebGL plan for
    /// one graphics stage. This is independent of emitted GLSL identifiers and
    /// can be compared directly with composed Naga entry-point reflection.
    pub fn source_bindings_for_stage(
        &self,
        stage: ShaderStage,
    ) -> Vec<quilting_shaders::ReflectedEntryBinding> {
        let mut bindings = self
            .uniform_blocks
            .iter()
            .filter(|binding| binding.source.stage == stage)
            .map(|binding| quilting_shaders::ReflectedEntryBinding {
                group: binding.source.group,
                binding: binding.source.binding,
                name: binding.source_name.to_string(),
                kind: quilting_shaders::ReflectedBindingKind::UniformBuffer,
            })
            .chain(
                self.samplers
                    .iter()
                    .filter(|binding| binding.source.stage == stage)
                    .map(|binding| quilting_shaders::ReflectedEntryBinding {
                        group: binding.source.group,
                        binding: binding.source.binding,
                        name: binding.source_name.to_string(),
                        kind: match binding.source_kind {
                            BindingKind::Texture { .. } => {
                                quilting_shaders::ReflectedBindingKind::SampledTexture
                            }
                            BindingKind::Sampler(_) => {
                                quilting_shaders::ReflectedBindingKind::Sampler
                            }
                            _ => unreachable!("opaque binding kind was validated at construction"),
                        },
                    }),
            )
            .collect::<Vec<_>>();
        bindings.sort();
        bindings
    }

    /// Return legacy binding coordinates that name distinct resources in more
    /// than one shader stage. WebGL can initialize those resources by their
    /// emitted GLSL names, but a WebGPU pipeline layout has one resource per
    /// `(group, binding)` and therefore cannot lower this plan losslessly.
    pub fn cross_stage_slot_conflicts(&self) -> Vec<WebGlStageBindingConflict> {
        let mut stages_by_slot = BTreeMap::<(u32, u32), BTreeSet<ShaderStage>>::new();
        for source in self
            .uniform_blocks
            .iter()
            .map(|binding| binding.source)
            .chain(self.samplers.iter().map(|binding| binding.source))
        {
            stages_by_slot
                .entry((source.group, source.binding))
                .or_default()
                .insert(source.stage);
        }
        stages_by_slot
            .into_iter()
            .filter_map(|((group, binding), stages)| {
                (stages.len() > 1).then(|| WebGlStageBindingConflict {
                    group,
                    binding,
                    stages: stages.into_iter().collect(),
                })
            })
            .collect()
    }

    /// Complete portable layout for the exact resources reachable by one
    /// linked graphics program. Missing group indices become explicit empty
    /// layouts because WebGPU addresses bind groups positionally.
    pub fn portable_layout(&self) -> Result<PipelineLayoutDescriptor, String> {
        let conflicts = self.cross_stage_slot_conflicts();
        if !conflicts.is_empty() {
            return Err(format!(
                "WebGL binding plan has {} cross-stage slot conflict(s)",
                conflicts.len()
            ));
        }
        let mut entries_by_group = BTreeMap::<u32, Vec<BindGroupLayoutEntry>>::new();
        for binding in &self.uniform_blocks {
            entries_by_group
                .entry(binding.source.group)
                .or_default()
                .push(BindGroupLayoutEntry {
                    binding: binding.source.binding,
                    visibility: shader_visibility(binding.source.stage),
                    kind: BindingKind::UniformBuffer {
                        dynamic_offset: false,
                        minimum_size: binding.minimum_size,
                    },
                });
        }
        for binding in &self.samplers {
            entries_by_group
                .entry(binding.source.group)
                .or_default()
                .push(BindGroupLayoutEntry {
                    binding: binding.source.binding,
                    visibility: shader_visibility(binding.source.stage),
                    kind: binding.source_kind,
                });
        }
        let Some(max_group) = entries_by_group.keys().next_back().copied() else {
            return PipelineLayoutDescriptor::new(Vec::new()).map_err(|error| error.to_string());
        };
        let groups = (0..=max_group)
            .map(|group| {
                BindGroupLayoutDescriptor::new(
                    group,
                    entries_by_group.remove(&group).unwrap_or_default(),
                )
                .map_err(|error| error.to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        PipelineLayoutDescriptor::new(groups).map_err(|error| error.to_string())
    }
}

const fn shader_visibility(stage: ShaderStage) -> ShaderVisibility {
    match stage {
        ShaderStage::Vertex => ShaderVisibility::VERTEX,
        ShaderStage::Fragment => ShaderVisibility::FRAGMENT,
        ShaderStage::Compute => ShaderVisibility::COMPUTE,
    }
}

/// Complete key for one linked and initialized WebGL program.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WebGlProgramKey {
    program: GraphicsProgramDescriptor,
    bindings: WebGlBindingPlan,
}

impl WebGlProgramKey {
    pub fn new(program: GraphicsProgramDescriptor, bindings: WebGlBindingPlan) -> Self {
        Self { program, bindings }
    }

    pub fn program(&self) -> &GraphicsProgramDescriptor {
        &self.program
    }

    pub fn bindings(&self) -> &WebGlBindingPlan {
        &self.bindings
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WebGlProgramMemoDiagnostics {
    pub device_epoch: u64,
    pub shaders: DeviceMemoDiagnostics,
    pub programs: DeviceMemoDiagnostics,
}

/// Sole owner of descriptor-lowered WebGL shader/program handles.
/// `Programs` is only a compact non-owning view of the primary render subset.
pub struct WebGlProgramMemo {
    shaders: DeviceMemo<ShaderModuleDescriptor, glow::Shader>,
    programs: DeviceMemo<WebGlProgramKey, glow::Program>,
}

impl WebGlProgramMemo {
    pub fn new(device_epoch: u64) -> Self {
        Self {
            shaders: DeviceMemo::new(device_epoch),
            programs: DeviceMemo::new(device_epoch),
        }
    }

    pub fn device_epoch(&self) -> u64 {
        debug_assert_eq!(self.shaders.device_epoch(), self.programs.device_epoch());
        self.programs.device_epoch()
    }

    pub fn diagnostics(&self) -> WebGlProgramMemoDiagnostics {
        WebGlProgramMemoDiagnostics {
            device_epoch: self.device_epoch(),
            shaders: self.shaders.diagnostics(),
            programs: self.programs.diagnostics(),
        }
    }

    /// Delete every linked program before deleting the cached shader modules.
    pub fn destroy(&mut self, gl: &glow::Context) {
        unsafe {
            for program in self.programs.drain() {
                gl.delete_program(program);
            }
            for shader in self.shaders.drain() {
                gl.delete_shader(shader);
            }
        }
    }
}

/// Compiled GLSL source for a vertex/fragment pair.
pub struct CompiledGlsl {
    pub vertex: String,
    pub fragment: String,
}

/// Fragment shader modes that quilting-shaders supports.
pub(crate) const FRAGMENT_MODES: &[&str] =
    &["matcap", "wire", "normals", "pbr", "stretch", "pick"];

fn fragment_source_and_entry(mode: &str) -> Result<(&'static str, &'static str), String> {
    match mode {
        "matcap" => Ok((quilting_shaders::sources::FRAG_MATCAP, "fs_matcap")),
        "wire" => Ok((quilting_shaders::sources::FRAG_WIRE, "fs_wire")),
        "normals" => Ok((quilting_shaders::sources::FRAG_NORMALS, "fs_normals")),
        "pbr" => Ok((quilting_shaders::sources::FRAG_PBR, "fs_pbr")),
        "stretch" => Ok((quilting_shaders::sources::FRAG_STRETCH, "fs_stretch")),
        "pick" => Ok((quilting_shaders::sources::FRAG_PICK, "fs_pick")),
        _ => Err(format!("unknown fragment mode: {mode}")),
    }
}

pub(crate) fn vertex_binding_entries(
    entry_point: &str,
) -> Result<(Vec<WebGlUniformBlockBinding>, Vec<WebGlSamplerBinding>), String> {
    let mut uniform_blocks = vec![
        WebGlUniformBlockBinding {
            name: "Uniforms_block_0Vertex".into(),
            binding_point: VERTEX_UNIFORMS_BINDING,
            source_name: "u".into(),
            source: WebGlBindingSite::new(ENTITY_BATCH_BIND_GROUP, 0, ShaderStage::Vertex),
            minimum_size: 352,
        },
        WebGlUniformBlockBinding {
            name: "JointMatrices_block_1Vertex".into(),
            binding_point: JOINT_MATRICES_BINDING,
            source_name: "joints".into(),
            source: WebGlBindingSite::new(FRAME_POSE_BIND_GROUP, 0, ShaderStage::Vertex),
            minimum_size: 8_464,
        },
    ];
    let mut samplers = vec![
        WebGlSamplerBinding {
            name: "_group_0_binding_1_vs".into(),
            texture_unit: SKINNING_TEX_UNIT,
            source_name: "skinning_tex".into(),
            source: WebGlBindingSite::new(FRAME_POSE_BIND_GROUP, 1, ShaderStage::Vertex),
            source_kind: UNFILTERABLE_TEXTURE_2D,
        },
        WebGlSamplerBinding {
            name: "_group_0_binding_2_vs".into(),
            texture_unit: MORPH_TEX_UNIT,
            source_name: "morph_tex".into(),
            source: WebGlBindingSite::new(FRAME_POSE_BIND_GROUP, 2, ShaderStage::Vertex),
            source_kind: UNFILTERABLE_TEXTURE_2D,
        },
        WebGlSamplerBinding {
            name: "_group_0_binding_3_vs".into(),
            texture_unit: FACE_DATA_TEX_UNIT,
            source_name: "face_data_tex".into(),
            source: WebGlBindingSite::new(FRAME_POSE_BIND_GROUP, 3, ShaderStage::Vertex),
            source_kind: UNFILTERABLE_TEXTURE_2D,
        },
        WebGlSamplerBinding {
            name: "_group_0_binding_4_vs".into(),
            texture_unit: SUPPRESSED_FACE_TEX_UNIT,
            source_name: "suppressed_face_tex".into(),
            source: WebGlBindingSite::new(FRAME_POSE_BIND_GROUP, 4, ShaderStage::Vertex),
            source_kind: UNFILTERABLE_TEXTURE_2D,
        },
    ];
    match entry_point {
        "vs_main" => {
            samplers.retain(|binding| {
                matches!(binding.source_name.as_ref(), "skinning_tex" | "morph_tex")
            });
        }
        "prepare_patches" => {
            samplers.retain(|binding| binding.source_name.as_ref() != "suppressed_face_tex");
        }
        "classify_patch_visibility" => {
            uniform_blocks.retain(|binding| binding.source_name.as_ref() == "u");
            samplers.retain(|binding| binding.source_name.as_ref() == "suppressed_face_tex");
        }
        _ => return Err(format!("unknown WebGL vertex entry point: {entry_point}")),
    }
    Ok((uniform_blocks, samplers))
}

fn primary_binding_plan(mode: &str) -> Result<WebGlBindingPlan, String> {
    let (mut uniform_blocks, mut samplers) = vertex_binding_entries("vs_main")?;
    match mode {
        "matcap" => uniform_blocks.push(WebGlUniformBlockBinding {
            name: "MatcapUniforms_block_0Fragment".into(),
            binding_point: WIRE_UNIFORMS_BINDING,
            source_name: "matcap_u".into(),
            source: WebGlBindingSite::new(MATERIAL_STYLE_BIND_GROUP, 0, ShaderStage::Fragment),
            minimum_size: 16,
        }),
        "wire" => uniform_blocks.push(WebGlUniformBlockBinding {
            name: "WireUniforms_block_0Fragment".into(),
            binding_point: WIRE_UNIFORMS_BINDING,
            source_name: "wire".into(),
            source: WebGlBindingSite::new(MATERIAL_STYLE_BIND_GROUP, 0, ShaderStage::Fragment),
            minimum_size: 16,
        }),
        "pbr" => uniform_blocks.push(WebGlUniformBlockBinding {
            name: "PbrUniforms_block_0Fragment".into(),
            binding_point: PBR_UNIFORMS_BINDING,
            source_name: "pbr".into(),
            source: WebGlBindingSite::new(MATERIAL_STYLE_BIND_GROUP, 0, ShaderStage::Fragment),
            minimum_size: 256,
        }),
        "normals" | "stretch" | "pick" => {}
        _ => return Err(format!("unknown fragment mode: {mode}")),
    }

    if mode == "pbr" {
        for (group, binding, texture_unit, source_name, source_kind) in [
            (
                MATERIAL_STYLE_BIND_GROUP,
                1,
                0,
                "base_color_tex",
                FILTERABLE_TEXTURE_2D,
            ),
            (
                MATERIAL_STYLE_BIND_GROUP,
                2,
                0,
                "base_color_sampler",
                FILTERING_SAMPLER,
            ),
            (
                MATERIAL_STYLE_BIND_GROUP,
                3,
                1,
                "metallic_roughness_tex",
                FILTERABLE_TEXTURE_2D,
            ),
            (
                MATERIAL_STYLE_BIND_GROUP,
                4,
                1,
                "metallic_roughness_sampler",
                FILTERING_SAMPLER,
            ),
            (
                MATERIAL_STYLE_BIND_GROUP,
                5,
                2,
                "normal_tex",
                FILTERABLE_TEXTURE_2D,
            ),
            (
                MATERIAL_STYLE_BIND_GROUP,
                6,
                2,
                "normal_sampler",
                FILTERING_SAMPLER,
            ),
            (
                MATERIAL_STYLE_BIND_GROUP,
                7,
                3,
                "emissive_tex",
                FILTERABLE_TEXTURE_2D,
            ),
            (
                MATERIAL_STYLE_BIND_GROUP,
                8,
                3,
                "emissive_sampler",
                FILTERING_SAMPLER,
            ),
            (
                MATERIAL_STYLE_BIND_GROUP,
                9,
                4,
                "occlusion_tex",
                FILTERABLE_TEXTURE_2D,
            ),
            (
                MATERIAL_STYLE_BIND_GROUP,
                10,
                4,
                "occlusion_sampler",
                FILTERING_SAMPLER,
            ),
            (
                MATERIAL_STYLE_BIND_GROUP,
                13,
                5,
                "env_prefiltered",
                FILTERABLE_TEXTURE_CUBE,
            ),
            (
                MATERIAL_STYLE_BIND_GROUP,
                14,
                5,
                "env_prefiltered_sampler",
                FILTERING_SAMPLER,
            ),
            (
                MATERIAL_STYLE_BIND_GROUP,
                15,
                6,
                "env_irradiance",
                FILTERABLE_TEXTURE_CUBE,
            ),
            (
                MATERIAL_STYLE_BIND_GROUP,
                16,
                6,
                "env_irradiance_sampler",
                FILTERING_SAMPLER,
            ),
            (
                PASS_RESOURCE_BIND_GROUP,
                0,
                8,
                "scene_color_tex",
                FILTERABLE_TEXTURE_2D,
            ),
            (
                PASS_RESOURCE_BIND_GROUP,
                1,
                8,
                "scene_color_sampler",
                FILTERING_SAMPLER,
            ),
            (
                PASS_RESOURCE_BIND_GROUP,
                4,
                10,
                "transmission_tex",
                FILTERABLE_TEXTURE_2D,
            ),
            (
                PASS_RESOURCE_BIND_GROUP,
                5,
                10,
                "transmission_tex_sampler",
                FILTERING_SAMPLER,
            ),
        ] {
            samplers.push(WebGlSamplerBinding {
                name: format!("_group_{group}_binding_{binding}_fs").into(),
                texture_unit,
                source_name: source_name.into(),
                source: WebGlBindingSite::new(group, binding, ShaderStage::Fragment),
                source_kind,
            });
        }
    }
    WebGlBindingPlan::new(uniform_blocks, samplers)
}

/// Pure descriptors for the six primary render programs. Application-level
/// auxiliary descriptors may be resolved through the same memo separately.
pub fn primary_program_descriptors() -> Result<Vec<(&'static str, WebGlProgramKey)>, String> {
    let compiler_catalog_revision = quilting_shaders::compiler_catalog_revision();
    let vertex = ShaderModuleDescriptor::new(
        "quilting primary vertex",
        quilting_shaders::sources::VERTEX_MAIN,
        Arc::clone(&compiler_catalog_revision),
        ShaderStage::Vertex,
        "vs_main",
        ShaderTarget::GlslEs300 {
            adjust_coordinate_space: false,
        },
        Vec::new(),
    )
    .map_err(|error| error.to_string())?;

    FRAGMENT_MODES
        .iter()
        .map(|&mode| {
            let (source, entry_point) = fragment_source_and_entry(mode)?;
            let fragment = ShaderModuleDescriptor::new(
                format!("quilting {mode} fragment"),
                source,
                Arc::clone(&compiler_catalog_revision),
                ShaderStage::Fragment,
                entry_point,
                ShaderTarget::GlslEs300 {
                    adjust_coordinate_space: false,
                },
                Vec::new(),
            )
            .map_err(|error| error.to_string())?;
            let program = GraphicsProgramDescriptor::new(
                vertex.clone(),
                Some(fragment),
                Vec::new(),
            )
            .map_err(|error| error.to_string())?;
            Ok((mode, WebGlProgramKey::new(program, primary_binding_plan(mode)?)))
        })
        .collect()
}

fn compile_module_glsl(descriptor: &ShaderModuleDescriptor) -> Result<String, String> {
    let ShaderTarget::GlslEs300 {
        adjust_coordinate_space,
    } = descriptor.target()
    else {
        return Err(format!(
            "WebGL cannot lower {:?} shader target for '{}'",
            descriptor.target(),
            descriptor.label()
        ));
    };
    let stage = match descriptor.stage() {
        ShaderStage::Vertex => quilting_shaders::EntryPointStage::Vertex,
        ShaderStage::Fragment => quilting_shaders::EntryPointStage::Fragment,
        ShaderStage::Compute => {
            return Err("WebGL2 does not support compute shader modules".into())
        }
    };
    let definitions = descriptor
        .definitions()
        .iter()
        .map(|definition| {
            let value = match definition.value {
                ShaderDefinitionValue::Bool(value) => {
                    quilting_shaders::ShaderDefValue::Bool(value)
                }
                ShaderDefinitionValue::I32(value) => {
                    quilting_shaders::ShaderDefValue::Int(value)
                }
                ShaderDefinitionValue::U32(value) => {
                    quilting_shaders::ShaderDefValue::UInt(value)
                }
            };
            (definition.name.to_string(), value)
        })
        .collect::<HashMap<_, _>>();
    let module = quilting_shaders::compile_shader(descriptor.source(), definitions)
        .map_err(|error| format!("{} WGSL: {error}", descriptor.label()))?;
    quilting_shaders::emit_graphics_entry_glsl(
        &module,
        stage,
        descriptor.entry_point(),
        adjust_coordinate_space,
    )
    .map_err(|error| format!("{} GLSL: {error}", descriptor.label()))
}

/// Compile all WGSL shaders to GLSL via quilting-shaders (naga).
/// Uses the "native" emission path (no Y-flip / Z-remap).
/// Returns raw GLSL strings for each program.
pub fn compile_all_glsl() -> Result<Vec<(&'static str, CompiledGlsl)>, String> {
    let descriptors = primary_program_descriptors()?;
    let shared_vertex = descriptors
        .first()
        .ok_or("primary shader catalog is empty")?
        .1
        .program()
        .vertex();
    let vertex = compile_module_glsl(shared_vertex)?;

    descriptors
        .into_iter()
        .map(|(mode, key)| {
            let fragment = compile_module_glsl(
                key.program()
                    .fragment()
                    .ok_or_else(|| format!("primary program '{mode}' has no fragment shader"))?,
            )?;
            Ok((mode, CompiledGlsl {
                vertex: vertex.clone(),
                fragment,
            }))
        })
        .collect()
}

/// Compile a single GL shader (vertex or fragment) from GLSL source.
fn compile_gl_shader(
    gl: &glow::Context,
    shader_type: u32,
    source: &str,
) -> Result<glow::Shader, String> {
    unsafe {
        let shader = gl.create_shader(shader_type)
            .map_err(|e| format!("create_shader: {e}"))?;
        gl.shader_source(shader, source);
        gl.compile_shader(shader);

        if !gl.get_shader_compile_status(shader) {
            let log = gl.get_shader_info_log(shader);
            gl.delete_shader(shader);
            return Err(format!("shader compile error:\n{log}\n\nSource:\n{source}"));
        }

        Ok(shader)
    }
}

/// Link a vertex + fragment shader into a GL program.
fn link_owned_program(
    gl: &glow::Context,
    vert: glow::Shader,
    frag: glow::Shader,
    transform_feedback_varyings: Option<&[&str]>,
) -> Result<glow::Program, String> {
    unsafe {
        let program = gl.create_program()
            .map_err(|e| {
                gl.delete_shader(vert);
                gl.delete_shader(frag);
                format!("create_program: {e}")
            })?;
        gl.attach_shader(program, vert);
        gl.attach_shader(program, frag);
        if let Some(varyings) = transform_feedback_varyings {
            gl.transform_feedback_varyings(program, varyings, glow::INTERLEAVED_ATTRIBS);
        }
        gl.link_program(program);

        if !gl.get_program_link_status(program) {
            let log = gl.get_program_info_log(program);
            gl.delete_program(program);
            gl.delete_shader(vert);
            gl.delete_shader(frag);
            return Err(format!("program link error: {log}"));
        }

        // Shaders can be detached after linking
        gl.detach_shader(program, vert);
        gl.detach_shader(program, frag);
        gl.delete_shader(vert);
        gl.delete_shader(frag);

        Ok(program)
    }
}

fn link_cached_program(
    gl: &glow::Context,
    vertex: glow::Shader,
    fragment: Option<glow::Shader>,
    transform_feedback_varyings: &[Arc<str>],
) -> Result<glow::Program, String> {
    unsafe {
        let program = gl.create_program().map_err(|error| format!("create_program: {error}"))?;
        gl.attach_shader(program, vertex);
        if let Some(fragment) = fragment {
            gl.attach_shader(program, fragment);
        }
        if !transform_feedback_varyings.is_empty() {
            let varyings = transform_feedback_varyings
                .iter()
                .map(AsRef::as_ref)
                .collect::<Vec<&str>>();
            gl.transform_feedback_varyings(program, &varyings, glow::INTERLEAVED_ATTRIBS);
        }
        gl.link_program(program);
        if !gl.get_program_link_status(program) {
            let log = gl.get_program_info_log(program);
            gl.delete_program(program);
            return Err(format!("program link error: {log}"));
        }
        gl.detach_shader(program, vertex);
        if let Some(fragment) = fragment {
            gl.detach_shader(program, fragment);
        }
        Ok(program)
    }
}

fn initialize_cached_program(
    gl: &glow::Context,
    program: glow::Program,
    bindings: &WebGlBindingPlan,
) -> Result<(), String> {
    unsafe {
        let planned_blocks = bindings
            .uniform_blocks()
            .iter()
            .map(|binding| (binding.name.as_ref(), binding.binding_point))
            .collect::<HashMap<_, _>>();
        let num_blocks = gl.get_program_parameter_i32(program, glow::ACTIVE_UNIFORM_BLOCKS);
        let mut active_blocks = BTreeSet::new();
        for index in 0..u32::try_from(num_blocks).unwrap_or(0) {
            let name = gl.get_active_uniform_block_name(program, index);
            let Some(&binding_point) = planned_blocks.get(name.as_str()) else {
                gl.delete_program(program);
                return Err(format!("active WebGL uniform block '{name}' has no binding plan"));
            };
            gl.uniform_block_binding(program, index, binding_point);
            active_blocks.insert(name);
        }
        if num_blocks > 2 {
            log_info(&format!(
                "All {num_blocks} UBO blocks: {:?}",
                active_blocks
            ));
        }

        gl.use_program(Some(program));
        for binding in bindings.samplers() {
            if let Some(location) = gl.get_uniform_location(program, binding.name.as_ref()) {
                let texture_unit = i32::try_from(binding.texture_unit).map_err(|_| {
                    gl.use_program(None);
                    gl.delete_program(program);
                    format!(
                        "texture unit {} for '{}' exceeds WebGL's signed uniform range",
                        binding.texture_unit, binding.name
                    )
                })?;
                gl.uniform_1_i32(Some(&location), texture_unit);
            }
        }
        gl.use_program(None);
        log_info(&format!(
            "UBO blocks bound for program: {:?}",
            active_blocks
        ));
    }
    Ok(())
}

impl WebGlProgramMemo {
    pub fn get_or_create(
        &mut self,
        gl: &glow::Context,
        key: WebGlProgramKey,
    ) -> Result<glow::Program, String> {
        let shaders = &mut self.shaders;
        let programs = &mut self.programs;
        programs
            .get_or_try_insert_with(key, |key| {
                let vertex = *shaders.get_or_try_insert_with(
                    key.program().vertex().clone(),
                    |descriptor| {
                        let source = compile_module_glsl(descriptor)?;
                        compile_gl_shader(gl, glow::VERTEX_SHADER, &source)
                    },
                )?;
                let fragment = key
                    .program()
                    .fragment()
                    .map(|descriptor| {
                        shaders
                            .get_or_try_insert_with(descriptor.clone(), |descriptor| {
                                let source = compile_module_glsl(descriptor)?;
                                compile_gl_shader(gl, glow::FRAGMENT_SHADER, &source)
                            })
                            .copied()
                    })
                    .transpose()?;
                let program = link_cached_program(
                    gl,
                    vertex,
                    fragment,
                    key.program().transform_feedback_varyings(),
                )?;
                initialize_cached_program(gl, program, key.bindings())?;
                Ok(program)
            })
            .copied()
    }
}

/// Create a GL program from GLSL vertex + fragment source.
pub fn create_program(
    gl: &glow::Context,
    vertex_src: &str,
    fragment_src: &str,
) -> Result<glow::Program, String> {
    let vert = compile_gl_shader(gl, glow::VERTEX_SHADER, vertex_src)?;
    let frag = compile_gl_shader(gl, glow::FRAGMENT_SHADER, fragment_src)
        .map_err(|e| {
            unsafe { gl.delete_shader(vert); }
            e
        })?;
    link_owned_program(gl, vert, frag, None)
}

/// Create a transform-feedback GL program from GLSL source. Varyings are
/// interleaved in the supplied order, which therefore defines the destination
/// buffer's record layout.
pub fn create_transform_feedback_program(
    gl: &glow::Context,
    vertex_src: &str,
    fragment_src: &str,
    varyings: &[&str],
) -> Result<glow::Program, String> {
    let vert = compile_gl_shader(gl, glow::VERTEX_SHADER, vertex_src)?;
    let frag = compile_gl_shader(gl, glow::FRAGMENT_SHADER, fragment_src)
        .map_err(|e| {
            unsafe { gl.delete_shader(vert); }
            e
        })?;
    link_owned_program(gl, vert, frag, Some(varyings))
}

/// Texture units matching the prototype's GL texture binding.
pub const SKINNING_TEX_UNIT: u32 = 15;
pub const MORPH_TEX_UNIT: u32 = 14;
pub const FACE_DATA_TEX_UNIT: u32 = 13;
pub const SUPPRESSED_FACE_TEX_UNIT: u32 = 12;

/// Compile all WGSL shaders to GLSL, create GL programs, and bind uniform blocks.
pub fn compile_programs(
    gl: &glow::Context,
    memo: &mut WebGlProgramMemo,
) -> Result<Programs, String> {
    let descriptors = primary_program_descriptors()?;
    let mut matcap = None;
    let mut wire = None;
    let mut normals = None;
    let mut pbr = None;
    let mut stretch = None;
    let mut pick = None;

    for (name, descriptor) in descriptors {
        log_info(&format!("Resolving primary program '{name}'"));
        let program = memo
            .get_or_create(gl, descriptor)
            .map_err(|error| format!("program '{name}': {error}"))?;
        match name {
            "matcap" => matcap = Some(program),
            "wire" => wire = Some(program),
            "normals" => normals = Some(program),
            "pbr" => pbr = Some(program),
            "stretch" => stretch = Some(program),
            "pick" => pick = Some(program),
            _ => {}
        }
    }
    log_info(&format!(
        "All {} primary shader programs resolved",
        FRAGMENT_MODES.len()
    ));

    Ok(Programs {
        matcap: matcap.ok_or("matcap program not compiled")?,
        wire: wire.ok_or("wire program not compiled")?,
        normals: normals.ok_or("normals program not compiled")?,
        pbr: pbr.ok_or("pbr program not compiled")?,
        stretch: stretch.ok_or("stretch program not compiled")?,
        pick: pick.ok_or("pick program not compiled")?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn primary_descriptors_share_one_exact_vertex_module() {
        let descriptors = primary_program_descriptors().unwrap();
        assert_eq!(
            descriptors
                .iter()
                .map(|(name, _)| *name)
                .collect::<Vec<_>>(),
            FRAGMENT_MODES
        );
        let vertex = descriptors[0].1.program().vertex();
        assert_eq!(vertex.source(), quilting_shaders::sources::VERTEX_MAIN);
        assert_eq!(vertex.entry_point(), "vs_main");
        assert_eq!(
            vertex.target(),
            ShaderTarget::GlslEs300 {
                adjust_coordinate_space: false
            }
        );
        assert!(descriptors
            .iter()
            .all(|(_, key)| key.program().vertex() == vertex));
        let shader_modules = descriptors
            .iter()
            .flat_map(|(_, key)| {
                [
                    key.program().vertex(),
                    key.program()
                        .fragment()
                        .expect("primary programs have fragment shaders"),
                ]
            })
            .cloned()
            .collect::<HashSet<_>>();
        assert_eq!(shader_modules.len(), 7);
        assert_eq!(
            descriptors.iter().map(|(_, key)| key.clone()).collect::<HashSet<_>>().len(),
            FRAGMENT_MODES.len()
        );
    }

    #[test]
    fn primary_binding_provenance_matches_composed_wgsl_reflection() {
        for (mode, key) in primary_program_descriptors().unwrap() {
            for descriptor in [
                key.program().vertex(),
                key.program()
                    .fragment()
                    .expect("primary programs have fragment shaders"),
            ] {
                assert!(descriptor.definitions().is_empty());
                let stage = match descriptor.stage() {
                    ShaderStage::Vertex => quilting_shaders::EntryPointStage::Vertex,
                    ShaderStage::Fragment => quilting_shaders::EntryPointStage::Fragment,
                    ShaderStage::Compute => unreachable!("graphics program contains compute"),
                };
                let module =
                    quilting_shaders::compile_shader(descriptor.source(), HashMap::new()).unwrap();
                let reflected = quilting_shaders::reflect_graphics_entry_bindings(
                    &module,
                    stage,
                    descriptor.entry_point(),
                )
                .unwrap();
                assert_eq!(
                    key.bindings()
                        .source_bindings_for_stage(descriptor.stage()),
                    reflected,
                    "{mode} {:?} bindings",
                    descriptor.stage()
                );
            }
        }
    }

    #[test]
    fn cold_renderer_descriptor_catalog_predicts_memo_diagnostics() {
        let mut keys = primary_program_descriptors()
            .unwrap()
            .into_iter()
            .map(|(_, key)| key)
            .collect::<Vec<_>>();
        keys.push(crate::prepare::patch_prepare_program_descriptor().unwrap());

        let shader_requests = keys.iter()
            .map(|key| 1 + usize::from(key.program().fragment().is_some()))
            .sum::<usize>();
        let shader_modules = keys.iter()
            .flat_map(|key| {
                std::iter::once(key.program().vertex())
                    .chain(key.program().fragment())
            })
            .cloned()
            .collect::<HashSet<_>>();
        let programs = keys.into_iter().collect::<HashSet<_>>();

        assert_eq!(shader_requests, 14);
        assert_eq!(shader_modules.len(), 9);
        assert_eq!(programs.len(), 7);
        assert_eq!(
            WebGlProgramMemoDiagnostics {
                device_epoch: 0,
                shaders: DeviceMemoDiagnostics {
                    hits: (shader_requests - shader_modules.len()) as u64,
                    misses: shader_modules.len() as u64,
                    failed_creations: 0,
                    invalidations: 0,
                    resident_entries: shader_modules.len(),
                },
                programs: DeviceMemoDiagnostics {
                    hits: 0,
                    misses: programs.len() as u64,
                    failed_creations: 0,
                    invalidations: 0,
                    resident_entries: programs.len(),
                },
            },
            WebGlProgramMemoDiagnostics {
                device_epoch: 0,
                shaders: DeviceMemoDiagnostics {
                    hits: 5,
                    misses: 9,
                    failed_creations: 0,
                    invalidations: 0,
                    resident_entries: 9,
                },
                programs: DeviceMemoDiagnostics {
                    hits: 0,
                    misses: 7,
                    failed_creations: 0,
                    invalidations: 0,
                    resident_entries: 7,
                },
            }
        );
    }

    #[test]
    fn emitted_primary_interfaces_are_covered_by_the_exact_plans() {
        for (mode, key) in primary_program_descriptors().unwrap() {
            let planned_blocks = key
                .bindings()
                .uniform_blocks()
                .iter()
                .map(|binding| binding.name.as_ref())
                .collect::<HashSet<_>>();
            let planned_samplers = key
                .bindings()
                .samplers()
                .iter()
                .map(|binding| binding.name.as_ref())
                .collect::<HashSet<_>>();
            for descriptor in [
                key.program().vertex(),
                key.program()
                    .fragment()
                    .expect("primary programs have fragment shaders"),
            ] {
                let source = compile_module_glsl(descriptor).unwrap();
                for line in source.lines() {
                    if let Some(after_uniform) = line.split("uniform ").nth(1) {
                        if let Some(block) = after_uniform
                            .split_whitespace()
                            .next()
                            .filter(|name| name.contains("_block_"))
                        {
                            assert!(
                                planned_blocks.contains(block),
                                "{mode} has unplanned uniform block {block}"
                            );
                        }
                    }
                    if line.contains("sampler") {
                        for token in line.split_whitespace() {
                            let sampler = token.trim_end_matches(';');
                            if sampler.starts_with("_group_") {
                                assert!(
                                    planned_samplers.contains(sampler),
                                    "{mode} has unplanned sampler {sampler}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn binding_plans_are_exact_canonical_program_identity() {
        let descriptors = primary_program_descriptors().unwrap();
        let pbr = descriptors
            .iter()
            .find(|(name, _)| *name == "pbr")
            .unwrap()
            .1
            .clone();
        let block_bindings = pbr
            .bindings()
            .uniform_blocks()
            .iter()
            .map(|binding| (binding.name.as_ref(), binding.binding_point))
            .collect::<Vec<_>>();
        assert_eq!(
            block_bindings,
            vec![
                ("JointMatrices_block_1Vertex", JOINT_MATRICES_BINDING),
                ("PbrUniforms_block_0Fragment", PBR_UNIFORMS_BINDING),
                ("Uniforms_block_0Vertex", VERTEX_UNIFORMS_BINDING),
            ]
        );
        assert_eq!(
            pbr.bindings()
                .samplers()
                .iter()
                .find(|binding| binding.name.as_ref() == "_group_3_binding_5_fs")
                .map(|binding| binding.texture_unit),
            Some(10)
        );

        let altered = WebGlBindingPlan::new(
            vec![WebGlUniformBlockBinding {
                name: "Uniforms_block_0Vertex".into(),
                binding_point: 99,
                source_name: "u".into(),
                source: WebGlBindingSite::new(0, 0, ShaderStage::Vertex),
                minimum_size: 352,
            }],
            Vec::new(),
        )
        .unwrap();
        assert_ne!(pbr, WebGlProgramKey::new(pbr.program().clone(), altered));
    }

    #[test]
    fn primary_plans_have_complete_non_conflicting_portable_layouts() {
        let descriptors = primary_program_descriptors().unwrap();
        for (mode, key) in &descriptors {
            assert!(
                key.bindings().cross_stage_slot_conflicts().is_empty(),
                "unexpected {mode} binding conflict"
            );
            let layout = key.bindings().portable_layout().unwrap();
            for (expected_group, group) in layout.groups().iter().enumerate() {
                assert_eq!(group.group(), expected_group as u32);
            }
        }

        let pbr = descriptors
            .iter()
            .find(|(mode, _)| *mode == "pbr")
            .unwrap()
            .1
            .bindings()
            .portable_layout()
            .unwrap();
        assert_eq!(pbr.groups().len(), 4);
        assert_eq!(pbr.groups()[FRAME_POSE_BIND_GROUP as usize].entries().len(), 3);
        assert_eq!(
            pbr.groups()[ENTITY_BATCH_BIND_GROUP as usize]
                .entries()
                .len(),
            1
        );
        assert_eq!(
            pbr.groups()[MATERIAL_STYLE_BIND_GROUP as usize]
                .entries()
                .len(),
            15
        );
        assert_eq!(
            pbr.groups()[PASS_RESOURCE_BIND_GROUP as usize]
                .entries()
                .len(),
            4
        );
        assert_eq!(
            pbr.groups()[MATERIAL_STYLE_BIND_GROUP as usize]
                .entries()
                .iter()
                .find(|entry| entry.binding == 13)
                .map(|entry| entry.kind),
            Some(FILTERABLE_TEXTURE_CUBE)
        );
        assert_eq!(
            pbr.groups()[PASS_RESOURCE_BIND_GROUP as usize]
                .entries()
                .iter()
                .find(|entry| entry.binding == 4)
                .map(|entry| entry.kind),
            Some(FILTERABLE_TEXTURE_2D)
        );
    }

    #[test]
    fn stage_specific_binding_provenance_is_program_identity() {
        let program = primary_program_descriptors().unwrap()[0]
            .1
            .program()
            .clone();
        let vertex_plan = WebGlBindingPlan::new(
            vec![WebGlUniformBlockBinding {
                name: "SameEmittedBlock".into(),
                binding_point: 7,
                source_name: "same_source".into(),
                source: WebGlBindingSite::new(0, 1, ShaderStage::Vertex),
                minimum_size: 16,
            }],
            Vec::new(),
        )
        .unwrap();
        let fragment_plan = WebGlBindingPlan::new(
            vec![WebGlUniformBlockBinding {
                name: "SameEmittedBlock".into(),
                binding_point: 7,
                source_name: "same_source".into(),
                source: WebGlBindingSite::new(0, 1, ShaderStage::Fragment),
                minimum_size: 16,
            }],
            Vec::new(),
        )
        .unwrap();

        let mut fragment_conflict = fragment_plan.uniform_blocks()[0].clone();
        fragment_conflict.name = "DistinctFragmentBlock".into();
        let conflicting = WebGlBindingPlan::new(
            vec![
                vertex_plan.uniform_blocks()[0].clone(),
                fragment_conflict,
            ],
            Vec::new(),
        )
        .unwrap();
        assert_eq!(
            conflicting.cross_stage_slot_conflicts(),
            vec![WebGlStageBindingConflict {
                group: 0,
                binding: 1,
                stages: vec![ShaderStage::Vertex, ShaderStage::Fragment],
            }]
        );
        assert!(conflicting.portable_layout().is_err());

        assert_ne!(
            WebGlProgramKey::new(program.clone(), vertex_plan),
            WebGlProgramKey::new(program, fragment_plan)
        );
    }

    #[test]
    fn binding_plan_order_is_canonical_and_duplicate_names_fail() {
        let left = WebGlBindingPlan::new(
            vec![
                WebGlUniformBlockBinding {
                    name: "z".into(),
                    binding_point: 2,
                    source_name: "z_source".into(),
                    source: WebGlBindingSite::new(0, 2, ShaderStage::Vertex),
                    minimum_size: 16,
                },
                WebGlUniformBlockBinding {
                    name: "a".into(),
                    binding_point: 1,
                    source_name: "a_source".into(),
                    source: WebGlBindingSite::new(0, 1, ShaderStage::Vertex),
                    minimum_size: 16,
                },
            ],
            Vec::new(),
        )
        .unwrap();
        let right = WebGlBindingPlan::new(
            vec![
                WebGlUniformBlockBinding {
                    name: "a".into(),
                    binding_point: 1,
                    source_name: "a_source".into(),
                    source: WebGlBindingSite::new(0, 1, ShaderStage::Vertex),
                    minimum_size: 16,
                },
                WebGlUniformBlockBinding {
                    name: "z".into(),
                    binding_point: 2,
                    source_name: "z_source".into(),
                    source: WebGlBindingSite::new(0, 2, ShaderStage::Vertex),
                    minimum_size: 16,
                },
            ],
            Vec::new(),
        )
        .unwrap();
        assert_eq!(left, right);
        let mut zero_sized = left.uniform_blocks()[0].clone();
        zero_sized.minimum_size = 0;
        assert!(WebGlBindingPlan::new(vec![zero_sized], Vec::new())
            .unwrap_err()
            .contains("minimum sizes"));
        assert!(WebGlBindingPlan::new(
            Vec::new(),
            vec![WebGlSamplerBinding {
                name: "not_opaque".into(),
                texture_unit: 0,
                source_name: "not_opaque".into(),
                source: WebGlBindingSite::new(0, 0, ShaderStage::Fragment),
                source_kind: BindingKind::StorageBuffer {
                    read_only: true,
                    dynamic_offset: false,
                    minimum_size: 16,
                },
            }],
        )
        .unwrap_err()
        .contains("sampled textures or samplers"));
        assert!(WebGlBindingPlan::new(
            vec![
                WebGlUniformBlockBinding {
                    name: "same".into(),
                    binding_point: 1,
                    source_name: "first_source".into(),
                    source: WebGlBindingSite::new(0, 1, ShaderStage::Vertex),
                    minimum_size: 16,
                },
                WebGlUniformBlockBinding {
                    name: "same".into(),
                    binding_point: 2,
                    source_name: "second_source".into(),
                    source: WebGlBindingSite::new(0, 2, ShaderStage::Vertex),
                    minimum_size: 16,
                },
            ],
            Vec::new(),
        )
        .is_err());

        assert!(WebGlBindingPlan::new(
            vec![
                WebGlUniformBlockBinding {
                    name: "first".into(),
                    binding_point: 1,
                    source_name: "first_source".into(),
                    source: WebGlBindingSite::new(0, 1, ShaderStage::Vertex),
                    minimum_size: 16,
                },
                WebGlUniformBlockBinding {
                    name: "second".into(),
                    binding_point: 2,
                    source_name: "second_source".into(),
                    source: WebGlBindingSite::new(0, 1, ShaderStage::Vertex),
                    minimum_size: 16,
                },
            ],
            Vec::new(),
        )
        .unwrap_err()
        .contains("duplicate stage-specific"));

        assert!(WebGlBindingPlan::new(
            vec![WebGlUniformBlockBinding {
                name: "compute".into(),
                binding_point: 0,
                source_name: "compute_source".into(),
                source: WebGlBindingSite::new(0, 0, ShaderStage::Compute),
                minimum_size: 16,
            }],
            Vec::new(),
        )
        .unwrap_err()
        .contains("compute stage"));
    }

    #[test]
    fn descriptor_lowering_rejects_non_webgl_and_compute_targets() {
        let catalog = quilting_shaders::compiler_catalog_revision();
        let wgsl = ShaderModuleDescriptor::new(
            "wgsl-only",
            quilting_shaders::sources::VERTEX_MAIN,
            Arc::clone(&catalog),
            ShaderStage::Vertex,
            "vs_main",
            ShaderTarget::Wgsl,
            Vec::new(),
        )
        .unwrap();
        assert!(compile_module_glsl(&wgsl)
            .unwrap_err()
            .contains("WebGL cannot lower"));

        let compute = ShaderModuleDescriptor::new(
            "compute",
            "@compute @workgroup_size(1) fn main() {}",
            catalog,
            ShaderStage::Compute,
            "main",
            ShaderTarget::GlslEs300 {
                adjust_coordinate_space: false,
            },
            Vec::new(),
        )
        .unwrap();
        assert_eq!(
            compile_module_glsl(&compute).unwrap_err(),
            "WebGL2 does not support compute shader modules"
        );
    }
}
