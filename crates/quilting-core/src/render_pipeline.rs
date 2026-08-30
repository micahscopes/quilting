//! Immutable shader and pipeline descriptions shared by rendering backends.
//!
//! These values are the pure side of pipeline creation: application/FRP code
//! may compare, retain, and memoize them without owning a GL/WebGPU handle.
//! Concrete backends lower a descriptor into device resources and keep those
//! resources in an epoch-scoped cache. Frame-varying uniforms, resource
//! handles, command encoders, and submission state never belong in these keys.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ShaderStage {
    Vertex,
    Fragment,
    Compute,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ShaderTarget {
    Wgsl,
    GlslEs300 { adjust_coordinate_space: bool },
}

/// Canonical finite `f32` for the few floating-point values that are part of
/// immutable pipeline state. Negative zero is normalized and NaN/infinity are
/// rejected before a value can become a memo key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FiniteF32(u32);

impl FiniteF32 {
    pub fn new(value: f32) -> Result<Self, RenderPipelineDescriptorError> {
        if !value.is_finite() {
            return Err(RenderPipelineDescriptorError::NonFinitePipelineFloat);
        }
        let canonical = if value == 0.0 { 0.0 } else { value };
        Ok(Self(canonical.to_bits()))
    }

    pub fn get(self) -> f32 {
        f32::from_bits(self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ShaderDefinitionValue {
    Bool(bool),
    I32(i32),
    U32(u32),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShaderDefinition {
    pub name: Arc<str>,
    pub value: ShaderDefinitionValue,
}

/// Complete, immutable input to one shader-module compilation.
///
/// The WGSL source itself is retained, rather than using a digest as the sole
/// cache identity. `Arc<str>` keeps descriptor cloning cheap while ordinary
/// equality still protects the cache from a digest collision or stale label.
#[derive(Debug, Clone)]
pub struct ShaderModuleDescriptor {
    label: Arc<str>,
    source: Arc<str>,
    source_fingerprint: u64,
    compiler_catalog_revision: Arc<str>,
    stage: ShaderStage,
    entry_point: Arc<str>,
    target: ShaderTarget,
    definitions: Vec<ShaderDefinition>,
}

impl ShaderModuleDescriptor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        label: impl Into<Arc<str>>,
        source: impl Into<Arc<str>>,
        compiler_catalog_revision: impl Into<Arc<str>>,
        stage: ShaderStage,
        entry_point: impl Into<Arc<str>>,
        target: ShaderTarget,
        mut definitions: Vec<ShaderDefinition>,
    ) -> Result<Self, RenderPipelineDescriptorError> {
        let label = label.into();
        let source = source.into();
        let compiler_catalog_revision = compiler_catalog_revision.into();
        let entry_point = entry_point.into();
        if label.is_empty() {
            return Err(RenderPipelineDescriptorError::EmptyShaderLabel);
        }
        if source.is_empty() {
            return Err(RenderPipelineDescriptorError::EmptyShaderSource);
        }
        if compiler_catalog_revision.is_empty() {
            return Err(RenderPipelineDescriptorError::EmptyCompilerCatalogRevision);
        }
        if entry_point.is_empty() {
            return Err(RenderPipelineDescriptorError::EmptyEntryPoint);
        }
        if definitions
            .iter()
            .any(|definition| definition.name.is_empty())
        {
            return Err(RenderPipelineDescriptorError::EmptyShaderDefinitionName);
        }
        definitions.sort_by(|left, right| left.name.cmp(&right.name));
        if definitions
            .windows(2)
            .any(|pair| pair[0].name == pair[1].name)
        {
            return Err(RenderPipelineDescriptorError::DuplicateShaderDefinition);
        }
        let source_fingerprint = source_fingerprint(source.as_bytes());
        Ok(Self {
            label,
            source,
            source_fingerprint,
            compiler_catalog_revision,
            stage,
            entry_point,
            target,
            definitions,
        })
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    /// Identifies both the shader compiler versions and the exact composable
    /// module catalog/configuration loaded into that compiler.
    pub fn compiler_catalog_revision(&self) -> &str {
        &self.compiler_catalog_revision
    }

    pub fn stage(&self) -> ShaderStage {
        self.stage
    }

    pub fn entry_point(&self) -> &str {
        &self.entry_point
    }

    pub fn target(&self) -> ShaderTarget {
        self.target
    }

    pub fn definitions(&self) -> &[ShaderDefinition] {
        &self.definitions
    }
}

impl PartialEq for ShaderModuleDescriptor {
    fn eq(&self, other: &Self) -> bool {
        self.source == other.source
            && self.compiler_catalog_revision == other.compiler_catalog_revision
            && self.stage == other.stage
            && self.entry_point == other.entry_point
            && self.target == other.target
            && self.definitions == other.definitions
    }
}

impl Eq for ShaderModuleDescriptor {}

impl Hash for ShaderModuleDescriptor {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash the cached source fingerprint, then rely on exact source
        // equality above to make collisions harmless. The diagnostic label is
        // deliberately not part of compilation identity.
        self.source_fingerprint.hash(state);
        self.compiler_catalog_revision.hash(state);
        self.stage.hash(state);
        self.entry_point.hash(state);
        self.target.hash(state);
        self.definitions.hash(state);
    }
}

/// Stable FNV-1a fingerprint used only to accelerate the in-memory hash key.
/// It is neither a durable identifier nor a security digest; exact source
/// equality remains authoritative for collision handling.
fn source_fingerprint(source: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in source {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShaderVisibility(u8);

impl ShaderVisibility {
    const VERTEX_BIT: u8 = 1 << 0;
    const FRAGMENT_BIT: u8 = 1 << 1;
    const COMPUTE_BIT: u8 = 1 << 2;

    pub const VERTEX: Self = Self(Self::VERTEX_BIT);
    pub const FRAGMENT: Self = Self(Self::FRAGMENT_BIT);
    pub const COMPUTE: Self = Self(Self::COMPUTE_BIT);

    pub const fn vertex_fragment() -> Self {
        Self(Self::VERTEX_BIT | Self::FRAGMENT_BIT)
    }

    pub const fn contains(self, stage: ShaderStage) -> bool {
        let bit = match stage {
            ShaderStage::Vertex => Self::VERTEX_BIT,
            ShaderStage::Fragment => Self::FRAGMENT_BIT,
            ShaderStage::Compute => Self::COMPUTE_BIT,
        };
        self.0 & bit != 0
    }

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TextureSampleKind {
    FloatFilterable,
    FloatUnfilterable,
    Sint,
    Uint,
    Depth,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TextureViewDimension {
    D1,
    D2,
    D2Array,
    Cube,
    CubeArray,
    D3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SamplerBindingKind {
    Filtering,
    NonFiltering,
    Comparison,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StorageTextureAccess {
    ReadOnly,
    WriteOnly,
    ReadWrite,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TextureFormat {
    R8Unorm,
    Rg8Unorm,
    Rgba8Unorm,
    Rgba8UnormSrgb,
    Bgra8Unorm,
    Bgra8UnormSrgb,
    Rgb10a2Unorm,
    R16Float,
    Rg16Float,
    Rgba16Float,
    R32Float,
    Rg32Float,
    Rgba32Float,
    R32Uint,
    Rgba32Uint,
    Depth24Plus,
    Depth24PlusStencil8,
    Depth32Float,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BindingKind {
    UniformBuffer {
        dynamic_offset: bool,
        minimum_size: u64,
    },
    StorageBuffer {
        read_only: bool,
        dynamic_offset: bool,
        minimum_size: u64,
    },
    Texture {
        sample_kind: TextureSampleKind,
        view_dimension: TextureViewDimension,
        multisampled: bool,
    },
    StorageTexture {
        access: StorageTextureAccess,
        format: TextureFormat,
        view_dimension: TextureViewDimension,
    },
    Sampler(SamplerBindingKind),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BindGroupLayoutEntry {
    pub binding: u32,
    pub visibility: ShaderVisibility,
    pub kind: BindingKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BindGroupLayoutDescriptor {
    group: u32,
    entries: Vec<BindGroupLayoutEntry>,
}

impl BindGroupLayoutDescriptor {
    pub fn new(
        group: u32,
        mut entries: Vec<BindGroupLayoutEntry>,
    ) -> Result<Self, RenderPipelineDescriptorError> {
        if entries.iter().any(|entry| entry.visibility.is_empty()) {
            return Err(RenderPipelineDescriptorError::EmptyShaderVisibility);
        }
        entries.sort_by_key(|entry| entry.binding);
        if entries
            .windows(2)
            .any(|pair| pair[0].binding == pair[1].binding)
        {
            return Err(RenderPipelineDescriptorError::DuplicateBinding);
        }
        Ok(Self { group, entries })
    }

    pub fn group(&self) -> u32 {
        self.group
    }

    pub fn entries(&self) -> &[BindGroupLayoutEntry] {
        &self.entries
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PipelineLayoutDescriptor {
    groups: Vec<BindGroupLayoutDescriptor>,
}

impl PipelineLayoutDescriptor {
    pub fn new(
        mut groups: Vec<BindGroupLayoutDescriptor>,
    ) -> Result<Self, RenderPipelineDescriptorError> {
        groups.sort_by_key(BindGroupLayoutDescriptor::group);
        if groups.windows(2).any(|pair| pair[0].group == pair[1].group) {
            return Err(RenderPipelineDescriptorError::DuplicateBindGroup);
        }
        Ok(Self { groups })
    }

    pub fn groups(&self) -> &[BindGroupLayoutDescriptor] {
        &self.groups
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum VertexStepMode {
    Vertex,
    Instance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum VertexFormat {
    Uint8x2,
    Uint8x4,
    Sint8x2,
    Sint8x4,
    Unorm8x2,
    Unorm8x4,
    Snorm8x2,
    Snorm8x4,
    Uint16x2,
    Uint16x4,
    Sint16x2,
    Sint16x4,
    Unorm16x2,
    Unorm16x4,
    Snorm16x2,
    Snorm16x4,
    Float16x2,
    Float16x4,
    Float32,
    Float32x2,
    Float32x3,
    Float32x4,
    Uint32,
    Uint32x2,
    Uint32x3,
    Uint32x4,
    Sint32,
    Sint32x2,
    Sint32x3,
    Sint32x4,
}

impl VertexFormat {
    const fn byte_size(self) -> u64 {
        match self {
            Self::Uint8x2 | Self::Sint8x2 | Self::Unorm8x2 | Self::Snorm8x2 => 2,
            Self::Uint8x4
            | Self::Sint8x4
            | Self::Unorm8x4
            | Self::Snorm8x4
            | Self::Uint16x2
            | Self::Sint16x2
            | Self::Unorm16x2
            | Self::Snorm16x2
            | Self::Float16x2
            | Self::Float32
            | Self::Uint32
            | Self::Sint32 => 4,
            Self::Uint16x4
            | Self::Sint16x4
            | Self::Unorm16x4
            | Self::Snorm16x4
            | Self::Float16x4
            | Self::Float32x2
            | Self::Uint32x2
            | Self::Sint32x2 => 8,
            Self::Float32x3 | Self::Uint32x3 | Self::Sint32x3 => 12,
            Self::Float32x4 | Self::Uint32x4 | Self::Sint32x4 => 16,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VertexAttributeDescriptor {
    pub location: u32,
    pub offset: u64,
    pub format: VertexFormat,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VertexBufferLayoutDescriptor {
    slot: u32,
    stride: u64,
    step_mode: VertexStepMode,
    attributes: Vec<VertexAttributeDescriptor>,
}

impl VertexBufferLayoutDescriptor {
    pub fn new(
        slot: u32,
        stride: u64,
        step_mode: VertexStepMode,
        mut attributes: Vec<VertexAttributeDescriptor>,
    ) -> Result<Self, RenderPipelineDescriptorError> {
        if stride == 0
            || attributes.iter().any(|attribute| {
                attribute
                    .offset
                    .checked_add(attribute.format.byte_size())
                    .is_none_or(|end| end > stride)
            })
        {
            return Err(RenderPipelineDescriptorError::InvalidVertexBufferLayout);
        }
        attributes.sort_by_key(|attribute| attribute.location);
        if attributes
            .windows(2)
            .any(|pair| pair[0].location == pair[1].location)
        {
            return Err(RenderPipelineDescriptorError::DuplicateVertexLocation);
        }
        Ok(Self {
            slot,
            stride,
            step_mode,
            attributes,
        })
    }

    pub fn slot(&self) -> u32 {
        self.slot
    }

    pub fn stride(&self) -> u64 {
        self.stride
    }

    pub fn step_mode(&self) -> VertexStepMode {
        self.step_mode
    }

    pub fn attributes(&self) -> &[VertexAttributeDescriptor] {
        &self.attributes
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PrimitiveTopology {
    PointList,
    LineList,
    LineStrip,
    TriangleList,
    TriangleStrip,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum IndexFormat {
    Uint16,
    Uint32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FrontFace {
    CounterClockwise,
    Clockwise,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CullMode {
    None,
    Front,
    Back,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PrimitiveStateDescriptor {
    pub topology: PrimitiveTopology,
    pub strip_index_format: Option<IndexFormat>,
    pub front_face: FrontFace,
    pub cull_mode: CullMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CompareFunction {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StencilOperation {
    Keep,
    Zero,
    Replace,
    Invert,
    IncrementClamp,
    DecrementClamp,
    IncrementWrap,
    DecrementWrap,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StencilFaceStateDescriptor {
    pub compare: CompareFunction,
    pub fail_op: StencilOperation,
    pub depth_fail_op: StencilOperation,
    pub pass_op: StencilOperation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DepthStencilStateDescriptor {
    pub format: TextureFormat,
    pub depth_write_enabled: bool,
    pub depth_compare: CompareFunction,
    pub stencil_front: StencilFaceStateDescriptor,
    pub stencil_back: StencilFaceStateDescriptor,
    pub stencil_read_mask: u32,
    pub stencil_write_mask: u32,
    pub depth_bias_constant: i32,
    pub depth_bias_slope_scale: FiniteF32,
    pub depth_bias_clamp: FiniteF32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BlendFactor {
    Zero,
    One,
    Source,
    OneMinusSource,
    SourceAlpha,
    OneMinusSourceAlpha,
    Destination,
    OneMinusDestination,
    DestinationAlpha,
    OneMinusDestinationAlpha,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BlendOperation {
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlendComponentDescriptor {
    pub source_factor: BlendFactor,
    pub destination_factor: BlendFactor,
    pub operation: BlendOperation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlendStateDescriptor {
    pub color: BlendComponentDescriptor,
    pub alpha: BlendComponentDescriptor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ColorWriteMask(u8);

impl ColorWriteMask {
    pub const RED: Self = Self(1 << 0);
    pub const GREEN: Self = Self(1 << 1);
    pub const BLUE: Self = Self(1 << 2);
    pub const ALPHA: Self = Self(1 << 3);
    pub const ALL: Self = Self(0b1111);

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub const fn bits(self) -> u8 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ColorTargetStateDescriptor {
    pub format: TextureFormat,
    pub blend: Option<BlendStateDescriptor>,
    pub write_mask: ColorWriteMask,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MultisampleStateDescriptor {
    pub count: u32,
    pub mask: u64,
    pub alpha_to_coverage_enabled: bool,
}

impl Default for MultisampleStateDescriptor {
    fn default() -> Self {
        Self {
            count: 1,
            mask: u64::MAX,
            alpha_to_coverage_enabled: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GraphicsProgramDescriptor {
    vertex: ShaderModuleDescriptor,
    fragment: Option<ShaderModuleDescriptor>,
    transform_feedback_varyings: Vec<Arc<str>>,
}

impl GraphicsProgramDescriptor {
    pub fn new(
        vertex: ShaderModuleDescriptor,
        fragment: Option<ShaderModuleDescriptor>,
        transform_feedback_varyings: Vec<Arc<str>>,
    ) -> Result<Self, RenderPipelineDescriptorError> {
        if vertex.stage != ShaderStage::Vertex
            || fragment
                .as_ref()
                .is_some_and(|module| module.stage != ShaderStage::Fragment)
        {
            return Err(RenderPipelineDescriptorError::InvalidGraphicsStages);
        }
        if fragment
            .as_ref()
            .is_some_and(|module| module.target != vertex.target)
        {
            return Err(RenderPipelineDescriptorError::ShaderTargetMismatch);
        }
        if transform_feedback_varyings
            .iter()
            .any(|varying| varying.is_empty())
        {
            return Err(RenderPipelineDescriptorError::EmptyTransformFeedbackVarying);
        }
        let mut unique_varyings = BTreeSet::new();
        if !transform_feedback_varyings
            .iter()
            .all(|varying| unique_varyings.insert(varying.as_ref()))
        {
            return Err(RenderPipelineDescriptorError::DuplicateTransformFeedbackVarying);
        }
        if !transform_feedback_varyings.is_empty()
            && !matches!(vertex.target, ShaderTarget::GlslEs300 { .. })
        {
            return Err(RenderPipelineDescriptorError::TransformFeedbackTargetMismatch);
        }
        Ok(Self {
            vertex,
            fragment,
            transform_feedback_varyings,
        })
    }

    pub fn vertex(&self) -> &ShaderModuleDescriptor {
        &self.vertex
    }

    pub fn fragment(&self) -> Option<&ShaderModuleDescriptor> {
        self.fragment.as_ref()
    }

    pub fn transform_feedback_varyings(&self) -> &[Arc<str>] {
        &self.transform_feedback_varyings
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RenderPipelineDescriptor {
    program: GraphicsProgramDescriptor,
    layout: PipelineLayoutDescriptor,
    vertex_buffers: Vec<VertexBufferLayoutDescriptor>,
    primitive: PrimitiveStateDescriptor,
    depth_stencil: Option<DepthStencilStateDescriptor>,
    color_targets: Vec<ColorTargetStateDescriptor>,
    multisample: MultisampleStateDescriptor,
}

impl RenderPipelineDescriptor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        program: GraphicsProgramDescriptor,
        layout: PipelineLayoutDescriptor,
        mut vertex_buffers: Vec<VertexBufferLayoutDescriptor>,
        primitive: PrimitiveStateDescriptor,
        depth_stencil: Option<DepthStencilStateDescriptor>,
        color_targets: Vec<ColorTargetStateDescriptor>,
        multisample: MultisampleStateDescriptor,
    ) -> Result<Self, RenderPipelineDescriptorError> {
        if multisample.count == 0 || !multisample.count.is_power_of_two() {
            return Err(RenderPipelineDescriptorError::InvalidMultisampleCount);
        }
        if program.fragment.is_none() && !color_targets.is_empty() {
            return Err(RenderPipelineDescriptorError::FragmentTargetMismatch);
        }
        if primitive.strip_index_format.is_some()
            && !matches!(
                primitive.topology,
                PrimitiveTopology::LineStrip | PrimitiveTopology::TriangleStrip
            )
        {
            return Err(RenderPipelineDescriptorError::StripIndexFormatMismatch);
        }
        vertex_buffers.sort_by_key(VertexBufferLayoutDescriptor::slot);
        if vertex_buffers
            .windows(2)
            .any(|pair| pair[0].slot == pair[1].slot)
        {
            return Err(RenderPipelineDescriptorError::DuplicateVertexBufferSlot);
        }
        let mut locations = vertex_buffers
            .iter()
            .flat_map(|buffer| buffer.attributes.iter().map(|attribute| attribute.location))
            .collect::<Vec<_>>();
        locations.sort_unstable();
        if locations.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(RenderPipelineDescriptorError::DuplicateVertexLocation);
        }
        Ok(Self {
            program,
            layout,
            vertex_buffers,
            primitive,
            depth_stencil,
            color_targets,
            multisample,
        })
    }

    pub fn program(&self) -> &GraphicsProgramDescriptor {
        &self.program
    }

    pub fn layout(&self) -> &PipelineLayoutDescriptor {
        &self.layout
    }

    pub fn vertex_buffers(&self) -> &[VertexBufferLayoutDescriptor] {
        &self.vertex_buffers
    }

    pub fn primitive(&self) -> PrimitiveStateDescriptor {
        self.primitive
    }

    pub fn depth_stencil(&self) -> Option<DepthStencilStateDescriptor> {
        self.depth_stencil
    }

    pub fn color_targets(&self) -> &[ColorTargetStateDescriptor] {
        &self.color_targets
    }

    pub fn multisample(&self) -> MultisampleStateDescriptor {
        self.multisample
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ComputePipelineDescriptor {
    module: ShaderModuleDescriptor,
    layout: PipelineLayoutDescriptor,
}

impl ComputePipelineDescriptor {
    pub fn new(
        module: ShaderModuleDescriptor,
        layout: PipelineLayoutDescriptor,
    ) -> Result<Self, RenderPipelineDescriptorError> {
        if module.stage != ShaderStage::Compute {
            return Err(RenderPipelineDescriptorError::InvalidComputeStage);
        }
        Ok(Self { module, layout })
    }

    pub fn module(&self) -> &ShaderModuleDescriptor {
        &self.module
    }

    pub fn layout(&self) -> &PipelineLayoutDescriptor {
        &self.layout
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderPipelineDescriptorError {
    EmptyShaderLabel,
    EmptyShaderSource,
    EmptyCompilerCatalogRevision,
    EmptyEntryPoint,
    EmptyShaderDefinitionName,
    DuplicateShaderDefinition,
    EmptyShaderVisibility,
    DuplicateBinding,
    DuplicateBindGroup,
    InvalidVertexBufferLayout,
    DuplicateVertexBufferSlot,
    DuplicateVertexLocation,
    InvalidGraphicsStages,
    ShaderTargetMismatch,
    InvalidComputeStage,
    EmptyTransformFeedbackVarying,
    DuplicateTransformFeedbackVarying,
    TransformFeedbackTargetMismatch,
    NonFinitePipelineFloat,
    InvalidMultisampleCount,
    FragmentTargetMismatch,
    StripIndexFormatMismatch,
}

impl fmt::Display for RenderPipelineDescriptorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::EmptyShaderLabel => "shader label is empty",
            Self::EmptyShaderSource => "shader source is empty",
            Self::EmptyCompilerCatalogRevision => {
                "shader compiler and module-catalog revision is empty"
            }
            Self::EmptyEntryPoint => "shader entry point is empty",
            Self::EmptyShaderDefinitionName => "shader definition name is empty",
            Self::DuplicateShaderDefinition => "shader definition names are not unique",
            Self::EmptyShaderVisibility => "binding visibility is empty",
            Self::DuplicateBinding => "bind-group binding is not unique",
            Self::DuplicateBindGroup => "pipeline bind-group index is not unique",
            Self::InvalidVertexBufferLayout => "vertex buffer layout is invalid",
            Self::DuplicateVertexBufferSlot => "vertex buffer slot is not unique",
            Self::DuplicateVertexLocation => "vertex attribute location is not unique",
            Self::InvalidGraphicsStages => "graphics program has invalid shader stages",
            Self::ShaderTargetMismatch => "graphics shader targets do not agree",
            Self::InvalidComputeStage => "compute pipeline does not use a compute shader",
            Self::EmptyTransformFeedbackVarying => "transform-feedback varying is empty",
            Self::DuplicateTransformFeedbackVarying => {
                "transform-feedback varying names are not unique"
            }
            Self::TransformFeedbackTargetMismatch => {
                "transform feedback requires the GLSL ES target"
            }
            Self::NonFinitePipelineFloat => "pipeline floating-point value is not finite",
            Self::InvalidMultisampleCount => "multisample count is not a positive power of two",
            Self::FragmentTargetMismatch => "color targets require a fragment stage",
            Self::StripIndexFormatMismatch => {
                "a strip index format is only valid for strip topology"
            }
        })
    }
}

impl Error for RenderPipelineDescriptorError {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn shader(stage: ShaderStage, definitions: Vec<ShaderDefinition>) -> ShaderModuleDescriptor {
        ShaderModuleDescriptor::new(
            format!("{stage:?}"),
            "@vertex fn main() {}",
            "naga-oil-0.19+naga-0.19",
            stage,
            "main",
            ShaderTarget::Wgsl,
            definitions,
        )
        .unwrap()
    }

    fn pipeline(vertex: ShaderModuleDescriptor) -> RenderPipelineDescriptor {
        let fragment = shader(ShaderStage::Fragment, Vec::new());
        let program = GraphicsProgramDescriptor::new(vertex, Some(fragment), Vec::new()).unwrap();
        RenderPipelineDescriptor::new(
            program,
            PipelineLayoutDescriptor::new(Vec::new()).unwrap(),
            vec![VertexBufferLayoutDescriptor::new(
                0,
                12,
                VertexStepMode::Vertex,
                vec![VertexAttributeDescriptor {
                    location: 0,
                    offset: 0,
                    format: VertexFormat::Float32x3,
                }],
            )
            .unwrap()],
            PrimitiveStateDescriptor {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::CounterClockwise,
                cull_mode: CullMode::Back,
            },
            None,
            vec![ColorTargetStateDescriptor {
                format: TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: ColorWriteMask::ALL,
            }],
            MultisampleStateDescriptor::default(),
        )
        .unwrap()
    }

    #[test]
    fn shader_definitions_are_canonical_and_duplicate_names_fail() {
        let first = ShaderDefinition {
            name: "first".into(),
            value: ShaderDefinitionValue::Bool(true),
        };
        let second = ShaderDefinition {
            name: "second".into(),
            value: ShaderDefinitionValue::U32(2),
        };
        assert_eq!(
            shader(ShaderStage::Vertex, vec![second.clone(), first.clone()]),
            shader(ShaderStage::Vertex, vec![first.clone(), second])
        );
        assert_eq!(
            ShaderModuleDescriptor::new(
                "duplicate",
                "source",
                "compiler",
                ShaderStage::Vertex,
                "main",
                ShaderTarget::Wgsl,
                vec![first.clone(), first],
            ),
            Err(RenderPipelineDescriptorError::DuplicateShaderDefinition)
        );
    }

    #[test]
    fn binding_and_vertex_orders_are_canonical() {
        let uniform = BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderVisibility::VERTEX,
            kind: BindingKind::UniformBuffer {
                dynamic_offset: false,
                minimum_size: 64,
            },
        };
        let texture = BindGroupLayoutEntry {
            binding: 3,
            visibility: ShaderVisibility::FRAGMENT,
            kind: BindingKind::Texture {
                sample_kind: TextureSampleKind::FloatFilterable,
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            },
        };
        let left = BindGroupLayoutDescriptor::new(0, vec![texture, uniform]).unwrap();
        let right = BindGroupLayoutDescriptor::new(0, vec![uniform, texture]).unwrap();
        assert_eq!(left, right);

        let mut altered = pipeline(shader(ShaderStage::Vertex, Vec::new()));
        altered.vertex_buffers[0].attributes[0].location = 1;
        assert_ne!(altered, pipeline(shader(ShaderStage::Vertex, Vec::new())));
    }

    #[test]
    fn labels_reuse_cache_but_source_changes_miss() {
        let original_vertex = shader(ShaderStage::Vertex, Vec::new());
        let relabeled_vertex = ShaderModuleDescriptor::new(
            "diagnostic label changed",
            original_vertex.source(),
            original_vertex.compiler_catalog_revision(),
            original_vertex.stage(),
            original_vertex.entry_point(),
            original_vertex.target(),
            original_vertex.definitions().to_vec(),
        )
        .unwrap();
        assert_eq!(original_vertex, relabeled_vertex);

        let original = pipeline(original_vertex);
        let relabeled = pipeline(relabeled_vertex);
        assert_eq!(original, relabeled);
        let changed_source = ShaderModuleDescriptor::new(
            "Vertex",
            "@vertex fn main() { var changed = 1u; }",
            "naga-oil-0.19+naga-0.19",
            ShaderStage::Vertex,
            "main",
            ShaderTarget::Wgsl,
            Vec::new(),
        )
        .unwrap();
        let changed = pipeline(changed_source);
        assert_ne!(original, changed);

        let mut cache = HashMap::new();
        cache.insert(original, "compiled-program-a");
        assert_eq!(cache.get(&relabeled), Some(&"compiled-program-a"));
        assert_eq!(cache.get(&changed), None);
    }

    #[test]
    fn fragmentless_and_nonindexed_strip_pipelines_are_representable() {
        let vertex = shader(ShaderStage::Vertex, Vec::new());
        let program = GraphicsProgramDescriptor::new(vertex, None, Vec::new()).unwrap();
        let pipeline = RenderPipelineDescriptor::new(
            program.clone(),
            PipelineLayoutDescriptor::new(Vec::new()).unwrap(),
            Vec::new(),
            PrimitiveStateDescriptor {
                topology: PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: FrontFace::CounterClockwise,
                cull_mode: CullMode::None,
            },
            None,
            Vec::new(),
            MultisampleStateDescriptor::default(),
        );
        assert!(pipeline.is_ok());

        let invalid = RenderPipelineDescriptor::new(
            program,
            PipelineLayoutDescriptor::new(Vec::new()).unwrap(),
            Vec::new(),
            PrimitiveStateDescriptor {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: Some(IndexFormat::Uint32),
                front_face: FrontFace::CounterClockwise,
                cull_mode: CullMode::None,
            },
            None,
            Vec::new(),
            MultisampleStateDescriptor::default(),
        );
        assert_eq!(
            invalid,
            Err(RenderPipelineDescriptorError::StripIndexFormatMismatch)
        );
    }

    #[test]
    fn pipeline_float_keys_are_finite_and_canonical() {
        fn assert_hash<T: std::hash::Hash>() {}
        assert_hash::<RenderPipelineDescriptor>();
        assert_hash::<ComputePipelineDescriptor>();

        assert_eq!(FiniteF32::new(-0.0), FiniteF32::new(0.0));
        assert_eq!(
            FiniteF32::new(f32::NAN),
            Err(RenderPipelineDescriptorError::NonFinitePipelineFloat)
        );
        assert_eq!(
            FiniteF32::new(f32::INFINITY),
            Err(RenderPipelineDescriptorError::NonFinitePipelineFloat)
        );

        let shader = shader(ShaderStage::Compute, Vec::new());
        let compute = ComputePipelineDescriptor::new(
            shader,
            PipelineLayoutDescriptor::new(Vec::new()).unwrap(),
        );
        assert!(compute.is_ok());
    }
}
