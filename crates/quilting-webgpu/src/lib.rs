//! Shadow WebGPU execution for Quilting's two-pass LOD classifier.
//!
//! This crate is intentionally outside the current WebGL2 authority path. It
//! consumes the exact backend-neutral payload frozen by `quilting-renderer`,
//! retains device pipelines and model buffers, and returns packed words for
//! conformance comparison. It owns no scene, FRP, or replicated state.

use futures_channel::oneshot;
use quilting_renderer::compute::{
    pack_wgsl_lod_atlas_words, pack_wgsl_lod_dispatch_words, pack_wgsl_lod_model_words,
    pack_wgsl_lod_subject_words, pack_wgsl_source_visibility_words, prepare_lod_atlas_lookup,
    prepare_lod_model, reconcile_and_pack_wgsl_lod_pass2, LodAtlasLookup, LodDispatchState,
    LodModelData, PreparedLodModel, WgslLodDispatchMetrics, WgslVisibilityCompactionSceneWords,
};
use std::borrow::Cow;
use wgpu::util::DeviceExt;

const LOD_WORKGROUP_SIZE: u32 = 64;
const DISPATCH_UNIFORM_BYTES: u64 = 272;
const SUBJECT_RECORD_BYTES: u64 = 160;
const JOINT_MATRIX_BYTES: u64 = 64;
const PASS1_RECORD_BYTES: u64 = 16;
const PACKED_RECORD_BYTES: u64 = 4;
const VISIBILITY_UNIFORM_BYTES: u64 = 16;
const VISIBILITY_BATCH_RECORD_BYTES: u64 = 16;
const VISIBILITY_RANGE_RECORD_BYTES: u64 = 20;
const INDEXED_INDIRECT_RECORD_BYTES: u64 = 20;

fn identity_matrix() -> [f32; 16] {
    [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]
}

fn identity_mobius() -> [f32; 16] {
    [
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
    ]
}

#[derive(Debug)]
pub enum LodWebGpuError {
    Shader(String),
    Payload(String),
    Conformance(String),
    Mapping(String),
    Poll(String),
}

impl std::fmt::Display for LodWebGpuError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Shader(message) => write!(formatter, "WebGPU LOD shader: {message}"),
            Self::Payload(message) => write!(formatter, "WebGPU LOD payload: {message}"),
            Self::Conformance(message) => write!(formatter, "WebGPU LOD conformance: {message}"),
            Self::Mapping(message) => write!(formatter, "WebGPU LOD mapping: {message}"),
            Self::Poll(message) => write!(formatter, "WebGPU LOD poll: {message}"),
        }
    }
}

impl std::error::Error for LodWebGpuError {}

/// Dynamic pose values uploaded for one classifier dispatch.
#[derive(Clone, Copy, Debug, Default)]
pub struct LodPose<'a> {
    /// Column-major 4x4 matrices, exactly `metrics.num_joints * 16` floats.
    pub joint_matrices: &'a [f32],
    /// Exactly one weight per immutable morph target.
    pub morph_weights: &'a [f32],
}

/// Bounded evidence returned after the shared device conformance matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LodDeviceConformance {
    pub full_pipeline_words: usize,
    pub coherence_words: usize,
    pub compacted_source_words: usize,
    pub compacted_range_words: usize,
    pub indirect_argument_words: usize,
    pub indirect_draws: usize,
}

/// Diagnostic copy of the exact same-device visibility outputs. The retained
/// GPU buffers remain suitable for direct storage/indirect consumption; this
/// owned projection exists only for conformance gates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VisibilityCompactionOutput {
    pub compacted_source_instances: Vec<u32>,
    pub compacted_ranges: Vec<[u32; 5]>,
    pub indirect_arguments: Vec<[u32; 5]>,
}

/// Device-local pipelines shared by every uploaded classifier model.
pub struct LodClassifierDevice {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pass1_pipeline: wgpu::ComputePipeline,
    pass2_pipeline: wgpu::ComputePipeline,
    visibility_count_pipeline: wgpu::ComputePipeline,
    visibility_scan_pipeline: wgpu::ComputePipeline,
    visibility_scatter_pipeline: wgpu::ComputePipeline,
}

/// Retained device buffers for one immutable prepared model and atlas lookup.
pub struct LodClassifierModel {
    prepared: PreparedLodModel,
    joint_capacity: usize,
    subject_rows: usize,
    uniform: wgpu::Buffer,
    joint_matrices: wgpu::Buffer,
    morph_weights: wgpu::Buffer,
    subject_states: wgpu::Buffer,
    packed_records: wgpu::Buffer,
    readback: wgpu::Buffer,
    pass1_bind_group: wgpu::BindGroup,
    pass2_bind_group: wgpu::BindGroup,
}

/// Retained scene shape and output buffers for deterministic visibility
/// compaction. Only `source_visibility` changes with the current pose.
pub struct VisibilityCompactionScene {
    batch_count: u32,
    source_count: u32,
    source_visibility: wgpu::Buffer,
    compacted_source_instances: wgpu::Buffer,
    compacted_ranges: wgpu::Buffer,
    indirect_arguments: wgpu::Buffer,
    count_bind_group: wgpu::BindGroup,
    scan_bind_group: wgpu::BindGroup,
    scatter_bind_group: wgpu::BindGroup,
}

impl LodClassifierDevice {
    /// Compile and retain the two flattened WGSL pipelines on an existing
    /// device. Device creation and adapter policy stay with the application.
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Result<Self, LodWebGpuError> {
        let pass1_source = quilting_shaders::compile_lod_pass1_wgsl()
            .map_err(|error| LodWebGpuError::Shader(error.to_string()))?;
        let pass2_source = quilting_shaders::compile_lod_pass2_wgsl()
            .map_err(|error| LodWebGpuError::Shader(error.to_string()))?;
        let visibility_count_source = quilting_shaders::compile_visibility_count_wgsl()
            .map_err(|error| LodWebGpuError::Shader(error.to_string()))?;
        let visibility_scan_source = quilting_shaders::compile_visibility_scan_wgsl()
            .map_err(|error| LodWebGpuError::Shader(error.to_string()))?;
        let visibility_scatter_source = quilting_shaders::compile_visibility_scatter_wgsl()
            .map_err(|error| LodWebGpuError::Shader(error.to_string()))?;
        let pass1_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("quilting LOD pass one"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(pass1_source)),
        });
        let pass2_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("quilting LOD pass two"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(pass2_source)),
        });
        let visibility_count_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("quilting visibility count"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(visibility_count_source)),
        });
        let visibility_scan_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("quilting visibility scan"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(visibility_scan_source)),
        });
        let visibility_scatter_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("quilting visibility scatter"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(visibility_scatter_source)),
        });
        let pass1_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("quilting LOD pass one"),
            layout: None,
            module: &pass1_module,
            entry_point: Some(quilting_shaders::LOD_PASS1_DEVICE_ENTRY_POINT),
            compilation_options: Default::default(),
            cache: None,
        });
        let pass2_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("quilting LOD pass two"),
            layout: None,
            module: &pass2_module,
            entry_point: Some(quilting_shaders::LOD_PASS2_DEVICE_ENTRY_POINT),
            compilation_options: Default::default(),
            cache: None,
        });
        let visibility_count_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("quilting visibility count"),
                layout: None,
                module: &visibility_count_module,
                entry_point: Some(quilting_shaders::VISIBILITY_COUNT_DEVICE_ENTRY_POINT),
                compilation_options: Default::default(),
                cache: None,
            });
        let visibility_scan_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("quilting visibility scan"),
                layout: None,
                module: &visibility_scan_module,
                entry_point: Some(quilting_shaders::VISIBILITY_SCAN_DEVICE_ENTRY_POINT),
                compilation_options: Default::default(),
                cache: None,
            });
        let visibility_scatter_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("quilting visibility scatter"),
                layout: None,
                module: &visibility_scatter_module,
                entry_point: Some(quilting_shaders::VISIBILITY_SCATTER_DEVICE_ENTRY_POINT),
                compilation_options: Default::default(),
                cache: None,
            });
        Ok(Self {
            device,
            queue,
            pass1_pipeline,
            pass2_pipeline,
            visibility_count_pipeline,
            visibility_scan_pipeline,
            visibility_scatter_pipeline,
        })
    }

    /// Run the same minimum exact matrix on native and browser devices.
    ///
    /// This covers one complete animated-capable two-pass dispatch plus all S3
    /// permutations, visible-only neighbor promotion, invisible records,
    /// priorities, and multiple atlas keys in the coherence pass.
    pub async fn run_conformance_matrix(&self) -> Result<LodDeviceConformance, LodWebGpuError> {
        let prepared = prepare_lod_model(LodModelData {
            positions: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            faces: vec![[0, 1, 2]],
            joint_indices: vec![[0; 4]; 3],
            joint_weights: vec![[0.0; 4]; 3],
            morph_deltas: Vec::new(),
            num_morph_targets: 0,
            face_nodes: vec![0],
        })
        .map_err(LodWebGpuError::Payload)?;
        let atlas = prepare_lod_atlas_lookup([[1, 1, 2]]).map_err(LodWebGpuError::Payload)?;
        let model_words = pack_wgsl_lod_model_words(&prepared).map_err(LodWebGpuError::Payload)?;
        let atlas_words = pack_wgsl_lod_atlas_words(&atlas);
        let expected = reconcile_and_pack_wgsl_lod_pass2(
            &[[1.0, 0.0, 0.0, 1.0]],
            &model_words.adjacency,
            &atlas_words,
        )
        .map_err(LodWebGpuError::Payload)?;
        let dispatch = LodDispatchState {
            subjects: Vec::new(),
            baseline_mobius: identity_mobius(),
            baseline_model: identity_matrix(),
            pole: [0.0; 4],
            mobius_power: 0.0,
            c_norm_sq: 0.0,
            has_pole: 0.0,
        };
        let metrics = WgslLodDispatchMetrics {
            view_projection: identity_matrix(),
            density: 1.0,
            pixel_floor: 0.0,
            max_lod: atlas.max_lod,
            viewport: [1024.0, 1024.0],
            num_joints: 2,
        };
        let mut joint_matrices = Vec::with_capacity(32);
        joint_matrices.extend_from_slice(&identity_matrix());
        joint_matrices.extend_from_slice(&identity_matrix());
        let mut resident = self.upload_model(prepared, &atlas)?;
        let actual = self
            .classify(
                &mut resident,
                &dispatch,
                metrics,
                LodPose {
                    joint_matrices: &joint_matrices,
                    morph_weights: &[],
                },
            )
            .await?;
        if actual != expected {
            return Err(LodWebGpuError::Conformance(format!(
                "full pipeline mismatch: expected {expected:?}, got {actual:?}"
            )));
        }
        let full_pipeline_words = actual.len();

        let pass1 = [
            [1.0, 2.0, 3.0, 11.0],
            [1.0, 3.0, 2.0, 12.0],
            [2.0, 1.0, 3.0, 13.0],
            [3.0, 1.0, 2.0, 14.0],
            [2.0, 3.0, 1.0, 15.0],
            [3.0, 2.0, 1.0, 16.0],
            [4.0, 5.0, 6.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 4.0, 2.0],
            [8.0, 8.0, 8.0, 0.0],
        ];
        let mut adjacency = vec![[u32::MAX, 0, 0, 0]; pass1.len() * 3];
        adjacency[7 * 3] = [8, 2, 0, 0];
        adjacency[7 * 3 + 1] = [9, 0, 0, 0];
        let mut atlas_lut = vec![u8::MAX as u32; 1_200];
        atlas_lut[321] = 17;
        atlas_lut[411] = 29;
        atlas_lut[654] = 31;
        let expected = reconcile_and_pack_wgsl_lod_pass2(&pass1, &adjacency, &atlas_lut)
            .map_err(LodWebGpuError::Payload)?;
        let actual = self
            .reconcile_conformance_records(&pass1, &adjacency, &atlas_lut)
            .await?;
        if actual != expected {
            return Err(LodWebGpuError::Conformance(format!(
                "coherence mismatch: expected {expected:?}, got {actual:?}"
            )));
        }

        let batch_zero_count = 130u32;
        let mut source_visibility = Vec::with_capacity(137);
        source_visibility.extend((0..batch_zero_count).map(|index| u8::from(index % 3 != 0)));
        source_visibility.extend([1, 1, 1, 1, 0, 1, 1]);
        let compaction_words = WgslVisibilityCompactionSceneWords {
            uniform: [3, 137, 0, 0],
            batches: vec![[0, 130, 6, 0], [130, 3, 12, 0], [133, 4, 18, 0]],
            source_eligibility: [vec![1; 130], vec![0; 3], vec![1; 4]].concat(),
        };
        let mut expected_sources = (0..batch_zero_count)
            .filter(|index| index % 3 != 0)
            .collect::<Vec<_>>();
        expected_sources.extend([133, 135, 136]);
        let expected_ranges = vec![[0, 0, 130, 0, 86], [1, 130, 3, 86, 0], [2, 133, 4, 86, 3]];
        let expected_indirect = vec![[6, 86, 0, 0, 0], [12, 0, 0, 0, 0], [18, 3, 0, 0, 0]];
        let mut compaction = self.upload_visibility_compaction_scene(compaction_words)?;
        let compacted = self
            .compact_visibility(&mut compaction, &source_visibility)
            .await?;
        if compacted.compacted_source_instances != expected_sources
            || compacted.compacted_ranges != expected_ranges
            || compacted.indirect_arguments != expected_indirect
        {
            return Err(LodWebGpuError::Conformance(format!(
                "visibility compaction mismatch: expected sources {expected_sources:?}, ranges \
                 {expected_ranges:?}, indirect {expected_indirect:?}; got {compacted:?}"
            )));
        }
        let indirect_draws = self
            .validate_indirect_draw_conformance(&compaction, &source_visibility)
            .await?;

        Ok(LodDeviceConformance {
            full_pipeline_words,
            coherence_words: actual.len(),
            compacted_source_words: compacted.compacted_source_instances.len(),
            compacted_range_words: compacted.compacted_ranges.len() * 5,
            indirect_argument_words: compacted.indirect_arguments.len() * 5,
            indirect_draws,
        })
    }

    /// Upload retained batch shape and static eligibility once. Output buffers
    /// include `INDIRECT` usage now, so a later render pass can consume the
    /// generated arguments without changing this residency contract.
    pub fn upload_visibility_compaction_scene(
        &self,
        words: WgslVisibilityCompactionSceneWords,
    ) -> Result<VisibilityCompactionScene, LodWebGpuError> {
        let batch_count = words.uniform[0];
        let source_count = words.uniform[1];
        if words.uniform[2..] != [0, 0]
            || words.batches.len() != batch_count as usize
            || words.source_eligibility.len() != source_count as usize
            || words.source_eligibility.iter().any(|&value| value > 1)
        {
            return Err(LodWebGpuError::Payload(
                "visibility compaction scene shape is malformed".to_string(),
            ));
        }
        let mut expected_first = 0u32;
        for batch in &words.batches {
            if batch[0] != expected_first || batch[3] != 0 {
                return Err(LodWebGpuError::Payload(
                    "visibility compaction batch ranges are not canonical".to_string(),
                ));
            }
            expected_first = expected_first.checked_add(batch[1]).ok_or_else(|| {
                LodWebGpuError::Payload(
                    "visibility compaction source count exceeds u32".to_string(),
                )
            })?;
        }
        if expected_first != source_count {
            return Err(LodWebGpuError::Payload(
                "visibility compaction batches do not cover the source stream".to_string(),
            ));
        }

        let uniform = buffer_init_or_zero(
            &self.device,
            "visibility compaction uniform",
            bytemuck::cast_slice(&words.uniform),
            wgpu::BufferUsages::UNIFORM,
        );
        let batches = buffer_init_or_zero(
            &self.device,
            "visibility compaction batches",
            bytemuck::cast_slice(&words.batches),
            wgpu::BufferUsages::STORAGE,
        );
        let source_eligibility = buffer_init_or_zero(
            &self.device,
            "visibility compaction eligibility",
            bytemuck::cast_slice(&words.source_eligibility),
            wgpu::BufferUsages::STORAGE,
        );
        let source_bytes = u64::from(source_count)
            .checked_mul(PACKED_RECORD_BYTES)
            .ok_or_else(|| {
                LodWebGpuError::Payload("visibility source buffer is too large".to_string())
            })?;
        let batch_count_bytes = u64::from(batch_count)
            .checked_mul(PACKED_RECORD_BYTES)
            .ok_or_else(|| {
                LodWebGpuError::Payload("visibility batch count buffer is too large".to_string())
            })?;
        let range_bytes = u64::from(batch_count)
            .checked_mul(VISIBILITY_RANGE_RECORD_BYTES)
            .ok_or_else(|| {
                LodWebGpuError::Payload("visibility range buffer is too large".to_string())
            })?;
        let indirect_bytes = u64::from(batch_count)
            .checked_mul(INDEXED_INDIRECT_RECORD_BYTES)
            .ok_or_else(|| {
                LodWebGpuError::Payload("visibility indirect buffer is too large".to_string())
            })?;
        debug_assert_eq!(
            std::mem::size_of_val(&words.uniform) as u64,
            VISIBILITY_UNIFORM_BYTES
        );
        debug_assert!(words
            .batches
            .iter()
            .all(|record| std::mem::size_of_val(record) as u64 == VISIBILITY_BATCH_RECORD_BYTES));

        let source_visibility = gpu_buffer(
            &self.device,
            "visibility compaction current visibility",
            source_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let batch_counts = gpu_buffer(
            &self.device,
            "visibility compaction batch counts",
            batch_count_bytes,
            wgpu::BufferUsages::STORAGE,
        );
        let compacted_source_instances = gpu_buffer(
            &self.device,
            "visibility compacted source instances",
            source_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let compacted_ranges = gpu_buffer(
            &self.device,
            "visibility compacted ranges",
            range_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let indirect_arguments = gpu_buffer(
            &self.device,
            "visibility indirect arguments",
            indirect_bytes,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_SRC,
        );
        let count_layout = self.visibility_count_pipeline.get_bind_group_layout(0);
        let count_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("visibility count bindings"),
            layout: &count_layout,
            entries: &[
                bind(0, &uniform),
                bind(1, &batches),
                bind(2, &source_eligibility),
                bind(3, &source_visibility),
                bind(4, &batch_counts),
            ],
        });
        let scan_layout = self.visibility_scan_pipeline.get_bind_group_layout(0);
        let scan_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("visibility scan bindings"),
            layout: &scan_layout,
            entries: &[
                bind(0, &uniform),
                bind(1, &batches),
                bind(2, &batch_counts),
                bind(3, &compacted_ranges),
                bind(4, &indirect_arguments),
            ],
        });
        let scatter_layout = self.visibility_scatter_pipeline.get_bind_group_layout(0);
        let scatter_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("visibility scatter bindings"),
            layout: &scatter_layout,
            entries: &[
                bind(0, &uniform),
                bind(1, &batches),
                bind(2, &source_eligibility),
                bind(3, &source_visibility),
                bind(4, &compacted_ranges),
                bind(5, &compacted_source_instances),
            ],
        });

        Ok(VisibilityCompactionScene {
            batch_count,
            source_count,
            source_visibility,
            compacted_source_instances,
            compacted_ranges,
            indirect_arguments,
            count_bind_group,
            scan_bind_group,
            scatter_bind_group,
        })
    }

    /// Validate and upload a CPU/shadow visibility fixture. A same-device
    /// producer can write [`VisibilityCompactionScene::source_visibility_buffer`]
    /// directly and skip this transfer.
    pub fn write_source_visibility(
        &self,
        scene: &VisibilityCompactionScene,
        source_visibility: &[u8],
    ) -> Result<(), LodWebGpuError> {
        let visibility = pack_wgsl_source_visibility_words(source_visibility, scene.source_count)
            .map_err(LodWebGpuError::Payload)?;
        if !visibility.is_empty() {
            self.queue.write_buffer(
                &scene.source_visibility,
                0,
                bytemuck::cast_slice(&visibility),
            );
        }
        Ok(())
    }

    /// Append count, deterministic batch scan, and stable parallel scatter to
    /// an application-owned command encoder. A caller can encode its visibility
    /// producer before this and indirect render passes after this, preserving
    /// one ordered GPU submission with no map/copy boundary.
    pub fn encode_visibility_compaction(
        &self,
        scene: &VisibilityCompactionScene,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        if scene.batch_count == 0 {
            return;
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("quilting visibility count"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.visibility_count_pipeline);
            pass.set_bind_group(0, &scene.count_bind_group, &[]);
            pass.dispatch_workgroups(scene.batch_count, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("quilting visibility scan"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.visibility_scan_pipeline);
            pass.set_bind_group(0, &scene.scan_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("quilting visibility scatter"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.visibility_scatter_pipeline);
            pass.set_bind_group(0, &scene.scatter_bind_group, &[]);
            pass.dispatch_workgroups(scene.batch_count, 1, 1);
        }
    }

    /// Convenience submission for CPU/shadow visibility input. This method
    /// returns immediately after queue submission and performs no readback.
    pub fn compact_visibility_on_device(
        &self,
        scene: &VisibilityCompactionScene,
        source_visibility: &[u8],
    ) -> Result<(), LodWebGpuError> {
        self.write_source_visibility(scene, source_visibility)?;
        if scene.batch_count == 0 {
            return Ok(());
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("quilting visibility compaction"),
            });
        self.encode_visibility_compaction(scene, &mut encoder);
        self.queue.submit([encoder.finish()]);
        Ok(())
    }

    /// Diagnostic wrapper around the same no-readback encoder path. Temporary
    /// staging buffers exist only for conformance calls and are not retained by
    /// live scene residency.
    pub async fn compact_visibility(
        &self,
        scene: &mut VisibilityCompactionScene,
        source_visibility: &[u8],
    ) -> Result<VisibilityCompactionOutput, LodWebGpuError> {
        self.write_source_visibility(scene, source_visibility)?;
        if scene.batch_count == 0 {
            return Ok(VisibilityCompactionOutput {
                compacted_source_instances: Vec::new(),
                compacted_ranges: Vec::new(),
                indirect_arguments: Vec::new(),
            });
        }

        let source_bytes = u64::from(scene.source_count) * PACKED_RECORD_BYTES;
        let range_bytes = u64::from(scene.batch_count) * VISIBILITY_RANGE_RECORD_BYTES;
        let indirect_bytes = u64::from(scene.batch_count) * INDEXED_INDIRECT_RECORD_BYTES;
        let source_readback = gpu_buffer(
            &self.device,
            "visibility source diagnostic readback",
            source_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let range_readback = gpu_buffer(
            &self.device,
            "visibility range diagnostic readback",
            range_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let indirect_readback = gpu_buffer(
            &self.device,
            "visibility indirect diagnostic readback",
            indirect_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("quilting visibility diagnostic compaction"),
            });
        self.encode_visibility_compaction(scene, &mut encoder);
        if source_bytes != 0 {
            encoder.copy_buffer_to_buffer(
                &scene.compacted_source_instances,
                0,
                &source_readback,
                0,
                source_bytes,
            );
        }
        encoder.copy_buffer_to_buffer(&scene.compacted_ranges, 0, &range_readback, 0, range_bytes);
        encoder.copy_buffer_to_buffer(
            &scene.indirect_arguments,
            0,
            &indirect_readback,
            0,
            indirect_bytes,
        );
        self.queue.submit([encoder.finish()]);

        let compacted_ranges = words_to_five_records(
            self.readback_words(&range_readback, range_bytes).await?,
            "visibility range",
        )?;
        let indirect_arguments = words_to_five_records(
            self.readback_words(&indirect_readback, indirect_bytes)
                .await?,
            "visibility indirect",
        )?;
        let survivor_count = compacted_ranges
            .last()
            .map_or(0u32, |range| range[3].saturating_add(range[4]));
        if survivor_count > scene.source_count {
            return Err(LodWebGpuError::Conformance(
                "visibility compaction emitted too many survivors".to_string(),
            ));
        }
        let mut compacted_source_instances = if source_bytes == 0 {
            Vec::new()
        } else {
            self.readback_words(&source_readback, source_bytes).await?
        };
        compacted_source_instances.truncate(survivor_count as usize);
        Ok(VisibilityCompactionOutput {
            compacted_source_instances,
            compacted_ranges,
            indirect_arguments,
        })
    }

    /// Prove that portable zero-based arguments emitted by compaction can be
    /// consumed immediately by real indexed-indirect draws on this device.
    /// This is part of the shared native/browser conformance matrix only.
    async fn validate_indirect_draw_conformance(
        &self,
        scene: &VisibilityCompactionScene,
        source_visibility: &[u8],
    ) -> Result<usize, LodWebGpuError> {
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("visibility indirect conformance shader"),
            source: wgpu::ShaderSource::Wgsl(
                r#"
                    @vertex
                    fn vertex_main(@builtin(vertex_index) vertex: u32) -> @builtin(position) vec4<f32> {
                        let positions = array<vec2<f32>, 3>(
                            vec2<f32>(-0.5, -0.5),
                            vec2<f32>(0.5, -0.5),
                            vec2<f32>(0.0, 0.5),
                        );
                        return vec4<f32>(positions[vertex % 3u], 0.0, 1.0);
                    }

                    @fragment
                    fn fragment_main() -> @location(0) vec4<f32> {
                        return vec4<f32>(0.25, 0.75, 1.0, 1.0);
                    }
                "#
                .into(),
            ),
        });
        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("visibility indirect conformance pipeline"),
                layout: None,
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vertex_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fragment_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });
        let indices = [0u32, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2];
        let index_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("visibility indirect conformance indices"),
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsages::INDEX,
            });
        let target = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("visibility indirect conformance target"),
            size: wgpu::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());

        self.write_source_visibility(scene, source_visibility)?;
        let error_scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("visibility compaction to indirect draw conformance"),
            });
        self.encode_visibility_compaction(scene, &mut encoder);
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("visibility indirect conformance pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target_view,
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
            pass.set_pipeline(&pipeline);
            pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            for batch_index in 0..scene.batch_count {
                pass.draw_indexed_indirect(
                    &scene.indirect_arguments,
                    u64::from(batch_index) * INDEXED_INDIRECT_RECORD_BYTES,
                );
            }
        }
        let submission = self.queue.submit([encoder.finish()]);
        #[cfg(not(target_arch = "wasm32"))]
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .map_err(|error| LodWebGpuError::Poll(error.to_string()))?;
        #[cfg(target_arch = "wasm32")]
        let _ = submission;
        if let Some(error) = error_scope.pop().await {
            return Err(LodWebGpuError::Conformance(format!(
                "compaction-to-indirect submission failed validation: {error}"
            )));
        }
        Ok(scene.batch_count as usize)
    }

    /// Upload immutable geometry/topology once and allocate retained dynamic
    /// buffers. The current implementation deliberately keeps readback for
    /// shadow conformance; an authoritative backend will consume packed words
    /// on-device instead.
    pub fn upload_model(
        &self,
        prepared: PreparedLodModel,
        atlas: &LodAtlasLookup,
    ) -> Result<LodClassifierModel, LodWebGpuError> {
        let words = pack_wgsl_lod_model_words(&prepared).map_err(LodWebGpuError::Payload)?;
        let atlas_words = pack_wgsl_lod_atlas_words(atlas);
        let face_count = prepared.residency.num_faces;
        let subject_rows = words
            .faces
            .iter()
            .map(|face| face[3] as usize)
            .max()
            .map_or(1, |row| row + 1);
        let joint_capacity = prepared
            .model
            .joint_indices
            .iter()
            .zip(&prepared.model.joint_weights)
            .flat_map(|(indices, weights)| indices.iter().zip(weights))
            .filter(|(_, weight)| **weight >= 1e-6)
            .map(|(joint, _)| usize::from(*joint) + 1)
            .max()
            .unwrap_or(1);

        let faces = storage_buffer(&self.device, "LOD faces", &words.faces, false);
        let positions = storage_buffer(&self.device, "LOD positions", &words.positions, false);
        let skinning = storage_buffer(&self.device, "LOD skinning", &words.skinning, false);
        let zero_record = [[0u32; 4]];
        let morph_source = if words.morph_deltas.is_empty() {
            zero_record.as_slice()
        } else {
            words.morph_deltas.as_slice()
        };
        let morph_deltas = storage_buffer(&self.device, "LOD morph deltas", morph_source, false);
        let adjacency = storage_buffer(&self.device, "LOD adjacency", &words.adjacency, false);
        let atlas_lut = storage_buffer(&self.device, "LOD atlas lookup", &atlas_words, false);

        let uniform = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LOD dispatch uniform"),
            size: DISPATCH_UNIFORM_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let joint_matrices = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LOD joint matrices"),
            size: joint_capacity as u64 * JOINT_MATRIX_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let morph_weights = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LOD morph weights"),
            size: prepared.model.num_morph_targets.max(1) as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let subject_states = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LOD subject states"),
            size: subject_rows as u64 * SUBJECT_RECORD_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let pass1_records = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LOD pass one records"),
            size: face_count as u64 * PASS1_RECORD_BYTES,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let packed_records = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LOD packed records"),
            size: face_count as u64 * PACKED_RECORD_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LOD diagnostic readback"),
            size: face_count as u64 * PACKED_RECORD_BYTES,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pass1_layout = self.pass1_pipeline.get_bind_group_layout(0);
        let pass1_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("LOD pass one bindings"),
            layout: &pass1_layout,
            entries: &[
                bind(0, &uniform),
                bind(1, &faces),
                bind(2, &positions),
                bind(3, &skinning),
                bind(4, &joint_matrices),
                bind(5, &morph_deltas),
                bind(6, &morph_weights),
                bind(7, &subject_states),
                bind(8, &pass1_records),
            ],
        });
        let pass2_layout = self.pass2_pipeline.get_bind_group_layout(0);
        let pass2_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("LOD pass two bindings"),
            layout: &pass2_layout,
            entries: &[
                bind(0, &uniform),
                bind(1, &pass1_records),
                bind(2, &adjacency),
                bind(3, &atlas_lut),
                bind(4, &packed_records),
            ],
        });

        Ok(LodClassifierModel {
            prepared,
            joint_capacity,
            subject_rows,
            uniform,
            joint_matrices,
            morph_weights,
            subject_states,
            packed_records,
            readback,
            pass1_bind_group,
            pass2_bind_group,
        })
    }

    /// Execute both passes in one command buffer and return the diagnostic
    /// packed words only after the device signals the map callback.
    pub async fn classify(
        &self,
        model: &mut LodClassifierModel,
        dispatch: &LodDispatchState,
        metrics: WgslLodDispatchMetrics,
        pose: LodPose<'_>,
    ) -> Result<Vec<u32>, LodWebGpuError> {
        let num_joints = metrics.num_joints as usize;
        if pose.joint_matrices.len() != num_joints.saturating_mul(16) {
            return Err(LodWebGpuError::Payload(
                "joint pose does not match the dispatch joint count".to_string(),
            ));
        }
        if pose.morph_weights.len() != model.prepared.model.num_morph_targets {
            return Err(LodWebGpuError::Payload(
                "morph weights do not match immutable targets".to_string(),
            ));
        }
        let subject_words = pack_wgsl_lod_subject_words(&model.prepared, dispatch)
            .map_err(LodWebGpuError::Payload)?;
        if subject_words.len() != model.subject_rows {
            return Err(LodWebGpuError::Payload(
                "subject table changed immutable shape".to_string(),
            ));
        }
        let uniform_words = pack_wgsl_lod_dispatch_words(&model.prepared, dispatch, metrics)
            .map_err(LodWebGpuError::Payload)?;
        self.queue
            .write_buffer(&model.uniform, 0, bytemuck::cast_slice(&uniform_words));
        self.queue.write_buffer(
            &model.subject_states,
            0,
            bytemuck::cast_slice(&subject_words),
        );
        // The retained buffer ends at the highest joint referenced by a
        // nonzero influence. A glTF skin may contain additional unused joints;
        // preserve the full dispatch count while uploading only the prefix the
        // shader can actually index for this model.
        let resident_joint_floats = num_joints.min(model.joint_capacity) * 16;
        if resident_joint_floats != 0 {
            self.queue.write_buffer(
                &model.joint_matrices,
                0,
                bytemuck::cast_slice(&pose.joint_matrices[..resident_joint_floats]),
            );
        }
        if !pose.morph_weights.is_empty() {
            self.queue.write_buffer(
                &model.morph_weights,
                0,
                bytemuck::cast_slice(pose.morph_weights),
            );
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("quilting LOD classifier"),
            });
        let groups = (model.prepared.residency.num_faces as u32).div_ceil(LOD_WORKGROUP_SIZE);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("quilting LOD pass one"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pass1_pipeline);
            pass.set_bind_group(0, &model.pass1_bind_group, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("quilting LOD pass two"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pass2_pipeline);
            pass.set_bind_group(0, &model.pass2_bind_group, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        let readback_bytes = model.prepared.residency.num_faces as u64 * PACKED_RECORD_BYTES;
        encoder.copy_buffer_to_buffer(&model.packed_records, 0, &model.readback, 0, readback_bytes);
        self.queue.submit([encoder.finish()]);

        self.readback_words(&model.readback, readback_bytes).await
    }

    /// Execute only the coherence/packing pass over caller-supplied
    /// intermediates. This diagnostic boundary exists to compare the device
    /// shader with the independent CPU pass-two oracle over a broad fixture
    /// matrix; it allocates temporary buffers and is not a runtime fast path.
    pub async fn reconcile_conformance_records(
        &self,
        pass1_records: &[[f32; 4]],
        adjacency: &[[u32; 4]],
        atlas_lut: &[u32],
    ) -> Result<Vec<u32>, LodWebGpuError> {
        let face_count = pass1_records.len();
        if face_count == 0
            || adjacency.len() != face_count.saturating_mul(3)
            || atlas_lut.len() < 1_200
            || atlas_lut.iter().any(|&index| index > u8::MAX as u32)
        {
            return Err(LodWebGpuError::Payload(
                "pass-two conformance payload is malformed".to_string(),
            ));
        }
        let face_count_u32 = u32::try_from(face_count).map_err(|_| {
            LodWebGpuError::Payload("pass-two conformance face count exceeds u32".to_string())
        })?;
        let mut uniform_words = [0u32; 68];
        uniform_words[64] = face_count_u32;
        let uniform = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("LOD pass two conformance uniform"),
                contents: bytemuck::cast_slice(&uniform_words),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let pass1 = storage_buffer(
            &self.device,
            "LOD pass two conformance records",
            pass1_records,
            false,
        );
        let adjacency = storage_buffer(
            &self.device,
            "LOD pass two conformance adjacency",
            adjacency,
            false,
        );
        let atlas = storage_buffer(
            &self.device,
            "LOD pass two conformance atlas",
            atlas_lut,
            false,
        );
        let output_bytes = face_count as u64 * PACKED_RECORD_BYTES;
        let output = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LOD pass two conformance output"),
            size: output_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LOD pass two conformance readback"),
            size: output_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layout = self.pass2_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("LOD pass two conformance bindings"),
            layout: &layout,
            entries: &[
                bind(0, &uniform),
                bind(1, &pass1),
                bind(2, &adjacency),
                bind(3, &atlas),
                bind(4, &output),
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("LOD pass two conformance encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LOD pass two conformance dispatch"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pass2_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(face_count_u32.div_ceil(LOD_WORKGROUP_SIZE), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output_bytes);
        self.queue.submit([encoder.finish()]);

        self.readback_words(&readback, output_bytes).await
    }

    async fn readback_words(
        &self,
        readback: &wgpu::Buffer,
        readback_bytes: u64,
    ) -> Result<Vec<u32>, LodWebGpuError> {
        let slice = readback.slice(..readback_bytes);

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
        let view = slice.get_mapped_range();
        let packed = bytemuck::cast_slice::<u8, u32>(&view).to_vec();
        drop(view);
        readback.unmap();
        Ok(packed)
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }
}

impl VisibilityCompactionScene {
    pub fn batch_count(&self) -> u32 {
        self.batch_count
    }

    pub fn source_count(&self) -> u32 {
        self.source_count
    }

    /// One `u32` visibility flag per stable source instance. A culling or LOD
    /// compute pass may bind this as writable storage before compaction.
    pub fn source_visibility_buffer(&self) -> &wgpu::Buffer {
        &self.source_visibility
    }

    /// Stable source-instance indirection consumed by a render vertex stage.
    pub fn compacted_source_instances_buffer(&self) -> &wgpu::Buffer {
        &self.compacted_source_instances
    }

    /// One 20-byte source/compacted range record per retained batch.
    pub fn compacted_ranges_buffer(&self) -> &wgpu::Buffer {
        &self.compacted_ranges
    }

    /// One exact 20-byte `DrawIndexedIndirect` record per retained batch.
    pub fn indirect_arguments_buffer(&self) -> &wgpu::Buffer {
        &self.indirect_arguments
    }
}

fn words_to_five_records(words: Vec<u32>, label: &str) -> Result<Vec<[u32; 5]>, LodWebGpuError> {
    if !words.len().is_multiple_of(5) {
        return Err(LodWebGpuError::Mapping(format!(
            "{label} readback does not contain five-word records"
        )));
    }
    Ok(words
        .chunks_exact(5)
        .map(|record| [record[0], record[1], record[2], record[3], record[4]])
        .collect())
}

fn bind(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn buffer_init_or_zero(
    device: &wgpu::Device,
    label: &'static str,
    contents: &[u8],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    let zero = [0u8; 4];
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: if contents.is_empty() { &zero } else { contents },
        usage,
    })
}

fn gpu_buffer(
    device: &wgpu::Device,
    label: &'static str,
    size: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(PACKED_RECORD_BYTES),
        usage,
        mapped_at_creation: false,
    })
}

fn storage_buffer<T: bytemuck::Pod>(
    device: &wgpu::Device,
    label: &'static str,
    records: &[T],
    copy_dst: bool,
) -> wgpu::Buffer {
    let mut usage = wgpu::BufferUsages::STORAGE;
    if copy_dst {
        usage |= wgpu::BufferUsages::COPY_DST;
    }
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(records),
        usage,
    })
}

/// Request a browser WebGPU device and execute the same exact matrix as the
/// native hardware gate. Enabled only for the standalone browser harness.
#[cfg(all(target_arch = "wasm32", feature = "browser-conformance"))]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_browser_lod_conformance() -> Result<String, wasm_bindgen::JsValue> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .map_err(browser_error)?;
    let adapter_info = adapter.get_info();
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("quilting browser LOD conformance"),
            ..Default::default()
        })
        .await
        .map_err(browser_error)?;
    let classifier = LodClassifierDevice::new(device, queue).map_err(browser_error)?;
    let report = classifier
        .run_conformance_matrix()
        .await
        .map_err(browser_error)?;
    Ok(format!(
        "adapter={} backend={:?} full_pipeline_words={} coherence_words={} \
         compacted_source_words={} compacted_range_words={} indirect_argument_words={} \
         indirect_draws={}",
        adapter_info.name,
        adapter_info.backend,
        report.full_pipeline_words,
        report.coherence_words,
        report.compacted_source_words,
        report.compacted_range_words,
        report.indirect_argument_words,
        report.indirect_draws,
    ))
}

#[cfg(all(target_arch = "wasm32", feature = "browser-conformance"))]
fn browser_error(error: impl std::fmt::Display) -> wasm_bindgen::JsValue {
    wasm_bindgen::JsValue::from_str(&error.to_string())
}
