//! Shadow WebGPU execution for Quilting's two-pass LOD classifier.
//!
//! This crate is intentionally outside the current WebGL2 authority path. It
//! consumes the exact backend-neutral payload frozen by `quilting-renderer`,
//! retains device pipelines and model buffers, and returns packed words for
//! conformance comparison. It owns no scene, FRP, or replicated state.

use futures_channel::oneshot;
use quilting_renderer::compute::{
    pack_wgsl_lod_atlas_words, pack_wgsl_lod_dispatch_words, pack_wgsl_lod_model_words,
    pack_wgsl_lod_subject_words, LodAtlasLookup, LodDispatchState, PreparedLodModel,
    WgslLodDispatchMetrics,
};
use std::borrow::Cow;
use wgpu::util::DeviceExt;

const LOD_WORKGROUP_SIZE: u32 = 64;
const DISPATCH_UNIFORM_BYTES: u64 = 272;
const SUBJECT_RECORD_BYTES: u64 = 160;
const JOINT_MATRIX_BYTES: u64 = 64;
const PASS1_RECORD_BYTES: u64 = 16;
const PACKED_RECORD_BYTES: u64 = 4;

#[derive(Debug)]
pub enum LodWebGpuError {
    Shader(String),
    Payload(String),
    Mapping(String),
    Poll(String),
}

impl std::fmt::Display for LodWebGpuError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Shader(message) => write!(formatter, "WebGPU LOD shader: {message}"),
            Self::Payload(message) => write!(formatter, "WebGPU LOD payload: {message}"),
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

/// Device-local pipelines shared by every uploaded classifier model.
pub struct LodClassifierDevice {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pass1_pipeline: wgpu::ComputePipeline,
    pass2_pipeline: wgpu::ComputePipeline,
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

impl LodClassifierDevice {
    /// Compile and retain the two flattened WGSL pipelines on an existing
    /// device. Device creation and adapter policy stay with the application.
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Result<Self, LodWebGpuError> {
        let pass1_source = quilting_shaders::compile_lod_pass1_wgsl()
            .map_err(|error| LodWebGpuError::Shader(error.to_string()))?;
        let pass2_source = quilting_shaders::compile_lod_pass2_wgsl()
            .map_err(|error| LodWebGpuError::Shader(error.to_string()))?;
        let pass1_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("quilting LOD pass one"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(pass1_source)),
        });
        let pass2_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("quilting LOD pass two"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(pass2_source)),
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
        Ok(Self {
            device,
            queue,
            pass1_pipeline,
            pass2_pipeline,
        })
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

        let slice = model.readback.slice(..readback_bytes);
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
        model.readback.unmap();
        Ok(packed)
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }
}

fn bind(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
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
