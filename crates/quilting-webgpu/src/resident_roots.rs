//! Retained root-only geometry planning and direct source-face preparation.
//!
//! Adaptive dyadic leaves remain a separate sparse overlay because their edge
//! LOD and permutation cannot be reconstructed from one source-face record.

use super::*;

/// Retained root-only geometry buckets derived from packed resident LOD.
pub struct ResidentGeometryBucketScene {
    pub(super) model_identity: u64,
    pub(super) face_count: u32,
    pub(super) atlas_count: u32,
    pub(super) bucket_count: u32,
    pub(super) chunk_count: u32,
    pub(super) eligibility_word_count: u32,
    pub(super) root_eligibility: wgpu::Buffer,
    pub(super) _chunk_counts: wgpu::Buffer,
    pub(super) _chunk_offsets: wgpu::Buffer,
    pub(super) _bucket_counts: wgpu::Buffer,
    pub(super) compacted_faces: wgpu::Buffer,
    pub(super) bucket_ranges: wgpu::Buffer,
    pub(super) indirect_arguments: wgpu::Buffer,
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
    pub indirect_arguments: Vec<[u32; 5]>,
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

/// Atomic retained aggregate for direct source-indexed root preparation.
/// Topology emission and the ordinary rational-QB preparation pass share the
/// same device buffer; no CPU topology upload exists in this path.
pub struct ResidentRootPreparationScene {
    pub(super) topology: ResidentRootTopologyScene,
    pub(super) patches: PatchPreparationScene,
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

    pub fn compacted_faces_buffer(&self) -> &wgpu::Buffer {
        &self.compacted_faces
    }

    pub fn bucket_ranges_buffer(&self) -> &wgpu::Buffer {
        &self.bucket_ranges
    }

    pub fn indirect_arguments_buffer(&self) -> &wgpu::Buffer {
        &self.indirect_arguments
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
}

impl LodClassifierDevice {
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
        self.upload_resident_root_preparation_words(model, words)
    }

    pub(super) fn upload_resident_root_preparation_words(
        &self,
        model: &LodClassifierModel,
        words: WgslResidentRootPreparationSceneWords,
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
        Ok(ResidentRootPreparationScene { topology, patches })
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
    ) -> Result<ResidentGeometryBucketScene, LodWebGpuError> {
        let face_count = u32::try_from(model.prepared.residency.num_faces)
            .map_err(|_| LodWebGpuError::Payload("resident geometry faces exceed u32".into()))?;
        self.upload_resident_geometry_bucket_scene_for_records(
            model.identity,
            face_count,
            &model.atlas_keys,
            &model.resident.packed_records,
            atlas,
        )
    }

    fn upload_resident_geometry_bucket_scene_for_records(
        &self,
        model_identity: u64,
        face_count: u32,
        atlas_keys: &[[u32; 3]],
        resident_records: &wgpu::Buffer,
        atlas: &PackedPatchAtlas,
    ) -> Result<ResidentGeometryBucketScene, LodWebGpuError> {
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
        let indirect_arguments = gpu_buffer(
            &self.device,
            "resident geometry indirect arguments",
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
                bind(2, &root_eligibility),
                bind(4, &chunk_counts),
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
                bind(8, &indirect_arguments),
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
                bind(2, &root_eligibility),
                bind(5, &chunk_offsets),
                bind(7, &bucket_ranges),
                bind(9, &compacted_faces),
            ],
        });
        Ok(ResidentGeometryBucketScene {
            model_identity,
            face_count,
            atlas_count,
            bucket_count,
            chunk_count,
            eligibility_word_count,
            root_eligibility,
            _chunk_counts: chunk_counts,
            _chunk_offsets: chunk_offsets,
            _bucket_counts: bucket_counts,
            compacted_faces,
            bucket_ranges,
            indirect_arguments,
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
        self.queue
            .write_buffer(&scene.root_eligibility, 0, bytemuck::cast_slice(words));
        Ok(())
    }

    pub fn write_resident_root_suppression(
        &self,
        scene: &ResidentGeometryBucketScene,
        suppressed_faces: &[u32],
    ) -> Result<(), LodWebGpuError> {
        let words = pack_wgsl_root_eligibility_bits(scene.face_count as usize, suppressed_faces)
            .map_err(LodWebGpuError::Payload)?;
        self.write_resident_root_eligibility_bits(scene, &words)
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
        let indirect_readback = gpu_buffer(
            &self.device,
            "resident geometry indirect diagnostic readback",
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
            &scene.indirect_arguments,
            0,
            &indirect_readback,
            0,
            indirect_bytes,
        );
        self.queue.submit([encoder.finish()]);
        let bucket_ranges = words_to_five_records(
            self.readback_words(&range_readback, range_bytes).await?,
            "resident geometry range",
        )?;
        let indirect_arguments = words_to_five_records(
            self.readback_words(&indirect_readback, indirect_bytes)
                .await?,
            "resident geometry indirect",
        )?;
        let survivor_count = bucket_ranges
            .last()
            .map_or(0u32, |range| range[3].saturating_add(range[4]));
        let mut compacted_faces = self.readback_words(&face_readback, face_bytes).await?;
        compacted_faces.truncate(survivor_count as usize);
        Ok(ResidentGeometryBucketOutput {
            compacted_faces,
            bucket_ranges,
            indirect_arguments,
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
                1, 2, 4, 6, 3, 0, 0, 1, 1, 1, 0, 3, 0, 0, 1, 1, 2, 3, 3, 0, 0,
            ],
            &[1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            &[0, 1, 2, 0, 1, 2, 0, 1, 2],
            &[],
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
        let scene = self.upload_resident_geometry_bucket_scene_for_records(
            identity,
            face_count as u32,
            &atlas_lookup.keys,
            &packed_buffer,
            &atlas,
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
        let atlas_draws = [[0, 3, 0, 0], [3, 3, 0, 0], [6, 3, 0, 0]];
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
            let expected =
                wgsl_resident_geometry_bucket_oracle_words(&packed, &atlas_draws, &eligibility)
                    .map_err(LodWebGpuError::Conformance)?;
            let actual = self
                .resident_geometry_buckets_for_diagnostics(&scene, &resident, &suppressed)
                .await?;
            if actual.compacted_faces != expected.compacted_faces
                || actual.bucket_ranges != expected.bucket_ranges
                || actual.indirect_arguments != expected.indirect_arguments
            {
                return Err(LodWebGpuError::Conformance(format!(
                    "resident geometry bucket mismatch for suppression {suppressed:?}: expected {expected:?}, got {actual:?}",
                )));
            }
            compared_words += actual.compacted_faces.len()
                + actual.bucket_ranges.len() * 5
                + actual.indirect_arguments.len() * 5;
        }
        Ok(compared_words)
    }
}
