//! Sparse adaptive dyadic replacement rendering over device-resident roots.
//!
//! The overlay owns only changed leaf topology, affine subject rows,
//! visibility, prepared output, and indirect batches. Immutable source-face
//! records stay in the resident-root preparation scene and are shared through
//! an `Arc`, so publishing a screen partition never rebuilds the baseline.

use super::*;

/// Retained GPU resources for only the adaptive replacement batches in one
/// validated backend-neutral scene.
pub struct AdaptiveOverlayScene {
    model_identity: u64,
    scene_revision: u64,
    suppressed_root_faces: Vec<u32>,
    source_batch_indices: Vec<u32>,
    batches: Vec<RenderBatchSnapshot>,
    pub(super) patches: PatchPreparationScene,
    visibility: VisibilityCompactionScene,
    face_visibility: FaceVisibilityExpansionScene,
    bindings: PatchRenderBindings,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdaptiveOverlayFrameEncoding {
    pub indirect_draw_calls: u32,
    pub source_patch_count: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidentAdaptiveFrameEncoding {
    pub logical_submission: RenderSubmissionStats,
    pub roots: ResidentRootFrameEncoding,
    pub overlay: Option<AdaptiveOverlayFrameEncoding>,
}

impl AdaptiveOverlayScene {
    pub fn scene_revision(&self) -> u64 {
        self.scene_revision
    }

    pub fn batch_count(&self) -> u32 {
        self.visibility.batch_count
    }

    pub fn patch_count(&self) -> u32 {
        self.patches.patch_count
    }

    pub fn source_batch_indices(&self) -> &[u32] {
        &self.source_batch_indices
    }

    pub fn prepared_records_buffer(&self) -> &wgpu::Buffer {
        &self.patches.prepared_records
    }
}

impl LodClassifierDevice {
    fn upload_adaptive_overlay_preparation_scene(
        &self,
        model: &LodClassifierModel,
        roots: &ResidentRootPreparationScene,
        words: WgslAdaptiveOverlayPreparationSceneWords,
    ) -> Result<PatchPreparationScene, LodWebGpuError> {
        let patch_count = words.uniform[0];
        let num_morph_targets = u32::try_from(model.prepared.model.num_morph_targets)
            .map_err(|_| LodWebGpuError::Payload("patch morph target count exceeds u32".into()))?;
        if roots.topology.model_identity != model.identity
            || roots.topology.face_count as usize != model.prepared.residency.num_faces
            || patch_count == 0
            || words.uniform[1] != model.prepared.residency.num_vertices
            || words.uniform[2] != 0
            || words.uniform[3] != num_morph_targets
            || words.topology.len() != patch_count as usize
            || words.subjects.is_empty()
        {
            return Err(LodWebGpuError::Payload(
                "adaptive overlay preparation shape is malformed".to_string(),
            ));
        }
        let topology = buffer_init_or_zero(
            &self.device,
            "adaptive overlay topology",
            bytemuck::cast_slice(&words.topology),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        self.allocate_patch_preparation_scene_with_source(
            model,
            words.uniform,
            topology,
            Arc::clone(&roots.patches.source_faces),
            &words.subjects,
        )
    }

    /// Extract and upload only adaptive replacement batches. `None` is the
    /// canonical representation of a scene with no suppressed roots.
    pub fn upload_adaptive_overlay_scene(
        &self,
        pipeline: &PatchRenderPipeline,
        model: &LodClassifierModel,
        roots: &ResidentRootPreparationScene,
        scene: &RenderSceneSnapshot,
    ) -> Result<Option<AdaptiveOverlayScene>, LodWebGpuError> {
        if roots.topology.model_identity != model.identity {
            return Err(LodWebGpuError::Payload(
                "adaptive overlay roots belong to a different resource epoch".to_string(),
            ));
        }
        let words = pack_wgsl_adaptive_overlay_scene_words(&model.prepared, scene)
            .map_err(LodWebGpuError::Payload)?;
        let source_batch_indices = words.source_batch_indices;
        if source_batch_indices.is_empty() {
            if !scene.suppressed_root_faces.is_empty() {
                return Err(LodWebGpuError::Payload(
                    "suppressed roots require a nonempty adaptive overlay".to_string(),
                ));
            }
            return Ok(None);
        }
        let patches =
            self.upload_adaptive_overlay_preparation_scene(model, roots, words.preparation)?;
        let visibility = self.upload_visibility_compaction_scene(words.visibility)?;
        let face_visibility = self.create_face_visibility_expansion_scene(
            model,
            &patches,
            &visibility,
            model.prepared.residency.num_faces,
        )?;
        let bindings = self.create_patch_render_bindings(pipeline, &patches, &visibility)?;
        let batches = source_batch_indices
            .iter()
            .map(|&index| scene.batches[index as usize].clone())
            .collect::<Vec<_>>();
        Ok(Some(AdaptiveOverlayScene {
            model_identity: model.identity,
            scene_revision: scene.revision,
            suppressed_root_faces: scene.suppressed_root_faces.clone(),
            source_batch_indices,
            batches,
            patches,
            visibility,
            face_visibility,
            bindings,
        }))
    }

    /// Publish the compact root mask only after a replacement overlay has
    /// been fully allocated. Callers perform this queue write and swap their
    /// retained overlay handle at one frame boundary; allocation failure never
    /// mutates the previously drawable baseline.
    pub fn publish_adaptive_overlay_suppression(
        &self,
        root_geometry: &ResidentGeometryBucketScene,
        overlay: Option<&AdaptiveOverlayScene>,
    ) -> Result<(), LodWebGpuError> {
        if overlay.is_some_and(|overlay| overlay.model_identity != root_geometry.model_identity) {
            return Err(LodWebGpuError::Payload(
                "adaptive overlay publication belongs to a different model epoch".to_string(),
            ));
        }
        let suppressed_faces =
            overlay.map_or(&[][..], |overlay| overlay.suppressed_root_faces.as_slice());
        self.write_resident_root_suppression(root_geometry, suppressed_faces)
    }

    pub fn write_adaptive_overlay_pose_state(
        &self,
        model: &LodClassifierModel,
        overlay: &AdaptiveOverlayScene,
        pose: LodPose<'_>,
        num_joints: u32,
    ) -> Result<(), LodWebGpuError> {
        if overlay.model_identity != model.identity {
            return Err(LodWebGpuError::Payload(
                "adaptive overlay belongs to a different WebGPU model".to_string(),
            ));
        }
        self.write_patch_pose(model, &overlay.patches, pose, num_joints)
    }

    fn write_adaptive_overlay_frames(
        &self,
        frame: &RenderFrame,
        overlay: &AdaptiveOverlayScene,
        use_qb: bool,
    ) -> Result<(), LodWebGpuError> {
        let mut words = overlay.bindings.frame_words.lock().map_err(|_| {
            LodWebGpuError::Payload("adaptive overlay frame lock was poisoned".to_string())
        })?;
        for (destination, batch) in words
            .chunks_exact_mut(PATCH_RENDER_FRAME_WORDS)
            .zip(&overlay.batches)
        {
            destination.copy_from_slice(
                &PatchRenderFrame::from_render_frame(frame, batch, use_qb).to_words()?,
            );
        }
        self.queue.write_buffer(
            &overlay.bindings.frames,
            0,
            bytemuck::cast_slice(words.as_slice()),
        );
        Ok(())
    }

    fn validate_adaptive_overlay_epoch(
        &self,
        frame: &RenderFrame,
        scene: &RenderSceneSnapshot,
        model: &LodClassifierModel,
        resident: &DeviceResidentLod<'_>,
        overlay: &AdaptiveOverlayScene,
        validate_frame: bool,
    ) -> Result<(), LodWebGpuError> {
        if validate_frame {
            frame.validate(scene).map_err(|error| {
                LodWebGpuError::Payload(format!("render frame contract: {error}"))
            })?;
        }
        if frame.style != RenderStyle::Normals {
            return Err(LodWebGpuError::Payload(
                "adaptive overlay renderer currently requires normals mode".to_string(),
            ));
        }
        if overlay.model_identity != model.identity
            || resident.model_identity != model.identity
            || resident.face_count as usize != model.prepared.residency.num_faces
            || overlay.scene_revision != scene.revision
            || overlay.suppressed_root_faces != scene.suppressed_root_faces
            || overlay.source_batch_indices.len() != overlay.batches.len()
            || overlay
                .source_batch_indices
                .iter()
                .zip(&overlay.batches)
                .any(|(&source, batch)| scene.batches.get(source as usize) != Some(batch))
        {
            return Err(LodWebGpuError::Payload(
                "adaptive overlay resources belong to a different scene epoch".to_string(),
            ));
        }
        Ok(())
    }

    /// Encode sparse leaf preparation, resident-face visibility expansion,
    /// compaction, and one indirect draw per adaptive batch. The target may
    /// use `Load` operations to compose over roots rendered earlier in the
    /// same command encoder.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_adaptive_overlay_normals<'resource>(
        &'resource self,
        encoder: &mut wgpu::CommandEncoder,
        frame: &RenderFrame,
        scene: &RenderSceneSnapshot,
        model: &LodClassifierModel,
        resident: &DeviceResidentLod<'_>,
        overlay: &'resource AdaptiveOverlayScene,
        pipeline: &'resource PatchRenderPipeline,
        atlas: &'resource PackedPatchAtlas,
        target: PatchRenderTarget<'resource>,
        pose: LodPose<'_>,
        num_joints: u32,
        use_qb: bool,
    ) -> Result<AdaptiveOverlayFrameEncoding, LodWebGpuError> {
        self.encode_adaptive_overlay_normals_impl(
            encoder, frame, scene, model, resident, overlay, pipeline, atlas, target, pose,
            num_joints, use_qb, true, true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_adaptive_overlay_normals_impl<'resource>(
        &'resource self,
        encoder: &mut wgpu::CommandEncoder,
        frame: &RenderFrame,
        scene: &RenderSceneSnapshot,
        model: &LodClassifierModel,
        resident: &DeviceResidentLod<'_>,
        overlay: &'resource AdaptiveOverlayScene,
        pipeline: &'resource PatchRenderPipeline,
        atlas: &'resource PackedPatchAtlas,
        target: PatchRenderTarget<'resource>,
        pose: LodPose<'_>,
        num_joints: u32,
        use_qb: bool,
        write_dynamic_pose: bool,
        validate_frame: bool,
    ) -> Result<AdaptiveOverlayFrameEncoding, LodWebGpuError> {
        self.validate_adaptive_overlay_epoch(
            frame,
            scene,
            model,
            resident,
            overlay,
            validate_frame,
        )?;
        if write_dynamic_pose {
            self.write_adaptive_overlay_pose_state(model, overlay, pose, num_joints)?;
        } else {
            self.write_patch_joint_count(&overlay.patches, num_joints);
        }
        self.write_adaptive_overlay_frames(frame, overlay, use_qb)?;

        for (batch_index, batch) in overlay.batches.iter().enumerate() {
            let draw = atlas.triangle_draw(batch.id.key.lod).ok_or_else(|| {
                LodWebGpuError::Payload(format!(
                    "packed WebGPU atlas is missing adaptive batch {batch_index} key {:?}",
                    batch.id.key.lod,
                ))
            })?;
            if draw.index_count != batch.triangle_index_count {
                return Err(LodWebGpuError::Payload(format!(
                    "adaptive atlas batch {batch_index} has {} indices; expected {}",
                    draw.index_count, batch.triangle_index_count,
                )));
            }
        }

        self.encode_patch_preparation(&overlay.patches, encoder);
        self.encode_resident_lod_visibility_expansion(
            &overlay.face_visibility,
            overlay.patches.patch_count,
            encoder,
        );
        self.encode_visibility_compaction(&overlay.visibility, encoder);

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
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("quilting adaptive overlay normals"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target.color_view,
                depth_slice: None,
                resolve_target: target.resolve_target,
                ops: wgpu::Operations {
                    load: color_load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        for (batch_index, batch) in overlay.batches.iter().enumerate() {
            let draw = atlas
                .triangle_draw(batch.id.key.lod)
                .expect("adaptive atlas was validated before encoding");
            let permutation_sign = if batch.id.key.parity_bucket == 0 {
                1
            } else {
                -1
            };
            let winding = if batch.transform.orientation_sign * permutation_sign < 0 {
                PatchWinding::Clockwise
            } else {
                PatchWinding::CounterClockwise
            };
            pipeline.draw_batch(
                &mut pass,
                &overlay.bindings,
                &overlay.visibility,
                draw,
                batch_index as u32,
                winding,
            )?;
        }
        drop(pass);
        Ok(AdaptiveOverlayFrameEncoding {
            indirect_draw_calls: overlay.visibility.batch_count,
            source_patch_count: overlay.patches.patch_count,
        })
    }

    /// Compose the device-generated retained baseline and sparse adaptive
    /// replacement layer without mapping, copying, or rebuilding root records.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_resident_adaptive_normals<'resource>(
        &'resource self,
        encoder: &mut wgpu::CommandEncoder,
        frame: &RenderFrame,
        scene: &RenderSceneSnapshot,
        model: &LodClassifierModel,
        resident: &DeviceResidentLod<'_>,
        roots: &'resource ResidentRootPreparationScene,
        root_geometry: &'resource ResidentGeometryBucketScene,
        root_pipeline: &'resource ResidentRootRenderPipeline,
        root_bindings: &'resource ResidentRootRenderBindings,
        overlay_pipeline: &'resource PatchRenderPipeline,
        overlay: Option<&'resource AdaptiveOverlayScene>,
        atlas: &'resource PackedPatchAtlas,
        target: PatchRenderTarget<'resource>,
        pose: LodPose<'_>,
        num_joints: u32,
        use_qb: bool,
    ) -> Result<ResidentAdaptiveFrameEncoding, LodWebGpuError> {
        frame
            .validate(scene)
            .map_err(|error| LodWebGpuError::Payload(format!("render frame contract: {error}")))?;
        if scene.suppressed_root_faces.is_empty() != overlay.is_none() {
            return Err(LodWebGpuError::Payload(
                "adaptive overlay presence does not match root suppression".to_string(),
            ));
        }
        let resident_suppression = root_geometry.suppressed_faces.lock().map_err(|_| {
            LodWebGpuError::Payload("resident root suppression lock was poisoned".to_string())
        })?;
        if resident_suppression.as_slice() != scene.suppressed_root_faces {
            return Err(LodWebGpuError::Payload(
                "resident root suppression was not published with the adaptive overlay".to_string(),
            ));
        }
        drop(resident_suppression);
        let PatchRenderTarget {
            color_view,
            resolve_target,
            depth_stencil_view,
            clear_color,
            clear_depth,
        } = target;
        let root_encoding = self.encode_resident_root_normals(
            encoder,
            frame,
            model,
            resident,
            roots,
            root_geometry,
            root_pipeline,
            root_bindings,
            atlas,
            PatchRenderTarget {
                color_view,
                resolve_target,
                depth_stencil_view,
                clear_color,
                clear_depth,
            },
            pose,
            num_joints,
            use_qb,
        )?;
        let overlay_encoding = overlay
            .map(|overlay| {
                self.encode_adaptive_overlay_normals_impl(
                    encoder,
                    frame,
                    scene,
                    model,
                    resident,
                    overlay,
                    overlay_pipeline,
                    atlas,
                    PatchRenderTarget {
                        color_view,
                        resolve_target,
                        depth_stencil_view,
                        clear_color: None,
                        clear_depth: None,
                    },
                    pose,
                    num_joints,
                    use_qb,
                    false,
                    false,
                )
            })
            .transpose()?;
        let logical_submission = frame
            .expected_submission_stats(scene)
            .map_err(|error| LodWebGpuError::Payload(format!("render frame contract: {error}")))?;
        Ok(ResidentAdaptiveFrameEncoding {
            logical_submission,
            roots: root_encoding,
            overlay: overlay_encoding,
        })
    }
}
