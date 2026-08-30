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
    bindings: PatchRenderBindings,
    _patch_frame_rows: wgpu::Buffer,
    prepared_visibility_bind_group: wgpu::BindGroup,
    pbr_scene_supported: bool,
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

    pub fn supports_resident_untextured_pbr(&self) -> bool {
        self.pbr_scene_supported
            && self
                .bindings
                .material_textures
                .as_ref()
                .is_some_and(|textures| {
                    textures
                        .residency()
                        .iter()
                        .all(|material| material.referenced_mask() == 0)
                })
            && self
                .bindings
                .pbr_environment
                .as_ref()
                .is_some_and(PbrEnvironmentBindings::is_resident)
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
        self.upload_adaptive_overlay_scene_with_resources(pipeline, model, roots, scene, None, None)
    }

    pub fn upload_adaptive_overlay_scene_with_pbr_resources(
        &self,
        pipeline: &PatchRenderPipeline,
        model: &LodClassifierModel,
        roots: &ResidentRootPreparationScene,
        scene: &RenderSceneSnapshot,
        textures: Option<&PbrTextureTable>,
        environment: Option<&PbrEnvironmentMap>,
    ) -> Result<Option<AdaptiveOverlayScene>, LodWebGpuError> {
        if pipeline.style != RenderStyle::Pbr {
            return Err(LodWebGpuError::Payload(
                "adaptive PBR resources require the PBR render pipeline".to_string(),
            ));
        }
        self.upload_adaptive_overlay_scene_with_resources(
            pipeline,
            model,
            roots,
            scene,
            textures,
            environment,
        )
    }

    fn upload_adaptive_overlay_scene_with_resources(
        &self,
        pipeline: &PatchRenderPipeline,
        model: &LodClassifierModel,
        roots: &ResidentRootPreparationScene,
        scene: &RenderSceneSnapshot,
        textures: Option<&PbrTextureTable>,
        environment: Option<&PbrEnvironmentMap>,
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
        let bindings = self.create_patch_render_bindings_with_environment(
            pipeline,
            scene,
            &patches,
            &visibility,
            textures,
            environment,
        )?;
        let batches = source_batch_indices
            .iter()
            .map(|&index| scene.batches[index as usize].clone())
            .collect::<Vec<_>>();
        let mut patch_frame_rows = Vec::with_capacity(patches.patch_count as usize);
        for (frame_row, batch) in batches.iter().enumerate() {
            let frame_row = u32::try_from(frame_row)
                .map_err(|_| LodWebGpuError::Payload("adaptive frame row exceeds u32".into()))?;
            patch_frame_rows.extend(std::iter::repeat_n(frame_row, batch.members.len()));
        }
        if patch_frame_rows.len() != patches.patch_count as usize {
            return Err(LodWebGpuError::Payload(
                "adaptive frame rows do not cover the prepared patch domain".to_string(),
            ));
        }
        let patch_frame_rows = buffer_init_or_zero(
            &self.device,
            "adaptive patch frame rows",
            bytemuck::cast_slice(&patch_frame_rows),
            wgpu::BufferUsages::STORAGE,
        );
        let prepared_visibility_layout = self.prepared_visibility_pipeline.get_bind_group_layout(0);
        let prepared_visibility_bind_group =
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("adaptive prepared visibility bindings"),
                layout: &prepared_visibility_layout,
                entries: &[
                    bind(0, &patches.uniform),
                    bind(1, &bindings.frames),
                    bind(2, &patches.prepared_records),
                    bind(3, &patch_frame_rows),
                    bind(4, &visibility.source_visibility),
                ],
            });
        Ok(Some(AdaptiveOverlayScene {
            model_identity: model.identity,
            scene_revision: scene.revision,
            suppressed_root_faces: scene.suppressed_root_faces.clone(),
            source_batch_indices,
            batches,
            patches,
            visibility,
            bindings,
            _patch_frame_rows: patch_frame_rows,
            prepared_visibility_bind_group,
            pbr_scene_supported: pipeline.style == RenderStyle::Pbr
                && supports_basic_pbr_frame(scene, RenderFrameOptions::default()),
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

    fn encode_adaptive_overlay_visibility(
        &self,
        overlay: &AdaptiveOverlayScene,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        if overlay.patches.patch_count == 0 {
            return;
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("quilting adaptive prepared visibility"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.prepared_visibility_pipeline);
        pass.set_bind_group(0, &overlay.prepared_visibility_bind_group, &[]);
        pass.dispatch_workgroups(
            overlay.patches.patch_count.div_ceil(LOD_WORKGROUP_SIZE),
            1,
            1,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn validate_adaptive_overlay_epoch(
        &self,
        frame: &RenderFrame,
        scene: &RenderSceneSnapshot,
        model: &LodClassifierModel,
        resident: &DeviceResidentLod<'_>,
        overlay: &AdaptiveOverlayScene,
        pipelines: &DiagnosticPatchRenderPipelines,
        validate_frame: bool,
    ) -> Result<(), LodWebGpuError> {
        if validate_frame {
            frame.validate(scene).map_err(|error| {
                LodWebGpuError::Payload(format!("render frame contract: {error}"))
            })?;
        }
        let pbr_supported = frame.style == RenderStyle::Pbr
            && overlay.supports_resident_untextured_pbr()
            && validate_basic_pbr_frame(scene, frame.options).is_ok();
        if !(supports_patch_presentation_style(frame.style) || pbr_supported)
            || render_draw_passes(frame.style)
                .iter()
                .filter(|draw| draw.pass != RenderPass::PbrTransparent)
                .any(|draw| pipelines.get_for_pass(draw.pass, draw.geometry).is_err())
        {
            return Err(LodWebGpuError::Payload(format!(
                "adaptive overlay renderer does not support {:?}",
                frame.style,
            )));
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
    /// compaction, and one indirect draw per adaptive batch and style pass.
    /// The target may use `Load` operations to compose over roots rendered
    /// earlier in the same command encoder.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_adaptive_overlay<'resource>(
        &'resource self,
        encoder: &mut wgpu::CommandEncoder,
        frame: &RenderFrame,
        scene: &RenderSceneSnapshot,
        model: &LodClassifierModel,
        resident: &DeviceResidentLod<'_>,
        overlay: &'resource AdaptiveOverlayScene,
        pipelines: &'resource DiagnosticPatchRenderPipelines,
        atlas: &'resource PackedPatchAtlas,
        target: PatchRenderTarget<'resource>,
        pose: LodPose<'_>,
        num_joints: u32,
        use_qb: bool,
    ) -> Result<AdaptiveOverlayFrameEncoding, LodWebGpuError> {
        self.encode_adaptive_overlay_impl(
            encoder, frame, scene, model, resident, overlay, pipelines, atlas, target, pose,
            num_joints, use_qb, true, true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_adaptive_overlay_impl<'resource>(
        &'resource self,
        encoder: &mut wgpu::CommandEncoder,
        frame: &RenderFrame,
        scene: &RenderSceneSnapshot,
        model: &LodClassifierModel,
        resident: &DeviceResidentLod<'_>,
        overlay: &'resource AdaptiveOverlayScene,
        pipelines: &'resource DiagnosticPatchRenderPipelines,
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
            pipelines,
            validate_frame,
        )?;
        if write_dynamic_pose {
            self.write_adaptive_overlay_pose_state(model, overlay, pose, num_joints)?;
        } else {
            self.write_patch_joint_count(&overlay.patches, num_joints);
        }
        self.write_adaptive_overlay_frames(frame, overlay, use_qb)?;

        let draw_passes = render_draw_passes(frame.style);
        for draw_pass in draw_passes {
            if draw_pass.pass == RenderPass::PbrTransparent {
                continue;
            }
            for (batch_index, batch) in overlay.batches.iter().enumerate() {
                let draw = atlas
                    .draw(batch.id.key.lod, draw_pass.geometry)
                    .ok_or_else(|| {
                        LodWebGpuError::Payload(format!(
                            "packed WebGPU atlas is missing adaptive batch {batch_index} key {:?} for {:?}",
                            batch.id.key.lod, draw_pass.geometry,
                        ))
                    })?;
                let expected = match draw_pass.geometry {
                    RenderGeometry::Triangles => batch.triangle_index_count,
                    RenderGeometry::Lines => batch.line_index_count,
                };
                if draw.index_count != expected {
                    return Err(LodWebGpuError::Payload(format!(
                        "adaptive atlas batch {batch_index} has {} {:?} indices; expected {expected}",
                        draw.index_count, draw_pass.geometry,
                    )));
                }
            }
        }

        self.encode_patch_preparation(&overlay.patches, encoder);
        self.encode_adaptive_overlay_visibility(overlay, encoder);
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
            label: Some("quilting adaptive overlay frame"),
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
        let mut indirect_draw_calls = 0u32;
        for draw_pass in draw_passes {
            if draw_pass.pass == RenderPass::PbrTransparent {
                continue;
            }
            let pipeline = pipelines.get_for_pass(draw_pass.pass, draw_pass.geometry)?;
            for (batch_index, batch) in overlay.batches.iter().enumerate() {
                let draw = atlas
                    .draw(batch.id.key.lod, draw_pass.geometry)
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
                    0,
                    winding,
                )?;
                indirect_draw_calls = indirect_draw_calls.saturating_add(1);
            }
        }
        drop(pass);
        Ok(AdaptiveOverlayFrameEncoding {
            indirect_draw_calls,
            source_patch_count: overlay.patches.patch_count,
        })
    }

    /// Compose the device-generated retained baseline and sparse adaptive
    /// replacement layer without mapping, copying, or rebuilding root records.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_resident_adaptive<'resource>(
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
        overlay_pipelines: &'resource DiagnosticPatchRenderPipelines,
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
        if frame.style == RenderStyle::Pbr {
            validate_basic_pbr_frame(scene, frame.options)?;
        }
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
        let root_encoding = self.encode_resident_roots(
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
                self.encode_adaptive_overlay_impl(
                    encoder,
                    frame,
                    scene,
                    model,
                    resident,
                    overlay,
                    overlay_pipelines,
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

    /// Submit one retained-root plus sparse adaptive frame to the offscreen
    /// target without mapping either the classifier or compaction outputs.
    #[allow(clippy::too_many_arguments)]
    pub fn render_offscreen_resident_adaptive(
        &self,
        frame: &RenderFrame,
        scene: &RenderSceneSnapshot,
        model: &LodClassifierModel,
        resident: &DeviceResidentLod<'_>,
        roots: &ResidentRootPreparationScene,
        root_geometry: &ResidentGeometryBucketScene,
        root_pipeline: &ResidentRootRenderPipeline,
        root_bindings: &ResidentRootRenderBindings,
        overlay_pipelines: &DiagnosticPatchRenderPipelines,
        overlay: Option<&AdaptiveOverlayScene>,
        atlas: &PackedPatchAtlas,
        target: &OffscreenPatchRenderTarget,
        pose: LodPose<'_>,
        num_joints: u32,
        use_qb: bool,
    ) -> Result<PatchFrameEncoding, LodWebGpuError> {
        if frame.view.viewport != target.size {
            return Err(LodWebGpuError::Payload(format!(
                "offscreen target {:?} does not match frame viewport {:?}",
                target.size, frame.view.viewport,
            )));
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("quilting resident adaptive offscreen frame"),
            });
        let encoding = self.encode_resident_adaptive(
            &mut encoder,
            frame,
            scene,
            model,
            resident,
            roots,
            root_geometry,
            root_pipeline,
            root_bindings,
            overlay_pipelines,
            overlay,
            atlas,
            PatchRenderTarget {
                color_view: &target.color_view,
                resolve_target: None,
                depth_stencil_view: Some(&target.depth_view),
                clear_color: Some(wgpu::Color {
                    r: 0.2,
                    g: 0.2,
                    b: 0.3,
                    a: 0.0,
                }),
                clear_depth: Some(1.0),
            },
            pose,
            num_joints,
            use_qb,
        )?;
        self.queue.submit([encoder.finish()]);
        Ok(resident_adaptive_frame_evidence(encoding))
    }

    /// Present one retained-root plus sparse adaptive frame through the same
    /// device-only encoder used by the offscreen path.
    #[allow(clippy::too_many_arguments)]
    pub fn present_resident_adaptive(
        &self,
        surface: &mut PatchPresentationSurface,
        frame: &RenderFrame,
        scene: &RenderSceneSnapshot,
        model: &LodClassifierModel,
        resident: &DeviceResidentLod<'_>,
        roots: &ResidentRootPreparationScene,
        root_geometry: &ResidentGeometryBucketScene,
        root_pipeline: &ResidentRootRenderPipeline,
        root_bindings: &ResidentRootRenderBindings,
        overlay_pipelines: &DiagnosticPatchRenderPipelines,
        overlay: Option<&AdaptiveOverlayScene>,
        atlas: &PackedPatchAtlas,
        pose: LodPose<'_>,
        num_joints: u32,
        use_qb: bool,
    ) -> Result<SurfacePresentation<PatchFrameEncoding>, LodWebGpuError> {
        surface.present_with(
            self,
            "quilting resident adaptive presentation frame",
            |encoder, mut target| {
                target.clear_color = Some(wgpu::Color {
                    r: 0.2,
                    g: 0.2,
                    b: 0.3,
                    a: 1.0,
                });
                target.clear_depth = Some(1.0);
                self.encode_resident_adaptive(
                    encoder,
                    frame,
                    scene,
                    model,
                    resident,
                    roots,
                    root_geometry,
                    root_pipeline,
                    root_bindings,
                    overlay_pipelines,
                    overlay,
                    atlas,
                    target,
                    pose,
                    num_joints,
                    use_qb,
                )
                .map(resident_adaptive_frame_evidence)
            },
        )
    }
}

fn resident_adaptive_frame_evidence(encoding: ResidentAdaptiveFrameEncoding) -> PatchFrameEncoding {
    let overlay_draws = encoding
        .overlay
        .map_or(0, |overlay| overlay.indirect_draw_calls);
    let overlay_patches = encoding
        .overlay
        .map_or(0, |overlay| overlay.source_patch_count);
    PatchFrameEncoding {
        logical_submission: encoding.logical_submission,
        indirect_draw_calls: encoding
            .roots
            .indirect_draw_calls
            .saturating_add(overlay_draws),
        source_instance_count: encoding
            .roots
            .source_face_count
            .saturating_add(overlay_patches),
    }
}
