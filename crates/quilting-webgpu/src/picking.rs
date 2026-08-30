//! Retained one-pixel picking for prepared QB patches.
//!
//! The pass reuses the ordinary scene bind group and compacted indirect draw
//! stream. A clip-space remap sends one full-viewport pixel to two 1x1 color
//! attachments, avoiding a full-resolution ID framebuffer. Only the explicit
//! staged result crosses the device boundary.

use super::{
    gpu_buffer, render_buffer_layout_visible, AdaptiveOverlayScene, LodClassifierDevice,
    LodWebGpuError, PackedPatchAtlas, PatchPreparationScene, PatchRenderBindings,
    PatchRenderPipeline, PatchRenderScene, ResidentGeometryBucketScene,
    ResidentRootPreparationScene, ResidentRootRenderBindings, ResidentRootRenderPipeline,
    VisibilityCompactionScene,
};
use futures_channel::oneshot;
use quilting_core::render::{RenderSceneSnapshot, ResidentRootDrawDomains};
use wgpu::util::DeviceExt;

const PICK_UNIFORM_BYTES: u64 = 16;
const PICK_COPY_ROW_BYTES: u64 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as u64;
const PICK_SURFACE_OFFSET: u64 = PICK_COPY_ROW_BYTES;
const PICK_READBACK_BYTES: u64 = 2 * PICK_COPY_ROW_BYTES;
const INDEXED_INDIRECT_RECORD_BYTES: u64 = 20;

/// One query against the exact viewport used to populate the current retained
/// frame tables. The epoch is not interpreted by the renderer; it travels with
/// the asynchronous sample so application authority can reject stale joins.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PatchPickRequest {
    pub viewport: [u32; 2],
    pub pixel: [u32; 2],
    pub target_epoch: u32,
}

impl PatchPickRequest {
    pub fn new(
        viewport: [u32; 2],
        pixel: [u32; 2],
        target_epoch: u32,
    ) -> Result<Self, LodWebGpuError> {
        let request = Self {
            viewport,
            pixel,
            target_epoch,
        };
        request.validate()?;
        Ok(request)
    }

    fn validate(self) -> Result<(), LodWebGpuError> {
        if self.viewport[0] == 0 || self.viewport[1] == 0 {
            return Err(LodWebGpuError::Payload(
                "patch pick viewport dimensions must be nonzero".to_string(),
            ));
        }
        if self.pixel[0] >= self.viewport[0] || self.pixel[1] >= self.viewport[1] {
            return Err(LodWebGpuError::Payload(format!(
                "patch pick pixel {},{} lies outside viewport {}x{}",
                self.pixel[0], self.pixel[1], self.viewport[0], self.viewport[1],
            )));
        }
        Ok(())
    }

    fn uniform_words(self) -> [u32; 4] {
        [
            self.viewport[0] as f32,
            self.viewport[1] as f32,
            self.pixel[0] as f32,
            self.pixel[1] as f32,
        ]
        .map(f32::to_bits)
    }
}

/// Exact transient renderer sample before joining packed node identity through
/// `hyperscape::InteractionTargetTable`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PatchPickSample {
    pub target_epoch: u32,
    pub packed_node: u32,
    pub source_face: u32,
    pub source_barycentric: [f32; 3],
    pub source_position: [f32; 3],
    pub output_distance: f32,
}

impl PatchPickSample {
    fn validate(self) -> Result<Self, LodWebGpuError> {
        if self
            .source_barycentric
            .into_iter()
            .chain(self.source_position)
            .chain([self.output_distance])
            .any(|value| !value.is_finite())
        {
            return Err(LodWebGpuError::Mapping(
                "patch pick readback contains non-finite values".to_string(),
            ));
        }
        if self.output_distance < 0.0
            || self
                .source_barycentric
                .into_iter()
                .any(|coordinate| coordinate < -1.0e-4)
        {
            return Err(LodWebGpuError::Mapping(
                "patch pick readback lies outside the rendered surface".to_string(),
            ));
        }
        Ok(self)
    }
}

/// Graphics state compatible with one ordinary prepared-patch pipeline family.
pub struct PatchPickPipeline {
    pipeline: wgpu::RenderPipeline,
    viewport_bind_group_layout: wgpu::BindGroupLayout,
    viewport_uniform: wgpu::Buffer,
    viewport_bind_group: wgpu::BindGroup,
}

/// Source-root graphics state compatible with the resident render family and
/// the viewport packet retained by [`PatchPickPipeline`].
pub struct ResidentRootPickPipeline {
    pipeline: wgpu::RenderPipeline,
}

/// Two one-pixel payload attachments and their private depth buffer.
pub struct PatchPickTarget {
    identity_and_bary: wgpu::Texture,
    identity_and_bary_view: wgpu::TextureView,
    surface_and_distance: wgpu::Texture,
    surface_and_distance_view: wgpu::TextureView,
    _depth: wgpu::Texture,
    depth_view: wgpu::TextureView,
}

/// CPU-visible facts about one encoded query. Survivor counts deliberately
/// remain device-local and do not add another readback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PatchPickEncoding {
    pub indirect_draw_calls: u32,
    pub request: PatchPickRequest,
}

/// Explicit asynchronous readback staged into the same command encoder as the
/// pick pass. The caller must submit that encoder before awaiting `read`.
pub struct StagedPatchPickReadback {
    #[cfg(not(target_arch = "wasm32"))]
    device: wgpu::Device,
    buffer: wgpu::Buffer,
    encoding: PatchPickEncoding,
}

impl StagedPatchPickReadback {
    pub fn encoding(&self) -> PatchPickEncoding {
        self.encoding
    }

    pub async fn read(self) -> Result<Option<PatchPickSample>, LodWebGpuError> {
        let slice = self.buffer.slice(..PICK_READBACK_BYTES);
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
        let sample = decode_patch_pick_bytes(&view, self.encoding.request)?;
        drop(view);
        self.buffer.unmap();
        Ok(sample)
    }
}

impl LodClassifierDevice {
    /// Create a query pipeline that shares group zero with `compatible`.
    /// Bindings produced for that prepared-patch family can therefore be used
    /// directly by the pick pass.
    pub fn create_patch_pick_pipeline(
        &self,
        compatible: &PatchRenderPipeline,
    ) -> Result<PatchPickPipeline, LodWebGpuError> {
        let module = self.memoized_render_shader_module(
            "quilting prepared QB pick",
            quilting_shaders::sources::PATCH_PICK_DEVICE,
            quilting_shaders::PATCH_PICK_DEVICE_VERTEX_ENTRY_POINT,
            quilting_shaders::compile_patch_pick_device_wgsl,
        )?;
        let viewport_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("quilting patch pick viewport layout"),
                    entries: &[render_buffer_layout_visible(
                        0,
                        wgpu::BufferBindingType::Uniform,
                        PICK_UNIFORM_BYTES,
                        false,
                        wgpu::ShaderStages::VERTEX,
                    )],
                });
        let viewport_uniform = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("quilting patch pick viewport"),
                contents: bytemuck::cast_slice(&[0u32; 4]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let viewport_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("quilting patch pick viewport bindings"),
            layout: &viewport_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: viewport_uniform.as_entire_binding(),
            }],
        });
        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("quilting patch pick pipeline layout"),
                bind_group_layouts: &[Some(&compatible.bind_group_layout), Some(&viewport_layout)],
                immediate_size: 0,
            });
        let attributes = wgpu::vertex_attr_array![0 => Float32x3];
        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("quilting patch pick pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some(quilting_shaders::PATCH_PICK_DEVICE_VERTEX_ENTRY_POINT),
                    compilation_options: Default::default(),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: 12,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &attributes,
                    }],
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth24Plus,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(quilting_shaders::PATCH_PICK_DEVICE_FRAGMENT_ENTRY_POINT),
                    compilation_options: Default::default(),
                    targets: &[
                        Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rgba32Uint,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                        Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rgba32Float,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                    ],
                }),
                multiview_mask: None,
                cache: None,
            });
        Ok(PatchPickPipeline {
            pipeline,
            viewport_bind_group_layout: viewport_layout,
            viewport_uniform,
            viewport_bind_group,
        })
    }

    /// Create the source-root half of retained adaptive picking. Group zero
    /// matches `compatible`; group one is the same one-pixel viewport packet
    /// used by ordinary and adaptive prepared-patch queries.
    pub fn create_resident_root_pick_pipeline(
        &self,
        compatible: &ResidentRootRenderPipeline,
        viewport: &PatchPickPipeline,
    ) -> Result<ResidentRootPickPipeline, LodWebGpuError> {
        let module = self.memoized_render_shader_module(
            "quilting resident root QB pick",
            quilting_shaders::sources::RESIDENT_ROOT_PICK_DEVICE,
            quilting_shaders::RESIDENT_ROOT_PICK_DEVICE_VERTEX_ENTRY_POINT,
            quilting_shaders::compile_resident_root_pick_device_wgsl,
        )?;
        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("quilting resident root pick pipeline layout"),
                bind_group_layouts: &[
                    Some(&compatible.bind_group_layout),
                    Some(&viewport.viewport_bind_group_layout),
                ],
                immediate_size: 0,
            });
        let attributes = wgpu::vertex_attr_array![0 => Float32x3];
        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("quilting resident root pick pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some(
                        quilting_shaders::RESIDENT_ROOT_PICK_DEVICE_VERTEX_ENTRY_POINT,
                    ),
                    compilation_options: Default::default(),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: 12,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &attributes,
                    }],
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth24Plus,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(
                        quilting_shaders::RESIDENT_ROOT_PICK_DEVICE_FRAGMENT_ENTRY_POINT,
                    ),
                    compilation_options: Default::default(),
                    targets: &[
                        Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rgba32Uint,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                        Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rgba32Float,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                    ],
                }),
                multiview_mask: None,
                cache: None,
            });
        Ok(ResidentRootPickPipeline { pipeline })
    }

    /// Allocate the constant-size payload target once for a retained backend.
    pub fn create_patch_pick_target(&self) -> PatchPickTarget {
        let extent = wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        };
        let texture = |label, format, usage| {
            self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let color_usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC;
        let identity_and_bary = texture(
            "quilting patch pick identity and barycentric",
            wgpu::TextureFormat::Rgba32Uint,
            color_usage,
        );
        let surface_and_distance = texture(
            "quilting patch pick surface and distance",
            wgpu::TextureFormat::Rgba32Float,
            color_usage,
        );
        let depth = texture(
            "quilting patch pick depth",
            wgpu::TextureFormat::Depth24Plus,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        );
        PatchPickTarget {
            identity_and_bary_view: identity_and_bary
                .create_view(&wgpu::TextureViewDescriptor::default()),
            surface_and_distance_view: surface_and_distance
                .create_view(&wgpu::TextureViewDescriptor::default()),
            depth_view: depth.create_view(&wgpu::TextureViewDescriptor::default()),
            identity_and_bary,
            surface_and_distance,
            _depth: depth,
        }
    }

    /// Append a one-pixel query after the current frame's preparation,
    /// visibility compaction, and frame-table writes. The encoded texture
    /// copies are ordered in the same submission; no synchronous map occurs.
    pub fn encode_patch_render_scene_pick(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &PatchPickPipeline,
        scene: &PatchRenderScene,
        atlas: &PackedPatchAtlas,
        target: &PatchPickTarget,
        request: PatchPickRequest,
    ) -> Result<StagedPatchPickReadback, LodWebGpuError> {
        let snapshot = scene.scene.snapshot();
        self.encode_patch_pick(
            encoder,
            pipeline,
            snapshot,
            &scene.bindings,
            &scene.patches,
            &scene.visibility,
            atlas,
            target,
            request,
        )
    }

    /// Convenience submission for a query against already-completed retained
    /// frame state. The returned map remains asynchronous; only command
    /// encoding and queue submission happen here.
    pub fn stage_patch_render_scene_pick(
        &self,
        pipeline: &PatchPickPipeline,
        scene: &PatchRenderScene,
        atlas: &PackedPatchAtlas,
        target: &PatchPickTarget,
        request: PatchPickRequest,
    ) -> Result<StagedPatchPickReadback, LodWebGpuError> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("quilting staged one-pixel patch pick"),
            });
        let readback = self.encode_patch_render_scene_pick(
            &mut encoder,
            pipeline,
            scene,
            atlas,
            target,
            request,
        )?;
        self.queue.submit([encoder.finish()]);
        Ok(readback)
    }

    /// Query a completed resident-root frame and its optional sparse adaptive
    /// replacement layer. This intentionally does not rerun preparation,
    /// visibility, or compaction: it consumes the same indirect state that
    /// produced the visible frame, then reads back only one 32-byte packet.
    #[allow(clippy::too_many_arguments)]
    pub fn stage_resident_adaptive_pick(
        &self,
        root_pipeline: &ResidentRootPickPipeline,
        overlay_pipeline: &PatchPickPipeline,
        scene: &RenderSceneSnapshot,
        roots: &ResidentRootPreparationScene,
        geometry: &ResidentGeometryBucketScene,
        root_bindings: &ResidentRootRenderBindings,
        overlay: Option<&AdaptiveOverlayScene>,
        atlas: &PackedPatchAtlas,
        target: &PatchPickTarget,
        request: PatchPickRequest,
    ) -> Result<StagedPatchPickReadback, LodWebGpuError> {
        request.validate()?;
        scene
            .validate()
            .map_err(|error| LodWebGpuError::Payload(format!("patch pick scene: {error}")))?;
        if scene.suppressed_root_faces.is_empty() != overlay.is_none() {
            return Err(LodWebGpuError::Payload(
                "resident pick overlay presence does not match root suppression".to_string(),
            ));
        }
        let resident_suppression = geometry.suppressed_faces.lock().map_err(|_| {
            LodWebGpuError::Payload("resident pick suppression lock was poisoned".to_string())
        })?;
        if resident_suppression.as_slice() != scene.suppressed_root_faces {
            return Err(LodWebGpuError::Payload(
                "resident pick suppression does not match the retained scene".to_string(),
            ));
        }
        drop(resident_suppression);

        let expected_domains = ResidentRootDrawDomains::build(scene, geometry.face_count as usize)
            .map_err(|error| {
                LodWebGpuError::Payload(format!("resident pick draw domains: {error}"))
            })?;
        if roots.topology.model_identity != geometry.model_identity
            || root_bindings.model_identity != geometry.model_identity
            || roots.draw_domains.domain_identity != geometry.domain_identity
            || root_bindings.domain_identity != geometry.domain_identity
            || roots.topology.face_count != geometry.face_count
            || roots.patches.patch_count != geometry.face_count
            || roots.draw_domains.face_count != geometry.face_count
            || roots.draw_domains.domains != expected_domains.domains
            || root_bindings.domain_count() != roots.draw_domains.domain_count
            || root_bindings.bucket_count != geometry.bucket_count
            || geometry.atlas_count != atlas.entry_count() as u32
        {
            return Err(LodWebGpuError::Payload(
                "resident pick resources belong to different scene epochs".to_string(),
            ));
        }
        if let Some(overlay) = overlay {
            let expected_sources = overlay.batches.iter().try_fold(0u32, |total, batch| {
                total
                    .checked_add(u32::try_from(batch.members.len()).map_err(|_| {
                        LodWebGpuError::Payload(
                            "resident pick overlay source count exceeds u32".to_string(),
                        )
                    })?)
                    .ok_or_else(|| {
                        LodWebGpuError::Payload(
                            "resident pick overlay source count overflowed".to_string(),
                        )
                    })
            })?;
            if overlay.model_identity != geometry.model_identity
                || overlay.scene_revision != scene.revision
                || overlay.suppressed_root_faces != scene.suppressed_root_faces
                || overlay.source_batch_indices.len() != overlay.batches.len()
                || overlay
                    .source_batch_indices
                    .iter()
                    .zip(&overlay.batches)
                    .any(|(&source, batch)| scene.batches.get(source as usize) != Some(batch))
                || overlay.visibility.batch_count != overlay.batches.len() as u32
                || overlay.visibility.source_count != expected_sources
                || overlay.bindings.frame_count != overlay.batches.len() as u32
                || overlay.patches.patch_count != expected_sources
            {
                return Err(LodWebGpuError::Payload(
                    "resident pick overlay belongs to a different scene epoch".to_string(),
                ));
            }
        }

        self.queue.write_buffer(
            &overlay_pipeline.viewport_uniform,
            0,
            bytemuck::cast_slice(&request.uniform_words()),
        );
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("quilting staged resident adaptive one-pixel pick"),
            });
        let mut indirect_draw_calls = 0u32;
        {
            let attachments = [
                Some(wgpu::RenderPassColorAttachment {
                    view: &target.identity_and_bary_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &target.surface_and_distance_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ];
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("quilting one-pixel resident root pick"),
                color_attachments: &attachments,
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &target.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&root_pipeline.pipeline);
            pass.set_bind_group(1, &overlay_pipeline.viewport_bind_group, &[]);
            pass.set_vertex_buffer(0, atlas.barycentric_buffer.slice(..));
            pass.set_index_buffer(
                atlas.triangle_index_buffer.slice(..),
                wgpu::IndexFormat::Uint32,
            );
            for bucket in 0..geometry.bucket_count {
                pass.set_bind_group(
                    0,
                    &root_bindings.bind_group,
                    &[bucket * root_bindings.bucket_index_uniform_stride],
                );
                pass.draw_indexed_indirect(
                    &geometry.triangle_indirect_arguments,
                    u64::from(bucket) * INDEXED_INDIRECT_RECORD_BYTES,
                );
                indirect_draw_calls = indirect_draw_calls.saturating_add(1);
            }
        }

        if let Some(overlay) = overlay {
            let attachments = [
                Some(wgpu::RenderPassColorAttachment {
                    view: &target.identity_and_bary_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &target.surface_and_distance_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ];
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("quilting one-pixel adaptive overlay pick"),
                color_attachments: &attachments,
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &target.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&overlay_pipeline.pipeline);
            pass.set_bind_group(1, &overlay_pipeline.viewport_bind_group, &[]);
            for (batch_index, batch) in overlay.batches.iter().enumerate() {
                if !batch.enabled {
                    continue;
                }
                let batch_index = u32::try_from(batch_index).map_err(|_| {
                    LodWebGpuError::Payload("resident pick overlay batch exceeds u32".to_string())
                })?;
                let draw = atlas.triangle_draw(batch.id.key.lod).ok_or_else(|| {
                    LodWebGpuError::Payload(format!(
                        "resident pick atlas is missing overlay batch {batch_index} key {:?}",
                        batch.id.key.lod,
                    ))
                })?;
                if draw.index_count != batch.triangle_index_count {
                    return Err(LodWebGpuError::Payload(format!(
                        "resident pick overlay batch {batch_index} has {} indices; expected {}",
                        draw.index_count, batch.triangle_index_count,
                    )));
                }
                let dynamic_offset = batch_index
                    .checked_mul(overlay.visibility.batch_index_uniform_stride)
                    .ok_or_else(|| {
                        LodWebGpuError::Payload(
                            "resident pick overlay batch offset exceeds u32".to_string(),
                        )
                    })?;
                pass.set_bind_group(0, &overlay.bindings.bind_group, &[dynamic_offset]);
                pass.set_vertex_buffer(0, draw.barycentric_buffer.slice(..));
                let index_width = match draw.index_format {
                    wgpu::IndexFormat::Uint16 => 2,
                    wgpu::IndexFormat::Uint32 => 4,
                };
                pass.set_index_buffer(
                    draw.index_buffer
                        .slice(u64::from(draw.first_index) * index_width..),
                    draw.index_format,
                );
                pass.draw_indexed_indirect(
                    &overlay.visibility.triangle_indirect_arguments,
                    u64::from(batch_index) * INDEXED_INDIRECT_RECORD_BYTES,
                );
                indirect_draw_calls = indirect_draw_calls.saturating_add(1);
            }
        }

        let buffer = gpu_buffer(
            &self.device,
            "quilting resident adaptive pick readback",
            PICK_READBACK_BYTES,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        copy_pick_texel(&mut encoder, &target.identity_and_bary, &buffer, 0);
        copy_pick_texel(
            &mut encoder,
            &target.surface_and_distance,
            &buffer,
            PICK_SURFACE_OFFSET,
        );
        self.queue.submit([encoder.finish()]);
        Ok(StagedPatchPickReadback {
            #[cfg(not(target_arch = "wasm32"))]
            device: self.device.clone(),
            buffer,
            encoding: PatchPickEncoding {
                indirect_draw_calls,
                request,
            },
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_patch_pick(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &PatchPickPipeline,
        snapshot: &RenderSceneSnapshot,
        bindings: &PatchRenderBindings,
        patches: &PatchPreparationScene,
        visibility: &VisibilityCompactionScene,
        atlas: &PackedPatchAtlas,
        target: &PatchPickTarget,
        request: PatchPickRequest,
    ) -> Result<StagedPatchPickReadback, LodWebGpuError> {
        request.validate()?;
        let expected_batches = u32::try_from(snapshot.batches.len())
            .map_err(|_| LodWebGpuError::Payload("patch pick batch count exceeds u32".into()))?;
        let expected_sources = snapshot.batches.iter().try_fold(0u32, |total, batch| {
            total
                .checked_add(u32::try_from(batch.members.len()).map_err(|_| {
                    LodWebGpuError::Payload("patch pick source count exceeds u32".into())
                })?)
                .ok_or_else(|| LodWebGpuError::Payload("patch pick source count overflowed".into()))
        })?;
        if visibility.batch_count != expected_batches
            || visibility.source_count != expected_sources
            || bindings.frame_count != expected_batches
            || patches.patch_count != expected_sources
        {
            return Err(LodWebGpuError::Payload(
                "patch pick residency does not match the retained scene".to_string(),
            ));
        }
        self.queue.write_buffer(
            &pipeline.viewport_uniform,
            0,
            bytemuck::cast_slice(&request.uniform_words()),
        );

        let mut indirect_draw_calls = 0u32;
        {
            let attachments = [
                Some(wgpu::RenderPassColorAttachment {
                    view: &target.identity_and_bary_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &target.surface_and_distance_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ];
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("quilting one-pixel patch pick"),
                color_attachments: &attachments,
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &target.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&pipeline.pipeline);
            pass.set_bind_group(1, &pipeline.viewport_bind_group, &[]);
            for (batch_index, batch) in snapshot.batches.iter().enumerate() {
                if !batch.enabled {
                    continue;
                }
                let batch_index = u32::try_from(batch_index).map_err(|_| {
                    LodWebGpuError::Payload("patch pick batch index exceeds u32".into())
                })?;
                let draw = atlas.triangle_draw(batch.id.key.lod).ok_or_else(|| {
                    LodWebGpuError::Payload(format!(
                        "patch pick atlas is missing batch {batch_index} key {:?}",
                        batch.id.key.lod,
                    ))
                })?;
                if draw.index_count != batch.triangle_index_count {
                    return Err(LodWebGpuError::Payload(format!(
                        "patch pick atlas batch {batch_index} has {} indices; expected {}",
                        draw.index_count, batch.triangle_index_count,
                    )));
                }
                let dynamic_offset = batch_index
                    .checked_mul(visibility.batch_index_uniform_stride)
                    .ok_or_else(|| {
                        LodWebGpuError::Payload("patch pick batch offset exceeds u32".into())
                    })?;
                pass.set_bind_group(0, &bindings.bind_group, &[dynamic_offset]);
                pass.set_vertex_buffer(0, draw.barycentric_buffer.slice(..));
                let index_width = match draw.index_format {
                    wgpu::IndexFormat::Uint16 => 2,
                    wgpu::IndexFormat::Uint32 => 4,
                };
                pass.set_index_buffer(
                    draw.index_buffer
                        .slice(u64::from(draw.first_index) * index_width..),
                    draw.index_format,
                );
                pass.draw_indexed_indirect(
                    &visibility.triangle_indirect_arguments,
                    u64::from(batch_index) * INDEXED_INDIRECT_RECORD_BYTES,
                );
                indirect_draw_calls = indirect_draw_calls.saturating_add(1);
            }
        }

        let buffer = gpu_buffer(
            &self.device,
            "quilting patch pick readback",
            PICK_READBACK_BYTES,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        copy_pick_texel(encoder, &target.identity_and_bary, &buffer, 0);
        copy_pick_texel(
            encoder,
            &target.surface_and_distance,
            &buffer,
            PICK_SURFACE_OFFSET,
        );
        Ok(StagedPatchPickReadback {
            #[cfg(not(target_arch = "wasm32"))]
            device: self.device.clone(),
            buffer,
            encoding: PatchPickEncoding {
                indirect_draw_calls,
                request,
            },
        })
    }
}

fn copy_pick_texel(
    encoder: &mut wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    buffer: &wgpu::Buffer,
    offset: u64,
) {
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset,
                bytes_per_row: Some(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT),
                rows_per_image: Some(1),
            },
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
}

fn decode_patch_pick_bytes(
    bytes: &[u8],
    request: PatchPickRequest,
) -> Result<Option<PatchPickSample>, LodWebGpuError> {
    if bytes.len() < PICK_READBACK_BYTES as usize {
        return Err(LodWebGpuError::Mapping(format!(
            "patch pick readback contains {} bytes; expected {PICK_READBACK_BYTES}",
            bytes.len(),
        )));
    }
    let identity = bytemuck::cast_slice::<u8, u32>(&bytes[..16]);
    if identity[0] == 0 {
        if identity[1] != 0 {
            return Err(LodWebGpuError::Mapping(
                "patch pick no-hit sentinel contains a node".to_string(),
            ));
        }
        return Ok(None);
    }
    if identity[1] == 0 {
        return Err(LodWebGpuError::Mapping(
            "patch pick hit contains no semantic node".to_string(),
        ));
    }
    let x = f32::from_bits(identity[2]);
    let y = f32::from_bits(identity[3]);
    let z = 1.0 - x - y;
    let surface_offset = PICK_SURFACE_OFFSET as usize;
    let surface = bytemuck::cast_slice::<u8, f32>(&bytes[surface_offset..surface_offset + 16]);
    PatchPickSample {
        target_epoch: request.target_epoch,
        packed_node: identity[1] - 1,
        source_face: identity[0] - 1,
        source_barycentric: [x, y, z],
        source_position: [surface[0], surface[1], surface[2]],
        output_distance: surface[3],
    }
    .validate()
    .map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_rejects_empty_or_out_of_bounds_pixels() {
        assert!(PatchPickRequest::new([0, 1], [0, 0], 3).is_err());
        assert!(PatchPickRequest::new([8, 4], [8, 0], 3).is_err());
        assert!(PatchPickRequest::new([8, 4], [0, 4], 3).is_err());
        assert_eq!(
            PatchPickRequest::new([8, 4], [7, 3], 3)
                .unwrap()
                .uniform_words(),
            [8.0, 4.0, 7.0, 3.0].map(f32::to_bits),
        );
    }

    #[test]
    fn readback_packet_preserves_epoch_identity_surface_and_distance() {
        let request = PatchPickRequest::new([1600, 900], [812, 417], 29).unwrap();
        let mut bytes = vec![0u8; PICK_READBACK_BYTES as usize];
        let identity = [43, 8, 0.125f32.to_bits(), 0.25f32.to_bits()];
        bytes[..16].copy_from_slice(bytemuck::cast_slice(&identity));
        let surface = [1.5f32, -2.0, 0.75, 4.25];
        let offset = PICK_SURFACE_OFFSET as usize;
        bytes[offset..offset + 16].copy_from_slice(bytemuck::cast_slice(&surface));
        assert_eq!(
            decode_patch_pick_bytes(&bytes, request).unwrap(),
            Some(PatchPickSample {
                target_epoch: 29,
                packed_node: 7,
                source_face: 42,
                source_barycentric: [0.125, 0.25, 0.625],
                source_position: [1.5, -2.0, 0.75],
                output_distance: 4.25,
            }),
        );
    }

    #[test]
    fn cleared_readback_is_no_hit_and_malformed_packets_fail() {
        let request = PatchPickRequest::new([1, 1], [0, 0], 5).unwrap();
        let mut bytes = vec![0u8; PICK_READBACK_BYTES as usize];
        assert_eq!(decode_patch_pick_bytes(&bytes, request).unwrap(), None);
        bytes[4..8].copy_from_slice(&1u32.to_ne_bytes());
        assert!(decode_patch_pick_bytes(&bytes, request).is_err());
        bytes[..8].copy_from_slice(bytemuck::cast_slice(&[1u32, 0]));
        assert!(decode_patch_pick_bytes(&bytes, request).is_err());
    }
}
