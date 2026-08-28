#![cfg(not(target_arch = "wasm32"))]

use quilting_core::batch::{RenderBatchId, RenderBatchKey, RenderBatchLayer, RenderBatchMember};
use quilting_core::instance_layout::{self, InstanceWriter};
use quilting_core::quaternion::{Mobius, Quat};
use quilting_core::render::{
    FocusFieldPacket, PbrDrawClass, RenderBatchSnapshot, RenderEntityTransform, RenderFrame,
    RenderFrameOptions, RenderGeometry, RenderPoseIdentity, RenderSceneSnapshot, RenderStyle,
    RenderView,
};
use quilting_core::screen_partition::ScreenPatchLeafId;
use quilting_renderer::compute::{
    pack_lod_classification, pack_wgsl_patch_preparation_scene_words,
    pack_wgsl_visibility_compaction_scene_words, prepare_lod_atlas_lookup,
    prepare_lod_dispatch_state, prepare_lod_model, unpack_lod_classification_fields,
    wgsl_visibility_compaction_oracle_words, LodAtlasLookup, LodDispatchState, LodModelData,
    LodSubjectState, PackedLodClassification, PreparedLodModel, WgslLodDispatchMetrics,
};
use quilting_webgpu::{LodClassifierDevice, LodPose, PatchAtlasDraw, PatchRenderTarget};
use wgpu::util::DeviceExt;

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

fn translation_matrix(x: f32, y: f32, z: f32) -> [f32; 16] {
    let mut matrix = identity_matrix();
    matrix[12] = x;
    matrix[13] = y;
    matrix[14] = z;
    matrix
}

fn identity_dispatch() -> LodDispatchState {
    LodDispatchState {
        subjects: Vec::new(),
        baseline_mobius: identity_mobius(),
        baseline_model: identity_matrix(),
        pole: [0.0; 4],
        mobius_power: 0.0,
        c_norm_sq: 0.0,
        has_pole: 0.0,
    }
}

fn compaction_member(face_index: u32, leaf_id: ScreenPatchLeafId) -> RenderBatchMember {
    RenderBatchMember {
        face_index,
        leaf_id,
        node_index: 0,
        edge_lods: [2; 3],
        permutation_index: 0,
        vertex_lods: [2; 3],
    }
}

fn compaction_batch(
    material_index: usize,
    faces: impl IntoIterator<Item = u32>,
    enabled: bool,
) -> RenderBatchSnapshot {
    RenderBatchSnapshot {
        id: RenderBatchId::complete(RenderBatchKey {
            lod: [2; 3],
            parity_bucket: 0,
            material_index,
            render_node_index: 0,
        }),
        members: faces
            .into_iter()
            .map(|face| compaction_member(face, ScreenPatchLeafId::ROOT))
            .collect(),
        triangle_index_count: 6 * (material_index as u32 + 1),
        line_index_count: 8 * (material_index as u32 + 1),
        transform: RenderEntityTransform {
            mobius: identity_mobius(),
            orientation_sign: 1,
            euclidean_model: identity_matrix(),
            euclidean_normal: identity_matrix(),
        },
        enabled,
        pbr_class: PbrDrawClass::Opaque,
    }
}

fn complete_atlas() -> LodAtlasLookup {
    let mut keys = Vec::with_capacity(220);
    for a in 0..=9 {
        for b in a..=9 {
            for c in b..=9 {
                keys.push([1u32 << a, 1u32 << b, 1u32 << c]);
            }
        }
    }
    prepare_lod_atlas_lookup(keys).unwrap()
}

fn shared_render_frame_fixture() -> (PreparedLodModel, Vec<f32>, RenderSceneSnapshot, RenderFrame) {
    let positions = vec![
        -0.8, -0.6, 0.0, -0.2, -0.6, 0.0, -0.5, 0.2, 0.0, 0.2, -0.6, 0.0, 0.8, -0.6, 0.0, 0.5, 0.2,
        0.0,
    ];
    let faces = vec![[0, 1, 2], [3, 4, 5]];
    let prepared = prepare_lod_model(LodModelData {
        positions: positions.clone(),
        faces: faces.clone(),
        joint_indices: vec![[0; 4]; 6],
        joint_weights: vec![[0.0; 4]; 6],
        morph_deltas: Vec::new(),
        num_morph_targets: 0,
        face_nodes: vec![0, 1],
    })
    .unwrap();
    let mut source_instances = vec![0.0; 2 * instance_layout::STRIDE];
    for (face_index, vertices) in faces.into_iter().enumerate() {
        let mut writer = InstanceWriter::new(&mut source_instances, face_index);
        for (corner, vertex) in vertices.into_iter().enumerate() {
            writer.set_position(
                corner,
                vertex,
                [
                    positions[vertex as usize * 3],
                    positions[vertex as usize * 3 + 1],
                    positions[vertex as usize * 3 + 2],
                ],
            );
            writer.set_normal(corner, [0.0, 0.0, 1.0]);
        }
        writer.set_uvs([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]);
        writer.set_node_id(face_index as u32);
    }

    let mut batches = Vec::new();
    for (material_index, face_index) in [0u32, 1].into_iter().enumerate() {
        let mut mobius = identity_mobius();
        if material_index == 1 {
            mobius = packed_mobius(Mobius::translation(Quat::from_point(0.1, 0.0, 0.0)));
        }
        batches.push(RenderBatchSnapshot {
            id: RenderBatchId::complete(RenderBatchKey {
                lod: [1; 3],
                parity_bucket: 0,
                material_index,
                render_node_index: material_index,
            }),
            members: vec![RenderBatchMember {
                face_index,
                leaf_id: ScreenPatchLeafId::ROOT,
                node_index: material_index,
                edge_lods: [1; 3],
                permutation_index: 0,
                vertex_lods: [1; 3],
            }],
            triangle_index_count: 3,
            line_index_count: 6,
            transform: RenderEntityTransform {
                mobius,
                orientation_sign: if material_index == 1 { -1 } else { 1 },
                euclidean_model: identity_matrix(),
                euclidean_normal: identity_matrix(),
            },
            enabled: true,
            pbr_class: PbrDrawClass::Opaque,
        });
    }
    let scene = RenderSceneSnapshot {
        revision: 91,
        suppressed_root_faces: Vec::new(),
        batches,
    };
    let frame = RenderFrame::build(
        7,
        RenderPoseIdentity {
            asset_revision: 4,
            pose_revision: 12,
        },
        RenderStyle::Normals,
        RenderView {
            viewport: [32, 32],
            mvp: identity_matrix(),
            model_view: identity_matrix(),
            camera_position: [0.0, 0.0, 3.0],
            selected_node: None,
            focus: FocusFieldPacket::default(),
        },
        RenderFrameOptions::default(),
        &scene,
    )
    .unwrap();
    (prepared, source_instances, scene, frame)
}

fn metrics(atlas: &LodAtlasLookup, pixel_floor: f32, num_joints: u32) -> WgslLodDispatchMetrics {
    WgslLodDispatchMetrics {
        view_projection: identity_matrix(),
        density: 1.0,
        pixel_floor,
        max_lod: atlas.max_lod,
        viewport: [1024.0, 1024.0],
        num_joints,
    }
}

fn packed_mobius(mobius: Mobius) -> [f32; 16] {
    let mut packed = [0.0; 16];
    for (chunk, quaternion) in packed
        .chunks_exact_mut(4)
        .zip([mobius.a, mobius.b, mobius.c, mobius.d])
    {
        chunk.copy_from_slice(&[
            quaternion.w as f32,
            quaternion.x as f32,
            quaternion.y as f32,
            quaternion.z as f32,
        ]);
    }
    packed
}

async fn classify_one(
    classifier: &LodClassifierDevice,
    prepared: PreparedLodModel,
    atlas: &LodAtlasLookup,
    dispatch: &LodDispatchState,
    metrics: WgslLodDispatchMetrics,
    pose: LodPose<'_>,
) -> PackedLodClassification {
    let mut resident = classifier.upload_model(prepared, atlas).unwrap();
    let packed = classifier
        .classify(&mut resident, dispatch, metrics, pose)
        .await
        .unwrap();
    assert_eq!(packed.len(), 1);
    unpack_lod_classification_fields(packed[0]).unwrap()
}

#[test]
fn native_classifier_matches_cpu_oracles_and_pass_one_invariants() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = match instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
        {
            Ok(adapter) => adapter,
            Err(error) if std::env::var_os("QUILTING_REQUIRE_WEBGPU").is_none() => {
                eprintln!("skipping native WebGPU conformance: {error}");
                return;
            }
            Err(error) => panic!("required native WebGPU adapter is unavailable: {error}"),
        };
        eprintln!("native WebGPU adapter: {:?}", adapter.get_info());
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("quilting native LOD conformance"),
                ..Default::default()
            })
            .await
            .expect("request native WebGPU device");
        let classifier = LodClassifierDevice::new(device, queue).unwrap();
        let report = classifier.run_conformance_matrix().await.unwrap();
        assert_eq!(report.full_pipeline_words, 1);
        assert_eq!(report.coherence_words, 10);
        assert_eq!(report.prepared_patch_words, 104);
        assert!(report.rendered_patch_pixels >= 8);
        assert_eq!(report.shared_frame_draws, 2);
        assert_eq!(report.compacted_source_words, 89);
        assert_eq!(report.compacted_range_words, 15);
        assert_eq!(report.indirect_argument_words, 15);
        assert_eq!(report.indirect_draws, 3);

        let (prepared, source_instances, render_scene, render_frame) =
            shared_render_frame_fixture();
        let preparation_words =
            pack_wgsl_patch_preparation_scene_words(&prepared, &render_scene, &source_instances)
                .unwrap();
        let model = classifier
            .upload_model(prepared, &complete_atlas())
            .unwrap();
        let patches = classifier
            .upload_patch_preparation_scene(&model, preparation_words)
            .unwrap();
        classifier
            .write_patch_pose(&model, &patches, LodPose::default(), 0)
            .unwrap();
        let visibility_words =
            pack_wgsl_visibility_compaction_scene_words(&render_scene, RenderGeometry::Triangles)
                .unwrap();
        let visibility = classifier
            .upload_visibility_compaction_scene(visibility_words)
            .unwrap();
        classifier
            .write_source_visibility(&visibility, &[1, 1])
            .unwrap();
        let pipeline = classifier
            .create_patch_render_pipeline(wgpu::TextureFormat::Rgba8Unorm, None, 1)
            .unwrap();
        let bindings = classifier
            .create_patch_render_bindings(&pipeline, &patches, &visibility)
            .unwrap();
        let barycentric_buffer =
            classifier
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("shared frame barycentrics"),
                    contents: bytemuck::cast_slice(&[
                        1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
                    ]),
                    usage: wgpu::BufferUsages::VERTEX,
                });
        let index_buffer =
            classifier
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("shared frame indices"),
                    contents: bytemuck::cast_slice(&[u32::MAX, u32::MAX, u32::MAX, 0, 1, 2]),
                    usage: wgpu::BufferUsages::INDEX,
                });
        let atlas = || PatchAtlasDraw {
            barycentric_buffer: &barycentric_buffer,
            index_buffer: &index_buffer,
            index_format: wgpu::IndexFormat::Uint32,
            first_index: 3,
            index_count: 3,
        };
        let atlases = [atlas(), atlas()];
        let target = classifier
            .device()
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("shared frame target"),
                size: wgpu::Extent3d {
                    width: 32,
                    height: 32,
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
        let error_scope = classifier
            .device()
            .push_error_scope(wgpu::ErrorFilter::Validation);
        let mut encoder =
            classifier
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("shared RenderFrame conformance"),
                });
        let encoding = classifier
            .encode_normals_render_frame(
                &mut encoder,
                &render_frame,
                &render_scene,
                &pipeline,
                &bindings,
                &patches,
                &visibility,
                &atlases,
                PatchRenderTarget {
                    color_view: &target_view,
                    resolve_target: None,
                    depth_stencil_view: None,
                    clear_color: Some(wgpu::Color::TRANSPARENT),
                    clear_depth: None,
                },
                true,
                |_, _, _| Ok(()),
            )
            .unwrap();
        assert_eq!(
            encoding.logical_submission,
            render_frame
                .expected_submission_stats(&render_scene)
                .unwrap(),
        );
        assert_eq!(encoding.indirect_draw_calls, 2);
        assert_eq!(encoding.source_instance_count, 2);
        classifier.queue().submit([encoder.finish()]);
        classifier
            .device()
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        if let Some(error) = error_scope.pop().await {
            panic!("shared RenderFrame validation: {error}");
        }

        let compaction_scene = RenderSceneSnapshot {
            revision: 19,
            suppressed_root_faces: Vec::new(),
            batches: vec![
                compaction_batch(0, 0..130, true),
                compaction_batch(1, 130..133, false),
                compaction_batch(2, 133..137, true),
            ],
        };
        let mut source_visibility = (0..130)
            .map(|index| u8::from(index % 3 != 0))
            .collect::<Vec<_>>();
        source_visibility.extend([1, 1, 1, 1, 0, 1, 1]);
        let expected = wgsl_visibility_compaction_oracle_words(
            &compaction_scene,
            &source_visibility,
            RenderGeometry::Triangles,
        )
        .unwrap();
        let words = pack_wgsl_visibility_compaction_scene_words(
            &compaction_scene,
            RenderGeometry::Triangles,
        )
        .unwrap();
        let mut resident = classifier
            .upload_visibility_compaction_scene(words)
            .unwrap();
        let actual = classifier
            .compact_visibility(&mut resident, &source_visibility)
            .await
            .unwrap();
        assert_eq!(
            actual.compacted_source_instances,
            expected.compacted_source_instances,
        );
        assert_eq!(actual.compacted_ranges, expected.compacted_ranges);
        assert_eq!(actual.indirect_arguments, expected.indirect_arguments);

        let mut roots = compaction_batch(0, [0, 1], true);
        roots.id.layer = RenderBatchLayer::RetainedRoot;
        let mut overlay = compaction_batch(0, [0], true);
        overlay.id.layer = RenderBatchLayer::AdaptiveOverlay;
        overlay.members[0].leaf_id = ScreenPatchLeafId::ROOT.child(0).unwrap();
        let replacement_scene = RenderSceneSnapshot {
            revision: 20,
            suppressed_root_faces: vec![0],
            batches: vec![roots, overlay],
        };
        let expected = wgsl_visibility_compaction_oracle_words(
            &replacement_scene,
            &[1, 1, 1],
            RenderGeometry::Lines,
        )
        .unwrap();
        let words =
            pack_wgsl_visibility_compaction_scene_words(&replacement_scene, RenderGeometry::Lines)
                .unwrap();
        let mut resident = classifier
            .upload_visibility_compaction_scene(words)
            .unwrap();
        let actual = classifier
            .compact_visibility(&mut resident, &[1, 1, 1])
            .await
            .unwrap();
        assert_eq!(
            actual.compacted_source_instances,
            expected.compacted_source_instances,
        );
        assert_eq!(actual.compacted_ranges, expected.compacted_ranges);
        assert_eq!(actual.indirect_arguments, expected.indirect_arguments);

        let atlas = complete_atlas();
        let simple_triangle = || {
            prepare_lod_model(LodModelData {
                positions: vec![-0.5, -0.5, 0.0, 0.5, -0.5, 0.0, 0.0, 0.5, 0.0],
                faces: vec![[0, 1, 2]],
                joint_indices: vec![[0; 4]; 3],
                joint_weights: vec![[0.0; 4]; 3],
                morph_deltas: Vec::new(),
                num_morph_targets: 0,
                face_nodes: vec![97],
            })
            .unwrap()
        };
        let visible = classify_one(
            &classifier,
            simple_triangle(),
            &atlas,
            &identity_dispatch(),
            metrics(&atlas, 1.0, 0),
            LodPose::default(),
        )
        .await;
        assert!(visible.visible());

        let mut translated_dispatch = identity_dispatch();
        translated_dispatch.baseline_model = translation_matrix(100.0, 0.0, 0.0);
        let culled = classify_one(
            &classifier,
            simple_triangle(),
            &atlas,
            &translated_dispatch,
            metrics(&atlas, 1.0, 0),
            LodPose::default(),
        )
        .await;
        assert_eq!(
            culled,
            unpack_lod_classification_fields(
                pack_lod_classification([1, 1, 1], 0, None, 0).unwrap()
            )
            .unwrap(),
        );

        let mut subject_dispatch = identity_dispatch();
        subject_dispatch.subjects.push(LodSubjectState {
            node: 97,
            mobius: identity_mobius(),
            model: translation_matrix(100.0, 0.0, 0.0),
            pole: [0.0; 4],
            mobius_power: 0.0,
            c_norm_sq: 0.0,
            has_pole: 0.0,
        });
        let subject_culled = classify_one(
            &classifier,
            simple_triangle(),
            &atlas,
            &subject_dispatch,
            metrics(&atlas, 1.0, 0),
            LodPose::default(),
        )
        .await;
        assert_eq!(subject_culled, culled);

        let morphed_triangle = || {
            prepare_lod_model(LodModelData {
                positions: vec![99.5, -0.5, 0.0, 100.5, -0.5, 0.0, 100.0, 0.5, 0.0],
                faces: vec![[0, 1, 2]],
                joint_indices: vec![[0; 4]; 3],
                joint_weights: vec![[0.0; 4]; 3],
                morph_deltas: vec![-100.0, 0.0, 0.0, -100.0, 0.0, 0.0, -100.0, 0.0, 0.0],
                num_morph_targets: 1,
                face_nodes: vec![0],
            })
            .unwrap()
        };
        let morph_offscreen = classify_one(
            &classifier,
            morphed_triangle(),
            &atlas,
            &identity_dispatch(),
            metrics(&atlas, 1.0, 0),
            LodPose {
                joint_matrices: &[],
                morph_weights: &[0.0],
            },
        )
        .await;
        let morph_onscreen = classify_one(
            &classifier,
            morphed_triangle(),
            &atlas,
            &identity_dispatch(),
            metrics(&atlas, 1.0, 0),
            LodPose {
                joint_matrices: &[],
                morph_weights: &[1.0],
            },
        )
        .await;
        assert!(!morph_offscreen.visible());
        assert!(morph_onscreen.visible());

        let skinned_triangle = || {
            prepare_lod_model(LodModelData {
                positions: vec![99.5, -0.5, 0.0, 100.5, -0.5, 0.0, 100.0, 0.5, 0.0],
                faces: vec![[0, 1, 2]],
                joint_indices: vec![[0; 4]; 3],
                joint_weights: vec![[1.0, 0.0, 0.0, 0.0]; 3],
                morph_deltas: Vec::new(),
                num_morph_targets: 0,
                face_nodes: vec![0],
            })
            .unwrap()
        };
        let identity_joint = identity_matrix();
        let translated_joint = translation_matrix(-100.0, 0.0, 0.0);
        let skin_offscreen = classify_one(
            &classifier,
            skinned_triangle(),
            &atlas,
            &identity_dispatch(),
            metrics(&atlas, 1.0, 1),
            LodPose {
                joint_matrices: &identity_joint,
                morph_weights: &[],
            },
        )
        .await;
        let skin_onscreen = classify_one(
            &classifier,
            skinned_triangle(),
            &atlas,
            &identity_dispatch(),
            metrics(&atlas, 1.0, 1),
            LodPose {
                joint_matrices: &translated_joint,
                morph_weights: &[],
            },
        )
        .await;
        assert!(!skin_offscreen.visible());
        assert!(skin_onscreen.visible());

        let pole_triangle = prepare_lod_model(LodModelData {
            positions: vec![-1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0, 0.0],
            faces: vec![[0, 1, 2]],
            joint_indices: vec![[0; 4]; 3],
            joint_weights: vec![[0.0; 4]; 3],
            morph_deltas: Vec::new(),
            num_morph_targets: 0,
            face_nodes: vec![0],
        })
        .unwrap();
        let reflection = Mobius::sphere_reflection(Quat::from_point(0.0, 0.0, 0.0), 1.0);
        let pole_dispatch =
            prepare_lod_dispatch_state(&[], &pole_triangle.residency, 1, packed_mobius(reflection));
        let pole_result = classify_one(
            &classifier,
            pole_triangle,
            &atlas,
            &pole_dispatch,
            metrics(&atlas, 0.0, 0),
            LodPose::default(),
        )
        .await;
        assert!(pole_result.visible());
        assert_eq!(pole_result.canonical, [512, 512, 512]);
    });
}
