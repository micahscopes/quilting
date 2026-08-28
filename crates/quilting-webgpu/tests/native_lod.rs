#![cfg(not(target_arch = "wasm32"))]

use quilting_core::batch::{
    FaceLodGrading, RenderBatchId, RenderBatchKey, RenderBatchLayer, RenderBatchMember,
};
use quilting_core::instance_layout::{self, InstanceWriter};
use quilting_core::quaternion::{Mobius, Quat};
use quilting_core::render::{
    FocusFieldPacket, PbrDrawClass, RenderBatchSnapshot, RenderEntityTransform, RenderFrame,
    RenderFrameOptions, RenderGeometry, RenderPoseIdentity, RenderSceneSnapshot, RenderStyle,
    RenderView,
};
use quilting_core::screen_partition::ScreenPatchLeafId;
use quilting_renderer::compute::{
    pack_lod_classification, pack_wgsl_visibility_compaction_scene_words, prepare_lod_atlas_lookup,
    prepare_lod_dispatch_state, prepare_lod_model, unpack_lod_classification_fields,
    wgsl_visibility_compaction_oracle_words, LodAtlasLookup, LodDispatchState, LodModelData,
    LodSubjectState, PackedLodClassification, PreparedLodModel, WgslLodDispatchMetrics,
};
use quilting_webgpu::{LodClassifierDevice, LodPose, PatchRenderSceneUpdate};

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
        assert_eq!(report.resident_lod_words, 20);
        assert_eq!(report.resident_visibility_words, 4);
        assert_eq!(report.resident_bucket_words, 323);
        assert_eq!(report.resident_root_topology_words, 240);
        assert_eq!(report.resident_root_prepared_words, 1040);
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
        let foreign_prepared = prepared.clone();
        let atlas = complete_atlas();
        let mut model = classifier.upload_model(prepared, &atlas).unwrap();
        let mut foreign_model = classifier.upload_model(foreign_prepared, &atlas).unwrap();
        let pipeline = classifier.create_offscreen_patch_render_pipeline().unwrap();
        let mut retained_scene = classifier
            .upload_patch_render_scene(
                &pipeline,
                &model,
                render_scene.clone(),
                &source_instances,
                render_scene.revision,
            )
            .unwrap();
        assert_eq!(retained_scene.patch_count(), 2);
        assert_eq!(retained_scene.batch_count(), 2);
        let mut reordered_scene = render_scene.clone();
        let (first, second) = reordered_scene.batches.split_at_mut(1);
        std::mem::swap(&mut first[0].members, &mut second[0].members);
        reordered_scene.revision += 1;
        let reordered_retained_scene = classifier
            .upload_patch_render_scene(
                &pipeline,
                &model,
                reordered_scene.clone(),
                &source_instances,
                reordered_scene.revision,
            )
            .unwrap();
        let mut visibility_dispatch = identity_dispatch();
        visibility_dispatch.subjects.push(LodSubjectState {
            node: 1,
            mobius: identity_mobius(),
            model: translation_matrix(100.0, 0.0, 0.0),
            pole: [0.0; 4],
            mobius_power: 0.0,
            c_norm_sq: 0.0,
            has_pole: 0.0,
        });
        {
            let classification = classifier
                .classify_on_device(
                    &mut model,
                    &visibility_dispatch,
                    metrics(&atlas, 1.0, 0),
                    LodPose::default(),
                )
                .unwrap();
            let resident_lod = classifier
                .reconcile_resident_lod_on_device(&classification, FaceLodGrading::TwoToOne);
            assert_eq!(
                classifier
                    .expand_resident_lod_visibility_for_diagnostics(&retained_scene, &resident_lod,)
                    .await
                    .unwrap(),
                [1, 0],
            );
            assert_eq!(
                classifier
                    .expand_resident_lod_visibility_for_diagnostics(
                        &reordered_retained_scene,
                        &resident_lod,
                    )
                    .await
                    .unwrap(),
                [0, 1],
            );
            let foreign_classification = classifier
                .classify_on_device(
                    &mut foreign_model,
                    &visibility_dispatch,
                    metrics(&atlas, 1.0, 0),
                    LodPose::default(),
                )
                .unwrap();
            let foreign_resident = classifier.reconcile_resident_lod_on_device(
                &foreign_classification,
                FaceLodGrading::TwoToOne,
            );
            let mut encoder =
                classifier
                    .device()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("foreign resident LOD visibility rejection"),
                    });
            assert!(classifier
                .encode_patch_render_resident_lod_visibility(
                    &retained_scene,
                    &foreign_resident,
                    &mut encoder,
                )
                .unwrap_err()
                .to_string()
                .contains("different WebGPU model"));
        }
        assert!(classifier
            .write_patch_render_pose_state(&foreign_model, &retained_scene, LodPose::default(), 0,)
            .unwrap_err()
            .to_string()
            .contains("different WebGPU model"));
        assert!(matches!(
            classifier
                .update_patch_render_scene_in_place(
                    &model,
                    &mut retained_scene,
                    reordered_scene.clone(),
                    &source_instances,
                    reordered_scene.revision,
                )
                .unwrap(),
            PatchRenderSceneUpdate::Updated,
        ));
        assert_eq!(retained_scene.scene(), &reordered_scene);
        assert_eq!(
            classifier
                .expand_face_visibility_for_diagnostics(&retained_scene, &[0b01])
                .await
                .unwrap(),
            [0, 1],
        );

        let mut resized_scene = reordered_scene.clone();
        let mut empty_batch = resized_scene.batches.last().unwrap().clone();
        empty_batch.id.key.material_index += 1;
        empty_batch.id.key.render_node_index += 1;
        empty_batch.members.clear();
        resized_scene.batches.push(empty_batch);
        resized_scene.revision += 1;
        let returned_scene = match classifier
            .update_patch_render_scene_in_place(
                &model,
                &mut retained_scene,
                resized_scene.clone(),
                &source_instances,
                resized_scene.revision,
            )
            .unwrap()
        {
            PatchRenderSceneUpdate::ShapeChanged(scene) => scene,
            PatchRenderSceneUpdate::Updated => panic!("batch-count change updated in place"),
        };
        assert_eq!(returned_scene, resized_scene);
        assert_eq!(retained_scene.scene(), &reordered_scene);

        assert!(matches!(
            classifier
                .update_patch_render_scene_in_place(
                    &model,
                    &mut retained_scene,
                    render_scene.clone(),
                    &source_instances,
                    render_scene.revision,
                )
                .unwrap(),
            PatchRenderSceneUpdate::Updated,
        ));
        assert_eq!(
            classifier
                .expand_face_visibility_for_diagnostics(&retained_scene, &[0b01])
                .await
                .unwrap(),
            [1, 0],
        );
        classifier
            .write_patch_render_scene_state(&model, &retained_scene, LodPose::default(), 0, &[1, 1])
            .unwrap();
        let packed_atlas = classifier
            .upload_packed_patch_atlas(
                &[1, 1, 1, 3, 3, 0, 0],
                &[1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                &[0, 0, 0, 0, 1, 2],
                &[],
            )
            .unwrap();
        assert_eq!(packed_atlas.entry_count(), 1);
        assert_eq!(packed_atlas.vertex_count(), 3);
        assert_eq!(packed_atlas.triangle_index_count(), 6);
        assert_eq!(packed_atlas.line_index_count(), 0);
        assert!(
            classifier
                .upload_packed_patch_atlas(
                    &[1, 1, 1, 0, 3, 0, 0],
                    &[1.0f32, 0.0, 0.0],
                    &[0, 1, 0],
                    &[],
                )
                .err()
                .expect("out-of-range packed atlas must fail")
                .to_string()
                .contains("out-of-range global vertex index")
        );
        assert!(classifier
            .upload_packed_patch_atlas(&[1, 1, 1, 5, 3, 0, 0], &[1.0f32, 0.0, 0.0], &[0], &[],)
            .err()
            .expect("out-of-range packed atlas metadata must fail")
            .to_string()
            .contains("range exceeds its index buffer"));
        let target = classifier
            .create_offscreen_patch_render_target([32, 32])
            .unwrap();
        let error_scope = classifier
            .device()
            .push_error_scope(wgpu::ErrorFilter::Validation);
        let encoding = classifier
            .render_offscreen_normals_patch_scene(
                &render_frame,
                &pipeline,
                &retained_scene,
                &packed_atlas,
                &target,
                true,
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
        classifier
            .device()
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        if let Some(error) = error_scope.pop().await {
            panic!("shared RenderFrame validation: {error}");
        }

        classifier
            .write_patch_render_pose_state(&model, &retained_scene, LodPose::default(), 0)
            .unwrap();
        classifier
            .write_patch_render_face_visibility_bits(&retained_scene, &[0b11])
            .unwrap();
        assert!(classifier
            .write_patch_render_face_visibility_bits(&retained_scene, &[u32::MAX])
            .unwrap_err()
            .to_string()
            .contains("nonzero padding"));
        let error_scope = classifier
            .device()
            .push_error_scope(wgpu::ErrorFilter::Validation);
        let encoding = classifier
            .render_offscreen_normals_patch_scene_with_face_visibility(
                &render_frame,
                &pipeline,
                &retained_scene,
                &packed_atlas,
                &target,
                true,
            )
            .unwrap();
        assert_eq!(encoding.indirect_draw_calls, 2);
        assert_eq!(encoding.source_instance_count, 2);
        classifier
            .device()
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        if let Some(error) = error_scope.pop().await {
            panic!("face visibility expansion validation: {error}");
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
