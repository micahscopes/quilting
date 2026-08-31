#![cfg(not(target_arch = "wasm32"))]

use quilting_core::batch::{
    FaceLodGrading, RenderBatchId, RenderBatchKey, RenderBatchLayer, RenderBatchMember,
};
use quilting_core::instance_layout::{self, InstanceWriter};
use quilting_core::material::{
    EnvironmentMapAsset, EnvironmentMapDescriptor, PbrMaterial, PbrTextureReferences,
    Rgba8TextureAsset, TextureAssetDescriptor, TextureWrapMode,
};
use quilting_core::quaternion::{Mobius, Quat};
use quilting_core::render::{
    FocusFieldPacket, MatcapStyle, PbrDrawClass, RenderBatchSnapshot, RenderEntityTransform,
    RenderFrame, RenderFrameOptions, RenderGeometry, RenderPoseIdentity, RenderSceneSnapshot,
    RenderStyle, RenderView, ValidatedRenderScene,
};
use quilting_core::render_evidence::{render_image_signature, RenderImageSignature};
use quilting_core::screen_partition::ScreenPatchLeafId;
use quilting_renderer::compute::{
    pack_lod_classification, pack_wgsl_visibility_compaction_scene_words, prepare_lod_atlas_lookup,
    prepare_lod_dispatch_state, prepare_lod_model, unpack_lod_classification_fields,
    wgsl_visibility_compaction_oracle_words, LodAtlasLookup, LodDispatchState, LodModelData,
    LodSubjectState, PackedLodClassification, PreparedLodModel, WgslLodDispatchMetrics,
};
use quilting_webgpu::{
    resident_root_render_domains, supports_basic_pbr_frame, supports_focus_pbr_frame,
    supports_patch_presentation_style, supports_resident_root_render_scene,
    supports_resident_root_render_style, LodClassifierDevice, LodPose, OffscreenPatchRenderTarget,
    PatchRenderSceneUpdate, PbrTextureTableUpdate, PoseUploadPolicy,
};

#[test]
fn live_patch_style_capability_has_one_authoritative_predicate() {
    for style in [
        RenderStyle::Matcap,
        RenderStyle::Wire,
        RenderStyle::Normals,
        RenderStyle::MatcapWire,
        RenderStyle::Lod,
        RenderStyle::Stretch,
    ] {
        assert!(supports_patch_presentation_style(style), "{style:?}");
    }
    assert!(!supports_patch_presentation_style(RenderStyle::Pbr));
    for style in [
        RenderStyle::Matcap,
        RenderStyle::Wire,
        RenderStyle::Normals,
        RenderStyle::MatcapWire,
        RenderStyle::Lod,
        RenderStyle::Stretch,
    ] {
        assert!(supports_resident_root_render_style(style), "{style:?}");
    }
    assert!(!supports_resident_root_render_style(RenderStyle::Pbr));
}

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
    let mut matte = PbrMaterial::default();
    matte.base_color = [0.75, 0.08, 0.04, 1.0];
    matte.roughness = 0.85;
    let mut metal = PbrMaterial::default();
    metal.base_color = [0.04, 0.18, 0.8, 1.0];
    metal.metallic = 0.9;
    metal.roughness = 0.18;
    let scene = RenderSceneSnapshot {
        revision: 91,
        materials: vec![matte, metal],
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

#[test]
fn basic_and_focus_pbr_capabilities_are_explicitly_disjoint() {
    let (_, _, scene, _) = shared_render_frame_fixture();
    let focus_options = RenderFrameOptions {
        focus_postprocess: Some(quilting_core::render::FocusPostprocessPacket {
            mode: quilting_core::render::FocusPostprocessMode::Spheroidal,
            blur_radius_pixels: 11,
            blur_strength: 1.0,
            focus_coordinate: 0.5,
            bandwidth: 0.1,
            normalize_range: false,
            stretch_range: [0.5, 0.5],
            gaussian_passes: 1,
            kawase_passes: 3,
            kawase_offset: 1.5,
        }),
        ..RenderFrameOptions::default()
    };
    assert!(supports_basic_pbr_frame(
        &scene,
        RenderFrameOptions::default(),
    ));
    assert!(!supports_focus_pbr_frame(
        &scene,
        RenderFrameOptions::default(),
    ));
    assert!(!supports_basic_pbr_frame(&scene, focus_options));
    assert!(supports_focus_pbr_frame(&scene, focus_options));
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

async fn offscreen_signature(
    classifier: &LodClassifierDevice,
    target: &OffscreenPatchRenderTarget,
) -> RenderImageSignature {
    render_image_signature(
        classifier
            .stage_offscreen_patch_render_target_image(target)
            .unwrap()
            .read()
            .await
            .unwrap()
            .view()
            .unwrap(),
        0,
    )
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
        let mut focus_resources = classifier
            .create_offscreen_focus_pbr_render_resources()
            .expect("create retained focus render resources");
        assert!(focus_resources
            .ensure_target(&classifier, [32, 32])
            .expect("create retained focus target"));
        assert!(!focus_resources
            .ensure_target(&classifier, [32, 32])
            .expect("reuse retained focus target"));
        let focus_pipelines = focus_resources.postprocess_pipelines();
        assert_eq!(
            focus_pipelines.output_format(),
            wgpu::TextureFormat::Rgba8Unorm,
        );
        let focus_pbr_pipeline = focus_resources.overlay_pipeline();
        assert_eq!(
            focus_pbr_pipeline.color_format(),
            wgpu::TextureFormat::Rgba8Unorm,
        );
        assert_eq!(
            focus_pbr_pipeline.raw_field_format(),
            wgpu::TextureFormat::Rgba16Float,
        );
        let focus_target = focus_resources.target().expect("retained focus target");
        let initial_shader_memo = classifier.render_shader_memo_diagnostics();
        assert_eq!(initial_shader_memo.misses, 3);
        assert_eq!(initial_shader_memo.hits, 0);
        assert_eq!(initial_shader_memo.failed_creations, 0);
        assert_eq!(initial_shader_memo.resident_entries, 3);
        let initial_focus_pipeline_memo = classifier.focus_postprocess_pipeline_memo_diagnostics();
        assert_eq!(initial_focus_pipeline_memo.misses, 1);
        assert_eq!(initial_focus_pipeline_memo.hits, 0);
        assert_eq!(initial_focus_pipeline_memo.failed_creations, 0);
        assert_eq!(initial_focus_pipeline_memo.resident_entries, 1);
        let initial_prepared_pipeline_memo = classifier.prepared_patch_pipeline_memo_diagnostics();
        assert_eq!(initial_prepared_pipeline_memo.misses, 1);
        assert_eq!(initial_prepared_pipeline_memo.hits, 0);
        assert_eq!(initial_prepared_pipeline_memo.failed_creations, 0);
        assert_eq!(initial_prepared_pipeline_memo.resident_entries, 1);

        let _reused_focus_overlay = classifier
            .create_offscreen_focus_pbr_patch_render_pipeline()
            .expect("reuse retained functional prepared-patch focus family");
        let reused_prepared_pipeline_memo = classifier.prepared_patch_pipeline_memo_diagnostics();
        assert_eq!(reused_prepared_pipeline_memo.misses, 1);
        assert_eq!(reused_prepared_pipeline_memo.hits, 1);
        assert_eq!(reused_prepared_pipeline_memo.failed_creations, 0);
        assert_eq!(reused_prepared_pipeline_memo.resident_entries, 1);
        assert_eq!(
            classifier.render_shader_memo_diagnostics(),
            initial_shader_memo,
            "a prepared-patch family hit must not revisit shader lowering",
        );

        let _reused_focus_pipelines = classifier
            .create_offscreen_focus_postprocess_pipelines()
            .expect("reuse retained functional focus pipeline family");
        let reused_focus_pipeline_memo = classifier.focus_postprocess_pipeline_memo_diagnostics();
        assert_eq!(reused_focus_pipeline_memo.misses, 1);
        assert_eq!(reused_focus_pipeline_memo.hits, 1);
        assert_eq!(reused_focus_pipeline_memo.failed_creations, 0);
        assert_eq!(reused_focus_pipeline_memo.resident_entries, 1);
        assert_eq!(
            classifier.render_shader_memo_diagnostics(),
            initial_shader_memo,
            "a functional pipeline-family hit must not revisit shader lowering",
        );
        let initial_root_pipeline_memo = classifier.resident_root_pipeline_memo_diagnostics();
        assert_eq!(initial_root_pipeline_memo.misses, 1);
        assert_eq!(initial_root_pipeline_memo.hits, 0);
        assert_eq!(initial_root_pipeline_memo.failed_creations, 0);
        assert_eq!(initial_root_pipeline_memo.resident_entries, 1);

        let _reused_root_pipeline = classifier
            .create_offscreen_resident_root_render_pipeline()
            .expect("reuse retained resident-root pipeline family");
        let reused_root_pipeline_memo = classifier.resident_root_pipeline_memo_diagnostics();
        assert_eq!(reused_root_pipeline_memo.misses, 1);
        assert_eq!(reused_root_pipeline_memo.hits, 1);
        assert_eq!(reused_root_pipeline_memo.failed_creations, 0);
        assert_eq!(reused_root_pipeline_memo.resident_entries, 1);
        assert_eq!(
            classifier.render_shader_memo_diagnostics(),
            initial_shader_memo,
            "a pipeline-family hit must not revisit shader lowering",
        );

        let texture_a = TextureAssetDescriptor {
            width: 2,
            height: 2,
            wrap_s: TextureWrapMode::Repeat,
            wrap_t: TextureWrapMode::ClampToEdge,
        };
        let texture_b = TextureAssetDescriptor {
            width: 1,
            height: 2,
            wrap_s: TextureWrapMode::MirroredRepeat,
            wrap_t: TextureWrapMode::Repeat,
        };
        let initial_a = [
            255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
        ];
        let initial_b = [13, 17, 19, 255, 23, 29, 31, 255];
        let initial_assets = [
            Rgba8TextureAsset::new(texture_a, &initial_a).unwrap(),
            Rgba8TextureAsset::new(texture_b, &initial_b).unwrap(),
        ];
        let error_scope = classifier
            .device()
            .push_error_scope(wgpu::ErrorFilter::Validation);
        let mut texture_table = classifier
            .upload_pbr_texture_table(&initial_assets)
            .unwrap();
        assert_eq!(
            texture_table.descriptors(),
            &[Some(texture_a), Some(texture_b)]
        );
        assert_eq!(texture_table.occupied_len(), 2);
        assert_eq!(texture_table.portable_atlas_plan().extent, [4, 4]);
        assert_eq!(texture_table.portable_atlas_plan().layer_count, 1);
        assert_eq!(
            texture_table.portable_atlas_plan().placements[1]
                .unwrap()
                .origin,
            [2, 0],
        );
        assert!(texture_table.linear_view(0).is_some());
        assert!(texture_table.srgb_view(0).is_some());
        assert!(texture_table.sampler(1).is_some());
        assert_eq!(
            classifier
                .read_pbr_texture_rgba8_for_diagnostics(&texture_table, 0)
                .await
                .unwrap(),
            initial_a,
        );
        assert_eq!(
            classifier
                .read_pbr_portable_atlas_rgba8_for_diagnostics(&texture_table, 0)
                .await
                .unwrap(),
            initial_a,
        );

        let sparse_assets = [Some(initial_assets[0]), None, Some(initial_assets[1])];
        let sparse_table = classifier
            .upload_pbr_texture_slot_table(&sparse_assets)
            .unwrap();
        assert_eq!(sparse_table.len(), 3);
        assert_eq!(sparse_table.occupied_len(), 2);
        assert_eq!(
            sparse_table.descriptors(),
            &[Some(texture_a), None, Some(texture_b)]
        );
        assert!(sparse_table.linear_view(1).is_none());
        assert_eq!(sparse_table.portable_atlas_plan().placements[1], None);
        assert_eq!(
            classifier
                .read_pbr_texture_rgba8_for_diagnostics(&sparse_table, 2)
                .await
                .unwrap(),
            initial_b,
        );
        assert_eq!(
            classifier
                .read_pbr_portable_atlas_rgba8_for_diagnostics(&sparse_table, 2)
                .await
                .unwrap(),
            initial_b,
        );

        let pbr_pipeline = classifier
            .create_diagnostic_patch_render_pipeline(
                RenderStyle::Pbr,
                wgpu::TextureFormat::Rgba8Unorm,
                Some(wgpu::TextureFormat::Depth24Plus),
                1,
            )
            .unwrap();
        let reused_shader_memo = classifier.render_shader_memo_diagnostics();
        assert_eq!(reused_shader_memo.misses, 3);
        assert_eq!(reused_shader_memo.hits, 1);
        assert_eq!(reused_shader_memo.resident_entries, 3);
        let mut first_material = PbrMaterial::default();
        first_material.textures = PbrTextureReferences {
            base_color: Some(0),
            normal: Some(1),
            transmission: Some(2),
            ..Default::default()
        };
        let mut second_material = PbrMaterial::default();
        second_material.textures.emissive = Some(99);
        let texture_bindings = classifier
            .create_pbr_material_texture_bindings(
                &pbr_pipeline,
                &[first_material, second_material],
                Some(&sparse_table),
            )
            .unwrap();
        assert_eq!(texture_bindings.material_count(), 2);
        assert_eq!(texture_bindings.residency().len(), 2);
        assert_eq!(texture_bindings.residency()[0].referenced_mask(), 0b10_0101);
        assert_eq!(texture_bindings.residency()[0].resident_mask(), 0b10_0001);
        assert_eq!(texture_bindings.residency()[0].unresolved_mask(), 0b000100);
        assert_eq!(texture_bindings.residency()[1].referenced_mask(), 0b001000);
        assert_eq!(texture_bindings.residency()[1].resident_mask(), 0);
        assert_eq!(texture_bindings.residency()[1].unresolved_mask(), 0b001000);
        let default_texture_bindings = classifier
            .create_pbr_material_texture_bindings(&pbr_pipeline, &[], None)
            .unwrap();
        assert_eq!(default_texture_bindings.material_count(), 1);
        assert_eq!(default_texture_bindings.residency(), &[Default::default()]);

        let updated_a = [7; 16];
        let updated_b = [11; 8];
        let updated_assets = [
            Rgba8TextureAsset::new(texture_a, &updated_a).unwrap(),
            Rgba8TextureAsset::new(texture_b, &updated_b).unwrap(),
        ];
        assert_eq!(
            classifier
                .update_pbr_texture_table_in_place(&mut texture_table, &updated_assets)
                .unwrap(),
            PbrTextureTableUpdate::Updated,
        );
        assert_eq!(
            classifier
                .read_pbr_texture_rgba8_for_diagnostics(&texture_table, 0)
                .await
                .unwrap(),
            updated_a,
        );
        assert_eq!(
            classifier
                .read_pbr_portable_atlas_rgba8_for_diagnostics(&texture_table, 0)
                .await
                .unwrap(),
            updated_a,
        );

        let rejected_a = [37; 16];
        let malformed_b = [41; 7];
        let malformed_assets = [
            Rgba8TextureAsset::new(texture_a, &rejected_a).unwrap(),
            Rgba8TextureAsset {
                descriptor: texture_b,
                pixels: &malformed_b,
            },
        ];
        assert!(classifier
            .update_pbr_texture_table_in_place(&mut texture_table, &malformed_assets)
            .unwrap_err()
            .to_string()
            .contains("requires 8 bytes, got 7"));
        assert_eq!(
            classifier
                .read_pbr_texture_rgba8_for_diagnostics(&texture_table, 0)
                .await
                .unwrap(),
            updated_a,
            "a rejected candidate must not publish its valid prefix",
        );

        let replacement_descriptor = TextureAssetDescriptor {
            width: 4,
            height: 1,
            ..texture_a
        };
        let replacement_pixels = [43; 16];
        let replacement_assets =
            [Rgba8TextureAsset::new(replacement_descriptor, &replacement_pixels).unwrap()];
        assert_eq!(
            classifier
                .update_pbr_texture_table_in_place(&mut texture_table, &replacement_assets)
                .unwrap(),
            PbrTextureTableUpdate::ShapeChanged,
        );
        assert_eq!(
            texture_table.descriptors(),
            &[Some(texture_a), Some(texture_b)]
        );
        texture_table = classifier
            .upload_pbr_texture_table(&replacement_assets)
            .unwrap();
        assert_eq!(texture_table.descriptors(), &[Some(replacement_descriptor)]);
        assert_eq!(
            classifier
                .read_pbr_texture_rgba8_for_diagnostics(&texture_table, 0)
                .await
                .unwrap(),
            replacement_pixels,
        );
        classifier
            .device()
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        if let Some(error) = error_scope.pop().await {
            panic!("PBR texture residency validation: {error}");
        }

        let environment_descriptor = EnvironmentMapDescriptor {
            prefiltered_face_size: 2,
            prefiltered_mip_count: 2,
            irradiance_face_size: 1,
        };
        let exact_half_values = [0.0, 0.5, 1.0, 2.0];
        let prefiltered = (0..environment_descriptor.prefiltered_rgba32f_len().unwrap())
            .map(|index| exact_half_values[index % exact_half_values.len()])
            .collect::<Vec<_>>();
        let irradiance = (0..environment_descriptor.irradiance_rgba32f_len().unwrap())
            .map(|index| exact_half_values[(index + 1) % exact_half_values.len()])
            .collect::<Vec<_>>();
        let environment_asset =
            EnvironmentMapAsset::new(environment_descriptor, &prefiltered, &irradiance).unwrap();
        let environment_error_scope = classifier
            .device()
            .push_error_scope(wgpu::ErrorFilter::Validation);
        let environment = classifier
            .upload_pbr_environment_map(environment_asset)
            .unwrap();
        assert_eq!(environment.descriptor(), environment_descriptor);
        assert_eq!(environment.prefiltered_mip_count(), 2);
        assert_eq!(
            classifier
                .read_pbr_environment_face_for_diagnostics(&environment, true, 0, 3)
                .await
                .unwrap(),
            prefiltered[48..64],
        );
        assert_eq!(
            classifier
                .read_pbr_environment_face_for_diagnostics(&environment, true, 1, 5)
                .await
                .unwrap(),
            prefiltered[116..120],
        );
        assert_eq!(
            classifier
                .read_pbr_environment_face_for_diagnostics(&environment, false, 0, 4)
                .await
                .unwrap(),
            irradiance[16..20],
        );
        let mut out_of_range = prefiltered.clone();
        out_of_range[7] = 70_000.0;
        let out_of_range_error = classifier
            .upload_pbr_environment_map(EnvironmentMapAsset {
                descriptor: environment_descriptor,
                prefiltered_rgba32f: &out_of_range,
                irradiance_rgba32f: &irradiance,
            })
            .err()
            .expect("out-of-range environment must be rejected");
        assert!(out_of_range_error
            .to_string()
            .contains("exceeds RGBA16F range"));
        classifier
            .device()
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        if let Some(error) = environment_error_scope.pop().await {
            panic!("PBR environment residency validation: {error}");
        }

        let report = classifier.run_conformance_matrix().await.unwrap();
        assert_eq!(report.full_pipeline_words, 1);
        assert_eq!(report.resident_lod_words, 20);
        assert_eq!(report.resident_visibility_words, 4);
        assert_eq!(report.resident_bucket_words, 303);
        assert_eq!(report.resident_root_topology_words, 240);
        assert_eq!(report.resident_root_prepared_words, 1040);
        assert_eq!(report.resident_root_domain_words, 18);
        assert!(report.resident_adaptive_rendered_pixels >= 32);
        assert_eq!(report.resident_adaptive_image.size, [32, 32]);
        assert_eq!(
            report.resident_adaptive_image.covered_pixels as usize,
            report.resident_adaptive_rendered_pixels,
        );
        assert_ne!(report.resident_adaptive_image.rgba8_hash, 0);
        assert_eq!(report.resident_root_indirect_draws, 4);
        assert_eq!(report.adaptive_overlay_patches, 1);
        assert_eq!(report.adaptive_overlay_indirect_draws, 2);
        assert_eq!(report.coherence_words, 10);
        assert_eq!(report.prepared_patch_words, 104);
        assert!(report.rendered_patch_pixels >= 8);
        assert_eq!(report.shared_frame_draws, 2);
        assert_eq!(report.compacted_source_words, 89);
        assert_eq!(report.compacted_range_words, 15);
        assert_eq!(report.indirect_argument_words, 30);
        assert_eq!(report.indirect_draws, 3);

        let (prepared, source_instances, render_scene, render_frame) =
            shared_render_frame_fixture();
        assert!(supports_basic_pbr_frame(
            &render_scene,
            RenderFrameOptions::default(),
        ));
        let mut textured_scene = render_scene.clone();
        textured_scene.materials[0].textures.base_color = Some(0);
        textured_scene.materials[0].textures.normal = Some(0);
        assert!(supports_basic_pbr_frame(
            &textured_scene,
            RenderFrameOptions::default(),
        ));
        assert!(!supports_basic_pbr_frame(
            &render_scene,
            RenderFrameOptions {
                focus_postprocess: Some(quilting_core::render::FocusPostprocessPacket {
                    mode: quilting_core::render::FocusPostprocessMode::Spheroidal,
                    blur_radius_pixels: 11,
                    blur_strength: 1.0,
                    focus_coordinate: 0.5,
                    bandwidth: 0.1,
                    normalize_range: false,
                    stretch_range: [0.5, 0.5],
                    gaussian_passes: 1,
                    kawase_passes: 3,
                    kawase_offset: 1.5,
                }),
                ..RenderFrameOptions::default()
            },
        ));
        let foreign_prepared = prepared.clone();
        let root_prepared = prepared.clone();
        let atlas = complete_atlas();
        let mut model = classifier.upload_model(prepared, &atlas).unwrap();
        let mut foreign_model = classifier.upload_model(foreign_prepared, &atlas).unwrap();
        let pipeline = classifier.create_offscreen_patch_render_pipeline().unwrap();
        let validated_scene = ValidatedRenderScene::new(render_scene.clone()).unwrap();
        let mut retained_scene = classifier
            .upload_validated_patch_render_scene(
                &pipeline,
                &model,
                validated_scene.clone(),
                &source_instances,
                None,
            )
            .unwrap();
        assert!(retained_scene
            .validated_scene()
            .shares_snapshot_with(&validated_scene));
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
                None,
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
                    &pipeline,
                    &model,
                    &mut retained_scene,
                    reordered_scene.clone(),
                    &source_instances,
                    reordered_scene.revision,
                    None,
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
                &pipeline,
                &model,
                &mut retained_scene,
                resized_scene.clone(),
                &source_instances,
                resized_scene.revision,
                None,
            )
            .unwrap()
        {
            PatchRenderSceneUpdate::ShapeChanged(scene) => scene,
            PatchRenderSceneUpdate::Updated => panic!("batch-count change updated in place"),
        };
        assert_eq!(returned_scene.snapshot(), &resized_scene);
        assert_eq!(retained_scene.scene(), &reordered_scene);

        assert!(matches!(
            classifier
                .update_patch_render_scene_in_place(
                    &pipeline,
                    &model,
                    &mut retained_scene,
                    render_scene.clone(),
                    &source_instances,
                    render_scene.revision,
                    None,
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
                &[1, 1, 1, 3, 3, 0, 6],
                &[1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                &[0, 0, 0, 0, 1, 2],
                &[0, 1, 1, 2, 2, 0],
            )
            .unwrap();
        assert_eq!(packed_atlas.entry_count(), 1);
        assert_eq!(packed_atlas.vertex_count(), 3);
        assert_eq!(packed_atlas.triangle_index_count(), 6);
        assert_eq!(packed_atlas.line_index_count(), 6);
        assert!(supports_resident_root_render_scene(&render_scene, 2));
        let mut relod_scene = render_scene.clone();
        for batch in &mut relod_scene.batches {
            batch.id.key.lod = [2; 3];
            batch.triangle_index_count = 6;
            batch.line_index_count = 12;
            for member in &mut batch.members {
                member.edge_lods = [2; 3];
                member.vertex_lods = [2; 3];
            }
        }
        assert_eq!(
            resident_root_render_domains(&render_scene, 2).unwrap(),
            resident_root_render_domains(&relod_scene, 2).unwrap(),
            "root residency must ignore CPU-authored LOD buckets",
        );
        let mut suppressed_scene = render_scene.clone();
        suppressed_scene.suppressed_root_faces.push(0);
        assert!(!supports_resident_root_render_scene(&suppressed_scene, 2));
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
        let focus_scene = classifier
            .upload_focus_pbr_patch_render_scene(
                focus_pbr_pipeline,
                &model,
                render_scene.clone(),
                &source_instances,
                render_scene.revision,
                None,
            )
            .unwrap();
        classifier
            .write_patch_render_scene_state(&model, &focus_scene, LodPose::default(), 0, &[1, 1])
            .unwrap();
        let mut focus_view = render_frame.view.clone();
        focus_view.focus = FocusFieldPacket {
            sphere: [0.0, 0.0, 0.0, 1.0],
            enabled: true,
        };
        let focus_options = RenderFrameOptions {
            focus_postprocess: Some(quilting_core::render::FocusPostprocessPacket {
                mode: quilting_core::render::FocusPostprocessMode::Spheroidal,
                blur_radius_pixels: 11,
                blur_strength: 1.0,
                focus_coordinate: 0.5,
                bandwidth: 0.1,
                normalize_range: false,
                stretch_range: [0.5, 0.5],
                gaussian_passes: 1,
                kawase_passes: 3,
                kawase_offset: 1.5,
            }),
            ..RenderFrameOptions::default()
        };
        let focus_plan = focus_scene
            .command_plan(RenderStyle::Pbr, focus_options)
            .unwrap();
        let focus_frame = RenderFrame::from_command_plan(
            render_frame.revision + 100,
            render_frame.pose,
            focus_view,
            focus_options,
            &focus_plan,
        )
        .unwrap();
        let focus_error_scope = classifier
            .device()
            .push_error_scope(wgpu::ErrorFilter::Validation);
        let focus_encoding = classifier
            .render_offscreen_focus_pbr_patch_scene_with_face_visibility(
                &focus_frame,
                focus_pbr_pipeline,
                focus_pipelines,
                &focus_scene,
                &packed_atlas,
                focus_target,
                &target,
                true,
            )
            .unwrap();
        assert_eq!(focus_encoding.scene.indirect_draw_calls, 2);
        assert_eq!(focus_encoding.postprocess.render_passes, 8);
        classifier
            .device()
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        if let Some(error) = focus_error_scope.pop().await {
            panic!("complete focus PBR frame validation: {error}");
        }
        let raw_focus = classifier
            .stage_focus_raw_field_image(focus_target)
            .unwrap()
            .read()
            .await
            .unwrap();
        assert_eq!(raw_focus.size(), [32, 32]);
        assert!(raw_focus.covered_texels() > 0);
        for channel in 0..3 {
            let [minimum, maximum] = raw_focus.covered_channel_range(channel).unwrap();
            assert!(minimum.is_finite());
            assert!(maximum.is_finite());
            assert!(minimum >= 0.0);
            assert!(maximum <= 1.0);
        }
        assert!(
            offscreen_signature(&classifier, &target)
                .await
                .covered_pixels
                > 0
        );
        let root_atlas = prepare_lod_atlas_lookup(vec![[1, 1, 1]]).unwrap();
        let mut root_model = classifier.upload_model(root_prepared, &root_atlas).unwrap();
        let root_preparation = classifier
            .upload_resident_root_preparation_scene(&root_model, &render_scene, &source_instances)
            .unwrap();
        let root_geometry = classifier
            .upload_resident_geometry_bucket_scene(
                &root_model,
                &packed_atlas,
                root_preparation.draw_domains(),
            )
            .unwrap();
        let root_pipeline = focus_resources.root_pipeline();
        let root_bindings = classifier
            .create_resident_root_render_bindings(root_pipeline, &root_preparation, &root_geometry)
            .unwrap();
        let root_focus_bindings = classifier
            .create_resident_root_render_bindings_with_pbr(
                root_pipeline,
                &root_preparation,
                &root_geometry,
                &render_scene,
                None,
                Some(&environment),
            )
            .unwrap();
        assert!(root_focus_bindings.supports_resident_basic_pbr());
        let mut root_metrics = metrics(&root_atlas, 1.0, 0);
        root_metrics.viewport = [32.0, 32.0];
        {
            let classification = classifier
                .classify_on_device(
                    &mut root_model,
                    &identity_dispatch(),
                    root_metrics,
                    LodPose::default(),
                )
                .unwrap();
            classifier.reconcile_resident_lod_on_device(&classification, FaceLodGrading::TwoToOne);
        }
        let resident = classifier.latest_resident_lod(&root_model).unwrap();
        let root_focus_error_scope = classifier
            .device()
            .push_error_scope(wgpu::ErrorFilter::Validation);
        let root_focus_encoding = classifier
            .render_offscreen_focus_resident_roots(
                &focus_frame,
                focus_plan.scene().snapshot(),
                &root_model,
                &resident,
                &root_preparation,
                &root_geometry,
                root_pipeline,
                &root_focus_bindings,
                &packed_atlas,
                focus_pipelines,
                focus_target,
                &target,
                LodPose::default(),
                0,
                PoseUploadPolicy::Publish,
                true,
            )
            .unwrap();
        assert_eq!(root_focus_encoding.scene.indirect_draw_calls, 2);
        assert_eq!(root_focus_encoding.postprocess.render_passes, 8);
        classifier
            .device()
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        if let Some(error) = root_focus_error_scope.pop().await {
            panic!("resident root focus frame validation: {error}");
        }
        let root_focus_raw = classifier
            .stage_focus_raw_field_image(focus_target)
            .unwrap()
            .read()
            .await
            .unwrap();
        assert!(root_focus_raw.covered_texels() > 0);
        assert_eq!(
            classifier
                .classify_resident_root_visibility_for_diagnostics(
                    &render_frame,
                    &root_model,
                    &resident,
                    &root_preparation,
                    &root_geometry,
                    &root_bindings,
                    LodPose::default(),
                    0,
                    true,
                )
                .await
                .unwrap(),
            [0b11],
        );
        let mut hidden_frame = render_frame.clone();
        hidden_frame.view.mvp = translation_matrix(100.0, 0.0, 0.0);
        assert_eq!(
            classifier
                .classify_resident_root_visibility_for_diagnostics(
                    &hidden_frame,
                    &root_model,
                    &resident,
                    &root_preparation,
                    &root_geometry,
                    &root_bindings,
                    LodPose::default(),
                    0,
                    true,
                )
                .await
                .unwrap(),
            [0],
        );
        let root_encoding = classifier
            .render_offscreen_resident_roots(
                &render_frame,
                &render_scene,
                &root_model,
                &resident,
                &root_preparation,
                &root_geometry,
                root_pipeline,
                &root_bindings,
                &packed_atlas,
                &target,
                LodPose::default(),
                0,
                PoseUploadPolicy::Publish,
                true,
            )
            .unwrap();
        assert_eq!(
            root_encoding.logical_submission,
            render_frame
                .expected_submission_stats(&render_scene)
                .unwrap(),
        );
        assert_eq!(root_encoding.indirect_draw_calls, 2);
        assert_eq!(root_encoding.source_instance_count, 2);
        assert!(
            offscreen_signature(&classifier, &target)
                .await
                .covered_pixels
                > 0
        );
        let lod_frame = RenderFrame::build(
            render_frame.revision + 1,
            render_frame.pose,
            RenderStyle::Lod,
            render_frame.view.clone(),
            render_frame.options,
            &render_scene,
        )
        .unwrap();
        let lod_error_scope = classifier
            .device()
            .push_error_scope(wgpu::ErrorFilter::Validation);
        let lod_root_encoding = classifier
            .render_offscreen_resident_roots(
                &lod_frame,
                &render_scene,
                &root_model,
                &resident,
                &root_preparation,
                &root_geometry,
                root_pipeline,
                &root_bindings,
                &packed_atlas,
                &target,
                LodPose::default(),
                0,
                PoseUploadPolicy::Publish,
                true,
            )
            .unwrap();
        assert_eq!(
            lod_root_encoding.logical_submission,
            lod_frame.expected_submission_stats(&render_scene).unwrap(),
        );
        assert_eq!(lod_root_encoding.indirect_draw_calls, 2);
        assert_eq!(lod_root_encoding.source_instance_count, 2);
        classifier
            .device()
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        if let Some(error) = lod_error_scope.pop().await {
            panic!("resident root LOD validation: {error}");
        }
        assert!(
            offscreen_signature(&classifier, &target)
                .await
                .covered_pixels
                > 0
        );
        let mut unhighlighted_root_wire = None;
        for style in [
            RenderStyle::Matcap,
            RenderStyle::Wire,
            RenderStyle::MatcapWire,
            RenderStyle::Stretch,
        ] {
            let style_frame = RenderFrame::build(
                lod_frame.revision + style as u64 + 1,
                render_frame.pose,
                style,
                render_frame.view.clone(),
                render_frame.options,
                &render_scene,
            )
            .unwrap();
            let style_error_scope = classifier
                .device()
                .push_error_scope(wgpu::ErrorFilter::Validation);
            let style_encoding = classifier
                .render_offscreen_resident_roots(
                    &style_frame,
                    &render_scene,
                    &root_model,
                    &resident,
                    &root_preparation,
                    &root_geometry,
                    root_pipeline,
                    &root_bindings,
                    &packed_atlas,
                    &target,
                    LodPose::default(),
                    0,
                    PoseUploadPolicy::Publish,
                    true,
                )
                .unwrap();
            assert_eq!(
                style_encoding.logical_submission,
                style_frame
                    .expected_submission_stats(&render_scene)
                    .unwrap(),
            );
            assert_eq!(
                style_encoding.indirect_draw_calls,
                if style == RenderStyle::MatcapWire {
                    4
                } else {
                    2
                },
            );
            classifier
                .device()
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();
            if let Some(error) = style_error_scope.pop().await {
                panic!("resident root {style:?} validation: {error}");
            }
            let signature = offscreen_signature(&classifier, &target).await;
            assert!(signature.covered_pixels > 0, "{style:?}");
            if style == RenderStyle::Wire {
                unhighlighted_root_wire = Some(signature);
            }
        }
        let highlighted_root_wire = RenderFrame::build(
            lod_frame.revision + 20,
            render_frame.pose,
            RenderStyle::Wire,
            render_frame.view,
            RenderFrameOptions {
                highlight_face: Some(0),
                ..render_frame.options
            },
            &render_scene,
        )
        .unwrap();
        let highlighted_root_encoding = classifier
            .render_offscreen_resident_roots(
                &highlighted_root_wire,
                &render_scene,
                &root_model,
                &resident,
                &root_preparation,
                &root_geometry,
                root_pipeline,
                &root_bindings,
                &packed_atlas,
                &target,
                LodPose::default(),
                0,
                PoseUploadPolicy::Publish,
                true,
            )
            .unwrap();
        assert_eq!(highlighted_root_encoding.indirect_draw_calls, 4);
        let highlighted_root_signature = offscreen_signature(&classifier, &target).await;
        assert_ne!(
            highlighted_root_signature.rgba8_hash,
            unhighlighted_root_wire.unwrap().rgba8_hash,
        );
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
        let baseline_signature = offscreen_signature(&classifier, &target).await;

        // Animated surface pins change the same backend-neutral entity
        // transform consumed by both renderers. Prove that a shape-compatible
        // WebGPU scene update publishes both its ordinary affine chart and its
        // conformal map without rebuilding residency or reading GPU state back
        // into the ordinary frame loop.
        let mut affine_scene = render_scene.clone();
        affine_scene.revision += 1;
        for batch in &mut affine_scene.batches {
            batch.transform.euclidean_model = translation_matrix(0.35, 0.0, 0.0);
        }
        assert!(matches!(
            classifier
                .update_patch_render_scene_in_place(
                    &pipeline,
                    &model,
                    &mut retained_scene,
                    affine_scene.clone(),
                    &source_instances,
                    affine_scene.revision,
                    None,
                )
                .unwrap(),
            PatchRenderSceneUpdate::Updated,
        ));
        let affine_frame = RenderFrame::build(
            render_frame.revision + 1,
            render_frame.pose,
            render_frame.style,
            render_frame.view,
            render_frame.options,
            &affine_scene,
        )
        .unwrap();
        classifier
            .render_offscreen_normals_patch_scene(
                &affine_frame,
                &pipeline,
                &retained_scene,
                &packed_atlas,
                &target,
                true,
            )
            .unwrap();
        let affine_signature = offscreen_signature(&classifier, &target).await;
        assert_ne!(baseline_signature.rgba8_hash, affine_signature.rgba8_hash);

        let mut pinned_scene = affine_scene.clone();
        pinned_scene.revision += 1;
        pinned_scene.batches[0].transform.mobius =
            packed_mobius(Mobius::translation(Quat::from_point(0.0, 0.35, 0.0)));
        assert!(matches!(
            classifier
                .update_patch_render_scene_in_place(
                    &pipeline,
                    &model,
                    &mut retained_scene,
                    pinned_scene.clone(),
                    &source_instances,
                    pinned_scene.revision,
                    None,
                )
                .unwrap(),
            PatchRenderSceneUpdate::Updated,
        ));
        let pinned_frame = RenderFrame::build(
            affine_frame.revision + 1,
            affine_frame.pose,
            affine_frame.style,
            affine_frame.view,
            affine_frame.options,
            &pinned_scene,
        )
        .unwrap();
        classifier
            .render_offscreen_normals_patch_scene(
                &pinned_frame,
                &pipeline,
                &retained_scene,
                &packed_atlas,
                &target,
                true,
            )
            .unwrap();
        let pinned_signature = offscreen_signature(&classifier, &target).await;
        assert_ne!(affine_signature.rgba8_hash, pinned_signature.rgba8_hash);

        assert!(matches!(
            classifier
                .update_patch_render_scene_in_place(
                    &pipeline,
                    &model,
                    &mut retained_scene,
                    render_scene.clone(),
                    &source_instances,
                    render_scene.revision,
                    None,
                )
                .unwrap(),
            PatchRenderSceneUpdate::Updated,
        ));

        classifier
            .write_patch_render_pose_state(&model, &retained_scene, LodPose::default(), 0)
            .unwrap();

        let visible_dispatch = identity_dispatch();
        let (visible_signature, visible_epoch) = {
            let classification = classifier
                .classify_on_device(
                    &mut model,
                    &visible_dispatch,
                    metrics(&atlas, 1.0, 0),
                    LodPose::default(),
                )
                .unwrap();
            let resident = classifier
                .reconcile_resident_lod_on_device(&classification, FaceLodGrading::TwoToOne);
            let epoch = resident.classification_epoch();
            classifier
                .render_offscreen_diagnostic_patch_scene_with_resident_lod_visibility(
                    &render_frame,
                    &pipeline,
                    &retained_scene,
                    &resident,
                    &packed_atlas,
                    &target,
                    true,
                )
                .unwrap();
            (offscreen_signature(&classifier, &target).await, epoch)
        };
        assert!(visible_signature.covered_pixels > 0);
        let latest = classifier.latest_resident_lod(&model).unwrap();
        assert_eq!(latest.classification_epoch(), visible_epoch);
        assert_eq!(latest.grading(), FaceLodGrading::TwoToOne);

        let mut hidden_dispatch = visible_dispatch;
        hidden_dispatch.baseline_model = translation_matrix(100.0, 0.0, 0.0);
        let hidden_signature = {
            let classification = classifier
                .classify_on_device(
                    &mut model,
                    &hidden_dispatch,
                    metrics(&atlas, 1.0, 0),
                    LodPose::default(),
                )
                .unwrap();
            let resident = classifier
                .reconcile_resident_lod_on_device(&classification, FaceLodGrading::TwoToOne);
            classifier
                .render_offscreen_diagnostic_patch_scene_with_resident_lod_visibility(
                    &render_frame,
                    &pipeline,
                    &retained_scene,
                    &resident,
                    &packed_atlas,
                    &target,
                    true,
                )
                .unwrap();
            offscreen_signature(&classifier, &target).await
        };
        assert_eq!(hidden_signature.covered_pixels, 0);
        classifier.invalidate_resident_lod(&model);
        assert!(classifier.latest_resident_lod(&model).is_none());

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

        let diagnostic_pipelines = classifier
            .create_diagnostic_patch_render_pipelines(
                wgpu::TextureFormat::Rgba8Unorm,
                Some(wgpu::TextureFormat::Depth24Plus),
                1,
            )
            .unwrap();
        let _wire_pipeline = classifier
            .create_diagnostic_patch_render_pipeline(
                RenderStyle::Wire,
                wgpu::TextureFormat::Rgba8Unorm,
                Some(wgpu::TextureFormat::Depth24Plus),
                1,
            )
            .unwrap();
        let mut diagnostic_scene = classifier
            .upload_patch_render_scene(
                diagnostic_pipelines.get(RenderStyle::Pbr).unwrap(),
                &model,
                render_scene.clone(),
                &source_instances,
                render_scene.revision,
                None,
            )
            .unwrap();
        assert!(!diagnostic_scene
            .pbr_environment_bindings()
            .unwrap()
            .is_resident());
        assert!(!diagnostic_scene.supports_resident_basic_pbr_frame(RenderFrameOptions::default()));
        assert!(
            !diagnostic_scene.supports_resident_patch_presentation_frame(
                RenderStyle::Pbr,
                RenderFrameOptions::default(),
            )
        );
        assert!(diagnostic_scene.supports_resident_patch_presentation_frame(
            RenderStyle::Wire,
            RenderFrameOptions::default(),
        ));
        classifier
            .write_patch_render_pose_state(&model, &diagnostic_scene, LodPose::default(), 0)
            .unwrap();
        classifier
            .write_patch_render_face_visibility_bits(&diagnostic_scene, &[0b11])
            .unwrap();
        let mut diagnostic_hashes = Vec::new();
        for (revision, style) in [
            RenderStyle::Pbr,
            RenderStyle::Matcap,
            RenderStyle::Lod,
            RenderStyle::Stretch,
            RenderStyle::Wire,
            RenderStyle::MatcapWire,
        ]
        .into_iter()
        .enumerate()
        {
            let options = RenderFrameOptions {
                highlight_face: (style == RenderStyle::Wire).then_some(0),
                matcap_style: MatcapStyle::GoldenSoft,
                ..render_frame.options
            };
            let plan = diagnostic_scene.command_plan(style, options).unwrap();
            assert!(plan
                .scene()
                .shares_snapshot_with(diagnostic_scene.validated_scene()));
            let frame = RenderFrame::from_command_plan(
                20 + revision as u64,
                render_frame.pose,
                render_frame.view,
                options,
                &plan,
            )
            .unwrap();
            let error_scope = classifier
                .device()
                .push_error_scope(wgpu::ErrorFilter::Validation);
            let encoding = classifier
                .render_offscreen_supported_patch_scene_with_face_visibility(
                    &frame,
                    &diagnostic_pipelines,
                    &diagnostic_scene,
                    &packed_atlas,
                    &target,
                    true,
                )
                .unwrap();
            let expected_draws = if style == RenderStyle::MatcapWire || style == RenderStyle::Wire {
                4
            } else {
                2
            };
            assert_eq!(encoding.indirect_draw_calls, expected_draws);
            assert_eq!(
                encoding.logical_submission,
                frame
                    .execution(diagnostic_scene.scene())
                    .unwrap()
                    .submission_stats()
            );
            let image = classifier
                .stage_offscreen_patch_render_target_image(&target)
                .unwrap()
                .read()
                .await
                .unwrap();
            let image = render_image_signature(image.view().unwrap(), 0);
            assert!(image.covered_pixels > 0, "{style:?} rendered no patches");
            diagnostic_hashes.push(image.rgba8_hash);
            classifier
                .device()
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();
            if let Some(error) = error_scope.pop().await {
                panic!("{style:?} shared-pipeline validation: {error}");
            }
        }
        for (left, left_hash) in diagnostic_hashes.iter().enumerate() {
            for right_hash in &diagnostic_hashes[left + 1..] {
                assert_ne!(left_hash, right_hash);
            }
        }

        classifier
            .replace_patch_render_scene_environment_bindings(
                diagnostic_pipelines.get(RenderStyle::Pbr).unwrap(),
                &mut diagnostic_scene,
                Some(&environment),
            )
            .unwrap();
        assert!(diagnostic_scene
            .pbr_environment_bindings()
            .unwrap()
            .is_resident());
        assert_eq!(
            diagnostic_scene
                .pbr_environment_bindings()
                .unwrap()
                .descriptor(),
            Some(environment_descriptor),
        );
        assert!(diagnostic_scene.supports_resident_basic_pbr_frame(RenderFrameOptions::default()));
        assert!(diagnostic_scene.supports_resident_patch_presentation_frame(
            RenderStyle::Pbr,
            RenderFrameOptions::default(),
        ));
        let environment_frame = RenderFrame::build(
            90,
            render_frame.pose,
            RenderStyle::Pbr,
            render_frame.view,
            render_frame.options,
            &render_scene,
        )
        .unwrap();
        classifier
            .render_offscreen_supported_patch_scene_with_face_visibility(
                &environment_frame,
                &diagnostic_pipelines,
                &diagnostic_scene,
                &packed_atlas,
                &target,
                true,
            )
            .unwrap();
        let environment_image = classifier
            .stage_offscreen_patch_render_target_image(&target)
            .unwrap()
            .read()
            .await
            .unwrap();
        let environment_signature = render_image_signature(environment_image.view().unwrap(), 0);
        assert!(environment_signature.covered_pixels > 0);
        assert_ne!(environment_signature.rgba8_hash, diagnostic_hashes[0]);

        let mut textured_diagnostic_scene = classifier
            .upload_patch_render_scene(
                diagnostic_pipelines.get(RenderStyle::Pbr).unwrap(),
                &model,
                textured_scene.clone(),
                &source_instances,
                textured_scene.revision,
                None,
            )
            .unwrap();
        assert_eq!(
            textured_diagnostic_scene.pbr_texture_residency().unwrap()[0].unresolved_mask(),
            0b101,
        );
        assert!(!textured_diagnostic_scene
            .supports_resident_basic_pbr_frame(RenderFrameOptions::default()));
        classifier
            .replace_patch_render_scene_texture_bindings(
                diagnostic_pipelines.get(RenderStyle::Pbr).unwrap(),
                &mut textured_diagnostic_scene,
                Some(&sparse_table),
            )
            .unwrap();
        assert_eq!(
            textured_diagnostic_scene.pbr_texture_residency().unwrap()[0].resident_mask(),
            0b101,
        );
        assert!(!textured_diagnostic_scene
            .supports_resident_basic_pbr_frame(RenderFrameOptions::default()));
        classifier
            .replace_patch_render_scene_environment_bindings(
                diagnostic_pipelines.get(RenderStyle::Pbr).unwrap(),
                &mut textured_diagnostic_scene,
                Some(&environment),
            )
            .unwrap();
        assert!(textured_diagnostic_scene
            .supports_resident_basic_pbr_frame(RenderFrameOptions::default()));
        classifier
            .write_patch_render_pose_state(
                &model,
                &textured_diagnostic_scene,
                LodPose::default(),
                0,
            )
            .unwrap();
        classifier
            .write_patch_render_face_visibility_bits(&textured_diagnostic_scene, &[0b11])
            .unwrap();
        let textured_frame = RenderFrame::build(
            91,
            render_frame.pose,
            RenderStyle::Pbr,
            render_frame.view,
            render_frame.options,
            &textured_scene,
        )
        .unwrap();
        classifier
            .render_offscreen_supported_patch_scene_with_face_visibility(
                &textured_frame,
                &diagnostic_pipelines,
                &textured_diagnostic_scene,
                &packed_atlas,
                &target,
                true,
            )
            .unwrap();
        let textured_image = classifier
            .stage_offscreen_patch_render_target_image(&target)
            .unwrap()
            .read()
            .await
            .unwrap();
        let textured_signature = render_image_signature(textured_image.view().unwrap(), 0);
        assert!(textured_signature.covered_pixels > 0);
        assert_ne!(textured_signature.rgba8_hash, diagnostic_hashes[0]);

        let compaction_scene = RenderSceneSnapshot {
            revision: 19,
            materials: Vec::new(),
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
        let words = pack_wgsl_visibility_compaction_scene_words(&compaction_scene).unwrap();
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
        assert_eq!(
            actual.triangle_indirect_arguments,
            expected.indirect_arguments
        );
        let expected_lines = wgsl_visibility_compaction_oracle_words(
            &compaction_scene,
            &source_visibility,
            RenderGeometry::Lines,
        )
        .unwrap();
        assert_eq!(
            actual.line_indirect_arguments,
            expected_lines.indirect_arguments
        );

        let mut roots = compaction_batch(0, [0, 1], true);
        roots.id.layer = RenderBatchLayer::RetainedRoot;
        let mut overlay = compaction_batch(0, [0], true);
        overlay.id.layer = RenderBatchLayer::AdaptiveOverlay;
        overlay.members[0].leaf_id = ScreenPatchLeafId::ROOT.child(0).unwrap();
        let replacement_scene = RenderSceneSnapshot {
            revision: 20,
            materials: Vec::new(),
            suppressed_root_faces: vec![0],
            batches: vec![roots, overlay],
        };
        let expected = wgsl_visibility_compaction_oracle_words(
            &replacement_scene,
            &[1, 1, 1],
            RenderGeometry::Lines,
        )
        .unwrap();
        let words = pack_wgsl_visibility_compaction_scene_words(&replacement_scene).unwrap();
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
        assert_eq!(actual.line_indirect_arguments, expected.indirect_arguments);
        let expected_triangles = wgsl_visibility_compaction_oracle_words(
            &replacement_scene,
            &[1, 1, 1],
            RenderGeometry::Triangles,
        )
        .unwrap();
        assert_eq!(
            actual.triangle_indirect_arguments,
            expected_triangles.indirect_arguments
        );

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
