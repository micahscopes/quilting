#![cfg(not(target_arch = "wasm32"))]

use quilting_core::quaternion::{Mobius, Quat};
use quilting_renderer::compute::{
    pack_lod_classification, pack_wgsl_lod_atlas_words, pack_wgsl_lod_model_words,
    prepare_lod_atlas_lookup, prepare_lod_dispatch_state, prepare_lod_model,
    reconcile_and_pack_wgsl_lod_pass2, unpack_lod_classification_fields, LodAtlasLookup,
    LodDispatchState, LodModelData, LodSubjectState, PackedLodClassification, PreparedLodModel,
    WgslLodDispatchMetrics,
};
use quilting_webgpu::{LodClassifierDevice, LodPose};

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

        let prepared = prepare_lod_model(LodModelData {
            positions: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            faces: vec![[0, 1, 2]],
            joint_indices: vec![[0; 4]; 3],
            joint_weights: vec![[0.0; 4]; 3],
            morph_deltas: Vec::new(),
            num_morph_targets: 0,
            face_nodes: vec![0],
        })
        .unwrap();
        let atlas = prepare_lod_atlas_lookup([[1, 1, 2]]).unwrap();
        let model_words = pack_wgsl_lod_model_words(&prepared).unwrap();
        let atlas_words = pack_wgsl_lod_atlas_words(&atlas);
        let expected = reconcile_and_pack_wgsl_lod_pass2(
            &[[1.0, 0.0, 0.0, 1.0]],
            &model_words.adjacency,
            &atlas_words,
        )
        .unwrap();
        let dispatch = identity_dispatch();
        let baseline_metrics = WgslLodDispatchMetrics {
            view_projection: identity_matrix(),
            density: 1.0,
            pixel_floor: 0.0,
            max_lod: atlas.max_lod,
            viewport: [1024.0, 1024.0],
            // The fixture does not reference either joint. This deliberately
            // proves that unused skin joints do not inflate retained buffers
            // or make a valid dispatch fail capacity validation.
            num_joints: 2,
        };
        let mut joint_matrices = Vec::with_capacity(32);
        joint_matrices.extend_from_slice(&identity_matrix());
        joint_matrices.extend_from_slice(&identity_matrix());
        let mut resident = classifier.upload_model(prepared, &atlas).unwrap();
        let actual = classifier
            .classify(
                &mut resident,
                &dispatch,
                baseline_metrics,
                LodPose {
                    joint_matrices: &joint_matrices,
                    morph_weights: &[],
                },
            )
            .await
            .unwrap();
        assert_eq!(actual, expected);

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
        // A visible neighbor promotes face 7's first edge; an invisible high
        // neighbor on its second edge must not promote it.
        adjacency[7 * 3] = [8, 2, 0, 0];
        adjacency[7 * 3 + 1] = [9, 0, 0, 0];
        let mut atlas_lut = vec![u8::MAX as u32; 1_200];
        atlas_lut[321] = 17;
        atlas_lut[411] = 29;
        atlas_lut[654] = 31;
        let expected = reconcile_and_pack_wgsl_lod_pass2(&pass1, &adjacency, &atlas_lut).unwrap();
        let actual = classifier
            .reconcile_conformance_records(&pass1, &adjacency, &atlas_lut)
            .await
            .unwrap();
        assert_eq!(actual, expected);

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
