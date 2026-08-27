#![cfg(not(target_arch = "wasm32"))]

use quilting_renderer::compute::{
    pack_wgsl_lod_atlas_words, pack_wgsl_lod_model_words, prepare_lod_atlas_lookup,
    prepare_lod_model, reconcile_and_pack_wgsl_lod_pass2, LodDispatchState, LodModelData,
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

#[test]
fn native_two_pass_classifier_matches_the_packed_cpu_oracle() {
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
                metrics,
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
    });
}
