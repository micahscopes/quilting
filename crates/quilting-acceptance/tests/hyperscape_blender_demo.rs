use hyperscape::{interchange::HyperscapeGltfRuntime, EntityFrame, EuclideanCoordinates};
use std::time::Duration;

const BLENDER_DEMO: &[u8] = include_bytes!("../../../examples/hyperscape-blender-demo.glb");

#[test]
fn blender_demo_has_renderable_fallback_and_full_timeline() {
    let scene = quilting_gltf::load_gltf(BLENDER_DEMO).unwrap();
    assert!(!scene.meshes.is_empty());
    assert!(scene
        .nodes
        .iter()
        .any(|node| node.name.as_deref() == Some("HS_Traveler") && node.mesh.is_some()));
    assert!(scene
        .nodes
        .iter()
        .any(|node| node.name.as_deref() == Some("HS_ProjectionCamera")));

    let asset = scene.hyperscape.unwrap();
    assert_eq!(asset.payload.frames.len(), 3);
    assert_eq!(asset.payload.walls.len(), 4);
    assert_eq!(asset.payload.anchors.len(), 2);
    assert_eq!(asset.payload.paths.len(), 1);
    assert_eq!(asset.payload.paths[0].coordinate_frame, Some(0));
    assert_eq!(
        asset.payload.paths[0]
            .transitions
            .iter()
            .map(|transition| transition.frame)
            .collect::<Vec<_>>(),
        vec![1, 2, 0]
    );
    assert_eq!(asset.payload.constraints.len(), 2);
    asset.validate().unwrap();
}

#[test]
fn blender_demo_runs_enter_reanchor_exit_timeline() {
    let (nodes, asset) = quilting_gltf::load_hyperscape_graph(BLENDER_DEMO).unwrap();
    let traveler_node = nodes
        .iter()
        .position(|node| node.name.as_deref() == Some("HS_Traveler"))
        .unwrap();
    let camera_node = nodes
        .iter()
        .position(|node| node.name.as_deref() == Some("HS_ProjectionCamera"))
        .unwrap();
    let asset = asset.unwrap();
    let bound_node_count = asset
        .node_bindings
        .iter()
        .filter(|binding| binding.is_some())
        .count();
    let wall_count = asset.payload.walls.len();
    assert_eq!(bound_node_count, 5);
    assert_eq!(wall_count, 4);
    let mut runtime = HyperscapeGltfRuntime::new(&nodes, &asset).unwrap();
    let traveler = runtime.entities()[traveler_node];

    assert!(runtime.packets_by_node().iter().any(|packet| {
        packet.subject_node == traveler_node && packet.camera_node == camera_node
    }));
    let initial_diagnostics = runtime.diagnostic_snapshot();
    assert_eq!(initial_diagnostics.frames.len(), 3);
    assert!(initial_diagnostics
        .entities
        .iter()
        .any(|entity| entity.node == traveler_node && entity.local == entity.euclidean));
    assert_eq!(initial_diagnostics.contacts.len(), 6);
    assert_eq!(
        initial_diagnostics
            .chamber_aggregates
            .classifications_last_tick,
        bound_node_count * wall_count
    );
    assert_eq!(
        initial_diagnostics
            .chamber_aggregates
            .counts
            .iter()
            .map(|(_, count)| count)
            .sum::<usize>(),
        bound_node_count
    );
    assert_eq!(
        initial_diagnostics
            .chamber_aggregates
            .aggregate_updates_last_tick,
        bound_node_count
    );
    assert!(initial_diagnostics
        .visibility_hints
        .iter()
        .all(|hint| !hint.can_cull));
    assert!(initial_diagnostics.visibility_hints.iter().any(|hint| {
        hint.comparable_chambers
            && !hint.same_chamber
            && !hint.separating_walls.is_empty()
            && hint
                .separating_walls
                .iter()
                .all(|wall| hint.contact_frontier.contains(wall))
    }));
    assert!(runtime
        .packets()
        .iter()
        .all(|packet| packet.origin_pole_denominator_norm_sq.is_finite()));

    for (seconds, expected_frame, expected_ambient) in [
        (2, 1, [-1.2, 1.2, 0.8]),
        (2, 2, [1.0, 1.2, 0.5]),
        (2, 0, [3.0, 1.2, 0.7]),
    ] {
        runtime.tick(Duration::from_secs(seconds));
        assert_eq!(
            runtime
                .app()
                .world()
                .get::<EntityFrame>(traveler)
                .map(|frame| frame.0 .0),
            Some(expected_frame)
        );
        let actual = runtime
            .app()
            .world()
            .get::<EuclideanCoordinates>(traveler)
            .unwrap()
            .0;
        for axis in 0..3 {
            assert!(
                (actual[axis] - expected_ambient[axis]).abs() < 1.0e-6,
                "axis {axis}: actual={actual:?}, expected={expected_ambient:?}"
            );
        }
    }
    let final_diagnostics = runtime.diagnostic_snapshot();
    let traveler_diagnostics = final_diagnostics
        .entities
        .iter()
        .find(|entity| entity.node == traveler_node)
        .unwrap();
    assert!(traveler_diagnostics.history.len() >= 4);
    assert!(final_diagnostics.transform_history_epoch >= 4);
    assert!(runtime.diagnostics().is_empty());
}
