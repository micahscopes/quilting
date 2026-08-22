use quilting_core::{
    ConformalGenerator, ConformalTransformChain, RoundSideOrientation, RoundWallGeometry,
};
use quilting_round_index::{
    classify_containment, classify_separation, enclose_negative_spheres, Containment,
    ContainmentCertificate, NodeId, NodeSpec, PoseKey, PredicateConfig, QueryResult, RefitBound,
    RoundIndex, RoundIndexError, RoundQuery, RoundSide, RoundSideAutomorphism, Separation,
    TopologyKey,
};

const EPS: f64 = 1.0e-9;

fn inside_sphere(center: [f64; 3], radius: f64) -> RoundSide {
    RoundSide::sphere(center, radius, RoundSideOrientation::Negative).unwrap()
}

fn outside_sphere(center: [f64; 3], radius: f64) -> RoundSide {
    RoundSide::sphere(center, radius, RoundSideOrientation::Positive).unwrap()
}

fn plane(normal: [f64; 3], offset: f64, orientation: RoundSideOrientation) -> RoundSide {
    RoundSide::plane(normal, offset, orientation).unwrap()
}

fn assert_close(a: f64, b: f64) {
    assert!((a - b).abs() < EPS, "{a} != {b}");
}

fn assert_point_close(a: [f64; 3], b: [f64; 3]) {
    for axis in 0..3 {
        assert_close(a[axis], b[axis]);
    }
}

fn assert_side_close(actual: RoundSide, expected: RoundSide) {
    assert_eq!(actual.orientation(), expected.orientation());
    match (actual.geometry(), expected.geometry()) {
        (
            RoundWallGeometry::Sphere {
                center: a,
                radius: ar,
            },
            RoundWallGeometry::Sphere {
                center: b,
                radius: br,
            },
        ) => {
            assert_point_close(a, b);
            assert_close(ar, br);
        }
        (
            RoundWallGeometry::Plane {
                unit_normal: an,
                offset: ao,
            },
            RoundWallGeometry::Plane {
                unit_normal: bn,
                offset: bo,
            },
        ) => {
            assert_point_close(an, bn);
            assert_close(ao, bo);
        }
        pair => panic!("geometry mismatch: {pair:?}"),
    }
}

#[test]
fn separation_handles_oriented_spheres_and_planes_conservatively() {
    let p = PredicateConfig::default();
    assert_eq!(
        classify_separation(
            &inside_sphere([0.0; 3], 1.0),
            &inside_sphere([4.0, 0.0, 0.0], 1.0),
            p,
        ),
        Separation::Disjoint
    );
    assert_eq!(
        classify_separation(
            &inside_sphere([0.0; 3], 1.0),
            &outside_sphere([0.0; 3], 3.0),
            p,
        ),
        Separation::Disjoint
    );
    assert_eq!(
        classify_separation(
            &outside_sphere([-4.0, 0.0, 0.0], 1.0),
            &outside_sphere([4.0, 0.0, 0.0], 1.0),
            p,
        ),
        Separation::IntersectsOrUnknown
    );

    let x_positive = plane([1.0, 0.0, 0.0], 0.0, RoundSideOrientation::Positive);
    assert_eq!(
        classify_separation(&inside_sphere([-3.0, 0.0, 0.0], 1.0), &x_positive, p),
        Separation::Disjoint
    );
    assert_eq!(
        classify_separation(&outside_sphere([-3.0, 0.0, 0.0], 1.0), &x_positive, p),
        Separation::IntersectsOrUnknown
    );

    let x_negative = plane([1.0, 0.0, 0.0], 0.0, RoundSideOrientation::Negative);
    let x_above_two = plane([1.0, 0.0, 0.0], 2.0, RoundSideOrientation::Positive);
    assert_eq!(
        classify_separation(&x_negative, &x_above_two, p),
        Separation::Disjoint
    );

    // Open tangent sides really are disjoint, but the floating predicate
    // deliberately declines to prune without clearance.
    assert_eq!(
        classify_separation(
            &inside_sphere([0.0; 3], 1.0),
            &inside_sphere([2.0, 0.0, 0.0], 1.0),
            p,
        ),
        Separation::IntersectsOrUnknown
    );
}

#[test]
fn every_certified_sampled_separation_has_no_sampled_counterexample() {
    let sides = [
        inside_sphere([-2.0, 0.0, 0.0], 0.7),
        outside_sphere([-2.0, 0.0, 0.0], 0.7),
        inside_sphere([2.0, 0.5, 0.0], 0.8),
        outside_sphere([2.0, 0.5, 0.0], 0.8),
        plane([1.0, 0.0, 0.0], -1.0, RoundSideOrientation::Negative),
        plane([1.0, 0.0, 0.0], 1.0, RoundSideOrientation::Positive),
        plane([0.0, 1.0, 0.0], 0.5, RoundSideOrientation::Negative),
    ];
    let config = PredicateConfig::new(1.0e-9).unwrap();
    for first in &sides {
        for second in &sides {
            if classify_separation(first, second, config) != Separation::Disjoint {
                continue;
            }
            for ix in -24..=24 {
                for iy in -24..=24 {
                    for iz in -4..=4 {
                        let point = [ix as f64 * 0.25, iy as f64 * 0.25, iz as f64 * 0.5];
                        assert!(
                            !(first.contains(point).unwrap() && second.contains(point).unwrap()),
                            "false separation for {first:?}, {second:?} at {point:?}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn containment_is_computable_where_clear_and_unknown_at_qb_style_boundary() {
    let config = PredicateConfig::default();
    assert_eq!(
        classify_containment(
            &inside_sphere([1.0, 0.0, 0.0], 1.0),
            &inside_sphere([0.0; 3], 5.0),
            config,
        ),
        Containment::Contained
    );
    assert_eq!(
        classify_containment(
            &inside_sphere([5.0, 0.0, 0.0], 2.0),
            &inside_sphere([0.0; 3], 5.0),
            config,
        ),
        Containment::NotContained
    );
    assert_eq!(
        classify_containment(
            &inside_sphere([4.0, 0.0, 0.0], 1.0),
            &inside_sphere([0.0; 3], 5.0),
            config,
        ),
        Containment::Unknown
    );
}

fn spec(
    id: u64,
    parent: Option<u64>,
    bound: RoundSide,
    payload: &'static str,
) -> NodeSpec<&'static str> {
    NodeSpec {
        id: NodeId(id),
        parent: parent.map(NodeId),
        parent_containment: ContainmentCertificate::Computed,
        bound,
        payload,
    }
}

#[test]
fn build_rejects_bad_references_cycles_duplicates_and_unproved_containment() {
    let root = spec(0, None, inside_sphere([0.0; 3], 5.0), "root");
    let outside = spec(1, Some(0), inside_sphere([10.0, 0.0, 0.0], 1.0), "outside");
    assert!(matches!(
        RoundIndex::build(vec![root.clone(), outside]),
        Err(RoundIndexError::ContainmentNotSatisfied { .. })
    ));

    let tangent = spec(1, Some(0), inside_sphere([4.0, 0.0, 0.0], 1.0), "tangent");
    assert!(matches!(
        RoundIndex::build(vec![root.clone(), tangent]),
        Err(RoundIndexError::ContainmentUnproven { .. })
    ));
    let mut externally_certified = spec(
        1,
        Some(0),
        inside_sphere([4.0, 0.0, 0.0], 1.0),
        "externally certified tangent/QB bound",
    );
    externally_certified.parent_containment = ContainmentCertificate::Trusted;
    assert!(RoundIndex::build(vec![root.clone(), externally_certified]).is_ok());
    assert!(matches!(
        RoundIndex::build(vec![root.clone(), root]),
        Err(RoundIndexError::DuplicateNode(NodeId(0)))
    ));
    assert!(matches!(
        RoundIndex::build(vec![spec(
            1,
            Some(99),
            inside_sphere([0.0; 3], 1.0),
            "orphan"
        )]),
        Err(RoundIndexError::UnknownParent { .. })
    ));
    assert!(matches!(
        RoundIndex::build(vec![
            spec(0, Some(1), inside_sphere([0.0; 3], 4.0), "a"),
            spec(1, Some(0), inside_sphere([0.0; 3], 3.0), "b"),
        ]),
        Err(RoundIndexError::Cycle(_))
    ));
    assert!(matches!(
        PredicateConfig::new(-1.0),
        Err(RoundIndexError::InvalidClearance)
    ));
}

fn two_leaf_index() -> RoundIndex<&'static str> {
    RoundIndex::build_for(
        TopologyKey {
            asset_revision: 7,
            topology_revision: 11,
        },
        vec![
            spec(0, None, inside_sphere([0.0; 3], 10.0), "root"),
            spec(1, Some(0), inside_sphere([-4.0, 0.0, 0.0], 0.5), "left"),
            spec(2, Some(0), inside_sphere([4.0, 0.0, 0.0], 0.5), "right"),
        ],
        PredicateConfig::default(),
    )
    .unwrap()
}

#[test]
fn traversal_matches_clear_brute_force_leaf_case() {
    let index = two_leaf_index();
    let query = RoundQuery::from(plane([1.0, 0.0, 0.0], 0.0, RoundSideOrientation::Positive));
    let result = index.query(&query);
    assert_eq!(result.candidate_leaves, vec![NodeId(2)]);
    assert_eq!(
        index.candidate_payloads(&result).collect::<Vec<_>>(),
        vec![(NodeId(2), &"right")]
    );
    assert_eq!(result.visited_nodes, 3);
    assert_eq!(result.pruned_nodes, 1);

    let all = index.query(&RoundQuery::whole_space());
    assert_eq!(
        all,
        QueryResult {
            candidate_leaves: vec![NodeId(1), NodeId(2)],
            visited_nodes: 3,
            pruned_nodes: 0,
        }
    );
}

#[test]
fn animated_refit_is_atomic_keeps_topology_and_updates_ancestors() {
    let mut index = two_leaf_index();
    let ids_before = index.nodes().map(|node| node.id()).collect::<Vec<_>>();
    let topology = index.topology_key();
    let pose = PoseKey {
        clip_revision: 3,
        sample: 120,
    };
    let report = index
        .refit(
            pose,
            &[(NodeId(2), inside_sphere([8.0, 1.0, 0.0], 0.75))],
            |_, _, children| {
                Ok(RefitBound {
                    bound: enclose_negative_spheres(children, 1.0e-8)?,
                    child_containment: ContainmentCertificate::Computed,
                })
            },
        )
        .unwrap();

    assert_eq!(report.updated_leaves, 1);
    assert_eq!(report.refit_internal_nodes, 1);
    assert_eq!(index.current_pose(), Some(pose));
    assert_eq!(index.topology_key(), topology);
    assert_eq!(
        index.nodes().map(|node| node.id()).collect::<Vec<_>>(),
        ids_before
    );
    for &child in index.node(NodeId(0)).unwrap().children() {
        assert_eq!(
            classify_containment(
                index.node(child).unwrap().bound(),
                index.node(NodeId(0)).unwrap().bound(),
                index.predicates(),
            ),
            Containment::Contained
        );
    }

    let new_region = RoundQuery::from(inside_sphere([8.0, 1.0, 0.0], 2.0));
    assert_eq!(index.query(&new_region).candidate_leaves, vec![NodeId(2)]);

    let bounds_before_failure = index
        .nodes()
        .map(|node| (node.id(), *node.bound()))
        .collect::<Vec<_>>();
    let failed = index.refit(
        PoseKey {
            clip_revision: 3,
            sample: 121,
        },
        &[(NodeId(2), inside_sphere([30.0, 0.0, 0.0], 1.0))],
        |_, _, _| {
            Ok(RefitBound {
                bound: inside_sphere([0.0; 3], 2.0),
                child_containment: ContainmentCertificate::Computed,
            })
        },
    );
    assert!(matches!(
        failed,
        Err(RoundIndexError::ContainmentNotSatisfied { .. })
    ));
    assert_eq!(index.current_pose(), Some(pose));
    assert_eq!(
        index
            .nodes()
            .map(|node| (node.id(), *node.bound()))
            .collect::<Vec<_>>(),
        bounds_before_failure
    );
}

#[test]
fn animated_refit_runs_deepest_first_and_keeps_every_parent_conservative() {
    let mut index = RoundIndex::build(vec![
        spec(0, None, inside_sphere([0.0; 3], 20.0), "root"),
        spec(1, Some(0), inside_sphere([-5.0, 0.0, 0.0], 5.0), "branch"),
        spec(2, Some(1), inside_sphere([-5.0, 0.0, 0.0], 1.0), "moving"),
        spec(3, Some(0), inside_sphere([5.0, 0.0, 0.0], 1.0), "fixed"),
    ])
    .unwrap();
    let mut order = Vec::new();
    index
        .refit(
            PoseKey {
                clip_revision: 9,
                sample: 4,
            },
            &[(NodeId(2), inside_sphere([-8.0, 2.0, 0.0], 1.5))],
            |id, _, children| {
                order.push(id);
                Ok(RefitBound {
                    bound: enclose_negative_spheres(children, 1.0e-8)?,
                    child_containment: ContainmentCertificate::Computed,
                })
            },
        )
        .unwrap();
    assert_eq!(order, vec![NodeId(1), NodeId(0)]);
    for parent in [NodeId(1), NodeId(0)] {
        let parent_node = index.node(parent).unwrap();
        for &child in parent_node.children() {
            assert_eq!(
                classify_containment(
                    index.node(child).unwrap().bound(),
                    parent_node.bound(),
                    index.predicates(),
                ),
                Containment::Contained
            );
        }
    }
}

#[test]
fn generators_transform_round_sides_and_round_trip() {
    let sphere = inside_sphere([1.0, -2.0, 0.5], 1.25);
    let translation = ConformalGenerator::translation([3.0, 4.0, -1.0]);
    assert_side_close(
        translation.push_side(&sphere).unwrap(),
        inside_sphere([4.0, 2.0, -0.5], 1.25),
    );

    let negative_scale = ConformalGenerator::uniform_scale(-2.0);
    assert_side_close(
        negative_scale.push_side(&sphere).unwrap(),
        inside_sphere([-2.0, 4.0, -1.0], 2.5),
    );

    let rotation =
        ConformalGenerator::rotation_axis_angle([0.0, 0.0, 1.0], std::f64::consts::FRAC_PI_2)
            .unwrap();
    assert_side_close(
        rotation
            .push_side(&inside_sphere([1.0, 0.0, 0.0], 0.5))
            .unwrap(),
        inside_sphere([0.0, 1.0, 0.0], 0.5),
    );

    let reflection = ConformalGenerator::sphere_reflection([0.0; 3], 1.0);
    let through_pole = inside_sphere([1.0, 0.0, 0.0], 1.0);
    assert_side_close(
        reflection.push_side(&through_pole).unwrap(),
        plane([-1.0, 0.0, 0.0], -0.5, RoundSideOrientation::Negative),
    );
    assert_side_close(
        reflection.push_side(&inside_sphere([0.0; 3], 2.0)).unwrap(),
        outside_sphere([0.0; 3], 0.5),
    );
    let source_plane = plane([1.0, 0.0, 0.0], 1.0, RoundSideOrientation::Negative);
    assert_side_close(
        reflection.push_side(&source_plane).unwrap(),
        outside_sphere([0.5, 0.0, 0.0], 0.5),
    );

    for generator in [translation, negative_scale, rotation, reflection] {
        let pushed = generator.push_side(&sphere).unwrap();
        assert_side_close(generator.pull_side(&pushed).unwrap(), sphere);
        let pushed_plane = generator.push_side(&source_plane).unwrap();
        assert_side_close(generator.pull_side(&pushed_plane).unwrap(), source_plane);
    }
}

#[test]
fn pulled_query_membership_matches_output_chart_membership() {
    let chain = ConformalTransformChain::new(vec![
        ConformalGenerator::translation([0.5, -0.2, 0.1]),
        ConformalGenerator::sphere_reflection([2.0, 0.0, 0.0], 1.5),
        ConformalGenerator::uniform_scale(1.7),
    ])
    .unwrap();
    let output_query = RoundQuery::new(vec![
        inside_sphere([0.0; 3], 4.0),
        plane([1.0, 0.0, 0.0], -1.0, RoundSideOrientation::Positive),
    ])
    .unwrap();
    let source_query = output_query.pullback(&chain).unwrap();

    for x in -4..=4 {
        for y in -3..=3 {
            for z in -2..=2 {
                let source = [x as f64 * 0.37, y as f64 * 0.41, z as f64 * 0.53];
                let output = chain.apply_point(source).unwrap();
                assert_eq!(
                    source_query.contains(source).unwrap(),
                    output_query.contains(output).unwrap(),
                    "pullback mismatch at {source:?} -> {output:?}"
                );
            }
        }
    }

    let index = two_leaf_index();
    assert_eq!(
        index.query_output_chart(&output_query, &chain).unwrap(),
        index.query(&source_query)
    );
}
