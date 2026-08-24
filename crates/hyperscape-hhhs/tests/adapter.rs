use futures::executor::block_on;
use futures::future::FutureExt;
use futures::stream::{Stream, StreamExt};
use futures_signals::signal_vec::{SignalVecExt, VecDiff};
use hhhs::{DagSnapshot, Digest, ReachIndex};
use hyperscape_hhhs::{
    decode_authored, encode_authored, AdapterError, DurableProject, ProjectId, StateRow,
    MAX_AUTHORED_PAYLOAD_BYTES, PAYLOAD_DOMAIN,
};
use hyperscape_protocol::{
    AssetDescriptor, AssetId, AuthoredCommand, AuthoredEnvelope, EntityId, MessageHeader,
    MessageId, PeerId, ProtocolVersion, WireTransform, CURRENT_PROTOCOL_VERSION,
};
use std::collections::BTreeSet;

fn project(value: u128) -> ProjectId {
    ProjectId::from_u128(value).unwrap()
}

fn asset_id(value: u128) -> AssetId {
    AssetId::from_u128(value).unwrap()
}

fn entity_id(value: u128) -> EntityId {
    EntityId::from_u128(value).unwrap()
}

fn envelope(sequence: u64, command: AuthoredCommand) -> AuthoredEnvelope {
    AuthoredEnvelope {
        header: MessageHeader {
            version: CURRENT_PROTOCOL_VERSION,
            message_id: MessageId::from_u128(0x1000 + u128::from(sequence)).unwrap(),
            sender: PeerId::from_u128(0x2000).unwrap(),
            sequence,
        },
        command,
    }
}

fn transform(x: f64) -> WireTransform {
    WireTransform {
        translation: [x, x + 1.0, x + 2.0],
        rotation_wxyz: [1.0, 0.0, 0.0, 0.0],
        scale: [1.0, 1.0, 1.0],
    }
}

fn set(sequence: u64, entity: EntityId, x: f64) -> AuthoredEnvelope {
    envelope(
        sequence,
        AuthoredCommand::SetEntityTransform {
            entity,
            transform: transform(x),
        },
    )
}

fn remove(sequence: u64, entity: EntityId) -> AuthoredEnvelope {
    envelope(sequence, AuthoredCommand::RemoveEntity { entity })
}

fn upsert(sequence: u64, asset: AssetId, uri: &str) -> AuthoredEnvelope {
    envelope(
        sequence,
        AuthoredCommand::UpsertAsset {
            asset: AssetDescriptor {
                id: asset,
                uri: uri.into(),
                media_type: Some("model/gltf-binary".into()),
                content_digest: Some([7; 32]),
            },
        },
    )
}

fn drain<S>(stream: &mut S) -> Vec<S::Item>
where
    S: Stream + Unpin,
{
    let mut items = Vec::new();
    while let Some(item) = stream.next().now_or_never().flatten() {
        items.push(item);
    }
    items
}

#[test]
fn frozen_payload_round_trips_deterministically() {
    let project = project(0xaaaa);
    let authored = upsert(3, asset_id(0xbbbb), "models/cube.glb");
    let first = encode_authored(project, &authored).unwrap();
    let second = encode_authored(project, &authored).unwrap();

    assert_eq!(first, second);
    assert_eq!(&first[..PAYLOAD_DOMAIN.len()], PAYLOAD_DOMAIN);
    assert_eq!(decode_authored(project, &first).unwrap(), authored);
    assert_eq!(
        Digest::of(&first).to_hex(),
        "ea8e5200857a3e116a20841028d026df030d3afae4ba87feead460b3adb3af90",
        "this golden digest freezes the complete v0.1 payload encoding"
    );
}

#[test]
fn wrong_project_domain_and_versions_are_rejected() {
    let expected = project(1);
    let bytes = encode_authored(expected, &set(1, entity_id(9), 2.0)).unwrap();

    assert!(matches!(
        decode_authored(project(2), &bytes),
        Err(AdapterError::WrongProject { .. })
    ));

    let mut wrong_domain = bytes.clone();
    wrong_domain[0] ^= 0xff;
    assert!(matches!(
        decode_authored(expected, &wrong_domain),
        Err(AdapterError::WrongDomain)
    ));

    let mut wrong_payload_version = bytes.clone();
    wrong_payload_version[PAYLOAD_DOMAIN.len()..PAYLOAD_DOMAIN.len() + 2]
        .copy_from_slice(&99u16.to_le_bytes());
    assert!(matches!(
        decode_authored(expected, &wrong_payload_version),
        Err(AdapterError::WrongPayloadVersion(ProtocolVersion {
            major: 99,
            minor: 1
        }))
    ));

    let mut wrong_protocol_version = bytes;
    let protocol_major = PAYLOAD_DOMAIN.len() + 4;
    wrong_protocol_version[protocol_major..protocol_major + 2]
        .copy_from_slice(&99u16.to_le_bytes());
    assert!(matches!(
        decode_authored(expected, &wrong_protocol_version),
        Err(AdapterError::WrongProtocolVersion(ProtocolVersion {
            major: 99,
            minor: 1
        }))
    ));
}

#[test]
fn oversized_payloads_are_rejected_before_binary_decode() {
    let project = project(13);
    let oversized = vec![0; MAX_AUTHORED_PAYLOAD_BYTES + 1];
    assert!(matches!(
        decode_authored(project, &oversized),
        Err(AdapterError::PayloadTooLarge { actual, max })
            if actual == MAX_AUTHORED_PAYLOAD_BYTES + 1 && max == MAX_AUTHORED_PAYLOAD_BYTES
    ));

    let authored = upsert(1, asset_id(13), &"x".repeat(MAX_AUTHORED_PAYLOAD_BYTES));
    assert!(matches!(
        encode_authored(project, &authored),
        Err(AdapterError::PayloadTooLarge { .. })
    ));
}

#[test]
fn aggregate_state_may_exceed_the_per_message_limit() {
    let project = project(14);
    let mut host = DurableProject::new(project).unwrap();
    let uri_a = format!("a{}", "x".repeat(600_000));
    let uri_b = format!("b{}", "x".repeat(600_000));

    block_on(host.admit(&upsert(1, asset_id(14), &uri_a))).unwrap();
    block_on(host.admit(&upsert(2, asset_id(15), &uri_b))).unwrap();

    let state = host.state().unwrap();
    assert_eq!(state.assets.len(), 2);
    assert_ne!(state.state_root, [0; 32]);
}

#[test]
fn persistence_failure_publishes_nothing() {
    let mut host = DurableProject::new(project(3)).unwrap();
    let before_log = host.durability().bytes().to_vec();
    host.durability_mut().fail_next_persist();

    let error = block_on(host.admit(&set(1, entity_id(1), 4.0))).unwrap_err();
    assert!(matches!(error, AdapterError::Repair(_)));
    assert_eq!(host.history_len(), 0);
    assert!(host.state().unwrap().entity_transforms.is_empty());
    assert_eq!(host.durability().bytes(), before_log);
}

#[test]
fn recovery_preserves_history_and_materialized_roots() {
    let project = project(4);
    let mut original = DurableProject::new(project).unwrap();
    block_on(original.admit(&upsert(1, asset_id(1), "scene.glb"))).unwrap();
    block_on(original.admit(&set(2, entity_id(2), 7.0))).unwrap();
    let before = original.state().unwrap();
    let log = original.durability().bytes().to_vec();

    let recovered = DurableProject::recover(project, log).unwrap();
    let after = recovered.state().unwrap();

    assert_eq!(after, before);
    assert_eq!(recovered.history_len(), 2);
    assert_ne!(after.history_root, [0; 32]);
    assert_ne!(after.state_root, [0; 32]);
}

#[test]
fn causal_successor_replaces_its_observed_predecessor() {
    let mut host = DurableProject::new(project(5)).unwrap();
    let entity = entity_id(3);
    let first = block_on(host.admit(&set(1, entity, 1.0))).unwrap();
    let second = block_on(host.admit(&set(2, entity, 9.0))).unwrap();

    assert!(second.entry().header.prevs.contains(&first.entry_hash()));
    assert_eq!(
        host.state().unwrap().entity_transforms[&entity],
        transform(9.0)
    );
}

#[test]
fn concurrent_writes_converge_independently_of_arrival_order() {
    let project = project(6);
    let entity = entity_id(4);
    let mut branch_a = DurableProject::new(project).unwrap();
    let mut branch_b = DurableProject::new(project).unwrap();
    let record_a = block_on(branch_a.admit(&set(1, entity, 11.0))).unwrap();
    let record_b = block_on(branch_b.admit(&set(2, entity, 22.0))).unwrap();
    assert!(record_a.entry().header.prevs.is_empty());
    assert!(record_b.entry().header.prevs.is_empty());

    let mut forward = DurableProject::new(project).unwrap();
    block_on(forward.apply_records(&[record_a.clone(), record_b.clone()])).unwrap();
    let mut reverse = DurableProject::new(project).unwrap();
    block_on(reverse.apply_records(&[record_b.clone(), record_a.clone()])).unwrap();

    let state_forward = forward.state().unwrap();
    let state_reverse = reverse.state().unwrap();
    assert_eq!(state_forward, state_reverse);
    let expected = if record_a.entry_hash().as_bytes() > record_b.entry_hash().as_bytes() {
        transform(11.0)
    } else {
        transform(22.0)
    };
    assert_eq!(state_forward.entity_transforms[&entity], expected);
}

#[test]
fn lazy_production_materialization_matches_reference_reach_index() {
    let project = project(12);
    let entity = entity_id(9);
    let mut chain = DurableProject::new(project).unwrap();
    let parent = block_on(chain.admit(&set(1, entity, 1.0))).unwrap();
    let child = block_on(chain.admit(&set(2, entity, 2.0))).unwrap();
    let mut fork = DurableProject::new(project).unwrap();
    let concurrent = block_on(fork.admit(&set(3, entity, 3.0))).unwrap();

    let snapshot = DagSnapshot::from_entries([
        parent.entry().clone(),
        child.entry().clone(),
        concurrent.entry().clone(),
    ]);
    let candidates: BTreeSet<_> = [
        parent.entry_hash(),
        child.entry_hash(),
        concurrent.entry_hash(),
    ]
    .into_iter()
    .collect();
    let reference = hhhs::register::resolve(&candidates, &ReachIndex::new(&snapshot)).unwrap();
    let expected = if reference == child.entry_hash() {
        transform(2.0)
    } else if reference == concurrent.entry_hash() {
        transform(3.0)
    } else {
        panic!("the causal parent cannot survive its child")
    };

    let mut materialized = DurableProject::new(project).unwrap();
    block_on(materialized.apply_records(&[concurrent, child, parent])).unwrap();
    assert_eq!(
        materialized.state().unwrap().entity_transforms[&entity],
        expected
    );
}

#[test]
fn concurrent_set_remove_race_uses_the_same_register_rule() {
    let project = project(7);
    let entity = entity_id(5);
    let mut set_branch = DurableProject::new(project).unwrap();
    let mut remove_branch = DurableProject::new(project).unwrap();
    let set_record = block_on(set_branch.admit(&set(1, entity, 3.0))).unwrap();
    let remove_record = block_on(remove_branch.admit(&remove(2, entity))).unwrap();

    let mut merged = DurableProject::new(project).unwrap();
    block_on(merged.apply_records(&[set_record.clone(), remove_record.clone()])).unwrap();
    let present = merged
        .state()
        .unwrap()
        .entity_transforms
        .contains_key(&entity);
    let set_wins = set_record.entry_hash().as_bytes() > remove_record.entry_hash().as_bytes();
    assert_eq!(present, set_wins);
}

#[test]
fn apply_records_lifts_reversed_causal_batches_and_retries_missing_children() {
    let project = project(9);
    let entity = entity_id(7);
    let mut origin = DurableProject::new(project).unwrap();
    let parent = block_on(origin.admit(&set(1, entity, 1.0))).unwrap();
    let child = block_on(origin.admit(&set(2, entity, 2.0))).unwrap();
    assert!(child.entry().header.prevs.contains(&parent.entry_hash()));

    let mut target = DurableProject::new(project).unwrap();
    let incomplete = block_on(target.apply_records(std::slice::from_ref(&child))).unwrap();
    assert!(incomplete.admitted.is_empty());
    assert_eq!(incomplete.deferred, vec![child.entry_hash()]);
    assert_eq!(target.history_len(), 0);

    let lifted = block_on(target.apply_records(&[child, parent])).unwrap();
    assert_eq!(lifted.admitted.len(), 2);
    assert!(lifted.deferred.is_empty());
    assert_eq!(lifted.lifted, 2);
    assert_eq!(
        target.state().unwrap().entity_transforms[&entity],
        transform(2.0)
    );
}

#[test]
fn foreign_project_records_are_refused_without_publication() {
    let mut foreign = DurableProject::new(project(10)).unwrap();
    let record = block_on(foreign.admit(&set(1, entity_id(8), 1.0))).unwrap();
    let mut local = DurableProject::new(project(11)).unwrap();
    let before_log = local.durability().bytes().to_vec();

    let report = block_on(local.apply_records(&[record])).unwrap();
    assert_eq!(report.refused.len(), 1);
    assert!(report.admitted.is_empty());
    assert_eq!(local.history_len(), 0);
    assert_eq!(local.durability().bytes(), before_log);
}

#[test]
fn reactive_views_start_current_coalesce_growth_and_retract_removed_rows() {
    let project = project(8);
    let entity = entity_id(6);
    let asset = asset_id(6);
    let mut host = DurableProject::new(project).unwrap();
    let mut stream = Box::pin(host.state_stream());
    assert!(drain(&mut stream).is_empty());

    block_on(host.admit(&upsert(1, asset, "coalesced.glb"))).unwrap();
    block_on(host.admit(&set(2, entity, 5.0))).unwrap();
    let coalesced = drain(&mut stream);
    assert_eq!(coalesced.len(), 1);
    assert_eq!(coalesced[0].added.len(), 2);
    assert!(coalesced[0].retracted.is_empty());

    let mut late = Box::pin(host.state_stream());
    let initial = drain(&mut late);
    assert_eq!(initial.len(), 1);
    assert_eq!(initial[0].added.len(), 2);

    block_on(host.admit(&remove(3, entity))).unwrap();
    let removed = drain(&mut stream);
    assert_eq!(removed.len(), 1);
    assert!(removed[0].added.is_empty());
    assert!(matches!(
        removed[0].retracted.as_slice(),
        [StateRow::EntityTransform { entity: retracted, .. }] if *retracted == entity
    ));

    let signal = host.state_signal_vec();
    let mut diffs = Box::pin(signal.to_stream());
    let initial_diffs = drain(&mut diffs);
    assert!(matches!(
        initial_diffs.as_slice(),
        [VecDiff::Replace { values }] if values.len() == 1
    ));

    block_on(host.admit(&set(4, entity, 12.0))).unwrap();
    assert!(drain(&mut diffs)
        .iter()
        .any(|diff| matches!(diff, VecDiff::InsertAt { .. })));
}
