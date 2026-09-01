use futures::executor::block_on;
use futures::future::FutureExt;
use futures::stream::{Stream, StreamExt};
use futures_signals::signal_vec::{SignalVecExt, VecDiff};
use hhhs::{DagSnapshot, Digest, ReachIndex};
use hhhs_store::decode_storage_transaction_log;
use hyperscape_hhhs::{
    decode_authored, encode_authored, AdapterError, ApplyRecordWithCheckpointReport,
    AuthoredRecordFrame, DurableProject, MemoryDurability, ProjectArchive, ProjectId,
    ProjectionKey, RecordFrameError, RecordRefusal, StateRow, AUTHORED_RECORD_FRAME_LANE,
    MAX_AUTHORED_PAYLOAD_BYTES, MAX_PROJECT_ARCHIVE_RECORDS, PAYLOAD_DOMAIN,
    PROJECT_ARCHIVE_DOMAIN,
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

fn resign_archive(bytes: &mut [u8]) {
    let checksum_offset = bytes.len() - 32;
    let checksum = Digest::of(&bytes[..checksum_offset]);
    bytes[checksum_offset..].copy_from_slice(checksum.as_bytes());
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
fn frozen_asset_options_round_trip_when_absent() {
    let project = project(0xaaab);
    let authored = envelope(
        4,
        AuthoredCommand::UpsertAsset {
            asset: AssetDescriptor {
                id: asset_id(0xbbbc),
                uri: "models/empty-options.glb".into(),
                media_type: None,
                content_digest: None,
            },
        },
    );

    let bytes = encode_authored(project, &authored).unwrap();
    assert_eq!(decode_authored(project, &bytes).unwrap(), authored);
}

#[test]
fn authored_record_frame_round_trips_without_granting_transport_authority() {
    let project_id = project(0xaacc);
    let authored = upsert(5, asset_id(0xbbcc), "models/carrier.glb");
    let mut source = DurableProject::new(project_id).unwrap();
    let record = block_on(source.admit(&authored)).unwrap();
    let frame = AuthoredRecordFrame::new(project_id, record.clone()).unwrap();

    let json = frame.encode_json().unwrap();
    assert_eq!(json, frame.encode_json().unwrap());
    assert!(json.contains(&format!(r#""lane":"{AUTHORED_RECORD_FRAME_LANE}""#)));
    assert!(!json.contains('='), "carrier base64url must be unpadded");
    let decoded = AuthoredRecordFrame::decode_json(&json).unwrap();
    assert_eq!(decoded.project_id(), project_id);
    assert_eq!(decoded.record(), &record);
    assert_eq!(
        decode_authored(project_id, &decoded.into_record().entry().payload).unwrap(),
        authored
    );
}

#[test]
fn authored_record_frame_rechecks_schema_project_and_record_payload() {
    let project_id = project(0xaacd);
    let mut source = DurableProject::new(project_id).unwrap();
    let record = block_on(source.admit(&set(6, entity_id(0xbbcd), 3.0))).unwrap();
    assert!(matches!(
        AuthoredRecordFrame::new(project(0xaace), record.clone()),
        Err(RecordFrameError::Adapter(AdapterError::WrongProject { .. }))
    ));

    let json = AuthoredRecordFrame::new(project_id, record)
        .unwrap()
        .encode_json()
        .unwrap();
    let mut wire: serde_json::Value = serde_json::from_str(&json).unwrap();
    wire["lane"] = serde_json::json!("authored");
    assert!(matches!(
        AuthoredRecordFrame::decode_json(&serde_json::to_string(&wire).unwrap()),
        Err(RecordFrameError::WrongLane(_))
    ));

    wire["lane"] = serde_json::json!(AUTHORED_RECORD_FRAME_LANE);
    wire["version"]["major"] = serde_json::json!(99);
    assert!(matches!(
        AuthoredRecordFrame::decode_json(&serde_json::to_string(&wire).unwrap()),
        Err(RecordFrameError::WrongVersion(ProtocolVersion {
            major: 99,
            minor: 1,
        }))
    ));

    wire["version"]["major"] = serde_json::json!(0);
    wire["unexpected"] = serde_json::json!(true);
    assert!(matches!(
        AuthoredRecordFrame::decode_json(&serde_json::to_string(&wire).unwrap()),
        Err(RecordFrameError::Json(_))
    ));
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
fn authored_entry_and_local_checkpoint_commit_and_restart_atomically() {
    let project = project(0x3003);
    let entity = entity_id(0x3003);
    let key = ProjectionKey::new("hyperscope/source-cursor", 1).unwrap();
    let checkpoint_bytes = b"browser revision 41".to_vec();
    let mut host = DurableProject::new(project).unwrap();
    let before_log = host.durability().bytes().to_vec();
    host.durability_mut().fail_next_persist();

    let error = block_on(host.admit_with_projection_checkpoint(
        &set(41, entity, 4.0),
        key.clone(),
        checkpoint_bytes.clone(),
    ))
    .unwrap_err();
    assert!(matches!(error, AdapterError::Repair(_)));
    assert_eq!(host.history_len(), 0);
    assert!(host.projection_checkpoint(&key).is_none());
    assert_eq!(host.durability().bytes(), before_log);

    let record = block_on(host.admit_with_projection_checkpoint(
        &set(41, entity, 4.0),
        key.clone(),
        checkpoint_bytes.clone(),
    ))
    .unwrap();
    let live = host.projection_checkpoint(&key).unwrap();
    let live_state = host.state().unwrap();
    assert_eq!(live.bytes(), checkpoint_bytes);
    assert_eq!(live.at.len(), 1);
    assert!(live.at.contains(&record.entry_hash()));
    assert_eq!(live.history_root.as_bytes(), &live_state.history_root);
    assert!(live.is_intact());

    let durable_bytes = host.durability().bytes().to_vec();
    let restarted = DurableProject::recover(project, durable_bytes).unwrap();
    assert_eq!(restarted.state().unwrap(), live_state);
    assert_eq!(restarted.projection_checkpoint(&key), Some(live));

    let portable = host.export_archive().unwrap();
    let imported = block_on(DurableProject::import_archive(&portable)).unwrap();
    assert_eq!(imported.state().unwrap(), live_state);
    assert!(imported.projection_checkpoint(&key).is_none());
}

#[test]
fn checkpoint_only_transition_is_durable_atomic_and_history_neutral() {
    let project_id = project(0x3004);
    let key = ProjectionKey::new("hyperscope/import-cursor", 1).unwrap();
    let checkpoint_bytes = 7_u64.to_le_bytes().to_vec();
    let mut host = DurableProject::new(project_id).unwrap();
    block_on(host.admit(&set(1, entity_id(0x3004), 4.0))).unwrap();
    let before_state = host.state().unwrap();
    let before_history_len = host.history_len();
    let before_failure_log = host.durability().bytes().to_vec();
    host.durability_mut().fail_next_persist();

    let error = block_on(host.persist_projection_checkpoint(key.clone(), checkpoint_bytes.clone()))
        .unwrap_err();
    assert!(matches!(error, AdapterError::Repair(_)));
    assert!(host.projection_checkpoint(&key).is_none());
    assert_eq!(host.history_len(), before_history_len);
    assert_eq!(host.state().unwrap(), before_state);
    assert_eq!(host.durability().bytes(), before_failure_log);

    let checkpoint =
        block_on(host.persist_projection_checkpoint(key.clone(), checkpoint_bytes.clone()))
            .unwrap();
    assert!(checkpoint.is_intact());
    assert_eq!(checkpoint.bytes(), checkpoint_bytes);
    assert_eq!(
        checkpoint.history_root.as_bytes(),
        &before_state.history_root
    );
    assert_eq!(host.projection_checkpoint(&key), Some(checkpoint.clone()));
    assert_eq!(host.history_len(), before_history_len);
    assert_eq!(host.state().unwrap(), before_state);

    let durable_log = host.durability().bytes().to_vec();
    let exact =
        block_on(host.persist_projection_checkpoint(key.clone(), checkpoint_bytes)).unwrap();
    assert_eq!(exact, checkpoint);
    assert_eq!(host.durability().bytes(), durable_log);

    let recovered = DurableProject::recover(project_id, durable_log).unwrap();
    assert_eq!(recovered.projection_checkpoint(&key), Some(checkpoint));
    assert_eq!(recovered.history_len(), before_history_len);
    assert_eq!(recovered.state().unwrap(), before_state);

    let archive = host.export_archive().unwrap();
    let imported = block_on(DurableProject::import_archive(&archive)).unwrap();
    assert!(imported.projection_checkpoint(&key).is_none());
    assert_eq!(imported.state().unwrap(), before_state);
}

#[test]
fn carrier_record_and_local_cursor_commit_atomically_and_retry_safely() {
    let project_id = project(0x3005);
    let key = ProjectionKey::new("hyperscope/carrier-cursor", 1).unwrap();
    let first_envelope = set(1, entity_id(0x3005), 5.0);
    let second_envelope = set(2, entity_id(0x3006), 6.0);
    let mut source = DurableProject::new(project_id).unwrap();
    let first = block_on(source.admit(&first_envelope)).unwrap();
    let second = block_on(source.admit(&second_envelope)).unwrap();
    let mut target = DurableProject::new(project_id).unwrap();
    let empty_log = target.durability().bytes().to_vec();

    let deferred = block_on(target.apply_record_with_projection_checkpoint(
        second.clone(),
        key.clone(),
        1_u64.to_le_bytes().to_vec(),
    ))
    .unwrap();
    assert!(matches!(
        deferred,
        ApplyRecordWithCheckpointReport::Deferred {
            entry,
            ref missing,
        } if entry == second.entry_hash() && missing == &[first.entry_hash()]
    ));
    assert_eq!(target.history_len(), 0);
    assert!(target.projection_checkpoint(&key).is_none());
    assert_eq!(target.durability().bytes(), empty_log);

    target.durability_mut().fail_next_persist();
    let failed = block_on(target.apply_record_with_projection_checkpoint(
        first.clone(),
        key.clone(),
        0_u64.to_le_bytes().to_vec(),
    ));
    assert!(matches!(failed, Err(AdapterError::Repair(_))));
    assert_eq!(target.history_len(), 0);
    assert!(target.projection_checkpoint(&key).is_none());
    assert_eq!(target.durability().bytes(), empty_log);

    let applied = block_on(target.apply_record_with_projection_checkpoint(
        first.clone(),
        key.clone(),
        0_u64.to_le_bytes().to_vec(),
    ))
    .unwrap();
    assert_eq!(
        applied,
        ApplyRecordWithCheckpointReport::Applied {
            entry: first.entry_hash()
        }
    );
    let first_log = target.durability().bytes().to_vec();
    let checkpoint = target.projection_checkpoint(&key).unwrap();
    assert_eq!(checkpoint.bytes(), 0_u64.to_le_bytes());
    assert_eq!(
        checkpoint.history_root.as_bytes(),
        &target.state().unwrap().history_root
    );

    let duplicate = block_on(target.apply_record_with_projection_checkpoint(
        first.clone(),
        key.clone(),
        99_u64.to_le_bytes().to_vec(),
    ))
    .unwrap();
    assert_eq!(
        duplicate,
        ApplyRecordWithCheckpointReport::AlreadyPresent {
            entry: first.entry_hash()
        }
    );
    assert_eq!(target.durability().bytes(), first_log);
    assert_eq!(target.projection_checkpoint(&key).unwrap(), checkpoint);

    let applied = block_on(target.apply_record_with_projection_checkpoint(
        second.clone(),
        key.clone(),
        1_u64.to_le_bytes().to_vec(),
    ))
    .unwrap();
    assert_eq!(
        applied,
        ApplyRecordWithCheckpointReport::Applied {
            entry: second.entry_hash()
        }
    );
    assert_eq!(target.history_len(), 2);
    assert_eq!(
        target.projection_checkpoint(&key).unwrap().bytes(),
        1_u64.to_le_bytes()
    );
    let recovered =
        DurableProject::recover(project_id, target.durability().bytes().to_vec()).unwrap();
    assert_eq!(recovered.state().unwrap(), target.state().unwrap());
    assert_eq!(
        recovered.projection_checkpoint(&key),
        target.projection_checkpoint(&key)
    );

    let mut foreign = DurableProject::new(project(0x3007)).unwrap();
    let foreign_record = block_on(foreign.admit(&set(3, entity_id(0x3007), 7.0))).unwrap();
    let refused = block_on(target.apply_record_with_projection_checkpoint(
        foreign_record.clone(),
        key,
        2_u64.to_le_bytes().to_vec(),
    ))
    .unwrap();
    assert_eq!(
        refused,
        ApplyRecordWithCheckpointReport::Refused {
            entry: foreign_record.entry_hash(),
            reason: RecordRefusal::Unauthorized,
        }
    );
    assert_eq!(target.history_len(), 2);
}

#[test]
fn recovery_preserves_history_and_materialized_roots() {
    let project = project(4);
    let mut original = DurableProject::new(project).unwrap();
    let first = upsert(1, asset_id(1), "scene.glb");
    let second = set(2, entity_id(2), 7.0);
    block_on(original.admit(&first)).unwrap();
    block_on(original.admit(&second)).unwrap();
    let before = original.state().unwrap();
    let log = original.durability().bytes().to_vec();
    assert_eq!(
        original.authored_history().unwrap(),
        vec![first.clone(), second.clone()]
    );

    let recovered = DurableProject::recover(project, log).unwrap();
    let after = recovered.state().unwrap();

    assert_eq!(after, before);
    assert_eq!(recovered.history_len(), 2);
    assert_eq!(recovered.authored_history().unwrap(), vec![first, second]);
    assert_ne!(after.history_root, [0; 32]);
    assert_ne!(after.state_root, [0; 32]);
}

#[test]
fn portable_archive_round_trips_deterministically_through_admission() {
    let project = project(0x4004);
    let mut original = DurableProject::new(project).unwrap();
    block_on(original.admit(&upsert(1, asset_id(1), "scene.glb"))).unwrap();
    block_on(original.admit(&set(2, entity_id(2), 7.0))).unwrap();
    block_on(original.admit(&set(3, entity_id(3), 11.0))).unwrap();

    let first = original.export_archive().unwrap();
    let second = original.export_archive().unwrap();
    let decoded = ProjectArchive::decode(&first).unwrap();
    let imported = block_on(DurableProject::import_archive(&first)).unwrap();

    assert_eq!(first, second);
    assert_eq!(decoded.project_id(), project);
    assert_eq!(decoded.record_count(), 3);
    assert_eq!(decoded.encode().unwrap(), first);
    assert_eq!(imported.state().unwrap(), original.state().unwrap());
    assert_eq!(imported.history_len(), original.history_len());
    assert_eq!(imported.export_archive().unwrap(), first);
    assert_ne!(imported.durability().bytes(), first);
    assert_eq!(
        Digest::of(&first).to_hex(),
        "282e486c7469a765455e8e41872af98176fcc8d2396be6de82bfb4aaf4af0c52",
        "this golden digest freezes the complete v0.1 archive encoding"
    );
}

#[test]
fn portable_archive_rejects_corruption_wrong_project_and_false_roots() {
    let project_id = project(0x4005);
    let mut original = DurableProject::new(project_id).unwrap();
    block_on(original.admit(&set(1, entity_id(4), 5.0))).unwrap();
    let archive = original.export_archive().unwrap();

    let mut corrupted = archive.clone();
    corrupted[PROJECT_ARCHIVE_DOMAIN.len() + 20] ^= 0x80;
    assert!(matches!(
        ProjectArchive::decode(&corrupted),
        Err(AdapterError::ArchiveChecksumMismatch)
    ));

    let mut wrong_project = archive.clone();
    let project_offset = PROJECT_ARCHIVE_DOMAIN.len() + 4;
    wrong_project[project_offset..project_offset + 16]
        .copy_from_slice(project(0x4999).as_uuid().as_bytes());
    resign_archive(&mut wrong_project);
    assert!(matches!(
        ProjectArchive::decode(&wrong_project),
        Err(AdapterError::WrongProject { .. })
    ));

    let mut wrong_history_root = archive.clone();
    let history_root_offset = PROJECT_ARCHIVE_DOMAIN.len() + 4 + 16 + 8;
    wrong_history_root[history_root_offset] ^= 1;
    resign_archive(&mut wrong_history_root);
    assert!(matches!(
        block_on(DurableProject::import_archive(&wrong_history_root)),
        Err(AdapterError::ArchiveHistoryRootMismatch)
    ));

    let mut wrong_state_root = archive;
    let state_root_offset = history_root_offset + 32;
    wrong_state_root[state_root_offset] ^= 1;
    resign_archive(&mut wrong_state_root);
    assert!(matches!(
        block_on(DurableProject::import_archive(&wrong_state_root)),
        Err(AdapterError::ArchiveStateRootMismatch)
    ));
}

#[test]
fn portable_archive_enforces_version_and_declared_record_bounds() {
    let original = DurableProject::new(project(0x4006)).unwrap();
    let archive = original.export_archive().unwrap();

    let mut wrong_version = archive.clone();
    wrong_version[PROJECT_ARCHIVE_DOMAIN.len()..PROJECT_ARCHIVE_DOMAIN.len() + 2]
        .copy_from_slice(&99_u16.to_le_bytes());
    resign_archive(&mut wrong_version);
    assert!(matches!(
        ProjectArchive::decode(&wrong_version),
        Err(AdapterError::WrongArchiveVersion(ProtocolVersion {
            major: 99,
            minor: 1
        }))
    ));

    let mut too_many = archive;
    let count_offset = PROJECT_ARCHIVE_DOMAIN.len() + 4 + 16;
    too_many[count_offset..count_offset + 8]
        .copy_from_slice(&((MAX_PROJECT_ARCHIVE_RECORDS as u64) + 1).to_le_bytes());
    resign_archive(&mut too_many);
    assert!(matches!(
        ProjectArchive::decode(&too_many),
        Err(AdapterError::TooManyArchiveRecords { actual, max })
            if actual == MAX_PROJECT_ARCHIVE_RECORDS + 1 && max == MAX_PROJECT_ARCHIVE_RECORDS
    ));
}

#[test]
fn archive_application_resumes_a_prefix_and_exact_retries_are_write_free() {
    let project = project(0x4007);
    let entity = entity_id(0x4007);
    let mut origin = DurableProject::new(project).unwrap();
    let parent = block_on(origin.admit(&set(1, entity, 1.0))).unwrap();
    block_on(origin.admit(&set(2, entity, 2.0))).unwrap();
    let archive = ProjectArchive::decode(&origin.export_archive().unwrap()).unwrap();

    let mut target = DurableProject::new(project).unwrap();
    block_on(target.apply_records(std::slice::from_ref(&parent))).unwrap();
    let resumed = block_on(target.apply_archive(&archive)).unwrap();
    assert_eq!(resumed.lifted, 1);
    assert_eq!(target.state().unwrap(), origin.state().unwrap());

    let durable_before_retry = target.durability().bytes().to_vec();
    let retried = block_on(target.apply_archive(&archive)).unwrap();
    assert_eq!(retried.lifted, 0);
    assert_eq!(target.durability().bytes(), durable_before_retry);
}

#[test]
fn archive_application_rejects_local_divergence_and_foreign_projects_before_writing() {
    let project_id = project(0x4008);
    let mut origin = DurableProject::new(project_id).unwrap();
    block_on(origin.admit(&set(1, entity_id(1), 1.0))).unwrap();
    let archive = ProjectArchive::decode(&origin.export_archive().unwrap()).unwrap();

    let mut divergent = DurableProject::new(project_id).unwrap();
    block_on(divergent.admit(&set(2, entity_id(2), 2.0))).unwrap();
    let divergent_before = divergent.durability().bytes().to_vec();
    assert!(matches!(
        block_on(divergent.apply_archive(&archive)),
        Err(AdapterError::ArchiveLocalDivergence { local_only: 1 })
    ));
    assert_eq!(divergent.durability().bytes(), divergent_before);

    let mut foreign = DurableProject::new(project(0x4998)).unwrap();
    let foreign_before = foreign.durability().bytes().to_vec();
    assert!(matches!(
        block_on(foreign.apply_archive(&archive)),
        Err(AdapterError::WrongArchiveProject { .. })
    ));
    assert_eq!(foreign.durability().bytes(), foreign_before);
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
fn trusted_transaction_recovery_keeps_the_sink_and_replay_in_lockstep() {
    let project = project(13);
    let entity = entity_id(10);
    let mut original = DurableProject::new(project).unwrap();
    block_on(original.admit(&set(1, entity, 4.0))).unwrap();
    block_on(original.admit(&set(2, entity, 8.0))).unwrap();
    let original_state = original.state().unwrap();
    let durable_bytes = original.durability().bytes().to_vec();
    let transactions = decode_storage_transaction_log(&durable_bytes).unwrap();
    let durability = MemoryDurability::from_bytes(durable_bytes).unwrap();

    let mut recovered =
        DurableProject::recover_trusted_transactions(project, durability, transactions).unwrap();
    assert_eq!(recovered.state().unwrap(), original_state);

    block_on(recovered.admit(&remove(3, entity))).unwrap();
    let restarted =
        DurableProject::recover(project, recovered.durability().bytes().to_vec()).unwrap();
    assert_eq!(restarted.history_len(), 3);
    assert!(!restarted
        .state()
        .unwrap()
        .entity_transforms
        .contains_key(&entity));
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
