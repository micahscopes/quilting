use futures::executor::block_on;
use hhhs::Digest;
use hhhs_replica::AsyncTransactionSink;
use hhhs_store::{StorageRecoveryState, StorageTransaction};
use hyperscape_hhhs::{AdapterError, DurableProject, MemoryDurability, ProjectId};
use hyperscape_protocol::{
    AssetDescriptor, AssetId, AuthoredCommand, AuthoredEnvelope, EntityId, EphemeralPresence,
    LocalPeerEnvelope, MessageHeader, MessageId, PeerId, PresenceEnvelope, ProtocolVersion,
    WireTransform, CURRENT_PROTOCOL_VERSION, LEGACY_PROTOCOL_VERSION,
};
use hyperscope_app::{
    AppStore, AuthoredRevision, CommitDisposition, LocalPeerDisposition, LocalPeerIngress,
};
use hyperscope_hhhs_shadow::{
    AuthoredHhhsShadow, AuthoredShadowCheckpoint, AuthoredShadowError, AuthoredShadowInitError,
    AuthoredShadowObservation, DurableAuthoredCoordinator, DurableAuthoredDispatchError,
    DurableAuthoredInitError, DurableAuthoredObservation, DurableAuthoredRestoreError,
    DurableAuthoredSession, DurableAuthoredSessionInitError, DurableCarrierError,
    DurableCarrierObservation, DurableLocalPeerError, AUTHORED_SHADOW_CHECKPOINT_DOMAIN,
};

fn project(value: u128) -> ProjectId {
    ProjectId::from_u128(value).unwrap()
}

fn resign_checkpoint(bytes: &mut [u8]) {
    let checksum_offset = bytes.len() - 32;
    let checksum = Digest::of(&bytes[..checksum_offset]);
    bytes[checksum_offset..].copy_from_slice(checksum.as_bytes());
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

fn asset(sequence: u64, id: u128) -> AuthoredEnvelope {
    envelope(
        sequence,
        AuthoredCommand::UpsertAsset {
            asset: AssetDescriptor {
                id: AssetId::from_u128(id).unwrap(),
                uri: format!("scene-{id}.glb"),
                media_type: Some("model/gltf-binary".into()),
                content_digest: None,
            },
        },
    )
}

fn transform(sequence: u64, entity: u128, x: f64) -> AuthoredEnvelope {
    envelope(
        sequence,
        AuthoredCommand::SetEntityTransform {
            entity: EntityId::from_u128(entity).unwrap(),
            transform: WireTransform {
                translation: [x, 0.0, 0.0],
                rotation_wxyz: [1.0, 0.0, 0.0, 0.0],
                scale: [1.0, 1.0, 1.0],
            },
        },
    )
}

fn remove(sequence: u64, entity: u128) -> AuthoredEnvelope {
    envelope(
        sequence,
        AuthoredCommand::RemoveEntity {
            entity: EntityId::from_u128(entity).unwrap(),
        },
    )
}

#[test]
fn applied_revision_matches_hhhs_in_exact_vector_order() {
    let store = AppStore::default();
    let mut shadow = AuthoredHhhsShadow::new(project(1)).unwrap();
    let revision = AuthoredRevision {
        projection_revision: 7,
        commands: vec![
            asset(1, 0xa),
            transform(2, 0xe, 1.0),
            remove(3, 0xe),
            transform(4, 0xe, 9.0),
        ],
    };
    let report = block_on(shadow.dispatch(&store, revision)).unwrap();
    assert_eq!(report.app.disposition, CommitDisposition::Applied);
    assert!(
        matches!(
            &report.shadow,
            Ok(AuthoredShadowObservation::Matched {
                projection_revision: 7,
                command_count: 4,
                history_len: 4,
                ..
            })
        ),
        "{:?}",
        report.shadow
    );
    assert_eq!(
        store.summary_snapshot().authored_projection_revision,
        Some(7)
    );
    let state = shadow.project_state().unwrap();
    assert_eq!(state.assets.len(), 1);
    assert_eq!(
        state.entity_transforms[&EntityId::from_u128(0xe).unwrap()].translation[0],
        9.0
    );
}

#[test]
fn stale_app_revision_is_a_zero_write_shadow_noop() {
    let store = AppStore::default();
    let mut shadow = AuthoredHhhsShadow::new(project(2)).unwrap();
    block_on(shadow.dispatch(
        &store,
        AuthoredRevision {
            projection_revision: 3,
            commands: vec![asset(1, 0xa)],
        },
    ))
    .unwrap();

    let report = block_on(shadow.dispatch(
        &store,
        AuthoredRevision {
            projection_revision: 3,
            commands: vec![asset(2, 0xb)],
        },
    ))
    .unwrap();
    assert_eq!(report.app.disposition, CommitDisposition::IgnoredStale);
    assert_eq!(
        report.shadow,
        Ok(AuthoredShadowObservation::IgnoredStale {
            projection_revision: 3,
            observed_projection_revision: Some(3),
            history_len: 1,
        })
    );
    assert_eq!(shadow.history_len(), 1);
    assert!(!shadow
        .project_state()
        .unwrap()
        .assets
        .contains_key(&AssetId::from_u128(0xb).unwrap()));
}

#[test]
fn archive_import_seeds_scene_state_without_laundering_a_projection_cursor() {
    let source_store = AppStore::default();
    let mut source = AuthoredHhhsShadow::new(project(0x21)).unwrap();
    block_on(source.dispatch(
        &source_store,
        AuthoredRevision {
            projection_revision: 41,
            commands: vec![asset(1, 0xa)],
        },
    ))
    .unwrap();
    let archive = source.export_archive().unwrap();

    let mut imported = block_on(AuthoredHhhsShadow::import_archive(&archive)).unwrap();
    let imported_store = AppStore::default();
    assert_eq!(imported.observed_projection_revision(), None);
    assert!(imported.align_store(&imported_store).unwrap().is_none());
    let report = block_on(imported.dispatch(
        &imported_store,
        AuthoredRevision {
            projection_revision: 1,
            commands: vec![transform(2, 0xe, 8.0)],
        },
    ))
    .unwrap();

    assert!(matches!(
        report.shadow,
        Ok(AuthoredShadowObservation::Matched {
            projection_revision: 1,
            history_len: 2,
            ..
        })
    ));
    let state = imported.project_state().unwrap();
    assert_eq!(state.assets.len(), 1);
    assert_eq!(
        state.entity_transforms[&EntityId::from_u128(0xe).unwrap()].translation[0],
        8.0
    );
}

#[test]
fn checkpoint_restart_aligns_a_fresh_store_and_continues_the_projection() {
    let source_store = AppStore::default();
    let mut source = AuthoredHhhsShadow::new(project(0x22)).unwrap();
    block_on(source.dispatch(
        &source_store,
        AuthoredRevision {
            projection_revision: 7,
            commands: vec![asset(1, 0xa)],
        },
    ))
    .unwrap();
    let checkpoint = source.checkpoint().unwrap();
    let archive = source.export_archive().unwrap();
    let project = block_on(DurableProject::import_archive(&archive)).unwrap();
    let mut recovered = AuthoredHhhsShadow::from_project_checkpoint(project, checkpoint).unwrap();
    let recovered_store = AppStore::default();

    let alignment = recovered.align_store(&recovered_store).unwrap().unwrap();
    assert_eq!(alignment.disposition, CommitDisposition::Applied);
    assert_eq!(
        recovered_store
            .summary_snapshot()
            .authored_projection_revision,
        Some(7)
    );
    assert!(recovered.align_store(&recovered_store).unwrap().is_none());
    assert_eq!(
        recovered.history_len(),
        1,
        "alignment is not authored history"
    );

    let continued = block_on(recovered.dispatch(
        &recovered_store,
        AuthoredRevision {
            projection_revision: 8,
            commands: vec![transform(2, 0xe, 3.0)],
        },
    ))
    .unwrap();
    assert!(matches!(
        continued.shadow,
        Ok(AuthoredShadowObservation::Matched {
            projection_revision: 8,
            history_len: 2,
            ..
        })
    ));
}

#[test]
fn source_checkpoint_codec_is_deterministic_and_rejects_corruption() {
    let source_store = AppStore::default();
    let mut source = AuthoredHhhsShadow::new(project(0x2201)).unwrap();
    let mut legacy_asset = asset(1, 0xa);
    legacy_asset.header.version = LEGACY_PROTOCOL_VERSION;
    let mut legacy_transform = transform(2, 0xe, 3.0);
    legacy_transform.header.version = LEGACY_PROTOCOL_VERSION;
    block_on(source.dispatch(
        &source_store,
        AuthoredRevision {
            projection_revision: 17,
            commands: vec![legacy_asset, legacy_transform],
        },
    ))
    .unwrap();
    let checkpoint = source.checkpoint().unwrap();
    let first = checkpoint.encode();
    let second = checkpoint.encode();

    assert_eq!(first, second);
    assert_eq!(
        AuthoredShadowCheckpoint::decode(&first).unwrap(),
        checkpoint
    );
    assert_eq!(
        Digest::of(&first).to_hex(),
        "85bd8ee953285a0004a544b4ee62a0327e5fee47f97db372a0a5556a5526fd7d",
        "this golden freezes the complete v0.1 source-checkpoint encoding"
    );

    let current_store = AppStore::default();
    let mut current_source = AuthoredHhhsShadow::new(project(0x2201)).unwrap();
    block_on(current_source.dispatch(
        &current_store,
        AuthoredRevision {
            projection_revision: 17,
            commands: vec![asset(1, 0xa), transform(2, 0xe, 3.0)],
        },
    ))
    .unwrap();
    assert_eq!(
        Digest::of(&current_source.checkpoint().unwrap().encode()).to_hex(),
        "13722147617f30ccf98f69c67df01737ad668ace31c011e29b913b2501e7bd25",
        "the unchanged checkpoint codec also binds a current protocol-v0.2 horizon"
    );

    let mut wrong_version = first.clone();
    wrong_version
        [AUTHORED_SHADOW_CHECKPOINT_DOMAIN.len()..AUTHORED_SHADOW_CHECKPOINT_DOMAIN.len() + 2]
        .copy_from_slice(&99_u16.to_le_bytes());
    resign_checkpoint(&mut wrong_version);
    assert!(matches!(
        AuthoredShadowCheckpoint::decode(&wrong_version),
        Err(AuthoredShadowInitError::WrongCheckpointVersion {
            major: 99,
            minor: 1
        })
    ));

    let mut malformed_option = first.clone();
    let option_offset = AUTHORED_SHADOW_CHECKPOINT_DOMAIN.len() + 4 + 16;
    malformed_option[option_offset] = 2;
    resign_checkpoint(&mut malformed_option);
    assert!(matches!(
        AuthoredShadowCheckpoint::decode(&malformed_option),
        Err(AuthoredShadowInitError::MalformedCheckpoint)
    ));

    let mut corrupt = first;
    corrupt[AUTHORED_SHADOW_CHECKPOINT_DOMAIN.len() + 20] ^= 1;
    assert!(matches!(
        AuthoredShadowCheckpoint::decode(&corrupt),
        Err(AuthoredShadowInitError::CheckpointChecksum)
    ));
}

#[test]
fn checkpoint_and_store_mismatches_are_rejected_without_history_writes() {
    let source_store = AppStore::default();
    let mut source = AuthoredHhhsShadow::new(project(0x23)).unwrap();
    block_on(source.dispatch(
        &source_store,
        AuthoredRevision {
            projection_revision: 7,
            commands: vec![asset(1, 0xa)],
        },
    ))
    .unwrap();
    let archive = source.export_archive().unwrap();
    let mut wrong_root = source.checkpoint().unwrap();
    wrong_root.state_root[0] ^= 1;
    let project = block_on(DurableProject::import_archive(&archive)).unwrap();
    assert!(matches!(
        AuthoredHhhsShadow::from_project_checkpoint(project, wrong_root),
        Err(AuthoredShadowInitError::CheckpointStateRoot)
    ));

    let checkpoint = source.checkpoint().unwrap();
    let project = block_on(DurableProject::import_archive(&archive)).unwrap();
    let recovered = AuthoredHhhsShadow::from_project_checkpoint(project, checkpoint).unwrap();
    let mismatched_store = AppStore::default();
    mismatched_store
        .dispatch(hyperscope_app::AppEvent::AuthoredRevision(
            AuthoredRevision {
                projection_revision: 99,
                commands: Vec::new(),
            },
        ))
        .unwrap();
    assert!(matches!(
        recovered.align_store(&mismatched_store),
        Err(AuthoredShadowInitError::StoreBaseline {
            expected: Some(7),
            actual: Some(99)
        })
    ));
    assert_eq!(recovered.history_len(), 1);
    assert_eq!(
        mismatched_store
            .summary_snapshot()
            .authored_projection_revision,
        Some(99)
    );
}

#[test]
fn durable_first_failure_publishes_nothing_and_reopen_recovers_the_cursor() {
    let project_id = project(0x2401);
    let store = AppStore::default();
    let mut coordinator = DurableAuthoredCoordinator::new(project_id).unwrap();
    let revision = AuthoredRevision {
        projection_revision: 41,
        commands: vec![asset(1, 0xa)],
    };
    let before_summary = store.summary_snapshot();
    let before_scene = store.authored_scene_snapshot();
    let before_log = coordinator.durable_bytes();
    coordinator.fail_next_persist();

    let error = block_on(coordinator.dispatch(&store, revision.clone())).unwrap_err();
    assert!(matches!(error, DurableAuthoredDispatchError::Adapter(_)));
    assert_eq!(coordinator.history_len(), 0);
    assert_eq!(coordinator.observed_projection_revision(), None);
    assert!(coordinator.fault().is_none());
    assert_eq!(coordinator.durable_bytes(), before_log);
    assert_eq!(store.summary_snapshot(), before_summary);
    assert_eq!(store.authored_scene_snapshot(), before_scene);

    coordinator = DurableAuthoredCoordinator::recover(project_id, before_log).unwrap();

    let applied = block_on(coordinator.dispatch(&store, revision)).unwrap();
    assert!(matches!(
        applied.durable,
        DurableAuthoredObservation::Applied {
            projection_revision: 41,
            history_len: 1,
            ..
        }
    ));
    assert_eq!(applied.app.unwrap().disposition, CommitDisposition::Applied);
    assert_eq!(coordinator.observed_projection_revision(), Some(41));
    assert_eq!(
        store.authored_scene_snapshot().projection_revision,
        Some(41)
    );

    let recovered =
        DurableAuthoredCoordinator::recover(project_id, coordinator.durable_bytes()).unwrap();
    assert_eq!(recovered.observed_projection_revision(), Some(41));
    assert_eq!(recovered.history_len(), 1);
    assert_eq!(
        recovered.project_state().unwrap(),
        coordinator.project_state().unwrap()
    );
}

#[test]
fn durable_recovery_restores_a_fresh_store_without_fabricating_history() {
    let project_id = project(0x2410);
    let source_store = AppStore::default();
    let mut source = DurableAuthoredCoordinator::new(project_id).unwrap();
    block_on(source.dispatch(
        &source_store,
        AuthoredRevision {
            projection_revision: 7,
            commands: vec![asset(1, 0xa)],
        },
    ))
    .unwrap();
    block_on(source.dispatch(
        &source_store,
        AuthoredRevision {
            projection_revision: 8,
            commands: vec![transform(2, 0xe, 3.0)],
        },
    ))
    .unwrap();

    let expected_scene = source_store.authored_scene_snapshot();
    let durable_bytes = source.durable_bytes();
    let mut recovered =
        DurableAuthoredCoordinator::recover(project_id, durable_bytes.clone()).unwrap();
    let recovered_state = recovered.project_state().unwrap();
    let fresh_store = AppStore::default();

    let commit = recovered.restore_store(&fresh_store).unwrap().unwrap();
    assert_eq!(commit.disposition, CommitDisposition::Applied);
    assert_eq!(fresh_store.authored_scene_snapshot(), expected_scene);
    assert_eq!(recovered.history_len(), 2);
    assert_eq!(recovered.project_state().unwrap(), recovered_state);
    assert_eq!(recovered.durable_bytes(), durable_bytes);

    let before_second_restore = fresh_store.summary_snapshot();
    assert_eq!(recovered.restore_store(&fresh_store).unwrap(), None);
    assert_eq!(fresh_store.summary_snapshot(), before_second_restore);
    assert_eq!(recovered.history_len(), 2);
    assert_eq!(recovered.durable_bytes(), durable_bytes);

    let mismatched_store = AppStore::default();
    mismatched_store
        .dispatch(hyperscope_app::AppEvent::AuthoredRevision(
            AuthoredRevision {
                projection_revision: 99,
                commands: vec![asset(99, 0xb)],
            },
        ))
        .unwrap();
    let mismatched_before = mismatched_store.authored_scene_snapshot();
    assert!(matches!(
        recovered.restore_store(&mismatched_store),
        Err(DurableAuthoredRestoreError::StoreProjectionRevision {
            expected: Some(8),
            actual: Some(99),
        })
    ));
    assert_eq!(
        mismatched_store.authored_scene_snapshot(),
        mismatched_before
    );
    assert_eq!(recovered.history_len(), 2);
    assert_eq!(recovered.durable_bytes(), durable_bytes);

    let divergent_store = AppStore::default();
    divergent_store
        .dispatch(hyperscope_app::AppEvent::AuthoredRevision(
            AuthoredRevision {
                projection_revision: 8,
                commands: vec![asset(98, 0xb)],
            },
        ))
        .unwrap();
    let divergent_before = divergent_store.authored_scene_snapshot();
    assert!(matches!(
        recovered.restore_store(&divergent_store),
        Err(DurableAuthoredRestoreError::StoreProjectionContent)
    ));
    assert_eq!(divergent_store.authored_scene_snapshot(), divergent_before);
    assert_eq!(recovered.history_len(), 2);
    assert_eq!(recovered.durable_bytes(), durable_bytes);

    let mut ingress = LocalPeerIngress::new(8).unwrap();
    let continued = block_on(recovered.accept_local_peer(
        &mut ingress,
        &fresh_store,
        LocalPeerEnvelope::Authored(asset(3, 0xb)),
        1.0,
    ))
    .unwrap();
    assert_eq!(continued.peer.disposition, LocalPeerDisposition::Applied);
    assert_eq!(continued.peer.projection_revision, Some(9));
    assert!(continued.durable.is_some());
    assert_eq!(recovered.history_len(), 3);
    assert_eq!(
        fresh_store.authored_scene_snapshot().projection_revision,
        Some(9)
    );
}

#[test]
fn durable_session_restart_rejects_replayed_peer_history_without_writes() {
    let project_id = project(0x2411);
    let source_store = AppStore::default();
    let mut source = DurableAuthoredSession::new(project_id).unwrap();
    let first = asset(7, 0xa);
    let applied = block_on(source.accept_local_peer(
        &source_store,
        LocalPeerEnvelope::Authored(first.clone()),
        1.0,
    ))
    .unwrap();
    assert_eq!(applied.peer.disposition, LocalPeerDisposition::Applied);
    assert_eq!(applied.peer.projection_revision, Some(0));
    assert_eq!(source.history_len(), 1);

    let durable_bytes = source.durable_bytes();
    let mut recovered = DurableAuthoredSession::recover(project_id, durable_bytes.clone()).unwrap();
    let recovered_store = AppStore::default();
    recovered.restore_store(&recovered_store).unwrap().unwrap();
    let before_replay = recovered_store.summary_snapshot();
    let before_state = recovered.project_state().unwrap();

    let replay = block_on(recovered.accept_local_peer(
        &recovered_store,
        LocalPeerEnvelope::Authored(first),
        2.0,
    ))
    .unwrap();
    assert_eq!(
        replay.peer.disposition,
        LocalPeerDisposition::IgnoredDuplicate
    );
    assert!(replay.durable.is_none());
    assert_eq!(recovered_store.summary_snapshot(), before_replay);
    assert_eq!(recovered.history_len(), 1);
    assert_eq!(recovered.project_state().unwrap(), before_state);
    assert_eq!(recovered.durable_bytes(), durable_bytes);

    let stale = block_on(recovered.accept_local_peer(
        &recovered_store,
        LocalPeerEnvelope::Authored(asset(6, 0xb)),
        2.1,
    ))
    .unwrap();
    assert_eq!(stale.peer.disposition, LocalPeerDisposition::IgnoredStale);
    assert!(stale.durable.is_none());
    assert_eq!(recovered_store.summary_snapshot(), before_replay);
    assert_eq!(recovered.history_len(), 1);
    assert_eq!(recovered.durable_bytes(), durable_bytes);

    let next = block_on(recovered.accept_local_peer(
        &recovered_store,
        LocalPeerEnvelope::Authored(asset(8, 0xb)),
        2.2,
    ))
    .unwrap();
    assert_eq!(next.peer.disposition, LocalPeerDisposition::Applied);
    assert_eq!(next.peer.projection_revision, Some(1));
    assert!(next.durable.is_some());
    assert_eq!(recovered.history_len(), 2);
    assert_eq!(
        recovered_store
            .authored_scene_snapshot()
            .projection_revision,
        Some(1)
    );
}

#[test]
fn concurrent_carrier_records_converge_independent_of_arrival_order() {
    let project_id = project(0x2414);
    let mut source_a = DurableProject::new(project_id).unwrap();
    let mut source_b = DurableProject::new(project_id).unwrap();
    let record_a = block_on(source_a.admit(&asset(1, 0xa))).unwrap();
    let record_b = block_on(source_b.admit(&asset(2, 0xb))).unwrap();

    let left_store = AppStore::default();
    let mut left = DurableAuthoredSession::new(project_id).unwrap();
    let left_first = block_on(left.accept_replica_record(&left_store, record_a.clone())).unwrap();
    let left_second = block_on(left.accept_replica_record(&left_store, record_b.clone())).unwrap();
    assert!(matches!(
        left_first.durable,
        DurableCarrierObservation::Applied {
            projection_revision: 0,
            history_len: 1,
            ..
        }
    ));
    assert!(matches!(
        left_second.durable,
        DurableCarrierObservation::Applied {
            projection_revision: 1,
            history_len: 2,
            ..
        }
    ));

    let right_store = AppStore::default();
    let mut right = DurableAuthoredSession::new(project_id).unwrap();
    block_on(right.accept_replica_record(&right_store, record_b)).unwrap();
    block_on(right.accept_replica_record(&right_store, record_a)).unwrap();

    assert_eq!(
        left.project_state().unwrap(),
        right.project_state().unwrap()
    );
    assert_eq!(
        left_store.authored_scene_snapshot(),
        right_store.authored_scene_snapshot()
    );
    assert_eq!(left_store.authored_scene_snapshot().assets.len(), 2);
}

#[test]
fn carrier_defers_children_and_duplicates_are_exact_zero_write_noops() {
    let project_id = project(0x2415);
    let mut source = DurableProject::new(project_id).unwrap();
    let parent = block_on(source.admit(&asset(1, 0xa))).unwrap();
    let child = block_on(source.admit(&asset(2, 0xb))).unwrap();

    let store = AppStore::default();
    let mut target = DurableAuthoredSession::new(project_id).unwrap();
    let before_deferred = target.durable_bytes();
    let deferred = block_on(target.accept_replica_record(&store, child.clone())).unwrap();
    assert!(matches!(
        deferred.durable,
        DurableCarrierObservation::Deferred {
            history_len: 0,
            ref missing,
            ..
        } if missing == &[parent.entry_hash()]
    ));
    assert_eq!(deferred.app, None);
    assert_eq!(target.durable_bytes(), before_deferred);
    assert_eq!(store.authored_scene_snapshot().projection_revision, None);

    block_on(target.accept_replica_record(&store, parent.clone())).unwrap();
    block_on(target.accept_replica_record(&store, child)).unwrap();
    assert_eq!(target.history_len(), 2);
    assert_eq!(target.observed_projection_revision(), Some(1));
    let settled_store = store.summary_snapshot();
    let settled_log = target.durable_bytes();

    let duplicate = block_on(target.accept_replica_record(&store, parent)).unwrap();
    assert!(matches!(
        duplicate.durable,
        DurableCarrierObservation::AlreadyPresent { history_len: 2, .. }
    ));
    assert_eq!(duplicate.app, None);
    assert_eq!(target.observed_projection_revision(), Some(1));
    assert_eq!(target.durable_bytes(), settled_log);
    assert_eq!(store.summary_snapshot(), settled_store);
}

#[test]
fn carrier_persistence_failure_is_atomic_and_retry_survives_restart() {
    let project_id = project(0x2416);
    let mut source = DurableProject::new(project_id).unwrap();
    let record = block_on(source.admit(&asset(1, 0xa))).unwrap();
    let store = AppStore::default();
    let mut target = DurableAuthoredSession::new(project_id).unwrap();
    let before_store = store.summary_snapshot();
    let before_log = target.durable_bytes();
    target.fail_next_persist();

    assert!(block_on(target.accept_replica_record(&store, record.clone())).is_err());
    assert_eq!(target.history_len(), 0);
    assert_eq!(target.observed_projection_revision(), None);
    assert_eq!(target.durable_bytes(), before_log);
    assert_eq!(store.summary_snapshot(), before_store);

    target = DurableAuthoredSession::recover(project_id, before_log).unwrap();

    let applied = block_on(target.accept_replica_record(&store, record.clone())).unwrap();
    assert!(matches!(
        applied.durable,
        DurableCarrierObservation::Applied {
            projection_revision: 0,
            history_len: 1,
            ..
        }
    ));
    let durable_bytes = target.durable_bytes();
    let mut recovered = DurableAuthoredSession::recover(project_id, durable_bytes.clone()).unwrap();
    let recovered_store = AppStore::default();
    recovered.restore_store(&recovered_store).unwrap().unwrap();
    let before_duplicate = recovered_store.summary_snapshot();
    let duplicate = block_on(recovered.accept_replica_record(&recovered_store, record)).unwrap();
    assert!(matches!(
        duplicate.durable,
        DurableCarrierObservation::AlreadyPresent { history_len: 1, .. }
    ));
    assert_eq!(recovered.durable_bytes(), durable_bytes);
    assert_eq!(recovered_store.summary_snapshot(), before_duplicate);
}

#[test]
fn explicit_archive_adoption_bootstraps_a_fresh_local_cursor_once() {
    let project_id = project(0x2412);
    let first = asset(1, 0xa);
    let mut source = DurableProject::new(project_id).unwrap();
    block_on(source.admit(&first)).unwrap();
    let archive = source.export_archive().unwrap();
    let imported = block_on(DurableProject::import_archive(&archive)).unwrap();
    assert!(matches!(
        DurableAuthoredSession::from_project(imported),
        Err(DurableAuthoredSessionInitError::Coordinator(
            DurableAuthoredInitError::MissingCursor
        ))
    ));

    let imported = block_on(DurableProject::import_archive(&archive)).unwrap();
    let mut adopted = block_on(DurableAuthoredSession::from_imported_project(imported)).unwrap();
    assert_eq!(adopted.observed_projection_revision(), Some(0));
    assert_eq!(adopted.history_len(), 1);
    let store = AppStore::default();
    adopted.restore_store(&store).unwrap().unwrap();
    assert_eq!(store.authored_scene_snapshot().projection_revision, Some(0));

    let adopted_bytes = adopted.durable_bytes();
    let recovered_project = DurableProject::recover(project_id, adopted_bytes.clone()).unwrap();
    let retried = block_on(DurableAuthoredSession::from_imported_project(
        recovered_project,
    ))
    .unwrap();
    assert_eq!(retried.observed_projection_revision(), Some(0));
    assert_eq!(retried.durable_bytes(), adopted_bytes);

    let next = block_on(adopted.accept_local_peer(
        &store,
        LocalPeerEnvelope::Authored(asset(2, 0xb)),
        2.0,
    ))
    .unwrap();
    assert_eq!(next.peer.projection_revision, Some(1));
    assert_eq!(adopted.history_len(), 2);
}

#[test]
fn archive_extension_advances_a_valid_local_prefix_cursor_once() {
    let project_id = project(0x2413);
    let first = asset(1, 0xa);
    let second = asset(2, 0xb);
    let local_store = AppStore::default();
    let mut local = DurableAuthoredSession::new(project_id).unwrap();
    block_on(local.accept_local_peer(
        &local_store,
        LocalPeerEnvelope::Authored(first.clone()),
        1.0,
    ))
    .unwrap();
    let local_bytes = local.durable_bytes();

    let mut archive_source = DurableProject::new(project_id).unwrap();
    block_on(archive_source.admit(&first)).unwrap();
    block_on(archive_source.admit(&second)).unwrap();
    let archive = archive_source.project_archive().unwrap();

    let mut extended = DurableProject::recover(project_id, local_bytes).unwrap();
    block_on(extended.apply_archive(&archive)).unwrap();
    let before_adoption_bytes = extended.durable_bytes();
    let mut adopted = block_on(DurableAuthoredSession::from_imported_project(extended)).unwrap();
    assert_eq!(adopted.observed_projection_revision(), Some(1));
    assert_eq!(adopted.history_len(), 2);
    assert_ne!(adopted.durable_bytes(), before_adoption_bytes);

    let store = AppStore::default();
    adopted.restore_store(&store).unwrap().unwrap();
    assert_eq!(store.authored_scene_snapshot().projection_revision, Some(1));
    assert_eq!(store.authored_scene_snapshot().assets.len(), 2);
    let next = block_on(adopted.accept_local_peer(
        &store,
        LocalPeerEnvelope::Authored(asset(3, 0xc)),
        3.0,
    ))
    .unwrap();
    assert_eq!(next.peer.projection_revision, Some(2));
}

#[test]
fn durable_first_rejects_multi_command_revisions_before_either_side_changes() {
    let store = AppStore::default();
    let mut coordinator = DurableAuthoredCoordinator::new(project(0x2402)).unwrap();
    let before_summary = store.summary_snapshot();
    let before_log = coordinator.durable_bytes();

    let error = block_on(coordinator.dispatch(
        &store,
        AuthoredRevision {
            projection_revision: 1,
            commands: vec![asset(1, 0xa), transform(2, 0xe, 3.0)],
        },
    ))
    .unwrap_err();

    assert!(matches!(
        error,
        DurableAuthoredDispatchError::UnsupportedCommandCount {
            projection_revision: 1,
            command_count: 2,
        }
    ));
    assert_eq!(coordinator.history_len(), 0);
    assert_eq!(coordinator.durable_bytes(), before_log);
    assert_eq!(store.summary_snapshot(), before_summary);
    assert!(coordinator.fault().is_none());
}

#[test]
fn durable_first_stale_revision_is_a_zero_write_noop() {
    let store = AppStore::default();
    let mut coordinator = DurableAuthoredCoordinator::new(project(0x2404)).unwrap();
    block_on(coordinator.dispatch(
        &store,
        AuthoredRevision {
            projection_revision: 3,
            commands: vec![asset(1, 0xa)],
        },
    ))
    .unwrap();
    let before_log = coordinator.durable_bytes();

    let stale = block_on(coordinator.dispatch(
        &store,
        AuthoredRevision {
            projection_revision: 3,
            commands: vec![asset(2, 0xb)],
        },
    ))
    .unwrap();

    assert_eq!(
        stale.app.unwrap().disposition,
        CommitDisposition::IgnoredStale
    );
    assert_eq!(
        stale.durable,
        DurableAuthoredObservation::IgnoredStale {
            projection_revision: 3,
            observed_projection_revision: Some(3),
            history_len: 1,
        }
    );
    assert_eq!(coordinator.history_len(), 1);
    assert_eq!(coordinator.durable_bytes(), before_log);
    assert!(!coordinator
        .project_state()
        .unwrap()
        .assets
        .contains_key(&AssetId::from_u128(0xb).unwrap()));
}

#[test]
fn durable_local_peer_reopen_is_not_deduplicated_after_persistence_failure() {
    let store = AppStore::default();
    let mut ingress = LocalPeerIngress::new(8).unwrap();
    let mut coordinator = DurableAuthoredCoordinator::new(project(0x2405)).unwrap();
    let envelope = asset(7, 0xa);
    coordinator.fail_next_persist();

    let error = block_on(coordinator.accept_local_peer(
        &mut ingress,
        &store,
        LocalPeerEnvelope::Authored(envelope.clone()),
        1.0,
    ))
    .unwrap_err();
    assert!(matches!(
        error,
        DurableLocalPeerError::Durable(DurableAuthoredDispatchError::Adapter(_))
    ));
    assert_eq!(coordinator.history_len(), 0);
    assert_eq!(store.authored_scene_snapshot().projection_revision, None);

    coordinator =
        DurableAuthoredCoordinator::recover(project(0x2405), coordinator.durable_bytes()).unwrap();

    let applied = block_on(coordinator.accept_local_peer(
        &mut ingress,
        &store,
        LocalPeerEnvelope::Authored(envelope.clone()),
        1.1,
    ))
    .unwrap();
    assert_eq!(applied.peer.disposition, LocalPeerDisposition::Applied);
    assert_eq!(applied.peer.projection_revision, Some(0));
    assert!(applied.durable.is_some());
    assert_eq!(coordinator.history_len(), 1);

    let duplicate = block_on(coordinator.accept_local_peer(
        &mut ingress,
        &store,
        LocalPeerEnvelope::Authored(envelope),
        1.2,
    ))
    .unwrap();
    assert_eq!(
        duplicate.peer.disposition,
        LocalPeerDisposition::IgnoredDuplicate
    );
    assert!(duplicate.durable.is_none());
    assert_eq!(coordinator.history_len(), 1);
}

#[test]
fn durable_local_peer_keeps_presence_out_of_hhhs() {
    let store = AppStore::default();
    let mut ingress = LocalPeerIngress::new(8).unwrap();
    let mut coordinator = DurableAuthoredCoordinator::new(project(0x2406)).unwrap();
    let presence = PresenceEnvelope {
        header: MessageHeader {
            version: CURRENT_PROTOCOL_VERSION,
            message_id: MessageId::from_u128(0x2406).unwrap(),
            sender: PeerId::from_u128(0x2407).unwrap(),
            sequence: 1,
        },
        presence: EphemeralPresence {
            ttl_millis: 500,
            camera: None,
            selection: Vec::new(),
            authoring_leases: Vec::new(),
            focus: None,
            active_cue: None,
            animation_seconds: Some(2.0),
        },
    };

    let receipt = block_on(coordinator.accept_local_peer(
        &mut ingress,
        &store,
        LocalPeerEnvelope::Presence(presence),
        2.0,
    ))
    .unwrap();
    assert_eq!(receipt.peer.disposition, LocalPeerDisposition::Applied);
    assert!(receipt.durable.is_none());
    assert_eq!(coordinator.history_len(), 0);
    assert_eq!(store.presence_snapshot().len(), 1);
    assert_eq!(store.authored_scene_snapshot().projection_revision, None);
}

struct BypassOnPersist {
    store: AppStore,
    inner: MemoryDurability,
    bypass: Option<AuthoredRevision>,
}

impl AsyncTransactionSink for BypassOnPersist {
    type Error = String;

    fn writer_lease_active(&self) -> bool {
        self.inner.writer_lease_active()
    }

    fn recovered_state(&self) -> StorageRecoveryState {
        self.inner.recovered_state()
    }

    async fn persist(
        &mut self,
        transaction: &StorageTransaction,
        expected: StorageRecoveryState,
    ) -> Result<(), Self::Error> {
        if let Some(revision) = self.bypass.take() {
            self.store
                .dispatch(hyperscope_app::AppEvent::AuthoredRevision(revision))
                .map_err(|error| error.to_string())?;
        }
        self.inner.persist(transaction, expected).await
    }
}

#[test]
fn inbound_carrier_survives_an_appstore_bypass_as_recoverable_authority() {
    let project_id = project(0x2417);
    let mut source = DurableProject::new(project_id).unwrap();
    let record = block_on(source.admit(&asset(1, 0xa))).unwrap();
    let store = AppStore::default();
    let durability = MemoryDurability::default();
    let durability_control = durability.control();
    let durable_project = DurableProject::with_sink(
        project_id,
        BypassOnPersist {
            store: store.clone(),
            inner: durability,
            bypass: Some(AuthoredRevision {
                projection_revision: 99,
                commands: vec![asset(99, 0xb)],
            }),
        },
    )
    .unwrap();
    let mut target = DurableAuthoredSession::from_project(durable_project).unwrap();

    let report = block_on(target.accept_replica_record(&store, record.clone())).unwrap();
    assert!(matches!(
        report.app,
        Some(Err(ref fault))
            if fault.projection_revision == 0
                && fault.history_len == 1
                && fault.reason.contains("baseline changed")
    ));
    assert!(matches!(
        report.durable,
        DurableCarrierObservation::Applied {
            projection_revision: 0,
            history_len: 1,
            ..
        }
    ));
    assert_eq!(target.observed_projection_revision(), Some(0));
    assert_eq!(
        store.authored_scene_snapshot().projection_revision,
        Some(99)
    );
    let settled_log = durability_control.bytes();

    assert!(matches!(
        block_on(target.accept_replica_record(&store, record)),
        Err(DurableCarrierError::Poisoned { .. })
    ));
    assert_eq!(durability_control.bytes(), settled_log);

    let recovered = DurableAuthoredSession::recover(project_id, settled_log).unwrap();
    let recovered_store = AppStore::default();
    recovered.restore_store(&recovered_store).unwrap().unwrap();
    assert_eq!(
        recovered_store
            .authored_scene_snapshot()
            .projection_revision,
        Some(0)
    );
    assert!(recovered_store
        .authored_scene_snapshot()
        .assets
        .iter()
        .any(|asset| asset.id == AssetId::from_u128(0xa).unwrap()));
}

#[test]
fn durable_first_detects_an_appstore_bypass_during_persistence() {
    let store = AppStore::default();
    let durable_project = DurableProject::with_sink(
        project(0x2403),
        BypassOnPersist {
            store: store.clone(),
            inner: MemoryDurability::default(),
            bypass: Some(AuthoredRevision {
                projection_revision: 99,
                commands: vec![asset(99, 0xb)],
            }),
        },
    )
    .unwrap();
    let mut coordinator = DurableAuthoredCoordinator::from_project(durable_project).unwrap();

    let report = block_on(coordinator.dispatch(
        &store,
        AuthoredRevision {
            projection_revision: 1,
            commands: vec![asset(1, 0xa)],
        },
    ))
    .unwrap();
    assert!(matches!(
        report.app,
        Err(ref fault)
            if fault.projection_revision == 1
                && fault.history_len == 1
                && fault.reason.contains("baseline changed")
    ));
    assert!(matches!(
        report.durable,
        DurableAuthoredObservation::Applied {
            projection_revision: 1,
            history_len: 1,
            ..
        }
    ));
    assert_eq!(coordinator.history_len(), 1);
    assert_eq!(coordinator.observed_projection_revision(), Some(1));
    assert_eq!(
        store.authored_scene_snapshot().projection_revision,
        Some(99)
    );
    assert!(coordinator
        .project_state()
        .unwrap()
        .assets
        .contains_key(&AssetId::from_u128(0xa).unwrap()));
    assert!(store
        .authored_scene_snapshot()
        .assets
        .iter()
        .any(|asset| asset.id == AssetId::from_u128(0xb).unwrap()));

    let poisoned = block_on(coordinator.dispatch(
        &store,
        AuthoredRevision {
            projection_revision: 100,
            commands: vec![asset(100, 0xc)],
        },
    ))
    .unwrap_err();
    assert!(matches!(
        poisoned,
        DurableAuthoredDispatchError::Poisoned {
            skipped_projection_revision: 100,
            ..
        }
    ));
    assert_eq!(coordinator.history_len(), 1);
}

#[test]
fn invalid_app_revision_never_reaches_hhhs() {
    let store = AppStore::default();
    let mut shadow = AuthoredHhhsShadow::new(project(3)).unwrap();
    let mut invalid = asset(1, 0xa);
    invalid.header.version = ProtocolVersion {
        major: 99,
        minor: 0,
    };

    let error = block_on(shadow.dispatch(
        &store,
        AuthoredRevision {
            projection_revision: 1,
            commands: vec![invalid],
        },
    ))
    .unwrap_err();
    assert!(error.to_string().contains("unsupported protocol version"));
    assert_eq!(store.summary_snapshot().authored_projection_revision, None);
    assert_eq!(shadow.history_len(), 0);
    assert!(shadow.fault().is_none());
}

#[test]
fn bypassed_authored_revision_is_detected_before_mirroring() {
    let store = AppStore::default();
    let mut shadow = AuthoredHhhsShadow::new(project(33)).unwrap();
    store
        .dispatch(hyperscope_app::AppEvent::AuthoredRevision(
            AuthoredRevision {
                projection_revision: 4,
                commands: vec![asset(1, 0xa)],
            },
        ))
        .unwrap();

    let report = block_on(shadow.dispatch(
        &store,
        AuthoredRevision {
            projection_revision: 9,
            commands: vec![asset(2, 0xb)],
        },
    ))
    .unwrap();
    assert_eq!(report.app.disposition, CommitDisposition::Applied);
    assert!(matches!(
        report.shadow,
        Err(AuthoredShadowError::Fault(ref fault))
            if fault.projection_revision == 9
                && fault.admitted_prefix == 0
                && fault.history_len == 0
                && fault.reason.contains("bypassed shadow baseline")
    ));
    assert_eq!(
        store.summary_snapshot().authored_projection_revision,
        Some(9)
    );
    assert_eq!(shadow.history_len(), 0);
}

#[derive(Debug)]
struct FailOnPersist {
    attempt: usize,
    fail_on: usize,
    inner: MemoryDurability,
}

impl AsyncTransactionSink for FailOnPersist {
    type Error = &'static str;

    fn writer_lease_active(&self) -> bool {
        self.inner.writer_lease_active()
    }

    fn recovered_state(&self) -> StorageRecoveryState {
        self.inner.recovered_state()
    }

    async fn persist(
        &mut self,
        transaction: &StorageTransaction,
        expected: StorageRecoveryState,
    ) -> Result<(), Self::Error> {
        self.attempt += 1;
        if self.attempt == self.fail_on {
            Err("injected batch persistence failure")
        } else {
            self.inner
                .persist(transaction, expected)
                .await
                .map_err(|_| "inner memory durability failed")
        }
    }
}

#[test]
fn first_persistence_failure_does_not_deny_the_app_commit() {
    let store = AppStore::default();
    let durability = MemoryDurability::default();
    let project = DurableProject::with_sink(
        project(4),
        FailOnPersist {
            attempt: 0,
            fail_on: 1,
            inner: durability,
        },
    )
    .unwrap();
    let mut shadow = AuthoredHhhsShadow::from_project(project).unwrap();

    let report = block_on(shadow.dispatch(
        &store,
        AuthoredRevision {
            projection_revision: 1,
            commands: vec![asset(1, 0xa)],
        },
    ))
    .unwrap();
    assert_eq!(report.app.disposition, CommitDisposition::Applied);
    assert!(matches!(
        &report.shadow,
        Err(AuthoredShadowError::Fault(ref fault))
            if fault.failed_command_index == 0
                && fault.admitted_prefix == 0
                && fault.history_len == 0
    ));
    assert_eq!(
        store.summary_snapshot().authored_projection_revision,
        Some(1)
    );
    assert_eq!(shadow.history_len(), 0);
}

#[test]
fn mid_revision_failure_reports_prefix_and_poison_prevents_naive_retry() {
    let store = AppStore::default();
    let project_id = project(5);
    let durability = MemoryDurability::default();
    let durability_control = durability.control();
    let project = DurableProject::with_sink(
        project_id,
        FailOnPersist {
            attempt: 0,
            fail_on: 2,
            inner: durability,
        },
    )
    .unwrap();
    let mut shadow = AuthoredHhhsShadow::from_project(project).unwrap();

    let report = block_on(shadow.dispatch(
        &store,
        AuthoredRevision {
            projection_revision: 1,
            commands: vec![asset(1, 0xa), transform(2, 0xe, 2.0), asset(3, 0xb)],
        },
    ))
    .unwrap();
    assert_eq!(report.app.disposition, CommitDisposition::Applied);
    assert!(
        matches!(
            &report.shadow,
            Err(AuthoredShadowError::Fault(ref fault))
                if fault.failed_command_index == 1
                    && fault.admitted_prefix == 1
                    && fault.history_len == 1
        ),
        "{:?}",
        report.shadow
    );
    assert_eq!(
        store.summary_snapshot().authored_projection_revision,
        Some(1)
    );
    assert_eq!(shadow.history_len(), 1);
    assert!(matches!(
        shadow.project_state(),
        Err(AdapterError::Repair(_))
    ));
    let recovered = DurableProject::recover(project_id, durability_control.bytes()).unwrap();
    assert_eq!(recovered.state().unwrap().assets.len(), 1);
    assert!(matches!(
        shadow.checkpoint(),
        Err(AuthoredShadowInitError::FaultedCheckpoint(ref fault))
            if fault.admitted_prefix == 1
    ));

    let later = block_on(shadow.dispatch(
        &store,
        AuthoredRevision {
            projection_revision: 2,
            commands: vec![asset(4, 0xc)],
        },
    ))
    .unwrap();
    assert_eq!(later.app.disposition, CommitDisposition::Applied);
    assert!(matches!(
        later.shadow,
        Err(AuthoredShadowError::Poisoned {
            skipped_projection_revision: 2,
            ..
        })
    ));
    assert_eq!(
        store.summary_snapshot().authored_projection_revision,
        Some(2)
    );
    assert_eq!(shadow.history_len(), 1, "poisoned shadow must not retry");
}
