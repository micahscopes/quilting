use futures::executor::block_on;
use futures::future::ready;
use hhhs::Digest;
use hhhs_replica::AsyncTransactionSink;
use hhhs_store::StorageTransaction;
use hyperscape_hhhs::{DurableProject, ProjectId};
use hyperscape_protocol::{
    AssetDescriptor, AssetId, AuthoredCommand, AuthoredEnvelope, EntityId, MessageHeader,
    MessageId, PeerId, ProtocolVersion, WireTransform, CURRENT_PROTOCOL_VERSION,
};
use hyperscope_app::{AppStore, AuthoredRevision, CommitDisposition};
use hyperscope_hhhs_shadow::{
    AuthoredHhhsShadow, AuthoredShadowCheckpoint, AuthoredShadowError, AuthoredShadowInitError,
    AuthoredShadowObservation, AUTHORED_SHADOW_CHECKPOINT_DOMAIN,
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
    block_on(source.dispatch(
        &source_store,
        AuthoredRevision {
            projection_revision: 17,
            commands: vec![asset(1, 0xa), transform(2, 0xe, 3.0)],
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
}

impl AsyncTransactionSink for FailOnPersist {
    type Error = &'static str;

    fn persist(
        &mut self,
        _transaction: &StorageTransaction,
    ) -> impl std::future::Future<Output = Result<(), Self::Error>> {
        self.attempt += 1;
        ready(if self.attempt == self.fail_on {
            Err("injected batch persistence failure")
        } else {
            Ok(())
        })
    }
}

#[test]
fn first_persistence_failure_does_not_deny_the_app_commit() {
    let store = AppStore::default();
    let project = DurableProject::with_sink(
        project(4),
        FailOnPersist {
            attempt: 0,
            fail_on: 1,
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
    let project = DurableProject::with_sink(
        project(5),
        FailOnPersist {
            attempt: 0,
            fail_on: 2,
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
    assert_eq!(shadow.project_state().unwrap().assets.len(), 1);
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
