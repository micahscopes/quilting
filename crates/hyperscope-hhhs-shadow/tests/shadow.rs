use futures::executor::block_on;
use futures::future::ready;
use hhhs_replica::AsyncTransactionSink;
use hhhs_store::StorageTransaction;
use hyperscape_hhhs::{DurableProject, ProjectId};
use hyperscape_protocol::{
    AssetDescriptor, AssetId, AuthoredCommand, AuthoredEnvelope, EntityId, MessageHeader,
    MessageId, PeerId, ProtocolVersion, WireTransform, CURRENT_PROTOCOL_VERSION,
};
use hyperscope_app::{AppStore, AuthoredRevision, CommitDisposition};
use hyperscope_hhhs_shadow::{AuthoredHhhsShadow, AuthoredShadowError, AuthoredShadowObservation};

fn project(value: u128) -> ProjectId {
    ProjectId::from_u128(value).unwrap()
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
