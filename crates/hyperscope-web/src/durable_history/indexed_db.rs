use super::{
    decode_ordered_rows, project_key_bounds, DurableHistoryError, OriginPersistence, DATABASE_NAME,
    DATABASE_VERSION, TRANSACTION_STORE,
};
use futures::try_join;
use hhhs_replica::AsyncTransactionSink;
use hhhs_store::{encode_storage_transaction, MemoryStorage, ReplicaStorage, StorageTransaction};
use hhhs_web_browser::{
    BrowserDurabilityHint, IndexedDbLogOptions, IndexedDbReplicaLog, ReplicaLogId,
};
use hyperscape_hhhs::{DurableProject, ProjectId};
use hyperscope_app::AppStore;
use indexed_db_futures::database::Database;
use indexed_db_futures::prelude::*;
use indexed_db_futures::typed_array::Uint8Array;

/// Browser durability is upstream HHHS's Web-Lock-owned IndexedDB log. This
/// alias deliberately exposes no database handle and cannot be cloned into a
/// second writer.
pub type IndexedDbDurability = IndexedDbReplicaLog;

fn log_id(project_id: ProjectId) -> ReplicaLogId {
    let mut label = Vec::with_capacity(48);
    label.extend_from_slice(b"hyperscape authored project v1\0");
    label.extend_from_slice(project_id.as_uuid().as_bytes());
    ReplicaLogId::derive(label)
}

fn log_options() -> IndexedDbLogOptions {
    IndexedDbLogOptions::new(DATABASE_NAME)
        .with_database_version(DATABASE_VERSION)
        .with_transaction_store(TRANSACTION_STORE)
        .with_durability(BrowserDurabilityHint::Strict)
}

async fn load_legacy_transactions(
    project_id: ProjectId,
) -> Result<Vec<StorageTransaction>, DurableHistoryError> {
    let db = Database::open(DATABASE_NAME)
        .with_version(DATABASE_VERSION)
        .await
        .map_err(indexed_db_error)?;
    let (lower, upper) = project_key_bounds(project_id);
    let transaction = db
        .transaction(TRANSACTION_STORE)
        .build()
        .map_err(indexed_db_error)?;
    let store = transaction
        .object_store(TRANSACTION_STORE)
        .map_err(indexed_db_error)?;
    let keys = store
        .get_all_keys::<String>()
        .with_query(lower.clone()..=upper.clone())
        .primitive()
        .map_err(indexed_db_error)?;
    let values = store
        .get_all::<Uint8Array>()
        .with_query(lower..=upper)
        .primitive()
        .map_err(indexed_db_error)?;
    let (keys, values) = try_join!(keys, values).map_err(indexed_db_error)?;
    let keys: Vec<_> = keys.collect::<Result<_, _>>().map_err(indexed_db_error)?;
    let values: Vec<_> = values.collect::<Result<_, _>>().map_err(indexed_db_error)?;
    drop(store);
    transaction.commit().await.map_err(indexed_db_error)?;
    db.close();
    if keys.len() != values.len() {
        return Err(DurableHistoryError::IndexedDb(format!(
            "legacy transaction key/value snapshot length mismatch: {} keys, {} values",
            keys.len(),
            values.len()
        )));
    }
    decode_ordered_rows(
        project_id,
        keys.into_iter()
            .zip(values)
            .map(|(key, value)| (key, value.into())),
    )
}

fn normalize_legacy_transactions(
    transactions: Vec<StorageTransaction>,
) -> Result<Vec<StorageTransaction>, DurableHistoryError> {
    let storage = MemoryStorage::new();
    let mut retained = Vec::with_capacity(transactions.len());
    let transaction_count = transactions.len();
    for (index, transaction) in transactions.into_iter().enumerate() {
        let before = storage.recovery_state();
        storage
            .commit(transaction.clone())
            .map_err(|error| DurableHistoryError::InvalidTransaction(error.to_string()))?;
        if storage.recovery_state() == before {
            if index + 1 != transaction_count {
                return Err(DurableHistoryError::InvalidTransaction(
                    "embedded legacy no-effect transaction is not a removable tail".into(),
                ));
            }
        } else {
            retained.push(transaction);
        }
    }
    Ok(retained)
}

async fn open_project_log(
    project_id: ProjectId,
) -> Result<IndexedDbDurability, DurableHistoryError> {
    let mut log = IndexedDbReplicaLog::open(log_id(project_id), log_options())
        .await
        .map_err(indexed_db_error)?;
    let legacy = normalize_legacy_transactions(load_legacy_transactions(project_id).await?)?;
    if legacy.is_empty() {
        return Ok(log);
    }
    if !log.transactions().is_empty() {
        let same = log.transactions().len() == legacy.len()
            && log.transactions().iter().zip(&legacy).all(|(left, right)| {
                encode_storage_transaction(left) == encode_storage_transaction(right)
            });
        if !same {
            return Err(DurableHistoryError::ProjectRecovery(
                "legacy and HHHS-owned browser logs disagree".into(),
            ));
        }
        return Ok(log);
    }
    for transaction in &legacy {
        let before = log.recovered_state();
        let expected = before
            .advance(
                transaction,
                before
                    .sequence()
                    .checked_add(1)
                    .ok_or(DurableHistoryError::SequenceOverflow)?,
            )
            .map_err(|error| DurableHistoryError::InvalidTransaction(error.to_string()))?;
        AsyncTransactionSink::persist(&mut log, transaction, expected)
            .await
            .map_err(indexed_db_error)?;
    }
    Ok(log)
}

/// Open and recover a browser-durable project without attaching it to current
/// application, UI, or rendering behavior.
pub async fn open_durable_project(
    project_id: ProjectId,
) -> Result<DurableProject<IndexedDbDurability>, DurableHistoryError> {
    let durability = open_project_log(project_id).await?;
    let transactions = durability.recovered_transactions();
    DurableProject::recover_trusted_transactions_with_sink(project_id, durability, transactions)
        .map_err(|error| DurableHistoryError::ProjectRecovery(error.to_string()))
}

/// Open this project's IndexedDB history, recover its paired ingress policy,
/// and align a fresh AppStore projection before returning control to a peer
/// transport.
///
/// This touches IndexedDB only. It does not initialize a renderer, canvas, or
/// GPU device.
pub async fn open_durable_authored_session(
    project_id: ProjectId,
    store: &AppStore,
) -> Result<super::OpenedDurableAuthoredSession<IndexedDbDurability>, DurableHistoryError> {
    let project = open_durable_project(project_id).await?;
    super::attach_durable_authored_session(project, store)
}

/// Validate a portable project archive in memory, then idempotently persist it
/// into this origin's dedicated authored-history database.
///
/// Rust performs both archive decoding and ordinary HHHS admission before the
/// browser database is touched. An exact retry or a locally stored prefix is
/// resumed safely. A local branch absent from the archive is rejected before
/// writing, rather than overwritten or silently merged.
pub async fn import_project_archive(
    bytes: &[u8],
) -> Result<DurableProject<IndexedDbDurability>, DurableHistoryError> {
    let validated = DurableProject::import_archive(bytes)
        .await
        .map_err(|error| DurableHistoryError::ProjectArchive(error.to_string()))?;
    import_validated_project_archive(validated).await
}

/// Persist a project that has already admitted an untrusted portable archive
/// in memory. Keeping this typed seam avoids decoding and admitting a large
/// archive twice when a caller also needs its validated project identity
/// before selecting an application session.
pub async fn import_validated_project_archive(
    validated: DurableProject,
) -> Result<DurableProject<IndexedDbDurability>, DurableHistoryError> {
    let archive = validated
        .project_archive()
        .map_err(|error| DurableHistoryError::ProjectArchive(error.to_string()))?;
    let mut durable = open_durable_project(archive.project_id()).await?;
    durable
        .apply_archive(&archive)
        .await
        .map_err(|error| DurableHistoryError::ProjectArchive(error.to_string()))?;
    Ok(durable)
}

/// Validate and persist a portable archive, explicitly adopt its local source
/// cursor, and restore a fresh AppStore before returning control to transport.
///
/// This touches IndexedDB but does not initialize a renderer or GPU device.
pub async fn import_durable_authored_session(
    bytes: &[u8],
    store: &AppStore,
) -> Result<super::OpenedDurableAuthoredSession<IndexedDbDurability>, DurableHistoryError> {
    let project = import_project_archive(bytes).await?;
    super::attach_imported_durable_authored_session(project, store).await
}

/// Persist and attach one already validated archive project without repeating
/// its in-memory HHHS admission pass.
pub async fn import_validated_durable_authored_session(
    validated: DurableProject,
    store: &AppStore,
) -> Result<super::OpenedDurableAuthoredSession<IndexedDbDurability>, DurableHistoryError> {
    let project = import_validated_project_archive(validated).await?;
    super::attach_imported_durable_authored_session(project, store).await
}

/// Query whether the browser currently protects this origin from eviction.
///
/// `BestEffort` and `Unknown` still allow strict, atomic IndexedDB commits, but
/// must not be presented as durable against storage pressure or private-mode
/// teardown.
pub async fn origin_persistence() -> Result<OriginPersistence, DurableHistoryError> {
    hhhs_web_browser::origin_persistence()
        .await
        .map_err(indexed_db_error)
        .map(|result| match result {
            hhhs_web_browser::OriginPersistence::Persistent => OriginPersistence::Persistent,
            hhhs_web_browser::OriginPersistence::BestEffort => OriginPersistence::BestEffort,
            hhhs_web_browser::OriginPersistence::Unknown => OriginPersistence::Unknown,
        })
}

/// Explicitly ask the browser to protect this origin from eviction and return
/// the resulting retention status.
///
/// Browsers may deny the request without prompting. Denial is reported as
/// `BestEffort`, not as successful durable retention.
pub async fn request_origin_persistence() -> Result<OriginPersistence, DurableHistoryError> {
    hhhs_web_browser::request_origin_persistence()
        .await
        .map_err(indexed_db_error)
        .map(|result| match result {
            hhhs_web_browser::OriginPersistence::Persistent => OriginPersistence::Persistent,
            hhhs_web_browser::OriginPersistence::BestEffort => OriginPersistence::BestEffort,
            hhhs_web_browser::OriginPersistence::Unknown => OriginPersistence::Unknown,
        })
}

fn indexed_db_error(error: impl std::fmt::Display) -> DurableHistoryError {
    DurableHistoryError::IndexedDb(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hhhs::{Entry, Position};
    use hhhs_store::{encode_storage_transaction, StorageRecoveryState};
    use hyperscape_protocol::{
        AssetDescriptor, AssetId, AuthoredCommand, AuthoredEnvelope, EntityId, LocalPeerEnvelope,
        MessageHeader, MessageId, PeerId, WireTransform, CURRENT_PROTOCOL_VERSION,
    };
    use hyperscope_app::LocalPeerDisposition;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    fn unique_project() -> ProjectId {
        let time = js_sys::Date::now() as u128;
        let random = (js_sys::Math::random() * u64::MAX as f64) as u128;
        ProjectId::from_u128(((time + 1) << 64) | (random + 1)).unwrap()
    }

    fn transaction(sequence: u64, payload: &[u8]) -> StorageTransaction {
        let mut transaction = StorageTransaction::new();
        transaction
            .expect_sequence(sequence)
            .push_entry(Entry::new(payload.to_vec(), Position::empty()));
        transaction
    }

    #[wasm_bindgen_test(async)]
    async fn indexeddb_rows_are_atomic_idempotent_and_collision_safe() {
        let project_id = unique_project();
        let mut durability = open_project_log(project_id).await.unwrap();
        assert!(durability.transactions().is_empty());
        let first = transaction(0, b"first");
        let first_state = StorageRecoveryState::default().advance(&first, 1).unwrap();
        AsyncTransactionSink::persist(&mut durability, &first, first_state)
            .await
            .unwrap();
        AsyncTransactionSink::persist(&mut durability, &first, first_state)
            .await
            .unwrap();

        let collision = transaction(0, b"different");
        assert!(matches!(
            AsyncTransactionSink::persist(&mut durability, &collision, first_state)
                .await
                .unwrap_err(),
            hhhs_web_browser::BrowserLogError::SequenceCollision { sequence: 0 }
        ));
        drop(durability);

        let recovered = open_project_log(project_id).await.unwrap();
        assert_eq!(recovered.transactions().len(), 1);
        assert_eq!(
            encode_storage_transaction(&recovered.transactions()[0]),
            encode_storage_transaction(&first)
        );
    }

    #[wasm_bindgen_test(async)]
    async fn durable_project_recovers_the_same_authored_state() {
        let project_id = unique_project();
        let entity = EntityId::from_u128(0x1111).unwrap();
        let authored = AuthoredEnvelope {
            header: MessageHeader {
                version: CURRENT_PROTOCOL_VERSION,
                message_id: MessageId::from_u128(0x2222).unwrap(),
                sender: PeerId::from_u128(0x3333).unwrap(),
                sequence: 1,
            },
            command: AuthoredCommand::SetEntityTransform {
                entity,
                transform: WireTransform {
                    translation: [1.0, 2.0, 3.0],
                    rotation_wxyz: [1.0, 0.0, 0.0, 0.0],
                    scale: [1.0, 1.0, 1.0],
                },
            },
        };

        let mut project = open_durable_project(project_id).await.unwrap();
        project.admit(&authored).await.unwrap();
        let before = project.state().unwrap();
        drop(project);

        let recovered = open_durable_project(project_id).await.unwrap();
        assert_eq!(recovered.state().unwrap(), before);
    }

    #[wasm_bindgen_test(async)]
    async fn durable_session_restores_appstore_and_rejects_recent_replay() {
        let project_id = unique_project();
        let entity = EntityId::from_u128(0x3111).unwrap();
        let authored = AuthoredEnvelope {
            header: MessageHeader {
                version: CURRENT_PROTOCOL_VERSION,
                message_id: MessageId::from_u128(0x3222).unwrap(),
                sender: PeerId::from_u128(0x3333).unwrap(),
                sequence: 7,
            },
            command: AuthoredCommand::SetEntityTransform {
                entity,
                transform: WireTransform {
                    translation: [1.0, 2.0, 3.0],
                    rotation_wxyz: [1.0, 0.0, 0.0, 0.0],
                    scale: [1.0, 1.0, 1.0],
                },
            },
        };
        let source_store = AppStore::default();
        let mut opened = open_durable_authored_session(project_id, &source_store)
            .await
            .unwrap();
        assert!(opened.restored_projection().is_none());
        let applied = opened
            .session_mut()
            .accept_local_peer(
                &source_store,
                LocalPeerEnvelope::Authored(authored.clone()),
                1.0,
            )
            .await
            .unwrap();
        assert_eq!(applied.peer.disposition, LocalPeerDisposition::Applied);
        assert_eq!(opened.session().history_len(), 1);
        let expected_scene = source_store.authored_scene_snapshot();
        drop(opened);

        let restored_store = AppStore::default();
        let mut reopened = open_durable_authored_session(project_id, &restored_store)
            .await
            .unwrap();
        assert!(reopened.restored_projection().is_some());
        assert_eq!(restored_store.authored_scene_snapshot(), expected_scene);
        let before_replay = restored_store.summary_snapshot();
        let replay = reopened
            .session_mut()
            .accept_local_peer(&restored_store, LocalPeerEnvelope::Authored(authored), 2.0)
            .await
            .unwrap();
        assert_eq!(
            replay.peer.disposition,
            LocalPeerDisposition::IgnoredDuplicate
        );
        assert!(replay.durable.is_none());
        assert_eq!(reopened.session().history_len(), 1);
        assert_eq!(restored_store.summary_snapshot(), before_replay);
    }

    #[wasm_bindgen_test(async)]
    async fn project_archive_import_is_validated_persistent_and_idempotent() {
        let project_id = unique_project();
        let entity = EntityId::from_u128(0x4444).unwrap();
        let authored = AuthoredEnvelope {
            header: MessageHeader {
                version: CURRENT_PROTOCOL_VERSION,
                message_id: MessageId::from_u128(0x5555).unwrap(),
                sender: PeerId::from_u128(0x6666).unwrap(),
                sequence: 1,
            },
            command: AuthoredCommand::SetEntityTransform {
                entity,
                transform: WireTransform {
                    translation: [4.0, 5.0, 6.0],
                    rotation_wxyz: [1.0, 0.0, 0.0, 0.0],
                    scale: [1.0, 1.0, 1.0],
                },
            },
        };
        let mut source = DurableProject::new(project_id).unwrap();
        source.admit(&authored).await.unwrap();
        let expected = source.state().unwrap();
        let archive = source.export_archive().unwrap();

        let imported = import_project_archive(&archive).await.unwrap();
        assert_eq!(imported.state().unwrap(), expected);
        drop(imported);

        let retried = import_project_archive(&archive).await.unwrap();
        assert_eq!(retried.state().unwrap(), expected);
        drop(retried);

        let recovered = open_durable_project(project_id).await.unwrap();
        assert_eq!(recovered.state().unwrap(), expected);
    }

    #[wasm_bindgen_test(async)]
    async fn imported_authored_session_persists_its_local_cursor_for_strict_restart() {
        let project_id = unique_project();
        let authored = AuthoredEnvelope {
            header: MessageHeader {
                version: CURRENT_PROTOCOL_VERSION,
                message_id: MessageId::from_u128(0x7777).unwrap(),
                sender: PeerId::from_u128(0x8888).unwrap(),
                sequence: 1,
            },
            command: AuthoredCommand::UpsertAsset {
                asset: AssetDescriptor {
                    id: AssetId::from_u128(0x9999).unwrap(),
                    uri: "portable.glb".into(),
                    media_type: Some("model/gltf-binary".into()),
                    content_digest: None,
                },
            },
        };
        let mut source = DurableProject::new(project_id).unwrap();
        source.admit(&authored).await.unwrap();
        let archive = source.export_archive().unwrap();
        let imported_store = AppStore::default();

        let imported = import_durable_authored_session(&archive, &imported_store)
            .await
            .unwrap();
        assert_eq!(imported.session().observed_projection_revision(), Some(0));
        assert!(imported.restored_projection().is_some());
        let expected_scene = imported_store.authored_scene_snapshot();
        drop(imported);

        let restarted_store = AppStore::default();
        let restarted = open_durable_authored_session(project_id, &restarted_store)
            .await
            .unwrap();
        assert_eq!(restarted.session().observed_projection_revision(), Some(0));
        assert_eq!(restarted_store.authored_scene_snapshot(), expected_scene);
    }
}
