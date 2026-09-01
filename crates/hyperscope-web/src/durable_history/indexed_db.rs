use super::{
    classify_existing_row, decode_ordered_rows, encode_transaction_row, project_key_bounds,
    transaction_key, DurableHistoryError, ExistingRow, OriginPersistence, DATABASE_NAME,
    DATABASE_VERSION, TRANSACTION_STORE,
};
use futures::try_join;
use hhhs_replica::AsyncTransactionSink;
use hhhs_store::StorageTransaction;
use hyperscape_hhhs::{DurableProject, ProjectId};
use hyperscope_app::AppStore;
use indexed_db_futures::database::Database;
use indexed_db_futures::prelude::*;
use indexed_db_futures::transaction::{TransactionDurability, TransactionMode, TransactionOptions};
use indexed_db_futures::typed_array::{Uint8Array, Uint8ArraySlice};
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::{spawn_local, JsFuture};

/// IndexedDB sink for exactly one Hyperscape project's authored history.
///
/// The sink is browser-local and intentionally not `Send`. A single local
/// actor must own it and its [`DurableProject`] from preparation through
/// finalization; `AsyncTransactionSink` does not require its future to be
/// `Send`. IndexedDB collision detection prevents silent overwrites, but does
/// not make multiple tabs valid shared writers. A second writer must stop and
/// recover after a collision; a future live integration should enforce one
/// owner with a Web Lock or SharedWorker.
///
/// Writes request IndexedDB's `strict` transaction durability. That does not
/// prevent whole-origin eviction or private-session loss; use
/// [`origin_persistence`] or [`request_origin_persistence`] and surface their
/// [`OriginPersistence`] result to the user.
pub struct IndexedDbDurability {
    db: Database,
    project_id: ProjectId,
}

impl IndexedDbDurability {
    /// Open the dedicated v1 database and recover this project's trusted local
    /// transaction rows.
    pub async fn open(
        project_id: ProjectId,
    ) -> Result<(Self, Vec<StorageTransaction>), DurableHistoryError> {
        let db = Database::open(DATABASE_NAME)
            .with_version(DATABASE_VERSION)
            .with_on_upgrade_needed(|_event, db| {
                if !db
                    .object_store_names()
                    .any(|name| name == TRANSACTION_STORE)
                {
                    db.create_object_store(TRANSACTION_STORE).build()?;
                }
                Ok(())
            })
            .await
            .map_err(indexed_db_error)?;
        if !db
            .object_store_names()
            .any(|name| name == TRANSACTION_STORE)
        {
            return Err(DurableHistoryError::IndexedDb(format!(
                "{DATABASE_NAME} v{DATABASE_VERSION} lacks {TRANSACTION_STORE}"
            )));
        }
        let mut version_changes = db.version_changes().map_err(indexed_db_error)?;
        spawn_local(async move {
            if version_changes.recv().await.is_some() {
                version_changes.db().clone().close();
            }
        });

        let durability = Self { db, project_id };
        let transactions = durability.load_transactions().await?;
        Ok((durability, transactions))
    }

    async fn load_transactions(&self) -> Result<Vec<StorageTransaction>, DurableHistoryError> {
        let (lower, upper) = project_key_bounds(self.project_id);
        let transaction = self
            .db
            .transaction(TRANSACTION_STORE)
            .build()
            .map_err(indexed_db_error)?;
        let store = transaction
            .object_store(TRANSACTION_STORE)
            .map_err(indexed_db_error)?;

        // Build both requests before yielding so the IndexedDB transaction
        // cannot become inactive between the key and value snapshots.
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

        if keys.len() != values.len() {
            return Err(DurableHistoryError::IndexedDb(format!(
                "transaction key/value snapshot length mismatch: {} keys, {} values",
                keys.len(),
                values.len()
            )));
        }
        decode_ordered_rows(
            self.project_id,
            keys.into_iter()
                .zip(values)
                .map(|(key, value)| (key, value.into())),
        )
    }

    fn strict_readwrite(
        &self,
    ) -> Result<indexed_db_futures::transaction::Transaction<'_>, DurableHistoryError> {
        let mut options = TransactionOptions::new();
        options.set_durability(TransactionDurability::Strict);
        self.db
            .transaction(TRANSACTION_STORE)
            .with_mode(TransactionMode::Readwrite)
            .with_options(options)
            .build()
            .map_err(indexed_db_error)
    }
}

impl Drop for IndexedDbDurability {
    fn drop(&mut self) {
        // End the detached version-change listener and release this connection
        // when the single owning project actor shuts down.
        self.db.clone().close();
    }
}

impl AsyncTransactionSink for IndexedDbDurability {
    type Error = DurableHistoryError;

    async fn persist(&mut self, transaction: &StorageTransaction) -> Result<(), Self::Error> {
        let (sequence, row) = encode_transaction_row(transaction)?;
        let key = transaction_key(self.project_id, sequence);
        let idb_transaction = self.strict_readwrite()?;
        let store = idb_transaction
            .object_store(TRANSACTION_STORE)
            .map_err(indexed_db_error)?;
        let existing = store
            .get::<Uint8Array, _, _>(key.clone())
            .primitive()
            .map_err(indexed_db_error)?
            .await
            .map_err(indexed_db_error)?;
        if let Some(existing) = &existing {
            // Refuse corrupted durable state rather than reporting it as an
            // ordinary writer collision or overwriting it.
            super::decode_transaction_row(existing.as_ref())?;
        }

        match classify_existing_row(sequence, existing.as_ref().map(AsRef::as_ref), &row)? {
            ExistingRow::Insert => {
                store
                    .add(Uint8ArraySlice::new(&row))
                    .with_key(key)
                    .without_key_type()
                    .primitive()
                    .map_err(indexed_db_error)?
                    .await
                    .map_err(indexed_db_error)?;
            }
            ExistingRow::ExactRetry => {}
        }

        drop(store);
        idb_transaction.commit().await.map_err(indexed_db_error)
    }
}

/// Open and recover a browser-durable project without attaching it to current
/// application, UI, or rendering behavior.
pub async fn open_durable_project(
    project_id: ProjectId,
) -> Result<DurableProject<IndexedDbDurability>, DurableHistoryError> {
    let (durability, transactions) = IndexedDbDurability::open(project_id).await?;
    DurableProject::recover_trusted_transactions(project_id, durability, transactions)
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

/// Query whether the browser currently protects this origin from eviction.
///
/// `BestEffort` and `Unknown` still allow strict, atomic IndexedDB commits, but
/// must not be presented as durable against storage pressure or private-mode
/// teardown.
pub async fn origin_persistence() -> Result<OriginPersistence, DurableHistoryError> {
    call_storage_manager_bool("persisted")
        .await
        .map(|result| match result {
            Some(true) => OriginPersistence::Persistent,
            Some(false) => OriginPersistence::BestEffort,
            None => OriginPersistence::Unknown,
        })
}

/// Explicitly ask the browser to protect this origin from eviction and return
/// the resulting retention status.
///
/// Browsers may deny the request without prompting. Denial is reported as
/// `BestEffort`, not as successful durable retention.
pub async fn request_origin_persistence() -> Result<OriginPersistence, DurableHistoryError> {
    if origin_persistence().await? == OriginPersistence::Persistent {
        return Ok(OriginPersistence::Persistent);
    }
    call_storage_manager_bool("persist")
        .await
        .map(|result| match result {
            Some(true) => OriginPersistence::Persistent,
            Some(false) => OriginPersistence::BestEffort,
            None => OriginPersistence::Unknown,
        })
}

async fn call_storage_manager_bool(method: &str) -> Result<Option<bool>, DurableHistoryError> {
    let global = js_sys::global();
    let navigator =
        js_sys::Reflect::get(&global, &JsValue::from_str("navigator")).map_err(js_error)?;
    if navigator.is_null() || navigator.is_undefined() {
        return Ok(None);
    }
    let storage =
        js_sys::Reflect::get(&navigator, &JsValue::from_str("storage")).map_err(js_error)?;
    if storage.is_null() || storage.is_undefined() {
        return Ok(None);
    }
    let function = js_sys::Reflect::get(&storage, &JsValue::from_str(method)).map_err(js_error)?;
    let Some(function) = function.dyn_ref::<js_sys::Function>() else {
        return Ok(None);
    };
    let promise = function
        .call0(&storage)
        .map_err(js_error)?
        .dyn_into::<js_sys::Promise>()
        .map_err(|_| {
            DurableHistoryError::IndexedDb(format!(
                "navigator.storage.{method}() did not return a Promise"
            ))
        })?;
    JsFuture::from(promise)
        .await
        .map_err(js_error)?
        .as_bool()
        .map(Some)
        .ok_or_else(|| {
            DurableHistoryError::IndexedDb(format!(
                "navigator.storage.{method}() did not resolve to a boolean"
            ))
        })
}

fn indexed_db_error(error: impl std::fmt::Display) -> DurableHistoryError {
    DurableHistoryError::IndexedDb(error.to_string())
}

fn js_error(error: JsValue) -> DurableHistoryError {
    DurableHistoryError::IndexedDb(format!("{error:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use hhhs::{Entry, Position};
    use hhhs_store::encode_storage_transaction;
    use hyperscape_protocol::{
        AuthoredCommand, AuthoredEnvelope, EntityId, LocalPeerEnvelope, MessageHeader, MessageId,
        PeerId, WireTransform, CURRENT_PROTOCOL_VERSION,
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
        let (mut durability, initial) = IndexedDbDurability::open(project_id).await.unwrap();
        assert!(initial.is_empty());
        let first = transaction(0, b"first");
        durability.persist(&first).await.unwrap();
        durability.persist(&first).await.unwrap();

        let collision = transaction(0, b"different");
        assert_eq!(
            durability.persist(&collision).await.unwrap_err(),
            DurableHistoryError::SequenceCollision { sequence: 0 }
        );
        drop(durability);

        let (_durability, recovered) = IndexedDbDurability::open(project_id).await.unwrap();
        assert_eq!(recovered.len(), 1);
        assert_eq!(
            encode_storage_transaction(&recovered[0]),
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
            .accept_local_peer(
                &restored_store,
                LocalPeerEnvelope::Authored(authored),
                2.0,
            )
            .await
            .unwrap();
        assert_eq!(replay.peer.disposition, LocalPeerDisposition::IgnoredDuplicate);
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
}
