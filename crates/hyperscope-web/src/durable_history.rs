//! Browser durability for Hyperscape's HHHS-authored operation lane.
//!
//! The platform-independent row codec and recovery validator live here so
//! corruption and ordering behavior can be tested natively. IndexedDB access
//! is compiled only for `wasm32`. A strict IndexedDB transaction establishes
//! atomic persist-before-publish ordering, but does not by itself protect an
//! origin from browser eviction or private-session teardown; callers should
//! surface [`OriginPersistence`] through the wasm status helpers.
//!
//! [`attach_durable_authored_session`] is the platform-neutral lifecycle seam:
//! it pairs recovered history with restart-safe ingress memory and restores the
//! AppStore projection before transport starts. The wasm IndexedDB adapter
//! exposes the same sequence through `open_durable_authored_session`.

#[cfg(any(target_arch = "wasm32", test))]
use hhhs::Digest;
use hhhs_replica::AsyncTransactionSink;
#[cfg(any(target_arch = "wasm32", test))]
use hhhs_store::{decode_storage_transaction, encode_storage_transaction, StorageTransaction};
use hyperscape_hhhs::DurableProject;
#[cfg(any(target_arch = "wasm32", test))]
use hyperscape_hhhs::ProjectId;
use hyperscope_app::{AppCommit, AppStore};
use hyperscope_hhhs_shadow::{DurableAuthoredSession, DurableAuthoredSessionInitError};
#[cfg(any(target_arch = "wasm32", test))]
use std::collections::BTreeMap;

#[cfg(target_arch = "wasm32")]
mod indexed_db;
#[cfg(target_arch = "wasm32")]
pub use indexed_db::{
    import_durable_authored_session, import_project_archive,
    import_validated_durable_authored_session, import_validated_project_archive,
    open_durable_authored_session, open_durable_project, IndexedDbDurability,
};
#[cfg(target_arch = "wasm32")]
pub use indexed_db::{origin_persistence, request_origin_persistence};

/// IndexedDB name dedicated to durable authored history.
///
/// It is intentionally distinct from the existing `hyperscope` version-3
/// cache/file database.
pub const DATABASE_NAME: &str = "hyperscape-authored-v1";
/// Frozen IndexedDB schema version for [`DATABASE_NAME`].
pub const DATABASE_VERSION: u32 = 1;
/// Object store containing one canonical transaction per project sequence.
pub const TRANSACTION_STORE: &str = "transactions";

/// Browser-level retention status for the origin containing the dedicated
/// authored-history database.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OriginPersistence {
    /// The browser reports that the origin has persistent storage protection.
    Persistent,
    /// IndexedDB works, but the browser may evict the origin under pressure.
    BestEffort,
    /// The StorageManager persistence API is unavailable in this context.
    Unknown,
}

#[cfg(any(target_arch = "wasm32", test))]
const ROW_DOMAIN: &[u8] = b"hyperscape IndexedDB HHHS transaction row v1\0";
#[cfg(any(target_arch = "wasm32", test))]
const ROW_DIGEST_BYTES: usize = 32;
#[cfg(any(target_arch = "wasm32", test))]
const KEY_SEQUENCE_HEX_BYTES: usize = 16;

/// Owned, thread-safe failures surfaced by the browser durability boundary.
///
/// JavaScript values and DOM exceptions are converted to strings immediately
/// so this type satisfies `AsyncTransactionSink::Error`'s `Send + Sync`
/// contract even though the IndexedDB future itself remains browser-local.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DurableHistoryError {
    #[error("HHHS storage transaction has no expected sequence")]
    MissingExpectedSequence,
    #[error("transaction row key is malformed: {0}")]
    MalformedKey(String),
    #[error("transaction row belongs to another project: {0}")]
    WrongProjectKey(String),
    #[error(
        "transaction row sequence {actual} is not the expected contiguous sequence {expected}"
    )]
    SequenceGap { expected: u64, actual: u64 },
    #[error("transaction row key sequence {key} does not match encoded sequence {encoded:?}")]
    SequenceMismatch { key: u64, encoded: Option<u64> },
    #[error("transaction sequence overflow")]
    SequenceOverflow,
    #[error("duplicate transaction row key: {0}")]
    DuplicateKey(String),
    #[error("transaction sequence {sequence} is already occupied by different durable bytes")]
    SequenceCollision { sequence: u64 },
    #[error("transaction row is truncated or has an unsupported format")]
    MalformedRow,
    #[error("transaction row checksum mismatch")]
    ChecksumMismatch,
    #[error("HHHS storage transaction is invalid: {0}")]
    InvalidTransaction(String),
    #[error("IndexedDB operation failed: {0}")]
    IndexedDb(String),
    #[error("trusted local project recovery failed: {0}")]
    ProjectRecovery(String),
    #[error("portable project archive failed validation or admission: {0}")]
    ProjectArchive(String),
    #[error("durable authored session initialization failed: {0}")]
    SessionInitialization(String),
    #[error("durable authored AppStore restoration failed: {0}")]
    ProjectionRestore(String),
}

/// A recovered durable authority paired with the result of aligning its
/// rebuildable AppStore projection.
pub struct OpenedDurableAuthoredSession<D> {
    session: DurableAuthoredSession<D>,
    restored_projection: Option<AppCommit>,
}

impl<D> OpenedDurableAuthoredSession<D> {
    pub fn session(&self) -> &DurableAuthoredSession<D> {
        &self.session
    }

    pub fn session_mut(&mut self) -> &mut DurableAuthoredSession<D> {
        &mut self.session
    }

    pub fn restored_projection(&self) -> Option<&AppCommit> {
        self.restored_projection.as_ref()
    }

    pub fn into_parts(self) -> (DurableAuthoredSession<D>, Option<AppCommit>) {
        (self.session, self.restored_projection)
    }
}

/// Attach a recovered platform durability host to the restart-safe authored
/// session and atomically align a fresh AppStore projection.
///
/// This function performs no browser or graphics work. Platform adapters such
/// as IndexedDB first recover a [`DurableProject`], then delegate here so every
/// host shares the same cursor, projection, and peer-dedup lifecycle.
pub fn attach_durable_authored_session<D>(
    project: DurableProject<D>,
    store: &AppStore,
) -> Result<OpenedDurableAuthoredSession<D>, DurableHistoryError> {
    let session = DurableAuthoredSession::from_project(project).map_err(
        |error: DurableAuthoredSessionInitError| {
            DurableHistoryError::SessionInitialization(error.to_string())
        },
    )?;
    restore_opened_session(session, store)
}

/// Attach a project returned by an explicit portable-archive import.
///
/// Unlike ordinary restart attachment, this may durably create or advance the
/// receiver-local projection cursor after the imported history has validated.
/// The public HHHS history remains unchanged. Exact re-import at the current
/// horizon is write-free.
pub async fn attach_imported_durable_authored_session<D>(
    project: DurableProject<D>,
    store: &AppStore,
) -> Result<OpenedDurableAuthoredSession<D>, DurableHistoryError>
where
    D: AsyncTransactionSink,
{
    let session = DurableAuthoredSession::from_imported_project(project)
        .await
        .map_err(|error: DurableAuthoredSessionInitError| {
            DurableHistoryError::SessionInitialization(error.to_string())
        })?;
    restore_opened_session(session, store)
}

fn restore_opened_session<D>(
    session: DurableAuthoredSession<D>,
    store: &AppStore,
) -> Result<OpenedDurableAuthoredSession<D>, DurableHistoryError> {
    let restored_projection = session
        .restore_store(store)
        .map_err(|error| DurableHistoryError::ProjectionRestore(error.to_string()))?;
    Ok(OpenedDurableAuthoredSession {
        session,
        restored_projection,
    })
}

#[cfg(any(target_arch = "wasm32", test))]
fn project_prefix(project_id: ProjectId) -> String {
    format!("p/{}/s/", project_id.as_uuid().simple())
}

#[cfg(any(target_arch = "wasm32", test))]
fn transaction_key(project_id: ProjectId, sequence: u64) -> String {
    format!("{}{sequence:016x}", project_prefix(project_id))
}

#[cfg(any(target_arch = "wasm32", test))]
fn project_key_bounds(project_id: ProjectId) -> (String, String) {
    let prefix = project_prefix(project_id);
    (
        format!("{prefix}{:016x}", 0_u64),
        format!("{prefix}{:016x}", u64::MAX),
    )
}

#[cfg(any(target_arch = "wasm32", test))]
fn parse_transaction_key(project_id: ProjectId, key: &str) -> Result<u64, DurableHistoryError> {
    let prefix = project_prefix(project_id);
    let Some(sequence) = key.strip_prefix(&prefix) else {
        return Err(DurableHistoryError::WrongProjectKey(key.into()));
    };
    if sequence.len() != KEY_SEQUENCE_HEX_BYTES
        || !sequence.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(DurableHistoryError::MalformedKey(key.into()));
    }
    u64::from_str_radix(sequence, 16).map_err(|_| DurableHistoryError::MalformedKey(key.into()))
}

#[cfg(any(target_arch = "wasm32", test))]
fn row_digest(transaction_bytes: &[u8]) -> Digest {
    let mut authenticated = Vec::with_capacity(ROW_DOMAIN.len() + transaction_bytes.len());
    authenticated.extend_from_slice(ROW_DOMAIN);
    authenticated.extend_from_slice(transaction_bytes);
    Digest::of(&authenticated)
}

#[cfg(any(target_arch = "wasm32", test))]
fn encode_transaction_row(
    transaction: &StorageTransaction,
) -> Result<(u64, Vec<u8>), DurableHistoryError> {
    let sequence = transaction
        .expected_sequence()
        .ok_or(DurableHistoryError::MissingExpectedSequence)?;
    validate_authored_transaction_shape(transaction)?;
    let transaction_bytes = encode_storage_transaction(transaction);
    decode_storage_transaction(&transaction_bytes)
        .map_err(|error| DurableHistoryError::InvalidTransaction(error.to_string()))?;
    let digest = row_digest(&transaction_bytes);
    let mut row = Vec::with_capacity(ROW_DOMAIN.len() + ROW_DIGEST_BYTES + transaction_bytes.len());
    row.extend_from_slice(ROW_DOMAIN);
    row.extend_from_slice(digest.as_bytes());
    row.extend_from_slice(&transaction_bytes);
    Ok((sequence, row))
}

#[cfg(any(target_arch = "wasm32", test))]
fn decode_transaction_row(row: &[u8]) -> Result<StorageTransaction, DurableHistoryError> {
    let payload_offset = ROW_DOMAIN.len() + ROW_DIGEST_BYTES;
    if row.len() < payload_offset || !row.starts_with(ROW_DOMAIN) {
        return Err(DurableHistoryError::MalformedRow);
    }
    let expected_digest = &row[ROW_DOMAIN.len()..payload_offset];
    let transaction_bytes = &row[payload_offset..];
    if row_digest(transaction_bytes).as_bytes() != expected_digest {
        return Err(DurableHistoryError::ChecksumMismatch);
    }
    let transaction = decode_storage_transaction(transaction_bytes)
        .map_err(|error| DurableHistoryError::InvalidTransaction(error.to_string()))?;
    validate_authored_transaction_shape(&transaction)?;
    Ok(transaction)
}

#[cfg(any(target_arch = "wasm32", test))]
fn validate_authored_transaction_shape(
    transaction: &StorageTransaction,
) -> Result<(), DurableHistoryError> {
    if transaction.entries().is_empty() && transaction.checkpoints().is_empty() {
        return Err(DurableHistoryError::InvalidTransaction(
            "authored-history transactions must contain an entry or projection checkpoint".into(),
        ));
    }
    Ok(())
}

#[cfg(any(target_arch = "wasm32", test))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExistingRow {
    Insert,
    ExactRetry,
}

#[cfg(any(target_arch = "wasm32", test))]
fn classify_existing_row(
    sequence: u64,
    existing: Option<&[u8]>,
    candidate: &[u8],
) -> Result<ExistingRow, DurableHistoryError> {
    match existing {
        None => Ok(ExistingRow::Insert),
        Some(bytes) if bytes == candidate => Ok(ExistingRow::ExactRetry),
        Some(_) => Err(DurableHistoryError::SequenceCollision { sequence }),
    }
}

#[cfg(any(target_arch = "wasm32", test))]
fn decode_ordered_rows(
    project_id: ProjectId,
    rows: impl IntoIterator<Item = (String, Vec<u8>)>,
) -> Result<Vec<StorageTransaction>, DurableHistoryError> {
    let mut ordered = BTreeMap::new();
    for (key, row) in rows {
        if ordered.insert(key.clone(), row).is_some() {
            return Err(DurableHistoryError::DuplicateKey(key));
        }
    }

    let mut expected = 0_u64;
    let mut transactions = Vec::with_capacity(ordered.len());
    for (key, row) in ordered {
        let sequence = parse_transaction_key(project_id, &key)?;
        if sequence != expected {
            return Err(DurableHistoryError::SequenceGap {
                expected,
                actual: sequence,
            });
        }
        let transaction = decode_transaction_row(&row)?;
        if transaction.expected_sequence() != Some(sequence) {
            return Err(DurableHistoryError::SequenceMismatch {
                key: sequence,
                encoded: transaction.expected_sequence(),
            });
        }
        transactions.push(transaction);
        expected = expected
            .checked_add(1)
            .ok_or(DurableHistoryError::SequenceOverflow)?;
    }
    Ok(transactions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;
    use hhhs::{Entry, Position};
    use hhhs_store::{ProjectionCheckpoint, ProjectionKey};
    use hyperscape_protocol::{
        AssetDescriptor, AssetId, AuthoredCommand, AuthoredEnvelope, LocalPeerEnvelope,
        MessageHeader, MessageId, PeerId, CURRENT_PROTOCOL_VERSION,
    };
    use hyperscope_app::{CommitDisposition, LocalPeerDisposition};

    fn project(value: u128) -> ProjectId {
        ProjectId::from_u128(value).unwrap()
    }

    fn transaction(sequence: Option<u64>, payload: &[u8]) -> StorageTransaction {
        let mut transaction = StorageTransaction::new();
        if let Some(sequence) = sequence {
            transaction.expect_sequence(sequence);
        }
        transaction.push_entry(Entry::new(payload.to_vec(), Position::default()));
        transaction
    }

    #[test]
    fn fixed_width_keys_sort_in_sequence_order_and_are_project_scoped() {
        let project_id = project(0x1234);
        let (lower, upper) = project_key_bounds(project_id);
        assert_eq!(lower, transaction_key(project_id, 0));
        assert_eq!(upper, transaction_key(project_id, u64::MAX));
        let mut keys = [10, 0, u64::MAX, 9].map(|sequence| transaction_key(project_id, sequence));
        keys.sort();
        assert!(matches!(
            parse_transaction_key(project(0x5678), &keys[0]),
            Err(DurableHistoryError::WrongProjectKey(_))
        ));
        assert_eq!(
            keys.map(|key| parse_transaction_key(project_id, &key).unwrap()),
            [0, 9, 10, u64::MAX]
        );
    }

    #[test]
    fn checksummed_rows_round_trip_and_reject_corruption() {
        let transaction = transaction(Some(7), b"payload");
        let (_, row) = encode_transaction_row(&transaction).unwrap();
        let recovered = decode_transaction_row(&row).unwrap();
        assert_eq!(
            encode_storage_transaction(&recovered),
            encode_storage_transaction(&transaction)
        );

        let mut corrupt = row;
        *corrupt.last_mut().unwrap() ^= 1;
        assert_eq!(
            decode_transaction_row(&corrupt).unwrap_err(),
            DurableHistoryError::ChecksumMismatch
        );
    }

    #[test]
    fn ordered_recovery_sorts_rows_and_requires_a_contiguous_sequence() {
        let project = project(0xaaaa);
        let (_, row0) = encode_transaction_row(&transaction(Some(0), b"zero")).unwrap();
        let (_, row1) = encode_transaction_row(&transaction(Some(1), b"one")).unwrap();
        let recovered = decode_ordered_rows(
            project,
            [
                (transaction_key(project, 1), row1.clone()),
                (transaction_key(project, 0), row0),
            ],
        )
        .unwrap();
        assert_eq!(recovered.len(), 2);
        assert_eq!(recovered[0].expected_sequence(), Some(0));
        assert_eq!(recovered[1].expected_sequence(), Some(1));

        assert_eq!(
            decode_ordered_rows(project, [(transaction_key(project, 1), row1)]).unwrap_err(),
            DurableHistoryError::SequenceGap {
                expected: 0,
                actual: 1,
            }
        );
    }

    #[test]
    fn recovery_cross_checks_the_key_and_encoded_sequence() {
        let project = project(0xbbbb);
        let (_, row) = encode_transaction_row(&transaction(Some(9), b"nine")).unwrap();
        assert_eq!(
            decode_ordered_rows(project, [(transaction_key(project, 0), row)]).unwrap_err(),
            DurableHistoryError::SequenceMismatch {
                key: 0,
                encoded: Some(9),
            }
        );
    }

    #[test]
    fn exact_retries_are_idempotent_but_collisions_are_refused() {
        let candidate = vec![1, 2, 3];
        assert_eq!(
            classify_existing_row(4, None, &candidate).unwrap(),
            ExistingRow::Insert
        );
        assert_eq!(
            classify_existing_row(4, Some(&candidate), &candidate).unwrap(),
            ExistingRow::ExactRetry
        );
        assert_eq!(
            classify_existing_row(4, Some(&[9]), &candidate).unwrap_err(),
            DurableHistoryError::SequenceCollision { sequence: 4 }
        );
    }

    #[test]
    fn transactions_without_an_expected_sequence_are_not_persistable() {
        assert_eq!(
            encode_transaction_row(&transaction(None, b"unsequenced")).unwrap_err(),
            DurableHistoryError::MissingExpectedSequence
        );
    }

    #[test]
    fn checkpoint_only_rows_round_trip_but_empty_transactions_are_rejected() {
        let mut checkpoint_only = StorageTransaction::new();
        checkpoint_only.expect_sequence(3).save_checkpoint(
            ProjectionCheckpoint::new(
                ProjectionKey::new("hyperscope/import-cursor", 1).unwrap(),
                Position::default(),
                Digest::of(b"history root"),
                9_u64.to_le_bytes().to_vec(),
            )
            .unwrap(),
        );
        let (_, row) = encode_transaction_row(&checkpoint_only).unwrap();
        let decoded = decode_transaction_row(&row).unwrap();
        assert_eq!(decoded.expected_sequence(), Some(3));
        assert!(decoded.entries().is_empty());
        assert_eq!(decoded.checkpoints().len(), 1);

        let mut empty = StorageTransaction::new();
        empty.expect_sequence(4);
        assert!(matches!(
            encode_transaction_row(&empty),
            Err(DurableHistoryError::InvalidTransaction(_))
        ));
    }

    #[test]
    fn host_attachment_restores_projection_and_recovered_peer_dedup() {
        let project_id = project(0xdddd);
        let envelope = AuthoredEnvelope {
            header: MessageHeader {
                version: CURRENT_PROTOCOL_VERSION,
                message_id: MessageId::from_u128(0xdd01).unwrap(),
                sender: PeerId::from_u128(0xdd02).unwrap(),
                sequence: 7,
            },
            command: AuthoredCommand::UpsertAsset {
                asset: AssetDescriptor {
                    id: AssetId::from_u128(0xdd03).unwrap(),
                    uri: "restored.glb".into(),
                    media_type: Some("model/gltf-binary".into()),
                    content_digest: None,
                },
            },
        };
        let source_store = AppStore::default();
        let mut source = DurableAuthoredSession::new(project_id).unwrap();
        block_on(source.accept_local_peer(
            &source_store,
            LocalPeerEnvelope::Authored(envelope.clone()),
            1.0,
        ))
        .unwrap();
        let expected_scene = source_store.authored_scene_snapshot();
        let durable_bytes = source.durability().bytes().to_vec();
        let project = DurableProject::recover(project_id, durable_bytes.clone()).unwrap();
        let restored_store = AppStore::default();

        let mut opened = attach_durable_authored_session(project, &restored_store).unwrap();
        assert_eq!(
            opened.restored_projection().unwrap().disposition,
            CommitDisposition::Applied
        );
        assert_eq!(restored_store.authored_scene_snapshot(), expected_scene);
        assert_eq!(opened.session().history_len(), 1);
        assert_eq!(opened.session().durability().bytes(), durable_bytes);

        let before_replay = restored_store.summary_snapshot();
        let replay = block_on(opened.session_mut().accept_local_peer(
            &restored_store,
            LocalPeerEnvelope::Authored(envelope),
            2.0,
        ))
        .unwrap();
        assert_eq!(replay.peer.disposition, LocalPeerDisposition::IgnoredDuplicate);
        assert!(replay.durable.is_none());
        assert_eq!(restored_store.summary_snapshot(), before_replay);
        assert_eq!(opened.session().history_len(), 1);
        assert_eq!(opened.session().durability().bytes(), durable_bytes);
    }

    #[test]
    fn explicit_import_attachment_persists_a_cursor_and_restores_projection() {
        let project_id = project(0xdd10);
        let envelope = AuthoredEnvelope {
            header: MessageHeader {
                version: CURRENT_PROTOCOL_VERSION,
                message_id: MessageId::from_u128(0xdd11).unwrap(),
                sender: PeerId::from_u128(0xdd12).unwrap(),
                sequence: 1,
            },
            command: AuthoredCommand::UpsertAsset {
                asset: AssetDescriptor {
                    id: AssetId::from_u128(0xdd13).unwrap(),
                    uri: "imported.glb".into(),
                    media_type: Some("model/gltf-binary".into()),
                    content_digest: None,
                },
            },
        };
        let mut source = DurableProject::new(project_id).unwrap();
        block_on(source.admit(&envelope)).unwrap();
        let archive = source.export_archive().unwrap();
        let imported = block_on(DurableProject::import_archive(&archive)).unwrap();
        let store = AppStore::default();

        let opened = block_on(attach_imported_durable_authored_session(imported, &store)).unwrap();
        assert_eq!(opened.session().observed_projection_revision(), Some(0));
        assert_eq!(opened.session().history_len(), 1);
        assert_eq!(
            opened.restored_projection().unwrap().disposition,
            CommitDisposition::Applied
        );
        assert_eq!(store.authored_scene_snapshot().projection_revision, Some(0));

        let durable_bytes = opened.session().durability().bytes().to_vec();
        let recovered = DurableProject::recover(project_id, durable_bytes).unwrap();
        assert!(attach_durable_authored_session(recovered, &AppStore::default()).is_ok());
    }
}
