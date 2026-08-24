//! Browser durability for Hyperscape's HHHS-authored operation lane.
//!
//! The platform-independent row codec and recovery validator live here so
//! corruption and ordering behavior can be tested natively. IndexedDB access
//! is compiled only for `wasm32`. A strict IndexedDB transaction establishes
//! atomic persist-before-publish ordering, but does not by itself protect an
//! origin from browser eviction or private-session teardown; callers should
//! surface [`OriginPersistence`] through the wasm status helpers.

#[cfg(any(target_arch = "wasm32", test))]
use hhhs::Digest;
#[cfg(any(target_arch = "wasm32", test))]
use hhhs_store::{decode_storage_transaction, encode_storage_transaction, StorageTransaction};
#[cfg(any(target_arch = "wasm32", test))]
use hyperscape_hhhs::ProjectId;
#[cfg(any(target_arch = "wasm32", test))]
use std::collections::BTreeMap;

#[cfg(target_arch = "wasm32")]
mod indexed_db;
#[cfg(target_arch = "wasm32")]
pub use indexed_db::{open_durable_project, IndexedDbDurability};
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
    if transaction.entries().is_empty() {
        return Err(DurableHistoryError::InvalidTransaction(
            "authored-history transactions must contain an entry".into(),
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
    use hhhs::{Entry, Position};

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
}
